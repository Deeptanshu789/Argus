#!/usr/bin/env python3
"""Carve a training-sized YOLO subset out of a downloaded plate dataset.

Reads YOLO (.txt), COCO (.json) or Pascal VOC (.xml) annotations and always
writes YOLO, because that is what Ultralytics trains on.

    python ml/prepare_dataset.py --src <downloaded-dataset> --subset 3000 --val 400

Nothing is auto-downloaded: dataset URLs and layouts rot, and a broken
downloader at hour 0 is the worst possible time to debug one. Fetch one by hand
into --src first. Sources that work:

  https://universe.roboflow.com/search?q=indian+number+plate   (export as YOLOv8)
  https://huggingface.co/datasets/keremberke/license-plate-object-detection
  https://www.kaggle.com/datasets?search=license+plate+yolo

The three-format support is not architecture astronautics: public plate datasets
ship in all three, the format is rarely stated on the listing page, and finding
out only after unzipping 155 MB is how an afternoon disappears.

ALL kept boxes collapse to class 0 (`license_plate`). Datasets that also label
vehicles get filtered by category name, not silently merged -- a detector
trained on "plate OR car, both class 0" finds cars.
"""
import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Category names that mean "the thing we want". Matched case-insensitively as
# substrings, so "License_Plate", "number plate" and "VRP-plate" all hit.
PLATE_WORDS = ("plate", "licence", "license", "number", "vrp", "anpr", "lp")

Box = tuple[float, float, float, float]     # cx, cy, w, h -- all normalized


def _clip(v: float) -> float:
    return max(0.0, min(1.0, v))


def _to_yolo(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Box | None:
    """Corner pixels -> normalized centre/size. None if degenerate or off-image."""
    if w <= 0 or h <= 0 or x2 <= x1 or y2 <= y1:
        return None
    cx, cy = _clip((x1 + x2) / 2 / w), _clip((y1 + y2) / 2 / h)
    bw, bh = _clip((x2 - x1) / w), _clip((y2 - y1) / h)
    return (cx, cy, bw, bh) if bw > 0 and bh > 0 else None


def _index_images(src: Path) -> dict[str, Path]:
    """Image basename -> path. Basename because COCO/VOC reference files by name
    while the images may sit in any subdirectory."""
    return {p.name: p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT}


# ------------------------------------------------------------------ loaders --

def load_yolo(src: Path, images: dict[str, Path]) -> dict[Path, list[Box]]:
    by_stem = {p.stem: p for p in images.values()}
    out: dict[Path, list[Box]] = {}
    for lbl in src.rglob("*.txt"):
        if lbl.name in {"classes.txt", "requirements.txt"} or lbl.stem not in by_stem:
            continue
        boxes = []
        for line in lbl.read_text(errors="replace").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            vals = []
            for v in parts[1:]:          # stop at the first non-number: some
                try:                     # exports append `difficult`, `crowd`...
                    vals.append(float(v))
                except ValueError:
                    break
            if len(vals) >= 8 and len(vals) % 2 == 0:
                # Roboflow's "YOLOv8" export is sometimes INSTANCE SEGMENTATION:
                # `class x1 y1 x2 y2 ...` polygon points rather than a box.
                # Reading fields 1-4 of that gives a box built from the first two
                # polygon vertices -- garbage that still trains, just on nonsense.
                # Collapse the polygon to its bounding box instead.
                # ponytail: 8 not 6, because `class cx cy w h conf track_id` is
                # also 6 even values and is FAR commoner than a 3-vertex polygon.
                # A real segmentation mask is a quad or better. A triangle label
                # loses its third vertex; nothing else here can tell them apart.
                xs, ys = vals[0::2], vals[1::2]
                x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
                cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            elif len(vals) >= 4:
                cx, cy, bw, bh = vals[:4]   # trailing conf/track_id ignored
            else:
                continue
            if bw > 0 and bh > 0:
                boxes.append((_clip(cx), _clip(cy), _clip(bw), _clip(bh)))
        if boxes:
            out[by_stem[lbl.stem]] = boxes
    return out


def load_coco(src: Path, images: dict[str, Path]) -> dict[Path, list[Box]]:
    out: dict[Path, list[Box]] = {}
    for js in src.rglob("*.json"):
        try:
            doc = json.loads(js.read_text(errors="replace"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not (isinstance(doc, dict) and "annotations" in doc and "images" in doc):
            continue

        cats = {c["id"]: str(c.get("name", "")).lower() for c in doc.get("categories", [])}
        plate_ids = {i for i, n in cats.items() if any(w in n for w in PLATE_WORDS)}
        # No category looks like a plate: either it is a single-class set with an
        # unhelpful name, or we would be guessing. Keep everything only when
        # there is exactly one class, so we can never silently merge cars in.
        if not plate_ids:
            plate_ids = set(cats) if len(cats) <= 1 else set()
            if cats and not plate_ids:
                print(f"warning: {js.name} has categories {sorted(cats.values())} and "
                      f"none look like a plate -- skipping this file")
                continue

        meta = {im["id"]: im for im in doc["images"]}
        for ann in doc["annotations"]:
            if plate_ids and ann.get("category_id") not in plate_ids:
                continue
            im = meta.get(ann.get("image_id"))
            bbox = ann.get("bbox")
            if not im or not bbox or len(bbox) != 4:
                continue
            path = images.get(Path(str(im.get("file_name", ""))).name)
            if not path:
                continue
            x, y, bw, bh = (float(v) for v in bbox)     # COCO: x, y, width, height
            box = _to_yolo(x, y, x + bw, y + bh, int(im.get("width", 0)), int(im.get("height", 0)))
            if box:
                out.setdefault(path, []).append(box)
    return out


def load_voc(src: Path, images: dict[str, Path]) -> dict[Path, list[Box]]:
    out: dict[Path, list[Box]] = {}
    for xml in src.rglob("*.xml"):
        try:
            root = ET.parse(xml).getroot()
        except ET.ParseError:
            continue
        size = root.find("size")
        if size is None:
            continue
        w = int(float(size.findtext("width", "0")))
        h = int(float(size.findtext("height", "0")))
        name = root.findtext("filename") or f"{xml.stem}.jpg"
        path = images.get(Path(name).name) or images.get(f"{xml.stem}.jpg")
        if not path:
            continue
        boxes = []
        for obj in root.findall("object"):
            label = (obj.findtext("name") or "").lower()
            if label and not any(word in label for word in PLATE_WORDS):
                continue
            bb = obj.find("bndbox")
            if bb is None:
                continue
            box = _to_yolo(float(bb.findtext("xmin", "0")), float(bb.findtext("ymin", "0")),
                           float(bb.findtext("xmax", "0")), float(bb.findtext("ymax", "0")), w, h)
            if box:
                boxes.append(box)
        if boxes:
            out[path] = boxes
    return out


def find_pairs(src: Path) -> tuple[dict[Path, list[Box]], str]:
    """First loader that finds anything wins. YOLO first because it needs no
    conversion and is what a Roboflow YOLOv8 export gives."""
    images = _index_images(src)
    if not images:
        raise SystemExit(f"no images found under {src}")
    for name, loader in (("YOLO", load_yolo), ("COCO", load_coco), ("VOC", load_voc)):
        found = loader(src, images)
        if found:
            return found, name
    raise SystemExit(
        f"{len(images)} images under {src} but no readable annotations "
        f"(.txt / .json / .xml).\n"
        f"This is probably a classification or OCR dataset -- plate crops with the "
        f"text as a label, no bounding boxes. Those train a reader, not a detector.\n"
        f"Try a Roboflow Universe export in YOLOv8 format instead."
    )


def write_split(pairs: list[tuple[Path, list[Box]]], dst: Path, split: str) -> None:
    for sub in ("images", "labels"):
        (dst / sub / split).mkdir(parents=True, exist_ok=True)
    for img, boxes in pairs:
        shutil.copy2(img, dst / "images" / split / img.name)
        (dst / "labels" / split / f"{img.stem}.txt").write_text(
            "".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for cx, cy, w, h in boxes)
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="downloaded dataset root")
    ap.add_argument("--dst", type=Path, default=Path("datasets/plates"))
    ap.add_argument("--subset", type=int, default=3000, help="train images")
    ap.add_argument("--val", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    found, fmt = find_pairs(args.src)
    pairs = sorted(found.items())
    print(f"{fmt} annotations: {len(pairs)} labelled images, "
          f"{sum(len(b) for b in found.values())} boxes")

    random.Random(args.seed).shuffle(pairs)
    val = pairs[: args.val]
    train = pairs[args.val : args.val + args.subset]
    if len(train) < args.subset:
        print(f"warning: only {len(train)} train images available, wanted {args.subset}")
    if not train:
        raise SystemExit("no training images left after the val split -- lower --val")

    if args.dst.exists():
        shutil.rmtree(args.dst)
    write_split(train, args.dst, "train")
    write_split(val, args.dst, "val")

    yaml = args.dst / "data.yaml"
    yaml.write_text(
        f"path: {args.dst.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: [license_plate]\n"
    )
    print(f"{len(train)} train / {len(val)} val -> {yaml}")


def _selfcheck() -> None:
    """All three readers produce identical YOLO boxes from the same geometry."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        # One 200x100 image, one box at pixels (50,20)-(150,60).
        # Expected YOLO: cx=0.5, cy=0.4, w=0.5, h=0.4
        want = (0.5, 0.4, 0.5, 0.4)

        y = root / "yolo"; (y / "images").mkdir(parents=True); (y / "labels").mkdir()
        (y / "images" / "a.jpg").write_bytes(b"x")
        (y / "labels" / "a.txt").write_text("0 0.5 0.4 0.5 0.4\n")
        got, fmt = find_pairs(y)
        assert fmt == "YOLO" and len(got) == 1
        assert all(abs(a - b) < 1e-6 for a, b in zip(list(got.values())[0][0], want))

        # Roboflow segmentation export: polygon points, not a box. Same geometry.
        seg = root / "seg"; (seg / "images").mkdir(parents=True); (seg / "labels").mkdir()
        (seg / "images" / "a.jpg").write_bytes(b"x")
        (seg / "labels" / "a.txt").write_text(
            "0 0.25 0.2 0.75 0.2 0.75 0.6 0.25 0.6 0.25 0.2\n")
        got, fmt = find_pairs(seg)
        assert fmt == "YOLO"
        assert all(abs(a - b) < 1e-6 for a, b in zip(list(got.values())[0][0], want)), \
            f"polygon must collapse to its bounding box, got {list(got.values())[0][0]}"

        # Same box carrying trailing fields -- `conf`, `track_id`, `difficult`.
        # 6 even values here must NOT be mistaken for a 3-vertex polygon.
        for tail in ("0.97", "0.97 3", "difficult"):
            t = root / f"tail{len(tail)}"
            (t / "images").mkdir(parents=True); (t / "labels").mkdir()
            (t / "images" / "a.jpg").write_bytes(b"x")
            (t / "labels" / "a.txt").write_text(f"0 0.5 0.4 0.5 0.4 {tail}\n")
            got, fmt = find_pairs(t)
            assert fmt == "YOLO", f"{tail!r} -> {fmt}"
            assert all(abs(a - b) < 1e-6 for a, b in zip(list(got.values())[0][0], want)), \
                f"trailing {tail!r} corrupted the box: {list(got.values())[0][0]}"

        c = root / "coco"; c.mkdir(); (c / "a.jpg").write_bytes(b"x")
        (c / "ann.json").write_text(json.dumps({
            "images": [{"id": 1, "file_name": "a.jpg", "width": 200, "height": 100}],
            "categories": [{"id": 7, "name": "License_Plate"}],
            "annotations": [{"image_id": 1, "category_id": 7, "bbox": [50, 20, 100, 40]}],
        }))
        got, fmt = find_pairs(c)
        assert fmt == "COCO", fmt
        assert all(abs(a - b) < 1e-6 for a, b in zip(list(got.values())[0][0], want))

        v = root / "voc"; v.mkdir(); (v / "a.jpg").write_bytes(b"x")
        (v / "a.xml").write_text(
            "<annotation><filename>a.jpg</filename>"
            "<size><width>200</width><height>100</height></size>"
            "<object><name>number plate</name>"
            "<bndbox><xmin>50</xmin><ymin>20</ymin><xmax>150</xmax><ymax>60</ymax></bndbox>"
            "</object></annotation>")
        got, fmt = find_pairs(v)
        assert fmt == "VOC", fmt
        assert all(abs(a - b) < 1e-6 for a, b in zip(list(got.values())[0][0], want))

        # A car box in the same COCO file must NOT become a plate. Training on
        # "plate OR car, both class 0" produces a detector that finds cars.
        m = root / "mixed"; m.mkdir(); (m / "a.jpg").write_bytes(b"x")
        (m / "ann.json").write_text(json.dumps({
            "images": [{"id": 1, "file_name": "a.jpg", "width": 200, "height": 100}],
            "categories": [{"id": 1, "name": "car"}, {"id": 2, "name": "plate"}],
            "annotations": [
                {"image_id": 1, "category_id": 1, "bbox": [0, 0, 200, 100]},
                {"image_id": 1, "category_id": 2, "bbox": [50, 20, 100, 40]},
            ],
        }))
        got, _ = find_pairs(m)
        boxes = list(got.values())[0]
        assert len(boxes) == 1, f"car box leaked in: {boxes}"
        assert all(abs(a - b) < 1e-6 for a, b in zip(boxes[0], want))

        # Degenerate boxes are dropped, not written as zero-area labels.
        d = root / "degen"; d.mkdir(); (d / "a.jpg").write_bytes(b"x")
        (d / "ann.json").write_text(json.dumps({
            "images": [{"id": 1, "file_name": "a.jpg", "width": 200, "height": 100}],
            "categories": [{"id": 1, "name": "plate"}],
            "annotations": [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 0, 0]}],
        }))
        try:
            find_pairs(d)
            raise AssertionError("a zero-area box must not count as an annotation")
        except SystemExit:
            pass

        # Images with no annotations at all: a clear message, not a crash.
        n = root / "none"; n.mkdir(); (n / "a.jpg").write_bytes(b"x")
        try:
            find_pairs(n)
            raise AssertionError("should have exited")
        except SystemExit as e:
            assert "classification or OCR dataset" in str(e)

        # Splits stay disjoint.
        big = root / "big"; (big / "images").mkdir(parents=True); (big / "labels").mkdir()
        for i in range(20):
            (big / "images" / f"{i}.jpg").write_bytes(b"x")
            (big / "labels" / f"{i}.txt").write_text("0 0.5 0.5 0.2 0.1\n")
        found, _ = find_pairs(big)
        pairs = sorted(found.items())
        random.Random(0).shuffle(pairs)
        val, train = pairs[:5], pairs[5:15]
        assert not ({p[0] for p in val} & {p[0] for p in train}), "val leaked into train"

        out = root / "out"
        write_split(train, out, "train")
        assert len(list((out / "images" / "train").iterdir())) == 10
        assert len(list((out / "labels" / "train").iterdir())) == 10
        assert (out / "labels" / "train" / f"{train[0][0].stem}.txt").read_text().startswith("0 ")

    print("selfcheck ok")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
