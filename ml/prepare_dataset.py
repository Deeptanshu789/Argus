#!/usr/bin/env python3
"""Carve a training-sized YOLO subset out of one or more downloaded plate datasets.

Reads YOLO (.txt), COCO (.json) or Pascal VOC (.xml) annotations and always
writes YOLO, because that is what Ultralytics trains on.

    python ml/prepare_dataset.py --src <downloaded-dataset> --subset 3000 --val 400

SEVERAL SOURCES MERGE INTO ONE SET. YOLO does not care where an image came
from, and no public Indian plate dataset is large enough on its own:

    python ml/prepare_dataset.py --src ~/indian-plates ~/kaggle-plates \
                                 --subset 8000 --val 800

Merging is not concatenation, for two reasons.

Basenames collide. Half these datasets export images as `0001.jpg`, and a plain
copy silently overwrites. Every image is written under a name carrying its
source index, so the collision cannot happen and the origin stays readable.

And the SAME IMAGES appear in several datasets. Public plate sets are largely
re-uploads of each other, often re-encoded, so the same photograph arrives with
a different byte count and a different name. One copy in train and another in
val is a leak: the model is validated on a picture it was trained on, val mAP
rises, and every honest measurement downstream is quietly wrong. --dedupe (on
by default) drops repeats by perceptual hash before anything is split.

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
    found: dict[str, Path] = {}
    shadowed = 0
    for p in src.rglob("*"):
        if p.suffix.lower() not in IMG_EXT:
            continue
        if p.name in found:
            # Two files with the same basename in different split folders. The
            # index is keyed by name because COCO and VOC reference images that
            # way, so one silently shadows the other and its annotations attach
            # to the wrong picture. Loud, because the training run would look
            # fine and score badly for no visible reason.
            shadowed += 1
            continue
        found[p.name] = p
    if shadowed:
        print(f"warning: {shadowed} image(s) share a basename with another and were "
              f"SKIPPED. Annotations reference images by name, so keeping both "
              f"would attach labels to the wrong file. Rename them or prepare "
              f"each split separately.")
    return found


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


def ahash(path: Path) -> str | None:
    """Perceptual hash: 8x8 grey, one bit per pixel against the mean.

    A byte hash only catches an untouched copy, and these datasets re-encode.
    Average hash survives a re-encode, a resize and a small quality change,
    which is exactly how the same photograph arrives twice. It is not a
    similarity search and is not meant to be: two DIFFERENT cars photographed
    the same way stay different, because the plate and the background differ in
    more than a handful of the 64 cells.
    """
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
    mean = small.mean()
    bits = 0
    for v in small.flatten():
        bits = (bits << 1) | int(v > mean)
    return f"{bits:016x}"


def dedupe(pairs: list[tuple[Path, list[Box]]]) -> tuple[list[tuple[Path, list[Box]]], int]:
    """Drop repeated photographs, keeping the first occurrence of each.

    First means "from the earliest --src", so the dataset listed first wins its
    own annotations for any image the later ones also carry. That is the useful
    order: put the set you trust first.
    """
    # ponytail: exact hash equality, so a re-encode harsh enough to flip one of
    # the 64 bits survives as a second copy. Matching within a Hamming distance
    # would catch those, and costs a pairwise comparison -- 10,000 images is 50
    # million of them. Worth doing only if a merge is measured to be letting
    # duplicates through; the report line below is what would show it.
    seen: set[str] = set()
    kept: list[tuple[Path, list[Box]]] = []
    dropped = 0
    for img, boxes in pairs:
        h = ahash(img)
        if h is None:              # unreadable: keep it, let training complain
            kept.append((img, boxes))
            continue
        if h in seen:
            dropped += 1
            continue
        seen.add(h)
        kept.append((img, boxes))
    return kept, dropped


def write_split(pairs: list[tuple[Path, list[Box]]], dst: Path, split: str,
                name_of: dict[Path, str] | None = None) -> None:
    for sub in ("images", "labels"):
        (dst / sub / split).mkdir(parents=True, exist_ok=True)
    for img, boxes in pairs:
        stem = (name_of or {}).get(img, img.stem)
        shutil.copy2(img, dst / "images" / split / f"{stem}{img.suffix}")
        (dst / "labels" / split / f"{stem}.txt").write_text(
            "".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for cx, cy, w, h in boxes)
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, nargs="+",
                    help="one or more downloaded dataset roots, merged in the "
                         "order given; on a duplicate image the earlier one wins")
    ap.add_argument("--dst", type=Path, default=Path("datasets/plates"))
    ap.add_argument("--subset", type=int, default=3000, help="train images")
    ap.add_argument("--val", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-dedupe", dest="dedupe", action="store_false",
                    help="keep repeated photographs. Only for measuring what "
                         "deduplication removed -- a repeat spanning train and "
                         "val inflates val mAP and hides it")
    args = ap.parse_args()

    # --dst is deleted and rewritten, and the default is a path that is itself a
    # perfectly good --src once a previous run has produced it. Deleting the
    # source mid-merge would take the images with it and leave a half-written
    # dataset that still looks valid.
    dst = args.dst.resolve()
    for src in args.src:
        src = src.resolve()
        if dst == src or dst in src.parents or src in dst.parents:
            raise SystemExit(
                f"--dst {dst} overlaps --src {src}.\n"
                f"--dst is deleted before it is written, which would destroy the "
                f"source. Point --dst somewhere else.")

    pairs: list[tuple[Path, list[Box]]] = []
    name_of: dict[Path, str] = {}
    for i, src in enumerate(args.src):
        found, fmt = find_pairs(src)
        one = sorted(found.items())
        for img, boxes in one:
            # Prefixed because half these datasets export images as 0001.jpg and
            # a plain copy into one folder overwrites silently.
            name_of[img] = f"s{i}_{img.stem}"
            pairs.append((img, boxes))
        print(f"[{i}] {src}: {fmt} annotations, {len(one)} labelled images, "
              f"{sum(len(b) for b in found.values())} boxes")

    if args.dedupe:
        before = len(pairs)
        pairs, dropped = dedupe(pairs)
        if dropped:
            print(f"dropped {dropped} repeated image(s) of {before} "
                  f"({dropped / before:.0%}) -- the same photograph in more than "
                  f"one source")

    if len(args.src) > 1:
        print(f"merged: {len(pairs)} images")

    random.Random(args.seed).shuffle(pairs)
    val = pairs[: args.val]
    train = pairs[args.val : args.val + args.subset]
    if len(train) < args.subset:
        print(f"warning: only {len(train)} train images available, wanted {args.subset}")
    if not train:
        raise SystemExit("no training images left after the val split -- lower --val")

    if args.dst.exists():
        shutil.rmtree(args.dst)
    write_split(train, args.dst, "train", name_of)
    write_split(val, args.dst, "val", name_of)

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

        # ---- merging two sources ----
        import cv2
        import numpy as np

        def photo(path: Path, seed: int, size=64) -> None:
            """Something with structure, like a photograph.

            NOT random noise: JPEG discards high frequencies, so a re-encode of
            pure noise genuinely is a different image at 8x8 and the fixture
            would be testing the codec rather than the hash. A gradient with a
            block in it is what a car against a road looks like to an 8x8
            average.
            """
            g = np.linspace(0, 255, size, dtype=np.uint8)
            img = np.repeat(g[None, :], size, axis=0)
            img = np.dstack([img, img, img])
            off = seed * 7 % (size // 2)
            img[off:off + size // 3, off:off + size // 2] = 20
            cv2.imwrite(str(path), img)

        # Same basename in both sources, DIFFERENT pictures. Both must survive,
        # and neither may overwrite the other.
        a = root / "srcA"; (a / "images").mkdir(parents=True); (a / "labels").mkdir()
        b = root / "srcB"; (b / "images").mkdir(parents=True); (b / "labels").mkdir()
        for folder, seed in ((a, 1), (b, 2)):
            photo(folder / "images" / "0001.jpg", seed)
            (folder / "labels" / "0001.txt").write_text("0 0.5 0.5 0.2 0.1\n")

        merged = [(x, y) for f in (a, b) for x, y in sorted(find_pairs(f)[0].items())]
        names = {img: f"s{i}_{img.stem}" for i, f in enumerate((a, b))
                 for img in find_pairs(f)[0]}
        kept, dropped = dedupe(merged)
        assert dropped == 0, "two different photographs must not be called duplicates"

        merged_out = root / "merged"
        write_split(kept, merged_out, "train", names)
        assert len(list((merged_out / "images" / "train").iterdir())) == 2, \
            "same basename from two sources must not overwrite"

        # The same photograph, re-encoded at a different quality and copied
        # under a different name -- which is how these datasets actually
        # overlap. It must be recognised as one image, not two.
        c = root / "srcC"; (c / "images").mkdir(parents=True); (c / "labels").mkdir()
        img = cv2.imread(str(a / "images" / "0001.jpg"))
        cv2.imwrite(str(c / "images" / "copy.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 55])
        (c / "labels" / "copy.txt").write_text("0 0.5 0.5 0.2 0.1\n")

        dup = merged + sorted(find_pairs(c)[0].items())
        kept, dropped = dedupe(dup)
        assert dropped == 1, f"a re-encoded copy must be caught, dropped {dropped}"
        assert len(kept) == 2
        assert kept[0][0].parent.parent.name == "srcA", \
            "the earlier --src must be the copy that is kept"

        # --dst inside --src would delete the source before reading it.
        import subprocess
        import sys as _sys
        r = subprocess.run(
            [_sys.executable, __file__, "--src", str(a), "--dst", str(a / "out")],
            capture_output=True, text=True)
        assert r.returncode != 0 and "overlaps --src" in (r.stdout + r.stderr), \
            "writing --dst inside --src must be refused, not silently destroy it"
        assert (a / "images" / "0001.jpg").exists(), "the source survived"

    print("selfcheck ok")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
