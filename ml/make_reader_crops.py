#!/usr/bin/env python3
"""Cut real plate crops out of a labelled detection dataset and read them, so
the reader can be trained on photographs instead of renders.

    python ml/make_reader_crops.py --src datasets/plates-merged-v2 \
                                   --dst datasets/reader-real

WHY THIS EXISTS. ml/train_reader.py's two public datasets are both synthetic.
A CRNN trained on them reaches 83% on held-out renders and reads 0 of 45 real
photographs -- it has learned a font, not a plate. The fix is real crops, and
this project already has 10,240 real photographs with hand-drawn plate boxes.
What it does not have is the TEXT on any of them.

So the text comes from PaddleOCR, which reads 39 of those 45 correctly. That is
pseudo-labelling, and its ceiling is honest: the reader cannot learn to be
better than PaddleOCR on the cases PaddleOCR gets wrong. It can be better
everywhere else -- these crops teach real fonts, real dirt, real angles and real
lighting, which is the part the renders cannot teach -- and it is far faster at
inference. Mixed with the synthetic set, whose labels are exact, the pseudo-
labels act as domain evidence rather than as the sole source of truth.

BOXES COME FROM THE DATASET'S OWN LABELS, NOT FROM THE DETECTOR. They are
hand-drawn and correct, they cost nothing to read, and using the detector here
would bake its mistakes into the reader's training set as well as the OCR's.

Only reads that correct_plate() accepts with little repair are kept. A crop
whose text is uncertain is dropped rather than guessed: a wrong label is worse
than a missing one, because the model has no way to tell it is wrong.
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sidecar as S  # noqa: E402

sys.stdout = S._PROTOCOL

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Width over height of a crop that could hold an Indian registration plate.
# Two lines is about 2:1, one line about 4.5:1; the bounds are loose enough for
# padding and a skewed view, and tight enough to throw out a box that is taller
# than it is wide.
PLATE_ASPECT = (1.2, 8.0)


def crops_of(img, boxes, pad: float):
    """Every labelled plate in one image, padded the way the sidecar pads."""
    h, w = img.shape[:2]
    for cx, cy, bw, bh in boxes:
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        if x2 - x1 < 24 or y2 - y1 < 8:
            continue                      # too small to carry readable text
        aspect = (x2 - x1) / (y2 - y1)
        if not PLATE_ASPECT[0] <= aspect <= PLATE_ASPECT[1]:
            # Not a plate whatever the label says. This dataset contains boxes
            # like 106x439 -- taller than they are wide, so nothing inside can
            # be a registration plate -- and OCR will still return SOMETHING
            # from one, which then gets written down as ground truth.
            continue
        px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
        crop = img[max(y1 - py, 0):min(y2 + py, h), max(x1 - px, 0):min(x2 + px, w)]
        if crop.size:
            yield crop


def read_boxes(label: Path) -> list[tuple[float, float, float, float]]:
    out = []
    for line in label.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            try:
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
            except ValueError:
                continue
            if bw > 0 and bh > 0:
                out.append((cx, cy, bw, bh))
    return out


def read_crop(reader, crop, min_conf: float, try_flip: bool = True):
    """(plate, confidence, image) for one crop, or None.

    EVERY CROP IS READ BOTH WAYS ROUND.

    About a third of the Roboflow export is horizontally flipped -- their
    commonest augmentation, and harmless to a detector, which still sees a
    plate. To a reader it is poison: the text runs backwards, PaddleOCR reads it
    anyway, and correct_plate() accepts the result because a reversed plate is
    often still a grammatical one. TN21TA0492 was being written down as
    TN21AT0492, and nothing downstream could ever tell.

    A real plate reads confidently one way round and badly the other, so the
    higher-scoring orientation IS the right one. That rejects the mirrors and
    repairs them in the same pass.

    try_flip=False for VIDEO. Mirrored plates are an artefact of how Roboflow
    exports augmented copies, not something a camera produces, and the second
    orientation doubles the OCR cost -- which is the whole cost of reading a
    frame of Delhi traffic with ten vehicles in it.
    """
    import cv2
    best = None
    for flipped in ((False, True) if try_flip else (False,)):
        view = cv2.flip(crop, 1) if flipped else crop
        for height, equalise in S._OCR_VARIANTS:
            lines = S._ocr_lines(reader, S._prepare_crop(view, height, equalise))
            if not lines:
                continue
            joined = "".join(t for t, _ in lines)
            mean = sum(sc for _, sc in lines) / len(lines)
            for raw, score in [(joined, mean)] + sorted(lines, key=lambda ts: -len(ts[0])):
                plate, penalty = S.correct_plate(raw)
                if plate is None:
                    continue
                conf = score - penalty
                if best is None or conf > best[1]:
                    best = (plate, conf, view)
            if best is not None:
                break
    return best if best and best[1] >= min_conf else None


PER_PLATE_CAP = 14


def voc_crops(xml: Path, out, dst: Path, kept: int, pad: float) -> int:
    """Crops from one Pascal VOC file whose <name> carries the plate text.

    The label is the dataset's own, so correct_plate() only has to agree that
    it is a real registration -- a file whose name is a class word like "plate"
    or a transcription that breaks Indian grammar is dropped rather than
    written, because a wrong label teaches the reader an untruth it cannot
    detect."""
    import cv2
    import xml.etree.ElementTree as ET
    try:
        root = ET.parse(xml).getroot()
    except ET.ParseError:
        return 0
    img_path = xml.with_suffix(".jpg")
    if not img_path.exists():
        img_path = xml.with_suffix(".jpeg")
    if not img_path.exists():
        img_path = xml.with_suffix(".png")
    img = cv2.imread(str(img_path))
    if img is None:
        return 0
    h, w = img.shape[:2]
    got = 0
    for obj in root.findall("object"):
        text, _ = S.correct_plate((obj.findtext("name") or "").upper())
        if not text:
            continue
        b = obj.find("bndbox")
        if b is None:
            continue
        x1, y1 = int(b.findtext("xmin", "0")), int(b.findtext("ymin", "0"))
        x2, y2 = int(b.findtext("xmax", "0")), int(b.findtext("ymax", "0"))
        bw, bh = x2 - x1, y2 - y1
        if bw < 24 or bh < 8:
            continue
        aspect = bw / bh
        if not PLATE_ASPECT[0] <= aspect <= PLATE_ASPECT[1]:
            continue
        x1, y1 = max(0, int(x1 - bw * pad)), max(0, int(y1 - bh * pad))
        x2, y2 = min(w, int(x2 + bw * pad)), min(h, int(y2 + bh * pad))
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        name = f"{kept + got:06d}_{text}.jpg"
        cv2.imwrite(str(dst / "images" / name), crop)
        out.write(f"{name}\t{text}\n")
        got += 1
    return got


def video_crops(path, reader, out, args, pad: float) -> int:
    """Real crops from traffic footage, through the sidecar's own two stages.

    Vehicle detector then plate detector, not the plate detector on the whole
    frame: that is what ml/sidecar.py does, so the crops carry the distances,
    angles and motion blur the reader will actually be handed. A frame with no
    vehicle costs one detector pass and nothing else.

    PER_PLATE_CAP stops one car that sits at a red light for nine seconds from
    contributing four hundred near-identical crops and teaching the model that
    plate above all others.
    """
    import cv2
    from ultralytics import YOLO

    vehicle = YOLO(S.VEHICLE_WEIGHTS)
    weights = S.find_plate_weights()
    plate = YOLO(weights) if weights else None
    if plate is None:
        print("  no plate detector; skipping video")
        return 0

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  cannot open {path}")
        return 0
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(src_fps / max(args.fps, 1)))

    per_plate: dict[str, int] = {}
    kept = raw = processed = 0
    while processed < args.max_frames:
        ok = cap.grab()
        if not ok:
            break
        raw += 1
        if raw % step:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        processed += 1

        res = vehicle(frame, classes=list(S.VEHICLE_CLASSES), imgsz=S.VEHICLE_IMGSZ,
                      conf=S.VEHICLE_CONF, verbose=False)[0]
        for box in (res.boxes or []):
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            if x2 - x1 < S.PLATE_MIN_VEHICLE_PX:
                continue          # never yields a readable plate; see sidecar.py
            veh = frame[max(y1, 0):y2, max(x1, 0):x2]
            if veh.size == 0:
                continue
            pres = plate(veh, imgsz=S.PLATE_IMGSZ, conf=S.PLATE_CONF, verbose=False)[0]
            if not len(pres.boxes):
                continue
            b = max(pres.boxes, key=lambda b: float(b.conf[0]))
            px1, py1, px2, py2 = (int(v) for v in b.xyxy[0])
            if not PLATE_ASPECT[0] <= (px2 - px1) / max(py2 - py1, 1) <= PLATE_ASPECT[1]:
                continue
            crop = S._pad(veh, px1, py1, px2, py2)
            if crop.size == 0:
                continue
            best = read_crop(reader, crop, args.min_conf, try_flip=False)
            if best is None:
                continue
            if per_plate.get(best[0], 0) >= PER_PLATE_CAP:
                continue
            per_plate[best[0]] = per_plate.get(best[0], 0) + 1
            name = f"v{abs(hash(path.name)) % 9999:04d}_{kept:06d}_{best[0]}.jpg"
            cv2.imwrite(str(args.dst / "images" / name), best[2])
            out.write(f"{name}\t{best[0]}\n")
            kept += 1
        if processed % 100 == 0:
            print(f"    {path.name[:34]}: {processed} frames, {kept} crops, "
                  f"{len(per_plate)} distinct plates", flush=True)
    cap.release()
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # SEVERAL SOURCES, and never the split that groundtruth_test50.csv is drawn
    # from. Training the reader on the 45 photographs it is scored against would
    # turn the only honest measurement in this project into a memory test.
    ap.add_argument("--src", nargs="*", type=Path,
                    default=[Path.home() / "indian-plates" / "train",
                             Path.home() / "indian-plates" / "valid"])
    ap.add_argument("--dst", type=Path, default=Path("datasets/reader-real"))
    ap.add_argument("--pad", type=float, default=None, help="default: sidecar PLATE_PAD")
    ap.add_argument("--min-conf", type=float, default=0.6,
                    help="OCR score minus correction penalty. Below this the read "
                         "is dropped rather than written as a label")
    ap.add_argument("--limit", type=int, default=0, help="stop after N images")
    # VIDEO IS THE ONLY PLENTIFUL SOURCE OF REAL CROPS.
    #
    # The labelled photograph sets yield under two thousand crops between them,
    # of which 49% are Maharashtra. A few minutes of traffic footage yields
    # thousands, from moving vehicles at the distances and angles the sidecar
    # actually sees -- which is the whole point. There are no boxes here, so the
    # plate detector draws them; its mistakes cost a junk crop that OCR then
    # fails to read, not a wrong label.
    # ALREADY-CROPPED PLATES, with no box file and no text in the filename.
    # Some published sets ship the crop itself rather than the photograph it
    # came from, so there is nothing to cut out -- only something to read.
    ap.add_argument("--crops", nargs="*", type=Path, default=[])
    # PASCAL VOC WHERE <name> IS THE PLATE TEXT, NOT A CLASS.
    # saisirishan/indian-vehicle-dataset files 1,697 photographs by state and
    # puts the registration itself in the object name. That is ground truth,
    # not a pseudo-label, so OCR never runs on these and the reader is not
    # capped at PaddleOCR's accuracy on them.
    ap.add_argument("--voc", nargs="*", type=Path, default=[])
    ap.add_argument("--video", nargs="*", type=Path, default=[])
    ap.add_argument("--fps", type=int, default=5, help="frames per second to sample")
    ap.add_argument("--max-frames", type=int, default=1500, help="per video")
    args = ap.parse_args()

    pad = S.PLATE_PAD if args.pad is None else args.pad
    import cv2

    images = [p for d in args.src for p in sorted(d.rglob("*"))
              if p.suffix.lower() in IMG_EXT]
    pre = [p for d in args.crops for p in sorted(d.rglob("*"))
           if p.suffix.lower() in IMG_EXT]
    vocs = [x for d in args.voc for x in sorted(d.rglob("*.xml"))]
    if not images and not args.video and not pre and not vocs:
        sys.exit(f"no images under {args.src}")
    print(f"{len(images)} image(s) from {len(args.src)} source(s), pad {pad}")

    reader = S.make_reader()
    if reader is None:
        sys.exit("no OCR available; nothing can be labelled")

    if args.dst.exists():
        shutil.rmtree(args.dst)
    (args.dst / "images").mkdir(parents=True)
    labels_file = args.dst / "labels.txt"

    kept = seen = no_box = no_read = low = mirrored = 0
    with labels_file.open("w") as out:
        for n, img_path in enumerate(images, 1):
            if args.limit and n > args.limit:
                break
            label = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
            if not label.exists():
                no_box += 1
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            for crop in crops_of(img, read_boxes(label), pad):
                seen += 1
                best = read_crop(reader, crop, args.min_conf)
                if best is None:
                    no_read += 1
                    continue
                if best[2] is not crop:
                    mirrored += 1
                name = f"{kept:06d}_{best[0]}.jpg"
                cv2.imwrite(str(args.dst / "images" / name), best[2])
                out.write(f"{name}\t{best[0]}\n")
                kept += 1
            if n % 250 == 0:
                print(f"  {n}/{len(images)} images, {kept} crops kept "
                      f"({no_read} unread, {low} below {args.min_conf})", flush=True)

        for n, crop_path in enumerate(pre, 1):
            crop = cv2.imread(str(crop_path))
            if crop is None:
                continue
            seen += 1
            best = read_crop(reader, crop, args.min_conf)
            if best is None:
                no_read += 1
                continue
            if best[2] is not crop:
                mirrored += 1
            name = f"{kept:06d}_{best[0]}.jpg"
            cv2.imwrite(str(args.dst / "images" / name), best[2])
            out.write(f"{name}\t{best[0]}\n")
            kept += 1
            if n % 250 == 0:
                print(f"  {n}/{len(pre)} crops, {kept} kept", flush=True)

        for n, xml in enumerate(vocs, 1):
            got = voc_crops(xml, out, args.dst, kept, pad)
            kept += got
            seen += 1
            if not got:
                no_read += 1
            if n % 250 == 0:
                print(f"  {n}/{len(vocs)} annotations, {kept} kept", flush=True)

        for vid in args.video:
            got = video_crops(vid, reader, out, args, pad)
            kept += got
            print(f"  {vid.name}: {got} crop(s)", flush=True)

    print(f"\n{kept} crop(s) written to {args.dst}")
    print(f"  {seen} labelled boxes, {no_read} unread, {low} below --min-conf, "
          f"{no_box} image(s) with no label file")
    print(f"  {mirrored} crop(s) were mirrored in the source and were un-mirrored")
    if seen:
        print(f"  yield {kept / seen * 100:.0f}% of labelled boxes")


if __name__ == "__main__":
    main()
