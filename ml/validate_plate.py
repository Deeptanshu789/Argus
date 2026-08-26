#!/usr/bin/env python3
"""Measure the trained plate detector, and the OCR that runs behind it.

    ./.venv/bin/python ml/validate_plate.py --data ~/indian-plates

Two numbers, and they answer different questions.

DETECTION (mAP50, precision, recall)
    Does the model find plates? Needs the dataset's labels, so it reports
    whatever `data.yaml` points at. This is the number Kaggle already printed;
    running it here re-measures the EXPORTED weights on a HELD-OUT set, which
    is the only version of the claim worth repeating to anyone.

OCR YIELD (--ocr)
    Of the plates the detector found, how many produce a string that
    `correct_plate()` will accept as a real Indian registration?

    This is NOT accuracy. The dataset has no ground-truth text, so nothing here
    can tell a correct read from a confident wrong one. Yield is an upper bound
    on accuracy and a lower bound on nothing. It is still the number that
    predicts what a demo looks like: a detector at 0.95 mAP paired with 20%
    yield shows boxes and no plates, and the failure is invisible in mAP alone.

Measuring real accuracy needs plate strings written down by hand for a few
dozen images. Half an hour of work, and worth it before anyone quotes a number.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sidecar import (  # noqa: E402
    PLATE_IMGSZ, correct_plate, find_plate_weights, make_reader,
)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path.home() / "indian-plates",
                    help="dataset root containing data.yaml")
    ap.add_argument("--model", default=None, help="default: the exported weights")
    ap.add_argument("--split", default="test", choices=["train", "val", "valid", "test"])
    ap.add_argument("--ocr", action="store_true", help="also measure OCR yield")
    ap.add_argument("--limit", type=int, default=150, help="images for the OCR pass")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    weights = args.model or find_plate_weights()
    if weights is None:
        sys.exit("no trained weights found.\n"
                 "Expected runs/detect/plate/weights/best.pt — train on Kaggle "
                 "first (WORKFLOW.md Stage 1), or pass --model.")
    print(f"model   : {weights}")

    from ultralytics import YOLO
    model = YOLO(weights)

    # ---------------------------------------------------------- detection --
    yaml_path = args.data / "data.yaml"
    if yaml_path.exists():
        print(f"dataset : {yaml_path}  split={args.split}\n")
        try:
            m = model.val(data=str(yaml_path), split=args.split,
                          imgsz=PLATE_IMGSZ, conf=0.001, verbose=False).box
            print(f"  mAP50      {m.map50:.3f}")
            print(f"  mAP50-95   {m.map:.3f}")
            print(f"  precision  {m.mp:.3f}")
            print(f"  recall     {m.mr:.3f}")
        except Exception as exc:
            print(f"  detection metrics unavailable: {type(exc).__name__}: {exc}")
            print(f"  (an OpenVINO export cannot be validated — pass "
                  f"--model runs/detect/plate/weights/best.pt)")
    else:
        print(f"no {yaml_path}; skipping detection metrics\n")

    if not args.ocr:
        print("\nadd --ocr to measure how many of those plates are actually readable")
        return

    # -------------------------------------------------------------- OCR --
    folder = args.data / args.split / "images"
    if not folder.is_dir():
        folder = args.data / args.split
    images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT)[: args.limit]
    if not images:
        sys.exit(f"no images under {folder}")

    import cv2
    reader = make_reader()
    if reader is None:
        sys.exit("OCR unavailable")

    from sidecar import _read_plate

    found = accepted = 0
    rejected: list[str] = []
    plates: list[str] = []
    t0 = time.time()

    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            continue
        res = model(img, imgsz=PLATE_IMGSZ, conf=args.conf, verbose=False)[0]
        if not len(res.boxes):
            continue
        found += 1
        box = max(res.boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        crop = img[max(y1, 0):y2, max(x1, 0):x2]
        # plate_model=None: the crop already IS the plate.
        plate, conf = _read_plate(reader, None, crop)
        if plate:
            accepted += 1
            plates.append(f"{plate} ({conf})")
        else:
            rejected.append(p.name)

    elapsed = time.time() - t0
    print(f"\nOCR over {len(images)} images, {elapsed:.0f}s "
          f"({elapsed / max(len(images), 1) * 1000:.0f} ms/image)")
    print(f"  plate detected     {found}/{len(images)}  "
          f"({found / max(len(images), 1):.0%})")
    print(f"  read and accepted  {accepted}/{found}  "
          f"({accepted / max(found, 1):.0%} yield)")
    print(f"  end to end         {accepted}/{len(images)}  "
          f"({accepted / max(len(images), 1):.0%})")
    if plates:
        print("\n  sample reads: " + ", ".join(plates[:8]))
    print("\nYield is an UPPER BOUND on accuracy: nothing here checks a read "
          "against\nthe true plate, so a confident wrong answer counts as a "
          "success.")


if __name__ == "__main__":
    main()
