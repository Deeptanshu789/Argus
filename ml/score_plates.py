#!/usr/bin/env python3
"""Measure REAL plate-reading accuracy against hand-written ground truth.

    ./.venv/bin/python ml/score_plates.py --data ~/indian-plates

`ml/validate_plate.py` reports YIELD: how many reads `correct_plate()` will
accept. Yield cannot tell a correct read from a confident wrong one, so it is
an upper bound and nothing more. This script compares against plates a human
transcribed from the images and reports the number you are allowed to quote.

Three outcomes matter and they are not interchangeable:

  correct    the string matches the plate exactly
  wrong      a plate was read and it is NOT the right one  <- the dangerous one
  missed     no plate was read at all

A wrong read is worse than a miss. A missed plate leaves the vehicle to be
matched by Re-ID; a wrong one puts a real registration on the wrong vehicle,
and in an ANPR system aimed at enforcement that is the failure that matters.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sidecar import (  # noqa: E402
    PLATE_IMGSZ, _pad, _read_plate, find_plate_weights, make_reader,
)


def char_errors(a: str, b: str) -> int:
    """Levenshtein distance. Two plates differing in one character are a much
    better result than two sharing nothing, and exact-match hides that."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path.home() / "indian-plates")
    ap.add_argument("--truth", type=Path, default=Path(__file__).parent / "groundtruth_test50.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--show", action="store_true", help="list every mismatch")
    ap.add_argument("--pad", type=float, default=None,
                    help="crop padding as a fraction of box size; default is the "
                         "sidecar's PLATE_PAD. A detector that draws tighter boxes "
                         "needs more padding, so this is worth re-measuring after "
                         "every retrain.")
    args = ap.parse_args()

    if args.pad is not None:
        import sidecar
        sidecar.PLATE_PAD = args.pad

    weights = args.model or find_plate_weights()
    if weights is None:
        sys.exit("no trained weights; train on Kaggle first (WORKFLOW.md Stage 1)")

    rows = [r for r in csv.reader(
        l for l in args.truth.read_text().splitlines() if l and not l.startswith("#"))]
    print(f"model : {weights}\ntruth : {args.truth} ({len(rows)} images)\n")

    import cv2
    from ultralytics import YOLO
    model = YOLO(weights)
    reader = make_reader()

    folder = args.data / args.split / "images"
    correct = wrong = missed = 0
    illegible_read = illegible_total = 0
    false_positive = notplate_total = 0
    chars = truth_chars = 0
    mismatches = []

    for expected, name in rows:
        path = folder / name
        img = cv2.imread(str(path))
        if img is None:
            print(f"  missing image: {name}")
            continue

        det = model(img, imgsz=PLATE_IMGSZ, conf=args.conf, verbose=False)[0]
        got = None
        if len(det.boxes):
            box = max(det.boxes, key=lambda b: float(b.conf[0]))
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            # Same padding the sidecar applies — measuring anything else would
            # report a number the running system does not produce.
            got, _ = _read_plate(reader, None, _pad(img, x1, y1, x2, y2))

        if expected == "-":                     # not a plate at all
            notplate_total += 1
            if got:
                false_positive += 1
                mismatches.append((name, "NOT A PLATE", got))
            continue
        if expected == "?":                     # illegible to a human too
            illegible_total += 1
            if got:
                illegible_read += 1
            continue

        truth_chars += len(expected)
        if got == expected:
            correct += 1
        elif got is None:
            missed += 1
            chars += len(expected)
            mismatches.append((name, expected, "(no read)"))
        else:
            wrong += 1
            chars += char_errors(got, expected)
            mismatches.append((name, expected, got))

    legible = correct + wrong + missed
    print(f"legible plates            {legible}")
    print(f"  correct                 {correct:3d}  ({correct / max(legible,1):.0%})")
    print(f"  wrong                   {wrong:3d}  ({wrong / max(legible,1):.0%})")
    print(f"  missed                  {missed:3d}  ({missed / max(legible,1):.0%})")
    print(f"  character error rate    {chars / max(truth_chars,1):.1%}")
    if wrong + correct:
        print(f"  precision when it answers {correct / (correct + wrong):.0%}"
              f"   <- trust in a read that IS produced")
    print(f"\nillegible to a human      {illegible_total}"
          f"   (model produced a plate for {illegible_read} of them)")
    print(f"boxes with no plate       {notplate_total}"
          f"   (false positives: {false_positive})")

    if args.show and mismatches:
        print("\nmismatches:")
        for name, want, got in mismatches:
            print(f"  {want:12} got {got:12}  {name[:46]}")


if __name__ == "__main__":
    main()
