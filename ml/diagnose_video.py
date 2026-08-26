#!/usr/bin/env python3
"""Find out WHERE a video loses its number plates.

    ./.venv/bin/python ml/diagnose_video.py --source clip.mp4

The pipeline drops a plate at one of four places, and the running system cannot
tell you which:

    1 vehicle not detected        nothing to crop
    2 plate not detected in crop  the plate box model found no plate
    3 OCR returned nothing        the crop was too small or too blurred
    4 correct_plate rejected it   text came back but was not a valid plate

"40 cars, 1 plate" is a different bug at each stage, so measure before changing
anything. This walks every processed frame with the OCR GATE DISABLED — it
reads every vehicle crop, not the first few frames of each track — because the
question here is what the footage can yield, not what the live budget spends.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sidecar as S  # noqa: E402

# sidecar.py points sys.stdout at stderr on import, so that no dependency's
# chatter can land on the newline-delimited JSON protocol the supervisor reads.
# This is a report for a human, not a protocol, so take stdout back.
sys.stdout = S._PROTOCOL


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d * 100:.0f}%)" if d else f"{n}/0"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--model", default=None,
                    help="plate weights. The OpenVINO export is frozen at the "
                         "size it was exported with, so testing another --imgsz "
                         "needs the .pt")
    ap.add_argument("--imgsz", type=int, default=None, help="override plate detector imgsz")
    ap.add_argument("--plate-conf", type=float, default=None)
    ap.add_argument("--pad", type=float, default=None)
    ap.add_argument("--show", action="store_true", help="print every OCR read")
    ap.add_argument("--truth", type=Path, default=None,
                    help="file of the plates a human can read in this clip, one "
                         "per line. Turns yield into ACCURACY.")
    args = ap.parse_args()

    if args.imgsz is not None:
        # THE PLATE detector, which is what this tool exists to sweep -- the
        # advice it prints when plate boxes are being missed is "try --imgsz
        # 960". Setting the vehicle size instead made that advice a no-op.
        S.PLATE_IMGSZ = args.imgsz
    if args.plate_conf is not None:
        S.PLATE_CONF = args.plate_conf
    if args.pad is not None:
        S.PLATE_PAD = args.pad

    import cv2
    from ultralytics import YOLO

    vehicle_model = YOLO(S.VEHICLE_WEIGHTS)
    weights = args.model or S.find_plate_weights()
    plate_model = YOLO(weights) if weights else None
    print(f"plate detector : {weights}")
    print(f"vehicle imgsz {S.VEHICLE_IMGSZ}  plate imgsz {S.PLATE_IMGSZ}  "
          f"plate_conf {S.PLATE_CONF}  pad {S.PLATE_PAD}\n")

    reader = S.make_reader()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(src_fps / max(args.fps, 1)))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"source {w}x{h} @ {src_fps:.0f} fps, every {step} frame(s)\n")

    frames = vehicles = with_plate_box = ocr_any = accepted = 0
    vehicle_w: list[int] = []
    read_at_w: list[int] = []
    tried_w: list[int] = []
    plate_w: list[int] = []
    raw_texts: Counter[str] = Counter()
    plates: Counter[str] = Counter()
    tracks_seen: set[str] = set()
    n = 0

    while frames < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if (n - 1) % step:
            continue
        frames += 1

        res = vehicle_model.track(frame, persist=True, tracker="ml/botsort.yaml",
                                  classes=list(S.VEHICLE_CLASSES), imgsz=S.VEHICLE_IMGSZ,
                                  conf=S.VEHICLE_CONF, verbose=False)[0]
        boxes = res.boxes
        if boxes is None or not len(boxes):
            continue

        for box in boxes:
            if box.id is not None:
                tracks_seen.add(str(int(box.id[0])))
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            crop = frame[max(y1, 0):y2, max(x1, 0):x2]
            if crop.size == 0:
                continue
            vehicles += 1
            vehicle_w.append(x2 - x1)
            tried_w.append(x2 - x1)

            if plate_model is None:
                continue
            pres = plate_model(crop, imgsz=S.PLATE_IMGSZ, conf=S.PLATE_CONF, verbose=False)[0]
            if not len(pres.boxes):
                continue
            with_plate_box += 1
            pbox = max(pres.boxes, key=lambda b: float(b.conf[0]))
            px1, py1, px2, py2 = (int(v) for v in pbox.xyxy[0])
            plate_w.append(px2 - px1)
            region = S._pad(crop, px1, py1, px2, py2)
            if region.size == 0:
                continue

            got_text = False
            for height, equalise in S._OCR_VARIANTS:
                lines = S._ocr_lines(reader, S._prepare_crop(region, height, equalise))
                if not lines:
                    continue
                got_text = True
                joined = "".join(t for t, _ in lines)
                raw_texts[joined] += 1
                plate, _ = S.correct_plate(joined)
                if plate is None:
                    for t, _sc in lines:
                        plate, _ = S.correct_plate(t)
                        if plate:
                            break
                if plate:
                    accepted += 1
                    plates[plate] += 1
                    read_at_w.append(x2 - x1)
                    if args.show:
                        print(f"  READ  {plate:12} from {joined!r} "
                              f"(plate box {px2 - px1}x{py2 - py1}px)")
                    break
                if args.show:
                    print(f"  reject {joined!r} (plate box {px2 - px1}x{py2 - py1}px)")
            if got_text:
                ocr_any += 1

    cap.release()

    def spread(name: str, xs: list[int]) -> None:
        if not xs:
            print(f"  {name}: none")
            return
        xs = sorted(xs)
        print(f"  {name}: min {xs[0]}  median {xs[len(xs) // 2]}  max {xs[-1]} px")

    print(f"\nframes processed        {frames}")
    print(f"distinct track ids      {len(tracks_seen)}")
    print(f"vehicle crops           {vehicles}")
    spread("vehicle box width", vehicle_w)
    print(f"\nplate box found         {pct(with_plate_box, vehicles)}")
    spread("plate box width", plate_w)
    print(f"OCR returned any text   {pct(ocr_any, with_plate_box)}")
    print(f"accepted by correct_plate {pct(accepted, ocr_any)}")
    print(f"\nEND TO END              {pct(accepted, vehicles)} of vehicle crops")

    # THE NUMBER THAT SETS THE OCR BUDGET. Attempting a read on a vehicle too
    # small to carry a legible plate spends the whole cost of plate detection
    # and two OCR passes to learn nothing, and those attempts are exactly the
    # ones a per-track ceiling then denies to the frames that would have worked.
    if read_at_w:
        rw = sorted(read_at_w)
        print(f"\nvehicle width WHEN A PLATE WAS READ:")
        print(f"  min {rw[0]}  10th pct {rw[len(rw) // 10]}  median {rw[len(rw) // 2]} px")
        below = sum(1 for w in tried_w if w < rw[0])
        print(f"  {pct(below, len(tried_w))} of crops are narrower than the "
              f"narrowest that ever read")

    if plates:
        print("\nplates read:")
        for p, c in plates.most_common(15):
            print(f"  {c:3}x  {p}")
    if raw_texts and not plates:
        print("\nraw OCR text that was rejected (top 15):")
        for t, c in raw_texts.most_common(15):
            print(f"  {c:3}x  {t!r}")

    if args.truth and args.truth.exists():
        want = [l.strip().upper() for l in args.truth.read_text().splitlines()
                if l.strip() and not l.startswith("#")]
        got = set(plates)

        def near(a: str, b: str) -> int:
            """Substitutions between two equal-length strings, else 99."""
            return sum(x != y for x, y in zip(a, b)) if len(a) == len(b) else 99

        exact = [w for w in want if w in got]
        one_off = [w for w in want
                   if w not in got and any(near(w, g) == 1 for g in got)]
        missed = [w for w in want if w not in got and w not in one_off]

        print(f"\nACCURACY against {args.truth.name} ({len(want)} legible plates)")
        print(f"  exactly right     {pct(len(exact), len(want))}")
        print(f"  one character out {pct(len(one_off), len(want))}")
        print(f"  not read at all   {pct(len(missed), len(want))}")
        for w in one_off:
            g = min((g for g in got if near(w, g) == 1), key=lambda g: near(w, g))
            diff = "".join(b if a != b else "." for a, b in zip(w, g))
            print(f"    {w} -> {g}   ({diff})")
        for w in missed:
            print(f"    {w} -> not read")
        # A read that matches nothing legible is either a plate the transcriber
        # could not make out, or an invention. Both are worth seeing.
        extra = [g for g in got if g not in want and not any(near(w, g) <= 1 for w in want)]
        if extra:
            print(f"  reads matching no legible plate: {', '.join(sorted(extra))}")

    print("\nWhere it went:")
    if vehicles and with_plate_box / vehicles < 0.2:
        print("  THE PLATE DETECTOR. It rarely finds a plate in the vehicle crop.")
        print("  Usually the plate is a handful of pixels wide at this distance,")
        print("  or the detector was trained on close-up plates and has not seen")
        print("  one this small. Try --imgsz 960 and a lower --plate-conf.")
    elif with_plate_box and ocr_any / with_plate_box < 0.5:
        print("  OCR. Plate boxes are found but no text comes back — the crop is")
        print("  too small or too motion-blurred to read.")
    elif ocr_any and accepted / ocr_any < 0.5:
        print("  correct_plate. Text is read but rejected as not a valid plate.")
        print("  Look at the raw text above: partial reads are the usual cause.")
    else:
        print("  Nothing dominates; the losses are spread across stages.")


if __name__ == "__main__":
    main()
