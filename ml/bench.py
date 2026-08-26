#!/usr/bin/env python3
"""Time each stage of the inference loop separately.

    ./.venv/bin/python ml/bench.py --source uploads/clip.mp4 --frames 60

The CPU budget in CLAUDE.md is stated as "4 streams x 5 FPS", but a budget you
cannot attribute is a budget you cannot cut: when the loop is too slow, the
question is always WHICH stage. This reports decode, detect+track, plate
detection and OCR separately, so an optimisation can be aimed rather than
guessed at, and so the same numbers can be compared before and after.

Pacing is deliberately skipped. `sidecar.run()` sleeps to hold the target frame
rate when reading a file, which is right for a demo and useless for a
measurement -- it would report the sleep, not the work.
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sidecar as S  # noqa: E402

sys.stdout = S._PROTOCOL  # importing the sidecar points stdout at stderr


def summarise(name: str, samples: list[float], processed: int) -> None:
    if not samples:
        print(f"  {name:<22} never ran")
        return
    total = sum(samples)
    print(f"  {name:<22} {statistics.mean(samples) * 1e3:7.1f} ms x {len(samples):4d}"
          f"  = {total:6.2f} s   {total / max(processed, 1) * 1e3:7.1f} ms/frame")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--frames", type=int, default=60, help="processed frames to time")
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--imgsz", type=int, default=None, help="override VEHICLE_IMGSZ")
    ap.add_argument("--no-ocr", action="store_true", help="cost of tracking alone")
    ap.add_argument("--grab", action="store_true",
                    help="decode skipped frames with grab() instead of read()")
    args = ap.parse_args()

    if args.imgsz:
        S.VEHICLE_IMGSZ = args.imgsz

    import cv2
    from ultralytics import YOLO

    S.apply_thread_limit()
    load = time.perf_counter()
    vehicle_model = YOLO(S.VEHICLE_WEIGHTS)
    weights = S.find_plate_weights()
    plate_model = YOLO(weights) if weights else None
    reader = None if args.no_ocr else S.make_reader()
    load = time.perf_counter() - load

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(src_fps / max(args.fps, 1)))

    decode: list[float] = []
    skip: list[float] = []
    track: list[float] = []
    ocr: list[float] = []
    colour: list[float] = []

    tracks: dict[str, S._Track] = {}
    processed = 0
    raw_no = 0
    wall = time.perf_counter()

    while processed < args.frames:
        want = (raw_no + 1) % step == 0
        t0 = time.perf_counter()
        if args.grab and not want:
            # grab() advances the decoder without converting the frame to a
            # numpy array. The pixels of a frame we are about to discard are
            # never needed.
            got = cap.grab()
            frame = None
        else:
            got, frame = cap.read()
        dt = time.perf_counter() - t0
        (decode if want else skip).append(dt)
        if not got:
            break
        raw_no += 1
        if not want:
            continue
        processed += 1

        t0 = time.perf_counter()
        result = vehicle_model.track(
            frame, persist=True, tracker="ml/botsort.yaml",
            classes=list(S.VEHICLE_CLASSES), imgsz=S.VEHICLE_IMGSZ, conf=S.VEHICLE_CONF,
            verbose=False)[0]
        track.append(time.perf_counter() - t0)

        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue
        for box in boxes:
            key = str(int(box.id[0]))
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            crop = frame[max(y1, 0):y2, max(x1, 0):x2]
            area = float((x2 - x1) * (y2 - y1))
            t = tracks.get(key)
            if t is None:
                t = tracks[key] = S._Track(key, "car", "", processed)
            t.frames += 1
            if area > t.best_area:
                t.best_area = area
                t0 = time.perf_counter()
                S._colour_name(crop)
                S._colour_hist(crop)
                colour.append(time.perf_counter() - t0)
            if reader is not None and t.wants_ocr(area, x2 - x1):
                t.attempts += 1
                t.last_ocr_area = area
                t0 = time.perf_counter()
                plate, _ = S._read_plate(reader, plate_model, crop)
                ocr.append(time.perf_counter() - t0)
                if plate:
                    t.record(plate, 0.9)
    cap.release()
    wall = time.perf_counter() - wall

    print(f"\nsource {args.source}  {src_fps:.0f} fps, every {step} frame(s), "
          f"imgsz={S.VEHICLE_IMGSZ}")
    print(f"model load {load:.1f} s (once per camera, not per frame)")
    print(f"{processed} processed frames in {wall:.2f} s "
          f"= {processed / max(wall, 1e-9):.2f} processed fps\n")
    summarise("decode (processed)", decode, processed)
    summarise("decode (skipped)", skip, processed)
    summarise("detect + track", track, processed)
    summarise("plate detect + OCR", ocr, processed)
    summarise("colour", colour, processed)

    per_frame = wall / max(processed, 1)
    print(f"\n  {'per processed frame':<22} {per_frame * 1e3:7.1f} ms")
    print(f"  {'streams at 5 fps':<22} {1 / (per_frame * 5):7.1f}  "
          f"(one core-second per second of video per stream)")


if __name__ == "__main__":
    main()
