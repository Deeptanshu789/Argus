#!/usr/bin/env python3
"""Visual demo: run the plate detector on images or video and draw what it finds.

    python ml/demo_detect.py --source photo.jpg
    python ml/demo_detect.py --source some_folder/
    python ml/demo_detect.py --source clip.mp4
    python ml/demo_detect.py --source photo.jpg --ocr     # also read the plates

Writes annotated copies to ml/demo_out/ and prints a summary. This is the
eyeball check on a trained detector -- a number like mAP50 tells you whether it
works on the val split; this tells you whether it works on YOUR footage.

CPU only, like everything at runtime here. Defaults to the OpenVINO export when
one exists because it is roughly 2x the PyTorch weights, and falls back to
best.pt otherwise.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sidecar import (
    PLATE_CONF, PLATE_IMGSZ, PLATE_MIN_VEHICLE_PX, VEHICLE_CLASSES,
    VEHICLE_CONF, VEHICLE_IMGSZ, VEHICLE_WEIGHTS,
    _pad, _read_plate, correct_plate, find_plate_weights, make_reader,
)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


_warned: set[str] = set()


def _warn_once(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        print(f"warning: {msg}")



def pipeline_video(src: Path, dst: Path, args) -> None:
    """Annotate a clip the way ml/sidecar.py actually sees it.

    The default mode of this script runs the plate detector straight at the
    frame, which is the right check on a still photograph and is NOT what the
    system does. `run()` detects vehicles, tracks them, and only then looks for
    a plate INSIDE a vehicle box -- so a demo that skips the vehicle stage
    cannot show why a plate was missed, and cannot show a plate being attached
    to a tracked vehicle at all.

    Reading goes through sidecar._read_plate(), the same function the sidecar
    calls, so the text drawn here is the text that would reach the database.
    The plate box is found once and handed on as an already-cropped region, so
    the detector is not run twice on the same crop just to draw a rectangle.
    """
    import cv2
    from ultralytics import YOLO

    vehicle = YOLO(VEHICLE_WEIGHTS)
    plate_path = pick_model(args.model)
    plate = YOLO(plate_path)
    reader = make_reader() if args.ocr else None
    print(f"vehicles: {VEHICLE_WEIGHTS}   plates: {plate_path}")

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        sys.exit(f"cannot decode video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, round(fps / 5))
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / step, (w, h))
    if not writer.isOpened():
        cap.release()
        sys.exit(f"cannot write {dst} (no mp4v encoder in this OpenCV build)")

    # Best read per track. A track is read in only a few of its frames, but the
    # label belongs on the box for as long as the vehicle is in shot -- that is
    # what the dashboard shows, and a label that blinks off tells the viewer
    # the plate was lost when it was simply not re-read.
    best: dict[int, tuple[str, float]] = {}
    frames = read_attempts = 0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step:
            i += 1
            continue
        i += 1
        res = vehicle.track(frame, persist=True, tracker="ml/botsort.yaml",
                            classes=list(VEHICLE_CLASSES), imgsz=VEHICLE_IMGSZ,
                            conf=VEHICLE_CONF, verbose=False)[0]
        for box in res.boxes or []:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            kind = VEHICLE_CLASSES.get(int(box.cls[0]), "car")
            tid = int(box.id[0]) if box.id is not None else -1
            width = x2 - x1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 170, 0), 2)
            tag = f"{kind} #{tid}" if tid >= 0 else kind
            cv2.putText(frame, tag, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 170, 0), 2)

            # The gate the real pipeline applies. Below it no plate has ever
            # been read on this footage, so paying for the attempt buys noise.
            if reader is None or width < PLATE_MIN_VEHICLE_PX:
                continue
            crop = frame[max(y1, 0):y2, max(x1, 0):x2]
            if crop.size == 0:
                continue
            pres = plate(crop, imgsz=PLATE_IMGSZ, conf=PLATE_CONF, verbose=False)[0]
            if not len(pres.boxes):
                continue
            pb = max(pres.boxes, key=lambda b: float(b.conf[0]))
            px1, py1, px2, py2 = (int(v) for v in pb.xyxy[0])
            cv2.rectangle(frame, (x1 + px1, y1 + py1), (x1 + px2, y1 + py2),
                          (0, 220, 0), 2)

            if tid not in best:
                read_attempts += 1
                region = _pad(crop, px1, py1, px2, py2)
                text, conf = _read_plate(reader, None, region)
                if text:
                    best[tid] = (text, conf or 0.0)

        # Labels last, so a plate box drawn afterwards cannot cover the text.
        for box in res.boxes or []:
            tid = int(box.id[0]) if box.id is not None else -1
            if tid not in best:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            text, conf = best[tid]
            label = f"{text} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, min(y2 + 4, h - th - 8)),
                          (x1 + tw + 6, min(y2 + th + 10, h)), (0, 220, 0), -1)
            cv2.putText(frame, label, (x1 + 3, min(y2 + th + 6, h - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        writer.write(frame)
        frames += 1
        if args.max_frames and frames >= args.max_frames:
            break

    cap.release(); writer.release()
    print(f"\nwrote {dst}  ({frames} frames at {fps / step:.0f} fps)")
    print(f"{len(best)} plate(s) read from {read_attempts} attempt(s) "
          f"on vehicles at least {PLATE_MIN_VEHICLE_PX}px wide")
    for tid, (text, conf) in sorted(best.items()):
        print(f"  #{tid:<4} {text}  {conf:.2f}")


def pick_model(explicit: str | None) -> str:
    """The detector the SIDECAR would load, not a second opinion about it.

    This kept its own copy of the candidate list, so when ml/sidecar.py's
    default moved to the trained plate-k12 weights this demo silently went on
    rendering with the older YOLOv8n -- a video captioned "the new detector"
    that showed the old one. One search, in one place, and ARGUS_PLATE_MODEL
    overrides both together.
    """
    if explicit:
        return explicit
    found = find_plate_weights()
    if found:
        return found
    sys.exit(
        "no trained weights found.\n"
        "Expected runs/detect/plate-k12/weights/best.pt -- train on Kaggle "
        "first (see WORKFLOW.md Stage 1), or pass --model explicitly."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="image, folder, or video")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", type=Path, default=Path("ml/demo_out"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=None,
                    help="default: the model's own export size, else 640")
    ap.add_argument("--ocr", action="store_true", help="also read the plate text")
    ap.add_argument("--limit", type=int, default=20, help="max images from a folder")
    ap.add_argument("--vehicles", action="store_true",
                    help="run the REAL pipeline: track vehicles, then read the "
                         "plate inside each one. Video only.")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop after this many processed frames (0 = all)")
    args = ap.parse_args()

    src = Path(args.source)
    if not src.exists():
        sys.exit(f"no such source: {src}")

    import cv2
    from ultralytics import YOLO

    model_path = pick_model(args.model)
    model = YOLO(model_path)

    if args.imgsz is None:
        # An OpenVINO IR is exported at ONE fixed input size and throws a shape
        # error on anything else. Read the size it was built with rather than
        # making the caller remember it.
        args.imgsz = 640
        meta = Path(model_path) / "metadata.yaml"
        if meta.exists():
            import yaml
            # imgsz is a YAML LIST (`imgsz:\n- 480\n- 480`), so the value is on
            # the following lines, not the same one. Parse it, do not regex it.
            got = yaml.safe_load(meta.read_text()).get("imgsz")
            if isinstance(got, (list, tuple)) and got:
                args.imgsz = int(got[0])
            elif isinstance(got, int):
                args.imgsz = got
    print(f"model : {model_path}  (imgsz {args.imgsz})")

    reader = make_reader() if args.ocr else None
    if args.ocr and reader is None:
        print("warning: OCR unavailable, boxes only")

    if src.is_dir():
        items = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT)[: args.limit]
    else:
        items = [src]
    if not items:
        sys.exit(f"no images under {src}")

    args.out.mkdir(parents=True, exist_ok=True)
    is_video = items[0].suffix.lower() in VID_EXT

    if args.vehicles:
        if not is_video:
            sys.exit("--vehicles needs a video; a still has nothing to track")
        pipeline_video(items[0], args.out / f"{items[0].stem}_pipeline.mp4", args)
        print(f"\nopen the results:  xdg-open {args.out}")
        return

    total_plates, total_ms, frames, texts = 0, 0.0, 0, []

    def annotate(frame, boxes) -> int:
        n = 0
        for b in boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
            conf = float(b.conf[0])
            label = f"plate {conf:.2f}"
            if reader is not None:
                text = read_plate(frame[max(y1, 0):y2, max(x1, 0):x2])
                if text:
                    label = f"{text} ({conf:.2f})"
                    texts.append(text)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            # Filled strip behind the text: green-on-green is unreadable against
            # a pale car, and this demo exists to be looked at.
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, max(y1 - th - 8, 0)), (x1 + tw + 6, y1), (0, 220, 0), -1)
            cv2.putText(frame, label, (x1 + 3, max(y1 - 5, th)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            n += 1
        return n

    def read_plate(crop) -> str | None:
        if crop.size == 0 or reader is None:
            return None
        try:
            res = reader.predict(crop)
        except Exception as exc:
            # Warn ONCE. Swallowing this silently makes a broken PaddleOCR
            # install look identical to "the plates were unreadable", and
            # paddle does throw here on some CPU/oneDNN builds.
            _warn_once(f"OCR failed ({type(exc).__name__}: {exc}), boxes only")
            return None
        raw = ""
        for page in res or []:
            got = page.get("rec_texts") if isinstance(page, dict) else None
            if got:
                raw = max(got, key=len)
                break
        if not raw:
            if res and not isinstance(res[0], dict):
                _warn_once(f"OCR returned {type(res[0]).__name__}, not the dict "
                           "this reader expects -- paddleocr API moved")
            return None
        fixed, _ = correct_plate(raw)
        # Show the raw read when correction rejects it -- "OCR saw something we
        # could not validate" is different from "OCR saw nothing", and the demo
        # should not hide which one happened.
        return fixed or f"?{raw}"

    if is_video:
        cap = cv2.VideoCapture(str(items[0]))
        # VideoCapture does not raise on a file it cannot decode -- it just
        # returns a closed handle whose every read fails, and VideoWriter then
        # silently refuses a 0x0 frame. Without this, the run reports success
        # and writes nothing.
        if not cap.isOpened():
            sys.exit(f"cannot decode video: {items[0]}\n"
                     "Not a video, or the codec is missing from this OpenCV build.")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 5 FPS, matching the real pipeline budget -- see CLAUDE.md. Only the
        # processed frames are written, at 5 FPS, so the boxes stay on screen.
        # Writing every frame at source FPS shows each box for one frame (40ms
        # at 25fps) -- a flicker, on the tool whose whole job is being looked at.
        step = max(1, round(src_fps / 5))
        out_fps = src_fps / step
        dst = args.out / f"{items[0].stem}_annotated.mp4"
        writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
        if not writer.isOpened():
            cap.release()
            sys.exit(f"cannot write {dst} (no mp4v encoder in this OpenCV build)")
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                t0 = time.time()
                r = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
                total_ms += (time.time() - t0) * 1000
                total_plates += annotate(frame, r.boxes)
                frames += 1
                writer.write(frame)
            i += 1
        cap.release(); writer.release()
        print(f"\nwrote {dst}  ({frames} frames at {out_fps:.0f} fps)")
    else:
        for p in items:
            frame = cv2.imread(str(p))
            if frame is None:
                print(f"  skip (unreadable): {p.name}")
                continue
            t0 = time.time()
            r = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            ms = (time.time() - t0) * 1000
            total_ms += ms
            n = annotate(frame, r.boxes)
            total_plates += n
            frames += 1
            # Flatten the path into the name: rglob can hand us train/a.jpg
            # and val/a.jpg, and a flat `a_annotated.jpg` loses one of them.
            stem = p.stem if p == src else "_".join(p.relative_to(src).with_suffix("").parts)
            dst = args.out / f"{stem}_annotated.jpg"
            cv2.imwrite(str(dst), frame)
            print(f"  {p.name:44} {n} plate(s)  {ms:5.0f} ms")

    print(f"\n{frames} frame(s), {total_plates} plate(s), "
          f"{total_ms / max(frames, 1):.0f} ms/frame average")
    if frames and total_plates == 0:
        print("NOTHING DETECTED. Try --conf 0.1, or check the images actually "
              "contain vehicles at a readable distance.")
    if texts:
        print(f"plates read: {', '.join(texts[:10])}"
              + (" ..." if len(texts) > 10 else ""))
    print(f"\nopen the results:  xdg-open {args.out}")


if __name__ == "__main__":
    main()
