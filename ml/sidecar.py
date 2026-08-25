#!/usr/bin/env python3
"""Inference sidecar — the ONLY Python that runs at application runtime.

One process per camera. Reads a video source, runs detection, tracking, OCR and
Re-ID, and emits newline-delimited JSON on stdout. Everything downstream —
cross-camera association, analytics, API, WebSocket, UI — is TypeScript.

    python ml/sidecar.py --camera CAM1 --source demo/cam1.mp4 --fps 5

Supervised by worker/ingest.ts, which parses each line against the SidecarEvent
schema in src/contract.ts. THAT SCHEMA IS THE CONTRACT: if you change an event
shape here, change it there in the same commit or the supervisor will reject
your output as a contract violation.

Event types, one JSON object per line on stdout:

  {"event":"ready","camera_id":...,"fps":...}
  {"event":"detection","camera_id":...,"track_id":...,"ts":...,"bbox":[x1,y1,x2,y2],
   "conf":...,"vehicle_type":...}
  {"event":"track_closed","camera_id":...,"track_id":...,"entry_time":...,
   "exit_time":...,"vehicle_type":...,"color":...,"plate_text":...,"plate_conf":...,
   "embedding":[...floats, same length on every camera...],"color_hist":[...]}
  {"event":"error","camera_id":...,"detail":...}

Diagnostics go to stderr. Anything non-JSON on stdout is a bug.

NO GPU. The demo machine is an AMD Ryzen AI 7 350 with no CUDA device. Models
load as OpenVINO int8 (see ml/export_onnx.py). Training happens separately on
Kaggle; this file never trains anything.

CPU budget, and the reason this file is structured the way it is:
  - decode and detect at 5 FPS, not 30
  - plate detection + OCR on a track's first few frames and its best crop only
  - Re-ID embedding on track exit only
4 streams x 5 FPS = 20 detections/sec, which a quantized YOLOv8n at 480px
clears on 16 Zen 5 threads. Per-frame OCR or Re-ID does not.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone

# Re-exported so the TS side and this file agree on the label set.
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# The three OCR confusions that account for most Indian-plate errors. This
# correction pass is worth more than model fine-tuning -- do not drop it.
DIGIT_FOR = {"O": "0", "Q": "0", "I": "1", "L": "1", "B": "8", "S": "5", "Z": "2"}
# NOT derived by inverting DIGIT_FOR: that map is many-to-one (O and Q both go
# to 0), so inverting it silently picks whichever key came last. Spelled out.
ALPHA_FOR = {"0": "O", "1": "I", "8": "B", "5": "S", "2": "Z"}
STATE_CODES = {
    "AP","AR","AS","BR","CG","CH","DD","DL","DN","GA","GJ","HP","HR","JH","JK",
    "KA","KL","LA","LD","MH","ML","MN","MP","MZ","NL","OD","PB","PY","RJ","SK",
    "TN","TR","TS","UK","UP","WB",
}


def emit(**event) -> None:
    """One JSON object per line, flushed. The supervisor reads line by line."""
    json.dump(event, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")
    sys.stdout.flush()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def correct_plate(raw: str) -> tuple[str | None, float]:
    """Indian plates are XX 00 XX 0000: two letters, two digits, one-or-two
    letters, four digits. Each position's character class is known, so an OCR
    confusion can be corrected by position rather than guessed at.

    Returns (plate, penalty). penalty is how many characters had to be fixed —
    the caller subtracts it from OCR confidence, so a heavily-corrected read
    does not masquerade as a confident one.
    """
    s = "".join(c for c in raw.upper() if c.isalnum())
    if not 9 <= len(s) <= 10:
        return None, 0.0
    # positions: 0-1 alpha, 2-3 digit, then 1-2 alpha, last 4 digit
    mid = len(s) - 4
    classes = ["A", "A", "D", "D"] + ["A"] * (mid - 4) + ["D"] * 4
    out, penalty = [], 0
    for ch, cls in zip(s, classes):
        want_digit = cls == "D"
        if want_digit and ch.isalpha():
            fixed = DIGIT_FOR.get(ch)
            if fixed is None:
                return None, 0.0
            out.append(fixed); penalty += 1
        elif not want_digit and ch.isdigit():
            fixed = ALPHA_FOR.get(ch)
            if fixed is None:
                return None, 0.0
            out.append(fixed); penalty += 1
        else:
            out.append(ch)
    plate = "".join(out)
    if plate[:2] not in STATE_CODES:
        return None, 0.0
    return plate, penalty * 0.05


# Camera ordering for the synthetic demo. The vehicle travels CAM1 -> CAM3 ->
# CAM2, and each sidecar staggers its output so the worker sees the legs in the
# order real event timing would deliver them.
DEMO_ROUTE = ["CAM1", "CAM3", "CAM2", "CAM4"]
DEMO_PLATE = "KA05MR7821"


def _demo_embedding(seed: int) -> list[float]:
    """Deterministic 512-dim vector. Same vehicle -> near-identical vectors, so
    layer 2 of the association engine fires exactly as it would on real tracker
    output. 512 here is arbitrary: what matters is that every demo sidecar
    agrees, which is the same rule the real ones follow."""
    import math
    return [round(math.sin((i + 1) * 0.017 + seed), 6) for i in range(512)]


def demo(camera: str, fps: int) -> None:
    """Emit a scripted event sequence -- no video, no models, no CV dependencies.

    Exists so the whole Python -> JSON -> worker -> association path can be run
    and watched today, before the real pipeline or a GPU-trained detector exist.
    Also gives the dashboard developer live traffic without sourcing footage.

        ARGUS_CAMERAS=CAM1=demo,CAM3=demo npm run worker

    Timestamps are backdated so a journey that takes minutes plays out
    instantly: CAM1's vehicle exits 200 s "ago", CAM3's arrives 30 s "ago", which
    is inside the 0.5x-2.0x window around the 168 s link.
    """
    import time

    try:
        leg = DEMO_ROUTE.index(camera)
    except ValueError:
        leg = 0
    # Stagger so the earlier leg is already in the worker's candidate window
    # when the later one closes -- the same ordering real cameras produce.
    time.sleep(leg * 2)

    base = time.time() - 200 + leg * 170
    entry = datetime.fromtimestamp(base, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    exit_ = datetime.fromtimestamp(base + 8, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    for i in range(fps):
        emit(event="detection", camera_id=camera, track_id="T0001", ts=entry,
             bbox=[400 + i * 12, 220, 560 + i * 12, 340], conf=0.93, vehicle_type="car")
        time.sleep(1.0 / max(fps, 1))

    emit(event="track_closed", camera_id=camera, track_id="T0001",
         entry_time=entry, exit_time=exit_, vehicle_type="car", color="white",
         # Third leg has an unreadable plate on purpose: it forces the match to
         # come from layers 2+3, which is exactly what Act 2 of the demo shows.
         plate_text=None if camera == "CAM2" else DEMO_PLATE,
         plate_conf=None if camera == "CAM2" else 0.97,
         embedding=_demo_embedding(3), color_hist=[10.0, 20.0, 30.0, 40.0])

    print(f"[{camera}] demo leg {leg} emitted", file=sys.stderr)


def run(camera: str, source: str, fps: int) -> None:
    """Real pipeline. Dev A fills this in; the event shapes above are fixed.

    Sketch, in the order things happen:
        cap = cv2.VideoCapture(source); skip frames to hit `fps`
        # model.track(persist=True, tracker="ml/botsort.yaml") gives boxes,
        # stable ids AND ReID features in one pass -- do not run a second model.
        results = vehicle_model.track(frame, persist=True,
                                      tracker="ml/botsort.yaml", classes=[2,3,5,7])
        for t in tracks.new_or_low_conf_plate:    # NOT every frame
            crop = best_crop(t); clahe(crop)
            plate_raw = ocr(plate_model(crop))
            t.plate, penalty = correct_plate(plate_raw)
        for t in tracks.just_exited:              # exit only
            emit(event="track_closed", embedding=t.reid_feature.tolist(), ...)
    """
    emit(event="error", camera_id=camera,
         detail="ml/sidecar.py run() not implemented yet -- see the docstring")
    raise SystemExit(2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--source", required=True, help="video file or RTSP url")
    ap.add_argument("--fps", type=int, default=5, help="process rate, not source rate")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        return _selfcheck()

    emit(event="ready", camera_id=args.camera, fps=args.fps)
    try:
        if args.source == "demo":
            return demo(args.camera, args.fps)
        run(args.camera, args.source, args.fps)
    except SystemExit:
        raise
    except Exception as exc:  # never die silently -- the supervisor restarts us
        emit(event="error", camera_id=args.camera, detail=f"{type(exc).__name__}: {exc}")
        raise


def _selfcheck() -> None:
    """Plate correction: the character-class rules hold and junk is rejected."""
    assert correct_plate("KA01MB1234") == ("KA01MB1234", 0.0)
    # O in a digit slot -> 0; I in a digit slot -> 1. Two fixes, penalty 0.10.
    p, pen = correct_plate("KAO1MB123I")
    assert p == "KA01MB1231", p
    assert abs(pen - 0.10) < 1e-9, pen
    # 0 in an alpha slot -> O.
    assert correct_plate("KA01M01234")[0] == "KA01MO1234"
    assert correct_plate("KA 05 MR 7821")[0] == "KA05MR7821", "spaces must be tolerated"
    assert correct_plate("MH04AQ5678")[0] == "MH04AQ5678"
    # Rejections.
    assert correct_plate("XX01MB1234")[0] is None, "unknown state code must be rejected"
    assert correct_plate("KA01MB12")[0] is None, "too short"
    assert correct_plate("KA01MB12345678")[0] is None, "too long"
    assert correct_plate("K#01MB1234")[0] is None, "punctuation strips to wrong length"

    import io
    buf, real = io.StringIO(), sys.stdout
    sys.stdout = buf
    emit(event="ready", camera_id="CAM1", fps=5)
    sys.stdout = real
    line = buf.getvalue()
    assert line.endswith("\n") and json.loads(line)["event"] == "ready"
    print("sidecar selfcheck ok", file=sys.stderr)


if __name__ == "__main__":
    main()
