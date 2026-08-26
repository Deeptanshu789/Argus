#!/usr/bin/env python3
"""Fine-tune YOLOv8n as a single-class Indian license-plate detector.

DEFAULTS TARGET KAGGLE GPU (T4 x2 / P100). ~20-40 s/epoch, so 50 epochs is
roughly half an hour. Run it from ml/kaggle_train.ipynb.

    python ml/train_plate.py --epochs 50

The build machine has no CUDA (AMD Radeon 860M, ROCm not installed), so the
local fallback is a separate, much slower path behind one flag:

    python ml/train_plate.py --cpu --epochs 1     # ~6-15 min for that one epoch

--cpu flips a preset, not a single setting: nano stays, but imgsz drops to 480
((480/640)^2 = 0.56x the FLOPs), the backbone freezes so the backward pass only
runs through neck and head, the dataset caches in RAM, and workers drop to 6 so
the laptop stays usable. Any flag you pass explicitly still wins over the preset.

Use --cpu only if Kaggle is unavailable. It is 15-25x slower for the same job.
"""
import argparse
import sys
import time
from pathlib import Path

# (imgsz, batch, workers, freeze, amp, cache)
PRESETS = {
    "gpu": dict(imgsz=640, batch=32, workers=4, freeze=0, amp=True, cache="ram"),
    "cpu": dict(imgsz=480, batch=16, workers=6, freeze=10, amp=False, cache="ram"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpu", action="store_true", help="local CPU fallback preset")
    ap.add_argument("--data", default="datasets/plates/data.yaml")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--project", default="runs/detect")
    ap.add_argument("--name", default="plate")
    ap.add_argument("--resume", action="store_true")
    # Preset-backed. Default None so we can tell "user asked" from "preset value".
    for flag, kind in [("--imgsz", int), ("--batch", int), ("--workers", int),
                       ("--freeze", int)]:
        ap.add_argument(flag, type=kind, default=None)
    ap.add_argument("--device", default=None, help="'cpu', '0', '0,1'")
    args = ap.parse_args()

    preset = PRESETS["cpu" if args.cpu else "gpu"]
    device = args.device or ("cpu" if args.cpu else 0)
    cfg = {k: (getattr(args, k, None) if getattr(args, k, None) is not None else v)
           for k, v in preset.items()}

    print(f"preset={'cpu' if args.cpu else 'gpu'} device={device} "
          f"epochs={args.epochs} {cfg}", flush=True)

    from ultralytics import YOLO

    # ABSOLUTE, deliberately. Ultralytics resolves a relative `project` against
    # its own runs directory, not the working directory, so `runs/detect` became
    # `runs/detect/runs/detect/<name>` and the weights landed somewhere nothing
    # looks for them -- including find_plate_weights() in ml/sidecar.py.
    project = Path(args.project).resolve()
    data = str(Path(args.data).expanduser().resolve())

    model = YOLO(args.model)
    t0 = time.time()
    model.train(
        data=data,
        epochs=args.epochs,
        patience=args.patience,
        device=device,
        project=str(project),
        name=args.name,
        resume=args.resume,
        exist_ok=True,
        **cfg,
    )
    mins = (time.time() - t0) / 60
    per = mins / max(args.epochs, 1)

    # Report where the weights REALLY are. The trainer knows; guessing from the
    # arguments is what hid the last run's output.
    save_dir = Path(getattr(model.trainer, "save_dir", project / args.name))
    best = save_dir / "weights" / "best.pt"
    print(f"\n{args.epochs} epoch(s) in {mins:.1f} min -> {per:.1f} min/epoch")
    print(f"weights: {best}")
    if not best.exists():
        print(f"WARNING: {best} does not exist -- training wrote nothing there.")
    elif save_dir.resolve() != (project / args.name).resolve():
        print(f"WARNING: expected {project / args.name}, got {save_dir}. "
              f"Move the weights before the sidecar can find them.")

    if args.cpu and args.epochs == 1:
        # Calibration guidance only matters on the slow path.
        if per < 15:
            print(f"CALIBRATION: budget epochs = floor(hours * 60 / {per:.1f}), cap 60.")
        elif per < 25:
            print("CALIBRATION: slow. Retry with --imgsz 416 and a 2000-image subset.")
        else:
            print("CALIBRATION: too slow for local. Use Kaggle — that is the default path.")


if __name__ == "__main__":
    sys.exit(main())
