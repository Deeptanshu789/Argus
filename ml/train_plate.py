#!/usr/bin/env python3
"""Fine-tune YOLOv8n as a single-class Indian license-plate detector on CPU.

This box has no CUDA (AMD Radeon 860M, ROCm not installed), so every default
here is chosen to make CPU training finish overnight instead of never:

  yolov8n     nano, not -s/-l. Single biggest lever.
  imgsz 480   (480/640)^2 = 0.56x the FLOPs of the usual 640.
  freeze 10   backbone frozen -> backward pass only through neck+head.
  cache ram   30 GB of RAM, a 3K-image subset fits; kills dataloader I/O.
  workers 6   leaves ~4 threads so the laptop stays usable during the run.
  patience 15 stop early rather than burn the whole epoch budget.

ALWAYS run --epochs 1 first and read the reported time. Published CPU
throughput for YOLOv8n varies ~3x across reports, so budget from YOUR measured
epoch, not from an estimate. See WORKFLOW.md Stage 1.
"""
import argparse
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="datasets/plates/data.yaml")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=1, help="1 = calibration run")
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--freeze", type=int, default=10)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--device", default="cpu", help="'0' for a CUDA box / Kaggle")
    ap.add_argument("--name", default="plate")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    t0 = time.time()
    YOLO(args.model).train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        freeze=args.freeze,
        patience=args.patience,
        device=args.device,
        cache="ram",
        amp=False,  # no AMP on CPU
        name=args.name,
        resume=args.resume,
    )
    mins = (time.time() - t0) / 60
    print(f"\n{args.epochs} epoch(s) in {mins:.1f} min -> {mins / args.epochs:.1f} min/epoch")
    if args.epochs == 1:
        per = mins
        if per < 15:
            print(f"CALIBRATION: budget epochs = floor(hours * 60 / {per:.1f}), cap 60.")
        elif per < 25:
            print("CALIBRATION: slow. Retry with --imgsz 416 and a 2000-image subset.")
        else:
            print("CALIBRATION: too slow for local. Move this job to Kaggle (--device 0).")


if __name__ == "__main__":
    main()
