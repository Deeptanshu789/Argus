#!/usr/bin/env python3
"""Export trained weights for CPU inference.

OpenVINO IR, not raw PyTorch: OpenVINO runs on AMD x86 CPUs and typically gives
2-3x over stock torch CPU. That margin is what makes 4 streams x 5 FPS fit.
ONNX is the fallback when OpenVINO misbehaves on a layer.

ponytail: ceiling here is CPU int8, 480px, 4 streams @ 5 FPS. Upgrade path is
the amdxdna NPU at /dev/accel/accel0 (Linux stack still immature) or any CUDA
box at the venue -- in both cases re-export, nothing else changes.
"""
import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="runs/detect/plate/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--format", default="openvino", choices=["openvino", "onnx"])
    ap.add_argument("--int8", action="store_true", default=True)
    ap.add_argument("--fp32", dest="int8", action="store_false")
    ap.add_argument("--data", default="datasets/plates/data.yaml",
                    help="int8 needs a calibration set")
    args = ap.parse_args()

    from ultralytics import YOLO

    kw = {"format": args.format, "imgsz": args.imgsz, "int8": args.int8}
    if args.int8 and args.format == "openvino":
        kw["data"] = args.data
    print("exported:", YOLO(args.weights).export(**kw))


if __name__ == "__main__":
    main()
