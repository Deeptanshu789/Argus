#!/usr/bin/env python3
"""Export trained weights for CPU inference.

OpenVINO IR, not raw PyTorch: OpenVINO runs on AMD x86 CPUs and typically gives
2-3x over stock torch CPU. That margin is what makes 4 streams x 5 FPS fit.
ONNX is the fallback when OpenVINO chokes on a layer.

    python ml/export_onnx.py --weights runs/detect/plate/weights/best.pt

USE --fp32 UNLESS YOU HAVE MEASURED A NEED FOR MORE.

Measured on the build machine (AMD Ryzen AI 7 350, 16 threads, imgsz 480):

    PyTorch CPU      17 ms/frame
    OpenVINO fp32     9 ms/frame

The budget is 4 streams x 5 FPS = 20 inferences/sec, i.e. 50 ms per inference.
fp32 already clears that by more than 5x. int8 buys roughly another 2x on a
number that is not the bottleneck -- so it is an optimisation to reach for if
end-to-end FPS actually misses, not a prerequisite.

That matters because int8 NEEDS CALIBRATION DATA: quantisation measures real
activation ranges on real images, and there is no way to do it from the weights
alone. Training happens on Kaggle and export happens on the laptop, so the
dataset is usually in the wrong place. --fp32 needs nothing but the weights.

If you do want int8 later, export on Kaggle where the dataset already is and
download the IR folder.

ponytail: ceiling is CPU int8, 480px, 4 streams @ 5 FPS. Upgrade path is the
amdxdna NPU at /dev/accel/accel0 (Linux stack still immature) or any CUDA box at
the venue -- in both cases re-export, nothing else changes.
"""
import argparse
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="runs/detect/plate/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--format", default="openvino", choices=["openvino", "onnx"])
    ap.add_argument("--int8", action="store_true", default=True)
    ap.add_argument("--fp32", dest="int8", action="store_false",
                    help="skip quantisation; needs no calibration data")
    ap.add_argument("--data", default="datasets/plates/data.yaml",
                    help="calibration set for int8")
    args = ap.parse_args()

    if not Path(args.weights).exists():
        sys.exit(
            f"no weights at {args.weights}\n"
            f"Bring them back from Kaggle first:\n"
            f"  mkdir -p runs/detect/plate/weights\n"
            f"  unzip ~/Downloads/argus-plate-weights.zip -d runs/detect/plate/weights"
        )

    needs_data = args.int8 and args.format == "openvino"
    if needs_data and not Path(args.data).exists():
        sys.exit(
            f"int8 needs calibration images and '{args.data}' is not here.\n"
            f"Quantisation measures activation ranges on real images -- it cannot be\n"
            f"done from the weights alone.\n\n"
            f"Either export on Kaggle, where the dataset already is:\n"
            f"    !python ml/export_onnx.py --weights runs/detect/plate/weights/best.pt\n"
            f"    then download the *_openvino_model folder\n\n"
            f"or export unquantised here and lose roughly 2x of the speedup:\n"
            f"    python ml/export_onnx.py --weights {args.weights} --fp32"
        )

    from ultralytics import YOLO

    kw = {"format": args.format, "imgsz": args.imgsz, "int8": args.int8}
    if needs_data:
        kw["data"] = args.data
    out = YOLO(args.weights).export(**kw)

    print(f"\nexported: {out}")
    if not args.int8:
        print("NOTE: fp32, not int8. Measure end-to-end FPS at 4 streams before")
        print("assuming this is fast enough -- int8 is worth about 2x.")


if __name__ == "__main__":
    main()
