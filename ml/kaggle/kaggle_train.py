#!/usr/bin/env python3
"""Train both models on a Kaggle GPU: the plate detector and the plate reader.

    kaggle kernels push -p ml/kaggle
    kaggle kernels status deeptanshu789/argus-plate-detector-and-reader
    kaggle kernels output deeptanshu789/argus-plate-detector-and-reader -p runs/kaggle

WHY THIS FILE EXISTS. Local CPU training costs 7.5 minutes per reader epoch and
roughly an hour per detector epoch at any useful size. A T4 does both in
minutes, which is the difference between one experiment a night and eight an
hour. The reader is where every remaining failure lives, so that ratio matters.

WHAT RUNS HERE. Two trainings, in this order, because the detector's weights are
worth having even if the reader run is cut short by the session limit:

  1. YOLO11s plate detector, fine-tuned from COCO weights. Heavier than the
     shipped YOLOv8n on purpose -- a GPU can afford it, and the export back to
     OpenVINO keeps inference on the laptop's CPU affordable.
  2. CRNN + CTC reader, from ml/train_reader.py in this repository, on the real
     Indian crops uploaded alongside plus the two synthetic Kaggle sets.

The code is not duplicated here. The kernel clones the public Argus repository
and calls the same ml/train_reader.py and ml/prepare_dataset.py that run
locally, so a fix made on the laptop is a fix made on Kaggle. Only the data
paths and the device differ.

DATA. Everything except the real crops is a public Kaggle dataset attached as a
kernel input, so nothing large is uploaded. The real crops cannot be: they are
built locally by ml/make_reader_crops.py out of hand-drawn boxes and PaddleOCR
reads, and they exist nowhere else.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

IN = Path("/kaggle/input")
OUT = Path("/kaggle/working")
# SCRATCH IS NOT OUTPUT. Everything under /kaggle/working is published as the
# kernel's output and has to be downloaded again to read a log; the clone and
# the merged detection set are both scratch, and putting them there once cost
# a 1.4 GB download to find one error message.
SCRATCH = Path("/tmp/argus")
REPO = SCRATCH / "repo"

# Detector size. YOLO11s is ~9M parameters against YOLOv8n's ~3M. The shipped
# laptop pipeline runs an OpenVINO export at 480px, and s still clears the
# 20 inferences/sec budget there; m does not, which is why this stops at s.
DET_MODEL = "yolo11s.pt"
DET_EPOCHS = 40           # over 27,000 images, not 3,000
DET_IMGSZ = 640            # train larger than we infer; the export sets 480

# 60, not 120. Each epoch now sees 178,266 samples against the local run's
# 78,650, so 60 epochs is roughly twice the gradient steps that run took --
# and it has to share a 12-hour session with the detector.
READER_EPOCHS = 60
READER_HIDDEN = 256        # 192 locally; a GPU can afford the wider LSTM
READER_BATCH = 256


def run(cmd: list[str], **kw) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def input_dirs() -> list[Path]:
    """Every plausible dataset root under /kaggle/input.

    NOT just its immediate children. Kaggle does not always mount one directory
    per dataset at the top level -- a run of this kernel saw every input nested
    under a single `datasets/` directory, and a search one level deep found
    nothing at all."""
    out = []
    for depth in ("*", "*/*", "*/*/*"):
        out += [d for d in sorted(IN.glob(depth)) if d.is_dir()]
    return out


def find_input(*needles: str) -> Path | None:
    """An attached dataset directory, matched loosely by name.

    Kaggle mounts each input under its own slug, and the slug is not always
    what the dataset was called. Matching on a fragment keeps this working when
    an input is swapped for a similar one."""
    for d in input_dirs():
        name = d.name.lower()
        if any(n in name for n in needles):
            return d
    return None


def clone_repo() -> None:
    """The trainers themselves, from the public repository.

    Cloning rather than pasting the code in keeps ONE definition of the reader
    architecture. A copy here would drift from ml/train_reader.py, and the two
    would silently disagree about tensor shapes the first time either changed."""
    if REPO.exists():
        return
    run(["git", "clone", "--depth", "1",
         "https://github.com/Deeptanshu789/Argus.git", str(REPO)])


def build_detection_set() -> Path:
    """One YOLO dataset out of every attached detection source.

    ml/prepare_dataset.py does the work: it reads YOLO, COCO and Pascal VOC,
    and drops repeated photographs by perceptual hash first. Public plate sets
    are largely re-uploads of each other, and the same image in train and val
    inflates val mAP silently."""
    dst = SCRATCH / "det"
    if (dst / "data.yaml").exists():
        return dst
    wanted = [find_input("kedarsai"), find_input("saisirishan"),
              find_input("tkm22092"), find_input("fareselmenshawii"),
              find_input("andrewmvd"), find_input("dataclusterlabs")]
    # A SOURCE WITH NO ANNOTATIONS ABORTS THE WHOLE MERGE, so drop those here
    # rather than letting one unlabelled set cost the run. Several published
    # "vehicle" sets are photographs only -- useful to a captioning model, not
    # to a detector, and there is nothing to convert.
    srcs = []
    for d in wanted:
        if d is None:
            continue
        if not any(d.rglob(f"*{ext}") for ext in (".txt", ".json", ".xml")):
            print(f"skipping {d.name}: no annotations, nothing to convert")
            continue
        srcs.append(d)
    if not srcs:
        sys.exit("no detection inputs attached")
    print("detection sources:", *[s.name for s in srcs], sep="\n  ")
    # --subset 0 means EVERYTHING. Its default of 3,000 is a guard for training
    # on this project's laptop, and on a GPU it silently threw away 25,000 of
    # the 28,509 merged images -- 94 batches an epoch instead of 890.
    run([sys.executable, REPO / "ml" / "prepare_dataset.py",
         "--src", *srcs, "--dst", dst, "--subset", "0", "--val", "1500"])
    return dst


# The last PyTorch series whose CUDA wheels still carry sm_60 kernels. Kaggle's
# own torch 2.10 does not, and a session that lands on a Tesla P100 cannot run
# a single tensor on the GPU without this.
P100_TORCH = ("torch==2.5.1", "torchvision==0.20.1")
P100_INDEX = "https://download.pytorch.org/whl/cu121"


def cuda_or_die() -> None:
    """Make the GPU usable, whichever one Kaggle handed us.

    Kaggle allocates whatever accelerator is free and ignores the metadata
    request often enough that it cannot be relied on: three runs asked for a T4
    and got a P100. Its preinstalled torch 2.10 builds no kernels for sm_60, so
    training dies on the first .to(device) with a traceback that blames
    ultralytics.

    Rather than fail and wait for a luckier draw, install a torch that does
    support the card and start again. The re-exec happens ONCE -- guarded by an
    environment variable, because a reinstall that does not fix the arch would
    otherwise loop until the session limit."""
    import torch
    if not torch.cuda.is_available():
        sys.exit("no CUDA device; this kernel needs a GPU accelerator")
    major, minor = torch.cuda.get_device_capability(0)
    name, arch = torch.cuda.get_device_name(0), f"sm_{major}{minor}"
    arches = torch.cuda.get_arch_list()
    print(f"GPU {name}  {arch}  torch {torch.__version__}")
    print("torch was built for:", " ".join(arches))
    if arch in arches:
        return

    if os.environ.get("ARGUS_TORCH_SWAPPED"):
        sys.exit(f"{name} is {arch} and even {P100_TORCH[0]} builds no kernels "
                 f"for it. Set the notebook's accelerator to T4 by hand.")
    print(f"\n{arch} unsupported -- installing {' '.join(P100_TORCH)} "
          f"and restarting", flush=True)
    run([sys.executable, "-m", "pip", "install", "-q", "--index-url", P100_INDEX,
         *P100_TORCH])
    os.environ["ARGUS_TORCH_SWAPPED"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


def train_detector(data: Path) -> None:
    from ultralytics import YOLO
    model = YOLO(DET_MODEL)
    model.train(
        data=str(data / "data.yaml"),
        epochs=DET_EPOCHS,
        imgsz=DET_IMGSZ,
        batch=32,
        device=0,
        workers=4,
        project=str(OUT / "runs"),
        name="plate-kaggle",
        exist_ok=True,
        patience=25,
        # Mirrored plates are the one augmentation this dataset must not get.
        # A flipped plate is still a plate to a DETECTOR, so it is harmless
        # here -- but the same export feeds ml/make_reader_crops.py, and a
        # reader trained on reversed text learns to read backwards.
        fliplr=0.0,
    )


def train_reader(crops: Path) -> None:
    """The CRNN, on real Indian crops plus whatever synthetic sets are attached.

    --real-repeat oversamples the real crops against the much larger synthetic
    pool. Synthetic teaches the alphabet; only the real crops teach real fonts,
    dirt, angles and lighting, and a model that sees them once per twenty
    batches learns the renders instead."""
    synth = [d for d in (find_input("synthetic-indian", "abtexp"),
                         find_input("commercial-vehicle", "raspberrypi5")) if d]
    cmd = [sys.executable, REPO / "ml" / "train_reader.py",
           "--epochs", READER_EPOCHS,
           "--batch", READER_BATCH,
           "--hidden", READER_HIDDEN,
           "--workers", 4,
           "--real", crops,
           "--real-repeat", 20,
           # 42% of the real crops are Maharashtra and 0.4% are Odisha. Without
           # this the model answers MH for every plate it cannot read, because
           # that is what a classifier does with a weak signal, and the result
           # is a grammatical registration nothing downstream can reject.
           "--balance-states",
           "--out", OUT / "runs" / "reader-kaggle"]
    if synth:
        cmd += ["--src", *synth]
        print("synthetic sources:", *[s.name for s in synth], sep="\n  ")
    else:
        # Real crops alone. Fewer samples, but every label is a real plate.
        cmd += ["--src", crops, "--val", 800]
        print("no synthetic input attached; training on real crops only")
    run(cmd)


def main() -> None:
    t0 = time.time()
    os.environ.setdefault("YOLO_VERBOSE", "true")
    # What Kaggle actually mounted. A missing input is the commonest way this
    # kernel fails, and the slug is not always the directory name.
    print("inputs:", *[f"  {d.relative_to(IN)}" for d in input_dirs()], sep="\n")
    clone_repo()
    run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])

    cuda_or_die()
    crops = find_input("argus-reader-crops", "reader-crops")
    if crops is None:
        sys.exit("the real crop dataset is not attached")

    data = build_detection_set()
    train_detector(data)
    print(f"\ndetector done at {(time.time() - t0) / 60:.0f} min", flush=True)

    train_reader(crops)
    print(f"\nreader done at {(time.time() - t0) / 60:.0f} min", flush=True)

    # Only the weights and the metric history are worth carrying home. Kaggle
    # caps the output directory, and the training images inside runs/ would
    # spend that budget on pictures we already have.
    summary = {}
    for run_dir in sorted((OUT / "runs").glob("*")):
        csv = run_dir / "results.csv"
        if csv.exists():
            summary[run_dir.name] = csv.read_text().splitlines()[-1]
        for keep in ("weights/best.pt", "best.pt", "results.csv", "args.yaml"):
            src = run_dir / keep
            if src.exists():
                dst = OUT / f"{run_dir.name}-{Path(keep).name}"
                shutil.copy(src, dst)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    shutil.rmtree(REPO, ignore_errors=True)


if __name__ == "__main__":
    main()
