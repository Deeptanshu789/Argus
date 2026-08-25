# Training the plate detector

Two routes: **Kaggle GPU** (recommended, ~45 min) and **this laptop's CPU**
(~5 h). Same dataset, same script, same output. Pick one; there is no reason to
do both except curiosity.

Every step has a **check** immediately after it. Run the check. A training run
that fails silently and finishes anyway is exactly how the first model ended up
seeing 15% of its dataset.

---

## What went wrong the first time

The shipped weights (`runs/detect/plate/weights/best.pt`, mAP50 0.928) were
trained on **1,365 of 8,823 images**. Not on purpose.

The dataset ships as three zips — `train.zip`, `valid.zip`, `test.zip` — and
**each one contains its own `_annotations.coco.json` at the archive root**. The
download cell unzipped train and valid into the same folder, so:

```
unzip train.zip   ->  6176 images + _annotations.coco.json  (6176 labels)
unzip valid.zip   ->  1765 images + _annotations.coco.json  (1765 labels, OVERWRITES)
                      7941 images on disk, 1765 of them labelled
```

`prepare_dataset.py` matches images to annotations, found 1,765 pairs, was asked
for 3,400, printed

```
warning: only 1365 train images available, wanted 3000
```

…and training continued. The warning scrolled past in a Kaggle log.

**Fixed** by unzipping each split into its own directory, and by an assertion in
the notebook that refuses to continue unless 8,823 images carry 8,823
annotations. `prepare_dataset.py` also now warns when two images share a
basename, since annotations reference images by name and one would silently
shadow the other.

Nothing about the training itself was wrong. The model is honest — it simply
never saw most of its data.

---

## Is more data going to help?

Measured against 45 hand-labelled plates in `ml/groundtruth_test50.csv`:

| | |
|---|---|
| Detector found the plate | **44 of 45** |
| Plate read correctly | 35 of 45 (78%) |
| Read wrongly | 2 (4%) |
| Not read | 8 (18%) |

**One** of the ten failures is a detection miss. The other nine are OCR. So more
detector training buys at most one image in forty-five — *on this test set*.

The honest case for retraining anyway: 1,365 images is a small training set, the
test set is only 45 plates, and 0.928 mAP50 on 15% of the data suggests real
headroom. It costs 45 minutes on Kaggle. Do it. Just do not expect it to move
end-to-end accuracy much, and do not skip the OCR work waiting for it.

---

## Route A — Kaggle GPU (recommended)

**Time:** ~10 min setup, ~45 min training.
**Cost:** free. 30 GPU-hours/week.

### A1. Enable the GPU

New notebook at <https://www.kaggle.com/code> → right sidebar → **Session
options**:

- Accelerator: **GPU T4 x2**
- Internet: **On** (needed to download the dataset)

Both require a phone-verified account. Settings → Phone verification, once.

**Check** — first cell:

```python
!nvidia-smi
```

Must print a Tesla T4 and a memory figure. No GPU listed means the accelerator
did not attach; restart the session rather than training on CPU by accident.

### A2. Get the code

```python
%cd /kaggle/working
!rm -rf Argus && git clone -q https://github.com/Deeptanshu789/Argus.git
%cd Argus
!pip install -q ultralytics
```

**Check:**

```python
!python -c "import ultralytics; print(ultralytics.__version__)"
!ls ml/prepare_dataset.py ml/train_plate.py
```

### A3. Download the dataset — one directory per split

**This is the step that broke last time.**

```python
BASE = "https://huggingface.co/datasets/keremberke/license-plate-object-detection/resolve/main/data"
SRC  = "/kaggle/working/plates"

for split in ("train", "valid", "test"):
    !mkdir -p {SRC}/{split}
    !curl -sL -o /tmp/{split}.zip {BASE}/{split}.zip
    !unzip -q -o /tmp/{split}.zip -d {SRC}/{split}
```

**Check — do not skip this:**

```python
import json, glob
files    = len(glob.glob(f"{SRC}/*/*.jpg"))
labelled = sum(len(json.load(open(j))["images"])
               for j in glob.glob(f"{SRC}/*/_annotations.coco.json"))
print(f"image files {files}, labelled in JSON {labelled}")
assert files == 8823,    f"expected 8823 images, found {files}"
assert labelled == 8823, f"expected 8823 labelled, found {labelled} -- a split overwrote another"
print("OK: all three splits intact")
```

If `labelled` is 1765 or 6176, the splits landed in one folder. Fix the paths
and re-run; do not proceed.

### A4. Convert to YOLO format

```python
!python ml/prepare_dataset.py --src {SRC} --subset 100000 --val 800
```

`--subset 100000` means "no cap" — the script prints what it actually found.

**Check** — the output must say **8023 train / 800 val**:

```
COCO annotations: 8823 labelled images, 9155 boxes
8023 train / 800 val -> datasets/plates/data.yaml
```

Any "only N train images available" line means data is still missing. Stop and
fix A3.

```python
!cat datasets/plates/data.yaml
!ls datasets/plates/images/train | wc -l    # 8023
!ls datasets/plates/labels/train | wc -l    # 8023
```

The two counts must match. An image without a label is a silent negative sample.

### A5. Train

```python
!python ml/train_plate.py --epochs 60 --device 0
```

Parameters, and why (`ml/train_plate.py`, `PRESETS["gpu"]`):

| | value | reason |
|---|---|---|
| `imgsz` | 640 | plates are small; 480 loses distant ones. GPU affords it |
| `batch` | 32 | fits T4 16 GB at 640 |
| `freeze` | 0 | full fine-tune. COCO has no plate class, so late layers alone are not enough |
| `amp` | True | half precision, ~2x faster on T4, no accuracy cost here |
| `cache` | ram | 8 k images at 640 fit; removes disk I/O from the loop |
| `patience` | 15 | stop when 15 epochs bring no improvement |
| `epochs` | 60 | ~45 s/epoch on T4 |

**Check** — the header must show the right data and device:

```
train: /kaggle/working/Argus/datasets/plates/images/train ... 8023 images
Model summary: ... 3,011,043 parameters
```

If it says fewer than 8023, A4 did not produce what you think.

### A6. Read the score

```python
import csv, pathlib
row = list(csv.DictReader(open("runs/detect/plate/results.csv")))[-1]
m50 = float(row["metrics/mAP50(B)"])
print(f"epochs {row['epoch']}  mAP50 {m50:.3f}  "
      f"precision {float(row['metrics/precision(B)']):.3f}  "
      f"recall {float(row['metrics/recall(B)']):.3f}")
print("BETTER than the shipped 0.928" if m50 > 0.928 else "NOT better — keep the old weights")
```

mAP50 above 0.92 is a working detector. Below 0.85, something is wrong with the
data, not the training.

### A7. Export

```python
!python ml/export_onnx.py --weights runs/detect/plate/weights/best.pt --fp32
!cd runs/detect/plate/weights && zip -qr /kaggle/working/argus-plate-weights.zip best.pt best_openvino_model
```

Use `--fp32`. int8 needs calibration images and buys nothing: measured 9 ms per
frame against a 50 ms budget.

**Check:**

```python
!ls -la /kaggle/working/argus-plate-weights.zip
```

Download it from the Output panel on the right.

### A8. Install locally

```bash
cd ~/code/Argus
mkdir -p runs/detect/plate-new/weights
unzip -o ~/Downloads/argus-plate-weights.zip -d runs/detect/plate-new/weights
```

Into `plate-new`, **not** over `plate`. Score both before replacing anything.

---

## Route B — this laptop, CPU only

**Time:** ~4 min 45 s per epoch, so 60 epochs is **about 5 hours**.
No GPU: the Radeon 860M has no ROCm installed and the XDNA2 NPU is
inference-only. This is CPU, all 16 Zen 5 threads.

Use this only if Kaggle is unavailable. It is 6x slower and the CPU preset
trades accuracy for time (`imgsz` 480 not 640, `freeze=10` not 0).

### B1. Environment

```bash
cd ~/code/Argus
./.venv/bin/python -c "import ultralytics, torch; print(ultralytics.__version__, torch.__version__)"
```

Must print a `+cpu` torch. If it does not, you have the 2.5 GB CUDA wheel that
cannot run on this machine — see CLAUDE.md, "Python install — two traps".

### B2. Dataset

```bash
BASE=https://huggingface.co/datasets/keremberke/license-plate-object-detection/resolve/main/data
mkdir -p ~/plates
for s in train valid test; do
  mkdir -p ~/plates/$s
  curl -L -o /tmp/$s.zip $BASE/$s.zip
  unzip -q -o /tmp/$s.zip -d ~/plates/$s
done
```

**Check:**

```bash
ls ~/plates/*/*.jpg | wc -l            # 8823
ls ~/plates/*/_annotations.coco.json   # three files, one per split
```

Three JSON files. One means they overwrote each other.

### B3. Convert

```bash
./.venv/bin/python ml/prepare_dataset.py --src ~/plates --subset 100000 --val 800
```

**Check** — must print `8023 train / 800 val`, with no "only N available"
warning.

### B4. Train

```bash
./.venv/bin/python ml/train_plate.py --epochs 60 --cpu --name plate-cpu
```

`--cpu` flips the whole preset, not just the device:

| | value | reason |
|---|---|---|
| `imgsz` | 480 | 640 on CPU is ~1.8x slower for a point of mAP |
| `batch` | 16 | 30 GB RAM, but small batches keep the loop responsive |
| `freeze` | 10 | freeze the backbone. Trains the head only — the difference between 5 hours and 15 |
| `amp` | False | no CPU benefit; can destabilise |
| `workers` | 6 | leave threads for the forward pass |

Runs in the foreground for hours. Use `tmux` or `nohup`:

```bash
nohup ./.venv/bin/python ml/train_plate.py --epochs 60 --cpu --name plate-cpu \
      > /tmp/train.log 2>&1 &
tail -f /tmp/train.log
```

**Check while running** — each epoch prints a line to
`runs/detect/plate-cpu/results.csv`. If the file has not grown in 10 minutes,
the run is stuck.

### B5. Score and export

```bash
./.venv/bin/python -c "
import csv; r=list(csv.DictReader(open('runs/detect/plate-cpu/results.csv')))[-1]
print('mAP50', r['metrics/mAP50(B)'], 'P', r['metrics/precision(B)'], 'R', r['metrics/recall(B)'])"
./.venv/bin/python ml/export_onnx.py --weights runs/detect/plate-cpu/weights/best.pt --fp32
```

---

## After either route — decide with numbers, not hope

Both routes land weights somewhere new. **Score the new against the old on data
neither has seen**, then swap only if it wins.

```bash
# Detection, on the held-out Indian set
./.venv/bin/python ml/validate_plate.py --model runs/detect/plate/weights/best.pt      --data ~/indian-plates
./.venv/bin/python ml/validate_plate.py --model runs/detect/plate-new/weights/best.pt  --data ~/indian-plates

# What actually matters: exact plate strings against hand-written truth
./.venv/bin/python ml/score_plates.py --model runs/detect/plate/weights/best.pt
./.venv/bin/python ml/score_plates.py --model runs/detect/plate-new/weights/best.pt --show
```

The shipped baseline, to beat:

```
mAP50 0.928   precision 0.957   recall 0.912
correct 35/45 (78%)   wrong 2 (4%)   missed 8 (18%)
```

`score_plates.py` is the one to trust. mAP measures boxes; **correct** measures
whether the system read the registration. A model with better mAP and worse
`correct` is a worse model for this project.

Swap only after the new weights win:

```bash
mv runs/detect/plate runs/detect/plate-old
mv runs/detect/plate-new runs/detect/plate
./.venv/bin/python ml/score_plates.py         # confirm the swap took
```

The sidecar picks up `runs/detect/plate/weights/best_openvino_model`
automatically — no code change, no restart beyond the worker.

---

## Where the effort actually pays

Nine of ten current failures are OCR, not detection. In rough order of value:

1. **Label more ground truth.** 45 plates is a thin test set; 200 would make
   every number below trustworthy. `ml/groundtruth_test50.csv` shows the format.
2. **Multiple crops per track.** The sidecar already reads a track's best few
   frames. Voting across them — take the plate string read most often — turns
   independent errors into a majority. Free accuracy from footage you already
   decode.
3. **A plate-specific OCR model.** PaddleOCR is trained on documents. A small
   CRNN trained on plate crops is the real ceiling-lifter, and also the biggest
   piece of work.
4. Retraining the detector on all 8,823 images. Worth the 45 minutes; not worth
   waiting on.
