# Training the plate detector

Dataset: **[Indian number plate, by Quobotic][ds]** on Roboflow Universe.
1,683 images, one class (`IndianNumberPlate`), CC BY 4.0, dataset version 1.

[ds]: https://universe.roboflow.com/quobotic/indian-number-plate

Two routes: **Kaggle GPU** (recommended, ~20 min end to end) and **this
laptop's CPU** (~1 h). Same dataset, same script, same output. Pick one.

Every step has a **check** immediately after it. Run the check. A training run
that fails silently and finishes anyway is exactly how the first model ended up
seeing 15% of its data.

---

## Read this before you start

This dataset is **1,683 images**. The one the shipped weights came from is
**8,823**. Five times smaller.

That is not a reason to avoid it — it is a reason to measure. A smaller,
better-matched set can beat a larger, more generic one, and this one is
specifically Indian plates. But do not assume a switch is an upgrade: score the
result against the current weights before replacing anything, using
`ml/score_plates.py` and the numbers in "Decide with numbers" below.

The third option is the one worth remembering: **train on both**. YOLO does not
care where an image came from, so pointing `data.yaml` at the union of the two
label sets is a few lines of file copying and gives 10,500 images.

---

## What went wrong the first time

The shipped weights (`runs/detect/plate/weights/best.pt`, mAP50 0.928) were
trained on **1,365 of 8,823 images**. Not on purpose.

That dataset shipped as three zips, and each contained its own
`_annotations.coco.json` **at the archive root**. The download cell unzipped
train and valid into the same folder, so valid's 1,765 annotations overwrote
train's 6,176. `prepare_dataset.py` printed

```
warning: only 1365 train images available, wanted 3000
```

…and training continued. The warning scrolled past in a Kaggle log.

**The Roboflow route cannot fail this way**, because Roboflow exports YOLO
format — one `.txt` label file next to each image, in per-split directories.
There is no shared annotation file to overwrite. The checks below still count
images against labels, because "cannot fail this way" is not the same as
"cannot fail".

---

## Is more detector training going to help?

Measured against 45 hand-labelled plates in `ml/groundtruth_test50.csv`:

| | |
|---|---|
| Detector found the plate | **44 of 45** |
| Plate read correctly | 35 of 45 (78%) |
| Read wrongly | 2 (4%) |
| Not read | 8 (18%) |

**One** of the ten failures is a detection miss. The other nine are OCR.

This was then confirmed the expensive way. A full retrain on all 8,023 training
images pushed detection from mAP50 0.928 to **0.991** — and end-to-end accuracy
went **down**, from 35 correct to 33, with wrong reads doubling from 2 to 4.

A better-fitted detector draws a *tighter* box, and a tighter crop is worse for
OCR: the last character falls outside it. `PLATE_PAD` in `ml/sidecar.py` exists
for exactly this, and `ml/score_plates.py --pad` re-measures it. **Any new
detector needs its padding re-measured before it is judged.**

So: train if you want a better detector. Do not expect a better system for
free, and do not skip the OCR work waiting for it.

---

## Route A — Kaggle GPU (recommended)

**Time:** ~10 min setup, ~10 min training. **Cost:** free, 30 GPU-h/week.

### A1. Get a Roboflow API key

Free account at <https://app.roboflow.com>. Then **Settings → API Keys →
Private API Key**. Copy it.

On Kaggle, store it as a secret rather than pasting it into a cell: **Add-ons →
Secrets → Add secret**, label `ROBOFLOW_API_KEY`. Notebooks get shared; keys in
cells get shared with them.

### A2. Enable the GPU

New notebook at <https://www.kaggle.com/code> → right sidebar → **Session
options**:

- Accelerator: **GPU T4 x2**
- Internet: **On** (needed to reach Roboflow)

Both need a phone-verified account. Settings → Phone verification, once.

**Check** — first cell:

```python
!nvidia-smi
```

Must print a Tesla T4 and a memory figure. No GPU listed means the accelerator
did not attach — restart the session rather than training on CPU by accident.

### A3. Get the code

```python
%cd /kaggle/working
!rm -rf Argus && git clone -q https://github.com/Deeptanshu789/Argus.git
%cd Argus
!pip install -q ultralytics roboflow
```

**Check:**

```python
import ultralytics, roboflow
print(ultralytics.__version__, roboflow.__version__)
```

### A4. Download the dataset

```python
from kaggle_secrets import UserSecretsClient
from roboflow import Roboflow

key = UserSecretsClient().get_secret("ROBOFLOW_API_KEY")
project = Roboflow(api_key=key).workspace("quobotic").project("indian-number-plate")
ds = project.version(1).download("yolov8", location="/kaggle/working/plates")
DATA = ds.location + "/data.yaml"
print(DATA)
```

`download("yolov8")` is what makes `prepare_dataset.py` unnecessary here — it
writes `train/images`, `train/labels`, `valid/…`, `test/…` and a `data.yaml`
already in the format Ultralytics wants.

**Check — do not skip this:**

```python
import glob, os, yaml

root = ds.location
total = 0
for split in ("train", "valid", "test"):
    imgs = glob.glob(f"{root}/{split}/images/*")
    labs = glob.glob(f"{root}/{split}/labels/*.txt")
    print(f"{split:6} {len(imgs):5} images  {len(labs):5} labels")
    # An image with no label file is a silent negative sample: YOLO trains on
    # it as "no plate here" and quietly teaches the model to miss.
    assert len(imgs) == len(labs), f"{split}: {len(imgs)} images but {len(labs)} labels"
    total += len(imgs)

print("total", total)
assert total >= 1683, f"expected at least 1683 images, found {total}"

cfg = yaml.safe_load(open(DATA))
print(cfg)
assert cfg["nc"] == 1, f"expected one class, got {cfg['nc']}: {cfg['names']}"
```

`total` may exceed 1,683 if the version applies augmentation — that is fine and
the assertion allows it. `total` *below* 1,683 means a split failed to
download; re-run rather than proceeding.

### A5. Fix the paths in data.yaml

Roboflow writes relative paths (`train: ../train/images`) that resolve against
the working directory, not the file. Trained from anywhere else, Ultralytics
looks in the wrong place — and when it finds nothing it warns and carries on
with an empty split.

```python
import yaml
cfg = yaml.safe_load(open(DATA))
for split in ("train", "val", "test"):
    if split in cfg:
        cfg[split] = f"{root}/{'valid' if split == 'val' else split}/images"
cfg["path"] = root
yaml.safe_dump(cfg, open(DATA, "w"))
print(open(DATA).read())
```

**Check** — every path must exist and be non-empty:

```python
import os
for split in ("train", "val", "test"):
    p = cfg.get(split)
    if p:
        n = len(os.listdir(p))
        print(f"{split:6} {p}  {n} files")
        assert n > 0, f"{split} path resolves to an empty directory"
```

### A6. Train

```python
!python ml/train_plate.py --data {DATA} --epochs 60 --device 0
```

Parameters, and why (`ml/train_plate.py`, `PRESETS["gpu"]`):

| | value | reason |
|---|---|---|
| `imgsz` | 640 | plates are small; 480 loses distant ones. GPU affords it |
| `batch` | 32 | fits T4 16 GB at 640 |
| `freeze` | 0 | full fine-tune. COCO has no plate class, so late layers alone are not enough |
| `amp` | True | half precision, ~2x faster on T4, no accuracy cost here |
| `cache` | ram | 1.7 k images at 640 fit easily; removes disk I/O from the loop |
| `patience` | 15 | stop when 15 epochs bring no improvement |
| `epochs` | 60 | ~10 s/epoch on a T4 at this dataset size |

**Check** — the header must show the right data and device:

```
train: /kaggle/working/plates/train/images ... N images
Model summary: ... 3,011,043 parameters
```

If N is far below the count A4 printed, `data.yaml` is still pointing
somewhere else. Go back to A5.

`patience=15` on a 1,683-image set will often stop the run well before 60
epochs. That is the mechanism working, not a failure.

### A7. Read the score

```python
import csv
row = list(csv.DictReader(open("runs/detect/plate/results.csv")))[-1]
m50 = float(row["metrics/mAP50(B)"])
print(f"epochs {row['epoch']}  mAP50 {m50:.3f}  "
      f"precision {float(row['metrics/precision(B)']):.3f}  "
      f"recall {float(row['metrics/recall(B)']):.3f}")
```

This is a score on *this dataset's* validation split, so it is not comparable
to the 0.928 or 0.991 quoted elsewhere in this file — those were measured on a
different set. The only comparable number is `ml/score_plates.py`, run locally
on the same 45 hand-labelled images for both models. See "Decide with numbers".

mAP50 above 0.90 is a working detector. Below 0.80 with this dataset, suspect
the paths, not the training.

### A8. Export

```python
!python ml/export_onnx.py --weights runs/detect/plate/weights/best.pt --fp32
!cd runs/detect/plate/weights && zip -qr /kaggle/working/argus-plate-weights.zip \
      best.pt best_openvino_model
```

Use `--fp32`. int8 needs calibration images and buys nothing here: measured
9 ms per frame against a 50 ms budget.

**Check:**

```python
!ls -la /kaggle/working/argus-plate-weights.zip
```

Download it from the Output panel on the right.

### A9. Install locally

```bash
cd ~/code/Argus
mkdir -p runs/detect/plate-new/weights
unzip -o ~/Downloads/argus-plate-weights.zip -d runs/detect/plate-new/weights
```

Into `plate-new`, **not** over `plate`. Score both before replacing anything.

---

## Route B — this laptop, CPU only

**Time:** measured on this machine at 8,023 images, `imgsz=480`, `freeze=10`,
with per-epoch validation: **4.2 min/epoch** (60 epochs took 255 minutes). This
dataset is about a fifth the size, so expect **roughly 1 minute an epoch, about
an hour for 60**.

No GPU: the Radeon 860M has no ROCm installed and the XDNA2 NPU is
inference-only. This is CPU, all 16 Zen 5 threads.

The CPU preset trades accuracy for time (`imgsz` 480 not 640, `freeze=10` not
0), so its weights are not equivalent to Route A's. At an hour, though, this is
now a reasonable first run rather than a fallback.

### B1. Environment

```bash
cd ~/code/Argus
./.venv/bin/pip install -q roboflow
./.venv/bin/python -c "import ultralytics, torch; print(ultralytics.__version__, torch.__version__)"
```

Must print a `+cpu` torch. If it does not, you have the 2.5 GB CUDA wheel that
cannot run on this machine — see CLAUDE.md, "Python install — two traps".

### B2. Dataset

Get the key from <https://app.roboflow.com> → Settings → API Keys. Pass it
through the environment; do not paste it into a file that gets committed.

```bash
export ROBOFLOW_API_KEY=xxxxxxxx
./.venv/bin/python - <<'PY'
import os
from roboflow import Roboflow
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
ds = (rf.workspace("quobotic").project("indian-number-plate")
        .version(1).download("yolov8", location=os.path.expanduser("~/plates-in")))
print("downloaded to", ds.location)
PY
```

**Check:**

```bash
for s in train valid test; do
  printf '%-6s %5s images %5s labels\n' $s \
    "$(ls ~/plates-in/$s/images 2>/dev/null | wc -l)" \
    "$(ls ~/plates-in/$s/labels 2>/dev/null | wc -l)"
done
```

Per split the two counts must be equal, and they must add up to at least 1,683.
An image with no label file trains the model that there is no plate in it.

### B3. Fix the paths in data.yaml

Same trap as A5 — relative paths that resolve against the working directory.

```bash
./.venv/bin/python - <<'PY'
import os, yaml
root = os.path.expanduser("~/plates-in")
p = f"{root}/data.yaml"
cfg = yaml.safe_load(open(p))
cfg["path"] = root
for split, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
    if split in cfg:
        cfg[split] = f"{root}/{folder}/images"
yaml.safe_dump(cfg, open(p, "w"))
print(open(p).read())
for split in ("train", "val", "test"):
    d = cfg.get(split)
    if d:
        assert os.path.isdir(d) and os.listdir(d), f"{split} -> {d} missing or empty"
print("paths OK")
PY
```

### B4. Train

```bash
nohup ./.venv/bin/python ml/train_plate.py \
      --data ~/plates-in/data.yaml --epochs 60 --cpu --name plate-rf \
      > /tmp/train.log 2>&1 &
tail -f /tmp/train.log
```

`--cpu` flips the whole preset, not just the device:

| | value | reason |
|---|---|---|
| `imgsz` | 480 | 640 on CPU is ~1.8x slower for a point of mAP |
| `batch` | 16 | 30 GB RAM, but small batches keep the loop responsive |
| `freeze` | 10 | freeze the backbone, train the head. The difference between one hour and three |
| `amp` | False | no CPU benefit, and it can destabilise |
| `workers` | 6 | leave threads for the forward pass |

**Check while running** — each epoch appends a line to
`runs/detect/plate-rf/results.csv`. If that file has not grown in 10 minutes,
the run is stuck, not slow.

```bash
watch -n 60 'wc -l runs/detect/plate-rf/results.csv'
```

### B5. Score and export

```bash
./.venv/bin/python -c "
import csv; r=list(csv.DictReader(open('runs/detect/plate-rf/results.csv')))[-1]
print('mAP50', r['metrics/mAP50(B)'], 'P', r['metrics/precision(B)'], 'R', r['metrics/recall(B)'])"
./.venv/bin/python ml/export_onnx.py --weights runs/detect/plate-rf/weights/best.pt --fp32
```

---

## Decide with numbers, not hope

Both routes land weights somewhere new. **Score the new against the old on data
neither has seen**, then swap only if it wins.

```bash
# Boxes only — useful, but not the number that decides anything
./.venv/bin/python ml/validate_plate.py --model runs/detect/plate/weights/best.pt     --data ~/indian-plates
./.venv/bin/python ml/validate_plate.py --model runs/detect/plate-new/weights/best.pt --data ~/indian-plates

# What actually matters: exact plate strings against hand-written truth
./.venv/bin/python ml/score_plates.py --model runs/detect/plate/weights/best.pt
./.venv/bin/python ml/score_plates.py --model runs/detect/plate-new/weights/best.pt --show
```

**Re-measure the padding for any new detector before judging it.** A new model
crops differently, and the crop is what OCR sees:

```bash
for p in 0.04 0.08 0.14 0.20; do
  echo "pad $p"
  ./.venv/bin/python ml/score_plates.py --model runs/detect/plate-new/weights/best.pt --pad $p \
    | grep -E "correct|wrong|missed"
done
```

Then set `PLATE_PAD` in `ml/sidecar.py` to whichever won.

Measured so far, on the same 45 hand-labelled plates:

| weights | training images | mAP50 | correct | wrong | missed |
|---|---|---|---|---|---|
| `plate` (shipped) | 1,365 | 0.928 | **35 (78%)** | **2** | 8 |
| `plate-cpu` (full retrain) | 8,023 | 0.991 | 33 (73%) | 4 | 8 |

The second model detects better by every box metric and reads worse. That table
is the entire argument for this section: **mAP measures boxes, `correct`
measures whether the system read the registration.** A model with better mAP and
worse `correct` is a worse model for this project.

Swap only after the new weights win on `correct`:

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
   every number in this file trustworthy, and it is an afternoon of typing.
   `ml/groundtruth_test50.csv` shows the format.
2. **Tune `PLATE_PAD`.** Already worth 62% → 78% once. Free, and it must be
   redone whenever the detector changes.
3. **A plate-specific OCR model.** PaddleOCR is trained on documents. A small
   CRNN trained on plate crops is the real ceiling-lifter, and also the biggest
   piece of work.
4. **Train on both datasets at once.** 1,683 Roboflow images plus 8,823 from the
   original set, in one `data.yaml`. More useful than either alone.
5. Retraining the detector on one dataset. Cheap, and already shown not to move
   end-to-end accuracy by itself.
