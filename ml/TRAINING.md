# Training the plate detector

Primary dataset: **[Indian number plate, by Quobotic][ds]** on Roboflow
Universe. One class (`IndianNumberPlate`), CC BY 4.0. Version 3 is what is on
this machine at `~/indian-plates`: 2,573 labelled images, 2,594 boxes.

[ds]: https://universe.roboflow.com/quobotic/indian-number-plate

**No single public Indian plate dataset is large enough.** `ml/prepare_dataset.py`
merges several into one training set — see "Route C". Merged from the two
already on this machine: **10,098 unique images**, after 1,298 duplicates were
removed -- 11% of the input, most of them mirror-augmented copies.

Three routes: **A, Kaggle GPU** (recommended, ~20 min end to end); **B, this
laptop's CPU** (~1 h); **C, merge several datasets and fine-tune the detector
that already works** rather than starting from COCO weights. A and B are the
same job on different hardware. C is a different job and can run on either.

Every step has a **check** immediately after it. Run the check. A training run
that fails silently and finishes anyway is exactly how the first model ended up
seeing 15% of its data.

---

## More data: which datasets, and what each is for

Two things are being trained here and they need different data.

A **detector** needs whole photographs with a box round the plate. That is what
Routes A, B and C train, and what everything below marked *detection* feeds.

A **reader** needs plate crops with the registration as a text label. Nothing
in this project trains one yet, and it is the piece that would fix the errors
the system still makes — every remaining mistake on real footage is `O` read for
`D` or `Q` in the series letters, where there is no closed set of valid answers
to check a guess against. The two synthetic sets below exist for exactly that.

Sizes and licences below were read from Kaggle's dataset API on 26 August 2026.
**The annotation format is not verified** — Kaggle does not publish file
listings without credentials, and these sets ship as YOLO, COCO, VOC or as bare
crops with no boxes at all. `ml/prepare_dataset.py` detects which of the three
box formats it is handed and says so; a set with no boxes gets a message saying
it trains a reader, not a detector.

| dataset | size | licence | for |
|---|---|---|---|
| [`tkm22092/indian-number-plate-images`](https://www.kaggle.com/datasets/tkm22092/indian-number-plate-images) | 1.5 GB | CC0 | detection |
| [`santoshvishwakarma99/indian-license-plate-dataset`](https://www.kaggle.com/datasets/santoshvishwakarma99/indian-license-plate-dataset) | 756 MB | CC0 | detection |
| [`gauravsanwal/indian-licence-plate`](https://www.kaggle.com/datasets/gauravsanwal/indian-licence-plate) | 385 MB | unstated | detection, "annotations in txt format" |
| [`dataclusterlabs/indian-number-plates-dataset`](https://www.kaggle.com/datasets/dataclusterlabs/indian-number-plates-dataset) | 152 MB | CC0 | detection |
| [`abtexp/synthetic-indian-license-plates`](https://www.kaggle.com/datasets/abtexp/synthetic-indian-license-plates) | 1.0 GB | CC0 | **reader** — all states, all vehicle types |
| [`raspberrypi5/indian-commercial-vehicle-number-plate`](https://www.kaggle.com/datasets/raspberrypi5/indian-commercial-vehicle-number-plate) | 831 MB | CC BY-NC-SA 4.0 | **reader** — built for TrOCR |
| [`fareselmenshawii/large-license-plate-dataset`](https://www.kaggle.com/datasets/fareselmenshawii/large-license-plate-dataset) | 2.5 GB | CC0 | detection, not India-specific |
| [`adilshamim8/license-plate-recognition`](https://www.kaggle.com/datasets/adilshamim8/license-plate-recognition) | 526 MB | CC BY 4.0 | detection, not India-specific |

Two cautions that are worth more than the list.

**Check the licence before it goes in a submission.** CC BY-NC-SA forbids
commercial use and forces the same licence downstream; CC0 and CC BY do not.
The one NC set above is marked.

**The unstated one is not free by default.** "Unknown" on Kaggle means the
uploader did not choose a licence, which is not the same as permitting reuse.

The often-cited **[Indian_LPR](https://github.com/sanchit2843/Indian_LPR)** —
16,192 images, 21,683 plates, with four-point plate outlines *and* per-character
labels — is **not downloadable**. The authors state they cannot publish Indian
road data for legal reasons. It is the best-matched dataset in existence for
this problem and it is worth not spending an hour rediscovering that.

---

## Read this before you start

The Roboflow set is **2,573 images**. The one the shipped weights came from is
**8,823**.

Neither is a reason to prefer the other — it is a reason to measure. A smaller,
better-matched set can beat a larger, more generic one, and this one is
specifically Indian plates. Do not assume a switch is an upgrade: score the
result against the current weights before replacing anything, using
`ml/score_plates.py` and the numbers in "Decide with numbers" below.

The third option is the one worth remembering: **train on both at once**, which
is Route C. YOLO does not care where an image came from.

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
| Plate read correctly | 39 of 45 (87%) |
| Read wrongly | 2 (4%) |
| Not read | 4 (9%) |

**One** of the six failures is a detection miss. The other five are OCR.

That 87% is up from 78% and no detector was retrained to get there. The whole
gain came from switching off two models PaddleOCR runs by default — see
`make_reader()` in `ml/sidecar.py`. Which is the point of this section: check
what the reader is doing before paying for a training run.

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
ds = project.version(3).download("yolov8", location="/kaggle/working/plates")
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
assert total >= 2573, f"expected at least 2573 images, found {total}"

cfg = yaml.safe_load(open(DATA))
print(cfg)
assert cfg["nc"] == 1, f"expected one class, got {cfg['nc']}: {cfg['names']}"
```

`total` may exceed 2,573 if the version applies augmentation — that is fine and
the assertion allows it. `total` *below* 2,573 means a split failed to
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
| `cache` | ram | 2.6 k images at 640 fit easily; removes disk I/O from the loop |
| `patience` | 15 | stop when 15 epochs bring no improvement |
| `epochs` | 60 | ~10 s/epoch on a T4 at this dataset size |

**Check** — the header must show the right data and device:

```
train: /kaggle/working/plates/train/images ... N images
Model summary: ... 3,011,043 parameters
```

If N is far below the count A4 printed, `data.yaml` is still pointing
somewhere else. Go back to A5.

`patience=15` on a 2,573-image set will often stop the run well before 60
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
        .version(3).download("yolov8", location=os.path.expanduser("~/plates-in")))
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

Per split the two counts must be equal, and they must add up to at least 2,573.
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

## Route C — merge several datasets, then fine-tune

Route A trains one dataset from stock `yolov8n.pt`. Route C trains the union of
several, starting from the detector that already works. It exists because the
two experiments this project has already run point at it: a single dataset is
too small, and a from-scratch retrain on a *bigger* single dataset produced a
better-fitting detector that read fewer plates.

### C1. Get the sources

Fetch each one by hand into its own directory — Roboflow via the snippet in B2,
Kaggle via the website or `kaggle datasets download -d <ref>`. Nothing here
auto-downloads: dataset URLs rot, and a broken downloader is the worst thing to
debug at hour 30.

### C2. Merge

```bash
./.venv/bin/python ml/prepare_dataset.py \
    --src ~/indian-plates ~/kaggle-plates datasets/plates \
    --dst datasets/plates-merged --subset 0 --val 1200
```

Order matters: on a duplicate image the **earliest `--src` wins**, so put the
set whose annotations you trust most first.

What it prints, and why each line is there:

```
[0] /home/deep/indian-plates: YOLO annotations, 2573 labelled images, 2594 boxes
[1] datasets/plates: YOLO annotations, 8823 labelled images, 9155 boxes
dropped 1298 repeated image(s) of 11396 (11%) -- the same photograph in more than one source
merged: 10098 images
8898 train / 1200 val -> datasets/plates-merged/data.yaml
```

`--subset 0` means "everything left after the validation split"; an absolute
figure is a number nobody can know before the merge runs. `--val` is capped at a
fifth of what is available, because an absolute validation count against an
unknown merged total is how a 3,000-image merge becomes 2,000 val and 1,000
train.

- **The format per source.** If one says COCO and you expected YOLO, that is the
  moment to notice, not after training.
- **The duplicate count.** Public plate datasets are largely re-uploads of one
  another. A photograph in train and again in val means the model is validated
  on an image it was trained on: val mAP rises and every number downstream is
  quietly wrong. Deduplication runs *before* the split, by perceptual hash, so a
  re-encoded copy under a different name is still caught, as is a mirrored one
  -- flipping is the commonest augmentation these datasets apply. **11% here**,
  and within a single source as well as across two. It was 4% before the hash
  was made mirror-invariant: 887 of these 11,396 images are a left-right flip
  of another one.
- **Images per source.** A source contributing far fewer images than it contains
  usually means unreadable annotations, not a small dataset.

`--dst` is deleted before it is written, so it may not be inside any `--src`.
The script refuses rather than destroying the source.

### C3. Fine-tune, do not restart

```bash
nohup ./.venv/bin/python ml/train_plate.py --cpu \
      --data datasets/plates-merged/data.yaml \
      --model runs/detect/plate/weights/best.pt \
      --name plate-ft --epochs 12 --patience 4 --lr0 0.002 \
      > /tmp/ft.log 2>&1 &
```

Two arguments carry the whole idea.

`--model runs/detect/plate/weights/best.pt` starts from the detector that
already reads 39 of 45 plates rather than from COCO weights that have never seen
one.

`--lr0 0.002` is what keeps it there. Ultralytics defaults to `lr0=0.01`, which
is correct for a random head and large enough to walk a trained detector away
from what it learned — arriving somewhere no better than a fresh run, which is
the experiment that already cost this project two correct reads.

On Kaggle, the same thing with the GPU preset:

```python
!python ml/train_plate.py --data datasets/plates-merged/data.yaml \
        --model runs/detect/plate/weights/best.pt \
        --name plate-ft --epochs 25 --lr0 0.002 --device 0
```

Getting the merged set onto Kaggle: run C2 locally, then upload
`datasets/plates-merged` as a Kaggle Dataset, or run C1 and C2 inside the
notebook — `prepare_dataset.py` is in the repo the notebook already clones.

### C4. Judge it

Exactly as in "Decide with numbers" below, and by `correct`, never by mAP. A
fine-tune that improves mAP and loses a correct read is a worse model for this
project. Re-sweep `PLATE_PAD` first: a detector that has moved has changed how
tightly it crops, and the crop is what OCR sees.

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

Measured on the same 45 hand-labelled plates, every row re-measured with the
CURRENT reader — a comparison across two different readers says nothing about
the detectors:

| weights | training images | mAP50 | pad | correct | wrong | missed | precision |
|---|---|---|---|---|---|---|---|
| `plate` (shipped) | 1,365 | 0.928 | 0.08 | **39 (87%)** | **2** | 4 | **95%** |
| `plate-cpu` (full retrain) | 8,023 | 0.991 | 0.08 | 39 (87%) | 2 | 4 | 95% |
| `plate-cpu` | 8,023 | 0.991 | 0.14 | 40 (89%) | 3 | 2 | 93% |
| `plate-cpu` | 8,023 | 0.991 | 0.20 | 39 (87%) | 3 | 3 | 93% |
| `plate-ft` (Route C fine-tune) | 9,785 merged | 0.982 | 0.08 | 39 (87%) | 2 | 4 | 95% |
| `plate-ft` | 9,785 merged | 0.982 | 0.14 | 39 (87%) | 2 | 4 | 95% |
| `plate-ft` | 9,785 merged | 0.982 | 0.20 | 38 (84%) | 3 | 4 | 93% |

On the real video (`ml/groundtruth_kiit.csv`, 5 legible plates, all five found
by every model): shipped weights read **4 exactly right**, `plate-ft` **3**.

### The detector is finished; this test set cannot see past it

Three detectors, trained on 1,365, 8,023 and 9,785 images, spanning mAP50 0.928
to 0.991, all read **exactly 39 of 45**. That is not a coincidence and it is not
noise. The detector finds a plate in **44 of the 45** photographs, so the most a
perfect detector could add is one plate. The remaining six failures are four
misses and two wrong reads, and every one of them is the reader.

Two conclusions follow, and they are the useful output of a lot of GPU time:

**Stop training detectors against this file.** It has no resolution left. More
detector work needs either footage the detector actually fails on, or a bigger
labelled test set — item 2 under "Where the effort actually pays".

**A fine-tune on more data is not free.** `plate-ft` ties on the photographs and
loses one plate on the video, which is the only test here with motion, glare and
distance in it. Higher mAP, same or slightly worse reading, for 78 minutes of
CPU. Shipped weights stay.

**This table used to say something different, and the difference is the lesson.**
Under the previous reader the shipped weights scored 35 and the full retrain 33,
and this file concluded that a better-fitted detector reads worse because it
crops tighter. Re-measured with the reader fixed, the two detectors are **exactly
level**. The gap was never the detector. It was the reader mangling a crop that
the tighter model happened to produce more often.

The 0.14 row is the one worth arguing about, and it is a refusal: one more plate
read correctly, one more read WRONGLY. A miss leaves the vehicle to Re-ID; a
wrong read puts a real registration on the wrong vehicle. Precision when it
answers is the column that decides, and it drops from 95% to 93%. Shipped
weights at 0.08 stay.

What survives unchanged: **mAP measures boxes, `correct` measures whether the
system read the registration**, and they can move independently in either
direction. Judge by `correct`, then by precision. And re-measure BOTH sides
whenever either the detector or the reader changes — a stale comparison is worse
than none, as this table demonstrated for two weeks.

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

Five of six current failures are OCR, not detection. In rough order of value:

1. **A plate-specific OCR model — STARTED, see `ml/train_reader.py`.** Every
   error left on real footage is `O` read for `D` or `Q` in the series letters.
   The state code has a closed set of 36 valid answers and `_state_code()` in
   `ml/sidecar.py` repairs it against that list; the series letters have no such
   set, so a guess there would corrupt genuine `O` series and cannot be made
   safely. PaddleOCR is trained on documents.

   A 2.11M-parameter CRNN with CTC loss now exists and trains on this laptop's
   CPU. What it measured:

   | training data | correct of 45 | CER | precision |
   |---|---|---|---|
   | PaddleOCR (shipped, for comparison) | **39 (87%)** | 9.4% | 95% |
   | 48,026 synthetic crops only | 0 (0%) | 82.6% | 0% |
   | + 1,756 real crops, 8 epochs | 16 (36%) | 34.9% | 42% |

   **The synthetic sets on their own are worthless for this.** Both are renders;
   a model trained on them reads held-out renders at 83% and real photographs at
   zero. It learns a font. Real crops are the whole game, and eight epochs on
   fewer than two thousand of them took it from nothing to a third, with the
   remaining errors becoming near-misses rather than nonsense.

   `ml/make_reader_crops.py` builds them: crop every hand-drawn plate box out of
   the Indian detection set, label it with PaddleOCR, keep only reads that
   `correct_plate()` accepts with little repair. `--video` does the same over
   traffic footage through the sidecar's own vehicle-then-plate stages, which is
   the only plentiful source. Two traps it handles, both of which silently
   poison the labels:

   - **43% of the Roboflow crops are horizontally mirrored.** Every crop is read
     both ways round and the higher-scoring orientation wins.
   - **Some dataset boxes are not plate-shaped** — one is 106x439, taller than
     it is wide. OCR returns something from them anyway.

   Never build crops from `~/indian-plates/test` or the KIIT clip. Those are the
   test sets.
2. **Label more ground truth.** 45 plates is a thin test set; 200 would make
   every number in this file trustworthy, and it is an afternoon of typing.
   `ml/groundtruth_test50.csv` shows the format, and `ml/groundtruth_kiit.csv`
   shows the same for a video, one line per vehicle rather than per track.
3. **Train on several datasets at once.** Route C. 10,240 unique images from the
   four now on this machine, and more from the table above — but note that only
   1,648 of them are INDIAN plates. `datasets/plates`, the bulk of every full
   retrain here, is a generic European set. Harmless for a detector, worthless
   for a reader.
4. **Tune `PLATE_PAD`.** Already worth 62% → 78% once. Free, and it must be
   redone whenever the detector changes.
5. Retraining the detector on one dataset from scratch. Cheap, and twice now
   shown not to move end-to-end accuracy by itself.
