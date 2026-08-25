# Argus — Workflow

The operational runbook: what to run, in what order, by whom. Planning and
rationale live in the [wiki](https://github.com/Deeptanshu789/Argus/wiki); this
file is the thing you keep open while building.

Two people:

- **Dev A** — model training, backend, ML pipeline, database, Docker
- **Dev B** — frontend, visualization, demo assets

---

## The one rule that shapes everything

**Freeze the API contract before either developer writes real code.**

With two people and no code review, an unannounced response-shape change costs
an hour of confused debugging at exactly the moment neither of you has an hour.
The contract is frozen, the mock server already serves it, and the frontend is
built against fixtures from minute one.

Training used to be the long pole; it no longer is. It runs on **Kaggle GPU** in
about half an hour (see Stage 1). What is still true: develop the pipeline
against stock `yolov8n.pt` with a stub plate region, and swap the trained
weights in at a single call site. Never block a developer on a training run.

## Stage 0 — One-time setup

**Dev A, ~30 min, hour 0**

Docker is not installed on the build machine. On Fedora:

```bash
sudo dnf -y install dnf-plugins-core
sudo dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo
sudo dnf -y install docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker $USER      # log out and back in for this to take effect
```

Python environment:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt          # mock API: fastapi + uvicorn
pip install ultralytics paddleocr torchreid opencv-python openvino \
            psycopg2-binary redis celery                # real pipeline
```

Bring up infra and apply the schema (Compose applies `db/schema.sql`
automatically on a first-time `up`; the explicit `psql` is for re-applying):

```bash
docker compose up -d db redis
psql postgresql://argus:argus@localhost:5432/argus -f db/schema.sql
```

**Dev B in parallel:** `npm create vite@latest frontend -- --template react-ts`,
then `npm i deck.gl maplibre-gl recharts`.

---

## Stage 1 — Train on Kaggle

**Repo owner. ~45 min wall clock, mostly unattended.**

Training runs on Kaggle's free GPU (T4 x2 / P100), not on the build laptop —
the laptop has no CUDA device, and the same job there takes 5-13 hours instead
of half an hour.

### 1. Get a dataset

Nothing is auto-downloaded: dataset URLs and layouts rot, and a broken
downloader at hour 0 is the worst possible time to debug one. On Kaggle, use
**Add Data** to attach one in YOLO format — Roboflow Universe exports work
directly.

- <https://universe.roboflow.com/search?q=indian+number+plate>
- <https://www.kaggle.com/datasets/praveen12345/indian-number-plate-detection>
- <https://huggingface.co/datasets/Dataclusterlabspvtltd/indian-number-plates-dataset>

### 2. Run the notebook

Upload `ml/kaggle_train.ipynb` to Kaggle, then:

| Setting | Value |
|---|---|
| Accelerator | **GPU T4 x2** (or P100) |
| Internet | **On** — needed for `pip install` and the repo clone |
| Add Data | your plate dataset |

Set `SRC` in the first code cell to the attached dataset path
(`/kaggle/input/<slug>`), then Run All.

The notebook clones this repo and calls `ml/prepare_dataset.py` and
`ml/train_plate.py` — no training code is duplicated in the notebook, so the
Kaggle path and the local path can never drift apart.

**Expect ~20-40 s/epoch. 50 epochs is roughly 20-35 minutes.**

GPU defaults are `imgsz=640, batch=32, amp=True, freeze=0`. Note `freeze=0` —
the backbone is *not* frozen. Freezing it is a CPU concession that costs
accuracy, and on a T4 there is no reason to pay it.

Kaggle allows 12 h per session and 30 h/week, so there is ample headroom. If a
session dies, re-run the setup cells and add `--resume`.

### 3. Bring the weights back

Download `argus-plate-weights.zip` from the notebook's Output panel, then
locally:

```bash
mkdir -p runs/detect/plate/weights
unzip argus-plate-weights.zip -d runs/detect/plate/weights
python ml/export_onnx.py --weights runs/detect/plate/weights/best.pt
```

**Go/no-go bar: `mAP50 >= 0.85` on val.** Single-class, tight-boxed plates reach
this readily. If it stalls below 0.7 the dataset conversion is wrong, not the
training — check that labels are class `0` and boxes are normalized `xywh`.

Since a GPU run is cheap, if the first result disappoints, raise `SUBSET`
toward the full 15,000 images and re-run. That option did not exist on CPU.

### Local fallback — only if Kaggle is unavailable

```bash
python ml/prepare_dataset.py --src <dataset> --subset 3000 --val 400
python ml/train_plate.py --cpu --epochs 1        # calibration — read the time
nohup nice -n 10 python ml/train_plate.py --cpu --epochs <N> > ml/train.log 2>&1 &
```

`--cpu` flips a whole preset: `imgsz=480`, frozen backbone, `workers=6` so the
laptop stays usable. Budget epochs from the **measured** minutes-per-epoch the
calibration run prints, not from an estimate — published CPU throughput for
YOLOv8n varies about 3x across reports. The script prints the recommendation
itself.

---

## Stage 2 — The API contract is already frozen

**Done. Both developers start from here.**

[[API-Contract]] specifies every REST endpoint and WebSocket message shape, and
`backend/mock.py` already serves all of them from seeded fixtures:

```bash
pip install -r backend/requirements.txt
uvicorn backend.mock:app --reload --port 8000
python backend/mock.py            # fixture selfcheck, no server needed
```

Every route and all four WebSocket message types are smoke-tested. Fixtures are
seeded with `Random(0)`, so reloads give identical data — a chart that
reshuffles on refresh makes it impossible to tell a UI bug from new data.

The frontend reads one constant:

```ts
export const API = import.meta.env.VITE_API_URL +
  (import.meta.env.VITE_MOCK ? "/api/mock" : "/api");
```

Switching the whole frontend from fixtures to live data is one environment
variable. That is the point.

The fixtures deliberately include the cases that break naive UIs:

- ~15% of tracks have `plate_text: null` — Re-ID-only matching is exactly what
  Act 2 of the demo shows off, so the UI must render it
- `GET /search` on an unknown plate returns **200 with empty arrays**, not 404
- one camera reports `status: "degraded"`
- hops carry `method` of `plate`, `reid`, or `spatial_temporal`, so the map can
  colour by which matching layer confirmed each leg
- `KA05MR7821` is a known-good plate that search will always find — use it in
  rehearsal

**Response shapes are frozen. Changing one is a contract change: announce it.**

---

## Stage 3 — Parallel build

**Hours 2-20.**

Both devs work their own trees per the wiki's Work-Split page. Dev A's pipeline
runs against **stock `yolov8n.pt` + a stub plate region** (bottom-center of the
vehicle crop) — not the training output.

### Hour 10 integration checkpoint — a hard gate

This must work before anything else proceeds:

```
camera feed -> vehicle detection -> plate OCR -> ByteTrack -> tracked vehicle
               with plate text and a 512-dim Re-ID embedding
```

If it does not work at hour 10, stop adding features and fix it.

### CPU inference budget — respect it or the demo drops frames

- Process video at **5 FPS**, not 30.
- Vehicle detection + ByteTrack: every processed frame.
- Plate detection + OCR: a track's **first few frames and its best crop only**.
- Re-ID embedding: **on track exit only** — that is the only moment
  cross-camera association needs it.

4 streams x 5 FPS = 20 vehicle-detection inferences/sec. Quantized YOLOv8n at
480px on 16 Zen 5 threads clears that with headroom; making OCR or Re-ID
per-frame does not.

---

## Stage 4 — Swap in the trained model

**Dev A, whenever the Kaggle weights land — likely early, since the run is
~30 min rather than overnight.**

```bash
python ml/export_onnx.py --weights runs/detect/plate/weights/best.pt
```

OpenVINO IR rather than raw PyTorch — it runs on AMD x86 CPUs and typically
gives 2-3x over stock torch CPU. That margin is what makes the framerate fit.
Fall back to `--format onnx` if OpenVINO chokes on a layer.

Then change one path in the ANPR service config and restart the backend.

Measure both: `mAP50` on val, and end-to-end FPS at 4 streams. **If FPS misses,
cut `imgsz` before cutting features.**

---

## Stage 5 — Integrate, harden, record

**Hours 20-36.**

- Real WebSocket pipeline replaces `/api/mock/*`.
- Tune association thresholds on a 2-camera test (two videos of the same route).
  Require 2 of 3 layers to agree before confirming a match — that is the defence
  against false trajectories.
- End-to-end run on all 4 streams; fix what breaks.
- **Record the backup demo video by hour 34. Non-negotiable.** A live demo
  failure with no fallback loses the whole thing.
- Presentation and Q&A rehearsal, hours 34-36.

---

## Daily-loop commands

| Want | Run |
|---|---|
| Infra only (usual dev mode) | `docker compose up -d db redis` |
| Whole stack | `docker compose up -d` |
| Real backend dev server | `uvicorn backend.main:app --reload` |
| Frontend dev server | `cd frontend && npm run dev` |
| Mock API (works today) | `uvicorn backend.mock:app --reload --port 8000` |
| Verify mock fixtures | `python backend/mock.py` |
| Check a local fallback run | `tail -f ml/train.log` |
| Resume a dead Kaggle run | re-run setup cells, then `--resume` |
| Run pipeline on one video | `python -m backend.pipeline --source demo/cam1.mp4 --camera CAM1` |
| Re-apply DB schema | `psql "$DATABASE_URL" -f db/schema.sql` |
| Verify dataset prep logic | `python ml/prepare_dataset.py --selfcheck` |
