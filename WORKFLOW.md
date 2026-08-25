# Argus — Workflow

The operational runbook: what to run, in what order, by whom. Planning and
rationale live in the [wiki](https://github.com/Deeptanshu789/Argus/wiki); this
file is the thing you keep open while building.

Two people:

- **Dev A** — model training, backend, ML pipeline, database, Docker
- **Dev B** — frontend, visualization, demo assets

---

## The one rule that shapes everything

**Start model training before you write any code, then walk away from it.**

Training the plate detector on this machine is CPU-only and takes 5-13 hours.
That is fine — it is unattended. It is only fine *if it starts at hour 0*.
Everything downstream is developed against stock `yolov8n.pt` with a stub plate
region, and the trained weights drop in later at a single call site.

The worst failure mode for this project is discovering at hour 20 that the
detector still needs eight hours of CPU.

---

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
pip install ultralytics paddleocr torchreid opencv-python openvino \
            fastapi "uvicorn[standard]" psycopg2-binary redis celery
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

## Stage 1 — Start training

**Dev A, hour 0. Do this before Stage 2.**

Datasets rot, so nothing is auto-downloaded. Fetch one by hand — Roboflow
Universe is the easiest since it exports YOLO format directly:

- <https://universe.roboflow.com/search?q=indian+number+plate>
- <https://huggingface.co/datasets/Dataclusterlabspvtltd/indian-number-plates-dataset>
- <https://www.kaggle.com/datasets/praveen12345/indian-number-plate-detection>

```bash
python ml/prepare_dataset.py --src ~/Downloads/indian-plates --subset 3000 --val 400
```

### The calibration run — do not skip this

```bash
python ml/train_plate.py --epochs 1
```

Read the reported minutes-per-epoch and budget from **your measured number**,
not from an estimate. Published CPU throughput for YOLOv8n varies about 3x
across reports, which is exactly why this step exists.

| Measured | Do this |
|---|---|
| **< 15 min/epoch** | `epochs = floor(available_hours * 60 / minutes_per_epoch)`, capped at 60 |
| **15-25 min/epoch** | Re-prepare at `--subset 2000`, retrain with `--imgsz 416`, re-calibrate |
| **> 25 min/epoch** | Local no longer fits. Move this one job to Kaggle (`--device 0` in a notebook, ~30 min) and bring the weights back |

### Launch the real run in the background

```bash
nohup nice -n 10 python ml/train_plate.py \
      --epochs <N> --workers 6 --patience 15 > ml/train.log 2>&1 &
```

`nice -n 10` and `--workers 6` leave ~4 threads free so the laptop stays usable
for backend work. It lengthens the run maybe 20-30%; take that trade.

Check with `tail -f ml/train.log`. **Do not wait on it.** Go to Stage 2.

If the run dies: `python ml/train_plate.py --resume`.

**Go/no-go bar:** `mAP50 >= 0.85` on val. Single-class, tight-boxed plates reach
this readily. If it stalls below 0.7, the dataset conversion is wrong, not the
training — check that labels are class `0` and boxes are normalized `xywh`.

---

## Stage 2 — Freeze the API contract

**Both devs together, hours 0-2. Nothing else starts until this is done.**

Write `API-Contract.md` in the wiki: every REST endpoint, every WebSocket
message shape, exactly. Then Dev A stubs `/api/mock/*` returning fixture JSON
in those shapes.

From that moment Dev B is unblocked permanently and never waits on the backend.
In a 2-person split this is the highest-leverage 90 minutes of the entire
build — without it the frontend serializes behind the backend for 20 hours.

Response shapes are frozen. Changing one is a contract change: announce it.

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

**Dev A, hour ~20, or whenever training finishes.**

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
| Backend dev server | `uvicorn backend.main:app --reload` |
| Frontend dev server | `cd frontend && npm run dev` |
| Check training progress | `tail -f ml/train.log` |
| Resume a dead training run | `python ml/train_plate.py --resume` |
| Run pipeline on one video | `python -m backend.pipeline --source demo/cam1.mp4 --camera CAM1` |
| Re-apply DB schema | `psql "$DATABASE_URL" -f db/schema.sql` |
| Verify dataset prep logic | `python ml/prepare_dataset.py --selfcheck` |
