# Argus — Claude Code context

City-wide ANPR + cross-camera vehicle trajectory tracking + traffic analytics.
Smart India Hackathon problem statement **SIH26127**, org **Bharat Electronics
Limited (BEL)**. 36-hour build, 2 people.

Full docs live in the GitHub wiki: <https://github.com/Deeptanshu789/Argus/wiki>
The original strategy document is `sih26127_implementation_plan.md`.
The operational runbook is `WORKFLOW.md` — read that before running anything.

## What actually differentiates this project

Any team can ship single-camera ANPR. The thing that wins is **Module C, the
cross-camera association engine**: matching the same vehicle across different
cameras via three layers — exact plate text (60-70% of cases), OSNet Re-ID
embedding cosine similarity > 0.75 (20-25%), and a spatial-temporal road-graph
feasibility check that rejects physically impossible matches. Effort goes here
first. Do not let dashboard polish eat Module C's hours.

## Hardware reality — read this before suggesting anything ML

This machine has **no NVIDIA GPU**.

- AMD Ryzen AI 7 350, 8 cores / 16 threads (Zen 5)
- Radeon 860M integrated GPU, **ROCm not installed** (unofficial support for
  this chip; deliberately not attempted)
- XDNA2 NPU present at `/dev/accel/accel0`, `amdxdna` loaded — **out of scope**,
  the Linux Ryzen AI stack is immature and it is inference-only
- 30 GB RAM, ~76 GB free on `/home`

So: **all inference is CPU** — the pipeline, the demo, everything that runs
here. Never propose `device=0`, `--half`, TensorRT, or a CUDA-only library for
runtime code. **Training is the one exception: it runs on Kaggle GPU**, which is
why `ml/train_plate.py` defaults to GPU settings and hides the CPU path behind
`--cpu`.

## Train exactly one model — on Kaggle

| Model | Decision |
|---|---|
| Plate detector (YOLOv8n) | **Fine-tune on Kaggle GPU.** COCO has no plate class. |
| Vehicle detector | Stock `yolov8n.pt`. COCO already has car/bus/truck/motorcycle. |
| Re-ID (OSNet) | Pretrained `osnet_x1_0` weights. CPU training is days. |
| OCR (PaddleOCR) | Pretrained + regex/state-code correction (`O↔0`, `I↔1`, `B↔8`). |

**Training runs on Kaggle, not on this machine.** Free T4 x2 / P100, ~20-40
s/epoch, so 50 epochs is ~20-35 min end to end. Use `ml/kaggle_train.ipynb`.

The local CPU fallback still exists behind `--cpu` (~6-15 min/epoch, 5-13 h for
the same job, 15-25x slower). Use it only if Kaggle is unavailable — do not
suggest it as the default.

**Inference is still CPU.** Only training moved; the demo runs on this laptop.
That is why `ml/export_onnx.py` targets OpenVINO int8, and why the 5 FPS /
sparse-OCR / Re-ID-on-exit budget below still applies in full.

Develop the pipeline against stock `yolov8n.pt` + a stub plate region anyway.
Trained weights are a one-line swap at a single call site, and that decoupling
is worth keeping even now that training is fast.

## Ownership

| | Dev A (repo owner) | Dev B |
|---|---|---|
| Owns | `ml/`, `backend/`, `db/`, Docker | `frontend/`, demo assets |
| Scope | ANPR pipeline, ByteTrack + Re-ID, cross-camera engine, analytics, FastAPI + WebSocket, Postgres/TimescaleDB, Redis, Celery | Camera grid, deck.gl city map, analytics charts, vehicle search, alerts panel, camera network graph data, demo video, deck |

`backend/` and `frontend/` are disjoint trees — merge conflicts should be near
zero. The seam between them is **`API-Contract.md` in the wiki**, frozen in
hour 0-2. Dev B builds against `/api/mock/*` fixtures and never blocks on the
backend. **If you change a response shape, it is a contract change — say so.**

## Stack

React 18 + deck.gl + MapLibre GL + Recharts · FastAPI + WebSocket · Postgres 16
+ TimescaleDB · Redis · Celery · Docker Compose · YOLOv8 (Ultralytics) +
ByteTrack + OSNet (torchreid) + PaddleOCR + OpenCV · OpenVINO for CPU inference.

Docker was **not installed** on this machine as of setup — see WORKFLOW.md
Stage 0.

## Commands

```bash
docker compose up -d db redis          # infra only, the usual dev mode
docker compose up -d                   # whole stack
uvicorn backend.mock:app --reload      # mock API (works today)
uvicorn backend.main:app --reload      # real backend
cd frontend && npm run dev             # frontend dev

# training: Kaggle, via ml/kaggle_train.ipynb (GPU defaults, no flags needed)
python ml/train_plate.py --epochs 50            # on Kaggle
python ml/train_plate.py --cpu --epochs 1       # local fallback only

python ml/export_onnx.py --weights runs/detect/plate/weights/best.pt

# mock API — unblocks the frontend. Implemented and tested.
uvicorn backend.mock:app --reload --port 8000
python backend/mock.py                          # fixture selfcheck
```

## Conventions

- Datasets, weights, `runs/` are gitignored. Never commit a `.pt`.
- `db/schema.sql` is a shared contract — announce changes to both devs.
- Video is processed at **5 FPS, not 30**. ANPR does not need 30.
- Plate detection + OCR run only on a track's first few frames and its best
  crop; Re-ID embedding runs **only on track exit**. Do not make these
  per-frame — that is what blows the CPU budget.
- Deliberate shortcuts with a known ceiling get a `ponytail:` comment naming
  the ceiling and the upgrade path.

## Non-negotiable

Record the backup demo video by hour 34. A live demo failure with no fallback
loses the whole thing.
