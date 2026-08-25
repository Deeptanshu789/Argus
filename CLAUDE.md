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

So: **everything is CPU.** Never propose `device=0`, `--half`, AMP, or a
CUDA-only library. Training config is tuned for this in `ml/train_plate.py`.

## Train exactly one model

| Model | Decision |
|---|---|
| Plate detector (YOLOv8n) | **Fine-tune locally on CPU.** COCO has no plate class. |
| Vehicle detector | Stock `yolov8n.pt`. COCO already has car/bus/truck/motorcycle. |
| Re-ID (OSNet) | Pretrained `osnet_x1_0` weights. CPU training is days. |
| OCR (PaddleOCR) | Pretrained + regex/state-code correction (`O↔0`, `I↔1`, `B↔8`). |

Local CPU training is ~6-15 min/epoch on a 3K-image subset at 480px, so ~5-13 h
for 50 epochs. Kaggle T4 does the same job in ~30-55 min, ~15-25x faster — but
the local run is unattended and overnight, which is why it was chosen. Always
run `--epochs 1` first and budget from the **measured** time, not an estimate.

**Kick training off at hour 0 and develop against stock `yolov8n.pt` + a stub
plate region.** Trained weights are a one-line swap at a single call site. The
worst failure mode for this project is finding out at hour 20 that the detector
still needs eight hours of CPU.

## Ownership

| | Dev A (repo owner) | Dev B |
|---|---|---|
| Owns | ML training, `ml/`, `backend/`, `db/`, Docker | `frontend/`, demo assets |
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
uvicorn backend.main:app --reload      # backend dev
cd frontend && npm run dev             # frontend dev

python ml/prepare_dataset.py --src <downloaded-dataset> --subset 3000 --val 400
python ml/train_plate.py --epochs 1    # CALIBRATION — always first
tail -f ml/train.log                   # check a background run
python ml/export_onnx.py --weights runs/detect/plate/weights/best.pt
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
