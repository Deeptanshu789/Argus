# Antigravity prompt — Dev A (Backend + CV pipeline)

Copy everything below the line into Antigravity as your task. It is
self-contained.

---

You are working on **Argus**, a city-wide ANPR and cross-camera vehicle
trajectory tracking system for Smart India Hackathon problem statement SIH26127
(Bharat Electronics Limited). Repo: `https://github.com/Deeptanshu789/Argus`

## Setup

```bash
git clone https://github.com/Deeptanshu789/Argus.git && cd Argus
git checkout -b dev-a-backend
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
pip install ultralytics paddleocr torchreid opencv-python openvino \
            psycopg2-binary redis celery
docker compose up -d db redis     # applies db/schema.sql on first run
```

**Read these first, in this order, before writing any code:**

1. `CLAUDE.md` — hardware constraints and project conventions
2. `WORKFLOW.md` — the runbook
3. The wiki's **API-Contract** page — the exact shapes you must serve
4. `db/schema.sql` — the tables you write into
5. `backend/mock.py` — a working reference implementation of every endpoint

## You own

`backend/`, `ml/`, `db/`, `docker-compose.yml`.

**Do not touch `frontend/`.** Another developer owns it and is working in
parallel against the mock API. If a view needs different data, change the API
*and announce the shape change* — never reach into their tree.

## The hard constraint: no GPU at runtime

The demo machine is an AMD Ryzen AI 7 350 with **no CUDA device**. ROCm is not
installed. Model *training* happens separately on Kaggle and is not your
problem — trained weights will be handed to you as a file.

Everything you write runs on CPU. Never use `device=0`, `--half`, TensorRT, or
any CUDA-only library. Export models with `ml/export_onnx.py`, which produces
OpenVINO int8 (2-3x over stock PyTorch CPU, and that margin is what makes the
demo hold framerate).

**Respect this inference budget or the demo drops frames:**

- Process video at **5 FPS, not 30**. ANPR does not need 30.
- Vehicle detection + tracking: every processed frame.
- Plate detection + OCR: **a track's first few frames and its best crop only.**
- Re-ID embedding: **on track exit only** — that is the sole moment cross-camera
  association needs it.

4 streams x 5 FPS = 20 vehicle-detection inferences/sec. A quantized YOLOv8n at
480px on 16 Zen 5 threads clears that with headroom. Making OCR or Re-ID
per-frame does not. If you measure a miss, **cut `imgsz` before cutting
features.**

## Build order

Work in this sequence. Each step has an acceptance check — do not move on until
it passes.

### 1. `backend/pipeline.py` — Modules A and B

Single-camera pipeline: video file or RTSP in, tracked vehicles out.

- Vehicle detection with **stock `yolov8n.pt`** (COCO already has car, bus,
  truck, motorcycle — do not train anything for this).
- Plate detection: use a **stub** at first — the bottom-center region of the
  vehicle crop. Trained plate weights drop in later at this one call site.
  Keep that swap to a single line.
- OCR with PaddleOCR on the plate crop.
- Post-process: validate `XX 00 XX 0000`, check the state code, and correct the
  three OCR confusions that cause most errors: `O`↔`0`, `I`↔`1`, `B`↔`8`.
  **This regex pass is worth more than model fine-tuning** — do not skip it.
- CLAHE contrast enhancement + upscaling before OCR for night and blurry plates.
- ByteTrack for frame-to-frame tracking within one camera.
- OSNet (`torchreid`, pretrained `osnet_x1_0` — **do not train it**) for a
  512-dim appearance embedding, computed on track exit.
- Write `tracks` and `detections` rows per `db/schema.sql`.

**Acceptance:** `python -m backend.pipeline --source demo/cam1.mp4 --camera CAM1`
produces rows in `tracks` with plate text on most vehicles and a 512-dim
embedding on every completed track.

### 2. `backend/association.py` — Module C ★ THE DIFFERENTIATOR

**This is where 40% of the project's value sits. Protect these hours. Any team
can ship single-camera ANPR; this is what wins.**

Match the same vehicle across different cameras, in three layers:

- **Layer 1 — plate text.** Exact match, both sides confident → confidence 0.99.
  Covers 60-70% of cases.
- **Layer 2 — appearance.** Cosine similarity of the 512-dim OSNet embeddings
  > 0.75, with colour-histogram similarity as a secondary signal. Covers 20-25%
  — the occluded and unreadable plates.
- **Layer 3 — spatial-temporal feasibility.** The `camera_links` table gives an
  expected travel time between any two cameras. Accept only if
  `0.5 x expected <= actual <= 2.0 x expected`. This rejects the physically
  impossible and disambiguates Layer 2.

Combined score when Layer 1 does not fire:
`0.6 * reid_sim + 0.2 * colour_sim + 0.2 * time_score`.

**Confirm a match only when 2 of the 3 layers agree.** This rule is the defence
against false trajectories, which are the single most damaging thing a judge can
see. A beautiful dashboard showing a wrong trajectory is worse than a plain one
showing a right one.

Write `matches` rows with the `method` that fired (`plate` / `reid` /
`spatial_temporal`) and stitch confirmed chains into `trajectories`.

**Acceptance:** given two demo videos of the same route treated as CAM1 and
CAM3, at least one vehicle is correctly matched across them, `matches` records
the method, and a `trajectories` row contains both track ids. Write a small
`assert`-based check that a deliberately infeasible pair (10 km apart, 30 s
gap) is **rejected**.

### 3. `backend/analytics.py` — Module D

Per-camera, per-time-bucket: vehicle count, type breakdown, speed estimate,
congestion score (density x inverse speed, normalized 0-100). Anomalies:
stationary > 5 min, wrong-way travel, volume spike. Write `analytics` and
`alerts` rows.

**Acceptance:** `GET /api/analytics` returns 12 buckets of real data and at
least one anomaly fires on demo footage.

### 4. `backend/main.py` — the real API

Serve every route from the API-Contract page at `/api/*`, backed by the
database. **`backend/mock.py` already implements all of them against fixtures —
match its shapes exactly.** Diff your responses against the mock's if unsure.

Then the WebSocket hub at `/ws`, emitting `detection`, `match`,
`trajectory_update`, `analytics`, and `alert` messages. Redis pub/sub between
the stream workers and the hub; Celery (`backend/tasks.py`) for per-camera
stream processing.

**Acceptance:** the frontend flips `VITE_MOCK` off, points at `/api`, and every
view still works with zero frontend changes. That is the whole test.

### 5. Swap in the trained weights

When the Kaggle-trained plate detector arrives:

```bash
mkdir -p runs/detect/plate/weights
unzip argus-plate-weights.zip -d runs/detect/plate/weights
python ml/export_onnx.py --weights runs/detect/plate/weights/best.pt
```

Change the one path in the ANPR config. Then measure `mAP50` and end-to-end FPS
at 4 streams.

## Rules

- **Never commit weights, datasets, or `runs/`** — they are gitignored, keep it
  that way.
- `db/schema.sql` and the API contract are shared. Announce changes; do not push
  a renamed field and let the other developer find it via a runtime error.
- Keep `backend/mock.py` working. It is the frontend's lifeline and the
  reference for what your real routes must return.
- Non-trivial logic (the association scoring, the plate regex, the congestion
  formula) leaves **one runnable `assert`-based check** behind. No test
  frameworks, no fixtures — the smallest thing that fails if the logic breaks.
- Mark deliberate shortcuts with a `ponytail:` comment naming the ceiling and
  the upgrade path.
- Prefer the simplest thing that works. This is a 36-hour build: no abstractions
  with one implementation, no config for a value that never changes.

## Merging

```bash
git add -A && git commit -m "feat(backend): <what>"
git pull --rebase origin main
git push -u origin dev-a-backend
```

Commit small and push often — a four-hour local branch is how you get a merge
conflict in a 36-hour build. `backend/`, `ml/`, `db/` and `frontend/` are
disjoint trees, so conflicts should be near zero. Open a PR into `main` when a
step's acceptance check passes; do not wait until everything is done.
