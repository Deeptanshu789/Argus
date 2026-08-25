# Argus — Claude Code context

City-wide ANPR + cross-camera vehicle trajectory tracking + traffic analytics.
Smart India Hackathon problem statement **SIH26127**, org **Bharat Electronics
Limited (BEL)**. 36-hour build, 2 people.

Full docs: <https://github.com/Deeptanshu789/Argus/wiki>
Runbook: `WORKFLOW.md`. Original strategy doc: `sih26127_implementation_plan.md`.

## The stack is TypeScript. Python is quarantined.

**Next.js 15 (App Router) + TypeScript for everything.** Python survives in
exactly two places and nowhere else:

| Python lives here | Why it cannot be TS |
|---|---|
| `ml/train_plate.py`, `ml/kaggle_train.ipynb` | Training. Runs on Kaggle, never in the app. |
| `ml/sidecar.py` | Inference. YOLO + PaddleOCR have no Node equivalents. |

`ml/sidecar.py` runs one process per camera and emits **newline-delimited JSON**
on stdout. `worker/ingest.ts` parses it. That pipe is the only place Python and
TypeScript meet.

**Do not add Python anywhere else.** No Flask, no FastAPI, no Celery. If you are
about to write a `.py` file outside `ml/`, you are solving the wrong problem.

## Architecture

```
ml/sidecar.py (one per camera)   ← the only runtime Python
  decode @5fps → YOLOv8 (OpenVINO int8) → BoT-SORT (track + ReID) → PaddleOCR
  stdout: {"event":"detection"|"track_closed"|"ready"|"error", ...}
                    │  newline-delimited JSON
worker/ingest.ts    ▼   spawns + restarts sidecars, validates against contract
  → Postgres/TimescaleDB (src/server/db.ts), → Redis pub/sub (src/server/bus.ts)
                    │
server.ts           ▼   ONE process: Next UI + /api handlers + /ws upgrade
  src/server/association.ts   Module C ★ (cross-camera, 3-layer)
  src/server/analytics.ts     Module D
  src/app/(dashboard)         Module E — four views (deck.gl, MapLibre, Recharts)
```

`src/contract.ts` holds zod schemas for every REST response, every WebSocket
message, and the sidecar's JSON events. **Types are inferred from it**, so the
mock, the real routes, the worker and the UI cannot drift apart without a type
error. Changing a schema is a contract change — announce it.

## What actually differentiates this project

Any team can ship single-camera ANPR. What wins is **Module C**
(`src/server/association.ts`): matching the same vehicle across cameras via
three layers — exact plate text (60-70% of cases), OSNet Re-ID embedding cosine
similarity > 0.75 (20-25%), and a road-graph travel-time feasibility check that
rejects physically impossible matches. **Confirm only when 2 of 3 agree.**

Effort goes here first. Do not let dashboard polish eat Module C's hours. Note
that it is plain TypeScript — cosine similarity over 512 floats, and a graph
lookup. It is business logic, not ML.

## Hardware — read before suggesting anything ML

**No NVIDIA GPU.** AMD Ryzen AI 7 350 (8c/16t, Zen 5), Radeon 860M iGPU with
ROCm **not installed**, XDNA2 NPU at `/dev/accel/accel0` (out of scope — Linux
stack immature, inference-only), 30 GB RAM.

- **All inference is CPU.** Never propose `device=0`, `--half`, TensorRT, or a
  CUDA-only library for runtime code.
- **Training is the one exception** — it runs on Kaggle GPU. That is why
  `ml/train_plate.py` defaults to GPU settings and hides the CPU path behind
  `--cpu`.

## Train exactly one model — on Kaggle

### Measured, on 169 held-out Indian plate photos the model never trained on

| Stage | Result |
|---|---|
| Detector mAP50 / precision / recall | **0.928 / 0.957 / 0.912** |
| Plate box found | 161/169 (95%) |
| Read and validated | 105/161 (65% yield) |
| **End to end** | **105/169 (62%)** |

`npm run` equivalent: `python ml/validate_plate.py --model
runs/detect/plate/weights/best.pt --data ~/indian-plates --ocr`.

**The detector is not the bottleneck — OCR is.** It finds a plate in 95% of
photos; a third of those still fail to read. Three fixes took end-to-end from
50% to 62% with no retraining at all:

1. `correct_plate()` accepted only `AA DD A(A) DDDD`. `DL9CAU4743` is a real
   Delhi plate (district code is digit-then-letter) and `MH05DK101` has a
   three-digit series. Both were thrown away.
2. PaddleOCR returns each line of a **two-line plate** separately, and the
   reader took only the longest — half a plate, rejected as too short. 34 of
   84 failures.
3. Two preprocessing variants fail on *different* crops, so a retry beats any
   single better filter: 58% and 55% alone, 65% combined.

Upscaling harder makes it **worse** (48 px 58%, 160 px 48%): PaddleOCR resizes
internally and an interpolated blur destroys the edges it needs.

Yield is an upper bound on accuracy — nothing checks a read against the true
plate, so a confident wrong answer counts as a success. Writing down the real
plate for fifty images is half an hour and is required before quoting a number.

| Model | Decision |
|---|---|
| Plate detector (YOLOv8n) | **Fine-tune on Kaggle GPU.** COCO has no plate class. |
| Vehicle detector | Stock `yolov8n.pt`. COCO already has car/bus/truck/motorcycle. |
| Re-ID (OSNet) | Pretrained `osnet_x1_0`. Training it is days, payoff small. |
| OCR (PaddleOCR) | Pretrained + `correct_plate()` in `ml/sidecar.py`. |

Kaggle T4: ~20-40 s/epoch, so 50 epochs is ~20-35 min. Local `--cpu` fallback is
~6-15 min/epoch, 15-25x slower — only if Kaggle is unavailable.

**The shipped weights used 3,400 of the 8,823 available images** (`SUBSET=3000`,
`VAL=400` in `ml/kaggle_train.ipynb`), 50 epochs at `imgsz=640`. So no, the
dataset is not exhausted. Retraining on all of it is one Kaggle hour and would
plausibly move mAP50 from 0.928 to ~0.94 — which is **+3 images out of 169**,
against the 56 that the detector finds and OCR cannot read. Do it only after
OCR yield stops being the limit.

Develop the sidecar against stock `yolov8n.pt` + a stub plate region; trained
weights are a one-line swap at a single call site.

## CPU inference budget — respect it or the demo drops frames

- Process video at **5 FPS, not 30**.
- Vehicle detection + BoT-SORT: every processed frame.
- Plate detection + OCR: **a track's first few frames and its best crop only.**
- Re-ID embedding: **on track exit only** — the sole moment Module C needs it.

4 streams x 5 FPS = 20 detections/sec. Quantized YOLOv8n at 480px on 16 Zen 5
threads clears that. Per-frame OCR or Re-ID does not. If FPS misses, **cut
`imgsz` before cutting features.**

### 5 FPS has a floor, and it is the tracker

Association matches a track to a detection by box overlap between *consecutive
processed frames*, and `fuse_score` folds the detection score into the cost. A
vehicle that moves too far between frames is never matched, so a **new track
starts every frame and none is ever confirmed** — `boxes.id` stays `None` and
the sidecar emits nothing at all. Measured:

| box width | movement per processed frame | IoU | tracked frames |
|---|---|---|---|
| 107 px | 60 px | 0.28 | **0 of 16** |
| 256 px | 28 px | 0.65 | 117 of 123 |

Rule of thumb: **a vehicle must move less than about half its own box width per
processed frame.** If real footage fragments into many short tracks, raise
`--fps` before touching thresholds. Fragmentation is worse than it looks —
Module C computes one Re-ID embedding per track, so a vehicle split six ways
gives six weak embeddings instead of one good one.

## Ownership

| | Dev A | Dev B |
|---|---|---|
| Owns | `src/server/**`, `src/app/api/**`, `worker/**`, `ml/**`, `db/**`, `server.ts` | `src/app/(dashboard)/**`, `src/components/**`, `demo/**` |
| Scope | Sidecar pipeline, Module C, analytics, REST + WebSocket, Drizzle/Postgres, Redis | Camera grid, deck.gl map, charts, vehicle search, alerts, camera graph data, demo video |

Shared, changed by neither alone: `src/contract.ts`, `db/schema.sql`.

Dev B builds against `/api/mock` (`src/server/mock.ts`) from minute one and
never blocks on the pipeline. **Changing a response shape is a contract change —
say so.**

## Commands

```bash
docker compose up -d db redis   # TimescaleDB + Redis
npm run db:setup     # apply db/schema.sql (idempotent) + seed camera topology
npm run dev          # Next UI + /api + /ws on :3000  (custom server)
npm run worker       # ingest supervisor + Python sidecars
npm run check        # tsc --noEmit
npm run selfcheck    # mock fixtures + Module C + Module D
npm run smoke        # every endpoint over real HTTP, judged by the zod contract

# End-to-end with no video, no CV deps, no GPU: synthetic sidecars that emit
# a vehicle travelling CAM1 -> CAM3 -> CAM2 (third leg has no readable plate).
ARGUS_PYTHON=python3 ARGUS_CAMERAS='CAM1=demo,CAM3=demo' npm run worker

python ml/sidecar.py --selfcheck --camera X --source X   # plate-correction check
python ml/demo_detect.py --source photo.jpg --ocr        # eyeball the detector
python ml/make_demo_clips.py                # synthetic demo/cam*.mp4 test clips
python ml/train_plate.py --epochs 50        # on Kaggle
python ml/export_onnx.py --weights runs/detect/plate/weights/best.pt
```

Real video end to end, no Kaggle needed (uses the exported weights if present):

```bash
python ml/make_demo_clips.py
ARGUS_CAMERAS='CAM1=demo/cam1.mp4,CAM3=demo/cam3.mp4,CAM2=demo/cam2.mp4' npm run worker
```

`next dev` alone boots the UI but **not** `/ws` — always `npm run dev`.

## Python install — two traps

```bash
sudo dnf -y install python3.12          # NOT 3.13: paddlepaddle has no 3.13 wheels
python3.12 -m venv .venv
./.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
./.venv/bin/pip install -r ml/requirements.txt
```

The CPU torch index is not optional housekeeping. PyPI's default wheel bundles
~2.5 GB of CUDA runtime that cannot run on this machine; the CPU wheel is
~200 MB and is what the sidecar actually uses.

**There is no `torchreid`.** The PyPI package of that name is an unofficial fork
stuck at 0.2.5; the real deep-person-reid is a source build. Ultralytics
BoT-SORT (`ml/botsort.yaml`, `with_reid: True, model: auto`) gives tracking and
appearance embeddings from a dependency we need anyway.

Measured on this machine: YOLOv8n + BoT-SORT with ReID at `imgsz=480` runs
**~17 ms/frame single stream** on CPU (synthetic frame, so treat it as a ceiling
— real footage with many detections is slower). The budget is 20 inferences/sec
across 4 streams, so there is real headroom.

Embedding dimension is therefore **not fixed at 512** — measured, this build's
`model: auto` encoder emits **64** floats. Every camera must emit the same dimension; `worker/ingest.ts`
checks this and fails loudly, because a mismatch makes cosine similarity return
0 and layer 2 silently stop firing.

## Data layer

`src/server/db.ts` holds **every** SQL statement in the application. postgres.js
tagged templates, not an ORM: Timescale's hypertables and `time_bucket()` are
raw SQL either way, and an ORM schema file would be a second definition of
`db/schema.sql` free to drift from it — the exact failure `src/contract.ts`
exists to prevent on the API side.

Every statement in `db/schema.sql` is **idempotent**. The Postgres container
runs its init scripts only on an empty volume, so without that property a schema
change cannot be applied without destroying data. `npm run db:setup` is always
safe to re-run.

The API validates each response against its zod schema **on the way out**. A
parse costs microseconds and turns "the dashboard renders blank" into a named
field in the server log.

## Timestamps carry milliseconds

Detections arrive five per second. A whole-second timestamp makes a track's
first and last frame parse to the same instant, `estimateSpeed()` divides by a
zero interval and returns null, and **every speed in the system is silently
blank** with no error to explain it. `ml/sidecar.py:now_iso()` emits
milliseconds and `src/server/db.ts` keeps them. `analytics.selfcheck.ts` asserts
both halves.

## The sidecar's stdout is a protocol, not a log

`ml/sidecar.py` captures the real stdout handle at import and points
`sys.stdout` at stderr. Ultralytics prints "Loading … for OpenVINO inference" to
stdout and PaddleOCR prints its cache paths there; either line lands mid-stream
and the supervisor drops the JSON around it. Only `emit()` can reach the
protocol channel — so no dependency's verbosity setting can ever break the pipe.

## PaddleOCR needs `enable_mkldnn=False`

Not tuning. Paddle's oneDNN executor raises
`ConvertPirAttribute2RuntimeAttribute` on this CPU build for **every**
`predict()` call, and the constructor succeeds without a hint. Omit the flag and
OCR silently reads nothing forever.

## Conventions

- Datasets, weights, `runs/`, `node_modules/`, `.next/` are gitignored. Never
  commit a `.pt`.
- API paths are constructed only in `src/lib/api.ts`. Hardcode one elsewhere and
  the mock/live switch breaks.
- WebSocket clients switch on `type` and **ignore unknown types** — that is what
  lets the server add message types without breaking an older frontend.
- `null` means "not known yet", never `""` or `0`. A search miss is a **200 with
  empty arrays**, never a 404.
- Non-trivial logic leaves one runnable `assert`-based check, not a test suite.
- Deliberate shortcuts get a `ponytail:` comment naming the ceiling and the
  upgrade path.

## State of play

| Piece | Status |
|---|---|
| `src/contract.ts`, `src/server/mock.ts` | Done, selfchecked |
| `src/server/association.ts` — **Module C ★** | Done, selfchecked |
| `src/server/analytics.ts` — Module D | Done, selfchecked |
| `src/server/db.ts`, `src/server/bus.ts` | Done — all SQL, Redis pub/sub |
| Real `/api/*` routes | Done, contract-validated, smoke-tested |
| `worker/ingest.ts` | Done — writes, associates, alerts, rollup, publishes |
| `ml/sidecar.py` | Done — `run()` is the real decode/track/OCR/ReID loop |
| `test/smoke.ts` | 38 checks incl. a live end-to-end pipeline run |
| Trained plate weights | Done — mAP50 0.950 on held-out Indian plates |
| `src/app/(dashboard)` | Not started — Dev B |
| Real traffic footage | Not started. `ml/make_demo_clips.py` is a fixture, not the demo |

## Non-negotiable

Record the backup demo video by hour 34. A live demo failure with no fallback
loses the whole thing.
