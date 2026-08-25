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
| `ml/sidecar.py` | Inference. YOLO + PaddleOCR + OSNet have no Node equivalents. |

`ml/sidecar.py` runs one process per camera and emits **newline-delimited JSON**
on stdout. `worker/ingest.ts` parses it. That pipe is the only place Python and
TypeScript meet.

**Do not add Python anywhere else.** No Flask, no FastAPI, no Celery. If you are
about to write a `.py` file outside `ml/`, you are solving the wrong problem.

## Architecture

```
ml/sidecar.py (one per camera)   ← the only runtime Python
  ffmpeg decode @5fps → YOLOv8 (OpenVINO int8) → ByteTrack → PaddleOCR → OSNet
  stdout: {"event":"detection"|"track_closed"|"ready"|"error", ...}
                    │  newline-delimited JSON
worker/ingest.ts    ▼   spawns + restarts sidecars, validates against contract
  → Postgres/TimescaleDB, → Redis pub/sub
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

| Model | Decision |
|---|---|
| Plate detector (YOLOv8n) | **Fine-tune on Kaggle GPU.** COCO has no plate class. |
| Vehicle detector | Stock `yolov8n.pt`. COCO already has car/bus/truck/motorcycle. |
| Re-ID (OSNet) | Pretrained `osnet_x1_0`. Training it is days, payoff small. |
| OCR (PaddleOCR) | Pretrained + `correct_plate()` in `ml/sidecar.py`. |

Kaggle T4: ~20-40 s/epoch, so 50 epochs is ~20-35 min. Local `--cpu` fallback is
~6-15 min/epoch, 15-25x slower — only if Kaggle is unavailable.

Develop the sidecar against stock `yolov8n.pt` + a stub plate region; trained
weights are a one-line swap at a single call site.

## CPU inference budget — respect it or the demo drops frames

- Process video at **5 FPS, not 30**.
- Vehicle detection + ByteTrack: every processed frame.
- Plate detection + OCR: **a track's first few frames and its best crop only.**
- Re-ID embedding: **on track exit only** — the sole moment Module C needs it.

4 streams x 5 FPS = 20 detections/sec. Quantized YOLOv8n at 480px on 16 Zen 5
threads clears that. Per-frame OCR or Re-ID does not. If FPS misses, **cut
`imgsz` before cutting features.**

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
npm run dev          # Next UI + /api + /ws on :3000  (custom server)
npm run worker       # ingest supervisor + Python sidecars
npm run check        # tsc --noEmit
npm run selfcheck    # mock fixtures + Module C + Module D
docker compose up -d db redis

# End-to-end with no video, no CV deps, no GPU: synthetic sidecars that emit
# a vehicle travelling CAM1 -> CAM3 -> CAM2 (third leg has no readable plate).
ARGUS_PYTHON=python3 ARGUS_CAMERAS='CAM1=demo,CAM3=demo' npm run worker

python ml/sidecar.py --selfcheck --camera X --source X   # plate-correction check
python ml/train_plate.py --epochs 50        # on Kaggle
python ml/export_onnx.py --weights runs/detect/plate/weights/best.pt
```

`next dev` alone boots the UI but **not** `/ws` — always `npm run dev`.

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
| `worker/ingest.ts` | Supervises sidecars, validates, associates. DB writes are TODO |
| `ml/sidecar.py` | Event contract, `correct_plate()`, `--source demo`. `run()` is TODO |
| `src/app/(dashboard)` | Not started — Dev B |
| Real `/api/*` routes, Drizzle layer | Not started — Dev A |
| Trained plate weights | Not started — needs a Kaggle run |

## Non-negotiable

Record the backup demo video by hour 34. A live demo failure with no fallback
loses the whole thing.
