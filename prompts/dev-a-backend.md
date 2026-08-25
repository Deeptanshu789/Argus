# Antigravity prompt — Dev A (Server, ingest, CV sidecar)

Copy everything below the line into Antigravity as your task. It is
self-contained.

---

You are working on **Argus**, a city-wide ANPR and cross-camera vehicle
trajectory tracking system for Smart India Hackathon problem statement SIH26127
(Bharat Electronics Limited). Repo: `https://github.com/Deeptanshu789/Argus`

## Setup

```bash
git clone https://github.com/Deeptanshu789/Argus.git && cd Argus
git checkout -b dev-a-server
npm install
cp .env.example .env.local
docker compose up -d db redis     # applies db/schema.sql on first run
npm run dev                       # UI + /api + /ws on :3000
```

Sanity check: `curl localhost:3000/api/mock/cameras` returns four cameras, and
`npm run selfcheck` passes.

Python venv, only when you start the sidecar:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r ml/requirements.txt
```

## What already exists — do not rebuild it

| Piece | State |
|---|---|
| `src/contract.ts` | Frozen. zod schemas, all types inferred from them. |
| `src/server/mock.ts` | Serves every route from seeded fixtures. |
| **`src/server/association.ts` — Module C ★** | **Written and selfchecked.** Do not rewrite it. |
| `src/server/analytics.ts` — Module D | Written and selfchecked. Do not rewrite it. |
| `worker/ingest.ts` | Supervises sidecars, validates output, calls the association engine. DB writes are marked `TODO(Dev A)`. |
| `ml/sidecar.py` | Event contract, `correct_plate()`, `--source demo`. `run()` is the stub you fill in. |

Watch the whole path run before you touch anything — no video, no models, no GPU:

```bash
ARGUS_PYTHON=python3 ARGUS_CAMERAS='CAM1=demo,CAM3=demo' npm run worker
# [MATCH] CAM1->CAM3 KA05MR7821 via plate conf 0.99 in 164s [plate+reid+spatial_temporal]
```

`npm run selfcheck` runs all three suites. Keep them green.

**Read these first, in this order, before writing any code:**

1. `src/contract.ts` — zod schemas for every response, message, and sidecar
   event. This is the specification.
2. `CLAUDE.md` — hardware constraints and conventions
3. `WORKFLOW.md` — the runbook
4. `db/schema.sql` — the tables you write into
5. `src/server/mock.ts` — a working reference for every endpoint's shape

## The stack is TypeScript. Python is quarantined.

Next.js 15 App Router + TypeScript. Python survives in exactly two places:

- `ml/train_plate.py` / `ml/kaggle_train.ipynb` — training, runs on Kaggle only
- `ml/sidecar.py` — inference, because YOLO/PaddleOCR/OSNet have no Node
  equivalents

**Do not add Python anywhere else.** No Flask, no FastAPI, no Celery. If you are
about to create a `.py` outside `ml/`, you are solving the wrong problem. Use
BullMQ on Redis for queues and `ws` for sockets — both are already dependencies.

## You own

`src/server/**`, `src/app/api/**`, `worker/**`, `ml/**`, `db/**`, `server.ts`.

**Do not touch `src/app/(dashboard)/**` or `src/components/**`.** Another
developer owns the UI and is working in parallel against the mock. If a view
needs different data, change the contract *and announce the change* — never
reach into their tree.

`src/contract.ts` and `db/schema.sql` are shared. Changing either is a contract
change: announce it, do not push a renamed field and let the other developer
find it via a type error.

## The hard constraint: no GPU at runtime

The demo machine is an AMD Ryzen AI 7 350 with **no CUDA device**. ROCm is not
installed. Training happens separately on Kaggle — weights arrive as a file.

Never use `device=0`, `--half`, TensorRT, or a CUDA-only library. Models load as
OpenVINO int8 via `ml/export_onnx.py` (2-3x over stock PyTorch CPU, and that
margin is what makes the demo hold framerate).

**Respect this inference budget or the demo drops frames:**

- Process video at **5 FPS, not 30**. ANPR does not need 30.
- Vehicle detection + tracking: every processed frame.
- Plate detection + OCR: **a track's first few frames and its best crop only.**
- Re-ID embedding: **on track exit only** — the sole moment cross-camera
  association needs it.

4 streams x 5 FPS = 20 detections/sec, which a quantized YOLOv8n at 480px clears
on 16 Zen 5 threads. Per-frame OCR or Re-ID does not. If you measure a miss,
**cut `imgsz` before cutting features.**

## Build order

Each step has an acceptance check. Do not move on until it passes.

### 1. `ml/sidecar.py` — implement `run()`

The file already defines the event contract, the plate-correction pass, and a
passing selfcheck. Fill in the pipeline:

- Decode with OpenCV at the target FPS (skip frames, do not decode all 30).
- Vehicle detection with **stock `yolov8n.pt`** — COCO already has car, bus,
  truck, motorcycle. Do not train anything for this.
- Plate detection: use a **stub** first — the bottom-center region of the vehicle
  crop. Trained plate weights drop in later at this one call site; keep the swap
  to a single line.
- CLAHE contrast enhancement + upscaling before OCR, for night and blur.
- PaddleOCR on the plate crop, then `correct_plate()` — already written and
  tested. **Do not skip the correction pass**; it is worth more than model
  fine-tuning on Indian plates.
- ByteTrack within the camera.
- OSNet (`torchreid`, pretrained `osnet_x1_0` — **do not train it**) for a
  512-dim embedding, on track exit.
- `emit()` `detection` and `track_closed` events. **Stdout is JSON only** —
  diagnostics go to stderr, or the supervisor logs a contract violation.

**Acceptance:** `python ml/sidecar.py --camera CAM1 --source demo/cam1.mp4`
prints valid JSON lines, one `track_closed` per completed track with a 512-float
embedding, and `python ml/sidecar.py --selfcheck --camera X --source X` passes.

### 2. `worker/ingest.ts` — fill in `handle()`

The supervisor, spawn/restart logic, and contract validation are done. Add:
insert `detections` and `tracks` rows via Drizzle, publish to Redis, and hand
closed tracks to the association engine.

**Acceptance:** running `npm run worker` against a demo video populates `tracks`
with plate text on most vehicles and an embedding on every closed track.

### 3. Persist Module C's output — the engine is already written

`src/server/association.ts` implements the three layers, the 2-of-3 rule, the
shortest-path travel-time check, and trajectory stitching. It is selfchecked.
**Read it, do not rewrite it.** Four decisions in there are load-bearing and
were arrived at by finding the bug:

- Layer 3 **vetoes** proven impossibility rather than being outvoted. Plain
  2-of-3 lets plate + appearance confirm a car covering 1.4 km in 30 s.
- A **missing route only abstains**, so real matches are not lost to incomplete
  topology data.
- Appearance confidence is **capped below** an exact plate read.
- `associateArrival` searches **backwards**, because the later camera's track
  closes last.

Your job here is persistence, not logic. In `worker/ingest.ts:handle()`:

- Replace the in-memory `recent` ring with
  `SELECT * FROM tracks WHERE exit_time > now() - interval '15 minutes'`.
- Replace the `LINKS` import from mock fixtures with `SELECT * FROM camera_links`.
- Insert `matches` rows from the returned `MatchResult`, extend `trajectories`,
  and publish `match` and `trajectory_update` messages.

Everything between those two lines — `associateArrival`, `toHop`, `stitch` —
stays exactly as it is.

**Acceptance:** given two demo videos of the same route as CAM1 and CAM3, a
`matches` row exists with the correct `method`, and a `trajectories` row
contains both track ids. `npm run selfcheck` still passes.

### 4. Persist Module D's output — also already written

`src/server/analytics.ts` gives you `bucketize`, `congestionScore`,
`estimateSpeed`, `headingDeg`, `detectStationary`, `detectWrongWay`, and
`detectVolumeSpike`. Selfchecked. Wire them:

- Write `analytics` and `alerts` rows.
- Schedule the periodic rollup with **BullMQ** — not a Python scheduler.
- **Measure `metersPerPixel` per camera** against a known landmark (a lane
  width, a crossing) and store it. It is a real calibration knob; no default can
  guess it, and every speed and congestion number depends on it.

**Acceptance:** `/api/analytics` returns 12 buckets of real data and at least one
anomaly fires on demo footage.

### 5. Real routes under `/api/*` and the live WebSocket

Mirror `src/app/api/mock/[...path]/route.ts` under `/api`, backed by Drizzle
queries. Every response is typed from `src/contract.ts`, so a drift is a compile
error, not a runtime surprise.

Then replace the canned loop in `server.ts` with the real hub: subscribe to
Redis and call `broadcast()`. Set `MOCK=0`.

**Acceptance:** the frontend removes `NEXT_PUBLIC_MOCK`, points at `/api`, and
every view still works with zero frontend changes. That is the whole test.

## Rules

- **Never commit weights, datasets, `runs/`, `node_modules/`, `.next/`** — all
  gitignored, keep it that way.
- Keep `npm run selfcheck` passing — all three suites. The mock is the
  dashboard's lifeline and the reference for what your real routes must return;
  the other two guard Module C and Module D.
- `npm run check` must pass before every push.
- Non-trivial logic (association scoring, congestion formula, plate regex) leaves
  **one runnable `assert`-based check**. No test frameworks, no fixtures — the
  smallest thing that fails if the logic breaks.
- Mark deliberate shortcuts with a `ponytail:` comment naming the ceiling and the
  upgrade path.
- Prefer the simplest thing that works. 36-hour build: no abstraction with one
  implementation, no config for a value that never changes.

## Merging

```bash
npm run check && npm run selfcheck
git add -A && git commit -m "feat(server): <what>"
git pull --rebase origin main
git push -u origin dev-a-server
```

Commit small and push often — a four-hour local branch is how you get a merge
conflict in a 36-hour build. Your files and the dashboard's are disjoint, so
conflicts should be near zero. Open a PR into `main` as each acceptance check
passes; do not wait until everything is done.
