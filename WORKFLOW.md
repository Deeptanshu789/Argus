# Argus — Workflow

The operational runbook: what to run, in what order, by whom. Planning and
rationale live in the [wiki](https://github.com/Deeptanshu789/Argus/wiki); this
file is the thing you keep open while building.

Two people:

- **Dev A** — server, ingest worker, CV sidecar, database
- **Dev B** — dashboard, visualization, demo assets

---

## Two rules that shape everything

**1. The contract is the boundary.** `src/contract.ts` holds zod schemas for
every REST response, every WebSocket message, and the sidecar's JSON events.
Types are inferred from it, so the mock, the real routes, the worker and the UI
cannot drift apart without a type error. It is already frozen and
`src/server/mock.ts` already serves it — the frontend is unblocked from minute
one. Changing a schema is a contract change: announce it.

**2. Python is quarantined to `ml/`.** Training (`train_plate.py`, on Kaggle) and
inference (`sidecar.py`, one process per camera, emitting JSON). Everything else
is TypeScript. If you are about to write a `.py` outside `ml/`, stop.

Training used to be the long pole; it is not any more. Kaggle GPU finishes the
plate detector in about half an hour. Develop the sidecar against stock
`yolov8n.pt` with a stub plate region and swap the weights in at a single call
site. Never block a developer on a training run.

---

## Stage 0 — Setup

```bash
git clone https://github.com/Deeptanshu789/Argus.git && cd Argus
npm install
cp .env.example .env.local

# infra
docker compose up -d db redis      # applies db/schema.sql on first run

# app: UI + /api + /ws, one process, port 3000
npm run dev
```

Verify: <http://localhost:3000> lists the mock routes, and
`curl localhost:3000/api/mock/cameras` returns four cameras.

Docker on Fedora, if not installed:

```bash
sudo dnf -y install dnf-plugins-core
sudo dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo
sudo dnf -y install docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo systemctl enable --now docker && sudo usermod -aG docker $USER   # re-login
```

Python, only for the sidecar (Dev A only, and only when starting Module A):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r ml/requirements.txt
```

---

## Stage 1 — Train on Kaggle

**Repo owner. ~45 min wall clock, mostly unattended. Not on either developer's
critical path.**

Training runs on Kaggle's free GPU (T4 x2 / P100), not on the build laptop —
the laptop has no CUDA device, and the same job there takes 5-13 hours instead
of half an hour.

### 1. Get a dataset

Nothing is auto-downloaded: dataset URLs and layouts rot, and a broken
downloader at hour 0 is the worst possible time to debug one. On Kaggle use
**Add Data**; Roboflow Universe exports YOLO format directly.

- <https://universe.roboflow.com/search?q=indian+number+plate>
- <https://www.kaggle.com/datasets/praveen12345/indian-number-plate-detection>
- <https://huggingface.co/datasets/Dataclusterlabspvtltd/indian-number-plates-dataset>

### 2. Run the notebook

Upload `ml/kaggle_train.ipynb`. Accelerator **GPU T4 x2**, Internet **On**,
dataset attached. Set `SRC` to `/kaggle/input/<slug>`, Run All.

The notebook clones this repo and calls `ml/prepare_dataset.py` and
`ml/train_plate.py` — no training code is duplicated, so the Kaggle path and the
local path cannot drift apart.

Expect **~20-40 s/epoch; 50 epochs is ~20-35 minutes.** GPU defaults are
`imgsz=640, batch=32, amp=True, freeze=0`. Note `freeze=0` — the backbone is
*not* frozen. Freezing it is a CPU concession that costs accuracy, and on a T4
there is no reason to pay it.

### 3. Bring the weights back

```bash
mkdir -p runs/detect/plate/weights
unzip argus-plate-weights.zip -d runs/detect/plate/weights
python ml/export_onnx.py --weights runs/detect/plate/weights/best.pt
```

**Go/no-go bar: `mAP50 >= 0.85` on val.** Below 0.7 means the dataset conversion
is wrong, not the training — check labels are class `0` and boxes normalized
`xywh`. A GPU run is cheap, so if the result disappoints, raise `SUBSET` toward
the full 15,000 images and re-run.

### Local fallback — only if Kaggle is unavailable

```bash
python ml/train_plate.py --cpu --epochs 1        # calibration — read the time
nohup nice -n 10 python ml/train_plate.py --cpu --epochs <N> > ml/train.log 2>&1 &
```

`--cpu` flips a whole preset (`imgsz=480`, frozen backbone, `workers=6` so the
laptop stays usable). Budget epochs from the **measured** minutes-per-epoch the
calibration run prints — published CPU throughput for YOLOv8n varies about 3x
across reports. The script prints the recommendation itself.

---

## Stage 2 — The contract is already frozen

**Done. Both developers start from here.**

```bash
npm run selfcheck     # validates every fixture against src/contract.ts
npm run check         # tsc --noEmit
```

`src/server/mock.ts` serves every shape from seeded fixtures, so reloads give
identical data — a chart that reshuffles on refresh makes it impossible to tell
a UI bug from new data.

The fixtures deliberately include the cases that break naive UIs:

- ~15% of tracks have `plate_text: null` — Re-ID-only matching is exactly what
  Act 2 of the demo shows off, so the UI must render it
- `GET /search` on an unknown plate returns **200 with empty arrays**, not 404
- one camera reports `status: "degraded"`
- hops carry `method` of `plate`, `reid`, or `spatial_temporal`, so the map can
  colour by which matching layer confirmed each leg
- `KA05MR7821` is a known-good plate search always finds — use it in rehearsal

The frontend reads `NEXT_PUBLIC_MOCK` through `src/lib/api.ts` and nothing else.
Switching the whole app to live data is one env var.

---

## Stage 3 — Parallel build

**Dev A** fills in `ml/sidecar.py:run()`, then `worker/ingest.ts:handle()`, then
`src/server/association.ts` (Module C), `src/server/analytics.ts` (Module D),
and the real `/api` route handlers.

**Dev B** builds the four dashboard views against `/api/mock`.

The sidecar develops against **stock `yolov8n.pt` + a stub plate region**, not
the training output, even though the weights arrive early now.

### Hour 10 integration checkpoint — a hard gate

```
video → sidecar JSON → worker → Postgres → /api → WebSocket → a box on screen
```

with plate text on most vehicles and a 512-dim embedding on every closed track.
If this does not run, **stop adding features and fix it.**

### CPU inference budget — respect it or the demo drops frames

- 5 FPS, not 30
- vehicle detection + ByteTrack: every processed frame
- plate detection + OCR: a track's first few frames and best crop only
- Re-ID embedding: on track exit only

---

## Stage 4 — Swap in the trained model

Change one path in the sidecar's config, restart the worker. Measure `mAP50` on
val and end-to-end FPS at 4 streams. **If FPS misses, cut `imgsz` before cutting
features.**

---

## Stage 5 — Integrate, harden, record

- Set `NEXT_PUBLIC_MOCK` off and `MOCK=0`; the real hub feeds `broadcast()`.
- Tune Module C thresholds on a 2-camera route test. Require **2 of 3 layers**
  to agree — that is the defence against false trajectories.
- **Record the backup demo video by hour 34. Non-negotiable.**

---

## Daily-loop commands

| Want | Run |
|---|---|
| App: UI + API + WebSocket | `npm run dev` |
| Ingest worker + sidecars | `npm run worker` |
| Typecheck | `npm run check` |
| Verify fixtures vs contract | `npm run selfcheck` |
| Infra only | `docker compose up -d db redis` |
| Whole stack | `docker compose up -d` |
| Plate-correction check | `python ml/sidecar.py --selfcheck --camera X --source X` |
| Dataset prep check | `python ml/prepare_dataset.py --selfcheck` |
| Re-apply DB schema | `psql "$DATABASE_URL" -f db/schema.sql` |
| One sidecar by hand | `python ml/sidecar.py --camera CAM1 --source demo/cam1.mp4` |
