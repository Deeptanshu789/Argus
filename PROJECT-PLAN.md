# Argus — Project Construction, Training Plan, and Wiki Publish

> **Superseded in one respect (2026-08-25):** training moved from strictly-local
> CPU to **Kaggle GPU**. The local CPU path survives as a fallback behind
> `ml/train_plate.py --cpu`. Everything else in this document still stands —
> notably that **inference is still CPU**, so the OpenVINO export, the 5 FPS
> budget, and the sparse OCR / Re-ID scheduling all still apply. Current
> instructions live in `WORKFLOW.md` and the wiki's Model-Training page.

## Context

`/home/deep/code/Argus` currently holds exactly one file: `sih26127_implementation_plan.md`, a 385-line strategy document for SIH problem statement 26127 (BEL — city-wide ANPR + cross-camera trajectory tracking + traffic analytics). It is a good *pitch* but not an executable project: it assumes a 6-person team, six AI co-pilots, and unspecified GPU hardware, and it contains no repo layout, no API contract, no runnable training procedure, and no assignment of work to real people.

Three things are missing and are the deliverable here:

1. **An executable project structure** — repo layout, the full stack made concrete (Docker Compose services, DB schema, run commands).
2. **A model-training plan that matches the actual machine.** This is the sharpest gap. The source doc assumes GPU training. The build machine has **no NVIDIA GPU**: AMD Ryzen AI 7 350 (16 threads, Zen 5), Radeon 860M integrated GPU, ROCm not installed, `amdxdna` NPU present at `/dev/accel/accel0` but with an immature Linux stack, 30 GB RAM, 76 GB free disk. User has chosen **strictly local CPU training**, so the plan must be built around that constraint honestly rather than restating GPU-era epoch counts.
3. **A two-person work split** for everything except model training, plus a `CLAUDE.md` local context file so future Claude Code sessions in this repo start with the project loaded.

The whole thing then gets published to the (currently empty, default) wiki at `https://github.com/Deeptanshu789/Argus/wiki`.

### Decisions already made by the user

| Question | Decision |
|---|---|
| Training compute | **Strictly local CPU only.** No cloud GPU, no ROCm install. |
| Stack | **Keep the full stack as written** — Postgres + TimescaleDB, Redis, Celery, Docker Compose. |
| People | **Dev A = the user** (model training + backend). **Dev B = one teammate** (frontend). |
| Split axis | **Backend vs Frontend**, with the REST/WebSocket API contract as the seam. |

> Docker is **not installed** on this machine (`docker --version` fails). Since the full stack was chosen, installing Docker Engine + Compose is now a hard prerequisite and is task #1 in the timeline.

---

## What gets built

### A. In the repo (`/home/deep/code/Argus`)

Deliberately small. The two devs scaffold their own halves in hour 0–2; this plan only creates what is needed *before* that, plus the training code that must start running on day one.

| File | Why it exists now |
|---|---|
| `CLAUDE.md` | Explicitly requested. Local context file loaded by every future Claude Code session in this repo. |
| `docker-compose.yml` | Postgres+TimescaleDB, Redis, backend, frontend, celery worker. Both devs need this before either can start. |
| `db/schema.sql` | Shared contract. Backend writes it, analytics reads it — must exist before either dev writes a query. |
| `ml/prepare_dataset.py` | Downloads/normalizes the Indian plate dataset to YOLO format, carves a CPU-sized subset. |
| `ml/train_plate.py` | The CPU training run. Started at hour 0, runs in background for the rest of day one. |
| `ml/export_onnx.py` | Exports trained weights to ONNX + INT8 for CPU inference. Determines whether the demo hits framerate. |
| `WORKFLOW.md` | Requested. The step-by-step operational runbook — what to run, in what order, by whom. Contents spelled out below and mirrored to the wiki. |
| `.gitignore` | Keeps datasets/weights/`runs/` out of git. |

Everything else — `backend/`, `frontend/` — is created by its owner at hour 0, not pre-scaffolded here.

### B. On the wiki (`Argus.wiki.git`, branch `master`)

Nine pages. Wiki repos are flat — page name = filename, `_Sidebar.md` drives navigation.

| Page | Contents |
|---|---|
| `Home.md` | Problem statement, what Argus is, hardware reality, quickstart, page index, risk table. |
| `Architecture.md` | Module A–E breakdown, service diagram, full tech stack table, Docker Compose services, repo layout, DB schema. |
| `Model-Training.md` | The CPU-only training plan (see below). Exact commands, local-vs-Kaggle time table, measured-first epoch budgeting, fallbacks. |
| `Workflow.md` | Mirror of the repo's `WORKFLOW.md` runbook — stage-by-stage commands. |
| `API-Contract.md` | The Dev A ↔ Dev B seam. Every REST endpoint, every WebSocket message shape, mock-data mode. |
| `Work-Split.md` | Dev A / Dev B ownership tables, files each owns, handoff points, merge policy. |
| `Build-Timeline.md` | The 36-hour schedule rewritten for 2 people (source doc assumed 6). |
| `Demo-Script.md` | The 4-act, 8-minute judge demo. |
| `_Sidebar.md` | Nav links. |

---

## The model training plan (CPU-only) — the core of this work

### Principle: train exactly one model, reuse the rest

The source doc implies training a plate detector, a Re-ID network, and tuning OCR. On CPU that is not affordable. Only one of the three actually needs local training:

| Model | Decision | Rationale |
|---|---|---|
| **Plate detector** (YOLOv8n) | **Fine-tune locally on CPU.** | COCO has no "license plate" class. This is the one model that genuinely does not exist off the shelf for Indian plates at the quality needed. |
| **Vehicle detector** | **Use stock `yolov8n.pt`.** No training. | COCO already has car/bus/truck/motorcycle. Training adds nothing. |
| **Re-ID** (OSNet) | **Use pretrained `osnet_x1_0` VeRi/Market weights.** No training. | Re-ID training on CPU is days, not hours. Pretrained embeddings are good enough for a cosine-similarity match with a 0.75 threshold, and Layer 1 (plate text) carries 60–70% of matches anyway. |
| **OCR** (PaddleOCR) | **Pretrained + regex/state-code post-correction.** No training. | The accuracy gap on Indian plates is closed far more cheaply by the `O↔0 / I↔1 / B↔8` correction pass and format validation than by fine-tuning a recognizer on CPU. |

Stretch only, if plate OCR accuracy is measurably the bottleneck after integration: fine-tune the small PaddleOCR **recognition** head (a CRNN, cheap relative to a detector) on synthetic Indian plate crops. Not on the critical path.

### CPU training configuration

Written for: 16 threads, 30 GB RAM, no GPU, ~1 usable overnight window.

```
model      yolov8n.pt          nano, not -s/-l — this is the single biggest CPU lever
imgsz      480                 not 640; plates are small but the crop pipeline compensates
batch      16
device     cpu
workers    8
freeze     10                  freeze the backbone — cuts backward-pass cost substantially
cache      ram                 30 GB is plenty for a ~3K-image subset; kills dataloader I/O
amp        False               no AMP on CPU
patience   15                  early-stop instead of burning the full epoch budget
```

Subset the dataset to **~3,000 train / 400 val images** rather than the full 15K. Plate detection is a single-class, geometrically simple task; the marginal value of images 3K–15K is far below the CPU cost.

### Measured-first epoch budgeting

Do not guess the epoch count. The procedure:

1. Run `epochs=1` and record the wall time.
2. Multiply by the hours available before the integration checkpoint.
3. Set `epochs` to fit that budget, with `patience=15` as the real stop condition.

Rough expectation to sanity-check against, **not** to rely on: ~6–12 min/epoch at these settings, so ~40–60 epochs in a 6–10 hour overnight run. If epoch 1 comes back at 25+ minutes, drop `imgsz` to 416 and the subset to 2,000 before committing to the run.

`mAP50 ≥ 0.85` on the val split is the go/no-go bar for plate detection. Single-class, tight-boxed plates reach this readily; if it stalls below 0.7, the dataset conversion is wrong, not the training.

### Training time estimate: local CPU vs Kaggle

Same job in both columns so the numbers are comparable: **YOLOv8n, single-class plate detector, ~3,000 train / 400 val images**.

| | **Local (Ryzen AI 7 350, CPU)** | **Kaggle (free T4 ×2 / P100)** |
|---|---|---|
| Config | `imgsz=480, batch=16, freeze=10, cache=ram, device=cpu, workers=8` | `imgsz=640, batch=32, AMP on, device=0` |
| Per epoch | **~6–15 min** | **~20–40 s** |
| 50 epochs | **~5–12.5 hours** | **~20–35 min** |
| Setup overhead | ~0 (data already local) | ~10–20 min (dataset upload + notebook) |
| **Total to weights** | **~5–13 hours** | **~30–55 min** |
| Speed ratio | 1× | **~15–25× faster** |
| Full 15K dataset, 100 epochs @ 640 | ~4–8 days — **not viable** | ~3–5 hours — viable |
| Constraints | Machine is thermally saturated and sluggish for the whole run | 12 h/session, 30 h/week quota; session can die and needs `resume=True` |

**Confidence in these numbers is moderate, not high.** Published CPU-training throughput for YOLOv8n varies by roughly 3× across reports, which is why the plan calls for a `epochs=1` calibration run before committing the epoch budget rather than trusting the table above. The Kaggle figures are firmer (T4 throughput is well characterized).

**Why local still wins here despite being 15–25× slower**, given the user's choice: the run is **unattended and overnight**. If it starts at hour 0, it costs ~0 developer hours, and it removes the dataset-upload step, the Kaggle session-death failure mode, and the dependency on external service availability during a timed hackathon. The 15× is wall-clock the team was going to spend sleeping anyway.

**The real cost of the local run is not time, it is the machine.** 16 threads pinned for 5–13 hours makes the laptop hot and slow for everything else — including Dev A's backend work. Mitigation, and this goes in the training script's defaults:

```
nice -n 10 python ml/train_plate.py --workers 6
```

Leave ~4 threads for interactive work. This lengthens the run maybe 20–30% and keeps the machine usable, which is the correct trade during a build.

**Escape hatch, documented but not planned:** if the `epochs=1` calibration comes back above ~25 min/epoch, local 50 epochs is a 20+ hour run and no longer fits. At that point the choice is (a) cut to `imgsz=416` + 2,000 images, or (b) move this one job to Kaggle for 30 minutes. The training script takes `--device` so nothing else in the repo changes.

### The scheduling insight that matters most

**Training is the long pole and it is unattended.** Kick off `ml/train_plate.py` in the background at **hour 0**, before writing a single line of backend code. It costs Dev A nothing to have it running. The single worst failure mode for this project is discovering at hour 20 that the plate detector still needs eight hours of CPU.

Until weights land, everything downstream runs against **stock `yolov8n.pt` vehicle boxes with a stub plate region** (bottom-center of the vehicle crop). The pipeline is developed and integrated end-to-end against the stub; the trained weights are a drop-in replacement at a single call site. This decouples the entire build from the training run.

### Inference: earning the framerate back on CPU

Training is the visible problem; **CPU inference is the one that decides whether the demo runs.** Plan:

1. Export both detectors to **OpenVINO IR** via `model.export(format="openvino", int8=True)`. OpenVINO runs on AMD x86 CPUs and typically gives 2–3× over stock PyTorch CPU. ONNX Runtime is the fallback if OpenVINO misbehaves.
2. Process video at **5 FPS**, not 30 — the source doc already identifies this, and it is correct. ANPR does not need 30 FPS.
3. Run the expensive stages **sparsely, not per-frame**:
   - Vehicle detection + ByteTrack: every processed frame.
   - Plate detection + OCR: only on a track's first N frames and on its best-quality crop — not every frame.
   - Re-ID embedding: only on track exit (that is the only moment cross-camera association needs it).
4. Budget: 4 streams × 5 FPS = 20 vehicle-detection inferences/sec. A quantized YOLOv8n at 480px on 16 Zen 5 threads clears this with headroom. OCR and Re-ID, being sparse, are amortized.

Set a hard `ponytail:` marker at the OpenVINO export site noting the ceiling (CPU int8, 5 FPS, 4 streams) and the upgrade path (the `amdxdna` NPU at `/dev/accel/accel0`, or any CUDA box at the venue).

### The NPU

The machine has an XDNA2 NPU and the `amdxdna` driver is loaded. It is **explicitly out of scope**: the Linux Ryzen AI stack is early, it is inference-only, and it would consume hours that the demo needs. Documented in the wiki as a known unexplored option, not a task.

---

## Work split: Dev A (user) / Dev B (teammate)

Model training is Dev A's alone, per the user. Non-training work splits Backend / Frontend, with the API contract as the seam.

**Dev A — ML + Backend**
- All model training, export, and the inference pipeline (Modules A, B)
- Cross-camera association engine (Module C) — the differentiator
- Analytics engine (Module D)
- FastAPI REST + WebSocket hub, Postgres/TimescaleDB schema and queries, Redis, Celery
- Docker Compose

**Dev B — Frontend + Demo**
- All four dashboard views (Module E): camera grid, deck.gl city map, analytics charts, vehicle search
- WebSocket client, state management
- Camera network graph definition (locations, road links, travel times) — feeds Dev A's Module C but is data authoring, not backend code
- Demo video sourcing/recording, backup recording, UI polish, presentation deck

**The seam.** `API-Contract.md` is written and frozen **first**, in hour 0–2, before either dev writes real code. Dev B builds the entire frontend against a `/api/mock/*` fixture endpoint that Dev A stubs immediately; the real pipeline is swapped behind the same shapes. Without this, a 2-person split serializes and the frontend blocks on the backend for 20 hours.

**Merge policy.** Two people, one `main`. Small, frequent commits; `backend/` and `frontend/` are disjoint trees so conflicts should be near-zero. Contract changes get announced, not silently pushed.

---

## Timeline reshaped for 2 people

The source doc's 36-hour plan allocates work across 6 members. Re-cut for 2, with training running unattended alongside:

| Hours | Dev A | Dev B |
|---|---|---|
| 0–2 | Install Docker; **kick off CPU training in background**; scaffold FastAPI + Compose + schema; freeze API contract; stub `/api/mock/*` | Scaffold React; consume the contract; camera grid shell against mock data |
| 2–6 | ANPR pipeline against stub weights; PaddleOCR + regex correction | deck.gl + MapLibre map shell; TripsLayer with fake trajectories |
| 6–10 | ByteTrack + OSNet embeddings; **integration checkpoint** | Camera grid overlay rendering; analytics chart shells |
| 10–16 | **Cross-camera association engine** (3-layer) | Camera network graph authoring; map trajectory animation wired to WS |
| 16–20 | Analytics engine; real WebSocket pipeline replaces mocks | Analytics view on live data; vehicle search UI |
| 20–24 | Swap in trained plate weights; OpenVINO export; measure FPS | Anomaly alert panel; dark theme, polish |
| 24–30 | Tune association thresholds on the 2-camera test; performance | Demo video prep; search + map interaction polish |
| 30–34 | End-to-end run on 4 streams; fix what breaks | **Record backup demo video** (non-negotiable) |
| 34–36 | Joint: presentation, Q&A rehearsal, edge-case notes | Joint |

Integration checkpoint at hour 10 is the same hard gate the source doc names, and it stays hard: camera feed → detection → OCR → track → embedding must work before anything else proceeds.

---

## `WORKFLOW.md` — the runbook to be written

This is the requested workflow file. It goes in the repo root and is mirrored as a wiki page. Its content, in order:

### Stage 0 — One-time setup (Dev A, ~30 min, hour 0)

```bash
# 1. Docker (Fedora) — required, not currently installed
sudo dnf -y install dnf-plugins-core
sudo dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo
sudo dnf -y install docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo systemctl enable --now docker && sudo usermod -aG docker $USER   # re-login after

# 2. Python env
python3 -m venv .venv && source .venv/bin/activate
pip install ultralytics paddleocr torchreid fastapi uvicorn[standard] \
            psycopg2-binary redis celery opencv-python openvino

# 3. Infra up
docker compose up -d db redis
psql "$DATABASE_URL" -f db/schema.sql
```

### Stage 1 — Start training FIRST, then everything else (Dev A, hour 0)

Order matters. This is the whole point of the workflow.

```bash
python ml/prepare_dataset.py --subset 3000 --val 400        # ~10 min
python ml/train_plate.py --epochs 1                          # CALIBRATION — read the time
```

Read the reported wall time for that one epoch, then:

- `< 15 min/epoch` → `epochs = floor(available_hours * 60 / minutes_per_epoch)`, capped at 60.
- `15–25 min/epoch` → drop to `--imgsz 416 --subset 2000`, re-calibrate.
- `> 25 min/epoch` after that → move this one job to Kaggle (`--device 0` in a notebook), take the weights back.

Then launch the real run in the background and **walk away from it**:

```bash
nohup nice -n 10 python ml/train_plate.py \
      --epochs <N> --workers 6 --patience 15 > ml/train.log 2>&1 &
```

Check on it with `tail -f ml/train.log`. Do not wait on it. Proceed to Stage 2 immediately.

### Stage 2 — Freeze the API contract (Dev A + Dev B together, hour 0–2)

Write `API-Contract.md`. Dev A stubs `/api/mock/*` returning fixture JSON in those exact shapes. Dev B is unblocked from this moment and never waits on the backend again.

**Nothing else starts until the contract is written.** In a 2-person split this is the single highest-leverage 90 minutes of the build.

### Stage 3 — Parallel build (hours 2–20)

Dev A and Dev B work their own trees per `Work-Split.md`. Dev A's pipeline runs against **stock `yolov8n.pt` + stub plate region**, not the training output. Integration checkpoint at hour 10 is a hard gate.

### Stage 4 — Swap in the trained model (Dev A, hour ~20, or whenever training finishes)

```bash
python ml/export_onnx.py --weights runs/detect/train/weights/best.pt   # OpenVINO int8
# change ONE path in the ANPR service config; restart backend
```

Then measure. `mAP50 ≥ 0.85` on val, and end-to-end FPS at 4 streams × 5 FPS. If FPS misses, cut `imgsz` before cutting features.

### Stage 5 — Integrate, harden, record (hours 20–36)

Real WebSocket replaces mocks. Association thresholds tuned on the 2-camera test. **Backup demo video recorded by hour 34** — non-negotiable, per the source doc's own risk table.

### Daily-loop commands (the short version)

| Want | Run |
|---|---|
| Bring stack up | `docker compose up -d` |
| Backend dev | `uvicorn backend.main:app --reload` |
| Frontend dev | `npm run dev` (in `frontend/`) |
| Check training | `tail -f ml/train.log` |
| Resume dead training | `python ml/train_plate.py --resume` |
| Run pipeline on a video | `python -m backend.pipeline --source demo/cam1.mp4 --camera CAM1` |

---

## Publishing to the wiki

> **Outward-facing action.** `Deeptanshu789/Argus` is a **public** repository. Everything pushed to its wiki becomes publicly visible on the internet immediately, and GitHub wiki content can be cached or indexed by third parties even if later deleted. The existing wiki has one page (the default `Home` containing only "Welcome to the Argus wiki!"), which this push will replace. I will ask for explicit confirmation before running `git push` on the wiki repo.

Procedure:

1. `git clone https://github.com/Deeptanshu789/Argus.wiki.git` into the scratchpad (not into the project directory — the wiki is a separate git repo and must not be nested inside it).
2. Write the nine pages.
3. `git add . && git commit && git push origin master`.

`gh auth status` confirms an authenticated `Deeptanshu789` account with HTTPS git protocol, so the credential helper should carry the push. If it prompts, the fallback is `gh auth setup-git`.

The repo itself (`CLAUDE.md`, `docker-compose.yml`, `ml/`, `db/`) is a **separate** commit to `main` on the main repo — note that `/home/deep/code/Argus` is **not currently a git repository** despite the remote existing, so it needs `git init` + remote add, or a fresh clone. I will confirm which before touching it.

---

## What is deliberately not being built

- **No cloud GPU path.** User chose local CPU; the plan commits to it rather than hedging.
- **No ROCm install.** Unofficial support for the 860M (Krackan Point) is a time sink with a real chance of ending in nothing.
- **No Re-ID or OCR training.** Pretrained weights, as argued above.
- **No `backend/` or `frontend/` scaffolding in this pass.** Each owner creates their own tree in hour 0. Pre-scaffolding another person's directory structure is the kind of speculative work that gets deleted.
- **No NPU work.** Documented, not attempted.

---

## Verification

**Repo files**
- `docker compose config` parses without error (after Docker install).
- `docker compose up -d db redis` brings both up; `psql` against the DB applies `db/schema.sql` cleanly.
- `python ml/prepare_dataset.py` produces a YOLO-format directory with matching image/label counts and a valid `data.yaml`.
- `python ml/train_plate.py --epochs 1` completes and prints wall-clock time — this is the calibration run that sets the real epoch budget, and it doubles as the smoke test that the whole training path works.
- `ml/export_onnx.py` on the resulting `best.pt` produces an OpenVINO IR that loads and runs one inference.

**CLAUDE.md**
- Start a fresh Claude Code session in `/home/deep/code/Argus` and confirm it can answer "what is this project and who owns the frontend" without reading anything else.

**Wiki**
- `curl -sI https://github.com/Deeptanshu789/Argus/wiki` returns 200 and each of the 9 pages resolves.
- Sidebar renders on every page; no broken internal links (wiki links are `[[Page-Name]]` by filename, so filename and link text must match exactly).
- Verify the pushed `Home` no longer shows the default "Welcome to the Argus wiki!" text.

**End-to-end (later, during the build — recorded here as the acceptance bar)**
- 4 demo videos → pipeline → dashboard shows live boxes, plate text, at least one confirmed cross-camera match with its trajectory drawn on the map, and a live congestion heatmap, sustained without dropped frames at 5 FPS.
