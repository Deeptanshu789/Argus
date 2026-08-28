# Argus — Claude Code context

City-wide ANPR + cross-camera vehicle trajectory tracking + traffic analytics.
Smart India Hackathon problem statement **SIH26127**, org **Bharat Electronics
Limited (BEL)**. 36-hour build, 2 people.

Full docs: <https://github.com/Deeptanshu789/Argus/wiki>
Runbook: `WORKFLOW.md`. VPS deployment: `DEPLOY.md`.
Original strategy doc: `sih26127_implementation_plan.md`.

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
  → Postgres/TimescaleDB (src/server/db.ts), → LISTEN/NOTIFY (src/server/bus.ts)
                    │
server.ts           ▼   ONE process: Next UI + /api handlers + /ws upgrade
  src/server/association.ts   Module C ★ (cross-camera, 3-layer)
  src/server/analytics.ts     Module D
  src/app/(dashboard)         Module E — six views (deck.gl, MapLibre, Recharts)
  src/server/frames.ts        phone JPEG in, MJPEG out for the sidecar
```

### A phone is just another camera

`/devices` issues a six-character pairing code, and the code can be satisfied
two ways because `getUserMedia` only works in a **secure context**:

| route | how | needs |
|---|---|---|
| `browser` | the phone opens `/cam/<code>`, captures at 5 fps and 640px, pushes JPEG over `/ws/cam` | HTTPS, or localhost |
| `url` | an IP-camera app on the phone serves RTSP/MJPEG; the code records the URL | nothing |

This laptop's own webcam works with no certificate at all — `localhost` is
always a secure context. A phone at `http://<lan-ip>:3000` is not, and the
browser refuses the camera with no way for the page to ask again. Run
`./scripts/dev-https.sh` once and `server.ts` serves HTTPS on 3443 alongside
the usual port; the phone accepts the self-signed warning once, and that
warning IS the step that makes the origin secure.

`src/server/frames.ts` bridges the two halves: JPEG in over a WebSocket, MJPEG
out at `/cam-stream/<camera_id>`, which the sidecar opens like any network
camera — **`ml/sidecar.py` needed no change at all**. Latest frame wins, no
queue: a phone on a slow network should show the newest frame, not work through
a backlog, and that also bounds memory at one frame per device.

The MJPEG route is served from `server.ts` directly, before Next sees the
request. Next buffers and transforms responses in ways an endless multipart
stream does not survive.

A paired device is a **real live camera**: it appears on the dashboard, the map
and the analytics beside CAM1..CAM4. It carries no `camera_links`, so layer 3
abstains on it rather than inventing a road graph.

`ready` is emitted **after the capture opens**, not at startup. The supervisor
treats it as proof a phone is really serving video; emitted earlier, a device
whose app had been closed would report itself healthy forever. Device sidecars
back off 5s, 10s, 20s… to a minute on repeated failure, because a dead stream
otherwise reloads YOLO and PaddleOCR every five seconds.

### Uploaded video is just another camera

`/upload` takes video files off the operator's machine. Each file becomes a row
in `cameras` with `is_upload = true`, so detections, tracks, Module C and the
analytics rollup work on it with **no special case anywhere**. Only two things
know an upload happened:

- `candidateTracks()` will not compare a track across the upload boundary — an
  uploaded video matches only the other videos in its own upload, and a live
  camera only other live cameras. Without that, an operator's clip matches demo
  footage and their results page fills with cameras they never sent.
- `getCameras/getTracks/getTrajectories/getAnalytics/getAlerts` exclude
  `is_upload` cameras, so uploaded footage never moves the city's numbers.
  Asking for one of those cameras **by name** still works, which is how the
  upload's own page reads it.

The server never spawns a sidecar. It writes the file, inserts a `pending` row,
and the worker picks it up — decoding video inside the web process is exactly
what the two-process split exists to prevent.

Uploaded files are assumed to cover the **same period**, as two cameras at a
junction would, so the interval between sightings is measured from the footage.
The optional travel-time field only feeds `camera_links`; it is not a playback
offset.

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

### Measured against 45 hand-labelled plates, `ml/groundtruth_test50.csv`

| Stage | Result |
|---|---|
| Detector mAP50 / precision / recall | **0.928 / 0.957 / 0.912** |
| Plate box found | 44/45 |
| **Read correctly** | **39/45 (87%)** |
| Read wrongly | 2 (4%) |
| Not read | 4 (9%) |
| Precision when it answers | 95% |

```bash
python ml/score_plates.py --model runs/detect/plate/weights/best.pt --show
```

`ml/validate_plate.py` reports *yield* over 169 photos instead, which is an
upper bound, not accuracy: nothing there checks a read against the true plate,
so a confident wrong answer counts as a success. Quote the 87%.

**78% of that came before touching a model.** The reader was running two of
PaddleOCR's document models on every plate crop — an orientation classifier and
an unwarper, both meant for scanned pages. On a 40 px crop the classifier
sometimes decides the text runs vertically and hands the recogniser a crop
turned on its side. Switching them off took the photo set from 35 correct to 39
and the real video from 2 exact to 4, and made OCR four times faster. See
`make_reader()` in `ml/sidecar.py`; it is the ONLY place a reader is built, so
that configuration cannot drift between the sidecar and the five scripts that
measure it.

### Real footage is a different problem from the test set

Measured on a 848x478 phone clip of live traffic, against plates transcribed by
eye (`ml/groundtruth_kiit.csv`): it read **1 of 5** legible plates. Three causes,
all found with `ml/diagnose_video.py`, which reports where each stage loses a
plate rather than only the total:

1. `correct_plate()` could not repair the STATE code. At 40 px, `OD` reads as
   `00`, and `ALPHA_FOR` maps `0` to `O` and nothing else, so `OO` failed the
   state check and twenty-six agreeing reads of one vehicle were thrown away.
   The state code is the one part of a plate with a **closed set** of valid
   answers, so `_state_code()` now scores every plausible pair against
   `STATE_CODES` and takes the cheapest real one.
2. The OCR growth retry was **dead code**. `run()` updated `best_area` before
   calling `wants_ocr(area)`, so the test read `area > area * 1.4` and never
   fired. Every track was read only in its first three frames — which for an
   approaching vehicle are the frames where it is furthest away. It now compares
   against the size at the last attempt.
3. **62% of OCR attempts were spent on vehicles too small to read.** No plate
   was ever read from a vehicle under 124 px wide. `PLATE_MIN_VEHICLE_PX` skips
   those before any cost is paid, which is what makes a larger attempt budget
   affordable.

Result on the same clip: **1 of 5 to 5 of 5 found**, 2 exactly right and 3 one
character out. Fixing the reader afterwards (see above) took it to **4 of 5
exactly right**. The one remaining error is `O` read for `D` in the SERIES
letters, which have no closed set to check against — the plate-specific OCR
model in `ml/TRAINING.md` is the real fix, and guessing there would corrupt
genuine `O` series.

**The detector is not the bottleneck — OCR is.** It finds a plate in 95% of
photos, and five of the six remaining failures are the read, not the box. Three
early fixes took end-to-end yield from 50% to 62% with no retraining at all,
and they are still the shape of every gain since:

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

Kaggle T4 at 2,573 images: ~15 s/epoch, so 60 epochs is ~15 min. Local `--cpu`
measured 4.2 min/epoch at 8,023 images, so roughly 1 min/epoch here — about an
hour for 60.

**The shipped weights trained on 1,365 of 8,823 images.** Each split zip carried
its own `_annotations.coco.json` at the archive root, so unzipping them into one
folder left only the last split's labels. The warning scrolled past in the
Kaggle log and training continued.

Retraining on all 8,023 fixed the mAP — 0.928 to **0.991** — and made the system
**worse**: 33 of 45 read correctly instead of 35, with wrong reads doubling. A
better-fitted detector crops tighter, and the last character falls outside the
box. `PLATE_PAD` in `ml/sidecar.py` and `score_plates.py --pad` exist for this.

That conclusion was half wrong and the correction matters. Re-measured with the
reader fixed, the two detectors score **identically** — 39 correct, 2 wrong,
both. The gap was never the detector; it was the reader mangling a crop that the
tighter model happened to produce more often. **Judge a detector by `correct`,
never by mAP — and re-measure both sides whenever either the detector or the
reader changes.** A stale comparison is worse than none.

**Four detectors now read exactly 39 of 45**: 1,365 images at mAP 0.928, 8,023
at 0.991, a 9,785-image merged fine-tune at 0.982, and a 10,240-image fine-tune
(`plate-v2`) at 0.987. The detector finds a plate in 44 of the 45 photographs,
so a perfect one could add one plate. This test set has no resolution left to
judge a detector with, and the six remaining failures are all the reader. The
9,785 fine-tune also read one plate FEWER on the real video — the only test here
with motion and glare in it — so the shipped weights stay. **Do not train
another detector against `groundtruth_test50.csv`**; label more truth or train a
reader.

### Most of that training data is not Indian

`datasets/plates` — 8,823 of the 10,240 merged images, and the bulk of every
"full retrain" above — is a GENERIC licence-plate set. Its crops are British and
European: `DS08 PCZ`, `LT07 FDJ`, one carrying an `EST` euroband. Only 1,648
merged images are Indian, from the Quobotic set.

That is why mAP rose from 0.928 to 0.991 while real reading did not move at all:
those runs fitted European plates and were validated on European plates. For a
DETECTOR this is harmless-to-useful — a plate is a rectangle of high-contrast
text whatever country issued it — which is exactly why every one of the four
scores 39/45. For a READER it is worthless, and `correct_plate()` rejects all of
it, because it validates Indian registration grammar.

**Never quote a plate-dataset size without saying how much of it is Indian.**

### A third of the Indian set is mirrored

Roboflow's commonest augmentation is a horizontal flip, and the export ships the
flipped copies as ordinary images. Harmless to a detector, which still sees a
plate. Poison to a reader: the text runs backwards, PaddleOCR reads it anyway,
and `correct_plate()` accepts the result because a reversed plate is often still
a grammatical one — `TN21TA0492` was being written down as `TN21AT0492` with
nothing downstream able to tell. `ml/make_reader_crops.py` reads every crop both
ways round and keeps the higher-scoring orientation, which rejects the mirrors
and repairs them in one pass. 758 of 1,756 crops (43%) needed it.

## The reader — `ml/train_reader.py`

The detector finds a plate in 44 of 45 photographs and the system reads 39, so
every remaining failure but one is the READ. `ml/train_reader.py` trains a CRNN
with CTC loss to replace PaddleOCR: 2.11M parameters, 32x256 grayscale input,
37 classes. Small on purpose — it trains on this laptop's CPU and runs in
milliseconds against PaddleOCR's 163 ms.

`ARGUS_READER=<checkpoint>` switches the sidecar to it, and `ARGUS_PLATE_MODEL`
does the same for the detector. **Opt-in by path, never by the file merely
existing**: a reader that has not been scored against `ml/groundtruth_test50.csv`
must not be able to replace the one in service by turning up on disk. Changing
the default is an edit to `READER_WEIGHTS` in `ml/sidecar.py`, made once the
score exists. `make_reader()` remains the single construction site and falls
back to PaddleOCR if the checkpoint will not load. `ARGUS_READER=paddle` forces
that fallback.

### Two-line plates are split and laid side by side

CTC alignment is monotone: the model emits characters left to right and can
never go back. A two-line plate asks it to read the top row across the image and
then the bottom row across the same columns, which monotone alignment cannot
express at all. `to_strip()` cuts at the gap between the rows and concatenates.
The same transform runs at training and at inference — if they ever diverge the
model reads noise.

### Synthetic data does not transfer, and the fix is measured

Both public reader datasets are renders. Trained on them alone the CRNN reaches
83% on held-out synthetic and reads **0 of 45** real photographs — it learned a
font, not a plate. Its errors were not nonsense but structured: `AP39E1493` read
as `KA39E1493`, `DL9CAU4743` as `LD9CAA47431`. Digits right, state letters
wrong.

`ml/make_reader_crops.py` fixes it by building real training data the project
already had the raw material for: crop every hand-drawn plate box out of the
Indian detection set and label it with PaddleOCR. Pseudo-labelling, with an
honest ceiling — the reader cannot learn to beat PaddleOCR where PaddleOCR is
wrong — but the renders cannot teach real fonts, dirt, angles or lighting, and
these can.

| reader | correct of 45 | CER | precision |
|---|---|---|---|
| PaddleOCR | **39 (87%)** | 9.4% | 95% |
| CRNN, synthetic only | 0 (0%) | 82.6% | 0% |
| CRNN, +1,756 real crops, 8 epochs | 16 (36%) | 34.9% | 42% |
| CRNN, 35 epochs local, 76,150 samples (`runs/reader-ft`) | 18 (40%) | 31.3% | 42% |
| CRNN, 60 epochs Kaggle, 178,266 samples (`runs/reader-k12`) | 25 (56%) | — | 60% |

Re-measured against the YOLO11s detector, which finds 65% more plate boxes and
so hands every reader more crops — including more crops too poor to read:

| reader, with `plate-k12` | correct of 45 | precision | KIIT exact | KIIT spurious |
|---|---|---|---|---|
| PaddleOCR | **38 (84%)** | **95%** | **4/5** | **6** |
| `reader-k12` best | 25 (56%) | 60% | 1/5 | 113 |
| `reader-ft` best (epoch 31) | 19 (42%) | 46% | 2/5 | 126 |
| `reader-ft` last (epoch 35) | 19 (42%) | 44% | — | — |

`reader-ft` reads one plate fewer with the better detector than with the worse
one (19 against 18 is noise; its precision falls from 42% to 46% on a smaller
base of answers). The extra boxes are extra chances to invent, because it
answers all 154 of them. **A better detector cannot help a reader that has no
way to say "I cannot read this."** Both CRNNs return text on 100% of crops.

### The confidence floor — the fix was 20 lines, not another training run

`READER_MIN_CONF` in `ml/sidecar.py` drops a CRNN read whose mean
per-character confidence is below the floor, returning nothing rather than a
guess. `ml/sweep_floor.py` picks the value by running the real
`ml/score_plates.py` at each threshold, so the numbers cannot drift from the
ones quoted anywhere else. With `reader-k12` and the `plate-k12` detector:

| floor | correct | wrong | missed | precision |
|---|---|---|---|---|
| 0.00 | 25 | 17 | 3 | 60% |
| 0.90 | 25 | 12 | 8 | 68% |
| 0.95 | 28 | 6 | 11 | 82% |
| **0.99** | **29** | **3** | **13** | **91%** |
| 0.995 | 27 | 3 | 15 | 90% |
| 0.999 | 24 | 3 | 18 | 89% |

0.99 is the knee. Wrong reads fall from 17 to 3 and precision from 60% to 91%
— most of PaddleOCR's lead recovered with no retraining at all.

**Correct reads go UP, 25 to 29**, which a filter that only removes answers
should not be able to do. `_read_plate()` tries two preprocessing variants and
keeps the first that yields a plate, so a confident-but-wrong read on the first
variant used to win and end the search. Rejecting it lets the second variant
run. The floor is not only a filter; it un-blocks a retry that was already
there.

On the KIIT clip the effect is sharper still, because a 44-pixel median box is
mostly unreadable: **reads matching no legible plate go from 114 to zero**, and
exactly-right is unchanged at 1 of 5. Only 2 of 154 crops clear the floor.
That is the floor working, not failing — the reader was never reading those
crops, it was inventing plates for them.

PaddleOCR is still ahead (38 correct, 95%, and 4 of 5 on the video). The gap is
now reading ability rather than restraint, which is the gap more real crops
close.

### 42% of the reader's crops are Maharashtra, and it shows

`ml/state_prior.py` counts the state code of every crop the reader trains on:
**MH 2,811 (42.3%), DL 615 (9.3%) … OD 29 (0.44%)**. A 97x gap between the
commonest state and the one the KIIT test footage was filmed in.

A classifier falls back on its prior wherever the evidence is weak, so on a
crop it cannot read the CRNN does not merely prefer MH — it emits MH, and
`correct_plate()` accepts it because `MH02DK1434` is a real registration.
Measured on that clip, **42% of the reads matching no legible plate began with
MH**.

**The obvious inference-time fix does not work, and the measurement is here so
nobody spends the day re-deriving it.** `best_state()` rescores the two state
letters as `log p(first) + log p(second) - tau * log prior(code)` over the
closed set of real codes — textbook logit adjustment, and `ARGUS_STATE_PRIOR_TAU`
sweeps it:

| tau | KIIT spurious | of those, MH | KIIT exact | photos correct | precision |
|---|---|---|---|---|---|
| **0.0 (shipped)** | 114 | 48 (42%) | 1/5 | 25 | 60% |
| 0.4 | 126 | 45 (35%) | 1/5 | 25 | 58% |
| 1.0 | 127 | 36 (28%) | 1/5 | 25 | 58% |

It does exactly what it claims — MH's share of the junk falls 42% to 28% — and
it improves nothing. Not one extra plate is read correctly on either test set,
and precision drops two points. **The correction redistributes the bias instead
of removing it.** MH was the symptom; the disease is that the reader cannot
read a 44 px crop at all, and guessing `OD` on a crop it cannot see is a
different wrong answer, not a right one. `tau` therefore ships at 0.0 and the
code stays as an instrumented negative result.

**The fix that can work is at training time.** `--balance-states` in
`ml/train_reader.py` weights each sample by `1/count(state)`, so every state's
characters appear equally often in a batch. Sampling, not capping: cutting MH
to the runner-up's 615 would discard 2,196 real crops, and real crops are this
reader's binding constraint — the synthetic ones taught it a font and read 0 of
45. `ml/kaggle/kaggle_train.py` passes it, so the next Kaggle reader is trained
without the prior rather than corrected after the fact.

Regenerate `STATE_PRIOR` with `ml/state_prior.py` whenever the crop set is
rebuilt. It is a constant in `ml/sidecar.py` and not a file beside the weights
because a deploy that shipped one and not the other would apply the wrong
correction silently.

**The floor ships OFF — `READER_MIN_CONF` defaults to 0.0.** At 0.99 only 2 of
154 crops from real video clear it, so a dashboard of live traffic shows almost
no plates, and being shown is what this build is for. The trade is the whole
first row of that table: 17 wrong reads instead of 3, 25 correct instead of 29,
and 114 invented plates instead of 0 on the KIIT clip.

Set `ARGUS_READER_MIN_CONF=0.99` before quoting any accuracy number. Every
figure in this section was measured with the floor on.

Eight epochs on under two thousand crops moved it from nothing to a third, and
the remaining errors became near-misses (`ML10B9306` -> `ML10B9308`). The
binding constraint is the amount of real data, not the architecture: 1,756
crops, 736 unique plates, 49% Maharashtra. Video is the only plentiful source —
`--video` runs the sidecar's own vehicle-then-plate stages over traffic footage.

### The Kaggle reader is in service, and it never abstains

`ml/kaggle/kaggle_train.py` trains both models on a Kaggle T4 in one session:
YOLO11s over 27,009 merged images, then the CRNN over 178,266 samples of which
73% are real Indian crops. Held out the reader scores **93.4% exact at 2.50%
CER** — the numbers asked for, and they are honest about that split.

They do not survive contact with the 45 photographs, where it reads 25. The
whole of the gap is one behaviour: **it answers every crop it is given.** On the
KIIT clip it returns text for 93 of 93 crops against PaddleOCR's 86, and the
reads that match no legible plate go from 9 to 71 — overwhelmingly `MH01`,
`MH02`, `MH04`, because 49% of the training crops are Maharashtra and an
unreadable crop makes it emit its prior. Exactly-right on that clip falls 4/5
to 2/5.

A CTC model has no abstention: softmax over 37 classes always has an argmax, and
`correct_plate()` cannot reject the output because the invented plates are
grammatical Indian registrations. PaddleOCR abstains because its detector finds
no text line at all. **That, not CER, is what a reader has to be judged on** —
a confident wrong plate is worse for ANPR than no plate, and the held-out split
cannot see the difference because every crop in it has a true label.

`runs/reader-k12` and `runs/detect/plate-k12` are the defaults in
`ml/sidecar.py`. `ARGUS_READER=paddle` puts the 39/45 reader back in one
command, and the YOLOv8n weights stay at `runs/detect/plate` for
`ARGUS_PLATE_MODEL`. The next gain is a confidence floor on the CTC path, not
another epoch.

**Never train the reader on `~/indian-plates/test` or on the KIIT clip.** They
are what `groundtruth_test50.csv` and `groundtruth_kiit.csv` score against, and
training on them turns the only honest measurements in this project into a
memory test.

Full procedure, three training routes, a catalogue of further Indian plate
datasets with licences, and the current dataset ([Quobotic Indian number plate on
Roboflow][ds]) are in `ml/TRAINING.md`. `ml/prepare_dataset.py` takes several
`--src` directories and merges them into one YOLO set, dropping repeated
photographs by perceptual hash first — public plate datasets are largely
re-uploads of each other, and the same image in train and val inflates val mAP
silently. The hash is mirror-invariant, because a horizontal flip is the
commonest augmentation these sets apply and it is the same photograph. The two
sets on this machine merge to **10,098 unique images** with **1,298 duplicates
removed, 11%** — 887 of those are flips that a plain hash let through.

[ds]: https://universe.roboflow.com/quobotic/indian-number-plate

Develop the sidecar against stock `yolov8n.pt` + a stub plate region; trained
weights are a one-line swap at a single call site.

## CPU inference budget — respect it or the demo drops frames

- Process video at **5 FPS, not 30**.
- Vehicle detection + BoT-SORT: every processed frame.
- Plate detection + OCR: **a track's first few frames and its best crop only.**
- Re-ID embedding: **on track exit only** — the sole moment Module C needs it.

4 streams x 5 FPS = 20 detections/sec. Quantized YOLOv8n at 480px on 16 Zen 5
threads clears that. Per-frame OCR or Re-ID does not.

**Measure before optimising — `ml/bench.py` times each stage separately**, and
on real footage the answer was not where the prose above implies. Per processed
frame, one stream, 848x478 traffic:

| stage | before | after |
|---|---|---|
| decode (5 of every 30 frames) | 1.6 ms | 1.6 ms |
| vehicle detect + BoT-SORT with Re-ID | 50 ms | 50 ms |
| plate detect + OCR, amortised | 289 ms | 64 ms |
| **whole loop** | **341 ms** | **121 ms** |

OCR was 85% of everything. Detection was never the problem, and neither was
decoding — which is why "cut `imgsz` before cutting features" is the wrong first
move here: 480 to 320 saves 4 ms of a 121 ms frame and costs distant vehicles.
Cut what the profile says.

Threads are the multi-stream lever. One sidecar takes all 16 hardware threads
for torch and ten more for Paddle; four of them each try to own the machine.
`ARGUS_THREADS` caps both, and `worker/ingest.ts` sets it to cores/4. Four
concurrent streams over the same clip: 102.7 s uncapped, 79.9 s capped.

**Do not export the vehicle detector to OpenVINO.** It is the obvious next
optimisation and it is wrong twice — slower (67 vs 51 ms, because BoT-SORT
reuses the detector's own features for Re-ID and an exported IR does not expose
them), and it changes the embedding dimension from 64 to 256, which would make
Module C's layer 2 silently stop matching. The constant in `ml/sidecar.py`
carries the measurement.

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
| Scope | Sidecar pipeline, Module C, analytics, REST + WebSocket, Postgres/TimescaleDB | Camera grid, deck.gl map, charts, vehicle search, alerts, camera graph data, demo video |

Shared, changed by neither alone: `src/contract.ts`, `db/schema.sql`.

Dev B builds against `/api/mock` (`src/server/mock.ts`) from minute one and
never blocks on the pipeline. **Changing a response shape is a contract change —
say so.**

## Commands

```bash
sudo ./scripts/postgres-local.sh   # native Postgres 16 + TimescaleDB, once
# or: docker compose up -d db          # the container route
npm run db:setup     # apply db/schema.sql (idempotent) + seed camera topology
npm run dev          # Next UI + /api + /ws on :3000  (custom server)
npm run worker       # ingest supervisor + Python sidecars
npm run worker:watch # same, restarting on edit (reloads every model: slow)
npm run check        # tsc --noEmit
npm run selfcheck    # mock fixtures + Module C + Module D
npm run smoke        # every endpoint over real HTTP, judged by the zod contract

# End-to-end with no video, no CV deps, no GPU: synthetic sidecars that emit
# a vehicle travelling CAM1 -> CAM3 -> CAM2 (third leg has no readable plate).
ARGUS_PYTHON=python3 ARGUS_CAMERAS='CAM1=demo,CAM3=demo' npm run worker

./scripts/dev-https.sh          # self-signed cert, so a PHONE can use its camera
npx tsx scripts/prune-uploads.ts --days 7 [--apply]   # delete old source video
npx tsx db/repair.ts            # report trajectories that break the contract
npx tsx db/repair.ts --apply    # delete them

python ml/sidecar.py --selfcheck --camera X --source X   # plate-correction check
python ml/demo_detect.py --source photo.jpg --ocr        # eyeball the detector
python ml/bench.py --source clip.mp4                     # ms per pipeline stage
python ml/diagnose_video.py --source clip.mp4 --truth ml/groundtruth_kiit.csv
python ml/prepare_dataset.py --src A B C --dst datasets/plates-merged  # merge datasets
python ml/sweep_floor.py --model runs/detect/plate-k12/weights/best.pt  # pick READER_MIN_CONF
python ml/sweep_floor.py --var ARGUS_STATE_PRIOR_TAU --floors 0,0.4,1.0  # or any knob
python ml/state_prior.py            # regenerate STATE_PRIOR after new crops
python ml/make_demo_clips.py                # synthetic demo/cam*.mp4 test clips
python ml/train_plate.py --epochs 50        # on Kaggle
python ml/export_onnx.py --weights runs/detect/plate/weights/best.pt --fp32

kaggle kernels push -p ml/kaggle            # BOTH models on one T4 session
kaggle kernels status deeptanshu789/argus-plate-detector-and-reader
kaggle kernels output deeptanshu789/argus-plate-detector-and-reader -p runs/kaggle

ARGUS_READER=paddle npm run worker          # the 39/45 reader, one command back
ARGUS_PLATE_MODEL=runs/detect/plate/weights/best_openvino_model npm run worker
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
| `src/server/db.ts`, `src/server/bus.ts` | Done — all SQL, LISTEN/NOTIFY pub/sub |
| Real `/api/*` routes | Done, contract-validated, smoke-tested |
| `worker/ingest.ts` | Done — writes, associates, alerts, rollup, publishes |
| `ml/sidecar.py` | Done — `run()` is the real decode/track/OCR/ReID loop |
| `ml/bench.py`, `ml/diagnose_video.py` | Per-stage timings, per-stage plate losses |
| `test/smoke.ts` | 38 checks incl. a live end-to-end pipeline run |
| Trained plate weights | Done — YOLO11s `plate-k12`, mAP50 0.974, 38/45 with PaddleOCR |
| Trained reader weights | In service — CRNN `reader-k12`, floor off, 25/45, precision 60% |
| `src/app/(dashboard)` | Done — live, map, analytics, search, upload, devices |
| Real traffic footage | Upload it at `/upload` — no code change needed |

## Deployment

`DEPLOY.md`, `docker-compose.prod.yml`, `Caddyfile`. Four containers; only Caddy
binds a public port. Two things that must stay true:

- **Argus authenticates nobody.** Caddy's basic auth is the entire access
  control. Never publish port 3000 — that routes around it.
- **`/cam/<code>` is deliberately open**, so a phone pairs by link without an
  operator password typed on a handset. The code guards pushing video IN; it
  reads nothing back out.

`./scripts/deploy-prepare.sh` generates the secrets and checks the box can run
this. Weights are not in the image: `runs/` is bind-mounted, so a model swap is
a restart rather than a rebuild.

## Non-negotiable

Record the backup demo video by hour 34. A live demo failure with no fallback
loses the whole thing.
