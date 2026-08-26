#!/usr/bin/env python3
"""Inference sidecar — the ONLY Python that runs at application runtime.

One process per camera. Reads a video source, runs detection, tracking, OCR and
Re-ID, and emits newline-delimited JSON on stdout. Everything downstream —
cross-camera association, analytics, API, WebSocket, UI — is TypeScript.

    python ml/sidecar.py --camera CAM1 --source demo/cam1.mp4 --fps 5

Supervised by worker/ingest.ts, which parses each line against the SidecarEvent
schema in src/contract.ts. THAT SCHEMA IS THE CONTRACT: if you change an event
shape here, change it there in the same commit or the supervisor will reject
your output as a contract violation.

Event types, one JSON object per line on stdout:

  {"event":"ready","camera_id":...,"fps":...}
  {"event":"detection","camera_id":...,"track_id":...,"ts":...,"bbox":[x1,y1,x2,y2],
   "conf":...,"vehicle_type":...}
  {"event":"track_closed","camera_id":...,"track_id":...,"entry_time":...,
   "exit_time":...,"vehicle_type":...,"color":...,"plate_text":...,"plate_conf":...,
   "embedding":[...floats, same length on every camera...],"color_hist":[...]}
  {"event":"error","camera_id":...,"detail":...}

Diagnostics go to stderr. Anything non-JSON on stdout is a bug.

NO GPU. The demo machine is an AMD Ryzen AI 7 350 with no CUDA device. Models
load as an OpenVINO export when one exists (see ml/export_onnx.py), PyTorch CPU
otherwise. Training happens separately on Kaggle; this file never trains
anything.

CPU budget, and the reason this file is structured the way it is:
  - decode and detect at 5 FPS, not 30
  - plate detection + OCR on a track's first few frames and its best crop only
  - Re-ID embedding on track exit only
4 streams x 5 FPS = 20 detections/sec, which a quantized YOLOv8n at 480px
clears on 16 Zen 5 threads. Per-frame OCR or Re-ID does not.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

# THREADS. Set before any of torch, OpenVINO, oneDNN or OpenCV is imported,
# because each reads its thread count once at import and ignores later changes.
#
# One sidecar left alone takes all 16 hardware threads for the vehicle detector
# AND all 16 for PaddleOCR, and the two then fight over the same cores. That is
# invisible on one stream and expensive on four: the city demo runs one process
# per camera, so the default of "everything" means four processes each trying to
# own the whole machine.
#
# worker/ingest.ts sets ARGUS_THREADS to the machine's thread count divided by
# the number of cameras it is about to spawn. Run by hand with nothing set, the
# sidecar keeps the library defaults, which is right for measuring one stream.
_THREADS = os.environ.get("ARGUS_THREADS")
if _THREADS and _THREADS.isdigit() and int(_THREADS) > 0:
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_var, _THREADS)

# Re-exported so the TS side and this file agree on the label set.
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# The three OCR confusions that account for most Indian-plate errors. This
# correction pass is worth more than model fine-tuning -- do not drop it.
DIGIT_FOR = {"O": "0", "Q": "0", "I": "1", "L": "1", "B": "8", "S": "5", "Z": "2"}
# NOT derived by inverting DIGIT_FOR: that map is many-to-one (O and Q both go
# to 0), so inverting it silently picks whichever key came last. Spelled out.
ALPHA_FOR = {"0": "O", "1": "I", "8": "B", "5": "S", "2": "Z"}
STATE_CODES = {
    "AP","AR","AS","BR","CG","CH","DD","DL","DN","GA","GJ","HP","HR","JH","JK",
    "KA","KL","LA","LD","MH","ML","MN","MP","MZ","NL","OD","PB","PY","RJ","SK",
    "TN","TR","TS","UK","UP","WB",
}

# What a character in the STATE SLOT might really be, cheapest reading first.
#
# The two-letter state code is the one part of a plate with a CLOSED set of
# valid answers, and that is worth exploiting: rather than guess each character
# independently and check the result, every plausible pair is scored and the
# best real state code wins.
#
# Measured need. On 848x478 footage a plate is around 40 px wide and PaddleOCR
# returns OD33AU6001 as "0033AU6001" — both O and D read as 0. ALPHA_FOR maps
# 0 to O and nothing else, so the pair became "OO", failed the STATE_CODES
# check, and twenty-six agreeing reads of one vehicle were all thrown away.
# One frame where the plate was large enough to resolve the D survived, which
# is why the video produced exactly one plate out of forty vehicles.
#
# Cost 1 is a substitution that looks identical at low resolution; cost 2 is
# one that needs the glyph to be genuinely degraded. Identity is free, so a
# clean read never loses to a repaired one.
STATE_ALTS: dict[str, tuple[tuple[str, int], ...]] = {
    # D and 0 are the same glyph at this size, so both cost 1. That is what
    # makes "00" resolve to OD at all.
    "0": (("O", 1), ("D", 1), ("Q", 2), ("U", 2)),
    "1": (("I", 1), ("L", 1), ("T", 2)),
    "2": (("Z", 1),),
    "3": (("B", 2), ("E", 2)),
    "4": (("A", 1),),
    "5": (("S", 1),),
    "6": (("G", 1), ("C", 2)),
    "7": (("T", 2), ("Z", 2)),
    "8": (("B", 1), ("R", 2)),
    "9": (("G", 2), ("P", 2), ("Q", 2)),
}
# DIGITS ONLY, and that is the whole principle. A state code contains no
# digits, so a digit in this slot is PROVABLY a misread and replacing it is a
# repair. A letter here might simply be correct, and substituting one letter
# for another is a guess dressed up as a repair.
#
# Measured: allowing letter-for-letter turned "MB06J9177" into MP06J9177 — a
# real Bengal plate reported as a Madhya Pradesh one, from a photo the stricter
# rule declines to read at all. A wrong plate attaches a registration to the
# wrong vehicle; a missing one only fails to help.

# A repaired state code cannot cost more than this. Two cheap substitutions, or
# one expensive one. Beyond that the read is a guess dressed up as a plate, and
# a wrong plate attaches a real registration to the wrong vehicle.
STATE_MAX_COST = 2

# Codes belonging to territories with very small vehicle fleets, penalised by
# one so they never win a TIE against a populous state.
#
# Without this, "00" repairs to OD (Odisha, ~10 million vehicles) and DD (Daman
# and Diu, a district of about 250,000 people) at exactly the same cost, and
# which one wins depends on dictionary iteration order. A prior this lopsided is
# not a guess, it is the only defensible way to break the tie.
RARE_STATES = {"DD", "DN", "LD", "LA", "PY", "CH", "SK", "MZ", "NL", "AR", "ML", "TR"}


def _state_code(pair: str) -> tuple[str, int] | None:
    """Best real state code for two characters, or None.

    Returns (code, cost). Cost is 0 for a clean read, and is folded into the
    caller's correction penalty so a repaired plate reports lower confidence
    than one that needed no help.
    """
    if len(pair) != 2:
        return None
    if pair in STATE_CODES:
        return pair, 0

    best: tuple[str, int] | None = None
    for a, ca in ((pair[0], 0), *STATE_ALTS.get(pair[0], ())):
        if not a.isalpha():
            continue
        for b, cb in ((pair[1], 0), *STATE_ALTS.get(pair[1], ())):
            if not b.isalpha():
                continue
            cost = ca + cb
            if cost > STATE_MAX_COST:
                continue
            code = a + b
            if code not in STATE_CODES:
                continue
            ranked = cost + (1 if code in RARE_STATES else 0)
            if best is None or ranked < best[1]:
                best = (code, ranked)
    return best


# THE PROTOCOL CHANNEL. Captured before anything else can touch it, then
# sys.stdout is pointed at stderr for the rest of the process.
#
# Ultralytics prints "Loading ... for OpenVINO inference" to stdout. PaddleOCR
# prints its model cache paths there. Neither is a bug in those libraries --
# but this stream is a protocol, and one stray line makes the supervisor log a
# contract violation and drop the JSON around it. Rather than chasing every
# library's verbosity flag, take stdout away from all of them: only emit() can
# reach the real handle, and every print() in this file or any dependency lands
# on stderr where the supervisor already forwards it as a diagnostic.
_PROTOCOL = sys.stdout
sys.stdout = sys.stderr


def emit(**event) -> None:
    """One JSON object per line, flushed. The supervisor reads line by line."""
    json.dump(event, _PROTOCOL, separators=(",", ":"))
    _PROTOCOL.write("\n")
    _PROTOCOL.flush()


# Unique per sidecar PROCESS, and prefixed onto every track id.
#
# The tracker numbers tracks from 1 on every start, so a restarted sidecar
# re-issues 1, 2, 3 for entirely different vehicles. `tracks` is keyed
# UNIQUE (camera_id, track_id), so the upsert in worker/ingest.ts merged the old
# vehicle and the new one into a single row: one database id, two sets of entry
# and exit times. Trajectories then formed cycles that visited the same track
# twice and carried timestamps that ran backwards.
#
# ponytail: process start time to the microsecond, hex, last 8 digits. Two
# sidecars for one camera cannot start in the same microsecond, and the id stays
# short enough to read in a log.
#
# ARGUS_RUN_ID pins it. Set it only when a replay is meant to be treated as the
# SAME observations rather than new ones -- re-running a recorded stream to test
# idempotence, or resuming an interrupted file after a crash. Two live cameras
# sharing a pinned id is harmless, because tracks are unique per (camera, track).
RUN_ID = os.environ.get("ARGUS_RUN_ID") or f"{int(time.time() * 1e6):x}"[-8:]


def now_iso() -> str:
    """MILLISECONDS, not seconds. At 5 FPS five consecutive frames fall inside
    one second, so a second-resolution timestamp makes a track's first and last
    detection look simultaneous -- and estimateSpeed() then divides by a zero
    interval, returns null, and every speed in the dashboard is blank with no
    error anywhere to explain it."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _fix(chunk: str, want_digit: bool) -> tuple[str, int] | None:
    """Force a run of characters into one class, correcting known OCR
    confusions. Returns (fixed, corrections) or None if a character cannot
    plausibly belong to that class at all."""
    out, fixes = [], 0
    for ch in chunk:
        if want_digit and ch.isalpha():
            got = DIGIT_FOR.get(ch)
            if got is None:
                return None
            out.append(got); fixes += 1
        elif not want_digit and ch.isdigit():
            got = ALPHA_FOR.get(ch)
            if got is None:
                return None
            out.append(got); fixes += 1
        else:
            out.append(ch)
    return "".join(out), fixes


def correct_plate(raw: str) -> tuple[str | None, float]:
    """Validate and repair an OCR read of an Indian registration plate.

    Returns (plate, penalty). penalty is how much of the string had to be
    corrected — the caller subtracts it from OCR confidence, so a heavily
    repaired read cannot present itself as confidently as a clean one.

    THE SHAPE, and why it is not one rigid pattern:

        <state 2 alpha> <district, starts with a digit> <series> <number>

    A first version demanded exactly `AA DD A(A) DDDD` and rejected half the
    plates measured on real photos. `DL9CAU4743` is a genuine Delhi plate whose
    district code is digit-then-letter; `MH05DK101` has a three-digit series
    number. Both are valid and both were thrown away, which matters more than
    it sounds: layer 1 of the association engine is plate text and carries
    60-70% of cross-camera matches, so a plate rejected here is a vehicle the
    system can only follow by appearance.

    What is still enforced, because these are what make a plate a plate:
      - the state code is real (checked against STATE_CODES)
      - the registration number is a trailing run of 2 to 4 digits
      - the district begins with a digit
    Correction is applied ONLY where the character class is known from
    position. The series letters in the middle are left exactly as read.
    """
    s = "".join(c for c in raw.upper() if c.isalnum())
    if not 9 <= len(s) <= 11:
        return None, 0.0

    state = _state_code(s[:2])
    if state is None:
        return None, 0.0

    best: tuple[str, int] | None = None
    # Prefer the longest trailing number and the longest district that parse:
    # "4743" is a better reading of DL9CAU4743 than "743", and a shorter one
    # silently drops a digit rather than failing.
    for num_len in (4, 3):
        for district_len in (2, 1):
            series = s[2 + district_len : len(s) - num_len]
            if not 1 <= len(series) <= 3:
                continue
            number = _fix(s[-num_len:], want_digit=True)
            district = _fix(s[2 : 2 + district_len], want_digit=True)
            fixed_series = _fix(series, want_digit=False)
            if number is None or district is None or fixed_series is None:
                continue
            plate = state[0] + district[0] + fixed_series[0] + number[0]
            fixes = state[1] + district[1] + fixed_series[1] + number[1]
            if best is None or fixes < best[1]:
                best = (plate, fixes)
        if best is not None:
            break          # a 4-digit number beat a 3-digit one; stop here

    if best is None:
        return None, 0.0
    return best[0], round(best[1] * 0.05, 4)


# Camera ordering for the synthetic demo. The vehicle travels CAM1 -> CAM3 ->
# CAM2, and each sidecar staggers its output so the worker sees the legs in the
# order real event timing would deliver them.
DEMO_ROUTE = ["CAM1", "CAM3", "CAM2", "CAM4"]
DEMO_PLATE = "KA05MR7821"


# The REAL tracker emits 64 floats, not 512 -- BoT-SORT's `model: auto` encoder
# decides that, and swapping the model changes it. Matching the real dimension
# here means the demo exercises the worker's dimension guard the same way live
# cameras do, instead of passing on a number nothing else produces.
DEMO_EMBED_DIM = 64


def _demo_embedding(seed: int) -> list[float]:
    """Deterministic unit vector. Same vehicle -> near-identical vectors, so
    layer 2 of the association engine fires exactly as it would on real tracker
    output. What matters is that every demo sidecar agrees on the dimension,
    which is the same rule the real ones follow."""
    import math
    raw = [math.sin((i + 1) * 0.017 + seed) for i in range(DEMO_EMBED_DIM)]
    norm = math.sqrt(sum(v * v for v in raw)) or 1.0
    return [round(v / norm, 6) for v in raw]


def demo(camera: str, fps: int) -> None:
    """Emit a scripted event sequence -- no video, no models, no CV dependencies.

    Exists so the whole Python -> JSON -> worker -> association path can be run
    and watched today, before the real pipeline or a GPU-trained detector exist.
    Also gives the dashboard developer live traffic without sourcing footage.

        ARGUS_CAMERAS=CAM1=demo,CAM3=demo npm run worker

    Timestamps are backdated so a journey that takes minutes plays out
    instantly: CAM1's vehicle exits 200 s "ago", CAM3's arrives 30 s "ago", which
    is inside the 0.5x-2.0x window around the 168 s link.
    """
    import time

    try:
        leg = DEMO_ROUTE.index(camera)
    except ValueError:
        leg = 0
    # Stagger so the earlier leg is already in the worker's candidate window
    # when the later one closes -- the same ordering real cameras produce.
    time.sleep(leg * 2)

    base = time.time() - 200 + leg * 170
    entry = datetime.fromtimestamp(base, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    exit_ = datetime.fromtimestamp(base + 8, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    for i in range(fps):
        emit(event="detection", camera_id=camera, track_id=f"{RUN_ID}-1", ts=entry,
             bbox=[400 + i * 12, 220, 560 + i * 12, 340], conf=0.93, vehicle_type="car")
        time.sleep(1.0 / max(fps, 1))

    emit(event="track_closed", camera_id=camera, track_id=f"{RUN_ID}-1",
         entry_time=entry, exit_time=exit_, vehicle_type="car", color="white",
         # Third leg has an unreadable plate on purpose: it forces the match to
         # come from layers 2+3, which is exactly what Act 2 of the demo shows.
         plate_text=None if camera == "CAM2" else DEMO_PLATE,
         plate_conf=None if camera == "CAM2" else 0.97,
         embedding=_demo_embedding(3), color_hist=[10.0, 20.0, 30.0, 40.0])

    print(f"[{camera}] demo leg {leg} emitted", file=sys.stderr)


# ---------------------------------------------------------------- real run --

# Stock COCO: car, motorcycle, bus, truck.
#
# DO NOT EXPORT THIS ONE TO OPENVINO. It is the obvious optimisation and it is
# wrong twice. Measured on the same 20 processed frames of real footage:
#
#   yolov8n.pt              50.9 ms/frame,  embedding dimension  64
#   yolov8n_openvino_model  67.3 ms/frame,  embedding dimension 256
#
# Slower, because ml/botsort.yaml sets `model: auto` and the tracker reuses the
# DETECTOR's own features for Re-ID. A .pt model hands those over from a forward
# pass it already ran; an exported IR does not expose them, so ultralytics runs
# a second network to get them. The export saves time on detection and spends
# more than it saved on appearance.
#
# And the dimension changes, which is the dangerous half. Module C compares
# embeddings across cameras by cosine similarity; vectors of different length
# score zero, so a fleet where one camera used the export and the others did not
# would show layer 2 quietly never firing rather than any error. worker/ingest.ts
# checks for exactly this and refuses to start, which is why the trap is
# survivable -- but it should not be walked into.
#
# The plate detector is a different case and IS exported: nothing tracks with
# it, so only its detection speed matters.
VEHICLE_WEIGHTS = "yolov8n.pt"
PLATE_CANDIDATES = (
    "runs/detect/plate/weights/best_openvino_model",
    "runs/detect/plate/weights/best.pt",
)
# Two sizes, not one, because the two detectors are constrained differently.
#
# The vehicle detector runs on the whole frame every processed frame and is the
# second-largest cost in the loop, so its size is the cheapest thing to trade.
# The plate detector runs on a vehicle crop only when a track wants a read, and
# it is an OpenVINO export -- an exported IR is compiled at ONE input size, and
# handing it another makes ultralytics letterbox back to the exported size,
# paying for the resize and gaining nothing. PLATE_IMGSZ must match whatever
# ml/export_onnx.py was run with.
#
# IMGSZ is kept as an alias because four scripts import it.
VEHICLE_IMGSZ = 480                    # see CLAUDE.md: cut this before cutting features
PLATE_IMGSZ = 480                      # must equal the export imgsz
IMGSZ = VEHICLE_IMGSZ
VEHICLE_CONF = 0.35
PLATE_CONF = 0.25

# A track is closed after this many PROCESSED frames without a sighting. At
# 5 FPS that is 2 seconds -- long enough to survive an occlusion behind a bus,
# short enough that the cross-camera match is not delayed past its own window.
MISS_LIMIT = 10

# OCR budget. Reading every frame would blow the inference budget on its own;
# these two rules keep a track to a handful of reads regardless of its length.
PLATE_FIRST_FRAMES = 2                 # early frames, before it drives out of view
PLATE_MAX_ATTEMPTS = 10                # hard ceiling per track
PLATE_GROWTH = 1.25                    # ...plus a retry each time the crop grows this

# Vehicles narrower than this never get an OCR attempt at all.
#
# MEASURED on 848x478 traffic footage: across 209 vehicle crops, not one plate
# was ever read from a vehicle under 124 px wide, and 62% of all crops were
# narrower than that. Those attempts cost a plate detection and two OCR passes
# each to learn nothing — and worse, they burned the per-track attempt ceiling
# before the vehicle got close enough to be legible. Forty vehicles yielded one
# plate.
#
# Set below the measured floor rather than at it, because that floor came from
# one clip. In ABSOLUTE pixels, not a fraction of the frame: legibility is a
# property of how many pixels fall on the plate, so a 4K camera reading the same
# car at the same distance genuinely does have more to work with.
#
# Re-measured on the same clip after the reader was fixed: the narrowest vehicle
# that produced a read moved from 124 px to 190, and 66% of crops now fall below
# it. Raising the gate to around 150 would skip more attempts that cannot
# succeed. NOT TAKEN, because that floor is one clip at one resolution and the
# stage it would save is no longer the expensive one -- OCR went from 722 ms per
# attempt to 166. The figure is recorded so the next person has it; raise this
# only with a second clip agreeing.
#
# ponytail: a single width threshold. A vehicle-type-aware one would help — a
# motorcycle plate is legible at a narrower crop than a truck's — if plate
# reading on two-wheelers ever becomes the limit.
PLATE_MIN_VEHICLE_PX = 110

# How many reads must AGREE before a track stops trying.
#
# Two was too few. OCR errors on a moving vehicle are mostly independent, which
# is what makes voting work — but some are not: at 40 px, D and O are the same
# glyph, and PaddleOCR reads OD02DR5938 as OD02OR5938 on frame after frame. Two
# such reads agree and stop the track before it ever gets close enough to
# resolve the D, even though a later frame would have read it correctly.
#
# THREE WAS TRIED AND MEASURED NOTHING. The hope was that a later, closer frame
# would resolve the D once the pair of wrong reads no longer stopped the track.
# It did not: on this footage PaddleOCR reads that D as an O in every frame, so
# a third opinion is the same opinion. Same 2 of 5 exact, 6 seconds slower.
#
# Left at two, with the experiment recorded, because the next person to look at
# a one-character error will have the same idea.
PLATE_AGREE_STOP = 2
                                       #    much bigger, i.e. the vehicle came closer


def _first_present(candidates) -> str | None:
    from pathlib import Path
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def find_plate_weights() -> str | None:
    """Trained plate detector, if one has been exported. Returns None when it
    has not: the sidecar must still run end to end on stock weights, or nothing
    downstream can be developed before the Kaggle run finishes."""
    return _first_present(PLATE_CANDIDATES)





def _colour_hist(crop) -> list[float]:
    """32-bin hue histogram, normalised. Layer 2's tie-breaker in Module C:
    cheap, lighting-tolerant, and it costs nothing next to the detector."""
    import cv2
    if crop.size == 0:
        return [0.0] * 32
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    total = float(hist.sum()) or 1.0
    return [round(float(v) / total, 6) for v in hist]


# Coarse colour names from mean HSV. Not a classifier -- a label for the UI.
def _colour_name(crop) -> str | None:
    import cv2
    import numpy as np
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, sat, val = (float(np.median(hsv[:, :, i])) for i in range(3))
    if val < 50:
        return "black"
    if sat < 40:
        return "white" if val > 170 else "silver"
    for lo, hi, name in ((0, 10, "red"), (10, 25, "orange"), (25, 35, "yellow"),
                         (35, 85, "green"), (85, 130, "blue"), (130, 165, "purple")):
        if lo <= h < hi:
            return name
    return "red"


class _Track:
    """Per-track state. Everything expensive is decided here: which frames get
    an OCR attempt, which crop is the best one, and what gets emitted on exit."""

    __slots__ = ("track_id", "vehicle_type", "entry_time", "last_seen_frame",
                 "frames", "attempts", "best_area", "last_ocr_area", "votes",
                 "embedding", "colour", "colour_hist")

    def __init__(self, track_id: str, vehicle_type: str, ts: str, frame_no: int):
        self.track_id = track_id
        self.vehicle_type = vehicle_type
        self.entry_time = ts
        self.last_seen_frame = frame_no
        self.frames = 0
        self.attempts = 0
        self.best_area = 0.0
        # Crop size at the most recent OCR attempt. wants_ocr() retries once the
        # vehicle has grown meaningfully SINCE THEN, which spends the budget as
        # it approaches rather than all at once when it is furthest away.
        self.last_ocr_area = 0.0
        # Every accepted read for this track, not just the best one. A vehicle
        # gets several OCR attempts as it crosses the frame; those reads fail
        # INDEPENDENTLY -- a smudge that turns 8 into B on one frame is gone on
        # the next -- while the correct string repeats. Two agreeing reads are
        # worth more than one confident one, and picking by confidence alone
        # throws that away. See plate() below.
        self.votes: dict[str, list[float]] = {}
        self.embedding: list[float] = []
        self.colour: str | None = None
        self.colour_hist: list[float] = []

    def record(self, plate: str, conf: float | None) -> None:
        self.votes.setdefault(plate, []).append(conf if conf is not None else 0.0)

    def plate(self) -> tuple[str | None, float | None]:
        """The winning read: most votes first, mean confidence as tie-break.

        Confidence reported is the mean of the votes for the winner, so a plate
        agreed on twice at 0.8 outranks a single 0.95 -- which is the right
        ordering, because a repeated reading is evidence and a lone one is not.
        """
        if not self.votes:
            return None, None
        best = max(self.votes.items(),
                   key=lambda kv: (len(kv[1]), sum(kv[1]) / len(kv[1])))
        return best[0], round(sum(best[1]) / len(best[1]), 3)

    def wants_ocr(self, area: float, width: int) -> bool:
        """Should this frame get an OCR attempt?

        The growth clause is what makes this work on real footage. A vehicle
        enters the frame far away and unreadable and grows as it approaches, so
        the FIRST frames of a track are the worst ones — spending every attempt
        there is spending them all on plates too small to read.

        It compares against the size at the LAST ATTEMPT, not against the
        largest crop seen. An earlier version used `best_area`, which run()
        updates before calling this, so the test read `area > area * 1.4` and
        was never true: the growth retry never fired once, and every track was
        read only in its first three frames. On a real 848x478 clip that turned
        forty vehicles into one plate.
        """
        # Too far away to carry a legible plate. Checked FIRST, before the
        # frame count, because a track's opening frames are the ones where the
        # vehicle is furthest away — spending the budget there is what made
        # real footage read almost nothing.
        if width < PLATE_MIN_VEHICLE_PX:
            return False
        # Stop once two attempts have AGREED. One confident read is not enough
        # to stop on: the confident wrong reads are exactly the dangerous ones.
        if any(len(v) >= PLATE_AGREE_STOP for v in self.votes.values()):
            return False
        if self.attempts >= PLATE_MAX_ATTEMPTS:
            return False
        # "First frames" now means the first frames at which it is BIG ENOUGH,
        # which for an approaching vehicle is where it becomes readable.
        if self.attempts < PLATE_FIRST_FRAMES:
            return True
        return area > self.last_ocr_area * PLATE_GROWTH


# Preprocessing variants, tried in order until one produces a plate that
# validates. MEASURED on 161 real plate crops:
#
#   upscale to 48 px + CLAHE   58%      <- best single variant
#   upscale to 96 px, no CLAHE 55%
#   upscale to 96 px + CLAHE   52%
#   upscale to 160 px + CLAHE  48%      <- more upscaling is WORSE
#   first two combined         65%
#
# Two lessons worth keeping. Upscaling harder does not help: PaddleOCR has its
# own resize and feeding it an interpolated blur destroys the edges it needs.
# And the two best variants fail on DIFFERENT crops, so a retry is worth far
# more than any single better filter. The retry only runs when the first read
# failed to validate, so the crops that already work cost one OCR call.
#
# RE-MEASURED after the reader changed (see make_reader), against the 45
# hand-labelled plates, since the numbers above describe a reader that no longer
# exists:
#
#   48+CLAHE, 96 plain           39 correct, 2 wrong, 4 missed, 95% precision
#   48 plain, 96 plain           39 correct, 2 wrong, 4 missed, 95% precision
#   48+CLAHE only                37 correct, 2 wrong, 6 missed, 95% precision
#   96 plain only                36 correct, 1 wrong, 8 missed, 97% precision
#   48+CLAHE, 96 plain, 64+CLAHE 39 correct, 3 wrong, 3 missed, 93% precision
#
# The pair is still right. CLAHE no longer earns its place on its own, but the
# pair costs nothing to keep. A THIRD variant is the tempting one and it is a
# bad trade: it converts a miss into a wrong answer, which is the wrong
# direction for a system that puts registrations on vehicles.
_OCR_VARIANTS = ((48, True), (96, False))


# Every OCR call is built here, and nowhere else. Four scripts used to
# construct PaddleOCR with their own copy of these arguments, which meant a
# measurement taken with one script did not necessarily describe what the
# sidecar ran.
#
# enable_mkldnn=False is NOT tuning. Paddle's oneDNN executor raises
# ConvertPirAttribute2RuntimeAttribute on this CPU build for every predict()
# call, and the constructor gives no hint that it will. Omit the flag and OCR
# silently reads nothing forever.
#
# The two doc_* models are the tuning. PaddleOCR 3.x runs a document
# orientation classifier and a document unwarping network before every read
# unless told not to. Both exist for photographed pages; a plate crop is
# neither rotated by 90 degrees nor curled at the corners. Measured on this
# machine, per predict() call on the same image: 2348 ms with them, 1794 ms
# without, identical text out.
#
# Turning the doc models off is not only cheaper, it READS BETTER. Measured
# against the 45 hand-labelled plates in ml/groundtruth_test50.csv, changing
# nothing else:
#
#   with doc orientation + unwarping    35 correct, 2 wrong, 8 missed, 95% precision
#   without                             39 correct, 2 wrong, 4 missed, 95% precision
#
# Four plates that were missed are now read, and no new wrong answer appeared.
# The orientation classifier is trained on pages, and on a 40 px plate crop it
# sometimes decides the text runs vertically and hands the recogniser a crop
# rotated by 90 degrees. There is nothing to recover from that.
#
# Capping the text detector's input size (text_det_limit_side_len) was tried
# and measured nothing: 170.8 ms per attempt uncapped, 163.4 ms at 320, 175.0 ms
# at 640, which is noise. _prepare_crop already hands it a crop 48 px tall, far
# under any limit meant for a scanned page. Not a knob worth having.
#
# ARGUS_OCR_VERSION selects the recogniser family, so a slower-but-better or
# faster-but-worse model can be measured against the same file without editing
# code. Measured, same conditions:
#
#   PP-OCRv6 (default)   39 correct, 2 wrong,  4 missed,  95% precision, 9.4% CER
#   PP-OCRv5             37 correct, 5 wrong,  3 missed,  88% precision, 8.2% CER
#   PP-OCRv4             29 correct, 7 wrong,  9 missed,  81% precision, 21.7% CER
#
# v5 reads more characters right and more PLATES wrong, which is the trade this
# project refuses: a wrong plate puts a real registration on the wrong vehicle.
# Judge by correct and by precision, never by character error rate.
OCR_VERSION = os.environ.get("ARGUS_OCR_VERSION") or "PP-OCRv6"


def apply_thread_limit() -> None:
    """Tell OpenCV and torch about ARGUS_THREADS. Call after importing them.

    The environment variables set at the top of this file only reach the BLAS
    libraries. OpenCV keeps its own pool for resize and colour conversion, and
    ultralytics resets torch's pool after import, so both have to be told again
    once they exist.
    """
    if not (_THREADS and _THREADS.isdigit()):
        return
    n = int(_THREADS)
    try:
        import cv2
        cv2.setNumThreads(n)
    except Exception:
        pass
    try:
        import torch
        torch.set_num_threads(n)
    except Exception:
        pass


def make_reader():
    """The PaddleOCR instance the whole project uses. None if it will not load."""
    try:
        from paddleocr import PaddleOCR
        kw = {}
        # Paddle defaults to 10 threads and torch to all 16, on a machine with
        # 16. Together they oversubscribe before a second camera even starts.
        if _THREADS and _THREADS.isdigit():
            kw["cpu_threads"] = int(_THREADS)
        return PaddleOCR(
            lang="en",
            ocr_version=OCR_VERSION,
            use_textline_orientation=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            enable_mkldnn=False,
            **kw,
        )
    except Exception as exc:
        print(f"OCR unavailable ({type(exc).__name__}: {exc})", file=sys.stderr)
        return None


def _prepare_crop(crop, height: int, equalise: bool):
    import cv2
    out = crop
    if out.shape[0] < height:
        scale = height / max(out.shape[0], 1)
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if equalise:
        grey = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        out = cv2.cvtColor(
            cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grey),
            cv2.COLOR_GRAY2BGR)
    return out


def _ocr_lines(reader, image) -> list[tuple[str, float]]:
    try:
        pages = reader.predict(image)
    except Exception as exc:
        print(f"ocr failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return []
    lines: list[tuple[str, float]] = []
    for page in pages or []:
        if not isinstance(page, dict):
            continue
        texts = page.get("rec_texts") or []
        scores = page.get("rec_scores") or [1.0] * len(texts)
        lines.extend((t, float(sc)) for t, sc in zip(texts, scores) if t.strip())
    return lines


# Grow the plate box before reading it. MEASURED against 45 hand-labelled
# plates, exact-match accuracy:
#
#    0% pad   28/45 (62%)   8 wrong, 9 missed
#    4% pad   30/45 (67%)   5 wrong, 10 missed
#    8% pad   35/45 (78%)   2 wrong, 8 missed     <- chosen
#   14% pad   33/45 (73%)   3 wrong, 9 missed
#   22% pad   35/45 (78%)   2 wrong, 8 missed
#
# The detector draws a box on the plate, not around it, and clips the outermost
# character often enough to matter. Six of eight WRONG reads were a 10-character
# plate read as 9 with the last digit gone -- "MH14EH7958" as "MH14EH795",
# which then parses as a perfectly valid plate with a three-digit series and is
# accepted with confidence. A wrong plate is worse than no plate: it attaches a
# real registration to the wrong vehicle. Padding removed six of those eight.
#
# RE-SWEPT after the reader changed, same 45 plates:
#
#    0% pad   38/45   2 wrong, 5 missed
#    4% pad   38/45   2 wrong, 5 missed
#    8% pad   39/45   2 wrong, 4 missed   <- kept
#   14% pad   39/45   2 wrong, 4 missed
#   22% pad   37/45   4 wrong, 4 missed
#
# Same answer, and the plateau from 8% to 14% is wide enough that a retrained
# detector drawing slightly tighter boxes will not fall off it. Re-sweep anyway
# after a retrain: this is the constant that pays for a detector's tightness.
PLATE_PAD = 0.08


def _pad(img, x1: int, y1: int, x2: int, y2: int):
    h, w = img.shape[:2]
    px = int((x2 - x1) * PLATE_PAD)
    py = int((y2 - y1) * PLATE_PAD)
    return img[max(y1 - py, 0):min(y2 + py, h), max(x1 - px, 0):min(x2 + px, w)]


def _read_plate(reader, plate_model, crop) -> tuple[str | None, float | None]:
    """Plate box -> OCR -> positional correction. Returns (plate, confidence).

    The correction penalty is subtracted from the OCR score, so a read that
    needed four fixes cannot present itself as confidently as a clean one.
    """
    if crop.size == 0 or reader is None:
        return None, None

    region = crop
    if plate_model is not None:
        res = plate_model(crop, imgsz=PLATE_IMGSZ, conf=PLATE_CONF, verbose=False)[0]
        if not len(res.boxes):
            return None, None
        box = max(res.boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        region = _pad(crop, x1, y1, x2, y2)
        if region.size == 0:
            return None, None

    for height, equalise in _OCR_VARIANTS:
        lines = _ocr_lines(reader, _prepare_crop(region, height, equalise))
        if not lines:
            continue

        # MANY INDIAN PLATES ARE TWO LINES -- "DL 9C AU" above "4743" -- and
        # PaddleOCR returns each line separately. Taking only the longest one
        # reads half the plate and correct_plate() rejects it as too short.
        # Measured on 169 photos, that single mistake caused 34 of 84 failures.
        joined = "".join(t for t, _ in lines)
        mean = sum(sc for _, sc in lines) / len(lines)
        candidates = [(joined, mean)] + sorted(lines, key=lambda ts: -len(ts[0]))

        best: tuple[str, float] | None = None
        for raw, score in candidates:
            plate, penalty = correct_plate(raw)
            if plate is None:
                continue
            conf = round(max(0.0, min(1.0, score - penalty)), 3)
            if best is None or conf > best[1]:
                best = (plate, conf)
        if best is not None:
            return best

    return None, None


def run(camera: str, source: str, fps: int, loop: bool = False) -> None:
    """The real pipeline: decode -> detect+track -> OCR -> emit.

    Order matters, and so does what is DELIBERATELY not done every frame:

      - vehicle detection + BoT-SORT: every processed frame. One model.track()
        call gives boxes, stable ids AND ReID features -- never run a second
        model for the appearance vector.
      - plate detection + OCR: a few frames per track (see _Track.wants_ocr).
      - Re-ID embedding: read on track EXIT only. It is the single moment
        Module C needs it, and the tracker has been maintaining it for free.
    """
    import cv2
    from ultralytics import YOLO

    apply_thread_limit()
    # Reported from inside the child, not from the supervisor that set it: the
    # supervisor can only say what it intended, and an env var that never
    # arrived looks identical from there.
    print(f"threads: {_THREADS or 'library default (all cores)'}", file=sys.stderr)
    vehicle_model = YOLO(VEHICLE_WEIGHTS)
    weights = find_plate_weights()
    plate_model = YOLO(weights) if weights else None
    if plate_model is None:
        print("no trained plate weights; OCR runs on the whole vehicle crop and "
              "will read few plates. Train on Kaggle, see WORKFLOW.md Stage 1.",
              file=sys.stderr)
    else:
        print(f"plate detector: {weights}", file=sys.stderr)

    reader = make_reader()
    if reader is None:
        print("tracking only; no plate will be read", file=sys.stderr)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        emit(event="error", camera_id=camera, detail=f"cannot open source: {source}")
        raise SystemExit(2)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(src_fps / max(fps, 1)))
    print(f"source {src_fps:.0f} fps, processing every {step} frame(s) "
          f"-> {src_fps / step:.1f} fps", file=sys.stderr)

    # "ready" means THE SOURCE IS OPEN, and is emitted here rather than at
    # startup for that reason. The supervisor treats it as proof that a paired
    # phone is really serving video: emitted before this line it would arrive
    # even when the stream could not be opened at all, and a device whose app
    # had been closed would report itself healthy forever.
    emit(event="ready", camera_id=camera, fps=fps)

    tracks: dict[str, _Track] = {}
    features: dict[str, list[float]] = {}
    processed = 0
    raw_no = 0
    # A track that never activates emits NOTHING, silently -- see the frame-rate
    # note in ml/botsort.yaml. Counting the two separately makes that visible
    # instead of looking like "the detector found no vehicles".
    frames_with_boxes = 0
    frames_with_ids = 0
    warned_no_ids = False
    started = time.monotonic()

    def close(key: str, why: str) -> None:
        t = tracks.pop(key, None)
        if t is None:
            return
        embedding = t.embedding or features.get(key) or []
        if len(embedding) < 32:
            # The contract requires a usable vector. Emitting a stub would make
            # layer 2 compare noise and quietly produce wrong matches, which is
            # worse than emitting nothing.
            print(f"track {key} closed with no ReID feature ({why}); dropped",
                  file=sys.stderr)
            return
        plate_text, plate_conf = t.plate()
        emit(event="track_closed", camera_id=camera, track_id=t.track_id,
             entry_time=t.entry_time, exit_time=now_iso(),
             vehicle_type=t.vehicle_type, color=t.colour,
             plate_text=plate_text, plate_conf=plate_conf,
             embedding=embedding, color_hist=t.colour_hist or [0.0] * 32)

    while True:
        # grab() for the frames we are about to throw away, read() only for the
        # ones we keep. At 5 fps off 30 fps footage, five of every six frames
        # are discarded, and read() decodes each one to a numpy array first.
        # grab() advances the decoder without that conversion. Measured, per
        # PROCESSED frame: 4.4 ms of discarded decoding becomes 2.8 ms.
        want = (raw_no + 1) % step == 0
        if want:
            ok, frame = cap.read()
        else:
            ok, frame = cap.grab(), None
        if not ok:
            if loop and cap.get(cv2.CAP_PROP_POS_FRAMES) > 0:
                # A file that ends is not a camera that failed. Rewind so a
                # rehearsal or a demo can run off a clip indefinitely.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                raw_no = 0
                continue
            break
        raw_no += 1
        if not want:
            continue
        processed += 1

        # Pace to the target rate. A file decodes as fast as the CPU allows, so
        # without this a 200 s clip is consumed in 30 s while timestamps come
        # from the wall clock -- which compresses the timeline, and layer 3 then
        # sees a vehicle reach the next camera in 4 s against a 168 s link and
        # correctly rejects a match that really happened. A live stream delivers
        # frames at its own rate and never waits here.
        behind = started + processed / max(fps, 1) - time.monotonic()
        if behind > 0:
            time.sleep(behind)

        ts = now_iso()

        result = vehicle_model.track(
            frame, persist=True, tracker="ml/botsort.yaml",
            classes=list(VEHICLE_CLASSES), imgsz=VEHICLE_IMGSZ, conf=VEHICLE_CONF,
            verbose=False)[0]

        # Harvest appearance vectors while the tracker still holds them. On exit
        # the STrack is gone, so caching here is what makes exit-only Re-ID work.
        tracker = getattr(vehicle_model.predictor, "trackers", [None])[0]
        for strack in getattr(tracker, "tracked_stracks", []):
            feat = getattr(strack, "smooth_feat", None)
            if feat is not None:
                features[f"{RUN_ID}-{int(strack.track_id)}"] = [
                    round(float(v), 6) for v in feat]

        seen: set[str] = set()
        boxes = result.boxes
        if boxes is not None and len(boxes):
            frames_with_boxes += 1
            if boxes.id is not None:
                frames_with_ids += 1
        # Vehicles are being DETECTED but no track is ever confirmed. Nothing
        # downstream can work, and without this warning the sidecar just looks
        # like an empty road. Almost always the frame rate: see ml/botsort.yaml.
        if (not warned_no_ids and frames_with_boxes >= 20 and frames_with_ids == 0):
            warned_no_ids = True
            print(f"WARNING: {frames_with_boxes} frames with vehicles and NOT ONE "
                  f"confirmed track. Association is failing every frame — vehicles "
                  f"are almost certainly moving too far between processed frames. "
                  f"Raise --fps (currently {fps}). See ml/botsort.yaml.",
                  file=sys.stderr)
        if boxes is not None and boxes.id is not None:
            for box in boxes:
                key = f"{RUN_ID}-{int(box.id[0])}"
                seen.add(key)
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                kind = VEHICLE_CLASSES.get(int(box.cls[0]), "car")
                conf = round(float(box.conf[0]), 3)

                emit(event="detection", camera_id=camera, track_id=key, ts=ts,
                     bbox=[x1, y1, x2, y2], conf=conf, vehicle_type=kind)

                t = tracks.get(key)
                if t is None:
                    t = tracks[key] = _Track(key, kind, ts, processed)
                t.last_seen_frame = processed
                t.frames += 1

                crop = frame[max(y1, 0):y2, max(x1, 0):x2]
                area = float((x2 - x1) * (y2 - y1))
                if area > t.best_area:
                    t.best_area = area
                    t.colour = _colour_name(crop)
                    t.colour_hist = _colour_hist(crop)
                if t.wants_ocr(area, x2 - x1):
                    t.attempts += 1
                    t.last_ocr_area = area
                    plate, plate_conf = _read_plate(reader, plate_model, crop)
                    if plate:
                        t.record(plate, plate_conf)

        if feats := features:
            for key in list(tracks):
                if key in seen and key in feats:
                    tracks[key].embedding = feats[key]

        for key in [k for k, t in tracks.items()
                    if processed - t.last_seen_frame > MISS_LIMIT]:
            close(key, "left view")

    cap.release()
    # End of stream is not a reason to lose the tracks still open: their
    # cross-camera match may be the one the demo is about to show.
    for key in list(tracks):
        close(key, "end of stream")
    print(f"[{camera}] {processed} frames processed, {frames_with_boxes} with "
          f"vehicles, {frames_with_ids} with confirmed tracks", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--source", required=True, help="video file or RTSP url")
    ap.add_argument("--fps", type=int, default=5, help="process rate, not source rate")
    ap.add_argument("--loop", action="store_true",
                    help="rewind a video file at EOF instead of exiting")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        return _selfcheck()

    try:
        if args.source == "demo":
            emit(event="ready", camera_id=args.camera, fps=args.fps)
            return demo(args.camera, args.fps)
        run(args.camera, args.source, args.fps, loop=args.loop)
    except SystemExit:
        raise
    except Exception as exc:  # never die silently -- the supervisor restarts us
        emit(event="error", camera_id=args.camera, detail=f"{type(exc).__name__}: {exc}")
        raise


def _selfcheck() -> None:
    """Plate correction: the character-class rules hold and junk is rejected."""
    assert correct_plate("KA01MB1234") == ("KA01MB1234", 0.0)
    # O in a digit slot -> 0; I in a digit slot -> 1. Two fixes, penalty 0.10.
    p, pen = correct_plate("KAO1MB123I")
    assert p == "KA01MB1231", p
    assert abs(pen - 0.10) < 1e-9, pen
    # 0 in an alpha slot -> O.
    assert correct_plate("KA01M01234")[0] == "KA01MO1234"
    assert correct_plate("KA 05 MR 7821")[0] == "KA05MR7821", "spaces must be tolerated"
    assert correct_plate("MH04AQ5678")[0] == "MH04AQ5678"
    # Real formats an earlier, stricter version threw away. Each of these was
    # measured on actual photographs; layer 1 of the association engine cannot
    # match a vehicle whose plate never entered the system.
    assert correct_plate("DL9CAU4743")[0] == "DL9CAU4743", \
        "Delhi district codes are digit-then-letter"
    assert correct_plate("MH05DK101")[0] == "MH05DK101", \
        "the registration number can be three digits"
    assert correct_plate("KL60N5344")[0] == "KL60N5344", "one series letter is valid"
    assert correct_plate("AP07AD5555")[0] == "AP07AD5555"
    # Rejections.
    assert correct_plate("XX01MB1234")[0] is None, "unknown state code must be rejected"
    assert correct_plate("KA01MB12")[0] is None, "too short"
    assert correct_plate("KA01MB12345678")[0] is None, "too long"
    assert correct_plate("K#01MB1234")[0] is None, "punctuation strips to wrong length"
    assert correct_plate("CRETA")[0] is None, "a badge is not a plate"
    assert correct_plate("06A929")[0] is None, "half a plate must not pass"

    # Colour histogram: normalised, right length, and it must not blow up on an
    # empty crop -- a zero-area box is what a box clipped at the frame edge gives.
    import numpy as np
    hist = _colour_hist(np.zeros((0, 0, 3), np.uint8))
    assert len(hist) == 32 and sum(hist) == 0.0, hist
    red = np.zeros((20, 20, 3), np.uint8); red[:, :, 2] = 200
    hist = _colour_hist(red)
    assert len(hist) == 32, len(hist)
    assert abs(sum(hist) - 1.0) < 1e-6, f"histogram must be normalised, got {sum(hist)}"
    assert _colour_name(red) == "red", _colour_name(red)
    assert _colour_name(np.zeros((20, 20, 3), np.uint8)) == "black"
    assert _colour_name(np.full((20, 20, 3), 255, np.uint8)) == "white"

    # Demo and real pipeline must agree on embedding dimension, or the worker's
    # guard fires the moment a demo camera runs beside a real one.
    assert len(_demo_embedding(1)) == DEMO_EMBED_DIM

    # The OCR budget is what keeps the CPU inference budget. If wants_ocr ever
    # returns True unconditionally, the sidecar silently stops meeting 5 FPS.
    big = PLATE_MIN_VEHICLE_PX + 10
    t = _Track("1", "car", now_iso(), 0)
    t.frames = 1
    assert t.wants_ocr(100.0, big), "the first readable frames must get an attempt"
    # A vehicle too far away to carry a legible plate is skipped BEFORE any of
    # the budget is spent. 62% of crops on real footage are in this state.
    assert not t.wants_ocr(100.0, PLATE_MIN_VEHICLE_PX - 1), \
        "a vehicle too small to read must not consume an attempt"
    t.frames, t.attempts, t.last_ocr_area = 50, PLATE_FIRST_FRAMES, 100.0
    assert not t.wants_ocr(101.0, big), "a mid-track frame with no growth must not"

    # AND IT MUST FIRE WHEN THE VEHICLE APPROACHES. The clause was dead for a
    # release -- it compared against best_area, which run() updates before
    # calling this, so the test was `area > area * 1.4` and never true. Every
    # track was then read only in its first three frames, which on real footage
    # are the frames where the vehicle is furthest away and least readable.
    assert t.wants_ocr(100.0 * PLATE_GROWTH + 1, big), \
        "a vehicle that has grown since the last attempt must get another"
    t.attempts = PLATE_MAX_ATTEMPTS
    assert not t.wants_ocr(10_000.0, big), "the attempt ceiling must hold"

    # Two agreeing reads end it, however large the vehicle gets.
    t2 = _Track("2", "car", now_iso(), 0)
    t2.frames, t2.attempts = 50, 1
    for i in range(PLATE_AGREE_STOP - 1):
        t2.record("OD33AU6001", 0.9)
        assert t2.wants_ocr(10_000.0, big), \
            f"{i + 1} agreeing read(s) must not stop a track short of PLATE_AGREE_STOP"
    t2.record("OD33AU6001", 0.9)
    assert not t2.wants_ocr(10_000.0, big), "PLATE_AGREE_STOP agreeing reads must stop it"

    # THE STATE SLOT. A plate 40 px wide reads OD as 00, and rejecting that
    # threw away twenty-six agreeing reads of one vehicle on real footage.
    assert _state_code("OD") == ("OD", 0), "a clean code must cost nothing"
    assert _state_code("00")[0] == "OD", "00 is OD, not OO"
    assert _state_code("0D")[0] == "OD"
    assert _state_code("O0")[0] == "OD"
    assert _state_code("05") is None, "no state code is reachable, so none is invented"
    assert correct_plate("0033AU6001")[0] == "OD33AU6001"
    # A repaired plate must report LOWER confidence than a clean one.
    assert correct_plate("0033AU6001")[1] > correct_plate("OD33AU6001")[1]
    t.attempts = PLATE_MAX_ATTEMPTS
    assert not t.wants_ocr(10_000.0, big), "the per-track attempt ceiling must hold"

    # Voting. Two agreeing reads beat one more-confident read, because OCR
    # errors on a moving vehicle are independent and the truth repeats.
    v = _Track("2", "car", now_iso(), 0)
    v.record("KA01MB1234", 0.80)
    v.record("KA01MB1284", 0.95)
    v.record("KA01MB1234", 0.82)
    plate, conf = v.plate()
    assert plate == "KA01MB1234", f"majority must win, got {plate}"
    assert abs(conf - 0.81) < 1e-6, conf
    v.record("KA01MB1234", 0.79)
    assert not v.wants_ocr(10_000.0, big), \
        "enough agreeing reads must stop further attempts"

    single = _Track("3", "car", now_iso(), 0)
    single.frames, single.last_ocr_area = 50, 100.0
    single.record("KA01MB1234", 0.99)
    assert single.plate()[0] == "KA01MB1234"
    assert single.wants_ocr(10_000.0, big), \
        "ONE confident read must NOT stop attempts -- confident wrong reads are " \
        "exactly the dangerous ones, and a second look is what catches them"
    assert _Track("4", "car", now_iso(), 0).plate() == (None, None)

    # emit() must write to the protocol channel, NOT to sys.stdout — that is the
    # whole point of the split, and a regression here silently corrupts the
    # stream with whatever a dependency decides to print.
    import io
    global _PROTOCOL
    buf, real = io.StringIO(), _PROTOCOL
    _PROTOCOL = buf
    emit(event="ready", camera_id="CAM1", fps=5)
    print("a library printing to stdout must NOT land on the protocol channel")
    _PROTOCOL = real
    line = buf.getvalue()
    assert line.count("\n") == 1, f"stdout leaked into the protocol channel: {line!r}"
    assert line.endswith("\n") and json.loads(line)["event"] == "ready"
    # make_reader() is the only place a PaddleOCR is built, and that is the
    # whole point of it: five scripts used to carry their own copy of these
    # arguments, so a number measured by one did not necessarily describe what
    # the sidecar ran. Checked by reading the source rather than by importing
    # paddle, because this selfcheck must run with no models present.
    from pathlib import Path as _Path
    here = _Path(__file__).parent
    # Assembled at runtime so this line does not match itself.
    needle = "PaddleOCR" + "("
    strays = []
    for f in sorted(here.glob("*.py")):
        n = f.read_text().count(needle)
        if f.name == "sidecar.py":
            n -= 1                              # the one inside make_reader
        if n > 0:
            strays.append(f.name)
    assert not strays, (
        f"{strays} construct PaddleOCR directly; call make_reader() instead, "
        f"or a measurement taken there will not describe what the sidecar runs")

    print("sidecar selfcheck ok", file=sys.stderr)


if __name__ == "__main__":
    main()
