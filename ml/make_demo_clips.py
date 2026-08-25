#!/usr/bin/env python3
"""Build synthetic multi-camera clips so the video path can be run today.

    ./.venv/bin/python ml/make_demo_clips.py --plates ~/indian-plates/test/images

Writes demo/cam1.mp4 .. demo/cam4.mp4. Each clip pans real photographed vehicles
across a frame, and the SAME vehicle appears on more than one camera at
staggered times — which is the only thing that makes a cross-camera match
possible to observe.

WHY THIS EXISTS, AND WHAT IT IS NOT
-----------------------------------
It is a test fixture, not the demo. A panning still has no perspective change,
no occlusion, no motion blur and no lighting shift, so it flatters the detector
and the tracker. Numbers measured on it are an upper bound.

It IS enough to prove the plumbing: decode -> YOLO -> BoT-SORT -> plate -> OCR
-> track_closed -> worker -> Postgres -> Module C, on video, with real plates
and real OCR. That path is otherwise untestable until someone sources footage,
and the backup demo video (CLAUDE.md, hour 34) still needs real traffic.
"""
import argparse
import sys
from pathlib import Path

FPS = 25
# How long a vehicle takes to cross the frame. This is not a cosmetic choice.
#
# Association at 5 FPS works on box overlap between CONSECUTIVE PROCESSED
# frames. A vehicle that crosses a 960 px frame in 4 s moves ~60 px per
# processed frame; at a typical 110 px box that is an IoU of 0.28, below what
# BoT-SORT will match, and every frame starts a new track. 10 s puts it at
# ~24 px, an IoU near 0.65, which matches comfortably.
#
# Ten seconds is also what real footage looks like: a vehicle at 30 km/h
# covers a 25 m field of view in about 3 s, and an ANPR camera watches a
# longer stretch than that.
SECONDS_PER_VEHICLE = 10
W, H = 960, 540

# Which vehicle appears on which camera, and how many seconds into the clip.
#
# THE OFFSETS ARE THE ROAD GRAPH. db/setup.ts seeds CAM1->CAM3 at 168 s and
# CAM1->CAM2 at 200 s, and layer 3 of the association engine accepts a match
# only within 0.5x-2.0x of that. Staging the same vehicle 4 s apart on two
# cameras does not make an easy match — it makes an IMPOSSIBLE one, which the
# engine correctly vetoes, and the fixture then proves nothing.
#
# So vehicle 0 leaves CAM1 at t=5, reaches CAM3 at t=173 (168 s later) and CAM2
# at t=205 (200 s later). Every sidecar starts at once and runs in real time, so
# clip time IS wall-clock time between cameras.
LEG_CAM1_CAM3 = 168
LEG_CAM1_CAM2 = 200
FILLER_EVERY = 24          # unrelated traffic, so the clips are not empty road

# The tracked vehicle. Exactly these three appearances, nowhere else.
TRACKED = {
    "cam1": [(0, 5)],
    "cam3": [(0, 5 + LEG_CAM1_CAM3)],
    "cam2": [(0, 5 + LEG_CAM1_CAM2)],
    "cam4": [],            # never on vehicle 0's route: a true negative
}

# Filler traffic, so the clips are not empty road between the three moments
# that matter. Vehicle 0 is EXCLUDED from it: a second, unscheduled appearance
# of the tracked vehicle would give the association engine a real ambiguity
# to resolve, and the fixture could then no longer say what a match proves.
ROUTE = {cam: list(passes) for cam, passes in TRACKED.items()}
for _cam, _first in (("cam1", 1), ("cam3", 2), ("cam2", 3), ("cam4", 1)):
    for _n in range(9):
        _idx = 1 + (_first + _n) % 3          # 1, 2 or 3 — never 0
        ROUTE[_cam].append((_idx, _n * FILLER_EVERY + 12))


def road_background():
    """A grey road with lane markings. Not scenery — something for the tracker's
    camera-motion estimate to sit still against."""
    import cv2
    import numpy as np
    bg = np.full((H, W, 3), 90, np.uint8)
    cv2.rectangle(bg, (0, 0), (W, int(H * 0.35)), (150, 160, 170), -1)   # sky
    for x in range(0, W, 90):                                            # lane dashes
        cv2.rectangle(bg, (x, H - 60), (x + 45, H - 52), (230, 230, 230), -1)
    cv2.line(bg, (0, int(H * 0.35)), (W, int(H * 0.35)), (70, 80, 70), 6)
    return bg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plates", type=Path, default=Path.home() / "indian-plates/test/images",
                    help="folder of vehicle photos to composite")
    ap.add_argument("--out", type=Path, default=Path("demo"))
    args = ap.parse_args()

    import cv2

    if not args.plates.is_dir():
        sys.exit(f"no such folder: {args.plates}\n"
                 "Point --plates at any folder of vehicle photos.")
    photos = sorted(p for p in args.plates.rglob("*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:4]
    if len(photos) < 4:
        sys.exit(f"need at least 4 photos in {args.plates}, found {len(photos)}")

    vehicles = []
    for p in photos:
        img = cv2.imread(str(p))
        if img is None:
            sys.exit(f"cannot read {p}")
        # Scale so the vehicle occupies a plausible slice of the frame. Too
        # large and the plate is trivially readable; too small and OCR has
        # nothing to work with at any distance.
        # The photo is a whole scene, so the car inside it is smaller than the
        # pasted region. Scale generously: a plate must be readable, which on a
        # real ANPR camera means the vehicle fills a good part of the frame.
        scale = (H * 0.8) / img.shape[0]
        vehicles.append(cv2.resize(img, None, fx=scale, fy=scale))
    print(f"vehicles: {', '.join(p.name for p in photos)}")

    args.out.mkdir(parents=True, exist_ok=True)
    bg = road_background()

    for cam, appearances in ROUTE.items():
        length = max(t for _, t in appearances) + SECONDS_PER_VEHICLE + 1
        dst = args.out / f"{cam}.mp4"
        writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        if not writer.isOpened():
            sys.exit(f"cannot write {dst} (no mp4v encoder in this OpenCV build)")

        for frame_no in range(length * FPS):
            frame = bg.copy()
            t = frame_no / FPS
            for idx, start in appearances:
                if not start <= t < start + SECONDS_PER_VEHICLE:
                    continue
                car = vehicles[idx]
                ch, cw = car.shape[:2]
                # Left to right across the full width, so the track has real
                # displacement for estimateSpeed() to measure.
                progress = (t - start) / SECONDS_PER_VEHICLE
                x = int(-cw + progress * (W + cw))
                y = H - ch - 30
                x0, x1 = max(x, 0), min(x + cw, W)
                if x1 <= x0:
                    continue
                frame[y:y + ch, x0:x1] = car[:, x0 - x:x1 - x]
            writer.write(frame)
        writer.release()
        print(f"  {dst}  {length}s, {len(appearances)} vehicle pass(es)")

    print("\nrun the pipeline on them:")
    print("  ARGUS_CAMERAS='CAM1=demo/cam1.mp4,CAM3=demo/cam3.mp4,CAM2=demo/cam2.mp4' "
          "npm run worker")


if __name__ == "__main__":
    main()
