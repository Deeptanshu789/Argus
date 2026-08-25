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
SECONDS_PER_VEHICLE = 4
W, H = 960, 540

# Which cameras each vehicle is seen by, and how many seconds into the clip.
# Vehicle 0 travels CAM1 -> CAM3 -> CAM2, matching the road graph seeded by
# db/setup.ts, so layer 3 has a link to check the timing against.
ROUTE = {
    "cam1": [(0, 0), (1, 5), (3, 10)],
    "cam3": [(0, 4), (2, 8)],
    "cam2": [(0, 9), (1, 13)],
    "cam4": [(2, 2), (3, 6)],
}


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
        scale = (H * 0.45) / img.shape[0]
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
