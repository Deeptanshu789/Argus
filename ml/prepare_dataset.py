#!/usr/bin/env python3
"""Carve a CPU-sized YOLO subset out of a downloaded Indian-plate dataset.

Datasets rot and change layout, so this does NOT download anything. Fetch one
of these by hand into --src first (any of them export in YOLO format):

  https://universe.roboflow.com/search?q=indian+number+plate      (easiest)
  https://huggingface.co/datasets/Dataclusterlabspvtltd/indian-number-plates-dataset
  https://www.kaggle.com/datasets/praveen12345/indian-number-plate-detection

--src must contain images/ and labels/ with matching stems, either flat or
already split into train/val subdirs (both are handled).

Full 15K images is ~4-8 days of CPU training. 3K is ~5-13 hours. See
WORKFLOW.md for why the subset is the right call here.
"""
import argparse
import random
import shutil
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_pairs(src: Path) -> list[tuple[Path, Path]]:
    """Every (image, label) pair under src, at any depth. Unlabeled images are
    dropped — a plate image with no box teaches the detector nothing."""
    labels = {p.stem: p for p in src.rglob("*.txt") if p.name != "classes.txt"}
    pairs = [
        (img, labels[img.stem])
        for img in src.rglob("*")
        if img.suffix.lower() in IMG_EXT and img.stem in labels
    ]
    return sorted(pairs)


def write_split(pairs, dst: Path, split: str) -> None:
    for sub in ("images", "labels"):
        (dst / sub / split).mkdir(parents=True, exist_ok=True)
    for img, lbl in pairs:
        shutil.copy2(img, dst / "images" / split / img.name)
        shutil.copy2(lbl, dst / "labels" / split / f"{img.stem}.txt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True, help="downloaded dataset root")
    ap.add_argument("--dst", type=Path, default=Path("datasets/plates"))
    ap.add_argument("--subset", type=int, default=3000, help="train images")
    ap.add_argument("--val", type=int, default=400, help="val images")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pairs = find_pairs(args.src)
    if not pairs:
        raise SystemExit(f"no image/label pairs under {args.src} — wrong layout?")

    random.Random(args.seed).shuffle(pairs)
    val = pairs[: args.val]
    train = pairs[args.val : args.val + args.subset]
    if len(train) < args.subset:
        print(f"warning: only {len(train)} train images available, wanted {args.subset}")

    if args.dst.exists():
        shutil.rmtree(args.dst)
    write_split(train, args.dst, "train")
    write_split(val, args.dst, "val")

    # Ultralytics resolves relative paths against this file's directory.
    yaml = args.dst / "data.yaml"
    yaml.write_text(
        f"path: {args.dst.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: [license_plate]\n"
    )
    print(f"{len(train)} train / {len(val)} val -> {yaml}")


def _selfcheck() -> None:
    """Split is disjoint, sized right, and never leaks val into train."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        (src / "images").mkdir(parents=True)
        (src / "labels").mkdir(parents=True)
        for i in range(20):
            (src / "images" / f"{i}.jpg").write_bytes(b"x")
            (src / "labels" / f"{i}.txt").write_text("0 0.5 0.5 0.2 0.1\n")
        (src / "images" / "orphan.jpg").write_bytes(b"x")  # no label -> dropped

        pairs = find_pairs(src)
        assert len(pairs) == 20, len(pairs)

        random.Random(0).shuffle(pairs)
        val, train = pairs[:5], pairs[5:15]
        assert len(val) == 5 and len(train) == 10
        assert not ({p[0] for p in val} & {p[0] for p in train}), "val leaked into train"

        dst = Path(td) / "out"
        write_split(train, dst, "train")
        assert len(list((dst / "images" / "train").iterdir())) == 10
        assert len(list((dst / "labels" / "train").iterdir())) == 10
    print("selfcheck ok")


if __name__ == "__main__":
    import sys

    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
