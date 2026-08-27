#!/usr/bin/env python3
"""Show how far a local training run has got, and how long is left.

    python ml/train_progress.py                 # newest run, one snapshot
    python ml/train_progress.py --watch 30      # refresh every 30 s

Ultralytics prints a per-epoch table and a tqdm bar, but neither survives a run
started in the background, and neither answers the only question that matters
during an 80-minute CPU train: how much of it is done. This reads the
results.csv the trainer appends after every epoch, so it works on a run started
from any terminal, or one that has already finished.

Nothing here writes: it is safe to run against a training in progress.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

BLOCKS = " ▁▂▃▄▅▆▇█"


def newest_run(root: Path) -> Path | None:
    runs = [d for d in root.glob("*/") if (d / "results.csv").exists()]
    return max(runs, key=lambda d: (d / "results.csv").stat().st_mtime, default=None)


def read_rows(csv_path: Path) -> list[dict[str, float]]:
    """Every completed epoch. Tolerates a half-written final line -- the trainer
    is appending to this file while we read it."""
    rows = []
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            try:
                rows.append({k.strip(): float(v) for k, v in row.items() if v not in (None, "")})
            except ValueError:
                pass          # the row being written right now
    return rows


def total_epochs(run: Path, fallback: int) -> int:
    args = run / "args.yaml"
    if args.exists():
        for line in args.read_text().splitlines():
            if line.startswith("epochs:"):
                return int(line.split(":", 1)[1])
    return fallback


def spark(xs: list[float], lo: float | None = None, hi: float | None = None) -> str:
    if not xs:
        return ""
    lo = min(xs) if lo is None else lo
    hi = max(xs) if hi is None else hi
    if hi - lo < 1e-9:
        return BLOCKS[4] * len(xs)
    return "".join(BLOCKS[min(8, int((x - lo) / (hi - lo) * 8))] for x in xs)


def hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def render(run: Path) -> str:
    csv_path = run / "results.csv"
    if not csv_path.exists():
        # The trainer writes this only after epoch 1, and on CPU that is
        # several minutes. Saying so beats a traceback.
        return f"{run}: training has not finished its first epoch yet"
    rows = read_rows(csv_path)
    if not rows:
        return f"{run}: no completed epochs yet"

    done = len(rows)
    total = total_epochs(run, done)
    # results.csv only gains a row when an epoch ENDS, so its clock stops for
    # however long the current epoch has been running -- six minutes of "12m45s
    # elapsed, 1h03m left" while both numbers quietly went stale. args.yaml is
    # written when the trainer starts, so its mtime is a real start time.
    started = run / "args.yaml"
    wall = time.time() - started.stat().st_mtime if started.exists() else 0.0
    elapsed = rows[-1].get("time", 0.0)
    if done < total:
        elapsed = max(elapsed, wall)      # a finished run's mtime is long past
    # Rate from the LAST FEW epochs, not the whole run. These machines change
    # load underneath a run -- another training job starts, one finishes -- and
    # an average over everything since the start then predicts an arrival time
    # that nothing about the last hour supports.
    recent = [r["time"] for r in rows[-6:] if "time" in r]
    per = ((recent[-1] - recent[0]) / (len(recent) - 1)
           if len(recent) > 1 else elapsed / done)
    left = max(total - done, 0) * per

    width = 40
    filled = round(width * done / max(total, 1))
    bar = "█" * filled + "·" * (width - filled)

    out = [f"{run}",
           f"[{bar}] {done}/{total} epochs  {done / max(total, 1) * 100:.0f}%",
           f"elapsed {hms(elapsed)}   {per / 60:.1f} min/epoch   "
           + (f"about {hms(left)} left" if done < total else "finished")]
    if done < total:
        out.append(f"epoch {done + 1} in progress "
                   f"({hms(elapsed - rows[-1].get('time', elapsed))} into it)")

    def col(key: str) -> list[float]:
        return [r[key] for r in rows if key in r]

    last = rows[-1]
    # Column-driven, so this works for the YOLO detector (mAP50, box_loss) and
    # the CRNN reader (accuracy, CER, ctc_loss) without knowing about either.
    keys = [k for k in rows[-1] if k not in ("epoch", "time")]
    metrics = [k for k in keys if k.startswith("metrics/")]
    losses = [k for k in keys if "loss" in k]

    out.append("")
    for key in metrics:
        xs = col(key)
        if not xs:
            continue
        # Fixed 0..1 scale: these are all fractions, and a sparkline autoscaled
        # to its own range makes 0.976 -> 0.982 look like a transformation.
        name = key.split("/", 1)[1].replace("(B)", "")
        out.append(f"  {name:<15} {last[key]:.4f}  {xs[-1] - xs[0]:+.4f}  {spark(xs, 0.0, 1.0)}")

    out.append("")
    for key in losses:
        xs = col(key)
        if not xs:
            continue
        name = key.replace("/", " ").replace("_", " ")
        out.append(f"  {name:<15} {last[key]:.4f}  {xs[-1] - xs[0]:+.4f}  {spark(xs)}")

    # Whichever metric the trainer selects best.pt on: mAP for the detector,
    # exact-match accuracy for the reader.
    pick = next((k for k in ("metrics/mAP50(B)", "metrics/accuracy") if k in last), None)
    if pick:
        best = max(rows, key=lambda r: r.get(pick, 0))
        out.append("")
        out.append(f"  best {pick.split('/', 1)[1].replace('(B)', '')} "
                   f"{best.get(pick, 0):.4f} at epoch {int(best['epoch'])}")
    # Neither mAP nor held-out accuracy is the number that decides this project;
    # say so where they are read.
    out.append("  These rank runs against each other. Whether a model is BETTER is")
    out.append("  ml/score_plates.py -- correct reads of real photographs.")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=None, help="run directory")
    ap.add_argument("--root", type=Path, default=Path("runs/detect"))
    ap.add_argument("--watch", type=int, default=0, help="seconds between refreshes")
    args = ap.parse_args()

    while True:
        run = args.run or newest_run(args.root)
        if run is None:
            sys.exit(f"no run with a results.csv under {args.root}")
        text = render(run)
        if args.watch:
            print("\033[2J\033[H" + time.strftime("%H:%M:%S") + "\n" + text, flush=True)
            time.sleep(args.watch)
        else:
            print(text)
            return


if __name__ == "__main__":
    main()
