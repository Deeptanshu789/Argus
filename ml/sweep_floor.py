#!/usr/bin/env python3
"""Choose READER_MIN_CONF by measuring, not by guessing.

    ./.venv/bin/python ml/sweep_floor.py

A CRNN reader cannot abstain: a softmax over 37 classes always has an argmax,
so it answers a 15-pixel smear as readily as a clean plate, and what it invents
is grammatical enough that correct_plate() keeps it. The floor in
CrnnReader.predict() is the fix, and this picks the number.

Every threshold is a real ml/score_plates.py run against the hand-labelled
plates, so the numbers here are the same numbers quoted anywhere else -- no
second scoring path that can drift from the first.

Read the table for the KNEE: the highest floor that has not yet started
throwing away correct reads. Precision alone always favours a floor of 1.0,
which answers nothing and is trivially perfect.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
PY = sys.executable


def score(floor: float, model: str | None, reader: str) -> dict[str, int]:
    env = {**os.environ, "ARGUS_READER": reader,
           "ARGUS_READER_MIN_CONF": str(floor)}
    cmd = [PY, str(HERE / "score_plates.py")]
    if model:
        cmd += ["--model", model]
    # stderr, not stdout. ml/sidecar.py points sys.stdout at stderr the moment
    # it is imported, so that no library's chatter can land on the sidecar's
    # JSON protocol channel -- and score_plates.py imports it. Reading .stdout
    # here got an empty string and a table of -1 that still printed a
    # confident-looking precision column.
    r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       cwd=HERE.parent)
    out = r.stdout + r.stderr
    got = {}
    for key in ("correct", "wrong", "missed"):
        m = re.search(rf"^\s+{key}\s+(\d+)", out, re.M)
        got[key] = int(m.group(1)) if m else -1
    return got


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reader", default="runs/reader-k12/best.pt")
    ap.add_argument("--model", default=None, help="plate detector weights")
    ap.add_argument("--floors", default="0.0,0.80,0.85,0.90,0.93,0.95,0.97,0.99")
    args = ap.parse_args()

    floors = [float(f) for f in args.floors.split(",")]
    print(f"reader: {args.reader}\n")
    print(f"{'floor':>6}  {'correct':>7}  {'wrong':>5}  {'missed':>6}  {'precision':>9}")
    rows = []
    for f in floors:
        r = score(f, args.model, args.reader)
        answered = r["correct"] + r["wrong"]
        prec = r["correct"] / answered if answered else 0.0
        rows.append((f, r, prec))
        print(f"{f:6.3f}  {r['correct']:7d}  {r['wrong']:5d}  "
              f"{r['missed']:6d}  {prec:8.0%}")

    base = rows[0][1]["correct"]
    knee = max((r for r in rows if r[1]["correct"] >= base), key=lambda r: r[0])
    print(f"\nhighest floor that keeps all {base} correct reads: {knee[0]:.3f} "
          f"(precision {knee[2]:.0%}, was {rows[0][2]:.0%})")


def demo() -> None:
    """The parsing is the only logic here worth a check: score_plates.py's
    layout changing silently would turn every row into -1 and the sweep would
    still print a confident table of nonsense."""
    sample = ("legible plates            45\n"
              "  correct                  25  (56%)\n"
              "  wrong                    17  (38%)\n"
              "  missed                    3  (7%)\n")
    got = {k: int(re.search(rf"^\s+{k}\s+(\d+)", sample, re.M).group(1))
           for k in ("correct", "wrong", "missed")}
    assert got == {"correct": 25, "wrong": 17, "missed": 3}, got
    print("sweep_floor selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        demo()
    else:
        main()
