#!/usr/bin/env python3
"""Regenerate STATE_PRIOR in ml/sidecar.py from the reader's training crops.

    ./.venv/bin/python ml/state_prior.py --labels datasets/reader-all/labels.txt

The table is a constant in ml/sidecar.py rather than a file beside the weights
because it has to travel with the code that reads it: a deploy that shipped the
weights and not the prior would silently apply the wrong correction, and there
would be nothing in any log to say so. Paste the output over the existing
block whenever the crop set is rebuilt.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path


def prior(labels: list[str]) -> list[tuple[str, float]]:
    """Fraction of crops carrying each state code, commonest first."""
    c = Counter(p[:2] for p in labels if len(p) >= 2 and p[:2].isalpha())
    n = sum(c.values())
    return [(k, v / n) for k, v in sorted(c.items(), key=lambda kv: -kv[1])]


def render(items: list[tuple[str, float]]) -> str:
    out, line = ["STATE_PRIOR = {"], "   "
    for k, v in items:
        piece = f' "{k}": {v:.5f},'
        if len(line) + len(piece) > 76:
            out.append(line)
            line = "   "
        line += piece
    out += [line, "}"]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path,
                    default=Path("datasets/reader-all/labels.txt"))
    args = ap.parse_args()
    rows = [l.rstrip("\n").split("\t")[-1]
            for l in args.labels.read_text().splitlines() if l.strip()]
    items = prior(rows)
    print(f"# {len(rows)} crops, {len(items)} states, "
          f"commonest {items[0][0]} at {items[0][1]:.1%}", file=sys.stderr)
    print(render(items))


def demo() -> None:
    items = prior(["MH01AB1234", "MH02CD5678", "OD05BQ2430", "7BADPLATE"])
    assert items[0] == ("MH", 2 / 3), items
    assert dict(items)["OD"] == 1 / 3, items
    assert "7B" not in dict(items), "a non-alphabetic pair is not a state code"
    assert render(items).startswith("STATE_PRIOR = {")
    print("state_prior selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        demo()
    else:
        main()
