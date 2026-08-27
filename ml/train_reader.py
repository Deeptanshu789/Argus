#!/usr/bin/env python3
"""Train a plate READER -- the model that turns a plate crop into text.

    python ml/train_reader.py --epochs 20
    python ml/train_reader.py --selfcheck        # no data needed, ~20 s

THIS IS THE BOTTLENECK, not the detector. Measured on ml/groundtruth_test50.csv
the detector finds a plate box in 44 of 45 photographs and the system reads 39
of them, so every remaining failure but one is the read. The commonest is `O`
for `D` in the SERIES letters, which unlike the state code have no closed set
for correct_plate() to check against -- nothing downstream can repair it, and
guessing would corrupt genuine `O` series.

A CRNN with CTC loss, not a transformer: 4M parameters, trains on this laptop's
CPU in about an hour, and runs in a few milliseconds per crop against
PaddleOCR's 163 ms. The whole point is that it is small.

TWO-LINE PLATES ARE SPLIT AND LAID SIDE BY SIDE FIRST.

CTC alignment is monotone -- the model emits characters left to right and can
never go back. A two-line plate asks it to read the top row across the image
and then the bottom row across the same columns, which monotone alignment
cannot express at all. So `to_strip()` cuts a two-line plate at the gap between
its rows and concatenates the halves into one wide line. The same transform
runs at training and at inference; if they ever diverge the model reads noise.

THE TRAINING DATA IS SYNTHETIC AND THE FOOTAGE IS NOT.

Both public sets are renders: clean type, even lighting, no motion. Real crops
off a 5 fps street camera are 40 px wide, blurred and glared. A reader trained
on clean renders learns clean renders. --degrade (on by default) is therefore
not a nicety -- it downsamples, blurs, re-compresses and re-lights every sample
so that what the model sees in training resembles what the sidecar will hand it.
Turn it off only to measure how much it is worth.
"""
import argparse
import os
import random
import re
import sys
import time
from pathlib import Path

# Same lever as the sidecar, and set before torch is imported for the same
# reason: the thread count is read from the environment at import, not later.
# Training the reader while the DETECTOR is also training on this machine needs
# it -- two runs that each grab all 16 hardware threads are slower than two that
# share them.
_THREADS = os.environ.get("ARGUS_THREADS")
if _THREADS and _THREADS.isdigit() and int(_THREADS) > 0:
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(_var, _THREADS)

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK = 0                                   # CTC blank is index 0
STOI = {c: i + 1 for i, c in enumerate(CHARS)}
ITOS = {i + 1: c for i, c in enumerate(CHARS)}

H, W = 32, 256                              # network input, after to_strip()
MAX_LEN = 13                                # longest Indian plate incl. 24BH series

# A plate laid out on ONE line is about four and a half times as wide as it is
# tall; on two lines about twice. The midpoint between them is the only number
# here that a different dataset might want to move, so it is a constant.
TWO_LINE_ASPECT = 2.9

# How small degrade() may shrink a crop, as a fraction of W.
#
# Derived from what the sidecar actually hands a reader, not picked. A vehicle
# narrower than PLATE_MIN_VEHICLE_PX (110 px) never yields a readable plate, and
# the plate is roughly a quarter of the vehicle's width, so the smallest plate
# worth reading is about 40 px across. Against a 256 px input that is 0.22.
#
# The first run used 0.12 -- thirty pixels across a ten-character plate, three
# pixels per character. No model can read that and no camera in this system
# produces it, so those samples taught the network to guess from a label it
# could not possibly have derived. It was still at CER 57% after five epochs.
DEGRADE_MIN_SCALE = 0.22

# Width of each direction of the recurrent stack. Stored in the checkpoint so
# that ml/sidecar.py rebuilds whatever was TRAINED, rather than whatever this
# constant happens to say on the day the weights are loaded.
HIDDEN = 192


def to_strip(gray):
    """Plate crop (grayscale ndarray) -> H x W single line, ready for the net.

    A two-line plate becomes its top row followed by its bottom row, so that
    reading order is left to right and CTC can express it."""
    import cv2
    import numpy as np

    h, w = gray.shape[:2]
    if h > 0 and w / h < TWO_LINE_ASPECT:
        # Split at the quietest row in the middle third: on a plate that is the
        # gap between the two lines of text. Falls back to the midpoint, which
        # is where the gap is on a well-framed crop anyway.
        lo = int(h * 0.35)
        ink = np.abs(np.diff(gray.astype(np.int16), axis=1)).sum(axis=1)[lo:int(h * 0.65)]
        if h >= 6 and ink.size:
            # The CENTRE of the quiet run, not its first row. The gap between
            # two lines of text is several rows deep and every one of them ties
            # for the minimum; argmin alone takes the topmost and shaves the
            # ascenders off the bottom line.
            quiet = np.flatnonzero(ink <= ink.min() + 1)
            cut = lo + int(np.median(quiet))
        else:
            cut = h // 2
        top, bottom = gray[:cut], gray[cut:]
        if top.size and bottom.size:
            tgt = max(top.shape[0], bottom.shape[0])
            top = cv2.resize(top, (int(top.shape[1] * tgt / top.shape[0]), tgt))
            bottom = cv2.resize(bottom, (int(bottom.shape[1] * tgt / bottom.shape[0]), tgt))
            gray = np.hstack([top, bottom])

    out = cv2.resize(gray, (W, H), interpolation=cv2.INTER_AREA)
    return out


def degrade(gray, rng: random.Random):
    """Make a clean render look like something a street camera produced.

    The single most important function in this file: without it the model
    scores near-perfectly on held-out synthetic data and reads nothing off real
    footage."""
    import cv2
    import numpy as np

    # Shrink to a realistic plate width and back. A plate 40 px wide in the
    # frame is the case that actually fails, and no amount of blur reproduces
    # what losing the pixels does.
    if rng.random() < 0.8:
        scale = rng.uniform(DEGRADE_MIN_SCALE, 0.75)
        small = cv2.resize(gray, (max(8, int(W * scale)), max(4, int(H * scale))))
        gray = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        k = rng.choice([3, 5])
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    if rng.random() < 0.3:                    # motion, mostly horizontal
        k = rng.choice([3, 5, 7])
        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1.0 / k
        gray = cv2.filter2D(gray, -1, kern)
    if rng.random() < 0.7:                    # lighting and glare
        alpha = rng.uniform(0.5, 1.6)
        beta = rng.uniform(-50, 50)
        gray = np.clip(gray.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        noise = np.random.normal(0, rng.uniform(3, 18), gray.shape)
        gray = np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.4:                    # JPEG, because every frame is one
        q = rng.randint(20, 70)
        ok, enc = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            gray = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return gray


# ----------------------------------------------------------------- the data --

# The label has to LOOK like a registration plate, not merely be alphanumeric.
# Half of one dataset names its files `plate_0000123_aug_std.jpg`; stripping the
# separators leaves PLATE0000123AUGSTD, which is alphanumeric, the right sort of
# length, and complete nonsense. Trained on a few thousand of those the model
# learns to emit letters that were never on any plate.
PLATE_SHAPES = (
    re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{1,4}$"),   # MH 05 DK 101
    re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$"),             # 24 BH 9662 FZ
)


def clean_label(text: str) -> str | None:
    text = "".join(ch for ch in text.upper() if ch.isalnum())
    if not (6 <= len(text) <= MAX_LEN) or any(ch not in STOI for ch in text):
        return None
    return text if any(r.match(text) for r in PLATE_SHAPES) else None


def index_sources(dirs: list[Path]) -> list[tuple[Path, str]]:
    """(image, plate text) for every usable sample under each directory.

    Two conventions cover both public sets and most others: a `labels.txt` of
    `filename<TAB>PLATE` lines, or the plate text as the filename itself."""
    items: list[tuple[Path, str]] = []
    for d in dirs:
        if not d.exists():
            print(f"skipping {d}: does not exist")
            continue
        before = len(items)
        labels: dict[str, str] = {}
        for f in d.rglob("labels.txt"):
            for line in f.read_text(errors="replace").splitlines():
                parts = line.replace("\t", " ").split()
                if len(parts) >= 2:
                    labels[parts[0]] = parts[-1]
        for p in d.rglob("*"):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            raw = labels.get(p.name, p.stem)
            text = clean_label(raw)
            if text:
                items.append((p, text))
        print(f"  {d}: {len(items) - before} sample(s)")
    return items


class Plates:
    """Deliberately not a torch Dataset subclass at import time -- this module
    is imported by the selfcheck before torch is known to be present."""

    def __init__(self, items, train: bool, do_degrade: bool):
        self.items = items
        self.train = train
        self.do_degrade = do_degrade

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import cv2
        import numpy as np
        import torch

        path, text = self.items[i]
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            gray = np.zeros((H, W), np.uint8)
        gray = to_strip(gray)
        if self.train and self.do_degrade:
            gray = degrade(gray, random.Random(random.randrange(1 << 30)))
        x = torch.from_numpy(gray).float().div_(127.5).sub_(1.0).unsqueeze(0)
        y = torch.tensor([STOI[c] for c in text], dtype=torch.long)
        return x, y, len(text)


def collate(batch):
    import torch
    xs, ys, lens = zip(*batch)
    return torch.stack(xs), torch.cat(ys), torch.tensor(lens, dtype=torch.long)


# ---------------------------------------------------------------- the model --

def build_model(hidden: int = HIDDEN):
    import torch.nn as nn

    def block(i, o, pool):
        return [nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(True), nn.MaxPool2d(*pool)]

    class CRNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                *block(1, 32, ((2, 2), (2, 2))),        # 16 x 128
                *block(32, 64, ((2, 2), (2, 2))),       #  8 x  64
                *block(64, 128, ((2, 1), (2, 1))),      #  4 x  64
                *block(128, 256, ((2, 1), (2, 1))),     #  2 x  64
                nn.Conv2d(256, 256, (2, 1)), nn.ReLU(True),   # 1 x 64
            )
            self.rnn = nn.LSTM(256, hidden, num_layers=2, bidirectional=True,
                               batch_first=True)
            self.out = nn.Linear(hidden * 2, len(CHARS) + 1)

        def forward(self, x):
            f = self.cnn(x).squeeze(2).permute(0, 2, 1)   # B, T, C
            f, _ = self.rnn(f)
            return self.out(f)                            # B, T, classes

    return CRNN()


def decode(logits) -> list[str]:
    """Greedy CTC: take the best class per step, drop repeats, drop blanks."""
    best = logits.argmax(-1).cpu().tolist()
    out = []
    for seq in best:
        prev, chars = BLANK, []
        for k in seq:
            if k != prev and k != BLANK:
                chars.append(ITOS[k])
            prev = k
        out.append("".join(chars))
    return out


def edit(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def evaluate(model, loader, device) -> tuple[float, float]:
    """(exact-match accuracy, character error rate)."""
    import torch
    model.eval()
    exact = total = chars = errs = 0
    with torch.no_grad():
        for x, y, lens in loader:
            texts = decode(model(x.to(device)))
            off = 0
            for t, n in zip(texts, lens.tolist()):
                want = "".join(ITOS[k] for k in y[off:off + n].tolist())
                off += n
                total += 1
                exact += t == want
                chars += len(want)
                errs += edit(t, want)
    model.train()
    return exact / max(total, 1), errs / max(chars, 1)


# ----------------------------------------------------------------- training --

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    home = Path.home() / "kaggle-plates"
    ap.add_argument("--src", nargs="+", type=Path,
                    default=[home / "synthetic-license-plates",
                             home / "commercial-vehicle-number-plate"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--val", type=int, default=2500)
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    # REAL CROPS ARE A SEPARATE ARGUMENT, and repeated, because there are two
    # thousand of them against forty-eight thousand renders. Mixed one for one
    # they would be 3% of the data and the model would go on learning the font
    # it already knows. Repeating them buys domain evidence, not new pictures --
    # the augmentation differs every epoch, so the model does not simply
    # memorise the same two thousand crops.
    ap.add_argument("--real", nargs="*", type=Path,
                    default=[Path("datasets/reader-real")])
    ap.add_argument("--real-repeat", type=int, default=12)
    ap.add_argument("--init", type=Path, default=None,
                    help="checkpoint to continue from. Fine-tuning the synthetic "
                         "model on real crops is the intended use; pass a much "
                         "lower --lr with it")
    ap.add_argument("--out", type=Path, default=Path("runs/reader"))
    ap.add_argument("--no-degrade", dest="degrade", action="store_false",
                    help="train on the renders as they are. Measures what the "
                         "degradation augmentation is worth; do not ship it")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        return _selfcheck()

    import torch
    from torch.utils.data import DataLoader

    if _THREADS and _THREADS.isdigit():
        torch.set_num_threads(int(_THREADS))
    # CPU is the DEVELOPMENT case, not the only one. This file was written for
    # a laptop with no NVIDIA GPU and hardcoded "cpu", which is invisible until
    # it runs somewhere that has one: on Kaggle it put a 3.17M-parameter model
    # and 178,266 samples through four CPU cores.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if amp else
                                 f", threads {torch.get_num_threads()}"))

    print("indexing:")
    items = index_sources(args.src)
    real = index_sources(args.real) if args.real else []
    if real:
        items += real * args.real_repeat
        print(f"  {len(real)} real crop(s) repeated {args.real_repeat}x = "
              f"{len(real) * args.real_repeat} of {len(items)} samples "
              f"({len(real) * args.real_repeat / len(items) * 100:.0f}%)")
    if len(items) < 100:
        sys.exit(f"only {len(items)} labelled crops found; check --src")
    random.Random(0).shuffle(items)
    val_n = min(args.val, len(items) // 5)
    val_items, train_items = items[:val_n], items[val_n:]
    print(f"{len(train_items)} train / {len(val_items)} val, "
          f"degrade={'on' if args.degrade else 'OFF'}")

    train_ds = Plates(train_items, True, args.degrade)
    # The validation set is degraded too. Held-out CLEAN renders measure how
    # well the model reads clean renders, which is not the question.
    val_ds = Plates(val_items, True, args.degrade)
    def one_thread(_worker):
        # Each loader worker augments ONE image at a time; letting OpenCV open
        # its own pool per worker oversubscribes the machine and makes the
        # augmentation slower than the network it is feeding.
        import cv2
        cv2.setNumThreads(0)

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, collate_fn=collate,
                          drop_last=True, worker_init_fn=one_thread,
                          persistent_workers=args.workers > 0,
                          pin_memory=amp,
                          prefetch_factor=4 if args.workers else None)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, collate_fn=collate,
                        worker_init_fn=one_thread)
    clean_ld = DataLoader(Plates(val_items, False, False), batch_size=args.batch,
                          shuffle=False, num_workers=args.workers,
                          collate_fn=collate, worker_init_fn=one_thread)

    model = build_model(args.hidden).to(device)
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"continuing from {args.init} "
              f"(epoch {ck.get('epoch', '?')}, exact {ck.get('accuracy', 0) * 100:.1f}%)")
    n_par = sum(p.numel() for p in model.parameters())
    print(f"CRNN {n_par / 1e6:.2f}M parameters, {H}x{W} input, {len(CHARS) + 1} classes")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr * 3, total_steps=args.epochs * len(train_ld), pct_start=0.2)
    ctc = torch.nn.CTCLoss(blank=BLANK, zero_infinity=True)

    args.out.mkdir(parents=True, exist_ok=True)
    # ml/train_progress.py reads the epoch count from here, exactly as it does
    # for an Ultralytics run. One line, and the progress bar works for both.
    (args.out / "args.yaml").write_text(
        f"epochs: {args.epochs}\nbatch: {args.batch}\nlr0: {args.lr}\n"
        f"degrade: {args.degrade}\ntrain: {len(train_items)}\nval: {len(val_items)}\n")
    csv = args.out / "results.csv"
    csv.write_text("epoch,time,train/ctc_loss,metrics/accuracy,"
               "metrics/accuracy_clean,metrics/cer,lr/pg0\n")
    # Selected on (exact-match, then fewest character errors). Exact match
    # alone leaves best.pt stuck on epoch 1 for as long as no whole plate is
    # read -- which for CTC is the first several epochs, because it learns
    # WHERE the characters go before it learns which they are. A run that ended
    # early would have shipped the weights that answer "TP" to everything.
    best_key, t0 = (-1.0, -1e9), time.time()
    scaler = torch.amp.GradScaler(enabled=amp)

    for ep in range(1, args.epochs + 1):
        run_loss = seen = 0
        for step, (x, y, lens) in enumerate(train_ld, 1):
            with torch.autocast(device, enabled=amp):
                logits = model(x.to(device, non_blocking=True))
            # THE LOSS STAYS IN FLOAT32. CTC sums probabilities over every
            # alignment of a 13-character label across 64 timesteps, and in
            # half precision those products underflow to zero -- the loss
            # becomes inf and the run dies in its first epoch.
            logp = logits.float().log_softmax(-1).permute(1, 0, 2)   # T, B, C
            inp_len = torch.full((x.size(0),), logits.size(1), dtype=torch.long)
            loss = ctc(logp, y, inp_len, lens)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * x.size(0)
            seen += x.size(0)
            if step % 50 == 0:
                done = (ep - 1) * len(train_ld) + step
                tot = args.epochs * len(train_ld)
                print(f"  epoch {ep}/{args.epochs} step {step}/{len(train_ld)} "
                      f"loss {run_loss / seen:.4f}  {done / tot * 100:.1f}% overall",
                      flush=True)

        acc, cer = evaluate(model, val_ld, device)
        # Clean as well as degraded. One number cannot separate "the model is
        # weak" from "these samples are unreadable"; the gap between two can.
        cacc, ccer = evaluate(model, clean_ld, device)
        el = time.time() - t0
        with csv.open("a") as fh:
            fh.write(f"{ep},{el:.1f},{run_loss / max(seen, 1):.5f},"
                     f"{acc:.5f},{cacc:.5f},{cer:.5f},{sched.get_last_lr()[0]:.6g}\n")
        print(f"epoch {ep}: loss {run_loss / max(seen, 1):.4f}  "
              f"exact {acc * 100:.1f}% (clean {cacc * 100:.1f}%)  "
              f"CER {cer * 100:.2f}% (clean {ccer * 100:.2f}%)  ({el / 60:.1f} min)",
              flush=True)

        ckpt = {"model": model.state_dict(), "chars": CHARS, "h": H, "w": W,
                "hidden": args.hidden, "accuracy": acc, "accuracy_clean": cacc,
                "cer": cer, "epoch": ep, "degrade": args.degrade}
        torch.save(ckpt, args.out / "last.pt")
        if (acc, -cer) > best_key:
            best_key = (acc, -cer)
            torch.save(ckpt, args.out / "best.pt")

    print(f"\nbest exact-match {best_key[0] * 100:.1f}% "
          f"(CER {-best_key[1] * 100:.2f}%) -> {args.out / 'best.pt'}")
    print("That is accuracy on DEGRADED RENDERS. The number that decides "
          "whether to ship it is\n  python ml/score_plates.py --show   "
          "-- real photographs, real plates.")


def _selfcheck() -> None:
    """Every claim in this file that could silently be false."""
    import numpy as np
    import torch

    # to_strip lays a two-line plate out as one line, and leaves a single line
    # alone. Build a two-line image whose halves differ, and check both halves
    # survive into the left and right of the output.
    two = np.zeros((100, 200), np.uint8)
    two[10:40, 10:190] = 255
    two[60:90, 10:190] = 128
    strip = to_strip(two)
    assert strip.shape == (H, W), strip.shape
    # The bright top row must end up LEFT of the dimmer bottom row. Comparing
    # centroids rather than halves, because the two rows are rarely the same
    # height and so do not occupy equal halves of the strip.
    xs = np.arange(W)[None, :]
    bright = (strip > 200).astype(np.float32)
    dim = ((strip > 90) & (strip < 190)).astype(np.float32)
    assert bright.sum() and dim.sum(), (bright.sum(), dim.sum())
    assert (xs * bright).sum() / bright.sum() < (xs * dim).sum() / dim.sum(), \
        "to_strip put the bottom line before the top one"

    one = np.zeros((40, 200), np.uint8)
    one[10:30, 10:190] = 255
    assert to_strip(one).shape == (H, W)

    # A wide single-line plate must NOT be split: aspect 5 is well past the
    # threshold, so the whole image is one row and nothing is stacked.
    assert 200 / 40 > TWO_LINE_ASPECT

    # Labels: accept real plates, reject junk, strip separators.
    assert clean_label("dl9cau4743") == "DL9CAU4743"
    assert clean_label("MH 05 DK 101") == "MH05DK101"
    assert clean_label("24BH9662FZ") == "24BH9662FZ"
    assert clean_label("AB") is None                    # too short
    assert clean_label("plate_0001") is None            # separators drop out,
    assert clean_label("IMG-20230614") is None          # still not a plate shape
    assert clean_label("PLATE0000123AUGSTD") is None    # a filename, not a plate
    assert clean_label("AS79Q5597") == "AS79Q5597"
    assert clean_label("KA41WV6686!") == "KA41WV6686"

    # degrade() must return something the net can still take.
    rng = random.Random(1)
    for _ in range(20):
        g = degrade(to_strip(two), rng)
        assert g.shape == (H, W) and g.dtype == np.uint8

    # CTC greedy decode collapses repeats and drops blanks.
    logits = torch.full((1, 6, len(CHARS) + 1), -10.0)
    for t, k in enumerate([STOI["K"], STOI["K"], BLANK, STOI["A"], BLANK, STOI["4"]]):
        logits[0, t, k] = 10.0
    assert decode(logits) == ["KA4"], decode(logits)

    assert edit("KA41", "KA41") == 0 and edit("KA41", "KA4I") == 1

    # The model runs, and its time axis is long enough for the longest plate.
    # A CRNN whose output is shorter than the label cannot fit it under CTC and
    # the loss goes to infinity with no other symptom.
    model = build_model()
    out = model(torch.zeros(2, 1, H, W))
    assert out.shape[0] == 2 and out.shape[2] == len(CHARS) + 1, out.shape
    assert out.shape[1] >= 2 * MAX_LEN + 1, (
        f"time axis {out.shape[1]} too short for {MAX_LEN}-character plates")

    # And it learns: 30 steps on four fixed samples must cut the loss.
    ctc = torch.nn.CTCLoss(blank=BLANK, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    xs = torch.randn(4, 1, H, W)
    words = ["KA41WV", "DL9CAU", "MH05DK", "OD02BC"]
    y = torch.tensor([STOI[c] for w in words for c in w])
    lens = torch.tensor([len(w) for w in words])
    first = last = None
    for i in range(30):
        logits = model(xs)
        loss = ctc(logits.log_softmax(-1).permute(1, 0, 2), y,
                   torch.full((4,), logits.size(1), dtype=torch.long), lens)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        first = loss.item() if i == 0 else first
        last = loss.item()
    assert last < first * 0.9, f"loss did not fall: {first:.3f} -> {last:.3f}"

    print(f"selfcheck ok (loss {first:.2f} -> {last:.2f} in 30 steps)")


if __name__ == "__main__":
    main()
