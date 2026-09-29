"""Measure how much validation text is also in the training set, for our block split
versus random line splits of the same size.

For every validation line we find the closest training label by character edit distance
(normalised by the validation line's length, i.e. a CER). If most validation lines have a
near-identical training line, validation scores are optimistic: the model can partly
memorise phrases instead of reading handwriting.

"Lookup CER" is the CER you would get by predicting each validation line as its closest
training label, using no image at all. It is a rough upper bound on the memorisation gain.

Usage:
    python src/split_leakage.py --train-csv data/Train.csv --split reports/split_v1.json
Needs rapidfuzz (pip install rapidfuzz); falls back to a slow pure-Python version.
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import clean_label  # noqa: E402

try:
    from rapidfuzz.distance import Levenshtein
    from rapidfuzz.process import cdist
    FAST = True
except ImportError:
    from official_metric import edit_distance
    FAST = False

THRESHOLDS = (0.0, 0.05, 0.10, 0.20, 0.30)


def nearest(val_texts, train_texts):
    """For each validation text: (edit distance to closest train text, index of that text)."""
    if FAST:
        d = cdist(val_texts, train_texts, scorer=Levenshtein.distance, dtype=np.int32, workers=-1)
        idx = d.argmin(axis=1)
        return d[np.arange(len(val_texts)), idx], idx
    dist, idx = [], []
    for v in val_texts:
        best, bi = 10 ** 9, -1
        for j, t in enumerate(train_texts):
            if abs(len(t) - len(v)) >= best:
                continue
            e = edit_distance(v, t)
            if e < best:
                best, bi = e, j
        dist.append(best)
        idx.append(bi)
    return np.array(dist), np.array(idx)


def measure(val_texts, train_texts):
    dist, idx = nearest(val_texts, train_texts)
    lens = np.array([max(1, len(v)) for v in val_texts])
    rate = dist / lens
    out = {f"share_cer<={t:.2f}": float((rate <= t).mean()) for t in THRESHOLDS}
    out["mean_nearest_cer"] = float(rate.mean())
    out["lookup_cer"] = float(dist.sum() / lens.sum())  # corpus CER of "predict closest train label"
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv", default="data/Train.csv")
    p.add_argument("--split", default="reports/split_v1.json")
    p.add_argument("--random-trials", type=int, default=5)
    p.add_argument("--out", default="reports/split_leakage.md")
    args = p.parse_args()

    df = pd.read_csv(args.train_csv, encoding="utf-8-sig", keep_default_na=False)
    text = {i: clean_label(t) for i, t in zip(df.ID, df.Target.astype(str))}
    with open(args.split) as f:
        split = json.load(f)
    val_ids, train_ids = split["val_ids"], split["train_ids"]
    print(f"engine: {'rapidfuzz' if FAST else 'pure python (slow)'}  train {len(train_ids)}  val {len(val_ids)}")

    rows = {"block split (split_v1)": measure([text[i] for i in val_ids], [text[i] for i in train_ids])}

    all_ids = list(df.ID)
    trials = []
    for seed in range(args.random_trials):
        rng = random.Random(seed)
        shuffled = all_ids[:]
        rng.shuffle(shuffled)
        v, t = shuffled[:len(val_ids)], shuffled[len(val_ids):]
        trials.append(measure([text[i] for i in v], [text[i] for i in t]))
    rows[f"random line split (mean of {args.random_trials})"] = {
        k: float(np.mean([r[k] for r in trials])) for k in trials[0]}

    cols = list(next(iter(rows.values())).keys())
    lines = ["# Split leakage", "",
             "For each validation line, the closest training label by character edit distance, "
             "as a fraction of the validation line's length (a CER). Higher shares at low CER mean "
             "more phrases are shared between train and validation, so validation scores are more optimistic.", "",
             "| split | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, r in rows.items():
        lines.append(f"| {name} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
    lines += ["",
              "`lookup_cer` is the CER of predicting each validation line as its closest training label "
              "with no image at all: the amount of validation score that memorised phrases alone can explain."]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
