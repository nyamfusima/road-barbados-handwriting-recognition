"""Derive the extra split files used by some ensemble members from split_v1.csv.

  split_v1_clean.csv  split_v1 minus the 3 corroborated bad training rows (see README).
  split_full.csv      every Train.csv line as "train" (4,098), plus 40 val lines from split_v1
                      duplicated as "val" -- a leaky sanity check only, not a real validation.

src/train_qwen.py merges the split on ID, so a duplicated ID with fold "val" is scored as val
while its "train" row still trains. The 40 val lines only affect the reported (leaky) val
score, not the trained model.

Usage:
  python scripts/make_splits.py --split data/splits/split_v1.csv --out-dir data/splits
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BAD_IDS = ["79tMUVyfIdy3GzkG", "F8DYDDp2AvW9Dytw", "JU7lRwk3jKkus24Z"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="data/splits/split_v1.csv")
    ap.add_argument("--out-dir", default="data/splits")
    ap.add_argument("--n-leaky-val", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    sp = pd.read_csv(a.split, encoding="utf-8-sig")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    clean = sp[~sp.ID.isin(BAD_IDS)]
    clean.to_csv(out / "split_v1_clean.csv", index=False)
    print(f"split_v1_clean: dropped {len(sp) - len(clean)} rows -> "
          f"{(clean.fold == 'train').sum()} train / {(clean.fold == 'val').sum()} val")

    leaky = sp[sp.fold == "val"].sample(a.n_leaky_val, random_state=a.seed).assign(fold="val")
    full = pd.concat([sp.assign(fold="train"), leaky], ignore_index=True)
    full.to_csv(out / "split_full.csv", index=False)
    print(f"split_full: {(full.fold == 'train').sum()} train / {(full.fold == 'val').sum()} leaky val")


if __name__ == "__main__":
    main()
