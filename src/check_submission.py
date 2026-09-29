"""Validate a submission CSV against SampleSubmission.csv before uploading.

Usage
  python src/check_submission.py submissions/trocr-large-v1.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--sample", default=None, help="default: first SampleSubmission.csv under data/raw")
    args = ap.parse_args()

    sample_path = Path(args.sample) if args.sample else next(Path("data/raw").rglob("SampleSubmission.csv"), None)
    if sample_path is None:
        sys.exit("[error] SampleSubmission.csv not found")
    sample = pd.read_csv(sample_path, keep_default_na=False)
    sub = pd.read_csv(args.submission, keep_default_na=False)
    errors, warns = [], []

    if list(sub.columns) != list(sample.columns):
        errors.append(f"columns {list(sub.columns)} != expected {list(sample.columns)}")
    idc = sample.columns[0]
    if idc in sub.columns:
        if len(sub) != len(sample):
            errors.append(f"rows {len(sub)} != expected {len(sample)}")
        if sub[idc].duplicated().any():
            errors.append(f"{int(sub[idc].duplicated().sum())} duplicate ids")
        missing = set(sample[idc].astype(str)) - set(sub[idc].astype(str))
        extra = set(sub[idc].astype(str)) - set(sample[idc].astype(str))
        if missing: errors.append(f"{len(missing)} ids missing")
        if extra: errors.append(f"{len(extra)} unexpected ids")
    for c in sub.columns[1:]:
        s = sub[c].astype(str)
        n_empty = int((s.str.strip() == "").sum())
        if n_empty: warns.append(f"`{c}`: {n_empty} empty predictions")
        if s.str.contains(r"[\n\r\t]").any(): warns.append(f"`{c}`: contains newlines/tabs")
        if (s != s.str.strip()).any(): warns.append(f"`{c}`: leading/trailing whitespace")

    for w in warns: print(f"[warn] {w}")
    for e in errors: print(f"[error] {e}")
    print("[PASS] submission format OK" if not errors else "[FAIL] fix errors before uploading")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
