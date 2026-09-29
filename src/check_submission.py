"""Check a submission file against SampleSubmission.csv before uploading to Zindi.

Usage:
    python src/check_submission.py --sub submission.csv --sample data/SampleSubmission.csv
Exits with status 1 if any hard error is found.
"""
import argparse
import sys

import pandas as pd


def check(sub_path, sample_path):
    errors, warnings = [], []
    sub = pd.read_csv(sub_path, encoding="utf-8-sig", keep_default_na=False)
    sample = pd.read_csv(sample_path, encoding="utf-8-sig", keep_default_na=False)

    if list(sub.columns) != ["ID", "Target"]:
        errors.append(f"columns are {list(sub.columns)}, expected ['ID', 'Target']")
        return errors, warnings
    if len(sub) != len(sample):
        errors.append(f"{len(sub)} rows, expected {len(sample)}")
    if sub["ID"].duplicated().any():
        errors.append(f"{sub['ID'].duplicated().sum()} duplicated IDs")
    missing, extra = set(sample["ID"]) - set(sub["ID"]), set(sub["ID"]) - set(sample["ID"])
    if missing:
        errors.append(f"{len(missing)} IDs missing from submission, e.g. {sorted(missing)[:3]}")
    if extra:
        errors.append(f"{len(extra)} unknown IDs in submission, e.g. {sorted(extra)[:3]}")

    tgt = sub["Target"].astype(str)
    empty = (tgt.str.strip() == "").sum()
    if empty:
        warnings.append(f"{empty} empty predictions")
    padded = (tgt != tgt.str.strip()).sum()
    if padded:
        warnings.append(f"{padded} predictions with leading/trailing whitespace")
    dbl = tgt.str.contains("  ").sum()
    if dbl:
        warnings.append(f"{dbl} predictions with double spaces")
    longest = tgt.str.len().max()
    if longest > 150:
        warnings.append(f"longest prediction is {longest} chars (train max is 120): possible runaway generation")
    return errors, warnings


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sub", required=True)
    p.add_argument("--sample", default="data/SampleSubmission.csv")
    args = p.parse_args()
    errs, warns = check(args.sub, args.sample)
    for w in warns:
        print("WARNING:", w)
    for e in errs:
        print("ERROR:", e)
    print("OK" if not errs else "FAILED")
    sys.exit(1 if errs else 0)
