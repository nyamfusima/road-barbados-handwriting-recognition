"""Validate and summarize the raw Zindi competition files before use.

Usage:
    python src/data_prep.py --input-dir data/raw
"""
import argparse
import os

import pandas as pd

EXPECTED_FILES = ["Train.csv", "Test.csv", "SampleSubmission.csv"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory containing the raw Zindi CSV files")
    args = parser.parse_args()

    missing = [f for f in EXPECTED_FILES if not os.path.exists(os.path.join(args.input_dir, f))]
    if missing:
        raise FileNotFoundError(f"Missing expected file(s) in {args.input_dir}: {', '.join(missing)}")

    train = pd.read_csv(os.path.join(args.input_dir, "Train.csv"))
    test = pd.read_csv(os.path.join(args.input_dir, "Test.csv"))
    sample_sub = pd.read_csv(os.path.join(args.input_dir, "SampleSubmission.csv"))

    print(f"Train: {train.shape[0]} rows, columns: {list(train.columns)}")
    print(f"Test:  {test.shape[0]} rows, columns: {list(test.columns)}")
    print(f"SampleSubmission: {sample_sub.shape[0]} rows, columns: {list(sample_sub.columns)}")

    test_ids = set(test["ID"])
    sub_ids = set(sample_sub["ID"])
    if test_ids != sub_ids:
        print(f"WARNING: Test IDs and SampleSubmission IDs differ "
              f"(test-only: {len(test_ids - sub_ids)}, submission-only: {len(sub_ids - test_ids)})")
    else:
        print("Test IDs match SampleSubmission IDs.")


if __name__ == "__main__":
    main()
