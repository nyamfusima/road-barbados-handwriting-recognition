"""Create the fixed validation split and save it as IDs only (safe to commit).

Train.csv has no page/document column, but neighbouring rows are different
crops of the same formulaic phrase (e.g. "Sealed and delivered in the presence
of"), so a random line split leaks phrases into validation. Rows appear to be
ordered by source, so we keep contiguous blocks of rows together and hold out
whole blocks.

Usage:
    python src/make_split.py --train-csv data/Train.csv --out reports/split_v1.json
"""
import argparse
import json
import random

import pandas as pd

SEED = 42
BLOCK = 40        # rows per block; blocks are never divided between train and val
VAL_FRACTION = 0.10


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv", default="data/Train.csv")
    p.add_argument("--out", default="reports/split_v1.json")
    args = p.parse_args()

    ids = pd.read_csv(args.train_csv, encoding="utf-8-sig")["ID"].tolist()
    blocks = [ids[i:i + BLOCK] for i in range(0, len(ids), BLOCK)]
    order = list(range(len(blocks)))
    random.Random(SEED).shuffle(order)
    n_val = max(1, round(len(blocks) * VAL_FRACTION))
    val_blocks = set(order[:n_val])

    val = [i for b, blk in enumerate(blocks) if b in val_blocks for i in blk]
    train = [i for b, blk in enumerate(blocks) if b not in val_blocks for i in blk]
    with open(args.out, "w") as f:
        json.dump({"seed": SEED, "block": BLOCK, "val_fraction": VAL_FRACTION,
                   "train_ids": train, "val_ids": val}, f)
    print(f"train {len(train)}  val {len(val)}  -> {args.out}")


if __name__ == "__main__":
    main()
