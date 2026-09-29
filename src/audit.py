"""Data audit: files, label lengths, charset, image sizes. Writes a markdown report.

Usage:
    python src/audit.py --data-dir data --image-dir images --out reports/day1_data_audit.md
"""
import argparse
import collections
import os

import numpy as np
import pandas as pd
from PIL import Image


def pct(a, q=(0, 1, 50, 95, 99, 100)):
    return ", ".join(f"p{x}={np.percentile(a, x):.0f}" for x in q)


def sizes(ids, image_dir):
    out = []
    for i in ids:
        path = os.path.join(image_dir, f"{i}.jpg")
        if os.path.exists(path):
            with Image.open(path) as im:
                out.append(im.size)
    return np.array(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--image-dir", default="images")
    p.add_argument("--out", default="reports/day1_data_audit.md")
    args = p.parse_args()

    train = pd.read_csv(os.path.join(args.data_dir, "Train.csv"), encoding="utf-8-sig")
    test = pd.read_csv(os.path.join(args.data_dir, "Test.csv"), encoding="utf-8-sig")
    tgt = train["Target"].fillna("").astype(str)
    lines = ["# Data audit", ""]

    def add(s=""):
        lines.append(s)

    add(f"- Train rows: {len(train)}, Test rows: {len(test)}, ID overlap: {len(set(train.ID) & set(test.ID))}")
    for name, df in (("train", train), ("test", test)):
        missing = [i for i in df.ID if not os.path.exists(os.path.join(args.image_dir, f"{i}.jpg"))]
        add(f"- Missing {name} images: {len(missing)}")
    add(f"- Empty train labels: {(tgt.str.strip() == '').sum()}, duplicate labels: {tgt.duplicated().sum()}")
    add(f"- Labels with leading/trailing whitespace: {(tgt != tgt.str.strip()).sum()}, double spaces: {tgt.str.contains('  ').sum()}")
    add()
    add("## Label length")
    add(f"- Characters: {pct(tgt.str.len())}")
    add(f"- Words: {pct(tgt.str.split().str.len())}")
    add()
    add("## Character set")
    counts = collections.Counter("".join(tgt))
    add(f"{len(counts)} distinct characters. Rare (<=5 occurrences): "
        + " ".join(f"`{c}`={n}" for c, n in sorted(counts.items(), key=lambda x: x[1]) if n <= 5))
    add()
    add("Non-alphanumeric characters (convention markers such as `^`, `~`, `\\`, `|`):")
    add(" ".join(f"`{c}`={n}" for c, n in sorted(counts.items(), key=lambda x: -x[1]) if not c.isalnum() and c != " "))
    add()
    add("## Image sizes (width x height)")
    for name, df in (("train", train), ("test", test)):
        s = sizes(df.ID, args.image_dir)
        add(f"- {name} width: {pct(s[:, 0])}")
        add(f"- {name} height: {pct(s[:, 1])}")
        add(f"- {name} aspect (w/h): {pct(s[:, 0] / s[:, 1])}")
    add()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
