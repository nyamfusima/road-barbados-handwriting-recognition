"""Error analysis for validation predictions (aggregate stats + short snippets).

Usage (Kaggle, CPU is enough):
  python src/error_analysis.py --pred /kaggle/input/.../val_predictions.csv \
      --train /kaggle/input/.../barbados-hw/Train.csv
"""
import argparse
import collections
import re

import pandas as pd
from rapidfuzz.distance import Levenshtein as L


def cer_w(P, T):
    return sum(L.distance(p, t) for p, t in zip(P, T)) / sum(len(t) for t in T)


def wer_w(P, T):
    return sum(L.distance(p.split(), t.split()) for p, t in zip(P, T)) / sum(len(t.split()) for t in T)


def score(P, T):
    c, w = cer_w(P, T), wer_w(P, T)
    return round(c, 4), round(w, 4), round(0.5 * c + 0.5 * w, 4)


WHAT_IF = {
    "case-insensitive": lambda s: s.lower(),
    "no punctuation": lambda s: re.sub(r"[^\w\s^]", "", s),
    "collapse whitespace": lambda s: " ".join(s.split()),
    "no ^ marker": lambda s: s.replace("^", ""),
    "u/v i/j merged + lower": lambda s: s.lower().replace("v", "u").replace("j", "i"),
    "ALL of the above": lambda s: " ".join(re.sub(r"[^\w\s]", "", s.lower().replace("v", "u").replace("j", "i")).split()),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--worst", type=int, default=12)
    a = ap.parse_args()

    tr = pd.read_csv(a.train, encoding="utf-8-sig", keep_default_na=False)
    pr = pd.read_csv(a.pred, keep_default_na=False)
    df = pr.merge(tr, on="ID", suffixes=("_p", "_t")).rename(columns={"Target_p": "p", "Target_t": "t"})
    P, T = list(df.p), list(df.t)
    print("rows:", len(df))
    print("\nBASE cer/wer/final:", score(P, T))

    print("\nWHAT-IF (error if this category were perfect):")
    for k, f in WHAT_IF.items():
        print(f"  {k:26s}", score([f(x) for x in P], [f(x) for x in T]))

    df["ed"] = [L.distance(p, t) for p, t in zip(P, T)]
    df["cer_line"] = df.ed / df.t.str.len().clip(lower=1)
    tot = df.ed.sum()
    s = df.sort_values("ed", ascending=False)
    for q in (0.05, 0.10, 0.20):
        n = int(len(df) * q)
        print(f"worst {int(q * 100)}% of lines ({n}) hold {s.ed.head(n).sum() / tot:.0%} of char errors")
    print("lines perfect:", round((df.ed == 0).mean(), 3), "| lines CER>50%:", int((df.cer_line > 0.5).sum()))

    d = df.p.str.len() - df.t.str.len()
    print("\nlen diff pred-true: mean", round(d.mean(), 2), "| >20 longer:", int((d > 20).sum()), "| >20 shorter:", int((d < -20).sum()))
    rep = df.p.str.contains(r"\b(?:\w{3,})\b", regex=True) & df.p.apply(
        lambda x: bool(re.search(r"\b(\w{3,})\b(?:\W+\1\b){2,}", x)))
    print("repetition loops:", int(rep.sum()))

    df["bucket"] = pd.cut(df.t.str.len(), [0, 30, 50, 70, 90, 200])
    print("\nCER by true-length bucket:")
    for b, g in df.groupby("bucket", observed=True):
        print(f"  {str(b):10s} n={len(g):4d} cer={g.ed.sum() / g.t.str.len().sum():.3f}")

    cc, wc = collections.Counter(), collections.Counter()
    for p, t in zip(P, T):
        for op, i, j in L.editops(t, p):
            cc[(t[i] if op != "insert" else "∅", p[j] if op != "delete" else "∅")] += 1
        tw, pw = t.split(), p.split()
        if len(tw) == len(pw):
            wc.update((x, y) for x, y in zip(tw, pw) if x != y)
    print("\nTOP 25 char edits (true->pred):", [(f"{x}->{y}", n) for (x, y), n in cc.most_common(25)])
    print("TOP 20 word swaps (true->pred):", [(f"{x}->{y}", n) for (x, y), n in wc.most_common(20)])

    print(f"\nWORST {a.worst}:")
    for _, r in s.head(a.worst).iterrows():
        print(f"  ed={r.ed:3d} | T: {r.t[:70]}\n         | P: {r.p[:70]}")


if __name__ == "__main__":
    main()
