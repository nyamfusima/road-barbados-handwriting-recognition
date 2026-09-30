"""Line-level MBR ensemble for OCR predictions.

For each ID, pick the candidate transcription that is closest (weighted
0.5*char-distance + 0.5*word-distance) to all other candidates -- the
"medoid". With 3+ models this behaves like a majority vote but never
invents text that no model produced.

Usage (validation scoring):
  python src/ensemble.py --preds a_val.csv b_val.csv c_val.csv \
      --weights 1 1 1.5 --truth Train.csv --out ens_val.csv
Usage (test):
  python src/ensemble.py --preds a_test.csv b_test.csv c_test.csv \
      --weights 1 1 1.5 --out submission_ens.csv
"""
import argparse

import pandas as pd

try:
    from rapidfuzz.distance import Levenshtein as L
except ImportError:  # pragma: no cover
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rapidfuzz"], check=True)
    from rapidfuzz.distance import Levenshtein as L


def dist(a: str, b: str) -> float:
    """Metric-shaped distance: 0.5 * normalised CER + 0.5 * normalised WER."""
    c = L.distance(a, b) / max(len(a), len(b), 1)
    wa, wb = a.split(), b.split()
    w = L.distance(wa, wb) / max(len(wa), len(wb), 1)
    return 0.5 * c + 0.5 * w


def pick(cands, weights):
    """Return the weighted medoid; ties go to the highest-weight model."""
    best, best_score = None, None
    order = sorted(range(len(cands)), key=lambda i: -weights[i])
    for i in order:
        s = sum(weights[j] * dist(cands[i], cands[j]) for j in range(len(cands)) if j != i)
        if best_score is None or s < best_score - 1e-12:
            best, best_score = cands[i], s
    return best


def score(pred, truth):
    """Length-weighted 0.5*CER + 0.5*WER (same as official_metric.py)."""
    ce = sum(L.distance(p, t) for p, t in zip(pred, truth))
    cn = sum(len(t) for t in truth)
    we = sum(L.distance(p.split(), t.split()) for p, t in zip(pred, truth))
    wn = sum(len(t.split()) for t in truth)
    c, w = ce / cn, we / wn
    return {"cer": round(c, 5), "wer": round(w, 5), "final": round(0.5 * c + 0.5 * w, 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float)
    ap.add_argument("--truth", help="Train.csv to score validation predictions")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    w = a.weights or [1.0] * len(a.preds)
    assert len(w) == len(a.preds), "one weight per prediction file"

    dfs = [pd.read_csv(p, keep_default_na=False, encoding="utf-8-sig")[["ID", "Target"]]
           .rename(columns={"Target": f"m{i}"}) for i, p in enumerate(a.preds)]
    df = dfs[0]
    for d in dfs[1:]:
        df = df.merge(d, on="ID", how="inner")
    assert len(df) == len(dfs[0]), f"ID mismatch: {len(df)} merged vs {len(dfs[0])}"
    cols = [f"m{i}" for i in range(len(a.preds))]

    df["Target"] = [pick([r[c] for c in cols], w) for _, r in df.iterrows()]
    df["Target"] = df["Target"].where(df["Target"].str.strip() != "", df[cols[int(max(range(len(w)), key=w.__getitem__))]])

    if a.truth:
        t = pd.read_csv(a.truth, keep_default_na=False, encoding="utf-8-sig")[["ID", "Target"]]
        m = df.merge(t, on="ID", suffixes=("", "_true"))
        print(f"scored rows: {len(m)}")
        for i, c in enumerate(cols):
            print(f"  model {i} ({a.preds[i]}):", score(list(m[c]), list(m.Target_true)))
        print("  ENSEMBLE:", score(list(m.Target), list(m.Target_true)))
        picked = pd.Series([next(i for i, c in enumerate(cols) if r[c] == r.Target) for _, r in df.iterrows()])
        print("  picked per model:", picked.value_counts().sort_index().to_dict())

    df[["ID", "Target"]].to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
