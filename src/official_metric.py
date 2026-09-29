"""PROVISIONAL metric until Zindi's wer.py / cer.py are available.

- CER and WER via Levenshtein distance, no text normalisation (labels keep historic spelling).
- Reports corpus-level (length-weighted) and per-line averages; calibrate against the leaderboard.
- Final = cer_weight * CER + wer_weight * WER (0.5 / 0.5 per Zindi evaluation page).

Usage
  python src/official_metric.py --gt val_gt.csv --pred val_pred.csv
"""
from __future__ import annotations

import argparse

import pandas as pd


def _lev(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _rates(refs, hyps, unit):
    split = (lambda s: s.split()) if unit == "word" else list
    errs = [_lev(split(r), split(h)) for r, h in zip(refs, hyps)]
    lens = [max(1, len(split(r))) for r in refs]
    corpus = sum(errs) / sum(lens)
    per_line = sum(min(1.0, e / n) for e, n in zip(errs, lens)) / max(1, len(refs))
    return corpus, per_line


def score(refs, hyps, cer_weight: float = 0.5, wer_weight: float = 0.5) -> dict:
    refs = [str(r) for r in refs]
    hyps = ["" if (h is None or (isinstance(h, float) and pd.isna(h))) else str(h) for h in hyps]
    cer_c, cer_l = _rates(refs, hyps, "char")
    wer_c, wer_l = _rates(refs, hyps, "word")
    return {
        "cer": round(cer_c, 5), "wer": round(wer_c, 5),
        "final": round(cer_weight * cer_c + wer_weight * wer_c, 5),
        "cer_per_line": round(cer_l, 5), "wer_per_line": round(wer_l, 5),
        "final_per_line": round(cer_weight * cer_l + wer_weight * wer_l, 5),
        "final_50_50": round(0.5 * cer_c + 0.5 * wer_c, 5),
    }


def score_csv(gt_csv, pred_csv, id_col: str = "ID", col: str = "Target", **kw) -> dict:
    gt = pd.read_csv(gt_csv, encoding="utf-8-sig", keep_default_na=False)
    pr = pd.read_csv(pred_csv, encoding="utf-8-sig", keep_default_na=False)
    m = gt.merge(pr[[id_col, col]], on=id_col, how="left", suffixes=("", "_pred"))
    missing = int(m[f"{col}_pred"].isna().sum())
    out = score(m[col], m[f"{col}_pred"].fillna(""), **kw)
    out["missing_predictions"] = missing
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--cer-weight", type=float, default=0.5)
    ap.add_argument("--wer-weight", type=float, default=0.5)
    a = ap.parse_args()
    print(score_csv(a.gt, a.pred, cer_weight=a.cer_weight, wer_weight=a.wer_weight))
