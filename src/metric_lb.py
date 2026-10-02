"""Exact Zindi leaderboard formula (from the community "Five things that DON'T work" thread):

  score = 0.5 * (1 - mean_word_edits_per_line / 12) + 0.5 * (1 - mean_char_edits_per_line / 55)

- word edits = Levenshtein distance between whitespace-split token lists (str.split())
- char edits = Levenshtein distance between the raw strings (after .strip())

Also reports our old val error (0.5*WER + 0.5*CER, corpus-level, as in src/official_metric.py)
for comparison. Uses rapidfuzz.distance.Levenshtein if installed, else a pure-Python DP.

Usage
  python src/metric_lb.py --gt data/raw/Train.csv --pred preds/val/v1.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from official_metric import score as _official_score  # noqa: E402

try:
    from rapidfuzz.distance import Levenshtein as _L

    def lev(a, b) -> int:
        return _L.distance(a, b)
except ImportError:  # pragma: no cover
    def lev(a, b) -> int:
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            cur = [i]
            for j, y in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
            prev = cur
        return prev[-1]


def _clean(h) -> str:
    return "" if (h is None or (isinstance(h, float) and pd.isna(h))) else str(h).strip()


def per_line_metrics(refs, hyps) -> dict:
    """Per-line word_edits, char_edits and lb_score lists (same order as input)."""
    refs = [str(r).strip() for r in refs]
    hyps = [_clean(h) for h in hyps]
    word_edits = [lev(r.split(), h.split()) for r, h in zip(refs, hyps)]
    char_edits = [lev(r, h) for r, h in zip(refs, hyps)]
    scores = [0.5 * (1 - we / 12) + 0.5 * (1 - ce / 55) for we, ce in zip(word_edits, char_edits)]
    return {"word_edits": word_edits, "char_edits": char_edits, "lb_score": scores}


def lb_score(refs, hyps) -> dict:
    """Aggregate (mean-over-lines) exact leaderboard score."""
    m = per_line_metrics(refs, hyps)
    n = max(1, len(refs))
    return {
        "lb_score": round(sum(m["lb_score"]) / n, 5),
        "mean_word_edits": round(sum(m["word_edits"]) / n, 4),
        "mean_char_edits": round(sum(m["char_edits"]) / n, 4),
    }


def old_val_error(refs, hyps) -> dict:
    """Corpus-level 0.5*WER + 0.5*CER, as in src/official_metric.py."""
    return _official_score(refs, hyps)


def score_csv(gt_csv, pred_csv, id_col: str = "ID", col: str = "Target") -> dict:
    """Scores pred_csv against the matching rows of gt_csv (inner join on ID), so a
    gt file that is a superset of pred's IDs (e.g. the full Train.csv) is fine --
    only pred's own ID set is scored, not every row of gt."""
    gt = pd.read_csv(gt_csv, encoding="utf-8-sig", keep_default_na=False)
    pr = pd.read_csv(pred_csv, encoding="utf-8-sig", keep_default_na=False)
    m = pr[[id_col, col]].merge(gt[[id_col, col]], on=id_col, how="left", suffixes=("_pred", ""))
    missing = int(m[col].isna().sum())
    refs, hyps = m[col].fillna("").tolist(), m[f"{col}_pred"].tolist()
    out = lb_score(refs, hyps)
    old = old_val_error(refs, hyps)
    out["old_val_error"] = old["final"]
    out["old_cer"] = old["cer"]
    out["old_wer"] = old["wer"]
    out["missing_gt_labels"] = missing
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    a = ap.parse_args()
    print(score_csv(a.gt, a.pred))
