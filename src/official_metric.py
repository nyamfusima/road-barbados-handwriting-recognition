"""PROVISIONAL CER / WER scoring, standing in for the official Zindi code.

The Zindi starter pack's eval_metrics.py imports `wer.py` and `cer.py`
(compute_weighted_wer / compute_weighted_cer with a `weight_factor`), but those
files were never shipped. The meaning of `weight_factor` is undocumented, so it
is accepted here and IGNORED. Do not tune toward it until it is confirmed.

Two variants are reported so we can calibrate against the public leaderboard:
  * corpus: total edit distance / total reference length (long lines count more)
  * line:   mean of per-line error rates (every line counts equally)

Two ways of combining CER and WER are reported until the Evaluation tab is confirmed:
  * 0.7 * CER + 0.3 * WER   (from the starter code)
  * 0.5 * CER + 0.5 * WER   (from the original team brief)

The function names and result keys of compute_weighted_wer / compute_weighted_cer
match what the starter eval_metrics.py expects, so the official files can replace
this module without touching callers.

Usage:
    python src/official_metric.py --gt data/Train.csv --pred preds.csv
"""
import argparse

import pandas as pd

COMBINATIONS = {"0.7/0.3": (0.7, 0.3), "0.5/0.5": (0.5, 0.5)}


def edit_distance(a, b):
    """Levenshtein distance between two sequences (strings or word lists)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _units(text, level):
    return text if level == "char" else text.split()


def _rates(refs, hyps, level):
    """Return (corpus_rate, mean_per_line_rate) at 'char' or 'word' level."""
    errs, lens, per_line = 0, 0, []
    for r, h in zip(refs, hyps):
        ru, hu = _units(r, level), _units(h, level)
        e = edit_distance(ru, hu)
        errs += e
        lens += len(ru)
        per_line.append(e / max(1, len(ru)))
    return errs / max(1, lens), sum(per_line) / max(1, len(per_line))


def per_line_errors(refs, hyps):
    """Per-line CER and WER lists, for worst-line analysis."""
    cers = [edit_distance(r, h) / max(1, len(r)) for r, h in zip(refs, hyps)]
    wers = [edit_distance(r.split(), h.split()) / max(1, len(r.split())) for r, h in zip(refs, hyps)]
    return cers, wers


def score(refs, hyps):
    """All variants for aligned lists of reference and hypothesis strings."""
    cer_corpus, cer_line = _rates(refs, hyps, "char")
    wer_corpus, wer_line = _rates(refs, hyps, "word")
    out = {"cer_corpus": cer_corpus, "wer_corpus": wer_corpus, "cer_line": cer_line, "wer_line": wer_line}
    for basis, (c, w) in (("corpus", (cer_corpus, wer_corpus)), ("line", (cer_line, wer_line))):
        for name, (wc, ww) in COMBINATIONS.items():
            out[f"final_{basis}_{name}"] = wc * c + ww * w
    return out


def _load(gt_csv, pred_csv):
    gt = pd.read_csv(gt_csv, encoding="utf-8-sig").fillna("")
    pred = pd.read_csv(pred_csv, encoding="utf-8-sig").fillna("")
    merged = gt.merge(pred, on="ID", suffixes=("_gt", "_pred"))
    return merged["Target_gt"].astype(str).tolist(), merged["Target_pred"].astype(str).tolist(), len(gt)


def _compute(gt_csv, pred_csv, level):
    refs, hyps, total = _load(gt_csv, pred_csv)
    corpus, _ = _rates(refs, hyps, level)
    return {"score": corpus, "matched": len(refs), "total_reference": total}


def compute_weighted_wer(gt_csv, pred_csv, weight_factor=0.5):
    """Starter-compatible interface. `weight_factor` is ignored (see module docstring)."""
    return _compute(gt_csv, pred_csv, "word")


def compute_weighted_cer(gt_csv, pred_csv, weight_factor=0.5):
    """Starter-compatible interface. `weight_factor` is ignored (see module docstring)."""
    return _compute(gt_csv, pred_csv, "char")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt", required=True)
    p.add_argument("--pred", required=True)
    args = p.parse_args()
    refs, hyps, total = _load(args.gt, args.pred)
    if len(refs) != total:
        print(f"WARNING: only {len(refs)}/{total} reference IDs found in predictions")
    for k, v in score(refs, hyps).items():
        print(f"{k:24s} {v:.4f}")
