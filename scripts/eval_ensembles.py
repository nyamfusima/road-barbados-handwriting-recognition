"""Evaluate medoid vs ROVER ensembles on validation; writes reports/ensemble_eval.md.

Members are passed on the command line so new ones can be added without editing code:

  python scripts/eval_ensembles.py \
      --val v1=preds/val/v1.csv v2=preds/val/v2.csv r64=preds/val/r64.csv 32b=preds/val/32b.csv \
      --weights v1=1 v2=1.2 r64=1.2 32b=1.3 \
      --truth data/raw/Train.csv --out reports/ensemble_eval.md

For each method, reports: LB score (exact leaderboard formula), mean word/char edits
per line, and our old val error (0.5*WER+0.5*CER). Also evaluates every 3-member
subset, and runs a paired/bootstrap comparison of the best method against the
given-weights medoid (our current production ensemble).
"""
from __future__ import annotations

import argparse
import itertools
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ensemble_rover import ensemble_ids, load_preds, parse_kv_list  # noqa: E402
from metric_lb import old_val_error, per_line_metrics  # noqa: E402


def load_truth(truth_csv: str) -> dict:
    t = pd.read_csv(truth_csv, encoding="utf-8-sig", keep_default_na=False)
    return dict(zip(t.ID, t.Target.astype(str)))


def run_method(member_preds: dict, weights: dict, ids_order: list, truth: dict,
               method: str, margin: float = 0.0, unit: str = "word") -> dict:
    result = ensemble_ids(member_preds, weights, method, margin=margin, unit=unit, ids_order=ids_order)
    refs = [truth[i] for i in ids_order]
    hyps = [result[i] for i in ids_order]
    m = per_line_metrics(refs, hyps)
    n = len(ids_order)
    old = old_val_error(refs, hyps)
    return {
        "lb_score": sum(m["lb_score"]) / n,
        "mean_word_edits": sum(m["word_edits"]) / n,
        "mean_char_edits": sum(m["char_edits"]) / n,
        "old_val_error": old["final"],
        "per_line_lb": m["lb_score"],
    }


def run_single(member_preds: dict, name: str, ids_order: list, truth: dict) -> dict:
    refs = [truth[i] for i in ids_order]
    hyps = [member_preds[name][i] for i in ids_order]
    m = per_line_metrics(refs, hyps)
    n = len(ids_order)
    old = old_val_error(refs, hyps)
    return {
        "lb_score": sum(m["lb_score"]) / n,
        "mean_word_edits": sum(m["word_edits"]) / n,
        "mean_char_edits": sum(m["char_edits"]) / n,
        "old_val_error": old["final"],
        "per_line_lb": m["lb_score"],
    }


def bootstrap_ci(diff: list, n_resamples: int = 1000, seed: int = 0) -> tuple:
    rng = random.Random(seed)
    n = len(diff)
    means = []
    for _ in range(n_resamples):
        s = sum(diff[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * n_resamples)]
    hi = means[int(0.975 * n_resamples) - 1]
    return lo, hi


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", nargs="+", required=True, help="name=path.csv pairs (validation predictions)")
    ap.add_argument("--weights", nargs="+", required=True, help="name=weight pairs")
    ap.add_argument("--truth", default="data/raw/Train.csv")
    ap.add_argument("--out", default="reports/ensemble_eval.md")
    a = ap.parse_args()

    val_paths = parse_kv_list(a.val)
    names = list(val_paths.keys())
    given_weights = {k: float(v) for k, v in parse_kv_list(a.weights).items()}
    equal_weights = {n: 1.0 for n in names}

    member_preds = {n: load_preds(p) for n, p in val_paths.items()}
    ids_order = list(member_preds[names[0]].keys())
    truth = load_truth(a.truth)

    rows = []  # (label, metrics_dict, members_used)

    print(f"members: {names}")
    print(f"given weights: {given_weights}")

    for n in names:
        rows.append((f"single: {n}", run_single(member_preds, n, ids_order, truth), (n,)))

    def ids_order_sub(order, preds):
        common = set.intersection(*(set(d.keys()) for d in preds.values()))
        return [i for i in order if i in common]

    # full member set: medoid (equal / given), rover (equal / given), rover margins 0/0.5, rover-char
    rows.append(("ALL equal-weight medoid", run_method(member_preds, equal_weights, ids_order, truth, "medoid"), tuple(names)))
    rows.append(("ALL given-weight medoid (current prod)", run_method(member_preds, given_weights, ids_order, truth, "medoid"), tuple(names)))
    rows.append(("ALL equal-weight rover (margin 0)", run_method(member_preds, equal_weights, ids_order, truth, "rover", margin=0.0), tuple(names)))
    rows.append(("ALL given-weight rover (margin 0)", run_method(member_preds, given_weights, ids_order, truth, "rover", margin=0.0), tuple(names)))
    rows.append(("ALL equal-weight rover (margin 0.5)", run_method(member_preds, equal_weights, ids_order, truth, "rover", margin=0.5), tuple(names)))
    rows.append(("ALL given-weight rover (margin 0.5)", run_method(member_preds, given_weights, ids_order, truth, "rover", margin=0.5), tuple(names)))
    rows.append(("ALL given-weight rover-char (margin 0)", run_method(member_preds, given_weights, ids_order, truth, "rover", margin=0.0, unit="char"), tuple(names)))

    # every 3-member subset
    for combo in itertools.combinations(names, len(names) - 1):
        sub_preds = {n: member_preds[n] for n in combo}
        sub_weights = {n: given_weights[n] for n in combo}
        label = f"subset {'+'.join(combo)}"
        sub_ids = ids_order_sub(ids_order, sub_preds)
        rows.append((f"{label} medoid", run_method(sub_preds, sub_weights, sub_ids, truth, "medoid"), combo))
        rows.append((f"{label} rover (margin 0)", run_method(sub_preds, sub_weights, sub_ids, truth, "rover", margin=0.0), combo))

    rows.sort(key=lambda r: -r[1]["lb_score"])

    # paired check: best method vs the given-weight medoid baseline
    baseline_label = "ALL given-weight medoid (current prod)"
    baseline = next(r for r in rows if r[0] == baseline_label)[1]
    best_label, best_metrics, _ = rows[0]
    if best_label == baseline_label:
        best_label, best_metrics, _ = rows[1]

    diffs = [b - m for b, m in zip(best_metrics["per_line_lb"], baseline["per_line_lb"])]
    improved = sum(1 for d in diffs if d > 1e-12)
    worsened = sum(1 for d in diffs if d < -1e-12)
    unchanged = len(diffs) - improved - worsened
    mean_diff = sum(diffs) / len(diffs)
    lo, hi = bootstrap_ci(diffs)

    lines = []
    lines.append("# Ensemble evaluation (validation, exact leaderboard metric)\n")
    lines.append(f"Members: {', '.join(names)}. Given weights: {given_weights}.\n")
    lines.append("Sorted by LB score (higher is better). `old_val_error` = 0.5*WER+0.5*CER corpus-level.\n")
    lines.append("| Method | Members | LB score | Mean word edits/line | Mean char edits/line | Old val error |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for label, m, members_used in rows:
        lines.append(f"| {label} | {'+'.join(members_used)} | {m['lb_score']:.5f} | "
                      f"{m['mean_word_edits']:.4f} | {m['mean_char_edits']:.4f} | {m['old_val_error']:.5f} |")

    lines.append("")
    lines.append(f"## Paired check: best ({best_label}) vs {baseline_label}")
    lines.append(f"- Lines improved: {improved}, worsened: {worsened}, unchanged: {unchanged} (of {len(diffs)})")
    lines.append(f"- Mean LB-score difference (best - baseline): {mean_diff:.5f}")
    lines.append(f"- Bootstrap 95% CI (1000 resamples of the 410 lines): [{lo:.5f}, {hi:.5f}]")
    if lo > 0:
        lines.append("- CI excludes 0: the improvement looks reliable, not noise.")
    elif hi < 0:
        lines.append("- CI excludes 0 (negative): the best method is reliably WORSE than the baseline.")
    else:
        lines.append("- CI straddles 0: the difference is not distinguishable from noise at this sample size.")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
