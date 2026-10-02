"""Medoid vote (baseline, same logic as src/ensemble.py) and ROVER confusion-network
voting for OCR line predictions.

ROVER, per line:
  1. pivot = the weighted medoid string (lowest weighted sum of blended char+word
     distance to the other members -- identical to src/ensemble.py's pick()).
  2. Align every member's token sequence (words, or characters for --method rover-char)
     onto the pivot's token sequence with Levenshtein alignment, backtracing match /
     substitution / insertion / deletion. This builds a confusion network: one voting
     slot per pivot token, plus one insertion slot between (and around) them.
  3. In each slot, sum member weights per candidate token (a deletion votes for "");
     insertion slots default every member's weight onto "" unless that member actually
     inserts there. Pick the max-weight candidate; a tie, or a margin not cleared
     (--margin), keeps the pivot's token.
  4. Join the chosen non-empty tokens (words with " ", characters with "").

Usage (build a submission):
  python src/ensemble_rover.py --method rover \
      --preds v1=preds/test/v1.csv v2=preds/test/v2.csv r64=preds/test/r64.csv \
      --weights v1=1 v2=1.2 r64=1.2 --out submissions/sub_best.csv [--margin 0.5]

Usage (score against ground truth):
  python src/ensemble_rover.py --method rover --preds v1=preds/val/v1.csv v2=preds/val/v2.csv \
      --weights v1=1 v2=1.2 --truth data/raw/Train.csv --out /tmp/scratch.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metric_lb import lev  # noqa: E402

METHODS = ("medoid", "rover", "rover-char")


def parse_kv_list(items: list[str]) -> dict:
    """['v1=path', 'v2=path'] -> {'v1': 'path', 'v2': 'path'}"""
    out = {}
    for it in items:
        k, v = it.split("=", 1)
        out[k] = v
    return out


def load_preds(path) -> dict:
    d = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    return dict(zip(d.ID, d.Target.astype(str)))


def dist(a: str, b: str) -> float:
    """Same blended distance as src/ensemble.py: 0.5*normalised-CER + 0.5*normalised-WER."""
    c = lev(a, b) / max(len(a), len(b), 1)
    wa, wb = a.split(), b.split()
    w = lev(wa, wb) / max(len(wa), len(wb), 1)
    return 0.5 * c + 0.5 * w


def medoid_pick(cands: list[str], weights: list[float]) -> str:
    """Weighted medoid; ties go to the highest-weight member (src/ensemble.py logic)."""
    order = sorted(range(len(cands)), key=lambda i: -weights[i])
    best, best_score = None, None
    for i in order:
        s = sum(weights[j] * dist(cands[i], cands[j]) for j in range(len(cands)) if j != i)
        if best_score is None or s < best_score - 1e-12:
            best, best_score = cands[i], s
    return best


def _align_ops(ref: list, hyp: list) -> list:
    """Levenshtein-align hyp onto ref. Returns ops in ref order:
      ("match"/"sub", ref_idx, hyp_idx)  ref_idx, hyp_idx are 1-indexed
      ("del", ref_idx, None)             ref token has no hyp counterpart
      ("ins", ref_idx_before, hyp_idx)   hyp token has no ref counterpart;
                                         ref_idx_before in 0..len(ref)
    """
    R, H = len(ref), len(hyp)
    D = [[0] * (H + 1) for _ in range(R + 1)]
    back = [[None] * (H + 1) for _ in range(R + 1)]
    for i in range(1, R + 1):
        D[i][0] = i
        back[i][0] = "u"
    for j in range(1, H + 1):
        D[0][j] = j
        back[0][j] = "l"
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            diag = D[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1)
            up = D[i - 1][j] + 1
            left = D[i][j - 1] + 1
            best = min(diag, up, left)
            D[i][j] = best
            back[i][j] = "d" if best == diag else ("u" if best == up else "l")
    ops = []
    i, j = R, H
    while i > 0 or j > 0:
        d = back[i][j]
        if d == "d":
            ops.append(("match" if ref[i - 1] == hyp[j - 1] else "sub", i, j))
            i, j = i - 1, j - 1
        elif d == "u":
            ops.append(("del", i, None))
            i -= 1
        else:
            ops.append(("ins", i, j))
            j -= 1
    ops.reverse()
    return ops


def _merge_insertions(ops: list) -> list:
    """Consecutive ('ins', same ref_idx_before, ...) ops are contiguous (insertions
    don't move the ref pointer); merge them into one span per member/slot."""
    merged = []
    for op in ops:
        if op[0] == "ins" and merged and merged[-1][0] == "ins" and merged[-1][1] == op[1]:
            merged[-1] = ("ins", op[1], merged[-1][2] + [op[2]])
        elif op[0] == "ins":
            merged.append(("ins", op[1], [op[2]]))
        else:
            merged.append(op)
    return merged


def _pick_slot(votes: dict, pivot_choice: str, margin: float) -> str:
    best_tok = max(votes, key=votes.get)
    if best_tok == pivot_choice:
        return pivot_choice
    if votes[best_tok] > votes.get(pivot_choice, 0.0) + margin:
        return best_tok
    return pivot_choice


def rover_vote(pivot_tokens: list, members: list, margin: float = 0.0, sep: str = " ") -> list:
    """members = [(token_list, weight), ...], aligned against pivot_tokens (itself one
    of the members). Returns the chosen token/span sequence with empties dropped."""
    P = len(pivot_tokens)
    total_w = sum(w for _, w in members)
    word_votes = [dict() for _ in range(P)]
    ins_votes = [{"": total_w} for _ in range(P + 1)]

    for tokens, w in members:
        ops = _merge_insertions(_align_ops(pivot_tokens, tokens))
        for op in ops:
            if op[0] in ("match", "sub"):
                _, ref_idx, hyp_idx = op
                cand = tokens[hyp_idx - 1]
                wv = word_votes[ref_idx - 1]
                wv[cand] = wv.get(cand, 0.0) + w
            elif op[0] == "del":
                ref_idx = op[1]
                wv = word_votes[ref_idx - 1]
                wv[""] = wv.get("", 0.0) + w
            else:
                ref_idx_before, hyp_idxs = op[1], op[2]
                cand = sep.join(tokens[h - 1] for h in hyp_idxs)
                iv = ins_votes[ref_idx_before]
                iv[""] = iv.get("", total_w) - w
                iv[cand] = iv.get(cand, 0.0) + w

    out = [_pick_slot(ins_votes[0], "", margin)]
    for k in range(1, P + 1):
        out.append(_pick_slot(word_votes[k - 1], pivot_tokens[k - 1], margin))
        out.append(_pick_slot(ins_votes[k], "", margin))
    return [t for t in out if t != ""]


def ensemble_ids(member_preds: dict, weights: dict, method: str, margin: float = 0.0,
                  unit: str = "word", ids_order: list | None = None) -> dict:
    """member_preds: {name: {ID: text}}. Returns {ID: text}."""
    names = list(member_preds.keys())
    common = set.intersection(*(set(d.keys()) for d in member_preds.values()))
    if ids_order is None:
        ids_order = list(member_preds[names[0]].keys())
    ids_order = [i for i in ids_order if i in common]

    sep = " " if unit == "word" else ""
    tokenize = (lambda s: s.split()) if unit == "word" else list

    out = {}
    for _id in ids_order:
        cands = [member_preds[n][_id] for n in names]
        ws = [weights[n] for n in names]
        pivot = medoid_pick(cands, ws)
        if method == "medoid":
            out[_id] = pivot
            continue
        pivot_tokens = tokenize(pivot)
        members = [(tokenize(c), w) for c, w in zip(cands, ws)]
        chosen = rover_vote(pivot_tokens, members, margin=margin, sep=sep)
        text = sep.join(chosen).strip()
        out[_id] = text if text != "" else pivot
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=METHODS, default="rover")
    ap.add_argument("--preds", nargs="+", required=True, help="name=path.csv pairs")
    ap.add_argument("--weights", nargs="+", help="name=weight pairs (default: 1 each)")
    ap.add_argument("--margin", type=float, default=0.0, help="ROVER min-support margin")
    ap.add_argument("--truth", help="Train.csv to score the result against (validation)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    preds_paths = parse_kv_list(a.preds)
    names = list(preds_paths.keys())
    weights = {n: 1.0 for n in names}
    if a.weights:
        weights.update({k: float(v) for k, v in parse_kv_list(a.weights).items()})

    member_preds = {n: load_preds(p) for n, p in preds_paths.items()}
    ids_order = list(member_preds[names[0]].keys())
    unit = "char" if a.method == "rover-char" else "word"
    method = "medoid" if a.method == "medoid" else "rover"
    result = ensemble_ids(member_preds, weights, method, margin=a.margin, unit=unit, ids_order=ids_order)

    df = pd.DataFrame({"ID": ids_order, "Target": [result[i] for i in ids_order]})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(df)} rows)")

    if a.truth:
        from metric_lb import lb_score, old_val_error
        t = pd.read_csv(a.truth, keep_default_na=False, encoding="utf-8-sig")[["ID", "Target"]]
        m = df.merge(t, on="ID", suffixes=("", "_true"))
        print(f"scored rows: {len(m)}")
        for n in names:
            hyps = [member_preds[n][i] for i in m.ID]
            print(f"  {n}:", lb_score(list(m.Target_true), hyps), old_val_error(list(m.Target_true), hyps))
        print(f"  ENSEMBLE ({a.method}):", lb_score(list(m.Target_true), list(m.Target)),
              old_val_error(list(m.Target_true), list(m.Target)))


if __name__ == "__main__":
    main()
