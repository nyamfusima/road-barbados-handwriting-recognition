"""Tests for src/official_metric.py. Run: python -m pytest tests -q   (or: python tests/test_official_metric.py)"""
import os
import random
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import official_metric as m  # noqa: E402

TRAIN = os.path.join(os.path.dirname(__file__), "..", "data", "Train.csv")


def test_edit_distance_known_values():
    assert m.edit_distance("kitten", "sitting") == 3
    assert m.edit_distance("", "abc") == 3 and m.edit_distance("abc", "") == 3
    assert m.edit_distance("same", "same") == 0
    assert m.edit_distance(["a", "b"], ["a", "c", "b"]) == 1


def test_hand_computed_score():
    # ref chars: 5 + 3 = 8; word counts 1 + 1 = 2
    # "hello"->"hallo" 1 char err, 1 word err; "cat"->"" 3 char err, 1 word err
    s = m.score(["hello", "cat"], ["hallo", ""])
    assert abs(s["cer_corpus"] - 4 / 8) < 1e-12
    assert abs(s["wer_corpus"] - 2 / 2) < 1e-12
    assert abs(s["final"] - (0.5 * 0.5 + 0.5 * 1.0)) < 1e-12
    # length-weighted means long lines count more than the plain mean of per-line CERs
    assert abs(s["cer_line"] - (1 / 5 + 3 / 3) / 2) < 1e-12 and s["cer_line"] != s["cer_corpus"]


def test_edge_cases():
    assert m.score(["a b"], ["a b"])["final"] == 0
    assert m.score(["a b c"], [""])["final"] == 1.0
    assert m.score([""], ["x"])["cer_corpus"] > 0  # empty reference must not divide by zero
    assert m.score(["ſtrange ^ & ~ ſ"], ["ſtrange ^ & ~ ſ"])["final"] == 0  # unicode and markers


def test_matches_independent_libraries():
    import jiwer
    from rapidfuzz.distance import Levenshtein
    refs = pd.read_csv(TRAIN, encoding="utf-8-sig").Target.tolist()
    rng = random.Random(0)
    hyps = []
    for r in refs:
        chars = list(r)
        for _ in range(rng.randint(0, 6)):
            k = rng.randrange(len(chars) + 1)
            op = rng.choice("ids")
            if op == "i":
                chars.insert(k, rng.choice("abcde ^"))
            elif op == "d" and chars:
                chars.pop(min(k, len(chars) - 1))
            elif chars:
                chars[min(k, len(chars) - 1)] = rng.choice("xyz")
        hyps.append("".join(chars))
    s = m.score(refs, hyps)
    # jiwer silently strips/collapses whitespace, so only compare it on normalised text
    norm = lambda xs: [" ".join(x.split()) for x in xs]  # noqa: E731
    sn = m.score(norm(refs), norm(hyps))
    assert abs(sn["cer_corpus"] - jiwer.cer(norm(refs), norm(hyps))) < 1e-9
    assert abs(s["wer_corpus"] - jiwer.wer(refs, hyps)) < 1e-9
    total = sum(Levenshtein.distance(r, h) for r, h in zip(refs, hyps))
    assert abs(s["cer_corpus"] - total / sum(len(r) for r in refs)) < 1e-9


def test_csv_interface_and_missing_ids():
    df = pd.DataFrame({"ID": ["a", "b", "c"], "Target": ["one two", "three", "four five six"]})
    with tempfile.TemporaryDirectory() as d:
        gt, pr = os.path.join(d, "gt.csv"), os.path.join(d, "pr.csv")
        df.to_csv(gt, index=False, encoding="utf-8-sig")
        df.iloc[[2, 0]].to_csv(pr, index=False)  # different order, ID "b" missing
        w = m.compute_weighted_wer(gt, pr, weight_factor=0.5)
        c = m.compute_weighted_cer(gt, pr, weight_factor=1.0)
    assert w["matched"] == 2 and w["total_reference"] == 3
    assert w["score"] == 0 and c["score"] == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
