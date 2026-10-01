"""Tests for run_batches in src/train_qwen.py (no GPU needed). Run: python tests/test_run_batches.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from train_qwen import run_batches  # noqa: E402


def test_order_restored_and_sorted_batches():
    keys = [9.0, 1.0, 5.0, 3.0, 7.0, 2.0, 8.0]
    seen = []

    def fn(idx):
        seen.append([keys[k] for k in idx])
        return [f"r{k}" for k in idx]

    out = run_batches(keys, 3, fn)
    assert out == [f"r{k}" for k in range(len(keys))], out          # original order restored
    assert [len(b) for b in seen] == [3, 3, 1], seen                   # batch sizes
    assert [k for b in seen for k in b] == sorted(keys), seen          # similar keys share a batch


def test_batch_of_one_matches_loop():
    out = run_batches([3, 1, 2], 1, lambda idx: [idx[0] * 10])
    assert out == [0, 10, 20]


def test_fallback_on_failure():
    calls = []

    def fn(idx):
        calls.append(len(idx))
        if len(idx) > 1:
            raise RuntimeError("out of memory")
        return [f"ok{idx[0]}"]

    cleaned = []
    out = run_batches([4, 3, 2, 1], 4, fn, on_error=lambda: cleaned.append(1))
    assert out == ["ok0", "ok1", "ok2", "ok3"], out
    assert calls == [4, 1, 1, 1, 1] and cleaned == [1], (calls, cleaned)


def test_single_item_failure_raises():
    try:
        run_batches([1], 4, lambda idx: (_ for _ in ()).throw(ValueError("bad")))
    except ValueError:
        return
    raise AssertionError("should have raised")


def test_empty():
    assert run_batches([], 8, lambda idx: []) == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
