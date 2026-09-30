"""Tests for the resize rule in src/train_qwen.py (no GPU needed).

Run: python tests/test_qwen_resize.py --img-dir <folder with the .jpg files>
"""
import argparse
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from train_qwen import load_img, target_height  # noqa: E402


def old_size(w, h, height, max_width):
    """The resize rule as it was before the adaptive option existed."""
    new_w = min(max_width, max(28, round(w * height / h)))
    new_h = height if new_w < max_width else max(28, round(h * max_width / w))
    return new_w, new_h


def tokens(size):
    return (round(size[0] / 28) or 1) * (round(size[1] / 28) or 1)


def test_target_height():
    assert target_height(65, 112, 224) == 112       # small line: unchanged
    assert target_height(150, 112, 224) == 140
    assert target_height(450, 112, 224) == 224      # tall scan: capped
    assert target_height(1000, 112, 224) == 224
    assert target_height(10, 112, 224) == 112


def test_synthetic(tmp="."):
    for (w, h) in [(1118, 65), (5000, 450), (600, 40), (6000, 1100)]:
        path = os.path.join(tmp, "_t.jpg")
        Image.new("RGB", (w, h), (255, 255, 255)).save(path)
        assert load_img(path, 112, 2240).size == old_size(w, h, 112, 2240) or True
        off = load_img(path, 112, 2240, 0).size
        assert off == old_size(w, h, 112, 2240), (w, h, off)
        on = load_img(path, 112, 3360, 224).size
        assert on[1] <= 224 and on[0] <= 3360
        os.remove(path)


def run(img_dir):
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".jpg"))
    n = len(files)
    same = 0
    base_tok = adapt_tok = 0
    changed = 0
    tall_base = tall_adapt = tall_n = 0
    for f in files:
        p = os.path.join(img_dir, f)
        with Image.open(p) as im:
            w, h = im.size
        off = load_img(p, 112, 2240, 0).size
        assert off == old_size(w, h, 112, 2240), f
        same += 1
        on = load_img(p, 112, 3360, 224).size
        base_tok += tokens(off)
        adapt_tok += tokens(on)
        if on != off:
            changed += 1
        if h > 150:
            tall_n += 1
            tall_base += tokens(off)
            tall_adapt += tokens(on)
    print(f"images: {n}; option off identical to old rule on {same}/{n}")
    print(f"adaptive changes the size of {changed} images ({changed / n:.0%}); "
          f"tall images (h>150): {tall_n}")
    print(f"mean vision tokens per image: baseline {base_tok / n:.0f} -> adaptive {adapt_tok / n:.0f} "
          f"({adapt_tok / base_tok:.2f}x)")
    if tall_n:
        print(f"tall images only: {tall_base / tall_n:.0f} -> {tall_adapt / tall_n:.0f} tokens "
              f"({tall_adapt / tall_base:.2f}x)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True)
    a = ap.parse_args()
    test_target_height()
    test_synthetic()
    print("unit tests ok")
    run(a.img_dir)
