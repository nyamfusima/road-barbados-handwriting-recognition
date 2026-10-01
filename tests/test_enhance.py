"""Tests for enhance_img in src/train_qwen.py (no GPU needed). Run: python tests/test_enhance.py"""
import os
import sys

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import laplace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from train_qwen import enhance_img, load_img  # noqa: E402


def _measure(img):
    a = np.asarray(img.convert("L"), dtype=np.float64)
    return laplace(a).var(), a.std()


def _synthetic_faint_line(size=(400, 80)):
    """A low-contrast, slightly blurred synthetic "page": narrow brightness range, soft edges."""
    rng = np.random.RandomState(0)
    a = 180 + rng.normal(0, 3, size[::-1])  # faint, low-contrast "paper"
    a[30:50, 50:350] -= 15  # a faint "stroke" band, low contrast against the paper
    img = Image.fromarray(np.clip(a, 0, 255).astype("uint8")).convert("RGB")
    return img.filter(ImageFilter.GaussianBlur(2))


def test_enhance_increases_blur_and_contrast_metrics():
    img = _synthetic_faint_line()
    b0, c0 = _measure(img)
    enh = enhance_img(img)
    b1, c1 = _measure(enh)
    assert b1 > b0 and c1 > c0, (b0, b1, c0, c1)


def test_enhance_keeps_image_mode_and_size():
    img = _synthetic_faint_line()
    enh = enhance_img(img)
    assert enh.size == img.size and enh.mode == "RGB"


def test_load_img_enhance_flag():
    img = _synthetic_faint_line()
    path = "_t.jpg"
    img.save(path)
    off = load_img(path, 112, 2240, enhance=False)
    on = load_img(path, 112, 2240, enhance=True)
    assert off.size == on.size  # enhancement must not change the resize outcome
    b0, _ = _measure(off)
    b1, _ = _measure(on)
    assert b1 > b0
    os.remove(path)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
