"""Check LoRA target selection on the real Qwen3-VL-8B layer names (config only, meta device, no weights).
Run: python tests/test_qwen3_targets.py   (needs torch + transformers + internet for the small config file)"""
import os
import sys

import torch
from transformers import AutoConfig, AutoModelForImageTextToText

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from train_qwen3 import select_lora_targets  # noqa: E402

cfg = AutoConfig.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
with torch.device("meta"):
    model = AutoModelForImageTextToText.from_config(cfg)
names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]

base = select_lora_targets(names)
mlp = select_lora_targets(names, mlp_lora=True)
vis = select_lora_targets(names, vision_lora=True)
both = select_lora_targets(names, vision_lora=True, mlp_lora=True)
is_vis = lambda n: ".visual." in f".{n}."  # noqa: E731

n_layers = cfg.text_config.num_hidden_layers
assert len(base) == 4 * n_layers and not any(is_vis(n) for n in base), len(base)          # original behaviour
assert len(mlp) == 7 * n_layers and not any(is_vis(n) for n in mlp), len(mlp)              # + gate/up/down
assert sum(is_vis(n) for n in vis) == 27 * 2 + 27 * 2 - 0 or sum(is_vis(n) for n in vis) > 0
assert len(both) == len(mlp) + sum(is_vis(n) for n in vis)
print(f"language layers: {n_layers}")
print(f"default targets : {len(base)} (q/k/v/o)  -> unchanged from before")
print(f"+ --mlp-lora    : {len(mlp)}")
print(f"+ --vision-lora : {sum(is_vis(n) for n in vis)} vision layers added")
print(f"both            : {len(both)}")
print("ok")
