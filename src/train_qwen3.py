"""LoRA fine-tune of Qwen3-VL on the Barbados line images with plain Transformers + PEFT (no Unsloth).

Unsloth's image processor broke on Qwen3-VL, so this script uses Qwen3VLForConditionalGeneration +
AutoProcessor directly. Data loading, split handling, --extra-train, image resizing, prompt, text
cleaning, metric and output files match src/train_qwen.py (PROMPT, clean and load_img are imported
from it).

Requires transformers >= 4.57, peft, accelerate (+ bitsandbytes for --load-4bit).

Colab A100 (bf16):
  python src/train_qwen3.py --data-dir $DATA --img-dir $IMG --split splits/split_v1.csv \
      --out runs/qwen3-8b --img-height 140 --predict-test
Kaggle T4 (fp16 + loss scaling, 4-bit base):
  python src/train_qwen3.py ... --load-4bit --predict-test
32B: --model Qwen/Qwen3-VL-32B-Instruct (needs --load-4bit even on a 40 GB A100).

Smoke test (40 steps, 48 val lines, prints timing + full-run estimate): add --smoke

Images: resized exactly like train_qwen.py (fixed height, width capped at --max-width), then each side
is rounded to the nearest multiple of 32 (Qwen3-VL: 16 px patches, 2x2 merge). The processor's
min/max pixels are set so it does not resize again (checked at start-up). --img-height 140 therefore
gives 128 px lines; use 160 to round up instead.

Outputs in --out: lora/ (adapter), ckpt/ (every 200 steps; --resume continues from the latest),
val_predictions.csv, val_metrics.json, submission.csv (if --predict-test), run_config.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from official_metric import score  # noqa: E402
from metric_lb import lb_score  # noqa: E402
from train_qwen import PROMPT, clean, load_img  # noqa: E402  (also pins CUDA_VISIBLE_DEVICES=0)

PATCH = 32  # 16 px patch x 2x2 spatial merge
LM_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
VISION_TARGETS = ("qkv", "proj", "linear_fc1", "linear_fc2")
# language model projections + vision-tower block attention/MLP linears (not the patch merger)
TARGET_REGEX = (r".*language_model.*\.(" + "|".join(LM_TARGETS) + r")"
                r"|.*visual\.blocks\.\d+\.(attn|mlp)\.(" + "|".join(VISION_TARGETS) + r")")
FALLBACK = "a"


def snap32(img: Image.Image, max_width: int) -> Image.Image:
    w, h = img.size
    nw = min(max(PATCH, round(w / PATCH) * PATCH), max(PATCH, max_width // PATCH * PATCH))
    nh = max(PATCH, round(h / PATCH) * PATCH)
    return img if (nw, nh) == (w, h) else img.resize((nw, nh), Image.BICUBIC)


def prepare_img(path: Path, a) -> Image.Image:
    return snap32(load_img(path, a.img_height, a.max_width), a.max_width)


def user_msg():
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]


def finalise(text: str) -> str:
    text = clean(text)
    return text if text else FALLBACK


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--img-height", type=int, default=140)
    ap.add_argument("--max-width", type=int, default=2240)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--extra-train", default=None, help="CSV with ID,Target of extra (pseudo-labelled) lines whose images are in --img-dir; appended to the training set only")
    ap.add_argument("--predict-test", action="store_true")
    ap.add_argument("--infer-bs", type=int, default=None, help="inference batch size (default 8 with bf16, 4 with fp16)")
    ap.add_argument("--resume", action="store_true", help="continue from the latest checkpoint in {out}/ckpt")
    ap.add_argument("--load-lora", default=None, help="path to a saved LoRA dir; loads it and skips training")
    ap.add_argument("--load-4bit", action="store_true", help="bitsandbytes NF4 base model (fits a 16 GB T4)")
    ap.add_argument("--smoke", action="store_true", help="40 steps on 300 lines, 48 val lines, timing estimate")
    return ap.parse_args(argv)


# ---------------------------------------------------------------- model
def load_model_and_processor(a, dtype, device):
    """Separate so it can be swapped for a tiny model in tests."""
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    kw = dict(dtype=dtype, attn_implementation="sdpa")
    if device == "cuda":
        kw["device_map"] = {"": 0}
    if a.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype)
    model = Qwen3VLForConditionalGeneration.from_pretrained(a.model, **kw)
    processor = AutoProcessor.from_pretrained(a.model)
    if device != "cuda":
        model.to(device)
    return model, processor, torch


def configure_processor(processor, a):
    """Pixel bounds wide enough that the processor never resizes our already-snapped images."""
    max_h = max(PATCH, round(a.img_height / PATCH) * PATCH)
    max_w = max(PATCH, a.max_width // PATCH * PATCH)
    lo, hi = PATCH * PATCH, max_h * max_w
    ip, new = processor.image_processor, {"shortest_edge": lo, "longest_edge": hi}
    ip.size = {**ip.size, **new} if isinstance(ip.size, dict) else type(ip.size)(**{**dict(ip.size), **new})
    for k, v in (("min_pixels", lo), ("max_pixels", hi)):  # older (transformers 4.x) image processors
        if hasattr(ip, k):
            setattr(ip, k, v)


def check_no_resize(processor, img: Image.Image):
    prompt = processor.apply_chat_template(user_msg(), tokenize=False, add_generation_prompt=True)
    enc = processor(text=[prompt], images=[img], return_tensors="pt")
    t, gh, gw = enc["image_grid_thw"][0].tolist()
    patch = processor.image_processor.patch_size
    w, h = img.size
    if (gh * patch, gw * patch) != (h, w):
        raise RuntimeError(f"processor resized {w}x{h} -> {gw * patch}x{gh * patch}; check min/max pixels")
    print(f"[image] {w}x{h} -> grid {gh}x{gw} patches, {gh * gw // 4} vision tokens (no processor resize)", flush=True)


def add_lora(model, a, torch, dtype):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    gc_kwargs = {"use_reentrant": False}
    if a.load_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs=gc_kwargs)
        # prepare_model_for_kbit_training upcasts every non-quantised weight to fp32, including the
        # embeddings and lm_head (~5 GB for 8B). They stay frozen, so cast matrices back to the compute
        # dtype to fit a T4; 1-D norm weights stay fp32.
        for p in model.parameters():
            if p.dtype == torch.float32 and p.ndim >= 2 and not p.requires_grad:
                p.data = p.data.to(dtype)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
        model.enable_input_require_grads()
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    cfg = LoraConfig(r=a.rank, lora_alpha=a.rank, lora_dropout=0.0, bias="none",
                     target_modules=TARGET_REGEX, task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    for p in model.parameters():  # fp32 adapters: needed for fp16 loss scaling, harmless for bf16
        if p.requires_grad:
            p.data = p.data.float()
    n_vis = sum(1 for n, _ in model.named_modules() if ".visual." in n and n.endswith("lora_A"))
    n_lm = sum(1 for n, _ in model.named_modules() if "language_model" in n and n.endswith("lora_A"))
    print(f"[lora] adapted modules: {n_lm} language, {n_vis} vision", flush=True)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------- batches
def train_batch(processor, items, torch):
    """items: list of (PIL image, target). Loss only on the assistant answer (+ end-of-turn token)."""
    tok = processor.tokenizer
    ids, labels, pix, grids, extra = [], [], [], [], {}
    for img, target in items:
        prompt = processor.apply_chat_template(user_msg(), tokenize=False, add_generation_prompt=True)
        full = processor.apply_chat_template(
            user_msg() + [{"role": "assistant", "content": [{"type": "text", "text": target}]}], tokenize=False)
        assert full.startswith(prompt), "chat template: full conversation does not extend the prompt"
        enc = processor(text=[prompt], images=[img], return_tensors="pt")
        ans = tok(full[len(prompt):].rstrip(), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        p_ids = enc["input_ids"][0]
        ids.append(torch.cat([p_ids, ans]))
        labels.append(torch.cat([torch.full_like(p_ids, -100), ans]))
        pix.append(enc["pixel_values"]); grids.append(enc["image_grid_thw"])
        for k, v in enc.items():  # per-token extras such as mm_token_type_ids (0 for answer tokens)
            if k not in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw") and v.ndim == 2 and v.shape[1] == len(p_ids):
                extra.setdefault(k, []).append(torch.cat([v[0], torch.zeros_like(ans, dtype=v.dtype)]))
    n = max(len(x) for x in ids)
    pad = tok.pad_token_id
    input_ids = torch.full((len(ids), n), pad, dtype=torch.long)
    lab = torch.full((len(ids), n), -100, dtype=torch.long)
    att = torch.zeros((len(ids), n), dtype=torch.long)
    ext = {k: torch.zeros((len(ids), n), dtype=v[0].dtype) for k, v in extra.items()}
    for i, (x, y) in enumerate(zip(ids, labels)):  # right padding for training
        input_ids[i, :len(x)] = x; lab[i, :len(y)] = y; att[i, :len(x)] = 1
        for k in ext:
            ext[k][i, :len(x)] = extra[k][i]
    return {"input_ids": input_ids, "attention_mask": att, "labels": lab,
            "pixel_values": torch.cat(pix), "image_grid_thw": torch.cat(grids), **ext}


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


# ---------------------------------------------------------------- checkpoints
def latest_ckpt(ckpt_dir: Path):
    cks = sorted(ckpt_dir.glob("step-*"), key=lambda p: int(p.name.split("-")[1])) if ckpt_dir.exists() else []
    return cks[-1] if cks else None


def save_ckpt(model, opt, sched, scaler, step, ckpt_dir: Path, torch, keep=2):
    d = ckpt_dir / f"step-{step}"
    model.save_pretrained(str(d))
    torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict(), "step": step}, d / "trainer_state.pt")
    for old in sorted(ckpt_dir.glob("step-*"), key=lambda p: int(p.name.split("-")[1]))[:-keep]:
        shutil.rmtree(old, ignore_errors=True)
    print(f"[ckpt] saved {d}", flush=True)


def load_ckpt(model, opt, sched, scaler, d: Path, torch):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    res = set_peft_model_state_dict(model, load_file(str(d / "adapter_model.safetensors")))
    if getattr(res, "unexpected_keys", None):
        raise RuntimeError(f"checkpoint keys do not match the model: {res.unexpected_keys[:5]}")
    st = torch.load(d / "trainer_state.pt", map_location="cpu", weights_only=False)
    opt.load_state_dict(st["optimizer"]); sched.load_state_dict(st["scheduler"]); scaler.load_state_dict(st["scaler"])
    print(f"[resume] from {d} (step {st['step']})", flush=True)
    return st["step"]


# ---------------------------------------------------------------- main
def main(argv=None, loader=None):
    a = parse_args(argv)
    t_start = time.time()

    import torch
    from transformers import get_linear_schedule_with_warmup, set_seed

    set_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = device == "cuda" and not use_bf16
    dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32
    infer_bs = a.infer_bs or (8 if use_bf16 else 4)
    print(f"[setup] device={device} dtype={dtype} 4bit={a.load_4bit} infer_bs={infer_bs}", flush=True)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    img_dir = Path(a.img_dir)
    files = {os.path.splitext(f)[0]: img_dir / f for f in os.listdir(img_dir)}

    df = pd.read_csv(f"{a.data_dir}/Train.csv", encoding="utf-8-sig", keep_default_na=False)
    df = df.merge(pd.read_csv(a.split, encoding="utf-8-sig"), on="ID")
    tr, va = df[df.fold == "train"], df[df.fold == "val"]
    if a.extra_train:
        ex = pd.read_csv(a.extra_train, encoding="utf-8-sig", keep_default_na=False)
        ex = ex[~ex.ID.isin(va.ID) & ex.ID.isin(files.keys()) & (ex.Target.str.strip() != "")]
        tr = pd.concat([tr[["ID", "Target"]], ex[["ID", "Target"]]], ignore_index=True)
        print(f"[extra-train] added {len(ex)} lines -> {len(tr)} training lines", flush=True)
    n_full_train, n_full_val = len(tr), len(va)
    n_test = len(pd.read_csv(f"{a.data_dir}/Test.csv", encoding="utf-8-sig", keep_default_na=False))
    if a.smoke:
        tr, va = tr.sample(min(300, len(tr)), random_state=0), va.sample(min(48, len(va)), random_state=0)
    print(f"[data] train {len(tr)} / val {len(va)} lines", flush=True)

    model, processor, _ = (loader or load_model_and_processor)(a, dtype, device)
    configure_processor(processor, a)
    check_no_resize(processor, prepare_img(files[(va.ID.iloc[0] if len(va) else tr.ID.iloc[0])], a))
    amp = (lambda: torch.autocast("cuda", dtype=dtype)) if device == "cuda" else (lambda: torch.autocast("cpu", enabled=False))

    s_per_step = None
    if a.load_lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.load_lora)
        print(f"[load-lora] {a.load_lora}; skipping training", flush=True)
    else:
        model = add_lora(model, a, torch, dtype)
        model.train()
        processor.tokenizer.padding_side = "right"
        data = list(zip(tr.ID, tr.Target.map(clean)))
        n_micro = math.ceil(len(data) / a.bs)
        steps_per_epoch = math.ceil(n_micro / a.grad_accum)
        total = 40 if a.smoke else math.ceil(steps_per_epoch * a.epochs)
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, math.ceil(0.05 * total), total)
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
        ckpt_dir = out / "ckpt"
        start = 0
        if a.resume:
            last = latest_ckpt(ckpt_dir)
            start = load_ckpt(model, opt, sched, scaler, last, torch) if last else 0
            if not last:
                print("[resume] no checkpoint found; starting from scratch", flush=True)
        print(f"[train] {total} optimizer steps ({steps_per_epoch}/epoch, bs {a.bs} x accum {a.grad_accum}), "
              f"warmup {math.ceil(0.05 * total)}, starting at {start}", flush=True)

        gstep, win_loss, win_t0, timed = 0, [], time.time(), []
        for epoch in range(math.ceil(total / steps_per_epoch)):
            perm = np.random.RandomState(a.seed + epoch).permutation(len(data))
            micro = [perm[i:i + a.bs] for i in range(0, len(perm), a.bs)]
            for g in range(0, len(micro), a.grad_accum):
                if gstep >= total:
                    break
                if gstep < start:  # resume: skip already-trained steps without loading images
                    gstep += 1
                    continue
                t0 = time.time()
                group = micro[g:g + a.grad_accum]
                step_loss = 0.0
                for mb in group:
                    batch = to_device(train_batch(
                        processor, [(prepare_img(files[data[i][0]], a), data[i][1]) for i in mb], torch), device)
                    with amp():
                        loss = model(**batch).loss / len(group)
                    scaler.scale(loss).backward()
                    step_loss += loss.item()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
                gstep += 1
                win_loss.append(step_loss)
                if gstep > start + 1:  # first step includes warm-up/compilation
                    timed.append(time.time() - t0)
                if gstep % 10 == 0 or gstep == total:
                    dt = (time.time() - win_t0) / len(win_loss)
                    print(f"step {gstep}/{total} loss {np.mean(win_loss):.4f} lr {sched.get_last_lr()[0]:.2e} "
                          f"{dt:.2f} s/step", flush=True)
                    win_loss, win_t0 = [], time.time()
                if gstep % 200 == 0 and gstep < total:
                    save_ckpt(model, opt, sched, scaler, gstep, ckpt_dir, torch)
        s_per_step = float(np.mean(timed)) if timed else None
        model.save_pretrained(str(out / "lora")); processor.save_pretrained(str(out / "lora"))
        print(f"[train] done; adapter -> {out / 'lora'}", flush=True)

    # ------------------------------------------------------------ inference
    model.eval()
    model.config.use_cache = True
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = True
    processor.tokenizer.padding_side = "left"
    prompt = processor.apply_chat_template(user_msg(), tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def predict(ids):
        preds = []
        for s in range(0, len(ids), infer_bs):
            chunk = ids[s:s + infer_bs]
            enc = processor(text=[prompt] * len(chunk), images=[prepare_img(files[i], a) for i in chunk],
                            padding=True, return_tensors="pt").to(device)
            with amp():
                gen = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False, num_beams=1,
                                     use_cache=True, pad_token_id=processor.tokenizer.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            preds += [finalise(t) for t in processor.batch_decode(new, skip_special_tokens=True)]
            if (s // infer_bs) % 25 == 0:
                print(f"  predicted {s + len(chunk)}/{len(ids)}", flush=True)
        return preds

    t0 = time.time()
    val_pred = predict(va.ID.tolist())
    s_per_line = (time.time() - t0) / max(1, len(va))
    refs = va.Target.map(clean).tolist()
    metrics = score(refs, val_pred)
    metrics.update(lb_score(refs, val_pred))
    pd.DataFrame({"ID": va.ID.values, "Target": val_pred}).to_csv(out / "val_predictions.csv", index=False)
    (out / "val_metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "run_config.json").write_text(json.dumps(vars(a), indent=2))
    print("VAL:", metrics, flush=True)

    if a.smoke:
        full_steps = math.ceil(math.ceil(math.ceil(n_full_train / a.bs) / a.grad_accum) * a.epochs)
        train_h = full_steps * (s_per_step or 0) / 3600
        infer_h = (n_full_val + n_test) * s_per_line / 3600
        print(f"SMOKE TIMING: {s_per_step or float('nan'):.2f} s/step, {s_per_line:.3f} s/line (infer_bs {infer_bs}); "
              f"wall {(time.time() - t_start) / 60:.1f} min", flush=True)
        print(f"SMOKE ESTIMATE full run: train {full_steps} steps = {train_h:.2f} h + val {n_full_val} & test {n_test} "
              f"lines = {infer_h:.2f} h -> total {train_h + infer_h:.2f} h", flush=True)

    if a.predict_test:
        te = pd.read_csv(f"{a.data_dir}/Test.csv", encoding="utf-8-sig", keep_default_na=False)
        sub = pd.DataFrame({"ID": te.ID, "Target": predict(te.ID.tolist())})
        sub["Target"] = sub.Target.map(finalise)
        sub.to_csv(out / "submission.csv", index=False)
        print(f"wrote {out / 'submission.csv'} ({len(sub)} rows, {int((sub.Target == FALLBACK).sum())} fallbacks)")


if __name__ == "__main__":
    main()
