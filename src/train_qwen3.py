"""Independent Transformers/PEFT Qwen3-VL fine-tuning pipeline.

Install in a fresh GPU runtime with:
    pip install -U "transformers>=4.57.0" peft accelerate bitsandbytes pandas pillow

The competition resize, prompt, and label cleaning intentionally match train_qwen.py.
The Qwen processor's visual resize is disabled to prevent a second resize; inputs are
aligned to its patch grid with edge padding only (no resampling).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metric_lb import lb_score  # noqa: E402
from official_metric import score as official_score  # noqa: E402

PROMPT = ("Transcribe this handwritten line from an 18th or 19th century Barbados archive exactly as written. "
          "Keep the original spelling, capitalisation, abbreviations and symbols such as ^ ~ & _. "
          "Output only the transcription.")
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def clean(text: str) -> str:
    text = str(text).strip().split("\n")[0]
    return re.sub(r"\s+", " ", text).strip()


def load_img(path: Path, height: int, max_width: int) -> tuple[Image.Image, tuple[int, int]]:
    """Exact competition resize from train_qwen.py (PIL BICUBIC, capped width)."""
    img = Image.open(path).convert("RGB")
    original = img.size
    w, h = img.size
    new_w = min(max_width, max(28, round(w * height / h)))
    new_h = height if new_w < max_width else max(28, round(h * max_width / w))
    return img.resize((new_w, new_h), Image.BICUBIC), original


def align_image_grid(image: Image.Image, factor: int) -> Image.Image:
    """Pad to processor patch-grid dimensions without applying a second resize."""
    w, h = image.size
    target_w = ((w + factor - 1) // factor) * factor
    target_h = ((h + factor - 1) // factor) * factor
    if (target_w, target_h) == (w, h):
        return image
    return ImageOps.expand(image, (0, 0, target_w - w, target_h - h), fill=(255, 255, 255))


def user_message(image: Image.Image) -> list[dict]:
    return [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--img-dir", default="images")
    ap.add_argument("--split", default="data/splits/split_v1.csv")
    ap.add_argument("--out", default="runs/qwen3-vl-8b-sft")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--img-height", type=int, default=140)
    ap.add_argument("--max-width", type=int, default=2240)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--predict-test", action="store_true")
    ap.add_argument("--infer-bs", type=int, default=1)
    ap.add_argument("--resume", default=None, help="checkpoint directory or PEFT adapter directory")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--vision-lora", action="store_true", help="also adapt compatible visual attention projections")
    return ap.parse_args()


def main() -> None:
    a = parse_args()
    if a.bs < 1 or a.grad_accum < 1 or a.infer_bs < 1:
        raise ValueError("batch size, gradient accumulation, and inference batch size must be positive")

    import torch
    import transformers
    from packaging.version import Version
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    import peft
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if Version(transformers.__version__) < Version("4.57.0"):
        raise RuntimeError(f"Transformers >=4.57.0 is required; found {transformers.__version__}")

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    root = Path(__file__).resolve().parents[1]
    data_dir = Path(a.data_dir)
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    img_dir = Path(a.img_dir)
    if not img_dir.is_absolute():
        img_dir = root / img_dir
    split_path = Path(a.split)
    if not split_path.is_absolute():
        split_path = root / split_path
    out = Path(a.out)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / "Train.csv", encoding="utf-8-sig", keep_default_na=False)
    split = pd.read_csv(split_path, encoding="utf-8-sig", keep_default_na=False)
    df = train.merge(split, on="ID", validate="one_to_one")
    tr = df[df.fold == "train"].copy()
    va = df[df.fold == "val"].copy()
    if not len(tr) or not len(va):
        raise ValueError(f"Empty train/validation split: train={len(tr)}, val={len(va)}")
    if a.smoke:
        tr = tr.sample(min(300, len(tr)), random_state=0).reset_index(drop=True)
        va = va.sample(min(48, len(va)), random_state=0).reset_index(drop=True)

    # IDs correspond to image stems in this competition, matching train_qwen.py.
    files = {p.stem: p for p in img_dir.iterdir() if p.is_file()}
    missing = sorted((set(tr.ID.astype(str)) | set(va.ID.astype(str))) - files.keys())
    if missing:
        raise FileNotFoundError(f"Missing image files for IDs, first examples: {missing[:5]}")

    cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda else None
    bf16 = bool(cuda and torch.cuda.is_bf16_supported() and not a.load_4bit)
    dtype = torch.bfloat16 if bf16 else (torch.float16 if cuda else torch.float32)
    load_kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    if a.load_4bit:
        if not cuda:
            raise RuntimeError("--load-4bit requires CUDA and bitsandbytes")
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        load_kwargs["device_map"] = {"": 0}
    elif cuda:
        load_kwargs["device_map"] = {"": 0}

    processor = AutoProcessor.from_pretrained(a.model)
    image_processor = processor.image_processor
    processor_factor = int((getattr(image_processor, "patch_size", None) or 14) *
                           (getattr(image_processor, "merge_size", None) or 2))
    original_resize = getattr(image_processor, "do_resize", True)
    image_processor.do_resize = False
    limits = {k: getattr(image_processor, k, None) for k in ("min_pixels", "max_pixels", "size", "patch_size", "merge_size")}
    print(f"processor resize: {original_resize} -> {image_processor.do_resize}; pixel limits: {limits}", flush=True)

    print("loading model", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(a.model, **load_kwargs)
    if a.resume:
        if a.load_4bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = PeftModel.from_pretrained(model, a.resume, is_trainable=True)
    else:
        if a.load_4bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        linear_names = [(name, module) for name, module in model.named_modules()
                        if isinstance(module, torch.nn.Linear)]
        targets = sorted({name for name, _ in linear_names
                          if ".visual." not in f".{name}." and name.rsplit(".", 1)[-1] in TARGET_MODULES})
        if not targets:
            raise RuntimeError("No language q_proj/k_proj/v_proj/o_proj linear layers found")
        if a.vision_lora:
            vision_targets = sorted(name for name, _ in linear_names
                                    if ".visual." in f".{name}." and name.rsplit(".", 1)[-1] in TARGET_MODULES)
            if not vision_targets:
                raise RuntimeError("--vision-lora requested, but no compatible visual q/k/v/o projection layers were found")
            targets.extend(vision_targets)
            print(f"vision LoRA targets: {vision_targets}", flush=True)
        model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0,
            bias="none", task_type="CAUSAL_LM", target_modules=targets))

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    model.print_trainable_parameters()

    class Lines(torch.utils.data.Dataset):
        def __init__(self, frame): self.frame = frame.reset_index(drop=True)
        def __len__(self): return len(self.frame)
        def __getitem__(self, ix):
            row = self.frame.iloc[ix]
            image, original_size = load_img(files[str(row.ID)], a.img_height, a.max_width)
            image = align_image_grid(image, processor_factor)
            base = user_message(image)
            prompt_inputs = processor.apply_chat_template(base, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt")
            full = base + [{"role": "assistant", "content": [{"type": "text", "text": clean(row.Target)}]}]
            inputs = processor.apply_chat_template(full, tokenize=True, add_generation_prompt=False,
                return_dict=True, return_tensors="pt")
            ids_tensor = inputs["input_ids"].squeeze(0)
            labels = ids_tensor.clone()
            prefix_len = prompt_inputs["input_ids"].shape[-1]
            labels[:prefix_len] = -100
            if "attention_mask" in inputs:
                labels[inputs["attention_mask"].squeeze(0) == 0] = -100
            inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == 1 else v
                      for k, v in inputs.items()}
            inputs["labels"] = labels
            return inputs

    def collate(rows):
        keys = rows[0].keys()
        out_batch = {}
        for key in keys:
            values = [row[key] for row in rows]
            if key == "input_ids": out_batch[key] = torch.nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
            elif key == "labels": out_batch[key] = torch.nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=-100)
            elif key == "attention_mask": out_batch[key] = torch.nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=0)
            elif len(values) == 1: out_batch[key] = values[0].unsqueeze(0)
            else: out_batch[key] = torch.cat(values, dim=0)
        return out_batch

    dataset = Lines(tr)
    loader = torch.utils.data.DataLoader(dataset, batch_size=a.bs, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=a.lr)
    steps_per_epoch = (len(loader) + a.grad_accum - 1) // a.grad_accum
    total_steps = steps_per_epoch * int(a.epochs) * a.grad_accum
    if a.smoke:
        total_steps = 40 * a.grad_accum  # 40 optimizer updates, matching Trainer max_steps semantics.
    warmup_steps = max(1, int((total_steps / a.grad_accum) * 0.05))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cuda and dtype == torch.float16))
    model.zero_grad(set_to_none=True)
    step = 0
    epoch = 0
    epoch_iter = iter(loader)
    train_start = time.perf_counter()
    step_seconds = []
    while step < total_steps:
        try:
            batch = next(epoch_iter)
        except StopIteration:
            epoch += 1
            epoch_iter = iter(loader)
            batch = next(epoch_iter)
        batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        lr_scale = min(1.0, (step // a.grad_accum + 1) / warmup_steps)
        for group in optimizer.param_groups: group["lr"] = a.lr * lr_scale
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=dtype, enabled=cuda):
            loss = model(**batch).loss / a.grad_accum
        scaler.scale(loss).backward()
        if (step + 1) % a.grad_accum == 0 or step + 1 == total_steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        step_seconds.append(time.perf_counter() - started)
        step += 1
        if step % (10 * a.grad_accum) == 0 or step == 1 or step == total_steps:
            print(f"step {((step - 1) // a.grad_accum) + 1}/{total_steps // a.grad_accum} "
                  f"loss={float(loss.detach()) * a.grad_accum:.4f} "
                  f"seconds/step={sum(step_seconds)/len(step_seconds):.2f}", flush=True)
    training_seconds = time.perf_counter() - train_start
    optimizer_steps = max(1, total_steps // a.grad_accum)
    seconds_per_step = training_seconds / optimizer_steps
    model.config.use_cache = True
    model.eval()
    model.save_pretrained(out / "lora")
    processor.save_pretrained(out / "lora")

    def predict(frame, desc):
        predictions, infer_seconds = [], 0.0
        started_all = time.perf_counter()
        identities = frame.ID.astype(str).tolist()
        old_padding_side = processor.tokenizer.padding_side
        processor.tokenizer.padding_side = "left"
        for start in range(0, len(identities), a.infer_bs):
            batch_ids = identities[start:start + a.infer_bs]
            images, original_sizes = [], []
            for identity in batch_ids:
                image, original_size = load_img(files[identity], a.img_height, a.max_width)
                images.append(align_image_grid(image, processor_factor))
                original_sizes.append(original_size)
            messages = [user_message(image) for image in images]
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", padding=True)
            if start == 0:
                grid = inputs.get("image_grid_thw")
                print(f"{desc} image dimensions: original={original_sizes[0]}, competition_resize={load_img(files[batch_ids[0]], a.img_height, a.max_width)[0].size}, "
                      f"processor_input={images[0].size}, processor_grid={grid.tolist() if grid is not None else None}, limits={limits}", flush=True)
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            started = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype, enabled=cuda):
                generated = model.generate(**inputs, max_new_tokens=64, do_sample=False, num_beams=1,
                                           use_cache=True, pad_token_id=processor.tokenizer.pad_token_id)
            infer_seconds += time.perf_counter() - started
            for generated_ids in generated:
                new_ids = generated_ids[inputs["input_ids"].shape[-1]:]
                pred = clean(processor.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
                predictions.append(pred or "[illegible]")  # submission contract forbids empty Target values
            if start + len(batch_ids) >= len(frame) or (start + len(batch_ids)) // 100 > start // 100:
                print(f"{desc}: {start + len(batch_ids)}/{len(frame)}", flush=True)
        processor.tokenizer.padding_side = old_padding_side
        return predictions, infer_seconds, time.perf_counter() - started_all

    val_pred, val_infer_seconds, val_wall = predict(va, "validation")
    val_metric = lb_score(va.Target.map(clean).tolist(), val_pred)
    extra_metrics = official_score(va.Target.map(clean).tolist(), val_pred)
    metrics = {**val_metric, "wer": extra_metrics["wer"], "cer": extra_metrics["cer"],
               "n_validation": len(va), "inference_seconds": round(val_wall, 2),
               "generation_seconds": round(val_infer_seconds, 2)}
    pd.DataFrame({"ID": va.ID.values, "Target": val_pred}).to_csv(out / "val_predictions.csv", index=False)
    (out / "val_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"VAL: {metrics['lb_score']:.5f}", flush=True)
    print(f"WER: {metrics['wer']:.5f} CER: {metrics['cer']:.5f} validation examples: {len(va)}", flush=True)
    estimated_training_seconds = steps_per_epoch * int(a.epochs) * seconds_per_step
    estimated_total_seconds = estimated_training_seconds + (val_wall / max(1, len(va))) * (410 + 1374)
    print(f"inference time: {val_wall:.1f}s; training seconds/step: {seconds_per_step:.2f}; "
          f"estimated full training: {steps_per_epoch * int(a.epochs) * seconds_per_step / 3600:.2f}h; "
          f"estimated end-to-end (train + val + test generation): {estimated_total_seconds / 3600:.2f}h", flush=True)

    if a.predict_test:
        test = pd.read_csv(data_dir / "Test.csv", encoding="utf-8-sig", keep_default_na=False)
        test_pred, _, _ = predict(test, "test")
        submission = pd.DataFrame({"ID": test.ID, "Target": test_pred})
        if len(submission) != 1374 or submission.Target.str.strip().eq("").any():
            raise ValueError(f"Invalid submission: rows={len(submission)}, empty targets={int(submission.Target.str.strip().eq('').sum())}")
        submission.to_csv(out / "submission.csv", index=False)
        print(f"submission rows: {len(submission)}", flush=True)

    try:
        import subprocess
        split_md5 = hashlib.md5(split_path.read_bytes()).hexdigest()
    except Exception:
        split_md5 = None
    config = {**vars(a), "model": a.model, "transformers_version": transformers.__version__,
              "torch_version": torch.__version__, "peft_version": peft.__version__,
              "cuda_version": torch.version.cuda, "gpu_name": gpu_name, "seed": a.seed,
              "learning_rate": a.lr, "epochs": a.epochs, "batch_size": a.bs,
              "gradient_accumulation": a.grad_accum, "lora_rank": a.rank, "lora_alpha": a.alpha,
              "image_height": a.img_height, "max_width": a.max_width, "prompt": PROMPT,
              "split_path": str(split_path), "split_md5": split_md5, "git_commit": get_git_commit(),
              "timestamp_utc": datetime.now(timezone.utc).isoformat(), "processor_pixel_limits": limits,
              "processor_resize_enabled": image_processor.do_resize, "processor_grid_factor": processor_factor,
              "training_seconds": round(training_seconds, 2), "training_seconds_per_step": round(seconds_per_step, 3),
              "estimated_full_training_seconds": round(estimated_training_seconds, 2),
              "estimated_total_runtime_seconds": round(estimated_total_seconds, 2),
              "validation_metrics": metrics,
              "gpu_max_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3) if cuda else None}
    (out / "run_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    if cuda:
        print(f"GPU peak allocated memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
