"""QLoRA fine-tune of a Qwen VL model on the Barbados line images (Unsloth, single T4).

Kaggle setup (GPU T4 x2; this script uses one GPU):
  pip install -q unsloth
  python src/train_qwen.py --data-dir $DATA --img-dir $IMG \
      --split /kaggle/working/splits/split_v1.csv --out /kaggle/working/runs/qwen-lora-v1 --predict-test

Smoke test (~15 min): add --smoke

Images are resized to a fixed height (default 112 px, a multiple of 28) keeping the
aspect ratio, with width capped, so long lines are never squashed and the number of
vision tokens stays small (<= ~320 per line).

Faster prediction: --pred-batch 8 predicts 8 lines per batch instead of one at a time (measure the gain
with a --smoke run first). Combine with --load-lora <saved lora dir> to re-predict without retraining.

Outputs in --out: lora/ (adapter), val_predictions.csv, val_metrics.json,
submission.csv (if --predict-test), run_config.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import pandas as pd  # noqa: E402
from PIL import Image, ImageFilter, ImageOps  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from official_metric import score  # noqa: E402

PROMPT = ("Transcribe this handwritten line from an 18th or 19th century Barbados archive exactly as written. "
          "Keep the original spelling, capitalisation, abbreviations and symbols such as ^ ~ & _. "
          "Output only the transcription.")


def clean(text: str) -> str:
    text = str(text).strip().split("\n")[0]
    return re.sub(r"\s+", " ", text).strip()


def enhance_img(img: Image.Image) -> Image.Image:
    """Autocontrast (stretch the tonal range, clipping the extreme 1% of pixels each end) plus a
    mild unsharp mask. Applied before resize, while the original detail is still there. Generic,
    code-driven, and the same for every image -- no per-image tuning."""
    img = ImageOps.autocontrast(img, cutoff=1)
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=100, threshold=3))
    return img


def load_img(path: Path, height: int, max_width: int, enhance: bool = False) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if enhance:
        img = enhance_img(img)
    w, h = img.size
    new_w = min(max_width, max(28, round(w * height / h)))
    new_h = height if new_w < max_width else max(28, round(h * max_width / w))
    return img.resize((new_w, new_h), Image.BICUBIC)


def to_conv(img, text=None):
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
    if text is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
    return {"messages": msgs}


def is_left_padded(attention_mask) -> bool:
    """True if every row ends with a real token (mask 1 in the last column), i.e. any padding is on
    the LEFT. Right padding makes generate() continue after pad tokens and silently ruins output."""
    return all(row[-1] == 1 for row in attention_mask.tolist())


def run_batches(keys, batch, fn, on_error=None):
    """Run fn on batches of item indices, sorted by `keys` so similar items share a batch
    (images of similar width need little padding), and return results in the ORIGINAL order.
    fn(indices) -> list of results. If a batch raises, its items are retried one at a time."""
    order = sorted(range(len(keys)), key=lambda k: keys[k])
    out = [None] * len(keys)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        try:
            res = fn(idx)
        except Exception as e:  # e.g. out of memory on a batch of wide images
            if len(idx) == 1:
                raise
            print(f"  batch of {len(idx)} failed ({type(e).__name__}); retrying one at a time", flush=True)
            if on_error:
                on_error()
            res = [fn([k])[0] for k in idx]
        for k, r in zip(idx, res):
            out[k] = r
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="unsloth/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--img-height", type=int, default=112)
    ap.add_argument("--max-width", type=int, default=2240)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--predict-test", action="store_true")
    ap.add_argument("--zero-shot", action="store_true", help="skip training; predict with the base model")
    ap.add_argument("--num-beams", type=int, default=1, help="beam search width at inference (1 = greedy)")
    ap.add_argument("--load-lora", default=None, help="path to a saved LoRA dir; loads it and skips training")
    ap.add_argument("--pred-batch", type=int, default=1,
                    help="lines predicted per batch (1 = one at a time, the original behaviour). "
                         "try 8 on a T4; a failing batch falls back to one at a time")
    ap.add_argument("--enhance", action="store_true",
                    help="autocontrast + mild unsharp mask on every image, train and predict alike. "
                         "27%% of images are in the blurriest/darkest quartile and have ~2x the "
                         "validation error of the sharpest quartile (reports/image_audit.csv)")
    a = ap.parse_args()

    from unsloth import FastVisionModel, is_bf16_supported
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTConfig, SFTTrainer
    import torch

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    img_dir = Path(a.img_dir)
    files = {os.path.splitext(f)[0]: img_dir / f for f in os.listdir(img_dir)}

    df = pd.read_csv(f"{a.data_dir}/Train.csv", encoding="utf-8-sig", keep_default_na=False)
    df = df.merge(pd.read_csv(a.split, encoding="utf-8-sig"), on="ID")
    tr, va = df[df.fold == "train"], df[df.fold == "val"]
    if a.smoke:
        tr, va = tr.sample(300, random_state=0), va.sample(48, random_state=0)

    model, tok = FastVisionModel.from_pretrained(a.load_lora or a.model, load_in_4bit=True,
                                                 use_gradient_checkpointing="unsloth")
    if not a.zero_shot and not a.load_lora:
        model = FastVisionModel.get_peft_model(
            model, finetune_vision_layers=True, finetune_language_layers=True,
            finetune_attention_modules=True, finetune_mlp_modules=True,
            r=a.rank, lora_alpha=a.rank, lora_dropout=0, bias="none", random_state=a.seed)

        train_data = [to_conv(load_img(files[i], a.img_height, a.max_width, a.enhance), clean(t))
                      for i, t in zip(tr.ID, tr.Target)]
        FastVisionModel.for_training(model)
        steps = dict(max_steps=40) if a.smoke else dict(num_train_epochs=a.epochs)
        trainer = SFTTrainer(
            model=model, tokenizer=tok, data_collator=UnslothVisionDataCollator(model, tok),
            train_dataset=train_data,
            args=SFTConfig(
                per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.grad_accum,
                warmup_ratio=0.03, learning_rate=a.lr, lr_scheduler_type="linear",
                fp16=not is_bf16_supported(), bf16=is_bf16_supported(), optim="adamw_8bit",
                weight_decay=0.01, logging_steps=10, save_strategy="steps", save_steps=200,
                save_total_limit=2, output_dir=str(out / "ckpt"), seed=a.seed, report_to="none",
                remove_unused_columns=False, dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True}, max_seq_length=1024, **steps))
        trainer.train()
        model.save_pretrained(str(out / "lora")); tok.save_pretrained(str(out / "lora"))

    FastVisionModel.for_inference(model)

    @torch.no_grad()
    def predict(ids):
        preds = []
        for n, i in enumerate(ids):
            img = load_img(files[i], a.img_height, a.max_width, a.enhance)
            text = tok.apply_chat_template(to_conv(img)["messages"], add_generation_prompt=True)
            inputs = tok(img, text, add_special_tokens=False, return_tensors="pt").to("cuda")
            gen = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False, num_beams=a.num_beams, use_cache=True)
            preds.append(clean(tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)))
            if n % 100 == 0:
                print(f"  predicted {n}/{len(ids)}", flush=True)
        return preds

    if hasattr(tok, "tokenizer"):
        tok.tokenizer.padding_side = "left"  # needed for batched generation

    @torch.no_grad()
    def predict_batched(ids):
        keys = []
        for i in ids:
            w, h = Image.open(files[i]).size
            keys.append(w / h)  # aspect ratio ~ resized width, so batches need little padding
        done = [0]

        def run(idx):
            imgs = [load_img(files[ids[k]], a.img_height, a.max_width, a.enhance) for k in idx]
            texts = [tok.apply_chat_template(to_conv(im)["messages"], add_generation_prompt=True) for im in imgs]
            inputs = tok(imgs, texts, add_special_tokens=False, return_tensors="pt", padding=True).to("cuda")
            if len(idx) > 1 and not is_left_padded(inputs["attention_mask"]):
                # never generate from a right-padded batch: raise so run_batches redoes these lines one at a time
                raise RuntimeError("batch is right-padded; falling back to one line at a time")
            gen = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False,
                                 num_beams=a.num_beams, use_cache=True)
            outs = tok.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            before = done[0]
            done[0] += len(idx)
            if done[0] // 100 > before // 100:
                print(f"  predicted {done[0]}/{len(ids)}", flush=True)
            return [clean(o) for o in outs]

        return run_batches(keys, a.pred_batch, run, on_error=torch.cuda.empty_cache)

    if a.pred_batch > 1:
        predict = predict_batched  # noqa: F811

    val_pred = predict(va.ID.tolist())
    metrics = score(va.Target.map(clean).tolist(), val_pred)
    pd.DataFrame({"ID": va.ID.values, "Target": val_pred}).to_csv(out / "val_predictions.csv", index=False)
    (out / "val_metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "run_config.json").write_text(json.dumps(vars(a), indent=2))
    print("VAL:", metrics)

    if a.predict_test:
        te = pd.read_csv(f"{a.data_dir}/Test.csv", encoding="utf-8-sig", keep_default_na=False)
        pd.DataFrame({"ID": te.ID, "Target": predict(te.ID.tolist())}).to_csv(out / "submission.csv", index=False)
        print("wrote", out / "submission.csv")


if __name__ == "__main__":
    main()
