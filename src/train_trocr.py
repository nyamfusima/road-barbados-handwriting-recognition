"""Fine-tune TrOCR on the Barbados line images.

Run on Kaggle (2 x T4, DDP):
  torchrun --nproc_per_node 2 src/train_trocr.py --data-dir $DATA --img-dir $IMG \
      --split /kaggle/working/splits/split_v1.csv --out /kaggle/working/runs/trocr-large-v1

Smoke test (about 5 min):
  torchrun --nproc_per_node 2 src/train_trocr.py ... --out /kaggle/working/runs/smoke --smoke

Key flags
  --fold N      cut each line into N horizontal pieces and stack them vertically
                (1 = off). Lines are ~17:1, so the default 384x384 resize squashes them;
                fold 3 or 4 brings them close to square.
  --aug         mild augmentation (rotation, blur, contrast) on training images only.
  --predict-test  write val predictions and a test submission with beam search at the end.

Outputs in --out: best/ (model + processor), val_predictions.csv, val_metrics.json,
submission.csv (if --predict-test), run_config.json.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, TrOCRProcessor,
                          VisionEncoderDecoderModel, set_seed)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from official_metric import score  # noqa: E402


# ---------------------------------------------------------------- helpers
def clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def fold_image(img: Image.Image, n: int, overlap: float = 0.04) -> Image.Image:
    """Cut a wide line into n overlapping pieces and stack them top to bottom."""
    if n <= 1:
        return img
    w, h = img.size
    step = w / n
    pad = int(step * overlap)
    pieces = [img.crop((max(0, int(i * step) - pad), 0, min(w, int((i + 1) * step) + pad), h)) for i in range(n)]
    pw = max(p.width for p in pieces)
    gap = max(2, h // 8)
    canvas = Image.new("RGB", (pw, n * h + (n - 1) * gap), (255, 255, 255))
    for i, p in enumerate(pieces):
        canvas.paste(p, (0, i * (h + gap)))
    return canvas


def augment(img: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        img = img.rotate(random.uniform(-1.5, 1.5), resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    if random.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))
    if random.random() < 0.3:
        from PIL import ImageEnhance
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.75, 1.25))
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
    return img


class LineDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, processor, fold, max_target_len, train=False, aug=False):
        self.ids = df["ID"].tolist()
        self.texts = df["Target"].tolist() if "Target" in df else None
        self.img_dir, self.proc, self.fold = Path(img_dir), processor, fold
        self.max_len, self.train, self.aug = max_target_len, train, aug
        self.eos = processor.tokenizer.eos_token_id
        self.ext = {}
        for f in os.listdir(self.img_dir):
            self.ext[os.path.splitext(f)[0]] = f

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = Image.open(self.img_dir / self.ext[self.ids[i]]).convert("RGB")
        if self.train and self.aug:
            img = augment(img)
        img = fold_image(img, self.fold)
        item = {"pixel_values": self.proc(images=img, return_tensors="pt").pixel_values[0]}
        if self.texts is not None:
            ids = self.proc.tokenizer(clean(self.texts[i]), add_special_tokens=False).input_ids
            item["labels"] = ids[: self.max_len - 1] + [self.eos]  # fairseq-style target: text + </s>
        return item


def collate(batch):
    out = {"pixel_values": torch.stack([b["pixel_values"] for b in batch])}
    if "labels" in batch[0]:
        L = max(len(b["labels"]) for b in batch)
        out["labels"] = torch.tensor([b["labels"] + [-100] * (L - len(b["labels"])) for b in batch])
    return out


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="microsoft/trocr-large-handwritten")
    ap.add_argument("--epochs", type=float, default=15)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--bs", type=int, default=4, help="per GPU")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true", help="use if you hit CUDA OOM")
    ap.add_argument("--max-target-len", type=int, default=48)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--beams", type=int, default=4, help="beams for final val/test predictions")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--predict-test", action="store_true")
    a = ap.parse_args()

    set_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rank0 = int(os.environ.get("RANK", 0)) == 0

    train_df = pd.read_csv(f"{a.data_dir}/Train.csv", encoding="utf-8-sig", keep_default_na=False)
    split = pd.read_csv(a.split, encoding="utf-8-sig")
    train_df = train_df.merge(split, on="ID")
    tr, va = train_df[train_df.fold == "train"], train_df[train_df.fold == "val"]
    if a.smoke:
        tr, va = tr.sample(400, random_state=0), va.sample(64, random_state=0)

    proc = TrOCRProcessor.from_pretrained(a.model)
    model = VisionEncoderDecoderModel.from_pretrained(a.model)
    tok = proc.tokenizer
    model.config.decoder_start_token_id = tok.eos_token_id   # matches TrOCR pretraining (</s> start)
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.generation_config.decoder_start_token_id = tok.eos_token_id
    model.generation_config.pad_token_id = tok.pad_token_id
    model.generation_config.eos_token_id = tok.eos_token_id
    model.generation_config.max_new_tokens = a.max_new_tokens
    model.generation_config.no_repeat_ngram_size = 0          # repeated words are legitimate
    if a.grad_ckpt:
        model.gradient_checkpointing_enable()

    ds_tr = LineDataset(tr, a.img_dir, proc, a.fold, a.max_target_len, train=True, aug=a.aug)
    ds_va = LineDataset(va, a.img_dir, proc, a.fold, a.max_target_len)

    def decode(ids):
        ids = np.where(ids < 0, tok.pad_token_id, ids)
        return [clean(t) for t in tok.batch_decode(ids, skip_special_tokens=True)]

    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        return score(decode(p.label_ids), decode(preds))

    steps = dict(eval_strategy="steps", eval_steps=40, save_strategy="steps", save_steps=40, max_steps=80) \
        if a.smoke else dict(eval_strategy="epoch", save_strategy="epoch", num_train_epochs=a.epochs)
    args = Seq2SeqTrainingArguments(
        output_dir=str(out / "ckpt"), per_device_train_batch_size=a.bs, per_device_eval_batch_size=16,
        gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, warmup_ratio=0.05,
        weight_decay=0.01, lr_scheduler_type="linear", fp16=True, logging_steps=20,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="final",
        greater_is_better=False, predict_with_generate=True, generation_max_length=a.max_new_tokens,
        generation_num_beams=1, dataloader_num_workers=2, report_to="none", seed=a.seed,
        ddp_find_unused_parameters=False, **steps)

    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_va,
                             data_collator=collate, compute_metrics=compute_metrics)
    trainer.train()

    # final validation with beam search on the best checkpoint
    res = trainer.predict(ds_va, num_beams=a.beams, max_new_tokens=a.max_new_tokens)
    preds = res.predictions[0] if isinstance(res.predictions, tuple) else res.predictions
    val_pred = decode(preds)[: len(va)]
    metrics = score(va["Target"].map(clean).tolist(), val_pred)
    metrics["beams"] = a.beams

    if rank0:
        trainer.save_model(str(out / "best")); proc.save_pretrained(str(out / "best"))
        pd.DataFrame({"ID": va["ID"].values, "Target": val_pred}).to_csv(out / "val_predictions.csv", index=False)
        (out / "val_metrics.json").write_text(json.dumps(metrics, indent=2))
        (out / "run_config.json").write_text(json.dumps(vars(a), indent=2))
        print("VAL (beam", a.beams, "):", metrics)

    if a.predict_test:
        te = pd.read_csv(f"{a.data_dir}/Test.csv", encoding="utf-8-sig", keep_default_na=False)
        ds_te = LineDataset(te, a.img_dir, proc, a.fold, a.max_target_len)
        res = trainer.predict(ds_te, num_beams=a.beams, max_new_tokens=a.max_new_tokens)
        preds = res.predictions[0] if isinstance(res.predictions, tuple) else res.predictions
        if rank0:
            pd.DataFrame({"ID": te["ID"], "Target": decode(preds)[: len(te)]}).to_csv(out / "submission.csv", index=False)
            print("wrote", out / "submission.csv")


if __name__ == "__main__":
    main()
