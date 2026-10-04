# How to reproduce

All commands run from the repo root. Set the data paths first (Kaggle example):

```bash
pip install -q unsloth
DATA=data/raw            # folder holding Train.csv, Test.csv (Kaggle: /kaggle/input/<your-dataset>)
IMG=data/raw/images      # folder holding the line images for train AND test
```

## 0. Data prep and splits

```bash
python src/data_prep.py --raw-dir $DATA --seed 42 --split-version v1   # -> data/splits/split_v1.csv
python scripts/make_splits.py --split data/splits/split_v1.csv --out-dir data/splits
#   -> data/splits/split_v1_clean.csv (3 bad rows dropped), data/splits/split_full.csv
```

`split_v1.csv`: 3,688 train / 410 val, seed 42, md5 `61d5a1a939e97a442dc635bcb63adb1d`.

Each training run writes `val_predictions.csv`, `submission.csv`, `lora/` and `run_config.json`
into `--out`. Copy the two CSVs into `preds/` under the member name, e.g.

```bash
cp runs/v2/val_predictions.csv preds/val/v2.csv
cp runs/v2/submission.csv      preds/test/v2.csv
```

Seed is the script default (`--seed 3407`) unless a command says otherwise.

## 1. Members in preds/ (used in the final votes)

v1: 7B, 2 epochs, h112 (Kaggle T4)
```bash
python src/train_qwen.py --data-dir $DATA --img-dir $IMG --split data/splits/split_v1.csv \
    --out runs/v1 --model unsloth/Qwen2.5-VL-7B-Instruct \
    --bs 1 --grad-accum 8 --epochs 2 --img-height 112 --predict-test
```

v2: 7B, 3 epochs, h140, the "v2 recipe" (Kaggle T4)
```bash
python src/train_qwen.py --data-dir $DATA --img-dir $IMG --split data/splits/split_v1.csv \
    --out runs/v2 --model unsloth/Qwen2.5-VL-7B-Instruct \
    --bs 1 --grad-accum 8 --epochs 3 --img-height 140 --predict-test
```

full: v2 recipe on all 4,098 train lines, plus 40 duplicated val lines as a leaky check (Kaggle T4).
No usable val predictions, so it's left out of the val ensemble evals.
```bash
python src/train_qwen.py --data-dir $DATA --img-dir $IMG --split data/splits/split_full.csv \
    --out runs/full --model unsloth/Qwen2.5-VL-7B-Instruct \
    --bs 1 --grad-accum 8 --epochs 3 --img-height 140 --predict-test
```

r64: v2 recipe, LoRA rank 64 (Kaggle T4)
```bash
python src/train_qwen.py --data-dir $DATA --img-dir $IMG --split data/splits/split_v1.csv \
    --out runs/r64 --model unsloth/Qwen2.5-VL-7B-Instruct \
    --bs 1 --grad-accum 8 --epochs 3 --img-height 140 --rank 64 --predict-test
```

32b: v2 recipe on the 32B base (Colab A100)
```bash
python src/train_qwen.py --data-dir $DATA --img-dir $IMG --split data/splits/split_v1.csv \
    --out runs/32b --model unsloth/Qwen2.5-VL-32B-Instruct-unsloth-bnb-4bit \
    --bs 1 --grad-accum 8 --epochs 3 --img-height 140 --predict-test
```

32b_pseudo: 32B v2 recipe + 1,374 test pseudo-labels from ens-7b4 (Colab A100)
```bash
python src/train_qwen.py --data-dir $DATA --img-dir $IMG --split data/splits/split_v1.csv \
    --out runs/32b_pseudo --model unsloth/Qwen2.5-VL-32B-Instruct-unsloth-bnb-4bit \
    --bs 1 --grad-accum 8 --epochs 3 --img-height 140 \
    --extra-train pseudo/test_pseudo_ens7b4.csv --predict-test
```

wide: 7B v2 recipe + max width 4480 (width fix), clean split (Kaggle T4)
```bash
python src/train_qwen.py --data-dir $DATA --img-dir $IMG --split data/splits/split_v1_clean.csv \
    --out runs/wide --model unsloth/Qwen2.5-VL-7B-Instruct \
    --bs 1 --grad-accum 8 --epochs 3 --img-height 140 --max-width 4480 --predict-test
```

## 2. Pseudo-label files (pseudo/)

These come from our own ensembles on the test images, with no manual edits. The ensembles and the
pseudo files are byte-identical.

```bash
# pseudo/test_pseudo_ens7b4.csv: medoid vote of v1, v2, full, r64
python src/ensemble_rover.py --method medoid \
    --preds v1=preds/test/v1.csv v2=preds/test/v2.csv full=preds/test/full.csv r64=preds/test/r64.csv \
    --weights v1=1 v2=1.2 full=1.1 r64=1.2 --out pseudo/test_pseudo_ens7b4.csv

# pseudo/test_pseudo_rover6.csv: identical to submissions/sub_rover6.csv (below)
cp submissions/sub_rover6.csv pseudo/test_pseudo_rover6.csv
```

## 3. Ensembles (char-level ROVER, margin 0)

submissions/sub_rover6.csv (6 members, public 0.9030):
```bash
python src/ensemble_rover.py --method rover-char \
    --preds v1=preds/test/v1.csv v2=preds/test/v2.csv full=preds/test/full.csv \
            r64=preds/test/r64.csv 32b=preds/test/32b.csv 32bp=preds/test/32b_pseudo.csv \
    --weights v1=1 v2=1.2 full=1.1 r64=1.2 32b=1.3 32bp=1.0 --margin 0.0 \
    --out submissions/sub_rover6.csv
```

submissions/sub_rover7.csv (7 members, best val 0.90257):
```bash
python src/ensemble_rover.py --method rover-char \
    --preds v1=preds/test/v1.csv v2=preds/test/v2.csv full=preds/test/full.csv \
            r64=preds/test/r64.csv 32b=preds/test/32b.csv 32bp=preds/test/32b_pseudo.csv \
            wide=preds/test/wide.csv \
    --weights v1=1 v2=1.2 full=1.1 r64=1.2 32b=1.3 32bp=1.0 wide=1.2 --margin 0.0 \
    --out submissions/sub_rover7.csv
```

submissions/sub_best.csv (= sub_rover5.csv, 5 members, public 0.90115):
```bash
python src/ensemble_rover.py --method rover-char \
    --preds v1=preds/test/v1.csv v2=preds/test/v2.csv full=preds/test/full.csv \
            r64=preds/test/r64.csv 32b=preds/test/32b.csv \
    --weights v1=1 v2=1.2 full=1.1 r64=1.2 32b=1.3 --margin 0.0 \
    --out submissions/sub_best.csv
```

Validation of the votes (exact LB metric; `full` is left out because it trained on val):
```bash
python scripts/eval_ensembles.py \
    --val v1=preds/val/v1.csv v2=preds/val/v2.csv r64=preds/val/r64.csv 32b=preds/val/32b.csv \
          32bp=preds/val/32b_pseudo.csv wide=preds/val/wide.csv \
    --weights v1=1 v2=1.2 r64=1.2 32b=1.3 32bp=1.0 wide=1.2 --truth $DATA/Train.csv
```

Check every file before uploading: `python src/check_submission.py submissions/sub_rover7.csv`.

## 4. Upcoming members (placeholders: fill in once run)

pseudo3: 7B v2 recipe + pseudo-labels from rover6, seed 21 (Kaggle T4)
```bash
python src/train_qwen.py --data-dir $DATA --img-dir $IMG --split data/splits/split_v1_clean.csv \
    --out runs/pseudo3 --model unsloth/Qwen2.5-VL-7B-Instruct \
    --bs 1 --grad-accum 8 --epochs 3 --img-height 140 \
    --extra-train pseudo/test_pseudo_rover6.csv --seed 21 --predict-test
```
- Val LB: TBD | Public: TBD | Ensemble weight: TBD

widepseudo: 7B v2 recipe + max width 4480 + pseudo rover6, seed 33 (Kaggle T4)
```bash
python src/train_qwen.py --data-dir $DATA --img-dir $IMG --split data/splits/split_v1_clean.csv \
    --out runs/widepseudo --model unsloth/Qwen2.5-VL-7B-Instruct \
    --bs 1 --grad-accum 8 --epochs 3 --img-height 140 --max-width 4480 \
    --extra-train pseudo/test_pseudo_rover6.csv --seed 33 --predict-test
```
- Val LB: TBD | Public: TBD | Ensemble weight: TBD

h168pseudo: 7B, h168, max width 4480 + pseudo rover6, seed 44 (Kaggle T4)
```bash
python src/train_qwen.py --data-dir $DATA --img-dir $IMG --split data/splits/split_v1_clean.csv \
    --out runs/h168pseudo --model unsloth/Qwen2.5-VL-7B-Instruct \
    --bs 1 --grad-accum 8 --epochs 3 --img-height 168 --max-width 4480 \
    --extra-train pseudo/test_pseudo_rover6.csv --seed 44 --predict-test
```
- Val LB: TBD | Public: TBD | Ensemble weight: TBD

Teammate runs (e.g. clean 32B seed 7, Colab A100):
```bash
# TBD: paste the exact command, hardware, split and seed here
```
- Val LB: TBD | Public: TBD | Ensemble weight: TBD

Final ensemble including the new members:
```bash
# TBD: python src/ensemble_rover.py --method rover-char --preds ... --weights ... --margin 0.0 --out submissions/sub_roverN.csv
```

## 5. qwen3-vl-8b-sft (Qwen3-VL-8B-Instruct, Transformers + PEFT)

This is an independent `src/train_qwen3.py` pipeline. It reproduces `train_qwen.py`'s
competition resize, prompt, label cleaning, split, and leaderboard metric. It uses language
LoRA on q/k/v/o projections by default, batch 1 x accumulation 8, 3 epochs, lr 2e-4,
5% warmup, AdamW, seed 3407, greedy decoding, and 64 new tokens. The processor resize is
disabled; the competition-resized image is padded to the processor patch grid without
resampling. Optional `--vision-lora` only enables compatible vision projections.

Install in a fresh Colab or Kaggle GPU session:
```bash
pip install -U "transformers>=4.57.0" peft accelerate bitsandbytes pandas pillow
```

Colab A100 smoke run (40 optimizer updates and 48 validation lines):
```bash
python src/train_qwen3.py --data-dir /content/work/raw --img-dir /content/work/images \
  --split /content/work/splits/split_v1.csv \
  --out /content/drive/MyDrive/barbados/qwen3-vl-8b-smoke --smoke
```

Colab A100 full run:
```bash
python src/train_qwen3.py --data-dir /content/work/raw --img-dir /content/work/images \
  --split /content/work/splits/split_v1.csv \
  --out /content/drive/MyDrive/barbados/qwen3-vl-8b-sft \
  --model Qwen/Qwen3-VL-8B-Instruct --epochs 3 --lr 2e-4 --bs 1 \
  --grad-accum 8 --rank 16 --alpha 16 --img-height 140 --max-width 2240 \
  --seed 3407 --predict-test
```

Kaggle T4 smoke run (4-bit NF4 base):
```bash
python src/train_qwen3.py --data-dir $DATA --img-dir $IMG \
  --split /kaggle/working/splits/split_v1.csv \
  --out /kaggle/working/qwen3-vl-8b-smoke --smoke --load-4bit
```

Kaggle T4 full run:
```bash
python src/train_qwen3.py --data-dir $DATA --img-dir $IMG \
  --split /kaggle/working/splits/split_v1.csv \
  --out /kaggle/working/qwen3-vl-8b-sft --model Qwen/Qwen3-VL-8B-Instruct \
  --epochs 3 --lr 2e-4 --bs 1 --grad-accum 8 --rank 16 --alpha 16 \
  --img-height 140 --max-width 2240 --seed 3407 --predict-test --load-4bit
```

Smoke and full expected times are measured from the current GPU and written to
`run_config.json`; no timing result exists until the first remote smoke. Output files are
`val_predictions.csv`, `val_metrics.json`, `run_config.json`, and (with `--predict-test`)
`submission.csv`. Val LB, WER, CER, and public result are pending.
