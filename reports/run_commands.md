# How to reproduce

Split file: `splits/split_v1.csv` (3,688 train / 410 val, seed 42), md5 `61d5a1a939e97a442dc635bcb63adb1d`.

## Models in the final vote

7B v1:
```
python src/train_qwen.py --split splits/split_v1.csv --model unsloth/Qwen2.5-VL-7B-Instruct \
    --bs 1 --grad-accum 8 --epochs 2 --img-height 112 --predict-test
```

7B v2 (same as v1, more epochs and height):
```
python src/train_qwen.py --split splits/split_v1.csv --model unsloth/Qwen2.5-VL-7B-Instruct \
    --bs 1 --grad-accum 8 --epochs 3 --img-height 140 --predict-test
```

7B full (v2 flags, trained on all train + 40 duplicated val lines as a leaky sanity check):
```
python src/train_qwen.py --split splits/split_full.csv --model unsloth/Qwen2.5-VL-7B-Instruct \
    --bs 1 --grad-accum 8 --epochs 3 --img-height 140 --predict-test
```

7B r64 (v2 flags, higher LoRA rank):
```
python src/train_qwen.py --split splits/split_v1.csv --model unsloth/Qwen2.5-VL-7B-Instruct \
    --bs 1 --grad-accum 8 --epochs 3 --img-height 140 --rank 64 --predict-test
```

32B (Colab A100, v2 flags on the 32B base):
```
python src/train_qwen.py --split splits/split_v1.csv \
    --model unsloth/Qwen2.5-VL-32B-Instruct-unsloth-bnb-4bit \
    --bs 1 --grad-accum 8 --epochs 3 --img-height 140 --predict-test
```

## Ensemble (submissions/sub_best.csv, char-level ROVER, 5 members)

```
python src/ensemble_rover.py --method rover-char \
    --preds v1=preds/test/v1.csv v2=preds/test/v2.csv full=preds/test/full.csv \
            r64=preds/test/r64.csv 32b=preds/test/32b.csv \
    --weights v1=1 v2=1.2 full=1.1 r64=1.2 32b=1.3 --margin 0.0 \
    --out submissions/sub_best.csv
```
