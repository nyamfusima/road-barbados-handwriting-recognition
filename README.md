# R.O.A.D. Barbados Historic Handwriting Challenge

This repo holds our solution to the Zindi **R.O.A.D. Barbados Historic Handwriting Challenge**: line-level
handwritten text recognition (HTR) on 18th–19th century Barbados archive documents. The task is to
transcribe each line image exactly as written, keeping historic spelling, capitalisation and abbreviation
marks (`^ ~ & _`).

No competition data is committed. Everything under `data/` is gitignored.

## Metric

The leaderboard score is higher-is-better. It averages a word-level and a character-level term, using mean
Levenshtein edits per line:

```
score = 0.5 * (1 - word_edits_per_line / 12) + 0.5 * (1 - char_edits_per_line / 55)
```

`src/metric_lb.py` implements this exactly and is what we use for all validation scores below.
`src/official_metric.py` reports the older 0.5·WER + 0.5·CER error, shown as "old val error" in the reports.

## Approach

1. **Base models.** Qwen2.5-VL-7B-Instruct and Qwen2.5-VL-32B-Instruct, fine-tuned with 4-bit
   **QLoRA via [Unsloth](https://github.com/unslothai/unsloth)** (`src/train_qwen.py`). LoRA goes on both
   the vision and language towers, with r=16 unless stated. One prompt is used throughout, asking for an
   exact transcription.
2. **Images.** Each line is resized to a fixed height, keeping its aspect ratio, with the width capped
   (`--max-width`). There is no other preprocessing.
3. **v2 recipe.** This is the standard setting for every strong member: 3 epochs, image height 140,
   lr 2e-4, batch 1 × grad-accum 8, AdamW-8bit, linear schedule, greedy decoding, max 64 new tokens.
4. **Diverse members.** Single models level off at a val LB score of about 0.885, so the gains come
   from diversity. Members differ in model size (7B / 32B), LoRA rank, training data (full train, clean
   split), image width cap, and pseudo-labels.
5. **Pseudo-labels.** Some members also train on the 1,374 test images, labelled by our own ensemble
   (`--extra-train`). The process is fully automated with no manual correction, which Zindi staff
   confirmed is allowed. The pseudo-label files are in `pseudo/`. Pseudo-labels make a single model
   slightly worse, but they still add useful diversity to the vote.
6. **Char-level ROVER ensemble** (`src/ensemble_rover.py --method rover-char`). For each line, the weighted
   medoid prediction is the pivot. Every member is aligned to the pivot character by character with
   Levenshtein alignment, which builds a confusion network. Each slot then takes a weighted vote, and ties
   keep the pivot's character. This beats a plain medoid vote on validation.

Things we tried that didn't help (details in `reports/experiment_log.md`): TrOCR-large; beam search;
larger inference height; image enhancement; frozen-vision "starter" settings; Qwen3-VL-8B.

## Hardware

| Members | Hardware | Typical time |
| --- | --- | --- |
| 7B (v1, v2, full, r64, wide) | Kaggle, 1× NVIDIA T4 16 GB | ~2–4 h train, ~1.5 h inference |
| 32B (32b, 32b_pseudo) | Google Colab, 1× NVIDIA A100 | ~6.5 h train with pseudo-labels |
| Ensembling / eval | CPU | a few minutes |

## Repo layout

```
src/train_qwen.py        QLoRA fine-tune + val/test prediction (Unsloth)
src/data_prep.py         validates raw Zindi files, writes data/splits/split_v1.csv + data audit
src/ensemble_rover.py    medoid / word-ROVER / char-ROVER voting
src/metric_lb.py         exact leaderboard metric
src/official_metric.py   0.5*WER + 0.5*CER (older internal metric)
src/check_submission.py  format check against SampleSubmission.csv
src/ensemble.py          original medoid vote (superseded by ensemble_rover.py)
src/error_analysis.py    per-line error breakdown
src/train_trocr.py       TrOCR baseline (dropped)
scripts/make_splits.py   derives split_v1_clean.csv and split_full.csv from split_v1.csv
scripts/eval_ensembles.py  sweeps vote methods / member subsets on val -> reports/ensemble_eval.md
preds/val, preds/test    per-member predictions (ID,Target)
pseudo/                  test pseudo-labels produced by our own ensembles
submissions/             ensemble submission files
reports/                 experiment log, submissions log, ensemble eval, run commands
```

## Reproduce end to end

[`reports/run_commands.md`](reports/run_commands.md) has the exact command for every member and every
ensemble. In outline:

```bash
# 0. Environment (Kaggle / Colab GPU)
pip install -q unsloth pandas

# 1. Data: put the Zindi files in data/raw/ (Train.csv, Test.csv, SampleSubmission.csv, line images)
DATA=data/raw; IMG=data/raw/images
python src/data_prep.py --raw-dir $DATA --seed 42 --split-version v1        # data/splits/split_v1.csv
python scripts/make_splits.py --split data/splits/split_v1.csv               # split_v1_clean, split_full

# 2. Train each member (example: v2), then copy its predictions into preds/
python src/train_qwen.py --data-dir $DATA --img-dir $IMG --split data/splits/split_v1.csv \
    --out runs/v2 --model unsloth/Qwen2.5-VL-7B-Instruct \
    --bs 1 --grad-accum 8 --epochs 3 --img-height 140 --predict-test
cp runs/v2/val_predictions.csv preds/val/v2.csv; cp runs/v2/submission.csv preds/test/v2.csv

# 3. Ensemble (final: 7-member char-ROVER)
python src/ensemble_rover.py --method rover-char \
    --preds v1=preds/test/v1.csv v2=preds/test/v2.csv full=preds/test/full.csv \
            r64=preds/test/r64.csv 32b=preds/test/32b.csv 32bp=preds/test/32b_pseudo.csv \
            wide=preds/test/wide.csv \
    --weights v1=1 v2=1.2 full=1.1 r64=1.2 32b=1.3 32bp=1.0 wide=1.2 --margin 0.0 \
    --out submissions/sub_rover7.csv
python src/check_submission.py submissions/sub_rover7.csv
```

The pseudo-label members depend on earlier ensembles. Build the ensemble first, then train on its output:

- ens-7b4 (medoid of v1, v2, full, r64) gives `pseudo/test_pseudo_ens7b4.csv`, which trains 32b_pseudo.
- rover6 gives `pseudo/test_pseudo_rover6.csv`, which trains the upcoming pseudo members.

**Note:** GPU training (Unsloth/QLoRA) is not fully deterministic, so freshly retrained members can differ
slightly (~±0.001 val) from the committed predictions in `preds/`. The committed `preds/` files are the
exact inputs used for our submissions, and the ensemble commands rebuild every submission byte-for-byte
from them.

## Results

Val = the 410-line `split_v1` validation fold, scored with the exact LB formula. Public = Zindi public
leaderboard, about 20–30% of test, so expect noise of about ±0.01.

### Single members

| Member | Model | Key config | Val LB score | Public |
| --- | --- | --- | --- | --- |
| v1 | 7B | 2 ep, h112 | 0.87963 | 0.8884 |
| v2 | 7B | v2 recipe (3 ep, h140) | 0.88513 | 0.8897 |
| full | 7B | v2 recipe, all train + val | n/a (trained on val) | 0.8873 |
| r64 | 7B | v2 recipe, LoRA r64 | 0.88535 | 0.8908 |
| 32b | 32B | v2 recipe | 0.88478 | not submitted |
| 32b_pseudo | 32B | v2 recipe + pseudo (ens-7b4) | 0.88101 | not submitted |
| wide | 7B | v2 recipe, max-width 4480, clean split | 0.88237 | not submitted |

### Votes

Val ensembles leave out `full`, which was trained on the val lines. The test submissions include it.

| Submission | Members (weights) | Method | Val LB score | Public |
| --- | --- | --- | --- | --- |
| ens-7b4 | v1 1, v2 1.2, full 1.1, r64 1.2 | medoid | n/a | 0.8969 |
| ens-7b4-32b | ens-7b4 + 32b 1.3 | medoid | 0.89771 | 0.9011 |
| sub_best / sub_rover5 | ens-7b4 + 32b 1.3 | char-ROVER | 0.89897 | 0.90115 |
| sub_rover6 | rover5 + 32b_pseudo 1.0 | char-ROVER | 0.90093 | **0.9030** |
| sub_rover7 | rover6 + wide 1.2 | char-ROVER | **0.90257** | 0.9025 |

**Selected finals:** `submissions/sub_rover6.csv` (best public, 0.9030) and `submissions/sub_rover7.csv`
(best val, 0.9026; public 0.9025).

The full sweep over methods and member subsets, with paired bootstrap CIs, is in
[`reports/ensemble_eval.md`](reports/ensemble_eval.md). The per-run history is in
[`reports/experiment_log.md`](reports/experiment_log.md) and
[`reports/submissions_log.md`](reports/submissions_log.md).

## Excluded training rows

Zindi staff allow dropping training rows that are clearly corrupted, as long as the drop is documented.
A participant posted 21 suspect IDs, but we drop only the 3 that a second participant independently
corroborated. They are removed in `split_v1_clean.csv` (`scripts/make_splits.py`) and used by `wide` and
later members:

| ID | Reason |
| --- | --- |
| `79tMUVyfIdy3GzkG` | image empty / unreadable |
| `F8DYDDp2AvW9Dytw` | transcription does not match the image |
| `JU7lRwk3jKkus24Z` | transcription does not match the image |

Earlier members (v1, v2, full, r64, 32b, 32b_pseudo) trained on the unfiltered `split_v1`.

## Licences and data

- **Qwen2.5-VL-7B-Instruct**: Apache-2.0
- **Qwen2.5-VL-32B-Instruct**: Apache-2.0 (Unsloth 4-bit quantisation of the same weights)
- Both allow commercial use, as the competition rules require. Qwen2.5-VL-3B (Qwen Research licence) was
  used only in early experiments and is in **no** final submission.
- **Unsloth**, **TRL**, **Transformers**: Apache-2.0.
- **No external data.** Training uses only the competition's `Train.csv` images and labels, plus
  pseudo-labels that our own models produced automatically on the competition's test images.
