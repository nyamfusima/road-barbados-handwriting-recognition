# R.O.A.D. Barbados Historic Handwriting Challenge

Code for the Zindi R.O.A.D. Barbados Historic Handwriting Challenge. This repo is
code-only — no competition data is committed to GitHub.

## Local setup

1. Download the competition files from Zindi and lay them out like this:
   ```
   data/     Train.csv, Test.csv, SampleSubmission.csv, Starters.zip
   images/   all line images (<ID>.jpg)
   ```
   `data/` and `images/` are gitignored, so nothing here ever reaches GitHub.
   Extract `Starters.zip` into `data/starters/` if you want the starter code.
2. Keep it that way — never `git add` files under `data/` or `images/`, raw
   CSVs, zips, images, checkpoints, model weights, or submission files.
   `.gitignore` already blocks these, but double-check `git status` before
   committing.
3. Validate the files:
   ```bash
   python src/data_prep.py --input-dir data
   ```
4. For training, use the local GPU or upload `data/` and `images/` to a
   **private** Kaggle Dataset — do not push data to GitHub.

## Data and validation

```bash
python src/audit.py        # writes reports/day1_data_audit.md (lengths, charset, image sizes)
python src/make_split.py   # writes reports/split_v1.json (train/val IDs, seed 42)
```

- `reports/split_v1.json` is the shared validation split (IDs only). Everyone
  should validate on it so scores are comparable. Train.csv has no page or
  document column and contains many crops of the same formulaic phrases, so the
  split holds out contiguous blocks of rows rather than random lines.
- **Metric.** Confirmed on the Zindi Evaluation tab: `final = 0.5 * CER + 0.5 * WER`,
  length-weighted (total edit distance / total reference length). The starter pack's
  `wer.py` / `cer.py` are still not included, so `src/official_metric.py` is our own
  implementation with the same function names (`compute_weighted_wer` /
  `compute_weighted_cer`); `weight_factor` is ignored. `score(refs, hyps)["final"]` is the
  decision metric. Per-line and 0.7/0.3 variants are diagnostics only.
- Before every upload: `python src/check_submission.py --sub submission.csv`

## Repo layout

- `src/` — project code
  - `src/data_prep.py` — validates the raw Zindi files
  - `src/audit.py` — data audit report
  - `src/make_split.py` — fixed validation split
  - `src/official_metric.py` — provisional CER / WER scoring (see above)
  - `src/check_submission.py` — submission format check
- `reports/` — audit report, split file, and (later) experiment and submission logs
- `data/`, `images/` — gitignored; local-only raw data and starter pack
