# Submissions log

Run `python src/check_submission.py <file>` before every upload.
Limits: 5 per day, 200 total. Competition closes 05 Oct 2026 (confirm time on Zindi).
Leaderboard score: higher is better (about 1 - error).
Public LB = ~20-30% of test (~300-400 lines), so expect +/-0.01 noise. Trust validation first.

| Date / time (SAST) | File | Run name | Val error | Public score | Rank | Final candidate? | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-29 ~18:00 | smoke-fold1/submission.csv | smoke-fold1 | 0.259 | 0.7519 | 536 | no | TrOCR-large 80 steps, fold 1, beam 4, leading-quote strip |
| 2026-09-29 ~22:00 | qwen-lora-v1/submission.csv | qwen-lora-v1 | 0.1353 | 0.8690 | 411 | no | Qwen2.5-VL-3B QLoRA r16, 2 ep, h112, greedy; val matched public |
| 2026-09-30 ~07:00 | qwen-7b-v1/submission.csv | qwen-7b-v1 | 0.1225 | 0.8884 | 359 | yes (current best) | Qwen2.5-VL-7B QLoRA r16, 2 ep, h112, greedy; public ~0.01 above val-implied |
| 2026-09-30 ~12:05 | qwen-3b-v2-clean.csv | qwen-3b-v2 | pending | 0.8744 | - | ensemble member | 3B, 3 ep, h140 (teammate run); +0.005 vs 3B v1, within noise |
| 2026-09-30 ~12:15 | submission_ens3.csv | ens3 | n/a | 0.8874 | - | no | Medoid ensemble 3B v1 + 3B v2 + 7B v1 (w 1,1,1.5); weak 3Bs outvote 7B |

## Leaderboard reference
- 29 Sep: 1st 0.9399, 5th 0.9331
- 30 Sep: rank 100 = 0.9240 (ranks 95-100 cluster at ~0.92403 -> likely a shared recipe), rank ~300 = 0.904
- Team target: top 100 -> need public ~0.925 (val error ~0.075)

## Lessons so far
- Validation tracks public within ~0.01; public set is small and noisy.
- Bigger model (3B -> 7B) gave -0.013 val error; h140 + 3 epochs gave only small gains on 3B.
- Ensembles only help when members are similarly strong.
- Error analysis (7B v1): no disasters; errors spread out; model modernises historic spelling; labels contain noise.

## Final selection
- Submission 1:
- Submission 2:
- Reasoning (validation + public evidence):