# Submissions log

Run `python src/check_submission.py <file>` before every upload.
Limits: 5 per day, 200 total. Competition CLOSES 04 Oct 2026 (reveal 05 Oct).
Leaderboard score: higher is better (about 1 - error).
Public LB = ~20-30% of test (~300-400 lines), so expect +/-0.01 noise. Trust validation first.

| Date / time (SAST) | File | Run name | Val error | Public score | Rank | Final candidate? | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-29 ~18:00 | smoke-fold1/submission.csv | smoke-fold1 | 0.259 | 0.7519 | 536 | no | TrOCR-large 80 steps, fold 1, beam 4 |
| 2026-09-29 ~22:00 | qwen-lora-v1/submission.csv | qwen-lora-v1 | 0.1353 | 0.8690 | 411 | NO - licence | Qwen2.5-VL-3B (Qwen Research licence, not allowed) |
| 2026-09-30 ~07:00 | qwen-7b-v1/submission.csv | qwen-7b-v1 | 0.1225 | 0.8884 | 359 | yes | 7B, 2 ep, h112 |
| 2026-09-30 ~12:05 | qwen-3b-v2-clean.csv | qwen-3b-v2 | n/a | 0.8744 | - | NO - licence | 3B, 3 ep, h140 (teammate) |
| 2026-09-30 ~12:15 | submission_ens3.csv | ens3 | n/a | 0.8874 | - | NO - licence | Medoid 3B v1 + 3B v2 + 7B v1; contains 3B |
| 2026-09-30 ~16:55 | submission_7b_v2.csv | qwen-7b-v2 | 0.1170 | 0.8897 | 356 | yes (current best) | 7B, 3 ep, h140 |
| 2026-10-01 | submission_7b_full.csv | qwen-7b-full | n/a (leaky 0.0908) | 0.8873 | - | yes | 7B, full data (train+val), 3 ep, h140 |
| 2026-10-01 | submission_ens_7b3.csv | ens-7b3 | n/a | pending | - | candidate | Medoid 7B v1 + v2 + full (1, 1.2, 1.1); all 7B |
| 2026-10-01 | submission_7b_r64.csv | qwen-7b-r64 | 0.1169 | 0.8908 | - | yes | 7B r64 |
| 2026-10-01 | submission_ens_7b4.csv | ens-7b4 | n/a | 0.8969 | - | yes (current best) | medoid 7B v1 + v2 + full + r64 (1, 1.2, 1.1, 1.2) |
| 2026-10-01 | v2 at h168 (submitted twice) | qwen-7b-v2-h168 | 0.1202 | 0.8859 | - | no | test-time height change hurts |
| 2026-10-01 | submission_ens_7b5.csv | ens-7b5 | n/a | 0.8957 | - | no | adding weaker h168 member lowered score |
| 2026-10-02 | submission_ens_7b4_32b.csv | ens-7b4-32b | n/a | 0.9011 | ~332 | yes | medoid 4x7B + 32B |
| 2026-10-02 | submissions/sub_best.csv | ens-rover5 | val 0.8990 | 0.90115 | - | yes (current best) | char-ROVER, 221 lines differ from medoid |

## Leaderboard reference
- 29 Sep: 1st 0.9399, 5th 0.9331
- 30 Sep: rank 100 = 0.9240 (ranks 95-100 at ~0.92403), rank ~300 = 0.904
- Team target: top 100 -> public ~0.925

## Lessons so far
- Validation tracks public within ~0.01; public set is small and noisy.
- Bigger model (3B -> 7B) gave -0.013 val error; h140 + 3 epochs gave only small gains on 3B.
- Ensembles only help when members are similarly strong.
- Error analysis (7B v1): no disasters; errors spread out; model modernises historic spelling; labels contain noise.

## Final selection
- RULE: only 7B-based (Apache 2.0) submissions are eligible. Excluded: qwen-lora-v1, qwen-3b-v2, ens3.
- Pseudo-labelling on test is allowed if fully automated.
- Current best: ens-rover5 (0.90115); runner-up ens-7b4-32b medoid (0.9011)
- Plan: Final 1 = best validated vote (char-ROVER); Final 2 = structurally different vote (e.g. medoid, or different member set)
- Deadline: select both before close on 04 Oct 2026
- Reasoning (validation + public evidence):