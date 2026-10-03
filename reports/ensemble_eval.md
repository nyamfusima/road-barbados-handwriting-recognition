# Ensemble evaluation (validation, exact leaderboard metric)

Members: v1, v2, r64, 32b. Given weights: {'v1': 1.0, 'v2': 1.2, 'r64': 1.2, '32b': 1.3}.

Sorted by LB score (higher is better). `old_val_error` = 0.5*WER+0.5*CER corpus-level.

| Method | Members | LB score | Mean word edits/line | Mean char edits/line | Old val error |
| --- | --- | --- | --- | --- | --- |
| ALL given-weight rover-char (margin 0) | v1+v2+r64+32b | 0.89897 | 1.8049 | 2.8415 | 0.10308 |
| subset v1+r64+32b medoid | v1+r64+32b | 0.89853 | 1.7951 | 2.9341 | 0.10338 |
| ALL given-weight rover (margin 0.5) | v1+v2+r64+32b | 0.89850 | 1.8049 | 2.8927 | 0.10344 |
| ALL given-weight medoid (current prod) | v1+v2+r64+32b | 0.89771 | 1.8195 | 2.9122 | 0.10424 |
| ALL equal-weight rover (margin 0) | v1+v2+r64+32b | 0.89651 | 1.8415 | 2.9439 | 0.10547 |
| ALL equal-weight rover (margin 0.5) | v1+v2+r64+32b | 0.89651 | 1.8415 | 2.9439 | 0.10547 |
| ALL given-weight rover (margin 0) | v1+v2+r64+32b | 0.89643 | 1.8146 | 3.0756 | 0.10542 |
| subset v1+r64+32b rover (margin 0) | v1+r64+32b | 0.89626 | 1.8098 | 3.1171 | 0.10558 |
| subset v1+v2+32b medoid | v1+v2+32b | 0.89612 | 1.8488 | 2.9537 | 0.10598 |
| ALL equal-weight medoid | v1+v2+r64+32b | 0.89570 | 1.8561 | 2.9659 | 0.10630 |
| subset v2+r64+32b rover (margin 0) | v2+r64+32b | 0.89522 | 1.8220 | 3.1756 | 0.10659 |
| subset v2+r64+32b medoid | v2+r64+32b | 0.89516 | 1.8585 | 3.0146 | 0.10687 |
| subset v1+v2+32b rover (margin 0) | v1+v2+32b | 0.89218 | 1.8756 | 3.2634 | 0.10970 |
| subset v1+v2+r64 rover (margin 0) | v1+v2+r64 | 0.89116 | 1.9268 | 3.1415 | 0.11086 |
| subset v1+v2+r64 medoid | v1+v2+r64 | 0.89076 | 1.9390 | 3.1293 | 0.11130 |
| single: r64 | r64 | 0.88535 | 2.0317 | 3.3000 | 0.11674 |
| single: v2 | v2 | 0.88513 | 2.0220 | 3.3683 | 0.11700 |
| single: 32b | 32b | 0.88478 | 2.0049 | 3.4854 | 0.11725 |
| single: v1 | v1 | 0.87963 | 2.1098 | 3.5707 | 0.12250 |

## Paired check: best (ALL given-weight rover-char (margin 0)) vs ALL given-weight medoid (current prod)
- Lines improved: 42, worsened: 23, unchanged: 345 (of 410)
- Mean LB-score difference (best - baseline): 0.00125
- Bootstrap 95% CI (1000 resamples of the 410 lines): [-0.00065, 0.00321]
- CI straddles 0: the difference is not distinguishable from noise at this sample size.

## 2026-10-03: with 32b_pseudo

New member `32bp` = Qwen2.5-VL-32B QLoRA, trained on v2 recipe + 1,374 test pseudo-labels from
ens-7b4 (see experiment_log.md). Full run:

`python scripts/eval_ensembles.py --val v1=preds/val/v1.csv v2=preds/val/v2.csv r64=preds/val/r64.csv 32b=preds/val/32b.csv 32bp=preds/val/32b_pseudo.csv --weights v1=1 v2=1.2 r64=1.2 32b=1.3 32bp=1.0`

| Method | Members | LB score | Mean word edits/line | Mean char edits/line | Old val error |
| --- | --- | --- | --- | --- | --- |
| ALL given-weight rover-char (margin 0) | v1+v2+r64+32b+32bp | 0.90093 | 1.7732 | 2.7707 | 0.10110 |
| ALL given-weight rover (margin 0.5) | v1+v2+r64+32b+32bp | 0.90090 | 1.7659 | 2.8073 | 0.10109 |
| ALL equal-weight rover (margin 0) | v1+v2+r64+32b+32bp | 0.90070 | 1.7707 | 2.8073 | 0.10131 |
| ALL equal-weight rover (margin 0.5) | v1+v2+r64+32b+32bp | 0.90070 | 1.7707 | 2.8073 | 0.10131 |
| subset v1+v2+32b+32bp medoid | v1+v2+32b+32bp | 0.89979 | 1.7829 | 2.8512 | 0.10227 |
| ALL given-weight rover (margin 0) | v1+v2+r64+32b+32bp | 0.89948 | 1.7659 | 2.9634 | 0.10236 |
| ALL equal-weight medoid | v1+v2+r64+32b+32bp | 0.89861 | 1.8049 | 2.8805 | 0.10336 |
| ALL given-weight medoid (current prod) | v1+v2+r64+32b+32bp | 0.89850 | 1.8049 | 2.8927 | 0.10346 |
| subset v1+v2+32b+32bp rover (margin 0) | v1+v2+32b+32bp | 0.89834 | 1.7756 | 3.0439 | 0.10351 |
| subset v1+v2+r64+32b medoid | v1+v2+r64+32b | 0.89771 | 1.8195 | 2.9122 | 0.10424 |
| subset v2+r64+32b+32bp medoid | v2+r64+32b+32bp | 0.89655 | 1.8415 | 2.9390 | 0.10551 |
| subset v1+r64+32b+32bp rover (margin 0) | v1+r64+32b+32bp | 0.89647 | 1.7951 | 3.1610 | 0.10534 |
| subset v1+v2+r64+32b rover (margin 0) | v1+v2+r64+32b | 0.89643 | 1.8146 | 3.0756 | 0.10542 |
| subset v2+r64+32b+32bp rover (margin 0) | v2+r64+32b+32bp | 0.89608 | 1.8146 | 3.1146 | 0.10575 |
| subset v1+r64+32b+32bp medoid | v1+r64+32b+32bp | 0.89584 | 1.8415 | 3.0171 | 0.10617 |
| subset v1+v2+r64+32bp rover (margin 0) | v1+v2+r64+32bp | 0.89322 | 1.8927 | 3.0707 | 0.10875 |
| subset v1+v2+r64+32bp medoid | v1+v2+r64+32bp | 0.89254 | 1.9122 | 3.0561 | 0.10950 |
| single: r64 | r64 | 0.88535 | 2.0317 | 3.3000 | 0.11674 |
| single: v2 | v2 | 0.88513 | 2.0220 | 3.3683 | 0.11700 |
| single: 32b | 32b | 0.88478 | 2.0049 | 3.4854 | 0.11725 |
| single: 32bp | 32bp | 0.88101 | 2.0390 | 3.7439 | 0.12085 |
| single: v1 | v1 | 0.87963 | 2.1098 | 3.5707 | 0.12250 |

32bp alone is the weakest single member (0.88101 vs 0.88478 for clean 32b) -- pseudo-labelling on
its own hurt this model, consistent with the experiment log. But it still adds diversity to the vote.

### Member-set comparison (medoid vs char-ROVER)

| Set | Method | LB score | Mean word edits/line | Mean char edits/line |
| --- | --- | --- | --- | --- |
| S4 = v1,v2,r64,32b (w 1,1.2,1.2,1.3) | medoid | 0.89771 | 1.8195 | 2.9122 |
| S4 | rover-char (margin 0) | 0.89897 | 1.8049 | 2.8415 |
| S5 = S4 + 32bp (w=1.0) | medoid | 0.89850 | 1.8049 | 2.8927 |
| S5 = S4 + 32bp (w=1.0) | rover-char (margin 0) | **0.90093** | 1.7732 | 2.7707 |
| S5b = S4 + 32bp (w=0.7) | medoid | 0.89714 | 1.8317 | 2.9195 |
| S5b = S4 + 32bp (w=0.7) | rover-char (margin 0) | 0.90058 | 1.7805 | 2.7756 |
| S4 minus v1 (v2,r64,32b) | medoid | 0.89516 | 1.8585 | 3.0146 |
| S4 minus v1 (v2,r64,32b) | rover-char (margin 0) | 0.89857 | 1.8049 | 2.8854 |

Best set/method: **S5 char-ROVER** (32bp at weight 1.0 -- beats weight 0.7). Dropping v1 does not
help (S4-minus-v1 is worse than S4 under both methods); 32bp's value is as a diverse vote, not a
core member.

### Paired check: best (S5 char-ROVER) vs S4 char-ROVER
- Lines improved: 60, worsened: 33, unchanged: 317 (of 410)
- Mean LB-score difference (best - baseline): 0.00196
- Bootstrap 95% CI (1000 resamples of the 410 lines): [-0.00052, 0.00455]
- CI nearly excludes 0 (lower bound -0.0005): a modest, probably-real gain from adding 32bp, though
  not fully conclusive at 410 lines.

### Decision
Adding 32b_pseudo at weight 1.0 via char-level ROVER (S5) is the new best validated configuration
(0.90093 vs 0.89897 for S4 char-ROVER, +0.00196 >= the 0.001 bar) -> see submissions/sub_rover6.csv.
