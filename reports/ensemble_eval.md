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
