# Experiment log

One row per run. Change one important thing at a time. Scores are on the split named in the row.

| Date | Run name | Workstream | Commit | Split | Model / method | Key config (lr, bs, epochs, img, max_tokens, beams, len_pen) | Val WER | Val CER | Val score (0.5W+0.5C) | Notes / worst errors | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-29 | smoke-fold1 | trocr | main @ 7d4d03b | split_v1 (seeded random, seed 42, 10% val); smoke = 400 train / 64 val lines | TrOCR large handwritten, full fine-tune, 2xT4 DDP | lr 4e-5, bs 2/GPU x 2 GPUs, grad-accum 4, grad-ckpt, 80 steps (3.2 epochs of 400 lines), img fold 1, no aug, max_new_tokens 48, beams 4, len_pen default | 0.375 | 0.156 | 0.266 | Pipeline check only, not a result. 229 s train, 2.3 s/step. Default bs 4 ran out of memory (14.5 GB/GPU); fixed with --bs 2 --grad-ckpt. One grad_norm nan at step 40, loss kept falling. Two checkpoints (~6.7 GB each) plus best model nearly filled the 20 GB disk. | keep (default for trocr-large-v1) |
| 2026-09-29 | smoke-fold4 | trocr | main @ 7d4d03b | same as smoke-fold1 | same as smoke-fold1 | same as smoke-fold1 except img fold 4 (line cut into 4 pieces, stacked) | 1.290 | 0.864 | 1.077 | Pipeline check only. Same speed and memory as fold 1, but still mostly wrong after 80 steps (train loss 40.6 vs 6.3); beam search made it worse (greedy 0.902 -> beam 1.077, WER > 1 = over-long output). First attempt crashed saving a checkpoint (disk full); rerun completed. Same config gave step-40 score 0.839 then 0.917, so noise is large. | drop for now; revisit in a longer run if GPU time is spare |

Workstreams: `data-metric`, `trocr`, `vlm`, `analysis`, `submission`.
Decision values: keep, rerun, improve, ensemble, drop.
