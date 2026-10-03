# Experiment log

One row per run. Change one important thing at a time. Scores are on the split named in the row.
Metric: 0.5 x WER + 0.5 x CER (length-weighted). Leaderboard score is about 1 - this error.

| Date | Run name | Workstream | Split | Model / method | Key config | Val WER | Val CER | Val error (0.5W+0.5C) | Notes | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-29 | trocr-zeroshot | trocr | 5 train lines | TrOCR-large-handwritten, no training | beam 4, max_tok 128 | n/a | n/a | n/a | Clean lines readable; tall crops fail (384x384 squash) | Fine-tune |
| 2026-09-29 | smoke-fold1 | trocr | split_v1 (64 val) | TrOCR-large full fine-tune | lr 4e-5, bs 2x2 GPUs, accum 8, 80 steps, fold 1, beam 4 | 0.36 | 0.16 | 0.259 | Public 0.7519 | Drop (VLM far ahead) |
| 2026-09-29 | qwen-smoke | vlm | split_v1 (48 val) | Qwen2.5-VL-3B QLoRA r16 | 40 steps on 300 lines, h112 | 0.41 | 0.22 | 0.315 | Pipeline check | Full run |
| 2026-09-29 | qwen-lora-v1 | vlm | split_v1 (410 val) | Qwen2.5-VL-3B QLoRA r16 | lr 2e-4, bs 2 x accum 4, 2 ep, h112, greedy | 0.203 | 0.068 | 0.1353 | Public 0.8690. LICENCE: 3B base is Qwen Research licence (non-commercial), not allowed for finals | Drop (licence) |
| 2026-09-30 | qwen-7b-v1 | vlm | split_v1 (410 val) | Qwen2.5-VL-7B QLoRA r16 | bs 1 x accum 8, 2 ep, h112, greedy | 0.187 | 0.058 | 0.1225 | Public 0.8884; 2h15m train, 3h40m total | Keep (ensemble member) |
| 2026-09-30 | qwen-3b-v2 | vlm | split_v1 | Qwen2.5-VL-3B QLoRA r16 | 3 ep, h140 (teammate) | n/a | n/a | n/a | Public 0.8744; licence-banned (3B) | Drop (licence) |
| 2026-09-30 | ens3 | ensemble | test only | Medoid 3B v1 + 3B v2 + 7B v1 | w 1,1,1.5 (src/ensemble.py) | n/a | n/a | n/a | Public 0.8874; weak members outvote 7B; contains 3B so not eligible | Drop |
| 2026-09-30 | qwen3-8b-smoke | vlm | split_v1 | Qwen3-VL-8B QLoRA | smoke 40 steps (teammate) | - | - | - | >2h for a 20-min smoke; broken (processor/stop token); no time to debug | Drop |
| 2026-09-30 | qwen-7b-v2 | vlm | split_v1 (410 val) | Qwen2.5-VL-7B QLoRA r16 | parent qwen-7b-v1; 3 ep, h140 | 0.179 | 0.055 | 0.1170 | Public 0.8897 (best); 3h54m train, 5h29m total; gain vs v1 within val noise | Keep (best single) |
| 2026-09-30 | 3b-bigimg | vlm | split_v1 | Qwen2.5-VL-3B QLoRA | teammate's larger-image resize (exact settings TBC) | 0.205 | 0.070 | 0.1374 | No gain vs 3B v1 (0.1353); bigger images do not help | Drop |
| 2026-09-30 | qwen-7b-full | vlm | full (train+val, 4,098 lines) | Qwen2.5-VL-7B QLoRA r16 | v2 recipe; 40 val lines duplicated as leaky sanity check; 1,539 steps | 0.138* | 0.044* | 0.0908* | *Leaky: even seen lines ~9% wrong -> underfitting signal. Public 0.8873 | Keep (ensemble member) |
| 2026-10-01 | ens-7b3 | ensemble | test only | Medoid 7B v1 + 7B v2 + 7B full | w 1, 1.2, 1.1 | n/a | n/a | n/a | built, not submitted (superseded by ens-7b4) | Superseded |
| 2026-10-01 | qwen-7b-v2-beam4 | vlm | split_v1 (410 val) | 7B v2 LoRA, inference only | num_beams 4 | 0.182 | 0.054 | 0.1179 | vs greedy 0.1170; no gain, 1.5x slower | Drop |
| 2026-10-01 | qwen-7b-r64 | vlm | split_v1 (410 val) | Qwen2.5-VL-7B QLoRA r64 | v2 recipe, rank 16->64 (206M trainable), 9.0 s/step | 0.180 | 0.054 | 0.1169 | = v2; capacity not the bottleneck. Public 0.8908 | Keep (ensemble member) |
| 2026-10-01 | 7b-enhance (teammate) | vlm | split_v1 | Qwen2.5-VL-7B QLoRA | --enhance (autocontrast + sharpen), h112, beams 3 | - | - | ~0.13 | worse than v2 | Drop |
| 2026-10-01 | qwen-7b-v2-h168 | vlm | split_v1 (410 val) | 7B v2 LoRA, inference only | predict at h168 (trained at h140) | 0.183 | 0.057 | 0.1202 | public 0.8859; 71% identical to v2 | Drop |
| 2026-10-01 | ens-7b4 | ensemble | test only | Medoid 7B v1 + v2 + full + r64 | w 1, 1.2, 1.1, 1.2 | n/a | n/a | n/a | Public 0.8969 (best); +0.007 over best single | Keep (best) |
| 2026-10-01 | ens-7b5 | ensemble | test only | ens-7b4 + v2@h168 | w 1, 1.2, 1.1, 1.2, 0.8 | n/a | n/a | n/a | Public 0.8957; weaker member diluted the vote | Drop |
| 2026-10-02 | qwen-7b-starter (teammate) | vlm | split_v1 (410 val) | Qwen2.5-VL-7B QLoRA | starter settings: frozen vision, lr 3e-5, alpha 32, dropout 0.05, accum 16 | 0.239 | 0.090 | 0.1647 | much worse than v2 (0.1170): underfits | Drop |
| 2026-10-02 | qwen-32b-smoke | vlm | split_v1 (48 val) | Qwen2.5-VL-32B QLoRA r16 (Colab A100) | smoke 40 steps, h140 | 0.365 | 0.173 | 0.269 | 3B smoke was 0.315 | Full run |
| 2026-10-02 | qwen-32b-v2recipe (teammate, Colab A100) | vlm | split_v1 (410 val) | Qwen2.5-VL-32B QLoRA r16 | v2 recipe (3 ep, h140, accum 8) | 0.178 | 0.057 | 0.1171 | = 7B v2; only ~27% identical to 7Bs on test (very diverse) | Keep (ensemble member) |
| 2026-10-02 | qwen-32b-pseudo (Colab A100) | vlm | split_v1 (410 val) | Qwen2.5-VL-32B QLoRA r16 | v2 recipe + 1,374 test pseudo-labels from ens-7b4; 1,899 steps, ~12 s/step, 6h32m train | 0.181 | 0.061 | 0.1208 | worse than clean 32B: pseudo-labels hurt; test preds 54% identical to ens-7b4 (copies the vote) | Drop as method; ensemble only if val eval says so |
| 2026-10-02 | 7b-full-pseudo seed 7 (teammate, Kaggle) | vlm | full + pseudo | Qwen2.5-VL-7B QLoRA | full data + pseudo, seed 7 | - | - | - | no val possible (trained on val); pseudo shown to hurt | Exclude unless public proves otherwise |
| 2026-10-02 | ens-7b4-32b | ensemble | test only | medoid 7B v1+v2+full+r64 + 32B | w 1, 1.2, 1.1, 1.2, 1.3 | n/a | n/a | n/a | public 0.9011 | Keep |
| 2026-10-02 | ens-rover5 | ensemble | val + test | char-level ROVER, same 5 members | same weights | | | val LB 0.8990 vs medoid 0.8977 (+0.0013, CI -0.0007..+0.0032) | public 0.90115 (best) | Keep (current best) |
| 2026-10-03 | ens-rover6 | ensemble | val only (test built, not submitted) | char-level ROVER, 5 members + 32b_pseudo (w=1.0) | same weights + 32bp=1.0 | | | val LB 0.90093 vs S4 char-ROVER 0.89897 (+0.00196, CI -0.00052..+0.00455) | submissions/sub_rover6.csv built; public 0.9030 (new best, Final 1); w=1.0 beat w=0.7 for 32bp | Keep (current best) |
| 2026-10-03 | qwen-7b-wide | vlm | split_v1_clean (410 val, 3 bad rows excluded) | Qwen2.5-VL-7B QLoRA r16 | v2 recipe + --max-width 4480 | 0.183 | 0.057 | 0.1197 | width fix not better alone than v2 (0.1170); very diverse (25% identical to 32B on test) | ensemble member if val eval says so |
| 2026-10-03 | ens-rover7 | ensemble | val only (test built, not submitted) | char-level ROVER, 6 members + wide (w=1.2) | same weights + wide=1.2 | | | val LB 0.90257 vs S6 char-ROVER 0.90093 (+0.00164, CI -0.00128..+0.00525) | submissions/sub_rover7.csv built; public 0.9025 (Final 2); w=1.2 beat w=1.0 and w=0.7 for wide | Keep (recommend submit) |

Workstreams: `data-metric`, `trocr`, `vlm`, `ensemble`, `analysis`, `submission`.
Decision values: keep, rerun, improve, ensemble, drop.

## Data facts (Day 1 audit)
- Train 4,098 / Test 1,374 lines; split_v1 = 3,688 train / 410 val, seed 42, md5 61d5a1a939e97a442dc635bcb63adb1d
- Median aspect ratio 17:1; 99.2% of lines wider than 8:1
- Token p99 = 28, max = 35, so max_new_tokens = 48
- Labels use ^ (872x), ~, &, _ as abbreviation marks; no long s; only 0.5% duplicate labels
- 0 train labels start with a quote, so stripping a leading quote in post-processing is safe

## Known code issue (found 2026-09-30)
- load_img in src/train_qwen.py caps width at --max-width (default 2240). Lines wider than ~16:1 get their HEIGHT reduced to fit, so over half of lines never reached the requested --img-height. A larger-image test (teammate 3B) and the community thread both show bigger images do not help, so no patch was applied.

## Error analysis (qwen-7b-v1 val, src/error_analysis.py)
- No disasters: 0 repetition loops, 1 line with CER > 50%, worst 20% of lines hold 52% of char errors
- Case ~0.016 of error; punctuation ~0.010; ^ ~0.003; whitespace ~0
- Model modernises historic spelling (prsents->presents, thiese->these, covent->covenant)
- Word splits/merges drive WER (space inserted 77x, deleted 55x)
- Short lines (30-50 chars) worst: CER 7.7% vs 3.9% for 70-90 chars
- Labels contain noise; some error is irreducible
- Multi-line crops: no staff ruling on test convention; ~0.1% of images are multi-line; our 7B v1 over-read only 1/410 val lines -> no special handling

## Community findings (Zindi thread "Five things that DON'T work", 17 Sep)
- Exact LB formula: 0.5*(1 - word_edits_per_line/12) + 0.5*(1 - char_edits_per_line/55); 1 word edit = 4.58 char edits
- Val noise sd ~0.005 on 820 lines -> ~0.007 on our 410; gains < 0.01 are not reliable
- Measured dead ends: lexicon recasing, markup (^ : &) fixes, OOV snapping, over-transcription guard, higher resolution for the high-res scan batch
- ~1 in 70 lines mislabelled; clearly corrupted train rows may be dropped if documented (thread 33891)
- Thread "OCR models, VLM, or custom?": a team slightly above 0.91 used Qwen + LoRA + GRPO + beam search and suspected bad data
- Multi-line crops: no staff ruling on test convention; ~0.1% of images are multi-line; our 7B v1 over-read only 1/410 val lines -> no special handling
- Competition CLOSES 04 Oct 2026 (reveal 05 Oct), not 05 Oct
- The 21 "corrupted" IDs were posted by a participant (Alpcan_Cepik), NOT confirmed by staff; staff only allow excluding clearly corrupted rows with documentation. Only 3 are corroborated by another participant: 79tMUVyfIdy3GzkG (empty/unreadable), F8DYDDp2AvW9Dytw and JU7lRwk3jKkus24Z (text doesn't match label). We exclude only these 3 in new runs.
- Kansallisarkisto/multicentury-htr-model is Apache-2.0 (allowed) but TrOCR-based (384x384 squash) -> skipped
- Pseudo-labelling measured: hurts (32B 0.1171 -> 0.1208)

## Rules (staff rulings)
- Only competition data for training/fine-tuning; no external datasets
- Pretrained base must allow commercial use. Qwen2.5-VL-3B (Qwen Research licence) is NOT allowed; Qwen2.5-VL-7B (Apache 2.0) is allowed
- Pseudo-labelling / self-training on test images IS allowed if fully automated

## Lessons so far
- Single models plateau at ~0.117 val (7B and 32B alike); capacity, beams, image size, preprocessing, starter settings and pseudo-labels don't break it
- Diversity is the working lever: adding the 32B (27% overlap) gave +0.004 public; weaker members dilute the vote
- Char-level ROVER ≈ medoid (+0.0013 val, +0.00005 public): vote method matters little, members matter more
- Local LB-formula val score tracks public (medoid val 0.8977 vs public 0.9011)

## Next levers
- 7B width fix: --max-width 4480 at h140 (lines >16:1 were being shrunk below h140), 3 bad rows excluded (Kaggle, Sat 3 Oct)
- Clean 32B seed 7 (teammate's Colab, Sat 3 Oct)
- Final vote: char-ROVER over the strongest diverse members; pick 2 finals before close on 4 Oct