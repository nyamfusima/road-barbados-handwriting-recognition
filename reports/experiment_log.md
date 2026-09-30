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
| 2026-10-01 | ens-7b3 | ensemble | test only | Medoid 7B v1 + 7B v2 + 7B full | w 1, 1.2, 1.1 | n/a | n/a | n/a | All 7B (Apache 2.0, licence-safe); public pending | pending |

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

## Rules (staff rulings)
- Only competition data for training/fine-tuning; no external datasets
- Pretrained base must allow commercial use. Qwen2.5-VL-3B (Qwen Research licence) is NOT allowed; Qwen2.5-VL-7B (Apache 2.0) is allowed
- Pseudo-labelling / self-training on test images IS allowed if fully automated

## Lessons so far
- Validation tracks public within ~0.01; public set is small (~20-30% of test) and noisy
- 3B -> 7B gave -0.013 val error; epochs/resolution changes gave gains within noise
- Ensembles only help when members are similarly strong
- Current recipe plateau: public ~0.887-0.890 across three 7B runs

## Next levers (untested)
- 7B with higher LoRA rank (capacity; underfitting signal)
- Pseudo-labelling on test images (allowed)
- Ensemble of strong 7B members only
- Drop clearly mislabelled training rows (documented)