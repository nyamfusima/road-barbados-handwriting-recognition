# Split leakage

For each validation line, the closest training label by character edit distance, as a fraction of the validation line's length (a CER). Higher shares at low CER mean more phrases are shared between train and validation, so validation scores are more optimistic.

| split | share_cer<=0.00 | share_cer<=0.05 | share_cer<=0.10 | share_cer<=0.20 | share_cer<=0.30 | mean_nearest_cer | lookup_cer |
|---|---|---|---|---|---|---|---|
| block split (split_v1) | 0.003 | 0.005 | 0.010 | 0.045 | 0.090 | 0.487 | 0.485 |
| random line split (mean of 5) | 0.004 | 0.008 | 0.019 | 0.048 | 0.112 | 0.482 | 0.479 |

`lookup_cer` is the CER of predicting each validation line as its closest training label with no image at all: the amount of validation score that memorised phrases alone can explain.
