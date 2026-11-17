# Final test-set evaluation: FD001

- **Date (UTC):** 2026-11-17 02:42
- **Commit:** `30cf99a6ba93094d66e99f1f8354336bb232b0f5` (uncommitted changes: false)
- **Test engines:** 100, each evaluated at its last observed cycle against the uncapped true RUL
- **Frozen setup:**
  - LightGBM on causal window features (windows 10, 30), training target capped at 125, trained on all training engines
    (ADRs 0001 and 0002).
  - Intervals: Mondrian split conformal with alpha 0.1, calibrated on the 5-fold out-of-fold
    residuals of the training engines (rows with true RUL <= 150; ADR 0003).
- **Uncertainty:** 95% bootstrap intervals over engines (10,000 resamples, seed 42).

## Point predictions

| Model | RMSE | MAE | PHM08 score |
|---|---|---|---|
| LightGBM + window features | 14.59 [12.46, 16.63] | 11.07 [9.21, 12.99] | 314.58 [227.26, 412.80] |
| Linear regression (current sensors) | 21.90 [19.10, 24.64] | 17.61 [15.09, 20.19] | 1313.99 [680.11, 2315.34] |
| Mean RUL baseline | 43.07 [38.49, 47.38] | 35.90 [31.18, 40.58] | 33629.23 [20688.63, 48145.35] |

**RMSE difference, LightGBM minus linear (paired bootstrap):** -7.31 [-10.20, -4.57] cycles. The interval is entirely below 0, so LightGBM is better on these engines.

RMSE and MAE are in cycles. The PHM08 score is a sum over engines (lower is better).

## Prediction intervals (90% target)

Overall: **93/100 engines covered (93.0%, 95% Wilson interval 86.3%–96.6%)**, mean width 48.9 cycles.

| Predicted RUL | Engines | Covered | Coverage [95% Wilson] | Mean width | q (cycles) |
|---|---|---|---|---|---|
| 0-30 | 23 | 19 | 83% [63%, 93%] | 12.7 | 6.38 |
| 30-60 | 12 | 10 | 83% [55%, 95%] | 31.9 | 15.97 |
| 60-90 | 12 | 11 | 92% [65%, 99%] | 57.6 | 28.78 |
| 90+ | 53 | 53 | 100% [93%, 100%] | 66.4 | 33.20 |

**Uncertainty:** with 100 engines, and fewer in each bin, coverage is estimated coarsely. For example, 18 of 20 engines covered (90%) has a 95% Wilson interval of about 70%–97%. Per-bin deviations from 90% of that size are expected noise, not evidence of miscalibration.

Per-engine predictions: `test_predictions_fd001.csv`.
