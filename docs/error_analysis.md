# Error analysis: LightGBM on FD001

Full analysis: [notebooks/02_error_analysis_fd001.ipynb](../notebooks/02_error_analysis_fd001.ipynb).

**Data-use rule:**

- Findings that could motivate model changes come only from the 5-fold out-of-fold CV
  predictions on the training engines.
- The official test predictions are only described, because the test set has already been used
  once.

**Sign convention:** error = predicted − true. A positive error is **late**, the dangerous
direction.

## Findings (cross-validation)

- **Late errors concentrate in mid-life.** At true RUL 50–100 the bias is about +9 cycles, and
  26–27% of rows are late by more than 20 cycles. Near failure the model is accurate (RUL 0–25:
  RMSE 3.9).
- **Short-lived engines are predicted late.** For the same true RUL, an engine seen for fewer
  cycles gets later predictions: at RUL 50–100, bias +18 to +20 with 30–100 cycles of history,
  vs −3 after more than 150. Given RUL, a short history means a fast-degrading engine, and the
  two cannot be separated in these data.
- **The worst engines are mostly long-lived and predicted early.** Four of the five worst
  (lifetimes at the 57th to 99th percentile) are early both below the cap (−11 to −18) and above
  it (−40 to −46).
  - Two of them start life with sensor levels shifted towards degradation, consistent with
    higher initial wear.
  - The other two are not explained by the four key sensors.
  - Per-engine RMSE ranges up to 25.6 (median 15.6), so no engine is an extreme outlier.
- **The RUL cap causes a structural, early bias.** No prediction exceeds 133.6 cycles. Rows
  with true RUL 126–150 are 16.1% of scored rows but 37.9% of the squared error. Against the
  capped target, the model's error there is only 13.6.

## Test set (descriptive only)

Of the six largest test errors:

- **Two are late:** short histories (71–74 cycles) at true RUL of about 78. That matches the CV
  mid-life pattern.
- **Two are early, from the cap or from long-lived engines:** true RUL 145, and true RUL 114
  after 152 cycles.
- **Two are early and unexplained** by the CV patterns.
- **Three of the six fall outside their 90% interval,** all on the early (safe) side.

## Future work (not applied)

Each item needs its own CV experiment and ADR:

- an asymmetric loss that penalises late errors;
- features that track each engine's degradation rate;
- normalising sensors by each engine's early-life baseline;
- reporting "RUL > 125" as a category rather than a number.
