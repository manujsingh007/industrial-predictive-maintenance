# Evaluation protocol (frozen)

This protocol is fixed **before** any model is trained, so results cannot be tuned to it. Every
model in this project is evaluated this way. Changing it requires an ADR in [`adr/`](adr/), and
results from different protocol versions must not be compared.

Scope: FD001 first. Each subset is evaluated separately with the same protocol.

## 1. Data splits

**Cross-validation (all model development).**

- Training engines are split **by engine**, never by row:
  `GroupKFold(n_splits=5, shuffle=True, random_state=42)` with `groups=unit`. The `shuffle` and
  `random_state` arguments need scikit-learn ≥ 1.6.
- All rows of an engine stay in the same fold. Rows of one engine are strongly correlated, so a
  row-level split would leak the engine's own history into validation.
- Everything learned from data, such as scalers and feature statistics, is fitted on the training
  folds only.
- The RUL cap is a hyperparameter chosen by mean CV score. Its starting value came from EDA on all
  training engines; see the leakage caveat in ADR 0001.
- **Validation rows:** all cycles of the held-out engines with true RUL ≤ 150. This roughly
  matches the FD001 test RUL range of 7–145 ([dataset.md](dataset.md)), and every training engine
  contributes.
- **Reported:** the mean and standard deviation over the 5 folds.

**Official test set (once).**

- Used **once**, at the end, for the final chosen model, after retraining on all training engines.
- Nothing is tuned on it, and the result is reported whatever it is.

## 2. Targets

- **Training target:** RUL capped at the value from
  [ADR 0001](adr/0001-rul-cap-and-sensor-selection.md): 125 by default, tuned over 110–150 by
  CV mean RMSE.
- **Ground truth for every metric:** the **uncapped** true RUL. Comparing against capped values
  would make lower caps look better automatically.

## 3. Metrics

On the test set, each engine is evaluated **at its last observed cycle**, against the true RUL
from `RUL_FD00x.txt`, so there is one prediction per engine. With *d* = predicted RUL − true RUL:

| Metric | Definition | Notes |
|---|---|---|
| RMSE | √(mean *d*²) | Primary metric; used for model selection |
| MAE | mean \|*d*\| | Easier to interpret, in cycles |
| PHM08 score | Σ (e^(−*d*/13) − 1) for *d* < 0; Σ (e^(*d*/10) − 1) for *d* ≥ 0 | Penalises late predictions more ([dataset.md §6](dataset.md#6-how-the-test-set-works)). It is a **sum**, so only compare it on the same set of engines |

## 4. Uncertainty

- **Test set:** 95% **bootstrap** confidence intervals for every metric. Resample the test
  **engines** with replacement, 10,000 times, `seed=42`, and take percentile intervals. Each
  resample has the same number of engines as the test set.
- **Two models:** bootstrap the **difference** in metrics on the same resampled engines.
  Overlapping individual intervals is not a valid test.
- FD001 has only 100 test engines, so expect wide intervals. Differences inside them are not
  evidence that one model is better.

## 5. Baselines

Every model must beat both baselines on CV RMSE before its test result means anything.
Both use the same capped target and the 14 kept sensors from ADR 0001.

1. **Mean RUL:** always predict the mean capped RUL of the training rows.
2. **Linear regression:** ordinary least squares on the standardised current-cycle values of the
   14 sensors, with no history or window features.

Baselines are reported with the same metrics, folds and intervals as every other model.
