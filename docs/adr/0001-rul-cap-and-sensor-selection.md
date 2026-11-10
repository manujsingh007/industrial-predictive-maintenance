# ADR 0001: RUL cap and sensor selection for FD001

- **Status:** Accepted
- **Date:** 2026-11-10
- **Evidence:** [`notebooks/01_eda_fd001.ipynb`](../../notebooks/01_eda_fd001.ipynb), FD001 training data only

## Context

RUL models need two decisions before training: what target to learn, and which inputs to use.

**The target.** Raw training RUL ranges from 0 up to 361 cycles. The EDA (section 5) shows that
the averaged sensor signal has moved only about 10% of its total change by RUL 130, and about 90%
of the change happens closer to failure. Far from failure, RUL 300 and RUL 200 look almost the
same in the sensors, so predicting the exact value there mostly fits noise. Every training engine
lives at least 128 cycles.

**The inputs.** Of the 21 sensors, the EDA (section 2) found:

- 6 with a single constant value;
- 1 (`s_6`) with only two values, one rounding step apart, the rarer value in 2% of rows.

The three operating settings show one operating condition and correlate with RUL at only
|r| ≤ 0.003 (section 6).

## Decision

1. **Cap the training target:** `rul_target = min(rul, 125)`.
   - 125 is the default, tuned over **110–150** with the cross-validation protocol in
     [evaluation.md](../evaluation.md).
   - Evaluation always compares against the **uncapped** true RUL.
2. **Drop 7 sensors and the 3 settings:**
   - constant: `s_1`, `s_5`, `s_10`, `s_16`, `s_18`, `s_19`;
   - near-zero variance: `s_6`;
   - settings: `setting_1`–`setting_3`.
3. **Keep 14 sensors:** `s_2`, `s_3`, `s_4`, `s_7`, `s_8`, `s_9`, `s_11`, `s_12`, `s_13`,
   `s_14`, `s_15`, `s_17`, `s_20`, `s_21`.
   - `s_9` and `s_14` are kept but **flagged**: only 71% and 58% of engines follow the fleet
     direction (section 3).
   - Dropping them is an experiment to evaluate, not a decision taken here.

## Alternatives considered

- **No cap (target goes linearly down to 0 from the first cycle).** Rejected: it forces the model to
  predict differences far from failure that the sensors barely show.
- **A cap of 169 or 89.** These are the onsets at the 5% and 20% thresholds (section 5). Both are
  defensible, so the range 110–150 is tuned rather than fixed at one value.
- **Drop sensors by std ≈ 0 or by coefficient of variation.** Rejected:
  - `std == 0` misses `s_5` and `s_16`, whose std shows floating-point noise of about 1e-15;
  - a CV rule would wrongly drop `s_8` and `s_13`, which have |r| = 0.56 with RUL.

  Counting distinct values is used instead.
- **Also drop `s_9` and `s_14` now.** Deferred: they change strongly for some engines, so
  validation results should decide.
- **Select features automatically inside each CV fold.** More rigorous, but unnecessary for
  dropping constant columns. It may be revisited for the flagged sensors.

## Consequences

- **No prediction ever exceeds the cap.** Test engines whose true RUL is above the cap carry a
  guaranteed error. FD001 has test RULs up to 145 ([dataset.md](../dataset.md)), so the cap
  directly trades error on those engines against noise far from failure. Evaluating against
  uncapped RUL makes that trade visible.
- **This ADR covers only FD001.** FD002–FD004 have six operating conditions, where settings
  matter and sensors may not be constant. They need their own EDA and ADR.
- **Caveat: possible leakage.** The cap and the drop list were chosen by looking at **all 100
  training engines**, including the engines that later serve as validation folds. Cross-validation
  scores for these choices may therefore be slightly optimistic. The risk is small for dropping
  constant columns and larger for the cap value. The official test set was never looked at, so
  the final test result is unaffected.
- **Changing these decisions needs a new ADR.** That applies to moving the cap outside 110–150 or
  changing the drop list.
