# ADR 0002: Primary RUL model for FD001

- **Status:** Accepted
- **Date:** 2026-11-16
- **Builds on:** [ADR 0001](0001-rul-cap-and-sensor-selection.md) (sensors, RUL cap 125) and the
  frozen [evaluation protocol](../evaluation.md)

## Context

The project needs one primary RUL model to take to the official test set (used once) and to
build on in later milestones, such as prediction intervals. All candidates were evaluated with
the frozen protocol:

- 5-fold engine cross-validation on the FD001 training data, seed 42;
- the **same folds** for every model;
- training target capped at 125;
- scored against the uncapped true RUL on validation rows with RUL ≤ 150.

| Model | Inputs | CV RMSE | CV MAE |
|---|---|---|---|
| Mean RUL baseline | none | 45.11 ± 0.09 | 38.59 ± 0.04 |
| Linear regression | 14 current sensors | 24.34 ± 0.82 | 19.89 ± 0.74 |
| Linear regression | window features (98) | 20.20 ± 0.87 | 16.53 ± 0.70 |
| 1D-CNN | sequences, 30 cycles × 14 sensors | 18.98 ± 1.04 | 14.49 ± 0.76 |
| **LightGBM** | **window features (98)** | **16.51 ± 1.06** | **12.36 ± 0.90** |

Values are in cycles, as mean ± std across the 5 folds. The runs are in MLflow experiment
`rul-fd001`. Window features are the 14 sensors plus a causal rolling mean, std and slope over 10
and 30 cycles.

**Fold-by-fold comparison** (valid because every model used the same folds): LightGBM beats the
1D-CNN in **all 5 folds**, by 1.76 to 3.20 RMSE (mean 2.47). The gap is consistent, not a
product of which engines happen to land in each fold.

## Decision

Use **LightGBM on causal window features** (`--model lgbm --features window`, the defaults in
`predmaint.models.gbdt`, windows 10 and 30, cap 125) as the primary RUL model for FD001.

## Alternatives considered

- **1D-CNN on raw sequences** (`--model cnn --features sequence`). It is second best, at 2.47
  RMSE behind LightGBM on average. Not chosen, for four reasons:
  - it is worse in every fold;
  - it needs the extra `deep` dependency group (PyTorch);
  - it trains more slowly;
  - on Apple MPS its results are not bit-for-bit reproducible.

  It remains in the codebase for future comparison.
- **Linear regression on window features.** Simpler and fully interpretable, but 3.69 RMSE worse
  than LightGBM. It shows the window features themselves carry much of the gain: 24.34 drops to
  20.20 with the same model.
- **XGBoost or another gradient-boosting library.** Not evaluated. With similar defaults, it would
  be expected to perform close to LightGBM.
- **Tune hyperparameters before choosing.** Deferred; see Consequences.

## Consequences

- **The comparison uses untuned defaults for both LightGBM and the CNN.** Tuning either could
  change the gap. A tuned model replaces LightGBM only through a new ADR, and only if it wins
  consistently across folds, not just on the mean.
- **The dataset is small.** It has 100 training engines, about 80 per CV training fold. Deep
  models usually benefit more from larger datasets, so this result says little about the CNN on
  bigger data or on FD002–FD004.
- **CV scores are slightly optimistic for the winner.** LightGBM was chosen by the same CV that
  reports its score. The official test set, used once, gives the unbiased estimate.
- **Advantages of the choice:**
  - training is fast and deterministic on CPU;
  - no deep-learning dependency is needed for the primary pipeline;
  - the input features have readable names, which makes it possible to inspect what the model
    uses.
- **Scope: FD001 only,** like ADR 0001. Other subsets need their own evaluation.
- **Reproducibility note:** the two baseline runs were logged from a working tree with
  uncommitted changes (`git_dirty=true`). Their numbers were re-run on 2026-11-16 with the
  current code and reproduced exactly (45.11 and 24.34 RMSE).
