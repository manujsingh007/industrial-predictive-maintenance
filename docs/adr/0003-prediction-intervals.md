# ADR 0003: Prediction intervals for the primary RUL model

- **Status:** Accepted
- **Date:** 2026-11-17
- **Builds on:** [ADR 0002](0002-model-choice.md) (LightGBM on window features, cap 125) and
  the [evaluation protocol](../evaluation.md)

## Context

A point estimate of RUL is not enough for maintenance planning: a planner needs to know how
far off it might be. We want 90% prediction intervals (alpha = 0.1) for the primary model,
evaluated without the method grading its own homework.

**Evaluation setup:**

- 5-fold engine cross-validation on FD001 training data, seed 42, the same folds for all
  methods;
- each fold's correction q is **calibrated only on the other four folds**, and coverage and
  width are measured on the held-out fold;
- rows with true RUL ≤ 150;
- MLflow experiment `rul-fd001`, commit `657aba9` with no uncommitted changes.

Reproduce with:

```bash
uv run python -m predmaint.training.run_conformal --subset FD001 --method all
```

**Overall** (all rows):

| Method | Coverage | Mean width (cycles) |
|---|---|---|
| Split conformal | 90.0% | 53.8 |
| CQR (LightGBM quantiles 0.05 / 0.95) | 90.0% | 61.1 |
| **Mondrian split conformal** | **89.9%** | **47.9** |

**By predicted-RUL bin** (coverage / mean width). This is the grouping available when an
interval is actually issued:

| Predicted RUL | Split | CQR | Mondrian |
|---|---|---|---|
| 0–30 | 100.0% / 42.9 | 100.0% / 42.7 | 90.0% / **12.1** |
| 30–60 | 99.1% / 56.4 | 99.6% / 66.3 | 89.8% / 31.9 |
| 60–90 | 89.0% / 56.3 | 94.4% / 74.7 | 89.3% / 57.4 |
| 90+ | **82.5%** / 56.4 | **80.2%** / 61.8 | 90.1% / 66.4 |

**By true RUL** (coverage / mean width):

| True RUL | Split | CQR | Mondrian |
|---|---|---|---|
| 0–25 | 100.0% / 41.7 | 100.0% / 41.4 | 94.2% / 12.8 |
| 25–50 | 98.0% / 55.9 | 98.4% / 66.7 | 88.2% / 31.0 |
| 50–75 | 85.8% / 56.4 | 95.1% / 72.9 | 85.8% / 51.6 |
| 75–100 | 88.3% / 56.4 | 95.5% / 67.8 | 91.6% / 62.6 |
| 100–150 | 83.8% / 56.4 | **75.2%** / 59.2 | 89.6% / 65.8 |

**What the tables show:**

- **Split conformal** hits 90% overall by averaging two failures. It uses one width (about
  ±28 cycles) everywhere, so it over-covers near failure (100%, with intervals about 43 cycles
  wide where errors are small) and under-covers predictions of 90+ (82.5%). The near-failure
  intervals are the ones a planner relies on most, and they are needlessly wide.
- **CQR** under-covers far from failure: 80.2% for predictions of 90+ and 75.2% for true RUL
  100–150. The most likely cause is the **RUL cap**. The quantile models are trained on the
  target capped at 125, so the 95% model learns that RUL never exceeds about 125, and its upper
  bound saturates there. For engines whose true RUL is 125–150, the band sits entirely below
  the truth. A single global correction q (about 14 cycles) cannot fix that without inflating
  every interval, so calibration reaches 90% overall by over-covering near failure (100%) and
  missing far from failure. It is also the widest method (61.1).
- **Mondrian** gives 89.3–90.1% in every predicted bin and is the narrowest overall (47.9).
  Near failure, its intervals are 12.1 cycles wide, about 3.5× narrower than either
  alternative. Its per-bin q is stable across folds (0–30: 6.3–6.5; 90+: 32.7–33.5 cycles), and
  no bin needed the global fallback.

## Decision

Use **Mondrian split conformal** for the primary model's intervals:

- the point prediction ± q of its predicted-RUL bin;
- bins **on predicted RUL**, left-closed: [0, 30), [30, 60), [60, 90), [90, ∞);
- alpha = 0.1;
- at least 50 calibration rows per bin, otherwise the global q;
- lower bound clipped at 0.

Bins must use the prediction: the true RUL is unknown when an interval is issued, and the
per-bin guarantee requires that bin membership does not depend on the label.

## Alternatives considered

- **Split conformal (one global q).** Simplest, with correct overall coverage, but too wide
  near failure and under-covering for predictions of 90+. Rejected.
- **CQR with LightGBM quantile models.** In principle the most adaptive, because the band's width
  can vary with each input, not only by bin. In practice here it is the widest method and
  under-covers far from failure, most likely because of the capped training target. It could be
  revisited with quantile models trained on the uncapped target, but that cause has not been
  tested yet.
- **Mondrian with finer or different bins.** Not evaluated. More bins mean fewer calibration
  rows per bin and noisier q. The current four bins all had ample rows.
- **Bins on true RUL.** Not possible: the true RUL is not available when an interval is issued.

## Consequences

- **Coverage is guaranteed per predicted bin, not per engine, and not per true-RUL range.**
  - About 10% of intervals will miss, and misses can cluster within a single engine, because
    an engine's consecutive cycles have correlated errors.
  - By true RUL, Mondrian's coverage still ranges from 85.8% (50–75) to 94.2% (0–25). An engine
    predicted at 100 cycles that is really at 140 is calibrated as a "90+" prediction.
- **Calibration assumes test engines resemble training engines (exchangeability).** If the
  official test set or real engines differ, for example in wear patterns, operating conditions
  or sensor drift, actual coverage can fall below 90% with no warning. The guarantee holds only
  approximately here even within the training data, because rows of one engine are correlated
  and each fold's residuals come from a different model.
- **Calibration for the final model:** q per bin is computed from all out-of-fold residuals of
  the training engines, then applied to the model retrained on all training engines.
- **The official test set** (used once) will report interval coverage and width per predicted
  bin alongside the point metrics, as an unbiased check.
- **Scope: FD001 only.** Changing the bins, alpha or the method requires a new ADR.
