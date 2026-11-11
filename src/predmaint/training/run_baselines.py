"""Cross-validate the baseline models on a training subset (test files are never read).

Run with: ``uv run python -m predmaint.training.run_baselines --subset FD001``
"""

import argparse

from predmaint.data.load import add_rul, load_train
from predmaint.features.selection import DEFAULT_RUL_CAP, KEPT_SENSORS, SUBSETS_WITH_ADR
from predmaint.models.baselines import make_linear_baseline, make_mean_baseline
from predmaint.training.cv import MAX_EVAL_RUL, cross_validate

BASELINES = {
    "mean RUL": make_mean_baseline,
    "linear regression": make_linear_baseline,
}
N_SPLITS = 5
SEED = 42


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Cross-validate the RUL baselines.")
    parser.add_argument(
        "--subset",
        required=True,
        choices=SUBSETS_WITH_ADR,
        help="training subset (only subsets covered by an ADR)",
    )
    parser.add_argument("--cap", type=float, default=DEFAULT_RUL_CAP, help="training RUL cap")
    args = parser.parse_args()

    train = add_rul(load_train(args.subset))

    print(
        f"{args.subset} baselines: {N_SPLITS}-fold engine CV (seed {SEED}), "
        f"{train['unit'].nunique()} training engines, {len(KEPT_SENSORS)} sensors\n"
        f"Target capped at {args.cap:g}; scored on validation rows with uncapped true RUL "
        f"<= {MAX_EVAL_RUL}\n"
    )
    header = f"{'model':<20} {'RMSE (mean ± std)':>18} {'MAE (mean ± std)':>18}"
    print(header)
    print("-" * len(header))
    for name, factory in BASELINES.items():
        result = cross_validate(factory, train, cap=args.cap, n_splits=N_SPLITS, seed=SEED)
        s = result.summary()
        rmse_text = f"{s.loc['mean', 'rmse']:.2f} ± {s.loc['std', 'rmse']:.2f}"
        mae_text = f"{s.loc['mean', 'mae']:.2f} ± {s.loc['std', 'mae']:.2f}"
        print(f"{name:<20} {rmse_text:>18} {mae_text:>18}")
    print("\nstd = sample standard deviation across folds; units are cycles.")


if __name__ == "__main__":
    main()
