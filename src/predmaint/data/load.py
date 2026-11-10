"""Load the raw C-MAPSS files into pandas DataFrames.

File format (see docs/dataset.md): no header, values separated by whitespace, 26 columns per row:
unit, cycle, 3 operational settings and 21 sensor measurements.
"""

from pathlib import Path

import pandas as pd

from predmaint.data.download import DEFAULT_RAW_DIR, SUBSETS

SETTING_COLUMNS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLUMNS = [f"s_{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTING_COLUMNS, *SENSOR_COLUMNS]


def _validate_subset(subset: str) -> None:
    """Raise ValueError unless ``subset`` is one of FD001..FD004."""
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}; expected one of {', '.join(SUBSETS)}")


def _data_file(data_dir: Path, name: str) -> Path:
    """Return the path to a raw data file, with a helpful error if it is missing."""
    path = Path(data_dir) / name
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Download the data with: uv run python -m predmaint.data.download"
        )
    return path


def _read_trajectories(path: Path) -> pd.DataFrame:
    """Read a train/test file and name its 26 columns."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"{path} has {df.shape[1]} columns, expected {len(COLUMNS)}")
    df.columns = COLUMNS
    return df.astype({"unit": "int64", "cycle": "int64"})


def load_train(subset: str, data_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Load the training trajectories (run to failure) of one subset.

    Args:
        subset: One of "FD001", "FD002", "FD003", "FD004".
        data_dir: Directory containing the raw C-MAPSS files.

    Returns:
        One row per (unit, cycle) with columns ``unit``, ``cycle``,
        ``setting_1``..``setting_3`` and ``s_1``..``s_21``.
    """
    _validate_subset(subset)
    return _read_trajectories(_data_file(data_dir, f"train_{subset}.txt"))


def load_test(subset: str, data_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Load the test trajectories (truncated before failure) of one subset.

    Args:
        subset: One of "FD001", "FD002", "FD003", "FD004".
        data_dir: Directory containing the raw C-MAPSS files.

    Returns:
        Same columns as :func:`load_train`.
    """
    _validate_subset(subset)
    return _read_trajectories(_data_file(data_dir, f"test_{subset}.txt"))


def add_rul(train_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``train_df`` with a ``rul`` column (remaining useful life in cycles).

    Training units run until failure, so RUL at a row is the unit's last cycle minus the
    current cycle: 0 at the failure cycle. The value is not capped; capping is a modelling
    decision made later.
    """
    last_cycle = train_df.groupby("unit")["cycle"].transform("max")
    return train_df.assign(rul=last_cycle - train_df["cycle"])


def load_test_rul(subset: str, data_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Load the true RUL of each test unit at its last observed cycle.

    ``RUL_FD00x.txt`` has one value per line, where line *i* belongs to test unit *i*.

    Args:
        subset: One of "FD001", "FD002", "FD003", "FD004".
        data_dir: Directory containing the raw C-MAPSS files.

    Returns:
        DataFrame with columns ``unit`` (1..n) and ``rul``.
    """
    _validate_subset(subset)
    path = _data_file(data_dir, f"RUL_{subset}.txt")
    rul = pd.read_csv(path, sep=r"\s+", header=None)
    if rul.shape[1] != 1:
        raise ValueError(f"{path} has {rul.shape[1]} columns, expected 1")
    values = rul[0].astype("int64")
    return pd.DataFrame({"unit": range(1, len(values) + 1), "rul": values})
