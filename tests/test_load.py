"""Tests for predmaint.data.load using tiny fake files (no real data)."""

from pathlib import Path

import pandas as pd
import pytest

from predmaint.data.load import (
    COLUMNS,
    add_rul,
    load_test,
    load_test_rul,
    load_train,
)


def write_trajectories(path: Path, cycles_per_unit: dict[int, int], n_cols: int = 26) -> None:
    """Write a fake file in the real C-MAPSS format: space-separated, two trailing spaces."""
    lines = []
    for unit, n_cycles in cycles_per_unit.items():
        for cycle in range(1, n_cycles + 1):
            values = [str(unit), str(cycle)] + [f"{0.5 * i:.2f}" for i in range(n_cols - 2)]
            lines.append(" ".join(values) + "  ")
    path.write_text("\n".join(lines) + "\n")


def write_rul(path: Path, values: list[int]) -> None:
    path.write_text("".join(f"{v} \n" for v in values))


# --- column names and parsing ------------------------------------------------


def test_load_train_column_names(tmp_path: Path) -> None:
    write_trajectories(tmp_path / "train_FD001.txt", {1: 3, 2: 2})
    df = load_train("FD001", data_dir=tmp_path)
    assert list(df.columns) == COLUMNS
    assert len(COLUMNS) == 26
    assert COLUMNS[:5] == ["unit", "cycle", "setting_1", "setting_2", "setting_3"]
    assert COLUMNS[5] == "s_1" and COLUMNS[-1] == "s_21"


def test_trailing_spaces_do_not_create_extra_columns(tmp_path: Path) -> None:
    write_trajectories(tmp_path / "test_FD002.txt", {1: 2})
    df = load_test("FD002", data_dir=tmp_path)
    assert df.shape == (2, 26)
    assert not df.isna().any().any()


def test_unit_and_cycle_are_integers(tmp_path: Path) -> None:
    write_trajectories(tmp_path / "train_FD001.txt", {1: 2})
    df = load_train("FD001", data_dir=tmp_path)
    assert df["unit"].dtype == "int64"
    assert df["cycle"].dtype == "int64"


def test_wrong_column_count_raises(tmp_path: Path) -> None:
    write_trajectories(tmp_path / "train_FD001.txt", {1: 2}, n_cols=27)
    with pytest.raises(ValueError, match="27 columns"):
        load_train("FD001", data_dir=tmp_path)


@pytest.mark.parametrize("bad", ["FD005", "fd001", "FD01", ""])
def test_invalid_subset_raises(tmp_path: Path, bad: str) -> None:
    for loader in (load_train, load_test, load_test_rul):
        with pytest.raises(ValueError, match="Unknown subset"):
            loader(bad, data_dir=tmp_path)


def test_missing_file_points_to_download(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="predmaint.data.download"):
        load_train("FD003", data_dir=tmp_path)


# --- RUL calculation ------------------------------------------------------------


def test_add_rul_counts_down_to_zero_per_unit() -> None:
    df = pd.DataFrame({"unit": [1, 1, 1, 2, 2], "cycle": [1, 2, 3, 1, 2]})
    assert add_rul(df)["rul"].tolist() == [2, 1, 0, 1, 0]


def test_add_rul_does_not_depend_on_row_order() -> None:
    df = pd.DataFrame({"unit": [2, 1, 1, 2, 1], "cycle": [2, 3, 1, 1, 2]})
    assert add_rul(df)["rul"].tolist() == [0, 0, 2, 1, 1]


def test_add_rul_is_not_capped_and_does_not_mutate_input() -> None:
    df = pd.DataFrame({"unit": [1] * 300, "cycle": range(1, 301)})
    out = add_rul(df)
    assert out["rul"].max() == 299
    assert "rul" not in df.columns


def test_add_rul_on_loaded_file(tmp_path: Path) -> None:
    write_trajectories(tmp_path / "train_FD001.txt", {1: 4, 2: 2})
    df = add_rul(load_train("FD001", data_dir=tmp_path))
    assert df.loc[df["unit"] == 1, "rul"].tolist() == [3, 2, 1, 0]
    assert df.loc[df["unit"] == 2, "rul"].tolist() == [1, 0]


# --- test RUL alignment -----------------------------------------------------------


def test_load_test_rul_columns_and_values(tmp_path: Path) -> None:
    write_rul(tmp_path / "RUL_FD001.txt", [112, 98, 69])
    rul = load_test_rul("FD001", data_dir=tmp_path)
    assert list(rul.columns) == ["unit", "rul"]
    assert rul["unit"].tolist() == [1, 2, 3]
    assert rul["rul"].tolist() == [112, 98, 69]


def test_test_rul_aligns_with_test_units(tmp_path: Path) -> None:
    write_trajectories(tmp_path / "test_FD001.txt", {1: 5, 2: 3, 3: 4})
    write_rul(tmp_path / "RUL_FD001.txt", [10, 20, 30])
    test = load_test("FD001", data_dir=tmp_path)
    rul = load_test_rul("FD001", data_dir=tmp_path)

    last = test.groupby("unit", as_index=False)["cycle"].max()
    merged = last.merge(rul, on="unit", how="outer", validate="one_to_one", indicator=True)
    assert (merged["_merge"] == "both").all()
    assert dict(zip(merged["unit"], merged["rul"], strict=True)) == {1: 10, 2: 20, 3: 30}
