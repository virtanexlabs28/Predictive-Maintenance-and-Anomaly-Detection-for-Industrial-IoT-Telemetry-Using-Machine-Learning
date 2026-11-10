"""Checks that the raw C-MAPSS files are intact and loaded correctly.

Run from the project root with:  python -m pytest tests -q
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data_loader as dl  # noqa: E402

SUBSETS = list(dl.SUBSETS)


def test_raw_files_match_recorded_checksums():
    results = dl.verify_checksums()
    assert results, "checksum list is empty"
    assert all(results.values()), f"changed or missing files: {[k for k, v in results.items() if not v]}"


@pytest.mark.parametrize("subset", SUBSETS)
@pytest.mark.parametrize("split", ["train", "test"])
def test_columns_and_types(subset, split):
    df = dl.load_subset(subset)[split]
    assert list(df.columns) == dl.COLUMN_NAMES
    assert len(dl.COLUMN_NAMES) == 26 and len(dl.SENSOR_COLS) == 21
    assert df["unit"].dtype.kind == "i" and df["cycle"].dtype.kind == "i"
    # every settings/sensor column must be numeric
    assert all(df[c].dtype.kind in "if" for c in dl.SETTING_COLS + dl.SENSOR_COLS)


@pytest.mark.parametrize("subset", SUBSETS)
@pytest.mark.parametrize("split", ["train", "test"])
def test_engine_counts_match_readme(subset, split):
    df = dl.load_subset(subset)[split]
    n = dl.EXPECTED_UNITS[subset][split]
    assert sorted(df["unit"].unique()) == list(range(1, n + 1))


@pytest.mark.parametrize("subset", SUBSETS)
@pytest.mark.parametrize("split", ["train", "test"])
def test_cycles_are_consecutive_from_one(subset, split):
    """Each engine's cycles must be 1, 2, 3, ... with no gaps or repeats,
    stored in order. RUL labels in Phase 3 rely on this."""
    df = dl.load_subset(subset)[split]
    for unit, cycles in df.groupby("unit")["cycle"]:
        assert cycles.tolist() == list(range(1, len(cycles) + 1)), f"unit {unit}"


@pytest.mark.parametrize("subset", SUBSETS)
def test_rul_file_has_one_positive_value_per_test_engine(subset):
    data = dl.load_subset(subset)
    rul = data["rul"]
    assert len(rul) == data["test"]["unit"].nunique()
    assert rul["unit"].tolist() == list(range(1, len(rul) + 1))
    assert rul["rul"].dtype.kind == "i" and (rul["rul"] > 0).all()


def test_unknown_subset_is_rejected():
    with pytest.raises(ValueError):
        dl.load_train("FD003")
