"""Checks for the Phase 2 exploration tables."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import exploration as ex  # noqa: E402


def test_no_missing_values_or_duplicates_in_any_file():
    report = ex.quality_report()
    assert len(report) == 4
    checks = ["missing_values", "non_finite_values", "duplicate_rows", "duplicate_unit_cycle"]
    assert (report[checks] == 0).all().all()


def test_condition_key_ignores_setting_noise_and_sign_of_zero():
    df = pd.DataFrame(
        {
            "setting_1": [-0.0007, 0.0019, 41.9987, 42.0033],
            "setting_2": [-0.0004, 0.0003, 0.8401, 0.8399],
            "setting_3": [100.0, 100.0, 100.0, 100.0],
        }
    )
    keys = ex.condition_key(df)
    assert keys[0] == keys[1]
    assert keys[2] == keys[3]
    assert keys[0] != keys[2]


def test_number_of_operating_conditions_matches_readme():
    assert len(ex.operating_conditions("FD001")) == 1
    fd002 = ex.operating_conditions("FD002")
    assert len(fd002) == 6
    assert fd002["rows"].sum() == 53759


def test_constant_columns_fd001():
    summary = ex.sensor_summary("FD001")
    constant = set(summary.loc[summary["constant"], "column"])
    assert constant == {"setting_3", "s1", "s5", "s10", "s16", "s18", "s19"}


def test_fd002_has_no_globally_constant_column_but_some_constant_within_condition():
    summary = ex.sensor_summary("FD002")
    assert not summary["constant"].any()
    within = set(summary.loc[summary["constant_within_condition"], "column"])
    assert within == {"setting_3", "s1", "s5", "s18", "s19"}


def test_correlations_are_valid():
    for subset in ("FD001", "FD002"):
        summary = ex.sensor_summary(subset)
        for col in ("spearman_vs_remaining", "spearman_within_condition"):
            values = summary[col].dropna()
            assert values.between(-1, 1).all()
        share = summary["variance_from_condition"].dropna()
        assert share.between(0, 1).all()
