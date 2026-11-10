"""Checks for Phase 4: the raw and summary-statistic feature representations."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data_loader as dl  # noqa: E402
from src import feature_engineering as fe  # noqa: E402
from src import preprocessing as pp  # noqa: E402


@pytest.fixture(scope="module")
def prepared():
    return {grid: pp.preprocess(*grid) for grid in pp.EXPERIMENT_GRID}


@pytest.fixture(scope="module")
def summary(prepared):
    return {
        grid: {split: fe.make_features(p[split], p["sensors"], "summary") for split in ("train", "test")}
        for grid, p in prepared.items()
    }


def test_feature_counts():
    expected = {("FD001", "global"): 15, ("FD002", "global"): 21, ("FD002", "condition"): 17}
    for grid, n_sensors in expected.items():
        sensors = pp.preprocess(*grid)["sensors"]
        assert len(fe.feature_names(sensors, "raw")) == n_sensors
        assert len(fe.feature_names(sensors, "summary")) == 3 * n_sensors


def test_raw_representation_is_the_scaled_sensors(prepared):
    for p in prepared.values():
        raw = fe.make_features(p["train"], p["sensors"], "raw")
        pd.testing.assert_frame_equal(raw[p["sensors"]], p["train"][p["sensors"]])
        assert list(raw.columns) == pp.ID_COLS + p["sensors"] + pp.TARGET_COLS


def test_rows_identifiers_and_targets_are_unchanged(prepared, summary):
    for grid, p in prepared.items():
        for split in ("train", "test"):
            out = summary[grid][split]
            for col in pp.ID_COLS + pp.TARGET_COLS:
                assert out[col].tolist() == p[split][col].tolist()
            names = fe.feature_names(p["sensors"], "summary")
            assert list(out.columns) == pp.ID_COLS + names + pp.TARGET_COLS
            assert np.isfinite(out[names].to_numpy()).all()


def test_last_value_equals_the_raw_reading(prepared, summary):
    for grid, p in prepared.items():
        for s in p["sensors"]:
            assert np.array_equal(summary[grid]["train"][f"{s}_last"], p["train"][s])


def test_window_statistics_match_a_direct_calculation(prepared, summary):
    """Recompute mean and std by hand for several engines and cycles."""
    for grid, p in prepared.items():
        table, feats = p["train"], summary[grid]["train"]
        sensor = "s11"
        for unit in (1, 7, table["unit"].max()):
            values = table.loc[table["unit"] == unit, sensor].to_numpy()
            rows = feats[feats["unit"] == unit].reset_index(drop=True)
            for cycle in (1, 2, 5, 29, 30, 31, 100, len(values)):
                window = values[max(0, cycle - fe.WINDOW):cycle]  # the last 30 readings up to this cycle
                assert rows.loc[cycle - 1, f"{sensor}_mean"] == pytest.approx(window.mean(), abs=1e-8)
                assert rows.loc[cycle - 1, f"{sensor}_std"] == pytest.approx(window.std(), abs=1e-8)


def test_windows_do_not_cross_engine_boundaries(summary):
    for tables in summary.values():
        first = tables["train"].groupby("unit").head(1)
        for col in [c for c in first.columns if c.endswith("_mean")]:
            assert np.allclose(first[col], first[col.replace("_mean", "_last")])
        std_cols = [c for c in first.columns if c.endswith("_std")]
        assert np.allclose(first[std_cols], 0)


def test_features_use_past_cycles_only(prepared):
    """Cutting off an engine's future must not change its earlier features."""
    p = prepared[("FD001", "global")]
    engine = p["train"][p["train"]["unit"] == 3]
    full = fe.make_features(engine, p["sensors"], "summary")
    truncated = fe.make_features(engine.iloc[:80], p["sensors"], "summary")
    names = fe.feature_names(p["sensors"], "summary")
    assert np.allclose(full[names].iloc[:80], truncated[names], atol=1e-10)


def test_std_is_never_negative_and_zero_for_constant_windows():
    table = pd.DataFrame({"unit": [1] * 40, "cycle": range(1, 41), "s2": [0.37] * 40})
    out = fe.make_features(table, ["s2"], "summary")
    assert (out["s2_std"] >= 0).all()
    assert np.allclose(out["s2_std"], 0, atol=1e-9)
    assert np.allclose(out["s2_mean"], 0.37)


def test_unsorted_rows_are_rejected():
    table = pd.DataFrame({"unit": [1, 1, 1], "cycle": [1, 3, 2], "s2": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError):
        fe.make_features(table, ["s2"], "summary")


def test_last_cycle_rows_give_one_row_per_test_engine_with_nasa_rul(summary):
    for (subset, _), tables in summary.items():
        last = fe.last_cycle_rows(tables["test"])
        assert len(last) == dl.EXPECTED_UNITS[subset]["test"]
        assert last["cycle"].tolist() == tables["test"].groupby("unit")["cycle"].max().tolist()
        assert last["rul_uncapped"].tolist() == dl.load_rul(subset)["rul"].tolist()


def test_engines_shorter_than_the_window_still_get_features(summary):
    test = summary[("FD002", "condition")]["test"]
    lengths = test.groupby("unit")["cycle"].max()
    assert lengths.min() < fe.WINDOW  # FD002 has test engines with fewer than 30 cycles
    last = fe.last_cycle_rows(test)
    short = last[last["cycle"] < fe.WINDOW]
    feature_cols = [c for c in short.columns if c.endswith(("_last", "_mean", "_std"))]
    assert len(short) > 0 and np.isfinite(short[feature_cols].to_numpy()).all()
