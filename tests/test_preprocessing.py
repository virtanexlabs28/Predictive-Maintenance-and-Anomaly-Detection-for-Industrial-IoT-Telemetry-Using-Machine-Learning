"""Checks for Phase 3: RUL labels, sensor removal, scaling and leakage."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data_loader as dl  # noqa: E402
from src import preprocessing as pp  # noqa: E402


@pytest.fixture(scope="module")
def results():
    return {(s, sc): pp.preprocess(s, sc) for s in dl.SUBSETS for sc in pp.SCHEMES}


# ---------------------------------------------------------------- RUL labels

@pytest.mark.parametrize("subset", dl.SUBSETS)
def test_train_rul_counts_down_to_zero_and_is_capped(subset):
    train = pp.add_train_rul(dl.load_train(subset))
    for _, engine in train.groupby("unit"):
        n = len(engine)
        assert engine["rul_uncapped"].tolist() == list(range(n - 1, -1, -1))
    assert (train["rul"] == train["rul_uncapped"].clip(upper=125)).all()
    assert train["rul"].max() == 125 and train["rul"].min() == 0


@pytest.mark.parametrize("subset", dl.SUBSETS)
def test_test_rul_matches_nasa_file_at_last_cycle(subset):
    data = dl.load_subset(subset)
    test = pp.add_test_rul(data["test"], data["rul"])
    last = test.groupby("unit").tail(1).set_index("unit")["rul_uncapped"]
    assert last.tolist() == data["rul"]["rul"].tolist()
    # one cycle earlier the engine has exactly one more cycle left
    for _, engine in test.groupby("unit"):
        assert (engine["rul_uncapped"].diff().dropna() == -1).all()


def test_custom_cap_is_respected():
    train = pp.add_train_rul(dl.load_train("FD001"), cap=100)
    assert train["rul"].max() == 100


# ------------------------------------------------------------ sensor removal

def test_constant_sensors_are_found_from_training_data():
    fd001, fd002 = dl.load_train("FD001"), dl.load_train("FD002")
    six = ["s1", "s5", "s10", "s16", "s18", "s19"]
    assert pp.find_constant_sensors(fd001, "global") == six
    assert pp.find_constant_sensors(fd001, "condition") == six
    assert pp.find_constant_sensors(fd002, "global") == []
    assert pp.find_constant_sensors(fd002, "condition") == ["s1", "s5", "s18", "s19"]


# -------------------------------------------------------------------- scaling

def test_structure_is_preserved(results):
    for (subset, _), res in results.items():
        raw = dl.load_subset(subset)
        for split in ("train", "test"):
            out = res[split]
            assert len(out) == len(raw[split])
            assert out["unit"].tolist() == raw[split]["unit"].tolist()
            assert out["cycle"].tolist() == raw[split]["cycle"].tolist()
            assert list(out.columns) == pp.ID_COLS + res["sensors"] + pp.TARGET_COLS
            assert not out.isna().any().any()
            assert np.isfinite(out[res["sensors"]].to_numpy()).all()


def test_scaler_statistics_come_from_training_rows_only(results):
    for (subset, scheme), res in results.items():
        raw_train = dl.load_train(subset)
        key = pp._group_key(raw_train, scheme)
        expected_mean = raw_train.groupby(key)[res["sensors"]].mean()
        pd.testing.assert_frame_equal(res["scaler"].mean_, expected_mean)


def test_global_scaling_gives_zero_mean_unit_variance_on_train(results):
    for subset in dl.SUBSETS:
        res = results[(subset, "global")]
        x = res["train"][res["sensors"]]
        assert np.allclose(x.mean(), 0, atol=1e-8)
        assert np.allclose(x.std(ddof=0), 1, atol=1e-8)


def test_condition_scaling_centres_every_condition_on_train(results):
    res = results[("FD002", "condition")]
    train = res["train"]
    grouped = train.groupby("condition")[res["sensors"]]
    assert np.allclose(grouped.mean(), 0, atol=1e-8)
    std = grouped.std(ddof=0).to_numpy()
    # each (condition, sensor) pair is either standardised or constant (all zeros)
    assert np.all(np.isclose(std, 1, atol=1e-8) | np.isclose(std, 0, atol=1e-12))


def test_fd001_schemes_are_identical(results):
    a, b = results[("FD001", "global")], results[("FD001", "condition")]
    assert a["sensors"] == b["sensors"]
    for split in ("train", "test"):
        assert np.allclose(a[split][a["sensors"]], b[split][b["sensors"]])


def test_test_engines_are_scaled_with_training_statistics(results):
    """Recompute one test value by hand from training statistics."""
    res = results[("FD002", "condition")]
    raw_train, raw_test = dl.load_train("FD002"), dl.load_test("FD002")
    row = raw_test.iloc[1234]
    cond = pp.condition_key(raw_test.iloc[[1234]]).iloc[0]
    same = raw_train[pp.condition_key(raw_train) == cond]["s11"]
    expected = (row["s11"] - same.mean()) / same.std(ddof=0)
    assert res["test"].loc[1234, "s11"] == pytest.approx(expected, rel=1e-9)


def test_zero_spread_group_becomes_zero_not_infinity():
    train = pd.DataFrame(
        {
            "setting_1": [0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
            "setting_2": [0.0, 0.0, 0.0, 0.25, 0.25, 0.25],
            "setting_3": [100.0] * 6,
            "s1": [14.62, 14.62, 14.62, 1.0, 2.0, 3.0],  # constant inside the first condition
        }
    )
    scaler = pp.SensorScaler("condition").fit(train, ["s1"])
    out = scaler.transform(train)
    assert out.loc[:2, "s1"].tolist() == [0.0, 0.0, 0.0]
    assert np.isfinite(out["s1"]).all()
    assert out.loc[3:, "s1"].mean() == pytest.approx(0)


def test_unseen_operating_condition_is_rejected():
    train = dl.load_train("FD001")
    scaler = pp.SensorScaler("condition").fit(train, ["s2"])
    with pytest.raises(ValueError):
        scaler.transform(dl.load_train("FD002").head(50))
