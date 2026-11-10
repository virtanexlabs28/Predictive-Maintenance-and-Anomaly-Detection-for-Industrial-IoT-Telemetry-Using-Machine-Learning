"""Checks for Phase 5: metrics, engine-level folds, leakage and reproducibility.

These tests use very small models so they run in seconds. They check the
procedure, not the reported accuracy.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data_loader as dl  # noqa: E402
from src import preprocessing as pp  # noqa: E402
from src import rul_prediction as rp  # noqa: E402
from src.evaluation import regression_metrics  # noqa: E402


@pytest.fixture(scope="module")
def folds():
    return rp.make_folds("FD002", "condition", "summary")


def test_regression_metrics_on_known_values():
    scores = regression_metrics([0, 10], [0, 20])
    assert scores["mae"] == pytest.approx(5.0)
    assert scores["rmse"] == pytest.approx(np.sqrt(50))
    assert scores["r2"] == pytest.approx(-1.0)
    perfect = regression_metrics([1, 2, 3], [1, 2, 3])
    assert perfect == {"mae": 0.0, "rmse": 0.0, "r2": 1.0}


def test_each_model_has_six_distinct_candidates():
    for model in rp.MODELS:
        cands = rp.candidates(model)
        assert len(cands) == 6
        assert len({str(c) for c in cands}) == 6


def test_folds_split_whole_engines(folds):
    assert len(folds) == rp.N_SPLITS
    all_val = []
    for fold in folds:
        assert not set(fold["train_units"]) & set(fold["val_units"])  # no engine on both sides
        assert len(fold["X_train"]) == len(fold["y_train"])
        assert len(fold["X_val"]) == len(fold["y_val"])
        all_val += fold["val_units"]
    # every training engine is validated exactly once
    assert sorted(all_val) == list(range(1, dl.EXPECTED_UNITS["FD002"]["train"] + 1))


def test_fold_scaler_is_fitted_on_training_folds_only(folds):
    raw = dl.load_train("FD002")
    fold = folds[0]
    train_rows = raw[raw["unit"].isin(fold["train_units"])]
    sensors = fold["scaler"].sensors_
    expected = train_rows.groupby(pp.condition_key(train_rows))[sensors].mean()
    pd.testing.assert_frame_equal(fold["scaler"].mean_, expected)
    # and it differs from statistics computed with the validation engines included
    everything = raw.groupby(pp.condition_key(raw))[sensors].mean()
    assert not np.allclose(fold["scaler"].mean_, everything)


@pytest.mark.parametrize("model", rp.MODELS)
def test_models_are_reproducible(folds, model):
    fold = folds[0]
    params = rp.candidates(model)[0]
    first = rp.make_model(model, params, n_estimators=10).fit(fold["X_train"], fold["y_train"])
    second = rp.make_model(model, params, n_estimators=10).fit(fold["X_train"], fold["y_train"])
    # The trees are identical. Random Forest averages them in parallel, and the
    # order of that sum can vary, so predictions agree to rounding (about 1e-13
    # cycles) rather than bit for bit.
    difference = np.abs(first.predict(fold["X_val"]) - second.predict(fold["X_val"])).max()
    assert difference < 1e-9


def test_fit_and_test_scores_one_prediction_per_test_engine():
    params = rp.candidates("xgboost")[0]
    metrics, predictions = rp.fit_and_test("FD001", "global", "raw", "xgboost", params, n_estimators=10)
    assert len(predictions) == 100 and predictions["unit"].tolist() == list(range(1, 101))
    assert predictions["true_rul_uncapped"].tolist() == dl.load_rul("FD001")["rul"].tolist()
    assert (predictions["true_rul"] == predictions["true_rul_uncapped"].clip(upper=125)).all()
    assert predictions["predicted_rul"].between(-5, 130).all()
    recomputed = regression_metrics(predictions["true_rul"], predictions["predicted_rul"])
    assert metrics["test_rmse"] == pytest.approx(recomputed["rmse"])
    assert metrics["n_features"] == 15


def test_selection_picks_lowest_mean_cv_rmse():
    scores = pd.DataFrame(
        {
            "subset": "FD001", "scheme": "global", "representation": "raw", "model": "xgboost",
            "params": ["a", "a", "b", "b"], "fold": [1, 2, 1, 2],
            "rmse": [10.0, 12.0, 9.0, 10.0], "mae": [8.0, 9.0, 7.0, 8.0], "r2": [0.8, 0.7, 0.85, 0.8],
        }
    )
    summary = rp.summarise_cv(scores)
    assert summary.loc[summary["selected"], "params"].tolist() == ["b"]
    assert summary.loc[summary["params"] == "a", "cv_rmse_mean"].iloc[0] == pytest.approx(11.0)
