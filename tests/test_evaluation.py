"""Checks for the bootstrap intervals and paired comparisons (synthetic data)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import evaluation as ev  # noqa: E402


def _predictions(noise: dict) -> pd.DataFrame:
    """Fake predictions for every configuration; ``noise`` sets the error size per model."""
    rng = np.random.default_rng(0)
    frames = []
    for subset, scheme in ev.DATA_CONFIGS:
        n = 100 if subset == "FD001" else 259
        true = rng.integers(5, 126, n).astype(float)
        for representation in ("raw", "summary"):
            for model in ev.MODEL_ORDER:
                frames.append(
                    pd.DataFrame(
                        {
                            "subset": subset, "scheme": scheme, "representation": representation,
                            "model": model, "unit": np.arange(1, n + 1), "true_rul": true,
                            "predicted_rul": true + rng.normal(0, noise[model], n),
                        }
                    )
                )
    return pd.concat(frames, ignore_index=True)


def test_intervals_contain_the_point_estimate_and_are_reproducible():
    predictions = _predictions({"random_forest": 10, "xgboost": 10})
    first, second = ev.bootstrap_intervals(predictions), ev.bootstrap_intervals(predictions)
    pd.testing.assert_frame_equal(first, second)  # fixed seed
    assert len(first) == 12
    for name in ("rmse", "mae", "r2"):
        assert (first[f"test_{name}_ci_low"] <= first[f"test_{name}"]).all()
        assert (first[f"test_{name}"] <= first[f"test_{name}_ci_high"]).all()
    # the point estimate is the plain metric on all engines
    one = predictions[(predictions["subset"] == "FD001") & (predictions["representation"] == "raw")
                      & (predictions["model"] == "xgboost")]
    expected = ev.regression_metrics(one["true_rul"], one["predicted_rul"])["rmse"]
    row = first[(first["subset"] == "FD001") & (first["representation"] == "raw") & (first["model"] == "xgboost")]
    assert np.isclose(row["test_rmse"].iloc[0], expected)


def test_paired_differences_detect_a_clearly_better_model():
    predictions = _predictions({"random_forest": 20, "xgboost": 5})  # XGBoost far more accurate
    diffs = ev.paired_rmse_differences(predictions)
    assert len(diffs) == 16
    model_rows = diffs[diffs["comparison"] == "model"]
    assert (model_rows["difference_B_minus_A"] < 0).all()
    assert model_rows["interval_excludes_zero"].all()


def test_paired_differences_do_not_flag_identical_predictions():
    predictions = _predictions({"random_forest": 10, "xgboost": 10})
    same = predictions.copy()
    rf = same[same["model"] == "random_forest"].sort_values(["subset", "scheme", "representation", "unit"])
    xgb_index = same[same["model"] == "xgboost"].sort_values(["subset", "scheme", "representation", "unit"]).index
    same.loc[xgb_index, "predicted_rul"] = rf["predicted_rul"].to_numpy()
    model_rows = ev.paired_rmse_differences(same)
    model_rows = model_rows[model_rows["comparison"] == "model"]
    assert np.allclose(model_rows["difference_B_minus_A"], 0)
    assert not model_rows["interval_excludes_zero"].any()
