"""Phase 5 - RUL prediction with Random Forest and XGBoost.

Covers Experiment 1 (model comparison) and Experiment 3 (feature
representation), including the normalisation comparison on FD002.

Procedure, identical for both models and every data configuration:

1. **Engine-level cross-validation.** The training engines are split into
   5 folds by engine number (GroupKFold), so all cycles of an engine are in
   the same fold. Preprocessing is refitted on the training folds each time.
2. **Hyperparameter choice.** Each model has a small grid of 6 candidates.
   The candidate with the lowest mean cross-validated RMSE is chosen.
3. **Final fit.** The chosen candidate is refitted on all training engines.
4. **Test.** One prediction per official test engine, at its last observed
   cycle, compared with the NASA RUL value. The test engines are used once,
   after all choices are made.
"""

from __future__ import annotations

import itertools
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from . import data_loader as dl
from . import feature_engineering as fe
from . import preprocessing as pp
from .evaluation import regression_metrics

METRIC_DIR = dl.PROJECT_ROOT / "results" / "metrics"
TABLE_DIR = dl.PROJECT_ROOT / "results" / "tables"

SEED = 42
N_SPLITS = 5
TARGET = "rul"  # the capped RUL is what the models learn

MODELS = ("random_forest", "xgboost")
MODEL_LABELS = {"random_forest": "Random Forest", "xgboost": "XGBoost"}

# Settings that stay the same for every candidate.
FIXED_PARAMS = {
    "random_forest": {"n_estimators": 200, "n_jobs": -1, "random_state": SEED},
    "xgboost": {
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "n_jobs": -1,
        "random_state": SEED,
    },
}

# Settings that are tuned: 6 candidates per model.
PARAM_GRIDS = {
    "random_forest": {"max_features": ["sqrt", 0.33], "min_samples_leaf": [5, 20, 50]},
    "xgboost": {"max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1]},
}

CONFIG_COLS = ["subset", "scheme", "representation", "model"]


def candidates(model: str) -> list[dict]:
    """Every combination of the tuned settings for one model."""
    grid = PARAM_GRIDS[model]
    return [dict(zip(grid, values)) for values in itertools.product(*grid.values())]


def make_model(model: str, params: dict, **overrides):
    """Create an untrained model. ``overrides`` replace fixed settings (used in tests)."""
    settings = {**FIXED_PARAMS[model], **params, **overrides}
    if model == "random_forest":
        return RandomForestRegressor(**settings)
    if model == "xgboost":
        return XGBRegressor(**settings)
    raise ValueError(f"Unknown model {model!r}. Use one of {MODELS}.")


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------

def make_folds(subset: str, scheme: str, representation: str) -> list[dict]:
    """Prepare the 5 train/validation splits for one data configuration.

    Engines are assigned to folds as whole units. For each split the sensors
    to drop and the scaling statistics are learned from the training folds
    only, then the features are built.
    """
    raw = pp.add_train_rul(dl.load_train(subset))
    folds = []
    splitter = GroupKFold(n_splits=N_SPLITS)
    for train_idx, val_idx in splitter.split(raw, groups=raw["unit"]):
        fitted = pp.fit_and_apply(raw.iloc[train_idx], {"val": raw.iloc[val_idx]}, scheme)
        names = fe.feature_names(fitted["sensors"], representation)
        train = fe.make_features(fitted["train"], fitted["sensors"], representation)
        val = fe.make_features(fitted["val"], fitted["sensors"], representation)
        folds.append(
            {
                "X_train": train[names], "y_train": train[TARGET],
                "X_val": val[names], "y_val": val[TARGET],
                "train_units": sorted(train["unit"].unique()),
                "val_units": sorted(val["unit"].unique()),
                "scaler": fitted["scaler"],
            }
        )
    return folds


def cross_validate(subset: str, scheme: str, representation: str, log=print, **overrides) -> pd.DataFrame:
    """Score every candidate of both models on the 5 folds.

    Returns one row per (model, candidate, fold) with MAE, RMSE and R2 on all
    cycles of the validation engines.
    """
    folds = make_folds(subset, scheme, representation)
    rows = []
    for model in MODELS:
        for params in candidates(model):
            start = time.perf_counter()
            for k, fold in enumerate(folds, start=1):
                estimator = make_model(model, params, **overrides)
                estimator.fit(fold["X_train"], fold["y_train"])
                scores = regression_metrics(fold["y_val"], estimator.predict(fold["X_val"]))
                rows.append(
                    {
                        "subset": subset, "scheme": scheme, "representation": representation,
                        "model": model, "params": json.dumps(params), "fold": k,
                        "val_engines": len(fold["val_units"]), "val_rows": len(fold["y_val"]),
                        **scores,
                    }
                )
            seconds = time.perf_counter() - start
            recent = pd.DataFrame(rows[-N_SPLITS:])
            log(
                f"  CV {subset}/{scheme}/{representation} {model:13s} {json.dumps(params):48s} "
                f"RMSE {recent['rmse'].mean():6.2f} +/- {recent['rmse'].std(ddof=1):4.2f}  ({seconds:5.1f}s)"
            )
    return pd.DataFrame(rows)


def summarise_cv(fold_scores: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over folds for every candidate, best first."""
    summary = (
        fold_scores.groupby(CONFIG_COLS + ["params"], sort=False)
        .agg(
            cv_rmse_mean=("rmse", "mean"), cv_rmse_std=("rmse", lambda s: s.std(ddof=1)),
            cv_mae_mean=("mae", "mean"), cv_mae_std=("mae", lambda s: s.std(ddof=1)),
            cv_r2_mean=("r2", "mean"), cv_r2_std=("r2", lambda s: s.std(ddof=1)),
        )
        .reset_index()
    )
    summary["selected"] = (
        summary.groupby(CONFIG_COLS)["cv_rmse_mean"].transform("min") == summary["cv_rmse_mean"]
    )
    return summary


# --------------------------------------------------------------------------
# Final fit and official test set
# --------------------------------------------------------------------------

def fit_and_test(subset: str, scheme: str, representation: str, model: str, params: dict, **overrides):
    """Fit on all training engines and score the official test engines.

    Returns ``(metrics, predictions)``. ``metrics`` has the scores against the
    capped target (main result) and the uncapped NASA values. ``predictions``
    has one row per test engine.
    """
    built = fe.build(subset, scheme, representation)
    estimator = make_model(model, params, **overrides)
    start = time.perf_counter()
    estimator.fit(built["train"][built["features"]], built["train"][TARGET])
    fit_seconds = time.perf_counter() - start

    last = fe.last_cycle_rows(built["test"])
    predicted = estimator.predict(last[built["features"]])
    capped = regression_metrics(last["rul"], predicted)
    uncapped = regression_metrics(last["rul_uncapped"], predicted)

    metrics = {
        "subset": subset, "scheme": scheme, "representation": representation, "model": model,
        "params": json.dumps(params), "n_features": len(built["features"]),
        "test_engines": len(last),
        "test_mae": capped["mae"], "test_rmse": capped["rmse"], "test_r2": capped["r2"],
        "test_mae_uncapped": uncapped["mae"], "test_rmse_uncapped": uncapped["rmse"],
        "test_r2_uncapped": uncapped["r2"],
        "fit_seconds": fit_seconds,
    }
    predictions = pd.DataFrame(
        {
            "subset": subset, "scheme": scheme, "representation": representation, "model": model,
            "unit": last["unit"].to_numpy(), "last_cycle": last["cycle"].to_numpy(),
            "true_rul_uncapped": last["rul_uncapped"].to_numpy(),
            "true_rul": last["rul"].to_numpy(),
            "predicted_rul": predicted,
        }
    )
    return metrics, predictions


# --------------------------------------------------------------------------
# Run everything
# --------------------------------------------------------------------------

def result_paths() -> dict:
    """Where the Phase 5 result files are stored."""
    return {
        "phase5_cv_fold_scores": METRIC_DIR / "phase5_cv_fold_scores.csv",
        "phase5_cv_summary": METRIC_DIR / "phase5_cv_summary.csv",
        "phase5_test_metrics": METRIC_DIR / "phase5_test_metrics.csv",
        "phase5_test_predictions": TABLE_DIR / "phase5_test_predictions.csv",
        "phase5_experiment_config": METRIC_DIR / "phase5_experiment_config.json",
    }


def _settings() -> dict:
    """Every setting that changes the numbers. Used to tell if saved results are current."""
    return {
        "seed": SEED,
        "target": f"RUL capped at {pp.RUL_CAP}",
        "summary_window": fe.WINDOW,
        "fixed_params": FIXED_PARAMS,
        "param_grids": PARAM_GRIDS,
    }


def results_are_current() -> bool:
    """True if all result files exist and were produced with the current settings."""
    paths = result_paths()
    if not all(path.exists() for path in paths.values()):
        return False
    saved = json.loads(paths["phase5_experiment_config"].read_text())
    current = json.loads(json.dumps(_settings()))  # same types as the saved JSON
    return all(saved.get(key) == value for key, value in current.items())


def run_all(log=print, force: bool = False) -> dict:
    """Run Experiments 1 and 3 and write every result file.

    Training all 372 models takes several minutes, so it is done once. If the
    saved results were produced with the current settings they are reused;
    pass ``force=True`` (or ``--force`` on the command line) to retrain.
    """
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    if not force and results_are_current():
        log("Saved Phase 5 results match the current settings; nothing was retrained.")
        return result_paths()
    started = time.perf_counter()

    fold_scores = []
    for subset, scheme in pp.EXPERIMENT_GRID:
        for representation in fe.REPRESENTATIONS:
            log(f"Cross-validation: {subset} / {scheme} / {representation}")
            fold_scores.append(cross_validate(subset, scheme, representation, log=log))
    fold_scores = pd.concat(fold_scores, ignore_index=True)
    cv_summary = summarise_cv(fold_scores)

    metrics, predictions = [], []
    for _, best in cv_summary[cv_summary["selected"]].iterrows():
        params = json.loads(best["params"])
        m, p = fit_and_test(best["subset"], best["scheme"], best["representation"], best["model"], params)
        for col in ("cv_rmse_mean", "cv_rmse_std", "cv_mae_mean", "cv_mae_std", "cv_r2_mean", "cv_r2_std"):
            m[col] = best[col]
        metrics.append(m)
        predictions.append(p)
        log(
            f"Test {m['subset']}/{m['scheme']}/{m['representation']} {m['model']:13s} "
            f"RMSE {m['test_rmse']:6.2f}  MAE {m['test_mae']:6.2f}  R2 {m['test_r2']:.3f}"
        )

    saved = {}
    outputs = {
        METRIC_DIR / "phase5_cv_fold_scores.csv": fold_scores,
        METRIC_DIR / "phase5_cv_summary.csv": cv_summary,
        METRIC_DIR / "phase5_test_metrics.csv": pd.DataFrame(metrics),
        TABLE_DIR / "phase5_test_predictions.csv": pd.concat(predictions, ignore_index=True),
    }
    for path, table in outputs.items():
        table.to_csv(path, index=False, float_format="%.6g")
        saved[path.stem] = path

    config = {
        "seed": SEED,
        "cv": f"GroupKFold by engine, {N_SPLITS} folds, preprocessing refitted on the training folds",
        "selection_rule": "lowest mean cross-validated RMSE on the capped RUL, all cycles of validation engines",
        "target": f"RUL capped at {pp.RUL_CAP}",
        "test_protocol": "one prediction per test engine at its last observed cycle",
        "summary_window": fe.WINDOW,
        "fixed_params": FIXED_PARAMS,
        "param_grids": PARAM_GRIDS,
        "total_minutes": round((time.perf_counter() - started) / 60, 1),
    }
    config_path = METRIC_DIR / "phase5_experiment_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    saved[config_path.stem] = config_path
    log(f"Done in {config['total_minutes']} minutes.")
    return saved


if __name__ == "__main__":
    def _log(message: str) -> None:
        print(message, flush=True)

    import sys

    for name, path in run_all(log=_log, force="--force" in sys.argv).items():
        print(f"{name}: {path.relative_to(dl.PROJECT_ROOT)}")
