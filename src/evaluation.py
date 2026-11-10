"""Evaluation metrics used in the thesis.

RUL prediction (regression):

* MAE  - mean absolute error: the average size of the error in cycles.
  Easy to interpret; every cycle of error counts the same.
* RMSE - root mean squared error: also in cycles, but large errors count
  more than small ones. This is the standard headline metric on C-MAPSS.
* R2   - coefficient of determination: the share of the variation in the
  true RUL that the predictions explain. 1 is perfect, 0 is no better than
  always predicting the mean, and negative values are worse than that.

Anomaly detection (classification of cycles): precision, recall and F1,
explained in ``detection_metrics``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import plot_style as ps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRIC_DIR = PROJECT_ROOT / "results" / "metrics"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

# The three data configurations, in the order used in every table and figure.
DATA_CONFIGS = [("FD001", "global"), ("FD002", "global"), ("FD002", "condition")]
CONFIG_LABELS = {
    ("FD001", "global"): "FD001",
    ("FD002", "global"): "FD002, dataset-wide normalisation",
    ("FD002", "condition"): "FD002, per-condition normalisation",
}
MODEL_ORDER = ["random_forest", "xgboost"]
MODEL_LABELS = {"random_forest": "Random Forest", "xgboost": "XGBoost"}
REPRESENTATION_LABELS = {"raw": "Raw sensors", "summary": "Summary statistics"}


def detection_metrics(is_anomaly, flagged) -> dict[str, float]:
    """Precision, recall and F1 for anomaly flags against reference labels.

    * Precision - of the cycles that were flagged, the share that are
      anomalous according to the reference label. Low precision = many
      false alarms.
    * Recall    - of the cycles labelled anomalous, the share that were
      flagged. Low recall = many missed anomalies.
    * F1        - harmonic mean of precision and recall; high only when both
      are high. Suitable here because anomalous cycles are a small minority,
      so plain accuracy would look good even for a detector that never fires.

    Also returns the counts behind the scores and the share of anomalous
    cycles (prevalence), which is needed to interpret precision.
    """
    truth = np.asarray(is_anomaly, dtype=bool)
    alarm = np.asarray(flagged, dtype=bool)
    tp = int((truth & alarm).sum())
    fp = int((~truth & alarm).sum())
    fn = int((truth & ~alarm).sum())
    tn = int((~truth & ~alarm).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "cycles": int(truth.size), "prevalence": float(truth.mean()) if truth.size else 0.0,
        "alarm_rate": float(alarm.mean()) if alarm.size else 0.0,
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE and R2 for one set of predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


# --------------------------------------------------------------------------
# Phase 5 result tables and figures (read from the saved result files, so
# figures can be redrawn without retraining any model)
# --------------------------------------------------------------------------

def load_rul_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test metrics (one row per model configuration) and per-engine predictions."""
    metrics = pd.read_csv(METRIC_DIR / "phase5_test_metrics.csv")
    predictions = pd.read_csv(TABLE_DIR / "phase5_test_predictions.csv")
    return metrics, predictions


def rul_results_table(metrics: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tidy results table for the thesis: one row per configuration."""
    if metrics is None:
        metrics, _ = load_rul_results()
    table = metrics.copy()
    table["data"] = [CONFIG_LABELS[(s, c)] for s, c in zip(table["subset"], table["scheme"])]
    table["representation"] = table["representation"].map(REPRESENTATION_LABELS)
    table["model"] = table["model"].map(MODEL_LABELS)
    table["cv_rmse"] = [f"{m:.2f} +/- {s:.2f}" for m, s in zip(table["cv_rmse_mean"], table["cv_rmse_std"])]
    intervals = bootstrap_intervals().set_index(CONFIG_KEYS)
    keys = list(zip(metrics["subset"], metrics["scheme"], metrics["representation"], metrics["model"]))
    table["test_rmse_95ci"] = [
        f"{intervals.loc[k, 'test_rmse_ci_low']:.2f} to {intervals.loc[k, 'test_rmse_ci_high']:.2f}" for k in keys
    ]
    columns = [
        "data", "representation", "model", "n_features", "cv_rmse",
        "test_mae", "test_rmse", "test_rmse_95ci", "test_r2",
        "test_mae_uncapped", "test_rmse_uncapped", "test_r2_uncapped", "params",
    ]
    return table[columns].round(3)


def _select(metrics: pd.DataFrame, subset: str, scheme: str, **where) -> pd.DataFrame:
    mask = (metrics["subset"] == subset) & (metrics["scheme"] == scheme)
    for column, value in where.items():
        mask &= metrics[column] == value
    return metrics[mask]


# --------------------------------------------------------------------------
# Uncertainty of the test scores (bootstrap over test engines)
# --------------------------------------------------------------------------
# A test set of 100 or 259 engines is a sample. Resampling those engines with
# replacement many times shows how much a score would move with a different
# sample of engines. No model is retrained; only saved predictions are used.

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42
CONFIG_KEYS = ["subset", "scheme", "representation", "model"]


def _resample_indices(n_engines: int) -> np.ndarray:
    """The same engine resamples for every configuration of a subset (paired)."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return rng.integers(0, n_engines, size=(N_BOOTSTRAP, n_engines))


def _bootstrap_scores(true: np.ndarray, predicted: np.ndarray, idx: np.ndarray) -> dict[str, np.ndarray]:
    t, p = true[idx], predicted[idx]
    squared = ((p - t) ** 2).mean(axis=1)
    return {
        "rmse": np.sqrt(squared),
        "mae": np.abs(p - t).mean(axis=1),
        "r2": 1 - squared / t.var(axis=1),
    }


def bootstrap_intervals(predictions: pd.DataFrame | None = None) -> pd.DataFrame:
    """95% bootstrap interval of test MAE, RMSE and R2 for every configuration."""
    if predictions is None:
        _, predictions = load_rul_results()
    rows = []
    for keys, group in predictions.groupby(CONFIG_KEYS, sort=False):
        group = group.sort_values("unit")
        true = group["true_rul"].to_numpy(float)
        predicted = group["predicted_rul"].to_numpy(float)
        scores = _bootstrap_scores(true, predicted, _resample_indices(len(group)))
        row = dict(zip(CONFIG_KEYS, keys))
        for name, values in scores.items():
            row[f"test_{name}"] = regression_metrics(true, predicted)[name]
            row[f"test_{name}_ci_low"], row[f"test_{name}_ci_high"] = np.percentile(values, [2.5, 97.5])
        rows.append(row)
    return pd.DataFrame(rows)


def paired_rmse_differences(predictions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Test RMSE differences for the three comparisons, with 95% intervals.

    Each row compares two configurations on the SAME test engines:
    ``difference = RMSE(B) - RMSE(A)``, so a negative value means B is better.
    If the interval contains 0, the data do not show a clear difference.
    """
    if predictions is None:
        _, predictions = load_rul_results()

    def errors(subset, scheme, representation, model):
        g = _select(predictions, subset, scheme, representation=representation, model=model).sort_values("unit")
        return g["true_rul"].to_numpy(float), g["predicted_rul"].to_numpy(float)

    def compare(kind, context, label_a, a, label_b, b):
        (true, pred_a), (_, pred_b) = errors(*a), errors(*b)
        idx = _resample_indices(len(true))
        boot_a = _bootstrap_scores(true, pred_a, idx)["rmse"]
        boot_b = _bootstrap_scores(true, pred_b, idx)["rmse"]
        low, high = np.percentile(boot_b - boot_a, [2.5, 97.5])
        rmse_a = regression_metrics(true, pred_a)["rmse"]
        rmse_b = regression_metrics(true, pred_b)["rmse"]
        return {
            "comparison": kind, "context": context, "A": label_a, "B": label_b,
            "rmse_A": rmse_a, "rmse_B": rmse_b, "difference_B_minus_A": rmse_b - rmse_a,
            "ci_low": low, "ci_high": high, "interval_excludes_zero": bool(low > 0 or high < 0),
        }

    rows = []
    for subset, scheme in DATA_CONFIGS:
        data = CONFIG_LABELS[(subset, scheme)]
        for rep in ("raw", "summary"):
            rows.append(compare(
                "model", f"{data}; {REPRESENTATION_LABELS[rep].lower()}",
                "Random Forest", (subset, scheme, rep, "random_forest"),
                "XGBoost", (subset, scheme, rep, "xgboost"),
            ))
    for subset, scheme in DATA_CONFIGS:
        data = CONFIG_LABELS[(subset, scheme)]
        for model in MODEL_ORDER:
            rows.append(compare(
                "representation", f"{data}; {MODEL_LABELS[model]}",
                "Raw sensors", (subset, scheme, "raw", model),
                "Summary statistics", (subset, scheme, "summary", model),
            ))
    for rep in ("raw", "summary"):
        for model in MODEL_ORDER:
            rows.append(compare(
                "normalisation", f"FD002; {REPRESENTATION_LABELS[rep].lower()}; {MODEL_LABELS[model]}",
                "Dataset-wide", ("FD002", "global", rep, model),
                "Per-condition", ("FD002", "condition", rep, model),
            ))
    return pd.DataFrame(rows)


def plot_actual_vs_predicted(representation: str = "summary"):
    """Figure 4 - predicted against true RUL for every test engine."""
    ps.apply_style()
    metrics, predictions = load_rul_results()
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.2), sharex=True, sharey=True)
    limit = 135

    for r, model in enumerate(MODEL_ORDER):
        for c, (subset, scheme) in enumerate(DATA_CONFIGS):
            ax = axes[r, c]
            points = _select(predictions, subset, scheme, model=model, representation=representation)
            row = _select(metrics, subset, scheme, model=model, representation=representation).iloc[0]
            ax.plot([0, limit], [0, limit], color=ps.MUTED, linewidth=1.0, linestyle="--", zorder=1)
            ax.scatter(
                points["true_rul"], points["predicted_rul"], s=16, color=ps.COLORS[subset],
                alpha=0.65, edgecolor="white", linewidth=0.4, zorder=2,
            )
            ax.text(
                0.04, 0.96,
                f"RMSE = {row['test_rmse']:.2f}\nMAE = {row['test_mae']:.2f}\n$R^2$ = {row['test_r2']:.3f}\nn = {len(points)} engines",
                transform=ax.transAxes, ha="left", va="top", fontsize=8, color=ps.INK_SOFT,
            )
            ax.set_xlim(0, limit)
            ax.set_ylim(0, limit)
            ax.set_aspect("equal")
            if r == 0:
                ax.set_title(CONFIG_LABELS[(subset, scheme)].replace(", ", "\n"), fontsize=9.5)
            if r == 1:
                ax.set_xlabel("True RUL at last observed cycle (capped at 125)")
            if c == 0:
                ax.set_ylabel(f"{MODEL_LABELS[model]}\nPredicted RUL (cycles)")
    axes[0, 0].plot([], [], color=ps.MUTED, linewidth=1.0, linestyle="--", label="Perfect prediction")
    axes[0, 0].legend(loc="lower right", fontsize=7.5)
    fig.suptitle(
        f"Actual vs predicted RUL on the official test engines ({REPRESENTATION_LABELS[representation].lower()})"
    )
    fig.tight_layout()
    return fig


def plot_model_comparison():
    """Figure 5 - Random Forest vs XGBoost on the test engines."""
    ps.apply_style()
    scores = bootstrap_intervals()
    measures = [("rmse", "RMSE (cycles)", ".1f"), ("mae", "MAE (cycles)", ".1f"), ("r2", "$R^2$", ".2f")]
    representations = ["raw", "summary"]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.8), sharex=True, sharey="col")
    x = np.arange(len(DATA_CONFIGS))
    width = 0.36

    for r, representation in enumerate(representations):
        for c, (name, label, fmt) in enumerate(measures):
            ax = axes[r, c]
            for i, model in enumerate(MODEL_ORDER):
                rows = [
                    _select(scores, s, sc, model=model, representation=representation).iloc[0]
                    for s, sc in DATA_CONFIGS
                ]
                values = np.array([row[f"test_{name}"] for row in rows])
                low = np.array([row[f"test_{name}_ci_low"] for row in rows])
                high = np.array([row[f"test_{name}_ci_high"] for row in rows])
                positions = x + (i - 0.5) * width
                ax.bar(
                    positions, values, width * 0.92, color=ps.MODEL_COLORS[model],
                    label=MODEL_LABELS[model] if (r, c) == (0, 0) else None,
                )
                ax.errorbar(
                    positions, values, yerr=[values - low, high - values], fmt="none",
                    ecolor=ps.INK, elinewidth=0.9, capsize=2.5, capthick=0.9,
                )
                for position, value, top in zip(positions, values, high):
                    ax.annotate(
                        format(value, fmt), (position, top), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5, color=ps.INK_SOFT,
                    )
            ax.grid(axis="x", visible=False)
            ax.set_xticks(x, ["FD001", "FD002\ndataset-wide", "FD002\nper-condition"])
            ax.margins(y=0.10)
            if r == 0:
                ax.set_title(label)
            if c == 0:
                ax.set_ylabel(REPRESENTATION_LABELS[representation], fontsize=10, color=ps.INK, fontweight="bold")
    fig.suptitle("Random Forest vs XGBoost: test-set performance by dataset and feature representation")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, fontsize=9)
    fig.text(
        0.5, 0.005,
        "Lower is better for RMSE and MAE; higher is better for $R^2$. One prediction per test engine, target "
        "capped at 125 cycles.\nError bars: 95% bootstrap interval over test engines.",
        ha="center", fontsize=8, color=ps.INK_SOFT,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    return fig


def plot_representation_comparison():
    """Figure 7 - raw sensors vs summary statistics (test RMSE)."""
    ps.apply_style()
    scores = bootstrap_intervals()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.6), sharey=True)
    dodge = {"random_forest": -0.09, "xgboost": 0.09}  # side-by-side so intervals do not overlap

    for ax, (subset, scheme) in zip(axes, DATA_CONFIGS):
        for model in MODEL_ORDER:
            rows = [
                _select(scores, subset, scheme, model=model, representation=rep).iloc[0]
                for rep in ("raw", "summary")
            ]
            values = np.array([row["test_rmse"] for row in rows])
            low = np.array([row["test_rmse_ci_low"] for row in rows])
            high = np.array([row["test_rmse_ci_high"] for row in rows])
            xs = np.array([0, 1]) + dodge[model]
            color = ps.MODEL_COLORS[model]
            ax.errorbar(
                xs, values, yerr=[values - low, high - values], color=color, linewidth=1.6,
                marker="o", markersize=6.5, elinewidth=1.1, capsize=3,
                label=MODEL_LABELS[model] if ax is axes[0] else None,
            )
            for xi, value, top in zip(xs, values, high):  # value label above the interval
                ax.annotate(
                    f"{value:.1f}", (xi, top), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=ps.INK_SOFT,
                )
        ax.set_xticks([0, 1], ["Raw sensors", "Summary statistics"])
        ax.set_xlim(-0.5, 1.5)
        ax.grid(axis="x", visible=False)
        ax.set_title(CONFIG_LABELS[(subset, scheme)].replace(", ", "\n"), fontsize=9.5)
    axes[0].set_ylabel("Test RMSE (cycles, lower is better)")
    fig.suptitle("Effect of the feature representation on test RMSE")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, fontsize=9)
    fig.text(
        0.5, 0.005,
        "Points: test RMSE, target capped at 125 cycles. Error bars: 95% bootstrap interval over test engines. "
        "The y-axis does not start at zero.",
        ha="center", fontsize=8, color=ps.INK_SOFT,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    return fig


def save_rul_figures() -> dict:
    """Draw and save the Phase 5 figures and the thesis results table."""
    saved = {}
    tables = {
        TABLE_DIR / "phase5_rul_results_table.csv": rul_results_table(),
        METRIC_DIR / "phase5_test_metrics_intervals.csv": bootstrap_intervals(),
        METRIC_DIR / "phase5_paired_differences.csv": paired_rmse_differences(),
    }
    for path, table in tables.items():
        table.to_csv(path, index=False, float_format="%.6g")
        saved[path.stem] = path
    figures = {
        "fig04_actual_vs_predicted_summary": plot_actual_vs_predicted("summary"),
        "fig05_model_comparison": plot_model_comparison(),
        "fig07_representation_comparison": plot_representation_comparison(),
    }
    for name, fig in figures.items():
        saved[name] = ps.save_figure(fig, name)
        plt.close(fig)
    return saved


# --------------------------------------------------------------------------
# Phase 6 result tables and figures (anomaly detection)
# --------------------------------------------------------------------------

EVALUATION_LABELS = {
    "cross-validation (training engines)": "Cross-validation, training engines (run to failure)",
    "official test engines": "Official test engines",
}
ANOMALY_ZONE_FILL = "#f0efec"


def load_anomaly_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Metrics, alarm rate by RUL bin, and per-cycle test scores."""
    metrics = pd.read_csv(METRIC_DIR / "phase6_anomaly_metrics.csv")
    rates = pd.read_csv(METRIC_DIR / "phase6_alarm_rate_by_rul.csv")
    scores = pd.read_csv(TABLE_DIR / "phase6_test_scores.csv")
    return metrics, rates, scores


def anomaly_results_table(metrics: pd.DataFrame | None = None, main_only: bool = True) -> pd.DataFrame:
    """Tidy precision/recall/F1 table for the thesis."""
    if metrics is None:
        metrics, _, _ = load_anomaly_results()
    table = metrics[metrics["main_setting"]].copy() if main_only else metrics.copy()
    table["data"] = [CONFIG_LABELS[(s, c)] for s, c in zip(table["subset"], table["scheme"])]
    table["representation"] = table["representation"].map(REPRESENTATION_LABELS)
    table["evaluation"] = table["evaluation"].map(
        {"cross-validation (training engines)": "Cross-validation", "official test engines": "Test"}
    )
    table["anomalous_share_percent"] = 100 * table["prevalence"]
    table["flagged_share_percent"] = 100 * table["alarm_rate"]
    columns = [
        "data", "representation", "evaluation", "scoring", "alarm_budget", "precision", "recall", "f1",
        "tp", "fp", "fn", "cycles", "anomalous_share_percent", "flagged_share_percent",
    ]
    return table[columns].round(3)


def plot_anomaly_trajectories():
    """Figure 6a - anomaly score over the life of one test engine per dataset."""
    ps.apply_style()
    _, _, scores = load_anomaly_results()
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), sharex="col")

    for c, (subset, scheme) in enumerate(DATA_CONFIGS):
        for r, representation in enumerate(("raw", "summary")):
            ax = axes[r, c]
            part = scores[(scores["subset"] == subset) & (scores["scheme"] == scheme)
                          & (scores["representation"] == representation)]
            # the test engine that is closest to failure at its last cycle (ties: lowest number)
            last = part.groupby("unit")["rul_uncapped"].min()
            unit = int(last[last == last.min()].index.min())
            engine = part[part["unit"] == unit]
            flagged = engine["flag_05"]

            zone = engine.loc[engine["is_anomaly"], "cycle"]
            ax.axvspan(zone.min() - 0.5, zone.max() + 0.5, color=ANOMALY_ZONE_FILL, zorder=0,
                       label="Reference anomaly zone (RUL $\\leq$ 30)")
            ax.axhline(engine["threshold"].iloc[0], color=ps.INK_SOFT, linewidth=1.0, linestyle="--",
                       label="Alarm threshold (5% budget)")
            ax.scatter(engine.loc[~flagged, "cycle"], engine.loc[~flagged, "score"], s=7,
                       color=ps.MUTED, label="Cycle not flagged", zorder=2)
            ax.scatter(engine.loc[flagged, "cycle"], engine.loc[flagged, "score"], s=13,
                       color=ps.COLORS[subset], label="Cycle flagged", zorder=3)
            if engine["warm_up"].any():
                ax.axvline(engine.loc[~engine["warm_up"], "cycle"].min(), color=ps.AXIS, linewidth=0.9, linestyle=":")
                ax.text(0.01, 0.03, "warm-up", transform=ax.transAxes, fontsize=7.5, color=ps.INK_SOFT)
            if r == 0:
                ax.set_title(f"{CONFIG_LABELS[(subset, scheme)].replace(', ', chr(10))}\n(test engine {unit}, "
                             f"true RUL {int(last.min())} at last cycle)", fontsize=9.5)
            if r == 1:
                ax.set_xlabel("Operating cycle")
            if c == 0:
                ax.set_ylabel(f"{REPRESENTATION_LABELS[representation]}\nAnomaly score")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Isolation Forest anomaly score over the observed life of a test engine")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=4, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def plot_alarm_rate_by_rul():
    """Figure 6b - share of cycles flagged as remaining useful life falls."""
    ps.apply_style()
    _, rates, _ = load_anomaly_results()
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), sharex=True, sharey=True)
    styles = {
        "raw": {"linestyle": "--", "marker": "o", "markerfacecolor": "white", "label": "Raw sensors"},
        "summary": {"linestyle": "-", "marker": "o", "label": "Summary statistics"},
    }

    for r, evaluation in enumerate(EVALUATION_LABELS):
        for c, (subset, scheme) in enumerate(DATA_CONFIGS):
            ax = axes[r, c]
            for representation, style in styles.items():
                part = rates[(rates["subset"] == subset) & (rates["scheme"] == scheme)
                             & (rates["representation"] == representation) & (rates["evaluation"] == evaluation)]
                part = part.sort_values("bin_upper", ascending=False)  # healthy on the left, failure on the right
                x = np.arange(len(part))
                ax.plot(x, 100 * part["alarm_rate"], color=ps.COLORS[subset], linewidth=1.6, markersize=4.5, **style)
            ax.axvspan(len(part) - 3.5, len(part) - 0.5, color=ANOMALY_ZONE_FILL, zorder=0)
            ax.axhline(5, color=ps.INK_SOFT, linewidth=0.9, linestyle=":")
            ticks = sorted(set(x[::2]) | {x[-1]})  # every second bin, plus the last one
            ax.set_xticks(ticks, part["rul_bin"].iloc[ticks], rotation=45, ha="right")
            ax.set_ylim(-3, 103)
            if r == 0:
                ax.set_title(CONFIG_LABELS[(subset, scheme)].replace(", ", "\n"), fontsize=9.5)
            if r == 1:
                ax.set_xlabel("Remaining useful life (cycles)")
            if c == 0:
                ax.set_ylabel(f"{EVALUATION_LABELS[evaluation].split(',')[0]}\nCycles flagged (%)")
    handles = [
        plt.Line2D([], [], color=ps.INK_SOFT, linewidth=1.6, markersize=4.5, **style) for style in styles.values()
    ]
    handles.append(plt.Line2D([], [], color=ps.INK_SOFT, linewidth=0.9, linestyle=":", label="5% alarm budget"))
    handles.append(plt.Rectangle((0, 0), 1, 1, color=ANOMALY_ZONE_FILL, label="Reference anomaly zone (RUL $\\leq$ 30)"))
    fig.suptitle("Share of cycles flagged by the Isolation Forest as engines approach failure")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=4, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def save_anomaly_figures() -> dict:
    """Draw and save the Phase 6 figures and the thesis results table."""
    saved = {}
    table_path = TABLE_DIR / "phase6_anomaly_results_table.csv"
    anomaly_results_table().to_csv(table_path, index=False, float_format="%.6g")
    saved[table_path.stem] = table_path
    figures = {
        "fig06a_anomaly_score_trajectories": plot_anomaly_trajectories(),
        "fig06b_alarm_rate_by_rul": plot_alarm_rate_by_rul(),
    }
    for name, fig in figures.items():
        saved[name] = ps.save_figure(fig, name)
        plt.close(fig)
    return saved


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    for name, path in {**save_rul_figures(), **save_anomaly_figures()}.items():
        print(f"saved {name}: {path.relative_to(PROJECT_ROOT)}")
