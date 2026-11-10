"""Phase 7 - bring the results together for Chapter 4.

Everything here READS the result files written by Phases 1-6. No model is
trained and no number is typed by hand, so the thesis tables cannot drift
away from the experiments.

Outputs:

* ``results/tables/thesis/tableNN_*.csv`` - the tables for Chapter 4
* ``results/figures/fig08_dataset_comparison`` - FD001 vs FD002
* ``reports/chapter4_tables_and_figures.md`` - all tables with captions and
  the list of figures, ready to copy into the thesis
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import evaluation as ev
from . import plot_style as ps

ROOT = ev.PROJECT_ROOT
TABLE_DIR = ROOT / "results" / "tables"
METRIC_DIR = ROOT / "results" / "metrics"
THESIS_DIR = TABLE_DIR / "thesis"
REPORT_PATH = ROOT / "reports" / "chapter4_tables_and_figures.md"

CONFIG_SHORT = {
    ("FD001", "global"): "FD001",
    ("FD002", "global"): "FD002 dataset-wide",
    ("FD002", "condition"): "FD002 per-condition",
}

# The eight visualisations required by the thesis brief and the files that provide them.
FIGURES = [
    ("Supporting: research framework", "fig00_research_framework",
     "Stages of the study from data to research questions (Chapter 3)."),
    ("1. Dataset distribution", "fig01_dataset_distribution",
     "Trajectory lengths of training and test engines and true test RUL, FD001 and FD002."),
    ("2. Sensor trends over operating cycles", "fig02a_sensor_trends_FD001",
     "All 21 sensors over the life of five FD001 training engines."),
    ("2. Sensor trends over operating cycles", "fig02b_operating_condition_effect",
     "Four sensors for one engine in FD001 (one operating condition) and FD002 (six)."),
    ("3. RUL degradation curves", "fig03a_rul_target",
     "Linear RUL and the piecewise linear target capped at 125 cycles."),
    ("Supporting: preprocessing", "fig03b_normalisation_effect_FD002",
     "One FD002 engine under dataset-wide and per-condition normalisation."),
    ("Supporting: feature representation", "fig03c_summary_features_example",
     "Raw readings, 30-cycle mean and 30-cycle standard deviation of sensor s11."),
    ("4. Actual vs predicted RUL", "fig04_actual_vs_predicted_summary",
     "Predicted against true RUL for every test engine, both models, three data configurations."),
    ("5. Random Forest vs XGBoost", "fig05_model_comparison",
     "Test RMSE, MAE and R2 with 95% bootstrap intervals."),
    ("6. Anomaly detection", "fig06a_anomaly_score_trajectories",
     "Anomaly score over the observed life of a test engine, with alarm threshold and reference zone."),
    ("6. Anomaly detection", "fig06b_alarm_rate_by_rul",
     "Share of cycles flagged as remaining useful life falls, cross-validation and test."),
    ("7. Feature representation comparison", "fig07_representation_comparison",
     "Test RMSE with raw sensors and with summary statistics, with 95% bootstrap intervals."),
    ("8. FD001 vs FD002", "fig08_dataset_comparison",
     "RUL prediction and anomaly detection results side by side for the two datasets."),
]


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def _data_label(frame: pd.DataFrame) -> list[str]:
    return [CONFIG_SHORT[(s, c)] for s, c in zip(frame["subset"], frame["scheme"])]


def table_datasets() -> pd.DataFrame:
    overview = pd.read_csv(TABLE_DIR / "phase1_dataset_overview.csv")
    conditions = pd.read_csv(TABLE_DIR / "phase2_operating_conditions.csv").groupby("subset").size()
    overview.insert(1, "operating_conditions", overview["subset"].map(conditions))
    overview.insert(2, "fault_mode", "HPC degradation")
    return overview


def table_operating_conditions() -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / "phase2_operating_conditions.csv")


def table_preprocessing() -> pd.DataFrame:
    summary = pd.read_csv(TABLE_DIR / "phase3_preprocessing_summary.csv")
    summary.insert(0, "data", _data_label(summary))
    return summary.drop(columns=["subset", "scheme"])


def table_feature_sets() -> pd.DataFrame:
    sets = pd.read_csv(TABLE_DIR / "phase4_feature_sets.csv")
    sets.insert(0, "data", _data_label(sets))
    sets["representation"] = sets["representation"].map(ev.REPRESENTATION_LABELS)
    return sets.drop(columns=["subset", "scheme"])


def table_model_settings() -> pd.DataFrame:
    """Fixed settings, tuned grid and (for the RUL models) the selected values."""
    rul = json.loads((METRIC_DIR / "phase5_experiment_config.json").read_text())
    anomaly = json.loads((METRIC_DIR / "phase6_experiment_config.json").read_text())
    rows = []
    for model, label in ev.MODEL_LABELS.items():
        rows.append(
            {
                "model": label, "task": "RUL prediction",
                "fixed_settings": ", ".join(f"{k}={v}" for k, v in rul["fixed_params"][model].items() if k != "n_jobs"),
                "tuned_settings": "; ".join(f"{k} in {v}" for k, v in rul["param_grids"][model].items()),
                "selection": "lowest mean RMSE in 5-fold engine-level cross-validation",
            }
        )
    rows.append(
        {
            "model": "Isolation Forest", "task": "Anomaly detection",
            "fixed_settings": ", ".join(f"{k}={v}" for k, v in anomaly["isolation_forest"].items() if k != "n_jobs"),
            "tuned_settings": "none",
            "selection": f"alarm threshold at a {int(100 * anomaly['alarm_budget_main'])}% budget on healthy training cycles",
        }
    )
    return pd.DataFrame(rows)


def table_selected_settings() -> pd.DataFrame:
    metrics, _ = ev.load_rul_results()
    out = metrics[["subset", "scheme", "representation", "model", "params", "cv_rmse_mean", "cv_rmse_std"]].copy()
    out.insert(0, "data", _data_label(out))
    out["representation"] = out["representation"].map(ev.REPRESENTATION_LABELS)
    out["model"] = out["model"].map(ev.MODEL_LABELS)
    out["params"] = [", ".join(f"{k}={v}" for k, v in json.loads(p).items()) for p in out["params"]]
    return out.drop(columns=["subset", "scheme"]).rename(columns={"params": "selected_settings"})


def table_rul_results() -> pd.DataFrame:
    table = ev.rul_results_table().drop(columns=["params"])
    table["data"] = table["data"].str.replace(", ", " ").str.replace(" normalisation", "")
    return table


def table_paired_differences() -> pd.DataFrame:
    return ev.paired_rmse_differences()


def table_anomaly_results() -> pd.DataFrame:
    """Cross-validation and test scores side by side, for both scorings."""
    tidy = ev.anomaly_results_table()
    tidy["data"] = tidy["data"].str.replace(", ", " ").str.replace(" normalisation", "")
    keys = ["data", "representation", "scoring"]
    wide = tidy.pivot_table(index=keys, columns="evaluation", values=["precision", "recall", "f1"], sort=False)
    wide = wide.reindex(columns=[(m, e) for e in ("Cross-validation", "Test") for m in ("precision", "recall", "f1")])
    wide.columns = [f"{'cv' if e == 'Cross-validation' else 'test'}_{m}" for m, e in wide.columns]
    share = tidy.pivot_table(index=keys, columns="evaluation", values="anomalous_share_percent", sort=False)
    wide["cv_anomalous_percent"] = share["Cross-validation"]
    wide["test_anomalous_percent"] = share["Test"]
    return wide.reset_index().sort_values("scoring", kind="stable").reset_index(drop=True)


def table_dataset_comparison(with_intervals: bool = True) -> pd.DataFrame:
    """FD001 against FD002 (both normalisation schemes) on the test engines.

    ``with_intervals`` adds "low"/"high" columns for the capped test RMSE
    (used for the error bars of Figure 8).
    """
    rul, _ = ev.load_rul_results()
    intervals = ev.bootstrap_intervals().set_index(ev.CONFIG_KEYS)
    anomaly, _, _ = ev.load_anomaly_results()
    anomaly = anomaly[anomaly["main_setting"] & (anomaly["evaluation"] == "official test engines")]

    rows = []
    for model in ev.MODEL_ORDER:
        for rep in ("raw", "summary"):
            for column, metric in (
                ("test_rmse", "Test RMSE, capped target"),
                ("test_rmse_uncapped", "Test RMSE, uncapped target"),
                ("test_mae", "Test MAE, capped target"),
                ("test_r2", "Test R2, capped target"),
                ("cv_rmse_mean", "Cross-validated RMSE"),
            ):
                row = {"task": "RUL prediction", "model": ev.MODEL_LABELS[model],
                       "features": ev.REPRESENTATION_LABELS[rep], "metric": metric}
                for subset, scheme in ev.DATA_CONFIGS:
                    value = ev._select(rul, subset, scheme, model=model, representation=rep)[column].iloc[0]
                    row[CONFIG_SHORT[(subset, scheme)]] = value
                    if column == "test_rmse":
                        ci = intervals.loc[(subset, scheme, rep, model)]
                        row[f"{CONFIG_SHORT[(subset, scheme)]} low"] = ci["test_rmse_ci_low"]
                        row[f"{CONFIG_SHORT[(subset, scheme)]} high"] = ci["test_rmse_ci_high"]
                rows.append(row)
    for rep in ("raw", "summary"):
        for scoring, column, metric in (
            ("all cycles", "recall", "Test recall"),
            ("all cycles", "precision", "Test precision, all cycles"),
            ("all cycles", "f1", "Test F1, all cycles"),
            ("healthy vs near failure", "f1", "Test F1, healthy vs near failure"),
        ):
            row = {"task": "Anomaly detection", "model": "Isolation Forest",
                   "features": ev.REPRESENTATION_LABELS[rep], "metric": metric}
            for subset, scheme in ev.DATA_CONFIGS:
                part = ev._select(anomaly, subset, scheme, representation=rep, scoring=scoring)
                row[CONFIG_SHORT[(subset, scheme)]] = part[column].iloc[0]
            rows.append(row)
    table = pd.DataFrame(rows)
    if not with_intervals:
        table = table.drop(columns=[c for c in table.columns if c.endswith((" low", " high"))])
    return table


def table_data_quality() -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / "phase2_data_quality.csv")


def table_sensor_exploration() -> pd.DataFrame:
    """Per-sensor exploration statistics for both subsets side by side (training engines)."""
    fd001 = pd.read_csv(TABLE_DIR / "phase2_sensor_summary_FD001.csv").set_index("column")
    fd002 = pd.read_csv(TABLE_DIR / "phase2_sensor_summary_FD002.csv").set_index("column")
    sensors = [c for c in fd001.index if c.startswith("s") and not c.startswith("setting")]
    return pd.DataFrame(
        {
            "sensor": sensors,
            "symbol": fd001.loc[sensors, "symbol"].to_numpy(),
            "fd001_unique_values": fd001.loc[sensors, "n_unique"].to_numpy(),
            "fd001_spearman": fd001.loc[sensors, "spearman_vs_remaining"].to_numpy(),
            "fd002_unique_values": fd002.loc[sensors, "n_unique"].to_numpy(),
            "fd002_variance_from_condition": fd002.loc[sensors, "variance_from_condition"].to_numpy(),
            "fd002_spearman": fd002.loc[sensors, "spearman_vs_remaining"].to_numpy(),
            "fd002_spearman_within_condition": fd002.loc[sensors, "spearman_within_condition"].to_numpy(),
        }
    )


def table_input_signal() -> pd.DataFrame:
    """Median and maximum absolute rank correlation of each kind of input with capped RUL."""
    signal = pd.read_csv(TABLE_DIR / "phase4_feature_signal.csv")
    signal.insert(0, "data", _data_label(signal))
    signal["statistic"] = signal["statistic"].map(
        {"last": "Last value (raw reading)", "mean": "30-cycle mean", "std": "30-cycle standard deviation"}
    )
    return signal.drop(columns=["subset", "scheme"])


def table_tuning_range() -> pd.DataFrame:
    """Best and worst mean cross-validated RMSE over the six candidates of each configuration."""
    cv = pd.read_csv(METRIC_DIR / "phase5_cv_summary.csv")
    keys = ["subset", "scheme", "representation", "model"]
    out = cv.groupby(keys, sort=False)["cv_rmse_mean"].agg(best="min", worst="max").reset_index()
    out["range"] = out["worst"] - out["best"]
    out.insert(0, "data", _data_label(out))
    out["representation"] = out["representation"].map(ev.REPRESENTATION_LABELS)
    out["model"] = out["model"].map(ev.MODEL_LABELS)
    return out.drop(columns=["subset", "scheme"])


def table_alarm_zones() -> pd.DataFrame:
    """Where the alarms fall on the test engines at the 5% budget."""
    _, _, scores = ev.load_anomaly_results()
    rows = []
    for (subset, scheme, representation), part in scores.groupby(["subset", "scheme", "representation"], sort=False):
        flagged = part[part["flag_05"]]
        share = flagged["zone"].value_counts(normalize=True)
        healthy = part[part["zone"] == "healthy"]
        rows.append(
            {
                "data": CONFIG_SHORT[(subset, scheme)],
                "representation": ev.REPRESENTATION_LABELS[representation],
                "alarms": len(flagged),
                "alarms_healthy_percent": 100 * share.get("healthy", 0.0),
                "alarms_transition_percent": 100 * share.get("transition", 0.0),
                "alarms_near_failure_percent": 100 * share.get("near_failure", 0.0),
                "healthy_cycles_flagged_percent": 100 * healthy["flag_05"].mean(),
            }
        )
    return pd.DataFrame(rows)


def table_budget_sensitivity() -> pd.DataFrame:
    """Precision, recall and F1 on the test engines (all cycles) at alarm budgets of 1, 5 and 10%."""
    full = ev.anomaly_results_table(main_only=False)
    part = full[(full["evaluation"] == "Test") & (full["scoring"] == "all cycles")].copy()
    part["data"] = part["data"].str.replace(", ", " ").str.replace(" normalisation", "")
    wide = part.pivot_table(index=["data", "representation"], columns="alarm_budget",
                            values=["precision", "recall", "f1"], sort=False)
    wide = wide.reindex(columns=[(m, b) for m in ("precision", "recall", "f1") for b in (0.01, 0.05, 0.10)])
    wide.columns = [f"{m}_{int(round(100 * b))}pct" for m, b in wide.columns]
    return wide.reset_index()


def table_above_cap() -> pd.DataFrame:
    """Predictions for test engines whose true RUL is above the 125-cycle cap."""
    _, predictions = ev.load_rul_results()
    above = predictions[predictions["true_rul_uncapped"] > predictions["true_rul"]]
    out = above.groupby(["subset", "scheme", "representation", "model"], sort=False).agg(
        engines_above_cap=("unit", "size"),
        mean_true_rul=("true_rul_uncapped", "mean"),
        mean_predicted_rul=("predicted_rul", "mean"),
        lowest_predicted_rul=("predicted_rul", "min"),
    ).reset_index()
    out.insert(0, "data", _data_label(out))
    out["representation"] = out["representation"].map(ev.REPRESENTATION_LABELS)
    out["model"] = out["model"].map(ev.MODEL_LABELS)
    return out.drop(columns=["subset", "scheme"])


def table_alarm_rate_by_rul() -> pd.DataFrame:
    """Share of cycles flagged (percent) by remaining-life bin, one column per case."""
    _, rates, _ = ev.load_anomaly_results()
    rates = rates.copy()
    rates["case"] = [
        f"{CONFIG_SHORT[(s, c)]}, {r}, {'CV' if e.startswith('cross') else 'test'}"
        for s, c, r, e in zip(rates["subset"], rates["scheme"], rates["representation"], rates["evaluation"])
    ]
    wide = rates.pivot_table(index=["bin_upper", "rul_bin"], columns="case", values="alarm_rate", sort=False)
    wide = (100 * wide).sort_index(level="bin_upper", ascending=False).reset_index().drop(columns="bin_upper")
    wide.columns.name = None
    return wide


def table_design_check_warm_up() -> pd.DataFrame:
    table = pd.read_csv(METRIC_DIR / "phase6_design_check_warm_up.csv")
    table.insert(0, "data", _data_label(table))
    for column in ("flagged_share_before_30", "flagged_share_from_30"):
        table[column.replace("share", "percent")] = 100 * table.pop(column)
    return table.drop(columns=["subset", "scheme"])


def table_design_check_sensors() -> pd.DataFrame:
    table = pd.read_csv(METRIC_DIR / "phase6_design_check_sensors.csv")
    table.insert(0, "data", _data_label(table))
    table["representation"] = table["representation"].map(ev.REPRESENTATION_LABELS)
    return table.drop(columns=["subset", "scheme"])


THESIS_TABLES = [
    ("table01_datasets", "Datasets used: engines, rows and trajectory lengths.", table_datasets),
    ("table02_operating_conditions", "Operating conditions in the training data (altitude in thousand feet / Mach number / throttle resolver angle).", table_operating_conditions),
    ("table03_preprocessing", "Preprocessing outcome per data configuration.", table_preprocessing),
    ("table04_feature_sets", "Size of the feature sets.", table_feature_sets),
    ("table05_model_settings", "Model settings.", table_model_settings),
    ("table06_selected_settings", "Settings selected by cross-validation for the RUL models, with mean and standard deviation of the cross-validated RMSE.", table_selected_settings),
    ("table07_rul_results", "RUL prediction results on the official test engines. Main result: target capped at 125. Interval: 95% bootstrap over test engines.", table_rul_results),
    ("table08_paired_differences", "Paired differences in test RMSE (B minus A) with 95% bootstrap intervals.", table_paired_differences),
    ("table09_anomaly_results", "Isolation Forest results at the 5% alarm budget, in cross-validation and on the official test engines.", table_anomaly_results),
    ("table10_dataset_comparison", "FD001 compared with FD002 on the official test engines (intervals for the capped RMSE are in Table 7).",
     lambda: table_dataset_comparison(with_intervals=False)),
    ("table11_data_quality", "Data quality checks on the four files.", table_data_quality),
    ("table12_sensor_exploration", "Exploration statistics per sensor (training engines). Spearman: rank correlation with the number of cycles remaining.", table_sensor_exploration),
    ("table13_input_signal", "Absolute rank correlation of each kind of model input with the capped RUL (training engines).", table_input_signal),
    ("table14_tuning_range", "Best and worst mean cross-validated RMSE over the six candidate settings.", table_tuning_range),
    ("table15_alarm_zones", "Distribution of Isolation Forest alarms on the test engines over the life zones (5% alarm budget).", table_alarm_zones),
    ("table16_budget_sensitivity", "Isolation Forest results on the test engines (all cycles) at alarm budgets of 1%, 5% and 10%.", table_budget_sensitivity),
    ("table17_design_check_warm_up", "Share of healthy cycles flagged by the summary-statistic detector without the warm-up rule (cross-validation on training engines).", table_design_check_warm_up),
    ("table18_design_check_sensors", "Isolation Forest cross-validation scores with and without the quantised sensors (training engines, all cycles, 5% budget).", table_design_check_sensors),
    ("table19_above_cap", "Predictions for test engines whose true RUL is above the 125-cycle cap.", table_above_cap),
    ("table20_alarm_rate_by_rul", "Share of cycles flagged by the Isolation Forest (percent) by remaining useful life; CV: cross-validation on training engines.", table_alarm_rate_by_rul),
]


# --------------------------------------------------------------------------
# Figure 8
# --------------------------------------------------------------------------

def plot_dataset_comparison():
    """Figure 8 - FD001 vs FD002 for RUL prediction and anomaly detection."""
    ps.apply_style()
    table = table_dataset_comparison()
    bar_style = {
        "FD001": {"color": ps.COLORS["FD001"], "label": "FD001 (one operating condition)"},
        "FD002 dataset-wide": {
            "facecolor": "#f7c3ad", "edgecolor": ps.COLORS["FD002"], "hatch": "///", "linewidth": 0,
            "label": "FD002 (six conditions), dataset-wide normalisation",
        },
        "FD002 per-condition": {"color": ps.COLORS["FD002"], "label": "FD002 (six conditions), per-condition normalisation"},
    }
    panels = [
        ("RUL prediction", "Test RMSE, capped target", "RUL prediction: test RMSE, target capped at 125\n(cycles, lower is better)", ".1f"),
        ("RUL prediction", "Test RMSE, uncapped target", "RUL prediction: test RMSE against uncapped NASA values\n(cycles, lower is better)", ".1f"),
        ("Anomaly detection", "Test recall", "Anomaly detection: recall on test engines\n(higher is better)", ".2f"),
        ("Anomaly detection", "Test F1, all cycles", "Anomaly detection: F1 on test engines, all cycles\n(higher is better)", ".2f"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
    width = 0.27

    for ax, (task, metric, title, fmt) in zip(axes.ravel(), panels):
        part = table[(table["task"] == task) & (table["metric"] == metric)].reset_index(drop=True)
        x = np.arange(len(part))
        for i, (config, style) in enumerate(bar_style.items()):
            positions = x + (i - 1) * width
            values = part[config].to_numpy(float)
            ax.bar(positions, values, width * 0.9, **style)
            tops = values
            if f"{config} low" in part and part[f"{config} low"].notna().all():
                low, high = part[f"{config} low"].to_numpy(float), part[f"{config} high"].to_numpy(float)
                ax.errorbar(positions, values, yerr=[values - low, high - values], fmt="none",
                            ecolor=ps.INK, elinewidth=0.9, capsize=2.5)
                tops = high
            for position, value, top in zip(positions, values, tops):
                ax.annotate(format(value, fmt), (position, top), xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5, color=ps.INK_SOFT)
        if task == "RUL prediction":
            labels = [f"{m}\n{f.lower()}" for m, f in zip(part["model"], part["features"])]
        else:
            labels = [f"Isolation Forest\n{f.lower()}" for f in part["features"]]
        ax.set_xticks(x, labels)
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("FD001 vs FD002: performance on the official test engines")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, fontsize=8.5)
    fig.text(0.5, 0.005, "Error bars (top left): 95% bootstrap interval over test engines. "
             "Anomaly detection at the 5% alarm budget.", ha="center", fontsize=8, color=ps.INK_SOFT)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    return fig


# --------------------------------------------------------------------------
# Research framework diagram (Chapter 3)
# --------------------------------------------------------------------------

def plot_research_framework():
    """Figure for Chapter 3 - the stages of the study from data to research questions."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    ps.apply_style()
    fig, ax = plt.subplots(figsize=(8.6, 9.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(-4, 110)
    ax.axis("off")

    def box(x, y, w, h, title, lines, fill="white"):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", linewidth=1.0,
            edgecolor=ps.INK_SOFT, facecolor=fill,
        ))
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=10, fontweight="bold", color=ps.INK)
        ax.text(x + w / 2, y + h - 6.6, "\n".join(lines), ha="center", va="top", fontsize=8.6,
                color=ps.INK_SOFT, linespacing=1.45)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                     linewidth=1.1, color=ps.INK_SOFT, shrinkA=0, shrinkB=0))

    tint = "#f4f3ef"
    box(10, 95, 80, 13, "1. Data", [
        "NASA C-MAPSS turbofan degradation data",
        "FD001: one operating condition      FD002: six operating conditions",
    ], fill=tint)
    box(10, 77, 80, 13, "2. Data exploration", [
        "Missing values and duplicates, operating conditions,",
        "sensor behaviour over engine life (training engines only)",
    ])
    box(10, 55, 80, 17, "3. Preprocessing", [
        "RUL labels: linear, capped at 125 cycles",
        "Removal of constant sensors",
        "Normalisation: dataset-wide  |  within each operating condition",
    ])
    box(10, 37, 80, 13, "4. Feature representation", [
        "A: raw sensor readings of each cycle",
        "B: last value, mean and standard deviation over the last 30 cycles",
    ])
    box(1, 13, 47, 19, "5a. RUL prediction (supervised)", [
        "Random Forest, XGBoost",
        "5-fold engine-level cross-validation",
        "Official test engines",
        "MAE, RMSE, R²",
    ])
    box(52, 13, 47, 19, "5b. Anomaly detection (unsupervised)", [
        "Isolation Forest fitted on healthy cycles",
        "Alarm budget: 5% of healthy cycles",
        "Reference label: RUL ≤ 30",
        "Precision, recall, F1",
    ])
    box(10, -3, 80, 12, "6. Analysis", [
        "Experiments 1 to 3  →  RQ1 (accuracy) and RQ2 (preprocessing and features)",
    ], fill=tint)

    for y_from, y_to in ((95, 90), (77, 72), (55, 50)):
        arrow(50, y_from, 50, y_to)
    arrow(34, 37, 26, 32)
    arrow(66, 37, 74, 32)
    arrow(26, 13, 34, 9)
    arrow(74, 13, 66, 9)
    ax.text(95, 63.5, "All statistics are fitted on training engines only", rotation=90, ha="center",
            va="center", fontsize=8.2, color=ps.INK_SOFT, style="italic")
    return fig


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _format(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return ""
        return f"{value:.3f}" if abs(value) <= 1 else f"{value:.2f}"
    return str(value).replace("|", "/")


def to_markdown(table: pd.DataFrame) -> str:
    """A plain Markdown table (no extra package needed)."""
    header = "| " + " | ".join(str(c).replace("_", " ") for c in table.columns) + " |"
    rule = "|" + "|".join("---" for _ in table.columns) + "|"
    body = ["| " + " | ".join(_format(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def figure_inventory() -> pd.DataFrame:
    """Every required visualisation and whether its files exist."""
    rows = [
        {
            "required_visualisation": requirement, "file": f"results/figures/{name}.png",
            "png": (ps.FIGURE_DIR / f"{name}.png").exists(),
            "pdf": (ps.FIGURE_DIR / f"{name}.pdf").exists(),
            "content": description,
        }
        for requirement, name, description in FIGURES
    ]
    return pd.DataFrame(rows)


def run_all() -> dict:
    """Write the thesis tables, Figure 8 and the Chapter 4 reference document."""
    THESIS_DIR.mkdir(parents=True, exist_ok=True)
    saved = {}

    for name, plot in (("fig08_dataset_comparison", plot_dataset_comparison),
                       ("fig00_research_framework", plot_research_framework)):
        fig = plot()
        saved[name] = ps.save_figure(fig, name)
        plt.close(fig)

    sections = [
        "# Chapter 4 - Tables and Figures",
        "",
        "Generated by `python -m src.results_summary` from the saved result files. Do not edit the numbers by "
        "hand; rerun the command instead.",
        "",
        "## Tables",
    ]
    for number, (name, caption, builder) in enumerate(THESIS_TABLES, start=1):
        table = builder()
        path = THESIS_DIR / f"{name}.csv"
        table.to_csv(path, index=False, float_format="%.6g")
        saved[name] = path
        sections += ["", f"### Table {number}. {caption}", "", f"Source file: `results/tables/thesis/{name}.csv`", "",
                     to_markdown(table)]

    inventory = figure_inventory()
    sections += ["", "## Figures", "", "Each figure is saved as PNG (300 dpi) and PDF in `results/figures/`.", "",
                 to_markdown(inventory[["required_visualisation", "file", "content"]])]
    REPORT_PATH.write_text("\n".join(sections) + "\n", encoding="utf-8")
    saved["chapter4_tables_and_figures"] = REPORT_PATH
    return saved


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    for name, path in run_all().items():
        print(f"saved {name}: {path.relative_to(ROOT)}")
    missing = figure_inventory().query("not png or not pdf")
    print("all required figures present" if missing.empty else f"MISSING FIGURES:\n{missing}")
