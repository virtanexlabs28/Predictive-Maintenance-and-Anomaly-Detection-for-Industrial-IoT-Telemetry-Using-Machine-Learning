"""Phase 3 - preprocessing and RUL label construction.

Steps, in order:

1. Build RUL labels (piecewise linear, capped at 125 cycles).
2. Remove sensors that carry no information, decided from TRAINING data only.
3. Standardise the remaining sensors (z-score) with statistics fitted on
   TRAINING data only.

Two normalisation schemes are provided so their effect can be compared (RQ2):

* ``"global"``    - one mean and standard deviation per sensor for the whole
  training file ("ordinary" normalisation).
* ``"condition"`` - one mean and standard deviation per sensor for each
  operating condition.

FD001 has a single operating condition, so both schemes give identical
numbers there. The comparison is only meaningful on FD002.

Leakage safeguards: the rule for dropping sensors and all scaling statistics
are computed from training engines. Test engines are only transformed.
``SensorScaler`` has separate ``fit`` and ``transform`` steps so that
cross-validation in Phase 5 can refit it inside every training fold.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import data_loader as dl
from . import plot_style as ps

PROCESSED_DIR = dl.PROJECT_ROOT / "data" / "processed"
TABLE_DIR = dl.PROJECT_ROOT / "results" / "tables"
METRIC_DIR = dl.PROJECT_ROOT / "results" / "metrics"

RUL_CAP = 125
SCHEMES = ("global", "condition")

# (subset, scheme) combinations used in the thesis. FD001 + "condition" is
# left out because it is numerically identical to FD001 + "global".
EXPERIMENT_GRID = [("FD001", "global"), ("FD002", "global"), ("FD002", "condition")]

ID_COLS = ["unit", "cycle", "condition"]
TARGET_COLS = ["rul_uncapped", "rul"]


# --------------------------------------------------------------------------
# Operating conditions
# --------------------------------------------------------------------------

def condition_key(df: pd.DataFrame) -> pd.Series:
    """Label each row with its operating condition.

    The three settings (altitude, Mach number, throttle resolver angle) take a
    small number of nominal values with tiny noise added. Rounding removes the
    noise, so rows flown at the same regime get the same label. The rule uses
    fixed rounding only - nothing is learned from the data - so applying it
    to test engines cannot leak information.
    """
    alt = df["setting_1"].round(0).astype(int).astype(str)
    # "+ 0.0" turns the float -0.0 into 0.0 so both print as "0.00"
    mach = (df["setting_2"].round(2) + 0.0).map("{:.2f}".format)
    tra = df["setting_3"].round(0).astype(int).astype(str)
    return alt + " / " + mach + " / " + tra


def _group_key(df: pd.DataFrame, scheme: str) -> pd.Series:
    """The group each row is normalised with: everything, or its condition."""
    if scheme == "global":
        return pd.Series("all", index=df.index)
    if scheme == "condition":
        return condition_key(df)
    raise ValueError(f"Unknown scheme {scheme!r}. Use one of {SCHEMES}.")


# --------------------------------------------------------------------------
# RUL labels
# --------------------------------------------------------------------------

def add_train_rul(train: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """Add RUL labels to run-to-failure training engines.

    ``rul_uncapped`` = cycles left until the engine's last recorded cycle
    (0 on the last row). ``rul`` = the same value limited to ``cap``.
    """
    out = train.copy()
    out["rul_uncapped"] = out.groupby("unit")["cycle"].transform("max") - out["cycle"]
    out["rul"] = out["rul_uncapped"].clip(upper=cap)
    return out


def add_test_rul(test: pd.DataFrame, rul_table: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """Add RUL labels to test engines.

    The NASA RUL file gives the true RUL at each test engine's LAST recorded
    cycle. Earlier rows are that value plus the cycles still to be observed.
    """
    out = test.copy()
    last_cycle = out.groupby("unit")["cycle"].transform("max")
    rul_at_last = out["unit"].map(rul_table.set_index("unit")["rul"])
    out["rul_uncapped"] = (last_cycle - out["cycle"]) + rul_at_last
    out["rul"] = out["rul_uncapped"].clip(upper=cap)
    return out


# --------------------------------------------------------------------------
# Sensor removal and scaling
# --------------------------------------------------------------------------

def find_constant_sensors(train: pd.DataFrame, scheme: str = "global") -> list[str]:
    """Sensors that would be constant after normalisation (training data only).

    * ``"global"``: the sensor has one unique value in the whole training file.
    * ``"condition"``: the sensor has one unique value inside EVERY operating
      condition, so after per-condition scaling it would be 0 everywhere.
    """
    n_unique = train.groupby(_group_key(train, scheme))[dl.SENSOR_COLS].nunique()
    return [c for c in dl.SENSOR_COLS if (n_unique[c] == 1).all()]


class SensorScaler:
    """Z-score scaler with statistics fitted per group on training data.

    ``scheme="global"`` uses one group (the whole file); ``scheme="condition"``
    uses one group per operating condition.
    """

    def __init__(self, scheme: str = "global"):
        if scheme not in SCHEMES:
            raise ValueError(f"Unknown scheme {scheme!r}. Use one of {SCHEMES}.")
        self.scheme = scheme

    def fit(self, train: pd.DataFrame, sensors: list[str]) -> "SensorScaler":
        grouped = train.groupby(_group_key(train, self.scheme))[list(sensors)]
        self.sensors_ = list(sensors)
        self.mean_ = grouped.mean()
        self.std_ = grouped.std(ddof=0)
        # A sensor that never changes inside a group has a true spread of 0,
        # but floating-point arithmetic can report something like 5e-15.
        # Dividing by that would turn rounding noise into large values.
        self.std_[grouped.nunique() == 1] = 0.0
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        key = _group_key(df, self.scheme)
        unseen = sorted(set(key.unique()) - set(self.mean_.index))
        if unseen:
            raise ValueError(f"Operating conditions not present in the training data: {unseen}")
        mean = self.mean_.loc[key].to_numpy()
        std = self.std_.loc[key].to_numpy()
        values = df[self.sensors_].to_numpy(dtype=float)
        scaled = np.zeros_like(values)  # stays 0 where the group's spread is 0
        np.divide(values - mean, std, out=scaled, where=std > 0)
        out = df.copy()
        out[self.sensors_] = scaled
        return out

    def parameters(self) -> pd.DataFrame:
        """Fitted means and standard deviations as a tidy table."""
        mean = self.mean_.rename_axis("group").reset_index().melt("group", var_name="sensor", value_name="mean")
        std = self.std_.rename_axis("group").reset_index().melt("group", var_name="sensor", value_name="std")
        return mean.merge(std, on=["group", "sensor"])


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------

def preprocess(subset: str, scheme: str = "global", cap: int = RUL_CAP) -> dict:
    """Run the whole preprocessing pipeline for one subset and scheme.

    Returns a dict with:

    * ``train`` / ``test`` - tables with unit, cycle, condition, the scaled
      sensors, ``rul_uncapped`` and ``rul``
    * ``sensors``  - names of the retained sensor columns (the model inputs)
    * ``dropped``  - sensors removed as constant
    * ``scaler``   - the fitted ``SensorScaler``
    """
    data = dl.load_subset(subset)
    fitted = fit_and_apply(
        add_train_rul(data["train"], cap),
        {"test": add_test_rul(data["test"], data["rul"], cap)},
        scheme,
    )
    return {"subset": subset, "scheme": scheme, **fitted}


def fit_and_apply(fit_rows: pd.DataFrame, other: dict[str, pd.DataFrame], scheme: str = "global") -> dict:
    """Learn the preprocessing from ``fit_rows`` and apply it to every table.

    ``fit_rows`` are labelled raw rows of the engines the model may learn
    from (all training engines, or the training folds during
    cross-validation). ``other`` holds labelled raw rows of engines that must
    stay unseen (test engines, or the validation fold). The sensors to drop
    and the scaling statistics come from ``fit_rows`` only.

    Returns ``train`` (the transformed ``fit_rows``), one entry per key of
    ``other``, plus ``sensors``, ``dropped`` and ``scaler``.
    """
    dropped = find_constant_sensors(fit_rows, scheme)
    sensors = [c for c in dl.SENSOR_COLS if c not in dropped]
    scaler = SensorScaler(scheme).fit(fit_rows, sensors)

    result = {"sensors": sensors, "dropped": dropped, "scaler": scaler}
    for name, df in {"train": fit_rows, **other}.items():
        out = scaler.transform(df)
        out["condition"] = condition_key(df)
        result[name] = out[ID_COLS + sensors + TARGET_COLS].reset_index(drop=True)
    return result


def label_summary(result: dict) -> dict:
    """Row counts and how much of the data sits on the RUL cap."""
    train, test = result["train"], result["test"]
    last_test = test.groupby("unit").tail(1)
    return {
        "subset": result["subset"],
        "scheme": result["scheme"],
        "sensors_retained": len(result["sensors"]),
        "sensors_dropped": ", ".join(result["dropped"]) or "none",
        "train_rows": len(train),
        "train_rows_at_cap_percent": round(100 * (train["rul_uncapped"] >= RUL_CAP).mean(), 1),
        "test_rows": len(test),
        "test_engines": test["unit"].nunique(),
        "test_engines_true_rul_above_cap": int((last_test["rul_uncapped"] > RUL_CAP).sum()),
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def _engines_spread_by_lifetime(train: pd.DataFrame, n: int) -> list[int]:
    lifetimes = train.groupby("unit")["cycle"].max().sort_values(kind="stable")
    positions = np.linspace(0, len(lifetimes) - 1, n).round().astype(int)
    return lifetimes.index[positions].tolist()


def plot_rul_target(n_engines: int = 3):
    """Figure 3a - the RUL target over the life of a few training engines."""
    ps.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, subset in zip(axes, dl.SUBSETS):
        train = add_train_rul(dl.load_train(subset))
        for i, unit in enumerate(_engines_spread_by_lifetime(train, n_engines)):
            engine = train[train["unit"] == unit]
            ax.plot(
                engine["cycle"], engine["rul_uncapped"], color=ps.MUTED, linewidth=1.0,
                linestyle="--", label="Linear RUL (uncapped)" if i == 0 else None,
            )
            ax.plot(
                engine["cycle"], engine["rul"], color=ps.COLORS[subset], linewidth=2.0,
                label=f"Piecewise linear target (capped at {RUL_CAP})" if i == 0 else None,
            )
            ax.annotate(
                f"engine {unit}", xy=(engine["cycle"].iloc[-1], 0), xytext=(0, -5),
                textcoords="offset points", ha="center", va="top", fontsize=7.5, color=ps.INK_SOFT,
            )
        ax.set_ylim(bottom=-32)  # room for the engine labels under the failure points
        ax.set_title(f"{subset}: shortest-, median- and longest-lived training engine")
        ax.set_xlabel("Operating cycle")
        ax.legend(loc="upper right")
    axes[0].set_ylabel("Remaining useful life (cycles)")
    fig.suptitle("RUL label construction: linear degradation capped at 125 cycles")
    fig.tight_layout()
    return fig


def plot_normalisation_effect(unit: int | None = None, sensors: tuple[str, ...] = ("s4", "s11")):
    """Figure 3b - one FD002 engine under the two normalisation schemes."""
    ps.apply_style()
    subset = "FD002"
    results = {scheme: preprocess(subset, scheme) for scheme in SCHEMES}
    if unit is None:  # median-lifetime training engine, as in Figure 2b
        lifetimes = results["global"]["train"].groupby("unit")["cycle"].max().sort_values(kind="stable")
        unit = int(lifetimes.index[len(lifetimes) // 2])
    titles = {
        "global": "Dataset-wide normalisation",
        "condition": "Normalisation within each operating condition",
    }

    fig, axes = plt.subplots(len(sensors), 2, figsize=(10, 2.5 * len(sensors) + 0.9), sharex=True)
    for c, scheme in enumerate(SCHEMES):
        train = results[scheme]["train"]
        engine = train[train["unit"] == unit]
        for r, col in enumerate(sensors):
            ax = axes[r, c]
            ax.plot(
                engine["cycle"], engine[col], color=ps.COLORS[subset],
                linewidth=0.6, marker="o", markersize=2.2,
            )
            symbol, description, _ = dl.SENSOR_INFO[col]
            ax.set_title(f"{col} - {description}", fontsize=9, loc="left")
            ax.set_ylabel(f"{symbol} (z-score)")
            if r == 0:
                ax.text(
                    0.0, 1.22, titles[scheme], transform=ax.transAxes,
                    fontsize=10, fontweight="bold", color=ps.INK,
                )
        axes[-1, c].set_xlabel("Operating cycle")
    fig.suptitle(f"Effect of the normalisation scheme on FD002 sensor readings (training engine {unit})", y=0.995)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Run everything
# --------------------------------------------------------------------------

def run_all() -> dict:
    """Write processed data, parameter tables, the config record and figures."""
    for folder in (PROCESSED_DIR, TABLE_DIR, METRIC_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    saved = {}
    summaries, parameters, config_runs = [], [], []

    for subset, scheme in EXPERIMENT_GRID:
        result = preprocess(subset, scheme)
        for split in ("train", "test"):
            path = PROCESSED_DIR / f"{subset}_{scheme}_{split}.csv"
            result[split].to_csv(path, index=False, float_format="%.6f")
            saved[path.stem] = path
        summaries.append(label_summary(result))
        params = result["scaler"].parameters()
        params.insert(0, "scheme", scheme)
        params.insert(0, "subset", subset)
        parameters.append(params)
        config_runs.append(
            {
                "subset": subset,
                "scheme": scheme,
                "sensors_retained": result["sensors"],
                "sensors_dropped": result["dropped"],
                "operating_conditions": sorted(result["train"]["condition"].unique()),
            }
        )

    summary_path = TABLE_DIR / "phase3_preprocessing_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    saved["phase3_preprocessing_summary"] = summary_path

    params_path = TABLE_DIR / "phase3_scaler_parameters.csv"
    pd.concat(parameters, ignore_index=True).to_csv(params_path, index=False, float_format="%.8g")
    saved["phase3_scaler_parameters"] = params_path

    config = {
        "rul_cap": RUL_CAP,
        "rul_definition": "train: last cycle of unit - cycle; test: (last observed cycle - cycle) + NASA RUL value; both clipped at rul_cap",
        "sensor_removal_rule": "one unique value in the training data within every normalisation group",
        "scaling": "z-score, population standard deviation, fitted on training engines only",
        "condition_rule": "round setting_1 to 0 dp, setting_2 to 2 dp, setting_3 to 0 dp",
        "runs": config_runs,
    }
    config_path = METRIC_DIR / "phase3_preprocessing_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    saved["phase3_preprocessing_config"] = config_path

    figures = {
        "fig03a_rul_target": plot_rul_target(),
        "fig03b_normalisation_effect_FD002": plot_normalisation_effect(),
    }
    for name, fig in figures.items():
        saved[name] = ps.save_figure(fig, name)
        plt.close(fig)
    return saved


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    for name, path in run_all().items():
        print(f"saved {name}: {path.relative_to(dl.PROJECT_ROOT)}")
