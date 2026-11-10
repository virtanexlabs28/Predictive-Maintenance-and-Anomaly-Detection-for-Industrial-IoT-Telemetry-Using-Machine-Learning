"""Phase 2 - data exploration for C-MAPSS FD001 and FD002.

Nothing here changes the data. The functions only describe it:

* ``quality_report``        - missing values and duplicates in every file
* ``operating_conditions``  - the flight regimes found in the settings columns
* ``sensor_summary``        - per-sensor statistics from the TRAINING engines
* ``plot_*``                - Figures 1, 2a and 2b of the thesis

Sensor statistics use training engines only. The test files are inspected
only for structure (size, missing values, trajectory length), so no sensor
information from held-out engines influences later modelling decisions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from . import data_loader as dl
from . import plot_style as ps
from .preprocessing import condition_key  # the rounding rule lives with the preprocessing code

TABLE_DIR = dl.PROJECT_ROOT / "results" / "tables"

# Sensors shown in Figure 2b. They sit at different points of the gas path
# (LPC outlet, LPT outlet, HPC outlet pressures) and all drift in FD001.
CONDITION_EFFECT_SENSORS = ["s2", "s4", "s7", "s11"]


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def quality_report() -> pd.DataFrame:
    """Missing values, non-finite values and duplicates for all four files."""
    rows = []
    for subset in dl.SUBSETS:
        data = dl.load_subset(subset)
        for split in ("train", "test"):
            df = data[split]
            values = df[dl.SETTING_COLS + dl.SENSOR_COLS]
            rows.append(
                {
                    "subset": subset,
                    "split": split,
                    "rows": len(df),
                    "missing_values": int(df.isna().sum().sum()),
                    "non_finite_values": int((~np.isfinite(values.to_numpy())).sum()),
                    "duplicate_rows": int(df.duplicated().sum()),
                    "duplicate_unit_cycle": int(df.duplicated(subset=dl.INDEX_COLS).sum()),
                    # same settings and sensor readings appearing on two different rows
                    "repeated_measurement_rows": int(values.duplicated().sum()),
                }
            )
    return pd.DataFrame(rows)


def operating_conditions(subset: str) -> pd.DataFrame:
    """Operating conditions in the training file and how often each occurs."""
    train = dl.load_train(subset)
    counts = condition_key(train).value_counts().sort_index()
    out = counts.rename_axis("altitude_kft / mach / throttle").reset_index(name="rows")
    out.insert(0, "subset", subset)
    out["share_percent"] = (100 * out["rows"] / len(train)).round(1)
    return out


def sensor_summary(subset: str) -> pd.DataFrame:
    """Describe every setting and sensor column of the training engines.

    Columns of the result:

    * ``n_unique`` / ``std``    - a column with one unique value carries no information
    * ``constant_within_condition`` - the column never changes inside any one
      operating condition, so it only repeats which condition the engine is in
    * ``spearman_vs_remaining`` - rank correlation with the number of cycles left
      before failure. Values near +1 or -1 mean the sensor moves steadily as
      the engine wears out; values near 0 mean no monotonic trend is visible.
    * ``variance_from_condition`` - share of the column's variance explained by
      the operating condition alone (0 = none, 1 = all of it)
    * ``spearman_within_condition`` - the same rank correlation after
      subtracting each operating condition's mean from the sensor
    """
    train = dl.load_train(subset)
    remaining = train.groupby("unit")["cycle"].transform("max") - train["cycle"]
    cond = condition_key(train)

    rows = []
    for col in dl.SETTING_COLS + dl.SENSOR_COLS:
        x = train[col]
        symbol, description, unit = dl.SENSOR_INFO.get(col, ("", "Operational setting", ""))
        row = {
            "subset": subset,
            "column": col,
            "symbol": symbol,
            "description": description,
            "unit": unit,
            "mean": x.mean(),
            "std": x.std() if x.nunique() > 1 else 0.0,
            "min": x.min(),
            "max": x.max(),
            "n_unique": x.nunique(),
            "constant": x.nunique() == 1,
            # one unique value inside every operating condition
            "constant_within_condition": bool(x.groupby(cond).nunique().max() == 1),
            "spearman_vs_remaining": np.nan,
            "variance_from_condition": np.nan,
            "spearman_within_condition": np.nan,
        }
        if not row["constant"]:
            row["spearman_vs_remaining"] = x.corr(remaining, method="spearman")
            centred = x - x.groupby(cond).transform("mean")
            share = 1 - centred.var(ddof=0) / x.var(ddof=0)
            row["variance_from_condition"] = round(float(share), 6) + 0.0  # hides float noise such as -2e-16
            if not row["constant_within_condition"]:
                row["spearman_within_condition"] = centred.corr(remaining, method="spearman")
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def _engines_spread_by_lifetime(train: pd.DataFrame, n: int) -> list[int]:
    """Pick n engines evenly from shortest-lived to longest-lived (deterministic)."""
    lifetimes = train.groupby("unit")["cycle"].max().sort_values(kind="stable")
    positions = np.linspace(0, len(lifetimes) - 1, n).round().astype(int)
    return lifetimes.index[positions].tolist()


def _median_lifetime_engine(train: pd.DataFrame) -> int:
    lifetimes = train.groupby("unit")["cycle"].max().sort_values(kind="stable")
    return int(lifetimes.index[len(lifetimes) // 2])


def plot_dataset_distribution():
    """Figure 1 - how long engines run in the training and test files."""
    ps.apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2), sharex="col")
    column_titles = [
        "Training engines:\ncycles until failure",
        "Test engines:\ncycles observed",
        "Test engines:\ntrue RUL at last observed cycle",
    ]
    bins = [np.arange(0, 401, 20), np.arange(0, 401, 20), np.arange(0, 201, 10)]

    for r, subset in enumerate(dl.SUBSETS):
        data = dl.load_subset(subset)
        series = [
            data["train"].groupby("unit")["cycle"].max(),
            data["test"].groupby("unit")["cycle"].max(),
            data["rul"]["rul"],
        ]
        for c, values in enumerate(series):
            ax = axes[r, c]
            ax.hist(values, bins=bins[c], color=ps.COLORS[subset], edgecolor="white", linewidth=0.8)
            ax.grid(axis="x", visible=False)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))  # counts are whole engines
            ax.text(
                0.97, 0.93,
                f"{subset}\nn = {len(values)} engines\nmedian = {values.median():.0f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=ps.INK_SOFT,
            )
            if r == 0:
                ax.set_title(column_titles[c])
            if r == 1:
                ax.set_xlabel("Remaining useful life (cycles)" if c == 2 else "Operating cycles")
            if c == 0:
                ax.set_ylabel("Number of engines")

    fig.suptitle("Trajectory lengths and test-set RUL in C-MAPSS FD001 and FD002")
    fig.tight_layout()
    return fig


def plot_sensor_trends(subset: str = "FD001", n_engines: int = 5):
    """Figure 2a - all 21 sensors over the life of a few training engines."""
    ps.apply_style()
    train = dl.load_train(subset)
    units = _engines_spread_by_lifetime(train, n_engines)
    constant = {c for c in dl.SENSOR_COLS if train[c].nunique() == 1}

    fig, axes = plt.subplots(7, 3, figsize=(10, 13), sharex=True)
    for ax, col in zip(axes.ravel(), dl.SENSOR_COLS):
        for unit in units:
            engine = train[train["unit"] == unit]
            ax.plot(engine["cycle"], engine[col], color=ps.COLORS[subset], linewidth=0.7, alpha=0.75)
        symbol, _, unit_label = dl.SENSOR_INFO[col]
        title = f"{col} - {symbol}" + (f" ({unit_label})" if unit_label != "-" else "")
        if col in constant:
            title += "  [constant]"
            centre = train[col].iloc[0]
            ax.set_ylim(centre - 1, centre + 1)
        ax.set_title(title, fontsize=9, loc="left")
        ax.ticklabel_format(axis="y", useOffset=False)
    for ax in axes[-1]:
        ax.set_xlabel("Operating cycle")
    for ax in axes[:, 0]:
        ax.set_ylabel("Sensor value")

    fig.suptitle(
        f"Sensor measurements over operating cycles for {n_engines} training engines ({subset})",
        y=0.995,
    )
    fig.text(
        0.5, 0.975,
        "Each line is one engine, from its first cycle until failure. "
        "Engines were chosen from the shortest-lived to the longest-lived.",
        ha="center", va="top", fontsize=8.5, color=ps.INK_SOFT,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_condition_effect(sensors: list[str] | None = None):
    """Figure 2b - the same sensors in FD001 (one regime) and FD002 (six regimes)."""
    ps.apply_style()
    sensors = sensors or CONDITION_EFFECT_SENSORS
    fig, axes = plt.subplots(len(sensors), 2, figsize=(10, 2.1 * len(sensors) + 0.8), sharex="col")

    for c, subset in enumerate(dl.SUBSETS):
        train = dl.load_train(subset)
        unit = _median_lifetime_engine(train)
        engine = train[train["unit"] == unit]
        for r, col in enumerate(sensors):
            ax = axes[r, c]
            ax.plot(
                engine["cycle"], engine[col], color=ps.COLORS[subset],
                linewidth=0.6, marker="o", markersize=2.2, alpha=0.9,
            )
            symbol, description, unit_label = dl.SENSOR_INFO[col]
            ax.set_title(f"{col} - {description}", fontsize=9, loc="left")
            ax.set_ylabel(f"{symbol} ({unit_label})")
            ax.ticklabel_format(axis="y", useOffset=False)
            if r == 0:
                n_cond = condition_key(train).nunique()
                label = "one operating condition" if n_cond == 1 else f"{n_cond} operating conditions"
                ax.text(
                    0.0, 1.26, f"{subset}: {label} (training engine {unit})",
                    transform=ax.transAxes, fontsize=10, fontweight="bold", color=ps.INK,
                )
        axes[-1, c].set_xlabel("Operating cycle")

    fig.suptitle("Effect of operating conditions on raw sensor readings: FD001 vs FD002", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 1.0))
    return fig


# --------------------------------------------------------------------------
# Run everything
# --------------------------------------------------------------------------

def run_all() -> dict[str, Path]:
    """Create every Phase 2 table and figure and return where each was saved."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    def save_table(df: pd.DataFrame, name: str) -> None:
        path = TABLE_DIR / f"{name}.csv"
        df.to_csv(path, index=False, float_format="%.6g")
        saved[name] = path

    save_table(quality_report(), "phase2_data_quality")
    save_table(
        pd.concat([operating_conditions(s) for s in dl.SUBSETS], ignore_index=True),
        "phase2_operating_conditions",
    )
    for subset in dl.SUBSETS:
        save_table(sensor_summary(subset), f"phase2_sensor_summary_{subset}")

    figures = {
        "fig01_dataset_distribution": plot_dataset_distribution(),
        "fig02a_sensor_trends_FD001": plot_sensor_trends("FD001"),
        "fig02b_operating_condition_effect": plot_condition_effect(),
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
