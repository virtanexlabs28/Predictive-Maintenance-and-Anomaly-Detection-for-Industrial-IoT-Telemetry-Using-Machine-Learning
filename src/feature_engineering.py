"""Phase 4 - the two feature representations compared under RQ2.

Both representations have ONE ROW PER ENGINE CYCLE, so the models see exactly
the same samples and targets. Only the input columns differ.

* ``"raw"``     - Representation A: the scaled sensor readings of that cycle.
* ``"summary"`` - Representation B: for every sensor, three statistics over
  the engine's most recent ``WINDOW`` cycles (the current cycle included):
  the last value, the mean and the standard deviation.

Rules that keep the summary features honest:

* Windows look backwards only. A row never uses readings from later cycles.
* Windows never cross from one engine into another.
* Early in an engine's life fewer than ``WINDOW`` cycles exist. The window
  then uses the cycles available so far (at cycle 1 the mean equals the
  reading and the standard deviation is 0). This keeps every row, which is
  required because some test engines are shorter than the window.
* Nothing is fitted here, so there are no statistics that could leak from
  test engines. Scaling is done beforehand in ``preprocessing.py``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from . import data_loader as dl
from . import plot_style as ps
from . import preprocessing as pp

TABLE_DIR = dl.PROJECT_ROOT / "results" / "tables"

WINDOW = 30
REPRESENTATIONS = ("raw", "summary")
STATISTICS = ("last", "mean", "std")


def feature_names(sensors: list[str], representation: str) -> list[str]:
    """Column names of the model inputs for one representation."""
    if representation == "raw":
        return list(sensors)
    if representation == "summary":
        return [f"{s}_{stat}" for s in sensors for stat in STATISTICS]
    raise ValueError(f"Unknown representation {representation!r}. Use one of {REPRESENTATIONS}.")


def _check_cycle_order(table: pd.DataFrame) -> None:
    """Rolling windows assume each engine's rows are stored in cycle order."""
    steps = table.groupby("unit", sort=False)["cycle"].diff().dropna()
    if not (steps == 1).all():
        raise ValueError("Rows must be sorted by cycle within each unit, with no gaps.")


def make_features(
    table: pd.DataFrame, sensors: list[str], representation: str = "raw", window: int = WINDOW
) -> pd.DataFrame:
    """Build the model-input table for one representation.

    ``table`` is a preprocessed table (scaled sensors). Identifier columns and
    RUL targets are carried over unchanged; only the sensor columns are
    replaced by the chosen representation.
    """
    names = feature_names(sensors, representation)
    ids = [c for c in pp.ID_COLS if c in table.columns]
    targets = [c for c in pp.TARGET_COLS if c in table.columns]

    if representation == "raw":
        features = table[sensors]
    else:
        _check_cycle_order(table)
        rolling = table.groupby("unit", sort=False)[sensors].rolling(window, min_periods=1)
        # groupby-rolling returns a (unit, row) index; drop "unit" to line up with the table
        mean = rolling.mean().reset_index(level=0, drop=True).reindex(table.index)
        std = rolling.std(ddof=0).reset_index(level=0, drop=True).reindex(table.index)
        features = pd.concat(
            [table[sensors].add_suffix("_last"), mean.add_suffix("_mean"), std.add_suffix("_std")],
            axis=1,
        )[names]

    return pd.concat([table[ids], features, table[targets]], axis=1)


def last_cycle_rows(table: pd.DataFrame) -> pd.DataFrame:
    """The final observed cycle of every engine (used to score the test set)."""
    return table.groupby("unit", sort=False).tail(1).reset_index(drop=True)


def build(subset: str, scheme: str = "global", representation: str = "raw", window: int = WINDOW) -> dict:
    """Preprocess one subset and return train/test feature tables.

    Returns a dict with ``train``, ``test``, ``features`` (input column
    names), ``sensors``, and the subset/scheme/representation labels.
    """
    prepared = pp.preprocess(subset, scheme)
    sensors = prepared["sensors"]
    return {
        "subset": subset,
        "scheme": scheme,
        "representation": representation,
        "window": window if representation == "summary" else None,
        "sensors": sensors,
        "features": feature_names(sensors, representation),
        "train": make_features(prepared["train"], sensors, representation, window),
        "test": make_features(prepared["test"], sensors, representation, window),
    }


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def feature_set_summary() -> pd.DataFrame:
    """Size of every feature set used in the experiments."""
    rows = []
    for subset, scheme in pp.EXPERIMENT_GRID:
        for representation in REPRESENTATIONS:
            built = build(subset, scheme, representation)
            train, test = built["train"], built["test"]
            partial = int((test.groupby("unit")["cycle"].max() < WINDOW).sum())
            rows.append(
                {
                    "subset": subset,
                    "scheme": scheme,
                    "representation": representation,
                    "sensors": len(built["sensors"]),
                    "features": len(built["features"]),
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "test_engines": test["unit"].nunique(),
                    "test_engines_shorter_than_window": partial if representation == "summary" else 0,
                }
            )
    return pd.DataFrame(rows)


def feature_signal_summary() -> pd.DataFrame:
    """How strongly each kind of feature tracks the capped RUL (training data).

    For every feature the absolute Spearman rank correlation with ``rul`` is
    computed; the table reports the median and maximum per statistic. This is
    a description of the inputs, not a model result.
    """
    rows = []
    for subset, scheme in pp.EXPERIMENT_GRID:
        built = build(subset, scheme, "summary")
        train = built["train"]
        rho = train[built["features"]].corrwith(train["rul"], method="spearman").abs()
        for stat in STATISTICS:
            part = rho[[f"{s}_{stat}" for s in built["sensors"]]]
            rows.append(
                {
                    "subset": subset,
                    "scheme": scheme,
                    "statistic": stat,
                    "features": len(part),
                    "median_abs_spearman": part.median(),
                    "max_abs_spearman": part.max(),
                    "strongest_feature": part.idxmax(),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------

def plot_summary_features(sensor: str = "s11"):
    """Figure 3c - one sensor of one engine as raw readings and window statistics."""
    ps.apply_style()
    titles = {
        ("FD001", "global"): "FD001",
        ("FD002", "global"): "FD002, dataset-wide normalisation",
        ("FD002", "condition"): "FD002, per-condition normalisation",
    }
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharex="col")
    symbol, description, _ = dl.SENSOR_INFO[sensor]

    for c, (subset, scheme) in enumerate(pp.EXPERIMENT_GRID):
        train = build(subset, scheme, "summary")["train"]
        lifetimes = train.groupby("unit")["cycle"].max().sort_values(kind="stable")
        unit = int(lifetimes.index[len(lifetimes) // 2])  # median-lifetime engine
        engine = train[train["unit"] == unit]
        color = ps.COLORS[subset]

        top, bottom = axes[0, c], axes[1, c]
        top.plot(
            engine["cycle"], engine[f"{sensor}_last"], linestyle="none", marker="o",
            markersize=2.4, color=ps.MUTED, alpha=0.8, label="Last value (reading of the cycle)",
        )
        top.plot(engine["cycle"], engine[f"{sensor}_mean"], color=color, linewidth=2.0,
                 label=f"Mean of last {WINDOW} cycles")
        top.set_title(f"{titles[(subset, scheme)]}\n(training engine {unit})", fontsize=9.5)
        if c == 0:  # same encoding in every column, so one legend is enough
            top.legend(loc="upper left", fontsize=7.5)

        bottom.plot(engine["cycle"], engine[f"{sensor}_std"], color=color, linewidth=2.0)
        bottom.set_ylim(bottom=0)
        bottom.set_xlabel("Operating cycle")
        for ax in (top, bottom):
            ax.axvline(WINDOW, color=ps.AXIS, linewidth=0.9, linestyle=":")
        bottom.text(
            WINDOW + 3, 0.04, "window full", transform=bottom.get_xaxis_transform(),
            fontsize=7.5, color=ps.INK_SOFT, va="bottom",
        )

    axes[0, 0].set_ylabel(f"{symbol} (z-score)")
    axes[1, 0].set_ylabel(f"Std. dev. of last {WINDOW} cycles")
    fig.suptitle(f"Summary-statistic features for sensor {sensor} ({description[0].lower()}{description[1:]})")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Run everything
# --------------------------------------------------------------------------

def run_all() -> dict:
    """Write the Phase 4 tables and figure and return where each was saved."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    saved = {}

    tables = {
        "phase4_feature_sets": feature_set_summary(),
        "phase4_feature_signal": feature_signal_summary(),
    }
    names = []
    for subset, scheme in pp.EXPERIMENT_GRID:
        sensors = pp.preprocess(subset, scheme)["sensors"]
        for representation in REPRESENTATIONS:
            for position, name in enumerate(feature_names(sensors, representation), start=1):
                names.append(
                    {"subset": subset, "scheme": scheme, "representation": representation,
                     "position": position, "feature": name}
                )
    tables["phase4_feature_names"] = pd.DataFrame(names)

    for name, table in tables.items():
        path = TABLE_DIR / f"{name}.csv"
        table.to_csv(path, index=False, float_format="%.6g")
        saved[name] = path

    fig = plot_summary_features()
    saved["fig03c_summary_features_example"] = ps.save_figure(fig, "fig03c_summary_features_example")
    plt.close(fig)
    return saved


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    for name, path in run_all().items():
        print(f"saved {name}: {path.relative_to(dl.PROJECT_ROOT)}")
