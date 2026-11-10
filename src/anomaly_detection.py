"""Phase 6 - unsupervised anomaly detection with Isolation Forest (Experiment 2).

How the detector is built (no failure labels are used):

1. **Healthy data.** Cycles of training engines that are at least
   ``HEALTHY_RUL`` (125) cycles away from failure. This is the flat part of
   the RUL target, where Phase 2 showed the sensors do not yet drift.
2. **Model.** An Isolation Forest is fitted on those healthy cycles only. It
   gives every cycle an anomaly score: the easier a cycle is to separate
   from the healthy data, the higher the score.
3. **Alarm threshold.** Set so that ``ALARM_BUDGET`` (5%) of the healthy
   training cycles would be flagged. This is a false-alarm budget chosen in
   advance; it is not tuned on any labels.

How the detector is evaluated:

* **Reference label.** A cycle is called anomalous when the engine has
  ``ANOMALY_RUL`` (30) or fewer cycles left. This label is constructed from
  the run-to-failure records. It marks late degradation; it is not a record
  of an observed fault, and an alarm outside that zone is not necessarily
  wrong (the engine may already be degrading).
* **Two scorings.** "all cycles" counts every cycle. "healthy vs near
  failure" leaves out the transition zone between 30 and 125 cycles, where
  the reference label is least meaningful.
* **Two evaluation sets.** Engine-level cross-validation on the training
  engines (full run-to-failure trajectories), and the official test engines.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupKFold

from . import data_loader as dl
from . import feature_engineering as fe
from . import preprocessing as pp
from .evaluation import detection_metrics

METRIC_DIR = dl.PROJECT_ROOT / "results" / "metrics"
TABLE_DIR = dl.PROJECT_ROOT / "results" / "tables"

SEED = 42
N_SPLITS = 5
HEALTHY_RUL = 125          # healthy: at least this many cycles before failure
ANOMALY_RUL = 30           # reference label: anomalous at or below this RUL
ALARM_BUDGET = 0.05        # share of healthy training cycles allowed to be flagged
BUDGET_SENSITIVITY = (0.01, 0.05, 0.10)

IF_PARAMS = {"n_estimators": 200, "max_samples": 256, "random_state": SEED, "n_jobs": -1}

ZONES = ("healthy", "transition", "near_failure")
SCORINGS = {
    "all cycles": ZONES,                                   # every cycle is scored
    "healthy vs near failure": ("healthy", "near_failure"),  # transition zone left out
}


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

def add_zone(table: pd.DataFrame) -> pd.DataFrame:
    """Add the life zone of every cycle and the reference label ``is_anomaly``."""
    out = table.copy()
    rul = out["rul_uncapped"]
    out["zone"] = np.select(
        [rul >= HEALTHY_RUL, rul <= ANOMALY_RUL], ["healthy", "near_failure"], default="transition"
    )
    out["is_anomaly"] = rul <= ANOMALY_RUL
    return out


def first_usable_cycle(representation: str) -> int:
    """First cycle at which the detector may raise an alarm.

    Summary-statistic features need a full window. During an engine's first
    29 cycles the window is still filling, and those half-filled windows look
    unusual to an unsupervised model for reasons unrelated to engine health.
    For that representation the detector therefore learns from, and raises
    alarms on, cycles from ``WINDOW`` (30) onwards. Raw readings need no
    warm-up.
    """
    return fe.WINDOW if representation == "summary" else 1


def fit_detector(train: pd.DataFrame, features: list[str], first_cycle: int = 1, **overrides) -> dict:
    """Fit an Isolation Forest on the healthy cycles of ``train``.

    Returns the model, the feature names and the anomaly scores of the healthy
    training cycles (needed to place the alarm threshold).
    """
    is_healthy = (train["rul_uncapped"] >= HEALTHY_RUL) & (train["cycle"] >= first_cycle)
    healthy = train.loc[is_healthy, features]
    model = IsolationForest(**{**IF_PARAMS, **overrides}).fit(healthy)
    return {
        "model": model,
        "features": features,
        "first_cycle": first_cycle,
        "healthy_scores": anomaly_score(model, healthy),
        "healthy_cycles": len(healthy),
    }


def anomaly_score(model: IsolationForest, X: pd.DataFrame) -> np.ndarray:
    """Anomaly score per cycle; higher means more unusual (range roughly 0.3 to 0.8)."""
    return -model.score_samples(X)


def alarm_threshold(healthy_scores: np.ndarray, budget: float = ALARM_BUDGET) -> float:
    """Score above which a cycle is flagged, so that ``budget`` of healthy cycles exceed it."""
    return float(np.quantile(healthy_scores, 1 - budget))


def score_cycles(detector: dict, table: pd.DataFrame) -> pd.DataFrame:
    """Score every cycle of ``table`` and flag it at each alarm budget."""
    out = add_zone(table)[["unit", "cycle", "rul_uncapped", "zone", "is_anomaly"]].copy()
    out["score"] = anomaly_score(detector["model"], table[detector["features"]])
    out["warm_up"] = out["cycle"] < detector["first_cycle"]  # no alarm possible yet
    for budget in BUDGET_SENSITIVITY:
        threshold = alarm_threshold(detector["healthy_scores"], budget)
        out[f"flag_{int(budget * 100):02d}"] = (out["score"] > threshold) & ~out["warm_up"]
    out["threshold"] = alarm_threshold(detector["healthy_scores"], ALARM_BUDGET)
    return out


def summarise(scored: pd.DataFrame, **labels) -> pd.DataFrame:
    """Precision, recall and F1 for every scoring and alarm budget."""
    rows = []
    for scoring, zones in SCORINGS.items():
        part = scored[scored["zone"].isin(zones)]
        for budget in BUDGET_SENSITIVITY:
            flags = part[f"flag_{int(budget * 100):02d}"]
            rows.append(
                {**labels, "scoring": scoring, "alarm_budget": budget,
                 "main_setting": budget == ALARM_BUDGET,
                 **detection_metrics(part["is_anomaly"], flags)}
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def cross_validate(
    subset: str, scheme: str, representation: str,
    first_cycle: int | None = None, exclude_sensors: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Out-of-fold scores for every cycle of every training engine.

    Engines are split into 5 folds. For each split the preprocessing and the
    detector are fitted on the training folds, and the held-out engines are
    scored along their whole run-to-failure trajectory.

    ``first_cycle`` and ``exclude_sensors`` are used only by ``design_checks``
    to show what happens without the warm-up rule or without some sensors.
    """
    raw = pp.add_train_rul(dl.load_train(subset))
    if first_cycle is None:
        first_cycle = first_usable_cycle(representation)
    parts = []
    for fold, (train_idx, val_idx) in enumerate(GroupKFold(N_SPLITS).split(raw, groups=raw["unit"]), start=1):
        fitted = pp.fit_and_apply(raw.iloc[train_idx], {"val": raw.iloc[val_idx]}, scheme)
        used = [s for s in fitted["sensors"] if s not in exclude_sensors]
        names = fe.feature_names(used, representation)
        train = fe.make_features(fitted["train"], fitted["sensors"], representation)
        val = fe.make_features(fitted["val"], fitted["sensors"], representation)
        scored = score_cycles(fit_detector(train, names, first_cycle), val)
        scored.insert(0, "fold", fold)
        parts.append(scored)
    return pd.concat(parts, ignore_index=True)


def fit_and_test(subset: str, scheme: str, representation: str) -> pd.DataFrame:
    """Fit on all training engines and score every cycle of the official test engines."""
    built = fe.build(subset, scheme, representation)
    detector = fit_detector(built["train"], built["features"], first_usable_cycle(representation))
    return score_cycles(detector, built["test"])


def alarm_rate_by_rul(scored: pd.DataFrame, width: int = 10) -> pd.DataFrame:
    """Share of cycles flagged (main budget) in bins of remaining useful life."""
    flag = f"flag_{int(ALARM_BUDGET * 100):02d}"
    rul = scored["rul_uncapped"]
    # upper edge of each bin: RUL 0-10 -> 10, 11-20 -> 20, ..., 121-124 -> 130
    upper = (np.ceil(rul.clip(lower=1) / width) * width).astype(int)
    upper = upper.where(rul < HEALTHY_RUL, HEALTHY_RUL + 2 * width)  # all healthy cycles in one bin
    out = scored.groupby(upper.rename("bin_upper"))[flag].agg(alarm_rate="mean", cycles="size").reset_index()

    def label(edge: int) -> str:
        if edge > HEALTHY_RUL + width:
            return f"{HEALTHY_RUL}+"
        low = 0 if edge == width else edge - width + 1
        return f"{low}-{min(edge, HEALTHY_RUL - 1)}"

    out.insert(1, "rul_bin", out["bin_upper"].map(label))
    return out


# --------------------------------------------------------------------------
# Evidence behind two design decisions (training engines only)
# --------------------------------------------------------------------------

QUANTISED_SENSORS = ("s6", "s10", "s16")  # sensors recorded with very few distinct values


def design_checks() -> dict[str, pd.DataFrame]:
    """Cross-validation evidence for the warm-up rule and for keeping all sensors.

    Both checks use training engines only, so the official test engines play
    no part in these design decisions.

    * ``warm_up``: summary-statistic detector WITHOUT the warm-up rule. Share
      of healthy cycles flagged before and from cycle 30.
    * ``sensors``: scores with all retained sensors and with the quantised
      sensors left out.
    """
    flag = f"flag_{int(ALARM_BUDGET * 100):02d}"
    warm_up, sensors = [], []
    for subset, scheme in pp.EXPERIMENT_GRID:
        labels = {"subset": subset, "scheme": scheme}
        scored = cross_validate(subset, scheme, "summary", first_cycle=1)
        healthy = scored[scored["zone"] == "healthy"]
        early = healthy["cycle"] < fe.WINDOW
        warm_up.append(
            {
                **labels,
                "healthy_cycles_before_30": int(early.sum()),
                "flagged_share_before_30": healthy.loc[early, flag].mean(),
                "healthy_cycles_from_30": int((~early).sum()),
                "flagged_share_from_30": healthy.loc[~early, flag].mean(),
            }
        )
        for representation in fe.REPRESENTATIONS:
            for name, excluded in (("all retained sensors", ()), ("without s6, s10, s16", QUANTISED_SENSORS)):
                result = summarise(cross_validate(subset, scheme, representation, exclude_sensors=excluded))
                main = result[(result["scoring"] == "all cycles") & result["main_setting"]].iloc[0]
                sensors.append(
                    {**labels, "representation": representation, "inputs": name,
                     "precision": main["precision"], "recall": main["recall"], "f1": main["f1"]}
                )
    return {"warm_up": pd.DataFrame(warm_up), "sensors": pd.DataFrame(sensors)}


# --------------------------------------------------------------------------
# Run everything
# --------------------------------------------------------------------------

def run_all(log=print) -> dict:
    """Run Experiment 2 for every data configuration and representation."""
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    metrics, test_scores, rates = [], [], []

    for subset, scheme in pp.EXPERIMENT_GRID:
        for representation in fe.REPRESENTATIONS:
            labels = {"subset": subset, "scheme": scheme, "representation": representation}
            for evaluation, scored in (
                ("cross-validation (training engines)", cross_validate(subset, scheme, representation)),
                ("official test engines", fit_and_test(subset, scheme, representation)),
            ):
                summary = summarise(scored, **labels, evaluation=evaluation)
                metrics.append(summary)
                rate = alarm_rate_by_rul(scored)
                for position, (key, value) in enumerate({**labels, "evaluation": evaluation}.items()):
                    rate.insert(position, key, value)
                rates.append(rate)
                if evaluation.startswith("official"):
                    for position, (key, value) in enumerate(labels.items()):
                        scored.insert(position, key, value)
                    test_scores.append(scored)
                main = summary[(summary["scoring"] == "all cycles") & summary["main_setting"]].iloc[0]
                log(
                    f"{subset}/{scheme}/{representation:8s} {evaluation:36s} "
                    f"P {main['precision']:.3f}  R {main['recall']:.3f}  F1 {main['f1']:.3f}"
                )

    log("Design checks on training engines ...")
    checks = design_checks()

    saved = {}
    outputs = {
        METRIC_DIR / "phase6_anomaly_metrics.csv": pd.concat(metrics, ignore_index=True),
        METRIC_DIR / "phase6_alarm_rate_by_rul.csv": pd.concat(rates, ignore_index=True),
        METRIC_DIR / "phase6_design_check_warm_up.csv": checks["warm_up"],
        METRIC_DIR / "phase6_design_check_sensors.csv": checks["sensors"],
        TABLE_DIR / "phase6_test_scores.csv": pd.concat(test_scores, ignore_index=True),
    }
    for path, table in outputs.items():
        table.to_csv(path, index=False, float_format="%.5g")
        saved[path.stem] = path

    config = {
        "seed": SEED,
        "isolation_forest": IF_PARAMS,
        "healthy_definition": f"training-engine cycles with RUL >= {HEALTHY_RUL}",
        "reference_label": f"anomalous if RUL <= {ANOMALY_RUL}",
        "alarm_budget_main": ALARM_BUDGET,
        "alarm_budgets_reported": list(BUDGET_SENSITIVITY),
        "threshold_rule": "quantile of the anomaly scores of the healthy training cycles; no labels used",
        "cv": f"GroupKFold by engine, {N_SPLITS} folds, preprocessing and detector refitted on the training folds",
        "scorings": {name: list(zones) for name, zones in SCORINGS.items()},
        "total_seconds": round(time.perf_counter() - started, 1),
    }
    config_path = METRIC_DIR / "phase6_experiment_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    saved[config_path.stem] = config_path
    log(f"Done in {config['total_seconds']} seconds.")
    return saved


if __name__ == "__main__":
    for name, path in run_all(log=lambda m: print(m, flush=True)).items():
        print(f"saved {name}: {path.relative_to(dl.PROJECT_ROOT)}")
