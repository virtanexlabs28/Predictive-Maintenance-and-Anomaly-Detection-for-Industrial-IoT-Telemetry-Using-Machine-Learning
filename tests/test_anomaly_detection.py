"""Checks for Phase 6: reference labels, healthy-only training, thresholds and scoring."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import anomaly_detection as ad  # noqa: E402
from src import feature_engineering as fe  # noqa: E402
from src.evaluation import detection_metrics  # noqa: E402


@pytest.fixture(scope="module")
def built():
    return {rep: fe.build("FD001", "global", rep) for rep in fe.REPRESENTATIONS}


@pytest.fixture(scope="module")
def detectors(built):
    return {
        rep: ad.fit_detector(b["train"], b["features"], ad.first_usable_cycle(rep)) for rep, b in built.items()
    }


def test_detection_metrics_on_known_values():
    scores = detection_metrics([1, 1, 0, 0, 0], [1, 0, 1, 0, 0])
    assert (scores["tp"], scores["fp"], scores["fn"], scores["tn"]) == (1, 1, 1, 2)
    assert scores["precision"] == scores["recall"] == scores["f1"] == pytest.approx(0.5)
    assert scores["prevalence"] == pytest.approx(0.4)
    silent = detection_metrics([1, 0, 0], [0, 0, 0])  # a detector that never fires
    assert silent["precision"] == silent["recall"] == silent["f1"] == 0.0


def test_zone_boundaries():
    table = pd.DataFrame({"rul_uncapped": [200, 125, 124, 31, 30, 0]})
    zoned = ad.add_zone(table)
    assert zoned["zone"].tolist() == ["healthy", "healthy", "transition", "transition", "near_failure", "near_failure"]
    assert zoned["is_anomaly"].tolist() == [False, False, False, False, True, True]


def test_detector_is_fitted_on_healthy_cycles_only(built, detectors):
    for rep, b in built.items():
        train = b["train"]
        expected = ((train["rul_uncapped"] >= ad.HEALTHY_RUL) & (train["cycle"] >= ad.first_usable_cycle(rep))).sum()
        assert detectors[rep]["healthy_cycles"] == expected
        assert len(detectors[rep]["healthy_scores"]) == expected
    assert ad.first_usable_cycle("raw") == 1 and ad.first_usable_cycle("summary") == fe.WINDOW


def test_threshold_respects_the_alarm_budget(detectors):
    scores = detectors["raw"]["healthy_scores"]
    for budget in ad.BUDGET_SENSITIVITY:
        flagged = (scores > ad.alarm_threshold(scores, budget)).sum()
        # the number of flagged healthy cycles matches the budget to within one cycle
        assert abs(flagged - budget * len(scores)) <= 1


def test_flags_grow_with_the_budget_and_respect_warm_up(built, detectors):
    for rep in fe.REPRESENTATIONS:
        scored = ad.score_cycles(detectors[rep], built[rep]["test"])
        assert len(scored) == len(built[rep]["test"])
        assert not (scored["flag_01"] & ~scored["flag_05"]).any()
        assert not (scored["flag_05"] & ~scored["flag_10"]).any()
        flags = scored[["flag_01", "flag_05", "flag_10"]].any(axis=1)
        assert not flags[scored["warm_up"]].any()
    summary_scored = ad.score_cycles(detectors["summary"], built["summary"]["test"])
    assert (summary_scored["warm_up"] == (summary_scored["cycle"] < fe.WINDOW)).all()
    raw_scored = ad.score_cycles(detectors["raw"], built["raw"]["test"])
    assert not raw_scored["warm_up"].any()


def test_scores_are_reproducible(built):
    b = built["raw"]
    first = ad.fit_detector(b["train"], b["features"])
    second = ad.fit_detector(b["train"], b["features"])
    a = ad.anomaly_score(first["model"], b["test"][b["features"]])
    c = ad.anomaly_score(second["model"], b["test"][b["features"]])
    assert np.abs(a - c).max() < 1e-12


def test_cross_validation_scores_every_training_cycle_once():
    scored = ad.cross_validate("FD001", "global", "raw")
    assert len(scored) == 20631
    assert not scored.duplicated(subset=["unit", "cycle"]).any()
    assert sorted(scored["fold"].unique()) == [1, 2, 3, 4, 5]
    assert scored.groupby("unit")["fold"].nunique().max() == 1  # an engine belongs to one fold
    # run-to-failure engines: 31 near-failure cycles each (RUL 0..30)
    assert scored["is_anomaly"].sum() == 100 * 31


def test_summarise_and_alarm_rate_tables():
    scored = ad.cross_validate("FD001", "global", "raw")
    summary = ad.summarise(scored, subset="FD001")
    assert len(summary) == len(ad.SCORINGS) * len(ad.BUDGET_SENSITIVITY)
    assert summary["main_setting"].sum() == len(ad.SCORINGS)
    everything = summary[summary["scoring"] == "all cycles"]["cycles"].iloc[0]
    reduced = summary[summary["scoring"] == "healthy vs near failure"]["cycles"].iloc[0]
    assert everything == len(scored) and reduced == (scored["zone"] != "transition").sum()
    rates = ad.alarm_rate_by_rul(scored)
    assert rates["cycles"].sum() == len(scored)
    assert rates["rul_bin"].iloc[0] == "0-10" and rates["rul_bin"].iloc[-1] == "125+"
    assert rates["rul_bin"].iloc[2] == "21-30" and rates["alarm_rate"].between(0, 1).all()
