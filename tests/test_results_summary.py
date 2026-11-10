"""Checks that the thesis tables agree with the saved experiment results.

These tests need the result files from Phases 1-6 (run the phase commands in
README.md first).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import evaluation as ev  # noqa: E402
from src import results_summary as rs  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (ev.METRIC_DIR / "phase5_test_metrics.csv").exists()
    or not (ev.METRIC_DIR / "phase6_anomaly_metrics.csv").exists(),
    reason="experiment results not generated yet",
)


def test_every_thesis_table_builds_and_is_not_empty():
    for name, caption, builder in rs.THESIS_TABLES:
        table = builder()
        assert len(table) > 0 and caption.endswith("."), name
        assert not table.columns.duplicated().any(), name


def test_rul_table_matches_the_saved_metrics():
    metrics, _ = ev.load_rul_results()
    table = rs.table_rul_results()
    assert len(table) == 12
    assert np.allclose(table["test_rmse"], metrics["test_rmse"].round(3))
    assert np.allclose(table["test_rmse_uncapped"], metrics["test_rmse_uncapped"].round(3))


def test_anomaly_table_matches_the_saved_metrics():
    metrics, _, _ = ev.load_anomaly_results()
    table = rs.table_anomaly_results()
    assert len(table) == 12  # 6 configurations x 2 scorings
    saved = metrics[
        metrics["main_setting"] & (metrics["subset"] == "FD002") & (metrics["scheme"] == "condition")
        & (metrics["representation"] == "summary") & (metrics["scoring"] == "all cycles")
        & (metrics["evaluation"] == "official test engines")
    ].iloc[0]
    row = table[
        (table["data"] == "FD002 per-condition") & (table["representation"] == "Summary statistics")
        & (table["scoring"] == "all cycles")
    ].iloc[0]
    assert row["test_recall"] == pytest.approx(saved["recall"], abs=1e-3)
    assert row["test_f1"] == pytest.approx(saved["f1"], abs=1e-3)


def test_dataset_comparison_has_all_three_configurations():
    table = rs.table_dataset_comparison(with_intervals=False)
    assert {"FD001", "FD002 dataset-wide", "FD002 per-condition"} <= set(table.columns)
    assert not any(c.endswith((" low", " high")) for c in table.columns)
    assert set(table["task"]) == {"RUL prediction", "Anomaly detection"}
    assert table[["FD001", "FD002 dataset-wide", "FD002 per-condition"]].notna().all().all()


def test_markdown_table_format():
    text = rs.to_markdown(pd.DataFrame({"a_b": [1, 2000], "c": [0.12345, 12.345], "d": [True, False]}))
    lines = text.splitlines()
    assert lines[0] == "| a b | c | d |"
    assert lines[1] == "|---|---|---|"
    assert lines[2] == "| 1 | 0.123 | yes |"
    assert lines[3] == "| 2,000 | 12.35 | no |" or lines[3] == "| 2,000 | 12.34 | no |"


def test_all_required_figures_exist():
    inventory = rs.figure_inventory()
    required = {r.split(".")[0] for r in inventory["required_visualisation"] if r[0].isdigit()}
    assert required == {"1", "2", "3", "4", "5", "6", "7", "8"}
    assert inventory["png"].all() and inventory["pdf"].all()
