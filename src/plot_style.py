"""Shared figure style so every thesis figure looks the same.

The two dataset colours are fixed for the whole project: FD001 is always blue
and FD002 is always orange. The pair was checked for colour-blind separation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

COLORS = {"FD001": "#2a78d6", "FD002": "#eb6834"}
# Model colours are used only in figures where the dataset is named in the
# panel title, so they never share a chart with the dataset colours.
MODEL_COLORS = {"random_forest": "#1baf7a", "xgboost": "#6250d6"}
INK = "#0b0b0b"        # titles and main text
INK_SOFT = "#52514e"   # axis labels and notes
MUTED = "#898781"      # tick labels, reference lines
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def apply_style() -> None:
    """Set Matplotlib defaults for clean, print-ready figures."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.labelsize": 9,
            "axes.labelcolor": INK_SOFT,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SOFT,
            "ytick.labelcolor": INK_SOFT,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            "figure.titleweight": "bold",
            "lines.linewidth": 1.2,
        }
    )


def save_figure(fig, name: str) -> Path:
    """Save a figure as 300-dpi PNG (for Word) and PDF (vector, for LaTeX)."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURE_DIR / f"{name}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{name}.pdf", bbox_inches="tight")
    return png
