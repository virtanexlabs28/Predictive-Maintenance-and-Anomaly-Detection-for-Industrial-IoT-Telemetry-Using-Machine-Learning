"""Load the NASA C-MAPSS turbofan degradation files used in this thesis.

Only the two subsets in the research scope are supported:

* FD001 - one operating condition, one fault mode (HPC degradation)
* FD002 - six operating conditions, one fault mode (HPC degradation)

Each raw file is a space-separated text table with 26 columns and no header:
unit number, cycle, three operational settings and 21 sensor measurements.
This module only reads the files and names the columns. It does not change
any value; cleaning and scaling happen later in ``preprocessing.py``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CMAPSS_DIR = RAW_DIR / "CMAPSSData"
CHECKSUM_FILE = RAW_DIR / "SHA256SUMS.txt"

SUBSETS = ("FD001", "FD002")

INDEX_COLS = ["unit", "cycle"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLUMN_NAMES = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# Sensor symbols, descriptions and units from Table 2 of Saxena et al. (2008),
# "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation".
# The dataset readme only numbers the sensors (1-21); matching them to the
# table in listed order is the convention used in the C-MAPSS literature.
SENSOR_INFO = {
    "s1": ("T2", "Total temperature at fan inlet", "deg R"),
    "s2": ("T24", "Total temperature at LPC outlet", "deg R"),
    "s3": ("T30", "Total temperature at HPC outlet", "deg R"),
    "s4": ("T50", "Total temperature at LPT outlet", "deg R"),
    "s5": ("P2", "Pressure at fan inlet", "psia"),
    "s6": ("P15", "Total pressure in bypass-duct", "psia"),
    "s7": ("P30", "Total pressure at HPC outlet", "psia"),
    "s8": ("Nf", "Physical fan speed", "rpm"),
    "s9": ("Nc", "Physical core speed", "rpm"),
    "s10": ("epr", "Engine pressure ratio (P50/P2)", "-"),
    "s11": ("Ps30", "Static pressure at HPC outlet", "psia"),
    "s12": ("phi", "Ratio of fuel flow to Ps30", "pps/psi"),
    "s13": ("NRf", "Corrected fan speed", "rpm"),
    "s14": ("NRc", "Corrected core speed", "rpm"),
    "s15": ("BPR", "Bypass ratio", "-"),
    "s16": ("farB", "Burner fuel-air ratio", "-"),
    "s17": ("htBleed", "Bleed enthalpy", "-"),
    "s18": ("Nf_dmd", "Demanded fan speed", "rpm"),
    "s19": ("PCNfR_dmd", "Demanded corrected fan speed", "rpm"),
    "s20": ("W31", "HPT coolant bleed", "lbm/s"),
    "s21": ("W32", "LPT coolant bleed", "lbm/s"),
}

# Number of engines per file, as stated in the dataset readme.
EXPECTED_UNITS = {
    "FD001": {"train": 100, "test": 100},
    "FD002": {"train": 260, "test": 259},
}


def _check_subset(subset: str) -> None:
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}. Use one of {SUBSETS}.")


def _read_trajectories(path: Path) -> pd.DataFrame:
    """Read one train/test file and attach the 26 column names."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See README.md for the download steps.")
    # Lines end with trailing spaces, so split on any run of whitespace.
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != len(COLUMN_NAMES):
        raise ValueError(f"{path.name}: expected {len(COLUMN_NAMES)} columns, found {df.shape[1]}.")
    df.columns = COLUMN_NAMES
    df[INDEX_COLS] = df[INDEX_COLS].astype(int)
    return df


def load_train(subset: str) -> pd.DataFrame:
    """Training engines: every trajectory runs until failure."""
    _check_subset(subset)
    return _read_trajectories(CMAPSS_DIR / f"train_{subset}.txt")


def load_test(subset: str) -> pd.DataFrame:
    """Test engines: every trajectory stops some time before failure."""
    _check_subset(subset)
    return _read_trajectories(CMAPSS_DIR / f"test_{subset}.txt")


def load_rul(subset: str) -> pd.DataFrame:
    """True RUL at the last recorded cycle of each test engine.

    The file has one value per line; line *i* belongs to test unit *i*.
    """
    _check_subset(subset)
    path = CMAPSS_DIR / f"RUL_{subset}.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See README.md for the download steps.")
    rul = pd.read_csv(path, sep=r"\s+", header=None, names=["rul"])
    rul.insert(0, "unit", range(1, len(rul) + 1))
    return rul


def load_subset(subset: str) -> dict[str, pd.DataFrame]:
    """Return the three tables of one subset as ``{"train", "test", "rul"}``."""
    return {
        "train": load_train(subset),
        "test": load_test(subset),
        "rul": load_rul(subset),
    }


def verify_checksums() -> dict[str, bool]:
    """Compare the raw files with the SHA-256 values recorded at download time.

    Returns ``{relative_path: matches}``. A missing file counts as a mismatch.
    """
    results = {}
    for line in CHECKSUM_FILE.read_text().splitlines():
        if not line.strip():
            continue
        expected, rel_path = line.split(maxsplit=1)
        path = RAW_DIR / rel_path
        if path.exists():
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            results[rel_path] = actual == expected
        else:
            results[rel_path] = False
    return results


def summarize(subset: str) -> pd.DataFrame:
    """One-row-per-split overview used for the dataset table in the thesis."""
    data = load_subset(subset)
    rows = []
    for split in ("train", "test"):
        df = data[split]
        lengths = df.groupby("unit")["cycle"].max()
        rows.append(
            {
                "subset": subset,
                "split": split,
                "engines": df["unit"].nunique(),
                "rows": len(df),
                "min_cycles": int(lengths.min()),
                "mean_cycles": round(float(lengths.mean()), 1),
                "max_cycles": int(lengths.max()),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Checksums:")
    for name, ok in verify_checksums().items():
        print(f"  {'OK      ' if ok else 'MISMATCH'} {name}")
    overview = pd.concat([summarize(s) for s in SUBSETS], ignore_index=True)
    print("\nDataset overview:")
    print(overview.to_string(index=False))
