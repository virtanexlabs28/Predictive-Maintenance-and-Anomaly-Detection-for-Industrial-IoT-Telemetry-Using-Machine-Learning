# Predictive Maintenance and Anomaly Detection for Industrial IoT Telemetry Using Machine Learning

Undergraduate thesis project. It applies three classical machine learning models to the
NASA C-MAPSS turbofan engine degradation data:

| Model | Task |
|---|---|
| Random Forest | Remaining Useful Life (RUL) prediction |
| XGBoost | Remaining Useful Life (RUL) prediction |
| Isolation Forest | Unsupervised anomaly detection |

## Research questions

- **RQ1 - Prognostic and diagnostic accuracy.** How accurately can selected machine learning
  models detect anomalies and predict RUL using single-condition and multi-condition
  industrial telemetry data?
- **RQ2 - Preprocessing and feature representation.** How do basic preprocessing techniques
  and feature representations affect the performance of machine learning models in
  industrial telemetry analysis?

## Scope

- Data: C-MAPSS subsets **FD001** (one operating condition) and **FD002** (six operating
  conditions). Both have one fault mode, HPC degradation. FD003 and FD004 are in the
  downloaded archive but are not used.
- No deep learning, no stacking, no dashboard or web application.
- C-MAPSS is a simulation. Results describe model behaviour on simulated engines and are
  not evidence of performance on real industrial equipment.

## Project structure

```text
data/raw/          original NASA files and their SHA-256 checksums
data/processed/    cleaned tables written by the preprocessing phase
notebooks/         one notebook per phase (exploration, preprocessing, ...)
src/               reusable Python modules
tests/             automated checks
results/           figures, tables and metrics produced by the code
reports/           research log and thesis chapters
```

## Setup

Python 3.12 on Windows. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Data

Source: NASA Prognostics Center of Excellence Data Set Repository
<https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>,
entry "Turbofan Engine Degradation Simulation".

1. Download `6. Turbofan Engine Degradation Simulation Data Set.zip` from the link on that
   page and save it as `data/raw/_download/cmapss_nasa.zip`.
2. The zip contains a second zip, `CMAPSSData.zip`. Extract its contents into
   `data/raw/CMAPSSData/`.
3. Confirm the files are the same version used in this thesis:

```powershell
.\.venv\Scripts\python.exe -m src.data_loader
```

Every line should read `OK`.

## Checks

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

## Reproducing the results

| Phase | Command (from the project root) | Output |
|---|---|---|
| 2 - Data exploration | `.\.venv\Scripts\python.exe -m src.exploration` | `results/tables/phase2_*.csv`, `results/figures/fig01*`, `fig02*` |
| 3 - Preprocessing | `.\.venv\Scripts\python.exe -m src.preprocessing` | `data/processed/*.csv`, `results/tables/phase3_*.csv`, `results/metrics/phase3_*.json`, `results/figures/fig03a*`, `fig03b*` |
| 4 - Feature representations | `.\.venv\Scripts\python.exe -m src.feature_engineering` | `results/tables/phase4_*.csv`, `results/figures/fig03c*` |
| 5 - RUL prediction (training) | `.\.venv\Scripts\python.exe -m src.rul_prediction` | `results/metrics/phase5_*`, `results/tables/phase5_test_predictions.csv` |
| 6 - Anomaly detection | `.\.venv\Scripts\python.exe -m src.anomaly_detection` | `results/metrics/phase6_*`, `results/tables/phase6_test_scores.csv` |
| 5 and 6 - Tables and figures | `.\.venv\Scripts\python.exe -m src.evaluation` | `results/tables/phase5_rul_results_table.csv`, `phase6_anomaly_results_table.csv`, `results/metrics/phase5_*intervals*`, `phase5_paired_differences.csv`, `results/figures/fig04*` to `fig07*` |

| 7 - Thesis tables and Figure 8 | `.\.venv\Scripts\python.exe -m src.results_summary` | `results/tables/thesis/table01` to `table10`, `results/figures/fig08*`, `reports/chapter4_tables_and_figures.md` |

Run the commands in this order. Each phase reads the files written by the earlier ones.

Phase 5 training takes about 16 minutes on a 16-core machine. Saved results are reused when
the settings have not changed; add `--force` to retrain.

The notebooks in `notebooks/` show the same results with explanations. To open them:

```powershell
.\.venv\Scripts\python.exe -m jupyter notebook
```

## Thesis chapters

The chapters are Word files in `reports/thesis/`. They are built from content files in
`reports/thesis/build/` (requires Node.js):

```powershell
cd reports\thesis\build
npm install
node chapter3.js
```

`reports/thesis/reference_verification.md` records how each cited reference was checked.

## Citation for the data

A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set",
NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

A. Saxena, K. Goebel, D. Simon and N. Eklund (2008). "Damage Propagation Modeling for
Aircraft Engine Run-to-Failure Simulation", Proceedings of the 1st International
Conference on Prognostics and Health Management (PHM08), Denver, CO.

## Progress

See `reports/research_log.md` for the record of decisions and results by phase.
