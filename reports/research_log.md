# Research Log

Running record of decisions, configurations and results. Every number here was produced
by code in this repository on the date shown.

## Scope decisions (2026-11-10)

| Item | Decision |
|---|---|
| Research questions | RQ1 (prognostic and diagnostic accuracy) and RQ2 (preprocessing and feature representation) only |
| Data | C-MAPSS FD001 and FD002 only |
| Models | Random Forest and XGBoost for RUL; Isolation Forest for anomaly detection |
| RUL target | Piecewise linear, capped at 125 cycles |
| Experiments | (1) RF vs XGBoost, (2) Isolation Forest, (3) raw vs summary-statistic features |
| Earlier 21-phase plan | Superseded. Its deep learning models, FD003/FD004, SHAP, conformal prediction and control-chart baselines are out of scope. No figures from that plan are reused. |

## Phase 1 - Project setup and data acquisition (2026-11-10)

### Environment

- Windows 11, Python 3.12.10, project virtual environment in `.venv`.
- Library versions are pinned in `requirements.txt`
  (pandas 2.2.2, numpy 2.4.6, scikit-learn 1.9.0, xgboost 3.3.0, scipy 1.18.0,
  matplotlib 3.11.0, seaborn 0.13.2).

### Dataset version

- Source page: NASA PCoE Data Set Repository, entry "Turbofan Engine Degradation Simulation".
- File: `6. Turbofan Engine Degradation Simulation Data Set.zip`, 12,429,152 bytes,
  server "Last-Modified" 2022-09-18, downloaded 2026-11-10.
- SHA-256 of the zip: `c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2`.
  Its MD5 matched the value reported by the server (`a83e8f128c59fc5614a4ca2e42a2e30c`).
- SHA-256 values of the six FD001/FD002 files and the readme are in `data/raw/SHA256SUMS.txt`.
  `src/data_loader.py::verify_checksums` re-checks them.
- The raw files are never edited. FD003 and FD004 are present in the archive but unused.

### What the files contain (verified)

| Subset | Split | Engines | Rows | Min cycles | Mean cycles | Max cycles |
|---|---|---|---|---|---|---|
| FD001 | train | 100 | 20,631 | 128 | 206.3 | 362 |
| FD001 | test | 100 | 13,096 | 31 | 131.0 | 303 |
| FD002 | train | 260 | 53,759 | 128 | 206.8 | 378 |
| FD002 | test | 259 | 33,991 | 21 | 131.2 | 367 |

Saved as `results/tables/phase1_dataset_overview.csv`.

- Every file has 26 columns: unit, cycle, 3 operational settings, 21 sensors.
- Engine counts match the dataset readme.
- No missing values in any of the four trajectory files.
- Cycles for every engine run 1, 2, 3, ... with no gaps or repeats.
- True test RUL (from `RUL_FD00x.txt`): FD001 min 7, median 86, max 145;
  FD002 min 6, median 80, max 194.
- Operating conditions in the training data, after rounding the three settings:
  FD001 has 1, FD002 has 6. This agrees with the readme.

### Decisions and notes

1. **Column names.** `unit`, `cycle`, `setting_1..3`, `s1..s21`. Short names keep code and
   plots readable; the physical meaning of each sensor is stored in
   `data_loader.SENSOR_INFO`.
2. **Sensor meanings.** Taken from Table 2 of Saxena et al. (2008). The readme only numbers
   the sensors, so the match between column order and table order is the convention used
   in the literature, not something the readme states. This should be said in Chapter 3.
3. **Readme typo.** The readme lists column 26 as "sensor measurement 26". There are 21
   sensors; the 26 refers to the column position.
4. **Trailing spaces.** Each line ends with spaces, so files are read with a
   whitespace-run separator. Reading with a single-space separator creates empty columns.
5. **Loader does no cleaning.** Loading is kept separate from preprocessing so that every
   change to the data is made in one place (Phase 3) and can be explained there.

### Points to carry forward

- The shortest test engine has 31 cycles in FD001 and 21 in FD002. A 30-cycle summary
  window (Phase 4) must therefore handle engines with fewer than 30 cycles.
- True test RUL goes up to 145 (FD001) and 194 (FD002), above the 125 cap. Phase 5 must
  state whether test targets are capped when computing the metrics.
- Open question for Phase 3: whether to compare dataset-wide normalization against
  normalization within each operating condition on FD002 (see the Phase 1 discussion).

### Verification

- `python -m src.data_loader`: all 8 checksum lines `OK`.
- `python -m pytest tests -q`: 16 passed.

## Phase 2 - Data exploration (2026-11-10)

Code: `src/exploration.py`, `src/plot_style.py`. Notebook: `notebooks/01_data_exploration.ipynb`
(executed, outputs embedded). Nothing in this phase changes the data.

**Leakage rule applied.** Sensor statistics come from training engines only. Test files were
inspected only for structure (size, missing values, duplicates, trajectory length) and for
the distribution of the published true RUL values.

### Data quality (`results/tables/phase2_data_quality.csv`)

All four trajectory files: 0 missing values, 0 non-finite values, 0 duplicate rows,
0 duplicate (unit, cycle) pairs, 0 rows repeating another row's settings and sensor readings.
No imputation or record removal is needed.

### Operating conditions (`results/tables/phase2_operating_conditions.csv`)

Found by rounding setting 1 to 0 decimals, setting 2 to 2 decimals and setting 3 to 0 decimals.

| Subset | Altitude (kft) / Mach / throttle | Training rows | Share |
|---|---|---|---|
| FD001 | 0 / 0.00 / 100 | 20,631 | 100% |
| FD002 | 0 / 0.00 / 100 | 8,044 | 15.0% |
| FD002 | 10 / 0.25 / 100 | 8,096 | 15.1% |
| FD002 | 20 / 0.70 / 100 | 8,122 | 15.1% |
| FD002 | 25 / 0.62 / 60 | 8,002 | 14.9% |
| FD002 | 35 / 0.84 / 100 | 8,037 | 15.0% |
| FD002 | 42 / 0.84 / 100 | 13,458 | 25.0% |

### Sensor statistics, training engines (`results/tables/phase2_sensor_summary_FD00x.csv`)

| Finding | FD001 | FD002 |
|---|---|---|
| Sensors constant over the whole file | s1, s5, s10, s16, s18, s19 | none |
| Sensors constant inside every operating condition | same six | s1, s5, s18, s19 |
| Settings constant | setting 3 | none (setting 3 constant inside each condition) |
| Median share of sensor variance explained by operating condition | 0 | 0.9998 |
| Largest absolute Spearman correlation with cycles remaining, raw | 0.718 (s11) | 0.247 (s13) |
| Same, after subtracting each condition's mean | 0.718 (s11) | 0.714 (s11) |

Other observations:

- FD001 s6 has two unique values (21.60, 21.61) and a raw correlation of -0.13 with cycles
  remaining. It is not strictly constant.
- FD001 strongest sensors by absolute correlation: s11 (-0.72), s4 (-0.70), s12 (+0.69),
  s7 (+0.68), s15 (-0.67), s21 (+0.66), s20 (+0.65).
- FD002 sensors with the lowest share of variance from the operating condition:
  s16 (0.885) and s14 (0.966). All others are above 0.99.
- These correlations describe training data. They are exploratory, not model results.

### Figures

| File (`results/figures/`, PNG 300 dpi and PDF) | Content |
|---|---|
| `fig01_dataset_distribution` | Training lifetimes, test observed lengths and true test RUL, FD001 and FD002 |
| `fig02a_sensor_trends_FD001` | All 21 sensors over the life of 5 training engines |
| `fig02b_operating_condition_effect` | s2, s4, s7, s11 for one median-lifetime engine in each subset |

Figure choices: FD001 is blue (`#2a78d6`) and FD002 is orange (`#eb6834`) in every figure
of the thesis; the pair passed a colour-blind separation check. Engines for Figure 2a are
picked evenly from shortest-lived to longest-lived (units chosen by rule, not by hand).
Figure 2b uses the median-lifetime training engine of each subset (FD001 unit 68,
FD002 unit 126).

### Implications for Phase 3

1. No cleaning for missing values or duplicates.
2. A "drop constant sensors" rule fitted on training data removes six sensors in FD001 and
   none in FD002. The thesis must report this difference; it is a property of the data.
3. In FD002 the operating condition explains almost all sensor variance and hides the wear
   trend. Normalising inside each operating condition restores correlations close to FD001.
   This is the evidence for comparing dataset-wide and per-condition normalisation under RQ2.
   Decision still to be made by the author.
4. Sensors are flat in early life and drift later, which supports the capped RUL target.

### Bug found and fixed during this phase

The first version of the operating-condition label printed Mach values rounded to "-0.00"
and "0.00" as different conditions, which split FD001 into two false conditions. Fixed in
`exploration.condition_key` and covered by a test.

### Verification

- `python -m pytest tests -q`: 22 passed (16 from Phase 1, 6 new).
- Notebook executed end to end with no errors; 3 figures rendered.

## Phase 3 - Preprocessing and RUL labels (2026-11-10)

Code: `src/preprocessing.py`. Notebook: `notebooks/02_preprocessing.ipynb` (executed).
Author decision recorded: compare dataset-wide and per-condition normalisation on FD002.

### Decisions

| # | Decision | Reason | Alternatives considered |
|---|---|---|---|
| 1 | Train RUL = last cycle of the engine minus current cycle (0 on the last row) | Training engines run to failure | Counting the last row as 1; this only shifts every label by one cycle |
| 2 | Test RUL = (last observed cycle - cycle) + NASA value | The NASA file gives RUL at the last observed cycle only | - |
| 3 | Target capped at 125 cycles; uncapped value kept as `rul_uncapped` | Fixed by the thesis scope. Phase 2 showed sensors are flat in early life | Uncapped linear target |
| 4 | Drop a sensor only if it has one unique value in the training data inside every normalisation group | Objective rule with no threshold to tune | Dropping near-constant sensors too (e.g. s6), which needs an arbitrary threshold. The 14-sensor set common in the FD001 literature also drops s6 |
| 5 | Z-score for both schemes | Same formula in both, so the only difference is the grouping | Min-max scaling. For tree models any dataset-wide rescaling gives identical splits |
| 6 | Operating condition found by fixed rounding of the three settings | Nothing is learned from data, so it cannot leak; it reproduces the six regimes exactly | K-means with K=6 (earlier plan), which adds a fitted model for the same result |
| 7 | Model inputs are the retained sensors only; the three settings are not inputs | Matches "raw sensor features" in the scope. Under dataset-wide normalisation the regime is still visible to the models through sensors such as s1 and s5 | Adding the settings as inputs |
| 8 | FD001 is run under one scheme only | Verified: both schemes give identical numbers on FD001 (largest difference 0.0) | - |

### Leakage safeguards

- Sensor removal and all means and standard deviations are computed from training engines.
- Test engines are transformed with training statistics. A test recomputes one test value by hand.
- `SensorScaler` has separate `fit` and `transform`. In Phase 5 cross-validation it must be
  refitted on the training folds only, so validation engines do not influence the statistics.
  The files in `data/processed/` use statistics from all training engines and are meant for
  the final fit and the official test set.

### Results (`results/tables/phase3_preprocessing_summary.csv`)

| | FD001 / global | FD002 / global | FD002 / condition |
|---|---|---|---|
| Sensors dropped | s1, s5, s10, s16, s18, s19 | none | s1, s5, s18, s19 |
| Sensors retained | 15 | 21 | 17 |
| Training rows | 20,631 | 53,759 | 53,759 |
| Training rows on the cap (RUL >= 125) | 39.4% | 39.5% | 39.5% |
| Test rows / engines | 13,096 / 100 | 33,991 / 259 | 33,991 / 259 |
| Test engines with true RUL above 125 | 11 | 57 | 57 |

Rank correlation of scaled sensors with capped RUL, training engines (descriptive only):

| | FD001 / global | FD002 / global | FD002 / condition |
|---|---|---|---|
| Strongest sensor | s11, 0.720 | s13, 0.250 | s11, 0.715 |
| Median over retained sensors | 0.635 | 0.078 | 0.454 |

### Known side effect: quantised sensors

s6, s10 and s16 take very few distinct values (2-3 inside an operating condition). Under
per-condition scaling their standard deviation is tiny and rare readings become very large
z-scores. Example: s10 in condition 10 / 0.25 / 100 is 1.26 on 8,093 training rows and 1.25
on 3, giving z = -51.9 for those 3. Rows with any |z| > 8 in FD002 / condition: 68 of 53,759
training rows and 28 of 33,991 test rows. Where a sensor is constant inside one condition
but not all (s10 in 1 condition, s16 in 5), the scaled value is set to 0.

Consequences: none for tree splits, which depend on order and not size. For Phase 4 the
window statistics of these sensors will contain spikes. For Phase 6 the Isolation Forest
reacts to rare values, so the choice of input sensors must be made and justified there.

### Outputs

- `data/processed/{FD001_global, FD002_global, FD002_condition}_{train,test}.csv`:
  `unit`, `cycle`, `condition`, scaled sensors, `rul_uncapped`, `rul`.
- `results/tables/phase3_scaler_parameters.csv`: every fitted mean and standard deviation.
- `results/metrics/phase3_preprocessing_config.json`: rules and settings.
- `results/figures/fig03a_rul_target` (RUL label construction) and
  `fig03b_normalisation_effect_FD002` (one engine under both schemes).

### Points to carry forward

- 11 FD001 and 57 FD002 test engines have a true RUL above the cap. Phase 5 must state
  whether metrics use capped or uncapped test targets.
- 39% of training rows sit on the cap, so the target distribution has a large spike at 125.

### Verification

- `python -m pytest tests -q`: 36 passed (14 new for preprocessing).
- Notebook executed end to end with no errors; 2 figures rendered.

## Phase 4 - Feature representations (2026-11-10)

Code: `src/feature_engineering.py`. Notebook: Part B of `notebooks/02_preprocessing.ipynb`
(executed). The five-notebook structure is kept; Phase 4 has no notebook of its own.

### Definitions

| Representation | One row is | Inputs per sensor |
|---|---|---|
| A - `raw` | one engine cycle | scaled reading of that cycle |
| B - `summary` | one engine cycle | `_last`, `_mean`, `_std` over the most recent 30 cycles, current cycle included |

Both representations have the same rows and targets. B contains A (`_last` is the raw
reading) plus two columns per sensor.

### Decisions

| # | Decision | Reason | Alternatives considered |
|---|---|---|---|
| 1 | One row per cycle in both representations | Same samples and targets, so differences in accuracy come from the inputs alone | One summary row per engine, which leaves 100 training rows in FD001 |
| 2 | Window of 30 cycles, fixed | Often used in C-MAPSS studies (source to be verified and cited in the literature review); no tuning keeps the scope small | Sweeping 15/30/50 (earlier plan), out of scope |
| 3 | Short histories use the cycles available (minimum 1) | Six FD002 test engines have fewer than 30 cycles and must still be scored; every training row is kept | Dropping rows with fewer than 30 cycles, which cannot score those test engines |
| 4 | Population standard deviation (divide by n) | Defined for a single reading (0), so cycle 1 needs no special case | Sample standard deviation, undefined for one reading |
| 5 | Statistics limited to last value, mean, standard deviation | Fixed by the thesis scope | Trend slope, min/max (earlier plan), out of scope |
| 6 | Feature tables are built in memory and not saved | They are rebuilt in seconds from the processed files; saving all six would add over 100 MB | - |

### Leakage safeguards

- Windows look backwards only. A test checks that cutting off an engine's future leaves
  its earlier features unchanged.
- Windows never cross engines. A test checks that every engine's first row has
  mean = reading and standard deviation = 0.
- Nothing is fitted in this step. Scaling statistics come from Phase 3 (training engines).

### Results (`results/tables/phase4_feature_sets.csv`)

| | FD001 | FD002 / global | FD002 / condition |
|---|---|---|---|
| Inputs, raw | 15 | 21 | 17 |
| Inputs, summary | 45 | 63 | 51 |
| Training rows (both) | 20,631 | 53,759 | 53,759 |
| Test engines shorter than the window | 0 | 6 | 6 |

Absolute Spearman correlation with capped RUL, training engines, median over sensors
(`results/tables/phase4_feature_signal.csv`, descriptive only):

| Statistic | FD001 | FD002 / global | FD002 / condition |
|---|---|---|---|
| Last value | 0.635 | 0.078 | 0.454 |
| Window mean | 0.707 | 0.048 | 0.688 |
| Window standard deviation | 0.320 | 0.072 | 0.258 |

The window mean relates to RUL more strongly than the single reading in FD001 and in
FD002 / condition. In FD002 / global no statistic helps, because the window averages
readings taken at different operating conditions.

Numerical check: the rolling statistics agree with a direct window-by-window calculation
over all training rows and sensors to within 3e-12.

### Property to state in the thesis

During an engine's first 29 cycles the window is not yet full, so the mean is noisier and
the standard deviation starts at 0. These values indirectly show that the engine is new.
This is information a maintenance system would have in practice (engine age), and it
applies equally to test engines, which also start at cycle 1. It is not leakage, but it
is one way representation B can differ from A beyond noise reduction.

### Outputs

- `results/tables/phase4_feature_sets.csv`, `phase4_feature_signal.csv`,
  `phase4_feature_names.csv` (every input column of every feature set).
- `results/figures/fig03c_summary_features_example`: sensor s11 of one engine as raw
  readings, window mean and window standard deviation, for the three data configurations.

### Verification

- `python -m pytest tests -q`: 47 passed (11 new for feature engineering).
- Notebook 02 executed end to end with no errors; 3 figures rendered.

## Phase 5 - RUL prediction: Experiments 1 and 3 (2026-11-10)

Code: `src/rul_prediction.py` (training), `src/evaluation.py` (metrics, intervals, figures).
Notebook: `notebooks/03_rul_prediction.ipynb` (executed).

### Training configuration

| Item | Setting |
|---|---|
| Target | RUL capped at 125 |
| Validation | GroupKFold by engine, 5 folds; sensor removal and scaling refitted on the training folds |
| Selection rule | Lowest mean cross-validated RMSE over all cycles of the validation engines |
| Random Forest, fixed | `n_estimators=200`, `random_state=42` |
| Random Forest, tuned | `max_features` in {sqrt, 0.33} x `min_samples_leaf` in {5, 20, 50} = 6 candidates |
| XGBoost, fixed | `n_estimators=300`, `subsample=0.8`, `colsample_bytree=0.8`, `tree_method=hist`, squared-error objective, `random_state=42` |
| XGBoost, tuned | `max_depth` in {3, 5, 7} x `learning_rate` in {0.05, 0.1} = 6 candidates |
| Test protocol | One prediction per test engine at its last observed cycle; test engines used once, after selection |
| Models trained | 360 cross-validation fits + 12 final fits; 16.2 minutes on 16 logical cores |

Full record: `results/metrics/phase5_experiment_config.json`.

Reproducibility: XGBoost predictions repeat exactly. Random Forest builds identical trees
but averages them in parallel, so predictions repeat to within rounding (below 1e-9 cycles).

Saved results are reused when the settings are unchanged (`python -m src.rul_prediction`);
`--force` retrains.

### Selected settings and results (`results/tables/phase5_rul_results_table.csv`)

Test metrics use the capped target. "Uncapped" compares the same predictions with the
original NASA values. Intervals are 95% bootstrap intervals over test engines (2,000
resamples, seed 42).

| Data | Features | Model | Selected | CV RMSE | Test MAE | Test RMSE (95% interval) | Test R2 | Test RMSE uncapped |
|---|---|---|---|---|---|---|---|---|
| FD001 | raw | RF | sqrt, leaf 5 | 18.54 +/- 0.73 | 11.83 | 16.86 (13.68 to 19.82) | 0.823 | 17.86 |
| FD001 | raw | XGB | depth 3, lr 0.05 | 18.65 +/- 0.67 | 11.76 | 16.99 (13.72 to 20.11) | 0.820 | 17.96 |
| FD001 | summary | RF | sqrt, leaf 5 | 17.32 +/- 0.60 | 11.91 | 16.41 (13.48 to 19.35) | 0.832 | 17.54 |
| FD001 | summary | XGB | depth 5, lr 0.05 | 17.11 +/- 0.90 | 11.59 | 15.83 (13.03 to 18.66) | 0.844 | 17.10 |
| FD002 global | raw | RF | sqrt, leaf 5 | 19.53 +/- 0.72 | 12.95 | 17.01 (15.27 to 18.71) | 0.843 | 29.16 |
| FD002 global | raw | XGB | depth 7, lr 0.05 | 19.70 +/- 0.78 | 12.97 | 17.18 (15.38 to 19.01) | 0.840 | 29.06 |
| FD002 global | summary | RF | 0.33, leaf 5 | 19.67 +/- 0.61 | 13.26 | 17.40 (15.66 to 19.14) | 0.836 | 29.68 |
| FD002 global | summary | XGB | depth 7, lr 0.05 | 19.58 +/- 0.65 | 13.09 | 16.92 (15.34 to 18.57) | 0.845 | 28.99 |
| FD002 condition | raw | RF | sqrt, leaf 5 | 19.63 +/- 0.69 | 13.00 | 17.54 (15.55 to 19.62) | 0.833 | 29.50 |
| FD002 condition | raw | XGB | depth 5, lr 0.05 | 19.62 +/- 0.75 | 13.22 | 17.73 (15.81 to 19.80) | 0.829 | 29.57 |
| FD002 condition | summary | RF | sqrt, leaf 5 | 19.34 +/- 1.01 | 12.63 | 16.81 (15.12 to 18.52) | 0.847 | 29.38 |
| FD002 condition | summary | XGB | depth 5, lr 0.05 | 19.04 +/- 1.03 | 11.55 | 15.86 (14.10 to 17.58) | 0.864 | 28.73 |

Across the six candidates of a configuration the mean CV RMSE varies by 0.2 to 1.1 cycles,
similar to the fold-to-fold standard deviation.

### Paired comparisons (`results/metrics/phase5_paired_differences.csv`)

Difference in test RMSE on the same engines, B minus A, with 95% paired bootstrap interval.

| Comparison (A -> B) | Context | Difference | Interval | Excludes 0 |
|---|---|---|---|---|
| RF -> XGB | FD001, raw | +0.13 | -0.57 to 0.79 | no |
| RF -> XGB | FD001, summary | -0.58 | -1.86 to 0.61 | no |
| RF -> XGB | FD002 global, raw | +0.18 | -0.23 to 0.58 | no |
| RF -> XGB | FD002 global, summary | -0.48 | -1.14 to 0.21 | no |
| RF -> XGB | FD002 condition, raw | +0.20 | -0.22 to 0.67 | no |
| RF -> XGB | FD002 condition, summary | -0.96 | -1.69 to -0.31 | yes |
| raw -> summary | FD001, RF | -0.45 | -1.82 to 1.10 | no |
| raw -> summary | FD001, XGB | -1.16 | -3.67 to 1.32 | no |
| raw -> summary | FD002 global, RF | +0.39 | -0.11 to 0.91 | no |
| raw -> summary | FD002 global, XGB | -0.27 | -0.91 to 0.40 | no |
| raw -> summary | FD002 condition, RF | -0.73 | -1.91 to 0.42 | no |
| raw -> summary | FD002 condition, XGB | -1.88 | -3.43 to -0.40 | yes |
| global -> condition | FD002, raw, RF | +0.53 | -0.06 to 1.16 | no |
| global -> condition | FD002, raw, XGB | +0.55 | -0.05 to 1.16 | no |
| global -> condition | FD002, summary, RF | -0.59 | -1.60 to 0.44 | no |
| global -> condition | FD002, summary, XGB | -1.06 | -2.44 to 0.39 | no |

### Findings

1. Both models reach a test RMSE of about 16 to 18 cycles (R2 0.82 to 0.86) on FD001 and FD002
   with the capped target.
2. Random Forest and XGBoost perform alike. With raw sensors they are within 0.2 cycles;
   with summary statistics XGBoost is 0.5 to 1.0 cycles lower.
3. Summary statistics lower test RMSE in five of six cases (0.3 to 1.9 cycles) and raise it
   in one.
4. Per-condition normalisation is about 0.5 cycles worse with raw sensors and 0.6 to 1.1
   cycles better with summary statistics.
5. Two of sixteen intervals exclude zero. No correction for multiple comparisons was
   applied, so these two are suggestive, not conclusive.

### Interpretation notes for Chapters 4 and 5

- **Why per-condition normalisation did not help raw-sensor models.** The Phase 2 to 4
  correlations looked at one sensor at a time. Tree models combine sensors, and under
  dataset-wide normalisation sensors such as s1 and s5 still reveal the operating
  condition, so the trees can separate the regimes themselves. This explanation is
  consistent with the results but was not tested directly.
- **Capped vs uncapped test targets.** On FD002 the uncapped RMSE is about 29 against about
  17 capped, because 57 of 259 test engines have a true RUL above 125 and the models cannot
  predict above about 125 (their mean prediction for those engines is about 110). On FD001
  the gap is about 1 cycle (11 of 100 engines). Published results must be compared with
  the column that matches each paper's convention; this needs checking paper by paper in
  the literature review.
- **CV vs test.** CV scores cover all cycles of validation engines; test scores cover one
  cycle per engine. They are not directly comparable.

### Additions beyond the original plan

- Bootstrap intervals and paired differences, computed from saved predictions with no
  retraining. Added because RQ2 asks whether representation "affects" performance, and
  that cannot be answered from point estimates alone.

### Figures

| File | Content |
|---|---|
| `fig04_actual_vs_predicted_summary` | True vs predicted RUL per test engine, both models, three data configurations (summary features) |
| `fig05_model_comparison` | RF vs XGBoost: RMSE, MAE, R2 with intervals |
| `fig07_representation_comparison` | Raw vs summary: test RMSE with intervals |

Model colours (RF `#1baf7a`, XGBoost `#6250d6`) are used only where the dataset is named in
the panel title, so they never share a chart with the dataset colours.

### Limitations to state

Single random seed; one official train/test split; small hyperparameter grid; simulated data.

### Verification

- `python -m pytest tests -q`: 58 passed (11 new: 8 procedure checks, 3 for intervals).
- Notebook 03 executed end to end with no errors; 3 figures rendered.

## Phase 6 - Anomaly detection: Experiment 2 (2026-11-10)

Code: `src/anomaly_detection.py`, `src/evaluation.py`. Notebook:
`notebooks/04_anomaly_detection.ipynb` (executed). Run time: about 36 seconds.

### Configuration (`results/metrics/phase6_experiment_config.json`)

| Item | Setting |
|---|---|
| Model | Isolation Forest, `n_estimators=200`, `max_samples=256`, `random_state=42`; not tuned |
| Healthy training data | Training-engine cycles with RUL >= 125 |
| Alarm threshold | Quantile of the healthy training scores so that 5% of healthy cycles are flagged; no labels used |
| Reference label | Cycle is anomalous if RUL <= 30 (constructed proxy for late degradation) |
| Scoring 1, "all cycles" | Every cycle is scored |
| Scoring 2, "healthy vs near failure" | Transition zone (RUL 31 to 124) left out |
| Evaluation sets | 5-fold GroupKFold on training engines (out-of-fold, whole trajectories); official test engines |
| Inputs | Same sensors and the same two representations as the RUL models |
| Warm-up | Summary representation only: learns from and alarms on cycles >= 30 |

### Decisions

| # | Decision | Reason | Alternatives considered |
|---|---|---|---|
| 1 | Train on healthy cycles only | The detector is meant to learn normal behaviour; fitting on all cycles would treat degradation as normal | Fit on all training cycles with a contamination rate |
| 2 | Fixed 5% alarm budget, set before computing any result | Keeps the detector fully unsupervised and avoids tuning on labels | Choosing the threshold by F1 on labelled engines, which uses failure labels |
| 3 | Reference label RUL <= 30 | Carried over from the earlier project plan; a common proxy for "close to failure" | Other cut-offs; not swept, to keep the scope small |
| 4 | Report a second scoring without the transition zone | The 30-cycle boundary is arbitrary and degradation is gradual, so alarms just outside it are not clearly wrong | Reporting "all cycles" only |
| 5 | Evaluate on training engines by cross-validation as well as on the test engines | Test engines stop early, so only 2.5 to 3.2% of their cycles are near failure; run-to-failure engines give the full picture | Test engines only |
| 6 | Run the detector on all three data configurations and both representations | Uses the configurations already built; shows whether preprocessing matters for this model too (RQ2) | One configuration per dataset |
| 7 | Warm-up for the summary representation | In cross-validation on training engines 14 to 15% of healthy cycles before cycle 30 were flagged against under 1% afterwards; half-filled windows look unusual for reasons unrelated to health | No warm-up, which lets window artefacts set the threshold |
| 8 | Keep the quantised sensors s6, s10, s16 as inputs | In cross-validation on training engines removing them changed F1 by at most 0.03 (slightly lower without them) | Dropping them, which would need an extra rule |

Decisions 7 and 8 were made from cross-validation on training engines only. The official
test engines were scored once, after the design was fixed.

### Results at the 5% budget (`results/tables/phase6_anomaly_results_table.csv`)

Scoring "all cycles":

| Data | Features | CV precision | CV recall | CV F1 | Test precision | Test recall | Test F1 |
|---|---|---|---|---|---|---|---|
| FD001 | raw | 0.439 | 0.997 | 0.610 | 0.193 | 0.985 | 0.323 |
| FD001 | summary | 0.430 | 0.997 | 0.601 | 0.181 | 1.000 | 0.307 |
| FD002 global | raw | 0.491 | 0.797 | 0.607 | 0.199 | 0.747 | 0.315 |
| FD002 global | summary | 0.215 | 0.074 | 0.110 | 0.063 | 0.078 | 0.070 |
| FD002 condition | raw | 0.466 | 0.978 | 0.631 | 0.196 | 0.959 | 0.325 |
| FD002 condition | summary | 0.448 | 0.987 | 0.617 | 0.187 | 0.969 | 0.313 |

Scoring "healthy vs near failure":

| Data | Features | CV precision | CV recall | CV F1 | Test precision | Test recall | Test F1 |
|---|---|---|---|---|---|---|---|
| FD001 | raw | 0.873 | 0.997 | 0.931 | 0.459 | 0.985 | 0.626 |
| FD001 | summary | 0.885 | 0.997 | 0.937 | 0.476 | 1.000 | 0.645 |
| FD002 global | raw | 0.853 | 0.797 | 0.824 | 0.463 | 0.747 | 0.572 |
| FD002 global | summary | 0.458 | 0.074 | 0.127 | 0.109 | 0.078 | 0.091 |
| FD002 condition | raw | 0.878 | 0.978 | 0.925 | 0.518 | 0.959 | 0.672 |
| FD002 condition | summary | 0.908 | 0.987 | 0.946 | 0.652 | 0.969 | 0.779 |

Share of anomalous cycles: 15.0% of all cycles in cross-validation; 2.5% (FD001) and 3.2%
(FD002) of test cycles. Near-failure cycles on the test sets: 332 from 25 engines (FD001)
and 1,087 from 61 engines (FD002).

Where test alarms fall at the 5% budget (FD001 and FD002 condition): 58 to 71% in the
transition zone, 10 to 23% on healthy cycles, 18 to 20% near failure. The share of healthy
test cycles flagged is close to the 5% budget.

Alarm-budget sensitivity, test, all cycles (F1 at 1% / 5% / 10%): FD001 raw 0.50 / 0.32 /
0.24; FD001 summary 0.45 / 0.31 / 0.24; FD002 condition raw 0.47 / 0.33 / 0.25; FD002
condition summary 0.40 / 0.31 / 0.27. The 5% value is the reported result; the others were
not used to choose a setting.

### Findings

1. Trained on healthy data only, the detector flags 96 to 100% of near-failure test cycles
   on FD001 and on FD002 with per-condition normalisation.
2. Precision against the 30-cycle label is about 0.19 on the test engines (F1 about 0.32).
   The alarm rate rises gradually as RUL falls (about 80 to 90% at RUL 31 to 40), so most
   "false alarms" are on engines already degrading.
3. Preprocessing matters strongly for this model. On FD002, per-condition normalisation
   raises test recall from 0.75 to 0.96 (raw) and from 0.08 to 0.97 (summary).
4. With suitable normalisation, raw and summary inputs give similar scores on all cycles.
   Without the transition zone, summary statistics give higher test precision on FD002
   condition (0.65 against 0.52).
5. Cross-validation and test scores differ mainly because of the share of anomalous cycles
   (15% against about 3%), which drives precision.

### Interpretation notes for Chapters 4 and 5

- Contrast with Phase 5: per-condition normalisation barely changed the supervised tree
  models but is decisive for the unsupervised detector. A supervised model can learn to
  separate operating conditions from the labels; an unsupervised one has no labels and
  treats regime changes as the main source of variation.
- The reference label is a proxy. The results show separation of late degradation from
  healthy operation in simulated data, not detection of real faults.
- No uncertainty intervals were computed for the detection scores. Cycles within an
  engine are strongly correlated, so an engine-level bootstrap would be needed.

### Figures

| File | Content |
|---|---|
| `fig06a_anomaly_score_trajectories` | Anomaly score over the observed life of the test engine closest to failure, per data configuration and representation |
| `fig06b_alarm_rate_by_rul` | Share of cycles flagged in bins of remaining life, cross-validation and test |

### Verification

- `python -m pytest tests -q`: 66 passed (8 new for anomaly detection).
- Notebook 04 executed end to end with no errors; 2 figures rendered.

## Phase 7 - Results consolidation (2026-11-10)

Code: `src/results_summary.py`. Notebook: `notebooks/05_results_analysis.ipynb` (executed).
This phase reads the saved result files only. No model was trained and no number was
typed by hand.

### Outputs

- `results/tables/thesis/table01` to `table10` (CSV): the tables for Chapter 4.
- `reports/chapter4_tables_and_figures.md`: all ten tables with captions and the figure
  list, generated from the CSV files. Regenerate with `python -m src.results_summary`.
- `results/figures/fig08_dataset_comparison`: FD001 vs FD002 for RUL prediction (capped
  and uncapped test RMSE) and anomaly detection (recall and F1).

### Required visualisations: all present (PNG and PDF)

| Requirement | File(s) |
|---|---|
| 1. Dataset distribution | `fig01_dataset_distribution` |
| 2. Sensor trends over operating cycles | `fig02a_sensor_trends_FD001`, `fig02b_operating_condition_effect` |
| 3. RUL degradation curves | `fig03a_rul_target` |
| 4. Actual vs predicted RUL | `fig04_actual_vs_predicted_summary` |
| 5. Random Forest vs XGBoost | `fig05_model_comparison` |
| 6. Anomaly detection | `fig06a_anomaly_score_trajectories`, `fig06b_alarm_rate_by_rul` |
| 7. Feature representation comparison | `fig07_representation_comparison` |
| 8. FD001 vs FD002 | `fig08_dataset_comparison` |
| Supporting (methodology) | `fig03b_normalisation_effect_FD002`, `fig03c_summary_features_example` |

### FD001 vs FD002 (official test engines; range over model and feature combinations)

| Metric | FD001 | FD002 dataset-wide | FD002 per-condition |
|---|---|---|---|
| RUL test RMSE, capped | 15.83 to 16.99 | 16.92 to 17.40 | 15.85 to 17.73 |
| RUL test RMSE, uncapped | 17.10 to 17.95 | 28.99 to 29.68 | 28.73 to 29.57 |
| RUL test MAE, capped | 11.59 to 11.91 | 12.95 to 13.26 | 11.55 to 13.22 |
| RUL test R2, capped | 0.82 to 0.84 | 0.84 to 0.84 | 0.83 to 0.86 |
| RUL cross-validated RMSE | 17.11 to 18.65 | 19.53 to 19.70 | 19.04 to 19.63 |
| Anomaly test recall | 0.985 to 1.000 | 0.078 to 0.747 | 0.959 to 0.969 |
| Anomaly test F1, all cycles | 0.307 to 0.323 | 0.070 to 0.315 | 0.313 to 0.325 |
| Anomaly test F1, healthy vs near failure | 0.626 to 0.645 | 0.091 to 0.572 | 0.672 to 0.779 |

### Evidence for the research questions

**RQ1 (accuracy on single- and multi-condition data).**

- RUL: with the capped target, test RMSE is 15.8 to 17.0 cycles on FD001 and 15.9 to 17.7
  on FD002; R2 is 0.82 to 0.86. Random Forest and XGBoost perform alike.
- RUL against uncapped NASA values: about 17 to 18 cycles on FD001 and about 29 on FD002.
  The FD002 gap comes from the share of test engines above the cap (22% against 11%).
- RUL in cross-validation (all cycles): FD002 is about 1 to 2.5 cycles worse than FD001.
- Anomaly detection: test recall 0.985 to 1.00 on FD001 and 0.96 to 0.97 on FD002 with
  per-condition normalisation; precision about 0.18 to 0.20 and F1 about 0.31 to 0.33
  against the 30-cycle label when all cycles are scored.

**RQ2 (preprocessing and feature representation).**

- Normalisation scheme on FD002: little effect on the supervised models (+0.5 cycles RMSE
  with raw sensors, -0.6 to -1.1 with summary statistics; all intervals include zero) and
  a decisive effect on the unsupervised detector (test recall from 0.75 to 0.96 with raw
  sensors and from 0.08 to 0.97 with summary statistics).
- Feature representation: summary statistics lower RUL test RMSE in five of six cases
  (0.3 to 1.9 cycles); one interval excludes zero. For the detector the two representations
  are similar when the normalisation is suitable.
- The effect of preprocessing depends on the model: supervised tree models are robust to
  the normalisation scheme, the unsupervised detector is not.
- Theoretical expectation, not tested: rescaling a whole column (min-max or dataset-wide
  z-score) keeps the order of values, so tree models make the same splits with or without
  it.

### Cautions for Chapter 5

- FD001 and FD002 differ in size as well as in operating conditions (100 against 260
  training engines), so differences between them cannot be attributed to operating
  conditions alone.
- The capped-versus-uncapped gap is a property of the evaluation convention and the test
  sets. It must not be presented as evidence that multi-condition data is harder.
- Simulated data; one seed; one train/test split; 16 paired comparisons without
  correction; no intervals for the detection scores; reference label and alarm budget are
  choices.

### Verification

- `python -m pytest tests -q`: 72 passed (6 new; they check that the thesis tables equal
  the saved experiment results and that every required figure exists).
- Notebook 05 executed end to end with no errors; 4 figures rendered.
- All five notebooks are executed and saved with outputs.

## Phase 8 - Thesis writing (started 2026-11-10)

Author decisions: chapters delivered as Word files; IEEE numbered citations unless the
university requires another style. Writing order: Chapters 3, 4, 2, 1, 5.

### How the Word files are produced

- Each chapter is a content file in `reports/thesis/build/` (`chapter3.js`, ...). The
  library `lib.js` turns it into a .docx and numbers sections, figures, tables, equations
  and citations automatically. Build with `node chapter3.js` from that folder
  (requires Node.js; `npm install` restores the `docx` package).
- Figures are embedded from `results/figures/`, so a regenerated figure is picked up by
  rebuilding the chapter.
- References are stored once in `build/references.json`. Each chapter currently ends with
  its own reference list; numbers will be unified when the chapters are assembled.
- Layout assumptions (no university template was supplied): A4, 2.54 cm margins, Times New
  Roman 12 pt, 1.5 line spacing, justified text, table captions above and figure captions
  below, page numbers centred in the footer.
- Layout is checked by converting the file to PDF with Microsoft Word
  (`build/docx_to_pdf.ps1`) and inspecting the pages.

### Reference policy

- Every reference is checked before it is cited; see `reports/thesis/reference_verification.md`.
- A reference is cited only for statements checked against a source that was read.
  Where only the abstract or a secondary source was available, the statement in the
  thesis is limited accordingly.

### Chapter 3 - Research Methodology (2026-11-10)

- File: `reports/thesis/Chapter_3_Research_Methodology.docx`; 19 pages, about 6,000 words
  including tables; 12 sections, 12 numbered equations, 7 tables, 4 figures, 9 references.
- New figure: `fig00_research_framework` (research framework diagram).
- Content is drawn from this log. Deviations from the original plan (normalisation
  comparison, Isolation Forest on all six cases, bootstrap intervals, warm-up for
  summary features) are stated in the chapter.
- Software table lists only libraries the code uses directly. Seaborn and SciPy are
  installed but not called by the project code, so they are not listed.
- Layout checked page by page in the Word-generated PDF; equations render correctly.

### Chapter 4 - Results and Analysis (2026-11-10)

- File: `reports/thesis/Chapter_4_Results_and_Analysis.docx`; 23 pages, about 6,700 words
  including tables; 7 sections, 17 tables, 9 figures, no external references.
- Every table is filled by the build script from `results/tables/thesis/*.csv`, so no
  table value was typed by hand. Numbers quoted in the running text were checked one by
  one against those files.
- Result files added so that every statement in the chapter can be reproduced:
  - `results/metrics/phase6_design_check_warm_up.csv` and `phase6_design_check_sensors.csv`:
    the cross-validation evidence (training engines only) for the warm-up rule and for
    keeping sensors s6, s10 and s16. These had been computed during Phase 6 but not saved.
    `python -m src.anomaly_detection` now writes them (run time about 105 seconds).
  - Thesis tables 11 to 20 in `results/tables/thesis/` (data quality, per-sensor
    exploration, input signal, tuning range, alarm zones, alarm-budget sensitivity, the two
    design checks, predictions above the cap, alarm rate by remaining life).
- Corrections to earlier wording, made while checking the chapter against the data:
  - Share of healthy test cycles flagged is 2.7% to 4.8%, "at or below" the 5% budget
    (the Phase 6 entry said "close to").
  - Inside the reference zone at least 92% of cycles are flagged in every 10-cycle bin of
    the four working cases (lowest value 92.8%).
- Structure follows the thesis brief: exploration, RUL prediction, anomaly detection,
  model comparison, feature representation and preprocessing comparison, FD001 vs FD002,
  summary.

### Chapter 2 - Literature Review (2026-11-10)

- File: `reports/thesis/Chapter_2_Literature_Review.docx`; 14 pages, about 5,800 words
  including references; 10 sections, 2 tables, 35 references.
- Sources were found and checked with `build/lookup_reference.py` (Crossref for
  bibliographic details; OpenAlex and Semantic Scholar for abstracts) and by reading open
  full texts where available. `reports/thesis/reference_verification.md` is now generated
  from `build/references.json` by `node make_verification_log.js` and lists every
  reference with how it was checked.
- Depth of checking: full text read for 5 sources (Saxena et al. 2008, Ramasso and Saxena
  2014, Laredo et al. 2019 preprint, Hou et al. 2020, Goel et al. 2026); abstract only for
  most others; 2 sources (Li et al. 2018, Babu et al. 2016) not read and described only
  through other papers. Statements are limited accordingly.
- Table 2.1 (reported RMSE of earlier studies) takes four rows from the comparison table of
  Hou et al. (2020), not from the original papers; the table note states this.
- Five research gaps are stated with respect to the reviewed sources only, and Table 2.2
  maps each to how the study addresses it.
### Chapters 1 and 5 (2026-11-10)

- Files: `reports/thesis/Chapter_1_Introduction.docx` (6 pages, 8 sections, 10 references)
  and `reports/thesis/Chapter_5_Discussion_and_Conclusion.docx` (10 pages, 7 sections,
  1 table, 8 references). No new sources: both cite only references already verified.
- Chapter 1 states five research objectives derived from the thesis brief and gives the
  two research questions word for word.
- Chapter 5: Table 5.1 is computed by the build script from
  `results/tables/thesis/table10_dataset_comparison.csv`. The comparison with the
  literature states that the conventions of the published studies could not be confirmed,
  so no claim of outperforming them is made.
- All five chapters now exist as separate Word files. Not yet done: one combined document
  with title page, abstract, table of contents, lists of figures and tables, and a single
  reference list with unified numbering.

- Useful for Chapter 5: published FD002 errors are larger than FD001 errors in every
  reviewed study (19.5 to 30.3 against 10.7 to 18.5), and the conventions behind those
  numbers differ, so comparison with this thesis must be made with care.
