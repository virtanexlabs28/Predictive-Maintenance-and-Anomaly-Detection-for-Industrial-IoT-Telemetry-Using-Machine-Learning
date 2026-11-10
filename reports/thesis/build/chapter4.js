/* Chapter 4 - Results and Analysis. Run with:  node chapter4.js
 *
 * Every table is filled from the result files in results/tables/thesis, so
 * the numbers in the tables are never typed by hand.
 */

const { buildChapter, readCsv, format } = require("./lib");
const { fixed, signed, count } = format;

const T = (name) => readCsv(`results/tables/thesis/${name}.csv`);
const yesNo = (value) => (String(value).toLowerCase() === "true" ? "Yes" : "No");
const titleCase = (text) => text.charAt(0).toUpperCase() + text.slice(1);

// ------------------------------------------------------------- table rows
const quality = T("table11_data_quality").map((r) => [
  r.subset, titleCase(r.split), count(r.rows), r.missing_values, r.non_finite_values, r.duplicate_rows, r.duplicate_unit_cycle,
]);

const conditions = T("table02_operating_conditions").map((r) => {
  const [altitude, mach, throttle] = r["altitude_kft / mach / throttle"].split(" / ");
  return [r.subset, altitude, mach, throttle, count(r.rows), fixed(r.share_percent, 1)];
});

const sensors = T("table12_sensor_exploration").map((r) => [
  r.sensor, r.symbol,
  count(r.fd001_unique_values), r.fd001_spearman === "" ? "constant" : signed(r.fd001_spearman, 2),
  count(r.fd002_unique_values), fixed(r.fd002_variance_from_condition, 3), signed(r.fd002_spearman, 2),
  r.fd002_spearman_within_condition === "" ? "constant" : signed(r.fd002_spearman_within_condition, 2),
]);

const signal = T("table13_input_signal").map((r) => [
  r.data, r.statistic, fixed(r.median_abs_spearman, 3), fixed(r.max_abs_spearman, 3), r.strongest_feature.replace("_", " "),
]);

const tuning = Object.fromEntries(T("table14_tuning_range").map((r) => [`${r.data}|${r.representation}|${r.model}`, r]));
const selected = T("table06_selected_settings").map((r) => {
  const range = tuning[`${r.data}|${r.representation}|${r.model}`];
  return [
    r.data, r.representation, r.model, r.selected_settings.replace(/_/g, " "),
    `${fixed(r.cv_rmse_mean)} ± ${fixed(r.cv_rmse_std)}`, fixed(range.worst), fixed(range.range),
  ];
});

const rul = T("table07_rul_results").map((r) => [
  r.data, r.representation, r.model, fixed(r.test_mae), fixed(r.test_rmse), r.test_rmse_95ci, fixed(r.test_r2, 3),
  fixed(r.test_mae_uncapped), fixed(r.test_rmse_uncapped), fixed(r.test_r2_uncapped, 3),
]);

const aboveCap = T("table19_above_cap").map((r) => [
  r.data, r.representation, r.model, r.engines_above_cap, fixed(r.mean_true_rul, 1), fixed(r.mean_predicted_rul, 1), fixed(r.lowest_predicted_rul, 1),
]);

const warmUp = T("table17_design_check_warm_up").map((r) => [
  r.data, count(r.healthy_cycles_before_30), fixed(r.flagged_percent_before_30, 1), count(r.healthy_cycles_from_30), fixed(r.flagged_percent_from_30, 1),
]);

const quantised = T("table18_design_check_sensors").map((r) => [
  r.data, r.representation, titleCase(r.inputs), fixed(r.precision, 3), fixed(r.recall, 3), fixed(r.f1, 3),
]);

const anomalyRows = (scoring) => T("table09_anomaly_results").filter((r) => r.scoring === scoring).map((r) => [
  r.data, r.representation, fixed(r.cv_precision, 3), fixed(r.cv_recall, 3), fixed(r.cv_f1, 3),
  fixed(r.test_precision, 3), fixed(r.test_recall, 3), fixed(r.test_f1, 3),
]);

const zones = T("table15_alarm_zones").map((r) => [
  r.data, r.representation, count(r.alarms), fixed(r.alarms_healthy_percent, 1), fixed(r.alarms_transition_percent, 1),
  fixed(r.alarms_near_failure_percent, 1), fixed(r.healthy_cycles_flagged_percent, 1),
]);

const budget = T("table16_budget_sensitivity").map((r) => [
  r.data, r.representation,
  fixed(r.precision_1pct, 3), fixed(r.precision_5pct, 3), fixed(r.precision_10pct, 3),
  fixed(r.recall_1pct, 3), fixed(r.recall_5pct, 3), fixed(r.recall_10pct, 3),
  fixed(r.f1_1pct, 3), fixed(r.f1_5pct, 3), fixed(r.f1_10pct, 3),
]);

const paired = (kind) => T("table08_paired_differences").filter((r) => r.comparison === kind).map((r) => [
  titleCase(r.context.replace(/ normalisation/g, "").replace(/; /g, ", ")),
  fixed(r.rmse_A), fixed(r.rmse_B), signed(r.difference_B_minus_A),
  `${fixed(r.ci_low)} to ${fixed(r.ci_high)}`, yesNo(r.interval_excludes_zero),
]);

const comparison = T("table10_dataset_comparison")
  .filter((r) => ["Test RMSE, capped target", "Test RMSE, uncapped target", "Cross-validated RMSE", "Test recall", "Test F1, all cycles", "Test F1, healthy vs near failure"].includes(r.metric))
  .map((r) => {
    const decimals = r.task === "RUL prediction" ? 2 : 3;
    return [r.model, r.features, r.metric, fixed(r.FD001, decimals), fixed(r["FD002 dataset-wide"], decimals), fixed(r["FD002 per-condition"], decimals)];
  });

// ---------------------------------------------------------------- content
const blocks = [
  { p: `This chapter presents the results of the study. It begins with the exploration of the data, which provides the evidence for the preprocessing decisions described in Chapter 3. It then reports the results of RUL prediction and of anomaly detection, followed by the three comparisons required by the research questions: between the two RUL models, between the feature representations and normalisation schemes, and between FD001 and FD002. All values were produced by the project code. Interpretation is kept to what the results show; their meaning for the research questions is discussed in Chapter 5.` },

  // ------------------------------------------------------------------ 4.1
  { h: 2, text: "Dataset Exploration Results" },
  { p: `The exploration describes the data before any change is made to them. Statistics of the sensors were computed from the training engines only. The test files were inspected only for their structure and for the published RUL values.` },

  { h: 3, text: "Data quality" },
  { p: `{ref:tab-quality} gives the results of the quality checks. None of the four files contains a missing value, a non-finite value, a duplicated row or a repeated combination of engine and cycle. No values had to be imputed and no rows were removed.` },
  { table: {
    id: "tab-quality",
    caption: "Data quality checks on the four files.",
    header: ["Subset", "Split", "Rows", "Missing values", "Non-finite values", "Duplicated rows", "Repeated engine and cycle"],
    widths: [0.9, 0.8, 1.0, 1.1, 1.2, 1.2, 1.6],
    align: ["left", "left", "right", "right", "right", "right", "right"],
    rows: quality,
  } },

  { h: 3, text: "Engine lifetimes and test-set RUL" },
  { p: `{ref:fig-distribution} shows how long the engines run. The training engines, which are run until failure, last between 128 and 362 cycles in FD001 and between 128 and 378 cycles in FD002. The median lifetime is 199 cycles in both subsets. The test engines are cut off before failure, so their records are shorter and more varied: between 31 and 303 cycles in FD001 and between 21 and 367 cycles in FD002.` },
  { p: `The true RUL of the test engines at their last recorded cycle ranges from 7 to 145 cycles in FD001 (median 86) and from 6 to 194 cycles in FD002 (median 80). It exceeds the 125-cycle cap for 11 of the 100 FD001 test engines and for 57 of the 259 FD002 test engines, that is, for 11% and 22% of the engines. This difference between the two test sets matters for the comparison of capped and uncapped results in Section 4.2.` },
  { figure: { id: "fig-distribution", file: "fig01_dataset_distribution.png", caption: "Lifetimes of the training engines, observed length of the test engines and true RUL of the test engines at their last recorded cycle, for FD001 and FD002." } },

  { h: 3, text: "Operating conditions" },
  { p: `{ref:tab-conditions} lists the operating conditions found by rounding the three operational settings. FD001 has one condition and FD002 has six, in agreement with the documentation of the dataset. In FD002 five of the conditions each account for about 15% of the rows and one accounts for 25%. Every operating condition that occurs in the test files also occurs in the training files.` },
  { table: {
    id: "tab-conditions",
    caption: "Operating conditions in the training data.",
    header: ["Subset", "Altitude (1,000 ft)", "Mach number", "Throttle resolver angle", "Rows", "Share of rows (%)"],
    widths: [1.0, 1.4, 1.2, 1.7, 1.0, 1.3],
    align: ["left", "right", "right", "right", "right", "right"],
    rows: conditions,
    note: "Values are the three operational settings after rounding (Section 3.4).",
  } },

  { h: 3, text: "Sensor behaviour" },
  { p: `{ref:fig-sensors} shows all 21 sensors over the life of five FD001 training engines, and {ref:tab-sensors} gives statistics for every sensor in both subsets. The table reports the Spearman rank correlation between each sensor and the number of cycles remaining before failure. A value far from zero means that the sensor moves steadily in one direction as the engine wears; a negative value means that the reading rises as failure approaches.` },
  { figure: { id: "fig-sensors", file: "fig02a_sensor_trends_FD001.png", widthCm: 14.2, caption: "The 21 sensor measurements over the operating cycles of five FD001 training engines, chosen from the shortest-lived to the longest-lived. Each line is one engine from its first cycle until failure." } },
  { table: {
    id: "tab-sensors",
    caption: "Exploration statistics for each sensor (training engines).",
    header: ["Sensor", "Symbol", "FD001: distinct values", "FD001: correlation", "FD002: distinct values", "FD002: variance share of condition", "FD002: correlation, raw", "FD002: correlation, within condition"],
    widths: [0.7, 1.1, 1.0, 1.1, 1.0, 1.3, 1.1, 1.3],
    align: ["left", "left", "right", "right", "right", "right", "right", "right"],
    fontSize: 18,
    rows: sensors,
    note: "Correlation: Spearman rank correlation with the number of cycles remaining before failure. Variance share of condition: share of the variance of the sensor that is explained by the operating condition alone. Within condition: correlation after subtracting the mean of each operating condition. \"constant\": the sensor does not vary (in FD002, not within any operating condition).",
  } },
  { p: `Three observations follow for FD001. First, six sensors (s1, s5, s10, s16, s18 and s19) take a single value and appear as flat lines in {ref:fig-sensors}. They carry no information. Second, sensor s6 takes only two values (21.60 and 21.61) and is almost unrelated to the remaining life, with a correlation of −0.13. Third, the other sensors stay roughly level during the early life of an engine and then drift as failure approaches. The strongest relations with remaining life are found for s11 (−0.72), s4 (−0.70), s12 (+0.69) and s7 (+0.68). Sensors s9 and s14 are weaker (−0.32 and −0.20), and {ref:fig-sensors} shows that they behave differently from one engine to another.` },
  { p: `The level-then-drifting shape supports the choice of a capped RUL target. In the early life of an engine the sensors do not yet show how much life remains, so a target that keeps decreasing through that period asks the model to predict something the inputs do not contain.` },

  { h: 3, text: "Effect of operating conditions in FD002" },
  { p: `In FD002 the picture is different. No sensor is constant over the whole file. Four sensors (s1, s5, s18 and s19) are constant within every operating condition, which means that they only repeat which condition the engine is in. For all sensors except s14 and s16, more than 99% of the variance is explained by the operating condition alone ({ref:tab-sensors}); the median over the 21 sensors is 0.9998. As a consequence the raw correlation with remaining life is weak for every sensor. The largest absolute value is 0.25 (s13), against 0.72 in FD001.` },
  { p: `{ref:fig-conditions} shows why. In FD001 the wear trend is visible directly in the readings. In FD002 the same sensors jump from one level to another at every cycle as the engine changes flight regime, and the jumps are far larger than the wear trend. After the mean of each operating condition is subtracted, the correlations in FD002 rise to values close to those of FD001; the largest becomes 0.71 (s11). The degradation signal is therefore present in FD002 but hidden behind the changes of regime.` },
  { figure: { id: "fig-conditions", file: "fig02b_operating_condition_effect.png", caption: "Four sensors over the life of one training engine in FD001 (left) and in FD002 (right). The engine with the median lifetime was chosen in each subset. Note the different ranges of the vertical axes." } },

  { h: 3, text: "Information in the model inputs" },
  { p: `The same question can be asked of the inputs that the models receive after preprocessing. {ref:tab-signal} gives, for each kind of input, the absolute rank correlation with the capped RUL on the training engines, summarised over the retained sensors.` },
  { table: {
    id: "tab-signal",
    caption: "Absolute Spearman rank correlation of the model inputs with the capped RUL (training engines).",
    header: ["Data configuration", "Kind of input", "Median over sensors", "Maximum", "Strongest input"],
    widths: [2.0, 2.3, 1.2, 1.0, 1.2],
    align: ["left", "left", "right", "right", "left"],
    rows: signal,
  } },
  { p: `Two points stand out. Under per-condition normalisation the median correlation of the raw readings in FD002 is 0.454, against 0.078 under dataset-wide normalisation and 0.635 in FD001. Second, the 30-cycle mean is more strongly related to the RUL than the single reading in FD001 (median 0.707 against 0.635) and in FD002 with per-condition normalisation (0.688 against 0.454). In FD002 with dataset-wide normalisation no statistic helps, because the window averages readings taken at different operating conditions.` },
  { p: `These correlations describe one input at a time on the training data. They are not model results. A model combines many inputs, and Sections 4.2 to 4.5 show that the picture given by {ref:tab-signal} carries over to the unsupervised detector but only partly to the supervised models.` },

  // ------------------------------------------------------------------ 4.2
  { h: 2, text: "RUL Prediction Results" },

  { h: 3, text: "Cross-validation and selected settings" },
  { p: `{ref:tab-selected} gives the settings selected by engine-level cross-validation for each of the twelve combinations of data configuration, feature representation and model, together with the cross-validated RMSE. The table also shows the mean RMSE of the worst of the six candidates and the difference between the worst and the best.` },
  { table: {
    id: "tab-selected",
    caption: "Selected settings and cross-validated RMSE (mean ± standard deviation over five folds, in cycles).",
    header: ["Data configuration", "Features", "Model", "Selected settings", "RMSE of selected", "RMSE of worst", "Range"],
    widths: [1.6, 1.3, 1.1, 2.3, 1.2, 0.8, 0.7],
    align: ["left", "left", "left", "left", "right", "right", "right"],
    fontSize: 18,
    rows: selected,
    note: "Scores cover all cycles of the validation engines against the capped target. Range: worst minus best mean RMSE over the six candidates.",
  } },
  { p: `The cross-validated RMSE lies between 17.1 and 19.7 cycles. Three points can be noted. First, the choice of settings matters little within the grids used: the difference between the best and the worst candidate is between 0.2 and 1.1 cycles, which is of the same order as the standard deviation between folds (0.6 to 1.0 cycles). Second, the selected settings are similar across cases. Random Forest selects the smallest leaf size in every case, and XGBoost selects the lower learning rate in every case. Third, the summary-statistic features lower the cross-validated RMSE on FD001 by 1.2 to 1.5 cycles and on FD002 with per-condition normalisation by 0.3 to 0.6 cycles, and leave it practically unchanged on FD002 with dataset-wide normalisation.` },

  { h: 3, text: "Results on the official test engines" },
  { p: `{ref:tab-rul} reports the results on the official test engines, with one prediction per engine at its last recorded cycle. The left part of the table uses the capped target, which is the main result. The right part compares the same predictions with the uncapped published values.` },
  { table: {
    id: "tab-rul",
    caption: "RUL prediction results on the official test engines.",
    header: ["Data configuration", "Features", "Model", "MAE", "RMSE", "95% interval of RMSE", "R²", "MAE, uncapped", "RMSE, uncapped", "R², uncapped"],
    widths: [1.5, 1.3, 1.1, 0.7, 0.7, 1.3, 0.7, 0.8, 0.8, 0.8],
    align: ["left", "left", "left", "right", "right", "center", "right", "right", "right", "right"],
    fontSize: 17,
    rows: rul,
    note: "MAE and RMSE in cycles. Interval: 95% bootstrap interval over test engines (2,000 resamples). FD001: 100 test engines. FD002: 259 test engines.",
  } },
  { p: `With the capped target the test RMSE lies between 15.83 and 17.73 cycles, the MAE between 11.55 and 13.26 cycles and R² between 0.820 and 0.864 across the twelve cases. The lowest RMSE on FD001 is 15.83 cycles (XGBoost with summary statistics) and the lowest on FD002 is 15.86 cycles (XGBoost with summary statistics and per-condition normalisation).` },
  { p: `The bootstrap intervals are wide. For FD001, with 100 test engines, they span about six cycles; for FD002, with 259 test engines, between three and four cycles. The intervals of all twelve cases overlap. Differences between cases therefore have to be examined with the paired comparisons of Sections 4.4 and 4.5, which use the same engines for both cases and are more sensitive than a comparison of separate intervals.` },
  { p: `{ref:fig-actual} plots the predicted against the true RUL for every test engine, using the summary-statistic features. Predictions are close for engines near failure and more scattered for engines with a longer remaining life. The vertical column of points at 125 contains the engines whose true RUL is 125 cycles or more.` },
  { figure: { id: "fig-actual", file: "fig04_actual_vs_predicted_summary.png", caption: "Predicted against true RUL for the official test engines (summary-statistic features). Each point is one engine; the dashed line marks a perfect prediction. True RUL is capped at 125." } },
  { p: `The test scores are lower than the cross-validation scores in every case. The two are not directly comparable: cross-validation scores cover every cycle of the validation engines, whereas test scores cover one cycle per engine.` },

  { h: 3, text: "Capped and uncapped targets" },
  { p: `Against the uncapped values the results change in different ways for the two datasets ({ref:tab-rul}). On FD001 the RMSE rises by about 1.0 to 1.3 cycles, to between 17.10 and 17.95 cycles. On FD002 it rises to between 28.73 and 29.68 cycles, and R² falls from about 0.84 to about 0.70.` },
  { p: `{ref:tab-above} explains the difference. It considers only the test engines whose true RUL is above the cap. Their true RUL is 135 cycles on average in FD001 and 159 cycles in FD002, but the models predict about 111 to 115 cycles for the former and about 110 to 111 cycles for the latter. A model trained on a target that never exceeds 125 cannot predict a longer life. The resulting error is small in FD001, where 11 engines are affected and their true RUL is at most 145 cycles. It is large in FD002, where 57 engines are affected and the true RUL reaches 194 cycles.` },
  { table: {
    id: "tab-above",
    caption: "Predictions for the test engines whose true RUL is above the cap of 125 cycles.",
    header: ["Data configuration", "Features", "Model", "Engines", "Mean true RUL", "Mean predicted RUL", "Lowest predicted RUL"],
    widths: [1.6, 1.4, 1.2, 0.8, 1.0, 1.2, 1.2],
    align: ["left", "left", "left", "right", "right", "right", "right"],
    fontSize: 18,
    rows: aboveCap,
  } },
  { p: `The gap between the capped and uncapped results on FD002 is therefore a consequence of the evaluation convention and of the composition of the test set. It should not be read as evidence that prediction is harder under several operating conditions. It also means that results from different studies can be compared only when they use the same convention.` },

  // ------------------------------------------------------------------ 4.3
  { h: 2, text: "Anomaly Detection Results" },

  { h: 3, text: "Design checks on the training engines" },
  { p: `Two design decisions for the detector were based on cross-validation on the training engines. The official test engines were not used for either decision.` },
  { p: `The first concerns the summary-statistic features. {ref:tab-warmup} shows what happens when the detector is allowed to raise alarms from the first cycle. Among the healthy cycles before cycle 30, between 13.9% and 14.8% are flagged. Among the healthy cycles from cycle 30 onwards, fewer than 1% are flagged. The alarm budget of 5% is thus spent almost entirely on the first cycles of each engine, where the 30-cycle window is still filling and its statistics look unusual for reasons unrelated to engine health. For this reason the detector that uses summary statistics learns from, and raises alarms on, cycles from the thirtieth onwards (Section 3.7). All results below use this rule.` },
  { table: {
    id: "tab-warmup",
    caption: "Healthy cycles flagged by the summary-statistic detector without the warm-up rule (cross-validation on training engines, 5% alarm budget).",
    header: ["Data configuration", "Healthy cycles before cycle 30", "Flagged (%)", "Healthy cycles from cycle 30", "Flagged (%)"],
    widths: [2.2, 1.6, 1.0, 1.6, 1.0],
    align: ["left", "right", "right", "right", "right"],
    rows: warmUp,
  } },
  { p: `The second decision concerns sensors s6, s10 and s16, which are recorded with very few distinct values. Because an Isolation Forest reacts to rare values, these sensors might cause false alarms. {ref:tab-quantised} compares the cross-validation scores with and without them. Leaving them out changes the F1-score by at most 0.03, and in most cases slightly lowers it. The sensors were therefore kept, so that the detector uses the same inputs as the RUL models.` },
  { table: {
    id: "tab-quantised",
    caption: "Cross-validation scores of the Isolation Forest with and without sensors s6, s10 and s16 (training engines, all cycles, 5% alarm budget).",
    header: ["Data configuration", "Features", "Inputs", "Precision", "Recall", "F1-score"],
    widths: [1.8, 1.5, 2.0, 0.9, 0.9, 0.9],
    align: ["left", "left", "left", "right", "right", "right"],
    fontSize: 18,
    rows: quantised,
    note: "In FD001, s10 and s16 are constant and already removed, so only s6 is affected.",
  } },

  { h: 3, text: "Detection results" },
  { p: `{ref:tab-anomaly-all} and {ref:tab-anomaly-reduced} give the precision, recall and F1-score at the alarm budget of 5%. The first table scores every cycle. The second leaves out the transition zone, that is, the cycles with between 31 and 124 cycles of life remaining. Each table reports the cross-validation on the training engines and the official test engines.` },
  { table: {
    id: "tab-anomaly-all",
    caption: "Isolation Forest results when all cycles are scored (5% alarm budget).",
    header: ["Data configuration", "Features", "CV precision", "CV recall", "CV F1", "Test precision", "Test recall", "Test F1"],
    widths: [1.9, 1.6, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
    align: ["left", "left", "right", "right", "right", "right", "right", "right"],
    fontSize: 18,
    rows: anomalyRows("all cycles"),
    note: "CV: engine-level cross-validation on training engines. Share of cycles labelled anomalous: 15.0% in cross-validation; 2.5% (FD001) and 3.2% (FD002) on the test engines.",
  } },
  { table: {
    id: "tab-anomaly-reduced",
    caption: "Isolation Forest results for healthy cycles against cycles near failure, with the transition zone left out (5% alarm budget).",
    header: ["Data configuration", "Features", "CV precision", "CV recall", "CV F1", "Test precision", "Test recall", "Test F1"],
    widths: [1.9, 1.6, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
    align: ["left", "left", "right", "right", "right", "right", "right", "right"],
    fontSize: 18,
    rows: anomalyRows("healthy vs near failure"),
    note: "Share of cycles labelled anomalous: 27.6% (FD001) and 27.5% (FD002) in cross-validation; 4.0% (FD001) and 5.0% (FD002) on the test engines.",
  } },
  { p: `Four of the six cases behave alike: FD001 with either feature representation, and FD002 with per-condition normalisation with either representation. They are referred to below as the four working cases. In these cases the recall on the test engines is between 0.959 and 1.000, so nearly all cycles near failure are flagged. The test set contains 332 such cycles from 25 engines in FD001 and 1,087 from 61 engines in FD002.` },
  { p: `Precision is much lower. When all cycles are scored, the precision on the test engines is between 0.181 and 0.196 in the four working cases, and the F1-score is between 0.307 and 0.325. When the transition zone is left out, the test precision rises to between 0.459 and 0.652 and the F1-score to between 0.626 and 0.779.` },
  { p: `The cross-validation scores are higher than the test scores: the F1-score of the four working cases is about 0.60 to 0.63 when all cycles are scored and 0.93 to 0.95 without the transition zone. Recall is almost the same in the two evaluations. The difference lies in precision, and it follows from the share of anomalous cycles. The training engines run until failure, so 15% of their cycles are labelled anomalous. The test engines stop early, so only about 3% of their cycles are. With the same rate of false alarms, a smaller share of true anomalies gives a lower precision.` },
  { p: `The two remaining cases use FD002 with dataset-wide normalisation. With raw sensor features the recall on the test engines falls to 0.747. With summary statistics the detector fails: it flags 7.8% of the cycles near failure, against 3.4% of the healthy cycles.` },

  { h: 3, text: "Where the alarms occur" },
  { p: `{ref:fig-trajectories} shows the anomaly score over the observed life of one test engine in each case, namely the engine that is closest to failure at its last recorded cycle. In the working cases the score stays below the threshold for most of the life of the engine and rises as failure approaches. It crosses the threshold some time before the reference zone begins. In the bottom middle panel, for FD002 with dataset-wide normalisation and summary statistics, the score does not rise.` },
  { figure: { id: "fig-trajectories", file: "fig06a_anomaly_score_trajectories.png", caption: "Anomaly score of the Isolation Forest over the observed life of the test engine closest to failure in each case. The shaded area is the reference anomaly zone (30 or fewer cycles remaining). With summary statistics no alarm is raised during the first 29 cycles (warm-up)." } },
  { p: `{ref:fig-alarmrate} generalises this to all engines. It shows the share of cycles that are flagged as a function of the remaining life. For healthy cycles the share is close to or below the budget of 5%. It then rises steadily as the remaining life falls. In the four working cases, between 81% and 90% of the cycles with 31 to 40 cycles remaining are flagged, and at least 92% of the cycles in every part of the reference zone. There is no sharp change at 30 cycles: the detector responds to gradual degradation, and the boundary of the reference label is a fixed line drawn across a continuous process.` },
  { figure: { id: "fig-alarmrate", file: "fig06b_alarm_rate_by_rul.png", caption: "Share of cycles flagged by the Isolation Forest as a function of remaining useful life, in cross-validation on the training engines (top) and on the official test engines (bottom)." } },
  { p: `{ref:tab-zones} shows where the alarms on the test engines fall. In the four working cases between 58% and 71% of all alarms are raised in the transition zone and between 18% and 20% inside the reference zone. Between 10% and 23% are raised on healthy cycles. Most of the alarms that the reference label counts as false are therefore raised on engines that are already degrading but still have more than 30 cycles of life. The last column confirms that the share of healthy test cycles that are flagged stays at or below the budget of 5%.` },
  { table: {
    id: "tab-zones",
    caption: "Distribution of the alarms on the test engines over the life zones (5% alarm budget).",
    header: ["Data configuration", "Features", "Alarms", "On healthy cycles (%)", "In transition zone (%)", "Near failure (%)", "Healthy cycles flagged (%)"],
    widths: [1.8, 1.5, 0.8, 1.1, 1.1, 1.0, 1.2],
    align: ["left", "left", "right", "right", "right", "right", "right"],
    fontSize: 18,
    rows: zones,
    note: "Healthy: 125 or more cycles remaining. Transition zone: 31 to 124 cycles remaining. Near failure: 30 or fewer cycles remaining.",
  } },

  { h: 3, text: "Sensitivity to the alarm budget" },
  { p: `{ref:tab-budget} shows how the results on the test engines change with the alarm budget. A stricter budget of 1% raises precision and the F1-score and lowers recall slightly; in the four working cases the F1-score becomes 0.40 to 0.50 and recall stays above 0.91. A looser budget of 10% has the opposite effect. The budget of 5% was fixed in advance and gives the reported results. The other two columns were not used to choose a setting; choosing the budget from the test results would make the reported scores too optimistic.` },
  { table: {
    id: "tab-budget",
    caption: "Isolation Forest results on the test engines (all cycles) at alarm budgets of 1%, 5% and 10%.",
    header: ["Data configuration", "Features", "P 1%", "P 5%", "P 10%", "R 1%", "R 5%", "R 10%", "F1 1%", "F1 5%", "F1 10%"],
    widths: [1.7, 1.5, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65],
    align: ["left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "right"],
    fontSize: 17,
    rows: budget,
    note: "P: precision. R: recall. F1: F1-score. The percentage is the alarm budget.",
  } },

  // ------------------------------------------------------------------ 4.4
  { h: 2, text: "Model Performance Comparison" },
  { p: `This section reports Experiment 1, the comparison of Random Forest with XGBoost. {ref:fig-models} shows the three metrics for both models in the six cases, and {ref:tab-paired-model} gives the paired differences in test RMSE.` },
  { figure: { id: "fig-models", file: "fig05_model_comparison.png", caption: "Random Forest and XGBoost on the official test engines: RMSE, MAE and R² for raw sensor features (top) and summary statistics (bottom). Error bars: 95% bootstrap intervals." } },
  { table: {
    id: "tab-paired-model",
    caption: "Paired difference in test RMSE between XGBoost and Random Forest (cycles).",
    header: ["Case", "RMSE, Random Forest", "RMSE, XGBoost", "Difference", "95% interval", "Interval excludes zero"],
    widths: [2.9, 1.1, 1.0, 1.0, 1.5, 1.1],
    align: ["left", "right", "right", "right", "center", "center"],
    fontSize: 18,
    rows: paired("model"),
    note: "Difference: XGBoost minus Random Forest on the same test engines; a negative value favours XGBoost. Interval: 95% paired bootstrap interval.",
  } },
  { p: `With raw sensor features the two models are practically equal. Random Forest has the lower RMSE in all three data configurations, but by no more than 0.20 cycles, and all three intervals include zero. With summary statistics XGBoost has the lower RMSE in all three configurations, by 0.48 to 0.96 cycles. One of these three intervals excludes zero, for FD002 with per-condition normalisation, where the difference is −0.96 cycles with an interval from −1.69 to −0.31.` },
  { p: `The MAE and R² give a similar picture ({ref:fig-models}): the differences are very small with raw features, and XGBoost is better in all three configurations with summary statistics. The cross-validation results agree in direction: with raw features the two models differ by less than 0.2 cycles in every configuration, and with summary statistics XGBoost is lower by 0.1 to 0.3 cycles ({ref:tab-selected}).` },
  { p: `In summary, neither model is clearly better. XGBoost shows a small advantage when summary statistics are used, and this advantage is distinguishable from zero in one case of six.` },

  // ------------------------------------------------------------------ 4.5
  { h: 2, text: "Feature Representation and Preprocessing Comparison" },
  { p: `This section reports Experiment 3. It first compares the two feature representations for the RUL models, then the two normalisation schemes on FD002, and finally considers both factors for the anomaly detector.` },

  { h: 3, text: "Raw sensor features and summary statistics" },
  { p: `{ref:fig-representation} shows the test RMSE of both models with each representation, and {ref:tab-paired-representation} gives the paired differences.` },
  { figure: { id: "fig-representation", file: "fig07_representation_comparison.png", caption: "Test RMSE with raw sensor features and with summary statistics. Error bars: 95% bootstrap intervals. The vertical axis does not start at zero." } },
  { table: {
    id: "tab-paired-representation",
    caption: "Paired difference in test RMSE between summary statistics and raw sensor features (cycles).",
    header: ["Case", "RMSE, raw", "RMSE, summary", "Difference", "95% interval", "Interval excludes zero"],
    widths: [2.9, 1.0, 1.1, 1.0, 1.5, 1.1],
    align: ["left", "right", "right", "right", "center", "center"],
    fontSize: 18,
    rows: paired("representation"),
    note: "Difference: summary statistics minus raw sensor features on the same test engines; a negative value favours summary statistics.",
  } },
  { p: `Summary statistics lower the test RMSE in five of the six cases, by 0.27 to 1.88 cycles. In the sixth case, Random Forest on FD002 with dataset-wide normalisation, the RMSE rises by 0.39 cycles. The largest gain is obtained by XGBoost on FD002 with per-condition normalisation (−1.88 cycles), and this is the only case in which the interval excludes zero. On FD001 the gains are 0.45 and 1.16 cycles, but with 100 test engines the intervals are wide and include zero.` },
  { p: `The cross-validation results point the same way and are somewhat clearer for FD001, where summary statistics lower the cross-validated RMSE by 1.2 to 1.5 cycles ({ref:tab-selected}). Taken together, the results indicate that summary statistics help modestly, that the benefit is larger for XGBoost than for Random Forest, and that it is absent when readings from different operating conditions are averaged without per-condition normalisation.` },

  { h: 3, text: "Dataset-wide and per-condition normalisation" },
  { p: `{ref:tab-paired-normalisation} compares the two normalisation schemes on FD002 for the RUL models.` },
  { table: {
    id: "tab-paired-normalisation",
    caption: "Paired difference in test RMSE between per-condition and dataset-wide normalisation on FD002 (cycles).",
    header: ["Case", "RMSE, dataset-wide", "RMSE, per-condition", "Difference", "95% interval", "Interval excludes zero"],
    widths: [2.9, 1.1, 1.1, 1.0, 1.5, 1.1],
    align: ["left", "right", "right", "right", "center", "center"],
    fontSize: 18,
    rows: paired("normalisation"),
    note: "Difference: per-condition minus dataset-wide normalisation on the same test engines; a negative value favours per-condition normalisation.",
  } },
  { p: `The effect depends on the feature representation. With raw sensor features, per-condition normalisation gives a slightly higher RMSE for both models (+0.53 and +0.55 cycles). With summary statistics it gives a lower RMSE (−0.59 and −1.06 cycles). All four intervals include zero, so none of these differences is established by the test data.` },
  { p: `This result contrasts with the exploration in Section 4.1, where per-condition normalisation raised the correlation of individual sensors with the RUL from 0.078 to 0.454 in the median. An explanation consistent with the results is that tree-based models combine several inputs. Under dataset-wide normalisation, sensors such as s1 and s5 still show which operating condition a row belongs to, so the trees can separate the conditions themselves and then apply thresholds within each condition. Per-condition normalisation would then add little for raw readings. It would still help when a window statistic is computed, because the mean over 30 cycles mixes readings from different conditions unless these have first been brought to a common level. This explanation was not tested directly.` },

  { h: 3, text: "Effect on the anomaly detector" },
  { p: `For the Isolation Forest the normalisation scheme is decisive ({ref:tab-anomaly-all}). On the test engines of FD002, per-condition normalisation raises the recall from 0.747 to 0.959 with raw sensor features and from 0.078 to 0.969 with summary statistics. The cross-validation results show the same change (from 0.797 to 0.978 and from 0.074 to 0.987). {ref:fig-alarmrate} shows the cause: under dataset-wide normalisation the share of flagged cycles rises only slowly with raw features and not at all with summary statistics.` },
  { p: `The feature representation matters less for the detector, provided that the normalisation is suitable. In the four working cases the two representations give almost the same scores when all cycles are scored; the F1-score on the test engines differs by 0.012 to 0.016. When the transition zone is left out, summary statistics give a higher test precision on FD002 with per-condition normalisation (0.652 against 0.518), because fewer healthy cycles are flagged ({ref:tab-zones}).` },
  { p: `The two kinds of model thus respond differently to the same preprocessing step. The supervised RUL models are barely affected by the normalisation scheme. The unsupervised detector depends on it.` },

  // ------------------------------------------------------------------ 4.6
  { h: 2, text: "Comparison of FD001 and FD002" },
  { p: `{ref:fig-datasets} and {ref:tab-datasets-compare} set the results for the single-condition dataset FD001 beside those for the multi-condition dataset FD002.` },
  { figure: { id: "fig-datasets", file: "fig08_dataset_comparison.png", caption: "FD001 and FD002 on the official test engines: RUL prediction (top) and anomaly detection (bottom). Error bars in the top left panel: 95% bootstrap intervals." } },
  { table: {
    id: "tab-datasets-compare",
    caption: "Results for FD001 and FD002 side by side.",
    header: ["Model", "Features", "Metric", "FD001", "FD002, dataset-wide", "FD002, per-condition"],
    widths: [1.4, 1.5, 2.6, 0.9, 1.2, 1.2],
    align: ["left", "left", "left", "right", "right", "right"],
    fontSize: 18,
    rows: comparison,
    note: "RMSE in cycles. Test: official test engines. Anomaly detection at the 5% alarm budget.",
  } },
  { p: `**RUL prediction with the capped target.** The two datasets give similar results. The test RMSE is between 15.83 and 16.99 cycles on FD001 and between 15.85 and 17.73 cycles on FD002. The intervals of the two datasets overlap widely ({ref:fig-datasets}). On this measure the models predict the RUL of engines operated under six conditions about as accurately as that of engines operated under one.` },
  { p: `**RUL prediction against the uncapped values.** Here FD002 is far worse, at about 29 cycles against 17 to 18 cycles for FD001. As shown in Section 4.2, this follows from the larger share of FD002 test engines with a true RUL above the cap (22% against 11%) and from their longer remaining life.` },
  { p: `**RUL prediction in cross-validation.** On all cycles of held-out training engines, FD002 is about 1 to 2.5 cycles worse than FD001: 19.0 to 19.7 cycles against 17.1 to 18.7 cycles. This comparison is not affected by the cap, since both use the capped target, and it covers the whole life of the engines. It suggests that FD002 is somewhat harder, although the difference is small.` },
  { p: `**Anomaly detection.** FD002 comes close to FD001 only with per-condition normalisation, where the recall is 0.96 to 0.97 against 0.985 to 1.000 for FD001. With dataset-wide normalisation the recall on FD002 is 0.747 with raw sensor features and 0.078 with summary statistics.` },
  { p: `Two cautions apply to this comparison. FD001 and FD002 differ in size as well as in the number of operating conditions: FD002 has 260 training engines against 100. A difference between the datasets cannot therefore be attributed to the operating conditions alone. Second, the two datasets have different test engines, so the paired comparison used elsewhere in this chapter is not available here.` },

  // ------------------------------------------------------------------ 4.7
  { h: 2, text: "Summary of Results" },
  { p: `The main results of the chapter are as follows.` },
  { numbered: [
    `**Data.** The data are complete and need no cleaning. Six sensors are constant in FD001. In FD002 the operating condition explains almost all of the variance of the sensors and hides the wear trend in the individual readings.`,
    `**RUL prediction.** With the capped target, Random Forest and XGBoost reach a test RMSE of 15.8 to 17.7 cycles and an R² of 0.82 to 0.86 on both datasets. Against the uncapped values the RMSE on FD002 is about 29 cycles, because 22% of its test engines have a true RUL above the cap.`,
    `**Model comparison.** The two models perform alike. XGBoost is better by 0.5 to 1.0 cycles with summary statistics, and one interval of six excludes zero.`,
    `**Feature representation.** Summary statistics lower the test RMSE in five of six cases, by up to 1.9 cycles. One interval of six excludes zero.`,
    `**Normalisation.** For the RUL models the normalisation scheme changes the test RMSE by 0.5 to 1.1 cycles in either direction, and no interval excludes zero. For the Isolation Forest it is decisive: on FD002, per-condition normalisation raises the test recall from 0.75 to 0.96 with raw features and from 0.08 to 0.97 with summary statistics.`,
    `**Anomaly detection.** With suitable normalisation the Isolation Forest flags 96% to 100% of the test cycles near failure. Its precision against the 30-cycle reference label is about 0.19 when all cycles are scored, mainly because it also flags engines in earlier stages of degradation.`,
    `**Single and multiple operating conditions.** With the capped target, RUL prediction is about equally accurate on FD001 and FD002. In cross-validation FD002 is 1 to 2.5 cycles worse. Anomaly detection on FD002 matches FD001 only with per-condition normalisation.`,
  ] },
  { p: `These results rest on simulated data, a single random seed and one official division into training and test engines. Sixteen paired comparisons were made and two intervals exclude zero; with that number of comparisons one such result could arise by chance, and no correction for multiple comparisons was applied. No uncertainty intervals were computed for the anomaly detection scores. Chapter 5 discusses what the results mean for the two research questions and how they relate to existing work.` },
];

buildChapter({
  number: 4,
  title: "Results and Analysis",
  blocks,
  outFile: "Chapter_4_Results_and_Analysis.docx",
}).catch((error) => {
  console.error(error);
  process.exit(1);
});
