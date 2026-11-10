/* Chapter 5 - Discussion and Conclusion. Run with:  node chapter5.js
 * Table values are computed from the result files.
 */

const { buildChapter, readCsv, format } = require("./lib");
const { fixed } = format;

const comparison = readCsv("results/tables/thesis/table10_dataset_comparison.csv");
const range = (metric, column, decimals = 2) => {
  const values = comparison.filter((r) => r.metric === metric).map((r) => Number(r[column]));
  return `${fixed(Math.min(...values), decimals)} to ${fixed(Math.max(...values), decimals)}`;
};
const row = (label, metric, decimals) => [
  label, range(metric, "FD001", decimals), range(metric, "FD002 dataset-wide", decimals), range(metric, "FD002 per-condition", decimals),
];

const blocks = [
  { p: `This chapter discusses the results presented in Chapter 4. It interprets the main findings, answers the two research questions, compares the results with the literature reviewed in Chapter 2, and sets out the contributions and limitations of the study and directions for future research.` },

  // ------------------------------------------------------------------ 5.1
  { h: 2, text: "Interpretation of Findings" },

  { h: 3, text: "Simple models predict RUL with a consistent level of accuracy" },
  { p: `Random Forest and XGBoost predicted the capped RUL of the test engines with an RMSE of about 16 to 18 cycles and explained 82% to 86% of its variation. This level was reached with a small number of settings, with fewer than 400 model fits in total, and with a training time of minutes. The level was also stable: it changed little between the two models, the two feature representations, the two normalisation schemes and the two datasets. The stability is itself a finding. It indicates that, for these data and this target, the accuracy of a tree-based model is determined more by the information in the sensors than by the particular choices examined here.` },
  { p: `The error is not uniform over the life of an engine. Predictions were close for engines near failure and more scattered for engines with a long remaining life (Figure 4.4). This is the useful direction for maintenance, since the predictions are most reliable when a decision is most urgent.` },

  { h: 3, text: "The two RUL models are closer than their designs suggest" },
  { p: `Random Forest and XGBoost build their ensembles on opposite principles, yet with raw sensor features their test RMSE differed by no more than 0.2 cycles. XGBoost was better by 0.5 to 1.0 cycles when summary statistics were used, and this difference could be distinguished from zero in one case of six. A plausible reading is that a boosted model makes better use of a larger set of related inputs, but the evidence is not strong enough to state this as a conclusion. For a practitioner the implication is that the choice between the two matters less than the preparation of the data and the care taken in validation.` },

  { h: 3, text: "Preprocessing matters for some models and not for others" },
  { p: `The clearest result of the study concerns normalisation within each operating condition. The exploration of the data suggested that it would be important: in FD002 the operating condition explains almost all of the variance of the sensors, and removing its effect raised the median correlation between individual sensors and the RUL from 0.08 to 0.45. For the Isolation Forest this expectation was fully borne out. Its recall on the test engines of FD002 rose from 0.75 to 0.96 with raw features and from 0.08 to 0.97 with summary statistics. For the supervised models it was not. Their test RMSE changed by about half a cycle to one cycle, in either direction, and none of the four differences could be distinguished from zero.` },
  { p: `The difference can be understood from how the models learn. A supervised model is told the target. If the operating condition can be recognised from the inputs, which is the case because some sensors depend almost entirely on it, a tree-based model can learn different rules for different conditions. An unsupervised detector is told nothing. It treats the largest source of variation in the data as the structure to be learned, and in FD002 the largest source of variation is the change of operating condition, not wear. This explanation is consistent with all the results, but the study did not test it directly, for example by removing the sensors that reveal the operating condition.` },
  { p: `A general lesson follows. A statistic computed on single inputs, such as a correlation, does not predict how a model that combines many inputs will respond to a preprocessing step. The effect of preprocessing has to be measured on the model for which it is intended.` },

  { h: 3, text: "Summary statistics help modestly" },
  { p: `Describing each cycle by the last value, mean and standard deviation of each sensor over the previous 30 cycles lowered the test RMSE in five of six cases, by 0.3 to 1.9 cycles, and one of these differences could be distinguished from zero. Cross-validation pointed the same way. The gain was absent where readings from different operating conditions were averaged without per-condition normalisation, which shows that the two choices interact: a window statistic is only as meaningful as the values that enter it. For the Isolation Forest the summary statistics changed little when all cycles were scored, but they required a warm-up rule, because half-filled windows look unusual to an unsupervised model.` },

  { h: 3, text: "The detector finds degradation, not a boundary" },
  { p: `Fitted on healthy cycles only, the Isolation Forest flagged 96% to 100% of the test cycles within 30 cycles of failure. Its precision against the reference label was about 0.19 when every cycle was scored. The two figures are not in conflict. The share of flagged cycles rose gradually as the remaining life fell, and most of the alarms counted as false were raised on engines that were already degrading but had more than 30 cycles left. The detector responds to a continuous process, while the reference label draws a line across it. Leaving out the transition zone raised the F1-score on the test engines to between 0.63 and 0.78.` },
  { p: `Two consequences should be stated plainly. First, the scores measure how well the detector separates late degradation from healthy operation in simulated data. They do not show that it detects real faults, and an alarm does not indicate that a mechanical failure is present. Second, precision depends strongly on how common the anomalous cycles are: the same detector had a precision of about 0.45 on run-to-failure engines, where 15% of cycles are near failure, and about 0.19 on the test engines, where about 3% are. A precision figure is therefore meaningful only together with that share.` },

  { h: 3, text: "The evaluation convention can change the conclusion" },
  { p: `On FD002 the RMSE was about 17 cycles against the capped target and about 29 cycles against the uncapped values, for the same predictions. The gap arises because 22% of the FD002 test engines have a true RUL above the cap, with a mean of 159 cycles, and a model trained on a capped target predicts about 110 cycles for them. On FD001, where 11% of the test engines are above the cap and by a smaller margin, the gap is about one cycle. Judged by the capped results, the two datasets are equally difficult. Judged by the uncapped results, FD002 appears far more difficult. A statement about the difficulty of multi-condition data therefore depends on a convention that is easily left unstated.` },

  // ------------------------------------------------------------------ 5.2
  { h: 2, text: "Answers to the Research Questions" },

  { h: 3, text: "RQ1: Prognostic and diagnostic accuracy" },
  { p: `*How accurately can selected machine learning models detect anomalies and predict Remaining Useful Life (RUL) using single-condition and multi-condition industrial telemetry data?*` },
  { p: `{ref:tab-rq1} summarises the evidence. Each cell gives the lowest and highest value over the combinations of model and feature representation.` },
  { table: {
    id: "tab-rq1",
    caption: "Accuracy on single-condition and multi-condition data (official test engines unless stated).",
    header: ["Measure", "FD001 (one condition)", "FD002, dataset-wide normalisation", "FD002, per-condition normalisation"],
    widths: [3.0, 1.7, 1.9, 1.9],
    align: ["left", "center", "center", "center"],
    fontSize: 18,
    rows: [
      row("RUL: test RMSE, capped target (cycles)", "Test RMSE, capped target", 2),
      row("RUL: test MAE, capped target (cycles)", "Test MAE, capped target", 2),
      row("RUL: test R², capped target", "Test R2, capped target", 3),
      row("RUL: test RMSE, uncapped target (cycles)", "Test RMSE, uncapped target", 2),
      row("RUL: cross-validated RMSE (cycles)", "Cross-validated RMSE", 2),
      row("Anomaly detection: recall", "Test recall", 3),
      row("Anomaly detection: precision, all cycles", "Test precision, all cycles", 3),
      row("Anomaly detection: F1-score, all cycles", "Test F1, all cycles", 3),
      row("Anomaly detection: F1-score, healthy vs near failure", "Test F1, healthy vs near failure", 3),
    ],
    note: "RUL: Random Forest and XGBoost with raw and summary-statistic features. Anomaly detection: Isolation Forest with raw and summary-statistic features, 5% alarm budget.",
  } },
  { p: `The answer has three parts.` },
  { numbered: [
    `**RUL prediction.** Random Forest and XGBoost predict the capped RUL with a test RMSE of 15.8 to 17.0 cycles on the single-condition data and 15.9 to 17.7 cycles on the multi-condition data, a mean absolute error of 11.6 to 13.3 cycles, and an R² of 0.82 to 0.86. The two models are about equally accurate.`,
    `**Anomaly detection.** An Isolation Forest fitted on healthy data flags 98.5% to 100% of the cycles near failure on the single-condition data and 96% to 97% on the multi-condition data, provided that the latter are normalised within each operating condition. Against a label that marks the last 30 cycles of life, its precision is about 0.19 and its F1-score about 0.32, because it also raises alarms during earlier stages of degradation.`,
    `**Single and multiple operating conditions.** With the capped target, RUL prediction is about as accurate on multi-condition data as on single-condition data. In cross-validation over the whole life of the engines the multi-condition data are 1 to 2.5 cycles worse. Against uncapped values they appear much worse, for a reason that lies in the evaluation and not in the operating conditions. For anomaly detection, multi-condition data are handled as well as single-condition data only with per-condition normalisation.`,
  ] },

  { h: 3, text: "RQ2: Preprocessing and feature representation" },
  { p: `*How do basic preprocessing techniques and feature representations affect the performance of machine learning models in industrial telemetry analysis?*` },
  { p: `The effect depends on the kind of model.` },
  { numbered: [
    `**Normalisation.** Normalising within each operating condition has little effect on the supervised tree-based models: the test RMSE on FD002 changes by +0.5 cycles with raw features and by −0.6 to −1.1 cycles with summary statistics, and no difference is distinguishable from zero. It has a decisive effect on the unsupervised detector, whose recall on FD002 rises from 0.75 to 0.96 with raw features and from 0.08 to 0.97 with summary statistics.`,
    `**Feature representation.** Replacing single readings by summary statistics over a 30-cycle window lowers the test RMSE of the supervised models in five of six cases, by 0.3 to 1.9 cycles; one difference is distinguishable from zero. The benefit is larger for XGBoost than for Random Forest. For the detector the representation makes little difference when the normalisation is suitable.`,
    `**Interaction.** The two choices are not independent. Summary statistics help the supervised models when the data are normalised per condition and not otherwise, and they make the detector fail completely when the data are not normalised per condition.`,
    `**Removal of constant sensors** removes six of 21 sensors in the single-condition data and none in the multi-condition data unless the operating condition is taken into account. Its effect on performance was not measured separately.`,
  ] },
  { p: `In short, basic preprocessing and feature representation affect the supervised models modestly and the unsupervised model strongly. Preprocessing that reflects the structure of the data, here the operating conditions, matters most where the model has no target to guide it.` },

  // ------------------------------------------------------------------ 5.3
  { h: 2, text: "Comparison with Existing Literature" },
  { p: `Comparison with published results has to be made with care, for the reasons given in Section 2.3: the studies do not all use the same RUL target or the same evaluation convention, and for most of them these could not be confirmed from the original papers. With that reservation, four comparisons can be made.` },
  { p: `**Level of accuracy.** The studies in Table 2.1 report an RMSE of 10.7 to 18.5 cycles for FD001. The results of this study, 15.8 to 17.0 cycles against the capped target and 17.1 to 18.0 against the uncapped values, lie within that range and above the best of the neural network results. For FD002 the studies report 19.5 to 30.3 cycles. The uncapped results of this study, about 29 cycles, are at the upper end of that range, close to the multi-layer perceptron of Laredo et al. {cite:laredo2019} (29.09), while the capped results, 15.9 to 17.7 cycles, are below it. It cannot be concluded from this that the models of this study outperform the published methods on FD002, because it is not known which convention each of them used. What the comparison does show is that the two conventions lead to opposite impressions of the same models.` },
  { p: `**Difficulty of multi-condition data.** Every study in Table 2.1 reports a larger error on FD002 than on FD001, and Laredo et al. {cite:laredo2019} conclude that the number of operating conditions has a large effect on the difficulty of the data. This study agrees in part. In cross-validation FD002 was 1 to 2.5 cycles worse than FD001, and against uncapped values it was much worse. Against the capped target the two were equal. The results suggest that a part of the difference reported for FD002 may come from its test set, in which a larger share of engines have a long remaining life, and not only from the operating conditions. This is a suggestion for further examination and not a claim about any particular published result.` },
  { p: `**Tree-based models and window statistics.** Grinsztajn et al. {cite:grinsztajn2022} found tree-based models to be strong on tabular data, and Goel et al. {cite:goel2026} report in a preprint that XGBoost with window statistics was competitive with neural networks on a single-condition subset. The results here point in the same direction: with a few simple statistics, XGBoost reached 15.83 cycles on FD001, comparable to the earlier neural network results in Table 2.1. The finding that normalisation barely affects the tree-based models is also consistent with what is known about them: they are little affected by a rescaling of their inputs {cite:galili2016} and are robust to uninformative features {cite:grinsztajn2022}.` },
  { p: `**Operating conditions and anomaly detection.** Ramasso and Saxena {cite:ramasso2014} describe the health of an engine as strongly occluded by the operating conditions, and the leading methods of the 2008 challenge treated the conditions explicitly. The exploration in this study quantifies the occlusion (more than 99% of the variance of most sensors), and the Isolation Forest results confirm that it has to be removed for an unsupervised method. The results add a qualification: for supervised tree-based models the explicit treatment made little difference. Finally, Liu et al. {cite:liu2008} report that Isolation Forest works when the training set contains no anomalies. The high recall obtained here with a model fitted on healthy cycles alone is in line with that property.` },

  // ------------------------------------------------------------------ 5.4
  { h: 2, text: "Research Contributions" },
  { p: `The study makes five contributions. They are modest in scale, as befits an undergraduate project, and they are supported by the evidence in Chapter 4.` },
  { numbered: [
    `**A like-for-like comparison of two classical models on single-condition and multi-condition data.** Random Forest and XGBoost were compared under one procedure, with validation by engine, on FD001 and FD002. The comparison shows that they perform alike and gives the size and uncertainty of the differences.`,
    `**Evidence that the effect of preprocessing depends on the kind of model.** By changing one factor at a time, the study shows that normalisation within operating conditions is nearly irrelevant for the supervised tree-based models and decisive for the unsupervised detector.`,
    `**A quantified effect of the evaluation convention.** The study shows that capping the test target changes the RMSE on FD002 from about 29 to about 17 cycles for identical predictions, and traces the difference to the composition of the test set.`,
    `**An evaluation of unsupervised anomaly detection on run-to-failure data with its limits stated.** The study defines a reference label, reports results with and without the ambiguous transition zone, shows how the alarm rate rises with degradation, and identifies a practical issue with window features in an unsupervised model.`,
    `**A reproducible pipeline.** All steps are implemented in tested code, the tables of the thesis are generated from stored results, and every reference was checked before it was cited.`,
  ] },

  // ------------------------------------------------------------------ 5.5
  { h: 2, text: "Limitations" },
  { p: `The conclusions are subject to the following limitations.` },
  { bullets: [
    `**Simulated data.** C-MAPSS is a simulation with a simplified model of the engine, of wear and of noise {cite:saxena2008}. The results do not demonstrate how the models would perform on real engines or on other equipment.`,
    `**Two subsets and one fault mode.** Only FD001 and FD002 were used. The findings may not hold for data with several fault modes.`,
    `**One seed and one split.** The models were trained with a single random seed and tested on the single official test set. The bootstrap intervals reflect the sampling of test engines but not the variation between seeds or between divisions of the data.`,
    `**Small tuning grids and fixed choices.** Each model had six candidate settings. The cap of 125 cycles and the window of 30 cycles were fixed and not varied.`,
    `**Constructed anomaly label.** The reference label (30 cycles) and the alarm budget (5%) are choices. Other values give other precision and recall figures. No uncertainty intervals were computed for the detection scores, because cycles within an engine are strongly related.`,
    `**Multiple comparisons.** Sixteen paired comparisons were made and two intervals exclude zero. No correction for multiple comparisons was applied, so these two results are suggestive and not conclusive.`,
    `**Confounded comparison of datasets.** FD001 and FD002 differ in size as well as in operating conditions, so differences between them cannot be attributed to the operating conditions alone.`,
    `**Untested explanation.** The explanation given for the small effect of normalisation on the supervised models is consistent with the results but was not tested.`,
    `**Selective literature review.** Most sources were checked at the level of the abstract, and some reported values were taken from comparison tables in other papers. The comparison with the literature is correspondingly limited.`,
  ] },

  // ------------------------------------------------------------------ 5.6
  { h: 2, text: "Future Research Directions" },
  { bullets: [
    `**More realistic data.** The procedure could be applied to the subsets with two fault modes and to the newer dataset of Arias Chao et al. {cite:ariaschao2021}, which uses recorded flight conditions and relates degradation to the operating history, and eventually to data from real equipment.`,
    `**Robustness.** Repeating the experiments with several seeds and with repeated cross-validation would show how much of the observed differences is due to chance. An engine-level bootstrap would give intervals for the detection scores.`,
    `**The fixed choices.** The cap on the RUL target, the length of the window and the reference label for anomalies could be varied systematically to show how sensitive the conclusions are to them.`,
    `**Testing the explanation.** The role of the sensors that reveal the operating condition could be tested by removing them, or by giving the operational settings to the models as inputs.`,
    `**Alarm rules.** In practice an alarm would be raised only after several consecutive flagged cycles. Such rules could reduce false alarms and should be evaluated at the level of engines, for example by how early a warning is given.`,
    `**Cost-sensitive evaluation.** The metrics used here treat early and late predictions alike. The scoring function defined with the dataset penalises late predictions more heavily {cite:saxena2008}, which is closer to the needs of maintenance.`,
    `**Comparison with neural networks under one protocol.** Training a neural network under the same validation procedure and the same conventions would allow a fair comparison with the classical models.`,
  ] },

  // ------------------------------------------------------------------ 5.7
  { h: 2, text: "Final Conclusion" },
  { p: `This thesis examined how three classical machine learning models perform on industrial telemetry and how basic preprocessing and feature representation affect them, using simulated turbofan engine data with one and with six operating conditions.` },
  { p: `Random Forest and XGBoost predicted the remaining useful life of test engines, capped at 125 cycles, with a root mean squared error of about 16 to 18 cycles, and performed alike. The accuracy was similar for single-condition and multi-condition data when judged against the capped target. An Isolation Forest fitted on healthy data alone detected nearly all cycles close to failure, and also raised alarms earlier in the degradation of an engine, which lowered its precision against a fixed reference label.` },
  { p: `Preprocessing mattered in a way that depended on the model. Normalising the sensor readings within each operating condition had little effect on the supervised tree-based models and a decisive effect on the unsupervised detector. Summary statistics over a time window helped the supervised models modestly. The study also showed that the convention used to evaluate RUL prediction can change the apparent difficulty of multi-condition data.` },
  { p: `The main lesson is that simple, well-understood models, applied with careful validation, give a sound and reproducible baseline for predictive maintenance, and that the preparation of the data deserves as much attention as the choice of model. These conclusions rest on simulated data and remain to be tested on real equipment.` },
];

buildChapter({
  number: 5,
  title: "Discussion and Conclusion",
  blocks,
  outFile: "Chapter_5_Discussion_and_Conclusion.docx",
}).catch((error) => {
  console.error(error);
  process.exit(1);
});
