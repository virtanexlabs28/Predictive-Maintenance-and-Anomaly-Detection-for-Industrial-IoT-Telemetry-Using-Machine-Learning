/* Chapter 3 - Research Methodology. Run with:  node chapter3.js */

const { buildChapter, math } = require("./lib");
const { t, sub, sup, frac, sum, sqrt } = math;

const blocks = [
  { p: `This chapter describes how the study was carried out. It presents the research framework, the dataset and the reasons for choosing it, the preprocessing steps, the construction of the Remaining Useful Life (RUL) target, the two feature representations, the three machine learning models, the training and validation procedures, the evaluation metrics and the design of the experiments. Each decision is stated together with its justification so that the work can be repeated.` },

  // ------------------------------------------------------------------ 3.1
  { h: 2, text: "Research Framework" },
  { p: `The study follows a quantitative, experimental design. It uses an existing public dataset of simulated turbofan engines and applies three machine learning models to two tasks. The first task is RUL prediction, a supervised regression problem addressed with Random Forest and XGBoost. The second task is anomaly detection, an unsupervised problem addressed with Isolation Forest. No new data were collected.` },
  { p: `{ref:fig-framework} shows the framework. The work proceeds in six stages: data acquisition, data exploration, preprocessing, feature representation, model training and evaluation, and analysis. The first research question (RQ1) concerns how accurately the selected models detect anomalies and predict RUL on single-condition and multi-condition data. It is answered by stages 5a and 5b, which are run on both datasets. The second research question (RQ2) concerns how preprocessing and feature representation affect performance. It is answered by repeating stage 5 under different choices in stages 3 and 4.` },
  { p: `Two rules apply to every stage. First, any quantity that is estimated from data, such as a mean used for normalisation or a model parameter, is estimated from training engines only. Engines used for validation or testing are never used to fit anything. Second, every number reported in this thesis is produced by the project code and stored in a result file. No result was entered by hand.` },
  { figure: { id: "fig-framework", file: "fig00_research_framework.png", widthCm: 12.5, caption: "Research framework of the study." } },

  // ------------------------------------------------------------------ 3.2
  { h: 2, text: "Dataset Description" },
  { p: `The study uses the Turbofan Engine Degradation Simulation Data Set published in the NASA Prognostics Data Repository {cite:nasa2008}. The data were generated with the Commercial Modular Aero-Propulsion System Simulation (C-MAPSS), a simulation of a large commercial turbofan engine of the 90,000 lb thrust class {cite:saxena2008}. Saxena et al. {cite:saxena2008} describe how the data were produced. An engine is simulated flight after flight while the efficiency and flow of one of its modules are gradually reduced, following an exponential wear model. One snapshot of sensor values is recorded per flight, and measurement noise is added. The simulation of an engine stops when a health index reaches zero, which is defined as failure.` },
  { p: `The repository contains four subsets, FD001 to FD004. This study uses FD001 and FD002. In both, the fault is degradation of the high-pressure compressor (HPC). They differ in the operating conditions: FD001 is simulated at a single condition (sea level), whereas FD002 is simulated at six conditions. {ref:tab-datasets} summarises the two subsets.` },
  { table: {
    id: "tab-datasets",
    caption: "Subsets of the C-MAPSS data used in this study.",
    header: ["Subset", "Operating conditions", "Fault mode", "Split", "Engines", "Rows", "Cycles per engine (min / mean / max)"],
    widths: [1.0, 1.3, 1.5, 0.8, 0.9, 1.0, 2.2],
    align: ["left", "center", "left", "left", "right", "right", "center"],
    rows: [
      ["FD001", "1", "HPC degradation", "Train", "100", "20,631", "128 / 206.3 / 362"],
      ["FD001", "1", "HPC degradation", "Test", "100", "13,096", "31 / 131.0 / 303"],
      ["FD002", "6", "HPC degradation", "Train", "260", "53,759", "128 / 206.8 / 378"],
      ["FD002", "6", "HPC degradation", "Test", "259", "33,991", "21 / 131.2 / 367"],
    ],
    note: "One row is one operating cycle (flight) of one engine. Counts were computed from the downloaded files.",
  } },
  { p: `Each subset has a training file and a test file. According to the documentation supplied with the data {cite:nasa2008}, every engine starts with a different, unknown degree of initial wear and manufacturing variation, which is considered normal. The engine operates normally at the start of its record and develops a fault at some point. In the training file the fault grows until the engine fails, so each training engine is a complete run-to-failure record. In the test file each record stops some time before failure, and a separate file gives the true RUL of each test engine at its last recorded cycle.` },
  { p: `Every file has 26 columns: the engine number, the cycle number, three operational settings and 21 sensor measurements. The three operational settings are altitude, Mach number and throttle resolver angle {cite:saxena2008}. {ref:tab-variables} lists the variables. The documentation numbers the sensors from 1 to 21 without naming them. The names in {ref:tab-variables} follow the order of the output table in Saxena et al. {cite:saxena2008}; this assignment is the convention in the literature and is not stated in the documentation itself.` },
  { table: {
    id: "tab-variables",
    caption: "Variables in the C-MAPSS files. Sensor symbols, descriptions and units from Saxena et al. {cite:saxena2008}.",
    header: ["Column", "Name in this study", "Symbol", "Description", "Unit"],
    widths: [0.7, 1.4, 1.1, 3.6, 0.9],
    fontSize: 18,
    rows: [
      ["1", "unit", "-", "Engine number", "-"],
      ["2", "cycle", "-", "Operating cycle (flight) number", "-"],
      ["3", "setting 1", "-", "Altitude", "1,000 ft"],
      ["4", "setting 2", "-", "Mach number", "-"],
      ["5", "setting 3", "TRA", "Throttle resolver angle", "-"],
      ["6", "s1", "T2", "Total temperature at fan inlet", "°R"],
      ["7", "s2", "T24", "Total temperature at LPC outlet", "°R"],
      ["8", "s3", "T30", "Total temperature at HPC outlet", "°R"],
      ["9", "s4", "T50", "Total temperature at LPT outlet", "°R"],
      ["10", "s5", "P2", "Pressure at fan inlet", "psia"],
      ["11", "s6", "P15", "Total pressure in bypass duct", "psia"],
      ["12", "s7", "P30", "Total pressure at HPC outlet", "psia"],
      ["13", "s8", "Nf", "Physical fan speed", "rpm"],
      ["14", "s9", "Nc", "Physical core speed", "rpm"],
      ["15", "s10", "epr", "Engine pressure ratio (P50/P2)", "-"],
      ["16", "s11", "Ps30", "Static pressure at HPC outlet", "psia"],
      ["17", "s12", "phi", "Ratio of fuel flow to Ps30", "pps/psi"],
      ["18", "s13", "NRf", "Corrected fan speed", "rpm"],
      ["19", "s14", "NRc", "Corrected core speed", "rpm"],
      ["20", "s15", "BPR", "Bypass ratio", "-"],
      ["21", "s16", "farB", "Burner fuel-air ratio", "-"],
      ["22", "s17", "htBleed", "Bleed enthalpy", "-"],
      ["23", "s18", "Nf_dmd", "Demanded fan speed", "rpm"],
      ["24", "s19", "PCNfR_dmd", "Demanded corrected fan speed", "rpm"],
      ["25", "s20", "W31", "HPT coolant bleed", "lbm/s"],
      ["26", "s21", "W32", "LPT coolant bleed", "lbm/s"],
    ],
    note: "LPC: low-pressure compressor. HPC: high-pressure compressor. HPT: high-pressure turbine. LPT: low-pressure turbine.",
  } },
  { p: `The data were downloaded on 10 November 2026 from the link given in the NASA repository. The checksum (SHA-256) of the downloaded archive and of each file used was recorded, and an automated check confirms that the files have not changed. The original files are never modified; all later steps read them and write new files.` },

  // ------------------------------------------------------------------ 3.3
  { h: 2, text: "Dataset Selection Justification" },
  { p: `The C-MAPSS data were selected for four reasons.` },
  { bullets: [
    `**Run-to-failure records with known failure times.** RUL prediction requires examples in which the time of failure is known. Such records are rarely available for real industrial equipment, because equipment is normally maintained before it fails. In the training files every engine is run until failure, so an exact RUL label can be computed for every cycle.`,
    `**A controlled comparison of single-condition and multi-condition operation.** FD001 and FD002 have the same fault mode and differ in the number of operating conditions. This matches RQ1, which asks about accuracy on single-condition and multi-condition telemetry. FD003 and FD004 add a second fault mode. Including them would change two factors at once, so they are outside the scope of this study.`,
    `**A fixed, public train and test split.** The repository provides separate training and test files and the true RUL of the test engines. Results can therefore be obtained on engines that play no part in model development, and other researchers can repeat the evaluation on the same engines.`,
    `**Established use in research.** The dataset is widely used for RUL estimation {cite:laredo2019}, which makes the methods and results of this study easier to relate to existing work.`,
  ] },
  { p: `The main limitation of this choice is that the data are simulated. The engine model, the wear model and the noise model are simplifications {cite:saxena2008}. Results obtained on these data describe how the models behave on the simulation. They do not demonstrate how the same models would perform on real engines or on other industrial equipment. This limitation is taken up again in Chapter 5.` },

  // ------------------------------------------------------------------ 3.4
  { h: 2, text: "Data Preprocessing" },
  { p: `Preprocessing was limited to the steps needed to make the data usable by the models. The steps are described in the order in which they are applied.` },

  { h: 3, text: "Loading and quality checks" },
  { p: `The files were read as whitespace-separated text and the 26 columns were named as in {ref:tab-variables}. All four files were then checked for missing values, non-finite values, duplicated rows and duplicated combinations of engine and cycle. None were found, so no values were imputed and no rows were removed. It was also verified that the cycles of every engine are numbered 1, 2, 3 and so on without gaps, because the RUL labels depend on this. The results of these checks are reported in Chapter 4.` },

  { h: 3, text: "Identification of operating conditions" },
  { p: `In FD002 each flight is flown at one of six operating conditions. The three operational settings take a small number of nominal values to which a small amount of noise has been added. The operating condition of a row was identified by rounding setting 1 to the nearest integer, setting 2 to two decimal places and setting 3 to the nearest integer. Rows with the same rounded values belong to the same condition. This rule recovers exactly one condition in FD001 and six in FD002, in agreement with the documentation. Because the rule is fixed in advance and nothing is estimated from the data, it can be applied to test engines without any risk of information leakage. A clustering algorithm such as k-means would give the same six groups but would add a fitted model with no benefit.` },

  { h: 3, text: "Removal of uninformative sensors" },
  { p: `A sensor that never changes cannot help any model. A sensor was removed if it takes a single value in the training data within every normalisation group. For dataset-wide normalisation the group is the whole training file. For per-condition normalisation there is one group per operating condition, so a sensor that is constant inside every condition is removed; after normalisation such a sensor would be zero everywhere. The decision uses training engines only. The rule has no threshold to choose, which keeps it objective.` },
  { p: `In FD001 the rule removes six sensors (s1, s5, s10, s16, s18 and s19) and 15 remain. In FD002 no sensor is constant over the whole file, so all 21 remain under dataset-wide normalisation; under per-condition normalisation four sensors (s1, s5, s18 and s19) are removed and 17 remain. Sensors that are nearly constant but not strictly constant, such as s6 in FD001, are kept. Removing them would require an arbitrary threshold. The three operational settings are used to identify the operating condition and are not given to the models as inputs.` },

  { h: 3, text: "Normalisation" },
  { p: `The retained sensors were standardised with the z-score. For sensor *j* and a row belonging to normalisation group *g*, the normalised value is` },
  { equation: { id: "eq-zscore", math: [sub("z", "j"), t(" = "), frac([sub("x", "j"), t(" − "), sub("μ", "j,g")], [sub("σ", "j,g")])] } },
  { p: `where *x* is the recorded value and μ and σ are the mean and standard deviation of sensor *j* in group *g*, computed from the training engines. Two schemes were prepared so that their effect could be compared under RQ2.` },
  { bullets: [
    `**Dataset-wide normalisation.** There is one group, the whole training file. This is the ordinary way of standardising a dataset.`,
    `**Per-condition normalisation.** There is one group for each operating condition. Each reading is compared with the typical reading at the same altitude, Mach number and throttle setting.`,
  ] },
  { p: `The same formula is used in both schemes, so the only difference between them is the grouping. FD001 has a single operating condition, and it was verified that the two schemes give identical values there. The comparison is therefore made on FD002 only. This gives three data configurations, which are used throughout the thesis: FD001, FD002 with dataset-wide normalisation and FD002 with per-condition normalisation. If a sensor is constant inside one operating condition but not in all of them, its normalised value in that condition is set to zero.` },
  { p: `The reason for comparing the two schemes is as follows. Tree-based models choose splits by comparing a feature with a threshold. Rescaling a whole column, as dataset-wide normalisation does, changes the thresholds but not the order of the values, so it is not expected to change the predictions of tree-based models. This is a theoretical expectation and was not tested in this study. Per-condition normalisation is different, because it shifts readings from different operating conditions by different amounts and therefore changes their order. It is the basic preprocessing step that can make a difference for the selected models. {ref:fig-normalisation} illustrates the two schemes for one FD002 engine.` },
  { figure: { id: "fig-normalisation", file: "fig03b_normalisation_effect_FD002.png", caption: "Two sensors of one FD002 training engine under dataset-wide normalisation (left) and per-condition normalisation (right)." } },

  { h: 3, text: "Prevention of data leakage" },
  { p: `Data leakage occurs when information that would not be available at prediction time is used to build a model, which makes its measured performance too optimistic {cite:kaufman2012}. Three safeguards were applied. The sensors to remove and the means and standard deviations for normalisation are computed from training engines and then applied unchanged to test engines. During cross-validation the same quantities are recomputed from the training folds only, so the engines held out for validation do not influence them. Finally, engines are never split across training and validation, as described in Section 3.8. Automated checks confirm each safeguard; for example, one check recomputes a normalised test value by hand from training statistics and compares it with the value produced by the code.` },
  { table: {
    id: "tab-preprocessing",
    caption: "Outcome of preprocessing for the three data configurations.",
    header: ["Data configuration", "Sensors removed", "Sensors retained", "Training rows", "Test rows", "Test engines"],
    widths: [2.3, 2.3, 1.0, 1.1, 1.0, 1.0],
    align: ["left", "left", "right", "right", "right", "right"],
    rows: [
      ["FD001", "s1, s5, s10, s16, s18, s19", "15", "20,631", "13,096", "100"],
      ["FD002, dataset-wide normalisation", "none", "21", "53,759", "33,991", "259"],
      ["FD002, per-condition normalisation", "s1, s5, s18, s19", "17", "53,759", "33,991", "259"],
    ],
  } },

  // ------------------------------------------------------------------ 3.5
  { h: 2, text: "RUL Label Construction" },
  { p: `The files do not contain RUL values for individual cycles, so the target had to be constructed. For a training engine *i* that fails after its last recorded cycle, the RUL at cycle *t* is the number of cycles that remain:` },
  { equation: { id: "eq-rul", math: [sub("RUL", "i"), t("(t) = "), sub("T", "i"), t(" − t")] } },
  { p: `where *T* is the last recorded cycle of the engine. The label is therefore zero on the last row and increases by one for each earlier cycle.` },
  { p: `A label that decreases linearly over the whole life assumes that the sensors can tell how much life remains even when the engine is new. The exploration in Chapter 4 shows that this is not the case: the sensors stay level during early life and begin to drift only as the fault develops. A piecewise linear target, which is constant in early life and then decreases linearly, has been proposed for this dataset {cite:laredo2019}. This study uses such a target with a cap of 125 cycles, as fixed in the scope of the study:` },
  { equation: { id: "eq-cap", math: [sub("y", "i"), t("(t) = min(125, "), sub("T", "i"), t(" − t)")] } },
  { p: `The capped value *y* is the target that the models learn. {ref:fig-rul} shows both labels for three engines of each subset. About 39% of the training rows lie on the cap (39.4% in FD001 and 39.5% in FD002).` },
  { figure: { id: "fig-rul", file: "fig03a_rul_target.png", caption: "Linear RUL and the piecewise linear target capped at 125 cycles for the shortest-, median- and longest-lived training engine of each subset." } },
  { p: `For test engines the true RUL is published only for the last recorded cycle. The RUL at an earlier cycle *t* of test engine *i* is that value plus the number of cycles still to be observed:` },
  { equation: { id: "eq-rul-test", math: [sub("RUL", "i"), t("(t) = ("), sub("L", "i"), t(" − t) + "), sub("R", "i")] } },
  { p: `where *L* is the last recorded cycle of the test engine and *R* is the published RUL at that cycle. The same cap is then applied. Both the capped and the uncapped values are kept. The true RUL exceeds 125 for 11 of the 100 FD001 test engines and for 57 of the 259 FD002 test engines. A model trained on a capped target cannot predict values much above the cap, so results are reported against the capped target as the main result and against the uncapped values as a second result (Section 3.9).` },

  // ------------------------------------------------------------------ 3.6
  { h: 2, text: "Feature Representation" },
  { p: `RQ2 asks whether the way sensor readings are presented to a model affects its performance. Two representations were compared. In both, one sample corresponds to one cycle of one engine, so both have exactly the same samples and the same targets. Any difference in model performance therefore comes from the inputs alone.` },
  { bullets: [
    `**Representation A, raw sensor features.** The inputs are the normalised readings of the retained sensors at the current cycle.`,
    `**Representation B, summary statistics.** For every retained sensor three values are computed over the most recent 30 cycles of the engine, including the current cycle: the last value, the mean and the standard deviation.`,
  ] },
  { p: `For sensor *j* at cycle *t*, the mean and standard deviation over the window are` },
  { equation: { id: "eq-mean", math: [sub("m", "j"), t("(t) = "), frac([t("1")], [t("n")]), sum([sub("x", "j"), t("(k)")], [t("k = t − n + 1")], [t("t")])] } },
  { equation: { id: "eq-std", math: [sub("s", "j"), t("(t) = "), sqrt([frac([t("1")], [t("n")]), sum([sup([t("("), sub("x", "j"), t("(k) − "), sub("m", "j"), t("(t))")], "2")], [t("k = t − n + 1")], [t("t")])])] } },
  { p: `where *n* = min(*t*, 30) is the number of cycles in the window. The last value is the reading of the current cycle, so Representation B contains Representation A and adds two columns per sensor. {ref:tab-features} gives the number of inputs in each case, and {ref:fig-summary} shows the three statistics for one sensor.` },
  { table: {
    id: "tab-features",
    caption: "Number of model inputs for each data configuration and feature representation.",
    header: ["Data configuration", "Sensors", "Inputs, Representation A", "Inputs, Representation B"],
    widths: [2.6, 1.0, 1.7, 1.7],
    align: ["left", "right", "right", "right"],
    rows: [
      ["FD001", "15", "15", "45"],
      ["FD002, dataset-wide normalisation", "21", "21", "63"],
      ["FD002, per-condition normalisation", "17", "17", "51"],
    ],
  } },
  { figure: { id: "fig-summary", file: "fig03c_summary_features_example.png", caption: "Summary-statistic features for sensor s11 of one training engine in each data configuration: readings and 30-cycle mean (top), 30-cycle standard deviation (bottom)." } },
  { p: `Four rules govern the calculation.` },
  { bullets: [
    `**The window looks backwards only.** A sample never uses readings from later cycles, since these would not be available when a prediction is made.`,
    `**The window never crosses from one engine to another.**`,
    `**Short histories use the cycles available.** During the first 29 cycles of an engine fewer than 30 readings exist, and the statistics are computed from the readings available so far. At the first cycle the mean equals the reading and the standard deviation is zero. This keeps every sample in both representations and is necessary because six FD002 test engines have fewer than 30 cycles.`,
    `**Nothing is estimated from the data.** The window statistics are computed separately for each engine from its own readings, so the step cannot leak information from test engines.`,
  ] },
  { p: `The window length of 30 cycles was fixed in advance and was not tuned. Windows of 20 to 30 cycles are reported for earlier studies on this dataset {cite:laredo2019}. One property of Representation B should be noted. While the window is still filling, the statistics indirectly reveal that the engine is new. This is information that would be available in practice, and it applies equally to test engines, whose records also begin at the first cycle. It is not leakage, but it is one way in which Representation B differs from Representation A in addition to reducing noise.` },

  // ------------------------------------------------------------------ 3.7
  { h: 2, text: "Model Architecture and Configurations" },
  { p: `Three models were used, as fixed by the scope of the study. Random Forest and XGBoost predict RUL. Isolation Forest detects anomalies. All three are ensembles of decision trees. The scikit-learn library {cite:pedregosa2011} provides the Random Forest and Isolation Forest implementations, and the XGBoost library {cite:chen2016} provides XGBoost.` },

  { h: 3, text: "Random Forest" },
  { p: `A Random Forest is a collection of decision trees, each grown on a random sample of the training data and using a random subset of the input variables when choosing each split. The prediction of the forest is the average of the predictions of its trees {cite:breiman2001}. Averaging many varied trees reduces the tendency of a single tree to fit noise. In this study Random Forest serves as the supervised baseline for RUL prediction.` },

  { h: 3, text: "XGBoost" },
  { p: `XGBoost is an implementation of gradient tree boosting {cite:chen2016}. Trees are added one at a time, and each new tree is fitted to reduce the errors that remain after the trees already built. A learning rate limits the contribution of each tree, and the objective includes a penalty on model complexity. XGBoost is compared with Random Forest because the two build their ensembles in different ways: Random Forest averages independent trees, whereas XGBoost builds trees in sequence.` },

  { h: 3, text: "Isolation Forest" },
  { p: `Isolation Forest detects anomalies without labels {cite:liu2008}. It builds trees that split the data at random values of randomly chosen variables. Observations that differ from the majority are separated from the rest after few splits, whereas ordinary observations need many. The average number of splits needed to isolate an observation is converted into an anomaly score, with a higher score indicating a more unusual observation.` },
  { p: `In this study the Isolation Forest is fitted on healthy data only, defined as the cycles of training engines that are at least 125 cycles from failure. This is the flat part of the RUL target, where the sensors have not yet begun to drift. The model therefore learns what normal operation looks like, and a cycle is unusual to the extent that it differs from normal operation. A cycle is flagged as anomalous when its score exceeds a threshold. The threshold is the value exceeded by 5% of the healthy training cycles. This alarm budget of 5% was fixed before any result was computed. No failure information is used to fit the model or to set the threshold, so the detector is fully unsupervised.` },
  { p: `One adjustment was needed for Representation B. During the first 29 cycles of an engine the window is still filling, and the resulting statistics look unusual to an unsupervised model for reasons unrelated to engine health. With Representation B the detector therefore learns from, and raises alarms on, cycles from the thirtieth onwards. This rule was adopted on the basis of cross-validation on the training engines, before the test engines were scored; the supporting evidence is given in Chapter 4.` },

  { h: 3, text: "Model settings" },
  { p: `{ref:tab-settings} lists the settings. For the two RUL models a small number of settings were tuned and the others were fixed. Each model has six candidate combinations, and the same selection procedure is applied to both (Section 3.8). The grids were kept small on purpose: the aim of the study is a fair comparison under a simple, repeatable procedure, not the best achievable accuracy. The Isolation Forest was not tuned, because tuning would require labels.` },
  { table: {
    id: "tab-settings",
    caption: "Model settings. Names are those of the software libraries.",
    header: ["Model", "Fixed settings", "Tuned settings (candidates)"],
    widths: [1.3, 3.2, 3.2],
    rows: [
      ["Random Forest", "200 trees (n_estimators); random seed 42", "max_features: square root of the number of inputs, or one third of the inputs; min_samples_leaf: 5, 20 or 50"],
      ["XGBoost", "300 trees (n_estimators); subsample 0.8; colsample_bytree 0.8; histogram tree method; squared-error objective; random seed 42", "max_depth: 3, 5 or 7; learning_rate: 0.05 or 0.1"],
      ["Isolation Forest", "200 trees (n_estimators); 256 samples per tree (max_samples); random seed 42; alarm budget 5% of healthy training cycles", "none"],
    ],
    note: "max_features is the number of inputs considered at each split; min_samples_leaf is the smallest number of samples allowed in a leaf; max_depth is the largest depth of a tree.",
  } },

  // ------------------------------------------------------------------ 3.8
  { h: 2, text: "Training and Validation Procedures" },

  { h: 3, text: "Engine-level cross-validation" },
  { p: `The cycles of one engine are strongly related to one another. If the rows were split at random, cycles of the same engine would appear in both the training part and the validation part, and a model could score well simply by recognising the engine. Validation was therefore carried out at the level of engines. The training engines of each subset were divided into five folds so that all cycles of an engine fall in the same fold (group k-fold cross-validation, with the engine number as the group). Each fold is used once for validation while the other four are used for training.` },
  { p: `Within each of the five splits the full preprocessing is repeated: the sensors to remove and the normalisation statistics are computed from the four training folds and applied to the validation fold, and the features are then built. The model is fitted on the training folds and evaluated on all cycles of the validation engines against the capped target.` },

  { h: 3, text: "Selection of settings and final models" },
  { p: `For every combination of data configuration, feature representation and model, each of the six candidates in {ref:tab-settings} was evaluated by cross-validation, and the candidate with the lowest mean root mean squared error over the five folds was selected. With three data configurations, two feature representations, two models, six candidates and five folds, this amounts to 360 model fits. The selected candidate was then fitted once more on all training engines of the subset, which gives 12 final models. The procedure is identical for Random Forest and XGBoost.` },

  { h: 3, text: "Evaluation on the official test engines" },
  { p: `The test engines were kept apart during all of the above. Each final model was applied to the test engines once, after its settings had been selected. The published RUL values refer to the last recorded cycle of each test engine, so one prediction is made for each test engine at that cycle and compared with the published value. The cross-validation scores and the test scores are therefore not directly comparable: the former cover all cycles of the validation engines, the latter one cycle per engine.` },

  { h: 3, text: "Procedure for anomaly detection" },
  { p: `The Isolation Forest was evaluated in two ways. First, the same five-fold engine-level cross-validation was used: in each split the preprocessing and the detector are fitted on the training folds, and every cycle of the validation engines is scored. This yields scores for the complete run-to-failure record of every training engine, each produced by a detector that had not seen that engine. Second, the detector was fitted on the healthy cycles of all training engines and applied to every cycle of the official test engines. The design decisions for the detector, including the adjustment for Representation B described in Section 3.7, were made using the cross-validation results only. The test engines were scored once, after the design had been fixed.` },

  { h: 3, text: "Reproducibility" },
  { p: `A single random seed (42) is used for all models. XGBoost reproduces its predictions exactly. Random Forest builds identical trees on every run but averages them in parallel, so its predictions agree between runs to within rounding (less than one billionth of a cycle). The anomaly scores of the Isolation Forest were likewise checked to agree between runs. The fold assignment is deterministic.` },

  // ------------------------------------------------------------------ 3.9
  { h: 2, text: "Evaluation Metrics" },

  { h: 3, text: "Metrics for RUL prediction" },
  { p: `Let *y* denote the true value and *ŷ* the predicted value for sample *i*, and let *N* be the number of samples. Three metrics are used.` },
  { equation: { id: "eq-mae", math: [t("MAE = "), frac([t("1")], [t("N")]), sum([t("|"), sub("y", "i"), t(" − "), sub("ŷ", "i"), t("|")], [t("i = 1")], [t("N")])] } },
  { equation: { id: "eq-rmse", math: [t("RMSE = "), sqrt([frac([t("1")], [t("N")]), sum([sup([t("("), sub("y", "i"), t(" − "), sub("ŷ", "i"), t(")")], "2")], [t("i = 1")], [t("N")])])] } },
  { equation: { id: "eq-r2", math: [sup([t("R")], "2"), t(" = 1 − "), frac([sum([sup([t("("), sub("y", "i"), t(" − "), sub("ŷ", "i"), t(")")], "2")], [t("i = 1")], [t("N")])], [sum([sup([t("("), sub("y", "i"), t(" − ȳ)")], "2")], [t("i = 1")], [t("N")])])] } },
  { p: `where *ȳ* is the mean of the true values.` },
  { bullets: [
    `**Mean absolute error (MAE)** is the average size of the error in cycles. It is easy to interpret, and every cycle of error counts equally.`,
    `**Root mean squared error (RMSE)** is also expressed in cycles but gives more weight to large errors. This is appropriate for maintenance, where a large error in the predicted life is more harmful than several small ones. RMSE is used to select model settings and as the main metric.`,
    `**The coefficient of determination (R²)** is the share of the variation in the true RUL that the predictions explain. A value of 1 is a perfect prediction and a value of 0 is no better than always predicting the mean. Unlike MAE and RMSE it has no unit, so it can be compared between datasets with different ranges of RUL.`,
  ] },
  { p: `The main results use the capped target, because this is the quantity the models were trained to predict. The same predictions are also compared with the uncapped published values, and both sets of results are reported. This matters when results are compared with other studies, which do not all follow the same convention.` },

  { h: 3, text: "Uncertainty of the test results" },
  { p: `A test set of 100 or 259 engines is a sample, and a metric computed on it would differ somewhat for another sample of engines. To indicate the size of this uncertainty, bootstrap intervals {cite:efron1979} were computed. The test engines were resampled with replacement 2,000 times, the metric was recalculated for each resample, and the 2.5th and 97.5th percentiles were taken as a 95% interval. When two configurations are compared, the same resamples are used for both and the interval is computed for the difference in RMSE. If this interval contains zero, the data do not show a clear difference between the two configurations. The intervals are computed from the stored predictions and involve no retraining. They were added to the original plan because RQ2 asks whether a choice affects performance, and this cannot be judged from single values alone.` },

  { h: 3, text: "Metrics for anomaly detection" },
  { p: `Evaluating a detector requires a statement of which cycles ought to be flagged. The dataset does not record when a fault becomes present, so a reference label was constructed: a cycle is labelled anomalous when the engine has 30 or fewer cycles of life remaining. This label marks late degradation. It is a proxy, not a record of an observed fault, and it has two consequences. An alarm does not show that a mechanical failure is present, and an alarm raised when more than 30 cycles remain is not necessarily wrong, since the engine may already be degrading.` },
  { p: `Each cycle falls into one of four groups: flagged and labelled anomalous (true positive, TP), flagged but not labelled anomalous (false positive, FP), not flagged but labelled anomalous (false negative, FN), and neither (true negative, TN). The metrics are` },
  { equation: { id: "eq-precision", math: [t("Precision = "), frac([t("TP")], [t("TP + FP")])] } },
  { equation: { id: "eq-recall", math: [t("Recall = "), frac([t("TP")], [t("TP + FN")])] } },
  { equation: { id: "eq-f1", math: [sub("F", "1"), t(" = "), frac([t("2 × Precision × Recall")], [t("Precision + Recall")])] } },
  { bullets: [
    `**Precision** is the share of flagged cycles that are labelled anomalous. Low precision means many false alarms.`,
    `**Recall** is the share of cycles labelled anomalous that were flagged. Low recall means that anomalies are missed.`,
    `**The F1-score** is the harmonic mean of precision and recall and is high only when both are high.`,
  ] },
  { p: `Accuracy is not used. Anomalous cycles are a small minority, so a detector that never raised an alarm would have a high accuracy while being useless. Precision also depends on how common anomalous cycles are, so the share of anomalous cycles is reported with every result.` },
  { p: `Two scorings are reported. In the first, every cycle is scored. In the second, cycles with between 31 and 124 cycles of life remaining are left out, so that healthy cycles are compared with cycles near failure. This transition zone is where the reference label is least meaningful, because the engine is neither clearly healthy nor close to failure. In addition to the alarm budget of 5%, results for budgets of 1% and 10% are shown to indicate how the balance between precision and recall changes. The 5% results are the reported results; the other two were not used to select a setting.` },

  // ------------------------------------------------------------------ 3.10
  { h: 2, text: "Experimental Design" },
  { p: `Three experiments were carried out. {ref:tab-experiments} relates them to the research questions.` },
  { table: {
    id: "tab-experiments",
    caption: "Experiments and their relation to the research questions.",
    header: ["Experiment", "Question", "What is compared", "Data", "Metrics"],
    widths: [1.5, 0.8, 2.7, 2.0, 1.5],
    rows: [
      ["1. Model performance", "RQ1", "Random Forest against XGBoost for RUL prediction", "FD001 and FD002", "MAE, RMSE, R²"],
      ["2. Anomaly detection", "RQ1, RQ2", "Isolation Forest against the reference label", "FD001 and FD002", "Precision, recall, F1-score"],
      ["3. Feature representation", "RQ2", "Raw sensor features against summary statistics; dataset-wide against per-condition normalisation (FD002)", "FD001 and FD002", "Difference in RMSE with 95% interval"],
    ],
  } },
  { p: `All three experiments use the same grid of six cases, formed by the three data configurations and the two feature representations ({ref:tab-features}). Experiment 1 compares the two RUL models within each case. Experiment 3 compares the cases with each other for each RUL model. Experiment 2 applies the Isolation Forest to each case.` },
  { p: `Two points extend the design that was first planned, and both are stated here for transparency. First, the comparison of normalisation schemes on FD002 was added to Experiment 3. RQ2 refers to preprocessing as well as to feature representation, and, as explained in Section 3.4, dataset-wide normalisation is not expected to change the results of tree-based models. Without this comparison the preprocessing part of RQ2 would have had nothing to measure. Second, the Isolation Forest was run on all six cases instead of one case per dataset. This required no additional data preparation and allows RQ2 to be examined for the unsupervised model as well.` },
  { p: `The comparison of FD001 with FD002 that RQ1 requires is made by setting the results of Experiments 1 and 2 for the two datasets side by side. FD001 and FD002 differ in size as well as in the number of operating conditions (100 against 260 training engines), so a difference between them cannot be attributed to the operating conditions alone.` },

  // ------------------------------------------------------------------ 3.11
  { h: 2, text: "Implementation" },
  { p: `The study was implemented in Python 3.12.10. {ref:tab-software} lists the libraries. The code is organised in modules that correspond to the stages of {ref:fig-framework}: data loading, exploration, preprocessing, feature construction, RUL prediction, anomaly detection, evaluation and summary of results. Five notebooks present the results of the stages with explanations.` },
  { table: {
    id: "tab-software",
    caption: "Software used.",
    header: ["Library", "Version", "Use in this study"],
    widths: [1.4, 0.9, 4.5],
    rows: [
      ["pandas", "2.2.2", "Reading the files, tables, window statistics"],
      ["NumPy", "2.4.6", "Numerical calculations, bootstrap resampling"],
      ["scikit-learn", "1.9.0", "Random Forest, Isolation Forest, group k-fold cross-validation, regression metrics"],
      ["XGBoost", "3.3.0", "XGBoost model"],
      ["Matplotlib", "3.11.0", "Figures"],
    ],
  } },
  { p: `Correctness is supported by 72 automated checks, which are run with a single command. They cover, among other things, the integrity of the downloaded files, the construction of the RUL labels, the fitting of preprocessing on training engines only, the backward-looking property of the window statistics, the separation of engines between folds, and the agreement of the tables in this thesis with the stored results. The tables and figures of Chapter 4 are generated from the stored result files by the code.` },

  // ------------------------------------------------------------------ 3.12
  { h: 2, text: "Chapter Summary" },
  { p: `This chapter has described the method of the study. Two subsets of the C-MAPSS data, one with a single operating condition and one with six, are preprocessed with rules that are fitted on training engines only. A piecewise linear RUL target capped at 125 cycles is constructed. Two normalisation schemes and two feature representations give six cases. Random Forest and XGBoost are tuned with engine-level cross-validation and evaluated once on the official test engines using MAE, RMSE and R², with bootstrap intervals to indicate uncertainty. An Isolation Forest is fitted on healthy cycles and evaluated against a constructed reference label using precision, recall and the F1-score. Chapter 4 presents the results.` },
];

buildChapter({
  number: 3,
  title: "Research Methodology",
  blocks,
  outFile: "Chapter_3_Research_Methodology.docx",
}).catch((error) => {
  console.error(error);
  process.exit(1);
});
