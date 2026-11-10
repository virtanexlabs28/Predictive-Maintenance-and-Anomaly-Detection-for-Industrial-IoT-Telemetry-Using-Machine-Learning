/* Chapter 1 - Introduction. Run with:  node chapter1.js */

const { buildChapter } = require("./lib");

const blocks = [
  // ------------------------------------------------------------------ 1.1
  { h: 2, text: "Background" },
  { p: `Industrial equipment wears out. Bearings, compressors, turbines and pumps degrade as they are used, and if the degradation is not noticed in time the equipment fails. A failure in service interrupts production, can damage other equipment and, in fields such as aviation, can endanger lives. Maintenance exists to prevent this, but maintenance also has a cost. Servicing equipment too late risks failure. Servicing it too early wastes the remaining life of parts that were still sound and takes equipment out of use without need.` },
  { p: `The way maintenance is planned therefore matters. The simplest policy is to repair equipment after it fails. A second policy is to service equipment at fixed intervals, whatever its actual condition. A third policy, predictive maintenance, uses information about the condition of the equipment to decide when to act. Zhu et al. {cite:zhu2026} describe predictive maintenance as a paradigm in which maintenance is carried out only after analytical models predict certain failures or degradations. The reviews of Carvalho et al. {cite:carvalho2019} and Zonta et al. {cite:zonta2020} link it to the avoidance of production shutdowns and to machine downtime, costs and the quality of production.` },
  { p: `Predictive maintenance rests on two analytical tasks. The first is to detect that the behaviour of a machine has become abnormal, which is the task of anomaly detection. The second is to estimate how long the machine can continue to operate, known as its Remaining Useful Life (RUL). Si et al. {cite:si2011} define the RUL of an asset as the useful life that remains at a particular time of operation. This thesis addresses both tasks with machine learning.` },

  // ------------------------------------------------------------------ 1.2
  { h: 2, text: "Industrial IoT and Predictive Maintenance" },
  { p: `Predictive maintenance needs data about the condition of equipment, and such data have become far more plentiful. Sisinni et al. {cite:sisinni2018} describe how the Internet of Things, which turns common objects into connected devices, is making its way into industry with purposely designed solutions, a development referred to as the Industrial Internet of Things (IIoT). Machines fitted with sensors report temperatures, pressures, speeds and flows continuously. Carvalho et al. {cite:carvalho2019} note that the amount of data extracted from production processes has increased strongly because of the spread of sensing technologies. In this thesis the stream of sensor readings sent by a machine is called its telemetry.` },
  { p: `The availability of telemetry has changed how the problem is approached. Where a physical model of the degradation of a machine is not available, or the machine is too complex for one to be built, its behaviour can be learned from recorded data instead {cite:laredo2019}. Machine learning methods have accordingly been described as a promising tool for predictive maintenance, with the caution that the performance of an application depends on the appropriate choice of method {cite:carvalho2019}.` },
  { p: `Telemetry is not easy to use, however. Sensor readings are noisy. A machine is operated under changing conditions, and a change in load or speed alters the readings as much as a developing fault does, or more. Examples of failure are scarce, because failures are rare in many safety-critical systems {cite:ariaschao2021}. And the moment at which a fault begins is usually not recorded, so there is often no direct record of which readings are abnormal. A study of predictive maintenance has to deal with each of these difficulties.` },

  // ------------------------------------------------------------------ 1.3
  { h: 2, text: "Research Problem" },
  { p: `Much of the research on RUL prediction uses the C-MAPSS turbofan engine datasets published by NASA {cite:saxena2008}, which contain simulated run-to-failure records of aircraft engines. The literature reviewed in Chapter 2 shows four features of this research that together define the problem addressed here.` },
  { bullets: [
    `**The methods have become complex.** The studies reviewed are mostly based on deep neural networks of increasing size. Classical tree-based models such as Random Forest and XGBoost, which are simpler to train and to explain and which are known to be strong on tabular data {cite:grinsztajn2022}, receive little attention on these data.`,
    `**Data with several operating conditions are less studied.** The subset with a single operating condition is the most used {cite:ramasso2014}. Where the subset with six operating conditions is reported, the errors are consistently larger.`,
    `**The effect of basic preprocessing is unclear.** Studies differ in how they normalise the sensor readings and in how they present them to the model, but these choices are fixed parts of each method. How much they matter, and whether they matter equally for different kinds of model, is seldom reported.`,
    `**Results are hard to compare and anomaly detection is rarely included.** The construction of the RUL target and the evaluation conventions vary between studies {cite:ramasso2014}, and the detection of anomalies is usually not examined alongside RUL prediction.`,
  ] },
  { p: `The problem can be stated as follows. It is not well established how accurately simple, classical machine learning models can predict RUL and detect anomalies in industrial telemetry under single and multiple operating conditions, nor how far basic preprocessing and feature representation determine their performance. Without this knowledge it is difficult to judge when a simple model is sufficient and which preparation of the data is necessary.` },

  // ------------------------------------------------------------------ 1.4
  { h: 2, text: "Research Objectives" },
  { p: `The aim of the study is to assess, with a simple and fully reproducible procedure, how three classical machine learning models perform on industrial telemetry for RUL prediction and anomaly detection, and how basic preprocessing and feature representation affect them. The specific objectives are:` },
  { numbered: [
    `to prepare the single-condition (FD001) and multi-condition (FD002) subsets of the C-MAPSS data with a documented preprocessing procedure that prevents data leakage;`,
    `to train Random Forest and XGBoost models for RUL prediction and to evaluate and compare their accuracy on both subsets;`,
    `to apply an Isolation Forest for unsupervised anomaly detection on both subsets and to evaluate it against a clearly defined reference label;`,
    `to compare two normalisation schemes and two feature representations and to determine their effect on the performance of the models;`,
    `to compare the results for single-condition and multi-condition data and to relate them to the existing literature.`,
  ] },

  // ------------------------------------------------------------------ 1.5
  { h: 2, text: "Research Questions" },
  { p: `The study answers two research questions.` },
  { p: `**RQ1: Prognostic and diagnostic accuracy.** How accurately can selected machine learning models detect anomalies and predict Remaining Useful Life (RUL) using single-condition and multi-condition industrial telemetry data?` },
  { p: `**RQ2: Preprocessing and feature representation.** How do basic preprocessing techniques and feature representations affect the performance of machine learning models in industrial telemetry analysis?` },
  { p: `RQ1 is addressed by Experiment 1, which compares Random Forest and XGBoost for RUL prediction, and by Experiment 2, which evaluates the Isolation Forest. RQ2 is addressed by Experiment 3, which compares raw sensor features with summary statistics and dataset-wide normalisation with normalisation within each operating condition. The experiments are described in Chapter 3.` },

  // ------------------------------------------------------------------ 1.6
  { h: 2, text: "Research Significance" },
  { p: `The study is significant in three respects.` },
  { bullets: [
    `**For practice.** Simple models are cheaper to build, faster to train and easier to explain than deep neural networks. Evidence about how well they perform, and about which preprocessing steps they need, helps a practitioner decide whether a simple model is adequate for a maintenance application before investing in a more complex one.`,
    `**For research.** The study isolates the effect of individual preprocessing and feature choices by changing one factor at a time under an otherwise identical procedure. It reports results under more than one evaluation convention and with intervals that indicate their uncertainty, which makes them easier to compare with other work.`,
    `**For reproducibility.** Every step, from the download of the data to the tables of results, is carried out by code that is tested and documented. The study can therefore be repeated and extended by others.`,
  ] },

  // ------------------------------------------------------------------ 1.7
  { h: 2, text: "Scope and Limitations" },
  { p: `The scope of the study was deliberately kept narrow so that each step could be carried out and explained carefully.` },
  { bullets: [
    `**Data.** Two subsets of the NASA C-MAPSS turbofan data are used: FD001, with one operating condition, and FD002, with six. Both have a single fault mode. The subsets with two fault modes are not used.`,
    `**Models.** Three models are used: Random Forest and XGBoost for RUL prediction, and Isolation Forest for anomaly detection. Neural networks and other deep learning methods are outside the scope.`,
    `**Preprocessing and features.** Only basic steps are considered: removal of constant sensors, two normalisation schemes, and two feature representations. Advanced feature engineering and dimensionality reduction are not used.`,
    `**Deliverable.** The study is an offline analysis. It does not include a monitoring system, a dashboard or any software for use in operation.`,
  ] },
  { p: `Three limitations should be kept in mind from the outset. First, the data are simulated. The results describe how the models behave on a simulation of an aircraft engine and do not show how they would perform on real engines or on other industrial equipment. Second, the dataset does not record when a fault begins, so the evaluation of anomaly detection relies on a reference label constructed for this study. Third, the experiments use one random seed and the single official division of the data into training and test engines. The limitations are discussed in full in Chapter 5.` },

  // ------------------------------------------------------------------ 1.8
  { h: 2, text: "Structure of the Thesis" },
  { p: `The thesis has five chapters. Chapter 2 reviews the literature on predictive maintenance, anomaly detection, RUL prediction, the three selected models and the preprocessing of sensor data, and identifies the gaps that the study addresses. Chapter 3 describes the methodology: the dataset, the preprocessing, the construction of the RUL target, the feature representations, the models, the training and validation procedures, the evaluation metrics and the design of the experiments. Chapter 4 presents the results of the data exploration and of the three experiments. Chapter 5 interprets the findings, answers the two research questions, compares the results with the literature, and sets out the contributions, the limitations and directions for future research.` },
];

buildChapter({
  number: 1,
  title: "Introduction",
  blocks,
  outFile: "Chapter_1_Introduction.docx",
}).catch((error) => {
  console.error(error);
  process.exit(1);
});
