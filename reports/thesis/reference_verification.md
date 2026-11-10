# Reference Verification Log

Every reference cited in the thesis is listed here with how it was checked. A reference
is cited only for statements that were checked against the source named in the last
column. The formatted entries are stored in `build/references.json`.

## Verified and cited

| Key | Reference | Bibliographic details checked against | Statements in the thesis checked against |
|---|---|---|---|
| nasa2008 | Saxena and Goebel, Turbofan Engine Degradation Simulation Data Set, NASA, 2008 | Citation text on the NASA repository page (2026-11-10) | Dataset readme, read in full |
| saxena2008 | Saxena, Goebel, Simon and Eklund, PHM 2008, pp. 1-9 | Crossref, DOI 10.1109/PHM.2008.4711414 (2026-11-10) | Full text (PDF supplied with the dataset), pages 2 to 7 |
| laredo2019 | Laredo, Chen, Schütze and Sun, Neural Networks 116, 2019, pp. 178-187 | Crossref title search, DOI 10.1016/j.neunet.2019.04.016 (2026-11-10) | arXiv preprint 1905.05918; wording may differ from the journal version |
| kaufman2012 | Kaufman, Rosset, Perlich and Stitelman, ACM TKDD 6(4), Art. 15, 2012 | Crossref, DOI 10.1145/2382577.2382579; university research-portal record (2026-11-10) | Abstract only |
| pedregosa2011 | Pedregosa et al., JMLR 12, 2011, pp. 2825-2830 | Journal page jmlr.org/papers/v12/pedregosa11a.html (2026-11-10) | Cited as the software reference only |
| chen2016 | Chen and Guestrin, KDD 2016, pp. 785-794 | Crossref, DOI 10.1145/2939672.2939785 (2026-11-10) | General description of the method only |
| breiman2001 | Breiman, Machine Learning 45(1), 2001, pp. 5-32 | Crossref, DOI 10.1023/A:1010933404324 (2026-11-10) | General description of the method only |
| liu2008 | Liu, Ting and Zhou, ICDM 2008, pp. 413-422 | Crossref, DOI 10.1109/ICDM.2008.17 (2026-11-10) | General description of the method only |
| efron1979 | Efron, Annals of Statistics 7(1), 1979, pp. 1-26 | Crossref, DOI 10.1214/aos/1176344552 (2026-11-10) | Cited as the origin of the bootstrap only |

## Statements deliberately limited

- **Piecewise linear RUL target.** Cited to Laredo et al. (2019), who state that such a
  model "has been proposed" for this dataset and attribute it to Ramasso (2014). The
  thesis does not attribute the idea to a specific original paper, because no original
  source was read.
- **Window length.** Laredo et al. (2019) report windows of 20 to 30 cycles in earlier
  work. The thesis cites this as a report of earlier studies, not as a direct finding.
- **Locations of conferences** are given only where confirmed (PHM 2008, Denver, from the
  dataset readme).

## Bibliographic details confirmed, content not yet read (do not cite for content)

| Reference | Status |
|---|---|
| Heimes, "Recurrent neural networks for remaining useful life estimation", PHM 2008, pp. 1-6, DOI 10.1109/PHM.2008.4711422 | Often credited with the capped RUL target. Full text not accessible, so not cited. |
| Li, Ding and Sun, "Remaining useful life estimation in prognostics using deep convolution neural networks", Reliability Engineering and System Safety 172, 2018, pp. 1-11, DOI 10.1016/j.ress.2017.11.021 | Abstract confirms a time-window deep CNN on C-MAPSS. The cap and window values it used were not confirmed. |
| Ramasso, "Investigating computational geometry for failure prognostics", Int. J. Prognostics and Health Management 5(1), 2014, pp. 1-18 | Known only through the reference list of Laredo et al. (2019). Not checked. |

These are candidates for Chapter 2 and must be checked there before use.
