# Verification report: Korolev AI iteration 03

Status: **rejected after edge-case testing**.

- `py_compile`: PASS.
- Empty corpus: all vectors, 33 matrices and tabular artifacts are created.
  However, `quality_summary.csv` contains 24 synthetic rows instead of being
  empty with headers as previously required.
- Mixed six-document fixture: PASS.
- Full supplied 50-document corpus: PASS.
- Full-corpus BGE weighted Jaccard range: `0.225772 .. 1.000000`.
- Full-corpus BGE weighted Dice range: `0.368375 .. 1.000000`.
- Additional required edge case, two empty documents from one topic: FAIL.
  All 24 `roc_auc` fields are `NaN`.
- Root cause: current scikit-learn returns `NaN` with
  `UndefinedMetricWarning` for single-class targets instead of throwing an
  exception, so the broad `try/except` does not activate. Average precision is
  also reported as `1.0`, contrary to the documented `0.5` fallback.

The full run is structurally sound, but the explicit no-NaN requirement makes
this iteration unacceptable.
