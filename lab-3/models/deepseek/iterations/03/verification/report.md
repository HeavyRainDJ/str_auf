# Verification report: DeepSeek iteration 03

Status: **accepted**.

- `py_compile`: PASS.
- Empty corpus (0 documents): PASS.
- Two completely empty documents from one topic: PASS; no NaN/Inf.
- Mixed fixture (6 documents; TXT UTF-8, TXT CP1251, HTML, DOCX, ODT,
  empty TXT): PASS.
- Full supplied corpus (50 documents): PASS.
- Required vector shapes: count BoW `50 x 7497`, lemma TF-IDF `50 x 4473`,
  BGE-M3 `50 x 1024`.
- All 33 matrices are present, finite, symmetric, and have valid diagonals.
- Long-format `all_pairs.csv`: 40,425 rows with the nine required columns.
- BGE weighted Jaccard range: `0.225772 .. 1.000000`.
- BGE weighted Dice range: `0.368375 .. 1.000000`.
- Best full-corpus ROC AUC: BGE Spearman `0.943644`; signed-split BGE
  weighted Jaccard and Dice both `0.942395`.
- Empty input, single-class quality fallback, CUDA selection, top-k, heatmaps,
  metadata and stage timings were verified.

The accepted code and full output were copied to `models/deepseek/final/`.
