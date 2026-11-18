# Verification report: Korolev AI iteration 04

Status: **accepted**.

- `py_compile`: PASS.
- Empty corpus (0 documents): PASS.
- Two completely empty documents from one topic: PASS; quality output contains
  no NaN/Inf and uses the documented `0.5` fallback.
- Mixed fixture (6 documents; TXT UTF-8, TXT CP1251, HTML, DOCX, ODT,
  empty TXT): PASS.
- Full supplied corpus (50 documents): PASS.
- Required vector shapes: count BoW `50 x 7549`, lemma TF-IDF `50 x 4633`,
  BGE-M3 `50 x 1024`.
- All 33 matrices are present, finite, symmetric, and have valid diagonals.
- BGE weighted Jaccard range: `0.225772 .. 1.000000`.
- BGE weighted Dice range: `0.368375 .. 1.000000`.
- Best full-corpus ROC AUC: BGE Spearman `0.943656`; signed-split BGE
  weighted Jaccard and Dice both `0.942395`.
- Metadata records library versions, signed-split, fallback rules and separate
  stage timings.

The accepted code and full output were copied to `models/korolev-ai/final/`.
