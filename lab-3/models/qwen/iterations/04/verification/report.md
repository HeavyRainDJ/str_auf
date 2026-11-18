# Verification report: Qwen iteration 04

Status: **accepted**.

- `py_compile`: PASS.
- Empty corpus (0 documents): PASS.
- Mixed fixture (6 documents; TXT UTF-8, TXT CP1251, HTML, DOCX, ODT,
  empty TXT): PASS.
- Full supplied corpus (50 documents): PASS.
- Required vector shapes: count BoW `50 x 7165`, lemma TF-IDF `50 x 4480`,
  BGE-M3 `50 x 1024`.
- All 33 matrices are present, finite, symmetric, and have valid diagonals.
- BGE weighted Jaccard range: `0.286407 .. 1.000000`.
- BGE weighted Dice range: `0.445283 .. 1.000000`.
- `run_metadata.json` explicitly documents the BGE signed-split method.
- Best full-corpus ROC AUC: BGE cosine / Euclidean closeness `0.953218`;
  signed-split BGE weighted Jaccard and Dice both `0.945851`.

The accepted code and full output were copied to `models/qwen/final/`.
