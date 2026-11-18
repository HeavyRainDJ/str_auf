# Verification report: Qwen iteration 03

- `py_compile`: PASS.
- Empty corpus: PASS (`verify_output.py`, 0 documents).
- Mixed six-document fixture: program completed, but validation FAILED.
- Failure: `bge_m3__weighted_jaccard.csv` had values outside `[0, 1]`
  (`min=-1.5862903273768258`, `max=1.0`).
- Root cause: the response described signed-split in prose and metadata but did
  not implement it in `build_matrices`; Jaccard and Dice still used raw signed
  BGE coordinates.

Status: rejected; correction requested in iteration 04.
