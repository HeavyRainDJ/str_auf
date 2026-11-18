# Verification report: Korolev AI iteration 01

Status: **rejected**.

- `py_compile`: PASS.
- Runtime import: FAIL before argument processing.
- Immediate failure: `ImportError: cannot import name 'OpenDocumentText' from 'odf'`.
- ODT handling is also conceptually invalid: `OpenDocumentText` creates a new
  document and `odf.text` does not provide `Table`, `TableRow`, or `TableCell`.
- BGE weighted Jaccard and Dice are computed directly on signed embeddings;
  signed-split is absent.
- An empty corpus exits without creating any required artifacts.
- All-empty text corpora make `CountVectorizer.fit_transform` and
  `TfidfVectorizer.fit_transform` fail on an empty vocabulary.
- `--model-cache` is incorrectly used as the model identifier, which replaces
  `BAAI/bge-m3` with the cache directory path.
- Explicit `--device cuda` is not validated when CUDA is unavailable.
- Nearest-neighbor indexing assumes at least `top_k + 1` documents and may
  include the wrong documents when ties occur; it fails for small corpora.
- Input directory existence/type is not validated; ordering uses a Windows
  string rather than a relative POSIX path.
- Empty output CSV files would not have stable explicit headers.
- The pinned requirement `pymorphy3==0.9.1` is not compatible with the tested
  Python 3.12 environment used by the other model runs.

The response was preserved unchanged in `raw/response.txt`; extracted files
are in `extracted/`.
