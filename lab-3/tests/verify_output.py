from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


REPRESENTATIONS = ("count_bow", "tfidf_lemma", "bge_m3")
SIMILARITIES = (
    "cosine",
    "weighted_jaccard",
    "weighted_dice",
    "pearson",
    "spearman",
    "euclidean_closeness",
    "manhattan_closeness",
    "chebyshev_closeness",
)
DISTANCES = ("euclidean", "manhattan", "chebyshev")


def fail(message: str) -> None:
    raise AssertionError(message)


def load_documents(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(value, dict):
        value = value.get("documents", [])
    if not isinstance(value, list):
        fail("documents.json must contain a list or {'documents': list}")
    return value


def find_matrix(directory: Path, representation: str, metric: str) -> Path:
    candidates = list(directory.glob(f"{representation}__*.csv"))
    aliases = {
        "euclidean_closeness": ("euclidean_closeness", "euclidean_close", "euclidean_proximity", "inv_euclidean", "euclidean_similarity"),
        "manhattan_closeness": ("manhattan_closeness", "manhattan_close", "manhattan_proximity", "inv_manhattan", "manhattan_similarity"),
        "chebyshev_closeness": ("chebyshev_closeness", "chebyshev_close", "chebyshev_proximity", "inv_chebyshev", "chebyshev_similarity"),
        "euclidean": ("euclidean", "euclidean_distance"),
        "manhattan": ("manhattan", "manhattan_distance"),
        "chebyshev": ("chebyshev", "chebyshev_distance"),
        "cosine": ("cosine", "cosine_similarity"),
        "weighted_jaccard": ("weighted_jaccard", "weighted_jaccard_similarity"),
        "weighted_dice": ("weighted_dice", "weighted_dice_similarity"),
        "pearson": ("pearson", "pearson_correlation"),
        "spearman": ("spearman", "spearman_correlation"),
    }[metric]
    exact = [p for p in candidates if p.stem.split("__", 1)[1] in aliases]
    if len(exact) != 1:
        fail(f"Expected one matrix for {representation}/{metric}, found {exact}")
    return exact[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--expect-documents", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    out = args.output
    n = args.expect_documents

    required = [
        "documents.json",
        "extracted_texts.jsonl",
        "all_pairs.csv",
        "nearest_neighbors.csv",
        "quality_summary.csv",
        "run_metadata.json",
        "vectors/count_bow.npz",
        "vectors/count_features.json",
        "vectors/tfidf_lemma.npz",
        "vectors/tfidf_features.json",
        "vectors/bge_m3.npy",
    ]
    missing = [name for name in required if not (out / name).exists()]
    if missing:
        fail(f"Missing required outputs: {missing}")

    docs = load_documents(out / "documents.json")
    if len(docs) != n:
        fail(f"Expected {n} documents, found {len(docs)}")
    bow = sp.load_npz(out / "vectors/count_bow.npz")
    tfidf = sp.load_npz(out / "vectors/tfidf_lemma.npz")
    bge = np.load(out / "vectors/bge_m3.npy")
    for name, matrix in (("count_bow", bow), ("tfidf_lemma", tfidf), ("bge_m3", bge)):
        if matrix.shape[0] != n:
            fail(f"{name} row count is {matrix.shape[0]}, expected {n}")

    for rep in REPRESENTATIONS:
        for metric in SIMILARITIES + DISTANCES:
            path = find_matrix(out / "matrices", rep, metric)
            frame = pd.read_csv(path, index_col=0, encoding="utf-8-sig")
            matrix = frame.to_numpy(dtype=float)
            if matrix.shape != (n, n):
                fail(f"{path.name}: shape {matrix.shape}, expected {(n, n)}")
            if not np.isfinite(matrix).all():
                fail(f"{path.name}: contains NaN or Inf")
            if metric in {"weighted_jaccard", "weighted_dice"}:
                if matrix.size and (matrix.min() < -1e-9 or matrix.max() > 1.0 + 1e-9):
                    fail(
                        f"{path.name}: weighted overlap similarity outside [0, 1] "
                        f"(min={matrix.min()}, max={matrix.max()})"
                    )
            if not np.allclose(matrix, matrix.T, atol=1e-7):
                fail(f"{path.name}: matrix is not symmetric")
            target = 0.0 if metric in DISTANCES else 1.0
            if n and not np.allclose(np.diag(matrix), target, atol=1e-7):
                fail(f"{path.name}: invalid diagonal")

    pairs = pd.read_csv(out / "all_pairs.csv", encoding="utf-8-sig")
    expected_pairs = n * (n - 1) // 2 * len(REPRESENTATIONS) * (len(SIMILARITIES) + len(DISTANCES))
    if len(pairs) != expected_pairs:
        fail(f"all_pairs.csv has {len(pairs)} rows, expected {expected_pairs}")
    if not pairs.empty and not np.isfinite(pairs["value"].to_numpy(dtype=float)).all():
        fail("all_pairs.csv contains NaN or Inf")

    neighbors = pd.read_csv(out / "nearest_neighbors.csv", encoding="utf-8-sig")
    expected_neighbors = n * min(args.top_k, max(n - 1, 0)) * len(REPRESENTATIONS) * len(SIMILARITIES)
    if len(neighbors) != expected_neighbors:
        fail(f"nearest_neighbors.csv has {len(neighbors)} rows, expected {expected_neighbors}")

    quality = pd.read_csv(out / "quality_summary.csv", encoding="utf-8-sig")
    expected_quality = 0 if n < 2 else len(REPRESENTATIONS) * len(SIMILARITIES)
    if len(quality) != expected_quality:
        fail(f"quality_summary.csv has {len(quality)} rows, expected {expected_quality}")
    numeric_columns = [
        name
        for name in ("mean_intra", "intra_mean", "mean_inter", "inter_mean", "diff", "auc", "roc_auc", "average_precision", "avg_precision")
        if name in quality.columns
    ]
    numeric = quality[numeric_columns]
    if not numeric.empty and not np.isfinite(numeric.to_numpy(dtype=float)).all():
        fail("quality_summary.csv contains NaN or Inf")

    print(f"PASS: {out} ({n} documents)")


if __name__ == "__main__":
    main()
