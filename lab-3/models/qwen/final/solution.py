#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
solution.py — воспроизводимый сравнительный эксперимент по попарной схожести
документов на трёх представлениях и одиннадцати мерах.

Поддерживаемые форматы: .txt, .html, .docx, .odt.
Тематическая метка = имя родительского каталога первого уровня.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Опциональные зависимости (импортируются лениво, чтобы пустой корпус работал
# без тяжёлых библиотек).
# ---------------------------------------------------------------------------
try:
    import pymorphy3
    _PYMORPHY = True
except Exception:
    _PYMORPHY = False

try:
    from bs4 import BeautifulSoup
    _BS4 = True
except Exception:
    _BS4 = False

try:
    import docx as _docx
    _DOCX = True
except Exception:
    _DOCX = False

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    _TORCH = True
except Exception:
    _TORCH = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except Exception:
    _MPL = False


# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
SUPPORTED_EXTS = {".txt", ".html", ".docx", ".odt"}
WORD_RE = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)
ODF_TEXT_NS = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
BGE_M3_NAME = "BAAI/bge-m3"
BGE_M3_DIM = 1024  # hidden size модели bge-m3

SIM_METRICS = [
    "cosine",
    "weighted_jaccard",
    "weighted_dice",
    "pearson",
    "spearman",
    "euclidean_closeness",
    "manhattan_closeness",
    "chebyshev_closeness",
]
DIST_METRICS = ["euclidean", "manhattan", "chebyshev"]
REP_NAMES = ["count_bow", "tfidf_lemma", "bge_m3"]


# ---------------------------------------------------------------------------
# Извлечение текста
# ---------------------------------------------------------------------------
def read_txt(path: Path) -> str:
    data = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(
        f"Cannot decode TXT with utf-8/utf-8-sig/cp1251: {path}"
    )


def read_html(path: Path) -> str:
    if not _BS4:
        raise RuntimeError("beautifulsoup4 is required to read HTML files")
    data = path.read_bytes()
    text = None
    for enc in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError(f"Cannot decode HTML with utf-8/utf-8-sig/cp1251: {path}")
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=False)


def read_docx(path: Path) -> str:
    if not _DOCX:
        raise RuntimeError("python-docx is required to read DOCX files")
    doc = _docx.Document(str(path))
    parts: List[str] = []
    for p in doc.paragraphs:
        t = p.text
        if t.strip():
            parts.append(t)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                t = cell.text
                if t.strip():
                    parts.append(t)
    return "\n".join(parts)


def read_odt(path: Path) -> str:
    parts: List[str] = []
    with zipfile.ZipFile(path) as z:
        with z.open("content.xml") as f:
            tree = ET.parse(f)
    root = tree.getroot()
    for el in root.iter():
        if el.tag in (f"{{{ODF_TEXT_NS}}}p", f"{{{ODF_TEXT_NS}}}h"):
            t = "".join(el.itertext()).strip()
            if t:
                parts.append(t)
    return "\n".join(parts)


READERS = {
    ".txt": read_txt,
    ".html": read_html,
    ".docx": read_docx,
    ".odt": read_odt,
}


def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext not in READERS:
        raise ValueError(f"Unsupported format: {path}")
    return READERS[ext](path)


# ---------------------------------------------------------------------------
# Токенизация
# ---------------------------------------------------------------------------
def tokenize_words(text: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


# ---------------------------------------------------------------------------
# Представление 1: count_bow
# ---------------------------------------------------------------------------
def build_count_bow(texts: List[str]) -> Tuple[sp.csr_matrix, List[str]]:
    doc_counters: List[Dict[str, int]] = []
    vocab_set: set = set()
    for t in texts:
        c: Dict[str, int] = {}
        for w in tokenize_words(t):
            c[w] = c.get(w, 0) + 1
        doc_counters.append(c)
        vocab_set.update(c.keys())
    vocab = sorted(vocab_set)
    w2i = {w: i for i, w in enumerate(vocab)}
    n, m = len(texts), len(vocab)
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for i, c in enumerate(doc_counters):
        for w, v in c.items():
            rows.append(i)
            cols.append(w2i[w])
            data.append(float(v))
    mat = sp.csr_matrix(
        (data, (rows, cols)), shape=(n, m), dtype=np.float64
    )
    return mat, vocab


# ---------------------------------------------------------------------------
# Представление 2: tfidf_lemma (pymorphy3)
# ---------------------------------------------------------------------------
def build_tfidf_lemma(texts: List[str]) -> Tuple[sp.csr_matrix, List[str]]:
    if not _PYMORPHY:
        raise RuntimeError("pymorphy3 is required for TF-IDF lemma")
    
    # Проверяем, есть ли хоть одно слово во всём корпусе
    has_words = any(WORD_RE.findall(t) for t in texts)
    if not has_words:
        n = len(texts)
        return sp.csr_matrix((n, 0), dtype=np.float64), []
    
    morph = pymorphy3.MorphAnalyzer()

    def analyzer(text: str) -> List[str]:
        out: List[str] = []
        for w in WORD_RE.findall(text):
            w = w.lower()
            parses = morph.parse(w)
            out.append(parses[0].normal_form if parses else w)
        return out

    vec = TfidfVectorizer(
        analyzer=analyzer,
        smooth_idf=True,
        norm="l2",
        token_pattern=r"(?u)\b\w+\b",
    )
    mat = vec.fit_transform(texts)
    return mat, list(vec.get_feature_names_out())


# ---------------------------------------------------------------------------
# Представление 3: bge_m3 (BAAI/bge-m3)
# ---------------------------------------------------------------------------
def _split_into_word_chunks(text: str, chunk_words: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i:i + chunk_words])
            for i in range(0, len(words), chunk_words)]


def build_bge_m3(
    texts: List[str],
    device: str,
    batch_size: int,
    chunk_words: int,
    model_cache: Optional[str],
) -> np.ndarray:
    if not _TORCH:
        raise RuntimeError("torch and transformers are required for BGE-M3")
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    tok = AutoTokenizer.from_pretrained(BGE_M3_NAME, cache_dir=model_cache)
    model = AutoModel.from_pretrained(BGE_M3_NAME, cache_dir=model_cache)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    max_len = getattr(tok, "model_max_length", 8192)
    if max_len is None or max_len > 100000:
        max_len = 8192
    hidden_size = int(model.config.hidden_size)

    all_doc_vecs: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_vecs: List[np.ndarray] = []
            for text in batch:
                if not text.strip():
                    batch_vecs.append(np.zeros(hidden_size, dtype=np.float32))
                    continue
                chunks = _split_into_word_chunks(text, chunk_words)
                chunk_vecs: List[np.ndarray] = []
                for j in range(0, len(chunks), batch_size):
                    sub = chunks[j:j + batch_size]
                    enc = tok(
                        sub,
                        padding=True,
                        truncation=True,
                        max_length=max_len,
                        return_tensors="pt",
                    )
                    enc = {k: v.to(device) for k, v in enc.items()}
                    out = model(**enc)
                    hidden = out.last_hidden_state
                    mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                    norms = pooled.norm(dim=1, keepdim=True).clamp(min=1e-9)
                    pooled = pooled / norms
                    chunk_vecs.append(pooled.cpu().numpy())
                if chunk_vecs:
                    cv = np.concatenate(chunk_vecs, axis=0)
                    doc_vec = cv.mean(axis=0)
                    nrm = float(np.linalg.norm(doc_vec))
                    if nrm > 0:
                        doc_vec = doc_vec / nrm
                else:
                    doc_vec = np.zeros(hidden_size, dtype=np.float32)
                batch_vecs.append(doc_vec.astype(np.float32))
            all_doc_vecs.append(np.stack(batch_vecs, axis=0))

    if all_doc_vecs:
        return np.concatenate(all_doc_vecs, axis=0).astype(np.float32)
    return np.zeros((0, hidden_size), dtype=np.float32)


# ---------------------------------------------------------------------------
# Signed-split для Jaccard/Dice
# ---------------------------------------------------------------------------
def split_signed(X_dense: np.ndarray, rep_name: str) -> np.ndarray:
    """
    Для bge_m3 создаёт X_overlap = [max(X,0) | max(-X,0)] — конкатенация
    положительной и отрицательной частей. Это гарантирует неотрицательность
    всех компонент, что необходимо для корректного вычисления weighted
    Jaccard и Dice (значения в [0,1]).
    
    Для count_bow и tfidf_lemma возвращает исходную матрицу (она уже неотрицательна).
    """
    if rep_name == "bge_m3":
        X_pos = np.maximum(X_dense, 0)
        X_neg = np.maximum(-X_dense, 0)
        return np.concatenate([X_pos, X_neg], axis=1)
    return X_dense


# ---------------------------------------------------------------------------
# Вычисление матриц мер
# ---------------------------------------------------------------------------
def _ensure_finite(X: np.ndarray) -> np.ndarray:
    X = np.where(np.isfinite(X), X, 0.0)
    return np.ascontiguousarray(X, dtype=np.float64)


def compute_all_matrices(
    X_dense: np.ndarray,
    X_sparse: Optional[sp.csr_matrix],
    rep_name: str,
) -> Dict[str, np.ndarray]:
    """
    Возвращает 11 матриц: 8 сходств, 3 расстояния.

    Для bge_m3 weighted Jaccard и Dice вычисляются по X_overlap = [max(X,0) | max(-X,0)],
    остальные меры — по исходному X_dense.

    Правила для вырожденных случаев:
      - диагональ сходств и близостей всегда 1;
      - диагональ расстояний всегда 0;
      - если оба вектора нулевые (i != j): cosine/jaccard/dice/pearson/spearman = 0;
      - если один из векторов постоянный (std = 0): pearson/spearman = 0 (кроме диагонали);
      - если представление имеет размерность (N,0): все расстояния = 0, 
        сходства для i != j = 0;
      - все значения ограничены конечными числами (NaN/Inf заменяются на 0).
    """
    n, d = X_dense.shape
    mats: Dict[str, np.ndarray] = {}

    # Обработка случая (N,0)
    if d == 0:
        for m in SIM_METRICS:
            M = np.zeros((n, n), dtype=np.float64)
            np.fill_diagonal(M, 1.0)
            mats[m] = M
        for m in DIST_METRICS:
            mats[m] = np.zeros((n, n), dtype=np.float64)
        return mats

    # --- Cosine ---
    if X_sparse is not None:
        C = cosine_similarity(X_sparse)
    else:
        C = cosine_similarity(X_dense)
    np.fill_diagonal(C, 1.0)
    mats["cosine"] = _ensure_finite(C)

    # --- Weighted Jaccard и Dice ---
    # Для bge_m3 используем signed-split
    X_overlap = split_signed(X_dense, rep_name)
    
    # Weighted Jaccard: sum(min)/sum(max)
    J = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                J[i, j] = 1.0
                continue
            a = X_overlap[i]
            b = X_overlap[j]
            smin = float(np.minimum(a, b).sum())
            smax = float(np.maximum(a, b).sum())
            v = (smin / smax) if smax > 0 else 0.0
            J[i, j] = J[j, i] = v
    mats["weighted_jaccard"] = _ensure_finite(J)

    # Weighted Dice: 2*sum(min)/(sum(x)+sum(y))
    sums = X_overlap.sum(axis=1)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                D[i, j] = 1.0
                continue
            smin = float(np.minimum(X_overlap[i], X_overlap[j]).sum())
            denom = float(sums[i] + sums[j])
            v = (2.0 * smin / denom) if denom > 0 else 0.0
            D[i, j] = D[j, i] = v
    mats["weighted_dice"] = _ensure_finite(D)

    # --- Pearson (по признакам, с ddof=0) ---
    Xc = X_dense - X_dense.mean(axis=1, keepdims=True)
    stds = Xc.std(axis=1, ddof=0)
    P = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                P[i, j] = 1.0
                continue
            if stds[i] == 0 or stds[j] == 0:
                v = 0.0
            else:
                v = float((Xc[i] * Xc[j]).sum() / (stds[i] * stds[j] * d))
            P[i, j] = P[j, i] = v
    mats["pearson"] = _ensure_finite(P)

    # --- Spearman (ранги по признакам, затем Pearson по рангам) ---
    ranks = np.zeros_like(X_dense, dtype=np.float64)
    for i in range(n):
        ranks[i] = stats.rankdata(X_dense[i], method="average")
    Rc = ranks - ranks.mean(axis=1, keepdims=True)
    rstds = Rc.std(axis=1, ddof=0)
    S = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                S[i, j] = 1.0
                continue
            if rstds[i] == 0 or rstds[j] == 0:
                v = 0.0
            else:
                v = float((Rc[i] * Rc[j]).sum() / (rstds[i] * rstds[j] * d))
            S[i, j] = S[j, i] = v
    mats["spearman"] = _ensure_finite(S)

    # --- Метрики ---
    E = cdist(X_dense, X_dense, metric="euclidean")
    M = cdist(X_dense, X_dense, metric="cityblock")
    Ch = cdist(X_dense, X_dense, metric="chebyshev")
    np.fill_diagonal(E, 0.0)
    np.fill_diagonal(M, 0.0)
    np.fill_diagonal(Ch, 0.0)
    mats["euclidean"] = _ensure_finite(E)
    mats["manhattan"] = _ensure_finite(M)
    mats["chebyshev"] = _ensure_finite(Ch)

    # --- Близости 1/(1+d) ---
    Ec = 1.0 / (1.0 + E)
    Mc = 1.0 / (1.0 + M)
    Chc = 1.0 / (1.0 + Ch)
    np.fill_diagonal(Ec, 1.0)
    np.fill_diagonal(Mc, 1.0)
    np.fill_diagonal(Chc, 1.0)
    mats["euclidean_closeness"] = _ensure_finite(Ec)
    mats["manhattan_closeness"] = _ensure_finite(Mc)
    mats["chebyshev_closeness"] = _ensure_finite(Chc)

    return mats


# ---------------------------------------------------------------------------
# Сохранение
# ---------------------------------------------------------------------------
def save_matrix_csv(path: Path, mat: np.ndarray, labels: List[str]) -> None:
    df = pd.DataFrame(mat, index=labels, columns=labels)
    df.to_csv(path, encoding="utf-8-sig", decimal=".")


def save_heatmap(mat: np.ndarray, labels: List[str], title: str, path: Path) -> None:
    if not _MPL:
        return
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    if len(labels) <= 30:
        short = [Path(l).name[:24] for l in labels]
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(short, rotation=90, fontsize=6)
        ax.set_yticklabels(short, fontsize=6)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Сводные таблицы
# ---------------------------------------------------------------------------
def build_all_pairs(
    doc_infos: List[Dict[str, Any]],
    matrices_by_rep: Dict[str, Dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    n = len(doc_infos)
    for i in range(n):
        for j in range(i + 1, n):
            same = doc_infos[i]["topic"] == doc_infos[j]["topic"]
            for rep in REP_NAMES:
                mats = matrices_by_rep[rep]
                for m in SIM_METRICS:
                    rows.append({
                        "doc_a": doc_infos[i]["rel_path"],
                        "doc_b": doc_infos[j]["rel_path"],
                        "topic_a": doc_infos[i]["topic"],
                        "topic_b": doc_infos[j]["topic"],
                        "same_topic": int(same),
                        "representation": rep,
                        "metric": m,
                        "type": "similarity",
                        "value": float(mats[m][i, j]),
                    })
                for m in DIST_METRICS:
                    rows.append({
                        "doc_a": doc_infos[i]["rel_path"],
                        "doc_b": doc_infos[j]["rel_path"],
                        "topic_a": doc_infos[i]["topic"],
                        "topic_b": doc_infos[j]["topic"],
                        "same_topic": int(same),
                        "representation": rep,
                        "metric": m,
                        "type": "distance",
                        "value": float(mats[m][i, j]),
                    })
    cols = ["doc_a", "doc_b", "topic_a", "topic_b", "same_topic",
            "representation", "metric", "type", "value"]
    return pd.DataFrame(rows, columns=cols)


def build_nearest_neighbors(
    doc_infos: List[Dict[str, Any]],
    matrices_by_rep: Dict[str, Dict[str, np.ndarray]],
    k: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    n = len(doc_infos)
    for rep in REP_NAMES:
        mats = matrices_by_rep[rep]
        for m in SIM_METRICS:
            M = mats[m]
            for i in range(n):
                sims = M[i].copy()
                sims[i] = -np.inf
                idx = np.argsort(-sims)[:k]
                for rank, j in enumerate(idx, start=1):
                    rows.append({
                        "doc": doc_infos[i]["rel_path"],
                        "topic": doc_infos[i]["topic"],
                        "representation": rep,
                        "metric": m,
                        "rank": rank,
                        "neighbor": doc_infos[j]["rel_path"],
                        "neighbor_topic": doc_infos[j]["topic"],
                        "same_topic": int(doc_infos[i]["topic"] == doc_infos[j]["topic"]),
                        "similarity": float(M[i, j]),
                    })
    cols = ["doc", "topic", "representation", "metric", "rank",
            "neighbor", "neighbor_topic", "same_topic", "similarity"]
    return pd.DataFrame(rows, columns=cols)


def build_quality_summary(
    doc_infos: List[Dict[str, Any]],
    matrices_by_rep: Dict[str, Dict[str, np.ndarray]],
) -> pd.DataFrame:
    n = len(doc_infos)
    topics = np.array([d["topic"] for d in doc_infos])
    same_topic = topics[:, None] == topics[None, :]
    triu = np.triu(np.ones((n, n), dtype=bool), k=1)
    intra_mask = triu & same_topic
    inter_mask = triu & ~same_topic
    rows: List[Dict[str, Any]] = []
    for rep in REP_NAMES:
        mats = matrices_by_rep[rep]
        for m in SIM_METRICS:
            M = mats[m]
            intra = M[intra_mask]
            inter = M[inter_mask]
            
            notes: List[str] = []
            
            if intra.size == 0:
                mean_intra = 0.0
                notes.append("no intra-topic pairs")
            else:
                mean_intra = float(intra.mean())
            
            if inter.size == 0:
                mean_inter = 0.0
                notes.append("no inter-topic pairs")
            else:
                mean_inter = float(inter.mean())
            
            diff = mean_intra - mean_inter
            
            if intra.size == 0 or inter.size == 0:
                auc = 0.5
                ap = 0.5
                notes.append("AUC/AP undefined (no intra or inter pairs)")
            else:
                labels = np.concatenate([np.ones(intra.size), np.zeros(inter.size)])
                scores = np.concatenate([intra, inter])
                try:
                    auc = float(roc_auc_score(labels, scores))
                except Exception:
                    auc = 0.5
                    notes.append("AUC computation failed")
                try:
                    ap = float(average_precision_score(labels, scores))
                except Exception:
                    ap = 0.5
                    notes.append("AP computation failed")
            
            rows.append({
                "representation": rep,
                "metric": m,
                "mean_intra": mean_intra,
                "mean_inter": mean_inter,
                "diff": diff,
                "auc": auc,
                "average_precision": ap,
                "note": "; ".join(notes) if notes else "",
            })
    cols = ["representation", "metric", "mean_intra", "mean_inter",
            "diff", "auc", "average_precision", "note"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise document similarity experiment")
    parser.add_argument("--input", required=True, help="корпус документов")
    parser.add_argument("--output", required=True, help="выходной каталог")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-words", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model-cache", default=None, help="кэш HF-модели")
    args = parser.parse_args()

    timings: Dict[str, float] = {}
    t_total = time.time()

    input_dir = Path(args.input).resolve()
    if not input_dir.exists():
        print(f"ERROR: input path does not exist: {input_dir}", file=sys.stderr)
        sys.exit(2)
    if not input_dir.is_dir():
        print(f"ERROR: input path is not a directory: {input_dir}", file=sys.stderr)
        sys.exit(2)

    if args.device == "cuda":
        if not _TORCH:
            print("ERROR: torch is not installed, cannot use CUDA", file=sys.stderr)
            sys.exit(2)
        if not torch.cuda.is_available():
            print("ERROR: --device cuda requested but CUDA is not available", file=sys.stderr)
            sys.exit(2)

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "vectors").mkdir(exist_ok=True)
    (output_dir / "matrices").mkdir(exist_ok=True)

    # --- Сбор и сортировка документов ---
    docs: List[Tuple[str, Path]] = []
    if input_dir.exists() and input_dir.is_dir():
        for p in input_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                rel = p.relative_to(input_dir).as_posix()
                docs.append((rel, p))
    docs.sort(key=lambda x: x[0])

    # --- Извлечение текста ---
    t1 = time.time()
    doc_infos: List[Dict[str, Any]] = []
    texts: List[str] = []
    for rel, p in docs:
        try:
            text = extract_text(p)
        except Exception as e:
            print(f"ERROR reading {p}: {e}", file=sys.stderr)
            sys.exit(2)
        rel_parts = Path(rel).parts
        topic = rel_parts[0] if len(rel_parts) > 1 else ""
        doc_infos.append({
            "rel_path": rel,
            "topic": topic,
            "format": p.suffix.lower(),
            "text_length": len(text),
        })
        texts.append(text)
    timings["extract"] = time.time() - t1

    n = len(texts)
    labels = [d["rel_path"] for d in doc_infos]

    # documents.json
    (output_dir / "documents.json").write_text(
        json.dumps(doc_infos, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # extracted_texts.jsonl
    with open(output_dir / "extracted_texts.jsonl", "w", encoding="utf-8-sig") as f:
        for d, t in zip(doc_infos, texts):
            f.write(json.dumps({"rel_path": d["rel_path"], "text": t},
                               ensure_ascii=False) + "\n")

    # --- Построение представлений ---
    matrices_by_rep: Dict[str, Dict[str, np.ndarray]] = {}

    # count_bow
    t1 = time.time()
    if n > 0:
        bow_mat, bow_vocab = build_count_bow(texts)
        sp.save_npz(output_dir / "vectors" / "count_bow.npz", bow_mat)
        (output_dir / "vectors" / "count_features.json").write_text(
            json.dumps(bow_vocab, ensure_ascii=False), encoding="utf-8"
        )
        bow_dense = bow_mat.toarray().astype(np.float64)
    else:
        bow_dense = np.zeros((0, 0), dtype=np.float64)
        bow_vocab = []
        bow_mat = sp.csr_matrix((0, 0), dtype=np.float64)
        sp.save_npz(output_dir / "vectors" / "count_bow.npz", bow_mat)
        (output_dir / "vectors" / "count_features.json").write_text(
            json.dumps([], ensure_ascii=False), encoding="utf-8"
        )
    timings["count_bow"] = time.time() - t1

    # tfidf_lemma
    t1 = time.time()
    if n > 0:
        tfidf_mat, tfidf_vocab = build_tfidf_lemma(texts)
        sp.save_npz(output_dir / "vectors" / "tfidf_lemma.npz", tfidf_mat)
        (output_dir / "vectors" / "tfidf_features.json").write_text(
            json.dumps(list(tfidf_vocab), ensure_ascii=False), encoding="utf-8"
        )
        tfidf_dense = tfidf_mat.toarray().astype(np.float64)
    else:
        tfidf_dense = np.zeros((0, 0), dtype=np.float64)
        tfidf_vocab = []
        tfidf_mat = sp.csr_matrix((0, 0), dtype=np.float64)
        sp.save_npz(output_dir / "vectors" / "tfidf_lemma.npz", tfidf_mat)
        (output_dir / "vectors" / "tfidf_features.json").write_text(
            json.dumps([], ensure_ascii=False), encoding="utf-8"
        )
    timings["tfidf_lemma"] = time.time() - t1

    # bge_m3 — не загружаем модель, если документов нет
    t1 = time.time()
    if n > 0:
        bge_dense = build_bge_m3(
            texts, args.device, args.batch_size, args.chunk_words, args.model_cache
        ).astype(np.float64)
        np.save(output_dir / "vectors" / "bge_m3.npy", bge_dense.astype(np.float32))
        bge_mat = None
    else:
        bge_dense = np.zeros((0, 0), dtype=np.float64)
        bge_mat = None
        np.save(output_dir / "vectors" / "bge_m3.npy", np.zeros((0, 0), dtype=np.float32))
    timings["bge_m3"] = time.time() - t1

    dense_by_rep = {
        "count_bow": bow_dense,
        "tfidf_lemma": tfidf_dense,
        "bge_m3": bge_dense,
    }
    sparse_by_rep = {
        "count_bow": bow_mat,
        "tfidf_lemma": tfidf_mat,
        "bge_m3": bge_mat,
    }

    # --- Матрицы мер ---
    t1 = time.time()
    for rep in REP_NAMES:
        if n == 0:
            matrices_by_rep[rep] = {}
            for m in SIM_METRICS + DIST_METRICS:
                empty_mat = np.zeros((0, 0), dtype=np.float64)
                save_matrix_csv(
                    output_dir / "matrices" / f"{rep}__{m}.csv", empty_mat, labels
                )
            continue
        mats = compute_all_matrices(dense_by_rep[rep], sparse_by_rep[rep], rep)
        matrices_by_rep[rep] = mats
        for mname, M in mats.items():
            save_matrix_csv(
                output_dir / "matrices" / f"{rep}__{mname}.csv", M, labels
            )
    timings["matrices"] = time.time() - t1

    # --- all_pairs.csv ---
    t1 = time.time()
    if n >= 2:
        ap_df = build_all_pairs(doc_infos, matrices_by_rep)
    else:
        ap_df = pd.DataFrame(columns=[
            "doc_a", "doc_b", "topic_a", "topic_b", "same_topic",
            "representation", "metric", "type", "value"
        ])
    ap_df.to_csv(output_dir / "all_pairs.csv", index=False, encoding="utf-8-sig")
    timings["all_pairs"] = time.time() - t1

    # --- nearest_neighbors.csv ---
    t1 = time.time()
    if n > 1:
        nn_df = build_nearest_neighbors(doc_infos, matrices_by_rep, min(args.top_k, n - 1))
    else:
        nn_df = pd.DataFrame(columns=[
            "doc", "topic", "representation", "metric", "rank",
            "neighbor", "neighbor_topic", "same_topic", "similarity"
        ])
    nn_df.to_csv(output_dir / "nearest_neighbors.csv", index=False, encoding="utf-8-sig")
    timings["nearest_neighbors"] = time.time() - t1

    # --- quality_summary.csv ---
    t1 = time.time()
    if n >= 2:
        qs_df = build_quality_summary(doc_infos, matrices_by_rep)
    else:
        qs_df = pd.DataFrame(columns=[
            "representation", "metric", "mean_intra", "mean_inter",
            "diff", "auc", "average_precision", "note"
        ])
    qs_df.to_csv(output_dir / "quality_summary.csv", index=False, encoding="utf-8-sig")
    timings["quality_summary"] = time.time() - t1

    # --- PNG-теплокарты cosine ---
    t1 = time.time()
    for rep in REP_NAMES:
        if n > 0 and "cosine" in matrices_by_rep[rep]:
            save_heatmap(
                matrices_by_rep[rep]["cosine"],
                labels,
                f"cosine similarity — {rep}",
                output_dir / f"heatmap_cosine_{rep}.png",
            )
    timings["heatmaps"] = time.time() - t1

    # --- run_metadata.json ---
    versions: Dict[str, Optional[str]] = {}
    for mod in ("numpy", "scipy", "pandas", "sklearn", "pymorphy3",
                "torch", "transformers", "bs4", "docx", "matplotlib"):
        try:
            m = importlib.import_module(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            versions[mod] = None

    meta = {
        "args": vars(args),
        "versions": versions,
        "n_documents": n,
        "count_bow_shape": list(bow_dense.shape),
        "tfidf_lemma_shape": list(tfidf_dense.shape),
        "bge_m3_shape": list(bge_dense.shape),
        "timings": timings,
        "total_time": time.time() - t_total,
        "methodology": {
            "signed_split": {
                "description": "For bge_m3 representation, weighted Jaccard and Dice "
                              "are computed on X_overlap = [max(X,0) | max(-X,0)] — "
                              "concatenation of positive and negative parts. This ensures "
                              "all components are non-negative, guaranteeing values in [0,1].",
                "applied_to": ["bge_m3"],
                "metrics": ["weighted_jaccard", "weighted_dice"],
            },
            "other_metrics": "cosine, pearson, spearman, and all distance metrics are "
                            "computed on the original X_dense for all representations.",
        },
        "edge_cases": {
            "empty_input": n == 0,
            "single_document": n == 1,
            "note_zero_vectors": "cosine/jaccard/dice/pearson/spearman = 0 "
                                "for distinct zero/constant vectors; diagonal = 1",
            "note_empty_bge": "BGE-M3 model is not loaded when the corpus is empty",
            "note_empty_representation": "For (N,0) representations: all distances = 0, "
                                        "similarities for i != j = 0, diagonal = 1",
            "note_quality_summary": "When intra/inter pairs are absent or AUC/AP undefined: "
                                   "mean = 0, AUC/AP = 0.5, reason noted in 'note' column",
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"Done. {n} documents processed in {meta['total_time']:.2f}s",
          file=sys.stderr)


if __name__ == "__main__":
    main()