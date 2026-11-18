#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import base64
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as stats
from bs4 import BeautifulSoup
from docx import Document
from odf.opendocument import load
from odf.table import Table, TableRow, TableCell
from odf.text import P, H
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from sklearn.preprocessing import normalize
import torch
from transformers import AutoModel, AutoTokenizer

# ---------- Конфигурация и типы ----------

@dataclass
class DocumentMeta:
    rel_path: str
    theme: str
    ext: str
    text_len: int

# ---------- Извлечение текста ----------

def extract_text_txt(path: Path) -> str:
    for enc in ['utf-8-sig', 'utf-8', 'cp1251']:
        try:
            with open(path, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Не удалось определить кодировку TXT: {path}")

def extract_text_html(path: Path) -> str:
    try:
        with open(path, 'rb') as f:
            raw = f.read()
    except Exception as e:
        raise ValueError(f"Ошибка чтения HTML {path}: {e}")

    encodings = ['utf-8-sig', 'utf-8', 'cp1251']
    for enc in encodings:
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Не удалось определить кодировку HTML: {path}")

    soup = BeautifulSoup(text, 'html.parser')
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()
    return soup.get_text(separator=' ', strip=True)

def extract_text_docx(path: Path) -> str:
    try:
        doc = Document(path)
    except Exception as e:
        raise ValueError(f"Ошибка чтения DOCX {path}: {e}")

    parts = []
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    parts.append(cell.text)
    return ' '.join(parts)

def extract_text_odt(path: Path) -> str:
    try:
        odt = load(str(path))
    except Exception as e:
        raise ValueError(f"Ошибка чтения ODT {path}: {e}")

    parts = []
    for element in odt.getElementsByType(P):
        text = ''.join(node.data for node in element.childNodes if hasattr(node, 'data'))
        if text.strip():
            parts.append(text)
    for element in odt.getElementsByType(H):
        text = ''.join(node.data for node in element.childNodes if hasattr(node, 'data'))
        if text.strip():
            parts.append(text)
    for table in odt.getElementsByType(Table):
        for row in table.getElementsByType(TableRow):
            for cell in row.getElementsByType(TableCell):
                cell_text = []
                for p in cell.getElementsByType(P):
                    text = ''.join(node.data for node in p.childNodes if hasattr(node, 'data'))
                    if text.strip():
                        cell_text.append(text)
                if cell_text:
                    parts.append(' '.join(cell_text))
    return ' '.join(parts)

def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == '.txt':
        return extract_text_txt(path)
    if ext == '.html':
        return extract_text_html(path)
    if ext == '.docx':
        return extract_text_docx(path)
    if ext == '.odt':
        return extract_text_odt(path)
    raise ValueError(f"Неподдерживаемый формат: {path}")

# ---------- Предобработка и векторизация ----------

def build_bow(docs_texts: List[str]) -> Tuple[sp.csr_matrix, List[str]]:
    vocab: Dict[str, int] = {}
    rows, cols, data = [], [], []
    for i, text in enumerate(docs_texts):
        tokens = re.findall(r'\w+', text.lower())
        counts = Counter(tokens)
        for token, cnt in counts.items():
            if token not in vocab:
                vocab[token] = len(vocab)
            rows.append(i)
            cols.append(vocab[token])
            data.append(cnt)
    matrix = sp.csr_matrix((data, (rows, cols)), shape=(len(docs_texts), len(vocab)), dtype=np.float32)
    return matrix, sorted(vocab.keys(), key=lambda x: vocab[x])

def build_tfidf(docs_texts: List[str]) -> Tuple[sp.csr_matrix, List[str]]:
    import pymorphy3
    morph = pymorphy3.MorphAnalyzer()
    def lemmatize(text: str) -> str:
        tokens = re.findall(r'\w+', text)
        lemmas = [morph.parse(t)[0].normal_form for t in tokens]
        return ' '.join(lemmas)
    lemmatized = [lemmatize(t) for t in docs_texts]
    vectorizer = TfidfVectorizer(
        norm='l2',
        smooth_idf=True,
        lowercase=True,
        token_pattern=r'(?u)\b\w+\b'
    )
    matrix = vectorizer.fit_transform(lemmatized)
    return matrix, vectorizer.get_feature_names_out().tolist()

def load_bge_m3(device: str, cache_dir: Optional[str] = None):
    model_name = "BAAI/bge-m3"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    model.to(device)
    model.eval()
    return tokenizer, model

def chunk_text(text: str, chunk_words: int) -> List[str]:
    tokens = re.findall(r'\w+', text)
    if not tokens:
        return []
    chunks = []
    for i in range(0, len(tokens), chunk_words):
        chunks.append(' '.join(tokens[i:i + chunk_words]))
    return chunks

def encode_bge_m3(texts: List[str], tokenizer, model, device: str, batch_size: int, chunk_words: int) -> np.ndarray:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_chunks = []
        chunk_map = []
        for j, text in enumerate(batch_texts):
            chunks = chunk_text(text, chunk_words)
            if not chunks:
                chunks = ['']
            batch_chunks.extend(chunks)
            chunk_map.extend([j] * len(chunks))

        enc = tokenizer(batch_chunks, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**enc)
            if hasattr(outputs, 'dense_vec'):
                vecs = outputs.dense_vec
            else:
                last_hidden = outputs.last_hidden_state
                attention_mask = enc['attention_mask']
                sum_emb = (last_hidden * attention_mask.unsqueeze(-1)).sum(dim=1)
                mask_sum = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
                vecs = sum_emb / mask_sum

        vecs = vecs.cpu().numpy()
        vecs = normalize(vecs, norm='l2')

        doc_vecs = np.zeros((len(batch_texts), vecs.shape[1]), dtype=np.float32)
        counts = np.zeros(len(batch_texts), dtype=np.int32)
        for idx, doc_idx in enumerate(chunk_map):
            doc_vecs[doc_idx] += vecs[idx]
            counts[doc_idx] += 1
        counts[counts == 0] = 1
        doc_vecs = doc_vecs / counts[:, None]
        doc_vecs = normalize(doc_vecs, norm='l2')
        all_embeddings.append(doc_vecs)
    return np.vstack(all_embeddings)

# ---------- Меры схожести и расстояний ----------

def safe_cosine(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx == 0 or ny == 0:
        return 0.0
    return float(np.dot(x, y) / (nx * ny))

def weighted_jaccard(x: np.ndarray, y: np.ndarray) -> float:
    min_sum = float(np.sum(np.minimum(x, y)))
    max_sum = float(np.sum(np.maximum(x, y)))
    if max_sum == 0:
        return 1.0
    return min_sum / max_sum

def weighted_dice(x: np.ndarray, y: np.ndarray) -> float:
    min_sum = float(np.sum(np.minimum(x, y)))
    sum_x, sum_y = float(np.sum(x)), float(np.sum(y))
    denom = sum_x + sum_y
    if denom == 0:
        return 1.0
    return 2 * min_sum / denom

def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0
    return float(stats.pearsonr(x, y)[0])

def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return 0.0
    return float(stats.spearmanr(x, y)[0])

def euclidean_dist(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.linalg.norm(x - y))

def manhattan_dist(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum(np.abs(x - y)))

def chebyshev_dist(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.max(np.abs(x - y)))

def build_matrices(vectors: np.ndarray, is_sparse: bool = False) -> Dict[str, np.ndarray]:
    n = vectors.shape[0]
    sim_cos = np.zeros((n, n), dtype=np.float32)
    sim_jac = np.zeros((n, n), dtype=np.float32)
    sim_dice = np.zeros((n, n), dtype=np.float32)
    corr_pearson = np.zeros((n, n), dtype=np.float32)
    corr_spearman = np.zeros((n, n), dtype=np.float32)
    dist_euc = np.zeros((n, n), dtype=np.float32)
    dist_man = np.zeros((n, n), dtype=np.float32)
    dist_cheb = np.zeros((n, n), dtype=np.float32)

    if is_sparse:
        vecs_dense = vectors.toarray()
    else:
        vecs_dense = vectors

    for i in range(n):
        sim_cos[i, i] = 1.0
        sim_jac[i, i] = 1.0
        sim_dice[i, i] = 1.0
        corr_pearson[i, i] = 1.0
        corr_spearman[i, i] = 1.0
        dist_euc[i, i] = 0.0
        dist_man[i, i] = 0.0
        dist_cheb[i, i] = 0.0
        for j in range(i + 1, n):
            xi, yi = vecs_dense[i], vecs_dense[j]

            c = safe_cosine(xi, yi)
            jac = weighted_jaccard(xi, yi)
            dice = weighted_dice(xi, yi)
            pears = pearson_corr(xi, yi)
            spear = spearman_corr(xi, yi)
            euc = euclidean_dist(xi, yi)
            man = manhattan_dist(xi, yi)
            cheb = chebyshev_dist(xi, yi)

            sim_cos[i, j] = sim_cos[j, i] = c
            sim_jac[i, j] = sim_jac[j, i] = jac
            sim_dice[i, j] = sim_dice[j, i] = dice
            corr_pearson[i, j] = corr_pearson[j, i] = pears
            corr_spearman[i, j] = corr_spearman[j, i] = spear
            dist_euc[i, j] = dist_euc[j, i] = euc
            dist_man[i, j] = dist_man[j, i] = man
            dist_cheb[i, j] = dist_cheb[j, i] = cheb

    close_euc = 1.0 / (1.0 + dist_euc)
    close_man = 1.0 / (1.0 + dist_man)
    close_cheb = 1.0 / (1.0 + dist_cheb)

    return {
        'cosine': sim_cos,
        'weighted_jaccard': sim_jac,
        'weighted_dice': sim_dice,
        'pearson': corr_pearson,
        'spearman': corr_spearman,
        'euclidean': dist_euc,
        'manhattan': dist_man,
        'chebyshev': dist_cheb,
        'euclidean_close': close_euc,
        'manhattan_close': close_man,
        'chebyshev_close': close_cheb,
    }

# ---------- Метрики качества ----------

def compute_quality(matrices: Dict[str, np.ndarray], themes: List[str]) -> pd.DataFrame:
    n = len(themes)
    same_theme = np.array([themes[i] == themes[j] for i in range(n) for j in range(n)]).reshape(n, n)
    records = []
    for name, mat in matrices.items():
        if 'distance' in name:
            continue
        values = mat[np.triu_indices(n, k=1)]
        labels = same_theme[np.triu_indices(n, k=1)]
        if len(np.unique(labels)) < 2:
            auc, ap = 0.5, 0.5
        else:
            try:
                auc = roc_auc_score(labels, values)
            except ValueError:
                auc = 0.5
            try:
                ap = average_precision_score(labels, values)
            except ValueError:
                ap = 0.5
        intra = float(values[labels].mean()) if labels.any() else 0.0
        inter = float(values[~labels].mean()) if (~labels).any() else 0.0
        diff = intra - inter
        records.append({
            'representation': name.split('__')[0],
            'metric': name,
            'type': 'similarity',
            'intra_mean': intra,
            'inter_mean': inter,
            'diff': diff,
            'roc_auc': auc,
            'avg_precision': ap
        })
    return pd.DataFrame(records)

# ---------- Сохранение ----------

def save_sparse_npz(matrix: sp.csr_matrix, path: Path):
    sp.save_npz(path, matrix)

def save_dense_npy(array: np.ndarray, path: Path):
    np.save(path, array)

def save_matrix_csv(matrix: np.ndarray, names: List[str], path: Path):
    df = pd.DataFrame(matrix, index=names, columns=names)
    df.to_csv(path, float_format='%.6f', encoding='utf-8-sig')

def save_all_pairs(docs: List[DocumentMeta], matrices: Dict[str, Dict[str, np.ndarray]], out_dir: Path):
    rows = []
    n = len(docs)
    for i in range(n):
        for j in range(i + 1, n):
            for rep, mats in matrices.items():
                for metric, mat in mats.items():
                    val = float(mat[i, j])
                    rows.append({
                        'doc_a': docs[i].rel_path,
                        'doc_b': docs[j].rel_path,
                        'theme_a': docs[i].theme,
                        'theme_b': docs[j].theme,
                        'same_theme': docs[i].theme == docs[j].theme,
                        'representation': rep,
                        'metric': metric,
                        'type': 'similarity' if 'distance' not in metric else 'distance',
                        'value': val
                    })
    pd.DataFrame(rows).to_csv(out_dir / 'all_pairs.csv', index=False, encoding='utf-8-sig')

def save_nearest_neighbors(docs: List[DocumentMeta], matrices: Dict[str, Dict[str, np.ndarray]], out_dir: Path, top_k: int):
    rows = []
    n = len(docs)
    for i in range(n):
        for rep, mats in matrices.items():
            for metric, mat in mats.items():
                if 'distance' in metric:
                    continue
                sims = mat[i].copy()
                sims[i] = -1.0
                nn_idx = np.argsort(-sims)[:top_k]
                for rank, j in enumerate(nn_idx, start=1):
                    rows.append({
                        'doc': docs[i].rel_path,
                        'representation': rep,
                        'metric': metric,
                        'neighbor_rank': rank,
                        'neighbor_doc': docs[j].rel_path,
                        'neighbor_theme': docs[j].theme,
                        'similarity': float(mat[i, j])
                    })
    pd.DataFrame(rows).to_csv(out_dir / 'nearest_neighbors.csv', index=False, encoding='utf-8-sig')

def save_heatmaps(matrices: Dict[str, Dict[str, np.ndarray]], docs: List[DocumentMeta], out_dir: Path):
    import matplotlib.pyplot as plt
    names = [d.rel_path for d in docs]
    for rep, mats in matrices.items():
        if 'cosine' in mats:
            plt.figure(figsize=(max(8, len(names) * 0.2), max(6, len(names) * 0.2)))
            plt.imshow(mats['cosine'], cmap='viridis', vmin=0, vmax=1)
            plt.colorbar(label='Cosine')
            plt.xticks(ticks=range(len(names)), labels=names, rotation=90)
            plt.yticks(ticks=range(len(names)), labels=names)
            plt.title(f'{rep} — Cosine Similarity')
            plt.tight_layout()
            plt.savefig(out_dir / f'{rep}__cosine.png', dpi=150)
            plt.close()

# ---------- Основная логика ----------

def discover_docs(root: Path) -> List[Path]:
    exts = {'.txt', '.html', '.docx', '.odt'}
    paths = []
    for path in sorted(root.rglob('*'), key=lambda p: p.relative_to(root).as_posix()):
        if path.is_file() and path.suffix.lower() in exts:
            paths.append(path)
    return paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--chunk-words', type=int, default=200)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--model-cache', default=None)
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    if not input_dir.exists():
        print(f"Каталог не найден: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'vectors').mkdir(exist_ok=True)
    (output_dir / 'matrices').mkdir(exist_ok=True)

    doc_paths = discover_docs(input_dir)
    if not doc_paths:
        print("Входной каталог не содержит документов. Создана только структура.", file=sys.stderr)
        meta_json = {'documents': [], 'params': vars(args), 'notes': 'Пустой корпус'}
        with open(output_dir / 'run_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(meta_json, f, ensure_ascii=False, indent=2)
        return

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    texts = []
    metas = []
    themes = []
    for path in doc_paths:
        try:
            text = extract_text(path)
        except Exception as e:
            print(f"Ошибка извлечения {path.relative_to(input_dir)}: {e}", file=sys.stderr)
            sys.exit(1)
        rel = path.relative_to(input_dir).as_posix()
        theme = path.parent.relative_to(input_dir).as_posix() if path.parent != input_dir else ''
        metas.append(DocumentMeta(rel_path=rel, theme=theme, ext=path.suffix.lower(), text_len=len(text)))
        texts.append(text)
        themes.append(theme)

    bow_matrix, bow_features = build_bow(texts)
    tfidf_matrix, tfidf_features = build_tfidf(texts)

    bge_vectors = None
    if texts:
        tokenizer, model = load_bge_m3(device, args.model_cache)
        bge_vectors = encode_bge_m3(texts, tokenizer, model, device, args.batch_size, args.chunk_words)

    bow_mats = build_matrices(bow_matrix, is_sparse=True)
    tfidf_mats = build_matrices(tfidf_matrix, is_sparse=True)
    bge_mats = build_matrices(bge_vectors, is_sparse=False) if bge_vectors is not None else {}

    matrices = {
        'count_bow': bow_mats,
        'tfidf_lemma': tfidf_mats,
        'bge_m3': bge_mats
    }

    with open(output_dir / 'documents.json', 'w', encoding='utf-8') as f:
        json.dump({'documents': [asdict(m) for m in metas]}, f, ensure_ascii=False, indent=2)

    with open(output_dir / 'extracted_texts.jsonl', 'w', encoding='utf-8') as f:
        for m, t in zip(metas, texts):
            f.write(json.dumps({'path': m.rel_path, 'text': t}, ensure_ascii=False) + '\n')

    save_sparse_npz(bow_matrix, output_dir / 'vectors' / 'count_bow.npz')
    with open(output_dir / 'vectors' / 'count_features.json', 'w', encoding='utf-8') as f:
        json.dump(bow_features, f, ensure_ascii=False)

    save_sparse_npz(tfidf_matrix, output_dir / 'vectors' / 'tfidf_lemma.npz')
    with open(output_dir / 'vectors' / 'tfidf_features.json', 'w', encoding='utf-8') as f:
        json.dump(tfidf_features, f, ensure_ascii=False)

    if bge_vectors is not None:
        save_dense_npy(bge_vectors, output_dir / 'vectors' / 'bge_m3.npy')

    for rep, mats in matrices.items():
        for metric, mat in mats.items():
            save_matrix_csv(mat, [m.rel_path for m in metas], output_dir / 'matrices' / f'{rep}__{metric}.csv')

    save_all_pairs(metas, matrices, output_dir)
    save_nearest_neighbors(metas, matrices, output_dir, args.top_k)

    quality = compute_quality(matrices, themes)
    quality.to_csv(output_dir / 'quality_summary.csv', index=False, encoding='utf-8-sig')

    import matplotlib
    matplotlib.use('Agg')
    save_heatmaps(matrices, metas, output_dir)

    import importlib.metadata
    meta = {
        'params': vars(args),
        'timestamps': {'start': time.time()},
        'library_versions': {
            'python': sys.version.split()[0],
            'numpy': importlib.metadata.version('numpy'),
            'scipy': importlib.metadata.version('scipy'),
            'pandas': importlib.metadata.version('pandas'),
            'torch': importlib.metadata.version('torch'),
            'transformers': importlib.metadata.version('transformers'),
            'pymorphy3': importlib.metadata.version('pymorphy3'),
            'scikit-learn': importlib.metadata.version('scikit-learn')
        },
        'shapes': {
            'count_bow': bow_matrix.shape,
            'tfidf_lemma': tfidf_matrix.shape,
            'bge_m3': bge_vectors.shape if bge_vectors is not None else None
        }
    }
    meta['timestamps']['end'] = time.time()
    with open(output_dir / 'run_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
