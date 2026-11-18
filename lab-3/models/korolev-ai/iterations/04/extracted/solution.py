import os
import sys
import json
import argparse
import time
import logging
import re
import platform
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import euclidean, cityblock, chebyshev
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import roc_auc_score, average_precision_score
from bs4 import BeautifulSoup
from docx import Document
from odf import opendocument, text, table, teletype
import pymorphy3
from sentence_transformers import SentenceTransformer
import torch
import transformers
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class TextExtractor:
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()

    def extract(self, file_path: Path) -> str:
        ext = file_path.suffix.lower()
        try:
            if ext == '.txt': return self._read_txt(file_path)
            elif ext == '.html': return self._read_html(file_path)
            elif ext == '.docx': return self._read_docx(file_path)
            elif ext == '.odt': return self._read_odt(file_path)
            else: raise ValueError(f"Unsupported extension {ext}")
        except Exception as e:
            logger.error(f"Error reading {file_path.as_posix()}: {e}")
            sys.exit(1)

    def _read_txt(self, path: Path) -> str:
        for enc in ['utf-8-sig', 'utf-8', 'cp1251']:
            try:
                return path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError(f"Unable to decode {path.as_posix()}")

    def _read_html(self, path: Path) -> str:
        content = self._read_txt(path)
        soup = BeautifulSoup(content, 'html.parser')
        for tag in soup(['script', 'style', 'noscript']):
            tag.decompose()
        return soup.get_text(separator=' ', strip=True)

    def _read_docx(self, path: Path) -> str:
        doc = Document(path)
        texts = [p.text for p in doc.paragraphs if p.text.strip()]
        for t in doc.tables:
            for row in t.rows:
                for cell in row.cells:
                    if cell.text.strip(): texts.append(cell.text)
        return "\n".join(texts)

    def _read_odt(self, path: Path) -> str:
        doc = opendocument.load(str(path))
        all_text_elements = []
        for element in doc.getElementsByType(text.P):
            all_text_elements.append(teletype.extractText(element))
        for element in doc.getElementsByType(text.H):
            all_text_elements.append(teletype.extractText(element))
        texts = [t for t in all_text_elements if t and t.strip()]
        return "\n".join(texts)

    def lemmatize_text(self, text: str) -> str:
        words = re.findall(r'\w+', text.lower())
        return " ".join([self.morph.parse(w)[0].normal_form for w in words])

def calculate_metrics(X: np.ndarray, is_dense_embedding=False):
    n = X.shape[0]
    if n == 0:
        return {}

    X_overlap = X
    if is_dense_embedding:
        X_overlap = np.concatenate([np.maximum(X, 0), np.maximum(-X, 0)], axis=1)

    res = {}
    metrics_sim = ['cosine', 'weighted_jaccard', 'weighted_dice', 'pearson', 'spearman']
    metrics_dist = ['euclidean', 'manhattan', 'chebyshev']

    for m in metrics_sim + metrics_dist:
        mat = np.eye(n) if m in metrics_sim else np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                val = 0.0
                vec_i, vec_j = X[i], X[j]
                if m == 'cosine':
                    norm_i, norm_j = np.linalg.norm(vec_i), np.linalg.norm(vec_j)
                    val = np.dot(vec_i, vec_j) / (norm_i * norm_j) if norm_i > 0 and norm_j > 0 else 0.0
                elif m == 'weighted_jaccard':
                    vi, vj = X_overlap[i], X_overlap[j]
                    mi, ma = np.minimum(vi, vj).sum(), np.maximum(vi, vj).sum()
                    val = mi / ma if ma != 0 else (1.0 if mi == 0 else 0.0)
                elif m == 'weighted_dice':
                    vi, vj = X_overlap[i], X_overlap[j]
                    mi, si = np.minimum(vi, vj).sum(), vi.sum() + vj.sum()
                    val = (2 * mi / si) if si != 0 else (1.0 if mi == 0 else 0.0)
                elif m == 'pearson':
                    if np.std(vec_i) == 0 or np.std(vec_j) == 0: val = 0.0
                    else: val, _ = pearsonr(vec_i, vec_j)
                elif m == 'spearman':
                    if np.std(vec_i) == 0 or np.std(vec_j) == 0: val = 0.0
                    else: val, _ = spearmanr(vec_i, vec_j)
                elif m == 'euclidean':
                    val = euclidean(vec_i, vec_j)
                elif m == 'manhattan':
                    val = cityblock(vec_i, vec_j)
                elif m == 'chebyshev':
                    val = chebyshev(vec_i, vec_j)
                mat[i, j] = mat[j, i] = val
        
        if m in metrics_sim:
            res[m] = ('similarity', mat)
        else:
            res[m] = ('distance', mat)
            prox = 1.0 / (1.0 + mat)
            np.fill_diagonal(prox, 1.0)
            res[f"{m}_proximity"] = ('similarity', prox)
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--chunk-words', type=int, default=512)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--model-cache', default=None)
    args = parser.parse_args()

    timers = {}
    start_total = time.time()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'vectors').mkdir(exist_ok=True)
    (out_dir / 'matrices').mkdir(exist_ok=True)

    input_path = Path(args.input)
    if not input_path.exists() or not input_path.is_dir():
        logger.error(f"Input path {args.input} does not exist or is not a directory.")
        sys.exit(1)

    # 1. Discovery
    t_disc = time.time()
    doc_files = []
    for root, _, files in os.walk(input_path):
        for f in files:
            if f.lower().endswith(('.txt', '.html', '.docx', '.odt')):
                doc_files.append(Path(root) / f)
    doc_files.sort(key=lambda x: x.relative_to(input_path).as_posix())
    timers['discovery'] = time.time() - t_disc

    # 2. Extraction
    t_ext = time.time()
    extractor = TextExtractor()
    docs_metadata = []
    texts = []
    with open(out_dir / 'extracted_texts.jsonl', 'w', encoding='utf-8') as f_jsonl:
        for dp in doc_files:
            rel_path = dp.relative_to(input_path).as_posix()
            topic = dp.parent.relative_to(input_path).name if dp.parent != input_path else "root"
            txt = extractor.extract(dp)
            texts.append(txt)
            docs_metadata.append({"path": rel_path, "topic": topic, "format": dp.suffix.lower(), "length": len(txt)})
            f_jsonl.write(json.dumps({"path": rel_path, "text": txt}, ensure_ascii=False) + "\n")
    with open(out_dir / 'documents.json', 'w', encoding='utf-8') as f:
        json.dump(docs_metadata, f, ensure_ascii=False, indent=4)
    timers['extraction'] = time.time() - t_ext

    N = len(texts)
    doc_paths = [d['path'] for d in docs_metadata]

    # 3. Vectorization
    t_bow = time.time()
    if N > 0 and any(t.strip() for t in texts):
        cv = CountVectorizer(lowercase=True, token_pattern=r'\w+')
        bow_sparse = cv.fit_transform(texts)
        bow_features = cv.get_feature_names_out().tolist()
    else:
        bow_sparse = sparse.csr_matrix((N, 0))
        bow_features = []
    sparse.save_npz(out_dir / 'vectors/count_bow.npz', bow_sparse)
    with open(out_dir / 'vectors/count_features.json', 'w', encoding='utf-8') as f:
        json.dump(bow_features, f, ensure_ascii=False)
    timers['bow'] = time.time() - t_bow

    t_tfidf = time.time()
    if N > 0 and any(t.strip() for t in texts):
        lemma_texts = [extractor.lemmatize_text(t) for t in texts]
        tfidf_vec = TfidfVectorizer(lowercase=True, token_pattern=r'\w+', smooth_idf=True, norm='l2')
        tfidf_sparse = tfidf_vec.fit_transform(lemma_texts)
        tfidf_features = tfidf_vec.get_feature_names_out().tolist()
    else:
        tfidf_sparse = sparse.csr_matrix((N, 0))
        tfidf_features = []
    sparse.save_npz(out_dir / 'vectors/tfidf_lemma.npz', tfidf_sparse)
    with open(out_dir / 'vectors/tfidf_features.json', 'w', encoding='utf-8') as f:
        json.dump(tfidf_features, f, ensure_ascii=False)
    timers['tfidf'] = time.time() - t_tfidf

    bge_matrix = np.zeros((N, 0))
    if N > 0:
        t_bge_load = time.time()
        device = args.device
        if device == 'auto': device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda' and not torch.cuda.is_available():
            logger.error("CUDA requested but not available.")
            sys.exit(1)
        model = SentenceTransformer('BAAI/bge-m3', device=device, cache_folder=args.model_cache)
        timers['bge_load'] = time.time() - t_bge_load
        
        t_bge_enc = time.time()
        dim = model.get_sentence_embedding_dimension()
        bge_vecs = []
        for t in texts:
            words = t.split()
            chunks = [" ".join(words[i:i + args.chunk_words]) for i in range(0, len(words), args.chunk_words)]
            if not chunks: chunk_vecs = np.zeros((1, dim))
            else: chunk_vecs = model.encode(chunks, batch_size=args.batch_size, show_progress_bar=False)
            doc_vec = np.mean(chunk_vecs, axis=0)
            norm = np.linalg.norm(doc_vec)
            if norm > 0: doc_vec = doc_vec / norm
            bge_vecs.append(doc_vec)
        bge_matrix = np.array(bge_vecs)
        timers['bge_encode'] = time.time() - t_bge_enc
    else:
        timers['bge_load'] = 0.0
        timers['bge_encode'] = 0.0
    np.save(out_dir / 'vectors/bge_m3.npy', bge_matrix)

    # 4. Matrices
    t_mat = time.time()
    reps_config = {
        'count_bow': (bow_sparse.toarray(), False),
        'tfidf_lemma': (tfidf_sparse.toarray(), False),
        'bge_m3': (bge_matrix, True)
    }
    all_rep_metrics = {}
    sim_metrics = ['cosine', 'weighted_jaccard', 'weighted_dice', 'pearson', 'spearman']
    dist_metrics = ['euclidean', 'manhattan', 'chebyshev']
    
    for rep_name, (data, is_dense) in reps_config.items():
        if N == 0:
            metrics_dict = {}
            for m in sim_metrics: metrics_dict[m] = ('similarity', np.zeros((0, 0)))
            for m in dist_metrics: 
                metrics_dict[m] = ('distance', np.zeros((0, 0)))
                metrics_dict[f"{m}_proximity"] = ('similarity', np.zeros((0, 0)))
            all_rep_metrics[rep_name] = metrics_dict
        elif data.shape[1] == 0:
            metrics_dict = {}
            for m in sim_metrics:
                mat = np.ones((N, N)) if m in ['weighted_jaccard', 'weighted_dice'] else np.eye(N)
                metrics_dict[m] = ('similarity', mat)
            for m in dist_metrics:
                mat = np.zeros((N, N))
                metrics_dict[m] = ('distance', mat)
                metrics_dict[f"{m}_proximity"] = ('similarity', np.ones((N, N)))
            all_rep_metrics[rep_name] = metrics_dict
        else:
            all_rep_metrics[rep_name] = calculate_metrics(data, is_dense)
        
        for m_name, (m_type, mat) in all_rep_metrics[rep_name].items():
            df = pd.DataFrame(mat, index=doc_paths, columns=doc_paths)
            df.to_csv(out_dir / f'matrices/{rep_name}__{m_name}.csv', encoding='utf-8-sig')
    timers['matrices'] = time.time() - t_mat

    # 5. All Pairs & Nearest Neighbors
    t_anal = time.time()
    all_pairs_list = []
    for rep_name, metrics in all_rep_metrics.items():
        for m_name, (m_type, mat) in metrics.items():
            for i in range(N):
                for j in range(i + 1, N):
                    all_pairs_list.append({
                        "path1": doc_paths[i], "path2": doc_paths[j],
                        "topic1": docs_metadata[i]['topic'], "topic2": docs_metadata[j]['topic'],
                        "same_topic": docs_metadata[i]['topic'] == docs_metadata[j]['topic'],
                        "representation": rep_name, "metric": m_name, "type": m_type, "value": mat[i, j]
                    })
    df_pairs = pd.DataFrame(all_pairs_list)
    if df_pairs.empty:
        df_pairs = pd.DataFrame(columns=["path1", "path2", "topic1", "topic2", "same_topic", "representation", "metric", "type", "value"])
    df_pairs.to_csv(out_dir / 'all_pairs.csv', index=False, encoding='utf-8-sig')

    nn_list = []
    for rep_name, metrics in all_rep_metrics.items():
        for m_name, (m_type, mat) in metrics.items():
            if m_type == 'similarity':
                k = min(args.top_k, N - 1)
                for i in range(N):
                    if k <= 0: continue
                    idx = np.argsort(mat[i])[::-1]
                    neighbors = [idx[j] for j in range(len(idx)) if idx[j] != i][:k]
                    for rank, n_idx in enumerate(neighbors):
                        nn_list.append({"doc": doc_paths[i], "rep": rep_name, "metric": m_name,
                                        "neighbor": doc_paths[n_idx], "rank": rank + 1, "value": mat[i, n_idx]})
    df_nn = pd.DataFrame(nn_list)
    if df_nn.empty:
        df_nn = pd.DataFrame(columns=["doc", "rep", "metric", "neighbor", "rank", "value"])
    df_nn.to_csv(out_dir / 'nearest_neighbors.csv', index=False, encoding='utf-8-sig')

    quality_rows = []
    for rep_name in reps_config.keys():
        for m_name, (m_type, _) in all_rep_metrics[rep_name].items():
            if m_type == 'distance': continue
            subset = df_pairs[(df_pairs['representation'] == rep_name) & (df_pairs['metric'] == m_name)]
            if subset.empty:
                if N >= 2: # Add row only if documents exist to form pairs
                    quality_rows.append({"rep": rep_name, "metric": m_name, "mean_in": 0.0, "mean_out": 0.0, "diff": 0.0, "roc_auc": 0.5, "avg_prec": 0.5})
                continue
            
            in_t = subset[subset['same_topic'] == True]['value']
            out_t = subset[subset['same_topic'] == False]['value']
            m_in = in_t.mean() if not in_t.empty else 0.0
            m_out = out_t.mean() if not out_t.empty else 0.0
            
            # Robust ROC AUC / AP check
            y_true = subset['same_topic']
            y_score = subset['value']
            if y_true.nunique() < 2:
                auc, ap = 0.5, 0.5
            else:
                try:
                    auc = roc_auc_score(y_true, y_score)
                    ap = average_precision_score(y_true, y_score)
                except:
                    auc, ap = 0.5, 0.5
            
            # Final check for NaN/Inf
            auc = auc if np.isfinite(auc) else 0.5
            ap = ap if np.isfinite(ap) else 0.5
            
            quality_rows.append({"rep": rep_name, "metric": m_name, "mean_in": m_in, "mean_out": m_out, "diff": m_in - m_out, "roc_auc": auc, "avg_prec": ap})
    
    df_q = pd.DataFrame(quality_rows)
    if df_q.empty:
        df_q = pd.DataFrame(columns=["rep", "metric", "mean_in", "mean_out", "diff", "roc_auc", "avg_prec"])
    df_q.to_csv(out_dir / 'quality_summary.csv', index=False, encoding='utf-8-sig')
    timers['quality'] = time.time() - t_anal

    # 6. Heatmaps
    t_hm = time.time()
    for rep_name in reps_config.keys():
        if rep_name in all_rep_metrics and 'cosine' in all_rep_metrics[rep_name]:
            _, cos_mat = all_rep_metrics[rep_name]['cosine']
            if cos_mat.size > 0:
                plt.figure(figsize=(8, 6))
                plt.imshow(cos_mat, cmap='viridis')
                plt.colorbar()
                plt.title(f'Cosine Similarity: {rep_name}')
                plt.savefig(out_dir / f'heatmap_{rep_name}_cosine.png')
                plt.close()
    timers['heatmaps'] = time.time() - t_hm

    # Final Metadata
    meta = {
        "params": vars(args),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": sparse.spmatrix.__module__.split('.')[0], # a bit hacky but works
            "pandas": pd.__version__,
            "scikit-learn": "1.4.0+", # version can be checked via sklearn.__version__
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentence-transformers": "3.0.0+",
            "pymorphy3": "2.0.0+"
        },
        "dimensions": {rep: data[0].shape for rep, data in reps_config.items()},
        "signed_split_rule": "For BGE-M3, Weighted Jaccard and Weighted Dice use X_overlap = [max(0, X), max(0, -X)]",
        "fallback_rules": "N=0 or D=0: cosine/pearson/spearman=0, jaccard/dice=1, dist=0, proximity=1 (outside diagonal)",
        "fallback_auc_ap": "0.5 used when classes are missing or NaN/Inf encountered",
        "stages_runtime": timers,
        "total_runtime": time.time() - start_total
    }
    # Explicit versions fix
    import sklearn
    meta["versions"]["scikit-learn"] = sklearn.__version__
    import scipy
    meta["versions"]["scipy"] = scipy.__version__
    import sentence_transformers
    meta["versions"]["sentence-transformers"] = sentence_transformers.__version__
    
    with open(out_dir / 'run_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=4)

if __name__ == "__main__":
    main()
