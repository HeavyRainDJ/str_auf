import os
import sys
import json
import argparse
import time
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cosine, euclidean, cityblock, chebyshev
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import roc_auc_score, average_precision_score
from bs4 import BeautifulSoup
from docx import Document
from odf import text, OpenDocumentText
import pymorphy3
from sentence_transformers import SentenceTransformer
import torch
import matplotlib.pyplot as plt

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class TextExtractor:
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()

    def extract(self, file_path: Path) -> str:
        ext = file_path.suffix.lower()
        try:
            if ext == '.txt':
                return self._read_txt(file_path)
            elif ext == '.html':
                return self._read_html(file_path)
            elif ext == '.docx':
                return self._read_docx(file_path)
            elif ext == '.odt':
                return self._read_odt(file_path)
            else:
                raise ValueError(f"Unsupported extension {ext}")
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            sys.exit(1)

    def _read_txt(self, path: Path) -> str:
        for enc in ['utf-8-sig', 'utf-8', 'cp1251']:
            try:
                return path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError(f"Unable to decode {path} with utf-8/cp1251")

    def _read_html(self, path: Path) -> str:
        content = self._read_txt(path)
        soup = BeautifulSoup(content, 'html.parser')
        for tag in soup(['script', 'style', 'noscript']):
            tag.decompose()
        return soup.get_text(separator=' ', strip=True)

    def _read_docx(self, path: Path) -> str:
        doc = Document(path)
        texts = []
        for para in doc.paragraphs:
            if para.text.strip():
                texts.append(para.text)
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        texts.append(cell.text)
        return "\n".join(texts)

    def _read_odt(self, path: Path) -> str:
        odt = OpenDocumentText(str(path))
        texts = []
        # Параграфы и заголовки
        for element in odt.getElementsByType(text.P):
            t = element.getText()
            if t.strip():
                texts.append(t)
        # Таблицы
        for table in odt.getElementsByType(text.Table):
            for row in table.getElementsByType(text.TableRow):
                for cell in row.getElementsByType(text.TableCell):
                    for p in cell.getElementsByType(text.P):
                        t = p.getText()
                        if t.strip():
                            texts.append(t)
        return "\n".join(texts)

    def lemmatize_text(self, text: str) -> str:
        words = re.findall(r'\w+', text.lower())
        return " ".join([self.morph.parse(w)[0].normal_form for w in words])

def calculate_metrics(X: np.ndarray, names: List[str]):
    n = X.shape[0]
    # Pre-calculate norms for cosine
    norms = np.linalg.norm(X, axis=1)
    
    results = {}
    
    # 1. Cosine
    cos_mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            if norms[i] == 0 or norms[j] == 0:
                val = 0.0
            else:
                val = np.dot(X[i], X[j]) / (norms[i] * norms[j])
            cos_mat[i, j] = cos_mat[j, i] = val
    results['cosine'] = ('similarity', cos_mat)

    # 2. Weighted Jaccard
    jacc_mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            mi = np.minimum(X[i], X[j]).sum()
            ma = np.maximum(X[i], X[j]).sum()
            val = mi / ma if ma != 0 else (1.0 if mi == 0 else 0.0)
            jacc_mat[i, j] = jacc_mat[j, i] = val
    results['weighted_jaccard'] = ('similarity', jacc_mat)

    # 3. Weighted Dice
    dice_mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            mi = np.minimum(X[i], X[j]).sum()
            si = X[i].sum() + X[j].sum()
            val = (2 * mi / si) if si != 0 else (1.0 if mi == 0 else 0.0)
            dice_mat[i, j] = dice_mat[j, i] = val
    results['weighted_dice'] = ('similarity', dice_mat)

    # 4. Pearson
    pear_mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            if np.std(X[i]) == 0 or np.std(X[j]) == 0:
                val = 0.0
            else:
                val, _ = pearsonr(X[i], X[j])
            pear_mat[i, j] = pear_mat[j, i] = val
    results['pearson'] = ('similarity', pear_mat)

    # 5. Spearman
    spear_mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            if np.std(X[i]) == 0 or np.std(X[j]) == 0:
                val = 0.0
            else:
                val, _ = spearmanr(X[i], X[j])
            spear_mat[i, j] = spear_mat[j, i] = val
    results['spearman'] = ('similarity', spear_mat)

    # 6. Metrics (Distances)
    metrics_list = [
        ('euclidean', euclidean),
        ('manhattan', cityblock),
        ('chebyshev', chebyshev)
    ]
    
    for m_name, m_func in metrics_list:
        dist_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                val = m_func(X[i], X[j])
                dist_mat[i, j] = dist_mat[j, i] = val
        
        # Proximity: 1 / (1 + dist)
        prox_mat = 1.0 / (1.0 + dist_mat)
        np.fill_diagonal(prox_mat, 1.0)
        
        results[m_name] = ('distance', dist_mat)
        results[f"{m_name}_proximity"] = ('similarity', prox_mat)

    return results

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

    start_time = time.time()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'vectors').mkdir(exist_ok=True)
    (out_dir / 'matrices').mkdir(exist_ok=True)

    # 1. Collection and Extraction
    corpus_path = Path(args.input)
    doc_files = []
    for root, dirs, files in os.walk(corpus_path):
        for f in files:
            if f.lower().endswith(('.txt', '.html', '.docx', '.odt')):
                doc_files.append(Path(root) / f)
    
    if not doc_files:
        logger.error("No documents found.")
        sys.exit(0)

    doc_files.sort(key=lambda x: str(x.relative_to(corpus_path)))
    
    extractor = TextExtractor()
    docs_metadata = []
    texts = []
    
    with open(out_dir / 'extracted_texts.jsonl', 'w', encoding='utf-8') as f_jsonl:
        for dp in doc_files:
            rel_path = str(dp.relative_to(corpus_path))
            topic = dp.parent.relative_to(corpus_path).name if dp.parent != corpus_path else "root"
            txt = extractor.extract(dp)
            texts.append(txt)
            docs_metadata.append({
                "path": rel_path,
                "topic": topic,
                "format": dp.suffix.lower(),
                "length": len(txt)
            })
            f_jsonl.write(json.dumps({"path": rel_path, "text": txt}, ensure_ascii=False) + "\n")

    with open(out_dir / 'documents.json', 'w', encoding='utf-8') as f:
        json.dump(docs_metadata, f, ensure_ascii=False, indent=4)

    # 2. Vectorization
    # BoW
    cv = CountVectorizer(lowercase=True, token_pattern=r'\w+')
    bow_sparse = cv.fit_transform(texts)
    sparse.save_npz(out_dir / 'vectors/count_bow.npz', bow_sparse)
    with open(out_dir / 'vectors/count_features.json', 'w', encoding='utf-8') as f:
        json.dump(cv.get_feature_names_out().tolist(), f, ensure_ascii=False)

    # TF-IDF Lemma
    lemma_texts = [extractor.lemmatize_text(t) for t in texts]
    tfidf_vec = TfidfVectorizer(lowercase=True, token_pattern=r'\w+', smooth_idf=True, norm='l2')
    tfidf_sparse = tfidf_vec.fit_transform(lemma_texts)
    sparse.save_npz(out_dir / 'vectors/tfidf_lemma.npz', tfidf_sparse)
    with open(out_dir / 'vectors/tfidf_features.json', 'w', encoding='utf-8') as f:
        json.dump(tfidf_vec.get_feature_names_out().tolist(), f, ensure_ascii=False)

    # BGE-M3
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_path = args.model_cache if args.model_cache else 'BAAI/bge-m3'
    model = SentenceTransformer(model_path, device=device)
    
    bge_vecs = []
    for t in texts:
        words = t.split()
        chunks = [" ".join(words[i:i + args.chunk_words]) for i in range(0, len(words), args.chunk_words)]
        if not chunks:
            chunk_vecs = np.zeros((1, model.get_sentence_embedding_dimension()))
        else:
            chunk_vecs = model.encode(chunks, batch_size=args.batch_size, show_progress_bar=False)
        
        doc_vec = np.mean(chunk_vecs, axis=0)
        norm = np.linalg.norm(doc_vec)
        if norm > 0:
            doc_vec = doc_vec / norm
        bge_vecs.append(doc_vec)
    
    bge_matrix = np.array(bge_vecs)
    np.save(out_dir / 'vectors/bge_m3.npy', bge_matrix)

    # 3. Similarity Matrices
    representations = {
        'count_bow': bow_sparse.toarray(),
        'tfidf_lemma': tfidf_sparse.toarray(),
        'bge_m3': bge_matrix
    }
    
    all_rep_metrics = {}
    doc_paths = [d['path'] for d in docs_metadata]

    for rep_name, data in representations.items():
        metrics = calculate_metrics(data, doc_paths)
        all_rep_metrics[rep_name] = metrics
        for m_name, (m_type, mat) in metrics.items():
            df = pd.DataFrame(mat, index=doc_paths, columns=doc_paths)
            df.to_csv(out_dir / f'matrices/{rep_name}__{m_name}.csv', encoding='utf-8-sig')

    # 4. All Pairs and Analysis
    all_pairs = []
    for rep_name, metrics in all_rep_metrics.items():
        for m_name, (m_type, mat) in metrics.items():
            for i in range(len(doc_paths)):
                for j in range(i + 1, len(doc_paths)):
                    all_pairs.append({
                        "path1": doc_paths[i],
                        "path2": doc_paths[j],
                        "topic1": docs_metadata[i]['topic'],
                        "topic2": docs_metadata[j]['topic'],
                        "same_topic": docs_metadata[i]['topic'] == docs_metadata[j]['topic'],
                        "representation": rep_name,
                        "metric": m_name,
                        "type": m_type,
                        "value": mat[i, j]
                    })
    
    df_pairs = pd.DataFrame(all_pairs)
    df_pairs.to_csv(out_dir / 'all_pairs.csv', index=False, encoding='utf-8-sig')

    # Nearest Neighbors
    nn_list = []
    for rep_name, metrics in all_rep_metrics.items():
        for m_name, (m_type, mat) in metrics.items():
            if m_type == 'similarity':
                for i in range(len(doc_paths)):
                    # Get indices of top k, excluding self
                    idx = np.argsort(mat[i])[::-1]
                    neighbors = [idx[j] for j in range(1, args.top_k + 1)]
                    for rank, n_idx in enumerate(neighbors):
                        nn_list.append({
                            "doc": doc_paths[i],
                            "rep": rep_name,
                            "metric": m_name,
                            "neighbor": doc_paths[n_idx],
                            "rank": rank + 1,
                            "value": mat[i, n_idx]
                        })
    pd.DataFrame(nn_list).to_csv(out_dir / 'nearest_neighbors.csv', index=False, encoding='utf-8-sig')

    # Quality Summary
    quality_rows = []
    for rep_name in representations.keys():
        for m_name in all_rep_metrics[rep_name].keys():
            m_type, _ = all_rep_metrics[rep_name][m_name]
            if m_type == 'distance': continue
            
            subset = df_pairs[(df_pairs['representation'] == rep_name) & (df_pairs['metric'] == m_name)]
            if subset.empty: continue
            
            in_theme = subset[subset['same_topic'] == True]['value']
            out_theme = subset[subset['same_topic'] == False]['value']
            
            mean_in = in_theme.mean() if not in_theme.empty else 0.0
            mean_out = out_theme.mean() if not out_theme.empty else 0.0
            
            y_true = subset['same_topic']
            y_score = subset['value']
            
            try:
                auc = roc_auc_score(y_true, y_score)
                ap = average_precision_score(y_true, y_score)
            except:
                auc, ap = 0.0, 0.0
            
            quality_rows.append({
                "rep": rep_name, "metric": m_name, 
                "mean_in": mean_in, "mean_out": mean_out, 
                "diff": mean_in - mean_out, "roc_auc": auc, "avg_prec": ap
            })
    pd.DataFrame(quality_rows).to_csv(out_dir / 'quality_summary.csv', index=False, encoding='utf-8-sig')

    # Heatmaps
    for rep_name in representations.keys():
        _, cos_mat = all_rep_metrics[rep_name]['cosine']
        plt.figure(figsize=(10, 8))
        plt.imshow(cos_mat, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Cosine Similarity')
        plt.title(f'Cosine Similarity - {rep_name}')
        plt.tight_layout()
        plt.savefig(out_dir / f'heatmap_{rep_name}_cosine.png')
        plt.close()

    # Metadata
    runtime = time.time() - start_time
    meta = {
        "params": vars(args),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "scikit-learn": "1.5.1"
        },
        "dimensions": {rep: representations[rep].shape for rep in representations},
        "total_runtime": runtime
    }
    with open(out_dir / 'run_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=4)

if __name__ == "__main__":
    main()
