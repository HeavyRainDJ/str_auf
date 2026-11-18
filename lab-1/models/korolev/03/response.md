```python
import os
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import spacy
from bs4 import BeautifulSoup
from docx import Document
from odf.opendocument import load
from odf.teletype import extractText
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

def read_text(file_path):
    ext = file_path.suffix.lower()
    try:
        if ext == '.txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='cp1251') as f:
                    return f.read()
        elif ext == '.html':
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
                for script_or_style in soup(['script', 'style']):
                    script_or_style.decompose()
                return soup.get_text(separator='\n')
        elif ext == '.docx':
            doc = Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        full_text.append(cell.text)
            return '\n'.join(full_text)
        elif ext == '.odt':
            odt_doc = load(file_path)
            full_text = []
            from odf import text
            for p in odt_doc.getElementsByType(text.P):
                full_text.append(extractText(p))
            return '\n'.join(full_text)
        else:
            return ""
    except Exception as e:
        raise e

def tokenize_simple(text):
    return re.findall(r'[a-z\u0400-\u04FF]+', text.lower())

def get_char_ngrams(word):
    s = f"^{word}$"
    ngrams = []
    for n in [2, 3]:
        for i in range(len(s) - n + 1):
            ngrams.append(s[i:i+n])
    return ngrams

def save_matrix_set(output_path, matrix, labels_rows, labels_cols, name):
    sparse.save_npz(output_path / f"{name}.npz", matrix)
    with open(output_path / f"{name}_rows.json", "w", encoding="utf-8") as f:
        json.dump(labels_rows, f, ensure_ascii=False)
    with open(output_path / f"{name}_cols.json", "w", encoding="utf-8") as f:
        json.dump(labels_cols, f, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to corpus folder')
    parser.add_argument('--output', required=True, help='Path to output folder')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    all_files = sorted([f for f in input_path.rglob('*') if f.suffix.lower() in {'.txt', '.docx', '.odt', '.html'}])
    
    docs_content = []
    read_errors = []
    
    for f_path in all_files:
        rel_path_posix = f_path.relative_to(input_path).as_posix()
        try:
            content = read_text(f_path)
            docs_content.append((rel_path_posix, content))
        except Exception as e:
            print(f"Error reading {rel_path_posix}: {e}")
            read_errors.append({"file": rel_path_posix, "error": str(e)})

    if read_errors:
        with open(output_path / "read_errors.json", "w", encoding="utf-8") as f:
            json.dump(read_errors, f, ensure_ascii=False, indent=4)

    try:
        nlp = spacy.load("ru_core_news_sm")
    except OSError:
        print("Error: spacy model 'ru_core_news_sm' not found.")
        sys.exit(1)

    corpus_words_global = set()
    processed_docs = []

    for path, text_content in docs_content:
        tokens = tokenize_simple(text_content)
        corpus_words_global.update(tokens)
        spacy_doc = nlp(text_content)
        processed_docs.append((path, tokens, spacy_doc))

    sorted_vocab = sorted(list(corpus_words_global))
    word_to_idx = {w: i for i, w in enumerate(sorted_vocab)}

    # --- 1. ВЕКТОРЫ СЛОВ (n-граммы) ---
    all_char_ngrams = set()
    word_ngram_counts = []
    for word in sorted_vocab:
        ngs = get_char_ngrams(word)
        all_char_ngrams.update(ngs)
        word_ngram_counts.append(Counter(ngs))
    
    sorted_char_features = sorted(list(all_char_ngrams))
    char_feat_to_idx = {f: i for i, f in enumerate(sorted_char_features)}
    
    rows, cols, data = [], [], []
    for w_idx, counts in enumerate(word_ngram_counts):
        for ng, count in counts.items():
            rows.append(w_idx)
            cols.append(char_feat_to_idx[ng])
            data.append(count)
    matrix_word_ngram = sparse.csr_matrix((data, (rows, cols)), 
                                          shape=(len(sorted_vocab), len(sorted_char_features)))

    # --- 2. ВЕКТОРЫ ДОКУМЕНТОВ (TF-IDF) ---
    texts_for_tfidf = [" ".join(tokens) for _, tokens, _ in processed_docs]
    if not sorted_vocab or not texts_for_tfidf:
        matrix_doc_tfidf = sparse.csr_matrix((len(processed_docs), 0))
        sorted_tfidf_features = []
    else:
        vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x.split(), 
            preprocessor=lambda x: x, 
            ngram_range=(1, 2), 
            smooth_idf=True, 
            norm='l2'
        )
        matrix_doc_tfidf_raw = vectorizer.fit_transform(texts_for_tfidf)
        raw_features = vectorizer.get_feature_names_out()
        sorted_tfidf_features = sorted(list(raw_features))
        feat_map = {f: i for i, f in enumerate(sorted_tfidf_features)}
        
        coo = matrix_doc_tfidf_raw.tocoo()
        new_cols = [feat_map[raw_features[c]] for c in coo.col]
        matrix_doc_tfidf = sparse.csr_matrix((coo.data, (coo.row, new_cols)), 
                                              shape=(len(processed_docs), len(sorted_tfidf_features)))

    # --- 3. СИНТАКСИС: СЛОВА ---
    word_syntax_counts = defaultdict(Counter)
    all_syntax_features = set()

    for _, _, doc in processed_docs:
        for token in doc:
            if not token.text.isalpha(): continue
            surf = token.text.lower()
            for child in token.children:
                if not child.text.isalpha(): continue
                feat_out = f"OUT|{child.dep_}|{child.lemma_.lower()}"
                word_syntax_counts[surf][feat_out] += 1
                all_syntax_features.add(feat_out)
                feat_in = f"IN|{child.dep_}|{token.lemma_.lower()}"
                word_syntax_counts[child.text.lower()][feat_in] += 1
                all_syntax_features.add(feat_in)

    sorted_syntax_word_features = sorted(list(all_syntax_features))
    syn_word_feat_to_idx = {f: i for i, f in enumerate(sorted_syntax_word_features)}
    
    rows, cols, data = [], [], []
    for word in sorted_vocab:
        counts = word_syntax_counts[word]
        for feat, count in counts.items():
            rows.append(word_to_idx[word])
            cols.append(syn_word_feat_to_idx[feat])
            data.append(count)
    matrix_word_syntax = sparse.csr_matrix((data, (rows, cols)), 
                                           shape=(len(sorted_vocab), len(sorted_syntax_word_features)))

    # --- 4. СИНТАКСИС: ДОКУМЕНТЫ ---
    doc_syntax_counts = []
    all_doc_syn_features = set()

    for _, _, doc in processed_docs:
        counts = Counter()
        for token in doc:
            for child in token.children:
                if child.dep_ == 'ROOT' or token == child or not token.text.isalpha() or not child.text.isalpha():
                    continue
                feat = f"{token.pos_}|{child.dep_}|{child.pos_}"
                counts[feat] += 1
                all_doc_syn_features.add(feat)
        doc_syntax_counts.append(counts)

    sorted_doc_syn_features = sorted(list(all_doc_syn_features))
    doc_syn_feat_to_idx = {f: i for i, f in enumerate(sorted_doc_syn_features)}

    rows, cols, data = [], [], []
    for d_idx, counts in enumerate(doc_syntax_counts):
        for feat, count in counts.items():
            rows.append(d_idx)
            cols.append(doc_syn_feat_to_idx[feat])
            data.append(count)
    matrix_doc_syntax = sparse.csr_matrix((data, (rows, cols)), 
                                          shape=(len(processed_docs), len(sorted_doc_syn_features)))

    doc_labels = [p for p, _, _ in processed_docs]
    save_matrix_set(output_path, matrix_word_ngram, sorted_vocab, sorted_char_features, "word_ngram")
    save_matrix_set(output_path, matrix_doc_tfidf, doc_labels, sorted_tfidf_features, "doc_tfidf")
    save_matrix_set(output_path, matrix_word_syntax, sorted_vocab, sorted_syntax_word_features, "word_syntax")
    save_matrix_set(output_path, matrix_doc_syntax, doc_labels, sorted_doc_syn_features, "doc_syntax")

    metadata = {
        "word_ngram": {"shape": matrix_word_ngram.shape},
        "doc_tfidf": {"shape": matrix_doc_tfidf.shape},
        "word_syntax": {"shape": matrix_word_syntax.shape},
        "doc_syntax": {"shape": matrix_doc_syntax.shape},
        "total_docs": len(processed_docs),
        "total_unique_words": len(sorted_vocab)
    }
    with open(output_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    if read_errors:
        sys.exit(1)

if __name__ == "__main__":
    main()
```