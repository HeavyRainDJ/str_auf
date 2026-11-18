#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторная работа: построение векторов слов и документов
для русскоязычного корпуса на основе n-грамм и синтаксического разбора.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
from docx2python import docx2python
from odf import text, teletype
from odf.opendocument import load as load_odt

try:
    import spacy
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False


def extract_text_from_file(file_path: Path) -> str:
    """Извлекает текст из файла."""
    try:
        suffix = file_path.suffix.lower()
        
        if suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
                
        elif suffix == '.html':
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
                for script in soup(["script", "style"]):
                    script.decompose()
                return soup.get_text(separator=' ')
                
        elif suffix == '.docx':
            try:
                doc_result = docx2python(str(file_path))
                return doc_result.text
            except Exception as e:
                raise RuntimeError(f"Ошибка при чтении DOCX файла {file_path}: {e}")
                
        elif suffix == '.odt':
            try:
                odt_doc = load_odt(str(file_path))
                paragraphs = []
                for paragraph in odt_doc.getElementsByType(text.P):
                    paragraphs.append(teletype.extractText(paragraph))
                return '\n'.join(paragraphs)
            except Exception as e:
                raise RuntimeError(f"Ошибка при чтении ODT файла {file_path}: {e}")
                
        else:
            raise RuntimeError(f"Неподдерживаемый формат файла: {file_path}")
            
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise e
        else:
            raise RuntimeError(f"Ошибка при чтении файла {file_path}: {e}")


def tokenize_text(text_content: str) -> list:
    """Токенизирует текст на слова."""
    word_pattern = re.compile(r'[a-zA-Zа-яА-ЯёЁ]+')
    words = word_pattern.findall(text_content)
    return [word.lower() for word in words]


def generate_ngrams(word: str, n: int) -> list:
    """Генерирует символьные n-граммы для слова с граничными символами."""
    padded_word = '^' + word + '$'
    return [padded_word[i:i+n] for i in range(len(padded_word) - n + 1)]


def build_word_ngram_vectors(documents: list) -> tuple:
    """Строит матрицу векторов слов на основе символьных n-грамм (2 и 3)."""
    all_words = set()
    for doc_words in documents:
        all_words.update(doc_words)
    all_words = sorted(all_words)
    
    if not all_words:
        empty_matrix = sparse.csr_matrix((0, 0), dtype=np.float64)
        return empty_matrix, [], []
    
    all_ngrams = set()
    for word in all_words:
        all_ngrams.update(generate_ngrams(word, 2))
        all_ngrams.update(generate_ngrams(word, 3))
    all_ngrams = sorted(all_ngrams)
    
    ngram_to_idx = {ngram: idx for idx, ngram in enumerate(all_ngrams)}
    
    rows, cols, data = [], [], []
    
    for word_idx, word in enumerate(all_words):
        ngram_counts = Counter()
        ngram_counts.update(generate_ngrams(word, 2))
        ngram_counts.update(generate_ngrams(word, 3))
        
        for ngram, count in ngram_counts.items():
            col_idx = ngram_to_idx[ngram]
            rows.append(word_idx)
            cols.append(col_idx)
            data.append(float(count))
    
    matrix = sparse.csr_matrix(
        (data, (rows, cols)), 
        shape=(len(all_words), len(all_ngrams)),
        dtype=np.float64
    )
    
    return matrix, all_words, all_ngrams


def build_document_tfidf_vectors(documents: list) -> tuple:
    """Строит TF-IDF матрицу документов по словным униграммам и биграммам."""
    document_strings = [' '.join(doc_words) for doc_words in documents]
    
    vectorizer = TfidfVectorizer(
        token_pattern=r'[a-zA-Zа-яА-ЯёЁ]+',
        lowercase=True,
        ngram_range=(1, 2),
        smooth_idf=True,
        norm='l2'
    )
    
    if all(not s.strip() for s in document_strings):
        empty_matrix = sparse.csr_matrix((len(document_strings), 0), dtype=np.float64)
        return empty_matrix
    
    tfidf_matrix = vectorizer.fit_transform(document_strings)
    feature_names = vectorizer.get_feature_names_out().tolist()
    
    return tfidf_matrix, feature_names


def build_syntax_word_vectors(documents: list, nlp) -> tuple:
    """Строит векторы слов на основе синтаксического разбора."""
    all_words = set()
    for doc_words in documents:
        all_words.update(doc_words)
    all_words = sorted(all_words)
    
    if not all_words:
        empty_matrix = sparse.csr_matrix((0, 0), dtype=np.float64)
        return empty_matrix, [], []
    
    word_contexts = defaultdict(Counter)
    
    for doc_words in documents:
        if not doc_words:
            continue
            
        doc_text = ' '.join(doc_words)
        try:
            spacy_doc = nlp(doc_text)
        except Exception as e:
            print(f"Предупреждение: ошибка при обработке документа: {e}", file=sys.stderr)
            continue
        
        for token in spacy_doc:
            if not re.fullmatch(r'[a-zA-Zа-яА-ЯёЁ]+', token.text):
                continue
                
            surface_form = token.text.lower()
            
            # OUT контекст: обходим детей текущего токена
            for child in token.children:
                if not re.fullmatch(r'[a-zA-Zа-яА-ЯёЁ]+', child.text):
                    continue
                context = f"OUT|{child.dep_}|{child.lemma_.lower()}"
                word_contexts[surface_form][context] += 1
            
            # IN контекст: если у токена есть голова (не self-edge)
            if token.head != token:
                if re.fullmatch(r'[a-zA-Zа-яА-ЯёЁ]+', token.head.text):
                    context = f"IN|{token.dep_}|{token.head.lemma_.lower()}"
                    word_contexts[surface_form][context] += 1
    
    all_contexts = set()
    for contexts in word_contexts.values():
        all_contexts.update(contexts.keys())
    all_contexts = sorted(all_contexts)
    
    if not all_contexts:
        empty_matrix = sparse.csr_matrix((len(all_words), 0), dtype=np.float64)
        return empty_matrix, all_words, []
    
    context_to_idx = {ctx: idx for idx, ctx in enumerate(all_contexts)}
    
    rows, cols, data = [], [], []
    
    for word_idx, word in enumerate(all_words):
        if word in word_contexts:
            for context, count in word_contexts[word].items():
                col_idx = context_to_idx[context]
                rows.append(word_idx)
                cols.append(col_idx)
                data.append(float(count))
    
    matrix = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(all_words), len(all_contexts)),
        dtype=np.float64
    )
    
    return matrix, all_words, all_contexts


def build_syntax_document_vectors(documents: list, nlp) -> tuple:
    """Строит векторы документов на основе синтаксических структур."""
    document_structures = []
    
    for doc_words in documents:
        if not doc_words:
            document_structures.append(Counter())
            continue
            
        doc_text = ' '.join(doc_words)
        try:
            spacy_doc = nlp(doc_text)
        except Exception as e:
            print(f"Предупреждение: ошибка при обработке документа: {e}", file=sys.stderr)
            document_structures.append(Counter())
            continue
        
        structure_counter = Counter()
        
        for token in spacy_doc:
            if not re.fullmatch(r'[a-zA-Zа-яА-ЯёЁ]+', token.text):
                continue
            
            if token.head != token:
                if re.fullmatch(r'[a-zA-Zа-яА-ЯёЁ]+', token.head.text):
                    structure = f"{token.head.pos_}|{token.dep_}|{token.pos_}"
                    structure_counter[structure] += 1
        
        document_structures.append(structure_counter)
    
    all_structures = set()
    for structures in document_structures:
        all_structures.update(structures.keys())
    all_structures = sorted(all_structures)
    
    if not all_structures:
        empty_matrix = sparse.csr_matrix((len(documents), 0), dtype=np.float64)
        return empty_matrix, []
    
    structure_to_idx = {struct: idx for idx, struct in enumerate(all_structures)}
    
    rows, cols, data = [], [], []
    
    for doc_idx, structures in enumerate(document_structures):
        for structure, count in structures.items():
            col_idx = structure_to_idx[structure]
            rows.append(doc_idx)
            cols.append(col_idx)
            data.append(float(count))
    
    matrix = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(documents), len(all_structures)),
        dtype=np.float64
    )
    
    return matrix, all_structures


def save_matrix_and_labels(matrix: sparse.csr_matrix, row_labels: list, col_labels: list, 
                          output_path: Path, name: str):
    """Сохраняет матрицу и метки в файлы."""
    sparse.save_npz(output_path / f"{name}.npz", matrix)
    
    with open(output_path / f"{name}_rows.json", 'w', encoding='utf-8') as f:
        json.dump(row_labels, f, ensure_ascii=False, indent=2)
    
    with open(output_path / f"{name}_features.json", 'w', encoding='utf-8') as f:
        json.dump(col_labels, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Построение векторов слов и документов')
    parser.add_argument('--input', required=True, help='Путь к входному каталогу')
    parser.add_argument('--output', required=True, help='Путь к выходной директории')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Ошибка: входной каталог не существует: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    supported_extensions = {'.txt', '.docx', '.odt', '.html'}
    document_files = []
    
    for root, _, files in os.walk(input_path):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in supported_extensions:
                document_files.append(file_path)
    
    document_files.sort()
    
    documents = []
    document_paths = []
    original_texts = []
    
    for file_path in document_files:
        try:
            text_content = extract_text_from_file(file_path)
            words = tokenize_text(text_content)
            documents.append(words)
            original_texts.append(text_content)
            rel_path = file_path.relative_to(input_path).as_posix()
            document_paths.append(rel_path)
        except RuntimeError as e:
            print(f"Ошибка: {e}", file=sys.stderr)
            sys.exit(1)
    
    if not documents:
        print("Предупреждение: не найдено документов", file=sys.stderr)
        return
    
    print("Построение векторов слов на основе n-грамм...")
    word_ngram_matrix, word_ngram_rows, word_ngram_cols = build_word_ngram_vectors(documents)
    save_matrix_and_labels(word_ngram_matrix, word_ngram_rows, word_ngram_cols, output_path, "word_char")
    
    print("Построение векторов документов на основе TF-IDF...")
    doc_tfidf_matrix, doc_tfidf_cols = build_document_tfidf_vectors(documents)
    save_matrix_and_labels(doc_tfidf_matrix, document_paths, doc_tfidf_cols, output_path, "doc_tfidf")
    
    if not NLP_AVAILABLE:
        print("Ошибка: spaCy не установлен", file=sys.stderr)
        sys.exit(1)
    
    try:
        nlp = spacy.load("ru_core_news_sm")
    except OSError:
        print("Ошибка: модель ru_core_news_sm не найдена", file=sys.stderr)
        sys.exit(1)
    
    if "parser" not in nlp.pipe_names:
        print("Ошибка: модель не содержит компонента parser", file=sys.stderr)
        sys.exit(1)
    
    print("Построение синтаксических векторов слов...")
    syntax_word_matrix, syntax_word_rows, syntax_word_cols = build_syntax_word_vectors(documents, nlp)
    save_matrix_and_labels(syntax_word_matrix, syntax_word_rows, syntax_word_cols, output_path, "word_syntax")
    
    print("Построение синтаксических векторов документов...")
    syntax_doc_matrix, syntax_doc_cols = build_syntax_document_vectors(documents, nlp)
    save_matrix_and_labels(syntax_doc_matrix, document_paths, syntax_doc_cols, output_path, "doc_syntax")
    
    metadata = {
        "word_char": {
            "shape": [int(word_ngram_matrix.shape[0]), int(word_ngram_matrix.shape[1])],
            "description": "Векторы слов на основе частот символьных n-грамм длины 2 и 3"
        },
        "doc_tfidf": {
            "shape": [int(doc_tfidf_matrix.shape[0]), int(doc_tfidf_matrix.shape[1])],
            "description": "TF-IDF векторы документов по униграммам и биграммам"
        },
        "word_syntax": {
            "shape": [int(syntax_word_matrix.shape[0]), int(syntax_word_matrix.shape[1])],
            "description": "Векторы слов на основе синтаксических контекстов"
        },
        "doc_syntax": {
            "shape": [int(syntax_doc_matrix.shape[0]), int(syntax_doc_matrix.shape[1])],
            "description": "Векторы документов на основе синтаксических структур"
        },
        "parameters": {
            "ngram_lengths": [2, 3],
            "tfidf_smooth_idf": True,
            "tfidf_norm": "l2",
            "ngram_boundary_symbols": ["^", "$"]
        }
    }
    
    with open(output_path / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Обработка завершена. Результаты сохранены в {output_path}")


if __name__ == "__main__":
    main()
