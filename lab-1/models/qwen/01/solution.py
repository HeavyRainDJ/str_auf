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
from typing import List, Tuple, Dict, Set, Optional

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
from docx2python import docx2python
from odf import text, teletype
from odf.opendocument import load as load_odt

try:
    import spacy
    from spacy.lang.ru import Russian
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False


def extract_text_from_file(file_path: Path) -> str:
    """
    Извлекает текст из файла в зависимости от его формата.
    
    Args:
        file_path: Путь к файлу
        
    Returns:
        Текстовое содержимое файла
        
    Raises:
        RuntimeError: При ошибке чтения файла
    """
    try:
        suffix = file_path.suffix.lower()
        
        if suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
                
        elif suffix == '.html':
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
                # Удаляем script и style теги
                for script in soup(["script", "style"]):
                    script.decompose()
                return soup.get_text()
                
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
                # Извлекаем абзацы
                for paragraph in odt_doc.getElementsByType(text.P):
                    paragraphs.append(teletype.extractText(paragraph))
                # Извлекаем таблицы
                for table in odt_doc.getElementsByType(text.Table):
                    for row in table.getElementsByType(text.TableRow):
                        cells = []
                        for cell in row.getElementsByType(text.TableCell):
                            cell_text = teletype.extractText(cell)
                            if cell_text.strip():
                                cells.append(cell_text)
                        if cells:
                            paragraphs.append(' '.join(cells))
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


def tokenize_text(text_content: str) -> List[str]:
    """
    Токенизирует текст на слова (последовательности русских/латинских букв).
    
    Args:
        text_content: Текст для токенизации
        
    Returns:
        Список слов в нижнем регистре
    """
    # Регулярное выражение для слов (русские/латинские буквы, сохраняем ё)
    word_pattern = re.compile(r'[a-zA-Zа-яА-ЯёЁ]+')
    words = word_pattern.findall(text_content)
    return [word.lower() for word in words]


def generate_ngrams(word: str, n: int) -> List[str]:
    """
    Генерирует символьные n-граммы для слова с граничными символами.
    
    Args:
        word: Слово
        n: Длина n-граммы
        
    Returns:
        Список n-грамм
    """
    padded_word = '^' + word + '$'
    return [padded_word[i:i+n] for i in range(len(padded_word) - n + 1)]


def build_word_ngram_vectors(documents: List[List[str]]) -> Tuple[sparse.csr_matrix, List[str], List[str]]:
    """
    Строит матрицу векторов слов на основе символьных n-грамм (2 и 3).
    
    Args:
        documents: Список документов, каждый документ - список слов
        
    Returns:
        Кортеж: (матрица CSR, метки строк (слова), метки столбцов (n-граммы))
    """
    # Собираем все уникальные слова
    all_words = set()
    for doc_words in documents:
        all_words.update(doc_words)
    all_words = sorted(all_words)
    
    if not all_words:
        # Пустой корпус
        empty_matrix = sparse.csr_matrix((0, 0), dtype=np.float64)
        return empty_matrix, [], []
    
    # Собираем все возможные n-граммы
    all_ngrams = set()
    for word in all_words:
        all_ngrams.update(generate_ngrams(word, 2))
        all_ngrams.update(generate_ngrams(word, 3))
    all_ngrams = sorted(all_ngrams)
    
    ngram_to_idx = {ngram: idx for idx, ngram in enumerate(all_ngrams)}
    
    # Строим матрицу
    rows, cols, data = [], [], []
    
    for word_idx, word in enumerate(all_words):
        # Подсчитываем частоты n-грамм для текущего слова
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


def build_document_tfidf_vectors(documents: List[List[str]]) -> Tuple[sparse.csr_matrix, List[str], List[str]]:
    """
    Строит TF-IDF матрицу документов по словным униграммам и биграммам.
    
    Args:
        documents: Список документов, каждый документ - список слов
        
    Returns:
        Кортеж: (матрица CSR, метки строк (документы), метки столбцов (термины))
    """
    # Преобразуем документы в строки
    document_strings = [' '.join(doc_words) for doc_words in documents]
    
    # Создаем TF-IDF векторизатор
    vectorizer = TfidfVectorizer(
        token_pattern=r'[a-zA-Zа-яА-ЯёЁ]+',  # Сохраняем ё
        lowercase=True,
        ngram_range=(1, 2),  # Униграммы и биграммы
        smooth_idf=True,
        norm='l2'
    )
    
    if all(not s.strip() for s in document_strings):
        # Все документы пустые
        empty_matrix = sparse.csr_matrix((len(document_strings), 0), dtype=np.float64)
        return empty_matrix, [], []
    
    tfidf_matrix = vectorizer.fit_transform(document_strings)
    feature_names = vectorizer.get_feature_names_out().tolist()
    
    return tfidf_matrix, [f"doc_{i}" for i in range(len(documents))], feature_names


def build_syntax_word_vectors(documents: List[List[str]], nlp) -> Tuple[sparse.csr_matrix, List[str], List[str]]:
    """
    Строит векторы слов на основе синтаксического разбора.
    
    Args:
        documents: Список документов, каждый документ - список слов
        nlp: spaCy модель для русского языка
        
    Returns:
        Кортеж: (матрица CSR, метки строк (слова), метки столбцов (контексты))
    """
    # Собираем все уникальные слова (surface forms)
    all_words = set()
    for doc_words in documents:
        all_words.update(doc_words)
    all_words = sorted(all_words)
    
    if not all_words:
        empty_matrix = sparse.csr_matrix((0, 0), dtype=np.float64)
        return empty_matrix, [], []
    
    # Словарь для накопления контекстов каждого слова
    word_contexts = defaultdict(Counter)
    
    # Обрабатываем каждый документ
    for doc_words in documents:
        if not doc_words:
            continue
            
        doc_text = ' '.join(doc_words)
        try:
            spacy_doc = nlp(doc_text)
        except Exception as e:
            print(f"Предупреждение: ошибка при обработке документа '{doc_text[:50]}...': {e}", file=sys.stderr)
            continue
        
        # Проходим по всем токенам
        for token in spacy_doc:
            # Пропускаем корневые токены и пунктуацию
            if token.dep_ == 'ROOT' or token.is_punct:
                continue
                
            # Проверяем, что токен состоит только из букв
            if not re.fullmatch(r'[a-zA-Zа-яА-ЯёЁ]+', token.text):
                continue
                
            surface_form = token.text.lower()
            
            # OUT контекст: зависимые токены
            for child in token.children:
                if child.is_punct:
                    continue
                if not re.fullmatch(r'[a-zA-Zа-яА-ЯёЁ]+', child.text):
                    continue
                context = f"OUT|{child.dep_}|{child.lemma_.lower()}"
                word_contexts[surface_form][context] += 1
            
            # IN контекст: головной токен
            if token.head != token and not token.head.is_punct:
                if re.fullmatch(r'[a-zA-Zа-яА-ЯёЁ]+', token.head.text):
                    context = f"IN|{token.dep_}|{token.head.lemma_.lower()}"
                    word_contexts[surface_form][context] += 1
    
    # Собираем все уникальные контексты
    all_contexts = set()
    for contexts in word_contexts.values():
        all_contexts.update(contexts.keys())
    all_contexts = sorted(all_contexts)
    
    if not all_contexts:
        empty_matrix = sparse.csr_matrix((len(all_words), 0), dtype=np.float64)
        return empty_matrix, all_words, []
    
    context_to_idx = {ctx: idx for idx, ctx in enumerate(all_contexts)}
    
    # Строим матрицу
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


def build_syntax_document_vectors(documents: List[List[str]], nlp) -> Tuple[sparse.csr_matrix, List[str], List[str]]:
    """
    Строит векторы документов на основе синтаксических структур.
    
    Args:
        documents: Список документов, каждый документ - список слов
        nlp: spaCy модель для русского языка
        
    Returns:
        Кортеж: (матрица CSR, метки строк (документы), метки столбцов (структурные тройки))
    """
    document_structures = []
    
    for doc_words in documents:
        if not doc_words:
            document_structures.append(Counter())
            continue
            
        doc_text = ' '.join(doc_words)
        try:
            spacy_doc = nlp(doc_text)
        except Exception as e:
            print(f"Предупреждение: ошибка при обработке документа '{doc_text[:50]}...': {e}", file=sys.stderr)
            document_structures.append(Counter())
            continue
        
        structure_counter = Counter()
        
        for token in spacy_doc:
            if token.dep_ == 'ROOT' or token.is_punct:
                continue
                
            if token.head != token and not token.head.is_punct:
                # Создаем структурную тройку: POS_головы|dep|POS_зависимого
                structure = f"{token.head.pos_}|{token.dep_}|{token.pos_}"
                structure_counter[structure] += 1
        
        document_structures.append(structure_counter)
    
    # Собираем все уникальные структуры
    all_structures = set()
    for structures in document_structures:
        all_structures.update(structures.keys())
    all_structures = sorted(all_structures)
    
    if not all_structures:
        empty_matrix = sparse.csr_matrix((len(documents), 0), dtype=np.float64)
        return empty_matrix, [f"doc_{i}" for i in range(len(documents))], []
    
    structure_to_idx = {struct: idx for idx, struct in enumerate(all_structures)}
    
    # Строим матрицу
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
    
    return matrix, [f"doc_{i}" for i in range(len(documents))], all_structures


def save_matrix_and_labels(matrix: sparse.csr_matrix, row_labels: List[str], col_labels: List[str], 
                          output_path: Path, name: str):
    """
    Сохраняет матрицу и метки в файлы.
    
    Args:
        matrix: Разреженная матрица CSR
        row_labels: Метки строк
        col_labels: Метки столбцов
        output_path: Путь к директории вывода
        name: Имя файла (без расширения)
    """
    # Сохраняем матрицу
    sparse.save_npz(output_path / f"{name}.npz", matrix)
    
    # Сохраняем метки строк
    with open(output_path / f"{name}_rows.json", 'w', encoding='utf-8') as f:
        json.dump(row_labels, f, ensure_ascii=False, indent=2)
    
    # Сохраняем метки столбцов
    with open(output_path / f"{name}_cols.json", 'w', encoding='utf-8') as f:
        json.dump(col_labels, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Построение векторов слов и документов для русскоязычного корпуса')
    parser.add_argument('--input', required=True, help='Путь к входному каталогу с документами')
    parser.add_argument('--output', required=True, help='Путь к выходной директории')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Ошибка: входной каталог не существует: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Собираем все файлы из подкаталогов
    supported_extensions = {'.txt', '.docx', '.odt', '.html'}
    document_files = []
    
    for root, _, files in os.walk(input_path):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in supported_extensions:
                document_files.append(file_path)
    
    document_files.sort()  # Для воспроизводимости
    
    if not document_files:
        print(f"Предупреждение: не найдено файлов с поддерживаемыми расширениями в {input_path}", file=sys.stderr)
    
    # Читаем и обрабатываем документы
    documents = []
    document_paths = []
    
    for file_path in document_files:
        try:
            text_content = extract_text_from_file(file_path)
            words = tokenize_text(text_content)
            documents.append(words)
            # Сохраняем относительный путь
            rel_path = file_path.relative_to(input_path)
            document_paths.append(str(rel_path))
        except RuntimeError as e:
            print(f"Ошибка: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Если нет документов, создаем пустые структуры
    if not documents:
        documents = [[]]
        document_paths = ["empty"]
    
    # 1. Векторы слов на основе n-грамм
    print("Построение векторов слов на основе n-грамм...")
    word_ngram_matrix, word_ngram_rows, word_ngram_cols = build_word_ngram_vectors(documents)
    save_matrix_and_labels(word_ngram_matrix, word_ngram_rows, word_ngram_cols, output_path, "word_ngram")
    
    # 2. Векторы документов на основе TF-IDF
    print("Построение векторов документов на основе TF-IDF...")
    doc_tfidf_matrix, doc_tfidf_rows, doc_tfidf_cols = build_document_tfidf_vectors(documents)
    # Используем реальные пути документов вместо doc_0, doc_1, ...
    if len(document_paths) == len(doc_tfidf_rows):
        doc_tfidf_rows = document_paths
    save_matrix_and_labels(doc_tfidf_matrix, doc_tfidf_rows, doc_tfidf_cols, output_path, "doc_tfidf")
    
    # 3 и 4. Синтаксические векторы (требуют spaCy)
    if not NLP_AVAILABLE:
        print("Ошибка: spaCy не установлен. Установите spaCy и модель ru_core_news_sm для синтаксического анализа.", file=sys.stderr)
        sys.exit(1)
    
    try:
        nlp = spacy.load("ru_core_news_sm")
    except OSError:
        print("Ошибка: модель ru_core_news_sm не найдена. Установите её командой: python -m spacy download ru_core_news_sm", file=sys.stderr)
        sys.exit(1)
    
    # Проверяем наличие компонентов синтаксического разбора
    if "parser" not in nlp.pipe_names:
        print("Ошибка: модель ru_core_news_sm не содержит компонента синтаксического разбора (parser).", file=sys.stderr)
        sys.exit(1)
    
    print("Построение синтаксических векторов слов...")
    syntax_word_matrix, syntax_word_rows, syntax_word_cols = build_syntax_word_vectors(documents, nlp)
    save_matrix_and_labels(syntax_word_matrix, syntax_word_rows, syntax_word_cols, output_path, "syntax_word")
    
    print("Построение синтаксических векторов документов...")
    syntax_doc_matrix, syntax_doc_rows, syntax_doc_cols = build_syntax_document_vectors(documents, nlp)
    if len(document_paths) == len(syntax_doc_rows):
        syntax_doc_rows = document_paths
    save_matrix_and_labels(syntax_doc_matrix, syntax_doc_rows, syntax_doc_cols, output_path, "syntax_doc")
    
    # Сохраняем метаданные
    metadata = {
        "word_ngram": {
            "shape": [int(word_ngram_matrix.shape[0]), int(word_ngram_matrix.shape[1])],
            "description": "Векторы слов на основе частот символьных n-грамм длины 2 и 3 с граничными символами ^ и $"
        },
        "doc_tfidf": {
            "shape": [int(doc_tfidf_matrix.shape[0]), int(doc_tfidf_matrix.shape[1])],
            "description": "TF-IDF векторы документов по словным униграммам и биграммам, smooth_idf=True, norm='l2'"
        },
        "syntax_word": {
            "shape": [int(syntax_word_matrix.shape[0]), int(syntax_word_matrix.shape[1])],
            "description": "Векторы слов на основе синтаксических контекстов: OUT|dep|lemma_зависимого и IN|dep|lemma_головы"
        },
        "syntax_doc": {
            "shape": [int(syntax_doc_matrix.shape[0]), int(syntax_doc_matrix.shape[1])],
            "description": "Векторы документов на основе частот структурных троек POS_головы|dep|POS_зависимого"
        },
        "parameters": {
            "ngram_lengths": [2, 3],
            "tfidf_smooth_idf": True,
            "tfidf_norm": "l2",
            "ngram_boundary_symbols": ["^", "$"],
            "syntax_exclude_root": True,
            "syntax_exclude_punct": True,
            "syntax_alphabetic_only": True
        }
    }
    
    with open(output_path / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Обработка завершена. Результаты сохранены в {output_path}")


if __name__ == "__main__":
    main()
