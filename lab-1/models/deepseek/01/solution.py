#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторная работа: векторизация слов и документов русского корпуса.

(1) Слова: символьные n-граммы (n=2,3) с граничными символами ^ и $.
(2) Документы: TF-IDF по словным униграммам и биграммам.
(3) Синтаксис: dependency parsing через spaCy ru_core_news_sm.
    - Вектор слова: контексты OUT|dep|lemma_зависимого и IN|dep|lemma_головы.
    - Вектор документа: POS_головы|dep|POS_зависимого.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer


# =============================================================================
# Чтение документов
# =============================================================================

def read_txt(path: Path) -> str:
    """Чтение TXT с автоопределением кодировки."""
    for enc in ("utf-8", "cp1251", "koi8-r"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    # fallback: utf-8 с заменой ошибок
    return path.read_text(encoding="utf-8", errors="replace")


def read_docx(path: Path) -> str:
    """Чтение DOCX: абзацы + таблицы (в порядке следования)."""
    from docx import Document
    from docx.oxml.ns import qn

    doc = Document(str(path))
    parts = []

    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            # это параграф
            para_text = "".join(node.text or "" for node in child.iter() if node.tag == qn("w:t"))
            if para_text.strip():
                parts.append(para_text.strip())
        elif child.tag == qn("w:tbl"):
            # это таблица
            from docx.table import Table
            table = Table(child, doc)
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))

    return "\n".join(parts)


def read_odt(path: Path) -> str:
    """Чтение ODT: абзацы + таблицы без дублирования."""
    from odf.opendocument import load
    from odf.text import P, H, Table, TableRow, TableCell
    from odf import teletype

    doc = load(str(path))
    parts = []
    seen_cells = set()

    # Обходим body в порядке следования
    body = doc.body

    def walk(element):
        for child in element.childNodes:
            if child.qname[1] in ("p", "h"):
                text = teletype.extractText(child).strip()
                if text:
                    parts.append(text)
            elif child.qname[1] == "table":
                for row in child.getElementsByType(TableRow):
                    cells = []
                    for cell in row.getElementsByType(TableCell):
                        cell_id = id(cell)
                        if cell_id in seen_cells:
                            continue
                        seen_cells.add(cell_id)
                        text = teletype.extractText(cell).strip()
                        if text:
                            cells.append(text)
                    if cells:
                        parts.append(" | ".join(cells))
            else:
                walk(child)

    walk(body)
    return "\n".join(parts)


def read_html(path: Path) -> str:
    """Чтение HTML: только видимый текст без script/style."""
    from bs4 import BeautifulSoup

    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")

    # Удаляем script и style
    for tag in soup(["script", "style"]):
        tag.decompose()

    return soup.get_text(separator="\n", strip=True)


READERS = {
    ".txt": read_txt,
    ".docx": read_docx,
    ".odt": read_odt,
    ".html": read_html,
    ".htm": read_html,
}


def read_document(path: Path) -> str:
    """Чтение документа по расширению. Ошибки явно видны."""
    ext = path.suffix.lower()
    reader = READERS.get(ext)
    if reader is None:
        raise ValueError(f"Неподдерживаемое расширение: {ext}")
    return reader(path)


# =============================================================================
# Токенизация
# =============================================================================

TOKEN_RE = re.compile(r"[а-яёА-ЯЁa-zA-Z]+", re.UNICODE)


def extract_words(text: str) -> list[str]:
    """
    Извлечение слов: lowercase последовательности русских/латинских букв.
    Ё сохраняется.
    """
    return [m.group().lower() for m in TOKEN_RE.finditer(text)]


# =============================================================================
# Символьные n-граммы для слов
# =============================================================================

def char_ngrams(word: str, n: int) -> list[str]:
    """
    N-граммы с граничными символами ^ и $.
    Для n=2: ^a, ab, b$, ...
    Для n=3: ^ab, abc, bc$, ...
    """
    padded = "^" + word + "$"
    return [padded[i:i + n] for i in range(len(padded) - n + 1)]


def build_word_char_ngram_matrix(
    documents: list[str],
    ngram_lengths: tuple[int, ...] = (2, 3),
) -> tuple[csr_matrix, list[str], list[str]]:
    """
    Строит матрицу: строки = уникальные слова, столбцы = n-граммы.
    Возвращает (matrix, word_labels, feature_labels).
    """
    # Собираем все слова корпуса
    all_words = []
    for text in documents:
        all_words.extend(extract_words(text))

    unique_words = sorted(set(all_words))
    if not unique_words:
        return csr_matrix((0, 0)), [], []

    # Собираем все n-граммы
    all_ngrams = set()
    word_ngram_counter = {}
    for word in unique_words:
        ngrams = []
        for n in ngram_lengths:
            ngrams.extend(char_ngrams(word, n))
        word_ngram_counter[word] = Counter(ngrams)
        all_ngrams.update(ngrams)

    feature_labels = sorted(all_ngrams)
    feature_index = {f: i for i, f in enumerate(feature_labels)}

    # Строим разреженную матрицу
    rows, cols, vals = [], [], []
    for row_idx, word in enumerate(unique_words):
        for ngram, count in word_ngram_counter[word].items():
            rows.append(row_idx)
            cols.append(feature_index[ngram])
            vals.append(count)

    matrix = csr_matrix(
        (vals, (rows, cols)),
        shape=(len(unique_words), len(feature_labels)),
        dtype=np.float64,
    )

    return matrix, unique_words, feature_labels


# =============================================================================
# TF-IDF для документов
# =============================================================================

def build_document_tfidf(
    documents: list[str],
    ngram_range: tuple[int, int] = (1, 2),
) -> tuple[csr_matrix, list[str]]:
    """
    TF-IDF по словным униграммам и биграммам.
    smooth_idf=True, norm='l2'.
    """
    # Токенизируем каждый документ
    tokenized = []
    for text in documents:
        words = extract_words(text)
        tokenized.append(" ".join(words))

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        smooth_idf=True,
        norm="l2",
        lowercase=False,  # уже lowercase
        token_pattern=r"(?u)\b\w+\b",
    )

    if not tokenized or all(t.strip() == "" for t in tokenized):
        return csr_matrix((len(tokenized), 0)), []

    matrix = vectorizer.fit_transform(tokenized)
    feature_names = vectorizer.get_feature_names_out().tolist()

    return matrix, feature_names


# =============================================================================
# Синтаксические векторы через spaCy
# =============================================================================

# Кэш модели
_NLP = None


def get_nlp():
    """Загрузка модели spaCy ru_core_news_sm. Ошибка явная."""
    global _NLP
    if _NLP is None:
        import spacy
        try:
            _NLP = spacy.load("ru_core_news_sm")
        except OSError as e:
            raise RuntimeError(
                "Модель ru_core_news_sm не найдена. "
                "Установите: python -m spacy download ru_core_news_sm"
            ) from e
    return _NLP


def is_letter_token(token) -> bool:
    """Токен состоит только из букв (хотя бы одной)."""
    return bool(TOKEN_RE.fullmatch(token.text))


def build_syntax_word_matrix(
    documents: list[str],
) -> tuple[csr_matrix, list[str], list[str]]:
    """
    Вектор слова: частоты контекстов OUT|dep|lemma_зависимого
    и IN|dep|lemma_головы.

    Ключ слова: lowercase surface form.
    Корневые и пунктуационные связи не включаются.
    Контексты только для пар, где оба токена состоят из букв.
    """
    nlp = get_nlp()

    word_contexts = defaultdict(Counter)  # word -> Counter(context)

    for text in documents:
        if not text.strip():
            continue
        doc = nlp(text)
        for token in doc:
            # Пропускаем корневые и пунктуационные связи
            if token.dep_ == "ROOT" or token.is_punct:
                continue

            head = token.head
            # Проверяем, что оба токена буквенные
            if not is_letter_token(token) or not is_letter_token(head):
                continue

            word_key = token.text.lower()
            if not word_key:
                continue

            # OUT: слово -> зависимый
            # Контекст: OUT|dep|lemma_зависимого
            context_out = f"OUT|{token.dep_}|{token.lemma_.lower()}"
            word_contexts[word_key][context_out] += 1

            # IN: слово -> голова
            # Контекст: IN|dep|lemma_головы
            context_in = f"IN|{token.dep_}|{head.lemma_.lower()}"
            word_contexts[word_key][context_in] += 1

    if not word_contexts:
        return csr_matrix((0, 0)), [], []

    word_labels = sorted(word_contexts.keys())
    all_contexts = set()
    for ctx_counter in word_contexts.values():
        all_contexts.update(ctx_counter.keys())

    if not all_contexts:
        return csr_matrix((len(word_labels), 0)), word_labels, []

    feature_labels = sorted(all_contexts)
    feature_index = {f: i for i, f in enumerate(feature_labels)}

    rows, cols, vals = [], [], []
    for row_idx, word in enumerate(word_labels):
        for ctx, count in word_contexts[word].items():
            rows.append(row_idx)
            cols.append(feature_index[ctx])
            vals.append(count)

    matrix = csr_matrix(
        (vals, (rows, cols)),
        shape=(len(word_labels), len(feature_labels)),
        dtype=np.float64,
    )

    return matrix, word_labels, feature_labels


def build_syntax_document_matrix(
    documents: list[str],
) -> tuple[csr_matrix, list[str]]:
    """
    Вектор документа: частоты структурных троек
    POS_головы|dep|POS_зависимого.
    """
    nlp = get_nlp()

    doc_contexts = []

    for text in documents:
        counter = Counter()
        if text.strip():
            doc = nlp(text)
            for token in doc:
                if token.dep_ == "ROOT" or token.is_punct:
                    continue
                head = token.head
                if not is_letter_token(token) or not is_letter_token(head):
                    continue
                context = f"{head.pos_}|{token.dep_}|{token.pos_}"
                counter[context] += 1
        doc_contexts.append(counter)

    # Собираем общее пространство признаков
    all_contexts = set()
    for c in doc_contexts:
        all_contexts.update(c.keys())

    feature_labels = sorted(all_contexts)
    feature_index = {f: i for i, f in enumerate(feature_labels)}

    rows, cols, vals = [], [], []
    for row_idx, counter in enumerate(doc_contexts):
        for ctx, count in counter.items():
            rows.append(row_idx)
            cols.append(feature_index[ctx])
            vals.append(count)

    matrix = csr_matrix(
        (vals, (rows, cols)),
        shape=(len(documents), len(feature_labels)),
        dtype=np.float64,
    )

    return matrix, feature_labels


# =============================================================================
# Сбор документов
# =============================================================================

def collect_documents(input_dir: Path) -> tuple[list[str], list[str]]:
    """
    Рекурсивный сбор документов.
    Возвращает (тексты, относительные пути).
    Пустые документы сохраняются как пустые строки.
    """
    texts = []
    rel_paths = []

    all_files = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in READERS
    )

    for path in all_files:
        rel = path.relative_to(input_dir)
        rel_str = str(rel).replace(os.sep, "/")

        try:
            text = read_document(path)
        except Exception as e:
            print(f"ОШИБКА чтения {path}: {e}", file=sys.stderr)
            text = ""

        texts.append(text)
        rel_paths.append(rel_str)

    return texts, rel_paths


# =============================================================================
# Сохранение
# =============================================================================

def save_matrix_npz(matrix: csr_matrix, path: Path) -> None:
    """Сохранение CSR-матрицы в .npz."""
    np.savez_compressed(
        path,
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        shape=matrix.shape,
    )


def save_json_list(items: list, path: Path) -> None:
    """Сохранение списка в UTF-8 JSON."""
    path.write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# =============================================================================
# main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Векторизация слов и документов русского корпуса."
    )
    parser.add_argument("--input", required=True, help="Каталог с документами")
    parser.add_argument("--output", required=True, help="Каталог для результатов")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.is_dir():
        print(f"Ошибка: {input_dir} не является каталогом", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Сбор документов
    print("Сбор документов...", file=sys.stderr)
    texts, doc_paths = collect_documents(input_dir)
    print(f"Найдено документов: {len(texts)}", file=sys.stderr)

    # 1. Символьные n-граммы слов
    print("Построение символьных n-грамм слов...", file=sys.stderr)
    word_char_matrix, word_char_rows, word_char_feats = build_word_char_ngram_matrix(texts)

    # 2. TF-IDF документов
    print("Построение TF-IDF документов...", file=sys.stderr)
    doc_tfidf_matrix, doc_tfidf_feats = build_document_tfidf(texts)

    # 3. Синтаксические векторы слов
    print("Построение синтаксических векторов слов...", file=sys.stderr)
    syntax_word_matrix, syntax_word_rows, syntax_word_feats = build_syntax_word_matrix(texts)

    # 4. Синтаксические векторы документов
    print("Построение синтаксических векторов документов...", file=sys.stderr)
    syntax_doc_matrix, syntax_doc_feats = build_syntax_document_matrix(texts)

    # Сохранение
    print("Сохранение...", file=sys.stderr)

    # 1. Слова: символьные n-граммы
    save_matrix_npz(word_char_matrix, output_dir / "word_char_ngrams.npz")
    save_json_list(word_char_rows, output_dir / "word_char_ngrams_rows.json")
    save_json_list(word_char_feats, output_dir / "word_char_ngrams_features.json")

    # 2. Документы: TF-IDF
    save_matrix_npz(doc_tfidf_matrix, output_dir / "doc_tfidf.npz")
    save_json_list(doc_paths, output_dir / "doc_tfidf_rows.json")
    save_json_list(doc_tfidf_feats, output_dir / "doc_tfidf_features.json")

    # 3. Слова: синтаксис
    save_matrix_npz(syntax_word_matrix, output_dir / "word_syntax.npz")
    save_json_list(syntax_word_rows, output_dir / "word_syntax_rows.json")
    save_json_list(syntax_word_feats, output_dir / "word_syntax_features.json")

    # 4. Документы: синтаксис
    save_matrix_npz(syntax_doc_matrix, output_dir / "doc_syntax.npz")
    save_json_list(doc_paths, output_dir / "doc_syntax_rows.json")
    save_json_list(syntax_doc_feats, output_dir / "doc_syntax_features.json")

    # metadata.json
    metadata = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "n_documents": len(texts),
        "n_word_types_char_ngrams": word_char_matrix.shape[0],
        "n_features_char_ngrams": word_char_matrix.shape[1],
        "n_features_doc_tfidf": doc_tfidf_matrix.shape[1],
        "n_word_types_syntax": syntax_word_matrix.shape[0],
        "n_features_syntax_word": syntax_word_matrix.shape[1],
        "n_features_syntax_doc": syntax_doc_matrix.shape[1],
        "char_ngram_lengths": [2, 3],
        "tfidf_ngram_range": [1, 2],
        "tfidf_smooth_idf": True,
        "tfidf_norm": "l2",
        "spacy_model": "ru_core_news_sm",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Готово.", file=sys.stderr)


if __name__ == "__main__":
    main()