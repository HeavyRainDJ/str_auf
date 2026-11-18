# Исправление ODT

Ошибка: `TableRow`/`TableCell` живут в `odf.table`, а не в `odf.text`. Явные импорты и правка `read_odt`:

```python
from odf.opendocument import load
from odf import teletype
from odf.table import Table, TableRow, TableCell
```

Ниже — полный `solution.py` с исправленным `read_odt`; всё остальное (CSR через `scipy.sparse.save_npz`, полный словарь, пустые входы, `read_errors.json`, направление контекстов, `.as_posix()`) сохранено.

```python
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
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


# =============================================================================
# Чтение документов
# =============================================================================

def read_txt(path: Path) -> str:
    """Чтение TXT с автоопределением кодировки."""
    raw = path.read_bytes()
    for enc in ("utf-8", "cp1251", "koi8-r"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def read_docx(path: Path) -> str:
    """Чтение DOCX: абзацы и таблицы в порядке следования body."""
    from docx import Document
    from docx.oxml.ns import qn
    from docx.table import Table as DocxTable

    doc = Document(str(path))
    parts = []

    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            para_text = "".join(
                node.text or "" for node in child.iter() if node.tag == qn("w:t")
            )
            if para_text.strip():
                parts.append(para_text.strip())
        elif child.tag == qn("w:tbl"):
            table = DocxTable(child, doc)
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))

    return "\n".join(parts)


def read_odt(path: Path) -> str:
    """Чтение ODT: абзацы + таблицы без дублирования ячеек."""
    from odf.opendocument import load
    from odf import teletype
    from odf.table import TableRow, TableCell

    doc = load(str(path))
    parts = []
    seen_cells = set()

    def walk(element):
        for child in element.childNodes:
            qname = getattr(child, "qname", None)
            if qname is None:
                continue
            tag = qname[1]
            if tag in ("p", "h"):
                text = teletype.extractText(child).strip()
                if text:
                    parts.append(text)
            elif tag == "table":
                for row in child.getElementsByType(TableRow):
                    cells = []
                    for cell in row.getElementsByType(TableCell):
                        cid = id(cell)
                        if cid in seen_cells:
                            continue
                        seen_cells.add(cid)
                        text = teletype.extractText(cell).strip()
                        if text:
                            cells.append(text)
                    if cells:
                        parts.append(" | ".join(cells))
            else:
                walk(child)

    walk(doc.body)
    return "\n".join(parts)


def read_html(path: Path) -> str:
    """Чтение HTML: только видимый текст без script/style."""
    from bs4 import BeautifulSoup

    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")
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
    """Чтение документа по расширению. Исключения пробрасываются вверх."""
    ext = path.suffix.lower()
    reader = READERS.get(ext)
    if reader is None:
        raise ValueError(f"Неподдерживаемое расширение: {ext}")
    return reader(path)


# =============================================================================
# Токенизация
# =============================================================================

TOKEN_RE = re.compile(r"[а-яёА-ЯЁa-zA-Z]+", re.UNICODE)


def extract_words(text: str) -> list:
    """Lowercase последовательности русских/латинских букв, ё сохраняется."""
    return [m.group().lower() for m in TOKEN_RE.finditer(text)]


# =============================================================================
# Символьные n-граммы для слов
# =============================================================================

def char_ngrams(word: str, n: int) -> list:
    """N-граммы с граничными символами ^ и $."""
    padded = "^" + word + "$"
    return [padded[i:i + n] for i in range(len(padded) - n + 1)]


def build_word_char_ngram_matrix(vocabulary: list, ngram_lengths=(2, 3)):
    """
    Матрица: строки = все уникальные слова корпуса (vocabulary),
    столбцы = все n-граммы длины n из ngram_lengths.
    """
    if not vocabulary:
        return csr_matrix((0, 0), dtype=np.float64), [], []

    all_ngrams = set()
    word_ngrams = {}
    for word in vocabulary:
        counter = Counter()
        for n in ngram_lengths:
            counter.update(char_ngrams(word, n))
        word_ngrams[word] = counter
        all_ngrams.update(counter.keys())

    if not all_ngrams:
        return csr_matrix((len(vocabulary), 0), dtype=np.float64), list(vocabulary), []

    feature_labels = sorted(all_ngrams)
    feature_index = {f: i for i, f in enumerate(feature_labels)}

    rows, cols, vals = [], [], []
    for row_idx, word in enumerate(vocabulary):
        for ngram, count in word_ngrams[word].items():
            rows.append(row_idx)
            cols.append(feature_index[ngram])
            vals.append(count)

    matrix = csr_matrix(
        (vals, (rows, cols)),
        shape=(len(vocabulary), len(feature_labels)),
        dtype=np.float64,
    )
    return matrix, list(vocabulary), feature_labels


# =============================================================================
# TF-IDF для документов
# =============================================================================

def build_document_tfidf(documents: list, ngram_range=(1, 2)):
    """TF-IDF по словным униграммам и биграммам, smooth_idf=True, norm='l2'."""
    tokenized = [" ".join(extract_words(text)) for text in documents]

    if not tokenized:
        return csr_matrix((0, 0), dtype=np.float64), []

    if all(t.strip() == "" for t in tokenized):
        return csr_matrix((len(tokenized), 0), dtype=np.float64), []

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        smooth_idf=True,
        norm="l2",
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
    )
    matrix = vectorizer.fit_transform(tokenized)
    feature_names = vectorizer.get_feature_names_out().tolist()
    return matrix.astype(np.float64), feature_names


# =============================================================================
# Синтаксические векторы через spaCy
# =============================================================================

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
    """Токен состоит ТОЛЬКО из русских/латинских букв."""
    return bool(TOKEN_RE.fullmatch(token.text))


def build_syntax_word_matrix(documents: list, vocabulary: list):
    """
    Вектор слова: частоты контекстов OUT|dep|lemma_зависимого
    и IN|dep|lemma_головы.

    Строки — ВСЕ слова из vocabulary (в том числе с нулевыми строками).
    """
    word_contexts = defaultdict(Counter)

    if vocabulary:
        nlp = get_nlp()
        for text in documents:
            if not text or not text.strip():
                continue
            doc = nlp(text)
            for token in doc:
                if token.dep_ == "ROOT" or token.is_punct:
                    continue
                head = token.head
                if head is token:
                    continue
                if not is_letter_token(token) or not is_letter_token(head):
                    continue

                head_key = head.text.lower()
                child_key = token.text.lower()
                if not head_key or not child_key:
                    continue

                dep = token.dep_
                word_contexts[head_key][f"OUT|{dep}|{token.lemma_.lower()}"] += 1
                word_contexts[child_key][f"IN|{dep}|{head.lemma_.lower()}"] += 1

    word_labels = list(vocabulary)

    if not word_labels:
        return csr_matrix((0, 0), dtype=np.float64), [], []

    all_contexts = set()
    for ctx_counter in word_contexts.values():
        all_contexts.update(ctx_counter.keys())

    if not all_contexts:
        return csr_matrix((len(word_labels), 0), dtype=np.float64), word_labels, []

    feature_labels = sorted(all_contexts)
    feature_index = {f: i for i, f in enumerate(feature_labels)}

    rows, cols, vals = [], [], []
    for row_idx, word in enumerate(word_labels):
        counter = word_contexts.get(word)
        if not counter:
            continue
        for ctx, count in counter.items():
            rows.append(row_idx)
            cols.append(feature_index[ctx])
            vals.append(count)

    matrix = csr_matrix(
        (vals, (rows, cols)),
        shape=(len(word_labels), len(feature_labels)),
        dtype=np.float64,
    )
    return matrix, word_labels, feature_labels


def build_syntax_document_matrix(documents: list):
    """
    Вектор документа: частоты структурных троек
    POS_головы|dep|POS_зависимого.
    """
    if not documents:
        return csr_matrix((0, 0), dtype=np.float64), []

    doc_counters = []
    has_any_text = any(text and text.strip() for text in documents)

    if has_any_text:
        nlp = get_nlp()
        for text in documents:
            counter = Counter()
            if text and text.strip():
                doc = nlp(text)
                for token in doc:
                    if token.dep_ == "ROOT" or token.is_punct:
                        continue
                    head = token.head
                    if head is token:
                        continue
                    if not is_letter_token(token) or not is_letter_token(head):
                        continue
                    ctx = f"{head.pos_}|{token.dep_}|{token.pos_}"
                    counter[ctx] += 1
            doc_counters.append(counter)
    else:
        doc_counters = [Counter() for _ in documents]

    all_contexts = set()
    for c in doc_counters:
        all_contexts.update(c.keys())

    if not all_contexts:
        return csr_matrix((len(documents), 0), dtype=np.float64), []

    feature_labels = sorted(all_contexts)
    feature_index = {f: i for i, f in enumerate(feature_labels)}

    rows, cols, vals = [], [], []
    for row_idx, counter in enumerate(doc_counters):
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

def collect_documents(input_dir: Path):
    """
    Возвращает (texts, rel_paths, errors).
    Ошибки чтения НЕ скрываются — попадают в errors и прерывают работу.
    """
    texts = []
    rel_paths = []
    errors = []

    all_files = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in READERS
    )

    for path in all_files:
        rel_str = path.relative_to(input_dir).as_posix()
        try:
            text = read_document(path)
        except Exception as e:
            errors.append({
                "path": rel_str,
                "error": f"{type(e).__name__}: {e}",
            })
            continue
        texts.append(text)
        rel_paths.append(rel_str)

    return texts, rel_paths, errors


# =============================================================================
# Сохранение
# =============================================================================

def save_csr(matrix: csr_matrix, path: Path):
    """Сохранение CSR-матрицы в формат scipy.sparse (.npz)."""
    save_npz(str(path), matrix)


def save_json_list(items, path: Path):
    """UTF-8 JSON со списком."""
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

    print("Сбор документов...", file=sys.stderr)
    texts, doc_paths, errors = collect_documents(input_dir)
    print(f"Найдено документов: {len(texts)}, ошибок чтения: {len(errors)}",
          file=sys.stderr)

    if errors:
        save_json_list(errors, output_dir / "read_errors.json")
        for err in errors:
            print(f"ОШИБКА чтения {err['path']}: {err['error']}", file=sys.stderr)
        print("Обнаружены ошибки чтения. Выход.", file=sys.stderr)
        sys.exit(2)

    all_words = []
    for text in texts:
        all_words.extend(extract_words(text))
    vocabulary = sorted(set(all_words))

    print("Построение символьных n-грамм слов...", file=sys.stderr)
    word_char_matrix, word_char_rows, word_char_feats = \
        build_word_char_ngram_matrix(vocabulary)

    print("Построение TF-IDF документов...", file=sys.stderr)
    doc_tfidf_matrix, doc_tfidf_feats = build_document_tfidf(texts)

    print("Построение синтаксических векторов слов...", file=sys.stderr)
    syntax_word_matrix, syntax_word_rows, syntax_word_feats = \
        build_syntax_word_matrix(texts, vocabulary)

    print("Построение синтаксических векторов документов...", file=sys.stderr)
    syntax_doc_matrix, syntax_doc_feats = build_syntax_document_matrix(texts)

    print("Сохранение...", file=sys.stderr)

    save_csr(word_char_matrix, output_dir / "word_char_ngrams.npz")
    save_json_list(word_char_rows, output_dir / "word_char_ngrams_rows.json")
    save_json_list(word_char_feats, output_dir / "word_char_ngrams_features.json")

    save_csr(doc_tfidf_matrix, output_dir / "doc_tfidf.npz")
    save_json_list(doc_paths, output_dir / "doc_tfidf_rows.json")
    save_json_list(doc_tfidf_feats, output_dir / "doc_tfidf_features.json")

    save_csr(syntax_word_matrix, output_dir / "word_syntax.npz")
    save_json_list(syntax_word_rows, output_dir / "word_syntax_rows.json")
    save_json_list(syntax_word_feats, output_dir / "word_syntax_features.json")

    save_csr(syntax_doc_matrix, output_dir / "doc_syntax.npz")
    save_json_list(doc_paths, output_dir / "doc_syntax_rows.json")
    save_json_list(syntax_doc_feats, output_dir / "doc_syntax_features.json")

    metadata = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "n_documents": len(texts),
        "n_word_types": len(vocabulary),
        "n_features_char_ngrams": int(word_char_matrix.shape[1]),
        "n_features_doc_tfidf": int(doc_tfidf_matrix.shape[1]),
        "n_features_syntax_word": int(syntax_word_matrix.shape[1]),
        "n_features_syntax_doc": int(syntax_doc_matrix.shape[1]),
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
```

## Что именно поменялось

В `read_odt`:

- было: `__import__("odf.text", fromlist=["TableRow"]).TableRow` (падало, т.к. `odf.text` не содержит `TableRow`);
- стало: `from odf.table import TableRow, TableCell` и `child.getElementsByType(TableRow)` / `row.getElementsByType(TableCell)`.

Логика обхода и `seen_cells` (антидубликат) сохранены. `teletype.extractText` используется и для параграфов, и для ячеек. Всё остальное — без изменений.

## Команды для прогонов

```bash
# форматы
python solution.py --input ./test_formats --output ./out_formats

# полный корпус из 50 файлов
python solution.py --input ./data --output ./out_full
```

## Статус проверки

Я **не запускал** эту версию на вашей машине и не могу подтвердить PASS без реального прогона. Исправление точечное и устраняет конкретный `AttributeError` из вашего трейсбека: классы `TableRow`/`TableCell` действительно определены в `odf.table`, а не в `odf.text`. Если после запуска `formats` и `full` появятся новые ошибки — пришлите трейсбек, поправлю точечно.