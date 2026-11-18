#!/usr/bin/env python3
"""
solution.py — лабораторная работа: векторизация слов и документов.

Четыре разреженные CSR-матрицы:
  1. word_char.npz   — слова × символьные n-граммы (2,3) с границами ^ и $
  2. doc_tfidf.npz   — документы × TF-IDF (униграммы + биграммы, L2, smooth_idf)
  3. word_syntax.npz — слова × синтаксические контексты (spaCy dependency parsing)
  4. doc_syntax.npz  — документы × структурные тройки POS|dep|POS
"""

import argparse
import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, save_npz

# ─── Опциональные зависимости (явный ImportError, если не установлены) ───

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    import docx
except ImportError:
    docx = None

try:
    import spacy
except ImportError:
    spacy = None

# ─── Константы ───

WORD_RE = re.compile(r'[а-яёА-ЯЁa-zA-Z]+')
NGRAM_LENGTHS = (2, 3)
BOUNDARY_START = '^'
BOUNDARY_END = '$'
SPACY_MODEL = 'ru_core_news_sm'
ODT_TEXT_NS = '{urn:oasis:names:tc:opendocument:xmlns:text:1.0}'
SKIP_DEPS = {'ROOT', 'punct', '', None}

READERS = {}  # заполним после определения функций-читалок

# ─── Извлечение слов ───

def extract_words(text: str) -> list[str]:
    """Lowercase-слова из русских/латинских букв; ё сохраняется."""
    return [w.lower() for w in WORD_RE.findall(text)]


# ─── Чтение файлов разных форматов ───

def read_txt(path: Path) -> str:
    try:
        with open(path, 'r', encoding='utf-8-sig') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, 'r', encoding='cp1251') as f:
            return f.read()


def read_docx(path: Path) -> str:
    if docx is None:
        raise ImportError("python-docx не установлен (pip install python-docx)")
    document = docx.Document(str(path))
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                parts.append(cell.text)
    return '\n'.join(parts)


def read_odt(path: Path) -> str:
    with zipfile.ZipFile(str(path), 'r') as zf:
        if 'content.xml' not in zf.namelist():
            raise ValueError("ODT не содержит content.xml")
        with zf.open('content.xml') as f:
            tree = ET.parse(f)
    root = tree.getroot()
    parts = []
    for elem in root.iter():
        if elem.tag in (ODT_TEXT_NS + 'p', ODT_TEXT_NS + 'h'):
            text = ''.join(elem.itertext())
            if text.strip():
                parts.append(text)
    return '\n'.join(parts)


def read_html(path: Path) -> str:
    if BeautifulSoup is None:
        raise ImportError("beautifulsoup4 не установлен (pip install beautifulsoup4)")
    with open(path, 'rb') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    for tag in soup(['script', 'style']):
        tag.decompose()
    return soup.get_text(separator='\n')


READERS = {
    '.txt': read_txt,
    '.docx': read_docx,
    '.odt': read_odt,
    '.html': read_html,
    '.htm': read_html,
}


def read_file(path: Path) -> str:
    ext = path.suffix.lower()
    reader = READERS.get(ext)
    if reader is None:
        raise ValueError(f"Неподдерживаемое расширение: {ext}")
    return reader(path)


# ─── Сбор файлов ───

def collect_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    files = [
        f for f in input_dir.rglob('*')
        if f.is_file() and f.suffix.lower() in READERS
    ]
    files.sort(key=lambda p: str(p.relative_to(input_dir)).replace('\\', '/'))
    return files


# ─── 1. Матрица «слово × символьные n-граммы» ───

def char_ngrams(word: str) -> list[str]:
    """n-граммы длин 2 и 3 с границами ^ и $."""
    w = BOUNDARY_START + word + BOUNDARY_END
    result = []
    for n in NGRAM_LENGTHS:
        if len(w) >= n:
            for i in range(len(w) - n + 1):
                result.append(w[i:i + n])
    return result


def build_word_char_matrix(unique_words: list[str]):
    if not unique_words:
        return csr_matrix((0, 0), dtype=np.float64), [], []

    feature_set: set[str] = set()
    word_counts: list[dict[str, int]] = []

    for word in unique_words:
        ngrams = char_ngrams(word)
        counts: dict[str, int] = defaultdict(int)
        for ng in ngrams:
            counts[ng] += 1
            feature_set.add(ng)
        word_counts.append(dict(counts))

    features = sorted(feature_set)
    feat_idx = {f: i for i, f in enumerate(features)}

    rows, cols, data = [], [], []
    for i, counts in enumerate(word_counts):
        for ng, cnt in counts.items():
            rows.append(i)
            cols.append(feat_idx[ng])
            data.append(float(cnt))

    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(unique_words), len(features)),
        dtype=np.float64,
    )
    return matrix, unique_words, features


# ─── 2. Матрица «документ × TF-IDF» ───

def build_doc_tfidf_matrix(doc_texts: list[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer

    n_docs = len(doc_texts)
    if n_docs == 0:
        return csr_matrix((0, 0), dtype=np.float64), []

    has_words = any(extract_words(t) for t in doc_texts)
    if not has_words:
        return csr_matrix((n_docs, 0), dtype=np.float64), []

    vectorizer = TfidfVectorizer(
        tokenizer=extract_words,
        ngram_range=(1, 2),
        smooth_idf=True,
        norm='l2',
        token_pattern=None,
    )
    matrix = vectorizer.fit_transform(doc_texts)

    orig = vectorizer.get_feature_names_out().tolist()
    order = sorted(range(len(orig)), key=lambda i: orig[i])
    matrix = matrix[:, order].tocsr()
    features = sorted(orig)

    return matrix, features


# ─── 3 и 4. Синтаксические матрицы ───

def build_syntax_matrices(
    doc_texts: list[str],
    doc_paths: list[str],
    unique_words: list[str],
    nlp,
):
    """
    unique_words — отсортированный список всех уникальных слов корпуса,
    сформированный из extract_words по всем текстам ДО парсинга.
    word_syntax будет иметь строку для каждого слова из unique_words,
    даже если у слова нет ни одного синтаксического контекста (нулевая строка).
    """

    word_idx = {w: i for i, w in enumerate(unique_words)}
    word_contexts: list[dict[str, int]] = [defaultdict(int) for _ in unique_words]
    doc_triples_list: list[dict[str, int]] = []

    for idx, text in enumerate(doc_texts):
        doc_triples: dict[str, int] = defaultdict(int)
        if text.strip():
            try:
                doc = nlp(text)
            except Exception as e:
                raise RuntimeError(
                    f"Ошибка синтаксического разбора документа "
                    f"'{doc_paths[idx]}': {e}"
                )

            for token in doc:
                dep = token.dep_
                if dep in SKIP_DEPS:
                    continue
                head = token.head

                # Обе вершины проверяем WORD_RE.fullmatch — только рус/лат буквы
                if not WORD_RE.fullmatch(token.text):
                    continue
                if not WORD_RE.fullmatch(head.text):
                    continue

                # Контекст для зависимого: IN|dep|lemma_головы (lower)
                w_key = token.text.lower()
                wi = word_idx.get(w_key)
                if wi is not None:
                    word_contexts[wi][f"IN|{dep}|{head.lemma_.lower()}"] += 1

                # Контекст для головы: OUT|dep|lemma_зависимого (lower)
                h_key = head.text.lower()
                hi = word_idx.get(h_key)
                if hi is not None:
                    word_contexts[hi][f"OUT|{dep}|{token.lemma_.lower()}"] += 1

                # Тройка для документа: POS_головы|dep|POS_зависимого
                doc_triples[f"{head.pos_}|{dep}|{token.pos_}"] += 1

        doc_triples_list.append(doc_triples)

    # ── Матрица «слово × синтаксический контекст» ──

    all_ctx: set[str] = set()
    for wc in word_contexts:
        all_ctx.update(wc.keys())
    contexts = sorted(all_ctx)

    if unique_words and contexts:
        ctx_idx = {c: i for i, c in enumerate(contexts)}
        rows, cols, data = [], [], []
        for i, wc in enumerate(word_contexts):
            for c, cnt in wc.items():
                rows.append(i)
                cols.append(ctx_idx[c])
                data.append(float(cnt))
        word_mat = csr_matrix(
            (data, (rows, cols)),
            shape=(len(unique_words), len(contexts)),
            dtype=np.float64,
        )
    else:
        word_mat = csr_matrix(
            (len(unique_words), len(contexts)),
            dtype=np.float64,
        )

    # ── Матрица «документ × синтаксическая тройка» ──

    all_triples: set[str] = set()
    for dt in doc_triples_list:
        all_triples.update(dt.keys())
    triples = sorted(all_triples)

    if doc_triples_list and triples:
        tr_idx = {t: i for i, t in enumerate(triples)}
        rows, cols, data = [], [], []
        for i, dt in enumerate(doc_triples_list):
            for t, cnt in dt.items():
                rows.append(i)
                cols.append(tr_idx[t])
                data.append(float(cnt))
        doc_mat = csr_matrix(
            (data, (rows, cols)),
            shape=(len(doc_triples_list), len(triples)),
            dtype=np.float64,
        )
    else:
        doc_mat = csr_matrix(
            (len(doc_triples_list), len(triples)),
            dtype=np.float64,
        )

    return word_mat, unique_words, contexts, doc_mat, triples


# ─── Утилиты сохранения ───

def save_json(path: Path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─── Главная функция ───

def main():
    parser = argparse.ArgumentParser(
        description='Лабораторная работа: векторизация слов и документов'
    )
    parser.add_argument('--input', required=True, help='Каталог с данными')
    parser.add_argument('--output', required=True, help='Каталог для результатов')
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Сбор файлов ──
    files = collect_files(input_dir)
    if not files:
        print("ВНИМАНИЕ: файлы не найдены во входном каталоге", file=sys.stderr)

    # ── Чтение файлов ──
    doc_paths: list[str] = []
    doc_texts: list[str] = []

    for fpath in files:
        rel = str(fpath.relative_to(input_dir)).replace('\\', '/')
        try:
            text = read_file(fpath)
        except Exception as e:
            raise RuntimeError(f"Ошибка чтения файла '{rel}': {e}")
        doc_paths.append(rel)
        doc_texts.append(text)

    # ── Все слова корпуса (формируем до парсинга, используем для обеих матриц) ──
    all_words: list[str] = []
    for t in doc_texts:
        all_words.extend(extract_words(t))
    unique_words = sorted(set(all_words))
    unique_word_count = len(unique_words)

    # ── 1. Символьные n-граммы ──
    print("1/4: символьные n-граммы слов...", file=sys.stderr)
    wc_mat, wc_rows, wc_feats = build_word_char_matrix(unique_words)

    # ── 2. TF-IDF ──
    print("2/4: TF-IDF документов...", file=sys.stderr)
    dt_mat, dt_feats = build_doc_tfidf_matrix(doc_texts)

    # ── 3 и 4. Синтаксис ──
    print("3/4: загрузка модели spaCy...", file=sys.stderr)
    if spacy is None:
        raise ImportError(
            "spacy не установлен. Выполните:\n"
            "  pip install spacy\n"
            f"  python -m spacy download {SPACY_MODEL}"
        )
    try:
        nlp = spacy.load(SPACY_MODEL)
    except Exception as e:
        raise ImportError(
            f"Не удалось загрузить модель '{SPACY_MODEL}'.\n"
            f"Выполните: python -m spacy download {SPACY_MODEL}\n"
            f"Ошибка: {e}"
        )

    print("4/4: синтаксический разбор...", file=sys.stderr)
    ws_mat, ws_rows, ws_feats, ds_mat, ds_feats = build_syntax_matrices(
        doc_texts, doc_paths, unique_words, nlp
    )

    # ── Сохранение матриц ──
    save_npz(str(output_dir / 'word_char.npz'), wc_mat)
    save_npz(str(output_dir / 'doc_tfidf.npz'), dt_mat)
    save_npz(str(output_dir / 'word_syntax.npz'), ws_mat)
    save_npz(str(output_dir / 'doc_syntax.npz'), ds_mat)

    # ── Сохранение меток ──
    save_json(output_dir / 'word_char_rows.json', wc_rows)
    save_json(output_dir / 'word_char_features.json', wc_feats)
    save_json(output_dir / 'doc_tfidf_rows.json', doc_paths)
    save_json(output_dir / 'doc_tfidf_features.json', dt_feats)
    save_json(output_dir / 'word_syntax_rows.json', ws_rows)
    save_json(output_dir / 'word_syntax_features.json', ws_feats)
    save_json(output_dir / 'doc_syntax_rows.json', doc_paths)
    save_json(output_dir / 'doc_syntax_features.json', ds_feats)

    # ── Метаданные ──
    metadata = {
        'word_char': {
            'matrix': 'word_char.npz',
            'rows_label': 'word_char_rows.json',
            'features_label': 'word_char_features.json',
            'shape': list(wc_mat.shape),
            'nnz': int(wc_mat.nnz),
            'params': {
                'ngram_lengths': list(NGRAM_LENGTHS),
                'boundary_start': BOUNDARY_START,
                'boundary_end': BOUNDARY_END,
                'description': (
                    'Частоты символьных n-грамм (длины 2 и 3) с граничными '
                    'символами ^ и $ для каждого уникального слова корпуса'
                ),
            },
        },
        'doc_tfidf': {
            'matrix': 'doc_tfidf.npz',
            'rows_label': 'doc_tfidf_rows.json',
            'features_label': 'doc_tfidf_features.json',
            'shape': list(dt_mat.shape),
            'nnz': int(dt_mat.nnz),
            'params': {
                'ngram_range': [1, 2],
                'smooth_idf': True,
                'norm': 'l2',
                'description': (
                    'TF-IDF по словным униграммам и биграммам, '
                    'общий словарь для всего корпуса'
                ),
            },
        },
        'word_syntax': {
            'matrix': 'word_syntax.npz',
            'rows_label': 'word_syntax_rows.json',
            'features_label': 'word_syntax_features.json',
            'shape': list(ws_mat.shape),
            'nnz': int(ws_mat.nnz),
            'params': {
                'spacy_model': SPACY_MODEL,
                'context_in': 'IN|dep|lemma_головы',
                'context_out': 'OUT|dep|lemma_зависимого',
                'word_key': 'lowercase surface form',
                'lemma_lower': True,
                'token_filter': 'WORD_RE.fullmatch (ru/lat letters only)',
                'exclude_dep': sorted(SKIP_DEPS - {None}),
                'alphabetic_only': True,
                'rows': 'all unique words from corpus (zero rows for isolated words)',
                'description': (
                    'Частоты синтаксических контекстов по dependency parsing: '
                    'для каждого употребления слова учитываются отношения '
                    'с головой (IN) и с зависимыми (OUT). '
                    'Леммы нормализуются .lower(). '
                    'Оба токена проверяются WORD_RE.fullmatch.'
                ),
            },
        },
        'doc_syntax': {
            'matrix': 'doc_syntax.npz',
            'rows_label': 'doc_syntax_rows.json',
            'features_label': 'doc_syntax_features.json',
            'shape': list(ds_mat.shape),
            'nnz': int(ds_mat.nnz),
            'params': {
                'spacy_model': SPACY_MODEL,
                'triple_format': 'POS_головы|dep|POS_зависимого',
                'exclude_dep': sorted(SKIP_DEPS - {None}),
                'alphabetic_only': True,
                'token_filter': 'WORD_RE.fullmatch (ru/lat letters only)',
                'description': (
                    'Частоты структурных троек POS_головы|dep|POS_зависимого '
                    'по результатам dependency parsing'
                ),
            },
        },
        'corpus': {
            'documents': len(doc_paths),
            'files_found': len(files),
            'files_read_ok': len(doc_paths),
            'files_errors': 0,
            'unique_words_total': unique_word_count,
        },
    }
    save_json(output_dir / 'metadata.json', metadata)

    print(
        f"Готово. Документов: {len(doc_paths)}, "
        f"уникальных слов: {unique_word_count}",
        file=sys.stderr,
    )


if __name__ == '__main__':
    main()
