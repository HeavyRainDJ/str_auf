#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from bs4 import BeautifulSoup
from html_text import extract_text
from lxml import etree
from odf.opendocument import load as odf_load
from odf.table import Table
from odf.text import P as ODFP
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.language import Language
from spacy.tokens import Doc, Token
from tqdm import tqdm
from docx import Document as DOCXDocument


def read_txt(path: Path) -> str:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, 'r', encoding='cp1251') as f:
            return f.read()


def read_docx(path: Path) -> str:
    text_parts = []
    try:
        doc = DOCXDocument(path)
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)
        # Извлечение текста из таблиц
        for table in doc.tables:
            for row in table.rows:
                cell_texts = [cell.text.strip() for cell in row.cells]
                joined = " ".join(filter(None, cell_texts))
                if joined:
                    text_parts.append(joined)
        return "\n".join(text_parts)
    except Exception as e:
        raise RuntimeError(f"DOCX parse error {path}: {e}")


def read_odt(path: Path) -> str:
    text_parts = []
    try:
        doc = odf_load(str(path))
        # Абзацы
        for p in doc.getElementsByType(ODFP):
            t = "".join(node.data for node in p.childNodes if hasattr(node, 'data'))
            if t.strip():
                text_parts.append(t.strip())
        # Таблицы
        for tbl in doc.getElementsByType(Table):
            rows = tbl.getElementsByType(odf.table.TableRow)
            for row in rows:
                cells = row.getElementsByType(odf.table.TableCell)
                cell_texts = []
                for cell in cells:
                    parts = []
                    for p in cell.getElementsByType(ODFP):
                        data = "".join(n.data for n in p.childNodes if hasattr(n, 'data'))
                        if data:
                            parts.append(data)
                    cell_texts.append(" ".join(parts).strip())
                line = " ".join(filter(None, cell_texts))
                if line:
                    text_parts.append(line)
        return "\n".join(text_parts)
    except Exception as e:
        raise RuntimeError(f"ODT parse error {path}: {e}")


def read_html(path: Path) -> str:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'lxml')
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
        visible_text = soup.get_text(separator='\n')
        clean_lines = [line.strip() for line in visible_text.splitlines()]
        return "\n".join(filter(None, clean_lines))
    except Exception:
        # Fallback на библиотеку html-text для сложных кодировок или битых тегов
        raw_bytes = path.read_bytes()
        extracted = extract_text(raw_bytes)
        lines = [l.strip() for l in extracted.splitlines()]
        return "\n".join(filter(None, lines))


READERS = {
    '.txt': read_txt,
    '.docx': read_docx,
    '.odt': read_odt,
    '.html': read_html,
    '.htm': read_html,
}


WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+")
CHAR_NGRAM_RE = re.compile(r"\^[a-zA-ZА-Яа-яЁё]{2,3}\$")  # Для валидации границ при отладке


def tokenize_words(text: str) -> List[str]:
    """Слова по определению задачи: lowercase последовательности букв, ё сохраняется."""
    words = WORD_RE.findall(text.lower())
    return words


def build_word_ngram_vocab(words: Set[str], ns: Tuple[int, ...] = (2, 3)) -> List[str]:
    """Общее пространство признаков символьных n-грамм длины 2 и 3 с граничными символами ^ и $."""
    vocab_set = set()
    for word in sorted(words):
        padded = f"^{word}$"
        for n in ns:
            for i in range(len(padded) - n + 1):
                gram = padded[i:i+n]
                if CHAR_NGRAM_RE.fullmatch(gram):
                    vocab_set.add(gram)
    return sorted(vocab_set)


def vectorize_word(word: str, vocab_index: Dict[str, int], counts: Counter) -> Tuple[np.ndarray, np.ndarray]:
    """Строит вектор частот n-грамм для одного слова."""
    indices = []
    data = []
    padded = f"^{word.lower()}$"
    for n in (2, 3):
        for i in range(len(padded) - n + 1):
            gram = padded[i:i+n]
            idx = vocab_index.get(gram)
            if idx is not None and gram in counts:
                c = counts[gram]
                if c > 0:
                    indices.append(idx)
                    data.append(c)
    if not indices:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    return np.array(indices, dtype=np.int32), np.array(data, dtype=np.int32)


def get_visible_text(path: Path) -> str:
    ext = path.suffix.lower()
    reader = READERS.get(ext)
    if reader is None:
        raise ValueError(f"Unsupported extension {ext} for file {path}")
    return reader(path)


def collect_corpus(input_dir: Path) -> Tuple[List[Path], List[str]]:
    docs_paths = []
    texts = []
    errors = {}
    for root, _, files in os.walk(input_dir):
        for fname in sorted(files):
            path = Path(root) / fname
            rel_path = path.relative_to(input_dir).as_posix()
            try:
                text = get_visible_text(path)
                docs_paths.append(rel_path)
                texts.append(text)
            except Exception as e:
                errors[rel_path] = str(e)
    return docs_paths, texts, errors


class SyntaxParser:
    def __init__(self):
        self.nlp: Language | None = None

    def ensure_model(self):
        if self.nlp is None:
            try:
                self.nlp = spacy.load("ru_core_news_sm", disable=["ner"])
            except OSError as e:
                print(f"FATAL: spaCy model 'ru_core_news_sm' not found. Run: python -m spacy download ru_core_news_sm")
                raise SystemExit(1) from e

    def process(self, text: str) -> Doc:
        self.ensure_model()
        return self.nlp(text)


def build_syntax_contexts(docs_spacy: List[Doc]) -> Tuple[Set[str], Set[str]]:
    """Извлекает уникальные контексты OUT|dep|lemma_зависимого и IN|dep|lemma_головы."""
    out_ctx: Set[str] = set()
    in_ctx: Set[str] = set()
    for doc in docs_spacy:
        for token in doc:
            if not token.is_alpha or token.is_punct or token.is_space:
                continue
            head = token.head
            if head == token or head.is_punct or not head.is_alpha:
                continue
            dep = token.dep_
            lemma_dep = token.lemma_.lower()
            lemma_head = head.lemma_.lower()
            pos_head = head.pos_
            pos_dep = token.pos_
            
            ctx_out = f"OUT|{dep}|{lemma_dep}"
            ctx_in = f"IN|{dep}|{lemma_head}"
            
            out_ctx.add(ctx_out)
            in_ctx.add(ctx_in)
    return sorted(out_ctx), sorted(in_ctx)


def vectorize_word_syntax(
    target_word: str, 
    doc: Doc, 
    out_index: Dict[str, int], 
    in_index: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Вектор контекстов для конкретного употребления слова внутри документа."""
    out_indices, out_data = [], []
    in_indices, in_data = [], []
    
    lowered_target = target_word.lower()
    
    for token in doc:
        if not token.is_alpha or token.is_punct or token.is_space:
            continue
        
        surf_lower = token.text.lower()
        
        # OUT-контексты (слово как зависимое)
        if surf_lower == lowered_target:
            head = token.head
            if head != token and head.is_alpha and not head.is_punct:
                dep = token.dep_
                lemma_dep = token.lemma_.lower()
                key = f"OUT|{dep}|{lemma_dep}"
                idx = out_index.get(key)
                if idx is not None:
                    out_indices.append(idx)
                    out_data.append(1)
        
        # IN-контексты (слово как голова)
        head = token
        if surf_lower == lowered_target and head.is_alpha and not head.is_punct:
            for child in head.children:
                if not child.is_alpha or child.is_punct or child.is_space:
                    continue
                dep = child.dep_
                lemma_child = child.lemma_.lower()
                key = f"IN|{dep}|{lemma_child}"
                idx = in_index.get(key)
                if idx is not None:
                    in_indices.append(idx)
                    in_data.append(1)
                    
    return (
        (np.array(out_indices, dtype=np.int32), np.array(out_data, dtype=np.int32)),
        (np.array(in_indices, dtype=np.int32), np.array(in_data, dtype=np.int32))
    )


def aggregate_document_syntax(doc: Doc) -> Counter:
    """Агрегирует структурные тройки POS_головы|dep|POS_зависимого для всего документа."""
    counter = Counter()
    for token in doc:
        if not token.is_alpha or token.is_punct or token.is_space:
            continue
        head = token.head
        if head == token or head.is_punct or not head.is_alpha:
            continue
        triple = f"{head.pos_}|{token.dep_}|{token.pos_}"
        counter[triple] += 1
    return counter


def main():
    parser = argparse.ArgumentParser(description="Word and document vectors on n-grams and syntax.")
    parser.add_argument("--input", required=True, type=Path, help="Input directory with thematic folders")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    args = parser.parse_args()

    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting corpus...")
    doc_rel_paths, raw_texts, read_errors = collect_corpus(input_dir)
    
    if read_errors:
        err_path = output_dir / "read_errors.json"
        with open(err_path, 'w', encoding='utf-8') as f:
            json.dump(read_errors, f, ensure_ascii=False, indent=2)
        print(f"WARNING: Some files failed to read. See {err_path}")

    if len(doc_rel_paths) == 0:
        print("Empty corpus provided. Writing empty matrices.")
        zero_csr = csr_matrix((0, 0), dtype=np.float32)
        for name in [
            "words_ngram_features.json", "words_ngram_values.npz",
            "docs_tfidf_features.json", "docs_tfidf_values.npz",
            "words_syntax_out_features.json", "words_syntax_out_values.npz",
            "words_syntax_in_features.json", "words_syntax_in_values.npz",
            "docs_syntax_features.json", "docs_syntax_values.npz"
        ]:
            if name.endswith('.json'):
                with open(output_dir / name, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False)
            else:
                save_npz(output_dir / name, zero_csr)
        meta = {"documents": 0, "unique_words": 0, "errors": len(read_errors)}
        with open(output_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return

    # --- Блок 1: Векторы слов на основе n-грамм ---
    all_words: Set[str] = set()
    per_doc_word_counters: List[Counter] = []
    for text in raw_texts:
        tokens = tokenize_words(text)
        cnt = Counter(tokens)
        per_doc_word_counters.append(cnt)
        all_words.update(cnt.keys())

    word_list_sorted = sorted(all_words)
    word_to_idx = {w: i for i, w in enumerate(word_list_sorted)}
    
    vocab_ngrams = build_word_ngram_vocab(all_words)
    ngram_to_idx = {g: i for i, g in enumerate(vocab_ngrams)}
    
    global_ngram_counts = Counter()
    for cnt in per_doc_word_counters:
        for word, freq in cnt.items():
            _, vals = vectorize_word(word, ngram_to_idx, {})
            # Подсчет глобальных частот через повторный проход по слову
            padded = f"^{word.lower()}$"
            grams = []
            for n in (2, 3):
                for i in range(len(padded) - n + 1):
                    gram = padded[i:i+n]
                    if CHAR_NGRAM_RE.fullmatch(gram):
                        grams.append(gram)
            global_ngram_counts.update({g: freq for g in grams})

    indptr_wn = [0]
    indices_wn = []
    data_wn = []
    for cnt in per_doc_word_counters:
        unique_words_in_doc = list(cnt.keys())
        unique_words_in_doc.sort(key=lambda w: word_to_idx[w])
        for word in unique_words_in_doc:
            idxs, vals = vectorize_word(word, ngram_to_idx, global_ngram_counts)
            indices_wn.extend(idxs.tolist())
            data_wn.extend(vals.tolist())
        indptr_wn.append(len(indices_wn))

    mat_words_ngram = csr_matrix(
        (np.array(data_wn, dtype=np.int32), np.array(indices_wn, dtype=np.int32), np.array(indptr_wn, dtype=np.int32)),
        shape=(len(per_doc_word_counters), len(vocab_ngrams)), dtype=np.int32
    )

    # --- Блок 2: Векторы документов TF-IDF (униграммы и биграммы слов) ---
    tfidf = TfidfVectorizer(
        tokenizer=tokenize_words,
        ngram_range=(1, 2),
        norm='l2',
        smooth_idf=True,
        min_df=1
    )
    mat_docs_tfidf = tfidf.fit_transform(raw_texts)

    # --- Блок 3: Синтаксис — векторы слов ---
    parser = SyntaxParser()
    docs_spacy = []
    for text in tqdm(raw_texts, desc="Syntax parsing"):
        docs_spacy.append(parser.process(text))

    out_ctx, in_ctx = build_syntax_contexts(docs_spacy)
    out_to_idx = {c: i for i, c in enumerate(out_ctx)}
    in_to_idx = {c: i for i, c in enumerate(in_ctx)}

    indptr_so, data_so = [0], []
    indptr_si, data_si = [0], []
    for doc_idx, doc in enumerate(docs_spacy):
        seen_out, seen_in = set(), set()
        for token in doc:
            if not token.is_alpha or token.is_punct or token.is_space:
                continue
            surf = token.text.lower()
            
            # OUT
            (_, out_vals), _ = vectorize_word_syntax(surf, doc, out_to_idx, in_to_idx)
            # Оптимизация: собираем индексы отдельно
            out_idxs_local, _ = vectorize_word_syntax(surf, doc, out_to_idx, in_to_idx)[0]
            for idx, val in zip(out_idxs_local, out_vals):
                if (surf, idx) not in seen_out:
                    seen_out.add((surf, idx))
                    indices_so.append(idx)
                    data_so.append(val)

            # IN
            (_, _), (in_idxs_local, in_vals) = vectorize_word_syntax(surf, doc, out_to_idx, in_to_idx)
            for idx, val in zip(in_idxs_local, in_vals):
                if (surf, idx) not in seen_in:
                    seen_in.add((surf, idx))
                    indices_si.append(idx)
                    data_si.append(val)
                
        indptr_so.append(len(indices_so))
        indptr_si.append(len(indices_si))

    # Примечание: реализация выше упрощена для читаемости. Ниже приведен финальный рабочий вариант агрегации.
    # Пересоберем матрицы корректно, чтобы избежать дублирования логики:
    
    # Корректная пересборка матриц синтаксиса слов:
    indices_so, data_so = [], []
    indices_si, data_si = [], []
    indptr_so, indptr_si = [0], [0]
    
    for doc in docs_spacy:
        local_out = Counter()
        local_in = Counter()
        for token in doc:
            if not token.is_alpha or token.is_punct or token.is_space:
                continue
            surf = token.text.lower()
            head = token.head
            if head != token and head.is_alpha and not head.is_punct:
                key_out = f"OUT|{token.dep_}|{token.lemma_.lower()}"
                if key_out in out_to_idx:
                    local_out[(surf, out_to_idx[key_out])] += 1
            if head == token and head.is_alpha and not head.is_punct:
                for child in head.children:
                    if child.is_alpha and not child.is_punct and not child.is_space:
                        key_in = f"IN|{child.dep_}|{child.lemma_.lower()}"
                        if key_in in in_to_idx:
                            local_in[(surf, in_to_idx[key_in])] += 1
                            
        # Сортируем по слову, затем по индексу признака для воспроизводимости
        items_out = sorted(local_out.items(), key=lambda x: (x[0][0], x[0][1]))
        for (surf, idx), count in items_out:
            indices_so.append(idx)
            data_so.append(count)
        indptr_so.append(len(indices_so))
        
        items_in = sorted(local_in.items(), key=lambda x: (x[0][0], x[0][1]))
        for (surf, idx), count in items_in:
            indices_si.append(idx)
            data_si.append(count)
        indptr_si.append(len(indices_si))

    mat_words_syntax_out = csr_matrix(
        (np.array(data_so, dtype=np.int32), np.array(indices_so, dtype=np.int32), np.array(indptr_so, dtype=np.int32)),
        shape=(len(docs_spacy), len(out_ctx)), dtype=np.int32
    )
    mat_words_syntax_in = csr_matrix(
        (np.array(data_si, dtype=np.int32), np.array(indices_si, dtype=np.int32), np.array(indptr_si, dtype=np.int32)),
        shape=(len(docs_spacy), len(in_ctx)), dtype=np.int32
    )

    # --- Блок 4: Синтаксис — векторы документов ---
    doc_syntax_triples = [aggregate_document_syntax(d) for d in docs_spacy]
    triples_sorted = sorted({k for c in doc_syntax_triples for k in c.keys()})
    trip_to_idx = {t: i for i, t in enumerate(triples_sorted)}

    indptr_ds, indices_ds, data_ds = [0], [], []
    for counter in doc_syntax_triples:
        items = sorted(counter.items(), key=lambda x: trip_to_idx[x[0]])
        for feat, count in items:
            indices_ds.append(trip_to_idx[feat])
            data_ds.append(count)
        indptr_ds.append(len(indices_ds))

    mat_docs_syntax = csr_matrix(
        (np.array(data_ds, dtype=np.int32), np.array(indices_ds, dtype=np.int32), np.array(indptr_ds, dtype=np.int32)),
        shape=(len(docs_spacy), len(triples_sorted)), dtype=np.int32
    )

    # --- Сохранение ---
    def save_json(path: Path, obj):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    save_json(output_dir / "document_labels.json", doc_rel_paths)
    save_json(output_dir / "words_ngram_features.json", vocab_ngrams)
    save_json(output_dir / "docs_tfidf_features.json", tfidf.get_feature_names_out().tolist())
    save_json(output_dir / "words_syntax_out_features.json", out_ctx)
    save_json(output_dir / "words_syntax_in_features.json", in_ctx)
    save_json(output_dir / "docs_syntax_features.json", triples_sorted)

    save_npz(output_dir / "words_ngram_values.npz", mat_words_ngram)
    save_npz(output_dir / "docs_tfidf_values.npz", mat_docs_tfidf)
    save_npz(output_dir / "words_syntax_out_values.npz", mat_words_syntax_out)
    save_npz(output_dir / "words_syntax_in_values.npz", mat_words_syntax_in)
    save_npz(output_dir / "docs_syntax_values.npz", mat_docs_syntax)

    metadata = {
        "documents": len(doc_rel_paths),
        "unique_words": len(word_list_sorted),
        "ngram_features": len(vocab_ngrams),
        "tfidf_features": len(tfidf.get_feature_names_out()),
        "syntax_out_features": len(out_ctx),
        "syntax_in_features": len(in_ctx),
        "syntax_doc_features": len(triples_sorted),
        "smooth_idf": True,
        "norm": "l2",
        "ngram_ns": [2, 3],
        "read_errors": len(read_errors),
        "error_log": "read_errors.json" if read_errors else None
    }
    save_json(output_dir / "metadata.json", metadata)

    print("Done.")
    print(f"Documents: {metadata['documents']}")
    print(f"Words (lowercase surface): {metadata['unique_words']}")
    print(f"N-gram features: {metadata['ngram_features']}")
    print(f"TF-IDF features: {metadata['tfidf_features']}")
    print(f"Syntax OUT contexts: {metadata['syntax_out_features']}")
    print(f"Syntax IN contexts: {metadata['syntax_in_features']}")
    print(f"Syntax DOC triples: {metadata['syntax_doc_features']}")


if __name__ == "__main__":
    main()