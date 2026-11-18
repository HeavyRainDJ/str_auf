Вот полностью исправленный и компактный `solution.py` (около 190 строк), устраняющий все указанные ошибки: передача исходного текста в spaCy, корректная обработка пустых данных и директорий, fallback кодировки для TXT, строгий выход при ошибках и объединённый проход для синтаксических матриц.

### `solution.py`
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, os, re, sys
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
except ImportError:
    print("Ошибка: spaCy не установлен", file=sys.stderr)
    sys.exit(1)

def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == '.txt':
        try:
            with open(file_path, 'r', encoding='utf-8') as f: return f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='cp1251') as f: return f.read()
    elif suffix == '.html':
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            for tag in soup(["script", "style"]): tag.decompose()
            return soup.get_text(separator=' ')
    elif suffix == '.docx':
        return docx2python(str(file_path)).text
    elif suffix == '.odt':
        doc = load_odt(str(file_path))
        return '\n'.join(teletype.extractText(p) for p in doc.getElementsByType(text.P))
    raise RuntimeError(f"Неподдерживаемый формат: {file_path}")

def tokenize(text: str) -> list:
    return [w.lower() for w in re.findall(r'[a-zA-Zа-яА-ЯёЁ]+', text)]

def build_ngrams(word: str, n: int) -> list:
    padded = '^' + word + '$'
    return [padded[i:i+n] for i in range(len(padded) - n + 1)]

def build_word_ngram(corpus: list) -> tuple:
    words = sorted({w for _, tokens, _ in corpus for w in tokens})
    if not words: return sparse.csr_matrix((0, 0), dtype=np.float64), [], []
    ngrams = sorted({ng for w in words for ng in build_ngrams(w, 2) + build_ngrams(w, 3)})
    ng2idx = {ng: i for i, ng in enumerate(ngrams)}
    rows, cols, data = [], [], []
    for w_idx, w in enumerate(words):
        for ng, cnt in Counter(build_ngrams(w, 2) + build_ngrams(w, 3)).items():
            rows.append(w_idx); cols.append(ng2idx[ng]); data.append(float(cnt))
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(words), len(ngrams)), dtype=np.float64), words, ngrams

def build_tfidf(corpus: list) -> tuple:
    docs = [' '.join(tokens) for _, tokens, _ in corpus]
    vec = TfidfVectorizer(token_pattern=r'[a-zA-Zа-яА-ЯёЁ]+', lowercase=True, ngram_range=(1, 2), smooth_idf=True, norm='l2')
    if not docs or all(not d.strip() for d in docs):
        return sparse.csr_matrix((len(docs), 0), dtype=np.float64), []
    return vec.fit_transform(docs), vec.get_feature_names_out().tolist()

def build_syntax_matrices(corpus: list, nlp) -> tuple:
    words = sorted({w for _, tokens, _ in corpus for w in tokens})
    word_ctx = defaultdict(Counter)
    doc_structs = []
    
    for raw_text, tokens, _ in corpus:
        ds_counter = Counter()
        if raw_text.strip():
            doc = nlp(raw_text)
            for token in doc:
                if not re.fullmatch(r'[a-zA-Zа-яА-ЯёЁ]+', token.text): continue
                surf = token.text.lower()
                
                for child in token.children:
                    if re.fullmatch(r'[a-zA-Zа-яА-ЯёЁ]+', child.text):
                        word_ctx[surf][f"OUT|{child.dep_}|{child.lemma_.lower()}"] += 1
                
                if token.head != token and re.fullmatch(r'[a-zA-Zа-яА-ЯёЁ]+', token.head.text):
                    word_ctx[surf][f"IN|{token.dep_}|{token.head.lemma_.lower()}"] += 1
                    ds_counter[f"{token.head.pos_}|{token.dep_}|{token.pos_}"] += 1
        doc_structs.append(ds_counter)

    all_w_ctx = sorted({c for ctxs in word_ctx.values() for c in ctxs})
    c2idx = {c: i for i, c in enumerate(all_w_ctx)}
    wr, wc, wd = [], [], []
    for w_idx, w in enumerate(words):
        for c, cnt in word_ctx[w].items(): wr.append(w_idx); wc.append(c2idx[c]); wd.append(float(cnt))
    w_mat = sparse.csr_matrix((wd, (wr, wc)), shape=(len(words), len(all_w_ctx)), dtype=np.float64)

    all_d_triples = sorted({t for ds in doc_structs for t in ds})
    t2idx = {t: i for i, t in enumerate(all_d_triples)}
    dr, dc, dd = [], [], []
    for d_idx, ds in enumerate(doc_structs):
        for t, cnt in ds.items(): dr.append(d_idx); dc.append(t2idx[t]); dd.append(float(cnt))
    d_mat = sparse.csr_matrix((dd, (dr, dc)), shape=(len(corpus), len(all_d_triples)), dtype=np.float64)
    
    return w_mat, words, all_w_ctx, d_mat, all_d_triples

def save_mat(mat, rows, cols, out_dir, prefix):
    sparse.save_npz(out_dir / f"{prefix}.npz", mat)
    with open(out_dir / f"{prefix}_rows.json", 'w', encoding='utf-8') as f: json.dump(rows, f, ensure_ascii=False, indent=2)
    with open(out_dir / f"{prefix}_features.json", 'w', encoding='utf-8') as f: json.dump(cols, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    in_path, out_path = Path(args.input), Path(args.output)
    if not in_path.is_dir(): print(f"Ошибка: каталог не существует: {in_path}", file=sys.stderr); sys.exit(1)
    out_path.mkdir(parents=True, exist_ok=True)
    
    exts = {'.txt', '.docx', '.odt', '.html'}
    files = sorted([Path(r)/f for r, _, fs in os.walk(in_path) for f in fs if Path(f).suffix.lower() in exts])
    
    if not files:
        empty = sparse.csr_matrix((0, 0), dtype=np.float64)
        for p in ["word_char", "doc_tfidf", "word_syntax", "doc_syntax"]: save_mat(empty, [], [], out_path, p)
        with open(out_path / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump({p: {"shape": [0, 0]} for p in ["word_char", "doc_tfidf", "word_syntax", "doc_syntax"]}, f, indent=2)
        return

    corpus = []
    for fp in files:
        try:
            raw = extract_text(fp)
            corpus.append((raw, tokenize(raw), fp.relative_to(in_path).as_posix()))
        except Exception as e:
            print(f"КРИТИЧЕСКАЯ ОШИБКА: сбой чтения {fp}: {e}", file=sys.stderr)
            sys.exit(1)

    w_ngram_mat, w_ngram_rows, w_ngram_cols = build_word_ngram(corpus)
    save_mat(w_ngram_mat, w_ngram_rows, w_ngram_cols, out_path, "word_char")
    
    tfidf_mat, tfidf_cols = build_tfidf(corpus)
    save_mat(tfidf_mat, [c[2] for c in corpus], tfidf_cols, out_path, "doc_tfidf")
    
    try:
        nlp = spacy.load("ru_core_news_sm")
        if "parser" not in nlp.pipe_names: raise ValueError("Нет компонента parser")
    except Exception as e:
        print(f"Ошибка spaCy: {e}", file=sys.stderr); sys.exit(1)
        
    w_syn_mat, w_syn_rows, w_syn_cols, d_syn_mat, d_syn_cols = build_syntax_matrices(corpus, nlp)
    save_mat(w_syn_mat, w_syn_rows, w_syn_cols, out_path, "word_syntax")
    save_mat(d_syn_mat, [c[2] for c in corpus], d_syn_cols, out_path, "doc_syntax")
    
    meta = {
        "word_char": {"shape": list(w_ngram_mat.shape), "desc": "Частоты символьных n-грамм (2,3) с ^ и $"},
        "doc_tfidf": {"shape": list(tfidf_mat.shape), "desc": "TF-IDF униграмм и биграмм, smooth_idf=True, norm=l2"},
        "word_syntax": {"shape": list(w_syn_mat.shape), "desc": "Контексты OUT|dep|lemma и IN|dep|lemma"},
        "doc_syntax": {"shape": list(d_syn_mat.shape), "desc": "Частоты троек POS_head|dep|POS_child"},
        "parameters": {"ngram_lengths": [2, 3], "tfidf_smooth_idf": True, "tfidf_norm": "l2"}
    }
    with open(out_path / "metadata.json", 'w', encoding='utf-8') as f: json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Успешно сохранено в {out_path}")

if __name__ == "__main__":
    main()
```

### Математический смысл и ограничения матриц
1. **`word_char`**: Матрица частот символьных n-грамм (длины 2 и 3) с границами `^` и `$`. *Смысл*: морфологическое сходство слов. *Ограничение*: не учитывает семантику, высокоразмерна для больших словарей.
2. **`doc_tfidf`**: TF-IDF векторы документов. $TF \times \log(\frac{N+1}{df+1}+1)$ с L2-нормализацией. *Смысл*: статистическая значимость терминов. *Ограничение*: игнорирует порядок слов (Bag-of-Words), чувствителен к длине документа.
3. **`word_syntax`**: Частоты синтаксических ролей слова (входящие/исходящие зависимости с леммами). *Смысл*: дистрибутивная семантика на основе грамматических связей. *Ограничение*: требует качественного парсера, разрежена для редких слов.
4. **`doc_syntax`**: Частоты структурных троек `POS_головы|dep|POS_зависимого`. *Смысл*: синтаксический профиль стиля/сложности текста. *Ограничение*: теряет лексическое содержание, зависит от точности POS-теггинга.

### Команды запуска
```bash
pip install scipy scikit-learn spacy beautifulsoup4 docx2python odfpy
python -m spacy download ru_core_news_sm
python solution.py --input ./data --output ./results
```