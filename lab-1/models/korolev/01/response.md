user---

Напиши полноценное воспроизводимое решение лабораторной работы на Python 3.11 для русскоязычного корпуса: построение векторов СЛОВ и отдельно ЦЕЛЫХ ДОКУМЕНТОВ (1) на основе n-грамм и (2) на основе синтаксического разбора.
Вход: каталог с вложенными тематическими папками, всего 50 файлов: TXT, DOCX, ODT, HTML. Темы: анамнез, биохимия крови, комментаторы, неорганическая химия, органическая химия, спортивная медицина, ставки на спорт. Названия папок нельзя использовать как признаки. Не отправляй данные во внешние API.
Чтобы честно сравнить реализации, используй следующие определения:

Слова: lowercase последовательности русских/латинских букв, ё сохраняется. Для каждого уникального слова в корпусе построй частотный вектор символьных n-грамм длины 2 и 3, с граничными символами ^ и $. Пространство признаков общее для всех слов.
Документы: TF-IDF по словным униграммам и биграммам, общий словарь для всего корпуса; smooth_idf=True, norm='l2'.
Синтаксис: реальный dependency parsing русского текста (например, spaCy ru_core_news_sm), а не регулярные выражения или одни POS-теги. Вектор каждого слова: частоты контекстов OUT|dep|lemma_зависимого и IN|dep|lemma_головы по всем употреблениям; ключ слова - lowercase surface form. Корневые и пунктуационные связи не включать. Контексты учитывать только для пар токенов, которые целиком состоят из букв. Отдельно вектор документа: частоты структурных троек POS_головы|dep|POS_зависимого. Общие пространства признаков по корпусу.
Выход: четыре разреженные CSR-матрицы scipy в .npz и UTF-8 JSON списки меток строк/признаков для каждой матрицы. Метки документов - относительные пути, слова и признаки отсортированы; порядок документов воспроизводимый. Сохрани metadata.json с размерами и параметрами. Пустые документы сохраняй нулевыми строками; полностью пустой корпус не должен падать. Ошибки чтения файлов должны быть явно видны, нельзя молча пропускать файлы или имитировать синтаксический разбор, если модель отсутствует. HTML: только видимый текст без script/style; DOCX: также таблицы; ODT: абзацы и таблицы без дублирования.
CLI: python solution.py --input PATH --output PATH. Дай полный код единственного solution.py, requirements.txt и команды установки модели/запуска. Не сокращай код заглушками. В solution.py не скачивай зависимости автоматически. Опиши математический смысл четырех матриц и ограничения. Код будет реально запускаться, включая пустые файлы и небольшой корпус с заранее известными n-граммами.


korolev--

Это комплексная задача, требующая интеграции инструментов обработки естественного языка (NLP), работы с различными форматами документов и линейной алгебры. 

### Математический смысл четырех матриц

1.  **Матрица слов (n-граммы):** Каждая строка — уникальное слово корпуса, каждый столбец — уникальная символьная n-грамма (биграмма или триграмма). Значение в ячейке — количество вхождений данной n-граммы в слово. Это позволяет моделировать морфологическую схожесть слов (например, слова с общим корнем будут иметь схожие векторы).
2.  **Матрица документов (TF-IDF):** Каждая строка — документ, каждый столбец — уникальная слово-униграмма или биграмма. Значение рассчитывается как $\text{tf} \times \text{idf}$. Это классический подход к определению тематики документа через значимость слов относительно всего корпуса.
3.  **Матрица слов (Синтаксис):** Каждая строка — уникальное слово (surface form). Признаки — уникальные синтаксические контексты (связи «зависимый $\to$ голова» и «голова $\to$ зависимый»). Это векторное представление семантической роли слова в языке.
4.  **Матрица документов (Синтаксис):** Каждая строка — документ. Признаки — тройки $\text{POS}_{\text{head}}|\text{dep}|\text{POS}_{\text{child}}$. Эта матрица описывает «структурный профиль» текста (насколько часто встречаются определенные грамматические конструкции), что может быть полезно для определения стиля или жанра.

**Ограничения:**
*   **Память:** При очень больших корпусах матрицы становятся огромными, поэтому используются `scipy.sparse.csr_matrix`.
*   **Качество разбора:** Точность векторов синтаксиса напрямую зависит от качества модели `ru_core_news_sm`.
*   **Токенизация:** Использование только букв для синтаксических векторов отсекает цифры и спецсимволы, что может быть критично для узкоспециальных текстов (например, биохимии).

---

### requirements.txt
```text
spacy==3.7.4
scikit-learn==1.3.2
scipy==1.11.4
python-docx==1.1.0
odfpy==1.4.0
beautifulsoup4==4.12.2
numpy==1.26.0
```

### Инструкция по установке и запуску

1. **Установка зависимостей:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Загрузка модели spaCy для русского языка:**
   ```bash
   python -m spacy download ru_core_news_sm
   ```

3. **Запуск решения:**
   ```bash
   python solution.py --input ./my_corpus --output ./results
   ```

---

### solution.py

```python
import os
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import spacy
from bs4 import BeautifulSoup
from docx import Document
from odf import text, teletype
from odf.opendocument import load
from scipy import sparse

def read_text(file_path):
    ext = file_path.suffix.lower()
    try:
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == '.html':
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
                for script_or_style in soup(['script', 'style']):
                    script_or_style.decompose()
                return soup.get_text(separator=' ')
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
            # Извлекаем параграфы и ячейки таблиц
            for element in odt_doc.getElementsByType(text.P):
                full_text.append(teletype.get_string(element))
            for table in odt_doc.getElementsByType(text.Table):
                for row in table.getElementsByType(text.TableRow):
                    for cell in row.getElementsByType(text.TableCell):
                        for p in cell.getElementsByType(text.P):
                            full_text.append(teletype.get_string(p))
            return '\n'.join(full_text)
        else:
            return ""
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        raise e

def tokenize_simple(text):
    # lowercase последовательности русских/латинских букв, ё сохраняется
    return re.findall(r'[a-z\u0400-\u04FF]+', text.lower())

def get_char_ngrams(word):
    # n-граммы длины 2 и 3 с граничными символами ^ и $
    s = f"^{word}$"
    ngrams = []
    for n in [2, 3]:
        for i in range(len(s) - n + 1):
            ngrams.append(s[i:i+n])
    return ngrams

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to corpus folder')
    parser.add_argument('--output', required=True, help='Path to output folder')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Сбор данных
    all_files = sorted([f for f in input_path.rglob('*') if f.suffix.lower() in {'.txt', '.docx', '.odt', '.html'}])
    
    docs_content = [] # Список (путь, текст)
    for f_path in all_files:
        content = read_text(f_path)
        rel_path = str(f_path.relative_to(input_path))
        docs_content.append((rel_path, content))

    if not docs_content:
        print("Corpus is empty.")
        # Создаем пустые файлы метаданных, чтобы не упасть
        with open(output_path / "metadata.json", "w") as f:
            json.dump({"status": "empty_corpus"}, f)
        return

    # Загрузка spaCy
    try:
        nlp = spacy.load("ru_core_news_sm")
    except OSError:
        print("Error: spacy model 'ru_core_news_sm' not found. Run: python -m spacy download ru_core_news_sm")
        return

    # --- РАСЧЕТЫ ---

    # Общий словарь слов для всех задач
    corpus_words_global = set()
    processed_docs = [] # (path, tokens, spacy_doc)

    for path, text_content in docs_content:
        tokens = tokenize_simple(text_content)
        corpus_words_global.update(tokens)
        spacy_doc = nlp(text_content)
        processed_docs.append((path, tokens, spacy_doc))

    sorted_vocab = sorted(list(corpus_words_global))
    word_to_idx = {w: i for i, w in enumerate(sorted_vocab)}

    # 1. ВЕКТОРЫ СЛОВ (n-граммы)
    all_char_ngrams = set()
    word_ngram_counts = [] # список Counter для каждого слова из sorted_vocab
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

    # 2. ВЕКТОРЫ ДОКУМЕНТОВ (TF-IDF)
    # Используем sklearn для TF-IDF по униграммам и биграммам
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Подготавливаем тексты в виде строк с пробелами между токенами
    texts_for_tfidf = [" ".join(tokens) for _, tokens, _ in processed_docs]
    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x.split(), 
        preprocessor=lambda x: x, 
        ngram_range=(1, 2), 
        smooth_idf=True, 
        norm='l2'
    )
    matrix_doc_tfidf = vectorizer.fit_transform(texts_for_tfidf)
    sorted_tfidf_features = sorted(vectorizer.get_feature_names_out())
    # Перестраиваем матрицу, чтобы признаки были отсортированы
    feat_map = {f: i for i, f in enumerate(sorted_tfidf_features)}
    old_cols = vectorizer.get_feature_names_out()
    col_indices = [feat_map[f] for f in old_cols]
    matrix_doc_tfidf = matrix_doc_tfidf[:, col_indices] # это не совсем верно для CSR, пересоберем
    
    # Правильная пересортировка признаков TF-IDF
    tfidf_data = matrix_doc_tfidf.tocoo()
    new_cols = [feat_map[vectorizer.get_feature_names_out()[c]] for c in tfidf_data.col]
    matrix_doc_tfidf = sparse.csr_matrix((tfidf_data.data, (tfidf_data.row, new_cols)), 
                                          shape=(len(processed_docs), len(sorted_tfidf_features)))

    # 3. СИНТАКСИС: СЛОВА
    # Вектор слова: частоты контекстов OUT|dep|lemma_child и IN|dep|lemma_head
    word_syntax_counts = defaultdict(Counter)
    all_syntax_features = set()

    for _, _, doc in processed_docs:
        for token in doc:
            # Только токены из букв
            if not token.text.isalpha():
                continue
            
            surf = token.text.lower()
            # Проверяем зависимости
            for child in token.children:
                if not child.text.isalpha(): continue
                # Контекст: слово является головой (OUT)
                feat_out = f"OUT|{child.dep_}|{child.lemma_.lower()}"
                word_syntax_counts[surf][feat_out] += 1
                all_syntax_features.add(feat_out)
                # Контекст: ребенок является зависимым (IN)
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

    # 4. СИНТАКСИС: ДОКУМЕНТЫ
    # Вектор документа: частоты POS_head|dep|POS_child
    doc_syntax_counts = []
    all_doc_syn_features = set()

    for _, _, doc in processed_docs:
        counts = Counter()
        for token in doc:
            for child in token.children:
                # Игнорируем пунктуацию и корневые связи (если голова - корень, связь всё равно есть, но по условию не включать корневые связи)
                # Обычно в spacy 'ROOT' это dep головы. Условие: не включать корневые связи.
                # Интерпретируем как: не считать связи, где dep == 'ROOT'
                if child.dep_ == 'ROOT' or not token.text.isalpha() or not child.text.isalpha():
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

    # --- СОХРАНЕНИЕ ---
    
    def save_matrix(matrix, labels_rows, labels_cols, name):
        sparse.save_npz(output_path / f"{name}.npz", matrix)
        with open(output_path / f"{name}_rows.json", "w", encoding="utf-8") as f:
            json.dump(labels_rows, f, ensure_ascii=False)
        with open(output_path / f"{name}_cols.json", "w", encoding="utf-8") as f:
            json.dump(labels_cols, f, ensure_ascii=False)

    save_matrix(matrix_word_ngram, sorted_vocab, sorted_char_features, "word_ngram")
    save_matrix(matrix_doc_tfidf, [p for p, _, _ in processed_docs], sorted_tfidf_features, "doc_tfidf")
    save_matrix(matrix_word_syntax, sorted_vocab, sorted_syntax_word_features, "word_syntax")
    save_matrix(matrix_doc_syntax, [p for p, _, _ in processed_docs], sorted_doc_syn_features, "doc_syntax")

    metadata = {
        "word_ngram": {"shape": matrix_word_ngram.shape},
        "doc_tfidf": {"shape": matrix_doc_tfidf.shape, "params": "TF-IDF, unigrams+bigrams, smooth_idf=True, norm=l2"},
        "word_syntax": {"shape": matrix_word_syntax.shape},
        "doc_syntax": {"shape": matrix_doc_syntax.shape},
        "total_docs": len(processed_docs),
        "total_unique_words": len(sorted_vocab)
    }
    with open(output_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print(f"Successfully processed {len(processed_docs)} files. Results saved to {output_path}")

if __name__ == "__main__":
    main()
```