# Lab 3 progress

Updated: 2026-10-21.

| Model | Current state | Evidence |
|---|---|---|
| GigaChat | Accepted, iteration 06 | Empty, mixed-6 and full-50 validation pass; final files in `models/gigachat/final` |
| Qwen | Accepted, iteration 04 | Empty, mixed-6 and full-50 validation pass; final files in `models/qwen/final` |
| DeepSeek | Accepted, iteration 03 | Empty, all-empty-2, mixed-6 and full-50 validation pass; final files in `models/deepseek/final` |
| Korolev AI | Accepted, iteration 04 | Empty, all-empty-2, mixed-6 and full-50 validation pass; final files in `models/korolev-ai/final` |

## Accepted GigaChat result

- 50 documents.
- Vector shapes: count BoW `50 x 7165`, lemma TF-IDF `50 x 4480`, BGE-M3 `50 x 1024`.
- 33 pairwise matrices validated.
- Best ROC AUC: BGE Spearman `0.915033`.
- Signed-split BGE weighted Jaccard/Dice ROC AUC: `0.912474` each.
- BGE weighted Jaccard range: `0.230147 .. 1.000000`.
- BGE weighted Dice range: `0.374178 .. 1.000000`.

## Accepted Qwen result

- 50 documents.
- Vector shapes: count BoW `50 x 7165`, lemma TF-IDF `50 x 4480`, BGE-M3 `50 x 1024`.
- 33 pairwise matrices validated.
- Best ROC AUC: BGE cosine / Euclidean closeness `0.953218`.
- Signed-split BGE weighted Jaccard/Dice ROC AUC: `0.945851` each.
- BGE weighted Jaccard range: `0.286407 .. 1.000000`.
- BGE weighted Dice range: `0.445283 .. 1.000000`.

## Accepted Korolev AI result

- 50 documents.
- Vector shapes: count BoW `50 x 7549`, lemma TF-IDF `50 x 4633`, BGE-M3 `50 x 1024`.
- 33 pairwise matrices validated.
- Best ROC AUC: BGE Spearman `0.943656`.
- Signed-split BGE weighted Jaccard/Dice ROC AUC: `0.942395` each.
- BGE weighted Jaccard range: `0.225772 .. 1.000000`.
- BGE weighted Dice range: `0.368375 .. 1.000000`.

## Accepted DeepSeek result

- 50 documents.
- Vector shapes: count BoW `50 x 7497`, lemma TF-IDF `50 x 4473`, BGE-M3 `50 x 1024`.
- 33 pairwise matrices validated; long-format `all_pairs.csv` has 40,425 rows.
- Best ROC AUC: BGE Spearman `0.943644`.
- Signed-split BGE weighted Jaccard/Dice ROC AUC: `0.942395` each.
- BGE weighted Jaccard range: `0.225772 .. 1.000000`.
- BGE weighted Dice range: `0.368375 .. 1.000000`.
