# Sentiment Analysis using BERT Embeddings

A research-oriented project exploring multiple methods for **lexicon-based sentiment analysis** using transformer models (BERT, DistilBERT, RoBERTa). The core idea: instead of relying on hand-crafted sentiment lexicons, word-level sentiment scores are derived from contextual embeddings.

---

## Project Structure

```
├── Sentiment_Analysis_Final.ipynb   # Main notebook: lexicon construction & evaluation methods
├── IPD_2.ipynb                      # Experimentation notebook: attention & gradient-based scoring
```

---

## Methods Implemented

### Embedding-Based Lexicon Construction (`Sentiment_Analysis_Final.ipynb`)

| Method | Description |
|---|---|
| **CLS Norm** | Uses the `[CLS]` token embedding from BERT; word scores derived via L2 norm |
| **Averaged BERT Embeddings** | Averages hidden state embeddings across positive/negative reviews to build word-level sentiment scores |
| **Zero-Shot (RoBERTa)** | Uses `cardiffnlp/twitter-roberta-base-sentiment` to score words without any task-specific training |
| **AFINN Baseline** | Classical lexicon-based method used as a comparison baseline |

### Attention & Gradient-Based Scoring (`IPD_2.ipynb`)

| Method | Description |
|---|---|
| **L2 Norm (DistilBERT)** | Extracts term weights from DistilBERT hidden states using L2 normalization |
| **Hidden State Method** | Uses raw hidden state representations for term weighting |
| **Gradient-Based** | Computes word importance via gradient magnitudes w.r.t. input embeddings |
| **CLS Cosine Similarity** | Measures cosine similarity between term and document `[CLS]` embeddings |
| **Attention Weights** | Uses self-attention scores from DistilBERT and RoBERTa as proxy for word importance |

---

## Datasets Used

- `amazon_cells_labelled.txt` — Amazon product reviews (tab-separated, binary labels)
- `imdb_labelled.txt` — IMDB movie reviews
- `sentiment_reviews.csv` — Aggregated review dataset (used in IPD_2)
- `yelp_reviews_sentiment.csv` / `amazon_reviews_sentiment.csv` — Extended evaluation sets

All datasets follow the format: `review \t label` (0 = negative, 1 = positive).

---

## Models Used

- [`bert-base-uncased`](https://huggingface.co/bert-base-uncased)
- [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased)
- [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
- [`cardiffnlp/twitter-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)

---

## Installation

```bash
pip install transformers torch scikit-learn pandas numpy nltk afinn tqdm
```

> **Note:** Notebooks were originally developed on Google Colab. If running locally, replace `google.colab.files.upload()` calls with standard `pd.read_csv("path/to/file")`.

---

## Evaluation Metrics

All methods are evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score (macro-averaged)

Computed via `sklearn.metrics`.

---

## Key Dependencies

| Package | Purpose |
|---|---|
| `transformers` | BERT, DistilBERT, RoBERTa models & tokenizers |
| `torch` | Model inference, gradient computation |
| `scikit-learn` | Metrics, cosine similarity |
| `nltk` | Stopword filtering, tokenization |
| `afinn` | AFINN baseline lexicon |
| `pandas` / `numpy` | Data handling |

---

## Notes

- All models run in **inference mode only** — no fine-tuning is performed.
- GPU acceleration is used automatically if available (`torch.cuda.is_available()`).
- Reviews are capped at **1000 samples** per run for compute efficiency.
- Stopwords are filtered before scoring in all embedding-based methods.
