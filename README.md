# ANUA Amazon Review — NLP Analysis Pipeline

A comprehensive NLP analysis of Amazon customer reviews for two ANUA skincare products: **Heartleaf Pore Control Cleansing Oil** and **Toner**. The project applies LDA topic modeling across multiple preprocessing strategies to extract structured consumer insights from unstructured review text, and explores an experimental signal-processing approach using Laplace-perspective trend analysis.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Methodology](#methodology)
  - [Preprocessing](#preprocessing)
  - [LDA Topic Modeling](#lda-topic-modeling)
  - [TF-IDF Filtered LDA](#tf-idf-filtered-lda)
  - [Laplace-Perspective Trend Analysis](#laplace-perspective-trend-analysis)
- [Results](#results)
- [Tableau Export](#tableau-export)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)

---

## Project Overview

The goal of this project is to identify what consumers genuinely value or criticize about ANUA skincare products by analyzing the language patterns within Amazon reviews. Rather than relying on star ratings alone, the project deconstructs review text into latent topics using unsupervised machine learning, enabling a more nuanced understanding of consumer sentiment at each satisfaction level.

The analysis was designed to answer:
- What topics dominate negative vs. positive reviews?
- Do the two products share similar complaint/praise patterns, or do they diverge?
- Which preprocessing strategy produces the most semantically coherent topics?
- Can a signal-processing lens (Laplace transform) reveal macro-level sentiment trends across the review timeline?

---

## Dataset

- **Products**: ANUA Heartleaf Pore Control Cleansing Oil (B0BN2PX8V3) and ANUA Toner (B08CMS8P67)
- **Total reviews**: 1,000 (500 per product)
- **Sampling strategy**: 100 reviews per star rating (1–5), deliberately balanced to prevent class imbalance in topic modeling
- **Language**: English only; image-only and empty reviews excluded
- **Source**: Amazon.com (U.S. marketplace)
- **Collection method**: Amazon blocks automated scraping, so reviews were manually collected using a Chrome extension tool ("Amazon Review Export") and saved as Excel files per rating tier. Each file was then merged into a unified CSV via `merge_cleansing_oil.py` and `merge_toner.py`.

---

## Project Structure

```
anua-review-nlp/
│
├── merge_cleansing_oil.py       # Merges 5 rating-tier Excel files → CSV (cleansing oil)
├── merge_toner.py               # Merges 5 rating-tier Excel files → CSV (toner)
├── LDA_pipeline.py              # Main LDA pipeline: 4 preprocessing modes × 2 products
├── export_tableau.py            # Expands topic keywords to row-per-keyword CSV for Tableau
├── topic_summary_dashboard.py   # Interactive Dash dashboard for result exploration
│
└── experiments/
    ├── LDA_tfidf_pipeline.py    # LDA with TF-IDF pre-filtering (experimental)
    ├── laplace_trend_v1.py      # Laplace-perspective trend analysis, v1
    └── laplace_trend_v2.py      # Laplace-perspective trend analysis, v2 (VADER + adaptive TF-IDF fallback)
```

---

## Pipeline

```
Raw Data (Excel files, 100 reviews × 5 ratings × 2 products)
        │
        ▼
merge_cleansing_oil.py / merge_toner.py
        │
        ▼
anua_cleansing_oil_reviews.csv / anua_toner_reviews.csv
(500 reviews each, 100 per star rating)
        │
        ├──────────────────────────────────────────────────┐
        ▼                                                  ▼
LDA_pipeline.py                              experiments/LDA_tfidf_pipeline.py
(4 preprocessing modes)                      (TF-IDF pre-filtering + LDA)
        │                                                  │
        ▼                                                  ▼
outputs_cleansing_oil/                       outputs_tfidf_cleansing_oil/
outputs_toner/                               outputs_tfidf_toner/
        │
        ▼
export_tableau.py
        │
        ▼
anua_tableau_master.csv
(row-per-keyword, Tableau-ready)
        │
        ▼
Tableau Visualization (in progress)


experiments/laplace_trend_v1.py / laplace_trend_v2.py
        │
        ▼
outputs_laplace_v1_cleansing_oil/ outputs_laplace_v1_toner/
outputs_laplace_cleansing_oil/    outputs_laplace_toner/
```

---

## Methodology

### Preprocessing

Four preprocessing strategies were applied to each review corpus, allowing direct comparison of how token representation affects topic quality:

| Mode | Description |
|---|---|
| `unigram` | Lemmatized single tokens; stopwords + custom domain words removed |
| `bigram` | Two-word phrases (e.g. `acne_prone`, `oil_cleanser`) to capture compound semantics |
| `unibi_mix` | Union of unigram and bigram tokens for broader contextual coverage |
| `adj_noun` | POS-filtered adjective-noun pairs (e.g. `gentle_cleanser`, `dry_skin`) for sentiment-rich phrases |

All modes use spaCy (`en_core_web_sm`) for tokenization and lemmatization. A custom domain stopword list (`CUSTOM_STOPWORDS`) removes product-specific terms (e.g. `anua`, `heartleaf`, `cleansing`) that would otherwise dominate every topic without adding discriminative value.

Additionally, `LDA_pipeline.py` applies a context-verb handling rule: verbs like `like` and `feel` are retained only when followed by a meaningful word, forming a compound token (e.g. `feel_soft`) rather than being discarded or kept as noise.

### LDA Topic Modeling

Gensim's `LdaModel` was applied independently to each combination of:
- Product (cleansing oil / toner)
- Star rating (1–5)
- Preprocessing mode (unigram / bigram / unibi_mix / adj_noun)

The optimal number of topics *k* was selected automatically by computing the **c_v coherence score** for k = 2 to 6 and choosing the value that maximized coherence. This avoids arbitrary topic count selection and ensures each rating group is modeled at its natural granularity.

All models used `passes=15` and `random_state=42` for reproducibility.

### TF-IDF Filtered LDA

`experiments/LDA_tfidf_pipeline.py` extends the base pipeline by applying **TF-IDF word selection** before constructing the LDA corpus. After preprocessing, documents are vectorized with `TfidfVectorizer` (max 3,000 features, `max_df=0.95`) and only words appearing in the TF-IDF vocabulary are retained. This filters out tokens that are either too rare or too ubiquitous to carry topical signal, producing cleaner and more discriminative topic distributions.

Representative coherence scores from this approach:

| Product | Mode | Rating | Coherence |
|---|---|---|---|
| cleansing_oil | adj_noun | 1 | 0.6057 |
| toner | bigram | 3 | 0.7040 |
| toner | adj_noun | 3 | 0.6477 |

Bigram and adj_noun modes consistently outperformed unigram and unibi_mix, suggesting that phrase-level features better capture the nuanced language of skincare reviews.

### Laplace-Perspective Trend Analysis

Beyond topic modeling, two experimental modules explore whether review sentiment follows interpretable temporal patterns using a signal-processing framework.

**Core idea**: treat the sequence of reviews (ordered by index) as a discrete time-series signal, convert it to a continuous signal, and apply a low-pass filter analogous to the Laplace-domain transfer function H(s) = 1/(1+sτ) to extract smooth macro-level trends.

**Pipeline**:
1. **Signal construction**: each review is scored on two axes — sentiment (VADER compound score) and topic entropy (via NMF decomposition of TF-IDF features). These are standardized and combined into a single scalar signal with configurable weights.
2. **Continuization**: the discrete signal is smoothed into a continuous function using Gaussian kernel regression with bandwidth *bw*.
3. **Low-pass filtering**: an n-th order ODE filter (equivalent to cascading H(s) = 1/(1+sτ) n times) extracts the underlying trend by suppressing high-frequency noise.
4. **Segmentation**: Savitzky-Golay derivative estimation labels each segment as **UP**, **DOWN**, or **FLAT** based on a slope threshold.
5. **Diagnostic**: a numerical Laplace transform is computed on the continuous signal for frequency-domain inspection.

**v1 vs. v2**:

| Feature | v1 | v2 |
|---|---|---|
| Sentiment scorer | Simple sign-lexicon | VADER (with sign-lexicon fallback) |
| TF-IDF | Fixed config | Adaptive with 3-tier fallback |
| NMF convergence | `max_iter=1000` | `max_iter=600` |
| Robustness | Prototype | Production-ready |

---

## Results

Output files are generated per product under the following directories (excluded from version control via `.gitignore` as they are fully reproducible):

| Directory | Contents |
|---|---|
| `outputs_cleansing_oil/` | LDA topic `.txt` files + coherence plots + `topic_summary.csv` |
| `outputs_toner/` | Same as above for toner |
| `outputs_tfidf_cleansing_oil/` | TF-IDF filtered LDA results |
| `outputs_tfidf_toner/` | Same as above for toner |
| `outputs_laplace_v1_cleansing_oil/` | Laplace v1 segment summary |
| `outputs_laplace_v1_toner/` | Same as above for toner |
| `outputs_laplace_cleansing_oil/` | Laplace v2 (VADER) segment summary |
| `outputs_laplace_toner/` | Same as above for toner |

---

## Tableau Export

`export_tableau.py` merges both products' `topic_summary.csv` files and expands the keyword column into a **row-per-keyword** format optimized for Tableau exploration.

**Output**: `anua_tableau_master.csv`

| Column | Description |
|---|---|
| `product` | `cleansing_oil` or `toner` |
| `preprocessing_mode` | `unigram`, `bigram`, `unibi_mix`, `adj_noun` |
| `star_rating` | 1–5 |
| `topic_id` | Topic index within the rating group |
| `keyword` | Individual keyword |
| `keyword_rank` | 1 = most representative keyword in topic |
| `coherence_score` | c_v coherence of the topic model |
| `optimal_k` | Number of topics selected for this group |
| `num_reviews` | Number of reviews in this rating group |

Tableau visualization is in progress.

---

## How to Run

```bash
# 1. Merge raw Excel files into CSV
python merge_cleansing_oil.py
python merge_toner.py

# 2. Run main LDA pipeline
python LDA_pipeline.py

# 3. Export Tableau-ready CSV
python export_tableau.py

# 4. (Optional) Run experimental pipelines
python experiments/LDA_tfidf_pipeline.py
python experiments/laplace_trend_v1.py
python experiments/laplace_trend_v2.py

# 5. Launch interactive dashboard
python topic_summary_dashboard.py
```

---

## Dependencies

```
pandas
spacy (en_core_web_sm)
gensim
scikit-learn
scipy
nltk
matplotlib
vaderSentiment
plotly
dash
numpy
```

Install all dependencies:
```bash
pip install pandas spacy gensim scikit-learn scipy nltk matplotlib vaderSentiment plotly dash numpy
python -m spacy download en_core_web_sm
```
