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
- [Key Findings](#key-findings)
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
├── merge_cleansing_oil.py          # Merges 5 rating-tier Excel files → CSV (cleansing oil)
├── merge_toner.py                  # Merges 5 rating-tier Excel files → CSV (toner)
├── LDA_pipeline.py                 # Main LDA pipeline: 4 preprocessing modes × 2 products
├── export_tableau.py               # Expands topic keywords to row-per-keyword CSV for Tableau
├── export_tableau_tfidf.py         # Same as above for TF-IDF filtered LDA results
├── export_tableau_combined.py      # Merges main LDA + TF-IDF LDA into single CSV with pipeline column
├── export_tableau_laplace.py       # Expands Laplace segment data to row-per-segment CSV for Tableau
├── topic_summary_dashboard.py      # Interactive Dash dashboard for result exploration
│
└── experiments/
    ├── LDA_tfidf_pipeline.py       # LDA with TF-IDF pre-filtering (experimental)
    ├── laplace_trend_v1.py         # Laplace-perspective trend analysis, v1
    └── laplace_trend_v2.py         # Laplace-perspective trend analysis, v2 (VADER + adaptive TF-IDF fallback)
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
        │                                                  │
        ▼                                                  ▼
export_tableau.py                            export_tableau_tfidf.py
        │                                                  │
        └──────────────┬───────────────────────────────────┘
                       ▼
            export_tableau_combined.py
                       │
                       ▼
            anua_tableau_combined.csv
            (main_lda + tfidf_lda, pipeline column for comparison)
                       │
                       ▼
            Tableau Visualization (in progress — dashboard link will be added upon completion)


experiments/laplace_trend_v1.py / laplace_trend_v2.py
        │
        ▼
outputs_laplace_v1_cleansing_oil/ outputs_laplace_v1_toner/
outputs_laplace_cleansing_oil/    outputs_laplace_toner/
        │
        ▼
export_tableau_laplace.py
        │
        ▼
anua_tableau_laplace.csv
(row-per-segment, UP/DOWN/FLAT labels)
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

Three Tableau-ready CSV files are generated from the analysis results:

**`anua_tableau_master.csv`** — Main LDA results, row-per-keyword format

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

**`anua_tableau_combined.csv`** — Main LDA + TF-IDF LDA merged, with `pipeline` column (`main_lda` / `tfidf_lda`) for direct side-by-side comparison of both approaches.

**`anua_tableau_laplace.csv`** — Laplace sentiment trend segments, row-per-segment format

| Column | Description |
|---|---|
| `product` | `cleansing_oil` or `toner` |
| `star_rating` | 1–5 |
| `version` | `v1` (sign-lexicon) or `v2` (VADER) |
| `segment_id` | Segment index within the rating group |
| `segment_start` | Start position in review sequence |
| `segment_end` | End position in review sequence |
| `segment_length` | Duration of the segment |
| `segment_label` | `UP`, `DOWN`, or `FLAT` |
| `sentiment_backend` | Sentiment scorer used |

Tableau visualization in progress — dashboard link will be added upon completion.

---

## Key Findings

### 1. Preprocessing Strategy — Phrase-Level Analysis Outperforms Word-Level

Comparing c_v coherence scores across four preprocessing modes reveals a consistent pattern: **phrase-level representations (bigram, adj_noun) significantly outperform word-level (unigram, unibi_mix)** across both products.

| Mode | Cleansing Oil | Toner |
|---|---|---|
| adj_noun | 0.5545 | 0.6002 |
| bigram | 0.5700 | 0.6094 |
| unibi_mix | 0.3958 | 0.3837 |
| unigram | 0.3965 | 0.4004 |

This suggests that skincare consumers naturally describe their experiences through compound expressions (e.g. `acne_prone`, `gentle_cleanser`, `dry_skin`) rather than isolated words. Single-token models fail to capture the relational semantics that matter most in beauty product reviews.

**Additionally**, coherence peaks at star rating 3 for toner bigram (0.70), indicating that mid-satisfaction reviews yield the most topically separable structure — a finding with direct implications for product improvement prioritization.

---

### 2. Distinct Consumer Concern Profiles by Product

LDA topic modeling reveals that the two products attract fundamentally different consumer concerns, suggesting distinct positioning and target segments.

**Cleansing Oil** — Pore & Sebum Management
- High-rating keywords: `pore_cleansing`, `pore_look`, `work_great` (★5), `sebaceous_filament` (★3–4)
- Dominant use case: double-cleansing routine, oily/combination skin management
- Negative keywords centered on: acne triggers (`hormonal_acne`, `cystic_acne`), sensitivity reactions

**Toner** — Hydration, Skin Texture & Routine Integration
- High-rating keywords: `great_toner`, `korean_brand`, `combo_skin`, `radiant_skin` (★4), `natural_ingredient`, `skincare_routine` (★5)
- Dominant use case: daily hydration, K-beauty routine integration
- Negative keywords centered on: formulation issues, irritation, unmet expectations

> **Business implication**: The two products should not share the same marketing message. Cleansing Oil campaigns should emphasize pore refinement and sebum control; Toner campaigns should lead with hydration efficacy and K-beauty credibility.

---

### 3. Formulation Change as a Primary Churn Driver (Toner)

A cluster of formulation-related keywords appears exclusively in low-rating toner reviews, forming a clear churn signal:

- `changed_ingredient`, `new_ingredient` → ★2
- `new_formula`, `old_formula`, `prior_formula` → ★1

The co-occurrence of `old_formula` and `prior_formula` at the lowest rating tier directly reflects customer sentiment of "the previous version was better." Notably, this pattern is **absent in cleansing oil reviews**, suggesting that a segment of established toner users perceived a formulation change and responded negatively.

Conversely, `natural_ingredient` (★5) and `great_ingredient` (★4) confirm that ingredient quality is the primary satisfaction driver when perceived positively.

> **Business implication**: Formulation changes should be preceded by structured customer feedback collection. A/B testing with loyal customers before full rollout could prevent churn from established fan bases.

---

### 4. TikTok Viral Effect — Opposite Outcomes by Product

The keyword `tik_tok` surfaces in reviews of both products but at starkly different rating tiers:

- **Cleansing Oil**: `tik_tok` appears at ★4–5 → TikTok-driven customers are satisfied
- **Toner**: `tik_tok` appears at ★1–2 → TikTok-driven customers are disappointed

This pattern was consistently observed across both the main LDA pipeline and the TF-IDF filtered pipeline, increasing confidence that it reflects a real signal rather than a pipeline artifact.

> **Business implication**: Tik-Tok-associated purchases of the toner correlate with lower satisfaction ratings. Whether this reflects a content-expectation mismatch or a self-selection effect cannot be determined from keyword co-occurrence alone, but the pattern is consistent enough to warrant content strategy review.

---

### 5. Skin Type Segmentation — Sensitive Skin as a Polarized Core Segment

`sensitive_skin` appears across all star ratings (★1–5) for both products — the only keyword to do so. Topic-level analysis reveals that this polarization is not random: `sensitive_skin` consistently clusters into two structurally distinct topics across both products, exposing a meaningful sub-segment split.

| Skin Type | Cleansing Oil | Toner |
|---|---|---|
| sensitive_skin | ★1, 2, 3, 5 | ★1, 2, 3, 4, 5 |
| prone_skin | ★1, 2, 3, 5 | ★1, 2, 3, 4 |
| combo_skin | — | ★4 |
| dry_skin | ★1, 2, 3, 5 | ★2, 4 |
| oily_skin | ★2, 3, 5 | — |

**Cleansing Oil — two distinct sensitive skin topics:**
- **Topic 1 (Trouble/Acne)**: `sensitive_skin` co-occurs with `bad_acne`, `cystic_acne`, `hormonal_acne`, `more_pimple`, `black_head`, `clogged_pore` → acne-prone sensitive skin customers experiencing breakouts
- **Topic 2 (Cleansing Efficacy)**: `sensitive_skin` co-occurs with `double_cleansing`, `pore_cleansing`, `gentle_formula`, `sebaceous_filament`, `noticeable_difference` → sensitive skin customers satisfied with gentle yet effective cleansing

**Toner — two distinct sensitive skin topics:**
- **Topic 1 (SNS-Driven Disappointment)**: `sensitive_skin` co-occurs with `major_breakout`, `red_bump`, `irritated_skin`, `high_hope`, `tik_tok`, `social_medium` → sensitive skin customers who purchased via SNS discovery and experienced breakouts
- **Topic 2 (Daily Hydration Routine)**: `sensitive_skin` co-occurs with `dry_state`, `daily_moisturizer`, `much_moisture`, `mild_rosacea`, `natural_ingredient`, `consistent_use` → dry/reactive sensitive skin customers satisfied with hydration and gentle formulation

The data indicates that "sensitive skin" encompasses at least two distinct sub-segments — **acne-prone sensitive skin** (reactive to occlusive/comedogenic ingredients) and **dry/reactive sensitive skin** (responsive to hydration and gentle formulas) — who have fundamentally different and often opposite responses to the same products.

> **Business implication**: Current marketing that targets "sensitive skin" as a monolithic segment is likely contributing to polarized outcomes. Product guidelines and content should differentiate between acne-prone sensitive skin and dry/reactive sensitive skin. For the toner specifically, SNS-driven acquisition of acne-prone sensitive skin customers — amplified by TikTok discovery (see Finding 4) — is generating a structurally predictable disappointment pattern that can be mitigated with upfront skin-type-specific messaging.

---

### 6. Laplace-Perspective Sentiment Trend Analysis

Applying a signal-processing framework to the review sequence reveals macro-level sentiment dynamics beyond what keyword frequency can capture.

**Key observations (v2, VADER-based):**
- `DOWN` segments are consistently longer than `UP` segments across all star ratings and both products
- Cleansing Oil ★5 shows the longest `DOWN` segment among all high-rating groups
- Toner ★1–2 shows minimal `UP` recovery segments

**Interpreting the DOWN-dominant pattern:**
The prevalence of extended DOWN segments even in high-rating reviews (★4–5) is consistent with a narrative structure in which customers describe initial concerns or adjustment periods before arriving at a positive verdict. This interpretation is inferred from segment shape alone and cannot be confirmed without sentence-level annotation.

It should be noted that this interpretation is inferred from segment shape alone; without sentence-level annotation, the exact cause of DOWN segments cannot be confirmed from the current data.

For low-rating reviews (★1–2), the near-absence of UP recovery in the toner aligns with the keyword evidence from Finding 3 (formulation change) and Finding 5 (SNS-driven mismatch) — customers who experienced formulation disappointment or skin-type mismatch rarely shift to positive sentiment within their review text, indicating structural dissatisfaction rather than a rough start.

**v1 vs v2 comparison:**
VADER-based scoring (v2) produces longer, more stable segments than the simple sign-lexicon (v1), confirming that sentiment backend quality directly impacts trend extraction reliability.

> **Business implication**: The DOWN-before-verdict structure in high-rating reviews suggests that satisfied customers frequently overcome an initial concern before reaching satisfaction. Proactive communication addressing the most common early-stage concerns (e.g. purging, adjustment period, skin-type fit) — particularly at the point of TikTok or SNS discovery — could reduce early abandonment before the positive outcome is reached. For low-rating toner reviews, the absence of any sentiment recovery indicates structural dissatisfaction (not just a rough start), requiring product-level or communication-level intervention rather than retention messaging alone.
> Specifically, a targeted post-purchase communication sequence in the early usage period — addressing purging, skin adjustment periods, and skin-type-specific usage guidance — could convert early-stage dropouts into satisfied customers, particularly for the sensitive skin segments identified in Finding 5. For toner low-rating reviews, where no sentiment recovery exists, the intervention point should be pre-purchase: aligning TikTok and SNS content with realistic product expectations for acne-prone sensitive skin (see Finding 4 and Finding 5) rather than post-purchase retention messaging.

---

### 7. Pipeline Comparison — Main LDA vs TF-IDF Filtered LDA

Running both pipelines on the same dataset provides a cross-validation layer:

- **Agreement**: `tik_tok` at toner ★1, `combo_skin` and `great_toner` at toner ★4 appear in both pipelines → high-confidence signals
- **Divergence**: TF-IDF filtering removes domain-specific medical terms (`hormonal_acne`, `sebaceous_filament`, `cystic_acne`) that the main pipeline retains → for skincare corpora, aggressive TF-IDF filtering risks losing clinically relevant vocabulary
- **TF-IDF advantage**: Surfaces additional usage-context keywords (`new_moisturizer`, `few_day`, `natural_ingredient`) not captured by the main pipeline

> **Methodological conclusion**: For domain-specific NLP on beauty/skincare text, the main LDA pipeline with custom stopwords and POS-aware tokenization outperforms generic TF-IDF filtering. TF-IDF filtered results are best used as a secondary validation layer rather than a replacement.

---

## How to Run

```bash
# 1. Merge raw Excel files into CSV
python merge_cleansing_oil.py
python merge_toner.py

# 2. Run main LDA pipeline
python LDA_pipeline.py

# 3. Export Tableau-ready CSVs
python export_tableau.py
python export_tableau_tfidf.py
python export_tableau_combined.py
python export_tableau_laplace.py

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
