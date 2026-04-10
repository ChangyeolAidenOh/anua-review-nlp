"""
ANUA Amazon Review - LDA Topic Modeling Pipeline
Products: Heartleaf Pore Control Cleansing Oil, Toner
Preprocessing: Unigram, Bigram, Unibi-Mix, Adj-Noun Pairs
"""

import os
import pandas as pd
import spacy
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from nltk.util import ngrams
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")

# Domain-specific stopwords for ANUA skincare reviews
CUSTOM_STOPWORDS = {
    "product", "skin", "oil", "face", "anua", "heartleaf",
    "cleansing", "cleanser", "toner", "use"
}

# Context verbs: kept only when followed by a meaningful word
CONTEXT_VERBS = {"like", "feel"}


# ── Preprocessing functions ───────────────────────────────────────────────────

def get_tokens(text):
    """Lemmatized unigrams with POS-aware context verb handling."""
    if pd.isnull(text):
        return []
    doc = nlp(text.lower())
    result = []
    for i, token in enumerate(doc):
        if not token.is_alpha or token.is_stop:
            continue
        if token.lemma_ in CUSTOM_STOPWORDS or len(token) <= 2:
            continue
        if token.pos_ == "VERB" and token.lemma_ in CONTEXT_VERBS:
            if i + 1 < len(doc) and doc[i + 1].is_alpha:
                result.append(f"{token.lemma_}_{doc[i + 1].lemma_}")
            continue
        result.append(token.lemma_)
    return result


def get_bigrams(tokens):
    """Two-word phrase combinations from token list."""
    return ["_".join(bg) for bg in ngrams(tokens, 2)]


def get_adj_noun(text):
    """Adjective-noun pairs for nuanced sentiment capture (e.g. gentle_cleanser)."""
    if pd.isnull(text):
        return []
    doc = nlp(text.lower())
    return [
        token.lemma_ + "_" + doc[i + 1].lemma_
        for i, token in enumerate(doc[:-1])
        if token.pos_ == "ADJ" and doc[i + 1].pos_ == "NOUN"
    ]


PREPROCESSING_MODES = {
    "unigram":   lambda text: get_tokens(text),
    "bigram":    lambda text: get_bigrams(get_tokens(text)),
    "unibi_mix": lambda text: get_tokens(text) + get_bigrams(get_tokens(text)),
    "adj_noun":  lambda text: get_adj_noun(text),
}


# ── Topic modeling ────────────────────────────────────────────────────────────

def find_optimal_k(texts, dictionary, corpus, mode, rating, output_dir, min_k=2, max_k=6):
    """
    Search k in [min_k, max_k], save coherence plot, return (best_k, best_score).
    Bug fix: output_dir passed as parameter so each product saves to its own folder.
    """
    scores = []
    for k in range(min_k, max_k + 1):
        lda = LdaModel(
            corpus=corpus, id2word=dictionary,
            num_topics=k, passes=15, random_state=42
        )
        cm = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence="c_v")
        scores.append((k, cm.get_coherence()))

    ks, cvs = zip(*scores)
    plt.figure()
    plt.plot(ks, cvs, marker="o")
    plt.title(f"Coherence Score by Topic Count\n({mode} | rating {rating})")
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence Score")
    plt.savefig(f"{output_dir}/plot_{mode}_rating{rating}.png")
    plt.close()

    best_k, best_score = max(scores, key=lambda x: x[1])
    return best_k, best_score


# ── Main pipeline ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    products = [
        ("cleansing_oil", "anua_cleansing_oil_reviews.csv"),
        ("toner",         "anua_toner_reviews.csv"),
    ]

    for product_name, file_path in products:
        print(f"\n{'='*50}\n▶ Processing: {product_name}\n{'='*50}")

        df = pd.read_csv(file_path)
        output_dir = f"outputs_{product_name}"
        os.makedirs(output_dir, exist_ok=True)

        summary_rows = []

        for mode_name, preprocess_func in PREPROCESSING_MODES.items():
            print(f"\n  [Mode: {mode_name}]")

            with open(f"{output_dir}/topics_{mode_name}.txt", "w", encoding="utf-8") as f_out:
                f_out.write(f"==== {product_name.upper()} | {mode_name.upper()} ====\n")

                for rating in range(1, 6):
                    subset = df[df["rating"] == rating]["Review Content"].dropna()
                    texts = subset.apply(preprocess_func).tolist()
                    texts = [t for t in texts if len(t) >= 3]

                    if not texts:
                        f_out.write(f"\n⭐ Rating {rating}: No valid reviews\n")
                        continue

                    try:
                        dictionary = corpora.Dictionary(texts)
                        corpus = [dictionary.doc2bow(text) for text in texts]

                        optimal_k, best_score = find_optimal_k(
                            texts, dictionary, corpus, mode_name, rating, output_dir
                        )

                        model = LdaModel(
                            corpus=corpus, id2word=dictionary,
                            num_topics=optimal_k, passes=15, random_state=42
                        )

                        f_out.write(f"\n⭐ Rating {rating} | Optimal K = {optimal_k} | Coherence = {best_score:.4f}\n")

                        keywords = []
                        for i, topic in model.show_topics(num_topics=optimal_k, num_words=8, formatted=False):
                            topic_words = ", ".join([word for word, _ in topic])
                            f_out.write(f"  🔹 Topic {i + 1}: {topic_words}\n")
                            keywords.append(topic_words)

                        summary_rows.append({
                            "product":     product_name,
                            "mode":        mode_name,
                            "rating":      rating,
                            "optimal_k":   optimal_k,
                            "coherence":   round(best_score, 4),
                            "num_reviews": len(texts),
                            "keywords":    " | ".join(keywords),
                        })

                        print(f"    ⭐ Rating {rating} → k={optimal_k}, coherence={best_score:.4f}")

                    except Exception as e:
                        f_out.write(f"\n⚠️ Rating {rating}: Error - {e}\n")
                        print(f"    ⚠️ Rating {rating}: Error - {e}")

        pd.DataFrame(summary_rows).to_csv(f"{output_dir}/topic_summary.csv", index=False)
        print(f"\n  ✅ Saved: {output_dir}/topic_summary.csv")
