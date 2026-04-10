import pandas as pd
import spacy
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.util import ngrams
import matplotlib.pyplot as plt
import os
import numpy as np

nlp = spacy.load("en_core_web_sm")

# 사용자 정의 도메인 스톱워드 추가
CUSTOM_STOPWORDS = {
    "product", "skin", "oil", "face", "anua", "heartleaf", "cleansing", "cleanser", "toner", "use"
}

# 전처리 함수들
def get_tokens(text):
    if pd.isnull(text): return []
    doc = nlp(text.lower())
    return [t.lemma_ for t in doc if t.is_alpha and not t.is_stop and t.lemma_ not in CUSTOM_STOPWORDS and len(t) > 2]

def get_bigrams(tokens):
    return ['_'.join(bg) for bg in ngrams(tokens, 2)]

def get_adj_noun(text):
    if pd.isnull(text): return []
    doc = nlp(text.lower())
    return [token.lemma_ + '_' + doc[i + 1].lemma_
            for i, token in enumerate(doc[:-1])
            if token.pos_ == "ADJ" and doc[i + 1].pos_ == "NOUN"]

preprocessing_modes = {
    "unigram": lambda text: get_tokens(text),
    "bigram": lambda text: get_bigrams(get_tokens(text)),
    "unibi_mix": lambda text: get_tokens(text) + get_bigrams(get_tokens(text)),
    "adj_noun": lambda text: get_adj_noun(text)
}

# 최적 k 탐색 함수 + plot 저장
def find_optimal_k(texts, dictionary, corpus, mode, rating, output_dir, min_k=2, max_k=6):
    scores = []
    for k in range(min_k, max_k + 1):
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, passes=15, random_state=42)
        cm = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence='c_v')
        scores.append((k, cm.get_coherence()))

    # plot 저장
    plt.figure()
    ks, cvs = zip(*scores)
    plt.plot(ks, cvs, marker='o')
    plt.title(f"Coherence Score by Topic Count\n({mode} | rating {rating})")
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence Score")
    plt.savefig(f"{output_dir}/plot_{mode}_rating{rating}.png")
    plt.close()

    best_k = max(scores, key=lambda x: x[1])[0]
    best_score = max(scores, key=lambda x: x[1])[1]
    return best_k, best_score

# === 진입점 ===
if __name__ == "__main__":
    df = pd.read_csv("../anua_cleansing_oil_reviews.csv")
    output_dir = "../outputs2"
    os.makedirs(output_dir, exist_ok=True)

    summary_rows = []

    for mode_name, preprocess_func in preprocessing_modes.items():
        with open(f"{output_dir}/topics_{mode_name}.txt", "w", encoding="utf-8") as f_out:
            f_out.write(f"==== {mode_name.upper()} ====\n")

            for rating in range(1, 6):
                subset = df[df["rating"] == rating]["Review Content"].dropna()
                texts = subset.apply(preprocess_func).tolist()
                texts = [t for t in texts if len(t) >= 3]

                if not texts or sum(len(t) for t in texts) == 0:
                    f_out.write(f"\n⭐ Rating {rating}: No valid reviews\n")
                    continue

                try:
                    # TF-IDF 기반 단어 선택 적용
                    joined_docs = [" ".join(t) for t in texts]
                    vectorizer = TfidfVectorizer(
                        min_df=1,
                        max_df=0.95,
                        max_features=3000,
                        stop_words="english"
                    )
                    tfidf_matrix = vectorizer.fit_transform(joined_docs)
                    selected_words = set(vectorizer.get_feature_names_out())
                    texts = [[word for word in doc if word in selected_words] for doc in texts]

                    dictionary = corpora.Dictionary(texts)
                    corpus = [dictionary.doc2bow(text) for text in texts]

                    if len(corpus) == 0:
                        f_out.write(f"\n⭐ Rating {rating}: Skipped due to empty corpus after TF-IDF filtering\n")
                        continue

                    optimal_k, best_score = find_optimal_k(texts, dictionary, corpus, mode_name, rating, output_dir)
                    f_out.write(f"\n⭐ Rating {rating} | Optimal K = {optimal_k} | Coherence = {best_score:.4f}\n")

                    model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=optimal_k, passes=15, random_state=42)
                    keywords = []
                    for i, topic in model.show_topics(num_topics=optimal_k, num_words=8, formatted=False):
                        topic_words = ", ".join([word for word, _ in topic])
                        f_out.write(f"  🔹 Topic {i + 1}: {topic_words}\n")
                        keywords.append(topic_words)

                    summary_rows.append({
                        "mode": mode_name,
                        "rating": rating,
                        "optimal_k": optimal_k,
                        "coherence": round(best_score, 4),
                        "num_reviews": len(texts),
                        "keywords": " | ".join(keywords)
                    })

                except Exception as e:
                    f_out.write(f"\n⚠️ Rating {rating}: Error occurred - {e}\n")

    pd.DataFrame(summary_rows).to_csv(f"{output_dir}/topic_summary.csv", index=False)
