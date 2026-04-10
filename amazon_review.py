
"""""
보완:POS filtering + 정규표현식 후처리
"""""
import pandas as pd
import spacy
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from nltk.util import ngrams
import matplotlib.pyplot as plt
import os

nlp = spacy.load("en_core_web_sm")

### Stopwords list to prevent less-meaningful words from the reviews.
## Custom stopwords
CUSTOM_STOPWORDS = {
    "product", "skin", "oil", "face", "anua", "heartleaf", "cleansing", "cleanser", "toner", "use"
}

### Exception words to the single-verbs
# 의미 없는 단독 동사 제거를 위한 예외 목록
CONTEXT_VERBS = {"like", "feel"}

### Preprocessing functions
# 전처리 함수들
def get_tokens(text):
    if pd.isnull(text): return []
    doc = nlp(text.lower())
    result = []
    for i, token in enumerate(doc):
        if token.is_alpha and not token.is_stop and token.lemma_ not in CUSTOM_STOPWORDS and len(token) > 2:
            # 단독 의미 없는 동사 제거: like, feel 등
            if token.pos_ == "VERB" and token.lemma_ in CONTEXT_VERBS:
                # 앞뒤 문맥 확인
                if i + 1 < len(doc) and doc[i + 1].is_alpha:
                    next_token = doc[i + 1].lemma_
                    result.append(f"{token.lemma_}_{next_token}")
                # 뒤에 nothing이 없으면 건너뜀
                continue
            result.append(token.lemma_)
    return result

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
def find_optimal_k(texts, dictionary, corpus, mode, rating, min_k=2, max_k=6):
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
    plt.savefig(f"outputs/plot_{mode}_rating{rating}.png")
    plt.close()

    # 최고값 반환
    best_k = max(scores, key=lambda x: x[1])[0]
    best_score = max(scores, key=lambda x: x[1])[1]
    return best_k, best_score

# === 진입점 ===
if __name__ == "__main__":
    for product_name, file_path in [
        ("cleansing_oil", "anua_cleansing_oil_reviews.csv"),
        ("toner", "anua_toner_reviews.csv")
    ]:
        df = pd.read_csv(file_path)

        # 결과 폴더 생성
        output_dir = f"outputs_{product_name}"
        os.makedirs(output_dir, exist_ok=True)

        summary_rows = []

        for mode_name, preprocess_func in preprocessing_modes.items():
            with open(f"{output_dir}/topics_{mode_name}.txt", "w", encoding="utf-8") as f_out:
                f_out.write(f"==== {mode_name.upper()} ====\n")

                for rating in range(1, 6):
                    subset = df[df["rating"] == rating]["Review Content"].dropna()
                    texts = subset.apply(preprocess_func).tolist()
                    texts = [t for t in texts if len(t) >= 3]

                    if not texts:
                        f_out.write(f"\n⭐ Rating {rating}: No valid reviews\n")
                        continue

                    dictionary = corpora.Dictionary(texts)
                    corpus = [dictionary.doc2bow(text) for text in texts]

                    optimal_k, best_score = find_optimal_k(texts, dictionary, corpus, mode_name, rating)
                    f_out.write(f"\n⭐ Rating {rating} | Optimal K = {optimal_k} | Coherence = {best_score:.4f}\n")

                    model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=optimal_k, passes=15, random_state=42)
                    keywords = []
                    for i, topic in model.show_topics(num_topics=optimal_k, num_words=8, formatted=False):
                        topic_words = ", ".join([word for word, _ in topic])
                        f_out.write(f"  🔹 Topic {i + 1}: {topic_words}\n")
                        keywords.append(topic_words)

                    summary_rows.append({
                        "product": product_name,
                        "mode": mode_name,
                        "rating": rating,
                        "optimal_k": optimal_k,
                        "coherence": round(best_score, 4),
                        "num_reviews": len(texts),
                        "keywords": " | ".join(keywords)
                    })

        pd.DataFrame(summary_rows).to_csv(f"{output_dir}/topic_summary.csv", index=False)


# from gensim.models import LdaModel, CoherenceModel
# from nltk.util import ngrams
# import matplotlib.pyplot as plt
# import os
#
# nlp = spacy.load("en_core_web_sm")
#
# # 사용자 정의 도메인 스톱워드
# CUSTOM_STOPWORDS = {
#     "product", "skin", "oil", "face", "anua", "heartleaf", "cleansing", "cleanser", "toner", "use"
# }
#
# def get_tokens(text):
#     if pd.isnull(text): return []
#     doc = nlp(text.lower())
#     return [t.lemma_ for t in doc if t.is_alpha and not t.is_stop and t.lemma_ not in CUSTOM_STOPWORDS and len(t) > 2]
#
# def get_bigrams(tokens):
#     return ['_'.join(bg) for bg in ngrams(tokens, 2)]
#
# def get_adj_noun(text):
#     if pd.isnull(text): return []
#     doc = nlp(text.lower())
#     return [token.lemma_ + '_' + doc[i + 1].lemma_
#             for i, token in enumerate(doc[:-1])
#             if token.pos_ == "ADJ" and doc[i + 1].pos_ == "NOUN"]
#
# preprocessing_modes = {
#     "unigram": lambda text: get_tokens(text),
#     "bigram": lambda text: get_bigrams(get_tokens(text)),
#     "unibi_mix": lambda text: get_tokens(text) + get_bigrams(get_tokens(text)),
#     "adj_noun": lambda text: get_adj_noun(text)
# }
#
# def find_optimal_k(texts, dictionary, corpus, mode, rating, out_dir, min_k=2, max_k=6):
#     scores = []
#     for k in range(min_k, max_k + 1):
#         lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, passes=15, random_state=42)
#         cm = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence='c_v')
#         scores.append((k, cm.get_coherence()))
#
#     # 저장
#     plt.figure()
#     ks, cvs = zip(*scores)
#     plt.plot(ks, cvs, marker='o')
#     plt.title(f"Coherence Score by Topic Count\n({mode} | rating {rating})")
#     plt.xlabel("Num Topics")
#     plt.ylabel("Coherence Score")
#     plt.savefig(f"{out_dir}/plot_{mode}_rating{rating}.png")
#     plt.close()
#
#     best_k = max(scores, key=lambda x: x[1])[0]
#     best_score = max(scores, key=lambda x: x[1])[1]
#     return best_k, best_score
#
# # === 진입점 ===
# if __name__ == "__main__":
#     input_files = [
#         ("anua_cleansing_oil_reviews.csv", "cleansing_oil"),
#         ("anua_toner_reviews.csv", "toner")
#     ]
#
#     for file_path, product_name in input_files:
#         df = pd.read_csv(file_path)
#         output_dir = f"outputs_{product_name}"
#         os.makedirs(output_dir, exist_ok=True)
#
#         summary_rows = []
#
#         for mode_name, preprocess_func in preprocessing_modes.items():
#             with open(f"{output_dir}/topics_{mode_name}.txt", "w", encoding="utf-8") as f_out:
#                 f_out.write(f"==== {product_name.upper()} | {mode_name.upper()} ====\n")
#
#                 for rating in range(1, 6):
#                     subset = df[df["rating"] == rating]["Review Content"].dropna()
#                     texts = subset.apply(preprocess_func).tolist()
#                     texts = [t for t in texts if len(t) >= 3]
#
#                     if not texts:
#                         f_out.write(f"\n⭐ Rating {rating}: No valid reviews\n")
#                         continue
#
#                     dictionary = corpora.Dictionary(texts)
#                     corpus = [dictionary.doc2bow(text) for text in texts]
#
#                     optimal_k, best_score = find_optimal_k(texts, dictionary, corpus, mode_name, rating, output_dir)
#                     f_out.write(f"\n⭐ Rating {rating} | Optimal K = {optimal_k} | Coherence = {best_score:.4f}\n")
#
#                     model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=optimal_k, passes=15, random_state=42)
#                     keywords = []
#                     for i, topic in model.show_topics(num_topics=optimal_k, num_words=8, formatted=False):
#                         topic_words = ", ".join([word for word, _ in topic])
#                         f_out.write(f"  🔹 Topic {i + 1}: {topic_words}\n")
#                         keywords.append(topic_words)
#
#                     summary_rows.append({
#                         "product": product_name,
#                         "mode": mode_name,
#                         "rating": rating,
#                         "optimal_k": optimal_k,
#                         "coherence": round(best_score, 4),
#                         "num_reviews": len(texts),
#                         "keywords": " | ".join(keywords)
#                     })
#
#         pd.DataFrame(summary_rows).to_csv(f"{output_dir}/topic_summary.csv", index=False)
#
#
#



# import pandas as pd
# import spacy
# from gensim import corpora
# from gensim.models import LdaModel, CoherenceModel
# from nltk.util import ngrams
# import matplotlib.pyplot as plt
# import os
#
# nlp = spacy.load("en_core_web_sm")
#
# # 사용자 정의 도메인 스톱워드 추가
# CUSTOM_STOPWORDS = {
#     "product", "skin", "oil", "face", "anua", "heartleaf", "cleansing", "cleanser", "toner", "use"
# }
#
# # 전처리 함수들
# def get_tokens(text):
#     if pd.isnull(text): return []
#     doc = nlp(text.lower())
#     return [t.lemma_ for t in doc if t.is_alpha and not t.is_stop and t.lemma_ not in CUSTOM_STOPWORDS and len(t) > 2]
#
# def get_bigrams(tokens):
#     return ['_'.join(bg) for bg in ngrams(tokens, 2)]
#
# def get_adj_noun(text):
#     if pd.isnull(text): return []
#     doc = nlp(text.lower())
#     return [token.lemma_ + '_' + doc[i + 1].lemma_
#             for i, token in enumerate(doc[:-1])
#             if token.pos_ == "ADJ" and doc[i + 1].pos_ == "NOUN"]
#
# preprocessing_modes = {
#     "unigram": lambda text: get_tokens(text),
#     "bigram": lambda text: get_bigrams(get_tokens(text)),
#     "unibi_mix": lambda text: get_tokens(text) + get_bigrams(get_tokens(text)),
#     "adj_noun": lambda text: get_adj_noun(text)
# }
#
# # 최적 k 탐색 함수 + plot 저장
# def find_optimal_k(texts, dictionary, corpus, mode, rating, min_k=2, max_k=6):
#     scores = []
#     for k in range(min_k, max_k + 1):
#         lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, passes=15, random_state=42)
#         cm = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence='c_v')
#         scores.append((k, cm.get_coherence()))
#
#     # plot 저장
#     plt.figure()
#     ks, cvs = zip(*scores)
#     plt.plot(ks, cvs, marker='o')
#     plt.title(f"Coherence Score by Topic Count\n({mode} | rating {rating})")
#     plt.xlabel("Num Topics")
#     plt.ylabel("Coherence Score")
#     plt.savefig(f"outputs/plot_{mode}_rating{rating}.png")
#     plt.close()
#
#     # 최고값 반환
#     best_k = max(scores, key=lambda x: x[1])[0]
#     best_score = max(scores, key=lambda x: x[1])[1]
#     return best_k, best_score
#
# # === 진입점 ===
# if __name__ == "__main__":
#     df = pd.read_csv("anua_cleansing_oil_reviews.csv")
#
#     # 결과 폴더 생성
#     os.makedirs("outputs", exist_ok=True)
#
#     summary_rows = []
#
#     for mode_name, preprocess_func in preprocessing_modes.items():
#         with open(f"outputs/topics_{mode_name}.txt", "w", encoding="utf-8") as f_out:
#             f_out.write(f"==== {mode_name.upper()} ====\n")
#
#             for rating in range(1, 6):
#                 subset = df[df["rating"] == rating]["Review Content"].dropna()
#                 texts = subset.apply(preprocess_func).tolist()
#                 texts = [t for t in texts if len(t) >= 3]
#
#                 if not texts:
#                     f_out.write(f"\n⭐ Rating {rating}: No valid reviews\n")
#                     continue
#
#                 dictionary = corpora.Dictionary(texts)
#                 corpus = [dictionary.doc2bow(text) for text in texts]
#
#                 optimal_k, best_score = find_optimal_k(texts, dictionary, corpus, mode_name, rating)
#                 f_out.write(f"\n⭐ Rating {rating} | Optimal K = {optimal_k} | Coherence = {best_score:.4f}\n")
#
#                 model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=optimal_k, passes=15, random_state=42)
#                 keywords = []
#                 for i, topic in model.show_topics(num_topics=optimal_k, num_words=8, formatted=False):
#                     topic_words = ", ".join([word for word, _ in topic])
#                     f_out.write(f"  🔹 Topic {i + 1}: {topic_words}\n")
#                     keywords.append(topic_words)
#
#                 summary_rows.append({
#                     "mode": mode_name,
#                     "rating": rating,
#                     "optimal_k": optimal_k,
#                     "coherence": round(best_score, 4),
#                     "num_reviews": len(texts),
#                     "keywords": " | ".join(keywords)
#                 })
#
#     # 요약 CSV 저장
#     pd.DataFrame(summary_rows).to_csv("outputs/topic_summary.csv", index=False)
#




# import pandas as pd
# import spacy
# from gensim import corpora
# from gensim.models import LdaModel, CoherenceModel
# from nltk.util import ngrams
# import matplotlib.pyplot as plt
# import os
#
# nlp = spacy.load("en_core_web_sm")
#
# # 전처리 함수 정의
# def get_tokens(text):
#     if pd.isnull(text):
#         return []
#     doc = nlp(text.lower())
#     return [t.lemma_ for t in doc if t.is_alpha and not t.is_stop and len(t) > 2]
#
# def get_bigrams(tokens):
#     return ['_'.join(bg) for bg in ngrams(tokens, 2)]
#
# def get_adj_noun(text):
#     if pd.isnull(text):
#         return []
#     doc = nlp(text.lower())
#     return [token.lemma_ + '_' + doc[i + 1].lemma_
#             for i, token in enumerate(doc[:-1])
#             if token.pos_ == "ADJ" and doc[i + 1].pos_ == "NOUN"]
#
# preprocessing_modes = {
#     "unigram": lambda text: get_tokens(text),
#     "bigram": lambda text: get_bigrams(get_tokens(text)),
#     "unibi_mix": lambda text: get_tokens(text) + get_bigrams(get_tokens(text)),
#     "adj_noun": lambda text: get_adj_noun(text)
# }
#
# # 최적 K 탐색 및 그래프 저장
# def find_optimal_k(texts, dictionary, corpus, mode, rating, product_name, min_k=2, max_k=6):
#     scores = []
#     for k in range(min_k, max_k + 1):
#         lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, passes=15, random_state=42)
#         cm = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence='c_v')
#         scores.append((k, cm.get_coherence()))
#
#     # 그래프 저장
#     plt.figure()
#     ks, cvs = zip(*scores)
#     plt.plot(ks, cvs, marker='o')
#     plt.title(f"Coherence Score by Topic Count\n({product_name} | {mode} | rating {rating})")
#     plt.xlabel("Num Topics")
#     plt.ylabel("Coherence Score")
#
#     out_dir = f"outputs_{product_name}"
#     os.makedirs(out_dir, exist_ok=True)
#     plt.savefig(f"{out_dir}/plot_{mode}_rating{rating}.png")
#     plt.close()
#
#     best_k = max(scores, key=lambda x: x[1])[0]
#     best_score = max(scores, key=lambda x: x[1])[1]
#     return best_k, best_score
#
# # === 진입점 ===
# if __name__ == "__main__":
#     review_files = {
#         "cleansing_oil": "anua_cleansing_oil_reviews.csv",
#         "toner": "anua_toner_reviews.csv"
#     }
#
#     for product_name, file_path in review_files.items():
#         df = pd.read_csv(file_path)
#         os.makedirs(f"outputs_{product_name}", exist_ok=True)
#         summary_rows = []
#
#         for mode_name, preprocess_func in preprocessing_modes.items():
#             with open(f"outputs_{product_name}/topics_{mode_name}.txt", "w", encoding="utf-8") as f_out:
#                 f_out.write(f"==== {mode_name.upper()} ====\n")
#
#                 for rating in range(1, 6):
#                     subset = df[df["rating"] == rating]["Review Content"].dropna()
#                     texts = subset.apply(preprocess_func).tolist()
#                     texts = [t for t in texts if len(t) >= 3]
#
#                     if not texts:
#                         f_out.write(f"\n⭐ Rating {rating}: No valid reviews\n")
#                         continue
#
#                     dictionary = corpora.Dictionary(texts)
#                     corpus = [dictionary.doc2bow(text) for text in texts]
#
#                     optimal_k, best_score = find_optimal_k(texts, dictionary, corpus, mode_name, rating, product_name)
#
#                     model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=optimal_k, passes=15, random_state=42)
#
#                     f_out.write(f"\n⭐ Rating {rating} | Optimal K = {optimal_k} | Coherence = {best_score:.4f}\n")
#
#                     topic_keywords = []
#                     for i, topic in model.print_topics(num_words=5):
#                         f_out.write(f"  🔹 Topic {i + 1}: {topic}\n")
#                         words = topic.split('+')
#                         words = [w.split('*')[1].replace('"', '').strip() for w in words]
#                         topic_keywords.append(", ".join(words))
#
#                     summary_rows.append({
#                         "product": product_name,
#                         "mode": mode_name,
#                         "rating": rating,
#                         "optimal_k": optimal_k,
#                         "coherence": round(best_score, 4),
#                         "num_reviews": len(texts),
#                         "keywords": " | ".join(topic_keywords)
#                     })
#
#         pd.DataFrame(summary_rows).to_csv(f"outputs_{product_name}/topic_summary.csv", index=False)






# import pandas as pd
# import spacy
# from gensim import corpora
# from gensim.models import LdaModel, CoherenceModel
# from nltk.util import ngrams
#
#
# # === 전처리 함수들 정의 === #
# def get_tokens(text):
#     if pd.isnull(text): return []
#     doc = nlp(text.lower())
#     return [t.lemma_ for t in doc if t.is_alpha and not t.is_stop and len(t) > 2]
#
# def get_bigrams(tokens):
#     return ['_'.join(bg) for bg in ngrams(tokens, 2)]
#
# def get_adj_noun(text):
#     if pd.isnull(text): return []
#     doc = nlp(text.lower())
#     return [token.lemma_ + '_' + doc[i + 1].lemma_
#             for i, token in enumerate(doc[:-1])
#             if token.pos_ == "ADJ" and doc[i + 1].pos_ == "NOUN"]
#
# preprocessing_modes = {
#     "unigram": lambda text: get_tokens(text),
#     "bigram": lambda text: get_bigrams(get_tokens(text)),
#     "unibi_mix": lambda text: get_tokens(text) + get_bigrams(get_tokens(text)),
#     "adj_noun": lambda text: get_adj_noun(text)
# }
#
#
# # === 최적 토픽 수 계산 === #
# def find_optimal_k(texts, dictionary, corpus, min_k=2, max_k=6):
#     best_k, best_score = min_k, -1
#     for k in range(min_k, max_k + 1):
#         lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, passes=15, random_state=42)
#         cm = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence='c_v')
#         score = cm.get_coherence()
#         if score > best_score:
#             best_k, best_score = k, score
#     return best_k
#
#
# # === 진입점 === #
# if __name__ == "__main__":
#     # spaCy 모델 로드
#     nlp = spacy.load("en_core_web_sm")
#
#     # 리뷰 데이터 불러오기
#     df = pd.read_csv("anua_cleansing_oil_reviews.csv")
#
#     # 분석 실행
#     for mode_name, preprocess_func in preprocessing_modes.items():
#         print(f"\n\n========================\n🔎 Preprocessing: {mode_name}\n========================")
#
#         for rating in range(1, 6):
#             print(f"\n⭐ Rating: {rating}")
#             subset = df[df["rating"] == rating]["Review Content"].dropna()
#             texts = subset.apply(preprocess_func).tolist()
#             texts = [t for t in texts if len(t) >= 3]
#
#             if not texts:
#                 print("No valid reviews.")
#                 continue
#
#             dictionary = corpora.Dictionary(texts)
#             corpus = [dictionary.doc2bow(text) for text in texts]
#
#             optimal_k = find_optimal_k(texts, dictionary, corpus)
#             print(f"→ Optimal number of topics: {optimal_k}")
#
#             model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=optimal_k, passes=15, random_state=42)
#             for i, topic in model.print_topics(num_words=8):
#                 print(f"🔹 Topic {i+1}: {topic}")
#
#




# import pandas as pd
# import nltk
# # nltk.download("wordnet")
# # nltk.download("omw-1.4")
# # nltk.download("vader_lexicon")
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
#
#
#
# #df = pd.read_csv("anua_cleansing_oil_reviews.csv")
#
# # 감정 분석기 초기화
# sia = SentimentIntensityAnalyzer()
#
# # 키워드 예시: acne 포함 리뷰
# acne_reviews = df[df["Review Content"].str.contains("acne", case=False, na=False)].copy()
#
# # 감정 점수 계산
# acne_reviews["sentiment_score"] = acne_reviews["Review Content"].apply(lambda x: sia.polarity_scores(x)["compound"])
# acne_reviews["sentiment"] = acne_reviews["sentiment_score"].apply(
#     lambda x: "positive" if x > 0.05 else ("negative" if x < -0.05 else "neutral")
# )
#
# # 결과 요약
# print(acne_reviews["sentiment"].value_counts())
