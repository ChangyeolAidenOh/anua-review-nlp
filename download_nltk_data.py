# import nltk
#
# # 경로 명시 (이게 핵심!!)
# nltk.data.path.append("/Users/aiden/nltk_data")
#
# # 리소스 다운로드
# nltk.download("wordnet", download_dir="/Users/aiden/nltk_data")
# nltk.download("omw-1.4", download_dir="/Users/aiden/nltk_data")
# nltk.download("vader_lexicon", download_dir="/Users/aiden/nltk_data")



import pandas as pd
import spacy
from gensim import corpora
from gensim.models import LdaModel
from nltk.util import ngrams
from pandas.core.common import random_state

# 데이터 불러오기
df = pd.read_csv("anua_cleansing_oil_reviews.csv")
nlp = spacy.load("en_core_web_sm")

# 🔹 전처리 함수들 정의

def preprocess_unigram(text):
    if pd.isnull(text): return []
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if token.is_alpha and not token.is_stop and len(token) > 2]

def preprocess_bigram(text):
    tokens = preprocess_unigram(text)
    return ['_'.join(bg) for bg in ngrams(tokens, 2)]

def preprocess_unibi(text):
    tokens = preprocess_unigram(text)
    bigrams = ['_'.join(bg) for bg in ngrams(tokens, 2)]
    return tokens + bigrams

def preprocess_adj_noun(text):
    if pd.isnull(text): return []
    doc = nlp(text.lower())
    return [
        token.lemma_ + '_' + doc[i + 1].lemma_
        for i, token in enumerate(doc[:-1])
        if token.pos_ == "ADJ" and doc[i + 1].pos_ == "NOUN"
    ]

# 🔹 분석 대상 전처리 목록
modes = {
    "unigram": preprocess_unigram,
    "bigram": preprocess_bigram,
    "unibi_mix": preprocess_unibi,
    "adj_noun": preprocess_adj_noun
}

# 🔹 실행 루프
for mode_name, preprocess_func in modes.items():
    print(f"\n\n=======================\n▶ LDA Topic for: {mode_name}\n=======================")
    subset = df["Review Content"].dropna()
    texts = subset.apply(preprocess_func).tolist()
    texts = [t for t in texts if len(t) >= 3]

    if not texts:
        print("No valid reviews to process.")
        continue

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=3, passes=15, random_state=42)

    for i, topic in lda_model.print_topics(num_words=8):
        print(f"🔹 Topic {i+1}: {topic}")


import pandas as pd
import spacy
from gensim import corpora
from gensim.models import LdaModel

"""
기본 토큰 단위
"""
# # 🔹 1. 데이터 불러오기
# df = pd.read_csv("anua_cleansing_oil_reviews.csv")
#
# # 🔹 2. spaCy 모델 로드 (처음엔 터미널에서: python -m spacy download en_core_web_sm)
# nlp = spacy.load("en_core_web_sm")
#
# # 🔹 3. 전처리 함수 정의
# def preprocess_spacy(text):
#     if pd.isnull(text):
#         return []
#     doc = nlp(text.lower())
#     return [
#         token.lemma_ for token in doc
#         if token.is_alpha and not token.is_stop and len(token) > 2
#     ]
#
# # 🔹 4. 별점별 리뷰 분석
# for rating in range(1, 6):
#     print(f"\n⭐ Rating: {rating}")
#     subset = df[df["rating"] == rating]["Review Content"].dropna()
#     texts = subset.apply(preprocess_spacy).tolist()
#
#     # 너무 짧은 텍스트 제거
#     texts = [t for t in texts if len(t) >= 3]
#
#     if not texts:
#         print("No valid reviews to process.")
#         continue
#
#     # 🔹 5. Dictionary + Corpus 생성
#     dictionary = corpora.Dictionary(texts)
#     corpus = [dictionary.doc2bow(text) for text in texts]
#
#     # 🔹 6. LDA 모델 학습
#     lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=3, passes=15, random_state=42)
#
#     # 🔹 7. 토픽 출력
#     for i, topic in lda_model.print_topics(num_words=8):
#         print(f"🔹 Topic {i+1}: {topic}")




"""
bi-gram
"""
import pandas as pd
import spacy
from gensim import corpora
from gensim.models import LdaModel
from nltk.util import ngrams

# 🔹 1. 데이터 불러오기
df = pd.read_csv("anua_cleansing_oil_reviews.csv")

# 🔹 2. spaCy 모델 로드
nlp = spacy.load("en_core_web_sm")

# 🔹 3. 전처리 함수 정의 (bi-gram 포함)
def preprocess_spacy_bigrams(text):
    if pd.isnull(text):
        return []
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc
        if token.is_alpha and not token.is_stop and len(token) > 2
    ]
    bigrams = ['_'.join(bg) for bg in ngrams(tokens, 2)]
    return bigrams

# 🔹 4. 별점별 리뷰 분석
for rating in range(1, 6):
    print(f"\n⭐ Rating: {rating}")
    subset = df[df["rating"] == rating]["Review Content"].dropna()
    texts = subset.apply(preprocess_spacy_bigrams).tolist()

    # 너무 짧은 텍스트 제거
    texts = [t for t in texts if len(t) >= 3]

    if not texts:
        print("No valid reviews to process.")
        continue

    # 🔹 5. Dictionary + Corpus 생성
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # 🔹 6. LDA 모델 학습
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=3, passes=15, random_state=42)

    # 🔹 7. 토픽 출력
    for i, topic in lda_model.print_topics(num_words=8):
        print(f"🔹 Topic {i+1}: {topic}")


"""
uni&bi 혼합 + POS
"""
import pandas as pd
import spacy
from gensim import corpora
from gensim.models import LdaModel
from nltk.util import ngrams

# Load dataset
df = pd.read_csv("anua_cleansing_oil_reviews.csv")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Preprocessing: unigram + bigram + adj+noun POS pair
def preprocess_combined(text):
    if pd.isnull(text):
        return []
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop and len(token) > 2]
    bigrams = ['_'.join(bg) for bg in ngrams(tokens, 2)]
    adj_noun = [
        token.lemma_ + '_' + doc[i + 1].lemma_
        for i, token in enumerate(doc[:-1])
        if token.pos_ == "ADJ" and doc[i + 1].pos_ == "NOUN"
    ]
    return tokens + bigrams + adj_noun

# Run LDA for each rating level
for rating in range(1, 6):
    print(f"\n⭐ Rating: {rating}")
    subset = df[df["rating"] == rating]["Review Content"].dropna()
    texts = subset.apply(preprocess_combined).tolist()
    texts = [t for t in texts if len(t) >= 3]

    if not texts:
        print("No valid reviews")
        continue

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=3, passes=15, random_state=42)

    for i, topic in lda_model.print_topics(num_words=8):
        print(f"🔹 Topic {i+1}: {topic}")




### New LDA for each rating level
for rating in range(1, 6):
    print(f"\n rating: {rating}")
    subst = df[df["rating"] == rating]["Review Content"].dropna()
    texts = subst.apply(preprocess_combined).tolist()
    texts = [t for t in texts if len(t) >= 3]

    if not texts:
        print("No valid reviews")
        continue

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = LdaModel(corpus = corpus, id2word = dictionary, num_topics = 3, passes = 15, random_state = 42)

    for i, topic in lda_model.print_topics(num_words = 8):
        int_setup = {i + 1}, {topic}
        print(f" Topic {i + 1}: {topic}")

### Secondary Filter based on the previous LDA
for key in int_setup:
    new_subst = df[df["rating"] == rating]["Review Contetn"].dropna()
    new_texts = new_subst.apply(preprocess_combined).tolist()

    if not new_texts:
        print("No valid reviews")
        continue
