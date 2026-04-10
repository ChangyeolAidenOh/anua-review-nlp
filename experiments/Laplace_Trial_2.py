from __future__ import annotations
import re
import math
import warnings
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter

# ========= Sentiment (VADER 우선, 미설치 시 simple lexicon으로 폴백) =========
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_AVAILABLE = True
    _vader = SentimentIntensityAnalyzer()
except Exception:
    _VADER_AVAILABLE = False
    _vader = None

# --- (폴백) 간단 감정 점수기: VADER 없을 때 사용 ---
_POS = {"great","good","excellent","amazing","love","nice","satisfied","happy",
        "recommend","perfect","awesome"}
_NEG = {"bad","terrible","awful","poor","hate","disappointed","broken","refund",
        "worst","noisy","leak","smell"}

def simple_sentiment_score(doc: str) -> float:
    tokens = re.findall(r"[a-z']+", doc.lower())
    pos = sum(t in _POS for t in tokens)
    neg = sum(t in _NEG for t in tokens)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)

def vader_sentiment_score(doc: str) -> float:
    if not _VADER_AVAILABLE:
        return simple_sentiment_score(doc)
    return _vader.polarity_scores(doc)['compound']  # [-1, 1]

# ========= 커널 회귀 (가우시안)로 이산 신호를 연속화 =========
def gaussian_kernel_regression(t: np.ndarray, x: np.ndarray,
                               t_query: np.ndarray, bw: float) -> np.ndarray:
    """
    t, t_query: (T,), x: (T,)
    f(t_query) = Σ K(tq, ti) * x_i / Σ K(tq, ti),  K = exp(-Δt^2/(2 bw^2))
    """
    d2 = cdist(t_query[:, None], t[:, None], metric='sqeuclidean')  # (Tq, T)
    K = np.exp(-d2 / (2.0 * bw * bw))
    wsum = K.sum(axis=1, keepdims=True) + 1e-12
    return (K @ x.reshape(-1, 1) / wsum).ravel()

# ========= 1차 저역통과 ODE 필터 (라플라스 H(s)=1/(1+sτ)와 동치) =========
def lowpass_ode_filter(t: np.ndarray, x: np.ndarray, tau: float, y0: float = 0.0) -> np.ndarray:
    """
    tau dy/dt + y = x  (Heun predictor-corrector)
    비균일 간격 t 지원
    """
    y = np.empty_like(x)
    y_prev = y0
    t_prev = t[0]
    y[0] = y_prev
    for i in range(1, len(t)):
        dt = max(t[i] - t_prev, 1e-12)
        # predictor (explicit Euler)
        y_pred = y_prev + ((x[i-1] - y_prev) / tau) * dt
        # corrector (Heun)
        slope_pred = (x[i] - y_pred) / tau
        y_new = y_prev + 0.5 * ((x[i-1] - y_prev) / tau + slope_pred) * dt
        y[i] = y_new
        y_prev, t_prev = y_new, t[i]
    return y

# ========= n차 저역통과: H(s) = (1/(1+sτ))^n =========
def lowpass_nth_order(t: np.ndarray, x: np.ndarray, tau: float, order: int = 2) -> np.ndarray:
    y = x.copy()
    for _ in range(order):
        y = lowpass_ode_filter(t, y, tau=tau, y0=y[0])
    return y

# ========= 라플라스 변환(진단용) =========
def laplace_transform_numeric(t: np.ndarray, f: np.ndarray, s_vals: np.ndarray) -> np.ndarray:
    """
    F(s) ~ Σ f(t_i) e^{-s t_i} Δt  (사다리꼴 대신 단순 가중합 근사)
    """
    dt = np.diff(t, prepend=t[0])
    F = []
    for s in s_vals:
        F.append(np.sum(f * np.exp(-s * t) * dt))
    return np.asarray(F)

# ========= 세그멘테이션: 기울기 기반 카테고리화 =========
def segment_and_label(t: np.ndarray, y: np.ndarray, win: int = 21, poly: int = 2,
                      slope_thr: float = 0.01) -> List[Tuple[float, float, str]]:
    """
    Savitzky-Golay로 미분 추정 → 기울기 부호로 UP/DOWN/FLAT 라벨링
    """
    if win >= len(y):  # 너무 짧으면 윈도우 조정(홀수 필요)
        win = max(5, len(y)//2*2 - 1)
    if win < 3:  # 데이터가 매우 짧은 경우
        return [(t[0], t[-1], "FLAT")]
    y_smooth = savgol_filter(y, window_length=win, polyorder=poly, mode='interp')
    dt = np.gradient(t)
    dy = np.gradient(y_smooth) / (dt + 1e-12)

    labels = np.where(dy > slope_thr, 1, np.where(dy < -slope_thr, -1, 0))

    segments = []
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            seg_label = {1: "UP", -1: "DOWN", 0: "FLAT"}[int(labels[i-1])]
            segments.append((t[start], t[i-1], seg_label))
            start = i
    seg_label = {1: "UP", -1: "DOWN", 0: "FLAT"}[int(labels[-1])]
    segments.append((t[start], t[-1], seg_label))
    return segments

# =========================
# 메인 파이프라인 (클래스)
# =========================

@dataclass
class LaplaceTextTrendConfig:
    # 신호 구성
    use_sentiment: bool = True
    use_tfidf_topic: bool = True
    topic_components: int = 8
    topic_to_signal: str = "entropy"  # "dominant", "entropy", "pc1"
    signal_mix: Dict[str, float] = None  # {"sentiment":0.6, "topic":0.4}
    # 연속화
    bandwidth: float = 6.0   # 가우시안 커널 폭(리뷰 인덱스 기준, 5~8 권장)
    # 라플라스 관점 필터
    tau: float = 12.0
    order: int = 2  # n차 저역통과(2~3)
    # 라벨링
    slope_thr: float = 0.003
    sg_win: int = 31
    sg_poly: int = 2
    # 라플라스 변환(옵션)
    s_min: float = 0.05
    s_max: float = 1.0
    s_points: int = 60

    def __post_init__(self):
        if self.signal_mix is None:
            self.signal_mix = {"sentiment": 0.6, "topic": 0.4}

class LaplaceTextTrend:
    def __init__(self, cfg: LaplaceTextTrendConfig):
        self.cfg = cfg
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.nmf: Optional[NMF] = None
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self._tfidf_used_cfg: Optional[dict] = None  # 어떤 폴백을 썼는지 기록

    # --- TF-IDF 적응형 폴백: 비어버리면 min_df 완화 → char n-gram 전환 ---
    def _fit_tfidf_with_fallbacks(self, docs: List[str]):
        N = len(docs)
        min_df1 = max(2, int(np.ceil(0.05 * N)))  # 데이터 크기에 따른 동적 min_df
        configs = [
            dict(stop_words="english", max_features=20000, ngram_range=(1, 2), min_df=min_df1),
            dict(stop_words="english", max_features=20000, ngram_range=(1, 2), min_df=1),
            dict(analyzer="char", ngram_range=(3, 5), min_df=1, max_features=30000),
        ]
        last_err = None
        for cfg in configs:
            vec = TfidfVectorizer(**cfg)
            try:
                X = vec.fit_transform(docs)
                if X.shape[1] > 0:
                    if last_err is not None:
                        warnings.warn(f"TF-IDF fallback used: {cfg}", RuntimeWarning)
                    self.vectorizer = vec
                    self._tfidf_used_cfg = cfg
                    return X
            except ValueError as e:
                last_err = e
                continue
        # 모든 시도 실패 → None 반환하여 토픽 비활성화 (파이프라인은 계속 진행)
        warnings.warn(f"TF-IDF produced no features after fallbacks. Last error: {last_err}",
                      RuntimeWarning)
        return None

    # 1) 텍스트 → 이산 스칼라 신호
    def build_discrete_signal(self, docs: List[str]) -> Tuple[np.ndarray, np.ndarray, Dict]:
        N = len(docs)
        if N == 0:
            raise ValueError("Empty docs.")
        t = np.arange(N, dtype=float)

        parts = []
        meta: Dict = {}

        # --- Sentiment (VADER → fallback) ---
        if self.cfg.use_sentiment:
            sents = np.array([vader_sentiment_score(d) for d in docs], dtype=float)
            parts.append(("sentiment", sents))
            meta["sentiment_backend"] = "vader" if _VADER_AVAILABLE else "simple_fallback"
            meta["sentiment_range"] = (float(sents.min()), float(sents.max()))

        # --- Topic signal (TF-IDF → NMF → scalarize) ---
        if self.cfg.use_tfidf_topic:
            X = self._fit_tfidf_with_fallbacks(docs)  # (N, V) 또는 None
            if X is None:
                meta["topic_disabled_reason"] = "tfidf_failed"
            else:
                n_samples, n_features = X.shape
                # NMF 컴포넌트 수: <= min(n_samples-1, n_features-1)
                k_desired = max(2, self.cfg.topic_components)
                k_max = max(2, min(n_samples - 1, n_features - 1))
                if k_max < 2:
                    warnings.warn("Too few features/samples for NMF. Topic signal disabled.",
                                  RuntimeWarning)
                    meta["topic_disabled_reason"] = "too_few_features_or_samples"
                    meta["tfidf_vocab_size"] = int(n_features)
                    meta["tfidf_used_cfg"] = self._tfidf_used_cfg
                else:
                    k = min(k_desired, k_max)
                    self.nmf = NMF(n_components=k, init="nndsvda",
                                   random_state=42, max_iter=600)
                    W = self.nmf.fit_transform(X)  # (N, k) 문서×토픽 강도

                    mode = self.cfg.topic_to_signal
                    if mode == "dominant":
                        topic_signal = W.argmax(axis=1) / max(1, k - 1)  # 0~1 정규화
                    elif mode == "entropy":
                        p = W / (W.sum(axis=1, keepdims=True) + 1e-12)
                        ent = -(p * np.log(p + 1e-12)).sum(axis=1)
                        topic_signal = (ent - ent.min()) / (ent.ptp() + 1e-12)
                    else:  # "pc1"
                        topic_signal = PCA(n_components=1, random_state=42)\
                            .fit_transform(W).ravel()
                        topic_signal = (topic_signal - topic_signal.mean()) / \
                                       (topic_signal.std() + 1e-12)

                    parts.append(("topic", topic_signal))
                    meta["topic_mode"] = mode
                    meta["topic_components_effective"] = int(k)
                    meta["tfidf_vocab_size"] = int(n_features)
                    meta["tfidf_used_cfg"] = self._tfidf_used_cfg

        # --- 혼합(표준화 후 가중 평균) ---
        x = np.zeros(N, dtype=float)
        weight_sum = 0.0
        for name, arr in parts:
            w = float(self.cfg.signal_mix.get(name, 0.0)) if self.cfg.signal_mix else 0.0
            if w == 0:
                continue
            arr_std = StandardScaler(with_mean=True, with_std=True)\
                .fit_transform(arr.reshape(-1, 1)).ravel()
            x += w * arr_std
            weight_sum += w
        if weight_sum == 0:
            raise ValueError("No signal components selected or weights are zero.")
        x /= weight_sum

        return t, x, meta

    # 2) 이산 → 연속
    def discretized_to_continuous(self, t: np.ndarray, x: np.ndarray,
                                  upsample: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        Tq = max(upsample * len(t), 200)
        t_cont = np.linspace(t.min(), t.max(), num=Tq)
        f_cont = gaussian_kernel_regression(t, x, t_cont, bw=self.cfg.bandwidth)
        return t_cont, f_cont

    # 3) (라플라스 관점) 저역통과 필터
    def lowpass_trend(self, t: np.ndarray, f: np.ndarray) -> np.ndarray:
        return lowpass_nth_order(t, f, tau=self.cfg.tau, order=self.cfg.order)

    # 4) 세그멘테이션 & 라벨
    def to_segments(self, t: np.ndarray, y: np.ndarray) -> List[Tuple[float, float, str]]:
        return segment_and_label(t, y, win=self.cfg.sg_win, poly=self.cfg.sg_poly,
                                 slope_thr=self.cfg.slope_thr)

    # 5) (옵션) 라플라스 변환값 계산
    def diagnostic_laplace(self, t: np.ndarray, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s_vals = np.linspace(self.cfg.s_min, self.cfg.s_max, self.cfg.s_points)
        F_vals = laplace_transform_numeric(t, f, s_vals)
        return s_vals, F_vals

    # 통합 실행
    def run(self, docs: List[str]) -> Dict:
        t, x, meta = self.build_discrete_signal(docs)
        t_cont, f_cont = self.discretized_to_continuous(t, x)
        y_trend = self.lowpass_trend(t_cont, f_cont)
        segments = self.to_segments(t_cont, y_trend)
        s_vals, F_vals = self.diagnostic_laplace(t_cont, f_cont)  # 원 신호 기준 진단

        return {
            "t_discrete": t, "x_discrete": x,
            "t_cont": t_cont, "f_cont": f_cont,
            "y_trend": y_trend,
            "segments": segments,
            "laplace_s": s_vals, "laplace_F": F_vals,
            "meta": meta
        }


# =========================
# 사용 예시
# =========================
if __name__ == "__main__":
    docs = [
        "Battery life is amazing and the build quality feels premium.",
        "Terrible packaging; the item arrived broken. Very disappointed.",
        "Sound is good, but the Bluetooth connection drops occasionally.",
        "Excellent value for money. Works perfectly for my daily commute.",
        "Strong plastic smell at first, but it faded after a day."
        
        "I got this cleanser to help my skin retain moisture because my lifelong oily skin has become dry combo skin over the past couple years,\
        and also wanted to reduce sebum filaments and blackheads. I was obsessed with this cleanser from the first wash that I used it,\
        and I used it religiously for 3 weeks morning and night along with their ANUA exfoliating toner and their rice ceramide skim barrier serum,\
        with my normal moisturizer. On day 3 I developed a cyst on my cheek, this has happened few times in my life\
        so I brushed it off and just figured my skin was purging — a good thing to me! Day 5, two more cysts appeared in other areas of my face.\
        That was weird to me, but again assumed skin was purging. Luckily I have an incredible retinol-AHA/BHA combo routine,\
        so that would fairly quickly diminish the cysts and the red marks that they’d leave.\
        By day 8 those initial three cysts were mostly gone… only to wake up on day 9 with THREE NEW CYSTS, all in different places…\
        I was so upset and could not understand what was happening! My skin just doesn’t do that, it never has!\
        On top of the cystic acne I was finding more and more large bumpy whiteheads scattered around my face, which is also abnormal for my skin.\
        That’s when I got a suspicion it was the cleanser, but I was in denial. I loved it. I was using it exactly correct too and everything.\
        At the end of day 14 I got a cyst on the inner end of my eyebrow, and by the afternoon of day 15 I had ANOTHER one in the middle of the same eyebrow!!!\
        I was crying in frustration at this point because these cysts were making my face SWELL! Especially ones that appeared along my jaw or on my cheek.\
        I had wore makeup only twice within this two week period so I was even more confused as to what could’ve been getting trapped underneath to cause all this.\
        Day 19 I noticed I was JUST beginning to develop a cyst on my cheek so I cut off the cleanser right then and there.\
        Went back to my SA cleanser while I look for a different brand oil cleanser and my skin is back to normal now.\
        That last cyst never fully developed after I stopped btw.\
        I have never had an issue with any type of cleanser, or even any product, before and certainly never anything like this.\
        My skin would have definitely been scarred in every area of my face (both cheeks, chin, both sides of jaw, just below forehead) with the frequency, size and rate of this cystic breakout,\
        had it not been for my combo routine I mentioned. Unbelievable!!!\
        I should note that I did not suspend using their exfoliating toner and rice ceramide serum, those have been AMAZING! Get a different oil cleanser, do not buy!!!"
    ]

    cfg = LaplaceTextTrendConfig(
        use_sentiment=True,
        use_tfidf_topic=True,
        topic_components=8,
        topic_to_signal="entropy",           # "dominant" / "entropy" / "pc1"
        signal_mix={"sentiment": 0.6, "topic": 0.4},
        bandwidth=6.0,                       # 5~8 사이 조정
        tau=12.0,                            # 10~20
        order=2,                             # 2~3
        slope_thr=0.003,                     # 0.002~0.004
        sg_win=31, sg_poly=2,
        s_min=0.05, s_max=1.0, s_points=60
    )

    pipe = LaplaceTextTrend(cfg)
    out = pipe.run(docs)

    # 결과 해석 포인트:
    # - out["y_trend"]: 저역통과(라플라스 관점)로 추출된 부드러운 추세
    # - out["segments"]: 각 구간 라벨(UP/DOWN/FLAT)
    # - out["laplace_F"]: f_cont의 라플라스 변환값(진단용; s↑일수록 고주파 민감)
    print("Sentiment backend:", out["meta"].get("sentiment_backend"))
    print("TF-IDF meta:", {k: out["meta"][k] for k in out["meta"] if k.startswith("tfidf") or k.startswith("topic")})
    print("Segments:", out["segments"])
