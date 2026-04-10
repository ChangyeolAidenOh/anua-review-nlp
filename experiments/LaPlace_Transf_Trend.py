"""""""""
Laplace transformation in LLM detecting feature trends.
- Raw text => Complex pattern => Categorized trend

Step 1 Preparation (Text -> Discrete Signal)
     1. Tokenization : Separate the Text data(Reveiws) in token
     2. Embedding : One-Hot coding = Frequency response
                  : Word2Vec, GloVe = Including meaning
                  : Sentence-BERT = Embodding based on sentence
     3. Feature Selection : Select specific dimension(Subject scores)
                          : PCA diminishing sequence into 1 dimension (Simplified process with discrete signal)

Step 2 Continuous from the given discrete signal
     1. Set the time-criteria : Ordering reviews (t = 1, 2, 3,...)
                              : Ddate of each reviews (When it was written: Real Numbers "t")
     2. Spline Borgan : Spline interpolate
                      : Gaussian Smoothing
"""""""""

import numpy as np
from scipy.interpolate import interp1d

# === 1. Transform text data to Scale values ===
tokens = ["good", "bad", "excellent", "poor"]
values = np.array([0.8, -0.5, 1.0, -0.7])  ## Example points_Temp
t = np.arange(len(values))

# === 2. Successive data set into interpolization ===
f_interp = interp1d(t, values, kind='cubic')
t_cont = np.linspace(0, len(values)-1, 100)

f_cont = f_interp(t_cont)

# === 3. Numerical transform(Laplace) ===
def laplace_transform(f, t, s):
    return np.trapz(f * np.exp(-s * t), t)

s_vals = np.linspace(0.1, 1.0, 50)
F_vals = [laplace_transform(f_cont, t_cont, s) for s in s_vals]

# === 4. Filtering ===
# In this case, there remains the only low-level.
F_filtered = np.where(s_vals < 0.5, F_vals, 0)

# === 5. Inverse transformation ===
# In real, applied numerical transformation


"""""""""
Realistic Scafold
"""""""""
from __future__ import annotations
import re
# import math
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter

### Optional Section of example emotion scale without VADER, based on sign-lexicon ###
# --- (옵션) 간단 감정 점수기: VADER 없을 때 쓸 수 있는 sign-lexicon 기반 ---
_POS = {"great","good","excellent","amazing","love","nice","satisfied","happy","recommend","perfect","awesome"}
_NEG = {"bad","terrible","awful","poor","hate","disappointed","broken","refund","worst","noisy","leak","smell"}

def simple_sentiment_score(doc: str) -> float:
    tokens = re.findall(r"[a-z']+", doc.lower())
    pos = sum(t in _POS for t in tokens)
    neg = sum(t in _NEG for t in tokens)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


########################################################################################################################
# Temporary parts, to try to adapt more simple segmentation, rather than just using sentiment score.#
# It doesn't need to be used to calculate the final LaPlace Transformation values.#

"""""""""
def simple_sentiment_score_2(doc: str) -> float:
    tokens_v2 = re.findall(r"[a-z']+", doc.lower())
    pos_v2 = sum(t in _POS for t in tokens_v2)
    neg_v2 = sum(t in _NEG for t in tokens_v2)
    if pos_v2 + neg_v2 == 0:
        return 0.0
    return (pos_v2 - neg_v2) / (pos_v2 + neg_v2)
"""""""""

### Successive gaussian kernel regression process##
def gaussian_kernel_regressiont(t: np.ndarray, x: np.ndarray, t_query: np.ndarray, bw: float) -> np.ndarray:
    d2_v2 = cdist(t_query[:, None], t[:, None], metric='sqeuclidean')
    k_v2 = np.exp(-bw * d2_v2)
    wsum_v2 = k_v2.sum(axis=1, keepdims=True) + 1e-12
    return (k_v2 @ x.reshape(-1, 1) / wsum_v2).ravel()

"""""""""
The previous step would diminish the possibilities to make them easier than before. Then, the initial work build unprocessed dataset without a kinds of
perspective regression optas. Secondary, the dims could be arranged by the reshape functions by returning pos_v2 and neg_v2.
"""""""""

########################################################################################################################


# --- 커널 회귀 (가우시안)로 이산 신호를 연속화 ---
def gaussian_kernel_regression(t: np.ndarray, x: np.ndarray, t_query: np.ndarray, bw: float) -> np.ndarray:
    # t, t_query: 1d, x: 1d
    # 가우시안 커널: K = exp(-(Δt)^2 / (2 bw^2))
    d2 = cdist(t_query[:,None], t[:,None], metric='sqeuclidean')  # (Tq, T)
    K = np.exp(-d2 / (2.0 * bw * bw))
    # 정규화 가중 평균
    wsum = K.sum(axis=1, keepdims=True) + 1e-12
    return (K @ x.reshape(-1,1) / wsum).ravel()

# --- 1차 저역통과 ODE 필터 (라플라스 영역 H(s)=1/(1+sτ)와 동치) ---
def lowpass_ode_filter(t: np.ndarray, x: np.ndarray, tau: float, y0: float = 0.0) -> np.ndarray:
    # 비균일 간격도 안전하게 처리
    y = np.empty_like(x)
    y_prev = y0
    t_prev = t[0]
    y[0] = y_prev
    for i in range(1, len(t)):
        dt = max(t[i] - t_prev, 1e-12)
        # explicit Euler: tau dy/dt + y = x  -> dy = (x - y)/tau * dt
        y_new = y_prev + ((x[i-1] - y_prev) / tau) * dt
        # Heun(개선 Euler) 한 번 더(안정성 ↑)
        y_pred = y_new
        slope_pred = (x[i] - y_pred) / tau
        y_new = y_prev + 0.5 * ((x[i-1] - y_prev) / tau + slope_pred) * dt
        y[i] = y_new
        y_prev, t_prev = y_new, t[i]
    return y

# --- n차 저역통과: 1차 필터를 연속으로 n번 적용 => H(s) = (1/(1+sτ))^n ---
def lowpass_nth_order(t: np.ndarray, x: np.ndarray, tau: float, order: int = 2) -> np.ndarray:
    y = x.copy()
    for _ in range(order):
        y = lowpass_ode_filter(t, y, tau=tau, y0=y[0])
    return y

# --- 라플라스 변환(진단용): 실수축 s>0에서의 수치적 적분 근사 ---
def laplace_transform_numeric(t: np.ndarray, f: np.ndarray, s_vals: np.ndarray) -> np.ndarray:
    # 단순 사다리꼴 적분
    # F(s) ~ Σ f(t_i) e^{-s t_i} Δt
    dt = np.diff(t, prepend=t[0])
    F = []
    for s in s_vals:
        F.append(np.sum(f * np.exp(-s * t) * dt))
    return np.asarray(F) 

# --- 세그멘테이션: 기울기 기반 카테고리화 ---
def segment_and_label(t: np.ndarray, y: np.ndarray, win: int = 21, poly: int = 2,
                      slope_thr: float = 0.01) -> List[Tuple[float,float,str]]:
    # Savitzky-Golay로 미분 추정
    if win >= len(y):  # 너무 짧으면 윈도우 조정
        win = max(5, len(y)//2*2 - 1)
    y_smooth = savgol_filter(y, window_length=win, polyorder=poly, mode='interp')
    dt = np.gradient(t)
    dy = np.gradient(y_smooth) / (dt + 1e-12)
    # 구간 라벨링: 상승 / 하락 / 안정
    labels = np.where(dy > slope_thr, 1, np.where(dy < -slope_thr, -1, 0))
    # 구간 묶기
    segments = []
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            seg_label = {1:"UP", -1:"DOWN", 0:"FLAT"}[int(labels[i-1])]
            segments.append((t[start], t[i-1], seg_label))
            start = i
    seg_label = {1:"UP", -1:"DOWN", 0:"FLAT"}[int(labels[-1])]
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
    topic_to_signal: str = "dominant"  # or "entropy", "pc1"
    signal_mix: Dict[str, float] = None  # {"sentiment":0.5, "topic":0.5}

    # 연속화
    bandwidth: float = 5.0   # 가우시안 커널 폭(리뷰 인덱스 기준)

    # 라플라스 관점 필터
    tau: float = 10.0
    order: int = 2  # n차 저역통과

    # 라벨링
    slope_thr: float = 0.002
    sg_win: int = 41
    sg_poly: int = 2

    # 라플라스 변환(옵션)
    s_min: float = 0.05
    s_max: float = 1.0
    s_points: int = 50

    def __post_init__(self):
        if self.signal_mix is None:
            self.signal_mix = {"sentiment":0.5, "topic":0.5}

class LaplaceTextTrend:
    def __init__(self, cfg: LaplaceTextTrendConfig):
        self.cfg = cfg
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.nmf: Optional[NMF] = None
        self.scaler = StandardScaler(with_mean=True, with_std=True)

    # 1) 텍스트 → 이산 스칼라 신호
    def build_discrete_signal(self, docs: List[str]) -> Tuple[np.ndarray, np.ndarray, Dict]:
        N = len(docs)
        t = np.arange(N, dtype=float)

        parts = []
        meta = {}

        if self.cfg.use_sentiment:
            sents = np.array([simple_sentiment_score(d) for d in docs], dtype=float)
            parts.append(("sentiment", sents))
            meta["sentiment_range"] = (sents.min(), sents.max())


        if self.cfg.use_tfidf_topic:
            self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2)
            X = self.vectorizer.fit_transform(docs)  # (N, V)
            k = self.cfg.topic_components
            self.nmf = NMF(n_components=k, init="nndsvda", random_state=42, max_iter=400)
            W = self.nmf.fit_transform(X)  # (N, k) 문서×토픽 강도
            # 토픽을 스칼라로 줄이는 방식
            if self.cfg.topic_to_signal == "dominant":
                topic_signal = W.argmax(axis=1) / max(1, k-1)  # 0~1 정규화
            elif self.cfg.topic_to_signal == "entropy":
                p = W / (W.sum(axis=1, keepdims=True) + 1e-12)
                ent = -(p*np.log(p+1e-12)).sum(axis=1)
                topic_signal = (ent - ent.min()) / (ent.ptp() + 1e-12)
            else:  # "pc1": 주성분 1축으로 투영
                topic_signal = PCA(n_components=1, random_state=42).fit_transform(W).ravel()
                topic_signal = (topic_signal - topic_signal.mean())/ (topic_signal.std()+1e-12)
            parts.append(("topic", topic_signal))
            meta["topic_mode"] = self.cfg.topic_to_signal

        # 혼합
        x = np.zeros(N, dtype=float)
        weight_sum = 0.0
        for name, arr in parts:
            w = self.cfg.signal_mix.get(name, 0.0)
            if w == 0:
                continue
            arr_std = self.scaler.fit_transform(arr.reshape(-1,1)).ravel()
            x += w * arr_std
            weight_sum += w
        if weight_sum == 0:
            raise ValueError("No signal components selected or weights are zero.")
        x /= weight_sum
        return t, x, meta

    # 2) 이산 → 연속
    def discretized_to_continuous(self, t: np.ndarray, x: np.ndarray, upsample: int = 10) -> Tuple[np.ndarray,np.ndarray]:
        t_cont = np.linspace(t.min(), t.max(), num=max(upsample*len(t), 200))
        f_cont = gaussian_kernel_regression(t, x, t_cont, bw=self.cfg.bandwidth)
        return t_cont, f_cont

    # 3) (라플라스 관점) 저역통과 필터
    def lowpass_trend(self, t: np.ndarray, f: np.ndarray) -> np.ndarray:
        return lowpass_nth_order(t, f, tau=self.cfg.tau, order=self.cfg.order)

    # 4) 세그멘테이션 & 라벨
    def to_segments(self, t: np.ndarray, y: np.ndarray) -> List[Tuple[float,float,str]]:
        return segment_and_label(t, y, win=self.cfg.sg_win, poly=self.cfg.sg_poly, slope_thr=self.cfg.slope_thr)

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
        "Great battery life and amazing build quality. Highly recommend.",
        "Terrible packaging, item arrived broken, very disappointed.",
        "Good sound but the connection is sometimes noisy.",
        "Excellent value for the price. Perfect for my daily use.",
        "Bad smell from the material, refund requested."
    ]
    cfg = LaplaceTextTrendConfig(
        use_sentiment=True,
        use_tfidf_topic=True,
        topic_components=6,
        topic_to_signal="entropy",   # 혹은 "dominant" / "pc1"
        signal_mix={"sentiment":0.6, "topic":0.4},
        bandwidth=3.5,
        tau=8.0,
        order=2,
        slope_thr=0.003,
        sg_win=31,
        sg_poly=2,
        s_min=0.05, s_max=1.2, s_points=60
    )
    pipe = LaplaceTextTrend(cfg)
    out = pipe.run(docs)

    # 결과 해석 포인트:
    # - out["y_trend"]: 저역통과(라플라스 관점)로 추출된 부드러운 추세
    # - out["segments"]: 각 구간 라벨(UP/DOWN/FLAT)
    # - out["laplace_F"]: f_cont의 라플라스 변환값(진단용; s↑일수록 고주파 민감)
    print("Segments:", out["segments"])

