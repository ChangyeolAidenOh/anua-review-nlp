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

# Sentiment scoring: use VADER if available, otherwise fall back to simple sign-lexicon
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_AVAILABLE = True
    _vader = SentimentIntensityAnalyzer()
except Exception:
    _VADER_AVAILABLE = False
    _vader = None

# Fallback sentiment scorer (used when VADER is not installed)
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
    return _vader.polarity_scores(doc)['compound']  # range: [-1, 1]

# Gaussian kernel regression: converts discrete signal to continuous
def gaussian_kernel_regression(t: np.ndarray, x: np.ndarray,
                               t_query: np.ndarray, bw: float) -> np.ndarray:
    """
    t, t_query: (T,), x: (T,)
    f(t_query) = sum K(tq, ti) * x_i / sum K(tq, ti),  K = exp(-delta_t^2 / (2 * bw^2))
    """
    d2 = cdist(t_query[:, None], t[:, None], metric='sqeuclidean')  # (Tq, T)
    K = np.exp(-d2 / (2.0 * bw * bw))
    wsum = K.sum(axis=1, keepdims=True) + 1e-12
    return (K @ x.reshape(-1, 1) / wsum).ravel()

# 1st order ODE low-pass filter — equivalent to Laplace H(s) = 1/(1+s*tau)
def lowpass_ode_filter(t: np.ndarray, x: np.ndarray, tau: float, y0: float = 0.0) -> np.ndarray:
    """
    tau dy/dt + y = x  (Heun predictor-corrector)
    Supports non-uniform time intervals
    """
    y = np.empty_like(x)
    y_prev = y0
    t_prev = t[0]
    y[0] = y_prev
    for i in range(1, len(t)):
        dt = max(t[i] - t_prev, 1e-12)
        # Predictor (explicit Euler)
        y_pred = y_prev + ((x[i-1] - y_prev) / tau) * dt
        # Corrector (Heun)
        slope_pred = (x[i] - y_pred) / tau
        y_new = y_prev + 0.5 * ((x[i-1] - y_prev) / tau + slope_pred) * dt
        y[i] = y_new
        y_prev, t_prev = y_new, t[i]
    return y

# n-th order low-pass filter: H(s) = (1/(1+s*tau))^n
def lowpass_nth_order(t: np.ndarray, x: np.ndarray, tau: float, order: int = 2) -> np.ndarray:
    y = x.copy()
    for _ in range(order):
        y = lowpass_ode_filter(t, y, tau=tau, y0=y[0])
    return y

# Laplace transform: numerical approximation (diagnostic use)
def laplace_transform_numeric(t: np.ndarray, f: np.ndarray, s_vals: np.ndarray) -> np.ndarray:
    """
    F(s) ~ sum f(t_i) * e^(-s * t_i) * delta_t  (simple weighted sum approximation)
    """
    dt = np.diff(t, prepend=t[0])
    F = []
    for s in s_vals:
        F.append(np.sum(f * np.exp(-s * t) * dt))
    return np.asarray(F)

# Gradient-based segmentation and labeling
def segment_and_label(t: np.ndarray, y: np.ndarray, win: int = 21, poly: int = 2,
                      slope_thr: float = 0.01) -> List[Tuple[float, float, str]]:
    """
    Estimates derivatives via Savitzky-Golay filter, then labels each segment UP/DOWN/FLAT
    """
    if win >= len(y):  # Adjust window size if too large (must be odd)
        win = max(5, len(y)//2*2 - 1)
    if win < 3:  # Edge case: very short data
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
# Main Pipeline (Class)
# =========================

@dataclass
class LaplaceTextTrendConfig:
    # Signal configuration
    use_sentiment: bool = True
    use_tfidf_topic: bool = True
    topic_components: int = 8
    topic_to_signal: str = "entropy"  # options: "dominant", "entropy", "pc1"
    signal_mix: Dict[str, float] = None  # e.g. {"sentiment": 0.6, "topic": 0.4}

    # Continuity (Gaussian kernel bandwidth, based on review index; recommended: 5~8)
    bandwidth: float = 6.0

    # Laplace low-pass filter
    tau: float = 12.0
    order: int = 2  # n-th order low-pass (recommended: 2~3)

    # Labeling
    slope_thr: float = 0.003
    sg_win: int = 31
    sg_poly: int = 2

    # Optional Laplace transform (diagnostic)
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
        self._tfidf_used_cfg: Optional[dict] = None  # tracks which fallback config was used

    # Adaptive TF-IDF fitting with fallbacks if vocabulary becomes empty
    def _fit_tfidf_with_fallbacks(self, docs: List[str]):
        N = len(docs)
        min_df1 = max(2, int(np.ceil(0.05 * N)))  # dynamic min_df based on dataset size
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
            except Exception as e:
                last_err = e
        warnings.warn("All TF-IDF configs failed. Topic signal will be disabled.", RuntimeWarning)
        return None

    # 1) Text -> Discrete scalar signal
    def build_discrete_signal(self, docs: List[str]) -> Tuple[np.ndarray, np.ndarray, Dict]:
        N = len(docs)
        if N == 0:
            raise ValueError("Empty docs.")
        t = np.arange(N, dtype=float)

        parts = []
        meta: Dict = {}

        # Sentiment signal (VADER with fallback)
        if self.cfg.use_sentiment:
            sents = np.array([vader_sentiment_score(d) for d in docs], dtype=float)
            parts.append(("sentiment", sents))
            meta["sentiment_backend"] = "vader" if _VADER_AVAILABLE else "simple_fallback"
            meta["sentiment_range"] = (float(sents.min()), float(sents.max()))

        # Topic signal (TF-IDF -> NMF -> scalar)
        if self.cfg.use_tfidf_topic:
            X = self._fit_tfidf_with_fallbacks(docs)  # (N, V) or None
            if X is None:
                meta["topic_disabled_reason"] = "tfidf_failed"
            else:
                n_samples, n_features = X.shape
                # Clamp NMF components to valid range
                k_desired = max(2, self.cfg.topic_components)
                k_max = max(2, min(n_samples - 1, n_features - 1))
                if k_max < 2:
                    warnings.warn("Too few features/samples for NMF. Topic signal disabled.", RuntimeWarning)
                    meta["topic_disabled_reason"] = "too_few_features_or_samples"
                    meta["tfidf_vocab_size"] = int(n_features)
                    meta["tfidf_used_cfg"] = self._tfidf_used_cfg
                else:
                    k = min(k_desired, k_max)
                    self.nmf = NMF(n_components=k, init="nndsvda", random_state=42, max_iter=600)
                    W = self.nmf.fit_transform(X)  # (N, k): document x topic weights

                    # Reduce topic matrix to a single scalar signal
                    mode = self.cfg.topic_to_signal
                    if mode == "dominant":
                        topic_signal = W.argmax(axis=1) / max(1, k - 1)  # normalized to [0, 1]
                    elif mode == "entropy":
                        p = W / (W.sum(axis=1, keepdims=True) + 1e-12)
                        ent = -(p * np.log(p + 1e-12)).sum(axis=1)
                        topic_signal = (ent - ent.min()) / (ent.ptp() + 1e-12)
                    else:  # "pc1": project onto first principal component
                        topic_signal = PCA(n_components=1, random_state=42).fit_transform(W).ravel()
                        topic_signal = (topic_signal - topic_signal.mean()) / (topic_signal.std() + 1e-12)

                    parts.append(("topic", topic_signal))
                    meta["topic_mode"] = mode
                    meta["topic_components_effective"] = int(k)
                    meta["tfidf_vocab_size"] = int(n_features)
                    meta["tfidf_used_cfg"] = self._tfidf_used_cfg

        # Weighted mixture of standardized signal components
        x = np.zeros(N, dtype=float)
        weight_sum = 0.0
        for name, arr in parts:
            w = float(self.cfg.signal_mix.get(name, 0.0)) if self.cfg.signal_mix else 0.0
            if w == 0:
                continue
            arr_std = StandardScaler(with_mean=True, with_std=True).fit_transform(arr.reshape(-1, 1)).ravel()
            x += w * arr_std
            weight_sum += w
        if weight_sum == 0:
            raise ValueError("No signal components selected or weights are zero.")
        x /= weight_sum

        return t, x, meta

    # 2) Discrete -> Continuous
    def discretized_to_continuous(self, t: np.ndarray, x: np.ndarray,
                                  upsample: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        Tq = max(upsample * len(t), 200)
        t_cont = np.linspace(t.min(), t.max(), num=Tq)
        f_cont = gaussian_kernel_regression(t, x, t_cont, bw=self.cfg.bandwidth)
        return t_cont, f_cont

    # 3) Laplace-perspective low-pass filter
    def lowpass_trend(self, t: np.ndarray, f: np.ndarray) -> np.ndarray:
        return lowpass_nth_order(t, f, tau=self.cfg.tau, order=self.cfg.order)

    # 4) Segmentation and labeling
    def to_segments(self, t: np.ndarray, y: np.ndarray) -> List[Tuple[float, float, str]]:
        return segment_and_label(t, y, win=self.cfg.sg_win, poly=self.cfg.sg_poly,
                                 slope_thr=self.cfg.slope_thr)

    # 5) Optional: compute Laplace transform values for diagnostics
    def diagnostic_laplace(self, t: np.ndarray, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s_vals = np.linspace(self.cfg.s_min, self.cfg.s_max, self.cfg.s_points)
        F_vals = laplace_transform_numeric(t, f, s_vals)
        return s_vals, F_vals

    # Run full pipeline
    def run(self, docs: List[str]) -> Dict:
        t, x, meta = self.build_discrete_signal(docs)
        t_cont, f_cont = self.discretized_to_continuous(t, x)
        y_trend = self.lowpass_trend(t_cont, f_cont)
        segments = self.to_segments(t_cont, y_trend)
        s_vals, F_vals = self.diagnostic_laplace(t_cont, f_cont)  # Based on original signal

        return {
            "t_discrete": t, "x_discrete": x,
            "t_cont": t_cont, "f_cont": f_cont,
            "y_trend": y_trend,
            "segments": segments,
            "laplace_s": s_vals, "laplace_F": F_vals,
            "meta": meta
        }


# =========================
# Usage example
# =========================
if __name__ == "__main__":
    import pandas as pd
    import os
    import json

    # Load review data
    products = [
        ("cleansing_oil", "../anua_cleansing_oil_reviews.csv"),
        ("toner",         "../anua_toner_reviews.csv"),
    ]

    cfg = LaplaceTextTrendConfig(
        use_sentiment=True,
        use_tfidf_topic=True,
        topic_components=8,
        topic_to_signal="entropy",           # options: "dominant" / "pc1"
        signal_mix={"sentiment": 0.6, "topic": 0.4},
        bandwidth=6.0,                       # recommended: 5~8
        tau=12.0,                            # recommended: 10~20
        order=2,                             # recommended: 2~3
        slope_thr=0.003,                     # recommended: 0.002~0.004
        sg_win=31, sg_poly=2,
        s_min=0.05, s_max=1.0, s_points=60
    )

    for product_name, file_path in products:
        print(f"\n{'='*50}\n▶ Processing: {product_name}\n{'='*50}")

        df = pd.read_csv(file_path)
        output_dir = f"../outputs_laplace_{product_name}"
        os.makedirs(output_dir, exist_ok=True)

        summary_rows = []

        for rating in range(1, 6):
            print(f"\n  Star rating {rating}")
            subset = df[df["rating"] == rating]["Review Content"].dropna().tolist()

            if len(subset) < 5:
                print(f"    Skipped: not enough reviews ({len(subset)})")
                continue

            try:
                pipe = LaplaceTextTrend(cfg)
                out = pipe.run(subset)

                # Save segment results
                summary_rows.append({
                    "product":   product_name,
                    "rating":    rating,
                    "n_reviews": len(subset),
                    "segments":  str(out["segments"]),
                    "sentiment_backend": out["meta"].get("sentiment_backend"),
                    "topic_mode":        out["meta"].get("topic_mode"),
                    "topic_components":  out["meta"].get("topic_components_effective"),
                    "tfidf_vocab_size":  out["meta"].get("tfidf_vocab_size"),
                })

                print(f"    Segments: {out['segments']}")
                print(f"    Sentiment backend: {out['meta'].get('sentiment_backend')}")

            except Exception as e:
                print(f"      Error: {e}")

        # Save summary CSV
        if summary_rows:
            import pandas as pd
            pd.DataFrame(summary_rows).to_csv(f"{output_dir}/laplace_summary.csv", index=False)
            print(f"\n  Result saved: {output_dir}/laplace_summary.csv")