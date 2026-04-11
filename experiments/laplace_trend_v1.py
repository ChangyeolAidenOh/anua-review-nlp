from __future__ import annotations
"""
Laplace transformation in LLM detecting feature trends.
- Raw text => Complex pattern => Categorized trend

Step 1 Preparation (Text -> Discrete Signal)
     1. Tokenization : Separate the text data (reviews) into tokens
     2. Embedding : One-Hot coding = Frequency response
                  : Word2Vec, GloVe = Including meaning
                  : Sentence-BERT = Embedding based on sentence
     3. Feature Selection : Select specific dimension (subject scores)
                          : PCA reducing sequence into 1 dimension (simplified process with discrete signal)

Step 2 Continuous from the given discrete signal
     1. Set the time-criteria : Ordering reviews (t = 1, 2, 3,...)
                              : Date of each review (when it was written: real numbers "t")
     2. Spline / Gaussian : Spline interpolation
                          : Gaussian smoothing
"""
import numpy as np
from scipy.interpolate import interp1d

# === 1. Transform text data to scale values ===
tokens = ["good", "bad", "excellent", "poor"]
values = np.array([0.8, -0.5, 1.0, -0.7])  # Example points
t = np.arange(len(values))

# === 2. Interpolate discrete data into continuous signal ===
f_interp = interp1d(t, values, kind='cubic')
t_cont = np.linspace(0, len(values)-1, 100)
f_cont = f_interp(t_cont)

# === 3. Numerical Laplace transform ===
def laplace_transform(f, t, s):
    return np.trapz(f * np.exp(-s * t), t)

s_vals = np.linspace(0.1, 1.0, 50)
F_vals = [laplace_transform(f_cont, t_cont, s) for s in s_vals]

# === 4. Low-pass filtering (keep only low-frequency components) ===
F_filtered = np.where(s_vals < 0.5, F_vals, 0)

# === 5. Inverse transformation ===
# In practice, numerical inverse Laplace transform would be applied here


"""
Realistic Scaffold
"""

import re
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter

# Optional fallback sentiment scorer based on sign-lexicon (used when VADER is unavailable)
_POS = {"great","good","excellent","amazing","love","nice","satisfied","happy","recommend","perfect","awesome"}
_NEG = {"bad","terrible","awful","poor","hate","disappointed","broken","refund","worst","noisy","leak","smell"}

def simple_sentiment_score(doc: str) -> float:
    tokens = re.findall(r"[a-z']+", doc.lower())
    pos = sum(t in _POS for t in tokens)
    neg = sum(t in _NEG for t in tokens)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


# Temporary: simpler segmentation attempt without sentiment score
# Not used in the final Laplace transformation pipeline

"""
def simple_sentiment_score_2(doc: str) -> float:
    tokens_v2 = re.findall(r"[a-z']+", doc.lower())
    pos_v2 = sum(t in _POS for t in tokens_v2)
    neg_v2 = sum(t in _NEG for t in tokens_v2)
    if pos_v2 + neg_v2 == 0:
        return 0.0
    return (pos_v2 - neg_v2) / (pos_v2 + neg_v2)
"""

# Early version of gaussian kernel regression (superseded below)
def gaussian_kernel_regression_v1(t: np.ndarray, x: np.ndarray, t_query: np.ndarray, bw: float) -> np.ndarray:
    d2_v2 = cdist(t_query[:, None], t[:, None], metric='sqeuclidean')
    k_v2 = np.exp(-bw * d2_v2)
    wsum_v2 = k_v2.sum(axis=1, keepdims=True) + 1e-12
    return (k_v2 @ x.reshape(-1, 1) / wsum_v2).ravel()

# Note: the version above was replaced because the bandwidth parameter now uses
# the standard 2*bw^2 denominator for consistency with the Gaussian kernel definition.


# Gaussian kernel regression: converts discrete signal to continuous
def gaussian_kernel_regression(t: np.ndarray, x: np.ndarray, t_query: np.ndarray, bw: float) -> np.ndarray:
    # Gaussian kernel: K = exp(-(delta_t)^2 / (2 * bw^2))
    d2 = cdist(t_query[:,None], t[:,None], metric='sqeuclidean')  # (Tq, T)
    K = np.exp(-d2 / (2.0 * bw * bw))
    # Normalized weighted average
    wsum = K.sum(axis=1, keepdims=True) + 1e-12
    return (K @ x.reshape(-1,1) / wsum).ravel()

# 1st order ODE low-pass filter — equivalent to Laplace H(s) = 1/(1+s*tau)
def lowpass_ode_filter(t: np.ndarray, x: np.ndarray, tau: float, y0: float = 0.0) -> np.ndarray:
    # Handles non-uniform time intervals safely
    y = np.empty_like(x)
    y_prev = y0
    t_prev = t[0]
    y[0] = y_prev
    for i in range(1, len(t)):
        dt = max(t[i] - t_prev, 1e-12)
        # Predictor: explicit Euler — tau dy/dt + y = x -> dy = (x - y)/tau * dt
        y_new = y_prev + ((x[i-1] - y_prev) / tau) * dt
        # Corrector: Heun's method for improved stability
        y_pred = y_new
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

# Laplace transform: numerical integral approximation on real axis s > 0
def laplace_transform_numeric(t: np.ndarray, f: np.ndarray, s_vals: np.ndarray) -> np.ndarray:
    # F(s) ~ sum f(t_i) * e^(-s * t_i) * delta_t
    dt = np.diff(t, prepend=t[0])
    F = []
    for s in s_vals:
        F.append(np.sum(f * np.exp(-s * t) * dt))
    return np.asarray(F)

# Gradient-based segmentation and labeling
def segment_and_label(t: np.ndarray, y: np.ndarray, win: int = 21, poly: int = 2,
                      slope_thr: float = 0.01) -> List[Tuple[float,float,str]]:
    # Estimate derivatives using Savitzky-Golay filter
    if win >= len(y):  # Adjust window if too large
        win = max(5, len(y)//2*2 - 1)
    y_smooth = savgol_filter(y, window_length=win, polyorder=poly, mode='interp')
    dt = np.gradient(t)
    dy = np.gradient(y_smooth) / (dt + 1e-12)
    # Interval labeling: UP / DOWN / FLAT
    labels = np.where(dy > slope_thr, 1, np.where(dy < -slope_thr, -1, 0))

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
# Main Pipeline (Class)
# =========================

@dataclass
class LaplaceTextTrendConfig:
    # Signal configuration
    use_sentiment: bool = True
    use_tfidf_topic: bool = True
    topic_components: int = 8
    topic_to_signal: str = "dominant"  # options: "entropy", "pc1"
    signal_mix: Dict[str, float] = None  # e.g. {"sentiment": 0.5, "topic": 0.5}

    # Continuity (Gaussian kernel bandwidth, based on review index)
    bandwidth: float = 5.0

    # Laplace low-pass filter
    tau: float = 10.0
    order: int = 2  # n-th order low-pass

    # Labeling
    slope_thr: float = 0.002
    sg_win: int = 41
    sg_poly: int = 2

    # Optional Laplace transform (diagnostic)
    s_min: float = 0.05
    s_max: float = 1.0
    s_points: int = 50

    def __post_init__(self):
        if self.signal_mix is None:
            self.signal_mix = {"sentiment": 0.5, "topic": 0.5}

class LaplaceTextTrend:
    def __init__(self, cfg: LaplaceTextTrendConfig):
        self.cfg = cfg
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.nmf: Optional[NMF] = None
        self.scaler = StandardScaler(with_mean=True, with_std=True)

    # 1) Text -> Discrete scalar signal
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
            self.nmf = NMF(n_components=k, init="nndsvda", random_state=42, max_iter=1000)
            W = self.nmf.fit_transform(X)  # (N, k): document x topic weights

            # Reduce topic matrix to a single scalar signal
            if self.cfg.topic_to_signal == "dominant":
                topic_signal = W.argmax(axis=1) / max(1, k-1)  # normalized to [0, 1]
            elif self.cfg.topic_to_signal == "entropy":
                p = W / (W.sum(axis=1, keepdims=True) + 1e-12)
                ent = -(p*np.log(p+1e-12)).sum(axis=1)
                topic_signal = (ent - ent.min()) / (ent.ptp() + 1e-12)
            else:  # "pc1": project onto first principal component
                topic_signal = PCA(n_components=1, random_state=42).fit_transform(W).ravel()
                topic_signal = (topic_signal - topic_signal.mean()) / (topic_signal.std()+1e-12)
            parts.append(("topic", topic_signal))
            meta["topic_mode"] = self.cfg.topic_to_signal

        # Weighted mixture of standardized signal components
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

    # 2) Discrete -> Continuous
    def discretized_to_continuous(self, t: np.ndarray, x: np.ndarray, upsample: int = 10) -> Tuple[np.ndarray,np.ndarray]:
        t_cont = np.linspace(t.min(), t.max(), num=max(upsample*len(t), 200))
        f_cont = gaussian_kernel_regression(t, x, t_cont, bw=self.cfg.bandwidth)
        return t_cont, f_cont

    # 3) Laplace-perspective low-pass filter
    def lowpass_trend(self, t: np.ndarray, f: np.ndarray) -> np.ndarray:
        return lowpass_nth_order(t, f, tau=self.cfg.tau, order=self.cfg.order)

    # 4) Segmentation and labeling
    def to_segments(self, t: np.ndarray, y: np.ndarray) -> List[Tuple[float,float,str]]:
        return segment_and_label(t, y, win=self.cfg.sg_win, poly=self.cfg.sg_poly, slope_thr=self.cfg.slope_thr)

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

    # Load review data
    products = [
        ("cleansing_oil", "../anua_cleansing_oil_reviews.csv"),
        ("toner",         "../anua_toner_reviews.csv"),
    ]

    cfg = LaplaceTextTrendConfig(
        use_sentiment=True,
        use_tfidf_topic=True,
        topic_components=6,
        topic_to_signal="entropy",           # options: "dominant" / "pc1"
        signal_mix={"sentiment": 0.6, "topic": 0.4},
        bandwidth=3.5,                       # recommended: 5~8
        tau=8.0,                             # recommended: 10~20
        order=2,                             # recommended: 2~3
        slope_thr=0.003,                     # recommended: 0.002~0.004
        sg_win=31, sg_poly=2,
        s_min=0.05, s_max=1.2, s_points=60
    )

    for product_name, file_path in products:
        print(f"\n{'='*50}\n▶ Processing: {product_name}\n{'='*50}")

        df = pd.read_csv(file_path)
        output_dir = f"../outputs_laplace_v1_{product_name}"
        os.makedirs(output_dir, exist_ok=True)

        summary_rows = []

        for rating in range(1, 6):
            print(f"\n  Star Rating {rating}")
            subset = df[df["rating"] == rating]["Review Content"].dropna().tolist()

            if len(subset) < 5:
                print(f"    Skipped: not enough reviews ({len(subset)})")
                continue

            try:
                pipe = LaplaceTextTrend(cfg)
                out = pipe.run(subset)

                summary_rows.append({
                    "product":         product_name,
                    "rating":          rating,
                    "n_reviews":       len(subset),
                    "segments":        str(out["segments"]),
                    "topic_mode":      out["meta"].get("topic_mode"),
                    "sentiment_range": str(out["meta"].get("sentiment_range")),
                })

                print(f"    Segments: {out['segments']}")

            except Exception as e:
                print(f"     Error: {e}")

        # Save summary CSV
        if summary_rows:
            pd.DataFrame(summary_rows).to_csv(f"{output_dir}/laplace_summary.csv", index=False)
            print(f"\n  Result saved: {output_dir}/laplace_summary.csv")