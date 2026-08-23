import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os

metrics_path = "metr_la_metrics.csv"
if not os.path.exists(metrics_path) and os.path.exists(os.path.join("..", metrics_path)):
    metrics_path = os.path.join("..", metrics_path)

df = pd.read_csv(metrics_path)
print(f"[+] Loaded {len(df)} sensors from {metrics_path}")

# ==========================================

# --- Build 4 alternative reliability score versions ---

# (a) Original heuristic weighting
df["reliability_original"] = 1 - (0.6*df.zero_rate + 0.2*df.cusum_flag_rate + 0.2*df.ewma_flag_rate)

# (b) Equal weighting
df["reliability_equal"] = 1 - (df.zero_rate + df.cusum_flag_rate + df.ewma_flag_rate) / 3

# (c) Zero-rate only
df["reliability_zero_only"] = 1 - df.zero_rate

# (d) PCA-derived weights (data-driven, not heuristic)
signals = df[["zero_rate", "cusum_flag_rate", "ewma_flag_rate"]].values
signals_std = (signals - signals.mean(axis=0)) / signals.std(axis=0)
pca = PCA(n_components=1)
pc1 = pca.fit_transform(signals_std).flatten()
# orient so higher = less reliable, matching the other scores' direction
if np.corrcoef(pc1, df.zero_rate)[0,1] < 0:
    pc1 = -pc1
pc1_norm = (pc1 - pc1.min()) / (pc1.max() - pc1.min())
df["reliability_pca"] = 1 - pc1_norm
print("PCA loadings (zero_rate, cusum_flag_rate, ewma_flag_rate):", pca.components_[0])

print(df[["node_id", "reliability_original", "reliability_equal",
    "reliability_zero_only", "reliability_pca"]].describe().round(4))


# ==========================================

# --- Correlation between the 4 versions, so you know upfront how different
#     they actually are before running the (much slower) R re-estimation ---
corr = df[["reliability_original", "reliability_equal",
           "reliability_zero_only", "reliability_pca"]].corr()
print(corr.round(3))
print("\nIf these are all >0.9 correlated with each other, the R re-estimation")
print("in the next file will likely show a stable causal conclusion regardless")
print("of weighting. If any pair is well below 0.9, expect the causal estimate")
print("to meaningfully shift between weighting schemes.")


# ==========================================

# traffic_regime, road_type, density, topology are already in df
df["disparity"] = df["persistence_error"]

df.to_csv("reliability_variants.csv", index=False)
print("Saved reliability_variants.csv")