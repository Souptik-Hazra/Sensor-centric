# Causal Fairness Attribution & Methodology Specification

## 1. Overview
This document specifies the formal causal framework and fairness metrics used in this dissertation to diagnose and attribute forecast accuracy disparities in spatiotemporal traffic prediction.

---

## 2. Structural Causal Model (SCM) & DAG
The causal relationships governing sensor density, operational reliability, spatial topology, and prediction error are formalized as a Directed Acyclic Graph (DAG):

$$\begin{aligned}
C &\to D \\
D &\to R, \quad D \to T, \quad D \to Y \\
R &\to Y, \quad T &\to Y
\end{aligned}$$

### Variable Definitions
- **$C$ (Confounders)**: Environmental and infrastructure covariates (road classification, traffic regime).
- **$D$ (Sensor Density)**: Local spatial density of loop detectors (treatment variable).
- **$R$ (Sensor Reliability)**: Time-varying operational health score of a sensor ($R \in [0, 1]$).
- **$T$ (Topology Sensitivity)**: Static graph-topological centrality and connectivity metric.
- **$Y$ (Forecast Disparity)**: Model residual error (or baseline persistence error) per sensor.

---

## 3. Counterfactual Fairness Decomposition (Plečko & Bareinboim, 2024)
Using the Standard Fairness Model (SFM) projection, total disparity across density levels ($x_0$ vs. $x_1$) is decomposed into three distinct causal pathways:

### A. Total Effect (TE)
$$\text{TE} = \mathbb{E}[Y \mid do(X = x_1)] - \mathbb{E}[Y \mid do(X = x_0)]$$

### B. Counterfactual Direct Effect (Ctf-DE)
Measures the unmediated disparity directly attributed to density $D$, keeping mediators fixed counterfactually:
$$\text{Ctf-DE} = \mathbb{E}[Y_{x_1, R_{x_0}, T_{x_0}}] - \mathbb{E}[Y_{x_0, R_{x_0}, T_{x_0}}]$$

### C. Counterfactual Indirect Effect (Ctf-IE)
Measures the disparity mediated through sensor reliability $R$ and topology $T$:
$$\text{Ctf-IE}_R = \mathbb{E}[Y_{x_1, R_{x_1}, T_{x_0}}] - \mathbb{E}[Y_{x_1, R_{x_0}, T_{x_0}}]$$
$$\text{Ctf-IE}_T = \mathbb{E}[Y_{x_1, R_{x_0}, T_{x_1}}] - \mathbb{E}[Y_{x_1, R_{x_0}, T_{x_0}}]$$

### D. Counterfactual Spurious Effect (Ctf-SE)
Measures disparity arising from backdoor confounding via $C$:
$$\text{Ctf-SE} = \text{TE} - (\text{Ctf-DE} + \text{Ctf-IE}_R + \text{Ctf-IE}_T)$$

---

## 4. Sensor Reliability Formulation ($R$)
Sensor operational reliability $R_i \in [0, 1]$ is constructed from three distinct anomaly indicators:
$$R_i = 1 - \left( w_1 \cdot \text{ZeroRate}_i + w_2 \cdot \text{CUSUM}_i + w_3 \cdot \text{EWMA}_i \right)$$

1. **ZeroRate**: Proportion of zero/stuck speed readings ($\le 1.0 \text{ mph}$).
2. **CUSUM**: Cumulative sum control chart flag rate for mean drift.
3. **EWMA**: Exponentially weighted moving average control chart flag rate for variance drift.

### Weighting Schemes Evaluated (Notebook 11)
- **Heuristic**: $w_1 = 0.6, w_2 = 0.2, w_3 = 0.2$
- **Equal**: $w_1 = 0.333, w_2 = 0.333, w_3 = 0.333$
- **Zero-Only**: $w_1 = 1.0, w_2 = 0, w_3 = 0$
- **PCA-Derived**: First principal component ($PC_1$) loading orientation.

---

## 5. Algorithmic Fairness Metrics

### A. Region-based Static Fairness (RSF)
$$\text{RSF} = \frac{2}{K(K-1)} \sum_{i=1}^{K-1} \sum_{j=i+1}^{K} \left| \text{MAPE}(R_i) - \text{MAPE}(R_j) \right|$$
- $K = 13$ spatial regional districts.
- Evaluates spatial disparity across geographic areas.

### B. Sensor-based Dynamic Fairness (SDF)
$$\text{SDF} = \frac{2}{N(N-1)} \sum_{i=1}^{N-1} \sum_{j=i+1}^{N} \left( \left| \text{Pos}_i - \text{Pos}_j \right| + \left| \text{Neg}_i - \text{Neg}_j \right| \right)$$
- $N = 207$ physical sensors.
- Tracks cumulative benefit (positive state) and sacrifice (negative state) balances over time.
