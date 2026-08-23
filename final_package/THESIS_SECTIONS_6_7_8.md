# 📝 Dissertation Sections: [6.1], [7.], and [8.]

---

## [6.1] FINDINGS IN LITERATURE SURVEY

Based on the exhaustive systematic review of 15 foundational papers in spatiotemporal Graph Neural Networks (ST-GNNs), non-graph time series baselines, AI fairness frameworks, microscopic traffic twins, and Judea Pearl Structural Causal Models (SCMs), four major findings and research gaps emerge:

### 1. Spatial Error Disparity in Deep ST-GNNs (Papers 1–6)
Existing state-of-the-art ST-GNN architectures (**DCRNN**, **Graph WaveNet**, **STGCN**, **AGCRN**, **GMAN**, **BGCN**) optimize exclusively for global Mean Absolute Error ($\text{MAE}$). In doing so, they conceal severe spatial error disparity across regional highway sub-networks ($\text{RSF} \ge 0.35$). Adaptive adjacency matrices ($\mathbf{\tilde{A}}_{\text{adp}}$) and latent node embeddings fail to disentangle physical sensor failure from traffic flow dynamics.

### 2. Failure of Non-Spatial Models (Paper 7)
Lightweight non-graph linear decomposition models (**DLinear**) eliminate graph convolution to achieve fast execution, but completely ignore spatial graph topology ($W_{ij}$). Consequently, prediction error degrades on complex highway networks ($\text{MAE} = 3.12$ mph vs GNN $\text{MAE} = 2.69$ mph), proving that spatial graph diffusion cannot be discarded.

### 3. Fairness–Accuracy Trade-Off in Software Re-Weighting (Papers 9–10)
Current AI fairness approaches (**FairTP**, **FairSTG**) operate strictly at **Pearl Level 1 (Observational Association)**. They treat disparity as a software loss re-weighting or re-sampling problem. This forces an explicit **Fairness–Accuracy Trade-off**, inflicting a **$+0.52\%$ to $+13.03\%$ prediction error penalty** without repairing the underlying physical root cause.

### 4. Lack of Pearl Level-3 Counterfactual $do$-Calculus in Digital Twins (Papers 11–13)
Existing traffic digital twins (**SUMO**, **TrafficLLM**, **DTCF**) rely on physical vehicle microsimulation, LLM prompt engineering, or Rubin's Potential Outcomes copulas. None of them incorporate **Judea Pearl Level-3 Structural Causal Models (SCMs)** or counterfactual abduction ($U_{\text{factual}} = u^*$) required to simulate interventional hardware repair actions ($do(R_i = 0.95)$).

### 💡 Core Research Gap Addressed by This Work
There is a critical need for a **Structural Causal Digital Twin** that bridges Pearl's SCM framework with spatiotemporal GNNs—attributing spatial disparity to physical hardware decay ($R_i$) via counterfactual mediation ($Ctf\text{-}IE_R = 61.3\%$) and achieving optimal equity (**MAE = 2.44 mph, RSF = 0.18**) with zero accuracy loss.

---

## [7.] METHODOLOGY

The proposed **4-Layer Structural Causal Digital Twin** integrates Judea Pearl's Level-3 SCM $do$-calculus with Plecko & Bareinboim's Structural Fairness Model (SFM), SIMD vectorization, and WebGL GIS visualization.

```
+-----------------------------------------------------------------------------------+
| LAYER 1: WebGL GIS Console (JavaScript / Leaflet)                                |
| • Interactive 207-Sensor Map  • 3-Horizon Selector (15m, 30m, 60m)  • Replay    |
+-----------------------------------------------------------------------------------+
                                        ▲
                                        │ digital_twin_state.json
+-----------------------------------------------------------------------------------+
| LAYER 2: Pearl Level-3 Causal Digital Twin Engine (Python CSR / SIMD)             |
| • do(R_i = 0.95) Repair  • BLAS-1 Rank-1 Outer Product  • Sparse CSR Diffusion      |
| • Multi-Fault Stress Sweeps  • 4.42 ms Interventional Evaluation Latency          |
+-----------------------------------------------------------------------------------+
                                        ▲
                                        │ SFM Coefficients & Pareto Frontier
+-----------------------------------------------------------------------------------+
| LAYER 3: Causal Identification & Plecko SFM Mediation (R faircause / dagitty)     |
| • SCM DAG Backdoor Criterion  • 1,000-Bootstrap 95% CIs  • Pareto Optimality      |
| • Counterfactual Attribution: Ctf-IE_R = 61.3%, Ctf-DE = 21.4%, Ctf-IE_T = -17.3% |
+-----------------------------------------------------------------------------------+
                                        ▲
                                        │ metr_la_metrics.csv
+-----------------------------------------------------------------------------------+
| LAYER 4: Telemetry Feature & SIMD Extraction Layer (Python SIMD)                  |
| • CUSUM Drift  • EWMA Volatility  • Zero-Rate Dropouts  • Graph Laplacian L_sym     |
+-----------------------------------------------------------------------------------+
```

### Layer 4: Telemetry Feature & SIMD Extraction Layer
- **Input Data:** METR-LA benchmark dataset (207 loop detectors, 34,272 time-steps).
- **Signal Extraction:** Computes stuck-zero dropout rates ($Z_i$), CUSUM calibration drift flags ($D_i$), and EWMA volatility spikes ($V_i$).
- **Composite Reliability Score:** Formulates composite hardware score $R_i = 1 - (0.60 Z_i + 0.20 D_i + 0.20 V_i)$.
- **Spatial Topology & Spectral Clustering:** Computes Haversine spatial density ($D_i$) and normalized graph Laplacian $\mathbf{L}_{\text{sym}} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{W} \mathbf{D}^{-1/2}$ to partition the network into 13 spectral regional clusters.

### Layer 3: Causal Identification & Plecko SFM Mediation Layer
- **SCM DAG Identifiability:** Constructs structural causal graph $G = (V, E)$ establishing conditional independence $R_i \perp\!\!\!\perp T_i \mid D_i$. Proves Backdoor Criterion satisfaction using adjustment set $\mathbf{Z} = \{\text{road\_type}, \text{traffic\_regime}\}$.
- **Plecko Counterfactual Mediation:** Decomposes spatial error disparity into four distinct causal pathways using 1,000 non-parametric bootstrap iterations:
  $$\text{TE} = \underbrace{Ctf\text{-}DE}_{\text{Density } (21.4\%)} + \underbrace{Ctf\text{-}IE_R}_{\text{Hardware Reliability } (61.3\%)} + \underbrace{Ctf\text{-}IE_T}_{\text{Topology } (-17.3\%)} + \underbrace{Ctf\text{-}SE}_{\text{Confounding } (0.0\%)}$$
- **Pareto Dominance Frontier:** Proves mathematically that Level-3 structural hardware repair $do(R_i = 0.95)$ strictly dominates Level-1 software loss re-weighting (FairSTG, FairTP).

### Layer 2: Pearl Level-3 Causal Digital Twin Engine
- **Counterfactual Intervention Operator:** Executes query $do(R_i = 0.95)$ for target highway districts via SIMD vectorization.
- **Sparse CSR Random-Walk Diffusion:** Computes dynamic edge degradation via BLAS rank-1 outer product scaling matrix $\mathbf{W}_{\text{degraded}} = \mathbf{W}_0 \odot (\mathbf{R} \mathbf{R}^T)$ and updates sparse CSR diffusion operators $\mathbf{P}_f = \mathbf{D}_o^{-1} \mathbf{W}_{\text{degraded}}$.
- **Compound Multi-Fault Stress Engine:** Evaluates network resilience under multi-fault perturbations (10% to 90% sensor dropouts), guaranteeing SLA compliance ($\text{RSF} \le 0.1930$).

### Layer 1: WebGL GIS Interactive Console
- **Interactive Map:** Renders 207 physical sensors with real-time hardware status indicators (Healthy, Drift, Failed).
- **Multi-Horizon Selector:** Toggle between 15-min, 30-min, and 60-min equity evaluation steps.
- **Intervention Stimulus Console:** Allows operators to trigger $do(R_i = 0.95)$ hardware repairs or $do(D_i += 5)$ density expansions with live 60 FPS map re-rendering.

---

## [8.] SOFTWARE REQUIREMENTS

### 8.1 Functional Requirements (FR)

| Req ID | Title | Description |
|:---:|:---|:---|
| **FR-1** | **Telemetry Feature Extraction** | The system shall compute zero-rate dropouts, CUSUM drift flags, EWMA volatility rates, and composite hardware reliability scores ($R_i$) across 207 sensors. |
| **FR-2** | **SCM DAG Identifiability** | The system shall verify Backdoor Criterion satisfaction and partition the graph into 13 spectral clusters using graph Laplacian $\mathbf{L}_{\text{sym}}$. |
| **FR-3** | **Plecko Counterfactual Mediation** | The system shall execute 1,000-bootstrap counterfactual decomposition, quantifying $Ctf\text{-}DE$, $Ctf\text{-}IE_R$, $Ctf\text{-}IE_T$, and $Ctf\text{-}SE$ with 95% confidence intervals. |
| **FR-4** | **Interventional Causal Simulator** | The system shall support counterfactual queries $do(R_i = 0.95)$ and $do(D_i += 5)$, re-calculating network RSF disparity in real time. |
| **FR-5** | **Pareto Frontier Evaluation** | The system shall evaluate Pareto optimality across 5 baseline strategies, identifying dominated software re-weighting approaches. |
| **FR-6** | **WebGL GIS Console & Multi-Horizon Export** | The system shall render an interactive 60 FPS GIS console and export multi-horizon equity metrics (15m, 30m, 60m) to CSV/JSON format. |

### 8.2 Non-Functional Requirements (NFR)

| Req ID | Category | Description |
|:---:|:---|:---|
| **NFR-1** | **Performance & Latency** | Interventional counterfactual queries shall execute with sub-millisecond to sub-5ms evaluation latency ($4.42\text{ ms}$). The WebGL GIS map shall render at $\ge 60\text{ FPS}$. |
| **NFR-2** | **Memory Efficiency** | Sparse CSR matrix representation shall achieve $>30\times$ memory compression over dense matrices. Tensor buffers shall use memory-mapped pre-allocation. |
| **NFR-3** | **Reliability & Robustness** | The system shall maintain SLA equity compliance ($\text{RSF} \le 0.1930$) under extreme multi-fault stress sweeps (up to 90% sensor dropout). |
| **NFR-4** | **Statistical Rigor** | Bootstrap mediation estimates shall achieve statistical power $>95\%$ at sample size $N=207$, verified via 500 Monte Carlo power iterations. |
| **NFR-5** | **Modularity & Polyglot Integration** | Python SIMD, R statistical engines (`faircause`, `dagitty`), and JavaScript GIS visualization modules shall communicate seamlessly via standard CSV/JSON data contracts. |
| **NFR-6** | **Hardware Independence** | The complete 13-script pipeline shall execute cleanly on standard consumer CPUs without requiring specialized GPU accelerators or native C++ compilers. |
