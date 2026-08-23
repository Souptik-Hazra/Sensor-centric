# 🎓 Viva Voce Thesis Defense Guide — Top 15 Adversarial Questions & Expert Answers

---

### Q1: Why use a Pearl Level-3 Structural Causal Model (SCM) instead of standard deep learning GNNs?
**Expert Answer:**
Standard ST-GNNs (DCRNN, Graph WaveNet) operate at **Pearl Level 1 (Observational Association)**. They learn statistical correlations $P(Y \mid X)$, which means they cannot distinguish between traffic speed variations caused by real physical congestion versus hardware sensor decay (e.g., stuck-zero dropouts, calibration drift). A Pearl Level-3 SCM enables counterfactual $do$-calculus queries $P(Y_{do(R_i = 0.95)} \mid x, y)$, allowing us to simulate physical hardware repair actions and prove that **61.3%** of regional prediction error disparity is caused by hardware unreliability ($R_i$), rather than network topology.

---

### Q2: How do you prove SCM DAG identifiability and satisfy the Backdoor Criterion?
**Expert Answer:**
Our SCM DAG defines the causal structure between Density ($D_i$), Reliability ($R_i$), Topology ($T_i$), and Error Disparity ($Y_i$). We establish that Hardware Reliability is conditionally independent of Topology given Spatial Density ($R_i \perp\!\!\!\perp T_i \mid D_i$). To identify the causal effect of $R_i$ on $Y_i$ without confounding bias, we block all backdoor pathways by conditioning on the adjustment set $\mathbf{Z} = \{\text{road\_type}, \text{traffic\_regime}\}$. The empirical positivity condition holds across all 207 sensors in METR-LA, ensuring non-zero probability for all confounder combinations.

---

### Q3: Why is Plecko & Bareinboim's Structural Fairness Model (SFM) superior to standard ML fairness metrics?
**Expert Answer:**
Standard fairness metrics (Equalized Odds, Demographic Parity) treat disparity as a monolithic software problem. Plecko & Bareinboim's SFM uses counterfactual mediation to mathematically decompose total outcome disparity ($\text{TE}$) into four distinct causal mechanisms:
$$\text{TE} = \underbrace{Ctf\text{-}DE}_{\text{Direct Density Effect } (21.4\%)} + \underbrace{Ctf\text{-}IE_R}_{\text{Hardware Reliability } (61.3\%)} + \underbrace{Ctf\text{-}IE_T}_{\text{Topology Buffer } (-17.3\%)} + \underbrace{Ctf\text{-}SE}_{\text{Confounding } (0.0\%)}$$
This proves that 61.3% of the disparity is mediated through hardware reliability, giving transportation authorities an actionable physical target for intervention rather than an abstract software penalty.

---

### Q4: Why did software fairness baselines (FairTP, FairSTG) fail on the Pareto Frontier?
**Expert Answer:**
Software fairness approaches (FairTP, FairSTG) operate by penalizing high-performing sensors or re-weighting loss functions during training. Because they do not repair the physical hardware root cause, they force an explicit **Fairness–Accuracy Trade-Off**, inflicting a **$+0.52\%$ to $+13.03\%$ prediction error penalty** ($\text{MAE} = 2.89$ mph). In contrast, our Level-3 Causal Digital Twin performs targeted hardware repair ($do(R_i = 0.95)$), achieving **MAE = 2.44 mph** and **RSF = 0.18**—strictly dominating all software approaches on the Pareto Dominance Frontier.

---

### Q5: Why benchmark DCRNN, Graph WaveNet, DLinear, and HA together?
**Expert Answer:**
This 4-model baseline selection forms a complete methodological spectrum:
1. **HA (4.86 mph)**: Naive statistical persistence baseline (proves machine learning is necessary).
2. **DLinear (3.12 mph)**: Non-graph deep learning baseline (proves spatial graph topology $W_{ij}$ is necessary).
3. **DCRNN (2.77 mph)**: Classic diffusion ST-GNN baseline (models spatial random walks, but ignores hardware decay).
4. **Graph WaveNet (2.69 mph)**: Adaptive ST-GNN baseline (learns $\mathbf{\tilde{A}}_{\text{adp}}$, but masks sensor dropouts).

Comparing against all four demonstrates that non-graph models fail, standard GNNs create severe spatial disparity ($\text{RSF} \ge 0.35$), and our Causal Twin resolves the disparity without accuracy loss.

---

### Q6: How is composite hardware reliability ($R_i$) calculated?
**Expert Answer:**
$R_i$ combines three empirical signal degradation indicators extracted from the 34,272 time-steps of METR-LA telemetry:
$$R_i = 1.0 - (0.60 \cdot Z_i + 0.20 \cdot D_i + 0.20 \cdot V_i)$$
where $Z_i$ is the stuck-zero dropout rate ($\text{speed} \le 1.0$ mph), $D_i$ is the CUSUM cumulative sum calibration drift rate, and $V_i$ is the EWMA volatility spike rate. PCA sensitivity analysis verified that alternative weighting schemes (equal weighting, zero-only, PCA-derived) are $> 0.91$ correlated, proving our causal conclusions are invariant to the weighting formula.

---

### Q7: How does the Digital Twin achieve sub-5ms interventional evaluation latency?
**Expert Answer:**
We employ **BLAS Rank-1 Outer Product SIMD Vectorization** and **Sparse CSR Matrix Operators**. When an operator executes query $do(R_i = 0.95)$ for a district, the degraded adjacency matrix is computed in $O(N)$ vector time:
$$\mathbf{W}_{\text{degraded}} = \mathbf{W}_0 \odot (\mathbf{R} \mathbf{R}^T)$$
Sparse CSR representation compresses memory by $> 30\times$, allowing the simulator to re-calculate global network diffusion operators ($\mathbf{P}_f$) in **$4.42\text{ ms}$**, enabling real-time WebGL GIS interactive simulation.

---

### Q8: What is Regional Static Fairness (RSF) and how is it measured?
**Expert Answer:**
RSF measures prediction error disparity across regional highway sub-networks. The 207 sensors are partitioned into $K=13$ spatial clusters using the normalized graph Laplacian $\mathbf{L}_{\text{sym}} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{W} \mathbf{D}^{-1/2}$. RSF is defined as the standard deviation of district-level Mean Absolute Errors:
$$\text{RSF} = \sqrt{\frac{1}{K} \sum_{k=1}^K (\text{MAE}_k - \overline{\text{MAE}})^2}$$
A lower RSF indicates higher spatial equity across all highway districts.

---

### Q9: What happens under extreme multi-fault compound stress?
**Expert Answer:**
We executed compound multi-fault stress sweeps injecting combined stuck-zeros, CUSUM drift, and EWMA volatility across 10%, 30%, 50%, 70%, and 90% sensor dropout rates. While baseline Graph WaveNet RSF degraded to 0.65, the Causal Digital Twin maintained **RSF $\le 0.1930$**, remaining strictly SLA-compliant even when 90% of physical sensors suffered hardware degradation.

---

### Q10: Why is sensor density expansion ($do(D_i += 5)$) less cost-effective than hardware repair?
**Expert Answer:**
Adding 5 new physical loop detectors costs \$250,000 (\$50,000/sensor). Because spatial density only accounts for **21.4%** of disparity ($Ctf\text{-}DE$), density expansion yields a minor RSF improvement at \$18,500 per RSF point. In contrast, upgrading existing physical sensor hardware ($do(R_i = 0.95)$) targets the **61.3%** primary causal channel ($Ctf\text{-}IE_R$), yielding a $5.24\times$ higher equity return per dollar spent.

---

### Q11: How do you prove statistical power and sample sufficiency at $N=207$?
**Expert Answer:**
We ran 500 Monte Carlo power iterations in R (`10_power_simulation.R`). At sample size $N=207$, the statistical power to detect the indirect hardware effect ($Ctf\text{-}IE_R = 61.3\%$) exceeds **95%** at significance level $\alpha = 0.05$. The 1,000 non-parametric bootstrap 95% confidence intervals confirm that the hardware mediation estimate $[54.2\%, 68.4\%]$ is bounded away from zero.

---

### Q12: How does your work differ from existing traffic digital twins (SUMO, TrafficLLM, DTCF)?
**Expert Answer:**
SUMO is a microscopic vehicle physics simulator that performs heuristic scenario replay without causal SCMs. TrafficLLM uses LLM prompt engineering for cybersecurity packets, suffering from 4.7% hallucination rates and lacking spatial graph convolutions. DTCF (Laudy 2026) uses Rubin's Potential Outcomes framework, where joint counterfactual dependence remains unidentifiable. Ours is the **first Pearl Level-3 SCM Digital Twin** combining $do$-calculus, sparse CSR diffusion operators, and Plecko SFM fairness mediation.

---

### Q13: How does the WebGL GIS Console function?
**Expert Answer:**
The interactive console (`digital_twin_gis_map.html`) is built using Leaflet.js and HTML5 Canvas rendering, maintaining **60 FPS** performance while displaying 207 sensors. It features a 3-horizon selector (15-min, 30-min, 60-min), a 288-step temporal replay slider, real-time hardware status pins, and an interactive stimulus panel allowing traffic operators to simulate $do(R_i = 0.95)$ repairs visually.

---

### Q14: Could this framework be deployed on real-world Smart City IoT networks?
**Expert Answer:**
Yes. The architecture is polyglot, modular, and hardware-independent. It communicates via lightweight CSV/JSON data contracts, executes in under 5ms per interventional query on standard consumer CPUs, and requires no proprietary GPU infrastructure. It can be directly integrated into municipal traffic management centers (TMCs) via REST APIs.

---

### Q15: What is the main thesis contribution in one sentence?
**Expert Answer:**
*"This dissertation presents the first 4-Layer Structural Causal Digital Twin for spatiotemporal traffic networks that proves hardware reliability accounts for 61.3% of regional prediction error disparity and achieves optimal equity (MAE = 2.44 mph, RSF = 0.18) via Pearl Level-3 interventional hardware repair without sacrificing overall forecasting accuracy."*
