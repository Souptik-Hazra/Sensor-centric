```md
# Literature Review

## Paper

### Title

**FairTP: A Prolonged Fairness Framework for Traffic Prediction**

### Authors

**Jiangnan Xia, Yu Yang, Jiaxing Shen, Senzhang Wang, Jiannong Cao**

### Year

**2025**

### Venue / Journal / Conference

**Proceedings of the Thirty-Ninth AAAI Conference on Artificial Intelligence (AAAI 2025)**

# 1. Objective & Problem Formulation

- The paper investigates **fairness in traffic prediction**, focusing on whether prediction performance is distributed equitably across different urban regions and road sensors rather than optimizing only overall forecasting accuracy.
- The primary problem arises from the **uneven deployment of traffic sensors across urban areas**, which creates imbalanced data volumes and can cause prediction models to perform poorly in regions with fewer sensors.
- Unequal prediction performance can lead to **unfair regional decision-making**, potentially affecting transportation services and the equity and quality of residents' lives.
- Existing fairness approaches may achieve fairness at particular time points but fail to maintain it as traffic conditions change.
- The paper therefore investigates **prolonged fair traffic prediction**, where fairness is considered across regions and sensors over time.
- Two fairness definitions are introduced: **region-based static fairness (RSF)** and **sensor-based dynamic fairness (SDF)**.
- The framework aims to improve fairness while avoiding substantial degradation of prediction accuracy.

# 2. Existing Systems & Background

- Existing traffic prediction research has progressed from deep learning approaches based on **CNNs and RNNs** toward graph-based spatio-temporal models.
- CNNs and RNNs have been widely used for traffic prediction because of their ability to model spatial and temporal patterns, but the paper notes that these approaches were designed primarily for **spatio-temporal grid data**.
- Graph Neural Networks subsequently became important because road traffic is naturally represented as a graph containing road sensors and their connections.
- Existing graph-based approaches combine GNNs with RNNs, TCNs, or attention mechanisms to capture spatial and temporal dependencies.
- The traffic forecasting models considered in the paper include **DCRNN, AGCRN, GWNET, ASTGCN, STGODE, DSTAGNN, DCGRN, and D2STGNN**.
- Existing traffic prediction models primarily emphasize **overall prediction accuracy**.
- Existing fairness research generally considers fairness among groups and may rely on sensitive attributes such as **race or gender**.
- Transportation-oriented fairness approaches discussed in the paper include **FairST** and **SA-Net**.
- FairST uses fairness metrics and regularization to promote equity across demographic groups.
- SA-Net incorporates **socio-demographic and ridership information** for fair demand prediction.
- Another dynamic fairness approach discussed in the paper relies on a similarity matrix constructed using **domain knowledge or human judgment**.
- The underlying assumption of conventional traffic forecasting is largely accuracy-oriented, whereas the paper argues that prediction quality must also be considered across regions, sensors, and time.

# 3. Limitations of Existing Systems & Research Gaps

- Existing traffic prediction models generally focus on **overall predictive accuracy** and overlook whether their predictions produce biased outcomes across different urban regions.
- Uneven sensor deployment creates **imbalanced traffic data**. Regions with fewer sensors consequently have less data representation and can experience larger prediction errors.
- Existing models do not explicitly address the fairness consequences of this **sensor-distribution imbalance**.
- Existing fairness approaches primarily address fairness at **specific time points**, making them inadequate for traffic environments whose conditions change over time.
- Static fairness can therefore break down as traffic conditions evolve.
- The paper identifies a lack of clear definitions for **measuring and quantifying fairness in dynamic traffic environments**.
- Existing group-fairness approaches may depend on sensitive attributes such as race or gender, which are not the basis of FairTP's traffic-sensor fairness formulation.
- Some transportation fairness approaches depend on external information such as socio-demographic and ridership data.
- A previously discussed dynamic fairness approach relies on an oracle similarity matrix created using domain knowledge or human judgment, which the authors regard as unsuitable for traffic scenarios.
- Directly optimizing regional fairness can reduce prediction accuracy in privileged regions with dense sensor coverage **without necessarily improving underprivileged regions**.
- Simply reducing prediction disparity between regions is therefore insufficient; the framework should also improve prediction opportunities for regions affected by sparse sensor coverage.
- The central research gap is the absence of a traffic prediction framework that simultaneously addresses **regional fairness, prolonged sensor-level fairness, changing traffic conditions, and uneven sensor distribution while preserving prediction accuracy**.

# 4. Proposed System & Technical Architecture

- FairTP is a **prolonged fairness traffic prediction framework** designed to integrate with existing traffic prediction models.
- The framework consists of three principal components:
  1. **State-guided balanced sampling module**
  2. **Spatio-temporal dependencies learning module**
  3. **State identification module**
- FairTP represents prolonged fairness through alternating sensor/area prediction states: **"benefit"** and **"sacrifice"**.
- A sensor is considered to be in a benefit state when its prediction performance improves relative to the defined threshold, while deterioration corresponds to the sacrifice state.
- The state information is used to guide subsequent sensor sampling.

- **Spatial Topology / Graph Construction:** The traffic environment is represented as a road network graph `G = (V,E)`, where `V` represents road sensors and `E` represents binary-valued road connectivity. FairTP does not introduce a new graph-construction mechanism; it operates with the existing road-network structure of the underlying ST model.
- **Spatial Encoding Module:** No single new spatial encoder is introduced. FairTP uses a **replaceable spatio-temporal traffic prediction model**, allowing existing models such as DCRNN, AGCRN, GWNET, ASTGCN, DSTAGNN, DCGRN, and D2STGNN to serve as the underlying ST model.
- **Temporal Encoding Module:** No independent temporal encoder is proposed by FairTP. Temporal modelling is delegated to the replaceable underlying ST forecasting model.
- **State Identification Module:** Sensor prediction performance is evaluated using MAPE relative to a threshold derived from the preceding training process. A discriminator is used to identify whether sensors are in benefit or sacrifice states.
- **State-Guided Balanced Sampling:** Stratified sampling first selects sensors proportionally across different regions. Sampling probabilities are then dynamically adjusted according to sensor states and regional imbalance. Sensors identified as being in a sacrifice state receive increased sampling opportunities.
- **Fairness Metrics:** RSF measures regional prediction-performance disparity at a given time, while SDF measures discrepancies between sensor states over a defined period.
- **Loss Function & Optimization:** FairTP combines prediction accuracy with fairness objectives through a composite loss containing the **MAE accuracy loss, RSF loss, and SDF loss**. The discriminator is additionally trained for sensor-state identification.
- The overall training process combines **prediction accuracy, shortdated regional fairness, prolonged sensor-level fairness, and adaptive sampling**.

# 5. Main Contributions

- The authors systematically investigate **prolonged fairness in traffic prediction** and introduce two fairness definitions: **region-based static fairness (RSF)** and **sensor-based dynamic fairness (SDF)**.
- They propose the **FairTP framework**, which integrates with existing traffic prediction models to improve fairness with limited accuracy degradation.
- They introduce a **state identification module** that distinguishes benefit and sacrifice states for traffic sensors.
- They design a **state-guided balanced sampling strategy** to mitigate performance disparities caused by uneven sensor distributions.
- Extensive experiments on two real-world traffic datasets demonstrate improvements in predictive fairness while maintaining or improving forecasting performance.

# 6. Experimental Setup

- **Datasets:**
  - **HK Didi dataset:** October 1, 2020 to March 31, 2021; six months of trajectory data; **938 road sensors**; traffic speed is used as the traffic feature.
  - **SD dataset:** San Diego County, **2019**; **716 road sensors**; sensor readings are used as traffic features.
  - HK is divided into **13 regions**.
  - SD is divided into **12 regions**.
  - The datasets are chronologically divided into training, validation, and testing sets using a **6:2:2 ratio**.
  - Sampling frequency: **Not discussed**.
- **Data Scope & Exclusions:**
  - HK date range: **October 1, 2020–March 31, 2021**.
  - SD: **2019**.
  - Chronological train/validation/test split: **6:2:2**.
  - Holiday exclusions: **Not discussed**.
  - Explicit seasonal exclusions: **Not discussed**.
  - Speed-threshold filtering: **Not discussed**.
  - Missing-data imputation procedure: **Not discussed**.
- **Evaluation Metrics:**
  - MAE
  - RMSE
  - MAPE
  - RSF
  - SDF
- **Baseline Models:**
  - **Traffic forecasting models:** DCRNN, AGCRN, GWNET, ASTGCN, STGODE, DSTAGNN, DCGRN, D2STGNN.
  - **Fairness-oriented baselines:** FairST and SA-Net.
- Prediction horizons of 15, 30, and 60 minutes: **Not discussed**.

# 7. Experimental Findings & Performance Breakdown

- FairTP consistently improves fairness across the evaluated traffic forecasting models.
- On the **HK dataset**, the paper reports **RSF improvements of 82.2%–127.2%** and **MAE reductions of 4.6%–6.7%** in the comparison with fairness baselines.
- On the **SD dataset**, the reported RSF improvements range from **19.0% to 71.9%**, with MAE performance improvements ranging from **27.8% to 44.3%** in the reported fairness comparison.
- In the broader accuracy–fairness trade-off analysis, HK MAE changes range from approximately **-1.14% to +13.03%**, while RSF improvements reach **27.84%–94.00%**.
- On SD, MAE performance improves by **4.86%–50.28%**, while RSF improves by **42.16%–89.01%**.
- The authors conclude that FairTP can improve fairness while minimizing prediction-performance degradation.
- Regional analysis shows improved prediction performance in underprivileged regions with fewer sensors.
- On HK, improvements are reported in underprivileged regions including **r8, r12, and r13**, while privileged regions such as **r1, r2, and r10** remain largely unaffected.
- On SD, improvements are reported in underprivileged regions including **r6, r7, r11, and r12**, while privileged regions such as **r1 and r2** do not show observable deterioration.
- The case study on **region r12** demonstrates close agreement between predicted traffic and ground-truth traffic, including sharp traffic fluctuations.

### Ablation Study Findings

- Three components are individually removed:
  - **noS:** removes the RSF loss.
  - **noD:** removes the SDF loss.
  - **noAS:** removes state-guided sampling and uses fixed stratified sampling.
- Removing any of the three components increases prediction error or worsens RSF/SDF performance.
- Removing **SDF** has the most significant impact, with the SDF measure deteriorating substantially.
- Removing **RSF** decreases both RSF and SDF performance, supporting the contribution of the static regional fairness constraint.
- Replacing adaptive state-guided sampling with fixed stratified sampling changes model performance, indicating that adaptive sampling contributes to the fairness-oriented framework.
- Parameter analysis reports the best fairness performance at **Td = 3** when dynamic time length is varied from 2 to 5.
- The reported optimal number of sampled sensors is **Nsam = 200** when the sampled number is varied from 100 to 300.
- Prediction results across explicit short-, medium-, and long-term horizons: **Not discussed**.

# 8. Strengths

- Addresses an underexplored fairness problem arising specifically from **uneven traffic-sensor deployment**.
- Moves beyond static fairness by introducing **sensor-based dynamic fairness over an extended period**.
- Provides both **regional and sensor-level fairness measures** through RSF and SDF.
- Does not require the underlying forecasting model to be replaced; the ST forecasting component is explicitly **replaceable**.
- Combines fairness objectives with **adaptive sensor sampling**, rather than relying solely on fairness regularization.
- Specifically targets underprivileged, sparsely sensed regions through state-guided sampling.
- Experimental results demonstrate fairness improvements without substantial prediction degradation.
- Regional experiments provide evidence that disadvantaged regions can improve without observable deterioration in several privileged regions.
- Ablation experiments support the importance of the major framework components.
- The approach is evaluated across multiple established traffic forecasting architectures rather than a single backbone.

# 9. Remaining Limitations & Vulnerabilities

## Author-Mentioned Limitations

**Not discussed.**

The provided Conclusion describes the framework, fairness definitions, integration with existing traffic models, and experimental findings, but does not explicitly identify a specific unresolved limitation, failure mode, or future-work problem.

## Broader Impact & Ethical Considerations

**Not discussed.**

No dedicated **Broader Impact**, **Ethics**, **Societal Implications**, or **Environmental Impact** section is present in the provided paper material. Although the paper discusses fairness, regional decision-making, and potential effects on equity and quality of life as motivation, it does not provide a separate broader-impact or ethics analysis.

# 10. Sensor-Centric & Causal Perspective

- **Sensor Reliability & Data Quality:** The paper addresses **uneven sensor distribution and resulting data imbalance**, and uses sensor prediction-performance states to guide sampling. It does not discuss noisy sensors, zero-dropouts, stuck readings, missing-data imputation, sensor drift, or hardware degradation.
- **Sensor State:** Sensors are explicitly assigned prediction-performance states of **benefit** or **sacrifice**, and these states influence subsequent sampling.
- **Spatial Equity & Disparity:** **Yes.** Regional prediction disparities are explicitly evaluated. The paper analyzes underprivileged regions with fewer sensors and compares their performance against more privileged regions.
- **Causal Reasoning Level:** **Pearl Level 1 — Observational Prediction.** The framework learns from observed traffic and sensor prediction-performance states. The paper does not formulate causal interventions or counterfactual inference.

# 11. Terminology & Acronym Collision Notes

- **DCRNN** — Diffusion Convolutional Recurrent Neural Network; established terminology in traffic forecasting and should be accompanied by its full name when first introduced.
- **AGCRN** — Adaptive Graph Convolutional Recurrent Network; established traffic-forecasting model name.
- **GWNET** — Graph WaveNet; established traffic-forecasting model terminology.
- **ASTGCN** — Attention-Based Spatial-Temporal Graph Convolutional Network; established model acronym.
- **DSTAGNN** — Dynamic Spatial-Temporal Aware Graph Neural Network; established model acronym.
- **DCGRN** — Dynamic Convolutional Graph Recurrent Network; should be disambiguated by its full name.
- **D2STGNN** — Decoupled Dynamic Spatial-Temporal Graph Neural Network; highly specific but should still be expanded at first occurrence.
- **FairST** and **SA-Net** are fairness-oriented model names and should be accompanied by their cited paper/context when used outside this literature review.
- **RSF** and **SDF** are central FairTP terminology but are potentially generic acronyms; their full definitions should always accompany first use:
  - RSF = **Region-based Static Fairness**
  - SDF = **Sensor-based Dynamic Fairness**
- **FairTP** is the paper-specific framework name and should be distinguished from unrelated uses of "fair" or "TP" in other literature.

# 12. Keywords (20–30)

- Traffic prediction
- Traffic forecasting
- Fair traffic prediction
- Prolonged fairness
- Region-based static fairness
- Sensor-based dynamic fairness
- RSF
- SDF
- Sensor-centric forecasting
- Traffic sensor distribution
- Uneven sensor deployment
- Data imbalance
- Regional prediction disparity
- Sensor state identification
- Benefit state
- Sacrifice state
- State-guided sampling
- Balanced sensor sampling
- Stratified sampling
- Spatio-temporal forecasting
- Graph neural networks
- Spatio-temporal graph neural networks
- Adaptive sampling
- Fairness-aware machine learning
- Prediction fairness
- Regional equity
- Sensor-level fairness
- Fairness-accuracy trade-off
- Under-sensored regions
- Intelligent transportation systems
```
