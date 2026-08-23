```markdown
# Literature Review

## Paper

### Title

**FairSTG: Countering Performance Heterogeneity via Collaborative Sample-Level Optimization**

### Authors

**Gengyu Lin, Zhengyang Zhou, Qihe Huang, Kuo Yang, Shifen Cheng, and Yang Wang**

### Year

**2025**

### Venue / Journal / Conference

**IEEE Transactions on Mobile Computing, Vol. 24, No. 5, May 2025**

---

# Objective

- The paper addresses **performance heterogeneity and prediction unfairness in spatiotemporal forecasting**.
- The central problem is that conventional spatiotemporal forecasting models primarily optimize **overall forecasting accuracy**, while prediction quality can differ substantially across individual samples, sensors, spatial regions, and timestamps.
- The objective is to develop a **model-independent fairness-aware spatiotemporal graph learning framework** that reduces sample-level prediction disparity while maintaining comparable forecasting performance.
- The authors identify that some samples are inherently more difficult for the forecasting model and consequently receive consistently poorer predictions.
- The paper therefore focuses on improving the treatment of **challenging samples** rather than optimizing only the average forecasting error.
- The motivation is supported by observed performance heterogeneity in spatiotemporal datasets, including substantial variance in prediction errors across samples.

### Important Note

> **Core problem:** Existing ST-GNNs can achieve good overall forecasting accuracy while still producing disproportionately high errors for particular samples or sensors.

---

# Existing Systems

## 1. Traditional Statistical Forecasting

The paper discusses traditional forecasting methods including:

- **ARIMA**
- **VAR**

These methods provide relatively simple and interpretable forecasting solutions.

### Limitation

They cannot simultaneously capture the **spatial and temporal dependencies** required by modern spatiotemporal forecasting problems.

---

## 2. Spatiotemporal Graph Neural Networks

Modern forecasting approaches use graph-based deep learning to jointly model spatial and temporal dependencies.

Representative models discussed in the paper include:

- **DCRNN**
- **STGCN**
- **AGCRN**
- **MTGNN**
- **D²STGNN**
- **ST-SSL**
- **HA-STGN**

These approaches model:

- spatial relationships,
- temporal dependencies,
- dynamic graph relationships,
- heterogeneous spatiotemporal patterns,
- non-Euclidean dependencies.

Examples include adaptive graph learning, graph convolution, self-supervised learning, directional road relationships, and time-aware graph attention.

---

## 3. Conventional Fairness-Aware Machine Learning

The paper discusses three major categories:

### Pre-processing
Training data are modified or rebalanced before model training.

### In-processing
Fairness constraints or fairness objectives are incorporated into model optimization.

### Post-processing
Model outputs or learned representations are modified after training.

These methods commonly address fairness using explicit sensitive attributes such as:

- gender,
- race,
- group identity.

---

## 4. Fairness in Mobile Computing

Related research has considered fairness in:

- edge computing,
- crowdsensing,
- intelligent traffic routing,
- resource allocation,
- task assignment.

However, these approaches primarily concern **allocation or system-level fairness**, rather than fairness in forecasting performance.

---

# Limitations of Existing Systems

## 1. Aggregate Accuracy Hides Performance Disparity

Existing forecasting models generally focus on metrics such as:

- MAE,
- MAPE,
- RMSE.

These metrics summarize overall prediction performance but may conceal large differences among individual samples.

The paper reports that:

- **MAE variance can be approximately 14 times the MAE on METR-LA**.
- **MAE variance can be approximately 9 times the MAE on PEMS-BAY**.

This demonstrates that a satisfactory average error does not necessarily imply consistent performance across samples.

### Important Note

> **Overall accuracy does not guarantee sample-level fairness.**

---

## 2. Spatial Performance Heterogeneity

The paper observes that sensors are more concentrated in **city centers than suburban or marginal areas**.

This creates differences in data representation and learning difficulty.

Underrepresented regions may have:

- fewer observations,
- poorer representation,
- greater learning difficulty,
- poorer prediction performance.

---

## 3. Temporal Performance Heterogeneity

Prediction performance can vary significantly at different timestamps **even for the same sensor**.

Therefore, performance disparity is not purely spatial.

The same sensor can have:

- relatively low prediction error under one temporal condition,
- substantially higher error under another condition.

---

## 4. Challenging Samples Are Not Explicitly Targeted

Existing forecasting models generally do not explicitly distinguish between:

- easy-to-learn samples,
- challenging-to-learn samples.

The paper identifies a subgroup of difficult samples whose forecasting errors are substantially higher.

In the analysis:

- **Top 30%** are treated as easy samples.
- **Bottom 30%** are treated as challenging samples.

Conventional aggregate optimization can therefore devote insufficient attention to the difficult subgroup.

---

## 5. Existing ST-GNNs Do Not Explicitly Optimize Prediction Fairness

Existing ST-GNNs effectively model:

**spatial dependencies + temporal dependencies**

but do not explicitly optimize:

**prediction-performance consistency across samples.**

Thus, the research problem is not that ST-GNNs cannot forecast traffic; rather, they do not explicitly address **unequal prediction quality**.

---

## 6. Conventional Fairness Methods Require Sensitive Attributes

Traditional fairness-aware machine learning commonly relies on explicitly defined sensitive attributes.

However, traffic forecasting datasets do not naturally provide equivalent sensitive-group labels for spatiotemporal samples.

Therefore, conventional fairness mechanisms cannot be directly transferred to this setting.

---

## 7. Limitations of Existing Similarity-Based Approaches

Existing methods may use:

- node-level similarity,
- time-invariant node embeddings.

The paper identifies that these approaches can overlook **temporal heterogeneity** and may not provide sufficiently fine-grained **sample-level representation enhancement**.

---

# Research Gaps Identified

The paper addresses several gaps:

1. Lack of explicit modeling of **sample-level prediction-performance heterogeneity** in spatiotemporal forecasting.
2. Lack of fairness-aware forecasting methods that operate **without explicit sensitive attributes**.
3. Lack of mechanisms for explicitly identifying **challenging spatiotemporal samples**.
4. Lack of sample-level collaborative representation enhancement for difficult observations.
5. Lack of a dedicated fairness objective based on **prediction-error disparity**.
6. Existing mobile-computing fairness approaches focus mainly on resource allocation, routing, and assignment rather than forecasting-performance fairness.
7. Existing similarity approaches do not sufficiently account for temporal heterogeneity at the sample level.

### Important Note

> **The central research gap is not spatial or temporal dependency modeling itself. It is the lack of explicit optimization for prediction-performance heterogeneity.**

---

# Proposed System

The proposed framework is **FairSTG (Fairness-aware SpatioTemporal Graph learning)**.

FairSTG is presented as a **model-independent framework** that can be integrated with different spatiotemporal forecasting backbones.

The framework consists of:

1. **Spatiotemporal Feature Extractor**
2. **Fairness Recognizer**
3. **Collaborative Feature Enhancement**
4. **Output Module**

The overall strategy is:

**Spatiotemporal representation**

→ identify learning difficulty

→ identify challenging samples

→ find similar well-learned samples

→ obtain compensatory representations

→ enhance challenging representations

→ apply fairness-aware optimization

→ generate forecasts.

---

# Proposed Architecture

## Spatial Topology / Graph Construction

- FairSTG operates on an **input spatiotemporal graph**.
- The framework is model-independent and can use different ST-GNN backbones.
- The experiments use:
  - **MTGNN**
  - **D²STGNN**
- The fairness recognizer uses graph-based representation learning and adaptive topology learning to capture relevant relationships.

### Important Note

> FairSTG does **not replace the underlying forecasting backbone** with one completely new ST-GNN. It adds a fairness-oriented framework around an existing spatiotemporal forecasting architecture.

---

## Spatial Encoding

- The selected ST-GNN backbone extracts spatial relationships and spatiotemporal representations.
- **MTGNN** and **D²STGNN** are used as the principal forecasting backbones.
- The fairness recognizer uses a **GCN-based architecture** to capture spatial correlations for identifying learning difficulty.

---

## Temporal Encoding

- Temporal dependencies are handled through the selected forecasting backbone.
- Temporal information is also used by the fairness recognizer.
- Relevant information includes:
  - sampling time,
  - sequence statistics,
  - temporal context,
  - weekday,
  - weather.

### Important Note

> FairSTG is primarily a **fairness-enhancement framework**, rather than a new standalone temporal encoder.

---

## Fairness Recognizer

- A **self-supervised fairness recognizer** identifies easy and challenging samples.
- It avoids requiring conventional explicit sensitive attributes.
- Samples are analyzed according to their prediction difficulty.
- The recognizer uses spatiotemporal and auxiliary information to learn the distinction between easy and challenging samples.

---

## Collaborative Feature Enhancement

- FairSTG searches for **well-learned samples with similar patterns** to challenging samples.
- Their representations are used as **compensatory representations**.
- These representations are aggregated and incorporated into the challenging sample representation.
- An adaptive representation **mix-up** mechanism is used to enhance the difficult sample.

### Important Note

> The central intuition is **"well-learned samples help difficult samples learn better."**

---

## Loss Function & Optimization

The framework jointly considers:

1. **Reweighted forecasting loss**
2. **Sample-level fairness loss**
3. **Self-supervised fairness-recognizer loss**

### Reweighted Forecasting Loss

Higher-error/challenging samples receive greater attention.

### Fairness Loss

Prediction-error variance is used to reduce performance disparity.

### Self-Supervised Loss

Trains the fairness recognizer to distinguish easy and challenging samples.

### Overall Objective

The framework jointly balances:

**Forecasting Accuracy + Fairness**

---

# Main Contributions

- Identifies and systematically studies **performance heterogeneity in spatiotemporal learning**.
- Proposes the **model-independent FairSTG framework** for reducing sample-level prediction unfairness.
- Introduces a **self-supervised fairness recognizer** for identifying challenging samples without explicit sensitive attributes.
- Introduces **collaborative feature enhancement** to transfer useful representations from well-learned samples to challenging samples.
- Uses **adaptive representation mix-up** to improve challenging-sample representations.
- Introduces a **sample-level fairness objective** based on prediction-error disparity.
- Demonstrates improved fairness while maintaining comparable forecasting performance across multiple datasets and forecasting backbones.

---

# Experimental Setup

## Datasets

### METR-LA

- **207 sensors/nodes**
- **34,272 samples**
- **5-minute sampling interval**
- Traffic-speed measurements from Los Angeles County highways.
- Reported period: **March 2012 – June 2012**

### PEMS-BAY

- **325 sensors/nodes**
- **52,116 samples**
- **5-minute sampling interval**
- Traffic-speed measurements from the Bay Area.
- Reported period: **January 2017 – May 2017**

### KnowAir

- **184 nodes**
- **11,688 samples**
- **3-hour sampling interval**
- PM2.5 observations from major Chinese cities.
- Reported period: **September 2016 – January 2017**

### ETT

- **7 nodes**
- **17,420 samples**
- **1-hour sampling interval**

---

## Data Scope & Exclusions

- Dataset time periods and sampling frequencies are reported above.
- Specific excluded holidays are **Not discussed**.
- Specific excluded seasons are **Not discussed**.
- Explicit speed-threshold filtering is **Not discussed**.
- Specific sensor-removal criteria are **Not discussed**.
- Explicit sensor-failure filtering is **Not discussed**.
- Detailed missing-data preprocessing is **Not discussed**.

---

## Evaluation Metrics

### Forecasting Metrics

- **MAE**
- **MAPE**
- **RMSE**

### Fairness Metrics

- **MAE variance**
- **MAPE variance**

### Additional Analysis

- Challenging-sample forecasting performance
- Easy-sample forecasting performance
- Sensor-level prediction improvement
- Fairness-recognizer accuracy

---

## Baseline Models

- **DCRNN**
- **STGCN**
- **AGCRN**
- **MTGNN**
- **D²STGNN**
- **ST-SSL**

### Main Backbones

- **MTGNN**
- **D²STGNN**

---

# Experimental Findings

## Overall Fairness Improvement

FairSTG reports fairness improvements ranging from:

**0.21%–20.05%**

with:

**16 of 24 comparisons achieving more than 4% improvement.**

---

## Horizon-12 Results

Reported MAE improvements:

| Dataset | MAE Improvement |
|---|---:|
| METR-LA | **5.59%** |
| PEMS-BAY | **8.15%** |
| KnowAir | **4.43%** |
| ETT | **2.76%** |

Reported MAPE improvements:

| Dataset | MAPE Improvement |
|---|---:|
| METR-LA | **12.36%** |
| PEMS-BAY | **12.01%** |
| KnowAir | **11.77%** |
| ETT | **12.98%** |

---

## Forecasting Accuracy

The paper reports forecasting improvements ranging from:

**0.43%–9.79%**

in the reported horizon-based comparisons.

However, forecasting performance also decreases in some cases:

- **0.52%–2.90%** degradation in reported cases.
- A maximum reported degradation of **7.23% on ETT horizon-3**.

### Important Note

> This demonstrates an explicit **fairness–accuracy trade-off**.

Increasing emphasis on fairness does not guarantee improved raw forecasting accuracy.

---

# Challenging-Sample Findings

- FairSTG improves performance on the **challenging sample subgroup**.
- Performance on the easy subgroup remains generally comparable.
- This supports the paper's argument that collaborative feature enhancement can compensate for poorer representations of difficult samples.

---

# Sensor-Level Findings

The PEMS-BAY analysis is particularly important for sensor-centric research.

The paper reports that sensors around **transportation hubs** can experience more serious prediction errors because of complex traffic patterns and greater learning difficulty.

Five nodes with the largest reported improvements are:

| Node | Improvement |
|---:|---:|
| **10** | **1.77** |
| **46** | **1.13** |
| **150** | **1.01** |
| **45** | **0.95** |
| **29** | **0.87** |

### Important Note

> This provides direct evidence that **forecasting performance disparity can occur at individual sensor locations**.

---

# Fairness Recognizer Findings

The self-supervised fairness recognizer achieves:

- **Training accuracy: 72.20%–87.40%**
- **Testing accuracy: 66.19%–81.49%**

The GCN-based recognizer generally performs better than the linear alternative.

---

# Ablation Study Findings

The paper evaluates:

### FairSTG

Full model.

### FairSTG-w/o-FE

Removes:

- fairness recognizer,
- collaborative feature enhancement.

### FairSTG-w/o-FO

Removes:

- fairness objective.

The results show that both:

- **collaborative feature enhancement**
- **explicit fairness optimization**

contribute to improved fairness.

### Dataset-Dependent Observation

For:

- **METR-LA**
- **PEMS-BAY**

the fairness objective has a particularly prominent role.

For:

- **KnowAir**
- **ETT**

collaborative feature enhancement has a more prominent role.

### Important Note

> The ablation study indicates that FairSTG's two major mechanisms are complementary rather than redundant.

---

# Hyperparameter Findings

The fairness coefficient is evaluated at:

**0.01, 0.1, 0.5, 1.0, 1.5**

The number of compensatory samples is evaluated at:

**5, 10, 20**

The paper observes:

- increasing the fairness coefficient improves fairness but can reduce forecasting accuracy;
- using too many compensatory samples can introduce noise into the mix-up representation;
- therefore, fairness and forecasting accuracy require careful balancing.

---

# Strengths

- Addresses a relatively underexplored problem of **sample-level prediction-performance heterogeneity**.
- Moves beyond aggregate forecasting accuracy.
- Explicitly considers fairness across individual samples.
- Does not require conventional sensitive attributes.
- Model-independent design allows integration with different ST-GNN backbones.
- Combines **representation-level enhancement** and **objective-level fairness optimization**.
- Explicitly targets challenging samples.
- Demonstrates results across multiple datasets.
- Evaluates both forecasting accuracy and fairness.
- Includes:
  - challenging-sample analysis,
  - sensor-level analysis,
  - ablation studies,
  - fairness-recognizer analysis,
  - hyperparameter analysis.
- Provides explicit evidence that different sensors can experience substantially different forecasting quality.

---

# Remaining Limitations

## Author Mentioned Limitations

The authors identify the following unresolved problems and future directions:

1. **Computational efficiency:**  
   Future work should develop an **approximate estimation algorithm for sample-pair similarity** to obtain a more computationally efficient collaborative learning paradigm.

2. **Forecasting-horizon adaptation:**  
   Future work should investigate **adaptive forecasting strategies for different horizons** to better balance forecasting performance and fairness.

3. **Root causes of unfairness:**  
   The authors identify the need to further investigate the **root causes of prediction unfairness from both model and data aspects**.

4. **Dataset adaptability:**  
   Future work should develop **data-adaptive fairness learning** to accommodate different datasets.

5. **Fairness–accuracy trade-off:**  
   Increasing the fairness emphasis can improve fairness while reducing forecasting accuracy.

6. **Compensatory-sample sensitivity:**  
   Excessive compensatory samples can introduce noise into the mix-up representations and degrade performance.

---

## Broader Impact / Ethical Considerations

**Not discussed.**

No dedicated **Broader Impact**, **Ethics**, or **Societal Implications** section was identified in the reviewed paper material.

---

# Sensor-Centric Perspective

## Sensor Reliability

The paper does **not** explicitly model:

- sensor health,
- hardware degradation,
- sensor failure,
- sensor drift,
- sensor confidence,
- dynamic sensor quality,
- stuck readings,
- abnormal sensor readings.

---

## Missing Data

Explicit mechanisms for:

- missing-value imputation,
- missing-sensor detection,
- sensor dropout handling,

are **Not discussed** in the highlighted material.

---

## Sensor Anomalies

Explicit detection of:

- stuck sensors,
- faulty sensors,
- abnormal sensor measurements,

is **Not discussed**.

---

## Data Quality

The paper considers:

- data representation,
- spatial under-representation,
- learning difficulty,
- spatiotemporal heterogeneity.

However, it does not explicitly treat **measurement quality as a sensor state**.

---

## Sensor-Level Performance

The paper does explicitly demonstrate **sensor-level prediction disparity**.

In particular:

- transportation-hub sensors can exhibit higher errors;
- individual sensors can benefit from FairSTG;
- spatial distribution of sensors affects representation and learning difficulty.

### Important Distinction

> **FairSTG identifies difficult-to-predict samples; it does not establish that the corresponding physical sensor is unreliable.**

A sensor can be difficult to predict because its traffic environment is complex rather than because the sensor itself is malfunctioning.

This distinction is highly relevant to a sensor-centric dissertation.

---

# Relevance to Sensor-Centric Traffic Forecasting

FairSTG is highly relevant because it establishes that **prediction performance is not uniform across sensors or temporal observations**.

Its relevance can be summarized as:

| FairSTG | Sensor-Centric Framework |
|---|---|
| Sample-level prediction fairness | Sensor-level reliability |
| Prediction-performance heterogeneity | Sensor/data-quality heterogeneity |
| Challenging samples | Potentially unreliable/problematic sensor observations |
| Prediction-error variance | Reliability/error-aware forecasting |
| Collaborative representation enhancement | Reliability-aware representation/forecasting |
| Fairness objective | Sensor-quality-aware objective |
| Model-performance perspective | Sensor-condition perspective |

FairSTG therefore provides a strong foundation for the **performance-disparity dimension** of sensor-centric forecasting.

However, it leaves a distinct gap concerning:

- physical sensor reliability,
- sensor health,
- sensor degradation,
- sensor drift,
- missing measurements,
- anomalous readings,
- dynamic sensor confidence,
- measurement-quality-aware forecasting.

### Key Research Gap for Sensor-Centric Work

> **FairSTG focuses on whether a sample is difficult for the forecasting model, whereas a sensor-centric framework can additionally investigate whether the underlying sensor or its measurements are responsible for the difficulty.**

This distinction should be preserved when positioning a future dissertation framework against FairSTG.

---

# Causal Perspective

- The framework operates at **Pearl Level 1 — Observational Prediction**.
- It learns from observed spatiotemporal data and prediction errors.
- No explicit intervention or simulation framework is introduced.
- **Pearl Level 2 — Intervention/Simulation:** Not discussed.
- **Pearl Level 3 — Counterfactual Abduction:** Not discussed.

---

# Terminology Notes

- **FairSTG** — the specific framework proposed by the paper.
- **ST-GNN / STG** — generic terminology for spatiotemporal graph neural networks and should not be treated as a unique architecture.
- **MTGNN** — established spatiotemporal graph forecasting model used as a FairSTG backbone.
- **D²STGNN** — established spatiotemporal graph forecasting model used as a FairSTG backbone.
- **DCRNN** — established traffic forecasting architecture used as a baseline/related method.
- **STGCN** — established spatiotemporal forecasting architecture.
- **AGCRN** — established adaptive graph convolutional recurrent forecasting architecture.
- **ST-SSL** — established spatiotemporal self-supervised learning approach.
- **Fairness Recognizer** — FairSTG component for identifying learning difficulty.
- **Collaborative Feature Enhancement** — FairSTG mechanism for transferring representations from well-learned to challenging samples.
- **MAE Variance / MAPE Variance** — fairness-related measures for prediction-performance disparity.

### Important Terminology Note

**STG, ST-GNN, and fairness-related abbreviations are generic enough to require the full method name when citing them in a dissertation.**

---

# Keywords (20–30)

- FairSTG
- Fairness-aware spatiotemporal learning
- Spatiotemporal forecasting
- Traffic forecasting
- Spatiotemporal graph neural networks
- ST-GNN
- Prediction fairness
- Sample-level fairness
- Performance heterogeneity
- Sensor-level disparity
- Challenging samples
- Easy samples
- Fairness recognizer
- Self-supervised learning
- Collaborative feature enhancement
- Compensatory representations
- Adaptive representation mix-up
- Fairness-aware optimization
- Reweighted forecasting loss
- Prediction-error variance
- MAE variance
- MAPE variance
- Adaptive graph learning
- MTGNN
- D²STGNN
- DCRNN
- STGCN
- AGCRN
- Sensor under-representation
- Fairness–accuracy trade-off

---

# Important Notes for Dissertation Literature Review

## 1. Core Problem to Remember

**FairSTG is fundamentally about prediction-performance heterogeneity, not sensor reliability.**

The paper asks:

> Why do some spatiotemporal samples receive worse predictions than others, and how can this disparity be reduced?

It does **not primarily ask**:

> Is the sensor itself reliable or faulty?

---

## 2. Strongest Existing-System Limitation

The strongest limitation of previous ST-GNNs identified by FairSTG is:

> **They optimize overall forecasting accuracy while overlooking unequal prediction performance across individual samples, sensors, spatial regions, and temporal contexts.**

---

## 3. Strongest Evidence Relevant to Sensors

The PEMS-BAY analysis is particularly important:

> Sensors around transportation hubs show more serious prediction errors because of complex traffic patterns and higher learning difficulty.

This demonstrates that **sensor location/context can be associated with forecasting difficulty**.

---

## 4. FairSTG's Main Novelty

The novelty is not simply another ST-GNN architecture.

Its major novelty is the combination of:

**challenging-sample recognition**

+

**collaborative feature enhancement**

+

**fairness-aware optimization**

to reduce sample-level prediction disparity.

---

## 5. Most Important Limitation of FairSTG

The authors themselves identify future work concerning:

- computational efficiency,
- horizon-adaptive fairness,
- root causes of unfairness,
- data-adaptive fairness.

For a **sensor-centric dissertation**, an additional distinction supported by the paper's scope is:

> FairSTG's fairness signal is based on prediction difficulty/performance rather than an explicit sensor-reliability state.

This should be stated carefully as a **scope distinction**, not as a claim that the authors explicitly criticized sensor reliability.

---

## 6. Most Useful Positioning Sentence

> **FairSTG demonstrates that conventional spatiotemporal forecasting models can exhibit substantial sample- and sensor-level prediction-performance heterogeneity despite satisfactory aggregate accuracy. By identifying challenging samples and enhancing their representations through collaborative feature transfer and fairness-aware optimization, FairSTG reduces prediction disparity without requiring explicit sensitive attributes. However, its fairness formulation remains centered on prediction performance rather than the underlying measurement quality or reliability of individual sensors, leaving sensor-condition-aware forecasting as a distinct research direction.**

---

## 7. Literature Review Classification

**Paper Category:** Fairness-aware Spatiotemporal Forecasting

**Primary Focus:** Sample-level prediction fairness

**Secondary Focus:** Sensor/spatial performance disparity

**Core Architecture:** Fairness framework + existing ST-GNN backbone

**Main Backbones:** MTGNN, D²STGNN

**Primary Novelty:** Collaborative sample-level optimization

**Fairness Signal:** Prediction-performance heterogeneity

**Sensor Reliability Modeling:** Not discussed

**Sensor Fault Detection:** Not discussed

**Missing Data Handling:** Not discussed

**Sensor Drift Modeling:** Not discussed

**Causal Reasoning:** Observational / Pearl Level 1

**Key Trade-Off:** Fairness ↔ Forecasting Accuracy

**Key Future Work:** Efficient similarity estimation, horizon-adaptive fairness, root-cause analysis, and data-adaptive fairness
```
