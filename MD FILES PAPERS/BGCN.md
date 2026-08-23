# Literature Review

## Paper 4

### Title
**Bayesian Graph Convolutional Network for Traffic Prediction**

### Authors
Jun Fu, Wei Zhou, Zhibo Chen

### Year
2021

---

# Objective

The paper aims to improve traffic forecasting by introducing a **Bayesian Graph Convolutional Network (BGCN)** that incorporates uncertainty into graph structure learning. Unlike conventional graph convolution methods that use manually defined or deterministically learned graph structures, the proposed framework jointly utilizes **observed road topology** and **historical traffic data** to infer a probabilistic graph representation. The model is designed to capture both **positive and negative spatial relationships** while preserving prior knowledge from the physical road network. Furthermore, BGCN is proposed as a **plug-and-play module** that can be integrated into existing graph-based traffic forecasting models to improve prediction accuracy and generalization.

---

# Existing Systems

The paper reviews representative traffic forecasting approaches from four categories.

## Traditional Statistical Methods

### Historical Average (HA)
- Uses seasonal historical averages for prediction.
- Assumes repetitive traffic patterns.

### ARIMA
- Linear autoregressive time-series forecasting model.

### Vector Auto Regression (VAR)
- Multivariate linear time-series forecasting model.
- Captures correlations among traffic variables.

### Support Vector Regression (SVR)
- Kernel-based regression model.
- Predicts traffic conditions independently for each road.

---

## Deep Learning Methods

### Stacked Autoencoder (SAE)
- Learns latent traffic representations.

### Deep Belief Network (DBN)
- Learns hierarchical traffic features.

### CNN-based Models
- Learn spatial-temporal representations using convolution.

### RNN-based Models
- Learn temporal dependencies from sequential observations.

### GRU Encoder–Decoder (GRU-ED)
- Encoder-decoder architecture built with stacked GRUs.

---

## Graph-based Traffic Forecasting Methods

### Temporal Graph Convolutional Network (TGCN)
- Combines graph convolution with recurrent neural networks.

### Diffusion Convolutional Recurrent Neural Network (DCRNN)
- Employs diffusion graph convolution within an encoder-decoder recurrent architecture.

### Spatio-Temporal Graph Convolutional Network (STGCN)
- Combines graph convolution with gated temporal convolution.

### Graph WaveNet
- Integrates graph convolution with dilated causal convolution.
- Learns latent graph structures through an attention mechanism.

### Spatial-Temporal Synchronous Graph Convolutional Network (STSGCN)
- Captures localized spatial-temporal dependencies through synchronous graph convolution.

### Adaptive Graph Convolutional Recurrent Network (AGCRN)
- Learns adaptive graph structures using node-adaptive parameter learning and a data-adaptive graph generation module.

---

## Bayesian Graph Neural Networks

Existing Bayesian GCNs:

- Introduce uncertainty into graph neural networks.
- Primarily estimate uncertainty in model weights.
- Mainly target semi-supervised node classification.

---

# Limitations of Existing Systems

## Traditional Statistical Methods

- Assume stationary traffic processes.
- Fail to capture nonlinear traffic dynamics.
- Ignore spatial dependencies among roads.
- Process roads independently.
- Perform poorly under complex traffic conditions and datasets with large variance.

---

## Deep Learning Methods

- Primarily model temporal dependencies.
- Limited capability for modeling spatial relationships.
- Do not fully exploit graph-structured road networks.

---

## Existing Graph Convolution Networks

- Use manually predefined graph structures.
- Depend on road topology or Euclidean distance.
- Cannot adequately capture latent spatial relationships.
- Physical road topology is not an optimal representation of traffic correlations.

---

## Adaptive Graph Learning Methods

The paper explicitly identifies three major limitations.

1. Learn graph structures from scratch while ignoring prior road topology.
2. SoftMax attention suppresses negative spatial relationships.
3. Graph structures remain deterministic without uncertainty modeling.

---

## Existing Bayesian GCNs

- Mainly model uncertainty in network weights.
- Do not estimate uncertainty in graph structures.
- Designed for node classification instead of traffic forecasting.

---

## Non-parametric Bayesian GCN

- Learns adjacency through neighborhood similarity.
- Produces symmetric graph structures.
- Cannot effectively represent asymmetric traffic influence.
- Does not learn graph structures end-to-end.
- Ignores dependence between graph structure and training data.

---

# Research Gaps Identified

The paper identifies several unresolved issues.

- Existing methods do not combine prior road topology with learned graph structures.
- Negative spatial correlations are largely ignored.
- Graph uncertainty is rarely modeled explicitly.
- Bayesian graph learning has not been fully explored for spatio-temporal traffic forecasting.
- Existing Bayesian methods estimate uncertainty only in network parameters rather than graph structures.
- Existing graph-learning methods cannot effectively model asymmetric traffic relationships.
- A unified Bayesian graph-learning framework is needed that jointly models graph uncertainty, prior topology, and spatial-temporal dependencies.

---

# Proposed System

The paper proposes a **Bayesian Graph Convolutional Network (BGCN)**.

The graph structure is modeled as a random variable drawn from a parametric generative model. The posterior graph structure is inferred using both observed road topology and historical traffic data.

The graph consists of two components:

- **Constant adjacency matrix** derived from road topology.
- **Learnable adjacency matrix** learned directly from traffic observations.

The framework:

- models graph uncertainty using Bayesian inference,
- approximates posterior inference through Monte Carlo Dropout,
- learns both positive and negative spatial relationships,
- produces asymmetric graph structures,
- serves as a plug-and-play module for existing graph-based traffic forecasting models.

---

# Main Contributions

- First application of Bayesian graph convolution to traffic forecasting.
- Bayesian inference framework for graph structure learning.
- Integration of prior topology and traffic observations.
- Explicit uncertainty modeling in graph structures.
- Learning of both positive and negative spatial correlations.
- Coarse-to-fine graph learning strategy.
- End-to-end graph learning with little additional computational cost.
- Plug-and-play Bayesian module compatible with existing graph neural networks.
- Superior performance on five benchmark traffic datasets.
- Improved performance when integrated into STGCN and AGCRN.

---

# Experimental Setup

## Datasets

- PeMS3
- PeMS4
- PeMS7
- PeMS8
- PeMS-Bay

---

## Data Scope & Exclusions

- Traffic data collected from California PeMS loop detectors.
- Traffic measurements aggregated into **5-minute intervals**, resulting in **288 observations per detector per day**.
- Dataset split into **training, validation, and testing sets in a 6:2:2 ratio**.
- Missing values are discarded.
- Traffic values are normalized using the **Z-score** method.
- The paper specifies the collection periods:
  - **PeMS3:** September 1–November 30, 2018.
  - **PeMS4:** January 1–February 28, 2018.
  - **PeMS7:** May 1–August 31, 2017.
  - **PeMS8:** July 1–August 31, 2016.
  - **PeMS-Bay:** January 1–May 31, 2017.

---

## Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

---

## Baseline Models

- Historical Average (HA)
- VAR
- GRU-ED
- DCRNN
- STGCN
- Graph WaveNet
- STSGCN
- AGCRN

---

# Experimental Findings

## Performance Comparison

BGCN achieved the best or nearly the best performance on all five benchmark datasets.

### PeMS3

- **MAE:** 14.35
- **RMSE:** 25.28
- **MAPE:** 14.47%

Compared with:

- Graph WaveNet: MAE 14.66
- AGCRN: MAE 15.70
- STGCN: MAE 16.29

---

### PeMS4

- **MAE:** 18.82
- **RMSE:** 30.34
- **MAPE:** 12.87%

Compared with:

- Graph WaveNet: MAE 19.23
- AGCRN: MAE 19.86
- STGCN: MAE 21.09

---

### PeMS7

- **MAE:** 20.09
- **RMSE:** 32.86
- **MAPE:** 8.45%

Compared with:

- Graph WaveNet: MAE 20.77
- AGCRN: MAE 21.81
- STGCN: MAE 22.63

---

### PeMS8

- **MAE:** 14.65
- **RMSE:** 23.43
- **MAPE:** 9.42%

Compared with:

- Graph WaveNet: MAE 15.43
- AGCRN: MAE 16.29
- STGCN: MAE 16.98

---

### PeMS-Bay

- **MAE:** 1.61
- **RMSE:** 3.63
- **MAPE:** 3.71%

Compared with:

- Graph WaveNet: MAE 1.65
- AGCRN: MAE 1.71
- STGCN: MAE 1.77

---

## Ablation Study

Removing major components degraded performance.

- Removing uncertainty:
  - MAE increased from **14.65 → 16.45**
  - RMSE increased from **23.43 → 25.49**
  - MAPE increased from **9.42 → 10.95**

- Removing learnable adjacency matrix:
  - MAE **17.49**
  - RMSE **27.60**
  - MAPE **11.15**

- Removing constant adjacency matrix:
  - MAE **15.06**
  - RMSE **23.96**
  - MAPE **9.53**

---

## Generalization Study

Integrating BGCN into other architectures improved performance.

### STGCN

- MAE improved:
  - **16.98 → 16.70**
- RMSE improved:
  - **26.58 → 25.85**
- MAPE improved:
  - **11.58 → 11.03**

### AGCRN

- MAE improved:
  - **16.29 → 15.63**
- RMSE improved:
  - **25.66 → 24.79**
- MAPE changed:
  - **10.32 → 10.38**

---

## Computational Cost

Compared with Graph WaveNet:

- Graph WaveNet:
  - Parameters: **305,228**
  - Training: **31.20 s**
  - Inference: **5.4 s**

- BGCN:
  - Parameters: **256,076**
  - Training: **33.14 s**
  - Inference: **4.4 s**

The paper also reports that AGCRN has fewer parameters than STGCN and Graph WaveNet but requires longer training and inference because of sequential RNN computation.

---

# Strengths

- Bayesian uncertainty modeling of graph structure.
- Integration of road topology with traffic-derived graph learning.
- Captures positive and negative spatial relationships.
- Learns asymmetric graph structures.
- Strong performance across five benchmark datasets.
- Better parameter efficiency than Graph WaveNet.
- Generalizes well to STGCN and AGCRN.
- Robust across different prediction horizons.

---

# Remaining Limitations

## Author Mentioned Limitations

The **Conclusion and Future Work** section states:

> "In the future, we focus on extending BGCN to other spatio-temporal time series forecasting tasks, such as forecasting ride demand."

No other unresolved technical limitation is explicitly mentioned in the conclusion.

---

## Broader Impact / Ethical Considerations

**Not discussed.**

No section on broader impacts, ethics, or societal implications is present.

---

## Sensor-Centric Perspective

From a sensor-centric viewpoint, the paper does **not** address:

- Sensor reliability estimation.
- Sensor confidence weighting.
- Faulty sensor detection.
- Sensor anomaly detection.
- Dynamic sensor quality assessment.
- Heterogeneous sensor fusion.
- Explicit uncertainty associated with individual sensors.
- Missing sensor recovery beyond simple preprocessing.

The uncertainty modeling is applied to the **graph structure**, not to the sensing process itself.

---

# Relevance to Sensor-Centric Traffic Forecasting

The paper is highly relevant because it learns graph structures directly from traffic sensor observations while incorporating uncertainty into graph learning. The learned graph captures richer spatial relationships than predefined road topology and improves forecasting accuracy across multiple datasets. However, the uncertainty model operates at the graph level rather than the sensor level. Consequently, future sensor-centric traffic forecasting frameworks can extend this work by integrating sensor reliability, confidence estimation, anomaly detection, missing-value recovery, heterogeneous sensor fusion, and dynamic sensor quality into graph construction and forecasting.

---

# Terminology Notes

- **BGCN (Bayesian Graph Convolutional Network)** is a generic acronym that could refer to different Bayesian graph convolution architectures in other graph learning or traffic forecasting literature. The acronym should therefore be interpreted in the context of this paper.
- **AGCRN**, **STGCN**, and **Graph WaveNet** are widely used architecture names in traffic forecasting literature but consistently refer to their established models.

---

# Keywords (20–30)

Traffic Forecasting, Bayesian Graph Convolutional Network, BGCN, Bayesian Inference, Graph Neural Network, Graph Convolution, Graph Structure Learning, Uncertainty Modeling, Graph Uncertainty, Monte Carlo Dropout, Adaptive Graph Learning, Latent Graph Structure, Road Topology, Spatial-Temporal Forecasting, Positive Spatial Correlation, Negative Spatial Correlation, Graph WaveNet, STGCN, DCRNN, AGCRN, STSGCN, PeMS Dataset, Intelligent Transportation Systems, Traffic Sensors, Deep Learning, Asymmetric Graph Learning, Plug-and-Play Module, Traffic Prediction