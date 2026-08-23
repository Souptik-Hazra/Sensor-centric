# Literature Review

## Paper 1

### Title
Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting

### Authors
Lei Bai, Lina Yao, Can Li, Xianzhi Wang, Can Wang

### Year
2020 (NeurIPS)

# Objective

The paper aims to improve multi-step traffic forecasting by addressing two key limitations of existing Graph Convolutional Network (GCN)-based methods: (1) the inability to learn node-specific traffic patterns due to shared model parameters, and (2) the reliance on manually predefined graph structures for modeling spatial dependencies. To achieve this, the authors propose adaptive parameter learning and adaptive graph generation modules that automatically learn node-specific representations and infer spatial interdependencies directly from traffic data without requiring a predefined graph.

# Existing Systems

The paper reviews several categories of traffic forecasting methods.

## Traditional Statistical Models

Representative methods include:

- Historical Average (HA)
- Auto-Regressive Integrated Moving Average (ARIMA)
- Vector Auto-Regression (VAR)

These methods model traffic forecasting as a time-series prediction problem using historical observations and simple statistical assumptions. They mainly capture temporal dependencies.

---

## Deep Learning-Based Temporal Models

Representative methods include:

- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- GRU Encoder–Decoder (GRU-ED)
- Temporal Convolution Networks (TCN)

These models automatically learn nonlinear temporal dependencies from historical traffic observations and significantly improve prediction compared with statistical models.

---

## CNN-Based Models

Representative method:

- DSANet

CNN-based methods use convolutional networks and attention mechanisms to model temporal and local spatial correlations.

---

## Correlated Time-Series Prediction Models

Recent correlated time-series forecasting methods employ:

- LSTM
- GRU
- Temporal CNN
- Transformer architectures

These models learn nonlinear temporal relationships among correlated variables.

---

## Graph Convolutional Network-Based Traffic Forecasting

Representative methods include:

- DCRNN
- Graph WaveNet
- STGCN
- ASTGCN
- STSGCN
- GMAN

These approaches model transportation systems as graphs, where roads or traffic sensors are represented as graph nodes. Graph convolution captures spatial dependencies while recurrent or convolutional modules capture temporal evolution.

---

## General Graph Neural Networks

The paper also reviews several generic graph learning methods.

- Graph Convolution Network (GCN)
- Graph Attention Network (GAT)
- DiffPool

These methods aggregate neighborhood information or learn hierarchical graph representations but are designed for general graph learning rather than traffic forecasting.

# Limitations of Existing Systems

The paper identifies several limitations of prior approaches.

## Statistical Models

- Unable to capture nonlinear traffic dynamics.
- Ignore spatial relationships among traffic sensors.
- Cannot model complex spatial-temporal dependencies.
- Poor performance on large-scale traffic forecasting tasks.

---

## Temporal Deep Learning Models

- Primarily focus on temporal dynamics.
- Do not explicitly model spatial interdependencies among traffic locations.
- Traffic sensors are treated almost independently.

---

## CNN-Based Models

- Assume traffic data are organized on regular spatial grids.
- Real transportation networks have irregular graph structures.
- Limited applicability to realistic road networks.

---

## Correlated Time-Series Models

- LSTM, GRU and TCN do not explicitly model dependencies among different traffic series.
- Transformer-based models require massive training samples because of the large number of trainable parameters.

---

## Existing GCN-Based Models

The paper highlights several shortcomings.

### Shared Parameter Learning

Existing GCN models use one shared parameter space for all traffic nodes.

This ignores node-specific traffic characteristics arising from:

- surrounding Points of Interest (POIs),
- weather,
- road functions,
- local traffic dynamics.

---

### Failure to Learn Node-Specific Patterns

Traffic series may exhibit

- similar,
- opposite,
- contradictory

patterns.

Adjacent roads may behave differently while distant roads may behave similarly.

Existing methods mainly learn shared patterns instead of individualized node behaviors.

---

### Dependence on Predefined Graphs

Most GCN-based methods require manually constructed adjacency matrices based on

- geographical distance,
- traffic similarity,
- node similarity,
- POI similarity.

These graphs remain fixed throughout training.

---

### Requirement for Domain Knowledge

Graph construction depends heavily on expert knowledge and manual engineering, limiting portability across different traffic networks.

---

### Sensitivity to Graph Quality

Prediction performance is strongly affected by graph quality.

Poorly constructed graphs reduce forecasting accuracy.

---

### Incomplete Spatial Dependency Modeling

Predefined graphs cannot represent complete spatial dependencies.

Hidden relationships among traffic nodes remain undiscovered.

---

### Graph Construction Bias

Distance-based and similarity-based graph construction methods are heuristic rather than prediction-oriented, introducing structural bias and incomplete connectivity.

---

### Static Graph Assumption

Most GCN variants operate on fixed graph structures and cannot naturally model dynamically evolving traffic relationships.

---

### Computational Issues

Assigning completely independent parameters to every node results in an extremely large parameter tensor, making optimization difficult and increasing overfitting risk.

---

### Long-Horizon Forecasting

Existing GCN models deteriorate more rapidly as prediction horizons increase.

---

### Computational Cost of Sequential Prediction

Autoregressive multi-step prediction increases inference time because future steps are generated sequentially.

# Research Gaps Identified

The paper identifies the following research gaps.

- Existing GCN models cannot effectively learn node-specific traffic patterns.
- Shared parameter learning ignores heterogeneous node behavior.
- Existing methods rely heavily on manually predefined graph structures.
- Hidden spatial dependencies cannot be automatically discovered from traffic data.
- Existing graph construction methods require domain knowledge and are sensitive to graph quality.
- Existing methods mainly learn shared spatial-temporal patterns rather than node-specific characteristics.
- Current graph structures are static and not optimized specifically for forecasting.
- Temporal models ignore explicit spatial relationships.
- Existing approaches struggle with long-term prediction.
- A unified framework that simultaneously learns node-specific parameters and adaptive graph structures is lacking.

# Proposed System

The paper proposes the **Adaptive Graph Convolutional Recurrent Network (AGCRN)**.

The model integrates two adaptive modules.

## Node Adaptive Parameter Learning (NAPL)

NAPL learns node-specific model parameters from a shared parameter pool using trainable node embeddings, allowing each traffic node to learn individualized spatial-temporal patterns.

## Data Adaptive Graph Generation (DAGG)

DAGG automatically infers hidden spatial dependencies directly from traffic data by learning adaptive node embeddings, eliminating the need for manually predefined adjacency matrices.

These modules are combined with Gated Recurrent Units (GRU) to construct AGCRN, enabling simultaneous learning of node-specific spatial and temporal correlations.

# Main Contributions

- Introduces Node Adaptive Parameter Learning (NAPL) for learning node-specific traffic patterns.
- Proposes Data Adaptive Graph Generation (DAGG) to infer spatial dependencies automatically from data.
- Develops Adaptive Graph Convolutional Recurrent Network (AGCRN), integrating NAPL, DAGG and GRU.
- Eliminates dependence on predefined graph structures.
- Learns node-specific spatial-temporal representations automatically.
- Demonstrates state-of-the-art performance on two real-world traffic datasets.
- Shows that adaptive modules can be applied individually or jointly to improve prediction.

# Experimental Setup

## Datasets

Two public traffic datasets were used.

### PeMSD4

- 307 loop detectors
- San Francisco Bay Area
- Data collected from **1 January 2018 to 28 February 2018**

### PeMSD8

- 170 loop detectors
- San Bernardino Area
- Data collected from **1 July 2016 to 31 August 2016**

---

## Data Scope & Exclusions

- Missing values were filled using linear interpolation.
- Data aggregated into **5-minute intervals** (288 observations per day).
- Standard normalization applied.
- One-hour historical observations used to predict the next hour.
- Twelve historical time steps used as input.
- Twelve future time steps predicted.
- Dataset split chronologically into **training : validation : testing = 6 : 2 : 2**.
- No excluded days, weather conditions or sensor filtering were discussed.

---

## Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

---

## Baseline Models

- HA
- VAR
- GRU-ED
- DSANet
- DCRNN
- STGCN
- ASTGCN
- STSGCN

# Experimental Findings

AGCRN consistently outperformed all baseline methods.

## PeMSD4

| Metric | AGCRN | Best Baseline | Relative Improvement |
|--------|-------:|--------------:|---------------------:|
| MAE | **19.83** | 21.16 (STGCN) | **6.29%** |
| RMSE | **32.26** | 33.44 (DCRNN) | **3.52%** |
| MAPE | **12.97%** | 13.83% (STGCN) | **6.22%** |

---

## PeMSD8

| Metric | AGCRN | Best Baseline | Relative Improvement |
|--------|-------:|--------------:|---------------------:|
| MAE | **15.95** | 16.82 (DCRNN) | **5.17%** |
| RMSE | **25.22** | 26.36 (DCRNN) | **4.32%** |
| MAPE | **10.09%** | 10.92% (DCRNN) | **7.60%** |

Additional findings include:

- GCN-based models outperform statistical models and DSANet, highlighting the importance of explicit spatial modeling.
- AGCRN performs best across almost all prediction horizons.
- Prediction degradation over long horizons is slower than competing GCN methods.
- NAPL improves longer-horizon (30-minute and 60-minute) prediction.
- DAGG improves spatial dependency inference compared with predefined graphs.
- Embedding dimension **10** achieves the best performance.
- AGCRN (embedding dimension 10) contains **748,810 parameters** and requires **35.56 s/epoch** training time.
- DCRNN contains **149,057 parameters** with **36.39 s/epoch** training time.
- STGCN has the shortest training time (**16.36 s/epoch**) but lower prediction accuracy.
- The adaptive modules consistently improve forecasting performance whether deployed individually or jointly.

# Strengths

- Eliminates reliance on predefined graph structures.
- Learns hidden spatial dependencies automatically.
- Captures node-specific traffic behaviors.
- Integrates adaptive graph learning with recurrent temporal modeling.
- Demonstrates strong performance on multiple real-world datasets.
- Improves long-term forecasting accuracy.
- Supports end-to-end optimization.
- Applicable to general correlated time-series prediction beyond traffic forecasting.

# Remaining Limitations

- Adaptive graph generation may incur high computational cost for extremely large graphs.
- Larger embedding dimensions increase parameter count and may cause overfitting.
- Selecting an appropriate embedding dimension requires balancing accuracy and complexity.
- Scalability to larger datasets and additional traffic forecasting architectures remains to be explored.

## Author Mentioned Limitations

The paper explicitly states in the future work discussion:

> "Our future work will focus on examining the scalability of our work from two perspectives: 1) data perspective – validating the performance of AGCRN on more time series prediction tasks; 2) model perspective – adapting NAPL and DAGG to more GCN-based traffic forecasting models."

## Broader Impact / Ethical Considerations

The paper contains a dedicated **Broader Impact** section.

The authors state that more accurate traffic forecasting can improve taxi dispatch, route planning, transportation efficiency, traveler time savings, operator income, and energy consumption. The adaptive modules are also applicable to other correlated time-series domains such as epidemic forecasting, economic growth prediction, and climate analysis.

The paper also discusses a potential negative societal impact. In ride-sharing platforms, emphasizing predicted high-demand regions could increase waiting times for travelers in low-demand areas, raising fairness concerns.

## Sensor-Centric Perspective

From a sensor-centric perspective, the paper provides several observations.

- Traffic sensors are treated as graph nodes whose relationships are learned adaptively.
- Hidden interdependencies among sensors are inferred directly from observations rather than predefined connectivity.
- The framework naturally supports heterogeneous sensor behavior through node-specific parameter learning.
- Missing sensor values are handled using linear interpolation during preprocessing.
- The paper does **not** discuss sensor confidence estimation, sensor reliability modeling, anomaly detection, faulty sensors, dynamic sensor quality, or uncertainty-aware sensing.

# Relevance to Sensor-Centric Traffic Forecasting

This paper is highly relevant to sensor-centric traffic forecasting because it shifts graph construction from manually engineered sensor relationships to data-driven adaptive sensor dependency learning. Each traffic sensor learns individualized representations while spatial relationships are inferred automatically, reducing dependence on prior road-network knowledge. The adaptive node-specific modeling aligns closely with sensor-centric frameworks that emphasize heterogeneous sensor behavior and evolving inter-sensor correlations.

# Terminology Notes

Several acronyms used in this paper are sufficiently generic that they may refer to different methods in other traffic forecasting literature.

- GCN
- GAT
- GRU
- TCN
- DCRNN
- STGCN
- ASTGCN
- STSGCN
- GMAN
- DAGG
- NAPL
- AGCRN

# Keywords (20–30)

Traffic forecasting, Graph Convolutional Network, Adaptive Graph Convolutional Recurrent Network, AGCRN, Node Adaptive Parameter Learning, NAPL, Data Adaptive Graph Generation, DAGG, Graph Neural Network, Spatial-temporal forecasting, Multi-step prediction, Correlated time series, Traffic flow prediction, Adaptive graph learning, Node embedding, Graph generation, Recurrent neural network, GRU, PeMSD4, PeMSD8, Spatial dependency, Temporal dependency, Loop detectors, Intelligent Transportation Systems, Deep learning, Multivariate forecasting, Traffic sensors, Adaptive parameter learning, Graph-based forecasting