# METR-LA Dataset Technical Specification & Implementation Guide

## 1. Overview & Provenance
- **Dataset Name**: METR-LA (Los Angeles Metropolitan Traffic)
- **Domain**: Spatiotemporal Traffic Flow/Speed Forecasting & Algorithmic Fairness Diagnosis
- **Origin**: Sensor network of automated loop detectors deployed on Los Angeles County freeways, monitored by Caltrans Performance Measurement System (PeMS).
- **Time Range**: March 1, 2012 – June 30, 2012 (119 days continuous)
- **Sampling Rate**: 5-minute aggregation intervals (288 time steps per 24-hour cycle)
- **Node Count**: 207 physical traffic sensors

---

## 2. Full Quantitative Statistics
| Property | Value | Technical Context |
| :--- | :--- | :--- |
| **Number of Sensors ($N$)** | `207` | Nodes in graph $G = (V, E, W)$ |
| **Total Time Steps ($T_{total}$)** | `34,272` | 288 steps/day $\times$ 119 days |
| **Total Observations** | `7,094,304` | $207 \text{ sensors} \times 34,272 \text{ timestamps}$ |
| **Historical Lookback ($T_{in}$)** | `12` | 60 minutes historical context ($12 \times 5 \text{ min}$) |
| **Prediction Horizon ($T_{out}$)** | `12` | 60 minutes future target ($12 \times 5 \text{ min}$) |
| **Feature Dimension ($F$)** | `2` | Dim 0: Traffic Speed (mph)<br>Dim 1: Time-of-Day (normalized $[0, 1)$) |
| **Missing Data Rate** | `8.109%` | Near-zero or missing sensor values ($0.0 \text{ mph}$) |
| **Data Split Ratio** | `70% / 10% / 20%` | Strict chronological split (Train / Val / Test) |
| **Train Samples** | `23,974` | Sliding windows (March 1 – April 24) |
| **Validation Samples** | `3,425` | Sliding windows (April 25 – May 10) |
| **Test Samples** | `6,850` | Sliding windows (May 11 – June 30) |
| **Global Mean Speed ($\mu$)** | `~53.7 mph` | Computed on training split |
| **Global Std Speed ($\sigma$)** | `~20.2 mph` | Computed on training split |

---

## 3. Directory & File Structure

```text
final_package/FairTP/data/metr-la/
└── 2019/
    ├── train.npz               # Training dataset array (x: [23974, 12, 207, 2], y: [23974, 12, 207, 2])
    ├── val.npz                 # Validation dataset array (x: [3425, 12, 207, 2], y: [3425, 12, 207, 2])
    ├── test.npz                # Test dataset array (x: [6850, 12, 207, 2], y: [6850, 12, 207, 2])
    ├── adj_mx.npy              # Thresholded Gaussian distance adjacency matrix (207, 207)
    ├── adj_mx_all1.npy         # Unweighted dense adjacency matrix fallback (207, 207)
    ├── adj_mx_mapping.json     # Dictionary mapping 0-indexed node IDs to physical sensor IDs
    ├── distances.csv           # Raw pairwise road network distances (from, to, distance)
    ├── sensor_locations.csv    # Sensor geographical metadata (sensor_id, latitude, longitude)
    ├── metr_la_district.json   # 13-cluster K-Means regional spatial partition (for RSF metric)
    └── his_initial200.npz      # Historical sensor sampling state array (207, 200) for FairTP engine
```

---

## 4. Mathematical Specifications & Preprocessing Rules

### A. Graph Adjacency Construction ($W$)
The spatial adjacency matrix $W \in \mathbb{R}^{207 \times 207}$ in `adj_mx.npy` is computed via thresholded Gaussian kernel over road network distances $d_{ij}$:
$$W_{ij} = \begin{cases} \exp\left(-\frac{d_{ij}^2}{\sigma^2}\right), & \text{if } d_{ij} \le \kappa \text{ and } \exp\left(-\frac{d_{ij}^2}{\sigma^2}\right) \ge \epsilon \\ 0, & \text{otherwise} \end{cases}$$
- **$\sigma$**: Standard deviation of pairwise road distances $d_{ij}$.
- **Threshold ($\kappa, \epsilon$)**: Set such that values $W_{ij} < 0.1$ are truncated to zero for graph sparsity.

### B. Input Normalization & Zero-Masking
1. **Z-Score Normalization**:
   $$x_{\text{norm}} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
   - Applied *only* to Feature 0 (Traffic Speed).
2. **Missing Value Masking**:
   - Zero values ($0.0 \text{ mph}$) represent sensor dropouts/failures (8.109% of dataset).
   - Loss functions and metrics (MAE, RMSE, MAPE) mask out zeros during loss evaluation:
     $$\text{Mask}_{i,t} = \mathbb{I}(y_{i,t} > 0.0)$$

### C. Time-of-Day Feature Encoding
Feature 1 represents daily periodic time step:
$$f_1(t) = \frac{t \bmod 288}{288.0} \in [0, 1)$$

---

## 5. Fairness Metrics & Spatial Regional Grouping

### A. 13-Cluster Spatial Regional Partition (`metr_la_district.json`)
To compute **Region-based Static Fairness (RSF)**, the 207 sensors are grouped into 13 geographic regional districts ($R_1, R_2, \dots, R_{13}$) using K-Means spatial clustering on latitudinal and longitudinal coordinates from `sensor_locations.csv`.

### B. Fairness Formulas
1. **Region-based Static Fairness (RSF)**:
   $$\text{RSF} = \frac{1}{\binom{13}{2}} \sum_{i < j} \left| \text{MAPE}(R_i) - \text{MAPE}(R_j) \right|$$
   - Measures disparity in forecast accuracy across regional districts. Lower = fairer.

2. **Sensor-based Dynamic Fairness (SDF)**:
   $$\text{SDF} = \frac{1}{\binom{207}{2}} \sum_{i < j} \left( \left| \text{Pos}_i - \text{Pos}_j \right| + \left| \text{Neg}_i - \text{Neg}_j \right| \right)$$
   - Tracks cumulative historical benefit vs. sacrifice states per sensor across training epochs.

---

## 6. Evaluation Loss & Metrics Reference
Standard evaluation functions implemented in `src/utils/metrics.py`:
- **Masked MAE**: $\frac{\sum |y - \hat{y}| \cdot \text{Mask}}{\sum \text{Mask}}$
- **Masked RMSE**: $\sqrt{\frac{\sum (y - \hat{y})^2 \cdot \text{Mask}}{\sum \text{Mask}}}$
- **Masked MAPE**: $\frac{\sum \frac{|y - \hat{y}|}{y} \cdot \text{Mask}}{\sum \text{Mask}}$
