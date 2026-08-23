# Model Architectures & Baseline Implementations

## 1. Overview
This project evaluates 4 distinct baseline architectures on `METR-LA` under the `FairTP` fairness-guided sampling framework:
1. **DCRNN**: Deep Spatiotemporal Graph Recurrent Neural Network
2. **GWNet (Graph WaveNet)**: Dilated Causal Convolutional Spatial Graph Network
3. **DLinear**: Decomposed Linear Time-Series Model with RevIN
4. **HA**: Historical Average Statistical Baseline

---

## 2. Model 1: DCRNN (Li et al., 2018)
- **Type**: Recurrent Graph Neural Network (Seq2Seq Architecture)
- **Spatial Mechanism**: Bidirectional Random-Walk Graph Diffusion Convolution:
  $$P_f = D_O^{-1} W, \quad P_b = D_I^{-1} W^T$$
  $$H^{(l)} = \sum_{k=0}^{K-1} \left( \theta_{k,1} P_f^k X + \theta_{k,2} P_b^k X \right)$$
- **Temporal Mechanism**: Gated Recurrent Unit (GRU) with diffusion graph convolution replacing matrix multiplications in transition gates.
- **Parameters**:
  - `num_rnn_layers`: 2
  - `n_filters`: 64
  - `max_diffusion_step`: 2
  - `cl_decay_steps`: 2,000 (Scheduled sampling)

---

## 3. Model 2: Graph WaveNet (Wu et al., 2019)
- **Type**: Fully Convolutional Spatiotemporal Network
- **Spatial Mechanism**: Dual Graph Convolution combining spatial distance graph $A_{sp}$ with a learned adaptive graph $A_{ad}$:
  $$A_{ad} = \text{Softmax}\left( \text{ReLU}\left( E_1 E_2^T \right) \right)$$
  where $E_1, E_2 \in \mathbb{R}^{N \times c}$ are node embeddings trained end-to-end.
- **Temporal Mechanism**: Stacked Dilated Causal 1D Convolutions with Gated Linear Units (GLU) and exponential receptive field expansion.
- **Parameters**:
  - `blocks`: 4
  - `layers`: 2
  - `residual_channels`: 32
  - `dilation_channels`: 32
  - `skip_channels`: 256
  - `end_channels`: 512

---

## 4. Model 3: True DLinear (Zeng et al., 2023)
- **Type**: Decomposed Linear Model with Instance Normalization
- **Preprocessing**: Reversible Instance Normalization (RevIN) to combat non-stationarity:
  $$X_{\text{norm}} = \gamma \odot \left( \frac{X - \mu(X)}{\sigma(X)} \right) + \beta$$
- **Decomposition**: Moving Average Series Decomposition separating trend ($X_{\text{trend}}$) and seasonal ($X_{\text{seasonal}}$) components:
  $$X_{\text{trend}} = \text{AvgPool1D}(X), \quad X_{\text{seasonal}} = X - X_{\text{trend}}$$
- **Linear Layer**: Individual channel-independent linear mappings:
  $$\hat{Y}_{\text{trend}} = W_{\text{trend}} X_{\text{trend}}, \quad \hat{Y}_{\text{seasonal}} = W_{\text{seasonal}} X_{\text{seasonal}}$$
  $$\hat{Y}_{\text{final}} = \text{RevIN}^{-1}\left( \hat{Y}_{\text{trend}} + \hat{Y}_{\text{seasonal}} \right)$$

---

## 5. Model 4: True HA (Historical Average)
- **Type**: Non-parametric Statistical Baseline
- **Periodicity**: 2,016 steps per week (288 steps/day $\times$ 7 days/week).
- **Computation**: Exact historical average for each node $i$, day-of-week $d$, and time-of-day $t$ computed strictly over the training split:
  $$\hat{Y}_{i, d, t} = \frac{1}{|K|} \sum_{k \in K} Y_{i, d, t, k}$$
- Evaluated using the exact same `fsample_engine` evaluation loop as DCRNN/GWNet for fair benchmarking.
