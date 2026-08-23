# Literature Review

## Paper 5

### Title
Are Transformers Effective for Time Series Forecasting?

### Authors
Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu

### Year
2022

# Objective

The paper investigates whether Transformer-based architectures are genuinely effective for Long-Term Time Series Forecasting (LTSF). It questions the widely accepted assumption that Transformers outperform previous forecasting approaches and examines whether their reported success is due to the Transformer architecture itself or to other design choices, particularly the Direct Multi-Step (DMS) forecasting strategy. To validate this hypothesis, the authors introduce an extremely simple one-layer linear baseline (LTSF-Linear) and conduct comprehensive empirical studies comparing it with state-of-the-art Transformer-based forecasting models. The study also advocates revisiting the validity of Transformer-based solutions for other time-series analysis tasks beyond LTSF.

# Existing Systems

The paper reviews the evolution of Time Series Forecasting (TSF) methods from traditional statistical techniques to modern Transformer-based Long-Term Time Series Forecasting (LTSF) models. Existing approaches include statistical methods, machine learning models, deep learning architectures, and Transformer-based forecasting systems.

## Statistical Methods

Traditional forecasting methods model future values using historical observations based on predefined statistical assumptions.

Representative methods include:

- AutoRegressive Integrated Moving Average (ARIMA)
- Exponential Smoothing
- Structural Time Series Models

These methods have been widely adopted since the 1970s. Although effective for relatively simple forecasting tasks, they require considerable domain expertise to select appropriate models and parameters.

## Machine Learning Methods

Machine learning approaches reduce reliance on statistical assumptions by learning forecasting relationships directly from data.

Representative method:

- Gradient Boosted Regression Trees (GBRT)

These methods generally outperform statistical models but still depend on handcrafted feature engineering and manually designed model structures.

## Deep Learning Methods

### Recurrent Neural Networks (RNNs)

Characteristics:

- Maintain hidden memory states.
- Learn sequential dependencies recursively.
- Generally employ Iterated Multi-Step (IMS) forecasting.

### Convolutional Neural Networks (CNNs)

Characteristics:

- Capture local temporal patterns using convolution filters.
- Can employ either Iterated Multi-Step (IMS) or Direct Multi-Step (DMS) forecasting depending on decoder design.

## Transformer-Based Long-Term Time Series Forecasting

The paper reviews five representative Transformer-based forecasting models.

### LogTrans

- LogSparse attention
- Sparse self-attention
- Approximately O(L log L) attention complexity

### Informer

- ProbSparse attention
- Self-attention distilling
- Generative decoder
- Direct Multi-Step forecasting

### Autoformer

- Seasonal-trend decomposition
- Auto-Correlation mechanism
- Direct Multi-Step forecasting

### Pyraformer

- Multi-resolution pyramidal attention
- Hierarchical temporal dependency modeling
- Spatial-temporal decoder

### FEDformer

- Frequency-enhanced decomposition
- Fourier block
- Wavelet block
- Frequency attention
- Mixture-of-experts decomposition

## Common Transformer Pipeline

Existing Transformer-based LTSF models generally follow four stages:

### Preprocessing

- Normalization
- Timestamp preparation
- Seasonal-trend decomposition

### Embedding

- Fixed positional encoding
- Channel projection
- Local timestamp embedding
- Global timestamp embedding
- Learnable temporal embedding
- Temporal convolution embedding

### Encoder

Different models replace vanilla self-attention using:

- LogSparse attention
- ProbSparse attention
- Auto-Correlation
- Multi-resolution pyramidal attention
- Frequency-enhanced attention

### Decoder

Common decoding strategies include:

- Autoregressive decoder
- Direct Multi-Step decoder
- Decomposition decoder
- Spatial-temporal decoder
- Frequency-attention decoder

## Forecasting Strategies

### Iterated Multi-Step (IMS)

Characteristics:

- Predicts one future step at a time.
- Autoregressive forecasting.
- Lower prediction variance.

Limitation:

- Suffers from accumulated prediction errors over long forecasting horizons.

### Direct Multi-Step (DMS)

Characteristics:

- Predicts the complete forecasting horizon simultaneously.

Advantages:

- Avoids autoregressive error accumulation.
- Better suited for long-term forecasting.

The paper argues that many reported improvements of Transformer-based forecasting methods originate from adopting DMS forecasting rather than from the Transformer architecture itself.

# Limitations of Existing Systems

The highlighted content identifies the following limitations of existing forecasting methods, particularly Transformer-based models.

1. Self-attention is permutation-invariant and therefore cannot naturally preserve temporal order, which is fundamental for time series forecasting.

2. Temporal information is inevitably lost during self-attention, even when positional encoding, timestamp embeddings, temporal embeddings, or sub-series tokenization are introduced.

3. Transformers possess semantic inductive bias rather than temporal inductive bias because they were originally designed for Natural Language Processing, Speech Recognition, and Computer Vision.

4. Existing Transformer architectures become increasingly complex by introducing decomposition modules, sparse attention, Auto-Correlation, frequency-domain processing, temporal embeddings, and specialized decoders without fundamentally solving temporal modeling.

5. Vanilla Transformer suffers from quadratic computational complexity.

6. Autoregressive decoding causes accumulated forecasting errors over long prediction horizons.

7. Existing Transformer models fail to effectively exploit long historical input sequences.

8. Increasing the look-back window often produces little improvement or even performance degradation.

9. Existing Transformer models primarily capture local temporal information instead of genuine long-range temporal dependencies.

10. Temporal-order experiments show that Transformer models are only weakly affected by shuffled input sequences, indicating poor preservation of sequence order.

11. Existing Transformer models tend to overfit temporal noise when longer historical sequences are provided.

12. Existing methods rely heavily on positional encoding and timestamp embeddings to compensate for the lack of temporal inductive bias.

13. Existing models show considerable variation across datasets, indicating limited robustness and generalization.

14. Transformer models frequently fail to capture prediction scale, prediction bias, and long-term trends.

15. Progressive removal of Transformer components improves forecasting performance, suggesting that self-attention and several auxiliary modules are unnecessary for existing LTSF benchmarks.

16. Limited training data is not the primary reason for Transformer underperformance.

17. Practical inference efficiency does not improve proportionally with reduced theoretical attention complexity.

18. Existing research emphasizes designing increasingly efficient attention mechanisms rather than improving forecasting capability.

19. Existing Transformer models perform poorly under train-test distribution shifts.

20. Existing models inadequately exploit periodic temporal structures.

21. Transformer performance fluctuates under different look-back window sizes.

22. Existing Transformer models exhibit unstable performance across both short-term and long-term forecasting tasks.

23. Learned Transformer representations are less interpretable than decomposition-based linear models.

24. Existing Transformer models do not explicitly separate trend and seasonal information during forecasting.

# Research Gaps Identified

The paper identifies the following research gaps:

- The suitability of Transformer architectures for Long-Term Time Series Forecasting has not been rigorously validated.
- Existing studies have not determined whether Transformer performance improvements originate from the Transformer architecture itself or from forecasting strategies such as Direct Multi-Step prediction.
- The contributions of self-attention, positional encoding, temporal embeddings, decomposition, and decoder design have not been systematically analyzed.
- Existing benchmark evaluations mainly compare against weaker IMS baselines that naturally suffer from accumulated prediction errors.
- Existing research emphasizes increasingly sophisticated Transformer architectures without establishing strong simple baselines.
- The capability of Transformer models to extract long-term temporal dependencies has not been comprehensively investigated.

# Proposed System

The paper proposes a family of simple linear forecasting models.

## LTSF-Linear

- Single linear layer.
- Direct Multi-Step forecasting.
- Directly regresses historical observations to future predictions.

## DLinear

- Applies seasonal-trend decomposition using moving-average decomposition.
- Separately models trend and seasonal components before linear prediction.

## NLinear

- Addresses train-test distribution shifts.
- Subtracts the last observed value before prediction and adds it back afterward, aligning predictions with the current data distribution.

# Main Contributions

- First systematic study questioning the effectiveness of Transformer-based LTSF models.
- Introduction of an extremely simple LTSF-Linear baseline.
- Introduction of DLinear and NLinear variants.
- Comprehensive empirical comparison with state-of-the-art Transformer models across nine benchmark datasets.
- Extensive investigation of long input sequences, temporal order, positional encoding, embedding strategies, forecasting strategies, look-back windows, computational efficiency, and distribution shifts.
- Demonstration that Transformer temporal modeling capability has been overestimated on current LTSF benchmarks.

# Experimental Setup

## Datasets

Nine multivariate real-world datasets:

- ETTh1
- ETTh2
- ETTm1
- ETTm2
- Traffic
- Electricity
- Exchange-Rate
- Weather
- ILI

## Data Scope & Exclusions

Dataset descriptions provided:

- ETTh1 and ETTh2: hourly electricity transformer data (July 2016 – July 2018).
- ETTm1 and ETTm2: 15-minute electricity transformer data (July 2016 – July 2018).
- Traffic: San Francisco freeway occupancy (2015–2016).
- Electricity: hourly electricity consumption of 321 clients (2012–2014).
- Exchange-Rate: daily exchange rates of eight countries (1990–2016).
- Weather: 21 weather indicators recorded every 10 minutes during 2020 in Germany.
- ILI: weekly influenza-like illness data (2002–2021).

Experiments evaluate multiple look-back window sizes depending on dataset frequency (e.g., 24–720 historical time steps) and forecasting horizons of 96, 192, 336, and 720 time steps.

No additional exclusions or filtering beyond preprocessing and decomposition are discussed.

## Evaluation Metrics

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

## Baseline Models

- Vanilla Transformer
- LogTrans
- Informer
- Autoformer
- Pyraformer
- FEDformer
- Repeat
- LTSF-Linear
- DLinear
- NLinear

# Experimental Findings

- LTSF-Linear consistently outperforms all evaluated Transformer models across the benchmark datasets.
- Reported improvements range from approximately **20% to 50%** over existing Transformer models on multivariate forecasting benchmarks.
- On the Exchange-Rate dataset, the simple Repeat baseline outperforms Transformer-based methods by approximately **45%**.
- LTSF-Linear achieves lower forecasting errors than Transformer models on all nine benchmark datasets.
- Increasing the look-back window generally improves LTSF-Linear performance, whereas Transformer models usually stop improving after approximately **96 input time steps** or even deteriorate.
- Temporal-order experiments show that shuffling input sequences causes only small degradation for Transformer models but significantly larger degradation for LTSF-Linear, indicating stronger reliance on temporal order.
- Progressive removal of self-attention and auxiliary Transformer modules improves forecasting accuracy.
- Practical efficiency experiments show that reduced theoretical attention complexity does not necessarily improve inference time.
- Clear train-test distribution shifts are observed in ETTh1, ETTh2, and ILI datasets, and NLinear effectively alleviates these shifts.
- Weight visualization demonstrates interpretable trend and seasonal patterns learned by DLinear across multiple datasets and forecasting horizons.

# Strengths

- Comprehensive evaluation across nine benchmark datasets.
- Comparison with multiple state-of-the-art Transformer models.
- Strong simple baselines (LTSF-Linear, DLinear, and NLinear).
- Thorough ablation study removing Transformer components progressively.
- Extensive analysis of temporal order, look-back windows, embedding strategies, forecasting strategies, and distribution shifts.
- Practical efficiency comparison including inference time, parameter count, and memory usage.
- Highly interpretable visualization of learned trend and seasonal weights.

# Remaining Limitations

- LTSF-Linear has limited model capacity.
- Performance is demonstrated primarily on current benchmark datasets.
- Applicability to more complex forecasting scenarios is not evaluated.
- Further improvements may require new model architectures, improved data processing techniques, and more challenging benchmark datasets.

## Author Mentioned Limitations

> "LTSF-Linear has a limited model capacity, and it merely serves as a simple yet competitive baseline for future research."

> "We believe there is a great potential for new model designs, data processing, and benchmarks to tackle the challenging LTSF problem."

## Broader Impact / Ethical Considerations

Not discussed.

## Sensor-Centric Perspective

The paper does not explicitly address:

- sensor failures,
- missing sensor observations,
- noisy sensor measurements,
- heterogeneous sensing,
- sensor confidence estimation,
- sensor anomaly detection.

However, the highlighted results indicate that existing forecasting models:

- struggle under train-test distribution shifts,
- fail to preserve temporal order,
- inadequately exploit long historical observations,
- overfit temporal noise.

These limitations are directly relevant to continuously collected sensor streams.

# Relevance to Sensor-Centric Traffic Forecasting

The paper is highly relevant to sensor-centric traffic forecasting because traffic prediction depends on continuously collected sensor measurements with strong temporal continuity. The study demonstrates that existing Transformer-based forecasting models inadequately preserve temporal order, fail to exploit long historical sensor observations, struggle under distribution shifts, overfit temporal noise, and do not consistently benefit from longer input histories. These findings motivate the development of sensor-centric forecasting frameworks that explicitly model temporal continuity, sensor reliability, evolving sensor distributions, and interpretable temporal relationships.

# Terminology Notes

The following names and acronyms may refer to different models, implementations, or datasets in other forecasting literature and should therefore be clearly defined when cited:

- Transformer
- Informer
- Autoformer
- FEDformer
- LogTrans
- Pyraformer
- LTSF-Linear
- DLinear
- NLinear
- IMS (Iterated Multi-Step)
- DMS (Direct Multi-Step)
- ETT (Electricity Transformer Temperature)

# Keywords (20–30)

Time Series Forecasting, Long-Term Time Series Forecasting, LTSF, Transformer, Self-Attention, Temporal Dependency, Temporal Order, Positional Encoding, Temporal Embedding, Direct Multi-Step Forecasting, Iterated Multi-Step Forecasting, LTSF-Linear, DLinear, NLinear, Informer, Autoformer, FEDformer, LogTrans, Pyraformer, Distribution Shift, Seasonal Decomposition, Trend Decomposition, Look-back Window, Frequency Attention, Benchmark Evaluation, Computational Efficiency, Interpretability, Temporal Noise, Multivariate Forecasting