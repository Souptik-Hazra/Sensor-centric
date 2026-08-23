# Literature Review

## Paper 1

### Title

**Graph WaveNet for Deep Spatial-Temporal Graph Modeling**

### Authors

Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, Chengqi Zhang

### Year

2019

# Objective

To improve traffic forecasting by jointly learning hidden spatial dependencies and long-range temporal dependencies through a unified graph neural network that does not rely solely on predefined graph structures.

# Existing Systems

- Existing traffic forecasting methods model spatial dependencies using predefined graph structures.
- Most methods combine Graph Convolutional Networks (GCNs) with Recurrent Neural Networks (RNNs) or Convolutional Neural Networks (CNNs).
- RNN-based approaches capture temporal dependencies through recurrent propagation.
- CNN-based approaches model temporal dependencies using temporal convolutions.
- Existing graph convolution methods assume that predefined adjacency matrices correctly represent node relationships.

# Limitations of Existing Systems

- Predefined graph structures cannot accurately represent all dependency relationships between traffic sensors.
- Hidden spatial dependencies cannot be discovered when graph connectivity is incomplete.
- RNN-based methods suffer from sequential computation and gradient vanishing/exploding, limiting long-range temporal modeling.
- CNN-based temporal models require many stacked layers or global pooling to obtain sufficiently large receptive fields.
- Existing adaptive graph learning methods are designed for static graphs and are not suitable for dynamic spatial-temporal graph modeling.
- Existing methods cannot jointly learn adaptive spatial dependencies and long-range temporal dependencies within a unified framework.

# Research Gaps Identified

- Automatic discovery of hidden spatial dependencies without requiring complete prior graph knowledge.
- Efficient modeling of long-range temporal dependencies.
- Unified learning of adaptive graph structures and temporal dynamics.
- Adaptive graph learning suitable for dynamic spatial-temporal traffic forecasting.

# Proposed System

Graph WaveNet introduces:

- A self-adaptive adjacency matrix learned directly from node embeddings.
- Diffusion graph convolution using both predefined and learned graph structures.
- Dilated causal temporal convolutions for modeling long-range temporal dependencies.
- Gated temporal convolution modules with residual and skip connections in an end-to-end framework.

# Main Contributions

- Introduces a self-adaptive adjacency matrix that automatically learns hidden spatial dependencies.
- Reduces reliance on manually constructed graph structures.
- Integrates adaptive graph learning and temporal convolution into a unified framework.
- Efficiently captures long-range temporal dependencies through dilated causal convolutions.
- Achieves state-of-the-art forecasting performance on benchmark traffic datasets.
- Demonstrates that combining predefined and adaptive graph structures improves forecasting performance.

# Experimental Setup

- **Datasets**
  - METR-LA
  - PEMS-BAY

- **Data Scope & Exclusions**
  - Not discussed.

- **Evaluation Metrics**
  - Mean Absolute Error (MAE)
  - Root Mean Square Error (RMSE)
  - Mean Absolute Percentage Error (MAPE)

- **Baseline Models**
  - ARIMA
  - FC-LSTM
  - WaveNet
  - DCRNN
  - STGCN
  - GGRU

# Experimental Findings

- Graph WaveNet achieved the best forecasting performance on both METR-LA and PEMS-BAY.
- The **Forward + Backward + Adaptive** graph configuration produced the best prediction accuracy.
- The **Adaptive-only** graph performed comparably to the predefined forward graph, showing that meaningful graph structures can be learned automatically.
- Graph WaveNet produced more stable prediction curves than WaveNet.
- Graph WaveNet achieved faster inference than DCRNN while maintaining superior prediction performance.
- The highlighted content reports that Graph WaveNet consistently obtained the lowest MAE, RMSE, and MAPE across 15-, 30-, and 60-minute forecasting horizons. No individual numerical values were highlighted.

# Strengths

- Learns hidden spatial dependencies automatically.
- Reduces dependence on predefined road-network graphs.
- Efficiently captures long-range temporal dependencies.
- Unified end-to-end spatial-temporal learning framework.
- Strong forecasting performance across multiple prediction horizons.
- Faster inference than recurrent graph-based forecasting methods.

# Remaining Limitations

## Author Mentioned Limitations

The authors state that future work will focus on:
- **Applying Graph WaveNet to larger-scale datasets.**
- **Learning dynamic spatial dependencies.**

## Broader Impact / Ethical Considerations

Not discussed.

## Sensor-Centric Perspective

Not discussed.

# Relevance to Sensor-Centric Traffic Forecasting

Graph WaveNet is highly relevant because it learns adaptive relationships among traffic sensors instead of relying entirely on predefined graph connectivity. The adaptive adjacency matrix provides a basis for discovering hidden sensor relationships, making the framework suitable for sensor-centric traffic forecasting.

# Terminology Notes

- **WaveNet** is a generic architecture name that also refers to other temporal convolution models in different domains.
- **GCN (Graph Convolutional Network)** represents a broad family of graph neural network architectures rather than a single model.

# Keywords (20–30)

Graph WaveNet, Traffic Forecasting, Graph Neural Network, Graph Convolution, Diffusion Convolution, Adaptive Adjacency Matrix, Adaptive Graph Learning, Spatial-Temporal Modeling, Temporal Convolution, Dilated Causal Convolution, Gated Temporal Convolution, Long-Range Temporal Dependencies, Hidden Spatial Dependencies, Dynamic Graph Learning, Traffic Sensors, Intelligent Transportation Systems, Time Series Forecasting, METR-LA, PEMS-BAY, Node Embeddings, End-to-End Learning, Residual Connections, Skip Connections, MAE, RMSE, MAPE