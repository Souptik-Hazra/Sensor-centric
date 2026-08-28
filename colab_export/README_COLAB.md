# 🚗 EquiTraffic-GPT METR-LA Google Colab Training Export

This standalone directory contains everything required to train, evaluate, and export PyTorch Graph WaveNet (GWNet) GNN models for the **METR-LA Highway Dataset (207 Sensors)** on **Google Colab** GPU.

---

### 📂 Directory Structure:

```text
colab_export/
├── EquiTraffic_GWNet_Training.ipynb   # METR-LA Google Colab Dissertation Training Notebook
├── README_COLAB.md                    # Setup & Execution Guide
├── code/                              # GWNet GNN Python Modules
│   ├── gwnet_model.py                 # Graph WaveNet Neural Architecture
│   ├── gwnet_dataset.py               # METR-LA Sequences & Spatial Adjacency Loader
│   ├── gwnet_loss.py                  # MAE, RMSE, MAPE, and R2 Evaluation Metrics
│   ├── gwnet_trainer.py               # PyTorch 2.x MLOps Model Trainer Engine
│   ├── gwnet_adapter.py               # Universal PeMS Serving Adapter
│   ├── gwnet_registry.py              # MLOps Model Version Registry Manifest
│   └── model_config.yaml              # GNN Model Configuration Parameters
├── data/                              # METR-LA Highway Sensors Dataset (207 Nodes)
│   ├── metr_la_metrics.csv            # Sensor Metadata & Persistence Errors
│   ├── sensor_locations.csv          # METR-LA Latitude/Longitude Coordinates
│   ├── distances.csv                  # Directed Spatial Graph Distance Matrix
│   ├── metr_la_his.npz                # Spatial-Temporal Speed Tensor (23,974 timesteps)
│   └── adj_metr_la.pkl                # Precomputed Binary Spatial Adjacency Matrix
└── checkpoints/                       # Model Weights Output Hierarchy
    └── v1.0.1/
        └── metr_la/                   # Saved PyTorch (.pt) Model Checkpoints
```

---

### 🚀 How to Run on Google Colab:

1. Open **Google Colab** (`colab.research.google.com`).
2. Select **Runtime -> Change runtime type -> T4 GPU**.
3. Open `EquiTraffic_GWNet_Training.ipynb` and click **Runtime -> Run all** (`Ctrl + F9`).
4. Training logs report **MAE (mph)**, **RMSE (mph)**, **MAPE (%)**, and **R²** determination coefficient for dissertation verification.
