# 🚗 EquiTraffic-GPT Google Colab Training Export

This standalone directory contains everything required to train, evaluate, and export PyTorch Graph WaveNet (GWNet) GNN models on **Google Colab** or any GPU cloud server.

---

### 📂 Directory Structure:

```text
colab_export/
├── EquiTraffic_GWNet_Training.ipynb   # Standalone Google Colab Training Notebook
├── code/                              # GWNet GNN Python Modules
│   ├── gwnet_model.py
│   ├── gwnet_dataset.py
│   ├── gwnet_loss.py
│   ├── gwnet_trainer.py
│   ├── gwnet_adapter.py
│   ├── gwnet_registry.py
│   ├── precompute_adjacency.py
│   └── model_config.yaml
├── data/                              # Datasets & Adjacency Matrices
│   ├── metr_la_metrics.csv
│   ├── sensor_locations.csv
│   ├── distances.csv
│   ├── metr_la_his.npz
│   ├── sd_meta.csv
│   ├── sd400_his.npz
│   ├── adj_metr_la.pkl
│   └── adj_sd400.pkl
└── checkpoints/                       # Saving Location for Checkpoint Weights
    └── v1.0.1/
        ├── metr_la/
        └── sd400/
```

---

### 🚀 How to Run on Google Colab:

1. Upload the entire `colab_export/` folder to your **Google Drive** or clone your GitHub repository.
2. Open `EquiTraffic_GWNet_Training.ipynb` in Google Colab.
3. Select **Runtime -> Change runtime type -> T4 GPU**.
4. Run all cells to train Graph WaveNet GNN models and save model weights (`.pt`) to `checkpoints/`.
