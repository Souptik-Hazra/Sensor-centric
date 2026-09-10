# EquiTraffic-GPT METR-LA Google Colab Export

This directory contains the self-contained Google Colab workflow for training and evaluating the PyTorch Graph WaveNet model on the 207-sensor METR-LA dataset.

## Run in Colab

1. Open `EquiTraffic_GWNet_Training.ipynb` in Google Colab.
2. Select **Runtime -> Change runtime type -> T4 GPU**.
3. Run all cells from top to bottom.
4. Download the checkpoint from the final cell.

The notebook order is:

1. Clone the repository and enter `colab_export`.
2. Verify GPU hardware and model architecture.
3. Train the model with validation metrics and checkpointing.
4. Generate dissertation evaluation figures.
5. Verify the model registry.
6. Package and download the trained checkpoint.

Use a fresh runtime after changing code. If imports fail, select **Runtime -> Restart session**, then run all cells again.

## Dissertation Figures

The evaluation cell saves figures under `figures/`:

- `metrics_and_speed_distribution.png`: validation MAE, RMSE, MAPE, R2, and observed speed distribution.
- `metr_la_speed_heatmap.png`: the first 24 hours of speed across all 207 sensors.
- `metr_la_sensor_topology.png`: spatial sensor coverage colored by mean observed speed.

The metrics are read from `code/model_registry.json`, so the charts represent the trained checkpoint.

## Metrics

- **MAE**: average prediction error in miles per hour.
- **RMSE**: error metric that emphasizes large mistakes.
- **MAPE**: relative percentage error; interpret carefully for near-zero speeds.
- **R2**: explained variance in observed speeds.

## Local VS Code Warning

The `/content/Sensor-centric/colab_export/code` path exists only inside Colab. VS Code may underline `from gwnet_model import GraphWaveNet` locally; the import works after the notebook clone/setup cell runs in Colab.

## Contents

- `code/`: model, dataset, loss, trainer, registry, and adapter modules.
- `data/`: METR-LA history, adjacency, distance, and sensor metadata files.
- `EquiTraffic_GWNet_Training.ipynb`: complete training and evaluation workflow.
