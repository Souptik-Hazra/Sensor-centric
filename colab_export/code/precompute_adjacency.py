"""
Pre-compute static spatial adjacency matrices for METR-LA and SD400 datasets into binary pickle files.
"""

import os
import pickle
import numpy as np
import pandas as pd
from gwnet_dataset import build_adj_matrix_from_distances

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'data')
os.makedirs(data_dir, exist_ok=True)

sensor_locs_path = os.path.join(data_dir, 'sensor_locations.csv')
distances_path = os.path.join(data_dir, 'distances.csv')

# 1. METR-LA (207 Nodes)
sensor_ids_la = list(range(207))
if os.path.exists(sensor_locs_path):
    df_locs = pd.read_csv(sensor_locs_path)
    sensor_ids_la = [int(row['sensor_id']) for _, row in df_locs.iterrows()]

adj_la = build_adj_matrix_from_distances(distances_path, sensor_ids_la)
pkl_la_path = os.path.join(data_dir, 'adj_metr_la.pkl')
with open(pkl_la_path, 'wb') as f:
    pickle.dump(adj_la, f)
print(f"[+] Saved static precomputed adjacency matrix: {pkl_la_path} (Shape: {adj_la.shape})")

# 2. SD400 (716 Nodes)
sd_meta_path = os.path.join(data_dir, 'sd_meta.csv')
num_sd_nodes = 716
if os.path.exists(sd_meta_path):
    df_sd = pd.read_csv(sd_meta_path)
    num_sd_nodes = len(df_sd)

adj_sd = np.eye(num_sd_nodes, dtype=np.float32)
pkl_sd_path = os.path.join(data_dir, 'adj_sd400.pkl')
with open(pkl_sd_path, 'wb') as f:
    pickle.dump(adj_sd, f)
print(f"[+] Saved static precomputed adjacency matrix: {pkl_sd_path} (Shape: {adj_sd.shape})")
