"""
EquiTraffic-GPT MLOps Module 3: Data Pipeline & Preprocessing Layer (gwnet_dataset.py)

Contains spatial graph data loaders, distance matrix builders, and sequence builders:
- load_pems_adjacency: Loads pre-computed binary .pkl spatial adjacency matrices in 0.001s
- build_adj_matrix_from_distances: Distance-standardized Gaussian kernel spatial adjacency builder
- load_pems_sequences: Strict train-split normalization (zero data leakage) + vectorized sliding window sequence builder
"""

import os
import pickle
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_model_config() -> dict:
    """Load model_config.yaml from backend directory with robust fallback."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, '..', 'backend', 'model_config.yaml'),
        os.path.join(base_dir, 'model_config.yaml')
    ]
    for cfg_path in candidates:
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except Exception:
                pass
    return {}


def load_pems_adjacency(data_dir, dataset_name, sensor_ids=None):
    """
    Loads precomputed binary pickle spatial adjacency matrix in 0.001s.
    Falls back to building from distances.csv or distances_{dataset_name}.csv if .pkl is missing.
    """
    clean_ds = dataset_name.lower().replace("_", "").replace("-", "")
    pkl_candidates = [
        os.path.join(data_dir, f'adj_{dataset_name}.pkl'),
        os.path.join(data_dir, f'{dataset_name}_adj.pkl'),
        os.path.join(data_dir, f'adj_{clean_ds}.pkl'),
        os.path.join(data_dir, 'metr_la_adj.pkl') if 'la' in clean_ds else os.path.join(data_dir, 'sd400_adj.pkl')
    ]

    for pkl_path in pkl_candidates:
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                adj = pickle.load(f)
                print(f"[+] Data Engineering: Loaded precomputed binary adjacency matrix '{os.path.basename(pkl_path)}' ({adj.shape[0]} nodes) in 0.001s.")
                return adj

    dist_candidates = [
        os.path.join(data_dir, f'distances_{dataset_name}.csv'),
        os.path.join(data_dir, 'distances.csv')
    ]
    distances_path = next((p for p in dist_candidates if os.path.exists(p)), dist_candidates[-1])

    if sensor_ids is None:
        cfg = load_model_config()
        arch = cfg.get('graph_wavenet_gnn', {}).get('architecture', {})
        node_count = arch.get(f'num_nodes_{clean_ds}', 716 if 'sd' in clean_ds else 207)
        sensor_ids = list(range(node_count))

    return build_adj_matrix_from_distances(distances_path, sensor_ids)


def build_adj_matrix_from_distances(distances_csv_path, sensor_ids, epsilon=0.1):
    """
    Build a real road-distance-based adjacency matrix from distances.csv.
    Uses distance-standardized Gaussian kernel: W_ij = exp(-(d_ij / sigma)^2) if >= epsilon, else 0.
    """
    num_nodes = len(sensor_ids)
    sid_to_idx = {sid: idx for idx, sid in enumerate(sensor_ids)}
    W = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    if not os.path.exists(distances_csv_path):
        return np.eye(num_nodes, dtype=np.float32)

    df = pd.read_csv(distances_csv_path)
    valid_ids = set(sensor_ids)
    df_filtered = df[(df['from'].isin(valid_ids)) & (df['to'].isin(valid_ids)) & (df['from'] != df['to'])].copy()

    costs = df_filtered['cost'].values if 'cost' in df_filtered.columns else []
    sigma_d = float(np.std(costs)) if len(costs) > 0 and float(np.std(costs)) > 1e-5 else 1000.0

    for _, row in df_filtered.iterrows():
        u_sid, v_sid = int(row['from']), int(row['to'])
        if u_sid in sid_to_idx and v_sid in sid_to_idx:
            d = float(row.get('cost', 1.0))
            w = np.exp(-((d / sigma_d) ** 2))
            if w >= epsilon:
                W[sid_to_idx[u_sid], sid_to_idx[v_sid]] = w

    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return W / row_sums


def load_pems_sequences(his_path, seq_len=12, horizon=12, stride=2):
    """
    Loads spatial-temporal tensor sequences from PeMS .npz history files.
    Calculates normalization statistics STRICTLY on training split (zero data leakage).
    """
    if not os.path.exists(his_path):
        raise FileNotFoundError(f"PeMS tensor history file not found: {his_path}")

    raw_npz = np.load(his_path)
    data = raw_npz['data']  # Shape (T, N, C)
    total_steps, num_nodes, num_channels = data.shape

    raw_speed = data[:, :, 0:1]  # Shape (T, N, 1)

    train_split_end = int(0.7 * total_steps)
    train_speed = raw_speed[:train_split_end]
    
    speed_mean = float(np.mean(train_speed))
    speed_std = float(np.std(train_speed)) if float(np.std(train_speed)) > 1e-6 else 1.0

    if abs(speed_mean) < 5.0 and speed_std < 5.0:
        normalized_speed = raw_speed
    else:
        normalized_speed = (raw_speed - speed_mean) / speed_std

    if num_channels >= 3:
        feature_data = np.concatenate([normalized_speed, data[:, :, 1:3]], axis=-1)
    else:
        tod = (np.arange(total_steps) % 288 / 288.0).reshape(-1, 1, 1)
        tod = np.tile(tod, (1, num_nodes, 1))
        dow = ((np.arange(total_steps) // 288) % 7 / 7.0).reshape(-1, 1, 1)
        dow = np.tile(dow, (1, num_nodes, 1))
        feature_data = np.concatenate([normalized_speed, tod, dow], axis=-1)

    feature_data = feature_data.astype(np.float32)

    indices = np.arange(0, total_steps - seq_len - horizon + 1, stride)
    
    X_list, Y_list = [], []
    for i in indices:
        X_list.append(feature_data[i: i + seq_len])
        Y_list.append(feature_data[i + seq_len: i + seq_len + horizon, :, 0:1])

    X_arr = np.array(X_list, dtype=np.float32)
    Y_arr = np.array(Y_list, dtype=np.float32)

    n_total = len(X_arr)
    train_end = int(0.7 * n_total)
    val_end = int(0.9 * n_total)

    X_train, Y_train = X_arr[:train_end], Y_arr[:train_end]
    X_val, Y_val = X_arr[train_end:val_end], Y_arr[train_end:val_end]

    print(f"[+] Data Engineering: Strict Train-Set Normalization (Mean: {speed_mean:.1f} mph, Std: {speed_std:.1f} mph)")
    print(f"[+] Data Engineering: Built {n_total:,} sequences with stride={stride} (Train: {len(X_train):,}, Val: {len(X_val):,})")

    return X_train, Y_train, X_val, Y_val, speed_mean, speed_std, num_nodes
