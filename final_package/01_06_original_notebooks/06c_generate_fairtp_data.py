import numpy as np
import pandas as pd
import json
import os
import pickle
import shutil
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from sklearn.cluster import KMeans

def main():
    out_dir = '../FairTP/data/metr-la/2019'
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading METR-LA dataset...")
    train = load_dataset("witgaw/METR-LA", split="train")

    CURRENT_SPEED = "x_t+0_d0" if "x_t+0_d0" in train.column_names else "x_t-0_d0"
    print(f"Using {CURRENT_SPEED} as current speed.")
    
    print("Extracting traffic data...")
    traffic = train.select_columns(["node_id", "t0_timestamp", CURRENT_SPEED]).to_pandas()
    traffic = traffic.rename(columns={CURRENT_SPEED: "speed"})
    traffic["node_id"] = traffic["node_id"].astype(int)
    traffic["t0_timestamp"] = pd.to_datetime(traffic["t0_timestamp"])
    traffic = traffic.sort_values(["node_id", "t0_timestamp"]).reset_index(drop=True)

    print("Pivoting data...")
    # Pivot so rows=time, cols=nodes
    df = traffic.pivot(index="t0_timestamp", columns="node_id", values="speed")
    df = df.sort_index()

    # Ensure we have exactly 207 columns in the right order
    assert df.shape[1] == 207
    num_samples, num_nodes = df.shape
    print(f"Data shape: {num_samples} time steps, {num_nodes} nodes.")
    
    # Base data (T, N, 1)
    data2 = np.expand_dims(df.values, axis=-1)

    # Add time features
    time_ind = (df.index.values - df.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
    time_of_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))

    dow = df.index.dayofweek
    dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
    day_of_week = dow_tiled / 7

    data = np.concatenate([data2, time_of_day, day_of_week], axis=-1)
    print("Data with time shape:", data.shape)
    print("Data without time shape:", data2.shape)
    
    seq_length_x = 12
    seq_length_y = 12

    min_t = seq_length_x
    max_t = num_samples - seq_length_y
    idx = np.arange(min_t, max_t, 1)

    num_idx = len(idx)
    
    # Matching FairTP 60/20/20
    num_train = round(num_idx * 0.6)
    num_val = round(num_idx * 0.2)

    idx_train = idx[:num_train]
    idx_val = idx[num_train: num_train + num_val]
    idx_test = idx[num_train + num_val:]

    print(f"Train size: {len(idx_train)}, Val size: {len(idx_val)}, Test size: {len(idx_test)}")
    
    # Normalize based on train set ONLY
    x_train = data2[:idx_val[0] - seq_length_x, :, 0]
    mean = x_train.mean()
    std = x_train.std()

    data2[..., 0] = (data2[..., 0] - mean) / std
    data[..., 0] = (data[..., 0] - mean) / std

    print(f"Normalized with mean={mean:.3f}, std={std:.3f}")
    
    print("Saving Features...")
    np.savez_compressed(os.path.join(out_dir, 'his.npz'), data=data, mean=mean, std=std)
    np.savez_compressed(os.path.join(out_dir, 'his_notime.npz'), data=data2, mean=mean, std=std)

    np.save(os.path.join(out_dir, 'idx_train.npy'), idx_train)
    np.save(os.path.join(out_dir, 'idx_val.npy'), idx_val)
    np.save(os.path.join(out_dir, 'idx_test.npy'), idx_test)
    print("Saved .npz and idx files.")
    
    print("Fetching Adjacency & Compute Districts...")
    adj_path = hf_hub_download(repo_id="witgaw/METR-LA", filename="sensor_graph/adj_mx.npy", repo_type="dataset")
    loc_path = hf_hub_download(repo_id="witgaw/METR-LA", filename="sensor_graph/sensor_locations.csv", repo_type="dataset")

    # Copy adj_mx.npy to out_dir
    shutil.copy(adj_path, os.path.join(out_dir, 'adj_mx.npy'))

    # Create the binary version
    adj = np.load(adj_path)
    adj_bin = (adj > 0).astype(float)
    np.save(os.path.join(out_dir, 'adj_mx_all1.npy'), adj_bin)
    print("Saved adjacency matrices.")
    
    # Generate Districts via KMeans on coordinates
    locs = pd.read_csv(loc_path)

    # Ensure order matches node_ids 0..206
    mapping_path = hf_hub_download(repo_id="witgaw/METR-LA", filename="sensor_graph/adj_mx_mapping.json", repo_type="dataset")
    with open(mapping_path, "r") as f:
        mapping = json.load(f)

    sensor_ids_ordered = mapping["sensor_ids"]
    coords = []
    for s_id in sensor_ids_ordered:
        row = locs[locs["sensor_id"] == int(s_id)]
        coords.append([row["latitude"].values[0], row["longitude"].values[0]])

    coords = np.array(coords)

    # KMeans clustering to create 13 districts (matching SD dataset)
    kmeans = KMeans(n_clusters=13, random_state=42)
    districts = kmeans.fit_predict(coords)

    sd_district = {}
    for i in range(13):
        sd_district[i] = np.where(districts == i)[0].tolist()

    with open(os.path.join(out_dir, 'metr_la_district.json'), 'wb') as f:
        pickle.dump(sd_district, f)
        
    print("Saved metr_la_district.json with 13 spatial clusters.")
    print("District sizes:", [len(v) for v in sd_district.values()])

if __name__ == '__main__':
    main()
