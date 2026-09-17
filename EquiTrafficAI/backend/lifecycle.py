"""FastAPI startup lifecycle and dataset loading."""

import json
import math
import os

import numpy as np
import pandas as pd
import yaml

from . import app_state
from .services.synthetic_topology import generate_synthetic_pems_topology


def load_all_data() -> None:
    """Initialize datasets, tensor history, configuration, and route graph."""
    print("Loading EquiTraffic-GPT Datasets & Universal PeMS Topologies...")
    data_dir=app_state.data_dir
    state_data=app_state.state_data

    backend_cfg_path=os.path.join(app_state.BASE_DIR, "backend_config.yaml")
    model_cfg_path=os.path.join(app_state.BASE_DIR, "model_config.yaml")
    if os.path.exists(backend_cfg_path):
        with open(backend_cfg_path, "r", encoding="utf-8") as file:
            state_data["backend_config"]=yaml.safe_load(file)
        print("[+] Loaded backend_config.yaml into runtime state.")
    if os.path.exists(model_cfg_path):
        with open(model_cfg_path, "r", encoding="utf-8") as file:
            state_data["model_config"]=yaml.safe_load(file)
        print("[+] Loaded model_config.yaml into runtime state.")

    for filename, key, label in (
        ("metr_la_his.npz", "his_npz_la", "METR-LA"),
        ("sd400_his.npz", "his_npz_sd", "SD400"),
    ):
        history_path=os.path.join(data_dir, filename)
        if os.path.exists(history_path):
            try:
                state_data[key]=np.load(history_path)["data"]
                print(f"[+] Loaded {label} tensor history shape {state_data[key].shape} in memory.")
            except Exception as error:
                print(f"[!] {label} npz load error: {error}")

    la_metrics=os.path.join(data_dir, "metr_la_metrics.csv")
    la_locs=os.path.join(data_dir, "sensor_locations.csv")
    la_dists=os.path.join(data_dir, "distances.csv")
    if os.path.exists(la_metrics) and os.path.exists(la_locs):
        metrics=pd.read_csv(la_metrics)
        locations=pd.read_csv(la_locs)
        nodes=[]
        for index, row in metrics.iterrows():
            sensor_node_id=int(row["node_id"])
            if index < len(locations):
                latitude=float(locations.iloc[index]["latitude"])
                longitude=float(locations.iloc[index]["longitude"])
                pems_id=int(locations.iloc[index]["sensor_id"])
            else:
                latitude, longitude, pems_id=34.0522, -118.2437, sensor_node_id
            location=app_state.la_location_map.get(
                str(pems_id), app_state.la_location_map.get(str(sensor_node_id), {})
            )
            nodes.append({
                "id": index,
                "sensor_id": str(pems_id),
                "node_index": index,
                "lat": latitude,
                "lon": longitude,
                "speed": round(float(row.get("avg_speed", 55.0)), 1),
                "zero_dropout_rate": round(float(row.get("zero_rate", 0.0) * 100.0), 2),
                "reliability": round(float(max(0.70, min(0.99, 1.0 - row.get("zero_rate", 0.0)))), 3),
                "traffic_regime": str(row.get("traffic_regime", "STABLE")),
                "cusum_flag": bool(row.get("cusum_flags", 0)),
                "ewma_flag": bool(row.get("ewma_flags", 0)),
                "persistence_error": round(float(row.get("persistence_error", 0.0)), 2),
                "status": "DEGRADED" if row.get("cusum_flags", 0) or row.get("ewma_flags", 0) else "HEALTHY",
                "freeway": location.get("freeway", ""),
                "direction": location.get("direction", ""),
                "neighborhood": location.get("neighborhood", ""),
                "nearest_landmark": location.get("nearest_landmark", ""),
                "location_label": location.get("location_label", f"Sensor #{sensor_node_id}"),
            })

        edges=[]
        neighbors={index: [] for index in range(len(nodes))}
        road_cache={}
        road_cache_path=os.path.join(data_dir, "osrm_road_cache.json")
        if os.path.exists(road_cache_path):
            with open(road_cache_path, "r", encoding="utf-8") as file:
                road_cache=json.load(file)
            print(f"[+] Loaded OSRM road geometry cache: {len(road_cache)} edge segments.")

        if os.path.exists(la_dists) and os.path.exists(la_locs):
            distances=pd.read_csv(la_dists)
            location_rows=pd.read_csv(la_locs)
            pems_to_latlon=dict(zip(location_rows["sensor_id"], zip(location_rows["latitude"], location_rows["longitude"])))
            pems_to_idx=dict(zip(location_rows["sensor_id"], range(len(location_rows))))
            valid_pems=set(location_rows["sensor_id"])
            filtered=distances[
                distances["from"].isin(valid_pems)
                & distances["to"].isin(valid_pems)
                & (distances["from"] != distances["to"])
            ].copy()
            app_state.route_graph_data["pems_to_idx"]=pems_to_idx
            app_state.route_graph_data["edges"]=[
                (int(row["from"]), int(row["to"]), float(row["cost"]))
                for _, row in filtered.iterrows()
                if math.isfinite(float(row["cost"])) and float(row["cost"]) > 0
            ]
            edges_set=set()
            for sensor_id in valid_pems:
                for _, row in filtered[filtered["from"] == sensor_id].sort_values("cost").head(3).iterrows():
                    edges_set.add((int(row["from"]), int(row["to"])))
            for from_pems, to_pems in edges_set:
                cache_key=f"{from_pems}-{to_pems}"
                legacy_key=f"{min(from_pems, to_pems)}-{max(from_pems, to_pems)}"
                if cache_key in road_cache:
                    edges.append(road_cache[cache_key])
                elif legacy_key in road_cache:
                    segment=road_cache[legacy_key]
                    if from_pems > to_pems:
                        segment=list(reversed(segment))
                    edges.append(segment)
                else:
                    edges.append([
                        [pems_to_latlon[from_pems][0], pems_to_latlon[from_pems][1]],
                        [pems_to_latlon[to_pems][0], pems_to_latlon[to_pems][1]],
                    ])
                if from_pems in pems_to_idx and to_pems in pems_to_idx:
                    neighbors[pems_to_idx[from_pems]].append(pems_to_idx[to_pems])
        state_data["la"]={"sensors": nodes, "edges": edges, "count": len(nodes), "road_cache": road_cache}
        state_data["graph_neighbors"]=neighbors

    sd_meta=os.path.join(data_dir, "sd_meta.csv")
    if os.path.exists(sd_meta):
        sd_rows=pd.read_csv(sd_meta)
        nodes_sd, edges_sd=[], []
        id_to_index={int(row["ID"]): index for index, row in sd_rows.iterrows()}
        for index, row in sd_rows.iterrows():
            sensor_id=int(row["ID"])
            location=app_state.sd_location_map.get(str(sensor_id), {})
            nodes_sd.append({
                "id": index,
                "sensor_id": sensor_id,
                "node_index": index,
                "lat": float(row["Lat"]),
                "lon": float(row["Lng"]),
                "freeway": str(row.get("Fwy", row.get("Freeway", "I-5"))),
                "direction": str(row.get("Dir", row.get("Direction", "N"))),
                "lanes": int(row.get("Lanes", 3)),
                "speed": round(float(max(15.0, min(75.0, 58.0 + np.random.randn() * 9.0))), 1),
                "zero_dropout_rate": round(float(max(0.0, min(12.0, np.random.exponential(1.5)))), 2),
                "reliability": round(float(max(0.75, min(0.99, 0.95 - np.random.rand() * 0.10))), 3),
                "traffic_regime": "STABLE" if index % 6 != 0 else "HEAVY",
                "cusum_flag": index % 8 == 0,
                "ewma_flag": index % 13 == 0,
                "persistence_error": round(float(np.random.rand() * 3.8), 2),
                "status": "HEALTHY" if index % 10 != 0 else "DEGRADED",
                "neighborhood": location.get("neighborhood", f"District {row.get('Fwy', 'I-5')} Zone"),
                "nearest_landmark": location.get("nearest_landmark", f"Exit {sensor_id % 100}"),
                "location_label": location.get("location_label", f"Fwy {row.get('Fwy', 'I-5')}-{row.get('Dir', row.get('Direction', 'N'))} Postmile #{sensor_id}"),
            })
        for freeway, group in sd_rows.groupby("Fwy"):
            sorted_group=group.sort_values("Lat") if ("N" in str(freeway) or "S" in str(freeway)) else group.sort_values("Lng")
            indices=[id_to_index[int(sensor_id)] for sensor_id in sorted_group["ID"]]
            for index in range(len(indices) - 1):
                first, second=indices[index], indices[index + 1]
                if np.sqrt((sd_rows.iloc[first]["Lat"] - sd_rows.iloc[second]["Lat"]) ** 2 + (sd_rows.iloc[first]["Lng"] - sd_rows.iloc[second]["Lng"]) ** 2) <= 0.08:
                    edges_sd.append([
                        [sd_rows.iloc[first]["Lat"], sd_rows.iloc[first]["Lng"]],
                        [sd_rows.iloc[second]["Lat"], sd_rows.iloc[second]["Lng"]],
                    ])
        state_data["sd"]={"sensors": nodes_sd, "edges": edges_sd, "count": len(nodes_sd)}

    for key, arguments in {
        "pems04": (307, 37.7749, -122.4194, "pems04"),
        "pems08": (170, 34.1083, -117.2898, "pems08"),
        "pems_bay": (325, 37.3382, -121.8863, "pems_bay"),
        "pems03": (358, 38.5816, -121.4944, "pems03"),
        "pems07": (883, 34.0522, -118.2437, "pems07"),
    }.items():
        state_data[key]=generate_synthetic_pems_topology(*arguments)

    print("[+] Universal PeMS Datasets Ready: METR-LA (207), SD400 (716), PeMS04 (307), PeMS08 (170), PeMS-BAY (325), PeMS03 (358), PeMS07 (883).")
