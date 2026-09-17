"""Synthetic topology generation for unsupported PeMS demo corridors."""

from typing import Any

import numpy as np


def generate_synthetic_pems_topology(
    num_nodes: int,
    center_lat: float,
    center_lon: float,
    dataset_id: str,
) -> dict[str, Any]:
    """Generate a deterministic synthetic sensor topology for a PeMS dataset."""
    np.random.seed(42)
    nodes, edges=[], []
    latitudes=center_lat + np.cumsum(np.random.randn(num_nodes) * 0.003)
    longitudes=center_lon + np.cumsum(np.random.randn(num_nodes) * 0.003)

    for index in range(num_nodes):
        nodes.append({
            "id": index,
            "sensor_id": 1000 + index,
            "speed": round(float(max(15.0, min(70.0, 52.0 + np.random.randn() * 8.0))), 1),
            "lat": round(float(latitudes[index]), 5),
            "lon": round(float(longitudes[index]), 5),
            "zero_dropout_rate": round(float(max(0.0, min(15.0, np.random.exponential(2.0)))), 2),
            "reliability": round(float(max(0.70, min(0.99, 0.94 - np.random.rand() * 0.15))), 3),
            "traffic_regime": "STABLE" if index % 5 != 0 else "CONGESTED",
            "cusum_flag": index % 7 == 0,
            "ewma_flag": index % 11 == 0,
            "persistence_error": round(float(np.random.rand() * 4.5), 2),
            "status": "HEALTHY" if index % 9 != 0 else "DEGRADED",
            "freeway": f"I-{5 + (index % 4) * 10}",
            "direction": "N" if index % 2 == 0 else "S",
            "neighborhood": f"District {dataset_id.upper()} Zone {index // 20}",
            "nearest_landmark": f"Corridor Marker #{index}",
            "location_label": f"{dataset_id.upper()} Highway Sensor #{index}",
        })
        if index > 0:
            edges.append([
                [nodes[index - 1]["lat"], nodes[index - 1]["lon"]],
                [nodes[index]["lat"], nodes[index]["lon"]],
            ])

    return {"sensors": nodes, "edges": edges, "count": len(nodes)}
