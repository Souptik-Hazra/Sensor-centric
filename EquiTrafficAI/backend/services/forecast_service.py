"""Forecast service for trained Graph WaveNet inference."""

from typing import Any

import numpy as np
from .speed_utils import standardized_speed_to_mph


def predict_congestion_15min(
    city: str,
    timestamp_index: int,
    state_data: dict[str, Any],
    gwnet_adapters: dict[str, Any],
) -> dict[str, Any]:
    """Return trained-model 15-minute speeds with historical fallback."""
    city_key = city.lower()
    city_data = state_data.get(city_key, state_data["la"])
    sensors = city_data.get("sensors", [])

    his_data = state_data.get("his_npz_sd" if city_key == "sd" else "his_npz_la")
    predicted_15min_speeds = {}
    forecast_source = "historical_fallback"

    adapter = gwnet_adapters.get(city_key)
    if his_data is not None and adapter is not None and adapter.checkpoint_loaded and his_data.shape[0] > 0:
        try:
            total_steps = his_data.shape[0]
            current_idx = max(0, min(total_steps - 1, timestamp_index))
            window = his_data[max(0, current_idx - 11):current_idx + 1]
            if len(window) < 12:
                window = np.concatenate(
                    [np.repeat(window[:1], 12 - len(window), axis=0), window],
                    axis=0,
                )
            model_output = adapter.predict_next_15min(window)
            # The model emits 12 five-minute horizons; index 2 is +15 min.
            future_15min_slice = model_output[2]
            forecast_source = "gwnet"
            for idx, sensor in enumerate(sensors):
                if idx < len(future_15min_slice):
                    value = standardized_speed_to_mph(
                        future_15min_slice[idx], 55.0
                    )
                    predicted_15min_speeds[sensor.get("id")] = round(value, 1)
        except Exception as error:
            print(f"[!] GWNet 15-min inference error; using historical fallback: {error}")

    if not predicted_15min_speeds and his_data is not None:
        try:
            total_steps = his_data.shape[0]
            start_idx = max(0, min(total_steps - 4, timestamp_index))
            future_15min_slice = his_data[start_idx + 3, :, 0]
            for idx, sensor in enumerate(sensors):
                if idx < len(future_15min_slice):
                    value = standardized_speed_to_mph(
                        future_15min_slice[idx], sensor.get("speed", 55.0)
                    )
                    predicted_15min_speeds[sensor.get("id")] = round(
                        value, 1
                    )
        except Exception as error:
            print(f"[!] Historical 15-min fallback error: {error}")

    congested_nodes = []
    for sensor in sensors:
        node_id = sensor.get("id")
        current_speed = sensor.get("speed", 55.0)
        future_speed = predicted_15min_speeds.get(
            node_id,
            round(max(10.0, current_speed - 12.0), 1),
        )

        if future_speed < 30.0:
            congested_nodes.append({
                "sensor_id": sensor.get("sensor_id", node_id),
                "id": node_id,
                "location_label": sensor.get("location_label", f"Corridor Sensor #{node_id}"),
                "current_speed": round(float(current_speed), 1),
                "predicted_speed": round(float(future_speed), 1),
                "speed_drop": round(float(current_speed - future_speed), 1),
                "lat": sensor.get("lat"),
                "lon": sensor.get("lon"),
                "warning": f"15-Min Neural Bottleneck Spike ({round(future_speed, 1)} mph)",
            })

    return {
        "city": city_key,
        "horizon": "15-min",
        "timestamp_index": timestamp_index,
        "source": forecast_source,
        "congested_sensors_count": len(congested_nodes),
        "congested_nodes": congested_nodes[:10],
        "predicted_speeds": {
            str(sensor_id): speed
            for sensor_id, speed in predicted_15min_speeds.items()
        },
    }
