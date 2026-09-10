"""Build a resumable, validated OSRM geometry cache for every directed edge."""

import json
import math
import os
import tempfile
import time

import pandas as pd
import requests


data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
locations_path = os.path.join(data_dir, "sensor_locations.csv")
distances_path = os.path.join(data_dir, "distances.csv")
cache_path = os.path.join(data_dir, "osrm_road_cache.json")

locations = pd.read_csv(locations_path)
distances = pd.read_csv(distances_path)
locs = {
    int(row.sensor_id): (float(row.latitude), float(row.longitude))
    for row in locations.itertuples()
}
valid_ids = set(locs)
edges = {
    (int(row["from"]), int(row["to"]))
    for _, row in distances.iterrows()
    if row["from"] in valid_ids
    and row["to"] in valid_ids
    and row["from"] != row["to"]
    and pd.notna(row["cost"])
    and float(row["cost"]) > 0
}


def haversine_miles(first, second):
    lat1, lon1 = first
    lat2, lon2 = second
    radius = 3958.8
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    value = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return radius * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))


def valid_geometry(coords, start, end):
    if not isinstance(coords, list) or len(coords) < 2:
        return False
    if any(not isinstance(point, list) or len(point) != 2 for point in coords):
        return False
    return (
        haversine_miles(coords[0], start) <= 0.25
        and haversine_miles(coords[-1], end) <= 0.25
    )


def save_cache(cache):
    directory = os.path.dirname(cache_path)
    handle, temporary_path = tempfile.mkstemp(prefix="osrm_cache_", suffix=".json", dir=directory)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            json.dump(cache, file, separators=(",", ":"))
            file.flush()
            os.fsync(file.fileno())
        for attempt in range(8):
            try:
                os.replace(temporary_path, cache_path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.5)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


cache = {}
if os.path.exists(cache_path):
    with open(cache_path, encoding="utf-8") as file:
        loaded = json.load(file)
    cache = {
        key: value
        for key, value in loaded.items()
        if "-" in key and valid_geometry(value, locs[int(key.split("-")[0])], locs[int(key.split("-")[1])])
    }

pending = sorted(edges - {
    (int(key.split("-")[0]), int(key.split("-")[1])) for key in cache
})
print(f"[OSRM Cache Builder] Graph edges: {len(edges)}")
print(f"[OSRM Cache Builder] Valid cached edges: {len(cache)}")
print(f"[OSRM Cache Builder] Missing edges to fetch: {len(pending)}")

session = requests.Session()
new_count = 0
osrm_count = 0
fallback_count = 0
retry_count = 0
checkpoint_interval = 25

for from_id, to_id in pending:
    start = locs[from_id]
    end = locs[to_id]
    key = f"{from_id}-{to_id}"
    coords = None
    for attempt in range(3):
        try:
            url = (
                "http://router.project-osrm.org/route/v1/driving/"
                f"{start[1]:.6f},{start[0]:.6f};{end[1]:.6f},{end[0]:.6f}"
                "?overview=full&geometries=geojson"
            )
            response = session.get(url, timeout=10)
            route_data = response.json() if response.status_code == 200 else {}
            raw_coords = route_data.get("routes", [{}])[0].get("geometry", {}).get("coordinates", [])
            candidate = [[point[1], point[0]] for point in raw_coords]
            if valid_geometry(candidate, start, end):
                coords = candidate
                osrm_count += 1
                break
        except (ValueError, IndexError, KeyError, requests.RequestException):
            pass
        if attempt < 2:
            retry_count += 1
            time.sleep(1.5 * (attempt + 1))

    if coords is None:
        coords = [[start[0], start[1]], [end[0], end[1]]]
        fallback_count += 1
    cache[key] = coords
    new_count += 1

    if new_count % checkpoint_interval == 0:
        save_cache(cache)
        print(f"  checkpoint: {len(cache)}/{len(edges)} cached")
    time.sleep(0.2)

save_cache(cache)
print("[OSRM Cache Builder] Complete")
print(f"  graph edges: {len(edges)}")
print(f"  cached edges: {len(cache)}")
print(f"  new OSRM geometries: {osrm_count}")
print(f"  straight-line fallbacks: {fallback_count}")
print(f"  retries: {retry_count}")
