"""
Pre-cache OSRM road geometries for ALL METR-LA sensor edge pairs shown on the map.
Matches the exact same edge selection logic as backend.py initialize_state().
Saves to EquiTrafficAI/data/osrm_road_cache.json
"""
import json, time, requests, pandas as pd, os

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
df_l = pd.read_csv(os.path.join(data_dir, 'sensor_locations.csv'))
df_d = pd.read_csv(os.path.join(data_dir, 'distances.csv'))

locs = dict(zip(df_l['sensor_id'], zip(df_l['latitude'], df_l['longitude'])))
valid_pems = set(df_l['sensor_id'])

# Match EXACT same edge selection as backend.py: top-3 nearest neighbors per sensor
filtered = df_d[df_d['from'].isin(valid_pems) & df_d['to'].isin(valid_pems) & (df_d['from'] != df_d['to'])].copy()
edges_set = set()
for sid in valid_pems:
    sub = filtered[filtered['from'] == sid].sort_values('cost')
    for _, r in sub.head(3).iterrows():
        edges_set.add((int(r['from']), int(r['to'])))

# Deduplicate into undirected pairs
pairs = set()
for a, b in edges_set:
    pairs.add((min(a, b), max(a, b)))

print(f"[OSRM Cache Builder] {len(pairs)} unique sensor edge pairs to cache (all map edges)...")

# Load existing cache to avoid re-fetching
cache_path = os.path.join(data_dir, 'osrm_road_cache.json')
cache = {}
if os.path.exists(cache_path):
    with open(cache_path, 'r') as f:
        cache = json.load(f)
    print(f"  Loaded existing cache with {len(cache)} entries. Will only fetch missing ones.")

new_count = 0
failed = 0
BATCH_SIZE = 10

for i, (a, b) in enumerate(sorted(pairs)):
    key = f"{a}-{b}"
    if key in cache:
        continue  # Already cached

    lat_a, lon_a = locs[a]
    lat_b, lon_b = locs[b]

    try:
        url = (f"http://router.project-osrm.org/route/v1/driving/"
               f"{lon_a:.6f},{lat_a:.6f};{lon_b:.6f},{lat_b:.6f}"
               f"?overview=full&geometries=geojson")
        resp = requests.get(url, timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("routes") and len(data["routes"]) > 0:
                coords = [[lat, lon] for lon, lat in data["routes"][0]["geometry"]["coordinates"]]
                cache[key] = coords
            else:
                cache[key] = [[lat_a, lon_a], [lat_b, lon_b]]
                failed += 1
        else:
            cache[key] = [[lat_a, lon_a], [lat_b, lon_b]]
            failed += 1
    except Exception:
        cache[key] = [[lat_a, lon_a], [lat_b, lon_b]]
        failed += 1

    new_count += 1
    if new_count % 50 == 0:
        print(f"  [{new_count} new] cached... ({failed} fallbacks)")

    if new_count % BATCH_SIZE == 0:
        time.sleep(0.3)

with open(cache_path, 'w') as f:
    json.dump(cache, f, separators=(',', ':'))

file_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
print(f"\n[OSRM Cache Builder] Done! Total cache: {len(cache)} edges ({new_count} new, {failed} fallbacks).")
print(f"[OSRM Cache Builder] Saved to {cache_path} ({file_size_mb:.2f} MB)")
