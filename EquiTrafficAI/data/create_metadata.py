import os
import json
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
metr_la_path = os.path.join(base_dir, 'metr_la_metrics.csv')
sd_path = os.path.join(base_dir, 'sd_meta.csv')

metadata = {
    "la": {},
    "sd": {}
}

# 1. Process METR-LA Sensors (207 nodes)
if os.path.exists(metr_la_path):
    df_la = pd.read_csv(metr_la_path)
    highways = ["US-101 Northbound", "I-5 Southbound", "I-10 Eastbound", "I-405 Southbound", "SR-110 Northbound"]
    exits = ["Hollywood Blvd", "Stadium Way", "Sunset Blvd", "Vermont Ave", "Western Ave", "Pasadena Fwy"]
    alternates = ["Sepulveda Blvd", "Mission Rd", "Olympic Blvd", "Wilshire Blvd", "Figueroa St"]
    
    for _, row in df_la.iterrows():
        node_id = str(int(row['node_id']))
        idx = int(node_id)
        h_name = highways[idx % len(highways)]
        e_name = exits[idx % len(exits)]
        a_name = alternates[idx % len(alternates)]
        speed_lim = 65 if row.get('road_type') == 'interstate' else 55
        
        metadata["la"][node_id] = {
            "highway_name": h_name,
            "speed_limit_mph": speed_lim,
            "nearest_exit": f"Exit {idx%20 + 1} - {e_name}",
            "alternate_route": a_name,
            "district": int(row.get('density', 0)) % 4
        }
    print(f"[+] Processed {len(df_la)} METR-LA Sensor Metadata entries.")

# 2. Process San Diego Sensors (716 nodes)
if os.path.exists(sd_path):
    df_sd = pd.read_csv(sd_path)
    for idx, row in df_sd.iterrows():
        node_id = str(int(row['ID']))
        fwy = str(row.get('Fwy', 'I-5'))
        direction = str(row.get('Direction', 'N'))
        lanes = int(row.get('Lanes', 4))
        
        metadata["sd"][node_id] = {
            "highway_name": f"{fwy} {direction}bound",
            "speed_limit_mph": 65,
            "nearest_exit": f"Exit {idx%30 + 1} - Main St",
            "alternate_route": "Pacific Highway",
            "district": int(row.get('District', 11)),
            "lanes": lanes
        }
    print(f"[+] Processed {len(df_sd)} San Diego Sensor Metadata entries.")

output_json = os.path.join(base_dir, 'sensor_metadata.json')
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)

file_size_kb = os.path.getsize(output_json) / 1024
print(f"[OK] Generated sensor_metadata.json ({file_size_kb:.1f} KB)")
