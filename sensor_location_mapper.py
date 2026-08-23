"""
METR-LA & SD400 Real-World Sensor Location Mapper

Maps every sensor node to its actual freeway, direction, nearest neighborhood,
and nearby landmarks using reverse geocoding from lat/lon coordinates.

This mapping feeds into:
1. LLM Engine — so Gemini Flash 2.5 Lite gives real location-aware advice
2. GWNet Adjacency — so spatial graph edges follow actual freeway topology
3. Frontend Map — so popups show real freeway names, not just node IDs
"""

import os
import json
import numpy as np
import pandas as pd

# ============================================================================
# METR-LA: 207 sensors on Los Angeles County highways
# Known freeways in METR-LA coverage area (lat 34.04–34.22, lon -118.54 to -118.18):
#   I-5, I-10, I-101 (US-101), I-110, I-134, I-210, I-405, SR-2, SR-134, SR-170
# ============================================================================

# LA Freeway corridor bounding boxes [lat_min, lat_max, lon_min, lon_max, fwy_name, direction]
LA_FREEWAY_CORRIDORS = [
    # I-5 (Golden State / Santa Ana Fwy) — runs N-S through central-east LA
    (34.04, 34.13, -118.28, -118.22, "I-5", "N/S", "Downtown LA / Glendale"),
    (34.13, 34.22, -118.28, -118.22, "I-5", "N/S", "Burbank / Sun Valley"),
    
    # I-10 (Santa Monica Fwy) — runs E-W through south-central LA
    (34.04, 34.08, -118.40, -118.22, "I-10", "E/W", "Mid-City / Downtown LA"),
    
    # US-101 (Hollywood Fwy) — runs NW-SE through Hollywood / Downtown
    (34.06, 34.12, -118.35, -118.24, "US-101", "N/S", "Hollywood / Echo Park"),
    (34.12, 34.18, -118.40, -118.35, "US-101", "N/S", "Studio City / Sherman Oaks"),
    
    # I-110 (Harbor Fwy) — runs N-S south of downtown
    (34.04, 34.08, -118.30, -118.26, "I-110", "N/S", "Downtown LA / South LA"),
    
    # I-405 (San Diego Fwy) — runs N-S on the west side
    (34.04, 34.10, -118.54, -118.44, "I-405", "N/S", "West LA / Culver City"),
    (34.10, 34.18, -118.54, -118.44, "I-405", "N/S", "Sherman Oaks / Encino"),
    (34.18, 34.22, -118.54, -118.44, "I-405", "N/S", "Granada Hills / Mission Hills"),
    
    # I-210 (Foothill Fwy) — runs E-W in the north
    (34.18, 34.22, -118.44, -118.18, "I-210", "E/W", "Pasadena / La Canada"),
    
    # SR-134 (Ventura Fwy east segment) — runs E-W through Glendale
    (34.13, 34.17, -118.35, -118.22, "SR-134", "E/W", "Glendale / Eagle Rock"),
    
    # SR-170 (Hollywood Fwy north segment) — runs N-S in North Hollywood
    (34.14, 34.22, -118.40, -118.36, "SR-170", "N/S", "North Hollywood / Sun Valley"),
    
    # SR-2 (Glendale Fwy) — runs N-S near Glendale
    (34.08, 34.16, -118.24, -118.20, "SR-2", "N/S", "Glendale / Eagle Rock"),
]

# LA Landmark proximity zones [lat, lon, radius_deg, landmark_name]
LA_LANDMARKS = [
    (34.0736, -118.2400, 0.015, "Dodger Stadium"),
    (34.0522, -118.2437, 0.012, "Downtown LA / City Hall"),
    (34.1381, -118.3534, 0.015, "Universal Studios / Studio City"),
    (34.0928, -118.3287, 0.012, "Hollywood / Vine"),
    (34.1478, -118.1445, 0.015, "Rose Bowl / Pasadena"),
    (34.0141, -118.2879, 0.015, "USC / Exposition Park"),
    (34.1613, -118.1676, 0.012, "JPL / La Canada"),
    (34.0195, -118.4912, 0.015, "Santa Monica / PCH"),
    (34.0430, -118.2673, 0.010, "Union Station"),
    (34.1866, -118.3815, 0.012, "Van Nuys / Sherman Oaks"),
    (34.2018, -118.4735, 0.012, "Northridge"),
    (34.0575, -118.4180, 0.012, "Westwood / UCLA"),
    (34.1008, -118.4950, 0.012, "Brentwood / I-405 Getty"),
]


def map_la_sensor_to_location(lat, lon, sensor_id):
    """Map a METR-LA sensor to its freeway, direction, neighborhood, and nearest landmark."""
    
    best_fwy = "Local Arterial"
    best_dir = ""
    best_neighborhood = "Los Angeles"
    
    # Find matching freeway corridor
    best_dist = float('inf')
    for lat_min, lat_max, lon_min, lon_max, fwy, direction, neighborhood in LA_FREEWAY_CORRIDORS:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            center_lat = (lat_min + lat_max) / 2
            center_lon = (lon_min + lon_max) / 2
            dist = np.sqrt((lat - center_lat)**2 + (lon - center_lon)**2)
            if dist < best_dist:
                best_dist = dist
                best_fwy = fwy
                best_dir = direction
                best_neighborhood = neighborhood
    
    # Find nearest landmark
    nearest_landmark = ""
    min_landmark_dist = float('inf')
    for lm_lat, lm_lon, radius, lm_name in LA_LANDMARKS:
        d = np.sqrt((lat - lm_lat)**2 + (lon - lm_lon)**2)
        if d < radius and d < min_landmark_dist:
            min_landmark_dist = d
            nearest_landmark = lm_name
    
    return {
        "sensor_id": int(sensor_id),
        "lat": float(lat),
        "lon": float(lon),
        "freeway": best_fwy,
        "direction": best_dir,
        "neighborhood": best_neighborhood,
        "nearest_landmark": nearest_landmark,
        "location_label": f"{best_fwy} {best_dir} near {best_neighborhood}" + (f" ({nearest_landmark})" if nearest_landmark else "")
    }


def build_la_sensor_map():
    """Build full METR-LA 207-sensor location map."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    df_loc = pd.read_csv(os.path.join(data_dir, 'sensor_locations.csv'))
    
    sensor_map = {}
    for idx, row in df_loc.iterrows():
        info = map_la_sensor_to_location(row['latitude'], row['longitude'], row['sensor_id'])
        info['node_index'] = idx
        sensor_map[int(row['sensor_id'])] = info
    
    return sensor_map


def build_sd_sensor_map():
    """Build full SD400 716-sensor location map from sd_meta.csv (already has Fwy, Direction, County)."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    df_sd = pd.read_csv(os.path.join(data_dir, 'sd_meta.csv'))
    
    # SD Landmark proximity zones
    SD_LANDMARKS = [
        (32.7831, -117.1196, 0.015, "Snapdragon Stadium / SDSU"),
        (32.7073, -117.1567, 0.012, "Petco Park / Gaslamp"),
        (32.7157, -117.1611, 0.012, "Downtown San Diego"),
        (32.7338, -117.1494, 0.012, "Balboa Park / San Diego Zoo"),
        (32.7335, -117.1960, 0.012, "Old Town San Diego"),
        (32.8328, -117.1440, 0.015, "Miramar / MCAS"),
        (32.8801, -117.2368, 0.012, "Del Mar / Torrey Pines"),
        (32.6282, -117.0493, 0.015, "Otay Mesa / Border"),
        (33.1270, -117.3044, 0.015, "Oceanside / Camp Pendleton"),
        (32.7785, -117.0714, 0.012, "La Mesa / El Cajon"),
        (32.7460, -117.1650, 0.012, "Hillcrest / Mission Hills"),
        (32.6697, -117.0983, 0.012, "Chula Vista"),
    ]
    
    sensor_map = {}
    for idx, row in df_sd.iterrows():
        lat, lon = float(row['Lat']), float(row['Lng'])
        fwy = str(row['Fwy'])
        direction = str(row['Direction'])
        
        # Parse freeway name cleanly
        fwy_clean = fwy.replace('-N', '').replace('-S', '').replace('-E', '').replace('-W', '')
        
        # Find nearest landmark
        nearest_landmark = ""
        min_dist = float('inf')
        for lm_lat, lm_lon, radius, lm_name in SD_LANDMARKS:
            d = np.sqrt((lat - lm_lat)**2 + (lon - lm_lon)**2)
            if d < radius and d < min_dist:
                min_dist = d
                nearest_landmark = lm_name
        
        # Determine neighborhood from lat/lon zones
        if lat < 32.60:
            neighborhood = "Otay Mesa / San Ysidro"
        elif lat < 32.65:
            neighborhood = "Chula Vista / National City"
        elif lat < 32.70:
            neighborhood = "National City / Barrio Logan"
        elif lat < 32.73:
            neighborhood = "Downtown SD / East Village"
        elif lat < 32.76:
            neighborhood = "Hillcrest / North Park"
        elif lat < 32.80:
            neighborhood = "Mission Valley / Kearny Mesa"
        elif lat < 32.85:
            neighborhood = "Clairemont / Miramar"
        elif lat < 32.90:
            neighborhood = "Mira Mesa / Scripps Ranch"
        elif lat < 32.95:
            neighborhood = "Poway / Rancho Bernardo"
        elif lat < 33.05:
            neighborhood = "Escondido / San Marcos"
        elif lat < 33.15:
            neighborhood = "Vista / Oceanside"
        else:
            neighborhood = "North County / Camp Pendleton"
        
        info = {
            "sensor_id": int(row['ID']),
            "node_index": idx,
            "lat": lat,
            "lon": lon,
            "freeway": fwy_clean,
            "direction": direction,
            "lanes": int(row['Lanes']),
            "county": str(row['County']),
            "neighborhood": neighborhood,
            "nearest_landmark": nearest_landmark,
            "location_label": f"{fwy_clean} {direction}B near {neighborhood}" + (f" ({nearest_landmark})" if nearest_landmark else "")
        }
        sensor_map[int(row['ID'])] = info
    
    return sensor_map


def save_sensor_maps():
    """Build and save both sensor maps to JSON for backend + LLM consumption."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    
    la_map = build_la_sensor_map()
    sd_map = build_sd_sensor_map()
    
    la_path = os.path.join(data_dir, 'la_sensor_location_map.json')
    sd_path = os.path.join(data_dir, 'sd_sensor_location_map.json')
    
    with open(la_path, 'w') as f:
        json.dump(la_map, f, indent=2)
    with open(sd_path, 'w') as f:
        json.dump(sd_map, f, indent=2)
    
    print(f"[+] METR-LA Sensor Location Map: {len(la_map)} sensors -> {la_path}")
    print(f"[+] SD400 Sensor Location Map: {len(sd_map)} sensors -> {sd_path}")
    
    # Print sample entries
    sample_la = list(la_map.values())[:5]
    print("\n--- METR-LA Sample Mappings ---")
    for s in sample_la:
        print(f"  Sensor #{s['sensor_id']}: {s['location_label']}")
    
    sample_sd = list(sd_map.values())[:5]
    print("\n--- SD400 Sample Mappings ---")
    for s in sample_sd:
        print(f"  Sensor #{s['sensor_id']}: {s['location_label']}")
    
    return la_map, sd_map


if __name__ == "__main__":
    save_sensor_maps()
