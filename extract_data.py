import re
import json

def extract():
    with open("digital_twin_gis_map.html", "r", encoding="utf-8") as f:
        content = f.read()

    # Extract sensors array
    sensors_match = re.search(r'const sensors = (\[.*?\]);', content, re.DOTALL)
    if sensors_match:
        sensors_json = sensors_match.group(1)
        # Some cleanup if keys are not quoted, but typically json.loads works if it's strict JSON
        # In HTML it might be JS objects, but let's try direct json.loads, or we can just write it as a JS module!
        # Writing as JS module is safer because it's already valid JS.
        with open("traffic-system-web/src/data/sensors.js", "w", encoding="utf-8") as f:
            f.write(f"export const sensors = {sensors_json};\n")
        print("Extracted sensors.js")

    # Extract tsData
    tsdata_match = re.search(r'const tsData = (\{.*?\});\n\s+const hasEmpiricalData', content, re.DOTALL)
    if tsdata_match:
        tsdata_json = tsdata_match.group(1)
        with open("traffic-system-web/src/data/traffic_data.js", "w", encoding="utf-8") as f:
            f.write(f"export const trafficData = {tsdata_json};\n")
        print("Extracted traffic_data.js")

if __name__ == "__main__":
    import os
    os.makedirs("traffic-system-web/src/data", exist_ok=True)
    extract()
