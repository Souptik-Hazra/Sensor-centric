import json
import re

html_path = r"c:\Users\User\Downloads\metr-la-dissertation-complete\final_package\07_13_methodology_validation\digital_twin_gis_map.html"
out_path = r"c:\Users\User\Downloads\metr-la-dissertation-complete\traffic-system-web\src\core\simulationData.json"

with open(html_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Extract modelBenchmarks
m_bench = re.search(r'const modelBenchmarks = ({.*?});', content)
if m_bench:
    benchmarks = json.loads(m_bench.group(1))
else:
    benchmarks = {}

# Extract empiricalProfiles
m_prof = re.search(r'const empiricalProfiles = ({.*?});\s*const hasEmpiricalData', content, re.DOTALL)
if m_prof:
    profiles = json.loads(m_prof.group(1))
else:
    profiles = {}

data = {
    "modelBenchmarks": benchmarks,
    "empiricalProfiles": profiles
}

import os
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(data, f)

print(f"Successfully extracted data to {out_path}")
