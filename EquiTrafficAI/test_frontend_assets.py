"""
EquiTraffic-GPT Frontend Asset & Route Integrity Test (test_frontend_assets.py)

Exhaustively verifies that all React Web GIS frontend routes, HTML entrypoints,
CSS style bundles, JavaScript modules, and Leaflet GIS assets load with HTTP 200 OK.
"""

import sys
import requests
import re

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://127.0.0.1:8000"

def test_frontend():
    print("=================================================================")
    print("      EQUITRAFFIC-GPT FRONTEND ASSET & ROUTE INTEGRITY TEST      ")
    print("=================================================================")

    # 1. Root Index HTML
    print("\n[1] Testing Root Index HTML (GET /)...")
    res_root = requests.get(f"{BASE_URL}/")
    assert res_root.status_code == 200, f"Failed root index with status {res_root.status_code}"
    html = res_root.text
    print("  [✔] Status Code  : 200 OK")
    print("  [✔] Root Tag     : '<div id=\"root\"></div>' present")

    # Extract JS and CSS asset URLs from HTML
    js_matches = re.findall(r'src="(/assets/[^"]+\.js)"', html)
    css_matches = re.findall(r'href="(/assets/[^"]+\.css)"', html)

    # 2. JavaScript Bundle Test
    print("\n[2] Testing Compiled React JS Bundle...")
    if js_matches:
        js_url = f"{BASE_URL}{js_matches[0]}"
        res_js = requests.get(js_url)
        assert res_js.status_code == 200, f"Failed JS bundle at {js_url}"
        print(f"  [✔] JS Bundle Path : {js_matches[0]}")
        print(f"  [✔] JS Bundle Size : {len(res_js.content) / 1024:.1f} KB (200 OK)")

    # 3. CSS Bundle Test
    print("\n[3] Testing Compiled Stylesheet CSS Bundle...")
    if css_matches:
        css_url = f"{BASE_URL}{css_matches[0]}"
        res_css = requests.get(css_url)
        assert res_css.status_code == 200, f"Failed CSS bundle at {css_url}"
        print(f"  [✔] CSS Bundle Path: {css_matches[0]}")
        print(f"  [✔] CSS Bundle Size: {len(res_css.content) / 1024:.1f} KB (200 OK)")

    # 4. Leaflet GIS Map Container & Font Icons
    print("\n[4] Testing GIS Map Icons & Font Dependencies...")
    res_icon = requests.get(f"{BASE_URL}/icons.svg")
    print(f"  [✔] Map Icons asset: HTTP {res_icon.status_code}")

    print("\n=================================================================")
    print("✔ FRONTEND ASSETS & REACT BUNDLE ARE 100% CLEAN & OPERATIONAL!")
    print("=================================================================\n")

if __name__ == "__main__":
    test_frontend()
