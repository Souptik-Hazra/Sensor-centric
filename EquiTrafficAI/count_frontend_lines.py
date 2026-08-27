"""
Count line sizes of all frontend source files in EquiTrafficAI/frontend/src/
"""

import os

frontend_src = r"c:\Users\User\Downloads\metr-la-dissertation-complete\EquiTrafficAI\frontend\src"

print("=================================================================")
print("             FRONTEND FILE LINE-COUNT & SIZE AUDIT               ")
print("=================================================================")

for root, dirs, files in os.walk(frontend_src):
    for file in sorted(files):
        if file.endswith(('.jsx', '.js', '.css')):
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, frontend_src)
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            size_kb = os.path.getsize(full_path) / 1024.0
            print(f"  - {rel_path:<42} : {len(lines):>4} lines ({size_kb:>5.1f} KB)")

print("=================================================================")
