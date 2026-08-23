import os
import glob
import pandas as pd
import numpy as np
import json
import pickle

base_dir = os.path.dirname(os.path.abspath(__file__))

print("==================================================================")
print("              FULL DATASET & FILE DISCOVERY SCAN                 ")
print("==================================================================")

all_files = []
for root, dirs, files in os.walk(base_dir):
    if 'node_modules' in root or '.git' in root or '.system_generated' in root or 'dist' in root:
        continue
    for f in files:
        all_files.append(os.path.join(root, f))

print(f"Total files found in workspace (excluding node_modules/git): {len(all_files)}\n")

data_extensions = ('.csv', '.npz', '.pkl', '.json', '.h5', '.parquet')
data_files = [f for f in all_files if f.endswith(data_extensions)]

print("--- DATA FILES DISCOVERED ---")
for df_path in sorted(data_files):
    rel_path = os.path.relpath(df_path, base_dir)
    size_mb = os.path.getsize(df_path) / (1024 * 1024)
    print(f"  • {rel_path} ({size_mb:.2f} MB)")

print("\n==================================================================")
print("              DETAILED DATASET FILE INSPECTION                   ")
print("==================================================================")

for df_path in sorted(data_files):
    rel_path = os.path.relpath(df_path, base_dir)
    print(f"\n>>> File: {rel_path}")
    size_mb = os.path.getsize(df_path) / (1024 * 1024)
    print(f"    Size: {size_mb:.2f} MB")
    
    try:
        if df_path.endswith('.csv'):
            df = pd.read_csv(df_path, nrows=5)
            with open(df_path, 'r', encoding='utf-8', errors='ignore') as f:
                row_count = sum(1 for _ in f) - 1
            print(f"    Type: CSV | Rows: {row_count:,} | Columns: {len(df.columns)}")
            print(f"    Columns: {list(df.columns)[:10]}{'...' if len(df.columns)>10 else ''}")
            print("    Sample Row 1:", df.iloc[0].to_dict() if len(df) > 0 else "Empty")

        elif df_path.endswith('.npz'):
            npz = np.load(df_path)
            print(f"    Type: NPZ Compressed Array | Keys: {list(npz.keys())}")
            for k in npz.keys():
                arr = npz[k]
                print(f"      - {k}: shape={arr.shape}, dtype={arr.dtype}, min={np.min(arr) if arr.size>0 else 'N/A'}, max={np.max(arr) if arr.size>0 else 'N/A'}")

        elif df_path.endswith('.json'):
            if size_mb > 10:
                print(f"    Type: JSON (Large, >10MB) - Skipping full parse for speed.")
            else:
                with open(df_path, 'r', encoding='utf-8', errors='ignore') as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, dict):
                            print(f"    Type: JSON Object | Top-level keys: {list(data.keys())[:10]}")
                        elif isinstance(data, list):
                            print(f"    Type: JSON Array | Length: {len(data):,}")
                    except Exception as je:
                        print(f"    Type: JSON (Custom/Parse Error: {je})")

        elif df_path.endswith('.pkl'):
            with open(df_path, 'rb') as f:
                try:
                    obj = pickle.load(f)
                    if isinstance(obj, np.ndarray):
                        print(f"    Type: PKL (NumPy Array) | Shape: {obj.shape}, dtype: {obj.dtype}")
                    elif isinstance(obj, list):
                        print(f"    Type: PKL (List) | Length: {len(obj)}")
                        if len(obj) > 0 and hasattr(obj[0], 'shape'):
                            print(f"      Item 0 shape: {obj[0].shape}")
                    elif isinstance(obj, dict):
                        print(f"    Type: PKL (Dict) | Keys: {list(obj.keys())[:10]}")
                    else:
                        print(f"    Type: PKL ({type(obj)})")
                except Exception as pe:
                    print(f"    Type: PKL (Pickle Load Error: {pe})")

    except Exception as e:
        print(f"    Error reading file: {e}")

print("\n==================================================================")
print("                       SCAN COMPLETE                              ")
print("==================================================================")
