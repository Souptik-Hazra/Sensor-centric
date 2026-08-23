#!/usr/bin/env python3
"""
inspect_pkl_keys.py
Inspects the full structure, dictionary keys, list elements, or data types
contained inside the pre-existing pickle files.
"""
import os
import sys
import pickle
import warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    fairtp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final_package', 'FairTP')
    
    files = [
        'HK_list_pred.pkl',
        'HK_list_label.pkl',
        'HK_list_pred_dcrnn_forstaticfair.pkl',
        'HK_list_label_dcrnn_forstaticfair.pkl',
        'SD_list_pred.pkl',
        'SD_list_label.pkl'
    ]
    
    for filename in files:
        filepath = os.path.join(fairtp_dir, filename)
        if not os.path.exists(filepath):
            print(f"[-] File not found: {filename}")
            continue
            
        print(f"\n=========================================")
        print(f"[+] File: {filename}")
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            print(f"  • Data Type : {type(data)}")
            
            if isinstance(data, list):
                print(f"  • List Length: {len(data)}")
                if len(data) > 0:
                    first_el = data[0]
                    print(f"  • First element type: {type(first_el)}")
                    if hasattr(first_el, 'shape'):
                        print(f"  • First element shape: {first_el.shape}")
                    elif isinstance(first_el, dict):
                        print(f"  • First element keys: {list(first_el.keys())}")
                    else:
                        print(f"  • First element value: {first_el}")
                        
            elif isinstance(data, dict):
                print(f"  • Dictionary Keys: {list(data.keys())}")
                for k, v in data.items():
                    print(f"    - Key: {k} | Type: {type(v)}")
                    if hasattr(v, 'shape'):
                        print(f"      Shape: {v.shape}")
            
            elif hasattr(data, 'shape'):
                print(f"  • Array Shape: {data.shape}")
                
        except Exception as e:
            print(f"  [-] Error loading {filename}: {e}")
            
    print("\n=========================================")

if __name__ == '__main__':
    main()
