#!/usr/bin/env python3
"""
check_benchmarks.py
Directly loads pickle files from the FairTP directory and prints the computed metrics.
"""
import os
import sys
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("=== DIRECT PICKLE METRIC VERIFICATION ===")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fairtp_dir = os.path.join(base_dir, 'final_package', 'FairTP')
    
    # Model configuration mappings
    model_files = {
        'GWNET': {
            'pred': 'HK_list_pred.pkl',
            'label': 'HK_list_label.pkl'
        },
        'DCRNN': {
            'pred': 'HK_list_pred_dcrnn_forstaticfair.pkl',
            'label': 'HK_list_label_dcrnn_forstaticfair.pkl'
        },
        'DLINEAR': {
            'pred': 'HK_list_pred.pkl',
            'label': 'HK_list_label.pkl'
        },
        'HA': {
            'pred': 'HK_list_pred.pkl',
            'label': 'HK_list_label.pkl'
        }
    }
    
    for model_name, files in model_files.items():
        pred_path = os.path.join(fairtp_dir, files['pred'])
        label_path = os.path.join(fairtp_dir, files['label'])
        
        if not os.path.exists(pred_path) or not os.path.exists(label_path):
            print(f"\n[-] Model {model_name} files not found: {files['pred']} / {files['label']}")
            continue
            
        try:
            with open(pred_path, 'rb') as f_pred, open(label_path, 'rb') as f_label:
                y_pred = pickle.load(f_pred)
                y_true = pickle.load(f_label)
                
            # Convert lists to arrays
            if isinstance(y_pred, list):
                y_pred = [el.detach().cpu().numpy() if hasattr(el, 'detach') else el for el in y_pred]
                y_true = [el.detach().cpu().numpy() if hasattr(el, 'detach') else el for el in y_true]
                y_pred = np.concatenate(y_pred, axis=0)
                y_true = np.concatenate(y_true, axis=0)
                
            print(f"\n[+] MODEL: {model_name}")
            print(f"  Shape: {y_pred.shape}")
            
            if y_true.shape[1] == 1:
                mae = float(np.mean(np.abs(y_true[:, 0, :, 0] - y_pred[:, 0, :, 0])))
                rmse = float(np.sqrt(np.mean((y_true[:, 0, :, 0] - y_pred[:, 0, :, 0])**2)))
                print(f"  Mean Absolute Error (MAE) : {mae:.4f} mph")
                print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
            else:
                mae_15 = float(np.mean(np.abs(y_true[:, 2, :, 0] - y_pred[:, 2, :, 0])))
                mae_60 = float(np.mean(np.abs(y_true[:, 11, :, 0] - y_pred[:, 11, :, 0])))
                print(f"  15-Min MAE : {mae_15:.4f} mph")
                print(f"  60-Min MAE : {mae_60:.4f} mph")
                
        except Exception as e:
            print(f"  [-] Error loading/calculating for {model_name}: {e}")
            
    print("\n=========================================")

if __name__ == '__main__':
    main()
