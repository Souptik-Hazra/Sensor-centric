#!/usr/bin/env python3
"""
17_causal_refutation_audit.py
Advanced Causal Inference Verification and SCM Refutation Audit using DoWhy.
Validates the structural paths: Density -> Reliability -> Disparity Error.
"""
import os
import pandas as pd
import numpy as np
import dowhy
from dowhy import CausalModel
import sys
import warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("=== STARTING ADVANCED SCM CAUSAL REFUTATION AUDIT (DOWHY) ===")
    
    # 1. Load telemetry metrics dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_path = os.path.join(base_dir, 'metr_la_metrics.csv')
    if not os.path.exists(metrics_path):
        metrics_path = 'metr_la_metrics.csv'
    if not os.path.exists(metrics_path):
        metrics_path = os.path.join(base_dir, '..', '..', 'metr_la_metrics.csv')
        
    df = pd.read_csv(metrics_path)
    
    # Derive regional sub-district mappings or spectral clusters if absent
    if 'district' not in df.columns:
        if 'spectral_cluster' in df.columns:
            df['district'] = df['spectral_cluster']
        else:
            df['district'] = pd.qcut(df['density'], q=4, labels=[0, 1, 2, 3]).astype(int)
            
    if 'reliability' not in df.columns:
        df['reliability'] = np.clip(1.0 - (0.60 * df['zero_rate'] + 0.20 * df['cusum_flags'] / df['observations'] + 0.20 * df['ewma_flags'] / df['observations']), 0.0, 1.0)

    # 2. Formulate the SCM and DAG
    # Treatment: reliability
    # Outcome: persistence_error (our disparity proxy)
    # Common Cause (Confounder): density
    
    import networkx as nx
    g = nx.DiGraph()
    g.add_nodes_from(['density', 'reliability', 'persistence_error'])
    g.add_edges_from([
        ('density', 'reliability'),
        ('reliability', 'persistence_error'),
        ('density', 'persistence_error')
    ])
    
    # 3. Initialize DoWhy model
    model = CausalModel(
        data=df,
        treatment='reliability',
        outcome='persistence_error',
        graph=g
    )
    
    # 4. Identify causal effect
    identified_estimand = model.identify_effect()
    print("\n[+] Causal Estimand Identified:")
    print(identified_estimand)
    
    # 5. Estimate causal effect using linear regression
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression"
    )
    print(f"\n[+] Estimated Causal Effect (Reliability -> Error): {estimate.value:.4f}")
    
    # 6. Run SCM Refutation Tests (Placebo Treatment Refuter)
    print("\n[+] Running Placebo Treatment Refutation Test (expecting effect to drop to ~0)...")
    refute_placebo = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="placebo_treatment_refuter",
        placebo_type="permute"
    )
    print(refute_placebo)
    
    # 7. Run SCM Refutation Tests (Random Common Cause Refuter)
    print("\n[+] Running Random Common Cause Refutation Test (expecting effect to remain stable)...")
    refute_random = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="random_common_cause"
    )
    print(refute_random)
    
    # Assert statistical validity: placebo effect should be very close to zero
    placebo_val = refute_placebo.new_effect
    assert abs(placebo_val) < 0.1, f"SCM validation failed: Placebo effect is non-zero ({placebo_val:.4f})"
    print("\n[SUCCESS] SCM Causal Graph Refutation and DAG Audit passed successfully!")

if __name__ == '__main__':
    main()
