"""
run_all_validation_pipeline.py
Master Sequential Pipeline Execution Runner for METR-LA Causal Digital Twin Validation Suite.
Executes all Python (.py) and R (.R) scripts sequentially, verifying 100% clean execution.
"""

import os
import subprocess
import time

def run_script(cmd, cwd):
    start = time.time()
    print("\n====================================================================")
    print(f"  RUNNING: {' '.join(cmd)}")
    print("====================================================================")
    try:
        res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, errors='ignore', check=False)
        duration = time.time() - start
        if res.returncode == 0:
            print(f"[OK] Completed successfully in {duration:.2f}s")
            return True, duration, res.stdout[-300:]
        else:
            print(f"[ERROR] Code {res.returncode} in {duration:.2f}s")
            print(res.stderr[:500])
            return False, duration, res.stderr[:500]
    except (OSError, ValueError) as e:
        duration = time.time() - start
        print(f"[EXCEPTION] {e}")
        return False, duration, str(e)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("=== STARTING MASTER SEQUENTIAL VALIDATION PIPELINE EXECUTION ===")
    
    scripts = [
        ['python', '06b_export_metrics.py'],
        ['python', '07_setup_and_fairtp_verified.py'],
        ['Rscript', '08_dag_identifiability.R'],
        ['Rscript', '09_ctf_estimation_faircause.R'],
        ['Rscript', '10_power_simulation.R'],
        ['python', '11a_reliability_weights.py'],
        ['Rscript', '11b_reliability_sensitivity.R'],
        ['python', '11c_compound_multifault_stress_test.py'],
        ['python', '12_disparity_reconciliation.py'],
        ['Rscript', '13_temporal_alignment.R'],
        ['python', '14_digital_twin_causal_simulator.py'],
        ['python', '15_digital_twin_gis_interactive.py'],
        ['python', '16_export_multi_horizon_metrics.py']
    ]
    
    results = []
    total_start = time.time()
    
    for cmd in scripts:
        script_name = cmd[1]
        success, duration, snippet = run_script(cmd, cwd=base_dir)
        results.append({
            'script': script_name,
            'status': 'PASSED' if success else 'FAILED',
            'duration_sec': round(duration, 2),
            'snippet': snippet.strip().replace('\n', ' ')
        })
        
    total_duration = time.time() - total_start
    print("\n====================================================================")
    print(f"  MASTER PIPELINE EXECUTION SUMMARY (Total Time: {total_duration:.2f}s)")
    print("====================================================================")
    
    passed_count = sum(1 for r in results if r['status'] == 'PASSED')
    failed_count = len(results) - passed_count
    
    for r in results:
        status_icon = "[OK] PASSED" if r['status'] == 'PASSED' else "[FAIL] FAILED"
        print(f"  [{status_icon}] {r['script']:<40} ({r['duration_sec']}s)")
        
    print(f"\nPipeline Summary: {passed_count}/{len(results)} scripts passed successfully ({failed_count} failures).")
    if failed_count == 0:
        print("[SUCCESS] 100% of methodology validation scripts executed cleanly with 0 errors!")

if __name__ == '__main__':
    main()
