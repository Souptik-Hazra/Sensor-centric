"""
run_all_tests_master.py
Master 1-Command Verification & Test Suite Runner for METR-LA Causal Digital Twin Framework.
Runs:
1. Full System 25-Testcase Battery (test_suite_25_cases.py)
2. Dedicated Digital Twin & GIS Map 25-Testcase Battery (test_suite_digital_twin_gis_25_cases.py)
3. 13-Script Master Methodology Validation Pipeline (run_all_validation_pipeline.py)
Total: 63 Verified Testcases & Pipeline Executions
"""

import sys
import os
import time
import subprocess
import warnings

warnings.filterwarnings('ignore')

def print_header(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)

def main():
    start_time = time.time()
    expected_total = 63
    print_header(f"MASTER 1-COMMAND SUITE: RUNNING ALL {expected_total} TESTCASES & PIPELINE SCRIPTS")
    
    root_dir = os.path.dirname(os.path.abspath(__file__))
    scratch_dir = os.path.join(root_dir, "..", "scratch")
    val_dir = os.path.join(root_dir, "final_package", "07_13_methodology_validation")
    
    if not os.path.exists(scratch_dir):
        scratch_dir = os.path.join(root_dir, "scratch")
    if not os.path.exists(val_dir):
        val_dir = root_dir

    suite_results = []
    missing_suites = []

    # 1. Battery 1: Full System 25 Testcases
    print_header("1. RUNNING FULL SYSTEM 25-TESTCASE BATTERY")
    battery1_script = os.path.join(scratch_dir, "test_suite_25_cases.py")
    if os.path.exists(battery1_script):
        res1 = subprocess.run([sys.executable, battery1_script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(res1.stdout)
        status1 = "PASSED (25/25)" if res1.returncode == 0 else "FAILED"
        suite_results.append(("Full System 25-Testcase Battery", status1))
    else:
        print(f"[!] Warning: {battery1_script} not found.")
        missing_suites.append("Full System 25-Testcase Battery")
        suite_results.append(("Full System 25-Testcase Battery", "MISSING"))

    # 2. Battery 2: Dedicated Digital Twin & GIS Map 25 Testcases
    print_header("2. RUNNING DIGITAL TWIN & GIS MAP 25-TESTCASE BATTERY")
    battery2_script = os.path.join(scratch_dir, "test_suite_digital_twin_gis_25_cases.py")
    if os.path.exists(battery2_script):
        res2 = subprocess.run([sys.executable, battery2_script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(res2.stdout)
        status2 = "PASSED (25/25)" if res2.returncode == 0 else "FAILED"
        suite_results.append(("Digital Twin & GIS Map 25-Testcase Battery", status2))
    else:
        print(f"[!] Warning: {battery2_script} not found.")
        missing_suites.append("Digital Twin & GIS Map 25-Testcase Battery")
        suite_results.append(("Digital Twin & GIS Map 25-Testcase Battery", "MISSING"))

    # 3. Master Sequential 13-Script Validation Pipeline
    print_header("3. RUNNING MASTER 13-SCRIPT METHODOLOGY VALIDATION PIPELINE")
    pipeline_script = os.path.join(val_dir, "run_all_validation_pipeline.py")
    if os.path.exists(pipeline_script):
        res3 = subprocess.run([sys.executable, pipeline_script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(res3.stdout)
        status3 = "PASSED (13/13)" if res3.returncode == 0 else "FAILED"
        suite_results.append(("13-Script Sequential Validation Pipeline", status3))
    else:
        print(f"[!] Warning: {pipeline_script} not found.")
        missing_suites.append("13-Script Sequential Validation Pipeline")
        suite_results.append(("13-Script Sequential Validation Pipeline", "MISSING"))

    total_duration = time.time() - start_time
    passed_count = sum(1 for _, status in suite_results if status.startswith("PASSED"))
    failed_count = sum(1 for _, status in suite_results if status == "FAILED")
    missing_count = sum(1 for _, status in suite_results if status == "MISSING")
    
    print_header(f"FINAL MASTER SUMMARY (Total Execution Time: {total_duration:.2f}s)")
    for name, status in suite_results:
        prefix = "[OK]" if status.startswith("PASSED") else "[WARN]"
        print(f"  [{prefix} {status}] {name}")

    total_suites = len(suite_results)
    print(
        f"\nSummary: {passed_count} passed, {failed_count} failed, {missing_count} missing "
        f"out of {total_suites} suites. Expected overall coverage was {expected_total} testcases/validations."
    )
    if failed_count == 0 and missing_count == 0:
        print("[SUCCESS] All discovered suites executed cleanly.")
    else:
        print("[WARN] The master suite did not complete full advertised coverage.")

if __name__ == '__main__':
    main()
