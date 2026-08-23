#!/bin/bash
set -e

echo "===================================================================="
echo "RUNNING FULL CAUSAL FAIRNESS ATTRIBUTION SUITE (SCRIPTS 06b - 13)"
echo "===================================================================="

echo ""
echo "[0/7] Ensuring metr_la_metrics.csv is exported..."
if [ ! -f metr_la_metrics.csv ]; then
    python 06b_export_metrics.py
else
    echo "metr_la_metrics.csv found - skipping re-export."
fi

echo ""
echo "[1/7] Running 07_setup_and_fairtp_verified.py..."
python 07_setup_and_fairtp_verified.py

echo ""
echo "[2/7] Running 08_dag_identifiability.R..."
Rscript 08_dag_identifiability.R

echo ""
echo "[3/7] Running 09_ctf_estimation_faircause.R..."
Rscript 09_ctf_estimation_faircause.R

echo ""
echo "[4/7] Running 10_power_simulation.R..."
Rscript 10_power_simulation.R

echo ""
echo "[5/7] Running 11a_reliability_weights.py and 11b_reliability_sensitivity.R..."
python 11a_reliability_weights.py
Rscript 11b_reliability_sensitivity.R

echo ""
echo "[6/7] Running 12_disparity_reconciliation.py..."
python 12_disparity_reconciliation.py

echo ""
echo "[7/7] Running 13_temporal_alignment.R..."
Rscript 13_temporal_alignment.R

echo ""
echo "===================================================================="
echo "ALL CAUSAL ANALYSIS SCRIPTS COMPLETED SUCCESSFULLY!"
echo "===================================================================="
