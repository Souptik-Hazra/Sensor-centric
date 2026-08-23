import os
import sys
import subprocess

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

fairtp_path = "FairTP"
if not os.path.exists(fairtp_path) and os.path.exists(os.path.join("..", "FairTP")):
    fairtp_path = os.path.join("..", "FairTP")

if not os.path.exists(fairtp_path):
    print("Cloning FairTP repository from GitHub... This may take a moment.")
    subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/jiangnanx129/FairTP.git', 'FairTP'], check=False)
    print("Clone complete!")
else:
    print(f"FairTP directory found at '{fairtp_path}' - skipping git clone.")

print("\nTop-level contents:")
for f in sorted(os.listdir(fairtp_path)):
    print(" ", f)
exp_dir = os.path.join(fairtp_path, "experiments")
if os.path.exists(exp_dir):
    print("\nexperiments/ contents (these are FairTP's own per-architecture folders):")
    for f in sorted(os.listdir(exp_dir)):
        print(" ", f)


# ==========================================

# --- View FairTP's actual RSF formula (static_cal) directly from their repo ---
engine_file = os.path.join(fairtp_path, "fsample_engine.py")
if os.path.exists(engine_file):
    with open(engine_file, encoding="utf-8") as fh:  # 'fh' avoids type conflict with loop var 'f' (str)
        src: str = fh.read()
    start = src.find("def static_cal")
    end = src.find("def dynamic_cal")
    print(src[start:end])
else:
    print(f"File {engine_file} not found.")


# ==========================================

# --- Clean, reusable RSF function, adapted from the above, tested on synthetic data ---
import numpy as np

def rsf_static_fairness(pred, label, eps=1e-6):
    """
    Region-based Static Fairness (RSF), adapted from FairTP's static_cal()
    (jiangnanx129/FairTP, fsample_engine.py).
    pred, label: arrays shaped (batch, time, n_regions)
    Returns: mean absolute pairwise MAPE-difference across all region pairs.
    Lower = fairer (regions perform more similarly to each other).
    """
    b, t, n = pred.shape
    pred = pred.reshape(b * t, n)
    label = label.reshape(b * t, n)
    mape = np.abs(pred - label) / (np.abs(label) + eps)
    diffs = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            diffs.append(np.abs(np.sum(mape[:, i] - mape[:, j])))
    return float(np.mean(diffs))

# Sanity check: near-identical regions -> low RSF; one degraded region -> high RSF
rng = np.random.default_rng(0)
pred_equal = rng.normal(50, 5, size=(32, 12, 5))
label_equal = pred_equal + rng.normal(0, 0.5, size=pred_equal.shape)
print("RSF, near-identical regions:", rsf_static_fairness(pred_equal, label_equal))

pred_uneven, label_uneven = pred_equal.copy(), label_equal.copy()
label_uneven[:, :, 0] += rng.normal(0, 15, size=(32, 12))
print("RSF, one region degraded:  ", rsf_static_fairness(pred_uneven, label_uneven))
assert rsf_static_fairness(pred_uneven, label_uneven) > rsf_static_fairness(pred_equal, label_equal), \
    "Sanity check failed: degraded region should increase RSF"
print("PASS: RSF increases when a region is degraded, as expected.")


# ==========================================

# --- Clean, reusable SDF function, tested on synthetic data ---
def sdf_dynamic_fairness(state_history_by_sensor):
    """
    Sensor-based Dynamic Fairness (SDF), adapted from FairTP's dynamic_cal().
    state_history_by_sensor: dict {sensor_id: [signed magnitudes over time]},
    positive = 'benefit' state, negative = 'sacrifice' state.
    Returns: mean absolute difference in cumulative positive/negative balance
    across all sensor pairs. Lower = fairer.
    """
    keys = list(state_history_by_sensor.keys())
    diffs = []
    for i in range(len(keys) - 1):
        for j in range(i + 1, len(keys)):
            si, sj = state_history_by_sensor[keys[i]], state_history_by_sensor[keys[j]]
            pos_i, pos_j = sum(x for x in si if x > 0), sum(x for x in sj if x > 0)
            neg_i, neg_j = sum(x for x in si if x < 0), sum(x for x in sj if x < 0)
            diffs.append(abs(pos_i - pos_j) + abs(neg_i - neg_j))
    return float(np.mean(diffs))

rng = np.random.default_rng(1)
balanced = {f"s{k}": list(rng.normal(0, 1, 50)) for k in range(6)}
print("SDF, balanced sensors:            ", sdf_dynamic_fairness(balanced))

unbalanced = {f"s{k}": list(rng.normal(0, 1, 50)) for k in range(6)}
unbalanced["s0"] = list(-np.abs(rng.normal(3, 1, 50)))
print("SDF, one sensor always sacrificed:", sdf_dynamic_fairness(unbalanced))
assert sdf_dynamic_fairness(unbalanced) > sdf_dynamic_fairness(balanced), \
    "Sanity check failed: a permanently-sacrificed sensor should increase SDF"
print("PASS: SDF increases when one sensor is chronically disadvantaged, as expected.")
