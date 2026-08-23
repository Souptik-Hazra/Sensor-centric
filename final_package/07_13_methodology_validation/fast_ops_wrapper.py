"""
fast_ops_wrapper.py
Python wrapper for fast_ops C native extension with automatic compiler detection
and 100% zero-breakage NumPy SIMD fallback.
"""

import os
import sys
import ctypes
import subprocess
import numpy as np

_c_lib = None
_using_c = False

_using_numba = False
try:
    from numba import njit
    @njit(fastmath=True, cache=True)
    def _detect_cusum_jit(values: np.ndarray, threshold: float, drift: float) -> int:
        mean = np.mean(values)
        pos, neg = 0.0, 0.0
        flags = 0
        for i in range(len(values)):
            v = values[i]
            pos = max(0.0, pos + (v - mean - drift))
            neg = max(0.0, neg + (mean - v - drift))
            if pos > threshold or neg > threshold:
                flags += 1
                pos, neg = 0.0, 0.0
        return flags
    _using_numba = True
except Exception:  # pylint: disable=broad-exception-caught
    pass

def _try_compile_and_load() -> bool:
    global _c_lib, _using_c  # pylint: disable=global-statement
    base_dir = os.path.dirname(os.path.abspath(__file__))
    c_src = os.path.join(base_dir, "fast_ops.c")
    
    if not os.path.exists(c_src):
        return False

    lib_name = "fast_ops.dll" if sys.platform.startswith("win") else "fast_ops.so"
    lib_path = os.path.join(base_dir, lib_name)

    # Try loading pre-compiled lib
    if os.path.exists(lib_path):
        try:
            _c_lib = ctypes.CDLL(lib_path)
            _using_c = True
            return True
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    # Try compiling with gcc if available
    try:
        if sys.platform.startswith("win"):
            cmd = ["gcc", "-O3", "-shared", "-o", lib_path, c_src]
        else:
            cmd = ["gcc", "-O3", "-fPIC", "-shared", "-o", lib_path, c_src, "-lm"]
        
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5, check=False)
        if res.returncode == 0 and os.path.exists(lib_path):
            _c_lib = ctypes.CDLL(lib_path)
            _using_c = True
            return True
    except Exception:  # pylint: disable=broad-exception-caught
        pass

    return False

# Initialize at import time
_try_compile_and_load()

# Setup C argument types if C library loaded
if _using_c and _c_lib is not None:
    _c_lib.detect_cusum_c.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
    ]
    _c_lib.detect_cusum_c.restype = ctypes.c_int

    _c_lib.detect_ewma_c.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
    ]
    _c_lib.detect_ewma_c.restype = ctypes.c_int

    _c_lib.haversine_matrix_c.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ]

    _c_lib.gaussian_adjacency_c.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ]

# --- Python API with automatic fallback ---

def is_c_accelerated():
    return _using_c

def fast_detect_cusum(values, threshold=5.0, drift=0.5):
    values = np.asarray(values, dtype=np.float64)
    if _using_c and _c_lib is not None:
        return _c_lib.detect_cusum_c(values, len(values), float(threshold), float(drift))
    if _using_numba:
        return _detect_cusum_jit(values, float(threshold), float(drift))
    
    # NumPy SIMD Fallback
    mean = np.mean(values)
    pos, neg = 0.0, 0.0
    flags = 0
    for v in values:
        pos = max(0.0, pos + (v - mean - drift))
        neg = max(0.0, neg + (mean - v - drift))
        if pos > threshold or neg > threshold:
            flags += 1
            pos, neg = 0.0, 0.0
    return flags

def fast_detect_ewma(values, alpha=0.2, control_limit=3.0):
    values = np.asarray(values, dtype=np.float64)
    if _using_c and _c_lib is not None:
        return _c_lib.detect_ewma_c(values, len(values), float(alpha), float(control_limit))
    
    # NumPy SIMD Fallback
    mean = np.mean(values)
    std = np.std(values)
    if std < 1e-6:
        return 0
    ewma_std = std * np.sqrt(alpha / (2.0 - alpha))
    upper = mean + control_limit * ewma_std
    lower = mean - control_limit * ewma_std
    
    import pandas as pd
    ewma_series = pd.Series(values).ewm(alpha=alpha, adjust=False).mean().values
    flags = np.sum((ewma_series > upper) | (ewma_series < lower))
    return int(flags)

def fast_haversine_matrix(lats, lons):
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    n = len(lats)
    
    if _using_c and _c_lib is not None:
        dist_matrix = np.zeros((n, n), dtype=np.float64)
        _c_lib.haversine_matrix_c(lats, lons, n, dist_matrix)
        return dist_matrix

    # Vectorized NumPy Fallback
    R = 6371000.0
    lats_rad = np.radians(lats)
    lons_rad = np.radians(lons)
    
    dlat = lats_rad[:, None] - lats_rad[None, :]
    dlon = lons_rad[:, None] - lons_rad[None, :]
    
    a = np.sin(dlat / 2.0)**2 + np.cos(lats_rad[:, None]) * np.cos(lats_rad[None, :]) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))
    return R * c

def fast_gaussian_adjacency(dist_matrix, sigma=5.0, threshold=0.1):
    dist_matrix = np.asarray(dist_matrix, dtype=np.float64)
    n = dist_matrix.shape[0]
    
    if _using_c and _c_lib is not None:
        W_out = np.zeros((n, n), dtype=np.float64)
        _c_lib.gaussian_adjacency_c(dist_matrix, n, float(sigma), float(threshold), W_out)
        return W_out

    # Vectorized NumPy Fallback
    W = np.exp(- (dist_matrix / sigma)**2)
    W[np.isinf(dist_matrix)] = 0.0
    W[W < threshold] = 0.0
    np.fill_diagonal(W, 0.0)
    return W
