#!/usr/bin/env python3
"""Property-based testing suite using Hypothesis."""
import unittest
import numpy as np
from hypothesis import given, strategies as st
from fast_ops_wrapper import fast_detect_cusum, fast_detect_ewma

class TestCausalTwinProperties(unittest.TestCase):
    @given(st.lists(st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False), min_size=5))
    def test_cusum_always_valid(self, values):
        val_arr = np.array(values, dtype=np.float64)
        flags = fast_detect_cusum(val_arr)
        self.assertGreaterEqual(flags, 0)
        self.assertLessEqual(flags, len(values))

    @given(st.lists(st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False), min_size=5))
    def test_ewma_always_valid(self, values):
        val_arr = np.array(values, dtype=np.float64)
        flags = fast_detect_ewma(val_arr)
        self.assertGreaterEqual(flags, 0)
        self.assertLessEqual(flags, len(values))

if __name__ == '__main__':
    unittest.main()
