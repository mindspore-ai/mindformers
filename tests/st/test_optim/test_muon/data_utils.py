# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Baseline data for Muon optimizer tests.
"""
import numpy as np

# Default tolerance for loss comparison
DEFAULT_RTOL = 1e-4
DEFAULT_ATOL = 1e-4

# Baseline losses for single card test cases
# learning_rate=0.02, weight_decay=0.1, momentum=0.95, nesterov=True
BASELINE_LOSSES_NESTEROV_TRUE = np.array([
    0.3881023, 7.8119774, 15.033653, 22.04059, 28.842205,
    35.438236, 41.858444, 48.087048, 54.100113, 59.940193,
    65.58313, 71.06186, 76.320015, 81.41474, 86.33582,
    91.08755, 95.67038, 100.0687, 104.30162, 108.35542
], dtype=np.float32)

# learning_rate=0.02, weight_decay=0.1, momentum=0.95, nesterov=False
BASELINE_LOSSES_NESTEROV_FALSE = np.array([
    0.3881023, 7.812128, 15.0343895, 22.049484, 28.864697,
    35.488247, 41.92078, 48.160828, 54.19826, 60.031258,
    65.703804, 71.1555, 76.45647, 81.57473, 86.521225,
    91.28078, 95.866196, 100.28407, 104.5376, 108.60419
], dtype=np.float32)

# learning_rate=0.01, weight_decay=0.05, momentum=0.9, nesterov=True
BASELINE_LOSSES_DIFF_LR = np.array([
    0.3881023, 7.8963957, 15.319717, 22.65504, 29.905792,
    37.07411, 44.15998, 51.16218, 58.080475, 64.89875,
    71.64437, 78.30286, 84.88207, 91.37878, 97.77704,
    104.11557, 110.35792, 116.513, 122.59845, 128.61096
], dtype=np.float32)


def compare_losses(actual_losses, expected_losses, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL):
    """
    Compare actual losses with expected baseline losses.

    Args:
        actual_losses (np.ndarray): Actual losses from the test run
        expected_losses (np.ndarray): Expected baseline losses
        rtol (float): Relative tolerance for comparison
        atol (float): Absolute tolerance for comparison

    Returns:
        bool: True if losses match within tolerance, False otherwise
    """
    return np.allclose(actual_losses, expected_losses, rtol=rtol, atol=atol)
