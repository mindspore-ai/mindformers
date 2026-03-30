# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Test ExpertParallel implementation"""
import subprocess
import os
from pathlib import Path
import numpy as np
import pytest


SCRIPT_DIR = Path(__file__).parent.resolve()


def test_expert_parallel():
    """Test ExpertParallel: run via msrun and compare outputs"""
    
    output_file = SCRIPT_DIR / "ep_output"
    
    # Run via msrun with 2 cards
    cmd = [
        "msrun",
        "--worker_num=2",
        "--local_worker_num=2",
        "--master_addr=127.0.0.1",
        "--master_port=8118",
        "--join=True",
        "python",
        str(SCRIPT_DIR / "run_expert_parallel.py"),
        "--hidden_size", "32",
        "--num_experts", "4",
        "--top_k", "2",
        "--batch_size", "2",
        "--seq_length", "4",
        "--output", str(output_file),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"msrun failed: {result.stderr}"
    
    # Load results
    output_path = str(output_file) + '.npz'
    assert Path(output_path).exists(), f"Output file not found: {output_path}"
    
    data = np.load(output_path)
    single_output = data['single']
    distributed_output = data['distributed']
    
    # Compare
    assert single_output.shape == distributed_output.shape
    np.testing.assert_allclose(single_output, distributed_output, rtol=1e-4, atol=1e-6)
