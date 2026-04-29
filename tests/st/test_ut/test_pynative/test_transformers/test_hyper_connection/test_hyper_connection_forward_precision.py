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
"""Test HyperConnectionModule forward precision."""

from pathlib import Path
import subprocess

import pytest

from mindformers.tools.logger import logger


def build_command_list(run_script_path, model_args):
    """Build the command with the specified parameters."""
    cmd_list = ["python", str(run_script_path)]
    for key, value in model_args.items():
        cmd_list.append(f"--{key}={value}")
    logger.info(f"Equivalent shell command for debugging (approximate): {' '.join(cmd_list)}")
    return cmd_list


class TestHyperConnection:
    """Test class for HyperConnectionModule."""

    def setup_method(self):
        """Prepare test paths."""
        self.current_dir = Path(__file__).parent.resolve()
        self.run_script_path = self.current_dir / "run_hyper_connection_forward_precision.py"

    def run_test(self, model_args):
        """Run HyperConnectionModule test script."""
        result = subprocess.run(
            build_command_list(self.run_script_path, model_args),
            shell=False,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            "HyperConnectionModule forward precision script failed with non-zero exit code: "
            f"{result.returncode}.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"
        )


class TestHyperConnectionSingleCard(TestHyperConnection):
    """Test HyperConnectionModule on single card."""

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_forward_precision(self):
        """Test deterministic forward precision."""
        self.run_test(
            {
                "seq_len": 4,
                "batch_size": 2,
                "rate": 4,
                "hidden_size": 8,
                "sinkhorn_iters": 20,
                "compute_dtype": "float32",
            }
        )
