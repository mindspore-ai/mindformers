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
"""Test the classes of style.py."""
import os
import random
import subprocess
import pytest
from tests.st.test_multi_cards_cases.utils import TaskType


_LEVEL_0_TASK_TIME = 45
_LEVEL_1_TASK_TIME = 0
_TASK_TYPE = TaskType.TWO_CARDS_TASK


cur_dir = os.path.dirname(os.path.abspath(__file__))
run_script_path = os.path.join(cur_dir, "run_test_style.py")
port_id = int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))


class TestPrepareModule:
    """Test cases for PrepareModuleInput/Output/InputOutput classes."""

    @pytest.mark.level0
    def test_multi_card_prepare_module_cases(self):
        """
        Feature: PrepareModuleInput/Output/InputOutput
        Description: Test PrepareModuleInput/Output/InputOutput with multi-card.
        Expectation: run script exits with code 0.
        """
        assert os.path.exists(
            run_script_path
        ), f"Run script not found: {run_script_path}"
        cmd = [
            "msrun",
            "--worker_num=2",
            "--local_worker_num=2",
            f"--master_port={port_id}",
            "--log_dir=./msrun_log_test_style",
            "--join=True",
            f"{run_script_path}",
            "--tp=2",
        ]
        result = subprocess.run(
            cmd, shell=False, capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, (
            f"PrepareModuleOO script failed with non-zero exit code: "
            f"{result.returncode}.\n"
            f"Stdout:\n{result.stdout}\nStderr:\n{result.stderr}"
        )
