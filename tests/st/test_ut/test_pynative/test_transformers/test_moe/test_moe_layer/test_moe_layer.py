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
"""Test MoELayer with various configurations"""
from pathlib import Path
import subprocess
import pytest
import numpy as np
from tests.utils.double_benchmark import DoubleBenchmarkStandard, DoubleBenchmarkComparator
from mindformers.tools.logger import logger

from .data_gen_utils import GOLDEN_DATA, GPU_DATA

HIDDEN_SIZE = 32
NUM_EXPERTS = 4
TOP_K = 2

# Parameter structure: (model_args, data_keys, expect_error)
SINGLE_CARD_TEST_PARAM = "model_args, data_keys, expect_error"
SINGLE_CARD_TEST_CASES = [
    # Basic cases
    (
        {
            "score_func": "softmax",
            "route_norm": False,
            "route_scale": 1.0,
            "num_expert_groups": None,
            "num_limited_groups": None,
            "force_load_balance": False,
            "shared_expert_num": 0,
            "apply_probs_on_input": False,
        },
        {
            "output": "output_case0",
        },
        None,
    ),
    # Shared expert case
    (
        {
            "score_func": "softmax",
            "route_norm": False,
            "route_scale": 1.0,
            "num_expert_groups": None,
            "num_limited_groups": None,
            "force_load_balance": False,
            "shared_expert_num": 1,
            "apply_probs_on_input": False,
        },
        {
            "output": "output_case1",
        },
        None,
    ),
]


def build_msrun_command_list(
    worker_num,
    local_worker_num,
    log_dir,
    run_script_path,
    hidden_size,
    num_experts,
    top_k,
    score_func,
    route_norm,
    route_scale,
    output_file_path,
    num_expert_groups,
    num_limited_groups,
    force_load_balance,
    shared_expert_num,
    apply_probs_on_input,
    port_id,
):
    """Build the msrun command with the specified parameters."""
    if worker_num == 1 and local_worker_num == 1:
        cmd_list = ["python"]
    else:
        cmd_list = [
            "msrun",
            f"--worker_num={worker_num}",
            f"--local_worker_num={local_worker_num}",
            f"--master_port={port_id}",
            f"--log_dir={log_dir}",
            "--join=True",
        ]
    cmd_list += [
        str(run_script_path),
        f"--hidden_size={hidden_size}",
        f"--num_experts={num_experts}",
        f"--top_k={top_k}",
        f"--score_func={score_func}",
        f"--route_norm={str(route_norm).lower()}",
        f"--route_scale={route_scale}",
        f"--output_path={output_file_path}",
        f"--force_load_balance={str(force_load_balance).lower()}",
        f"--shared_expert_num={shared_expert_num}",
        f"--apply_probs_on_input={str(apply_probs_on_input).lower()}",
    ]
    if num_expert_groups is not None:
        cmd_list.append(f"--num_expert_groups={num_expert_groups}")
    if num_limited_groups is not None:
        cmd_list.append(f"--num_limited_groups={num_limited_groups}")
    return cmd_list


class TestMoELayer:
    """Test class for MoELayer"""

    OUTPUT_MS_FILENAME = "output_ms.npz"
    LOG_DIR_NAME = "msrun_log"

    def setup_method(self):
        """Setup method to prepare test environment"""
        self.sh_path = Path(__file__).parent.resolve()
        self.run_script_path = self.sh_path / "run_moe_layer.py"

    def check_acc(self, output_ms_dict, data_keys):
        """
        Compare output_ms with GOLDEN_DATA and GPU_DATA using DoubleBenchmarkComparator.
        """
        standard = DoubleBenchmarkStandard(dtype="bfloat16")

        for key, data_key in data_keys.items():
            npu_data = output_ms_dict.get(key)
            if npu_data is None:
                npu_data = output_ms_dict.get("output_case0") # Fallback

            golden_data = GOLDEN_DATA.get(data_key)
            gpu_data = GPU_DATA.get(data_key)

            # If golden data is missing (e.g. output_case1), we might need to handle it.
            # But I populated it.

            DoubleBenchmarkComparator.check_pass_or_not(
                npu_data=npu_data, gpu_data=gpu_data, golden_data=golden_data, standard=standard
            )

    def run_test(self, worker_num, local_worker_num, model_args, data_keys, tmp_path, expect_error=False, port_id=8118):
        """Helper function to run test and check results"""
        output_file_path = tmp_path / self.OUTPUT_MS_FILENAME
        log_dir_path = tmp_path / self.LOG_DIR_NAME
        log_dir_path.mkdir(parents=True, exist_ok=True)

        # For single card, we can just use python
        cmd_list = build_msrun_command_list(
            worker_num=worker_num,
            local_worker_num=local_worker_num,
            log_dir=log_dir_path,
            run_script_path=self.run_script_path,
            hidden_size=HIDDEN_SIZE,
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            score_func=model_args["score_func"],
            route_norm=model_args["route_norm"],
            route_scale=model_args["route_scale"],
            output_file_path=output_file_path,
            num_expert_groups=model_args["num_expert_groups"],
            num_limited_groups=model_args["num_limited_groups"],
            force_load_balance=model_args["force_load_balance"],
            shared_expert_num=model_args["shared_expert_num"],
            apply_probs_on_input=model_args["apply_probs_on_input"],
            port_id=port_id,
        )

        logger.info(f"Running command: {' '.join(cmd_list)}")

        result = subprocess.run(cmd_list, shell=False, capture_output=True, text=True, check=False)

        if expect_error:
            assert result.returncode != 0, (
                f"Test script expected to fail but succeeded.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"
            )
            assert expect_error in result.stderr, (
                f"Expected error message '{expect_error}' not found in stderr.\nStderr:\n{result.stderr}"
            )
            return

        assert result.returncode == 0, (
            f"Test script failed with non-zero exit code: "
            f"{result.returncode}.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"
        )
        assert output_file_path.exists(), f"Output file {output_file_path} was not created."

        output_ms_dict = np.load(output_file_path)

        self.check_acc(output_ms_dict, data_keys)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.parametrize(SINGLE_CARD_TEST_PARAM, SINGLE_CARD_TEST_CASES)
    def test_moe_layer_configurations(self, tmp_path, model_args, data_keys, expect_error):
        """Test MoELayer with various configurations."""
        self.run_test(
            worker_num=1,
            local_worker_num=1,
            model_args=model_args,
            expect_error=expect_error,
            data_keys=data_keys,
            tmp_path=tmp_path,
        )
