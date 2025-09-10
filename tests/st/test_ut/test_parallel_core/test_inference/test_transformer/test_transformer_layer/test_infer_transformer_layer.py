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
"""mcore transformer layer UT of inference"""
from pathlib import Path
import subprocess
import pytest
import numpy as np

from mindformers.tools.logger import logger

from tests.utils.double_benchmark import DoubleBenchmarkStandard, DoubleBenchmarkComparator
from tests.st.test_ut.test_parallel_core.test_inference.test_transformer.test_transformer_layer.data_gen_utils import (
    get_init_params,
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEQ_LENGTH,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_FFN_HIDDEN_SIZE,
    DEFAULT_NUM_HEADS,
    GPU_DATA,
    GOLDEN_DATA,
)


SINGLE_CARD_TEST_PARAM = "model_args, data_keys, expect_error"
SINGLE_CARD_TEST_CASES = [
    (
        # 并行策略: 单卡, Transformer Layer组成模块: Standard Norm, SelfAttention, MLP
        # expected result: 功能跑通, 精度对齐。
        {"num_experts": None, "moe_grouped_gemm": False, "qk_layernorm": False, "multi_latent_attention": False,
         "qk_l2_norm": False, "sandwich_norm": False},
        {"output": "output_standard"},
        False,
    ),
    (
        # 并行策略: 单卡, Transformer Layer组成模块: Standard Norm, QK Norm, SelfAttention, MLP
        # expected result: 功能跑通, 精度对齐。
        {"num_experts": None, "moe_grouped_gemm": False, "qk_layernorm": True, "multi_latent_attention": False,
         "qk_l2_norm": False, "sandwich_norm": False},
        {"output": "output_qk"},
        False,
    ),
    (
        # 并行策略: 单卡, Transformer Layer组成模块: Standard Norm, Sandwich Norm, SelfAttention, MLP
        # expected result: 功能跑通, 精度对齐。
        {"num_experts": None, "moe_grouped_gemm": False, "qk_layernorm": False, "multi_latent_attention": False,
         "qk_l2_norm": False, "sandwich_norm": True},
        {"output": "output_sandwich"},
        False,
    ),
]


def build_msrun_command_list(
        worker_num, local_worker_num, log_dir, run_script_path,
        model_args, # This will be a dictionary
        output_path_param, tensor_parallel,
        # Default dimensions, can be overridden by model_args if specific tests need them
        batch_size=DEFAULT_BATCH_SIZE, seq_length=DEFAULT_SEQ_LENGTH,
        hidden_size=DEFAULT_HIDDEN_SIZE, ffn_hidden_size=DEFAULT_FFN_HIDDEN_SIZE,
        num_attention_heads=DEFAULT_NUM_HEADS, port=8118
):
    """ Build the msrun command with the specified parameters. """
    if worker_num == 1:
        cmd_list = ["python"]
    else:
        cmd_list = [
            "msrun",
            f"--worker_num={worker_num}",
            f"--local_worker_num={local_worker_num}",  # Should match NPU cards available
            f"--master_port={port}",  # Ensure port is unique per test run if parallelized at pytest level
            f"--log_dir={log_dir}",
            "--join=True"]
    cmd_list += [str(run_script_path),
                 f"--batch_size={model_args.get('batch_size', batch_size)}",
                 f"--seq_length={model_args.get('seq_length', seq_length)}",
                 f"--hidden_size={model_args.get('hidden_size', hidden_size)}",
                 f"--ffn_hidden_size={model_args.get('ffn_hidden_size', ffn_hidden_size)}",
                 f"--num_attention_heads={model_args.get('num_attention_heads', num_attention_heads)}",
                 # Add submodule choices
                 f"--moe_grouped_gemm={str(model_args['moe_grouped_gemm']).lower()}",
                 f"--qk_layernorm={str(model_args['qk_layernorm']).lower()}",
                 f"--multi_latent_attention={str(model_args['multi_latent_attention']).lower()}",
                 f"--qk_l2_norm={str(model_args['qk_l2_norm']).lower()}",
                 f"--sandwich_norm={str(model_args['sandwich_norm']).lower()}",
                 f"--output_path={output_path_param}",
                 f"--tensor_parallel={tensor_parallel}",
                 ]
    if model_args['num_experts'] is not None:
        cmd_list.append(f"--num_experts={model_args['num_experts']}")
    logger.info(f"Equivalent shell command for debugging (approximate): {' '.join(cmd_list)}")
    return cmd_list


class TestInferTransformerLayer:
    """Test class for Transformer Layer with different configurations"""
    OUTPUT_MS_FILENAME = "output_ms.npz"
    LOG_DIR_NAME = "msrun_log"

    def setup_method(self):
        """Setup method to prepare test environment"""
        self.sh_path = Path(__file__).parent.resolve()
        self.run_script_path = self.sh_path / "run_infer_transformer_layer.py"
        assert self.run_script_path.exists(), f"Run script not found: {self.run_script_path}"

    @staticmethod
    def check_function(output_ms_dict, model_args, data_keys):
        """
        Compare the shapes of output_ms and input_ms whether they are the same.
        """
        for key, _ in data_keys.items():
            output_data = output_ms_dict.get(key)
            input_data = get_init_params(
                model_args.get('batch_size', DEFAULT_BATCH_SIZE),
                model_args.get('seq_length', DEFAULT_SEQ_LENGTH),
                model_args.get('hidden_size', DEFAULT_HIDDEN_SIZE))["hidden_states"]

            assert output_data.shape == input_data.shape, \
                (f"The shapes of output data and input data are different, "
                 f"got output shape: {output_data.shape} and input shape: {input_data.shape}")

    @staticmethod
    def check_acc(output_ms_dict, data_keys):
        """Compare output_ms with GOLDEN_DATA and GPU_DATA using DoubleBenchmarkComparator."""
        standard = DoubleBenchmarkStandard(dtype="bfloat16")

        for key, data_key in data_keys.items():
            npu_data = output_ms_dict.get(key)
            golden_data = GOLDEN_DATA.get(data_key)
            gpu_data = GPU_DATA.get(data_key)

            DoubleBenchmarkComparator.check_pass_or_not(
                npu_data=npu_data,
                gpu_data=gpu_data,
                golden_data=golden_data,
                standard=standard
            )

    def check_result(
            self,
            output_file_path,
            model_args,
            data_keys,
            result,
            expect_error
    ):
        """Helper function to check results"""
        if expect_error:
            assert result.returncode != 0, (
                f"Expected an error but test script passed. "
                f"Stdout:\n{result.stdout}\n"
                f"Stderr:\n{result.stderr}"
            )
        else:
            assert result.returncode == 0, (
                f"Test script failed with non-zero exit code: "
                f"{result.returncode}.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"
            )
            assert output_file_path.exists(), (
                f"Output file {output_file_path} was not created."
            )

            output_ms_dict = np.load(output_file_path)

            # check whether the function of rotary embedding module works properly.
            self.check_function(output_ms_dict, model_args, data_keys)
            # Check whether the accuracy is consistent with vLLM
            # self.check_acc(output_ms_dict, data_keys)

    def run_test(
            self,
            worker_num,
            local_worker_num,
            model_args,
            data_keys,
            tmp_path,
            tensor_parallel=1,
            expect_error=False,
            port=8118
    ):
        """Helper function to run test and check results"""
        output_file_path = tmp_path / self.OUTPUT_MS_FILENAME
        log_dir_path = tmp_path / self.LOG_DIR_NAME
        log_dir_path.mkdir(parents=True, exist_ok=True)

        cmd_list = build_msrun_command_list(
            worker_num=worker_num,
            local_worker_num=local_worker_num,
            log_dir=log_dir_path,
            run_script_path=self.run_script_path,
            model_args=model_args,
            output_path_param=output_file_path,
            tensor_parallel=tensor_parallel,
            port=port
        )

        logger.info(f"Running command: {' '.join(cmd_list)}")
        cmd_result = subprocess.run(
            cmd_list, shell=False, capture_output=True, text=True, check=False)

        self.check_result(output_file_path, model_args, data_keys, cmd_result, expect_error)


class TestInferTransformerLayerSingleCard(TestInferTransformerLayer):
    """Test class for InferTransformerLayer with single card configurations"""
    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.parametrize(
        SINGLE_CARD_TEST_PARAM,
        SINGLE_CARD_TEST_CASES
    )
    def test_single_card_cases(self, model_args, data_keys, expect_error, tmp_path):
        """Test single card cases with various configurations."""
        logger.info(f"--- Running Single Card Test: model_args={model_args} ---")
        self.run_test(
            worker_num=1, local_worker_num=1,
            model_args=model_args, expect_error=expect_error,
            data_keys=data_keys,
            tmp_path=tmp_path
        )
