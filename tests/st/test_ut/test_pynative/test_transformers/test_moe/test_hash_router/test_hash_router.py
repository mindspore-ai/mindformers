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
"""
Hash route accuracy comparison test (double benchmark comparison).

Test Procedure:
    1. Run `run_hash_router.py` on the NPU via a subprocess call.
    2. Load the NPU output and compare it with `GPU_DATA` / `GOLDEN_DATA` using DoubleBenchmarkComparator.
    3. Perform precise element-by-element verification using `selected_experts_indices`.

usage:
    pytest tests/st/test_ut/test_pynative/test_transformers/test_moe/test_hash_router/test_hash_router.py -v
"""

from pathlib import Path
import subprocess

import numpy as np
import pytest

from tests.utils.double_benchmark import DoubleBenchmarkComparator, DoubleBenchmarkStandard

from .hash_routing_data_gen import (
    CASE_KEYS, CASE_INPUT_REGISTRY, GOLDEN_DATA, GPU_DATA,
)

SINGLE_CARD_TEST_PARAM = "case_key"


class TestHashRouterPrecision:
    """Double-benchmark precision test for hash router."""

    OUTPUT_MS_FILENAME = "output_ms.npz"
    LOG_DIR_NAME = "msrun_log"

    def setup_method(self):
        """Setup test environment."""
        self.sh_path = Path(__file__).parent.resolve()
        self.run_script_path = self.sh_path / "run_hash_router.py"

    def _assert_data_loaded(self):
        """Ensure there is output benchmark data."""
        assert len(GOLDEN_DATA) > 0, (
            "GOLDEN_DATA is empty! Run Megatron CPU benchmark script first, "
            "then copy the printed GOLDEN_DATA snippet into "
            "hash_routing_data_gen.get_golden()."
        )
        assert len(GPU_DATA) > 0, (
            "GPU_DATA is empty! Run Megatron GPU benchmark script first, "
            "then copy the printed GPU_DATA snippet into "
            "hash_routing_data_gen.get_gpu_datas()."
        )

    def check_acc(self, output_ms_dict, case_key):
        """Check the output accuracy with double-benchmark."""
        standard = DoubleBenchmarkStandard(dtype="float32")

        for field in ["top_scores"]:
            npu_val = output_ms_dict.get(field)
            golden_key = f"{field}_{case_key}"
            gpu_key = f"{field}_{case_key}"

            assert npu_val is not None, f"'{field}' not in NPU output"
            assert golden_key in GOLDEN_DATA, f"'{golden_key}' not in GOLDEN_DATA"
            assert gpu_key in GPU_DATA, f"'{gpu_key}' not in GPU_DATA"

            DoubleBenchmarkComparator.check_pass_or_not(
                npu_data=npu_val,
                gpu_data=GPU_DATA[gpu_key],
                golden_data=GOLDEN_DATA[golden_key],
                standard=standard,
            )

    def check_indices_exact(self, output_ms_dict, case_key):
        """Check the indices exact."""
        npu_val = output_ms_dict.get("selected_experts_indices")
        golden_key = f"selected_experts_indices_{case_key}"

        assert npu_val is not None, "selected_experts_indices not in NPU output"
        assert golden_key in GOLDEN_DATA, f"'{golden_key}' not in GOLDEN_DATA"

        golden_val = GOLDEN_DATA[golden_key]

        if not np.array_equal(npu_val, golden_val):
            mismatch_count = int(np.sum(npu_val != golden_val))
            raise AssertionError(
                f"selected_experts_indices mismatch: {mismatch_count}/{npu_val.size} "
                f"elements differ.\n"
                f"First 5 mismatch positions: "
                f"{np.argwhere(npu_val != golden_val)[:5].flatten().tolist()}"
            )

    def run_test(
            self,
            case_key,
            tmp_path,
            worker_num=1,
            local_worker_num=1,
            port_id=8118,
            enable_aux=False,
    ):
        """
        Test hash routing test cases with single-card.
        """
        self._assert_data_loaded()

        output_path = tmp_path / self.OUTPUT_MS_FILENAME
        log_dir = tmp_path / self.LOG_DIR_NAME
        log_dir.mkdir(parents=True, exist_ok=True)

        if worker_num == 1 and local_worker_num == 1:
            cmd = ["python"]
        else:
            cmd = [
                "msrun",
                f"--worker_num={worker_num}",
                f"--local_worker_num={local_worker_num}",
                f"--master_port={port_id}",
                f"--log_dir={log_dir}",
                "--join=True",
            ]

        cmd += [
            str(self.run_script_path),
            f"--case-key={case_key}",
            f"--output-path={output_path}",
        ]
        if enable_aux:
            cmd.append("--enable-aux")

        result = subprocess.run(cmd, shell=False, capture_output=True, text=True, check=False)

        assert result.returncode == 0, (
            f"Hash router NPU script failed (code={result.returncode}).\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        assert output_path.exists(), f"Output not created: {output_path}"

        output_ms_dict = dict(np.load(output_path, allow_pickle=True))

        self.check_indices_exact(output_ms_dict, case_key)
        self.check_acc(output_ms_dict, case_key)
        return output_ms_dict

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.parametrize(SINGLE_CARD_TEST_PARAM, CASE_KEYS)
    def test_single_card_precision(self, case_key, tmp_path):
        """
        Test All hash routing test cases with single-card.
        """
        self.run_test(case_key, tmp_path, worker_num=1, local_worker_num=1)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_hash_seq_aux_matches_megatron_natural_topk(self, tmp_path):
        """Hash seq-aux follows Megatron's logits-derived natural top-k map."""
        case_key = "v50_e4_k1_softmax_s1.0_l4_b2_h16"
        output = self.run_test(case_key, tmp_path, enable_aux=True)
        data = CASE_INPUT_REGISTRY[case_key]

        hidden = data["hidden_states"].astype(np.float32)
        weight = data["weight"].astype(np.float32)
        logits = hidden.reshape(-1, hidden.shape[-1]) @ weight.T
        logits -= logits.max(axis=-1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)

        selected = output["selected_experts_indices"].astype(np.int64)
        routing_map = np.zeros_like(probabilities, dtype=np.float32)
        np.put_along_axis(routing_map, selected, 1.0, axis=1)

        seq_length, batch_size = hidden.shape[:2]
        num_experts = data["num_experts"]
        top_k = data["top_k"]
        probabilities = probabilities.reshape(seq_length, -1)
        routing_map = routing_map.reshape(seq_length, -1)
        expected_raw_aux = (
            (probabilities.sum(axis=0) * routing_map.sum(axis=0)).sum()
            * num_experts
            / (top_k * seq_length * seq_length * batch_size)
        )

        natural_topk = np.argmax(probabilities.reshape(-1, num_experts), axis=1)
        natural_map = np.zeros_like(probabilities.reshape(-1, num_experts))
        np.put_along_axis(natural_map, natural_topk[:, None], 1.0, axis=1)
        natural_map = natural_map.reshape(seq_length, -1)
        natural_raw_aux = (
            (probabilities.sum(axis=0) * natural_map.sum(axis=0)).sum()
            * num_experts
            / (top_k * seq_length * seq_length * batch_size)
        )

        assert abs(expected_raw_aux - natural_raw_aux) > 1.0e-4
        np.testing.assert_allclose(
            output["load_balancing_loss"], natural_raw_aux, rtol=0.0, atol=2.0e-6
        )
