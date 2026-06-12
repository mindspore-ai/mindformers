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
"""Test Multi-Token Prediction (MTP) with various configurations."""
import os
import subprocess
from pathlib import Path
import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.pynative.transformers.multi_token_prediction import (
    MultiTokenPredictionBlock,
    MultiTokenPredictionLayer,
    roll_tensor as ms_roll_tensor,
    save_to_mtp_losses_tracker,
    get_mtp_layer_wise_logging_tracker,
    track_mtp_metrics,
)
from mindformers.pynative.base_models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_mtp_block_spec,
)
from mindformers.tools.logger import logger


_SEED = 42
_HIDDEN_SIZE = 64
_NUM_HEADS = 8
_MTP_NUM_LAYERS = 1   # pynative currently supports only mtp_num_layers=1


def _build_config(tp=1, cp=1, **overrides):
    """Build a TransformerConfig for MTP constructor tests."""
    return TransformerConfig(
        mtp_num_layers=_MTP_NUM_LAYERS,
        num_layers=4,
        hidden_size=_HIDDEN_SIZE,
        num_attention_heads=_NUM_HEADS,
        sequence_parallel=(tp > 1),
        context_parallel_size=cp,
        **overrides,
    )


def _build_mtp_block(config, fused_norm=True, normalization="RMSNorm"):
    """Build a MultiTokenPredictionBlock from GPT layer specs."""
    transformer_layer_spec = get_gpt_layer_local_spec(
        fused_norm=fused_norm, normalization=normalization,
    )
    mtp_block_spec = get_gpt_mtp_block_spec(
        config=config, spec=transformer_layer_spec, normalization=normalization,
    )
    return MultiTokenPredictionBlock(config=config, spec=mtp_block_spec)


def _param_count(cell):
    """Count total number of parameters in a cell."""
    total = 0
    for p in cell.get_parameters():
        total += np.prod(p.shape)
    return total


class TestMultiTokenPredictionLayer:
    """Test MTP layer constructor and parameter counts."""

    def test_constructor_tp1(self):
        """Verify layer count, weight shapes, and total parameter count for tp=1."""
        ms.set_seed(_SEED)
        config = _build_config(tp=1, cp=1)
        block = _build_mtp_block(config)

        assert len(block.layers) == _MTP_NUM_LAYERS
        for i in range(_MTP_NUM_LAYERS):
            layer = block.layers[i]
            assert isinstance(layer, MultiTokenPredictionLayer)
            assert layer.enorm.weight.shape == (_HIDDEN_SIZE,)
            assert layer.hnorm.weight.shape == (_HIDDEN_SIZE,)
            assert layer.eh_proj.weight.shape[0] == _HIDDEN_SIZE
            assert layer.eh_proj.weight.shape[1] == _HIDDEN_SIZE * 2
            assert layer.transformer_layer is not None

        assert _param_count(block) == 58240, \
            f"tp=1 expected 58240, got {_param_count(block)}"


class TestRollTensor:
    """Unit tests for the roll_tensor utility with various shifts and dims."""

    def test_roll_left_basic(self):
        """Shift a 1-D tensor left by one position."""
        tensor = Tensor([1, 2, 3, 4, 5], dtype=ms.float32)
        rolled = ms_roll_tensor(tensor, shifts=-1, dims=-1)
        expected = Tensor([2, 3, 4, 5, 0], dtype=ms.float32)
        assert np.allclose(rolled.asnumpy(), expected.asnumpy())

    def test_roll_left_2d(self):
        """Shift a 2-D tensor left by one position along the last dim."""
        tensor = Tensor([[1, 2, 3], [4, 5, 6]], dtype=ms.float32)
        rolled = ms_roll_tensor(tensor, shifts=-1, dims=-1)
        expected = Tensor([[2, 3, 0], [5, 6, 0]], dtype=ms.float32)
        assert np.allclose(rolled.asnumpy(), expected.asnumpy())

    def test_roll_positive_shift(self):
        """Shift a 1-D tensor right by two positions."""
        tensor = Tensor([1, 2, 3, 4, 5], dtype=ms.float32)
        rolled = ms_roll_tensor(tensor, shifts=2, dims=-1)
        expected = Tensor([0, 0, 1, 2, 3], dtype=ms.float32)
        assert np.allclose(rolled.asnumpy(), expected.asnumpy())

    def test_roll_no_shift(self):
        """Verify that zero shift returns the original tensor unchanged."""
        tensor = Tensor([1, 2, 3], dtype=ms.int32)
        rolled = ms_roll_tensor(tensor, shifts=0, dims=-1)
        assert np.allclose(rolled.asnumpy(), tensor.asnumpy())


class TestMTPLossLoggingHelper:
    """Test the MTP loss logging tracker and metrics aggregation."""
    _NUM_LAYERS = 4

    def setup_method(self):
        """Clear the global logging tracker before each test."""
        get_mtp_layer_wise_logging_tracker().clear()

    def teardown_method(self):
        """Clear the global logging tracker after each test."""
        get_mtp_layer_wise_logging_tracker().clear()

    def test_save_loss_to_tracker(self):
        """Verify a single loss value is saved to the correct tracker slot."""
        loss = Tensor(1.3, dtype=ms.float32)
        save_to_mtp_losses_tracker(loss=loss, layer_number=2, num_layers=self._NUM_LAYERS)
        tracker = get_mtp_layer_wise_logging_tracker()
        assert "values" in tracker
        assert tracker["values"].shape == (self._NUM_LAYERS,)
        assert np.isclose(tracker["values"][2].asnumpy(), loss.asnumpy())

    def test_track_mtp_metrics(self):
        """Populate all slots and verify aggregated metrics match the inputs."""
        loss = Tensor(2.3, dtype=ms.float32)
        for i in range(self._NUM_LAYERS):
            save_to_mtp_losses_tracker(loss=loss, layer_number=i, num_layers=self._NUM_LAYERS)
        mtp_metrics = track_mtp_metrics()
        assert mtp_metrics is not None
        assert mtp_metrics.shape == (self._NUM_LAYERS,)
        for i in range(self._NUM_LAYERS):
            assert np.isclose(mtp_metrics[i].asnumpy(), loss.asnumpy())


SINGLE_CARD_TEST_PARAM = "model_args"
SINGLE_CARD_TEST_CASES = [
    # case 1: learned_absolute position embeddings
    {"position_embedding_type": "learned_absolute"},
    # case 2: rope position embeddings
    {"position_embedding_type": "rope"},
]


def _build_command_list(run_script_path, model_args):
    """Build a command-line argument list from a dict of model arguments."""
    cmd_list = ["python", str(run_script_path)]
    for k, v in model_args.items():
        if isinstance(v, bool):
            if v:
                cmd_list.append(f"--{k}")
        else:
            cmd_list.append(f"--{k}={v}")
    logger.info(f"Test case shell command: {' '.join(cmd_list)}")
    return cmd_list


class TestMTPForwardAgainstMegatron:
    """End-to-end test that runs MTP forward pass via subprocess."""

    def setup_method(self):
        """Resolve the path to the run script before each test."""
        self.current_dir = Path(__file__).parent.resolve()
        self.run_script_path = self.current_dir / "run_multi_token_prediction.py"

    def _run_test(self, model_args):
        """Execute the MTP script in a subprocess and assert success."""
        cmd_list = _build_command_list(self.run_script_path, model_args)
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = "0"
        result = subprocess.run(
            cmd_list, shell=False, capture_output=True, text=True, check=False, env=env,
        )
        assert result.returncode == 0, (
            f"MTP script failed (exit {result.returncode}).\n"
            f"Stdout:\n{result.stdout}\nStderr:\n{result.stderr}"
        )

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @pytest.mark.parametrize(SINGLE_CARD_TEST_PARAM, SINGLE_CARD_TEST_CASES)
    def test_single_card_mtp_cases(self, model_args):
        """Parametrized test that runs MTP with different position embedding types."""
        self._run_test(model_args)
