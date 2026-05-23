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
"""Test LossCallback"""

from unittest.mock import MagicMock, patch
import time
import pytest

from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindformers.pynative.callback.loss_callback import LossCallback


# pylint: disable=protected-access
@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestLossCallback:
    """Test LossCallback class"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.callback = LossCallback(log_interval=1)
        self.args = MagicMock()
        self.state = MagicMock()
        self.state.global_step = 1
        self.state.max_steps = 10
        self.state.global_batch_size = 2
        self.state.total_flops = 0

    @staticmethod
    def _make_model_mock(
        moe_aux_loss_coeff=0.0,
        moe_router_load_balancing_type="aux_loss",
        moe_router_enable_expert_bias=False,
        num_layers=12,
        first_k_dense_replace=0,
        moe_layer_freq=None,
        mtp_num_layers=0,
    ):
        """Create a mock model with a properly configured TransformerConfig mock."""
        model = MagicMock()
        model_config = MagicMock()
        model_config.moe_aux_loss_coeff = moe_aux_loss_coeff
        model_config.moe_router_load_balancing_type = moe_router_load_balancing_type
        model_config.moe_router_enable_expert_bias = moe_router_enable_expert_bias
        model_config.num_layers = num_layers
        model_config.moe_layer_freq = moe_layer_freq
        model_config.first_k_dense_replace = first_k_dense_replace
        model_config.mtp_num_layers = mtp_num_layers
        model.get_gpt_transformer_config.return_value = model_config
        model.cells_and_names.return_value = []
        return model

    def test_init(self):
        """Test initialization"""
        callback = LossCallback(log_interval=5)
        assert callback.log_interval == 5
        assert callback.step_time is not None
        assert callback.epoch_time is not None

    def test_on_train_begin(self):
        """Test on_train_begin"""
        self.callback.step_time = 0
        self.callback.epoch_time = 0
        self.callback.on_train_begin(self.args, self.state)
        assert self.callback.step_time != 0
        assert self.callback.epoch_time != 0

    def test_on_epoch_begin(self):
        """Test on_epoch_begin"""
        self.callback.epoch_time = 0
        self.callback.on_epoch_begin(self.args, self.state)
        assert self.callback.epoch_time != 0

    def test_on_step_begin(self):
        """Test on_step_begin"""
        self.callback.step_time = 0
        self.callback.on_step_begin(self.args, self.state)
        assert self.callback.step_time != 0

    def test_on_epoch_end(self):
        """Test on_epoch_end"""
        self.callback.on_epoch_end(self.args, self.state)

    @patch("mindformers.pynative.callback.loss_callback.logger")
    @patch("mindformers.pynative.callback.loss_callback.get_world_size")
    @patch("mindformers.pynative.callback.loss_callback.num_floating_point_operations")
    @patch(
        "mindformers.pynative.callback.loss_callback.convert_transformer_config_to_args_for_tflops"
    )
    def test_on_step_end(
        self, mock_convert, mock_flops, mock_get_world_size, mock_logger
    ):
        """Test on_step_end"""
        mock_get_world_size.return_value = 1
        mock_flops.return_value = 1000
        mock_convert.return_value = MagicMock()

        model = self._make_model_mock()

        # Case 1: loss is None
        self.callback.on_step_end(self.args, self.state, loss=None)
        mock_logger.info.assert_not_called()

        # Case 2: step not in log interval
        self.callback.log_interval = 2
        self.state.global_step = 1
        self.callback.on_step_end(self.args, self.state, loss=1.0, model=model)
        mock_logger.info.assert_not_called()

        # Case 3: Normal logging
        self.callback.log_interval = 1
        self.state.global_step = 1
        self.callback.step_time = time.time() - 0.1

        lr_scheduler = MagicMock(spec=LearningRateSchedule)
        lr_scheduler.side_effect = lambda x: 0.01

        self.callback.on_step_end(
            self.args,
            self.state,
            loss=1.0,
            grad_norm=0.5,
            model=model,
            lr_scheduler=lr_scheduler,
        )

        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args[0][0]
        assert "loss:   1.000000" in call_args
        assert "grad_norm:   0.500000" in call_args
        assert "lr: 1.000000e-02" in call_args
        assert "throughput:" in call_args

    @patch("mindformers.pynative.callback.loss_callback.logger")
    @patch("mindformers.pynative.callback.loss_callback.get_world_size")
    @patch("mindformers.pynative.callback.loss_callback.num_floating_point_operations")
    @patch(
        "mindformers.pynative.callback.loss_callback.convert_transformer_config_to_args_for_tflops"
    )
    def test_on_step_end_edge_cases(
        self, mock_convert, mock_flops, mock_get_world_size, mock_logger
    ):
        """Test on_step_end edge cases"""
        mock_get_world_size.return_value = 1
        mock_flops.return_value = 1000
        mock_convert.return_value = MagicMock()
        model = self._make_model_mock()

        # Case: grad_norm is None, lr_scheduler is None
        self.callback.step_time = time.time() - 0.1
        self.callback.on_step_end(
            self.args,
            self.state,
            loss=1.0,
            grad_norm=None,
            model=model,
            lr_scheduler=None,
        )
        call_args = mock_logger.info.call_args[0][0]
        assert "grad_norm: NaN" in call_args
        assert "lr:" not in call_args

    def test_parse_lr_info(self):
        """Test _parse_lr_info"""
        # Case 1: LearningRateSchedule
        lr_scheduler = MagicMock(spec=LearningRateSchedule)
        lr_scheduler.side_effect = lambda x: 0.01
        lr = self.callback._parse_lr_info(lr_scheduler, 1)
        assert lr == 0.01

        # Case 2: Not LearningRateSchedule
        lr = self.callback._parse_lr_info(None, 1)
        assert lr is None

    def test_to_float(self):
        """Test _to_float"""
        # Case 1: None
        assert self.callback._to_float(None) is None

        # Case 2: Object with asnumpy
        mock_tensor = MagicMock()
        mock_tensor.asnumpy.return_value = 1.5
        assert self.callback._to_float(mock_tensor) == 1.5

        # Case 3: Object with item (but no asnumpy)
        mock_tensor_item = MagicMock()
        del mock_tensor_item.asnumpy
        mock_tensor_item.item.return_value = 2.5
        assert self.callback._to_float(mock_tensor_item) == 2.5

        # Case 4: Scalar
        assert self.callback._to_float(3.5) == 3.5
