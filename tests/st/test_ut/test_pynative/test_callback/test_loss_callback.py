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
        # on_epoch_end is currently empty, just ensuring it runs without error
        try:
            self.callback.on_epoch_end(self.args, self.state)
        except Exception as e:
            pytest.fail(f"on_epoch_end raised exception: {e}")

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

        model = MagicMock()
        model.get_gpt_transformer_config.return_value = MagicMock()

        # Case 1: loss is None
        self.callback.on_step_end(self.args, self.state, loss=None)
        mock_logger.info.assert_not_called()

        # Case 2: step not in log interval
        self.callback.log_interval = 2
        self.state.global_step = 1
        self.callback.on_step_end(self.args, self.state, loss=1.0)
        mock_logger.info.assert_not_called()

        # Case 3: Normal logging
        self.callback.log_interval = 1
        self.state.global_step = 1
        # Set step_time to be slightly in the past to avoid division by zero if implementation changes
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
        model = MagicMock()

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
        # lr should not be in logs if None (based on implementation reading)
        # Wait, let's check implementation:
        # lr = log_info.get("learning_rate")
        # if lr is not None: log_parts.append(...)
        # So check it's NOT present
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
        del mock_tensor_item.asnumpy  # Ensure no asnumpy
        mock_tensor_item.item.return_value = 2.5
        assert self.callback._to_float(mock_tensor_item) == 2.5

        # Case 4: Scalar
        assert self.callback._to_float(3.5) == 3.5
