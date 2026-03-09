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
"""Loss callback for logging training loss."""
import time
from typing import Dict, Any
from copy import deepcopy

from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore.mint.distributed import get_world_size

from mindformers.pynative.callback.callback import TrainerCallback
from mindformers.tools.logger import logger
from mindformers.models.utils import (
    convert_transformer_config_to_args_for_tflops,
    num_floating_point_operations,
)


class LossCallback(TrainerCallback):
    """
    Callback for logging loss and other training metrics during training.

    This callback logs the loss, learning rate, gradient norm, and step time
    at specified intervals.

    Args:
        log_interval (int): Number of steps between loss logging. Default: 1
    """

    def __init__(self, log_interval: int = 1):
        """
        Initialize the LossCallback.
        """
        super().__init__()
        self.log_interval = log_interval
        self.step_time = time.time()
        self.epoch_time = time.time()

    def on_train_begin(self, args, state, **kwargs):
        """
        Called at the beginning of training.

        Args:
            args: Training arguments.
            state: Trainer state.
            **kwargs: Additional keyword arguments.
        """
        self.step_time = time.time()
        self.epoch_time = time.time()
    
    def on_train_end(self, args, state, **kwargs):
        """
        Called at the end of training.

        Args:
            args: Training arguments.
            state: Trainer state.
            **kwargs: Additional keyword arguments.
        """
        # update train state
        state.is_train_begin = False
        state.is_train_end = True

    def on_epoch_begin(self, args, state, **kwargs):
        """
        Called at the beginning of each epoch.

        Args:
            args: Training arguments.
            state: Trainer state.
            **kwargs: Additional keyword arguments.
        """
        self.epoch_time = time.time()

    def on_step_begin(self, args, state, **kwargs):
        """
        Called at the beginning of each training step.

        Args:
            args: Training arguments.
            state: Trainer state.
            **kwargs: Additional keyword arguments.
        """
        self.step_time = time.time()

    def on_step_end(self, args, state, **kwargs):
        """
        Called at the end of each training step.

        Logs the loss value and optionally computes statistics.

        Args:
            args: Training arguments.
            state: Trainer state.
            **kwargs: Additional keyword arguments, including:
                - loss: The current step loss value.
                - grad_norm: The gradient norm (optional).
                - lr_scheduler: The learning rate scheduler (optional).
        """
        loss = kwargs.get("loss")
        if loss is None or state.global_step % self.log_interval != 0:
            return

        grad_norm = kwargs.get("grad_norm")

        # Calculate the time cost for the current step in milliseconds
        cur_time = time.time()
        step_time_cost = int((cur_time - self.step_time) * 1000)

        # Calculate total FLOPs for the model
        model = kwargs.get("model")
        model_config = deepcopy(model.get_gpt_transformer_config())
        model_config = convert_transformer_config_to_args_for_tflops(model_config)
        state.total_flops = num_floating_point_operations(
            model_config, state.global_batch_size
        )

        # Calculate the throughput for the current step
        throughput = state.total_flops / (step_time_cost * 10**12 * get_world_size())

        # Prepare log information dictionary
        log_info = {
            "loss": self._to_float(loss),
            "grad_norm": self._to_float(grad_norm),
            "cur_step": state.global_step,
            "max_steps": state.max_steps,
            "step_time": step_time_cost,
            "throughput": throughput,
            "learning_rate": self._parse_lr_info(
                kwargs.get("lr_scheduler"), state.global_step
            ),
        }

        # Print the collected log information
        self._print_log(log_info)

    def on_epoch_end(self, args, state, **kwargs):
        """
        Called at the end of each epoch.

        Args:
            args: Training arguments.
            state: Trainer state.
            **kwargs: Additional keyword arguments.
        """

    def _print_log(self, log_info: Dict[str, Any]):
        """
        Format and print the log information.

        Args:
            log_info (Dict[str, Any]): A dictionary containing the metrics to log.
        """
        cur_step = log_info.get("cur_step", 0)
        max_steps = log_info.get("max_steps", 0)
        # Construct the log message parts
        log_parts = [
            f"step:[{cur_step:5d}/{max_steps:5d}]",
            f"loss: {log_info.get('loss'):10.6f}",
            f"per_step_time: {log_info.get('step_time'):6d}ms",
        ]

        lr = log_info.get("learning_rate")
        if lr is not None:
            log_parts.append(f"lr: {lr:10.6e}")

        grad_norm = log_info.get("grad_norm")
        if grad_norm is not None:
            log_parts.append(f"grad_norm: {grad_norm:10.6f}")
        else:
            log_parts.append("grad_norm: NaN")

        throughput = log_info.get("throughput")
        if throughput is not None:
            log_parts.append(f"throughput: {throughput:6.2f}T")

        # Log the formatted message
        logger.info(f"{{ {', '.join(log_parts)} }}")

    def _parse_lr_info(self, lr_scheduler, step):
        """
        Parse the learning rate for the current step.

        Args:
            lr_scheduler: The learning rate scheduler object.
            step (int): The current global step.

        Returns:
            float: The learning rate value, or None if not available.
        """
        if isinstance(lr_scheduler, LearningRateSchedule):
            return self._to_float(lr_scheduler(step))
        return None

    @staticmethod
    def _to_float(data):
        """
        Convert data to a float value.

        Args:
            data: The input data (Tensor, numpy array, or scalar).

        Returns:
            float: The float representation of the data, or None if input is None.
        """
        if data is None:
            return data

        if hasattr(data, "asnumpy"):
            return float(data.asnumpy())
        if hasattr(data, "item"):
            return data.item()
        return float(data)
