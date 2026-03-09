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
"""Checkpoint callback for saving model checkpoints during training."""
import os
from mindformers.pynative.callback.callback import TrainerCallback
from mindformers.tools.logger import logger
from mindformers.checkpoint import save_checkpoint
from mindformers.checkpoint.checkpoint import CommonInfo, AsyncSaveManager


class CheckpointCallback(TrainerCallback):
    """
    Callback for saving model checkpoints during training.

    This callback saves model checkpoints at specified intervals and at the end of training.
    It can save both the model parameters and optimizer state.

    Args:
        save_path (str): Directory where checkpoints will be saved.
        save_interleaved_steps (int): Number of steps between checkpoint saves. Default: 1000.
        no_save_optim (bool): Whether to skip saving optimizer state. Default: False.
        save_max (int): Maximum number of checkpoints to keep. Default: 3.
        prefix (str): Prefix for checkpoint file names. Default: "checkpoint".
        async_save (bool): Whether to enable asynchronous saving. Default: False.
        remove_redundancy (bool): Whether to remove redundancy when saving. Default: False.
    """

    def __init__(
        self,
        save_path: str,
        save_max: int = 3,
        save_interleaved_steps: int = 1000,
        no_save_optim: bool = False,
        async_save: bool = False,
        prefix: str = "checkpoint",
        remove_redundancy: bool = False,
    ):
        """
        Initialize the CheckpointCallback.
        """
        super().__init__()
        self.save_path = save_path
        self.save_max = save_max
        self.save_interleaved_steps = save_interleaved_steps
        self.no_save_optim = no_save_optim
        self.prefix = prefix
        self.async_save = async_save
        self.remove_redundancy = remove_redundancy

        if not self.save_path:
            raise ValueError("save_path must be provided for CheckpointCallback.")

        # Initialize the async save manager
        self.async_save_manager = None
        if self.async_save:
            self.async_save_manager = AsyncSaveManager(async_save=True)
            logger.info("AsyncSaveManager created")

    def on_train_begin(self, args, state, **kwargs):
        """
        Called at the beginning of training.

        Creates the save directory if it doesn't exist.

        Args:
            args: Training arguments.
            state: Trainer state.
            **kwargs: Additional keyword arguments.
        """
        if self.save_path and not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
            logger.info(f"Created checkpoint directory: {self.save_path}")

    def on_step_end(self, args, state, **kwargs):
        """
        Called at the end of each training step.

        Saves checkpoint if the current step matches the save interval.

        Args:
            args: Training arguments.
            state: Trainer state.
            **kwargs: Additional keyword arguments including:
                - model: The model to save.
                - optimizer: The optimizer to save (saved unless no_save_optim is True).
        """
        # Skip saving when the interval is not reached.
        if state.global_step % self.save_interleaved_steps != 0:
            return

        self._save_checkpoint(args, state, **kwargs)

    def on_train_end(self, args, state, **kwargs):
        """
        Called at the end of training.

        Always saves a final checkpoint at the end of training.

        Args:
            args: Training arguments.
            state: Trainer state.
            **kwargs: Additional keyword arguments.
        """
        self._save_checkpoint(args, state, **kwargs)
        logger.info("Training completed. Final checkpoint saved.")

    def _save_checkpoint(self, args, state, **kwargs):
        """
        Save a checkpoint using mindformers.checkpoint.save_checkpoint.

        Args:
            args: Training arguments.
            state: Trainer state.
            **kwargs: Additional keyword arguments including model and optimizer.
        """
        # Skip saving when no target path is configured.
        if not self.save_path:
            return

        # args is currently unused, keep for callback signature compatibility.
        _ = args
        model = kwargs.get("model", None)
        optimizer = kwargs.get("optimizer", None)

        if model is None:
            logger.warning("No model provided to CheckpointCallback, skipping save.")
            return

        # CommonInfo provides metadata required by save_checkpoint.
        common_info = self._create_common_info(state)

        try:
            # Prepare the async manager before any save operation.
            if self.async_save_manager is not None:
                self.async_save_manager.prepare_before_save()

            # Save model parameters and (optionally) optimizer state.
            save_checkpoint(
                iteration=common_info.global_step,
                network=model,
                optimizer=None if self.no_save_optim else optimizer,
                async_save_manager=self.async_save_manager,
                common_info=common_info,
                keep_max_num=self.save_max,
                user_prefix=self.prefix,
                save_checkpoint_path=self.save_path,
                remove_redundancy=self.remove_redundancy,
            )

            logger.info(
                f"Checkpoint saved at step {common_info.global_step} to {self.save_path} "
                f"(async={self.async_save}, remove_redundancy={self.remove_redundancy})"
            )

        except (RuntimeError, ValueError, FileNotFoundError) as e:
            logger.error(f"Error saving checkpoint: {e}")

    def _create_common_info(self, state) -> CommonInfo:
        """
        Create CommonInfo from TrainerState.

        Args:
            state: Trainer state containing training information.

        Returns:
            CommonInfo: An instance of CommonInfo.
        """
        common_info = CommonInfo()

        if getattr(state, "epoch", None):
            common_info.epoch_num = int(state.epoch)
        else:
            common_info.epoch_num = 1

        if getattr(state, "max_steps", None):
            common_info.step_num = int(state.max_steps)

        if getattr(state, "loss_scale", None):
            common_info.loss_scale = state.loss_scale
        else:
            common_info.loss_scale = 1.0

        if getattr(state, "global_step", None):
            common_info.global_step = state.global_step

        if getattr(state, "global_batch_size", None):
            common_info.global_batch_size = state.global_batch_size

        logger.debug(
            f"Created CommonInfo: epoch={common_info.epoch_num}, "
            f"step={common_info.step_num}, global_step={common_info.global_step}"
        )

        return common_info
