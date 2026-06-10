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
"""Trainer for training models with MindFormers."""

import os
import enum
from typing import Optional, Callable, List, Dict, Any, Union
import json
import numpy as np

# mindspore modules
import mindspore as ms
from mindspore import Tensor, ops, manual_seed, set_deterministic, nn
from mindspore.dataset import Dataset
from mindspore.common import set_seed
from mindspore.mint.distributed import (
    init_process_group,
    destroy_process_group,
    get_world_size,
    all_reduce,
    new_group,
)
from mindspore.graph.api import _no_grad

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat

from mindformers.tools.logger import logger
from mindformers.pynative.config import TrainConfig
from mindformers.models import PreTrainedModel
from mindformers.checkpoint.checkpoint import load_checkpoint
from mindformers.pynative.callback import (
    CallbackHandler,
    TrainerCallback,
    LossCallback,
    CheckpointCallback,
    MonitorCallback,
    configure_max_logits_tracking,
    ensure_max_logits_reset_callback,
)
from mindformers.pynative.tools.profiler import Profiler
import mindformers.tools.register.register as register_module
import mindformers.dataset.handler.base_handler as handler_module
from mindformers.pynative.distributed.parallel_dims import ParallelDims
from mindformers.pynative.distributed.parallelize import parallelize_model
from mindformers.pynative.distributed.utils import get_loss_sense

from mindformers.checkpoint.checkpoint import CommonInfo, get_checkpoint_path

from .train_state import TrainerState
from ..tools.monitor import MonitorGroup
from .utils import (
    _build_model,
    _build_dataset,
    _build_optimizer,
    _build_lr_scheduler,
    _build_callback,
    _build_lora_model,
    _calculate_global_grad_norm,
    compute_parameters,
)


def patch_legacy_model_inference():
    """Patch legacy model inference."""
    return False


def patch_pynative_modules():
    """Patch is_legacy_model for pynative trainer."""
    register_module.is_legacy_model = patch_legacy_model_inference
    handler_module.is_legacy_model = patch_legacy_model_inference
    from mindformers.core import context as core_context
    core_context.is_legacy_model = patch_legacy_model_inference


class TrainMode(enum.Enum):
    """Training mode enumeration."""

    FINETUNE = "finetune"
    PRETRAIN = "pretrain"


def _cast_grad_to_fp32(grad):
    """Backward hook: cast gradient to fp32 if it is in a lower-precision dtype."""
    if grad.dtype in (ms.bfloat16, ms.float16):
        return ops.cast(grad, ms.float32)
    return grad


class Trainer:
    """
    Trainer for training models in MindFormers.
    """

    def __init__(
        self,
        config: Optional[Union[str, dict]] = None,
        model: PreTrainedModel = None,
        run_mode: Optional[str] = "train",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        optimizer: Optional[Any] = None,
        lr_scheduler: Optional[Any] = None,
        callbacks: Optional[List] = None,
        compute_loss_func: Optional[Callable] = None,
    ):
        """
        Initialize the Trainer.

        Args:
            config (Union[str, dict], optional): The configuration of the trainer.
                It can be a path to a yaml file or a dictionary. Defaults to None.
            model (PreTrainedModel, optional): The model to train. Defaults to None.
            run_mode (str, optional): The running mode. Can be "train", "finetune", "eval" or "predict".
                Defaults to "train".
            train_dataset (Dataset, optional): The training dataset. Defaults to None.
            eval_dataset (Dataset, optional): The evaluation dataset. Defaults to None.
            optimizer (Any, optional): The optimizer. Defaults to None.
            lr_scheduler (Any, optional): The learning rate scheduler. Defaults to None.
            callbacks (List, optional): The list of callbacks. Defaults to None.
            compute_loss_func (Callable, optional): The function to compute loss. Defaults to None.
        """
        # Verify instance validity when config is yaml file
        if isinstance(config, str) and any(
            [model, train_dataset, eval_dataset, optimizer, lr_scheduler, callbacks]
        ):
            logger.warning(
                "When config is a yaml file, (model, dataset, optimizer, lr_scheduler, callbacks) "
                "instances should not be provided. They will be built from config."
            )

        # patch legacy model inference for pynative mode
        patch_pynative_modules()

        # apply backward compat for `loss.backward()`
        enable_mindspore_backward_compat()

        # Initialize config
        self.config = self._init_config(config, run_mode)
        configure_max_logits_tracking(self.config, callbacks, optimizer)

        self._setup_seed_and_determinism()

        self.world_size = get_world_size()
        logger.info(f"Current world size: {self.world_size}.")
        self.enable_parallel = self.world_size > 1

        if self.enable_parallel:
            init_process_group()
            self.communication_init = True
            logger.info("Distributed communication is initialized.")
        
        # init states for pipeline parallel
        self.pp_enabled = self.config.parallelism.pipeline_parallel > 1
        self.parallel_dims = None
        self.metric_reduce_group = None
        self.metric_reduce_group_size = None

        self.global_batch_size = self.config.training.global_batch_size
        self.gradient_accumulation_steps = None
        self._compute_data_parallel_size()

        # Create model
        self.model = self._create_model(model, self.config.model)
        if self.enable_parallel:
            # Apply parallelism to model
            self.model, self.schedule, self.has_first, self.has_last = \
                self._apply_parallelism(self.model, self.config.parallelism)

        # After parallelism, parameters will be reset
        self.model = self.model if isinstance(self.model, list) else [self.model]
        for m in self.model:
            m.to_empty()
            with _no_grad():
                m.init_states()

        # Create train dataset
        self.train_dataset = self._create_dataset(
            train_dataset, getattr(self.config, "train_dataset", None)
        )
        if self.train_dataset is not None:
            self.train_epoch_step = self._get_dataset_size(self.train_dataset)
        else:
            self.train_epoch_step = self.config.training.steps
        self.train_num_epochs = max(
            self.config.training.steps // self.train_epoch_step, 1
        )

        # Create evaluate dataset
        self.eval_dataset = self._create_dataset(
            eval_dataset, getattr(self.config, "eval_dataset", None)
        )

        # Create optimizer and scheduler
        self.optimizer, self.lr_scheduler = self._create_optimizer_and_scheduler(
            optimizer,
            lr_scheduler,
            getattr(self.config, "optimizer", None),
            getattr(self.config, "lr_scheduler", None),
        )

        # Create callback handler
        self.callback_handler = self._create_callback_handler(callbacks, self.config)

        # Store other parameters
        self.compute_loss_func = compute_loss_func

        # Initialize monitor
        self.monitor = MonitorGroup(self.config, model=self.model[0])

        # Initialize training state
        self.communication_init = False
        self.state = None
        self._model_states_initialized = False

    @staticmethod
    def _init_config(config: Union[str, dict], run_mode: str = "train") -> TrainConfig:
        """
        Initialize trainer config from yaml file or dict instance.

        Args:
            config (Union[str, dict]): The configuration of the trainer.
            run_mode (str, optional): The running mode. Defaults to "train".

        Returns:
            TrainConfig: The initialized trainer configuration.

        Raises:
            ValueError: If config is None or invalid type.
            FileNotFoundError: If config file not found.
        """
        if config is None:
            raise ValueError(
                "config cannot be None. Please provide a yaml file path or dict."
            )

        if isinstance(config, str):
            if not os.path.exists(config):
                raise FileNotFoundError(f"Config file not found: {config}")

            logger.info(f"Loading config from yaml file: {config}")

            if run_mode == "train":
                config = TrainConfig.load_from_yaml(config)
            else:
                raise ValueError(f"Invalid run mode: {run_mode}")

        elif isinstance(config, dict):
            config = TrainConfig.from_dict(config)

        else:
            raise ValueError(
                f"Invalid config type: {type(config)}, expected str or dict."
            )

        logger.info(f"Current config: {json.dumps(config.to_dict(), indent=2)}")
        return config

    def _setup_seed_and_determinism(self):
        """Set random seed and deterministic mode."""
        # Set deterministic mode
        if self.config.training.deterministic:
            set_deterministic(True)

        # Set random seed
        seed = self.config.training.seed
        set_seed(seed)
        manual_seed(seed)
        np.random.seed(seed)

    def _apply_parallelism(self, model, parallelism):
        """
        Apply parallelism strategies (FSDP/TP/CP/AC) to the model.
        This follows TorchTitan's parallelization order.
        """
        logger.info("Applying parallelism to model...")

        # Build ParallelDims
        self.parallel_dims = ParallelDims(
            dp_replicate=parallelism.data_parallel_replicate,
            dp_shard=parallelism.data_parallel_shard,
            cp=parallelism.context_parallel,
            tp=parallelism.tensor_parallel,
            pp=parallelism.pipeline_parallel,
            ep=parallelism.expert_parallel,
            etp=parallelism.expert_tensor_parallel,
            world_size=self.world_size,
        )

        # Apply unified parallelization
        model, schedule, has_first, has_last = parallelize_model(
            model=model,
            parallel_dims=self.parallel_dims,
            parallelism=parallelism,
            recompute=self.config.recompute,
            recompute_comm=self.config.recompute_comm,
            swap=self.config.swap,
            accumulate_allreduce_grads_in_fp32=(
                self.config.optimizer.accumulate_allreduce_grads_in_fp32
            ),
            gradient_accumulation_steps=self.num_accumulation_steps,
        )
        return model, schedule, has_first, has_last

    def _create_model(self, model, model_config: Optional[Dict]) -> Any:
        """
        Create or validate model instance.

        Args:
            model (PreTrainedModel): The model instance.
            model_config (Optional[Dict]): The model configuration.

        Returns:
            Any: The created or validated model instance.

        Raises:
            ValueError: If neither model instance nor model config is provided.
        """
        # If user provided model instance, use it directly
        if model is not None:
            logger.info("Using user-provided model instance.")
            return model

        # Build model from config
        if model_config is None:
            raise ValueError("Either model instance or config.model must be provided.")

        logger.info("Building model from config...")
        with ms.DeviceCtx("meta"):
            logger.info("[DelayInit] Using meta device for model construction (PyNative mode).")
            model = _build_model(self.config)

        # Apply LoRA if provided (after base model is built)
        lora_config = getattr(self.config, "lora_config", None)
        if model is not None and lora_config is not None:
            logger.info("Applying LoRA configuration to model...")
            _build_lora_model(model, lora_config)

        # Compute parameters of the model
        compute_parameters(model)
        return model

    def _compute_data_parallel_size(self):
        """
        Compute the data parallel size based on the parallelism settings.
        """
        parallelism = self.config.parallelism
        tensor_parallel = parallelism.tensor_parallel
        pipeline_parallel = parallelism.pipeline_parallel
        context_parallel = parallelism.context_parallel

        data_parallel = self.world_size // (
            tensor_parallel * pipeline_parallel * context_parallel
        )
        if data_parallel <= 0:
            raise ValueError(
                f"tensor_parallel({tensor_parallel}) * pipeline_parallel({pipeline_parallel}) * "
                f"context_parallel({context_parallel}) must be less than or equal to world_size({self.world_size})."
            )

        parallelism.data_parallel = data_parallel
        if parallelism.data_parallel_shard < 0:
            dp_replicate = 1
        else:
            dp_replicate = max(data_parallel // parallelism.data_parallel_shard, 1)
        parallelism.data_parallel_replicate = dp_replicate
        parallelism.data_parallel_shard = data_parallel // dp_replicate

        logger.info(
            f"Got parallelism settings: {data_parallel=}"
            f"(dp_shard[{parallelism.data_parallel_shard}] * dp_replicate[{dp_replicate}]), "
            f"{tensor_parallel=}, {pipeline_parallel=}, {context_parallel=}"
        )

        # calculate gradient accumulation steps
        if (
            parallelism.data_parallel * self.config.training.local_batch_size
            > self.global_batch_size
        ):
            raise ValueError(
                "The product of data_parallel and local_batch_size exceeds global_batch_size, "
                "please increase global_batch_size or decrease local_batch_size."
            )

        self.num_accumulation_steps = self.global_batch_size // (
            parallelism.data_parallel * self.config.training.local_batch_size
        )
        parallelism.pipeline_parallel_microbatch_size = self.num_accumulation_steps
        logger.info(
            f"Calculate global_batch_size={self.global_batch_size}, "
            f"num_accumulation_steps={self.num_accumulation_steps}."
        )

    def _create_dataset(self, dataset, dataset_config: Optional[Dict]) -> Optional[Any]:
        """
        Create or validate dataset instance.

        Args:
            dataset (Dataset): The dataset instance.
            dataset_config (Optional[Dict]): The dataset configuration.

        Returns:
            Optional[Any]: The created or validated dataset instance.
        """
        # If user provided dataset instance, use it directly
        if dataset is not None:
            logger.info("Using user-provided dataset instance.")
            return dataset

        # If no config, return None
        if dataset_config is None:
            return None

        # Build dataset from config
        logger.info("Building dataset from config...")
        dataset = _build_dataset(
            dataset_config,
            self.global_batch_size,
            self.config.parallelism,
            self.num_accumulation_steps,
        )

        return dataset

    @staticmethod
    def _get_dataset_size(dataset) -> int:
        """
        Get the size of a dataset.
        """
        if hasattr(dataset, "get_dataset_size"):
            return dataset.get_dataset_size()
        if hasattr(dataset, "__len__"):
            return len(dataset)
        # Cannot determine dataset size; raise error per spec
        raise ValueError("Unable to determine dataset size from the provided dataset.")

    def _create_optimizer_and_scheduler(
        self,
        optimizer,
        lr_scheduler,
        optim_config: Optional[Dict],
        lr_config: Optional[Dict],
    ) -> tuple:
        """
        Create optimizer and learning rate scheduler.

        Args:
            optimizer (Any): The optimizer instance.
            lr_scheduler (Any): The learning rate scheduler instance.
            optim_config (Optional[Dict]): The optimizer configuration.
            lr_config (Optional[Dict]): The learning rate scheduler configuration.

        Returns:
            tuple: A tuple containing the optimizer and learning rate scheduler.
        """
        # If user provided instances, use them directly
        if optimizer is not None and lr_scheduler is not None:
            logger.info("Using user-provided optimizer and lr_scheduler instances.")
            return optimizer, lr_scheduler

        # Build from config
        if optim_config is None or lr_config is None:
            logger.warning("No optimizer or lr_scheduler config provided.")
            return None, None

        logger.info("Building optimizer and lr_scheduler from config...")

        # Build learning rate scheduler
        lr_scheduler = _build_lr_scheduler(lr_config, self.config.training.steps)

        # Build optimizer
        optimizer = _build_optimizer(optim_config, self.model, lr_scheduler)
        return optimizer, lr_scheduler

    def _create_built_in_callbacks(self) -> List[TrainerCallback]:
        """
        Create default callbacks.
        """
        checkpoint = self.config.checkpoint
        if not checkpoint.enable_save:
            return [LossCallback(), MonitorCallback()]

        # build checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_path=checkpoint.save_path,
            save_max=checkpoint.save_max,
            save_interleaved_steps=checkpoint.save_interleaved_steps,
            no_save_optim=checkpoint.no_save_optim,
            async_save=checkpoint.async_save,
            prefix=checkpoint.prefix,
            remove_redundancy=checkpoint.remove_redundancy,
            save_global_layout_cache=checkpoint.save_global_layout_cache,
        )

        return [
            LossCallback(),
            MonitorCallback(),
            checkpoint_callback,
        ]

    def _create_callback_handler(
        self, callbacks: Optional[List], config: Any
    ) -> CallbackHandler:
        """
        Create callback handler.

        Args:
            callbacks (Optional[List]): The list of callbacks.
            config (Any): The configuration.

        Returns:
            CallbackHandler: The created callback handler.
        """
        # Prepare initial callback list
        callback_list = self._create_built_in_callbacks()
        if callbacks:
            callback_list.extend(callbacks)

        # Build callbacks from config and extend
        callback_config = getattr(config, "callbacks", [])
        for callback in callback_config:
            callback_list.append(_build_callback(callback))

        callback_list = ensure_max_logits_reset_callback(
            callback_list,
            getattr(self.config.model, "track_max_attention_logit", False),
        )

        # Create handler with complete list
        cb_handler = CallbackHandler(
            callbacks=callback_list,
            model=self.model[0],
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
        )

        return cb_handler

    def _init_train_state(self):
        """Initialize training state."""
        return TrainerState(
            epoch_step=self.train_epoch_step,
            max_steps=self.config.training.steps,
            save_steps=self.config.checkpoint.save_interleaved_steps,
            global_batch_size=self.global_batch_size,
            num_accumulation_steps=self.num_accumulation_steps,
            is_train_begin=True,
            is_train_end=False,
        )

    def _register_grad_hooks(self):
        """Register backward hooks to cast low-precision gradients to fp32.

        When enabled, bf16/fp16 parameter gradients are cast to fp32 during backward
        so that gradient accumulation and all-reduce operate in fp32 precision,
        aligning with Megatron-LM's accumulate_allreduce_grads_in_fp32 behavior.
        """
        if not self.config.optimizer.accumulate_allreduce_grads_in_fp32:
            return

        low_precision_types = (ms.bfloat16, ms.float16)
        hook_count = 0
        for m in self.model:
            for param in m.trainable_params():
                if param.dtype in low_precision_types:
                    hook_count += 1
                    param.register_hook(_cast_grad_to_fp32)

        if hook_count > 0:
            logger.info(
                "[GradReduceFP32] Registered fp32-cast hooks on %d low-precision parameters",
                hook_count,
            )

    def train(
        self,
        checkpoint_path: Optional[str] = None,
        mode: str = "pretrain",
    ):
        """
        Execute the training loop.

        Args:
            checkpoint_path (Optional[str], optional): Path to load checkpoint from. Defaults to None.
            mode (str, optional): Training mode, can be "pretrain" or "finetune". Defaults to "pretrain".

        Raises:
            ValueError: If mode is not "pretrain" or "finetune".
        """
        # Validate mode
        if mode not in ["pretrain", "finetune"]:
            raise ValueError(f"mode must be 'pretrain' or 'finetune', got: {mode}")

        # Initialize training state
        self.state = self._init_train_state()

        if self.parallel_dims and self.parallel_dims.pp_enabled:
            loss_mesh = self.parallel_dims.get_mesh("loss_mesh")
            self.metric_reduce_group_size = loss_mesh.size()
            self.metric_reduce_group = loss_mesh.get_group()

        # Load checkpoint
        checkpoint_path = checkpoint_path or self.config.checkpoint.load_path
        if checkpoint_path:
            for m in self.model:
                self._load_checkpoint(checkpoint_path, m, self.optimizer)

        # Register gradient hooks for fp32 accumulation
        self._register_grad_hooks()

        # Call train begin callback
        self.callback_handler.on_train_begin(self.config, self.state)

        # Execute training loop
        self._inner_train_loop()

        # Call train end callback
        self.callback_handler.on_train_end(self.config, self.state)

        if self.communication_init:
            destroy_process_group()

    def _load_checkpoint(
        self,
        checkpoint_path: str,
        model,
        optimizer=None,
        global_step: Optional[int] = None,
    ):
        """
        Load checkpoint from file.

        Args:
            checkpoint_path (str): The path to the checkpoint file.
            model (PreTrainedModel): The model instance.
            optimizer (Any, optional): The optimizer instance. Defaults to None.
            global_step (Optional[int], optional): The global step. Defaults to None.

        Raises:
            ValueError: If model is None.
        """
        if model is None:
            raise ValueError("model is None, cannot load checkpoint.")

        checkpoint_path = get_checkpoint_path(checkpoint_path)
        logger.info(f"Loading checkpoint from: {checkpoint_path}")

        common_file = os.path.join(checkpoint_path, "common.json")
        common_info = CommonInfo.load_common(common_file)

        checkpoint = self.config.checkpoint

        if not checkpoint.no_load_optim:
            if optimizer is None:
                raise ValueError("If no_load_optim is False, optimizer is required.")

            global_step = common_info.global_step
            if common_info.global_batch_size != self.global_batch_size:
                global_step = int(
                    common_info.global_step
                    * (common_info.global_batch_size / self.global_batch_size)
                )
                logger.info(
                    f"Scaled global step: {common_info.global_step} -> {global_step} "
                    f"(batch size changed from {common_info.global_batch_size} to {self.global_batch_size})"
                )
            self.train_dataset.set_init_step(global_step)
            logger.info(f"Resume dataset from: {global_step}")

            self.state.global_step = global_step

        load_checkpoint(
            checkpoint=checkpoint_path,
            network=model,
            optimizer=optimizer if not checkpoint.no_load_optim else None,
            global_step=global_step,
            balanced_load=checkpoint.load_balanced,
        )

        # When optimizer state is not loaded (weights-only resume), the fp32 master
        # weights still hold their pre-load init values. Refresh them from the freshly
        # loaded model params so master and model start aligned.
        if checkpoint.no_load_optim and optimizer is not None:
            logger.info("no_load_optim=True: refreshing fp32 master weights from loaded model params.")
            optimizer.reload_main_params_from_model()

    def _inner_train_loop(self):
        """
        Internal training loop with gradient accumulation support.
        """
        # Create dataset iterator
        dataset_iter = self._create_dataset_iterator(self.train_dataset)

        for m in self.model:
            if hasattr(m, "set_train"):
                m.set_train(True)

        # Training loop
        logger.info("Start training loop...")
        step = self.state.global_step

        with Profiler(self.config.profiler) as prof:
            while step < self.state.max_steps:
                # Epoch begin callback
                if step % self.state.epoch_step == 0 and step > 0:
                    self.callback_handler.on_epoch_begin(self.config, self.state)
                    self.state.update_epoch()

                # Get batch data
                try:
                    inputs = self.get_batch(dataset_iter)
                except StopIteration:
                    # Recreate iterator if dataset exhausted
                    dataset_iter = self._create_dataset_iterator(self.train_dataset)
                    inputs = self.get_batch(dataset_iter)

                # Step begin callback
                self.callback_handler.on_step_begin(self.config, self.state)

                # Gradient accumulation: accumulate over micro-batches
                loss = 0.0
                grad_norm = 0.0
                self.monitor.reset()
                self.monitor.record("moe_tpe_step_begin")

                if self.parallel_dims and self.parallel_dims.pp_enabled:
                    loss, grad_norm = self.training_pp_step(inputs)
                else:
                    loss, grad_norm = self.training_step(
                        inputs, loss=loss, grad_norm=grad_norm, step=step)

                # Update state
                self.state.global_step += 1
                step = self.state.global_step

                # Step end callback (pass loss)
                self.callback_handler.on_step_end(
                    self.config, self.state, loss=loss, grad_norm=grad_norm, monitor=self.monitor,
                    metric_reduce_group=self.metric_reduce_group,
                    metric_reduce_group_size=self.metric_reduce_group_size,
                )

                # Epoch end callback (pass loss)
                if step % self.state.epoch_step == 0:
                    self.callback_handler.on_epoch_end(self.config, self.state)

                # Profiler step notification
                prof.step()

    def _create_dataset_iterator(self, dataset):
        """
        Create an iterator for the dataset using MindSpore Dataset API.

        Args:
            dataset (Dataset): The dataset instance.

        Returns:
            DatasetIterator: The dataset iterator.

        Raises:
            ValueError: If dataset is None.
            TypeError: If dataset does not support create_dict_iterator.
        """
        if dataset is None:
            raise ValueError("dataset is None, cannot create dataset iterator.")

        if not hasattr(dataset, "create_dict_iterator"):
            raise TypeError(
                f"Dataset type {type(dataset)} does not support create_dict_iterator()"
            )
        return dataset.create_dict_iterator(
            output_numpy=False, num_epochs=self.train_num_epochs
        )

    def get_batch(self, dataset_iter) -> Dict[str, Any]:
        """
        Get a batch of data from the dataset.

        Args:
            dataset_iter (DatasetIterator): Dataset iterator.

        Returns:
            Dict[str, Any]: Dictionary containing batch data.
        """
        use_distribute_dataset = getattr(self.config, "use_distribute_dataset", False)
        use_remove_redundant_dataset = getattr(
            self.config, "use_remove_redundant_dataset", False
        )

        if use_distribute_dataset:
            data = self._get_batch_distributed(dataset_iter)
        elif use_remove_redundant_dataset:
            data = self._get_batch_remove_redundant(dataset_iter)
        else:
            data = self._get_batch_naive(dataset_iter)

        # Ensure dict output
        if data is not None and not isinstance(data, dict):
            if isinstance(data, (tuple, list)):
                data = {"input_ids": data[0]} if len(data) > 0 else {}
        return data if data is not None else {}

    def _get_batch_distributed(self, dataset_iter):
        """Fetch next batch in distributed dataset mode (simplified)."""
        return next(dataset_iter)

    def _get_batch_remove_redundant(self, dataset_iter):
        """Fetch next batch in remove-redundant mode (simplified)."""
        return next(dataset_iter)

    def _get_batch_naive(self, dataset_iter):
        """Fetch next batch in naive loading mode (simplified)."""
        return next(dataset_iter)
    
    def _split_micro_batch(self, inputs: Dict[str, Any], acc_step: int):
        """
        Split a batch of data into a micro-batch for gradient accumulation.

        Args:
            inputs (Dict[str, Any]): Input data dictionary.
            acc_step (int): Current accumulation step.

        Returns:
            Dict[str, Any]: Dictionary containing micro-batch data.
        """
        local_batch_size = self.config.training.local_batch_size
        start_idx = acc_step * local_batch_size
        end_idx = (acc_step + 1) * local_batch_size

        micro_inputs = {
            k: v[start_idx:end_idx] for k, v in inputs.items()
        }
        return micro_inputs

    def training_step(self, inputs, loss, grad_norm, step):
        """
        Accumulate gradients over multiple micro-batches.
        """
        for micro_step in range(self.num_accumulation_steps):
            micro_inputs = self._split_micro_batch(inputs, micro_step)

            try:
                micro_loss = self._forward_backward(self.model, micro_inputs)
            except Exception as e:
                raise RuntimeError(f"Error in training step {step}.") from e
            if isinstance(micro_loss, DTensor):
                micro_loss = micro_loss.to_local()
            loss += micro_loss

            self.monitor.record("local_loss", micro_loss,
                                context={"micro_step": micro_step, "model": self.model})
            self.monitor.record("local_norm",
                                context={"micro_step": micro_step, "model": self.model})

            # Only perform optimizer step after gradient accumulation
            if micro_step >= self.num_accumulation_steps - 1:
                # Compute grad norm and update optimizer
                grad_norm = self._optimizer_update()

                # Loss reduction for parallel training
                if self.enable_parallel:
                    all_reduce(loss, op=ops.ReduceOp.SUM)
                    loss /= self.world_size
                self.monitor.record("device_loss", loss)
                self.monitor.record("device_norm")
        return loss, grad_norm

    def training_pp_step(self, inputs: Dict[str, Any]):
        """
        Perform a training step for pipeline parallelism.

        Args:
            inputs (Dict[str, Any]): Input data dictionary.

        Returns:
            Tuple[Tensor, Tensor]: Tuple containing loss and global norm.
        """
        loss = self.compute_pp_loss(inputs)
        global_norm = self._optimizer_update()

        if self.has_last:
            if isinstance(loss, DTensor):
                loss = loss.to_local()
            all_reduce(loss, op=ops.ReduceOp.SUM, group=self.metric_reduce_group)
            loss /= self.metric_reduce_group_size

        return loss, global_norm

    def compute_loss(self, model, inputs: Dict[str, Any]):
        """
        Compute loss for the model.

        Args:
            model (PreTrainedModel): Model instance.
            inputs (Dict[str, Any]): Input data dictionary.

        Returns:
            Tensor: Computed loss value.
        """
        # Forward pass
        outputs = model[0](**inputs)

        # Compute loss
        if self.compute_loss_func is not None:
            # Get labels from inputs
            labels = inputs.get("labels", None)

            # Use user-defined loss function
            loss = self.compute_loss_func(outputs, labels)
        else:
            # Extract loss from model output
            # We don't use .loss here since the model may return tuples instead of ModelOutput
            if isinstance(outputs, dict):
                loss = outputs["loss"]
            elif isinstance(outputs, (Tensor, DTensor)):
                loss = outputs
            else:
                # Assume first element is loss
                loss = outputs[0]

        # return value must be Tensor or DTensor
        return loss

    def compute_pp_loss(self, inputs: Dict[str, Any]):
        """
        Compute loss for the model.

        Args:
            model (PreTrainedModel): Model instance.
            inputs (Dict[str, Any]): Input data dictionary.

        Returns:
            Tensor: Computed loss value.
        """
        # Forward pass
        outputs = self.schedule.run(**inputs)

        # Compute loss
        loss = 0.0
        if self.compute_loss_func is not None:
            # Get labels from inputs
            labels = inputs.get("labels", None)

            # Use user-defined loss function
            loss = self.compute_loss_func(outputs, labels)
        else:
            # Extract loss from model output
            # We don't use .loss here since the model may return tuples instead of ModelOutput
            if isinstance(outputs, dict):
                loss = outputs["loss"]
            elif isinstance(outputs, list):
                # Assume first element is loss
                if len(outputs) == 0:
                    loss = None
                else:
                    for output in outputs:
                        if isinstance(output[0], DTensor):
                            loss += output[0].to_local()
                        else:
                            loss += output[0]
                    loss /= len(outputs)

        # return value must be Tensor or DTensor
        return loss

    def _forward_backward(self, model, inputs: Dict[str, Any]):
        """
        Compute loss and perform backward pass for a single micro-batch.
        Loss is scaled by 1/num_accumulation_steps for gradient accumulation.

        Args:
            model (PreTrainedModel): Model instance.
            inputs (Dict[str, Any]): Input data dictionary.

        Returns:
            Tensor: Unscaled loss value for reporting.
        """
        loss = self.compute_loss(model, inputs)
        sense = get_loss_sense(
            enable_parallel=self.enable_parallel,
            parallelism=self.config.parallelism,
            gradient_accumulation_steps=self.num_accumulation_steps,
            apply_gradient_accumulation=False,
        )

        if self.num_accumulation_steps > 1:
            loss = loss / self.num_accumulation_steps

        loss.backward(sense)
        return loss

    def _optimizer_update(self):
        """
        Compute gradient norm, clip gradients, update optimizer and zero gradients.

        Returns:
            Tensor: Global gradient norm.
        """
        global_norm, grads = _calculate_global_grad_norm(
            self.optimizer.parameters,
            enable_parallel=self.enable_parallel,
            max_norm=self.config.training.max_norm,
            parallelism=self.config.parallelism,
        )

        with _no_grad():
            self.optimizer(grads)

        # zero grad
        for m in self.model:
            if hasattr(m, "zero_grad"):
                m.zero_grad()
            else:
                for param in m.trainable_params():
                    param.grad = None

        return global_norm
