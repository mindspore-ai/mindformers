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

from mindspore import mint, manual_seed, value_and_grad, set_deterministic
from mindspore.dataset import Dataset
from mindspore.common import set_seed
from mindspore.mint.distributed import (
    init_process_group,
    destroy_process_group,
    get_world_size,
)
from mindspore.nn.utils import no_init_parameters
from mindspore.ops import clip_by_global_norm

try:
    from hyper_parallel import parallelize_value_and_grad
except ImportError:
    parallelize_value_and_grad = None

from mindformers.tools.logger import logger
from mindformers.pynative.config import TrainConfig
from mindformers.models import PreTrainedModel
from mindformers.checkpoint.checkpoint import load_checkpoint
from mindformers.pynative.callback import (
    CallbackHandler,
    TrainerCallback,
    LossCallback,
    CheckpointCallback,
)

from mindformers.checkpoint.checkpoint import CommonInfo, get_checkpoint_path

from .train_state import TrainerState
from .utils import (
    _build_model,
    _build_dataset,
    _build_optimizer,
    _build_lr_scheduler,
    _build_callback,
    _build_lora_model,
    compute_parameters,
)


class TrainMode(enum.Enum):
    """Training mode enumeration."""

    FINETUNE = "finetune"
    PRETRAIN = "pretrain"


class Trainer:
    """
    Trainer for training models in MindFormers.
    """

    def __init__(
        self,
        config: Union[str, dict] = None,
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

        # Initialize config
        self.config = self._init_config(config, run_mode)

        self.world_size = get_world_size()
        logger.info(f"Current world size: {self.world_size}.")
        self.use_parallel = self.world_size > 1
        if self.use_parallel:
            raise ValueError("Parallel training is not supported yet.")

        self._setup_seed_and_determinism()

        self.global_batch_size = self.config.training.global_batch_size
        self.gradient_accumulation_steps = None
        self._compute_data_parallel_size()

        # Create model
        self.model = self._create_model(model, self.config.model)

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

        # Initialize training state
        self.communication_init = False
        self.state = None

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
        with no_init_parameters():
            model = _build_model(model_config)

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
        self.config.parallelism.data_parallel = data_parallel
        logger.info(
            f"Got parallelism settings: {data_parallel=}, {tensor_parallel=}, "
            f"{pipeline_parallel=}, {context_parallel=}"
        )

        # calculate gradient accumulation steps
        if (
            data_parallel * self.config.training.local_batch_size
            > self.global_batch_size
        ):
            raise ValueError(
                "The product of data_parallel and local_batch_size exceeds global_batch_size, "
                "please increase global_batch_size or decrease local_batch_size."
            )

        self.num_accumulation_steps = self.global_batch_size // (
            data_parallel * self.config.training.local_batch_size
        )
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
            self.config.parallelism.data_parallel,
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

    def _create_build_in_callbacks(self) -> List[TrainerCallback]:
        """
        Create default callbacks.
        """
        # build checkpoint callback
        checkpoint = self.config.checkpoint
        checkpoint_callback = CheckpointCallback(
            save_path=checkpoint.save_path,
            save_max=checkpoint.save_max,
            save_interleaved_steps=checkpoint.save_interleaved_steps,
            no_save_optim=checkpoint.no_save_optim,
            async_save=checkpoint.async_save,
            prefix=checkpoint.prefix,
            remove_redundancy=checkpoint.remove_redundancy,
        )

        return [
            LossCallback(),
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
        callback_list = self._create_build_in_callbacks()
        if callbacks:
            callback_list.extend(callbacks)

        # Build callbacks from config and extend
        callback_config = getattr(config, "callbacks", [])
        for callback in callback_config:
            callback_list.append(_build_callback(callback))

        # Create handler with complete list
        cb_handler = CallbackHandler(
            callbacks=callback_list,
            model=self.model,
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
            is_train_begin=True,
            is_train_end=False,
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

        # MindSpore communication setting
        if self.use_parallel:
            init_process_group()
            self.communication_init = True
            logger.info("Distributed communication is initialized.")

        # Initialize training state
        self.state = self._init_train_state()

        # Initialize parallel config and wrappers
        if self.use_parallel:
            self._init_parallel_config(self.model, self.config.parallelism)

        # Load checkpoint
        checkpoint_path = checkpoint_path or self.config.checkpoint.load_path
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path, self.model, self.optimizer)

        # Call train begin callback
        self.callback_handler.on_train_begin(self.config, self.state)

        # Execute training loop
        self._inner_train_loop()

        # Call train end callback
        self.callback_handler.on_train_end(self.config, self.state)

        if self.communication_init:
            destroy_process_group()

    def _init_parallel_config(self, model, parallelism):
        """Initialize parallel configuration."""
        _, _ = model, parallelism
        raise ValueError("Parallel training is not supported yet.")

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
            optimizer=optimizer,
            global_step=global_step,
            balanced_load=checkpoint.load_balanced,
        )

    def _inner_train_loop(self):
        """
        Internal training loop.
        """
        # Create dataset iterator
        dataset_iter = self._create_dataset_iterator(self.train_dataset)

        if self.use_parallel:
            grad_fn = parallelize_value_and_grad(
                self.compute_loss, self.optimizer.parameters
            )

        else:
            grad_fn = value_and_grad(
                self.compute_loss, None, self.optimizer.parameters, has_aux=False
            )

        if hasattr(self.model, "set_train"):
            self.model.set_train(True)

        # Training loop
        logger.info("Start training loop...")
        step = self.state.global_step
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

            # Training step
            try:
                loss, grad_norm = self.training_step(self.model, inputs, grad_fn)
            except Exception as e:
                raise RuntimeError(f"Error in training step {step}.") from e

            # Update state
            self.state.global_step += 1
            step = self.state.global_step

            # Step end callback (pass loss)
            self.callback_handler.on_step_end(
                self.config, self.state, loss=loss, grad_norm=grad_norm
            )

            # Epoch end callback (pass loss)
            if step % self.state.epoch_step == 0:
                self.callback_handler.on_epoch_end(self.config, self.state)

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
        outputs = model(**inputs)

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
            else:
                # Assume first element is loss
                loss = outputs[0]

        # return value must be Tensor or DTensor
        return loss

    def training_step(self, model, inputs: Dict[str, Any], grad_fn: Callable):
        """
        Perform a single training step.

        Args:
            model (PreTrainedModel): Model instance.
            inputs (Dict[str, Any]): Input data dictionary.
            grad_fn (Callable): Gradient computation function.

        Returns:
            tuple: Tuple containing loss value and gradient norm (loss, grad_norm).
        """
        # Forward and compute loss
        loss, grads = grad_fn(model, inputs)

        # Gradient norm calculation
        overflow = False
        grad_norm = 0.0
        for grad in grads:
            if grad.isinf().any() or grad.isnan().any():
                overflow = True
                break
            grad_norm += mint.sum(mint.square(grad))

        if overflow:
            return loss, None
        grad_norm = mint.sqrt(grad_norm)

        # Clip gradients
        grads = clip_by_global_norm(grads)

        # Optimizer step
        self.optimizer(grads)

        return loss, grad_norm
