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
from mindspore import Tensor, ops, manual_seed, set_deterministic
from mindspore.dataset import Dataset
from mindspore.common import set_seed
from mindspore.mint.distributed import (
    init_process_group,
    destroy_process_group,
    get_world_size,
    get_rank,
    all_reduce,
    new_group,
    broadcast_object_list,
)
from mindspore.graph.api import _no_grad
from hyper_parallel.platform.mindspore.pipeline_parallel._utils import _MicroBatch
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat

from mindformers.tools.logger import logger
from mindformers.pynative.config import TrainConfig
from mindformers.pynative.trainer.dynamic_batch import build_dynamic_scheduler
from mindformers.models import PreTrainedModel
from mindformers.checkpoint.checkpoint import load_checkpoint, load_hf_checkpoint
from mindformers.checkpoint.utils import is_hf_checkpoint
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
from mindformers.pynative.pet.lora_adapter import freeze_base_params

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
    set_auxiliary_loss_backward_scale,
    _sync_mtp_embedding_weights_after_init,
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

        self.communication_init = False
        if self.enable_parallel:
            init_process_group()
            self.communication_init = True
            logger.info("Distributed communication is initialized.")

        # init states for pipeline parallel
        self.schedule = None
        self.metric_reduce_group = None
        self.metric_reduce_group_size = None
        self.pp_metric_reduce_group = None
        self.pp_metric_reduce_group_size = None
        # On the single-card / non-PP path there is no pipeline split, so the
        # only stage is also the last stage. ``_apply_parallelism`` overwrites
        # these when PP is enabled.
        self.has_first = True
        self.has_last = True

        self.global_batch_size = self.config.training.global_batch_size
        self.dynamic_scheduler = None
        self._dataset_iter = None
        # consumed_samples: total samples processed so far.
        self._consumed_samples = 0
        self._base_units = 0  # data_parallel * local_batch_size
        # True when resuming (set in _load_checkpoint). On epoch-wrap iterator
        # recreate, a resumed DYNAMIC run resets the dataset offset to 0
        # (set_init_step(0)) so the new epoch starts at item 0 — otherwise the
        # persisted resume offset makes the recreated iterator restart at the
        # checkpoint position and diverge from fresh training after the first epoch.
        self._resumed = False
        rampup = getattr(self.config.training, "rampup_batch_size", None)
        self.dynamic_batch_enabled = rampup is not None
        self._compute_data_parallel_size()

        # init data broadcast group
        self.enable_data_broadcast = getattr(self.config.train_dataset, "use_distribute_dataset", False)
        self._init_distributed_dataset_group()

        # Create model
        self.model = self._create_model(model, self.config.model)
        self.parallel_dims = None
        if self.enable_parallel:
            # Apply parallelism to model
            self._apply_parallelism(self.model, self.config.parallelism)

        # Set the backward scale for auxiliary losses (aux/mtp/index) once, for
        # every code path. These losses inject their gradient through dedicated
        # autoscalers (not the main ``loss.backward``), so the scale must fold in
        # ``1 / num_accumulation_steps`` to stay aligned with the main loss. The
        # single-card path skips ``parallelize_model`` entirely, so without this
        # the autoscalers keep their default ``1.0`` and gradient accumulation
        # diverges from the no-accumulation run.
        set_auxiliary_loss_backward_scale(
            parallelism=self.config.parallelism,
            enable_parallel=self.enable_parallel,
            gradient_accumulation_steps=self.num_accumulation_steps,
            indexer_loss_tp_replica_size=(
                self.config.parallelism.tensor_parallel
                if getattr(self.config.model, "experimental_attention_variant", None) == "dsa"
                else 1
            ),
        )

        # After parallelism, parameters will be reset
        self.model = self.model if isinstance(self.model, list) else [self.model]
        for m in self.model:
            m.to_empty()
            with _no_grad():
                m.init_states()

        # PP rebuilds separate input-embedding copies for the first stage and
        # the MTP stage. Delayed initialization above gives those replicas
        # different values even though their gradients are synchronized and
        # checkpoint metadata treats them as one shared parameter. Canonicalize
        # the initialized local shards before optimizer states are derived.
        with _no_grad():
            _sync_mtp_embedding_weights_after_init(self.model)

        # LoRA: freeze base params AFTER parallelism + init_states. During distribute_module
        # all params must be trainable so they materialise as DTensors (including frozen
        # norms/biases and NoParallel base weights, whose TP layout inference needs DTensor
        # gamma); freezing earlier leaves them as plain Tensors and breaks TP. Only
        # lora_a/lora_b stay trainable, so the optimizer (built below) sees adapters only.
        if getattr(self.config, "lora_config", None) is not None:
            # Under pipeline parallelism freeze runs per stage; a stage may legitimately hold
            # no adapters, so aggregate the trainable count and only error if NO stage has any.
            total_trainable = 0
            for m in self.model:
                trainable, _ = freeze_base_params(m)
                total_trainable += trainable
            if total_trainable == 0:
                raise RuntimeError(
                    "LoRA enabled but no trainable adapters were found across any pipeline "
                    "stage. Check lora_config.target_modules against the model's module names."
                )
            compute_parameters(self.model[0])

        # Create train dataset
        self.train_dataset = self._create_dataset(
            train_dataset, getattr(self.config, "train_dataset", None)
        )
        if self.train_dataset is not None:
            self.train_epoch_step = self._get_dataset_size(self.train_dataset)
        else:
            self.train_epoch_step = self.config.training.steps
        if self.dynamic_batch_enabled and self.dynamic_scheduler is not None:
            # Dynamic batch consumes a varying (growing) number of micro-batches per
            # optimizer step. Size num_epochs to a safe UPPER BOUND (max_steps *
            # max_accum, +1 epoch margin) so the iterator never exhausts mid-run.
            # A mid-run StopIteration would force an iterator recreate that restarts
            # the data stream and diverge fresh vs resumed training (the data
            # position is driven by consumed_samples on the trainer, not by the
            # iterator's internal offset). The bound is safe because num_epochs only
            # caps capacity (the iterator is lazy); the loop still stops at max_steps,
            # and consumed + remaining <= max_steps * max_accum <= (num_epochs-1)*E.
            max_accum = self.dynamic_scheduler.max_gbs // self._base_units
            max_total_micros = self.config.training.steps * max_accum
            self.train_num_epochs = max(
                (max_total_micros + self.train_epoch_step - 1)
                // self.train_epoch_step + 1, 1)
        else:
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
        self.monitor = MonitorGroup(self.config, model=self.model)

        # Initialize training state
        self.state = None

    def _init_distributed_dataset_group(self):
        """Initialize per-data-shard batch broadcast group."""
        self.data_broadcast_group = None
        self.data_broadcast_src_rank = None
        self.data_broadcast_group_ranks = None

        if not self.enable_data_broadcast:
            return

        parallelism = self.config.parallelism
        data_domain_size = parallelism.context_parallel * parallelism.tensor_parallel
        if self.world_size <= 1 or data_domain_size <= 1:
            logger.info(
                "use_distribute_dataset is enabled but no cross-rank data domain exists; "
                "fall back to normal dataset iteration."
            )
            self.enable_data_broadcast = False
            return

        if not self.enable_parallel:
            raise RuntimeError("Data broadcast requires distributed communication.")

        pp_domain_size = self.world_size // parallelism.pipeline_parallel
        rank = get_rank()
        pp_rank = rank // pp_domain_size
        rank_in_pp = rank % pp_domain_size
        data_domain_id = rank_in_pp // data_domain_size
        start = pp_rank * pp_domain_size + data_domain_id * data_domain_size
        self.data_broadcast_group_ranks = list(range(start, start + data_domain_size))
        self.data_broadcast_src_rank = self.data_broadcast_group_ranks[0]
        self.data_broadcast_group = new_group(self.data_broadcast_group_ranks)
        if not self.data_broadcast_group:
            raise RuntimeError(
                f"Failed to create data broadcast group for ranks "
                f"{self.data_broadcast_group_ranks}."
            )
        logger.info(
            "Distributed dataset broadcast is enabled: group ranks=%s, src_rank=%s.",
            self.data_broadcast_group_ranks,
            self.data_broadcast_src_rank,
        )

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

        # Build ParallelDims (dp_replicate / dp_shard derived inside from_config)
        self.parallel_dims = ParallelDims.from_config(parallelism, self.world_size)

        # Apply unified parallelization
        self.model, self.schedule, self.has_first, self.has_last = parallelize_model(
            model=model,
            parallel_dims=self.parallel_dims,
            parallelism=parallelism,
            recompute=self.config.recompute,
            recompute_comm=self.config.recompute_comm,
            swap=self.config.swap,
            gradient_accumulation_steps=self.num_accumulation_steps,
        )

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
        base_units = parallelism.data_parallel * self.config.training.local_batch_size
        self._base_units = base_units
        if self.dynamic_batch_enabled:
            # Dynamic batch : the scheduler maps
            # consumed_samples -> GBS. Initial values come from consumed_samples=0.
            self.dynamic_scheduler = build_dynamic_scheduler(self.config, self.global_batch_size, base_units)
            self.num_accumulation_steps = self.dynamic_scheduler.num_accum_at_consumed(0)
            self.global_batch_size = self.dynamic_scheduler.gbs_at_consumed(0)
            logger.info(
                "Dynamic batch enabled: initial global_batch_size=%d, "
                "num_accumulation_steps=%d.",
                self.global_batch_size, self.num_accumulation_steps,
            )
        else:
            if base_units > self.global_batch_size:
                raise ValueError(
                    "The product of data_parallel and local_batch_size exceeds global_batch_size, "
                    "please increase global_batch_size or decrease local_batch_size."
                )
            self.num_accumulation_steps = self.global_batch_size // base_units
            logger.info(
                f"Calculate global_batch_size={self.global_batch_size}, "
                f"num_accumulation_steps={self.num_accumulation_steps}."
            )
        parallelism.pipeline_parallel_microbatch_size = self.num_accumulation_steps

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
            self.config.parallelism,
            local_batch_size=self.config.training.local_batch_size,
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
            return [MonitorCallback(), LossCallback()]

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
            MonitorCallback(),
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
            num_accumulation_steps=self.num_accumulation_steps,
            is_train_begin=True,
            is_train_end=False,
        )

    def _register_grad_hooks(self):
        """Register backward hooks to cast low-precision gradients to fp32 (single-card only).

        bf16/fp16 parameter gradients are cast to fp32 during backward so that
        gradient accumulation operates in fp32 precision.

        Multi-card runs go through ``fully_shard`` instead, where HSDP's
        ``apply_grad_on_fp32_main_grad`` policy accumulates the reduced gradient onto
        a fp32 ``param.main_grad`` buffer, so these hooks are unnecessary there.
        """
        if self.enable_parallel:
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
        # Sync dynamic fields that were computed before state existed.
        self.state.num_accumulation_steps = self.num_accumulation_steps
        self.state.global_batch_size = self.global_batch_size

        if self.parallel_dims is not None and self.parallel_dims.pp_enabled:
            pp_mesh = self.parallel_dims.get_mesh("pp")
            self.pp_metric_reduce_group_size = pp_mesh.size()
            self.pp_metric_reduce_group = pp_mesh.get_group()

        # ``loss_mesh`` is the dp x cp domain and is the right group to all-reduce
        # per-step metric statistics over (loss, MoE aux loss, MoE tokens_per_expert
        # for the auxiliary-loss-free expert_bias update). It must be set even when
        # PP is disabled, otherwise multi-card DP / CP runs would compute the
        # expert_bias delta from per-rank local token counts instead of the
        # global-batch counts, diverging from the single-card baseline.
        if self.parallel_dims is not None and (
                self.parallel_dims.pp_enabled or self.parallel_dims.dp_cp_enabled):
            loss_mesh = self.parallel_dims.get_mesh("loss")
            self.metric_reduce_group_size = loss_mesh.size()
            self.metric_reduce_group = loss_mesh.get_group()

        # Load checkpoint
        checkpoint_path = checkpoint_path or self.config.checkpoint.load_path
        if checkpoint_path:
            for m in self.model:
                self._load_checkpoint(checkpoint_path, m, self.optimizer)

        # Register gradient hooks for fp32 accumulation
        self._register_grad_hooks()

        self.monitor.setup(self.config, self.state)

        # Call train begin callback
        self.callback_handler.on_train_begin(self.config, self.state)

        # Execute training loop
        self._inner_train_loop()

        # Call train end callback
        self.callback_handler.on_train_end(self.config, self.state)

        self.monitor.close()

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
        checkpoint = self.config.checkpoint

        if is_hf_checkpoint(checkpoint_path):
            load_hf_checkpoint(
                pretrained_model_dir=checkpoint_path,
                network=model,
                balanced_load=checkpoint.load_balanced,
                reshard_worker_num=checkpoint.reshard_worker_num
            )
        else:
            common_file = os.path.join(checkpoint_path, "common.json")
            common_info = CommonInfo.load_common(common_file)

            checkpoint = self.config.checkpoint

            if not checkpoint.no_load_optim:
                if optimizer is None:
                    raise ValueError("If no_load_optim is False, optimizer is required.")

                global_step = common_info.global_step
                self._resumed = True
                if self.dynamic_batch_enabled:
                    # Dynamic: skip consumed_samples // base_units micro-batches. Use the
                    # persisted consumed_samples; fall back to recomputing from global_step
                    # for old checkpoints without the field.
                    saved_consumed = getattr(common_info, "consumed_samples", None)
                    if saved_consumed is not None:
                        self._consumed_samples = int(saved_consumed)
                    else:
                        self._consumed_samples = self.dynamic_scheduler.consumed_samples_at_step(global_step)
                    self.state.consumed_samples = self._consumed_samples
                    skip_micros = self._consumed_samples // self._base_units
                    if skip_micros > 0:
                        self.train_dataset.set_init_step(skip_micros)
                    logger.info(
                        "Resume dataset (dynamic batch): skip %d micro-batches "
                        "(consumed_samples=%d, global_step=%d).",
                        skip_micros, self._consumed_samples, global_step,
                    )
                else:
                    # Static: set_init_step by global_step, not consumed_samples.
                    if common_info.global_batch_size and common_info.global_batch_size != self.global_batch_size:
                        global_step = int(
                            common_info.global_step
                            * (common_info.global_batch_size / self.global_batch_size)
                        )
                        logger.info(
                            f"Scaled global step: {common_info.global_step} -> {global_step} "
                            f"(batch size changed from {common_info.global_batch_size} to {self.global_batch_size})"
                        )
                    self.train_dataset.set_init_step(global_step)
                    self._consumed_samples = global_step * self.global_batch_size
                    self.state.consumed_samples = self._consumed_samples
                    logger.info(f"Resume dataset (static batch) from: {global_step}")

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
        # Create dataset iterator. On resume, set_init_step(skip) (called in
        # _load_checkpoint) makes this first iterator start at the resume position.
        # Epoch-wrap recreates (in _next_batch) reset the offset to 0 so each new
        # epoch starts at its first item, matching fresh training.
        self._dataset_iter = self._create_dataset_iterator(self.train_dataset)

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

                # Dynamic batch (sample-driven): refresh num_accumulation_steps and
                # derived items from consumed_samples BEFORE running this step, so the
                # step consumes the right number of micro-batches. No-op when the
                # accumulation value is unchanged.
                if self.dynamic_batch_enabled:
                    self._update_dynamic_batch(self._consumed_samples)

                # Step begin callback
                self.callback_handler.on_step_begin(self.config, self.state)

                # Gradient accumulation: accumulate over micro-batches
                loss = 0.0
                grad_norm = 0.0
                self.monitor.reset()
                self.monitor.record("moe_tpe_step_begin")

                if self.parallel_dims and self.parallel_dims.pp_enabled:
                    loss, grad_norm = self.training_pp_step()
                else:
                    loss, grad_norm = self.training_step(step=step)

                # Advance consumed_samples by this step's GBS (dynamic drives the
                # schedule with it; static uses it for logging). Persisted via
                # state.consumed_samples for dynamic-batch resume.
                self._consumed_samples += self.global_batch_size
                self.state.consumed_samples = self._consumed_samples

                # Update state
                self.state.global_step += 1
                step = self.state.global_step

                # Step end callback (pass loss)
                self.callback_handler.on_step_end(
                    self.config, self.state, loss=loss, grad_norm=grad_norm, monitor=self.monitor,
                    metric_reduce_group=self.metric_reduce_group,
                    metric_reduce_group_size=self.metric_reduce_group_size,
                    pp_metric_reduce_group=self.pp_metric_reduce_group,
                    pp_metric_reduce_group_size=self.pp_metric_reduce_group_size,
                    has_last=self.has_last,
                )
                self.monitor.flush(step)

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
        use_remove_redundant_dataset = getattr(
            self.config, "use_remove_redundant_dataset", False
        )

        if self.enable_data_broadcast:
            data = self._get_batch_data_broadcast(dataset_iter)
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

    def _get_batch_data_broadcast(self, dataset_iter):
        """Fetch a batch on the source rank and broadcast it within the data domain."""
        rank = get_rank()
        if rank == self.data_broadcast_src_rank:
            try:
                data = next(dataset_iter)
            except StopIteration:
                obj = [{"end": True, "data": None}]
                broadcast_object_list(
                    obj,
                    src=self.data_broadcast_src_rank,
                    group=self.data_broadcast_group,
                )
                raise
            obj = [{"end": False, "data": self._pack_broadcast_data(data)}]
        else:
            obj = [None]

        broadcast_object_list(
            obj,
            src=self.data_broadcast_src_rank,
            group=self.data_broadcast_group,
        )
        payload = obj[0]
        if payload["end"]:
            raise StopIteration
        return self._unpack_broadcast_data(payload["data"])

    @classmethod
    def _pack_broadcast_data(cls, data):
        """Convert tensors in a nested batch object into pickle-friendly arrays."""
        if isinstance(data, Tensor):
            return {"__data_broadcast_type__": "tensor", "value": data.asnumpy()}
        if isinstance(data, np.ndarray):
            return {"__data_broadcast_type__": "ndarray", "value": data}
        if isinstance(data, dict):
            return {k: cls._pack_broadcast_data(v) for k, v in data.items()}
        if isinstance(data, tuple):
            return {
                "__data_broadcast_type__": "tuple",
                "value": [cls._pack_broadcast_data(v) for v in data],
            }
        if isinstance(data, list):
            return [cls._pack_broadcast_data(v) for v in data]
        return data

    @classmethod
    def _unpack_broadcast_data(cls, data):
        """Restore a nested broadcast batch object for model execution."""
        if isinstance(data, dict):
            data_type = data.get("__data_broadcast_type__")
            if data_type in ("tensor", "ndarray"):
                return Tensor(data["value"])
            if data_type == "tuple":
                return tuple(cls._unpack_broadcast_data(v) for v in data["value"])
            return {k: cls._unpack_broadcast_data(v) for k, v in data.items()}
        if isinstance(data, list):
            return [cls._unpack_broadcast_data(v) for v in data]
        return data

    def training_step(self, step=None):
        """Accumulate gradients over micro-batches.

        Each micro-batch is fetched individually via ``_next_batch`` at
        ``local_batch_size`` granularity. This path is shared by both static
        (fixed ``num_accumulation_steps``) and dynamic (varying
        ``num_accumulation_steps``) batch training.

        Args:
            step (int, optional): Current optimizer step for error reporting.

        Returns:
            Tuple[Tensor, Tensor]: Accumulated loss and global gradient norm.
        """
        loss = 0.0
        grad_norm = 0.0
        for micro_step in range(self.num_accumulation_steps):
            micro_inputs = self._next_batch()

            try:
                micro_loss = self._forward_backward(self.model, micro_inputs)
            except Exception as e:
                raise RuntimeError(f"Error in training step {step}.") from e
            if isinstance(micro_loss, DTensor):
                micro_loss = micro_loss.to_local()
            loss += micro_loss

            self.monitor.record("local_loss", micro_loss,
                                context={"micro_step": micro_step, "model": self.model})

            # Only perform optimizer step after gradient accumulation
            if micro_step >= self.num_accumulation_steps - 1:
                # Compute grad norm and update optimizer
                grad_norm = self._optimizer_update()

                # Loss reduction for parallel training
                if self.enable_parallel:
                    all_reduce(loss, op=ops.ReduceOp.SUM)
                    loss /= self.world_size
                if self.monitor.should_record("device_loss"):
                    self.monitor.record("device_loss", loss)
                if self.monitor.should_record("device_norm"):
                    self.monitor.record("device_norm")
        return loss, grad_norm

    def training_pp_step(self):
        """Perform a training step for pipeline parallelism.

        Micro-batches are collected at ``local_batch_size`` granularity and
        concatenated along dim 0 for the PP schedule to split equally by
        ``micro_batch_num``. This path is shared by both static and dynamic
        batch training.

        Returns:
            Tuple[Tensor, Tensor]: Tuple containing loss and global norm.
        """
        inputs = self._collect_micro_batches(self.num_accumulation_steps)
        loss = self.compute_pp_loss(inputs)
        global_norm = self._optimizer_update()

        if self.has_last:
            if isinstance(loss, DTensor):
                loss = loss.to_local()
            all_reduce(loss, op=ops.ReduceOp.SUM, group=self.metric_reduce_group)
            loss /= self.metric_reduce_group_size

        return loss, global_norm


    def _next_batch(self) -> Dict[str, Any]:
        """Fetch the next micro-batch; on exhaustion (epoch wrap) recreate the
        iterator and retry."""
        try:
            return self.get_batch(self._dataset_iter)
        except StopIteration:
            if self._resumed and self.dynamic_batch_enabled:
                self.train_dataset.set_init_step(0)
            self._dataset_iter = self._create_dataset_iterator(self.train_dataset)
            return self.get_batch(self._dataset_iter)

    def _update_dynamic_batch(self, consumed_samples: int) -> None:
        """Refresh ``num_accumulation_steps`` and derived items per the scheduler for
        the current ``consumed_samples``. No-op when the value is unchanged; otherwise
        syncs state, PP microbatch size, aux-loss scale, and (for PP) the schedule."""
        new_accum = self.dynamic_scheduler.num_accum_at_consumed(consumed_samples)
        if new_accum == self.num_accumulation_steps:
            return
        new_gbs = self.dynamic_scheduler.gbs_at_consumed(consumed_samples)
        self.num_accumulation_steps = new_accum
        self.global_batch_size = new_gbs
        # Sync to state so callbacks (LossCallback etc.) read the current
        # per-step micro-batch count for aux / MTP / indexer loss division.
        self.state.num_accumulation_steps = new_accum
        self.state.global_batch_size = new_gbs
        self.config.parallelism.pipeline_parallel_microbatch_size = new_accum
        # Auxiliary loss (MoE/MTP/indexer) backward scaling includes
        # 1/num_accumulation_steps and must be refreshed synchronously.
        set_auxiliary_loss_backward_scale(
            parallelism=self.config.parallelism,
            enable_parallel=self.enable_parallel,
            gradient_accumulation_steps=new_accum,
        )
        if self.parallel_dims and self.parallel_dims.pp_enabled:
            self._update_pipeline_microbatch(new_accum)

    def _update_pipeline_microbatch(self, new_num_accum: int) -> None:
        """Update schedule micro_batch_num at runtime and rebuild exec_order for PP.

        v1 supports dynamic batch under PP: update ``micro_batch_num`` and
        ``split_micro_batch``, call ``build_exec_order`` to rebuild the 1F1B
        schedule (overlap_b_f callbacks stored in ``_custom_fn_map`` are preserved
        across rebuilds), and refresh ``loss_scale`` on every stage (which embeds
        ``1/num_accumulation_steps``).

        Args:
            new_num_accum (int): New gradient accumulation steps (== PP micro_batch_num).
        """
        sched = self.schedule
        if sched is None:
            return
        sched.micro_batch_num = new_num_accum
        # Rebuild split_micro_batch using the same platform micro_batch factory
        # that was used when the schedule was constructed.
        # Must fail-fast on error: if split_micro_batch stays at the old count
        # while micro_batch_num/exec_order have been updated, the mismatch between
        # split_microbatches and run_microbatches counts will cause silent data
        # corruption.
        try:
            sched.split_micro_batch = _MicroBatch(
                new_num_accum, sched._args_batch_dim, sched._kwargs_batch_dim)
        except Exception as e:
            raise RuntimeError(
                f"Rebuild split_micro_batch for micro_batch_num={new_num_accum} failed; "
                f"aborting to avoid split/schedule mismatch."
            ) from e
        sched.build_exec_order()
        new_loss_scale = float(get_loss_sense(
            parallelism=self.config.parallelism,
            enable_parallel=self.enable_parallel,
            gradient_accumulation_steps=new_num_accum,
            apply_gradient_accumulation=True,
        ).asnumpy().item())
        for stage in sched.stages:
            stage.loss_scale = new_loss_scale
        logger.info("Pipeline schedule rebuilt for micro_batch_num=%d.", new_num_accum)

    def _collect_micro_batches(self, num_micro_batches: int) -> Dict[str, Any]:
        """Fetch ``num_micro_batches`` micro-batches and concatenate along dim 0 into a single batch.

        After stacking, batch dim = ``num_micro_batches * local_batch_size`` == ``micro_batch_num * local_batch_size``,
        which the PP ``schedule.run`` then splits equally by ``micro_batch_num``.

        Args:
            num_micro_batches (int): Number of micro-batches (== current ``num_accumulation_steps``).

        Returns:
            Dict[str, Any]: Concatenated input dict.
        """
        batches = [self._next_batch() for _ in range(num_micro_batches)]
        if not batches:
            return {}
        stacked = {}
        for key in batches[0]:
            values = [b[key] for b in batches]
            first = values[0]
            # Use concat along dim 0 (not stack: stack adds a new dim, turning
            # (lbs, seq) into (n, lbs, seq), which breaks the schedule's
            # equal-split of dim 0 by micro_batch_num).
            if isinstance(first, Tensor):
                stacked[key] = ops.concat(values, 0)
            elif isinstance(first, np.ndarray):
                stacked[key] = np.concatenate(values, axis=0)
            else:
                stacked[key] = values
        return stacked

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
