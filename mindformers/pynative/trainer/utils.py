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
"""Pynative Trainer Utils."""

from functools import partial
from typing import List, Tuple, Union
import numpy as np

from hyper_parallel.core.dtensor.dtensor import DTensor

from mindspore import nn, Tensor, mint
from mindspore.common import dtype as mstype
import mindspore.dataset as ms_dataset
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore.mint.distributed import (
    get_world_size,
    get_rank,
)

from mindformers.tools.logger import logger
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindformers.dataset.dataloader.blended_megatron_dataloader import (
    BlendedMegatronDatasetDataLoader,
)
from mindformers.dataset.dataloader.hf_dataloader import HFDataLoader
from mindformers.dataset.dataloader.utils import _get_mindrecord_files
from mindformers.pynative.optimizer.adamw import AdamW
from mindformers.pynative.optimizer import Muon
from mindformers.pynative.distributed.context_parallel import attach_context_parallel_runtime_hints


def compute_parameters(model: nn.Cell) -> None:
    """
    Compute and log the number of total and trainable parameters in the model.

    Args:
        model: MindSpore model instance.

    Note:
        - Parameter size is computed by multiplying all dimensions of each tensor.
        - The result is logged in million (M) units.
    """
    model_params = model.get_parameters()
    total_params = []
    trainable_params = []
    for param in model_params:
        param_shape = np.prod(param.shape)
        total_params.append(param_shape)
        if param.requires_grad:
            trainable_params.append(param_shape)
    logger.info(
        f"Got total parameters: {sum(total_params) // 1.e6}M, "
        f"trainable parameters: {sum(trainable_params) // 1.e6}M"
    )


def _get_no_wd_params(model: nn.Cell) -> Tuple[Union[dict, list], dict]:
    """
    Get parameters or parameter name keywords that should NOT apply weight decay.

    This function queries optional model interfaces:
        - model.no_weight_decay()
        - model.no_weight_decay_keywords()

    Args:
        model: MindSpore model instance.

    Returns:
        Tuple[Union[dict, list], dict]:
        - no_wd_params: Explicit parameter names without weight decay, can be a dict or a list.
        - no_wd_keywords: Keywords used to match parameter names.
    """
    no_wd_params = {}
    no_wd_keywords = {}

    if hasattr(model, "no_weight_decay"):
        no_wd_params = model.no_weight_decay()
        logger.info(f"Get no weight decay params: {no_wd_params}")

    if hasattr(model, "no_weight_decay_keywords"):
        no_wd_keywords = model.no_weight_decay_keywords()
        logger.info(f"Get no weight decay keywords: {no_wd_keywords}")

    return no_wd_params, no_wd_keywords


def get_param_groups(model: nn.Cell, weight_decay: float = 0.0) -> List[dict]:
    """
    Create optimizer parameter groups.

    Weight decay will be disabled for:
        - 1D parameters (e.g., LayerNorm / bias)
        - Bias parameters
        - Parameters explicitly specified by model
        - Parameters matched by no-weight-decay keywords

    Args:
        model: MindSpore model instance.
        weight_decay (float): Base weight decay value.

    Returns:
        List[dict]: Parameter groups compatible with MindSpore optimizers.
    """
    # Get no weight decay params and keywords from model interface
    no_wd_params, no_wd_keywords = _get_no_wd_params(model)

    def _is_keywords_in_name(p_name):
        """Check whether parameter name matches any no-wd keyword."""
        for keyword in no_wd_keywords:
            if keyword in p_name:
                return True
        return False

    def _init_param_group(wd_value, wd_factor, lr=None):
        """Initialize a parameter group."""
        init_group = {"weight_decay": wd_value * wd_factor, "params": []}
        if lr is not None:
            init_group["lr"] = lr
        return init_group

    # Iterate over trainable parameters and assign them to groups
    logger.info("Assigning parameters to groups...")
    parameter_groups = {}

    for param in model.trainable_params():
        param_name = param.name

        no_wd = (
            len(param.shape) == 1
            or param_name.endswith(".bias")
            or param_name in no_wd_params
            or _is_keywords_in_name(param_name)
        )
        if no_wd:
            wd_mul = 0.0
            group_name = "no_weight_decay"
        else:
            wd_mul = 1.0
            group_name = "weight_decay"

        # Initialize group if not exists
        if group_name not in parameter_groups:
            parameter_groups[group_name] = _init_param_group(weight_decay, wd_mul)

        # Append parameter to its group
        parameter_groups[group_name]["params"].append(param)
        logger.info(
            f"param_name: {param_name:<60} | weight_decay: "
            f"{parameter_groups[group_name]['weight_decay']:>8.4f}"
        )
    return list(parameter_groups.values())


def _sync_eod_compression_flag(dataset_config, model_config):
    """Sync use_eod_attn_mask_compression from dataset config to model config.

    The non-pynative trainer does this in ``_get_columns_with_strategy``.
    The pynative trainer must do it before model construction so that
    ``TransformerConfig.__post_init__`` picks the correct input_layout.
    """
    if dataset_config is None:
        return
    dataloader_cfg = getattr(dataset_config, "dataloader", None)
    if dataloader_cfg is None:
        return
    inner_cfg = getattr(dataloader_cfg, "config", None)
    if inner_cfg is None:
        return
    create_compressed = False
    if isinstance(inner_cfg, dict):
        create_compressed = inner_cfg.get("create_compressed_eod_mask", False)
    else:
        create_compressed = getattr(inner_cfg, "create_compressed_eod_mask", False)
    if create_compressed:
        model_config.use_eod_attn_mask_compression = True


def _build_model(config, parallelism_config=None, dataset_config=None):
    """
    Build model instance from MindFormer config.

    Args:
        config: Model configuration object.
        parallelism_config: Optional parallelism config when ``config`` is only a model config.
        dataset_config: Optional train dataset config used for model-side EOD compression flags.

    Returns:
        Model instance.
    """
    if hasattr(config, "model"):
        parallelism_config = getattr(config, "parallelism", parallelism_config)
        dataset_config = getattr(config, "train_dataset", dataset_config)
        config = config.model

    model_config = MindFormerRegister.get_instance_from_cfg(
        config.to_dict(), MindFormerModuleType.CONFIG
    )
    _sync_eod_compression_flag(dataset_config, model_config)
    model_config = attach_context_parallel_runtime_hints(model_config, parallelism_config)

    model = MindFormerRegister.get_instance_from_cfg(
        model_config.to_dict(),
        MindFormerModuleType.MODELS,
        default_args={"config": model_config},  # additional default arguments for the model
    )
    return model


def _build_lora_model(model, config):
    """
    Build LoRA wrapped model.

    Args:
        model: Base model instance.
        config: LoRA configuration.

    Raises:
        NotImplementedError: LoRA is not supported yet.
    """
    _, _ = model, config
    raise NotImplementedError("Lora model is not implemented yet.")


def _build_dataset(
    config,
    global_batch_size: int,
    dataset_parallel: int = 1,
    num_grad_acc: int = 1,
):
    """
    Build dataset and dataloader.

    Args:
        config: Dataset configuration.
        global_batch_size (int): Global batch size.
        dataset_parallel (int): Dataset parallel degree.
        num_grad_acc (int): Gradient accumulation steps.

    Returns:
        MindSpore Dataset instance.

    Raises:
        ValueError: If world size is not divisible by dataset parallel size.
    """
    def _set_ms_dataset_config():
        """
        Apply MindSpore dataset global configurations.
        """
        ms_dataset.config.set_prefetch_size(config.prefetch_size)
        ms_dataset.config.set_numa_enable(config.numa_enable)
        ms_dataset.config.set_num_parallel_workers(config.num_parallel_workers)

    def _compute_shard_info():
        """
        Compute dataset shard number and shard id for data parallel training.
        """
        if dataset_parallel == 1:
            return None, None

        world_size = get_world_size()
        if world_size % dataset_parallel != 0:
            raise ValueError(
                f"The world size {world_size} is not divisible by "
                f"the dataset parallel {dataset_parallel}."
            )

        dp_world_size = world_size // dataset_parallel
        dp_rank = get_rank() // dp_world_size
        return dataset_parallel, dp_rank

    def _actual_seq_len_batch_map(*cols, micro_batch_size):
        """
        Adjust actual sequence length offsets under gradient accumulation.

        This is required when using compressed EOD mask with micro-batching.
        """
        columns = cols[:-1]
        actual_seq_len = columns[-1]

        # Skip if batch size is 1
        if len(actual_seq_len) == 1:
            return columns

        batch_size = len(actual_seq_len) // micro_batch_size
        cur_seq_len = []

        for micro_idx in range(micro_batch_size):
            offset = 0
            start = micro_idx * batch_size
            end = (micro_idx + 1) * batch_size

            for seq_idx in range(start, end):
                per_seq = actual_seq_len[seq_idx] + offset
                offset = per_seq[-1]
                cur_seq_len.append(per_seq)

        # Replace actual_seq_len column only
        columns = columns[:-1] + (cur_seq_len,)
        return columns

    _set_ms_dataset_config()
    num_shards, shard_id = _compute_shard_info()

    dataloader_config = config.dataloader.to_dict()
    dataloader_config.update({"num_shards": num_shards, "shard_id": shard_id})
    dataloader_type = dataloader_config.pop("type")

    create_compressed_eod_mask = False
    if dataloader_type == "BlendedMegatronDatasetDataLoader":
        dataset = BlendedMegatronDatasetDataLoader(**dataloader_config)
        create_compressed_eod_mask = dataloader_config['config']['create_compressed_eod_mask']
    elif dataloader_type == "HFDataLoader":
        dataloader_config['use_broadcast_data'] = False
        dataset = HFDataLoader(**dataloader_config)
        create_compressed_eod_mask = dataloader_config['create_compressed_eod_mask']
    elif dataloader_type == "MindDataset":
        dataloader_config['dataset_files'] = _get_mindrecord_files(dataloader_config.pop("dataset_files", None))
        # MindDataset does not support column_names
        dataloader_config['columns_list'] = dataloader_config.pop('column_names', None)
        dataloader_config.pop('python_multiprocessing', False)
        dataset = ms_dataset.MindDataset(**dataloader_config)
        if 'actual_seq_len' in dataset.get_col_names():
            create_compressed_eod_mask = True
    else:
        raise ValueError(f"Unsupported dataloader type: {dataloader_type}")

    per_batch_map_func = None
    if create_compressed_eod_mask:
        per_batch_map_func = partial(
            _actual_seq_len_batch_map, micro_batch_size=num_grad_acc
        )

    dataset = dataset.batch(
        batch_size=global_batch_size // dataset_parallel,
        drop_remainder=config.drop_remainder,
        per_batch_map=per_batch_map_func,
    )
    return dataset


def _build_optimizer(
    config,
    model,
    learning_rate: Union[float, LearningRateSchedule],
):
    """
    Build optimizer.

    Args:
        config: Optimizer configuration.
        model: Model instance.
        learning_rate (float or LearningRateSchedule): Learning rate.

    Returns:
        Optimizer instance.
    """
    grouped_params = get_param_groups(
        model=model,
        weight_decay=config.weight_decay,
    )

    config = config.to_dict()
    optim_type = config.pop("type", None)

    if optim_type == "AdamW":
        optimizer = AdamW(
            params=grouped_params,
            learning_rate=learning_rate,
            **config,
        )
    elif optim_type == "Muon":
        optimizer = Muon(
            params=grouped_params,
            learning_rate=learning_rate,
            model=model,
            **config,
        )
    else:
        raise NotImplementedError(
            f"Optimizer type {optim_type} is not implemented yet."
        )

    return optimizer


def _build_lr_scheduler(config, total_steps: int):
    """
    Build learning rate scheduler.

    Args:
        config: LR scheduler configuration.
        total_steps (int): Total training steps.

    Returns:
        Learning rate scheduler instance.
    """
    config = config.to_dict()
    config["total_steps"] = total_steps
    return MindFormerRegister.get_instance_from_cfg(
        config, MindFormerModuleType.LR
    )


def _build_callback(config):
    """
    Build callback instance.

    Args:
        config: Callback configuration.

    Returns:
        Callback instance.
    """
    return MindFormerRegister.get_instance_from_cfg(
        config.to_dict(),
        MindFormerModuleType.CALLBACK,
    )


def _get_grad_factor(grad) -> float:
    """
    Compute the scaling factor for a gradient based on its distributed tensor placements.
    """
    from hyper_parallel.core.dtensor.placement_types import Replicate
    # Non-DTensor gradients do not need scaling
    if not isinstance(grad, DTensor):
        return 1.0
    # Gradients without mesh/placement metadata cannot be analyzed
    if not hasattr(grad, 'device_mesh') or not hasattr(grad, 'placements'):
        return 1.0

    mesh = grad.device_mesh
    placements = grad.placements
    factor = 1.0
    # Accumulate the product of mesh sizes along all replicated dimensions
    for dim, p in enumerate(placements):
        # Direct isinstance check when Replicate is importable
        if Replicate is not None and isinstance(p, Replicate):
            factor *= mesh.size(dim)
        # Fallback: compare by class name when Replicate import failed
        elif type(p).__name__ == 'Replicate':
            factor *= mesh.size(dim)
    return factor


def _calculate_global_grad_norm(
    parameters,
    enable_parallel: bool = False,
    max_norm: float = 1.0,
    eps: float = 1e-6,
):
    """Calculate the global gradient norm across all parameters and clip gradients in-place.

    Computes the global L2 norm of gradients, synchronizes it across distributed
    processes, and clips gradients in-place if the norm exceeds max_norm.

    Args:
        parameters: Iterable of parameters whose gradients will be clipped.
        enable_parallel (bool): Whether to enable parallel training. Default: False.
        max_norm (float): Maximum allowed gradient norm. Default: 1.0.
        eps (float): Small epsilon added to the norm for numerical stability when computing the clipping coefficient.
           Default: 1e-6.

    Returns:
        Tuple of (global_norm, grads) where global_norm is the computed global
        gradient norm and grads is a tuple of gradient tensors.
    """
    global_norm = Tensor(0.0)
    grads = []
    for param in parameters:
        if param.grad is not None:
            grad = param.grad
            grads.append(grad)
            local_grad = grad.to_local() if isinstance(grad, DTensor) else grad
            local_grad = local_grad.pow(2).sum() / _get_grad_factor(grad)
            global_norm += local_grad

    if enable_parallel:
        from mindspore import ops
        from mindspore.mint.distributed import all_reduce
        all_reduce(global_norm, op=ops.ReduceOp.SUM)

    global_norm = mint.sqrt(global_norm)

    # Clip gradients in-place if the global norm exceeds max_norm
    clip_coef = max_norm / (global_norm + eps)
    if clip_coef < 1:
        scale = clip_coef.item()
        for grad in grads:
            if grad is not None:
                grad.mul_(scale)

    return global_norm, tuple(grads)

def _get_loss_sense(parallelism, enable_parallel: bool = False):
    """
    Calculate the loss scaling factor for distributed training.

    Args:
        loss: The loss tensor, either a regular Tensor or a DTensor.
        parallelism: Parallelism configuration.
        enable_parallel (bool): Whether to enable parallel training. Default: False.

    Returns:
        A scalar Tensor containing the loss scaling factor.
            - Returns [1.0] if loss is not a DTensor (non-distributed case).
            - Returns [1.0 / (num_devices // pipeline_parallel)] if loss is a DTensor.
    """
    if not enable_parallel:
        return Tensor([1.0,], dtype=mstype.float32)

    num_devices = get_world_size()
    pipeline_parallel = int(parallelism.pipeline_parallel)
    sense = 1. / (num_devices // pipeline_parallel)
    logger.debug(
        f"Got loss sense({sense}) = 1. / (num_devices({num_devices}) // pipeline_parallel({pipeline_parallel}))"
    )
    return Tensor([sense,], dtype=mstype.float32)
