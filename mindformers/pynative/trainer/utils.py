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

from fnmatch import fnmatch
from functools import partial
from typing import List, Optional, Set, Tuple, Union
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
from mindformers.pynative.pet.lora_adapter import build_lora_model


def get_grad(param):
    """Return the gradient to consume for optimizer update / norm / monitoring.

    Multi-card (fully_shard) runs route the reduced gradient onto a fp32
    ``param.main_grad`` buffer and leave ``param.grad`` at None (HSDP
    ``apply_grad_on_fp32_main_grad`` policy); single-card runs accumulate fp32
    gradients directly on ``param.grad`` via backward hooks. This accessor hides
    the difference from callers.
    """
    main_grad = getattr(param, "main_grad", None)
    return main_grad if main_grad is not None else param.grad


def _is_dtensor_like(value):
    """Return whether value behaves like a DTensor gradient."""
    return isinstance(value, DTensor) or hasattr(value, "to_local")


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


def _get_no_wd_params(model: nn.Cell) -> Tuple[Set[str], Set[str]]:
    """
    Get parameter names and name keywords that should NOT apply weight decay.

    This function queries optional model interfaces:
        - model.no_weight_decay()
        - model.no_weight_decay_keywords()

    Args:
        model: MindSpore model instance.

    Returns:
        Tuple[Set[str], Set[str]]:
        - no_wd_params: Explicit parameter names without weight decay.
        - no_wd_keywords: Keywords used to match parameter names.
    """
    no_wd_params: Set[str] = set()
    no_wd_keywords: Set[str] = set()

    if hasattr(model, "no_weight_decay"):
        no_wd_params = set(model.no_weight_decay())
        logger.info(f"Get no weight decay params: {no_wd_params}")

    if hasattr(model, "no_weight_decay_keywords"):
        no_wd_keywords = set(model.no_weight_decay_keywords())
        logger.info(f"Get no weight decay keywords: {no_wd_keywords}")

    return no_wd_params, no_wd_keywords


def _normalize_weight_decay_rules(rules, field_name: str) -> List[str]:
    """Normalize custom weight-decay rules to a list of strings."""
    if rules is None:
        return []
    if not isinstance(rules, (list, tuple)):
        raise TypeError(
            f"{field_name} should be a list of strings, but got {type(rules)}."
        )
    for rule in rules:
        if not isinstance(rule, str):
            raise TypeError(
                f"All items in {field_name} should be strings, but got {type(rule)}."
            )
    return list(rules)


def _match_weight_decay_rule(param_name: str, rules: List[str]) -> Optional[str]:
    """
    Match a parameter name against custom rules.

    Rules support exact names, substring matches, and shell-style wildcards.
    """
    for rule in rules:
        if rule == param_name or rule in param_name or fnmatch(param_name, rule):
            return rule
    return None


def get_param_groups(
    model: nn.Cell,
    weight_decay: float = 0.0,
    weight_decay_include: Optional[List[str]] = None,
    weight_decay_exclude: Optional[List[str]] = None,
) -> List[dict]:
    """
    Create optimizer parameter groups.

    Weight decay will be disabled for:
        - Parameters matched by weight_decay_exclude
        - 1D parameters (e.g., LayerNorm / bias)
        - Bias parameters
        - Parameters explicitly specified by model
        - Parameters matched by no-weight-decay keywords

    Parameters matched by weight_decay_include always use weight decay, even if
    they match the default no-weight-decay rules. The priority is:
    weight_decay_include > weight_decay_exclude > default no-weight-decay rules.

    Args:
        model: MindSpore model instance.
        weight_decay (float): Base weight decay value.
        weight_decay_include (Optional[List[str]]): Parameter name rules that
            force weight decay on. Rules support exact names, substring matches,
            and shell-style wildcards.
        weight_decay_exclude (Optional[List[str]]): Parameter name rules that
            force weight decay off. Rules support exact names, substring matches,
            and shell-style wildcards.

    Returns:
        List[dict]: Parameter groups compatible with MindSpore optimizers.
    """
    # Get no weight decay params and keywords from model interface
    no_wd_params, no_wd_keywords = _get_no_wd_params(model)
    weight_decay_include = _normalize_weight_decay_rules(
        weight_decay_include, "weight_decay_include"
    )
    weight_decay_exclude = _normalize_weight_decay_rules(
        weight_decay_exclude, "weight_decay_exclude"
    )
    if weight_decay_include:
        logger.info(f"Get weight decay include rules: {weight_decay_include}")
    if weight_decay_exclude:
        logger.info(f"Get weight decay exclude rules: {weight_decay_exclude}")

    def _matches_keyword(p_name):
        return any(keyword in p_name for keyword in no_wd_keywords)

    def _classify(param, p_name):
        """Return (no_wd, reason) for a parameter."""
        rule = _match_weight_decay_rule(p_name, weight_decay_include)
        if rule is not None:
            return False, f"custom_include:{rule}"
        rule = _match_weight_decay_rule(p_name, weight_decay_exclude)
        if rule is not None:
            return True, f"custom_exclude:{rule}"

        default_no_wd_rules = (
            ("1d_param", len(param.shape) == 1),
            ("bias", p_name.endswith(".bias")),
            ("model_no_weight_decay", p_name in no_wd_params),
            ("model_no_weight_decay_keywords", _matches_keyword(p_name)),
        )
        for reason, hit in default_no_wd_rules:
            if hit:
                return True, reason
        return False, "default"

    # Iterate over trainable parameters and assign them to groups
    logger.info("Assigning parameters to groups...")
    parameter_groups = {}

    for param in model.trainable_params():
        param_name = param.name
        no_wd, wd_reason = _classify(param, param_name)
        wd_value = 0.0 if no_wd else weight_decay
        group_name = "no_weight_decay" if no_wd else "weight_decay"

        if group_name not in parameter_groups:
            parameter_groups[group_name] = {"weight_decay": wd_value, "params": []}
        parameter_groups[group_name]["params"].append(param)

        logger.info(
            f"param_name: {param_name:<60} | weight_decay: "
            f"{wd_value:>8.4f} | reason: {wd_reason}"
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
    Inject LoRA adapters into the base model in place (PyNative, FSDP-only first version).

    Replaces target ``Linear`` layers with ``LinearWithLoRA`` and reuses the original
    weights. Base-parameter freezing is performed separately by the trainer, after
    parallelism + ``init_states`` (see ``freeze_base_params``).

    Args:
        model: Base model instance (built on meta device).
        config: LoRA configuration (``target_modules`` regex required).

    Returns:
        The same model instance, mutated in place.
    """
    return build_lora_model(model, config)


def _build_dataset(
    config,
    global_batch_size: int,
    parallelism,
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
    dataset_parallel = parallelism.data_parallel
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

        shard_id = get_rank() % (world_size // parallelism.pipeline_parallel) // \
            parallelism.tensor_parallel // parallelism.context_parallel
        return dataset_parallel, shard_id

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
    config = config.to_dict()
    optim_type = config.pop("type", None)
    grouped_params = []
    for m in model:
        params = get_param_groups(
            model=m,
            weight_decay=config["weight_decay"],
            weight_decay_include=config.pop("weight_decay_include", None),
            weight_decay_exclude=config.pop("weight_decay_exclude", None),
        )
        grouped_params.extend(params)

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
            model=model[0],
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
    if not _is_dtensor_like(grad):
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


def _group_grads_by_device_and_dtype(grads):
    """Group gradients by device and dtype for batched processing."""
    groups = {}
    for grad in grads:
        local_grad = grad.to_local() if _is_dtensor_like(grad) else grad
        key = (local_grad.device, local_grad.dtype)
        if key not in groups:
            groups[key] = []
        groups[key].append((grad, local_grad))
    return groups


def _compute_grad_norm_sq_fused(local_grads, grad_factors):
    """Compute sum of squared norms using batched vector_norm."""
    if not local_grads:
        return Tensor(0.0, dtype=mstype.float32)

    norm_sqs = []
    for local_grad, factor in zip(local_grads, grad_factors):
        grad_fp32 = local_grad.astype(mstype.float32)
        norm_sq = mint.sum(mint.square(grad_fp32)) / factor
        norm_sqs.append(norm_sq)

    if len(norm_sqs) == 1:
        return norm_sqs[0]

    stacked = mint.stack(norm_sqs)
    return mint.sum(stacked)


def _calculate_global_grad_norm(
    parameters,
    enable_parallel: bool = False,
    max_norm: float = 1.0,
    eps: float = 1e-6,
):
    """Calculate the global gradient norm across all parameters and clip gradients in-place.

    Computes the global L2 norm of gradients, synchronizes it across distributed
    processes, and clips gradients in-place if the norm exceeds max_norm.

    Performance optimizations (MindSpore mint API based):
    - Uses mint.square + mint.sum for efficient L2 norm computation (better than pow(2).sum())
    - Groups gradients by device/dtype to minimize kernel launch overhead
    - Uses mint.stack for batched aggregation of per-tensor norms
    - FP32 precision for norm computation to avoid fp16/bf16 numerical instability

    Note: Unlike PyTorch's foreach API or DeepSpeed's fused kernel, this implementation
    uses mint.stack + grouped processing due to MindSpore API limitations. Performance
    is comparable to foreach-based approaches for typical model sizes.

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
    grads = []
    for param in parameters:
        grad = get_grad(param)
        if grad is not None:
            grads.append(grad)

    if not grads:
        return Tensor(0.0, dtype=mstype.float32), ()

    grouped_grads = _group_grads_by_device_and_dtype(grads)

    group_norm_sqs = []
    for (_, _), grad_list in grouped_grads.items():
        local_grads = [item[1] for item in grad_list]
        grad_factors = [_get_grad_factor(item[0]) for item in grad_list]
        group_norm_sq = _compute_grad_norm_sq_fused(local_grads, grad_factors)
        group_norm_sqs.append(group_norm_sq)

    if len(group_norm_sqs) == 1:
        total_norm_sq = group_norm_sqs[0]
    else:
        total_norm_sq = mint.sum(mint.stack(group_norm_sqs))

    if enable_parallel:
        from mindspore import ops
        from mindspore.mint.distributed import all_reduce
        all_reduce(total_norm_sq, op=ops.ReduceOp.SUM)

    global_norm = mint.sqrt(total_norm_sq)

    # Clip gradients in-place if the global norm exceeds max_norm
    clip_coef = max_norm / (global_norm + eps)
    if clip_coef < 1:
        scale = clip_coef.item()
        for (_, _), grad_list in grouped_grads.items():
            for grad, local_grad in grad_list:
                if _is_dtensor_like(grad):
                    local_grad.mul_(scale)
                else:
                    grad.mul_(scale)

    return global_norm, tuple(grads)
