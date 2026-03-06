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

from mindspore import nn
import mindspore.dataset as ms_dataset
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore.mint.distributed import (
    get_world_size,
    get_rank,
)

from mindformers.tools.logger import logger
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindformers.core.optim.adamw import AdamW
from mindformers.dataset.dataloader.blended_megatron_dataloader import (
    BlendedMegatronDatasetDataLoader,
)
from mindformers.dataset.dataloader.hf_dataloader import HFDataLoader
from mindformers.dataset.dataloader.utils import _get_mindrecord_files


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


def _build_model(config):
    """
    Build model instance from MindFormer config.

    Args:
        config: Model configuration object.

    Returns:
        Model instance.
    """
    model_config = MindFormerRegister.get_instance_from_cfg(
        config.to_dict(), MindFormerModuleType.CONFIG
    )

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
        world_size = get_world_size()
        if world_size % dataset_parallel != 0:
            raise ValueError(
                f"The world size {world_size} is not divisible by "
                f"the dataset parallel {dataset_parallel}."
            )

        dp_world_size = world_size // dataset_parallel
        dp_rank = get_rank() // dp_world_size
        return dp_world_size, dp_rank

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
