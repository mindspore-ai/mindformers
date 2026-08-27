# Copyright 2025 Huawei Technologies Co., Ltd
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
"""save / load parallelization strategy."""
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from mindspore import save_checkpoint
from mindspore.nn import Cell
from mindspore.mint.distributed import scatter_object_list

from mindformers.checkpoint.layout_adapter import LayoutAdapter
from mindformers.checkpoint.sharded_tensor import (
    ShardedTensor,
    get_all_sharded_tensor,
    get_all_sharded_tensor_on_rank0,
    get_cur_sharded_tensor,
    get_param_redundancy_after_balanced
)
from mindformers.tools.logger import logger
from mindformers.tools.utils import get_real_rank, get_real_group_size
from mindformers.checkpoint.metadata import save_metadata
from mindformers.checkpoint.utils import (
    _reverse_sharded_tensor_shard_id,
    get_checkpoint_iter_dir,
    get_metadata_filename,
    get_checkpoint_name,
    FileType,
    _get_shard_size,
    sharded_tensor_shard_id
)


@dataclass
class BalancedShardPlan:
    """
    Per-rank view of a balanced shard distribution.

    Holds only what the local rank needs, so that the global (potentially quadratic-size)
    shard metadata never has to be materialized on non-zero ranks in PyNative mode on
    large-scale clusters.

    Attributes:
        cur_rank_shards (Dict[str, Tuple[ShardedTensor, Tuple[int, ...]]]): Shards assigned to
            the current rank, mapping unique shard IDs to tuples of the corresponding
            `ShardedTensor` object and the rank group holding redundant copies of the shard.
        param_redundancy (Dict[Tuple[int, ...], List[str]]): Redundant parameter groups that
            contain the current rank, mapping rank groups to lists of parameter names.
        total_files_num (int): Total number of checkpoint files written by all ranks.
        cur_rank_file_id (Optional[int]): Checkpoint file identifier of the current rank,
            None if no shard is assigned to the current rank.
        full_assignment (Optional[Dict[int, Dict[str, Tuple[int, ...]]]]): The complete shard
            assignment `{rank_id: {shard_id: rank_group}}`, populated on rank 0 only (used to
            write 'metadata.json'). None on all other ranks.
    """

    cur_rank_shards: Dict[str, Tuple[ShardedTensor, Tuple[int, ...]]] = field(default_factory=dict)
    """Shards assigned to the current rank: {shard_id: (ShardedTensor, rank_group)}."""

    param_redundancy: Dict[Tuple[int, ...], List[str]] = field(default_factory=dict)
    """Redundant rank groups containing the current rank: {rank_group: [param_name]}."""

    total_files_num: int = 0
    """Total number of checkpoint files written by all ranks."""

    cur_rank_file_id: Optional[int] = None
    """Checkpoint file identifier of the current rank."""

    full_assignment: Optional[Dict[int, Dict[str, Tuple[int, ...]]]] = None
    """Complete shard assignment, populated on rank 0 only."""

    @property
    def cur_rank_sharded_tensors(self) -> Dict[str, ShardedTensor]:
        """ShardedTensor instances assigned to the current rank, keyed by parameter name."""
        return {
            sharded_tensor.key: sharded_tensor
            for sharded_tensor, _ in self.cur_rank_shards.values()
        }

    @property
    def cur_rank_param_names(self) -> List[str]:
        """Parameter names of the shards assigned to the current rank."""
        return [sharded_tensor.key for sharded_tensor, _ in self.cur_rank_shards.values()]


class BalancedSaveStrategy():
    """
    A class that implements a balanced saving strategy for model checkpoints in a distributed training environment.

    This strategy aims to evenly distribute the saving of model parameters across multiple ranks to optimize
    the checkpointing process. It takes into account the shared distribution of parameters among ranks and
    ensures that each rank saves only the parameters it is responsible for. Additionally, it provides options
    for caching the distribution information and saving metadata about the checkpoint files.

    Attributes:
        network: The neural network model to be saved.
        user_prefix (str): A user-defined prefix that can be used to customize the naming of checkpoint files.
            Defaults to an empty string.
        do_cache_distribution (bool): A flag indicating whether to cache the shared distribution information.
            Caching can improve performance if the distribution remains the same across multiple checkpoint saves.
            Defaults to False.
        cached_distribution (dict or None): The cached shared distribution information, if caching is enabled.
            Initially set to None.
        checkpoint_path (str or None): The directory path where the checkpoint files will be saved.
            Defaults to None.
        file_type (str): Specific file types corresponding to shard weights.
    """

    def __init__(self, network, user_prefix=None, do_cache_distribution=False, checkpoint_path=None,
                 filter_func=None, file_type=FileType.MODEL, global_sharded_tensor_metas=None, plan_cache=None):
        """
        Initialize the BalancedSaveStrategy object.

        Args:
            network: The neural network model (or optimizer) to be saved.
            user_prefix (str): A user-defined prefix for checkpoint file names.
            do_cache_distribution (bool): Whether to cache the shard distribution on this instance.
            checkpoint_path (str): The directory path where checkpoint files are saved.
            filter_func (Callable): Filter selecting the parameters of this file type.
            file_type (FileType): Model or optimizer checkpoint file type.
            global_sharded_tensor_metas (dict): The global ShardedTensor metadata of this file
                type, keyed by rank ID. Only rank 0 holds the real dictionary; all other ranks
                pass None. Mandatory on rank 0 in PyNative multi-rank scenarios, where rank 0
                computes the balanced shard distribution from it. The dictionary must already
                be filtered to this file type's parameters by the caller.
            plan_cache (dict): Optional caller-owned dictionary used to cache the built
                BalancedShardPlan across saves (keyed by `file_type`). The parallel layout is
                static during training, so a cached plan is reused directly without any
                collective communication on subsequent saves.
        """
        super().__init__()
        self.user_prefix = user_prefix
        self.do_cache_distribution = do_cache_distribution
        self.total_files_num = None
        self.cur_rank_file_id = None
        self.cached_distribution = None
        self.rank_id = get_real_rank()
        self.checkpoint_path = checkpoint_path
        self.network = network
        self.ckpt_format = "safetensors"
        self.filter_func = filter_func
        self.file_type = file_type
        self.global_sharded_tensor_metas = global_sharded_tensor_metas
        self.plan_cache = plan_cache

    def get_total_files(self):
        """
        Get the total number of checkpoint files required for all ranks.

        If the total number of files has not been calculated yet, this method will calculate it based on the
        shared distribution of parameters among ranks.

        Returns:
            The total number of checkpoint files.
        """
        if self.total_files_num is None:
            plan = self.apply_saving_parallelization()
            self.total_files_num = plan.total_files_num

        return self.total_files_num

    def get_cur_rank_file_id(self):
        """
        Get the identifier for the current rank's checkpoint file.

        If the identifier has not been calculated yet, this method will calculate it based on the shared
        distribution of parameters among ranks.

        Returns:
            The identifier for the current rank's checkpoint file.
        """
        if self.cur_rank_file_id is None:
            plan = self.apply_saving_parallelization()
            self.cur_rank_file_id = plan.cur_rank_file_id

        return self.cur_rank_file_id

    def save(self, iteration, async_save: bool = False):
        """
        Save the model checkpoint using the balanced saving strategy.

        This method determines which parameters should be saved by the current rank based on the shared distribution,
            generates the appropriate checkpoint file name,
            and saves the selected parameters in the specified format.

        Args:
            iteration (int): The current iteration number.
            async_save (bool): Whether to use async save. Defaults to False.

        Returns:
            BalancedShardPlan: The per-rank view of the balanced shard distribution used for
                this save. Its `full_assignment` field is populated on rank 0 only. The caller
                is responsible for writing 'metadata.json' (see `save_balanced_metadata`).
        """
        plan = self.apply_saving_parallelization()

        if self.total_files_num is None:
            self.total_files_num = plan.total_files_num
        if self.cur_rank_file_id is None:
            self.cur_rank_file_id = plan.cur_rank_file_id

        save_ckpt_path = get_checkpoint_iter_dir(self.checkpoint_path, iteration)
        save_file_name = os.path.join(
            save_ckpt_path,
            get_checkpoint_name(None, self.user_prefix, self.cur_rank_file_id, self.total_files_num, self.file_type)
        )

        cur_rank_param_names = set(plan.cur_rank_param_names)

        def choice_func(param_name):
            if param_name in cur_rank_param_names:
                return True
            return False

        start_time = time.time()
        save_checkpoint(
            LayoutAdapter.preprocess_params(self.network),
            save_file_name,
            format=self.ckpt_format,
            choice_func=choice_func,
            integrated_save=False,
            async_save=async_save
        )
        logger.info(
            f"Non-redundancy {self.file_type.value} checkpoint successfully saved at '{save_file_name}.safetensors. "
            f"Save time: {time.time() - start_time:.4f} seconds."
        )

        return plan

    def apply_saving_parallelization(self):
        """
        Get the shared distribution of parameters among ranks.

        If a caller-owned plan cache is provided and already holds this file type's plan
        (built at a previous save), the cached plan is returned directly without any
        collective communication. If instance caching is enabled and the distribution has
        been cached, this method will return the cached distribution. Otherwise, it builds
        the distribution and caches it if caching is enabled.

        Returns:
            BalancedShardPlan: The per-rank view of the balanced shard distribution. See
            `BalancedShardPlan` for the contained fields.
        """
        if self.plan_cache is not None and self.file_type in self.plan_cache:
            return self.plan_cache[self.file_type]

        if self.do_cache_distribution and self.cached_distribution is not None:
            plan = self.cached_distribution
        elif LayoutAdapter.is_pynative_mode() and get_real_group_size() > 1:
            plan = _build_balanced_shard_plan_from_global_metas(
                self.network, self.filter_func, self.global_sharded_tensor_metas
            )
        else:
            plan = build_balanced_shard_plan(self.network, self.filter_func)

        if self.do_cache_distribution:
            self.cached_distribution = plan
        if self.plan_cache is not None:
            self.plan_cache[self.file_type] = plan

        return plan


def save_balanced_metadata(checkpoint_path, iteration, user_prefix, balanced_plans, sharded_tensor_metas):
    """
    Write one combined 'metadata.json' on rank 0 for the balanced (non-redundancy) saving flow.

    Only rank 0 performs the write; all other ranks return immediately (this function
    contains no collective communication). The file content is the union of all file types:
    `state_dict_metadata` is built from the full global ShardedTensor metadata (network and
    optimizer parameters combined), and `storage_data` covers the shard assignments of every
    file type's balanced plan.

    Args:
        checkpoint_path (str): The root directory of the checkpoint.
        iteration (int): The current iteration number.
        user_prefix (str): The user-defined prefix of checkpoint file names.
        balanced_plans (List[Tuple[BalancedShardPlan, FileType]]): The balanced plans of all
            saved file types (model, and optimizer when saved). `full_assignment` of each
            plan is populated on rank 0 only.
        sharded_tensor_metas (Dict[int, Dict[str, ShardedTensor]]): The global ShardedTensor
            metadata keyed by rank ID (network + optimizer combined), held on rank 0 only;
            None on all other ranks.
    """
    if get_real_rank() != 0:
        return

    param_file_mapping = []
    for plan, file_type in balanced_plans:
        cur_rank_id = 0
        for rank_id, shard_groups in (plan.full_assignment or {}).items():
            if not shard_groups:
                continue
            save_file_name = get_checkpoint_name(
                None, user_prefix, cur_rank_id, plan.total_files_num, file_type
            )
            for shard_id, rank_group in shard_groups.items():
                param_file_mapping.append((
                    save_file_name + ".safetensors",
                    rank_id,
                    rank_group,
                    _reverse_sharded_tensor_shard_id(shard_id)
                ))
            cur_rank_id += 1

    metadata_file_path = get_metadata_filename(checkpoint_path, iteration)
    save_metadata(sharded_tensor_metas, param_file_mapping, metadata_file_path)
    logger.info(
        f"The 'metadata.json' of non-redundancy weight saved successfully at '{metadata_file_path}'."
    )


def distribute_shards(shard_coverage, shard_sizes, total_ranks):
    """
    Distribute shards to ranks using a greedy algorithm based on the following priority:
    1. Shards with fewer covering ranks are assigned first.
    2. For shards with the same number of covering ranks, larger shards are assigned first.
    3. For shards with the same size, the shard ID is used as a tiebreaker.

    Args:
        shard_coverage (dict): A dictionary mapping shard IDs to a list of ranks that contain the shard.
        shard_sizes (dict): A dictionary mapping shard IDs to their size in bytes.
        total_ranks (int): The total number of ranks.

    Returns:
        Dict[str, Tuple[int, Tuple[int, ...]]]: A dictionary where each key is a unique shard ID,
        and the corresponding value is a 2-element tuple:
        1. Selected target rank (int): The rank assigned to store the shard (chosen to minimize current load).
        2. Rank group (Tuple[int, ...]): Ranks that originally contain the shard (from `shard_coverage`),
            representing redundant copies for fault tolerance or parallel access.
    """
    coverage_map = {
        k: tuple(sorted(v))
        for k, v in shard_coverage.items()
    }
    rank_loads = {
        rank: 0
        for rank in range(total_ranks)
    }
    shard_assignment = {}
    sorted_shards = sorted(
        coverage_map.items(),
        key=lambda item: (len(item[1]), -shard_sizes[item[0]], item[0])
    )

    for shard_id, available_ranks in sorted_shards:
        selected_rank = min(available_ranks, key=lambda rank: rank_loads[rank])
        shard_assignment[shard_id] = (selected_rank, available_ranks)
        rank_loads[selected_rank] += shard_sizes[shard_id]

    return shard_assignment


def apply_balance_shard_strategy(network: List[Cell], filter_func: Callable[[str], bool] = None):
    """
    Distributes and balances sharded tensor storage across ranks in a parallel group,
    generating rank-specific shard assignments.

    Collects sharded tensor metadata from the input MindSpore Network Cell (filtered by an optional function),
    computes unique shard identifiers and their sizes, and distributes shards to ranks using a load-balanced strategy.
    The result maps each target rank to its assigned shards along with the group of ranks that share redundant copies
    of those shards.

    Core Workflow:
    1. Extract all sharded tensor metadata from the network using `get_all_sharded_tensor`, applying the `filter_func`
       to select target tensors (e.g., exclude non-trainable parameters).
    2. Generate unique shard IDs for each tensor shard (via `sharded_tensor_shard_id`) by combining the tensor key
       and global offset, then track which ranks originally own each shard.
    3. Calculate the byte size of each unique shard using its local shape and data type (via `_get_shard_size`),
       avoiding redundant size computations for identical shards.
    4. Distribute shards to ranks for storage using the `distribute_shards` function, which implements a load-balanced
       algorithm to evenly distribute the storage load across the parallel group.
    5. Compile a rank-to-shard mapping: for each rank, store its assigned shards and the corresponding rank group
       (ranks with redundant copies of the same shard).

    Args:
        network (Cell): A MindSpore Network Cell containing parameters and their associated sharding metadata.
        filter_func (Optional[Callable[[str], bool]]): An optional filtering function that takes a tensor key (str)
            and returns a boolean. Only tensors for which the function returns `True` are included in the shard
            distribution. Defaults to `None` (all sharded tensors in the network are included).

    Returns:
        Dict[int, Dict[str, Tuple]]: A nested dictionary where:
        - Outer keys: Target rank IDs (int) in the parallel group.
        - Outer values: Dictionaries mapping unique shard IDs (str) to tuples containing:
        1. Corresponding `ShardedTensor` object with complete shard metadata (local shape, dtype, global offset, etc.).
        2. Rank group (tuple of ints): Ranks that have redundant copies of the shard (supports fault tolerance or
            parallel access).
    """
    total_shard_metadata = get_all_sharded_tensor(network, filter_func)
    shard_id_to_ranks = defaultdict(list)
    shard_to_size = {}
    shards_in_this_parallelization_group = set()
    shard_id_to_tensor = {}

    for rank, sharded_tensor_metas in total_shard_metadata.items():
        for sharded_tensor in sharded_tensor_metas.values():
            shard_id = sharded_tensor_shard_id(sharded_tensor.key, sharded_tensor.global_offset)
            shard_id_to_ranks[shard_id].append(rank)

            if shard_id not in shard_to_size:
                shard_to_size[shard_id] = _get_shard_size(sharded_tensor.local_shape, sharded_tensor.dtype)
                shard_id_to_tensor[shard_id] = sharded_tensor
            shards_in_this_parallelization_group.add(shard_id)

    shard_id_to_ranks = {
        k: v
        for k, v in shard_id_to_ranks.items()
        if k in shards_in_this_parallelization_group
    }

    shard_to_saving_rank = distribute_shards(
        shard_id_to_ranks, shard_to_size, len(total_shard_metadata)
    )

    rank_id_to_sharded_tensors = {}
    for shard_id, rank_info in shard_to_saving_rank.items():
        selected_rank_id, rank_group = rank_info
        sharded_tensor = shard_id_to_tensor[shard_id]
        if selected_rank_id in rank_id_to_sharded_tensors:
            rank_id_to_sharded_tensors[selected_rank_id][shard_id] = (sharded_tensor, rank_group)
        else:
            rank_id_to_sharded_tensors[selected_rank_id] = {shard_id: (sharded_tensor, rank_group)}

    rank_id_to_sharded_tensors = {
        k: rank_id_to_sharded_tensors.get(k, None)
        for k in sorted(rank_id_to_sharded_tensors)
    }

    return rank_id_to_sharded_tensors


def build_balanced_shard_plan(
        network: List[Cell],
        filter_func: Callable[[str], bool] = None
) -> BalancedShardPlan:
    """
    Builds the balanced shard distribution plan for the current rank.

    In Graph mode (or single-rank jobs), the plan is derived from the full shard distribution
    computed locally by `apply_balance_shard_strategy`. In PyNative multi-rank scenarios,
    every rank builds only its own ShardedTensor metadata locally (no global layout is
    materialized anywhere), the per-rank metadata is gathered onto rank 0, where the
    de-redundancy and balanced assignment are computed, and each rank's assignment is
    scattered back from rank 0. This avoids the CPU memory explosion and the O(world_size)
    gather overhead per rank on large-scale clusters (e.g. 10K cards).

    Note:
        This is a collective interface in PyNative multi-rank scenarios: **all ranks must
        call it** (a `gather_object` towards rank 0 followed by a `scatter_object_list`
        from rank 0 are performed).

    Args:
        network (Cell): A MindSpore Network Cell containing parameters and their associated
            sharding metadata.
        filter_func (Optional[Callable[[str], bool]]): An optional filtering function that takes
            a tensor key (str) and returns a boolean. Only tensors for which the function returns
            `True` are included in the shard distribution. Defaults to `None` (all sharded tensors
            in the network are included).

    Returns:
        BalancedShardPlan: The per-rank view of the balanced shard distribution.
    """
    if LayoutAdapter.is_pynative_mode() and get_real_group_size() > 1:
        return _build_balanced_shard_plan_with_gather(network, filter_func)
    return _build_balanced_shard_plan_legacy(network, filter_func)


def _build_balanced_shard_plan_legacy(
        network: List[Cell],
        filter_func: Callable[[str], bool] = None
) -> BalancedShardPlan:
    """
    Builds the balanced shard plan from the full shard distribution computed locally.

    Used in Graph mode and single-rank jobs, where the global strategy metadata is available
    locally on every rank without extra communication.
    """
    rank_id_to_sharded_tensors = apply_balance_shard_strategy(network, filter_func)

    local_rank = get_real_rank()
    cur_rank_shards = rank_id_to_sharded_tensors.get(local_rank) or {}
    param_redundancy = get_param_redundancy_after_balanced(rank_id_to_sharded_tensors)

    total_files_num = 0
    cur_rank_file_id = None
    full_assignment = {}
    for rank_id, sharded_tensors in rank_id_to_sharded_tensors.items():
        if rank_id == local_rank:
            cur_rank_file_id = total_files_num
        if sharded_tensors:
            total_files_num += 1
            full_assignment[rank_id] = {
                shard_id: rank_group for shard_id, (_, rank_group) in sharded_tensors.items()
            }

    return BalancedShardPlan(
        cur_rank_shards=cur_rank_shards,
        param_redundancy=param_redundancy,
        total_files_num=total_files_num,
        cur_rank_file_id=cur_rank_file_id,
        full_assignment=full_assignment
    )


def _build_local_shard_map(
        network: List[Cell],
        filter_func: Callable[[str], bool] = None
) -> Dict[str, ShardedTensor]:
    """
    Build the current rank's ShardedTensor metadata locally and index it by shard ID.

    Purely local operation (no collective communication).
    """
    local_metas = get_cur_sharded_tensor(network, filter_func) or {}
    local_shard_map = {}
    for _, sharded_tensor in local_metas.items():
        shard_id = sharded_tensor_shard_id(sharded_tensor.key, sharded_tensor.global_offset)
        local_shard_map[shard_id] = sharded_tensor
    return local_shard_map


def _compute_balanced_assignment_on_rank0(
        global_sharded_tensor_metas: Dict[int, Dict[str, ShardedTensor]],
        world_size: int
) -> Tuple[List[Tuple], Dict[int, Dict[str, Tuple[int, ...]]]]:
    """
    Compute the balanced shard assignment on rank 0 from the global ShardedTensor metadata.

    Uses the same `distribute_shards` algorithm as the legacy implementation, producing
    identical assignment results. Only rank 0 calls this function.

    Args:
        global_sharded_tensor_metas (Dict[int, Dict[str, ShardedTensor]]): The global
            ShardedTensor metadata keyed by rank ID, already filtered to the target
            parameters by the caller.
        world_size (int): The total number of ranks.

    Returns:
        Tuple containing:
        - scatter_payloads (List[Tuple]): Per-rank payloads `(assigned_shards,
          param_redundancy, cur_rank_file_id, total_files_num)` to be scattered by rank 0.
        - full_assignment (Dict[int, Dict[str, Tuple[int, ...]]]): The complete shard
          assignment `{rank_id: {shard_id: rank_group}}`, kept on rank 0 for writing
          'metadata.json'.
    """
    shard_id_to_ranks = defaultdict(list)
    shard_to_size = {}
    for rank_id, metas in global_sharded_tensor_metas.items():
        if not metas:
            continue
        for sharded_tensor in metas.values():
            shard_id = sharded_tensor_shard_id(sharded_tensor.key, sharded_tensor.global_offset)
            shard_id_to_ranks[shard_id].append(rank_id)
            if shard_id not in shard_to_size:
                shard_to_size[shard_id] = _get_shard_size(sharded_tensor.local_shape, sharded_tensor.dtype)

    shard_to_saving_rank = distribute_shards(shard_id_to_ranks, shard_to_size, world_size)

    full_assignment = {}
    for shard_id, (selected_rank_id, rank_group) in shard_to_saving_rank.items():
        full_assignment.setdefault(selected_rank_id, {})[shard_id] = rank_group
    full_assignment = {k: full_assignment[k] for k in sorted(full_assignment)}

    total_files_num = len(full_assignment)
    cur_rank_file_ids = {rank_id: file_id for file_id, rank_id in enumerate(full_assignment)}
    param_redundancy_per_rank = defaultdict(dict)
    for shard_id, (_, rank_group) in shard_to_saving_rank.items():
        if len(rank_group) <= 1:
            continue
        param_name, _ = _reverse_sharded_tensor_shard_id(shard_id)
        for rank_id in rank_group:
            param_redundancy_per_rank[rank_id].setdefault(tuple(rank_group), []).append(param_name)

    scatter_payloads = []
    for rank_id in range(world_size):
        assigned_shards = full_assignment.get(rank_id, {})
        scatter_payloads.append(
            (assigned_shards, param_redundancy_per_rank.get(rank_id, {}),
             cur_rank_file_ids.get(rank_id), total_files_num)
        )

    return scatter_payloads, full_assignment


def _scatter_balanced_shard_plan(
        scatter_payloads: Optional[List[Tuple]],
        full_assignment: Optional[Dict[int, Dict[str, Tuple[int, ...]]]],
        local_shard_map: Dict[str, ShardedTensor]
) -> BalancedShardPlan:
    """
    Receive the per-rank shard assignment scattered from rank 0 and rebuild the local plan.

    Every rank rebuilds its assigned shards from its own local ShardedTensor metadata. This
    is valid because the ShardedTensor of a given shard (identified by parameter name and
    global offset) has identical content on every rank that owns the shard.

    Note:
        Collective interface: **all ranks must call it** (a `scatter_object_list` from rank 0
        is performed).
    """
    recv_payload = [(None, None, None, None)]
    scatter_object_list(recv_payload, scatter_payloads, src=0)
    assigned_shards, param_redundancy, cur_rank_file_id, total_files_num = recv_payload[0]

    cur_rank_shards = {
        shard_id: (local_shard_map[shard_id], rank_group)
        for shard_id, rank_group in assigned_shards.items()
    }

    return BalancedShardPlan(
        cur_rank_shards=cur_rank_shards,
        param_redundancy=param_redundancy,
        total_files_num=total_files_num,
        cur_rank_file_id=cur_rank_file_id,
        full_assignment=full_assignment
    )


def _build_balanced_shard_plan_with_gather(
        network: List[Cell],
        filter_func: Callable[[str], bool] = None
) -> BalancedShardPlan:
    """
    Builds the balanced shard plan for PyNative multi-rank scenarios.

    Core workflow:
    1. Every rank builds its own ShardedTensor metadata locally (purely local operation).
    2. The global ShardedTensor metadata is collected onto rank 0 via
       `get_all_sharded_tensor_on_rank0` (each rank only contributes its parameter names
       plus the layouts it owns, so the gathered payload is deduplicated cluster-wide),
       where the shard coverage and the balanced assignment are computed with the same
       `distribute_shards` algorithm used by the legacy implementation, producing identical
       assignment results.
    3. Rank 0 scatters each rank's assignment (shard IDs with their rank groups), its
       redundant parameter groups, and its checkpoint file numbering back
       (`scatter_object_list`).
    4. Every rank rebuilds its assigned shards from its own local ShardedTensor metadata.

    Only rank 0 additionally keeps the complete shard-level assignment (without ShardedTensor
    objects) for writing 'metadata.json' in the balanced saving flow.
    """
    local_rank = get_real_rank()
    world_size = get_real_group_size()

    local_shard_map = _build_local_shard_map(network, filter_func)
    global_sharded_tensor_metas = get_all_sharded_tensor_on_rank0(network, filter_func)

    scatter_payloads = None
    full_assignment = None
    if local_rank == 0:
        scatter_payloads, full_assignment = _compute_balanced_assignment_on_rank0(
            global_sharded_tensor_metas, world_size
        )

    return _scatter_balanced_shard_plan(scatter_payloads, full_assignment, local_shard_map)


def _build_balanced_shard_plan_from_global_metas(
        network: List[Cell],
        filter_func: Callable[[str], bool] = None,
        global_sharded_tensor_metas: Optional[Dict[int, Dict[str, ShardedTensor]]] = None
) -> BalancedShardPlan:
    """
    Builds the balanced shard plan from the caller-provided global ShardedTensor metadata.

    Same as `_build_balanced_shard_plan_with_gather`, except that rank 0 reuses the global
    ShardedTensor metadata fetched and cached by the caller instead of gathering it again.
    The only collective communication is the `scatter_object_list` of the per-rank
    assignments from rank 0.

    Args:
        network (Cell): A MindSpore Network Cell containing parameters and their associated
            sharding metadata.
        filter_func (Optional[Callable[[str], bool]]): Filter selecting the parameters of
            this file type, applied to each rank's local metadata.
        global_sharded_tensor_metas (Optional[Dict[int, Dict[str, ShardedTensor]]]): The
            global ShardedTensor metadata keyed by rank ID, already filtered to the target
            parameters by the caller. Mandatory on rank 0; None on all other ranks.

    Returns:
        BalancedShardPlan: The per-rank view of the balanced shard distribution.

    Raises:
        ValueError: If `global_sharded_tensor_metas` is None on rank 0.
    """
    local_rank = get_real_rank()
    world_size = get_real_group_size()

    local_shard_map = _build_local_shard_map(network, filter_func)

    scatter_payloads = None
    full_assignment = None
    if local_rank == 0:
        if global_sharded_tensor_metas is None:
            raise ValueError(
                "Building the balanced shard plan in PyNative multi-rank scenarios requires "
                "the global ShardedTensor metadata on rank 0. Please fetch and cache it once "
                "via `get_all_sharded_tensor_on_rank0` before saving, and pass it through."
            )
        scatter_payloads, full_assignment = _compute_balanced_assignment_on_rank0(
            global_sharded_tensor_metas, world_size
        )

    return _scatter_balanced_shard_plan(scatter_payloads, full_assignment, local_shard_map)
