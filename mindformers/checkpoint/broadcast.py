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
"""broadcast params across redundant rank groups."""
import numpy as np

from mindspore import Tensor
from mindspore.communication import create_group, destroy_group
from mindspore.ops import ReduceOp
from mindspore.communication._comm_helper import _get_group_map, _remove_group_info
from mindspore.runtime import synchronize
from mindspore.mint.distributed import all_gather_object
from mindspore.ops.communication import all_reduce

from mindformers.tools.utils import get_real_rank, get_real_group_size
from mindformers.tools.logger import logger


def _get_sorted_group_map():
    """Get the world group map."""
    group_map = _get_group_map()
    if group_map:
        group_map = {key: group_map[key] for key in sorted(group_map.keys())}
    return group_map


def _get_param_index_in_group(total_param_loaded, group, param):
    """Get param_index in group."""
    param_rank_index = []
    for rank_id in group:
        if rank_id < len(total_param_loaded):
            if param in total_param_loaded[rank_id]:
                param_rank_index.append(rank_id)
        else:
            raise ValueError("rank_id should be smaller than total rank num")
    return param_rank_index


def _remove_param_not_load(param_name, param_not_load):
    """Remove param_name from param_not_load."""
    if param_not_load is not None and param_name in param_not_load:
        param_not_load.remove(param_name)


def _create_allreduce_input(params, group, net_param_dict, total_param_loaded, param_not_load, cur_rank):
    """Creates allreduce input."""
    allreduce_input = []
    for param in params:
        if param not in net_param_dict:
            continue
        param_rank_index = _get_param_index_in_group(total_param_loaded, group, param)
        if not param_rank_index:
            continue
        if len(param_rank_index) == 1:
            real_param = net_param_dict[param]
            _remove_param_not_load(real_param.name, param_not_load)
            if cur_rank != param_rank_index[0]:
                real_param.set_data(Tensor(np.zeros(real_param.shape), dtype=real_param.dtype), real_param.sliced)
            allreduce_input.append(real_param)
        elif len(param_rank_index) > 1:
            raise ValueError(f"For param {param} in group {group} should be in one rank, but in {param_rank_index}.")
    return allreduce_input


def _get_group_name(group_map, group):
    """get group name"""
    group_name = "remove_redundancy" + str(group)
    is_manual_communication_group = True
    if group_map:
        for name, rank_list in group_map.items():
            if list(group) == rank_list:
                group_name = name
                is_manual_communication_group = False
                break
    return group_name, is_manual_communication_group


def _communicate_allreduce(allreduce_input, group_map, group):
    """Communicate allreduce input."""
    if not allreduce_input or len(group) == 1:
        return
    group_name, is_manual_communication_group = _get_group_name(group_map, group)
    if is_manual_communication_group:
        create_group(group_name, list(group))
    try:
        for real_param in allreduce_input:
            value = real_param.to_local() if hasattr(real_param, 'to_local') else real_param
            data = Tensor(value)
            reduced, _ = all_reduce(data, op=ReduceOp.SUM, group=group_name)
            real_param.set_data(reduced, real_param.sliced)
    finally:
        if is_manual_communication_group:
            synchronize()
            destroy_group(group_name)
            _remove_group_info(group_name)


def single_parameter_broadcast(net, param_redundancy, param_not_load, param_loaded):
    """
    Broadcasts unique parameter values across redundant rank groups to eliminate duplicate parameter storage.

    This function coordinates parameter sharing among ranks that hold redundant copies of the same parameters
    (identified by `param_redundancy`). It synchronizes the
    loading status of parameters across all ranks, and performs allreduce communication within redundant groups
    to ensure all ranks in a group access the same parameter values—thus removing redundant parameter storage.

    Core workflow:
    1. Capture the current rank ID.
    2. Retrieve the network's parameter dictionary.
    3. Track the current rank's parameter loading status and synchronize this status across all ranks via allgather.
    4. For each group of ranks with redundant parameters:
        a. Filter parameters to only those present in the network's parameter dictionary.
        b. Create input data for allreduce communication, including valid parameter values from loaded ranks.
        c. Execute allreduce to broadcast consistent parameter values to all ranks in the group.
    5. Synchronize all ranks to complete the process.

    Args:
        net: MindSpore Network instance containing the parameters to be broadcasted and synchronized.
        param_redundancy (dict): Mapping of redundant rank groups (tuples of rank IDs) to lists of parameter keys.
            Each entry represents parameters that are duplicated across the ranks in the corresponding group.
        param_not_load (set): Set of parameter keys that have not been loaded by any rank (excluded from broadcast).
        param_loaded (set): Set tracking rank IDs that have successfully loaded their assigned parameters.
            Updated to include the current rank before synchronizing loading status.
    """
    if get_real_group_size() == 1:
        return
    cur_rank = get_real_rank()
    net_param_dict = net.parameters_dict()
    param_loaded.add(cur_rank)
    total_num = get_real_group_size()
    total_param_loaded = [None] * total_num
    synchronize()
    all_gather_object(total_param_loaded, param_loaded)

    group_map = _get_sorted_group_map()
    for group in sorted(param_redundancy):
        if cur_rank not in group:
            continue
        params = param_redundancy[group]
        logger.debug(f"Rank group: {group}")
        logger.debug(f"AllReduce params: {params}")
        params = sorted(param for param in params if param in net_param_dict)
        allreduce_input = _create_allreduce_input(
            params, group, net_param_dict, total_param_loaded, param_not_load, cur_rank)
        _communicate_allreduce(allreduce_input, group_map, group)
    synchronize()
    logger.info("End loading the parameter broadcast for removing redundant parameters.")
