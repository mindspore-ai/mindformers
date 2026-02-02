# Copyright 2026 Huawei Technologies Co., Ltd
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
# implementation of Expert Parallel for the GroupedMLP in MoE

import numpy as np
import hashlib

from mindspore import mint, nn, ops, Tensor
from mindspore.common import dtype as mstype
from mindspore.communication import get_rank
from mindspore.communication.management import create_group

from hyper_parallel.core.placement_types import Shard
from hyper_parallel.core.device_mesh import DeviceMesh

from mindformers.pynative.distributed.style import ParallelStyle
from mindformers.pynative.distributed.utils import distribute_module
from mindformers.pynative.transformers.moe.experts import GroupedMLP


GROUP_NAME = {}


class ExpertParallel(ParallelStyle):
    """
    Expert Parallel implementation for MoE.

    This class implements expert parallelism for Mixture of Experts (MoE) models,
    distributing different experts across devices for parallel computation.

    Note:
        Only supports `GroupedMLP` modules. Other module types will raise
        `TypeError` exception.

    """

    def __init__(self):
        super().__init__()
        self.ctx = None

        self.cast = ops.cast
        self.shape = ops.shape
        self.reshape = mint.reshape
        self.concat = mint.cat
        self.transpose = mint.transpose
        self.sort = mint.sort
        self.fmod = mint.fmod
        self.index_select = mint.index_select
        self.one_hot = mint.nn.functional.one_hot
        self.sum = mint.sum
        self.cumsum = mint.cumsum
        self.mul = mint.mul
        self.strided_slice = ops.strided_slice

    # performing all-to-all dispatch on the input
    def _token_dispatch(self, device_mesh, cell, args):
        tokens, probs, topk_indices, num_tokens_per_expert = args
        tokens_shape = self.shape(tokens)
        tokens = self.reshape(tokens, (-1, tokens_shape[-1]))

        ep_degree = device_mesh.mesh_shape[0]
        num_experts = num_tokens_per_expert.shape[-1]
        moe_router_topk = topk_indices.shape[-1]
        self.pad_tokens = Tensor(np.zeros((num_experts, tokens.shape[-1])), dtype=tokens.dtype)
        self.pad_probs = Tensor(np.zeros((num_experts, moe_router_topk)), dtype=probs.dtype)
        self.pad_topk_indices = Tensor(
            np.arange(num_experts * moe_router_topk).reshape(num_experts, moe_router_topk) % num_experts,
            dtype=topk_indices.dtype)
        tokens = self.concat((self.pad_tokens, tokens), dim=0)
        probs = self.concat((self.pad_probs, probs), dim=0)
        topk_indices = self.concat((self.pad_topk_indices, topk_indices), dim=0)

        topk_indices_shape = self.shape(topk_indices)
        topk_indices = self.transpose(topk_indices, 1, 0)  # (B*S, k) --> (k, B*S)
        topk_indices = self.reshape(topk_indices, (-1,))  # (k, B*S) --> (k*T,)

        sorted_topk_indices, token_indices_experts_sorted = self.sort(self.cast(topk_indices, mstype.float32), dim=-1)

        _, unsort_token_indices_experts = self.sort(self.cast(token_indices_experts_sorted, mstype.float32), dim=-1)
        unsort_token_indices_experts = self.reshape(unsort_token_indices_experts,
                                                    (topk_indices_shape[1], topk_indices_shape[0]))
        unsort_token_indices_experts = self.transpose(unsort_token_indices_experts, 1, 0)  # (k, B*S) --> (B*S, k)

        inter_map = self.fmod(token_indices_experts_sorted, topk_indices_shape[0])
        index = self.reshape(inter_map, (-1,))
        routed_input = self.index_select(tokens, 0, index)
        routed_input = self.reshape(routed_input, (tokens_shape[0], -1, tokens_shape[-1]))
        num_tokens_per_expert = self.sum(self.one_hot(self.cast(topk_indices, mstype.int32), num_experts), dim=0)
        num_tokens_per_expert = self.cast(num_tokens_per_expert, mstype.float32)

        # generate the input splits and output splits for all-to-all
        self.ep_group = get_ep_group_name(get_rank(), ep_degree)
        num_tokens_per_expert_group = ops.AlltoAll(
            split_count=ep_degree,
            split_dim=-1,
            concat_dim=-1,
            group=self.ep_group
        )(num_tokens_per_expert)

        num_tokens_per_expert_reshaped = self.reshape(num_tokens_per_expert, (ep_degree, -1))
        input_splits = self.cast(self.sum(num_tokens_per_expert_reshaped, dim=-1, keepdim=False), mstype.int64)
        num_tokens_per_expert_group_reshaped = self.reshape(num_tokens_per_expert_group, (ep_degree, -1))
        output_splits = self.cast(self.sum(num_tokens_per_expert_group_reshaped, dim=-1, keepdim=False), mstype.int64)
        num_tokens_per_expert = self.cumsum(self.sum(num_tokens_per_expert_group_reshaped, dim=-2, keepdim=False), 0)
        num_tokens_per_expert = self.cast(num_tokens_per_expert, mstype.int64)

        # perform expert parallel AlltoAll communication
        original_shape = routed_input.shape

        global_input_tokens = ops.AlltoAllV(group=self.ep_group, block_size=cell.hidden_size)(
            self.reshape(routed_input, (-1,)), input_splits, output_splits
        )
        global_input_tokens = self.reshape(global_input_tokens, (1, -1, cell.hidden_size))
        routing_map = self.reshape(self.cast(sorted_topk_indices, mstype.float32), (-1,))
        routing_map = ops.AlltoAllV(group=self.ep_group, block_size=1)(
            routing_map, input_splits, output_splits
        )
        routing_map = self.reshape(routing_map, (1, -1))

        # sort tokens by local expert
        _, sorted_map = self.sort(routing_map)
        _, unsorted_map = self.sort(self.cast(sorted_map, mstype.float32))
        index = self.reshape(sorted_map, (self.shape(sorted_map)[0] * self.shape(sorted_map)[1],))
        global_input_tokens_shape = self.shape(global_input_tokens)
        global_input_tokens = self.reshape(global_input_tokens, (-1, global_input_tokens_shape[-1]))
        global_input_tokens = self.index_select(global_input_tokens, 0, index)
        global_input_tokens = self.reshape(global_input_tokens,
                                           (-1, global_input_tokens_shape[-1]))
        self.ctx = (
            probs, unsorted_map, unsort_token_indices_experts,
            input_splits, output_splits, original_shape
        )
        return global_input_tokens, probs, topk_indices, num_tokens_per_expert

    @staticmethod
    def _get_parameter_shard_plan():
        # shard on the expert dimension
        return {
            "weight1": (Shard(0),),
            "weight2": (Shard(0),)
        }

    # performing all-to-all combine on the output
    def _token_combine(self, device_mesh, cell, args, routed_output):
        probs, unsorted_map, unsort_token_indices_experts, \
            input_splits, output_splits, original_shape = self.ctx
        routed_output = self.reshape(routed_output, (1, -1, cell.hidden_size))
        # unsort tokens by local expert
        index = self.reshape(unsorted_map, (-1,))
        routed_output_shape = routed_output.shape
        routed_output = self.reshape(routed_output, (-1, routed_output_shape[-1]))
        routed_output = self.index_select(routed_output, 0, index)
        routed_output = self.reshape(routed_output, (routed_output_shape[0], -1, routed_output_shape[-1]))

        # perform expert parallel AlltoAll communication
        permutated_local_input_tokens = ops.AlltoAllV(group=self.ep_group, block_size=cell.hidden_size)(
            self.reshape(routed_output, (-1,)), output_splits, input_splits
        )
        permutated_local_input_tokens = self.reshape(permutated_local_input_tokens, original_shape)

        # AlltoAll output to output
        index = self.reshape(unsort_token_indices_experts, (-1,))
        permutated_local_input_tokens = self.reshape(permutated_local_input_tokens,
                                                     (-1, self.shape(permutated_local_input_tokens)[-1]))
        routed_output = self.index_select(permutated_local_input_tokens, 0, index)
        unsort_token_indices_experts_shape = self.shape(unsort_token_indices_experts)
        routed_output = self.reshape(routed_output,
                                     (unsort_token_indices_experts_shape[0], unsort_token_indices_experts_shape[1], -1))
        probs = self.reshape(probs, (self.shape(probs)[0], self.shape(probs)[1], 1))
        routed_output = self.mul(routed_output, self.cast(probs, routed_output.dtype))
        routed_output = self.sum(routed_output, dim=1, keepdim=False)

        routed_output = self.strided_slice(routed_output, (self.pad_topk_indices.shape[0], 0),
                                          (routed_output.shape[0], routed_output.shape[-1]), (1, 1))
        return routed_output

    def _apply(self, module: nn.Cell, device_mesh: DeviceMesh) -> nn.Cell:
        # only supports GroupedMLP
        if not isinstance(module, GroupedMLP):
            raise TypeError(f"Expert parallel only supports GroupedMLP, but got {type(module)}")

        return distribute_module(
            module,
            device_mesh,
            parameter_shard_plan=self._get_parameter_shard_plan(),
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )


class DeredundancyExpertParallel(ExpertParallel):
    """
    Deredundancy Expert Parallel strategy with multi-machine.

    This design reduces communication overhead and improves parallel efficiency.

    Args:
        nums_per_device (int, optional): Number of NPUs per device. Default: ``8``.

    """

    def __init__(self, nums_per_device: int = 8):
        super().__init__()
        self.nums_per_device = nums_per_device
        self.rank_id = get_rank()
        self.local_expert_start_index = None
        self.local_expert_end_index = None

        self.squeeze = mint.squeeze
        self.zeros = mint.zeros
        self.ones = mint.ones
        self.cat = mint.cat
        self.unsqueeze = mint.unsqueeze
        self.logical_and = mint.logical_and
        self.nonzero = mint.nonzero

    def _token_dispatch(self, device_mesh, cell, args):
        tokens, probs, topk_indices, num_tokens_per_expert = args
        ep_degree = device_mesh.mesh_shape[0]
        num_experts = self.shape(num_tokens_per_expert)[0]

        tokens_shape = self.shape(tokens)
        tokens = self.reshape(tokens, (-1, tokens_shape[-1]))
        tokens = self.squeeze(tokens, 0)
        probs = self.cast(self.squeeze(probs, 0), mstype.bfloat16)
        topk_indices = self.cast(self.squeeze(topk_indices, 0), mstype.int32)

        # communication group
        inter_ep = self.nums_per_device
        outer_ep = ep_degree // inter_ep
        node_expert_num = num_experts // outer_ep
        ep_idx = self.rank_id % ep_degree
        self.local_expert_start_index = ep_idx // inter_ep * node_expert_num
        self.local_expert_end_index = self.local_expert_start_index + node_expert_num
        self.oep_group = get_oep_group_name(self.rank_id, ep_degree, inter_ep)
        self.iep_group = get_iep_group_name(self.rank_id, inter_ep)

        top_k = self.shape(topk_indices)[-1]
        node_expert_num = self.local_expert_end_index - self.local_expert_start_index
        safe_tokens = self.zeros((node_expert_num, tokens_shape[-1]), dtype=mstype.bfloat16)
        tokens = self.cat((safe_tokens, tokens), dim=0)
        safe_topk_indices = self.cumsum(self.ones((node_expert_num, top_k)), dim=0)
        safe_topk_indices = self.cast(safe_topk_indices - 1 + self.local_expert_start_index, mstype.int32)
        topk_indices = self.cat((safe_topk_indices, topk_indices), dim=0)
        safe_probs = self.zeros((node_expert_num, top_k), dtype=mstype.bfloat16)
        probs = self.cat((safe_probs, probs), dim=0)

        # prepare counter
        iepones = [node_expert_num // inter_ep for i in range(inter_ep)]
        expert_ids = ops.AllGather(group=self.oep_group)(topk_indices)
        expert_ids = self.reshape(expert_ids, (-1, top_k))
        excounter = self.sum(self.one_hot(self.cast(self.reshape(expert_ids, (-1,)), mstype.int32), num_experts), dim=0)
        excounter = excounter[self.local_expert_start_index: self.local_expert_end_index]
        excounter_reshaped = self.reshape(excounter, (inter_ep, -1))
        local_excounter = ops.AlltoAllV(group=self.iep_group, block_size=1)(excounter.reshape(-1), iepones, iepones)
        local_excounter_reshaped = self.reshape(local_excounter, (inter_ep, -1))
        exrl = self.sum(local_excounter_reshaped, dim=1, keepdim=False)
        exrl = self.cast(exrl, mstype.int64) # [outer_ep]
        exgl = self.cumsum(self.sum(local_excounter_reshaped, dim=-2, keepdim=False), 0, mstype.int32)
        exgl = self.cast(exgl, mstype.int64)
        exsl = self.sum(excounter_reshaped, dim=1, keepdim=False)
        exsl = self.cast(exsl, mstype.int64) # [outer_ep]

        # 1. allgather
        permuted_probs = self.reshape(ops.AllGather(group=self.oep_group)(probs), (-1, top_k))

        # 2. exdispatch
        hidden_size = tokens_shape[-1]
        expert_ids = self.reshape(expert_ids, (-1,)) # (n, k) --> (nk)
        sorted_expert_ids, dispatch_idx = self.sort(self.cast(expert_ids, mstype.float32))
        sorted_expert_ids = self.cast(sorted_expert_ids, mstype.int32)
        permuted_probs = self.reshape(permuted_probs, (-1,)) # (n, k) --> (nk)
        sorted_router_coeff = self.index_select(permuted_probs, 0, dispatch_idx)
        dispatch_idx_floordiv_k = dispatch_idx // top_k
        mask = self.logical_and(
            sorted_expert_ids >= self.local_expert_start_index, sorted_expert_ids < self.local_expert_end_index
        )
        routed_input = ops.AllGather(group=self.oep_group)(tokens)
        routed_input = self.reshape(routed_input, (-1, hidden_size))
        idx = self.reshape(self.nonzero(self.reshape(mask, (-1,))), (-1,))
        dispatch_idx = self.index_select(dispatch_idx_floordiv_k, 0, idx)
        sorted_expert_ids = self.index_select(sorted_expert_ids, 0, idx)
        sorted_router_coeff = self.index_select(sorted_router_coeff, 0, idx)

        excombine_whiteboard = self.mul(routed_input, Tensor(0.0, dtype=mstype.bfloat16))
        routed_input = self.index_select(routed_input, 0, dispatch_idx)

        # 3. inner alltoallv
        routed_input = ops.AlltoAllV(
            group=self.iep_group, block_size=hidden_size
        )(self.reshape(routed_input, (-1,)), exsl, exrl)
        routed_input = self.reshape(routed_input, (-1, hidden_size))
        sorted_expert_ids = ops.AlltoAllV(
            group=self.iep_group, block_size=1
        )(self.reshape(sorted_expert_ids, (-1,)), exsl, exrl)

        # 4. resort
        _, sort_map = mint.sort(self.cast(sorted_expert_ids, mstype.float32))
        _, unsort_map = mint.sort(self.cast(sort_map, mstype.float32))
        routed_input = mint.index_select(routed_input, 0, sort_map)

        self.ctx = (sorted_router_coeff, unsort_map, exrl, exsl, excombine_whiteboard, dispatch_idx, tokens_shape)
        return routed_input, sorted_router_coeff, expert_ids, exgl

    def _token_combine(self, device_mesh, cell, args, routed_output):
        probs, unsorted_map, exrl, exsl, excombine_whiteboard, dispatch_idx, token_orig_shape = self.ctx
        # -4. unresort
        routed_output = self.index_select(routed_output, 0, unsorted_map)

        # -3. allToAllv
        hidden_size = token_orig_shape[-1]
        routed_output = ops.AlltoAllV(
            group=self.iep_group, block_size=hidden_size
        )(self.reshape(routed_output, (-1,)), exrl, exsl)
        routed_output = self.reshape(routed_output, (-1, hidden_size))

        # -2. excombine
        routed_output = self.mul(self.unsqueeze(probs, 1), routed_output)
        routed_output = excombine_whiteboard.index_add_(0, self.reshape(dispatch_idx, (-1,)), routed_output)

        # -1 reduce scatter
        routed_output = ops.ReduceScatter(group=self.oep_group)(routed_output)
        node_expert_num = self.local_expert_end_index - self.local_expert_start_index
        routed_output = routed_output[node_expert_num:]

        return routed_output


def get_group(rank_list):
    """check whether a group has been created."""
    rank_list_str = "-".join([str(i) for i in rank_list])
    if rank_list_str in GROUP_NAME:
        return GROUP_NAME[rank_list_str]

    hashed = hashlib.sha256(rank_list_str.encode()).hexdigest()[:48]
    group_name = str(hashed)
    create_group(group_name, rank_list)
    GROUP_NAME[rank_list_str] = group_name
    return group_name


def get_ep_group_name(rank_id, expert_model_parallel_size):
    """Get expert model parallel group."""
    rank_start = rank_id // expert_model_parallel_size * expert_model_parallel_size
    rand_end = rank_id // expert_model_parallel_size * expert_model_parallel_size + expert_model_parallel_size
    rank_list = list(range(rank_start, rand_end))
    return get_group(rank_list)


def get_oep_group_name(rank_id, expert_model_parallel_size, npu_nums_per_device):
    """
    Generates a unique group name for a set of ranks involved in outer expert partitioning (oep)
    and creates a communication group with this name.
    This method calculates a range of ranks based on the current rank id
    and the expert partition size, hashes this range to create a unique
    identifier, and then establishes a new communication group using this identifier.
    """
    rank_start = rank_id // expert_model_parallel_size * expert_model_parallel_size
    rank_start = rank_start + rank_id % npu_nums_per_device
    rand_end = rank_start + expert_model_parallel_size
    rank_list = list(range(rank_start, rand_end, npu_nums_per_device))
    return get_group(rank_list)


def get_iep_group_name(rank_id, npu_nums_per_device):
    """
    Generates a unique group name for a set of ranks involved in inner expert partitioning (iep)
    and creates a communication group with this name.
    This method calculates a range of ranks based on the current rank id
    and the expert partition size, hashes this range to create a unique
    identifier, and then establishes a new communication group using this identifier.
    """
    rank_start = rank_id // npu_nums_per_device * npu_nums_per_device
    rand_end = rank_start + npu_nums_per_device
    rank_list = list(range(rank_start, rand_end))
    return get_group(rank_list)