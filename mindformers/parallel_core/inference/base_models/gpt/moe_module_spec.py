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
"""MoE module spec."""
import os
from typing import Optional

from mindformers.parallel_core.inference.tensor_parallel.batch_invariant_layers import (
    BatchInvariantRowParallelLinear,
    BatchInvariantReplicatedLinear,
    BatchInvariantMergedColumnParallelLinear
)
from mindformers.parallel_core.utils.spec_utils import ModuleSpec
from mindformers.parallel_core.inference.transformer.mlp import MLPSubmodules
from mindformers.parallel_core.inference.transformer.moe.moe_layer import MoELayer, MoESubmodules
from mindformers.parallel_core.inference.transformer.moe.experts import GroupedMLP
from mindformers.parallel_core.inference.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
)
from mindformers.parallel_core.inference.tensor_parallel.layers import (
    MergedColumnParallelLinear,
    RowParallelLinear,
    ReplicatedLinear,
)
from mindformers.parallel_core.inference.tensor_parallel.grouped_layers import (
    ColumnParallelGroupedLinear,
    RowParallelGroupedLinear
)
from mindformers.parallel_core.inference.transformer.moe.router import TopKRouter, BatchInvariantTopKRouter
from mindformers.parallel_core.inference.transformer.moe.shared_experts import SharedExpertMLP, BatchInvariantSharedExpertMLP
from mindformers.parallel_core.inference.utils import is_batch_invariant

def get_moe_module_spec(
        num_experts: Optional[int] = None,
        moe_grouped_gemm: Optional[bool] = True,
        use_alltoall: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for MoE"""
    if not num_experts:
        raise ValueError(f"Using MoE module, num_experts must be int, but num_experts get {num_experts}.")

    if not moe_grouped_gemm:
        raise NotImplementedError("moe_grouped_gemm = 'False' is not supported now.")

    # experts spec
    ## use legacy GroupedMLP
    expert_module = GroupedMLP
    expert_submodule = MLPSubmodules(
        linear_fc1=ColumnParallelGroupedLinear,
        linear_fc2=RowParallelGroupedLinear,
    )

    experts = ModuleSpec(module=expert_module, submodules=expert_submodule)

    # shared experts spec
    if is_batch_invariant():
        shared_expert_module = BatchInvariantSharedExpertMLP
        shared_expert_submodule = MLPSubmodules(
            # When using AlltoAll, shared experts use unsplit linear
            linear_fc1=BatchInvariantReplicatedLinear if use_alltoall else BatchInvariantMergedColumnParallelLinear,
            linear_fc2=BatchInvariantReplicatedLinear if use_alltoall else BatchInvariantRowParallelLinear,
        )
        shared_experts = ModuleSpec(module=shared_expert_module, params={"gate": False}, submodules=shared_expert_submodule)
        moe_submodule = MoESubmodules(
            router=BatchInvariantTopKRouter,
            experts=experts,
            shared_experts=shared_experts,
            token_dispatcher=MoEAlltoAllTokenDispatcher if use_alltoall else MoEAllGatherTokenDispatcher,
        )
    else:
        shared_expert_module = SharedExpertMLP
        shared_expert_submodule = MLPSubmodules(
            # When using AlltoAll, shared experts use unsplit linear
            linear_fc1=ReplicatedLinear if use_alltoall else MergedColumnParallelLinear,
            linear_fc2=ReplicatedLinear if use_alltoall else RowParallelLinear,
        )
        shared_experts = ModuleSpec(module=shared_expert_module, params={"gate": False}, submodules=shared_expert_submodule)
        moe_submodule = MoESubmodules(
            router=TopKRouter,
            experts=experts,
            shared_experts=shared_experts,
            token_dispatcher=MoEAlltoAllTokenDispatcher if use_alltoall else MoEAllGatherTokenDispatcher,
        )

    # MoE module spec
    moe_module_spec = ModuleSpec(
        module=MoELayer,
        submodules=moe_submodule,
    )

    return moe_module_spec
