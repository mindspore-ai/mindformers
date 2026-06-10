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
"""distributed modules"""
from mindformers.pynative.distributed.style import (
    PrepareModuleInput,
    PrepareModuleOutput,
    PrepareModuleInputOutput,
    ParallelStyle,
)
from mindformers.pynative.distributed.fsdp import (
    get_fsdp_reshard_after_forward_policy,
    disable_fsdp_gradient_division,
)
from mindformers.pynative.distributed.parallelize import (
    parallelize_module,
    parallelize_model,
    register_parallelize_fn,
    register_parallelize,
)
from mindformers.pynative.distributed.pipeline_parallel import PpLayerSetting, StageModelBuilder
from mindformers.pynative.distributed.ep_overlap import OverlapExpertParallel, apply_chunk_overlap_hooks

__all__ = [
    "PrepareModuleInput",
    "PrepareModuleOutput",
    "PrepareModuleInputOutput",
    "ParallelStyle",
    "get_fsdp_reshard_after_forward_policy",
    "disable_fsdp_gradient_division",
    "parallelize_module",
    "parallelize_model",
    "register_parallelize_fn",
    "register_parallelize",
    "PpLayerSetting",
    "StageModelBuilder",
    "OverlapExpertParallel",
    "apply_chunk_overlap_hooks",
]
