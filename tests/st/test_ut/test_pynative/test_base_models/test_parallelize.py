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
"""Tests for GPT model parallelization helpers."""

from types import SimpleNamespace

import pytest

from mindformers.pynative.base_models.gpt import parallelize


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tag_dsv4_tp_replicated_grad_norm_params_excludes_sharded_and_indexer_weights():
    """Only already-global full-attention gradients are TP replica-counted."""
    parameters = {
        "linear_proj.weight": SimpleNamespace(),
        "linear_q_up_proj.weight": SimpleNamespace(),
        "core_attention.attn_sink": SimpleNamespace(),
        "core_attention.indexer.weight": SimpleNamespace(),
    }
    unrelated = SimpleNamespace()

    marked_attention = SimpleNamespace(
        _tp_full_attention_replica_count=2,
        parameters_and_names=parameters.items,
    )
    unmarked_cell = SimpleNamespace(
        parameters_and_names=lambda: (("weight", unrelated),)
    )
    model = SimpleNamespace(
        cells_and_names=lambda: (
            ("self_attention", marked_attention),
            ("unrelated", unmarked_cell),
        )
    )
    tag_replicated_params = getattr(
        parallelize, "_tag_dsv4_tp_replicated_grad_norm_params"
    )
    tag_replicated_params(model)

    assert getattr(parameters["linear_proj.weight"], "_grad_norm_replica_count") == 2
    assert getattr(parameters["core_attention.attn_sink"], "_grad_norm_replica_count") == 2
    assert not hasattr(parameters["linear_q_up_proj.weight"], "_grad_norm_replica_count")
    assert not hasattr(parameters["core_attention.indexer.weight"], "_grad_norm_replica_count")
    assert not hasattr(unrelated, "_grad_norm_replica_count")
