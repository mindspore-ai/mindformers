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
"""Unit tests for MindFormers DSA context-parallel style selection."""

import pytest

from hyper_parallel.core.context_parallel.context_parallel import ContextParallel
from hyper_parallel.core.context_parallel.dsa_context_parallel import (
    DSASequenceReplicateCache,
    DSASparseAttentionContextParallel,
)
from mindformers.pynative.distributed.context_parallel import (
    ContextParallelModelIOStyle,
    ContextParallelAttentionStyle,
    DSAContextParallelAttentionStyle,
    build_context_parallel_attention_style,
)
from mindformers.pynative.distributed.style import HPContextParallelAdapter, build_hp_dsa_cp_style


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize(
    "method,cp_size,ulysses_degree,expected_degree",
    [("colossal", 4, None, 1), ("ulysses", 4, None, 4), ("hybrid", 8, 2, 2)],
)
def test_dense_teacher_uses_requested_cp_topology(
        method, cp_size, ulysses_degree, expected_degree
):
    """Dense teacher attention follows the configured CP topology."""
    style = build_hp_dsa_cp_style(
        method=method,
        cp_size=cp_size,
        ulysses_degree_in_cp=ulysses_degree,
        input_layout="TND",
        use_sparse_loss=False,
    )

    assert isinstance(style.attention_style, HPContextParallelAdapter)
    assert isinstance(style.attention_style.hp_style, ContextParallel)
    assert style.attention_style.hp_style.ulysses_degree == expected_degree
    assert style.indexer_loss_style.mode == "colossal"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("async_enabled", [False, True])
def test_dsa2_shares_gathers_only_in_synchronous_mode(async_enabled):
    """Synchronous DSA2 shares global KV; async keeps its producer lifecycle."""
    style = build_hp_dsa_cp_style(
        method="colossal",
        cp_size=4,
        input_layout="TND",
        async_enabled=async_enabled,
        use_sparse_loss=True,
    )

    assert isinstance(style.attention_style, DSASparseAttentionContextParallel)
    cache = style.attention_style.shared_replicate_cache
    if async_enabled:
        assert cache is None
        assert style.attention_style.share_key_value is False
        assert style.indexer_loss_style.shared_replicate_cache is None
    else:
        assert isinstance(cache, DSASequenceReplicateCache)
        assert style.attention_style.share_key_value is True
        assert style.indexer_loss_style.shared_replicate_cache is cache


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_tnd_dsa_model_io_requires_actual_seq_len():
    """TND CP must not silently fall back to an O(S^2) dense-mask path."""
    style = ContextParallelModelIOStyle(
        cp_mesh=object(),
        cp_method="ulysses",
        ulysses_degree_in_cp=4,
        require_actual_seq_len=True,
    )

    with pytest.raises(ValueError, match="requires actual_seq_len"):
        style._prepare_inputs(None, (), {})


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_mla_and_dense_dsa_select_different_ulysses_styles():
    """Generic MLA and dense DSA must retain separate CP ownership."""
    mla_style = build_context_parallel_attention_style(
        method="ulysses",
        cp_size=4,
        ulysses_degree_in_cp=4,
        input_layout="BNSD",
        attention_variant=None,
    )
    dsa_style = build_context_parallel_attention_style(
        method="ulysses",
        cp_size=4,
        ulysses_degree_in_cp=4,
        input_layout="BNSD",
        attention_variant="dsa",
        dsa_use_sparse_loss=False,
    )

    assert isinstance(mla_style, ContextParallelAttentionStyle)
    assert not isinstance(mla_style, DSAContextParallelAttentionStyle)
    assert isinstance(dsa_style, DSAContextParallelAttentionStyle)
