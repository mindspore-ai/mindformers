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
"""Tests for configurable mHC head selection."""

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor

from mindformers.models.deepseek4.configuration_deepseek_v4 import DeepseekV4Config
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.pynative.base_models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_mtp_block_spec,
)
from mindformers.pynative.layers.identity_op import IdentityOp
from mindformers.pynative.transformers.hyper_connection import HyperConnectionHead
from mindformers.pynative.transformers.transformer_block import (
    TransformerBlock,
    TransformerBlockSubmodules,
)


def _transformer_config(**kwargs):
    """Build a minimal mHC config."""
    defaults = {
        "num_layers": 1,
        "hidden_size": 4,
        "num_attention_heads": 1,
        "ffn_hidden_size": 16,
        "params_dtype": "fp32",
        "compute_dtype": "fp32",
        "layernorm_compute_dtype": "fp32",
        "enable_hyper_connections": True,
        "num_residual_streams": 2,
    }
    defaults.update(kwargs)
    return TransformerConfig(**defaults)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_hc_head_is_disabled_by_default_and_converted():
    """Enabling mHC alone must not silently add learnable head parameters."""
    model_config = DeepseekV4Config(enable_hyper_connections=True, head_dim=128)

    assert model_config.enable_hc_head is False
    assert model_config.convert_to_transformer_config().enable_hc_head is False

    model_config = DeepseekV4Config(
        enable_hyper_connections=True,
        enable_hc_head=True,
        head_dim=128,
    )
    assert model_config.convert_to_transformer_config().enable_hc_head is True


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_hc_head_requires_hyper_connections():
    """A learned head cannot consume the single-stream layout."""
    with pytest.raises(ValueError, match="enable_hc_head requires enable_hyper_connections"):
        _transformer_config(
            enable_hyper_connections=False,
            enable_hc_head=True,
        )


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_hc_head_spec_is_injected_into_decoder_and_mtp():
    """The provider-selected head spec is propagated to both output sites."""
    config = _transformer_config(enable_hc_head=True, mtp_num_layers=1)
    decoder_spec = get_gpt_decoder_block_spec(config, hc_head=HyperConnectionHead)

    assert decoder_spec.hc_head is HyperConnectionHead

    mtp_spec = get_gpt_mtp_block_spec(config, decoder_spec)
    assert mtp_spec.layer_specs[0].submodules.hc_head is HyperConnectionHead


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_configured_hc_head_is_built_from_spec():
    """A provider-selected head spec creates the learnable module explicitly."""
    ms.set_device("CPU")
    config = _transformer_config(enable_hc_head=True)
    spec = TransformerBlockSubmodules(
        layer_specs=[],
        layer_norm=IdentityOp,
        hc_head=HyperConnectionHead,
    )
    block = TransformerBlock(
        config,
        spec,
        pre_process=False,
        post_process=True,
        layer_start=0,
        layer_end=-1,
    )

    assert isinstance(block.hc_head, HyperConnectionHead)
    assert "hc_head.hc_fn.weight" in block.parameters_dict()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_unconfigured_hc_head_falls_back_to_mean_collapse():
    """Without a learned head spec, preserve the original parameter-free collapse."""
    ms.set_device("CPU")
    config = _transformer_config(enable_hc_head=False)
    spec = TransformerBlockSubmodules(
        layer_specs=[],
        layer_norm=IdentityOp,
        hc_head=None,
    )
    block = TransformerBlock(
        config,
        spec,
        pre_process=False,
        post_process=True,
        layer_start=0,
        layer_end=-1,
    )

    packed = np.arange(16, dtype=np.float32).reshape(2, 1, 8)
    actual = block(Tensor(packed), attention_mask=None).asnumpy()
    expected = packed.reshape(2, 1, 2, 4).mean(axis=2)

    assert block.hc_head is None
    np.testing.assert_array_equal(actual, expected)
