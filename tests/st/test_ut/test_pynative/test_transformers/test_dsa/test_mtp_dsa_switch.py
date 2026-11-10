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
"""Tests for ``mtp_enable_dsa``: keeping DSA on the decoder while the MTP layers run
ordinary attention, which is what graph mode does unconditionally."""
from types import SimpleNamespace

import pytest

from mindformers.parallel_core.transformer_config import MLATransformerConfig
from mindformers.pynative.base_models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_mtp_block_spec,
)
from mindformers.pynative.layers.flash_attention import FlashAttention
from mindformers.pynative.transformers.multi_latent_attention import MLASelfAttention
from mindformers.pynative.transformers.transformer_layer import (
    HyperConnectionTransformerLayer,
    TransformerLayer,
)
from mindformers.pynative.transformers.experimental_attention_variant.dsa import DSAttention
from mindformers.pynative.transformers.experimental_attention_variant.dsa_attention import (
    DSASelfAttention,
)
from mindformers.pynative.transformers.experimental_attention_variant.dsa_transformer_layer import (
    DSAHyperConnectionTransformerLayer,
    DSATransformerLayer,
)


def _config(**overrides):
    """Build a minimal DSA-shaped config with one MTP layer."""
    kwargs = {
        "num_layers": 2,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "ffn_hidden_size": 128,
        "multi_latent_attention": True,
        "kv_lora_rank": 512,
        "qk_pos_emb_head_dim": 64,
        "mtp_num_layers": 1,
        "experimental_attention_variant": "dsa",
        "dsa_indexer_n_heads": 64,
        "dsa_indexer_head_dim": 128,
    }
    kwargs.update(overrides)
    return MLATransformerConfig(**kwargs)


class TestConfigResolution:
    """``mtp_enable_dsa`` defaults to whether the model runs DSA at all."""

    @pytest.mark.parametrize("variant", ["dsa", None])
    def test_unset_follows_the_attention_variant(self, variant):
        """Leaving it unset must not change behaviour for existing configs."""
        config = _config(experimental_attention_variant=variant)
        assert config.mtp_enable_dsa is (variant == "dsa")

    def test_explicit_false_survives_post_init(self):
        """The whole point of the field: decoder DSA on, MTP DSA off."""
        config = _config(mtp_enable_dsa=False)
        assert config.experimental_attention_variant == "dsa"
        assert config.mtp_enable_dsa is False

    def test_mtp_dsa_without_a_dsa_model_is_rejected(self):
        """MTP cannot run DSA when the model does not -- there is no indexer config to build it
        from, and silently ignoring the request is what this whole switch exists to prevent."""
        with pytest.raises(ValueError, match="requires experimental_attention_variant"):
            _config(experimental_attention_variant=None, mtp_enable_dsa=True)


def _specs(config):
    """The decoder block spec and the transformer layer wrapped inside the MTP layer."""
    decoder_spec = get_gpt_decoder_block_spec(config)
    mtp_spec = get_gpt_mtp_block_spec(config, decoder_spec)
    return decoder_spec, mtp_spec.layer_specs[0].submodules.transformer_layer


class TestMtpLayerSpec:
    """Both the layer class and the self-attention spec carry DSA, so both must be swapped."""

    def test_mtp_follows_the_decoder_by_default(self):
        _, inner = _specs(_config())
        assert inner.module is DSATransformerLayer
        assert inner.submodules.self_attention.module is DSASelfAttention

    def test_mtp_drops_dsa_when_disabled(self):
        decoder_spec, inner = _specs(_config(mtp_enable_dsa=False))
        # The decoder keeps DSA ...
        last_decoder = decoder_spec.layer_specs[-1]
        assert last_decoder.module is DSATransformerLayer
        assert last_decoder.submodules.self_attention.submodules.core_attention.module is DSAttention
        # ... while the MTP layer runs plain MLA, so no indexer parameter under ``mtp.``.
        assert inner.module is TransformerLayer
        assert inner.submodules.self_attention.module is MLASelfAttention
        assert inner.submodules.self_attention.submodules.core_attention is FlashAttention

    def test_mlp_and_norms_are_untouched(self):
        """Only attention differs between the two flavours; the MTP layer still mirrors the
        last decoder layer everywhere else."""
        decoder_spec, inner = _specs(_config(mtp_enable_dsa=False))
        last_decoder = decoder_spec.layer_specs[-1]
        assert inner.submodules.mlp == last_decoder.submodules.mlp
        assert inner.submodules.input_layernorm is last_decoder.submodules.input_layernorm
        assert inner.submodules.pre_mlp_layernorm is last_decoder.submodules.pre_mlp_layernorm


class TestCombinedWithHyperConnections:
    """The two MTP switches are orthogonal and compose in either order."""

    def test_dsa_off_keeps_mhc(self):
        _, inner = _specs(_config(enable_hyper_connections=True, num_residual_streams=2,
                                  mtp_enable_dsa=False))
        assert inner.module is HyperConnectionTransformerLayer
        assert inner.submodules.self_attention.module is MLASelfAttention

    def test_mhc_off_keeps_dsa(self):
        _, inner = _specs(_config(enable_hyper_connections=True, num_residual_streams=2,
                                  mtp_enable_hyper_connections=False))
        assert inner.module is DSATransformerLayer
        assert inner.submodules.self_attention.module is DSASelfAttention

    def test_both_off_is_a_plain_layer(self):
        decoder_spec, inner = _specs(_config(enable_hyper_connections=True, num_residual_streams=2,
                                             mtp_enable_hyper_connections=False,
                                             mtp_enable_dsa=False))
        assert decoder_spec.layer_specs[-1].module is DSAHyperConnectionTransformerLayer
        assert inner.module is TransformerLayer
        assert inner.submodules.self_attention.module is MLASelfAttention


@pytest.mark.parametrize("sparse,uses_dsa,training,expected", [
    (False, True, True, True), (True, True, True, True),
    (False, False, True, False), (True, False, True, False),
    (False, True, False, False), (True, True, False, False),
])
def test_mtp_early_backward_respects_stage_and_dsa_switch(sparse, uses_dsa, training, expected):
    """The merged two-stage entry stays inactive for plain MTP and evaluation."""
    from mindformers.pynative.transformers.multi_token_prediction import MultiTokenPredictionLayer

    layer = SimpleNamespace(training=training, uses_dsa=uses_dsa,
                            config=_config(dsa_indexer_layerwise_backward=True,
                                           dsa_indexer_use_sparse_loss=sparse))
    assert MultiTokenPredictionLayer._layerwise_indexer_backward(layer) is expected
