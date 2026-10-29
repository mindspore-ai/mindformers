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
"""KDA CP configuration and production GPT parallelization entry points."""

import os
from types import SimpleNamespace

import pytest

if os.getenv("RUN_KDA_TESTS") != "1":
    pytest.skip("Pending HP KDA dependency; opt in with RUN_KDA_TESTS=1.", allow_module_level=True)
# pylint: disable=wrong-import-position
import numpy as np
import mindspore as ms
from mindspore import nn

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.spec_utils import build_module
from mindformers.pynative.base_models.gpt import parallelize
from mindformers.pynative.base_models.gpt.kda import get_kda_module_spec
from mindformers.pynative.distributed.context_parallel import (
    ContextParallelModelIOStyle,
    apply_context_parallel_model_io,
    build_context_parallel_attention_style,
)
from mindformers.pynative.base_models.gpt.gpt_model import GPTModel


def make_config(**kwargs):
    """Construct a valid small KDA model with four heads."""
    values = {"hidden_size": 8, "num_layers": 1, "num_attention_heads": 4,
              "experimental_attention_variant": "kda", "linear_conv_kernel_dim": 4,
              "linear_key_head_dim": 128, "linear_value_head_dim": 128,
              "linear_num_key_heads": 4, "linear_num_value_heads": 4,
              "compute_dtype": "bfloat16", "params_dtype": "float32"}
    return TransformerConfig(**(values | kwargs))


class FakeMesh:
    """Two-rank chronological CP mesh for setup-only tests."""

    rank_list = (0, 1)
    ndim = 1
    mesh = SimpleNamespace(numel=lambda: 2)

    size = staticmethod(lambda: 2)
    get_local_rank = staticmethod(lambda: 0)
    get_group = staticmethod(lambda: "kda_cp_test")


class Model(nn.Cell):
    """Expose the same decoder hierarchy consumed by production parallelization."""

    def __init__(self, config):
        super().__init__()
        self.config, self.decoder = config, nn.Cell()
        self.decoder.layers = nn.CellList([nn.Cell()])
        self.decoder.layers[0].self_attention = build_module(get_kda_module_spec(), config=config)


def test_kda_cp_config_and_tp_boundary():
    """Only TP and the Ulysses subgroup divide heads, not the state CP group."""
    assert make_config(context_parallel_size=2, use_flash_attention=False).context_parallel_size == 2
    assert make_config(context_parallel_size=3).context_parallel_size == 3
    assert make_config(tensor_model_parallel_size=2).tensor_model_parallel_size == 2
    assert make_config(tensor_model_parallel_size=2, context_parallel_size=2).context_parallel_size == 2
    assert make_config(tensor_model_parallel_size=2, context_parallel_size=4).context_parallel_size == 4
    make_config(tensor_model_parallel_size=4, context_parallel_size=16, hidden_size=3584,
                num_attention_heads=32, linear_num_key_heads=32, linear_num_value_heads=32)
    assert build_context_parallel_attention_style("hybrid", 16, 4, attention_variant="kda").hp_style.state_parallel


@pytest.mark.parametrize("low_rank", [None, 3])
def test_legacy_kda_checkpoint_reshape_preserves_elements(tmp_path, low_rank):
    """Old flat weights can be loaded without changing keys or element order."""
    with ms.DeviceCtx("CPU"):
        layer = build_module(get_kda_module_spec(), config=make_config(
            kda_f_lora_rank=low_rank, add_bias_linear=True))
        legacy = {}
        for name in ("in_proj.weight", "in_proj.bias", "conv1d.weight"):
            module_name, parameter_name = name.split(".")
            shape = getattr(getattr(layer, module_name), parameter_name).shape
            flat_shape = (shape[0] * shape[1],) + tuple(shape[2:])
            data = np.arange(np.prod(shape), dtype=np.float32).reshape(flat_shape)
            legacy[name] = ms.Parameter(ms.Tensor(data), requires_grad=False, name=name)
        ms.save_checkpoint(legacy, str(tmp_path / "legacy.ckpt"))
        converted = layer.reshape_legacy_parameters(ms.load_checkpoint(str(tmp_path / "legacy.ckpt")))
        for name, source in legacy.items():
            np.testing.assert_array_equal(converted[name].asnumpy().reshape(-1), source.asnumpy().reshape(-1))
        assert layer.reshape_legacy_parameters(converted).keys() == converted.keys()
        with pytest.raises(ValueError, match="must have shape"):
            layer.reshape_legacy_parameters({"in_proj.weight": ms.Tensor(np.ones((2, 2), np.float32))})


@pytest.mark.parametrize("method,async_enabled,degree", [
    ("hybrid", False, None), ("hybrid", False, 1),
    ("ulysses", True, None), ("ulysses", False, 1),
])
def test_reject_unsupported_kda_cp_setup(monkeypatch, method, async_enabled, degree):
    """Unsupported schedules fail at the actual model entry, before FA lookup."""
    with ms.DeviceCtx("meta"):
        model = Model(make_config())
    monkeypatch.setattr(parallelize, "_unwrap_gptmodel", lambda value: value)
    dims = SimpleNamespace(cp=2, tp_enabled=False)
    parallelism = SimpleNamespace(context_parallel_method=method, context_parallel_async=async_enabled,
                                  ulysses_degree_in_cp=degree)
    with pytest.raises(ValueError, match="KDA|Hybrid|Ulysses"):
        parallelize.apply_context_parallel_attention(model, FakeMesh(), dims, parallelism)


@pytest.mark.parametrize("embedding,explicit_rope,requires_positions,variant", [
    ("none", False, False, "kda"), ("learned_absolute", False, True, "kda"),
    ("rope", True, True, "kda"), ("none", False, False, None),
])
def test_production_setup_preserves_parameter_names_and_maskless_io(
        monkeypatch, embedding, explicit_rope, requires_positions, variant):
    """Exercise the HP CP boundary around convolution and the KDA kernel together."""
    config = make_config(context_parallel_size=2, use_flash_attention=False,
                         position_embedding_type=embedding, use_rotary_position_ids=explicit_rope)
    with ms.DeviceCtx("meta"):
        model = Model(config)
    model.rotary_pos_emb = SimpleNamespace(use_rotary_position_ids=False)
    before = tuple(name for name, _ in model.parameters_and_names())
    config.experimental_attention_variant = variant
    monkeypatch.setattr(parallelize, "_unwrap_gptmodel", lambda value: value)
    dims = SimpleNamespace(cp=2, tp_enabled=False, cp_enabled=True, get_mesh=lambda _: FakeMesh())
    parallelism = SimpleNamespace(context_parallel_method="ulysses", context_parallel_async=False,
                                  ulysses_degree_in_cp=None)
    parallelize.apply_context_parallel_attention(model, FakeMesh(), dims, parallelism)
    attention = model.decoder.layers[0].self_attention
    assert attention.core_attention._forward_pre_hook
    assert not attention.conv1d._forward_pre_hook
    assert before == tuple(name for name, _ in model.parameters_and_names())
    assert model.rotary_pos_emb.use_rotary_position_ids is False
    original = ContextParallelModelIOStyle._apply
    def capture(style, module):
        assert style.require_position_ids == requires_positions
        return original(style, module)
    monkeypatch.setattr(ContextParallelModelIOStyle, "_apply", capture)
    assert apply_context_parallel_model_io(model, dims, parallelism, config) is model
    assert model._forward_pre_hook


def test_kda_model_io_slices_all_token_fields_without_position_ids():
    """KDA needs neither a dense mask nor artificial position_ids to slice a batch."""
    style = ContextParallelModelIOStyle(FakeMesh(), cp_method="ulysses",
                                        build_attention_mask=False, require_position_ids=False)
    inputs = {"input_ids": np.arange(16).reshape(2, 8), "labels": np.arange(16).reshape(2, 8) + 1,
              "loss_mask": np.ones((2, 8)), "decoder_input": np.arange(48).reshape(8, 2, 3)}
    _, result = style._prepare_inputs(None, (), inputs)
    for name in ("input_ids", "labels", "loss_mask"):
        np.testing.assert_array_equal(result[name], inputs[name][:, :4])
    np.testing.assert_array_equal(result["decoder_input"], inputs["decoder_input"][:4])
    assert result.get("attention_mask") is None
    assert result["position_ids"] is None
    with pytest.raises(NotImplementedError, match="custom attention_mask"):
        style._prepare_inputs(None, (), inputs | {"attention_mask": np.ones((8, 8))})


@pytest.mark.parametrize("mixed", [False, True])
def test_gpt_mask_generation_respects_kda_and_fa_layers(mixed):
    """Only mixed models generate a causal mask; explicit masks remain unsupported."""
    attentions = [SimpleNamespace(attention_variant="kda")]
    if mixed:
        attentions.append(SimpleNamespace(attention_variant=None))
    model = SimpleNamespace(
        config=SimpleNamespace(experimental_attention_variant=None),
        cells_and_names=lambda: [(f"layer{i}.self_attention", cell) for i, cell in enumerate(attentions)],
        decoder=SimpleNamespace(layers=[SimpleNamespace(self_attention=cell) for cell in attentions]),
        use_eod_attn_mask_compression=False, use_attn_mask_compression=True,
        casual_mask=lambda: "causal-mask",
    )
    result = GPTModel._preprocess_input_labels_and_masks(model, "tokens", loss_mask="loss-mask")
    assert result[1] == ("causal-mask" if mixed else None)
    with pytest.raises(NotImplementedError, match="explicit attention_mask"):
        GPTModel._preprocess_input_labels_and_masks(model, "tokens", attention_mask="custom", loss_mask="loss-mask")
