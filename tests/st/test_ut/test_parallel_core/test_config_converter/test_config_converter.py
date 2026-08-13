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
"""Test ConfigConverter."""

from copy import deepcopy

import pytest

from mindformers.parallel_core.config_converter import ConfigConverter, ConversionContext
from mindformers.parallel_core.transformer_config import TransformerConfig, MLATransformerConfig
from mindformers.parallel_core.transformer_config_utils import (
    ConfigLogHandler,
    ConfigConversionTracer,
    get_cp_comm_type,
)


class _TestConfigConverter(ConfigConverter):
    """Test ConfigConverter subclass"""
    CONFIG_MAPPING = {
        # Model Architecture
        "head_dim": "kv_channels",
        "attention_bias": "add_qkv_bias",
        "num_hidden_layers": "num_layers",
    }


def _make_minimal_model_config(**overrides):
    """mock model_config input"""
    config = {
        "num_layers": 2,
        "hidden_size": 128,
        "num_attention_heads": 8,
    }
    config.update(overrides)
    return config


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_conversion_context_creation():
    """
    Feature: ConversionContext can create and hold mapping, result, log_handler, tracer correctly.
    Description: Construct ConversionContext with valid parameters.
    Expectation: The attributes are accessible and of the correct type.
    """
    mapping = {"a": "b"}
    reversed_mapping = {"b": ["a"]}
    result = {}
    log_handler = ConfigLogHandler()
    tracer = ConfigConversionTracer()
    ctx = ConversionContext(
        mapping=mapping,
        reversed_mapping=reversed_mapping,
        result=result,
        log_handler=log_handler,
        tracer=tracer,
    )
    assert ctx.mapping is mapping
    assert ctx.reversed_mapping is reversed_mapping
    assert ctx.result is result
    assert ctx.log_handler is log_handler
    assert ctx.tracer is tracer


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_convert_basic_returns_transformer_config():
    """
    Feature: ConfigConverter.convert returns TransformerConfig by default.
    Description: Use the ConfigConverter subclass to convert a valid model_config.
    Expectation: TransformerConfig is returned, and the key fields are consistent with the input (after mapping).
    """
    model_config = deepcopy(_make_minimal_model_config())
    result = _TestConfigConverter.convert(model_config, is_mla_model=False)
    assert isinstance(result, TransformerConfig)
    assert result.num_layers == 2
    assert result.hidden_size == 128
    assert result.num_attention_heads == 8


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_convert_use_rotary_position_ids_is_independent_from_eod_reset():
    """PyNative explicit rotary positions use their own model-side configuration."""
    model_config = deepcopy(_make_minimal_model_config(
        use_rotary_position_ids=True,
        use_eod_reset=False,
    ))
    result = _TestConfigConverter.convert(model_config, is_mla_model=False)

    assert result.use_rotary_position_ids is True
    assert result.use_eod_reset is False


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_convert_is_mla_model_returns_mla_transformer_config():
    """
    Feature: ConfigConverter.convert returns MLATransformerConfig when is_mla_model=True.
    Description: Transfer is_mla_model=True and multi_latent_attention.
    Expectation: MLATransformerConfig is returned and multi_latent_attention is True.
    """
    model_config = deepcopy(_make_minimal_model_config(multi_latent_attention=True))
    result = _TestConfigConverter.convert(model_config, is_mla_model=True)
    assert isinstance(result, MLATransformerConfig)
    assert result.multi_latent_attention is True


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_convert_parallel_config_nested():
    """
    Feature: _pre_process supports key-by-key application mapping when recompute in parallel_config is a dictionary.
    Description: The value of parallel_config.recompute is dict, and the key is in the mapping.
    Expectation: The conversion is complete and no error is reported. (If the recompute internal key is used in
    the mapping, the recompute internal key will be applied.)
    """
    model_config = deepcopy(_make_minimal_model_config())
    model_config["parallel_config"] = {
        "model_parallel": 4,
        "pipeline_stage": 2,
        "recompute": {"recompute": True},
        "context_parallel_algo": "colossalai_cp",
    }
    result = _TestConfigConverter.convert(model_config, is_mla_model=False)
    assert isinstance(result, TransformerConfig)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_convert_context_parallel_algo_transform():
    """
    Feature: The mapping rule with transform (such as context_parallel_algo -> cp_comm_type) is correctly applied.
    Description: When context_parallel_algo is set to ulysses_cp, a2a should be obtained through get_cp_comm_type.
    Expectation: The value of cp_comm_type in the conversion result is a2a.
    """
    model_config = deepcopy(_make_minimal_model_config(context_parallel_algo="ulysses_cp"))
    result = _TestConfigConverter.convert(model_config, is_mla_model=False)
    assert result.cp_comm_type == "a2a"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_convert_context_parallel_algo_invalid_raises():
    """
    Feature: When the value of context_parallel_algo is invalid, get_cp_comm_type throws an error. As a result,
    the conversion record is incorrect and check_and_raise throws ValueError.
    Description: Set context_parallel_algo to an unsupported value.
    Expectation: ValueError is thrown when convert is invoked.
    """
    model_config = deepcopy(_make_minimal_model_config(context_parallel_algo="invalid_cp"))
    with pytest.raises(ValueError):
        _TestConfigConverter.convert(model_config, is_mla_model=False)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_pre_process_without_parallel_config():
    """
    Feature: When parallel_config does not exist in model_config, _pre_process does not report an error
    and does not modify the result.
    Description: Normal conversion of model_config without the parallel_config key.
    Expectation: The conversion is successful, and the top-level model_parallel is used for parallel correlation.
    """
    model_config = deepcopy(_make_minimal_model_config())
    assert "parallel_config" not in model_config or model_config.get("parallel_config") is None
    result = _TestConfigConverter.convert(model_config, is_mla_model=False)
    assert result.tensor_model_parallel_size == 1


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_convert_instantiation_type_error_raises_runtime_error():
    """
    Feature:  If the result cannot be instantiated as TransformerConfig (e.g., due to invalid keys), raise RuntimeError.
    Description: By injecting TransformerConfig with a nonexistent key into result through subclasses,
    the config_cls(**result) raises a TypeError.。
    Expectation: Raise a RuntimeError with the cause being a TypeError.
    """

    class BadConverter(ConfigConverter):
        CONFIG_MAPPING = {}

        @classmethod
        def _pre_process(cls, model_config, ctx):
            super()._pre_process(model_config, ctx)
            ctx.result["_nonexistent_key_xyz"] = 1

    model_config = deepcopy(_make_minimal_model_config())
    with pytest.raises(RuntimeError) as exc_info:
        BadConverter.convert(model_config, is_mla_model=False)
    assert isinstance(exc_info.value.__cause__, TypeError)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_cp_comm_type_values():
    """
    Feature: The mapping of get_cp_comm_type to context_parallel_algo is correct.
    Description: colossalai_cp -> all_gather, ulysses_cp -> a2a.
    Expectation: The returned value is the same as expected. If the value is invalid, the valueError is thrown.
    """
    assert get_cp_comm_type("colossalai_cp") == "all_gather"
    assert get_cp_comm_type("ulysses_cp") == "a2a"
    with pytest.raises(ValueError):
        get_cp_comm_type("unknown")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_get_final_mapping_has_scalar_specs():
    """
    Feature: _get_final_mapping returns a 1:1 mapping (each source key maps to exactly one target spec).
    Description: Inspect the structure of the mapping returned by _get_final_mapping.
    Expectation: Every value is a scalar — either a str (target_key) or a (target_key, transform_func) tuple —
                 never a list. This guards the invariant relied on by _apply_mapping_rule and _get_reversed_mapping.
    """
    log_handler = ConfigLogHandler()
    mapping = ConfigConverter._get_final_mapping(log_handler, is_mla_model=False)

    assert mapping, "mapping should not be empty"
    for src_key, target_spec in mapping.items():
        # Must NOT be a list — that would indicate the old multi-spec structure.
        assert not isinstance(target_spec, list), (
            f"source key '{src_key}' maps to a list {target_spec}; expected a scalar spec"
        )
        # Must be either a plain target_key string or a (target_key, transform_func) tuple.
        if isinstance(target_spec, tuple):
            assert len(target_spec) == 2, (
                f"source key '{src_key}' has tuple spec {target_spec} of wrong length; expected (target_key, func)"
            )
            target_key, trans_func = target_spec
            assert isinstance(target_key, str) and callable(trans_func), (
                f"source key '{src_key}' has malformed tuple spec {target_spec}"
            )
        else:
            assert isinstance(target_spec, str), (
                f"source key '{src_key}' has unexpected spec type {type(target_spec).__name__}: {target_spec}"
            )
