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
"""Manual KDA TP/CP regression; pending HP state APIs keep this worker outside CI."""

import argparse

import numpy as np
import mindspore as ms
from mindspore import Tensor, mint, nn
from mindspore.communication import init, get_rank, get_group_size
from mindspore.graph.api import _no_grad
from mindspore.ops.communication import all_reduce

from hyper_parallel import DeviceMesh, SkipDTensorDispatch
from hyper_parallel.core.dtensor.layout import _get_slice_tensor_by_layout
from hyper_parallel.core.fully_shard.api import fully_shard, MixedPrecisionPolicy
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.spec_utils import build_module
from mindformers.parallel_core.utils.model_mixin import TrainModelMixin
from mindformers.pynative.base_models.gpt.kda import get_kda_module_spec
from mindformers.pynative.base_models.gpt.parallelize import _apply_layers_tp, _build_layer_fsdp_policy
from mindformers.pynative.distributed.context_parallel import build_context_parallel_attention_style


def weighted_loss(cell, weights):
    """Bind one step's loss weights before constructing its gradient function."""
    def loss(value):
        return (cell(value) * weights).sum()
    return loss


def assert_close(actual, expected, atol, rtol, name="tensor"):
    """Reject nonfinite values as well as deviations from the baseline."""
    actual = actual.astype(ms.float32).asnumpy()
    expected = expected.astype(ms.float32).asnumpy()
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), name
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=name)


def check_convolution_layout(core, tokens, heads):
    """Require the intended local sequence/head layout at convolution entry."""
    original = core._convolve

    def checked(query, key, value, weight):
        """Reject a regression that moves convolution ahead of CP redistribution."""
        assert query.shape[1:3] == key.shape[1:3] == value.shape[1:3] == (tokens, heads), query.shape
        assert weight.shape[1] == heads * query.shape[-1], weight.shape
        return original(query, key, value, weight)

    core._convolve = checked


def check_tensor_parallel(tp_mesh, cp_mesh, lora=False, bias=False, fsdp=False,
                          method="ulysses", degree=None, heads=4, tokens=64, batch=2, future_only=False):
    """Exercise production TP setup and compare local shards with two serial SGD steps."""
    config = TransformerConfig(
        hidden_size=8, num_layers=1, num_attention_heads=heads,
        experimental_attention_variant="kda", linear_conv_kernel_dim=4,
        linear_key_head_dim=128, linear_value_head_dim=128, linear_num_key_heads=heads, linear_num_value_heads=heads,
        compute_dtype="bfloat16", params_dtype="float32", add_bias_linear=bias,
        kda_f_lora_rank=3 if lora else None, kda_gate_lora_rank=5 if lora else None,
    )
    with ms.DeviceCtx("meta"):
        parallel = build_module(get_kda_module_spec(), config=config)
    base = build_module(get_kda_module_spec(), config=config)
    rng = np.random.default_rng(142)
    block = nn.Cell()
    block.self_attention = parallel
    block.input_layernorm, block.pre_mlp_layernorm, block.mlp = nn.Identity(), nn.Identity(), nn.Identity()
    with ms.DeviceCtx("meta"):
        _apply_layers_tp(block, tp_mesh, enable_ep=False)
    if cp_mesh.size() > 1:
        build_context_parallel_attention_style(
            method=method, cp_size=cp_mesh.size(), attention_variant="kda", ulysses_degree_in_cp=degree,
        )._apply(parallel.core_attention, cp_mesh)
    if fsdp:
        enable_mindspore_backward_compat()
        shard_plan, replicas = _build_layer_fsdp_policy(block, cp_mesh.size())
        for param in (parallel.in_proj.weight, parallel.conv1d.weight):
            assert shard_plan(param).dim == 1 and all(id(param) != id(replica) for replica in replicas)
        with ms.DeviceCtx("meta"):
            fully_shard(parallel, mesh=cp_mesh, reshard_after_forward=False,
                        shard_placement_fn=shard_plan, replicate_params=replicas,
                        mp_policy=MixedPrecisionPolicy(param_dtype=ms.float32, reduce_dtype=ms.float32,
                                                       output_dtype=ms.float32, cast_forward_inputs=False))
        parallel.set_reduce_op_type("sum")
    class InitModel(nn.Cell, TrainModelMixin):
        """Use the same materialization/reset entry points as Trainer."""
        def __init__(self):
            super().__init__()
            self.model = parallel

    initializer = InitModel()
    initializer.to_empty()
    initialized = {}
    for _ in range(2):
        ms.set_seed(142)
        ms.manual_seed(142)
        with _no_grad():
            initializer.init_states()
        for name in ("A_log", "dt_bias"):
            value = getattr(parallel, name).full_tensor()
            mean, _ = all_reduce(value / get_group_size())
            assert_close(value, mean, 1e-6, 1e-6, name + " replicas")
            if name in initialized:
                assert_close(value, initialized[name], 0, 0, name + " reseed")
            initialized[name] = value.copy()
            sample = value.asnumpy()
            if name == "dt_bias":
                assert np.unique(sample, axis=0).shape[0] == sample.shape[0], name + " repeated shards"
    step = mint.nn.functional.softplus(initialized["dt_bias"]).asnumpy()
    assert (step >= .001 - 1e-7).all() and (step <= .1 + 1e-7).all()
    assert not initialized["A_log"].asnumpy().any()
    for (_, source), (_, target) in zip(base.parameters_and_names(), parallel.parameters_and_names()):
        source.set_data(target.full_tensor())
    # Independent pre-TP flat parameter layout for numerical/gradient comparison.
    base.in_proj.weight = ms.Parameter(base.in_proj.weight.reshape(-1, config.hidden_size), name="weight")
    if base.in_proj.bias is not None:
        base.in_proj.bias = ms.Parameter(base.in_proj.bias.reshape(-1), name="bias")
    base.in_proj.output_groups = 1
    base.conv1d.weight = ms.Parameter(base.conv1d.weight.reshape(-1, 1, 4), name="weight")
    head_degree = cp_mesh.size() if method == "ulysses" else (1 if method == "colossal" else degree)
    check_convolution_layout(parallel.core_attention, tokens // (cp_mesh.size() // head_degree),
                             heads // tp_mesh.size() // head_degree)
    original_params, parallel_params = tuple(base.trainable_params()), tuple(parallel.trainable_params())
    cp_tokens = tokens // cp_mesh.size()
    local_tokens = cp_tokens // tp_mesh.size()
    start = cp_mesh.get_local_rank() * cp_tokens + tp_mesh.get_local_rank() * local_tokens
    stop = start + local_tokens
    for _ in range(2):
        inputs = Tensor(rng.normal(0, .2, (tokens, batch, 8)).astype(np.float32), ms.bfloat16)
        weight_array = rng.normal(size=(tokens, batch, 8)).astype(np.float32)
        if future_only:
            weight_array[:-cp_tokens] = 0
        weights = Tensor(weight_array) / (tokens * batch * 8)
        assert_close(parallel(inputs[start:stop]), base(inputs)[start:stop], 2e-3, 2e-2)
        if fsdp:
            for parameter in original_params:
                parameter.grad = None
            base_inputs = inputs.copy()
            base_inputs.requires_grad = True
            weighted_loss(base, weights)(base_inputs).backward()
            base_input_grad = base_inputs.grad
            base_grads = tuple(parameter.grad for parameter in original_params)
            parallel.zero_grad()
            local_inputs = inputs[start:stop].copy()
            local_inputs.requires_grad = True
            weighted_loss(parallel, weights[start:stop])(local_inputs).backward()
            local_input_grad = local_inputs.grad
            local_grads = tuple(parameter.grad for parameter in parallel_params)
        else:
            base_input_grad, base_grads = ms.grad(weighted_loss(base, weights), 0, original_params)(inputs)
            local_input_grad, local_grads = ms.grad(
                weighted_loss(parallel, weights[start:stop]), 0, parallel_params)(inputs[start:stop])
        assert_close(local_input_grad, base_input_grad[start:stop], 3e-5, 3e-2)
        for (name, source), target, full_grad, local_grad in zip(
                base.parameters_and_names(), parallel_params, base_grads, local_grads):
            value = local_grad.to_local() if hasattr(local_grad, "to_local") else local_grad
            if not fsdp:
                if cp_mesh.size() > 1:
                    value, _ = all_reduce(value, group=cp_mesh.get_group())
                if target.placements[0].is_replicate():
                    value, _ = all_reduce(value, group=tp_mesh.get_group())
            expected = _get_slice_tensor_by_layout(full_grad.reshape(target.shape), target.layout)
            assert_close(value, expected, 4e-5, 3e-2, name)
            with _no_grad():
                source.set_data(source - .01 * full_grad)
                with SkipDTensorDispatch():
                    target.to_local().copy_(target.to_local() - .01 * value)
            assert_close(
                target.to_local(), _get_slice_tensor_by_layout(source.reshape(target.shape), target.layout),
                2e-6, 1e-3, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--method", choices=("ulysses", "colossal", "hybrid"), default="ulysses")
    parser.add_argument("--degree", type=int)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--future-only", action="store_true")
    for option in ("fsdp", "lora", "bias"):
        parser.add_argument("--" + option, action="store_true")
    cli_options = parser.parse_args()
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
    init()
    ranks = np.arange(get_group_size()).reshape(-1, cli_options.tp)
    full_mesh = DeviceMesh("npu", ranks, mesh_dim_names=("cp", "tp"))
    check_tensor_parallel(
        full_mesh["tp"], full_mesh["cp"], cli_options.lora,
        cli_options.bias, cli_options.fsdp, cli_options.method, cli_options.degree,
        cli_options.heads, cli_options.tokens, cli_options.batch, cli_options.future_only)
    print("KDA TP/CP INIT AND TWO STEPS PASS", get_rank(), flush=True)
