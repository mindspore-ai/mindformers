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
"""Run HyperConnectionModule forward precision check."""

import argparse

import numpy as np
import mindspore as ms
from mindspore import Tensor

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.pynative.transformers.hyper_connection import (
    HyperConnectionHead,
    HyperConnectionModule,
    HyperConnectionOutputCell,
)


def build_config(args):
    """Build a minimal HyperConnectionModule config."""
    return TransformerConfig(
        num_layers=1,
        num_attention_heads=2,
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.hidden_size * 4,
        params_dtype="fp32",
        compute_dtype=args.compute_dtype,
        layernorm_compute_dtype="fp32",
        enable_hyper_connections=True,
        num_residual_streams=args.rate,
        mhc_sinkhorn_iterations=args.sinkhorn_iters,
        mhc_init_gating_factor=args.init_gating_factor,
    )


def sigmoid(x):
    """Numpy sigmoid."""
    return 1.0 / (1.0 + np.exp(-x))


def sinkhorn(logits, iterations):
    """Reference Sinkhorn-Knopp projection."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    matrix = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
    matrix = matrix + 1e-6
    matrix = matrix / (np.sum(matrix, axis=2, keepdims=True) + 1e-6)
    for _ in range(iterations - 1):
        matrix = matrix / (np.sum(matrix, axis=3, keepdims=True) + 1e-6)
        matrix = matrix / (np.sum(matrix, axis=2, keepdims=True) + 1e-6)
    return matrix


def rms_norm(hidden_states, eps):
    """Reference RMSNorm with fixed gamma=1."""
    return hidden_states / np.sqrt(np.mean(np.square(hidden_states), axis=-1, keepdims=True) + eps)


def set_deterministic_params(module, rate):
    """Set deterministic parameters and return their numpy values."""
    params = module.parameters_dict()
    weight = np.linspace(
        -0.08, 0.08, num=np.prod(params["mapping_proj.weight"].shape), dtype=np.float32
    ).reshape(params["mapping_proj.weight"].shape)
    bias = np.linspace(-0.3, 0.3, num=params["bias"].shape[0], dtype=np.float32)

    params["mapping_proj.weight"].set_data(Tensor(weight, dtype=ms.float32))
    params["bias"].set_data(Tensor(bias, dtype=ms.float32))
    params["alpha_pre"].set_data(Tensor(np.array([0.25], dtype=np.float32), dtype=ms.float32))
    params["alpha_post"].set_data(Tensor(np.array([0.5], dtype=np.float32), dtype=ms.float32))
    params["alpha_res"].set_data(Tensor(np.array([0.75], dtype=np.float32), dtype=ms.float32))

    alpha = np.concatenate(
        [
            np.full((rate,), 0.25, dtype=np.float32),
            np.full((rate,), 0.5, dtype=np.float32),
            np.full((rate * rate,), 0.75, dtype=np.float32),
        ],
        axis=-1,
    )
    return weight, bias, alpha


def set_deterministic_head_params(head):
    """Set deterministic learned-head parameters and return their numpy values."""
    params = head.parameters_dict()
    weight = np.linspace(
        -0.05, 0.07, num=np.prod(params["hc_fn.weight"].shape), dtype=np.float32
    ).reshape(params["hc_fn.weight"].shape)
    base = np.linspace(-0.2, 0.3, num=params["hc_base"].shape[0], dtype=np.float32)
    scale = np.array([0.75], dtype=np.float32)
    params["hc_fn.weight"].set_data(Tensor(weight, dtype=ms.float32))
    params["hc_base"].set_data(Tensor(base, dtype=ms.float32))
    params["hc_scale"].set_data(Tensor(scale, dtype=ms.float32))
    return weight, base, scale


def hyper_connection_reference(hidden_states, weight, bias, alpha, config):
    """Compute HyperConnectionModule forward with numpy."""
    seq_len, batch_size, n_hidden = hidden_states.shape
    rate = config.num_residual_streams
    hidden_size = config.hidden_size

    norm_x = rms_norm(hidden_states.astype(np.float32), config.mhc_layernorm_epsilon)
    projected = np.matmul(norm_x.reshape((seq_len, batch_size, 1, n_hidden)), weight.T)
    projected = projected * alpha + bias

    h_pre, h_post, h_res = np.split(projected, [rate, rate * 2], axis=3)
    h_pre = (sigmoid(h_pre) + 1e-6).astype(np.float32)
    h_post = (2.0 * sigmoid(h_post)).reshape((seq_len, batch_size, rate, 1)).astype(np.float32)
    h_res = h_res.reshape((seq_len, batch_size, rate, rate)).astype(np.float32)
    h_res = sinkhorn(h_res, config.mhc_sinkhorn_iterations).astype(np.float32)

    streams = hidden_states.reshape((seq_len, batch_size, rate, hidden_size)).astype(np.float32)
    aggregated = np.squeeze(np.matmul(h_pre, streams), axis=2).astype(np.float32)
    return aggregated, h_res, h_post


def assert_close(name, actual, expected, rtol, atol):
    """Assert arrays are close and report max error."""
    if np.allclose(actual, expected, rtol=rtol, atol=atol):
        return
    diff = np.abs(actual - expected)
    index = np.unravel_index(np.argmax(diff), diff.shape)
    raise AssertionError(
        f"{name} mismatch: max_abs={diff[index]:.8e}, index={index}, "
        f"actual={actual[index]}, expected={expected[index]}"
    )


def run_forward_precision(args):
    """Run one deterministic forward precision case."""
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_deterministic(True)
    ms.set_seed(args.seed)
    np.random.seed(args.seed)

    config = build_config(args)
    module = HyperConnectionModule(config=config, layer_number=1)
    module.reset_parameter()
    module.set_train(False)
    weight, bias, alpha = set_deterministic_params(module, args.rate)

    input_size = args.seq_len * args.batch_size * args.rate * args.hidden_size
    hidden_states = np.linspace(-1.0, 1.0, num=input_size, dtype=np.float32).reshape(
        (args.seq_len, args.batch_size, args.rate * args.hidden_size)
    )

    actual = module(Tensor(hidden_states, dtype=ms.float32))
    actual = tuple(item.asnumpy().astype(np.float32) for item in actual)
    expected = hyper_connection_reference(hidden_states, weight, bias, alpha, config)

    for name, actual_item, expected_item in zip(("aggregated", "h_res", "h_post"), actual, expected):
        assert_close(name, actual_item, expected_item, args.rtol, args.atol)

    # Use a deliberately non-symmetric residual matrix to lock down the
    # MindSpeed convention: output stream j receives sum_i H_res[i, j] x_i.
    output_cell = HyperConnectionOutputCell(args.rate, args.hidden_size, dtype=ms.float32)
    streams = hidden_states.reshape(
        (args.seq_len, args.batch_size, args.rate, args.hidden_size)
    ).astype(np.float32)
    h_res = np.linspace(
        -0.4,
        0.7,
        num=args.seq_len * args.batch_size * args.rate * args.rate,
        dtype=np.float32,
    ).reshape((args.seq_len, args.batch_size, args.rate, args.rate))
    h_post = np.linspace(
        0.1,
        0.9,
        num=args.seq_len * args.batch_size * args.rate,
        dtype=np.float32,
    ).reshape((args.seq_len, args.batch_size, args.rate, 1))
    sublayer_out = np.linspace(
        -0.5,
        0.5,
        num=args.seq_len * args.batch_size * args.hidden_size,
        dtype=np.float32,
    ).reshape((args.seq_len, args.batch_size, args.hidden_size))
    expected_output = (
        np.einsum("sbij,sbih->sbjh", h_res, streams)
        + h_post * np.expand_dims(sublayer_out, axis=2)
    ).reshape((args.seq_len, args.batch_size, args.rate * args.hidden_size))
    actual_output = output_cell(
        Tensor(h_res),
        Tensor(h_post),
        Tensor(hidden_states),
        Tensor(sublayer_out),
    ).asnumpy()
    assert_close("output_cell", actual_output, expected_output, args.rtol, args.atol)

    # The learned head is optional at the model-spec level. When explicitly selected,
    # verify that its numerical path still matches MindSpeed's FP32 formulation.
    head = HyperConnectionHead(config)
    head_weight, head_base, head_scale = set_deterministic_head_params(head)
    flat = hidden_states.astype(np.float32)
    rms_inv = 1.0 / np.sqrt(
        np.mean(np.square(flat), axis=-1, keepdims=True) + config.mhc_layernorm_epsilon
    )
    mixes = np.matmul(flat, head_weight.T) * rms_inv
    pre = sigmoid(mixes * head_scale + head_base) + 1e-6
    expected_head = np.sum(np.expand_dims(pre, -1) * streams, axis=2)
    actual_head = head(Tensor(hidden_states, dtype=ms.float32)).asnumpy()
    assert_close("hc_head", actual_head, expected_head, args.rtol, args.atol)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="HyperConnectionModule forward precision test")
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--rate", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=8)
    parser.add_argument("--sinkhorn_iters", type=int, default=20)
    parser.add_argument("--init_gating_factor", type=float, default=0.01)
    parser.add_argument("--compute_dtype", choices=["float32"], default="float32")
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser.parse_args()


if __name__ == "__main__":
    run_forward_precision(parse_args())
