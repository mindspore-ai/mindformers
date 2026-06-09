#  Copyright 2026 Huawei Technologies Co., Ltd
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ============================================================================
"""Get reference data."""

import numpy as np

import mindspore as ms

from mindformers.parallel_core.transformer_config import TransformerConfig


def get_init_params(config: TransformerConfig, seq_length=4, batch_size=2):
    """Generate initialization parameters"""
    rng = np.random.default_rng(42)
    data = list(range(seq_length))
    hidden_size = config.hidden_size
    ffn = config.ffn_hidden_size if config.ffn_hidden_size else 4 * hidden_size
    vocab = config.vocab_size
    mlp_size = ffn * 2 if config.gated_linear_unit else ffn

    state_dict = {
        "layers.0.enorm.weight": rng.normal(0, 0.15, size=hidden_size),
        "layers.0.hnorm.weight": rng.normal(0, 0.15, size=hidden_size),
        "layers.0.eh_proj.weight": rng.normal(0, 0.15, size=(hidden_size, hidden_size * 2)),
        "layers.0.final_layernorm.weight": rng.normal(0, 0.15, size=hidden_size),
        "layers.0.transformer_layer.input_layernorm.weight": rng.normal(0, 0.15, size=hidden_size),
        "layers.0.transformer_layer.self_attention.linear_proj.weight": rng.normal(
            0, 0.15, size=(hidden_size, hidden_size)
        ),
        "layers.0.transformer_layer.self_attention.linear_qkv.weight": rng.normal(
            0, 0.15, size=(hidden_size * 3, hidden_size)
        ),
        "layers.0.transformer_layer.pre_mlp_layernorm.weight": rng.normal(0, 0.15, size=hidden_size),
        "layers.0.transformer_layer.mlp.linear_fc1.weight": rng.normal(0, 0.15, size=(mlp_size, hidden_size)),
        "layers.0.transformer_layer.mlp.linear_fc2.weight": rng.normal(0, 0.15, size=(hidden_size, mlp_size)),
        "word_embeddings.weight": rng.normal(0, 0.15, size=(vocab, hidden_size)),
        "position_embeddings.weight": rng.normal(0, 0.15, size=(seq_length, hidden_size)),
        "weight": rng.normal(0, 0.15, size=(vocab, hidden_size)),
    }
    for k in state_dict:
        state_dict[k] = ms.Parameter(ms.tensor(state_dict[k], dtype=ms.float32))

    input_data = {
        "input_ids": np.tile(data, (batch_size, 1)).astype(np.int32),
        "labels": (1 + np.tile(data, (batch_size, 1))).astype(np.int32),
        "position_ids": np.tile(data, (batch_size, 1)).astype(np.int32),
        "loss_mask": np.ones((batch_size, seq_length), dtype=np.float32),
        "attention_mask": np.ones((batch_size, 1, seq_length, seq_length), dtype=bool),
        "hidden_states": rng.normal(0, 0.15, size=(seq_length, batch_size, hidden_size)),
    }
    input_data["hidden_states"] = ms.tensor(input_data["hidden_states"], dtype=ms.bfloat16)

    return input_data, state_dict


def get_golden_datas():
    """Generate golden data for test."""
    return np.array([2.2798521519], dtype=np.float16)


def get_gpu_datas():
    """Generate gpu data for test."""
    return np.array([2.2798521519], dtype=np.float16)
