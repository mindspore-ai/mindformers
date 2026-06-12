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
"""Run Multi-Token Prediction (MTP) accuracy test with configurable parameters via args.

Follows the same pattern as ``run_mlp.py``.
"""
import argparse

import numpy as np
import mindspore as ms
from mindspore import Tensor, mint, ops

from data_gen_utils import get_init_params, get_golden_datas, get_gpu_datas
from tests.utils.double_benchmark import DoubleBenchmarkStandard, DoubleBenchmarkComparator
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.spec_utils import ModuleSpec
from mindformers.pynative.transformers.multi_token_prediction import (
    MultiTokenPredictionBlock,
    MultiTokenPredictionBlockSubmodules,
    MultiTokenPredictionLayer,
    MultiTokenPredictionLayerSubmodules,
    roll_tensor,
    save_to_mtp_losses_tracker,
    track_mtp_metrics,
)
from mindformers.pynative.transformers.attention import SelfAttention, SelfAttentionSubmodules
from mindformers.pynative.transformers.mlp import MLP, MLPSubmodules
from mindformers.pynative.transformers.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from mindformers.pynative.layers.linear import Linear
from mindformers.pynative.layers.layer_norm import FusedRMSNorm
from mindformers.pynative.layers.flash_attention import FlashAttention
from mindformers.pynative.layers.identity_op import IdentityOp
from mindformers.pynative.base_models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding,
)
from mindformers.pynative.loss import CrossEntropyLoss


def _build_mtp_block_spec():
    """Build :class:`ModuleSpec` for the small-scale MTP test."""
    transfmr_spec = ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=FusedRMSNorm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                submodules=SelfAttentionSubmodules(
                    linear_qkv=Linear,
                    core_attention=FlashAttention,
                    linear_proj=Linear,
                ),
            ),
            pre_cross_attn_layernorm=IdentityOp,
            cross_attention=IdentityOp,
            pre_mlp_layernorm=FusedRMSNorm,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(linear_fc1=Linear, linear_fc2=Linear),
            ),
        ),
    )
    mtp_layer_spec = ModuleSpec(
        module=MultiTokenPredictionLayer,
        submodules=MultiTokenPredictionLayerSubmodules(
            enorm=FusedRMSNorm,
            hnorm=FusedRMSNorm,
            eh_proj=Linear,
            transformer_layer=transfmr_spec,
            layer_norm=FusedRMSNorm,
        ),
    )
    return ModuleSpec(
        module=MultiTokenPredictionBlock,
        submodules=MultiTokenPredictionBlockSubmodules(
            layer_specs=[mtp_layer_spec],
        ),
    )


class MTPRunner:
    """Class to manage MTP model, weights, and golden-data comparison."""

    def __init__(self, args_from_parser):
        self.args = args_from_parser

        self.config = TransformerConfig(
            hidden_size=self.args.hidden_size,
            seq_length=self.args.seq_length,
            num_attention_heads=self.args.num_heads,
            num_layers=1,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            add_bias_linear=False,
            mtp_loss_scaling_factor=self.args.mtp_loss_scaling_factor,
            mtp_num_layers=self.args.mtp_num_layers,
            vocab_size=self.args.vocab_size,
            compute_dtype=self.args.compute_dtype,
            ffn_hidden_size=self.args.ffn_hidden_size,
            gated_linear_unit=False,
            position_embedding_type=self.args.position_embedding_type,
        )
        self.input_data, self.state_dict = get_init_params(self.config)

    def build_model(self):
        """Build and initialize MTP model with embedding and output layers."""
        spec = _build_mtp_block_spec()
        mtp_block = MultiTokenPredictionBlock(self.config, spec)
        ms.load_param_into_net(mtp_block, self.state_dict, strict_load=False)

        emb_layer = LanguageModelEmbedding(
            config=self.config,
            vocab_size=self.config.vocab_size,
            max_sequence_length=self.config.seq_length,
            position_embedding_type=self.config.position_embedding_type,
        )
        ms.load_param_into_net(emb_layer, self.state_dict, strict_load=False)

        output_layer = Linear(
            self.config.hidden_size,
            self.config.vocab_size,
            compute_dtype=self.config.compute_dtype,
            params_dtype=self.config.params_dtype,
            bias=False,
        )
        ms.load_param_into_net(output_layer, self.state_dict, strict_load=False)

        return mtp_block, emb_layer, output_layer

    def run(self):
        """Run the MTP forward pass and compare against golden data."""
        mtp_block, emb_layer, output_layer = self.build_model()
        ce_loss = CrossEntropyLoss(False)

        hidden_states = mtp_block(
            Tensor(self.input_data["input_ids"], dtype=ms.int32),
            Tensor(self.input_data["position_ids"], dtype=ms.int32),
            self.input_data["hidden_states"],
            Tensor(self.input_data["attention_mask"], dtype=ms.bool_),
            embedding=emb_layer,
        )

        hs_list = mint.chunk(hidden_states, 1 + self.config.mtp_num_layers, dim=0)
        mtp_labels = Tensor(self.input_data["labels"], dtype=ms.int32)
        loss_mask_t = Tensor(self.input_data["loss_mask"], dtype=ms.float32)

        for i in range(self.config.mtp_num_layers):
            logits = output_layer(hs_list[i + 1], weight=self.state_dict["weight"])
            logits = logits.transpose(0, 1).reshape((-1, logits.shape[-1]))
            mtp_labels = roll_tensor(mtp_labels, shifts=-1, dims=-1)
            loss_mask_t = roll_tensor(loss_mask_t, shifts=-1, dims=-1)
            mask_flat = loss_mask_t.reshape(-1)
            mask_flat = ops.cast(mask_flat, ms.float32)
            loss = ce_loss(logits, mtp_labels.reshape(-1))
            loss = mint.mul(loss, mask_flat)
            loss_sum = loss.sum()
            num_tokens = mask_flat.sum()
            scale = self.config.mtp_loss_scaling_factor / self.config.mtp_num_layers
            save_to_mtp_losses_tracker(
                scale * loss_sum / num_tokens, i, self.config.mtp_num_layers,
            )

        mtp_metrics = track_mtp_metrics()
        assert mtp_metrics is not None, "track_mtp_metrics returned None"
        assert not ops.isnan(mtp_metrics).any(), "MTP metrics contain NaN"
        assert not ops.isinf(mtp_metrics).any(), "MTP metrics contain Inf"

        output_npu = mtp_metrics.asnumpy().astype(np.float32)
        standard = DoubleBenchmarkStandard(dtype="bfloat16")
        output_gpu = get_gpu_datas()
        output_golden = get_golden_datas()
        DoubleBenchmarkComparator.check_pass_or_not(output_npu, output_gpu, output_golden, standard)

        print(
            f"Accuracy Test Case Finished: position_embedding_type={self.args.position_embedding_type}, "
            f"mtp_num_layers={self.args.mtp_num_layers}."
        )


def main():
    parser = argparse.ArgumentParser(description="Run MTP test")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_length", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=16)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=100)
    parser.add_argument("--ffn_hidden_size", type=int, default=64)
    parser.add_argument("--mtp_num_layers", type=int, default=1)
    parser.add_argument("--mtp_loss_scaling_factor", type=float, default=0.5)
    parser.add_argument("--position_embedding_type", type=str, default="learned_absolute",
                        choices=["learned_absolute", "rope"])
    parser.add_argument("--compute_dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    ms.set_deterministic(True)
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_seed(42)

    runner = MTPRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
