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
"""Run MoELayer accuracy test with configurable parameters via args"""
import argparse
import os
from pathlib import Path
import numpy as np
import mindspore as ms
from mindspore import context
from mindspore.communication import init

from data_gen_utils import get_init_params
from mindformers.pynative.transformers.moe.moe_layer import MoELayer
from mindformers.parallel_core.transformer_config import TransformerConfig

SCRIPT_DIR = Path(__file__).parent.resolve()


class MoELayerRunner:
    """Class to manage MoELayer model and weights"""

    def __init__(self, args_from_parser):
        self.args = args_from_parser
        self.hidden_size = self.args.hidden_size
        self.num_experts = self.args.num_experts
        self.top_k = self.args.top_k
        self.score_func = self.args.score_func
        self.route_norm = self.args.route_norm
        self.route_scale = self.args.route_scale
        self.num_expert_groups = self.args.num_expert_groups
        self.num_limited_groups = self.args.num_limited_groups
        self.force_load_balance = self.args.force_load_balance
        self.batch_size = self.args.batch_size
        self.seq_length = self.args.seq_length
        self.shared_expert_num = self.args.shared_expert_num
        self.apply_probs_on_input = self.args.apply_probs_on_input

        init_params = get_init_params(
            self.hidden_size, self.num_experts, self.batch_size, self.seq_length, self.shared_expert_num
        )

        self.inputs = ms.Tensor(init_params.get("inputs"), dtype=ms.float32)
        self.router_weight = init_params.get("router_weight")
        self.experts_weight1 = init_params.get("experts_weight1")
        self.experts_weight2 = init_params.get("experts_weight2")
        self.shared_fc1 = init_params.get("shared_fc1")
        self.shared_fc2 = init_params.get("shared_fc2")

        # RANK_ID and worker_num are set by msrun environment
        rank_id_str = os.environ.get("RANK_ID")
        self.rank_id = int(rank_id_str) if rank_id_str is not None else None
        self.worker_num = int(os.environ.get("MS_WORKER_NUM", "1"))

        # Set parallel context for multi-card
        if self.rank_id is not None:
            ms.set_auto_parallel_context(
                parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, full_batch=True
            )
            init()

    def build_model(self):
        """Build and initialize MoELayer model"""
        # Create config
        config = TransformerConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=4,
            num_layers=1,
            hidden_act="fusedswiglu",
            num_moe_experts=self.num_experts,
            moe_router_topk=self.top_k,
            moe_router_score_function=self.score_func,
            norm_topk_prob=self.route_norm,
            moe_router_topk_scaling_factor=self.route_scale,
            moe_router_num_groups=self.num_expert_groups,
            moe_router_group_topk=self.num_limited_groups,
            moe_router_force_expert_balance=self.force_load_balance,
            add_bias_linear=False,
            # MoE specific configs
            moe_ffn_hidden_size=self.hidden_size * 4, # Standard MLP expansion
            shared_expert_num=self.shared_expert_num,
            moe_apply_probs_on_input=self.apply_probs_on_input,
            gated_linear_unit=True,
            # For testing, we can set aux loss coeff to enable expert_bias buffer creation
            moe_aux_loss_coeff=0.01
        )

        net = MoELayer(config)

        state_dict = {
            "router.weight": ms.Parameter(self.router_weight),
            "experts.weight1": ms.Parameter(self.experts_weight1),
            "experts.weight2": ms.Parameter(self.experts_weight2)
        }

        if self.shared_expert_num > 0:
            state_dict["shared_experts.linear_fc1.weight"] = ms.Parameter(self.shared_fc1)
            state_dict["shared_experts.linear_fc2.weight"] = ms.Parameter(self.shared_fc2)

        ms.load_param_into_net(net, state_dict)

        return net

    def run(self):
        """Run the model with given inputs"""
        net = self.build_model()
        net.set_train(True)
        inputs = self.inputs.to(ms.bfloat16)
        output, _ = net(inputs)

        output_ms = {
            "output_case0": output,
        }

        if self.rank_id is None or int(self.rank_id) == 0:
            output_np = {k: v.asnumpy().astype(np.float32)
                         for k, v in output_ms.items() if v is not None}
            output_path = self.args.output_path
            np.savez(output_path, **output_np)


def main():
    parser = argparse.ArgumentParser(description="Run MoELayer test")
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--score_func", type=str, default="softmax")
    parser.add_argument("--route_norm", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--route_scale", type=float, default=1.0)
    parser.add_argument("--num_expert_groups", type=int, default=None)
    parser.add_argument("--num_limited_groups", type=int, default=None)
    parser.add_argument("--force_load_balance", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_length", type=int, default=4)
    parser.add_argument("--shared_expert_num", type=int, default=0)
    parser.add_argument("--apply_probs_on_input", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--output_path", type=str, default="output_ms.npz")

    args = parser.parse_args()

    ms.set_deterministic(True)
    context.set_context(mode=context.PYNATIVE_MODE)
    ms.set_seed(42)

    # Prepare input
    runner = MoELayerRunner(args)
    runner.run()

if __name__ == "__main__":
    main()
