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
"""ExpertParallel test runner"""
import argparse
import os
import numpy as np
import mindspore as ms
from mindspore import context, Parameter, Tensor
from mindspore.communication import init

from data_gen_utils import get_init_params
from mindformers.pynative.transformers.moe.experts import GroupedMLP
from mindformers.pynative.distributed.expert_parallel import ExpertParallel
from mindformers.parallel_core.transformer_config import TransformerConfig
from hyper_parallel.core.device_mesh import init_device_mesh
from hyper_parallel import DTensor


class ExpertParallelRunner:
    """Run ExpertParallel: single-card baseline vs distributed versions"""

    def __init__(self, args):
        self.hidden_size = args.hidden_size
        self.num_experts = args.num_experts
        self.top_k = args.top_k
        self.batch_size = args.batch_size
        self.seq_length = args.seq_length

        # Set parallel context for distributed mode
        rank_id_str = os.environ.get("RANK_ID")
        self.rank_id = int(rank_id_str) if rank_id_str is not None else 0
        
        if self.rank_id == 0 and self.rank_id is not None:
            ms.set_auto_parallel_context(
                parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, full_batch=True
            )
            init()

        # Generate data with fixed seed
        np.random.seed(42)
        ms.set_seed(42)
        
        init_params = get_init_params(
            self.hidden_size, self.num_experts, self.batch_size, self.seq_length
        )

        self.tokens = ms.Tensor(init_params["tokens"], dtype=ms.float32)
        self.probs = ms.Tensor(init_params["probs"], dtype=ms.float32)
        self.topk_indices = ms.Tensor(init_params["topk_indices"], dtype=ms.int32)
        self.num_tokens_per_expert = ms.Tensor(init_params["num_tokens_per_expert"], dtype=ms.float32)
        self.weight1 = init_params["weight1"]
        self.weight2 = init_params["weight2"]

    def build_model(self):
        """Build GroupedMLP model with fixed weights"""
        config = TransformerConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=4,
            num_layers=1,
            hidden_act="fusedswiglu",
            num_moe_experts=self.num_experts,
            moe_router_topk=self.top_k,
            moe_router_score_function="softmax",
            norm_topk_prob=False,
            moe_router_topk_scaling_factor=1.0,
            moe_router_num_groups=None,
            moe_router_group_topk=None,
            moe_router_force_expert_balance=False,
            add_bias_linear=False,
            moe_ffn_hidden_size=self.hidden_size * 4,
            shared_expert_num=1,
            moe_apply_probs_on_input=False,
            gated_linear_unit=True,
            moe_aux_loss_coeff=0.01
        )
        net = GroupedMLP(config)
        net.weight1 = Parameter(Tensor(self.weight1))
        net.weight2 = Parameter(Tensor(self.weight2))
        return net

    def run_distributed(self):
        """Run single-card and Expert Parallel versions, return both outputs
        
        Returns:
            tuple: (single_output_np, distributed_output_np)
        """
        context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
        ms.set_seed(42)

        # Single-card version
        net_single = self.build_model()
        net_single.set_train(False)
        output_single = net_single(self.tokens, self.probs, self.topk_indices, self.num_tokens_per_expert)
        output_single_np = output_single.asnumpy()

        # Distributed version (Expert Parallel on 2 cards)
        net_distributed = self.build_model()
        net_distributed.set_train(False)
        
        mesh = init_device_mesh(device_type="npu", mesh_shape=(2,), mesh_dim_names=("ep",))
        ep = ExpertParallel()
        net_distributed = ep._apply(net_distributed, mesh)
        
        output_distributed = net_distributed(self.tokens, self.probs, self.topk_indices, self.num_tokens_per_expert)
        
        # Handle DTensor
        if isinstance(output_distributed, DTensor):
            output_distributed = output_distributed.to_local()
        output_distributed_np = output_distributed.asnumpy()

        return output_single_np, output_distributed_np


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="ExpertParallel test")
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_length", type=int, default=4)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    """Main entry point for msrun"""
    args = parse_args()
    runner = ExpertParallelRunner(args)
    single_output, distributed_output = runner.run_distributed()
    
    # Save results on rank 0
    if args.output and runner.rank_id == 0:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        output_path = args.output if args.output.endswith('.npz') else args.output + '.npz'
        np.savez(output_path, single=single_output, distributed=distributed_output)


if __name__ == "__main__":
    main()
