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
"""The MindFormers hash routing script runs.

It loads predefined test cases from `hash_routing_data_gen`, builds TopKRouter, runs `construct`,
    and saves the output as a .npz file.

Usage (single card):
    python run_hash_router.py --case-key v50_e4_k2_softmax_s1.0_l4_b2_h16 --output-path out.npz

Usage (multi card):
    msrun --worker_num=2 --local_worker_num=2 --master_port=8118 \
        run_hash_router.py --case-key ... --output-path out.npz
"""

import argparse
import os

import numpy as np
import mindspore as ms
from mindspore import Tensor, context
from mindspore.communication import init

from hash_routing_data_gen import CASE_INPUT_REGISTRY
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.pynative.transformers.moe.router import TopKRouter


class HashRouterRunner:
    """Load deterministic data, build router, run construct, save output."""

    def __init__(self, args_from_parser):
        self.args = args_from_parser
        self.case_key = args_from_parser.case_key
        self.output_path = args_from_parser.output_path
        self.data = CASE_INPUT_REGISTRY[self.case_key]

        rank_id_str = os.environ.get("RANK_ID")
        self.rank_id = int(rank_id_str) if rank_id_str is not None else None
        if self.rank_id is not None:
            ms.set_auto_parallel_context(
                parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, full_batch=True
            )
            init()

    def build_model(self):
        """Build TopKRouter model."""
        config = TransformerConfig(
            hidden_size=self.data["hidden_size"],
            num_attention_heads=4,
            num_layers=4,
            num_moe_experts=self.data["num_experts"],
            moe_router_topk=self.data["top_k"],
            moe_router_score_function=self.data["score_func"],
            moe_router_topk_scaling_factor=self.data["route_scale"],
            moe_router_dtype="float32",
            moe_n_hash_layers=1,
            moe_router_pre_softmax=(self.data["top_k"] == 1),
            actual_vocab_size=self.data["vocab_size"],
            hidden_act="fusedswiglu",
            gated_linear_unit=True,
            moe_ffn_hidden_size=self.data["hidden_size"] * 4,
            add_bias_linear=False,
        )
        router = TopKRouter(config, layer_number=0)
        router.reset_parameter()

        weight_ms = ms.Parameter(
            Tensor(self.data["weight"], dtype=ms.float32),
            name="weight",
        )
        ms.load_param_into_net(router, {"weight": weight_ms})

        tid2eid = Tensor(self.data["tid2eid"], dtype=ms.int32)
        router.tid2eid.set_data(tid2eid)

        return router

    def run(self):
        """Run the router module with given inputs."""
        router = self.build_model()

        x = Tensor(self.data["hidden_states"], dtype=ms.float32)
        input_ids = Tensor(self.data["input_ids"], dtype=ms.int32)

        top_scores, selected_experts_indices, _ = router.construct(
            x, expert_bias=None, input_ids=input_ids
        )

        output_ms = {
            "top_scores": top_scores.asnumpy().astype(np.float32),
            "selected_experts_indices": selected_experts_indices.asnumpy().astype(np.int64),
        }

        if self.rank_id is None or int(self.rank_id) == 0:
            np.savez(self.output_path, **output_ms)


def main():
    parser = argparse.ArgumentParser(description="Run MindFormers hash router test")
    parser.add_argument("--case-key", type=str, required=True,
                        help="Test case key from hash_routing_data_gen")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Path to save .npz output")
    args = parser.parse_args()

    if args.case_key not in CASE_INPUT_REGISTRY:
        available = "\n  ".join(CASE_INPUT_REGISTRY.keys())
        raise KeyError(f"Unknown case_key '{args.case_key}'. Available:\n  {available}")

    ms.set_deterministic(True)
    context.set_context(mode=context.PYNATIVE_MODE)
    ms.set_seed(42)

    runner = HashRouterRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
