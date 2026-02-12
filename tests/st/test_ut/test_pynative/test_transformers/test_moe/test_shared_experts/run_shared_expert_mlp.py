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
"""Run SharedExpertMLP accuracy test with configurable parameters via args"""
import argparse

import mindspore as ms
from mindspore import nn
from data_gen_utils import get_init_params, get_golden_datas, get_gpu_datas
from tests.utils.double_benchmark import DoubleBenchmarkComparator, DoubleBenchmarkStandard
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.pynative.transformers.moe.shared_experts import SharedExpertMLP, MLPSubmodules
from mindformers.pynative.layers.linear import Linear


class SharedExpertMLPRunner:
    """Class to manage SharedExpertMLP model and weights"""

    def __init__(self, args_from_parser):
        self.args = args_from_parser
        self.use_shared_expert_gating = self.args.gate

        self.transformer_config = self._get_config()
        self.input_, self.state_dict = get_init_params(self.transformer_config)

    def _get_config(self):
        """get TransformerConfig for test"""
        return TransformerConfig(
            num_layers=1,
            num_attention_heads=2,
            hidden_size=16,
            ffn_hidden_size=16,
            moe_shared_expert_intermediate_size=16,
            hidden_act="silu",
            add_bias_linear=False,
            compute_dtype='bfloat16',
            params_dtype='float32',
            moe_router_dtype='float32',
            use_shared_expert_gating=self.use_shared_expert_gating
        )

    def build_model(self):
        """Build and initialize SharedExpertMLP model"""
        net = TestModel(self.transformer_config)
        ms.load_param_into_net(net, self.state_dict)
        return net

    def run(self):
        """Run the model with given inputs"""
        net = self.build_model()

        output = net(self.input_)
        output_npu = output.asnumpy()
        standard = DoubleBenchmarkStandard(dtype="bfloat16")
        output_gpu = get_gpu_datas(self.args)
        output_golden = get_golden_datas(self.args)
        DoubleBenchmarkComparator.check_pass_or_not(output_npu, output_gpu, output_golden, standard)
        print(f"Accuracy Test Case Finished: use_shared_expert_gating={self.use_shared_expert_gating}.")


class TestModel(nn.Cell):
    """Model for test"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.mlp = SharedExpertMLP(config=config, submodules=MLPSubmodules(linear_fc1=Linear,
                                                                           linear_fc2=Linear))

    def construct(self, hidden_states):
        """This avoids graph compilation errors due to unsupported return types."""
        mlp_output = self.mlp(hidden_states)
        return mlp_output


def main():
    parser = argparse.ArgumentParser(description="Run SharedExpertMLP test")
    parser.add_argument(
        '--gate',
        action='store_true',
        help='use a gated linear unit in the SharedExpertMLP')

    parser.set_defaults(gate=False)
    args = parser.parse_args()

    ms.set_deterministic(True)
    ms.set_context(mode=ms.PYNATIVE_MODE)

    # Prepare input and run
    runner = SharedExpertMLPRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
