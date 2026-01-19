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
"""Data generation utilities for TransformerLayer test"""
import numpy as np
import mindspore as ms

# Default configuration for data generation
# These can be overridden by arguments in the test script if needed
DEFAULT_SEQ_LENGTH = 2
DEFAULT_BATCH_SIZE = 2
DEFAULT_HIDDEN_SIZE = 24
DEFAULT_FFN_HIDDEN_SIZE = 48 # Typically 4 * hidden_size, but smaller for faster tests
DEFAULT_NUM_HEADS = 4


def get_init_params(seq_length=DEFAULT_SEQ_LENGTH,
                    batch_size=DEFAULT_BATCH_SIZE,
                    hidden_size=DEFAULT_HIDDEN_SIZE,
                    compute_dtype=ms.bfloat16):
    """
    Generates initial parameters (inputs) for the TransformerLayer model.
    """
    np.random.seed(42)
    hidden_states_shape = (seq_length, batch_size, hidden_size)
    hidden_states_np = 0.01 * np.random.randn(*hidden_states_shape).astype(np.float32) # Initial data in fp32
    attention_mask_np = np.random.choice([True, False], size=(batch_size, 1, seq_length, seq_length)).astype(np.int32)
    init_params = {
        "hidden_states": ms.Tensor(hidden_states_np, dtype=compute_dtype),
        "attention_mask": ms.Tensor(attention_mask_np, dtype=compute_dtype), # Usually bool or float
    }
    return init_params


def get_fixed_weights(hidden_size=DEFAULT_HIDDEN_SIZE,
                      ffn_hidden_size=DEFAULT_FFN_HIDDEN_SIZE,
                      param_init_dtype=ms.float32):
    """Generate fixed weights for TransformerLayer.

    Args:
        hidden_size (int, optional): Hidden dimension of the transformer layer. Default: DEFAULT_HIDDEN_SIZE.
        ffn_hidden_size (int, optional): Feed-forward network hidden dimension. Default: DEFAULT_FFN_HIDDEN_SIZE.
        param_init_dtype (mindspore.dtype, optional): Data type for parameter initialization. Default: ms.float32.

    Returns:
        dict: Dictionary containing initialized transformer layer weights.
    """
    np.random.seed(42)
    qkv_weight = 0.01 * np.random.randn(3 * hidden_size, hidden_size).astype(np.float32)
    qkv_bias = 0.01 * np.random.randn(3 * hidden_size).astype(np.float32)
    proj_weight = 0.01 * np.random.randn(hidden_size, hidden_size).astype(np.float32)
    proj_bias = 0.01 * np.random.randn(hidden_size).astype(np.float32)
    input_ln_gamma = 0.01 * np.random.randn(hidden_size).astype(np.float32)
    input_ln_beta = 0.01 * np.random.randn(hidden_size).astype(np.float32)
    pre_mlp_ln_gamma = 0.01 * np.random.randn(hidden_size).astype(np.float32)
    pre_mlp_ln_beta = 0.01 * np.random.randn(hidden_size).astype(np.float32)
    mlp_fc1_weight = 0.01 * np.random.randn(ffn_hidden_size, hidden_size).astype(np.float32)
    mlp_fc1_bias = 0.01 * np.random.randn(ffn_hidden_size).astype(np.float32)
    mlp_fc2_weight = 0.01 * np.random.randn(hidden_size, ffn_hidden_size).astype(np.float32)
    mlp_fc2_bias = 0.01 * np.random.randn(hidden_size).astype(np.float32)
    weight_dict = {
        "input_layernorm.gamma": ms.Tensor(input_ln_gamma, dtype=param_init_dtype),
        "input_layernorm.beta": ms.Tensor(input_ln_beta, dtype=param_init_dtype),
        "self_attention.linear_qkv.weight": ms.Tensor(qkv_weight, dtype=param_init_dtype),
        "self_attention.linear_qkv.bias": ms.Tensor(qkv_bias, dtype=param_init_dtype),
        "self_attention.linear_proj.weight": ms.Tensor(proj_weight, dtype=param_init_dtype),
        "self_attention.linear_proj.bias": ms.Tensor(proj_bias, dtype=param_init_dtype),
        "pre_mlp_layernorm.gamma": ms.Tensor(pre_mlp_ln_gamma, dtype=param_init_dtype),
        "pre_mlp_layernorm.beta": ms.Tensor(pre_mlp_ln_beta, dtype=param_init_dtype),
        "mlp.linear_fc1.weight": ms.Tensor(mlp_fc1_weight, dtype=param_init_dtype),
        "mlp.linear_fc1.bias": ms.Tensor(mlp_fc1_bias, dtype=param_init_dtype),
        "mlp.linear_fc2.weight": ms.Tensor(mlp_fc2_weight, dtype=param_init_dtype),
        "mlp.linear_fc2.bias": ms.Tensor(mlp_fc2_bias, dtype=param_init_dtype),
    }
    return weight_dict

GOLDEN_DATA = {
    "output_default": np.array(
        [[[-0.0039093159, -0.0094388910, -0.0234587993,  0.0433887206, -0.0065255468, -0.0102961604,
           0.0304493606,  0.0299217571,  0.0047993637,  0.0060305987, -0.0051112385, -0.0151664773,
           0.0070123109, -0.0172300264, -0.0431940109, -0.0282073356, -0.0261407960, -0.0201630425,
          -0.0079966392, -0.0287125688,  0.0072326753, -0.0177765712,  0.0336738043,  0.0000863587],
         [-0.0143408421, -0.0069154087, -0.0414097235,  0.0319417939, -0.0101610739, -0.0109111620,
           0.0085853236,  0.0407416523,  0.0093610268, -0.0100708334,  0.0077591301, -0.0227215271,
           0.0066808863, -0.0176695306, -0.0392333269, -0.0206004754, -0.0085618747, -0.0216183402,
          -0.0000576600, -0.0175904669, -0.0221993309, -0.0226927958,  0.0283830240,  0.0248611048]],

        [[-0.0054175197, -0.0256709047, -0.0266851541,  0.0243020765, -0.0109246895, -0.0018515757,
           0.0249271523,  0.0315530859,  0.0010830322, -0.0025258809,  0.0028285750, -0.0007606138,
          -0.0002181875,  0.0000578081, -0.0370167345, -0.0345540531, -0.0078531718, -0.0097700190,
           0.0003877311, -0.0045704865, -0.0038178815, -0.0219577774,  0.0366174951,  0.0297042765],
         [-0.0092844358,  0.0076036649, -0.0561106429,  0.0363665968, -0.0033172830, -0.0109820329,
           0.0155247431,  0.0023542009,  0.0072811274,  0.0041908016,  0.0143108396, -0.0156996809,
          -0.0035146144, -0.0031273505, -0.0167904720, -0.0192591306, -0.0212960839, -0.0182052627,
           0.0020543011, -0.0048622359, -0.0144246891, -0.0188029818,  0.0290469266, -0.0003483640]]],
           dtype=np.float32),
}

GPU_DATA = {
    "output_default": np.array(
        [[[-0.0039062500, -0.0094604492, -0.0234375000,  0.0434570312, -0.0065002441, -0.0103149414,
           0.0303955078,  0.0299072266,  0.0048522949,  0.0059814453, -0.0051269531, -0.0151367188,
           0.0070190430, -0.0172119141, -0.0432128906, -0.0281982422, -0.0261230469, -0.0202636719,
          -0.0079956055, -0.0286865234,  0.0072326660, -0.0177001953,  0.0336914062,  0.0001220703],
         [-0.0142822266, -0.0068969727, -0.0415039062,  0.0319824219, -0.0102539062, -0.0109863281,
           0.0086059570,  0.0407714844,  0.0093383789, -0.0100097656,  0.0077514648, -0.0227050781,
           0.0066528320, -0.0177001953, -0.0393066406, -0.0205078125, -0.0085449219, -0.0217285156,
          -0.0000610352, -0.0177001953, -0.0220947266, -0.0227050781,  0.0283203125,  0.0249023438]],

        [[-0.0054321289, -0.0256347656, -0.0266113281,  0.0242919922, -0.0109252930, -0.0018768311,
           0.0249023438,  0.0314941406,  0.0010986328, -0.0025634766,  0.0028228760, -0.0007476807,
          -0.0002136230,  0.0001220703, -0.0368652344, -0.0346679688, -0.0078735352, -0.0097656250,
           0.0003814697, -0.0046386719, -0.0038299561, -0.0219726562,  0.0366210938,  0.0296630859],
         [-0.0092773438,  0.0075683594, -0.0561523438,  0.0363769531, -0.0033264160, -0.0109863281,
           0.0155029297,  0.0023193359,  0.0072937012,  0.0041809082,  0.0142822266, -0.0157470703,
          -0.0034790039, -0.0031127930, -0.0167236328, -0.0191650391, -0.0212402344, -0.0183105469,
           0.0020446777, -0.0048828125, -0.0144653320, -0.0187988281,  0.0290527344, -0.0003051758]]],
           dtype=np.float16),
}

if __name__ == '__main__':
    # Example of how to generate and save data if needed for external use
    params = get_init_params()
    print("Generated hidden_states shape:", params["hidden_states"].shape)
    print("Generated attention_mask shape:", params["attention_mask"].shape)
    print("Data generation utilities ready.")
