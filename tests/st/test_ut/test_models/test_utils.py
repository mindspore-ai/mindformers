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
"""test utils"""
from unittest.mock import patch

import pytest
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.models.utils import (
    num_floating_point_operations,
    convert_transformer_config_to_args_for_tflops
)


class TestNumFloatingPointOperations:
    """Test num_floating_point_operations function"""

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_standard_transformer_basic(self):
        """Test standard transformer: no GQA, no MoE, no MTP, no MLA, no linear attention, no SwiGLU"""
        config = TransformerConfig(
            num_layers=4,
            hidden_size=128,
            num_attention_heads=8,
            seq_length=512,
            ffn_hidden_size=512,
            vocab_size=10000
        )
        args = convert_transformer_config_to_args_for_tflops(config)
        args.mtp_num_layers = None
        batch_size = 2
        flops = num_floating_point_operations(args, batch_size)
        expected_flops = 14306770944.0
        assert isinstance(flops, (int, float))
        assert flops == expected_flops, f"Expected flops: {expected_flops}, but got: {flops}"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_transformer_with_gqa(self):
        """Test transformer with Group Query Attention"""
        config = TransformerConfig(
            num_layers=4,
            hidden_size=128,
            num_attention_heads=8,
            num_query_groups=4,
            seq_length=512,
            ffn_hidden_size=512,
            vocab_size=10000
        )
        args = convert_transformer_config_to_args_for_tflops(config)
        batch_size = 2
        flops = num_floating_point_operations(args, batch_size)
        expected_flops = 13904117760.0
        assert isinstance(flops, (int, float))
        assert flops == expected_flops, f"Expected flops: {expected_flops}, but got: {flops}"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_transformer_with_moe_int_freq(self):
        """Test transformer with MoE using int frequency"""
        config = TransformerConfig(
            num_layers=4,
            hidden_size=128,
            num_attention_heads=8,
            seq_length=512,
            ffn_hidden_size=512,
            vocab_size=10000,
            num_moe_experts=8,
            moe_layer_freq=2,
            moe_router_topk=2,
            add_bias_linear=False
        )
        args = convert_transformer_config_to_args_for_tflops(config)
        batch_size = 2
        flops = num_floating_point_operations(args, batch_size)
        expected_flops = 15917383680.0
        assert isinstance(flops, (int, float))
        assert flops == expected_flops, f"Expected flops: {expected_flops}, but got: {flops}"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_transformer_with_moe_list_freq(self):
        """Test transformer with MoE using list frequency"""
        config = TransformerConfig(
            num_layers=4,
            hidden_size=128,
            num_attention_heads=8,
            seq_length=512,
            ffn_hidden_size=512,
            vocab_size=10000,
            num_moe_experts=8,
            moe_layer_freq=[1, 0, 1, 0],
            moe_router_topk=2,
            add_bias_linear=False
        )
        args = convert_transformer_config_to_args_for_tflops(config)
        batch_size = 2
        flops = num_floating_point_operations(args, batch_size)
        expected_flops = 15917383680.0
        assert isinstance(flops, (int, float))
        assert flops == expected_flops, f"Expected flops: {expected_flops}, but got: {flops}"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_transformer_with_moe_shared_expert(self):
        """Test transformer with MoE and shared expert"""
        config = TransformerConfig(
            num_layers=4,
            hidden_size=128,
            num_attention_heads=8,
            seq_length=512,
            ffn_hidden_size=512,
            vocab_size=10000,
            num_moe_experts=8,
            moe_layer_freq=2,
            moe_router_topk=2,
            moe_shared_expert_intermediate_size=1024,
            add_bias_linear=False
        )
        args = convert_transformer_config_to_args_for_tflops(config)
        batch_size = 2
        flops = num_floating_point_operations(args, batch_size)
        expected_flops = 19138609152.0
        assert isinstance(flops, (int, float))
        assert flops == expected_flops, f"Expected flops: {expected_flops}, but got: {flops}"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_transformer_with_mtp_and_moe(self):
        """Test transformer with MTP and MoE (covers both MTP and MTP+MoE paths)"""
        config = TransformerConfig(
            num_layers=4,
            hidden_size=128,
            num_attention_heads=8,
            seq_length=512,
            ffn_hidden_size=512,
            vocab_size=10000,
            mtp_num_layers=1,
            num_moe_experts=8,
            moe_layer_freq=2,
            moe_router_topk=2,
            add_bias_linear=False
        )
        args = convert_transformer_config_to_args_for_tflops(config)
        batch_size = 2
        flops = num_floating_point_operations(args, batch_size)
        expected_flops = 25596002304.0
        assert isinstance(flops, (int, float))
        assert flops == expected_flops, f"Expected flops: {expected_flops}, but got: {flops}"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_transformer_with_mla_no_q_lora_rank(self):
        """Test transformer with Multi-Latent Attention: q_lora_rank=None"""
        config = TransformerConfig(
            num_layers=4,
            hidden_size=128,
            num_attention_heads=8,
            seq_length=512,
            ffn_hidden_size=512,
            vocab_size=10000,
            multi_latent_attention=True
        )
        args = convert_transformer_config_to_args_for_tflops(config)
        # Set MLA-specific parameters
        args.qk_head_dim = 64
        args.qk_pos_emb_head_dim = 32
        args.v_head_dim = 64
        args.kv_lora_rank = 256
        # Set q_lora_rank to None to test that branch
        args.q_lora_rank = None
        batch_size = 2
        flops = num_floating_point_operations(args, batch_size)
        expected_flops = 30519853056.0
        assert isinstance(flops, (int, float))
        assert flops == expected_flops, f"Expected flops: {expected_flops}, but got: {flops}"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_transformer_with_mla_with_q_lora_rank(self):
        """Test transformer with Multi-Latent Attention: q_lora_rank has value"""
        config = TransformerConfig(
            num_layers=4,
            hidden_size=128,
            num_attention_heads=8,
            seq_length=512,
            ffn_hidden_size=512,
            vocab_size=10000,
            multi_latent_attention=True
        )
        args = convert_transformer_config_to_args_for_tflops(config)
        # Set MLA-specific parameters
        args.qk_head_dim = 64
        args.qk_pos_emb_head_dim = 32
        args.v_head_dim = 64
        args.kv_lora_rank = 256
        args.q_lora_rank = 512
        batch_size = 2
        flops = num_floating_point_operations(args, batch_size)
        expected_flops = 39390806016.0
        assert isinstance(flops, (int, float))
        assert flops == expected_flops, f"Expected flops: {expected_flops}, but got: {flops}"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_transformer_with_attention_output_gate(self):
        """Test transformer with attention output gate"""
        config = TransformerConfig(
            num_layers=4,
            hidden_size=128,
            num_attention_heads=8,
            seq_length=512,
            ffn_hidden_size=512,
            vocab_size=10000
        )
        args = convert_transformer_config_to_args_for_tflops(config)
        args.attention_output_gate = True
        batch_size = 2
        flops = num_floating_point_operations(args, batch_size)
        expected_flops = 14709424128.0
        assert isinstance(flops, (int, float))
        assert flops == expected_flops, f"Expected flops: {expected_flops}, but got: {flops}"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_transformer_with_linear_attention(self):
        """Test transformer with linear attention: gated_delta_net with int and list frequency"""
        config = TransformerConfig(
            num_layers=4,
            hidden_size=128,
            num_attention_heads=8,
            seq_length=512,
            ffn_hidden_size=512,
            vocab_size=10000
        )
        args = convert_transformer_config_to_args_for_tflops(config)
        args.experimental_attention_variant = "gated_delta_net"
        args.linear_key_head_dim = 64
        args.linear_value_head_dim = 64
        args.linear_num_key_heads = 4
        args.linear_num_value_heads = 8
        args.linear_conv_kernel_dim = 128
        batch_size = 2
        
        # Test int frequency
        args.linear_attention_freq = 2
        flops = num_floating_point_operations(args, batch_size)
        expected_flops = 19163774976.0
        assert isinstance(flops, (int, float))
        assert flops == expected_flops, f"Expected flops: {expected_flops}, but got: {flops}"
        
        # Test list frequency
        args.linear_attention_freq = [1, 1, 0, 1]
        flops = num_floating_point_operations(args, batch_size)
        expected_flops = 21592276992.0
        assert isinstance(flops, (int, float))
        assert flops == expected_flops, f"Expected flops: {expected_flops}, but got: {flops}"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_transformer_with_swiglu(self):
        """Test transformer with SwiGLU activation"""
        config = TransformerConfig(
            num_layers=4,
            hidden_size=128,
            num_attention_heads=8,
            seq_length=512,
            ffn_hidden_size=512,
            vocab_size=10000,
            hidden_act="swiglu"
        )
        args = convert_transformer_config_to_args_for_tflops(config)
        batch_size = 2
        flops = num_floating_point_operations(args, batch_size)
        expected_flops = 15917383680.0
        assert isinstance(flops, (int, float))
        assert flops == expected_flops, f"Expected flops: {expected_flops}, but got: {flops}"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_hybrid_model_basic_and_mtp(self):
        """Test hybrid model: mamba_num_heads=None with mtp_num_layers=None, and mtp_num_layers != None"""
        config = TransformerConfig(
            num_layers=4,
            hidden_size=128,
            num_attention_heads=8,
            seq_length=512,
            ffn_hidden_size=512,
            vocab_size=10000
        )
        args = convert_transformer_config_to_args_for_tflops(config)
        args.hybrid_layer_pattern = "*M-*"
        args.mamba_state_dim = 128
        args.mamba_head_dim = 64
        args.mamba_num_groups = 8
        batch_size = 2
        
        # Test mamba_num_heads=None and mtp_num_layers=None
        args.mamba_num_heads = None
        args.mtp_num_layers = None  # Test mtp_num_layers=None branch
        flops = num_floating_point_operations(args, batch_size)
        expected_flops = 13202620416.0
        assert isinstance(flops, (int, float))
        assert flops == expected_flops, f"Expected flops: {expected_flops}, but got: {flops}"
        
        # Test mtp_num_layers != None
        args.mamba_num_heads = 128
        args.mtp_num_layers = 1
        flops = num_floating_point_operations(args, batch_size)
        expected_flops = 21164457984.0
        assert isinstance(flops, (int, float))
        assert flops == expected_flops, f"Expected flops: {expected_flops}, but got: {flops}"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_hybrid_model_with_pattern_variants(self):
        """Test hybrid model with different pattern variants (MTP pattern and pipeline stage separators)"""
        config = TransformerConfig(
            num_layers=4,
            hidden_size=128,
            num_attention_heads=8,
            seq_length=512,
            ffn_hidden_size=512,
            vocab_size=10000
        )
        args = convert_transformer_config_to_args_for_tflops(config)
        args.mamba_state_dim = 128
        args.mamba_head_dim = 64
        args.mamba_num_groups = 8
        args.mamba_num_heads = 128
        batch_size = 2
        
        # Test MTP pattern
        args.hybrid_layer_pattern = "M*M*/MM/MM"  # main pattern + MTP pattern with 2 depths
        flops = num_floating_point_operations(args, batch_size)
        expected_flops = 27594326016.0
        assert isinstance(flops, (int, float))
        assert flops == expected_flops, f"Expected flops: {expected_flops}, but got: {flops}"
        
        # Test pipeline stage separator
        args.hybrid_layer_pattern = "M-M-|M-M*-"  # pattern with pipeline stage separator
        flops = num_floating_point_operations(args, batch_size)
        expected_flops = 23970447360.0
        assert isinstance(flops, (int, float))
        assert flops == expected_flops, f"Expected flops: {expected_flops}, but got: {flops}"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_hybrid_model_invalid_pattern(self):
        """Test hybrid model with invalid pattern symbol (should raise ValueError)"""
        config = TransformerConfig(
            num_layers=4,
            hidden_size=128,
            num_attention_heads=8,
            seq_length=512,
            ffn_hidden_size=512,
            vocab_size=10000
        )
        args = convert_transformer_config_to_args_for_tflops(config)
        args.hybrid_layer_pattern = "*M-X"  # 'X' is invalid symbol
        args.mamba_state_dim = 128
        args.mamba_head_dim = 64
        args.mamba_num_groups = 8
        args.mamba_num_heads = 128
        batch_size = 2
        with pytest.raises(ValueError, match="is not a valid layer symbol"):
            num_floating_point_operations(args, batch_size)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_hybrid_model_inconsistent_mtp_pattern(self):
        """Test hybrid model with inconsistent MTP patterns (should raise ValueError)"""
        config = TransformerConfig(
            num_layers=4,
            hidden_size=128,
            num_attention_heads=8,
            seq_length=512,
            ffn_hidden_size=512,
            vocab_size=10000
        )
        args = convert_transformer_config_to_args_for_tflops(config)
        args.hybrid_layer_pattern = "M*M*/MM/MMM"  # Inconsistent MTP patterns
        args.mamba_state_dim = 128
        args.mamba_head_dim = 64
        args.mamba_num_groups = 8
        args.mamba_num_heads = 128
        batch_size = 2
        with pytest.raises(ValueError, match="All MTP patterns must be identical"):
            num_floating_point_operations(args, batch_size)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_hybrid_model_with_moe_latent_size(self):
        """Test hybrid model with MoE and moe_latent_size"""
        config = TransformerConfig(
            num_layers=4,
            hidden_size=128,
            num_attention_heads=8,
            seq_length=512,
            ffn_hidden_size=512,
            vocab_size=10000,
            add_bias_linear=False
        )
        args = convert_transformer_config_to_args_for_tflops(config)
        args.hybrid_layer_pattern = "*M-E"  # attention, mamba, mlp, moe
        args.mamba_state_dim = 128
        args.mamba_head_dim = 64
        args.mamba_num_groups = 8
        args.mamba_num_heads = 128
        args.num_experts = 8
        args.moe_router_topk = 2
        args.moe_ffn_hidden_size = 512
        args.moe_latent_size = 256
        batch_size = 2
        flops = num_floating_point_operations(args, batch_size)
        expected_flops = 16118710272.0
        assert isinstance(flops, (int, float))
        assert flops == expected_flops, f"Expected flops: {expected_flops}, but got: {flops}"

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_transformer_with_moe_invalid_freq(self):
        """Test transformer with MoE using invalid moe_layer_freq type (should raise RuntimeError)"""
        config = TransformerConfig(
            num_layers=4,
            hidden_size=128,
            num_attention_heads=8,
            seq_length=512,
            ffn_hidden_size=512,
            vocab_size=10000,
            num_moe_experts=8,
            moe_layer_freq=2,
            moe_router_topk=2,
            add_bias_linear=False
        )
        args = convert_transformer_config_to_args_for_tflops(config)
        # Set invalid moe_layer_freq type to test RuntimeError branch
        args.moe_layer_freq = "invalid"
        batch_size = 2
        with pytest.raises(RuntimeError, match="Illegal --moe-layer-freq argument provided!"):
            num_floating_point_operations(args, batch_size)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_transformer_with_linear_attention_none_freq(self):
        """Test transformer with linear attention but linear_attention_freq=None (should raise ValueError)"""
        config = TransformerConfig(
            num_layers=4,
            hidden_size=128,
            num_attention_heads=8,
            seq_length=512,
            ffn_hidden_size=512,
            vocab_size=10000
        )
        args = convert_transformer_config_to_args_for_tflops(config)
        args.experimental_attention_variant = "gated_delta_net"
        args.linear_attention_freq = None
        args.linear_key_head_dim = 64
        args.linear_value_head_dim = 64
        args.linear_num_key_heads = 4
        args.linear_num_value_heads = 8
        args.linear_conv_kernel_dim = 128
        batch_size = 2
        with pytest.raises(ValueError, match="Linear attention type.*but linear_attention_freq is None"):
            num_floating_point_operations(args, batch_size)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_transformer_with_linear_attention_invalid_freq(self):
        """Test transformer with linear attention but invalid linear_attention_freq type (should raise ValueError)"""
        config = TransformerConfig(
            num_layers=4,
            hidden_size=128,
            num_attention_heads=8,
            seq_length=512,
            ffn_hidden_size=512,
            vocab_size=10000
        )
        args = convert_transformer_config_to_args_for_tflops(config)
        args.experimental_attention_variant = "gated_delta_net"
        args.linear_attention_freq = "invalid"  # Invalid type
        args.linear_key_head_dim = 64
        args.linear_value_head_dim = 64
        args.linear_num_key_heads = 4
        args.linear_num_value_heads = 8
        args.linear_conv_kernel_dim = 128
        batch_size = 2
        with pytest.raises(ValueError, match="Invalid linear_attention_freq"):
            num_floating_point_operations(args, batch_size)

    @pytest.mark.level1
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_transformer_with_linear_attention_invalid_variant(self):
        """Test transformer with invalid experimental_attention_variant (should raise ValueError)"""
        config = TransformerConfig(
            num_layers=4,
            hidden_size=128,
            num_attention_heads=8,
            seq_length=512,
            ffn_hidden_size=512,
            vocab_size=10000
        )
        args = convert_transformer_config_to_args_for_tflops(config)
        args.experimental_attention_variant = "invalid_variant"  # Not "gated_delta_net"
        args.linear_attention_freq = 2
        args.linear_key_head_dim = 64
        args.linear_value_head_dim = 64
        args.linear_num_key_heads = 4
        args.linear_num_value_heads = 8
        args.linear_conv_kernel_dim = 128
        batch_size = 2
        with patch('mindformers.models.utils.is_linear_attention_variant', return_value=True):
            with pytest.raises(ValueError, match="Invalid experimental_attention_variant"):
                num_floating_point_operations(args, batch_size)
