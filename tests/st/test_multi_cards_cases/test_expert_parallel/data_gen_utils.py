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
"""Generate data for ExpertParallel (EP) test."""

import numpy as np


def get_init_params(hidden_size=32, num_experts=4, batch_size=2, seq_length=4):
    """Generate initialization parameters for ExpertParallel test.
    
    Args:
        hidden_size: Hidden dimension size
        num_experts: Number of experts
        batch_size: Batch size of input tokens
        seq_length: Sequence length
    
    Returns:
        Dictionary containing:
        - tokens: Input tokens (batch_size, seq_length, hidden_size)
        - probs: Router probabilities (batch_size * seq_length, top_k)
        - topk_indices: Selected expert indices (batch_size * seq_length, top_k)
        - num_tokens_per_expert: Token count per expert (num_experts,)
        - weight1: Expert first layer weight (num_experts, hidden_size, ffn_hidden_size*2)
        - weight2: Expert second layer weight (num_experts, ffn_hidden_size, hidden_size)
    """
    np.random.seed(42)
    
    # tokens: (batch_size, seq_length, hidden_size)
    tokens = 0.01 * np.random.randn(batch_size, seq_length, hidden_size).astype(np.float32)
    
    top_k = 2  # Top-k for routing
    total_tokens = batch_size * seq_length
    
    # probs: (total_tokens, top_k) - routing probabilities
    probs = np.random.dirichlet(np.ones(top_k), total_tokens).astype(np.float32)
    
    # topk_indices: (total_tokens, top_k) - selected expert indices
    topk_indices = np.random.randint(0, num_experts, size=(total_tokens, top_k)).astype(np.int32)
    
    # num_tokens_per_expert: (num_experts,) - token distribution per expert
    # This is computed based on topk_indices
    num_tokens_per_expert = np.zeros(num_experts, dtype=np.float32)
    for i in range(total_tokens):
        for j in range(top_k):
            expert_idx = topk_indices[i, j]
            num_tokens_per_expert[expert_idx] += 1
    
    # Expert weights
    ffn_hidden_size = hidden_size * 4
    
    # weight1: (num_experts, hidden_size, ffn_hidden_size*2)
    # For GroupedMLP, this is reshaped to (num_experts * hidden_size, ffn_hidden_size * 2)
    weight1 = 0.01 * np.random.randn(num_experts * hidden_size, ffn_hidden_size * 2).astype(np.float32)
    
    # weight2: (num_experts, ffn_hidden_size, hidden_size)
    # For GroupedMLP, this is reshaped to (num_experts * ffn_hidden_size, hidden_size)
    weight2 = 0.01 * np.random.randn(num_experts * ffn_hidden_size, hidden_size).astype(np.float32)
    
    params = {
        "tokens": tokens,
        "probs": probs,
        "topk_indices": topk_indices,
        "num_tokens_per_expert": num_tokens_per_expert,
        "weight1": weight1,
        "weight2": weight2,
    }
    
    return params
