# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
MoE Auxiliary Loss computation and gradient injection.
"""

from typing import Optional,Tuple

from hyper_parallel.core.dtensor.dtensor import DTensor
from mindspore import Tensor, mint, nn
from mindspore.common import dtype as mstype
from mindspore.common._grad_function import _Function


# MOE logging
_MOE_LAYER_WISE_LOGGING_TRACKER: dict = {}


def switch_load_balancing_loss_func(
    probs:  Tensor,
    tokens_per_expert: Tensor,
    total_num_tokens: int,
    topk: int,
    num_experts: int,
    moe_aux_loss_coeff: float,
    fused: bool = False,
) -> Tensor:
    """Calculate the auxiliary loss for load balancing.
    Refer to the Switch Transformer (https://arxiv.org/abs/2101.03961)
    and Global Load Balancing Loss(https://arxiv.org/abs/2501.11873) for details.

    ### Detailed explanation of the auxiliary loss #######

    The formula for the auxiliary loss is:
        loss = E * Σ_{i=1}^{E} (f_i * P_i)
    where:
        f_i = 1 / (T * topk) * Σ_{x∈B} routing_map(x, i)
             (fraction of tokens dispatched to expert i)
        P_i = 1 / T * Σ_{x∈B} probs(x, i)
             (averaged router probability allocated for expert i)
        E is the number of experts
        T is the total number of tokens in the batch B

    For distributed training with sequence or context parallelism, each rank can
    process a subset of the batch.
        loss = E * Σ_{i=1}^{E} (f_i * Σ_{j=1}^{N} P_ij)
             = E * Σ_{i=1}^{E} Σ_{j=1}^{N} (f_i * P_ij)
             = Σ_{j=1}^{N} E * (Σ_{i=1}^{E} f_i * P_ij)

    where:
        f_i = 1 / (T * topk) * Σ_{x∈B} routing_map(x, i)
             (fraction of tokens dispatched to expert i in the global batch)
        P_ij = 1 / T * Σ_{x∈B_j} probs(x, i)
              (averaged router probability allocated for expert i in local batch of the j-th rank)
        N is the number of ranks
        B_j is the batch of tokens in the j-th rank
        T is the total number of tokens in the global batch B

    Note:
    To calculate the auxiliary loss at different levels (micro-batch or global batch):
    - probs: Should always be from the local batch being processed
    - tokens_per_expert: Should represent token counts at the desired level
      (either micro-batch or global batch)
    - total_num_tokens: Should match the total token count at the same level as tokens_per_expert

    #########################################################

    Args:
        probs (Tensor): Softmax probabilities output by the router for each token.
                              Shape in [num_tokens, num_experts].
        tokens_per_expert (Tensor): Number of tokens assigned to each expert in the batch.
                                          Shape in [num_experts]
        total_num_tokens (int): Total number of tokens in the batch.
        topk (int): The number of experts selected for each token.
        num_experts (int): The number of experts.
        moe_aux_loss_coeff (float): The coefficient for the auxiliary loss.
        fused (bool): Whether to use the fused version of the auxiliary loss.

    Returns:
        Tensor: The auxiliary loss for load balancing.
    """
    if fused:
        raise ValueError("Fused version is not supported yet.")
    aggregated_probs_per_expert = probs.sum(dim=0)
    aux_loss = mint.sum(aggregated_probs_per_expert * tokens_per_expert) * (
        num_experts * moe_aux_loss_coeff / (topk * total_num_tokens * total_num_tokens)
    )
    return aux_loss


class _MoEAuxLossAutoScaler(_Function):
    """An AutoScaler that triggers the backward pass and scales the grad for auxiliary loss."""

    main_loss_backward_scale: Optional[Tensor] = None

    @staticmethod
    def forward(ctx, output: Tensor, aux_loss: Tensor) -> Tensor:
        """Preserve the aux_loss by storing it in the context to avoid garbage collection.

        Args:
            output (Tensor): The output tensor.
            aux_loss (Tensor): The auxiliary loss tensor.

        Returns:
            Tensor: The output tensor.
        """
        ctx.aux_loss = aux_loss
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """Compute and scale the gradient for auxiliary loss..

        Args:
            grad_output (Tensor): The gradient of the output.

        Returns:
            Tuple[Tensor, Tensor]: The gradient of the output, scaled auxiliary loss gradient.
        """
        aux_loss = ctx.aux_loss
        if _MoEAuxLossAutoScaler.main_loss_backward_scale is None:
            # Prefer mint operator to create Tensor
            _MoEAuxLossAutoScaler.main_loss_backward_scale = Tensor(
                1.0, dtype=aux_loss.dtype
            )

        aux_loss_backward_scale = _MoEAuxLossAutoScaler.main_loss_backward_scale
        scaled_aux_loss_grad = mint.ones_like(aux_loss) * aux_loss_backward_scale
        return grad_output, scaled_aux_loss_grad

    @staticmethod
    def set_loss_scale(scale: Tensor) -> None:
        """set the scale of the aux loss.

        Args:
            scale (Tensor): The scale value to set. Please ensure that the scale passed in
                            matches the scale of the main_loss.
        """
        if _MoEAuxLossAutoScaler.main_loss_backward_scale is None:
            _MoEAuxLossAutoScaler.main_loss_backward_scale = scale
        else:
            _MoEAuxLossAutoScaler.main_loss_backward_scale.copy_(scale)

class MoEAuxLossAutoScaler(nn.Cell):
    """
    Module wrapper for the custom LogSoftmax function.
    """

    @staticmethod
    def construct(output: Tensor, aux_loss: Tensor):
        """
        Forward pass for LogSoftmax.
        """
        return _MoEAuxLossAutoScaler.apply(output, aux_loss)

def compute_routing_scores_for_aux_loss(
    logits: Tensor,
    topk: int,
    score_function: str,
) -> Tuple[Tensor, Tensor]:
    """Compute routing scores based on the score function.

    Args:
        logits (Tensor): The logits tensor after gating, shape: [num_tokens, num_experts].
        topk (int): The number of top-k indices to compute.
        score_function (str): The score function to use. Can be either "softmax" or "sigmoid".
        fused (bool, optional): Whether to use the fused version. Defaults to False.
        padding_mask (Tensor, optional): Boolean mask indicating non-padding tokens.
                                         Shape in [num_tokens]. True for valid tokens,
                                         False for padding tokens. Defaults to None.

    Returns:
        Tuple[Tensor, Tensor]: The routing map and the normalized routing scores.
    """
    if score_function == "softmax":
        scores = mint.softmax(logits, dim=-1, dtype=mstype.float32)
    elif score_function == "sigmoid":
        scores = mint.sigmoid(logits)
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    _, top_indices = mint.topk(scores, k=topk, dim=1)
    routing_map = mint.zeros_like(logits).int()
    routing_map.scatter_(1, top_indices, 1).bool()
    return routing_map, scores


def get_moe_layer_wise_logging_tracker() -> dict:
    """Return the moe layer wise tracker."""
    # pylint: disable=W0602
    global _MOE_LAYER_WISE_LOGGING_TRACKER
    return _MOE_LAYER_WISE_LOGGING_TRACKER


def save_to_aux_losses_tracker(
    loss: Tensor,
    layer_number: int,
    num_layers: int,
) -> None:
    """Save the auxiliary loss for logging.
    Args:
        name (str): The name of the loss.
        loss (Tensor): The loss tensor.
        layer_number (int): Layer index of the loss.
        num_layers (int): The number of total layers.
    """
    # Skip aux loss logging if layer_number is None.
    if layer_number is None:
        return

    tracker = get_moe_layer_wise_logging_tracker()
    if not tracker:
        tracker["values"] = mint.zeros(num_layers)
    if isinstance(loss, DTensor):
        loss = loss.to_local()
    tracker["values"][layer_number] = tracker["values"][layer_number] + loss  # Aggregate the loss for the layer.


def clear_aux_losses_tracker() -> None:
    """Clear the auxiliary losses."""
    tracker = get_moe_layer_wise_logging_tracker()
    for _ in tracker:
        tracker["values"].zero_()
