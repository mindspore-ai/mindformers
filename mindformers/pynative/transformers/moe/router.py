# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mixture of Experts (MoE) modules for pynative mode."""
from typing import Tuple, Optional

from mindspore import nn, Tensor, mint, ops
from mindspore.common.parameter import Parameter
from mindspore.common import dtype as mstype

from mindformers.parallel_core.transformer_config import TransformerConfig
from mindformers.parallel_core.utils.init_method import init_method_normal
from mindformers.tools.logger import logger
from .moe_utils import (
    compute_routing_scores_for_aux_loss,
    switch_load_balancing_loss_func,
    save_to_aux_losses_tracker,
    MoEAuxLossAutoScaler,
)


class TopKRouter(nn.Cell):
    """This class implements token-choice routing. In token-choice top-K routing, each token is
    routed to top K experts based on the router scores.

    Optionally supports node-limited (group-limited) routing where experts are divided into groups
    (e.g., by node), and only num_limited_groups groups are considered before selecting top_k experts.
    This reduces cross-node communication in distributed settings.

    Args:
        config (TransformerConfig): Transformer configuration object containing:
            - hidden_size (int): Dimension of input tokens.
            - num_moe_experts (int): Number of experts in each moe layer.
            - moe_router_num_groups (int | None): Number of expert groups for node-limited routing. If None, standard
              top-k routing is used. Must be a divisor of num_experts.
            - moe_router_group_topk (int | None): Number of groups to select in node-limited routing. Required when
              moe_router_num_groups is set.
            - moe_router_topk (int): Number of experts each token will be routed to in token-choice routing.
            - moe_router_score_function (Literal["softmax", "sigmoid", "sqrtsoftplus"]): Score function
              applied to router logits.
            - norm_topk_prob (bool): Whether to normalize the routing scores when using sigmoid.
            - moe_router_topk_scaling_factor (float): Scaling factor applied to the routing scores.
            - moe_router_force_expert_balance (bool): Whether to force load balance via round-robin
              routing. Default: False.
    """

    def __init__(
            self,
            config: TransformerConfig,
            layer_number: int = 0,
            is_mtp_layer: bool = False,
    ):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.is_mtp_layer = is_mtp_layer

        # Extract parameters from config
        dim = config.hidden_size
        num_experts = config.num_moe_experts
        num_expert_groups = config.moe_router_num_groups
        num_limited_groups = config.moe_router_group_topk
        top_k = config.moe_router_topk
        score_func = config.moe_router_score_function
        route_norm = config.norm_topk_prob
        route_scale = (
            config.moe_router_topk_scaling_factor
            if config.moe_router_topk_scaling_factor is not None
            else 1.0
        )
        self.weight = Parameter(
            mint.empty((num_experts, dim), dtype=config.moe_router_dtype), name="weight"
        )
        self.num_experts = num_experts
        self.num_expert_groups = num_expert_groups or 0
        self.num_limited_groups = num_limited_groups
        self.top_k = top_k
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale
        self._debug_force_load_balance = config.moe_router_force_expert_balance
        # NOTE: add config for per token loss
        self.calculate_per_token_loss = False
        self.moe_aux_loss_auto_scaler = MoEAuxLossAutoScaler()

        # Hash-based MoE routing (RFC §3.4.2.2, aligns Megatron router.py:181-202).
        # The leading moe_n_hash_layers decoder layers (0-based decoder layer index) select
        # experts from a fixed tid2eid lookup table instead of learned top-k routing
        # (aligns mindspeed convert_ckpt_deepseek4.py:246 hash_layer = i < n_hash_layers).
        # MTP layers never use hash routing.
        if config.moe_n_hash_layers > 0 and layer_number is None:
            raise ValueError(f"layer_number({layer_number}) is required for the hash-based router.")
        self.is_hash_layer = (
            not self.is_mtp_layer
            and config.moe_n_hash_layers > 0
            and layer_number < config.moe_n_hash_layers
        )
        if self.is_hash_layer:
            # DSv4-Pro ships a pre-trained tid2eid table in its inference checkpoint;
            # no public initialization recipe is documented. Round-robin is used here
            # only as a placeholder so the layer is runnable from scratch.
            # framework adaptation: torch register_buffer -> MindSpore Parameter(requires_grad=False).
            # Build the int32 index table directly (no astype chain, NPU-friendly).
            vocab_size = config.actual_vocab_size
            self.tid2eid = Parameter(
                mint.empty((vocab_size, top_k), dtype=mstype.int32),
                name="tid2eid",
                requires_grad=False,
            )
            logger.info(f"Using hash routing for layer {layer_number}")
        else:
            self.tid2eid = None

        # Auxiliary loss components. Hash layers do not produce aux loss
        # (aligns Megatron router.py:200-202: expert bias / aux loss disabled on hash layers).
        self.moe_aux_loss_coeff = 0.0
        if config.moe_aux_loss_coeff and not self.is_hash_layer:
            self.moe_aux_loss_coeff = config.moe_aux_loss_coeff

        self.aux_loss_type = config.moe_router_load_balancing_type
        # Initialize global tokens per expert for global aux loss
        if self.aux_loss_type == "global_aux_loss":
            self.global_tokens_per_expert = Parameter(
                mint.empty(num_experts, dtype=mstype.float32),
                name="global_tokens_per_expert",
                requires_grad=False,
            )
            self.ga_steps = Parameter(
                mint.empty((), dtype=mstype.float32),
                name="ga_steps",
                requires_grad=False,
            )
        else:
            self.global_tokens_per_expert = None
            self.ga_steps = None

        # Initialize operators in __init__
        self.linear = mint.nn.functional.linear
        self.sigmoid = mint.nn.functional.sigmoid
        self.softmax = mint.nn.functional.softmax
        self.softplus = mint.nn.functional.softplus
        self.sqrt = mint.sqrt
        self.cast = ops.cast
        self.arange = mint.arange
        self.gather = mint.gather
        self.reshape = mint.reshape
        self.topk = mint.topk
        self.sum = mint.sum
        self.ones_like = mint.ones_like
        self.div = mint.div
        self.mul = mint.mul
        self.histc = mint.histc
        self.stop_gradient = ops.stop_gradient


    def reset_parameter(self):
        """Reset router weights for delayed initialization.

        Under delayed (meta-device) init the model is built on meta then materialized via
        ``to_empty()``, which leaves all Parameters as uninitialized real memory. Besides the
        learned ``weight``, the router also owns deterministic non-grad Parameters
        (``tid2eid``, ``global_tokens_per_expert``, ``ga_steps``) whose ``__init__`` values do
        not survive ``to_empty()``. Re-run their deterministic init in-place here so hash
        routing reads a valid table instead of garbage.
        """
        self.weight.normal_(mean=0.0, std=0.01)

        if self.is_hash_layer:
            # Rebuild the int32 hash table exactly as __init__ does (see is_hash_layer branch).
            vocab_size = self.config.actual_vocab_size
            ids = self.arange(vocab_size, dtype=mstype.int32)
            tid2eid = mint.stack(
                [(ids + k) % self.num_experts for k in range(self.top_k)], dim=1
            )
            self.tid2eid.set_data(tid2eid)

        if self.aux_loss_type == "global_aux_loss":
            self.global_tokens_per_expert.zero_()
            self.ga_steps.zero_()

    def _debug_force_load_balance_routing(
            self, scores: Tensor, selected_experts_indices: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Balanced round-robin expert assignment.
        Returns (selected_experts_indices [N, K] LongTensor, top_scores [N, K] FloatTensor).
        """
        n_tokens = scores.shape[0]
        # Round-robin indices with exact balance
        round_robin_indices = (
                self.reshape(
                    self.arange(n_tokens * self.top_k, dtype=mstype.int64),
                    (n_tokens, self.top_k),
                )
                % self.num_experts
        )
        # Overwrite the values of the normal top-k indices instead of using the
        # locally built round-robin table directly: the top-k indices carry
        # scores' (possibly distributed) layout under TP, while a standalone
        # arange-based index has no layout and breaks the gather below.
        selected_experts_indices = self.cast(
            selected_experts_indices * 0 + round_robin_indices, mstype.int64
        )
        top_scores = self.gather(scores, dim=1, index=selected_experts_indices)  # [N,K]
        return selected_experts_indices, top_scores

    def _get_node_limited_routing_scores(
            self,
            scores_for_choice: Tensor,
    ) -> Tensor:
        """Select num_limited_groups groups based on group scores,
        and set expert scores in non-selected groups as -inf

        Args:
            scores_for_choice: Router scores with expert_bias (if any), shape (bs*slen, num_experts)

        Returns:
            scores_for_choice: shape (bs*slen, num_experts)
        """
        if self.num_limited_groups is None:
            raise ValueError(
                "num_limited_groups must be set when num_expert_groups is set"
            )
        if self.num_expert_groups is None:
            raise ValueError(
                "num_expert_groups must be set when using node-limited routing"
            )
        if self.num_experts % self.num_expert_groups != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by "
                f"num_expert_groups ({self.num_expert_groups})"
            )
        experts_per_group = self.num_experts // self.num_expert_groups
        if experts_per_group < 2:
            raise ValueError(f"experts_per_group ({experts_per_group}) must be >= 2")
        scores_grouped = self.reshape(
            scores_for_choice, (-1, self.num_expert_groups, experts_per_group)
        )
        top2_scores_in_group, _ = self.topk(scores_grouped, 2, dim=-1)
        group_scores = self.sum(top2_scores_in_group, dim=-1)
        _, group_idx = self.topk(
            group_scores, k=self.num_limited_groups, dim=-1, sorted=False
        )
        group_mask = self.ones_like(group_scores, dtype=mstype.bool)
        group_mask.scatter_(1, group_idx, False)  # False = selected groups (keep)
        # Mask out experts from non-selected groups
        scores_for_choice = scores_grouped.masked_fill(
            group_mask.unsqueeze(-1), float("-inf")
        )
        scores_for_choice = self.reshape(scores_for_choice, (-1, self.num_experts))

        return scores_for_choice

    def _hash_routing(
            self, logits: Tensor, input_ids: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Hash-based routing: expert indices come from the tid2eid lookup table.

        Scores are still computed from the gating logits for weight computation,
        but expert selection is determined by the pre-computed hash table.
        Aligns Megatron router.py:616-655.

        Framework adaptation (RFC §3.4.2.2): unlike Megatron, which scatters the
        result back to a dense ``[num_tokens, num_experts]`` routing_probs / routing_map,
        the PyNative dispatcher consumes the ``[b*s, top_k]`` form, so this returns
        ``(top_scores, selected_experts_indices)`` directly from the tid2eid lookup
        (no dense scatter).

        Args:
            logits (Tensor): Gating logits, shape ``[num_tokens, num_experts]``.
            input_ids (Tensor): Token IDs, shape ``[b, s]``.

        Returns:
            Tuple[Tensor, Tensor]:
                - top_scores (Tensor): combine weights for selected experts, ``[num_tokens, top_k]``.
                - selected_experts_indices (Tensor): expert indices, ``[num_tokens, top_k]`` int64.
        """
        # Scores are only used for combine weights; expert selection is from tid2eid.
        if self.score_func == "sigmoid":
            scores = self.sigmoid(self.cast(logits, mstype.float32))
        elif self.score_func == "softmax":
            scores = self.softmax(self.cast(logits, mstype.float32), dim=1)
        elif self.score_func == "sqrtsoftplus":
            scores = self.sqrt(self.softplus(self.cast(logits, mstype.float32)))
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        # input_ids is [b, s] from the model, but hidden_states are [s, b, h] and get
        # flattened to [s*b, h]; transpose then flatten to align token order
        # (aligns Megatron router.py:640-642 .T.reshape(-1)).
        flat_ids = self.reshape(input_ids.T, (-1,))
        # int64 keeps the returned indices consistent with the non-hash path (construct)
        # and matches the documented int64 return type long().
        selected_experts_indices = self.cast(
            mint.index_select(self.tid2eid, 0, flat_ids),
            mstype.int64
        )

        top_scores = self.gather(scores, dim=1, index=selected_experts_indices)
        if self.score_func != "softmax":
            denominator = self.sum(top_scores, dim=-1, keepdim=True) + 1e-20
            top_scores = self.div(top_scores, denominator)
        top_scores = self.mul(top_scores, self.route_scale)

        return top_scores, selected_experts_indices

    def construct(
            self, x: Tensor, expert_bias: Optional[Tensor] = None,
            input_ids: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x (Tensor): Input tensor with shape ``(bs*slen, dim)``.
            expert_bias (Tensor | None, optional): Optional bias tensor for experts with shape ``(num_experts,)``.
                Used for load balancing. Defaults to None.
            input_ids (Tensor | None, optional): Token IDs with shape ``(b, s)``. Required on hash
                layers (consumed by the tid2eid lookup); ignored on non-hash layers. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - top_scores (Tensor):
                    Routing scores for selected experts with shape ``(bs*slen, top_k)``.
                - selected_experts_indices (Tensor):
                    Expert indices selected for each token with shape ``(bs*slen, top_k)``.
                - num_tokens_per_expert (Tensor):
                    Number of tokens assigned to each expert with shape ``(num_experts,)``.
        """
        # preprocess input shape (slen, bs, num_experts)
        seq_length, bsz, dim = x.shape
        x = self.reshape(x, (-1, dim))

        # scores shape (bs*slen, num_experts)
        # Compute gate in float32 to help stability of expert load balancing.
        x = self.cast(x, self.config.moe_router_dtype)
        logits = self.linear(x, self.weight)

        # Hash-based routing: select experts from tid2eid, weights from gating scores.
        if self.is_hash_layer:
            if input_ids is None:
                raise ValueError(
                    "input_ids is required for hash-based routing but was None. "
                    "Ensure moe_n_hash_layers is set correctly and input_ids are passed."
                )
            top_scores, selected_experts_indices = self._hash_routing(logits, input_ids)
            num_tokens_per_expert = self.histc(
                selected_experts_indices,
                bins=self.num_experts,
                min=0,
                max=self.num_experts,
            )
            return top_scores, selected_experts_indices, num_tokens_per_expert

        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion
        if self.score_func == "sigmoid":
            scores = self.sigmoid(self.cast(logits, mstype.float32))
        elif self.score_func == "softmax":
            scores = self.softmax(self.cast(logits, mstype.float32), dim=1)
        elif self.score_func == "sqrtsoftplus":
            scores = self.sqrt(self.softplus(self.cast(logits, mstype.float32)))
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        scores_for_choice = scores if expert_bias is None else scores + expert_bias
        # Apply node-limited routing if configured
        if self.num_expert_groups > 0:
            scores_for_choice = self._get_node_limited_routing_scores(scores_for_choice)
        _, selected_experts_indices = self.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False
        )
        selected_experts_indices = self.cast(selected_experts_indices, mstype.int64)

        # top scores shape (bs*slen, top_k)
        # NOTE: The expert_bias is only used for routing. The gating value
        #       top_scores is still derived from the original scores.
        top_scores = self.gather(scores, dim=1, index=selected_experts_indices)

        # debug override: balanced round-robin routing
        if self._debug_force_load_balance:
            (
                selected_experts_indices,
                top_scores,
            ) = self._debug_force_load_balance_routing(scores, selected_experts_indices)

        if self.route_norm:
            denominator = self.sum(top_scores, dim=-1, keepdim=True) + 1e-20
            top_scores = self.div(top_scores, denominator)

        top_scores = self.mul(top_scores, self.route_scale)
        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = self.histc(
            selected_experts_indices,
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        # Compute auxiliary load-balancing loss and inject gradient into top_scores
        if self.moe_aux_loss_coeff > 0:
            top_scores = self._compute_aux_loss(
                logits,
                top_scores,
                seq_length,
                bsz,
            )

        return top_scores, selected_experts_indices, num_tokens_per_expert

    # pylint: disable=W0613
    def _compute_aux_loss(
            self,
            logits: Tensor,
            top_scores: Tensor,
            seq_length: int,
            bsz: int,
    ) -> Tensor:
        """
        Compute auxiliary load-balancing loss and inject its gradient into top_scores.

        Uses scores (with gradient) and routing_map (without gradient) to compute
        the load-balancing loss. The loss gradient is injected into the computation
        graph via MoEAuxLossAutoScaler, so it flows back to the router weights during
        backpropagation without modifying the main loss scalar.

        Args:
            logits (Tensor): Raw router logits before score function.
                Used to recompute routing_map and scores_for_aux_loss via
                ``compute_routing_scores_for_aux_loss``.
            top_scores (Tensor): Top-K routing scores for selected experts,
                shape ``[T, K]``. The aux loss gradient will be injected into
                this tensor via ``MoEAuxLossAutoScaler``.
            seq_length (int): Sequence length of the input, used by
                ``seq_aux_loss`` to reshape per-sequence statistics.
            bsz (int): Batch size, used by ``seq_aux_loss`` to average the
                loss across sequences.

        Returns:
            Tensor: top_scores with aux loss gradient injected.
        """

        routing_map_for_aux_loss, scores_for_aux_loss = compute_routing_scores_for_aux_loss(
            logits,
            self.top_k,
            self.score_func,
        )

        if self.aux_loss_type == "seq_aux_loss":
            top_scores = self._apply_seq_aux_loss(
                top_scores, scores_for_aux_loss, routing_map_for_aux_loss, seq_length, bsz
            )
        elif self.aux_loss_type == "global_aux_loss":
            top_scores = self._apply_global_aux_loss(
                top_scores, scores_for_aux_loss, routing_map_for_aux_loss
            )
        else:
            raise ValueError(f"Unknown aux_loss_type {self.aux_loss_type}")

        return top_scores

    def _apply_seq_aux_loss(
            self,
            top_scores: Tensor,
            scores_for_aux_loss: Tensor,
            routing_map: Tensor,
            seq_length: int,
            bsz: int,
    ) -> Tensor:
        """
        Compute per-sequence auxiliary loss and inject gradient.

        Args:
            top_scores (Tensor): Top-K routing scores, shape ``[T, K]``.
            scores_for_aux_loss (Tensor): Router probabilities, shape ``[T, E]``.
            routing_map (Tensor): Token-expert assignment map (detached), shape ``[T, E]``.
            seq_length (int): Sequence length.
            bsz (int): Batch size.

        Returns:
            Tensor: top_scores with aux loss gradient injected.
        """
        # NOTE: AllReduce tokens_per_expert over tp_cp_group for multi-card
        scores_for_aux_loss = scores_for_aux_loss.reshape((seq_length, -1),)
        tokens_per_expert = routing_map.reshape((seq_length, -1),).sum(dim=0)
        total_num_tokens = seq_length

        aux_loss = (
                switch_load_balancing_loss_func(
                    probs=scores_for_aux_loss,
                    tokens_per_expert=tokens_per_expert,
                    total_num_tokens=total_num_tokens,
                    topk=self.top_k,
                    num_experts=self.config.num_moe_experts,
                    moe_aux_loss_coeff=self.moe_aux_loss_coeff,
                )
                / bsz
        )

        top_scores = self._attach_and_log_aux_loss(
            top_scores, aux_loss, self.moe_aux_loss_coeff
        )
        return top_scores

    def _apply_global_aux_loss(
            self,
            top_scores: Tensor,
            scores_for_aux_loss: Tensor,
            routing_map: Tensor,
    ) -> Tensor:
        """
        Compute global-batch auxiliary loss with EMA accumulation and inject gradient.

        Args:
            top_scores (Tensor): Top-K routing scores, shape ``[T, K]``.
            scores_for_aux_loss (Tensor): Router probabilities, shape ``[T, E]``.
            routing_map (Tensor): Token-expert assignment map (detached), shape ``[T, E]``.

        Returns:
            Tensor: top_scores with aux loss gradient injected.
        """
        tokens_per_expert = routing_map.sum(dim=0)
        self.global_tokens_per_expert += tokens_per_expert
        self.ga_steps += 1
        averated_tokens_per_expert = self.global_tokens_per_expert / self.ga_steps

        num_tokens = scores_for_aux_loss.shape[0]
        # total_num_tokens = num_tokens * self.tp_dp_cp_group.size()
        total_num_tokens = num_tokens

        global_aux_loss = switch_load_balancing_loss_func(
            probs=scores_for_aux_loss,
            tokens_per_expert=averated_tokens_per_expert,
            total_num_tokens=total_num_tokens,
            topk=self.top_k,
            num_experts=self.config.num_moe_experts,
            moe_aux_loss_coeff=self.moe_aux_loss_coeff,
        )
        top_scores = self._attach_and_log_aux_loss(
            top_scores,
            global_aux_loss,
            self.moe_aux_loss_coeff,
        )
        return top_scores

    def _attach_and_log_aux_loss(
            self,
            top_scores: Tensor,
            aux_loss: Tensor,
            aux_loss_coeff: float,
    ) -> Tensor:
        """
        Inject aux_loss gradient into top_scores via MoEAuxLossAutoScaler
        and record the loss in MoETracker.

        Args:
            top_scores (Tensor): Top-K routing scores, shape ``[T, K]``.
            aux_loss (Tensor): Scalar auxiliary loss.
            aux_loss_coeff (float): Coefficient for aux_loss.

        Returns:
            Tensor: top_scores with aux loss gradient attached.
        """
        num_layers = self.config.num_layers
        if self.config.mtp_num_layers is not None:
            num_layers += self.config.mtp_num_layers

        save_to_aux_losses_tracker(
            aux_loss / aux_loss_coeff,
            self.layer_number,
            num_layers,
        )

        if self.calculate_per_token_loss:
            # Scale the aux_loss by the number of tokens.
            # The expected final scaling for aux_loss gradients is 1/(num_micro_batches * dp_size).
            # After commit 02648000, Megatron started using the number of total tokens to scale
            # gradients under the argument of calculate_per_token_loss,
            # which scales both the main_loss gradient and aux_loss gradient by
            # 1/(num_local_tokens * dp_size * num_micro_batches) in finalize_model_grads function.
            # To correct this scaling, we need to scale the aux_loss by num_local_tokens here.
            top_scores = self.moe_aux_loss_auto_scaler(top_scores, aux_loss * top_scores.shape[0])
        else:
            top_scores = self.moe_aux_loss_auto_scaler(top_scores, aux_loss)

        return top_scores

    def reset_global_aux_loss_tracker(self):
        """Reset the global aux loss tracker."""
        if self.global_tokens_per_expert is not None:
            self.global_tokens_per_expert = mint.zeros_like(self.global_tokens_per_expert)
            self.ga_steps = mint.zeros_like(self.ga_steps)
