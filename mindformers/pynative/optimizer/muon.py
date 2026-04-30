# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Muon optimizer for pynative mode.

This module provides the Muon optimizer implementation for pynative mode training.
Unlike the static graph version, this version:
- Uses Python for loops instead of hyper_map + MultitypeFuncGraph
- Uses direct reshape operations instead of Morph operator
"""

from mindspore import mint
from mindspore.common import dtype as mstype
from mindspore.common.parameter import ParameterTuple
from mindspore.ops import functional as F, operations as P
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.common.tensor import Tensor

from hyper_parallel import DTensor
from hyper_parallel import SkipDTensorDispatch
from hyper_parallel.core.dtensor.dtensor import distribute_tensor

from mindformers.core import context as core_context
from mindformers.tools.logger import logger
from mindformers.pynative.optimizer.adamw import _run_adamw_opt


def _create_state_parameter(old_param, prefix, init='zeros'):
    """Create optimizer state parameter with the same shape and dtype as the original parameter."""
    param = old_param.clone(init)
    param.name = prefix + "." + old_param.name
    return param


def _to_full(tensor):
    """Return the full (gathered) tensor for a DTensor, otherwise pass through."""
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _to_local(tensor):
    """Return the local tensor for a DTensor, otherwise pass through."""
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def newton_schulz(x, dim_a, dim_b, eps, ns_steps, ns_coefficients, matmul_op, addmm_op):
    """Apply Newton-Schulz iteration."""
    a, b, c = ns_coefficients

    if dim_a > dim_b:
        x = x.mT
    # Ensure spectral norm is at most 1
    x = mint.nn.functional.normalize(x, p=2, dim=(-2, -1), eps=eps)
    # Perform the NS iterations
    for _ in range(ns_steps):
        # a_mat = x @ x.T
        a_mat = matmul_op(x, x.mT)

        # b_mat = b * a_mat + c * (a_mat @ a_mat)
        b_mat = addmm_op(a_mat, a_mat, a_mat, beta=b, alpha=c)

        # x = a * x + (b_mat @ x)
        x = addmm_op(x, b_mat, x, beta=a, alpha=1)
    if dim_a > dim_b:
        x = x.mT
    return x


def _apply_muon_update(
    gradient, muon_m, momentum, use_nesterov, param, lr, weight_decay,
    matched_adamw_rms, muon_split_fn, muon_merge_fn, param_name,
    eps, ns_steps, ns_coefficients):
    """Apply Muon optimizer update.

    Works for both DTensor (multi-card) and regular Tensor (single-card) gradients.
    When ``gradient`` is a DTensor, the full tensor is gathered to run Newton-Schulz,
    and the result is redistributed back to the original sharding before applying the
    update. Regular tensors skip the gather/redistribute entirely.
    """
    op_cast = P.Cast()
    grad_is_dtensor = isinstance(gradient, DTensor)
    device_mesh = gradient.device_mesh if grad_is_dtensor else None
    placements = gradient.placements if grad_is_dtensor else None

    gradient = _to_local(gradient)
    muon_m = _to_local(muon_m)

    m_fp32 = op_cast(muon_m, mstype.float32)
    gradient_fp32 = op_cast(gradient, mstype.float32)
    next_m = m_fp32 * momentum + gradient_fp32

    if use_nesterov:
        gradient_fp32 = gradient_fp32 + next_m * momentum
    else:
        gradient_fp32 = next_m

    ns_inputs = op_cast(gradient_fp32, mstype.bfloat16)
    ns_inputs = DTensor.from_local(ns_inputs, device_mesh, placements) if grad_is_dtensor else ns_inputs
    ns_inputs = _to_full(ns_inputs)
    ns_inputs_list = muon_split_fn(param_name, ns_inputs)
    x_list = []

    for ns_inputs_item in ns_inputs_list:
        ns_input_shape = ns_inputs_item.shape
        dim_a, dim_b = ns_input_shape[-2:]
        if len(ns_input_shape) == 2:
            matmul_op, addmm_op = mint.mm, mint.addmm
        else:
            matmul_op, addmm_op = mint.bmm, mint.baddbmm
        x = newton_schulz(
            ns_inputs_item, dim_a, dim_b, eps, ns_steps, ns_coefficients, matmul_op, addmm_op)
        x_list.append(x)

    x_ret = muon_merge_fn(param_name, x_list)
    x_ret = distribute_tensor(x_ret, device_mesh, placements).to_local() if grad_is_dtensor else x_ret

    with SkipDTensorDispatch():
        param_fp32 = op_cast(param, mstype.float32) * (1 - lr * weight_decay)
        adjusted_ratio = mint.sqrt(op_cast(max(dim_a, dim_b), mstype.float32)) * matched_adamw_rms
        adjusted_lr = lr * adjusted_ratio
        update_with_lr = adjusted_lr * x_ret
        next_param = param_fp32 - update_with_lr.reshape(param_fp32.shape)
        param.copy_(op_cast(next_param, F.dtype(param)))
        muon_m.copy_(op_cast(next_m, F.dtype(muon_m)))
    return op_cast(next_param, F.dtype(param))


class Muon(Optimizer):
    """
    Muon optimizer implementation for pynative mode.

    Args:
        params: model parameters to optimize.
        learning_rate (float): Learning rate. Default: ``2e-2``.
        weight_decay (float): Weight decay factor. Default: ``0.1``.
        matched_adamw_rms (float): RMS matching parameter for AdamW. Default: ``0.2``.
        momentum (float): Momentum factor. Default: ``0.95``.
        nesterov (bool): Whether to use Nesterov momentum. Default: ``True``.
        ns_steps (int): Number of Newton-Schulz steps. Default: ``5``.
        ns_coefficients (tuple): Newton-Schulz coefficients. Default: ``(3.4445, -4.7750, 2.0315)``.
        adamw_betas (tuple): Beta parameters for AdamW. Default: ``(0.95, 0.95)``.
        adamw_eps (float): Epsilon for AdamW. Default: ``1e-8``.
        qk_clip_enabled (bool): Whether to apply QK clip scaling. Default: ``True``.
        qk_clip_threshold (float): QK clip threshold. Default: ``100``.
        model: The model model. Default: ``None``.
    """

    def __init__(
        self,
        params,
        learning_rate=2e-2,
        weight_decay=0.1,
        matched_adamw_rms=0.2,
        momentum=0.95,
        eps=1e-7,
        nesterov=True,
        ns_steps=5,
        ns_coefficients=(3.4445, -4.7750, 2.0315),
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        qk_clip_enabled=True,
        qk_clip_threshold=100,
        model=None,
        **kwargs,
    ):
        super().__init__(learning_rate, params, weight_decay)
        if kwargs.get('swap', False):
            raise ValueError("Muon does not support swap.")

        self._verify_model(model)

        self.beta1, self.beta2 = adamw_betas
        self.adamw_eps = adamw_eps
        self.muon_momentum = momentum
        self.eps = eps
        self.matched_adamw_rms = matched_adamw_rms
        self.use_nesterov = nesterov
        self._verify_config(ns_steps, ns_coefficients, qk_clip_enabled, qk_clip_threshold)
        self.ns_steps = ns_steps
        self.ns_coefficients = tuple(ns_coefficients)
        self.qk_clip_enabled = qk_clip_enabled
        self.param_name_tuple = tuple(p.name for p in self._parameters)

        self.muon_split_fn, self.muon_merge_fn = model.make_model_muon_fns()
        self.logit_threshold = Tensor([qk_clip_threshold], dtype=mstype.float32) if qk_clip_enabled else None

        self._initialize_state(model)

        self.model = model

    @staticmethod
    def _verify_model(model):
        """Verify if the model is compatible with Muon optimizer."""
        if model is None:
            raise ValueError("Model must be provided for Muon optimizer.")

        if core_context.is_legacy_model():
            raise ValueError("Muon does not support Legacy Model.")

        config = model.get_gpt_transformer_config()

        if not config.multi_latent_attention:
            raise ValueError("Current Muon implementation only supports models with Multi-Latent Attention enabled.")

    @staticmethod
    def _verify_config(ns_steps, ns_coefficients, qk_clip_enabled, qk_clip_threshold):
        """Validate Newton-Schulz and QK-clip hyperparameters."""
        if ns_steps <= 0:
            raise ValueError(f"ns_steps must be positive, got {ns_steps!r}.")
        if len(ns_coefficients) != 3:
            raise ValueError(f"ns_coefficients must have 3 elements, got {ns_coefficients!r}.")
        if qk_clip_enabled and qk_clip_threshold <= 0:
            raise ValueError(f"qk_clip_threshold must be positive, got {qk_clip_threshold!r}.")

    def _initialize_state(self, model):
        """Create Muon momentum and AdamW moment state in one pass over self._parameters."""
        muon_filter = model.get_muon_filter()

        muon_m = []
        moments1 = []
        moments2 = []
        state_indices = []
        use_muon = []

        for param in self._parameters:
            if muon_filter(param):
                state_indices.append(len(muon_m))
                muon_m.append(_create_state_parameter(param, "muon_m"))
                use_muon.append(True)
                logger.info(f"Muon apply: {type(param)=}, {param.name=}")
            else:
                state_indices.append(len(moments1))
                moments1.append(_create_state_parameter(param, "adam_m"))
                moments2.append(_create_state_parameter(param, "adam_v"))
                use_muon.append(False)
                logger.info(f"Adam apply: {type(param)=}, {param.name=}")

        self.muon_m = ParameterTuple(muon_m)
        self.moments1 = ParameterTuple(moments1)
        self.moments2 = ParameterTuple(moments2)
        self.use_muon = tuple(use_muon)
        self.state_indices = tuple(state_indices)

    def construct(self, gradients):
        """Construct method for optimizer.

        Args:
            gradients: Gradients for optimization.

        Returns:
            Updated gradients after optimization.
        """
        gradients = self.flatten_gradients(gradients)
        weight_decay = self.get_weight_decay()
        lr = self.get_lr()
        self.assignadd(self.global_step, self.global_step_increase_tensor)

        step = self.global_step
        bias_correction1 = 1.0 - self.beta1 ** step
        bias_correction2 = 1.0 - self.beta2 ** step
        one_minus_beta2 = 1.0 - self.beta2

        optim_result = []
        for i, (param, gradient, use_muon) in enumerate(zip(self._parameters, gradients, self.use_muon)):
            param_name = self.param_name_tuple[i]

            if "max_logits_val" in param_name:
                optim_result.append(P.Cast()(gradient, F.dtype(param)))
                continue

            if not self.optim_filter[i]:
                optim_result.append(gradient)
                continue

            if self.is_group:
                param_lr = lr[i] if self.is_group_lr else lr
                param_wd = weight_decay[i]
            else:
                param_lr = lr
                param_wd = weight_decay

            if use_muon:
                muon_m = self.muon_m[self.state_indices[i]]
                result = _apply_muon_update(
                    gradient, muon_m, self.muon_momentum,
                    self.use_nesterov, param, param_lr, param_wd,
                    self.matched_adamw_rms, self.muon_split_fn, self.muon_merge_fn,
                    param_name, self.eps,
                    self.ns_steps, self.ns_coefficients)
            else:
                state_idx = self.state_indices[i]
                with SkipDTensorDispatch():
                    _run_adamw_opt(
                        self.beta1, self.beta2, self.adamw_eps,
                        param_lr, param_wd, param, gradient,
                        self.moments1[state_idx], self.moments2[state_idx],
                        True, bias_correction1, bias_correction2, one_minus_beta2)
                    result = param

            optim_result.append(result)

        if self.qk_clip_enabled:
            self.model.allreduce_max_attention_logit()
            self.model.apply_qk_clip_scaling(
                self.logit_threshold,
                self.muon_split_fn,
                self.muon_merge_fn,
            )

        return optim_result
