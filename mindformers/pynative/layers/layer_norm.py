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
"""Fused normalization layers for transformer models.

This module provides fused implementations of commonly used normalization
layers, including LayerNorm and RMSNorm, implemented as MindSpore `nn.Cell`s.

These fused variants are designed for transformer-style models and support:
    - Parameter dtype control (e.g., fp32 parameters)
    - Computation dtype control (e.g., fp16/bf16 compute)
    - Factory-style construction via `FusedNorm`

Currently supported normalization types:
    - FusedLayerNorm
    - FusedRMSNorm
"""

__all__ = ["get_norm_cls"]

from mindspore import nn, dtype, Parameter, mint
from mindspore.ops import cast, rms_norm
from mindspore.common.initializer import initializer
from mindspore.mint.nn.functional import layer_norm


class FusedLayerNorm(nn.Cell):
    """
    Fused Layer Normalization cell.

    This class implements Layer Normalization over the last dimension
    of the input tensor using MindSpore's fused `layer_norm` functional
    interface.

    The implementation supports:
        - Explicit parameter dtype (gamma / beta)
        - Separate computation dtype for numerical stability
        - Automatic casting back to the original input dtype

    Args:
        dim (Union[int, tuple[int], list[int]]):
            Shape of the normalized dimension(s), typically the hidden size.
        eps (float, optional):
            Small constant added to variance for numerical stability.
            Default: 1e-5.
        compute_dtype (mindspore.dtype, optional):
            Data type used during LayerNorm computation.
            Default: mindspore.float32.

    Inputs:
        x (Tensor):
            Input tensor of shape (..., hidden_size). Layer normalization
            is applied over the last dimension.

    Outputs:
        Tensor:
            Normalized tensor with the same shape and dtype as the input.
    """

    def __init__(self, dim, eps=1e-5, compute_dtype=dtype.float32):
        super().__init__()
        self.compute_type = compute_dtype
        self.eps = eps

        self.layer_norm = layer_norm
        self.gamma = Parameter(
            mint.empty(dim, dtype=self.compute_type),
            name="gamma"
        )
        self.beta = Parameter(
            mint.empty(dim, dtype=self.compute_type),
            name="beta"
        )
        self.cast = cast

    def construct(self, x):
        """Apply fused Layer Normalization."""
        original_type = x.dtype
        compute_type = self.compute_type
        x = self.cast(x, compute_type)
        output = self.layer_norm(x, x.shape[-1], self.gamma, self.beta, self.eps)
        return self.cast(output, original_type)

    def reset_parameter(self):
        """Reset LayerNorm parameters for delayed initialization."""
        self.gamma.fill_(1.0)
        self.beta.zero_()


class FusedRMSNorm(nn.Cell):
    """
    Fused RMS Normalization cell.

    This class implements RMSNorm using MindSpore's fused `RmsNorm` operator.
    RMSNorm normalizes inputs based on the root mean square of activations
    and applies a learnable scale parameter.

    Compared to LayerNorm, RMSNorm:
        - Does not use a bias term
        - Often provides better numerical stability in large models

    Args:
        dim (Union[int, tuple[int], list[int]]):
            Shape of the normalized dimension(s), typically the hidden size.
        eps (float, optional):
            Small constant added for numerical stability.
            Default: 1e-5.
        compute_dtype (mindspore.dtype, optional):
            Data type used during RMSNorm computation.
            Default: mindspore.float32.

    Inputs:
        x (Tensor):
            Input tensor of shape (..., hidden_size). Normalization
            is applied over the last dimension.

    Outputs:
        Tensor:
            Normalized tensor with the same shape and dtype as the input.
    """

    def __init__(self, dim, eps=1e-5, compute_dtype=dtype.float32):
        super().__init__()
        self.compute_type = compute_dtype

        self.eps = eps
        self.weight = Parameter(mint.empty(dim, dtype=self.compute_type))

        self.norm = rms_norm
        self.cast = cast

    def construct(self, x):
        """Apply fused RMS Normalization."""
        original_type = x.dtype
        compute_type = self.compute_type
        x = self.cast(x, compute_type)
        weight = self.cast(self.weight, compute_type) if self.weight.dtype != compute_type else self.weight
        output = self.norm(x, weight, self.eps)[0]
        return self.cast(output, original_type)

    def reset_parameter(self):
        """Reset RMSNorm parameters for delayed initialization."""
        self.weight.fill_(1.0)


def get_norm_cls(
        normalization,
        fused_norm=True,
):
    """
    Supported normalization types:
        - "LayerNorm": returns a `FusedLayerNorm`
        - "RMSNorm": returns a `FusedRMSNorm`

    Args:
        normalization (str):
            Name of the normalization type. Must be either
            "LayerNorm" or "RMSNorm".
        fused_norm (bool):
            Whether to use fused normalization or not.

    Returns:
        nn.Cell:
            A class of `FusedLayerNorm` or `FusedRMSNorm`.

    Raises:
        ValueError:
            If an unsupported normalization type is specified.
    """

    if fused_norm:
        if normalization == "LayerNorm":
            return FusedLayerNorm
        if normalization == "RMSNorm":
            return FusedRMSNorm
        raise ValueError("Only 'LayerNorm' and 'RMSNorm' are currently supported.")
    raise ValueError("Only fused normalization layers are supported.")
