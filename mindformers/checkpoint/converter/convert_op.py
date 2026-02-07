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
"""Converter operator for translate MindSpore Transformers weight with Hugging Face weight."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

from mindformers.parallel_core.transformer_config import TransformerConfig


@dataclass
class ConvertOp(ABC):
    """
    Base class for weight conversion operations.

    Referenced from ROLL library design, supports bidirectional conversion:
    - HF → MF: Used when loading HuggingFace weights
    - MF → HF: Used when exporting to HuggingFace format

    Attributes:
        hf_names: List of HuggingFace weight names
        mf_names: List of MindSpore Transformers weight names
        mf_config: MindSpore Transformers model configuration (for getting num_heads and other parameters)
    """
    hf_names: Union[str, List[str]]
    mf_names: Union[str, List[str]]
    mf_config: TransformerConfig = None

    def __post_init__(self):
        if isinstance(self.hf_names, str):
            self.hf_names = [self.hf_names]
        if isinstance(self.mf_names, str):
            self.mf_names = [self.mf_names]

    def __call__(
            self,
            name_to_weight: Dict[str, np.ndarray],
            mf_to_hf: bool = False
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Execute conversion.

        Args:
            name_to_weight: Input weight dictionary
            mf_to_hf: Conversion direction
                - False: HF → MF (default)
                - True: MF → HF

        Returns:
            Converted weight dictionary, returns None when weights are incomplete
        """
        required_names = self.mf_names if mf_to_hf else self.hf_names
        if len(required_names) > len(name_to_weight):
            return None

        if mf_to_hf:
            return self.mf_to_hf(name_to_weight)
        return self.hf_to_mf(name_to_weight)

    @staticmethod
    def _name_to_pattern(name: str):
        return name.replace(".", r"\.").replace("{}", "(.*)")

    def is_required_name(self, name, mf_name: bool):
        required_names = self.mf_names if mf_name else self.hf_names
        if name in required_names:
            return True
        for pattern in required_names:
            re_pattern = self._name_to_pattern(pattern)
            if re.match(re_pattern, name):
                return True
        return False

    def _to_names_and_weights(
            self,
            from_names: List[str],
            to_names: List[str],
            name_to_weight: Dict[str, np.ndarray]
    ) -> Tuple[List[str], List[np.ndarray]]:
        """Extract weights from input dictionary and compute target names"""
        weights = []
        match = None
        for from_name in from_names:
            if from_name in name_to_weight:
                weight = name_to_weight[from_name]
            elif "{}" in from_name:
                re_pattern = self._name_to_pattern(from_name)
                for name in name_to_weight:
                    match = re.findall(re_pattern, name)
                    if match:
                        weight = name_to_weight[name]
                        break
                if not match:
                    raise ValueError(f"Cannot find match {from_name} in {name_to_weight.keys()}")
            else:
                raise ValueError(f"Cannot find {from_name} in {name_to_weight.keys()}")
            weights.append(weight)

        if match:
            to_names = [to_name.format(*match) for to_name in to_names]

        return to_names, weights

    def hf_to_mf(self, name_to_weight: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """HF → MF conversion"""
        names, weights = self._to_names_and_weights(
            self.hf_names, self.mf_names, name_to_weight
        )
        mf_weights = self._hf_to_mf(weights)
        if not isinstance(mf_weights, list):
            mf_weights = [mf_weights]

        if len(names) != len(mf_weights):
            raise ValueError(f"Names and weights length mismatch: names: {names}, weights: {mf_weights}")

        return {
            names[i]: mf_weights[i]
            for i in range(len(names))
        }

    def mf_to_hf(self, name_to_weight: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """MF → HF conversion"""
        names, weights = self._to_names_and_weights(
            self.mf_names, self.hf_names, name_to_weight
        )
        hf_weights = self._mf_to_hf(weights)
        if not isinstance(hf_weights, list):
            hf_weights = [hf_weights]

        if len(names) != len(hf_weights):
            raise ValueError(f"Names and weights length mismatch: names: {names}, weights: {hf_weights}")
        return {
            names[i]: hf_weights[i]
            for i in range(len(names))
        }

    @abstractmethod
    def _hf_to_mf(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """Convert HuggingFace weights to MindSpore Transformers weights (implemented by subclasses)"""
        raise NotImplementedError()

    @abstractmethod
    def _mf_to_hf(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """Convert MindSpore Transformers weights to HuggingFace weights (implemented by subclasses)"""
        raise NotImplementedError()


@dataclass
class RenameConvertOp(ConvertOp):
    """
    Rename operation (1:1 mapping).

    For bidirectional conversion, only parameter names are modified, weight values remain unchanged.
    """

    def __post_init__(self):
        super().__post_init__()

        if not ((len(self.hf_names) == 1) and (len(self.mf_names) == 1)):
            raise ValueError(f"RenameConvertOp only supports one name as target: {self.hf_names} {self.mf_names}")

    def _hf_to_mf(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        return weights

    def _mf_to_hf(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        return weights


@dataclass
class ConcatConvertOp(ConvertOp):
    """
    Concatenation operation (N:1 mapping).

    HF → MF: np.concatenate() concatenates multiple weights
    MF → HF: np.split() splits into multiple weights
    """
    dim: int = 0
    split_sizes: List[int] = None  # Optional: specify size of each HF weight

    def __post_init__(self):
        super().__post_init__()

        if not (len(self.hf_names) == 1) != (len(self.mf_names) == 1):
            raise ValueError(
                f"ConcatConvertOp only supports the mapping of 'N hf_name to 1 mf_name' "
                f"or 'N mf_name to 1 hf_name'，"
                f"but got hf_names: `{self.hf_names}`, mf_names: `{self.mf_names}`."
            )

    def _hf_to_mf(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        # Step 1: Stack along custom dimension self.dim (ndim after stacking = input_ndim + 1)
        stacked = np.stack(weights, axis=self.dim)  # Add 1 dimension for weight index

        # Step 2: Dynamically construct transpose dimensions, swap "stack dimension" and "original concat dimension" to achieve interleaving
        axes = list(range(stacked.ndim))  # List of dimension indices after stacking
        axes[self.dim], axes[self.dim + 1] = axes[self.dim + 1], axes[self.dim]  # Swap adjacent dimensions
        transposed = stacked.transpose(axes)  # After transpose, still maintains ndim = input_ndim + 1

        # Step 3: Construct new shape, merge two related dimensions while preserving original ndim (no flattening)
        # Idea: Replace original concat dimension with (num_weights * single_weight_concat_dim_size), remove dimension added by stacking
        new_shape = list(transposed.shape)
        new_shape[self.dim] = transposed.shape[self.dim] * transposed.shape[self.dim + 1]
        new_shape.pop(self.dim + 1)

        # Step 4: Reshape to get final interleaved result (ndim = input_ndim)
        interleaved_concat = transposed.reshape(new_shape)

        return [interleaved_concat]

    def _mf_to_hf(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """Split concatenated MF weights back to HF weights"""
        raise ValueError("Currently, ConcatConvertOp does not support MF → HF conversion")


@dataclass
class StackConvertOp(ConvertOp):
    """
    Stack operation (N:1 mapping).

    HF → MF: np.stack() stacks multiple weights
    MF → HF: np.split() splits into multiple weights
    """
    dim: int = 0

    def __post_init__(self):
        super().__post_init__()

        if not (len(self.hf_names) == 1) != (len(self.mf_names) == 1):
            raise ValueError(
                f"StackConvertOp only supports the mapping of 'N hf_name to 1 mf_name' "
                f"or 'N mf_name to 1 hf_name'，"
                f"but got hf_names: `{self.hf_names}`, mf_names: `{self.mf_names}`."
            )

    def _hf_to_mf(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        return [np.stack(weights, axis=self.dim)]

    def _mf_to_hf(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """Split stacked MF weights back to HF weights"""
        raise ValueError("Currently, StackConvertOp does not support MF → HF conversion")


@dataclass
class QKVConvertOp(ConvertOp):
    """
    QKV fusion operation (3:1 mapping).

    HF → MF: Interleave and concatenate Q, K, V into QKV in GQA format
    MF → HF: Split QKV into independent Q, K, V

    GQA format: [ng, (nh/ng + 2) * kv_channels, hidden_size]
    where ng = num_query_groups, nh = num_attention_heads
    """
    num_attention_heads: int = None
    num_query_groups: int = None
    kv_channels: int = None
    hidden_size: int = None
    tensor_model_parallel_size: int = None

    def __post_init__(self):
        super().__post_init__()

        if len(self.hf_names) != 3:
            raise ValueError(f"QKVConvertOp only support three hf_names: `{self.hf_names}`.")
        if len(self.mf_names) != 1:
            raise ValueError(f"QKVConvertOp only support one mf_name: `{self.mf_names}`.")

    def set_model_config(self, config):
        """Set parameters from model configuration"""
        self.num_attention_heads = config.num_attention_heads
        self.num_query_groups = config.num_query_groups
        self.kv_channels = config.kv_channels
        self.hidden_size = config.hidden_size
        self.tensor_model_parallel_size = config.tensor_model_parallel_size

    def _hf_to_mf(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """Convert Q, K, V weights to QKV fused weight"""
        if len(weights) < 3:
            raise ValueError(f"Expected at least 3 weights for QKV conversion, but got {len(weights)}")
        q_weight, k_weight, v_weight = weights
        nh = self.num_attention_heads // self.tensor_model_parallel_size
        ng = self.num_query_groups // self.tensor_model_parallel_size
        dim = self.kv_channels

        if nh % ng != 0:
            raise ValueError(
                f"Number of attention heads per group ({nh}) must be divisible by number of query groups ({ng})")

        # Reshape and concatenate (GQA interleaved format)
        mf_qkv_weight = np.concatenate([
            q_weight.reshape((ng, dim * nh // ng, -1)),
            k_weight.reshape((ng, dim, -1)),
            v_weight.reshape((ng, dim, -1)),
        ], axis=1).reshape((-1, self.hidden_size))

        return [mf_qkv_weight]

    def _mf_to_hf(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """
        Split QKV fused weight into independent Q, K, V weights.

        Referenced from ROLL's QKVConvertOp._mf_to_hf() implementation.
        """
        raise ValueError("Currently, QKVConvertOp does not support MF → HF conversion")


@dataclass
class QKVBiasConvertOp(ConvertOp):
    """
    QKV Bias fusion operation (3:1 mapping).

    Similar to QKVConvertOp, but handles 1D bias vectors.
    """
    num_attention_heads: int = None
    num_query_groups: int = None
    kv_channels: int = None
    tensor_model_parallel_size: int = None

    def set_model_config(self, config):
        """Set parameters from model configuration"""
        self.num_attention_heads = config.num_attention_heads
        self.num_query_groups = config.num_query_groups
        self.kv_channels = config.kv_channels
        self.tensor_model_parallel_size = config.tensor_model_parallel_size

    def _hf_to_mf(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """Convert Q, K, V bias to QKV fused bias"""
        if len(weights) < 3:
            raise ValueError(f"Expected at least 3 bias weights for QKV bias conversion, but got {len(weights)}")
        q_bias, k_bias, v_bias = weights
        nh = self.num_attention_heads // self.tensor_model_parallel_size
        ng = self.num_query_groups // self.tensor_model_parallel_size
        dim = self.kv_channels

        if nh % ng != 0:
            raise ValueError(
                f"Number of attention heads per group ({nh}) must be divisible by number of query groups ({ng})"
            )

        mf_qkv_bias = np.concatenate([
            q_bias.reshape((ng, dim)),
            k_bias.reshape((ng, dim)),
            v_bias.reshape((ng, dim)),
        ], axis=1).reshape(-1)

        return [mf_qkv_bias]

    def _mf_to_hf(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """Split QKV fused bias into independent Q, K, V bias"""
        raise ValueError("Currently, QKVBiasConvertOp does not support MF → HF conversion")
