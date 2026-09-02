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
from mindspore import get_auto_parallel_context
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
            if isinstance(match[0], tuple):
                match = match[0]
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
    Concatenation operation (N:1 mapping), with optional interleaving.

    HF → MF: interleaves values along ``dim`` by default for fused gated-MLP
    layouts; set ``interleaved=False`` for a plain ``np.concatenate`` layout.
    MF → HF: the exact inverse, a strided gather or a plain split.

    For a fused gated-MLP ``linear_fc1`` the layout is not a property of the
    mapping but of the model: ``use_interleaved_weight_layout_mlp`` picks
    ``MLPInterleaved`` over ``MLP`` (and ``SharedExpertMLPInterleaved`` over
    ``SharedExpertMLP``), which read the fused weight differently.
    :meth:`set_model_config` therefore takes ``interleaved`` from the config for
    those ops, so a declaration cannot go stale against the model it describes.
    """
    dim: int = 0
    split_sizes: List[int] = None  # Optional: specify size of each HF weight
    interleaved: bool = True  # Preserve the existing fused gated-MLP layout by default.

    def __post_init__(self):
        super().__post_init__()

        if not (len(self.hf_names) == 1) != (len(self.mf_names) == 1):
            raise ValueError(
                f"ConcatConvertOp only supports the mapping of 'N hf_name to 1 mf_name' "
                f"or 'N mf_name to 1 hf_name'，"
                f"but got hf_names: `{self.hf_names}`, mf_names: `{self.mf_names}`."
            )

    def is_gated_mlp_fc1(self) -> bool:
        """True for the fused gate|up weight of an MLP / shared-expert MLP.

        Keyed on ``linear_fc1`` so that concatenations which are not a gated MLP
        (MTP's ``eh_proj``, say) keep whatever their declaration asked for.
        """
        return any('linear_fc1' in name for name in self.mf_names)

    def set_model_config(self, config: TransformerConfig):
        """Take the fused-fc1 layout from the model config.

        ``use_interleaved_weight_layout_mlp`` selects ``MLPInterleaved`` over
        ``MLP`` in the layer spec, i.e. whether the fused gate|up weight is read
        row-interleaved or as two contiguous halves. Reading it here keeps the
        converter in step with the model the config builds; MoE routed experts
        are unaffected (they are ``ExpertsConvertOp`` and always contiguous).
        """
        self.mf_config = config
        if self.is_gated_mlp_fc1():
            interleaved = getattr(config, 'use_interleaved_weight_layout_mlp', None)
            if interleaved is not None:
                self.interleaved = bool(interleaved)

    def _hf_to_mf(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        if not self.interleaved:
            return [np.concatenate(weights, axis=self.dim)]

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
        """Split the concatenated MF weight back into the HF weights.

        Exact inverse of :meth:`_hf_to_mf`: when ``interleaved`` is set the parts
        alternate along ``dim`` (part0[0], part1[0], part0[1], ...), so they are
        recovered with a strided gather rather than a contiguous split.
        """
        weight = weights[0]
        num_parts = len(self.hf_names)
        if num_parts < 2:
            raise ValueError(
                f"ConcatConvertOp MF → HF requires at least 2 hf_names, got `{self.hf_names}`.")

        if self.interleaved:
            length = weight.shape[self.dim]
            if length % num_parts != 0:
                raise ValueError(
                    f"ConcatConvertOp: dim {self.dim} of the MF weight ({length}) is not "
                    f"divisible by the number of hf_names ({num_parts}).")
            return [np.ascontiguousarray(
                np.take(weight, np.arange(k, length, num_parts), axis=self.dim))
                    for k in range(num_parts)]

        if self.split_sizes:
            if len(self.split_sizes) != num_parts:
                raise ValueError(
                    f"ConcatConvertOp: split_sizes {self.split_sizes} does not match "
                    f"the number of hf_names ({num_parts}).")
            offsets = np.cumsum(self.split_sizes)[:-1]
            return [np.ascontiguousarray(w) for w in np.split(weight, offsets, axis=self.dim)]

        return [np.ascontiguousarray(w) for w in np.split(weight, num_parts, axis=self.dim)]


@dataclass
class ExpertsConvertOp(ConvertOp):
    """
    Experts operation (N:1 mapping).

    HF → MF: Converts HF experts weights to MF weights
    MF → HF: Converts MF experts weights to HF weights
    """
    num_moe_experts: int = None
    hidden_size: int = None
    moe_ffn_hidden_size: Optional[int] = None
    expert_parallel_size: int = None
    optimizer_parallel_size: int = None

    def __post_init__(self):
        super().__post_init__()
        if len(self.mf_names) != 1:
            raise ValueError(f"ExpertsConvertOp only support one mf_name: `{self.mf_names}`.")

    def set_model_config(self, config: TransformerConfig):
        self.mf_config = config
        self.num_moe_experts = config.num_moe_experts
        self.hidden_size = config.hidden_size
        self.moe_ffn_hidden_size = config.moe_ffn_hidden_size
        self.expert_parallel_size = config.expert_model_parallel_size
        self.optimizer_parallel_size = get_auto_parallel_context('optimizer_weight_shard_size')
        if not self.num_moe_experts:
            raise ValueError("The number of experts is not set.")

    def _hf_to_mf(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """Convert experts HF weights to MF weights"""
        result = []
        if len(weights) == 2:
            gate_weight = weights[0]
            up_weight = weights[1]
            gate_weight = gate_weight.reshape(self.num_moe_experts // self.expert_parallel_size
                                              // self.optimizer_parallel_size, -1, self.hidden_size)
            up_weight = up_weight.reshape(self.num_moe_experts // self.expert_parallel_size
                                          // self.optimizer_parallel_size, -1, self.hidden_size)
            gate_weight = gate_weight.transpose(0, 2, 1)
            up_weight = up_weight.transpose(0, 2, 1)
            if gate_weight is None or up_weight is None:
                raise ValueError("Experts are missing weight.")
            # Concatenate gate_proj and up_proj
            weight1 = np.concatenate([gate_weight, up_weight], axis=2)
            weight1 = weight1.reshape(self.num_moe_experts // self.expert_parallel_size
                                      // self.optimizer_parallel_size * self.hidden_size, -1)
            result.append(weight1)
        elif len(weights) == 1:
            weight2 = weights[0]
            weight2 = weight2.reshape(self.num_moe_experts // self.expert_parallel_size
                                      // self.optimizer_parallel_size, -1, self.moe_ffn_hidden_size)
            weight2 = weight2.transpose(0, 2, 1)
            weight2 = weight2.reshape(self.num_moe_experts // self.expert_parallel_size
                                      // self.optimizer_parallel_size * self.moe_ffn_hidden_size, -1)
            result.append(weight2)
        else:
            raise ValueError("The number of weights does not match the expected number of experts.")

        return result

    def _num_local_experts(self) -> int:
        """Number of experts actually stored in this checkpoint shard."""
        if not self.num_moe_experts:
            raise ValueError(
                "ExpertsConvertOp requires set_model_config() (or num_moe_experts, "
                "hidden_size, expert_parallel_size and optimizer_parallel_size) to be "
                "set before MF → HF conversion.")
        return (self.num_moe_experts
                // (self.expert_parallel_size or 1)
                // (self.optimizer_parallel_size or 1))

    def _mf_to_hf(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """Unstack the MF experts weight into one array per HF slot.

        Returns one array per entry of ``hf_names``, each shaped
        ``[num_local_experts, ...]``. Exact inverse of :meth:`_hf_to_mf`.
        Naming is handled by :meth:`mf_to_hf`, because a single MF tensor maps
        to ``len(hf_names) * num_local_experts`` HF tensors.
        """
        num_local = self._num_local_experts()
        weight = weights[0]

        if len(self.hf_names) == 2:
            # weight1: [E * hidden, 2 * ffn] -> gate/up, each [E, ffn, hidden]
            stacked = weight.reshape(num_local, self.hidden_size, -1)
            gate_weight, up_weight = np.split(stacked, 2, axis=2)
            return [np.ascontiguousarray(gate_weight.transpose(0, 2, 1)),
                    np.ascontiguousarray(up_weight.transpose(0, 2, 1))]
        if len(self.hf_names) == 1:
            # weight2: [E * ffn, hidden] -> down, [E, hidden, ffn]
            stacked = weight.reshape(num_local, -1, self.hidden_size)
            return [np.ascontiguousarray(stacked.transpose(0, 2, 1))]
        raise ValueError(
            f"ExpertsConvertOp MF → HF supports 1 or 2 hf_names, got `{self.hf_names}`.")

    def mf_to_hf(self, name_to_weight: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """MF → HF conversion, expanding the stacked weight into per-expert names.

        The base implementation assumes a 1:1 correspondence between names on
        each side. Experts are the one case where a single MF tensor yields
        ``len(hf_names) * num_local_experts`` HF tensors, so the naming is done
        here instead of in :meth:`ConvertOp.mf_to_hf`.
        """
        mf_pattern = self.mf_names[0]
        weight, match = None, None
        if mf_pattern in name_to_weight:
            weight = name_to_weight[mf_pattern]
        else:
            re_pattern = self._name_to_pattern(mf_pattern)
            for name in name_to_weight:
                found = re.findall(re_pattern, name)
                if found:
                    weight, match = name_to_weight[name], found
                    break
        if weight is None:
            raise ValueError(f"Cannot find {mf_pattern} in {list(name_to_weight)}")

        prefix_args = ()
        if match:
            first = match[0]
            prefix_args = first if isinstance(first, tuple) else (first,)

        per_slot = self._mf_to_hf([weight])
        num_local = self._num_local_experts()

        hf_weights = {}
        for slot, hf_pattern in enumerate(self.hf_names):
            for expert_id in range(num_local):
                hf_name = hf_pattern.format(*prefix_args, expert_id)
                hf_weights[hf_name] = per_slot[slot][expert_id]
        return hf_weights


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
        """Unstack the MF weight back into the HF weights (inverse of np.stack)."""
        if len(self.mf_names) != 1:
            raise ValueError(
                f"StackConvertOp MF → HF only supports 'N hf_name to 1 mf_name', "
                f"but got mf_names: `{self.mf_names}`.")
        weight = weights[0]
        num_parts = len(self.hf_names)
        if weight.shape[self.dim] != num_parts:
            raise ValueError(
                f"StackConvertOp: dim {self.dim} of the MF weight "
                f"({weight.shape[self.dim]}) must equal the number of hf_names ({num_parts}).")
        return [np.ascontiguousarray(np.take(weight, k, axis=self.dim)) for k in range(num_parts)]


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
    optimizer_parallel_size: int = None

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
        self.optimizer_parallel_size = get_auto_parallel_context('optimizer_weight_shard_size')

    def _hf_to_mf(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """Convert Q, K, V weights to QKV fused weight"""
        if len(weights) < 3:
            raise ValueError(f"Expected at least 3 weights for QKV conversion, but got {len(weights)}")
        q_weight, k_weight, v_weight = weights
        nh = self.num_attention_heads // self.tensor_model_parallel_size // self.optimizer_parallel_size
        ng = self.num_query_groups // self.tensor_model_parallel_size // self.optimizer_parallel_size
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
        """Split the fused GQA QKV weight into independent Q, K, V weights."""
        qkv_weight = weights[0]
        nh = self.num_attention_heads // self.tensor_model_parallel_size // self.optimizer_parallel_size
        ng = self.num_query_groups // self.tensor_model_parallel_size // self.optimizer_parallel_size
        dim = self.kv_channels

        if nh % ng != 0:
            raise ValueError(
                f"Number of attention heads per group ({nh}) must be divisible by "
                f"number of query groups ({ng})")

        q_per_group = dim * nh // ng
        grouped = qkv_weight.reshape((ng, q_per_group + 2 * dim, -1))
        q_weight = grouped[:, :q_per_group, :].reshape((-1, self.hidden_size))
        k_weight = grouped[:, q_per_group:q_per_group + dim, :].reshape((-1, self.hidden_size))
        v_weight = grouped[:, q_per_group + dim:, :].reshape((-1, self.hidden_size))
        return [np.ascontiguousarray(q_weight),
                np.ascontiguousarray(k_weight),
                np.ascontiguousarray(v_weight)]


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
    optimizer_parallel_size: int = None

    def set_model_config(self, config):
        """Set parameters from model configuration"""
        self.num_attention_heads = config.num_attention_heads
        self.num_query_groups = config.num_query_groups
        self.kv_channels = config.kv_channels
        self.tensor_model_parallel_size = config.tensor_model_parallel_size
        self.optimizer_parallel_size = get_auto_parallel_context('optimizer_weight_shard_size')

    def _hf_to_mf(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """Convert Q, K, V bias to QKV fused bias"""
        if len(weights) < 3:
            raise ValueError(f"Expected at least 3 bias weights for QKV bias conversion, but got {len(weights)}")
        q_bias, k_bias, v_bias = weights
        nh = self.num_attention_heads // self.tensor_model_parallel_size // self.optimizer_parallel_size
        ng = self.num_query_groups // self.tensor_model_parallel_size // self.optimizer_parallel_size
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
        """Split the fused QKV bias into independent Q, K, V biases.

        Exact inverse of :meth:`_hf_to_mf`, which lays the bias out as
        ``(ng, 3 * kv_channels)``.
        """
        qkv_bias = weights[0]
        ng = self.num_query_groups // self.tensor_model_parallel_size // self.optimizer_parallel_size
        dim = self.kv_channels
        grouped = qkv_bias.reshape((ng, 3 * dim))
        return [np.ascontiguousarray(grouped[:, :dim].reshape(-1)),
                np.ascontiguousarray(grouped[:, dim:2 * dim].reshape(-1)),
                np.ascontiguousarray(grouped[:, 2 * dim:].reshape(-1))]


@dataclass
class ScaleSplitConvertOp(ConvertOp):
    """
    Scale-split operation (1:N mapping).

    Splits a single HF tensor along dim=0 into N separate MF tensors.

    This is used for HyperConnection ``hc_scale`` parameters in DeepSeek-V4,
    where the HF checkpoint stores a combined tensor (e.g. shape ``[3]``)
    that needs to be split into separate ``alpha_pre``, ``alpha_post`` and
    ``alpha_res`` scalars (each shape ``[1]``).

    Example::

        hf_names  = ["layers.{}.hc_attn_scale"]
        mf_names  = ["decoder.layers.{}.attn_hc.alpha_pre",
                     "decoder.layers.{}.attn_hc.alpha_post",
                     "decoder.layers.{}.attn_hc.alpha_res"]

        HF shape [3]  →  MF shapes [1], [1], [1]
    """

    def __post_init__(self):
        super().__post_init__()
        if len(self.hf_names) != 1:
            raise ValueError(
                f"ScaleSplitConvertOp requires exactly 1 hf_name, "
                f"but got {len(self.hf_names)}: `{self.hf_names}`."
            )
        if len(self.mf_names) < 2:
            raise ValueError(
                f"ScaleSplitConvertOp requires at least 2 mf_names, "
                f"but got {len(self.mf_names)}: `{self.mf_names}`."
            )

    def _hf_to_mf(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """Split single HF weight into N MF weights along dim=0."""
        weight = weights[0]
        num_splits = len(self.mf_names)
        if weight.shape[0] != num_splits:
            raise ValueError(
                f"ScaleSplitConvertOp: hf weight first dimension ({weight.shape[0]}) "
                f"must match the number of mf_names ({num_splits})."
            )
        splits = np.split(weight, num_splits, axis=0)
        # np.split on [3] with 3 splits returns arrays of shape [1] each
        return list(splits)

    def _mf_to_hf(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """Concatenate N MF weights back to single HF weight along dim=0."""
        return [np.concatenate(weights, axis=0)]
