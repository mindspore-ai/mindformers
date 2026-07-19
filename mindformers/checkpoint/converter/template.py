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
"""Converter Template of Load Hugging Face checkpoint."""

import re
import functools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

import numpy as np
from mindspore import Parameter, Tensor, get_auto_parallel_context
from mindspore.common import dtype as mstype

from mindformers.checkpoint.sharded_tensor import ShardedTensor
from mindformers.tools.logger import logger
from mindformers.checkpoint.converter.convert_op import ConvertOp, ExpertsConvertOp, QKVConvertOp, QKVBiasConvertOp
from mindformers.parallel_core.transformer_config_utils import convert_to_transformer_config
# Mapping from MindSpore dtypes to the corresponding "canonical" numpy dtype.
# When a numpy weight is already in this dtype, we skip the explicit Tensor(…)
# cast and let downstream Parameter(…) wrapping create the Tensor directly.
_MS_DTYPE_TO_NP: Dict = {
    mstype.float32: np.float32,
    mstype.float16: np.float16,
    mstype.bfloat16: np.float32,   # bf16 np has no native dtype; stored as uint16 → skip is unsafe
    mstype.int32: np.int32,
    mstype.int16: np.int16,
    mstype.int8: np.int8,
    mstype.uint8: np.uint8,
}


def _is_np_dtype_compatible(np_dtype: np.dtype, ms_dtype) -> bool:
    """Return True when *np_dtype* is already bit-compatible with *ms_dtype*."""
    canonical = _MS_DTYPE_TO_NP.get(ms_dtype)
    return canonical is not None and np_dtype == canonical


@dataclass
class WeightTemplate:
    """Weight conversion template, supports bidirectional conversion HF ↔ MF"""
    weight_converters: List[ConvertOp] = field(default_factory=list)

    # Built by __post_init__
    hf_name_to_converter: Dict[str, ConvertOp] = field(default_factory=dict, init=False)
    mf_name_to_converter: Dict[str, ConvertOp] = field(default_factory=dict, init=False)
    name_to_weight: Dict[str, np.ndarray] = field(default_factory=dict, init=False)
    network_tensor_info: Dict[str, ShardedTensor] = field(default_factory=dict, init=False)

    # Pre-compiled regex patterns and sorted lookup lists (built in __post_init__)
    _compiled_hf_patterns: Dict[str, 're.Pattern'] = field(default_factory=dict, init=False)
    _compiled_mf_patterns: Dict[str, 're.Pattern'] = field(default_factory=dict, init=False)
    _sorted_hf_patterns: List[tuple] = field(default_factory=list, init=False)
    _sorted_mf_patterns: List[tuple] = field(default_factory=list, init=False)
    _expert_converters: List[ConvertOp] = field(default_factory=list, init=False)
    _qkv_converters: List[ConvertOp] = field(default_factory=list, init=False)

    def __post_init__(self):
        """Build hf_name_to_converter and mf_name_to_converter mapping dictionaries"""
        # Build mapping dictionaries directly from ConvertOp names
        for converter in self.weight_converters:
            for hf_name in converter.hf_names:
                self.hf_name_to_converter[hf_name] = converter
            for mf_name in converter.mf_names:
                self.mf_name_to_converter[mf_name] = converter

        # Pre-compile all regex patterns for HF names (used in preprocess_hf_weights, get_mf_name, etc.)
        for converter in self.weight_converters:
            for hf_name in converter.hf_names:
                if "{}" in hf_name:
                    re_pattern = hf_name.replace(".", r"\.").replace("{}", r"(\d+)")
                    self._compiled_hf_patterns[hf_name] = re.compile(re_pattern)

        # Pre-compile regex patterns for MF names
        for converter in self.weight_converters:
            for mf_name in converter.mf_names:
                if "{}" in mf_name:
                    re_pattern = mf_name.replace(".", r"\.").replace("{}", r"(\d+)")
                    self._compiled_mf_patterns[mf_name] = re.compile(re_pattern)

        # Pre-sort patterns by length (longest first) for get_convert_op — done once, not per call.
        # Store (compiled_regex, pattern, converter) tuples.
        self._sorted_hf_patterns = sorted(
            [(re.compile(p.replace(".", r"\.").replace("{}", "(.*)")), p, c)
             for p, c in self.hf_name_to_converter.items()],
            key=lambda x: len(x[1]), reverse=True
        )
        self._sorted_mf_patterns = sorted(
            [(re.compile(p.replace(".", r"\.").replace("{}", "(.*)")), p, c)
             for p, c in self.mf_name_to_converter.items()],
            key=lambda x: len(x[1]), reverse=True
        )

        # Separate expert and QKV converters for fast direct lookup
        self._expert_converters = [c for c in self.weight_converters if isinstance(c, ExpertsConvertOp)]
        self._qkv_converters = [c for c in self.weight_converters if isinstance(c, QKVConvertOp)]

        self.release()

    def release(self):
        """Release cached weights"""
        if len(self.name_to_weight) > 0:
            weights_not_converted = [
                (name, weight.size())
                for name, weight in self.name_to_weight.items()
            ]
            logger.warning(f"weights not converted {len(weights_not_converted)} {weights_not_converted}")
        self.name_to_weight.clear()

    def set_model_config(self, config):
        """Set model configuration to required ConvertOp"""
        self.mf_config = config
        for converter in self.weight_converters:
            converter.mf_config = config
            if hasattr(converter, 'set_model_config'):
                converter.set_model_config(config)

    def _cast_weight_to_dtype(self, mf_name: str, weight: np.ndarray):
        """
        Cast weight to the target dtype specified in the model config.

        When the numpy ``weight`` is already in the target dtype (e.g. float32 →
        float32), the explicit ``Tensor(…)`` call is skipped so that downstream
        ``Parameter(…)`` wrapping can create the Tensor directly without an
        intermediate copy.

        Args:
            mf_name: MindFormers parameter name
            weight: numpy array weight

        Returns:
            numpy array or Tensor cast to target dtype.
        """
        config = getattr(self, 'mf_config', None)
        if config is None:
            return weight

        # For embedding weights, prefer embedding_params_dtype
        is_embedding = 'embedding' in mf_name.lower()
        if is_embedding:
            target_dtype = getattr(config, 'embedding_params_dtype', None)
            if target_dtype is not None:
                if _is_np_dtype_compatible(weight.dtype, target_dtype):
                    return weight
                return Tensor(weight, dtype=target_dtype)

        # tid2eid is an integer lookup table (token-id → expert-id). It is stored as
        # BFloat16 in HuggingFace checkpoints but the model expects Int32. Cast to
        # int32 explicitly; otherwise load_param_into_net will raise a type mismatch
        # error (Int32 in net vs BFloat16 in parameter_dict).
        if 'tid2eid' in mf_name:
            return Tensor(weight.astype(np.int32), dtype=mstype.int32)

        target_dtype = getattr(config, 'params_dtype', None)
        if target_dtype is not None:
            if _is_np_dtype_compatible(weight.dtype, target_dtype):
                return weight
            return Tensor(weight, dtype=target_dtype)

        return weight

    def get_mf_name(self, hf_name: str) -> str:
        """Get MF parameter name corresponding to HF parameter name"""
        converter = self.get_convert_op(hf_name, self.hf_name_to_converter)
        if converter is None:
            return hf_name

        mf_names = converter.mf_names

        # Use pre-compiled patterns to extract layer index
        for hf_pattern in converter.hf_names:
            compiled_re = self._compiled_hf_patterns.get(hf_pattern)
            if compiled_re is not None:
                match = compiled_re.match(hf_name)
                if match:
                    layer_index = match.group(1)
                    return [name.replace("{}", layer_index) for name in mf_names]

        return mf_names

    def get_hf_names_for_mf(self, mf_name: str) -> List[str]:
        """Get all HF parameter names required to generate a MF parameter"""
        raise ValueError("Conversion from MindSpore Transformer weights to Hugging Face is currently not supported.")

    def get_mf_state_dict(
            self,
            hf_state_dict: Dict[str, np.ndarray]
    ) -> Dict[str, Parameter]:
        """Convert HF parameter dictionary from Reshard output to MF parameter dictionary"""
        mf_state_dict: Dict[str, Parameter] = {}

        for hf_name, weight in hf_state_dict.items():
            result = self.add_hf_weight(hf_name, weight)
            if result is not None:
                for mf_name, mf_weight in result.items():
                    if not isinstance(mf_weight, Parameter):
                        mf_weight = self._cast_weight_to_dtype(mf_name, mf_weight)
                        mf_weight = Parameter(mf_weight, name=mf_name)
                    mf_state_dict[mf_name] = mf_weight
            elif self.get_convert_op(hf_name, self.mf_name_to_converter) is not None:
                # The name is already an MF name (e.g. preprocessed by preprocess_hf_weights
                # for MoE experts / QKV). Add it directly without going through a converter.
                if not isinstance(weight, Parameter):
                    weight = self._cast_weight_to_dtype(hf_name, weight)
                    weight = Parameter(weight, name=hf_name)
                mf_state_dict[hf_name] = weight
            else:
                # Only warn if weight is not cached (invalid/unregistered weight)
                # If weight is cached, it means we found a converter but weights are incomplete
                # (waiting for more weights, e.g., QKV needs 3 weights, Concat needs 2 weights)
                # This is normal and will be converted when all required weights are available
                if hf_name not in self.name_to_weight:
                    logger.warning(f"hf_name: {hf_name} added but not converted (no matching converter found)")

        self.release()
        return mf_state_dict

    def _extract_layer_key(self, weight_name: str, pattern: str) -> Optional[str]:
        """
        Extract layer key from weight name based on pattern.
        Uses pre-compiled patterns when available.
        """
        if "{}" not in pattern:
            return None

        # Try pre-compiled patterns first (both HF and MF namespaces)
        compiled_re = self._compiled_hf_patterns.get(pattern) or self._compiled_mf_patterns.get(pattern)
        if compiled_re is not None:
            match = compiled_re.match(weight_name)
        else:
            re_pattern = pattern.replace(".", r"\.").replace("{}", r"(\d+)")
            match = re.match(re_pattern, weight_name)

        if match:
            layer_index = match.group(1)
            prefix_end = pattern.find("{}")
            prefix = pattern[:prefix_end]
            return prefix + layer_index
        return None

    def _extract_layer_key_from_patterns(self, weight_name: str, patterns: List[str]) -> Optional[str]:
        """
        Extract layer key from weight name by trying to match against patterns.
        
        Args:
            weight_name: Full weight name
            patterns: List of patterns to try matching
        
        Returns:
            Layer key string or None if no pattern matches
        """
        for pattern in patterns:
            if "{}" in pattern:
                layer_key = self._extract_layer_key(weight_name, pattern)
                if layer_key:
                    return layer_key
        return None

    def _should_collect_weight(
            self,
            cached_name: str,
            op: ConvertOp,
            layer_key: Optional[str],
            is_mf_name: bool
    ) -> bool:
        """
        Check if a cached weight should be collected for conversion.
        
        Args:
            cached_name: Name of the cached weight
            op: ConvertOp that requires this weight
            layer_key: Layer key to match (None for non-layer weights)
            is_mf_name: Whether checking MF names (True) or HF names (False)
        
        Returns:
            True if weight should be collected, False otherwise
        """
        if not op.is_required_name(cached_name, mf_name=is_mf_name):
            return False

        # If this is a layer weight, ensure it's from the same layer
        if layer_key is not None:
            patterns = op.mf_names if is_mf_name else op.hf_names
            cached_layer_key = self._extract_layer_key_from_patterns(cached_name, patterns)
            if cached_layer_key != layer_key:
                return False

        return True

    def add_hf_weight(
            self,
            hf_name: str,
            weight: np.ndarray
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Add a single HF weight and attempt conversion.
        
        Returns:
            Dict[str, np.ndarray]: Conversion result if successful
            None: If weight is invalid/unregistered OR if weights are incomplete (waiting for more weights)
        """
        if "qkv" in hf_name:
            return {hf_name: weight}
        # Temporarily store weight using full name
        self.name_to_weight[hf_name] = weight

        # Get converter corresponding to this weight
        op = self.get_convert_op(hf_name, self.hf_name_to_converter)
        if op is None:
            # No converter found, this is an invalid/unregistered weight
            self.name_to_weight.pop(hf_name, None)
            return None
        if (get_auto_parallel_context('enable_parallel_optimizer') and
                isinstance(op, (ExpertsConvertOp, QKVConvertOp, QKVBiasConvertOp))):
            parallel_size = get_auto_parallel_context('optimizer_weight_shard_size')
            if isinstance(op, ExpertsConvertOp):
                parallel_size = op.expert_parallel_size
            if isinstance(op, (QKVConvertOp, QKVBiasConvertOp)):
                parallel_size = op.tensor_model_parallel_size
            # Get real optimizer parallel size for experts and qkv
            real_optimizer_parallel_size = (
                        self.network_tensor_info[op.mf_names[0].replace('{}', '0')].axis_fragmentations[0]
                        // parallel_size)
            op.optimizer_parallel_size = real_optimizer_parallel_size
        else:
            op.optimizer_parallel_size = 1

        # Extract layer key from current weight name (if it's a layer weight)
        layer_key = self._extract_layer_key_from_patterns(hf_name, op.hf_names)

        # Collect all weights required by this converter from the same layer
        name_to_weight = {}
        for cached_name in list(self.name_to_weight.keys()):
            if self._should_collect_weight(cached_name, op, layer_key, is_mf_name=False):
                name_to_weight[cached_name] = self.name_to_weight.pop(cached_name)

        # Execute conversion
        convert_res = op(name_to_weight)
        if convert_res is None:
            # Weights incomplete, restore cached weights and return None
            # This is normal for multi-weight converters (e.g., QKV, Concat)
            self.name_to_weight.update(name_to_weight)
            return None

        # Conversion successful, ConvertOp already replaced wildcards in names
        return convert_res

    def get_convert_op(self, name, pattern_to_convert_ops: Dict[str, ConvertOp]) -> Optional[ConvertOp]:
        """Find ConvertOp that matches the given name"""
        # Fast path: exact match
        if name in pattern_to_convert_ops:
            return pattern_to_convert_ops[name]

        # Use pre-sorted, pre-compiled patterns when available
        if pattern_to_convert_ops is self.hf_name_to_converter:
            sorted_patterns = self._sorted_hf_patterns
        elif pattern_to_convert_ops is self.mf_name_to_converter:
            sorted_patterns = self._sorted_mf_patterns
        else:
            # Fallback for any other dict (should not happen in practice)
            sorted_patterns = sorted(
                [(re.compile(p.replace(".", r"\.").replace("{}", "(.*)")), p, c)
                 for p, c in pattern_to_convert_ops.items()],
                key=lambda x: len(x[1]), reverse=True
            )

        for compiled_re, _, converter in sorted_patterns:
            if compiled_re.match(name):
                return converter
        return None

    def get_hf_state_dict(
            self,
            mf_state_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Convert MindFormers parameter dictionary to HuggingFace format"""
        hf_state_dict: Dict[str, np.ndarray] = {}

        for mf_name, weight in mf_state_dict.items():
            # Get conversion result corresponding to this weight
            result = self.add_mf_weight(mf_name, weight)
            if result is not None:
                hf_state_dict.update(result)

        self.release()
        return hf_state_dict

    def add_mf_weight(
            self,
            name: str,
            weight: np.ndarray
    ) -> Optional[Dict[str, np.ndarray]]:
        """Add a single MF weight and convert to HF format"""
        # Temporarily store weight using full name
        self.name_to_weight[name] = weight

        # Get converter corresponding to this weight
        op = self.get_convert_op(name, self.mf_name_to_converter)
        if op is None:
            # No converter found, this is an invalid/unregistered weight
            self.name_to_weight.pop(name, None)
            return None

        # Extract layer key from current weight name (if it's a layer weight)
        layer_key = self._extract_layer_key_from_patterns(name, op.mf_names)

        # Collect all weights required by this converter from the same layer
        name_to_weight = {}
        for cached_name in list(self.name_to_weight.keys()):
            if self._should_collect_weight(cached_name, op, layer_key, is_mf_name=True):
                name_to_weight[cached_name] = self.name_to_weight.pop(cached_name)

        # Execute conversion
        convert_res = op(name_to_weight, mf_to_hf=True)

        if convert_res is None:
            # Weights incomplete, restore cached weights and return None
            self.name_to_weight.update(name_to_weight)
            return convert_res

        # Conversion successful, ConvertOp already replaced wildcards in names
        return convert_res

    def check_weights_for_experts(self, weight_name: str):
        """
        Check if the given weight name belongs to MoE routed experts.

        This method checks whether a parameter corresponds to Mixture of Experts (MoE)
        expert weights by searching for the "experts" keyword in the weight name,
        excluding shared experts which are handled by regular ConvertOps.

        Args:
            weight_name (str): The name of the weight parameter to check

        Returns:
            bool: True if the weight is an MoE routed expert weight, False otherwise
        """
        # Only routed experts (e.g. "ffn.experts.0.w1") need preprocess stacking.
        # Shared experts (e.g. "ffn.shared_experts.w1") use ConcatConvertOp.
        if "shared_experts" in weight_name:
            return False
        if "experts" in weight_name:
            return True
        return False

    def get_expert_id(self, hf_name: str) -> Optional[int]:
        """Extract the expert index from an HF expert parameter name.

        Uses pre-compiled patterns from registered ExpertsConvertOp entries.
        For patterns with two placeholders (e.g. ``layers.{}.ffn.experts.{}.w1.weight``),
        the last captured group is the expert id.  For single-placeholder patterns
        (e.g. ``mtp.{}.ffn.experts.{}.w2.weight``), the second group is the expert id.

        Args:
            hf_name: HuggingFace parameter name, e.g.
                     ``layers.0.ffn.experts.42.w1.weight``

        Returns:
            Expert index as integer, or ``None`` if no expert converter matches.
        """
        for converter in self._expert_converters:
            for hf_pattern in converter.hf_names:
                compiled_re = self._compiled_hf_patterns.get(hf_pattern)
                if compiled_re is None:
                    continue
                match = compiled_re.findall(hf_name)
                if not match:
                    continue
                # Pattern with 2 placeholders → tuple of (layer_id, expert_id)
                if isinstance(match[0], tuple):
                    return int(match[0][-1])
                # Pattern with 1 placeholder → expert_id is the only group
                return int(match[0])
        return None

    def _find_expert_converter_for_hf(self, hf_name: str) -> Optional[ConvertOp]:
        """Return the ExpertsConvertOp whose hf_pattern matches *hf_name*, or None."""
        for converter in self._expert_converters:
            for hf_pattern in converter.hf_names:
                compiled_re = self._compiled_hf_patterns.get(hf_pattern)
                if compiled_re is not None and compiled_re.match(hf_name):
                    return converter
        return None

    def get_expert_source_idx(self, hf_name: str) -> Optional[int]:
        """Return the index of *hf_name* in the multi-source ExpertsConvertOp hf_names list.

        For single-source converters the result is always 0.
        Returns ``None`` when *hf_name* does not belong to any expert converter.
        """
        converter = self._find_expert_converter_for_hf(hf_name)
        if converter is None:
            return None
        for idx, pattern in enumerate(converter.hf_names):
            compiled_re = self._compiled_hf_patterns.get(pattern)
            if compiled_re is not None and compiled_re.match(hf_name):
                return idx
        return None

    def is_multi_source_expert(self, dst_name: str) -> bool:
        """Return True when the destination expert parameter combines multiple HF sources."""
        converter = self._find_expert_converter(dst_name)
        if converter is None:
            # dst_name may be the HF-style name — try dst_metas-based fallback
            return False
        return len(converter.hf_names) > 1

    def check_weights_for_qkv(self, weight_name: str):
        """
        Check if the given weight name belongs to QKV projection weights.

        This method determines whether a parameter is a QKV weight by checking two conditions:
        1. Whether QKV weights can be divided across tensor parallel and optimizer parallel dimensions
        2. Whether the weight name contains Q/K/V projection identifiers

        Args:
            weight_name (str): The name of the weight parameter to check

        Returns:
            bool: True if the weight is a QKV projection weight that needs special handling, False otherwise
        """
        for converter in self.weight_converters:
            if (isinstance(converter, QKVConvertOp) and converter.optimizer_parallel_size != -1 and
                    converter.num_query_groups >=
                    converter.tensor_model_parallel_size * converter.optimizer_parallel_size):
                return False
        if any(proj in weight_name for proj in ("q_proj", "k_proj", "v_proj")):
            return True
        return False

    def stack_hf_experts_weight(
            self,
            dst_name: str,
            num_moe_experts: int,
            src_tensor: ShardedTensor
    ) -> ShardedTensor:
        """
        Stack expert weights for HuggingFace MoE (Mixture of Experts) model.

        This method checks if the destination parameter name corresponds to MoE expert weights,
        and if so, reshapes the source tensor by stacking all experts along the first dimension.
        The global shape is expanded by multiplying with the number of experts.

        For multi-source converters (e.g., w1 + w3 → weight1), the global shape is set to
        the 3D merged layout: (num_experts, hidden_size, num_sources * moe_ffn_hidden_size).

        Args:
            dst_name (str): Destination parameter name to check if it's an expert weight
            num_moe_experts (int): Total number of MoE experts to stack
            src_tensor (ShardedTensor): Source sharded tensor containing expert weights

        Returns:
            ShardedTensor: Modified tensor with updated shape information if it's an expert weight,
                          otherwise returns the original tensor unchanged
        """
        if self.check_weights_for_experts(dst_name):
            converter = self._find_expert_converter(dst_name)
            # Use model-config dimensions for the post-merge shape calculation
            # instead of the stored (file) shape.  This is needed because
            # quantized checkpoints (e.g. I8 → packed INT4) may store weights
            # with a smaller inner dimension than the actual model dimension.
            hidden_size = getattr(self.mf_config, 'hidden_size', None)
            moe_ffn_hidden_size = getattr(self.mf_config, 'moe_ffn_hidden_size', None)
            if hidden_size is None:
                hidden_size = src_tensor.global_shape[1]
            if moe_ffn_hidden_size is None:
                moe_ffn_hidden_size = src_tensor.global_shape[0]

            if converter is not None and len(converter.hf_names) > 1:
                # Multi-source expert weight (e.g., w1 + w3 → weight1).
                # The preprocess_hf_weights step will merge them into a 3D tensor:
                # (num_experts, hidden_size, num_sources * moe_ffn_hidden_size).
                num_sources = len(converter.hf_names)
                param_shape = (num_moe_experts,
                               hidden_size,
                               num_sources * moe_ffn_hidden_size)
            else:
                # Single-source: 3D shape (num_experts, moe_ffn_hidden, hidden).
                # For w2 (down_proj): shape after stacking is
                #   (num_experts, moe_ffn_hidden_size, hidden_size).
                param_shape = (num_moe_experts, moe_ffn_hidden_size, hidden_size)
            src_tensor.global_shape = param_shape
            src_tensor.local_shape = param_shape
            src_tensor.axis_fragmentations = [1] * len(param_shape)
        return src_tensor

    def _find_expert_converter(self, dst_name: str) -> Optional[ConvertOp]:
        """Find the ExpertsConvertOp that matches the given destination parameter name."""
        for converter in self._expert_converters:
            for mf_name in converter.mf_names:
                compiled_re = self._compiled_mf_patterns.get(mf_name)
                if compiled_re is not None and compiled_re.match(dst_name):
                    return converter
        return None

    def get_num_moe_experts(self, dst_name, src_names_num):
        """Get the number of moe experts"""
        for converter in self._expert_converters:
            for mf_name in converter.mf_names:
                compiled_re = self._compiled_mf_patterns.get(mf_name)
                if compiled_re is not None and compiled_re.search(dst_name):
                    return src_names_num // len(converter.hf_names)
        if self._expert_converters:
            raise ValueError(f"Can not find the num_moe_experts for dst_name: {dst_name}.")
        return None


    def _handle_expert_param_match(self, parameter, experts_weights,
                                     experts_per_card, expert_offset,
                                     hf_names, mf_names, hf_name, match):
        """Process a matched expert parameter: collect, stack, and merge when ready.

        Returns:
            tuple: (target_name, processed_tensor) or (None, None) if incomplete
        """
        num_placeholders = hf_name.count("{}")
        if num_placeholders == 2:
            layer_id = int(match[0][0])
            expert_id = int(match[0][1])
            stacked_key = hf_name.format(layer_id, "stack")
        elif num_placeholders == 1:
            layer_id = None
            expert_id = int(match[0][0])
            stacked_key = hf_name.format("stack")
        else:
            raise ValueError(
                f"Expected 1 or 2 '{{}}' placeholders in hf_name, "
                f"got {num_placeholders}: '{hf_name}'"
            )

        local_expert_id = expert_id - expert_offset

        # Counter-based tracking: O(1) per insertion instead of O(n) any() scan
        if stacked_key not in experts_weights:
            experts_weights[stacked_key] = {
                "weights": [None] * experts_per_card,
                "count": 0,
                "expert_offset": expert_offset,
            }
        info = experts_weights[stacked_key]
        if 0 <= local_expert_id < experts_per_card and info["weights"][local_expert_id] is None:
            info["weights"][local_expert_id] = parameter
            info["count"] += 1

        if info["count"] < experts_per_card:
            return None, None

        # For multi-source converters (e.g., w1 + w3 → weight1), check siblings
        if len(hf_names) > 1:
            all_ready = True
            all_stacked = []
            for hf_name_check in hf_names:
                if num_placeholders == 2:
                    key = hf_name_check.format(layer_id, "stack")
                else:
                    key = hf_name_check.format("stack")
                if key not in experts_weights:
                    all_ready = False
                    break
                info_check = experts_weights[key]
                if info_check["count"] < experts_per_card:
                    all_ready = False
                    break
                stacked = np.stack(info_check["weights"], axis=0)
                # The HF checkpoint may store expert weights in transposed layout
                # (moe_ffn_hidden, hidden) instead of (hidden, moe_ffn_hidden).
                # Transpose so that stacking produces (num_experts, hidden, moe_ffn_hidden).
                moe_ffn_hidden_size = getattr(self.mf_config, 'moe_ffn_hidden_size', None)
                if moe_ffn_hidden_size is not None and stacked.shape[1] == moe_ffn_hidden_size:
                    stacked = stacked.transpose(0, 2, 1)
                stacked = stacked.reshape(experts_per_card * stacked.shape[1], -1)
                all_stacked.append(stacked)

            if not all_ready:
                return None, None

            # Merge all sources into 3D: (experts_per_card, hidden_size, total_ffn_hidden_size)
            merged_parts = []
            for stacked in all_stacked:
                leading_dim = stacked.shape[0] // experts_per_card
                reshaped = stacked.reshape(experts_per_card, leading_dim, -1)
                merged_parts.append(reshaped)
            merged = np.concatenate(merged_parts, axis=2)

            # Free individual expert weight tensors — the merged tensor
            # already owns the data. Each source contributes 256 × tensor
            # elements, so this roughly halves peak host memory for experts.
            for hf_name_check in hf_names:
                if num_placeholders == 2:
                    key = hf_name_check.format(layer_id, "stack")
                else:
                    key = hf_name_check.format("stack")
                experts_weights.pop(key, None)

            return mf_names[0].format(layer_id), merged

        # Single-source: stack and reshape to 3D, keeping original
        # (experts_per_card, moe_ffn_hidden, hidden) layout for w2.
        stack_parameter = np.stack(info["weights"], axis=0)
        # The HF checkpoint may store expert weights in transposed layout
        # (hidden, moe_ffn_hidden) instead of (moe_ffn_hidden, hidden).
        # Transpose so that stacking produces (num_experts, moe_ffn_hidden, hidden).
        hidden_size = getattr(self.mf_config, 'hidden_size', None)
        if hidden_size is not None and stack_parameter.shape[1] == hidden_size:
            stack_parameter = stack_parameter.transpose(0, 2, 1)
        experts_weights.pop(stacked_key, None)
        stack_parameter = stack_parameter.reshape(experts_per_card, stack_parameter.shape[1], -1)
        return mf_names[0].format(layer_id), stack_parameter

    def preprocess_hf_weights(self,
                                                     src_name,
                                                     parameter,
                                                     experts_weights,
                                                     qkv_weights,
                                                     num_moe_experts,
                                                     expert_offset=0,
                                                     experts_per_card=None):
        """
        Get stacked tensor and rename source parameter names for MoE experts and QKV.

        Uses pre-compiled regex patterns and direct converter lookup
        (via _expert_converters / _qkv_converters) to avoid iterating through all converters.
        Counter-based tracking replaces O(n) any() scanning with O(1) per insertion.

        Args:
            src_name (str): Source parameter name
            parameter (np.ndarray): Parameter tensor to be processed
            experts_weights (dict): {stacked_key: {"weights": [...], "count": N}}
            qkv_weights (dict): {mf_name: {src_name: weight, ...}}
            num_moe_experts (int): Total number of MoE experts
            expert_offset (int): First expert index assigned to this card (0 when no filtering).
            experts_per_card (int): Number of experts assigned to this card
                (defaults to *num_moe_experts* when ``None``).

        Returns:
            tuple: (target parameter name, processed tensor) or (None, None) if incomplete
        """
        if experts_per_card is None:
            experts_per_card = num_moe_experts

        # ── Expert weights: use pre-collected expert converters only ──
        for converter in self._expert_converters:
            hf_names = converter.hf_names
            mf_names = converter.mf_names
            for hf_name in hf_names:
                compiled_re = self._compiled_hf_patterns.get(hf_name)
                if compiled_re is None:
                    continue
                match = compiled_re.findall(src_name)
                if not match:
                    continue
                return self._handle_expert_param_match(
                    parameter, experts_weights,
                    experts_per_card, expert_offset,
                    hf_names, mf_names, hf_name, match
                )

        # ── QKV weights: use pre-collected QKV converters only ──
        for converter in self._qkv_converters:
            hf_names = converter.hf_names
            mf_names = converter.mf_names
            for hf_name in hf_names:
                compiled_re = self._compiled_hf_patterns.get(hf_name)
                if compiled_re is None:
                    continue
                match = compiled_re.findall(src_name)
                if not match:
                    continue
                layer_id = int(match[0])
                src_name_key = hf_name.format(layer_id)
                mf_name = mf_names[0].format(layer_id)
                if mf_name not in qkv_weights:
                    qkv_weights[mf_name] = {}
                qkv_weights[mf_name][src_name_key] = parameter
                if len(qkv_weights[mf_name]) < 3:
                    return None, None
                q_param = qkv_weights[mf_name][hf_names[0].format(layer_id)]
                k_param = qkv_weights[mf_name][hf_names[1].format(layer_id)]
                v_param = qkv_weights[mf_name][hf_names[2].format(layer_id)]
                qkv_param = np.concatenate([
                    q_param.reshape((converter.num_query_groups,
                                     converter.kv_channels * converter.num_attention_heads
                                     // converter.num_query_groups, -1)),
                    k_param.reshape((converter.num_query_groups, converter.kv_channels, -1)),
                    v_param.reshape((converter.num_query_groups, converter.kv_channels, -1)),
                ], axis=1).reshape((-1, converter.hidden_size))
                return mf_name, qkv_param

        raise ValueError(f"Can not find the weight name {src_name}.")


def register_hf_weight_template(func: Callable = None):
    """
    Decorator: Decorates the `__init__` method of model classes,
        automatically creates and binds WeightTemplate to network instance.

    After decoration, after __init__ execution completes, it will automatically:
    1. Create WeightTemplate based on class attributes weight_converters
    2. Call template.set_model_config(config) to set model configuration
    3. Bind template instance to self.template attribute

    Requires model class or its parent class to define the following class attributes:
    - weight_converters: List[ConvertOp]  # Required
      Each ConvertOp's hf_names and mf_names should contain full paths with wildcards,
      e.g., "model.layers.{}.self_attn.q_proj.weight" -> "decoder.layers.{}.self_attention.linear_qkv.weight"

    Args:
        func: The decorated __init__ method
    """

    def decorator(init_func: Callable):
        @functools.wraps(init_func)
        def wrapper(self, *args, **kwargs):
            # Execute original __init__ method
            result = init_func(self, *args, **kwargs)

            # Get model configuration (assuming first argument is config)
            config = args[0] if args else kwargs.get('config')

            # Get class attributes
            cls = self.__class__

            # Get required class attributes
            if not hasattr(cls, 'weight_converters'):
                raise AttributeError(f"Class {cls.__name__} must define 'weight_converters' attribute")

            weight_converters = getattr(cls, 'weight_converters', [])

            # Create WeightTemplate
            template = WeightTemplate(
                weight_converters=weight_converters
            )

            # Set model configuration
            if config is not None:
                # Prefer the model-specific converter (e.g. DeepseekV4ConfigConverter)
                # when __init__ has already populated self.transformer_config.
                tf_config = getattr(self, 'transformer_config', None)
                if tf_config is None:
                    tf_config = convert_to_transformer_config(config)
                template.set_model_config(tf_config)

            # Bind template instance to self.template
            self.template = template

            return result

        return wrapper

    # If called as `@register_template` (no parentheses), then func is the decorated function
    if func is None:
        return decorator
    # If called as `@register_template()` (with parentheses), then func is the decorated function
    return decorator(func)
