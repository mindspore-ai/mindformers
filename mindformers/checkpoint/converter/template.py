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
from mindspore import Parameter, get_auto_parallel_context

from mindformers.checkpoint.sharded_tensor import ShardedTensor
from mindformers.tools.logger import logger
from mindformers.checkpoint.converter.convert_op import ConvertOp, ExpertsConvertOp, QKVConvertOp, QKVBiasConvertOp
from mindformers.parallel_core.transformer_config_utils import convert_to_transformer_config


@dataclass
class WeightTemplate:
    """Weight conversion template, supports bidirectional conversion HF ↔ MF"""
    weight_converters: List[ConvertOp] = field(default_factory=list)

    # Built by __post_init__
    hf_name_to_converter: Dict[str, ConvertOp] = field(default_factory=dict, init=False)
    mf_name_to_converter: Dict[str, ConvertOp] = field(default_factory=dict, init=False)
    name_to_weight: Dict[str, np.ndarray] = field(default_factory=dict, init=False)
    network_tensor_info: Dict[str, ShardedTensor] = field(default_factory=dict, init=False)

    def __post_init__(self):
        """Build hf_name_to_converter and mf_name_to_converter mapping dictionaries"""
        # Build mapping dictionaries directly from ConvertOp names
        # ConvertOp's hf_names and mf_names should already contain full paths with wildcards
        for converter in self.weight_converters:
            # Build mapping from HF names to ConvertOp
            for hf_name in converter.hf_names:
                self.hf_name_to_converter[hf_name] = converter
            # Build mapping from MF names to ConvertOp
            for mf_name in converter.mf_names:
                self.mf_name_to_converter[mf_name] = converter
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
        for converter in self.weight_converters:
            converter.mf_config = config
            if hasattr(converter, 'set_model_config'):
                converter.set_model_config(config)

    def get_mf_name(self, hf_name: str) -> str:
        """Get MF parameter name corresponding to HF parameter name"""
        # Find corresponding ConvertOp by matching against hf_name_to_converter
        converter = self.get_convert_op(hf_name, self.hf_name_to_converter)
        if converter is None:
            # Converter not found, return original name (may be unregistered parameter)
            return hf_name

        # Get converted MF names (ConvertOp will handle wildcard replacement)
        # For get_mf_name, we need to manually extract and replace wildcards
        mf_names = converter.mf_names

        # Extract layer index from hf_name if it matches a pattern with wildcard
        for hf_pattern in converter.hf_names:
            if "{}" in hf_pattern:
                # Try to match pattern and extract layer index
                re_pattern = hf_pattern.replace(".", r"\.").replace("{}", r"(\d+)")
                match = re.match(re_pattern, hf_name)
                if match:
                    layer_index = match.group(1)
                    # Replace wildcard in mf_names with extracted layer index
                    return [name.replace("{}", layer_index) for name in mf_names]

        # No wildcard pattern matched, return names as is (for non-layer weights)
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
                        mf_weight = Parameter(mf_weight, name=mf_name)
                    mf_state_dict[mf_name] = mf_weight
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
        For patterns with wildcard, extract the layer index part.
        For patterns without wildcard, return None (non-layer weights).
        
        Args:
            weight_name: Full weight name (e.g., "model.layers.0.self_attn.q_proj.weight")
            pattern: Pattern with wildcard (e.g., "model.layers.{}.self_attn.q_proj.weight")
        
        Returns:
            Layer key string (e.g., "model.layers.0") or None for non-layer weights
        """
        if "{}" not in pattern:
            return None

        # Convert pattern to regex and extract layer index
        re_pattern = pattern.replace(".", r"\.").replace("{}", r"(\d+)")
        match = re.match(re_pattern, weight_name)
        if match:
            layer_index = match.group(1)
            # Extract layer prefix part (e.g., "model.layers.0")
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
        if name in pattern_to_convert_ops:
            return pattern_to_convert_ops[name]
        for pattern in sorted(pattern_to_convert_ops, key=len, reverse=True):
            # Convert pattern with wildcard {} to regex pattern
            # Use same logic as ConvertOp._name_to_pattern for consistency
            re_pattern = pattern.replace(".", r"\.").replace("{}", "(.*)")
            if re.match(re_pattern, name):
                return pattern_to_convert_ops[pattern]
        # No matching ConvertOp found, return None (invalid/unregistered weight)
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
        Check if the given weight name belongs to MoE experts.

        This method checks whether a parameter corresponds to Mixture of Experts (MoE)
        expert weights by searching for the "experts" keyword in the weight name.

        Args:
            weight_name (str): The name of the weight parameter to check

        Returns:
            bool: True if the weight is an MoE expert weight (contains "experts"), False otherwise
        """
        if "experts" in weight_name:
            return True
        return False

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

        Args:
            dst_name (str): Destination parameter name to check if it's an expert weight
            num_moe_experts (int): Total number of MoE experts to stack
            src_tensor (ShardedTensor): Source sharded tensor containing expert weights

        Returns:
            ShardedTensor: Modified tensor with updated shape information if it's an expert weight,
                          otherwise returns the original tensor unchanged
        """
        if self.check_weights_for_experts(dst_name):
            # Calculate stacked shape by multiplying first dimension with number of experts
            param_shape = (src_tensor.global_shape[0] * num_moe_experts, src_tensor.global_shape[1])
            src_tensor.global_shape = param_shape
            src_tensor.local_shape = param_shape
            src_tensor.axis_fragmentations = [1] * len(param_shape)
        return src_tensor

    def get_num_moe_experts(self, dst_name, src_names_num):
        """Get the number of moe experts"""
        for converter in self.weight_converters:
            if isinstance(converter, ExpertsConvertOp):
                for mf_name in converter.mf_names:
                    re_pattern = mf_name.replace(".", r"\.").replace("{}", "(.*)")
                    match = re.findall(re_pattern, dst_name)
                    if match:
                        return src_names_num // len(converter.hf_names)
                raise ValueError("Can not find the num_moe_experts.")
        return None


    def preprocess_hf_weights(self,
                                             src_name,
                                             parameter,
                                             experts_weights,
                                             qkv_weights,
                                             num_moe_experts):
        """
        Get stacked tensor and rename source parameter names for MoE experts and QKV.

        This method iterates through weight converters, matches the source parameter name
        against corresponding converters, stacks MoE expert weights, and concatenates QKV weights.

        Args:
            src_name (str): Source parameter name, e.g., "model.layers.0.self_attn.q_proj.weight"
            parameter (np.ndarray): Parameter tensor to be processed
            experts_weights (dict): Dictionary storing MoE expert weights for accumulating multiple experts
            qkv_weights (dict): Dictionary storing Q/K/V weights for accumulating three projections
            num_moe_experts (int): Total number of MoE experts

        Returns:
            tuple: (target parameter name, processed tensor)
                - If weight collection is complete, returns converted name and stacked/concatenated tensor
                - If weights are still being collected, returns (None, None)

        Raises:
            ValueError: Raised when no matching weight name is found
        """
        for converter in self.weight_converters:
            hf_names = converter.hf_names
            mf_names = converter.mf_names
            for hf_name in hf_names:
                # Convert Hugging Face format name pattern to regex for matching
                re_pattern = hf_name.replace(".", r"\.").replace("{}", r"(\d+)")
                match = re.findall(re_pattern, src_name)
                # Stack the experts, with the number of experts being num_moe_experts
                if match and isinstance(converter, ExpertsConvertOp):
                    # Extract layer ID and expert ID from match results
                    layer_id = int(match[0][0])
                    expert_id = int(match[0][1])
                    src_name = hf_name.format(layer_id, "stack")
                    # Initialize expert weights list and store current expert parameter
                    if not src_name in experts_weights:
                        experts_weights[src_name] = [None] * num_moe_experts
                    experts_weights[src_name][expert_id] = parameter
                    # Check if all expert weights have been collected
                    if None in experts_weights[src_name]:
                        return None, None

                    # Stack all expert weights along new axis and reshape
                    shape = parameter.shape
                    stack_parameter = np.stack(experts_weights[src_name], axis=0)
                    stack_parameter = stack_parameter.reshape(num_moe_experts * shape[0], -1)
                    return src_name, stack_parameter
                # concat hole qkv weight with interleave
                if match and isinstance(converter, QKVConvertOp):
                    # Extract layer ID and build target name
                    layer_id = int(match[0])
                    src_name = hf_name.format(layer_id)
                    mf_name = mf_names[0].format(layer_id)
                    # Initialize QKV weights storage structure
                    if not mf_name in qkv_weights:
                        qkv_weights[mf_name] = {}
                    qkv_weights[mf_name][src_name] = parameter
                    # Check if Q/K/V three weights have all been collected
                    if len(qkv_weights[mf_name].keys()) < 3:
                        return None, None
                    # Concatenate Q/K/V weights in interleaved manner
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
