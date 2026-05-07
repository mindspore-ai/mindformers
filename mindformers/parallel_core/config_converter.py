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
"""Config Converter."""

from abc import ABC
from dataclasses import fields, dataclass, asdict
from typing import Any, Dict, Tuple, Union, Callable, List

from mindformers.tools import logger
from mindformers.parallel_core.transformer_config import TransformerConfig, MLATransformerConfig
from mindformers.parallel_core.transformer_config_utils import get_cp_comm_type, ConfigLogHandler, \
    ConfigConversionTracer, DEFAULT_WHITE_KEY


@dataclass
class ConversionContext:
    mapping: Dict[str, Any]
    reversed_mapping: Dict[str, List[str]]
    result: Dict[str, Any]
    log_handler: 'ConfigLogHandler'
    tracer: 'ConfigConversionTracer'


class ConfigConverter(ABC):
    """
    ConfigConverter: convert to TransformerConfig or MLATransformerConfig.
    """
    # Define explicit mapping rules, in the format:
    # "source_key" or  ("src_key1","src_key2") -> "target_key" or ("target_key", transform_func)
    CONFIG_MAPPING = {
        # Parallel Config
        ("data_parallel", "data_parallel_size"): "data_parallel_size",
        ("model_parallel", "tensor_model_parallel_size"): "tensor_model_parallel_size",
        ("pipeline_stage", "pipeline_model_parallel_size"): "pipeline_model_parallel_size",
        ("pp_interleave_num", "virtual_pipeline_model_parallel_size"): "virtual_pipeline_model_parallel_size",
        ("use_seq_parallel", "sequence_parallel"): "sequence_parallel",
        ("context_parallel", "context_parallel_size"): "context_parallel_size",
        ("expert_parallel", "expert_model_parallel_size"): "expert_model_parallel_size",
        ("expert_model_parallel", "expert_tensor_parallel_size"): "expert_tensor_parallel_size",
        "context_parallel_algo": ("cp_comm_type", get_cp_comm_type),
        "ulysses_degree_in_cp": "hierarchical_context_parallel_sizes",
        # CPU Offloading
        ("swap", "cpu_offloading"): "cpu_offloading",
        ("layer_swap", "cpu_offloading_num_layers"): "cpu_offloading_num_layers",
    }
    _RUNTIME_HINT_PREFIX = "_mf_runtime_"

    @classmethod
    def convert(cls, model_config: Dict[str, Any],
                is_mla_model: bool = False):
        """
        convert model config to TransformerConfig or MLATransformerConfig.
        Args:
            model_config (Dict[str, Any]): Source model configuration dictionary
            is_mla_model (bool, optional): if it is a mla model. Default: False
         Returns:
            Union[TransformerConfig, MLATransformerConfig]: Converted configuration object
        """
        # Initializes the conversion context.
        # including the mapping, conversion result, conversion error, and conversion record.
        log_handler = ConfigLogHandler()
        tracer = ConfigConversionTracer()
        mapping = cls._get_final_mapping(log_handler, is_mla_model)
        result: Dict[str, Any] = {}
        reversed_mapping: Dict[str, List[str]] = {}
        cls._get_reversed_mapping(mapping, reversed_mapping)
        ctx = ConversionContext(
            mapping=mapping,
            reversed_mapping=reversed_mapping,
            result=result,
            log_handler=log_handler,
            tracer=tracer
        )

        # Start conversion
        # Step 1: customized pre-processing, subclass override
        cls._pre_process(model_config, ctx)
        # Step 2: Apply mapping
        missing_keys = set()
        for src_key in model_config:
            if src_key in mapping:
                src_value = model_config[src_key]
                cls._apply_mapping_rule(src_key, src_value, ctx)
            else:
                missing_keys.add(src_key)
        # check HF config unmapping keys
        missing_keys = cls._filter_internal_runtime_hints(missing_keys)
        if missing_keys - DEFAULT_WHITE_KEY:
            log_handler.add_warning("Unmapped Keys",
                                    f"following keys:{missing_keys - DEFAULT_WHITE_KEY} not in mapping."
                                    f"Please check if need add mapping rule or add into @ignore_and_delete_parameter")
        # Step 3: check and raise warn/error info
        log_handler.check_and_raise()
        # Step 4: Print conversion detail
        tracer.print_summary()
        # Step 5: Instantiation
        try:
            config_cls = MLATransformerConfig if is_mla_model else TransformerConfig
            converted_config = config_cls(**result)
            cls._attach_internal_runtime_hints(converted_config, model_config)
            config_repr = "{\n" + ",\n".join(f"  {repr(k)}: {repr(v)}" for k, v in
                                             asdict(converted_config).items()) + "\n}"
            logger.info(f"The final converted {config_cls.__name__} is: {config_repr}")
            return converted_config
        except TypeError as e:
            raise RuntimeError(
                f"Failed to instantiate {'MLATransformerConfig' if is_mla_model else 'TransformerConfig'} because: {e}"
            ) from e

    @classmethod
    def _filter_internal_runtime_hints(cls, keys):
        """Exclude internal runtime hints from user-facing unmapped-key warnings."""
        return {key for key in keys if not key.startswith(cls._RUNTIME_HINT_PREFIX)}

    @classmethod
    def _attach_internal_runtime_hints(cls, converted_config, model_config):
        """Carry internal runtime hints to the converted config without exposing them as config keys."""
        for key, value in model_config.items():
            if key.startswith(cls._RUNTIME_HINT_PREFIX) and not hasattr(converted_config, key):
                setattr(converted_config, key, value)

    @classmethod
    def _get_final_mapping(cls, log_handler, is_mla_model: bool = False) -> Dict[str, Union[str, Tuple[str, Callable]]]:
        """
        Merge the CONFIG_MAPPING of all classes in MRO and add the default mapping of the same name (uncovered fields).
        Rule:
          - Process in MRO order (sub-category → parent class);
          - Subclass rules keep priority for target behavior.
          - Extra source aliases without transform can still map to an occupied target.
          - Finally, add a default rule with the same name  for the unmapped Megatron fields.
        """
        seen_target_keys = set()
        seen_source_keys = set()
        final_rules = []  # (sources, target_key, trans_func or None)

        # step1: Explicit CONFIG_MAPPING Merge (based on the MRO priority, subClass2 -> subClass1 --> common)
        for klass in cls.__mro__:
            if not hasattr(klass, 'CONFIG_MAPPING'):
                continue

            mapping = klass.CONFIG_MAPPING
            for src_keys, target_spec in mapping.items():
                if not isinstance(src_keys, tuple):
                    src_keys = (src_keys,)

                if isinstance(target_spec, str):
                    target_key = target_spec
                    trans_func = None
                elif isinstance(target_spec, tuple) and len(target_spec) == 2:
                    target_key, trans_func = target_spec
                else:
                    log_handler.add_error("Invalid Mapping Rule",
                                          f"src_key:{src_keys} with target spec '{target_spec}' "
                                          f"must be single target_key or (target_key, transform_func) "
                                          f"Other mapping rule should be implemented in post_process"
                                          )
                    continue
                # Keep the subclass target priority while preserving other source aliases for that target.
                new_src_keys = tuple(src_key for src_key in src_keys if src_key not in seen_source_keys)
                if not new_src_keys:
                    continue
                if target_key not in seen_target_keys:
                    seen_target_keys.add(target_key)
                    final_rules.append((new_src_keys, target_key, trans_func))
                    seen_source_keys.update(new_src_keys)
                elif trans_func is None:
                    final_rules.append((new_src_keys, target_key, trans_func))
                    seen_source_keys.update(new_src_keys)
        # step2: Default map with the same name
        megatron_fields = {f.name for f in fields(TransformerConfig)}
        existing_rule_keys = {(src_keys, target_key) for src_keys, target_key, _ in final_rules}
        all_default_fields = set(megatron_fields)
        # append mla field if needed
        if is_mla_model:
            mla_fields = {f.name for f in fields(MLATransformerConfig)}
            all_default_fields.update(mla_fields - megatron_fields)
        for field in all_default_fields:
            if field in seen_source_keys:
                continue
            rule_key = ((field,), field)
            if rule_key not in existing_rule_keys:
                final_rules.append((rule_key[0], rule_key[1], None))
        # step3: Build the final mapping dict (source_key -> target_spec)
        final: Dict[str, List[Union[str, Tuple[str, Callable]]]] = {}
        for src_keys, target_key, trans_func in final_rules:
            for src in src_keys:
                if src not in final:
                    final[src] = []
                if trans_func is None:
                    final[src].append(target_key)
                else:
                    final[src].append((target_key, trans_func))
        return final

    @classmethod
    def _get_reversed_mapping(cls, mapping, reversed_mapping):
        """
        Retrieved reversed final mapping used for tracer.
        """
        for src_key, target_spec in mapping.items():
            keys = []
            for item in target_spec:
                if isinstance(item, str):
                    keys.append(item)
                elif isinstance(item, tuple) and len(item) == 2:
                    keys.append(item[0])
            for target_key in keys:
                reversed_mapping.setdefault(target_key, []).append(src_key)

    @classmethod
    def _apply_mapping_rule(cls, src_key: str, src_value: Any, ctx: ConversionContext):
        """
        Apply final mapping rules to config.
        """
        rules = ctx.mapping[src_key]
        rule_list = rules if isinstance(rules, list) else [rules]
        for rule in rule_list:
            trans_func_name = None
            if isinstance(rule, str):
                target_key = rule
                target_value = src_value
            elif isinstance(rule, tuple) and len(rule) == 2:
                target_key, trans_func = rule
                try:
                    target_value = trans_func(src_value)
                    trans_func_name = getattr(trans_func, "__name__", str(trans_func))
                except (TypeError, ValueError, KeyError, AttributeError) as e:
                    ctx.log_handler.add_error(
                        "Function Convert Error",
                        f"`{src_key}` → `{target_key}` with func {trans_func_name}: {e}"
                    )
                    continue
            else:
                ctx.log_handler.add_error("Invalid Mapping Rule",
                                          f"src_key:{src_key} with target spec '{rule}' "
                                          f"must be single target_key or (target_key, transform_func) "
                                          f"Other mapping rule should be implemented in post_process"
                                          )
                continue
            # Conflicts check
            if target_key in ctx.result:
                if ctx.result[target_key] == target_value:
                    continue
                existing_sources = ctx.reversed_mapping.get(target_key, [])
                ctx.log_handler.add_error(
                    "Mapping Conflicts",
                    f"target key'{target_key}' is mapped by multiple source keys: {existing_sources} "
                    f"current convert is {src_key}:{src_value}, please check other source key"
                )
                continue

            ctx.result[target_key] = target_value
            ctx.tracer.record(
                source_key=src_key,
                source_value=src_value,
                target_key=target_key,
                target_value=target_value,
                trans_func_name=trans_func_name,
            )

    @classmethod
    def _pre_process(cls, model_config: Dict[str, Any], ctx: ConversionContext) -> None:
        """
        Subclasses can be overridden to add additional logic
        The 'parallel_config' nested structure in model_config is handled by default.
        """
        parallel_config = model_config.pop('parallel_config', None)
        if not parallel_config or not isinstance(parallel_config, dict):
            return

        for key, value in parallel_config.items():
            if key == 'recompute' and isinstance(value, dict):
                for recompute_key, recompute_value in value.items():
                    if recompute_key in ctx.mapping:
                        cls._apply_mapping_rule(recompute_key, recompute_value, ctx)
            elif key in ctx.mapping:
                cls._apply_mapping_rule(key, value, ctx)
