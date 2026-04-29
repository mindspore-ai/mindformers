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
"""Pynative transformer helpers (recompute)."""
import inspect
import re
from typing import Sequence, Tuple, Set

import regex
from mindspore import nn
from hyper_parallel.core.activation_checkpoint import CheckpointPolicy, checkpoint_wrapper

from mindformers.pynative.config.config import (
    RecomputeConfig,
    RecomputeCommConfig,
    TrainConfig,
)
from mindformers.pynative.base_models.gpt.parallelize import _unwrap_gptmodel
from mindformers.tools.logger import logger

__all__ = [
    "apply_recompute_and_swap",
    "apply_recompute",
]

_LAYER_ID_SPEC_PATTERN = re.compile(r"^(\d+)(?:-(\d+))?$")


def _validate_recompute_config(config: TrainConfig, num_layers: int) -> None:
    """Validate recompute configuration."""

    pipeline_parallel = config.parallelism.pipeline_parallel
    rc = config.recompute
    rc_comm = config.recompute_comm

    need_recompute = rc.mode != "None"
    need_comm = rc_comm.enable
    if (need_recompute or need_comm) and pipeline_parallel != 1:
        logger.error("[Recompute Config] recompute/recompute_comm requires pipeline_parallel=1 "
                     f"(pipeline parallelism is not supported); got pipeline_parallel={pipeline_parallel}")
        raise ValueError(
            "TrainConfig: recompute/recompute_comm requires pipeline_parallel=1 "
            f"(pipeline parallelism is not supported); got pipeline_parallel={pipeline_parallel}"
        )
    if need_recompute:
        _validate_recompute_structure(rc, pipeline_parallel)
        _validate_recompute_layer_specs_no_pp(rc, num_layers)
    if need_comm:
        _validate_recompute_comm_structure(rc_comm, pipeline_parallel)
        _validate_recompute_comm_layer_specs_no_pp(rc_comm, num_layers)


def _validate_recompute_structure(recompute_cfg: RecomputeConfig, pp: int) -> None:
    """Validate recompute configuration structure and mode-specific requirements."""
    pfx = "TrainConfig.recompute"
    if recompute_cfg.select_module is not None:
        if not isinstance(recompute_cfg.select_module, dict):
            logger.error(f"[Recompute Config] {pfx}.select_module must be a dict "
                         f"(module path -> list of layer ranges), "
                         f"got {type(recompute_cfg.select_module).__name__}")
            raise TypeError(
                f"{pfx}.select_module must be a dict "
                f"(module path -> list of layer ranges), "
                f"got {type(recompute_cfg.select_module).__name__}"
            )
        for key, ranges in recompute_cfg.select_module.items():
            if not isinstance(key, str) or not key.strip():
                logger.error(f"[Recompute Config] {pfx}.select_module: invalid module path key {key!r}")
                raise ValueError(f"{pfx}.select_module: invalid module path key {key!r}")
            if not isinstance(ranges, (list, tuple)):
                logger.error(f"[Recompute Config] {pfx}.select_module[{key!r}]: "
                             f"value must be a list or tuple of layer ranges, "
                             f"got {type(ranges).__name__}")
                raise TypeError(
                    f"{pfx}.select_module[{key!r}]: value must be a list or tuple of layer ranges, "
                    f"got {type(ranges).__name__}"
                )
        _validate_select_module_size_for_pp(
            recompute_cfg.select_module,
            f"{pfx}.select_module",
            pp,
            "mode is 'select'",
        )
    if recompute_cfg.mode == "full" and not recompute_cfg.full_recompute_layer:
        logger.error(f"[Recompute Config] {pfx}: mode is 'full' but full_recompute_layer is missing or empty")
        raise ValueError(
            f"{pfx}: mode is 'full' but full_recompute_layer is missing or empty"
        )
    if recompute_cfg.full_recompute_layer is not None and len(recompute_cfg.full_recompute_layer) != pp:
        logger.error(f"[Recompute Config] {pfx}.full_recompute_layer must contain "
                     f"exactly one layer range string when pipeline_parallel={pp}; "
                     f"got {len(recompute_cfg.full_recompute_layer)} entries")
        raise ValueError(
            f"{pfx}.full_recompute_layer must contain exactly one layer range string when "
            f"pipeline_parallel={pp}; got {len(recompute_cfg.full_recompute_layer)} entries"
        )
    if recompute_cfg.mode == "select" and not recompute_cfg.select_module:
        logger.error(f"[Recompute Config] {pfx}: mode is 'select' but select_module is missing or empty")
        raise ValueError(f"{pfx}: mode is 'select' but select_module is missing or empty")


def _validate_recompute_comm_structure(recompute_comm_cfg: RecomputeCommConfig, pp: int) -> None:
    """Validate communication recompute configuration structure."""
    pfx = "TrainConfig.recompute_comm"
    if recompute_comm_cfg.select_module is not None:
        if not isinstance(recompute_comm_cfg.select_module, dict):
            logger.error(f"[Recompute Config] {pfx}.select_module must be a dict "
                         f"(comm op path -> list of layer ranges), "
                         f"got {type(recompute_comm_cfg.select_module).__name__}")
            raise TypeError(
                f"{pfx}.select_module must be a dict (comm op path -> list of layer ranges), "
                f"got {type(recompute_comm_cfg.select_module).__name__}"
            )
        for key, ranges in recompute_comm_cfg.select_module.items():
            if not isinstance(key, str) or not key.strip():
                logger.error(f"[Recompute Config] {pfx}.select_module: invalid comm op path key {key!r}")
                raise ValueError(f"{pfx}.select_module: invalid comm op path key {key!r}")
            if not isinstance(ranges, (list, tuple)):
                logger.error(f"[Recompute Config] {pfx}.select_module[{key!r}]: "
                             f"value must be a list or tuple of layer ranges, "
                             f"got {type(ranges).__name__}")
                raise TypeError(
                    f"{pfx}.select_module[{key!r}]: value must be a list or tuple of layer ranges, "
                    f"got {type(ranges).__name__}"
                )
        _validate_select_module_size_for_pp(
            recompute_comm_cfg.select_module,
            f"{pfx}.select_module",
            pp,
            "communication recompute is enabled",
        )
    if recompute_comm_cfg.enable and not recompute_comm_cfg.select_module:
        logger.error(f"[Recompute Config] {pfx}: enable is True but select_module is missing or empty")
        raise ValueError(f"{pfx}: enable is True but select_module is missing or empty")


def _validate_recompute_layer_specs_no_pp(recompute_cfg: RecomputeConfig, num_layers: int) -> None:
    """Validate and normalize layer spec strings in recompute configuration."""
    pfx = "TrainConfig.recompute_config.recompute"
    if recompute_cfg.mode == "None":
        return
    if recompute_cfg.full_recompute_layer:
        recompute_cfg.full_recompute_layer[0] = _validate_layer_id_range(
            f"{pfx}.full_recompute_layer", recompute_cfg.full_recompute_layer[0], num_layers
        )
    elif recompute_cfg.mode == "select":
        for key, ranges in recompute_cfg.select_module.items():
            ranges[0] = _validate_layer_id_range(
                f"{pfx}.select_module[{key!r}]", ranges[0], num_layers
            )


def _validate_recompute_comm_layer_specs_no_pp(
        recompute_comm_cfg: RecomputeCommConfig, num_layers: int) -> None:
    """Validate comm recompute layer specs when pipeline_parallel is 1."""
    if not recompute_comm_cfg.enable:
        return
    pfx = "TrainConfig.recompute_config.recompute_comm"
    for key, ranges in recompute_comm_cfg.select_module.items():
        ranges[0] = _validate_layer_id_range(
            f"{pfx}.select_module[{key!r}][0]", ranges[0], num_layers
        )


def _validate_select_module_size_for_pp(
    select_module: dict,
    label_prefix: str,
    pp: int,
    reason: str,
) -> None:
    """Validate that each select_module entry has exactly pp layer range strings."""
    for key, ranges in select_module.items():
        if len(ranges) != pp:
            logger.error(f"[Recompute Config] {label_prefix}[{key!r}]: must contain "
                         f"exactly one layer range string when {reason}; "
                         f"got {len(ranges)} entries")
            raise ValueError(
                f"{label_prefix}[{key!r}]: must contain exactly one layer range string when "
                f"{reason}; got {len(ranges)} entries"
            )


def _validate_layer_id_range(label: str, value: object, num_layers: int) -> str:
    """Validate one layer spec."""
    normalized = str(value).strip()
    m = _LAYER_ID_SPEC_PATTERN.match(normalized)
    if not m:
        logger.error(f"[Recompute Config] {label}: invalid layer spec {value!r}; "
                     "expected one non-negative integer or 'start-end' (e.g. '5', '0-19')")
        raise ValueError(
            f"{label}: invalid layer spec {value!r}; expected one non-negative integer or 'start-end' "
            "(e.g. '5', '0-19')"
        )
    lo = int(m.group(1))
    hi = int(m.group(2)) if m.group(2) is not None else lo
    if lo > hi:
        logger.error(f"[Recompute Config] {label}: range start must be <= end, got {value!r}")
        raise ValueError(f"{label}: range start must be <= end, got {value!r}")
    if hi >= num_layers:
        logger.error(f"[Recompute Config] {label}: layer id {hi} out of range "
                     f"[0, {num_layers - 1}], got {value!r}")
        raise ValueError(
            f"{label}: layer id {hi} out of range [0, {num_layers - 1}], got {value!r}"
        )
    return normalized


def regex_match(pattern, string, timeout=1):
    """Match pattern against string with timeout protection."""
    try:
        return regex.fullmatch(pattern, string, timeout=timeout)
    except TimeoutError as e:
        logger.warning(f"{e} Please check and fix it.")
    return None


def _parse_recompute_spec_lo_hi(item: str) -> Tuple[int, int]:
    """Parse a single layer spec string into (lo, hi) tuple."""
    text = str(item).strip()
    if "-" in text:
        left, right = text.split("-", 1)
        lo = int(left.strip())
        hi = int(right.strip())
        return lo, hi
    lo = int(text)
    return lo, lo


def _parse_layer_ids(specs: Sequence[str]) -> set:
    """Parse a sequence of layer spec strings into a set of layer ids."""
    if not specs:
        return set()
    out = set()
    for item in specs:
        lo, hi = _parse_recompute_spec_lo_hi(item)
        out.update(range(lo, hi + 1))
    return out


def _get_single_layer_whitelist(layer, whitelist, info=''):
    """Recursively get all modules and operators in a single layer."""
    cell_names = set(layer._cells.keys())
    # Collect operators: non-cell attributes that are functions
    for attr in dir(layer):
        if attr.startswith('_') or attr in cell_names:
            continue

        try:
            value = getattr(layer, attr)
            if inspect.isfunction(value):
                op_path = f"{info}.{attr}" if info else attr
                whitelist.append(op_path)
        except AttributeError:
            continue

    # Collect child cells and recurse into them
    for name, cell in layer._cells.items():
        current_path = f"{info}.{name}" if info else name
        whitelist.append(current_path)
        _get_single_layer_whitelist(cell, whitelist, current_path)


def _get_modules_and_ops_list(model, num_layers):
    """Get a list of all modules and operators for each layer in the model."""
    layer_configs = []

    for layer_id in range(num_layers):
        layer = model.layers[layer_id]
        layer_whitelist = []
        _get_single_layer_whitelist(layer, layer_whitelist)
        layer_configs.append(layer_whitelist)

    return layer_configs


def _expand_select_module(config_list, select_module):
    """Expand wildcard patterns in select_module against the model whitelist."""
    layer_to_modules = {}

    for module_name, raw_ranges_str in select_module.items():
        raw_layer_ids = _parse_layer_ids(raw_ranges_str)
        matched = False

        for layer_id in raw_layer_ids:
            # Match user pattern against each item in the layer whitelist
            for item in config_list[layer_id]:
                if regex_match(module_name, item):
                    matched = True
                    if layer_id not in layer_to_modules:
                        layer_to_modules[layer_id] = []
                    if item not in layer_to_modules[layer_id]:
                        layer_to_modules[layer_id].append(item)

        if not matched:
            logger.warning(f"select_module pattern '{module_name}' did not match any module in the model, "
                           "please check your recompute config.")

    return layer_to_modules


def _add_modules_dedup(module_names, layer_id, layer_to_modules, parent_modules=None):
    """Add module names to layer_to_modules with dedup and parent coverage check."""
    check_modules = parent_modules if parent_modules is not None else layer_to_modules
    for module_name in module_names:
        # Skip if a parent module is already configured (child is implicitly covered)
        is_covered = False
        if layer_id in check_modules:
            for potential_parent in check_modules[layer_id]:
                if module_name.startswith(potential_parent + "."):
                    is_covered = True
                    break
        if is_covered:
            continue
        if layer_id not in layer_to_modules:
            layer_to_modules[layer_id] = []
        layer_to_modules[layer_id].append(module_name)


def _clean_and_parse_config(full_target_ids, select_module):
    """
    Deduplicate the expanded select_module into a Layer ID -> [Module Names] map.
    Parents are processed before children. Children are skipped if their parent is configured.
    Assumes select_module is already in {layer_id: [module_names]} format.
    """
    layer_to_modules = {}

    for layer_id, module_names in select_module.items():
        if layer_id in full_target_ids:
            continue

        # Sort by hierarchy depth so parents are processed before children
        sorted_names = sorted(module_names, key=lambda name: (name.count('.'), name))
        _add_modules_dedup(sorted_names, layer_id, layer_to_modules)

    logger.info("--- Final Select Recompute Configuration Map---")
    for layer_id in sorted(layer_to_modules.keys()):
        modules = layer_to_modules[layer_id]
        logger.info(f"layer{layer_id}: {', '.join(modules) if modules else '(No Module)'}")
    logger.info("---------------------------------------------------------------------")

    return layer_to_modules


def _clean_and_parse_comm_config(full_target_ids, select_layer_to_modules, comm_modules):
    """Deduplicate comm select_module, skipping layers in full_target_ids or covered by parent."""
    layer_to_modules = {}
    for layer_id, module_names in comm_modules.items():
        if layer_id in full_target_ids:
            continue
        _add_modules_dedup(module_names, layer_id, layer_to_modules, parent_modules=select_layer_to_modules)
            

    logger.info("--- Final Comm Select Recompute Configuration Map---")
    for layer_id in sorted(layer_to_modules.keys()):
        modules = layer_to_modules[layer_id]
        logger.info(f"layer{layer_id}: {', '.join(modules) if modules else '(No Module)'}")
    logger.info("---------------------------------------------------------------------")

    return layer_to_modules


def _policy_fn_recompute(ctx, op, *args, **kwargs): # pylint: disable=W0613
    return CheckpointPolicy.MUST_RECOMPUTE


def _set_pattern_recompute(layer, p_list, add_prim_attr=False, info=''):
    """Recursively traverse layer cells along p_list path and apply checkpoint_wrapper."""
    log_list = []
    log = ''
    # Pop the next path segment to match
    if p_list:
        p = p_list.pop(0)
    else:
        return info
    if p_list:
        # Still have path segments left: recurse into matching child cells
        # pylint: disable=W0212
        for name, cell in layer._cells.items():
            if p == name:
                log = _set_pattern_recompute(cell, p_list, add_prim_attr, info + f'.{name}')
                if log:
                    log_list.append(log[1:])
    else:
        # Last path segment: apply checkpoint_wrapper to the target
        for name, cell in layer._cells.items():
            if p == name:
                if add_prim_attr:
                    logger.info(f"For communication recompute, {info.replace('.', '', 1)}.{name} "
                                "is expected to be operation but got cell, "
                                "this configuration will not be effective.")
                    continue
                setattr(layer, name, checkpoint_wrapper(cell))
                log = f"{info}.{name}"
        for attr in dir(layer):
            if p == attr:
                operator = getattr(layer, attr)
                setattr(layer, attr, checkpoint_wrapper(operator, policy_fn=_policy_fn_recompute))
                log = f"{info}.{attr}"

    # Restore p_list so the caller's list is unchanged after recursion
    p_list.insert(0, p)
    if log_list:
        return " " + ", ".join(log_list)
    return log


def _set_select_recompute(layer, layer_id, layer_to_modules, add_prim_attr=False):
    """Set select recompute or comm recompute for a layer."""
    if layer_id in layer_to_modules:
        log_ops = []
        for pattern in layer_to_modules[layer_id]:
            log = _set_pattern_recompute(layer, pattern.split(r'.'), add_prim_attr)
            if log:
                log_ops.append(log[1:])
        log_ops_str = ', '.join(log_ops)
        if log_ops_str:
            comm = 'comm ' if add_prim_attr else ''
            logger.info(f"Set select {comm}recompute at layer {layer_id}: {log_ops_str}")


def apply_recompute(
    model: nn.Cell,
    recompute_config: RecomputeConfig,
    recompute_comm_config: RecomputeCommConfig,
) -> None:
    """Apply ``checkpoint_wrapper`` using recompute and recompute_comm configs."""
    rc = recompute_config
    rc_comm = recompute_comm_config
    need_recompute = rc.mode != "None"
    need_comm = rc_comm.enable

    full_target_ids = set()
    layer_to_modules = {}
    comm_layer_to_modules = {}
    config_list = []

    if need_recompute or need_comm:
        config_list = _get_modules_and_ops_list(model, model.config.num_layers)

    if need_recompute:
        full_target_ids = _parse_layer_ids(rc.full_recompute_layer)
        if rc.mode == "select":
            select_module_list = _expand_select_module(config_list, rc.select_module)
            layer_to_modules = _clean_and_parse_config(full_target_ids, select_module_list)

    if need_comm:
        comm_select_module_list = _expand_select_module(config_list, rc_comm.select_module)
        comm_layer_to_modules = _clean_and_parse_comm_config(full_target_ids, layer_to_modules, comm_select_module_list)

    for layer_id in range(model.config.num_layers):
        if need_recompute and layer_id in full_target_ids:
            model.layers[layer_id] = checkpoint_wrapper(model.layers[layer_id])
            logger.info(f"Set full recompute at layer {layer_id}")

        if need_recompute and rc.mode == "select":
            _set_select_recompute(model.layers[layer_id], layer_id, layer_to_modules, add_prim_attr=False)

        if need_comm:
            _set_select_recompute(model.layers[layer_id], layer_id, comm_layer_to_modules, add_prim_attr=True)


def apply_recompute_and_swap(model: nn.Cell, config: TrainConfig) -> None:
    if config.recompute.mode != "None" or config.recompute_comm.enable:
        gpt_model = _unwrap_gptmodel(model)
        decoder = gpt_model.decoder
        _validate_recompute_config(config, decoder.config.num_layers)
        apply_recompute(decoder, config.recompute, config.recompute_comm)
