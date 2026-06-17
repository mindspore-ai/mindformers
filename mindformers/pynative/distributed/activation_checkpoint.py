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

import contextlib
import inspect
import re
from typing import Sequence, Tuple

import regex
import mindspore as ms

from mindspore import nn
from hyper_parallel.core.activation_checkpoint import(
    CheckpointPolicy,
    checkpoint_wrapper,
    swap_wrapper,
    SwapManager,
)

from mindformers.pynative.config.config import (
    RecomputeConfig,
    RecomputeCommConfig,
    SwapConfig,
)
from mindformers.tools.logger import logger

__all__ = [
    "apply_ac",
    "apply_recompute",
    "apply_swap",
]

_LAYER_ID_SPEC_PATTERN = re.compile(r"^(\d+)(?:-(\d+))?$")
_config_list = {}

# Generic activation-recompute marker. During recompute the wrapped forward re-runs in
# the backward pass; any per-forward side effect that must happen once (MoE aux-loss
# logging today; reused as-is by other aux losses / modules later) has to be skipped on
# that re-run. ``recompute_context_fn`` is a context_fn for ms.recompute /
# checkpoint_wrapper whose recompute_ctx is entered only on the backward re-run;
# ``is_in_recompute`` lets callers detect and skip it. A context_fn is used rather than
# the selective-checkpoint policy_fn because policy_fn is a per-op save/recompute
# decision hook present only for `select` mode, whereas this context_fn brackets the
# whole recompute re-run uniformly for both full and select modes.
# ``_RECOMPUTE_DEPTH`` is a nesting depth (not per-layer state): the guard only needs
# "are we inside any recompute". PyNative runs forward/recompute sequentially and each
# marker brackets one recompute via try/finally, so depth is balanced and never leaks
# across layers; nested recompute simply keeps depth > 0.
_RECOMPUTE_DEPTH = 0


@contextlib.contextmanager
def recompute_marker():
    """Mark the enclosed region as a backward-time recompute re-run."""
    global _RECOMPUTE_DEPTH
    _RECOMPUTE_DEPTH += 1
    try:
        yield
    finally:
        _RECOMPUTE_DEPTH -= 1


def recompute_context_fn():
    """``context_fn`` factory for ms.recompute(use_reentrant=False) / checkpoint_wrapper.

    Returns ``(forward_ctx, recompute_ctx)``: the original forward runs under a no-op
    context (side effects happen normally) while the backward-time recompute runs under
    the marker, so the re-run is detected and skipped.
    """
    return contextlib.nullcontext(), recompute_marker()


def is_in_recompute() -> bool:
    """Return True while inside a backward-time activation recompute re-run."""
    return _RECOMPUTE_DEPTH > 0

def _validate_recompute_config(
    recompute,
    recompute_comm,
    num_layers: int,
) -> None:
    """Validate recompute configuration."""
    rc = recompute
    rc_comm = recompute_comm

    need_recompute = rc.mode != "None"
    need_comm = rc_comm.enable
    if need_recompute:
        _validate_recompute_structure(rc)
        _validate_recompute_layer_specs(rc, num_layers)
    if need_comm:
        _validate_recompute_comm_structure(rc_comm)
        _validate_recompute_comm_layer_specs(rc_comm, num_layers)


def _validate_recompute_structure(recompute_cfg: RecomputeConfig) -> None:
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
    if recompute_cfg.mode == "full" and not recompute_cfg.full_recompute_layer:
        logger.error(f"[Recompute Config] {pfx}: mode is 'full' but full_recompute_layer is missing or empty")
        raise ValueError(
            f"{pfx}: mode is 'full' but full_recompute_layer is missing or empty"
        )
    if recompute_cfg.mode == "select" and not recompute_cfg.select_module:
        logger.error(f"[Recompute Config] {pfx}: mode is 'select' but select_module is missing or empty")
        raise ValueError(f"{pfx}: mode is 'select' but select_module is missing or empty")


def _validate_recompute_comm_structure(recompute_comm_cfg: RecomputeCommConfig) -> None:
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
    if recompute_comm_cfg.enable and not recompute_comm_cfg.select_module:
        logger.error(f"[Recompute Config] {pfx}: enable is True but select_module is missing or empty")
        raise ValueError(f"{pfx}: enable is True but select_module is missing or empty")


def _validate_recompute_layer_specs(recompute_cfg: RecomputeConfig, num_layers: int) -> None:
    """Validate and normalize layer spec strings in recompute configuration."""
    pfx = "TrainConfig.recompute_config.recompute"
    if recompute_cfg.mode == "None":
        return
    if recompute_cfg.full_recompute_layer:
        _validate_layer_specs(
            recompute_cfg.full_recompute_layer, f"{pfx}.full_recompute_layer", num_layers
        )
    elif recompute_cfg.mode == "select":
        for key, ranges in recompute_cfg.select_module.items():
            _validate_layer_specs(
                ranges, f"{pfx}.select_module[{key!r}]", num_layers
            )


def _validate_recompute_comm_layer_specs(
        recompute_comm_cfg: RecomputeCommConfig, num_layers: int) -> None:
    """Validate comm recompute layer specs."""
    if not recompute_comm_cfg.enable:
        return
    pfx = "TrainConfig.recompute_config.recompute_comm"
    for key, ranges in recompute_comm_cfg.select_module.items():
        _validate_layer_specs(
            ranges, f"{pfx}.select_module[{key!r}]", num_layers
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


def _validate_layer_specs(specs, label_prefix, num_layers, prefetch=None):
    """Validate layer spec strings.

    Checks: each spec is valid and specs are in ascending order.
    When prefetch is specified, validates that the maximum layer id
    plus prefetch does not exceed num_layers.
    """
    if not specs:
        return
    prev_hi = -1
    for i, spec in enumerate(specs):
        specs[i] = _validate_layer_id_range(f"{label_prefix}[{i}]", spec, num_layers)
        lo, hi = _parse_spec_lo_hi(specs[i])
        if lo <= prev_hi:
            logger.error(f"[Recompute Config] {label_prefix}: layer specs must be in strictly "
                         f"ascending order; spec[{i}] '{specs[i]}' starts at {lo} "
                         f"but previous spec ends at {prev_hi}")
            raise ValueError(
                f"{label_prefix}: layer specs must be in strictly ascending order; "
                f"spec[{i}] '{specs[i]}' starts at {lo} but previous spec ends at {prev_hi}"
            )
        prev_hi = hi
    if prefetch and prev_hi >= 0 and prev_hi + prefetch >= num_layers:
        logger.error(f"[Swap Config] {label_prefix}: max layer id {prev_hi} plus prefetch "
                     f"{prefetch} exceeds num_layers ({num_layers}); "
                     f"max allowed layer id is {num_layers - prefetch - 1}")
        raise ValueError(
            f"{label_prefix}: max layer id {prev_hi} plus prefetch {prefetch} "
            f"exceeds num_layers ({num_layers}); "
            f"max allowed layer id is {num_layers - prefetch - 1}"
        )


def _validate_swap_config(swap, num_layers: int) -> None:
    """Validate swap configuration."""
    sc = swap

    _validate_swap_structure(sc, num_layers)
    _validate_swap_layers(sc, num_layers)


def _validate_swap_entry_shared_fields(item: dict, label: str) -> None:
    """Validate shared swap entry fields: `layers`."""
    layers = item["layers"]
    if not isinstance(layers, (list, tuple)):
        raise TypeError(
            f"{label}.layers: expected list/tuple of layer ids or ranges like ['5'], ['0-19'], "
            f"got {type(layers).__name__}: {layers!r}"
        )


def _validate_swap_structure(swap_cfg: SwapConfig, num_layers: int) -> None:
    """Validate swap configuration structure."""
    pfx = "TrainConfig.swap_config"

    prefetch = swap_cfg.default_prefetch
    if prefetch >= num_layers or prefetch < 1:
        logger.error(f"[Swap Config] {pfx}.default_prefetch: value {prefetch} "
                     f"out of range [1, {num_layers - 1}]")
        raise ValueError(
            f"{pfx}.default_prefetch: value {prefetch} "
            f"out of range [1, {num_layers - 1}]"
        )
    if swap_cfg.layer_swap is not None:
        if not isinstance(swap_cfg.layer_swap, list):
            logger.error(f"[Swap Config] {pfx}.layer_swap must be a list, "
                         f"got {type(swap_cfg.layer_swap).__name__}")
            raise TypeError(f"{pfx}.layer_swap must be a list, got {type(swap_cfg.layer_swap).__name__}")
        for idx, item in enumerate(swap_cfg.layer_swap):
            label = f"{pfx}.layer_swap[{idx}]"
            if not isinstance(item, dict):
                raise TypeError(f"{label} must be a dict, got {type(item).__name__}")
            if "layers" not in item:
                raise ValueError(
                    f"{label}: each entry must include 'layers', got {item!r}"
                )
            _validate_swap_entry_shared_fields(item, label)

    if swap_cfg.op_swap is not None:
        if not isinstance(swap_cfg.op_swap, list):
            logger.error(f"[Swap Config] {pfx}.op_swap must be a list, "
                         f"got {type(swap_cfg.op_swap).__name__}")
            raise TypeError(f"{pfx}.op_swap must be a list, got {type(swap_cfg.op_swap).__name__}")
        for idx, item in enumerate(swap_cfg.op_swap):
            label = f"{pfx}.op_swap[{idx}]"
            if not isinstance(item, dict):
                raise TypeError(f"{label} must be a dict, got {type(item).__name__}")
            for k in ("op_name", "layers"):
                if k not in item:
                    raise ValueError(
                        f"{label}: each entry must include 'op_name' and 'layers', got {item!r}"
                    )
            opn = item["op_name"]
            if not isinstance(opn, str) or not opn.strip():
                raise ValueError(f"{label}.op_name must be a non-empty str, got {opn!r}")
            _validate_swap_entry_shared_fields(item, label)


def _validate_swap_layers(swap_cfg: SwapConfig, num_layers: int) -> None:
    """Validate swap layer specs."""
    pfx = "TrainConfig.swap_config"
    prefetch = swap_cfg.default_prefetch
    layer_items = (
        []
        if swap_cfg.layer_swap is None
        else ([swap_cfg.layer_swap] if isinstance(swap_cfg.layer_swap, dict) else swap_cfg.layer_swap)
    )
    for idx, item in enumerate(layer_items):
        label = f"{pfx}.layer_swap[{idx}]"
        _validate_layer_specs(item["layers"], f"{label}.layers", num_layers, prefetch)

    op_items = (
        []
        if swap_cfg.op_swap is None
        else ([swap_cfg.op_swap] if isinstance(swap_cfg.op_swap, dict) else swap_cfg.op_swap)
    )
    for idx, item in enumerate(op_items):
        label = f"{pfx}.op_swap[{idx}]"
        _validate_layer_specs(item["layers"], f"{label}.layers", num_layers, prefetch)


def regex_match(pattern, string, timeout=1):
    """Match pattern against string with timeout protection."""
    try:
        return regex.fullmatch(pattern, string, timeout=timeout)
    except TimeoutError as e:
        logger.warning(f"{e} Please check and fix it.")
    return None


def _parse_spec_lo_hi(item: str) -> Tuple[int, int]:
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
        lo, hi = _parse_spec_lo_hi(item)
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


def _get_modules_and_ops_list(model):
    """Get a dict of all modules and operators for each layer in the model."""
    layer_configs = {}

    for layer_id in range(model.layer_start, model.layer_end + 1):
        layer = model.layers[layer_id]
        layer_whitelist = []
        _get_single_layer_whitelist(layer, layer_whitelist)
        layer_configs[layer_id] = layer_whitelist

    return layer_configs


def _expand_select_module(config_list, select_module):
    """Expand wildcard patterns in select_module against the model whitelist."""
    layer_to_modules = {}

    for module_name, raw_ranges_str in select_module.items():
        raw_layer_ids = _parse_layer_ids(raw_ranges_str)
        matched = False

        for layer_id in raw_layer_ids:
            if layer_id not in config_list:
                continue
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


def _clean_and_parse_config(full_target_ids, select_module, label=None):
    """
    Deduplicate the expanded select_module into a Layer ID -> [Module Names] map.
    Parents are processed before children. Children are skipped if their parent is configured.
    Assumes select_module is already in {layer_id: [module_names]} format.
    """
    layer_to_modules = {}

    for layer_id, module_names in select_module.items():
        if layer_id in full_target_ids:
            continue

        sorted_names = sorted(module_names, key=lambda name: (name.count('.'), name))
        _add_modules_dedup(sorted_names, layer_id, layer_to_modules)

    if label:
        logger.info(f"--- Final Select {label} Configuration Map---")
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
                setattr(layer, name, checkpoint_wrapper(cell, context_fn=recompute_context_fn))
                log = f"{info}.{name}"
        for attr in dir(layer):
            if p == attr:
                operator = getattr(layer, attr)
                setattr(layer, attr, checkpoint_wrapper(operator, output_recompute=True))
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

    if _config_list:
        config_list = _config_list
    else:
        config_list = _get_modules_and_ops_list(model)

    if need_recompute:
        full_target_ids = _parse_layer_ids(rc.full_recompute_layer)
        if rc.mode == "select":
            select_module_list = _expand_select_module(config_list, rc.select_module)
            layer_to_modules = _clean_and_parse_config(full_target_ids, select_module_list, label="Recompute")

    if need_comm:
        comm_select_module_list = _expand_select_module(config_list, rc_comm.select_module)
        comm_layer_to_modules = _clean_and_parse_comm_config(full_target_ids, layer_to_modules, comm_select_module_list)

    if not hasattr(model, "layers"):
        raise ValueError(f"{type(model)} must have 'layers' attribute.")

    for layer_id in range(model.layer_start, model.layer_end + 1):
        if need_recompute and layer_id in full_target_ids:
            model.layers[layer_id] = checkpoint_wrapper(
                model.layers[layer_id], context_fn=recompute_context_fn
            )
            logger.info(f"Set full recompute at layer {layer_id}")

        if need_recompute and rc.mode == "select":
            _set_select_recompute(model.layers[layer_id], layer_id, layer_to_modules, add_prim_attr=False)

        if need_comm:
            _set_select_recompute(model.layers[layer_id], layer_id, comm_layer_to_modules, add_prim_attr=True)


def _build_policy_fn_swap(config):
    """Build swap policy from model config.

    Keep attention masks on device since FlashAttention backward consumes the
    shared local uint8 mask directly and the generic swap lifecycle may break
    that assumption.
    """
    seq_length = getattr(config, "seq_length", None)
    use_mask_compression = bool(
        getattr(config, "use_attn_mask_compression", False)
        or getattr(config, "use_eod_attn_mask_compression", False)
    )

    def _policy_fn_swap(x):
        if x.dtype != ms.uint8:
            return CheckpointPolicy.MUST_SWAP

        ndim = getattr(x, "ndim", 0)
        if ndim == 4 and x.shape[1] == 1:
            if seq_length is None or (x.shape[2] == x.shape[3] == seq_length):
                return CheckpointPolicy.MUST_SAVE

        if use_mask_compression and ndim == 2 and x.shape[0] == x.shape[1]:
            return CheckpointPolicy.MUST_SAVE

        return CheckpointPolicy.MUST_SWAP

    return _policy_fn_swap


def _expand_op_swap(config_list, op_swap):
    """Expand wildcard patterns in select_module against the model whitelist."""
    layer_to_modules = {}

    for item in op_swap:
        module_name, raw_ranges_str = list(item.items())[0]
        raw_layer_ids = _parse_layer_ids(raw_ranges_str)
        matched = False

        for layer_id in raw_layer_ids:
            if layer_id not in config_list:
                continue
            for item in config_list[layer_id]:
                if regex_match(module_name, item):
                    matched = True
                    if layer_id not in layer_to_modules:
                        layer_to_modules[layer_id] = []
                    if item not in layer_to_modules[layer_id]:
                        layer_to_modules[layer_id].append(item)

        if not matched:
            logger.warning(f"op swap pattern '{module_name}' did not match any module in the model, "
                           "please check your swap config.")

    return layer_to_modules


def parse_op_swap(op_swap) -> list:
    """Parse swap config."""
    op_swap_list = []
    if op_swap:
        for item in op_swap:
            op_name = item["op_name"]
            layers = item["layers"]
            op_swap_list.append({op_name: layers})

    return op_swap_list


def _set_pattern_swap(layer, p_list, policy_fn, info=''):
    """Recursively traverse layer cells along p_list path and apply swap_wrapper."""
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
                log = _set_pattern_swap(cell, p_list, policy_fn, info + f'.{name}')
                if log:
                    log_list.append(log[1:])
    else:
        # Last path segment: apply swap_wrapper to the target
        for name, cell in layer._cells.items():
            if p == name:
                setattr(layer, name, swap_wrapper(cell, policy_fn=policy_fn))
                log = f"{info}.{name}"
        for attr in dir(layer):
            if p == attr:
                operator = getattr(layer, attr)
                setattr(layer, attr, swap_wrapper(operator, policy_fn=policy_fn))
                log = f"{info}.{attr}"

    # Restore p_list so the caller's list is unchanged after recursion
    p_list.insert(0, p)
    if log_list:
        return " " + ", ".join(log_list)
    return log


def _set_op_swap(layer, layer_id, layer_to_modules, policy_fn):
    """Set op swap for a layer."""
    log_ops = []
    for pattern in layer_to_modules[layer_id]:
        log = _set_pattern_swap(layer, pattern.split(r'.'), policy_fn)
        if log:
            log_ops.append(log[1:])
    log_ops_str = ', '.join(log_ops)
    if log_ops_str:
        logger.info(f"Set select swap at layer {layer_id}: {log_ops_str}")


def apply_swap(
    model: nn.Cell,
    swap: SwapConfig,
) -> None:
    """Apply ``checkpoint_wrapper`` using ``swap_config``."""
    sc = swap
    prefetch = sc.default_prefetch
    full_target_ids = _parse_layer_ids(sc.layer_swap[0].get("layers", [])) if sc.layer_swap else set()
    layer_to_modules = {}
    if _config_list:
        config_list = _config_list
    else:
        config_list = _get_modules_and_ops_list(model)
    if sc.op_swap:
        op_swap_list = _expand_op_swap(config_list, parse_op_swap(sc.op_swap))
        layer_to_modules = _clean_and_parse_config(full_target_ids, op_swap_list, label="Swap")
    policy_fn = _build_policy_fn_swap(model.config)

    if not hasattr(model, "layers"):
        raise ValueError(f"{type(model)} must have 'layers' attribute.")

    for layer_id in range(model.layer_start, model.layer_end + 1):
        if layer_id in full_target_ids:
            model.layers[layer_id] = swap_wrapper(model.layers[layer_id], policy_fn=policy_fn)
            SwapManager().set_forward_prefetch_layer(model.layers[layer_id], model.layers[layer_id + prefetch])
            logger.info(f"Set layer swap at layer {layer_id}")

        if sc.op_swap and layer_id in layer_to_modules:
            _set_op_swap(model.layers[layer_id], layer_id, layer_to_modules, policy_fn)
            SwapManager().set_forward_prefetch_layer(model.layers[layer_id], model.layers[layer_id + prefetch])


def _check_recompute_swap_overlap(recompute, recompute_comm, swap):
    """Check if recompute and swap configs overlap on the same modules."""
    rc = recompute
    rc_comm = recompute_comm
    sc = swap

    recompute_layers = set()
    if rc.mode != "None":
        recompute_layers = _parse_layer_ids(rc.full_recompute_layer)
    swap_layers = set()
    if sc.enable and sc.layer_swap:
        swap_layers = _parse_layer_ids(sc.layer_swap[0].get("layers", []))

    recompute_select = {}
    if rc.mode == "select" and rc.select_module:
        recompute_select = _expand_select_module(_config_list, rc.select_module)
    if rc_comm.enable and rc_comm.select_module:
        comm_select = _expand_select_module(_config_list, rc_comm.select_module)
        for layer_id, mods in comm_select.items():
            if layer_id in recompute_select:
                recompute_select[layer_id] = list(set(recompute_select[layer_id]) | set(mods))
            else:
                recompute_select[layer_id] = list(mods)
    if recompute_select:
        recompute_select = _clean_and_parse_config(recompute_layers, recompute_select)
    swap_select = {}
    if sc.op_swap:
        swap_select = _expand_op_swap(_config_list, parse_op_swap(sc.op_swap))

    overlap = swap_layers & (recompute_layers | set(recompute_select.keys()))
    if overlap:
        logger.error(f"[Recompute/Swap Config] layers {sorted(overlap)} are configured for "
                     "both recompute and layer swap, which is not allowed")
        raise ValueError(
            f"layers {sorted(overlap)} are configured for both recompute "
            "and layer swap, which is not allowed"
        )

    layer_overlap_errors = []
    for layer_id, sw_mods in swap_select.items():
        if layer_id in recompute_layers:
            layer_overlap_errors.append(
                f"layer {layer_id}: layer is configured for full recompute, op swap is not allowed"
            )
            continue
        if layer_id not in recompute_select:
            continue
        overlap_pairs = []
        for sw_mod in sw_mods:
            for rc_mod in recompute_select[layer_id]:
                if sw_mod == rc_mod or sw_mod.startswith(rc_mod + ".") or rc_mod.startswith(sw_mod + "."):
                    overlap_pairs.append((sw_mod, rc_mod))
        if overlap_pairs:
            pairs_str = ", ".join(f"'{sw}'/'{rc}'" for sw, rc in overlap_pairs)
            layer_overlap_errors.append(
                f"layer {layer_id}: modules {pairs_str} have parent-child overlap"
            )
    if layer_overlap_errors:
        error_detail = "; ".join(layer_overlap_errors)
        logger.error(f"[Recompute/Swap Config] {error_detail}, which is not allowed")
        raise ValueError(f"{error_detail}, which is not allowed")


def apply_ac(
    model,
    recompute,
    recompute_comm,
    swap,
    pp,
):
    """Apply activation checkpointing to the model."""
    global _config_list
    enable_recompute = (recompute.mode != "None" or recompute_comm.enable)
    enable_swap = swap.enable
    if not (enable_recompute or enable_swap):
        return

    num_layers = None
    if hasattr(model, "config"):
        num_layers = model.config.num_layers
    if not num_layers:
        raise ValueError(f"{type(model)} must have 'config.num_layers' attribute.")

    _config_list = _get_modules_and_ops_list(model)

    if pp > 1 and enable_swap:
        logger.error("[Swap Config] swap is not supported with pipeline parallel")
        raise ValueError("swap is not supported with pipeline parallel")

    if enable_recompute and enable_swap:
        _check_recompute_swap_overlap(recompute, recompute_comm, swap)

    if enable_recompute:
        _validate_recompute_config(recompute, recompute_comm, num_layers)
        apply_recompute(model, recompute, recompute_comm)

    if enable_swap:
        _validate_swap_config(swap, num_layers)
        apply_swap(model, swap)
