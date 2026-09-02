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
import contextvars
import inspect
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Sequence, Tuple

import regex
import mindspore as ms

from mindspore import nn
from hyper_parallel.core.activation_checkpoint import(
    CheckpointPolicy,
    swap_wrapper,
    SwapManager,
)

from mindformers.pynative.config.config import (
    RecomputeConfig,
    RecomputeCommConfig,
    SwapConfig,
)
from mindformers.pynative.distributed.checkpoint_backend import (
    create_checkpoint_backend,
)
from mindformers.tools.logger import logger

__all__ = [
    "apply_ac",
    "apply_recompute",
    "apply_swap",
    "get_recompute_metadata",
    "is_in_recompute",
    "recompute_context_fn",
    "save_for_recompute",
]

_LAYER_ID_SPEC_PATTERN = re.compile(r"^(\d+)(?:-(\d+))?$")
_config_list = {}

# Generic activation-recompute marker. During recompute the wrapped forward re-runs in
# the backward pass; any per-forward side effect that must happen once (MoE aux-loss
# logging today; reused as-is by other aux losses / modules later) has to be skipped on
# that re-run. ``recompute_context_fn`` is shared by both checkpoint backends;
# its recompute context is entered only on the backward re-run.
# ``is_in_recompute`` lets callers detect and skip it. A context_fn is used rather than
# the selective-checkpoint policy_fn because policy_fn is a per-op save/recompute
# decision hook present only for `select` mode, whereas this context_fn brackets the
# whole recompute re-run uniformly for both full and select modes.
# Besides marking the backward-time replay, the context owns tiny Python metadata
# produced during the original forward and needed again during replay. This is used
# by EP to retain host-side split lists, avoiding a second counts D2H. The state is
# created per checkpoint invocation (rather than stored on the module), so pipeline
# micro-batches, nested checkpoints and concurrent forward/backward threads do not
# overwrite one another. A queue per key supports the same module being called more
# than once within one checkpointed region.
class _RecomputeInvocationState:
    """Metadata shared by one checkpoint invocation's forward and replay contexts."""

    def __init__(self):
        self.metadata = defaultdict(deque)


_RECOMPUTE_CONTEXT = contextvars.ContextVar("mindformers_recompute_context", default=None)


@contextlib.contextmanager
def _recompute_phase(state, is_recompute):
    """Activate one invocation-local forward or replay phase."""
    token = _RECOMPUTE_CONTEXT.set((state, is_recompute))
    try:
        yield
    finally:
        _RECOMPUTE_CONTEXT.reset(token)


@contextlib.contextmanager
def recompute_marker(state=None):
    """Mark the enclosed region as a backward-time recompute re-run."""
    if state is None:
        state = _RecomputeInvocationState()
    with _recompute_phase(state, is_recompute=True):
        yield


def recompute_context_fn():
    """Create matching forward/replay contexts for either checkpoint backend.

    Returns ``(forward_ctx, recompute_ctx)`` sharing invocation-local metadata. The
    original forward records small non-tensor values needed by replay, while the
    backward-time recompute consumes them and is marked so one-shot side effects are
    skipped.
    """
    state = _RecomputeInvocationState()
    return (
        _recompute_phase(state, is_recompute=False),
        recompute_marker(state),
    )


def is_in_recompute() -> bool:
    """Return True while inside a backward-time activation recompute re-run."""
    context = _RECOMPUTE_CONTEXT.get()
    return context is not None and context[1]


def save_for_recompute(key, value) -> bool:
    """Save a small Python metadata value for this checkpoint invocation's replay.

    The call is a no-op outside an original checkpoint forward. Values are queued
    by ``key`` so repeated calls are replayed in original execution order.

    Returns:
        bool: Whether the value was saved in an active checkpoint context.
    """
    context = _RECOMPUTE_CONTEXT.get()
    if context is None or context[1]:
        return False
    state, _ = context
    state.metadata[key].append(value)
    return True


def get_recompute_metadata(key):
    """Consume metadata saved by the matching original checkpoint forward.

    Returns ``None`` outside a recompute replay. During replay a missing value is
    an invariant violation: silently performing the host operation again would
    hide an ordering/key bug and reintroduce the synchronization being avoided.
    """
    context = _RECOMPUTE_CONTEXT.get()
    if context is None or not context[1]:
        return None
    state, _ = context
    values = state.metadata.get(key)
    if not values:
        raise RuntimeError(
            f"No forward metadata was saved for recompute key {key!r}. "
            "Ensure the save and replay paths use the same checkpoint-local key and call order."
        )
    return values.popleft()


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
    _validate_exclude_op(rc, num_layers, need_recompute)


def _validate_exclude_op(recompute_cfg: RecomputeConfig, num_layers: int, recompute_enabled: bool = True) -> None:
    """Validate the ``exclude_op`` config."""
    pfx = "TrainConfig.recompute.exclude_op"
    names = recompute_cfg.exclude_op
    if names is None:
        return
    if not recompute_enabled and names:
        logger.warning("[Recompute Config] exclude_op is configured but recompute mode is 'None'; "
                       "exclude_op has no effect without recompute.")
    if not isinstance(names, dict):
        logger.error(f"[Recompute Config] {pfx} must be a dict "
                     f"(module path -> list of layer ranges), "
                     f"got {type(names).__name__}")
        raise TypeError(
            f"{pfx} must be a dict "
            f"(module path -> list of layer ranges), "
            f"got {type(names).__name__}"
        )
    for key, ranges in names.items():
        if not isinstance(key, str) or not key.strip():
            logger.error(f"[Recompute Config] {pfx}: invalid module path key {key!r}")
            raise ValueError(f"{pfx}: invalid module path key {key!r}")
        if not isinstance(ranges, (list, tuple)):
            logger.error(f"[Recompute Config] {pfx}[{key!r}]: "
                         f"value must be a list or tuple of layer ranges, "
                         f"got {type(ranges).__name__}")
            raise TypeError(
                f"{pfx}[{key!r}]: value must be a list or tuple of layer ranges, "
                f"got {type(ranges).__name__}"
            )
        _validate_layer_specs(ranges, f"{pfx}[{key!r}]", num_layers)


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
    if recompute_cfg.mode == "select":
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


def _iter_class_owned_callable_attrs(layer, excluded_names):
    """Yield replaceable callable descriptors declared by custom Cell classes.

    Stop before MindSpore's ``nn.Cell`` base so framework methods are not
    exposed as ``exclude_op`` targets. Read-only properties are intentionally
    skipped because the exclusion wrapper cannot be installed back on them.
    """
    seen = set(excluded_names)
    for cls in type(layer).__mro__:
        if cls in (nn.Cell, object):
            break
        for attr, descriptor in vars(cls).items():
            if attr in seen or attr.startswith('_') or attr == "construct":
                continue
            if isinstance(descriptor, property):
                if descriptor.fset is None:
                    continue
            elif not (
                    inspect.isfunction(descriptor)
                    or isinstance(descriptor, (staticmethod, classmethod))):
                continue
            try:
                value = getattr(layer, attr)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                continue
            if callable(value):
                seen.add(attr)
                yield attr


def _get_single_layer_whitelist(layer, whitelist, info=''):
    """Recursively get all modules and operators in a single layer."""
    cell_names = set(layer._cells.keys())

    # Collect comm ops from the registry.
    if hasattr(layer, '_comm_ops'):
        for name in layer._comm_ops:
            op_path = f"{info}.{name}" if info else name
            whitelist.append(op_path)

    # Collect instance-owned callable attributes. Restricting this to
    # ``__dict__`` avoids exposing inherited Cell methods such as ``construct``
    # and ``set_train``, while still covering Python functions, ``mint.*``
    # functions, Primitive instances, and user callable objects assigned to a
    # Cell attribute.
    for attr, value in vars(layer).items():
        if attr.startswith('_') or attr in cell_names:
            continue
        if callable(value):
            op_path = f"{info}.{attr}" if info else attr
            whitelist.append(op_path)

    # Collect bound methods and replaceable callable properties declared on
    # custom Cell classes. Instance attributes above take precedence.
    instance_names = set(vars(layer))
    for attr in _iter_class_owned_callable_attrs(
            layer, cell_names.union(instance_names)):
        op_path = f"{info}.{attr}" if info else attr
        whitelist.append(op_path)

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


def _expand_select_module(config_list, module_dict, label="select_module", config_name="recompute config"):
    """Expand wildcard patterns against the model whitelist."""
    layer_to_modules = {}

    for module_name, raw_ranges_str in module_dict.items():
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
            logger.warning(f"{label} pattern '{module_name}' did not match any module in the model, "
                           f"please check your {config_name}.")

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
    Deduplicate the expanded module pattern dict into a Layer ID -> [Module Names] map.
    Parents are processed before children. Children are skipped if their parent is configured.
    Assumes input is already in {layer_id: [module_names]} format.
    """
    layer_to_modules = {}

    for layer_id, module_names in select_module.items():
        if layer_id in full_target_ids:
            continue

        sorted_names = sorted(module_names, key=lambda name: (name.count('.'), name))
        _add_modules_dedup(sorted_names, layer_id, layer_to_modules)

    if label:
        logger.info(f"--- Final {label} Configuration Map---")
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


def _create_checkpoint_backend(use_reentrant):
    """Create a checkpoint backend bound to MindFormers replay context."""
    return create_checkpoint_backend(use_reentrant, recompute_context_fn)


def _install_callable_wrapper(layer, name, wrapped):
    """Replace an instance callable, bound method, or settable property."""
    if name in vars(layer):
        setattr(layer, name, wrapped)
        return

    descriptor = inspect.getattr_static(layer, name, None)
    if isinstance(descriptor, property):
        descriptor.fset(layer, wrapped)
        return
    if inspect.isfunction(descriptor) or isinstance(
            descriptor, (staticmethod, classmethod)):
        object.__setattr__(layer, name, wrapped)
        return
    setattr(layer, name, wrapped)


def _set_pattern_recompute(
        layer, p_list, backend, add_prim_attr=False, info=''):
    """Find the configured path and install its recompute wrapper."""
    log_list = []
    log = ''
    # Pop the next path segment to match
    if p_list:
        p = p_list.pop(0)
    else:
        return info
    if p_list:
        for name, cell in layer._cells.items():
            if p == name:
                log = _set_pattern_recompute(
                    cell,
                    p_list,
                    backend,
                    add_prim_attr,
                    info + f'.{name}',
                )
                if log:
                    log_list.append(log[1:])
    else:
        for name, cell in layer._cells.items():
            if p == name:
                if add_prim_attr:
                    logger.info(f"For communication recompute, {info.replace('.', '', 1)}.{name} "
                                "is expected to be operation but got cell, "
                                "this configuration will not be effective.")
                    continue
                setattr(
                    layer,
                    name,
                    backend.wrap_cell(cell),
                )
                log = f"{info}.{name}"
        if not log and p not in layer._cells and hasattr(layer, p):
            operator = getattr(layer, p)
            if callable(operator):
                _install_callable_wrapper(
                    layer, p, backend.wrap_callable(operator))
                log = f"{info}.{p}"

    # Restore p_list so the caller's list is unchanged after recursion
    p_list.insert(0, p)
    if log_list:
        return " " + ", ".join(log_list)
    return log


def _set_select_recompute(
        layer, layer_id, layer_to_modules, backend, add_prim_attr=False):
    """Set select recompute or comm recompute for a layer."""
    if layer_id in layer_to_modules:
        log_ops = []
        for pattern in layer_to_modules[layer_id]:
            log = _set_pattern_recompute(
                layer,
                pattern.split(r'.'),
                backend,
                add_prim_attr,
            )
            if log:
                log_ops.append(log[1:])
        log_ops_str = ', '.join(log_ops)
        if log_ops_str:
            comm = 'comm ' if add_prim_attr else ''
            logger.info(
                f"Set select {comm}{backend.log_prefix}recompute at layer "
                f"{layer_id}: {log_ops_str}"
            )


def _set_pattern_exclude(layer, p_list, backend, info=''):
    """Apply the matching exclude wrapper to a module path within a layer.

    Matches two target types at the final path segment:
    1. Cell in ``_cells``
    2. Comm op in ``_comm_ops`` registry
    3. Replaceable callable attribute or custom-class descriptor
    """
    log_list = []
    log = ''
    if p_list:
        p = p_list.pop(0)
    else:
        return info
    if p_list:
        # Compound slot match (e.g. "input" + "allgather" → "input.allgather").
        slot = p + '.' + '.'.join(p_list)
        if hasattr(layer, '_comm_ops') and slot in layer._comm_ops:
            entry = layer._comm_ops[slot]
            entry['fn'] = backend.wrap_exclude(entry['fn'])
            log = f"{info}.{slot}"
            p_list.clear()
        else:
            for name, cell in layer._cells.items():
                if p == name:
                    log = _set_pattern_exclude(
                        cell, p_list, backend, info + f'.{name}')
                    if log:
                        log_list.append(log[1:])
    else:
        # Leaf segment: match cells, comm ops, then functional operators.
        for name, cell in layer._cells.items():
            if p == name:
                setattr(
                    layer,
                    name,
                    backend.wrap_exclude(cell),
                )
                log = f"{info}.{name}"
        if hasattr(layer, '_comm_ops') and p in layer._comm_ops:
            entry = layer._comm_ops[p]
            entry['fn'] = backend.wrap_exclude(entry['fn'])
            log = f"{info}.{p}"
        if not log and p not in layer._cells and hasattr(layer, p):
            operator = getattr(layer, p)
            if callable(operator):
                _install_callable_wrapper(
                    layer, p, backend.wrap_exclude(operator))
                log = f"{info}.{p}"

    # Restore p_list so the caller's list is unchanged after recursion
    p_list.insert(0, p)
    if log_list:
        return " " + ", ".join(log_list)
    return log


def _set_exclude_recompute(
        layer, layer_id, exclude_layer_to_modules, backend):
    """Apply exclude recompute using the surrounding layer's checkpoint type."""
    if layer_id not in exclude_layer_to_modules:
        return
    log_ops = []
    for pattern in exclude_layer_to_modules[layer_id]:
        log = _set_pattern_exclude(
            layer, pattern.split(r'.'), backend)
        if log:
            log_ops.append(log[1:])
    log_ops_str = ', '.join(log_ops)
    if log_ops_str:
        logger.info(f"Set exclude recompute at layer {layer_id}: {log_ops_str}")


@dataclass(frozen=True)
class _RecomputePlan:
    """Backend-neutral checkpoint targets resolved from user configuration."""

    full_target_ids: set
    select_layer_to_modules: dict
    exclude_layer_to_modules: dict
    comm_layer_to_modules: dict


def _build_recompute_plan(model, recompute_config, recompute_comm_config):
    """Resolve layer ranges and patterns without selecting an implementation."""
    rc = recompute_config
    rc_comm = recompute_comm_config
    need_recompute = rc.mode != "None"

    config_list = _config_list or _get_modules_and_ops_list(model)
    full_target_ids = set()
    select_layer_to_modules = {}
    exclude_layer_to_modules = {}
    comm_layer_to_modules = {}

    if need_recompute:
        full_target_ids = _parse_layer_ids(rc.full_recompute_layer)
        if rc.mode == "select":
            select_modules = _expand_select_module(
                config_list, rc.select_module)
            select_layer_to_modules = _clean_and_parse_config(
                full_target_ids, select_modules, label="Recompute")

    if need_recompute and isinstance(rc.exclude_op, dict) and rc.exclude_op:
        exclude_modules = _expand_select_module(
            config_list, rc.exclude_op, label="exclude_op")
        exclude_layer_to_modules = _clean_and_parse_config(
            set(), exclude_modules, label="ExcludeOp")

    if rc_comm.enable:
        comm_modules = _expand_select_module(
            config_list,
            rc_comm.select_module,
            label="comm_select_module",
        )
        comm_layer_to_modules = _clean_and_parse_comm_config(
            full_target_ids, select_layer_to_modules, comm_modules)

    return _RecomputePlan(
        full_target_ids=full_target_ids,
        select_layer_to_modules=select_layer_to_modules,
        exclude_layer_to_modules=exclude_layer_to_modules,
        comm_layer_to_modules=comm_layer_to_modules,
    )


def apply_recompute(
    model,
    recompute_config: RecomputeConfig,
    recompute_comm_config: RecomputeCommConfig,
    backend=None,
) -> None:
    """Build and apply a backend-neutral activation-recompute plan.

    ``model`` may be a bare decoder ``TransformerBlock`` or a ``_MtpLayerView``
    that also exposes the MTP layers as the last global layer ids.
    """
    rc = recompute_config
    rc_comm = recompute_comm_config
    need_recompute = rc.mode != "None"
    need_comm = rc_comm.enable
    if backend is None:
        backend = _create_checkpoint_backend(rc.use_reentrant)
        backend.validate_recompute(rc, rc_comm)

    if not hasattr(model, "layers"):
        raise ValueError(f"{type(model)} must have 'layers' attribute.")
    plan = _build_recompute_plan(model, rc, rc_comm)

    for layer_id in range(model.layer_start, model.layer_end + 1):
        layer = model.layers[layer_id]
        if _disables_activation_recompute(layer):
            logger.info(
                "Skip activation recompute at DSA warm-up layer %s: "
                "the frozen trunk is detached by the layer backward boundary.",
                layer_id,
            )
            continue
        # Step 1: exclude (must be before checkpoint_wrapper)
        if plan.exclude_layer_to_modules:
            active_excludes, inactive_excludes = backend.partition_excludes(
                layer_id,
                plan.exclude_layer_to_modules.get(layer_id, ()),
                plan.full_target_ids,
                plan.select_layer_to_modules,
            )
            for pattern in inactive_excludes:
                logger.info(
                    "Skip exclude_op '%s' at layer %s because it is outside "
                    "every %s checkpoint boundary.",
                    pattern,
                    layer_id,
                    backend.name,
                )
            if active_excludes:
                _set_exclude_recompute(
                    model.layers[layer_id],
                    layer_id,
                    {layer_id: active_excludes},
                    backend,
                )

        # Step 2: full recompute
        if need_recompute and layer_id in plan.full_target_ids:
            model.layers[layer_id] = backend.wrap_cell(model.layers[layer_id])
            logger.info(
                f"Set full recompute at layer {layer_id} ({backend.name})")

        # Step 3: select recompute
        if need_recompute and rc.mode == "select":
            _set_select_recompute(
                model.layers[layer_id],
                layer_id,
                plan.select_layer_to_modules,
                backend,
            )

        # Step 4: comm recompute
        if need_comm:
            _set_select_recompute(
                model.layers[layer_id],
                layer_id,
                plan.comm_layer_to_modules,
                backend,
                add_prim_attr=True,
            )



def _disables_activation_recompute(module):
    """Return whether a module contains a DSA1 layer that cuts its trunk graph."""
    if getattr(module, "disable_activation_recompute", False):
        return True
    return any(_disables_activation_recompute(cell) for cell in module._cells.values())

def _tensor_storage_ptr(tensor):
    """Return a tensor's storage pointer, or ``None`` for empty/non-tensors."""
    if not isinstance(tensor, ms.Tensor):
        return None
    storage = tensor.untyped_storage()
    if storage.size() == 0:
        return None
    return storage.data_ptr()


def _collect_tensor_storage_ptrs(tree):
    """Collect tensor storage pointers from nested layer inputs."""
    storage_ptrs = set()

    def _collect(value):
        storage_ptr = _tensor_storage_ptr(value)
        if storage_ptr is not None:
            storage_ptrs.add(storage_ptr)
            return
        if isinstance(value, dict):
            for item in value.values():
                _collect(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                _collect(item)

    _collect(tree)
    return storage_ptrs


class _SwapPolicy:
    """Keep storage owned by enclosing model inputs out of per-layer swap."""

    def __init__(self):
        self._external_input_storage_ptrs = set()

    def capture_external_inputs(self, inputs):
        """Refresh storage identities at the current enclosing model boundary."""
        self._external_input_storage_ptrs = _collect_tensor_storage_ptrs(inputs)

    def __call__(self, tensor):
        storage_ptr = _tensor_storage_ptr(tensor)
        if storage_ptr is not None and storage_ptr in self._external_input_storage_ptrs:
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.MUST_SWAP


def _build_policy_fn_swap():
    """Build a storage-aware layer-swap policy independent of tensor shape."""
    return _SwapPolicy()


class _SwapInputBoundary:
    """Describe one enclosing input boundary and the layers governed by it."""

    def __init__(self, module, layer_ids, exclude_arg_names=()):
        self.module = module
        self.layer_ids = frozenset(layer_ids)
        self.exclude_arg_names = frozenset(exclude_arg_names)


def _register_swap_input_storage_hook(boundary, policy_fn):
    """Capture selected boundary inputs by construct argument name."""
    module = boundary.module
    construct_signature = inspect.signature(module.construct)
    unknown_names = boundary.exclude_arg_names.difference(construct_signature.parameters)
    if unknown_names:
        raise ValueError(
            f"Swap input boundary excludes unknown construct arguments: {sorted(unknown_names)}"
        )

    def _capture_input_storage(mod, args, kwargs):  # pylint: disable=unused-argument
        bound_inputs = construct_signature.bind_partial(*args, **kwargs).arguments
        selected_inputs = {
            name: value for name, value in bound_inputs.items()
            if name not in boundary.exclude_arg_names
        }
        policy_fn.capture_external_inputs(selected_inputs)

    old_handle = getattr(module, "_swap_input_storage_hook_handle", None)
    if old_handle is not None:
        old_handle.remove()
    handle = module.register_forward_pre_hook(_capture_input_storage, with_kwargs=True)
    module._swap_input_storage_hook_handle = handle


def _build_swap_boundary_policies(model, input_boundaries, active_layer_ids):
    """Build isolated policies and map every governed layer to its boundary policy."""
    if input_boundaries is None:
        input_boundaries = (
            _SwapInputBoundary(model, range(model.layer_start, model.layer_end + 1)),
        )

    default_policy = _build_policy_fn_swap()
    policy_by_layer = {}
    valid_layer_ids = set(range(model.layer_start, model.layer_end + 1))
    for boundary in input_boundaries:
        unknown_layer_ids = boundary.layer_ids.difference(valid_layer_ids)
        if unknown_layer_ids:
            raise ValueError(f"Swap input boundary contains unknown layer ids: {sorted(unknown_layer_ids)}")
        governed_layer_ids = boundary.layer_ids.intersection(active_layer_ids)
        if not governed_layer_ids:
            continue
        duplicate_layer_ids = governed_layer_ids.intersection(policy_by_layer)
        if duplicate_layer_ids:
            raise ValueError(f"Swap input boundaries overlap at layer ids: {sorted(duplicate_layer_ids)}")

        policy_fn = _build_policy_fn_swap()
        _register_swap_input_storage_hook(boundary, policy_fn)
        for layer_id in governed_layer_ids:
            policy_by_layer[layer_id] = policy_fn

    return default_policy, policy_by_layer


def _expand_op_swap(config_list, op_swap):
    """Expand wildcard patterns in op_swap against the model whitelist."""
    op_swap_dict = {}
    for item in op_swap:
        op_swap_dict.update(item)
    return _expand_select_module(config_list, op_swap_dict, label="op_swap", config_name="swap config")


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
    input_boundaries=None,
) -> None:
    """Apply swap wrappers using the configured policy."""
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
    if not hasattr(model, "layers"):
        raise ValueError(f"{type(model)} must have 'layers' attribute.")
    active_layer_ids = full_target_ids.union(layer_to_modules)
    default_policy, policy_by_layer = _build_swap_boundary_policies(
        model, input_boundaries, active_layer_ids
    )

    prefetch_pairs = set()
    for layer_id in range(model.layer_start, model.layer_end + 1):
        policy_fn = policy_by_layer.get(layer_id, default_policy)
        if layer_id in full_target_ids:
            model.layers[layer_id] = swap_wrapper(model.layers[layer_id], policy_fn=policy_fn)
            prefetch_pairs.add((layer_id, layer_id + prefetch))
            logger.info(f"Set layer swap at layer {layer_id}")

        if sc.op_swap and layer_id in layer_to_modules:
            _set_op_swap(model.layers[layer_id], layer_id, layer_to_modules, policy_fn)
            prefetch_pairs.add((layer_id, layer_id + prefetch))

    for layer_id, prefetch_layer_id in sorted(prefetch_pairs):
        SwapManager().set_forward_prefetch_layer(model.layers[layer_id], model.layers[prefetch_layer_id])


class _MtpLayerIndex:
    """Global-layer-id indexer over decoder layers plus MTP layers.

    Decoder layers keep their global ids (``id < num_layers``); MTP layer ``i``
    is exposed as global id ``num_layers + i`` (local index ``id - num_layers``
    in the MTP block's plain ``nn.CellList``). Supports get/set so full-recompute
    can replace a layer cell with its ``checkpoint_wrapper``.
    """

    def __init__(self, decoder_layers, mtp_layers, num_layers):
        self._decoder_layers = decoder_layers
        self._mtp_layers = mtp_layers
        self._num_layers = num_layers

    def __getitem__(self, idx):
        if idx < self._num_layers:
            return self._decoder_layers[idx]
        return self._mtp_layers[idx - self._num_layers]

    def __setitem__(self, idx, cell):
        if idx < self._num_layers:
            self._decoder_layers[idx] = cell
        else:
            self._mtp_layers[idx - self._num_layers] = cell


class _MtpLayerView:
    """Read/write view presenting the decoder block and the MTP block as a single
    global-indexed layer container for the recompute and swap machinery.

    MTP layers are treated as the *last* layers: MTP layer ``i`` is addressable
    as global layer ``num_layers + i`` in the config, so users target them with
    the same layer-id namespace as the decoder.
    """

    def __init__(self, decoder, mtp_block):
        num_layers = decoder.config.num_layers
        self.config = decoder.config
        self.layer_start = decoder.layer_start
        self.layer_end = num_layers + len(mtp_block.layers) - 1
        self.layers = _MtpLayerIndex(decoder.layers, mtp_block.layers, num_layers)


def apply_ac(
    model,
    recompute,
    recompute_comm,
    swap,
    pp,
    mtp_block=None,
):
    """Apply activation checkpointing to the model.

    ``model`` is the decoder ``TransformerBlock``. When ``mtp_block`` is provided
    (DeepSeek-V3 MTP, only on the last PP stage), its layers extend the
    layer-id namespace as the *last* layers: MTP layer ``i`` is addressable as
    global layer ``num_layers + i``.
    """
    global _config_list
    enable_recompute = (recompute.mode != "None" or recompute_comm.enable)
    enable_swap = swap.enable
    if not (enable_recompute or enable_swap):
        return

    backend = None
    if enable_recompute:
        backend = _create_checkpoint_backend(recompute.use_reentrant)
        backend.validate_recompute(recompute, recompute_comm)
        backend.validate_swap(swap)

    num_layers = None
    if hasattr(model, "config"):
        num_layers = model.config.num_layers
    if not num_layers:
        raise ValueError(f"{type(model)} must have 'config.num_layers' attribute.")

    mtp_num_layers = getattr(model.config, "mtp_num_layers", 0) or 0
    total_num_layers = num_layers + mtp_num_layers

    if mtp_block is not None:
        ac_model = _MtpLayerView(model, mtp_block)
    else:
        ac_model = model

    _config_list = _get_modules_and_ops_list(ac_model)

    if pp > 1 and enable_swap:
        logger.error("[Swap Config] swap is not supported with pipeline parallel")
        raise ValueError("swap is not supported with pipeline parallel")

    if enable_recompute:
        _validate_recompute_config(recompute, recompute_comm, total_num_layers)
        apply_recompute(
            ac_model, recompute, recompute_comm, backend=backend)

    if enable_swap:
        _validate_swap_config(swap, total_num_layers)
        input_boundaries = [
            _SwapInputBoundary(
                model,
                range(model.layer_start, model.layer_end + 1),
                exclude_arg_names={"hidden_states"},
            )
        ]
        if mtp_block is not None:
            input_boundaries.append(
                _SwapInputBoundary(mtp_block, range(num_layers, total_num_layers))
            )
        apply_swap(ac_model, swap, input_boundaries=input_boundaries)
