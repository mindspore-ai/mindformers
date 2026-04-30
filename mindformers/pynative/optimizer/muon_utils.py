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
"""Muon tensor layout helpers for pynative mode."""

from fnmatch import fnmatch

from mindspore import mint


def muon_split(tensor, part_a: int, part_b: int):
    """Split periodic row layout ``[A, B] * N`` into two tensors."""
    tensor = tensor.T
    *prefix, _ = tensor.shape
    t = mint.reshape(tensor, tuple(prefix) + (-1, part_a + part_b))
    return (
        mint.reshape(t[..., :part_a], tuple(prefix) + (-1,)).T,
        mint.reshape(t[..., part_a:], tuple(prefix) + (-1,)).T,
    )


def muon_merge(tensor_a, tensor_b, part_a: int, part_b: int):
    """Inverse of ``muon_split``."""
    tensor_a = tensor_a.T
    tensor_b = tensor_b.T
    *prefix, _ = tensor_a.shape
    a = mint.reshape(tensor_a, tuple(prefix) + (-1, part_a))
    b = mint.reshape(tensor_b, tuple(prefix) + (-1, part_b))
    return mint.reshape(mint.cat([a, b], dim=-1), tuple(prefix) + (-1,)).T


def _eval_tuple(spec, name, tensor):
    """Evaluate a schema tuple or tuple factory."""
    return spec(name, tensor) if callable(spec) else spec


_IDENTITY_RULE = {"kind": None}


def _split_periodic(param_name, rule, tensor):
    """Split a periodic row layout."""
    part_a, part_b = _eval_tuple(rule["parts"], param_name, tensor)[:2]
    return list(muon_split(tensor, part_a, part_b))


def _merge_periodic(param_name, rule, parts_list):
    """Merge a periodic row layout."""
    part_a, part_b = _eval_tuple(rule["parts"], param_name, parts_list[0])[:2]
    return muon_merge(parts_list[0], parts_list[1], part_a, part_b)


def _split_reshape_concat(param_name, rule, tensor):
    """Reshape a concat expert weight and split the last dimension."""
    _, hidden_size, total_intermediate = _eval_tuple(rule["reshape"], param_name, tensor)
    t3 = mint.reshape(tensor, (-1, hidden_size, total_intermediate))
    half_intermediate = total_intermediate // 2
    return [t3[..., :half_intermediate], t3[..., half_intermediate:]]


def _merge_reshape_concat(param_name, rule, parts_list):
    """Merge a split concat expert weight and restore it to 2D."""
    _, _, total_intermediate = _eval_tuple(rule["reshape"], param_name, parts_list[0])
    return mint.reshape(mint.cat(parts_list, dim=-1), (-1, total_intermediate))


def _split_reshape_only(param_name, rule, tensor):
    """Reshape a weight to 3D for Muon."""
    _, intermediate_size, hidden_size = _eval_tuple(rule["reshape"], param_name, tensor)
    return [mint.reshape(tensor, (-1, intermediate_size, hidden_size))]


def _merge_reshape_only(param_name, rule, parts_list):
    """Restore a reshape-only Muon block to 2D."""
    _, _, hidden_size = _eval_tuple(rule["reshape"], param_name, parts_list[0])
    return mint.reshape(parts_list[0], (-1, hidden_size))


def _split_block_split(param_name, rule, tensor):
    """Split a tensor by plain and periodic row blocks."""
    parts = []
    offset = 0
    for block in _eval_tuple(rule["blocks"], param_name, tensor):
        if isinstance(block, int):
            parts.append(tensor[offset:offset + block])
            offset += block
            continue
        part_a, part_b, num_blocks = block
        size = (part_a + part_b) * num_blocks
        parts.extend(muon_split(tensor[offset:offset + size], part_a, part_b))
        offset += size
    return parts


def _merge_block_split(param_name, rule, parts_list):
    """Merge plain and periodic row blocks."""
    merged_blocks = []
    part_idx = 0
    for block in _eval_tuple(rule["blocks"], param_name, parts_list[0]):
        if isinstance(block, int):
            merged_blocks.append(parts_list[part_idx])
            part_idx += 1
            continue
        part_a, part_b, _ = block
        merged_blocks.append(
            muon_merge(parts_list[part_idx], parts_list[part_idx + 1], part_a, part_b)
        )
        part_idx += 2
    return mint.cat(merged_blocks, dim=0)


_SPLIT_HANDLERS = {
    "periodic": _split_periodic,
    "reshape_concat": _split_reshape_concat,
    "reshape_only": _split_reshape_only,
    "block_split": _split_block_split,
}
_MERGE_HANDLERS = {
    "periodic": _merge_periodic,
    "reshape_concat": _merge_reshape_concat,
    "reshape_only": _merge_reshape_only,
    "block_split": _merge_block_split,
}


def make_muon_fns(schema):
    """
    Build split/merge functions from a model-specific schema.

    Supported rule kinds:
      - periodic: split ``[A, B] * num_blocks`` rows.
      - reshape_concat: reshape to 3D and split the last dimension in half.
      - reshape_only: reshape to 3D for Newton-Schulz, then restore to 2D.
      - block_split: slice dim 0 into plain slabs and periodic slabs.
    """

    def _match_rule(param_name):
        for rule in schema:
            if any(fnmatch(param_name, pat) for pat in rule["patterns"]):
                return rule
        return _IDENTITY_RULE

    handler_map = {}

    def _get_handlers(param_name):
        if param_name not in handler_map:
            rule = _match_rule(param_name)
            handler_map[param_name] = (rule, _SPLIT_HANDLERS.get(rule["kind"]), _MERGE_HANDLERS.get(rule["kind"]))
        return handler_map[param_name]

    def split_fn(param_name, tensor):
        rule, split_handler, _ = _get_handlers(param_name)
        return [tensor] if split_handler is None else split_handler(param_name, rule, tensor)

    def merge_fn(param_name, parts_list):
        rule, _, merge_handler = _get_handlers(param_name)
        return parts_list[0] if merge_handler is None else merge_handler(param_name, rule, parts_list)

    return split_fn, merge_fn
