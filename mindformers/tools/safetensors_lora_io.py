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
"""Bounded-memory and atomic safetensors output for LoRA merging."""

import json
import os
import struct
import tempfile
from typing import Optional

import numpy as np
import mindspore as ms
from mindspore import Tensor

from mindformers.checkpoint.safetensors_utils import load_safetensors_file, read_safetensors_header
from mindformers.tools.lora_merge_utils import (
    MOE_ADAPTER_LABEL,
    build_adapter_plan,
    classify_adapter,
    validate_adapter_pair,
)


_STREAM_DTYPE_TO_NP = {
    "BF16": np.uint16,
    "F32": np.float32,
    "F16": np.float16,
}


def atomic_save_checkpoint(save_list, out_file: str) -> None:
    """Write a full-load result beside its destination and commit it atomically."""
    out_dir = os.path.dirname(os.path.abspath(out_file))
    fd, temp_file = tempfile.mkstemp(prefix=".lora-merge-", suffix=".safetensors", dir=out_dir)
    os.close(fd)
    os.remove(temp_file)
    try:
        ms.save_checkpoint(save_list, temp_file, format="safetensors")
        os.replace(temp_file, out_file)
    finally:
        for candidate in (temp_file, temp_file + ".safetensors"):
            if os.path.exists(candidate):
                os.remove(candidate)


def _stream_copy(source, target, start, end, chunk_size=16 * 1024 * 1024):
    """Copy one raw safetensors payload range without materialising it."""
    source.seek(start)
    remaining = end - start
    while remaining:
        block = source.read(min(chunk_size, remaining))
        if not block:
            raise IOError("Unexpected EOF while streaming safetensors payload.")
        target.write(block)
        remaining -= len(block)


def _encode_value(value: np.ndarray, dtype: str) -> bytes:
    """Encode a merged FP32 value back to the source safetensors dtype."""
    if dtype == "BF16":
        return Tensor(value.astype(np.float32), ms.bfloat16).asnumpy().tobytes(order="C")
    np_dtype = _STREAM_DTYPE_TO_NP.get(dtype)
    if np_dtype is None:
        raise ValueError(f"Streaming merge cannot safely rewrite dtype '{dtype}'.")
    return value.astype(np_dtype, copy=False).tobytes(order="C")


def _decode_value(value: np.ndarray, dtype: str) -> np.ndarray:
    """Convert a raw safetensors slice to FP32 without reading adjacent slices."""
    if dtype == "BF16":
        return (value.astype(np.uint32) << 16).view(np.float32)
    return value.astype(np.float32)


def _write_delta(source_file, data_start, meta, target, adapter, chunk_rows=64):
    """Write one merged tensor in bounded row chunks and return its byte count."""
    rank_reduction_weight, rank_expansion_weight, scaling, label, adapter_name = adapter
    dtype = meta["dtype"]
    raw_dtype = _STREAM_DTYPE_TO_NP.get(dtype)
    if raw_dtype is None:
        raise ValueError(f"Streaming merge cannot safely rewrite dtype '{dtype}'.")
    shape = tuple(meta["shape"])
    offset = data_start + int(meta["data_offsets"][0])
    base_weight = np.memmap(
        source_file, mode="r", dtype=raw_dtype, offset=offset, shape=shape, order="C")
    written = 0
    if label == MOE_ADAPTER_LABEL:
        if len(shape) != 3:
            raise ValueError(f"MoE base tensor for '{adapter_name}' must be 3-D, got {shape}.")
        for expert in range(shape[0]):
            for row in range(0, shape[1], chunk_rows):
                end = min(row + chunk_rows, shape[1])
                value = _decode_value(np.asarray(base_weight[expert, row:end, :]), dtype)
                encoded = _encode_value(
                    value + (rank_reduction_weight[expert, row:end, :]
                             @ rank_expansion_weight[expert]) * scaling,
                    dtype,
                )
                target.write(encoded)
                written += len(encoded)
    else:
        if len(shape) != 2:
            raise ValueError(f"Dense base tensor for '{adapter_name}' must be 2-D, got {shape}.")
        for row in range(0, shape[0], chunk_rows):
            end = min(row + chunk_rows, shape[0])
            value = _decode_value(np.asarray(base_weight[row:end, :]), dtype)
            encoded = _encode_value(
                value + (rank_expansion_weight[row:end, :] @ rank_reduction_weight) * scaling,
                dtype,
            )
            target.write(encoded)
            written += len(encoded)
    del base_weight
    return written


def _load_adapter_pair(src_file, header, spec, lora_alpha: int):
    """Load, validate, and prepare exactly one adapter pair for streaming."""
    for name in (spec.rank_reduction_name, spec.rank_expansion_name):
        dtype = header[name].get("dtype")
        if dtype not in _STREAM_DTYPE_TO_NP:
            raise NotImplementedError(
                f"LoRA adapter '{name}' has dtype '{dtype}', which is not supported. "
                "Only F32, F16 and BF16 adapters are supported."
            )

    adapter_values = load_safetensors_file(
        src_file,
        filter_names={spec.rank_reduction_name, spec.rank_expansion_name},
        dequantize=False,
    )
    rank_reduction_weight = adapter_values[spec.rank_reduction_name].astype(np.float32)
    rank_expansion_weight = adapter_values[spec.rank_expansion_name].astype(np.float32)
    rank = validate_adapter_pair(
        rank_reduction_weight,
        rank_expansion_weight,
        spec.label,
        spec.rank_reduction_name,
    )

    base_meta = header[spec.base_name]
    base_shape = tuple(base_meta["shape"])
    if base_meta["dtype"] not in _STREAM_DTYPE_TO_NP:
        raise NotImplementedError(
            f"Merging LoRA into base weight '{spec.base_name}' with dtype "
            f"'{base_meta['dtype']}' is not supported. Only F32, F16 and BF16 are supported."
        )
    if spec.label == MOE_ADAPTER_LABEL:
        delta_shape = (
            rank_reduction_weight.shape[0],
            rank_reduction_weight.shape[1],
            rank_expansion_weight.shape[2],
        )
    else:
        delta_shape = (rank_expansion_weight.shape[0], rank_reduction_weight.shape[1])
    if delta_shape != base_shape:
        if (tuple(reversed(delta_shape[-2:])) == base_shape[-2:]
                and delta_shape[:-2] == base_shape[:-2]):
            return None
        raise ValueError(
            f"Cannot reconcile shapes for adapter '{spec.rank_reduction_name}': "
            f"delta={delta_shape}, base={base_shape}."
        )
    return (
        rank_reduction_weight,
        rank_expansion_weight,
        float(lora_alpha) / rank,
        spec.label,
        spec.rank_reduction_name,
    )


def stream_merge_single_file(src_file: str, dst_file: str, lora_alpha: int) -> Optional[int]:
    """Merge a colocated checkpoint while frozen base tensors remain on disk."""
    if os.path.realpath(os.path.abspath(src_file)) == os.path.realpath(os.path.abspath(dst_file)):
        raise ValueError("src_file and dst_file must be different paths for streaming LoRA merge.")
    if isinstance(lora_alpha, bool) or not isinstance(lora_alpha, int) or lora_alpha <= 0:
        raise ValueError("lora_alpha must be a positive integer.")
    header, data_start = read_safetensors_header(src_file)
    tensor_names = {
        name for name, meta in header.items() if isinstance(meta, dict) and "dtype" in meta
    }
    plan = build_adapter_plan(tensor_names)
    if not plan:
        return None

    specs_by_base = {spec.base_name: spec for spec in plan}

    output_header = {"__metadata__": header["__metadata__"]} if "__metadata__" in header else {}
    ordered_names = [name for name in header if name in tensor_names and classify_adapter(name) is None]
    offset = 0
    for name in ordered_names:
        meta = dict(header[name])
        byte_count = int(meta["data_offsets"][1]) - int(meta["data_offsets"][0])
        meta["data_offsets"] = [offset, offset + byte_count]
        output_header[name] = meta
        offset += byte_count
    encoded_header = json.dumps(output_header, separators=(",", ":")).encode("utf-8")

    with open(src_file, "rb") as source, open(dst_file, "wb") as target:
        target.write(struct.pack("<Q", len(encoded_header)))
        target.write(encoded_header)
        for name in ordered_names:
            meta = header[name]
            start = data_start + int(meta["data_offsets"][0])
            end = data_start + int(meta["data_offsets"][1])
            spec = specs_by_base.get(name)
            if spec is None:
                _stream_copy(source, target, start, end)
                continue
            adapter = _load_adapter_pair(src_file, header, spec, lora_alpha)
            if adapter is None:
                return None
            written = _write_delta(src_file, data_start, meta, target, adapter)
            del adapter
            if written != end - start:
                raise ValueError(f"Encoded size changed for '{name}', refusing invalid safetensors.")
    return len(plan)
