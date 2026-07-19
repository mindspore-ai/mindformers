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
"""Safetensors utilities for loading checkpoint files with unsupported dtypes (BF16, F8_E8M0, etc.).

MindSpore's built-in ``ms_load_checkpoint`` does not handle float8 formats (F8_E8M0, F8_E4M3, F8_E5M2)
or bfloat16 in all cases. This module provides drop-in replacements that parse the safetensors
header and tensor data directly, converting unsupported dtypes on the fly.
"""

import json
import os
import struct
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Safetensors dtype → numpy dtype mapping
# ---------------------------------------------------------------------------
_SAFETENSOR_DTYPE_TO_NP: Dict[str, np.dtype] = {
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,          # numpy has no native bf16; read as uint16 and convert later
    "F8_E8M0": np.uint8,        # float8 formats → read as raw uint8, convert later
    "F8_E4M3": np.uint8,
    "F8_E5M2": np.uint8,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}

# Dtypes that do NOT have a one-to-one numpy equivalent and need post-processing.
_SPECIAL_DTYPES: Set[str] = {"BF16", "F8_E8M0", "F8_E4M3", "F8_E5M2"}

# Safetensors dtype → MindSpore-compatible numpy dtype (for metadata / shape-only purposes).
# For actual tensor loading, use load_safetensors_file which converts to float32.
_SAFETENSOR_DTYPE_TO_NP_META: Dict[str, np.dtype] = {
    **{k: v for k, v in _SAFETENSOR_DTYPE_TO_NP.items() if k not in _SPECIAL_DTYPES},
    "BF16": np.float32,        # report as float32 for metadata (same size: 32 bits → 2 bytes)
    "F8_E8M0": np.float32,     # report as float32 for metadata
    "F8_E4M3": np.float32,
    "F8_E5M2": np.float32,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_safetensors_header(file_path: str) -> Tuple[Dict[str, dict], int]:
    """Read the JSON header of a safetensors file without loading any tensor data.

    The safetensors format is:
        - 8 bytes: uint64 (little-endian) header size
        - N bytes: JSON header
        - remaining bytes: tensor data

    Returns:
        A tuple of ``(header_dict, data_start_offset)`` where:

        *header_dict*: mapping each tensor name to its metadata::

            {"param_name": {"dtype": "F32", "shape": [1024, 512], "data_offsets": [0, 2097152]}}

        *data_start_offset*: byte offset where tensor data begins (``8 + header_size``).

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        ValueError: If the file is not a valid safetensors file.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Safetensors file not found: {file_path}")

    with open(file_path, "rb") as f:
        header_bytes = f.read(8)
        if len(header_bytes) < 8:
            raise ValueError(f"File too small to be a valid safetensors file: {file_path}")
        header_size = struct.unpack("<Q", header_bytes)[0]
        header_str = f.read(header_size).decode("utf-8")

    try:
        header = json.loads(header_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse safetensors JSON header in {file_path}: {e}") from e

    return header, 8 + header_size


# Dtypes that are quantized and need dequantization with a scale factor.
_QUANTIZED_DTYPES: Set[str] = {"I8", "F8_E4M3", "F8_E5M2"}

# E2M1 finite-only FP4 values in checkpoint code-point order. DeepSeek-V4
# routed-expert weights store two of these values in every I8 byte.
_FP4_E2M1_LUT = np.asarray(
    (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0),
    dtype=np.float32,
)


# Threshold (bytes) below which adjacent byte ranges are merged into a single
# read to reduce syscall count.  1 MiB is chosen because a single extra disk
# read typically costs more than reading an additional 1 MiB sequentially.
_CONTIGUOUS_MERGE_THRESHOLD: int = 1024 * 1024


def load_safetensors_file(
    file_path: str,
    filter_names: Optional[Set[str]] = None,
    dequantize: bool = True,
) -> Dict[str, np.ndarray]:
    """Load selected tensors from a single safetensors file.

    Unsupported dtypes (BF16, F8_E8M0, …) are automatically converted to float32.

    When *dequantize* is ``True`` (the default), quantized weight tensors (I8,
    F8_E4M3, F8_E5M2) are paired with their ``.scale`` tensors and dequantized
    on the fly::

        dequantized = decode(weight) * decode(scale)   # block-wise broadcast

    The scale tensor is consumed during dequantization and **not** returned in
    the result dict.

    **Lazy / sparse loading:** Only the byte ranges for tensors listed in
    *filter_names* (plus any required scale tensors) are read from disk.
    Adjacent ranges whose gap is below *CONTIGUOUS_MERGE_THRESHOLD* (1 MiB)
    are merged into a single ``f.read()`` to reduce syscall overhead while
    still avoiding bulk I/O for tensors that are not needed.

    Args:
        file_path: Path to the safetensors file.
        filter_names: If set, only load tensors whose names are in this set.
            If ``None``, load all tensors in the file.
        dequantize: Whether to automatically dequantize I8/F8 weights with
            their scale factors.

    Returns:
        Dict mapping tensor name → numpy array.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        ValueError: If an unknown dtype is encountered.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Safetensors file not found: {file_path}")

    # --- Open once; read header and tensor data in the same context ---
    with open(file_path, "rb") as f:
        header_bytes = f.read(8)
        if len(header_bytes) < 8:
            raise ValueError(f"File too small to be a valid safetensors file: {file_path}")
        header_size = struct.unpack("<Q", header_bytes)[0]
        header_str = f.read(header_size).decode("utf-8")
        try:
            header = json.loads(header_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse safetensors JSON header in {file_path}: {e}") from e

        data_start = 8 + header_size

        # --- Pre-scan: build scale lookup and collect tensors to load ---
        weight_to_scale: Dict[str, str] = {}
        to_load_names: List[str] = []

        for name, info in header.items():
            # Skip metadata entries (e.g., __metadata__) that don't have tensor fields
            if not isinstance(info, dict) or "dtype" not in info:
                continue
            st_dtype = info["dtype"]

            # Track quantized weights that have a corresponding scale tensor
            if dequantize and name.endswith(".weight") and st_dtype in _QUANTIZED_DTYPES:
                scale_name = name[:-len(".weight")] + ".scale"
                if scale_name in header:
                    weight_to_scale[name] = scale_name

            # Apply filter
            if filter_names is not None and name not in filter_names:
                continue

            # Scale tensors are consumed during dequantization — don't return them directly
            if dequantize and name in weight_to_scale.values():
                continue

            to_load_names.append(name)

        if not to_load_names:
            return {}

        # --- Build contiguous byte ranges from needed-tensor offsets ---
        # Collect every byte range we must read: requested tensors + their scale tensors.
        seen_offsets: Set[Tuple[int, int]] = set()
        range_entries: List[Tuple[int, int, List[str]]] = []  # (start, end, [names])

        def _add_range(name):
            offs = tuple(header[name]["data_offsets"])
            if offs not in seen_offsets:
                seen_offsets.add(offs)
                range_entries.append((offs[0], offs[1], [name]))

        for name in to_load_names:
            _add_range(name)
            if name in weight_to_scale:
                _add_range(weight_to_scale[name])

        # Sort by start offset (safetensors stores them in this order already,
        # but sort explicitly for correctness).
        range_entries.sort(key=lambda x: x[0])

        # Merge adjacent / nearby ranges.
        merged: List[Tuple[int, int, List[str]]] = []
        for start, end, names in range_entries:
            if merged and start <= merged[-1][1] + _CONTIGUOUS_MERGE_THRESHOLD:
                prev = merged[-1]
                merged[-1] = (prev[0], max(prev[1], end), prev[2] + names)
            else:
                merged.append((start, end, names))

        # --- Read each merged range and build a name→buffer lookup ---
        tensor_buf_map: Dict[str, Tuple[int, bytes]] = {}  # name → (buf_start, buf_bytes)

        for range_start, range_end, range_names in merged:
            f.seek(data_start + range_start)
            buf = f.read(range_end - range_start)
            for n in range_names:
                tensor_buf_map[n] = (range_start, buf)

    # --- Inline helper: extract a tensor from the lazy-read buffers ---
    def _extract(name):
        """Look up the pre-read buffer for *name* and decode the tensor."""
        info = header[name]
        offsets = info["data_offsets"]
        st_dtype = info["dtype"]
        shape = info["shape"]

        np_dtype = _SAFETENSOR_DTYPE_TO_NP.get(st_dtype)
        if np_dtype is None:
            raise ValueError(
                f"Unsupported safetensors dtype '{st_dtype}' for tensor '{name}' "
                f"in file {file_path}. Supported dtypes: {list(_SAFETENSOR_DTYPE_TO_NP.keys())}"
            )

        buf_start, buf = tensor_buf_map[name]
        raw = buf[offsets[0] - buf_start:offsets[1] - buf_start]
        tensor = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
        return tensor, st_dtype

    # --- Load and post-process each tensor ---
    result: Dict[str, np.ndarray] = {}

    for name in to_load_names:
        info = header[name]
        st_dtype = info["dtype"]

        if name in weight_to_scale:
            # === Dequantization path: weight + scale ===
            scale_name = weight_to_scale[name]
            if st_dtype == "I8":
                # Packed E2M1 FP4 → unpack first, then dequantize.
                weight, _ = _extract(name)
                weight = _unpack_fp4_e2m1(weight)
            else:
                weight, w_dtype = _extract(name)
                if w_dtype in _SPECIAL_DTYPES:
                    weight = _convert_special_dtype(weight, w_dtype)
                elif w_dtype in _QUANTIZED_DTYPES:
                    weight = weight.astype(np.float32, copy=False)

            scale, s_dtype = _extract(scale_name)
            if s_dtype in _SPECIAL_DTYPES:
                scale = _convert_special_dtype(scale, s_dtype)
            elif s_dtype in _QUANTIZED_DTYPES:
                scale = scale.astype(np.float32, copy=False)

            result[name] = _dequantize_blockwise(weight, scale)

        elif st_dtype in _SPECIAL_DTYPES:
            tensor, _ = _extract(name)
            result[name] = _convert_special_dtype(tensor, st_dtype)

        elif st_dtype in _QUANTIZED_DTYPES and dequantize:
            tensor, _ = _extract(name)
            result[name] = tensor.astype(np.float32, copy=False)

        else:
            tensor, _ = _extract(name)
            result[name] = tensor

    return result


def _unpack_fp4_e2m1(weight: np.ndarray) -> np.ndarray:
    """Unpack I8 storage containing two finite E2M1 FP4 values per byte.

    The low nibble is the first value and the high nibble is the second value,
    matching the checkpoint producer and HuggingFace's FP4 dequantizer. Nibble
    values index :data:`_FP4_E2M1_LUT`; they are not two's-complement integers.

    For an input ending in dimension ``N``, the output ends in ``2 * N`` with
    columns ``2j`` and ``2j+1`` carrying the low and high nibbles respectively.

    Args:
        weight: int8 numpy array containing packed E2M1 FP4 data.

    Returns:
        float32 numpy array with the final dimension doubled.
    """
    w_u8 = weight.view(np.uint8) if weight.dtype == np.int8 else np.asarray(weight, dtype=np.uint8)
    low = _FP4_E2M1_LUT[w_u8 & 0xF]
    high = _FP4_E2M1_LUT[(w_u8 >> 4) & 0xF]
    return np.stack((low, high), axis=-1).reshape(*weight.shape[:-1], 2 * weight.shape[-1])


def _dequantize_blockwise(weight: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Dequantize a block-wise quantized weight tensor.

    The scale factor is broadcast from per-block granularity to element-wise::

        dequantized[i, j] = weight[i, j] * scale[i // block_M, j // block_N]

    where ``block_M = weight.shape[0] // scale.shape[0]`` and
    ``block_N = weight.shape[1] // scale.shape[1]``.

    Uses 4-D reshape + NumPy broadcasting (no ``np.repeat`` materialisation)
    to keep memory overhead independent of the weight size.

    Args:
        weight: Decoded float32 weight tensor of shape ``[M, N]``.
        scale: Decoded float32 scale tensor of shape ``[M_blk, N_blk]``.

    Returns:
        Dequantized float32 tensor of shape ``[M, N]``.
    """
    m, n = weight.shape
    m_blk, n_blk = scale.shape
    block_m = m // m_blk
    block_n = n // n_blk

    w_f32 = weight.astype(np.float32, copy=False)
    s_f32 = scale.astype(np.float32, copy=False)

    # Reshape to 4-D: (M_blk, block_M, N_blk, block_N)
    w_reshaped = w_f32.reshape(m_blk, block_m, n_blk, block_n)
    # Broadcast scale along the block axes: (M_blk, 1, N_blk, 1)
    # NumPy broadcasts without materialising a full copy of the scale.
    s_reshaped = s_f32[:, np.newaxis, :, np.newaxis]

    return (w_reshaped * s_reshaped).reshape(m, n)


def safetensor_dtype_to_np_meta(dtype_str: str) -> np.dtype:
    """Map a safetensors dtype string to a MindSpore-compatible numpy dtype for metadata.

    Unsupported float types (BF16, F8_*) are mapped to float32 so that downstream
    shape/size calculations work correctly.
    """
    np_dtype = _SAFETENSOR_DTYPE_TO_NP_META.get(dtype_str)
    if np_dtype is None:
        raise ValueError(
            f"Unknown safetensors dtype '{dtype_str}'. "
            f"Known dtypes: {list(_SAFETENSOR_DTYPE_TO_NP_META.keys())}"
        )
    return np_dtype


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _convert_special_dtype(tensor: np.ndarray, dtype_str: str) -> np.ndarray:
    """Convert a raw-integer tensor (uint8/uint16) to float32 based on dtype semantics.

    - BF16: reinterpret uint16 as bfloat16, upcast to float32.
    - F8_E8M0: 2^(value - 127).  value=0 → 0.0.
    - F8_E4M3: IEEE 754-like float8 (1s 4e 3m, bias=7).
    - F8_E5M2: IEEE 754-like float8 (1s 5e 2m, bias=15).
    - I8/I32/I16: kept as-is (integer quantized weights; dequantization is handled
      separately when scale factors are available).
    """
    if dtype_str == "BF16":
        # BF16 stored as uint16:
        #   float32 bits = (uint16_value << 16)
        tensor_u32 = tensor.astype(np.uint32) << 16
        return tensor_u32.view(np.float32).copy()

    if dtype_str == "F8_E8M0":
        # F8_E8M0 represents powers of 2:  val=0 → 0.0,  val=1..254 → 2^(val-127).
        # Use bit manipulation for perfect accuracy and speed:
        #   float32 bit pattern = (val as uint32) << 23   (sign=0, exp=val, mantissa=0)
        # Special cases: val=0 stays 0.0, val=255 → +Inf.
        tensor_u32 = tensor.astype(np.uint32) << 23
        result = tensor_u32.view(np.float32).copy()
        result[tensor == 0] = 0.0
        return result

    if dtype_str == "F8_E4M3":
        # 1 sign | 4 exp (bias=7) | 3 mantissa
        return _decode_fp8_e4m3(tensor.ravel()).reshape(tensor.shape)

    if dtype_str == "F8_E5M2":
        # 1 sign | 5 exp (bias=15) | 2 mantissa
        return _decode_fp8_e5m2(tensor.ravel()).reshape(tensor.shape)

    # Unknown special dtype — fall back to float32 cast
    return tensor.astype(np.float32)


def _decode_fp8_e4m3(data: np.ndarray) -> np.ndarray:
    """Decode an array of uint8 E4M3 values to float32.

    E4M3 layout (MSB→LSB): s e3 e2 e1 e0 m2 m1 m0
    Normal:   (-1)^s × 2^(exp - 7) × (1 + mantissa / 8)
    Subnormal (exp=0000): (-1)^s × 2^(-6) × (0 + mantissa / 8)
    """
    data = np.asarray(data, dtype=np.uint8)
    sign = (data >> 7).astype(np.int8)
    exp = (data >> 3) & 0x0F
    mant = data & 0x07

    result = np.zeros(len(data), dtype=np.float32)
    normal = exp > 0
    subnormal = exp == 0

    # Normal numbers
    result[normal] = (1.0 + mant[normal].astype(np.float32) / 8.0) * \
        np.power(2.0, exp[normal].astype(np.float32) - 7.0)

    # Subnormal numbers
    result[subnormal] = (mant[subnormal].astype(np.float32) / 8.0) * \
        np.power(2.0, -6.0)

    # Apply sign
    neg = sign != 0
    result[neg] = -result[neg]

    return result


def _decode_fp8_e5m2(data: np.ndarray) -> np.ndarray:
    """Decode an array of uint8 E5M2 values to float32.

    E5M2 layout (MSB→LSB): s e4 e3 e2 e1 e0 m1 m0
    Normal:   (-1)^s × 2^(exp - 15) × (1 + mantissa / 4)
    Subnormal (exp=00000): (-1)^s × 2^(-14) × (0 + mantissa / 4)
    """
    data = np.asarray(data, dtype=np.uint8)
    sign = (data >> 7).astype(np.int8)
    exp = (data >> 2) & 0x1F
    mant = data & 0x03

    result = np.zeros(len(data), dtype=np.float32)
    normal = exp > 0
    subnormal = exp == 0

    result[normal] = (1.0 + mant[normal].astype(np.float32) / 4.0) * \
        np.power(2.0, exp[normal].astype(np.float32) - 15.0)

    result[subnormal] = (mant[subnormal].astype(np.float32) / 4.0) * \
        np.power(2.0, -14.0)

    neg = sign != 0
    result[neg] = -result[neg]

    return result
