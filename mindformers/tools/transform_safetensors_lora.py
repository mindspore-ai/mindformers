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
"""Merge LoRA adapter weights into the base model."""
import os
import re
import json
import glob
import argparse
from typing import Optional, List, Tuple, Dict

import numpy as np
import mindspore as ms
from mindspore import DeviceCtx, Tensor

from mindformers.tools.logger import logger
from mindformers.checkpoint.safetensors_utils import (
    load_safetensors_file,
    read_safetensors_header,
)
from mindformers.checkpoint.checkpoint import get_checkpoint_path
from mindformers.checkpoint.metadata import get_metadata_of_checkpoint
from mindformers.checkpoint.reshard import ReshardLoader
from mindformers.checkpoint.sharded_tensor import build_sharded_tensor

# Safetensors dtype string -> MindSpore dtype (preserves each tensor's original
# storage dtype on save).
_SAFETENSOR_DTYPE_TO_MS = {
    "F64": ms.float64,
    "F32": ms.float32,
    "F16": ms.float16,
    "BF16": ms.bfloat16,
    "I64": ms.int64,
    "I32": ms.int32,
    "I16": ms.int16,
    "I8": ms.int8,
    "U8": ms.uint8,
    "U16": ms.uint16,
    "U32": ms.uint32,
    "U64": ms.uint64,
    "BOOL": ms.bool_,
}


# Adapter naming conventions, most-specific suffix first. (A-suffix, B-suffix, label).
_ADAPTER_CONVENTIONS: List[Tuple[str, str, str]] = [
    (".lora_A.weight", ".lora_B.weight", "parallel_core"),
    (".mindpet_delta_lora_a", ".mindpet_delta_lora_b", "mindpet"),
    (".lora_a", ".lora_b", "pynative"),
]

# metadata.json ``storage_data`` keys are stringified tuples, e.g. ``('param.name', (0,))``.
_STORAGE_KEY_PATTERN = re.compile(r"^\('(.+)', \([0-9, ]*\)\)$")


def _ms_dtype_of(dtype_str: str):
    """Map a safetensors dtype string to a MindSpore dtype (float32 fallback)."""
    dtype = _SAFETENSOR_DTYPE_TO_MS.get(dtype_str)
    if dtype is None:
        logger.warning("Unknown safetensors dtype '%s'; falling back to float32.", dtype_str)
        return ms.float32
    return dtype


def _is_optimizer_shard(file_name: str) -> bool:
    """True for optimizer-state shard files (``*-opt-*`` / ``*-optimizer-*``)."""
    base = os.path.basename(file_name)
    return "-opt-" in base or "-optimizer-" in base


def _is_optimizer_param(name: str) -> bool:
    """Filter stray optimizer state keys (adam_m/adam_v) if present in a shard."""
    return name.startswith("adam_m") or name.startswith("adam_v") or ".opt." in name


def _classify_adapter(name: str) -> Optional[Tuple[str, str, Tuple[str, str, str]]]:
    """Classify a parameter name as a LoRA adapter.

    Returns ``(prefix, kind, convention)`` with ``kind`` being ``'A'`` or ``'B'``
    and ``prefix`` the base-weight prefix, or ``None`` if not an adapter.
    """
    for convention in _ADAPTER_CONVENTIONS:
        a_suffix, b_suffix, _ = convention
        if name.endswith(a_suffix):
            return name[:-len(a_suffix)], "A", convention
        if name.endswith(b_suffix):
            return name[:-len(b_suffix)], "B", convention
    return None


def _storage_key_param(key) -> Optional[str]:
    """Extract the parameter name from a metadata.json ``storage_data`` key."""
    if isinstance(key, tuple):
        return key[0]
    match = _STORAGE_KEY_PATTERN.match(key)
    return match.group(1) if match else None


def _find_model_safetensors(ckpt_dir: str) -> List[str]:
    """List model safetensors files of a single-rank directory (optimizer shards excluded)."""
    files = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))
    files = [f for f in files if not _is_optimizer_shard(f)]
    if not files:
        raise FileNotFoundError(
            f"No model safetensors files found under '{ckpt_dir}'. "
            f"Expected one or more '*.safetensors' (excluding optimizer shards)."
        )
    return files


def _read_adapter_config(src_path: str) -> Optional[dict]:
    """Read a sibling ``adapter_config.json`` (up to 3 ancestor dirs), if any."""
    candidates = []
    if os.path.isfile(src_path):
        candidates.append(os.path.dirname(src_path))
    elif os.path.isdir(src_path):
        candidates.append(src_path)
    # Walk up to three ancestor directories so a config saved at the run output
    # root is found when shards live deeper under it.
    d = candidates[0] if candidates else ""
    for _ in range(3):
        d = os.path.dirname(d)
        if d:
            candidates.append(d)
    for d in candidates:
        if not d:
            continue
        p = os.path.join(d, "adapter_config.json")
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Found adapter_config.json at '%s' but failed to read: %s", p, e)
                return None
    return None


def _load_plain_params(files: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Load single-rank safetensors shards into one dict, recording dtypes."""
    params: Dict[str, np.ndarray] = {}
    dtypes: Dict[str, object] = {}
    for fpath in files:
        header, _ = read_safetensors_header(fpath)
        for name, meta in header.items():
            if name != "__metadata__":
                dtypes[name] = _ms_dtype_of(meta.get("dtype"))
        for name, arr in load_safetensors_file(fpath, dequantize=False).items():
            if _is_optimizer_param(name):
                continue
            if name in params:
                raise ValueError(
                    f"Parameter '{name}' appears in more than one model shard "
                    f"({files}). Duplicate names across shards indicate a "
                    f"multi-rank distributed checkpoint without 'metadata.json'; "
                    f"merge it from the trainer-saved iteration directory "
                    f"(which contains metadata.json) instead."
                )
            params[name] = arr
    return params, dtypes


def _load_replicated_rank_params(ckpt_dir: str):
    """Load weights from a legacy static-graph multi-rank save (``rank_*/`` dirs)."""
    rank_dirs = sorted(d for d in glob.glob(os.path.join(ckpt_dir, "rank_*")) if os.path.isdir(d))
    if not rank_dirs:
        return None
    ref_dir = rank_dirs[0]
    ref_params, dtypes = _load_plain_params(_find_model_safetensors(ref_dir))
    for rank_dir in rank_dirs[1:]:
        params, _ = _load_plain_params(_find_model_safetensors(rank_dir))
        if set(params) != set(ref_params):
            raise ValueError(
                f"Rank directories under '{ckpt_dir}' hold different parameter "
                f"sets ('{ref_dir}' vs '{rank_dir}'), i.e. this is a sharded "
                f"(e.g. pipeline-parallel) checkpoint. Merge it from a single "
                f"rank directory (e.g. '{ref_dir}') instead."
            )
        mismatched = [
            name for name in ref_params
            if params[name].shape != ref_params[name].shape
            or not np.array_equal(params[name], ref_params[name])
        ]
        if mismatched:
            raise ValueError(
                f"Rank directories under '{ckpt_dir}' hold different values for "
                f"{len(mismatched)} parameters (e.g. '{mismatched[0]}'), i.e. "
                f"this is a sharded (tensor/expert-parallel) checkpoint that "
                f"cannot be reassembled without a strategy file. Merge it from "
                f"a single rank directory (e.g. '{ref_dir}') or re-save with "
                f"'integrated_save' enabled instead."
            )
    logger.info("Found %d rank directories with identical weights under '%s'; using '%s'.",
                len(rank_dirs), ckpt_dir, ref_dir)
    return ref_params, dtypes, ref_dir


def _load_distributed_params(ckpt_dir: str) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Reassemble full weights from a multi-rank checkpoint dir via ``ReshardLoader``."""
    src_metas, file_mappings = get_metadata_of_checkpoint(ckpt_dir)

    # Keep model shards only (optimizer shards are dropped).
    file_mappings = {
        key: [entry for entry in entries if not _is_optimizer_shard(entry["file_name"])]
        for key, entries in file_mappings.items()
    }
    file_mappings = {key: entries for key, entries in file_mappings.items() if entries}
    kept_names = {name for name in (_storage_key_param(k) for k in file_mappings) if name}
    src_metas = {name: chunks for name, chunks in src_metas.items() if name in kept_names}

    # Destination layout: every parameter as one full tensor.
    dst_metas = {}
    dtypes = {}
    for name, chunks in src_metas.items():
        ref = chunks[0]
        global_shape = tuple(ref.global_shape)
        dtypes[name] = ref.dtype
        dst_metas[name] = build_sharded_tensor(
            param_name=name, param_dtype=ref.dtype,
            local_shape=global_shape, global_shape=global_shape,
            global_offset=(0,) * len(global_shape),
            axis_fragmentations=(1,) * len(global_shape))

    with DeviceCtx("CPU"):
        loader = ReshardLoader(
            checkpoint_dir=ckpt_dir,
            dst_sharded_tensor_metas=dst_metas,
            src_sharded_tensor_metas=src_metas,
            param_file_mappings=file_mappings,
        )
        state_dict = loader.load()

    params = {name: param.asnumpy() for name, param in state_dict.items()}
    logger.info("Reassembled %d full parameters from distributed shards in '%s'.",
                len(params), ckpt_dir)
    return params, dtypes


def _load_checkpoint_params(src_path: str):
    """Load model weights from any supported checkpoint form."""
    if os.path.isfile(src_path):
        params, dtypes = _load_plain_params([src_path])
        return params, dtypes, [src_path]
    if not os.path.isdir(src_path):
        raise FileNotFoundError(f"src_ckpt_path does not exist: {src_path}")

    # Legacy static-graph multi-rank saves are handled before tracker-based
    # resolution: they contain no direct *.safetensors files.
    for probe in (src_path, os.path.join(src_path, "checkpoint")):
        replicated = _load_replicated_rank_params(probe)
        if replicated is not None:
            params, dtypes, rank_dir = replicated
            return params, dtypes, [rank_dir]
    # A checkpoint save root is resolved to its latest iteration directory.
    ckpt_dir = get_checkpoint_path(src_path)
    if os.path.isfile(os.path.join(ckpt_dir, "metadata.json")):
        # Multi-rank distributed checkpoint: reassemble shards to full weights.
        params, dtypes = _load_distributed_params(ckpt_dir)
        return params, dtypes, [ckpt_dir]
    params, dtypes = _load_plain_params(_find_model_safetensors(ckpt_dir))
    return params, dtypes, [ckpt_dir]


def merge_lora(src_ckpt_path: str,
               dst_ckpt_path: str,
               lora_alpha: Optional[int] = None) -> str:
    """Merge LoRA adapters into base weights and save a clean checkpoint."""
    params, dtypes, sources = _load_checkpoint_params(src_ckpt_path)
    logger.info("Loaded %d parameters from: %s", len(params), sources)

    # Resolve alpha: explicit flag -> adapter_config.json -> default 16.
    if lora_alpha is None:
        cfg = _read_adapter_config(src_ckpt_path)
        alpha_in_cfg = cfg.get("lora_alpha") if cfg else None
        if alpha_in_cfg is not None:
            lora_alpha = int(alpha_in_cfg)
            logger.info("Read lora_alpha=%d from adapter_config.json.", lora_alpha)
    if lora_alpha is None:
        lora_alpha = 16
        logger.info("No lora_alpha given or in adapter_config.json; using default %d.", lora_alpha)

    # Collect every A-adapter with its paired B-adapter and base weight keys.
    a_adapters = []  # (a_key, b_key, base_key, label)
    label_counts = {}
    for name in params:
        cls = _classify_adapter(name)
        if cls is None or cls[1] != "A":
            continue
        prefix, _, (_, b_suffix, label) = cls
        a_adapters.append((name, prefix + b_suffix, prefix + ".weight", label))
        label_counts[label] = label_counts.get(label, 0) + 1
    if not a_adapters:
        raise ValueError(
            f"No LoRA adapter parameters found in '{src_ckpt_path}'. Expected one "
            f"of the supported naming conventions: <base>.lora_A.weight / "
            f"<base>.mindpet_delta_lora_a / <base>.lora_a (and the matching B)."
        )
    logger.info("Found %d LoRA adapter(s): %s", len(a_adapters), label_counts)

    merged_count = 0
    for a_key, b_key, base_key, label in a_adapters:
        if base_key not in params:
            raise ValueError(
                f"Cannot locate the base weight for adapter '{a_key}': expected "
                f"'{base_key}'. The checkpoint may be an adapter-only export "
                f"(missing frozen base weights)."
            )
        if b_key not in params:
            raise ValueError(
                f"Missing paired B-adapter for '{a_key}': expected '{b_key}' "
                f"(convention '{label}')."
            )

        # Matmul in float32 for numerical stability; cast back on save.
        a_mat = params[a_key].astype(np.float32)      # (r, in)
        b_mat = params[b_key].astype(np.float32)      # (out, r)
        w_mat = params[base_key].astype(np.float32)   # (out, in)
        r = int(a_mat.shape[0])
        scaling = float(lora_alpha) / float(r)
        delta = scaling * (b_mat @ a_mat)  # (out, in)

        # Shape safety net: fall back to delta.T when the base weight is stored
        # transposed (a transpose_b=False convention).
        if delta.shape != w_mat.shape:
            delta_t = delta.T
            if delta_t.shape == w_mat.shape:
                logger.warning(
                    "Adapter '%s': (B@A) shape %s != base '%s' shape %s; the base "
                    "weight is stored transposed — using (B@A).T.", a_key, delta.shape,
                    base_key, w_mat.shape,
                )
                delta = delta_t
            else:
                raise ValueError(
                    f"Cannot reconcile shapes for adapter '{a_key}': "
                    f"(B@A)={tuple(delta.shape)}, (B@A).T={tuple(delta_t.shape)}, "
                    f"base '{base_key}'={tuple(w_mat.shape)}."
                )

        params[base_key] = w_mat + delta
        merged_count += 1
        logger.info(
            "Merged '%s' [%s]: r=%d, alpha=%d, scaling=%.6f -> '%s'.",
            a_key, label, r, lora_alpha, scaling, base_key,
        )

    # Drop every LoRA adapter parameter; keep all other (base) params as-is.
    params = {name: value for name, value in params.items() if _classify_adapter(name) is None}

    # Cast back to the original storage dtype to reproduce the base layout.
    save_list = [
        {"name": name, "data": Tensor(value, dtypes.get(name, ms.float32))}
        for name, value in params.items()
    ]

    # ms.save_checkpoint appends '.safetensors' when the name lacks it.
    out_base = dst_ckpt_path[:-len(".safetensors")] if dst_ckpt_path.endswith(".safetensors") \
        else dst_ckpt_path
    out_dir = os.path.dirname(os.path.abspath(out_base))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    ms.save_checkpoint(save_list, out_base, format="safetensors")
    out_file = out_base + ".safetensors"

    logger.info(
        "LoRA merge succeeded: %d adapters folded, %d parameters saved to '%s'.",
        merged_count, len(save_list), out_file,
    )
    return out_file


def _parse_args():
    """Parse command-line arguments (see the module docstring for usage)."""
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter weights into the base model. Supports "
                    "parallel_core (.lora_A.weight), mindpet (.mindpet_delta_lora_a) "
                    "and pynative (.lora_a) naming conventions (auto-detected). "
                    "Multi-rank distributed checkpoints (with metadata.json) are "
                    "reassembled to full weights automatically. Output is always "
                    "a single full-weight safetensors file.")
    parser.add_argument("--src_ckpt_path", required=True, type=str,
                        help="Path to the fine-tuned checkpoint: a '*.safetensors' file, "
                             "a single-rank shard directory, a checkpoint save root, or a "
                             "multi-rank distributed iteration directory (metadata.json). "
                             "Optimizer shards are auto-excluded.")
    parser.add_argument("--dst_ckpt_path", required=True, type=str,
                        help="Output path for the merged safetensors checkpoint "
                             "('.safetensors' optional).")
    parser.add_argument("--lora_alpha", default=None, type=int,
                        help="LoRA scaling numerator; effective scale is alpha/r. If omitted, "
                             "read from a sibling adapter_config.json, else default 16.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    merge_lora(
        src_ckpt_path=args.src_ckpt_path,
        dst_ckpt_path=args.dst_ckpt_path,
        lora_alpha=args.lora_alpha,
    )
