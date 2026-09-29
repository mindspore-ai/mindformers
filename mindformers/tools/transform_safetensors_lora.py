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
import glob
import argparse
import hashlib
import tempfile
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
from mindformers.tools.lora_merge_utils import (
    MOE_ADAPTER_LABEL,
    AdapterConfig,
    build_adapter_plan,
    classify_adapter,
    read_adapter_config,
    validate_adapter_base,
    validate_adapter_pair,
)
from mindformers.tools.safetensors_lora_io import (
    atomic_save_checkpoint as _atomic_save_checkpoint,
    stream_merge_single_file as _stream_merge_single_file,
)

# Safetensors dtype string -> dtype used when saving arrays returned by
# ``load_safetensors_file``. Float8 values are decoded to float32 by that loader.
_SAFETENSOR_DTYPE_TO_MS = {
    "F32": ms.float32,
    "F16": ms.float16,
    "BF16": ms.bfloat16,
    "I64": ms.int64,
    "I32": ms.int32,
    "I16": ms.int16,
    "I8": ms.int8,
    "U8": ms.uint8,
    "BOOL": ms.bool_,
}

_MOE_ADAPTER_LABEL = MOE_ADAPTER_LABEL

# metadata.json ``storage_data`` keys are stringified tuples, e.g. ``('param.name', (0,))``.
_STORAGE_KEY_PATTERN = re.compile(r"^\('(.+)', \([0-9, ]*\)\)$")
_OPTIMIZER_SAFETENSORS_PATTERN = re.compile(r"(?:.+-)?opt-\d+-\d+\.safetensors")


def _ms_dtype_of(dtype_str: str):
    """Map a safetensors dtype string to a MindSpore dtype (float32 fallback)."""
    if dtype_str in {"F8_E8M0", "F8_E4M3", "F8_E5M2"}:
        raise NotImplementedError(
            f"Merging LoRA into a checkpoint containing quantized dtype '{dtype_str}' "
            "is not supported. Dequantize the checkpoint first."
        )
    dtype = _SAFETENSOR_DTYPE_TO_MS.get(dtype_str)
    if dtype is None:
        logger.warning("Unknown safetensors dtype '%s'; falling back to float32.", dtype_str)
        return ms.float32
    return dtype


def _ensure_mergeable_base_dtype(dtype, base_name: str) -> None:
    """Reject quantized/non-floating target weights before doing any arithmetic."""
    if dtype not in (ms.float32, ms.float16, ms.bfloat16):
        raise NotImplementedError(
            f"Merging LoRA into base weight '{base_name}' with dtype '{dtype}' is not supported. "
            "Only F32, F16 and BF16 base weights are supported."
        )


def _ensure_mergeable_adapter_dtype(dtype, adapter_name: str) -> None:
    """Reject adapter storage that would be interpreted as raw numeric codes."""
    if dtype not in (ms.float32, ms.float16, ms.bfloat16):
        raise NotImplementedError(
            f"LoRA adapter '{adapter_name}' has dtype '{dtype}', which is not supported. "
            "Only F32, F16 and BF16 adapters are supported."
        )


def _is_optimizer_shard(file_name: str) -> bool:
    """Return whether a Safetensors file name belongs to optimizer state.

    Beta1 does not expose the checkpoint helper added on master. Keep the same
    full-name matching rule locally so model names containing ``-opt-`` are not
    mistaken for optimizer shards.
    """
    return _OPTIMIZER_SAFETENSORS_PATTERN.fullmatch(os.path.basename(file_name)) is not None


def _is_optimizer_param(name: str) -> bool:
    """Filter stray optimizer state keys (adam_m/adam_v) if present in a shard."""
    return name.startswith("adam_m") or name.startswith("adam_v") or ".opt." in name


def _classify_adapter(name: str) -> Optional[Tuple[str, str, Tuple[str, str, str]]]:
    """Compatibility wrapper for the shared adapter classifier."""
    return classify_adapter(name)


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


def _read_adapter_config(src_path: str):
    """Compatibility wrapper returning the resolved config and manifest."""
    return read_adapter_config(src_path)


def _adapter_config_value(config: Optional[dict], key: str):
    """Read an adapter option from PEFT or MindFormers manifest layout."""
    if not config:
        return None
    if key in config:
        return config[key]
    lora_config = config.get("lora_config")
    return lora_config.get(key) if isinstance(lora_config, dict) else None


def _merge_delta(rank_reduction_weight: np.ndarray, rank_expansion_weight: np.ndarray,
                 base_shape: tuple,
                 scaling: float, label: str, adapter_name: str) -> np.ndarray:
    """Build a float32 LoRA delta for dense 2-D or routed-expert 3-D weights."""
    if label == _MOE_ADAPTER_LABEL:
        # MindFormers GroupedMLP layout: A=[E,in,r], B=[E,r,out], W=[E,in,out].
        delta = np.matmul(rank_reduction_weight, rank_expansion_weight) * scaling
    else:
        # Linear layout: A=[r,in], B=[out,r], W=[out,in].
        delta = (rank_expansion_weight @ rank_reduction_weight) * scaling

    if delta.shape == base_shape:
        return delta
    delta_t = np.swapaxes(delta, -1, -2)
    if delta_t.shape == base_shape:
        logger.warning("Adapter '%s' uses a transposed base layout; transposing delta %s -> %s.",
                       adapter_name, delta.shape, delta_t.shape)
        return delta_t
    raise ValueError(
        f"Cannot reconcile shapes for adapter '{adapter_name}': delta={tuple(delta.shape)}, "
        f"delta.T={tuple(delta_t.shape)}, base={tuple(base_shape)}.")


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


def _directory_digest(rank_dir: str) -> Tuple[int, bytes]:
    """Hash model shards without materialising another full model in memory."""
    digest = hashlib.sha256()
    files = _find_model_safetensors(rank_dir)
    for file_path in files:
        with open(file_path, "rb") as stream:
            while True:
                block = stream.read(16 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
    return len(files), digest.digest()


def _load_replicated_rank_params(ckpt_dir: str):
    """Load weights from a legacy static-graph multi-rank save (``rank_*/`` dirs)."""
    rank_dirs = sorted(d for d in glob.glob(os.path.join(ckpt_dir, "rank_*")) if os.path.isdir(d))
    if not rank_dirs:
        return None
    ref_dir = rank_dirs[0]
    ref_digest = _directory_digest(ref_dir)
    for rank_dir in rank_dirs[1:]:
        if _directory_digest(rank_dir) != ref_digest:
            raise ValueError(
                f"Rank directories under '{ckpt_dir}' hold different values for "
                f"their model shards ('{ref_dir}' vs '{rank_dir}'), i.e. this is "
                f"a sharded (tensor/expert/pipeline-parallel) checkpoint that "
                f"cannot be reassembled without a strategy file. Merge it from "
                f"a single rank directory (e.g. '{ref_dir}') or re-save with "
                f"'integrated_save' enabled instead."
            )
    ref_params, dtypes = _load_plain_params(_find_model_safetensors(ref_dir))
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


def _resolve_single_model_file(src_path: str) -> Optional[str]:
    """Resolve a locally complete, single safetensors model file if available.

    The streaming path intentionally does not attempt to reassemble distributed
    metadata shards: that operation itself requires a full global tensor.  It
    is used for the common full-checkpoint/HF-shard case, where every base
    tensor and its adapters are colocated in one safetensors file.
    """
    if os.path.isfile(src_path):
        return src_path if not _is_optimizer_shard(src_path) else None
    if not os.path.isdir(src_path):
        return None
    # Legacy static-graph saves place replicated copies below ``rank_*``
    # directories and have no direct safetensors file at the save root.  They
    # are handled (including replica-consistency checks) by
    # ``_load_replicated_rank_params`` in the full-load path.
    for probe in (src_path, os.path.join(src_path, "checkpoint")):
        if any(os.path.isdir(path) for path in glob.glob(os.path.join(probe, "rank_*"))):
            return None
    ckpt_dir = get_checkpoint_path(src_path)
    if os.path.isfile(os.path.join(ckpt_dir, "metadata.json")):
        return None
    files = _find_model_safetensors(ckpt_dir)
    return files[0] if len(files) == 1 else None


def merge_lora(base_ckpt_path: Optional[str] = None,
               dst_ckpt_path: Optional[str] = None,
               lora_alpha: Optional[int] = None,
               adapter_ckpt_path: Optional[str] = None,
               src_ckpt_path: Optional[str] = None) -> str:
    """Merge dense and routed-expert LoRA adapters into their base weights.

    ``base_ckpt_path`` may contain both base weights and adapters. When
    ``adapter_ckpt_path`` is provided, base weights and adapters are loaded
    independently. A combined single-rank safetensors checkpoint takes the
    streaming path; separate and distributed checkpoints use full reassembly.
    """
    if base_ckpt_path is None:
        base_ckpt_path = src_ckpt_path
    elif src_ckpt_path is not None \
            and os.path.realpath(base_ckpt_path) != os.path.realpath(src_ckpt_path):
        raise ValueError("base_ckpt_path and legacy src_ckpt_path refer to different checkpoints.")
    if not base_ckpt_path:
        raise ValueError("base_ckpt_path is required.")
    if not dst_ckpt_path:
        raise ValueError("dst_ckpt_path is required.")

    adapter_source = adapter_ckpt_path or base_ckpt_path
    same_source = os.path.realpath(adapter_source) == os.path.realpath(base_ckpt_path)

    # Resolve alpha: explicit flag -> adapter_config.json -> default 16.
    resolved_config = _read_adapter_config(adapter_source)
    if isinstance(resolved_config, AdapterConfig):
        cfg = resolved_config.values
        manifest = resolved_config.manifest
        manifest_path = resolved_config.path
    else:
        # Preserve the private helper's historical monkeypatch contract in unit tests.
        cfg = resolved_config
        manifest = None
        manifest_path = None
    if not same_source:
        validate_adapter_base(manifest, base_ckpt_path, manifest_path)
    if lora_alpha is None:
        alpha_in_cfg = _adapter_config_value(cfg, "lora_alpha")
        if alpha_in_cfg is not None:
            lora_alpha = int(alpha_in_cfg)
            logger.info("Read lora_alpha=%d from adapter_config.json.", lora_alpha)
    if lora_alpha is None:
        lora_alpha = 16
        logger.info("No lora_alpha given or in adapter_config.json; using default %d.", lora_alpha)
    if isinstance(lora_alpha, bool):
        raise ValueError("lora_alpha must be a positive integer.")
    try:
        lora_alpha = int(lora_alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("lora_alpha must be a positive integer.") from exc
    if lora_alpha <= 0:
        raise ValueError("lora_alpha must be a positive integer.")
    out_base = dst_ckpt_path[:-len(".safetensors")] if dst_ckpt_path.endswith(".safetensors") \
        else dst_ckpt_path
    out_dir = os.path.dirname(os.path.abspath(out_base))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out_file = out_base + ".safetensors"
    for source_path in (base_ckpt_path, adapter_source):
        if os.path.isfile(source_path) \
                and os.path.realpath(os.path.abspath(source_path)) == os.path.realpath(os.path.abspath(out_file)):
            raise ValueError("dst_ckpt_path must not overwrite a base or adapter checkpoint.")

    stream_source = _resolve_single_model_file(base_ckpt_path) if same_source else None
    if stream_source:
        if os.path.realpath(os.path.abspath(stream_source)) == os.path.realpath(os.path.abspath(out_file)):
            raise ValueError("dst_ckpt_path must not overwrite a base or adapter checkpoint.")
        fd, temp_file = tempfile.mkstemp(
            prefix=".lora-stream-", suffix=".safetensors", dir=out_dir)
        os.close(fd)
        os.remove(temp_file)
        try:
            merged_count = _stream_merge_single_file(stream_source, temp_file, int(lora_alpha))
            if merged_count is not None:
                os.replace(temp_file, out_file)
                logger.info("LoRA streaming merge succeeded: %d adapters folded into '%s'.",
                            merged_count, out_file)
                return out_file
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    params, dtypes, base_sources = _load_checkpoint_params(base_ckpt_path)
    logger.info("Loaded %d base parameters from: %s", len(params), base_sources)
    if not same_source:
        # An explicit adapter path is authoritative. Ignore any stale adapter
        # tensors that may happen to coexist with the selected base weights.
        params = {name: value for name, value in params.items() if _classify_adapter(name) is None}
        dtypes = {name: dtype for name, dtype in dtypes.items() if name in params}
        adapter_params, adapter_dtypes, adapter_sources = _load_checkpoint_params(adapter_source)
        adapter_params = {
            name: value for name, value in adapter_params.items()
            if _classify_adapter(name) is not None
        }
        if not adapter_params:
            raise ValueError(
                f"No LoRA adapter parameters found in '{adapter_source}'."
            )
        params.update(adapter_params)
        dtypes.update({name: adapter_dtypes[name] for name in adapter_params if name in adapter_dtypes})
        logger.info("Loaded %d adapter parameters from: %s", len(adapter_params), adapter_sources)

    # Build one validation plan shared with the streaming path.
    adapter_plan = build_adapter_plan(params)
    label_counts = {}
    for spec in adapter_plan:
        label = spec.label
        label_counts[label] = label_counts.get(label, 0) + 1
    if not adapter_plan:
        raise ValueError(
            f"No LoRA adapter parameters found in '{adapter_source}'. Expected one "
            f"of the supported naming conventions: <base>.lora_A.weight / "
            f"<base>.mindpet_delta_lora_a / <base>.lora_a (and the matching B)."
        )
    logger.info("Found %d LoRA adapter(s): %s", len(adapter_plan), label_counts)

    merged_count = 0
    for spec in adapter_plan:
        rank_reduction_key = spec.rank_reduction_name
        rank_expansion_key = spec.rank_expansion_name
        base_key, label = spec.base_name, spec.label
        # Matmul in float32 for numerical stability; cast back on save.
        _ensure_mergeable_base_dtype(dtypes.get(base_key, ms.float32), base_key)
        _ensure_mergeable_adapter_dtype(dtypes.get(rank_reduction_key), rank_reduction_key)
        _ensure_mergeable_adapter_dtype(dtypes.get(rank_expansion_key), rank_expansion_key)
        rank_reduction_weight = params[rank_reduction_key].astype(np.float32)
        rank_expansion_weight = params[rank_expansion_key].astype(np.float32)
        base_weight = params[base_key].astype(np.float32)
        lora_rank = validate_adapter_pair(
            rank_reduction_weight,
            rank_expansion_weight,
            label,
            rank_reduction_key,
        )
        scaling = float(lora_alpha) / float(lora_rank)
        weight_delta = _merge_delta(
            rank_reduction_weight,
            rank_expansion_weight,
            base_weight.shape,
            scaling,
            label,
            rank_reduction_key,
        )

        params[base_key] = base_weight + weight_delta
        merged_count += 1
        logger.info(
            "Merged '%s' [%s]: r=%d, alpha=%d, scaling=%.6f -> '%s'.",
            rank_reduction_key, label, lora_rank, lora_alpha, scaling, base_key,
        )

    # Drop every LoRA adapter parameter; keep all other (base) params as-is.
    params = {name: value for name, value in params.items() if _classify_adapter(name) is None}

    # Cast back to the original storage dtype to reproduce the base layout.
    save_list = [
        {"name": name, "data": Tensor(value, dtypes.get(name, ms.float32))}
        for name, value in params.items()
    ]

    # Distributed checkpoint fallback necessarily materialises full tensors, but
    # the final destination is still committed atomically.
    _atomic_save_checkpoint(save_list, out_file)

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
                    "pynative (.lora_a), and routed-expert (_lora_a) naming "
                    "conventions (auto-detected). "
                    "Multi-rank distributed checkpoints (with metadata.json) are "
                    "reassembled to full weights automatically. Output is always "
                    "a single full-weight safetensors file.")
    parser.add_argument("--base_ckpt_path", "--src_ckpt_path", dest="base_ckpt_path",
                        required=True, type=str,
                        help="Path to the base checkpoint, or to a combined base+LoRA "
                             "checkpoint when --adapter_ckpt_path is omitted. The legacy "
                             "name --src_ckpt_path is retained as an alias.")
    parser.add_argument("--adapter_ckpt_path", default=None, type=str,
                        help="Optional separate LoRA adapter checkpoint. When omitted, "
                             "adapters are read from --base_ckpt_path.")
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
        base_ckpt_path=args.base_ckpt_path,
        dst_ckpt_path=args.dst_ckpt_path,
        lora_alpha=args.lora_alpha,
        adapter_ckpt_path=args.adapter_ckpt_path,
    )
