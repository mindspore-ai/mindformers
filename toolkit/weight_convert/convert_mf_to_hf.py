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
"""
Convert MindFormers (MF) checkpoint weights to HuggingFace (HF) format.

Driven by ``mindformers.checkpoint.converter.template.WeightTemplate`` in its
MF -> HF direction, over the ``weight_converters`` the model class declares, so
this script owns no weight mapping: names and layouts come from the single
source of truth in ``mindformers/models/<model>/utils.py``. Declaring a weight
there makes it exportable; anything undeclared is reported as unconverted rather
than exported under an invented name. Depends only on MindSpore / MindFormers
and ``safetensors`` — no PyTorch, as BF16/FP16 tensors are cast on CPU by
MindSpore and written through ``safetensors.serialize_file``.

Usage::

    python convert_mf_to_hf.py --yaml_config test.yaml \\
        --input_path /path/to/mf_checkpoint/ --output_path /path/to/hf_output/
"""

import argparse
import importlib
import inspect
import json
import os
import re
import sys
import multiprocessing
from collections import defaultdict
from dataclasses import replace
from glob import glob

import numpy as np
import yaml
from safetensors import serialize_file

import mindspore as ms
from mindspore.ops.operations import Cast

from mindformers.tools.logger import logger
from mindformers.checkpoint.converter.convert_op import (
    ConcatConvertOp, ExpertsConvertOp, QKVConvertOp, QKVBiasConvertOp)
from mindformers.checkpoint.converter.template import WeightTemplate
from mindformers.checkpoint.safetensors_utils import read_safetensors_header
from mindformers.checkpoint.utils import get_sharded_tensor_shard_id

ms.set_device(device_target='CPU')
cpu_cast = Cast().set_device('CPU')

# Target dtype for floating-point weights.
DTYPE_MAP = {
    'fp32': ms.float32, 'float32': ms.float32,
    'bf16': ms.bfloat16, 'bfloat16': ms.bfloat16,
    'fp16': ms.float16, 'float16': ms.float16,
}

MS_FLOAT_DTYPES = (ms.float16, ms.bfloat16, ms.float32, ms.float64)

# safetensors dtype tag -> (numpy dtype to read the raw bytes with, MindSpore dtype).
# BF16 has no numpy equivalent, so it is read as uint16 and widened by hand.
ST_TAG_TO_DTYPE = {
    'BF16': (np.uint16, ms.bfloat16), 'F16': (np.float16, ms.float16),
    'F32': (np.float32, ms.float32), 'F64': (np.float64, ms.float64),
    'I8': (np.int8, ms.int8), 'I16': (np.int16, ms.int16),
    'I32': (np.int32, ms.int32), 'I64': (np.int64, ms.int64),
    'U8': (np.uint8, ms.uint8), 'BOOL': (np.bool_, ms.bool_),
}

# MindSpore float dtype -> safetensors raw-API dtype name. The raw
# ``serialize_file`` API takes lowercase names and writes the F32/BF16/... tags.
ST_FLOAT_DTYPE = {ms.float32: 'float32', ms.float16: 'float16',
                  ms.bfloat16: 'bfloat16', ms.float64: 'float64'}

# Non-float weights (the router's ``tid2eid`` index table, say) must NOT be cast
# to bf16; these numpy dtype names pass straight through as safetensors names.
ST_NUMPY_DTYPE = frozenset({'bool', 'uint8', 'int8', 'int16', 'int32', 'int64',
                            'float16', 'float32', 'float64'})

# Exportable models are discovered, not listed: any class under
# ``mindformers/models/*/utils.py`` declaring ``weight_converters`` is exportable,
# keyed by its own ``config_class.model_type``. A hand-kept list goes stale.
MODEL_CONVERTERS = {}


def discover_model_converters():
    """Populate MODEL_CONVERTERS from the models that declare weight_converters."""
    if MODEL_CONVERTERS:
        return MODEL_CONVERTERS
    models_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), 'mindformers', 'models')
    for name in sorted(os.listdir(models_dir)):
        if not os.path.isfile(os.path.join(models_dir, name, 'utils.py')):
            continue
        try:
            module = importlib.import_module(f'mindformers.models.{name}.utils')
        except ImportError as exc:            # optional deps, unrelated models
            logger.debug(f"Skipping mindformers.models.{name}.utils: {exc}")
            continue
        for obj in vars(module).values():
            if not (inspect.isclass(obj) and getattr(obj, 'weight_converters', None)):
                continue
            model_type = getattr(getattr(obj, 'config_class', None), 'model_type', None)
            if model_type and model_type not in MODEL_CONVERTERS:
                MODEL_CONVERTERS[model_type] = obj
    return MODEL_CONVERTERS

# Optimizer state lives beside the model weights in a checkpoint directory and is
# not part of an export. Filtered by file name first, then by parameter prefix as
# a net for layouts that keep both in one file.
OPTIMIZER_FILE_RE = re.compile(r'-opt-|optimizer')
OPTIMIZER_PREFIXES = frozenset({'adam_m', 'adam_v', 'muon_m', 'muon_v',
                                'main_param', 'exp_avg', 'exp_avg_sq', 'momentum'})

# Parameters the reference DeepSeek-V4 release keeps in fp32 regardless of the
# checkpoint's weight dtype (verified against the published shards: every other
# non-quantised tensor there is bf16, these ten are F32). They are tiny, and they
# are gains/biases rather than matmul weights, so precision matters more than
# size. ``--uniform_dtype`` turns this off.
FP32_HF_SUFFIXES = (
    '.attn_sink', '.compressor.ape',   # the latter covers .indexer.compressor.ape
    '.ffn.gate.bias',
    '.hc_attn_base', '.hc_attn_fn', '.hc_attn_scale',
    '.hc_ffn_base', '.hc_ffn_fn', '.hc_ffn_scale',
    'hc_head_base', 'hc_head_fn', 'hc_head_scale',
)


class DtypePolicy:
    """Decides the output dtype of each HF tensor.

    ``default_dtype`` of None selects source mode: every tensor is written with
    the dtype it has in the MF checkpoint, which makes the export bit-exact and
    is what a load-and-compare-loss check needs.
    """

    def __init__(self, default_dtype, uniform=False):
        self.default = default_dtype
        self.uniform = uniform

    def for_name(self, hf_name, source_dtype=None):
        """Return the dtype to write ``hf_name`` with."""
        if self.default is None:
            if source_dtype is None:
                raise ValueError(f"source dtype unknown for {hf_name}")
            return source_dtype
        if not self.uniform and hf_name.endswith(FP32_HF_SUFFIXES):
            return ms.float32
        return self.default

    def __str__(self):
        if self.default is None:
            return "source (each tensor keeps its MF dtype)"
        if self.uniform:
            return f"{self.default} (uniform)"
        return f"{self.default}, fp32 for {len(FP32_HF_SUFFIXES)} scalar/HC parameters"


# =============================================================================
# Tensor IO helpers (MindSpore only, no torch)
# =============================================================================

def to_numpy(value):
    """Return (numpy view, source dtype) for an MF Parameter/Tensor.

    Floats are up-cast to fp32 because numpy has no bf16; integer and bool
    weights (e.g. the router ``tid2eid`` table) are kept as-is so they are never
    silently turned into floats. The source dtype is returned so ``--dtype
    source`` can write each tensor back exactly as it was stored.
    """
    if value.dtype in MS_FLOAT_DTYPES:
        return cpu_cast(value, ms.float32).numpy(), value.dtype
    return value.asnumpy(), value.dtype


def pack(array, dtype):
    """Build a safetensors entry {dtype, shape, data} from a numpy array."""
    array = np.ascontiguousarray(array)
    if array.dtype.kind != 'f':
        if array.dtype.name not in ST_NUMPY_DTYPE:
            raise TypeError(f"Unsupported non-float weight dtype: {array.dtype}")
        return {"dtype": array.dtype.name, "shape": list(array.shape),
                "data": array.tobytes()}
    tensor = cpu_cast(ms.Tensor(array), dtype)
    return {
        "dtype": ST_FLOAT_DTYPE[dtype],
        "shape": list(array.shape),
        "data": tensor.get_bytes(),
    }


def save_safetensors(entries, file_path):
    """Write {name: {dtype, shape, data}} to a safetensors file.

    No ``__metadata__`` block is written: the reference DeepSeek-V4 shards do not
    carry one, and MindFormers' HF metadata reader iterates the header without
    skipping it.
    """
    serialize_file(entries, file_path)


# =============================================================================
# Template construction
# =============================================================================

def build_template(model_type, config):
    """Build the MF -> HF WeightTemplate for ``model_type``.

    The expert / QKV / fc1 converters need values from the config, set here
    rather than through ``set_model_config``: an offline full checkpoint is
    unsharded (EP = 1, optimizer shard = 1) with no auto-parallel context. Each
    op is copied first — ``weight_converters`` is a class attribute of shared
    instances, so configuring in place would leave this run's geometry on it.
    """
    model_cls = MODEL_CONVERTERS.get(model_type)
    if model_cls is None:
        raise ValueError(f"No weight_converters registered for model type '{model_type}'.")

    hidden_size = config.get('hidden_size')
    converters = [_configure_op(op, config, hidden_size)
                  for op in model_cls.weight_converters]
    return WeightTemplate(weight_converters=converters)


def _configure_op(op, config, hidden_size):
    """Return a copy of ``op`` with the values an offline export has to supply."""
    if isinstance(op, ExpertsConvertOp):
        return replace(op, num_moe_experts=num_routed_experts(config),
                       hidden_size=hidden_size,
                       moe_ffn_hidden_size=config.get('moe_intermediate_size'),
                       expert_parallel_size=1, optimizer_parallel_size=1)
    if isinstance(op, ConcatConvertOp) and any('linear_fc1' in n for n in op.mf_names):
        # The fused gate|up layout belongs to the model, not the mapping:
        # `use_interleaved_weight_layout_mlp` picks MLPInterleaved over MLP,
        # which read it differently. See ConcatConvertOp.set_model_config.
        fc1_layout = config.get('use_interleaved_weight_layout_mlp')
        return op if fc1_layout is None else replace(op, interleaved=bool(fc1_layout))
    if isinstance(op, (QKVConvertOp, QKVBiasConvertOp)):
        heads = config.get('num_attention_heads')
        return replace(op, num_attention_heads=heads,
                       num_query_groups=config.get('num_key_value_heads') or heads,
                       kv_channels=config.get('head_dim'), hidden_size=hidden_size,
                       tensor_model_parallel_size=1, optimizer_parallel_size=1)
    return op


def num_routed_experts(config):
    """Routed expert count: ``n_routed_experts`` (DeepSeek) or ``num_experts``."""
    return config.get('n_routed_experts') or config.get('num_experts') or 0


def check_not_sharded(mf_name, array, config):
    """Fail on a weight whose size says the checkpoint is still rank-sharded.

    Exporting a shard yields a checkpoint that looks well-formed but holds a
    fraction of every tensor, which nothing downstream can detect. These few
    weights have element counts derivable from the config, so they gate cheaply.
    The comparison is on total element count rather than any single dimension:
    experts.weight1 is [E, hidden, 2*ffn] here but [E*hidden, 2*ffn] after
    ExpertsConvertOp, and only the total is stable across both layouts.
    """
    vocab = config.get('vocab_size')
    hidden = config.get('hidden_size')
    experts = num_routed_experts(config)
    ffn = config.get('moe_intermediate_size')

    if mf_name in ('embedding.word_embeddings.weight', 'output_layer.weight'):
        label, want = 'vocab_size * hidden_size', vocab * hidden if vocab and hidden else 0
    elif mf_name.endswith('.mlp.experts.weight1'):
        label = 'n_routed_experts * hidden_size * 2 * moe_intermediate_size'
        want = experts * hidden * 2 * ffn if experts and hidden and ffn else 0
    elif mf_name.endswith('.mlp.experts.weight2'):
        label = 'n_routed_experts * moe_intermediate_size * hidden_size'
        want = experts * ffn * hidden if experts and hidden and ffn else 0
    else:
        return
    if not want:                      # geometry missing from the YAML: nothing to check
        return

    got = int(array.size)
    if got != want:
        ratio = f" ({want // got}x smaller)" if got and want % got == 0 else ""
        raise ValueError(
            f"{mf_name} holds {got} elements, but the YAML config implies "
            f"{want} ({label}){ratio}.\n"
            f"The checkpoint is most likely still sharded across ranks (FSDP / expert "
            f"parallel / remove_redundancy). Exporting it would produce a checkpoint that "
            f"holds only a slice of every weight. Aggregate the checkpoint first, or fix "
            f"the config if the geometry in the YAML is wrong."
        )


class ConversionStats:
    """Tracks which MF weights the template claimed, across all layers."""

    def __init__(self):
        self.converted = set()
        self.unclaimed = set()

    def report(self, all_keys, pending):
        """Print the final coverage summary. Returns True when everything converted."""
        leftovers = sorted((set(all_keys) - self.converted) | self.unclaimed | set(pending))
        if not leftovers:
            logger.info(f"All {len(all_keys)} MF weights were converted.")
            return True
        logger.warning(f"{len(leftovers)} MF weight(s) were NOT converted and are "
                       f"missing from the output:")
        for name in leftovers[:50]:
            reason = "no converter declares it" if name in self.unclaimed else "incomplete group"
            logger.warning(f"  - {name}  ({reason})")
        if len(leftovers) > 50:
            logger.warning(f"  ... and {len(leftovers) - 50} more")
        return False


def convert_weights(template, mf_weights, policy, stats, config, verify_whole=False):
    """Push a batch of MF weights through the template, returning HF entries.

    ``verify_whole`` gates the rank-shard size check. It applies to plain
    checkpoints, where one rank's shard is indistinguishable from a whole
    checkpoint, but not to the sharded reader, which has already reassembled
    each parameter and verified it against ``global_shape``.
    """
    hf_entries = {}
    for mf_name in list(mf_weights):
        array, source_dtype = mf_weights[mf_name]
        if verify_whole:
            check_not_sharded(mf_name, array, config)
        cached_before = set(template.name_to_weight)
        result = template.add_mf_weight(mf_name, array)
        if result is None:
            # Either cached while waiting for the rest of its group, or no
            # converter claims it at all (add_mf_weight drops it in that case).
            if mf_name not in template.name_to_weight:
                stats.unclaimed.add(mf_name)
            continue
        # A completed group also consumes whatever was cached for it earlier.
        stats.converted.add(mf_name)
        stats.converted.update(cached_before - set(template.name_to_weight))
        for hf_name, value in result.items():
            hf_entries[hf_name] = pack(value, policy.for_name(hf_name, source_dtype))
    return hf_entries


# =============================================================================
# YAML config
# =============================================================================

def parse_yaml_config(yaml_path):
    """Parse MindFormers YAML config and extract model configuration."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model_config = config.get('model') or {}

    result = dict(model_config)
    result.update({
        'model_type': model_config.get('model_type', 'unknown'),
        'architectures': model_config.get('architectures', ''),
        'num_layers': model_config.get('num_hidden_layers', 1),
        'num_nextn_predict_layers': model_config.get('num_nextn_predict_layers', 0) or 0,
        'n_routed_experts': (model_config.get('n_routed_experts')
                             or model_config.get('num_experts') or 0),
        'compute_dtype': model_config.get('compute_dtype') or 'bfloat16',
        'params_dtype': model_config.get('params_dtype') or 'bfloat16',
        'tie_word_embeddings': model_config.get('tie_word_embeddings', False),
    })
    result['_raw_model_section'] = model_config
    return result


def resolve_dtype(config, dtype_override=None):
    """Resolve the output floating-point dtype. None means 'keep the source dtype'."""
    if dtype_override == 'source':
        return None
    dtype_str = dtype_override or config.get('params_dtype') or config.get('compute_dtype')
    dtype = DTYPE_MAP.get(str(dtype_str).lower())
    if dtype is None:
        raise ValueError(f"Unsupported dtype '{dtype_str}'. Supported: {sorted(DTYPE_MAP)}")
    return dtype


# =============================================================================
# Checkpoint layout discovery
# =============================================================================

INDEX_FILE_NAMES = ("param_name_map.json", "ms-model.safetensors.index.json",
                    "model.safetensors.index.json")

LAYER_KEY_RE = re.compile(r'^(decoder|mtp)\.layers\.(\d+)\.')


def _layer_key(weight_key):
    """Bucket a MF weight under its layer: an int for decoder, 'mtp_N' for MTP.

    Weights outside any layer (embedding, final norm, output head) are their own
    bucket, so ``_extra_keys_for_layer`` can ask for them by name.
    """
    m = LAYER_KEY_RE.match(weight_key)
    if not m:
        return weight_key
    return int(m.group(2)) if m.group(1) == 'decoder' else f'mtp_{m.group(2)}'


def build_layer_file_map(file_path):
    """Map layer index (or weight name) -> set of safetensors files holding it.

    Returns (layer_st_map, all_weight_names).
    """
    layer_st_map = defaultdict(set)

    weight_map_file = next(
        (p for p in (os.path.join(file_path, n) for n in INDEX_FILE_NAMES)
         if os.path.exists(p)), None)

    if weight_map_file:
        with open(weight_map_file, encoding='utf-8') as f:
            weights_map = json.load(f)
        if isinstance(weights_map, dict) and "weight_map" in weights_map:
            weights_map = weights_map["weight_map"]
    else:
        safetensors_files = sorted(glob(os.path.join(file_path, "*.safetensors")))
        # save_checkpoint writes model and optimizer state side by side as
        # "<prefix>-model-<rank>-<total>" / "<prefix>-opt-<rank>-<total>". The
        # optimizer files hold adam_m / adam_v / muon_m / main_param entries that
        # no converter claims; reading them wastes time and buries the real
        # unconverted report under optimizer names.
        model_files = [f for f in safetensors_files
                       if not OPTIMIZER_FILE_RE.search(os.path.basename(f))]
        skipped = len(safetensors_files) - len(model_files)
        if skipped:
            logger.info(f"Skipping {skipped} optimizer shard(s); exporting model weights only.")
        safetensors_files = model_files
        if not safetensors_files:
            raise ValueError(f"No model safetensors files found in {file_path}")
        weights_map = {}
        seen_in = defaultdict(list)
        for st_path in safetensors_files:
            shard = os.path.basename(st_path)
            header, _ = read_safetensors_header(st_path)
            for key in header:
                if key == '__metadata__' or key.split('.')[0] in OPTIMIZER_PREFIXES:
                    continue
                weights_map[key] = shard
                seen_in[key].append(shard)
        repeated = {k: v for k, v in seen_in.items() if len(v) > 1}
        if repeated:
            sample = sorted(repeated)[:5]
            raise ValueError(
                f"{file_path} looks like a rank-sharded checkpoint: "
                f"{len(repeated)} weight name(s) appear in more than one file, e.g.\n" +
                "\n".join(f"  {k}: {repeated[k]}" for k in sample) +
                "\nEach rank file then holds only its slice of those weights, and merging "
                "them by name would silently keep one slice and drop the rest. Aggregate the "
                "checkpoint into an unsharded one before exporting."
            )

    for weight_key, value in weights_map.items():
        layer_st_map[_layer_key(weight_key)].add(os.path.join(file_path, value))

    return layer_st_map, set(weights_map.keys())


def read_matched_file(layer_st_map, layer_indices, prefixes, extra_keys=None):
    """Read just the tensors a layer needs from the files backing it.

    Only the entries under ``prefixes`` (plus ``extra_keys``) are read, one
    tensor at a time. A single-file checkpoint maps every layer to that one
    file, so loading it whole would give each worker a full copy of the model.
    """
    extra_keys = tuple(extra_keys or ())
    st_file_list = set()
    for key in list(layer_indices) + list(extra_keys):
        st_file_list.update(layer_st_map.get(key, ()))

    weights = {}
    for st_file in sorted(st_file_list):
        header, data_start = read_safetensors_header(st_file)
        for name in header:
            # `__metadata__` is a free-form annotation block, not a tensor entry.
            if name == '__metadata__' or name.split('.')[0] in OPTIMIZER_PREFIXES:
                continue
            if name.startswith(prefixes) or name in extra_keys:
                weights[name] = read_one_tensor(st_file, header, data_start, name)
    return weights


# =============================================================================
# Distributed (rank-sharded) MindFormers checkpoints
# =============================================================================


def read_one_tensor(file_path, header, data_start, name):
    """Read a single tensor out of an already-parsed safetensors file.

    Reading tensor by tensor keeps peak memory to the layer being converted: a
    plain checkpoint is often one file holding every layer, so loading it whole
    would cost each worker a full copy of the model.
    """
    info = header.get(name)
    if info is None:
        raise KeyError(f"{name} not found in {os.path.basename(file_path)}")
    tag = info['dtype']
    if tag not in ST_TAG_TO_DTYPE:
        raise TypeError(
            f"{name} in {os.path.basename(file_path)} has dtype {tag}, which this "
            f"tool does not handle (quantised checkpoints are out of scope).")
    np_dtype, ms_dtype = ST_TAG_TO_DTYPE[tag]
    start, end = info['data_offsets']
    with open(file_path, 'rb') as f:
        f.seek(data_start + start)
        raw = f.read(end - start)
    array = np.frombuffer(raw, dtype=np_dtype).reshape(info['shape'])
    if tag == 'BF16':
        array = (array.astype(np.uint32) << 16).view(np.float32)
    return array, ms_dtype


class ShardedCheckpointReader:
    """Reads a rank-sharded MindFormers checkpoint as whole parameters.

    ``save_checkpoint`` writes one file per rank plus a ``metadata.json`` giving
    each parameter's ``global_shape``, the fragment count per axis, and the file
    holding each fragment — enough to return the unsharded parameter with no
    merge step. Tensors are read one at a time from the safetensors data block;
    loading whole rank files would re-read every file once per layer.
    """

    def __init__(self, file_path):
        with open(os.path.join(file_path, 'metadata.json'), encoding='utf-8') as f:
            metadata = json.load(f)
        self.file_path = file_path
        self._meta = metadata['state_dict_metadata']
        self._storage = metadata['storage_data']
        self._headers = {}
        self.names = sorted(
            name for name in self._meta
            if name.split('.')[0] not in OPTIMIZER_PREFIXES
        )

    @staticmethod
    def is_distributed_checkpoint(file_path):
        """True when file_path holds a MindFormers distributed checkpoint."""
        meta_file = os.path.join(file_path, 'metadata.json')
        if not os.path.exists(meta_file):
            return False
        try:
            with open(meta_file, encoding='utf-8') as f:
                return 'state_dict_metadata' in json.load(f)
        except (ValueError, OSError):
            return False

    def _header(self, file_name):
        if file_name not in self._headers:
            self._headers[file_name] = read_safetensors_header(
                os.path.join(self.file_path, file_name))
        return self._headers[file_name]

    def read(self, name):
        """Return the whole parameter as (numpy array, MF dtype)."""
        entry = self._meta[name]
        fragments = entry.get('axis_fragmentations') or []
        split_axes = [i for i, n in enumerate(fragments) if n and n > 1]
        if len(split_axes) > 1:
            raise ValueError(
                f"{name} is split along more than one axis ({fragments}); this tool "
                f"only reassembles single-axis sharding.")
        axis = split_axes[0] if split_axes else None    # None: replicated

        pieces = []
        for chunk in sorted(entry['chunk'], key=lambda c: c['global_offset'][0]):
            locations = self._storage[
                get_sharded_tensor_shard_id(name, chunk['global_offset'])]
            shard_file = locations[0]['file_name']
            header, data_start = self._header(shard_file)
            array, ms_dtype = read_one_tensor(
                os.path.join(self.file_path, shard_file), header, data_start, name)
            if axis is None:                      # replicated: any copy will do
                return array, ms_dtype
            pieces.append(array)

        merged = np.concatenate(pieces, axis=axis) if len(pieces) > 1 else pieces[0]
        expected = tuple(entry['global_shape'])
        if merged.shape != expected:
            raise ValueError(
                f"{name} reassembled to {merged.shape} but metadata.json declares "
                f"{expected}; the checkpoint's shard layout is not what this tool "
                f"expects.")
        return merged, ms_dtype

    def read_layer(self, prefixes, extra_names=()):
        """Read every parameter under any of ``prefixes``, plus ``extra_names``."""
        wanted = [n for n in self.names if any(n.startswith(p) for p in prefixes)]
        wanted += [n for n in extra_names if n in self._meta and n not in wanted]
        return {name: self.read(name) for name in wanted}


# =============================================================================
# Conversion driver
# =============================================================================

def _shard_name(index, total):
    return f"model-{index + 1:05d}-of-{total:05d}.safetensors"


def _flush_shard(hf_entries, shard_file, output_path, weight_map, index, total):
    """Write one shard and update the index bookkeeping. Returns bytes written."""
    written = 0
    for name, entry in hf_entries.items():
        weight_map[name] = shard_file
        written += len(entry["data"])
    save_safetensors(hf_entries, os.path.join(output_path, shard_file))
    logger.info(f"Saved layer {index + 1}/{total} -> {shard_file} ({len(hf_entries)} tensors)")
    return written


def _extra_keys_for_layer(layer_id, total_layers):
    """Non-layer MF weights that must be loaded alongside ``layer_id``."""
    extra_keys = []
    if layer_id == 0:
        extra_keys += ['embedding.word_embeddings.weight', 'decoder.hc_head.hc_base',
                       'decoder.hc_head.hc_fn.weight', 'decoder.hc_head.hc_scale']
    if layer_id == total_layers - 1:
        extra_keys += ['decoder.final_layernorm.weight', 'output_layer.weight']
    return extra_keys


def _report_tied_head(weight_map, config, template):
    """Explain a missing output head rather than leaving it unmentioned.

    Its HF name differs per model (``head.weight`` vs ``lm_head.weight``), so it
    is asked of the model's own converters rather than assumed.
    """
    head_names = template.get_hf_names_for_mf('output_layer.weight')
    if not head_names:
        return                      # this model declares no output head at all
    if any(name in weight_map for name in head_names):
        return

    head = head_names[0]
    if config.get('tie_word_embeddings'):
        logger.info(f"Note: no output_layer.weight and tie_word_embeddings=True; "
                    f"'{head}' is left out so it is tied to the embedding on load.")
    else:
        logger.warning(f"No 'output_layer.weight' found and tie_word_embeddings is "
                       f"False; the exported checkpoint has no '{head}'.")


# Per-process state for the worker pool. Each worker builds its own template and
# its own reader: WeightTemplate caches partially-filled converter groups in
# ``name_to_weight``, so a shared one would have workers overwriting each other.
_WORKER = {}


# Per-worker ceiling on MindSpore's CPU op threads. The work here is concatenate
# / copy / dtype-cast, i.e. memory-bandwidth bound rather than compute bound, so
# threads past a handful buy nothing and only add contention. Without a ceiling a
# big machine hands each worker dozens of threads it cannot use.
_MAX_THREADS_PER_WORKER = 8


def _available_cpus():
    """CPUs this process may actually run on, not the machine's total."""
    if hasattr(os, 'sched_getaffinity'):
        return len(os.sched_getaffinity(0)) or 1
    return os.cpu_count() or 1


def _threads_per_worker(workers):
    """Split one core-sized thread budget between the worker processes.

    Each process runs MindSpore's own CPU op thread pool, which sizes itself from
    the machine's core count and ignores OMP_NUM_THREADS — so N workers otherwise
    means N full-sized pools and a badly oversubscribed box, however modest N
    looks. Keeping total threads at roughly one per core is what leaves the
    machine responsive enough to still log into.
    """
    threads = max(1, min(_available_cpus() // workers, _MAX_THREADS_PER_WORKER))
    ms.set_context(runtime_num_threads=threads)
    return threads


def _policy_to_spec(policy):
    """Reduce a DtypePolicy to picklable primitives (ms dtypes do not pickle)."""
    if policy.default is None:
        return (None, policy.uniform)
    return (ST_FLOAT_DTYPE[policy.default], policy.uniform)


def _policy_from_spec(spec):
    name, uniform = spec
    return DtypePolicy(None if name is None else DTYPE_MAP[name], uniform)


def _worker_init(input_path, output_path, config, policy_spec, layer_st_map,
                 threads_per_worker):
    """Build this process's template and checkpoint reader exactly once."""
    # Also applied here, not just in the parent: a spawned worker re-imports
    # MindSpore from scratch and would otherwise start a full-sized thread pool.
    ms.set_context(runtime_num_threads=threads_per_worker)
    discover_model_converters()
    _WORKER['input_path'] = input_path
    _WORKER['output_path'] = output_path
    _WORKER['config'] = config
    _WORKER['policy'] = _policy_from_spec(policy_spec)
    _WORKER['template'] = build_template(config['model_type'], config)
    _WORKER['stats'] = ConversionStats()
    _WORKER['layer_st_map'] = layer_st_map
    _WORKER['sharded'] = (ShardedCheckpointReader(input_path)
                          if layer_st_map is None else None)


def _layer_prefixes(layer_id, num_layers):
    """MF name prefix owning ``layer_id`` (decoder layer or MTP layer)."""
    if layer_id < num_layers:
        return (f'decoder.layers.{layer_id}.',)
    return (f'mtp.layers.{layer_id - num_layers}.',)


def _read_layer_source(layer_id, total_layers, num_layers, sharded, layer_st_map):
    """Read just the MF weights layer ``layer_id`` needs."""
    extra_keys = _extra_keys_for_layer(layer_id, total_layers)
    prefixes = _layer_prefixes(layer_id, num_layers)
    if sharded is not None:
        return sharded.read_layer(list(prefixes), extra_keys), True
    layer_indices = ([layer_id] if layer_id < num_layers
                     else [f'mtp_{layer_id - num_layers}'])
    return read_matched_file(layer_st_map, layer_indices, prefixes, extra_keys), False


def _convert_layer(layer_id):
    """Convert one layer and write its shard. Runs inside a pool worker."""
    cfg = _WORKER['config']
    num_layers = cfg['num_layers']
    total_layers = num_layers + cfg.get('num_nextn_predict_layers', 0)
    template = _WORKER['template']
    stats = _WORKER['stats']

    source, from_sharded = _read_layer_source(
        layer_id, total_layers, num_layers, _WORKER['sharded'], _WORKER['layer_st_map'])
    hf_entries = convert_weights(template, source, _WORKER['policy'], stats, cfg,
                                 verify_whole=not from_sharded)

    weight_map, written = {}, 0
    if hf_entries:
        shard_file = _shard_name(layer_id, total_layers)
        written = _flush_shard(hf_entries, shard_file, _WORKER['output_path'],
                               weight_map, layer_id, total_layers)
    # The template's leftover cache is per-process, so report it with the layer.
    pending = sorted(template.name_to_weight)
    template.name_to_weight.clear()
    return weight_map, written, stats.converted, stats.unclaimed, pending


def convert_model(input_path, output_path, config, policy, workers=1):
    """Convert an MF checkpoint to HF format, one shard per layer.

    Layers are independent — each reads only its own parameters and writes its
    own shard — so ``workers`` > 1 spreads them over processes, which pays off
    because the per-layer cost is dominated by reading and reassembling rank
    shards. Peak memory scales with ``workers``: each holds a whole layer, and
    under ``--dtype source`` a bf16 tensor is staged through fp32.
    """
    model_type = config['model_type']
    num_layers = config['num_layers']
    num_mtp = config.get('num_nextn_predict_layers', 0)
    total_layers = num_layers + num_mtp

    os.makedirs(output_path, exist_ok=True)
    template = build_template(model_type, config)
    stats = ConversionStats()

    logger.info(f"Loading MindSpore checkpoint from: {input_path}")
    logger.info(f"Model type: {model_type}, num_layers: {num_layers}, MTP: {num_mtp}")
    logger.info(f"Output dtype: {policy}")
    logger.info(f"Driving {len(template.weight_converters)} declared converters "
                f"({len(template.mf_name_to_converter)} MF name patterns)")

    layer_st_map, sharded = None, None
    if ShardedCheckpointReader.is_distributed_checkpoint(input_path):
        sharded = ShardedCheckpointReader(input_path)
        all_keys = set(sharded.names)
        logger.info(f"Distributed checkpoint: reassembling {len(all_keys)} parameters "
                    f"from their rank shards via metadata.json")
    else:
        layer_st_map, all_keys = build_layer_file_map(input_path)

    weight_map = {}
    total_size = 0
    pending = []

    if workers > 1:
        workers = min(workers, total_layers)
        threads = _threads_per_worker(workers)
        logger.info(f"Converting {total_layers} layers across {workers} worker processes "
                    f"({threads} compute threads each, {_available_cpus()} CPUs available)")
        # 'fork' matches the sibling converters under toolkit/weight_convert and skips
        # re-importing MindSpore in every worker (~20 s each). It does fork a process
        # that has already initialised MindSpore; that is exercised and produces
        # byte-identical output, with MF2HF_MP_START=spawn as an escape hatch.
        ctx = multiprocessing.get_context(os.environ.get('MF2HF_MP_START', 'fork'))
        init_args = (input_path, output_path, config,
                     _policy_to_spec(policy), layer_st_map, threads)
        with ctx.Pool(processes=workers, initializer=_worker_init,
                      initargs=init_args) as pool:
            for w_map, written, converted, unclaimed, layer_pending in pool.imap_unordered(
                    _convert_layer, range(total_layers)):
                weight_map.update(w_map)
                total_size += written
                stats.converted |= converted
                stats.unclaimed |= unclaimed
                pending.extend(layer_pending)
    else:
        for layer_id in range(total_layers):
            source, from_sharded = _read_layer_source(
                layer_id, total_layers, num_layers, sharded, layer_st_map)
            hf_entries = convert_weights(template, source, policy, stats, config,
                                         verify_whole=not from_sharded)
            if hf_entries:
                total_size += _flush_shard(
                    hf_entries, _shard_name(layer_id, total_layers), output_path,
                    weight_map, layer_id, total_layers)
        pending = sorted(template.name_to_weight)
        template.name_to_weight.clear()

    _save_index(output_path, weight_map, total_size)
    _report_tied_head(weight_map, config, template)
    stats.report(all_keys, sorted(set(pending)))
    logger.info("Conversion complete!")


def _save_index(output_path, weight_map, total_size):
    """Save model.safetensors.index.json.

    The map is sorted so the file does not depend on the order layers finished
    in: with ``workers`` > 1 they complete out of order, and an unsorted map
    would make two exports of one checkpoint differ byte for byte.
    """
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(weight_map.items())),
    }
    index_file = os.path.join(output_path, "model.safetensors.index.json")
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2)
    logger.info(f"Saved index: {index_file} ({len(weight_map)} tensors, {total_size} bytes)")


def hf_config_fields(model_type):
    """Field names the model's own config class declares.

    The YAML ``model:`` section mixes HF fields with MindFormers-only ones
    (``compute_dtype``, ``moe_grouped_gemm`` ...) that ``transformers`` does not
    understand. Filtering by the model's own ``config_class`` keeps this script
    free of any config mapping of its own, as the weight names come from
    ``weight_converters``.

    The result is a **subset** of a published config.json: fields the YAML never
    carries (``bos_token_id``, ``use_cache``, ``rope_scaling``) cannot be
    emitted, and a genuine HF field ``config_class`` does not declare is dropped
    with the MindFormers-only ones (``head_dim`` on ``Qwen3MoeConfig``). Closing
    either needs a per-model MF -> HF *config* converter mirroring
    ``weight_converters``; the framework has only the HF -> MF direction today.
    """
    cfg_cls = getattr(MODEL_CONVERTERS[model_type], 'config_class', None)
    if cfg_cls is None:
        return None
    fields = set()
    for klass in cfg_cls.__mro__:
        init = klass.__dict__.get('__init__')
        if not init:
            continue
        for name, param in inspect.signature(init).parameters.items():
            if name != 'self' and param.kind in (param.POSITIONAL_OR_KEYWORD,
                                                 param.KEYWORD_ONLY):
                fields.add(name)
    return fields


def save_hf_config(output_path, config, model_type):
    """Write a config.json holding only fields the HF config class declares."""
    model_section = dict(config.get('_raw_model_section') or {})
    if not model_section:
        return
    fields = hf_config_fields(model_type)
    if fields is None:
        logger.warning(f"{model_type} declares no config_class; writing the YAML "
                       f"'model' section unfiltered, MindFormers-only fields included.")
        fields = set(model_section)
    hf_config = {k: v for k, v in model_section.items() if k in fields}
    dropped = sorted(set(model_section) - set(hf_config))

    # HF metadata present in every published config.json but not a config_class field.
    arch = model_section.get('architectures')
    if arch:
        hf_config['architectures'] = [arch] if isinstance(arch, str) else arch
    hf_config['model_type'] = model_type
    if model_section.get('params_dtype'):
        hf_config['torch_dtype'] = str(model_section['params_dtype'])

    config_file = os.path.join(output_path, "config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(hf_config, f, indent=2)
    logger.info(f"Saved config: {config_file} ({len(hf_config)} fields; a subset of a "
                f"published config.json, see hf_config_fields). Dropped {len(dropped)} "
                f"field(s) the HF config class does not declare: {', '.join(dropped)}")


def _writes_into_input(input_path, output_path):
    """True if exporting to ``output_path`` would clobber the input checkpoint.

    Covers both "same directory" and "output nested inside input": in the latter
    the shards written first would be picked up as input by later layers.
    """
    src = os.path.realpath(input_path)
    dst = os.path.realpath(output_path)
    if os.path.isfile(src):
        src = os.path.dirname(src)
    return dst == src or dst.startswith(src + os.sep)


# =============================================================================
# Main entry point
# =============================================================================

def main():
    """Parse CLI arguments and run the requested conversion."""
    parser = argparse.ArgumentParser(
        description='Convert MindFormers weights to HuggingFace safetensors, one '
                    'shard per layer.')
    parser.add_argument('--yaml_config', type=str, required=True,
                        help='Path to the MindFormers YAML config file.')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to the input MF checkpoint (directory or .ckpt file).')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to the output directory for HF weights.')
    parser.add_argument('--model_type', type=str, default=None,
                        help='Override the model type read from the YAML config.')
    parser.add_argument('--dtype', type=str, default=None,
                        choices=sorted(DTYPE_MAP) + ['source'],
                        help='Output dtype for float weights; "source" keeps each tensor\'s '
                             'MF dtype, making the export bit-exact. Default: params_dtype.')
    parser.add_argument('--uniform_dtype', action='store_true',
                        help='Write every float tensor with --dtype. By default attention '
                             'sink, APE and HyperConnection params stay fp32, as published.')
    parser.add_argument('--no_config_json', action='store_true',
                        help='Do not write config.json into the output directory.')
    parser.add_argument('--max_worker', type=int, default=16,
                        help='Processes converting layers in parallel; 1 is serial. Peak '
                             'memory scales with it (one layer per worker). '
                             'MF2HF_MP_START=spawn avoids forking.')
    args = parser.parse_args()

    if not os.path.exists(args.yaml_config):
        logger.error(f"YAML config file not found: {args.yaml_config}")
        sys.exit(1)

    config = parse_yaml_config(args.yaml_config)
    model_type = config['model_type']
    logger.info(f"YAML: model_type={model_type}, architecture={config['architectures']}, "
                f"num_layers={config['num_layers']}")

    discover_model_converters()
    model_key = args.model_type or model_type
    if model_key not in MODEL_CONVERTERS:
        logger.error(f"Unsupported model type '{model_key}'; "
                     f"this tool exports {list(MODEL_CONVERTERS)}.")
        sys.exit(1)

    if not os.path.exists(args.input_path):
        logger.error(f"Input path not found: {args.input_path}")
        sys.exit(1)
    if _writes_into_input(args.input_path, args.output_path):
        logger.error(f"output_path '{args.output_path}' is the input checkpoint or "
                     f"lives inside it; exporting there would overwrite the weights "
                     f"being read. Choose a directory outside '{args.input_path}'.")
        sys.exit(1)

    os.makedirs(args.output_path, exist_ok=True)
    policy = DtypePolicy(resolve_dtype(config, args.dtype), uniform=args.uniform_dtype)
    convert_model(args.input_path, args.output_path, config, policy, workers=args.max_worker)
    if not args.no_config_json:
        save_hf_config(args.output_path, config, model_key)


if __name__ == '__main__':
    main()
