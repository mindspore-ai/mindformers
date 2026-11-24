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
"""Convert Xing4 HF weights on CPU, one layer at a time, using the runtime schema."""
import argparse
import json
import multiprocessing
import sys
from contextlib import ExitStack
from pathlib import Path

# Select this checkout before importing MindFormers when invoked as a script.
# pylint: disable=wrong-import-position
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mindformers.checkpoint.checkpoint import CommonInfo
from mindformers.checkpoint.converter.convert_op import ExpertsConvertOp
from mindformers.checkpoint.metadata import save_metadata, tensor_to_ms_type
from mindformers.checkpoint.sharded_tensor import build_sharded_tensor
from mindformers.tools.logger import logger
from toolkit.weight_convert.convert_mf_to_hf import (
    DTYPE_MAP, DtypePolicy, _layer_key, _writes_into_input, _threads_per_worker, pack, parse_yaml_config,
    read_one_tensor, read_safetensors_header, save_safetensors,
)
from toolkit.weight_convert.xing4.export_adapter import Xing4ExportAdapter
# pylint: enable=wrong-import-position


_WORKER = {}


def _layer_groups(template):
    layers = {}
    for op in template.weight_converters:
        layers.setdefault(_layer_key(op.mf_names[0]), []).append(op)
    return list(layers.values())


def _init_worker(source, target, config, dtype, strict_load, names, headers, workers, template=None):
    """Build process-local rules and readers; no MindSpore dtype objects cross processes."""
    if workers > 1:
        _threads_per_worker(workers)
    template = template or Xing4ExportAdapter.build_weight_template(config)
    _WORKER.clear()
    _WORKER.update(source=Path(source), target=Path(target), template=template,
                   layers=_layer_groups(template), dtype=dtype, strict_load=strict_load,
                   names=names, headers=headers,
                   policy=DtypePolicy(None if dtype == 'source' else DTYPE_MAP[dtype]))


def _convert_layer(shard_id):
    """Write one independent shard and return only small, serializable metadata."""
    source, target = _WORKER['source'], _WORKER['target']
    template, policy = _WORKER['template'], _WORKER['policy']
    names, headers = _WORKER['names'], _WORKER['headers']
    dtype = _WORKER['dtype']
    entries, info, consumed = {}, {}, set()
    filename = f"model-{shard_id + 1:05d}-of-{len(_WORKER['layers']):05d}.safetensors"
    for op in _WORKER['layers'][shard_id]:
        required = ([name.format(expert) for name in op.hf_names for expert in range(op.num_moe_experts)]
                    if isinstance(op, ExpertsConvertOp) else op.hf_names)
        missing = set(required) - names.keys()
        if missing:
            message = f'Missing HF weights: {sorted(missing)[:10]}; check model type and DSA/MTP settings'
            if _WORKER['strict_load']:
                raise ValueError(message)
            logger.warning(message)
            continue
        weights, dtypes = {}, set()
        for name in required:
            file = names[name]
            if file not in headers:
                headers[file] = read_safetensors_header(str(source / file))
            weights[name], source_dtype = read_one_tensor(str(source / file), *headers[file], name)
            dtypes.add(source_dtype)
        if len(dtypes) != 1 and dtype == 'source':
            raise ValueError(f'Mixed source dtypes within conversion group: {required[:2]}')
        source_dtype = dtypes.pop() if dtype == 'source' else policy.default
        for name, array in template._numpy_group(op, weights).items():
            out_dtype = policy.for_name(name, source_dtype)
            if dtype != 'source' and name.endswith('.expert_bias'):
                out_dtype = DTYPE_MAP['fp32']
            entries[name] = pack(array, out_dtype)
            info[name] = (str(out_dtype), tuple(array.shape))
        consumed.update(required)
    if entries:
        save_safetensors(entries, str(target / filename))
        logger.info(f'Saved {filename}: {len(entries)} MF tensors')
    return filename, info, consumed


def convert_checkpoint(input_path, output_path, config, dtype='source', strict_load=False, workers=1):
    """Write unsharded MF safetensors and loader-compatible metadata."""
    source, target = Path(input_path), Path(output_path)
    if workers < 1:
        raise ValueError('max_worker must be positive')
    if _writes_into_input(str(source), str(target)):
        raise ValueError('output_path must be outside the input checkpoint')
    target.mkdir(parents=True, exist_ok=True)
    index_path = source / 'model.safetensors.index.json'
    headers = {}
    if index_path.exists():
        names = json.loads(index_path.read_text())['weight_map']
    else:
        names = {}
        for file in sorted(source.glob('*.safetensors')):
            header = read_safetensors_header(str(file))
            headers[file.name] = header
            for name in header[0]:
                if name != '__metadata__':
                    if name in names:
                        raise ValueError(f'Duplicate HF tensor: {name}')
                    names[name] = file.name
    if not names:
        raise ValueError('No HF safetensors weights found')
    template = Xing4ExportAdapter.build_weight_template(config)
    layers = _layer_groups(template)
    workers = min(workers, len(layers))
    init_args = (str(source), str(target), config, dtype, strict_load, names, headers, workers)
    metas, locations, weight_map, consumed = {}, [], {}, set()
    logger.info(f'Converting {len(layers)} shards with {workers} process(es)')
    with ExitStack() as stack:
        if workers > 1:
            pool = stack.enter_context(multiprocessing.get_context('spawn').Pool(
                workers, initializer=_init_worker, initargs=init_args))
            results = pool.imap(_convert_layer, range(len(layers)))
        else:
            _init_worker(*init_args, template=template)
            results = map(_convert_layer, range(len(layers)))
        for filename, info, used in results:
            for name, (dtype_name, shape) in info.items():
                out_dtype = tensor_to_ms_type[dtype_name.lower()]
                offset = (0,)
                ndim = len(shape)
                metas[name] = build_sharded_tensor(name, out_dtype, shape, shape, (1,) * ndim, offset)
                locations.append((filename, 0, [0], (name, offset)))
                weight_map[name] = filename
            consumed.update(used)
    unclaimed = set(names) - consumed - template.shared_weight_aliases.keys()
    if unclaimed:
        logger.warning(f'HF weights not converted: {sorted(unclaimed)[:10]} ({len(unclaimed)} total)')
    if not weight_map:
        raise ValueError('No weights were converted; check the source checkpoint and model configuration')
    save_metadata({0: metas}, locations, str(target / 'metadata.json'))
    (target / 'param_name_map.json').write_text(json.dumps(weight_map, indent=2))
    CommonInfo(epoch_num=0, step_num=0, global_step=0, loss_scale=1.0,
               global_batch_size=1, consumed_samples=0, ckpt_status=0).save_common(str(target / 'common.json'))
    logger.info(f'Converted {len(consumed)} HF tensors to {len(weight_map)} MF tensors')


def main():
    """Parse the model YAML and export HF weights to an unsharded MF checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yaml_config', required=True)
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--dtype', choices=sorted(DTYPE_MAP) + ['source'], default='source')
    parser.add_argument('--strict_load', action='store_true', help='Fail instead of warning on missing groups')
    parser.add_argument('--max_worker', type=int, default=1,
                        help='Layer conversion processes. Default: 1; each holds one layer in memory.')
    args = parser.parse_args()
    convert_checkpoint(args.input_path, args.output_path, parse_yaml_config(args.yaml_config),
                       dtype=args.dtype, strict_load=args.strict_load, workers=args.max_worker)


if __name__ == '__main__':
    main()
