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
"""UTs for the YAML handling of toolkit/weight_convert/convert_mf_to_hf.py."""
import importlib.util
import os
import json
import subprocess
import sys

import numpy as np

import pytest

from mindformers.tools.check_rules import YAML_MAX_NESTING_DEPTH

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
CONVERTER = os.path.join(REPO_ROOT, "toolkit", "weight_convert", "convert_mf_to_hf.py")


def _load_converter():
    """Import the converter script as a module; it is not part of an installed package."""
    spec = importlib.util.spec_from_file_location("convert_mf_to_hf", CONVERTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _nested_mapping_yaml(depth):
    """Return a YAML document whose mappings nest ``depth`` levels deep."""
    lines = [f"{'  ' * level}level{level}:" for level in range(depth)]
    lines[-1] += " leaf"
    return "\n".join(lines) + "\n"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_parse_yaml_config_rejects_overly_nested_yaml(tmp_path):
    """
    Feature: convert_mf_to_hf.parse_yaml_config
    Description: Parse a YAML file nested one level deeper than YAML_MAX_NESTING_DEPTH.
    Expectation: The depth check rejects it with ValueError before yaml.safe_load builds it.
    """
    converter = _load_converter()
    yaml_path = tmp_path / "too_deep.yaml"
    yaml_path.write_text(_nested_mapping_yaml(YAML_MAX_NESTING_DEPTH + 1), encoding="utf-8")

    with pytest.raises(ValueError, match="YAML nesting depth"):
        converter.parse_yaml_config(str(yaml_path))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_parse_yaml_config_reads_model_section(tmp_path):
    """
    Feature: convert_mf_to_hf.parse_yaml_config
    Description: Parse an ordinary MindFormers YAML after the depth check has consumed the file once.
    Expectation: The model section is read in full (the file is rewound before yaml.safe_load).
    """
    converter = _load_converter()
    yaml_path = tmp_path / "model.yaml"
    yaml_path.write_text(
        "model:\n"
        "  model_type: deepseek_v3\n"
        "  num_hidden_layers: 4\n"
        "  n_routed_experts: 8\n"
        "  compute_dtype: float16\n",
        encoding="utf-8",
    )

    config = converter.parse_yaml_config(str(yaml_path))

    assert config["model_type"] == "deepseek_v3"
    assert config["num_layers"] == 4
    assert config["n_routed_experts"] == 8
    assert config["compute_dtype"] == "float16"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("model_type,layers,mtp", [("xing4_v4", 40, 1), ("xing4_v41", 45, 0)])
def test_xing4_dynamic_export_schema(tmp_path, model_type, layers, mtp):
    """
    Feature: Xing4 offline export.
    Description: Discover both dynamic schemas and export defaults under native HF names.
    Expectation: DSA and MTP rules match the selected model, without MF metadata in HF config.
    """
    converter = _load_converter()
    converter.discover_model_converters()
    config = {"model_type": model_type, "_raw_model_section": {"model_type": model_type}}
    template = converter.build_template(model_type, config)
    assert f"decoder.layers.{layers - 1}.self_attention.core_attention.indexer.linear_wk.weight" in \
        template.mf_name_to_converter
    assert any(name.startswith('mtp.') for name in template.mf_name_to_converter) == bool(mtp)
    assert all(op.expert_parallel_size == op.optimizer_parallel_size == 1 for op in template._expert_converters)
    converter.save_hf_config(str(tmp_path), config, model_type)
    hf = json.loads((tmp_path / 'config.json').read_text())
    assert hf['num_hidden_layers'] == layers and hf['num_nextn_predict_layers'] == mtp
    assert hf['model_type'] == ('xing4_0' if mtp else 'deepseek_v32')
    assert hf['scoring_func'] == 'sigmoid'
    assert 'architectures' in hf and 'params_dtype' not in hf


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_xing4_export_preserves_reference_metadata_and_mtp_aliases(tmp_path):
    """
    Feature: Native HF reload compatibility.
    Description: Export MTP aliases, native metadata and learned router corrections.
    Expectation: Shared weights are present, custom-code metadata survives and router bias stays FP32.
    """
    converter = _load_converter()
    converter.discover_model_converters()
    config = {'model_type': 'xing4_v4', '_raw_model_section': {'model_type': 'xing4_v4'}}
    template = converter.build_template('xing4_v4', config)
    value = np.arange(12, dtype=np.float32).reshape(3, 4)
    weights = template.add_mf_weight('embedding.word_embeddings.weight', value)
    np.testing.assert_array_equal(weights['model.layers.40.embed_tokens.weight'], value)
    reference = tmp_path / 'original.json'
    reference.write_text(json.dumps({'model_type': 'xing4_0', 'num_hidden_layers': 99,
                                     'auto_map': {'AutoConfig': 'configuration_xing4_0.Xing4_0Config'}}))
    converter.save_hf_config(str(tmp_path), config, 'xing4_v4', str(reference))
    exported = json.loads((tmp_path / 'config.json').read_text())
    assert exported['num_hidden_layers'] == 40 and exported['auto_map']['AutoConfig'].endswith('Xing4_0Config')
    policy = converter.DtypePolicy(converter.ms.bfloat16)
    assert policy.for_name('model.layers.3.mlp.gate.e_score_correction_bias') == converter.ms.float32
    reference.write_text(json.dumps({'model_type': 'deepseek_v32'}))
    with pytest.raises(ValueError, match='different model type'):
        converter.save_hf_config(str(tmp_path), config, 'xing4_v4', str(reference))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("flat_index", [False, True])
def test_sharded_reader_restores_two_axes(tmp_path, missing, flat_index):
    """
    Feature: Distributed checkpoint export.
    Description: Read four fragments split across two axes, with a replica and shuffled metadata.
    Expectation: Recover exact values, or reject an incomplete set of fragments.
    """
    converter = _load_converter()
    expected = np.arange(24, dtype=np.float32).reshape(4, 6)
    chunks, storage = [], {}
    for index, (row, col) in enumerate(((2, 3), (0, 3), (2, 0), (0, 0))):
        filename = f'rank-{index}.safetensors'
        tile = expected[row:row + 2, col:col + 3]
        converter.save_safetensors({'weight': converter.pack(tile, converter.ms.float32)}, str(tmp_path / filename))
        offset = [(row // 2) * 2 + col // 3] if flat_index else [row, col]
        chunks.append({'global_offset': offset, 'local_shape': [2, 3]})
        key = converter.get_sharded_tensor_shard_id('weight', offset)
        storage[key] = [{'file_name': filename}, {'file_name': 'unused-replica.safetensors'}]
    if missing:
        chunks.pop()
    metadata = {'state_dict_metadata': {'weight': {'global_shape': [4, 6], 'axis_fragmentations': [2, 2],
                                                'chunk': chunks}}, 'storage_data': storage}
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata))
    reader = converter.ShardedCheckpointReader(str(tmp_path))
    if missing:
        with pytest.raises(ValueError, match='chunks, expected'):
            reader.read('weight')
    else:
        actual, dtype = reader.read('weight')
        np.testing.assert_array_equal(actual, expected)
        assert dtype == converter.ms.float32


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('workers', [1, 2])
def test_xing4_forward_partial_checkpoint_policy(tmp_path, workers):
    """
    Feature: Offline HF to MF conversion.
    Description: Convert an embedding-only source with default and strict missing-weight policies.
    Expectation: Default conversion writes readable MF metadata; strict conversion rejects missing groups.
    """
    forward = __import__('toolkit.weight_convert.xing4.convert_xing4_hf_weight',
                         fromlist=['convert_checkpoint'])
    converter = _load_converter()
    source = tmp_path / 'source'
    source.mkdir()
    expected = np.arange(32, dtype=np.float32).reshape(8, 4)
    converter.save_safetensors({'model.embed_tokens.weight': converter.pack(expected, converter.ms.float32)},
                              str(source / 'model.safetensors'))
    config = {'model_type': 'xing4_v41', 'num_hidden_layers': 1, 'first_k_dense_replace': 1,
              'vocab_size': 8, 'hidden_size': 4}
    destination = tmp_path / 'mf'
    forward.convert_checkpoint(str(source), str(destination), config, workers=workers)
    actual, _ = converter.ShardedCheckpointReader(str(destination)).read('embedding.word_embeddings.weight')
    np.testing.assert_array_equal(actual, expected)
    common = forward.CommonInfo.load_common(str(destination / 'common.json'))
    assert common.global_step == common.consumed_samples == 0 and common.global_batch_size == 1
    mapping = json.loads((destination / 'param_name_map.json').read_text())
    assert set(mapping) == {'embedding.word_embeddings.weight'}
    header, _ = converter.read_safetensors_header(str(destination / mapping['embedding.word_embeddings.weight']))
    assert header['embedding.word_embeddings.weight']['shape'] == [8, 4]
    with pytest.raises(ValueError, match='Missing HF weights'):
        forward.convert_checkpoint(str(source), str(tmp_path / 'strict'), config, strict_load=True, workers=workers)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('size,fragments,flat', [(8, 2, True), (8, 2, False), (4, 4, True), (4, 4, False)])
def test_one_dimensional_checkpoint_offsets(tmp_path, size, fragments, flat):
    """Recover 1-D native shard indices and complete element offsets without ambiguity."""
    converter = _load_converter()
    expected = np.arange(size, dtype=np.float32)
    width = size // fragments
    chunks, storage = [], {}
    for index in reversed(range(fragments)):
        offset = [index if flat else index * width]
        filename = f'rank-{index}.safetensors'
        converter.save_safetensors({'weight': converter.pack(expected[index * width:(index + 1) * width],
                                                            converter.ms.float32)}, str(tmp_path / filename))
        chunks.append({'global_offset': offset, 'local_shape': [width]})
        storage[converter.get_sharded_tensor_shard_id('weight', offset)] = [{'file_name': filename}]
    metadata = {'state_dict_metadata': {'weight': {'global_shape': [size], 'axis_fragmentations': [fragments],
                                                'chunk': chunks}}, 'storage_data': storage}
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata))
    actual, _ = converter.ShardedCheckpointReader(str(tmp_path)).read('weight')
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_xing4_templates_do_not_share_ops():
    """Building another offline template must leave the first instance's geometry unchanged."""
    converter = _load_converter()
    converter.discover_model_converters()
    config = {'model_type': 'xing4_v41', 'num_hidden_layers': 4}
    first = converter.build_template('xing4_v41', config)
    first._expert_converters[0].expert_parallel_size = 8
    second = converter.build_template('xing4_v41', config)
    assert first._expert_converters[0].expert_parallel_size == 8
    assert not {id(op) for op in first.weight_converters} & {id(op) for op in second.weight_converters}


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('missing,overrides', [
    ('hc_eps', {}), ('index_topk', {'experimental_attention_variant': None})])
def test_xing4_optional_hf_fields(monkeypatch, missing, overrides):
    """Optional fields removed by the model schema do not break native config export."""
    from mindformers.models.xing4.configuration_xing4 import Xing4Config
    from toolkit.weight_convert.xing4.export_adapter import Xing4ExportAdapter
    defaults = dict(Xing4Config._HF_DEFAULTS)
    defaults.pop(missing)
    monkeypatch.setattr(Xing4Config, '_HF_DEFAULTS', defaults)
    result = Xing4ExportAdapter.export_hf_config({'model_type': 'xing4_v41', **overrides})
    assert result['model_type'] == 'deepseek_v32' and missing not in result


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_adapter_import_error_is_not_silenced(monkeypatch):
    """An unavailable adapter reports its cause while unrelated model exporters remain usable."""
    converter = _load_converter()
    original = converter.importlib.import_module
    cause = ImportError('broken adapter dependency')

    def importing(name):
        if name.endswith('.export_adapter'):
            raise cause
        return original(name)

    monkeypatch.setattr(converter.importlib, 'import_module', importing)
    warnings = []
    monkeypatch.setattr(converter.logger, 'warning', warnings.append)
    available = converter.discover_model_converters()
    assert available and any('export_adapter' in message and str(cause) in message for message in warnings)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('dtype,tag', [('bf16', 'BF16'), ('fp16', 'F16')])
def test_forward_router_correction_stays_fp32(tmp_path, dtype, tag):
    """Explicit low-precision conversion keeps learned router corrections in FP32."""
    from toolkit.weight_convert.xing4.convert_xing4_hf_weight import convert_checkpoint
    converter = _load_converter()
    source, destination = tmp_path / 'hf', tmp_path / 'mf'
    source.mkdir()
    correction = np.array([0.000123456, 0.000234567], np.float32)
    weights = {'model.layers.1.mlp.gate.weight': np.arange(8, dtype=np.float32).reshape(2, 4),
               'model.layers.1.mlp.gate.e_score_correction_bias': correction}
    converter.save_safetensors({name: converter.pack(value, converter.ms.float32) for name, value in weights.items()},
                              str(source / 'model.safetensors'))
    config = {'model_type': 'xing4_v41', 'num_hidden_layers': 2, 'first_k_dense_replace': 1,
              'hidden_size': 4, 'n_routed_experts': 2, 'num_experts_per_tok': 1}
    convert_checkpoint(str(source), str(destination), config, dtype=dtype)
    mapping = json.loads((destination / 'param_name_map.json').read_text())
    header, _ = converter.read_safetensors_header(str(destination / mapping['decoder.layers.1.mlp.expert_bias']))
    assert header['decoder.layers.1.mlp.expert_bias']['dtype'] == 'F32'
    assert header['decoder.layers.1.mlp.router.weight']['dtype'] == tag
    actual, _ = converter.ShardedCheckpointReader(str(destination)).read('decoder.layers.1.mlp.expert_bias')
    np.testing.assert_array_equal(actual, correction)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_direct_script_conversion_outside_repo(tmp_path):
    """Both CLI scripts find their checkout without requiring the repository in PYTHONPATH."""
    converter = _load_converter()
    source, mf_path, hf_path = (tmp_path / name for name in ('source', 'mf', 'hf'))
    source.mkdir()
    values = np.arange(32, dtype=np.float32).reshape(8, 4)
    converter.save_safetensors({'model.embed_tokens.weight': converter.pack(values, converter.ms.float32)},
                              str(source / 'model.safetensors'))
    spec = tmp_path / 'model.yaml'
    spec.write_text('model:\n  model_type: xing4_v41\n  num_hidden_layers: 1\n'
                    '  first_k_dense_replace: 1\n  vocab_size: 8\n  hidden_size: 4\n')
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(path for path in env.get('PYTHONPATH', '').split(os.pathsep)
                                      if path and os.path.realpath(path) != REPO_ROOT)
    for script, src, dst, extra in (
            ('xing4/convert_xing4_hf_weight.py', source, mf_path, ['--max_worker', '2']),
            ('convert_mf_to_hf.py', mf_path, hf_path, ['--max_worker', '1'])):
        command = [sys.executable, os.path.join(REPO_ROOT, 'toolkit/weight_convert', script),
                   '--yaml_config', str(spec), '--input_path', str(src), '--output_path', str(dst),
                   '--dtype', 'source', *extra]
        run = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90, check=False)
        assert run.returncode == 0, run.stdout + run.stderr
    index = json.loads((hf_path / 'model.safetensors.index.json').read_text())['weight_map']
    file = str(hf_path / index['model.embed_tokens.weight'])
    header, offset = converter.read_safetensors_header(file)
    actual, _ = converter.read_one_tensor(file, header, offset, 'model.embed_tokens.weight')
    np.testing.assert_array_equal(actual, values)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_forward_serial_parallel_outputs_match(tmp_path):
    """Serial and spawned workers produce the same tensors and loader metadata across source shards."""
    from toolkit.weight_convert.xing4.convert_xing4_hf_weight import convert_checkpoint
    converter = _load_converter()
    source = tmp_path / 'source'
    source.mkdir()
    shapes = {'model.embed_tokens.weight': (8, 4), 'model.norm.weight': (4,), 'lm_head.weight': (8, 4),
              'model.layers.0.self_attn.q_a_proj.weight': (64, 4),
              'model.layers.0.self_attn.kv_a_proj_with_mqa.weight': (576, 4),
              'model.layers.1.self_attn.indexer.wk.weight': (128, 4),
              'model.layers.1.mlp.gate.e_score_correction_bias': (2,)}
    index = {}
    for shard in (0, 1):
        filename = f'source-{shard}.safetensors'
        entries = {name: converter.pack(np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 32,
                                        converter.ms.float32)
                   for i, (name, shape) in enumerate(shapes.items()) if i % 2 == shard}
        converter.save_safetensors(entries, str(source / filename))
        index.update(dict.fromkeys(entries, filename))
    (source / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': index}))
    config = {'model_type': 'xing4_v41', 'num_hidden_layers': 2, 'first_k_dense_replace': 1,
              'hidden_size': 4, 'vocab_size': 8, 'q_lora_rank': 64,
              'n_routed_experts': 2, 'num_experts_per_tok': 1}
    serial, parallel = tmp_path / 'serial', tmp_path / 'parallel'
    convert_checkpoint(str(source), str(serial), config, workers=1)
    convert_checkpoint(str(source), str(parallel), config, workers=2)
    for name in ('metadata.json', 'param_name_map.json', 'common.json'):
        assert (serial / name).read_bytes() == (parallel / name).read_bytes()
    expected, actual = converter.ShardedCheckpointReader(str(serial)), converter.ShardedCheckpointReader(str(parallel))
    assert expected.names == actual.names
    for name in expected.names:
        wanted, wanted_dtype = expected.read(name)
        value, value_dtype = actual.read(name)
        assert wanted_dtype == value_dtype
        np.testing.assert_array_equal(value, wanted)
