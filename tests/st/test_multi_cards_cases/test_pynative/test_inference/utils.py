# Copyright 2026 Huawei Technologies Co, Ltd
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
"""Utils for PyNative inference multi-cards tests."""

import json
import os
import random
import shutil
import subprocess

import yaml

from tests.st.test_multi_cards_cases.test_pynative.test_models.test_deepseek3.utils import (
    save_model_checkpoints,
)

SEED = 42
VOCAB_SIZE = 64

# Inference-specific overrides applied on top of the shared training yaml
# (test_models/test_deepseek3/pynative_ds3.yaml). The miniature vocab/layer
# count and float32 compute keep greedy decoding deterministic across
# parallel strategies; seq_length must stay >= 2048 for the CP compressed
# attention mask. Training-only sections are dropped because predict mode
# skips all training-side initialization.
INFERENCE_OVERRIDES = {
    "model": {
        "vocab_size": VOCAB_SIZE,
        "num_hidden_layers": 4,
        "seq_length": 2048,
        "compute_dtype": "float32",
    },
    "parallelism": {
        "pipeline_parallel_layers_per_stage": "auto",
        "sequence_parallel": False,
    },
    "training": {
        "steps": 1,
        "local_batch_size": 1,
        "global_batch_size": 1,
    },
    "checkpoint": {"no_save_optim": True},
}


# Ten fixed English prompts; every word must exist in the generated
# WordLevel vocab so tokenization is deterministic.
PROMPTS = [
    "the model reads the input",
    "a small test runs fast",
    "we check the output text",
    "this line has five words",
    "greedy decode picks one token",
    "the rank writes the result file",
    "parallel ranks share the batch",
    "the last layer returns logits",
    "each step appends one new token",
    "the final answer is correct",
]


def build_wordlevel_tokenizer(vocab_dir):
    """Write a deterministic WordLevel tokenizer.json into the checkpoint dir.

    The inference path loads the tokenizer from ``checkpoint.load_path``,
    so the generated file is placed next to the saved ``iteration_1`` weights.
    """
    os.makedirs(vocab_dir, exist_ok=True)
    # Explicit class + special tokens so AutoTokenizer does not fall back to
    # the AutoConfig branch and resolves pad/eos ids from the vocab.
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "pad_token": "<pad>",
        "eos_token": "<unk>",
    }
    tokenizer_config_path = os.path.join(vocab_dir, "tokenizer_config.json")
    existing_config = None
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, "r") as fp:
            existing_config = json.load(fp)
    if existing_config != tokenizer_config:
        with open(tokenizer_config_path, "w") as fp:
            json.dump(tokenizer_config, fp)

    tokenizer_path = os.path.join(vocab_dir, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        return tokenizer_path

    words = sorted({
        word for prompt in PROMPTS for word in prompt.split()
    })
    vocab = {"<unk>": 0, "<pad>": 1}
    for word in words:
        vocab[word] = len(vocab)
    # Pad the vocab up to the model vocab_size so every id is valid.
    for idx in range(len(vocab), VOCAB_SIZE):
        vocab[f"<extra_{idx}>"] = idx

    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": 0, "content": "<unk>", "single_word": False,
             "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 1, "content": "<pad>", "single_word": False,
             "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"},
    }
    with open(tokenizer_path, "w") as fp:
        json.dump(tokenizer, fp)
    return tokenizer_path


def build_prompts_file(prompts_path):
    """Write the fixed prompt list as an instruction-only jsonl file."""
    if os.path.exists(prompts_path):
        return prompts_path
    with open(prompts_path, "w") as fp:
        for prompt in PROMPTS:
            fp.write(json.dumps({"instruction": prompt}) + "\n")
    return prompts_path


def build_case_config(base_config, local_config_path, checkpoint_path, updates):
    """Write a deterministic inference case config and return its path."""
    with open(base_config, "r") as fp:
        configs = yaml.safe_load(fp)

    configs["run_mode"] = "predict"
    configs["checkpoint"]["load_path"] = checkpoint_path
    configs["training"]["seed"] = SEED
    configs["training"]["deterministic"] = True
    configs.pop("recompute", None)
    prompts_path = os.path.join(
        os.path.dirname(os.path.abspath(local_config_path)), "prompts.jsonl"
    )
    build_prompts_file(prompts_path)
    configs["inference"] = {
        "max_new_tokens": 8,
        "batch_size": 4,
        "input_data": prompts_path,
        "output": local_config_path.replace(".yaml", "_result.jsonl"),
    }

    # Case updates come last so per-case parallelism wins over the defaults.
    merged = {k: dict(v) for k, v in INFERENCE_OVERRIDES.items()}
    for section, values in updates.items():
        merged.setdefault(section, {}).update(values)
    for section, values in merged.items():
        configs.setdefault(section, {}).update(values)

    with open(local_config_path, "w") as fp:
        yaml.dump(configs, fp, indent=2)
    return local_config_path


def run_inference_msrun(run_script_path, config_path, log_dir, worker_num):
    """Run inference with msrun and return the worker log directory."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    port_id = int(os.environ.get("ASCEND_PORT_ID", random.randint(50000, 65535)))
    log_path = os.path.join(cur_dir, log_dir)
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    cmd = [
        "msrun",
        f"--worker_num={worker_num}",
        f"--local_worker_num={worker_num}",
        f"--master_port={port_id}",
        f"--log_dir={log_path}",
        "--join=True",
        f"{run_script_path}",
        "--config",
        f"{config_path}",
    ]
    env = os.environ.copy()
    socket_start = min(port_id + 1, 65400)
    socket_end = min(socket_start + worker_num + 32, 65535)
    env.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", f"{socket_start}-{socket_end}")
    result = subprocess.run(
        cmd, shell=False, capture_output=True, text=True, check=False, env=env,
    )
    assert result.returncode == 0, (
        f"Inference msrun script failed with exit code {result.returncode}.\n"
        f"Stdout:\n{result.stdout}\nStderr:\n{result.stderr}"
    )
    return log_path


def run_inference_python(run_script_path, config_path):
    """Run single-card inference with Python directly."""
    cmd = ["python", run_script_path, "--config", config_path]
    result = subprocess.run(
        cmd, shell=False, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, (
        f"Inference script failed with exit code {result.returncode}.\n"
        f"Stdout:\n{result.stdout}\nStderr:\n{result.stderr}"
    )


def collect_generated_ids(result_path):
    """Read the rank-0 written jsonl and return {sample_idx: generated_ids}."""
    samples = {}
    with open(result_path, "r") as fp:
        for line in fp:
            payload = json.loads(line)
            if "error" in payload:
                raise AssertionError(f"Inference failed for sample: {payload}")
            samples[payload["sample_idx"]] = payload["generated_ids"]
    assert samples, f"No generated samples found in {result_path}"
    return samples


def build_inference_base_config():
    """Materialize the shared training yaml with inference overrides applied.

    Returns the path of a local yaml (``pynative_infer_base.yaml``) used both
    for checkpoint generation (miniature vocab/layers) and case configs, so
    the saved weights match the model shape the inference runs build.
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    train_config = os.path.normpath(os.path.join(
        cur_dir, os.pardir, "test_models", "test_deepseek3", "pynative_ds3.yaml"
    ))
    local_config = os.path.join(cur_dir, "pynative_infer_base.yaml")
    with open(train_config, "r") as fp:
        configs = yaml.safe_load(fp)
    configs["run_mode"] = "predict"
    configs.pop("recompute", None)
    for section, values in INFERENCE_OVERRIDES.items():
        configs.setdefault(section, {}).update(values)
    with open(local_config, "w") as fp:
        yaml.dump(configs, fp, indent=2)
    return local_config


def run_inference_case(config_name, log_dir, worker_num, updates=None, run_reference=True):
    """Prepare deterministic weights/tokenizer/config, run one inference case.

    Returns ``(multi_ids, reference_ids)``; ``reference_ids`` is ``None`` when
    ``run_reference`` is False. The single-card reference runs the same config
    with all parallel degrees collapsed to 1, so parallel runs can be compared
    token-by-token against it.
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    base_config = build_inference_base_config()
    checkpoint_path = os.path.join(cur_dir, "checkpoints")
    run_script_path = os.path.join(cur_dir, "run_inference.py")

    save_model_checkpoints(base_config, checkpoint_path)
    build_wordlevel_tokenizer(checkpoint_path)

    local_config_path = os.path.join(cur_dir, config_name)
    build_case_config(base_config, local_config_path, checkpoint_path, updates or {})

    reference_ids = None
    if run_reference:
        ref_config = os.path.join(cur_dir, config_name.replace(".yaml", "_ref.yaml"))
        ref_updates = dict(updates or {})
        ref_updates["parallelism"] = {
            "data_parallel_shard": 1,
            "tensor_parallel": 1,
            "context_parallel": 1,
            "pipeline_parallel": 1,
        }
        build_case_config(base_config, ref_config, checkpoint_path, ref_updates)
        run_inference_python(run_script_path, ref_config)
        reference_ids = collect_generated_ids(
            os.path.join(cur_dir, config_name.replace(".yaml", "_ref_result.jsonl"))
        )

    log_path = run_inference_msrun(run_script_path, local_config_path, log_dir, worker_num)
    multi_ids = collect_generated_ids(local_config_path.replace(".yaml", "_result.jsonl"))
    return multi_ids, reference_ids, log_path


def assert_parallel_matches_reference(multi_ids, reference_ids):
    """Assert every multi-card sample matches the single-card reference."""
    assert set(multi_ids) == set(reference_ids), (
        f"Sample set mismatch: multi={sorted(multi_ids)}, "
        f"reference={sorted(reference_ids)}"
    )
    for idx in sorted(reference_ids):
        assert multi_ids[idx] == reference_ids[idx], (
            f"Sample {idx} diverges from single-card reference:\n"
            f"multi={multi_ids[idx]}\nreference={reference_ids[idx]}"
        )
