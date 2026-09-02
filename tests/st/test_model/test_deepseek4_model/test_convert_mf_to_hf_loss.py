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
"""Test that exporting DeepSeek-V4 weights to HuggingFace format preserves loss.

A network is populated with known weights and saved through
``mindformers.checkpoint.checkpoint.save_checkpoint`` — the same call the training
loop makes. That checkpoint is exported by
``toolkit/weight_convert/convert_mf_to_hf.py``, loaded back into a second network
via ``load_hf_checkpoint``, and the loss on a fixed batch is compared with the
loss the first network produced.

Equal loss means the whole path is sound: names, layouts and values all survived
the export, and they survived it in the sense the model actually consumes them —
which a weight-level comparison alone cannot show.

The losses must match exactly. The export runs with ``--dtype source`` so each
tensor keeps the dtype it was saved with, making the weight round trip
bit-exact, and the run is put in deterministic mode so the two forward passes
reduce in the same order. If this ever fails, compare the reloaded network's
parameters with the saved checkpoint first: identical parameters but different
loss points at determinism or at the load path casting dtypes, not at the
converter.

Two notes on how the checkpoint is produced:

* The weights are seeded pseudo-randomly rather than trained. Building this model
  outside the trainer leaves almost every parameter zero-filled, and zero weights
  would hide exactly the bugs this test exists to catch — a transposed or
  misplaced tensor changes nothing when every tensor is zero. Random weights make
  the loss sensitive to every element. Only the numbers' provenance differs; the
  checkpoint still goes through the production save path.
* The DSA indexer needs two things the trainer normally arranges and a bare
  single-card build does not. ``hyper_parallel`` patches torch-style autograd
  onto MindSpore tensors — including ``Tensor.detach``, which the indexer uses —
  from ``enable_mindspore_backward_compat()``, reached in production via
  ``fully_shard`` / pipeline setup; the test calls it directly. And
  ``dsa_indexer_loss_coeff`` defaults to None, which the indexer's KL loss then
  multiplies by. With both in place the indexer runs, so ``compress_ratios``
  uses 4 and its weights are covered by the forward pass, not just exported.

Requires one Ascend device.
"""

import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest
import yaml

import mindspore as ms

from mindformers.checkpoint.checkpoint import CommonInfo, save_checkpoint, load_hf_checkpoint
from mindformers.models.deepseek4.configuration_deepseek_v4 import DeepseekV4Config

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 4))
CONVERTER = os.path.join(REPO_ROOT, "toolkit", "weight_convert", "convert_mf_to_hf.py")
# The exporter is a separate deliverable; without it this case has nothing to
# exercise, so it skips rather than failing on a missing file.

SEQ_LEN = 64
BATCH = 2
WEIGHT_SCALE = 0.02

# Geometry of the tiny model. DSv4 hybrid attention requires
# qk_head_dim + qk_pos_emb_head_dim == v_head_dim, hence head_dim == v_head_dim.
MODEL_KWARGS = {
    "vocab_size": 64,
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "head_dim": 64,
    "v_head_dim": 64,
    "qk_rope_head_dim": 16,
    "q_lora_rank": 32,
    "o_lora_rank": 32,
    "o_groups": 2,
    "moe_intermediate_size": 32,
    "n_routed_experts": 4,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    # Required: the converter handles mlp.experts.weight1/weight2, which is the
    # stacked grouped-GEMM layout. Without it the experts are separate parameters
    # and ExpertsConvertOp has nothing to convert.
    "moe_grouped_gemm": True,
    "num_nextn_predict_layers": 0,
    "num_hash_layers": 1,
    # 0 = plain MLA, 4 = compressor + DSA indexer. Mixing both exercises the two
    # attention variants a real checkpoint contains.
    "compress_ratios": [0, 4, 0, 4],
    # Must be set whenever the indexer runs: it defaults to None and the indexer
    # multiplies its KL loss by it. Zero keeps the loss out of the comparison
    # while still routing attention through the indexer's weights.
    "dsa_indexer_loss_coeff": 0.0,
    # The fused DSA kernels need aclnnSparseFlashMla, which is absent from some
    # CANN builds. The unfused path computes the same thing and keeps this test
    # runnable anywhere; kernel selection is irrelevant to weight conversion.
    "apply_dsa_kernel_fusion": False,
    "max_position_embeddings": 256,
    "seq_length": SEQ_LEN,
    "params_dtype": "bfloat16",
    "compute_dtype": "bfloat16",
}


def _build_network():
    """Build the DeepSeek-V4 training network, as run_mindformer would."""
    # pylint: disable=import-outside-toplevel
    from mindformers.models.deepseek4.modeling_deepseek_v4 import PyNativeDeepseekV4ForCausalLM

    assert PyNativeDeepseekV4ForCausalLM is not None, (
        "PyNativeDeepseekV4ForCausalLM is unavailable; check the hyper_parallel install")
    return PyNativeDeepseekV4ForCausalLM(DeepseekV4Config(**MODEL_KWARGS))


def _seed_weights(network, seed):
    """Fill every trainable floating-point parameter with pseudo-random data.

    Non-trainable buffers (``q_rms_gamma``, the HyperConnection ``rms_weight``,
    ``tokens_per_expert``) are left alone: they are constants rather than
    weights, the HuggingFace checkpoint has no counterpart for them, and
    ``weight_converters`` therefore does not declare them. Both networks build
    them identically, so leaving them untouched keeps the loss comparison a
    statement about the exported weights alone.
    """
    rng = np.random.default_rng(seed)
    filled = 0
    for name, param in sorted(network.parameters_dict().items()):
        if param.dtype == ms.int32:
            # router.tid2eid is a token-id -> expert-id lookup table allocated
            # with mint.empty. Left alone it holds whatever was in that memory,
            # and out-of-range expert ids make the MoE gather read out of bounds
            # — a nan loss, or a "GatherElementsV2 ... ub address out of bounds"
            # fault. Zero is a valid expert id for every configuration.
            param.set_data(ms.Tensor(np.zeros(param.shape, np.int32), param.dtype))
            continue
        if param.requires_grad:
            values = (rng.standard_normal(param.shape) * WEIGHT_SCALE).astype(np.float32)
            filled += 1
        else:
            # Buffers are allocated with mint.empty, i.e. uninitialised memory.
            # They must be given defined values or the run is not reproducible —
            # and garbage in tokens_per_expert drives the grouped-GEMM grouping
            # out of bounds. Norm gains are ones, counters are zeros, and both
            # networks get the same thing so they cancel in the comparison.
            fill = 1.0 if ("gamma" in name or "rms_weight" in name) else 0.0
            values = np.full(param.shape, fill, dtype=np.float32)
        param.set_data(ms.Tensor(values, param.dtype))
    assert filled > 0, "network exposed no trainable floating-point parameters"
    return filled


def _buffer_names(network):
    """Names of the non-trainable parameters, which are not part of the export."""
    return {name for name, param in network.parameters_dict().items()
            if not param.requires_grad}


def _unconverted_names(stdout):
    """Parse the converter's report of weights it could not place."""
    return set(re.findall(r"^  - (\S+)\s+\(", stdout, flags=re.M))


def _fixed_batch():
    """A deterministic batch, identical for every network instance."""
    rng = np.random.default_rng(1234)
    input_ids = rng.integers(0, MODEL_KWARGS["vocab_size"], size=(BATCH, SEQ_LEN + 1))
    tokens = ms.Tensor(input_ids[:, :-1].astype(np.int32))
    labels = ms.Tensor(input_ids[:, 1:].astype(np.int32))
    loss_mask = ms.Tensor(np.ones((BATCH, SEQ_LEN), np.float32))
    return tokens, labels, loss_mask


def _eval_loss(network, batch):
    """Return the loss for one forward pass.

    The network must stay in training mode: GPTModel returns logits instead of a
    loss when ``self.training`` is false. Dropout is disabled in this config, so
    the pass is still deterministic.
    """
    tokens, labels, loss_mask = batch
    network.set_train(True)
    loss = network(input_ids=tokens, labels=labels, loss_mask=loss_mask)
    if isinstance(loss, (tuple, list)):
        loss = loss[0]
    return float(loss.astype(ms.float32).mean())


def _write_yaml(path):
    """Write the config the converter reads, mirroring MODEL_KWARGS."""
    model = dict(MODEL_KWARGS)
    model.update({
        "model_type": "deepseek_v4",
        "architectures": "DeepseekV4ForCausalLM",
        "tie_word_embeddings": False,
    })
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"model": model}, f)


def _run_converter(yaml_path, input_path, output_path):
    """Invoke the offline converter and return its stdout."""
    proc = subprocess.run(
        [sys.executable, CONVERTER,
         "--yaml_config", str(yaml_path),
         "--input_path", str(input_path),
         "--output_path", str(output_path),
         # Keep each tensor's dtype so the round trip cannot lose precision.
         "--dtype", "source"],
        capture_output=True, text=True, cwd=REPO_ROOT, check=False)
    assert proc.returncode == 0, f"converter failed:\n{proc.stdout}\n{proc.stderr[-3000:]}"
    return proc.stdout


def _saved_weight_dir(root):
    """Locate the directory save_checkpoint wrote the model safetensors into."""
    for dir_path, _, file_names in os.walk(root):
        if any(f.endswith(".safetensors") for f in file_names):
            return dir_path
    raise AssertionError(f"no safetensors written under {root}")


def _save_and_export(network, tmp_path):
    """Save ``network`` through the training save path and export it to HF format.

    Returns (mf checkpoint dir, exported HF dir). Also checks the converter's
    report: a trainable weight it could not place would be invisible to the loss
    whenever the forward pass does not read it, so every name it reports must be
    a non-trainable buffer.
    """
    save_root = tmp_path / "mf_ckpt"
    save_checkpoint(iteration=0, network=network, save_checkpoint_path=str(save_root),
                    common_info=CommonInfo(epoch_num=0, step_num=0, global_step=0,
                                           loss_scale=1.0, global_batch_size=BATCH,
                                           consumed_samples=0, ckpt_status=0))
    mf_dir = _saved_weight_dir(str(save_root))

    yaml_path = tmp_path / "deepseek_v4_tiny.yaml"
    hf_dir = tmp_path / "hf_out"
    _write_yaml(yaml_path)
    stdout = _run_converter(yaml_path, mf_dir, hf_dir)

    unexpected = _unconverted_names(stdout) - _buffer_names(network)
    assert not unexpected, (
        f"the converter left trainable weights unconverted: {sorted(unexpected)}\n{stdout}")

    index_file = os.path.join(hf_dir, "model.safetensors.index.json")
    with open(index_file, encoding="utf-8") as f:
        assert json.load(f)["weight_map"], "export produced an empty index"
    return mf_dir, hf_dir


@pytest.mark.skipif(not os.path.exists(CONVERTER),
                    reason=f"offline exporter not present: {CONVERTER}")
@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_exported_weights_reproduce_loss(tmp_path):
    """
    Feature: Offline MF -> HF weight conversion for DeepSeek-V4.
    Description: Save a populated network the way training saves it, export the
                 checkpoint to HuggingFace format, load the export into a second
                 network holding different weights, and re-run the same batch.
    Expectation: The reloaded network reports exactly the source network's loss,
                 while the same network before loading reports a different one —
                 so the match is attributable to the weights, not to the batch.
    """
    # Determinism is required for an exact comparison: without it, Ascend kernels
    # may reduce in a different order between the two forward passes.
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", deterministic="ON")
    # pylint: disable=import-outside-toplevel
    from hyper_parallel.platform.mindspore.autograd_compat import (
        enable_mindspore_backward_compat)
    # Production reaches this through fully_shard / pipeline setup; a bare
    # single-card build has to ask for it. Without it the DSA indexer fails on
    # the missing Tensor.detach.
    enable_mindspore_backward_compat()
    ms.set_seed(0)

    batch = _fixed_batch()

    # 1. Populate a network and record its loss.
    source = _build_network()
    _seed_weights(source, seed=42)
    loss_source = _eval_loss(source, batch)
    assert np.isfinite(loss_source) and loss_source != 0.0, (
        f"source loss {loss_source} is degenerate; the batch or the weights are "
        f"not exercising the model")

    # 2-3. Save the way training does, then export to HuggingFace format.
    mf_dir, hf_dir = _save_and_export(source, tmp_path)

    # 4. A network holding other weights must disagree, otherwise a load that did
    #    nothing at all would still satisfy the comparison below.
    target = _build_network()
    _seed_weights(target, seed=7)
    loss_before_load = _eval_loss(target, batch)
    # Finiteness matters as much as difference: a nan would satisfy "!=" without
    # establishing anything about what the load did.
    assert np.isfinite(loss_before_load), (
        f"loss before load is {loss_before_load}; a non-finite baseline cannot "
        f"show that loading changed anything")
    assert loss_before_load != loss_source, (
        f"loss {loss_before_load} already equals the source loss {loss_source}; "
        f"the test cannot distinguish a successful load")

    # 5. Load the export back and compare.
    load_hf_checkpoint(str(hf_dir), target)
    loss_reloaded = _eval_loss(target, batch)

    print(f"\nloss source      = {loss_source!r}"
          f"\nloss before load = {loss_before_load!r}"
          f"\nloss reloaded    = {loss_reloaded!r}")

    assert loss_reloaded == loss_source, (
        f"loss changed across the export round trip: "
        f"source {loss_source!r}, reloaded {loss_reloaded!r}, "
        f"delta {abs(loss_reloaded - loss_source)!r}. The round trip is expected "
        f"to be bit-exact; compare the reloaded parameters against {mf_dir} to "
        f"see whether a weight or the forward pass is at fault.")
