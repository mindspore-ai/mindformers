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
"""Run DeepSeek-V3 training with recompute callsite probes."""
import argparse
from collections import defaultdict
import functools
import json
import random

import numpy as np
import mindspore as ms
from mindspore.mint.distributed import get_rank

from mindformers.pynative.base_models.gpt import parallelize as gpt_parallelize
from mindformers.pynative.distributed.activation_checkpoint import is_in_recompute
from mindformers.pynative.trainer import Trainer as PynativeTrainer

SEED = 42
PROBE_PREFIX = "RECOMPUTE_PROBE_JSON="
_COUNTS = defaultdict(lambda: {"forward": 0, "recompute": 0})


def _record(name):
    _COUNTS[name]["recompute" if is_in_recompute() else "forward"] += 1


def _resolve_attr(root, path):
    target = root
    for part in path.split("."):
        target = getattr(target, part)
    return target


def _probe_cell(root, path, layer_id):
    cell = _resolve_attr(root, path)
    key = f"layer{layer_id}.{path}"
    cell.register_forward_hook(lambda _c, _a, o, k=key: (_record(k), o)[1])


def _wrap_recording(original, key):
    @functools.wraps(original)
    def _wrapped(*args, **kwargs):
        _record(key)
        return original(*args, **kwargs)
    return _wrapped


def _probe_callable(root, path, layer_id):
    """Wrap the callable at ``path`` so calls bump the replay counter."""
    if "." in path:
        owner_path, attr = path.rsplit(".", 1)
        owner = _resolve_attr(root, owner_path)
    else:
        owner, attr = root, path
    setattr(owner, attr, _wrap_recording(getattr(owner, attr), f"layer{layer_id}.{path}"))


def _iter_comm_ops(root, prefix=""):
    if hasattr(root, "_comm_ops"):
        for slot in root._comm_ops:
            yield f"{prefix}.{slot}" if prefix else slot, root, slot
    for name, cell in root._cells.items():
        yield from _iter_comm_ops(cell, f"{prefix}.{name}" if prefix else name)


def _probe_comm_ops(layer, layer_id):
    """Wrap TP/EP comm ops on ``layer`` so calls bump the replay counter."""
    for path, owner, slot in _iter_comm_ops(layer):
        is_tp = layer_id in range(4, 8) and (
            slot.endswith("allgather") or slot.endswith("reducescatter"))
        is_ep = layer_id in range(1, 8) and (
            slot.endswith("expert_counts.alltoall")
            or slot.endswith("input.alltoallsingle")
            or slot.endswith("output.alltoallsingle"))
        if not (is_tp or is_ep):
            continue
        original = owner._comm_ops[slot]["fn"]
        owner._comm_ops[slot]["fn"] = _wrap_recording(
            original, f"layer{layer_id}.{path}")


def _install_probes(decoder):
    _probe_cell(decoder.layers[0], "input_layernorm", 0)
    _probe_cell(decoder.layers[8], "input_layernorm", 8)
    for lid in range(0, 2):
        _probe_cell(decoder.layers[lid], "self_attention.linear_proj", lid)
    for lid, path in ((2, "add"), (3, "add"), (4, "self_attention.reshape"),
                       (5, "self_attention.cast")):
        _probe_callable(decoder.layers[lid], path, lid)
    for lid in range(1, 8):
        _probe_comm_ops(decoder.layers[lid], lid)


def _patch_apply_ac():
    original = gpt_parallelize.apply_ac

    @functools.wraps(original)
    def _patched(model, *args, **kwargs):
        _install_probes(model)
        return original(model, *args, **kwargs)

    gpt_parallelize.apply_ac = _patched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    ms.set_seed(SEED)
    _patch_apply_ac()
    trainer = PynativeTrainer(config=args.config)
    rank = get_rank()
    trainer.train()
    payload = {"rank": rank, "counts": dict(sorted(_COUNTS.items()))}
    print(f"{PROBE_PREFIX}{json.dumps(payload, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
