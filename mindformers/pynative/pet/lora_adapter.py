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
"""LoRA injection and freezing for PyNative mode (first version: FSDP only).

``build_lora_model`` walks the (meta-device) base model and replaces every
:class:`mindformers.pynative.layers.linear.Linear` whose dotted path matches
``target_modules`` with a :class:`LinearWithLoRA`, reusing the original weight.
MoE routed-expert grouped GEMM is not a ``Linear`` and is therefore skipped.

``freeze_base_params`` sets ``requires_grad=False`` on everything except the
``lora_a``/``lora_b`` adapters. The first version calls it *after* parallelism +
``init_states`` so that frozen base weights are still materialised as DTensors
(the ``distribute_module`` skip of ``requires_grad is False`` params is avoided).
"""
__all__ = ["build_lora_model", "freeze_base_params"]

import re

from mindformers.tools.logger import logger
from mindformers.pynative.layers.linear import Linear
from mindformers.pynative.pet.lora_layer import LinearWithLoRA


def _cfg_get(cfg, key, default):
    """Read ``key`` from a LoraConfig / BaseConfig / dict, falling back to default."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _resolve_lora_kwargs(cfg) -> dict:
    return {
        "lora_rank": int(_cfg_get(cfg, "lora_rank", 8)),
        "lora_alpha": int(_cfg_get(cfg, "lora_alpha", 16)),
        "lora_dropout": float(_cfg_get(cfg, "lora_dropout", 0.0)),
        "lora_a_std": float(_cfg_get(cfg, "lora_a_std", 0.01)),
        "lora_b_std": float(_cfg_get(cfg, "lora_b_std", 0.0)),
    }


def _replace_linears(parent, prefix, target, exclude, lora_kwargs, replaced):
    """Recursively replace matching ``Linear`` children with ``LinearWithLoRA``."""
    # pylint: disable=W0212
    for name, cell in list(parent._cells.items()):
        if cell is None:
            continue
        full = f"{prefix}.{name}" if prefix else name
        if exclude is not None and exclude.search(full):
            continue
        if (isinstance(cell, Linear)
                and not isinstance(cell, LinearWithLoRA)
                and target.search(full)):
            new_cell = LinearWithLoRA.from_base(cell, **lora_kwargs)
            # Replace via _cells (not setattr) so Cell.__setattr__ does NOT re-derive param
            # names: the reused base weight/bias keep their original framework names, which
            # is what a pretrained checkpoint is keyed by (e.g. "decoder...linear_qkv.weight",
            # no "model." prefix). Name the new adapters off that same prefix for consistency.
            parent._cells[name] = new_cell   # pylint: disable=protected-access
            base_name = getattr(cell.weight, "name", None)
            prefix = base_name[:-len(".weight")] if (base_name and base_name.endswith(".weight")) else full
            new_cell.lora_a.name = prefix + ".lora_a"
            new_cell.lora_b.name = prefix + ".lora_b"
            replaced.append(full)
        else:
            _replace_linears(cell, full, target, exclude, lora_kwargs, replaced)


def _stash_lora_config(model, config):
    """Remember the LoRA config on ``model.config`` so pipeline-parallel re-injection
    (which rebuilds fresh per-stage models from config, discarding this injection) can
    recover it. Best-effort: a frozen/slots config simply skips the stash."""
    cfg_obj = getattr(model, "config", None)
    if cfg_obj is None:
        return
    try:
        setattr(cfg_obj, "_mf_lora_config", config)
    except Exception:  # pylint: disable=broad-except
        pass


def build_lora_model(model, config, strict=True):
    """Inject LoRA adapters into ``model`` in place.

    Args:
        model: The base model (built on meta device).
        config: LoRA config object exposing ``target_modules`` (regex, required),
            ``exclude_layers`` (regex, optional), ``lora_rank``, ``lora_alpha``,
            ``lora_dropout``, ``lora_a_std``.
        strict: If True (default), raise when no ``Linear`` matches ``target_modules``.
            Pipeline parallelism re-injects per stage and passes ``strict=False`` because
            a stage may legitimately hold no targetable layer (e.g. an embedding-only
            stage); the "no adapters anywhere" check is then done across all stages.

    Returns:
        The same ``model`` instance, mutated in place.
    """
    target_pattern = _cfg_get(config, "target_modules", None)
    if not target_pattern:
        raise ValueError("lora_config.target_modules is required to enable LoRA.")
    exclude_pattern = _cfg_get(config, "exclude_layers", None)
    target = re.compile(target_pattern)
    exclude = re.compile(exclude_pattern) if exclude_pattern else None
    lora_kwargs = _resolve_lora_kwargs(config)

    # Stash the config for pipeline-parallel re-injection regardless of match outcome.
    _stash_lora_config(model, config)

    replaced = []
    _replace_linears(model, "", target, exclude, lora_kwargs, replaced)

    if not replaced:
        if not strict:
            logger.info("LoRA: no target Linear in this stage/submodel (skipped).")
            return model
        raise ValueError(
            f"LoRA target_modules='{target_pattern}' matched no Linear layers. "
            "Check the regex against the model's module names."
        )
    logger.info(
        "LoRA injected into %d Linear layers (rank=%d, alpha=%d, dropout=%.3g). "
        "Example targets: %s",
        len(replaced), lora_kwargs["lora_rank"], lora_kwargs["lora_alpha"],
        lora_kwargs["lora_dropout"], replaced[:4],
    )
    # NOTE: freezing is deferred to AFTER parallelism + init_states (see Trainer). During
    # distribute_module, params must be trainable so they all materialise as DTensors;
    # freezing here would leave frozen non-LoRA params (norms/biases, NoParallel base
    # weights) as plain Tensors and break TP layout inference.
    return model


def _is_lora_param_name(name: str) -> bool:
    return name.endswith(".lora_a") or name.endswith(".lora_b") or name in ("lora_a", "lora_b")


def freeze_base_params(model):
    """Freeze every parameter except the LoRA adapters (``lora_a``/``lora_b``).

    Returns:
        Tuple[int, int]: (num_trainable, num_frozen) parameter tensors.
    """
    trainable = frozen = 0
    for param in model.get_parameters():
        if _is_lora_param_name(param.name):
            param.requires_grad = True
            trainable += 1
        else:
            param.requires_grad = False
            frozen += 1
    # NOTE: under pipeline parallelism this is called per stage, and a stage may legitimately
    # hold no adapters; the "no adapters anywhere" check is done by the caller across all stages.
    logger.info("LoRA freeze: %d trainable adapter tensors, %d frozen base tensors.",
                trainable, frozen)
    return trainable, frozen
