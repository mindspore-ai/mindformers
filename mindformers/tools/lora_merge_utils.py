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
"""Configuration and parameter planning for LoRA checkpoint merging."""

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from mindformers.checkpoint.utils import get_base_checkpoint_fingerprint


MIND_FORMERS_ADAPTER_FORMAT = "mindformers_pynative_lora_adapter_v1"
MOE_ADAPTER_LABEL = "pynative_moe"

# Most-specific suffix first. (rank-reduction suffix, rank-expansion suffix,
# implementation label). The suffix strings retain each checkpoint format's
# external A/B naming contract.
ADAPTER_CONVENTIONS: Tuple[Tuple[str, str, str], ...] = (
    (".lora_A.weight", ".lora_B.weight", "parallel_core"),
    (".mindpet_delta_lora_a", ".mindpet_delta_lora_b", "mindpet"),
    (".lora_a", ".lora_b", "pynative"),
    ("_lora_a", "_lora_b", MOE_ADAPTER_LABEL),
)


@dataclass(frozen=True)
class AdapterSpec:
    """One validated low-rank adapter pair and its target base parameter."""

    rank_reduction_name: str
    rank_expansion_name: str
    base_name: str
    label: str


@dataclass(frozen=True)
class AdapterConfig:
    """Resolved adapter configuration and optional MindFormers manifest."""

    values: Optional[dict]
    manifest: Optional[dict]
    path: Optional[str]


def classify_adapter(name: str) -> Optional[Tuple[str, str, Tuple[str, str, str]]]:
    """Return ``(prefix, kind, convention)`` for a supported adapter name."""
    for convention in ADAPTER_CONVENTIONS:
        rank_reduction_suffix, rank_expansion_suffix, _ = convention
        if name.endswith(rank_reduction_suffix):
            return name[:-len(rank_reduction_suffix)], "A", convention
        if name.endswith(rank_expansion_suffix):
            return name[:-len(rank_expansion_suffix)], "B", convention
    return None


def adapter_base_key(prefix: str, label: str) -> str:
    """Return the base parameter name for an adapter prefix."""
    return prefix if label == MOE_ADAPTER_LABEL else prefix + ".weight"


def validate_adapter_pair(rank_reduction_weight, rank_expansion_weight,
                          label: str, adapter_name: str) -> int:
    """Validate one A/B pair and return its positive LoRA rank.

    Both merge backends use this check so they agree on supported layouts before
    performing any matrix multiplication.
    """
    if label == MOE_ADAPTER_LABEL:
        valid = (rank_reduction_weight.ndim == 3 and rank_expansion_weight.ndim == 3
                 and rank_reduction_weight.shape[0] == rank_expansion_weight.shape[0]
                 and rank_reduction_weight.shape[2] == rank_expansion_weight.shape[1])
        rank = rank_reduction_weight.shape[2] if rank_reduction_weight.ndim == 3 else 0
    else:
        valid = (rank_reduction_weight.ndim == 2 and rank_expansion_weight.ndim == 2
                 and rank_reduction_weight.shape[0] == rank_expansion_weight.shape[1])
        rank = rank_reduction_weight.shape[0] if rank_reduction_weight.ndim == 2 else 0
    if not valid:
        raise ValueError(
            f"Adapter '{adapter_name}' has incompatible shapes: "
            f"A={rank_reduction_weight.shape}, B={rank_expansion_weight.shape}."
        )
    if rank <= 0:
        raise ValueError(f"Adapter '{adapter_name}' has an invalid LoRA rank {rank}; rank must be positive.")
    return int(rank)


def build_adapter_plan(names: Iterable[str]) -> List[AdapterSpec]:
    """Build and validate all adapter pairs before any weight is changed."""
    name_set = set(names)
    rank_reduction_entries: Dict[Tuple[str, str], Tuple[str, Tuple[str, str, str]]] = {}
    rank_expansion_entries: Dict[Tuple[str, str], Tuple[str, Tuple[str, str, str]]] = {}

    for name in name_set:
        classified = classify_adapter(name)
        if classified is None:
            continue
        prefix, kind, convention = classified
        key = (prefix, convention[2])
        target = rank_reduction_entries if kind == "A" else rank_expansion_entries
        target[key] = (name, convention)

    unpaired_rank_expansions = sorted(
        value[0] for key, value in rank_expansion_entries.items()
        if key not in rank_reduction_entries
    )
    if unpaired_rank_expansions:
        raise ValueError(
            "Found LoRA B adapter(s) without matching A adapter: "
            f"{unpaired_rank_expansions[:4]}."
        )

    specs = []
    targeted_bases = {}
    for key, (rank_reduction_name, convention) in sorted(rank_reduction_entries.items()):
        prefix, label = key
        _, rank_expansion_suffix, _ = convention
        rank_expansion_name = prefix + rank_expansion_suffix
        base_name = adapter_base_key(prefix, label)
        if rank_expansion_name not in name_set:
            raise ValueError(
                f"Missing paired B-adapter for '{rank_reduction_name}': "
                f"expected '{rank_expansion_name}' "
                f"(convention '{label}')."
            )
        if base_name not in name_set:
            raise ValueError(
                f"Cannot locate the base weight for adapter '{rank_reduction_name}': "
                f"expected '{base_name}'."
            )
        if base_name in targeted_bases:
            raise ValueError(
                f"Multiple LoRA adapters target the same base weight '{base_name}': "
                f"'{targeted_bases[base_name]}' and '{rank_reduction_name}'."
            )
        targeted_bases[base_name] = rank_reduction_name
        specs.append(AdapterSpec(rank_reduction_name, rank_expansion_name, base_name, label))

    return specs


def _load_json_object(path: str) -> dict:
    """Load a JSON object; a present but invalid file is always an error."""
    try:
        with open(path, "r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to read adapter configuration '{path}': {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Adapter configuration '{path}' must contain a JSON object.")
    return value


def _candidate_directories(src_path: str) -> List[str]:
    """Return the source directory and up to three ancestors without duplicates."""
    if os.path.isfile(src_path):
        current = os.path.dirname(os.path.abspath(src_path))
    elif os.path.isdir(src_path):
        current = os.path.abspath(src_path)
    else:
        return []
    result = []
    for _ in range(4):
        if current and current not in result:
            result.append(current)
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return result


def read_adapter_config(src_path: str) -> AdapterConfig:
    """Resolve a MindFormers manifest first, then a generic adapter config.

    A file is skipped only when it does not exist. A present malformed file or
    an incompatible MindFormers manifest is rejected instead of silently using
    a default alpha.
    """
    directories = _candidate_directories(src_path)
    for directory in directories:
        path = os.path.join(directory, "mindformers_adapter_config.json")
        if not os.path.isfile(path):
            continue
        manifest = _load_json_object(path)
        if manifest.get("format") != MIND_FORMERS_ADAPTER_FORMAT:
            raise ValueError(
                f"Unsupported MindFormers adapter manifest format in '{path}': "
                f"{manifest.get('format')!r}."
            )
        config = manifest.get("lora_config")
        if not isinstance(config, dict):
            raise ValueError(f"MindFormers adapter manifest '{path}' has no valid 'lora_config' object.")
        return AdapterConfig(config, manifest, path)

    for directory in directories:
        path = os.path.join(directory, "adapter_config.json")
        if os.path.isfile(path):
            return AdapterConfig(_load_json_object(path), None, path)
    return AdapterConfig(None, None, None)


def validate_adapter_base(manifest: Optional[dict], base_path: str, manifest_path: Optional[str]) -> None:
    """Validate an adapter-only checkpoint against its recorded base fingerprint."""
    if manifest is None:
        return
    expected = manifest.get("base_fingerprint")
    if not isinstance(expected, str) or not expected:
        raise ValueError(
            f"MindFormers adapter manifest '{manifest_path}' has no valid 'base_fingerprint'."
        )
    actual = get_base_checkpoint_fingerprint(base_path)
    if actual is None:
        raise ValueError(f"Cannot fingerprint base checkpoint '{base_path}'.")
    if actual != expected:
        raise ValueError(
            f"Base checkpoint fingerprint does not match adapter manifest '{manifest_path}'."
        )
