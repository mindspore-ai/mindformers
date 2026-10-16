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
"""Common utilities for PyNative parameter-efficient tuning."""


def is_lora_param_name(name: str) -> bool:
    """Return whether a parameter name follows a supported PyNative LoRA convention."""
    return (name.endswith((".lora_a", ".lora_b", "_lora_a", "_lora_b"))
            or name in ("lora_a", "lora_b"))


def is_lora_checkpoint_param_name(name: str) -> bool:
    """Return whether a parameter is required by an exact LoRA training resume.

    Besides trainable LoRA weights, auxiliary-loss-free MoE routing owns mutable,
    non-trainable state.  Dropping that state from an adapter checkpoint resets
    expert routing on resume and changes the very next forward pass.
    """
    return is_lora_param_name(name) or name.endswith((".expert_bias", ".tokens_per_expert"))


def init_lora_parameters(lora_a_params, lora_b_params, lora_a_std, lora_b_std):
    """Apply the shared LoRA initialization policy to one or more A/B parameters."""
    for param in lora_a_params:
        param.normal_(mean=0.0, std=lora_a_std)
    for param in lora_b_params:
        if lora_b_std > 0.0:
            param.normal_(mean=0.0, std=lora_b_std)
        else:
            param.zero_()
