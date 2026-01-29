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
"""Utility functions for config module."""

from typing import Any, List, Union, Type, get_origin, get_args
from dataclasses import is_dataclass


def check_type(value: Any, expected_type: Type, field_name: str):
    """Recursively check if value matches expected_type."""
    if expected_type is Any:
        return

    origin = get_origin(expected_type)

    if origin is Union:
        _check_union(value, expected_type, field_name)
        return

    if value is None:
        raise TypeError(f"Field '{field_name}' cannot be None")

    if origin is list or origin is List:
        _check_list(value, expected_type, field_name)
        return

    if is_dataclass(expected_type):
        _check_dataclass(value, expected_type, field_name)
        return

    _check_scalar(value, expected_type, field_name)


def _check_union(value: Any, expected_type: Type, field_name: str):
    """Handle Union and Optional types."""
    args = get_args(expected_type)

    if value is None:
        if type(None) not in args:
            raise TypeError(f"Field '{field_name}' cannot be None")
        return

    # Try to match against any valid type
    matched = False
    for t in args:
        if t is type(None):
            continue
        try:
            check_type(value, t, field_name)
            matched = True
            break
        except TypeError:
            continue

    if not matched:
        raise TypeError(
            f"Expected {expected_type} for '{field_name}', got {type(value).__name__} ({value})"
        )


def _check_list(value: Any, expected_type: Type, field_name: str):
    """Handle List types."""
    if not isinstance(value, list):
        raise TypeError(f"Expected list for '{field_name}', got {type(value).__name__}")
    args = get_args(expected_type)
    if args:
        inner_type = args[0]
        for i, item in enumerate(value):
            check_type(item, inner_type, f"{field_name}[{i}]")


def _check_dataclass(value: Any, expected_type: Type, field_name: str):
    """Handle Dataclass types."""
    if not isinstance(value, expected_type):
        raise TypeError(
            f"Expected {expected_type.__name__} for '{field_name}', got {type(value).__name__}"
        )
    if hasattr(value, "validate"):
        value.validate()


def _check_scalar(value: Any, expected_type: Type, field_name: str):
    """Handle scalar types (bool, int, float, str)."""
    if expected_type is bool:
        if not isinstance(value, bool):
            raise TypeError(
                f"Expected bool for '{field_name}', got {type(value).__name__} ({value})"
            )
    elif expected_type is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(
                f"Expected int for '{field_name}', got {type(value).__name__} ({value})"
            )
    elif expected_type is float:
        if not isinstance(value, (float, int)) or isinstance(value, bool):
            raise TypeError(
                f"Expected float for '{field_name}', got {type(value).__name__} ({value})"
            )
    elif expected_type is str:
        if not isinstance(value, str):
            raise TypeError(
                f"Expected str for '{field_name}', got {type(value).__name__} ({value})"
            )
