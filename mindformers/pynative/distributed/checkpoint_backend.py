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
"""Activation-checkpoint implementation backends."""

import importlib
import inspect

from hyper_parallel.core.activation_checkpoint import (
    checkpoint_exclude_wrapper,
    checkpoint_wrapper,
)


def _accepts_kwarg(func, name):
    """Whether ``func`` accepts keyword ``name``; False if it cannot be inspected."""
    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


REENTRANT_CHECKPOINT_MODULE = (
    "hyper_parallel.platform.mindspore.activation_checkpoint."
    "reentrant_checkpoint"
)


class CheckpointBackend:
    """Implementation-specific activation-checkpoint operations."""

    name = "checkpoint"
    log_prefix = ""

    def __init__(self, context_fn):
        self._context_fn = context_fn

    def wrap_cell(self, cell, **checkpoint_kwargs):
        """Wrap a Cell checkpoint boundary.

        ``checkpoint_kwargs`` carries optional knobs (currently ``early_stop``) that not
        every backend accepts; callers must gate them on :meth:`supports_kwarg`.
        """
        raise NotImplementedError

    def supports_kwarg(self, name):
        """Whether :meth:`wrap_cell` accepts keyword ``name`` on this backend.

        Each backend answers for the function it actually calls -- the two differ, so
        probing one on behalf of the other silently mis-gates the keyword.
        """
        raise NotImplementedError

    def wrap_callable(self, operator):
        """Wrap a callable checkpoint boundary."""
        raise NotImplementedError

    def wrap_exclude(self, operator):
        """Wrap an excluded child within a checkpoint boundary."""
        raise NotImplementedError

    @staticmethod
    def validate_recompute(recompute, recompute_comm):
        """Validate backend-specific recompute capabilities."""
        del recompute, recompute_comm

    @staticmethod
    def validate_swap(swap):
        """Validate whether activation swap can use this backend."""
        del swap

    @staticmethod
    def partition_excludes(
            layer_id, patterns, full_target_ids, select_layer_to_modules):
        """Return active and inactive excludes for one layer."""
        del layer_id, full_target_ids, select_layer_to_modules
        return list(patterns), []


class NonReentrantCheckpointBackend(CheckpointBackend):
    """Saved-tensor-hook checkpoint implementation."""

    name = "non-reentrant"

    def wrap_cell(self, cell, **checkpoint_kwargs):
        return checkpoint_wrapper(cell, context_fn=self._context_fn, **checkpoint_kwargs)

    def supports_kwarg(self, name):
        # ``CheckpointWrapper`` stores whatever it is given and forwards it verbatim to
        # ``checkpoint``, so that is the signature to probe -- not ``checkpoint_wrapper``'s.
        # ``early_stop`` only reached ``checkpoint`` in later HyperParallel versions; on
        # older ones it would be passed on to the wrapped cell's ``construct`` and blow up.
        del self
        try:
            from hyper_parallel.core.activation_checkpoint import checkpoint  # pylint: disable=C0415
        except ImportError:
            return False
        return _accepts_kwarg(checkpoint, name)

    def wrap_callable(self, operator):
        del self
        return checkpoint_wrapper(operator, output_recompute=True)

    def wrap_exclude(self, operator):
        del self
        return checkpoint_exclude_wrapper(operator)


class ReentrantCheckpointBackend(CheckpointBackend):
    """HyperParallel custom-backward checkpoint implementation.

    The HyperParallel MR 1291 API is imported only when this backend is
    selected. Non-reentrant users therefore remain compatible with an older
    HyperParallel installation that does not provide the reentrant module.
    """

    name = "HyperParallel reentrant"
    log_prefix = "reentrant "

    def __init__(self, context_fn):
        super().__init__(context_fn)
        try:
            api = importlib.import_module(REENTRANT_CHECKPOINT_MODULE)
            self._checkpoint_wrapper = api.reentrant_checkpoint_wrapper
            self._exclude_wrapper = api.reentrant_checkpoint_exclude_wrapper
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "recompute.use_reentrant=True requires a HyperParallel version "
                "that provides activation_checkpoint.reentrant_checkpoint "
                "(MR 1291 or later)."
            ) from exc

    def wrap_cell(self, cell, **checkpoint_kwargs):
        return self._checkpoint_wrapper(
            cell, context_fn=self._context_fn, **checkpoint_kwargs)

    def supports_kwarg(self, name):
        # The reentrant wrapper consumes its own keywords and never forwards them to
        # ``checkpoint``, so probe it directly. It takes no ``early_stop`` and needs none:
        # it replays the whole forward, so there is no early stop to disable.
        return _accepts_kwarg(self._checkpoint_wrapper, name)

    def wrap_callable(self, operator):
        return self._checkpoint_wrapper(
            operator, context_fn=self._context_fn)

    def wrap_exclude(self, operator):
        return self._exclude_wrapper(operator)

    @staticmethod
    def validate_recompute(recompute, recompute_comm):
        if recompute.mode == "None":
            raise ValueError(
                "recompute.use_reentrant=True requires "
                "recompute.mode='full' or recompute.mode='select'."
            )
        if recompute_comm.enable:
            raise ValueError(
                "recompute_comm.enable=True is not supported with "
                "recompute.use_reentrant=True."
            )

    @staticmethod
    def validate_swap(swap):
        if swap.enable:
            raise ValueError(
                "swap.enable=True is not supported with "
                "recompute.use_reentrant=True."
            )

    @staticmethod
    def partition_excludes(
            layer_id, patterns, full_target_ids, select_layer_to_modules):
        """Keep excludes only when enclosed by a reentrant boundary."""
        active = []
        inactive = []
        select_parents = select_layer_to_modules.get(layer_id, ())
        for pattern in patterns:
            covered_by_select = any(
                pattern == parent or pattern.startswith(parent + ".")
                for parent in select_parents
            )
            if layer_id in full_target_ids or covered_by_select:
                active.append(pattern)
            else:
                inactive.append(pattern)
        return active, inactive


def create_checkpoint_backend(use_reentrant, context_fn):
    """Create the requested backend without loading unused implementations."""
    if use_reentrant:
        return ReentrantCheckpointBackend(context_fn)
    return NonReentrantCheckpointBackend(context_fn)
