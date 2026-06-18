# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file is a direct port of TorchTitan's ParallelDims
# (torchtitan/distributed/parallel_dims.py), adapted for MindSpore with the
# *minimal* changes required to run on top of ``hyper_parallel``:
#     - torch DeviceMesh / init_device_mesh -> hyper_parallel equivalents.
#     - device_type hard-coded to "npu".
#     - logger -> mindformers logger.
#     - ``from_config`` (TorchTitan ParallelismConfig factory) dropped; the
#       MindFormers trainer constructs ParallelDims directly.
# ============================================================================
"""Multi-dimensional parallelism dims and device-mesh construction (pynative)."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from hyper_parallel import DeviceMesh, init_device_mesh

from mindformers.tools.logger import logger

if TYPE_CHECKING:
    # Only used for type annotations (strings under ``from __future__ import
    # annotations``); never imported at runtime.
    MeshAxisName = str
    NamedPlacement = object


__all__ = ["ParallelDims"]

device_type = "npu"


@dataclass
class ParallelDims:
    """Multi-dimensional parallelism config and device-mesh factory.

    Ported from TorchTitan: a 1D world mesh is unflattened into named
    dataloading / dense / sparse views, and sub-meshes are looked up by axis
    name via ``get_mesh``.
    """
    dp_replicate: int
    dp_shard: int
    cp: int
    tp: int
    pp: int
    ep: int
    world_size: int
    full_dtensor: bool = False
    # Cache by axis name(s); DeviceMesh equality is by identity, so reuse
    # is required for ``mesh in spmd_meshes()`` checks.
    _single_axis_meshes: dict[str, DeviceMesh] = field(default_factory=dict)
    _multi_axis_meshes: dict[tuple[str, ...], DeviceMesh] = field(default_factory=dict)
    _world_mesh: DeviceMesh | None = None
    _spmd_meshes: list[DeviceMesh] = field(default_factory=list)

    @classmethod
    def from_config(cls, parallelism, world_size: int) -> "ParallelDims":
        """Build ParallelDims from a MindFormers ``ParallelismConfig``.

        MindFormers field semantics differ from TorchTitan's ``*_degree`` fields:
          - ``parallelism.data_parallel`` is the *total* DP degree, computed by
            the trainer as ``world_size // (tp * pp * cp)`` and stored back on
            the config (it is not a standalone config field).
          - ``parallelism.data_parallel_shard`` is the FSDP shard degree; the
            default ``-1`` means "no replicate" (pure FSDP over all of DP).
          - ``dp_replicate`` is derived as ``total_dp // shard``.
        """
        data_parallel = parallelism.data_parallel
        dp_shard_cfg = parallelism.data_parallel_shard
        if dp_shard_cfg > 0:
            # HSDP: shard within ``data_parallel_shard`` ranks, replicate across
            # the rest so that dp_replicate * dp_shard == data_parallel.
            dp_shard = dp_shard_cfg
            dp_replicate = max(data_parallel // dp_shard, 1)
        else:
            # Pure FSDP over the whole DP group.
            dp_replicate = 1
            dp_shard = data_parallel
        return cls(
            dp_replicate=dp_replicate,
            dp_shard=dp_shard,
            cp=parallelism.context_parallel,
            tp=parallelism.tensor_parallel,
            pp=parallelism.pipeline_parallel,
            ep=parallelism.expert_parallel,
            world_size=world_size,
            full_dtensor=getattr(parallelism, "full_dtensor", False),
        )

    def __post_init__(self):
        self._validate()

    def _validate(self):
        """Validate degrees and auto-derive ``dp_shard`` when set to -1."""
        dp_replicate, dp_shard, cp, tp, pp, ep = (
            self.dp_replicate,
            self.dp_shard,
            self.cp,
            self.tp,
            self.pp,
            self.ep,
        )
        for d in (dp_replicate, cp, tp, pp, ep):
            if d < 1:
                raise ValueError("Parallelism degree should be >= 1, except for dp_shard")

        if not (dp_shard == -1 or dp_shard >= 1):
            raise ValueError("dp_shard must be -1 or >= 1.")
        if dp_shard < 0:
            self.dp_shard = dp_shard = self.world_size // (dp_replicate * cp * tp * pp)
        if dp_shard < 1:
            raise ValueError(
                f"Auto-derived dp_shard({dp_shard}) is invalid (must be >= 1): "
                f"world_size({self.world_size}) // (dp_replicate({dp_replicate}) * "
                f"cp({cp}) * tp({tp}) * pp({pp}) = {dp_replicate * cp * tp * pp}) < 1. "
                f"world_size must be a multiple of dp_replicate * cp * tp * pp."
            )

        if dp_replicate * dp_shard * cp * tp * pp != self.world_size:
            raise ValueError(
                f"Invalid parallel dims: dp_replicate({dp_replicate}) * dp_shard({dp_shard}) * "
                f"cp({cp}) * tp({tp}) * pp({pp}) != WORLD_SIZE({self.world_size})"
            )

        # EP is carved out of the (dp_shard * cp * tp) region within a single
        # dp_replicate group: the sparse mesh ["pp", "dp_replicate", "efsdp",
        # "ep"] nests ``ep`` inside ``dp_replicate``, so ``efsdp * ep`` must
        # equal ``dp_shard * cp * tp``. If ``ep`` does not divide that region,
        # ``efsdp = dp_shard * cp * tp // ep`` silently floors to a wrong value
        # (e.g. 0 when ``ep`` exceeds the region), which only surfaces later as
        # an opaque mesh-size mismatch. Reject it here with an actionable error.
        ep_region = dp_shard * cp * tp
        if ep_region % ep != 0:
            raise ValueError(
                f"Invalid expert_parallel: ep({ep}) must divide dp_shard({dp_shard}) * "
                f"cp({cp}) * tp({tp}) = {ep_region}, and thus cannot exceed {ep_region}. "
                f"EP is sharded within a single dp_replicate group, so to use a larger ep "
                f"increase dp_shard*cp*tp (e.g. raise dp_shard) instead of relying on "
                f"dp_replicate."
            )

    def _mesh_exist(self, name: str, degree: int) -> bool:
        """Whether axis ``name`` needs a real (non-fake) process group."""
        if name == "fsdp":
            # Always keep fsdp mesh with real backend so fully_shard()
            # can apply MixedPrecisionPolicy even at degree 1.
            return True
        if name == "loss":
            # Always keep the loss mesh (dp x cp domain) real, even at degree 1:
            # the trainer all-reduces per-step metrics (loss, MoE aux loss,
            # tokens_per_expert) over this group and divides by its size, so a
            # valid size-1 group must exist on PP-only / single-card runs.
            return True
        if name == "dp_shard" and self.full_dtensor:
            # Under full_dtensor ``dp_shard`` is the DP storage axis (no
            # flattened ``fsdp``); keep alive at size 1 so ``fully_shard``
            # can install MixedPrecisionPolicy and FSDP can discriminate
            # the DP submesh on TP/DDP/PP-only.
            return True
        if name == "efsdp":
            # We always keep the efsdp if EP is larger than 1 because we need
            # FSDP wrapping to help the MoE layers do mixed precision training.
            return self.ep > 1
        return degree > 1

    def build_mesh(self) -> DeviceMesh:
        """
        Build the device mesh with the required mesh dimensions.

        The following mesh dimensions will be created:

            pp:      Pipeline Parallelism (PP).
            batch:   Used by data loading to determine the global batch size and which
                     part of the data each rank should read. This dimension includes both
                     ``dp_replicate`` and ``dp_shard``.
            loss:    Used by all-reduce when computing the loss. Includes ``dp_replicate``,
                     ``dp_shard``, and ``cp`` degrees, as all of them parallelize the data,
                     essentially require the weight gradients reduction.
            dp_replicate: For DDP or HSDP replicate dimension.
            fsdp:    For FSDP dimension. This includes ``dp_shard`` and ``cp``. Note that
                     we always assume that when ``cp`` is used, FSDP is also applied to
                     utilize its weight all-gather and gradients reduce_scatter even if
                     there may be no data parallelism (e.g., global batch size is 1).
            cp:      Context Parallelism (CP).
            tp:      Tensor Parallelism (TP).
            ep:      Expert Parallelism (EP).
            efsdp:   FSDP in the EP region.

        Note: Most dimensions above are created by unflattening the world mesh, except for loss,
        which is created by flattening the batch and cp dimensions.
        This API performs the following unflatten operations from the world mesh:

            ["pp", "batch", "cp", "tp"]  # dataloading_mesh
            ["pp", "dp_replicate", "fsdp", "tp"]  # dense_mesh
            ["pp", "dp_replicate", "efsdp", "ep"]  # sparse_mesh
        """

        def unflatten_mesh(
            world_mesh: DeviceMesh,
            dim_names: tuple[str, ...],
            dim_degrees: tuple[int, ...],
        ):
            """Unflatten the world mesh to create the required mesh dimensions.

            Uses fake backend for dimensions with degree 1 or for 'batch' dimension
            to avoid unnecessary process group creation.
            """
            backend_override = {}
            for name, degree in zip(dim_names, dim_degrees, strict=True):
                if not self._mesh_exist(name, degree):
                    backend_override[name] = "fake"

            return world_mesh._unflatten(
                0,
                dim_degrees,
                dim_names,
                backend_override=backend_override,
            )

        logger.info(
            f"Building device mesh with parallelism: "
            f"pp={self.pp}, dp_replicate={self.dp_replicate}, dp_shard={self.dp_shard}, "
            f"cp={self.cp}, tp={self.tp}, ep={self.ep}"
        )

        batch = self.dp_replicate * self.dp_shard
        fsdp = self.dp_shard * self.cp
        efsdp = fsdp * self.tp // self.ep

        self._world_mesh = init_device_mesh(
            device_type, (self.world_size,), mesh_dim_names=("world",)
        )
        dataloading_mesh = unflatten_mesh(
            self._world_mesh,
            ("pp", "batch", "cp", "tp"),
            (self.pp, batch, self.cp, self.tp),
        )
        loss_mesh = dataloading_mesh["batch", "cp"]._flatten("loss_mesh")
        if self.full_dtensor:
            # Under full_dtensor, ``dp_shard`` and ``cp`` cannot be folded
            # together: activations carry a ``cp`` dimension, so parameters
            # need a ``cp`` axis as well. ``fully_shard`` folds ``dp_shard``
            # and ``cp`` internally at initialization time.
            candidate_spmd_dense_axes = ["dp_replicate", "dp_shard", "cp", "tp"]
            full_dense_mesh = unflatten_mesh(
                self._world_mesh,
                tuple(["pp"] + candidate_spmd_dense_axes),
                (self.pp, self.dp_replicate, self.dp_shard, self.cp, self.tp),
            )
        else:
            # Legacy path folds ``dp_shard`` and ``cp`` into ``fsdp``.
            candidate_spmd_dense_axes = ["dp_replicate", "fsdp", "tp"]
            full_dense_mesh = unflatten_mesh(
                self._world_mesh,
                ("pp", "dp_replicate", "fsdp", "tp"),
                (self.pp, self.dp_replicate, fsdp, self.tp),
            )

        full_sparse_mesh = unflatten_mesh(
            self._world_mesh,
            ("pp", "dp_replicate", "efsdp", "ep"),
            (self.pp, self.dp_replicate, efsdp, self.ep),
        )

        self._global_meshes = {
            "dataloading": dataloading_mesh,
            "loss": loss_mesh,
            "dense": full_dense_mesh,
            "sparse": full_sparse_mesh,
        }

        self._single_axis_meshes = {
            "pp": dataloading_mesh["pp"],
            "batch": dataloading_mesh["batch"],
            "loss": loss_mesh,
            "dp_replicate": full_dense_mesh["dp_replicate"],
            "cp": dataloading_mesh["cp"],
            "tp": dataloading_mesh["tp"],
            "ep": full_sparse_mesh["ep"],
            "efsdp": full_sparse_mesh["efsdp"],
        }
        if self.full_dtensor:
            self._single_axis_meshes["dp_shard"] = full_dense_mesh["dp_shard"]
        else:
            self._single_axis_meshes["fsdp"] = full_dense_mesh["fsdp"]

        self._validate_meshes()

        candidate_spmd_sparse_axes = ["dp_replicate", "efsdp", "ep"]
        activated_spmd_dense_mesh = self.get_activated_mesh(candidate_spmd_dense_axes)
        activated_spmd_sparse_mesh = self.get_activated_mesh(candidate_spmd_sparse_axes)
        self._spmd_meshes = [
            m
            for m in (activated_spmd_dense_mesh, activated_spmd_sparse_mesh)
            if m is not None
        ]

        logger.info(
            f"Successfully created meshes with active dimensions: "
            f"{list(self.get_all_one_dimensional_meshes().keys())}"
        )

        return self._world_mesh

    def _validate_meshes(self):
        """Validate that created meshes have the expected sizes."""
        expected_sizes = {
            "pp": self.pp,
            "batch": self.dp_replicate * self.dp_shard,
            "loss": self.dp_replicate * self.dp_shard * self.cp,
            "dp_replicate": self.dp_replicate,
            "cp": self.cp,
            "tp": self.tp,
            "ep": self.ep,
            "efsdp": self.dp_shard * self.cp * self.tp // self.ep,
        }
        if self.full_dtensor:
            expected_sizes["dp_shard"] = self.dp_shard
        else:
            expected_sizes["fsdp"] = self.dp_shard * self.cp

        for mesh_name, expected_size in expected_sizes.items():
            actual_size = self._single_axis_meshes[mesh_name].size()
            if actual_size != expected_size:
                raise RuntimeError(
                    f"Mesh '{mesh_name}' has unexpected size: "
                    f"expected {expected_size}, got {actual_size}"
                )

    def get_optional_mesh(self, dims: str | list[str]) -> DeviceMesh | None:
        """Get a device mesh by dimension name(s), returning None if not enabled.

        Args:
            dims: Names of the mesh dimension. Valid options include:
                 'pp', 'batch', 'loss', 'dp_replicate', 'fsdp',
                 'cp', 'tp', 'ep', 'efsdp'.

        Returns:
            DeviceMesh for the requested dimension(s), or None if:
            - The dimension size is 1 (parallelism not enabled)
            - The dimension doesn't exist
            Note: 'fsdp' always exists (for mixed precision via fully_shard()),
            and 'efsdp' exists when ep > 1, even if their size is 1.

        Raises:
            ValueError: If the requested dimension name(s) is not valid.
        """
        if not self._single_axis_meshes:
            self.build_mesh()

        if isinstance(dims, str):
            dims = [dims]

        for mesh_name in dims:
            if mesh_name not in self._single_axis_meshes:
                raise ValueError(
                    f"Invalid mesh dim: '{mesh_name}'. "
                    f"Valid dimensions are: {list(self._single_axis_meshes.keys())}"
                )

        if any(
            not self._mesh_exist(dim, self._single_axis_meshes[dim].size())
            for dim in dims
        ):
            return None

        if len(dims) == 1:
            return self._single_axis_meshes[dims[0]]

        # Cache to ensure mesh equality by object identity.
        key = tuple(dims)
        if key in self._multi_axis_meshes:
            return self._multi_axis_meshes[key]

        candidates = [
            (name, global_mesh)
            for name, global_mesh in self._global_meshes.items()
            if global_mesh.mesh_dim_names is not None
            and set(dims).issubset(set(global_mesh.mesh_dim_names))
        ]
        if not candidates:
            raise ValueError(f"Invalid mesh name combinations {dims}.")
        submesh = candidates[0][1][key]
        self._multi_axis_meshes[key] = submesh
        return submesh

    def get_mesh(self, dims: str | list[str]) -> DeviceMesh:
        """Get a device mesh by dimension name(s), raising if not available.

        Args:
            dims: Names of the mesh dimension. Valid options include:
                 'pp', 'batch', 'loss', 'dp_replicate', 'fsdp',
                 'cp', 'tp', 'ep', 'efsdp'.

        Returns:
            DeviceMesh for the requested dimension(s).

        Raises:
            ValueError: If the mesh is not available (dimension size = 1 or not enabled),
                or if the requested dimension name(s) is not valid.
        """
        mesh = self.get_optional_mesh(dims)
        if mesh is None:
            enabled_str = (
                "enabled (size > 1)" if isinstance(dims, str) else "all enabled"
            )
            raise ValueError(
                f"Mesh '{dims}' is not available. "
                f"Ensure the corresponding parallelism dimension is {enabled_str}."
            )
        return mesh

    def spmd_meshes(self) -> list[DeviceMesh]:
        """Valid full-SPMD meshes, restricted to enabled axes.

        Returns the full-SPMD meshes; today we have dense and sparse.
        """
        if not self._spmd_meshes:
            self.build_mesh()
        return self._spmd_meshes

    def get_activated_mesh(self, axes: list[str]) -> DeviceMesh | None:
        """Submesh of ``axes`` filtered to those actually enabled in this run.

        Returns a mesh containing the axes in ``axes`` that are enabled. If
        none of the axes in ``axes`` is enabled, returns ``None``. This
        differs from ``get_optional_mesh``, which returns ``None`` as soon
        as any axis in ``axes`` is not enabled.
        """
        if not self._single_axis_meshes:
            self.build_mesh()
        axes = [
            axis
            for axis in axes
            if axis in self._single_axis_meshes
            and self.get_optional_mesh(axis) is not None
        ]
        return self.get_optional_mesh(axes) if axes else None

    def resolve_mesh(self, axes: Iterable[MeshAxisName | str]) -> DeviceMesh | None:
        """Resolve the device mesh for a set of mesh axis names."""
        axes_list = list(axes)
        if not self.full_dtensor:
            in_band = ("tp", "ep")
            axes_list = [axis for axis in axes_list if axis in in_band]
        mesh = self.get_activated_mesh(axes_list)
        if mesh is None:
            return None
        if mesh.mesh_dim_names is None:
            raise RuntimeError("DeviceMesh must have named axes")
        if self.full_dtensor and mesh not in self.spmd_meshes():
            raise ValueError(
                f"Resolved mesh {list(mesh.mesh_dim_names)} does not match any "
                f"SPMD mesh. Valid meshes: "
                f"{[list(m.mesh_dim_names or ()) for m in self.spmd_meshes()]}."
            )
        return mesh

    def resolve_shared_mesh(
        self, placements: Iterable["NamedPlacement | None"]
    ) -> DeviceMesh | None:
        """Resolve the mesh shared by a list of NamedPlacements."""
        non_none = [p for p in placements if p is not None]
        if not non_none:
            return None
        axes = non_none[0].keys()
        for p in non_none[1:]:
            if p.keys() != axes:
                raise ValueError(
                    f"Inconsistent mesh axes within a boundary: "
                    f"{sorted(k.value for k in axes)} vs "
                    f"{sorted(k.value for k in p.keys())}"
                )
        return self.resolve_mesh(axes)

    def get_all_one_dimensional_meshes(self) -> dict[str, DeviceMesh]:
        """Get all enabled one-dimensional device meshes.

        Returns a dictionary mapping mesh dimension names to their
        corresponding DeviceMesh objects. Only includes meshes where
        ``ndim == 1`` and parallelism is enabled (size > 1).
        """
        if not self._single_axis_meshes:
            self.build_mesh()
        return {
            k: v
            for k, v in self._single_axis_meshes.items()
            if v.ndim == 1 and v.size() > 1
        }

    @property
    def world_mesh(self) -> DeviceMesh:
        if self._world_mesh is None:
            self._world_mesh = self.build_mesh()
        return self._world_mesh

    @property
    def dp_enabled(self):
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self):
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self):
        return self.dp_shard > 1

    @property
    def cp_enabled(self):
        return self.cp > 1

    @property
    def dp_cp_enabled(self):
        return self.dp_enabled or self.cp_enabled

    @property
    def fsdp_enabled(self):
        return self.dp_shard_enabled or self.cp_enabled

    @property
    def tp_enabled(self):
        return self.tp > 1

    @property
    def pp_enabled(self):
        return self.pp > 1

    @property
    def ep_enabled(self):
        return self.ep > 1

    @property
    def fsdp(self) -> int:
        """FSDP shard degree, including CP (``dp_shard * cp``).

        Consumed by parallelize.py when collecting MoE replicate params; kept
        as a property on top of the TorchTitan core (which computes this inline).
        """
        return self.dp_shard * self.cp

    @property
    def non_data_parallel_size(self):
        return self.cp * self.tp * self.pp

    @property
    def seq_len_divisor(self):
        # Sequence Parallel requires that seq_len be divisible by TP degree.
        # Context Parallel requires that seq_len be divisible by 2 * CP degree,
        # when load balancing is enabled (by default).
        return self.tp * (self.cp * 2)
