# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file is derived from TorchTitan and adapted for MindSpore.
# Modifications:
#     - Adapted to MindSpore framework: replaced torch's DeviceMesh with mindspore's DeviceMesh.
#     - Added dp_replicate/dp_shard separation for HSDP support.
#     - Added scenarios for EP-reused TP communication groups and EP-reused CP-TP communication groups.
# ============================================================================
""" For parallel dims """
__all__ = [
    "ParallelDims",
]

from dataclasses import dataclass, field
from typing import List, Optional, Union

from hyper_parallel import DeviceMesh, init_device_mesh
from mindformers.tools.logger import logger


@dataclass
class ParallelDims:
    """
    Multi-dimensional parallelism configuration.
    """
    dp_replicate: int
    dp_shard: int
    cp: int
    tp: int
    pp: int
    ep: int
    etp: int
    world_size: int

    _world_mesh: DeviceMesh = field(default=None, repr=False)

    def __post_init__(self):
        self._validate()

    def _validate(self):
        """
        Validate that the parallelism configuration is valid and consistent with the world size.

        - All parallelism degrees (except dp_shard) are at least 1.
        - dp_shard can be -1 (auto-derived) or >= 1.
        - The product of dp_replicate * dp_shard * cp * tp * pp equals world_size.
        - When ep > 1, etp must be tp or 1 (aligned with TorchTitan).
        """
        dp_replicate, dp_shard, cp, tp, pp, ep, etp = (
            self.dp_replicate,
            self.dp_shard,
            self.cp,
            self.tp,
            self.pp,
            self.ep,
            self.etp,
        )

        for name, d in [
            ("dp_replicate", dp_replicate), ("cp", cp), ("tp", tp),
            ("pp", pp), ("ep", ep), ("etp", etp),
        ]:
            if d < 1:
                raise ValueError(f"{name} should be >= 1.")

        if dp_shard != -1 and dp_shard < 1:
            raise ValueError("dp_shard must be -1 or >= 1.")
        if dp_shard < 0:
            self.dp_shard = dp_shard = self.world_size // (dp_replicate * cp * tp * pp)
        if dp_shard < 1:
            raise ValueError(
                f"Auto-derived dp_shard={dp_shard} is invalid. "
                f"Check world_size and other dims."
            )

        if dp_replicate * dp_shard * cp * tp * pp != self.world_size:
            raise ValueError(
                f"Invalid parallel dims: dp_replicate({dp_replicate}) * dp_shard({dp_shard}) * "
                f"cp({cp}) * tp({tp}) * pp({pp}) != WORLD_SIZE({self.world_size})"
            )

        if ep > 1:
            if etp not in {tp, 1}:
                raise ValueError(
                    f"When ep > 1, etp must be tp({tp}) or 1, got etp={etp}"
                )

        # Note: EP divisibility checks are removed here.
        # Mesh correctness is validated after build_mesh() via _validate_meshes().

    def build_mesh(self) -> DeviceMesh:
        """
        Build a DeviceMesh based on whether expert parallelism (ep) is enabled.
        """
        if self.ep == 1 and self.etp == 1:
            dims = [self.pp, self.dp_replicate, self.dp_shard, self.cp, self.tp]
            names = ["pp", "dp_replicate", "dp_shard", "cp", "tp"]
            mesh = init_device_mesh(
                device_type="npu",
                mesh_shape=tuple(dims),
                mesh_dim_names=tuple(names)
            )

            mesh[("dp_replicate", "dp_shard")].flatten(mesh_dim_name="batch")
            mesh[("dp_replicate", "dp_shard", "cp")].flatten(mesh_dim_name="loss_mesh")
            mesh[("dp_shard", "cp")].flatten(mesh_dim_name="fsdp")

        elif self.ep <= self.tp:
            tp_mod_ep = self.tp // self.ep
            dims = [self.pp, self.dp_replicate, self.dp_shard, self.cp, tp_mod_ep, self.ep]
            names = ["pp", "dp_replicate", "dp_shard", "cp", "tp_mod_ep", "ep"]

            mesh = init_device_mesh(
                device_type="npu",
                mesh_shape=tuple(dims),
                mesh_dim_names=tuple(names)
            )

            mesh[("dp_replicate", "dp_shard")].flatten(mesh_dim_name="batch")
            mesh[("dp_replicate", "dp_shard", "cp")].flatten(mesh_dim_name="loss_mesh")
            mesh[("dp_shard", "cp")].flatten(mesh_dim_name="fsdp")
            mesh[("dp_shard", "cp", "tp_mod_ep")].flatten(mesh_dim_name="efsdp")
            mesh[("tp_mod_ep", "ep")].flatten(mesh_dim_name="tp")
        
        elif self.tp < self.ep <= (self.cp * self.tp):
            cp_tp_mod_ep = (self.cp * self.tp) // self.ep
            ep_mod_tp = self.ep // self.tp
            dims = [self.pp, self.dp_replicate, self.dp_shard, cp_tp_mod_ep, ep_mod_tp, self.tp]
            names = ["pp", "dp_replicate", "dp_shard", "cp_tp_mod_ep", "ep_mod_tp", "tp"]

            mesh = init_device_mesh(
                device_type="npu",
                mesh_shape=tuple(dims),
                mesh_dim_names=tuple(names)
            )

            mesh[("dp_replicate", "dp_shard")].flatten(mesh_dim_name="batch")
            mesh[("dp_replicate", "dp_shard", "cp_tp_mod_ep", "ep_mod_tp")].flatten(mesh_dim_name="loss_mesh")
            mesh[("dp_shard", "cp_tp_mod_ep", "ep_mod_tp")].flatten(mesh_dim_name="fsdp")
            mesh[("dp_shard", "cp_tp_mod_ep")].flatten(mesh_dim_name="efsdp")
            mesh[("cp_tp_mod_ep", "ep_mod_tp")].flatten(mesh_dim_name="cp")
            mesh[("ep_mod_tp", "tp")].flatten(mesh_dim_name="ep")
        
        else:
            efsdp = (self.dp_shard  * self.cp * self.tp) // self.ep
            ep_mod_cp_tp = self.ep // (self.cp * self.tp)
            dims = [self.pp, self.dp_replicate, efsdp, ep_mod_cp_tp, self.cp, self.tp]
            names = ["pp", "dp_replicate", "efsdp", "ep_mod_cp_tp", "cp", "tp"]

            mesh = init_device_mesh(
                device_type="npu",
                mesh_shape=tuple(dims),
                mesh_dim_names=tuple(names)
            )

            mesh[("dp_replicate", "efsdp", "ep_mod_cp_tp")].flatten(mesh_dim_name="batch")
            mesh[("dp_replicate", "efsdp", "ep_mod_cp_tp", "cp")].flatten(mesh_dim_name="loss_mesh")
            mesh[("efsdp", "ep_mod_cp_tp", "cp")].flatten(mesh_dim_name="fsdp")
            mesh[("efsdp", "ep_mod_cp_tp")].flatten(mesh_dim_name="dp_shard")
            mesh[("ep_mod_cp_tp", "cp", "tp")].flatten(mesh_dim_name="ep")

        self._world_mesh = mesh
        self._validate_meshes()
        return mesh

    def _validate_meshes(self):
        """Validate that created meshes have the expected sizes.

        Called after build_mesh() to catch configuration errors that would
        previously require complex pre-validation logic (e.g., EP divisibility).
        """
        expected_sizes = {
            "pp": self.pp,
            "batch": self.dp_replicate * self.dp_shard,
            "dp_replicate": self.dp_replicate,
            "fsdp": self.dp_shard * self.cp,
            "cp": self.cp,
            "tp": self.tp,
        }

        for mesh_name, expected_size in expected_sizes.items():
            try:
                sub_mesh = self._world_mesh[mesh_name]
                actual_size = sub_mesh.size()
                if actual_size != expected_size:
                    raise AssertionError(
                        f"Mesh '{mesh_name}' has unexpected size: "
                        f"expected {expected_size}, got {actual_size}"
                    )
            except KeyError:
                raise RuntimeError(
                    f"Mesh '{mesh_name}' was not created. "
                    f"This indicates a bug in _build_mesh_with_ep() or _build_mesh_without_ep(). "
                    f"Check that all expected flatten() calls use correct dimension names."
                ) from None

    @property
    def world_mesh(self) -> DeviceMesh:
        if self._world_mesh is None:
            self._world_mesh = self.build_mesh()
        return self._world_mesh

    def get_optional_mesh(self, dims: Union[str, List[str]]) -> Optional[DeviceMesh]:
        """Get a device mesh by dimension name(s), returning None if not enabled.

        Args:
            dims: Mesh dimension name(s).

        Returns:
            DeviceMesh for the requested dimension(s), or None if the
            parallelism is not enabled (size == 1).
        """
        if self._world_mesh is None:
            self.build_mesh()

        if isinstance(dims, str):
            dims = [dims]

        if len(dims) == 1:
            return self._world_mesh[dims[0]]
        return self._world_mesh[tuple(dims)]

    def get_mesh(self, dims: Union[str, List[str]]) -> DeviceMesh:
        """Get a device mesh by dimension name(s), raising if not available.

        Same as get_optional_mesh but raises ValueError instead of returning None.
        """
        mesh = self.get_optional_mesh(dims)
        if mesh is None:
            raise ValueError(
                f"Mesh '{dims}' is not available. "
                f"Ensure the corresponding parallelism dimension is enabled (size > 1)."
            )
        return mesh

    @property
    def dp_enabled(self) -> bool:
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self) -> bool:
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self) -> bool:
        return self.dp_shard > 1

    @property
    def fsdp_enabled(self) -> bool:
        """FSDP is enabled when dp_shard > 1 or cp > 1.

        Note: When CP is enabled, FSDP is also applied to utilize its
        weight all-gather and gradient reduce-scatter, even if dp_shard is 1.
        In this case, the fsdp mesh size is dp_shard * cp, and parameters
        are sharded within the CP group.
        """
        return self.dp_shard > 1 or self.cp > 1

    @property
    def cp_enabled(self) -> bool:
        return self.cp > 1

    @property
    def dp_cp_enabled(self) -> bool:
        return self.dp_enabled or self.cp_enabled

    @property
    def tp_enabled(self) -> bool:
        return self.tp > 1

    @property
    def pp_enabled(self) -> bool:
        return self.pp > 1

    @property
    def ep_enabled(self) -> bool:
        return self.ep > 1

    @property
    def etp_enabled(self) -> bool:
        return self.etp > 1

    @property
    def fsdp_gradient_divide_factor(self) -> int:
        """Factor for gradient division in distributed training.

        Equals dp_replicate * dp_shard * cp.
        """
        return self.dp_replicate * self.dp_shard * self.cp
