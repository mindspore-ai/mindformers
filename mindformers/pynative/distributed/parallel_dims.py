# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file is derived from TorchTitan and adapted for MindSpore.
# Modifications:
#     - Adapted to MindSpore framework: replaced torch's DeviceMesh with mindspore's DeviceMesh.
#     - Add scenarios for EP-reused TP communication groups and EP-reused CP-TP communication groups.
# ============================================================================
""" For parallel dims """
__all__ = [
    "ParallelDims",
]

from dataclasses import dataclass

from hyper_parallel.core.device_mesh import DeviceMesh, init_device_mesh


@dataclass
class ParallelDims:
    """
    Multi-dimensional parallelism configuration.
    """
    dp: int
    cp: int
    tp: int
    pp: int
    ep: int
    etp: int
    world_size: int

    _world_mesh: DeviceMesh = None

    def __post_init__(self):
        self._validate()

    def _validate(self):
        """
        Validate that the parallelism configuration is valid and consistent with the world size.

        - All parallelism degrees (dp, cp, tp, pp, ep, etp) are at least 1.
        - The product of dp * cp * tp * pp equals world_size.
        - etp is currently fixed to 1.
        - ep (expert parallelism) is compatible with tp, cp, and dp according to hierarchical constraints:
            * If ep <= tp: tp must be divisible by ep.
            * If tp < ep <= cp * tp: cp * tp must be divisible by ep.
            * If ep > cp * tp: dp * cp * tp must be divisible by ep.
        """
        dp, cp, tp, pp, ep, etp = (
            self.dp,
            self.cp,
            self.tp,
            self.pp,
            self.ep,
            self.etp,
        )
        for d in (dp, cp, tp, pp, ep, etp):
            if d < 1:
                raise ValueError("Parallelism degree should be >= 1.")

        if dp * cp * tp * pp != self.world_size:
            raise ValueError(
                f"Invalid parallel dims: dp({dp}) * cp({cp}) * tp({tp}) * "
                f"pp({pp}) != WORLD_SIZE({self.world_size})"
            )

        if etp != 1:
            raise ValueError("Currently we only support ETP=1")

        cp_tp = cp * tp
        dp_cp_tp = dp * cp * tp

        if ep <= tp:
            if tp % ep != 0:
                raise ValueError(
                    f"Invalid parallel dims: tp({tp}) % ep({ep}) != 0 when ep <= tp"
                )
        elif tp < ep <= cp_tp:
            if cp_tp % ep != 0:
                raise ValueError(
                    f"Invalid parallel dims: (cp({cp}) * tp({tp})) % ep({ep}) != 0 "
                    f"when tp < ep <= (cp * tp)"
                )
        else:
            if dp_cp_tp % ep != 0:
                raise ValueError(
                    f"Invalid parallel dims: (dp({dp}) * cp({cp}) * tp({tp})) "
                    f"% ep({ep}) != 0 when ep > (cp * tp)"
                )

    def build_mesh(self) -> DeviceMesh:
        """
        Build a DeviceMesh based on whether expert parallelism (ep) is enabled.
        """
        if self.ep > 1:
            return self._build_mesh_with_ep()
        return self._build_mesh_without_ep()

    def _build_mesh_with_ep(self) -> DeviceMesh:
        """
        Construct a DeviceMesh when expert parallelism (ep > 1) is enabled.

        The mesh is constructed differently depending on the relative size of ep compared to tp and cp*tp:
        1. If ep <= tp: split tp into two sub-dimensions — one for non-expert computation and one for ep.
        2. If tp < ep <= cp * tp: split cp dimension to accommodate ep, while keeping tp intact.
        3. If ep > cp * tp: split dp dimension to accommodate ep.

        After constructing the base mesh, it flattens logical dimensions to expose commonly used submeshes:
        - "dp", "cp", "tp": individual parallelism axes
        - "dp_cp": combined data and context parallelism group (used for loss reduction)
        - "ep": expert parallelism group
        """
        # Create all the submesh here to ensure all required process groups are
        # initialized:
        dp_mesh_dim_names = []
        cp_mesh_dim_names = []
        tp_mesh_dim_names = []
        dp_cp_mesh_dim_names = []
        ep_mesh_dim_names = []

        if self.ep <= self.tp:
            tp_shard_mod_ep = self.tp // self.ep
            tp_shard_in_ep = self.ep
            dims = [self.pp, self.dp, self.cp, tp_shard_mod_ep, tp_shard_in_ep]
            names = ["pp", "dp", "cp", "tp_shard_mod_ep", "tp_shard_in_ep"]

            mesh = init_device_mesh(
                device_type="npu",
                mesh_shape=tuple(dims),
                mesh_dim_names=tuple(names)
            )

            tp_mesh_dim_names.append("tp_shard_mod_ep")
            tp_mesh_dim_names.append("tp_shard_in_ep")

            dp_cp_mesh_dim_names.append("dp")
            dp_cp_mesh_dim_names.append("cp")

            ep_mesh_dim_names.append("tp_shard_in_ep")
        elif self.tp < self.ep <= (self.cp * self.tp):
            cp_shard_mod_ep = (self.cp * self.tp) // self.ep
            cp_shard_in_ep = self.ep // self.tp
            dims = [self.pp, self.dp, cp_shard_mod_ep, cp_shard_in_ep, self.tp]
            names = ["pp", "dp", "cp_shard_mod_ep", "cp_shard_in_ep", "tp"]

            mesh = init_device_mesh(
                device_type="npu",
                mesh_shape=tuple(dims),
                mesh_dim_names=tuple(names)
            )

            cp_mesh_dim_names.append("cp_shard_mod_ep")
            cp_mesh_dim_names.append("cp_shard_in_ep")

            dp_cp_mesh_dim_names.append("dp")
            dp_cp_mesh_dim_names.append("cp_shard_mod_ep")
            dp_cp_mesh_dim_names.append("cp_shard_in_ep")

            ep_mesh_dim_names.append("cp_shard_in_ep")
            ep_mesh_dim_names.append("tp")
        else:
            dp_shard_mod_ep = (self.dp * self.cp * self.tp) // self.ep
            dp_shard_in_ep = self.ep // (self.cp * self.tp)
            dims = [self.pp, dp_shard_mod_ep, dp_shard_in_ep, self.cp, self.tp]
            names = ["pp", "dp_shard_mod_ep", "dp_shard_in_ep", "cp", "tp"]

            mesh = init_device_mesh(
                device_type="npu",
                mesh_shape=tuple(dims),
                mesh_dim_names=tuple(names)
            )

            dp_mesh_dim_names.append("dp_shard_mod_ep")
            dp_mesh_dim_names.append("dp_shard_in_ep")

            dp_cp_mesh_dim_names.append("dp_shard_mod_ep")
            dp_cp_mesh_dim_names.append("dp_shard_in_ep")
            dp_cp_mesh_dim_names.append("cp")

            ep_mesh_dim_names.append("dp_shard_in_ep")
            ep_mesh_dim_names.append("cp")
            ep_mesh_dim_names.append("tp")

        if dp_mesh_dim_names:
            mesh[tuple(dp_mesh_dim_names)].flatten(mesh_dim_name="dp")

        if cp_mesh_dim_names:
            mesh[tuple(cp_mesh_dim_names)].flatten(mesh_dim_name="cp")

        if tp_mesh_dim_names:
            mesh[tuple(tp_mesh_dim_names)].flatten(mesh_dim_name="tp")

        if dp_cp_mesh_dim_names:
            mesh[tuple(dp_cp_mesh_dim_names)].flatten(mesh_dim_name="dp_cp")

        if ep_mesh_dim_names:
            mesh[tuple(ep_mesh_dim_names)].flatten(mesh_dim_name="ep")

        return mesh

    def _build_mesh_without_ep(self) -> DeviceMesh:
        """
        Construct a DeviceMesh when expert parallelism is disabled (ep = 1).

        Builds a mesh with dimensions: [pp, dp, cp, tp, ep].
        "ep" mesh shape defaults to 1.
        Flattens the DP and CP dimensions into a combined "dp_cp" submesh,
        which is typically used for cross-entropy loss reduction.
        """
        # Create all the submesh here to ensure all required process groups are
        # initialized:
        dp_cp_mesh_dim_names = []

        dims = [self.pp, self.dp, self.cp, self.tp, self.ep]
        names = ["pp", "dp", "cp", "tp", "ep"]

        mesh = init_device_mesh(
            device_type="npu",
            mesh_shape=tuple(dims),
            mesh_dim_names=tuple(names)
        )

        dp_cp_mesh_dim_names.append("dp")
        dp_cp_mesh_dim_names.append("cp")

        if dp_cp_mesh_dim_names:
            mesh[tuple(dp_cp_mesh_dim_names)].flatten()

        return mesh

    @property
    def world_mesh(self) -> DeviceMesh:
        if self._world_mesh is None:
            self._world_mesh = self.build_mesh()
        return self._world_mesh

    @property
    def dp_enabled(self):
        return self.dp > 1

    @property
    def cp_enabled(self):
        return self.cp > 1

    @property
    def dp_cp_enabled(self):
        return self.dp_enabled or self.cp_enabled

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
    def etp_enabled(self):
        return self.etp > 1
