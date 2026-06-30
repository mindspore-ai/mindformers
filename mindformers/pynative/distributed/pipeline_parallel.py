# Copyright (c) Meta Platforms, Inc. and affiliates
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
import inspect
import importlib
from typing import List, Tuple

import mindspore as ms
from mindspore import nn

from hyper_parallel import PipelineStage

from mindformers.tools.logger import logger

class ScaledLossPipelineStage(PipelineStage):
    """PipelineStage that scales the last-stage main-loss backward seed.

    In the base ``PipelineStage``, ``get_last_stage_sens`` returns ``1.0``
    (or ``1.0 / repeat_num`` for replicated DTensor outputs) as the backward
    seed, which causes gradients to be **summed** across micro-batches.

    However, the single-card training path divides
    ``loss / num_accumulation_steps`` before calling backward. To keep the
    pipeline-parallel path consistent with this behavior -- and to align with
    the scaling already applied to MoE-aux / MTP losses via
    ``main_loss_sense`` -- the framework sets ``loss_scale`` on every stage
    to::

        loss_scale = 1 / (dp * tp * cp) / grad_accum_steps

    and this class multiplies the backward seed by that factor.

    The base ``PipelineStage`` additionally divides DTensor outputs by their
    layout ``repeat_num``. That is correct for a generic replicated DTensor
    backward seed, but MindFormers' main loss scale already includes the TP
    division. For actual replicated loss outputs, multiply the seed by
    ``repeat_num`` before applying ``loss_scale``; local loss outputs are left
    unchanged.

    **Scope of scaling:**
    Only the main-loss seed is scaled here. MoE-aux and MTP gradients are
    already scaled at backward time by their respective auto-scalers, so
    they must **not** be scaled again (doing so on ``param.grad`` would
    double-scale them).
    """

    loss_scale = 1.0

    def __init__(self, *args, tp_mesh=None, **kwargs):
        super().__init__(*args, **kwargs)
        # ``tp_mesh`` (and any future per-axis within-stage submesh) is captured
        # here and NOT forwarded to the base ``PipelineStage``, which has no such
        # parameter. Used by ``_get_layout_rank_list`` below.
        self._axis_meshes = {"tp": tp_mesh} if tp_mesh is not None else {}

    def _get_layout_rank_list(self, layout, sender_rank=None):
        """Resolve a received activation's submesh ranks from explicit axis meshes.

        The base impl resolves the sender's submesh from ``self.mesh.root_mesh``
        by name (e.g. ``root["tp"]``). But mindformers builds ``pp_mesh`` as
        ``dataloading_mesh["pp"]`` whose ``root_mesh`` collapses to the flat 1-D
        ``("world",)`` mesh — that root's flatten-mapping has no ``tp`` axis, so
        the base hits its ``root.ndim <= 1`` "PP-only" guard and wrongly returns
        a single rank, crashing the subsequent ``reshape`` for a TP/SP-sharded
        activation (``mesh_shape=(2,)``).

        We instead resolve any axis present in ``self._axis_meshes`` directly
        from the current rank's submesh, whose ``rank_list`` is exactly this
        rank's group along that axis. Anything else falls back to the base.
        """
        pp_dim_names = set(self.mesh.mesh_dim_names or ()) if self.mesh is not None else set()
        layout_dim_names = tuple(
            name for name in (getattr(layout, "alias_name", None) or ()) if name not in pp_dim_names
        )
        if len(layout_dim_names) == 1 and layout_dim_names[0] in self._axis_meshes:
            return tuple(self._axis_meshes[layout_dim_names[0]].rank_list)
        return super()._get_layout_rank_list(layout, sender_rank)

    def get_last_stage_sens(self, last_stage_outputs):
        """Return the backward seed for the last pipeline stage, scaled by ``loss_scale``.

        The base ``PipelineStage.get_last_stage_sens`` divides DTensor outputs
        by their layout ``repeat_num``. Because MindFormers' main loss scale
        already includes the TP division, the seed for a replicated loss output
        is multiplied back by ``repeat_num`` here (via ``restore_repeat_num``)
        before ``loss_scale`` is applied; local (non-replicated) loss outputs
        are left unchanged. ``loss_scale == 1.0`` short-circuits and returns the
        base seed untouched.
        """
        p_sens = super().get_last_stage_sens(last_stage_outputs)
        scale = self.loss_scale
        if scale == 1.0:
            return p_sens
        if isinstance(last_stage_outputs, (list, tuple)):
            outputs = last_stage_outputs
        else:
            outputs = (last_stage_outputs,)

        def restore_repeat_num(sens, output):
            if hasattr(output, "layout") and hasattr(output.layout, "repeat_num"):
                return sens * output.layout.repeat_num()
            return sens

        if isinstance(p_sens, list):
            return [restore_repeat_num(s, output) * scale for s, output in zip(p_sens, outputs)]
        return restore_repeat_num(p_sens, outputs[0]) * scale


class PpLayerSetting:
    """
    Layer setting for pipeline parallelism.
    """
    def __init__(self, num_layers: int, parallelism_config):
        self.num_layers = num_layers
        self.pp = parallelism_config.pipeline_parallel
        self.pp_interleave_num = parallelism_config.pipeline_parallel_interleave_num
        self.total_stages = self.pp * self.pp_interleave_num
        self.avg_layer = None

        self.layers_per_stage = parallelism_config.pipeline_parallel_layers_per_stage
        if self.layers_per_stage == "auto":
            self._compute_auto()
        else:
            self._init_explicit(self.layers_per_stage)

    def _init_explicit(self, layers_per_stage: List[str]):
        """
        Parse the layer range string to a tuple of (layer_start, layer_end).
        """
        after_parse_layer = {}
        if len(layers_per_stage) != self.pp:
            raise ValueError(
                f"The length of 'pipeline_parallel_layers_per_stage' ({len(layers_per_stage)}) "
                f"must equal 'pipeline_parallel' ({self.pp}), "
                f"but got {len(layers_per_stage)} entries for {self.pp} PP ranks."
            )
        for pp_rank, stage_str in enumerate(layers_per_stage):
            ranges = stage_str.split(',')
            if len(ranges) != self.pp_interleave_num:
                raise ValueError(
                    f"PP rank {pp_rank} has {len(ranges)} chunk(s) in "
                    f"'pipeline_parallel_layers_per_stage' (got '{stage_str}'), "
                    f"but 'pipeline_parallel_interleave_num' is {self.pp_interleave_num}. "
                    f"Each PP rank must have exactly {self.pp_interleave_num} comma-separated chunk(s)."
                )
            for chunk_id, r in enumerate(ranges):
                layer_ids = r.strip().split('-')
                if len(layer_ids) <= 2:
                    try:
                        layer_start, layer_end = int(layer_ids[0]), int(layer_ids[-1])
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid layer range format: {r}. "
                            f"Layer IDs must be integers, got '{layer_ids[0]}' and '{layer_ids[-1]}'"
                        ) from exc
                    if layer_start > layer_end:
                        raise ValueError(
                            f"Invalid layer range format: {r}. "
                            "layer_start must be less than or equal to layer_end."
                        )
                else:
                    raise ValueError(
                        f"Invalid layer range format: {r}. "
                        "Expected format is 'start-end' or 'single_layer'."
                    )
                after_parse_layer[str(chunk_id * self.pp + pp_rank)] = (layer_start, layer_end)

        self.layers_per_stage = after_parse_layer

    def _compute_auto(self):
        if self.num_layers % self.total_stages != 0:
            raise ValueError(
                f"Number of layers ({self.num_layers}) is not divisible by total stages ({self.total_stages})."
                f"Please configure the model distribution via 'pipeline_parallel_layers_per_stage'." 
            )
        self.avg_layer = self.num_layers // self.total_stages

    def get_layer_range(self, stage_id: int) -> Tuple[int, int]:
        if self.avg_layer:
            layer_start_id = stage_id * self.avg_layer
            layer_end_id = (stage_id + 1) * self.avg_layer - 1
            return (layer_start_id, layer_end_id)
        return self.layers_per_stage[str(stage_id)]

    @property
    def num_virtual_stages(self) -> int:
        return self.pp * self.pp_interleave_num

class StageModelBuilder:
    """
    Builder for creating pipeline parallel stages.
    """
    def __init__(self, layer_setting: PpLayerSetting):
        self.layer_setting = layer_setting

    def build_stages(
        self,
        model_cls: type,
        model_config,
        pp_mesh,
        tp_mesh=None,
    ) -> tuple[list[PipelineStage], list[nn.Cell]]:
        """
        Build pipeline parallel stages.
        """
        num_virtual_stages = self.layer_setting.num_virtual_stages
        local_stage_indices = self._get_local_stage_indices(pp_mesh)
        
        stages = []
        model_parts = []
        
        for stage_idx in local_stage_indices:
            layer_start, layer_end = self.layer_setting.get_layer_range(stage_idx)
            is_first = stage_idx == 0
            is_last = stage_idx == num_virtual_stages - 1
            
            with ms.DeviceCtx("meta"):
                stage_model = model_cls(
                    model_config,
                    layer_start=layer_start,
                    layer_end=layer_end,
                    pre_process=is_first,
                    post_process=is_last,
                    stage_idx=stage_idx,
                    vp_size=num_virtual_stages,
                )
            model_parts.append(stage_model)
            
            stage = ScaledLossPipelineStage(
                submodule=stage_model,
                stage_index=stage_idx,
                stage_num=num_virtual_stages,
                device=pp_mesh.device_type,
                group=pp_mesh.get_group(),
                mesh=pp_mesh,
                tp_mesh=tp_mesh,
            )
            stages.append(stage)
        
        return stages, model_parts

    def _get_local_stage_indices(self, pp_mesh) -> list[int]:
        pp_rank = pp_mesh.get_local_rank()
        pp_size = pp_mesh.size()
        interleave_num = self.layer_setting.pp_interleave_num
        return [chunk_id * pp_size + pp_rank for chunk_id in range(interleave_num)]


def _infer_schedule_type(parallelism):
    """Infer the schedule type based on parallelism config.

    - If ``pipeline_parallel_schedule`` is explicitly set (not empty/"auto"), return it.
    - Otherwise default to ``"1f1b"``.
    - When the result is ``"1f1b"`` but ``interleave_num > 1``, auto-upgrade to
      ``"interleaved_1f1b"`` because 1F1B does not support interleaved stages.
    """
    schedule_type = getattr(parallelism, "pipeline_parallel_schedule", None)
    if schedule_type and schedule_type not in (""):
        schedule_type = schedule_type.lower()
    else:
        schedule_type = "1f1b"

    if schedule_type == "1f1b" and parallelism.pipeline_parallel_interleave_num > 1:
        schedule_type = "interleaved_1f1b"

    return schedule_type


_SCHEDULE_REGISTRY = {
    "1f1b": ("hyper_parallel.core.pipeline_parallel", "Schedule1F1B"),
    "interleaved_1f1b": ("hyper_parallel.core.pipeline_parallel", "ScheduleInterleaved1F1B"),
    "gpipe": ("hyper_parallel.core.pipeline_parallel", "ScheduleGPipe"),
}


def register_schedule(name, module_path, class_name):
    """Register a custom pipeline schedule type at runtime.

    Args:
        name (str): Schedule type name (e.g. ``"my_schedule"``).
        module_path (str): Dotted module path to import from
            (e.g. ``"my_pkg.schedules"``).
        class_name (str): Name of the Schedule class inside the module
            (e.g. ``"ScheduleMyAlgo"``).
    """
    key = name.lower()
    if key in _SCHEDULE_REGISTRY:
        raise ValueError(
            f"Schedule '{key}' is already registered. "
            f"Existing: {_SCHEDULE_REGISTRY[key]}."
        )
    _SCHEDULE_REGISTRY[key] = (module_path, class_name)


def _resolve_schedule_cls(schedule_type):
    """Resolve a schedule type string to its class via the registry.

    Performs a lazy import so that ``hyper_parallel`` (and its platform
    backends) are only loaded when a schedule is actually instantiated.
    """
    key = schedule_type.lower()
    if key not in _SCHEDULE_REGISTRY:
        supported = ", ".join(sorted(_SCHEDULE_REGISTRY))
        raise ValueError(
            f"Unsupported pipeline_parallel_schedule '{schedule_type}'. "
            f"Supported values: {supported}"
        )
    module_path, class_name = _SCHEDULE_REGISTRY[key]
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Failed to import Schedule class '{class_name}' from "
            f"'{module_path}' for schedule type '{schedule_type}'. "
            f"Ensure the module and class exist. Error: {e}"
        ) from e


def _create_schedule(schedule_type, stages, micro_batch_num, parallelism, swap=False, has_moe=True):
    """Create a pipeline schedule instance based on the specified schedule type.

    Uses ``inspect.signature`` to detect the constructor parameters of each
    Schedule class so that only the arguments it actually accepts are passed.
    This avoids hard-coding parameter lists for every Schedule variant and
    automatically adapts to future additions.

    Supported schedule types (from ``hyper_parallel.core.pipeline_parallel``):

    * ``"1f1b"``            → :class:`Schedule1F1B`
    * ``"interleaved_1f1b"`` → :class:`ScheduleInterleaved1F1B`
    * ``"gpipe"``           → :class:`ScheduleGPipe`

    Args:
        schedule_type (str): Schedule type name.
        stages: Pipeline stages.
        micro_batch_num (int): Number of micro-batches.
        parallelism: Parallelism config object.
        swap (bool): Whether to enable activation swap.
        has_moe (bool): Whether the model contains MoE layers.  When
            ``False`` (dense-only model), ``overlap_b_f`` and
            ``enable_dxdw_split`` are forced to ``False`` because there is
            no EP alltoall communication to overlap with compute.
    """
    schedule_type = schedule_type.lower()
    interleave_num = parallelism.pipeline_parallel_interleave_num

    if schedule_type == "1f1b" and interleave_num > 1:
        raise ValueError(
            f"pipeline_parallel_schedule='1f1b' is incompatible with "
            f"pipeline_parallel_interleave_num={interleave_num}. "
            f"Use 'interleaved_1f1b' schedule instead, or set interleave_num=1."
        )

    if schedule_type == "interleaved_1f1b" and interleave_num <= 1:
        raise ValueError(
            f"pipeline_parallel_schedule='{schedule_type}' requires "
            f"pipeline_parallel_interleave_num > 1, but got {interleave_num}. "
            f"Use '1f1b' schedule instead, or set interleave_num > 1."
        )

    schedule_cls = _resolve_schedule_cls(schedule_type)

    overlap_b_f = getattr(parallelism, "pipeline_parallel_overlap_b_f", False)
    enable_dxdw_split = getattr(parallelism, "pipeline_parallel_enable_dxdw_split", False)

    if not has_moe and overlap_b_f:
        logger.warning(
            "overlap_b_f=True is set but the model has no MoE layers (dense-only). "
            "Forcing overlap_b_f=False because there is no EP alltoall "
            "communication to overlap with compute."
        )
        overlap_b_f = False
        enable_dxdw_split = False

    if not overlap_b_f and enable_dxdw_split:
        logger.warning(
            "enable_dxdw_split=True requires overlap_b_f=True, but overlap_b_f "
            "is False. Forcing enable_dxdw_split=False."
        )
        enable_dxdw_split = False

    all_kwargs = {
        "overlap_p2p": getattr(parallelism, "pipeline_parallel_overlap_p2p", False),
        "overlap_b_f": overlap_b_f,
        "enable_dxdw_split": enable_dxdw_split,
        "p2p_transport": getattr(parallelism, "pipeline_parallel_p2p_transport", "plain"),
        "swap": swap,
    }

    sig = inspect.signature(schedule_cls.__init__)
    accepted_params = set(sig.parameters) - {"self"}
    filtered_kwargs = {k: v for k, v in all_kwargs.items() if k in accepted_params}

    return schedule_cls(stages, micro_batch_num, **filtered_kwargs)
