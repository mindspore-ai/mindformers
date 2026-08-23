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
"""Load/save checkpoint APIs for distributed parallel layout management."""
from typing import Dict, List, Optional, Tuple, Union

import mindspore as ms
from mindspore import Parameter
from mindspore.nn.cell import Cell
from mindspore.mint.distributed import gather_object
from mindspore.parallel.strategy import get_strategy_metadata

from mindformers.core.context.build_context import get_context
from mindformers.tools.logger import logger

try:
    from hyper_parallel.core.distributed_checkpoint import get_global_layout, get_current_layout
    from hyper_parallel.core.dtensor.dtensor import DTensor
except ImportError as e:
    get_global_layout = None
    get_current_layout = None
    DTensor = None


from mindformers.tools.utils import get_real_rank, get_real_group_size


class LayoutAdapter:
    """
    Adapter class for extracting and managing distributed parallel layout information from MindSpore networks.

    This class provides unified interfaces to retrieve sharding strategy metadata across different
    parallel modes (PyNative and Graph mode), supporting both single-rank and multi-rank scenarios.
    """

    @staticmethod
    def is_pynative_mode() -> bool:
        """
        Check if the current execution mode is PyNative mode.

        Returns:
            bool: True if running in PyNative mode, False if in Graph mode.
        """
        return get_context("mode") == ms.context.PYNATIVE_MODE

    @staticmethod
    def get_all_layouts(network: Union[Cell, List[Cell]]) -> Dict[int, Dict[str, list]]:
        """
        Retrieve distributed parallel layout information for all ranks in the network.

        This method automatically detects the execution mode and delegates to the appropriate
        backend implementation to collect layout metadata across all participating ranks.

        Args:
            network (Cell): The MindSpore network cell containing distributed parameters
                and their sharding strategies.

        Returns:
            Dict[int, Dict[str, list]]: A nested dictionary where:
                - Outer keys are rank IDs (int).
                - Inner dictionaries map parameter names (str) to layout information lists containing:
                    [ms.Layout object, parameter type, full shape].
                Returns empty dict if no layout information is available.
        """
        if LayoutAdapter.is_pynative_mode():
            return LayoutAdapter._get_layout_from_pynative(network)
        return LayoutAdapter._get_layout_from_graph(network)


    @staticmethod
    def get_current_layout(network: Union[Cell, List[Cell]]) -> Dict[str, list]:
        """
        Retrieve distributed parallel layout information for the current rank only.

        This method extracts layout metadata specific to the executing rank, which is useful
        for rank-specific checkpoint operations and local tensor reconstruction.

        In PyNative mode this is a purely local operation (no collective communication):
        it directly uses the current-rank layout provided by the hyper_parallel framework
        instead of gathering the global layout and indexing into it. This avoids the
        O(world_size) memory and communication overhead per rank on large-scale clusters.

        Args:
            network (Cell): The MindSpore network cell containing distributed parameters
                and their sharding strategies.

        Returns:
            Dict[str, list]: A dictionary mapping parameter names (str) to layout information
                lists containing [ms.Layout object, parameter type, full shape] for the
                current rank. Returns empty dict if no layout information is available.
        """
        rank_id = get_real_rank()
        if LayoutAdapter.is_pynative_mode():
            return LayoutAdapter._get_current_layout_from_pynative(network)
        return LayoutAdapter._get_layout_from_graph(network)[rank_id]

    @staticmethod
    def get_all_layouts_on_rank0(
            network: Union[Cell, List[Cell]]
    ) -> Optional[Tuple[Dict[str, list], Dict[int, List[str]]]]:
        """
        Collect a deduplicated layout index of the whole job onto rank 0 only.

        Gathering every rank's full layout would make rank 0 receive
        ``world_size`` copies of each parameter's layout, each one holding a rank list of
        ``world_size`` entries: the payload and the work on rank 0 are both quadratic in the world
        size, which costs tens of minutes on a 10K-card job.

        A parameter's layout describes its *global* sharding, so it is byte-for-byte identical on
        every rank that owns the parameter, and the offsets it yields already cover all ranks. One
        copy per parameter is therefore enough to rebuild the layout of every rank.

        In PyNative mode, each rank sends only its parameter names plus the layouts it owns
        (the lowest rank of the parameter's rank list is chosen as the owner, decided locally
        without communication). In Graph mode, the strategy metadata is produced locally by the
        compiler on every rank, so rank 0 simply builds the index from its local view. Either
        way the payload and rank 0's work are linear in the world size.

        Note:
            All ranks must call this interface. In PyNative multi-rank scenarios it performs a
            ``gather_object`` towards rank 0 (non-zero ranks take part in the gather and simply
            get None). In Graph mode only rank 0 builds the result locally, without any
            communication.

        Args:
            network (Cell): The MindSpore network cell containing distributed parameters
                and their sharding strategies.

        Returns:
            Optional[Tuple[Dict[str, list], Dict[int, List[str]]]]: On rank 0 (or in
                single-rank jobs), a tuple of
                - the layout of each parameter, `{param_name: [ms.Layout or None, type, full_shape]}`
                - the parameter names owned by each rank, `{rank_id: [param_name, ...]}`, in the
                  order the owning rank reported them.
                On all other ranks, None.

        Raises:
            RuntimeError: On rank 0, if a rank reported a parameter whose layout no rank claimed
                ownership of, which would silently drop that parameter from 'metadata.json'.
        """
        if not LayoutAdapter.is_pynative_mode():
            # Graph mode strategy metadata is produced locally by the compiler on every
            # rank without any communication, so only rank 0 needs to build the index.
            if get_real_rank() != 0:
                return None
            return LayoutAdapter._build_layout_index_from_graph(network)

        if get_current_layout is None:
            raise ImportError("hyper_parallel is required for PyNative mode. Please install it.")

        local_layout_dict = LayoutAdapter._get_raw_current_layout_from_pynative(network)
        contribution = LayoutAdapter._build_local_layout_contribution(local_layout_dict)

        if get_real_group_size() == 1:
            return LayoutAdapter._merge_layout_contributions([contribution])

        if get_real_rank() != 0:
            gather_object(contribution, None, dst=0)
            return None

        gathered = [None] * get_real_group_size()
        gather_object(contribution, gathered, dst=0)
        return LayoutAdapter._merge_layout_contributions(gathered)

    @staticmethod
    def _build_layout_index_from_graph(
            network: Union[Cell, List[Cell]]
    ) -> Tuple[Dict[str, list], Dict[int, List[str]]]:
        """
        Build the deduplicated layout index from Graph-mode strategy metadata (rank 0 only).

        The compiler produces the full strategy metadata locally on every rank, so the index
        is extracted from the local view without any communication. A parameter's layout is
        identical on every rank that owns it, so the first occurrence is kept.
        """
        graph_layout = LayoutAdapter._get_layout_from_graph(network)
        param_layouts = {}
        rank_param_names = {}
        for rank_id, params in graph_layout.items():
            rank_param_names[int(rank_id)] = list(params.keys())
            for param_name, param_info in params.items():
                if param_name not in param_layouts:
                    param_layouts[param_name] = param_info
        return param_layouts, rank_param_names

    @staticmethod
    def _build_local_layout_contribution(
            local_layout_dict: Dict[str, Dict[str, dict]]
    ) -> Tuple[Dict[int, List[str]], Dict[str, dict]]:
        """
        Build what the current rank contributes to the global layout index.

        Args:
            local_layout_dict (Dict[str, Dict[str, dict]]): Raw current-rank layout dictionary, as
                returned by `_get_raw_current_layout_from_pynative`.

        Returns:
            Tuple[Dict[int, List[str]], Dict[str, dict]]: The parameter names held by this rank
                (keyed by rank ID, order preserved), and the raw layouts this rank owns. A sharded
                parameter is owned by the lowest rank of its rank list, so exactly one rank sends
                it; an unsharded parameter has no rank list and is sent by every rank holding it,
                which is cheap because its raw layout is just a type and a shape.
        """
        cur_rank = get_real_rank()
        rank_param_names: Dict[int, List[str]] = {}
        owned_layouts: Dict[str, dict] = {}
        for rank_id, param_layouts in local_layout_dict.items():
            rank_param_names[int(rank_id)] = list(param_layouts.keys())
            for param_name, param_info in param_layouts.items():
                if param_name in owned_layouts:
                    continue
                rank_list = param_info.get('rank_list') if 'device_matrix' in param_info else None
                if rank_list is None or min(rank_list) == cur_rank:
                    owned_layouts[param_name] = param_info
        return rank_param_names, owned_layouts

    @staticmethod
    def _merge_layout_contributions(
            contributions: List[Tuple[Dict[int, List[str]], Dict[str, dict]]]
    ) -> Tuple[Dict[str, list], Dict[int, List[str]]]:
        """
        Merge the per-rank contributions into the global layout index on rank 0.

        Args:
            contributions: One `(rank_param_names, owned_layouts)` pair per rank, as built by
                `_build_local_layout_contribution`.

        Returns:
            Tuple[Dict[str, list], Dict[int, List[str]]]: The converted layout of each parameter,
                and the parameter names owned by each rank.

        Raises:
            RuntimeError: If a rank reported a parameter that no rank claimed ownership of.
        """
        raw_layouts: Dict[str, dict] = {}
        rank_param_names: Dict[int, List[str]] = {}
        for contribution in contributions:
            if not contribution:
                continue
            cur_rank_param_names, owned_layouts = contribution
            rank_param_names.update(cur_rank_param_names)
            for param_name, param_info in owned_layouts.items():
                if param_name not in raw_layouts:
                    raw_layouts[param_name] = param_info

        reported = set()
        for param_names in rank_param_names.values():
            reported.update(param_names)
        missing = reported - raw_layouts.keys()
        if missing:
            raise RuntimeError(
                f"The layout of {len(missing)} parameter(s) was reported by some rank but owned by none, "
                f"for instance {sorted(missing)[:5]}. This means the owner rank did not take part in the "
                f"gather, and 'metadata.json' would be incomplete."
            )

        logger.info(
            ".........Layout index gathered: %d distinct parameters over %d ranks (%d parameter "
            "instances).........",
            len(raw_layouts), len(rank_param_names), sum(len(n) for n in rank_param_names.values())
        )

        # Each parameter's ms.Layout is built once here and shared by every rank that owns it.
        return LayoutAdapter._convert_rank_layout_dict(raw_layouts), rank_param_names

    @staticmethod
    def _get_raw_current_layout_from_pynative(network: Union[Cell, List[Cell]]) -> Dict[str, Dict[str, dict]]:
        """
        Extract the raw current-rank layout dictionary from PyNative mode execution.

        This is a purely local operation (no collective communication). When multiple
        networks are given, their current-rank layout dictionaries are merged.

        Args:
            network (Cell): The MindSpore network cell containing distributed parameters
                and their sharding strategies in PyNative mode.

        Returns:
            Dict[str, Dict[str, dict]]: A dictionary keyed by the current rank ID (str),
                whose values map parameter names to raw layout info dicts as provided by
                the hyper_parallel framework.
        """
        if get_current_layout is None:
            raise ImportError("hyper_parallel is required for PyNative mode. Please install it.")
        local_layout_dict = {}
        network = network if isinstance(network, list) else [network]
        for net in network:
            layout_dict = get_current_layout(net)
            for rank_id, metas in layout_dict.items():
                if rank_id in local_layout_dict:
                    local_layout_dict[rank_id].update(metas)
                else:
                    local_layout_dict[rank_id] = metas
        return local_layout_dict

    @staticmethod
    def _get_current_layout_from_pynative(network: Union[Cell, List[Cell]]) -> Dict[str, list]:
        """
        Extract converted current-rank layout information from PyNative mode execution.

        Purely local operation; see `get_current_layout` for the public contract.
        """
        local_layout_dict = LayoutAdapter._get_raw_current_layout_from_pynative(network)
        if not local_layout_dict:
            return {}
        result = LayoutAdapter._convert_global_layout_dict(local_layout_dict)
        return result.get(get_real_rank(), {})

    @staticmethod
    def _convert_rank_layout_dict(current_layout_dict: Dict[str, dict]) -> Dict[str, list]:
        """
        Convert one rank's raw layout dictionary into structured layout information lists.

        Args:
            current_layout_dict (Dict[str, dict]): Raw parameter layout information of a
                single rank, as provided by the hyper_parallel framework.

        Returns:
            Dict[str, list]: A dictionary mapping parameter names (str) to layout
                information lists: [ms.Layout object with device_matrix/alias_name/rank_list/
                tensor_map, parameter type, full shape].
        """
        rank_layout = {}
        for param_name, param_info in current_layout_dict.items():
            if 'device_matrix' not in param_info:
                rank_layout[param_name] = [None, param_info['type'], param_info['full_shape']]
            else:
                layout_info = ms.Layout(
                    device_matrix=tuple(param_info['device_matrix']),
                    alias_name=tuple(param_info['alias_name']),
                    rank_list=list(param_info['rank_list'])
                )
                layout_info._tensor_map = tuple(param_info['tensor_map'])
                rank_layout[param_name] = [layout_info, param_info['type'], param_info['full_shape']]
        return rank_layout

    @staticmethod
    def _convert_global_layout_dict(global_layout_dict: Dict) -> Dict[int, Dict[str, list]]:
        """
        Convert a raw global layout dictionary (keyed by rank ID) into structured form.

        Args:
            global_layout_dict (Dict): Raw global layout dictionary keyed by rank ID
                (int or str), whose values are raw per-rank layout dictionaries.

        Returns:
            Dict[int, Dict[str, list]]: The converted global layout dictionary keyed by
                rank ID (int).
        """
        result = {}
        for rank_id, current_layout_dict in global_layout_dict.items():
            result[int(rank_id)] = LayoutAdapter._convert_rank_layout_dict(current_layout_dict)
        return result

    @staticmethod
    def _get_layout_from_pynative(network: Cell) -> Dict[int, Dict[str, list]]:
        """
        Extract distributed parallel layout information from PyNative mode execution.

        This method retrieves global layout metadata from the hyper_parallel framework and
        constructs ms.Layout objects for each parameter across all ranks. It handles the
        conversion of raw layout dictionaries into structured Layout instances with proper
        tensor map configuration.

        Args:
            network (Cell): The MindSpore network cell containing distributed parameters
                and their sharding strategies in PyNative mode.

        Returns:
            Dict[int, Dict[str, list]]: A nested dictionary where:
                - Outer keys are rank IDs (int).
                - Inner dictionaries map parameter names (str) to layout information lists:
                    [ms.Layout object with device_matrix/alias_name/rank_list/tensor_map,
                     parameter type, full shape].
                Returns empty dict if global layout is not available.
        """
        if get_global_layout is None:
            raise ImportError("hyper_parallel is required for PyNative mode. Please install it.")
        global_layout_dict = {}
        network = network if isinstance(network, list) else [network]
        for net in network:
            layout_dict = get_global_layout(net)
            for rank_id, metas in layout_dict.items():
                if rank_id in global_layout_dict:
                    global_layout_dict[rank_id].update(metas)
                else:
                    global_layout_dict[rank_id] = metas

        if not global_layout_dict:
            return {}
        return LayoutAdapter._convert_global_layout_dict(global_layout_dict)

    @staticmethod
    def _get_layout_from_graph(network: Cell) -> Dict[int, Dict[str, list]]:
        """
        Extract distributed parallel layout information from Graph mode execution.

        This method retrieves strategy metadata that was generated during graph compilation,
        which contains the optimized sharding decisions made by MindSpore's parallel compiler.

        Args:
            network (Cell): The MindSpore network cell containing distributed parameters
                and their compiled sharding strategies in Graph mode.

        Returns:
            Dict[int, Dict[str, list]]: A nested dictionary where:
                - Outer keys are rank IDs (int).
                - Inner dictionaries map parameter names (str) to layout information lists
                  obtained from the compiled strategy metadata.
        """
        return get_strategy_metadata(network)

    @staticmethod
    def preprocess_params(network: Cell):
        """
        Preprocess network parameters by converting DTensor types to Parameter in PyNative mode.

        This function is primarily used for parameter conversion in distributed training scenarios.
        In PyNative mode, if parameters are of DTensor type (distributed tensor), they are converted
        to Parameter form for subsequent saving or processing operations.

        Args:
            network (Cell): MindSpore neural network cell object containing parameters to be processed

        Returns:
            state_dict (Dict[str, Parameter]): Returns the original network object if not in PyNative mode;
                                If in PyNative mode, returns a dictionary where keys are parameter names
                                and values are processed parameter objects (DTensor converted to Parameter)
        """
        if not LayoutAdapter.is_pynative_mode():
            return network
        if DTensor is None:
            raise ImportError("DTensor is required for PyNative mode. Please install it.")
        state_dict = {}
        network = network if isinstance(network, list) else [network]
        for net in network:
            for param_name, param in net.parameters_dict().items():
                if isinstance(param, DTensor):
                    param_value = Parameter([])
                    param_value.data = param.to_local()
                    param_value.name = param_name
                    param_value.requires_grad = param.requires_grad
                    state_dict[param.name] = param_value
                else:
                    state_dict[param.name] = param

        return state_dict
