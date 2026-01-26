# Copyright 2025 Huawei Technologies Co., Ltd
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
"""resharding tensor"""

import os
import copy
from time import time
from concurrent.futures import ThreadPoolExecutor

import operator
from typing import Dict, List, Tuple, Optional, Any, Union
from functools import reduce
import numpy as np

from mindspore import Parameter, Tensor
from mindspore import load_checkpoint as ms_load_checkpoint

from mindformers.tools.logger import logger
from mindformers.tools.utils import get_real_rank
from mindformers.checkpoint.sharded_tensor import ShardedTensor
from mindformers.checkpoint.utils import get_sharded_tensor_shard_id


def check_layout(layout: Optional[Any], name: str) -> None:
    """
    Validates that a layout contains required attributes with correct types.

    Args:
        layout: Layout object to validate
        name: Name of the layout (for error messages)

    Raises:
        ValueError: If layout missing required attributes or has size mismatches
        TypeError: If layout components are not tuples/lists
    """
    if not layout:
        return

    # Check for required attributes
    required_attrs = ['_device_shape', '_tensor_map', '_rank_list']
    for attr in required_attrs:
        if not hasattr(layout, attr):
            raise ValueError(
                f"Layout {name} must contain attribute {attr}"
            )

    # Validate component types
    def check_type_is_sequence(obj: Any, obj_name: str) -> None:
        if not isinstance(obj, (tuple, list)):
            raise TypeError(
                f"Layout {name} {obj_name} must be tuple or list, "
                f"but got {type(obj).__name__}"
            )

    layout_dict = layout.to_dict()
    check_type_is_sequence(layout_dict['device_matrix'], 'device_matrix')
    check_type_is_sequence(layout_dict['tensor_map'], 'tensor_map')
    check_type_is_sequence(layout_dict['rank_list'], 'rank_list')

    # Validate rank list size matches device count
    dev_num = reduce(operator.mul, layout_dict['device_matrix'])
    if len(layout_dict['rank_list']) != dev_num:
        raise ValueError(
            f"Layout {name} rank_list size ({len(layout_dict['rank_list'])}) "
            f"must match device count ({dev_num})"
        )


def rank_id_to_dev_id_list(dev_matrix: Tuple[int, ...], rank_id: int) -> List[int]:
    """
    Converts a rank ID to a list of device IDs based on the device matrix.

    Args:
        dev_matrix: Shape of the device matrix
        rank_id: Global rank ID to convert

    Returns:
        List of device IDs corresponding to the rank
    """
    dims = len(dev_matrix)
    dev_id_list = [0] * dims

    for i in range(dims - 1, -1, -1):
        dev_id_list[i] = rank_id % dev_matrix[i]
        rank_id = rank_id // dev_matrix[i]

    return dev_id_list


def infer_intersection(
        area_a: Tuple[Tuple[int, int], ...],
        area_b: Tuple[Tuple[int, int], ...]
) -> Optional[Tuple[Tuple[int, int], ...]]:
    """
    Calculates the intersection of two tensor slice areas.

    Args:
        area_a: First area to intersect
        area_b: Second area to intersect

    Returns:
        Tuple of intersection boundaries or None if no intersection
    """

    # Validate input formats
    def is_valid_axis_list(axis_list: Any) -> None:
        if not isinstance(axis_list, (tuple, list)):
            raise TypeError("Area must be a tuple of ranges")
        for axis_range in axis_list:
            if (not isinstance(axis_range, (tuple, list)) \
                    or len(axis_range) != 2):
                raise TypeError("Each axis range must be a 2-element tuple")

    is_valid_axis_list(area_a)
    is_valid_axis_list(area_b)

    # Check dimension compatibility
    if len(area_a) != len(area_b):
        raise ValueError(
            f"Area dimension mismatch: {len(area_a)} vs {len(area_b)}"
        )

    # Calculate intersection for each dimension
    intersection: List[Tuple[int, int]] = []
    for axis_range_a, axis_range_b in zip(area_a, area_b):
        left = max(axis_range_a[0], axis_range_b[0])
        right = min(axis_range_a[1], axis_range_b[1])

        if left >= right:  # No intersection in this dimension
            return None

        intersection.append((left, right))

    return tuple(intersection)


def infer_slice_area_by_rank(
        dev_matrix: Tuple[int, ...],
        tensor_map: Union[List[int], Tuple[int, ...]],
        rank_id: int,
        full_shape: Tuple[int, ...]
) -> Tuple[Tuple[int, int], ...]:
    """
    Calculates the tensor slice boundaries for a specific rank.

    Args:
        dev_matrix: Shape of the device matrix
        tensor_map: Mapping of tensor dimensions to device dimensions
        rank_id: Rank ID to calculate slice for
        full_shape: Complete shape of the original tensor

    Returns:
        Tuple of (start, end) boundaries for each tensor dimension
    """

    # Helper to get device count along a dimension
    def _get_dev_num_along_dim(dim: int) -> int:
        return dev_matrix[-dim - 1] if dim != -1 else 1

    dims = len(full_shape)
    dev_id_list = rank_id_to_dev_id_list(dev_matrix, rank_id)
    area: List[Tuple[int, int]] = []

    for axis in range(dims):
        mapping = tensor_map[axis]
        if isinstance(mapping, int):
            mapping = (mapping,)  # Convert to tuple for consistent handling

        # Calculate total number of splits for this axis
        split_num = 1
        for dim in mapping:
            split_num *= _get_dev_num_along_dim(dim)

        # Calculate slice ID for this rank
        slice_id = 0
        coef = 1
        for dim in reversed(mapping):
            if dim == -1:
                continue
            slice_id += dev_id_list[-dim - 1] * coef
            coef *= _get_dev_num_along_dim(dim)

        # Calculate start/end indices for this slice
        slice_size = full_shape[axis] // split_num
        start = slice_id * slice_size
        end = start + slice_size
        area.append((start, end))

    return tuple(area)


class ReshardHandler:
    """
    Handles tensor resharding between different distributed layouts.

    This class manages the process of reshaping and redistributing tensors between
    different parallel layouts. It calculates necessary tensor slices, validates
    input layouts, and assembles the final tensor for the target rank.

    Args:
        param_name: Name of the parameter (without pipeline stage prefix)
        full_shape: Complete shape of the tensor before sharding
        from_layout: Source layout containing device matrix, tensor map, and rank list
        to_layout: Target layout containing device matrix, tensor map, and rank list
        to_rank_id: Target rank ID to receive the resharded tensor

    Raises:
        ValueError: If both layouts are None or layouts contain invalid attributes
        TypeError: If layout components are not tuples/lists
    """

    def __init__(
            self,
            param_name: str,
            full_shape: Tuple[int, ...],
            from_layout: Optional[Any],
            to_layout: Optional[Any],
            to_rank_id: int
    ):
        # Validate input layouts
        check_layout(from_layout, 'from_layout')
        check_layout(to_layout, 'to_layout')

        # Initialize basic attributes
        self.param_name = param_name
        self.full_shape = full_shape

        # Process source layout configuration
        if from_layout is None:
            self.from_dev_matrix = (1,)
            self.from_tensor_map = tuple(0 for _ in full_shape)
            self.from_rank_list = [0]
        else:
            from_layout_dict = from_layout.to_dict()
            self.from_dev_matrix = from_layout_dict["device_matrix"]
            self.from_tensor_map = from_layout_dict["tensor_map"]
            self.from_rank_list = from_layout_dict["rank_list"]

        # Process target layout configuration
        if to_layout is None:
            self.to_dev_matrix = (1,)
            self.to_tensor_map = tuple(0 for _ in full_shape)
            self.to_rank_list = [0]
            self.to_rank_id = 0
        else:
            to_layout_dict = to_layout.to_dict()
            self.to_dev_matrix = to_layout_dict["device_matrix"]
            self.to_tensor_map = to_layout_dict["tensor_map"]
            self.to_rank_list = to_layout_dict["rank_list"]
            self.to_rank_id = to_rank_id

        # Calculate device counts and internal rank mappings
        self.from_dev_num = len(self.from_rank_list)
        self.inner_from_rank_list = range(self.from_dev_num)
        self.inner_to_rank_id = self.to_rank_list.index(self.to_rank_id)

        # Compute redundancy information
        self.inner_deredundancy_from_rank_list = (
            self._infer_inner_deredundancy_rank_list_by_from_layout()
            if from_layout else [0]
        )
        self.global_union_area_map: Dict[int, Tuple[Tuple[int, int], ...]] = {}

    def _infer_inner_deredundancy_rank_list_by_from_layout(self) -> List[int]:
        """
        Infers ranks containing non-redundant data from the source layout.

        Returns:
            List of ranks with unique data slices
        """
        inner_deredundancy_rank_list: List[int] = []
        from_dev_map = set()
        dev_dim = len(self.from_dev_matrix)

        # Collect relevant device dimensions from tensor map
        for map_dev in self.from_tensor_map:
            if isinstance(map_dev, (list, tuple)):
                for map_dev_inner in map_dev:
                    from_dev_map.add(dev_dim - map_dev_inner - 1)
            else:
                from_dev_map.add(dev_dim - map_dev - 1)

        # Filter ranks with non-redundant data
        for rank_id in self.inner_from_rank_list:
            dev_id_list = rank_id_to_dev_id_list(self.from_dev_matrix, rank_id)
            if any(dim not in from_dev_map and dev_id_list[dim] > 0 for dim in range(dev_dim)):
                continue
            inner_deredundancy_rank_list.append(rank_id)

        return inner_deredundancy_rank_list

    def infer_all_tensor_offset(self) -> Dict[int, Tuple[Tuple[int, int], ...]]:
        """
        Calculates required tensor slices from each source rank.

        Determines which parts of the tensor need to be collected from each source
        rank to assemble the target tensor slice.

        Returns:
            Dictionary mapping source ranks to their required slice offsets
        """
        # Calculate target area for current rank
        self.to_area = infer_slice_area_by_rank(
            self.to_dev_matrix,
            self.to_tensor_map,
            self.inner_to_rank_id,
            self.full_shape
        )

        # Calculate required slices from each source rank
        local_union_areas_map: Dict[int, Tuple[Tuple[int, int], ...]] = {}
        self.global_union_area_map.clear()

        for inner_rank_id in self.inner_deredundancy_from_rank_list:
            # Get source area for this rank
            from_area = infer_slice_area_by_rank(
                self.from_dev_matrix,
                self.from_tensor_map,
                inner_rank_id,
                self.full_shape
            )

            # Find overlapping area between source and target
            union_area = infer_intersection(from_area, self.to_area)
            if union_area is not None:
                source_rank = self.from_rank_list[inner_rank_id]
                self.global_union_area_map[source_rank] = union_area

                # Calculate relative offsets within source slice
                local_union_areas_map[source_rank] = tuple(
                    (union_range[0] - from_range[0], union_range[1] - from_range[0])
                    for union_range, from_range in zip(union_area, from_area)
                )

        return local_union_areas_map

    def get_real_tensor(self, from_tensor_map: Dict[int, Tensor]) -> Tensor:
        """
        Assembles the final tensor for the target rank from collected slices.

        Args:
            from_tensor_map: Dictionary mapping source ranks to their tensor slices

        Returns:
            Assembled tensor for the target rank

        Raises:
            ValueError: If input slices are missing or have incorrect shapes
        """
        if not from_tensor_map:
            raise ValueError("Input from_tensor_map cannot be empty")

        # Validate input slices
        for from_rank_id, from_area in self.global_union_area_map.items():
            if from_rank_id not in from_tensor_map:
                raise ValueError(
                    f"Missing slice data from rank {from_rank_id}. "
                    "Please provide all required slices from infer_all_tensor_offset."
                )

            # Validate slice shape matches expected size
            expected_shape = tuple(end - start for start, end in from_area)
            actual_shape = from_tensor_map[from_rank_id].shape
            if expected_shape != actual_shape:
                raise ValueError(
                    f"Slice from rank {from_rank_id} has incorrect shape. "
                    f"Expected {expected_shape}, got {actual_shape}."
                )

        # Create target tensor and assign slices
        to_slice_shape = [end - start for start, end in self.to_area]
        current_slice = next(iter(from_tensor_map.values()))
        if isinstance(current_slice, Tensor):
            real_tensor = Tensor(np.zeros(to_slice_shape), current_slice.dtype)
        else:
            real_tensor = np.zeros(to_slice_shape, current_slice.dtype)

        for from_rank_id, from_slice in from_tensor_map.items():
            from_area = self.global_union_area_map[from_rank_id]

            # Calculate assignment indices in target tensor
            assign_slices = tuple(
                slice(from_axis[0] - to_axis[0], from_axis[1] - to_axis[0])
                for from_axis, to_axis in zip(from_area, self.to_area)
            )

            real_tensor[assign_slices] = from_slice

        return real_tensor


def smart_slice(tensor, slice_ranges, load_from_multi_rank=False):
    """
    Slices a tensor based on specified slice ranges and determines if it's a full slice.

    Args:
        tensor: The tensor to slice (can be Parameter, Tensor, or have .shape attribute)
        slice_ranges: List of (start, end) tuples specifying slice ranges for each dimension
        load_from_multi_rank: If True, forces slicing even for full slices (for multi-rank loading)

    Returns:
        The original tensor if full slice and not load_from_multi_rank,  otherwise the sliced numpy array.

    Raises:
        ValueError: If slice dimension count doesn't match tensor dimension count
    """
    # Get tensor shape - handle both Parameter and Tensor types
    tensor_shape = tensor.shape

    if len(slice_ranges) != len(tensor_shape):
        raise ValueError(
            f"Slice dimension count ({len(slice_ranges)}) does not "
            f"match tensor dimension count ({len(tensor_shape)})"
        )

    # Check if this is a full slice
    is_full_slice = all(
        start == 0 and end == dim_size
        for (start, end), dim_size in zip(slice_ranges, tensor_shape)
    )

    # Perform the slice
    slice_indices = tuple(slice(start, end) for start, end in slice_ranges)
    if not load_from_multi_rank:
        if is_full_slice:
            return tensor
        return tensor[slice_indices]

    if isinstance(tensor, (Tensor, Parameter)):
        # MindSpore Tensor/Parameter
        sliced_tensor = copy.deepcopy(tensor.asnumpy()[slice_indices])
    else:
        # Numpy array or other array-like
        sliced_tensor = tensor[slice_indices]

    return sliced_tensor


def balance_load(params: List[dict], num_groups: int) -> List[List[dict]]:
    """
    Balances parameter load across worker groups to minimize load imbalance.

    Uses a greedy load balancing algorithm:
        1. Sorts parameters by total size (descending) to prioritize large parameters
        2. Greedily assigns each parameter to the worker group with the smallest current total size
    This ensures even distribution of computational load across workers.

    Args:
        params: List of parameter metadata dicts (each with "size" key)
        num_groups: Number of worker groups to split parameters into

    Returns:
        List of worker groups, where each group is a list of parameter metadata dicts.
        Groups are balanced by total tensor size to avoid uneven workload distribution.
    """
    # Sort parameters from largest to smallest to optimize load balancing
    sorted_params = sorted(params, key=lambda x: x["size"], reverse=True)

    # Initialize worker groups with empty params and zero total size
    groups = [{"total_size": 0, "params": []} for _ in range(num_groups)]

    # Assign each parameter to the least loaded group
    for param in sorted_params:
        min_group = min(groups, key=lambda g: g["total_size"])
        min_group["total_size"] += param["size"]
        min_group["params"].append(param)

    # Extract only the parameter lists (discard size tracking)
    return [group["params"] for group in groups]


class ReshardLoader:
    """
    An abstract Reshard loader.

    Provides plug-and-play distributed weight loading capabilities, supporting:
        - Lazy loading: Delayed reading using `ms_load_checkpoint`
        - On-demand slicing: Reading only the necessary slices
        - Multi-threaded concatenation: Parallel processing of reshard operations

    Design notes:
        - The `template` parameter is only used for HF weight loading scenarios.
        - When loading self-trained weights, `template` is `None`, directly using the source parameter name.
        - When loading HF weights, parameter name mapping is completed using `template.get_mf_name()`.
    """

    def __init__(
            self,
            checkpoint_dir: str,
            dst_sharded_tensor_metas: Dict[str, ShardedTensor],
            src_sharded_tensor_metas: Dict[str, List[ShardedTensor]],
            param_file_mappings: Dict[Tuple[str, Tuple], List[Dict]],
            reshard_worker_num: int = 1,
            template: Optional["WeightTemplate"] = None
    ):
        """
        Initialize the ReshardLoader.

        Args:
            checkpoint_dir: Path to the checkpoint directory.
            dst_sharded_tensor_metas: ShardedTensor dictionary for all parameters to be loaded in the current task.
                Format: {
                    param_name: ShardedTensor
                }
                Includes network and optimizer parameters in fine-tuning scenarios.
            src_sharded_tensor_metas: ShardedTensor dictionary for all parameters in the checkpoint to be loaded.
                Format: {
                    param_name: [ShardedTensor, ...]
                }
                - key is the original parameter name recorded in the checkpoint file.
                - value is a list of ShardedTensor (distributed weights may have multiple slices).
            param_file_mappings: Storage information for all slices in the checkpoint to be loaded.
                Format: {
                    (param_name, global_offset): [
                        {
                            'file_name': 'xxx.safetensors',
                            'storage_rank': rank_id,
                            'rank_group': [...]
                        },
                    ...
                    ]
                }
            reshard_worker_num: Number of threads for reshard processing (used in concatenation phase).
            template (Optional): Hugging Face weight conversion template instance.
        """
        self.checkpoint_dir = checkpoint_dir
        self.dst_metas = dst_sharded_tensor_metas
        self.src_metas = src_sharded_tensor_metas
        self.param_file_mappings = param_file_mappings
        self.reshard_worker_num = reshard_worker_num
        self.rank_id = get_real_rank()

        # Hugging Face Weight convert Template
        self.template = template

        # Pre-build a bidirectional mapping dictionary, to avoid repeated calls to `template.get_mf_name`.
        self.src_to_dst_mapping, self.dst_to_src_mapping = self._build_bidirectional_mapping()

        # Calculate the offset of all parameters
        start_time = time()
        self.params_info = self._compute_all_offsets()
        self.build_all_offsets_time = time() - start_time

    def _build_bidirectional_mapping(self) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
        """
        Pre-construct a bidirectional mapping dictionary.

        For Hugging Face weights, iterate through all `src_name` values at once,
            calling `template.get_mf_name()` to construct the mapping relationship,
            avoiding subsequent redundant calculations.

        Returns:
            Tuple[src_to_dst_mapping, dst_to_src_mapping]
            - src_to_dst_mapping: {src_name: dst_name},
                the mapping from source parameter names to target parameter names
            - dst_to_src_mapping: {dst_name: [src_name_1, src_name_2, ...]},
                the mapping from target parameter names to a list of source parameter names.

        Example (Concat QKV of Hugging Face weight scenario):
            src_to_dst_mapping = {
                "q_proj.weight": "linear_qkv.weight",
                "k_proj.weight": "linear_qkv.weight",
                "v_proj.weight": "linear_qkv.weight"
            }
            dst_to_src_mapping = {
                "linear_qkv.weight": ["q_proj.weight", "k_proj.weight", "v_proj.weight"]
            }
        """
        src_to_dst: Dict[str, str] = {}
        dst_to_src: Dict[str, List[str]] = {}

        if self.template is None:
            # MindSpore Transformers weights scenario: `src_name` and `dst_name` are the same.
            for src_name in self.src_metas.keys():
                if src_name in self.dst_metas:
                    dst_name = src_name
                    src_to_dst[src_name] = dst_name
                    dst_to_src[dst_name] = [src_name]
        else:
            # Hugging Face weighting scenario: using templates for mapping.
            for src_name in self.src_metas.keys():
                dst_name = self.template.get_mf_name(src_name)[0]
                if dst_name in self.dst_metas:
                    src_to_dst[src_name] = dst_name

                    if dst_name not in dst_to_src:
                        dst_to_src[dst_name] = []
                    dst_to_src[dst_name].append(src_name)

        return src_to_dst, dst_to_src

    def get_dst_name(self, src_name: str) -> str:
        """
        Retrieves the target parameter name corresponding to the source parameter name.

        Preferably uses pre-built mappings to avoid duplicate calls to `template.get_mf_name()`.
        """
        return self.src_to_dst_mapping.get(src_name, None)

    def get_src_names(self, dst_name: str) -> List[str]:
        """
        Retrieves all source parameter names corresponding to the target parameter name.

        Used in Hugging Face checkpoint scenarios, such as retrieving [q, k, v] corresponding to qkv.
        """
        return self.dst_to_src_mapping.get(dst_name, None)

    def _compute_all_offsets(self) -> Dict[str, Dict]:
        """
        Calculate the offset information for all weight parameters associated
            with the parameters to be loaded on the current card.

        Key design:
            - First traverse dst_metas (parameters to be loaded on the current card)
                to determine which target parameters are needed.
            - For each target parameter, find all related source parameters through dst_to_src_mapping.
            - Calculate offset information for each source parameter,
                using the source parameter's shape and target parameter's layout.
            - Return in {src_name: {...}} format, use `get_dst_name()` for lookup when needed.

        Process:
            1. Traverse dst_metas to get target parameters to be loaded on the current card.
            2. For each target parameter, get the list of all related source parameter names.
            3. For each source parameter, create a ReshardHandler and calculate all_offset.

        Returns:
            params_info: {
                src_name: {
                    "all_offset": {rank: slice_range, ...},
                    "reshard_handler": ReshardHandler
                }
            }

            Where src_name is the original parameter name saved in the weight file,
                and the corresponding target parameter name can be obtained via `self.get_dst_name(src_name)`.
        """
        params_info = {}

        # 1. Iterate through the target parameters that need to be loaded in the current card.
        for dst_name, dst_tensor in self.dst_metas.items():
            # 2. Get all source parameter names corresponding to the target parameter.
            src_names = self.get_src_names(dst_name)

            if not src_names:
                logger.warning(f"No source parameters found for dst_param: {dst_name}, skipping")
                continue

            # 3. For each source parameter, calculate the offset information.
            for src_name in src_names:
                # Skip processed source parameters (to avoid duplication).
                if src_name in params_info:
                    continue

                # Check if the source parameters are in `src_metas`.
                if src_name not in self.src_metas:
                    logger.warning(f"Source parameter {src_name} not in src_metas, skipping")
                    continue

                src_tensor_list = self.src_metas[src_name]
                src_tensor = (
                    src_tensor_list[0]
                    if isinstance(src_tensor_list, list)
                    else src_tensor_list
                )

                # Create ReshardHandler
                # Note: Use the source parameter's `global_shape` and the target parameter's `layout`.
                reshard_handler = ReshardHandler(
                    param_name=src_name,
                    full_shape=src_tensor.global_shape,
                    from_layout=src_tensor.layout,
                    to_layout=dst_tensor.layout,
                    to_rank_id=self.rank_id
                )

                all_offset = reshard_handler.infer_all_tensor_offset()
                params_info[src_name] = {
                    "all_offset": all_offset,
                    "reshard_handler": reshard_handler
                }

        return params_info

    def _organize_file_load_info(self) -> Dict[str, List[Tuple[str, int, Tuple]]]:
        """
        Organize loading information by file.

        Based on param_file_mappings, determine which file stores each parameter slice,
            and group the parameters to be loaded by file.

        Core logic:
            1. Iterate through params_info, get the list of search_ranks for each source parameter (all_offset.keys());
            2. Get all storage slice information for the src_name in param_file_mappings;
            3. For each search_rank, iterate through all storage information to find matching slices;
            4. Matching conditions: storage_rank == search_rank or search_rank in rank_group;
            5. After finding a match, record the file name and slice information.

        Explanation of redundancy removal save scenario:
            - When weights use redundancy removal saving (remove_redundancy=True),
                the same slice might be saved by only one rank in the rank_group (storage_rank).
            - At this time, search_rank and storage_rank may be inconsistent.
            - Need to judge through rank_group: as long as search_rank exists in rank_group,
                the slice can be loaded from that file.

        Returns:
            {file_name: [(src_name, search_rank, param_slice), ...]}

        Data structure explanation:
        - param_file_mappings format:
            {
                (param_name, global_offset): [
                    {
                        "file_name": "model-0000001-0000008.safetensors",
                        "storage_rank": 0,
                        "rank_group": [0, 1, 2, 3]  # redundancy removal
                    },
                    ...
                ]
            }
        """
        files_to_load: Dict[str, List[Tuple[str, int, Tuple]]] = {}

        for src_name, src_info in self.params_info.items():
            all_offset = src_info["all_offset"]

            # 1. Retrieves storage information for all slices specified by the src_name parameter.
            param_storage_infos = self._get_param_storage_infos(src_name)

            # 2. For each search_rank that needs to be loaded, find the matching storage file.
            for search_rank, param_slice in all_offset.items():
                # Find the corresponding storage file name based on search_rank.
                file_name = self._find_file_for_rank(
                    src_name, search_rank, param_storage_infos
                )

                if file_name not in files_to_load:
                    files_to_load[file_name] = []
                files_to_load[file_name].append(
                    (src_name, search_rank, param_slice)
                )

        return files_to_load

    def _get_param_storage_infos(
            self,
            param_name: str
    ) -> Dict[int, Dict]:
        """
        Gets the storage information for all slices of a specified parameter.

        Args:
            param_name: Source parameter name

        Returns:
            A dict, format is: {
                storage_rank_id: {
                    "file_name": "xxx.safetensors",
                    "rank_group": [...]
                },
                ...
            }

        Implementation logic:
            1. Get the ShardedTensor list for the parameter from src_metas;
            2. Iterate through each ShardedTensor to get its global_offset;
            3. Call get_sharded_tensor_shard_id() to generate the mapping key;
            4. Look up the corresponding storage information from param_file_mappings using the key;
            5. Reorganize the result with storage_rank as the key.

        Advantages of this implementation:
            - Directly utilizes the existing slice information in src_metas,
                avoiding traversing the entire param_file_mappings.
            - Uses get_sharded_tensor_shard_id to ensure the key format is consistent with when saving.
            - Using storage_rank as the key facilitates quick lookup based on search_rank in subsequent steps.
        """
        result: Dict[int, Dict] = {}

        # 1. Retrieve the list of ShardedTensors for this parameter from `src_metas`.
        if param_name not in self.src_metas:
            raise ValueError(f"Parameter '{param_name}' not found in src_metas")

        src_tensor_list = self.src_metas[param_name]
        if not isinstance(src_tensor_list, list):
            src_tensor_list = [src_tensor_list]

        # 2. Iterate through each ShardedTensor to obtain storage information.
        for sharded_tensor in src_tensor_list:
            # 3. Generate `mapping_key`.
            mapping_key = get_sharded_tensor_shard_id(param_name, sharded_tensor.global_offset)

            # 4. Find storage information from `param_file_mappings`.
            if mapping_key not in self.param_file_mappings:
                raise ValueError(
                    f"Storage info not found for param '{param_name}' "
                    f"with key={mapping_key}. The source checkpoint may be incomplete."
                )

            # 5. Organize the results using `storage_rank` as the key.
            for storage_info in self.param_file_mappings[mapping_key]:
                storage_rank = storage_info.get("storage_rank")
                result[storage_rank] = {
                    "file_name": storage_info["file_name"],
                    "rank_group": storage_info.get("rank_group", [])
                }

        return result

    def _find_file_for_rank(
            self,
            param_name: str,
            search_rank: int,
            param_storage_infos: Dict[int, Dict]
    ) -> str:
        """
        Find the corresponding storage file name based on search_rank.

        Args:
            param_name: Parameter name (for error message).
            search_rank: Rank ID to search.
            param_storage_infos: Storage information for the parameter.
                Format: {
                    storage_rank: {
                        "file_name": ...,
                        "rank_group": [...]
                    },
                    ...
                }

        Returns:
            A string. File name storing this slice.

        Raises:
            ValueError: If the corresponding storage file cannot be found,
                indicating the source checkpoint is incomplete.

        Matching rules:
            1. Priority direct match: `search_rank == storage_rank`;
            2. Redundancy removal scenario: `search_rank` in `rank_group`.
        """
        # Rule 1: Direct Match.
        if search_rank in param_storage_infos:
            return param_storage_infos[search_rank]["file_name"]

        # Rule 2: Check rank_group (for redundancy removal and saving scenarios).
        for _, info in param_storage_infos.items():
            rank_group = info.get("rank_group", [])
            if search_rank in rank_group:
                return info["file_name"]

        # Throw an error when not found the file.
        raise ValueError(
            f"Cannot find storage file for parameter '{param_name}' "
            f"at search_rank={search_rank}. "
            f"Available storage_ranks: {list(param_storage_infos.keys())}. "
            f"The source checkpoint may be incomplete or corrupted."
        )

    def _load_and_slice(
            self,
            files_to_load: Dict
    ) -> Tuple[Dict, Dict]:
        """
        Lazy loading and slicing.

        Use ms.load_checkpoint to lazily load the weight files, then perform slicing.

        Key design:
            - For MindSpore Transformers weights: Get the target parameter directly after slicing.
            - For Hugging Face weights: Slice q, k, v separately,
                which will be concatenated later by `Template.convert()`.

        Returns:
            A tuple, contains (params_info_need_reshard, src_sliced_tensors).
            - params_info_need_reshard: Information of parameters that need resharding.
            - src_sliced_tensors: Source parameters after slicing, format: {src_name: sliced_tensor}.
        """
        src_sliced_tensors: Dict[str, Any] = {}  # Store source parameters after slicing
        params_info_need_reshard: Dict[str, Dict] = {}

        for file_name, param_infos in files_to_load.items():
            file_path = os.path.join(self.checkpoint_dir, file_name)

            # Collect the source parameter names that need to be loaded,
            # using the source parameter names when reading from the weight file.
            src_names = list(set(info[0] for info in param_infos))  # `src_name` is the first character in the tuple.

            # Lazy loading
            state_dict_from_file = ms_load_checkpoint(
                file_path,
                format='safetensors',
                choice_func=lambda x, src_names_local=src_names: x in src_names_local
            )

            # Slicing
            for src_name, search_rank, param_slice in param_infos:
                if src_name not in state_dict_from_file:
                    continue

                parameter = state_dict_from_file[src_name]
                src_info = self.params_info[src_name]
                reshard_handler = src_info["reshard_handler"]
                all_offset = src_info["all_offset"]
                need_reshard = len(all_offset) > 1

                sliced_tensor = smart_slice(
                    parameter, param_slice, need_reshard
                )

                if not need_reshard:
                    # Save slice results directly without resharding.
                    src_sliced_tensors[src_name] = sliced_tensor
                else:
                    # Resharding is required, recorded in `params_info_need_reshard`.
                    if src_name not in params_info_need_reshard:
                        params_info_need_reshard[src_name] = {
                            "reshard_handler": reshard_handler,
                            "tensor_map": {}
                        }
                    params_info_need_reshard[src_name]["tensor_map"][search_rank] = sliced_tensor

        return params_info_need_reshard, src_sliced_tensors

    def _parallel_reshard(
            self,
            params_info_need_reshard: Dict,
            src_sliced_tensors: Dict
    ) -> Dict[str, Any]:
        """
        Parallel concatenation of parameters requiring resharding.

        Concatenate the slices from each rank into the complete source parameters.

        Returns:
            Updated src_sliced_tensors: {src_name: complete parameters after resharding}
        """
        if not params_info_need_reshard:
            return src_sliced_tensors

        # Prepare worker tasks.
        tasks = []
        for src_name, info in params_info_need_reshard.items():
            tensor_map = info["tensor_map"]
            size = sum(np.prod(t.shape) for t in tensor_map.values())
            tasks.append({
                "src_name": src_name,
                "reshard_handler": info["reshard_handler"],
                "tensor_map": tensor_map,
                "size": size
            })

        # Load balancing distribution.
        worker_groups = balance_load(tasks, self.reshard_worker_num)

        def process_group(group):
            """Process a group of parameters."""
            results = {}
            for task in group:
                real_tensor = task["reshard_handler"].get_real_tensor(task["tensor_map"])
                real_tensor = Parameter(real_tensor, name=task["src_name"], requires_grad=False)
                results[task["src_name"]] = real_tensor
            return results

        # Multi-threaded execution.
        if self.reshard_worker_num > 1:
            with ThreadPoolExecutor(max_workers=self.reshard_worker_num) as executor:
                futures = [executor.submit(process_group, group) for group in worker_groups]
                for future in futures:
                    src_sliced_tensors.update(future.result())
        else:
            for group in worker_groups:
                src_sliced_tensors.update(process_group(group))

        return src_sliced_tensors

    def load(self) -> Dict[str, Parameter]:
        """
        Execute Reshard loading, return the parameter dictionary after Reshard.

        Returns:
            A parameter dictionary in the format of {param_name: Parameter}

        Return value description:
            - MindSpore Transformers weight scenario:
                `param_name` is consistent with the key in `dst_sharded_tensor_metas`.
            - Hugging Face weight scenario:
                `param_name` is the original Hugging Face parameter name (subsequently converted by Template.convert()).

        Note:
            For Hugging Face weights, the returned result is after Reshard
                but before Convert (such as QKV concatenation).
            Subsequent call to `Template.convert()` is required to complete the final conversion
                to get the dictionary with MF parameter names.
        """
        logger.info("ReshardLoader: Starting load...")

        # 1. Organization the file loading information.
        start_time = time()
        files_to_load = self._organize_file_load_info()

        # 2. Lazy loading and slicing.
        params_info_need_reshard, src_sliced_tensors = self._load_and_slice(files_to_load)
        self.build_all_tensor_map_time = time() - start_time

        # 3. Parallel splicing requires the reshard parameter.
        start_time = time()
        src_sliced_tensors = self._parallel_reshard(params_info_need_reshard, src_sliced_tensors)
        self.apply_parallel_load_strategy_time = time() - start_time

        # 4. Construct the return result
        if self.template is None:
            # For MindSpore Transformers weights: organize directly by `mf_param_name`;
            state_dict = {}
            start_time = time()
            for src_name in list(src_sliced_tensors.keys()):
                tensor = src_sliced_tensors.pop(src_name)
                if not isinstance(tensor, Parameter):
                    logger.info(f"{src_name}, type: {type(tensor)}")
                    tensor = Parameter(tensor, name=src_name, requires_grad=False)
                state_dict[src_name] = tensor
            del src_sliced_tensors
            self.convert_parameter_time = time() - start_time
        else:
            # For Hugging Face weights: return the `src_name`, and converted by `Template.convert()`.
            state_dict = src_sliced_tensors

        logger.info(f"ReshardLoader: Loaded {len(state_dict)} parameters.")
        logger.info(f"Build all_offsets cost: {round(self.build_all_offsets_time, 6)}s.")
        logger.info(f"Build all_tensor_map cost: {round(self.build_all_tensor_map_time, 6)}s.")
        logger.info(f"Apply parallel_load_strategy cost: {round(self.apply_parallel_load_strategy_time, 6)}s.")
        if self.template is None:
            logger.info(f"Convert to parameter cost: {round(self.convert_parameter_time, 6)}s.")

        return state_dict
