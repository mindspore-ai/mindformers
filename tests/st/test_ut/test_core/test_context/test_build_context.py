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
"""Test build_context.py"""

import multiprocessing
import os
import tempfile
from unittest.mock import patch

import pytest

from mindformers.core.context import (
    build_context,
    build_mf_context,
    build_parallel_context,
    get_context,
    set_context,
)
from mindformers.core.context.build_context import (
    Context,
    MFContextOperator,
    MSContextOperator,
    set_cpu_affinity,
    set_ms_affinity,
)
from mindformers.tools.register import MindFormerConfig


@pytest.fixture(autouse=True)
def isolated_context():
    """Reset both context singletons and isolate one-time CPU affinity setup."""
    Context.reset_instance()
    MFContextOperator.reset_instance()
    with patch("mindformers.core.context.build_context._setup_affinity"):
        yield
    Context.reset_instance()
    MFContextOperator.reset_instance()


def get_config_tpl():
    return {"context": {"mode": "PYNATIVE_MODE", "device_target": "CPU"}, "parallel": {}}


def run_in_subprocess(func, *args):
    """Run testcase in subprocess and check it is successfully."""
    process = multiprocessing.Process(target=func, args=args)
    process.start()
    process.join()
    assert process.exitcode == 0


def run_deterministic_setting(mode, switch, shuffle, lccl, expected_shuffle, expected_lccl):
    """Check the MindSpore API call and environment variables owned by MindFormers."""
    os.environ.clear()
    os.environ.update({key: value for key, value in {
        "CUSTOM_MATMUL_SHUFFLE": shuffle,
        "LCCL_DETERMINISTIC": lccl,
    }.items() if value is not None})
    config = get_config_tpl()
    config["run_mode"] = mode
    with patch("mindformers.core.context.build_context.set_ms_deterministic") as set_deterministic:
        build_context(config)
        if mode in ("train", "finetune"):
            set_context(train_precision_sync=switch)
            assert get_context("train_precision_sync") == switch
        else:
            set_context(infer_precision_sync=switch)
            assert get_context("infer_precision_sync") == switch
        set_deterministic.assert_called_once_with(switch)
    assert os.getenv("CUSTOM_MATMUL_SHUFFLE") == expected_shuffle
    assert os.getenv("LCCL_DETERMINISTIC") == expected_lccl


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("mode,switch,shuffle,lccl,expected_shuffle,expected_lccl", [
    ("train", False, None, None, "on", "0"),
    ("train", True, None, None, "on", "0"),
    ("finetune", True, None, None, "on", "0"),
    ("predict", False, "off", "1", "on", "0"),
    ("predict", True, "on", "0", "off", "1"),
])
def test_deterministic(mode, switch, shuffle, lccl, expected_shuffle, expected_lccl):
    """Training and inference precision switches call MindSpore with the requested value."""
    run_in_subprocess(run_deterministic_setting, mode, switch, shuffle, lccl, expected_shuffle, expected_lccl)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_mf_context_singleton():
    """
    Feature: Test whether MFContextOperator is a singleton.
    Description: The MFContextOperator instance created twice is the same object.
    Expectation: The MFContextOperator instance created twice are different object.
    """
    config_tpl = get_config_tpl()
    mf_ctx = build_mf_context(config_tpl)
    another_mf_ctx = build_mf_context(config_tpl)
    assert mf_ctx is another_mf_ctx


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_context_singleton():
    """
    Feature: Test Context singleton pattern.
    Description: Test that Context is a singleton.
    Expectation: Multiple Context instances are the same object.
    """

    def is_singleton_context():
        config_tpl = get_config_tpl()
        ctx1 = build_context(config_tpl)
        ctx2 = build_context(config_tpl)
        assert ctx1 is ctx2

    run_in_subprocess(is_singleton_context)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_context_set_mf_ctx_run_mode():
    """
    Feature: Test Context.set_mf_ctx_run_mode method.
    Description: Test setting run_mode with valid and invalid values.
    Expectation: Valid run_mode is set, invalid run_mode raises ValueError.
    """
    Context.reset_instance()
    config_tpl = get_config_tpl()
    ctx = build_context(config_tpl)

    # Test valid run_mode
    ctx.set_mf_ctx_run_mode("train")
    assert ctx.mf_ctx_opr.run_mode == "train"

    # Test invalid run_mode
    with pytest.raises(ValueError) as exc_info:
        ctx.set_mf_ctx_run_mode("invalid_mode")
    assert "Invalid value" in str(exc_info.value)

    # Test None run_mode
    ctx.set_mf_ctx_run_mode(None)
    Context.reset_instance()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_ms_context_operator_set_save_graphs_path():
    """
    Feature: Test MSContextOperator save_graphs_path setting.
    Description: Test save_graphs_path setting via environment variable.
    Expectation: save_graphs_path is set when save_graphs is True.
    """
    config = MindFormerConfig(
        context={"save_graphs": True, "save_graphs_path": "/tmp/graphs"},
        parallel={},
    )
    operator = MSContextOperator(config)
    assert operator.get_context("save_graphs_path") == "/tmp/graphs"


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_ms_context_applier_apply_device():
    """
    Feature: Test _MSContextApplier._apply_device method.
    Description: Test that device_id and device_target are applied via mindspore.set_device().
    Expectation: mindspore.set_device is called and attributes are cached.
    """
    def run_apply_device_test():
        config = MindFormerConfig(
            context={"device_target": "CPU", "device_id": 0},
            use_parallel=False,
        )
        operator = MSContextOperator(config)

        # Verify attributes are cached
        assert operator.get_context("device_target") == "CPU"
        assert operator.get_context("device_id") == 0

    run_in_subprocess(run_apply_device_test)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_ms_context_applier_apply_memory():
    """
    Feature: Test _MSContextApplier._apply_memory method.
    Description: Test that max_device_memory is applied via mindspore.runtime.set_memory().
    Expectation: mindspore.runtime.set_memory is called and attributes are cached.
    """
    config = MindFormerConfig(
        context={"max_device_memory": "2GB"}, parallel={}
    )
    operator = MSContextOperator(config)

    # Verify attributes are cached
    assert operator.get_context("max_device_memory") == "2GB"


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_ms_context_applier_apply_env():
    """
    Feature: Test _MSContextApplier._apply_env method.
    Description: Test that save_graphs and save_graphs_path are applied via environment variables.
    Expectation: Environment variables are set and attributes are cached.
    """
    # Clear environment first
    os.environ.pop("MS_DEV_SAVE_GRAPHS", None)
    os.environ.pop("MS_DEV_SAVE_GRAPHS_PATH", None)

    config = MindFormerConfig(
        context={"save_graphs": True, "save_graphs_path": "/tmp/graphs"},
        parallel={},
    )
    operator = MSContextOperator(config)

    # Verify environment variables are set
    assert os.getenv("MS_DEV_SAVE_GRAPHS") == "1"
    assert os.getenv("MS_DEV_SAVE_GRAPHS_PATH") == "/tmp/graphs"

    # Verify attributes are cached
    assert operator.get_context("save_graphs") is True
    assert operator.get_context("save_graphs_path") == "/tmp/graphs"


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_ms_context_applier_apply_extra():
    """
    Feature: Test _MSContextApplier._apply_extra method.
    Description: Test that max_call_depth is applied via mindspore.set_recursion_limit().
    Expectation: mindspore.set_recursion_limit is called and attribute is cached.
    """
    config = MindFormerConfig(context={"max_call_depth": 200}, parallel={})
    operator = MSContextOperator(config)

    # Verify attribute is cached
    assert operator.get_context("max_call_depth") == 200


def _make_parallel_speed_up_temp_file():
    """
    Create a temp file with valid parallel_speed_up JSON content.
    Returns (path, parallel_json_path). Caller must os.unlink(path) when done.
    """
    fd, path = tempfile.mkstemp(suffix=".json", prefix="parallel_speed_up_")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write('{"dataset_broadcast_opt_level": 3}')
    return path, os.path.abspath(path)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_ms_context_operator_ascend_config_precision_mode():
    """
    Feature: Test MSContextOperator with ascend_config precision_mode.
    Description: Test that precision_mode is applied via new API and removed from ascend_config.
    Expectation: precision_mode is applied via new API; ascend_config is cached without precision_mode.
    """
    path, parallel_json_path = _make_parallel_speed_up_temp_file()
    try:
        config = MindFormerConfig(
            context={
                "ascend_config": {
                    "precision_mode": "must_keep_origin_dtype",
                    "parallel_speed_up_json_path": parallel_json_path,
                }
            },
            parallel={},
        )
        operator = MSContextOperator(config)

        cached_config = operator.get_context("ascend_config")
        assert cached_config is not None
        assert cached_config == config.get('context').get('ascend_config')
    finally:
        if os.path.exists(path):
            os.unlink(path)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_ms_context_operator_ascend_config_only_parallel_speed_up():
    """
    Feature: Test MSContextOperator with ascend_config containing only parallel_speed_up_json_path.
    Description: Test that parallel_speed_up_json_path is passed to set_context when precision_mode is absent.
    Expectation: ascend_config is cached correctly; set_context is called with a real parallel_speed_up JSON path.
    """
    path, parallel_json_path = _make_parallel_speed_up_temp_file()
    try:
        config = MindFormerConfig(
            context={
                "ascend_config": {
                    "parallel_speed_up_json_path": parallel_json_path
                }
            },
            parallel={},
        )
        operator = MSContextOperator(config)

        cached_config = operator.get_context("ascend_config")
        assert cached_config is not None
        assert (
            cached_config.get("parallel_speed_up_json_path") == parallel_json_path
        )
    finally:
        if os.path.exists(path):
            os.unlink(path)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
def test_build_parallel_context():
    """
    Feature: Test build_parallel_context function.
    Description: Test building parallel context.
    Expectation: ParallelOperator is returned.
    """
    config_tpl = get_config_tpl()
    parallel_opr = build_parallel_context(config_tpl)
    assert parallel_opr is not None
    assert hasattr(parallel_opr, "parallel_ctx")
    assert hasattr(parallel_opr, "parallel")


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@patch('mindformers.tools.utils.get_real_group_size')
@patch('mindformers.tools.utils.get_real_local_rank')
@patch('mindspore.runtime.set_cpu_affinity')
def test_set_ms_affinity_with_affinity_config(mock_set_affinity, mock_rank, mock_group_size):
    """
    Feature: Test set_ms_affinity with affinity_config.
    Description: Verify affinity_config overrides affinity_cpu_list and passes module config.
    Expectation: MindSpore set_cpu_affinity called with config values.
    """
    mock_group_size.return_value = 8
    mock_rank.return_value = 1
    affinity_config = {
        'device_1': {
            'affinity_cpu_list': [0, 1],
            'module_to_cpu_dict': {'module_a': [2, 3]}
        }
    }
    set_ms_affinity(affinity_config, [4, 5])
    mock_set_affinity.assert_called_once_with(
        True,
        [0, 1],
        {'module_a': [2, 3]}
    )


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@patch('os.access')
@patch('os.path.exists')
@patch('mindspore.runtime.set_cpu_affinity')
def test_set_ms_affinity_with_json_file(mock_set_affinity, mock_exists, mock_access):
    """
    Feature: Test set_ms_affinity with JSON file path.
    Description: Verify that set_cpu_affinity is called correctly when affinity_config is a .json file path.
    Expectation: MindSpore set_cpu_affinity is called with (True, None, None, json_path).
    """
    mock_exists.return_value = True
    mock_access.return_value = True

    json_path = "/tmp/test_affinity_config.json"
    
    set_ms_affinity(json_path, None)

    mock_exists.assert_called_once_with(json_path)
    mock_access.assert_called_once_with(json_path, os.R_OK)
    mock_set_affinity.assert_called_once_with(True, None, None, json_path)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@patch('mindformers.tools.utils.get_real_group_size')
@patch('mindformers.tools.utils.get_real_local_rank')
@patch('mindspore.runtime.set_cpu_affinity')
def test_set_ms_affinity_without_device_entry(mock_set_affinity, mock_rank, mock_group_size):
    """
    Feature: Test set_ms_affinity when device entry missing.
    Description: Verify defaults are used when affinity_config lacks device info.
    Expectation: MindSpore set_cpu_affinity called with None values.
    """
    mock_group_size.return_value = 8
    mock_rank.return_value = 0
    affinity_config = {
        'device_1': {
            'affinity_cpu_list': [4, 5],
            'module_to_cpu_dict': {'module_a': [6, 7]}
        }
    }
    set_ms_affinity(affinity_config, None)
    mock_set_affinity.assert_called_once_with(
        True,
        None,
        None
    )


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@patch('mindformers.tools.utils.get_real_group_size')
@patch('mindformers.tools.utils.get_real_local_rank')
def test_set_ms_affinity_with_invalid_device_id(mock_rank, mock_group_size):
    """
    Feature: Test set_ms_affinity when device entry missing.
    Description: Verify defaults are used when affinity_config lacks device info.
    Expectation: MindSpore set_cpu_affinity called with None values.
    """
    mock_group_size.return_value = 1
    mock_rank.return_value = 0
    affinity_config = {
        'device_1': {
            'affinity_cpu_list': [4, 5],
            'module_to_cpu_dict': {'module_a': [6, 7]}
        }
    }
    with pytest.raises(ValueError) as exc_info:
        set_ms_affinity(affinity_config, None)
    assert 'Invalid device id' in str(exc_info.value)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize(
    "cann_workqueue_cores, cpu_affinity",
    (
        ([0, 1], [2, 3]),
        ([0, 1, 2, 3], [0, 1, 2, 3]),
    ),
)
@patch("mindformers.utils.cann_cores_utils.get_cann_workqueue_cores")
@patch("psutil.Process")
@patch("psutil.cpu_count", return_value=8)
@patch("mindspore.dataset.config.set_numa_enable")
def test_set_cpu_affinity(
    mock_set_numa,
    mock_cpu_count,
    mock_process_cls,
    mock_get_cores,
    monkeypatch,
    cann_workqueue_cores,
    cpu_affinity,
):
    """
    Feature: Test set_cpu_affinity fallback behavior.
    Description: Verify that the original CPU list is used when CANN occupies all candidate cores,
    and CPU affinity excludes CANN workqueue cores when available.
    Expectation: Process cpu_affinity receives unfiltered CPU list.
    """
    mock_get_cores.return_value = cann_workqueue_cores
    monkeypatch.setenv("CPU_AFFINITY", "True")
    process_mock = mock_process_cls.return_value

    set_cpu_affinity(rank_id=0, rank_size=2)

    mock_set_numa.assert_called_once_with(True)
    mock_cpu_count.assert_called_once()
    mock_get_cores.assert_called_once_with(0)
    process_mock.cpu_affinity.assert_called_once_with(cpu_affinity)
