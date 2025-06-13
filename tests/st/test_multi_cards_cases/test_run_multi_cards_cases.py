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
"""multi cards testcases scheduler."""

import glob
import importlib.util
import os
import random
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import pytest

from tests.st.test_multi_cards_cases.utils import TaskType, TaskInfo


NODE_WORK_NUM = 8
WORK_DIR = os.path.dirname(os.path.abspath(__file__))


class CardsGroup:
    """
    CardsGroup represents a collection of card groups, where each group is a list of integers.

    Attributes:
        group_list (List[List[int]]): A list containing groups of cards, each represented as a list of integers.
        ocupated (List[bool]): A list of boolean values indicating whether each group is occupied.

    Args:
        group_list (List[List[int]], optional): The initial list of card groups. Defaults to an empty list.

    Example:
        cards = CardsGroup([[1, 2], [3, 4]])
    """
    def __init__(self, group_list: List[List[int]]):
        self.group_list = group_list
        self.ocupated = [False] * len(self.group_list)


class CardStateManager:
    """
    Manages the allocation and release of card groups for different task types.
    Uses global locks to avoid pickling issues with multiprocessing.
    """
    # Move locks to be global to avoid pickling issues
    _global_locks = {
        TaskType.TWO_CARDS_TASK: threading.Lock(),
        TaskType.FOUR_CARDS_TASK: threading.Lock(),
        TaskType.EIGHT_CARDS_TASK: threading.Lock()
    }

    def __init__(self):
        self.two_cards_group = CardsGroup([
            [0, 1], [2, 3], [4, 5], [6, 7]
        ])
        self.four_cards_group = CardsGroup([
            [0, 1, 2, 3], [4, 5, 6, 7]
        ])
        self.eight_cards_group = CardsGroup([
            [0, 1, 2, 3, 4, 5, 6, 7]
        ])
        self.task_group_mapping = {
            TaskType.TWO_CARDS_TASK: self.two_cards_group,
            TaskType.FOUR_CARDS_TASK: self.four_cards_group,
            TaskType.EIGHT_CARDS_TASK: self.eight_cards_group
        }
        # Randomly generate four port numbers between 50000 and 65535, and save them as a list
        self.task_port_ids: List[int] = random.sample(range(50000, 65536), 4)

    def allocate(self, task_type: TaskType):
        """
        Allocate a card group for the given task type.

        Args:
            task_type (TaskType): The type of task to allocate cards for.

        Returns:
            list: The allocated card group.

        Raises:
            KeyError: If the task type is not supported.
            ValueError: If no card group is available.
        """
        if task_type not in self.task_group_mapping:
            raise KeyError(
                f"{task_type} is not support, only support task type in "
                f"{list(self.task_group_mapping.keys())}"
            )
        # Use global locks
        lock = CardStateManager._global_locks[task_type]
        group = self.task_group_mapping[task_type]
        with lock:
            for idx, _ in enumerate(group.group_list):
                if not group.ocupated[idx]:
                    group.ocupated[idx] = True
                    return group.group_list[idx], self.task_port_ids[idx]
        raise ValueError("No card is ready!!!")

    def free(self, task_type, card_list):
        """
        Release the card group for the given task type.

        Args:
            task_type (TaskType): The type of task to release cards for.
            card_list (list): The card group to release.

        Raises:
            KeyError: If the task type is not supported.
            ValueError: If the card list is invalid.
        """
        if task_type not in self.task_group_mapping:
            raise KeyError(
                f"{task_type} is not support, only support task type in "
                f"{list(self.task_group_mapping.keys())}"
            )
        with CardStateManager._global_locks[task_type]:
            group_list = self.task_group_mapping[task_type].group_list
            for group_idx, group in enumerate(group_list):
                if group == card_list:
                    self.task_group_mapping[task_type].ocupated[group_idx] = False
                    return
        raise ValueError(
            f"card list {card_list} is error."
        )


class TaskScheduler(ABC):
    """
    Abstract base class for scheduling and executing groups of tasks.
    This class manages tasks grouped by their type and provides a framework for scheduling and running these groups.
    Subclasses must implement the `schedule_group` method to define how each group of tasks is scheduled.
    Attributes:
        grouped_tasks (dict): A dictionary mapping group types to lists of TaskInfo objects.
    Methods:
        add_tasks(task: TaskInfo):
            Adds a task to the appropriate group in `grouped_tasks`. If the group does not exist, it is created.
        worker(task: TaskInfo):
            Static method that simulates the execution of a task by sleeping for
            a duration based on the task's `task_time`. Returns the duration.
        schedule_group(tasks: List[TaskInfo]):
            Abstract method to schedule a group of tasks. Must be implemented by subclasses.
        run():
            Executes all groups of tasks in sequence, timing each group and the total execution.
            Returns the total completion time.
    """
    def __init__(self):
        self.grouped_tasks = {}
        self.card_manager = CardStateManager()
        self.stop_event = threading.Event()

    def add_task(self, task_info: TaskInfo):
        """
        Adds a task to the list of tasks associated with the specified group type.

        If the group type already has tasks, the new task is appended to the list.
        If not, a new list is created for the group type with the given task as its first element.

        Args:
            task_info: The task object to be added.
            group_type (GroupType): The type of group to which the task should be added.

        """
        if task_info.task_type in self.grouped_tasks:
            self.grouped_tasks[task_info.task_type].append(task_info)
        else:
            self.grouped_tasks[task_info.task_type] = [task_info]

    def worker(self, task_info: TaskInfo, card_manager: CardStateManager):
        """
        Executes a task by allocating cards, simulating execution time, and freeing the cards.
        Args:
            task_info (TaskInfo): Information about the task, including type and execution time.
            card_manager (CardStateManager): Manager responsible for allocating and freeing cards.
        Returns:
            int: The duration of the task execution.
        Notes:
            - Allocates cards based on the task type.
            - Runs pytest and prints the result to the log.
            - Frees the allocated cards after execution.
        """
        if self.stop_event.is_set():
            print(f"🛑 Task canceled before start: {task_info.task_command}")
            return 0
        try:
            card_list, port_id = card_manager.allocate(task_info.task_type)
            visible_devices = ",".join(str(card) for card in card_list)
            env = os.environ.copy()
            env["ASCEND_RT_VISIBLE_DEVICES"] = visible_devices
            env["ASCEND_PORT_ID"] = str(port_id)

            print(f"🏃 Running: {task_info.task_command} on cards {card_list} (port {port_id})")
            start_time = time.time()

            process = subprocess.Popen(
                task_info.task_command,
                shell=True,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8"
            )
            while process.poll() is None:
                if self.stop_event.is_set():
                    process.terminate()  # Send terminate signal
                    print(f"🛑 Terminated: {task_info.task_command}")
                    return 0
                time.sleep(0.1)  # Avoid busy waiting on CPU
            stdout, _ = process.communicate()
            elapsed = time.time() - start_time
            if process.returncode == 0:
                print(f"✅ PASSED: {task_info.task_command} | Time: {elapsed:.3f}s")
            else:
                print(f"❌ FAILED: {task_info.task_command} (exit code {process.returncode}) | Time: {elapsed:.3f}s")
                print(f"Output:\n{stdout}")
                raise RuntimeError(f"Task failed: {task_info.task_command}")
        # pylint: disable=W0703
        except Exception as exc:
            if not self.stop_event.is_set():
                print(f"🔥 Triggering a global stop due to failure: {exc}")
                self.stop_event.set()
        finally:
            try:
                card_manager.free(task_info.task_type, card_list)
            # pylint: disable=W0703
            except Exception as free_error:
                print(f"🚨 Error freeing cards: {free_error}")
        return task_info.task_time

    @abstractmethod
    def schedule_group(self, tasks):
        """
        Abstract method to schedule a group of tasks.

        Subclasses must implement this method to define how to schedule and execute
        a group of TaskInfo objects. This method should handle the logic for running
        the tasks in the group, such as determining concurrency, ordering, and execution.

        Args:
            tasks (List[TaskInfo]): The list of tasks to be scheduled and executed.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("The 'schedule_group' method must be implemented by subclasses of TaskScheduler.")

    def run(self):
        """
        Executes all groups of tasks in sequence, timing each group and the total execution.
        Returns the total completion time.
        If any task fails, stops execution of remaining tasks.
        """
        self.stop_event.clear()
        total_start = time.time()
        last_end = total_start
        success = True

        try:
            for group_type, tasks in self.grouped_tasks.items():
                if self.stop_event.is_set() or not tasks:
                    continue
                print(f"\n=== Processing Group: {group_type.value} ({len(tasks)} tasks) ===")
                group_start = last_end
                self.schedule_group(tasks)
                group_end = time.time()
                last_end = group_end
                print(f"Group {group_type.value} completed in {group_end - group_start:.3f}s")
        # pylint: disable=W0703
        except Exception as e:
            print(f"\n⛔ Execution stopped due to failure: {e}")
            self.stop_event.set()
        finally:
            total_time = time.time() - total_start
            status = "INTERRUPTED" if self.stop_event.is_set() else "COMPLETED"
            success = (status == "COMPLETED")
            print(f"\nTotal execution {status} in {total_time:.3f}s")
        return success, total_time


class GreedyTaskScheduler(TaskScheduler):
    """
    GreedyTaskScheduler implements a greedy scheduling strategy for task execution.
    This scheduler prioritizes tasks with the longest execution time, sorting them in descending order
    before scheduling. It determines the concurrency level based on the number of available resources
    (e.g., cards) in the group, assuming all tasks in the group share the same group_type.
    Methods
    -------
    schedule_group(tasks):
        Schedule a group of tasks using a greedy strategy, prioritizing longer tasks first.
        Executes tasks in parallel up to the concurrency limit defined by the group's resources.
    Parameters
    ----------
    tasks : list
        A list of task objects to be scheduled. Each task must have 'task_time' and 'group_type' attributes.
    Notes
    -----
    - The concurrency is set to the value of 'group_type' of the first task in the sorted list.
    - Uses multiprocessing.Pool to execute tasks in parallel.
    """
    def schedule_group(self, tasks):
        """
        Schedule and execute a group of tasks using a greedy strategy that prioritizes
        tasks with the longest execution time.
        Args:
            tasks (list): A list of task objects to be scheduled.
                Each task is expected to have a 'task_time' attribute and a 'task_type' attribute.
        Returns:
            None
        Strategy:
            - Tasks are sorted in descending order based on their execution time.
            - The number of concurrent threads is determined by dividing NODE_WORK_NUM
                by the value of the task type (assuming all tasks in the group have the same type).
            - The actual number of threads used is the minimum of the calculated concurrency and the number of tasks.
            - Tasks are executed concurrently using a ThreadPoolExecutor.
            - If an exception occurs in any worker,
                all remaining tasks are cancelled and a global stop event is triggered.
            - The thread pool is gracefully shut down after execution.
        Notes:
            - If the stop event is set during execution, remaining tasks are not started.
            - Any exception raised by a worker will be propagated to the caller.
        """
        # Sort tasks in descending order by execution time
        sorted_tasks = sorted(tasks, key=lambda t: t.task_time, reverse=True)
        if not sorted_tasks:
            return
        # Concurrency is determined by the number of cards per group (assuming all tasks have the same group_type)
        concurrency = NODE_WORK_NUM // sorted_tasks[0].task_type.value if sorted_tasks else 1

        actual_threads = min(concurrency, len(sorted_tasks))
        with ThreadPoolExecutor(max_workers=actual_threads) as executor:
            # futures = [executor.submit(self.worker, task, self.card_manager) for task in sorted_tasks]
            futures = {executor.submit(self.worker, task, self.card_manager): task for task in sorted_tasks}
            try:
                for future in as_completed(futures):
                    if self.stop_event.is_set():
                        break
                    # This will re-raise any exception from the worker, allowing run() to catch it
                    future.result()
            except Exception as exc:
                # Cancel all not-yet-started futures
                if not self.stop_event.is_set():
                    print(f"Group failure triggering global stop: {exc}")
                    self.stop_event.set()
                raise
            finally:
                # Gracefully shut down the thread pool
                executor.shutdown(wait=False)


def collect_task_cases(level_mark: str):
    """
    Collects all test cases under test_X_cards_cases directories for the specified level mark.
    Recursively collects all subdirectories and their .py files.
    Returns a GreedyTaskScheduler with all collected TaskInfo objects.
    """
    scheduler = GreedyTaskScheduler()
    # Recursively find all Python files starting with test_ in current directory and subdirectories
    py_files = glob.glob(os.path.join(WORK_DIR, "**", "test_*.py"), recursive=True)
    # Exclude this file itself
    current_file = os.path.abspath(__file__)
    py_files = [f for f in py_files if os.path.abspath(f) != current_file]
    for py_file in py_files:
        # Dynamically import the module
        module_name = os.path.splitext(os.path.basename(py_file))[0]
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Get _TASK_TYPE and _LEVEL_0_TASK_TIME or _LEVEL_1_TASK_TIME
        task_type = getattr(module, "_TASK_TYPE")
        if level_mark == "level0":
            task_time = getattr(module, "_LEVEL_0_TASK_TIME")
        else:
            task_time = getattr(module, "_LEVEL_1_TASK_TIME")
        task_info = TaskInfo(
            task_time=task_time,
            task_command=f"pytest -vs --disable-warnings -m '{level_mark}' {py_file}",
            task_type=task_type
        )
        scheduler.add_task(task_info)
    return scheduler


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_multi_cards_level0_cases():
    """
    Feature: Multi-card Level 0 Test Execution
    Description: This test function gathers all task cases labeled as "level0" using the `collect_task_cases` function,
    initializes a scheduler with these cases, and executes them by invoking the scheduler's `run` method.
    Expectation: All "level0" multi-card task cases are executed successfully without errors.
    """
    scheduler = collect_task_cases("level0")
    success, total_time = scheduler.run()

    # 检查执行结果
    assert success, "One, or more tasks failed during execution."
    print(f"All level0 tasks completed successfully in {total_time:.3f}s")

@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_multi_cards_level1_cases():
    """
    Feature: Multi-card Level 0 Test Execution
    Description: This test function gathers all task cases labeled as "level1" using the `collect_task_cases` function,
    initializes a scheduler with these cases, and executes them by invoking the scheduler's `run` method.
    Expectation: All "level1" multi-card task cases are executed successfully without errors.
    """
    scheduler = collect_task_cases("level1")
    success, total_time = scheduler.run()

    # 检查执行结果
    assert success, "One, or more tasks failed during execution."
    print(f"All level1 tasks completed successfully in {total_time:.3f}s")
