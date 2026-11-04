#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pytest configuration control"""

import os
from typing import List, Optional

import pytest


def duration_estimate(seconds: float):
    """
    Decorator: annotate a test case with estimated duration (seconds).

    This decorator marks test cases with their expected execution time,
    allowing pytest to reorder tests for optimal parallel execution.

    Args:
        seconds: Estimated execution time in seconds

    Example:
        @duration_estimate(120)
        def test_something():
            ...
    """

    def decorator(func):
        # Store the time cost as a public attribute on the function
        func.duration_estimate = seconds
        return func

    return decorator


def _set_process_desc(desc: str):
    try:
        import setproctitle

        setproctitle.setproctitle(desc)
    except ModuleNotFoundError:
        pass


def pytest_addoption(parser: pytest.Parser):
    """Register custom options with pytest

    :param parser: pytest.Parser instance
    """
    parser.addoption("--device", nargs="+", type=int, help="Device ID, default 0")
    parser.addoption("--test_case_info", action="store", default="", help="Test case info.")
    parser.addoption(
        "--cards-per-case",
        type=int,
        default=1,
        help="Number of cards required for each test case. Default is 1 (single-card cases).",
    )


def pytest_configure(config):
    """Set enable_slice=true on root scope at pytest startup.

    Overrides the false default in tile_fwk_config.json. The root scope is not affected by
    config::Reset() / reset_options(), so Clear() in UT SetUp will not clear this setting.
    Individual test cases can still override via @pypto.options(pass_options={"enable_slice": False}).
    """
    try:
        import pypto.pypto_impl

        pypto.pypto_impl.SetGlobalConfig({"pass.enable_slice": True})
    except Exception:
        pass


def _is_case_match_cards(item, target_cards) -> bool:
    """
    Check whether a test case matches the target number of cards
    """
    cards_marker = item.get_closest_marker("world_size")
    if cards_marker is None:
        return True
    required_cards = cards_marker.args
    if not required_cards:
        return True

    if isinstance(required_cards[0], int):
        return target_cards == required_cards[0]

    return True


def pytest_configure_node(node):
    """pytest-xdist callback, invoked before the main pytest process forks worker processes.

    :param node: worker node
    """
    # Get DeviceId list; when --device is passed externally, it is an STest scenario, otherwise UTest
    device_id_lst: Optional[List[int]] = node.config.getoption("--device")
    cards_per_case: int = node.config.getoption("--cards-per-case", 1)

    if device_id_lst:
        if cards_per_case > 1:
            # Multi-card mode
            if len(device_id_lst) % cards_per_case != 0:
                raise ValueError(f"Cannot divide {len(device_id_lst)} devices into groups of {cards_per_case}")

            # Calculate the device group to assign to this worker
            num_groups = len(device_id_lst) // cards_per_case
            worker_idx = int(str(node.gateway.id).lstrip("gw"))

            if worker_idx >= num_groups:
                # Not enough device groups; do not assign devices to this worker
                node.gateway.id = "NoDevices"
                node.gateway.remote_exec('import os; os.environ.pop("TILE_FWK_DEVICE_ID", None)')
                node.gateway.remote_exec('import os; os.environ.pop("TILE_FWK_DEVICE_ID_LIST", None)')
                return

            # Assign device group
            start_idx = worker_idx * cards_per_case
            end_idx = start_idx + cards_per_case
            device_group = device_id_lst[start_idx:end_idx]
            device_group_str = ",".join(map(str, device_group))

            node.gateway.id = f"Devices[{device_group_str}]"
            node.gateway.remote_exec(f'import os; os.environ["TILE_FWK_DEVICE_ID_LIST"] = "{device_group_str}"')
        else:
            # Single-card mode, keep original logic
            worker_idx = int(str(node.gateway.id).lstrip("gw"))
            if worker_idx >= len(device_id_lst):
                raise ValueError(f"WorkerIdx[{worker_idx}] out of DeviceIdLst{device_id_lst} range.")
            device_id: int = device_id_lst[worker_idx]

            # Rename worker and set DeviceId in the worker
            node.gateway.id = f"Device[{device_id}]"  # Reflected in output
            node.gateway.remote_exec(f'import os; os.environ["TILE_FWK_DEVICE_ID"] = "{device_id}"')
    else:
        node.gateway.remote_exec('import os; os.environ.pop("TILE_FWK_DEVICE_ID", None)')


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    # Prefer device group list
    device_list_str: Optional[str] = os.environ.get("TILE_FWK_DEVICE_ID_LIST", None)
    if device_list_str is not None:
        device_list = device_list_str.split(",")
        _set_process_desc(f"Devices[{','.join(device_list)}]")
    else:
        device_id: Optional[str] = os.environ.get("TILE_FWK_DEVICE_ID", None)
        if device_id is not None:
            _set_process_desc(f"Device[{device_id}]")
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Manually save coverage data in forked child processes.

    pytest-forked child processes exit via ``os._exit()``, which skips coverage's
    ``atexit`` data write, causing all coverage data to be lost in ``--forked`` mode.
    This hook reuses the Coverage instance already created by pytest-cov (with source
    and data_suffix configured), manually calling stop + save after test execution.
    Data is written to suffixed files and automatically merged by pytest-cov's
    combine() in finish().
    """
    cov_file = os.environ.get("COVERAGE_FILE")
    is_forked = item.config.getoption("forked", default=False)
    if not (cov_file and is_forked):
        yield
        return

    cov_plugin = item.config.pluginmanager.get_plugin("_cov")
    if cov_plugin is None or cov_plugin.cov_controller is None:
        yield
        return

    cov = cov_plugin.cov_controller.cov
    if cov is None:
        yield
        return

    try:
        yield
    finally:
        cov.stop()
        cov.save()


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Called after the test case process starts"""
    item_path = str(item.fspath).replace("\\", "/")
    if "/tests/ut/" in item_path:
        try:
            import pypto

            pypto.pypto_impl.Reset()
        except Exception:
            pass

    # Get the number of cards for the current run mode
    device_list_str: Optional[str] = os.environ.get("TILE_FWK_DEVICE_ID_LIST", None)
    if device_list_str is not None:
        # Multi-card
        device_list = device_list_str.split(",")
    else:
        # Single-card
        device_id: Optional[str] = os.environ.get("TILE_FWK_DEVICE_ID", None)

    # Set process description
    case_name: str = str(item.name)
    if device_list_str is not None:
        device_list = device_list_str.split(",")
        _set_process_desc(f"Case(Devices[{','.join(device_list)}]::{case_name})")
    else:
        device_id: Optional[str] = os.environ.get("TILE_FWK_DEVICE_ID", None)
        if device_id is not None:
            _set_process_desc(f"Case(Device[{device_id}]::{case_name})")
    return None  # Continue with the default test flow


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item):
    """Clean up the C++ Program singleton's global state after test case execution.

    PyPTO's C++ side ``Program`` is a process-level singleton. Test cases accumulate
    ``Function`` / ``Operation`` graph nodes via ``pypto.function()``. ``EndFunction``
    only ends the current function recording and does not clean up global graph objects
    left by previous cases, causing subsequent cases in the same process to access
    dangling ``shared_ptr<Operation>`` during ``GetGraphInfo`` /
    ``RemoveCallOpViewAssemble`` stages, triggering segfaults.

    In ``--forked`` mode each case runs in an independent child process. However the
    parent (xdist worker) process accumulates state across forks, so ``Reset()`` is
    called in both setup (pre-fork, cleans parent) and teardown (post-test, cleans
    forked child before exit).

    Only applies to UTest (python/tests/ut) paths to avoid affecting STest NPU device state.
    """
    item_path = str(item.fspath).replace("\\", "/")
    if "/tests/ut/" not in item_path:
        return
    try:
        import pypto

        pypto.pypto_impl.Reset()
    except Exception:
        pass


def _get_test_time_cost(item):
    """
    Get the estimated duration of a test case

    Args:
        item: pytest test item

    Returns:
        int or None: duration in seconds, or None if not marked
    """
    # Check if the function has a duration_estimate attribute
    if hasattr(item.function, 'duration_estimate'):
        return item.function.duration_estimate

    # Check if the class has a duration_estimate attribute
    if hasattr(item, 'cls') and item.cls and hasattr(item.cls, 'duration_estimate'):
        return item.cls.duration_estimate

    time_marker = item.get_closest_marker("duration_estimate")
    if time_marker and time_marker.args:
        return time_marker.args[0]

    return None


def _get_soc_version():
    """
    Get soc version from torch_npu
    """
    try:
        import torch_npu

        soc_version = torch_npu.npu.get_soc_version()
        return soc_version
    except Exception as e:
        pytest.exit(f"Error: Failed to get soc version, error info: {str(e)}", returncode=1)
        return None


def _is_case_match_soc(item, target_soc):
    """
    Check whether a test case matches the target soc version
    """
    soc_marker = item.get_closest_marker("soc")
    if soc_marker is None:
        supported_socs = ["910"]
    else:
        # Parse supported versions from marker (compatible with single/multiple version formats)
        supported_socs = soc_marker.args
        if isinstance(supported_socs[0], str):
            supported_socs = [soc.strip() for soc in supported_socs]
        elif isinstance(supported_socs[0], list):
            supported_socs = [soc.strip() for soc in supported_socs[0]]
    # Core matching logic: soc 260 → "950" (A5), 270 → "960" (A6), others → "910"
    if target_soc == 260:
        target_tag = "950"
    elif target_soc == 270:
        target_tag = "960"
    else:
        target_tag = "910"
    return target_tag in supported_socs


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    """
    Perform global reordering after all conftest.py scopes have been processed
    """
    if not items:
        return
    first_item = items[0]
    item_path = str(first_item.fspath).replace("\\", "/")
    has_ut = "/ut/" in item_path.lower()

    def _is_verify_case(item):
        # Host pass_verify cases: python/tests/ut/interpreter (formerly tests/verify)
        return "ut/interpreter" in str(item.fspath).lower().replace("\\", "/")

    if has_ut:
        filtered_items = items
    else:
        verify_items = [item for item in items if _is_verify_case(item)]
        other_items = [item for item in items if not _is_verify_case(item)]
        # ut/interpreter covers Host pass_verify / SIM, does not depend on NPU soc detection
        if other_items:
            target_soc = _get_soc_version()
            filtered_items = [item for item in other_items if _is_case_match_soc(item, target_soc)]
            filtered_items.extend(verify_items)
        else:
            filtered_items = verify_items

    # Filter cases by card count requirement
    cards_per_case = config.getoption("--cards-per-case", 1)

    # Filter out non-matching cases at collection stage
    card_filtered_items = [item for item in filtered_items if _is_case_match_cards(item, cards_per_case)]

    # Separate test cases with and without duration markers
    timed_tests = []
    untimed_tests = []

    for item in card_filtered_items:
        time_cost = _get_test_time_cost(item)
        if time_cost is not None:
            timed_tests.append((item, time_cost))
        else:
            untimed_tests.append(item)

    timed_tests.sort(key=lambda x: x[1], reverse=True)
    reordered_items = [item for item, _ in timed_tests] + untimed_tests

    items[:] = reordered_items
