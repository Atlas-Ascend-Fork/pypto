#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Executable file execution helper module

This module provides execution of executable files (mainly GTest test programs),
test case list retrieval (ordered by estimated duration refresh),
primarily used for test case management and execution acceleration in the test framework.

Main features:
- Wrapping and execution of executable files
- Automatic retrieval of GTest test case lists
- Test case duration estimation retrieval and case list reordering (from executable metadata or JSON cache)
- Environment variable configuration and management
- Execution timeout control

Main classes:
- Exec: Executable file wrapper class, providing complete execution and management functionality
- CaseDesc: Test case description, including name and estimated duration

Usage example:
    exec_obj = Exec(file=Path("test_executable"), envs={"ENV_VAR": "value"}, timeout=300)
    case_list, case_dict = exec_obj.get_case_name_info(case_duration_json=Path("duration.json"))
    ret, cmd, duration = exec_obj.run(params=["--gtest_filter=TestSuite.TestCase"])
"""

import dataclasses
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import Dict, List, Optional, Tuple


class Exec:
    """Executable file wrapper class

    Provides executable file execution, test case list retrieval, and other functionalities.
    """

    @dataclasses.dataclass
    class CaseDesc:
        """Test case description

        Contains case name and estimated/actual execution duration
        """

        name: Optional[str] = None
        duration: Optional[float] = None

        def __init__(self, name: str, duration: Optional[float] = None):
            self.name = name
            self.duration = duration

    def __init__(self, file: Path, envs: Optional[Dict[str, str]] = None, timeout: Optional[int] = None):
        """
        :param file: Path to the executable file
        :type file: Path
        :param envs: Additional environment variables to set when executing the executable
        :type envs: Optional[Dict[str, str]]
        :param timeout: Execution timeout in seconds. If empty, reads from environment variable
            PYPTO_TESTS_CASE_EXECUTE_TIMEOUT; if not specified, no timeout limit
        :type timeout: Optional[int]
        """
        self.file: Path = Path(file).resolve()
        self.envs: Dict[str, str] = envs if envs is not None else {}
        self.timeout: Optional[int] = None
        env_timeout = os.environ.get("PYPTO_TESTS_CASE_EXECUTE_TIMEOUT", None)
        if env_timeout:
            self.timeout = int(env_timeout)
        if timeout and timeout > 0:
            self.timeout = timeout  # Parameter setting takes priority over environment variable

    @property
    def brief(self) -> str:
        """Get a brief description of the execution process

        Includes filename and ASAN/UBSAN status information

        :return: Brief description string
        """
        asan = "ON" if "ASAN_OPTIONS" in self.envs.keys() else "OFF"
        ubsan = "ON" if "UBSAN_OPTIONS" in self.envs.keys() else "OFF"
        return f"({self.file.name}) XSAN(ASAN:{asan} UBSAN:{ubsan})"

    def get_case_name_info(
        self, case_name_list: Optional[List[str]] = None, duration_json: Optional[Path] = None
    ) -> Tuple[int, List[CaseDesc], Dict[str, CaseDesc]]:
        """Get test case information and sort

        Retrieves the case list based on the specified case list or executable file,
        supplements duration estimation information, and sorts in descending order by estimated duration.

        :param case_name_list: Specified list of case names. If empty or contains "*",
            retrieves all cases from the executable
        :type case_name_list: Optional[List[str]]
        :param case_duration_json: Path to the case duration cache JSON file, used to supplement estimated duration
        :type case_duration_json: Optional[Path]
        :return: Sorted list of case descriptions and case description dictionary
        :rtype: Tuple[List[CaseDesc], Dict[str, CaseDesc]]
        """
        # Determine the list of case names to be executed
        if case_name_list is None or len(case_name_list) == 0 or "*" in case_name_list:
            case_name_list = self._get_case_name_list_origin()
            logging.info("Determine TestCase from file, get %s cases", len(case_name_list))
        else:
            logging.info("Determine TestCase from args, get %s cases", len(case_name_list))
        desc_dict = {name: self.CaseDesc(name=name) for name in case_name_list}

        # Supplement and refresh estimated duration information
        # (first from duration defined in cases, then refresh from JSON,
        # to ensure it is closer to actual duration)
        self._mdf_case_desc_dict(case_desc_dict=desc_dict, path=duration_json)

        # Reorder test cases
        desc_list = desc_dict.values()
        desc_list = sorted(
            desc_list, key=lambda x: x.duration if x.duration is not None else float('-inf'), reverse=True
        )
        ordered_cnt = 0
        for desc in desc_list:
            if desc.duration:
                ordered_cnt += 1
            else:
                break  # After sorting, cases with duration estimates are placed first
        normal_cnt = len(case_name_list) - ordered_cnt
        logging.info("Determine TestCase Order, OrderedCase(%s), NormalCase(%s)", ordered_cnt, normal_cnt)
        return ordered_cnt, desc_list, desc_dict

    def run(
        self,
        params: Optional[List[str]] = None,
        check: bool = False,
        capture_output: bool = True,
        envs: Optional[Dict[str, str]] = None,
    ) -> Tuple[subprocess.CompletedProcess, str, timedelta]:
        """Execute the executable file

        :param params: Additional command parameters
        :type params: Optional[List[str]]
        :param check: Check parameter passed through to subprocess.run
        :type check: bool
        :param capture_output: capture_output parameter passed through to subprocess.run
        :type capture_output: bool
        :param envs: Additional environment variables to set at runtime
        :type envs: Optional[Dict[str, str]]
        :return: Return value, executed command, execution duration
        :rtype: Tuple[CompletedProcess, str, timedelta]
        """
        cmd = self._get_run_cmd(params=params)
        # Environment variable priority: function parameter specified >
        # class-level environment variable (command line specified) > existing system variables
        envs = envs if envs is not None else {}
        act_env = os.environ.copy()  # System environment variables
        act_env.update(self.envs)  # Additional specified environment variables
        act_env.update(envs)  # Environment variables specified at function call time
        cwd = str(self.file.parent)
        ts = datetime.now(tz=timezone.utc)
        try:
            ret = subprocess.run(
                shlex.split(cmd),
                env=act_env,
                cwd=cwd,
                timeout=self.timeout,
                capture_output=capture_output,
                check=check,
                text=True,
                encoding='utf-8',
            )
        except subprocess.CalledProcessError as e:
            # 透出被捕获的 stdout/stderr(如 ASAN/LSAN 报告), 便于 CI 日志定位真实失败原因
            logging.error(
                "Execute failed(retcode=%s): %s\n--- stdout ---\n%s\n--- stderr ---\n%s",
                e.returncode, e.cmd, e.stdout, e.stderr,
            )
            raise
        return ret, cmd, datetime.now(tz=timezone.utc) - ts

    def _get_run_cmd(self, params: Optional[List[str]] = None) -> str:
        """Build the execution command string

        :param params: List of command line parameters
        :return: Complete execution command string
        """
        cmd = f"./{self.file.name}"
        if params:
            cmd += " " + " ".join(params)
        return cmd

    def _get_case_name_list_origin(self) -> List[str]:
        """
        Get the original test case list from the executable:

        :return: List of case names
        :rtype: List[str]
        """
        case_name_list = []
        (
            ret,
            _,
            _,
        ) = self.run(params=["--gtest_list_tests"], check=True)  # GoogleTest native parameter
        for line in ret.stdout.split('\n'):
            line = line.rstrip()
            if not line or line.startswith('#') or "GoogleTestVerification" in line:
                continue
            if line.endswith('.'):
                current_suite = line[:-1]
            elif line.startswith('  '):
                test_name = line.strip()
                full_name = f"{current_suite}.{test_name}"
                case_name_list.append(full_name)
        return case_name_list

    def _get_case_desc_list_origin(self) -> List[CaseDesc]:
        """Get the test case list with duration from the executable

        Uses custom parameter --gtest_list_tests_with_meta to get case names and estimated durations

        :return: List of case descriptions, containing case names and estimated durations
        """
        case_desc_list = []
        (
            ret,
            _,
            _,
        ) = self.run(params=["--gtest_list_tests_with_meta"], check=True)
        pattern = re.compile(r'^([\w\.]+)\|(\d+\.?\d*)$', re.MULTILINE)
        matches = pattern.findall(ret.stdout)
        for test_name, cost_str in matches:
            case_desc_list.append(self.CaseDesc(name=test_name, duration=float(cost_str.strip())))
        return case_desc_list

    def _mdf_case_desc_dict(self, case_desc_dict: Dict[str, CaseDesc], path: Optional[Path] = None):
        """Refresh duration estimates in the case description dictionary

        Prioritizes retrieval from executable metadata, then refreshes from JSON cache file

        :param case_desc_dict: Case description dictionary, key is case name, value is CaseDesc object
        :param path: Path to the case duration cache JSON file
        """
        # Refresh case duration estimates based on durations defined within cases
        update_cnt = 0
        case_desc_list = self._get_case_desc_list_origin()
        for item in case_desc_list:
            desc = case_desc_dict.get(item.name, None)
            if not desc:
                continue
            desc.duration = item.duration
            update_cnt += 1
        logging.info("Determine TestCase Order, %s case's estimate update by local define", update_cnt)

        # Refresh case duration estimates based on JSON cache file
        case_duration_dict = {}
        if path is not None and path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    case_duration_dict = json.load(f)
            except json.JSONDecodeError:
                case_duration_dict = {}

        update_cnt = 0
        for case_name, duration in case_duration_dict.items():
            desc = case_desc_dict.get(case_name, None)
            if not desc:
                continue
            desc.duration = float(duration)
            update_cnt += 1
        logging.info("Determine TestCase Order, %s case's estimate update by cache file", update_cnt)
