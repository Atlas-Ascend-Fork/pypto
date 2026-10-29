#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-only regression tests for the swimlane pipe usage report."""

import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

PROFILING_DIR = Path(__file__).resolve().parents[1] / "profiling"
sys.path.insert(0, str(PROFILING_DIR))
try:
    import draw_swim_lane as swim
finally:
    sys.path.pop(0)


class TestPipeUsage(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.directory = Path(self.temp_dir.name)
        self.original_cores = swim.total_cores
        self.original_start = swim.mininum_start_time
        self.original_end = swim.max_end_time
        self.original_args = getattr(swim, "args", None)
        self.original_leaf_funcs = swim.pipe_time.leaf_funcs
        self.original_pipe_list = swim.pipe_time.pipe_list
        self.addCleanup(self.restore_globals)
        swim.pipe_time.leaf_funcs = {}
        swim.pipe_time.pipe_list = []

    def restore_globals(self):
        swim.total_cores = self.original_cores
        swim.mininum_start_time = self.original_start
        swim.max_end_time = self.original_end
        if self.original_args is None:
            swim.__dict__.pop("args", None)
        else:
            swim.args = self.original_args
        swim.pipe_time.leaf_funcs = self.original_leaf_funcs
        swim.pipe_time.pipe_list = self.original_pipe_list

    @staticmethod
    def make_core(index, core_type, name, start, end):
        task = swim.TaskInfo(index)
        task.func_name = name
        task.exec_start = start
        task.exec_end = end
        core = swim.CoreInfo(index, core_type)
        core.tasks = [task]
        return core

    def report(self, cores, leaves, start=0, end=10):
        leaf_path = self.directory / "leaf_cycles.json"
        leaf_path.write_text(json.dumps(leaves), encoding="utf-8")
        swim.args = SimpleNamespace(pipe_exec_time=str(leaf_path))
        swim.total_cores = {index: core for index, core in enumerate(cores)}
        swim.mininum_start_time = start
        swim.max_end_time = end
        report_path = self.directory / "pipe_usage.csv"
        swim.calculate_pipe_usage(report_path)
        with report_path.open(encoding="utf-8", newline="") as stream:
            return list(csv.reader(stream, skipinitialspace=True))

    @staticmethod
    def leaf(name, cycles, pipes):
        return {"FuncName": name, "FuncHash": 1, "TotalCycles": cycles, "pipes": pipes}

    @staticmethod
    def find_row(rows, name):
        return next(row for row in rows if row and row[0] == name)

    def test_mixed_cores_produce_correct_pipe_usage(self):
        cores = [self.make_core(0, "AIC", "cube", 0, 10), self.make_core(1, "AIV", "vec", 2, 10)]
        leaves = [
            self.leaf("cube", 1800, {"CUBE": 900, "MTE_IN": 360, "MTE1": 180, "MTE_OUT": 180}),
            self.leaf("vec", 3600, {"VECTOR_ALU": 1800, "MTE_IN": 900, "MTE_OUT": 450}),
        ]
        rows = self.report(cores, leaves)
        self.assertEqual(rows[:3], [["Total Core Num:2"], ["AIC: 1"], ["AIV: 1"]])
        cube = self.find_row(rows, "CUBE")
        vector = self.find_row(rows, "VECTOR_ALU")
        self.assertAlmostEqual(float(cube[1]), 0.5)
        self.assertAlmostEqual(float(cube[3].rstrip("%")), 5.0)
        self.assertAlmostEqual(float(vector[1]), 1.0)
        self.assertAlmostEqual(float(vector[3].rstrip("%")), 10.0)
        self.assertAlmostEqual(float(self.find_row(rows, "MTE_IN")[1]), 2.0)
        aic = self.find_row(rows, "AIC_0")
        aiv = self.find_row(rows, "AIV_1")
        self.assertEqual([float(aic[index]) for index in (1, 2, 4, 6, 8, 10)], [10.0, 2.0, 1.0, 1.0, 0.5, 0.0])
        self.assertEqual([float(aiv[index]) for index in (1, 2, 4, 6, 8, 10)], [8.0, 2.0, 0.0, 1.0, 0.0, 1.0])
        self.assertAlmostEqual(float(aiv[11].rstrip("%")), 12.5)

    def test_single_core_types_and_unit_duration(self):
        for core_type, pipe, absent_pipe in (("AIC", "CUBE", "VECTOR_ALU"), ("AIV", "VECTOR_ALU", "CUBE")):
            with self.subTest(core_type=core_type):
                rows = self.report(
                    [self.make_core(0, core_type, "leaf", 0, 1)], [self.leaf("leaf", 1800, {pipe: 900})], end=1
                )
                self.assertEqual(rows[0], ["Total Core Num:1"])
                self.assertEqual(float(self.find_row(rows, absent_pipe)[1]), 0.0)
                self.assertEqual(float(self.find_row(rows, pipe)[1]), 0.5)
                self.assertEqual(float(self.find_row(rows, core_type + "_0")[1]), 1.0)

    def test_unmatched_leaf_does_not_block_report(self):
        core = self.make_core(0, "AIC", "missing", 0, 10)
        matched = self.make_core(1, "AIV", "vec", 0, 10)
        rows = self.report([core, matched], [self.leaf("vec", 1800, {"VECTOR_ALU": 900})])
        self.assertEqual(float(self.find_row(rows, "CUBE")[1]), 0.0)
        self.assertEqual(float(self.find_row(rows, "VECTOR_ALU")[1]), 0.5)
        self.assertEqual(float(self.find_row(rows, "AIC_0")[8]), 0.0)

    def test_empty_and_zero_duration_reports_have_finite_zero_usage(self):
        for cores in ([], [swim.CoreInfo(0, "AIC")], [self.make_core(0, "AIC", "leaf", 5, 5)]):
            with self.subTest(cores=cores):
                rows = self.report(cores, [], start=5, end=5)
                self.assertEqual(rows[0], ["Total Core Num:0"])
                for pipe in ("CUBE", "VECTOR_ALU", "MTE_IN", "MTE1", "MTE_OUT"):
                    row = self.find_row(rows, pipe)
                    self.assertEqual(float(row[1]), 0.0)
                    self.assertEqual(float(row[3].rstrip("%")), 0.0)

    def test_fake_task_does_not_hide_real_task(self):
        core = self.make_core(0, "AIC", "leaf", 0, 10)
        fake = swim.TaskInfo(10)
        fake.is_fake = True
        fake.exec_start, fake.exec_end = -1, 0
        core.tasks.insert(0, fake)
        rows = self.report([core], [self.leaf("leaf", 1800, {"CUBE": 900})])
        self.assertEqual(rows[0], ["Total Core Num:1"])
        self.assertEqual(float(self.find_row(rows, "CUBE")[1]), 0.5)
        self.assertEqual(float(self.find_row(rows, "AIC_0")[1]), 10.0)


if __name__ == "__main__":
    unittest.main()
