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

"""Running the pipeline transform, once and in one place.

Every test here starts the same way — probe-parse the kernel, then transform it under the
probe's facts — and then reads a different part of the result, so the run is shared and the
reading is not.

The buffer declarations stay in the test modules that use them: the parser reads a factory's
own source to place a declaration, so which file a factory lives in is part of what the
declaration tests assert.
"""

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypto_pro._errors import span_source
from pypto_pro.language.parser._span_tracker import SpanTracker
from pypto_pro.runtime.pipeline import transform_pipeline
from pypto_pro.runtime.pipeline._analyzer import probe_kernel_facts
from test_common import assert_ir_matches_dump

from pypto.pypto_impl import ir


@dataclass
class Transformed:
    """One transform's result, in the forms the tests read it in."""

    # The kernel definition the transform ran on.
    kd: Any
    # The transformed kernel.
    tree: ast.AST
    # The KernelFacts the probe parse collected.
    facts: Any

    @property
    def source(self) -> str:
        """The transformed kernel as source."""
        return ast.unparse(self.tree)

    def parse_target(self, target):
        """The target's IR, parsed from the transformed kernel as @pl.jit would parse it."""
        self.kd._func_def = self.tree
        self.kd._pipeline = None
        return self.kd.parse_target_program(target)


def transform(kernel) -> Transformed:
    """Run the pipeline transform on ``kernel`` the way @pl.jit does."""
    kd = kernel.to_kernel_def()
    facts = probe_kernel_facts(kd, None)
    spans = SpanTracker(kd._source_file, kd._source_lines, kd._line_offset, kd._col_offset)
    with span_source(spans.get_span):
        tree = transform_pipeline(
            kd._func_def, kd._closure_vars, kd._pipeline, facts=facts,
            tilingkey_consts=kd._tilingkey_consts, datatype_consts=kd._datatype_consts,
        )
    return Transformed(kd, tree, facts)


def assert_targets_match_dumps(result: Transformed, test_file: str, case: str) -> None:
    """Each target's IR matches its dump ``<test file>__<case>__<target>.pypto`` beside the test file."""
    test_path = Path(test_file)
    for target in (ir.SectionKind.Cube, ir.SectionKind.Vector):
        program, _ = result.parse_target(target)
        golden_path = test_path.with_name(f"{test_path.stem}__{case}__{target.name.lower()}.pypto")
        assert_ir_matches_dump(program, golden_path)
