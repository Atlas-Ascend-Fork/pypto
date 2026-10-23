# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""GEMM access analysis before native staging expansion."""

import ast
import re
import textwrap

from pypto_pro import ir
from pypto_pro._errors import InvalidArgument, InvalidOperation, NotSupported, span_source
import pypto_pro.language as pl
from pypto_pro.runtime.pipeline._cross_core_scanner import AccessScanTables, scan_stage_accesses
import pytest


def _scan(body, *, tracked=("ga", "gb"), shared=()):
    source = (
        "def stage(gc, ga, gb, la, lb, bg):\n"
        "    c = gc.current()\n    a = ga.next()\n    b = gb.next()\n    bias = bg.current()\n"
        + textwrap.indent(textwrap.dedent(body).strip(), "    ") + "\n"
    )
    function = ast.parse(source).body[0]
    names = ("gc", "ga", "gb", "la", "lb", "bg")
    tables = AccessScanTables(
        cross_buffers=dict.fromkeys(tracked),
        all_buffer_memory={"ga": pl.MemorySpace.Mat, "gb": pl.MemorySpace.Mat,
                           "gc": pl.MemorySpace.Acc, "la": pl.MemorySpace.Left,
                           "lb": pl.MemorySpace.Right, "bg": pl.MemorySpace.Bias},
        group_names=set(names),
        kernel_slot_to_buffer={}, tuple_fields={}, vf_func_defs={}, helper_func_defs={},
        addr_shared=set(shared), closure_vars={"pl": pl, "product": pl.gemm},
    )
    def location(node):
        return ir.Span("pipeline_gemm.py", node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)
    with span_source(location):
        return scan_stage_accesses(function, [ast.Name(id=name, ctx=ast.Load()) for name in names], tables)


@pytest.mark.parametrize("init", [True, False])
@pytest.mark.parametrize("tracked", [("ga",), ("gb",), ("ga", "gb")])
def test_l1_reads_use_staging_pipe_with_local_accumulator(tracked, init):
    bias = ", bias=bias" if init else ""
    actual = _scan(f"pl.gemm(c, a, b, l0a=la, l0b=lb, init={init}{bias})", tracked=tracked)
    assert actual == [(name, "R", "MTE1") for name in tracked]


def test_same_l1_buffer_is_one_read():
    assert _scan("pl.gemm(c, a, a, l0a=la, l0b=lb)", tracked=("ga",)) == [("ga", "R", "MTE1")]


def test_imported_gemm_alias_and_tile_alias_are_traced():
    assert _scan("x = a\nproduct(c, x, b, l0a=la, l0b=lb)") == [
        ("ga", "R", "MTE1"), ("gb", "R", "MTE1"),
    ]


def test_address_reused_l1_buffer_is_tracked():
    assert _scan("pl.gemm(c, a, b, l0a=la, l0b=lb)", tracked=(), shared=("ga",)) == [
        ("ga", "R", "MTE1"),
    ]


def test_native_expansion_has_the_same_l1_accesses():
    sugar = _scan("pl.gemm(c, a, b, l0a=la, l0b=lb)")
    primitive = _scan("x = la.next()\ny = lb.next()\npl.move(x, a)\npl.move(y, b)\npl.matmul(c, x, y)")
    # The scanner resolves the move's pipe from both sides' declared memory spaces.
    assert sugar == primitive


def test_purely_local_gemm_requires_no_cross_core_pipe():
    assert _scan("pl.gemm(c, a, b, l0a=la, l0b=lb)", tracked=()) == []


@pytest.mark.parametrize("init", [True, False])
@pytest.mark.parametrize("address_reuse", [True, False])
def test_tracked_accumulator_is_rejected(init, address_reuse):
    options = {"tracked": (), "shared": ("gc",)} if address_reuse else {"tracked": ("gc",)}
    with pytest.raises(InvalidOperation) as error:
        _scan(f"pl.gemm(c, a, b, l0a=la, l0b=lb, init={init})", **options)
    assert re.search("cannot hand over its accumulator", error.value.message)
    assert error.value.span["line"] == 6
    assert error.value.span["filename"] == "pipeline_gemm.py"


@pytest.mark.parametrize("name,keyword", [("la", "l0a"), ("lb", "l0b"), ("bg", "bias")])
def test_tracked_keyword_operands_are_not_silently_ignored(name, keyword):
    with pytest.raises(InvalidOperation) as caught:
        _scan("pl.gemm(c, a, b, l0a=la, l0b=lb, bias=bias)", tracked=(name,))
    assert re.search(f"cannot hand over its {keyword} buffer", caught.value.message)


def test_inline_slot_selection_keeps_pipeline_contract():
    with pytest.raises(NotSupported) as caught:
        _scan("pl.gemm(c, ga.next(), b, l0a=la, l0b=lb)")
    assert "A slot pick must stand alone" in caught.value.message


def test_inline_tracked_subview_uses_its_l1_buffer():
    assert _scan("pl.gemm(c, a[:, :], b, l0a=la, l0b=lb)", tracked=("ga",)) == [
        ("ga", "R", "MTE1"),
    ]


def test_untraceable_tracked_expression_is_rejected():
    with pytest.raises(NotSupported) as caught:
        _scan("pl.gemm(c, a + 0, b, l0a=la, l0b=lb)", tracked=("ga",))
    assert re.search("tracked.*operand must be assigned to a simple variable", caught.value.message)


def test_mx_handover_remains_explicitly_unsupported():
    with pytest.raises(InvalidArgument) as caught:
        _scan("pl.gemm_mx(c, a, b, bias, bias, l0a=la, l0b=lb, scale_a=la, scale_b=lb)")
    assert re.search("pl.gemm_mx.*no entry", caught.value.message)


@pytest.mark.parametrize("keyword", ["l0a", "l0b", "scale_a", "scale_b"])
@pytest.mark.parametrize("address_reuse", [False, True])
def test_mx_tracked_keyword_operands_are_rejected(keyword, address_reuse):
    operands = dict(l0a="la", l0b="lb", scale_a="la", scale_b="lb")
    operands[keyword] = "bias"
    keywords = ", ".join(f"{name}={value}" for name, value in operands.items())
    options = {"tracked": (), "shared": ("bg",)} if address_reuse else {"tracked": ("bg",)}
    with pytest.raises(InvalidArgument) as caught:
        _scan(f"pl.gemm_mx(c, a, b, a, b, {keywords})", **options)
    assert "no entry" in caught.value.message
    assert "['bg']" in caught.value.message


def test_purely_local_mx_is_not_rejected_by_access_scanning():
    assert not _scan("pl.gemm_mx(c, a, b, a, b, l0a=la, l0b=lb, scale_a=la, scale_b=lb)", tracked=())
