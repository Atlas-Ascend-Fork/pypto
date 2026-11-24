# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Valueless operations execute as statements and cannot supply expression operands."""

import ast
from textwrap import indent

from pypto_pro import ir
from pypto_pro._errors import PyptoProError
import pypto_pro.language as pl
from pypto_pro.language.parser._ast_parser import ASTParser
from pypto_pro.runtime.jit import _codegen_target_cce
from pypto_pro.runtime.platform import NpuArch
import pytest

_CALLS = [
    ("pl.mark_stamp(pl.PipeType.S, 0)", "system.mark_stamp", "mark_stamp<PIPE_S, 0>"),
    ("pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)",
     "system.sync_src_dyn", "set_flag("),
    ("pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)",
     "system.sync_dst_dyn", "wait_flag("),
    ("pl.system.bar_all()", "system.bar_all", "pipe_barrier(PIPE_ALL)"),
    ("pl.system.bar_mte2()", "system.bar_mte2", "pipe_barrier(PIPE_MTE2)"),
    ("pl.system.dsb()", "system.dsb", "dsb("),
    ("pl.system.dcci(out)", "system.dcci", "dcci("),
    ("pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=0)", "system.mutex_lock_dyn", "get_buf("),
    ("pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=0)", "system.mutex_unlock_dyn", "rls_buf("),
    ("pl.system.wait_pre_task_end()", "system.wait_pre_task_end", "WaitPreTaskEnd<true>"),
    ("pl.system.set_next_task_start()", "system.set_next_task_start", "SetNextTaskStart<"),
    ("pl.system.icache_preload(1)", "system.icache_preload", "ICachePreLoad("),
    ("pl.setval(out, 0, 7)", "block.setval", "out_0_ptr + 0"),
    ("pl.printf('probe=%d', flag)", "debug.printf", "pypto::aicore::printf("),
    ("pl.trap()", "debug.trap", "trap();"),
]


def ignore(value):
    pass


def optional(value=None):
    pass


def identity(value):
    return value


def _parse(body):
    source = "def kernel(out: pl.Tensor[[16], pl.DT_INT32], flag: pl.DT_INT32):\n" + indent(body, "    ") + "\n"
    parser = ASTParser(
        __file__, source.splitlines(), ir.SectionKind.Vector,
        closure_vars={"pl": pl, "ignore": ignore, "optional": optional, "identity": identity, "absent": None},
        debug_info=ir.IRDebugInfo(),
    )
    return parser.parse_function(ast.parse(source).body[0])


def _codegen(function):
    program = ir.Program([function], function.name, ir.Span.unknown(), ir.IRDebugInfo())
    return _codegen_target_cce(program, NpuArch.DAV_3510, "", ir.SectionKind.Vector).content


@pytest.mark.parametrize("call,ir_name,instruction", _CALLS, ids=[c[1] for c in _CALLS])
@pytest.mark.parametrize("body", [
    "{call}",
    "value = {call}",
])
def test_call_executes_once_when_result_unused(call, ir_name, instruction, body):
    function = _parse(body.format(call=call))
    calls = [stmt.value for stmt in function.body.stmts
             if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call) and stmt.value.name == ir_name]
    assert len(calls) == 1
    assert _codegen(function).count(instruction) == 1


@pytest.mark.parametrize("call,ir_name,instruction", _CALLS, ids=[c[1] for c in _CALLS])
@pytest.mark.parametrize("body", [
    "value = {call}\nvalue + 1",
    "value = {call}\nignore(value)",
    "value = {call}\nunused = (1, value)",
    "value = {call}\nvalue == 0",
    "unused = ({call}, 1)",
    "unused = (1, {call})",
    "selected = ({call}, 1)[1]",
    "_, selected = ({call}, 1)",
    "unused = [1, {call}]",
    "unused = pl.make_tuple(first={call}, second=1)",
    "ignore({call})",
    "pl.system.wait_pre_task_end(force={call})",
    "if {call}:\n    pass",
    "while {call}:\n    pass",
    "{call} + 1",
    "-{call}",
    "not {call}",
    "{call} == 0",
    "0 < {call}",
    "flag and {call}",
    "flag or {call}",
    "pl.min({call}, 1)",
    "return {call}",
    "value = {call} if flag else 1",
    "value = 1 if flag else {call}",
    "value = {call} if True else 1",
])
def test_void_result_cannot_be_used_as_value(call, ir_name, instruction, body):
    with pytest.raises((PyptoProError, ValueError)):
        _parse(body.format(call=call))


@pytest.mark.parametrize("body", [
    "unused = (None, 1)",
    "unused = (1, None)",
    "unused = (None, None)",
    "selected = (None, 1)[1]",
    "unused = [None, 1]",
    "unused = pl.make_tuple(first=None, second=1)",
    "unused = ((1, None), 2)",
])
def test_tuple_rejects_valueless_elements_even_when_unused(body):
    with pytest.raises(PyptoProError, match="Tuple elements must have values"):
        _parse(body)


@pytest.mark.parametrize("body", ["ignore(None)", "optional()", "optional(None)", "optional(absent)"])
def test_explicit_none_remains_a_valid_optional_argument(body):
    _codegen(_parse(body))


def test_value_returning_calls_work_in_arguments_tuples_and_comparisons():
    function = _parse("values = (pl.get_block_idx(), 1)\nvalue = identity(pl.min(values[0], flag))\nvalue == 0")
    _codegen(function)


def test_cached_void_call_is_checked_when_reused_as_an_operand():
    source = "pl.system.bar_all()"
    parser = ASTParser(__file__, [source], ir.SectionKind.Vector, closure_vars={"pl": pl},
                       debug_info=ir.IRDebugInfo())
    node = ast.parse(source).body[0].value
    parser.parse_expression(node, nested=False)
    with pytest.raises(PyptoProError, match="returns no value"):
        parser.parse_expression(node)


def test_value_returning_side_effect_is_preserved():
    function = _parse("ignore(pl.atomic_add(out, 0, 1))")
    assert _codegen(function).count("atomicAdd(") == 1


def test_bare_none_still_works_as_a_return_value():
    function = _parse("return None")
    assert isinstance(function.body.stmts[-1], ir.ReturnStmt)
