# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""GEMM expansion contracts, native synchronization and slot safety."""

from parser.test_gemm_ir.common import parse_gemm
from pathlib import Path
import re

from pypto_pro.language.parser.diagnostics import (
    InvalidArgument,
    InvalidOperation,
    InvalidType,
    InvalidVal,
    NotSupported,
)
import pytest


def test_compile_time_flags_and_splits_accept_local_constant_expressions(tmp_path):
    actual = parse_gemm(tmp_path, """
        chunk = 32 * 2
        initialize = not False
        pl.gemm(c, a, b, l0a=la, l0b=lb, splitk=chunk, init=initialize)
    """, l0k=64)
    expected = parse_gemm(tmp_path, "pl.gemm(c, a, b, l0a=la, l0b=lb, splitk=64, init=True)", l0k=64)
    assert re.findall(r"@([\w.]+)\(", actual) == re.findall(r"@([\w.]+)\(", expected)


@pytest.mark.parametrize("operand", ["l0a=la.next(), l0b=lb", "l0a=la, l0b=lb.next()"])
def test_compact_is_checked_through_cursor_expressions(tmp_path, operand):
    with pytest.raises(InvalidVal) as caught:
        parse_gemm(tmp_path, f"pl.gemm(c, a, b, {operand}, k=64)", compact=0)
    assert re.search("compact=1", caught.value.message)


@pytest.mark.parametrize("flag", ["trans_a", "trans_b"])
def test_static_transpose_and_dynamic_slot_rejection(tmp_path, flag):
    group, var = ("ga", "a") if flag == "trans_a" else ("gb", "b")
    with pytest.raises(NotSupported) as caught:
        parse_gemm(tmp_path, f"{var} = {group}.next()\npl.gemm(c, a, b, l0a=la, l0b=lb, {flag}=True)",
                   trans_a=flag == "trans_a", trans_b=flag == "trans_b")
    assert re.search("runtime index", caught.value.message)


def test_transpose_flag_rejects_an_existing_zn_view(tmp_path):
    with pytest.raises(NotSupported) as caught:
        parse_gemm(tmp_path, """
            b = pl.reinterpret(b, layout=pl.ZN)
            pl.gemm(c, a, b, l0a=la, l0b=lb, trans_b=True)
        """, trans_b=True)
    assert re.search("requires an NZ L1 Tile", caught.value.message)


@pytest.mark.parametrize("space", ["Mat", "Vec"])
def test_split_n_rejects_accumulator_in_other_memory_spaces(tmp_path, space):
    with pytest.raises(InvalidOperation) as caught:
        parse_gemm(tmp_path, (
            "bad = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, "
            f"target_memory=pl.MemorySpace.{space}), addrs=131072, mutex_ids=[9])[0]\n"
            "pl.gemm(bad, a, b, l0a=la, l0b=lb, splitn=32)"
        ), l0n=32)
    assert "acc must be in L0C (Acc)" in caught.value.message
    span = caught.value.span
    line = Path(span["filename"]).read_text().splitlines()[span["begin_line"] - 1]
    assert line[span["begin_column"] - 1:span["end_column"]] == "bad"


def test_split_n_rejects_dynamic_address_even_with_reused_mutex_id(tmp_path):
    with pytest.raises(NotSupported) as caught:
        parse_gemm(tmp_path, """
            gc = pl.make_tile_group(
                type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
                addrs=[0, 16384], mutex_ids=[8, 8])
            c = gc.next()
            pl.gemm(c, a, b, l0a=la, l0b=lb, splitn=32)
        """,
               acc_depth=2, l0n=32)
    assert re.search("runtime-selected accumulator", caught.value.message)


@pytest.mark.parametrize("call,kwargs,error,match", [
    ("pl.gemm(c, a, b, l0a=la)", {}, InvalidArgument, "missing a required argument: 'l0b'"),
    ("pl.gemm(c, a, b, l0a=la, l0b=lb, typo=True)", {}, InvalidArgument, "unexpected keyword"),
    ("pl.gemm(c, a, b, la, lb)", {}, InvalidArgument, "positional"),
    ("pl.gemm(c, a, b, l0a=la, l0b=lb, init=extent)", {}, InvalidType, "compile-time bool"),
    ("pl.gemm(c, a, b, l0a=la, l0b=lb, init=1)", {}, InvalidType, "compile-time bool"),
    ("pl.gemm(c, a, b, l0a=la, l0b=lb, trans_a=extent)", {}, InvalidType, "compile-time bool"),
    ("pl.gemm(c, a, b, l0a=la, l0b=lb, splitk=extent)", {}, InvalidType, "compile-time int"),
    ("pl.gemm(c, a, b, l0a=la, l0b=lb, splitk=True)", {}, InvalidType, "compile-time int"),
    ("pl.gemm(c, a, b, l0a=la, l0b=lb, splitn=1.5)", {}, InvalidType, "compile-time int"),
    ("pl.gemm(c, a, b, l0a=la, l0b=lb, splitk=0)", {}, InvalidVal, "must be at least 1"),
    ("pl.gemm(c, a, b, l0a=la, l0b=lb, splitk=64)", {}, InvalidVal, "must equal the split size"),
    ("pl.gemm(c, a, b, l0a=la, l0b=lb, splitn=24)", {}, InvalidVal, "multiple of 16"),
    ("pl.gemm(c, a, b, l0a=la, l0b=lb, splitn=32, splitk=64)", {}, InvalidArgument, "mutually exclusive"),
    ("pl.gemm(c, a, b, l0a=la, l0b=lb, splitn=32, m=48)", {}, NotSupported, "cannot combine"),
    ("pl.gemm(c, a, b, l0a=la, l0b=lb, k=192)", {}, InvalidVal, "outside"),
    ("pl.gemm(c, a, b, l0a=la, l0b=lb, k=-64)", {}, InvalidVal, "outside"),
    ("pl.gemm(c, a, b, l0a=la, l0b=lb, k=True)", {}, InvalidType, "integers"),
    ("pl.gemm(c, a, b, l0a=la, l0b=lb, m=1.5)", {}, InvalidType, "integers"),
    ("pl.gemm(c, a, b, l0a=la, l0b=lb)", {"l0k": 64}, InvalidVal, "extent must match"),
    ("pl.gemm(c, a, b, l0a=la, l0b=lb, k=128)", {"l0k": 64}, InvalidVal, "capacity"),
    ("pl.gemm(c, a, b, l0a=la, l0b=lb, trans_b=True)", {}, InvalidVal, "disagree on K"),
    ("pl.gemm(c, a, b, l0a=la, l0b=lb, m=48)", {"acc_compact": 0}, InvalidVal, "compact=1"),
    ("value = pl.gemm(c, a, b, l0a=la, l0b=lb)", {}, NotSupported, "standalone statement"),
])
def test_invalid_calls_report_the_gemm_location(tmp_path, call, kwargs, error, match):
    with pytest.raises(error) as raised:
        parse_gemm(tmp_path, call, **kwargs)
    assert re.search(match, raised.value.message)
    assert raised.value.span is not None


@pytest.mark.parametrize("option,error,marked", [
    ("typo=True", InvalidArgument, "typo"),
    ("init=extent", InvalidType, "extent"),
    ("splitk=extent", InvalidType, "extent"),
    ("splitk=0", InvalidVal, "0"),
    ("splitn=24", InvalidVal, "24"),
    ("k=192", InvalidVal, "192"),
])
def test_invalid_parameter_keeps_its_exact_source_span(tmp_path, option, error, marked):
    with pytest.raises(error) as raised:
        parse_gemm(tmp_path, f"pl.gemm(c, a, b, l0a=la, l0b=lb, {option})")
    span = raised.value.span
    lines = Path(span["filename"]).read_text().splitlines()
    assert span["begin_line"] == span["end_line"]
    assert "pl.gemm(" in lines[span["begin_line"] - 1]
    assert lines[span["begin_line"] - 1][span["begin_column"] - 1:span["end_column"]] == marked


def test_call_error_inherits_the_statement_span(tmp_path):
    call = "pl.gemm(c, a, b, l0a=la, l0b=lb, splitn=32, splitk=64)"
    with pytest.raises(InvalidArgument) as raised:
        parse_gemm(tmp_path, call)
    assert re.search("mutually exclusive", raised.value.message)
    span = raised.value.span
    line = Path(span["filename"]).read_text().splitlines()[span["begin_line"] - 1]
    assert line[span["begin_column"] - 1:span["end_column"]] == call


@pytest.mark.parametrize("k", ["32", "96", "64 + 32", "128 - 32"])
def test_mx_static_k_alignment(tmp_path, k):
    with pytest.raises(InvalidVal) as caught:
        parse_gemm(tmp_path,
               f"pl.gemm_mx(c, a, b, s1, s2, l0a=la, l0b=lb, scale_a=sa, scale_b=sb, k={k})",
               mx=True)
    assert re.search("multiple of 64", caught.value.message)


@pytest.mark.parametrize("option", ["bias=c", "splitk=64", "splitn=32"])
def test_mx_unsupported_options_are_explicit(tmp_path, option):
    with pytest.raises(NotSupported) as caught:
        parse_gemm(tmp_path,
               f"pl.gemm_mx(c, a, b, s1, s2, l0a=la, l0b=lb, scale_a=sa, scale_b=sb, {option})",
               mx=True)
    assert re.search("no bias operand|does not split", caught.value.message)


def test_mx_runtime_slot_pairing_still_checks_every_possible_address(tmp_path):
    with pytest.raises(InvalidVal) as caught:
        parse_gemm(tmp_path,
               "pl.gemm_mx(c, a, b, s1, s2, l0a=la, l0b=lb, scale_a=sa, scale_b=sb)",
               k=64, mx=True, bad_plane=True)
    assert re.search("slot.*address|address.*slot", caught.value.message)


def test_mx_narrowed_k_requires_compact_planes(tmp_path):
    with pytest.raises(InvalidVal) as caught:
        parse_gemm(tmp_path,
               "pl.gemm_mx(c, a, b, s1, s2, l0a=la, l0b=lb, scale_a=sa, scale_b=sb, k=64)",
               mx=True, plane_compact=0)
    assert re.search("compact=1.*MX scale group axis", caught.value.message)
