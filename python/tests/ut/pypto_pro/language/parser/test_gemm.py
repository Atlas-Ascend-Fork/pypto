# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""GEMM expansion contracts, native synchronization and slot safety."""

import importlib.util
import itertools
from pathlib import Path
import re
import textwrap

from pypto_pro import ir
from pypto_pro.language.parser.diagnostics import (
    InvalidArgument,
    InvalidOperation,
    InvalidType,
    InvalidVal,
    NotSupported,
)
import pytest

_SERIAL = itertools.count()


def _parse(tmp_path, body, *, k=128, n=64, l0k=None, l0n=None, compact=1,
           acc_compact=1, acc_depth=1, trans_a=False, trans_b=False,
           auto=False, mx=False, plane_compact=0, bad_plane=False, static_validshape=False):
    """Use a real source file so both the parser and location diagnostics are exercised."""
    filename = tmp_path / f"gemm_case_{next(_SERIAL)}.py"
    ak, bn = l0k or k, l0n or n
    a_shape = [k, 64] if trans_a else [64, k]
    b_shape = [n, k] if trans_b else [k, n]
    dtype = "DT_FP8E4M3FN" if mx else "DT_FP16"
    def group_sync(values):
        return f'mutex_ids="auto", depth={len(values)}' if auto else f"mutex_ids={values}"
    decls = [
        ("ga", a_shape, dtype, "Mat", "NZ", 0, [0, 1], 0),
        ("gb", b_shape, dtype, "Mat", "NZ", 65536, [2, 3], 0),
        ("la", [64, ak], dtype, "Left", "NZ", 0, [4, 5], compact),
        ("lb", [ak, bn], dtype, "Right", "ZN", 0, [6, 7], compact),
        ("gc", [64, n], "DT_FP32", "Acc", "NZ", 0, list(range(8, 8 + acc_depth)), acc_compact),
    ]
    declarations = []
    for name, shape, dt, space, layout, addr, mutex, comp in decls:
        valid = f", valid_shape={shape}" if static_validshape else ""
        declarations.append(
            f"{name} = pl.make_tile_group(type=pl.TileType(shape={shape}, dtype=pl.{dt}, "
            f"target_memory=pl.MemorySpace.{space}, layout=pl.{layout}, compact={comp}{valid}), "
            f"addrs={addr}, {group_sync(mutex)})"
        )
    if mx:
        data_bytes = 64 * ak
        for name, shape, space, layout, addr, mutex in (
            ("gsa", [64, k // 32], "Mat", "ZZ", 131072, [10, 11]),
            ("gsb", [k // 32, n], "Mat", "NN", 135168, [12, 13]),
            ("sa", [64, ak // 32], "ScaleLeft", "ZZ",
             [0, (data_bytes >> 4) + (64 if bad_plane else 0)], [14, 15]),
            ("sb", [ak // 32, bn], "ScaleRight", "NN", [0, data_bytes >> 4], [16, 17]),
        ):
            declarations.append(
                f"{name} = pl.make_tile_group(type=pl.TileType(shape={shape}, dtype=pl.DT_FP8E8M0, "
                f"target_memory=pl.MemorySpace.{space}, layout=pl.{layout}, compact={plane_compact}), "
                f"addrs={addr}, {group_sync(mutex)})"
            )
        declarations += ["s1 = gsa[0]", "s2 = gsb[0]"]
    declarations += ["a = ga[0]", "b = gb[0]", "c = gc[0]"]
    source = (
        "import pypto_pro.language as pl\nfrom pypto_pro.language import gemm as product\n"
        "@pl.jit(auto_mutex=True)\n"
        "def kernel(arg: pl.Tensor[[1], pl.DT_FP16], extent: pl.DT_INT32):\n"
        + textwrap.indent("\n".join(declarations) + "\n" + textwrap.dedent(body).strip(), "    ") + "\n"
    )
    filename.write_text(source)
    spec = importlib.util.spec_from_file_location(filename.stem, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    program, _ = module.kernel.to_kernel_def().parse_target_program(ir.SectionKind.Cube)
    program = ir.Pass.insert_sync()(program)
    return ir.python_print(program.get_function("kernel"))


def _sequence(text):
    result = []
    aliases = {}
    mutex_ids = {}
    for line in text.splitlines():
        alias = re.match(r"\s*(\w+)(?::.*?)? = (\w+)(?:\[[^\]]+\])?$", line)
        if alias:
            aliases[alias.group(1)] = aliases.get(alias.group(2), alias.group(2))
        getter = re.match(r"\s*(\w+).* = ir.call @block.tile_mutex_id\((\w+), (\d+)\)", line)
        if getter:
            mutex_ids[getter.group(1)] = (aliases.get(getter.group(2), getter.group(2)), getter.group(3))
        block = re.search(r"@(block\.[a-z_0-9]+)\(", line)
        if block and block.group(1) not in ("block.make_mutex_tile", "block.tile_mutex_id"):
            result.append(block.group(1))
        lock = re.search(r"@(system\.mutex_(?:un)?lock)_dyn\((\w+),", line)
        if lock:
            pipe = re.search(r'pipe="pipe": (\d+)', line).group(1)
            result.append((lock.group(1), pipe, mutex_ids.get(lock.group(2), lock.group(2))))
    return result


def _validshapes(text):
    return [line.split("@block.set_validshape(", 1)[1].split(",", 1)[1]
            for line in text.splitlines() if "@block.set_validshape(" in line]


@pytest.mark.parametrize("init", [True, False])
def test_native_staging_and_mutex_equivalence(tmp_path, init):
    sugar = f"pl.gemm(c, a, b, l0a=la, l0b=lb, init={init})"
    manual = (
        "x = la.next()\ny = lb.next()\n"
        "pl.set_validshape(x, [64, 128])\npl.set_validshape(y, [128, 64])\n"
        "pl.set_validshape(c, [64, 64])\npl.move(x, a)\npl.move(y, b)\n"
        + ("pl.matmul(c, x, y)" if init else "pl.matmul_acc(c, c, x, y)")
    )
    actual = _parse(tmp_path, sugar)
    assert _sequence(actual) == _sequence(_parse(tmp_path, manual))
    assert "block.gemm" not in actual
    assert {entry[1] for entry in _sequence(actual) if isinstance(entry, tuple)} == {"0", "3"}


@pytest.mark.parametrize("axis,shapes", [
    ("m", [[48, 128], [48, 64]]),
    ("n", [[128, 48], [64, 48]]),
    ("k", [[64, 64], [64, 64]]),
])
def test_effective_shapes_preserve_other_declared_axes(tmp_path, axis, shapes):
    value = 64 if axis == "k" else 48
    actual = _parse(tmp_path, f"pl.gemm(c, a, b, l0a=la, l0b=lb, {axis}={value})")
    assert actual.count("@block.set_validshape(") == 3
    for shape in shapes:
        assert repr(shape) in actual
    for line in actual.splitlines():
        if "@block.set_validshape(" in line:
            assert not re.search(r"\((a|b)_\d+,", line)


def test_runtime_extent_is_bound_once(tmp_path):
    actual = _parse(tmp_path, "pl.gemm(c, a, b, l0a=la, l0b=lb, m=pl.min(64, extent))")
    assert actual.count("ir.min(") == 1
    assert actual.count("@block.set_validshape(") == 3


def test_inline_cursor_expressions_are_evaluated_once(tmp_path):
    actual = _parse(tmp_path, "pl.gemm(gc.next(), ga.next(), gb.next(), l0a=la.next(), l0b=lb.next())")
    manual = _parse(tmp_path, """
        z = gc.next()
        x = ga.next()
        y = gb.next()
        u = la.next()
        v = lb.next()
        pl.set_validshape(u, [64, 128])
        pl.set_validshape(v, [128, 64])
        pl.set_validshape(z, [64, 64])
        pl.move(u, x)
        pl.move(v, y)
        pl.matmul(z, u, v)
    """)
    assert actual.count("@struct.set(") == manual.count("@struct.set(")
    assert _sequence(actual) == _sequence(manual)


@pytest.mark.parametrize("view", ["input", "staging"])
def test_inline_views_keep_operand_order_and_single_cursor_evaluation(tmp_path, view):
    a = "pl.reinterpret(a, shape=[64, 128], layout=pl.NZ)" if view == "input" else "ga.next()"
    u = "pl.reinterpret(la[0], shape=[64, 128], layout=pl.NZ)" if view == "staging" else "la.next()"
    actual = _parse(tmp_path, f"pl.gemm(gc.next(), {a}, gb.next(), l0a={u}, l0b=lb.next())")
    manual = _parse(tmp_path, f"""
        z = gc.next()
        x = {a}
        y = gb.next()
        u = {u}
        v = lb.next()
        pl.set_validshape(u, [64, 128])
        pl.set_validshape(v, [128, 64])
        pl.set_validshape(z, [64, 64])
        pl.move(u, x)
        pl.move(v, y)
        pl.matmul(z, u, v)
    """)
    assert actual.count("@struct.set(") == manual.count("@struct.set(")
    assert _sequence(actual) == _sequence(manual)


@pytest.mark.parametrize("operand", ["l0a=la.next(), l0b=lb", "l0a=la, l0b=lb.next()"])
def test_compact_is_checked_through_cursor_expressions(tmp_path, operand):
    with pytest.raises(InvalidVal) as caught:
        _parse(tmp_path, f"pl.gemm(c, a, b, {operand}, k=64)", compact=0)
    assert re.search("compact=1", caught.value.message)


@pytest.mark.parametrize("flag", ["trans_a", "trans_b"])
def test_static_transpose_and_dynamic_slot_rejection(tmp_path, flag):
    actual = _parse(tmp_path, f"pl.gemm(c, a, b, l0a=la, l0b=lb, {flag}=True)",
                    trans_a=flag == "trans_a", trans_b=flag == "trans_b")
    assert 'blayout="blayout": 1, slayout="slayout": 2' in actual
    group, var = ("ga", "a") if flag == "trans_a" else ("gb", "b")
    with pytest.raises(NotSupported) as caught:
        _parse(tmp_path, f"{var} = {group}.next()\npl.gemm(c, a, b, l0a=la, l0b=lb, {flag}=True)",
               trans_a=flag == "trans_a", trans_b=flag == "trans_b")
    assert re.search("runtime index", caught.value.message)


def test_bare_imported_alias_can_emit_transpose_constants(tmp_path):
    actual = _parse(tmp_path, "product(c, a, b, l0a=la, l0b=lb, trans_b=True)", trans_b=True)
    assert "@block.matmul(" in actual


def test_transpose_flag_rejects_an_existing_zn_view(tmp_path):
    with pytest.raises(NotSupported) as caught:
        _parse(tmp_path, """
            b = pl.reinterpret(b, layout=pl.ZN)
            pl.gemm(c, a, b, l0a=la, l0b=lb, trans_b=True)
        """, trans_b=True)
    assert re.search("requires an NZ L1 Tile", caught.value.message)


def test_explicit_k_can_use_smaller_compact_staging(tmp_path):
    actual = _parse(tmp_path, "pl.gemm(c, a, b, l0a=la, l0b=lb, k=64)", l0k=64)
    assert actual.count("@block.set_validshape(") == 3


def test_generated_names_do_not_capture_kernel_variables(tmp_path):
    actual = _parse(tmp_path, """
        _pl_gemm1_l0a = la.next()
        pl.gemm(c, a, b, l0a=la, l0b=lb)
        pl.move(_pl_gemm1_l0a, a)
    """)
    assert "_pl_gemm2_l0a" in actual
    assert actual.count("@block.move(") == 3


@pytest.mark.parametrize("init", [True, False])
@pytest.mark.parametrize("tail", [False, True])
def test_split_k_controls_loop_and_initialization(tmp_path, init, tail):
    actual = _parse(tmp_path, f"pl.gemm(c, a, b, l0a=la, l0b=lb, splitk=64, init={init})",
                    k=96 if tail else 128, l0k=64)
    assert f"ir.range(0, {96 if tail else 128}, 64)" in actual
    assert ("@block.matmul(" in actual) == init
    assert "@block.matmul_acc(" in actual
    assert actual.count("@block.set_validshape(") == 3


@pytest.mark.parametrize("auto", [True, False])
@pytest.mark.parametrize("n,split", [(64, 32), (64, 48), (80, 48)])
@pytest.mark.parametrize("view", [False, True])
def test_split_n_places_contiguous_alias_strips(tmp_path, auto, n, split, view):
    base = 65536 if view else 0
    body = f"pl.gemm(c, a, b, l0a=la, l0b=lb, splitn={split})"
    if view:
        sync = 'mutex_ids="auto", depth=1' if auto else 'mutex_ids=[9]'
        body = (
            f"wide = pl.make_tile_group(type=pl.TileType(shape=[64, {n * 2}], dtype=pl.DT_FP32, "
            f"target_memory=pl.MemorySpace.Acc, compact=1), addrs={base}, {sync})\n"
            f"c = pl.reinterpret(wide[0], shape=[64, {n}])\n" + body
        )
    actual = _parse(tmp_path, body, n=n, l0n=split, auto=auto)
    # Both strips are views of the selected accumulator, including an already reinterpreted source.
    slices = re.findall(r"(\w+) = ir.call @block.subview\((c_\d+), \[0, (\d+)\],", actual)
    assert [int(col) for _, _, col in slices] == [0, split]
    assert len({source for _, source, _ in slices}) == 1
    for (window, _, _), width in zip(slices, (split, ((n - split + 15) // 16) * 16)):
        assert f"@block.reinterpret({window}, [64, {width}]," in actual
    assert f"ir.range(0, {n}, {split})" in actual
    # InsertSync uses the original accumulator ID for every strip after AUTO assignment.
    accumulator = "wide" if view else "gc"
    mutex_id = re.search(
        rf"_tg_{accumulator}_tiles_0_0 = ir.call @block.make_mutex_tile\([^\n]*?, \[\], \[(\d+)\]", actual
    ).group(1)
    locks = [entry for entry in _sequence(actual) if isinstance(entry, tuple) and entry[0] == "system.mutex_lock"]
    assert locks.count(("system.mutex_lock", "3", mutex_id)) == actual.count("@block.matmul(")
    assert "-214748" not in actual


@pytest.mark.parametrize("space", ["Mat", "Vec"])
def test_split_n_rejects_accumulator_in_other_memory_spaces(tmp_path, space):
    with pytest.raises(InvalidOperation) as caught:
        _parse(tmp_path, (
            "bad = pl.make_tile_group(type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, "
            f"target_memory=pl.MemorySpace.{space}), addrs=131072, mutex_ids=[9])[0]\n"
            "pl.gemm(bad, a, b, l0a=la, l0b=lb, splitn=32)"
        ), l0n=32)
    assert "acc must be in L0C (Acc)" in caught.value.message
    span = caught.value.span
    line = Path(span["filename"]).read_text().splitlines()[span["begin_line"] - 1]
    assert line[span["begin_column"] - 1:span["end_column"]] == "bad"


def test_split_n_accepts_static_slot_of_multibuffer_accumulator(tmp_path):
    actual = _parse(tmp_path, "c = gc[1]\npl.gemm(c, a, b, l0a=la, l0b=lb, splitn=32)",
                    acc_depth=2, l0n=32)
    assert "@block.matmul(" in actual
    assert 'memref_addr="memref_addr": 16384' in actual


def test_split_n_rejects_dynamic_address_even_with_reused_mutex_id(tmp_path):
    with pytest.raises(NotSupported) as caught:
        _parse(tmp_path, """
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
    ("pl.gemm(c, a, b, l0a=la, l0b=lb, splitk=extent)", {}, InvalidType, "compile-time int"),
    ("pl.gemm(c, a, b, l0a=la, l0b=lb, splitk=0)", {}, InvalidVal, "positive"),
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
        _parse(tmp_path, call, **kwargs)
    assert re.search(match, raised.value.message)
    assert raised.value.span is not None


@pytest.mark.parametrize("option,error,marked", [
    ("typo=True", InvalidArgument, "typo"),
    ("init=extent", InvalidType, "extent"),
    ("splitk=extent", InvalidType, "extent"),
    ("splitn=24", InvalidVal, "24"),
    ("k=192", InvalidVal, "192"),
])
def test_invalid_parameter_keeps_its_exact_source_span(tmp_path, option, error, marked):
    with pytest.raises(error) as raised:
        _parse(tmp_path, f"pl.gemm(c, a, b, l0a=la, l0b=lb, {option})")
    span = raised.value.span
    lines = Path(span["filename"]).read_text().splitlines()
    assert span["begin_line"] == span["end_line"]
    assert "pl.gemm(" in lines[span["begin_line"] - 1]
    assert lines[span["begin_line"] - 1][span["begin_column"] - 1:span["end_column"]] == marked


def test_call_error_inherits_the_statement_span(tmp_path):
    call = "pl.gemm(c, a, b, l0a=la, l0b=lb, splitn=32, splitk=64)"
    with pytest.raises(InvalidArgument) as raised:
        _parse(tmp_path, call)
    assert re.search("mutually exclusive", raised.value.message)
    span = raised.value.span
    line = Path(span["filename"]).read_text().splitlines()[span["begin_line"] - 1]
    assert line[span["begin_column"] - 1:span["end_column"]] == call


def test_none_optional_arguments_match_the_defaults(tmp_path):
    actual = _parse(tmp_path, "pl.gemm(c, a, b, l0a=la, l0b=lb, m=None, splitk=None, bias=None)")
    assert _sequence(actual) == _sequence(_parse(tmp_path, "pl.gemm(c, a, b, l0a=la, l0b=lb)"))


@pytest.mark.parametrize("options,partial,kwargs", [
    ("", "m=48, n=40, k=64", {}),
    (", splitk=64", "m=48, n=40, k=96", {"l0k": 64}),
    (", splitn=48", "n=64, k=64", {"n": 80, "l0n": 48}),
    (", scale_a=su, scale_b=sv", "m=48, n=40, k=64", {"mx": True, "plane_compact": 1}),
])
def test_omitted_extents_restore_reused_tiles(tmp_path, options, partial, kwargs):
    mx = kwargs.get("mx", False)
    operands = "c, a, b, s1, s2" if mx else "c, a, b"
    call = f"pl.gemm{'_mx' if mx else ''}({operands}, l0a=u, l0b=v{options}"
    prefix = "u = la[0]\nv = lb[0]\n" + ("su = sa[0]\nsv = sb[0]\n" if mx else "")
    prefix += f"{call}, {partial})\n"
    actual = _parse(tmp_path, prefix + call + ")", **kwargs)
    full = f"n={kwargs.get('n', 64)}, k=128" + ("" if "splitn" in options else ", m=64")
    explicit = _parse(tmp_path, prefix + call + f", {full})", **kwargs)
    assert _sequence(actual) == _sequence(explicit)
    assert _validshapes(actual) == _validshapes(explicit)


@pytest.mark.parametrize("options,kwargs", [
    ("m=64, n=64, k=128", {}),
    ("splitk=64, m=64, n=64, k=128", {"l0k": 64}),
    ("splitn=32, n=64, k=128", {"l0n": 32}),
    ("m=64, n=64, k=128, scale_a=sa, scale_b=sb", {"mx": True}),
    ("k=64", {"l0k": 64}),
])
def test_full_physical_extents_accept_noncompact_tiles(tmp_path, options, kwargs):
    mx = kwargs.get("mx", False)
    operands = "c, a, b, s1, s2" if mx else "c, a, b"
    actual = _parse(tmp_path, f"pl.gemm{'_mx' if mx else ''}({operands}, l0a=la, l0b=lb, {options})",
                    compact=0, acc_compact=0, **kwargs)
    assert "@block.matmul" in actual


def test_full_static_descriptors_need_no_runtime_update(tmp_path):
    actual = _parse(tmp_path, "pl.gemm(c, a, b, l0a=la, l0b=lb, m=64, n=64, k=128)",
                    static_validshape=True, compact=0, acc_compact=0)
    assert "@block.matmul(" in actual
    assert "@block.set_validshape(" not in actual


@pytest.mark.parametrize("init", [True, False])
def test_mx_adjacent_pairs_and_native_mutex_equivalence(tmp_path, init):
    sugar = f"pl.gemm_mx(c, a, b, s1, s2, l0a=la, l0b=lb, scale_a=sa, scale_b=sb, init={init})"
    manual = """
        x = la.next()
        y = lb.next()
        sx = sa.next()
        sy = sb.next()
        pl.set_validshape(x, [64, 64])
        pl.set_validshape(y, [64, 64])
        pl.set_validshape(c, [64, 64])
        pl.set_validshape(sx, [64, 2])
        pl.set_validshape(sy, [2, 64])
        pl.move(x, a)
        pl.move(sx, s1)
        pl.move(y, b)
        pl.move(sy, s2)
    """ + ("\npl.matmul_mx(c, x, y, sx, sy)" if init else "\npl.matmul_mx_acc(c, c, x, y, sx, sy)")
    # Dedent the shared prologue before appending the final statement.
    manual = textwrap.dedent(manual.rsplit("\npl.", 1)[0]) + "\npl." + manual.rsplit("\npl.", 1)[1]
    actual = _parse(tmp_path, sugar, k=64, mx=True, auto=True)
    assert _sequence(actual) == _sequence(_parse(tmp_path, manual, k=64, mx=True, auto=True))
    assert [entry for entry in _sequence(actual) if isinstance(entry, str)] == [
        *["block.set_validshape"] * 5,
        "block.move", "block.move", "block.move", "block.move",
        "block.matmul_mx" if init else "block.matmul_mx_acc",
    ]


@pytest.mark.parametrize("axis,value", [("m", 48), ("n", 48), ("k", 64)])
def test_mx_valid_extents_reach_plane_axes(tmp_path, axis, value):
    actual = _parse(tmp_path,
                    f"pl.gemm_mx(c, a, b, s1, s2, l0a=la, l0b=lb, scale_a=sa, scale_b=sb, {axis}={value})",
                    mx=True, plane_compact=1 if axis == "k" else 0)
    assert actual.count("@block.set_validshape(") == 5
    if axis == "k":
        assert "[64, 2]" in actual and "[2, 64]" in actual


def test_mx_runtime_k_reaches_planes_as_group_count(tmp_path):
    actual = _parse(tmp_path,
                    "pl.gemm_mx(c, a, b, s1, s2, l0a=la, l0b=lb, scale_a=sa, scale_b=sb, k=extent)",
                    mx=True, plane_compact=1)
    assert actual.count("@block.set_validshape(") == 5
    assert actual.count("// 32") >= 1 or "@scalar.floordiv(" in actual


@pytest.mark.parametrize("k", ["32", "96", "64 + 32", "128 - 32"])
def test_mx_static_k_alignment(tmp_path, k):
    with pytest.raises(InvalidVal) as caught:
        _parse(tmp_path,
               f"pl.gemm_mx(c, a, b, s1, s2, l0a=la, l0b=lb, scale_a=sa, scale_b=sb, k={k})",
               mx=True)
    assert re.search("multiple of 64", caught.value.message)


@pytest.mark.parametrize("option", ["bias=c", "splitk=64", "splitn=32"])
def test_mx_unsupported_options_are_explicit(tmp_path, option):
    with pytest.raises(NotSupported) as caught:
        _parse(tmp_path,
               f"pl.gemm_mx(c, a, b, s1, s2, l0a=la, l0b=lb, scale_a=sa, scale_b=sb, {option})",
               mx=True)
    assert re.search("no bias operand|does not split", caught.value.message)


def test_mx_runtime_slot_pairing_still_checks_every_possible_address(tmp_path):
    with pytest.raises(InvalidVal) as caught:
        _parse(tmp_path,
               "pl.gemm_mx(c, a, b, s1, s2, l0a=la, l0b=lb, scale_a=sa, scale_b=sb)",
               k=64, mx=True, bad_plane=True)
    assert re.search("slot.*address|address.*slot", caught.value.message)


def test_mx_narrowed_k_requires_compact_planes(tmp_path):
    with pytest.raises(InvalidVal) as caught:
        _parse(tmp_path,
               "pl.gemm_mx(c, a, b, s1, s2, l0a=la, l0b=lb, scale_a=sa, scale_b=sb, k=64)",
               mx=True, plane_compact=0)
    assert re.search("compact=1.*MX scale group axis", caught.value.message)
