# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Aicpu target validation: frontend contract + validate_target_ops backend.

``mode="aicpu_kernel"`` alone selects the Aicpu execution domain: the kernel
takes no section, and any ``with pl.*()`` inside it is rejected at parse time.
Everything the control CPU cannot honor is rejected as early as possible:

- frontend: any section inside an aicpu kernel, pipeline-stage calls (the one
  form with no downstream guard -- a stage would inline silently);
- backend (``ir.Pass.validate_target_ops(SectionKind.Aicpu)`` -- the Aicpu
  CoreMask): everything that still reaches a program is checked against the
  GM/ptr/struct allow-list -- debug probes, AI-Core queries, tile ops, system
  and channel interfaces all die here with "cannot execute on AICPU", still
  before the bisheng AICPU cross-compile.
"""

from __future__ import annotations

import importlib
import re

from pypto_pro._errors import InvalidOperation
import pypto_pro.language as pl
import pypto_pro.language.distribute as pld
import pytest

from pypto.pypto_impl import ir

jit = importlib.import_module("pypto_pro.runtime.jit")
aicpu_compile = importlib.import_module("pypto_pro.runtime.aicpu_compile")

CORES_CV = "Supported cores for this call: Cube and Vector."
CORES_VEC = "Supported cores for this call: Vector."


def _parse(kernel):
    """Parse one aicpu kernel for the Aicpu target and validate its ops."""
    from pypto_pro.runtime.kernel import validate_target_ops_or_raise

    program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Aicpu)
    validate_target_ops_or_raise(ir.SectionKind.Aicpu, program)
    return program


# ---------------------------------------------------------------------------
# Frontend: no sections at all in an aicpu kernel
# ---------------------------------------------------------------------------


def test_aicpu_kernel_needs_no_section():
    # mode="aicpu_kernel" alone selects the execution domain: the body parses
    # directly into the Aicpu program with no section wrapper.
    @pl.jit(mode="aicpu_kernel")
    def kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        n = pl.numel(x)
        for i in pl.range(n):
            pl.setval(x, i, pl.getval(x, i) * 2)

    _parse(kernel)


@pytest.mark.parametrize("attr", ["section_vector", "section_cube"])
def test_ai_core_sections_in_aicpu_kernel_are_rejected(attr, tmp_path):
    # source must live in a real file: the parser reads the kernel's source
    path = tmp_path / "kernel_case.py"
    path.write_text(
        "import pypto_pro.language as pl\n\n\n"
        "@pl.jit(mode='aicpu_kernel')\n"
        "def kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):\n"
        "    pl.setval(x, 0, 1)\n"
        f"    with pl.{attr}():\n"
        "        pl.setval(x, 1, 0)\n",
        encoding="utf-8",
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("kernel_case", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(InvalidOperation, match=f"Section 'pl.{attr}' cannot be used inside an aicpu kernel"):
        mod.kernel.to_kernel_def().parse_target_program(ir.SectionKind.Aicpu)


def test_section_aicpu_in_aicpu_kernel_is_rejected():
    # The section no longer exists: even the old spelling is rejected as a
    # plain section inside an aicpu kernel.
    @pl.jit(mode="aicpu_kernel")
    def kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        with pl.section_aicpu():
            pl.setval(x, 0, 1)

    with pytest.raises(InvalidOperation, match="Section 'pl.section_aicpu' cannot be used inside an aicpu kernel"):
        kernel.to_kernel_def().parse_target_program(ir.SectionKind.Aicpu)


# ---------------------------------------------------------------------------
# Backend: Tile interfaces are rejected (make_tile reaches IR, S pipe -> CV)
# ---------------------------------------------------------------------------


def test_tile_construction_is_rejected():
    @pl.jit(mode="aicpu_kernel")
    def kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        tf = pl.TileType(shape=[16, 16], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
        tile = pl.make_tile(tf, addr=0)  # noqa: F841 -- DSL body, parsed as AST

    with pytest.raises(InvalidOperation, match=f"cannot execute on AICPU. {re.escape(CORES_CV)}"):
        _parse(kernel)


# ---------------------------------------------------------------------------
# SIMT / VF / stage
# ---------------------------------------------------------------------------


def test_simt_launch_is_intercepted():
    @pl.vector_function(mode="simt", max_threads=8)
    def helper(v: pl.DT_INT32) -> pl.DT_INT32:
        return v

    @pl.jit(mode="aicpu_kernel")
    def kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        helper[1, 1](1)  # noqa: F841 -- DSL body, parsed as AST

    # The aicpu kernel is parsed for the Aicpu target (interlock), and the
    # launch's own scope rule rejects every non-Vector parse target.
    with pytest.raises(
        InvalidOperation,
        match="SIMT vector functions must be invoked inside 'with pl\\.section_vector\\(\\):'",
    ):
        _parse(kernel)


def test_vf_function_call_is_rejected_by_the_backend():
    @pl.vector_function
    def helper(v: pl.DT_INT32):
        total = v + 1  # noqa: F841 -- DSL body, parsed as AST

    @pl.jit(mode="aicpu_kernel")
    def kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        helper(1)  # noqa: F841 -- DSL body, parsed as AST

    # The call parses (a simd.call reaches the IR); the Aicpu CoreMask in
    # validate_target_ops is the layer that rejects it -- still before bisheng.
    with pytest.raises(InvalidOperation, match=f"cannot execute on AICPU. {re.escape(CORES_VEC)}"):
        _parse(kernel)


def test_vf_op_is_intercepted():
    from pypto_pro.language import Vf as vf  # noqa: N813

    @pl.jit(mode="aicpu_kernel")
    def kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        m = vf.create_mask()  # noqa: F841 -- DSL body, parsed as AST

    # The VF execution-domain gate fires first (vf.* needs @pl.vector_function).
    with pytest.raises(InvalidOperation, match="Operation 'vf.create_mask' can only be used inside"):
        _parse(kernel)


def test_pipeline_stage_call_is_intercepted():
    @pl.pipeline.stage
    def stage_helper(v: pl.DT_INT32):
        return v

    @pl.jit(mode="aicpu_kernel")
    def kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        stage_helper(1)  # noqa: F841 -- DSL body, parsed as AST

    with pytest.raises(InvalidOperation, match="pipeline stage 'stage_helper' cannot be called"):
        _parse(kernel)


def test_window_parameter_is_rejected_at_codegen():
    # The parse itself accepts the WindowType parameter; the rejection fires in
    # the AICPU bare-body codegen (EmitSingleFunctionSignature), still before
    # any bisheng invocation.
    @pl.jit(mode="aicpu_kernel")
    def kernel(win: pld.Window, x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        pl.setval(x, 0, 1)

    program = _parse(kernel)
    with pytest.raises(RuntimeError, match="AICPU bare body does not support window parameter"):
        aicpu_compile.generate_aicpu_bare_body(
            program, __import__("pypto_pro.runtime.platform", fromlist=["NpuArch"]).NpuArch.DAV_3510, "kernel"
        )


def test_tiling_class_parameter_is_rejected_at_codegen():
    from dataclasses import dataclass

    @dataclass
    class Tiling:  # the tiling-class shape the AI Core path expands
        n: int

    @pl.jit(mode="aicpu_kernel")
    def kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32], t: Tiling):
        pl.setval(x, 0, 1)

    program = _parse(kernel)
    with pytest.raises(RuntimeError, match="AICPU bare body does not support tiling struct parameter"):
        aicpu_compile.generate_aicpu_bare_body(
            program, __import__("pypto_pro.runtime.platform", fromlist=["NpuArch"]).NpuArch.DAV_3510, "kernel"
        )


def test_channel_creation_is_rejected():
    # The channel creation lowers to a dist op, which the Aicpu CoreMask
    # rejects (channel ops run on the AIV).
    @pl.jit(mode="aicpu_kernel")
    def kernel(win: pld.Window, x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        ch = pld.channel(win, 0)  # noqa: F841 -- DSL body, parsed as AST

    with pytest.raises(InvalidOperation, match=f"cannot execute on AICPU. {re.escape(CORES_VEC)}"):
        _parse(kernel)


# ---------------------------------------------------------------------------
# The op surface: positives
# ---------------------------------------------------------------------------


def test_non_surface_op_receiving_tensor_is_rejected():
    @pl.jit(mode="aicpu_kernel")
    def kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        pl.init_output(x, size=4, value=0)

    with pytest.raises(InvalidOperation, match=f"cannot execute on AICPU. {re.escape(CORES_VEC)}"):
        _parse(kernel)


def test_struct_and_tuple_surface_parse():
    @pl.jit(mode="aicpu_kernel")
    def kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32], out: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        plan = pl.struct("Plan", total=0, cursor=0)
        fields = pl.make_tuple(a=1, b=2)  # map surface: the tuple literal
        total = pl.numel(x)
        pl.setval(out, 0, pl.getval(x, 0) + plan.total + fields.a + total)  # noqa: F841

    _parse(kernel)


def test_ptr_surface_and_scalar_math_parse():
    @pl.jit(mode="aicpu_kernel")
    def kernel(x: pl.Ptr[pl.DT_INT32], out: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32], n: pl.DT_INT32):
        base = pl.make_ptr(x, pl.DT_INT64)
        view = pl.make_tensor(base, [n], [1])
        lo = pl.min(pl.getval(view, 0), pl.max(0, n))
        pl.setval(out, 0, pl.astype(lo, pl.DT_INT32))

    _parse(kernel)


def test_tensor_view_surface_parses_and_folds_to_the_source_pointer():
    # make_ptr/make_tensor accept a Tensor operand, with the AI Core view
    # semantics: pure metadata over the same GM pointer, nothing copied.
    @pl.jit(mode="aicpu_kernel")
    def kernel(x: pl.Tensor[[64, 128], pl.DT_FP16], out: pl.Tensor[[64, 128], pl.DT_FP16]):
        p = pl.make_ptr(x)
        row1 = pl.make_tensor(p, [128], [1])
        trans = pl.make_tensor(x, [128, 64], [1, 128])
        pl.setval(out, 0, pl.getval(row1, 0) + pl.getval(trans, 0))

    program = _parse(kernel)
    body = aicpu_compile.generate_aicpu_bare_body(
        program, __import__("pypto_pro.runtime.platform", fromlist=["NpuArch"]).NpuArch.DAV_3510, "kernel"
    )
    # The views fold back to the tensor's own pointer: reinterpret_cast and
    # direct dereference -- no copy anywhere.
    assert "((half*)(x_0_ptr))" in body
    assert "copy" not in body.lower()


def test_addptr_surface_is_rejected():
    @pl.jit(mode="aicpu_kernel")
    def kernel(x: pl.Ptr[pl.DT_INT32], out: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        shifted = pl.addptr(x, 8)  # noqa: F841 -- DSL body, parsed as AST

    with pytest.raises(InvalidOperation, match=f"cannot execute on AICPU. {re.escape(CORES_CV)}"):
        _parse(kernel)


def test_dynamic_input_surface_is_rejected():
    @pl.jit(mode="aicpu_kernel")
    def kernel(desc: pl.Ptr[pl.DT_INT32], out: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        first = pl.dynamic_input_ptr(desc, 0)  # noqa: F841 -- DSL body, parsed as AST

    with pytest.raises(InvalidOperation, match=f"cannot execute on AICPU. {re.escape(CORES_CV)}"):
        _parse(kernel)


def test_inline_python_helper_still_works():
    def double(v):
        return v * 2

    @pl.jit(mode="aicpu_kernel")
    def kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32], out: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        pl.setval(out, 0, double(pl.getval(x, 0)))

    program = _parse(kernel)
    assert program.get_function(kernel.__name__) is not None


# ---------------------------------------------------------------------------
# Frontend gate stops the bisheng cross-compile itself
# ---------------------------------------------------------------------------


def test_rejected_kernel_never_reaches_the_aicpu_build(monkeypatch, tmp_path):
    builds = []

    def _fake_build(*args, **kwargs):
        builds.append(args)

    monkeypatch.setattr(aicpu_compile, "build_aicpu_jit_so", _fake_build)
    monkeypatch.setenv("ASCEND_WORK_PATH", str(tmp_path))

    @pl.jit(mode="aicpu_kernel")
    def kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32], out: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        pl.printf("%d", pl.getval(x, 0))
        pl.setval(out, 0, pl.getval(x, 0))

    with pytest.raises(InvalidOperation, match=f"cannot execute on AICPU. {re.escape(CORES_CV)}"):
        aicpu_compile.compile_aicpu_variant(kernel, None, (None, None, None, None), None, ())
    assert builds == []


# ---------------------------------------------------------------------------
# Backend: the Aicpu CoreMask in validate_target_ops
# ---------------------------------------------------------------------------


def _program_with_call(op_name, args, params=()):
    span = ir.Span.unknown()
    call = ir.Call(ir.Op(op_name), list(args), span)
    body = ir.SeqStmts([ir.AssignStmt([], call, span)], span)
    function = ir.Function("kernel", list(params), [], body, span)
    return ir.Program([function], "probe", span)


def _gm_tensor_param(name="tensor"):
    span = ir.Span.unknown()
    return ir.Var(name, ir.TensorType([4], ir.DataType.INT32, None), span)


def test_validate_target_ops_rejects_ai_core_only_op():
    program = _program_with_call("system.bar_m", [])
    with pytest.raises(
        ValueError,
        match="This operation cannot execute on AICPU. Supported cores for this call: Cube.",
    ):
        ir.Pass.validate_target_ops(ir.SectionKind.Aicpu)(program)


def test_validate_target_ops_rejects_unclassified_op():
    program = _program_with_call("system.nop", [])
    with pytest.raises(
        ValueError,
        match="This operation cannot execute on AICPU. Supported cores for this call: Cube and Vector.",
    ):
        ir.Pass.validate_target_ops(ir.SectionKind.Aicpu)(program)


def test_validate_target_ops_accepts_gm_scalar_access():
    tensor = _gm_tensor_param()
    zero = ir.ConstInt(0, ir.DataType.INT64, ir.Span.unknown())
    for op_name, args in (
        ("block.getval", [tensor, zero]),
        ("block.setval", [tensor, zero, zero]),
    ):
        program = _program_with_call(op_name, args, [tensor])
        ir.Pass.validate_target_ops(ir.SectionKind.Aicpu)(program)


@pytest.mark.parametrize(
    "op_name",
    [
        "ptr.make_tensor",
        "ptr.make_ptr",
        "struct.create",
        "struct.set",
    ],
)
def test_validate_target_ops_accepts_aicpu_surface_ops(op_name):
    span = ir.Span.unknown()
    ptr = ir.Var("ptr", ir.PtrType(ir.DataType.INT32), span)
    zero = ir.ConstInt(0, ir.DataType.INT64, span)
    shape = ir.MakeTuple([zero], span)
    args_by_op = {
        "ptr.make_tensor": [ptr, shape, shape],
        "ptr.make_ptr": [ptr],
        "struct.create": [],
        "struct.set": [],
    }
    program = _program_with_call(op_name, args_by_op[op_name], [ptr])
    # The Aicpu CoreMask admits the surface ops, and the shared Cube/Vector
    # masks keep admitting them (the Aicpu bit never narrows the AI cores).
    ir.Pass.validate_target_ops(ir.SectionKind.Aicpu)(program)
    ir.Pass.validate_target_ops(ir.SectionKind.Cube)(program)
    ir.Pass.validate_target_ops(ir.SectionKind.Vector)(program)


def test_validate_target_ops_rejects_addptr_on_aicpu():
    # Outside the aicpu surface: the call falls to the unclassified default
    # (CV), so the Aicpu target rejects it while the AI cores keep it.
    span = ir.Span.unknown()
    ptr = ir.Var("ptr", ir.PtrType(ir.DataType.INT32), span)
    zero = ir.ConstInt(0, ir.DataType.INT64, span)
    program = _program_with_call("ptr.addptr", [ptr, zero], [ptr])
    with pytest.raises(
        ValueError,
        match="This operation cannot execute on AICPU. Supported cores for this call: Cube and Vector.",
    ):
        ir.Pass.validate_target_ops(ir.SectionKind.Aicpu)(program)
    ir.Pass.validate_target_ops(ir.SectionKind.Cube)(program)
    ir.Pass.validate_target_ops(ir.SectionKind.Vector)(program)


def test_validate_target_ops_rejects_dynamic_input_on_aicpu():
    span = ir.Span.unknown()
    ptr = ir.Var("ptr", ir.PtrType(ir.DataType.INT32), span)
    zero = ir.ConstInt(0, ir.DataType.INT64, span)
    for op_name, args in (
        ("dynamic_input_ptr", [ptr, zero]),
        ("dynamic_input_dim", [ptr, zero, zero]),
    ):
        program = _program_with_call(op_name, args, [ptr])
        with pytest.raises(
            ValueError,
            match="This operation cannot execute on AICPU. Supported cores for this call: Cube and Vector.",
        ):
            ir.Pass.validate_target_ops(ir.SectionKind.Aicpu)(program)
        ir.Pass.validate_target_ops(ir.SectionKind.Cube)(program)
        ir.Pass.validate_target_ops(ir.SectionKind.Vector)(program)


def test_validate_target_ops_rejects_foreign_target():
    with pytest.raises(ValueError, match="ValidateTargetOps requires a Cube, Vector or Aicpu target"):
        ir.Pass.validate_target_ops(ir.SectionKind.VF)


def test_validate_target_ops_rejects_mat_tile_getval():
    # Mat-memory tile scalar access falls to the unclassified CV default:
    # the AI cores keep it, the Aicpu target rejects it.
    from pypto_pro.ir.op.block_ops import make_mutex_tile_expr

    span = ir.Span.unknown()
    tile_expr = make_mutex_tile_expr([16, 16], ir.DataType.FP32, ir.MemorySpace.Mat, addr=0, mutex_ids=[0])
    tile = ir.Var("tile", tile_expr.type, span)
    zero = ir.ConstInt(0, ir.DataType.INT64, span)
    program = _program_with_call("block.getval", [tile, zero])
    ir.Pass.validate_target_ops(ir.SectionKind.Cube)(program)
    ir.Pass.validate_target_ops(ir.SectionKind.Vector)(program)
    with pytest.raises(
        ValueError,
        match="This operation cannot execute on AICPU. Supported cores for this call: Cube and Vector.",
    ):
        ir.Pass.validate_target_ops(ir.SectionKind.Aicpu)(program)


def test_validate_target_ops_rejects_tile_backed_getval():
    from pypto_pro.ir.op.block_ops import make_mutex_tile_expr

    span = ir.Span.unknown()
    tile_expr = make_mutex_tile_expr([16, 16], ir.DataType.FP32, ir.MemorySpace.Vec, addr=0, mutex_ids=[0])
    tile = ir.Var("tile", tile_expr.type, span)
    zero = ir.ConstInt(0, ir.DataType.INT64, span)
    program = _program_with_call("block.getval", [tile, zero])
    with pytest.raises(
        ValueError,
        match="This operation cannot execute on AICPU. Supported cores for this call: Vector.",
    ):
        ir.Pass.validate_target_ops(ir.SectionKind.Aicpu)(program)


# ---------------------------------------------------------------------------
# The shared routing table + validation gate (kernel.py)
# ---------------------------------------------------------------------------


def test_targets_for_func_type_routes_every_entry_type():
    from pypto_pro.runtime.kernel import targets_for_func_type

    assert targets_for_func_type(ir.FunctionType.AicpuKernel) == (ir.SectionKind.Aicpu,)
    assert targets_for_func_type(ir.FunctionType.SimtKernel) == (ir.SectionKind.Vector,)
    assert targets_for_func_type(ir.FunctionType.Opaque) == (ir.SectionKind.Cube, ir.SectionKind.Vector)


def test_validate_target_ops_or_raise_maps_the_cpp_error():
    from pypto_pro.runtime.kernel import validate_target_ops_or_raise

    program = _program_with_call("system.bar_m", [])
    with pytest.raises(InvalidOperation, match="cannot execute on AICPU"):
        validate_target_ops_or_raise(ir.SectionKind.Aicpu, program)


def test_aicpu_chain_rejects_non_aicpu_kernel(monkeypatch, tmp_path):
    # The routing table owns the target: a plain SIMD kernel reaching the AICPU
    # chain is a caller bug and is rejected before any parse or build.
    from pypto_pro._errors import InvalidVal

    builds = []
    monkeypatch.setattr(aicpu_compile, "build_aicpu_jit_so", lambda *a, **k: builds.append(a))
    monkeypatch.setenv("ASCEND_WORK_PATH", str(tmp_path))

    @pl.jit(auto_mutex=False)
    def plain_kernel(x: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32]):
        pl.setval(x, 0, 0)

    with pytest.raises(InvalidVal, match="the AICPU chain requires @pl.jit\\(mode='aicpu_kernel'\\)"):
        aicpu_compile.compile_aicpu_variant(plain_kernel, None, (None, None, None, None), None, ())
    assert builds == []
