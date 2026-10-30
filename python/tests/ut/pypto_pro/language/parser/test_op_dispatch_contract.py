# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Dispatcher contract UTs, runnable without the native IR extension.

The IR objects below are test doubles. The registry, dispatcher, argument gate
and span tracker are loaded from their production files. These tests cover the
Python boundary, not native IR construction, lowering or device execution.
"""

from __future__ import annotations

import ast
import contextlib
import enum
import functools
import importlib.util
import inspect
from pathlib import Path
import struct
import sys
import types

import pytest

_SOURCE = Path(__file__).resolve().parents[5] / "pypto_pro"


class _DType(enum.Enum):
    INT32 = "int32"
    INT64 = "int64"
    UINT32 = "uint32"
    UINT64 = "uint64"
    FP32 = "fp32"
    FP64 = "fp64"
    BOOL = "bool"

    def is_int(self):
        return self in (self.INT32, self.INT64, self.UINT32, self.UINT64)

    def is_float(self):
        return self in (self.FP32, self.FP64)


class _Span:
    def __init__(self, filename="test.py", begin_line=1, begin_column=1, end_line=1, end_column=1):
        self.filename, self.begin_line, self.begin_column = filename, begin_line, begin_column
        self.end_line, self.end_column = end_line, end_column

    @staticmethod
    def unknown():
        return _Span()


class _ScalarType:
    def __init__(self, dtype):
        self.dtype = dtype


class _Expr:
    def __init__(self, value_type=None, span=None):
        self.type, self.span = value_type, span


class _Const(_Expr):
    def __init__(self, value, dtype, span):
        super().__init__(_ScalarType(dtype), span)
        self.value = value


class _ConstInt(_Const):
    def __init__(self, value, dtype=_DType.INT64, span=None):
        super().__init__(value, dtype, span or _Span.unknown())


class _ConstFloat(_Const):
    def __init__(self, value, dtype=_DType.FP32, span=None):
        super().__init__(value, dtype, span or _Span.unknown())


class _ConstBool(_Const):
    def __init__(self, value, span=None):
        super().__init__(value, _DType.BOOL, span or _Span.unknown())


def _make_constant(value, dtype, span):
    if dtype == _DType.BOOL:
        return _ConstBool(bool(value), span)
    if dtype.is_float():
        return _ConstFloat(float(value), dtype, span)
    return _ConstInt(int(value), dtype, span)


class _TupleType:
    pass


class _NoneType:
    pass


class _MakeTuple(_Expr):
    def __init__(self, elements, span):
        super().__init__(_TupleType(), span)
        self.elements = elements


class _TileType:
    def __init__(self, dtype):
        self.dtype = dtype


class _DispatchError(Exception):
    def __init__(self, message, *, span=None, hint=None, parser_retry=False):
        super().__init__(message)
        self.span, self.hint, self.parser_retry = span, hint, parser_retry


def _constant_value(value):
    if isinstance(getattr(value, "type", None), _NoneType):
        return True, None
    if isinstance(value, _Const):
        return True, value.value
    if isinstance(value, _MakeTuple):
        converted = [_constant_value(element) for element in value.elements]
        return (True, tuple(v for _, v in converted)) if all(known for known, _ in converted) else (False, None)
    return (False, None) if isinstance(value, (_Expr, ast.AST)) else (True, value)


@pytest.fixture
def runtime(monkeypatch):
    def module(name, **attributes):
        result = types.ModuleType(name)
        result.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, result)
        return result

    def load(name, relative):
        spec = importlib.util.spec_from_file_location(name, _SOURCE / relative)
        result = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, result)
        spec.loader.exec_module(result)
        return result

    ir = types.SimpleNamespace(
        Expr=_Expr, BinaryExpr=type("BinaryExpr", (_Expr,), {}), ScalarType=_ScalarType,
        ConstInt=_ConstInt, ConstFloat=_ConstFloat, ConstBool=_ConstBool, MakeTuple=_MakeTuple,
        TileType=_TileType, TensorType=type("TensorType", (_TileType,), {}),
        TupleType=_TupleType, PtrType=type("PtrType", (), {}), NoneType=_NoneType,
        Span=_Span, DataType=_DType, MemorySpace=types.SimpleNamespace(Vec="vec"),
        FunctionType=types.SimpleNamespace(SimtVF="simt", SimtCallee="simt_callee", SimtKernel="simt_kernel"),
        SectionKind=types.SimpleNamespace(Cube="cube", Vector="vector"),
    )
    for name in (
        "pypto", "pypto_pro", "pypto_pro.ir", "pypto_pro.ir.op",
        "pypto_pro.language", "pypto_pro.language.parser",
    ):
        module(name, __path__=[])
    module("pypto.pypto_impl", ir=ir)
    module("pypto.pypto_impl.ir", **ir.__dict__, Call=_Expr, Function=_Expr)
    ir.RoundMode = enum.Enum("RoundMode", ["CAST_NONE"])
    module(
        "pypto_pro.ir._utils", _get_span_or_capture=lambda span: span or _Span(),
        _normalize_expr=lambda value, span: value,
    )
    error_names = (
        "CommonExternal", "InvalidArgument", "InvalidOperation", "InvalidShape",
        "InvalidType", "InvalidVal", "NotSupported",
    )
    errors = module("pypto_pro._errors", **{name: type(name, (_DispatchError,), {}) for name in error_names})
    errors.error_class_of_spec_message = lambda message: None
    ambient = []

    @contextlib.contextmanager
    def current_span(span, arguments=None):
        ambient.append((span, arguments))
        try:
            yield
        finally:
            ambient.pop()

    errors.current_span = current_span
    errors.argument_span = lambda name, fallback=None: (
        (ambient[-1][1] or {}).get(name, fallback) if ambient else fallback
    )
    api = module("pypto_pro.language._api")
    simt = types.SimpleNamespace()
    module("pypto_pro.language._simt_api", Simt=simt)
    module("pypto_pro.language._system_api", System=types.SimpleNamespace())
    vf = types.SimpleNamespace()
    module("pypto_pro.language._vf_api", Vf=vf)
    module(
        "pypto_pro.language.parser._expr_evaluator",
        ExprEvaluator=types.SimpleNamespace(ir_to_python_value=_constant_value),
    )
    module(
        "pypto_pro.language.parser.diagnostics",
        check_fits_dtype=lambda *args, **kwargs: None,
        make_const_int=lambda value, dtype=None, *, span, **kwargs: _make_constant(value, dtype or _DType.INT64, span),
    )
    registry = load("pypto_pro.ir.op._op_registry", "ir/op/_op_registry.py")
    checker = load("pypto_pro.language.parser._argument_checker", "language/parser/_argument_checker.py")
    tracker = load("pypto_pro.language.parser._span_tracker", "language/parser/_span_tracker.py")

    def declare(name, function):
        @functools.wraps(function)
        def declaration(*args, **kwargs):
            raise AssertionError("DSL declaration must not be executed")
        declaration.bind_declared = inspect.signature(function).bind
        owner, _, short = name.rpartition(".")
        setattr({"simt": simt, "vf": vf}.get(owner, api), short, declaration)

    # Reuse the real frontend helpers without importing the native runtime.
    helper_names = {
        "_dispatch_ast_op",
        "_dispatch_op", "_check_declared_params", "_check_declared_types", "_validate_op_scope",
        "require_const_value", "_parse_vf_op", "parse_op_call",
    }
    call_tree = ast.parse((_SOURCE / "language/parser/_call_parser.py").read_text(encoding="utf-8"))
    mixin = next(node for node in call_tree.body if isinstance(node, ast.ClassDef) and node.name == "CallParserMixin")
    helper_class = ast.ClassDef(
        name="ParserHelpers", bases=[], keywords=[], decorator_list=[],
        body=[node for node in mixin.body if isinstance(node, ast.FunctionDef) and node.name in helper_names],
    )
    expression_tree = ast.parse((_SOURCE / "language/parser/_expression_parser.py").read_text(encoding="utf-8"))
    expression_mixin = next(node for node in expression_tree.body if isinstance(node, ast.ClassDef)
                            and node.name == "ExpressionParserMixin")
    helper_class.body.extend(
        node for node in expression_mixin.body
        if isinstance(node, ast.FunctionDef) and node.name in ("parse_list", "parse_tuple_literal")
    )
    helper_module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), helper_class,
    ], type_ignores=[])
    namespace = {
        "ast": ast, "ir": ir, "inspect": inspect, "_argument_checker": checker,
        "_OP_REGISTRY": registry._OP_REGISTRY, "_AST_OP_REGISTRY": registry._AST_OP_REGISTRY,
        "ExprEvaluator": types.SimpleNamespace(ir_to_python_value=_constant_value), **errors.__dict__,
    }
    exec(compile(ast.fix_missing_locations(helper_module), "_call_parser_helpers", "exec"), namespace)

    class Parser(namespace["ParserHelpers"]):
        _VF_OP_DST_COUNT = {}
        _auto_mutex = False

        def __init__(self, source, values=None):
            self.span_tracker = tracker.SpanTracker("test.py", [source])
            self.call = ast.parse(source, mode="eval").body
            self.values = values or {}
            self.evaluated = []
            self._current_func_type = "simt"
            self.folded = None
            self.inline_vf_depth = 0
            self.target = ir.SectionKind.Cube
            self.expr_evaluator = types.SimpleNamespace(
                ir_to_python_value=_constant_value, try_eval_expr=lambda node: (False, None),
            )

        def parse_expression(self, node, *, nested=True):
            self.evaluated.append(ast.unparse(node))
            if isinstance(node, ast.Name):
                return self.values[node.id]
            if isinstance(node, ast.Constant):
                dtype = {int: _DType.INT64, float: _DType.FP32, bool: _DType.BOOL}.get(type(node.value))
                if dtype is not None:
                    return _make_constant(node.value, dtype, self.span_tracker.get_span(node))
                return _Expr(_NoneType()) if node.value is None else node.value
            if isinstance(node, ast.List):
                return self.parse_list(node)
            if isinstance(node, ast.Tuple):
                return self.parse_tuple_literal(node)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
                value = self.parse_expression(node.operand)
                return _make_constant(-value.value if isinstance(node.op, ast.USub) else value.value,
                              value.type.dtype, self.span_tracker.get_span(node))
            if isinstance(node, ast.Subscript):
                raise AssertionError("atomic reference must not emit a scalar load")
            raise AssertionError(ast.dump(node))

        def _make_scalar_constant(self, value, dtype, span):
            return _make_constant(value, dtype, span)

        def _parse_scalar_subscript_index(self, container, node, span):
            return self.parse_expression(node)

        def _fold_const_binop(self, op_name, result, span):
            return self.folded

        def run(self, name):
            if not name.startswith("simt.") and name in registry._AST_OP_REGISTRY:
                self._current_func_type = "block"
            if name.startswith("simt.") or name in registry._AST_OP_REGISTRY:
                return self.parse_op_call(self.call)
            return self._dispatch_op(name, self.call, registry._OP_REGISTRY[name])

        def _route_ir_node_method(self, call):
            return None

        def _extract_op_name(self, node):
            return ast.unparse(node).removeprefix("pl.")

    def simt_module():
        name = "pypto_pro.ir.op.simt_ops"
        # A cached module may still refer to the native IR classes from earlier tests.
        return load(name, "ir/op/simt_ops.py")

    def load_functions(relative, names, **extra):
        """Run production helpers whose containing module requires the native IR."""
        tree = ast.parse((_SOURCE / relative).read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        assert {node.name for node in functions} == set(names)
        for node in functions:
            node.decorator_list = []
        module_tree = ast.Module(body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions,
        ], type_ignores=[])
        function_globals = {
            "ast": ast, "ir": ir, "_ir_core": ir,
            **errors.__dict__, **extra,
        }
        exec(compile(ast.fix_missing_locations(module_tree), relative, "exec"), function_globals)
        return types.SimpleNamespace(**{name: function_globals[name] for name in names})

    tuple_helpers = load_functions(
        "ir/_utils.py", ("_normalize_expr", "_to_make_tuple"), _ir=ir, DataType=_DType,
    )
    namespace["_to_make_tuple"] = tuple_helpers._to_make_tuple

    return types.SimpleNamespace(
        registry=registry, Parser=Parser, errors=errors, ir=ir,
        declare=declare, ambient=ambient, load=load, simt_module=simt_module, load_functions=load_functions,
    )


def test_builder_receives_ir_for_operands_and_static_keyword_attributes(runtime):
    def declaration(lhs: "int", rhs: "int", /, *, axis: "int" = 0):
        pass
    runtime.declare("example", declaration)

    @runtime.registry.op_impl("example")
    def build(self, lhs, rhs, *, axis=0, span):
        assert isinstance(lhs, _Expr) and isinstance(rhs, _Expr)
        assert isinstance(axis, _Expr) and axis.value == 2
        assert runtime.ambient[-1][0] is span
        return lhs, rhs, axis

    assert runtime.registry._OP_REGISTRY["example"] is build
    x = _Expr(_ScalarType(_DType.INT64))
    parser = runtime.Parser("pl.example(x, 3, axis=2)", {"x": x})
    lhs, rhs, axis = parser.run("example")
    assert lhs is x and rhs.value == 3 and axis.value == 2
    assert parser.evaluated == ["x", "3", "2"]
    assert runtime.ambient == []


@pytest.mark.parametrize("source,positional,keywords,evaluated", [
    ("pl.example(rhs=3, lhs=2)", (), ("rhs", "lhs"), ["3", "2"]),
    ("pl.example(2, rhs=3)", (2,), ("rhs",), ["2", "3"]),
    ("pl.example(2, 3)", (2, 3), (), ["2", "3"]),
])
def test_operands_keep_their_call_form_and_source_order(runtime, source, positional, keywords, evaluated):
    def declaration(lhs: "int", rhs: "int"):
        pass
    runtime.declare("example", declaration)

    @runtime.registry.op_impl("example")
    def build(self, *args, span, **kwargs):
        assert all(isinstance(value, _Expr) for value in (*args, *kwargs.values()))
        return args, kwargs

    parser = runtime.Parser(source)
    args, kwargs = parser.run("example")
    assert tuple(value.value for value in args) == positional
    assert tuple(kwargs) == keywords
    expected_kwargs = {name: {"lhs": 2, "rhs": 3}[name] for name in keywords}
    assert {name: value.value for name, value in kwargs.items()} == expected_kwargs
    assert parser.evaluated == evaluated


@pytest.mark.parametrize("source", ["pl.example(flag=False)", "pl.example(flag=True)"])
def test_boolean_keyword_is_ir_until_its_builder_requires_a_constant(runtime, source):
    @runtime.registry.op_impl("example")
    def build(self, *, flag, span):
        assert isinstance(flag, _Expr) and flag.type.dtype is _DType.BOOL
        return self.require_const_value(
            flag, "flag", expects="bool", check=lambda value: type(value) is bool, hint="Use a constant",
        )

    assert runtime.Parser(source).run("example") is ("True" in source)


def test_parameter_name_does_not_make_an_integer_an_enum(runtime):
    def declaration(*, mode: "int"):
        pass
    runtime.declare("example", declaration)

    @runtime.registry.op_impl("example")
    def build(self, *, mode, span):
        return mode

    assert runtime.Parser("pl.example(mode=1)").run("example").value == 1


def test_declared_enum_rejects_an_integer_without_a_parameter_name_table(runtime):
    def declaration(*, selection):
        pass
    declaration.__annotations__["selection"] = "RoundMode"
    runtime.declare("example", declaration)

    @runtime.registry.op_impl("example")
    def build(self, *, selection, span):
        return selection

    source = "pl.example(selection=1)"
    with pytest.raises(runtime.errors.InvalidType, match="selection expects") as error:
        runtime.Parser(source).run("example")
    assert error.value.span.begin_column == source.index("1") + 1
    enum_value = runtime.ir.RoundMode.CAST_NONE
    assert runtime.Parser("pl.example(selection=choice)", {"choice": enum_value}).run("example") is enum_value


def test_order_builder_extracts_integer_list_and_rejects_runtime_axes(runtime):
    helpers = runtime.load_functions(
        "ir/op/block_ops.py", ("_prepare_order",), _is_int=lambda value: type(value) is int,
    )

    @runtime.registry.op_impl("example")
    def build(self, *, span, **kwargs):
        helpers._prepare_order(self, kwargs)
        return kwargs.get("order")

    assert runtime.Parser("pl.example()").run("example") is None
    assert runtime.Parser("pl.example(order=[1, 0])").run("example") == [1, 0]
    source = "pl.example(order=[1, axis])"
    with pytest.raises(runtime.errors.InvalidType, match="compile-time integer list") as error:
        runtime.Parser(source, {"axis": _Expr(_ScalarType(_DType.INT64))}).run("example")
    assert error.value.span.begin_column == source.index("[") + 1


@pytest.mark.parametrize("sequence", ['["auto", 3]', '("auto", 3)'])
def test_sequence_rejects_strings_before_native_tuple_construction(runtime, sequence):
    @runtime.registry.op_impl("example")
    def build(self, value, *, span):
        raise AssertionError("must not reach builder")

    with pytest.raises(runtime.errors.InvalidType, match="Cannot convert .*str.* to IR expression"):
        runtime.Parser(f"pl.example({sequence})").run("example")


@pytest.mark.parametrize("dim, op", [(0, "block.row_sum"), (1, "block.col_sum")])
def test_dim_builder_selects_instruction_from_an_ir_constant(runtime, dim, op):
    runtime.ir.create_op_call = lambda name, args, attrs, span: name
    integer_helpers = runtime.load_functions(
        "ir/_utils.py", ("_const_int_value", "require_const_int"), _ir=runtime.ir,
    )
    helpers = runtime.load_functions(
        "ir/op/block_ops.py", ("_create_dim_op",), _DIM_REDUCE_DTYPES={}, block_ir_op=lambda name: f"block.{name}",
        require_const_int=integer_helpers.require_const_int,
    )
    source = _Expr(_ScalarType(_DType.INT32))

    @runtime.registry.op_impl("example")
    def build(self, *, dim=0, span):
        return helpers._create_dim_op([source, source], row_op="row_sum", col_op="col_sum", dim=dim, span=span)

    assert runtime.Parser(f"pl.example(dim={dim})").run("example") == op
    assert runtime.Parser("pl.example()").run("example") == "block.row_sum"
    with pytest.raises(runtime.errors.InvalidType, match="compile-time integer"):
        runtime.Parser("pl.example(dim=n)", {"n": source}).run("example")
    for value in ("True", "1.5"):
        with pytest.raises(runtime.errors.InvalidType, match="compile-time integer"):
            runtime.Parser(f"pl.example(dim={value})").run("example")
    for value, bound in ((-1, "at least 0"), (2, "at most 1")):
        with pytest.raises(runtime.errors.InvalidVal, match=bound):
            runtime.Parser(f"pl.example(dim={value})").run("example")


@pytest.mark.parametrize("enabled", [False, True])
def test_layout_transform_extracts_boolean_instead_of_testing_ir_object_truthiness(runtime, enabled):
    helpers = runtime.load_functions(
        "ir/op/system_ops.py", ("_build_system_set_mm_layout_transform",),
        set_mm_layout_transform=lambda *, enabled, span: enabled,
    )
    runtime.registry.op_impl("example")(helpers._build_system_set_mm_layout_transform)
    assert runtime.Parser(f"pl.example(enabled={enabled})").run("example") is enabled


@pytest.mark.parametrize("scale", [0, 1, 0.5])
def test_literal_scale_keeps_float_bit_encoding_and_runtime_bits_stay_ir(runtime, scale):
    helpers = runtime.load_functions(
        "ir/op/block_ops.py", ("_prepare_scale", "_resolve_scale_param", "_encode_deq_scalar"),
        struct=struct, Expr=_Expr, ConstInt=_ConstInt, DataType=_DType, _validate_scaling_tile=lambda value: None,
    )

    @runtime.registry.op_impl("example")
    def build(self, *, scale, span):
        kwargs = {"scale": scale}
        helpers._prepare_scale(self, kwargs)
        return helpers._resolve_scale_param(kwargs["scale"], span)

    expected = struct.unpack("!I", struct.pack("!f", scale))[0]
    encoded, tile = runtime.Parser(f"pl.example(scale={scale})").run("example")
    assert tile is None and isinstance(encoded, _ConstInt)
    assert encoded.type.dtype == _DType.UINT64 and encoded.value == expected
    bits = _Expr(_ScalarType(_DType.INT32))
    assert runtime.Parser("pl.example(scale=bits)", {"bits": bits}).run("example") == (bits, None)
    assert runtime.Parser("pl.example(scale=None)").run("example") == (None, None)
    with pytest.raises(runtime.errors.InvalidType, match="compile-time number"):
        runtime.Parser("pl.example(scale=False)").run("example")


@pytest.mark.parametrize("operation", ["set_cross_core", "wait_cross_core"])
@pytest.mark.parametrize("pipe", ["V", "S"])
def test_cross_core_constant_and_runtime_ids_keep_their_instruction_forms(runtime, operation, pipe):
    runtime.ir.create_op_call = lambda name, args, attrs, span: (name, args, attrs)
    integer_helpers = runtime.load_functions(
        "ir/_utils.py", ("_const_int_value", "require_const_int", "require_int"), _ir=runtime.ir,
    )
    helpers = runtime.load_functions(
        "ir/op/system_ops.py", (operation, f"_build_{operation}", "_validate_integer_id"),
        Expr=_Expr, MAX_EVENT_ID=15, require_int=integer_helpers.require_int,
        PipeType=types.SimpleNamespace(S="S"),
        CrossCoreSyncMode=types.SimpleNamespace(INTRA_BLOCK=0, UNICAST_BLOCK=1),
        _get_span_or_capture=lambda span: span, _validate_concrete_pipe=lambda *args: None,
        _check_id_range=lambda *args, **kwargs: None,
    )
    runtime.registry.op_impl("example")(getattr(helpers, f"_build_{operation}"))
    constant = runtime.Parser("pl.example(pipe=pipe, event_id=7)", {"pipe": pipe}).run("example")
    assert constant == (f"system.{operation}", [], {"pipe": pipe, "event_id": 7, "sync_mode": 0})
    event_id = _Expr(_ScalarType(_DType.INT32))
    dynamic = runtime.Parser(
        "pl.example(pipe=pipe, event_id=event)", {"pipe": pipe, "event": event_id},
    ).run("example")
    assert dynamic[0] == f"system.{operation}_dyn" and dynamic[1] == [event_id]


def test_invalid_signature_rejects_before_evaluating_any_argument(runtime):
    def declaration(value: "int", /):
        pass
    runtime.declare("example", declaration)

    @runtime.registry.op_impl("example")
    def build(self, value, *, span):
        raise AssertionError("must not reach builder")

    parser = runtime.Parser("pl.example(undefined, extra=side_effect())")
    with pytest.raises(runtime.errors.InvalidArgument, match="unexpected keyword"):
        parser.run("example")
    assert parser.evaluated == []


def test_value_type_gate_runs_before_the_builder(runtime):
    def declaration(value: "int", /):
        pass
    runtime.declare("example", declaration)

    @runtime.registry.op_impl("example")
    def build(self, value, *, span):
        raise AssertionError("must not reach builder")

    parser = runtime.Parser("pl.example(tile)", {"tile": _Expr(_TileType(_DType.FP32))})
    with pytest.raises(runtime.errors.InvalidType, match="expects.*int") as error:
        parser.run("example")
    assert error.value.span.begin_column == 12


def test_specific_builder_type_checks_can_supply_their_own_diagnostics(runtime):
    def declaration(value: "int", /):
        pass
    runtime.declare("example", declaration)

    @runtime.registry.op_impl("example")
    def build(self, value, *, span):
        return self.require_const_value(
            value, 0, expects="integer", check=lambda v: type(v) is int, hint="Use a literal",
        )

    parser = runtime.Parser("pl.example(value)", {"value": _Expr(_ScalarType(_DType.INT64))})
    with pytest.raises(runtime.errors.InvalidType, match="must be a compile-time integer"):
        parser.run("example")


@pytest.mark.parametrize("source,key", [("pl.example(value)", 0), ("pl.example(value=value)", "value")])
def test_const_diagnostic_prefers_the_call_site_over_the_variable_definition(runtime, source, key):
    @runtime.registry.op_impl("example")
    def build(self, value, *, span):
        return self.require_const_value(value, key, expects="integer", hint="Use a constant")

    definition_span = _Span(filename="definition.py", begin_line=7, end_line=7)
    value = _Expr(_ScalarType(_DType.INT64), definition_span)
    parser = runtime.Parser(source, {"value": value})
    with pytest.raises(runtime.errors.InvalidType, match="got 'value'") as error:
        parser.run("example")
    assert error.value.span.filename == "test.py"
    assert error.value.span.begin_line == 1
    assert error.value.span.begin_column == source.rindex("value") + 1
    assert value.span is definition_span


def test_shape_element_diagnostic_keeps_its_source_location(runtime):
    @runtime.registry.op_impl("example")
    def build(self, *, shape, span):
        return self.require_const_value(shape.elements[1], "shape[1]", expects="integer", hint="Use a constant")

    value = _Expr(_ScalarType(_DType.INT64), _Span(begin_column=22, end_column=22))
    parser = runtime.Parser("pl.example(shape=[1, x])", {"x": value})
    with pytest.raises(runtime.errors.InvalidType, match=r"'shape\[1\]' must be a compile-time integer") as error:
        parser.run("example")
    assert error.value.span.begin_column == 22


def test_builder_uses_ir_source_and_span_without_inspecting_its_signature(runtime, monkeypatch):
    @runtime.registry.op_impl("example")
    def build(self, value, *, span):
        assert self is parser
        assert self.span_tracker.get_source_text(value.span) == "x"
        assert value.span.begin_column == 12
        assert not isinstance(value, ast.AST)
        return value

    value = _Expr(_ScalarType(_DType.INT64), _Span(begin_column=12, end_column=12))
    parser = runtime.Parser("pl.example(x)", {"x": value})
    monkeypatch.setattr(inspect, "signature", lambda *args, **kwargs: pytest.fail("Dispatcher must always pass self"))
    assert parser.run("example") is parser.values["x"]


def test_assert_condition_text_uses_the_call_site_for_a_preexisting_ir_value(runtime):
    source = "pl.pto_assert(flag)"
    def declaration(condition: "bool", /):
        pass
    runtime.declare("pto_assert", declaration)
    helpers = runtime.load_functions(
        "ir/op/debug_ops.py", ("_build_pto_assert",), pto_assert=lambda *args, **kwargs: kwargs["condition_text"],
    )
    runtime.registry.op_impl("pto_assert")(helpers._build_pto_assert)
    definition_span = _Span(filename="definition.py", begin_line=7, end_line=7)
    value = _Expr(_ScalarType(_DType.BOOL), definition_span)
    parser = runtime.Parser(source, {"flag": value})
    assert parser.run("pto_assert") == "flag"
    assert value.span is definition_span


def test_vf_attribute_conversion_consumes_ir_values(runtime):
    def declaration(*, flag: "bool"):
        pass
    runtime.declare("vf.example", declaration)
    runtime.ir.create_op_call = lambda name, args, attrs, span: attrs
    parser = runtime.Parser("vf.example(flag=False)")
    assert parser._parse_vf_op("example", parser.call) == {"flag": False}


def test_vf_runtime_attribute_rejection_keeps_ir_location(runtime):
    def declaration(*, flag: "bool"):
        pass
    runtime.declare("vf.example", declaration)
    source = "vf.example(flag=runtime_flag)"
    column = source.index("runtime_flag") + 1
    value = _Expr(_ScalarType(_DType.BOOL), _Span(begin_column=column, end_column=column + len("runtime_flag") - 1))
    parser = runtime.Parser(source, {"runtime_flag": value})
    with pytest.raises(runtime.errors.InvalidType, match="compile-time attribute") as error:
        parser._parse_vf_op("example", parser.call)
    assert error.value.span.begin_column == source.index("runtime_flag") + 1


def test_duplicate_registration_is_rejected(runtime):
    def build(self, span, /):
        pass
    runtime.registry.op_impl("example")(build)
    with pytest.raises(runtime.errors.InvalidOperation, match="Duplicate"):
        runtime.registry.op_impl("example")(build)


@pytest.mark.parametrize("source", ["pl.example(*xs)", "pl.example(**xs)"])
def test_unpacking_does_not_evaluate(runtime, source):
    @runtime.registry.op_impl("example")
    def build(self, *args, span, **kwargs):
        raise AssertionError("must not reach builder")
    parser = runtime.Parser(source)
    with pytest.raises(runtime.errors.InvalidArgument):
        parser.run("example")
    assert parser.evaluated == []


@pytest.mark.parametrize("source", [
    "pl.const(value, dtype)", "pl.const(1 + 2, dtype)", "pl.const(+2, dtype)",
])
def test_legacy_const_rejects_nonliteral_values_before_evaluation(runtime, source):
    def declaration(value, dtype, /):
        pass
    runtime.declare("const", declaration)
    helpers = runtime.load_functions("ir/op/scalar_ops.py", ("_parse_typed_constant",))
    runtime.registry.op_impl_ast("const")(helpers._parse_typed_constant)
    parser = runtime.Parser(source, {"dtype": _DType.INT32})
    with pytest.raises(runtime.errors.InvalidType, match="numeric literal"):
        parser.run("const")
    assert parser.evaluated == []


@pytest.mark.parametrize(("source", "expected", "dtype"), [
    ("pl.const(-2, dtype)", -2, _DType.INT32),
    ("pl.const(True, dtype)", 1, _DType.INT32),
    ("pl.const(-True, dtype)", -1, _DType.INT32),
    ("pl.const(True, dtype)", True, _DType.BOOL),
    ("pl.const(0.5, dtype)", 0.5, _DType.FP32),
])
def test_legacy_const_preserves_source_literal_forms(runtime, source, expected, dtype):
    def declaration(value, dtype, /):
        pass
    runtime.declare("const", declaration)
    helpers = runtime.load_functions("ir/op/scalar_ops.py", ("_parse_typed_constant",))
    runtime.registry.op_impl_ast("const")(helpers._parse_typed_constant)
    parser = runtime.Parser(source, {"dtype": dtype})
    result = parser.run("const")
    assert result.value == expected and result.type.dtype == dtype
    expected_type = _ConstBool if dtype == _DType.BOOL else _ConstFloat if dtype.is_float() else _ConstInt
    assert type(result) is expected_type
    assert parser.evaluated == ["dtype"]


@pytest.mark.parametrize(("source", "op_name", "error_name", "message"), [
    ("pl.struct(name, value=undefined)", "struct", "InvalidVal", "string struct name"),
    ("pl.struct('A' + 'B', value=undefined)", "struct", "InvalidVal", "string struct name"),
    ("pl.struct_array(size, 'S', value=undefined)", "struct_array", "InvalidType", "integer size"),
    ("pl.struct_array(1 + 2, 'S', value=undefined)", "struct_array", "InvalidType", "integer size"),
    ("pl.struct_array(-1, 'S', value=undefined)", "struct_array", "InvalidType", "integer size"),
])
def test_legacy_struct_literals_reject_before_field_evaluation(runtime, source, op_name, error_name, message):
    runtime.load("pypto_pro.language.parser._struct_parser", "language/parser/_struct_parser.py")
    parser = runtime.Parser(source)
    with pytest.raises(getattr(runtime.errors, error_name), match=message):
        parser.run(op_name)
    assert parser.evaluated == []


def test_legacy_struct_fields_can_use_framework_parameter_names(runtime):
    structure = runtime.load("pypto_pro.language.parser._struct_parser", "language/parser/_struct_parser.py")

    class Parser(runtime.Parser, structure.StructParserMixin):
        pass

    parser = Parser("pl.struct('S', span=1, self=2)")
    parser.named_fields = lambda value: None
    parser._make_struct_create = lambda values, name, names, span: (name, dict(zip(names, values)))
    name, fields = parser.run("struct")
    assert name == "S" and {name: value.value for name, value in fields.items()} == {"span": 1, "self": 2}


@pytest.mark.parametrize("source", [
    "pl.simt.atomic_example(tile[0], 1)", "pl.simt.atomic_example(value=1, target=tile[0])",
])
def test_atomic_legacy_parser_preserves_target_and_contextual_literal_dtype(runtime, source):
    def declaration(target: "int", value: "int"):
        pass
    runtime.declare("simt.atomic_example", declaration)
    simt = runtime.simt_module()

    @runtime.registry.op_impl_ast("simt.atomic_example")
    def parse(self, call):
        return simt._parse_atomic_call(
            self, "simt.atomic_example", lambda *args, **kwargs: args, ("value",), call,
        )

    tile = _Expr(_TileType(_DType.INT32))
    parser = runtime.Parser(source, {"tile": tile})
    container, offset, value = parser.run("simt.atomic_example")
    assert container is tile and offset.value == 0
    assert value.value == 1 and value.type.dtype == _DType.INT32
    assert parser.evaluated == ["tile", "0"]


def test_atomic_operand_variable_retains_its_declared_dtype(runtime):
    def declaration(target, value):
        pass
    runtime.declare("simt.example", declaration)
    simt = runtime.simt_module()

    @runtime.registry.op_impl_ast("simt.example")
    def parse(self, call):
        return simt._parse_atomic_call(self, "simt.example", lambda *args, **kwargs: args[2], ("value",), call)

    value = _Expr(_ScalarType(_DType.INT64))
    parser = runtime.Parser("pl.simt.example(tile[0], value)", {"tile": _Expr(_TileType(_DType.INT32)), "value": value})
    assert parser.run("simt.example") is value


def test_atomic_integer_target_rejects_float_literal(runtime):
    def declaration(target, value):
        pass
    runtime.declare("simt.example", declaration)
    simt = runtime.simt_module()

    @runtime.registry.op_impl_ast("simt.example")
    def parse(self, call):
        return simt._parse_atomic_call(self, "simt.example", lambda *args, **kwargs: None, ("value",), call)

    parser = runtime.Parser("pl.simt.example(tile[0], 1.5)", {"tile": _Expr(_TileType(_DType.INT32))})
    with pytest.raises(runtime.errors.InvalidType, match="requires an integer value"):
        parser.run("simt.example")


def test_warp_legacy_parser_preserves_control_dtypes(runtime):
    runtime.ir.create_op_call = lambda name, args, attrs, span: (name, args)
    runtime.simt_module()
    value = _Expr(_ScalarType(_DType.INT32))
    parser = runtime.Parser("pl.simt.warp_shfl_up(value, delta=1)", {"value": value})
    name, operands = parser.run("simt.warp_shfl_up")
    assert name == "simt.warp_shfl_up" and operands[0] is value
    assert operands[1].value == 1 and operands[1].type.dtype == _DType.UINT32
    assert operands[2].value == 32 and operands[2].type.dtype == _DType.INT32
    assert parser.evaluated == ["value"]


@pytest.mark.parametrize("operation", ["min", "max"])
def test_production_scalar_min_max_retains_frontend_folding(runtime, operation):
    def declaration(lhs: "int", rhs: "int", /):
        pass
    runtime.declare(operation, declaration)
    result = runtime.ir.BinaryExpr()
    setattr(runtime.ir, f"{operation}_", lambda lhs, rhs, span: result)
    runtime.load("pypto_pro.ir.op.scalar_ops", "ir/op/scalar_ops.py")
    parser = runtime.Parser(f"pl.{operation}(2, 3)")
    parser.folded = _ConstInt(2 if operation == "min" else 3, _DType.INT64, _Span())
    assert parser.run(operation) is parser.folded
    parser.folded = None
    assert parser.run(operation) is result


def test_legacy_handler_receives_unparsed_call(runtime):
    @runtime.registry.op_impl_ast("example")
    def parse(self, call):
        assert call is self.call
        assert runtime.ambient[-1][0].begin_column == 1
        return "legacy"

    parser = runtime.Parser("pl.example(undefined, side_effect())")
    assert parser.run("example") == "legacy"
    assert parser.evaluated == []


@pytest.mark.parametrize("first", ["op_impl", "op_impl_ast"])
def test_duplicate_registration_is_rejected_across_both_registries(runtime, first):
    def build(*args):
        pass
    getattr(runtime.registry, first)("example")(build)
    for decorator in (runtime.registry.op_impl, runtime.registry.op_impl_ast):
        with pytest.raises(runtime.errors.InvalidOperation, match="Duplicate"):
            decorator("example")(build)


def test_production_scalar_min_receives_ir_values(runtime):
    def declaration(lhs: "int", rhs: "int", /):
        pass
    runtime.declare("min", declaration)
    runtime.ir.min_ = lambda lhs, rhs, span: (lhs, rhs, span)
    runtime.load("pypto_pro.ir.op.scalar_ops", "ir/op/scalar_ops.py")
    value = _Expr(_ScalarType(_DType.INT64))
    parser = runtime.Parser("pl.min(x, 3)", {"x": value})
    lhs, rhs, span = parser.run("min")
    assert lhs is value and isinstance(rhs, _Expr) and rhs.value == 3
    assert parser.evaluated == ["x", "3"]


def test_production_named_tuple_accepts_fields_named_like_framework_parameters(runtime):
    runtime.load("pypto_pro.language.parser._struct_parser", "language/parser/_struct_parser.py")
    parser = runtime.Parser("pl.make_tuple(span=1, context=2, self=3, _fields=4)")
    parser.make_named_tuple = lambda values, names, span: dict(zip(names, values))
    fields = parser.run("make_tuple")
    assert {name: value.value for name, value in fields.items()} == {"span": 1, "context": 2, "self": 3, "_fields": 4}


def test_production_atomic_builder_receives_reference_and_typed_value(runtime):
    def declaration(target: "int", value: "int"):
        pass
    runtime.declare("simt.atomic_add", declaration)
    runtime.ir.create_op_call = lambda name, args, attrs, span: (name, args, attrs)
    runtime.load("pypto_pro.ir.op.simt_ops", "ir/op/simt_ops.py")
    tile = _Expr(_TileType(_DType.INT32))
    parser = runtime.Parser("pl.simt.atomic_add(value=1, target=tile[0])", {"tile": tile})
    name, args, attrs = parser.run("simt.atomic_add")
    assert name == "simt.atomic_add" and args[0] is tile
    assert args[1].value == 0 and args[2].type.dtype == _DType.INT32
    assert parser.evaluated == ["tile", "0"]


def test_legacy_atomic_rejects_a_scalar_target_container(runtime):
    def declaration(target, value):
        pass
    runtime.declare("simt.atomic_add", declaration)
    runtime.simt_module()
    source = "pl.simt.atomic_add(scalar[0], 1)"
    parser = runtime.Parser(source, {"scalar": _Expr(_ScalarType(_DType.INT32))})
    with pytest.raises(runtime.errors.InvalidType, match="target container must be a Tile or Tensor") as error:
        parser.run("simt.atomic_add")
    assert error.value.span.begin_column == source.index("scalar[0]") + 1


@pytest.mark.parametrize("op_name", ["thread_idx", "syncthreads", "cast", "sqrt", "atomic_add", "warp_all", "launch"])
@pytest.mark.parametrize("function_type", ["ordinary_kernel", "inline_helper"])
def test_simt_scope_rejects_before_evaluating_operands(runtime, op_name, function_type):
    runtime.simt_module()
    parser = runtime.Parser(f"pl.simt.{op_name}(undefined)")
    parser._current_func_type = function_type

    with pytest.raises(runtime.errors.InvalidOperation, match="can only be used inside a SIMT function"):
        parser.run(f"simt.{op_name}")
    assert parser.evaluated == []


@pytest.mark.parametrize("function_type", ["simt", "simt_callee", "simt_kernel"])
def test_simt_function_rejects_non_simt_ops_before_evaluating_operands(runtime, function_type):
    parser = runtime.Parser("pl.add(undefined)")
    parser._current_func_type = function_type
    with pytest.raises(runtime.errors.NotSupported, match="is not supported inside a SIMT function"):
        parser.parse_op_call(parser.call)
    assert parser.evaluated == []


@pytest.mark.parametrize("function_type", ["simt", "simt_callee", "simt_kernel"])
def test_simt_scope_accepts_simt_function_kinds(runtime, function_type):
    runtime.simt_module()
    runtime.ir.create_op_call = lambda name, args, attrs, span: name
    parser = runtime.Parser("pl.simt.syncthreads()")
    parser._current_func_type = function_type
    assert parser.run("simt.syncthreads") == "simt.syncthreads"


def test_indexed_simt_launch_passes_scope_validation_in_an_ordinary_kernel(runtime):
    parser = runtime.Parser("entry[1](undefined)")
    parser._current_func_type = "ordinary_kernel"
    parser._validate_op_scope("simt.launch", parser.call)
    assert parser.evaluated == []


@pytest.mark.parametrize("op_name", ["simt.launch", "simt.syncthreads"])
def test_simt_operations_remain_forbidden_inside_inline_vf(runtime, op_name):
    parser = runtime.Parser("entry[1](undefined)")
    parser._current_func_type = "ordinary_kernel"
    parser.inline_vf_depth = 1
    parser._TILE_GROUP_ACCESSORS = frozenset()
    parser._VF_SCALAR_PL_OPS = frozenset()
    with pytest.raises(runtime.errors.NotSupported, match="is not supported inside @pl.vector_function"):
        parser._validate_op_scope(op_name, parser.call)
    assert parser.evaluated == []


@pytest.mark.parametrize("source,op_name,parameters", [
    ("pl.simt.abs(value=tile)", "abs", ("value",)),
    ("pl.simt.min(1, rhs=tile)", "min", ("lhs", "rhs")),
    ("pl.simt.min(rhs=tile, lhs=1)", "min", ("lhs", "rhs")),
])
def test_production_simt_scalar_diagnostic_uses_the_declaration(runtime, source, op_name, parameters):
    namespace = {}
    exec(f"def declaration({', '.join(parameters)}): pass", namespace)
    declaration = namespace["declaration"]
    declaration.__annotations__ = {name: "Union[int, float]" for name in parameters}
    runtime.declare(f"simt.{op_name}", declaration)
    runtime.simt_module()
    parser = runtime.Parser(source, {"tile": _Expr(_TileType(_DType.INT32))})

    with pytest.raises(runtime.errors.InvalidType) as error:
        parser.run(f"simt.{op_name}")
    name = "value" if op_name == "abs" else "rhs"
    assert f"{name} expects Union[int, float]" in str(error.value)
    assert error.value.span.begin_column == source.index("tile") + 1


@pytest.mark.parametrize("shape", ["[1, 32]", "(1, 32)"])
def test_tile_type_legacy_parser_parses_dimensions_without_a_tuple_anchor(runtime, shape):
    helpers = runtime.load_functions(
        "ir/op/block_ops.py", ("_parse_tile_type",),
        TileType=lambda **kwargs: kwargs, _is_int=lambda value: type(value) is int,
    )
    runtime.registry.op_impl_ast("TileType")(helpers._parse_tile_type)
    source = f"pl.TileType(shape={shape}, valid_shape=[1, 16], dtype=dtype)"
    parser = runtime.Parser(source, {"dtype": _DType.INT32})
    result = parser.run("TileType")
    assert result["shape"] == [1, 32] and result["valid_shape"] == [1, 16]
    assert result["dtype"] == _DType.INT32
    assert parser.evaluated == ["1", "32", "1", "16", "dtype"]


def test_tile_type_runtime_shape_dimension_keeps_its_diagnostic(runtime):
    helpers = runtime.load_functions(
        "ir/op/block_ops.py", ("_parse_tile_type",),
        TileType=lambda **kwargs: kwargs, _is_int=lambda value: type(value) is int,
    )
    runtime.registry.op_impl_ast("TileType")(helpers._parse_tile_type)
    source = "pl.TileType(shape=[1, n], dtype=dtype)"
    column = source.index("n]") + 1
    value = _Expr(_ScalarType(_DType.INT64), _Span(begin_column=column, end_column=column))
    parser = runtime.Parser(source, {"n": value, "dtype": _DType.INT32})
    with pytest.raises(runtime.errors.InvalidType, match="must be a compile-time integer") as error:
        parser.run("TileType")
    assert error.value.span.begin_column == source.index("n]") + 1


def test_legacy_adds_still_reaches_native_ir_validation(runtime):
    runtime.ir.create_op_call = lambda name, args, attrs, span: (name, args, attrs, span)
    runtime.load("pypto_pro.ir.op.scalar_ops", "ir/op/scalar_ops.py")
    tile = _Expr(_TileType(_DType.INT32))
    tensor = _Expr(runtime.ir.TensorType(_DType.INT32))
    parser = runtime.Parser("pl.adds(out, src, 1)", {"out": tile, "src": tensor})
    name, args, attrs, span = parser.run("adds")
    assert name == "block.adds" and args[:2] == [tile, tensor] and args[2].value == 1
    assert attrs == {} and span.begin_column == 1


def test_registered_implementations_have_no_ast_parsing_or_tables():
    for path in (_SOURCE / "ir/op").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in ("register_table", "OpSpec"), path
            if isinstance(node, ast.FunctionDef) and any(
                isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "op_impl" for decorator in node.decorator_list
            ):
                parameters = [*node.args.posonlyargs, *node.args.args]
                assert parameters[0].arg == "self", (path, node.name)
                assert "span" in [argument.arg for argument in node.args.kwonlyargs], (path, node.name)
                for child in ast.walk(node):
                    if isinstance(child, ast.Attribute):
                        assert child.attr not in ("parse_expression", "parse_op_kwargs"), path
