# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Native stamps, target selection and validation before device compilation."""
import ast
import importlib
from pathlib import Path
import re

from pypto_pro import ir
from pypto_pro._errors import InvalidArgument, InvalidType, NotSupported, PyptoProError
import pypto_pro.language as pl
from pypto_pro.language.parser._ast_parser import ASTParser
from pypto_pro.runtime.jit import (
    _add_kernel_header,
    _assemble_cv_source,
    _parse_and_codegen_targets,
    _write_codegen_artifacts,
)
from pypto_pro.runtime.platform import NpuArch
import pytest


@pl.pipeline.stage
def stage_body():
    pl.mark_stamp(pl.PipeType.V, 3)


@pl.pipeline.stage
def plain_stage():
    pl.system.bar_all()


@pl.jit(auto_mutex=False)
def plain_kernel(flag: pl.DT_INT32):
    with pl.section_vector():
        plain_stage()


@pl.jit(auto_mutex=False)
def stamped_kernel(flag: pl.DT_INT32):
    with pl.section_vector():
        pl.mark_stamp(pl.PipeType.MTE2, 0)
        if flag == 0:
            return
        for i in pl.range(0, 2):
            stage_body()
        pl.mark_stamp(pl.PipeType.MTE3, 4095)


def generate(tmp_path, kernel=stamped_kernel, arch=NpuArch.DAV_3510):
    return _assemble_cv_source(*_parse_and_codegen_targets(kernel.to_kernel_def(), arch, str(tmp_path)))


@pytest.fixture
def forbid_codegen(monkeypatch):
    def unexpected_codegen(*args, **kwargs):
        pytest.fail("Invalid mark_stamp reached CCE code generation")

    jit = importlib.import_module("pypto_pro.runtime.jit")
    monkeypatch.setattr(jit, "_codegen_target_cce", unexpected_codegen)


@pytest.mark.parametrize('arch', [NpuArch.DAV_3510, NpuArch.DAV_9201])
def test_no_manual_calls_means_no_pmu_instrumentation(tmp_path, arch):
    result = _add_kernel_header(generate(tmp_path, plain_kernel, arch))
    assert "mark_stamp" not in result.content
    assert 'asm volatile("nop")' not in result.content
    assert '__pypto_pmu_selected' not in result.content


def test_manual_stamps_drain_after_early_return(tmp_path):
    result = generate(tmp_path)
    assert set(re.findall(r'mark_stamp<PIPE_(\w+), (\d+)>', result.content)) == {
        ('MTE2', '0'), ('V', '3'), ('MTE3', '4095'),
    }
    wrapper = result.content[result.content.rindex('__aicore__ inline void'):]
    body = result.content[:result.content.rindex('__aicore__ inline void')]
    assert (body.index('return;') < body.index('}();') < body.index('pipe_barrier(PIPE_ALL)')
            < body.index('asm volatile("nop")'))
    assert 'pipe_barrier' not in wrapper
    assert 'mark_stamp' not in wrapper
    assert '__pypto_pmu_selected' not in result.content
    assert 'get_block_idx' not in result.content
    assert 'get_subblockid' not in result.content
    assert 'if (' not in wrapper


@pytest.mark.parametrize('target, pipes', [
    ('cube', ('S', 'MTE2', 'MTE1', 'M', 'FIX', 'MTE3')),
    ('vector', ('S', 'MTE2', 'V', 'MTE3')),
])
def test_all_supported_pipes_add_no_extra_sync_or_spacing(tmp_path, target, pipes):
    for pipe_name in pipes:
        pipe = getattr(pl.PipeType, pipe_name)

        @pl.jit(auto_mutex=False)
        def ordered(flag: pl.DT_INT32):
            if target == 'cube':
                with pl.section_cube():
                    pl.mark_stamp(pipe, 21)
            else:
                with pl.section_vector():
                    pl.mark_stamp(pipe, 21)

        result = generate(tmp_path, ordered)
        body = result.content.split('}();')[0]
        assert f'bisheng::cce::mark_stamp<PIPE_{pipe_name}, 21>();' in body
        assert all(token not in body for token in (
            'pipe_barrier', 'nop', 'set_flag', 'wait_flag', 'if (', 'get_block_idx', 'get_subblockid',
        ))


def test_unsupported_arch_rejected_before_bisheng(tmp_path):
    with pytest.raises(NotSupported, match='PMU stamps require arch 3510') as error:
        generate(tmp_path, arch=NpuArch.DAV_9201)
    assert Path(error.value.span['filename']) == Path(__file__)


@pytest.mark.parametrize('stamp_id, message', [
    (-1, 'at least 0'), (4096, '4095'), (65536, '4095'),
    (True, 'compile-time integer'), (False, 'compile-time integer'),
    (1.0, 'id expects int'), (1.5, 'id expects int'),
    ('1', 'id expects int'), (None, 'id expects int'),
])
def test_invalid_id_rejected_before_codegen(tmp_path, forbid_codegen, stamp_id, message):
    @pl.jit(auto_mutex=False)
    def invalid(flag: pl.DT_INT32):
        with pl.section_vector():
            pl.mark_stamp(pl.PipeType.S, stamp_id)

    with pytest.raises(PyptoProError, match=message) as error:
        generate(tmp_path, invalid)
    assert Path(error.value.span['filename']) == Path(__file__)


@pytest.mark.parametrize('pipe', [0, 'V', None, True])
def test_invalid_pipe_type_rejected_before_codegen(tmp_path, forbid_codegen, pipe):
    @pl.jit(auto_mutex=False)
    def invalid(flag: pl.DT_INT32):
        with pl.section_vector():
            pl.mark_stamp(pipe, 0)

    with pytest.raises(InvalidType, match='PipeType'):
        generate(tmp_path, invalid)


@pytest.mark.parametrize('target, pipe', [
    ('cube', pl.PipeType.V), ('cube', pl.PipeType.ALL),
    ('vector', pl.PipeType.M), ('vector', pl.PipeType.MTE1),
    ('vector', pl.PipeType.FIX), ('vector', pl.PipeType.ALL),
])
def test_target_pipe_mismatch_rejected_before_codegen(tmp_path, forbid_codegen, target, pipe):
    @pl.jit(auto_mutex=False)
    def invalid(flag: pl.DT_INT32):
        if target == 'cube':
            with pl.section_cube():
                pl.mark_stamp(pipe, 0)
        else:
            with pl.section_vector():
                pl.mark_stamp(pipe, 0)

    with pytest.raises(InvalidArgument, match='unavailable'):
        generate(tmp_path, invalid)


def test_duplicate_id_across_pipes_rejected_before_codegen(tmp_path, forbid_codegen):
    @pl.jit(auto_mutex=False)
    def invalid(flag: pl.DT_INT32):
        with pl.section_vector():
            pl.mark_stamp(pl.PipeType.S, 7)
            pl.mark_stamp(pl.PipeType.V, 7)

    with pytest.raises(InvalidArgument, match='Duplicate mark_stamp'):
        generate(tmp_path, invalid)


def test_duplicate_id_in_inline_helper_rejected(tmp_path, forbid_codegen):
    @pl.jit(auto_mutex=False)
    def invalid(flag: pl.DT_INT32):
        with pl.section_vector():
            stage_body()
            stage_body()

    with pytest.raises(InvalidArgument, match='Duplicate mark_stamp'):
        generate(tmp_path, invalid)


def test_mixed_targets_allow_same_id_on_all_blocks(tmp_path):
    @pl.jit(auto_mutex=False)
    def mixed(flag: pl.DT_INT32):
        with pl.section_cube():
            pl.mark_stamp(pl.PipeType.M, 0)
        with pl.section_vector():
            pl.mark_stamp(pl.PipeType.V, 0)

    result = generate(tmp_path, mixed)
    assert 'mark_stamp<PIPE_M, 0>' in result.content
    assert 'mark_stamp<PIPE_V, 0>' in result.content
    assert result.content.count('asm volatile("nop")') == 2
    assert '__pypto_pmu_selected' not in result.content
    assert 'get_block_idx' not in result.content
    assert 'get_subblockid' not in result.content
    assert 'if (' not in result.content


def test_unstamped_target_has_no_guard_or_padding(tmp_path):
    @pl.jit(auto_mutex=False)
    def mixed(flag: pl.DT_INT32):
        with pl.section_cube():
            pl.system.bar_all()
        with pl.section_vector():
            pl.mark_stamp(pl.PipeType.V, 0)

    result = generate(tmp_path, mixed)
    cube, vector = result.content.split('#if defined(__DAV_VEC__)', 1)
    assert 'nop' not in cube
    assert 'asm volatile("nop")' in vector
    assert '__pypto_pmu_selected' not in result.content


@pytest.mark.parametrize('target', ['cube', 'vector'])
def test_explicit_nonzero_block_filter_is_preserved(tmp_path, target):
    @pl.jit(auto_mutex=False)
    def filtered(flag: pl.DT_INT32):
        if target == 'cube':
            with pl.section_cube():
                if pl.get_block_idx() == 1:
                    pl.mark_stamp(pl.PipeType.S, 7)
        else:
            with pl.section_vector():
                if pl.get_block_idx() == 1:
                    pl.mark_stamp(pl.PipeType.S, 7)

    result = generate(tmp_path, filtered)
    block = re.search(r'auto (\w+) = \(int64_t\)\(__pypto_block_idx_value\(\)\);', result.content)
    assert block is not None
    condition = re.search(rf'auto (\w+) = \({block.group(1)} == 1\);', result.content)
    assert condition is not None
    assert re.search(
        rf'if \({condition.group(1)}\) {{\s*bisheng::cce::mark_stamp<PIPE_S, 7>\(\);\s*}}', result.content,
    )
    assert result.content.count('if (') == 1
    assert '__pypto_pmu_selected' not in result.content


def test_artifacts_contain_only_source_and_tiling_headers(tmp_path):
    result = _add_kernel_header(generate(tmp_path))
    _write_codegen_artifacts(result)
    assert (tmp_path / 'kernel.cpp').read_text() == result.content
    assert {path.name for path in tmp_path.iterdir()} == {'kernel.cpp', *result.tiling_headers}


def test_compile_time_id_expression_and_reparse(tmp_path):
    @pl.jit(auto_mutex=False)
    def constant(flag: pl.DT_INT32):
        with pl.section_vector():
            pl.mark_stamp(pl.PipeType.S, 1 + 2)

    kernel_def = constant.to_kernel_def()
    for _ in range(2):
        result = _assemble_cv_source(*_parse_and_codegen_targets(kernel_def, NpuArch.DAV_3510, str(tmp_path)))
        assert 'mark_stamp<PIPE_S, 3>' in result.content


@pytest.mark.parametrize('call', [
    'pl.mark_stamp()', 'pl.mark_stamp(pl.PipeType.S)',
    'pl.mark_stamp(pl.PipeType.S, 0, 1)', 'pl.mark_stamp(pl.PipeType.S, 0, id=1)',
    'pl.mark_stamp(pl.PipeType.S, 0, name="removed")',
    'pl.mark_stamp(pl.PipeType.S, 0, stamp_nops=1)',
])
def test_invalid_signature_rejected_at_source(call):
    source = f'def kernel():\n    {call}\n'
    parser = ASTParser(
        __file__, source.splitlines(), ir.SectionKind.Vector, closure_vars={'pl': pl}, debug_info=ir.IRDebugInfo(),
    )
    with pytest.raises(InvalidArgument) as error:
        parser.parse_function(ast.parse(source).body[0])
    assert error.value.span['line'] == 2


@pytest.mark.parametrize('body', [
    'if pl.mark_stamp(pl.PipeType.S, 0):\n        pass',
    'while pl.mark_stamp(pl.PipeType.S, 0):\n        pass',
    'pl.mark_stamp(pl.PipeType.S, 0) + 1',
    'pl.min(pl.mark_stamp(pl.PipeType.S, 0), 1)',
    'value = stamp(pl.PipeType.S, 0)\n    value + 1',
])
def test_stamp_cannot_be_used_as_a_value(body):
    source = f'def kernel():\n    {body}\n'
    parser = ASTParser(
        __file__, source.splitlines(), ir.SectionKind.Vector,
        closure_vars={'pl': pl, 'stamp': pl.mark_stamp}, debug_info=ir.IRDebugInfo(),
    )
    with pytest.raises((PyptoProError, ValueError)):
        parser.parse_function(ast.parse(source).body[0])


def test_stamp_alias_as_statement(tmp_path):
    stamp = pl.mark_stamp

    @pl.jit(auto_mutex=False)
    def alias(flag: pl.DT_INT32):
        with pl.section_vector():
            stamp(pl.PipeType.S, 0)

    result = generate(tmp_path, alias)
    assert 'mark_stamp<PIPE_S, 0>' in result.content


@pytest.mark.parametrize('loop_id', [False, True])
def test_runtime_id_rejected_before_codegen(tmp_path, forbid_codegen, loop_id):
    @pl.jit(auto_mutex=False)
    def invalid(stamp_id: pl.DT_INT32):
        with pl.section_vector():
            if loop_id:
                for i in pl.range(0, stamp_id):
                    pl.mark_stamp(pl.PipeType.V, i)
            else:
                pl.mark_stamp(pl.PipeType.V, stamp_id)

    with pytest.raises(InvalidType, match="'id' must be a compile-time integer"):
        generate(tmp_path, invalid)


def test_name_keyword_rejected_before_codegen(tmp_path, forbid_codegen):
    @pl.jit(auto_mutex=False)
    def invalid(flag: pl.DT_INT32):
        with pl.section_vector():
            pl.mark_stamp(pl.PipeType.V, 0, name='removed')

    with pytest.raises(PyptoProError, match='name'):
        generate(tmp_path, invalid)


@pytest.mark.parametrize('mode', ['simd', 'simt'])
@pytest.mark.parametrize('nested', [False, True])
def test_vector_function_and_nested_callee_rejected_before_codegen(tmp_path, forbid_codegen, mode, nested):
    @pl.vector_function(mode=mode)
    def invalid_body():
        pl.mark_stamp(pl.PipeType.S, 0)

    @pl.vector_function(mode=mode)
    def outer():
        invalid_body()

    @pl.jit(auto_mutex=False)
    def simd_kernel(flag: pl.DT_INT32):
        with pl.section_vector():
            if nested:
                outer()
            else:
                invalid_body()

    @pl.jit(mode='simt', max_threads=32)
    def simt_kernel():
        if nested:
            outer()
        else:
            invalid_body()

    with pytest.raises(NotSupported, match='mark_stamp.*(vector_function|SIMT)'):
        generate(tmp_path, simd_kernel if mode == 'simd' else simt_kernel)


@pytest.mark.parametrize('func_type', [
    ir.FunctionType.SimdVF, ir.FunctionType.SimdCallee,
    ir.FunctionType.SimtVF, ir.FunctionType.SimtCallee, ir.FunctionType.SimtKernel,
])
def test_all_vector_and_simt_function_types_reject_stamp_at_source(func_type):
    source = 'def kernel():\n    pl.mark_stamp(pl.PipeType.S, 0)\n'
    parser = ASTParser(
        __file__, source.splitlines(), ir.SectionKind.Vector, closure_vars={'pl': pl}, debug_info=ir.IRDebugInfo(),
    )
    with pytest.raises(NotSupported, match='mark_stamp.*(vector_function|SIMT)') as error:
        parser.parse_function(ast.parse(source).body[0], func_type=func_type)
    assert error.value.span['line'] == 2
