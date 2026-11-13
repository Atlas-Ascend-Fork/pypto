# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""The common SIMD tile ABI also carries metadata through pointer bindings and control flow."""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.jit import _assemble_cv_source, _parse_and_codegen_targets
from pypto_pro.runtime.platform import NpuArch


@pl.vector_function(mode="simd")
def copy_valid(dst, src):
    for row in pl.range(dst.valid_shape[0]):
        mask = vf.update_mask(dst.valid_shape[1], dtype=pl.DT_FP32)
        value = vf.load_align(src, row * 64)
        vf.store_align(dst, value, mask, row * 64)


@pl.vector_function(mode="simd")
def pointer_bindings(dst, src, count):
    alias = src + 0
    copy_valid(dst, alias)
    copy_valid(dst, src + 0)
    for _ in pl.range(count):
        alias = alias + 0
    copy_valid(dst, alias)


@pl.jit(auto_mutex=False)
def pointer_kernel(count: pl.DT_INT32):
    with pl.section_vector():
        tt = pl.TileType(shape=[2, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[1, 37])
        src = pl.make_tile(tt, addr=0)
        dst = pl.make_tile(tt, addr=0x1000)
        pointer_bindings(dst, src, count)


def test_simd_valid_shape_codegen(assert_9201_golden):
    cube, vector = _parse_and_codegen_targets(pointer_kernel.to_kernel_def(), NpuArch.DAV_3510, "")
    source = _assemble_cv_source(cube, vector).content
    assert_9201_golden(source, "cce")
    assert "__simd_callee__ inline void copy_valid__callee(__ubuf__ float* dst_0, __ubuf__ float* src_0)" in source
    assert (
        "__simd_vf__ inline void pointer_bindings(__ubuf__ float* dst_0, __ubuf__ float* src_0, int32_t count_0)"
        in source
    )
    assert "__valid_row" not in source
    assert "__valid_col" not in source

    copy_body = source[source.index("__simd_callee__"):source.index("__simd_vf__")]
    assert "auto _expr_tmp_0_0 = (int64_t)(1);" in copy_body
    assert "const uint16_t row__iterator_0_ub = (uint16_t)(_expr_tmp_0_0);" in copy_body
    assert "row__iterator_0 < row__iterator_0_ub" in copy_body
    assert "auto _expr_tmp_1_0 = (int64_t)(37);" in copy_body
    assert "uint32_t _vf_mask_scalar_0 = (uint32_t)_expr_tmp_1_0;" in copy_body
    assert "mask_0 = plt_b32(_vf_mask_scalar_0, POST_UPDATE);" in copy_body

    bindings = [
        "__ubuf__ float* alias_0 = (src_0 + (0));",
        "copy_valid__callee(dst_0, alias_0);",
        "__ubuf__ float* _expr_tmp_0_0 = (src_0 + (0));",
        "copy_valid__callee(dst_0, _expr_tmp_0_0);",
        "const uint16_t ___iterator_0_ub = (uint16_t)(count_0);",
        "__ubuf__ float* alias_1 = alias_0;",
        "for (uint16_t ___iterator_0 = 0; ___iterator_0 < ___iterator_0_ub; ___iterator_0 += 1)",
        "__ubuf__ float* alias_2 = (alias_1 + (0));",
        "alias_1 = alias_2;",
        "copy_valid__callee(dst_0, alias_1);",
    ]
    positions = [source.index(binding) for binding in bindings]
    assert positions == sorted(positions)
    assert (
        "pointer_bindings((__ubuf__ float *)dst_0.tile.data(), (__ubuf__ float *)src_0.tile.data(), count_0);"
        in source
    )
