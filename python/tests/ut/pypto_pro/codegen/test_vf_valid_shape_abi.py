# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""VF signatures, entry calls and nested calls pass only dynamic valid dimensions."""

import pypto_pro.language as pl
from pypto_pro.runtime.jit import _assemble_cv_source, _parse_and_codegen_targets
from pypto_pro.runtime.platform import NpuArch
import pytest


def _make_kernel(mode, dynamic_row, dynamic_col):
    valid_shape = [-1 if dynamic_row else 1, -1 if dynamic_col else 37]

    @pl.vector_function(mode=mode)
    def leaf(tile):
        rows = tile.valid_shape[0]  # noqa: F841
        cols = tile.valid_shape[1]  # noqa: F841

    @pl.vector_function(mode=mode, **({"max_threads": 32} if mode == "simt" else {}))
    def entry(tile):
        leaf(tile)

    @pl.jit(auto_mutex=False)
    def kernel(rows: pl.DT_INT32, cols: pl.DT_INT32):
        with pl.section_vector():
            tt = pl.TileType(
                shape=[2, 64],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Vec,
                valid_shape=valid_shape,
            )
            tile = pl.make_tile(tt, addr=0)
            if dynamic_row and dynamic_col:
                pl.set_validshape(tile, [rows, cols])
            if mode == "simt":
                entry[32](tile)
            else:
                entry(tile)

    return kernel


@pytest.mark.parametrize("mode", ["simd", "simt"])
@pytest.mark.parametrize("dynamic_row,dynamic_col", [(False, False), (True, False), (False, True), (True, True)])
def test_vf_valid_shape_abi(mode, dynamic_row, dynamic_col, assert_9201_golden):
    kernel = _make_kernel(mode, dynamic_row, dynamic_col)
    cube, vector = _parse_and_codegen_targets(kernel.to_kernel_def(), NpuArch.DAV_3510, "")
    source = _assemble_cv_source(cube, vector).content
    assert_9201_golden(source, "cce")
    params = ["__ubuf__ float* tile_0"]
    callee_args = ["tile_0"]
    pointer_cast = "(__ubuf__ float *)" if mode == "simd" else "(__ubuf__ float*)"
    entry_args = [f"{pointer_cast}tile_0.tile.data()"]
    for axis, dynamic, constant, variable in (("row", dynamic_row, 1, "rows"), ("col", dynamic_col, 37, "cols")):
        metadata = f"tile_0__valid_{axis}"
        if dynamic:
            params.append(f"uint32_t {metadata}")
            callee_args.append(metadata)
            entry_args.append(f"(uint32_t)tile_0.tile.GetValid{axis.title()}()")
        else:
            assert metadata not in source
        expected_value = metadata if dynamic else str(constant)
        assert f"auto {variable}_0 = (int64_t)({expected_value});" in source

    leaf_name = "leaf__callee" if mode == "simd" else "leaf"
    signature = ", ".join(params)
    assert f"__{mode}_callee__ inline void {leaf_name}({signature})" in source
    assert f"{leaf_name}({', '.join(callee_args)});" in source
    if mode == "simd":
        assert f"__simd_vf__ inline void entry({signature})" in source
        assert f"entry({', '.join(entry_args)});" in source
    else:
        assert f"__simt_vf__ __launch_bounds__(32) inline void entry({signature})" in source
        assert f"cce::async_invoke<entry>(cce::dim3{{32, 1, 1}}, {', '.join(entry_args)});" in source
    if dynamic_row and dynamic_col:
        assert "tile_0.tile.SetValidShape(rows_0, cols_0);" in source
