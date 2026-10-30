/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TILEOP_VECTOR_QUANT_HIF4_TILE_H
#define TILEOP_VECTOR_QUANT_HIF4_TILE_H
#include "pto_tile.h"
#include "quant_hif4.h"

using hfloat4 = float4_e1m2x2_t;
#define OP_TILE_OP_QUANT_HIF4 TQuantHIF4

template <int SCALE_ALG = 0, int AXIS = -1, int COMPUTE_COLS = 0, int COMPUTE_ROWS = 0, typename T0, typename T1,
          typename T2, typename T3>
TILEOP void TQuantHIF4(T0 dst, T1 exp, T2 scratch, T3 src)
{
    (void)AXIS;
    constexpr auto rows = TileOp::GetTensorTileShapeDim<T3, DIM_4TH, MAX_DIMS>();
    constexpr auto cols = TileOp::GetTensorTileShapeDim<T3, DIM_5TH, MAX_DIMS>();
    static_assert(COMPUTE_COLS == 0 || COMPUTE_COLS == cols, "HiF4 compute columns must match the 2D tile.");
    static_assert(COMPUTE_ROWS == 0 || COMPUTE_ROWS == rows, "HiF4 compute rows must match the 2D tile.");
    static_assert(TileOp::GetTensorStrideDim<T3, DIM_4TH, MAX_DIMS>() == cols || rows == 1,
                  "HiF4 source rows must be contiguous within a 2D tile.");
    static_assert(TileOp::GetTensorStrideDim<T0, DIM_4TH, MAX_DIMS>() == cols || rows == 1,
                  "HiF4 destination rows must be contiguous within a 2D tile.");
    constexpr hif4::ScratchLayout allocation(rows, cols);
    static_assert(GetAllAxisTileProduct<T2>() * sizeof(typename T2::Type) >= allocation.bytes,
                  "HiF4 scratch must cover one 2D tile.");
    constexpr auto expRowStride = TileOp::GetTensorStrideDim<T1, DIM_4TH, MAX_DIMS>();
    constexpr auto expCols = expRowStride > 0 ? expRowStride : TileOp::GetTensorTileShapeDim<T1, DIM_5TH, MAX_DIMS>();
    constexpr auto expRows = TileOp::GetTensorTileShapeDim<T1, DIM_4TH, MAX_DIMS>();
    using ExpTile = pto::Tile<pto::TileType::Vec, uint32_t, expRows, expCols, pto::BLayout::RowMajor, -1, -1>;
    using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, allocation.bytes, pto::BLayout::RowMajor, 1,
                                  allocation.bytes>;
    const auto sl = src.GetLayout();
    const auto el = exp.GetLayout();
    auto srcTile = PtoTile<T3>(src);
    auto dstTile = PtoTile<T0>(dst);
    ExpTile expTile(el.template GetShapeDim<DIM_4TH, MAX_DIMS>(), el.template GetShapeDim<DIM_5TH, MAX_DIMS>());
    ScratchTile scratchTile;
    pto::TASSIGN(scratchTile, (uint64_t)scratch.GetAddr());
    for (LoopVar n0 = 0; n0 < sl.template GetShapeDim<DIM_1ST, MAX_DIMS>(); ++n0) {
        for (LoopVar n1 = 0; n1 < sl.template GetShapeDim<DIM_2ND, MAX_DIMS>(); ++n1) {
            for (LoopVar n2 = 0; n2 < sl.template GetShapeDim<DIM_3RD, MAX_DIMS>(); ++n2) {
                const auto offsets = TileOffset(n0, n1, n2);
                srcTile.Assign(src, offsets);
                dstTile.Assign(dst, offsets);
                auto expOffset = n0 * el.template GetStrideDim<DIM_1ST, MAX_DIMS>() +
                                 n1 * el.template GetStrideDim<DIM_2ND, MAX_DIMS>() +
                                 n2 * el.template GetStrideDim<DIM_3RD, MAX_DIMS>();
                pto::TASSIGN(expTile, (uint64_t)(exp.GetAddr() + expOffset * sizeof(uint32_t)));
                hif4::TQUANT<SCALE_ALG>(dstTile.Data(), srcTile.Data(), expTile, scratchTile);
            }
        }
    }
}
#endif
