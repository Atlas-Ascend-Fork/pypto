/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file logicaland.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_LOGICALAND__H
#define TILEOP_TILE_OPERATOR_LOGICALAND__H
#include "utils/sync.h"
#include <type_traits>

#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <typename Type>
using CVT_TYPE = std::conditional_t<std::is_same_v<Type, float> || std::is_same_v<Type, bfloat16_t> ||
                                        std::is_same_v<Type, int32_t> || std::is_same_v<Type, int64_t>,
                                    float, half>;

template <typename T, typename U>
using FINAL_TYPE = std::conditional_t<std::is_same_v<CVT_TYPE<T>, CVT_TYPE<U>>, CVT_TYPE<T>, float>;

template <typename T, typename U, typename Cvt>
TILEOP auto& StandardizedSrcTile([[maybe_unused]] T& dst, U& src, Cvt& cvt)
{
    if constexpr (std::is_same_v<typename T::DType, typename U::DType>) {
        return src;
    } else {
        if constexpr (!std::is_same_v<typename T::DType, float> || sizeof(typename U::DType) > 1) {
            pto::TCVT(dst, src, pto::RoundMode::CAST_NONE);
        } else {
            pto::TCVT(cvt, src, pto::RoundMode::CAST_NONE);
            SyncV();
            pto::TCVT(dst, cvt, pto::RoundMode::CAST_NONE);
        }
        return dst;
    }
}

template <typename T, typename U, typename R, typename V>
TILEOP void CalculateLogicalTile(T& dst, U& src, R& res, V& startAddrUBTile)
{
    pto::TCMPS(res, src, static_cast<typename U::DType>(0), pto::CmpMode::EQ);
    SyncV();
    // EQ selects the original zero; every nonzero value selects scalar one.
    pto::TSELS(dst, res, src, startAddrUBTile, static_cast<typename U::DType>(1));
}

template <typename T, typename U, typename R, typename Cvt, typename V>
TILEOP void CalculateSrcLogicalTile(T& dst, U& src, R& res, Cvt& cvt, V& startAddrUBTile)
{
    auto tile = StandardizedSrcTile(dst, src, cvt);
    SyncV();
    CalculateLogicalTile(dst, tile, res, startAddrUBTile);
}

template <typename T, typename U, typename TMP>
TILEOP void SelectLogicalResult(T& dst, U& src1, U& src2, TMP& tmp)
{
    pto::TMIN(src1, src1, src2);
    SyncV();
    if constexpr (std::is_same_v<typename TMP::DType, typename U::DType>) {
        pto::TCVT(dst, src1, pto::RoundMode::CAST_NONE);
    } else {
        pto::TCVT(tmp, src1, pto::RoundMode::CAST_NONE);
        SyncV();
        pto::TCVT(dst, tmp, pto::RoundMode::CAST_NONE);
    }
}

template <typename T, typename U1, typename U2, typename L, typename Res, typename Cvt, typename V>
TILEOP void LogicalAndImpl(T& dst, U1& src1, U2& src2, L& tmp1, L& tmp2, Res& res, Cvt& cvt, V& startAddrUBTile)
{
    CalculateSrcLogicalTile(tmp1, src1, res, cvt, startAddrUBTile);
    SyncV();
    CalculateSrcLogicalTile(tmp2, src2, res, cvt, startAddrUBTile);
    SyncV();
    SelectLogicalResult(dst, tmp1, tmp2, cvt);
    // Finish reading cvt before the next tile reuses it.
    SyncV();
}

template <typename T0, typename T1, typename T2, typename T3>
TILEOP void TLogicalAnd(T0 dst, T1 src1, T2 src2, T3 tmp)
{
    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    using DType = FINAL_TYPE<typename T1::Type, typename T2::Type>;
    constexpr auto tileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    constexpr auto alignElems = TileOp::BLOCK_SIZE / sizeof(typename T0::Type);
    constexpr auto tmpW = (tileW + alignElems - 1) / alignElems * alignElems;
    // TCMP reads full repeats. Pad the allocation tail, not every row.
    constexpr auto compareElems = TileOp::REPEAT_BYTE / sizeof(half);
    constexpr auto tailElems = (tmpW + compareElems - 1) / compareElems * compareElems - tmpW;
    constexpr auto bufferElems = tileH * tmpW + tailElems;
    constexpr auto maskBytesPerRow = (tmpW + TileOp::BITS_PER_BYTE - 1) / TileOp::BITS_PER_BYTE;
    constexpr auto maskW = (maskBytesPerRow + TileOp::BLOCK_SIZE - 1) / TileOp::BLOCK_SIZE * TileOp::BLOCK_SIZE;
    constexpr auto maskBytes = tileH * maskW;
    constexpr auto dataBytes = bufferElems * sizeof(DType);
    constexpr auto cvtBytes = std::is_same_v<DType, float> ? bufferElems * sizeof(half) : 0;
    constexpr auto sharedBytes = maskBytes > cvtBytes ? maskBytes : cvtBytes;
    constexpr auto selectTmpCols = TileOp::BLOCK_SIZE / sizeof(uint32_t);

    using TmpTile = pto::Tile<pto::TileType::Vec, DType, tileH, tmpW, pto::BLayout::RowMajor, -1, -1>;
    using CvtTile = pto::Tile<pto::TileType::Vec, half, tileH, tmpW, pto::BLayout::RowMajor, -1, -1>;
    using MaskTile = pto::Tile<pto::TileType::Vec, uint8_t, tileH, maskW, pto::BLayout::RowMajor, -1, -1>;
    using SelectTmpTile = pto::Tile<pto::TileType::Vec, uint32_t, 1, selectTmpCols, pto::BLayout::RowMajor, -1, -1>;

    auto dstTile = PtoTile<T0>(dst);
    auto src1Tile = MakeElementwiseOperandExecTile(dst, src1);
    auto src2Tile = MakeElementwiseOperandExecTile(dst, src2);
    TmpTile tmp1Tile(dstShape3, dstShape4);
    TmpTile tmp2Tile(dstShape3, dstShape4);
    CvtTile cvtTile(dstShape3, dstShape4);
    auto validMaskCols = (dstShape4 + TileOp::BITS_PER_BYTE - 1) / TileOp::BITS_PER_BYTE;
    MaskTile cmpTile(dstShape3, validMaskCols);
    SelectTmpTile selectTmpTile(1, selectTmpCols);

    // Keep this layout in sync with TiledLogicalAndOperation.
    auto tmp1Addr = (uint64_t)tmp.GetAddr();
    auto tmp2Addr = tmp1Addr + dataBytes;
    auto sharedAddr = tmp2Addr + dataBytes;
    auto selectTmpAddr = sharedAddr + sharedBytes;
    pto::TASSIGN(tmp1Tile, tmp1Addr);
    pto::TASSIGN(tmp2Tile, tmp2Addr);
    // Each comparison follows conversion; the final conversion follows both selections.
    pto::TASSIGN(cmpTile, sharedAddr);
    pto::TASSIGN(cvtTile, std::is_same_v<DType, float> ? sharedAddr : tmp1Addr);
    pto::TASSIGN(selectTmpTile, selectTmpAddr);

    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                AssignElementwiseOperandExecTile(src1Tile, src1, tileOffsets);
                AssignElementwiseOperandExecTile(src2Tile, src2, tileOffsets);
                LogicalAndImpl(dstTile.Data(), src1Tile, src2Tile, tmp1Tile, tmp2Tile, cmpTile, cvtTile, selectTmpTile);
            }
        }
    }
}
#endif // TILEOP_TILE_OPERATOR_LOGICALAND__H
