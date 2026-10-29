/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file signbit.h
 * \brief signbit operator: identify sign bit, negative returns true, positive returns false
 */

#ifndef TILEOP_TILE_OPERATOR_SIGNBIT__H
#define TILEOP_TILE_OPERATOR_SIGNBIT__H
#include "pto_tile.h"
#include "utils/sync.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"
#include <type_traits>

#define OP_TILE_OP_SIGNBIT TSignbit
template <typename LastUse = LastUse2Dim<0, 0>, typename T0, typename T1, typename T2>
TILEOP void TSignbit(T0 dst, T1 src, T2 tmp)
{
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto srcTypeSize = sizeof(typename T1::Type);

    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    auto srcExecShape3 = GetElementwiseOperandExecShapeDim<DIM_4TH, MAX_DIMS>(dst, src);
    auto srcExecShape4 = GetElementwiseOperandExecShapeDim<DIM_5TH, MAX_DIMS>(dst, src);

    auto dstStride0 = dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto dstStride1 = dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto dstStride2 = dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();

    auto srcStride0 = srcLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto srcStride1 = srcLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto srcStride2 = srcLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();

    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();

    using SrcExecConfig = ElementwiseOperandExecConfig<T0, T1>;
    constexpr auto srcTileH = SrcExecConfig::tileH;
    constexpr auto srcTileW = SrcExecConfig::tileW;

    constexpr auto tmpTileW = (srcTileW + TileOp::BLOCK_NELEM_B16 - 1) / TileOp::BLOCK_NELEM_B16 *
                              TileOp::BLOCK_NELEM_B16;
    constexpr auto tmpTileW32Bit = (srcTileW + TileOp::BLOCK_NELEM_B32 - 1) / TileOp::BLOCK_NELEM_B32 *
                                   TileOp::BLOCK_NELEM_B32;

    using DstTile = pto::Tile<pto::TileType::Vec, uint8_t, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
    using SrcTile = pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileH, srcTileW, pto::BLayout::RowMajor, -1,
                              -1>;
    using SrcUint32Tile = pto::Tile<pto::TileType::Vec, uint32_t, srcTileH, tmpTileW32Bit, pto::BLayout::RowMajor, -1,
                                    -1>;
    using SrcInt32Tile = pto::Tile<pto::TileType::Vec, int32_t, srcTileH, tmpTileW32Bit, pto::BLayout::RowMajor, -1,
                                   -1>;
    using SrcUint16Tile = pto::Tile<pto::TileType::Vec, uint16_t, srcTileH, tmpTileW, pto::BLayout::RowMajor, -1, -1>;
    using SrcInt16Tile = pto::Tile<pto::TileType::Vec, int16_t, srcTileH, tmpTileW, pto::BLayout::RowMajor, -1, -1>;
    using SrcUint8Tile = pto::Tile<pto::TileType::Vec, uint8_t, srcTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
    using SrcInt8Tile = pto::Tile<pto::TileType::Vec, int8_t, srcTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
    DstTile dstTile(dstShape3, dstShape4);
    SrcTile srcTile(srcExecShape3, srcExecShape4);
    SrcUint32Tile srcUint32Tile(srcExecShape3, srcExecShape4);
    SrcUint16Tile srcUint16Tile(srcExecShape3, srcExecShape4);
    SrcInt32Tile srcInt32Tile(srcExecShape3, srcExecShape4);
    SrcInt16Tile srcInt16Tile(srcExecShape3, srcExecShape4);
    SrcUint8Tile srcUint8Tile(srcExecShape3, srcExecShape4);
    SrcInt8Tile srcInt8Tile(srcExecShape3, srcExecShape4);

    using TempUint16Tile = pto::Tile<pto::TileType::Vec, uint16_t, srcTileH, tmpTileW, pto::BLayout::RowMajor, -1, -1>;
    using TempInt16Tile = pto::Tile<pto::TileType::Vec, int16_t, srcTileH, tmpTileW, pto::BLayout::RowMajor, -1, -1>;
    using TempFp16Tile = pto::Tile<pto::TileType::Vec, half, srcTileH, tmpTileW, pto::BLayout::RowMajor, -1, -1>;
    TempUint16Tile tempUint16Tile(srcExecShape3, srcExecShape4);
    TempInt16Tile tempInt16Tile(srcExecShape3, srcExecShape4);
    TempFp16Tile tempFp16Tile(srcExecShape3, srcExecShape4);
    pto::TASSIGN(tempUint16Tile, (uint64_t)(tmp.GetAddr()));
    pto::TASSIGN(tempInt16Tile, (uint64_t)(tmp.GetAddr()));
    pto::TASSIGN(tempFp16Tile, (uint64_t)(tmp.GetAddr()));

    constexpr auto shiftAmountUint32 = 31;
    constexpr auto shiftAmountUint16 = 15;
    constexpr auto zero = 0;
    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));

                if constexpr (std::is_same<typename T1::Type, uint8_t>::value) {
                    pto::TEXPANDS(dstTile, static_cast<uint8_t>(zero));
                    continue;
                } else if constexpr (std::is_same<typename T1::Type, float>::value ||
                                     std::is_same<typename T1::Type, int32_t>::value) {
                    pto::TASSIGN(srcUint32Tile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                    pto::TASSIGN(srcInt32Tile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                    pto::TSHRS(srcUint32Tile, srcUint32Tile, static_cast<uint16_t>(shiftAmountUint32));
                    SyncV();
                    pto::TCVT(tempInt16Tile, srcInt32Tile, pto::RoundMode::CAST_NONE);
                } else if constexpr (std::is_same<typename T1::Type, half>::value ||
                                     std::is_same<typename T1::Type, bfloat16_t>::value ||
                                     std::is_same<typename T1::Type, int16_t>::value) {
                    pto::TASSIGN(srcUint16Tile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                    pto::TSHRS(tempUint16Tile, srcUint16Tile, shiftAmountUint16);
                } else if constexpr (std::is_same<typename T1::Type, int8_t>::value ||
                                     std::is_same<typename T1::Type, bool>::value) {
                    pto::TASSIGN(srcInt8Tile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                    pto::TCVT(tempFp16Tile, srcInt8Tile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TSHRS(tempUint16Tile, tempUint16Tile, shiftAmountUint16);
                }
                SyncV();
                pto::TCVT(dstTile, tempFp16Tile, pto::RoundMode::CAST_CEIL);
            }
        }
    }
}

#endif
