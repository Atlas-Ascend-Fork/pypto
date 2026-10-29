/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file extract.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_EXTRACT__H
#define TILEOP_TILE_OPERATOR_EXTRACT__H
#include "utils/sync.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#define OP_TILE_OP_EXTRACT TExtract
template <int k, int extractMode, int isLargest, typename T0, typename T1>
TILEOP void TExtract(T0 dst, T1 src)
{
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto dstStride0 = dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto dstStride1 = dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto dstStride2 = dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto dstStride3 = dstLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();

    if (dstShape0 == 0 || dstShape1 == 0 || dstShape2 == 0 || dstShape3 == 0 || dstShape4 == 0) {
        return;
    }

    // Pattern gather packs rows contiguously, so invoke it separately for each physical row.
    auto srcShape4 = dstShape4 * 2;
    auto srcStride0 = srcLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto srcStride1 = srcLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto srcStride2 = srcLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto srcStride3 = srcLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();

    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, DIM_5TH, MAX_DIMS>();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto srcTypeSize = sizeof(typename T1::Type);

    for (LoopVar n0Index = 0; n0Index < dstShape0; n0Index++) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; n1Index++) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; n2Index++) {
                for (LoopVar n3Index = 0; n3Index < dstShape3; n3Index++) {
                    using DstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, 1, dstTileW,
                                                    pto::BLayout::RowMajor, -1, -1>;
                    using SrcTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, 1, srcTileW,
                                                    pto::BLayout::RowMajor, -1, -1>;
                    DstTileDefine dstTile(1, dstShape4);
                    SrcTileDefine srcTile(1, srcShape4);
                    auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2 +
                                     n3Index * dstStride3;
                    auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2 +
                                     n3Index * srcStride3;
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                    constexpr auto pattern = (extractMode == 0) ? pto::MaskPattern::P0101 : pto::MaskPattern::P1010;
                    pto::TGATHER<DstTileDefine, SrcTileDefine, pattern>(dstTile, srcTile);
                    SyncV();

                    if constexpr (extractMode == 0 && isLargest == 0) {
                        constexpr int32_t INT32_MIN_VALUE = static_cast<int32_t>(0x80000000U);
                        using DstAddTileDefine = pto::Tile<pto::TileType::Vec, int32_t, 1, dstTileW,
                                                           pto::BLayout::RowMajor, -1, -1>;
                        DstAddTileDefine dstAddTile(1, dstShape4);
                        pto::TASSIGN(dstAddTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                        pto::TADDS(dstAddTile, dstAddTile, INT32_MIN_VALUE);
                        SyncV();
                    }
                }
            }
        }
    }
}

#define OP_TILE_OP_EXTRACTSINGLE TExtractSingle
template <int extractMode, int isLargest, typename T0, typename T1>
TILEOP void TExtractSingle(T0 dst, T1 src)
{
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto dstStride0 = dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto dstStride1 = dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto dstStride2 = dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto dstStride3 = dstLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();

    if (dstShape0 == 0 || dstShape1 == 0 || dstShape2 == 0 || dstShape3 == 0 || dstShape4 == 0) {
        return;
    }

    auto srcShape4 = srcLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto srcStride0 = srcLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto srcStride1 = srcLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto srcStride2 = srcLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto srcStride3 = srcLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();

    if (srcShape4 == 0) {
        return;
    }

    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, DIM_5TH, MAX_DIMS>();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto srcTypeSize = sizeof(typename T1::Type);

    for (LoopVar n0Index = 0; n0Index < dstShape0; n0Index++) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; n1Index++) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; n2Index++) {
                for (LoopVar n3Index = 0; n3Index < dstShape3; n3Index++) {
                    using DstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, 1, dstTileW,
                                                    pto::BLayout::RowMajor, -1, -1>;
                    using SrcTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, 1, srcTileW,
                                                    pto::BLayout::RowMajor, -1, -1>;
                    DstTileDefine dstTile(1, dstShape4);
                    SrcTileDefine srcTile(1, srcShape4);
                    auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2 +
                                     n3Index * dstStride3;
                    auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2 +
                                     n3Index * srcStride3;
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                    constexpr auto pattern = (extractMode == 0) ? pto::MaskPattern::P0101 : pto::MaskPattern::P1010;
                    pto::TGATHER<DstTileDefine, SrcTileDefine, pattern>(dstTile, srcTile);
                    SyncV();

                    if constexpr (extractMode == 0 && isLargest == 0) {
                        constexpr int32_t INT32_MIN_VALUE = static_cast<int32_t>(0x80000000U);
                        using DstAddTileDefine = pto::Tile<pto::TileType::Vec, int32_t, 1, dstTileW,
                                                           pto::BLayout::RowMajor, -1, -1>;
                        DstAddTileDefine dstAddTile(1, dstShape4);
                        pto::TASSIGN(dstAddTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                        pto::TADDS(dstAddTile, dstAddTile, INT32_MIN_VALUE);
                        SyncV();
                    }
                }
            }
        }
    }
}

#endif
