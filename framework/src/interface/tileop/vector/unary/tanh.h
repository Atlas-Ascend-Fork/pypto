/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tanh.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_TANH__H
#define TILEOP_TILE_OPERATOR_TANH__H
#include "basic.h"
#include "utils/sync.h"
#include <type_traits>

constexpr float TANH_POLY_015 = 0.0157396831f;
constexpr float TANH_POLY_NEG_052 = -0.0523039624f;
constexpr float TANH_POLY_133 = 0.133152977f;
constexpr float TANH_POLY_NEG_0333 = -0.333327681f;
constexpr float TANH_THRESHOLD = 0.55f;
constexpr float TANH_CLIP_VALUE = 20.0f;
constexpr float TANH_TWO = 2.0f;

template <typename LastUse, typename T, typename DstTile, typename SrcTile, typename TmpTile, typename CmpTile,
          typename AddrUBTile>
TILEOP void TanhFP32(DstTile dstTile, SrcTile srcTile, TmpTile tmpTile, TmpTile tmpTile2, CmpTile cmpTile,
                     AddrUBTile startAddrUBTile)
{
    pto::TMUL(tmpTile, srcTile, srcTile);
    SyncV();

    pto::TMULS(dstTile, tmpTile, static_cast<T>(TANH_POLY_015));
    SyncV();

    pto::TADDS(dstTile, dstTile, static_cast<T>(TANH_POLY_NEG_052));
    SyncV();

    pto::TMUL(dstTile, dstTile, tmpTile);
    SyncV();
    pto::TADDS(dstTile, dstTile, static_cast<T>(TANH_POLY_133));
    SyncV();

    pto::TMUL(dstTile, dstTile, tmpTile);
    SyncV();
    pto::TADDS(dstTile, dstTile, static_cast<T>(TANH_POLY_NEG_0333));
    SyncV();

    pto::TMUL(dstTile, dstTile, tmpTile);
    SyncV();

    pto::TMUL(dstTile, dstTile, srcTile);
    SyncV();
    pto::TADD(dstTile, dstTile, srcTile);
    SyncV();

    pto::TABS(tmpTile, srcTile);
    SyncV();

    pto::TCMPS(cmpTile, tmpTile, static_cast<T>(TANH_THRESHOLD), pto::CmpMode::LT);
    SyncV();

    pto::TMINS(tmpTile, srcTile, static_cast<T>(TANH_CLIP_VALUE));
    SyncV();

    pto::TMULS(tmpTile, tmpTile, static_cast<T>(TANH_TWO));
    SyncV();

    pto::TEXP(tmpTile, tmpTile);
    SyncV();

    pto::TADDS(tmpTile2, tmpTile, static_cast<T>(-1.0f));
    SyncV();

    pto::TADDS(tmpTile, tmpTile, static_cast<T>(1.0f));
    SyncV();

    pto::TDIV(tmpTile2, tmpTile2, tmpTile);
    SyncV();

    pto::TSEL(dstTile, cmpTile, dstTile, tmpTile2, startAddrUBTile);
}

template <typename LastUse, typename T, typename DstTile, typename SrcTile, typename TmpTile, typename CmpTile,
          typename AddrUBTile>
TILEOP void TanhCast(DstTile dstTile, SrcTile srcTile, TmpTile tmpTile1, TmpTile tmpTile2, TmpTile tmpTile3,
                     TmpTile tmpTile4, CmpTile cmpTile, AddrUBTile startAddrUBTile)
{
    pto::TCVT(tmpTile1, srcTile, pto::RoundMode::CAST_NONE);
    SyncV();

    pto::TMUL(tmpTile2, tmpTile1, tmpTile1);
    SyncV();

    pto::TMULS(tmpTile3, tmpTile2, static_cast<float>(TANH_POLY_015));
    SyncV();

    pto::TADDS(tmpTile3, tmpTile3, static_cast<float>(TANH_POLY_NEG_052));
    SyncV();

    pto::TMUL(tmpTile3, tmpTile3, tmpTile2);
    SyncV();
    pto::TADDS(tmpTile3, tmpTile3, static_cast<float>(TANH_POLY_133));
    SyncV();

    pto::TMUL(tmpTile3, tmpTile3, tmpTile2);
    SyncV();
    pto::TADDS(tmpTile3, tmpTile3, static_cast<float>(TANH_POLY_NEG_0333));
    SyncV();

    pto::TMUL(tmpTile3, tmpTile3, tmpTile2);
    SyncV();

    pto::TMUL(tmpTile3, tmpTile3, tmpTile1);
    SyncV();
    pto::TADD(tmpTile3, tmpTile3, tmpTile1);
    SyncV();

    pto::TABS(tmpTile2, tmpTile1);
    SyncV();

    pto::TMINS(tmpTile1, tmpTile1, static_cast<float>(TANH_CLIP_VALUE));
    SyncV();

    pto::TMULS(tmpTile1, tmpTile1, static_cast<float>(TANH_TWO));
    SyncV();

    pto::TEXP(tmpTile1, tmpTile1);
    SyncV();

    pto::TADDS(tmpTile4, tmpTile1, static_cast<float>(-1.0f));
    SyncV();

    pto::TADDS(tmpTile1, tmpTile1, static_cast<float>(1.0f));
    SyncV();

    pto::TDIV(tmpTile4, tmpTile4, tmpTile1);
    SyncV();

    pto::TCMPS(cmpTile, tmpTile2, static_cast<float>(TANH_THRESHOLD), pto::CmpMode::LT);
    SyncV();

    pto::TSEL(tmpTile3, cmpTile, tmpTile3, tmpTile4, startAddrUBTile);
    SyncV();

    pto::TCVT(dstTile, tmpTile3, pto::RoundMode::CAST_NONE);
}

#define OP_TILE_OP_TANH Ttanh
template <typename LastUse = LastUse2Dim<0, 0>, typename T0, typename T1, typename T3>
TILEOP void TTanh(T0 dst, T1 src, T3 tmp)
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

    using DstTile = pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1,
                              -1>;
    using SrcTile = pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileH, srcTileW, pto::BLayout::RowMajor, -1,
                              -1>;

    DstTile dstTile(dstShape3, dstShape4);
    SrcTile srcTile(srcExecShape3, srcExecShape4);

    using AddrUBTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, TileOp::BLOCK_SIZE, pto::BLayout::RowMajor, -1, -1>;

    constexpr bool isFp32 = std::is_same<typename T0::Type, float>::value;
    constexpr auto computeTileH = isFp32 ? dstTileH : srcTileH;
    constexpr auto computeTileW = isFp32 ? dstTileW : srcTileW;
    constexpr auto alignFp32 = 8;
    AddrUBTile startAddrUBTile(1, TileOp::BLOCK_SIZE / TileOp::BITS_PER_BYTE);
    constexpr auto tmpTileW = (computeTileW + alignFp32 - 1) / alignFp32 * alignFp32;
    constexpr auto cmpTileW = (computeTileW / alignFp32 + TileOp::BLOCK_SIZE - 1) / TileOp::BLOCK_SIZE *
                              TileOp::BLOCK_SIZE;
    auto tmpOffset = dstShape3 * tmpTileW;
    auto cmpOffset = dstShape3 * cmpTileW;
    constexpr size_t FP32_CMP_SLOT = 2;
    constexpr size_t CAST_TMP3_SLOT = 2;
    constexpr size_t CAST_TMP4_SLOT = 3;
    constexpr size_t CAST_CMP_SLOT = 4;
    using TmpTile = pto::Tile<pto::TileType::Vec, float, computeTileH, tmpTileW, pto::BLayout::RowMajor, -1, -1>;
    using CmpTile = pto::Tile<pto::TileType::Vec, uint8_t, computeTileH, cmpTileW, pto::BLayout::RowMajor, -1, -1>;

    if constexpr (isFp32) {
        TmpTile tmpTile1(dstShape3, dstShape4);
        TmpTile tmpTile2(dstShape3, dstShape4);
        CmpTile cmpTile(dstTileH, dstShape4 / TileOp::BITS_PER_BYTE);
        pto::TASSIGN(tmpTile1, (uint64_t)(tmp.GetAddr()));
        pto::TASSIGN(tmpTile2, (uint64_t)(tmp.GetAddr() + tmpOffset * sizeof(float)));
        pto::TASSIGN(cmpTile, (uint64_t)(tmp.GetAddr() + FP32_CMP_SLOT * tmpOffset * sizeof(float)));
        pto::TASSIGN(startAddrUBTile, (uint64_t)(tmp.GetAddr() + FP32_CMP_SLOT * tmpOffset * sizeof(float) +
                                                 cmpOffset * sizeof(uint8_t)));

        for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
            for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
                for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                    auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                    auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                    TanhFP32<LastUse, typename T0::Type, DstTile, SrcTile, TmpTile, CmpTile, AddrUBTile>(
                        dstTile, srcTile, tmpTile1, tmpTile2, cmpTile, startAddrUBTile);
                }
            }
        }
    } else if constexpr (std::is_same<typename T0::Type, half>::value ||
                         std::is_same<typename T0::Type, bfloat16_t>::value) {
        TmpTile tmpTile1(srcExecShape3, srcExecShape4);
        TmpTile tmpTile2(srcExecShape3, srcExecShape4);
        TmpTile tmpTile3(srcExecShape3, srcExecShape4);
        TmpTile tmpTile4(srcExecShape3, srcExecShape4);
        CmpTile cmpTile(srcTileH, srcExecShape4 / TileOp::BITS_PER_BYTE);
        pto::TASSIGN(tmpTile2, (uint64_t)(tmp.GetAddr()));
        pto::TASSIGN(tmpTile1, (uint64_t)(tmp.GetAddr() + tmpOffset * sizeof(float)));
        pto::TASSIGN(tmpTile3, (uint64_t)(tmp.GetAddr() + CAST_TMP3_SLOT * tmpOffset * sizeof(float)));
        pto::TASSIGN(tmpTile4, (uint64_t)(tmp.GetAddr() + CAST_TMP4_SLOT * tmpOffset * sizeof(float)));
        pto::TASSIGN(cmpTile, (uint64_t)(tmp.GetAddr() + CAST_CMP_SLOT * tmpOffset * sizeof(float)));
        pto::TASSIGN(startAddrUBTile, (uint64_t)(tmp.GetAddr() + CAST_CMP_SLOT * tmpOffset * sizeof(float) +
                                                 cmpOffset * sizeof(uint8_t)));

        for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
            for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
                for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                    auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                    auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                    TanhCast<LastUse, typename T0::Type, DstTile, SrcTile, TmpTile, CmpTile, AddrUBTile>(
                        dstTile, srcTile, tmpTile1, tmpTile2, tmpTile3, tmpTile4, cmpTile, startAddrUBTile);
                }
            }
        }
    }
}

#endif
