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
 * \file tan.h
 * \brief tan operator: rational approximation aligned to CANN TBE tan_compute_v2
 *
 * Algorithm: tan(x) = x * P(x^2) / ((x^2 + R3) * (x + pi/2) * (x - pi/2))
 * Range reduction: 4-part Cody-Waite with pi decomposition
 * Polynomial: P(x^2) = (x^2 * R0 + R1) * x^2 + R2
 */

#ifndef TILEOP_TILE_OPERATOR_TAN__H
#define TILEOP_TILE_OPERATOR_TAN__H
#include "utils/layout.h"
#include "utils/sync.h"
#include "utils/tile_tensor.h"
#include <type_traits>

template <typename LastUse, typename DstTile, typename SrcTile, typename TmpTile, typename TmpInt32Tile,
          typename Tmp2Tile, typename Tmp3Tile, typename Tmp4Tile, typename Tmp5Tile, typename Tmp6Tile,
          typename Tmp7Tile, typename Tmp8Tile>
TILEOP void TanImpl(DstTile dstTile, SrcTile srcTile, TmpTile tmpTile, TmpInt32Tile tmpInt32Tile, Tmp2Tile tmp2Tile,
                    Tmp3Tile tmp3Tile, Tmp4Tile tmp4Tile, Tmp5Tile tmp5Tile, Tmp6Tile tmp6Tile, Tmp7Tile tmp7Tile,
                    Tmp8Tile tmp8Tile)
{
    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;

    // Constants matching CANN TBE tan_compute_v2 exactly
    constexpr float INV_PI = 0.3183098733425140380859375f;
    constexpr float PI_V2 = 3.140625f;
    constexpr float KPI_1 = 0.0009670257568359375f;
    constexpr float KPI_2 = 6.2771141529083251953125e-7f;
    constexpr float KPI_3 = 1.21644916362129151821136474609375e-10f;
    constexpr float KPI_4 = -1.0291767438275201129727065563201904296875e-13f;
    constexpr float PI_DOWN = 1.57079637050628662109375f;
    constexpr float PI_DOWN_NEG = -1.57079637050628662109375f;
    constexpr float EPS = 0.00000004371139000189375f;
    constexpr float EPS_NEG = -0.00000004371139000189375f;
    constexpr float R0 = 0.0698520831551998762793f;
    constexpr float R1 = -6.8711573651634203789f;
    constexpr float R2 = 61.20362572811089435388f;
    constexpr float R3 = -24.8048928861126769186219f;

    // k = round(x / pi)
    pto::TMULS(tmpTile, srcTile, INV_PI);
    SyncV();
    pto::TCVT(tmpInt32Tile, tmpTile, pto::RoundMode::CAST_RINT);
    SyncV();
    pto::TCVT(tmpTile, tmpInt32Tile, pto::RoundMode::CAST_RINT);
    SyncV();

    // KPI_0: input_x = x - k * PI_V2
    pto::TMULS(tmp3Tile, tmpTile, PI_V2);
    SyncV();
    pto::TSUB(tmp2Tile, srcTile, tmp3Tile);
    SyncV();

    // KPI_1: input_x -= k * KPI_1
    pto::TMULS(tmp3Tile, tmpTile, KPI_1);
    SyncV();
    pto::TSUB(tmp2Tile, tmp2Tile, tmp3Tile);
    SyncV();

    // res_down1 = input_x + PI_DOWN (first factor: x + pi/2)
    pto::TADDS(tmp4Tile, tmp2Tile, PI_DOWN);
    SyncV();
    // res_down2 = input_x + PI_DOWN_NEG (second factor: x - pi/2)
    pto::TADDS(tmp5Tile, tmp2Tile, PI_DOWN_NEG);
    SyncV();

    // KPI_2: input_x -= k*KPI_2, res_down1 -= k*KPI_2, res_down2 -= k*KPI_2
    pto::TMULS(tmp3Tile, tmpTile, KPI_2);
    SyncV();
    pto::TSUB(tmp2Tile, tmp2Tile, tmp3Tile);
    SyncV();
    pto::TSUB(tmp4Tile, tmp4Tile, tmp3Tile);
    SyncV();
    pto::TSUB(tmp5Tile, tmp5Tile, tmp3Tile);
    SyncV();
    // res_down1 += EPS_NEG, res_down2 += EPS
    pto::TADDS(tmp4Tile, tmp4Tile, EPS_NEG);
    SyncV();
    pto::TADDS(tmp5Tile, tmp5Tile, EPS);
    SyncV();

    // KPI_3: input_x -= k*KPI_3, res_down1 -= k*KPI_3, res_down2 -= k*KPI_3
    pto::TMULS(tmp3Tile, tmpTile, KPI_3);
    SyncV();
    pto::TSUB(tmp2Tile, tmp2Tile, tmp3Tile);
    SyncV();
    pto::TSUB(tmp4Tile, tmp4Tile, tmp3Tile);
    SyncV();
    pto::TSUB(tmp5Tile, tmp5Tile, tmp3Tile);
    SyncV();

    // KPI_4: input_x -= k*KPI_4, res_down1 -= k*KPI_4, res_down2 -= k*KPI_4
    pto::TMULS(tmp3Tile, tmpTile, KPI_4);
    SyncV();
    pto::TSUB(tmp2Tile, tmp2Tile, tmp3Tile);
    SyncV();
    pto::TSUB(tmp4Tile, tmp4Tile, tmp3Tile);
    SyncV();
    pto::TSUB(tmp5Tile, tmp5Tile, tmp3Tile);
    SyncV();

    // Polynomial evaluation
    // x2 = input_x * input_x
    pto::TMUL(tmp6Tile, tmp2Tile, tmp2Tile);
    SyncV();

    // Numerator: res_up = ((x2 * R0 + R1) * x2 + R2) * input_x
    pto::TMULS(tmp7Tile, tmp6Tile, R0);
    SyncV();
    pto::TADDS(tmp7Tile, tmp7Tile, R1);
    SyncV();
    pto::TMUL(tmp7Tile, tmp7Tile, tmp6Tile);
    SyncV();
    pto::TADDS(tmp7Tile, tmp7Tile, R2);
    SyncV();
    pto::TMUL(tmp7Tile, tmp7Tile, tmp2Tile);
    SyncV();

    // Denominator: res_down = (x2 + R3) * res_down1 * res_down2
    pto::TADDS(tmp8Tile, tmp6Tile, R3);
    SyncV();
    pto::TMUL(tmp8Tile, tmp8Tile, tmp4Tile);
    SyncV();
    pto::TMUL(tmp8Tile, tmp8Tile, tmp5Tile);
    SyncV();

    // Result: tan(x) = numerator / denominator
    pto::TDIV(dstTile, tmp7Tile, tmp8Tile);
    SyncV();
}

template <typename LastUse, typename DstTile, typename SrcTile, typename TmpTile, typename TmpFp32Tile,
          typename TmpInt32Tile, typename Tmp2Tile, typename Tmp3Tile, typename Tmp4Tile, typename Tmp5Tile,
          typename Tmp6Tile, typename Tmp7Tile, typename Tmp8Tile>
TILEOP void TanHalfImpl(DstTile dstTile, SrcTile srcTile, TmpTile tmpTile, TmpFp32Tile tmpFp32Tile,
                        TmpInt32Tile tmpInt32Tile, Tmp2Tile tmp2Tile, Tmp3Tile tmp3Tile, Tmp4Tile tmp4Tile,
                        Tmp5Tile tmp5Tile, Tmp6Tile tmp6Tile, Tmp7Tile tmp7Tile, Tmp8Tile tmp8Tile)
{
    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;

    pto::TCVT(tmpFp32Tile, srcTile, pto::RoundMode::CAST_NONE);
    SyncV();
    TanImpl<LastUse, TmpFp32Tile, TmpFp32Tile, TmpFp32Tile, TmpInt32Tile, Tmp2Tile, Tmp3Tile, Tmp4Tile, Tmp5Tile,
            Tmp6Tile, Tmp7Tile, Tmp8Tile>(tmpFp32Tile, tmpFp32Tile, tmpFp32Tile, tmpInt32Tile, tmp2Tile, tmp3Tile,
                                          tmp4Tile, tmp5Tile, tmp6Tile, tmp7Tile, tmp8Tile);
    SyncV();
    pto::TCVT(dstTile, tmpFp32Tile, pto::RoundMode::CAST_RINT);
    SyncV();
}

#define OP_TILE_OP_TAN Ttan
template <typename LastUse = LastUse2Dim<0, 0>, typename T0, typename T1, typename T2>
TILEOP void Ttan(T0 dst, T1 src, T2 tmp)
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

    using DstTile = pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1,
                              -1>;
    using SrcTile = pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileH, srcTileW, pto::BLayout::RowMajor, -1,
                              -1>;
    using TmpFp32Tile = pto::Tile<pto::TileType::Vec, float, srcTileH, tmpTileW32Bit, pto::BLayout::RowMajor, -1, -1>;
    using TmpInt32Tile = pto::Tile<pto::TileType::Vec, int32_t, srcTileH, tmpTileW32Bit, pto::BLayout::RowMajor, -1,
                                   -1>;
    using Tmp2Tile = pto::Tile<pto::TileType::Vec, float, srcTileH, tmpTileW32Bit, pto::BLayout::RowMajor, -1, -1>;
    using Tmp3Tile = pto::Tile<pto::TileType::Vec, float, srcTileH, tmpTileW32Bit, pto::BLayout::RowMajor, -1, -1>;
    using Tmp4Tile = pto::Tile<pto::TileType::Vec, float, srcTileH, tmpTileW32Bit, pto::BLayout::RowMajor, -1, -1>;
    using Tmp5Tile = pto::Tile<pto::TileType::Vec, float, srcTileH, tmpTileW32Bit, pto::BLayout::RowMajor, -1, -1>;
    using Tmp6Tile = pto::Tile<pto::TileType::Vec, float, srcTileH, tmpTileW32Bit, pto::BLayout::RowMajor, -1, -1>;
    using Tmp7Tile = pto::Tile<pto::TileType::Vec, float, srcTileH, tmpTileW32Bit, pto::BLayout::RowMajor, -1, -1>;
    using Tmp8Tile = pto::Tile<pto::TileType::Vec, float, srcTileH, tmpTileW32Bit, pto::BLayout::RowMajor, -1, -1>;

    DstTile dstTile(dstShape3, dstShape4);
    SrcTile srcTile(srcExecShape3, srcExecShape4);
    TmpFp32Tile tmpFp32Tile(srcExecShape3, srcExecShape4);
    TmpInt32Tile tmpInt32Tile(srcExecShape3, srcExecShape4);
    Tmp2Tile tmp2Tile(srcExecShape3, srcExecShape4);
    Tmp3Tile tmp3Tile(srcExecShape3, srcExecShape4);
    Tmp4Tile tmp4Tile(srcExecShape3, srcExecShape4);
    Tmp5Tile tmp5Tile(srcExecShape3, srcExecShape4);
    Tmp6Tile tmp6Tile(srcExecShape3, srcExecShape4);
    Tmp7Tile tmp7Tile(srcExecShape3, srcExecShape4);
    Tmp8Tile tmp8Tile(srcExecShape3, srcExecShape4);

    auto tmpFp32TileSize = dstShape3 * tmpTileW32Bit * sizeof(float);
    constexpr size_t TMP2_SLOT = 2;
    constexpr size_t TMP3_SLOT = 3;
    constexpr size_t TMP4_SLOT = 4;
    constexpr size_t TMP5_SLOT = 5;
    constexpr size_t TMP6_SLOT = 6;
    constexpr size_t TMP7_SLOT = 7;
    constexpr size_t TMP8_SLOT = 8;
    pto::TASSIGN(tmpFp32Tile, (uint64_t)(tmp.GetAddr()));
    pto::TASSIGN(tmpInt32Tile, (uint64_t)(tmp.GetAddr() + tmpFp32TileSize));
    pto::TASSIGN(tmp2Tile, (uint64_t)(tmp.GetAddr() + TMP2_SLOT * tmpFp32TileSize));
    pto::TASSIGN(tmp3Tile, (uint64_t)(tmp.GetAddr() + TMP3_SLOT * tmpFp32TileSize));
    pto::TASSIGN(tmp4Tile, (uint64_t)(tmp.GetAddr() + TMP4_SLOT * tmpFp32TileSize));
    pto::TASSIGN(tmp5Tile, (uint64_t)(tmp.GetAddr() + TMP5_SLOT * tmpFp32TileSize));
    pto::TASSIGN(tmp6Tile, (uint64_t)(tmp.GetAddr() + TMP6_SLOT * tmpFp32TileSize));
    pto::TASSIGN(tmp7Tile, (uint64_t)(tmp.GetAddr() + TMP7_SLOT * tmpFp32TileSize));
    pto::TASSIGN(tmp8Tile, (uint64_t)(tmp.GetAddr() + TMP8_SLOT * tmpFp32TileSize));

    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));

                if constexpr (std::is_same<typename T1::Type, float>::value) {
                    TanImpl<LastUse, DstTile, SrcTile, TmpFp32Tile, TmpInt32Tile, Tmp2Tile, Tmp3Tile, Tmp4Tile,
                            Tmp5Tile, Tmp6Tile, Tmp7Tile, Tmp8Tile>(dstTile, srcTile, tmpFp32Tile, tmpInt32Tile,
                                                                    tmp2Tile, tmp3Tile, tmp4Tile, tmp5Tile, tmp6Tile,
                                                                    tmp7Tile, tmp8Tile);
                } else if constexpr (std::is_same<typename T1::Type, half>::value ||
                                     std::is_same<typename T1::Type, bfloat16_t>::value) {
                    TanHalfImpl<LastUse, DstTile, SrcTile, TmpFp32Tile, TmpFp32Tile, TmpInt32Tile, Tmp2Tile, Tmp3Tile,
                                Tmp4Tile, Tmp5Tile, Tmp6Tile, Tmp7Tile, Tmp8Tile>(
                        dstTile, srcTile, tmpFp32Tile, tmpFp32Tile, tmpInt32Tile, tmp2Tile, tmp3Tile, tmp4Tile,
                        tmp5Tile, tmp6Tile, tmp7Tile, tmp8Tile);
                }
            }
        }
    }
}

#endif
