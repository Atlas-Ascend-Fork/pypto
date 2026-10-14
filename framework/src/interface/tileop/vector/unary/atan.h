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
 * \file atan.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_ATAN__H
#define TILEOP_TILE_OPERATOR_ATAN__H
#include "../pto_tile.h"
#include "utils/sync.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#include <cmath>

namespace AtanConstants {
constexpr int64_t NUM_VALUE_3 = 3;
constexpr int64_t NUM_VALUE_4 = 4;
constexpr int64_t NUM_VALUE_8 = 8;
constexpr int64_t NUM_VALUE_16 = 16;
constexpr int64_t NUM_VALUE_32 = 32;
} // namespace AtanConstants

template <typename DST, typename SRC, typename TMP1, typename TMP2, typename CMP>
TILEOP void AtanCalc(DST dst, SRC src, TMP1 tmp1, TMP2 tmp2, CMP cmp)
{
    constexpr float a[] = {-0.333329409,  0.199887753,  -0.141718030,  0.105184801,
                           -0.0725297481, 0.0398497507, -0.0143969795, 0.00245002890};
    constexpr int POLY_LAST_INDEX = 7;
    constexpr int POLY_SECOND_LAST_INDEX = POLY_LAST_INDEX - 1;
    constexpr int HORNER_START_INDEX = POLY_SECOND_LAST_INDEX - 1;
    constexpr float pi2 = 1.570796326794896619;
    pto::TABS(tmp1, src);
    pto::TEXPANDS(dst, 1.0);
    SyncV();
    pto::TDIV(tmp2, dst, tmp1);
    pto::TCMPS(cmp, tmp1, 1.0, pto::CmpMode::GT);
    SyncV();
    pto::TSEL(tmp2, cmp, tmp2, tmp1, dst);
    SyncV();
    pto::TMUL(tmp1, tmp2, tmp2);
    SyncV();
    pto::TMULS(dst, tmp1, a[POLY_LAST_INDEX]);
    SyncV();
    pto::TADDS(dst, dst, a[POLY_SECOND_LAST_INDEX]);
    SyncV();
    for (int i = HORNER_START_INDEX; i >= 0; --i) {
        pto::TMUL(dst, dst, tmp1);
        SyncV();
        pto::TADDS(dst, dst, a[i]);
        SyncV();
    }
    pto::TMUL(dst, dst, tmp1);
    SyncV();
    pto::TMUL(dst, dst, tmp2);
    SyncV();
    pto::TADD(dst, dst, tmp2);
    SyncV();
    pto::TNEG(tmp1, dst);
    SyncV();
    pto::TADDS(tmp1, tmp1, pi2);
    SyncV();
    pto::TSEL(dst, cmp, tmp1, dst, tmp2);
    SyncV();
    pto::TNEG(tmp1, dst);
    pto::TCMPS(cmp, src, 0.0, pto::CmpMode::GE);
    SyncV();
    pto::TSEL(dst, cmp, dst, tmp1, tmp2);
    SyncV();
}

template <typename DST>
TILEOP void AtanGetShape(DST dst, size_t dstShape[])
{
    const auto dstLayout = dst.GetLayout();
    dstShape[DIM_1ST] = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    dstShape[DIM_2ND] = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    dstShape[DIM_3RD] = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    dstShape[DIM_4TH] = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    dstShape[DIM_5TH] = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
}

#define OP_TILE_OP_ATAN TAtan
template <typename DST, typename TMP, typename SRC>
TILEOP void TAtan(DST dst, TMP tmp, SRC src)
{
    size_t dstShape[MAX_DIMS];
    AtanGetShape(dst, dstShape);
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<DST, DIM_5TH, MAX_DIMS>();
    constexpr auto cmpTileW = ((tileW + AtanConstants::NUM_VALUE_8 - 1) / AtanConstants::NUM_VALUE_8 +
                               AtanConstants::NUM_VALUE_32 - 1) /
                              AtanConstants::NUM_VALUE_32 * AtanConstants::NUM_VALUE_32;
    auto cmpSize = (dstShape[DIM_5TH] + AtanConstants::NUM_VALUE_8 - 1) / AtanConstants::NUM_VALUE_8;
    using CmpTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, 1, cmpTileW, pto::BLayout::RowMajor, -1, -1>;
    auto dstTile = PtoTile<DST>(dst);
    auto srcExecTile = MakeElementwiseOperandExecTile(dst, src);
    auto tmp1Tile = PtoTile<DST>(dst);
    auto tmp2Tile = PtoTile<DST>(dst);
    CmpTileDefine cmpTile(dstShape[DIM_4TH], cmpSize);
    for (LoopVar n0Index = 0; n0Index < dstShape[DIM_1ST]; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape[DIM_2ND]; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape[DIM_3RD]; ++n2Index) {
                auto dstOffset = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, dstOffset);
                AssignElementwiseOperandExecTile(srcExecTile, src, dstOffset);
                auto tmp1Offset = GenTileOffset(dst, dstOffset) * AtanConstants::NUM_VALUE_3;
                auto tmp2Offset = tmp1Offset + dstShape[DIM_4TH] * tileW;
                auto cmpOffset = tmp2Offset + dstShape[DIM_4TH] * tileW;
                tmp1Tile.Assign(tmp.GetAddr(), tmp1Offset);
                tmp2Tile.Assign(tmp.GetAddr(), tmp2Offset);
                pto::TASSIGN(cmpTile, tmp.GetAddr() + cmpOffset * sizeof(typename DST::Type));
                AtanCalc(dstTile.Data(), srcExecTile, tmp1Tile.Data(), tmp2Tile.Data(), cmpTile);
            }
        }
    }
}

template <typename HDST, typename FSRC, typename UDST, typename UTMP, typename CMP>
TILEOP void Atan2Cast(HDST dstH, FSRC srcF, UDST dstU, UTMP tmpU, CMP cmp)
{
    constexpr uint16_t sign = 0x8000u;
    constexpr uint16_t val = 0x4000u;
    pto::TCVT(dstH, srcF, pto::RoundMode::CAST_NONE);
    SyncV();
    pto::TANDS(tmpU, dstU, sign);
    SyncV();
    pto::TORS(dstU, tmpU, val);
    SyncV();
    pto::TCMPS(cmp, dstH, 0.0, pto::CmpMode::GE);
    SyncV();
}

template <typename DST, typename SRC0, typename SRC1, typename TMP1, typename TMP2, typename TMP3, typename CMP>
TILEOP void Atan2Sp(DST dst, SRC0 src0, SRC1 src1, TMP1 tmp1, TMP2 tmp2, TMP3 tmp3, CMP cmp)
{
    constexpr float pi = 3.14159265358979323;
    constexpr float pi2 = 1.570796326794896619;
    pto::TADDS(tmp2, tmp1, pi);
    pto::TSUBS(tmp3, tmp1, pi);
    SyncV();
    pto::TSEL(tmp2, cmp, tmp2, tmp3, dst);
    SyncV();
    pto::TCMPS(cmp, src1, 0.0, pto::CmpMode::LT);
    SyncV();
    pto::TSEL(dst, cmp, tmp2, tmp1, tmp3);
    SyncV();
    pto::TEXPANDS(tmp1, pi2);
    pto::TEXPANDS(tmp2, -pi2);
    pto::TCMPS(cmp, src0, 0.0, pto::CmpMode::GT);
    SyncV();
    pto::TSEL(tmp1, cmp, tmp1, tmp2, tmp3);
    SyncV();
    pto::TEXPANDS(tmp2, 0.0);
    pto::TCMPS(cmp, src0, 0.0, pto::CmpMode::NE);
    SyncV();
    pto::TSEL(tmp1, cmp, tmp1, tmp2, tmp3);
    SyncV();
    pto::TCMPS(cmp, src1, 0.0, pto::CmpMode::NE);
    SyncV();
    pto::TSEL(dst, cmp, dst, tmp1, tmp3);
    SyncV();
    pto::TEXPANDS(tmp1, NAN);
    pto::TCMP(cmp, src0, src0, pto::CmpMode::EQ);
    SyncV();
    pto::TSEL(dst, cmp, dst, tmp1, tmp3);
    SyncV();
    pto::TEXPANDS(tmp1, NAN);
    pto::TCMP(cmp, src1, src1, pto::CmpMode::EQ);
    SyncV();
    pto::TSEL(dst, cmp, dst, tmp1, tmp3);
    SyncV();
}

template <typename DST, typename SRC0, typename SRC1, typename TMP1, typename TMP2, typename TMP3, typename CMP>
TILEOP void Atan2Div(DST dst, SRC0 src0, SRC1 src1, TMP1 tmp1, TMP2 tmp2, TMP3 tmp3, CMP cmp)
{
    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(dst, src0, src1);
    pto::TCMP(cmp, src0, src1, pto::CmpMode::NE);
    pto::TMULS(tmp1, src0, -1.0);
    pto::TEXPANDS(tmp2, 1.0);
    SyncV();
    pto::TSEL(dst, cmp, dst, tmp2, tmp3);
    SyncV();
    pto::TEXPANDS(tmp2, -1.0);
    pto::TCMP(cmp, tmp1, src1, pto::CmpMode::NE);
    SyncV();
    pto::TSEL(dst, cmp, dst, tmp2, tmp3);
    SyncV();
}

#define OP_TILE_OP_ATAN2 TAtan2
template <typename DST, typename SRC0, typename SRC1, typename TMP>
TILEOP void TAtan2(DST dst, SRC0 src0, SRC1 src1, TMP tmp)
{
    size_t dstShape[MAX_DIMS];
    AtanGetShape(dst, dstShape);
    constexpr size_t dstDtypeSize = sizeof(typename DST::Type);
    constexpr auto tileH = TileOp::GetTensorTileShapeDim<DST, DIM_4TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<DST, DIM_5TH, MAX_DIMS>();
    constexpr auto cmpTileW = ((tileW + AtanConstants::NUM_VALUE_8 - 1) / AtanConstants::NUM_VALUE_8 +
                               AtanConstants::NUM_VALUE_32 - 1) /
                              AtanConstants::NUM_VALUE_32 * AtanConstants::NUM_VALUE_32;
    constexpr auto b2TileW = (tileW + AtanConstants::NUM_VALUE_16 - 1) / AtanConstants::NUM_VALUE_16 *
                             AtanConstants::NUM_VALUE_16;
    using CmpTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, cmpTileW, pto::BLayout::RowMajor, -1, -1>;
    using UIntTileDefine = pto::Tile<pto::TileType::Vec, uint16_t, tileH, b2TileW, pto::BLayout::RowMajor, -1, -1>;
    using HalfTileDefine = pto::Tile<pto::TileType::Vec, half, tileH, b2TileW, pto::BLayout::RowMajor, -1, -1>;
    auto dstTile = PtoTile<DST>(dst);
    auto src0ExecTile = MakeElementwiseOperandExecTile(dst, src0);
    auto src1ExecTile = MakeElementwiseOperandExecTile(dst, src1);
    auto tmp1Tile = PtoTile<DST>(dst);
    auto tmp2Tile = PtoTile<DST>(dst);
    auto tmp3Tile = PtoTile<DST>(dst);
    CmpTileDefine cmpTile(dstShape[DIM_4TH],
                          (dstShape[DIM_5TH] + AtanConstants::NUM_VALUE_8 - 1) / AtanConstants::NUM_VALUE_8);
    UIntTileDefine dstUIntTile(dstShape[DIM_4TH], dstShape[DIM_5TH]);
    UIntTileDefine tmp2UIntTile(dstShape[DIM_4TH], dstShape[DIM_5TH]);
    HalfTileDefine dstHalfTile(dstShape[DIM_4TH], dstShape[DIM_5TH]);
    for (LoopVar n0Index = 0; n0Index < dstShape[DIM_1ST]; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape[DIM_2ND]; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape[DIM_3RD]; ++n2Index) {
                auto dstOffset = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, dstOffset);
                AssignElementwiseOperandExecTile(src0ExecTile, src0, dstOffset);
                AssignElementwiseOperandExecTile(src1ExecTile, src1, dstOffset);
                auto tileOffset = GenTileOffset(dst, dstOffset);
                pto::TASSIGN(dstUIntTile, dst.GetAddr() + tileOffset * dstDtypeSize);
                pto::TASSIGN(dstHalfTile, dst.GetAddr() + tileOffset * dstDtypeSize);
                auto tmp1Offset = tileOffset * AtanConstants::NUM_VALUE_4;
                auto tmp2Offset = tmp1Offset + dstShape[DIM_4TH] * tileW;
                auto tmp3Offset = tmp2Offset + dstShape[DIM_4TH] * tileW;
                auto cmpOffset = tmp3Offset + dstShape[DIM_4TH] * tileW;
                tmp1Tile.Assign(tmp.GetAddr(), tmp1Offset);
                tmp2Tile.Assign(tmp.GetAddr(), tmp2Offset);
                tmp3Tile.Assign(tmp.GetAddr(), tmp3Offset);
                pto::TASSIGN(tmp2UIntTile, tmp.GetAddr() + tmp2Offset * dstDtypeSize);
                pto::TASSIGN(cmpTile, tmp.GetAddr() + cmpOffset * dstDtypeSize);
                Atan2Div(dstTile.Data(), src0ExecTile, src1ExecTile, tmp1Tile.Data(), tmp2Tile.Data(), tmp3Tile.Data(),
                         cmpTile);
                AtanCalc(tmp1Tile.Data(), dstTile.Data(), tmp2Tile.Data(), tmp3Tile.Data(), cmpTile);
                Atan2Cast(dstHalfTile, src0ExecTile, dstUIntTile, tmp2UIntTile, cmpTile);
                Atan2Sp(dstTile.Data(), src0ExecTile, src1ExecTile, tmp1Tile.Data(), tmp2Tile.Data(), tmp3Tile.Data(),
                        cmpTile);
            }
        }
    }
}

#endif
