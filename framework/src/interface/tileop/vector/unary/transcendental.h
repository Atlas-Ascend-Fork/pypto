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
 * \file transcendental.h
 * \brief Unary tile operation implementations.
 */

#ifndef TILEOP_TILE_OPERATOR_VEC_UNARY_TRANSCENDENTAL_H
#define TILEOP_TILE_OPERATOR_VEC_UNARY_TRANSCENDENTAL_H

#include "utils/sync.h"
#include "basic.h"

#define OP_TILE_OP_LOG TLog
template <auto PrecisionType = pto::LogAlgorithm::DEFAULT, typename T0, typename T1>
TILEOP void TLog(T0 dst, T1 src)
{
    UnaryCompute<UnaryOp::LN, PrecisionType, LastUse2Dim<0, 0>>(dst, src);
}

#define OP_TILE_OP_EXP2 TExp2
template <typename T0, typename T1, typename T2, typename T3>
TILEOP void TExp2(T0 dst, T1 tmp, T2 tmp2, T3 src)
{
    constexpr float EXP2_BASE = 2.0f;
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto tmpTile = PtoTile<T1>(tmp);
    auto tmpTile2 = PtoTile<T2>(tmp2);
    auto srcExecTile = MakeElementwiseOperandExecTile(dst, src);
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                tmpTile.Assign(tmp, tileOffsets);
                tmpTile2.Assign(tmp2, tileOffsets);
                AssignElementwiseOperandExecTile(srcExecTile, src, tileOffsets);

                if constexpr (std::is_same_v<typename T3::Type, float>) {
                    pto::TEXPANDS(tmpTile2.Data(), EXP2_BASE);
                    SyncV();
                    pto::TLOG(tmpTile2.Data(), tmpTile2.Data());
                    SyncV();
                    pto::TMUL(tmpTile2.Data(), srcExecTile, tmpTile2.Data());
                    SyncV();
                    pto::TEXP(dstTile.Data(), tmpTile2.Data());
                    SyncV();
                } else {
                    pto::TCVT(tmpTile.Data(), srcExecTile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TEXPANDS(tmpTile2.Data(), EXP2_BASE);
                    SyncV();
                    pto::TLOG(tmpTile2.Data(), tmpTile2.Data());
                    SyncV();
                    pto::TMUL(tmpTile.Data(), tmpTile.Data(), tmpTile2.Data());
                    SyncV();
                    if constexpr (std::is_same_v<typename T3::Type, half> ||
                                  std::is_same_v<typename T3::Type, bfloat16_t>) {
                        pto::TEXP(tmpTile2.Data(), tmpTile.Data());
                        SyncV();
                        pto::TCVT(dstTile.Data(), tmpTile2.Data(), pto::RoundMode::CAST_RINT);
                    } else {
                        pto::TEXP(dstTile.Data(), tmpTile.Data());
                    }
                }
            }
        }
    }
}

#define OP_TILE_OP_EXPM1 TExpm1
template <typename T0, typename T1, typename T2>
TILEOP void TExpm1(T0 dst, T1 tmp, T2 src)
{
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto tmpTile = PtoTile<T1>(tmp);
    auto srcExecTile = MakeElementwiseOperandExecTile(dst, src);
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                tmpTile.Assign(tmp, tileOffsets);
                AssignElementwiseOperandExecTile(srcExecTile, src, tileOffsets);

                if constexpr (std::is_same_v<typename T2::Type, float>) {
                    pto::TEXP(dstTile.Data(), srcExecTile);
                    SyncV();
                    pto::TADDS(dstTile.Data(), dstTile.Data(), -1.0f);
                } else {
                    pto::TCVT(tmpTile.Data(), srcExecTile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TEXP(tmpTile.Data(), tmpTile.Data());
                    SyncV();
                    if constexpr (std::is_same_v<typename T2::Type, half> ||
                                  std::is_same_v<typename T2::Type, bfloat16_t>) {
                        pto::TADDS(tmpTile.Data(), tmpTile.Data(), -1.0f);
                        SyncV();
                        pto::TCVT(dstTile.Data(), tmpTile.Data(), pto::RoundMode::CAST_RINT);
                    } else {
                        pto::TADDS(dstTile.Data(), tmpTile.Data(), -1.0f);
                    }
                }
            }
        }
    }
}

#ifdef __DAV_V220
template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
TILEOP void TSinhCompute(T0 dstTile, T1 srcTile, T2 tmp0Tile, T3 tmp1Tile, T4 tmp2Tile, T5 tmp3Tile, T6 tmp1MaskTile)
{
    constexpr float SCALAR_ZERO_0199 = 0.0001998459335617813754003f;
    constexpr float SCALAR_ZERO_0833 = 0.00833308538698833f;
    constexpr float SCALAR_ZERO_166 = 0.16666668254541f;
    constexpr float SCALAR_ZERO_48 = 0.48f;
    constexpr float SCALAR_ONE = 1.0f;
    constexpr float SCALAR_ZERO_POINT_FIVE = 0.5f;
    constexpr float SCALAR_NEGATIVE_15 = -1.5f;
    constexpr float SCALAR_NEGATIVE_ONE = -1.0f;
    constexpr float SCALAR_ZERO = 0.0f;

    // sinh(x) = x + x^3 / 3! + x^5 / 5! + x^7 / 7! for small x
    pto::TABS(tmp0Tile, srcTile);
    SyncV();
    pto::TMUL(tmp1Tile, tmp0Tile, tmp0Tile);
    SyncV();
    pto::TMULS(tmp2Tile, tmp1Tile, SCALAR_ZERO_0199);
    SyncV();
    pto::TADDS(tmp2Tile, tmp2Tile, SCALAR_ZERO_0833);
    SyncV();
    pto::TMUL(tmp2Tile, tmp2Tile, tmp1Tile);
    SyncV();
    pto::TADDS(tmp2Tile, tmp2Tile, SCALAR_ZERO_166);
    SyncV();
    pto::TMUL(tmp2Tile, tmp2Tile, tmp1Tile);
    SyncV();
    pto::TADDS(tmp2Tile, tmp2Tile, SCALAR_ONE);
    SyncV();
    pto::TMUL(tmp2Tile, tmp2Tile, tmp0Tile);
    SyncV();

    // sinh(x) = 1/2 * (e^{x/2} - e^{-3x/2}) * e^{x/2} for large x
    pto::TMULS(tmp1Tile, tmp0Tile, SCALAR_ZERO_POINT_FIVE);
    SyncV();
    pto::TEXP<pto::ExpAlgorithm::HIGH_PRECISION>(tmp1Tile, tmp1Tile);
    SyncV();
    pto::TMULS(tmp3Tile, tmp0Tile, SCALAR_NEGATIVE_15);
    SyncV();
    pto::TEXP<pto::ExpAlgorithm::HIGH_PRECISION>(tmp3Tile, tmp3Tile);
    SyncV();
    pto::TSUB(tmp3Tile, tmp1Tile, tmp3Tile);
    SyncV();
    pto::TMULS(tmp3Tile, tmp3Tile, SCALAR_ZERO_POINT_FIVE);
    SyncV();
    pto::TMUL(tmp3Tile, tmp3Tile, tmp1Tile);
    SyncV();

    pto::TCMPS(tmp1MaskTile, tmp0Tile, SCALAR_ZERO_48, pto::CmpMode::LT);
    SyncV();
    pto::TSEL(dstTile, tmp1MaskTile, tmp2Tile, tmp3Tile, tmp0Tile);
    SyncV();

    pto::TMULS(tmp2Tile, dstTile, SCALAR_NEGATIVE_ONE);
    SyncV();
    pto::TCMPS(tmp1MaskTile, srcTile, SCALAR_ZERO, pto::CmpMode::GE);
    SyncV();
    pto::TSEL(dstTile, tmp1MaskTile, dstTile, tmp2Tile, tmp0Tile);
    SyncV();
}
#else
template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
TILEOP void TSinhCompute(T0 dstTile, T1 srcTile, T2 tmp0Tile, T3 tmp1Tile, T4 tmp2Tile, T5 tmp3Tile, T6 tmp3MaskTile)
{
    constexpr float kC2 = 0.16666667f;
    constexpr float kC3 = 0.008333347f;
    constexpr float kC4 = 0.00019841270f;
    constexpr float kC5 = 0.0000027557319f;
    constexpr float kNegLn2Hi = -0.6931471824645996f;
    constexpr float kLn2Lo = 1.9046542e-9f;
    constexpr float kOverflow = 89.4159927368164f;
    constexpr float kInf = std::numeric_limits<float>::infinity();

    // Recover the rounding residual of shifted = |x| - ln2_hi.
    pto::TABS(tmp0Tile, srcTile);
    pto::TADDS(tmp1Tile, tmp0Tile, kNegLn2Hi);
    pto::TSUB(tmp2Tile, tmp1Tile, tmp0Tile); // z = shifted - |x|
    pto::TSUB(tmp3Tile, tmp1Tile, tmp2Tile);
    pto::TSUB(tmp3Tile, tmp0Tile, tmp3Tile);
    pto::TMULS(tmp2Tile, tmp2Tile, -1.0f);
    pto::TADDS(tmp2Tile, tmp2Tile, kNegLn2Hi);
    pto::TADD(tmp3Tile, tmp3Tile, tmp2Tile);
    pto::TADDS(tmp3Tile, tmp3Tile, kLn2Lo);
    pto::TEXP(tmp1Tile, tmp1Tile);
    // t = exp(shifted) * (1 + residual), with separate multiply/add rounding.
    pto::TMUL(tmp2Tile, tmp1Tile, tmp3Tile);
    pto::TADD(tmp2Tile, tmp1Tile, tmp2Tile);
    // The residual is dead; its slot now stores the mask. Discard corrections
    // on overflowing / NaN lanes, as in the reference implementation.
    pto::TCMPS(tmp3MaskTile, tmp0Tile, kOverflow, pto::CmpMode::LT);
    pto::TSEL(tmp1Tile, tmp3MaskTile, tmp2Tile, tmp1Tile, dstTile);
    pto::TEXPANDS(tmp2Tile, 0.25f);
    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(tmp2Tile, tmp2Tile, tmp1Tile);
    pto::TSUB(tmp1Tile, tmp1Tile, tmp2Tile);

    // Guard overflow before restoring the sign of the large-argument result.
    pto::TCMPS(tmp3MaskTile, tmp0Tile, kOverflow, pto::CmpMode::GE);
    pto::TEXPANDS(tmp2Tile, kInf);
    pto::TSEL(tmp1Tile, tmp3MaskTile, tmp2Tile, tmp1Tile, dstTile);
    pto::TCMPS(tmp3MaskTile, srcTile, 0.0f, pto::CmpMode::LT);
    pto::TMULS(tmp2Tile, tmp1Tile, -1.0f);
    pto::TSEL(tmp1Tile, tmp3MaskTile, tmp2Tile, tmp1Tile, dstTile);

    // For |x| < 1, use x + x^3 * (((c5*x^2 + c4)*x^2 + c3)*x^2 + c2).
    // Keep the signed input and the reference operation order, including -0.
    pto::TMUL(tmp0Tile, srcTile, srcTile);
    pto::TEXPANDS(dstTile, kC5);
    pto::TMUL(dstTile, dstTile, tmp0Tile);
    pto::TADDS(dstTile, dstTile, kC4);
    pto::TMUL(dstTile, dstTile, tmp0Tile);
    pto::TADDS(dstTile, dstTile, kC3);
    pto::TMUL(dstTile, dstTile, tmp0Tile);
    pto::TADDS(dstTile, dstTile, kC2);
    pto::TMUL(tmp2Tile, tmp0Tile, srcTile);
    pto::TMUL(dstTile, dstTile, tmp2Tile);
    pto::TADD(dstTile, srcTile, dstTile);
    pto::TABS(tmp0Tile, srcTile);
    pto::TCMPS(tmp3MaskTile, tmp0Tile, 1.0f, pto::CmpMode::LT);
    pto::TSEL(dstTile, tmp3MaskTile, dstTile, tmp1Tile, tmp2Tile);
}
#endif

#define OP_TILE_OP_SINH TSinh
template <typename T0, typename T1, typename T2>
TILEOP void TSinh(T0 dst, T1 src, T2 tmp)
{
    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    constexpr auto tileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);

    auto tileShapeSize = dstShape0 * dstShape1 * dstShape2 * dstShape3 * tileW;

    using DstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, tileH, tileW, pto::BLayout::RowMajor, -1,
                                    -1>;
    using MaskTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, tileW * 4, pto::BLayout::RowMajor, -1, -1>;
    DstTileDefine dstTile(dstShape3, dstShape4);
    auto srcExecTile = MakeElementwiseOperandExecTile(dst, src);
    DstTileDefine tmp0Tile(dstShape3, dstShape4);
    DstTileDefine tmp1Tile(dstShape3, dstShape4);
    DstTileDefine tmp2Tile(dstShape3, dstShape4);
    DstTileDefine tmp3Tile(dstShape3, dstShape4);
#ifdef __DAV_V220
    MaskTileDefine tmp1MaskTile(dstShape3, dstShape4);
#else
    MaskTileDefine tmp3MaskTile(dstShape3, dstShape4);
#endif
    constexpr size_t TMP2_SLOT = 2;
    constexpr size_t TMP3_SLOT = 3;

    for (LoopVar n0Index = 0; n0Index < dstShape0; n0Index++) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; n1Index++) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; n2Index++) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto dstOffset = GenTileOffset(dst, tileOffsets);
                auto tmpByteOffset = TileOp::GetPackedByteOffset<typename T2::Type>(GenTileOffset(tmp, tileOffsets));
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                AssignElementwiseOperandExecTile(srcExecTile, src, tileOffsets);

                pto::TASSIGN(tmp0Tile, (uint64_t)(tmp.GetAddr() + tmpByteOffset));
                pto::TASSIGN(tmp1Tile, (uint64_t)(tmp.GetAddr() + tmpByteOffset + tileShapeSize * dstTypeSize));
                pto::TASSIGN(tmp2Tile,
                             (uint64_t)(tmp.GetAddr() + tmpByteOffset + TMP2_SLOT * tileShapeSize * dstTypeSize));
                pto::TASSIGN(tmp3Tile,
                             (uint64_t)(tmp.GetAddr() + tmpByteOffset + TMP3_SLOT * tileShapeSize * dstTypeSize));
#ifdef __DAV_V220
                pto::TASSIGN(tmp1MaskTile, (uint64_t)(tmp.GetAddr() + tmpByteOffset + tileShapeSize * dstTypeSize));

                TSinhCompute(dstTile, srcExecTile, tmp0Tile, tmp1Tile, tmp2Tile, tmp3Tile, tmp1MaskTile);
#else
                pto::TASSIGN(tmp3MaskTile,
                             (uint64_t)(tmp.GetAddr() + tmpByteOffset + TMP3_SLOT * tileShapeSize * dstTypeSize));
                TSinhCompute(dstTile, srcExecTile, tmp0Tile, tmp1Tile, tmp2Tile, tmp3Tile, tmp3MaskTile);
#endif
            }
        }
    }
}

#ifdef __DAV_V220
template <typename T0, typename T1, typename T2>
TILEOP void TCoshCompute(T0 dstTile, T1 srcTile, T2 tmpTile)
{
    constexpr float SCALAR_ZERO_POINT_FIVE = 0.5f;
    constexpr float SCALAR_NEGATIVE_ONE_POINT_FIVE = -1.5f;

    // cosh(x) = 0.5 * (exp(x / 2) + exp(-3 * x / 2)) * exp(x / 2)
    pto::TABS(tmpTile, srcTile);
    SyncV();
    pto::TMULS(dstTile, tmpTile, SCALAR_NEGATIVE_ONE_POINT_FIVE);
    SyncV();
    pto::TMULS(tmpTile, tmpTile, SCALAR_ZERO_POINT_FIVE);
    SyncV();
    pto::TEXP<pto::ExpAlgorithm::HIGH_PRECISION>(tmpTile, tmpTile);
    SyncV();
    pto::TEXP<pto::ExpAlgorithm::HIGH_PRECISION>(dstTile, dstTile);
    SyncV();
    pto::TADD(dstTile, dstTile, tmpTile);
    SyncV();
    pto::TMULS(dstTile, dstTile, SCALAR_ZERO_POINT_FIVE);
    SyncV();
    pto::TMUL(dstTile, dstTile, tmpTile);
    SyncV();
}
#else
template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
TILEOP void TCoshCompute(T0 dstTile, T1 srcTile, T2 tmp0Tile, T3 tmp1Tile, T4 tmp2Tile, T5 tmp0IntTile, T6 tmp2MaskTile)
{
    constexpr float kLog2e = 1.442695041f;
    constexpr float kNegLn2Hi = -0.6931471825f;
    constexpr float kLn2Lo = 1.9046542e-9f;
    constexpr float kExpMagic = 12583037.0f;
    constexpr int32_t kFp32MantissaBits = 23;
    constexpr float kNClamp = 126.0f;
    constexpr float kHalfInv8 = 0.125f;
    constexpr float kTwo = 2.0f;
    constexpr float kOvfThreshold = 90.0f;
    constexpr float kInf = std::numeric_limits<float>::infinity();

    // ax = abs(x), temporarily stored in dstTile.
    pto::TABS(dstTile, srcTile);
    // n = trunc(ax * log2(e))
    pto::TMULS(tmp0Tile, dstTile, kLog2e);
    pto::TCVT(tmp0Tile, tmp0Tile, pto::RoundMode::CAST_TRUNC);
    // n = min(n, 126)
    pto::TMINS(tmp0Tile, tmp0Tile, kNClamp);
    // r = ax
    pto::TADDS(tmp1Tile, dstTile, 0.0f);
    // r += n * (-ln2_hi)
    pto::TMULS(tmp2Tile, tmp0Tile, 0.0f);
    pto::TADDS(tmp2Tile, tmp2Tile, kNegLn2Hi);
    pto::TMULA(tmp1Tile, tmp0Tile, tmp2Tile);
    // r += n * (+ln2_lo)
    pto::TMULS(tmp2Tile, tmp2Tile, 0.0f);
    pto::TADDS(tmp2Tile, tmp2Tile, kLn2Lo);
    pto::TMULA(tmp1Tile, tmp0Tile, tmp2Tile);
    // p2 = 2^(n - 2)
    pto::TADDS(tmp0Tile, tmp0Tile, kExpMagic);
    pto::TSHLS(tmp0IntTile, tmp0IntTile, kFp32MantissaBits);
    // er = exp(r) * p2 = exp(ax) / 4
    pto::TEXP(tmp1Tile, tmp1Tile);
    pto::TMUL(tmp1Tile, tmp1Tile, tmp0Tile);
    // Generate the overflow mask before overwriting ax in dstTile.
    // NaN >= 90 is false, so the naturally computed NaN is preserved.
    pto::TCMPS(tmp2MaskTile, dstTile, kOvfThreshold, pto::CmpMode::GE);
    // dst = 0.125 / er
    pto::TRECIP(dstTile, tmp1Tile);
    pto::TMULS(dstTile, dstTile, kHalfInv8);
    // tmp0 = 2 * er
    pto::TMULS(tmp0Tile, tmp1Tile, kTwo);
    // dst = 2 * er + 0.125 / er
    pto::TADD(dstTile, tmp0Tile, dstTile);
    pto::TADDS(tmp0Tile, tmp0Tile, kInf);
    // Select +inf on overflow; otherwise preserve the regular result.
    // er in tmp1Tile is dead and can be used as TSEL scratch.
    pto::TSEL(dstTile, tmp2MaskTile, tmp0Tile, dstTile, tmp1Tile);
}
#endif

#define OP_TILE_OP_COSH TCosh
template <typename T0, typename T1, typename T2>
TILEOP void TCosh(T0 dst, T1 src, T2 tmp)
{
    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    constexpr auto tileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    auto tileShapeSize = dstShape0 * dstShape1 * dstShape2 * dstShape3 * tileW;

    using DataTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, tileH, tileW, pto::BLayout::RowMajor, -1,
                                     -1>;
    DataTileDefine dstTile(dstShape3, dstShape4);
    auto srcExecTile = MakeElementwiseOperandExecTile(dst, src);

#ifdef __DAV_V220
    DataTileDefine tmpTile(dstShape3, dstShape4);
#else
    using IntTileDefine = pto::Tile<pto::TileType::Vec, int32_t, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using MaskTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, tileW * sizeof(float), pto::BLayout::RowMajor,
                                     -1, -1>;

    DataTileDefine tmp0Tile(dstShape3, dstShape4);
    DataTileDefine tmp1Tile(dstShape3, dstShape4);
    DataTileDefine tmp2Tile(dstShape3, dstShape4);
    IntTileDefine tmp0IntTile(dstShape3, dstShape4);
    MaskTileDefine tmp2MaskTile(dstShape3, dstShape4);
    constexpr size_t TMP2_SLOT = 2;
#endif

    for (LoopVar n0Index = 0; n0Index < dstShape0; n0Index++) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; n1Index++) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; n2Index++) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto dstOffset = GenTileOffset(dst, tileOffsets);
                auto tmpByteOffset = TileOp::GetPackedByteOffset<typename T2::Type>(GenTileOffset(tmp, tileOffsets));
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                AssignElementwiseOperandExecTile(srcExecTile, src, tileOffsets);

#ifdef __DAV_V220
                pto::TASSIGN(tmpTile, (uint64_t)(tmp.GetAddr() + tmpByteOffset));
                TCoshCompute(dstTile, srcExecTile, tmpTile);
#else
                auto tmp0Addr = static_cast<uint64_t>(tmp.GetAddr() + tmpByteOffset);
                auto tmp1Addr = static_cast<uint64_t>(tmp.GetAddr() + tmpByteOffset + tileShapeSize * dstTypeSize);
                auto tmp2Addr = static_cast<uint64_t>(tmp.GetAddr() + tmpByteOffset +
                                                      TMP2_SLOT * tileShapeSize * dstTypeSize);

                pto::TASSIGN(tmp0Tile, tmp0Addr);
                pto::TASSIGN(tmp1Tile, tmp1Addr);
                pto::TASSIGN(tmp2Tile, tmp2Addr);
                pto::TASSIGN(tmp0IntTile, tmp0Addr);
                pto::TASSIGN(tmp2MaskTile, tmp2Addr);

                TCoshCompute(dstTile, srcExecTile, tmp0Tile, tmp1Tile, tmp2Tile, tmp0IntTile, tmp2MaskTile);
#endif
            }
        }
    }
}

template <UnaryOp op, typename T0, typename T1, typename T2, typename T3, typename T4>
TILEOP void reduceKCompute(T0 dst, T1 tmp0, T2 tmp1, T3 tmp2, T4 src0)
{
    // define the number of x div pi
    constexpr float PI_FOR_X_TODIV = 0.3183098733425140380859375;
    // define the PI for compute
    constexpr float PI_V2 = 3.140625;
    constexpr float KPI_FIRS_PI_MULS = 0.0009670257568359375;
    constexpr float KPI_TWI_PI_MULS = 6.2771141529083251953125e-7;
    constexpr float KPI_THIR_PI_MULS = 1.21644916362129151821136474609375e-10;
    constexpr float KPI_FOR_PI_MULS = -1.0290623200529979163359041220560e-13;
    constexpr float POINT_FIVE = 0.5;
    constexpr float K2_SCA = -2.0;
    constexpr float M4_SCA = 4.0;
    constexpr float TRIG_ZERO = 0.0;
    constexpr float TRIG_ONE = 1.0;
    // define the number of down of pi_div
    constexpr float PI_DOWN = 1.57079637050628662109375;
    // kpi_2
    constexpr float PI_RESDOWN_ADDS_NEG = -0.00000004371139000189375;

    pto::TMULS(tmp0, src0, TRIG_ZERO);
    SyncV();
    pto::TADD(tmp2, src0, tmp0);
    SyncV();
    //  k=round(x/π), x0=x-kπ, x0 belongs to [-π/2, π/2]
    //  cos(x) = (-1)^k * sin(x0 + π/2)
    pto::TMULS(tmp0, tmp2, PI_FOR_X_TODIV);
    SyncV();
    if constexpr (op == UnaryOp::SIN) {
        pto::TCVT(tmp1, tmp0, pto::RoundMode::CAST_ROUND);
        SyncV();
    }
    if constexpr (op == UnaryOp::COS) {
        pto::TADDS(tmp0, tmp0, POINT_FIVE);
        SyncV();
        pto::TCVT(tmp1, tmp0, pto::RoundMode::CAST_RINT);
        SyncV();
    }
    pto::TCVT(tmp0, tmp1, pto::RoundMode::CAST_NONE);
    SyncV();

    // x -= k * pi_0
    pto::TMULS(dst, tmp0, PI_V2);
    SyncV();
    pto::TSUB(tmp2, tmp2, dst);
    SyncV();
    // x -= k * pi_1
    pto::TMULS(dst, tmp0, KPI_FIRS_PI_MULS);
    SyncV();
    pto::TSUB(tmp2, tmp2, dst);
    SyncV();
    // x = x + PI_DOWN
    if constexpr (op == UnaryOp::COS) {
        pto::TADDS(tmp2, tmp2, PI_DOWN);
        SyncV();
    }
    // x -= k * pi_2
    pto::TMULS(dst, tmp0, KPI_TWI_PI_MULS);
    SyncV();
    pto::TSUB(tmp2, tmp2, dst);
    SyncV();
    // x -= k * pi_3
    pto::TMULS(dst, tmp0, KPI_THIR_PI_MULS);
    SyncV();
    pto::TSUB(tmp2, tmp2, dst);
    SyncV();
    // x -= k * pi_4
    pto::TMULS(dst, tmp0, KPI_FOR_PI_MULS);
    SyncV();
    pto::TSUB(tmp2, tmp2, dst);
    SyncV();

    if constexpr (op == UnaryOp::COS) {
        // x = x + PI_RESDOWN_ADDS_NEG
        pto::TADDS(tmp2, tmp2, PI_RESDOWN_ADDS_NEG);
        SyncV();
    }
    // kover2
    pto::TMULS(dst, tmp0, POINT_FIVE);
    SyncV();
    pto::TCVT(tmp1, dst, pto::RoundMode::CAST_FLOOR);
    SyncV();
    pto::TCVT(dst, tmp1, pto::RoundMode::CAST_NONE);
    SyncV();
    // kover2floorm4
    pto::TMULS(dst, dst, M4_SCA);
    SyncV();
    // k2
    pto::TMULS(tmp0, tmp0, K2_SCA);
    SyncV();
    // sign
    pto::TADD(dst, dst, tmp0);
    SyncV();
    pto::TADDS(dst, dst, TRIG_ONE);
    SyncV();
}

template <UnaryOp op, typename T0, typename T1, typename T2, typename T3, typename T4>
TILEOP void SinCosCompute(T0 dst, T1 tmp0, T2 tmp1, T3 tmp2, T4 src0)
{
    constexpr float RES_MULTI_SCA = 2.604926501e-6;
    constexpr float RES_ADDICT_UP = -0.0001980894471;
    constexpr float ADD2S = 0.008333049340;
    constexpr float ADD3S = -0.1666665792;
    constexpr float TRIG_ONE = 1.0;

    // x^2
    pto::TMUL(tmp0, tmp2, tmp2);
    SyncV();
    // sin(x) = x * P(x)
    // P(x) = (((x^2 * R0 + R1) * x^2 + R2) * x^2 + R3) * x^2 + 1.0
    // roundTensor = mul(x^2, 2.604926501e-6)
    pto::TMULS(tmp1, tmp0, RES_MULTI_SCA);
    SyncV();
    pto::TADDS(tmp1, tmp1, RES_ADDICT_UP);
    SyncV();
    // roundTensor = mul(roundTensor, x^2)
    pto::TMUL(tmp1, tmp0, tmp1);
    SyncV();
    pto::TADDS(tmp1, tmp1, ADD2S);
    SyncV();
    // roundTensor = mul(roundTensor, x^2)
    pto::TMUL(tmp1, tmp0, tmp1);
    SyncV();
    pto::TADDS(tmp1, tmp1, ADD3S);
    SyncV();
    // roundTensor = mul(roundTensor, x^2)
    pto::TMUL(tmp1, tmp0, tmp1);
    SyncV();
    pto::TADDS(tmp1, tmp1, TRIG_ONE);
    SyncV();
    // sin(x) = x * P(x)
    pto::TMUL(tmp1, tmp2, tmp1);
    SyncV();
    pto::TMUL(dst, dst, tmp1);
    SyncV();
    return;
}
// P(x) = (((((0.053443748819x^2+0.75517016694e1)x^2+0.10162808918e3)x^2
//          +0.13938061484e4)x^2+0.50637915060e4)x^2+0.29639384698e5)x
template <UnaryOp op, typename T0, typename T1, typename T2>
TILEOP void TrigCompute(T0 dst, T1 tmp, T2 src)
{
    static_assert(op == UnaryOp::SIN || op == UnaryOp::COS, "TrigCompute only supports SIN and COS");

    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto shape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto shape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    constexpr auto tileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);

    using TmpFP32Tile = pto::Tile<pto::TileType::Vec, typename T0::Type, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using TmpINT32Tile = pto::Tile<pto::TileType::Vec, int32_t, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;

    TmpFP32Tile dstTile(shape3, shape4);
    TmpFP32Tile tmp0Tile(shape3, shape4);
    TmpINT32Tile tmp1Tile(shape3, shape4);
    TmpFP32Tile tmp2Tile(shape3, shape4);
    TmpFP32Tile tmp3Tile(shape3, shape4);
    constexpr size_t TMP2_SLOT = 2;
    auto srcExecTile = MakeElementwiseOperandExecTile(dst, src);
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + GenTileOffset(dst, tileOffsets) * dstTypeSize));
                AssignElementwiseOperandExecTile(srcExecTile, src, tileOffsets);
                pto::TASSIGN(tmp0Tile, (uint64_t)(tmp.GetAddr()));
                pto::TASSIGN(tmp1Tile, (uint64_t)(tmp.GetAddr() + tileW * shape3 * dstTypeSize));
                pto::TASSIGN(tmp2Tile, (uint64_t)(tmp.GetAddr() + TMP2_SLOT * tileW * shape3 * dstTypeSize));
                pto::TASSIGN(tmp3Tile, (uint64_t)(tmp.GetAddr() + tileW * shape3 * dstTypeSize));

                reduceKCompute<op>(dstTile, tmp0Tile, tmp1Tile, tmp2Tile, srcExecTile);
                SyncV();
                SinCosCompute<op>(dstTile, tmp0Tile, tmp3Tile, tmp2Tile, srcExecTile);
            }
        }
    }
}

#define OP_TILE_OP_SIN TSin
template <typename T0, typename T1, typename T2>
TILEOP void TSin(T0 dst, T1 tmp, T2 src)
{
    TrigCompute<UnaryOp::SIN>(dst, tmp, src);
}

#define OP_TILE_OP_COS TCos
template <typename T0, typename T1, typename T2>
TILEOP void TCos(T0 dst, T1 tmp, T2 src)
{
    TrigCompute<UnaryOp::COS>(dst, tmp, src);
}

#endif // TILEOP_TILE_OPERATOR_VEC_UNARY_TRANSCENDENTAL_H
