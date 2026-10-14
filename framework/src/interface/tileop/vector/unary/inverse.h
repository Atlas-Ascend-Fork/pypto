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
 * \file inverse.h
 * \brief Unary tile operation implementations.
 */

#ifndef TILEOP_TILE_OPERATOR_VEC_UNARY_INVERSE_H
#define TILEOP_TILE_OPERATOR_VEC_UNARY_INVERSE_H

#include "utils/sync.h"
#include "basic.h"

// Horner evaluation of arcsin Taylor on t in [0, 1/sqrt(2)]:
//   arcsin(t) = t * (c0 + c1*s + c2*s^2 + ... + c7*s^7),  s = t^2
template <typename TOut, typename TIn, typename TScratch>
TILEOP void ArcsinPolyHorner(TOut outTile, TIn tTile, TScratch sScratch)
{
    constexpr float ASIN_C0 = 1.0f;        // 1
    constexpr float ASIN_C1 = 0.16666667f; // 1/6
    constexpr float ASIN_C2 = 0.075f;      // 3/40
    constexpr float ASIN_C3 = 0.04464286f; // 5/112
    constexpr float ASIN_C4 = 0.03038194f; // 35/1152
    constexpr float ASIN_C5 = 0.02237216f; // 63/2816
    constexpr float ASIN_C6 = 0.01735276f; // 231/13312
    constexpr float ASIN_C7 = 0.01396484f; // 143/10240

    // s = t^2
    pto::TMUL(sScratch, tTile, tTile);
    SyncV();
    // acc = c7*s + c6
    pto::TMULS(outTile, sScratch, ASIN_C7);
    SyncV();
    pto::TADDS(outTile, outTile, ASIN_C6);
    SyncV();
    // acc = acc*s + c5
    pto::TMUL(outTile, outTile, sScratch);
    SyncV();
    pto::TADDS(outTile, outTile, ASIN_C5);
    SyncV();
    // acc = acc*s + c4
    pto::TMUL(outTile, outTile, sScratch);
    SyncV();
    pto::TADDS(outTile, outTile, ASIN_C4);
    SyncV();
    // acc = acc*s + c3
    pto::TMUL(outTile, outTile, sScratch);
    SyncV();
    pto::TADDS(outTile, outTile, ASIN_C3);
    SyncV();
    // acc = acc*s + c2
    pto::TMUL(outTile, outTile, sScratch);
    SyncV();
    pto::TADDS(outTile, outTile, ASIN_C2);
    SyncV();
    // acc = acc*s + c1
    pto::TMUL(outTile, outTile, sScratch);
    SyncV();
    pto::TADDS(outTile, outTile, ASIN_C1);
    SyncV();
    // acc = acc*s + c0
    pto::TMUL(outTile, outTile, sScratch);
    SyncV();
    pto::TADDS(outTile, outTile, ASIN_C0);
    SyncV();
    // result = acc * t
    pto::TMUL(outTile, outTile, tTile);
    SyncV();
}

template <bool IsAsin, typename TDst, typename TSrc, typename TTmp0, typename TTmp1, typename TTmp2, typename TMask>
TILEOP void TAsinAcosTileImpl(TDst dstTile, TSrc srcTile, TTmp0 tmp0Tile, TTmp1 tmp1Tile, TTmp2 tmp2Tile,
                              TMask maskTile)
{
    constexpr float ASIN_THRESHOLD = 0.70710678f; // 1/sqrt(2)
    constexpr float PI_HALF = 1.57079633f;
    constexpr float SCALAR_ONE = 1.0f;
    constexpr float SCALAR_NEGATIVE_ONE = -1.0f;
    constexpr float SCALAR_ZERO = 0.0f;

    // ---- 1) tmp0 = |x| ----
    pto::TABS(tmp0Tile, srcTile);
    SyncV();

    // ---- 2) Reduce both branches to t in [0, 1/sqrt(2)] ----
    // tmp1 = sqrt(1 - x^2), the reduced argument for the large branch
    pto::TMUL(tmp1Tile, tmp0Tile, tmp0Tile);
    SyncV();
    pto::TMULS(tmp1Tile, tmp1Tile, SCALAR_NEGATIVE_ONE);
    SyncV();
    pto::TADDS(tmp1Tile, tmp1Tile, SCALAR_ONE);
    SyncV();
    pto::TSQRT(tmp1Tile, tmp1Tile);
    SyncV();

    // tmp2 = |x| for the small branch, sqrt(1 - x^2) for the large branch
    pto::TCMPS(maskTile, tmp0Tile, ASIN_THRESHOLD, pto::CmpMode::LE);
    SyncV();
    // dstTile is still unused and serves as TSEL scratch on A2/A3.
    pto::TSEL(tmp2Tile, maskTile, tmp0Tile, tmp1Tile, dstTile);
    SyncV();

    // ---- 3) Evaluate the shared polynomial once ----
    // tmp1 is no longer needed and is reused as Horner scratch.
    ArcsinPolyHorner(dstTile, tmp2Tile, tmp1Tile);

    // tmp0 = pi/2 - poly(t), used only by the large branch
    pto::TMULS(tmp0Tile, dstTile, SCALAR_NEGATIVE_ONE);
    SyncV();
    pto::TADDS(tmp0Tile, tmp0Tile, PI_HALF);
    SyncV();
    pto::TSEL(dstTile, maskTile, dstTile, tmp0Tile, tmp1Tile);
    SyncV();
    // dst now == arcsin(|x|), >= 0

    // ---- 4) Sign restore ----
    if constexpr (IsAsin) {
        // arcsin is odd: dst = src >= 0 ? dst : -dst
        pto::TMULS(tmp0Tile, dstTile, SCALAR_NEGATIVE_ONE);
        SyncV();
        pto::TCMPS(maskTile, srcTile, SCALAR_ZERO, pto::CmpMode::GE);
        SyncV();
        pto::TSEL(dstTile, maskTile, dstTile, tmp0Tile, tmp1Tile);
        SyncV();
    } else {
        // arccos(x) = pi/2 - sign(src)*arcsin(|x|)
        //   src >= 0: pi/2 - dst
        //   src <  0: pi/2 + dst
        pto::TMULS(tmp0Tile, dstTile, SCALAR_NEGATIVE_ONE);
        SyncV();
        pto::TCMPS(maskTile, srcTile, SCALAR_ZERO, pto::CmpMode::GE);
        SyncV();
        pto::TSEL(dstTile, maskTile, tmp0Tile, dstTile, tmp1Tile);
        SyncV();
        pto::TADDS(dstTile, dstTile, PI_HALF);
        SyncV();
    }
}

// Unified body for TAsin / TAcos.
//   |x| <= 1/sqrt(2):  arcsin(|x|) via 8-term Taylor on |x|
//   |x| >  1/sqrt(2):  arcsin(|x|) = pi/2 - arcsin(sqrt(1 - x^2))
template <bool IsAsin, typename T0, typename T1, typename T2>
TILEOP void TAsinAcosImpl(T0 dst, T1 src, T2 tmp)
{
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto shape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto shape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    constexpr auto tileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);

    using DataTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, tileH, tileW, pto::BLayout::RowMajor, -1,
                                     -1>;
    using MaskTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, tileW * 4, pto::BLayout::RowMajor, -1, -1>;

    DataTileDefine dstTile(shape3, shape4);
    auto srcExecTile = MakeElementwiseOperandExecTile(dst, src);
    DataTileDefine tmp0Tile(shape3, shape4); // |x|, then large-branch result / negated result
    DataTileDefine tmp1Tile(shape3, shape4); // large-branch argument, then Horner / TSEL scratch
    DataTileDefine tmp2Tile(shape3, shape4); // shared reduced argument
    MaskTileDefine maskTile(shape3, shape4);

    size_t tmpStride = shape3 * tileW * dstTypeSize;
    constexpr size_t TMP2_SLOT = 2;
    constexpr size_t MASK_SLOT = 3;
    pto::TASSIGN(tmp0Tile, (uint64_t)(tmp.GetAddr() + 0 * tmpStride));
    pto::TASSIGN(tmp1Tile, (uint64_t)(tmp.GetAddr() + 1 * tmpStride));
    pto::TASSIGN(tmp2Tile, (uint64_t)(tmp.GetAddr() + TMP2_SLOT * tmpStride));
    pto::TASSIGN(maskTile, (uint64_t)(tmp.GetAddr() + MASK_SLOT * tmpStride));

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + GenTileOffset(dst, tileOffsets) * dstTypeSize));
                AssignElementwiseOperandExecTile(srcExecTile, src, tileOffsets);
                TAsinAcosTileImpl<IsAsin>(dstTile, srcExecTile, tmp0Tile, tmp1Tile, tmp2Tile, maskTile);
            }
        }
    }
}

#define OP_TILE_OP_ASIN TAsin
template <typename T0, typename T1, typename T2>
TILEOP void TAsin(T0 dst, T1 src, T2 tmp)
{
    TAsinAcosImpl<true>(dst, src, tmp);
}

#define OP_TILE_OP_ACOS TAcos
template <typename T0, typename T1, typename T2>
TILEOP void TAcos(T0 dst, T1 src, T2 tmp)
{
    TAsinAcosImpl<false>(dst, src, tmp);
}

#define OP_TILE_OP_ASINH TASinh
template <typename T0, typename T1, typename T2>
TILEOP void TASinh(T0 dst, T1 src, T2 tmp)
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

    constexpr float CONST_BRANCH_CONDITION = 0.00024414063;
    constexpr float CONST_ZERO = 0.0f;
    constexpr float CONST_ONE = 1.0f;
    constexpr float CONST_NEG_ONE = -1.0f;
    constexpr float CONST_COMPARE_VALUE_MIN = 1e-45f;
    constexpr float CONST_COMPARE_VALUE_MAX = 3.4028235e34f;
    constexpr float CONST_LOG_TWO_VALUE = 6.93147180559945286227e-01f;

    auto tileShapeSize = dstShape0 * dstShape1 * dstShape2 * dstShape3 * tileW;

    using DataTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, tileH, tileW, pto::BLayout::RowMajor, -1,
                                     -1>;
    using MaskTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, 4 * tileW, pto::BLayout::RowMajor, -1, -1>;
    auto srcExecTile = MakeElementwiseOperandExecTile(dst, src);
    DataTileDefine dstTile(dstShape3, dstShape4);
    DataTileDefine tmp0Tile(dstShape3, dstShape4);
    DataTileDefine tmp1Tile(dstShape3, dstShape4);
    DataTileDefine tmp2Tile(dstShape3, dstShape4);
    DataTileDefine tmp3Tile(dstShape3, dstShape4);
    MaskTileDefine tmp2MaskTile(dstShape3, dstShape4);
    constexpr size_t TMP2_SLOT = 2;
    constexpr size_t TMP3_SLOT = 3;

    for (LoopVar n0Index = 0; n0Index < dstShape0; n0Index++) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; n1Index++) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; n2Index++) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto dstOffset = GenTileOffset(dst, tileOffsets);
                auto tmpByteOffset = TileOp::GetPackedByteOffset<typename T2::Type>(GenTileOffset(tmp, tileOffsets));
                AssignElementwiseOperandExecTile(srcExecTile, src, tileOffsets);
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));

                pto::TASSIGN(tmp0Tile, (uint64_t)(tmp.GetAddr() + tmpByteOffset));
                pto::TASSIGN(tmp1Tile, (uint64_t)(tmp.GetAddr() + tmpByteOffset + tileShapeSize * dstTypeSize));
                pto::TASSIGN(tmp2Tile,
                             (uint64_t)(tmp.GetAddr() + tmpByteOffset + TMP2_SLOT * tileShapeSize * dstTypeSize));
                pto::TASSIGN(tmp3Tile,
                             (uint64_t)(tmp.GetAddr() + tmpByteOffset + TMP3_SLOT * tileShapeSize * dstTypeSize));
                pto::TASSIGN(tmp2MaskTile,
                             (uint64_t)(tmp.GetAddr() + tmpByteOffset + TMP2_SLOT * tileShapeSize * dstTypeSize));

                pto::TABS(tmp0Tile, srcExecTile); // |x|
                SyncV();
                pto::TDIVS(tmp1Tile, CONST_ONE, tmp0Tile); // 1/|x|
                SyncV();
                pto::TMUL(tmp2Tile, tmp1Tile, tmp1Tile); // 1/(|x|)^2
                SyncV();

                pto::TADDS(tmp3Tile, tmp2Tile, CONST_ONE); // 1 + 1/(|x|)^2
                SyncV();
                pto::TSQRT(tmp3Tile, tmp3Tile); // sqrt(1 + 1/(|x|)^2)
                SyncV();
                pto::TADD(tmp1Tile, tmp3Tile, tmp1Tile); // sqrt(1 + 1/(|x|)^2) + 1/|x|
                SyncV();
                pto::TDIV(tmp1Tile, tmp0Tile, tmp1Tile); // |x| / (sqrt(1 + 1/(|x|)^2) + 1/|x|)
                SyncV();
                pto::TADD(tmp1Tile, tmp0Tile, tmp1Tile); // r = |x| + |x| / (sqrt(1 + 1/(|x|)^2) + 1/|x|)
                SyncV();
                pto::TADDS(tmp3Tile, tmp1Tile, CONST_ONE); // r + 1
                SyncV();

                pto::TADDS(dstTile, tmp3Tile, CONST_NEG_ONE); // clamp(r, s_min, s_max)
                SyncV();
                pto::TMAXS(dstTile, dstTile, CONST_COMPARE_VALUE_MIN);
                SyncV();
                pto::TMINS(dstTile, dstTile, CONST_COMPARE_VALUE_MAX);
                SyncV();

                pto::TLOG(tmp3Tile, tmp3Tile); // log(r + 1)
                SyncV();
                pto::TMUL(tmp1Tile, tmp1Tile, tmp3Tile); // r * log(r + 1)
                SyncV();
                pto::TDIV(tmp1Tile, tmp1Tile, dstTile); // r * log(r + 1) / clamp(r, s_min, s_max)
                SyncV();

                pto::TLOG(tmp3Tile, tmp0Tile); // log(|x|)
                SyncV();
                pto::TADDS(tmp3Tile, tmp3Tile, CONST_LOG_TWO_VALUE); // log(|x|) + log2
                SyncV();
                pto::TADD(tmp2Tile, tmp3Tile, tmp2Tile); // log(|x|) + log2 + 1/(|x|)^2
                SyncV();
                pto::TMIN(tmp1Tile, tmp1Tile, tmp2Tile); // min
                SyncV();

                pto::TCMPS(tmp2MaskTile, tmp0Tile, CONST_BRANCH_CONDITION, pto::CmpMode::LT);
                SyncV();
                pto::TSEL(tmp0Tile, tmp2MaskTile, tmp0Tile, tmp1Tile, tmp3Tile);
                SyncV();
                pto::TMULS(tmp1Tile, tmp0Tile, CONST_NEG_ONE);
                SyncV();

                pto::TCMPS(tmp2MaskTile, srcExecTile, CONST_ZERO, pto::CmpMode::GE);
                SyncV();
                pto::TSEL(dstTile, tmp2MaskTile, tmp0Tile, tmp1Tile, tmp3Tile);
                SyncV();
            }
        }
    }
}

#define OP_TILE_OP_ACOSH TACosh
template <typename T0, typename T1, typename T2>
TILEOP void TACosh(T0 dst, T1 src, T2 tmp)
{
    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    constexpr float CONST_ONE = 1.0f;
    constexpr float CONST_NEG_ONE = -1.0f;
    constexpr float CONST_COMPARE_VALUE_MIN = 1e-45f;
    constexpr float CONST_COMPARE_VALUE_MAX = 3.4028235e34f;
    constexpr float CONST_LOG_TWO_VALUE = 6.93147180559945286227e-01f;

    constexpr auto tileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);

    auto tileShapeSize = dstShape0 * dstShape1 * dstShape2 * dstShape3 * tileW;

    using DataTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, tileH, tileW, pto::BLayout::RowMajor, -1,
                                     -1>;
    auto srcExecTile = MakeElementwiseOperandExecTile(dst, src);
    DataTileDefine dstTile(dstShape3, dstShape4);
    DataTileDefine tmp0Tile(dstShape3, dstShape4);
    DataTileDefine tmp1Tile(dstShape3, dstShape4);
    DataTileDefine tmp2Tile(dstShape3, dstShape4);
    constexpr size_t TMP2_SLOT = 2;
    for (LoopVar n0Index = 0; n0Index < dstShape0; n0Index++) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; n1Index++) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; n2Index++) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto dstOffset = GenTileOffset(dst, tileOffsets);
                auto tmpByteOffset = TileOp::GetPackedByteOffset<typename T2::Type>(GenTileOffset(tmp, tileOffsets));
                AssignElementwiseOperandExecTile(srcExecTile, src, tileOffsets);
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));

                pto::TASSIGN(tmp0Tile, (uint64_t)(tmp.GetAddr() + tmpByteOffset));
                pto::TASSIGN(tmp1Tile, (uint64_t)(tmp.GetAddr() + tmpByteOffset + tileShapeSize * dstTypeSize));
                pto::TASSIGN(tmp2Tile,
                             (uint64_t)(tmp.GetAddr() + tmpByteOffset + TMP2_SLOT * tileShapeSize * dstTypeSize));

                pto::TADDS(tmp0Tile, srcExecTile, CONST_NEG_ONE); // t
                SyncV();
                pto::TADD(tmp1Tile, tmp0Tile, tmp0Tile); // 2t
                pto::TMUL(tmp2Tile, tmp0Tile, tmp0Tile); // t^2
                SyncV();
                pto::TADD(tmp1Tile, tmp1Tile, tmp2Tile); // t^2 + 2t
                SyncV();
                pto::TSQRT(tmp1Tile, tmp1Tile); // sqrt(t^2 + 2t)
                SyncV();
                pto::TADD(tmp1Tile, tmp1Tile, tmp0Tile); // t + sqrt(t^2 + 2t) = r
                SyncV();
                pto::TADDS(tmp2Tile, tmp1Tile, CONST_ONE); // r + 1
                SyncV();
                pto::TADDS(tmp0Tile, tmp2Tile, CONST_NEG_ONE); // clamp(r, s_min, s_max)
                pto::TLOG(dstTile, tmp2Tile);                  // log(r + 1)
                SyncV();
                pto::TMAXS(tmp0Tile, tmp0Tile, CONST_COMPARE_VALUE_MIN);
                pto::TMUL(dstTile, dstTile, tmp1Tile); // r * log(r + 1)
                SyncV();
                pto::TMINS(tmp0Tile, tmp0Tile, CONST_COMPARE_VALUE_MAX);
                pto::TLOG<pto::LogAlgorithm::HIGH_PRECISION>(tmp1Tile, srcExecTile); // log(x)
                SyncV();
                pto::TDIV(dstTile, dstTile, tmp0Tile);               // r * log(r + 1) / clamp(r, s_min, s_max)
                pto::TADDS(tmp1Tile, tmp1Tile, CONST_LOG_TWO_VALUE); // log(x) + log(2)
                SyncV();
                pto::TMIN(dstTile, dstTile, tmp1Tile);
                SyncV();
            }
        }
    }
}
#endif // TILEOP_TILE_OPERATOR_VEC_UNARY_INVERSE_H
