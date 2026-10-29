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
 * \file erf.h
 * \brief Unary tile operation implementations.
 */

#ifndef TILEOP_TILE_OPERATOR_VEC_UNARY_ERF_H
#define TILEOP_TILE_OPERATOR_VEC_UNARY_ERF_H

#include "utils/sync.h"
#include "basic.h"

template <typename T0, typename T1, typename T2>
TILEOP void ErfComputeP(T0 dst, T1 tmp0, T2 tmp1)
{
    constexpr float SCALAR_P0 = 0.29639384698e5;
    constexpr float SCALAR_P1 = 0.50637915060e4;
    constexpr float SCALAR_P2 = 0.13938061484e4;
    constexpr float SCALAR_P3 = 0.10162808918e3;
    constexpr float SCALAR_P4 = 0.75517016694e1;
    constexpr float SCALAR_P5 = 0.053443748819;
    // x^2
    pto::TMULS(tmp1, tmp0, SCALAR_P5);
    SyncV();
    pto::TADDS(tmp1, tmp1, SCALAR_P4);
    SyncV();
    pto::TMUL(tmp1, tmp0, tmp1);
    SyncV();
    pto::TADDS(tmp1, tmp1, SCALAR_P3);
    SyncV();
    pto::TMUL(tmp1, tmp0, tmp1);
    SyncV();
    pto::TADDS(tmp1, tmp1, SCALAR_P2);
    SyncV();
    pto::TMUL(tmp1, tmp0, tmp1);
    SyncV();
    pto::TADDS(tmp1, tmp1, SCALAR_P1);
    SyncV();
    pto::TMUL(tmp1, tmp0, tmp1);
    SyncV();
    pto::TADDS(tmp1, tmp1, SCALAR_P0);
    SyncV();
    pto::TMUL(tmp1, dst, tmp1);
    return;
}
// Q(x) = ((((x^2+0.31212858877e2)x^2+0.39856963806e3)x^2+0.30231248150e4)x^2+0.13243365831e5)x^2+0.26267224157e5
template <typename T0, typename T1>
TILEOP void ErfComputeQ(T0 tmp0, T1 tmp2)
{
    constexpr float SCALAR_Q0 = 0.26267224157e5;
    constexpr float SCALAR_Q1 = 0.13243365831e5;
    constexpr float SCALAR_Q2 = 0.30231248150e4;
    constexpr float SCALAR_Q3 = 0.39856963806e3;
    constexpr float SCALAR_Q4 = 0.31212858877e2;

    pto::TADDS(tmp2, tmp0, SCALAR_Q4);
    SyncV();
    pto::TMUL(tmp2, tmp0, tmp2);
    SyncV();
    pto::TADDS(tmp2, tmp2, SCALAR_Q3);
    SyncV();
    pto::TMUL(tmp2, tmp0, tmp2);
    SyncV();
    pto::TADDS(tmp2, tmp2, SCALAR_Q2);
    SyncV();
    pto::TMUL(tmp2, tmp0, tmp2);
    SyncV();
    pto::TADDS(tmp2, tmp2, SCALAR_Q1);
    SyncV();
    pto::TMUL(tmp2, tmp0, tmp2);
    SyncV();
    pto::TADDS(tmp2, tmp2, SCALAR_Q0);
    return;
}
// Erf(x) = P(x) / Q(x)
template <typename T0, typename T1, typename T2, typename T3, typename T4>
TILEOP void ErfPadeCompute(T0 dst, T1 tmp0, T2 tmp1, T3 tmp2, T4 src)
{
    constexpr float ERF_BOUNDARY_MAX = 3.92;

    pto::TMINS(dst, src, ERF_BOUNDARY_MAX);
    SyncV();
    pto::TMAXS(dst, dst, -ERF_BOUNDARY_MAX);
    SyncV();
    // x^2
    pto::TMUL(tmp0, dst, dst);
    SyncV();
    ErfComputeP(dst, tmp0, tmp1);
    SyncV();
    ErfComputeQ(tmp0, tmp2);
    SyncV();
    pto::TDIV(dst, tmp1, tmp2);
    SyncV();
    return;
}

template <typename T0, typename T1, typename T2>
TILEOP void ErfSubsectionSmallCompute(T0 dst, T1 tmp2, T2 src)
{
    using FloatIntUnion = union {
        uint32_t i;
        float f;
    };
    pto::TMUL(dst, src, src);
    pto::TMULS(tmp2, dst, FloatIntUnion{.i = 0x38B1E96A}.f);
    pto::TADDS(tmp2, tmp2, FloatIntUnion{.i = 0xBA574D20}.f);
    pto::TMUL(tmp2, dst, tmp2);
    pto::TADDS(tmp2, tmp2, FloatIntUnion{.i = 0x3BAAD5EA}.f);
    pto::TMUL(tmp2, dst, tmp2);
    pto::TADDS(tmp2, tmp2, FloatIntUnion{.i = 0xBCDC1BE7}.f);
    pto::TMUL(tmp2, dst, tmp2);
    pto::TADDS(tmp2, tmp2, FloatIntUnion{.i = 0x3DE718AF}.f);
    pto::TMUL(tmp2, dst, tmp2);
    pto::TADDS(tmp2, tmp2, FloatIntUnion{.i = 0xBEC093AC}.f);
    pto::TMUL(tmp2, dst, tmp2);
    pto::TADDS(tmp2, tmp2, FloatIntUnion{.i = 0x3E0375D3}.f);
    pto::TMUL(tmp2, src, tmp2);
    pto::TADD(tmp2, tmp2, src);
    return;
}

template <typename T0, typename T1, typename T2, typename T3>
TILEOP void ErfSubsectionLargeCompute(T0 dst, T1 tmp0, T2 tmp1, T3 src)
{
    using FloatIntUnion = union {
        uint32_t i;
        float f;
    };
    constexpr float LOG2_VALUE = 2.0f;
    constexpr float ZERO_VALUE = 0.0f;

    pto::TABS(tmp1, src);
    pto::TMULS(dst, tmp1, FloatIntUnion{0x38EB4C3A}.f);
    pto::TADDS(dst, dst, FloatIntUnion{0xBAAE005B}.f);
    pto::TMUL(dst, tmp1, dst);
    pto::TADDS(dst, dst, FloatIntUnion{0x3C09919F}.f);
    pto::TMUL(dst, tmp1, dst);
    pto::TADDS(dst, dst, FloatIntUnion{0xBD24D99A}.f);
    pto::TMUL(dst, tmp1, dst);
    pto::TADDS(dst, dst, FloatIntUnion{0x3E235519}.f);
    pto::TMUL(dst, tmp1, dst);
    pto::TADDS(dst, dst, FloatIntUnion{0x3F69B4F9}.f);
    pto::TMUL(dst, tmp1, dst);
    pto::TADDS(dst, dst, FloatIntUnion{0x3F210A14}.f);
    pto::TNEG(tmp1, tmp1);
    pto::TMUL(dst, tmp1, dst);
    pto::TADD(dst, dst, tmp1);

    pto::TEXPANDS(tmp1, LOG2_VALUE);
    pto::TLOG<pto::LogAlgorithm::DEFAULT>(tmp1, tmp1);
    pto::TMUL(dst, tmp1, dst);
    pto::TEXP<pto::ExpAlgorithm::DEFAULT>(dst, dst);
    pto::TEXPANDS(tmp1, FloatIntUnion{0x3F800000}.f);
    pto::TSUB(dst, tmp1, dst);
    pto::TCMPS(tmp0, src, ZERO_VALUE, pto::CmpMode::GE);
    pto::TNEG(tmp1, dst);
    // tmp0=1取正值，tmp0=0取负值
    pto::TSEL(dst, tmp0, dst, tmp1, tmp1);
    return;
}

template <typename T0, typename T1, typename T2, typename T3, typename T4>
TILEOP void ErfSubsectionCompute(T0 dst, T1 tmp0, T2 tmp1, T3 tmp2, T4 src)
{
    using FloatIntUnion = union {
        uint32_t i;
        float f;
    };
    // tmp2
    ErfSubsectionSmallCompute(dst, tmp2, src);
    // dst
    ErfSubsectionLargeCompute(dst, tmp0, tmp1, src);

    pto::TABS(tmp1, src);
    pto::TCMPS(tmp0, tmp1, FloatIntUnion{0x3F8060FE}.f, pto::CmpMode::GE);
    // A5 TSEL的tmp未使用
    pto::TSEL(dst, tmp0, dst, tmp2, tmp2);
    return;
}

template <typename T0, typename T1, typename T2>
TILEOP void ErfCompute(T0 dst, T1 tmp, T2 src)
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

    using TmpFP32Tile = pto::Tile<pto::TileType::Vec, typename T2::Type, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using TmpMaskTile = pto::Tile<pto::TileType::Vec, uint8_t, tileH, tileW * 4, pto::BLayout::RowMajor, -1, -1>;

    TmpFP32Tile dstTile(shape3, shape4);
    TmpFP32Tile tmp0Tile(shape3, shape4);
    TmpFP32Tile tmp2Tile(shape3, shape4);
    TmpFP32Tile tmp3Tile(shape3, shape4);
    auto src0Tile = MakeElementwiseOperandExecTile(dst, src);
    constexpr size_t TMP2_SLOT = 2;
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                pto::TASSIGN(dstTile,
                             (uint64_t)(dst.GetAddr() + GenTileOffset(dst, tileOffsets) * sizeof(typename T2::Type)));
                pto::TASSIGN(src0Tile,
                             (uint64_t)(src.GetAddr() + GenTileOffset(src, tileOffsets) * sizeof(typename T2::Type)));
                pto::TASSIGN(tmp0Tile, (uint64_t)(tmp.GetAddr()));
                pto::TASSIGN(tmp2Tile, (uint64_t)(tmp.GetAddr() + TMP2_SLOT * tileW * shape3 * sizeof(float)));
                pto::TASSIGN(tmp3Tile, (uint64_t)(tmp.GetAddr() + tileW * shape3 * sizeof(float)));
#ifdef __DAV_V220
                ErfPadeCompute(dstTile, tmp0Tile, tmp3Tile, tmp2Tile, src0Tile);
#else
                TmpMaskTile tmpmaskTile(shape3, shape4);
                pto::TASSIGN(tmpmaskTile, (uint64_t)(tmp.GetAddr()));
                ErfSubsectionCompute(dstTile, tmpmaskTile, tmp3Tile, tmp2Tile, src0Tile);
#endif
            }
        }
    }
}

#define OP_TILE_OP_ERF TErf
template <typename T0, typename T1, typename T2>
TILEOP void TErf(T0 dst, T1 tmp, T2 src)
{
    ErfCompute(dst, tmp, src);
}

constexpr float ERFC_FP32_MIN = 2.168404344971009e-19f;
constexpr float ERFC_BOUNDARY_MAX = 10.0f;
constexpr float ERFC_NEG_BOUNDARY_MAX = -10.0f;
constexpr float ERFC_NEG_ONE = -1.0f;
constexpr float ERFC_ONE = 1.0f;

constexpr float ERFC_R0 = 0.1735313680e-7f;
constexpr float ERFC_R1 = -0.9856738394e-6f;
constexpr float ERFC_R2 = 0.2517003236e-4f;
constexpr float ERFC_R3 = -0.3848015171e-3f;
constexpr float ERFC_R4 = 0.5681528564e0f;
constexpr float ERFC_R5 = 0.5245623129e1f;
constexpr float ERFC_R6 = 0.2107740710e2f;
constexpr float ERFC_R7 = 0.4212761755e2f;
constexpr float ERFC_R8 = 0.4380524149e2f;

constexpr float ERFC_S1 = 0.9349684299e1f;
constexpr float ERFC_S2 = 0.3756930664e2f;
constexpr float ERFC_S3 = 0.8058268949e2f;
constexpr float ERFC_S4 = 0.9155653738e2f;
constexpr float ERFC_S5 = 0.4380524152e2f;

template <typename DstTileType, typename SrcTileType>
TILEOP inline void ErfcClip(DstTileType& dst, const SrcTileType& src)
{
    pto::TMINS(dst, src, ERFC_BOUNDARY_MAX);
    SyncV();
    pto::TMAXS(dst, dst, ERFC_NEG_BOUNDARY_MAX);
    SyncV();
}

template <typename TileType>
TILEOP inline void ErfcPreCompute(TileType& xb, const TileType& clipped_x, TileType& xa)
{
    pto::TABS(xa, clipped_x);
    SyncV();
    pto::TADDS(xa, xa, ERFC_FP32_MIN);
    SyncV();
    pto::TDIV(xb, clipped_x, xa);
    SyncV();
}

template <typename TileType>
TILEOP inline void ErfcComputeR(TileType& tmpCompBuf2, TileType& tmpCompBuf3, const TileType& z)
{
    pto::TMULS(tmpCompBuf2, z, ERFC_R0);
    SyncV();
    pto::TADDS(tmpCompBuf3, tmpCompBuf2, ERFC_R1);
    SyncV();
    pto::TMUL(tmpCompBuf2, z, tmpCompBuf3);
    SyncV();
    pto::TADDS(tmpCompBuf3, tmpCompBuf2, ERFC_R2);
    SyncV();
    pto::TMUL(tmpCompBuf2, z, tmpCompBuf3);
    SyncV();
    pto::TADDS(tmpCompBuf3, tmpCompBuf2, ERFC_R3);
    SyncV();
    pto::TMUL(tmpCompBuf2, z, tmpCompBuf3);
    SyncV();
    pto::TADDS(tmpCompBuf3, tmpCompBuf2, ERFC_R4);
    SyncV();
    pto::TMUL(tmpCompBuf2, z, tmpCompBuf3);
    SyncV();
    pto::TADDS(tmpCompBuf3, tmpCompBuf2, ERFC_R5);
    SyncV();
    pto::TMUL(tmpCompBuf2, z, tmpCompBuf3);
    SyncV();
    pto::TADDS(tmpCompBuf3, tmpCompBuf2, ERFC_R6);
    SyncV();
    pto::TMUL(tmpCompBuf2, z, tmpCompBuf3);
    SyncV();
    pto::TADDS(tmpCompBuf3, tmpCompBuf2, ERFC_R7);
    SyncV();
    pto::TMUL(tmpCompBuf2, z, tmpCompBuf3);
    SyncV();
    pto::TADDS(tmpCompBuf3, tmpCompBuf2, ERFC_R8);
    SyncV();
}

template <typename TileType>
TILEOP inline void ErfcComputeS(TileType& tmpCompBuf2, TileType& tmpCompBuf4, const TileType& z)
{
    pto::TADDS(tmpCompBuf2, z, ERFC_S1);
    SyncV();
    pto::TMUL(tmpCompBuf4, z, tmpCompBuf2);
    SyncV();
    pto::TADDS(tmpCompBuf2, tmpCompBuf4, ERFC_S2);
    SyncV();
    pto::TMUL(tmpCompBuf4, z, tmpCompBuf2);
    SyncV();
    pto::TADDS(tmpCompBuf2, tmpCompBuf4, ERFC_S3);
    SyncV();
    pto::TMUL(tmpCompBuf4, z, tmpCompBuf2);
    SyncV();
    pto::TADDS(tmpCompBuf2, tmpCompBuf4, ERFC_S4);
    SyncV();
    pto::TMUL(tmpCompBuf4, z, tmpCompBuf2);
    SyncV();
    pto::TADDS(tmpCompBuf2, tmpCompBuf4, ERFC_S5);
    SyncV();
}

template <typename TileType>
TILEOP inline void ErfcPublicSteps(TileType& tmpCompBuf1, TileType& tmpCompBuf2, TileType& tmpCompBuf3,
                                   TileType& tmpCompBuf4)
{
    ErfcComputeR(tmpCompBuf2, tmpCompBuf3, tmpCompBuf1);
    ErfcComputeS(tmpCompBuf2, tmpCompBuf4, tmpCompBuf1);

    pto::TDIV(tmpCompBuf2, tmpCompBuf3, tmpCompBuf2);
    SyncV();
    pto::TMUL(tmpCompBuf1, tmpCompBuf1, tmpCompBuf1);
    SyncV();
    pto::TMULS(tmpCompBuf1, tmpCompBuf1, ERFC_NEG_ONE);
    SyncV();
    pto::TEXP(tmpCompBuf1, tmpCompBuf1);
    SyncV();
    pto::TMUL(tmpCompBuf2, tmpCompBuf1, tmpCompBuf2);
    SyncV();
}

template <typename TileType>
TILEOP inline void ErfcPostCompute(TileType& dst, const TileType& xb, TileType& tmpCompBuf2, TileType& tmpCompBuf3)
{
    pto::TMULS(tmpCompBuf3, xb, ERFC_NEG_ONE);
    SyncV();
    pto::TADDS(tmpCompBuf3, tmpCompBuf3, ERFC_ONE);
    SyncV();
    pto::TMUL(tmpCompBuf2, tmpCompBuf2, xb);
    SyncV();
    pto::TADD(dst, tmpCompBuf2, tmpCompBuf3);
    SyncV();
}

#define OP_TILE_OP_ERFC TErfc
template <typename T0, typename T1, typename T2>
TILEOP void TErfc(T0 dst, T1 tmp, T2 src)
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
    if (shape0 == 0 || shape1 == 0 || shape2 == 0 || shape3 == 0 || shape4 == 0) {
        return;
    }

    using TmpFP32Tile = pto::Tile<pto::TileType::Vec, typename T0::Type, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    TmpFP32Tile dstTile(shape3, shape4);
    auto srcExecTile = MakeElementwiseOperandExecTile(dst, src);
    TmpFP32Tile tmpCompBuf1(shape3, shape4);
    TmpFP32Tile tmpCompBuf2(shape3, shape4);
    TmpFP32Tile tmpCompBuf3(shape3, shape4);
    TmpFP32Tile tmpCompBuf4(shape3, shape4);
    constexpr size_t TMP_BUFFER3_SLOT = 2;
    constexpr size_t TMP_BUFFER4_SLOT = 3;

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + GenTileOffset(dst, tileOffsets) * dstTypeSize));
                AssignElementwiseOperandExecTile(srcExecTile, src, tileOffsets);

                pto::TASSIGN(tmpCompBuf1, (uint64_t)(tmp.GetAddr()));
                pto::TASSIGN(tmpCompBuf2, (uint64_t)(tmp.GetAddr() + 1 * tileW * shape3 * dstTypeSize));
                pto::TASSIGN(tmpCompBuf3, (uint64_t)(tmp.GetAddr() + TMP_BUFFER3_SLOT * tileW * shape3 * dstTypeSize));
                pto::TASSIGN(tmpCompBuf4, (uint64_t)(tmp.GetAddr() + TMP_BUFFER4_SLOT * tileW * shape3 * dstTypeSize));

                ErfcClip(dstTile, srcExecTile);
                ErfcPreCompute(dstTile, dstTile, tmpCompBuf1);
                ErfcPublicSteps(tmpCompBuf1, tmpCompBuf2, tmpCompBuf3, tmpCompBuf4);
                ErfcPostCompute(dstTile, dstTile, tmpCompBuf2, tmpCompBuf3);
            }
        }
    }
}

#endif // TILEOP_TILE_OPERATOR_VEC_UNARY_ERF_H
