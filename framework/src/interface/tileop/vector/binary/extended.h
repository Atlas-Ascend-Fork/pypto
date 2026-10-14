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
 * \file extended.h
 * \brief Binary tile operation implementations.
 */

#ifndef TILEOP_TILE_OPERATOR_VEC_BINARY_EXTENDED_H
#define TILEOP_TILE_OPERATOR_VEC_BINARY_EXTENDED_H

#include "utils/sync.h"
#include "basic.h"

template <BinaryOp op, auto PrecisionType = 0, typename T0, typename T1, typename T2, typename T3>
TILEOP void BinaryTmpComputeImpl(T0 dst, T1 src0, T2 src1, T3 tmp)
{
    if constexpr (op == BinaryOp::BITWISEXOR) {
        pto::TXOR(dst, src0, src1, tmp);
        return;
    }
    if constexpr (op == BinaryOp::POW) {
        pto::TPOW<PrecisionType>(dst, src0, src1, tmp);
        return;
    }
    if constexpr (op == BinaryOp::REM) {
        pto::TREM<PrecisionType>(dst, src0, src1, tmp);
        return;
    }
}

template <BinaryOp op, auto PrecisionType = 0, typename T0, typename T1, typename T2, typename T3>
TILEOP void BinaryTmpCompute(T0 dst, T1 src0, T2 src1, T3 tmp)
{
    if constexpr (op != BinaryOp::REM && TileOp::IsConstContinous<T0, T1, T2, T3>() == true) {
        auto dstTile = PtoTile<T0, pto::BLayout::RowMajor, true>().Data();
        using Src0ExecDtype = typename ElementwiseOperandExecConfig<T0, T1>::OperandDtype;
        using Src1ExecDtype = typename ElementwiseOperandExecConfig<T0, T2>::OperandDtype;
        using TmpExecDtype = typename ElementwiseOperandExecConfig<T0, T3>::OperandDtype;
        using Src0TileDefine = typename PtoTile<T0, pto::BLayout::RowMajor, true, Src0ExecDtype>::Type;
        using Src1TileDefine = typename PtoTile<T0, pto::BLayout::RowMajor, true, Src1ExecDtype>::Type;
        using TmpTileDefine = typename PtoTile<T0, pto::BLayout::RowMajor, true, TmpExecDtype>::Type;
        Src0TileDefine src0ExecTile;
        Src1TileDefine src1ExecTile;
        TmpTileDefine tmpExecTile;
        pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
        pto::TASSIGN(src0ExecTile, (uint64_t)src0.GetAddr());
        pto::TASSIGN(src1ExecTile, (uint64_t)src1.GetAddr());
        pto::TASSIGN(tmpExecTile, (uint64_t)tmp.GetAddr());
        BinaryTmpComputeImpl<op, PrecisionType>(dstTile, src0ExecTile, src1ExecTile, tmpExecTile);
        return;
    }
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto src0ExecTile = MakeElementwiseOperandExecTile(dst, src0);
    auto src1ExecTile = MakeElementwiseOperandExecTile(dst, src1);
    if constexpr (op == BinaryOp::REM) {
        auto tmpTile = PtoTile<T3>(tmp);
        for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
            for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
                for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                    auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                    dstTile.Assign(dst, tileOffsets);
                    AssignElementwiseOperandExecTile(src0ExecTile, src0, tileOffsets);
                    AssignElementwiseOperandExecTile(src1ExecTile, src1, tileOffsets);
                    tmpTile.Assign(tmp, tileOffsets);
                    BinaryTmpComputeImpl<op, PrecisionType>(dstTile.Data(), src0ExecTile, src1ExecTile, tmpTile.Data());
                }
            }
        }
    } else {
        auto tmpExecTile = MakeElementwiseOperandExecTile(dst, tmp);
        for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
            for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
                for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                    auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                    dstTile.Assign(dst, tileOffsets);
                    AssignElementwiseOperandExecTile(src0ExecTile, src0, tileOffsets);
                    AssignElementwiseOperandExecTile(src1ExecTile, src1, tileOffsets);
                    AssignElementwiseOperandExecTile(tmpExecTile, tmp, tileOffsets);
                    BinaryTmpComputeImpl<op, PrecisionType>(dstTile.Data(), src0ExecTile, src1ExecTile, tmpExecTile);
                }
            }
        }
    }
}

#define OP_TILE_OP_BITWISEXOR TBitwiseXor
template <typename T0, typename T1, typename T2, typename T3>
TILEOP void TBitwiseXor(T0 dst, T1 src0, T2 src1, T3 tmp)
{
    BinaryTmpCompute<BinaryOp::BITWISEXOR>(dst, src0, src1, tmp);
}

#ifdef __DAV_V220
template <typename DstTile, typename Src0Tile, typename Src1Tile, typename TmpTile, typename MaskTile>
TILEOP void TPowFloatTile(DstTile dstTile, Src0Tile src0Tile, Src1Tile src1Tile, TmpTile tmp0Tile, TmpTile tmp1Tile,
                          MaskTile mask0Tile, MaskTile mask1Tile, MaskTile selTmpTile)
{
    constexpr float scalarHalf = 0.5f;
    constexpr float scalarNegTwo = -2.0f;
    constexpr float scalarNegOne = -1.0f;
    constexpr float scalarOne = 1.0f;
    constexpr float scalarZero = 0.0f;
    const float nanValue = __builtin_nanf("");
    const float infValue = __builtin_huge_valf();
    const float negInfValue = -infValue;

    // mag = exp(y * log(|x|)) -> dst
    pto::TABS(tmp0Tile, src0Tile);
    SyncV();
    pto::TLOG(tmp0Tile, tmp0Tile);
    SyncV();
    pto::TMUL(tmp0Tile, src1Tile, tmp0Tile);
    SyncV();
    pto::TEXP(dstTile, tmp0Tile);
    SyncV();

    // parity = y - 2 * floor(0.5 * y) -> tmp0; odd integer exponent = (parity == 1) -> mask0
    pto::TMULS(tmp1Tile, src1Tile, scalarHalf);
    SyncV();
    pto::TCVT(tmp1Tile, tmp1Tile, pto::RoundMode::CAST_FLOOR);
    SyncV();
    pto::TMULS(tmp1Tile, tmp1Tile, scalarNegTwo);
    SyncV();
    pto::TADD(tmp0Tile, src1Tile, tmp1Tile);
    SyncV();
    pto::TCMPS(mask0Tile, tmp0Tile, scalarOne, pto::CmpMode::EQ);
    SyncV();

    // negative base -> mask1
    pto::TCMPS(mask1Tile, src0Tile, scalarZero, pto::CmpMode::LT);
    SyncV();

    // x < 0 and odd integer exponent: dst = -mag (mask0 = xneg AND isOdd, in-place)
    pto::TAND(mask0Tile, mask1Tile, mask0Tile);
    SyncV();
    pto::TMULS(tmp0Tile, dstTile, scalarNegOne);
    SyncV();
    pto::TPARTSEL(dstTile, mask0Tile, tmp0Tile, dstTile, selTmpTile);
    SyncV();

    // floor(y) -> tmp1; non-integer exponent = (y != floor(y)) -> mask0 (reused)
    pto::TCVT(tmp1Tile, src1Tile, pto::RoundMode::CAST_FLOOR);
    SyncV();
    pto::TCMP(mask0Tile, src1Tile, tmp1Tile, pto::CmpMode::NE);
    SyncV();

    // x < 0 (finite) and non-integer exponent: dst = NaN.
    // -inf ^ non-integer = +mag (already in dst), so exclude x == -inf via x > -inf.
    pto::TAND(mask1Tile, mask1Tile, mask0Tile); // mask1 = xneg AND !isInt
    SyncV();
    pto::TCMPS(mask0Tile, src0Tile, negInfValue, pto::CmpMode::GT);
    SyncV();
    pto::TAND(mask1Tile, mask1Tile, mask0Tile); // mask1 = xneg AND !isInt AND x > -inf
    SyncV();
    pto::TEXPANDS(tmp0Tile, nanValue);
    SyncV();
    pto::TPARTSEL(dstTile, mask1Tile, tmp0Tile, dstTile, selTmpTile);
    SyncV();

    // y == 0: dst = 1
    pto::TCMPS(mask0Tile, src1Tile, scalarZero, pto::CmpMode::EQ);
    SyncV();
    pto::TEXPANDS(tmp0Tile, scalarOne);
    SyncV();
    pto::TPARTSEL(dstTile, mask0Tile, tmp0Tile, dstTile, selTmpTile);
    SyncV();

    // x == 1: dst = 1 (tmp0 still holds 1)
    pto::TCMPS(mask0Tile, src0Tile, scalarOne, pto::CmpMode::EQ);
    SyncV();
    pto::TPARTSEL(dstTile, mask0Tile, tmp0Tile, dstTile, selTmpTile);
    SyncV();

    // x == -1 and y is +/-inf: dst = 1 (mag is NaN because log(1) == 0)
    pto::TABS(tmp1Tile, src1Tile);
    SyncV();
    pto::TCMPS(mask0Tile, src0Tile, scalarNegOne, pto::CmpMode::EQ);
    SyncV();
    pto::TCMPS(mask1Tile, tmp1Tile, infValue, pto::CmpMode::EQ);
    SyncV();
    pto::TAND(mask1Tile, mask0Tile, mask1Tile); // mask1 = x==-1 AND |y|==inf
    SyncV();
    pto::TPARTSEL(dstTile, mask1Tile, tmp0Tile, dstTile, selTmpTile);
    SyncV();
}
#endif

#define OP_TILE_OP_POW TPow
template <auto PrecisionType = pto::PowAlgorithm::DEFAULT, typename T0, typename T1, typename T2, typename T3>
TILEOP void TPow(T0 dst, T1 src0, T2 src1, T3 tmp)
{
#ifdef __DAV_V220
    if constexpr (std::is_same_v<typename T0::Type, float>) {
        const auto dstLayout = dst.GetLayout();
        auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
        auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
        auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
        auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
        auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
        if (dstShape0 == 0 || dstShape1 == 0 || dstShape2 == 0) {
            return;
        }
        constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
        constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
        constexpr auto src0TileH = TileOp::GetTensorTileShapeDim<T1, DIM_4TH, MAX_DIMS>();
        constexpr auto src0TileW = TileOp::GetTensorTileShapeDim<T1, DIM_5TH, MAX_DIMS>();
        constexpr auto src1TileH = TileOp::GetTensorTileShapeDim<T2, DIM_4TH, MAX_DIMS>();
        constexpr auto src1TileW = TileOp::GetTensorTileShapeDim<T2, DIM_5TH, MAX_DIMS>();
        constexpr auto computeTileH = dstTileH;
        constexpr auto computeTileW = dstTileW;
        constexpr auto dataTypeSize = sizeof(typename T0::Type);
        constexpr auto floatSlot = computeTileH * computeTileW * dataTypeSize; // bytes, 32B-aligned
        constexpr auto BITS_PER_BYTE = TileOp::BITS_PER_BYTE;
        constexpr auto MASK_ALIGNMENT = TileOp::BLOCK_SIZE;
        constexpr auto MASK_COUNT = 2;
        constexpr auto maskCols = ((computeTileW + BITS_PER_BYTE - 1) / BITS_PER_BYTE + MASK_ALIGNMENT - 1) /
                                  MASK_ALIGNMENT * MASK_ALIGNMENT;
        constexpr auto maskSlot = computeTileH * maskCols; // bytes
        constexpr auto maskBase = MASK_COUNT * floatSlot;
        using DstDataTile = pto::Tile<pto::TileType::Vec, float, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
        using Src0DataTile = pto::Tile<pto::TileType::Vec, float, src0TileH, src0TileW, pto::BLayout::RowMajor, -1, -1>;
        using Src1DataTile = pto::Tile<pto::TileType::Vec, float, src1TileH, src1TileW, pto::BLayout::RowMajor, -1, -1>;
        using TmpDataTile = pto::Tile<pto::TileType::Vec, float, computeTileH, computeTileW, pto::BLayout::RowMajor, -1,
                                      -1>;
        using MaskTile = pto::Tile<pto::TileType::Vec, uint8_t, computeTileH, maskCols, pto::BLayout::RowMajor, -1, -1>;
        DstDataTile dstTile(dstShape3, dstShape4);
        Src0DataTile src0Tile(dstShape3, dstShape4);
        Src1DataTile src1Tile(dstShape3, dstShape4);
        TmpDataTile tmp0Tile(dstShape3, dstShape4);
        TmpDataTile tmp1Tile(dstShape3, dstShape4);
        const auto validMaskCols = (dstShape4 + BITS_PER_BYTE - 1) / BITS_PER_BYTE;
        MaskTile mask0Tile(dstShape3, validMaskCols);
        MaskTile mask1Tile(dstShape3, validMaskCols);
        MaskTile selTmpTile(dstShape3, validMaskCols);
        for (LoopVar n0Index = 0; n0Index < dstShape0; n0Index++) {
            for (LoopVar n1Index = 0; n1Index < dstShape1; n1Index++) {
                for (LoopVar n2Index = 0; n2Index < dstShape2; n2Index++) {
                    auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                    auto dstOffset = GenTileOffset(dst, tileOffsets);
                    auto src0Offset = GenTileOffset(src0, tileOffsets);
                    auto src1Offset = GenTileOffset(src1, tileOffsets);
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dataTypeSize));
                    pto::TASSIGN(src0Tile, (uint64_t)(src0.GetAddr() + src0Offset * dataTypeSize));
                    pto::TASSIGN(src1Tile, (uint64_t)(src1.GetAddr() + src1Offset * dataTypeSize));
                    pto::TASSIGN(tmp0Tile, (uint64_t)(tmp.GetAddr() + 0 * floatSlot));
                    pto::TASSIGN(tmp1Tile, (uint64_t)(tmp.GetAddr() + 1 * floatSlot));
                    pto::TASSIGN(mask0Tile, (uint64_t)(tmp.GetAddr() + maskBase + 0 * maskSlot));
                    pto::TASSIGN(mask1Tile, (uint64_t)(tmp.GetAddr() + maskBase + 1 * maskSlot));
                    pto::TASSIGN(selTmpTile, (uint64_t)(tmp.GetAddr() + maskBase + MASK_COUNT * maskSlot));
                    TPowFloatTile(dstTile, src0Tile, src1Tile, tmp0Tile, tmp1Tile, mask0Tile, mask1Tile, selTmpTile);
                }
            }
        }
        return;
    }
#endif
    BinaryTmpCompute<BinaryOp::POW, PrecisionType>(dst, src0, src1, tmp);
}

#define OP_TILE_OP_REM TRem
template <auto PrecisionType = pto::RemAlgorithm::DEFAULT, typename T0, typename T1, typename T2, typename T3>
TILEOP void TRemainder(T0 dst, T1 src0, T2 src1, T3 tmp)
{
    BinaryTmpCompute<BinaryOp::REM, PrecisionType>(dst, src0, src1, tmp);
}

#endif // TILEOP_TILE_OPERATOR_VEC_BINARY_EXTENDED_H
