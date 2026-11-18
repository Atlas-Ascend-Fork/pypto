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
 * \file basic.h
 * \brief Binary tile operation implementations.
 */

#ifndef TILEOP_TILE_OPERATOR_VEC_BINARY_BASIC_H
#define TILEOP_TILE_OPERATOR_VEC_BINARY_BASIC_H

#include "utils/sync.h"
#include "../pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"
#include "../binary_brcinline.h"
#include "../unary/basic.h"

template <BinaryOp op, auto PrecisionType = 0, typename LastUse, typename T0, typename T1, typename T2>
TILEOP void BinaryComputeImpl(T0 dst, T1 src0, T2 src1)
{
    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;
    constexpr auto n3 = Std::tuple_element<DIM_3RD, LastUse>::type::value;
    if constexpr (op == BinaryOp::ADD) {
        PTO_WITH_LAST_USE(pto::TADD(dst, src0, src1), n1, n2, n3);
        return;
    }

    if constexpr (op == BinaryOp::SUB) {
        PTO_WITH_LAST_USE(pto::TSUB(dst, src0, src1), n1, n2, n3);
        return;
    }

    if constexpr (op == BinaryOp::MUL) {
        PTO_WITH_LAST_USE(pto::TMUL(dst, src0, src1), n1, n2, n3);
        return;
    }

    if constexpr (op == BinaryOp::DIV) {
        PTO_WITH_LAST_USE(pto::TDIV<PrecisionType>(dst, src0, src1), n1, n2, n3);
        return;
    }

    if constexpr (op == BinaryOp::MAX) {
        PTO_WITH_LAST_USE(pto::TMAX(dst, src0, src1), n1, n2, n3);
        return;
    }

    if constexpr (op == BinaryOp::MIN) {
        PTO_WITH_LAST_USE(pto::TMIN(dst, src0, src1), n1, n2, n3);
        return;
    }

    if constexpr (op == BinaryOp::BITWISEAND) {
        pto::TAND(dst, src0, src1);
        return;
    }

    if constexpr (op == BinaryOp::BITWISEOR) {
        pto::TOR(dst, src0, src1);
        return;
    }

    if constexpr (op == BinaryOp::EXPANDEXPDIF) {
        pto::TCOLEXPANDEXPDIF(dst, src0, src1);
        return;
    }

    if constexpr (op == BinaryOp::MOD) {
        pto::TFMOD<PrecisionType>(dst, src0, src1);
        return;
    }
}

template <BinaryOp op, auto PrecisionType = 0, BrcMode brcmode, typename LastUse, typename T0, typename T1, typename T2>
TILEOP void BinaryBrcDispatch(T0 dst, T1 src0, T2 src1)
{
    if constexpr (brcmode == BrcMode::BRC_W) {
        BinaryExpandDispatch<BinaryExpandMode::ROW, op, PrecisionType, LastUse>(dst, src0, src1);
    } else if constexpr (brcmode == BrcMode::BRC_H) {
        BinaryExpandDispatch<BinaryExpandMode::COL, op, PrecisionType, LastUse>(dst, src0, src1);
    } else if constexpr (brcmode == BrcMode::BRC_W0_H1) {
        pto::TCOLEXPAND(dst, src1);
        SyncV();
        BinaryExpandDispatch<BinaryExpandMode::ROW, op, PrecisionType, LastUse>(dst, src0, dst);
    } else if constexpr (brcmode == BrcMode::BRC_H0_W1) {
        pto::TCOLEXPAND(dst, src0);
        SyncV();
        BinaryExpandDispatch<BinaryExpandMode::ROW, op, PrecisionType, LastUse>(dst, dst, src1);
    } else if constexpr (brcmode == BrcMode::NONE) {
        BinaryComputeImpl<op, PrecisionType, LastUse>(dst, src0, src1);
    }
}

template <BrcMode brcmode, typename Src0Tensor, typename Src1Tensor, typename Src0TileInfo, typename Src1TileInfo,
          int... BrcOperands, typename T1, typename T2>
TILEOP void A5Expand1DimBrcWSrc(T1 src0, T2 src1, uint64_t src0Addr, uint64_t src1Addr)
{
    constexpr bool src0NeedExpand = brcmode == BrcMode::BRC_W &&
                                    GetBrcOperandAt<DIM_5TH, BrcOperands...>() == BRC_LEFT &&
                                    Std::tuple_size<typename Src0Tensor::Shape>::value == 1 && Src0TileInfo::tileW != 1;
    constexpr bool src1NeedExpand = brcmode == BrcMode::BRC_W &&
                                    GetBrcOperandAt<DIM_5TH, BrcOperands...>() == BRC_RIGHT &&
                                    Std::tuple_size<typename Src1Tensor::Shape>::value == 1 && Src1TileInfo::tileW != 1;
    if constexpr (src0NeedExpand) {
        using FillDst = pto::Tile<pto::TileType::Vec, typename T1::DType, T1::Rows, T1::Cols, pto::BLayout::RowMajor,
                                  T1::Rows, T1::Cols>;
        FillDst fillDst;
        pto::TASSIGN(fillDst, src0Addr);
        pto::TROWEXPAND(fillDst, src0);
    } else if constexpr (src1NeedExpand) {
        using FillDst = pto::Tile<pto::TileType::Vec, typename T2::DType, T2::Rows, T2::Cols, pto::BLayout::RowMajor,
                                  T2::Rows, T2::Cols>;
        FillDst fillDst;
        pto::TASSIGN(fillDst, src1Addr);
        pto::TROWEXPAND(fillDst, src1);
    }
}

template <BinaryOp op, auto PrecisionType = 0, typename LastUse, typename T0, typename T1, typename T2>
TILEOP void BinaryElementwiseCompute(T0 dst, T1 src0, T2 src1)
{
    if constexpr (TileOp::IsConstContinous<T0, T1, T2>() == true) {
        auto dstTile = PtoTile<T0, pto::BLayout::RowMajor, true>().Data();
        using Src0ExecDtype = typename ElementwiseOperandExecConfig<T0, T1>::OperandDtype;
        using Src1ExecDtype = typename ElementwiseOperandExecConfig<T0, T2>::OperandDtype;
        using Src0TileDefine = typename PtoTile<T0, pto::BLayout::RowMajor, true, Src0ExecDtype>::Type;
        using Src1TileDefine = typename PtoTile<T0, pto::BLayout::RowMajor, true, Src1ExecDtype>::Type;
        Src0TileDefine src0ExecTile;
        Src1TileDefine src1ExecTile;
        pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
        pto::TASSIGN(src0ExecTile, (uint64_t)src0.GetAddr());
        pto::TASSIGN(src1ExecTile, (uint64_t)src1.GetAddr());
        BinaryComputeImpl<op, PrecisionType, LastUse>(dstTile, src0ExecTile, src1ExecTile);
        return;
    }

    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstTile = PtoTile<T0>(dst);
    auto src0ExecTile = MakeElementwiseOperandExecTile(dst, src0);
    auto src1ExecTile = MakeElementwiseOperandExecTile(dst, src1);
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                AssignElementwiseOperandExecTile(src0ExecTile, src0, tileOffsets);
                AssignElementwiseOperandExecTile(src1ExecTile, src1, tileOffsets);
                BinaryComputeImpl<op, PrecisionType, LastUse>(dstTile.Data(), src0ExecTile, src1ExecTile);
            }
        }
    }
}

template <BinaryOp op, auto PrecisionType = 0, typename LastUse, int... BrcOperands, typename T0, typename T1,
          typename T2>
TILEOP void BinaryCompute(T0 dst, T1 src0, T2 src1)
{
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    if (shape0 == 0 || shape1 == 0 || shape2 == 0) {
        return;
    }
    using Src0TileInfo = TensorTileInfo<T1>;
    using Src1TileInfo = TensorTileInfo<T2>;
    constexpr BrcMode brcmode = GetBrcMode<BrcOperands...>();
    if constexpr (!TileOp::HasBrcOperand<BrcOperands...>() && op != BinaryOp::EXPANDEXPDIF) {
        BinaryElementwiseCompute<op, PrecisionType, LastUse>(dst, src0, src1);
        return;
    } else if constexpr (brcmode == BrcMode::BRC_HW) {
        BinaryMixBrcCompute<op, PrecisionType, Src0TileInfo, Src1TileInfo, LastUse, BrcOperands...>(dst, src0, src1);
        return;
    } else if constexpr (TileOp::IsConstContinous<T0, T1, T2>() == true && !TileOp::HasBrcOperand<BrcOperands...>()) {
        auto dstTile = PtoTile<T0, pto::BLayout::RowMajor, true>().Data();
        using Src0PtoTile = typename std::conditional<
            (Src0TileInfo::tileW == 1 && GetBrcOperandAt<DIM_5TH, BrcOperands...>() == BRC_LEFT),
            PtoTile<T1, pto::BLayout::ColMajor, true>, PtoTile<T1, pto::BLayout::RowMajor, true>>::type;
        using Src1PtoTile = typename std::conditional<
            (Src1TileInfo::tileW == 1 && GetBrcOperandAt<DIM_5TH, BrcOperands...>() == BRC_RIGHT),
            PtoTile<T2, pto::BLayout::ColMajor, true>, PtoTile<T2, pto::BLayout::RowMajor, true>>::type;
        auto src0Tile = Src0PtoTile().Data();
        auto src1Tile = Src1PtoTile().Data();
        pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
        pto::TASSIGN(src0Tile, (uint64_t)src0.GetAddr());
        pto::TASSIGN(src1Tile, (uint64_t)src1.GetAddr());
        BinaryBrcDispatch<op, PrecisionType, brcmode, LastUse>(dstTile, src0Tile, src1Tile);
        return;
    }

    using Src0PtoTile = typename std::conditional<(Src0TileInfo::tileW == 1 &&
                                                   GetBrcOperandAt<DIM_5TH, BrcOperands...>() == BRC_LEFT),
                                                  PtoTile<T1, pto::BLayout::ColMajor>, PtoTile<T1>>::type;
    using Src1PtoTile = typename std::conditional<(Src1TileInfo::tileW == 1 &&
                                                   GetBrcOperandAt<DIM_5TH, BrcOperands...>() == BRC_RIGHT),
                                                  PtoTile<T2, pto::BLayout::ColMajor>, PtoTile<T2>>::type;
    auto dstTile = PtoTile<T0>(dst);
    auto src0Tile = Src0PtoTile(src0);
    auto src1Tile = Src1PtoTile(src1);
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto dsttileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto src0tileOffsets = TileOffset(
                    (Src0TileInfo::tile0 == 1 || GetBrcOperandAt<DIM_1ST, BrcOperands...>() == BRC_LEFT) ? 0 : n0Index,
                    (Src0TileInfo::tile1 == 1 || GetBrcOperandAt<DIM_2ND, BrcOperands...>() == BRC_LEFT) ? 0 : n1Index,
                    (Src0TileInfo::tile2 == 1 || GetBrcOperandAt<DIM_3RD, BrcOperands...>() == BRC_LEFT) ? 0 : n2Index);
                auto src1tileOffsets = TileOffset(
                    (Src1TileInfo::tile0 == 1 || GetBrcOperandAt<DIM_1ST, BrcOperands...>() == BRC_RIGHT) ? 0 : n0Index,
                    (Src1TileInfo::tile1 == 1 || GetBrcOperandAt<DIM_2ND, BrcOperands...>() == BRC_RIGHT) ? 0 : n1Index,
                    (Src1TileInfo::tile2 == 1 || GetBrcOperandAt<DIM_3RD, BrcOperands...>() == BRC_RIGHT) ? 0 :
                                                                                                            n2Index);
                dstTile.Assign(dst, dsttileOffsets);
                src0Tile.Assign(src0, src0tileOffsets);
                src1Tile.Assign(src1, src1tileOffsets);
#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6) || defined(__LITE_NPU)
                if constexpr (GetBrcOperandAt<DIM_5TH, BrcOperands...>() != BRC_NONE) {
                    A5Expand1DimBrcWSrc<brcmode, T1, T2, Src0TileInfo, Src1TileInfo, BrcOperands...>(
                        src0Tile.Data(), src1Tile.Data(), (uint64_t)src0.GetAddr(), (uint64_t)src1.GetAddr());
                }
#endif
                BinaryBrcDispatch<op, PrecisionType, brcmode, LastUse>(dstTile.Data(), src0Tile.Data(),
                                                                       src1Tile.Data());
            }
        }
    }
}

#define OP_TILE_OP_ADD TAdd
template <typename LastUse = LastUse3Dim<0, 0, 0>, int... BrcOperands, typename T0, typename T1, typename T2>
TILEOP void TAdd(T0 dst, T1 src0, T2 src1)
{
    BinaryCompute<BinaryOp::ADD, 0, LastUse, BrcOperands...>(dst, src0, src1);
}

#define OP_TILE_OP_SUB TSub
template <typename LastUse = LastUse3Dim<0, 0, 0>, int... BrcOperands, typename T0, typename T1, typename T2>
TILEOP void TSub(T0 dst, T1 src0, T2 src1)
{
    BinaryCompute<BinaryOp::SUB, 0, LastUse, BrcOperands...>(dst, src0, src1);
}

#define OP_TILE_OP_MUL TMul
template <typename LastUse = LastUse3Dim<0, 0, 0>, int... BrcOperands, typename T0, typename T1, typename T2>
TILEOP void TMul(T0 dst, T1 src0, T2 src1)
{
    BinaryCompute<BinaryOp::MUL, 0, LastUse, BrcOperands...>(dst, src0, src1);
}

#define OP_TILE_OP_DIV TDiv
template <auto PrecisionType = pto::DivAlgorithm::DEFAULT, typename LastUse = LastUse3Dim<0, 0, 0>, int... BrcOperands,
          typename T0, typename T1, typename T2>
TILEOP void TDiv(T0 dst, T1 src0, T2 src1)
{
    BinaryCompute<BinaryOp::DIV, PrecisionType, LastUse, BrcOperands...>(dst, src0, src1);
}

#define OP_TILE_OP_MAX TMax
template <typename LastUse = LastUse3Dim<0, 0, 0>, int... BrcOperands, typename T0, typename T1, typename T2>
TILEOP void TMax(T0 dst, T1 src0, T2 src1)
{
    BinaryCompute<BinaryOp::MAX, 0, LastUse, BrcOperands...>(dst, src0, src1);
}

#define OP_TILE_OP_MIN TMin
template <typename LastUse = LastUse3Dim<0, 0, 0>, int... BrcOperands, typename T0, typename T1, typename T2>
TILEOP void TMin(T0 dst, T1 src0, T2 src1)
{
    BinaryCompute<BinaryOp::MIN, 0, LastUse, BrcOperands...>(dst, src0, src1);
}

#define OP_TILE_OP_BITWISEAND TBitwiseAnd
template <typename LastUse = LastUse3Dim<0, 0, 0>, int... BrcOperands, typename T0, typename T1, typename T2>
TILEOP void TBitwiseAnd(T0 dst, T1 src0, T2 src1)
{
    BinaryCompute<BinaryOp::BITWISEAND, 0, LastUse, BrcOperands...>(dst, src0, src1);
}

#define OP_TILE_OP_BITWISEOR TBitwiseOr
template <typename LastUse = LastUse3Dim<0, 0, 0>, int... BrcOperands, typename T0, typename T1, typename T2>
TILEOP void TBitwiseOr(T0 dst, T1 src0, T2 src1)
{
    BinaryCompute<BinaryOp::BITWISEOR, 0, LastUse, BrcOperands...>(dst, src0, src1);
}

#define OP_TILE_OP_Mod TMod
template <auto PrecisionType = pto::FmodAlgorithm::DEFAULT, typename LastUse = LastUse3Dim<0, 0, 0>, int... BrcOperands,
          typename T0, typename T1, typename T2>
TILEOP void TMod(T0 dst, T1 src0, T2 src1)
{
    BinaryCompute<BinaryOp::MOD, PrecisionType, LastUse, BrcOperands...>(dst, src0, src1);
}

#define OP_TILE_OP_AXPY TAxpy
template <int... BrcOperands, typename T0, typename T1, typename Scalar>
TILEOP void TAxpy(T0 dst, T1 src0, Scalar alpha)
{
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    if constexpr (!TileOp::HasBrcOperand<BrcOperands...>()) {
        auto dstTile = PtoTile<T0>(dst);
        auto src0ExecTile = MakeElementwiseOperandExecTile(dst, src0);
        for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
            for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
                for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                    auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                    dstTile.Assign(dst, tileOffsets);
                    AssignElementwiseOperandExecTile(src0ExecTile, src0, tileOffsets);
                    pto::TAXPY(dstTile.Data(), src0ExecTile, static_cast<typename T1::Type>(alpha));
                }
            }
        }
        return;
    }

    using SrcTileInfo = TensorTileInfo<T1>;

    using SrcPtoTile = typename std::conditional<(SrcTileInfo::tileW == 1 &&
                                                  GetBrcOperandAt<DIM_5TH, BrcOperands...>() == BRC_RIGHT),
                                                 PtoTile<T1, pto::BLayout::ColMajor>, PtoTile<T1>>::type;

    auto dstTile = PtoTile<T0>(dst);
    auto src0Tile = SrcPtoTile(src0);

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto dsttileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto src0tileOffsets = TileOffset(
                    (SrcTileInfo::tile0 == 1 || GetBrcOperandAt<DIM_1ST, BrcOperands...>() == BRC_RIGHT) ? 0 : n0Index,
                    (SrcTileInfo::tile1 == 1 || GetBrcOperandAt<DIM_2ND, BrcOperands...>() == BRC_RIGHT) ? 0 : n1Index,
                    (SrcTileInfo::tile2 == 1 || GetBrcOperandAt<DIM_3RD, BrcOperands...>() == BRC_RIGHT) ? 0 : n2Index);
                dstTile.Assign(dst, dsttileOffsets);
                src0Tile.Assign(src0, src0tileOffsets);
                pto::TAXPY(dstTile.Data(), src0Tile.Data(), static_cast<typename T1::Type>(alpha));
            }
        }
    }
}

#endif // TILEOP_TILE_OPERATOR_VEC_BINARY_BASIC_H
