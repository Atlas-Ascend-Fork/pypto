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
 * \brief Basic binary-scalar tile operation implementations.
 */

#ifndef TILEOP_TILE_OPERATOR_VEC_BINARY_SCALAR_BASIC_H
#define TILEOP_TILE_OPERATOR_VEC_BINARY_SCALAR_BASIC_H

#include "utils/sync.h"
#include "../pto_tile.h"
#include "../nan_compare.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <BinaryScalarOp op, auto PrecisionType = 0, typename LastUse, typename T0, typename T1, typename Scalar>
TILEOP void BinaryScalarComputeImpl(T0 dst, T1 src0, Scalar src1)
{
    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;
    if constexpr (op == BinaryScalarOp::ADD) {
        PTO_WITH_LAST_USE(pto::TADDS(dst, src0, src1), n1, n2);
    }

    if constexpr (op == BinaryScalarOp::SUB) {
        if constexpr (std::is_same<Scalar, half>::value) {
            PTO_WITH_LAST_USE(
                pto::TADDS(dst, src0, static_cast<half>(static_cast<float>(-1) * static_cast<float>(src1))), n1, n2);
        } else {
            PTO_WITH_LAST_USE(pto::TADDS(dst, src0, -src1), n1, n2);
        }
    }

    if constexpr (op == BinaryScalarOp::MUL) {
        PTO_WITH_LAST_USE(pto::TMULS(dst, src0, src1), n1, n2);
    }

    if constexpr (op == BinaryScalarOp::DIV) {
        PTO_WITH_LAST_USE(pto::TDIVS<PrecisionType>(dst, src0, src1), n1, n2);
    }

    if constexpr (op == BinaryScalarOp::MAX) {
        PTO_WITH_LAST_USE(pto::TMAXS(dst, src0, src1), n1, n2);
    }

    if constexpr (op == BinaryScalarOp::MIN) {
        PTO_WITH_LAST_USE(pto::TMINS(dst, src0, src1), n1, n2);
    }

    if constexpr (op == BinaryScalarOp::BITWISEAND) {
        pto::TANDS(dst, src0, src1);
    }

    if constexpr (op == BinaryScalarOp::BITWISEOR) {
        pto::TORS(dst, src0, src1);
    }

    if constexpr (op == BinaryScalarOp::MOD) {
        pto::TFMODS<PrecisionType>(dst, src0, src1);
    }

    if constexpr (op == BinaryScalarOp::LRELU) {
        pto::TLRELU(dst, src0, src1);
    }
}

template <BinaryScalarOp op, auto PrecisionType = 0, typename LastUse, typename T0, typename T1, typename Scalar>
TILEOP void BinaryScalarCompute(T0 dst, T1 src0, Scalar src1)
{
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    if (shape0 == 0 || shape1 == 0 || shape2 == 0) {
        return;
    }

    auto dstTile = PtoTile<T0>(dst);
    auto src0Tile = PtoTile<T1>(src0);
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                src0Tile.Assign(src0, tileOffsets);
                BinaryScalarComputeImpl<op, PrecisionType, LastUse>(dstTile.Data(), src0Tile.Data(), src1);
            }
        }
    }
}

#define OP_TILE_OP_ADDS TAddS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TAddS(T0 dst, T1 src0, Scalar src1)
{
    BinaryScalarCompute<BinaryScalarOp::ADD, 0, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_SUBS TSubS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TSubS(T0 dst, T1 src0, Scalar src1)
{
    BinaryScalarCompute<BinaryScalarOp::SUB, 0, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_MULS TMulS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TMulS(T0 dst, T1 src0, Scalar src1)
{
    BinaryScalarCompute<BinaryScalarOp::MUL, 0, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_DIVS TDivS
template <auto PrecisionType = pto::DivAlgorithm::DEFAULT, typename LastUse = LastUse2Dim<0, 0>, typename Scalar,
          typename T0, typename T1>
TILEOP void TDivS(T0 dst, T1 src0, Scalar src1)
{
    BinaryScalarCompute<BinaryScalarOp::DIV, PrecisionType, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_MAXS TMaxS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TMaxS(T0 dst, T1 src0, Scalar src1)
{
    BinaryScalarCompute<BinaryScalarOp::MAX, 0, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_MINS TMinS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TMinS(T0 dst, T1 src0, Scalar src1)
{
    BinaryScalarCompute<BinaryScalarOp::MIN, 0, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_LRELU TLReLU
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TLReLU(T0 dst, T1 src0, Scalar src1)
{
    BinaryScalarCompute<BinaryScalarOp::LRELU, 0, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_BITWISEANDS TBitwiseAndS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TBitwiseAndS(T0 dst, T1 src0, Scalar src1)
{
    BinaryScalarCompute<BinaryScalarOp::BITWISEAND, 0, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_BITWISEORS TBitwiseOrS
template <typename LastUse = LastUse2Dim<0, 0>, typename Scalar, typename T0, typename T1>
TILEOP void TBitwiseOrS(T0 dst, T1 src0, Scalar src1)
{
    BinaryScalarCompute<BinaryScalarOp::BITWISEOR, 0, LastUse>(dst, src0, src1);
}

#define OP_TILE_OP_MODS TModS
template <auto PrecisionType = pto::FmodSAlgorithm::DEFAULT, typename LastUse = LastUse2Dim<0, 0>, typename Scalar,
          typename T0, typename T1>
TILEOP void TModS(T0 dst, T1 src0, Scalar src1)
{
    BinaryScalarCompute<BinaryScalarOp::MOD, PrecisionType, LastUse>(dst, src0, src1);
}

template <bool Max, int H, int W, typename D, typename X, typename T, typename Scalar>
TILEOP void TNanScalar(D dst, X src, T tmp, Scalar scalar)
{
    constexpr size_t maskBytes = H * ((W + 255) / 256 * 32);
    constexpr size_t compareBytes = H * ((W * sizeof(typename X::Type) + 255) / 256 * 256);
    static_assert(maskBytes + 32 + compareBytes <= TileOp::GetTileSizeBytes<T>(),
                  "NaN scalar scratch exceeds its planned capacity");
    using Mask = pto::Tile<pto::TileType::Vec, uint8_t, H, ((W + 7) / 8 + 31) / 32 * 32, pto::BLayout::RowMajor, -1,
                           -1>;
    using Selector = pto::Tile<pto::TileType::Vec, uint8_t, 1, 32>;
    auto dl = dst.GetLayout();
    if (dl.template GetShapeDim<3, 5>() == 0 || dl.template GetShapeDim<4, 5>() == 0)
        return;
    Mask mask(dl.template GetShapeDim<3, 5>(), (W + 7) / 8);
    Selector selector;
    pto::TASSIGN(mask, tmp.GetAddr());
    pto::TASSIGN(selector, tmp.GetAddr() + maskBytes);
    auto d = PtoTile<D>(dst);
    auto x = PtoTile<X>(src);
    for (LoopVar n0 = 0; n0 < dl.template GetShapeDim<0, 5>(); ++n0)
        for (LoopVar n1 = 0; n1 < dl.template GetShapeDim<1, 5>(); ++n1)
            for (LoopVar n2 = 0; n2 < dl.template GetShapeDim<2, 5>(); ++n2) {
                auto off = TileOffset(n0, n1, n2);
                d.Assign(dst, off);
                x.Assign(src, off);
                auto out = d.Data();
                auto in = x.Data();
                NanCompareMask<H, W>(mask, in, tmp.GetAddr() + maskBytes + 32);
                if constexpr (Max)
                    pto::TMAXS(out, in, static_cast<typename X::Type>(scalar));
                else
                    pto::TMINS(out, in, static_cast<typename X::Type>(scalar));
                SyncV();
                pto::TSELS(out, mask, out, selector, static_cast<typename X::Type>(scalar));
                SyncV();
            }
}
#endif // TILEOP_TILE_OPERATOR_VEC_BINARY_SCALAR_BASIC_H
