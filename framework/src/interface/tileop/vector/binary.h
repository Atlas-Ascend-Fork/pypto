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
 * \file binary.h
 * \brief Binary tile operation implementations.
 */

#ifndef TILEOP_TILE_OPERATOR_BINARY_H
#define TILEOP_TILE_OPERATOR_BINARY_H

#include "pair_binary.h"
#include "nan_compare.h"
#include "binary/basic.h"
#include "binary/extended.h"
#include "binary/floor_div.h"
#include "binary/gcd.h"
#include "binary/divmod.h"

template <int H, int W, typename D, typename A, typename B, typename M, typename S>
TILEOP void NanRepair(D& dst, A& a, B& b, M& mask, S& selector, uint64_t compareAddress)
{
    NanCompareMask<H, W>(mask, a, compareAddress);
    pto::TPARTSEL(dst, mask, dst, b, selector);
    SyncV();
    NanCompareMask<H, W>(mask, b, compareAddress);
    pto::TPARTSEL(dst, mask, dst, a, selector);
    SyncV();
}

template <typename T>
TILEOP auto MakeNanOverlapTile(T, unsigned h, unsigned w)
{
    using Physical = typename PtoTile<T>::Type;
    return pto::Tile<pto::TileType::Vec, typename Physical::DType, Physical::Rows, Physical::Cols,
                     pto::BLayout::RowMajor, -1, -1>(h, w);
}

template <int Axis, typename A, typename B>
TILEOP unsigned NanOverlap(A a, B b)
{
    const auto lhs = a.GetLayout().template GetShapeDim<Axis, 5>();
    const auto rhs = b.GetLayout().template GetShapeDim<Axis, 5>();
    return lhs < rhs ? lhs : rhs;
}

// PAIR preserves the non-overlapping tail; ordinary binary ops use the output valid shape.
template <bool Max, bool Pair, int H, int W, typename D, typename A, typename B, typename T>
TILEOP void TNanBinary(D dst, A a, B b, T tmp)
{
    if constexpr (Pair && Max) {
        TPairMax(dst, a, b);
    } else if constexpr (Pair) {
        TPairMin(dst, a, b);
    } else if constexpr (Max) {
        TMax(dst, a, b);
    } else {
        TMin(dst, a, b);
    }
    SyncV();
    constexpr size_t maskBytes = H * ((W + 255) / 256 * 32);
    constexpr size_t compareBytes = H * ((W * sizeof(typename D::Type) + 255) / 256 * 256);
    static_assert(maskBytes + 32 + compareBytes <= TileOp::GetTileSizeBytes<T>(),
                  "NaN binary scratch exceeds its planned capacity");
    using Mask = pto::Tile<pto::TileType::Vec, uint8_t, H, ((W + 7) / 8 + 31) / 32 * 32, pto::BLayout::RowMajor, -1,
                           -1>;
    using Selector = pto::Tile<pto::TileType::Vec, uint8_t, 1, 32>;
    const auto layout = dst.GetLayout();
    const auto d0 = Pair ? NanOverlap<0>(a, b) : layout.template GetShapeDim<0, 5>();
    const auto d1 = Pair ? NanOverlap<1>(a, b) : layout.template GetShapeDim<1, 5>();
    const auto d2 = Pair ? NanOverlap<2>(a, b) : layout.template GetShapeDim<2, 5>();
    const auto h = Pair ? NanOverlap<3>(a, b) : layout.template GetShapeDim<3, 5>();
    const auto w = Pair ? NanOverlap<4>(a, b) : layout.template GetShapeDim<4, 5>();
    if (h == 0 || w == 0)
        return;
    // Scratch is independent of runtime valid extents.
    Mask mask(h, (W + 7) / 8);
    Selector selector;
    pto::TASSIGN(mask, tmp.GetAddr());
    pto::TASSIGN(selector, tmp.GetAddr() + maskBytes);
    auto d = MakeNanOverlapTile(dst, h, w);
    auto x = MakeNanOverlapTile(a, h, w);
    auto y = MakeNanOverlapTile(b, h, w);
    for (LoopVar n0 = 0; n0 < d0; ++n0) {
        for (LoopVar n1 = 0; n1 < d1; ++n1) {
            for (LoopVar n2 = 0; n2 < d2; ++n2) {
                auto offset = TileOffset(n0, n1, n2);
                AssignElementwiseOperandExecTile(d, dst, offset);
                AssignElementwiseOperandExecTile(x, a, offset);
                AssignElementwiseOperandExecTile(y, b, offset);
                NanRepair<H, W>(d, x, y, mask, selector, tmp.GetAddr() + maskBytes + 32);
            }
        }
    }
}
#endif // TILEOP_TILE_OPERATOR_BINARY_H
