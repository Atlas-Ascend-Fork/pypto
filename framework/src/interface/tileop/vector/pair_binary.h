/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pair_binary.h
 * \brief
 */
#ifndef TILEOP_TILE_OPERATOR_PAIR_BINARY_H
#define TILEOP_TILE_OPERATOR_PAIR_BINARY_H

#include "utils/layout.h"
#include "utils/sync.h"
#include "pto_tile.h"
#include "utils/tile_tensor.h"

template <int W, typename D, typename S>
TILEOP void CopyPairArgCompareInput(D& dst, S& src, uint64_t addr)
{
    using E = typename D::DType;
    constexpr int repeat = 256 / sizeof(E);
    constexpr int tailStart = W / repeat * repeat;
    pto::TEXTRACT(dst, src, 0, 0);
    SyncV();
    const auto h = dst.GetValidRow();
    const auto w = dst.GetValidCol();
    if (w > tailStart) {
        using Tail = pto::Tile<pto::TileType::Vec, E, 1, repeat, pto::BLayout::RowMajor, 1, -1>;
        using Padded = pto::Tile<pto::TileType::Vec, E, 1, repeat, pto::BLayout::RowMajor, 1, repeat,
                                 pto::SLayout::NoneBox, 512, pto::PadValue::Zero>;
        const auto tailAddr = addr + ((h - 1) * W + tailStart) * sizeof(E);
        Tail tail(w - tailStart);
        Padded padded;
        pto::TASSIGN(tail, tailAddr);
        pto::TASSIGN(padded, tailAddr);
        pto::TFILLPAD_INPLACE(padded, tail);
        SyncV();
    }
}

template <bool Max, bool Finalize, int H, int W, typename V, typename I, typename A, typename AI, typename B,
          typename BI, typename T>
TILEOP void NanPairArgTile(V& value, I& index, A& a, AI& ai, B& b, BI& bi, T& tmp)
{
    constexpr size_t maskBytes = H * ((W + 255) / 256 * 32);
    constexpr size_t rowBytes = W * sizeof(int32_t);
    constexpr size_t indexBytes = H * rowBytes;
    constexpr size_t compareOffset = maskBytes + indexBytes + 32;
    // rowPad covers short rows. A wide partial repeat needs a guarded copy.
    constexpr bool GuardTail = rowBytes > 256 && rowBytes % 256 != 0;
    constexpr size_t compareBytes = GuardTail ? indexBytes + 256 - rowBytes % 256 : 0;
    static_assert(compareOffset + 2 * compareBytes <= TileOp::GetTileSizeBytes<T>(),
                  "NaN pair arg scratch exceeds its planned capacity");
    using Mask = pto::Tile<pto::TileType::Vec, uint8_t, H, ((W + 7) / 8 + 31) / 32 * 32, pto::BLayout::RowMajor, -1,
                           -1>;
    using Tie = pto::Tile<pto::TileType::Vec, int32_t, H, W, pto::BLayout::RowMajor, -1, -1>;
    using CompareValue = pto::Tile<pto::TileType::Vec, typename V::DType, H, W, pto::BLayout::RowMajor, -1, -1>;
    using Selector = pto::Tile<pto::TileType::Vec, uint8_t, 1, 32>;
    const auto h = value.GetValidRow();
    const auto w = value.GetValidCol();
    if (h == 0 || w == 0) {
        return;
    }
    Mask mask(h, (w + 7) / 8);
    Tie tie(h, w);
    Selector selector;
    const auto scratch = tmp.GetAddr();
    pto::TASSIGN(mask, scratch);
    pto::TASSIGN(tie, scratch + maskBytes);
    pto::TASSIGN(selector, scratch + maskBytes + indexBytes);
    CompareValue cmpA(h, w), cmpB(h, w);
    const auto compareAddr = scratch + compareOffset;
    if constexpr (GuardTail) {
        pto::TASSIGN(cmpA, compareAddr);
        pto::TASSIGN(cmpB, compareAddr + compareBytes);
        CopyPairArgCompareInput<W>(cmpA, a, compareAddr);
        CopyPairArgCompareInput<W>(cmpB, b, compareAddr + compareBytes);
        pto::TCMP(mask, cmpA, cmpB, Max ? pto::CmpMode::GE : pto::CmpMode::LE);
    } else {
        pto::TCMP(mask, a, b, Max ? pto::CmpMode::GE : pto::CmpMode::LE);
    }
    SyncV();
    pto::TPARTSEL(index, mask, ai, bi, selector);
    SyncV();
    if constexpr (Max)
        pto::TMAX(value, a, b);
    else
        pto::TMIN(value, a, b);
    SyncV();
    pto::TMIN(tie, ai, bi);
    SyncV();
    if constexpr (GuardTail)
        pto::TCMP(mask, cmpA, cmpB, pto::CmpMode::EQ);
    else
        pto::TCMP(mask, a, b, pto::CmpMode::EQ);
    SyncV();
    pto::TPARTSEL(index, mask, tie, index, selector);
    SyncV();
    // Invalid partials already lose through the identity value and minimum-index tie break.
    if constexpr (Finalize) {
        // Only the final pair converts an all-invalid result to index zero.
        Tie compareIndex(h, w);
        if constexpr (GuardTail) {
            pto::TASSIGN(compareIndex, compareAddr);
            CopyPairArgCompareInput<W>(compareIndex, ai, compareAddr);
            pto::TCMPS(mask, compareIndex, INT32_MAX, pto::CmpMode::EQ);
        } else {
            pto::TCMPS(mask, ai, INT32_MAX, pto::CmpMode::EQ);
        }
        SyncV();
        pto::TNOT(mask, mask);
        SyncV();
        pto::TSELS(tie, mask, ai, selector, 0);
        SyncV();
        if constexpr (GuardTail) {
            CopyPairArgCompareInput<W>(compareIndex, bi, compareAddr);
            pto::TCMPS(mask, compareIndex, INT32_MAX, pto::CmpMode::EQ);
        } else {
            pto::TCMPS(mask, bi, INT32_MAX, pto::CmpMode::EQ);
        }
        SyncV();
        pto::TNOT(mask, mask);
        SyncV();
        pto::TPARTSEL(index, mask, index, tie, selector);
        SyncV();
    }
}

template <typename T0, typename T1, size_t index = 0, size_t shapeSize = 0>
TILEOP int GetReduceAxisIndex(T0 src0, T1 src1)
{
    // 使用validshape进行比较，如果index大于等于shapesize，仍然没有找到reduce轴，则默认R轴为首轴，合并成RA进行处理
    if constexpr (index < shapeSize) {
        const auto src0Layout = src0.GetLayout();
        const auto src1Layout = src1.GetLayout();
        auto src0CurShape = src0Layout.template GetShapeDim<index, shapeSize>();
        auto src1CurShape = src1Layout.template GetShapeDim<index, shapeSize>();
        if (src0CurShape != src1CurShape) {
            return index;
        }
        return GetReduceAxisIndex<T0, T1, index + 1, shapeSize>(src0, src1);
    } else {
        return 0;
    }
}

// 左闭右开
template <int startIndex, int endIndex, typename TileShape>
TILEOP constexpr int GetTileShapeMergeResult()
{
    if constexpr (startIndex >= endIndex) {
        return 1;
    } else {
        constexpr auto curShape = Std::tuple_element<startIndex, TileShape>::type::value;
        return curShape * GetTileShapeMergeResult<startIndex + 1, endIndex, TileShape>();
    }
}

template <int reduceAxisIndex, size_t shapeSize, bool isHeight, typename T0>
TILEOP constexpr int GetOneDimensionTileSize()
{
    // R轴非尾轴
    if constexpr (reduceAxisIndex < shapeSize - 1) {
        if constexpr (isHeight) {
            return Std::tuple_element<reduceAxisIndex, typename T0::TileShape>::type::value;
        } else {
            return GetTileShapeMergeResult<reduceAxisIndex + 1, shapeSize, typename T0::TileShape>();
        }
    } else {
        if constexpr (isHeight) {
            return TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
        } else {
            return TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
        }
    }
}

template <int reduceAxisIndex, size_t shapeSize, typename T0>
TILEOP int GetHeightValidSize(T0 tensor)
{
    const auto tensorLayout = tensor.GetLayout();
    if constexpr (reduceAxisIndex == shapeSize - 1) {
        return tensorLayout.template GetShapeDim<reduceAxisIndex - 1, shapeSize>();
    } else {
        return tensorLayout.template GetShapeDim<reduceAxisIndex, shapeSize>();
    }
}

template <int reduceAxisIndex, size_t shapeSize, typename T0>
TILEOP int GetWidthValidSize(T0 tensor)
{
    const auto tensorLayout = tensor.GetLayout();
    if constexpr (reduceAxisIndex == shapeSize - 1) {
        return tensorLayout.template GetShapeDim<reduceAxisIndex, shapeSize>();
    } else {
        return GetTileShapeMergeResult<reduceAxisIndex + 1, shapeSize, typename T0::TileShape>();
    }
}

template <PairBinaryOp op, typename T0, typename T1, typename T2>
TILEOP void PairBinaryComputeImpl(T0 dst, T1 src0, T2 src1)
{
    if constexpr (op == PairBinaryOp::ADD) {
        pto::TPARTADD(dst, src0, src1);
        return;
    }
    if constexpr (op == PairBinaryOp::MAX) {
        pto::TPARTMAX(dst, src0, src1);
        return;
    }
    if constexpr (op == PairBinaryOp::MIN) {
        pto::TPARTMIN(dst, src0, src1);
        return;
    }
    if constexpr (op == PairBinaryOp::MUL) {
        pto::TPARTMUL(dst, src0, src1);
        return;
    }
}

template <PairBinaryOp op, int reduceAxisIndex, typename T0, typename T1, typename T2>
TILEOP void InnerPairBinaryCompute(T0 dst, T1 src0, T2 src1)
{
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
    if constexpr (reduceAxisIndex < shapeSize) {
        const auto dstLayout = dst.GetLayout();
        const auto src0Layout = src0.GetLayout();
        const auto src1Layout = src1.GetLayout();
        constexpr auto dstTypeSize = sizeof(typename T0::Type);
        constexpr auto src0TypeSize = sizeof(typename T1::Type);
        constexpr auto src1TypeSize = sizeof(typename T2::Type);

        size_t dstShape0 = 1;
        size_t dstShape1 = 1;
        size_t dstShape2 = 1;
        size_t dstStride0 = 0;
        size_t dstStride1 = 0;
        size_t dstStride2 = 0;
        size_t src0Stride0 = 0;
        size_t src0Stride1 = 0;
        size_t src0Stride2 = 0;
        size_t src1Stride0 = 0;
        size_t src1Stride1 = 0;
        size_t src1Stride2 = 0;

        if constexpr ((reduceAxisIndex == shapeSize - 1 && reduceAxisIndex >= DIM_3RD) ||
                      (reduceAxisIndex < shapeSize - 1 && reduceAxisIndex >= 1)) {
            dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, shapeSize>();
            dstStride0 = dstLayout.template GetStrideDim<DIM_1ST, shapeSize>();
            src0Stride0 = src0Layout.template GetStrideDim<DIM_1ST, shapeSize>();
            src1Stride0 = src1Layout.template GetStrideDim<DIM_1ST, shapeSize>();
        }
        if constexpr ((reduceAxisIndex == shapeSize - 1 && reduceAxisIndex >= DIM_4TH) ||
                      (reduceAxisIndex < shapeSize - 1 && reduceAxisIndex >= DIM_3RD)) {
            dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, shapeSize>();
            dstStride1 = dstLayout.template GetStrideDim<DIM_2ND, shapeSize>();
            src0Stride1 = src0Layout.template GetStrideDim<DIM_2ND, shapeSize>();
            src1Stride1 = src1Layout.template GetStrideDim<DIM_2ND, shapeSize>();
        }
        if constexpr ((reduceAxisIndex == shapeSize - 1 && reduceAxisIndex >= DIM_5TH) ||
                      (reduceAxisIndex < shapeSize - 1 && reduceAxisIndex >= DIM_4TH)) {
            dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, shapeSize>();
            dstStride2 = dstLayout.template GetStrideDim<DIM_3RD, shapeSize>();
            src0Stride2 = src0Layout.template GetStrideDim<DIM_3RD, shapeSize>();
            src1Stride2 = src1Layout.template GetStrideDim<DIM_3RD, shapeSize>();
        }

        constexpr auto dstTileH = GetOneDimensionTileSize<reduceAxisIndex, shapeSize, true, T0>();
        constexpr auto dstTileW = GetOneDimensionTileSize<reduceAxisIndex, shapeSize, false, T0>();
        constexpr auto src0TileH = GetOneDimensionTileSize<reduceAxisIndex, shapeSize, true, T1>();
        constexpr auto src0TileW = GetOneDimensionTileSize<reduceAxisIndex, shapeSize, false, T1>();
        constexpr auto src1TileH = GetOneDimensionTileSize<reduceAxisIndex, shapeSize, true, T2>();
        constexpr auto src1TileW = GetOneDimensionTileSize<reduceAxisIndex, shapeSize, false, T2>();
        int dstShape3;
        int src0Shape3;
        int src1Shape3;
        if constexpr (shapeSize != 1) {
            dstShape3 = GetHeightValidSize<reduceAxisIndex, shapeSize, T0>(dst);
            src0Shape3 = GetHeightValidSize<reduceAxisIndex, shapeSize, T1>(src0);
            src1Shape3 = GetHeightValidSize<reduceAxisIndex, shapeSize, T2>(src1);
        } else {
            dstShape3 = 1;
            src0Shape3 = 1;
            src1Shape3 = 1;
        }

        auto dstShape4 = GetWidthValidSize<reduceAxisIndex, shapeSize, T0>(dst);
        auto src0Shape4 = GetWidthValidSize<reduceAxisIndex, shapeSize, T1>(src0);
        auto src1Shape4 = GetWidthValidSize<reduceAxisIndex, shapeSize, T2>(src1);

        using DstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW,
                                        pto::BLayout::RowMajor, -1, -1>;
        using Src0TileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, src0TileH, src0TileW,
                                         pto::BLayout::RowMajor, -1, -1>;
        using Src1TileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, src1TileH, src1TileW,
                                         pto::BLayout::RowMajor, -1, -1>;

        for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
            for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
                for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                    DstTileDefine dstTile(dstShape3, dstShape4);
                    Src0TileDefine src0Tile(src0Shape3, src0Shape4);
                    Src1TileDefine src1Tile(src1Shape3, src1Shape4);
                    auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                    auto src0Offset = n0Index * src0Stride0 + n1Index * src0Stride1 + n2Index * src0Stride2;
                    auto src1Offset = n0Index * src1Stride0 + n1Index * src1Stride1 + n2Index * src1Stride2;
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                    pto::TASSIGN(src0Tile, (uint64_t)(src0.GetAddr() + src0Offset * src0TypeSize));
                    pto::TASSIGN(src1Tile, (uint64_t)(src1.GetAddr() + src1Offset * src1TypeSize));
                    PairBinaryComputeImpl<op>(dstTile, src0Tile, src1Tile);
                }
            }
        }
    }
}

template <PairBinaryOp op, int reduceAxisIndex, typename T0, typename T1, typename T2, typename T3, typename T4,
          typename T5>
TILEOP void InnerPairArgBinaryCompute(T0 dstValue, T1 dstIndex, T2 src0Value, T3 src0Index, T4 src1Value, T5 src1Index)
{
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
    if constexpr (reduceAxisIndex < shapeSize) {
        const auto dstLayout = dstValue.GetLayout();
        const auto src0Layout = src0Value.GetLayout();
        const auto src1Layout = src1Value.GetLayout();
        constexpr auto dstTypeSize = sizeof(typename T0::Type);
        constexpr auto src0TypeSize = sizeof(typename T2::Type);
        constexpr auto src1TypeSize = sizeof(typename T4::Type);

        size_t dstShape0 = 1;
        size_t dstShape1 = 1;
        size_t dstShape2 = 1;
        size_t dstStride0 = 0;
        size_t dstStride1 = 0;
        size_t dstStride2 = 0;
        size_t src0Stride0 = 0;
        size_t src0Stride1 = 0;
        size_t src0Stride2 = 0;
        size_t src1Stride0 = 0;
        size_t src1Stride1 = 0;
        size_t src1Stride2 = 0;

        if constexpr ((reduceAxisIndex == shapeSize - 1 && reduceAxisIndex >= DIM_3RD) ||
                      (reduceAxisIndex < shapeSize - 1 && reduceAxisIndex >= 1)) {
            dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, shapeSize>();
            dstStride0 = dstLayout.template GetStrideDim<DIM_1ST, shapeSize>();
            src0Stride0 = src0Layout.template GetStrideDim<DIM_1ST, shapeSize>();
            src1Stride0 = src1Layout.template GetStrideDim<DIM_1ST, shapeSize>();
        }
        if constexpr ((reduceAxisIndex == shapeSize - 1 && reduceAxisIndex >= DIM_4TH) ||
                      (reduceAxisIndex < shapeSize - 1 && reduceAxisIndex >= DIM_3RD)) {
            dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, shapeSize>();
            dstStride1 = dstLayout.template GetStrideDim<DIM_2ND, shapeSize>();
            src0Stride1 = src0Layout.template GetStrideDim<DIM_2ND, shapeSize>();
            src1Stride1 = src1Layout.template GetStrideDim<DIM_2ND, shapeSize>();
        }
        if constexpr ((reduceAxisIndex == shapeSize - 1 && reduceAxisIndex >= DIM_5TH) ||
                      (reduceAxisIndex < shapeSize - 1 && reduceAxisIndex >= DIM_4TH)) {
            dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, shapeSize>();
            dstStride2 = dstLayout.template GetStrideDim<DIM_3RD, shapeSize>();
            src0Stride2 = src0Layout.template GetStrideDim<DIM_3RD, shapeSize>();
            src1Stride2 = src1Layout.template GetStrideDim<DIM_3RD, shapeSize>();
        }

        constexpr auto dstTileH = GetOneDimensionTileSize<reduceAxisIndex, shapeSize, true, T0>();
        constexpr auto dstTileW = GetOneDimensionTileSize<reduceAxisIndex, shapeSize, false, T0>();
        constexpr auto src0TileH = GetOneDimensionTileSize<reduceAxisIndex, shapeSize, true, T2>();
        constexpr auto src0TileW = GetOneDimensionTileSize<reduceAxisIndex, shapeSize, false, T2>();
        constexpr auto src1TileH = GetOneDimensionTileSize<reduceAxisIndex, shapeSize, true, T4>();
        constexpr auto src1TileW = GetOneDimensionTileSize<reduceAxisIndex, shapeSize, false, T4>();
        int dstShape3;
        int src0Shape3;
        int src1Shape3;
        if constexpr (shapeSize != 1) {
            dstShape3 = GetHeightValidSize<reduceAxisIndex, shapeSize, T0>(dstValue);
            src0Shape3 = GetHeightValidSize<reduceAxisIndex, shapeSize, T2>(src0Value);
            src1Shape3 = GetHeightValidSize<reduceAxisIndex, shapeSize, T4>(src1Value);
        } else {
            dstShape3 = 1;
            src0Shape3 = 1;
            src1Shape3 = 1;
        }

        auto dstShape4 = GetWidthValidSize<reduceAxisIndex, shapeSize, T0>(dstValue);
        auto src0Shape4 = GetWidthValidSize<reduceAxisIndex, shapeSize, T2>(src0Value);
        auto src1Shape4 = GetWidthValidSize<reduceAxisIndex, shapeSize, T4>(src1Value);

        using DstValueTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW,
                                             pto::BLayout::RowMajor, -1, -1>;
        using DstIndexTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, dstTileH, dstTileW,
                                             pto::BLayout::RowMajor, -1, -1>;
        using Src0ValueTileDefine = pto::Tile<pto::TileType::Vec, typename T2::Type, src0TileH, src0TileW,
                                              pto::BLayout::RowMajor, -1, -1>;
        using Src0IndexTileDefine = pto::Tile<pto::TileType::Vec, typename T3::Type, src0TileH, src0TileW,
                                              pto::BLayout::RowMajor, -1, -1>;
        using Src1ValueTileDefine = pto::Tile<pto::TileType::Vec, typename T4::Type, src1TileH, src1TileW,
                                              pto::BLayout::RowMajor, -1, -1>;
        using Src1IndexTileDefine = pto::Tile<pto::TileType::Vec, typename T5::Type, src1TileH, src1TileW,
                                              pto::BLayout::RowMajor, -1, -1>;

        for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
            for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
                for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                    DstValueTileDefine dstValueTile(dstShape3, dstShape4);
                    DstIndexTileDefine dstIndexTile(dstShape3, dstShape4);
                    Src0ValueTileDefine src0ValueTile(src0Shape3, src0Shape4);
                    Src0IndexTileDefine src0IndexTile(src0Shape3, src0Shape4);
                    Src1ValueTileDefine src1ValueTile(src1Shape3, src1Shape4);
                    Src1IndexTileDefine src1IndexTile(src1Shape3, src1Shape4);
                    auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                    auto src0Offset = n0Index * src0Stride0 + n1Index * src0Stride1 + n2Index * src0Stride2;
                    auto src1Offset = n0Index * src1Stride0 + n1Index * src1Stride1 + n2Index * src1Stride2;
                    pto::TASSIGN(dstValueTile, (uint64_t)(dstValue.GetAddr() + dstOffset * dstTypeSize));
                    pto::TASSIGN(dstIndexTile, (uint64_t)(dstIndex.GetAddr() + dstOffset * dstTypeSize));
                    pto::TASSIGN(src0ValueTile, (uint64_t)(src0Value.GetAddr() + src0Offset * src0TypeSize));
                    pto::TASSIGN(src0IndexTile, (uint64_t)(src0Index.GetAddr() + src0Offset * src0TypeSize));
                    pto::TASSIGN(src1ValueTile, (uint64_t)(src1Value.GetAddr() + src1Offset * src1TypeSize));
                    pto::TASSIGN(src1IndexTile, (uint64_t)(src1Index.GetAddr() + src1Offset * src1TypeSize));
                    if constexpr (op == PairBinaryOp::ARGMAX) {
                        pto::TPARTARGMAX(dstValueTile, src0ValueTile, src1ValueTile, dstIndexTile, src0IndexTile,
                                         src1IndexTile);

                    } else {
                        pto::TPARTARGMIN(dstValueTile, src0ValueTile, src1ValueTile, dstIndexTile, src0IndexTile,
                                         src1IndexTile);
                    }
                }
            }
        }
    }
}

template <PairBinaryOp op, typename T0, typename T1, typename T2>
TILEOP void PairBinaryCompute(T0 dst, T1 src0, T2 src1)
{
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
    size_t reduceAxisIndex = GetReduceAxisIndex<T1, T2, 0, shapeSize>(src0, src1);
    if (reduceAxisIndex == 0) {
        InnerPairBinaryCompute<op, 0>(dst, src0, src1);
    } else if (reduceAxisIndex == 1) {
        InnerPairBinaryCompute<op, 1>(dst, src0, src1);
    } else if (reduceAxisIndex == DIM_3RD) {
        InnerPairBinaryCompute<op, DIM_3RD>(dst, src0, src1);
    } else if (reduceAxisIndex == DIM_4TH) {
        InnerPairBinaryCompute<op, DIM_4TH>(dst, src0, src1);
    } else if (reduceAxisIndex == DIM_5TH) {
        InnerPairBinaryCompute<op, DIM_5TH>(dst, src0, src1);
    }
}

template <PairBinaryOp op, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
TILEOP void PairArgBinaryCompute(T0 dstValue, T1 dstIndex, T2 src0Value, T3 src0Index, T4 src1Value, T5 src1Index)
{
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
    size_t reduceAxisIndex = GetReduceAxisIndex<T2, T4, 0, shapeSize>(src0Value, src1Value);
    if (reduceAxisIndex == 0) {
        InnerPairArgBinaryCompute<op, 0>(dstValue, dstIndex, src0Value, src0Index, src1Value, src1Index);
    } else if (reduceAxisIndex == 1) {
        InnerPairArgBinaryCompute<op, 1>(dstValue, dstIndex, src0Value, src0Index, src1Value, src1Index);
    } else if (reduceAxisIndex == DIM_3RD) {
        InnerPairArgBinaryCompute<op, DIM_3RD>(dstValue, dstIndex, src0Value, src0Index, src1Value, src1Index);
    } else if (reduceAxisIndex == DIM_4TH) {
        InnerPairArgBinaryCompute<op, DIM_4TH>(dstValue, dstIndex, src0Value, src0Index, src1Value, src1Index);
    } else if (reduceAxisIndex == DIM_5TH) {
        InnerPairArgBinaryCompute<op, DIM_5TH>(dstValue, dstIndex, src0Value, src0Index, src1Value, src1Index);
    }
}

#define OP_TILE_OP_PAIRSUM TPairSum
template <typename T0, typename T1, typename T2>
TILEOP void TPairSum(T0 dst, T1 src0, T2 src1)
{
    PairBinaryCompute<PairBinaryOp::ADD>(dst, src0, src1);
}

#define OP_TILE_OP_PAIRMAX TPairMax
template <typename T0, typename T1, typename T2>
TILEOP void TPairMax(T0 dst, T1 src0, T2 src1)
{
    PairBinaryCompute<PairBinaryOp::MAX>(dst, src0, src1);
}

#define OP_TILE_OP_PAIRMIN TPairMin
template <typename T0, typename T1, typename T2>
TILEOP void TPairMin(T0 dst, T1 src0, T2 src1)
{
    PairBinaryCompute<PairBinaryOp::MIN>(dst, src0, src1);
}

#define OP_TILE_OP_PAIRPROD TPairProd
template <typename T0, typename T1, typename T2>
TILEOP void TPairProd(T0 dst, T1 src0, T2 src1)
{
    PairBinaryCompute<PairBinaryOp::MUL>(dst, src0, src1);
}

#define OP_TILE_OP_PAIRARGMAX TPairArgMax
template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
TILEOP void TPairArgMax(T0 dstValue, T1 dstIndex, T2 src0Value, T3 src0Index, T4 src1Value, T5 src1Index)
{
    PairArgBinaryCompute<PairBinaryOp::ARGMAX>(dstValue, dstIndex, src0Value, src0Index, src1Value, src1Index);
}

#define OP_TILE_OP_PAIRARGMIN TPairArgMin
template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
TILEOP void TPairArgMin(T0 dstValue, T1 dstIndex, T2 src0Value, T3 src0Index, T4 src1Value, T5 src1Index)
{
    PairArgBinaryCompute<PairBinaryOp::ARGMIN>(dstValue, dstIndex, src0Value, src0Index, src1Value, src1Index);
}

template <bool Max, bool Finalize, int H, int W, typename V, typename I, typename A, typename AI, typename B,
          typename BI, typename T>
TILEOP void TNanPairArg(V value, I index, A a, AI ai, B b, BI bi, T tmp)
{
    const auto layout = value.GetLayout();
    PtoTile<V> dstValue(value);
    PtoTile<I> dstIndex(index);
    PtoTile<A> leftValue(a);
    PtoTile<AI> leftIndex(ai);
    PtoTile<B> rightValue(b);
    PtoTile<BI> rightIndex(bi);
    for (LoopVar n1 = 0; n1 < layout.template GetShapeDim<1, 5>(); ++n1) {
        for (LoopVar n2 = 0; n2 < layout.template GetShapeDim<2, 5>(); ++n2) {
            const auto offset = TileOffset(0, n1, n2);
            dstValue.Assign(value, offset);
            dstIndex.Assign(index, offset);
            leftValue.Assign(a, offset);
            leftIndex.Assign(ai, offset);
            rightValue.Assign(b, offset);
            rightIndex.Assign(bi, offset);
            NanPairArgTile<Max, Finalize, H, W>(dstValue.Data(), dstIndex.Data(), leftValue.Data(), leftIndex.Data(),
                                                rightValue.Data(), rightIndex.Data(), tmp);
        }
    }
}
#endif
