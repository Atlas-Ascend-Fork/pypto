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
 * \file reduce.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_REDUCE__H
#define TILEOP_TILE_OPERATOR_REDUCE__H
#include "utils/sync.h"
#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

namespace reduce_detail {
constexpr size_t MERGED_TILE_AXIS_OFFSET = 3;
constexpr size_t TILE_HEIGHT_OFFSET = 2;
} // namespace reduce_detail

template <ReduceOp op, typename LastUse, typename T0, typename T1, typename T2>
TILEOP void ReduceComputeImpl(T0 dst, T1 src, T2 tmp)
{
    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;
    constexpr auto n3 = Std::tuple_element<DIM_3RD, LastUse>::type::value;
    if constexpr (op == ReduceOp::SUM) {
        PTO_WITH_LAST_USE(pto::TROWSUM(dst, src, tmp), n1, n2, n3);
    } else if constexpr (op == ReduceOp::MAX) {
        PTO_WITH_LAST_USE(pto::TROWMAX(dst, src, tmp), n1, n2, n3);
    } else if constexpr (op == ReduceOp::MIN) {
        PTO_WITH_LAST_USE(pto::TROWMIN(dst, src, tmp), n1, n2, n3);
    } else if constexpr (op == ReduceOp::PROD) {
        PTO_WITH_LAST_USE(pto::TROWPROD(dst, src, tmp), n1, n2, n3);
    } else if constexpr (op == ReduceOp::ARGMAX) {
        pto::TROWARGMAX(dst, src, tmp);
    } else if constexpr (op == ReduceOp::ARGMIN) {
        pto::TROWARGMIN(dst, src, tmp);
    }
}

template <ReduceOp op, typename LastUse, typename T0, typename T1, typename T2>
TILEOP void ReduceLastAxisCompute(T0 dst, T1 src, T2 tmp)
{
    constexpr auto tmpTileH = TileOp::GetTensorTileShapeDim<T2, DIM_4TH, MAX_DIMS>();
    constexpr auto tmpTileW = TileOp::GetTensorTileShapeDim<T2, DIM_5TH, MAX_DIMS>();
    using TmpTileDefine = pto::Tile<pto::TileType::Vec, typename T2::Type, tmpTileH, tmpTileW, pto::BLayout::RowMajor,
                                    tmpTileH, tmpTileW>;
    TmpTileDefine tmpTile;

    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto dstStride0 = dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto dstStride1 = dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto dstStride2 = dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();

    const auto srcLayout = src.GetLayout();
    auto srcShape0 = srcLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto srcShape1 = srcLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto srcShape2 = srcLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto srcShape3 = srcLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto srcShape4 = srcLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    if (srcShape0 == 0 || srcShape1 == 0 || srcShape2 == 0 || srcShape3 == 0 || srcShape4 == 0) {
        return;
    }
    auto srcStride0 = srcLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto srcStride1 = srcLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto srcStride2 = srcLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, DIM_4TH, MAX_DIMS>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, DIM_5TH, MAX_DIMS>();
    constexpr auto srcTypeSize = sizeof(typename T1::Type);
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                using DstTileDefine = typename std::conditional<
                    (dstTileW == 1),
                    pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::ColMajor, -1,
                              -1>,
                    pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1,
                              -1> >::type;
                using SrcTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileH, srcTileW,
                                                pto::BLayout::RowMajor, -1, -1>;
                DstTileDefine dstTile(dstShape3, dstShape4);
                SrcTileDefine srcTile(srcShape3, srcShape4);
                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                pto::TASSIGN(tmpTile, (uint64_t)(tmp.GetAddr()));
                ReduceComputeImpl<op, LastUse>(dstTile, srcTile, tmpTile);
            }
        }
    }
}

#define OP_TILE_OP_ROWSUMSINGLE TRowSumSingle
template <typename LastUse = LastUse3Dim<0, 0, 0>, typename T0, typename T1, typename T2>
TILEOP void TRowSumSingle(T0 dst, T1 src, T2 tmp)
{
    ReduceLastAxisCompute<ReduceOp::SUM, LastUse>(dst, src, tmp);
}

template <ReduceOp op, typename T0, typename T1, typename T2, typename T3>
TILEOP void ArgReduceLastAxisCompute(T0 dstValue, T1 dstIndex, T2 src, T3 tmp)
{
    constexpr auto tmpTileH = TileOp::GetTensorTileShapeDim<T3, DIM_4TH, MAX_DIMS>();
    constexpr auto tmpTileW = TileOp::GetTensorTileShapeDim<T3, DIM_5TH, MAX_DIMS>();
    using TmpTileDefine = pto::Tile<pto::TileType::Vec, typename T2::Type, tmpTileH, tmpTileW, pto::BLayout::RowMajor,
                                    tmpTileH, tmpTileW>;
    TmpTileDefine tmpTile;

    const auto dstLayout = dstValue.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto dstStride0 = dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto dstStride1 = dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto dstStride2 = dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();

    const auto srcLayout = src.GetLayout();
    auto srcShape0 = srcLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto srcShape1 = srcLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto srcShape2 = srcLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto srcShape3 = srcLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto srcShape4 = srcLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    if (srcShape0 == 0 || srcShape1 == 0 || srcShape2 == 0 || srcShape3 == 0 || srcShape4 == 0) {
        return;
    }
    auto srcStride0 = srcLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto srcStride1 = srcLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto srcStride2 = srcLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T2, DIM_4TH, MAX_DIMS>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T2, DIM_5TH, MAX_DIMS>();
    constexpr auto srcTypeSize = sizeof(typename T2::Type);
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                using DstValueTileDefine = typename std::conditional<
                    (dstTileW == 1),
                    pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::ColMajor, -1,
                              -1>,
                    pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1,
                              -1> >::type;
                using DstIndexTileDefine = typename std::conditional<
                    (dstTileW == 1),
                    pto::Tile<pto::TileType::Vec, typename T1::Type, dstTileH, dstTileW, pto::BLayout::ColMajor, -1,
                              -1>,
                    pto::Tile<pto::TileType::Vec, typename T1::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1,
                              -1> >::type;
                using SrcTileDefine = pto::Tile<pto::TileType::Vec, typename T2::Type, srcTileH, srcTileW,
                                                pto::BLayout::RowMajor, -1, -1>;
                DstValueTileDefine dstValueTile(dstShape3, dstShape4);
                DstIndexTileDefine dstIndexTile(dstShape3, dstShape4);
                SrcTileDefine srcTile(srcShape3, srcShape4);
                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                pto::TASSIGN(dstValueTile, (uint64_t)(dstValue.GetAddr() + dstOffset * dstTypeSize));
                pto::TASSIGN(dstIndexTile, (uint64_t)(dstIndex.GetAddr() + dstOffset * dstTypeSize));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                pto::TASSIGN(tmpTile, (uint64_t)(tmp.GetAddr()));
                if constexpr (op == ReduceOp::ROWARGMAXWITHVALUE) {
                    pto::TROWARGMAX(dstValueTile, dstIndexTile, srcTile, tmpTile);
                } else if constexpr (op == ReduceOp::ROWARGMINWITHVALUE) {
                    pto::TROWARGMIN(dstValueTile, dstIndexTile, srcTile, tmpTile);
                }
            }
        }
    }
}

#define OP_TILE_OP_ROWARGMAXWITHVALUESINGLE TRowArgMaxWithValueSingle
template <typename LastUse = LastUse4Dim<0, 0, 0, 0>, typename T0, typename T1, typename T2, typename T3>
TILEOP void TRowArgMaxWithValueSingle(T0 dstValue, T1 dstIndex, T2 src, T3 tmp)
{
    ArgReduceLastAxisCompute<ReduceOp::ROWARGMAXWITHVALUE>(dstValue, dstIndex, src, tmp);
}

#define OP_TILE_OP_ROWARGMINWITHVALUESINGLE TRowArgMinWithValueSingle
template <typename LastUse = LastUse4Dim<0, 0, 0, 0>, typename T0, typename T1, typename T2, typename T3>
TILEOP void TRowArgMinWithValueSingle(T0 dstValue, T1 dstIndex, T2 src, T3 tmp)
{
    ArgReduceLastAxisCompute<ReduceOp::ROWARGMINWITHVALUE>(dstValue, dstIndex, src, tmp);
}

#define OP_TILE_OP_ROWMAXSINGLE TRowMaxSingle
template <typename LastUse = LastUse3Dim<0, 0, 0>, typename T0, typename T1, typename T2>
TILEOP void TRowMaxSingle(T0 dst, T1 src, T2 tmp)
{
    ReduceLastAxisCompute<ReduceOp::MAX, LastUse>(dst, src, tmp);
}

#define OP_TILE_OP_ROWMINSINGLE TRowMinSingle
template <typename LastUse = LastUse3Dim<0, 0, 0>, typename T0, typename T1, typename T2>
TILEOP void TRowMinSingle(T0 dst, T1 src, T2 tmp)
{
    ReduceLastAxisCompute<ReduceOp::MIN, LastUse>(dst, src, tmp);
}

#define OP_TILE_OP_ROWPRODSINGLE TRowProdSingle
template <typename LastUse = LastUse3Dim<0, 0, 0>, typename T0, typename T1, typename T2>
TILEOP void TRowProdSingle(T0 dst, T1 src, T2 tmp)
{
    ReduceLastAxisCompute<ReduceOp::PROD, LastUse>(dst, src, tmp);
}

template <ReduceOp op, int axis, typename T0, typename T1>
TILEOP void TRowMaxMinProdLineDynamic(T0 dst, T1 src)
{
    constexpr auto srcShapeSize = Std::tuple_size<typename T1::Shape>::value;
    constexpr auto dstShapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, axis + dstShapeSize - MAX_DIMS>();
    constexpr auto dstTileW = TileOp::GetAnyAxisMergeResult<
        axis + dstShapeSize - reduce_detail::MERGED_TILE_AXIS_OFFSET, dstShapeSize, typename T0::TileShape>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, axis + srcShapeSize - MAX_DIMS>();
    constexpr auto srcTileW = TileOp::GetAnyAxisMergeResult<
        axis + srcShapeSize - reduce_detail::MERGED_TILE_AXIS_OFFSET, srcShapeSize, typename T1::TileShape>();
    using DstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor,
                                    -1, -1>;
    using SrcTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileH, srcTileW, pto::BLayout::RowMajor,
                                    -1, -1>;
    constexpr auto typeSize = sizeof(typename T1::Type);
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    size_t dstShape[] = {static_cast<size_t>(dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>()),
                         static_cast<size_t>(dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>()),
                         static_cast<size_t>(dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>()),
                         static_cast<size_t>(dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>()),
                         static_cast<size_t>(dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>())};
    size_t dstStride[] = {static_cast<size_t>(dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>()),
                          static_cast<size_t>(dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>()),
                          static_cast<size_t>(dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>()),
                          static_cast<size_t>(dstLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>())};
    size_t srcShape[] = {static_cast<size_t>(srcLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>()),
                         static_cast<size_t>(srcLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>()),
                         static_cast<size_t>(srcLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>()),
                         static_cast<size_t>(srcLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>()),
                         static_cast<size_t>(srcLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>())};
    size_t srcStride[] = {static_cast<size_t>(srcLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>()),
                          static_cast<size_t>(srcLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>()),
                          static_cast<size_t>(srcLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>()),
                          static_cast<size_t>(srcLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>())};
    for (LoopVar n0Index = 0, n0Size = (axis == 0 ? (size_t)1 : dstShape[0]); n0Index < n0Size; ++n0Index) {
        for (LoopVar n1Index = 0, n1Size = (axis == 1 ? (size_t)1 : dstShape[1]); n1Index < n1Size; ++n1Index) {
            for (LoopVar n2Index = 0, n2Size = (axis == DIM_3RD ? (size_t)1 : dstShape[DIM_3RD]); n2Index < n2Size;
                 ++n2Index) {
                for (LoopVar n3Index = 0, n3Size = (axis == DIM_4TH ? (size_t)1 : dstShape[DIM_4TH]); n3Index < n3Size;
                     ++n3Index) {
                    DstTileDefine dstTile(dstShape[axis], dstShape[DIM_5TH]);
                    SrcTileDefine srcTile(srcShape[axis], srcShape[DIM_5TH]);
                    auto dstOffset = n0Index * dstStride[DIM_1ST] + n1Index * dstStride[DIM_2ND] +
                                     n2Index * dstStride[DIM_3RD] + n3Index * dstStride[DIM_4TH];
                    auto srcOffset = n0Index * srcStride[DIM_1ST] + n1Index * srcStride[DIM_2ND] +
                                     n2Index * srcStride[DIM_3RD] + n3Index * srcStride[DIM_4TH];
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * typeSize));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * typeSize));
                    if constexpr (op == ReduceOp::MAX) {
                        pto::TCOLMAX(dstTile, srcTile);
                    } else if constexpr (op == ReduceOp::MIN) {
                        pto::TCOLMIN(dstTile, srcTile);
                    } else if constexpr (op == ReduceOp::PROD) {
                        pto::TCOLPROD(dstTile, srcTile);
                    }
                }
            }
        }
    }
}

#define OP_TILE_OP_ROWMAXLINE TRowMaxLine
template <int axis, typename T0, typename T1>
TILEOP void TRowMaxLine(T0 dst, T1 src)
{
    TRowMaxMinProdLineDynamic<ReduceOp::MAX, axis>(dst, src);
}

#define OP_TILE_OP_ROWMINLINE TRowMinLine
template <int axis, typename T0, typename T1>
TILEOP void TRowMinLine(T0 dst, T1 src)
{
    TRowMaxMinProdLineDynamic<ReduceOp::MIN, axis>(dst, src);
}

#define OP_TILE_OP_ROWPRODLINE TRowProdLine
template <int axis, typename T0, typename T1>
TILEOP void TRowProdLine(T0 dst, T1 src)
{
    TRowMaxMinProdLineDynamic<ReduceOp::PROD, axis>(dst, src);
}

template <int axis, ReduceOp op, typename DstTileDefine, typename SrcTileDefine, typename TmpTileDefine, typename T0,
          typename T1, typename T2>
TILEOP void ColReduceWithTmpImp(T0 dst, T1 src, T2 tmp)
{
    constexpr auto srcTypeSize = sizeof(typename T1::Type);
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    size_t dstShape[] = {static_cast<size_t>(dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>()),
                         static_cast<size_t>(dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>()),
                         static_cast<size_t>(dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>()),
                         static_cast<size_t>(dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>()),
                         static_cast<size_t>(dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>())};
    size_t dstStride[] = {static_cast<size_t>(dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>()),
                          static_cast<size_t>(dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>()),
                          static_cast<size_t>(dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>()),
                          static_cast<size_t>(dstLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>())};
    size_t srcShape[] = {static_cast<size_t>(srcLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>()),
                         static_cast<size_t>(srcLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>()),
                         static_cast<size_t>(srcLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>()),
                         static_cast<size_t>(srcLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>()),
                         static_cast<size_t>(srcLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>())};
    size_t srcStride[] = {static_cast<size_t>(srcLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>()),
                          static_cast<size_t>(srcLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>()),
                          static_cast<size_t>(srcLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>()),
                          static_cast<size_t>(srcLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>())};
    for (LoopVar n0Index = 0, n0Size = (axis == 0 ? (size_t)1 : dstShape[0]); n0Index < n0Size; ++n0Index) {
        for (LoopVar n1Index = 0, n1Size = (axis == 1 ? (size_t)1 : dstShape[1]); n1Index < n1Size; ++n1Index) {
            for (LoopVar n2Index = 0, n2Size = (axis == DIM_3RD ? (size_t)1 : dstShape[DIM_3RD]); n2Index < n2Size;
                 ++n2Index) {
                for (LoopVar n3Index = 0, n3Size = (axis == DIM_4TH ? (size_t)1 : dstShape[DIM_4TH]); n3Index < n3Size;
                     ++n3Index) {
                    DstTileDefine dstTile(dstShape[axis], dstShape[DIM_5TH]);
                    SrcTileDefine srcTile(srcShape[axis], srcShape[DIM_5TH]);
                    TmpTileDefine tmpTile;
                    auto dstOffset = n0Index * dstStride[DIM_1ST] + n1Index * dstStride[DIM_2ND] +
                                     n2Index * dstStride[DIM_3RD] + n3Index * dstStride[DIM_4TH];
                    auto srcOffset = n0Index * srcStride[DIM_1ST] + n1Index * srcStride[DIM_2ND] +
                                     n2Index * srcStride[DIM_3RD] + n3Index * srcStride[DIM_4TH];
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                    pto::TASSIGN(tmpTile, (uint64_t)(tmp.GetAddr()));
                    if constexpr (op == ReduceOp::SUM) {
                        pto::TCOLSUM(dstTile, srcTile, tmpTile, true);
                    } else if constexpr (op == ReduceOp::ARGMAX) {
                        pto::TCOLARGMAX(dstTile, srcTile, tmpTile);
                    } else if constexpr (op == ReduceOp::ARGMIN) {
                        pto::TCOLARGMIN(dstTile, srcTile, tmpTile);
                    }
                }
            }
        }
    }
}

template <int axis, ReduceOp op, typename T0, typename T1, typename T2>
TILEOP void ColReduceWithTmp(T0 dst, T1 src, T2 tmp)
{
    constexpr auto srcShapeSize = Std::tuple_size<typename T1::Shape>::value;
    constexpr auto dstShapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr auto tmpShapeSize = Std::tuple_size<typename T2::Shape>::value;
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, axis + dstShapeSize - MAX_DIMS>();
    constexpr auto dstTileW = TileOp::GetAnyAxisMergeResult<
        axis + dstShapeSize - reduce_detail::MERGED_TILE_AXIS_OFFSET, dstShapeSize, typename T0::TileShape>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, axis + srcShapeSize - MAX_DIMS>();
    constexpr auto srcTileW = TileOp::GetAnyAxisMergeResult<
        axis + srcShapeSize - reduce_detail::MERGED_TILE_AXIS_OFFSET, srcShapeSize, typename T1::TileShape>();
    using DstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor,
                                    -1, -1>;
    using SrcTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileH, srcTileW, pto::BLayout::RowMajor,
                                    -1, -1>;
    constexpr auto tmpTileH = TileOp::GetTensorTileShapeDim<T2, tmpShapeSize - reduce_detail::TILE_HEIGHT_OFFSET>();
    constexpr auto tmpTileW = TileOp::GetTensorTileShapeDim<T2, tmpShapeSize - 1>();
    using TmpTileDefine = pto::Tile<pto::TileType::Vec, typename T2::Type, tmpTileH, tmpTileW, pto::BLayout::RowMajor,
                                    tmpTileH, tmpTileW>;
    ColReduceWithTmpImp<axis, op, DstTileDefine, SrcTileDefine, TmpTileDefine>(dst, src, tmp);
}

#define OP_TILE_OP_ROWSUMLINE TRowSumLine
template <int axis, typename T0, typename T1, typename T2>
TILEOP void TRowSumLine(T0 dst, T1 src, T2 tmp)
{
    ColReduceWithTmp<axis, ReduceOp::SUM, T0, T1, T2>(dst, src, tmp);
}

#define OP_TILE_OP_ROWARGMAXLINE TRowArgMaxLine
template <int axis, typename T0, typename T1, typename T2>
TILEOP void TRowArgMaxLine(T0 dst, T1 src, T2 tmp)
{
    ColReduceWithTmp<axis, ReduceOp::ARGMAX, T0, T1, T2>(dst, src, tmp);
}

template <int axis, ReduceOp op, typename DstValueTileDefine, typename DstIndexTileDefine, typename SrcTileDefine,
          typename TmpTileDefine, typename T0, typename T1, typename T2, typename T3>
TILEOP void ColArgReduceImp(T0 dstValue, T1 dstIndex, T2 src, T3 tmp)
{
    constexpr auto srcTypeSize = sizeof(typename T2::Type);
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    const auto dstLayout = dstValue.GetLayout();
    const auto srcLayout = src.GetLayout();
    size_t dstShape[] = {static_cast<size_t>(dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>()),
                         static_cast<size_t>(dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>()),
                         static_cast<size_t>(dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>()),
                         static_cast<size_t>(dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>()),
                         static_cast<size_t>(dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>())};
    size_t dstStride[] = {static_cast<size_t>(dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>()),
                          static_cast<size_t>(dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>()),
                          static_cast<size_t>(dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>()),
                          static_cast<size_t>(dstLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>())};
    size_t srcShape[] = {static_cast<size_t>(srcLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>()),
                         static_cast<size_t>(srcLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>()),
                         static_cast<size_t>(srcLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>()),
                         static_cast<size_t>(srcLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>()),
                         static_cast<size_t>(srcLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>())};
    size_t srcStride[] = {static_cast<size_t>(srcLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>()),
                          static_cast<size_t>(srcLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>()),
                          static_cast<size_t>(srcLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>()),
                          static_cast<size_t>(srcLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>())};
    for (LoopVar n0Index = 0, n0Size = (axis == 0 ? (size_t)1 : dstShape[0]); n0Index < n0Size; ++n0Index) {
        for (LoopVar n1Index = 0, n1Size = (axis == 1 ? (size_t)1 : dstShape[1]); n1Index < n1Size; ++n1Index) {
            for (LoopVar n2Index = 0, n2Size = (axis == DIM_3RD ? (size_t)1 : dstShape[DIM_3RD]); n2Index < n2Size;
                 ++n2Index) {
                for (LoopVar n3Index = 0, n3Size = (axis == DIM_4TH ? (size_t)1 : dstShape[DIM_4TH]); n3Index < n3Size;
                     ++n3Index) {
                    DstValueTileDefine dstValueTile(dstShape[axis], dstShape[DIM_5TH]);
                    DstIndexTileDefine dstIndexTile(dstShape[axis], dstShape[DIM_5TH]);
                    SrcTileDefine srcTile(srcShape[axis], srcShape[DIM_5TH]);
                    TmpTileDefine tmpTile;
                    auto dstOffset = n0Index * dstStride[DIM_1ST] + n1Index * dstStride[DIM_2ND] +
                                     n2Index * dstStride[DIM_3RD] + n3Index * dstStride[DIM_4TH];
                    auto srcOffset = n0Index * srcStride[DIM_1ST] + n1Index * srcStride[DIM_2ND] +
                                     n2Index * srcStride[DIM_3RD] + n3Index * srcStride[DIM_4TH];
                    pto::TASSIGN(dstValueTile, (uint64_t)(dstValue.GetAddr() + dstOffset * dstTypeSize));
                    pto::TASSIGN(dstIndexTile, (uint64_t)(dstIndex.GetAddr() + dstOffset * dstTypeSize));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                    pto::TASSIGN(tmpTile, (uint64_t)(tmp.GetAddr()));
                    if constexpr (op == ReduceOp::COLARGMAXWITHVALUE) {
                        pto::TCOLARGMAX(dstValueTile, dstIndexTile, srcTile, tmpTile);
                    } else if constexpr (op == ReduceOp::COLARGMINWITHVALUE) {
                        pto::TCOLARGMIN(dstValueTile, dstIndexTile, srcTile, tmpTile);
                    }
                }
            }
        }
    }
}

template <int axis, ReduceOp op, typename T0, typename T1, typename T2, typename T3>
TILEOP void ColArgReduce(T0 dstValue, T1 dstIndex, T2 src, T3 tmp)
{
    constexpr auto srcShapeSize = Std::tuple_size<typename T2::Shape>::value;
    constexpr auto dstShapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr auto tmpShapeSize = Std::tuple_size<typename T3::Shape>::value;
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, axis + dstShapeSize - MAX_DIMS>();
    constexpr auto dstTileW = TileOp::GetAnyAxisMergeResult<
        axis + dstShapeSize - reduce_detail::MERGED_TILE_AXIS_OFFSET, dstShapeSize, typename T0::TileShape>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T2, axis + srcShapeSize - MAX_DIMS>();
    constexpr auto srcTileW = TileOp::GetAnyAxisMergeResult<
        axis + srcShapeSize - reduce_detail::MERGED_TILE_AXIS_OFFSET, srcShapeSize, typename T2::TileShape>();
    using DstValueTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW,
                                         pto::BLayout::RowMajor, -1, -1>;
    using DstIndexTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, dstTileH, dstTileW,
                                         pto::BLayout::RowMajor, -1, -1>;
    using SrcTileDefine = pto::Tile<pto::TileType::Vec, typename T2::Type, srcTileH, srcTileW, pto::BLayout::RowMajor,
                                    -1, -1>;
    constexpr auto tmpTileH = TileOp::GetTensorTileShapeDim<T3, tmpShapeSize - reduce_detail::TILE_HEIGHT_OFFSET>();
    constexpr auto tmpTileW = TileOp::GetTensorTileShapeDim<T3, tmpShapeSize - 1>();
    using TmpTileDefine = pto::Tile<pto::TileType::Vec, typename T3::Type, tmpTileH, tmpTileW, pto::BLayout::RowMajor,
                                    tmpTileH, tmpTileW>;
    ColArgReduceImp<axis, op, DstValueTileDefine, DstIndexTileDefine, SrcTileDefine, TmpTileDefine>(dstValue, dstIndex,
                                                                                                    src, tmp);
}

#define OP_TILE_OP_ROWARGMAXWITHVALUELINE TRowArgMaxWithValueLine
template <int axis, typename T0, typename T1, typename T2, typename T3>
TILEOP void TRowArgMaxWithValueLine(T0 dstValue, T1 dstIndex, T2 src, T3 tmp)
{
    ColArgReduce<axis, ReduceOp::COLARGMAXWITHVALUE, T0, T1, T2, T3>(dstValue, dstIndex, src, tmp);
}

#define OP_TILE_OP_ROWARGMINWITHVALUELINE TRowArgMinWithValueLine
template <int axis, typename T0, typename T1, typename T2, typename T3>
TILEOP void TRowArgMinWithValueLine(T0 dstValue, T1 dstIndex, T2 src, T3 tmp)
{
    ColArgReduce<axis, ReduceOp::COLARGMINWITHVALUE, T0, T1, T2, T3>(dstValue, dstIndex, src, tmp);
}

namespace NanTile {
__aicore__ constexpr size_t Align(size_t n, size_t a = 32) { return (n + a - 1) / a * a; }

__aicore__ constexpr size_t RowScratchWidth(size_t rowBytes)
{
    if (rowBytes <= 256)
        return 32;
    if (rowBytes <= 512)
        return 256;
    return rowBytes / 512 * 256;
}

} // namespace NanTile

template <bool Max, bool Column, typename D, typename X, typename Q>
TILEOP void NanValueReduce(D& dst, X& src, Q& tmp)
{
    if constexpr (Column) {
        if constexpr (Max)
            pto::TCOLMAX(dst, src);
        else
            pto::TCOLMIN(dst, src);
    } else {
        if constexpr (Max)
            pto::TROWMAX(dst, src, tmp);
        else
            pto::TROWMIN(dst, src, tmp);
    }
    SyncV();
}

template <bool Column, typename D, typename X>
TILEOP void NanExpand(D& dst, X& src)
{
    if constexpr (Column)
        pto::TCOLEXPAND(dst, src);
    else
        pto::TROWEXPAND(dst, src);
    SyncV();
}

template <bool Column, typename V, typename I, typename X, typename Q>
TILEOP void NanCandidateReduce(V& value, I& index, X& scores, Q& tmp)
{
    if constexpr (Column)
        pto::TCOLARGMAX(value, index, scores, tmp);
    else
        pto::TROWARGMAX(value, index, scores, tmp);
    SyncV();
}

// All results used by CMP/SEL have an explicit row-major layout. The input is read-only.
template <bool Max, bool Column, bool Arg, int WorkWidth, int32_t IndexOffset, bool Finalize, typename V, typename I,
          typename X, typename T>
TILEOP void NanReduceTile(V& dst, I& dstIndex, X& src, T& tmp)
{
    // BF16 and non-FP32 arg inputs are converted by the operation layer.
    using E = typename X::DType;
    constexpr int H = X::Rows;
    constexpr int W = NanTile::Align(WorkWidth * sizeof(E)) / sizeof(E);
    constexpr int RH = Column ? 1 : H;
    constexpr int RW = Column ? W : 32 / sizeof(E);
    constexpr size_t rowBytes = W * sizeof(E);
    constexpr size_t dataBytes = H * rowBytes;
    constexpr size_t maskRowBytes = NanTile::Align((W + 7) / 8);
    constexpr size_t resultRowBytes = RW * sizeof(E);
    constexpr size_t resultBytes = RH * resultRowBytes;
    // A2 row arg needs packed values and indices. The native row scratch width covers both.
    constexpr int QW = Column && Arg ? 192 : NanTile::RowScratchWidth(rowBytes) / sizeof(E);
    constexpr int QH = Column || (!Arg && rowBytes <= 256) ? 1 : H;
    constexpr size_t reduceBytes = Column && !Arg ? 0 : QH * QW * sizeof(E);

    constexpr size_t workOffset = Arg ? dataBytes : 0;
    // Comparisons can read a full repeat beyond the last physical work row.
    constexpr size_t compareGuard = NanTile::Align(rowBytes, 256) - rowBytes;
    constexpr size_t maskOffset = workOffset + dataBytes + compareGuard;
    constexpr size_t valueOffset = maskOffset + H * maskRowBytes;
    constexpr size_t flagOffset = valueOffset + (Column ? 0 : resultBytes);
    constexpr size_t indexOffset = flagOffset + resultBytes;
    constexpr size_t reduceOffset = indexOffset + (Arg && !Column ? resultBytes : 0);
    constexpr size_t selectorOffset = reduceOffset + reduceBytes;
    constexpr size_t selectorEnd = selectorOffset + 32;
    // A2 compares read a complete repeat even when the result row is shorter.
    constexpr size_t flagReadEnd = indexOffset + NanTile::Align(resultRowBytes, 256) - resultRowBytes;
    constexpr size_t requiredBytes = selectorEnd > flagReadEnd ? selectorEnd : flagReadEnd;
    static_assert(requiredBytes <= TileOp::GetTileSizeBytes<T>(), "NaN reduction scratch exceeds its planned capacity");

    using Data = pto::Tile<pto::TileType::Vec, E, H, W, pto::BLayout::RowMajor, -1, -1>;
    using Result = pto::Tile<pto::TileType::Vec, E, RH, RW, pto::BLayout::RowMajor, -1, -1>;
    using Index = pto::Tile<pto::TileType::Vec, int32_t, RH, RW, pto::BLayout::RowMajor, -1, -1>;
    using Mask = pto::Tile<pto::TileType::Vec, uint8_t, H, maskRowBytes, pto::BLayout::RowMajor, -1, -1>;
    using Selector = pto::Tile<pto::TileType::Vec, uint8_t, 1, 32>;
    using ReduceTmp = pto::Tile<pto::TileType::Vec, E, QH, QW>;
    unsigned h = src.GetValidRow(), w = src.GetValidCol();
    unsigned rh = Column ? 1 : h, rw = Column ? w : 1;
    Data work(h, w), converted(h, w);
    Result value(rh, rw), flag(rh, rw);
    Index index(rh, rw);
    Mask mask(h, (W + 7) / 8), resultMask(rh, (RW + 7) / 8);
    Selector selector;
    ReduceTmp q;
    const auto base = tmp.GetAddr();
    const auto workAddr = base + workOffset;
    const auto reduceAddr = base + reduceOffset;
    pto::TASSIGN(work, workAddr);
    pto::TASSIGN(converted, base);
    if constexpr (Column)
        pto::TASSIGN(value, reinterpret_cast<uint64_t>(dst.data()));
    else
        pto::TASSIGN(value, base + valueOffset);
    pto::TASSIGN(flag, base + flagOffset);
    if constexpr (Arg) {
        if constexpr (Column)
            pto::TASSIGN(index, reinterpret_cast<uint64_t>(dstIndex.data()));
        else
            pto::TASSIGN(index, base + indexOffset);
    }
    pto::TASSIGN(mask, base + maskOffset);
    pto::TASSIGN(resultMask, base + maskOffset);
    pto::TASSIGN(selector, base + selectorOffset);
    pto::TASSIGN(q, reduceAddr);
    union {
        uint32_t u;
        float f;
    } inf = {0x7f800000U}, nan = {0x7fc00000U};
    const E identity = static_cast<E>(Max ? -inf.f : inf.f);
    if constexpr (Arg) {
        // Retain compact candidates independently of shared source padding.
        if constexpr (X::Cols == W)
            pto::TMOV(converted, src);
        else
            pto::TEXTRACT(converted, src, 0, 0);
        SyncV();
    }
    // A2 TCMP reads full repeats. Compare a local copy when a numeric tile
    // could otherwise read past the end of the source allocation.
    if constexpr (!Arg && W * sizeof(E) % 256 != 0) {
        if constexpr (X::Cols == W) {
            pto::TMOV(work, src);
        } else {
            pto::TEXTRACT(work, src, 0, 0);
        }
        SyncV();
#ifdef __DAV_V220
        constexpr unsigned repeat = 256 / sizeof(E);
        constexpr unsigned tailStart = W / repeat * repeat;
        if constexpr (tailStart != W) {
            if (w > tailStart) {
                // TCMP reads the last repeat in full. Fill the invalid lanes in
                // scratch, including the part that crosses the physical row end.
                using Repeat = pto::Tile<pto::TileType::Vec, E, 1, repeat, pto::BLayout::RowMajor, 1, -1>;
                using Padded = pto::Tile<pto::TileType::Vec, E, 1, repeat, pto::BLayout::RowMajor, 1, repeat,
                                         pto::SLayout::NoneBox, 512, pto::PadValue::Zero>;
                const auto address = workAddr + ((h - 1) * W + tailStart) * sizeof(E);
                Repeat valid(w - tailStart);
                Padded padded;
                pto::TASSIGN(valid, address);
                pto::TASSIGN(padded, address);
                pto::TFILLPAD_INPLACE(padded, valid);
                SyncV();
            }
        }
#endif
        pto::TCMP(mask, work, work, pto::CmpMode::EQ);
        SyncV();
        pto::TSELS(work, mask, work, selector, identity);
        SyncV();
    } else if constexpr (Arg) {
        pto::TCMP(mask, converted, converted, pto::CmpMode::EQ);
        SyncV();
        pto::TSELS(work, mask, converted, selector, identity);
        SyncV();
    } else {
        pto::TCMP(mask, src, src, pto::CmpMode::EQ);
        SyncV();
        pto::TSELS(work, mask, src, selector, identity);
        SyncV();
    }
    NanValueReduce<Max, Column>(value, work, q);
    if constexpr (!Arg) {
        pto::TEXPANDS(work, static_cast<E>(1));
        SyncV();
        pto::TSELS(work, mask, work, selector, static_cast<E>(0));
        SyncV();
        NanValueReduce<true, Column>(flag, work, q);
        pto::TCMPS(resultMask, flag, static_cast<E>(0), pto::CmpMode::NE);
        SyncV();
        pto::TSELS(value, resultMask, value, selector, static_cast<E>(nan.f));
        SyncV();
    } else {
        NanExpand<Column>(work, value);
        pto::TCMP(mask, converted, work, pto::CmpMode::EQ);
        SyncV();
        pto::TEXPANDS(work, static_cast<E>(1));
        SyncV();
        pto::TSELS(work, mask, work, selector, static_cast<E>(0));
        SyncV();
        NanCandidateReduce<Column>(flag, index, work, q);
        if constexpr (IndexOffset != 0) {
            // Apply the block offset before installing the sentinel for invalid rows or columns.
            pto::TADDS(index, index, IndexOffset);
            SyncV();
        }
        pto::TCMPS(resultMask, flag, static_cast<E>(0), pto::CmpMode::NE);
        SyncV();
        pto::TSELS(index, resultMask, index, selector, Finalize ? 0 : INT32_MAX);
        SyncV();
        if constexpr (!Column) {
            pto::TROWMIN(dstIndex, index, q);
            SyncV();
        }
    }
    if constexpr (!Column) {
        // Row results must be packed into the output's column-major layout.
        pto::TROWMIN(dst, value, q);
        SyncV();
    }
}

// Same iteration/stride convention as the existing row and column reduction wrappers.
template <int Axis, bool Max, bool Arg, int H, int WorkWidth, int32_t IndexOffset = 0, bool Finalize = false,
          typename V, typename I, typename X, typename T>
TILEOP void TNanReduce(V dst, I dstIndex, X src, T tmp)
{
    constexpr bool Column = Axis != 4;
    constexpr auto Rank = Std::tuple_size<typename X::Shape>::value;
    // Source columns describe physical strides; work columns use the planned compact layout.
    constexpr int W = Column ? TileOp::GetAnyAxisMergeResult<Axis + Rank - 3, Rank, typename X::TileShape>() :
                               TileOp::GetTensorTileShapeDim<X, 4, MAX_DIMS>();
    using E = typename X::Type;
    constexpr int DH = Column ? 1 : TileOp::GetTensorTileShapeDim<V, 3, 5>();
    constexpr int DW = Column ? NanTile::Align(WorkWidth * sizeof(E)) / sizeof(E) :
                                TileOp::GetTensorTileShapeDim<V, 4, 5>();
    using Value = pto::Tile<pto::TileType::Vec, typename V::Type, DH, DW,
                            DW == 1 ? pto::BLayout::ColMajor : pto::BLayout::RowMajor, -1, -1>;
    constexpr int IH = Column ? 1 : TileOp::GetTensorTileShapeDim<I, 3, 5>();
    constexpr int IW = Column ? DW : TileOp::GetTensorTileShapeDim<I, 4, 5>();
    using Idx = pto::Tile<pto::TileType::Vec, int32_t, IH, IW,
                          IW == 1 ? pto::BLayout::ColMajor : pto::BLayout::RowMajor, -1, -1>;
    using Source = pto::Tile<pto::TileType::Vec, typename X::Type, H, W, pto::BLayout::RowMajor, -1, -1>;
    const auto layout = src.GetLayout();
    const size_t dim3 = layout.template GetShapeDim<3, 5>();
    const unsigned h = Column ? layout.template GetShapeDim<Axis, 5>() : dim3;
    const unsigned w = layout.template GetShapeDim<4, 5>();
    if (Column ? w == 0 : h == 0) {
        return;
    }
    const size_t limit1 = Axis == 1 ? 1 : layout.template GetShapeDim<1, 5>();
    const size_t limit2 = Axis == 2 ? 1 : layout.template GetShapeDim<2, 5>();
    const size_t limit3 = !Column || Axis == 3 ? 1 : dim3;
    // The five-dimensional layout left-pads a rank-four tile with one singleton axis.
    for (LoopVar n1 = 0; n1 < limit1; ++n1) {
        for (LoopVar n2 = 0; n2 < limit2; ++n2) {
            for (LoopVar n3 = 0; n3 < limit3; ++n3) {
                const auto offset = TileOffset4Dim(0, n1, n2, n3);
                const auto srcOffset = GenTileOffset(src, offset);
                const auto dstOffset = GenTileOffset(dst, offset);
                Source x(h, w);
                Value v(Column ? 1 : h, Column ? w : 1);
                Idx i(Column ? 1 : h, Column ? w : 1);
                pto::TASSIGN(x, src.GetAddr() + srcOffset * sizeof(typename X::Type));
                pto::TASSIGN(v, dst.GetAddr() + dstOffset * sizeof(typename V::Type));
                pto::TASSIGN(i, dstIndex.GetAddr() + GenTileOffset(dstIndex, offset) * sizeof(int32_t));
                if (Column ? h == 0 : w == 0) {
                    // An empty physical R tile must initialize its partial result.
                    union {
                        uint32_t u;
                        float f;
                    } infinity = {0x7f800000U}, nan = {0x7fc00000U};
                    auto identity = static_cast<typename V::Type>(Max ? -infinity.f : infinity.f);
                    pto::TEXPANDS(v, !Arg ? static_cast<typename V::Type>(nan.f) : identity);
                    if constexpr (Arg) {
                        pto::TEXPANDS(i, Finalize ? 0 : INT32_MAX);
                    }
                    SyncV();
                    continue;
                }
                NanReduceTile<Max, Column, Arg, WorkWidth, IndexOffset, Finalize>(v, i, x, tmp);
            }
        }
    }
}
#endif
