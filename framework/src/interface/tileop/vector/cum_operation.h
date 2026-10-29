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
 * \file cum_operation.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_CUM_OPERATION__H
#define TILEOP_TILE_OPERATOR_CUM_OPERATION__H
#include "utils/layout.h"
#include "utils/sync.h"
#include "utils/tile_tensor.h"
#include <array>

template <typename T0, typename T1, unsigned tileH, unsigned tileW, unsigned dstTypeSize, int is_sum>
TILEOP void CumOperationTool(T0 dst, T1 src, uint64_t tmpStride)
{
    using tmpTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, tileH, tileW, pto::BLayout::RowMajor, -1,
                                    -1>;
    tmpTileDefine tmpDstTile(tileH, tileW), tmpSrcTile(tileH, tileW);
    pto::TASSIGN(tmpDstTile, (uint64_t)(dst.GetAddr() + tmpStride));
    pto::TASSIGN(tmpSrcTile, (uint64_t)(src.GetAddr() + tmpStride));
    pto::TMOV(tmpDstTile, tmpSrcTile);

#pragma clang loop unroll(disable)
    for (LoopVar i = 1; i < tileH; i *= 2) {
#ifdef __DAV_V220
        pipe_barrier(PIPE_V);
#else
        pipe_barrier(PIPE_ALL);
#endif
        using TileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, tileH, tileW, pto::BLayout::RowMajor, -1,
                                     -1>;
        TileDefine src1Tile(tileH - i, tileW), src0Tile(tileH - i, tileW), dstTile(tileH - i, tileW);
        pto::TASSIGN(src0Tile, (uint64_t)(src.GetAddr() + tmpStride));
        pto::TASSIGN(src1Tile, (uint64_t)(src.GetAddr() + tmpStride + i * tileW * dstTypeSize));
        pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + tmpStride + i * tileW * dstTypeSize));
        if (is_sum) {
            pto::TADD(dstTile, src0Tile, src1Tile);
        } else {
            pto::TMUL(dstTile, src0Tile, src1Tile);
        }
#ifdef __DAV_V220
        pipe_barrier(PIPE_V);
#else
        pipe_barrier(PIPE_ALL);
#endif
        pto::TADDS(src1Tile, dstTile, 0);
    }
}

template <typename T0, int is_sum>
TILEOP void CumOperationScalarTool(__ubuf__ typename T0::Type* dstAddr, __ubuf__ typename T0::Type* srcAddr,
                                   uint64_t stride, uint64_t lastShape, uint64_t idx)
{
    if (is_sum) {
        for (LoopVar tmpIdx = 0; tmpIdx < lastShape - idx; tmpIdx++) {
            dstAddr[stride + tmpIdx + idx] = srcAddr[stride + tmpIdx] + srcAddr[stride + tmpIdx + idx];
        }
    } else {
        for (LoopVar tmpIdx = 0; tmpIdx < lastShape - idx; tmpIdx++) {
            dstAddr[stride + tmpIdx + idx] = srcAddr[stride + tmpIdx] * srcAddr[stride + tmpIdx + idx];
        }
    }
    for (LoopVar tmpIdx = 0; tmpIdx < lastShape - idx; tmpIdx++) {
        srcAddr[stride + tmpIdx + idx] = dstAddr[stride + tmpIdx + idx];
    }
}

template <int axis, int is_sum, typename T0, typename T1>
TILEOP void TCumOperationFullTile(T0 dst, T1 src)
{
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    const auto dstLayout = dst.GetLayout();
    auto n0DstStride = dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto n1DstStride = dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto n2DstStride = dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto n3DstStride = dstLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();
    auto n0DstShape = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto n1DstShape = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto n2DstShape = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto n3DstShape = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto n4DstShape = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    constexpr auto dst1RawShape = Std::tuple_element<shapeSize - 1, typename T0::TileShape>::type::value;
    constexpr size_t SECOND_LAST_DIM_OFFSET = 2;
    constexpr size_t THIRD_LAST_DIM_OFFSET = 3;
    constexpr size_t FOURTH_LAST_DIM_OFFSET = 4;

    if constexpr (axis == 0) {
        constexpr auto tileH = Std::tuple_element<DIM_1ST, typename T0::TileShape>::type::value;
        constexpr auto tileW = TileOp::GetNonFirstAxisMergeResult<shapeSize, typename T0::TileShape>();
        uint64_t tmpStride = 0;
        CumOperationTool<T0, T1, tileH, tileW, dstTypeSize, is_sum>(dst, src, tmpStride);
        return;
    } else if constexpr (axis == 1) {
        int loops = n0DstShape;
        constexpr auto
            tileH = Std::tuple_element<shapeSize - FOURTH_LAST_DIM_OFFSET, typename T0::TileShape>::type::value;
        constexpr auto
            dst2RawShape = Std::tuple_element<shapeSize - SECOND_LAST_DIM_OFFSET, typename T0::TileShape>::type::value;
        constexpr auto
            dst3RawShape = Std::tuple_element<shapeSize - THIRD_LAST_DIM_OFFSET, typename T0::TileShape>::type::value;
        constexpr int tileW = dst3RawShape * dst2RawShape * dst1RawShape;

        for (LoopVar loop = 0; loop < loops; loop++) {
            uint64_t tmpStride = loop * n0DstStride * dstTypeSize;
            CumOperationTool<T0, T1, tileH, tileW, dstTypeSize, is_sum>(dst, src, tmpStride);
        }
        return;
    } else if constexpr (axis == DIM_3RD) {
        constexpr auto
            dst2RawShape = Std::tuple_element<shapeSize - SECOND_LAST_DIM_OFFSET, typename T0::TileShape>::type::value;
        constexpr auto
            dst3RawShape = Std::tuple_element<shapeSize - THIRD_LAST_DIM_OFFSET, typename T0::TileShape>::type::value;
        constexpr int tileH = dst3RawShape;
        constexpr int tileW = dst2RawShape * dst1RawShape;

        for (LoopVar j = 0; j < n0DstShape; j++) {
            for (LoopVar k = 0; k < n1DstShape; k++) {
                uint64_t tmpStride = (j * n0DstStride + k * n1DstStride) * dstTypeSize;
                CumOperationTool<T0, T1, tileH, tileW, dstTypeSize, is_sum>(dst, src, tmpStride);
            }
        }
        return;
    } else if constexpr (axis == DIM_4TH) {
        constexpr auto
            dst2RawShape = Std::tuple_element<shapeSize - SECOND_LAST_DIM_OFFSET, typename T0::TileShape>::type::value;
        constexpr int tileH = dst2RawShape;
        constexpr int tileW = dst1RawShape;

        for (LoopVar m = 0; m < n0DstShape; m++) {
            for (LoopVar j = 0; j < n1DstShape; j++) {
                for (LoopVar k = 0; k < n2DstShape; k++) {
                    uint64_t tmpStride = (m * n0DstStride + j * n1DstStride + k * n2DstStride) * dstTypeSize;
                    CumOperationTool<T0, T1, tileH, tileW, dstTypeSize, is_sum>(dst, src, tmpStride);
                }
            }
        }
        return;
    } else {
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        auto srcAddr = (__ubuf__ typename T1::Type*)((uint64_t)(src.GetAddr()));
        auto dstAddr = (__ubuf__ typename T0::Type*)((uint64_t)(dst.GetAddr()));

        for (LoopVar n = 0; n < n0DstShape; n++) {
            for (LoopVar j = 0; j < n1DstShape; j++) {
                for (LoopVar k = 0; k < n2DstShape; k++) {
                    for (LoopVar m = 0; m < n3DstShape; m++) {
                        int tmpStride = n * n0DstStride + j * n1DstStride + k * n2DstStride + m * n3DstStride;
                        dstAddr[tmpStride] = srcAddr[tmpStride];
                        for (LoopVar i = 1; i < n4DstShape;) {
                            CumOperationScalarTool<T0, is_sum>(dstAddr, srcAddr, tmpStride, n4DstShape, i);
                            i = i * 2;
                        }
                    }
                }
            }
        }
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    }
}
// Partial tiles must preserve the independent source and destination strides.
template <int axis, int is_sum, typename T0, typename T1>
TILEOP void TCumOperation(T0 dst, T1 src)
{
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    size_t shape[] = {dstLayout.template GetShapeDim<0, MAX_DIMS>(), dstLayout.template GetShapeDim<1, MAX_DIMS>(),
                      dstLayout.template GetShapeDim<2, MAX_DIMS>(), dstLayout.template GetShapeDim<3, MAX_DIMS>(),
                      dstLayout.template GetShapeDim<4, MAX_DIMS>()};
    size_t dstStride[] = {
        dstLayout.template GetStrideDim<0, MAX_DIMS>(), dstLayout.template GetStrideDim<1, MAX_DIMS>(),
        dstLayout.template GetStrideDim<2, MAX_DIMS>(), dstLayout.template GetStrideDim<3, MAX_DIMS>(),
        dstLayout.template GetStrideDim<4, MAX_DIMS>()};
    size_t srcStride[] = {
        srcLayout.template GetStrideDim<0, MAX_DIMS>(), srcLayout.template GetStrideDim<1, MAX_DIMS>(),
        srcLayout.template GetStrideDim<2, MAX_DIMS>(), srcLayout.template GetStrideDim<3, MAX_DIMS>(),
        srcLayout.template GetStrideDim<4, MAX_DIMS>()};
    constexpr size_t dstRaw[] = {
        TileOp::GetTensorTileShapeDim<T0, 0, MAX_DIMS>(), TileOp::GetTensorTileShapeDim<T0, 1, MAX_DIMS>(),
        TileOp::GetTensorTileShapeDim<T0, 2, MAX_DIMS>(), TileOp::GetTensorTileShapeDim<T0, 3, MAX_DIMS>(),
        TileOp::GetTensorTileShapeDim<T0, 4, MAX_DIMS>()};
    constexpr size_t srcRaw[] = {
        TileOp::GetTensorTileShapeDim<T1, 0, MAX_DIMS>(), TileOp::GetTensorTileShapeDim<T1, 1, MAX_DIMS>(),
        TileOp::GetTensorTileShapeDim<T1, 2, MAX_DIMS>(), TileOp::GetTensorTileShapeDim<T1, 3, MAX_DIMS>(),
        TileOp::GetTensorTileShapeDim<T1, 4, MAX_DIMS>()};
    bool fullTile = true;
    for (int dim = 0; dim < MAX_DIMS; ++dim) {
        if (shape[dim] == 0) {
            return;
        }
        fullTile = fullTile && shape[dim] >= dstRaw[dim] && dstRaw[dim] == srcRaw[dim] &&
                   dstStride[dim] == srcStride[dim];
    }
    if (fullTile) {
        TCumOperationFullTile<axis, is_sum>(dst, src);
        return;
    }
    using Dtype = typename T0::Type;
    const auto count = shape[axis];
    shape[axis] = 1;
    for (LoopVar n0 = 0; n0 < shape[0]; ++n0) {
        for (LoopVar n1 = 0; n1 < shape[1]; ++n1) {
            for (LoopVar n2 = 0; n2 < shape[2]; ++n2) {
                for (LoopVar n3 = 0; n3 < shape[3]; ++n3) {
                    auto dstOffset = n0 * dstStride[0] + n1 * dstStride[1] + n2 * dstStride[2] + n3 * dstStride[3];
                    auto srcOffset = n0 * srcStride[0] + n1 * srcStride[1] + n2 * srcStride[2] + n3 * srcStride[3];
                    if constexpr (axis == DIM_5TH) {
                        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
                        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
                        auto dstPtr = reinterpret_cast<__ubuf__ Dtype*>(dst.GetAddr()) + dstOffset;
                        auto srcPtr = reinterpret_cast<__ubuf__ Dtype*>(src.GetAddr()) + srcOffset;
                        for (LoopVar i = 0; i < count; ++i) {
                            dstPtr[i * dstStride[axis]] = srcPtr[i * srcStride[axis]];
                        }
                        // Hillis-Steele scan with the same association order as the full-tile path; iterate from
                        // the high index down so each round reads pre-round values.
                        for (LoopVar offset = 1; offset < count; offset *= 2) {
                            for (LoopVar i = count - 1; i >= offset; --i) {
                                if constexpr (is_sum) {
                                    dstPtr[i * dstStride[axis]] = dstPtr[i * dstStride[axis]] +
                                                                  dstPtr[(i - offset) * dstStride[axis]];
                                } else {
                                    dstPtr[i * dstStride[axis]] = dstPtr[i * dstStride[axis]] *
                                                                  dstPtr[(i - offset) * dstStride[axis]];
                                }
                            }
                        }
                        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
                        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
                    } else {
                        using DstRow = pto::Tile<pto::TileType::Vec, Dtype, 1, dstRaw[4], pto::BLayout::RowMajor, -1,
                                                 -1>;
                        // Each tile covers one row; the source address uses its own row stride.
                        using SrcRow = DstRow;
                        // The runtime shape is view-level and may be wider than this tile, whose share is only
                        // the declared columns, so clamp the row width to the tile declaration.
                        const auto rowWidth = shape[4] < dstRaw[4] ? shape[4] : dstRaw[4];
                        DstRow dstRow(1, rowWidth), previousRow(1, rowWidth);
                        SrcRow srcRow(1, rowWidth);
                        const auto dstRowBytes = dstStride[axis] * sizeof(Dtype);
                        const auto srcRowBytes = srcStride[axis] * sizeof(Dtype);
                        const auto dstBase = dst.GetAddr() + dstOffset * sizeof(Dtype);
                        const auto srcBase = src.GetAddr() + srcOffset * sizeof(Dtype);
                        // Copy the source rows into the destination rows first: every scan round reads the
                        // pre-round values of all rows at once.
                        for (LoopVar i = 0; i < count; ++i) {
                            pto::TASSIGN(dstRow, dstBase + i * dstRowBytes);
                            pto::TASSIGN(srcRow, srcBase + i * srcRowBytes);
                            pto::TMOV(dstRow, srcRow);
                            pipe_barrier(PIPE_ALL);
                        }
                        // Hillis-Steele scan with the same association order as the full-tile path; iterate from
                        // the high index down so each round reads pre-round values.
                        for (LoopVar offset = 1; offset < count; offset *= 2) {
                            for (LoopVar i = count - 1; i >= offset; --i) {
                                pto::TASSIGN(dstRow, dstBase + i * dstRowBytes);
                                pto::TASSIGN(previousRow, dstBase + (i - offset) * dstRowBytes);
                                if constexpr (is_sum) {
                                    pto::TADD(dstRow, dstRow, previousRow);
                                } else {
                                    pto::TMUL(dstRow, dstRow, previousRow);
                                }
                                pipe_barrier(PIPE_ALL);
                            }
                        }
                    }
                }
            }
        }
    }
}
#endif
