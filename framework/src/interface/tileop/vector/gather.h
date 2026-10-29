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
 * \file gather.h
 * \brief
 */
#ifndef TILEOP_TILE_OPERATOR_GATHER__H
#define TILEOP_TILE_OPERATOR_GATHER__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#define OP_TILE_OP_GATHER_ELEMENT TgatherElement
template <int axis, typename T0, typename T1, typename T2, typename T3>
TILEOP void TgatherElement(T0 dst, T1 src0, T2 src1, T3 tmp)
{
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
    const auto srcLayout = src0.GetLayout();
    auto n0SrcStride = srcLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto n1SrcStride = srcLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto n2SrcStride = srcLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto n3SrcStride = srcLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();

    const auto idxLayout = src1.GetLayout();
    auto n0IdxShape = idxLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto n1IdxShape = idxLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto n2IdxShape = idxLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto n3IdxShape = idxLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto n4IdxShape = idxLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto n0IdxStride = idxLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto n1IdxStride = idxLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto n2IdxStride = idxLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto n3IdxStride = idxLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();

    const auto dstLayout = dst.GetLayout();
    auto n0DstStride = dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto n1DstStride = dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto n2DstStride = dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto n3DstStride = dstLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();

    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, DIM_5TH, MAX_DIMS>();
    constexpr auto idxTileW = TileOp::GetTensorTileShapeDim<T2, DIM_5TH, MAX_DIMS>();

    constexpr bool scalarFlag = (sizeof(typename T2::Type) == 8) ? true : false;
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto srcTileShape1 = TileOp::GetOutterAxisMergeResult<shapeSize, typename T1::TileShape>();
    using srcTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileShape1, srcTileW,
                                    pto::BLayout::RowMajor>;
    using idxTileDefine = pto::Tile<pto::TileType::Vec, typename T2::Type, 1, idxTileW, pto::BLayout::RowMajor, -1, -1>;
    using dstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, 1, dstTileW, pto::BLayout::RowMajor, -1, -1>;
    srcTileDefine srcTile;
    idxTileDefine idxTile(1, n4IdxShape);
    idxTileDefine tmpTile(1, n4IdxShape);
    dstTileDefine dstTile(1, n4IdxShape);
    if constexpr (scalarFlag) {
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    }
    auto srcAddr = (__ubuf__ typename T1::Type*)((uint64_t)(src0.GetAddr()));
    auto idxAddr = (__ubuf__ typename T2::Type*)((uint64_t)(src1.GetAddr()));
    auto dstAddr = (__ubuf__ typename T0::Type*)((uint64_t)(dst.GetAddr()));
    auto tmpAddr = (__ubuf__ typename T2::Type*)((uint64_t)(tmp.GetAddr()));
    auto newIdxValue = 0;
    for (LoopVar i = 0; i < n0IdxShape; ++i) {
        for (LoopVar j = 0; j < n1IdxShape; ++j) {
            for (LoopVar k = 0; k < n2IdxShape; ++k) {
                for (LoopVar l = 0; l < n3IdxShape; ++l) {
                    if constexpr (scalarFlag == false) {
                        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
                        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
                    }
                    for (LoopVar m = 0; m < n4IdxShape; ++m) {
                        auto dstOffset = i * n0DstStride + j * n1DstStride + k * n2DstStride + l * n3DstStride + m;
                        auto orgIdxValue = *(idxAddr + i * n0IdxStride + j * n1IdxStride + k * n2IdxStride +
                                             l * n3IdxStride + m);
                        if constexpr (axis == 0) {
                            newIdxValue = orgIdxValue * n0SrcStride + j * n1SrcStride + k * n2SrcStride +
                                          l * n3SrcStride + m;
                        } else if constexpr (axis == 1) {
                            newIdxValue = i * n0SrcStride + orgIdxValue * n1SrcStride + k * n2SrcStride +
                                          l * n3SrcStride + m;
                        } else if constexpr (axis == DIM_3RD) {
                            newIdxValue = i * n0SrcStride + j * n1SrcStride + orgIdxValue * n2SrcStride +
                                          l * n3SrcStride + m;
                        } else if constexpr (axis == DIM_4TH) {
                            newIdxValue = i * n0SrcStride + j * n1SrcStride + k * n2SrcStride +
                                          orgIdxValue * n3SrcStride + m;
                        } else {
                            newIdxValue = i * n0SrcStride + j * n1SrcStride + k * n2SrcStride + l * n3SrcStride +
                                          orgIdxValue;
                        }
                        if constexpr (scalarFlag) {
                            dstAddr[dstOffset] = srcAddr[newIdxValue];
                        } else {
                            *(tmpAddr + m) = newIdxValue;
                        }
                    }
                    if constexpr (scalarFlag == false) {
                        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
                        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
                        auto dstAddrOffset = i * n0DstStride + j * n1DstStride + k * n2DstStride + l * n3DstStride;
                        pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstAddrOffset * dstTypeSize));
                        pto::TASSIGN(srcTile, (uint64_t)(src0.GetAddr()));
                        pto::TASSIGN(idxTile, (uint64_t)(tmp.GetAddr()));
                        pto::TASSIGN(tmpTile, (uint64_t)(tmp.GetAddr() + idxTileW * sizeof(typename T2::Type)));
                        pto::TGATHER(dstTile, srcTile, idxTile, tmpTile);
                    }
                }
            }
        }
    }
    if constexpr (scalarFlag) {
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    }
}

template <int axis, typename T0, typename T1, typename T2, typename T3, typename C>
TILEOP void TgatherElement(T0 dst, T1 tmp, T2 src, T3 idx, C srcCoordinate)
{
    constexpr size_t expectSize = 5;
    const auto srcLayout = src.GetLayout();
    auto n0SrcShape = srcLayout.template GetShapeDim<DIM_1ST, expectSize>();
    auto n1SrcShape = srcLayout.template GetShapeDim<DIM_2ND, expectSize>();
    auto n2SrcShape = srcLayout.template GetShapeDim<DIM_3RD, expectSize>();
    auto n3SrcShape = srcLayout.template GetShapeDim<DIM_4TH, expectSize>();
    auto n4SrcShape = srcLayout.template GetShapeDim<DIM_5TH, expectSize>();
    auto n0SrcStride = srcLayout.template GetStrideDim<DIM_1ST, expectSize>();
    auto n1SrcStride = srcLayout.template GetStrideDim<DIM_2ND, expectSize>();
    auto n2SrcStride = srcLayout.template GetStrideDim<DIM_3RD, expectSize>();
    auto n3SrcStride = srcLayout.template GetStrideDim<DIM_4TH, expectSize>();
    auto n4SrcStride = srcLayout.template GetStrideDim<DIM_5TH, expectSize>();

    const auto idxLayout = idx.GetLayout();
    auto n0IdxStride = idxLayout.template GetStrideDim<DIM_1ST, expectSize>();
    auto n1IdxStride = idxLayout.template GetStrideDim<DIM_2ND, expectSize>();
    auto n2IdxStride = idxLayout.template GetStrideDim<DIM_3RD, expectSize>();
    auto n3IdxStride = idxLayout.template GetStrideDim<DIM_4TH, expectSize>();

    const auto dstLayout = dst.GetLayout();
    auto n0DstShape = dstLayout.template GetShapeDim<DIM_1ST, expectSize>();
    auto n1DstShape = dstLayout.template GetShapeDim<DIM_2ND, expectSize>();
    auto n2DstShape = dstLayout.template GetShapeDim<DIM_3RD, expectSize>();
    auto n3DstShape = dstLayout.template GetShapeDim<DIM_4TH, expectSize>();
    auto n4DstShape = dstLayout.template GetShapeDim<DIM_5TH, expectSize>();
    auto n0DstStride = dstLayout.template GetStrideDim<DIM_1ST, expectSize>();
    auto n1DstStride = dstLayout.template GetStrideDim<DIM_2ND, expectSize>();
    auto n2DstStride = dstLayout.template GetStrideDim<DIM_3RD, expectSize>();
    auto n3DstStride = dstLayout.template GetStrideDim<DIM_4TH, expectSize>();

    auto srcOffset = srcLayout.template GetGmOffset<C, expectSize>(srcCoordinate);
    using DstType = typename T0::Type;
    using SrcType = typename T2::Type;
    using IdxType = typename T3::Type;
    __ubuf__ DstType* dstAddr = (__ubuf__ DstType*)((uint64_t)(dst.GetAddr()));
    __gm__ SrcType* srcAddr = (__gm__ SrcType*)((uint64_t)(src.GetAddr())) + srcOffset;
    __ubuf__ IdxType* idxAddr = (__ubuf__ IdxType*)((uint64_t)(idx.GetAddr()));

    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

#if defined(__DAV_V310)
    if constexpr (std::is_same_v<IdxType, int32_t> || std::is_same_v<IdxType, uint32_t>) {
        constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, expectSize>();
        using DstTile = pto::Tile<pto::TileType::Vec, DstType, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
        using OffsetTile = pto::Tile<pto::TileType::Vec, IdxType, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
        using TableShape = pto::Shape<-1, -1, -1, -1, -1>;
        using TableStride = pto::Stride<-1, -1, -1, -1, -1>;
        using TableGlobal = pto::GlobalTensor<SrcType, TableShape, TableStride>;
        TableGlobal tableGM(srcAddr, TableShape(n0SrcShape, n1SrcShape, n2SrcShape, n3SrcShape, n4SrcShape),
                            TableStride(n0SrcStride, n1SrcStride, n2SrcStride, n3SrcStride, n4SrcStride));
        const auto tmpHalfSize = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, expectSize>();
        __ubuf__ IdxType* offsetAddr = (__ubuf__ IdxType*)((uint64_t)(tmp.GetAddr()));
        __ubuf__ IdxType* laneAddr = offsetAddr + tmpHalfSize;
        const unsigned validCol = static_cast<unsigned>(n4DstShape);
        OffsetTile offsetTile(1U, validCol);
        OffsetTile indicesTile(1U, validCol);
        OffsetTile laneTile(1U, validCol);
        pto::TASSIGN(offsetTile, (uint64_t)(offsetAddr));
        pto::TASSIGN(laneTile, (uint64_t)(laneAddr));
        if constexpr (axis != DIM_5TH) {
            pto::TCI<OffsetTile, IdxType, 0>(laneTile, static_cast<IdxType>(0));
            set_flag(PIPE_S, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
            if (n4SrcStride != 1) {
                pto::TMULS(laneTile, laneTile, static_cast<IdxType>(n4SrcStride));
            }
        }

        int64_t axisStride = n4SrcStride;
        if constexpr (axis == 0) {
            axisStride = n0SrcStride;
        } else if constexpr (axis == 1) {
            axisStride = n1SrcStride;
        } else if constexpr (axis == DIM_3RD) {
            axisStride = n2SrcStride;
        } else if constexpr (axis == DIM_4TH) {
            axisStride = n3SrcStride;
        }

        for (LoopVar i = 0; i < n0DstShape; ++i) {
            for (LoopVar j = 0; j < n1DstShape; ++j) {
                for (LoopVar k = 0; k < n2DstShape; ++k) {
                    for (LoopVar l = 0; l < n3DstShape; ++l) {
                        auto idxRowOffset = i * n0IdxStride + j * n1IdxStride + k * n2IdxStride + l * n3IdxStride;
                        int64_t outerBase = 0;
                        if constexpr (axis != 0) {
                            outerBase += i * n0SrcStride;
                        }
                        if constexpr (axis != 1) {
                            outerBase += j * n1SrcStride;
                        }
                        if constexpr (axis != DIM_3RD) {
                            outerBase += k * n2SrcStride;
                        }
                        if constexpr (axis != DIM_4TH) {
                            outerBase += l * n3SrcStride;
                        }

                        pto::TASSIGN(indicesTile, (uint64_t)(idxAddr + idxRowOffset));
                        pto::TMULS(offsetTile, indicesTile, static_cast<IdxType>(axisStride));
                        if constexpr (axis != DIM_5TH) {
                            pto::TADD(offsetTile, offsetTile, laneTile);
                            if (outerBase != 0) {
                                pto::TADDS(offsetTile, offsetTile, static_cast<IdxType>(outerBase));
                            }
                        } else if (outerBase != 0) {
                            pto::TADDS(offsetTile, offsetTile, static_cast<IdxType>(outerBase));
                        }

                        DstTile dstTile(1U, validCol);
                        auto dstRowOffset = i * n0DstStride + j * n1DstStride + k * n2DstStride + l * n3DstStride;
                        pto::TASSIGN(dstTile, (uint64_t)(dstAddr + dstRowOffset));
                        pipe_barrier(PIPE_ALL);
                        pto::MGATHER<pto::Coalesce::Elem, pto::GatherOOB::Zero>(dstTile, tableGM, offsetTile);
                        pipe_barrier(PIPE_ALL);
                    }
                }
            }
        }
    } else
#endif
    {
        for (LoopVar i = 0; i < n0DstShape; ++i) {
            for (LoopVar j = 0; j < n1DstShape; ++j) {
                for (LoopVar k = 0; k < n2DstShape; ++k) {
                    for (LoopVar l = 0; l < n3DstShape; ++l) {
                        for (LoopVar m = 0; m < n4DstShape; ++m) {
                            auto idxElementOffset = i * n0IdxStride + j * n1IdxStride + k * n2IdxStride +
                                                    l * n3IdxStride + m;
                            int64_t index = static_cast<int64_t>(idxAddr[idxElementOffset]);
                            auto srcElementOffset = i * n0SrcStride + j * n1SrcStride + k * n2SrcStride +
                                                    l * n3SrcStride + m;
                            if constexpr (axis == 0) {
                                srcElementOffset = index * n0SrcStride + j * n1SrcStride + k * n2SrcStride +
                                                   l * n3SrcStride + m;
                            } else if constexpr (axis == 1) {
                                srcElementOffset = i * n0SrcStride + index * n1SrcStride + k * n2SrcStride +
                                                   l * n3SrcStride + m;
                            } else if constexpr (axis == DIM_3RD) {
                                srcElementOffset = i * n0SrcStride + j * n1SrcStride + index * n2SrcStride +
                                                   l * n3SrcStride + m;
                            } else if constexpr (axis == DIM_4TH) {
                                srcElementOffset = i * n0SrcStride + j * n1SrcStride + k * n2SrcStride +
                                                   index * n3SrcStride + m;
                            } else {
                                srcElementOffset = i * n0SrcStride + j * n1SrcStride + k * n2SrcStride +
                                                   l * n3SrcStride + index;
                            }
                            auto dstElementOffset = i * n0DstStride + j * n1DstStride + k * n2DstStride +
                                                    l * n3DstStride + m;
                            dstAddr[dstElementOffset] = srcAddr[srcElementOffset];
                        }
                    }
                }
            }
        }
        dcci(src.GetAddr(), 0);
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
}

template <int axis, size_t index0, size_t index1, size_t index2, size_t index3, size_t index4, typename T0, typename T1,
          typename T2, typename C1, typename C2>
TILEOP void Tgather(T0 dst, T1 src, T2 idx, C1 srcCoordinate, C2 idxCoordinate)
{
    constexpr size_t srcExpectSize = 4;
    constexpr size_t idxExpectSize = 2;
    const auto srcLayout = src.GetLayout();
    auto n0SrcStride = srcLayout.template GetStrideDim<DIM_1ST, srcExpectSize>();
    auto n1SrcStride = srcLayout.template GetStrideDim<DIM_2ND, srcExpectSize>();
    auto n2SrcStride = srcLayout.template GetStrideDim<DIM_3RD, srcExpectSize>();
    auto n3SrcStride = srcLayout.template GetStrideDim<DIM_4TH, srcExpectSize>();

    const auto idxLayout = idx.GetLayout();
    auto n0IdxStride = idxLayout.template GetStrideDim<DIM_1ST, idxExpectSize>();

    const auto dstLayout = dst.GetLayout();
    auto n0DstStride = dstLayout.template GetStrideDim<index0, MAX_DIMS>();
    auto n1DstStride = dstLayout.template GetStrideDim<index1, MAX_DIMS>();
    auto n2DstStride = dstLayout.template GetStrideDim<index2, MAX_DIMS>();
    auto n3DstStride = dstLayout.template GetStrideDim<index3, MAX_DIMS>();

    auto n0DstShape = dstLayout.template GetShapeDim<index0, MAX_DIMS>();
    auto n1DstShape = dstLayout.template GetShapeDim<index1, MAX_DIMS>();
    auto n2DstShape = dstLayout.template GetShapeDim<index2, MAX_DIMS>();
    auto n3DstShape = dstLayout.template GetShapeDim<index3, MAX_DIMS>();
    auto n4DstShape = dstLayout.template GetShapeDim<index4, MAX_DIMS>();

    auto srcOffset = srcLayout.template GetGmOffset<C1, MAX_DIMS>(srcCoordinate);
    auto idxOffset = idxLayout.template GetGmOffset<C2, MAX_DIMS>(idxCoordinate);
    using idxType = typename T2::Type;
    using srcType = typename T1::Type;
    using dstType = typename T0::Type;
    __gm__ srcType* srcAddr = (__gm__ srcType*)((uint64_t)(src.GetAddr()));
    __gm__ idxType* idxAddr = (__gm__ idxType*)((uint64_t)(idx.GetAddr()));
    srcAddr += srcOffset;
    idxAddr += idxOffset;
    __ubuf__ dstType* dstAddr = (__ubuf__ dstType*)((uint64_t)(dst.GetAddr()));
    constexpr auto tileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    using ShapeDim5 = pto::Shape<-1, -1, -1, -1, -1>;
    using StrideDim5 = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = pto::GlobalTensor<srcType, ShapeDim5, StrideDim5>;
    using TileDefine = pto::Tile<pto::TileType::Vec, dstType, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    if constexpr (axis == 0) {
        __gm__ idxType* idx0 = idxAddr;
        for (LoopVar i = 0; i < n0DstShape; i++) {
            __gm__ dstType* src0 = srcAddr;
            __ubuf__ dstType* dst0 = dstAddr;
            for (LoopVar j = 0; j < n1DstShape; j++) {
                __ubuf__ dstType* dst1 = dst0;
                uint64_t index = idx0[j];
                src0 = srcAddr + index * n0SrcStride;
                for (LoopVar k = 0; k < n2DstShape; k++) {
                    TileDefine dstTile(n3DstShape, n4DstShape);
                    GlobalData srcGlobal(src0, pto::Shape(1, 1, 1, n3DstShape, n4DstShape),
                                         pto::Stride(0, 0, 0, n2SrcStride, n3SrcStride));
                    pto::TASSIGN(dstTile, (uint64_t)dst1);
                    pto::TLOAD(dstTile, srcGlobal);
                    dst1 += n2DstStride;
                    src0 += n1SrcStride;
                }
                dst0 += n1DstStride;
            }
            dstAddr += n0DstStride;
            idx0 += n0IdxStride;
        }
    } else if constexpr (axis == 1) {
        for (LoopVar i = 0; i < n0DstShape; i++) { // a
            __gm__ dstType* src0 = srcAddr;
            __gm__ idxType* idx0 = idxAddr;
            __ubuf__ dstType* dst0 = dstAddr;
            for (LoopVar j = 0; j < n1DstShape; j++) { // e
                __ubuf__ dstType* dst1 = dst0;
                for (LoopVar k = 0; k < n2DstShape; k++) {
                    uint64_t index = idx0[k];
                    src0 = srcAddr + index * n1SrcStride;
                    TileDefine dstTile(n3DstShape, n4DstShape);
                    GlobalData srcGlobal(src0, pto::Shape(1, 1, 1, n3DstShape, n4DstShape),
                                         pto::Stride(0, 0, 0, n2SrcStride, n3SrcStride));
                    pto::TASSIGN(dstTile, (uint64_t)dst1);
                    pto::TLOAD(dstTile, srcGlobal);
                    dst1 += n2DstStride;
                }
                dst0 += n1DstStride;
                idx0 += n0IdxStride;
            }
            dstAddr += n0DstStride;
            srcAddr += n0SrcStride;
        }
    } else if constexpr (axis == DIM_3RD) {
        for (LoopVar i = 0; i < n0DstShape; i++) {
            __gm__ dstType* src0 = srcAddr;
            __ubuf__ dstType* dst0 = dstAddr;
            for (LoopVar j = 0; j < n1DstShape; j++) { // b
                __gm__ idxType* idx0 = idxAddr;
                __gm__ dstType* src1 = src0;
                __ubuf__ dstType* dst1 = dst0;
                for (LoopVar k = 0; k < n2DstShape; k++) { // e
                    __ubuf__ dstType* dst2 = dst1;
                    for (LoopVar l = 0; l < n3DstShape; l++) { // f
                        uint64_t index = idx0[l];
                        src1 = src0 + index * n2SrcStride;
                        TileDefine dstTile(1, n4DstShape);
                        GlobalData srcGlobal(src1, pto::Shape(1, 1, 1, 1, n4DstShape),
                                             pto::Stride(0, 0, 0, n2SrcStride, n3SrcStride));
                        pto::TASSIGN(dstTile, (uint64_t)dst2);
                        pto::TLOAD(dstTile, srcGlobal);
                        dst2 += n3DstStride;
                    }
                    dst1 += n2DstStride;
                    idx0 += n0IdxStride;
                }
                src0 += n1SrcStride;
                dst0 += n1DstStride;
            }
            srcAddr += n0SrcStride;
            dstAddr += n0DstStride;
        }
    } else if constexpr (axis == DIM_4TH) {
        for (LoopVar i = 0; i < n0DstShape; i++) {
            __gm__ dstType* src0 = srcAddr;
            __ubuf__ dstType* dst0 = dstAddr;
            for (LoopVar j = 0; j < n1DstShape; j++) { // b
                __gm__ dstType* src1 = src0;
                __ubuf__ dstType* dst1 = dst0;
                for (LoopVar k = 0; k < n2DstShape; k++) { // c
                    __gm__ dstType* src2 = src1;
                    __ubuf__ dstType* dst2 = dst1;
                    __gm__ idxType* idx0 = idxAddr;
                    for (LoopVar l = 0; l < n3DstShape; l++) { // e
                        __ubuf__ dstType* dst3 = dst2;
                        for (LoopVar p = 0; p < n4DstShape; p++) {
                            dcci(idx0, 0);
                            uint64_t index = idx0[p];
                            src2 = src1 + index;
                            *dst3 = *src2;
                            dst3++;
                        }
                        dst2 += n3DstStride;
                        idx0 += n0IdxStride;
                    }
                    src1 += n2SrcStride;
                    dst1 += n2DstStride;
                }
                src0 += n1SrcStride;
                dst0 += n1DstStride;
            }
            srcAddr += n0SrcStride;
            dstAddr += n0DstStride;
        }
        dcci(src.GetAddr(), 0);
        dcci(idx.GetAddr(), 0);
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    }
}

/**
 * 辅助函数，计算 token id 对应物理偏移
 */
template <typename T2, typename T3, unsigned blockSize>
INLINE T2 CalaOffset2PageAttention(__gm__ T3* blockTable, T2 index)
{
    T2 blockID = index / blockSize;            // 这个token对应第几个块，逻辑块
    blockID = blockTable[blockID];             // 页表中存放这个逻辑块到物理块的映射，得到物理块
    T2 blockOffset = index % blockSize;        // token在块内偏移
    index = blockID * blockSize + blockOffset; // 物理块*blockSize 得到物理块的偏移，加上token在块中的偏移
    return index;
}
/**
 * 定制版本，只支持 ds v3.2，使用之前请仔细确认
 * param 2维
 * indices 2维
 * axis -2
 * result 2维 {和标准实现不同}
 * [a,b] [1,c] -2  [c,b]
 *
 * 模板参数
 * T input 参数类型
 * T2 indices 参数类型
 * UBOutputS*  output在ub上的步长
 *
 * 运行时参数
 * dst result，ub上
 * param 输入，在gm上
 * indices 输入索引，在gm上
 * GMParamShape* ,param validshape，用于指导拷贝长度
 * GMParamStride*,param 的步长，用于计算偏移
 * GMParamOffset*,param 的偏移，用于确定分块
 * GMIndicesShape* ,indices 的 validshape ，用于指导循环，
 * GMIndicesStride* ,步长，用于计算偏移
 * blocktable[e,f] e batch的维度  f ceil(maxtoken/blockSize)
 */
template <unsigned blockSize, typename T0, typename T1, typename T2, typename T3, typename C1, typename C2, typename C3>
TILEOP void TgatherInUB(T0 dst, T1 param, T2 indices, T3 blockTable, C1 paramCoordinate, C2 indicesCoordinate,
                        C3 blockTableCoordinate)
{
    constexpr size_t paramExpectSize = 2;
    constexpr size_t dstExpectSize = 2;
    const auto paramLayout = param.GetLayout();
    auto n0ParamLayoutStride = paramLayout.template GetStrideDim<DIM_1ST, paramExpectSize>();
    auto n1ParamLayoutStride = paramLayout.template GetStrideDim<DIM_2ND, paramExpectSize>();

    const auto indicesLayout = indices.GetLayout();

    const auto blockTableLayout = blockTable.GetLayout();

    const auto dstLayout = dst.GetLayout();
    auto n0DstLayoutStride = dstLayout.template GetStrideDim<DIM_1ST, dstExpectSize>();
    auto n0DstShape = dstLayout.template GetShapeDim<DIM_1ST, dstExpectSize>();
    auto n1DstShape = dstLayout.template GetShapeDim<DIM_2ND, dstExpectSize>();

    auto paramOffset = paramLayout.template GetGmOffset<C1, MAX_DIMS>(paramCoordinate);
    auto indicesOffset = indicesLayout.template GetGmOffset<C2, MAX_DIMS>(indicesCoordinate);
    auto blockTableOffset = blockTableLayout.template GetGmOffset<C3, MAX_DIMS>(blockTableCoordinate);
    using dstType = typename T0::Type;
    using paramType = typename T1::Type;
    using indicesType = typename T2::Type;
    using blockTableType = typename T3::Type;
    __gm__ paramType* paramAddr = (__gm__ paramType*)((uint64_t)(param.GetAddr()));
    __gm__ indicesType* indicesAddr = (__gm__ indicesType*)((uint64_t)(indices.GetAddr()));
    __gm__ blockTableType* blockTableAddr = (__gm__ blockTableType*)((uint64_t)(blockTable.GetAddr()));
    paramAddr += paramOffset;
    indicesAddr += indicesOffset;
    blockTableAddr += blockTableOffset;
    /**
     * 思路
     * [token_size,dim]
     * [1,topk]
     * [1,pagetable]
     * 主要是这个遍历indices 的1轴，
     */
    __ubuf__ dstType* dstAddr = (__ubuf__ dstType*)((uint64_t)(dst.GetAddr()));
    // auto n0ParamStride = paramLayout.template GetStrideDim<index0, paramExpectSize>();

    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr size_t TILE_HEIGHT_OFFSET = 2;
    constexpr auto tileH = Std::tuple_element<shapeSize - TILE_HEIGHT_OFFSET, typename T0::TileShape>::type::value;
    constexpr auto tileW = Std::tuple_element<shapeSize - 1, typename T0::TileShape>::type::value;
    using ShapeDim5 = pto::Shape<-1, -1, -1, -1, -1>;
    using StrideDim5 = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = pto::GlobalTensor<paramType, ShapeDim5, StrideDim5>;
    using TileDefine = pto::Tile<pto::TileType::Vec, dstType, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    __gm__ paramType* paramTmp = paramAddr;

    for (int j = 0; j < n0DstShape; j++) {
        uint64_t index_1 = indicesAddr[j];
        index_1 = CalaOffset2PageAttention<uint64_t, blockTableType, blockSize>(blockTableAddr, index_1);
        paramTmp = paramAddr + index_1 * n0ParamLayoutStride; // 得到了实际的地址

        TileDefine dstTile(1, n1DstShape);
        GlobalData srcGlobal(paramTmp, pto::Shape(1, 1, 1, 1, n1DstShape),
                             pto::Stride(0, 0, 0, n0ParamLayoutStride, n1ParamLayoutStride));
        pto::TASSIGN(dstTile, (uint64_t)dstAddr);
        pto::TLOAD(dstTile, srcGlobal);

        dstAddr += n0DstLayoutStride;
    }
}

#define OP_TILE_OP_GATHER_MASK TGatherMask
template <int patternMode, typename T0, typename T1>
TILEOP void TGatherMask(T0 dst, T1 src)
{
    static_assert(patternMode >= static_cast<int>(pto::MaskPattern::P0101) &&
                      patternMode <= static_cast<int>(pto::MaskPattern::P1111),
                  "patternMode must be in range [1, 7]");
    constexpr auto pattern = static_cast<pto::MaskPattern>(patternMode);
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto dstStride0 = dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto dstStride1 = dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto dstStride2 = dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto dstStride3 = dstLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();
    auto srcShape4 = srcLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto srcStride0 = srcLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto srcStride1 = srcLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto srcStride2 = srcLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto srcStride3 = srcLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, DIM_5TH, MAX_DIMS>();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto srcTypeSize = sizeof(typename T1::Type);
    if (dstShape3 == 0 || dstShape4 == 0) {
        return;
    }
    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                for (LoopVar n3Index = 0; n3Index < dstShape3; ++n3Index) {
                    using DstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, 1, dstTileW,
                                                    pto::BLayout::RowMajor, -1, -1>;
                    using SrcTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, 1, srcTileW,
                                                    pto::BLayout::RowMajor, -1, -1>;
                    DstTileDefine dstTile(1, dstShape4);
                    SrcTileDefine srcTile(1, srcShape4);
                    auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2 +
                                     n3Index * dstStride3;
                    auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2 +
                                     n3Index * srcStride3;
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                    pto::TGATHER<DstTileDefine, SrcTileDefine, pattern>(dstTile, srcTile);
                }
            }
        }
    }
}

#endif
