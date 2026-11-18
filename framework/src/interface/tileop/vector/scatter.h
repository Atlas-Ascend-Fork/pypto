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
 * \file scatter.h
 * \brief
 */
#ifndef TILEOP_TILE_OPERATOR_SCATTER__H
#define TILEOP_TILE_OPERATOR_SCATTER__H
#include <type_traits>
#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6)
#include "tileop_common.h"
#endif
#include "utils/layout.h"
#include "utils/tile_tensor.h"

constexpr unsigned SCATTER_MODE_MAX = 3;
template <int axis, int scatterMode, typename T0, typename T1, typename Scalar>
TILEOP void TscatterElementS(T0 dst, T1 src1, Scalar src2)
{
    static_assert(scatterMode < SCATTER_MODE_MAX, "Unsupport scatterMode");
    const auto dstLayout = dst.GetLayout();
    auto n0DstStride = dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto n1DstStride = dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto n2DstStride = dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto n3DstStride = dstLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();

    const auto idxLayout = src1.GetLayout();
    auto n0IdxStride = idxLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto n1IdxStride = idxLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto n2IdxStride = idxLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto n3IdxStride = idxLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();
    auto n0IdxShape = idxLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto n1IdxShape = idxLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto n2IdxShape = idxLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto n3IdxShape = idxLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto n4IdxShape = idxLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    auto idxAddr = (__ubuf__ typename T1::Type*)((uint64_t)(src1.GetAddr()));
    auto dstAddr = (__ubuf__ typename T0::Type*)((uint64_t)(dst.GetAddr()));
    for (LoopVar i = 0; i < n0IdxShape; ++i) {
        for (LoopVar j = 0; j < n1IdxShape; ++j) {
            for (LoopVar k = 0; k < n2IdxShape; ++k) {
                for (LoopVar l = 0; l < n3IdxShape; ++l) {
                    for (LoopVar m = 0; m < n4IdxShape; ++m) {
                        typename T1::Type index = *(idxAddr + i * n0IdxStride + j * n1IdxStride + k * n2IdxStride +
                                                    l * n3IdxStride + m);
                        typename T1::Type dstOffset = 0;
                        if constexpr (axis == 0) {
                            dstOffset = index * n0DstStride + j * n1DstStride + k * n2DstStride + l * n3DstStride + m;
                        } else if constexpr (axis == 1) {
                            dstOffset = i * n0DstStride + index * n1DstStride + k * n2DstStride + l * n3DstStride + m;
                        } else if constexpr (axis == DIM_3RD) {
                            dstOffset = i * n0DstStride + j * n1DstStride + index * n2DstStride + l * n3DstStride + m;
                        } else if constexpr (axis == DIM_4TH) {
                            dstOffset = i * n0DstStride + j * n1DstStride + k * n2DstStride + index * n3DstStride + m;
                        } else {
                            dstOffset = i * n0DstStride + j * n1DstStride + k * n2DstStride + l * n3DstStride + index;
                        }
                        if constexpr (scatterMode == 0) {
                            dstAddr[dstOffset] = src2;
                        } else if constexpr (scatterMode == 1) {
                            if constexpr (std::is_integral_v<typename T0::Type>) {
                                dstAddr[dstOffset] = static_cast<typename T0::Type>(
                                    static_cast<typename T0::Type>(src2) + dstAddr[dstOffset]);
                            } else {
                                dstAddr[dstOffset] = static_cast<typename T0::Type>(
                                    static_cast<float>(src2) + static_cast<float>(dstAddr[dstOffset]));
                            }
                        } else {
                            if constexpr (std::is_integral_v<typename T0::Type>) {
                                dstAddr[dstOffset] = static_cast<typename T0::Type>(
                                    static_cast<typename T0::Type>(src2) * dstAddr[dstOffset]);
                            } else {
                                dstAddr[dstOffset] = static_cast<typename T0::Type>(
                                    static_cast<float>(src2) * static_cast<float>(dstAddr[dstOffset]));
                            }
                        }
                    }
                }
            }
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
}

template <int axis, int scatterMode, typename T0, typename T1, typename T2, typename T3>
TILEOP void Tscatter(T0 dst, T1 src1, T2 src2, T3 tmp)
{
    static_assert(scatterMode < SCATTER_MODE_MAX, "Unsupport scatterMode");
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
    const auto dstLayout = dst.GetLayout();
    auto dstStride0 = dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto dstStride1 = dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto dstStride2 = dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto dstStride3 = dstLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();

    const auto idxLayout = src1.GetLayout();
    auto idxStride0 = idxLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto idxStride1 = idxLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto idxStride2 = idxLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto idxStride3 = idxLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();
    auto idxShape0 = idxLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto idxShape1 = idxLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto idxShape2 = idxLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto idxShape3 = idxLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto idxShape4 = idxLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    const auto srcLayout = src2.GetLayout();
    auto srcStride0 = srcLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto srcStride1 = srcLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto srcStride2 = srcLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto srcStride3 = srcLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();

    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    constexpr auto idxTileW = TileOp::GetTensorTileShapeDim<T1, DIM_5TH, MAX_DIMS>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T2, DIM_5TH, MAX_DIMS>();

    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto idxTypeSize = sizeof(typename T1::Type);
    constexpr auto srcTypeSize = sizeof(typename T2::Type);
#ifdef __DAV_V220
    /* A2 A3不支持vscatter指令，调用pto封装接口会导致性能劣化，因此pypto自行用scalar计算实现，A5正常调用pto接口 */
    constexpr bool scalarFlag = true;
#else
    constexpr bool scalarFlag = ((sizeof(typename T1::Type) == sizeof(int64_t)) || (scatterMode > 0) ||
                                 (dstTypeSize == sizeof(int16_t) && idxTypeSize == sizeof(int32_t)) ||
                                 (dstTypeSize == sizeof(int8_t) && idxTypeSize == sizeof(int32_t)) ||
                                 (dstTypeSize == sizeof(int32_t) && idxTypeSize == sizeof(int32_t))) ?
                                    true :
                                    false;
#endif
    constexpr auto dstTileShapeH = TileOp::GetOutterAxisMergeResult<shapeSize, typename T0::TileShape>();
    using dstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileShapeH, dstTileW,
                                    pto::BLayout::RowMajor>;
    using idxTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, 1, idxTileW, pto::BLayout::RowMajor, -1, -1>;
    using srcTileDefine = pto::Tile<pto::TileType::Vec, typename T2::Type, 1, srcTileW, pto::BLayout::RowMajor>;
    dstTileDefine dstTile;
    idxTileDefine idxTile(1, idxShape4);
    srcTileDefine srcTile;

    if constexpr (scalarFlag) {
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    }
    auto dstAddr = (__ubuf__ typename T0::Type*)((uint64_t)(dst.GetAddr()));
    auto idxAddr = (__ubuf__ typename T1::Type*)((uint64_t)(src1.GetAddr()));
    auto srcAddr = (__ubuf__ typename T2::Type*)((uint64_t)(src2.GetAddr()));
    auto tmpAddr = (__ubuf__ typename T3::Type*)((uint64_t)(tmp.GetAddr()));
    typename T1::Type dstOffset = 0;
    for (LoopVar i = 0; i < idxShape0; ++i) {
        for (LoopVar j = 0; j < idxShape1; ++j) {
            for (LoopVar k = 0; k < idxShape2; ++k) {
                for (LoopVar l = 0; l < idxShape3; ++l) {
                    if constexpr (scalarFlag == false) {
                        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
                        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
                    }
                    for (LoopVar m = 0; m < idxShape4; ++m) {
                        typename T1::Type index = *(idxAddr + i * idxStride0 + j * idxStride1 + k * idxStride2 +
                                                    l * idxStride3 + m);
                        typename T1::Type src2Offset = i * srcStride0 + j * srcStride1 + k * srcStride2 +
                                                       l * srcStride3 + m;
                        if constexpr (axis == 0) {
                            dstOffset = index * dstStride0 + j * dstStride1 + k * dstStride2 + l * dstStride3 + m;
                        } else if constexpr (axis == 1) {
                            dstOffset = i * dstStride0 + index * dstStride1 + k * dstStride2 + l * dstStride3 + m;
                        } else if constexpr (axis == DIM_3RD) {
                            dstOffset = i * dstStride0 + j * dstStride1 + index * dstStride2 + l * dstStride3 + m;
                        } else if constexpr (axis == DIM_4TH) {
                            dstOffset = i * dstStride0 + j * dstStride1 + k * dstStride2 + index * dstStride3 + m;
                        } else {
                            dstOffset = i * dstStride0 + j * dstStride1 + k * dstStride2 + l * dstStride3 + index;
                        }
                        /* idx类型为int64或scatter操作为add或multiply，退化为标量实现 */
                        if constexpr (scalarFlag) {
                            if constexpr (scatterMode == 0) {
                                dstAddr[dstOffset] = srcAddr[src2Offset];
                            } else if constexpr (scatterMode == 1) {
                                dstAddr[dstOffset] = srcAddr[src2Offset] + dstAddr[dstOffset];
                            } else {
                                dstAddr[dstOffset] = srcAddr[src2Offset] * dstAddr[dstOffset];
                            }
                        } else {
                            *(tmpAddr + m) = dstOffset;
                        }
                    }
                    if constexpr (scalarFlag == false) {
                        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
                        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
                        auto srcOffset = i * srcStride0 + j * srcStride1 + k * srcStride2 + l * srcStride3;
                        pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr()));
                        pto::TASSIGN(idxTile, (uint64_t)(tmp.GetAddr()));
                        pto::TASSIGN(srcTile, (uint64_t)(src2.GetAddr() + srcOffset * srcTypeSize));
                        pto::TSCATTER(dstTile, srcTile, idxTile);
                    }
                }
            }
        }
    }
    if constexpr (scalarFlag) {
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    } else {
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    }
}

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6)
struct ScatterGmTileInfo {
    uint64_t shape[3];
    uint64_t idxStride[4];
    uint64_t srcStride[4];
    uint64_t dstStride[5];
};

template <int axis, int scatterMode, typename DstType, typename IdxType, typename SrcType>
__simt_vf__ AICORE LAUNCH_BOUND(1024) inline void ScatterGmSimt(__gm__ DstType* dst, __ubuf__ const IdxType* indices,
                                                                __ubuf__ const SrcType* values, ScatterGmTileInfo info,
                                                                uint32_t rows, uint32_t cols)
{
    static_assert(scatterMode == 0, "GM Scatter SIMT only supports overwrite");
    const uint32_t warps = rows < 32 ? rows : 32;
    for (uint32_t row = threadIdx.y; row < rows; row += warps) {
        uint32_t remaining = row;
        const uint32_t l = remaining % info.shape[2];
        remaining /= info.shape[2];
        const uint32_t k = remaining % info.shape[1];
        remaining /= info.shape[1];
        const uint32_t j = remaining % info.shape[0];
        const uint32_t i = remaining / info.shape[0];
        const int64_t idxRow = i * info.idxStride[0] + j * info.idxStride[1] + k * info.idxStride[2] +
                               l * info.idxStride[3];
        const int64_t srcRow = i * info.srcStride[0] + j * info.srcStride[1] + k * info.srcStride[2] +
                               l * info.srcStride[3];
        const int64_t baseOffset = (axis == 0 ? 0 : i * info.dstStride[0]) + (axis == 1 ? 0 : j * info.dstStride[1]) +
                                   (axis == 2 ? 0 : k * info.dstStride[2]) + (axis == 3 ? 0 : l * info.dstStride[3]);
        for (uint32_t m = threadIdx.x; m < cols; m += 32) {
            const IdxType index = indices[idxRow + m];
            const int64_t offset = baseOffset + (axis == 4 ? static_cast<int64_t>(index) * info.dstStride[4] :
                                                             static_cast<int64_t>(index) * info.dstStride[axis] +
                                                                 static_cast<int64_t>(m) * info.dstStride[4]);
            dst[offset] = static_cast<DstType>(values[srcRow + m]);
        }
    }
}

template <int axis, int scatterMode, typename DstType, typename IdxType>
__simt_vf__ AICORE LAUNCH_BOUND(1024) inline void ScatterScalarSimt(__gm__ DstType* dst,
                                                                    __ubuf__ const IdxType* indices, DstType value,
                                                                    ScatterGmTileInfo info, uint32_t rows,
                                                                    uint32_t cols)
{
    static_assert(scatterMode == 0, "GM scalar Scatter only supports overwrite");
    const uint32_t warps = rows < 32 ? rows : 32;
    for (uint32_t row = threadIdx.y; row < rows; row += warps) {
        uint32_t remaining = row;
        const uint32_t l = remaining % info.shape[2];
        remaining /= info.shape[2];
        const uint32_t k = remaining % info.shape[1];
        remaining /= info.shape[1];
        const uint32_t j = remaining % info.shape[0];
        const uint32_t i = remaining / info.shape[0];
        const int64_t idxRow = i * info.idxStride[0] + j * info.idxStride[1] + k * info.idxStride[2] +
                               l * info.idxStride[3];
        const int64_t baseOffset = (axis == 0 ? 0 : i * info.dstStride[0]) + (axis == 1 ? 0 : j * info.dstStride[1]) +
                                   (axis == 2 ? 0 : k * info.dstStride[2]) + (axis == 3 ? 0 : l * info.dstStride[3]);
        for (uint32_t m = threadIdx.x; m < cols; m += 32) {
            const IdxType index = indices[idxRow + m];
            const int64_t offset = baseOffset + (axis == 4 ? static_cast<int64_t>(index) * info.dstStride[4] :
                                                             static_cast<int64_t>(index) * info.dstStride[axis] +
                                                                 static_cast<int64_t>(m) * info.dstStride[4]);
            dst[offset] = value;
        }
    }
}

template <int axis, int scatterMode, typename T0, typename T1, typename Scalar>
TILEOP void TscatterElementSInplaceImpl(T0 dst, T1 indices, Scalar scalar, __gm__ std::remove_cv_t<Scalar>* dstAddr)
{
    static_assert(scatterMode == 0, "GM Scatter only supports overwrite");
    constexpr auto expectSize = MAX_DIMS;
    const auto dstLayout = dst.GetLayout();
    const auto idxLayout = indices.GetLayout();
    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();
    auto dstStride3 = dstLayout.template GetStrideDim<3, expectSize>();
    auto dstStride4 = dstLayout.template GetStrideDim<4, expectSize>();
    auto idxStride0 = idxLayout.template GetStrideDim<0, expectSize>();
    auto idxStride1 = idxLayout.template GetStrideDim<1, expectSize>();
    auto idxStride2 = idxLayout.template GetStrideDim<2, expectSize>();
    auto idxStride3 = idxLayout.template GetStrideDim<3, expectSize>();
    auto idxShape0 = idxLayout.template GetShapeDim<0, expectSize>();
    auto idxShape1 = idxLayout.template GetShapeDim<1, expectSize>();
    auto idxShape2 = idxLayout.template GetShapeDim<2, expectSize>();
    auto idxShape3 = idxLayout.template GetShapeDim<3, expectSize>();
    auto idxShape4 = idxLayout.template GetShapeDim<4, expectSize>();
    auto idxAddr = reinterpret_cast<__ubuf__ const typename T1::Type*>(indices.GetAddr());
    using DstType = std::remove_cv_t<Scalar>;
    using IdxType = typename T1::Type;
    const uint32_t rows = idxShape0 * idxShape1 * idxShape2 * idxShape3;
    const ScatterGmTileInfo info{{idxShape1, idxShape2, idxShape3},
                                 {idxStride0, idxStride1, idxStride2, idxStride3},
                                 {0, 0, 0, 0},
                                 {dstStride0, dstStride1, dstStride2, dstStride3, dstStride4}};
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    if (rows != 0 && idxShape4 != 0) {
        const uint32_t warps = rows < 32 ? rows : 32;
        cce::async_invoke<ScatterScalarSimt<axis, scatterMode, DstType, IdxType>>(
            cce::dim3{32, warps}, dstAddr, idxAddr, static_cast<DstType>(scalar), info, rows, idxShape4);
    }
    dcci(static_cast<__gm__ void*>(0), cache_line_t::ENTIRE_DATA_CACHE);
    dsb(DSB_DDR);
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
}

template <int axis, int scatterMode, typename T0, typename C, typename T1, typename Scalar>
TILEOP void TscatterElementSInplace(T0 dst, C coordinate, T1 indices, Scalar scalar)
{
    using DstType = std::remove_cv_t<Scalar>;
    auto gmOffset = dst.GetLayout().template GetGmOffset<C, MAX_DIMS>(coordinate);
    auto dstAddr = reinterpret_cast<__gm__ DstType*>(dst.GetAddr()) + gmOffset;
    TscatterElementSInplaceImpl<axis, scatterMode>(dst, indices, scalar, dstAddr);
}

template <int axis, int scatterMode, typename T0, typename C, typename T1, typename T2, typename T3>
TILEOP void Tscatter(T0 dst, C coordinate, T1 src1, T2 src2, T3 tmp)
{
    static_assert(scatterMode == 0, "GM Scatter only supports overwrite");
    constexpr auto expectSize = MAX_DIMS;
    const auto dstLayout = dst.GetLayout();
    const auto idxLayout = src1.GetLayout();
    const auto srcLayout = src2.GetLayout();

    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();
    auto dstStride3 = dstLayout.template GetStrideDim<3, expectSize>();
    auto dstStride4 = dstLayout.template GetStrideDim<4, expectSize>();
    auto idxStride0 = idxLayout.template GetStrideDim<0, expectSize>();
    auto idxStride1 = idxLayout.template GetStrideDim<1, expectSize>();
    auto idxStride2 = idxLayout.template GetStrideDim<2, expectSize>();
    auto idxStride3 = idxLayout.template GetStrideDim<3, expectSize>();
    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();
    auto srcStride3 = srcLayout.template GetStrideDim<3, expectSize>();

    auto idxShape0 = idxLayout.template GetShapeDim<0, expectSize>();
    auto idxShape1 = idxLayout.template GetShapeDim<1, expectSize>();
    auto idxShape2 = idxLayout.template GetShapeDim<2, expectSize>();
    auto idxShape3 = idxLayout.template GetShapeDim<3, expectSize>();
    auto idxShape4 = idxLayout.template GetShapeDim<4, expectSize>();
    auto gmOffset = dstLayout.template GetGmOffset<C, expectSize>(coordinate);

    // GM TileTensor carries an address-space-qualified type; UB src has the scalar type.
    using DstType = typename T2::Type;
    using IdxType = typename T1::Type;
    using SrcType = typename T2::Type;
    auto dstAddr = reinterpret_cast<__gm__ DstType*>(dst.GetAddr()) + gmOffset;
    auto idxAddr = reinterpret_cast<__ubuf__ IdxType*>(src1.GetAddr());
    auto srcAddr = reinterpret_cast<__ubuf__ SrcType*>(src2.GetAddr());
    const uint32_t rows = idxShape0 * idxShape1 * idxShape2 * idxShape3;
    const ScatterGmTileInfo info{{idxShape1, idxShape2, idxShape3},
                                 {idxStride0, idxStride1, idxStride2, idxStride3},
                                 {srcStride0, srcStride1, srcStride2, srcStride3},
                                 {dstStride0, dstStride1, dstStride2, dstStride3, dstStride4}};

    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    if (rows != 0 && idxShape4 != 0) {
        const uint32_t warps = rows < 32 ? rows : 32;
        cce::async_invoke<ScatterGmSimt<axis, scatterMode, DstType, IdxType, SrcType>>(
            cce::dim3{32, warps}, dstAddr, idxAddr, srcAddr, info, rows, idxShape4);
    }
    dcci(static_cast<__gm__ void*>(0), cache_line_t::ENTIRE_DATA_CACHE);
    dsb(DSB_DDR);
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
}
#endif

#endif
