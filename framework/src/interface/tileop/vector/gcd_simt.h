/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TILEOP_TILE_OPERATOR_VEC_GCD_SIMT_H
#define TILEOP_TILE_OPERATOR_VEC_GCD_SIMT_H

#if defined(__DAV_V310)
#include "utils/layout.h"
#include "utils/tile_tensor.h"
#include "binary_brcinline.h"

template <typename T>
__simt_callee__ AICORE inline T GcdSimtValue(T lhs, T rhs)
{
    const int32_t x = static_cast<int32_t>(lhs);
    const int32_t y = static_cast<int32_t>(rhs);
    uint32_t a = x < 0 ? 0u - static_cast<uint32_t>(x) : static_cast<uint32_t>(x);
    uint32_t b = y < 0 ? 0u - static_cast<uint32_t>(y) : static_cast<uint32_t>(y);
    if (a == 0) {
        return static_cast<T>(b);
    }
    if (b == 0) {
        return static_cast<T>(a);
    }

    const uint8_t shift = __builtin_ffs(a | b) - 1;
    a >>= __builtin_ffs(a) - 1;
    b >>= __builtin_ffs(b) - 1;
    while (b != 0) {
        b >>= __builtin_ffs(b) - 1;
        if (a > b) {
            const uint32_t temp = a;
            a = b;
            b = temp;
        }
        b -= a;
    }
    // The magnitude of the signed minimum value wraps on conversion to the result dtype.
    return static_cast<T>(a << shift);
}

template <typename T, bool scalarMode>
__simt_vf__ AICORE LAUNCH_BOUND(1024) inline void GcdTileSimt(__ubuf__ T* dst, __ubuf__ const T* src0,
                                                              __ubuf__ const T* src1, T scalar, uint32_t rows,
                                                              uint32_t cols, uint32_t dstRowStride,
                                                              uint32_t dstColStride, uint32_t src0RowStride,
                                                              uint32_t src0ColStride, uint32_t src1RowStride,
                                                              uint32_t src1ColStride)
{
    const uint32_t count = rows * cols;
    for (uint32_t index = __cce_simt_get_TID_X(); index < count; index += 1024) {
        const uint32_t row = index / cols;
        const uint32_t col = index % cols;
        const T rhs = scalarMode ? scalar : src1[row * src1RowStride + col * src1ColStride];
        dst[row * dstRowStride + col * dstColStride] = GcdSimtValue(src0[row * src0RowStride + col * src0ColStride],
                                                                    rhs);
    }
}

template <int... BrcOperands, typename T0, typename T1, typename T2>
TILEOP void TGcdSimt(T0 dst, T1 src0, T2 src1)
{
    const auto dstLayout = dst.GetLayout();
    const auto src0Layout = src0.GetLayout();
    const auto src1Layout = src1.GetLayout();
    const auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    const auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    const auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    const auto rows = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    const auto cols = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    if (shape0 == 0 || shape1 == 0 || shape2 == 0 || rows == 0 || cols == 0) {
        return;
    }
    using Src0Info = TensorTileInfo<T1>;
    using Src1Info = TensorTileInfo<T2>;
    using Type = typename T0::Type;
    const uint32_t dstRowStride = dstLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();
    const uint32_t dstColStride = dstLayout.template GetStrideDim<DIM_5TH, MAX_DIMS>();
    const uint32_t src0RowStride = (src0Layout.template GetShapeDim<DIM_4TH, MAX_DIMS>() == 1 ||
                                    GetBrcOperandAt<DIM_4TH, BrcOperands...>() == BRC_LEFT) ?
                                       0 :
                                       src0Layout.template GetStrideDim<DIM_4TH, MAX_DIMS>();
    const uint32_t src0ColStride = (src0Layout.template GetShapeDim<DIM_5TH, MAX_DIMS>() == 1 ||
                                    GetBrcOperandAt<DIM_5TH, BrcOperands...>() == BRC_LEFT) ?
                                       0 :
                                       src0Layout.template GetStrideDim<DIM_5TH, MAX_DIMS>();
    const uint32_t src1RowStride = (src1Layout.template GetShapeDim<DIM_4TH, MAX_DIMS>() == 1 ||
                                    GetBrcOperandAt<DIM_4TH, BrcOperands...>() == BRC_RIGHT) ?
                                       0 :
                                       src1Layout.template GetStrideDim<DIM_4TH, MAX_DIMS>();
    const uint32_t src1ColStride = (src1Layout.template GetShapeDim<DIM_5TH, MAX_DIMS>() == 1 ||
                                    GetBrcOperandAt<DIM_5TH, BrcOperands...>() == BRC_RIGHT) ?
                                       0 :
                                       src1Layout.template GetStrideDim<DIM_5TH, MAX_DIMS>();
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    for (LoopVar i = 0; i < shape0; ++i) {
        for (LoopVar j = 0; j < shape1; ++j) {
            for (LoopVar k = 0; k < shape2; ++k) {
                const auto dstOffset = GenTileOffset(dst, TileOffset(i, j, k));
                const auto src0Offset = GenTileOffset(
                    src0,
                    TileOffset(
                        (Src0Info::tile0 == 1 || GetBrcOperandAt<DIM_1ST, BrcOperands...>() == BRC_LEFT) ? 0 : i,
                        (Src0Info::tile1 == 1 || GetBrcOperandAt<DIM_2ND, BrcOperands...>() == BRC_LEFT) ? 0 : j,
                        (Src0Info::tile2 == 1 || GetBrcOperandAt<DIM_3RD, BrcOperands...>() == BRC_LEFT) ? 0 : k));
                const auto src1Offset = GenTileOffset(
                    src1,
                    TileOffset(
                        (Src1Info::tile0 == 1 || GetBrcOperandAt<DIM_1ST, BrcOperands...>() == BRC_RIGHT) ? 0 : i,
                        (Src1Info::tile1 == 1 || GetBrcOperandAt<DIM_2ND, BrcOperands...>() == BRC_RIGHT) ? 0 : j,
                        (Src1Info::tile2 == 1 || GetBrcOperandAt<DIM_3RD, BrcOperands...>() == BRC_RIGHT) ? 0 : k));
                cce::async_invoke<GcdTileSimt<Type, false>>(
                    cce::dim3{1024}, reinterpret_cast<__ubuf__ Type*>(dst.GetAddr()) + dstOffset,
                    reinterpret_cast<__ubuf__ const Type*>(src0.GetAddr()) + src0Offset,
                    reinterpret_cast<__ubuf__ const Type*>(src1.GetAddr()) + src1Offset, Type{}, rows, cols,
                    dstRowStride, dstColStride, src0RowStride, src0ColStride, src1RowStride, src1ColStride);
            }
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
}

template <typename Scalar, typename T0, typename T1>
TILEOP void TGcdScalarSimt(T0 dst, T1 src, Scalar scalar)
{
    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    const auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    const auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    const auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    const auto rows = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    const auto cols = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    if (shape0 == 0 || shape1 == 0 || shape2 == 0 || rows == 0 || cols == 0) {
        return;
    }
    using Type = typename T0::Type;
    const uint32_t dstRowStride = dstLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();
    const uint32_t dstColStride = dstLayout.template GetStrideDim<DIM_5TH, MAX_DIMS>();
    const uint32_t srcRowStride = srcLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();
    const uint32_t srcColStride = srcLayout.template GetStrideDim<DIM_5TH, MAX_DIMS>();
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    for (LoopVar i = 0; i < shape0; ++i) {
        for (LoopVar j = 0; j < shape1; ++j) {
            for (LoopVar k = 0; k < shape2; ++k) {
                const auto offsets = TileOffset(i, j, k);
                const auto dstOffset = GenTileOffset(dst, offsets);
                const auto srcOffset = GenTileOffset(src, offsets);
                cce::async_invoke<GcdTileSimt<Type, true>>(
                    cce::dim3{1024}, reinterpret_cast<__ubuf__ Type*>(dst.GetAddr()) + dstOffset,
                    reinterpret_cast<__ubuf__ const Type*>(src.GetAddr()) + srcOffset,
                    reinterpret_cast<__ubuf__ const Type*>(src.GetAddr()) + srcOffset, static_cast<Type>(scalar), rows,
                    cols, dstRowStride, dstColStride, srcRowStride, srcColStride, 0, 0);
            }
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
}
#endif
#endif // TILEOP_TILE_OPERATOR_VEC_GCD_SIMT_H
