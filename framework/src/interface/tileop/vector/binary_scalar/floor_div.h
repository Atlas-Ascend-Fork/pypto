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
 * \file floor_div.h
 * \brief Binary-scalar floor division tile operation implementation.
 */

#ifndef TILEOP_TILE_OPERATOR_VEC_BINARY_SCALAR_FLOOR_DIV_H
#define TILEOP_TILE_OPERATOR_VEC_BINARY_SCALAR_FLOOR_DIV_H

#include "utils/sync.h"
#include "../pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"
#include "../binary/floor_div.h"

template <typename T0, typename Scalar, typename T2, typename DstTile, typename SrcTile, typename Offset,
          typename Shape>
TILEOP void FloorDivSFloatingCompute(DstTile& dstTile, SrcTile& src0Tile, Scalar src1, T2 tmp, Offset dstOffset,
                                     size_t tileShapeSize, Shape dstShape3, Shape dstShape4)
{
    constexpr auto tileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    if constexpr (std::is_same_v<typename T0::Type, half> || std::is_same_v<typename T0::Type, bfloat16_t>) {
        using Fp32TileDefine = pto::Tile<pto::TileType::Vec, float, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
        Fp32TileDefine tmp0Tile(dstShape3, dstShape4);
        pto::TASSIGN(tmp0Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 0, sizeof(float)));
        pto::TCVT(tmp0Tile, src0Tile, pto::RoundMode::CAST_NONE);
        SyncV();
        pto::TDIVS<pto::DivAlgorithm::HIGH_PRECISION>(tmp0Tile, tmp0Tile, static_cast<float>(src1));
        SyncV();
        pto::TCVT(tmp0Tile, tmp0Tile, pto::RoundMode::CAST_FLOOR);
        SyncV();
        pto::TCVT(dstTile, tmp0Tile, pto::RoundMode::CAST_RINT);
        SyncV();
    } else if constexpr (std::is_same_v<typename T0::Type, float>) {
        pto::TDIVS<pto::DivAlgorithm::HIGH_PRECISION>(dstTile, src0Tile, static_cast<float>(src1));
        SyncV();
        pto::TCVT(dstTile, dstTile, pto::RoundMode::CAST_FLOOR);
        SyncV();
    }
}

#ifdef __DAV_V220
template <typename T0, typename Scalar, typename T2, typename DstTile, typename SrcTile, typename Offset,
          typename Shape>
TILEOP void FloorDivSV220Int32Compute(DstTile& dstTile, SrcTile& src0Tile, Scalar src1, T2 tmp, Offset dstOffset,
                                      size_t tileShapeSize, Shape dstShape3, Shape dstShape4)
{
    constexpr auto tileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    if constexpr (std::is_same_v<typename T0::Type, int32_t>) {
        using Fp32TileDefine = pto::Tile<pto::TileType::Vec, float, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
        using Int32TileDefine = pto::Tile<pto::TileType::Vec, int32_t, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
        using MaskTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, 4 * tileW, pto::BLayout::RowMajor, -1, -1>;

        Fp32TileDefine tmp0Fp32Tile(dstShape3, dstShape4);
        Fp32TileDefine tmp2Fp32Tile(dstShape3, dstShape4);
        Int32TileDefine tmp0I32Tile(dstShape3, dstShape4);
        Int32TileDefine tmp2I32Tile(dstShape3, dstShape4);
        Int32TileDefine tmp3I32Tile(dstShape3, dstShape4);
        Int32TileDefine tmp4I32Tile(dstShape3, dstShape4);
        Int32TileDefine tmp5I32Tile(dstShape3, dstShape4);
        MaskTileDefine tmp1MaskTile(dstShape3, dstShape4);

        pto::TASSIGN(tmp0Fp32Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 0, sizeof(int32_t)));
        pto::TASSIGN(tmp2Fp32Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 2, sizeof(int32_t)));
        pto::TASSIGN(tmp0I32Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 0, sizeof(int32_t)));
        pto::TASSIGN(tmp2I32Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 2, sizeof(int32_t)));
        pto::TASSIGN(tmp3I32Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 3, sizeof(int32_t)));
        pto::TASSIGN(tmp4I32Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 4, sizeof(int32_t)));
        pto::TASSIGN(tmp5I32Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 5, sizeof(int32_t)));
        pto::TASSIGN(tmp1MaskTile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 1, sizeof(int32_t)));
        auto divisor = static_cast<int32_t>(src1);

        // Step 1: approximate quotient by float32 division, then floor and cast to int32.
        // q = floor(float32(x1) / float32(x2))
        pto::TCVT(tmp0Fp32Tile, src0Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
        SyncV();
        pto::TDIVS<pto::DivAlgorithm::HIGH_PRECISION>(tmp0Fp32Tile, tmp0Fp32Tile, static_cast<float>(divisor));
        SyncV();
        pto::TCVT(dstTile, tmp0Fp32Tile, pto::RoundMode::CAST_FLOOR);
        SyncV();

        // Step 2: compute exact int32 remainder: r = x1 - q * x2.
        pto::TMULS(tmp0I32Tile, dstTile, divisor);
        SyncV();
        pto::TSUB(tmp0I32Tile, src0Tile, tmp0I32Tile);
        SyncV();

        // Step 3: refine q with floor(float32(r) / float32(x2)).
        pto::TCVT(tmp2Fp32Tile, tmp0I32Tile, pto::RoundMode::CAST_NONE);
        SyncV();
        pto::TDIVS<pto::DivAlgorithm::HIGH_PRECISION>(tmp2Fp32Tile, tmp2Fp32Tile, static_cast<float>(divisor));
        SyncV();
        pto::TCVT(tmp0I32Tile, tmp2Fp32Tile, pto::RoundMode::CAST_FLOOR);
        SyncV();

        // Step 4: apply the remainder-based correction.
        // q_corrected = q + correction
        pto::TADD(dstTile, dstTile, tmp0I32Tile);
        SyncV();

        // Step 5: recompute r2 with q_corrected.
        pto::TMULS(tmp0I32Tile, dstTile, divisor);
        SyncV();
        pto::TSUB(tmp0I32Tile, src0Tile, tmp0I32Tile); // r2
        SyncV();

        // Step 6: final +/-1 correction. A valid floor-div remainder must satisfy
        // 0 <= r2 * sign(x2) < abs(x2).
        auto absSrc1 = divisor;
        if (divisor < 0) {
            pto::TMULS(tmp0I32Tile, tmp0I32Tile, -1); // r2_adj = -r2
            SyncV();
            absSrc1 = -divisor;
        }

        pto::TADDS(tmp3I32Tile, tmp0I32Tile, -absSrc1); // diff = r2_adj - abs(x2)
        SyncV();

        // Build tensor constants and use TSEL instead of TSELS to avoid the A2/A3
        // tensor-scalar select path, whose first lane can be unstable across calls.
        pto::TSUB(tmp4I32Tile, tmp0I32Tile, tmp0I32Tile); // zero
        SyncV();

        // If r2_adj < 0, q_corrected is too large: final_corr = -1; otherwise 0.
        pto::TCVT(tmp2Fp32Tile, tmp0I32Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
        SyncV();
        pto::TCMPS(tmp1MaskTile, tmp2Fp32Tile, 0.0f, pto::CmpMode::LT);
        SyncV();
        pto::TADDS(tmp2I32Tile, tmp4I32Tile, -1);
        SyncV();
        pto::TSEL(tmp0I32Tile, tmp1MaskTile, tmp2I32Tile, tmp4I32Tile, tmp5I32Tile);
        SyncV();

        // If diff >= 0, q_corrected is too small: final_corr = 1.
        pto::TCVT(tmp2Fp32Tile, tmp3I32Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
        SyncV();
        pto::TCMPS(tmp1MaskTile, tmp2Fp32Tile, 0.0f, pto::CmpMode::GE);
        SyncV();
        pto::TADDS(tmp2I32Tile, tmp4I32Tile, 1);
        SyncV();
        pto::TSEL(tmp0I32Tile, tmp1MaskTile, tmp2I32Tile, tmp0I32Tile, tmp5I32Tile);
        SyncV();

        // res = q_corrected + final_corr
        pto::TADD(dstTile, dstTile, tmp0I32Tile);
        SyncV();
    }
}

template <typename T0, typename Scalar, typename T2, typename DstTile, typename SrcTile, typename Offset,
          typename Shape>
TILEOP void FloorDivSV220Int8Compute(DstTile& dstTile, SrcTile& src0Tile, Scalar src1, T2 tmp, Offset dstOffset,
                                     size_t tileShapeSize, Shape dstShape3, Shape dstShape4)
{
    constexpr auto tileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    if constexpr (std::is_same_v<typename T0::Type, int8_t> || std::is_same_v<typename T0::Type, uint8_t>) {
        using HalfTileDefine = pto::Tile<pto::TileType::Vec, half, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
        using Fp32TileDefine = pto::Tile<pto::TileType::Vec, float, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
        HalfTileDefine tmp0Tile(dstShape3, dstShape4);
        Fp32TileDefine tmp1Tile(dstShape3, dstShape4);
        pto::TASSIGN(tmp0Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 0, sizeof(float)));
        pto::TASSIGN(tmp1Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 1, sizeof(float)));
        pto::TCVT(tmp0Tile, src0Tile, pto::RoundMode::CAST_NONE);
        SyncV();
        pto::TCVT(tmp1Tile, tmp0Tile, pto::RoundMode::CAST_NONE);
        SyncV();
        pto::TDIVS<pto::DivAlgorithm::HIGH_PRECISION>(tmp1Tile, tmp1Tile, static_cast<float>(src1));
        SyncV();
        pto::TCVT(tmp0Tile, tmp1Tile, pto::RoundMode::CAST_FLOOR);
        SyncV();
        pto::TCVT(dstTile, tmp0Tile, pto::RoundMode::CAST_FLOOR, pto::SaturationMode::ON);
        SyncV();
    }
}
#else
template <typename T0, typename Scalar, typename T2, typename DstTile, typename SrcTile, typename Offset,
          typename Shape>
TILEOP void FloorDivSLegacyIntegerCompute(DstTile& dstTile, SrcTile& src0Tile, Scalar src1, T2 tmp, Offset dstOffset,
                                          size_t tileShapeSize, Shape dstShape3, Shape dstShape4)
{
    constexpr auto tileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    if constexpr (std::is_same_v<typename T0::Type, uint8_t>) {
        using HalfTileDefine = pto::Tile<pto::TileType::Vec, half, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
        using Int16TileDefine = pto::Tile<pto::TileType::Vec, int16_t, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
        HalfTileDefine tmp0Tile(dstShape3, dstShape4);
        Int16TileDefine tmp1Tile(dstShape3, dstShape4);
        pto::TASSIGN(tmp0Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 0, sizeof(float)));
        pto::TASSIGN(tmp1Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 1, sizeof(float)));
        pto::TCVT(tmp0Tile, src0Tile, pto::RoundMode::CAST_NONE);
        pto::TCVT(tmp1Tile, tmp0Tile, pto::RoundMode::CAST_NONE);
        pto::TDIVS(tmp1Tile, tmp1Tile, static_cast<int16_t>(src1));
        pto::TCVT(dstTile, tmp1Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::ON);
    } else if constexpr (std::is_same_v<typename T0::Type, int8_t>) {
        using HalfTileDefine = pto::Tile<pto::TileType::Vec, half, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
        HalfTileDefine tmp0Tile(dstShape3, dstShape4);
        pto::TASSIGN(tmp0Tile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 0, sizeof(float)));
        pto::TCVT(tmp0Tile, src0Tile, pto::RoundMode::CAST_NONE);
        if (src1 == 0) {
            pto::TEXPANDS(tmp0Tile, static_cast<half>(0.0f));
        } else {
            pto::TDIVS<pto::DivAlgorithm::HIGH_PRECISION>(tmp0Tile, tmp0Tile,
                                                          static_cast<half>(static_cast<float>(src1)));
        }
        pto::TCVT(dstTile, tmp0Tile, pto::RoundMode::CAST_FLOOR);
    } else if constexpr (std::is_same_v<typename T0::Type, int32_t>) {
        using Int32TileDefine = pto::Tile<pto::TileType::Vec, int32_t, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
        using MaskTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, 4 * tileW, pto::BLayout::RowMajor, -1, -1>;
        Int32TileDefine tmp0DataTile(dstShape3, dstShape4);
        Int32TileDefine tmp1DataTile(dstShape3, dstShape4);
        MaskTileDefine tmp2MaskTile(dstShape3, dstShape4);
        MaskTileDefine tmp3MaskTile(dstShape3, dstShape4);
        pto::TASSIGN(tmp0DataTile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 0, sizeof(float)));
        pto::TASSIGN(tmp1DataTile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 1, sizeof(float)));
        pto::TASSIGN(tmp2MaskTile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 2, sizeof(float)));
        pto::TASSIGN(tmp3MaskTile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 1, sizeof(float)));

        if (src1 == 0) {
            pto::TCMPS(tmp2MaskTile, src0Tile, 0, pto::CmpMode::LT);
            pto::TSELS(dstTile, tmp2MaskTile, dstTile, tmp0DataTile, INT32_NEGATIVE_DIVIDEND_ZERO_DIVISOR_RESULT);
            pto::TCMPS(tmp2MaskTile, src0Tile, 0, pto::CmpMode::GE);
            pto::TSELS(dstTile, tmp2MaskTile, dstTile, tmp0DataTile, INT32_NONNEGATIVE_DIVIDEND_ZERO_DIVISOR_RESULT);
        } else {
            if (src1 < 0) {
                pto::TCMPS(tmp2MaskTile, src0Tile, 0, pto::CmpMode::GE);
            } else {
                pto::TCMPS(tmp2MaskTile, src0Tile, 0, pto::CmpMode::LT);
            }
            pto::TDIVS(dstTile, src0Tile, static_cast<int32_t>(src1));
            pto::TMULS(tmp0DataTile, dstTile, static_cast<int32_t>(src1));
            pto::TSUB(tmp0DataTile, src0Tile, tmp0DataTile);
            pto::TCMPS(tmp3MaskTile, tmp0DataTile, 0, pto::CmpMode::NE);
            pto::TAND(tmp2MaskTile, tmp2MaskTile, tmp3MaskTile);
            pto::TADDS(tmp1DataTile, dstTile, -1);
            pto::TSEL(dstTile, tmp2MaskTile, tmp1DataTile, dstTile, tmp0DataTile);
        }
    } else if constexpr (std::is_same_v<typename T0::Type, int64_t>) {
        using Int64TileDefine = pto::Tile<pto::TileType::Vec, int64_t, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
        using MaskTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, 8 * tileW, pto::BLayout::RowMajor, -1, -1>;
        Int64TileDefine tmp0DataTile(dstShape3, dstShape4);
        Int64TileDefine tmp1DataTile(dstShape3, dstShape4);
        MaskTileDefine tmp2MaskTile(dstShape3, dstShape4);
        MaskTileDefine tmp3MaskTile(dstShape3, dstShape4);
        pto::TASSIGN(tmp0DataTile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 0, sizeof(int64_t)));
        pto::TASSIGN(tmp1DataTile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 1, sizeof(int64_t)));
        pto::TASSIGN(tmp2MaskTile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 2, sizeof(int64_t)));
        pto::TASSIGN(tmp3MaskTile, FloorDivTmpAddr(tmp, dstOffset, tileShapeSize, 3, sizeof(int64_t)));

        if (src1 == 0) {
            pto::TCMPS(tmp2MaskTile, src0Tile, 0, pto::CmpMode::LT);
            pto::TSELS(dstTile, tmp2MaskTile, dstTile, tmp0DataTile, INT64_NEGATIVE_DIVIDEND_ZERO_DIVISOR_RESULT);
            pto::TCMPS(tmp2MaskTile, src0Tile, 0, pto::CmpMode::GE);
            pto::TSELS(dstTile, tmp2MaskTile, dstTile, tmp0DataTile, INT64_NONNEGATIVE_DIVIDEND_ZERO_DIVISOR_RESULT);
        } else {
            if (src1 < 0) {
                pto::TCMPS(tmp2MaskTile, src0Tile, 0, pto::CmpMode::GE);
            } else {
                pto::TCMPS(tmp2MaskTile, src0Tile, 0, pto::CmpMode::LT);
            }
            pto::TDIVS(dstTile, src0Tile, static_cast<int64_t>(src1));
            pto::TMULS(tmp0DataTile, dstTile, static_cast<int64_t>(src1));
            pto::TSUB(tmp0DataTile, src0Tile, tmp0DataTile);
            pto::TCMPS(tmp3MaskTile, tmp0DataTile, 0, pto::CmpMode::NE);
            pto::TAND(tmp2MaskTile, tmp2MaskTile, tmp3MaskTile);
            pto::TADDS(tmp1DataTile, dstTile, -1);
            pto::TSEL(dstTile, tmp2MaskTile, tmp1DataTile, dstTile, tmp0DataTile);
        }
    }
}
#endif

#define OP_TILE_OP_FLOORDIVS TFloorDivS
template <typename Scalar, typename T0, typename T1, typename T2>
TILEOP void TFloorDivS(T0 dst, T1 src0, Scalar src1, T2 tmp)
{
    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    if (dstShape0 == 0 || dstShape1 == 0 || dstShape2 == 0 || dstShape3 == 0 || dstShape4 == 0) {
        return;
    }

    constexpr auto tileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr size_t FLOOR_DIV_TMP_PLANE_COUNT = 6;
    constexpr auto tmpPlaneSize = TileOp::GetAnyAxisMergeResult<DIM_1ST, Std::tuple_size<typename T2::TileShape>::value,
                                                                typename T2::TileShape>() /
                                  FLOOR_DIV_TMP_PLANE_COUNT;
    using DataTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, tileH, tileW, pto::BLayout::RowMajor, -1,
                                     -1>;
    auto src0Tile = MakeElementwiseOperandExecTile(dst, src0);
    DataTileDefine dstTile(dstShape3, dstShape4);

    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto srcOffset = GenTileOffset(src0, tileOffsets);
                auto dstOffset = GenTileOffset(dst, tileOffsets);
                auto tmpByteOffset = TileOp::GetPackedByteOffset<typename T2::Type>(GenTileOffset(tmp, tileOffsets));
                pto::TASSIGN(src0Tile, (uint64_t)(src0.GetAddr() + srcOffset * dstTypeSize));
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                FloorDivSFloatingCompute<T0>(dstTile, src0Tile, src1, tmp, tmpByteOffset, tmpPlaneSize, dstShape3,
                                             dstShape4);
#ifdef __DAV_V220
                FloorDivSV220Int32Compute<T0>(dstTile, src0Tile, src1, tmp, tmpByteOffset, tmpPlaneSize, dstShape3,
                                              dstShape4);
                FloorDivSV220Int8Compute<T0>(dstTile, src0Tile, src1, tmp, tmpByteOffset, tmpPlaneSize, dstShape3,
                                             dstShape4);
#else
                FloorDivSLegacyIntegerCompute<T0>(dstTile, src0Tile, src1, tmp, tmpByteOffset, tmpPlaneSize, dstShape3,
                                                  dstShape4);
#endif
            }
        }
    }
}

#endif // TILEOP_TILE_OPERATOR_VEC_BINARY_SCALAR_FLOOR_DIV_H
