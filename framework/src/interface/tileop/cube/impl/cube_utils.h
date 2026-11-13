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
 * \file cube_utils.h
 * \brief Utility Functions and Constant Definitions
 */

#ifndef TILEOP_TILE_OPERATOR_CUBE_UTILS__H
#define TILEOP_TILE_OPERATOR_CUBE_UTILS__H

#include "utils/layout.h"
#include "utils/tile_tensor.h"

#ifdef __NPU_ARCH__
#if __NPU_ARCH__ == 3510
#define PTO_NPU_ARCH_A5
#endif
#endif

constexpr int16_t SHAPE_DIM2 = 2;
constexpr int16_t SHAPE_DIM3 = 3;
constexpr int16_t SHAPE_DIM4 = 4;
constexpr int16_t MX_SCALE_MAX_DIM = 5;
constexpr uint16_t BLOCK_CUBE_M_N = 16;
constexpr uint16_t BLOCK_ALIGN_BYTE = 32;
constexpr int64_t FP4_BLOCK_ALIGN_BYTE = 64;
constexpr int64_t MX_BLOCK_ALIGN_BYTE = 64;
// Bit 46 of the fixpipe deqScalar register selects signed (vs unsigned) int8 requantization.
constexpr uint64_t DEQ_SCALAR_INT8OUT_SIGN_BIT = 46;

template <CopyOutMode mode, bool isAcc, uint8_t reluMode>
struct TStoreConfig {
    static constexpr CopyOutMode kMode = mode;
    static constexpr bool kIsAcc = isAcc;
    static constexpr uint8_t kReluMode = reluMode;
};

template <int64_t idx, typename U>
INLINE int64_t GetShape(const U& tileTensor)
{
    static_assert(idx >= 0, "Idx should be non-negative");
    static_assert(idx < static_cast<int64_t>(Std::tuple_size<typename U::Shape>::value),
                  "Idx should be less than shape size");
    const auto tileLayout = tileTensor.GetLayout();
    return tileLayout.template GetShapeDim<idx>();
}

template <int64_t idx, typename U>
INLINE int64_t GetStride(const U& tileTensor)
{
    static_assert(idx >= 0, "Idx should be non-negative");
    static_assert(idx < static_cast<int64_t>(Std::tuple_size<typename U::Stride>::value),
                  "Idx should be less than stride size");
    const auto tileLayout = tileTensor.GetLayout();
    return tileLayout.template GetStrideDim<idx>();
}

template <int64_t fromTail, typename U>
INLINE int64_t GetTailShape(const U& tileTensor)
{
    constexpr auto shapeSize = Std::tuple_size<typename U::Shape>::value;
    static_assert(fromTail < static_cast<int64_t>(shapeSize), "fromTail should be less than shape size");
    return GetShape<static_cast<int64_t>(shapeSize - 1 - fromTail)>(tileTensor);
}

template <int64_t fromTail, typename U>
INLINE int64_t GetTailStride(const U& tileTensor)
{
    constexpr auto shapeSize = Std::tuple_size<typename U::Stride>::value;
    static_assert(fromTail < static_cast<int64_t>(shapeSize), "fromTail should be less than stride size");
    return GetStride<static_cast<int64_t>(shapeSize - 1 - fromTail)>(tileTensor);
}

// 折叠 batch 轴 offset：sum(coord[i] * stride[i])，i 属于 [0, dim-2)。
// 仅 TLoad/TStore 等 GM 搬运使用（offset 可能为非 0 的 batch 偏移）；
// L1/L0 内部搬运的前导 offset 由构图层保证为 0，不走此折叠。
// dim 上限 5：4D batch 的 MX scale tile 为 {1, 1, m, kScale, 2}
template <typename Coord, typename U>
INLINE int64_t GetBatchDimsOffset(const U& tileTensor, const Coord& coord)
{
    constexpr auto shapeSize = Std::tuple_size<typename U::Shape>::value;
    static_assert(
        shapeSize == SHAPE_DIM2 || shapeSize == SHAPE_DIM3 || shapeSize == SHAPE_DIM4 || shapeSize == MX_SCALE_MAX_DIM,
        "Shape dim size should be 2, 3, 4 or 5");
    int64_t batchOffset = 0;
    if constexpr (shapeSize == SHAPE_DIM3) {
        batchOffset = TileOp::GetTupleElement<Coord, 0, SHAPE_DIM3, 0>(coord) * GetStride<0>(tileTensor);
    } else if constexpr (shapeSize == SHAPE_DIM4) {
        batchOffset = TileOp::GetTupleElement<Coord, 0, SHAPE_DIM4, 0>(coord) * GetStride<0>(tileTensor) +
                      TileOp::GetTupleElement<Coord, 1, SHAPE_DIM4, 0>(coord) * GetStride<1>(tileTensor);
    } else if constexpr (shapeSize == MX_SCALE_MAX_DIM) {
        batchOffset = TileOp::GetTupleElement<Coord, 0, MX_SCALE_MAX_DIM, 0>(coord) * GetStride<0>(tileTensor) +
                      TileOp::GetTupleElement<Coord, 1, MX_SCALE_MAX_DIM, 0>(coord) * GetStride<1>(tileTensor) +
                      TileOp::GetTupleElement<Coord, SHAPE_DIM2, MX_SCALE_MAX_DIM, 0>(coord) * GetStride<2>(tileTensor);
    }
    return batchOffset;
}

INLINE int64_t CalNZOffset(const int64_t& srcShape0, const int64_t& srcShape1, const int64_t& offset0,
                           const int64_t& offset1, const int64_t& c0Size)
{
    int64_t batchSize = srcShape0 * srcShape1;
    int64_t offsetElem = offset1 + offset0 * srcShape1;
    int64_t batchIndex = offsetElem / batchSize;
    int64_t gmOffset = batchIndex * batchSize + (offset1 * srcShape0) + (offset0 - batchIndex * srcShape0) * c0Size;
    return gmOffset;
}

template <typename T>
constexpr INLINE bool CheckIsB4()
{
#if defined PTO_NPU_ARCH_A5
    return std::is_same<typename T::Type, float4_e2m1x2_t>::value ||
           std::is_same<typename T::Type, float4_e1m2x2_t>::value;
#else
    return false;
#endif
}

// 校验 tensor 所有维度 valid shape 非零
template <typename T>
INLINE bool CheckTensorDimsNonZero(const T& tensor)
{
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(
        shapeSize == SHAPE_DIM2 || shapeSize == SHAPE_DIM3 || shapeSize == SHAPE_DIM4 || shapeSize == MX_SCALE_MAX_DIM,
        "Shape dim size should be 2, 3, 4 or 5");
    if constexpr (shapeSize >= SHAPE_DIM3) {
        if (GetShape<SHAPE_DIM3 - 1>(tensor) == 0) {
            return false;
        }
    }
    if constexpr (shapeSize >= SHAPE_DIM4) {
        if (GetShape<SHAPE_DIM4 - 1>(tensor) == 0) {
            return false;
        }
    }
    if constexpr (shapeSize >= MX_SCALE_MAX_DIM) {
        if (GetShape<MX_SCALE_MAX_DIM - 1>(tensor) == 0) {
            return false;
        }
    }
    return GetShape<0>(tensor) != 0 && GetShape<1>(tensor) != 0;
}

template <typename T, typename U>
INLINE bool CheckShapeValid(const T& dst, const U& src)
{
    return CheckTensorDimsNonZero(dst) && CheckTensorDimsNonZero(src);
}

template <typename L0TileData>
INLINE bool CheckBASEMNValid(const L0TileData& l0Tile)
{
    int64_t l0Shape0 = GetShape<0>(l0Tile);
    int64_t l0Shape1 = GetShape<1>(l0Tile);
    if constexpr (L0TileData::FORMAT == Hardware::L0A) {
        return l0Shape0 > 0;
    } else {
        return l0Shape1 > 0;
    }
}

// 折叠 MX scale 的 batch 轴 offset：sum(coord[i] * stride[i])，i 属于 [0, dim-3)。
template <typename Coord, typename U>
INLINE int64_t GetBatchDimsOffsetMX(const U& tileTensor, const Coord& coord)
{
    constexpr auto shapeSize = Std::tuple_size<typename U::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM3 || shapeSize == SHAPE_DIM4 || shapeSize == MX_SCALE_MAX_DIM,
                  "Shape dim size should be 3, 4 or 5");
    int64_t batchOffset = 0;
    if constexpr (shapeSize == SHAPE_DIM4) {
        batchOffset = TileOp::GetTupleElement<Coord, 0, SHAPE_DIM4, 0>(coord) * GetStride<0>(tileTensor);
    } else if constexpr (shapeSize == MX_SCALE_MAX_DIM) {
        batchOffset = TileOp::GetTupleElement<Coord, 0, MX_SCALE_MAX_DIM, 0>(coord) * GetStride<0>(tileTensor) +
                      TileOp::GetTupleElement<Coord, 1, MX_SCALE_MAX_DIM, 0>(coord) * GetStride<1>(tileTensor);
    }
    return batchOffset;
}

// create Scale Tile Data
template <typename FpTileData>
INLINE auto CreateScaleTileData(FpTileData& fixbuf)
{
    constexpr uint64_t shapeSize = Std::tuple_size<typename FpTileData::Shape>::value;
    constexpr int64_t
        scaleTileH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename FpTileData::TileShape>::type::value;
    constexpr int64_t scaleTileW = Std::tuple_element<shapeSize - 1, typename FpTileData::TileShape>::type::value;
    int64_t scaleShape0 = GetTailShape<1>(fixbuf);
    int64_t scaleShape1 = GetTailShape<0>(fixbuf);
    using scaleTileData = pto::Tile<pto::TileType::Scaling, uint64_t, scaleTileH, scaleTileW, pto::BLayout::RowMajor,
                                    -1, -1>;
    return scaleTileData(scaleShape0, scaleShape1);
}

// Type compatibility check for quantization mode
template <typename l0cDType, typename ubDType>
constexpr INLINE bool IsSupportedQuantMode()
{
    return (std::is_same<l0cDType, int32_t>::value && std::is_same<ubDType, half>::value) ||
           std::is_same<ubDType, int8_t>::value;
}

// Type compatibility check for basic mode
template <typename l0cDType, typename ubDType>
constexpr INLINE bool IsSupportedBasicMode()
{
    return (std::is_same<l0cDType, float>::value && std::is_same<ubDType, half>::value) ||
           (std::is_same<l0cDType, float>::value && std::is_same<ubDType, bfloat16_t>::value);
}

// Handle K=0 scenario: fill L1 with zero and return early
// Returns true if early return is needed, false otherwise
template <CopyInMode copyMode, int64_t kIndex, typename TileData>
INLINE bool HandleZeroKScenario(TileData& dst)
{
    if constexpr (kIndex != -1) {
        static_assert(kIndex >= 0 && kIndex <= SHAPE_DIM4 - 1, "K index should be in [0, SHAPE_DIM4 - 1]");
        if constexpr (copyMode != CopyInMode::ND2ND) {
            int64_t validK = GetShape<kIndex>(dst);
            if (validK == 0) {
                TFillPadZeroK(dst);
                return true; // Signal that early return is needed
            }
        }
    }
    return false; // No early return needed
}

#endif // TILEOP_TILE_OPERATOR_CUBE_UTILS__H
