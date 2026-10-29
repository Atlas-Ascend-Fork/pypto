/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file convbp_vec_tile_inference.cpp
 * \brief Infer VecTileShape for ConvBp dW (ConvBackwardFilter) TransData from op attributes.
 */

#include "convbp_vec_tile_inference.h"
#include "interface/operation/operation.h"
#include "interface/operation/operation_impl.h"
#include "interface/utils/common.h"
#include "tilefwk/platform.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu {
namespace tile_fwk {
namespace ConvBp {

namespace {
constexpr int64_t MKN_N_VAL = 16;

inline size_t GetUbLimit()
{
    size_t ubSize = Platform::Instance().GetAIVCore().GetMemorySize(MemoryType::MEM_UB);
    return ubSize / 4;
}

inline int64_t VecTileBytes(const std::vector<int64_t>& tile, int64_t dtypeSize)
{
    int64_t total = dtypeSize;
    for (int64_t v : tile) {
        total *= v;
    }
    return total;
}

inline void GrowDim(std::vector<int64_t>& tile, size_t dimIdx, int64_t step, int64_t maxVal, int64_t dtypeSize,
                    size_t ubLimit)
{
    while (tile[dimIdx] + step <= maxVal) {
        std::vector<int64_t> tmp = tile;
        tmp[dimIdx] += step;
        if (VecTileBytes(tmp, dtypeSize) > static_cast<int64_t>(ubLimit)) {
            break;
        }
        tile = tmp;
    }
}

VecTile InferBpNchwToNzVecTile(const std::vector<int64_t>& nchwShape, int64_t c0, int64_t dtypeSize, size_t ubLimit)
{
    VecTile vecTile;
    vecTile.tile = {1, c0, 1, c0};
    GrowDim(vecTile.tile, 3, c0, nchwShape[3], dtypeSize, ubLimit);
    GrowDim(vecTile.tile, 2, 1, nchwShape[2], dtypeSize, ubLimit);
    GrowDim(vecTile.tile, 1, c0, nchwShape[1], dtypeSize, ubLimit);
    return vecTile;
}

VecTile InferBpFzToNchwVecTile(int64_t n1Full, int64_t dstW, int64_t n0, int64_t c0, int64_t dtypeSize)
{
    if (n1Full <= 0 || dstW <= 0 || n0 <= 0 || c0 <= 0) {
        return VecTile{};
    }
    int64_t wPad = (dstW + c0 - 1) / c0 * c0;
    int64_t yTileSizeElem = (dtypeSize == 1) ? 32 : 16;
    int64_t ubSize = static_cast<int64_t>(Platform::Instance().GetAIVCore().GetMemorySize(MemoryType::MEM_UB));
    int64_t perN1Bytes = 3 * wPad * n0 * c0 * dtypeSize;
    int64_t fixedBytes = c0 * ((wPad + yTileSizeElem - 1) / yTileSizeElem * yTileSizeElem) * dtypeSize;
    int64_t n1Cap = 1;
    if (perN1Bytes > 0 && ubSize > fixedBytes) {
        n1Cap = std::max<int64_t>(1, (ubSize - fixedBytes) / perN1Bytes);
    }
    VecTile vecTile;
    vecTile.tile = {dstW, std::min(n1Full, n1Cap), n0, c0};
    return vecTile;
}
} // namespace

ConvBpVecTileShapes InferConvBpVecTileShapes(const Operation& convBpOp, DataType dtype)
{
    ConvBpVecTileShapes vecTiles;
    int64_t dtypeSize = static_cast<int64_t>(BytesOf(dtype));
    auto gradShape = convBpOp.GetVectorIntAttribute(CONV_ORI_GRAD_OUTPUT_SHAPE_ATTR);
    auto resShape = convBpOp.GetVectorIntAttribute(CONV_ORI_RES_SHAPE_ATTR);
    int64_t c0 = ALIGN_SIZE_32 / dtypeSize;
    int64_t n0 = MKN_N_VAL;
    int64_t groups = convBpOp.HasAttr(CONV_GROUPS_ATTR) ? convBpOp.GetIntAttribute(CONV_GROUPS_ATTR) : 1;
    int64_t coutPerGroup = groups > 0 ? resShape[0] / groups : resShape[0];
    int64_t n1Full = (coutPerGroup + n0 - 1) / n0;

    vecTiles.inVecTile = InferBpNchwToNzVecTile(gradShape, c0, dtypeSize, GetUbLimit());
    vecTiles.outVecTile = InferBpFzToNchwVecTile(n1Full, resShape[NCHW_W_IDX], n0, c0, dtypeSize);
    return vecTiles;
}

VecTile SelectConvBpVecTile(const ConvBpVecTileShapes& vecTiles, TileOpFormat targetFormat)
{
    if (targetFormat == TileOpFormat::TILEOP_ND) {
        CONV_LOGI("ConvBp transdata vec tile for output after infer is: %s",
                  IntVecToStr(vecTiles.outVecTile.tile).c_str());
        return vecTiles.outVecTile;
    }
    if (targetFormat == TileOpFormat::TILEOP_NC1HWC0) {
        CONV_LOGI("ConvBp transdata vec tile for input after infer is: %s",
                  IntVecToStr(vecTiles.inVecTile.tile).c_str());
        return vecTiles.inVecTile;
    }
    return VecTile{};
}

} // namespace ConvBp
} // namespace tile_fwk
} // namespace npu
