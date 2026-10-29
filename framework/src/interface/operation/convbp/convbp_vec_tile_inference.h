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
 * \file convbp_vec_tile_inference.h
 * \brief Infer VecTileShape for ConvBp dW (ConvBackwardFilter) TransData from op attributes.
 */

#ifndef FRAMEWORK_SRC_INTERFACE_OPERATION_CONVBP_CONVBP_VEC_TILE_INFERENCE_H
#define FRAMEWORK_SRC_INTERFACE_OPERATION_CONVBP_CONVBP_VEC_TILE_INFERENCE_H

#include "tilefwk/tile_shape.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/data_type.h"

namespace npu {
namespace tile_fwk {

class Operation;

namespace ConvBp {

struct ConvBpVecTileShapes {
    VecTile inVecTile;
    VecTile outVecTile;
};

ConvBpVecTileShapes InferConvBpVecTileShapes(const Operation& convBpOp, DataType dtype);

VecTile SelectConvBpVecTile(const ConvBpVecTileShapes& vecTiles, TileOpFormat targetFormat);

} // namespace ConvBp
} // namespace tile_fwk
} // namespace npu

#endif // FRAMEWORK_SRC_INTERFACE_OPERATION_CONVBP_CONVBP_VEC_TILE_INFERENCE_H
