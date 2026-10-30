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
 * \file digamma.cpp
 * \brief
 */

#include "unary_tiled.h"
#include "binary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

Tensor Digamma(const Tensor& self)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Digamma");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Digamma");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "Digamma");

    auto castSelf = Cast(self, DataType::DT_FP32);
    auto result = CALL(UnaryOperation<UnaryOpType::DIGAMMA>, *Program::GetInstance().GetCurrentFunction(),
                       castSelf.GetStorage());
    auto castResult = Cast(result, self.GetDataType());
    return castResult;
}

void DigammaOperationTileFunc(Function& function, const TileShape& tileShape,
                              const std::vector<LogicalTensorPtr>& iOperand,
                              const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto shape = tileShape.GetVecTile().tile;
    std::vector<int64_t> tmpShape;
    tmpShape.assign(shape.begin(), shape.end());
    auto alignSize = BLOCK_SIZE / BytesOf(DT_FP32);
    tmpShape[tmpShape.size() - 1] = (tmpShape[tmpShape.size() - 1] + alignSize - 1) / alignSize * alignSize;
    // 3个中间变量
    uint64_t intermediateBytes = static_cast<int64_t>(BytesOf(DT_FP32)) * NUM_VALUE_21 *
                                 std::accumulate(tmpShape.begin(), tmpShape.end(), 1LL, std::multiplies<int64_t>());

    return TiledUnaryOperation<UnaryOpType::DIGAMMA>(function, tileShape, iOperand[0], oOperand[0], intermediateBytes);
}

REGISTER_OPERATION_TILED_FUNC(OP_DIGAMMA, Opcode::OP_DIGAMMA, DigammaOperationTileFunc);

} // namespace npu::tile_fwk
