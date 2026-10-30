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
 * \file error_function.cpp
 * \brief
 */

#include "unary_tiled.h"
#include "binary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

Tensor Erf(const Tensor& self)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Erf");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Erf");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "Erf");

    auto castSelf = Cast(self, DataType::DT_FP32);
    auto result = CALL(UnaryOperation<UnaryOpType::ERF>, *Program::GetInstance().GetCurrentFunction(),
                       castSelf.GetStorage());
    auto castResult = Cast(result, self.GetDataType());
    return castResult;
}

Tensor Erfc(const Tensor& self)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Erfc");

    std::unordered_set<DataType> supportedTypes = {DT_BF16, DT_FP16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Erfc");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "Erfc");
    if (self.GetDataType() != DataType::DT_FP32) {
        auto castSelf = Cast(self, DataType::DT_FP32);
        auto result = CALL(UnaryOperation<UnaryOpType::ERFC>, *Program::GetInstance().GetCurrentFunction(),
                           castSelf.GetStorage());
        auto castResult = Cast(result, self.GetDataType());
        return castResult;
    }
    auto result = CALL(UnaryOperation<UnaryOpType::ERFC>, *Program::GetInstance().GetCurrentFunction(),
                       self.GetStorage());
    return result;
}

void ErfOperationTileFunc(Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
                          const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto shape = tileShape.GetVecTile().tile;
    std::vector<int64_t> tmpShape;
    tmpShape.assign(shape.begin(), shape.end());
    auto alignSize = BLOCK_SIZE / BytesOf(DT_FP32);
    tmpShape[tmpShape.size() - 1] = (tmpShape[tmpShape.size() - 1] + alignSize - 1) / alignSize * alignSize;
    // 3个中间变量
    uint64_t intermediateBytes = static_cast<int64_t>(BytesOf(DT_FP32)) * NUM_VALUE_3 *
                                 std::accumulate(tmpShape.begin(), tmpShape.end(), 1LL, std::multiplies<int64_t>());

    return TiledUnaryOperation<UnaryOpType::ERF>(function, tileShape, iOperand[0], oOperand[0], intermediateBytes);
}

void ErfcOperationTileFunc(Function& function, const TileShape& tileShape,
                           const std::vector<LogicalTensorPtr>& iOperand, const std::vector<LogicalTensorPtr>& oOperand,
                           [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto shape = tileShape.GetVecTile().tile;
    std::vector<int64_t> tmpShape;
    tmpShape.assign(shape.begin(), shape.end());
    auto alignSize = BLOCK_SIZE / BytesOf(DT_FP32);
    tmpShape[tmpShape.size() - 1] = (tmpShape[tmpShape.size() - 1] + alignSize - 1) / alignSize * alignSize;
    uint64_t intermediateBytes = static_cast<int64_t>(BytesOf(DT_FP32)) * NUM_VALUE_4 *
                                 std::accumulate(tmpShape.begin(), tmpShape.end(), 1LL, std::multiplies<int64_t>());
    return TiledUnaryOperation<UnaryOpType::ERFC>(function, tileShape, iOperand[0], oOperand[0], intermediateBytes);
}

REGISTER_OPERATION_TILED_FUNC(OP_ERF, Opcode::OP_ERF, ErfOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ERFC, Opcode::OP_ERFC, ErfcOperationTileFunc);

} // namespace npu::tile_fwk
