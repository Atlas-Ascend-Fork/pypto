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
 * \file one_hot.cpp
 * \brief
 */

#include "unary.h"
#include "binary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

void TiledOneHot(Function& function, const TileShape& tileShape, size_t cur, Input& input, Input& output,
                 int numClasses)
{
    if (cur == output.tensor.GetShape().size()) {
        auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto outputTile = output.tensor.GetStorage()->View(function, output.tileInfo.shape, output.tileInfo.offset);
        auto& newOp = function.AddOperation(Opcode::OP_ONEHOT, {inputTile}, {outputTile});
        newOp.SetAttribute(OP_ATTR_PREFIX + "numClasses", numClasses);
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < output.tensor.GetShape()[cur]; i += vecTile[cur]) {
        if (cur < input.tensor.GetShape().size()) {
            input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
            input.tileInfo.offset[cur] = i;
        }
        output.tileInfo.shape[cur] = std::min(output.tensor.GetShape()[cur] - i, vecTile[cur]);
        output.tileInfo.offset[cur] = i;
        TiledOneHot(function, tileShape, cur + 1, input, output, numClasses);
    }
}

void TiledOneHot(Function& function, const TileShape& tileShape, const LogicalTensorPtr& self,
                 const LogicalTensorPtr& result, int numClasses)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, self->shape.size() == self->offset.size())
        << "Shape size and offset size should be equal";
    CHECK(VectorErrorCode::ERR_CONFIG_TILE, numClasses == tileShape.GetVecTile()[result->shape.size() - 1])
        << "The numClasses and last axis of tileshape should be equal";

    TileInfo inputTileInfo(self->shape.size(), self->offset.size());
    TileInfo outputTileInfo(result->shape.size(), result->offset.size());
    auto input = Input{self, inputTileInfo};
    auto output = Input{result, outputTileInfo};
    TiledOneHot(function, tileShape, 0, input, output, numClasses);
}

Tensor TensorOneHot(Function& function, const LogicalTensorPtr& self, int numClasses)
{
    Shape shape(self->shape);
    std::vector<SymbolicScalar> validShape(self->dynValidShape_);
    shape.push_back(static_cast<int64_t>(numClasses));
    validShape.push_back(SymbolicScalar(numClasses));
    auto result = std::make_shared<LogicalTensor>(function, DataType::DT_INT64, shape, validShape);
    auto& op = function.AddOperation(Opcode::OP_ONEHOT, {self}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "numClasses", numClasses);
    return result;
}

Tensor OneHot(const Tensor& self, int numClasses)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "OneHot");

    std::unordered_set<DataType> supportedTypes = {DT_INT8, DT_INT16, DT_INT32, DT_INT64};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "ONEHOT");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_3, "ONEHOT");
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, numClasses > 0) << "numClasses must be greater than 0";
    auto res = CALL(OneHot, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), numClasses);
    CheckTensorShapeSize(res.GetStorage(), "ONEHOT");
    return res;
}

void OneHotOperationTileFunc(Function& function, const TileShape& tileShape,
                             const std::vector<LogicalTensorPtr>& iOperand,
                             const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    int numClasses = op.GetIntAttribute(OP_ATTR_PREFIX + "numClasses");
    TiledOneHot(function, tileShape, iOperand[0], oOperand[0], numClasses);
}

REGISTER_OPERATION_TILED_FUNC(OP_ONEHOT, Opcode::OP_ONEHOT, OneHotOperationTileFunc);

} // namespace npu::tile_fwk
