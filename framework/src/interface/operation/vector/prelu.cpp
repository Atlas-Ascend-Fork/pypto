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
 * \file prelu.cpp
 * \brief
 */
#include "tilefwk/data_type.h"
#include "unary.h"
#include "binary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/error_code.h"
#include "passes/tile_graph_pass/graph_constraint/axis_combine.h"

namespace npu::tile_fwk {

void TiledPReLUOperation(Function& function, const TileShape& tileShape, size_t cur, Input& input, Input& weight,
                         const LogicalTensorPtr& result)
{
    if (cur == 0 && input.tensor.GetShape().size() == 1) {
        // For a 1D input, weight has shape [1] and does not need tiling. Initialize its tile info directly.
        weight.tileInfo.shape[0] = 1;
        weight.tileInfo.offset[0] = 0;
    }

    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto weightTile = weight.tensor.GetStorage()->View(function, weight.tileInfo.shape, weight.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);
        int axis = NUM_VALUE_5 - cur + 1;
        constexpr size_t ALIGN_SIZE = NUM_VALUE_32;
        constexpr size_t SIZEOFBYTE = NUM_VALUE_8;
        int64_t tmpSize = ALIGN_SIZE;
        if (axis == NUM_VALUE_4) {
            tmpSize = (input.tileInfo.shape[cur - 1] + SIZEOFBYTE - 1) / SIZEOFBYTE;
            tmpSize = (tmpSize + ALIGN_SIZE - 1) / ALIGN_SIZE * ALIGN_SIZE + ALIGN_SIZE;
        }
        std::vector<int64_t> tmpShape({tmpSize});
        auto tmpTensor = std::make_shared<LogicalTensor>(function, DT_UINT8, tmpShape);
        auto& op = function.AddOperation(Opcode::OP_PRELU, {tile, weightTile}, {resultTile, tmpTensor});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);

        size_t dimSize = input.tensor.GetShape().size();
        if (dimSize == NUM_VALUE_2) {
            std::vector<bool> dimMap({true, false});
            op.SetAttr(OpAttributeKey::rowPad, dimMap);
        }
        return;
    }
    auto& vecTile = tileShape.GetVecTile();

    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        // A 1D input does not require tiling the weight.
        if (input.tensor.GetShape().size() > 1 && cur == 1) {
            weight.tileInfo.shape[0] = std::min(weight.tensor.GetShape()[0] - i, vecTile[cur]);
            weight.tileInfo.offset[0] = i;
        }
        TiledPReLUOperation(function, tileShape, cur + 1, input, weight, result);
    }
}

void TiledPReLUOperation(Function& function, const TileShape& tileShape, const LogicalTensorPtr& input,
                         const LogicalTensorPtr& weight, const LogicalTensorPtr& result)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, input->shape.size() == input->offset.size())
        << "The shape size of input and offset must be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, weight->shape.size() == weight->offset.size())
        << "The shape size of weight and offset must be equal";

    TileInfo inputTileInfo(input->shape.size(), input->offset.size());
    TileInfo weightTileInfo(weight->shape.size(), weight->offset.size());
    auto inputArg = Input{input, inputTileInfo};
    auto weightArg = Input{weight, weightTileInfo};
    TiledPReLUOperation(function, tileShape, 0, inputArg, weightArg, result);
}

void PReLUOperationOperandCheck(const LogicalTensorPtr& selfTensor, const LogicalTensorPtr& weightTensor)
{
    CheckTensorDimRange(selfTensor, 1, NUM_VALUE_4, "PReLU");
    CheckTensorDimRange(weightTensor, 1, 1, "PReLU");

    if (selfTensor->shape.size() == 1) {
        // For a 1D input, weight must have shape [1].
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, weightTensor->shape[0] == 1)
            << "The weight size should be [1] when input is 1D";
    } else {
        // For a 2D, 3D, or 4D input, weight must match the second dimension of self.
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, weightTensor->shape[0] == selfTensor->shape[1])
            << "The weight size should be equal to the input's second dimension";
    }
}

void PReLUOperationTileFunc(Function& function, const TileShape& tileShape,
                            const std::vector<LogicalTensorPtr>& iOperand,
                            const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    TiledPReLUOperation(function, tileShape, iOperand[0], iOperand[1], oOperand[0]);
}

LogicalTensorPtr TensorPReLUOperation(Function& function, const Tensor& self, const Tensor& weight)
{
    auto selfTensor = self.GetStorage();
    auto weightTensor = weight.GetStorage();

    PReLUOperationOperandCheck(selfTensor, weightTensor);

    auto result = std::make_shared<LogicalTensor>(function, selfTensor->Datatype(), selfTensor->shape,
                                                  selfTensor->GetDynValidShape());
    function.AddOperation(Opcode::OP_PRELU, {selfTensor, weightTensor}, {result});
    return result;
}

Tensor PReLU(const Tensor& self, const Tensor& weight)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "PReLU");
    CheckVectorOperand(weight.GetStorage(), "PReLU");

    CheckTensorsDataTypeConsistency(self.GetStorage(), weight.GetStorage(), "PReLU");
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "PReLU");

    RETURN_CALL(PReLUOperation, *Program::GetInstance().GetCurrentFunction(), self, weight);
}

REGISTER_OPERATION_TILED_FUNC(OP_PRELU, Opcode::OP_PRELU, PReLUOperationTileFunc);

} // namespace npu::tile_fwk
