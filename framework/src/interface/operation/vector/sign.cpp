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
 * \file sign.cpp
 * \brief
 */

#include "unary.h"
#include "binary.h"
#include "binary_tiled.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"
#include "tilefwk/error_code.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {

void TiledSignOperation(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                        const LogicalTensorPtr& result)
{
    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);

        constexpr size_t ALIGN_SIZE = NUM_VALUE_32;
        int64_t tmpSize = ALIGN_SIZE / BytesOf(DT_FP16);
        if (input.tensor.GetDataType() == DT_INT8) {
            tmpSize = MultiplyLastTwoDims<float16>(input.tileInfo.shape);
        } else if (input.tensor.GetDataType() == DT_FP16 || input.tensor.GetDataType() == DT_FP32) {
            const int64_t typeSize = BytesOf(input.tensor.GetDataType());
            const int64_t alignElements = ALIGN_SIZE / typeSize;
            const auto& shape = input.tileInfo.shape;
            const int64_t tmpH = shape.size() > 1 ? shape[shape.size() - 2] : 1;
            const int64_t tmpW = (shape.back() + alignElements - 1) / alignElements * alignElements;
            // One work tile, one mask tile with the same byte footprint, and one scalar block.
            const int64_t tmpBytes = 2 * tmpH * tmpW * typeSize + ALIGN_SIZE;
            tmpSize = tmpBytes / BytesOf(DT_FP16);
        }

        std::vector<int64_t> tmpShape({tmpSize});
        auto tmpTensor = std::make_shared<LogicalTensor>(function, DT_FP16, tmpShape);
        function.AddOperation(Opcode::OP_SIGN, {tile}, {resultTile, tmpTensor});

        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledSignOperation(function, tileShape, cur + 1, input, result);
    }
}

void TiledSignOperation(Function& function, const TileShape& tileShape, const LogicalTensorPtr& self,
                        const LogicalTensorPtr& result)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, self->shape.size() == self->offset.size())
        << "Shape size and offset size should be equal";

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{self, tileInfo};
    TiledSignOperation(function, tileShape, 0, input, result);
}

void TiledSignbitOperation(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                           const LogicalTensorPtr& result)
{
    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);

        int64_t tmpSize = MultiplyLastTwoDims<float16>(input.tileInfo.shape);
        std::vector<int64_t> tmpShape({tmpSize});
        auto tmpTensor = std::make_shared<LogicalTensor>(function, DT_FP16, tmpShape);
        function.AddOperation(Opcode::OP_SIGNBIT, {tile}, {resultTile, tmpTensor});

        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledSignbitOperation(function, tileShape, cur + 1, input, result);
    }
}

void TiledSignbitOperation(Function& function, const TileShape& tileShape, const LogicalTensorPtr& self,
                           const LogicalTensorPtr& result)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, self->shape.size() == self->offset.size())
        << "Shape size and offset size should be equal";

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{self, tileInfo};
    TiledSignbitOperation(function, tileShape, 0, input, result);
}

LogicalTensorPtr TensorSignOperation(Function& function, LogicalTensorPtr self)
{
    auto result = std::make_shared<LogicalTensor>(function, self->tensor->datatype, self->shape,
                                                  self->GetDynValidShape());
    function.AddOperation(Opcode::OP_SIGN, {self}, {result});
    return result;
}

LogicalTensorPtr TensorSignbitOperation(Function& function, LogicalTensorPtr self)
{
    auto result = std::make_shared<LogicalTensor>(function, DataType::DT_BOOL, self->shape, self->GetDynValidShape());
    function.AddOperation(Opcode::OP_SIGNBIT, {self}, {result});
    return result;
}

Tensor Sign(const Tensor& self)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Sign");

    const auto& combinations = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_SIGN);
    CheckTensorDataTypeCombination({self.GetDataType()}, combinations, "SIGN");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "SIGN");
    RETURN_CALL(SignOperation, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

Tensor Signbit(const Tensor& self)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Signbit");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32, DT_INT8};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "SIGNBIT");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "SIGNBIT");
    RETURN_CALL(SignbitOperation, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

Tensor CopySign(const Tensor& self, const Tensor& other)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "CopySign");
    CheckVectorOperand(other.GetStorage(), "CopySign");

    CheckTensorsDataTypeConsistency(self.GetStorage(), other.GetStorage(), "COPYSIGN");
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "COPYSIGN");

    DataType selfDType = self.GetDataType();
    DataType otherDType = other.GetDataType();
    Tensor castSelf = self;
    Tensor castOther = other;
    if (selfDType == DT_INT16 || selfDType == DT_INT32) {
        castSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                        self.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    }
    if (otherDType == DT_INT16 || otherDType == DT_INT32) {
        castOther = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                         other.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    }
    RETURN_CALL(BinaryOperation<BinaryOpType::COPYSIGN>, *Program::GetInstance().GetCurrentFunction(), castSelf,
                castOther);
}

void SignOperationTileFunc(Function& function, const TileShape& tileShape,
                           const std::vector<LogicalTensorPtr>& iOperand, const std::vector<LogicalTensorPtr>& oOperand,
                           [[maybe_unused]] const Operation& op)
{
    TiledSignOperation(function, tileShape, iOperand[0], oOperand[0]);
}

void SignbitOperationTileFunc(Function& function, const TileShape& tileShape,
                              const std::vector<LogicalTensorPtr>& iOperand,
                              const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    TiledSignbitOperation(function, tileShape, iOperand[0], oOperand[0]);
}

REGISTER_OPERATION_TILED_FUNC(OP_SIGN, Opcode::OP_SIGN, SignOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_SIGNBIT, Opcode::OP_SIGNBIT, SignbitOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_COPYSIGN, Opcode::OP_COPYSIGN, BinaryOperationTileFunc<BinaryOpType::COPYSIGN>);

} // namespace npu::tile_fwk
