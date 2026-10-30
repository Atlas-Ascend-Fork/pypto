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
 * \file hyperbolic.cpp
 * \brief
 */

#include "unary_tiled.h"
#include "binary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {
namespace {

int64_t CmpResAlign(const std::vector<int64_t>& vec)
{
    constexpr size_t ALIGN_SIZE = NUM_VALUE_32;
    constexpr size_t ALIGN_BIT = NUM_VALUE_8;
    int64_t axis2 = (vec[vec.size() - 1] + ALIGN_BIT - 1) / ALIGN_BIT * ALIGN_BIT;
    axis2 = (axis2 + ALIGN_SIZE - 1) / ALIGN_SIZE * ALIGN_SIZE;
    return axis2 * vec[vec.size() - NUM_VALUE_2];
}

} // namespace

Tensor Sinh(const Tensor& self)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Sinh");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_FP32, DT_BF16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "SINH");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "SINH");

    auto castSelf = Cast(self, DataType::DT_FP32);
    auto result = CALL(UnaryOperation<UnaryOpType::SINH>, *Program::GetInstance().GetCurrentFunction(),
                       castSelf.GetStorage());
    auto castResult = Cast(result, self.GetDataType());
    return castResult;
}

Tensor Cosh(const Tensor& self)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Cosh");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_FP32, DT_BF16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "COSH");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "COSH");

    auto castSelf = Cast(self, DataType::DT_FP32);
    auto result = CALL(UnaryOperation<UnaryOpType::COSH>, *Program::GetInstance().GetCurrentFunction(),
                       castSelf.GetStorage());
    auto castResult = Cast(result, self.GetDataType());
    return castResult;
}

Tensor Atanh(const Tensor& self)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Atanh");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "ATANH");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "ATANH");

    auto castSelf = Cast(self, DataType::DT_FP32);
    auto result = CALL(UnaryOperation<UnaryOpType::ATANH>, *Program::GetInstance().GetCurrentFunction(),
                       castSelf.GetStorage());
    auto castResult = Cast(result, self.GetDataType());
    return castResult;
}

Tensor ASinh(const Tensor& self)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "ASinh");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_FP32, DT_BF16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "ASinh");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "ASinh");

    auto castSelf = Cast(self, DataType::DT_FP32);
    auto result = CALL(UnaryOperation<UnaryOpType::ASINH>, *Program::GetInstance().GetCurrentFunction(),
                       castSelf.GetStorage());
    auto castResult = Cast(result, self.GetDataType());
    return castResult;
}

Tensor ACosh(const Tensor& self)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "ACosh");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_FP32, DT_BF16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "ACosh");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "ACosh");

    auto castSelf = Cast(self, DataType::DT_FP32);
    auto result = CALL(UnaryOperation<UnaryOpType::ACOSH>, *Program::GetInstance().GetCurrentFunction(),
                       castSelf.GetStorage());
    auto castResult = Cast(result, self.GetDataType());
    return castResult;
}

void TiledTanhOperation(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                        const LogicalTensorPtr& result)
{
    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);

        constexpr size_t ALIGN_SIZE = NUM_VALUE_32;
        int64_t tmpSize = MultiplyLastTwoDims<float>(tileShape.GetVecTile().tile);
        int64_t cmpsize = CmpResAlign(tileShape.GetVecTile().tile);
        if (input.tensor.GetDataType() != DT_FP32) {
            tmpSize = NUM_VALUE_4 * tmpSize * sizeof(float);
        } else {
            tmpSize = NUM_VALUE_2 * tmpSize * sizeof(float);
        }
        tmpSize = tmpSize + cmpsize + ALIGN_SIZE;

        std::vector<int64_t> tmpShape({tmpSize});
        auto tmpTensor = std::make_shared<LogicalTensor>(function, DT_INT8, tmpShape);
        function.AddOperation(Opcode::OP_TANH, {tile}, {resultTile, tmpTensor});

        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledTanhOperation(function, tileShape, cur + 1, input, result);
    }
}

void TiledTanhOperation(Function& function, const TileShape& tileShape, const LogicalTensorPtr& self,
                        const LogicalTensorPtr& result)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, self->shape.size() == self->offset.size())
        << "Shape size and offset size should be equal";

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{self, tileInfo};
    TiledTanhOperation(function, tileShape, 0, input, result);
}

LogicalTensorPtr TensorTanhOperation(Function& function, LogicalTensorPtr self)
{
    auto result = std::make_shared<LogicalTensor>(function, self->tensor->datatype, self->shape,
                                                  self->GetDynValidShape());
    function.AddOperation(Opcode::OP_TANH, {self}, {result});
    return result;
}

Tensor Tanh(const Tensor& self)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Tanh");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "TANH");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "TANH");

    RETURN_CALL(TanhOperation, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

template <UnaryOpType T, int64_t TmpBlockNum>
void Fp32AlignedTmpUnaryOperationTileFunc(Function& function, const TileShape& tileShape,
                                          const std::vector<LogicalTensorPtr>& iOperand,
                                          const std::vector<LogicalTensorPtr>& oOperand,
                                          [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto shape = tileShape.GetVecTile();
    int dim = shape.size();
    auto alignSize = BLOCK_SIZE / BytesOf(DT_FP32);
    std::vector<int64_t> tmpShape = shape.tile;
    tmpShape[dim - 1] = AlignUp(tmpShape[dim - 1], alignSize) * TmpBlockNum;
    uint64_t intermediateBytes = std::accumulate(tmpShape.begin(), tmpShape.end(), 1LL, std::multiplies<int64_t>()) *
                                 BytesOf(DT_FP32);
    return TiledUnaryOperation<T>(function, tileShape, iOperand[0], oOperand[0], intermediateBytes);
}

void AtanhOperationTileFunc(Function& function, const TileShape& tileShape,
                            const std::vector<LogicalTensorPtr>& iOperand,
                            const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto shape = tileShape.GetVecTile();
    int dim = shape.size();
    auto alignSize = BLOCK_SIZE / BytesOf(DT_FP32);
    std::vector<int64_t> tmpShape = shape.tile;
    tmpShape[dim - 1] = AlignUp(tmpShape[dim - 1], alignSize) * NUM_VALUE_5;
    uint64_t intermediateBytes = std::accumulate(tmpShape.begin(), tmpShape.end(), 1LL, std::multiplies<int64_t>()) *
                                 BytesOf(DT_FP32);
    return TiledUnaryOperation<UnaryOpType::ATANH>(function, tileShape, iOperand[0], oOperand[0], intermediateBytes);
}

void TanhOperationTileFunc(Function& function, const TileShape& tileShape,
                           const std::vector<LogicalTensorPtr>& iOperand, const std::vector<LogicalTensorPtr>& oOperand,
                           [[maybe_unused]] const Operation& op)
{
    TiledTanhOperation(function, tileShape, iOperand[0], oOperand[0]);
}

REGISTER_OPERATION_TILED_FUNC(OP_SINH, Opcode::OP_SINH,
                              (Fp32AlignedTmpUnaryOperationTileFunc<UnaryOpType::SINH, NUM_VALUE_4>));
REGISTER_OPERATION_TILED_FUNC(OP_COSH, Opcode::OP_COSH,
                              (Fp32AlignedTmpUnaryOperationTileFunc<UnaryOpType::COSH, NUM_VALUE_3>));
REGISTER_OPERATION_TILED_FUNC(OP_ATANH, Opcode::OP_ATANH, AtanhOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ASINH, Opcode::OP_ASINH,
                              (Fp32AlignedTmpUnaryOperationTileFunc<UnaryOpType::ASINH, NUM_VALUE_4>));
REGISTER_OPERATION_TILED_FUNC(OP_ACOSH, Opcode::OP_ACOSH,
                              (Fp32AlignedTmpUnaryOperationTileFunc<UnaryOpType::ACOSH, NUM_VALUE_3>));
REGISTER_OPERATION_TILED_FUNC(OP_TANH, Opcode::OP_TANH, TanhOperationTileFunc);

} // namespace npu::tile_fwk
