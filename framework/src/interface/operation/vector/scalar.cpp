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
 * \file scalar.cpp
 * \brief Scalar operation implementations.
 */

#include "tilefwk/data_type.h"
#include "unary.h"
#include "binary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {
template <BinaryOpType T>
void TiledBinaryOperationAllScalar(Function& function, const TileShape& tileShape, size_t cur, LogicalInput& input1,
                                   Element& value, const LogicalTensorPtr& result, TileInfo& resultTileInfo,
                                   bool reverseOperand)
{
    if (cur == input1.tensor->GetShape().size()) {
        auto inputTile1 = input1.tensor->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        // 确认接口
        auto& op = function.AddOperation(GetBinaryOpNameCode<T, true>(), {inputTile1}, {resultTile});
        op.SetAttribute(OpAttributeKey::scalar, value);
        op.SetAttribute(OP_ATTR_PREFIX + "reverseOperand", reverseOperand);
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input1.tileInfo.offset[cur] = i % input1.tensor->GetShape()[cur];
        input1.tileInfo.shape[cur] = std::min(input1.tensor->GetShape()[cur] - input1.tileInfo.offset[cur],
                                              vecTile[cur]);

        TiledBinaryOperationAllScalar<T>(function, tileShape, cur + 1, input1, value, result, resultTileInfo,
                                         reverseOperand);
    }
}

template <BinaryOpType T>
void TiledBinaryOperationAllScalar(Function& function, const TileShape& tileShape, LogicalTensorPtr operand1,
                                   Element value, const LogicalTensorPtr& result, bool reverseOperand)
{
    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = LogicalInput{operand1, tileInfo1};
    TiledBinaryOperationAllScalar<T>(function, tileShape, 0, input1, value, result, resultTileInfo, reverseOperand);
}

Tensor ScalarAddS(const Tensor& operand, const Element& value, bool reverseOperand)
{
    DECLARE_TRACER();
    CheckVectorOperand(operand.GetStorage(), "ScalarAddS");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(operand.GetStorage(), supportedTypes, "S_ADD");

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_ADD>, *Program::GetInstance().GetCurrentFunction(),
                operand.GetStorage(), value, reverseOperand);
}

Tensor ScalarSubS(const Tensor& operand, const Element& value, bool reverseOperand)
{
    DECLARE_TRACER();
    CheckVectorOperand(operand.GetStorage(), "ScalarSubS");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(operand.GetStorage(), supportedTypes, "S_SUB");

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_SUB>, *Program::GetInstance().GetCurrentFunction(),
                operand.GetStorage(), value, reverseOperand);
}

Tensor ScalarMulS(const Tensor& operand, const Element& value, bool reverseOperand)
{
    DECLARE_TRACER();
    CheckVectorOperand(operand.GetStorage(), "ScalarMulS");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(operand.GetStorage(), supportedTypes, "S_MUL");

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_MUL>, *Program::GetInstance().GetCurrentFunction(),
                operand.GetStorage(), value, reverseOperand);
}

Tensor ScalarDivS(const Tensor& operand, const Element& value, bool reverseOperand)
{
    DECLARE_TRACER();
    CheckVectorOperand(operand.GetStorage(), "ScalarDivS");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(operand.GetStorage(), supportedTypes, "S_DIV");

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_DIV>, *Program::GetInstance().GetCurrentFunction(),
                operand.GetStorage(), value, reverseOperand);
}

Tensor ScalarMaxS(const Tensor& operand, const Element& value, bool reverseOperand)
{
    DECLARE_TRACER();
    CheckVectorOperand(operand.GetStorage(), "ScalarMaxS");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(operand.GetStorage(), supportedTypes, "S_MAX");

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_MAX>, *Program::GetInstance().GetCurrentFunction(),
                operand.GetStorage(), value, reverseOperand);
}

template <BinaryOpType T>
void TiledBinaryOperationAllScalar(Function& function, const TileShape& tileShape, size_t cur, LogicalInput& input1,
                                   LogicalInput& input2, const LogicalTensorPtr& result, TileInfo& resultTileInfo)
{
    if (cur == input1.tensor->GetShape().size()) {
        auto inputTile1 = input1.tensor->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto inputTile2 = input2.tensor->View(function, input2.tileInfo.shape, input2.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        function.AddOperation(GetBinaryOpNameCode<T, false>(), {inputTile1, inputTile2}, {resultTile});
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input1.tileInfo.offset[cur] = i % input1.tensor->GetShape()[cur];
        input1.tileInfo.shape[cur] = std::min(input1.tensor->GetShape()[cur] - input1.tileInfo.offset[cur],
                                              vecTile[cur]);
        input2.tileInfo.offset[cur] = i % input2.tensor->GetShape()[cur];
        input2.tileInfo.shape[cur] = std::min(input2.tensor->GetShape()[cur] - input2.tileInfo.offset[cur],
                                              vecTile[cur]);
        TiledBinaryOperationAllScalar<T>(function, tileShape, cur + 1, input1, input2, result, resultTileInfo);
    }
}

template <BinaryOpType T>
void TiledBinaryOperationAllScalar(Function& function, const TileShape& tileShape, LogicalTensorPtr operand1,
                                   LogicalTensorPtr operand2, const LogicalTensorPtr& result)
{
    CheckBinOpOperandsValid(operand1, operand2);

    if (operand1->shape != result->shape) {
        auto targetShape = result->shape;
        auto tmp = std::make_shared<LogicalTensor>(function, operand1->Datatype(), targetShape);
        Expand(function, tileShape, operand1, {operand2}, tmp);
        operand1 = tmp;
    }

    if (operand2->shape != result->shape) {
        auto targetShape = result->shape;
        auto tmp = std::make_shared<LogicalTensor>(function, operand2->Datatype(), targetShape);
        Expand(function, tileShape, operand2, {operand1}, tmp);
        operand2 = tmp;
    }

    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo tileInfo2(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = LogicalInput{operand1, tileInfo1};
    auto input2 = LogicalInput{operand2, tileInfo2};
    TiledBinaryOperationAllScalar<T>(function, tileShape, 0, input1, input2, result, resultTileInfo);
}

Tensor ScalarAdd(const Tensor& operand1, const Tensor& operand2)
{
    DECLARE_TRACER();
    CheckVectorOperand(operand1.GetStorage(), "ScalarAdd");
    CheckVectorOperand(operand2.GetStorage(), "ScalarAdd");

    CheckTensorsDataTypeConsistency(operand1.GetStorage(), operand2.GetStorage(), "S_ADD");
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(operand1.GetStorage(), supportedTypes, "S_ADD");

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_ADD>, *Program::GetInstance().GetCurrentFunction(),
                operand1.GetStorage(), operand2.GetStorage());
}
Tensor ScalarSub(const Tensor& operand1, const Tensor& operand2)
{
    DECLARE_TRACER();
    CheckVectorOperand(operand1.GetStorage(), "ScalarSub");
    CheckVectorOperand(operand2.GetStorage(), "ScalarSub");

    CheckTensorsDataTypeConsistency(operand1.GetStorage(), operand2.GetStorage(), "S_SUB");
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(operand1.GetStorage(), supportedTypes, "S_SUB");

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_SUB>, *Program::GetInstance().GetCurrentFunction(),
                operand1.GetStorage(), operand2.GetStorage());
}

Tensor ScalarMul(const Tensor& operand1, const Tensor& operand2)
{
    DECLARE_TRACER();
    CheckVectorOperand(operand1.GetStorage(), "ScalarMul");
    CheckVectorOperand(operand2.GetStorage(), "ScalarMul");

    CheckTensorsDataTypeConsistency(operand1.GetStorage(), operand2.GetStorage(), "S_MUL");
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(operand1.GetStorage(), supportedTypes, "S_MUL");

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_MUL>, *Program::GetInstance().GetCurrentFunction(),
                operand1.GetStorage(), operand2.GetStorage());
}

Tensor ScalarDiv(const Tensor& operand1, const Tensor& operand2)
{
    DECLARE_TRACER();
    CheckVectorOperand(operand1.GetStorage(), "ScalarDiv");
    CheckVectorOperand(operand2.GetStorage(), "ScalarDiv");

    CheckTensorsDataTypeConsistency(operand1.GetStorage(), operand2.GetStorage(), "S_DIV");
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(operand1.GetStorage(), supportedTypes, "S_DIV");

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_DIV>, *Program::GetInstance().GetCurrentFunction(),
                operand1.GetStorage(), operand2.GetStorage());
}

Tensor ScalarMax(const Tensor& operand1, const Tensor& operand2)
{
    DECLARE_TRACER();
    CheckVectorOperand(operand1.GetStorage(), "ScalarMax");
    CheckVectorOperand(operand2.GetStorage(), "ScalarMax");

    CheckTensorsDataTypeConsistency(operand1.GetStorage(), operand2.GetStorage(), "S_MAX");
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(operand1.GetStorage(), supportedTypes, "S_MAX");

    RETURN_CALL(BinaryOperationAllScalar<BinaryOpType::S_MAX>, *Program::GetInstance().GetCurrentFunction(),
                operand1.GetStorage(), operand2.GetStorage());
}

// OP_ADDS OP_SUBS OP_MULS OP_DIVS OP_MAXS OP_MINS OP_BITWISEANDS OP_BITWISEORS OP_BITWISEXORS

template <BinaryOpType T>
void BinaryOperationAllScalarResTileFunc(Function& function, const TileShape& tileShape,
                                         const std::vector<LogicalTensorPtr>& iOperand,
                                         const std::vector<LogicalTensorPtr>& oOperand,
                                         [[maybe_unused]] const Operation& op)
{
    TiledBinaryOperationAllScalar<T>(function, tileShape, iOperand[0], op.GetElementAttribute(OpAttributeKey::scalar),
                                     oOperand[0], op.GetBoolAttribute(OP_ATTR_PREFIX + "reverseOperand"));
}

// OP_S_ADD OP_S_SUB OP_S_MUL OP_S_DIV OP_S_MAX
template <BinaryOpType T>
void BinaryOperationAllScalarTileFunc(Function& function, const TileShape& tileShape,
                                      const std::vector<LogicalTensorPtr>& iOperand,
                                      const std::vector<LogicalTensorPtr>& oOperand,
                                      [[maybe_unused]] const Operation& op)
{
    BinaryOperationOperandCheck(iOperand, oOperand);
    CheckTensorsDataTypeConsistency(iOperand[0], iOperand[1], GetBinaryOpName<T>());
    TiledBinaryOperationAllScalar<T>(function, tileShape, iOperand[0], iOperand[1], oOperand[0]);
}

REGISTER_OPERATION_TILED_FUNC(OP_S_ADDS, Opcode::OP_S_ADDS, BinaryOperationAllScalarResTileFunc<BinaryOpType::S_ADD>);
REGISTER_OPERATION_TILED_FUNC(OP_S_SUBS, Opcode::OP_S_SUBS, BinaryOperationAllScalarResTileFunc<BinaryOpType::S_SUB>);
REGISTER_OPERATION_TILED_FUNC(OP_S_MULS, Opcode::OP_S_MULS, BinaryOperationAllScalarResTileFunc<BinaryOpType::S_MUL>);
REGISTER_OPERATION_TILED_FUNC(OP_S_DIVS, Opcode::OP_S_DIVS, BinaryOperationAllScalarResTileFunc<BinaryOpType::S_DIV>);
REGISTER_OPERATION_TILED_FUNC(OP_S_MAXS, Opcode::OP_S_MAXS, BinaryOperationAllScalarResTileFunc<BinaryOpType::S_MAX>);
REGISTER_OPERATION_TILED_FUNC(OP_S_MINS, Opcode::OP_S_MINS, BinaryOperationAllScalarResTileFunc<BinaryOpType::S_MIN>);

REGISTER_OPERATION_TILED_FUNC(OP_S_ADD, Opcode::OP_S_ADD, BinaryOperationAllScalarTileFunc<BinaryOpType::S_ADD>);
REGISTER_OPERATION_TILED_FUNC(OP_S_SUB, Opcode::OP_S_SUB, BinaryOperationAllScalarTileFunc<BinaryOpType::S_SUB>);
REGISTER_OPERATION_TILED_FUNC(OP_S_MUL, Opcode::OP_S_MUL, BinaryOperationAllScalarTileFunc<BinaryOpType::S_MUL>);
REGISTER_OPERATION_TILED_FUNC(OP_S_DIV, Opcode::OP_S_DIV, BinaryOperationAllScalarTileFunc<BinaryOpType::S_DIV>);
REGISTER_OPERATION_TILED_FUNC(OP_S_MAX, Opcode::OP_S_MAX, BinaryOperationAllScalarTileFunc<BinaryOpType::S_MAX>);
REGISTER_OPERATION_TILED_FUNC(OP_S_MIN, Opcode::OP_S_MIN, BinaryOperationAllScalarTileFunc<BinaryOpType::S_MIN>);

} // namespace npu::tile_fwk
