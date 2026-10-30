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
 * \file exponential.cpp
 * \brief
 */

#include "unary_tiled.h"
#include "binary.h"
#include "binary_tiled.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"
#include "tilefwk/error_code.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {

Tensor Exp(const Tensor& self, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Exp");

    const auto& combinations = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_EXP);
    CheckTensorDataTypeCombination({self.GetDataType()}, combinations, "Exp");

    auto [result, op] = TensorUnaryOperationWithOp<UnaryOpType::EXP>(*Program::GetInstance().GetCurrentFunction(),
                                                                     self.GetStorage());
    op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    return Tensor(result);
}

void ExpOperationTileFunc(Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
                          const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    int64_t precisionType = static_cast<int64_t>(PrecisionType::INTRINSIC);
    if (op.HasAttr(OpAttributeKey::precisionType)) {
        precisionType = op.GetIntAttribute(OpAttributeKey::precisionType);
    }
    return TiledUnaryOperation<UnaryOpType::EXP>(function, tileShape, iOperand[0], oOperand[0], 0, precisionType);
}

Tensor TensorExp2(Function& function, const LogicalTensorPtr& self)
{
    auto result = std::make_shared<LogicalTensor>(function, self->Datatype(), self->GetShape(),
                                                  self->GetDynValidShape());
    if (self->Datatype() == DataType::DT_INT32 || self->Datatype() == DataType::DT_INT16) {
        result = std::make_shared<LogicalTensor>(function, DT_FP32, self->GetShape(), self->GetDynValidShape());
    }
    function.AddOperation(Opcode::OP_EXP2, {self}, {result});
    return result;
}

Tensor Exp2(const Tensor& self)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Exp2");

    std::unordered_set<DataType> supportedTypes = {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "EXP2");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "EXP2");

    RETURN_CALL(Exp2, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

void TiledExp2(Function& function, const TileShape& tileShape, size_t cur, Input& input, const LogicalTensorPtr& result)
{
    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);
        std::vector<int64_t> srcTileShape(input.tileInfo.shape);
        auto tileShapeLen = srcTileShape.size();
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, SHAPE_DIM1 <= tileShapeLen && tileShapeLen <= SHAPE_DIM4)
            << "Length of tile shape only supports 1~4";
        std::vector<int64_t> tmpShape;
        std::vector<int64_t> tmpShape2;
        if (srcTileShape.size() == 1) {
            tmpShape2.assign(srcTileShape.end() - SHAPE_DIM1, srcTileShape.end());
        } else {
            tmpShape2.assign(srcTileShape.end() - SHAPE_DIM2, srcTileShape.end());
        }
        auto alignSize2 = BLOCK_SIZE / BytesOf(DT_FP32);
        tmpShape2[tmpShape2.size() - 1] = (tmpShape2[tmpShape2.size() - 1] + alignSize2 - 1) / alignSize2 * alignSize2;
        if (input.tensor.GetDataType() == DT_FP32) {
            tmpShape = {BLOCK_SIZE / sizeof(float)};
        } else {
            if (srcTileShape.size() == 1) {
                tmpShape.assign(srcTileShape.end() - SHAPE_DIM1, srcTileShape.end());
            } else {
                tmpShape.assign(srcTileShape.end() - SHAPE_DIM2, srcTileShape.end());
            }
            auto alignSize = BLOCK_SIZE / BytesOf(DT_FP32);
            tmpShape[tmpShape.size() - 1] = (tmpShape[tmpShape.size() - 1] + alignSize - 1) / alignSize * alignSize;
        }
        auto tmpTensor = std::make_shared<LogicalTensor>(function, DT_FP32, tmpShape);
        auto tmpTensorNext = std::make_shared<LogicalTensor>(function, DT_FP32, tmpShape2);

        function.AddOperation(Opcode::OP_EXP2, {tile}, {resultTile, tmpTensor, tmpTensorNext});
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledExp2(function, tileShape, cur + 1, input, result);
    }
}

void TiledExp2(Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand,
               const LogicalTensorPtr& result)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, operand->shape.size() == operand->offset.size())
        << "The shape size of operand and offset must be equal";

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};

    TiledExp2(function, tileShape, 0, input, result);
}

void Exp2OperationTileFunc(Function& function, const TileShape& tileShape,
                           const std::vector<LogicalTensorPtr>& iOperand, const std::vector<LogicalTensorPtr>& oOperand,
                           [[maybe_unused]] const Operation& op)
{
    TiledExp2(function, tileShape, iOperand[0], oOperand[0]);
}

Tensor TensorExpm1(Function& function, const LogicalTensorPtr& self)
{
    auto result = std::make_shared<LogicalTensor>(function, self->Datatype(), self->GetShape(),
                                                  self->GetDynValidShape());
    if (self->Datatype() == DataType::DT_INT32 || self->Datatype() == DataType::DT_INT16) {
        result = std::make_shared<LogicalTensor>(function, DT_FP32, self->GetShape(), self->GetDynValidShape());
    }
    function.AddOperation(Opcode::OP_EXPM1, {self}, {result});
    return result;
}

Tensor Expm1(const Tensor& self)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Expm1");

    std::unordered_set<DataType> supportedTypes = {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "EXPM1");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "EXPM1");

    RETURN_CALL(Expm1, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

void TiledExpm1(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                const LogicalTensorPtr& result)
{
    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);
        std::vector<int64_t> srcTileShape(input.tileInfo.shape);
        auto tileShapeLen = srcTileShape.size();
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, SHAPE_DIM1 <= tileShapeLen && tileShapeLen <= SHAPE_DIM4)
            << "Length of tile shape only supports 1~4";
        std::vector<int64_t> tmpShape;
        if (input.tensor.GetDataType() == DT_FP32) {
            tmpShape = {BLOCK_SIZE / sizeof(float)};
        } else {
            if (srcTileShape.size() == 1) {
                tmpShape.assign(srcTileShape.end() - SHAPE_DIM1, srcTileShape.end());
            } else {
                tmpShape.assign(srcTileShape.end() - SHAPE_DIM2, srcTileShape.end());
            }
            auto alignSize = BLOCK_SIZE / BytesOf(DT_FP32);
            tmpShape[tmpShape.size() - 1] = (tmpShape[tmpShape.size() - 1] + alignSize - 1) / alignSize * alignSize;
        }
        auto tmpTensor = std::make_shared<LogicalTensor>(function, DT_FP32, tmpShape);
        function.AddOperation(Opcode::OP_EXPM1, {tile}, {resultTile, tmpTensor});
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledExpm1(function, tileShape, cur + 1, input, result);
    }
}

void TiledExpm1(Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand,
                const LogicalTensorPtr& result)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, operand->shape.size() == operand->offset.size())
        << "The shape size of operand and offset must be equal";

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};

    TiledExpm1(function, tileShape, 0, input, result);
}

void Expm1OperationTileFunc(Function& function, const TileShape& tileShape,
                            const std::vector<LogicalTensorPtr>& iOperand,
                            const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    TiledExpm1(function, tileShape, iOperand[0], oOperand[0]);
}

void TiledExpandExpDifOperation(Function& function, const TileShape& tileShape, size_t cur, LogicalInput& input1,
                                LogicalInput& input2, const LogicalTensorPtr& result, TileInfo& resultTileInfo)
{
    if (cur == input1.tensor->GetShape().size()) {
        auto inputTile1 = input1.tensor->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto inputTile2 = input2.tensor->View(function, input2.tileInfo.shape, input2.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto opName = GetBinaryOpNameCode<BinaryOpType::EXPANDEXPDIF>();
        auto& op = function.AddOperation(opName, {inputTile1, inputTile2}, {resultTile});
        size_t shapeSize = input1.tensor->GetShape().size();
        std::vector<int64_t> brcOperand(shapeSize, 0);
        size_t brcAxesCount = 0;
        for (size_t i = 0; i < shapeSize; i++) {
            brcOperand[i] = BrcAxisBinaryOp(input1.tensor, input2.tensor, i);
            if (brcOperand[i] != 0) {
                brcAxesCount++;
            }
        }
        if (brcAxesCount > 0) {
            op.SetAttribute(OpAttributeKey::brcOperand, brcOperand);
        }
    } else {
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
            TiledExpandExpDifOperation(function, tileShape, cur + 1, input1, input2, result, resultTileInfo);
        }
    }
}

void TiledExpandExpDifOperation(Function& function, const TileShape& tileShape, LogicalTensorPtr operand1,
                                LogicalTensorPtr operand2, const LogicalTensorPtr& result)
{
    CheckBinOpOperandsValid(operand1, operand2);

    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo tileInfo2(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = LogicalInput{operand1, tileInfo1};
    auto input2 = LogicalInput{operand2, tileInfo2};
    TiledExpandExpDifOperation(function, tileShape, 0, input1, input2, result, resultTileInfo);
}

Tensor ExpandExpDif(const Tensor& input, const Tensor& other)
{
    DECLARE_TRACER();
    CheckVectorOperand(input.GetStorage(), "ExpandExpDif");
    CheckVectorOperand(other.GetStorage(), "ExpandExpDif");

    CheckTensorsDataTypeConsistency(input.GetStorage(), other.GetStorage(), "EXPANDEXPDIF");
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(input.GetStorage(), supportedTypes, "EXPANDEXPDIF");
    CheckBinaryInputTensors(input.GetStorage(), other.GetStorage(), "EXPANDEXPDIF");
    config::SetOperationOption(KEY_COMBINE_AXIS, true);
    RETURN_CALL(BinaryOperation<BinaryOpType::EXPANDEXPDIF>, *Program::GetInstance().GetCurrentFunction(), input,
                other);
}

void ExpandExpDifOperationTileFunc(Function& function, const TileShape& tileShape,
                                   const std::vector<LogicalTensorPtr>& iOperand,
                                   const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    BinaryOperationOperandCheck(iOperand, oOperand);
    TiledExpandExpDifOperation(function, tileShape, iOperand[0], iOperand[1], oOperand[0]);
}

DataType GetPowRealResultDataType(DataType selfType, DataType otherType)
{
    if (selfType == DT_INT32) {
        return otherType;
    }
    if (otherType == DT_INT32) {
        return selfType;
    }
    if (selfType == DT_BF16) {
        return otherType == DT_FP16 ? DT_FP32 : otherType;
    }
    if (otherType == DT_BF16) {
        return selfType == DT_FP16 ? DT_FP32 : selfType;
    }
    return selfType == DT_FP16 && otherType == DT_FP16 ? DT_FP16 : DT_FP32;
}

DataType GetPowCalcResultDataType(DataType selfType, DataType otherType)
{
    if (selfType == DT_INT32 && otherType == DT_INT32) {
        return DT_INT32;
    }
    return DT_FP32;
}

LogicalTensorPtr CastToResultType(const LogicalTensorPtr& tensor, DataType originType, DataType resultType)
{
    if (originType == resultType) {
        return tensor;
    }
    RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), tensor, resultType,
                CastMode::CAST_NONE);
}

void PowCheck(const Tensor& self, const Tensor& other)
{
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT32, DT_FP32, DT_INT8, DT_UINT8, DT_INT16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "POW");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "POW");
    CheckTensorShapeSize(self.GetStorage(), "POW");
    CheckTensorShapeSize(other.GetStorage(), "POW");
    CheckTensorsDimConsistency({self.GetStorage(), other.GetStorage()}, "POW");
    CheckTensorsShapeConsistencyOrBroadcast({self.GetStorage(), other.GetStorage()}, "POW");
    CheckTensorsFormatConsistency(self.GetStorage(), other.GetStorage(), "POW");
}

Tensor Pow(const Tensor& self, const Tensor& other, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Pow");
    CheckVectorOperand(other.GetStorage(), "Pow");

    PowCheck(self, other);
    DataType selfType = self.GetDataType();
    DataType otherType = other.GetDataType();
    if (IsInteger(selfType) && IsInteger(otherType)) {
        CheckTensorsDataTypeConsistency(self.GetStorage(), other.GetStorage(), "Pow");
        auto [result, op] = TensorBinaryOperationWithOp<BinaryOpType::POW>(*Program::GetInstance().GetCurrentFunction(),
                                                                           self.GetStorage(), other.GetStorage());
        op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(PrecisionType::INTRINSIC));
        return result;
    }
    DataType realResultType = GetPowRealResultDataType(selfType, otherType);
    DataType calcResultType = GetPowCalcResultDataType(selfType, otherType);
    auto selfSt = CastToResultType(self.GetStorage(), selfType, calcResultType);
    auto otherSt = CastToResultType(other.GetStorage(), otherType, calcResultType);
    auto [result, op] = TensorBinaryOperationWithOp<BinaryOpType::POW>(*Program::GetInstance().GetCurrentFunction(),
                                                                       selfSt, otherSt);
    op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    if (realResultType != calcResultType) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), result,
                    realResultType, CastMode::CAST_NONE);
    }
    return result;
}

LogicalTensorPtr IntegerPow(const Tensor& self, int32_t intExponent)
{
    // 快速幂
    auto result = GenAllOneTensor(self.GetShape(), self.GetStorage()->GetDynValidShape(), self.GetDataType());
    auto current = CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), self, result);

    while (intExponent != NUM_VALUE_0) {
        if (intExponent % NUM_VALUE_2 != NUM_VALUE_0) {
            result = CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), result,
                          current);
        }
        current = CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), current,
                       current);
        intExponent /= NUM_VALUE_2;
    }
    return result;
}

LogicalTensorPtr GeneralPow(const Tensor& self, double exponent)
{
    // 如果指数小于0，先计算a^(-b)，最后再取倒数
    bool expLessThanZero = exponent < NUM_VALUE_0;
    exponent = std::abs(exponent);

    LogicalTensorPtr result;
    int32_t intExponent = static_cast<int32_t>(std::floor(exponent));
    if (exponent - intExponent < NUM_VALUE_EPS) {
        result = IntegerPow(self, intExponent);
    } else {
        auto lnSelf = CALL(UnaryOperation<UnaryOpType::LN>, *Program::GetInstance().GetCurrentFunction(),
                           self.GetStorage());
        auto exponentLnSelf = CALL(BinaryOperationScalar<BinaryOpType::MUL>,
                                   *Program::GetInstance().GetCurrentFunction(), lnSelf,
                                   Element(DataType::DT_FP32, exponent));
        result = CALL(UnaryOperation<UnaryOpType::EXP>, *Program::GetInstance().GetCurrentFunction(), exponentLnSelf);
    }

    // 指数小于零，结果取倒数
    if (expLessThanZero) {
        auto oneTensor = GenAllOneTensor(self.GetShape(), self.GetStorage()->GetDynValidShape(), self.GetDataType());
        // 求倒数
        RETURN_CALL(BinaryOperation<BinaryOpType::DIV>, *Program::GetInstance().GetCurrentFunction(), oneTensor,
                    result);
    }
    return result;
}

Tensor Pow(const Tensor& self, const Element& other)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Pow");

    LogicalTensorPtr castSelf = self.GetStorage();
    if ((self.GetDataType() == DT_INT32 || self.GetDataType() == DT_INT16) && other.GetDataType() != DT_INT32) {
        castSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), castSelf,
                        DataType::DT_FP32, CastMode::CAST_NONE);
    }
    double exponent = other.Cast<double>();
    // 指数为0，输出全1
    if (std::abs(exponent) < NUM_VALUE_EPS) {
        return GenAllOneTensor(self.GetShape(), self.GetStorage()->GetDynValidShape(), self.GetDataType());
    }
    DataType dataType = castSelf->Datatype();
    bool shouldUpToFp32 = dataType == DT_FP16 || dataType == DT_BF16;
    if (shouldUpToFp32) {
        castSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), castSelf,
                        DataType::DT_FP32, CastMode::CAST_NONE);
    }
    auto result = castSelf;
    if (std::abs(exponent - NUM_VALUE_0_5) < NUM_VALUE_EPS) {
        result = CALL(UnaryOperation<UnaryOpType::SQRT>, *Program::GetInstance().GetCurrentFunction(), result);
    } else if (std::abs(exponent - NUM_VALUE_2) < NUM_VALUE_EPS) {
        result = CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), result, result);
    } else if (std::abs(exponent - NUM_VALUE_3) < NUM_VALUE_EPS) {
        auto doubleSelf = CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), result,
                               result);
        result = CALL(BinaryOperation<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(), doubleSelf,
                      result);
    } else {
        result = GeneralPow(result, exponent);
    }
    if (shouldUpToFp32) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), result, dataType,
                    CastMode::CAST_NONE);
    }
    return result;
}

REGISTER_OPERATION_TILED_FUNC(OP_EXP, Opcode::OP_EXP, ExpOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_EXP2, Opcode::OP_EXP2, Exp2OperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_EXPM1, Opcode::OP_EXPM1, Expm1OperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_EXPANDEXPDIF, Opcode::OP_EXPANDEXPDIF, ExpandExpDifOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_POW, Opcode::OP_POW, BinaryOperationTileFunc<BinaryOpType::POW>);

} // namespace npu::tile_fwk
