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
 * \file unary_basic.cpp
 * \brief
 */

#include <cmath>
#include "unary_tiled.h"
#include "binary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"
#include "tilefwk/error_code.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {

Tensor Rsqrt(const Tensor& self, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Rsqrt");

    const auto& combinations = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_RSQRT);
    CheckTensorDataTypeCombination({self.GetDataType()}, combinations, "Rsqrt");

    auto castSelf = self.GetStorage();
    if (self.GetDataType() != DataType::DT_FP32) {
        castSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                        self.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    }
    auto [sqrtResult, sqrtOp] = TensorUnaryOperationWithOp<UnaryOpType::SQRT>(
        *Program::GetInstance().GetCurrentFunction(), castSelf);
    sqrtOp->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    auto ones = CALL(FullOperation, *Program::GetInstance().GetCurrentFunction(), Element(DataType::DT_FP32, 1.0),
                     SymbolicScalar(), DataType::DT_FP32, self.GetShape(), self.GetStorage()->GetDynValidShape());
    auto [divResult, divOp] = TensorBinaryOperationWithOp<BinaryOpType::DIV>(
        *Program::GetInstance().GetCurrentFunction(), ones, sqrtResult);
    divOp->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    if (self.GetDataType() != DataType::DT_FP32) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), divResult,
                    self.GetDataType(), CastMode::CAST_NONE);
    }
    return divResult;
}

Tensor Sqrt(const Tensor& self, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Sqrt");

    const auto& combinations = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_SQRT);
    CheckTensorDataTypeCombination({self.GetDataType()}, combinations, "Sqrt");

    auto [result, op] = TensorUnaryOperationWithOp<UnaryOpType::SQRT>(*Program::GetInstance().GetCurrentFunction(),
                                                                      self.GetStorage());
    op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    return Tensor(result);
}

Tensor Relu(const Tensor& self)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Relu");

    const auto& combinations = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_RELU);
    CheckTensorDataTypeCombination({self.GetDataType()}, combinations, "Relu");
    RETURN_CALL(UnaryOperation<UnaryOpType::RELU>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

Tensor Reciprocal(const Tensor& operand, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckVectorOperand(operand.GetStorage(), "Reciprocal");

    const auto& combinations = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_RECIPROCAL);
    CheckTensorDataTypeCombination({operand.GetDataType()}, combinations, "Reciprocal");

    auto [result, op] = TensorUnaryOperationWithOp<UnaryOpType::RECIPROCAL>(
        *Program::GetInstance().GetCurrentFunction(), operand.GetStorage());
    op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    return Tensor(result);
}

Tensor Abs(const Tensor& self)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Abs");

    const auto& combinations = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_ABS);
    CheckTensorDataTypeCombination({self.GetDataType()}, combinations, "Abs");
    RETURN_CALL(UnaryOperation<UnaryOpType::ABS>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

Tensor Hub(const Tensor& self)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Hub");

    RETURN_CALL(UnaryOperation<UnaryOpType::HUB>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

Tensor Neg(const Tensor& self)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Neg");

    const auto& combinations = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_NEG);
    CheckTensorDataTypeCombination({self.GetDataType()}, combinations, "NEG");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "NEG");

    if (IsFloat(self.GetStorage()->Datatype())) {
        RETURN_CALL(BinaryOperationScalar<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(),
                    self.GetStorage(), Element(self.GetStorage()->Datatype(), -1.0));
    } else {
        RETURN_CALL(BinaryOperationScalar<BinaryOpType::MUL>, *Program::GetInstance().GetCurrentFunction(),
                    self.GetStorage(), Element(self.GetStorage()->Datatype(), -1));
    }
}

Tensor Duplicate(const Tensor& operand)
{
    DECLARE_TRACER();
    CheckVectorOperand(operand.GetStorage(), "Duplicate");

    RETURN_CALL(UnaryOperation<UnaryOpType::DUPLICATE>, *Program::GetInstance().GetCurrentFunction(),
                operand.GetStorage());
}

Tensor IsFinite(const Tensor& self)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "IsFinite");

    std::unordered_set<DataType> supportedTypes = {DT_FP16,  DT_FP32,   DT_BF16,   DT_INT16, DT_INT4,   DT_INT8,
                                                   DT_INT32, DT_UINT16, DT_UINT32, DT_UINT8, DT_UINT64, DT_INT64};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "IsFinite");
    RETURN_CALL(UnaryOperation<UnaryOpType::ISFINITE>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(),
                DT_BOOL);
}

Tensor IsNan(const Tensor& self)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "IsNan");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_FP32, DT_BF16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "ISNAN");

    RETURN_CALL(UnaryOperation<UnaryOpType::ISNAN>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(),
                DT_BOOL);
}

Tensor Ceil(const Tensor& self)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Ceil");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Ceil");

    if (self.GetDataType() == DataType::DT_INT16 || self.GetDataType() == DataType::DT_INT32) {
        RETURN_CALL(UnaryOperation<UnaryOpType::CEIL>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
    }

    auto castSelf = self.GetStorage();
    if (self.GetDataType() != DataType::DT_FP32) {
        castSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                        self.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    }

    auto ceilResult = CALL(UnaryOperation<UnaryOpType::CEIL>, *Program::GetInstance().GetCurrentFunction(), castSelf);
    if (self.GetDataType() != DataType::DT_FP32) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), ceilResult,
                    self.GetDataType(), CastMode::CAST_NONE);
    }
    return ceilResult;
}

Tensor Floor(const Tensor& self)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Floor");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Floor");

    if (self.GetDataType() == DataType::DT_INT16 || self.GetDataType() == DataType::DT_INT32) {
        RETURN_CALL(UnaryOperation<UnaryOpType::FLOOR>, *Program::GetInstance().GetCurrentFunction(),
                    self.GetStorage());
    }

    auto castSelf = self.GetStorage();
    if (self.GetDataType() != DataType::DT_FP32) {
        castSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                        self.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    }

    auto floorResult = CALL(UnaryOperation<UnaryOpType::FLOOR>, *Program::GetInstance().GetCurrentFunction(), castSelf);
    if (self.GetDataType() != DataType::DT_FP32) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), floorResult,
                    self.GetDataType(), CastMode::CAST_NONE);
    }
    return floorResult;
}

Tensor Trunc(const Tensor& self)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Trunc");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Trunc");

    if (self.GetDataType() == DataType::DT_INT16 || self.GetDataType() == DataType::DT_INT32) {
        RETURN_CALL(UnaryOperation<UnaryOpType::TRUNC>, *Program::GetInstance().GetCurrentFunction(),
                    self.GetStorage());
    }

    auto castSelf = self.GetStorage();
    if (self.GetDataType() != DataType::DT_FP32) {
        castSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                        self.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    }

    auto truncResult = CALL(UnaryOperation<UnaryOpType::TRUNC>, *Program::GetInstance().GetCurrentFunction(), castSelf);
    if (self.GetDataType() != DataType::DT_FP32) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), truncResult,
                    self.GetDataType(), CastMode::CAST_NONE);
    }
    return truncResult;
}

Tensor TensorRound(Function& function, const LogicalTensorPtr& self, const int& decimals = 0)
{
    auto result = std::make_shared<LogicalTensor>(function, self->Datatype(), self->GetShape(),
                                                  self->GetDynValidShape());
    auto& op = function.AddOperation(Opcode::OP_ROUND, {self}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "decimals", decimals);
    return result;
}

Tensor Round(const Tensor& self, const int& decimals)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Round");

    const auto& combinations = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_ROUND);
    CheckTensorDataTypeCombination({self.GetDataType()}, combinations, "ROUND");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "ROUND");

    RETURN_CALL(Round, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), decimals);
}

void TiledRound(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                const LogicalTensorPtr& result, const int& decimals = 0)
{
    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);
        std::vector<int64_t> srcTileShape(input.tileInfo.shape);
        auto tileShapeLen = srcTileShape.size();
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, SHAPE_DIM1 <= tileShapeLen && tileShapeLen <= SHAPE_DIM4)
            << "Length of tile shape only supports 1~4";
        std::vector<int64_t> tmpShape;
        if (result->Datatype() == DT_FP32) {
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
        auto& newOp = function.AddOperation(Opcode::OP_ROUND, {tile}, {resultTile, tmpTensor});
        float powDecimals = std::pow(static_cast<float>(NUM_VALUE_10), static_cast<float>(decimals));
        const int32_t maxFp32Len = 38;
        if (decimals > maxFp32Len) {
            powDecimals = INFINITY;
        }
        newOp.SetAttribute(OP_ATTR_PREFIX + "decimals", decimals);
        newOp.SetAttribute(OpAttributeKey::scalar, Element(DataType::DT_FP32, powDecimals));
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledRound(function, tileShape, cur + 1, input, result, decimals);
    }
}

void TiledRound(Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand,
                const LogicalTensorPtr& result, const int& decimals = 0)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, operand->shape.size() == operand->offset.size())
        << "The shape size of operand and offset must be equal";

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};

    TiledRound(function, tileShape, 0, input, result, decimals);
}

void RsqrtOperationTileFunc(Function& function, const TileShape& tileShape,
                            const std::vector<LogicalTensorPtr>& iOperand,
                            const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::RSQRT>(function, tileShape, iOperand[0], oOperand[0]);
}

void ReluOperationTileFunc(Function& function, const TileShape& tileShape,
                           const std::vector<LogicalTensorPtr>& iOperand, const std::vector<LogicalTensorPtr>& oOperand,
                           [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::RELU>(function, tileShape, iOperand[0], oOperand[0]);
}

void SqrtOperationTileFunc(Function& function, const TileShape& tileShape,
                           const std::vector<LogicalTensorPtr>& iOperand, const std::vector<LogicalTensorPtr>& oOperand,
                           [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    int64_t precisionType = static_cast<int64_t>(PrecisionType::INTRINSIC);
    if (op.HasAttr(OpAttributeKey::precisionType)) {
        precisionType = op.GetIntAttribute(OpAttributeKey::precisionType);
    }
    return TiledUnaryOperation<UnaryOpType::SQRT>(function, tileShape, iOperand[0], oOperand[0], 0, precisionType);
}

void ReciprocalOperationTileFunc(Function& function, const TileShape& tileShape,
                                 const std::vector<LogicalTensorPtr>& iOperand,
                                 const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    int64_t precisionType = static_cast<int64_t>(PrecisionType::INTRINSIC);
    if (op.HasAttr(OpAttributeKey::precisionType)) {
        precisionType = op.GetIntAttribute(OpAttributeKey::precisionType);
    }
    return TiledUnaryOperation<UnaryOpType::RECIPROCAL>(function, tileShape, iOperand[0], oOperand[0], 0,
                                                        precisionType);
}

void AbsOperationTileFunc(Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
                          const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::ABS>(function, tileShape, iOperand[0], oOperand[0]);
}

void HubOperationTileFunc(Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
                          const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::HUB>(function, tileShape, iOperand[0], oOperand[0]);
}

void IsFiniteOperationTileFunc(Function& function, const TileShape& tileShape,
                               const std::vector<LogicalTensorPtr>& iOperand,
                               const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto shape = tileShape.GetVecTile().tile;
    // tileShape 对应的中间变量结果，类型为 FP16
    uint32_t intermediateBytes = static_cast<int64_t>(BytesOf(DT_FP16)) *
                                 std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
    uint32_t workspaceSize = intermediateBytes;
    return TiledUnaryOperation<UnaryOpType::ISFINITE>(function, tileShape, iOperand[0], oOperand[0], workspaceSize);
}

void IsNanOperationTileFunc(Function& function, const TileShape& tileShape,
                            const std::vector<LogicalTensorPtr>& iOperand,
                            const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto tmpShape = tileShape.GetVecTile().tile;
    int dim = static_cast<int>(tmpShape.size());
    auto alignSize = BLOCK_SIZE / BytesOf(DT_FP16);
    int64_t tmpW = AlignUp(tmpShape[dim - 1], alignSize);
    int64_t tmpH = (dim >= NUM_VALUE_2) ? tmpShape[dim - NUM_VALUE_2] : 1;

    constexpr int64_t kNumBlocks = NUM_VALUE_3;
    int64_t blockBytes = tmpH * tmpW * BytesOf(DT_FP32);
    uint32_t workspaceSize = kNumBlocks * blockBytes + BLOCK_SIZE;
    return TiledUnaryOperation<UnaryOpType::ISNAN>(function, tileShape, iOperand[0], oOperand[0], workspaceSize);
}

void CeilOperationTileFunc(Function& function, const TileShape& tileShape,
                           const std::vector<LogicalTensorPtr>& iOperand, const std::vector<LogicalTensorPtr>& oOperand,
                           [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::CEIL>(function, tileShape, iOperand[0], oOperand[0]);
}

void FloorOperationTileFunc(Function& function, const TileShape& tileShape,
                            const std::vector<LogicalTensorPtr>& iOperand,
                            const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::FLOOR>(function, tileShape, iOperand[0], oOperand[0]);
}

void TruncOperationTileFunc(Function& function, const TileShape& tileShape,
                            const std::vector<LogicalTensorPtr>& iOperand,
                            const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    return TiledUnaryOperation<UnaryOpType::TRUNC>(function, tileShape, iOperand[0], oOperand[0]);
}

void RoundOperationTileFunc(Function& function, const TileShape& tileShape,
                            const std::vector<LogicalTensorPtr>& iOperand,
                            const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    int decimals = op.GetIntAttribute(OP_ATTR_PREFIX + "decimals");
    TiledRound(function, tileShape, iOperand[0], oOperand[0], decimals);
}

REGISTER_OPERATION_TILED_FUNC(OP_RSQRT, Opcode::OP_RSQRT, RsqrtOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_RELU, Opcode::OP_RELU, ReluOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_SQRT, Opcode::OP_SQRT, SqrtOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_RECIPROCAL, Opcode::OP_RECIPROCAL, ReciprocalOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ABS, Opcode::OP_ABS, AbsOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_HUB, Opcode::OP_HUB, HubOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ISFINITE, Opcode::OP_ISFINITE, IsFiniteOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ISNAN, Opcode::OP_ISNAN, IsNanOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_CEIL, Opcode::OP_CEIL, CeilOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_FLOOR, Opcode::OP_FLOOR, FloorOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_TRUNC, Opcode::OP_TRUNC, TruncOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ROUND, Opcode::OP_ROUND, RoundOperationTileFunc);

} // namespace npu::tile_fwk
