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
 * \file trigonometric.cpp
 * \brief
 */

#include "unary_tiled.h"
#include "binary.h"
#include "binary_tiled.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

Tensor Atan(const Tensor& self)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Atan");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_FP32, DT_BF16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Atan");
    auto castSelf = self.GetStorage();
    if (self.GetDataType() != DataType::DT_FP32) {
        castSelf = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                        self.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    }
    auto res = CALL(UnaryOperation<UnaryOpType::ATAN>, *Program::GetInstance().GetCurrentFunction(), castSelf);
    if (self.GetDataType() != DataType::DT_FP32) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), res,
                    self.GetDataType(), CastMode::CAST_NONE);
    }
    return res;
}

Tensor Atan2(const Tensor& y, const Tensor& x)
{
    DECLARE_TRACER();
    CheckVectorOperand(y.GetStorage(), "Atan2");
    CheckVectorOperand(x.GetStorage(), "Atan2");

    CheckTensorsDataTypeConsistency(y.GetStorage(), x.GetStorage(), "ATAN2");
    std::unordered_set<DataType> supportedTypes = {DT_FP32, DT_FP16, DT_BF16};
    CheckTensorDataType(y.GetStorage(), supportedTypes, "ATAN2");
    auto castY = y.GetStorage();
    auto castX = x.GetStorage();
    DataType dataType = y.GetDataType();
    if (dataType != DataType::DT_FP32) {
        castY = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), y.GetStorage(),
                     DataType::DT_FP32, CastMode::CAST_NONE);
        castX = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), x.GetStorage(),
                     DataType::DT_FP32, CastMode::CAST_NONE);
    }
    auto res = CALL(BinaryOperation<BinaryOpType::ATAN2>, *Program::GetInstance().GetCurrentFunction(), castY, castX);
    if (dataType != DataType::DT_FP32) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), res, dataType,
                    CastMode::CAST_NONE);
    }
    return res;
}

Tensor Sin(const Tensor& self)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Sin");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Sin");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "Sin");

    auto castSelf = Cast(self, DataType::DT_FP32);
    auto result = CALL(UnaryOperation<UnaryOpType::SIN>, *Program::GetInstance().GetCurrentFunction(),
                       castSelf.GetStorage());
    auto castResult = Cast(result, self.GetDataType());
    return castResult;
}

Tensor Cos(const Tensor& self)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Cos");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Cos");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "Cos");

    auto castSelf = Cast(self, DataType::DT_FP32);
    auto result = CALL(UnaryOperation<UnaryOpType::COS>, *Program::GetInstance().GetCurrentFunction(),
                       castSelf.GetStorage());
    auto castResult = Cast(result, self.GetDataType());
    return castResult;
}

Tensor Asin(const Tensor& self)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Asin");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_FP32, DT_BF16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "ASIN");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "ASIN");

    auto castSelf = Cast(self, DataType::DT_FP32);
    auto result = CALL(UnaryOperation<UnaryOpType::ASIN>, *Program::GetInstance().GetCurrentFunction(),
                       castSelf.GetStorage());
    auto castResult = Cast(result, self.GetDataType());
    return castResult;
}

Tensor Acos(const Tensor& self)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Acos");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_FP32, DT_BF16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "ACOS");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "ACOS");

    auto castSelf = Cast(self, DataType::DT_FP32);
    auto result = CALL(UnaryOperation<UnaryOpType::ACOS>, *Program::GetInstance().GetCurrentFunction(),
                       castSelf.GetStorage());
    auto castResult = Cast(result, self.GetDataType());
    return castResult;
}

void TiledTanOperation(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                       const LogicalTensorPtr& result, DataType srcDtype)
{
    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);

        int64_t tmpSize = NUM_VALUE_9 * MultiplyLastTwoDims<float>(input.tileInfo.shape);
        std::vector<int64_t> tmpShape({tmpSize});
        auto tmpTensor = std::make_shared<LogicalTensor>(function, DT_FP32, tmpShape);
        function.AddOperation(Opcode::OP_TAN, {tile}, {resultTile, tmpTensor});

        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledTanOperation(function, tileShape, cur + 1, input, result, srcDtype);
    }
}

void TiledTanOperation(Function& function, const TileShape& tileShape, const LogicalTensorPtr& self,
                       const LogicalTensorPtr& result, DataType srcDtype)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, self->shape.size() == self->offset.size())
        << "Shape size and offset size should be equal";

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{self, tileInfo};
    TiledTanOperation(function, tileShape, 0, input, result, srcDtype);
}

LogicalTensorPtr TensorTanOperation(Function& function, LogicalTensorPtr self)
{
    auto srcDtype = self->tensor->datatype;
    LogicalTensorPtr operandCast = self;
    if (srcDtype == DataType::DT_FP16 || srcDtype == DataType::DT_BF16) {
        operandCast = TensorCastOperation<CastOpType::CAST>(function, self, DataType::DT_FP32, CastMode::CAST_NONE);
    }
    auto result = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, self->shape, self->GetDynValidShape());
    function.AddOperation(Opcode::OP_TAN, {operandCast}, {result});
    if (srcDtype == DataType::DT_FP16 || srcDtype == DataType::DT_BF16) {
        auto resultCast = TensorCastOperation<CastOpType::CAST>(function, result, srcDtype, CastMode::CAST_NONE);
        return resultCast;
    }
    return result;
}

Tensor Tan(const Tensor& operand)
{
    DECLARE_TRACER();
    CheckVectorOperand(operand.GetStorage(), "Tan");

    CheckTensorDimRange(operand.GetStorage(), 1, NUM_VALUE_4, "Tan");

    auto dType = operand.GetStorage()->Datatype();
    CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED,
          dType == DataType::DT_FP32 || dType == DataType::DT_FP16 || dType == DataType::DT_BF16)
        << "The datatype is not supported";
    RETURN_CALL(TanOperation, *Program::GetInstance().GetCurrentFunction(), operand.GetStorage());
}

void AtanOperationTileFunc(Function& function, const TileShape& tileShape,
                           const std::vector<LogicalTensorPtr>& iOperand, const std::vector<LogicalTensorPtr>& oOperand,
                           [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto shape = tileShape.GetVecTile();
    auto tmpShape = shape.tile;
    auto alignSize = BLOCK_SIZE / BytesOf(DT_FP32);
    tmpShape[shape.size() - 1] = AlignUp(tmpShape[shape.size() - 1], alignSize) * NUM3;
    auto workspace = BytesOf(DT_FP32) *
                     std::accumulate(tmpShape.begin(), tmpShape.end(), 1LL, std::multiplies<int64_t>());
    return TiledUnaryOperation<UnaryOpType::ATAN>(function, tileShape, iOperand[0], oOperand[0], workspace);
}

void SinOperationTileFunc(Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
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

    return TiledUnaryOperation<UnaryOpType::SIN>(function, tileShape, iOperand[0], oOperand[0], intermediateBytes);
}

void CosOperationTileFunc(Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
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

    return TiledUnaryOperation<UnaryOpType::COS>(function, tileShape, iOperand[0], oOperand[0], intermediateBytes);
}

template <UnaryOpType T>
void AsinAcosOperationTileFunc(Function& function, const TileShape& tileShape,
                               const std::vector<LogicalTensorPtr>& iOperand,
                               const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto shape = tileShape.GetVecTile().tile;
    int dim = static_cast<int>(shape.size());
    auto repeatElems = REPEAT_BYTE / BytesOf(DT_FP32);
    int64_t tmpW = (shape[dim - 1] + repeatElems - 1) / repeatElems * repeatElems;
    int64_t tmpH = (dim >= NUM_VALUE_2) ? shape[dim - NUM_VALUE_2] : 1;
    // 3 H x W tile and 1 mask tile
    uint64_t intermediateBytes = static_cast<uint64_t>(BytesOf(DT_FP32)) * NUM_VALUE_4 * tmpH * tmpW;
    return TiledUnaryOperation<T>(function, tileShape, iOperand[0], oOperand[0], intermediateBytes);
}

void TanOperationTileFunc(Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
                          const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    TiledTanOperation(function, tileShape, iOperand[0], oOperand[0], iOperand[0]->tensor->datatype);
}

REGISTER_OPERATION_TILED_FUNC(OP_ATAN, Opcode::OP_ATAN, AtanOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ATAN2, Opcode::OP_ATAN2, BinaryOperationTileFunc<BinaryOpType::ATAN2>);
REGISTER_OPERATION_TILED_FUNC(OP_SIN, Opcode::OP_SIN, SinOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_COS, Opcode::OP_COS, CosOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_ASIN, Opcode::OP_ASIN, AsinAcosOperationTileFunc<UnaryOpType::ASIN>);
REGISTER_OPERATION_TILED_FUNC(OP_ACOS, Opcode::OP_ACOS, AsinAcosOperationTileFunc<UnaryOpType::ACOS>);
REGISTER_OPERATION_TILED_FUNC(OP_TAN, Opcode::OP_TAN, TanOperationTileFunc);

} // namespace npu::tile_fwk
