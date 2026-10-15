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
 * \file logical.cpp
 * \brief
 */

#include "unary.h"
#include "binary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"
#include "tilefwk/error_code.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {

void TiledLogicalNotOperation(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                              const LogicalTensorPtr& result)
{
    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);

        constexpr int64_t COUNT_NUM = 2048;
        constexpr int64_t vcmp_bit_size = COUNT_NUM / NUM_VALUE_8;
        constexpr size_t ALIGN_SIZE = NUM_VALUE_32;

        DataType select_dtype;
        if (input.tensor.GetDataType() == DT_FP32 || input.tensor.GetDataType() == DT_BF16) {
            select_dtype = DT_FP32;
        } else if (input.tensor.GetDataType() == DT_INT16 || input.tensor.GetDataType() == DT_UINT16) {
            select_dtype = DT_INT16;
        } else if (input.tensor.GetDataType() == DT_INT32 || input.tensor.GetDataType() == DT_UINT32) {
            select_dtype = DT_INT32;
        } else if (input.tensor.GetDataType() == DT_INT64 || input.tensor.GetDataType() == DT_UINT64) {
            select_dtype = DT_INT64;
        } else {
            select_dtype = DT_FP16;
        }

        int64_t total_size;
        if (input.tensor.GetDataType() == DT_INT16 || input.tensor.GetDataType() == DT_UINT16 ||
            input.tensor.GetDataType() == DT_INT32 || input.tensor.GetDataType() == DT_UINT32 ||
            input.tensor.GetDataType() == DT_INT64 || input.tensor.GetDataType() == DT_UINT64) {
            // New integer path: one buffer of 2048 elements at compute width
            // For int64, two-step gather reuses the same buffer
            total_size = COUNT_NUM * BytesOf(select_dtype);
        } else {
            // Existing path: vcmpBitResult + compareCondition + oneCondition + castCondition + startAddrUB
            total_size = COUNT_NUM * NUM_VALUE_2 + COUNT_NUM * BytesOf(select_dtype) * NUM_VALUE_2 + vcmp_bit_size +
                         NUM_VALUE_8;
        }
        total_size = (total_size + ALIGN_SIZE - 1) / ALIGN_SIZE * ALIGN_SIZE;
        std::vector<int64_t> tmpShape({total_size});

        auto tmpTensor = std::make_shared<LogicalTensor>(function, DT_INT8, tmpShape);
        auto& op = function.AddOperation(Opcode::OP_LOGICALNOT, {tile}, {resultTile, tmpTensor});
        if (input.tensor.GetDataType() == DT_FP32 || input.tensor.GetDataType() == DT_BF16 ||
            input.tensor.GetDataType() == DT_FP16) {
            std::vector<bool> dimMap({true});
            op.SetAttr(OpAttributeKey::rowPad, dimMap);
        }
        return;
    }

    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledLogicalNotOperation(function, tileShape, cur + 1, input, result);
    }
}

void TiledLogicalNotOperation(Function& function, const TileShape& tileShape, const LogicalTensorPtr& self,
                              const LogicalTensorPtr& result)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, self->shape.size() == self->offset.size())
        << "Shape size and offset size should be equal";

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{self, tileInfo};
    TiledLogicalNotOperation(function, tileShape, 0, input, result);
}

LogicalTensorPtr TensorLogicalNotOperation(Function& function, LogicalTensorPtr self)
{
    auto result = std::make_shared<LogicalTensor>(function, DT_BOOL, self->shape, self->GetDynValidShape());
    function.AddOperation(Opcode::OP_LOGICALNOT, {self}, {result});
    return result;
}

Tensor LogicalNot(const Tensor& self)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "LogicalNot");

    const auto& combinations = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_LOGICALNOT);
    CheckTensorDataTypeCombination({self.GetDataType()}, combinations, "LOGICALNOT");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "LOGICALNOT");
    CheckTensorShapeSize(self.GetStorage(), "LOGICALNOT");
    RETURN_CALL(LogicalNotOperation, *Program::GetInstance().GetCurrentFunction(), self.GetStorage());
}

void TiledLogicalAndOperation(Function& function, const TileShape& tileShape, size_t cur, Input& input0, Input& input1,
                              const LogicalTensorPtr& result, TileInfo& resultTileInfo)
{
    if (cur == input0.tensor.GetShape().size()) {
        auto tile0 = input0.tensor.GetStorage()->View(function, input0.tileInfo.shape, input0.tileInfo.offset);
        auto tile1 = input1.tensor.GetStorage()->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);

        constexpr size_t ALIGN_SIZE = NUM_VALUE_32;
        auto needsFloat = [](DataType dtype) { return dtype == DT_FP32 || dtype == DT_BF16 || dtype == DT_INT32; };
        auto compute_dtype = needsFloat(input0.tensor.GetDataType()) || needsFloat(input1.tensor.GetDataType()) ?
                                 DT_FP32 :
                                 DT_FP16;
        const auto& vecTile = tileShape.GetVecTile().tile;
        auto tileDims = vecTile.size();
        int64_t tileH = tileDims >= NUM_VALUE_2 ? vecTile[tileDims - NUM_VALUE_2] : 1;
        // Keep the row strides and shared mask/conversion storage in sync with TLogicalAnd.
        int64_t tmpW = AlignUp(vecTile.back(), ALIGN_SIZE);
        auto compareElems = REPEAT_BYTE / BytesOf(DT_FP16);
        int64_t bufferElems = tileH * tmpW + AlignUp(tmpW, compareElems) - tmpW;
        int64_t maskW = AlignUp((tmpW + NUM_VALUE_7) / NUM_VALUE_8, ALIGN_SIZE);
        int64_t compute_array_size = bufferElems * BytesOf(compute_dtype);
        int64_t half_array_size = compute_dtype == DT_FP32 ? bufferElems * BytesOf(DT_FP16) : 0;
        int64_t total_bytes = NUM_VALUE_2 * compute_array_size + std::max(tileH * maskW, half_array_size) + ALIGN_SIZE;
        if (!ConfigManager::Instance().GetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true)) {
            // The legacy CCE implementation still uses fixed 64-element buffers.
            total_bytes = NUM_VALUE_64 * (NUM_VALUE_4 * BytesOf(DT_FP32) + BytesOf(DT_FP16)) + NUM_VALUE_3 * ALIGN_SIZE;
        }
        std::vector<int64_t> tmp_shape({static_cast<int64_t>(total_bytes)});
        auto tmp_tensor = std::make_shared<LogicalTensor>(function, DT_UINT8, tmp_shape);

        function.AddOperation(Opcode::OP_LOGICALAND, {tile0, tile1}, {resultTile, tmp_tensor});
        return;
    }

    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        input0.tileInfo.offset[cur] = i % input0.tensor.GetShape()[cur];
        input1.tileInfo.offset[cur] = i % input1.tensor.GetShape()[cur];
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        input0.tileInfo.shape[cur] = std::min(input0.tensor.GetShape()[cur] - input0.tileInfo.offset[cur],
                                              vecTile[cur]);
        input1.tileInfo.shape[cur] = std::min(input1.tensor.GetShape()[cur] - input1.tileInfo.offset[cur],
                                              vecTile[cur]);
        TiledLogicalAndOperation(function, tileShape, cur + 1, input0, input1, result, resultTileInfo);
    }
}

void TiledLogicalAndOperation(Function& function, const TileShape& tileShape, LogicalTensorPtr operand0,
                              LogicalTensorPtr operand1, const LogicalTensorPtr& result)
{
    BroadcastOperandTensor(operand0, operand1, result, function, tileShape);
    BroadcastOperandTensor(operand1, operand0, result, function, tileShape);

    TileInfo tileInfo0(result->shape.size(), result->offset.size());
    TileInfo tileInfo1(result->shape.size(), result->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input0 = Input{operand0, tileInfo0};
    auto input1 = Input{operand1, tileInfo1};
    TiledLogicalAndOperation(function, tileShape, 0, input0, input1, result, resultTileInfo);
}

LogicalTensorPtr TensorLogicalAndOperation(Function& function, const Tensor& self, const Tensor& other)
{
    auto operandT0 = self.GetStorage();
    auto operandT1 = other.GetStorage();
    if (operandT0->shape.size() != operandT1->shape.size()) {
        std::vector<int> broadCastShape = GetBroadCastShape(operandT0, operandT1);
        operandT0 = BinaryOperationBroadCast(operandT0, broadCastShape);
        operandT1 = BinaryOperationBroadCast(operandT1, broadCastShape);
    }

    std::vector<SymbolicScalar> resultValidShape;
    std::vector<int64_t> resultShape = BinaryOperationResultShape(operandT0, operandT1);
    if ((!operandT0->GetDynValidShape().empty()) && (!operandT1->GetDynValidShape().empty())) {
        for (size_t i = 0; i < resultShape.size(); ++i) {
            if (resultShape[i] == operandT0->shape[i]) {
                resultValidShape.push_back(operandT0->GetDynValidShape()[i]);
            } else {
                resultValidShape.push_back(operandT1->GetDynValidShape()[i]);
            }
        }
    }

    auto result = std::make_shared<LogicalTensor>(function, DT_BOOL, resultShape, resultValidShape);
    function.AddOperation(Opcode::OP_LOGICALAND, {operandT0, operandT1}, {result});
    return result;
}

Tensor LogicalAnd(const Tensor& self, const Tensor& other)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "LogicalAnd");
    CheckTensorFormat(other.GetStorage(), {TileOpFormat::TILEOP_NZ}, "LogicalAnd");

    std::unordered_set<DataType> supportedTypes = {DT_FP32,  DT_FP16, DT_BF16,  DT_INT8,
                                                   DT_UINT8, DT_BOOL, DT_INT16, DT_INT32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "LOGICALAND");
    CheckTensorDataType(other.GetStorage(), supportedTypes, "LOGICALAND");
    CheckBinaryInputTensors(self.GetStorage(), other.GetStorage(), "LOGICALAND");
    RETURN_CALL(LogicalAndOperation, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(),
                other.GetStorage());
}

void LogicNotOperationTileFunc(Function& function, const TileShape& tileShape,
                               const std::vector<LogicalTensorPtr>& iOperand,
                               const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    TiledLogicalNotOperation(function, tileShape, iOperand[0], oOperand[0]);
}

void LogicAndOperationTileFunc(Function& function, const TileShape& tileShape,
                               const std::vector<LogicalTensorPtr>& iOperand,
                               const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    TiledLogicalAndOperation(function, tileShape, iOperand[0], iOperand[1], oOperand[0]);
}

REGISTER_OPERATION_TILED_FUNC(OP_LOGICALNOT, Opcode::OP_LOGICALNOT, LogicNotOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_LOGICALAND, Opcode::OP_LOGICALAND, LogicAndOperationTileFunc);

} // namespace npu::tile_fwk
