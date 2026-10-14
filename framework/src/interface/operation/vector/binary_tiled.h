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
 * \file binary_tiled.h
 * \brief
 */
#pragma once

#include "tilefwk/data_type.h"
#include "binary.h"
#include "tensor_transformation.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/error_code.h"
#include "passes/tile_graph_pass/graph_constraint/axis_combine.h"

namespace npu::tile_fwk {

template <BinaryOpType T>
void AddTiledBinaryOperation(Function& function, const TileShape& tileShape, LogicalInput& input1, LogicalInput& input2,
                             const LogicalTensorPtr& result, TileInfo& resultTileInfo, size_t shapeSize,
                             [[maybe_unused]] int64_t precisionType)
{
    auto inputTile1 = input1.tensor->View(function, input1.tileInfo.shape, input1.tileInfo.offset);
    auto inputTile2 = input2.tensor->View(function, input2.tileInfo.shape, input2.tileInfo.offset);
    auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
    auto opName = GetBinaryOpName<T>();
    Operation* op = nullptr;
    if (opName == "BITWISEXOR" || opName == "COPYSIGN" || opName == "POW") {
        std::vector<int64_t> tmpShape(resultTileInfo.shape);
        auto alignSize = BLOCK_SIZE / BytesOf(result->Datatype());
        auto alignedLast = AlignUp(tmpShape[resultTileInfo.shape.size() - 1], alignSize);
        if (opName == "POW" && result->Datatype() == DT_FP32 &&
            Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_2201) {
            auto maskColsFloat = AlignUp((alignedLast + NUM_VALUE_8 - 1) / NUM_VALUE_8, BLOCK_SIZE) /
                                 BytesOf(result->Datatype());
            alignedLast = NUM_VALUE_2 * alignedLast + NUM_VALUE_3 * maskColsFloat;
        }
        tmpShape[resultTileInfo.shape.size() - 1] = alignedLast;
        auto tempTensor = std::make_shared<LogicalTensor>(function, result->Datatype(), tmpShape);
        op = &function.AddOperation(GetBinaryOpNameCode<T, false, false>(), {inputTile1, inputTile2},
                                    {resultTile, tempTensor});
    } else if (opName == "REM") {
        int64_t alignSize = static_cast<int64_t>(NUM_VALUE_256 / BytesOf(result->Datatype()));
        int64_t alignedLast = AlignUp(resultTileInfo.shape[shapeSize - 1], alignSize);
        int64_t maskBytes = (alignedLast + NUM_VALUE_8 - 1) / NUM_VALUE_8;
        int64_t maskFloats = AlignUp(maskBytes, NUM_VALUE_256) / static_cast<int64_t>(BytesOf(result->Datatype()));
        int64_t tmpCols = alignedLast + maskFloats + alignSize;
        std::vector<int64_t> tmpShape{1, tmpCols};
        auto tmpTensor = std::make_shared<LogicalTensor>(function, result->Datatype(), tmpShape);
        op = &function.AddOperation(GetBinaryOpNameCode<T, false, false>(), {inputTile1, inputTile2},
                                    {resultTile, tmpTensor});
    } else if (opName == "ATAN2") {
        std::vector<int64_t> tmpShape(resultTileInfo.shape);
        auto lastDim = tmpShape[resultTileInfo.shape.size() - 1];
        auto alignSize = BLOCK_SIZE / BytesOf(result->Datatype());
        tmpShape[resultTileInfo.shape.size() - 1] = AlignUp(lastDim, alignSize) * NUM_VALUE_4 * BytesOf(DT_FP32);
        std::vector<SymbolicScalar> tmpValidShape;
        for (auto s : tmpShape) {
            tmpValidShape.emplace_back(s);
        }
        LogicalTensorPtr workspace = std::make_shared<LogicalTensor>(function, DT_UINT8, tmpShape, tmpValidShape);
        op = &function.AddOperation(GetBinaryOpNameCode<T, false, false>(), {inputTile1, inputTile2},
                                    {resultTile, workspace});
    } else if (opName == "FLOORDIV") {
        auto tmpShape = resultTileInfo.shape;
        auto alignSize = BLOCK_SIZE / BytesOf(result->Datatype());
        tmpShape.back() = AlignUp(tmpShape.back(), alignSize);
        tmpShape.front() *= NUM_VALUE_6;
        auto tmpType = result->Datatype() == DT_INT64 ? DT_INT64 : DT_FP32;
        auto tempTensor = std::make_shared<LogicalTensor>(function, tmpType, tmpShape);
        op = &function.AddOperation(GetBinaryOpNameCode<T, false, false>(), {inputTile1, inputTile2},
                                    {resultTile, tempTensor});
    } else if (opName == "GCD") {
        auto vecTile = tileShape.GetVecTile().tile;
        int64_t tileW = vecTile.empty() ? 1 : vecTile.back();
        int64_t tileH = vecTile.size() > 1 ? vecTile[vecTile.size() - NUM_VALUE_2] : 1;
        int64_t tileFootprint = tileH * tileW;
        int64_t maskCols = ((tileW + NUM_VALUE_7) / NUM_VALUE_8 + NUM_VALUE_31) / BLOCK_SIZE * BLOCK_SIZE;
        int64_t intermediateBytes = (NUM_VALUE_6 * tileFootprint * BytesOf(DT_INT32)) +
                                    (NUM_VALUE_2 * tileFootprint * BytesOf(DT_FP32)) + NUM_VALUE_4 * tileH * maskCols +
                                    GCD_TSEL_TMP_BYTES;
        Shape tempShape(resultTile->shape.size(), 1);
        tempShape[tempShape.size() - 1] = intermediateBytes;
        auto tempTensor = std::make_shared<LogicalTensor>(function, DT_UINT8, tempShape);
        op = &function.AddOperation(GetBinaryOpNameCode<T, false, false>(), {inputTile1, inputTile2},
                                    {resultTile, tempTensor});
    } else {
        op = &function.AddOperation(GetBinaryOpNameCode<T, false, false>(), {inputTile1, inputTile2}, {resultTile});
    }

    if (op != nullptr) {
        std::vector<int64_t> brcOperand(shapeSize, 0);
        size_t brcAxesCount = 0;
        for (size_t i = 0; i < shapeSize; i++) {
            brcOperand[i] = BrcAxisBinaryOp(input1.tensor, input2.tensor, i);
            if (brcOperand[i] != 0) {
                brcAxesCount++;
            }
        }
        if (brcAxesCount > 0) {
            if (brcOperand[shapeSize - 1] != 0 && brcAxesCount >= NUM_VALUE_2) {
                op->SetAttribute(OpAttributeKey::excludeBufferReuse, true);
            }
            op->SetAttribute(OpAttributeKey::brcOperand, brcOperand);
        }
        if (opName == "POW" && result->Datatype() == DT_FP32 &&
            Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_2201) {
            std::vector<bool> dimMap(op->iOperand.size(), true);
            op->SetAttr(OpAttributeKey::rowPad, dimMap);
        }
    }
    if constexpr (T == BinaryOpType::DIV || T == BinaryOpType::MOD || T == BinaryOpType::POW ||
                  T == BinaryOpType::REM) {
        op->SetAttribute(OpAttributeKey::precisionType, precisionType);
    }
}

template <BinaryOpType T>
void TiledBinaryOperation(Function& function, const TileShape& tileShape, size_t cur, LogicalInput& input1,
                          LogicalInput& input2, const LogicalTensorPtr& result, TileInfo& resultTileInfo,
                          int64_t precisionType)
{
    size_t shapeSize = input1.tensor->GetShape().size();
    if (cur == shapeSize) {
        AddTiledBinaryOperation<T>(function, tileShape, input1, input2, result, resultTileInfo, shapeSize,
                                   precisionType);
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
        TiledBinaryOperation<T>(function, tileShape, cur + 1, input1, input2, result, resultTileInfo, precisionType);
    }
}

// Determine the target shape for expand before tileop
template <BinaryOpType T>
std::pair<std::vector<int64_t>, std::vector<int64_t>> GetBrcExpandShape(Function& function, LogicalTensorPtr operand1,
                                                                        LogicalTensorPtr operand2,
                                                                        LogicalTensorPtr result)
{
    auto operand1Shape = result->shape;
    auto operand2Shape = result->shape;
    size_t shapeSize = result->shape.size();

    bool isInWhiteList = SUPPORT_BRC_INLINE.count(GetBinaryOpNameCode<T>());
    bool isCombineAxisEnabled = function.paramConfigs_.combineAxis && isInWhiteList;
    if (isInWhiteList) {
        // Outer axis: handled by tileop loop with stride control, keep operand shape.
        if (shapeSize > NUM_VALUE_2) {
            for (size_t i = 0; i < shapeSize - NUM_VALUE_2; i++) {
                operand1Shape[i] = operand1->shape[i];
                operand2Shape[i] = operand2->shape[i];
            }
        }
        // The 2nd last axis: skip expand, brcinline
        if (shapeSize > 1) {
            operand1Shape[shapeSize - NUM_VALUE_2] = operand1->shape[shapeSize - NUM_VALUE_2];
            operand2Shape[shapeSize - NUM_VALUE_2] = operand2->shape[shapeSize - NUM_VALUE_2];
        }
        // The last axis: brcinline when combineAxis is enabled
        if (shapeSize > 0 && isCombineAxisEnabled) {
            operand1Shape[shapeSize - 1] = operand1->shape[shapeSize - 1];
            operand2Shape[shapeSize - 1] = operand2->shape[shapeSize - 1];
        }
    }
    return {operand1Shape, operand2Shape};
}

template <BinaryOpType T>
void TiledBinaryOperation(Function& function, const TileShape& tileShape, LogicalTensorPtr operand1,
                          LogicalTensorPtr operand2, const LogicalTensorPtr& result, int64_t precisionType)
{
    CheckBinOpOperandsValid(operand1, operand2);
    auto [dstShape1, dstShape2] = GetBrcExpandShape<T>(function, operand1, operand2, result);
    BroadcastOperandTensor(operand1, operand2, result, function, tileShape, dstShape1);
    BroadcastOperandTensor(operand2, operand1, result, function, tileShape, dstShape2);

    TileInfo tileInfo1(operand1->shape.size(), operand1->offset.size());
    TileInfo tileInfo2(operand2->shape.size(), operand2->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto input1 = LogicalInput{operand1, tileInfo1};
    auto input2 = LogicalInput{operand2, tileInfo2};
    TiledBinaryOperation<T>(function, tileShape, 0, input1, input2, result, resultTileInfo, precisionType);
}

template <BinaryOpType T>
void BinaryOperationTileFunc(Function& function, const TileShape& tileShape,
                             const std::vector<LogicalTensorPtr>& iOperand,
                             const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    BinaryOperationOperandCheck(iOperand, oOperand);
    int64_t precisionType = static_cast<int64_t>(PrecisionType::INTRINSIC);
    if constexpr (T == BinaryOpType::DIV || T == BinaryOpType::MOD || T == BinaryOpType::POW ||
                  T == BinaryOpType::REM) {
        precisionType = GetOperationPrecisionType(op);
    }
    TiledBinaryOperation<T>(function, tileShape, iOperand[0], iOperand[1], oOperand[0], precisionType);
}

} // namespace npu::tile_fwk
