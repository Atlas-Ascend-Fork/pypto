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
 * \\file gather_mask.cpp
 * \\brief
 */

#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/operation/operation_common.h"
#include "interface/operation/vector/gather_mask_common.h"
#include "tensor_transformation.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

void CheckGatherMaskViewShape(const Tensor& input, const std::vector<int64_t>& shape, uint8_t patternMode)
{
    if (patternMode < static_cast<uint8_t>(GatherMaskPattern::P0101) ||
        patternMode > static_cast<uint8_t>(GatherMaskPattern::P1000)) {
        return;
    }
    const auto& producers = input.GetStorage()->GetProducers();
    for (auto* op : producers) {
        if (op->GetOpcode() == Opcode::OP_VIEW) {
            const auto& inputShape = op->GetIOperands()[0]->GetShape();
            if (inputShape.size() == shape.size() && inputShape.back() > 0) {
                CHECK(VectorErrorCode::ERR_PARAM_INVALID, shape.back() >= inputShape.back())
                    << "GatherMask requires the last axis of self.shape not to be split by view. "
                    << "self.shape last axis: " << inputShape.back() << ", viewshape last axis: " << shape.back();
            }
        }
    }
}

int64_t UpdateGatherMaskShape(std::vector<int64_t>& shape, const VecTile& vecTile, GatherMaskPattern pattern)
{
    int64_t divisor = 1;
    if (IsHalfGatherMaskPattern(pattern)) {
        divisor = NUM_VALUE_2;
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, shape.back() % divisor == 0)
            << "The last axis of input shape should be divisible by 2 when patternMode is 1 or 2";
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, vecTile.tile.back() % divisor == 0)
            << "The last axis of tileshape should be divisible by 2 when patternMode is 1 or 2";
    } else if (IsQuarterGatherMaskPattern(pattern)) {
        divisor = NUM_VALUE_4;
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, shape.back() % divisor == 0)
            << "The last axis of input shape should be divisible by 4 when patternMode is 3, 4, 5 or 6";
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, vecTile.tile.back() % divisor == 0)
            << "The last axis of tileshape should be divisible by 4 when patternMode is 3, 4, 5 or 6";
    } else {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, pattern == GatherMaskPattern::P1111)
            << "Just support patternMode is 1, 2, 3, 4, 5, 6, 7";
    }
    shape.back() = shape.back() / divisor;
    return divisor;
}

void UpdateGatherMaskValidShape(Tensor& result, const Tensor& input, int64_t divisor)
{
    if (input.GetStorage()->GetDynValidShape().empty()) {
        return;
    }
    std::vector<SymbolicScalar> outValidShape;
    for (auto dim : input.GetStorage()->GetDynValidShape()) {
        outValidShape.push_back(dim);
    }
    outValidShape.back() = outValidShape.back() / divisor;
    result.GetStorage()->UpdateDynValidShape(outValidShape);
}

void TensorGatherMask(Function& function, const LogicalTensorPtr& self, const LogicalTensorPtr& result,
                      const uint8_t& patternMode)
{
    if (patternMode != 0) {
        auto& op = function.AddOperation(Opcode::OP_GATHER_MASK_BUILDIN, {self}, {result});
        op.SetAttribute(OP_ATTR_PREFIX + "patternMode", patternMode);
        return;
    }
}

Tensor GatherMask(const Tensor& self, const uint8_t patternMode)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "GatherMask");

    std::unordered_set<DataType> supportedTypes = {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, DT_UINT16, DT_UINT32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "GATHERMASK");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "GATHERMASK");
    auto shape = self.GetShape();
    CheckGatherMaskViewShape(self, shape, patternMode);
    const auto pattern = static_cast<GatherMaskPattern>(patternMode);
    auto divisor = UpdateGatherMaskShape(shape, TileShape::Current().GetVecTile(), pattern);
    auto result = Tensor(self.GetDataType(), shape);
    UpdateGatherMaskValidShape(result, self, divisor);
    CALL(GatherMask, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), result.GetStorage(), patternMode);
    return result;
}

void TiledGatherMaskBuildIn(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                            const LogicalTensorPtr& result, TileInfo& resultTileInfo, const uint8_t patternMode)
{
    const auto pattern = static_cast<GatherMaskPattern>(patternMode);
    if (cur == input.tensor.GetShape().size()) {
        auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto& op = function.AddOperation(Opcode::OP_GATHER_MASK, {inputTile}, {resultTile});
        op.SetAttribute(OP_ATTR_PREFIX + "patternMode", patternMode);
        return;
    }

    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        // update input && result && resultDices shape and offset info
        input.tileInfo.offset[cur] = i % input.tensor.GetShape()[cur];
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - input.tileInfo.offset[cur], vecTile[cur]);

        if ((cur == input.tensor.GetShape().size() - 1) && IsHalfGatherMaskPattern(pattern)) {
            resultTileInfo.offset[cur] = i / NUM_VALUE_2;
            resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur],
                                                 vecTile[cur] / NUM_VALUE_2);
        } else if ((cur == input.tensor.GetShape().size() - 1) && IsQuarterGatherMaskPattern(pattern)) {
            resultTileInfo.offset[cur] = i / NUM_VALUE_4;
            resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur],
                                                 vecTile[cur] / NUM_VALUE_4);
        } else {
            resultTileInfo.offset[cur] = i;
            resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        }
        TiledGatherMaskBuildIn(function, tileShape, cur + 1, input, result, resultTileInfo, patternMode);
    }
}

void TiledGatherMaskBuildIn(Function& function, const TileShape& tileShape, const LogicalTensorPtr operand,
                            const LogicalTensorPtr resOperand, const uint8_t patternMode)
{
    TileInfo tileInfo(operand->shape.size(), operand->offset.size());
    TileInfo resultTileInfo(resOperand->shape.size(), resOperand->offset.size());
    tileInfo.shape = operand->shape;
    resultTileInfo.shape = resOperand->shape;
    auto input = Input{operand, tileInfo};
    TiledGatherMaskBuildIn(function, tileShape, 0, input, resOperand, resultTileInfo, patternMode);
}

void GatherMaskBuildInOperationTileFunc(Function& function, const TileShape& tileShape,
                                        const std::vector<LogicalTensorPtr>& iOperand,
                                        const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    uint8_t patternMode = op.GetIntAttribute(OP_ATTR_PREFIX + "patternMode");
    if (patternMode >= 1 && patternMode <= NUM_VALUE_6) {
        const auto& viewShape = iOperand[0]->GetShape();
        std::vector<int64_t> originalShape;
        if (iOperand[0]->GetAttr("ORIGINAL_SHAPE", originalShape) && !originalShape.empty() &&
            originalShape.size() == viewShape.size() && originalShape.back() > 0 && viewShape.back() > 0) {
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, viewShape.back() >= originalShape.back())
                << "GatherMask requires the last axis of self.shape not to be split by view. "
                << "self.shape last axis: " << originalShape.back() << ", viewshape last axis: " << viewShape.back();
        }
    }
    TiledGatherMaskBuildIn(function, tileShape, iOperand[0], oOperand[0], patternMode);
}

REGISTER_OPERATION_TILED_FUNC(OP_GATHER_MASK_BUILDIN, Opcode::OP_GATHER_MASK_BUILDIN,
                              GatherMaskBuildInOperationTileFunc);

} // namespace npu::tile_fwk
