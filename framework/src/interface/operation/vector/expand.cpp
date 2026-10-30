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
 * \file expand.cpp
 * \brief
 */

#include "unary.h"
#include <sstream>
#include <string>
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "tilefwk/error_code.h"
#include "tilefwk/platform.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {

struct ExpandInfo {
    const std::shared_ptr<LogicalTensor>& srcTensor;
    const std::shared_ptr<LogicalTensor>& result;
    std::vector<int64_t>& viewShape;
    std::vector<int64_t>& offset;
    const std::vector<int> expandDims;
    ExpandInfo(const std::shared_ptr<LogicalTensor>& srcTensor0, const std::shared_ptr<LogicalTensor>& result0,
               std::vector<int64_t>& viewShape0, std::vector<int64_t>& offset0, const std::vector<int> expandDims0)
        : srcTensor(srcTensor0), result(result0), viewShape(viewShape0), offset(offset0), expandDims(expandDims0)
    {}
};

void CheckExpandTensorValid(const LogicalTensorPtr& operand, const LogicalTensorPtr& result)
{
    const auto& operand_shape = operand->shape;
    const auto& result_shape = result->shape;

    if (operand_shape.size() != result_shape.size()) {
        std::ostringstream oss;
        oss << "The number of dimensions must match! "
            << "Operand shape: " << operand_shape.size() << "D (" << operand_shape << ") "
            << "Result shape: " << result_shape.size() << "D (" << result_shape << ")";
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, false) << oss.str();
    }

    for (size_t i = 0; i < result_shape.size(); ++i) {
        if (operand_shape[i] != result_shape[i] && operand_shape[i] != 1) {
            std::ostringstream oss;
            oss << "The size of tensor a (" << operand_shape[i] << ") must match the size of tensor b ("
                << result_shape[i] << ") at non-singleton dimension " << i << ". "
                << "Operand shape: (" << operand_shape << ") "
                << "Result shape: (" << result_shape << ")";
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, false) << oss.str();
        }
    }
}

void ExpandTile(Function& function, const struct ExpandInfo& expandInfo)
{
    auto resultTile = expandInfo.result->View(function, expandInfo.viewShape, expandInfo.offset);

    std::vector<int64_t> srcShape(expandInfo.srcTensor->shape.size(), 1);
    for (size_t i = 0; i < expandInfo.result->shape.size(); i++) {
        srcShape[i] = std::min(expandInfo.viewShape[i], expandInfo.srcTensor->shape[i]);
    }

    std::vector<int64_t> srcOffset = expandInfo.offset;
    for (size_t j = 0; j < srcOffset.size(); j++) {
        if (expandInfo.srcTensor->shape[j] < expandInfo.result->shape[j]) {
            srcOffset[j] = expandInfo.offset[j] % expandInfo.srcTensor->shape[j];
        }
    }
    auto srcTile = expandInfo.srcTensor->View(function, srcShape, srcOffset);
    auto& newOp = function.AddOperation("TILE_EXPAND", {srcTile}, {resultTile});
    newOp.SetAttribute(OpAttributeKey::expandDims, expandInfo.expandDims);
    newOp.SetAttribute(OP_ATTR_PREFIX + "validShape", resultTile->GetDynValidShape());
}

void ExpandTile(Function& function, const TileShape& tileShape, int dimIdx, const struct ExpandInfo& expandInfo,
                std::vector<SymbolicScalar> validShape)
{
    if (static_cast<size_t>(dimIdx) == expandInfo.result->shape.size()) {
        ExpandTile(function, expandInfo);
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < expandInfo.result->shape[dimIdx]; i += vecTile[dimIdx]) {
        expandInfo.offset[dimIdx] = i;
        expandInfo.viewShape[dimIdx] = std::min(expandInfo.result->shape[dimIdx] - i,
                                                static_cast<int64_t>(vecTile[dimIdx]));
        ExpandTile(function, tileShape, dimIdx + 1, expandInfo, validShape);
    }
}

void Expand(Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand,
            const std::vector<LogicalTensorPtr>& other, const LogicalTensorPtr& result)
{
    CheckExpandTensorValid(operand, result);
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, function.GetGraphType() == GraphType::TILE_GRAPH)
        << "The GetGraphType of function is incorrect";
    std::vector<int64_t> offset(result->shape.size(), 0);
    std::vector<int64_t> viewShape(result->shape.size(), 1);
    std::vector<SymbolicScalar> outValidShape;
    std::vector<int> expandDims;
    for (size_t i = 0; i < result->shape.size(); ++i) {
        if (operand->shape[i] != result->shape[i]) {
            expandDims.push_back(i);
            for (auto it : other) {
                if (it != nullptr && it->shape[i] == result->shape[i]) {
                    if (it->GetDynValidShape().empty()) {
                        outValidShape.push_back(it->shape[i]);
                    } else {
                        outValidShape.push_back(it->GetDynValidShape()[i]);
                    }
                    break;
                }
            }
        } else {
            if (operand->GetDynValidShape().empty()) {
                outValidShape.push_back(operand->shape[i]);
            } else {
                outValidShape.push_back(operand->GetDynValidShape()[i]);
            }
        }
    }

    result->UpdateDynValidShape(outValidShape);
    struct ExpandInfo expandInfo(operand, result, viewShape, offset, expandDims);
    ExpandTile(function, tileShape, 0, expandInfo, outValidShape);
}

void ExpandWithResultValidShape(Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand,
                                const LogicalTensorPtr& result, const std::vector<SymbolicScalar> resultValidShape)
{
    CheckExpandTensorValid(operand, result);
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, function.GetGraphType() == GraphType::TILE_GRAPH)
        << "The GetGraphType of function is incorrect";
    std::vector<int64_t> offset(result->shape.size(), 0);
    std::vector<int64_t> viewShape(result->shape.size(), 1);
    std::vector<int> expandDims;
    for (size_t i = 0; i < result->shape.size(); ++i) {
        if (operand->shape[i] != result->shape[i]) {
            expandDims.push_back(i);
        }
    }
    result->UpdateDynValidShape(resultValidShape);
    struct ExpandInfo expandInfo(operand, result, viewShape, offset, expandDims);
    ExpandTile(function, tileShape, 0, expandInfo, resultValidShape);
}

void TiledExpand(Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand,
                 const LogicalTensorPtr& result, const std::vector<SymbolicScalar>& validShape)
{
    CheckExpandTensorValid(operand, result);
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, function.GetGraphType() == GraphType::TILE_GRAPH)
        << "The GetGraphType of function is incorrect";

    std::vector<int64_t> offset(result->shape.size(), 0);
    std::vector<int64_t> viewShape(result->shape.size(), 1);
    std::vector<int> expandDims;
    for (size_t i = 0; i < result->shape.size(); ++i) {
        if (operand->shape[i] != result->shape[i]) {
            expandDims.push_back(i);
        }
    }
    result->UpdateDynValidShape(validShape);
    struct ExpandInfo expandInfo(operand, result, viewShape, offset, expandDims);
    ExpandTile(function, tileShape, 0, expandInfo, validShape);
}

Tensor TensorExpandOperation(Function& function, const LogicalTensorPtr& operand, const std::vector<int64_t>& dstShape,
                             const std::vector<SymbolicScalar>& validShape)
{
    auto result = std::make_shared<LogicalTensor>(function, operand->Datatype(), dstShape, validShape);
    auto& op = function.AddOperation(Opcode::OP_EXPAND, {operand}, {result});

    op.SetAttribute(OP_ATTR_PREFIX + "shape", dstShape);
    op.SetAttribute(OP_ATTR_PREFIX + "validShape", validShape);
    return result;
}

Tensor TensorJustNeedCopyOperation(Function& function, const LogicalTensorPtr& operand,
                                   const std::vector<int64_t>& dstShape, const std::vector<SymbolicScalar>& validShape)
{
    auto result = std::make_shared<LogicalTensor>(function, operand->Datatype(), dstShape, validShape);
    function.AddOperation(Opcode::OP_REGISTER_COPY, {operand}, {result});
    return result;
}

Tensor Expand(const Tensor& self, const std::vector<int64_t>& dstShape, std::vector<SymbolicScalar> validShape)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Expand");

    const auto& combinations = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_EXPAND);
    CheckTensorDataTypeCombination({self.GetDataType()}, combinations, "EXPAND");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "EXPAND");
    CheckDstShapeSize(dstShape, "EXPAND");
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, self.GetShape().size() == dstShape.size())
        << "The shape size of self and dst should be equal";
    if (validShape.empty()) {
        for (size_t i = 0; i < dstShape.size(); ++i) {
            if (self.GetShape()[i] != dstShape[i]) {
                validShape.emplace_back(dstShape[i]);
            } else {
                validShape.emplace_back(self.GetShape()[i]);
            }
        }
    }
    bool needExpand = false;
    for (size_t i = 0; i < dstShape.size(); ++i) {
        if (self.GetShape()[i] != dstShape[i]) {
            needExpand = true;
        }
    }
    if (needExpand) {
        RETURN_CALL(ExpandOperation, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), dstShape,
                    validShape);
    } else {
        RETURN_CALL(JustNeedCopyOperation, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), dstShape,
                    validShape);
    }
}

void ExpandOperationTileFunc(Function& function, const TileShape& tileShape,
                             const std::vector<LogicalTensorPtr>& iOperand,
                             const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    std::vector<SymbolicScalar> validShape;
    op.GetAttr(OP_ATTR_PREFIX + "validShape", validShape);
    TiledExpand(function, tileShape, iOperand[0], oOperand[0], validShape);
}

REGISTER_OPERATION_TILED_FUNC(OP_EXPAND, Opcode::OP_EXPAND, ExpandOperationTileFunc);

} // namespace npu::tile_fwk
