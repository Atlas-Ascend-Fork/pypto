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
 * \file full.cpp
 * \brief
 */

#include "unary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "tilefwk/error_code.h"
#include "tilefwk/platform.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {

void TiledFull(Function& function, const TileShape& tileShape, size_t cur, const Element& value,
               const SymbolicScalar& dynValue, std::vector<int64_t>& shape,
               const std::vector<SymbolicScalar>& validShape, const LogicalTensorPtr& results, TileInfo& resultTileInfo)
{
    if (cur == results->shape.size()) {
        auto resultTile = results->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto& op = function.AddOperation("TILE_VEC_DUP", {}, {resultTile});
        op.SetAttribute(OpAttributeKey::scalar, value);
        if (dynValue.IsValid()) {
            op.SetAttribute(OpAttributeKey::dynScalar, dynValue);
        }
        op.SetAttribute(OP_ATTR_PREFIX + "shape", resultTileInfo.shape);
        op.SetAttribute(OP_ATTR_PREFIX + "validShape", resultTile->GetDynValidShape());
        return;
    }

    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < results->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(results->shape[cur] - i, vecTile[cur]);
        TiledFull(function, tileShape, cur + 1, value, dynValue, shape, validShape, results, resultTileInfo);
    }
}

void TiledFull(Function& function, const TileShape& tileShape, const Element& value, const SymbolicScalar& dynValue,
               std::vector<int64_t>& shape, const std::vector<SymbolicScalar>& validShape,
               const LogicalTensorPtr& results)
{
    TileInfo resultTileInfo(results->shape.size(), results->offset.size());
    TiledFull(function, tileShape, 0, value, dynValue, shape, validShape, results, resultTileInfo);
}

Tensor TensorFullOperation(Function& function, const Element& src, const SymbolicScalar& dynValue, DataType dtype,
                           const std::vector<int64_t>& dstShape, const std::vector<SymbolicScalar>& validShape)
{
    auto result = std::make_shared<LogicalTensor>(function, dtype, dstShape, validShape);
    auto& op = function.AddOperation(Opcode::OP_VEC_DUP, {}, {result}); // 输入没有tensor
    op.SetAttribute(OpAttributeKey::scalar, src);
    if (dynValue.IsValid()) {
        op.SetAttribute(OpAttributeKey::dynScalar, dynValue);
    }
    op.SetAttribute(OP_ATTR_PREFIX + "shape", dstShape);
    op.SetAttribute(OP_ATTR_PREFIX + "validShape", validShape);
    return result;
}

Tensor Full(const Element& src, DataType dtype, const std::vector<int64_t>& dstShape,
            std::vector<SymbolicScalar> validShape)
{
    DECLARE_TRACER();
    const auto& combinations = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_VEC_DUP);
    CheckTensorDataTypeCombination({dtype}, combinations, "FULL");
    CheckDstShapeDimRange(dstShape, 1, NUM_VALUE_4, "FULL");
    CheckDstShapeSize(dstShape, "FULL");
    if (validShape.empty()) {
        for (auto x : dstShape)
            validShape.emplace_back(x);
    }
    RETURN_CALL(FullOperation, *Program::GetInstance().GetCurrentFunction(), src, SymbolicScalar(), dtype, dstShape,
                validShape);
}

Tensor Full(const SymbolicScalar& dynSrc, DataType dtype, const std::vector<int64_t>& dstShape,
            std::vector<SymbolicScalar> validShape)
{
    DECLARE_TRACER();
    const auto& combinations = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_VEC_DUP);
    CheckTensorDataTypeCombination({dtype}, combinations, "FULL");
    CheckDstShapeDimRange(dstShape, 1, NUM_VALUE_4, "FULL");
    CheckDstShapeSize(dstShape, "FULL");
    if (validShape.empty()) {
        for (auto x : dstShape)
            validShape.emplace_back(x);
    }
    RETURN_CALL(FullOperation, *Program::GetInstance().GetCurrentFunction(), Element(dtype, (int64_t)0), dynSrc, dtype,
                dstShape, validShape);
}

void FullOperationTileFunc(Function& function, const TileShape& tileShape,
                           [[maybe_unused]] const std::vector<LogicalTensorPtr>& iOperand,
                           const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    Element scalar = op.GetElementAttribute(OpAttributeKey::scalar);
    SymbolicScalar dynScalar;
    if (op.HasAttr(OpAttributeKey::dynScalar)) {
        dynScalar = op.GetSymbolicScalarAttribute(OpAttributeKey::dynScalar);
    }
    std::vector<int64_t> shape = op.GetVectorIntAttribute(OP_ATTR_PREFIX + "shape");
    std::vector<SymbolicScalar> validShape;
    op.GetAttr(OP_ATTR_PREFIX + "validShape", validShape);
    TiledFull(function, tileShape, scalar, dynScalar, shape, validShape, oOperand[0]);
}

REGISTER_OPERATION_TILED_FUNC(OP_VEC_DUP, Opcode::OP_VEC_DUP, FullOperationTileFunc);

} // namespace npu::tile_fwk
