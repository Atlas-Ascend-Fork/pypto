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
 * \file triangular.cpp
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

struct TriULTileInfoPara {
    TileInfo inputTileInfo;
    TileInfo dstTileInfo;
};

struct TriULPara {
    const LogicalTensorPtr& input;
    const LogicalTensorPtr& dstTensor;
    const SymbolicScalar diagonal;
    const bool isUpper;
};

void InnerTiledTriUL(size_t cur, Function& function, const TileShape& tileShape, const TriULPara& triULPara,
                     TriULTileInfoPara& triULTileInfo)
{
    const LogicalTensorPtr& input = triULPara.input;
    const LogicalTensorPtr& dstTensor = triULPara.dstTensor;
    SymbolicScalar realDiagonal = triULPara.diagonal;
    const bool isUpper = triULPara.isUpper;
    auto& vecTile = tileShape.GetVecTile();

    if (cur == dstTensor->GetShape().size()) {
        auto dstTile = dstTensor->View(function, triULTileInfo.dstTileInfo.shape, triULTileInfo.dstTileInfo.offset);
        auto inputTile = input->View(function, triULTileInfo.inputTileInfo.shape, triULTileInfo.inputTileInfo.offset);
        realDiagonal = realDiagonal + dstTile->GetOffset()[cur - NUM_VALUE_2] - dstTile->GetOffset()[cur - 1];
        auto& op = function.AddOperation(Opcode::OP_TRIUL, {inputTile}, {dstTile});
        op.SetAttribute(OpAttributeKey::dynScalar, realDiagonal);
        op.SetAttribute(OpAttributeKey::isUpper, isUpper);
        return;
    }
    int64_t tmpTile = vecTile[cur];

    for (int i = 0; i < input->GetShape()[cur]; i += tmpTile) {
        triULTileInfo.dstTileInfo.offset[cur] = i;
        triULTileInfo.dstTileInfo.shape[cur] = std::min(dstTensor->GetShape()[cur] - i, tmpTile);
        triULTileInfo.inputTileInfo.offset[cur] = i;
        triULTileInfo.inputTileInfo.shape[cur] = std::min(input->GetShape()[cur] - i, tmpTile);
        InnerTiledTriUL(cur + 1, function, tileShape, triULPara, triULTileInfo);
    }
}

void TiledTriUL(Function& function, const TileShape& tileShape, const TriULPara& triULPara)
{
    TriULTileInfoPara triULTileInfo{
        TileInfo(triULPara.input->GetShape().size(), triULPara.input->GetOffset().size()),
        TileInfo(triULPara.dstTensor->GetShape().size(), triULPara.dstTensor->GetOffset().size())};

    InnerTiledTriUL(0, function, tileShape, triULPara, triULTileInfo);
}

void CheckTriULOperationParams(const Tensor& input, const std::string& opName)
{
    const auto& combinations = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_TRIUL);
    CheckTensorDataTypeCombination({input.GetDataType()}, combinations, opName);
    CheckTensorDimRange(input.GetStorage(), NUM_VALUE_2, NUM_VALUE_5, opName);
}

void TensorTriUL(Function& function, const TriULPara& triULPara)
{
    if (triULPara.input->Datatype() == DT_INT8) {
        LogicalTensorPtr inputConverted = std::make_shared<LogicalTensor>(
            function, DT_FP16, triULPara.input->GetShape(), triULPara.input->GetDynValidShape());
        auto& castinputOp = GraphUtils::AddDynOperation(function, Opcode::OP_CAST, {triULPara.input}, {inputConverted});
        castinputOp.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_NONE);
        castinputOp.SetAttribute(OP_ATTR_PREFIX + "satmode", static_cast<int64_t>(SaturationMode::ON));
        LogicalTensorPtr dstConverted = std::make_shared<LogicalTensor>(
            function, DT_FP16, triULPara.dstTensor->GetShape(), inputConverted->GetDynValidShape());
        auto& op = GraphUtils::AddDynOperation(function, Opcode::OP_TRIUL, {inputConverted}, {dstConverted});
        op.SetAttribute(OpAttributeKey::dynScalar, triULPara.diagonal);
        op.SetAttribute(OpAttributeKey::isUpper, triULPara.isUpper);
        triULPara.dstTensor->UpdateDynValidShape(dstConverted->GetDynValidShape());
        auto& castDstOp = GraphUtils::AddDynOperation(function, Opcode::OP_CAST, {dstConverted}, {triULPara.dstTensor});
        castDstOp.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_TRUNC);
        castDstOp.SetAttribute(OP_ATTR_PREFIX + "satmode", static_cast<int64_t>(SaturationMode::ON));
    } else {
        triULPara.dstTensor->UpdateDynValidShape(triULPara.input->GetDynValidShape());
        auto& op = GraphUtils::AddDynOperation(function, Opcode::OP_TRIUL, {triULPara.input}, {triULPara.dstTensor});
        op.SetAttribute(OpAttributeKey::dynScalar, triULPara.diagonal);
        op.SetAttribute(OpAttributeKey::isUpper, triULPara.isUpper);
    }
}

Tensor TriU(const Tensor& input, const SymbolicScalar& diagonal)
{
    DECLARE_TRACER();
    CheckVectorOperand(input.GetStorage(), "TriU");

    CheckTriULOperationParams(input, "TRIU");
    Tensor result(input.GetDataType(), input.GetShape());
    CALL(TriUL, *Program::GetInstance().GetCurrentFunction(),
         {input.GetStorage(), result.GetStorage(), diagonal, true});
    return result;
}

Tensor TriL(const Tensor& input, const SymbolicScalar& diagonal)
{
    DECLARE_TRACER();
    CheckVectorOperand(input.GetStorage(), "TriL");

    CheckTriULOperationParams(input, "TRIL");
    Tensor result(input.GetDataType(), input.GetShape());
    CALL(TriUL, *Program::GetInstance().GetCurrentFunction(),
         {input.GetStorage(), result.GetStorage(), diagonal, false});
    return result;
}

void TriULOperationTileFunc(Function& function, const TileShape& tileShape,
                            const std::vector<LogicalTensorPtr>& iOperand,
                            const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    SymbolicScalar diagonal = op.GetSymbolicScalarAttribute(OpAttributeKey::dynScalar);
    bool isUpper = op.GetBoolAttribute(OpAttributeKey::isUpper);
    TiledTriUL(function, tileShape, {iOperand[0], oOperand[0], diagonal, isUpper});
}

REGISTER_OPERATION_TILED_FUNC(OP_TRIUL, Opcode::OP_TRIUL, TriULOperationTileFunc);

} // namespace npu::tile_fwk
