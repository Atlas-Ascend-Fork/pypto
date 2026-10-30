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
 * \\file gather.cpp
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
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {

namespace {
bool IsFp8DataType(DataType dtype) { return dtype == DT_FP8E4M3 || dtype == DT_FP8E5M2 || dtype == DT_FP8E8M0; }

void CheckFp8ArchSupport(const Tensor& tensor, const std::string& opName)
{
    if (!IsFp8DataType(tensor.GetDataType())) {
        return;
    }
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510)
        << opName << ": DT_FP8E4M3, DT_FP8E5M2 and DT_FP8E8M0 are only supported on DAV_3510 architecture.";
}
} // namespace

void TiledGatherOperation(Function& function, const TileShape& tileShape, size_t cur, Input& paramsInput,
                          Input& indicesInput, int axis, const LogicalTensorPtr& result, TileInfo& resultTileInfo)
{
    if (cur == result->shape.size()) {
        // add Operation
        auto paramsTile = paramsInput.tensor.GetStorage()->View(function, paramsInput.tileInfo.shape,
                                                                paramsInput.tileInfo.offset);
        auto indicesTile = indicesInput.tensor.GetStorage()->View(function, indicesInput.tileInfo.shape,
                                                                  indicesInput.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        if (function.IsStatic()) {
            auto& op = function.AddOperation(Opcode::OP_GATHER_FROM_UB, {paramsTile, indicesTile}, {resultTile});
            op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        } else {
            auto& op = function.AddOperation(Opcode::OP_GATHER, {paramsTile, indicesTile}, {resultTile});
            op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        }

        return;
    }

    // 按照resultShape进行切分
    auto& vecTile = tileShape.GetVecTile();
    int64_t tmpTile = vecTile[cur];
    for (int i = 0; i < result->shape[cur]; i += tmpTile) {
        if (cur < static_cast<size_t>(axis)) {
            // 在result中gather轴的外层轴
            paramsInput.tileInfo.offset[cur] = i % paramsInput.tensor.GetShape()[cur];
            paramsInput.tileInfo.shape[cur] = std::min(
                paramsInput.tensor.GetShape()[cur] - paramsInput.tileInfo.offset[cur], tmpTile);
        } else if (cur >= static_cast<size_t>(axis) &&
                   (cur < static_cast<size_t>(axis) + indicesInput.tensor.GetShape().size())) {
            // 当前属于indices的gather轴
            // params[axis]不切
            paramsInput.tileInfo.offset[axis] = 0;
            paramsInput.tileInfo.shape[axis] = paramsInput.tensor.GetShape()[axis];
            // 处理indices的tileInfo
            indicesInput.tileInfo.offset[cur - axis] = i % indicesInput.tensor.GetShape()[cur - axis];
            indicesInput.tileInfo.shape[cur - axis] = std::min(
                indicesInput.tensor.GetShape()[cur - axis] - indicesInput.tileInfo.offset[cur - axis], tmpTile);
        } else {
            // 在result中gather轴的内层轴
            int paramHighAxis = cur - indicesInput.tensor.GetShape().size() + 1;
            paramsInput.tileInfo.offset[paramHighAxis] = i % paramsInput.tensor.GetShape()[paramHighAxis];
            paramsInput.tileInfo.shape[paramHighAxis] = std::min(
                paramsInput.tensor.GetShape()[paramHighAxis] - paramsInput.tileInfo.offset[paramHighAxis], tmpTile);
        }

        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], tmpTile);
        TiledGatherOperation(function, tileShape, cur + 1, paramsInput, indicesInput, axis, result, resultTileInfo);
    }
}

std::vector<int64_t> GatherOperationResultShape(LogicalTensorPtr params, LogicalTensorPtr indices, int axis)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, params->shape.size() == params->offset.size())
        << "The size of params shape and offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, indices->shape.size() == indices->offset.size())
        << "The size of indices shape and offset should be equal";
    int paramsRank = params->shape.size();
    if (axis < 0) {
        axis = axis + paramsRank;
    }
    // result shape: params.shape[:aixs] + indices.shape + params.shape[axis+1:]
    std::vector<int64_t> resultShape = params->shape;
    resultShape.erase(resultShape.begin() + axis);
    resultShape.insert(resultShape.begin() + axis, indices->shape.begin(), indices->shape.end());

    return resultShape;
}

void TiledGatherOperation(Function& function, const TileShape& tileShape, const LogicalTensorPtr& params,
                          const LogicalTensorPtr& indices, int axis, const LogicalTensorPtr& result)
{
    // Check Operands Valid
    std::vector<int64_t> expectedShape = GatherOperationResultShape(params, indices, axis);
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, result->shape.size() == expectedShape.size())
        << "The size of result shape and expectedShape should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, result->shape.size() == result->offset.size())
        << "The size of result shape and offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, params->shape.size() == params->offset.size())
        << "The size of params shape and offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, indices->shape.size() == indices->offset.size())
        << "The size of indices shape and offset should be equal";

    CHECK(VectorErrorCode::ERR_PARAM_SHAPE_DIM_UNSUPPORTED, result->shape.size() <= NUM_VALUE_5)
        << "Not support shape size of result greater than 5";
    CHECK(VectorErrorCode::ERR_PARAM_SHAPE_DIM_UNSUPPORTED, indices->shape.size() <= NUM_VALUE_2)
        << "Not support shape size of indices greater than 2";
    if (axis < 0) {
        axis += params->shape.size();
    }
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, axis >= 0 && axis < static_cast<int>(params->shape.size()))
        << "The axis should be greater than or equal to 0 and less than shape size of params";
    TileInfo paramsTileInfo(params->shape.size(), params->offset.size());
    TileInfo indicesTileInfo(indices->shape.size(), indices->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto paramsInput = Input{params, paramsTileInfo};
    auto indicesInput = Input{indices, indicesTileInfo};
    TiledGatherOperation(function, tileShape, 0, paramsInput, indicesInput, axis, result, resultTileInfo);
}

LogicalTensorPtr TensorGatherOperation(Function& function, const LogicalTensorPtr& params,
                                       const LogicalTensorPtr& indices, int axis)
{
    const auto& paramsDynShape = params->GetDynValidShape();
    const auto& indicesDynShape = indices->GetDynValidShape();
    const int paramsRank = paramsDynShape.size();
    if (axis < 0) {
        axis += paramsRank;
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, axis >= 0 && axis < paramsRank)
            << "The configuration of the axis is incorrect";
    }
    std::vector<int64_t> resultShape = GatherOperationResultShape(params, indices, axis);
    auto result = std::make_shared<LogicalTensor>(function, params->Datatype(), resultShape);
    std::vector<SymbolicScalar> outValidShape = paramsDynShape;
    outValidShape.erase(outValidShape.begin() + axis);
    outValidShape.insert(outValidShape.begin() + axis, indicesDynShape.begin(), indicesDynShape.end());
    auto& op = GraphUtils::AddDynOperation(function, Opcode::OP_GATHER, {params, indices}, {result}, {outValidShape});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);

    return result;
}

void CheckGatherParamsInvalid(const Tensor& params, const Tensor& indices, int axis, const std::string& opName)
{
    const auto& combinations = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_GATHER);
    CheckTensorDataTypeCombination({params.GetDataType(), indices.GetDataType()}, combinations, opName);
    CheckFp8ArchSupport(params, opName);
    CheckTensorDimRange(params.GetStorage(), 1, NUM_VALUE_4, opName);
    CheckTensorDimRange(indices.GetStorage(), 1, NUM_VALUE_2, opName);
    CheckTensorShapeSize(params.GetStorage(), opName);
    CheckTensorShapeSize(indices.GetStorage(), opName);
    CheckAxisRange(params, axis);
    CheckTensorsFormatConsistency(params.GetStorage(), indices.GetStorage(), opName);
}

Tensor Gather(const Tensor& params, const Tensor& indices, int axis)
{
    DECLARE_TRACER();
    CheckVectorOperand(params.GetStorage(), "Gather");
    CheckVectorOperand(indices.GetStorage(), "Gather");

    CheckGatherParamsInvalid(params, indices, axis, "GATHER");
    RETURN_CALL(GatherOperation, *Program::GetInstance().GetCurrentFunction(), params.GetStorage(),
                indices.GetStorage(), axis);
}

Tensor TensorIndex(const Tensor& params, const Tensor& indices)
{
    DECLARE_TRACER();
    CheckGatherParamsInvalid(params, indices, 0, "TENSORINDEX");
    // TensorIndex默认按0轴进行gather
    RETURN_CALL(GatherOperation, *Program::GetInstance().GetCurrentFunction(), params.GetStorage(),
                indices.GetStorage(), 0);
}

void TiledGatherElementOperation(Function& function, const TileShape& tileShape, size_t cur, Input& paramsInput,
                                 Input& indicesInput, int axis, const LogicalTensorPtr& result,
                                 TileInfo& resultTileInfo)
{
    if (cur == result->shape.size()) {
        // add Operation
        auto paramsTile = paramsInput.tensor.GetStorage()->View(function, paramsInput.tileInfo.shape,
                                                                paramsInput.tileInfo.offset);
        auto indicesTile = indicesInput.tensor.GetStorage()->View(function, indicesInput.tileInfo.shape,
                                                                  indicesInput.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);
        Shape tmpShape({indicesTile->GetShape()[indicesTile->GetShape().size() - 1]});
        tmpShape[0] = NUM_VALUE_2 * AlignUp(tmpShape[0], BLOCK_SIZE / BytesOf(resultTile->Datatype()));
        auto tmpBuffer = std::make_shared<LogicalTensor>(function, indicesTile->Datatype(), tmpShape);
        auto& op = function.AddOperation(Opcode::OP_GATHER_ELEMENT, {paramsTile, indicesTile}, {resultTile, tmpBuffer});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        return;
    }

    // 按照resultShape进行切分
    auto& vecTile = tileShape.GetVecTile();
    int64_t tmpTile = vecTile[cur];
    for (int i = 0; i < result->shape[cur]; i += tmpTile) {
        if (cur == static_cast<size_t>(axis)) {
            // params[axis]不切
            paramsInput.tileInfo.offset[cur] = 0;
            paramsInput.tileInfo.shape[cur] = paramsInput.tensor.GetShape()[cur];
            // 处理indices的tileInfo
            indicesInput.tileInfo.offset[cur] = i % indicesInput.tensor.GetShape()[cur];
            indicesInput.tileInfo.shape[cur] = std::min(
                indicesInput.tensor.GetShape()[cur] - indicesInput.tileInfo.offset[cur], tmpTile);
        } else {
            paramsInput.tileInfo.offset[cur] = i % paramsInput.tensor.GetShape()[cur];
            paramsInput.tileInfo.shape[cur] = std::min(
                paramsInput.tensor.GetShape()[cur] - paramsInput.tileInfo.offset[cur], tmpTile);
            // 处理indices的tileInfo
            indicesInput.tileInfo.offset[cur] = i % indicesInput.tensor.GetShape()[cur];
            indicesInput.tileInfo.shape[cur] = std::min(
                indicesInput.tensor.GetShape()[cur] - indicesInput.tileInfo.offset[cur], tmpTile);
        }

        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], tmpTile);
        TiledGatherElementOperation(function, tileShape, cur + 1, paramsInput, indicesInput, axis, result,
                                    resultTileInfo);
    }
}

void TiledGatherElementOperation(Function& function, const TileShape& tileShape, const LogicalTensorPtr& params,
                                 const LogicalTensorPtr& indices, int axis, const LogicalTensorPtr& result)
{
    // Check Operands Valid
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, result->shape.size() == result->offset.size())
        << "The size of result shape and offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, params->shape.size() == params->offset.size())
        << "The size of params shape and offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, indices->shape.size() == indices->offset.size())
        << "The size of indices shape and offset should be equal";

    TileInfo paramsTileInfo(params->shape.size(), params->offset.size());
    TileInfo indicesTileInfo(indices->shape.size(), indices->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto paramsInput = Input{params, paramsTileInfo};
    auto indicesInput = Input{indices, indicesTileInfo};
    TiledGatherElementOperation(function, tileShape, 0, paramsInput, indicesInput, axis, result, resultTileInfo);
}

LogicalTensorPtr TensorGatherElementOperation(Function& function, const LogicalTensorPtr& params,
                                              const LogicalTensorPtr& indices, int axis)
{
    auto result = std::make_shared<LogicalTensor>(function, params->Datatype(), indices->shape);
    std::vector<std::vector<SymbolicScalar>> outValidShape;
    outValidShape.push_back(indices->GetDynValidShape());
    auto& op = GraphUtils::AddDynOperation(function, Opcode::OP_GATHER_ELEMENT, {params, indices}, {result},
                                           outValidShape);
    op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);

    return result;
}

Tensor GatherElements(const Tensor& params, const Tensor& indices, int axis)
{
    DECLARE_TRACER();
    CheckVectorOperand(params.GetStorage(), "GatherElements");
    CheckVectorOperand(indices.GetStorage(), "GatherElements");

    std::vector<LogicalTensorPtr> tensors = {params.GetStorage(), indices.GetStorage()};
    CheckTensorsDimConsistency(tensors, "GATHERELEMENTS");
    CheckAxisRange(params, axis); // 支持负轴
    for (size_t i = 0; i < params.GetShape().size(); ++i) {
        if (static_cast<int>(i) == axis) {
            continue;
        }
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, indices.GetShape()[i] <= params.GetShape()[i])
            << "The shape of params and indices should be equal";
    }
    const auto& combinations = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_GATHER_ELEMENT);
    CheckTensorDataTypeCombination({params.GetDataType(), indices.GetDataType()}, combinations, "GATHERELEMENTS");
    CheckTensorDimRange(params.GetStorage(), 1, NUM_VALUE_5, "GATHERELEMENTS");
    CheckTensorsFormatConsistency(params.GetStorage(), indices.GetStorage(), "GATHERELEMENTS");

    RETURN_CALL(GatherElementOperation, *Program::GetInstance().GetCurrentFunction(), params.GetStorage(),
                indices.GetStorage(), axis);
}

void GatherOperationTileFunc(Function& function, const TileShape& tileShape,
                             const std::vector<LogicalTensorPtr>& iOperand,
                             const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    int axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
    TiledGatherOperation(function, tileShape, iOperand[0], iOperand[1], axis, oOperand[0]);
}

void GatherElementOperationTileFunc(Function& function, const TileShape& tileShape,
                                    const std::vector<LogicalTensorPtr>& iOperand,
                                    const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    int axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
    TiledGatherElementOperation(function, tileShape, iOperand[0], iOperand[1], axis, oOperand[0]);
}

REGISTER_OPERATION_TILED_FUNC(OP_GATHER, Opcode::OP_GATHER, GatherOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_GATHER_ELEMENT, Opcode::OP_GATHER_ELEMENT, GatherElementOperationTileFunc);

} // namespace npu::tile_fwk
