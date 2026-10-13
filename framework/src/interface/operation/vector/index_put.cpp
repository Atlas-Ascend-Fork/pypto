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
 * \\file index_put.cpp
 * \\brief
 */

#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation_common.h"
#include "interface/operation/vector/gather_mask_common.h"
#include "tensor_transformation.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

void TiledIndexPut(Function& function, const TileShape& tileShape, Input& inputSelf, Input& inputValues,
                   std::vector<Input>& inputIndices, const LogicalTensorPtr result, bool accumulate, size_t cur)
{
    size_t selfDim = inputSelf.tileInfo.shape.size();
    size_t valuesDim = inputValues.tileInfo.shape.size();
    size_t indicesCount = inputIndices.size();
    if (cur == valuesDim) {
        auto inputSelfTile = inputSelf.tensor.GetStorage()->View(function, inputSelf.tileInfo.shape,
                                                                 inputSelf.tileInfo.offset);
        auto inputValuesTile = inputValues.tensor.GetStorage()->View(function, inputValues.tileInfo.shape,
                                                                     inputValues.tileInfo.offset);
        std::vector<LogicalTensorPtr> inputsTile;
        inputsTile.push_back(inputSelfTile);
        inputsTile.push_back(inputValuesTile);
        for (size_t j = 0; j < indicesCount; j++) {
            auto inputIndicesTile = inputIndices[j].tensor.GetStorage()->View(function, inputIndices[j].tileInfo.shape,
                                                                              inputIndices[j].tileInfo.offset);
            inputsTile.push_back(inputIndicesTile);
        }
        bool useSimt = Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510 && selfDim == indicesCount &&
                       (inputSelf.tensor.GetDataType() == DT_FP32 || inputSelf.tensor.GetDataType() == DT_FP16 ||
                        inputSelf.tensor.GetDataType() == DT_BF16) &&
                       (inputIndices[0].tensor.GetDataType() == DT_INT32 ||
                        inputIndices[0].tensor.GetDataType() == DT_UINT32);
        LogicalTensors outputs{result};
        if (useSimt) {
            Shape tmpShape({inputValuesTile->GetShape()[0]});
            outputs.push_back(
                std::make_shared<LogicalTensor>(function, inputIndices[0].tensor.GetDataType(), tmpShape));
        }
        auto& newOp = function.AddOperation(Opcode::OP_INDEX_PUT, inputsTile, outputs);
        newOp.SetAttribute(OpAttributeKey::inplaceIdx, 0);
        newOp.SetAttribute(OpAttributeKey::accumulate, accumulate);
        newOp.SetAttribute(OpAttributeKey::indicesSize, static_cast<int>(indicesCount));
        if (useSimt) {
            newOp.SetAttribute(OP_ATTR_PREFIX + "requires_simt", true);
        }
        return;
    }
    const auto& vecTile = tileShape.GetVecTile();
    int64_t tileSize = inputValues.tensor.GetShape()[cur];
    if (cur < vecTile.size()) {
        tileSize = vecTile[cur];
    }
    for (int64_t i = 0, size = inputValues.tensor.GetShape()[cur]; i < size; i += tileSize) {
        if (cur != 0) {
            size_t selfIndex = selfDim - valuesDim + cur;
            inputSelf.tileInfo.shape[selfIndex] = std::min(inputSelf.tensor.GetShape()[selfIndex] - i, tileSize);
            inputSelf.tileInfo.offset[selfIndex] = i;
        }
        inputValues.tileInfo.shape[cur] = std::min(inputValues.tensor.GetShape()[cur] - i, tileSize);
        inputValues.tileInfo.offset[cur] = i;
        if (cur == 0) {
            for (size_t j = 0; j < indicesCount; ++j) {
                inputIndices[j].tileInfo.shape[cur] = std::min(inputIndices[j].tensor.GetShape()[cur] - i, tileSize);
                inputIndices[j].tileInfo.offset[cur] = i;
            }
        }
        TiledIndexPut(function, tileShape, inputSelf, inputValues, inputIndices, result, accumulate, cur + 1);
    }
}

void TiledIndexPut(Function& function, const TileShape& tileShape, const LogicalTensorPtr& self,
                   const LogicalTensorPtr& values, const std::vector<LogicalTensorPtr>& indices,
                   const LogicalTensorPtr& result, bool accumulate)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, self->GetShape().size() == self->GetOffset().size());
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, values->GetShape().size() == values->GetOffset().size());
    for (size_t i = 0; i < indices.size(); i++) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, indices[i]->GetShape().size() == indices[i]->GetOffset().size());
    }
    TileInfo valuesTileInfo(values->shape.size(), values->offset.size());
    TileInfo selfTileInfo(self->shape.size(), self->offset.size());
    auto inputValues = Input{values, valuesTileInfo};
    auto inputSelf = Input{self, selfTileInfo};
    for (size_t i = 0, size = self->shape.size(); i < size; ++i) {
        inputSelf.tileInfo.shape[i] = self->shape[i];
        inputSelf.tileInfo.offset[i] = 0;
    }
    std::vector<Input> inputIndices;
    for (size_t i = 0, size = indices.size(); i < size; ++i) {
        TileInfo indicesTileInfoTemp(indices[i]->shape.size(), indices[i]->offset.size());
        auto inputIndicesTemp = Input{indices[i], indicesTileInfoTemp};
        inputIndices.push_back(inputIndicesTemp);
    }
    TiledIndexPut(function, tileShape, inputSelf, inputValues, inputIndices, result, accumulate, 0);
}

void TensorIndexPut(Function& function, const LogicalTensorPtr& self, const LogicalTensors& indices,
                    const LogicalTensorPtr& values, const LogicalTensorPtr& dst, bool accumulate)
{
    LogicalTensors iOperands = indices;
    iOperands.insert(iOperands.begin(), {self, values});
    auto& op = function.AddOperation(Opcode::OP_INDEX_PUT, iOperands, {dst});
    op.SetAttribute(OpAttributeKey::inplaceIdx, 0);
    op.SetAttribute(OpAttributeKey::accumulate, accumulate);
    op.SetAttribute(OpAttributeKey::indicesSize, static_cast<int>(indices.size()));
}

static void CheckIndexPutParamsInvalid(const Tensor& self, const std::vector<Tensor>& indices, const Tensor& values)
{
    const auto& combinations = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_INDEX_PUT);
    int indicesShape = -1;
    for (size_t i = 0; i < indices.size(); i++) {
        CheckTensorDataTypeCombination({self.GetDataType(), indices[i].GetDataType(), values.GetDataType()},
                                       combinations, "INDEXPUT");
        CheckTensorDimRange(indices[i].GetStorage(), 1, 1, "INDEXPUT");
        if (indicesShape == -1) {
            indicesShape = indices[i].GetShape()[0];
        } else {
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, indices[i].GetShape()[0] == indicesShape)
                << "Tensors in indices should have the same shape";
        }
        CheckTensorShapeSize(indices[i].GetStorage(), "INDEXPUT");
    }
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, indices.size() >= NUM_VALUE_1 && indices.size() <= NUM_VALUE_4)
        << "indicesSize is out of range [1, 4]";
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "INDEXPUT");
    CheckTensorDimRange(values.GetStorage(), 1, NUM_VALUE_4, "INDEXPUT");
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, self.GetShape().size() + 1 == indices.size() + values.GetShape().size())
        << "unsupport the inputs shape combination: dimSelf + 1 != indicesSize + dimValues";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, values.GetShape()[0] == indicesShape)
        << "valuesFirstDim should be equal to indicesShape";
    for (size_t i = 1; i < values.GetShape().size(); i++) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID,
              self.GetShape()[self.GetShape().size() - i] == values.GetShape()[values.GetShape().size() - i])
            << "valuesShape should match selfShape";
    }
    CheckTensorShapeSize(self.GetStorage(), "INDEXPUT");
    CheckTensorShapeSize(values.GetStorage(), "INDEXPUT");
    CheckTensorsFormatConsistency(self.GetStorage(), values.GetStorage(), "INDEXPUT");
    for (size_t i = 0; i < indices.size(); i++) {
        CheckTensorsFormatConsistency(self.GetStorage(), indices[i].GetStorage(), "INDEXPUT");
        CheckTensorsFormatConsistency(values.GetStorage(), indices[i].GetStorage(), "INDEXPUT");
    }
}

void IndexPut_(Tensor& self, const std::vector<Tensor>& indices, const Tensor& values, bool accumulate)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "IndexPut_");
    CheckTensorFormat(values.GetStorage(), {TileOpFormat::TILEOP_NZ}, "IndexPut_");
    for (const auto& tensor : indices) {
        CheckTensorFormat(tensor.GetStorage(), {TileOpFormat::TILEOP_NZ}, "IndexPut_");
    }

    CheckIndexPutParamsInvalid(self, indices, values);

    std::vector<LogicalTensorPtr> indicesLogical;
    for (size_t i = 0; i < indices.size(); i++) {
        indicesLogical.push_back(indices[i].GetStorage());
    }
    Tensor dst(self.GetDataType(), self.GetShape());
    CALL(IndexPut, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), indicesLogical, values.GetStorage(),
         dst.GetStorage(), accumulate);
    Program::GetInstance().GetCurrentFunction()->SetSameMemId(self.GetStorage(), dst.GetStorage());
    self = dst;
}

void IndexPutOperationTileFunc(Function& function, const TileShape& tileShape,
                               const std::vector<LogicalTensorPtr>& iOperand,
                               const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    std::vector<LogicalTensorPtr> indices = iOperand;
    constexpr size_t num2 = NUM_VALUE_2;
    indices.erase(indices.begin(), indices.begin() + num2);
    bool accumulate = op.GetBoolAttribute(OpAttributeKey::accumulate);
    TiledIndexPut(function, tileShape, iOperand[0], iOperand[1], indices, oOperand[0], accumulate);
}

REGISTER_OPERATION_TILED_FUNC(OP_INDEX_PUT, Opcode::OP_INDEX_PUT, IndexPutOperationTileFunc);

} // namespace npu::tile_fwk
