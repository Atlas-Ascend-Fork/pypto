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
 * \file axpy.cpp
 * \brief
 */
#include "tilefwk/data_type.h"
#include "unary.h"
#include "binary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/error_code.h"
#include "passes/tile_graph_pass/graph_constraint/axis_combine.h"

namespace npu::tile_fwk {

void TiledAxpyOperation(Function& function, const TileShape& tileShape, size_t cur, LogicalInput& inputSelf,
                        LogicalInput& inputOther, const Element& alpha, const LogicalTensorPtr& result,
                        TileInfo& resultTileInfo)
{
    size_t shapeSize = inputSelf.tensor->GetShape().size();
    if (cur == shapeSize) {
        auto selfTile = inputSelf.tensor->View(function, inputSelf.tileInfo.shape, inputSelf.tileInfo.offset);
        auto otherTile = inputOther.tensor->View(function, inputOther.tileInfo.shape, inputOther.tileInfo.offset);
        auto resultTile = result->View(function, resultTileInfo.shape, resultTileInfo.offset);

        auto& op = function.AddOperation(Opcode::OP_AXPY, {selfTile, otherTile}, {resultTile});
        op.SetAttribute(OpAttributeKey::scalar, alpha);
        std::vector<int64_t> brcOperand(shapeSize, 0);
        size_t brcAxesCount = 0;
        for (size_t i = 0; i < shapeSize; i++) {
            int brcResult = BrcAxisBinaryOp(inputSelf.tensor, inputOther.tensor, i);
            brcOperand[i] = (brcResult == 1) ? 0 : brcResult;
            if (brcOperand[i] != 0) {
                brcAxesCount++;
            }
        }
        if (brcOperand[shapeSize - 1] != 0) {
            op.SetAttribute(OpAttributeKey::excludeBufferReuse, true);
            op.SetAttribute(OpAttributeKey::brcbIdx, brcOperand[shapeSize - 1]);
        }
        if (brcAxesCount > 0) {
            op.SetAttribute(OpAttributeKey::brcOperand, brcOperand);
        }
        return;
    }

    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < result->shape[cur]; i += vecTile[cur]) {
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(result->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        inputSelf.tileInfo.offset[cur] = i % inputSelf.tensor->GetShape()[cur];
        inputSelf.tileInfo.shape[cur] = std::min(inputSelf.tensor->GetShape()[cur] - inputSelf.tileInfo.offset[cur],
                                                 vecTile[cur]);
        inputOther.tileInfo.offset[cur] = i % inputOther.tensor->GetShape()[cur];
        inputOther.tileInfo.shape[cur] = std::min(inputOther.tensor->GetShape()[cur] - inputOther.tileInfo.offset[cur],
                                                  vecTile[cur]);
        TiledAxpyOperation(function, tileShape, cur + 1, inputSelf, inputOther, alpha, result, resultTileInfo);
    }
}

void TiledAxpyOperation(Function& function, const TileShape& tileShape, LogicalTensorPtr self, LogicalTensorPtr other,
                        const Element& alpha, const LogicalTensorPtr& result)
{
    CheckBinOpOperandsValid(self, other);
    BroadcastOperandTensor(other, self, result, function, tileShape);

    TileInfo selfTileInfo(self->shape.size(), self->offset.size());
    TileInfo otherTileInfo(other->shape.size(), other->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());
    auto inputSelf = LogicalInput{self, selfTileInfo};
    auto inputOther = LogicalInput{other, otherTileInfo};

    TiledAxpyOperation(function, tileShape, 0, inputSelf, inputOther, alpha, result, resultTileInfo);
}

void AxpyOperationTileFunc(Function& function, const TileShape& tileShape,
                           const std::vector<LogicalTensorPtr>& iOperand, const std::vector<LogicalTensorPtr>& oOperand,
                           const Operation& op)
{
    auto alpha = op.GetElementAttribute(OpAttributeKey::scalar);
    TiledAxpyOperation(function, tileShape, iOperand[0], iOperand[1], alpha, oOperand[0]);
}

LogicalTensorPtr TensorAxpyOperation(Function& function, const Tensor& self, const Tensor& other, float alpha)
{
    auto selfTensor = self.GetStorage();
    auto otherTensor = other.GetStorage();
    if (selfTensor->shape.size() != otherTensor->shape.size()) {
        std::vector<int> broadCastShape = GetBroadCastShape(selfTensor, otherTensor);
        selfTensor = BinaryOperationBroadCast(selfTensor, broadCastShape);
        otherTensor = BinaryOperationBroadCast(otherTensor, broadCastShape);
    }

    CheckTensorShapeSize(selfTensor, "AXPY");
    CheckTensorShapeSize(otherTensor, "AXPY");
    CheckBinOpOperandsValid(selfTensor, otherTensor);
    CheckTensorsFormatConsistency(selfTensor, otherTensor, "AXPY");

    std::vector<SymbolicScalar> resultValidShape;
    std::vector<int64_t> resultShape = BinaryOperationResultShape(selfTensor, otherTensor);
    size_t shapeSize = resultShape.size();
    if ((!selfTensor->GetDynValidShape().empty()) && (!otherTensor->GetDynValidShape().empty())) {
        for (size_t i = 0; i < shapeSize; ++i) {
            if (resultShape[i] == selfTensor->shape[i]) {
                resultValidShape.push_back(selfTensor->GetDynValidShape()[i]);
            } else {
                resultValidShape.push_back(otherTensor->GetDynValidShape()[i]);
            }
        }
    }
    // AXPY: y = alpha * x + y, y is in-place updated, cannot broadcast
    // Validate: if any dimension of y is 1 but x is not 1, it's invalid
    for (size_t i = 0; i < shapeSize; i++) {
        if ((selfTensor->shape[i] == 1) && (otherTensor->shape[i] != 1)) {
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, false)
                << "AXPY: self tensor cannot broadcast, self.shape[" << i << "]=" << selfTensor->shape[i]
                << " but other.shape[" << i << "]=" << otherTensor->shape[i];
        }
    }

    auto result = std::make_shared<LogicalTensor>(function, selfTensor->Datatype(), resultShape, resultValidShape,
                                                  selfTensor->Format());
    auto& op = function.AddOperation(Opcode::OP_AXPY, {selfTensor, otherTensor}, {result});
    op.SetAttribute(OpAttributeKey::scalar, Element(selfTensor->Datatype(), alpha));
    std::map<int, int> inplaceInfo = {{0, 0}};
    op.SetAttr(OpAttributeKey::inplaceInfo, inplaceInfo);

    return result;
}

Tensor Axpy(const Tensor& self, const Tensor& other, float alpha)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Axpy");
    CheckVectorOperand(other.GetStorage(), "Axpy");

    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "AXPY");
    CheckTensorDimRange(other.GetStorage(), 1, NUM_VALUE_4, "AXPY");

    auto selfDtype = self.GetDataType();
    auto otherDtype = other.GetDataType();
    if (selfDtype == otherDtype) {
        std::unordered_set<DataType> supportedTypes = {DT_FP32, DT_FP16};
        CheckTensorDataType(self.GetStorage(), supportedTypes, "AXPY");
    } else {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, selfDtype == DT_FP32 && otherDtype == DT_FP16)
            << "AXPY: when dtype mismatch, only dst(y)=fp32 with src(x)=fp16 is supported.";
    }
    RETURN_CALL(AxpyOperation, *Program::GetInstance().GetCurrentFunction(), self, other, alpha);
}

REGISTER_OPERATION_TILED_FUNC(OP_AXPY, Opcode::OP_AXPY, AxpyOperationTileFunc);

} // namespace npu::tile_fwk
