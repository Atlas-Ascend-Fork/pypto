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
 * \file pack.cpp
 * \brief
 */

#include "unary.h"
#include "interface/utils/operator_tracer.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

Tensor Pack(const Tensor& self)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Pack");

    std::unordered_set<DataType> supportedTypes = {DT_BF16,  DT_FP16,   DT_FP32,  DT_UINT8,  DT_INT8, DT_UINT16,
                                                   DT_INT16, DT_UINT32, DT_INT32, DT_UINT64, DT_INT64};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "Pack");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "Pack");
    std::vector<int64_t> tmpShape;
    auto selfShape = self.GetShape();
    int64_t dim = 1;
    for (size_t i = 0; i < selfShape.size(); ++i) {
        dim *= selfShape[i];
    }
    tmpShape.emplace_back(dim);
    auto tmp = Reshape(self, tmpShape).GetStorage();
    auto& function = *Program::GetInstance().GetCurrentFunction();
    int64_t byte = BytesOf(self.GetDataType());
    tmpShape[0] *= byte;
    std::vector<SymbolicScalar> tmpDynValidShape{tmp->GetDynValidShape()[0] * byte};
    auto result = std::make_shared<LogicalTensor>(function, DT_UINT8, tmpShape, tmpDynValidShape, tmp->Format());
    [[maybe_unused]] auto useless = &function.AddOperation(Opcode::OP_PACK, {tmp}, {result});
    return result;
}

Tensor UnPack(const Tensor& self, DataType dstDataType)
{
    DECLARE_TRACER();

    std::unordered_set<DataType> supportedSrcTypes = {DT_UINT8};
    std::unordered_set<DataType> supportedDstTypes = {DT_FP32,  DT_FP16,  DT_BF16,   DT_INT8,   DT_INT16, DT_INT32,
                                                      DT_INT64, DT_UINT8, DT_UINT64, DT_UINT16, DT_UINT32};
    CheckVectorOperand(self.GetStorage(), "UnPack");

    CheckTensorDataType(self.GetStorage(), supportedSrcTypes, "UnPack");
    CheckTensorDimRange(self.GetStorage(), 1, 1, "UnPack");
    CheckTensorDataType(dstDataType, supportedDstTypes, "UnPack");
    int64_t dstByte = BytesOf(dstDataType);
    auto srcShape = self.GetShape();
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, dstByte != 0 && srcShape[0] * BytesOf(DT_UINT8) % dstByte == 0)
        << "Cannot unpack tile to " << DataType2String(dstDataType, true) << ": total bytes not divisible by "
        << dstByte << " bytes";

    std::vector<int64_t> resultShape;
    resultShape.emplace_back(srcShape[0] / dstByte);
    const auto& srcStorage = self.GetStorage();
    std::vector<SymbolicScalar> dstDynValidShape{srcStorage->GetDynValidShape()[0] / dstByte};
    auto& function = *Program::GetInstance().GetCurrentFunction();
    auto result = std::make_shared<LogicalTensor>(function, dstDataType, resultShape, dstDynValidShape,
                                                  srcStorage->Format());

    [[maybe_unused]] auto useless = &function.AddOperation(Opcode::OP_UNPACK, {self.GetStorage()}, {result});
    return result;
}

void PackOperationTileFunc(Function& function, const TileShape& tileShape,
                           const std::vector<LogicalTensorPtr>& iOperand, const std::vector<LogicalTensorPtr>& oOperand,
                           [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto vecTile = tileShape.GetVecTile();
    int64_t byte = BytesOf(iOperand[0]->Datatype());
    int64_t tile = 1;
    for (size_t i = 0; i < vecTile.size(); ++i) {
        tile *= vecTile[i];
    }
    for (int64_t i = 0; i < iOperand[0]->shape[0]; i += tile) {
        std::vector<int64_t> inputShape{std::min(iOperand[0]->shape[0] - i, tile)};
        std::vector<int64_t> outputShape{inputShape[0] * byte};
        std::vector<int64_t> inputOffset{i};
        std::vector<int64_t> outputOffset{i * byte};
        auto inputTile = iOperand[0]->View(function, inputShape, inputOffset);
        auto resultTile = oOperand[0]->View(function, outputShape, outputOffset);
        [[maybe_unused]] auto useless = &function.AddOperation(Opcode::OP_PACK, {inputTile}, {resultTile});
    }
}

void UnPackOperationTileFunc(Function& function, const TileShape& tileShape,
                             const std::vector<LogicalTensorPtr>& iOperand,
                             const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    auto vecTile = tileShape.GetVecTile();
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, vecTile.size() != 0) << "UnPack tile shape invalid: vecTile is empty";
    auto dstByte = BytesOf(oOperand[0]->Datatype());
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, dstByte != 0 && vecTile[0] % dstByte == 0)
        << "UnPack tile shape invalid: vecTile[0]=" << vecTile[0] << " must be divisible by dstByte=" << dstByte;

    for (int64_t i = 0; i < iOperand[0]->shape[0]; i += vecTile[0]) {
        std::vector<int64_t> inputShape{std::min(iOperand[0]->shape[0] - i, vecTile[0])};
        std::vector<int64_t> offset{i};
        std::vector<int64_t> outputShape{static_cast<int64_t>(inputShape[0] / dstByte)};
        std::vector<int64_t> outputOffset{static_cast<int64_t>(i / dstByte)};
        auto inputTile = iOperand[0]->View(function, inputShape, offset);
        auto resultTile = oOperand[0]->View(function, outputShape, outputOffset);
        [[maybe_unused]] auto useless = &function.AddOperation(Opcode::OP_UNPACK, {inputTile}, {resultTile});
    }
}

REGISTER_OPERATION_TILED_FUNC(OP_PACK, Opcode::OP_PACK, PackOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_UNPACK, Opcode::OP_UNPACK, UnPackOperationTileFunc);

} // namespace npu::tile_fwk
