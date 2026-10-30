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
 * \file cumulative.cpp
 * \brief
 */

#include "unary.h"
#include "binary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

struct CumOperationTileInfoPara {
    TileInfo inputTileInfo;
    TileInfo dstTileInfo;
};

struct CumOperationPara {
    const LogicalTensorPtr& input;
    const LogicalTensorPtr& dstTensor;
    const int axis;
    const bool is_sum;
};

// CumSum/CumProd tiling (rank 1-4):
//   Phase 1 — loop and tile every non-cum dimension (0..rank-1 except axis).
//   Phase 2 — within each fixed non-cum fiber, tile only along axis and propagate
//             carry (last cum-axis element) between consecutive tiles on that axis.
// Carry is scoped per non-cum fiber so higher-rank slices do not clobber each other.

namespace {

std::vector<int> BuildNonCumDimIndices(int rank, int axis)
{
    std::vector<int> dims;
    dims.reserve(rank > 0 ? static_cast<size_t>(rank - 1) : 0);
    for (int d = 0; d < rank; ++d) {
        if (d != axis) {
            dims.push_back(d);
        }
    }
    return dims;
}

void EmitCumAxisTile(Function& function, const CumOperationPara& cumOperationPara, CumOperationTileInfoPara& tileInfo,
                     LogicalTensorPtr& lastCarry)
{
    const LogicalTensorPtr& input = cumOperationPara.input;
    const LogicalTensorPtr& dstTensor = cumOperationPara.dstTensor;
    const int axis = cumOperationPara.axis;
    const bool is_sum = cumOperationPara.is_sum;

    auto dstTile = dstTensor->View(function, tileInfo.dstTileInfo.shape, tileInfo.dstTileInfo.offset);
    auto inputTile = input->View(function, tileInfo.inputTileInfo.shape, tileInfo.inputTileInfo.offset);

    LogicalTensorPtr srcTile = std::make_shared<LogicalTensor>(function, dstTile->Datatype(), dstTile->GetShape(),
                                                               inputTile->GetDynValidShape());
    if (is_sum) {
        auto& op = function.AddOperation(Opcode::OP_CUM_SUM, {inputTile}, {srcTile});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        op.SetAttribute(OP_ATTR_PREFIX + "flag", is_sum);
    } else {
        auto& op = function.AddOperation(Opcode::OP_CUM_PROD, {inputTile}, {srcTile});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        op.SetAttribute(OP_ATTR_PREFIX + "flag", is_sum);
    }

    if (tileInfo.dstTileInfo.offset[axis] > 0) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, lastCarry != nullptr)
            << "carry must be set when cum axis tile offset > 0";
        LogicalTensorPtr lastTile = std::make_shared<LogicalTensor>(function, srcTile->Datatype(), srcTile->GetShape(),
                                                                    srcTile->GetDynValidShape());
        auto& eop = function.AddOperation("TILE_EXPAND", {lastCarry}, {lastTile});
        eop.SetAttribute(OpAttributeKey::expandDims, std::vector<int>{axis});
        if (is_sum) {
            function.AddOperation(Opcode::OP_ADD, {srcTile, lastTile}, {dstTile});
        } else {
            function.AddOperation(Opcode::OP_MUL, {srcTile, lastTile}, {dstTile});
        }
    } else {
        function.AddOperation(Opcode::OP_REGISTER_COPY, {srcTile}, {dstTile});
    }

    std::vector<int64_t> lastShape = tileInfo.dstTileInfo.shape;
    lastShape[axis] = 1;
    std::vector<int64_t> lastViewOffset(lastShape.size(), 0);
    lastViewOffset[axis] = tileInfo.dstTileInfo.shape[axis] - 1;
    lastCarry = dstTile->View(function, lastShape, lastViewOffset);
}

void TiledCumAlongAxis(Function& function, const TileShape& tileShape, const CumOperationPara& cumOperationPara,
                       CumOperationTileInfoPara& tileInfo)
{
    const int axis = cumOperationPara.axis;
    const auto& input = cumOperationPara.input;
    auto& vecTile = tileShape.GetVecTile();
    const int64_t cumTileSize = vecTile[axis];
    const int64_t cumDim = input->GetShape()[axis];

    LogicalTensorPtr lastCarry = nullptr;
    for (int64_t i = 0; i < cumDim; i += cumTileSize) {
        tileInfo.dstTileInfo.offset[axis] = i;
        tileInfo.dstTileInfo.shape[axis] = std::min(cumDim - i, cumTileSize);
        tileInfo.inputTileInfo.offset[axis] = i;
        tileInfo.inputTileInfo.shape[axis] = std::min(cumDim - i, cumTileSize);
        EmitCumAxisTile(function, cumOperationPara, tileInfo, lastCarry);
    }
}

void InnerTiledCumNonAxisDims(size_t idx, const std::vector<int>& nonAxisDims, Function& function,
                              const TileShape& tileShape, const CumOperationPara& cumOperationPara,
                              CumOperationTileInfoPara& tileInfo)
{
    if (idx == nonAxisDims.size()) {
        TiledCumAlongAxis(function, tileShape, cumOperationPara, tileInfo);
        return;
    }

    const int dim = nonAxisDims[idx];
    const auto& input = cumOperationPara.input;
    auto& vecTile = tileShape.GetVecTile();
    const int64_t dimSize = input->GetShape()[dim];
    const int64_t tileSize = vecTile[dim];

    for (int64_t i = 0; i < dimSize; i += tileSize) {
        tileInfo.dstTileInfo.offset[dim] = i;
        tileInfo.dstTileInfo.shape[dim] = std::min(dimSize - i, tileSize);
        tileInfo.inputTileInfo.offset[dim] = i;
        tileInfo.inputTileInfo.shape[dim] = std::min(dimSize - i, tileSize);
        InnerTiledCumNonAxisDims(idx + 1, nonAxisDims, function, tileShape, cumOperationPara, tileInfo);
    }
}

} // namespace

void TiledCumOperation(Function& function, const TileShape& tileShape, const CumOperationPara& cumOperationPara)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          cumOperationPara.input->GetShape().size() == cumOperationPara.input->GetOffset().size())
        << "Shape size and offset size should be equal";

    const int rank = static_cast<int>(cumOperationPara.input->GetShape().size());
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          rank >= static_cast<int>(MIN_TENSOR_DIM) && rank <= static_cast<int>(MAX_TENSOR_DIM))
        << "CumSum/CumProd tiling supports rank 1-4";

    CumOperationTileInfoPara tileInfo{
        TileInfo(cumOperationPara.input->GetShape().size(), cumOperationPara.input->GetOffset().size()),
        TileInfo(cumOperationPara.dstTensor->GetShape().size(), cumOperationPara.dstTensor->GetOffset().size())};

    const std::vector<int> nonAxisDims = BuildNonCumDimIndices(rank, cumOperationPara.axis);
    InnerTiledCumNonAxisDims(0, nonAxisDims, function, tileShape, cumOperationPara, tileInfo);
}

LogicalTensorPtr AddCumCast(Function& function, const LogicalTensorPtr& input, DataType dataType, CastMode mode)
{
    auto result = std::make_shared<LogicalTensor>(function, dataType, input->GetShape(), input->GetDynValidShape());
    auto& op = function.AddOperation(Opcode::OP_CAST, {input}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "mode", mode);
    return result;
}

void AddCumCompute(Function& function, Opcode opcode, const LogicalTensorPtr& input, const LogicalTensorPtr& output,
                   const CumOperationPara& cumOperationPara)
{
    auto& op = function.AddOperation(opcode, {input}, {output});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", cumOperationPara.axis);
    op.SetAttribute(OP_ATTR_PREFIX + "flag", cumOperationPara.is_sum);
}

void AddCumOutputCast(Function& function, const LogicalTensorPtr& input, const CumOperationPara& cumOperationPara)
{
    cumOperationPara.dstTensor->UpdateDynValidShape(input->GetDynValidShape());
    auto& op = function.AddOperation(Opcode::OP_CAST, {input}, {cumOperationPara.dstTensor});
    op.SetAttribute(OP_ATTR_PREFIX + "mode", CastMode::CAST_NONE);
}

void TensorCumInt16Operation(Function& function, const CumOperationPara& cumOperationPara)
{
    auto inputFp32 = AddCumCast(function, cumOperationPara.input, DT_FP32, CastMode::CAST_NONE);
    auto inputInt32 = AddCumCast(function, inputFp32, DT_INT32, CastMode::CAST_TRUNC);
    auto outputInt32 = std::make_shared<LogicalTensor>(function, DT_INT32, cumOperationPara.dstTensor->GetShape(),
                                                       inputInt32->GetDynValidShape());
    AddCumCompute(function, Opcode::OP_CUM_SUM, inputInt32, outputInt32, cumOperationPara);
    AddCumOutputCast(function, outputInt32, cumOperationPara);
}

void TensorCumFp16Operation(Function& function, const CumOperationPara& cumOperationPara)
{
    auto inputFp32 = AddCumCast(function, cumOperationPara.input, DT_FP32, CastMode::CAST_NONE);
    auto outputFp32 = std::make_shared<LogicalTensor>(function, DT_FP32, cumOperationPara.dstTensor->GetShape(),
                                                      inputFp32->GetDynValidShape());
    auto opcode = cumOperationPara.is_sum ? Opcode::OP_CUM_SUM : Opcode::OP_CUM_PROD;
    AddCumCompute(function, opcode, inputFp32, outputFp32, cumOperationPara);
    AddCumOutputCast(function, outputFp32, cumOperationPara);
}

void TensorCumInt32Operation(Function& function, const CumOperationPara& cumOperationPara)
{
    auto outputInt32 = std::make_shared<LogicalTensor>(function, DT_INT32, cumOperationPara.dstTensor->GetShape(),
                                                       cumOperationPara.input->GetDynValidShape());
    AddCumCompute(function, Opcode::OP_CUM_SUM, cumOperationPara.input, outputInt32, cumOperationPara);
    AddCumOutputCast(function, outputInt32, cumOperationPara);
}

void TensorCumOperation(Function& function, const CumOperationPara& cumOperationPara)
{
    auto dataType = cumOperationPara.input->Datatype();
    if (dataType == DT_INT16) {
        TensorCumInt16Operation(function, cumOperationPara);
        return;
    }
    if (dataType == DT_BF16 || dataType == DT_FP16) {
        TensorCumFp16Operation(function, cumOperationPara);
        return;
    }
    if (dataType == DT_INT32) {
        TensorCumInt32Operation(function, cumOperationPara);
        return;
    }

    cumOperationPara.dstTensor->UpdateDynValidShape(cumOperationPara.input->GetDynValidShape());
    auto opcode = cumOperationPara.is_sum ? Opcode::OP_CUM_SUM : Opcode::OP_CUM_PROD;
    AddCumCompute(function, opcode, cumOperationPara.input, cumOperationPara.dstTensor, cumOperationPara);
}

void CheckCumOperation(const Tensor& input, const int& axis, const bool& is_sum)
{
    if (is_sum) {
        std::unordered_set<DataType> supportedTypes = {DT_FP32, DT_FP16, DT_INT32, DT_INT16, DT_BF16};
        CheckTensorDataType(input.GetStorage(), supportedTypes, "CUMSUM");
        CheckTensorDimRange(input.GetStorage(), 1, NUM_VALUE_4, "CUMSUM");
    } else {
        std::unordered_set<DataType> supportedTypes = {DT_FP32, DT_FP16, DT_BF16};
        CheckTensorDataType(input.GetStorage(), supportedTypes, "CUMPROD");
        CheckTensorDimRange(input.GetStorage(), 1, NUM_VALUE_4, "CUMPROD");
    }
    int tmpAxis0 = axis;
    CheckAxisRange(input, tmpAxis0);
    if (input.GetShape().size() == 1) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, tmpAxis0 == 0) << "when input.GetShape().size() is 1, axis must be 0";
    }
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, tmpAxis0 == 0 || static_cast<size_t>(tmpAxis0) < input.GetShape().size())
        << "The tmpAxis0 should be 0 and less than shape size";
}

Tensor CumOperation(const Tensor& input, const int& axis, const bool& is_sum)
{
    DECLARE_TRACER();
    CheckCumOperation(input, axis, is_sum);

    auto resultDType = input.GetDataType();
    int shapeSize = input.GetShape().size();
    int tmpAxis0 = axis < 0 ? shapeSize + axis : axis;

    if (resultDType == DataType::DT_INT16 || resultDType == DataType::DT_INT32) {
        resultDType = DataType::DT_INT64;
    }

    const int n_1 = shapeSize - 1;
    const int n_2 = shapeSize - NUM_VALUE_2;
    if ((resultDType != DataType::DT_INT64) && tmpAxis0 > 0 && tmpAxis0 == n_1) {
        Tensor tmpInput = Transpose(input, {n_2, n_1});
        const int transposedAxis = n_2;

        VecTile oriVectile = TileShape::Current().GetVecTile();
        VecTile tmpVectile = TileShape::Current().GetVecTile();
        int64_t tmp = tmpVectile.tile[n_2];
        tmpVectile.tile[n_2] = tmpVectile.tile[n_1];
        tmpVectile.tile[n_1] = tmp;
        TileShape::Current().SetVecTile(tmpVectile);

        auto tmpValidShape = input.GetStorage()->dynValidShape_;
        SymbolicScalar tmpValid = tmpValidShape[n_2];
        tmpValidShape[n_2] = tmpValidShape[n_1];
        tmpValidShape[n_1] = tmpValid;

        Tensor result(tmpInput.GetDataType(), tmpInput.GetShape());
        CALL(CumOperation, *Program::GetInstance().GetCurrentFunction(),
             {tmpInput.GetStorage(), result.GetStorage(), transposedAxis, is_sum});
        Tensor tmpresult = Transpose(result, {n_2, n_1});
        TileShape::Current().SetVecTile(oriVectile);
        return tmpresult;
    } else {
        Tensor result(resultDType, input.GetShape());
        CALL(CumOperation, *Program::GetInstance().GetCurrentFunction(),
             {input.GetStorage(), result.GetStorage(), tmpAxis0, is_sum});
        return result;
    }
}

Tensor CumSum(const Tensor& input, const int& axis)
{
    DECLARE_TRACER();
    CheckVectorOperand(input.GetStorage(), "CumSum");

    bool is_sum = true;
    Tensor result = CumOperation(input, axis, is_sum);
    return result;
}

Tensor CumProd(const Tensor& input, const int& axis)
{
    DECLARE_TRACER();
    CheckVectorOperand(input.GetStorage(), "CumProd");

    bool is_sum = false;
    Tensor result = CumOperation(input, axis, is_sum);
    return result;
}

void CumSumOperationTileFunc(Function& function, const TileShape& tileShape,
                             const std::vector<LogicalTensorPtr>& iOperand,
                             const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    int axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
    bool is_sum = op.GetBoolAttribute(OP_ATTR_PREFIX + "flag");
    TiledCumOperation(function, tileShape, {iOperand[0], oOperand[0], axis, is_sum});
}

REGISTER_OPERATION_TILED_FUNC(OP_CUM_SUM, Opcode::OP_CUM_SUM, CumSumOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_CUM_PROD, Opcode::OP_CUM_PROD, CumSumOperationTileFunc);

} // namespace npu::tile_fwk
