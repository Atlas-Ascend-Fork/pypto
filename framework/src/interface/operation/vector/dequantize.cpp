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
 * \file dequantize.cpp
 * \brief Quantization operation implementation for INT8 symmetric and asymmetric quantization
 */

#include "interface/utils/operator_tracer.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation_common.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/operation/vector/tensor_transformation.h"

#include <unordered_set>

namespace npu::tile_fwk {

namespace {
constexpr size_t QUANT_MX_MIN_RANK = 1;

// =============================================================================
// Dequantization Operations (INT8/INT16 -> FP32)
// TDequant always requires 4 params: dst, src, scale, offset (symmetric: offset=0)
// =============================================================================

void TiledDequantize(Function& function, const TileShape& tileShape, size_t cur, Input& srcInput, Input& scaleInput,
                     Input& offsetInput, Input& dstInput, int64_t axis)
{
    if (cur == dstInput.tensor.GetShape().size()) {
        auto srcTile = srcInput.tensor.GetStorage()->View(function, srcInput.tileInfo.shape, srcInput.tileInfo.offset);
        auto scaleTile = scaleInput.tensor.GetStorage()->View(function, scaleInput.tileInfo.shape,
                                                              scaleInput.tileInfo.offset);
        auto offsetTile = offsetInput.tensor.GetStorage()->View(function, offsetInput.tileInfo.shape,
                                                                offsetInput.tileInfo.offset);
        auto dstTile = dstInput.tensor.GetStorage()->View(function, dstInput.tileInfo.shape, dstInput.tileInfo.offset);

        auto& op = function.AddOperation(Opcode::OP_DEQUANTIZE, {srcTile, scaleTile, offsetTile}, {dstTile});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        return;
    }

    auto& vecTile = tileShape.GetVecTile();
    for (int64_t i = 0; i < dstInput.tensor.GetShape()[cur]; i += vecTile[cur]) {
        dstInput.tileInfo.shape[cur] = std::min(dstInput.tensor.GetShape()[cur] - i, vecTile[cur]);
        dstInput.tileInfo.offset[cur] = i;

        if (cur < srcInput.tensor.GetShape().size()) {
            srcInput.tileInfo.shape[cur] = std::min(srcInput.tensor.GetShape()[cur] - i, vecTile[cur]);
            srcInput.tileInfo.offset[cur] = i;
        }

        if (cur < scaleInput.tensor.GetShape().size()) {
            int64_t scaleIdx = i % scaleInput.tensor.GetShape()[cur];
            scaleInput.tileInfo.shape[cur] = std::min(scaleInput.tensor.GetShape()[cur] - scaleIdx, vecTile[cur]);
            scaleInput.tileInfo.offset[cur] = scaleIdx;
        }

        if (cur < offsetInput.tensor.GetShape().size()) {
            int64_t offsetIdx = i % offsetInput.tensor.GetShape()[cur];
            offsetInput.tileInfo.shape[cur] = std::min(offsetInput.tensor.GetShape()[cur] - offsetIdx, vecTile[cur]);
            offsetInput.tileInfo.offset[cur] = offsetIdx;
        }

        TiledDequantize(function, tileShape, cur + 1, srcInput, scaleInput, offsetInput, dstInput, axis);
    }
}

void TiledDequantize(Function& function, const TileShape& tileShape, const LogicalTensorPtr& src,
                     const LogicalTensorPtr& scale, const LogicalTensorPtr& offset, const LogicalTensorPtr& dst,
                     int64_t axis)
{
    TileInfo srcTileInfo(src->shape.size(), src->offset.size());
    TileInfo scaleTileInfo(scale->shape.size(), scale->offset.size());
    TileInfo offsetTileInfo(offset->shape.size(), offset->offset.size());
    TileInfo dstTileInfo(dst->shape.size(), dst->offset.size());

    auto srcInput = Input{Tensor(src), srcTileInfo};
    auto scaleInput = Input{Tensor(scale), scaleTileInfo};
    auto offsetInput = Input{Tensor(offset), offsetTileInfo};
    auto dstInput = Input{Tensor(dst), dstTileInfo};

    TiledDequantize(function, tileShape, 0, srcInput, scaleInput, offsetInput, dstInput, axis);
}

LogicalTensorPtr TensorDequantizeOperation(Function& function, const LogicalTensorPtr& src,
                                           const LogicalTensorPtr& scale, const LogicalTensorPtr& offset, int64_t axis)
{
    auto result = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, src->shape, src->GetDynValidShape());
    auto& op = function.AddOperation(Opcode::OP_DEQUANTIZE, {src, scale, offset}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
    return result;
}

// Helper: create zero tensor for symmetric dequantization
static LogicalTensorPtr CreateZeroOffsetTensor(Function& function, const LogicalTensorPtr& scale)
{
    Element zeroVal(DataType::DT_FP32, (int64_t)0);

    Tensor zeroTensor = TensorFullOperation(function, zeroVal, SymbolicScalar(), DataType::DT_FP32, scale->shape,
                                            scale->GetDynValidShape());

    return zeroTensor.GetStorage();
}

Tensor PrepareDequantize1DInput(const Tensor& input, VecTile& originalVecTile)
{
    originalVecTile = TileShape::Current().GetVecTile();
    if (!originalVecTile.tile.empty()) {
        VecTile extendedVecTile = originalVecTile;
        extendedVecTile.tile.insert(extendedVecTile.tile.begin(), 1);
        TileShape::Current().SetVecTile(extendedVecTile);
    }

    std::vector<int64_t> newShape = {1, input.GetShape()[0]};
    auto originalValidShape = input.GetStorage()->GetDynValidShape();
    std::vector<SymbolicScalar> extendedValidShape;
    if (!originalValidShape.empty()) {
        extendedValidShape.push_back(SymbolicScalar(1));
        extendedValidShape.push_back(originalValidShape[0]);
    }
    return Reshape(input, newShape, extendedValidShape);
}

Tensor RestoreDequantize1DResult(Tensor result, const Tensor& input, const VecTile& originalVecTile)
{
    std::vector<int64_t> originalShape = {input.GetShape()[0]};
    auto originalValidShape = input.GetStorage()->GetDynValidShape();
    result = Reshape(result, originalShape, originalValidShape);
    if (!originalVecTile.tile.empty()) {
        TileShape::Current().SetVecTile(originalVecTile);
    }
    return result;
}

Tensor DequantizeAlongLastAxis(const Tensor& input, const Tensor& scale, int axis, const Tensor& zeroPoints)
{
    if (zeroPoints.GetStorage() != nullptr) {
        return CALL(DequantizeOperation, *Program::GetInstance().GetCurrentFunction(), input.GetStorage(),
                    scale.GetStorage(), zeroPoints.GetStorage(), axis);
    }
    auto zeroOffset = CreateZeroOffsetTensor(*Program::GetInstance().GetCurrentFunction(), scale.GetStorage());
    return CALL(DequantizeOperation, *Program::GetInstance().GetCurrentFunction(), input.GetStorage(),
                scale.GetStorage(), zeroOffset, axis);
}

Tensor DequantizeAlongSecondLastAxis(const Tensor& input, const Tensor& scale, const Tensor& zeroPoints)
{
    int ndim = static_cast<int>(input.GetShape().size());
    int lastDim = ndim - 1;
    int secondLastDim = ndim - NUM_VALUE_2;
    Tensor transposedInput = Transpose(input, {secondLastDim, lastDim});
    VecTile oriVectile = TileShape::Current().GetVecTile();
    VecTile tmpVectile = TileShape::Current().GetVecTile();
    std::swap(tmpVectile[secondLastDim], tmpVectile[lastDim]);
    TileShape::Current().SetVecTile(tmpVectile);
    auto tmpValidShape = input.GetStorage()->dynValidShape_;
    std::swap(tmpValidShape[secondLastDim], tmpValidShape[lastDim]);
    transposedInput.GetStorage()->UpdateDynValidShape(tmpValidShape);

    Tensor dequantizedResult = DequantizeAlongLastAxis(transposedInput, scale, -1, zeroPoints);
    dequantizedResult.GetStorage()->UpdateDynValidShape(tmpValidShape);
    TileShape::Current().SetVecTile(tmpVectile);
    Tensor result = Transpose(dequantizedResult, {secondLastDim, lastDim});
    result.GetStorage()->UpdateDynValidShape(input.GetStorage()->dynValidShape_);
    TileShape::Current().SetVecTile(oriVectile);
    return result;
}

} // namespace

// Public Dequantize API
Tensor Dequantize(const Tensor& input, const Tensor& scale, DataType otype, int axis, const Tensor& zeroPoints)
{
    DECLARE_TRACER();
    CheckVectorOperand(input.GetStorage(), "Dequantize");
    CheckVectorOperand(scale.GetStorage(), "Dequantize");

    if (zeroPoints.GetStorage() != nullptr) {
        CheckVectorOperand(zeroPoints.GetStorage(), "Dequantize");
    }

    CHECK(VectorErrorCode::ERR_PARAM_SHAPE_DIM_UNSUPPORTED,
          input.GetShape().size() >= SHAPE_DIM1 && input.GetShape().size() <= SHAPE_DIM5)
        << "The shape.size() only supports 1~5";
    bool is1DInput = (input.GetShape().size() == 1);
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, !(is1DInput && axis == -NUM_VALUE_2))
        << "1D input only supports axis=-1, axis=-2 is not supported";

    VecTile originalVecTile;
    Tensor processedInput = is1DInput ? PrepareDequantize1DInput(input, originalVecTile) : input;

    CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED,
          input.GetDataType() == DataType::DT_INT8 || input.GetDataType() == DataType::DT_INT16)
        << "Dequantize input dtype must be INT8 or INT16, but got dtype=" << static_cast<int>(input.GetDataType());
    CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, otype == DataType::DT_FP32)
        << "Dequantize output type must be FP32, but got dtype=" << static_cast<int>(otype);

    int ndim = static_cast<int>(processedInput.GetShape().size());
    int normalizedAxis = axis >= 0 ? axis - ndim : axis;
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, normalizedAxis == -NUM_VALUE_1 || normalizedAxis == -NUM_VALUE_2)
        << "Dequantize axis must be -1 (per-row) or -2 (per-column), but got axis=" << axis
        << " (normalized=" << normalizedAxis << ")";

    Tensor result = normalizedAxis == -NUM_VALUE_2 ?
                        DequantizeAlongSecondLastAxis(processedInput, scale, zeroPoints) :
                        DequantizeAlongLastAxis(processedInput, scale, normalizedAxis, zeroPoints);
    return is1DInput ? RestoreDequantize1DResult(result, input, originalVecTile) : result;
}

Tensor DequantizeSymmetric(const Tensor& src, const Tensor& scale, int64_t axis)
{
    CheckVectorOperand(src.GetStorage(), "DequantizeSymmetric");
    CheckVectorOperand(scale.GetStorage(), "DequantizeSymmetric");

    return Dequantize(src, scale, DataType::DT_FP32, axis, Tensor());
}

Tensor DequantizeAsymmetric(const Tensor& src, const Tensor& scale, const Tensor& zeroPoints, int64_t axis)
{
    CheckVectorOperand(src.GetStorage(), "DequantizeAsymmetric");
    CheckVectorOperand(scale.GetStorage(), "DequantizeAsymmetric");
    CheckVectorOperand(zeroPoints.GetStorage(), "DequantizeAsymmetric");

    return Dequantize(src, scale, DataType::DT_FP32, axis, zeroPoints);
}

// Tile Function Registration
void DequantizeOperationTileFunc(Function& function, const TileShape& tileShape,
                                 const std::vector<LogicalTensorPtr>& iOperand,
                                 const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    int64_t axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
    TiledDequantize(function, tileShape, iOperand[0], iOperand[1], iOperand[NUM_VALUE_2], oOperand[0], axis);
}

REGISTER_OPERATION_TILED_FUNC(OP_DEQUANTIZE, Opcode::OP_DEQUANTIZE, DequantizeOperationTileFunc);

} // namespace npu::tile_fwk
