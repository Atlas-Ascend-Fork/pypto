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
 * \file quantize.cpp
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
// Symmetric Quantization (FP32 -> INT8)
// =============================================================================

void TiledQuantizeSymmetric(Function& function, const TileShape& tileShape, size_t cur, Input& srcInput,
                            Input& scaleInput, Input& dstInput, int64_t axis, uint32_t workspaceSize)
{
    if (cur == dstInput.tensor.GetShape().size()) {
        auto srcTile = srcInput.tensor.GetStorage()->View(function, srcInput.tileInfo.shape, srcInput.tileInfo.offset);
        auto scaleTile = scaleInput.tensor.GetStorage()->View(function, scaleInput.tileInfo.shape,
                                                              scaleInput.tileInfo.offset);
        auto dstTile = dstInput.tensor.GetStorage()->View(function, dstInput.tileInfo.shape, dstInput.tileInfo.offset);

        Operation* op = nullptr;
        if (workspaceSize == 0) {
            op = &function.AddOperation(Opcode::OP_QUANTIZE_SYM, {srcTile, scaleTile}, {dstTile});
        } else {
            LogicalTensorPtr workspace = std::make_shared<LogicalTensor>(function, DT_INT32,
                                                                         std::vector<int64_t>{workspaceSize});
            op = &function.AddOperation(Opcode::OP_QUANTIZE_SYM, {srcTile, scaleTile}, {dstTile, workspace});
        }
        op->SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        return;
    }

    auto& vecTile = tileShape.GetVecTile();
    for (int64_t i = 0; i < dstInput.tensor.GetShape()[cur]; i += vecTile[cur]) {
        // Update dst tile info
        dstInput.tileInfo.shape[cur] = std::min(dstInput.tensor.GetShape()[cur] - i, vecTile[cur]);
        dstInput.tileInfo.offset[cur] = i;

        // Update src tile info - src has same shape as dst
        if (cur < srcInput.tensor.GetShape().size()) {
            srcInput.tileInfo.shape[cur] = std::min(srcInput.tensor.GetShape()[cur] - i, vecTile[cur]);
            srcInput.tileInfo.offset[cur] = i;
        }

        // Update scale tile info - scale may have different shape depending on axis
        if (cur < scaleInput.tensor.GetShape().size()) {
            // If scale's dimension is 1 (broadcast dimension), use modulo which gives 0
            // Otherwise use the same index as src/dst
            int64_t scaleIdx = i % scaleInput.tensor.GetShape()[cur];
            scaleInput.tileInfo.shape[cur] = std::min(scaleInput.tensor.GetShape()[cur] - scaleIdx, vecTile[cur]);
            scaleInput.tileInfo.offset[cur] = scaleIdx;
        }

        TiledQuantizeSymmetric(function, tileShape, cur + 1, srcInput, scaleInput, dstInput, axis, workspaceSize);
    }
}

void TiledQuantizeSymmetric(Function& function, const TileShape& tileShape, const LogicalTensorPtr& src,
                            const LogicalTensorPtr& scale, const LogicalTensorPtr& dst, int64_t axis,
                            uint32_t workspaceSize)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, src->shape.size() == src->offset.size())
        << "Source shape size and offset size should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, dst->shape.size() == dst->offset.size())
        << "Destination shape size and offset size should be equal";

    TileInfo srcTileInfo(src->shape.size(), src->offset.size());
    TileInfo scaleTileInfo(scale->shape.size(), scale->offset.size());
    TileInfo dstTileInfo(dst->shape.size(), dst->offset.size());

    auto srcInput = Input{Tensor(src), srcTileInfo};
    auto scaleInput = Input{Tensor(scale), scaleTileInfo};
    auto dstInput = Input{Tensor(dst), dstTileInfo};

    TiledQuantizeSymmetric(function, tileShape, 0, srcInput, scaleInput, dstInput, axis, workspaceSize);
}

// =============================================================================
// Asymmetric Quantization (FP32 -> UINT8)
// =============================================================================

void TiledQuantizeAsymmetric(Function& function, const TileShape& tileShape, size_t cur, Input& srcInput,
                             Input& scaleInput, Input& offsetInput, Input& dstInput, int64_t axis,
                             uint32_t workspaceSize)
{
    if (cur == dstInput.tensor.GetShape().size()) {
        auto srcTile = srcInput.tensor.GetStorage()->View(function, srcInput.tileInfo.shape, srcInput.tileInfo.offset);
        auto scaleTile = scaleInput.tensor.GetStorage()->View(function, scaleInput.tileInfo.shape,
                                                              scaleInput.tileInfo.offset);
        auto offsetTile = offsetInput.tensor.GetStorage()->View(function, offsetInput.tileInfo.shape,
                                                                offsetInput.tileInfo.offset);
        auto dstTile = dstInput.tensor.GetStorage()->View(function, dstInput.tileInfo.shape, dstInput.tileInfo.offset);

        Operation* op = nullptr;
        if (workspaceSize == 0) {
            op = &function.AddOperation(Opcode::OP_QUANTIZE_ASYM, {srcTile, scaleTile, offsetTile}, {dstTile});
        } else {
            LogicalTensorPtr workspace = std::make_shared<LogicalTensor>(function, DT_INT32,
                                                                         std::vector<int64_t>{workspaceSize});
            op = &function.AddOperation(Opcode::OP_QUANTIZE_ASYM, {srcTile, scaleTile, offsetTile},
                                        {dstTile, workspace});
        }
        op->SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        return;
    }

    auto& vecTile = tileShape.GetVecTile();
    for (int64_t i = 0; i < dstInput.tensor.GetShape()[cur]; i += vecTile[cur]) {
        // Update dst tile info
        dstInput.tileInfo.shape[cur] = std::min(dstInput.tensor.GetShape()[cur] - i, vecTile[cur]);
        dstInput.tileInfo.offset[cur] = i;

        // Update src tile info - src has same shape as dst
        if (cur < srcInput.tensor.GetShape().size()) {
            srcInput.tileInfo.shape[cur] = std::min(srcInput.tensor.GetShape()[cur] - i, vecTile[cur]);
            srcInput.tileInfo.offset[cur] = i;
        }

        // Update scale tile info - scale may have different shape depending on axis
        if (cur < scaleInput.tensor.GetShape().size()) {
            int64_t scaleIdx = i % scaleInput.tensor.GetShape()[cur];
            scaleInput.tileInfo.shape[cur] = std::min(scaleInput.tensor.GetShape()[cur] - scaleIdx, vecTile[cur]);
            scaleInput.tileInfo.offset[cur] = scaleIdx;
        }

        // Update offset tile info - offset has same shape as scale
        if (cur < offsetInput.tensor.GetShape().size()) {
            int64_t offsetIdx = i % offsetInput.tensor.GetShape()[cur];
            offsetInput.tileInfo.shape[cur] = std::min(offsetInput.tensor.GetShape()[cur] - offsetIdx, vecTile[cur]);
            offsetInput.tileInfo.offset[cur] = offsetIdx;
        }

        TiledQuantizeAsymmetric(function, tileShape, cur + 1, srcInput, scaleInput, offsetInput, dstInput, axis,
                                workspaceSize);
    }
}

void TiledQuantizeAsymmetric(Function& function, const TileShape& tileShape, const LogicalTensorPtr& src,
                             const LogicalTensorPtr& scale, const LogicalTensorPtr& offset, const LogicalTensorPtr& dst,
                             int64_t axis, uint32_t workspaceSize)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, src->shape.size() == src->offset.size())
        << "Source shape size and offset size should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, dst->shape.size() == dst->offset.size())
        << "Destination shape size and offset size should be equal";

    TileInfo srcTileInfo(src->shape.size(), src->offset.size());
    TileInfo scaleTileInfo(scale->shape.size(), scale->offset.size());
    TileInfo offsetTileInfo(offset->shape.size(), offset->offset.size());
    TileInfo dstTileInfo(dst->shape.size(), dst->offset.size());

    auto srcInput = Input{Tensor(src), srcTileInfo};
    auto scaleInput = Input{Tensor(scale), scaleTileInfo};
    auto offsetInput = Input{Tensor(offset), offsetTileInfo};
    auto dstInput = Input{Tensor(dst), dstTileInfo};

    TiledQuantizeAsymmetric(function, tileShape, 0, srcInput, scaleInput, offsetInput, dstInput, axis, workspaceSize);
}

// =============================================================================
// Tensor-level Quantization Operations
// =============================================================================

LogicalTensorPtr TensorQuantizeSymmetricOperation(Function& function, const LogicalTensorPtr& src,
                                                  const LogicalTensorPtr& scale, int64_t axis)
{
    // Output is INT8 for symmetric quantization
    auto result = std::make_shared<LogicalTensor>(function, DataType::DT_INT8, src->shape, src->GetDynValidShape());
    auto& op = function.AddOperation(Opcode::OP_QUANTIZE_SYM, {src, scale}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
    return result;
}

LogicalTensorPtr TensorQuantizeAsymmetricOperation(Function& function, const LogicalTensorPtr& src,
                                                   const LogicalTensorPtr& scale, const LogicalTensorPtr& offset,
                                                   int64_t axis)
{
    // Output is UINT8 for asymmetric quantization
    auto result = std::make_shared<LogicalTensor>(function, DataType::DT_UINT8, src->shape, src->GetDynValidShape());
    auto& op = function.AddOperation(Opcode::OP_QUANTIZE_ASYM, {src, scale, offset}, {result});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
    return result;
}

// =============================================================================
// Public Quantize API
// =============================================================================

Tensor PrepareQuantize1DInput(const Tensor& input, VecTile& originalVecTile)
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

Tensor RestoreQuantize1DResult(Tensor result, const Tensor& input, const VecTile& originalVecTile)
{
    std::vector<int64_t> originalShape = {input.GetShape()[0]};
    auto originalValidShape = input.GetStorage()->GetDynValidShape();
    result = Reshape(result, originalShape, originalValidShape);
    if (!originalVecTile.tile.empty()) {
        TileShape::Current().SetVecTile(originalVecTile);
    }
    return result;
}

Tensor QuantizeAlongLastAxis(const Tensor& input, const Tensor& scale, DataType dtype, int axis,
                             const Tensor& zeroPoints)
{
    bool isAsymmetric = (zeroPoints.GetStorage() != nullptr);
    if (isAsymmetric) {
        CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, dtype == DataType::DT_UINT8)
            << "Asymmetric quantization output type should be UINT8";
        return CALL(QuantizeAsymmetricOperation, *Program::GetInstance().GetCurrentFunction(), input.GetStorage(),
                    scale.GetStorage(), zeroPoints.GetStorage(), axis);
    }
    CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, dtype == DataType::DT_INT8)
        << "Symmetric quantization output type should be INT8";
    return CALL(QuantizeSymmetricOperation, *Program::GetInstance().GetCurrentFunction(), input.GetStorage(),
                scale.GetStorage(), axis);
}

Tensor QuantizeAlongSecondLastAxis(const Tensor& input, const Tensor& scale, DataType dtype, const Tensor& zeroPoints)
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

    Tensor quantizedResult = QuantizeAlongLastAxis(transposedInput, scale, dtype, -1, zeroPoints);
    quantizedResult.GetStorage()->UpdateDynValidShape(tmpValidShape);
    TileShape::Current().SetVecTile(tmpVectile);
    Tensor result = Transpose(quantizedResult, {secondLastDim, lastDim});
    result.GetStorage()->UpdateDynValidShape(input.GetStorage()->dynValidShape_);
    TileShape::Current().SetVecTile(oriVectile);
    return result;
}

} // namespace

Tensor Quantize(const Tensor& input, const Tensor& scale, DataType dtype, int axis, const Tensor& zeroPoints)
{
    DECLARE_TRACER();
    CheckVectorOperand(input.GetStorage(), "Quantize");
    CheckVectorOperand(scale.GetStorage(), "Quantize");

    if (zeroPoints.GetStorage() != nullptr) {
        CheckVectorOperand(zeroPoints.GetStorage(), "Quantize");
    }

    CHECK(VectorErrorCode::ERR_PARAM_SHAPE_DIM_UNSUPPORTED,
          input.GetShape().size() >= SHAPE_DIM1 && input.GetShape().size() <= SHAPE_DIM5)
        << "The shape.size() only supports 1~5";
    bool is1DInput = (input.GetShape().size() == 1);
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, !(is1DInput && axis == -NUM_VALUE_2))
        << "1D input only supports axis=-1, axis=-2 is not supported";

    VecTile originalVecTile;
    Tensor processedInput = is1DInput ? PrepareQuantize1DInput(input, originalVecTile) : input;
    std::vector<DataType> SUPPORT_INPUT_DATATYPES = {DataType::DT_FP32};
    CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED,
          std::find(SUPPORT_INPUT_DATATYPES.begin(), SUPPORT_INPUT_DATATYPES.end(), input.GetDataType()) !=
              SUPPORT_INPUT_DATATYPES.end())
        << "The input datatype is not supported";

    int ndim = static_cast<int>(processedInput.GetShape().size());
    int normalizedAxis = axis >= 0 ? axis - ndim : axis;
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, normalizedAxis == -NUM_VALUE_1 || normalizedAxis == -NUM_VALUE_2)
        << "Only axis=-1 (per-row) and axis=-2 (per-column) are supported";

    Tensor result = normalizedAxis == -NUM_VALUE_2 ?
                        QuantizeAlongSecondLastAxis(processedInput, scale, dtype, zeroPoints) :
                        QuantizeAlongLastAxis(processedInput, scale, dtype, normalizedAxis, zeroPoints);
    return is1DInput ? RestoreQuantize1DResult(result, input, originalVecTile) : result;
}

// =============================================================================
// Tile Function Registration
// =============================================================================

void QuantizeSymmetricOperationTileFunc(Function& function, const TileShape& tileShape,
                                        const std::vector<LogicalTensorPtr>& iOperand,
                                        const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    int64_t axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");

    // Calculate workspace size: same size as src (with int32_t type)
    auto shape = tileShape.GetVecTile();
    int dim = shape.size();
    // tmpbuf: same size as src, with int32_t type
    int64_t tmpRows = (dim >= NUM_VALUE_2) ? shape.tile[dim - NUM_VALUE_2] : 1;
    int64_t tmpCols = (dim >= 1) ? shape.tile[dim - 1] : 1;

    // tmpbuf need 32-byte alignment
    constexpr int64_t alignElements = NUM_VALUE_8; // 8 * 4 = 32 bytes
    tmpCols = (tmpCols + alignElements - 1) / alignElements * alignElements;
    tmpRows = (tmpRows + alignElements - 1) / alignElements * alignElements;

    // workspaceSize is element count, not bytes (LogicalTensor constructor takes shape)
    uint32_t workspaceSize = tmpRows * tmpCols;

    TiledQuantizeSymmetric(function, tileShape, iOperand[0], iOperand[1], oOperand[0], axis, workspaceSize);
}

void QuantizeAsymmetricOperationTileFunc(Function& function, const TileShape& tileShape,
                                         const std::vector<LogicalTensorPtr>& iOperand,
                                         const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    int64_t axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");

    // Calculate workspace size: same size as src (with int32_t type)
    auto shape = tileShape.GetVecTile();
    int dim = shape.size();
    // tmpbuf: same size as src, with int32_t type
    int64_t tmpRows = (dim >= NUM_VALUE_2) ? shape.tile[dim - NUM_VALUE_2] : 1;
    int64_t tmpCols = (dim >= 1) ? shape.tile[dim - 1] : 1;

    // tmpbuf need 32-byte alignment
    constexpr int64_t alignElements = NUM_VALUE_8; // 8 * 4 = 32 bytes
    tmpCols = (tmpCols + alignElements - 1) / alignElements * alignElements;
    tmpRows = (tmpRows + alignElements - 1) / alignElements * alignElements;

    // workspaceSize is element count, not bytes (LogicalTensor constructor takes shape)
    uint32_t workspaceSize = tmpRows * tmpCols;

    TiledQuantizeAsymmetric(function, tileShape, iOperand[0], iOperand[1], iOperand[NUM_VALUE_2], oOperand[0], axis,
                            workspaceSize);
}

REGISTER_OPERATION_TILED_FUNC(OP_QUANTIZE_SYM, Opcode::OP_QUANTIZE_SYM, QuantizeSymmetricOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_QUANTIZE_ASYM, Opcode::OP_QUANTIZE_ASYM, QuantizeAsymmetricOperationTileFunc);

} // namespace npu::tile_fwk
