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
 * \file quant_mx.cpp
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
constexpr size_t QUANT_MX_MAX_RANK = NUM_VALUE_4;
constexpr int64_t QUANT_MX_GROUP_COLS = NUM_VALUE_32;
constexpr int64_t QUANT_MX_SCALE_GROUP_COLS = NUM_VALUE_64;
constexpr int64_t QUANT_MX_SCALE_PAIR_SIZE = NUM_VALUE_2;
constexpr int64_t QUANT_MX_TILE_ALIGN_BYTES = 256;
const std::unordered_set<DataType> QUANT_MX_SUPPORTED_INPUT_TYPES = {DataType::DT_FP16, DataType::DT_BF16,
                                                                     DataType::DT_FP32};
const std::unordered_set<DataType> QUANT_MX_SUPPORTED_OUTPUT_TYPES = {DataType::DT_FP8E4M3, DataType::DT_FP4_E2M1};
const std::vector<NPUArch> QUANT_MX_SUPPORTED_ARCHITECTURES = {NPUArch::DAV_3510};

int64_t CeilDiv(int64_t dividend, int64_t divisor)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, divisor != 0) << "CeilDiv divisor must not be zero.";
    return (dividend + divisor - 1) / divisor;
}

void CheckQuantMXDtype(DataType quantDtype)
{
    CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED,
          QUANT_MX_SUPPORTED_OUTPUT_TYPES.find(quantDtype) != QUANT_MX_SUPPORTED_OUTPUT_TYPES.end())
        << "QuantMX currently only supports DT_FP8E4M3 and DT_FP4_E2M1 output. Current quant dtype: "
        << DataType2String(quantDtype);
}

void CheckQuantMXDtypeCombination(DataType inputDtype, DataType quantDtype)
{
    if (quantDtype == DataType::DT_FP8E4M3) {
        CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED,
              inputDtype == DataType::DT_FP32 || inputDtype == DataType::DT_FP16 || inputDtype == DataType::DT_BF16)
            << "QuantMX DT_FP8E4M3 output only supports DT_FP32, DT_FP16, and DT_BF16 input.";
        return;
    }
    if (quantDtype == DataType::DT_FP4_E2M1) {
        CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED,
              inputDtype == DataType::DT_FP16 || inputDtype == DataType::DT_BF16)
            << "QuantMX DT_FP4_E2M1 output only supports DT_FP16 and DT_BF16 input.";
    }
}

void CheckQuantMXMode(DequantScaleRoundingMode mode)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          mode == DequantScaleRoundingMode::ROUND_DOWN || mode == DequantScaleRoundingMode::ROUND_UP)
        << "QuantMX currently only supports ROUND_DOWN (OCP) and ROUND_UP (NV) modes.";
}

void CheckQuantMXPerformanceMode(int64_t performanceMode)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, performanceMode == 0 || performanceMode == 1)
        << "QuantMX performance mode must be 0 or 1. Current performance mode: " << performanceMode;
}

DequantScaleRoundingMode GetQuantMXMode(const Operation& op)
{
    int64_t modeValue = 0;
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, op.GetAttr(OpAttributeKey::mxQuantMode, modeValue))
        << "QuantMX missing required attribute: " << OpAttributeKey::mxQuantMode;
    return static_cast<DequantScaleRoundingMode>(modeValue);
}

int64_t GetQuantMXAxis(const Operation& op)
{
    int64_t axis = 0;
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, op.GetAttr(OpAttributeKey::mxQuantAxis, axis))
        << "QuantMX missing required attribute: " << OpAttributeKey::mxQuantAxis;
    return axis;
}

int64_t GetQuantMXPerformanceMode(const Operation& op)
{
    int64_t performanceMode = 0;
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, op.GetAttr(OpAttributeKey::mxQuantPerformanceMode, performanceMode))
        << "QuantMX missing required attribute: " << OpAttributeKey::mxQuantPerformanceMode;
    return performanceMode;
}

int64_t NormalizeQuantMXAxis(int64_t axis, size_t rank)
{
    if (axis < 0) {
        axis += static_cast<int64_t>(rank);
    }
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, axis >= 0 && axis < static_cast<int64_t>(rank))
        << "QuantMX axis is out of range. Current axis: " << axis << ", input rank: " << rank;
    return axis;
}

void CheckQuantMXAxis(int64_t axis, size_t rank)
{
    const int64_t normalizedAxis = NormalizeQuantMXAxis(axis, rank);
    const int64_t lastAxis = static_cast<int64_t>(rank) - 1;
    const int64_t dnAxis = static_cast<int64_t>(rank) - NUM_VALUE_2;
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          normalizedAxis == lastAxis || (rank >= NUM_VALUE_2 && normalizedAxis == dnAxis))
        << "QuantMX currently only supports the last axis and second-last axis. Current axis: " << axis
        << ", input rank: " << rank;
}

bool IsQuantMXDnAxis(int64_t normalizedAxis, size_t rank)
{
    return rank >= NUM_VALUE_2 && normalizedAxis == static_cast<int64_t>(rank) - NUM_VALUE_2;
}

void CheckQuantMXInput(const Tensor& input, DataType quantDtype, DequantScaleRoundingMode mode, int64_t axis,
                       bool performanceMode)
{
    const auto inputDtype = input.GetDataType();
    CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED,
          QUANT_MX_SUPPORTED_INPUT_TYPES.find(inputDtype) != QUANT_MX_SUPPORTED_INPUT_TYPES.end())
        << "QuantMX currently only supports DT_FP16, DT_BF16, and DT_FP32 input.";
    CheckQuantMXDtype(quantDtype);
    CheckQuantMXDtypeCombination(inputDtype, quantDtype);
    CheckQuantMXMode(mode);
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, input.Format() == TileOpFormat::TILEOP_ND)
        << "QuantMX only supports TILEOP_ND input.";
    CHECK(VectorErrorCode::ERR_PARAM_SHAPE_DIM_UNSUPPORTED,
          QUANT_MX_MIN_RANK <= input.GetShape().size() && input.GetShape().size() <= QUANT_MX_MAX_RANK)
        << "QuantMX only supports 1D to 4D input.";
    const auto& inputShape = input.GetShape();
    CheckQuantMXAxis(axis, inputShape.size());
    const int64_t normalizedAxis = NormalizeQuantMXAxis(axis, inputShape.size());
    if (IsQuantMXDnAxis(normalizedAxis, inputShape.size())) {
        const int64_t dnDim = inputShape[normalizedAxis];
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, dnDim % QUANT_MX_SCALE_GROUP_COLS == 0)
            << "QuantMX axis=-2 requires the second-last dimension to be 64-aligned. Current dim: " << dnDim;
        if (quantDtype == DT_FP4_E2M1) {
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, inputShape.back() % QUANT_MX_SCALE_GROUP_COLS == 0)
                << "QuantMX FP4 axis=-2 requires view shape's last dim to be 64-aligned. Current dim: "
                << inputShape.back();
        }
        return;
    }
    if (!performanceMode) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, inputShape.back() % QUANT_MX_SCALE_GROUP_COLS == 0)
            << "QuantMX non-performance mode requires input last dim to be a multiple of 64. Current last dim: "
            << inputShape.back();
    }
    if (performanceMode) {
        const int64_t lastDimBytes = inputShape.back() * BytesOf(inputDtype);
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, lastDimBytes % QUANT_MX_TILE_ALIGN_BYTES == 0)
            << "QuantMX performance mode requires view shape's last dim to be 256-byte aligned. Current last dim "
               "bytes: "
            << lastDimBytes;
    }
}

void CheckQuantMXPerformanceTileShape(const LogicalTensorPtr& input, const VecTile& vecTile, int64_t axis,
                                      int64_t performanceMode)
{
    if (performanceMode == 0) {
        return;
    }
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, vecTile.size() == input->GetShape().size())
        << "QuantMX performance mode tile shape rank must match input rank.";
    const int64_t normalizedAxis = NormalizeQuantMXAxis(axis, input->GetShape().size());
    if (IsQuantMXDnAxis(normalizedAxis, input->GetShape().size())) {
        return;
    }
    const int64_t lastTileDim = vecTile[vecTile.size() - 1];
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, lastTileDim == input->GetShape().back())
        << "QuantMX performance mode requires tile shape last dim to be the same as input last dim. Current tile "
           "last dim: "
        << lastTileDim << ", input last dim: " << input->GetShape().back();
}

std::vector<int64_t> BuildQuantMXPerformanceGroupedShape(const std::vector<int64_t>& inputShape)
{
    if (inputShape.size() == 1) {
        return {CeilDiv(inputShape[0], QUANT_MX_GROUP_COLS)};
    }

    std::vector<int64_t> groupedShape;
    groupedShape.reserve(inputShape.size() - 1);
    for (size_t i = 0; i + NUM_VALUE_2 < inputShape.size(); ++i) {
        groupedShape.push_back(inputShape[i]);
    }
    groupedShape.push_back(inputShape[inputShape.size() - NUM_VALUE_2] *
                           CeilDiv(inputShape.back(), QUANT_MX_GROUP_COLS));
    return groupedShape;
}

std::vector<int64_t> BuildQuantMXScalingShape(const std::vector<int64_t>& groupedShape, DataType inputDtype)
{
    auto scalingShape = groupedShape;
    if (inputDtype == DataType::DT_FP32) {
        scalingShape.back() *= QUANT_MX_SCALE_PAIR_SIZE;
    }
    return scalingShape;
}

std::vector<int64_t> BuildQuantMXExpStorageShape(const std::vector<int64_t>& inputShape)
{
    auto expStorageShape = inputShape;
    expStorageShape.back() = CeilDiv(inputShape.back(), QUANT_MX_GROUP_COLS);
    return expStorageShape;
}

std::vector<int64_t> BuildQuantMXDnExpShape(const std::vector<int64_t>& inputShape)
{
    auto expShape = inputShape;
    expShape[expShape.size() - NUM_VALUE_2] /= QUANT_MX_SCALE_GROUP_COLS;
    expShape.back() *= QUANT_MX_SCALE_PAIR_SIZE;
    return expShape;
}

std::vector<int64_t> BuildQuantMXDnScratchShape(const std::vector<int64_t>& inputShape)
{
    auto scratchShape = inputShape;
    scratchShape[scratchShape.size() - NUM_VALUE_2] /= QUANT_MX_GROUP_COLS;
    return scratchShape;
}

std::vector<int64_t> BuildQuantMXPerformanceVecTile(const std::vector<int64_t>& inputVecTile)
{
    if (inputVecTile.size() == 1) {
        return {CeilDiv(inputVecTile[0], QUANT_MX_GROUP_COLS)};
    }

    std::vector<int64_t> groupedVecTile;
    groupedVecTile.reserve(inputVecTile.size() - 1);
    for (size_t i = 0; i + NUM_VALUE_2 < inputVecTile.size(); ++i) {
        groupedVecTile.push_back(inputVecTile[i]);
    }
    groupedVecTile.push_back(inputVecTile[inputVecTile.size() - NUM_VALUE_2] *
                             CeilDiv(inputVecTile.back(), QUANT_MX_GROUP_COLS));
    return groupedVecTile;
}

std::vector<int64_t> BuildQuantMXPerformanceGroupedOffset(const std::vector<int64_t>& inputOffset,
                                                          const std::vector<int64_t>& inputShape,
                                                          const std::vector<int64_t>& inputTileShape)
{
    if (inputOffset.size() == 1) {
        return {inputOffset[0] / QUANT_MX_GROUP_COLS};
    }

    std::vector<int64_t> groupedOffset;
    groupedOffset.reserve(inputOffset.size() - 1);
    for (size_t i = 0; i + NUM_VALUE_2 < inputOffset.size(); ++i) {
        groupedOffset.push_back(inputOffset[i]);
    }
    const int64_t groupCols = CeilDiv(inputShape.back(), QUANT_MX_GROUP_COLS);
    const int64_t tileRows = inputTileShape[inputTileShape.size() - NUM_VALUE_2];
    groupedOffset.push_back(inputOffset[inputOffset.size() - NUM_VALUE_2] * groupCols +
                            tileRows * (inputOffset.back() / QUANT_MX_GROUP_COLS));
    return groupedOffset;
}

std::vector<int64_t> BuildQuantMXScalingOffset(const std::vector<int64_t>& groupedOffset, DataType inputDtype)
{
    auto scalingOffset = groupedOffset;
    if (inputDtype == DataType::DT_FP32) {
        scalingOffset.back() *= QUANT_MX_SCALE_PAIR_SIZE;
    }
    return scalingOffset;
}

std::vector<SymbolicScalar> BuildQuantMXPerformanceGroupedValidShape(const std::vector<SymbolicScalar>& inputValidShape)
{
    if (inputValidShape.size() == 1) {
        return {(inputValidShape[0] + QUANT_MX_GROUP_COLS - 1) / QUANT_MX_GROUP_COLS};
    }

    std::vector<SymbolicScalar> groupedValidShape;
    groupedValidShape.reserve(inputValidShape.size() - 1);
    for (size_t i = 0; i + NUM_VALUE_2 < inputValidShape.size(); ++i) {
        groupedValidShape.push_back(inputValidShape[i]);
    }
    groupedValidShape.push_back(inputValidShape[inputValidShape.size() - NUM_VALUE_2] *
                                ((inputValidShape.back() + QUANT_MX_GROUP_COLS - 1) / QUANT_MX_GROUP_COLS));
    return groupedValidShape;
}

std::vector<SymbolicScalar> BuildQuantMXScalingValidShape(const std::vector<SymbolicScalar>& groupedValidShape,
                                                          DataType inputDtype)
{
    auto scalingValidShape = groupedValidShape;
    if (inputDtype == DataType::DT_FP32) {
        scalingValidShape.back() = scalingValidShape.back() * QUANT_MX_SCALE_PAIR_SIZE;
    }
    return scalingValidShape;
}

std::vector<SymbolicScalar> BuildQuantMXExpStorageValidShape(const std::vector<SymbolicScalar>& inputValidShape)
{
    auto expStorageValidShape = inputValidShape;
    expStorageValidShape.back() = (expStorageValidShape.back() + QUANT_MX_GROUP_COLS - 1) / QUANT_MX_GROUP_COLS;
    return expStorageValidShape;
}

std::vector<SymbolicScalar> BuildQuantMXDnExpValidShape(const std::vector<SymbolicScalar>& inputValidShape)
{
    auto expValidShape = inputValidShape;
    expValidShape[expValidShape.size() - NUM_VALUE_2] = expValidShape[expValidShape.size() - NUM_VALUE_2] /
                                                        QUANT_MX_SCALE_GROUP_COLS;
    expValidShape.back() = expValidShape.back() * QUANT_MX_SCALE_PAIR_SIZE;
    return expValidShape;
}

std::vector<SymbolicScalar> BuildQuantMXDnScratchValidShape(const std::vector<SymbolicScalar>& inputValidShape)
{
    auto scratchValidShape = inputValidShape;
    scratchValidShape[scratchValidShape.size() -
                      NUM_VALUE_2] = scratchValidShape[scratchValidShape.size() - NUM_VALUE_2] / QUANT_MX_GROUP_COLS;
    return scratchValidShape;
}

std::vector<int64_t> BuildQuantMXScaleShape(const std::vector<int64_t>& inputShape)
{
    auto scaleShape = inputShape;
    scaleShape.back() = CeilDiv(scaleShape.back(), QUANT_MX_SCALE_GROUP_COLS);
    scaleShape.push_back(QUANT_MX_SCALE_PAIR_SIZE);
    return scaleShape;
}

std::vector<int64_t> BuildQuantMXDnScaleShape(const std::vector<int64_t>& inputShape)
{
    auto scaleShape = inputShape;
    scaleShape[scaleShape.size() - NUM_VALUE_2] /= QUANT_MX_SCALE_GROUP_COLS;
    scaleShape.push_back(QUANT_MX_SCALE_PAIR_SIZE);
    return scaleShape;
}

std::vector<SymbolicScalar> BuildQuantMXScaleValidShape(const std::vector<SymbolicScalar>& inputValidShape)
{
    auto scaleValidShape = inputValidShape;
    scaleValidShape.back() = (scaleValidShape.back() + QUANT_MX_SCALE_GROUP_COLS - 1) / QUANT_MX_SCALE_GROUP_COLS;
    scaleValidShape.push_back(SymbolicScalar(QUANT_MX_SCALE_PAIR_SIZE));
    return scaleValidShape;
}

std::vector<SymbolicScalar> BuildQuantMXDnScaleValidShape(const std::vector<SymbolicScalar>& inputValidShape)
{
    auto scaleValidShape = inputValidShape;
    scaleValidShape[scaleValidShape.size() - NUM_VALUE_2] = scaleValidShape[scaleValidShape.size() - NUM_VALUE_2] /
                                                            QUANT_MX_SCALE_GROUP_COLS;
    scaleValidShape.push_back(SymbolicScalar(QUANT_MX_SCALE_PAIR_SIZE));
    return scaleValidShape;
}

void CheckQuantMXTileShape(const LogicalTensorPtr& input, const VecTile& vecTile, DataType quantDtype, int64_t axis,
                           int64_t performanceMode)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, vecTile.size() == input->GetShape().size())
        << "QuantMX tile shape rank must match input rank.";
    const int64_t normalizedAxis = NormalizeQuantMXAxis(axis, input->GetShape().size());
    if (IsQuantMXDnAxis(normalizedAxis, input->GetShape().size())) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, vecTile[normalizedAxis] > 0)
            << "QuantMX axis=-2 tile shape second-last dim must be positive.";
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, vecTile[normalizedAxis] % QUANT_MX_SCALE_GROUP_COLS == 0)
            << "QuantMX axis=-2 tile shape second-last dim must be 64-aligned. Current dim: "
            << vecTile[normalizedAxis];
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, vecTile[vecTile.size() - 1] > 0)
            << "QuantMX axis=-2 tile shape last dim must be positive.";
        if (quantDtype == DT_FP4_E2M1) {
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, vecTile[vecTile.size() - 1] % QUANT_MX_SCALE_GROUP_COLS == 0)
                << "QuantMX FP4 axis=-2 requires tile shape's last dim to be 64-aligned. Current dim: "
                << vecTile[vecTile.size() - 1];
        }
        return;
    }
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, vecTile[vecTile.size() - 1] > 0)
        << "QuantMX tile shape last dim must be positive.";

    if (performanceMode == 0) {
        return;
    }
    const int64_t lastDimBytes = vecTile[vecTile.size() - 1] * BytesOf(input->Datatype());
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, lastDimBytes % QUANT_MX_TILE_ALIGN_BYTES == 0)
        << "QuantMX performance mode requires tile shape's last dim to be 256-byte aligned. Current last dim bytes: "
        << lastDimBytes;
}

struct QuantMXTileContext {
    Function& function;
    Input& input;
    const LogicalTensorPtr& dst;
    const LogicalTensorPtr& exp;
    const LogicalTensorPtr& maxScratch;
    const LogicalTensorPtr& scalingScratch;
    DequantScaleRoundingMode mode;
    int64_t axis;
    int64_t performanceMode;
};

struct QuantMXTileParams {
    std::vector<int64_t> expShape;
    std::vector<int64_t> expOffset;
    std::vector<int64_t> groupedShape;
    std::vector<int64_t> groupedOffset;
    std::vector<int64_t> scalingShape;
    std::vector<int64_t> scalingOffset;
};

void CheckQuantMXTileAlignment(const QuantMXTileContext& ctx)
{
    const int64_t normalizedAxis = NormalizeQuantMXAxis(ctx.axis, ctx.input.tensor.GetShape().size());
    if (IsQuantMXDnAxis(normalizedAxis, ctx.input.tensor.GetShape().size())) {
        return;
    }
    if (ctx.performanceMode == 0) {
        return;
    }
    const int64_t lastDimBytes = ctx.input.tileInfo.shape.back() * BytesOf(ctx.input.tensor.GetDataType());
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, lastDimBytes % QUANT_MX_TILE_ALIGN_BYTES == 0)
        << "QuantMX performance mode requires tile width to be 256-byte aligned. Current last dim bytes: "
        << lastDimBytes;
}

QuantMXTileParams BuildQuantMXTileParams(const Input& input, int64_t performanceMode)
{
    QuantMXTileParams params;
    params.groupedShape = BuildQuantMXPerformanceGroupedShape(input.tileInfo.shape);
    params.groupedOffset = BuildQuantMXPerformanceGroupedOffset(input.tileInfo.offset, input.tensor.GetShape(),
                                                                input.tileInfo.shape);
    params.scalingShape = BuildQuantMXScalingShape(params.groupedShape, input.tensor.GetDataType());
    params.scalingOffset = BuildQuantMXScalingOffset(params.groupedOffset, input.tensor.GetDataType());
    if (performanceMode == 0) {
        params.expShape = input.tileInfo.shape;
        params.expShape.back() = CeilDiv(params.expShape.back(), QUANT_MX_GROUP_COLS);
        params.expOffset = input.tileInfo.offset;
        params.expOffset.back() /= QUANT_MX_GROUP_COLS;
        return params;
    }
    params.expShape = params.groupedShape;
    params.expOffset = params.groupedOffset;
    return params;
}

QuantMXTileParams BuildQuantMXDnTileParams(const Input& input)
{
    QuantMXTileParams params;
    params.expShape = input.tileInfo.shape;
    params.expShape[params.expShape.size() - NUM_VALUE_2] /= QUANT_MX_SCALE_GROUP_COLS;
    params.expShape.back() *= QUANT_MX_SCALE_PAIR_SIZE;
    params.expOffset = input.tileInfo.offset;
    params.expOffset[params.expOffset.size() - NUM_VALUE_2] /= QUANT_MX_SCALE_GROUP_COLS;
    params.expOffset.back() *= QUANT_MX_SCALE_PAIR_SIZE;
    params.groupedShape = input.tileInfo.shape;
    params.groupedShape[params.groupedShape.size() - NUM_VALUE_2] /= QUANT_MX_GROUP_COLS;
    params.groupedOffset = input.tileInfo.offset;
    params.groupedOffset[params.groupedOffset.size() - NUM_VALUE_2] /= QUANT_MX_GROUP_COLS;
    params.scalingShape = params.groupedShape;
    params.scalingOffset = params.groupedOffset;
    return params;
}

void EmitQuantMXTile(const QuantMXTileContext& ctx, const QuantMXTileParams& params)
{
    auto storage = ctx.input.tensor.GetStorage();
    auto srcTile = storage->View(ctx.function, ctx.input.tileInfo.shape, ctx.input.tileInfo.offset);
    auto dstTile = ctx.dst->View(ctx.function, ctx.input.tileInfo.shape, ctx.input.tileInfo.offset);
    auto expTile = ctx.exp->View(ctx.function, params.expShape, params.expOffset);
    auto maxTile = ctx.maxScratch->View(ctx.function, params.groupedShape, params.groupedOffset);
    auto scalingTile = ctx.scalingScratch->View(ctx.function, params.scalingShape, params.scalingOffset);
    auto& tiledOp = ctx.function.AddOperation(Opcode::OP_QUANT_MX, {srcTile}, {dstTile, expTile, maxTile, scalingTile});
    tiledOp.SetAttribute(OpAttributeKey::mxQuantMode, static_cast<int64_t>(ctx.mode));
    tiledOp.SetAttribute(OpAttributeKey::mxQuantAxis, ctx.axis);
    tiledOp.SetAttribute(OpAttributeKey::mxQuantPerformanceMode, ctx.performanceMode);
}

void TiledQuantMXOperationImpl(const QuantMXTileContext& ctx, const TileShape& tileShape, size_t cur)
{
    if (cur == ctx.input.tensor.GetShape().size()) {
        CheckQuantMXTileAlignment(ctx);
        const int64_t normalizedAxis = NormalizeQuantMXAxis(ctx.axis, ctx.input.tensor.GetShape().size());
        const auto params = IsQuantMXDnAxis(normalizedAxis, ctx.input.tensor.GetShape().size()) ?
                                BuildQuantMXDnTileParams(ctx.input) :
                                BuildQuantMXTileParams(ctx.input, ctx.performanceMode);
        EmitQuantMXTile(ctx, params);
        return;
    }

    const auto& vecTile = tileShape.GetVecTile();
    int64_t step = std::max<int64_t>(1, std::min<int64_t>(vecTile[cur], ctx.input.tensor.GetShape()[cur]));
    for (int64_t i = 0; i < ctx.input.tensor.GetShape()[cur]; i += step) {
        ctx.input.tileInfo.shape[cur] = std::min(ctx.input.tensor.GetShape()[cur] - i, step);
        ctx.input.tileInfo.offset[cur] = i;
        TiledQuantMXOperationImpl(ctx, tileShape, cur + 1);
    }
}

void TiledQuantMXOperation(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                           const LogicalTensorPtr& dst, const LogicalTensorPtr& exp, const LogicalTensorPtr& maxScratch,
                           const LogicalTensorPtr& scalingScratch, DequantScaleRoundingMode mode, int64_t axis,
                           int64_t performanceMode)
{
    QuantMXTileContext ctx{function, input, dst, exp, maxScratch, scalingScratch, mode, axis, performanceMode};
    TiledQuantMXOperationImpl(ctx, tileShape, cur);
}

void QuantMXTileFunc(Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
                     const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, iOperand.size() == NUM_VALUE_1) << "QuantMX expects 1 input tensor.";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, oOperand.size() == NUM_VALUE_4) << "QuantMX expects 4 output tensors.";

    const auto& src = iOperand[0];
    const auto& dst = oOperand[0];
    const auto& exp = oOperand[1];
    const auto& maxScratch = oOperand[NUM_VALUE_2];
    const auto& scalingScratch = oOperand[NUM_VALUE_3];
    CheckQuantMXDtype(dst->Datatype());
    auto mode = GetQuantMXMode(op);
    auto axis = GetQuantMXAxis(op);
    auto performanceMode = GetQuantMXPerformanceMode(op);
    CheckQuantMXMode(mode);
    CheckQuantMXPerformanceMode(performanceMode);
    CheckQuantMXAxis(axis, src->GetShape().size());
    CheckQuantMXTileShape(src, tileShape.GetVecTile(), dst->Datatype(), axis, performanceMode);
    CheckQuantMXPerformanceTileShape(src, tileShape.GetVecTile(), axis, performanceMode);
    TileInfo inputTileInfo(src->shape.size(), src->offset.size());
    auto input = Input{Tensor(src), inputTileInfo};
    TiledQuantMXOperation(function, tileShape, 0, input, dst, exp, maxScratch, scalingScratch, mode, axis,
                          performanceMode);
}
} // namespace

std::tuple<Tensor, Tensor> QuantMX(const Tensor& input, DataType quantDtype, DequantScaleRoundingMode mode,
                                   int64_t axis, bool performanceMode)
{
    DECLARE_TRACER();
    CheckVectorOperand(input.GetStorage(), "QuantMX");

    CheckSupportedNPUArch(QUANT_MX_SUPPORTED_ARCHITECTURES, "QuantMX");
    CheckQuantMXPerformanceMode(static_cast<int64_t>(performanceMode));
    CheckQuantMXInput(input, quantDtype, mode, axis, performanceMode);
    const auto oldVecTile = TileShape::Current().GetVecTile();
    const auto& inputShape = input.GetShape();
    const int64_t normalizedAxis = NormalizeQuantMXAxis(axis, inputShape.size());
    const bool isDnAxis = IsQuantMXDnAxis(normalizedAxis, inputShape.size());
    if (!oldVecTile.tile.empty() && isDnAxis && quantDtype == DT_FP4_E2M1) {
        CheckQuantMXTileShape(input.GetStorage(), oldVecTile, quantDtype, normalizedAxis,
                              static_cast<int64_t>(performanceMode));
    }
    if (performanceMode && !oldVecTile.tile.empty()) {
        CheckQuantMXPerformanceTileShape(input.GetStorage(), oldVecTile, normalizedAxis,
                                         static_cast<int64_t>(performanceMode));
    }

    const std::vector<int64_t> groupedShape = isDnAxis ? BuildQuantMXDnScratchShape(inputShape) :
                                                         BuildQuantMXPerformanceGroupedShape(inputShape);
    const std::vector<int64_t> expShape = isDnAxis ? BuildQuantMXDnExpShape(inputShape) :
                                                     (performanceMode ? groupedShape :
                                                                        BuildQuantMXExpStorageShape(inputShape));
    const std::vector<int64_t> scaleShape = isDnAxis ? BuildQuantMXDnScaleShape(inputShape) :
                                                       BuildQuantMXScaleShape(inputShape);

    const auto scratchDtype = input.GetDataType();
    const auto maxScratchShape = groupedShape;
    const auto scalingShape = isDnAxis ? groupedShape : BuildQuantMXScalingShape(groupedShape, scratchDtype);
    auto quantized = Tensor(quantDtype, inputShape, "", TileOpFormat::TILEOP_ND);
    auto exp = Tensor(DataType::DT_FP8E8M0, expShape, "", TileOpFormat::TILEOP_ND);
    auto maxScratch = Tensor(scratchDtype, maxScratchShape, "", TileOpFormat::TILEOP_ND);
    auto scalingScratch = Tensor(scratchDtype, scalingShape, "", TileOpFormat::TILEOP_ND);

    std::vector<SymbolicScalar> scaleValidShape;
    const auto& inputValidShape = input.GetStorage()->GetDynValidShape();
    if (!inputValidShape.empty()) {
        quantized.GetStorage()->UpdateDynValidShape(inputValidShape);
        const auto groupedValidShape = isDnAxis ? BuildQuantMXDnScratchValidShape(inputValidShape) :
                                                  BuildQuantMXPerformanceGroupedValidShape(inputValidShape);
        const auto expValidShape = isDnAxis ? BuildQuantMXDnExpValidShape(inputValidShape) :
                                              (performanceMode ? groupedValidShape :
                                                                 BuildQuantMXExpStorageValidShape(inputValidShape));
        exp.GetStorage()->UpdateDynValidShape(expValidShape);
        maxScratch.GetStorage()->UpdateDynValidShape(groupedValidShape);
        const auto scalingValidShape = isDnAxis ? groupedValidShape :
                                                  BuildQuantMXScalingValidShape(groupedValidShape, scratchDtype);
        scalingScratch.GetStorage()->UpdateDynValidShape(scalingValidShape);
        scaleValidShape = isDnAxis ? BuildQuantMXDnScaleValidShape(inputValidShape) :
                                     BuildQuantMXScaleValidShape(inputValidShape);
    }

    auto& op = Program::GetInstance().GetCurrentFunction()->AddOperation(
        Opcode::OP_QUANT_MX, {input.GetStorage()},
        {quantized.GetStorage(), exp.GetStorage(), maxScratch.GetStorage(), scalingScratch.GetStorage()});
    op.SetAttribute(OpAttributeKey::mxQuantMode, static_cast<int64_t>(mode));
    op.SetAttribute(OpAttributeKey::mxQuantAxis, normalizedAxis);
    op.SetAttribute(OpAttributeKey::mxQuantPerformanceMode, static_cast<int64_t>(performanceMode ? 1 : 0));
    if (!isDnAxis && performanceMode && !oldVecTile.tile.empty()) {
        TileShape::Current().SetVecTile(BuildQuantMXPerformanceVecTile(oldVecTile.tile));
    }
    auto scale = Reshape(exp, scaleShape, scaleValidShape);
    if (!isDnAxis && performanceMode && !oldVecTile.tile.empty()) {
        TileShape::Current().SetVecTile(oldVecTile);
    }
    return std::tie(quantized, scale);
}

REGISTER_OPERATION_TILED_FUNC(QuantMX, Opcode::OP_QUANT_MX, QuantMXTileFunc);

} // namespace npu::tile_fwk
