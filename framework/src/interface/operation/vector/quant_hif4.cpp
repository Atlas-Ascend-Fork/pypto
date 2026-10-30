/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "interface/utils/operator_tracer.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation_common.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "tilefwk/hif4_scratch.h"

namespace npu::tile_fwk {
namespace {
constexpr int64_t kBlockSize = 64;

std::vector<int64_t> ExpShape(std::vector<int64_t> shape)
{
    shape.back() = (shape.back() + kBlockSize - 1) / kBlockSize;
    return shape;
}

std::vector<int64_t> TileStep(const std::vector<int64_t>& shape, const VecTile& tile)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, tile.size() == shape.size())
        << "QuantHiF4 tile shape rank must match input rank.";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, tile[tile.size() - 1] > 0 && tile[tile.size() - 1] % kBlockSize == 0)
        << "QuantHiF4 tile shape last dimension must be a positive multiple of 64.";
    auto step = shape;
    for (size_t i = 0; i < shape.size(); ++i) {
        step[i] = std::min(shape[i], std::max<int64_t>(1, tile[i]));
    }
    step.back() = std::min<int64_t>(hif4::Align64(shape.back()), tile[tile.size() - 1]);
    return step;
}

int64_t ScratchBytes(const std::vector<int64_t>& capacity)
{
    size_t rows = capacity.size() > 1 ? capacity[capacity.size() - 2] : 1;
    return hif4::ScratchLayout(rows, capacity.back()).bytes;
}

void EmitTiles(Function& function, const TileShape& tileShape, const LogicalTensorPtr& src,
               const std::vector<LogicalTensorPtr>& outputs, std::vector<int64_t>& shape, std::vector<int64_t>& offset,
               size_t dim, int64_t scaleAlg, int64_t axis)
{
    if (dim < shape.size()) {
        const auto step = TileStep(src->GetShape(), tileShape.GetVecTile())[dim];
        for (int64_t i = 0; i < src->GetShape()[dim]; i += step) {
            shape[dim] = std::min(step, src->GetShape()[dim] - i);
            offset[dim] = i;
            EmitTiles(function, tileShape, src, outputs, shape, offset, dim + 1, scaleAlg, axis);
        }
        return;
    }
    auto expOffset = offset;
    expOffset.back() /= kBlockSize;
    auto input = src->View(function, shape, offset);
    auto dst = outputs[0]->View(function, shape, offset);
    auto exp = outputs[1]->View(function, ExpShape(shape), expOffset);
    auto scratch = outputs[2]->View(function, outputs[2]->GetShape(), {0});
    auto& tiled = function.AddOperation(Opcode::OP_QUANT_HIF4, {input}, {dst, exp, scratch});
    tiled.SetAttribute(OpAttributeKey::hif4ScaleAlg, scaleAlg);
    tiled.SetAttribute(OpAttributeKey::hif4QuantAxis, axis);
}

void QuantHiF4TileFunc(Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& inputs,
                       const std::vector<LogicalTensorPtr>& outputs, const Operation& op)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, inputs.size() == 1 && outputs.size() == 3)
        << "QuantHiF4 expects one input and three outputs (dst, exp, scratch).";
    int64_t scaleAlg = 0;
    int64_t axis = -1;
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, op.GetAttr(OpAttributeKey::hif4ScaleAlg, scaleAlg))
        << "QuantHiF4 missing scale algorithm.";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, op.GetAttr(OpAttributeKey::hif4QuantAxis, axis))
        << "QuantHiF4 missing axis.";
    std::vector<int64_t> shape(inputs[0]->GetShape().size()), offset(shape.size());
    EmitTiles(function, tileShape, inputs[0], outputs, shape, offset, 0, scaleAlg, axis);
}
} // namespace

std::tuple<Tensor, Tensor> QuantHiF4(const Tensor& input, DataType quantDtype, int64_t scaleAlg, int64_t axis)
{
    DECLARE_TRACER();
    CheckSupportedNPUArch({NPUArch::DAV_9201}, "QuantHiF4");
    CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, input.GetDataType() == DT_BF16 && quantDtype == DT_HF4)
        << "QuantHiF4 requires BF16 input and HF4 output.";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, input.Format() == TileOpFormat::TILEOP_ND)
        << "QuantHiF4 requires ND input.";
    const auto& shape = input.GetShape();
    CHECK(VectorErrorCode::ERR_PARAM_SHAPE_DIM_UNSUPPORTED, !shape.empty() && shape.size() <= 4)
        << "QuantHiF4 supports 1D to 4D input.";
    if (axis < 0) {
        axis += static_cast<int64_t>(shape.size());
    }
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, axis == static_cast<int64_t>(shape.size()) - 1)
        << "QuantHiF4 supports the last axis.";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, scaleAlg == 0 || scaleAlg == 1) << "QuantHiF4 scaleAlg must be 0 or 1.";
    const auto& valid = input.GetStorage()->GetDynValidShape();
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, shape.back() > 0 && shape.back() % kBlockSize == 0)
        << "QuantHiF4 requires a positive 64-aligned ViewShape last dimension; valid last dimension must be even.";
    const auto& tile = TileShape::Current().GetVecTile();
    const auto capacity = tile.tile.empty() ? shape : TileStep(shape, tile);
    const auto scratchBytes = ScratchBytes(capacity);
    Tensor dst(quantDtype, shape, "", TileOpFormat::TILEOP_ND);
    Tensor exp(DT_UINT32, ExpShape(shape), "", TileOpFormat::TILEOP_ND);
    Tensor scratch(DT_UINT8, {scratchBytes}, "", TileOpFormat::TILEOP_ND);
    if (!valid.empty()) {
        dst.GetStorage()->UpdateDynValidShape(valid);
        auto expValid = valid;
        expValid.back() = (expValid.back() + kBlockSize - 1) / kBlockSize;
        exp.GetStorage()->UpdateDynValidShape(expValid);
        scratch.GetStorage()->UpdateDynValidShape({scratchBytes});
    }
    auto& op = Program::GetInstance().GetCurrentFunction()->AddOperation(
        Opcode::OP_QUANT_HIF4, {input.GetStorage()}, {dst.GetStorage(), exp.GetStorage(), scratch.GetStorage()});
    op.SetAttribute(OpAttributeKey::hif4ScaleAlg, scaleAlg);
    op.SetAttribute(OpAttributeKey::hif4QuantAxis, axis);
    return std::tie(dst, exp);
}
REGISTER_OPERATION_TILED_FUNC(QuantHiF4, Opcode::OP_QUANT_HIF4, QuantHiF4TileFunc);
} // namespace npu::tile_fwk
