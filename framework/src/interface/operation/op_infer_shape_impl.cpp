/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file op_infer_shape_impl.cpp
 * \brief
 */

#include "op_infer_shape_impl.h"
#include "interface/operation/attr_holder.h"
#include "interface/operation/operation.h"
#include "interface/operation/operation_impl.h"
#include "interface/operation/vector/gather_mask_common.h"
#include "interface/tensor/symbolic_scalar.h"
#include "interface/utils/common.h"
#include "tilefwk/error_code.h"
#include "tilefwk/error_code.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {
const std::string COPY_OUT_FORCE_INFER_SHAPE = "copy_out_force_infer_shape";

void BinaryBrcinlineInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    auto validShape0 = op->GetIOperands()[0]->GetDynValidShape();
    auto validShape1 = op->GetIOperands()[1]->GetDynValidShape();

    std::vector<SymbolicScalar> outputValidShape = validShape0;
    std::vector<int64_t> brcOperand;
    if (op->GetAttr(OpAttributeKey::brcOperand, brcOperand)) {
        for (size_t i = 0; i < outputValidShape.size(); i++) {
            if (brcOperand[i] == 1) {
                outputValidShape[i] = validShape1[i];
            }
        }
    }

    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outputValidShape);
    }
}

REGISTER_INFER_SHAPE_FUNC(OP_ADD, Opcode::OP_ADD, BinaryBrcinlineInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_SUB, Opcode::OP_SUB, BinaryBrcinlineInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_MUL, Opcode::OP_MUL, BinaryBrcinlineInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_DIV, Opcode::OP_DIV, BinaryBrcinlineInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_MAXIMUM, Opcode::OP_MAXIMUM, BinaryBrcinlineInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_MINIMUM, Opcode::OP_MINIMUM, BinaryBrcinlineInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_AXPY, Opcode::OP_AXPY, BinaryBrcinlineInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_EXPANDEXPDIF, Opcode::OP_EXPANDEXPDIF, BinaryBrcinlineInferFunc);

void ElewiseInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    auto inputNum = op->GetIOperands().size();
    auto shapeDimNum = op->GetIOperands()[0]->GetDynValidShape().size();
    // 将每个输入的同一维shape值填充到一个vector中，便于后续对每一维进行筛选
    std::vector<std::vector<SymbolicScalar>> dimValidShape(shapeDimNum,
                                                           std::vector<SymbolicScalar>(inputNum, SymbolicScalar()));
    std::vector<std::vector<int64_t>> dimShape(shapeDimNum, std::vector<int64_t>(inputNum, 0));
    for (size_t i = 0; i < op->GetIOperands().size(); ++i) {
        auto iOperand = op->GetInputOperand(i);
        auto validShape = op->GetIOperands()[i]->GetDynValidShape();
        for (size_t dimIdx = 0; dimIdx < validShape.size(); ++dimIdx) {
            dimValidShape[dimIdx][i] = validShape[dimIdx];
        }
        auto shape = op->GetIOperands()[i]->GetShape();
        for (size_t dimIdx = 0; dimIdx < shape.size() && dimIdx < shapeDimNum; ++dimIdx) {
            if (dimIdx == shape.size() - 1 && iOperand->GetProducers().size() == 1 &&
                (*iOperand->GetProducers().begin())->GetOpcode() == Opcode::OP_BRCB) {
                dimShape[dimIdx][i] = 1;
            } else {
                dimShape[dimIdx][i] = shape[dimIdx];
            }
        }
    }
    std::vector<int64_t> inputIdx(shapeDimNum, 0);
    std::vector<SymbolicScalar> inputValidShape;
    for (size_t i = 0; i < shapeDimNum; ++i) {
        size_t oneDimNum = 0;
        size_t noOneIndex = 0;
        for (size_t j = 0; j < dimShape[i].size(); ++j) {
            if (dimShape[i][j] == 1) {
                oneDimNum++;
            } else {
                noOneIndex = j;
            }
        }
        if (oneDimNum > 0 && oneDimNum < dimShape[i].size()) {
            inputValidShape.push_back(dimValidShape[i][noOneIndex]);
            inputIdx[i] = noOneIndex;
            continue;
        }

        auto flag = false;
        auto minDim = SymbolicScalar();
        for (auto dim : dimValidShape[i]) {
            if (!(dim.IsImmediate())) {
                inputValidShape.push_back(dim);
                flag = true;
                break;
            } else {
                minDim = minDim.ConcreteValid() ? std::min(minDim.Concrete(), dim.Concrete()) : dim.Concrete();
            }
        }
        // 全部都是Immediate值，取用最小的
        if (!flag) {
            inputValidShape.push_back(minDim);
        }
    }

    int64_t mode = 0;
    if (op->GetAttr(OP_ATTR_PREFIX + "cmp_mode", mode) && mode == 1) {
        inputValidShape[inputValidShape.size() - 1] = inputValidShape[inputValidShape.size() - 1] /
                                                      NUM_VALUE_8; // 8 bit to 1 byte
    }

    int64_t whereBitMode = 0;
    if (op->GetAttr(OP_ATTR_PREFIX + "whereBitMode", whereBitMode) && whereBitMode == 1 && inputIdx.back() == 0) {
        inputValidShape[inputValidShape.size() - 1] = inputValidShape[inputValidShape.size() - 1] * NUM_VALUE_8;
    }

    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(inputValidShape);
    }
}

REGISTER_INFER_SHAPE_FUNC(OP_ADDS, Opcode::OP_ADDS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_MULS, Opcode::OP_MULS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_DIVS, Opcode::OP_S_DIVS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_POW, Opcode::OP_POW, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_POWS, Opcode::OP_POWS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_EXP, Opcode::OP_EXP, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_EXP2, Opcode::OP_EXP2, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_EXPM1, Opcode::OP_EXPM1, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_SIGN, Opcode::OP_SIGN, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_SIGNBIT, Opcode::OP_SIGNBIT, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_TANH, Opcode::OP_TANH, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_NEG, Opcode::OP_NEG, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ROUND, Opcode::OP_ROUND, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_RSQRT, Opcode::OP_RSQRT, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_RELU, Opcode::OP_RELU, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_LOG1P, Opcode::OP_LOG1P, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_SQRT, Opcode::OP_SQRT, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_CEIL, Opcode::OP_CEIL, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_FLOOR, Opcode::OP_FLOOR, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_TRUNC, Opcode::OP_TRUNC, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_BITWISENOT, Opcode::OP_BITWISENOT, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_GCD, Opcode::OP_GCD, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_GCDS, Opcode::OP_GCDS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ABS, Opcode::OP_ABS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_LN, Opcode::OP_LN, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ISFINITE, Opcode::OP_ISFINITE, ElewiseInferFunc);

void IsNanInferShapeFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    auto inputValidShape = op->GetIOperands()[0]->GetDynValidShape();
    if (inputValidShape.empty()) {
        return;
    }
    outValidShapes.emplace_back(inputValidShape);
    if (op->GetOOperands().size() > 1) {
        outValidShapes.emplace_back(std::vector<SymbolicScalar>{SymbolicScalar(op->GetOOperands()[1]->GetShape()[0])});
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_ISNAN, Opcode::OP_ISNAN, IsNanInferShapeFunc);

REGISTER_INFER_SHAPE_FUNC(OP_HUB, Opcode::OP_HUB, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ERF, Opcode::OP_ERF, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_DIGAMMA, Opcode::OP_DIGAMMA, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_SIN, Opcode::OP_SIN, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_COS, Opcode::OP_COS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ERFC, Opcode::OP_ERFC, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ATAN, Opcode::OP_ATAN, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ATAN2, Opcode::OP_ATAN2, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_DIVS, Opcode::OP_DIVS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_RECIPROCAL, Opcode::OP_RECIPROCAL, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_SUBS, Opcode::OP_SUBS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_MAXS, Opcode::OP_MAXS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_MINS, Opcode::OP_MINS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_BITWISEANDS, Opcode::OP_BITWISEANDS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_BITWISEORS, Opcode::OP_BITWISEORS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_BITWISEXORS, Opcode::OP_BITWISEXORS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_ADDS, Opcode::OP_S_ADDS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_SUBS, Opcode::OP_S_SUBS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_MULS, Opcode::OP_S_MULS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_MAXS, Opcode::OP_S_MAXS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_MINS, Opcode::OP_S_MINS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_LRELU, Opcode::OP_LRELU, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_ADD, Opcode::OP_S_ADD, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_SUB, Opcode::OP_S_SUB, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_MUL, Opcode::OP_S_MUL, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_DIV, Opcode::OP_S_DIV, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_MAX, Opcode::OP_S_MAX, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_S_MIN, Opcode::OP_S_MIN, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_CUM_SUM, Opcode::OP_CUM_SUM, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_CUM_PROD, Opcode::OP_CUM_PROD, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_TRIUL, Opcode::OP_TRIUL, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_REGISTER_COPY, Opcode::OP_REGISTER_COPY, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_CMP, Opcode::OP_CMP, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_CMPS, Opcode::OP_CMPS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_HYPOT, Opcode::OP_HYPOT, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_MOD, Opcode::OP_MOD, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_MODS, Opcode::OP_MODS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_BITWISERIGHTSHIFT, Opcode::OP_BITWISERIGHTSHIFT, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_BITWISELEFTSHIFT, Opcode::OP_BITWISELEFTSHIFT, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_BITWISERIGHTSHIFTS, Opcode::OP_BITWISERIGHTSHIFTS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_BITWISELEFTSHIFTS, Opcode::OP_BITWISELEFTSHIFTS, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_SBITWISERIGHTSHIFT, Opcode::OP_SBITWISERIGHTSHIFT, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_SBITWISELEFTSHIFT, Opcode::OP_SBITWISELEFTSHIFT, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_BITWISEAND, Opcode::OP_BITWISEAND, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_BITWISEOR, Opcode::OP_BITWISEOR, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_BITWISEXOR, Opcode::OP_BITWISEXOR, ElewiseInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_COPYSIGN, Opcode::OP_COPYSIGN, ElewiseInferFunc);

void RemInferShapeFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    ElewiseInferFunc(op, outValidShapes);
    outValidShapes.erase(outValidShapes.begin() + 1, outValidShapes.end());
    std::vector<SymbolicScalar> tmpValidShape;
    tmpValidShape.push_back(SymbolicScalar(op->GetOOperands()[1]->GetShape()[0]));
    tmpValidShape.push_back(SymbolicScalar(op->GetOOperands()[1]->GetShape()[1]));
    outValidShapes.push_back(tmpValidShape);
}
REGISTER_INFER_SHAPE_FUNC(OP_REM, Opcode::OP_REM, RemInferShapeFunc);
REGISTER_INFER_SHAPE_FUNC(OP_REMS, Opcode::OP_REMS, RemInferShapeFunc);
REGISTER_INFER_SHAPE_FUNC(OP_REMRS, Opcode::OP_REMRS, RemInferShapeFunc);

void PackInferShapeFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    auto input = op->GetIOperands()[0];
    outValidShapes.push_back({input->GetDynValidShape()[0] * BytesOf(input->Datatype())});
}
REGISTER_INFER_SHAPE_FUNC(OP_PACK, Opcode::OP_PACK, PackInferShapeFunc);

void UnPackInferShapeFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    auto input = op->GetIOperands()[0];
    auto output = op->GetOOperands()[0];
    outValidShapes.push_back({input->GetDynValidShape()[0] / BytesOf(output->Datatype())});
}
REGISTER_INFER_SHAPE_FUNC(OP_UNPACK, Opcode::OP_UNPACK, UnPackInferShapeFunc);

void FloorDivInferShapeFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    auto inputValidShape = op->GetIOperands()[0]->GetDynValidShape();
    if (inputValidShape.empty()) {
        return;
    }
    outValidShapes.emplace_back(inputValidShape);
    inputValidShape.front() = inputValidShape.front() * NUM_VALUE_6;
    outValidShapes.emplace_back(inputValidShape);
}
REGISTER_INFER_SHAPE_FUNC(OP_FLOORDIV, Opcode::OP_FLOORDIV, FloorDivInferShapeFunc);
REGISTER_INFER_SHAPE_FUNC(OP_FLOORDIVS, Opcode::OP_FLOORDIVS, FloorDivInferShapeFunc);

template <int64_t TailScale>
void InferShapeWithTailScaleFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    auto inputValidShape = op->GetIOperands()[0]->GetDynValidShape();
    if (inputValidShape.empty()) {
        return;
    }
    size_t ndim = inputValidShape.size();
    outValidShapes.emplace_back(inputValidShape);
    outValidShapes.emplace_back(std::vector<SymbolicScalar>{inputValidShape[ndim - 1] * TailScale});
}
REGISTER_INFER_SHAPE_FUNC(OP_SINH, Opcode::OP_SINH, InferShapeWithTailScaleFunc<NUM_VALUE_4>);
REGISTER_INFER_SHAPE_FUNC(OP_COSH, Opcode::OP_COSH, InferShapeWithTailScaleFunc<1>);
REGISTER_INFER_SHAPE_FUNC(OP_TAN, Opcode::OP_TAN, InferShapeWithTailScaleFunc<NUM_VALUE_7>);
REGISTER_INFER_SHAPE_FUNC(OP_ASINH, Opcode::OP_ASINH, InferShapeWithTailScaleFunc<NUM_VALUE_4>);
REGISTER_INFER_SHAPE_FUNC(OP_ACOSH, Opcode::OP_ACOSH, InferShapeWithTailScaleFunc<NUM_VALUE_3>);

void AsinAcosInferShapeFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    auto inputValidShape = op->GetIOperands()[0]->GetDynValidShape();
    if (inputValidShape.empty()) {
        return;
    }
    size_t ndim = inputValidShape.size();
    outValidShapes.emplace_back(inputValidShape);
    // tmp: 5 个 H x W 中间块，1D 输入退化为 H=1
    if (ndim >= NUM_VALUE_2) {
        outValidShapes.emplace_back(
            std::vector<SymbolicScalar>{inputValidShape[ndim - NUM_VALUE_2] * inputValidShape[ndim - 1] * NUM_VALUE_5});
    } else {
        outValidShapes.emplace_back(std::vector<SymbolicScalar>{inputValidShape[ndim - 1] * NUM_VALUE_5});
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_ASIN, Opcode::OP_ASIN, AsinAcosInferShapeFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ACOS, Opcode::OP_ACOS, AsinAcosInferShapeFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ATANH, Opcode::OP_ATANH, InferShapeWithTailScaleFunc<NUM_VALUE_5>);

void PadInferShapeFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    auto inputValidShape = op->GetIOperands()[0]->GetDynValidShape();
    if (inputValidShape.empty()) {
        return;
    }
    size_t ndim = inputValidShape.size();
    std::vector<SymbolicScalar> outValidShape = inputValidShape;

    int64_t padRight = 0;
    int64_t padBottom = 0;
    op->GetAttr(OP_ATTR_PREFIX + "pad_right", padRight);
    op->GetAttr(OP_ATTR_PREFIX + "pad_bottom", padBottom);
    if (ndim >= 1 && padRight > 0) {
        outValidShape[ndim - 1] = outValidShape[ndim - 1] + padRight;
    }
    if (ndim >= NUM_VALUE_2 && padBottom > 0) {
        outValidShape[ndim - NUM_VALUE_2] = outValidShape[ndim - NUM_VALUE_2] + padBottom;
    }
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_PAD, Opcode::OP_PAD, PadInferShapeFunc);

void InterleaveInferShapeFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    std::vector<SymbolicScalar> outValidShape = op->GetIOperands()[0]->GetDynValidShape();
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_INTERLEAVE, Opcode::OP_INTERLEAVE, InterleaveInferShapeFunc);

void DeInterleaveInferShapeFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    std::vector<SymbolicScalar> outValidShape = op->GetIOperands()[0]->GetDynValidShape();
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_DEINTERLEAVE, Opcode::OP_DEINTERLEAVE, DeInterleaveInferShapeFunc);

void DeInterleaveSingleInferShapeFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    std::vector<SymbolicScalar> outValidShape = op->GetIOperands()[0]->GetDynValidShape();
    if (!outValidShape.empty()) {
        outValidShape[outValidShape.size() - 1] = outValidShape[outValidShape.size() - 1] / NUM_VALUE_2;
    }
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_DEINTERLEAVE_SINGLE, Opcode::OP_DEINTERLEAVE_SINGLE, DeInterleaveSingleInferShapeFunc);

void FillPadInferShapeFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    std::vector<SymbolicScalar> outValidShape;
    for (auto c : op->GetOOperands()[0]->oriShape) {
        outValidShape.push_back(SymbolicScalar(c));
    }
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_FILLPAD, Opcode::OP_FILLPAD, FillPadInferShapeFunc);

void IndexOutCastInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    std::vector<SymbolicScalar> outValidShape;
    /* 这里取IOperands索引是依据AddOperation中ioprand中的输入的顺序，这里使用的是dst参数的ioprand，即第2个索引 */
    size_t input_dim = NUM_VALUE_2;
    auto inValidShape = op->GetIOperands()[input_dim]->GetDynValidShape();

    for (size_t i = 0; i < inValidShape.size(); ++i) {
        outValidShape.push_back(inValidShape[i]);
    }

    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }

    auto indexOutCastOpAttribute = std::dynamic_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
    if (indexOutCastOpAttribute == nullptr) {
        VECTOR_LOGW("IndexOutCast [%d] has no copyOpAttr.", op->GetOpMagic());
        return;
    }
    indexOutCastOpAttribute->SetFromDynValidShape(
        OpImmediate::Specified(op->GetIOperands()[input_dim]->GetDynValidShape()));
}
REGISTER_INFER_SHAPE_FUNC(OP_INDEX_OUTCAST, Opcode::OP_INDEX_OUTCAST, IndexOutCastInferFunc);

void CastInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    auto srcValidShape = op->GetIOperands()[0]->GetDynValidShape();
    outValidShapes.push_back(srcValidShape);
    if (op->GetOOperands().size() > 1) {
        std::vector<SymbolicScalar> tmpValidShape;
        for (size_t i = 0; i < op->GetOOperands()[1]->GetShape().size(); ++i) {
            tmpValidShape.push_back(SymbolicScalar(op->GetOOperands()[1]->GetShape()[i]));
        }
        outValidShapes.push_back(tmpValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_CAST, Opcode::OP_CAST, CastInferFunc);

void GatherElementInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    std::vector<SymbolicScalar> outValidShape;
    /* 这里取IOperands索引是依据AddOperation中ioprand中的输入的顺序，这里使用的是src参数的ioprand，即第1个索引 */
    auto inValidShape = op->GetIOperands()[1]->GetDynValidShape();

    for (size_t i = 0; i < inValidShape.size(); ++i) {
        outValidShape.push_back(inValidShape[i]);
    }

    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_GATHER_ELEMENT, Opcode::OP_GATHER_ELEMENT, GatherElementInferFunc);

void GatherMaskFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    std::vector<std::vector<SymbolicScalar>> inputValidShapes;
    for (auto inputTensor : op->GetIOperands()) {
        inputValidShapes.push_back(inputTensor->GetDynValidShape());
    }
    if (inputValidShapes.empty()) {
        return;
    }
    std::vector<SymbolicScalar> res(inputValidShapes[0]);
    uint8_t patternMode = op->GetIntAttribute(OP_ATTR_PREFIX + "patternMode");
    const auto pattern = static_cast<GatherMaskPattern>(patternMode);
    if (IsHalfGatherMaskPattern(pattern)) {
        res.back() = res.back() / NUM_VALUE_2;
    } else if (IsQuarterGatherMaskPattern(pattern)) {
        res.back() = res.back() / NUM_VALUE_4;
    }
    outValidShapes.push_back(res);
}
REGISTER_INFER_SHAPE_FUNC(OP_GATHER_MASK, Opcode::OP_GATHER_MASK, GatherMaskFunc);

void ScatterInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    if (op->GetOpcode() != Opcode::OP_SCATTER_INPLACE && op->GetOpcode() != Opcode::OP_SCATTER_ELEMENT_INPLACE) {
        std::vector<SymbolicScalar> outValidShape;
        auto inValidShape = op->GetIOperands()[0]->GetDynValidShape();
        for (size_t i = 0; i < inValidShape.size(); ++i) {
            outValidShape.push_back(inValidShape[i]);
        }
        for (auto output : op->GetOOperands()) {
            outValidShapes.push_back(outValidShape);
        }
        return;
    }
    outValidShapes.push_back(op->GetIOperands()[0]->GetDynValidShape());
    for (size_t i = 1; i < op->GetOOperands().size(); ++i) {
        std::vector<SymbolicScalar> tmpValidShape;
        for (auto dim : op->GetOOperands()[i]->GetShape()) {
            tmpValidShape.emplace_back(dim);
        }
        outValidShapes.push_back(tmpValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_SCATTER_ELEMENT, Opcode::OP_SCATTER_ELEMENT, ScatterInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_SCATTER_ELEMENT_INPLACE, Opcode::OP_SCATTER_ELEMENT_INPLACE, ScatterInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_SCATTER, Opcode::OP_SCATTER, ScatterInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_SCATTER_INPLACE, Opcode::OP_SCATTER_INPLACE, ScatterInferFunc);

void IndexAddInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    auto selfValidShape = op->GetIOperands()[0]->GetDynValidShape();
    auto srcValidShape = op->GetIOperands()[1]->GetDynValidShape();
    outValidShapes.push_back(selfValidShape);
    outValidShapes.push_back({1, srcValidShape[srcValidShape.size() - 1]});
}
REGISTER_INFER_SHAPE_FUNC(OP_INDEX_ADD, Opcode::OP_INDEX_ADD, IndexAddInferFunc);

void IndexAddUBInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    auto inValidShape = op->GetIOperands()[0]->GetDynValidShape();
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(inValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_INDEX_ADD_UB, Opcode::OP_INDEX_ADD_UB, IndexAddUBInferFunc);

void LogicalNotInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    ElewiseInferFunc(op, outValidShapes);
    outValidShapes.erase(outValidShapes.begin() + 1, outValidShapes.end());
    auto data_type = op->GetIOperands()[0]->Datatype();

    DataType select_dtype;
    if (data_type == DT_FP32 || data_type == DT_BF16) {
        select_dtype = DT_FP32;
    } else if (data_type == DT_INT16 || data_type == DT_UINT16) {
        select_dtype = DT_INT16;
    } else if (data_type == DT_INT32 || data_type == DT_UINT32) {
        select_dtype = DT_INT32;
    } else if (data_type == DT_INT64 || data_type == DT_UINT64) {
        select_dtype = DT_INT64;
    } else {
        select_dtype = DT_FP16;
    }
    constexpr int64_t COUNT_SIZE = 2048;
    constexpr int64_t vcmp_bit_size = COUNT_SIZE / NUM_VALUE_8;
    constexpr size_t ALIGN_SIZE = NUM_VALUE_32;

    int64_t total_size;
    if (data_type == DT_INT16 || data_type == DT_UINT16 || data_type == DT_INT32 || data_type == DT_UINT32) {
        total_size = COUNT_SIZE * BytesOf(select_dtype);
    } else {
        total_size = COUNT_SIZE * NUM_VALUE_2 + COUNT_SIZE * BytesOf(select_dtype) * NUM_VALUE_2 + vcmp_bit_size +
                     NUM_VALUE_8;
    }
    total_size = (total_size + ALIGN_SIZE - 1) / ALIGN_SIZE * ALIGN_SIZE;
    int64_t shape = total_size / BytesOf(select_dtype);
    outValidShapes.push_back({shape});
}
REGISTER_INFER_SHAPE_FUNC(OP_LOGICALNOT, Opcode::OP_LOGICALNOT, LogicalNotInferFunc);

void LogicalAndInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    ElewiseInferFunc(op, outValidShapes);
    outValidShapes.erase(outValidShapes.begin() + 1, outValidShapes.end());
    if (op->GetOOperands().size() > 1) {
        outValidShapes.push_back({SymbolicScalar(op->GetOOperands()[1]->GetShape()[0])});
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_LOGICALAND, Opcode::OP_LOGICALAND, LogicalAndInferFunc);

void ViewTypeInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    auto inputOperand = op->GetIOperands()[0];
    auto outputOperand = op->GetOOperands()[0];

    auto validShape = inputOperand->GetDynValidShape();
    auto changedDim = validShape[validShape.size() - 1] * BytesOf(inputOperand->Datatype()) /
                      BytesOf(outputOperand->Datatype());
    validShape[validShape.size() - 1] = changedDim;

    outValidShapes.push_back(validShape);
}
REGISTER_INFER_SHAPE_FUNC(OP_VIEW_TYPE, Opcode::OP_VIEW_TYPE, ViewTypeInferFunc);

void IndexPutInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    outValidShapes.push_back(op->GetIOperands()[0]->GetDynValidShape());
    if (op->GetOOperands().size() > 1) {
        outValidShapes.push_back(op->GetIOperands()[2]->GetDynValidShape());
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_INDEX_PUT, Opcode::OP_INDEX_PUT, IndexPutInferFunc);

void PairReduceInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    auto dimSize = op->GetIOperands()[0]->GetDynValidShape().size();
    std::vector<SymbolicScalar> outValidShape;
    for (size_t i = 0; i < dimSize; i++) {
        outValidShape.push_back(op->GetIOperands()[0]->GetDynValidShape()[i]);
    }
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_PAIRMAX, Opcode::OP_PAIRMAX, PairReduceInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_PAIRMIN, Opcode::OP_PAIRMIN, PairReduceInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_PAIRSUM, Opcode::OP_PAIRSUM, PairReduceInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_PAIRPROD, Opcode::OP_PAIRPROD, PairReduceInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_PAIRARGMAX, Opcode::OP_PAIRARGMAX, PairReduceInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_PAIRARGMIN, Opcode::OP_PAIRARGMIN, PairReduceInferFunc);

// elewise brc infer shape func
void ElewiseBrcInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    std::vector<SymbolicScalar> outValidShape;

    // elewisebrc dim is immediate and dim is 1, another dim is dst shape dim
    for (size_t i = 0; i < op->GetIOperands()[0]->GetDynValidShape().size(); ++i) {
        auto leftIShapeDim = op->GetIOperands()[0]->GetDynValidShape()[i];
        if (leftIShapeDim.IsImmediate() && leftIShapeDim.Concrete() == 1) {
            outValidShape.push_back(op->GetIOperands()[1]->GetDynValidShape()[i]);
        } else {
            outValidShape.push_back(leftIShapeDim);
        }
    }
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_ADD_BRC, Opcode::OP_ADD_BRC, ElewiseBrcInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_SUB_BRC, Opcode::OP_SUB_BRC, ElewiseBrcInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_MUL_BRC, Opcode::OP_MUL_BRC, ElewiseBrcInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_DIV_BRC, Opcode::OP_DIV_BRC, ElewiseBrcInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_MAX_BRC, Opcode::OP_MAX_BRC, ElewiseBrcInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_GCD_BRC, Opcode::OP_GCD_BRC, ElewiseBrcInferFunc);

// broadcast infer shape func
void BroadcastInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    std::vector<SymbolicScalar> outValidShape;
    if (op->GetAttr(OP_ATTR_PREFIX + "validShape", outValidShape)) {
        for (auto output : op->GetOOperands()) {
            outValidShapes.push_back(outValidShape);
        }
        return;
    }
    auto outTensor = op->GetOOperands()[0]; // one in, one out
    // broadcast 1对应的维度采用tileshap
    for (size_t i = 0; i < op->GetIOperands()[0]->GetDynValidShape().size(); ++i) {
        if (op->GetIOperands()[0]->oriShape[i] != 1) {
            outValidShape.push_back(op->GetIOperands()[0]->GetDynValidShape()[i]);
        } else {
            outValidShape.push_back(SymbolicScalar(op->GetOOperands()[0]->GetShape()[i]));
        }
    }
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_EXPAND, Opcode::OP_EXPAND, BroadcastInferFunc);

// one hot infer shape func
void OneHotInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    std::vector<SymbolicScalar> outValidShape(op->GetIOperands()[0]->GetDynValidShape());
    int lastDim = op->GetIntAttribute(OP_ATTR_PREFIX + "numClasses");
    outValidShape.push_back(SymbolicScalar(lastDim));
    outValidShapes.push_back(outValidShape);
}
REGISTER_INFER_SHAPE_FUNC(OP_ONEHOT, Opcode::OP_ONEHOT, OneHotInferFunc);

// Quantize infer shape func
void QuantizeInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    // First output (dst): use input's valid shape
    std::vector<SymbolicScalar> dstValidShape(op->GetIOperands()[0]->GetDynValidShape());
    outValidShapes.push_back(dstValidShape);

    // Second output (tmpbuf): same size as src, with int32_t type (4 bytes per element)
    auto inputValidShape = op->GetIOperands()[0]->GetDynValidShape();
    if (!inputValidShape.empty()) {
        // tmpbuf has same shape as input src
        std::vector<SymbolicScalar> tmpValidShape = inputValidShape;

        // For axis=-2, align the second-to-last dimension to 32 bytes (8 int32 elements)
        int64_t axis = 0;
        if (op->GetAttr(OP_ATTR_PREFIX + "axis", axis) && axis == -NUM_VALUE_2 && tmpValidShape.size() >= NUM_VALUE_2) {
            constexpr int64_t alignElements = NUM_VALUE_8; // 8 * 4 = 32 bytes
            size_t lastDimIndex = tmpValidShape.size() - NUM_VALUE_2;
            tmpValidShape[lastDimIndex] = (tmpValidShape[lastDimIndex] + alignElements - 1) / alignElements *
                                          alignElements;
        }

        outValidShapes.push_back(tmpValidShape);
    } else {
        // No dynamic shape info, use input's static shape
        std::vector<SymbolicScalar> tmpValidShape;
        auto inputShape = op->GetIOperands()[0]->GetShape();
        for (auto dim : inputShape) {
            tmpValidShape.push_back(SymbolicScalar(dim));
        }

        // For axis=-2, align the second-to-last dimension to 32 bytes (8 int32 elements)
        int64_t axis = 0;
        if (op->GetAttr(OP_ATTR_PREFIX + "axis", axis) && axis == -NUM_VALUE_2 && tmpValidShape.size() >= NUM_VALUE_2) {
            constexpr int64_t alignElements = NUM_VALUE_8; // 8 * 4 = 32 bytes
            size_t lastDimIndex = tmpValidShape.size() - NUM_VALUE_2;
            tmpValidShape[lastDimIndex] = (tmpValidShape[lastDimIndex] + alignElements - 1) / alignElements *
                                          alignElements;
        }

        outValidShapes.push_back(tmpValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_QUANTIZE_SYM, Opcode::OP_QUANTIZE_SYM, QuantizeInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_QUANTIZE_ASYM, Opcode::OP_QUANTIZE_ASYM, QuantizeInferFunc);

void DequantizeInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    std::vector<SymbolicScalar> outValidShape(op->GetIOperands()[0]->GetDynValidShape());
    outValidShapes.push_back(outValidShape);
}
REGISTER_INFER_SHAPE_FUNC(OP_DEQUANTIZE, Opcode::OP_DEQUANTIZE, DequantizeInferFunc);

// Range infer shape func
void RangeInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    std::vector<SymbolicScalar> outValidShape;
    Element size = op->GetElementAttribute(OP_ATTR_PREFIX + "SIZE");
    outValidShape.push_back(SymbolicScalar(size.GetSignedData()));
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_RANGE, Opcode::OP_RANGE, RangeInferFunc);

void UniformInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    std::vector<SymbolicScalar> outValidShape;
    auto shapeAttr = op->GetVectorIntAttribute(OP_ATTR_PREFIX + "SHAPE");
    for (auto dim : shapeAttr) {
        outValidShape.push_back(SymbolicScalar(static_cast<int64_t>(dim)));
    }
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_UNIFORM, Opcode::OP_UNIFORM, UniformInferFunc);

// reduce infer shape func
void ReduceInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    std::vector<std::vector<SymbolicScalar>> inputValidShapes;
    for (auto inputTensor : op->GetIOperands()) {
        inputValidShapes.push_back(inputTensor->GetDynValidShape());
    }
    if (inputValidShapes.empty()) {
        return;
    }
    auto outValidShape = inputValidShapes[0];
    int axis = op->GetIntAttribute(OP_ATTR_PREFIX + "AXIS");
    outValidShape[axis] = SymbolicScalar(1);
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_ROWSUMLINE, Opcode::OP_ROWSUMLINE, ReduceInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ROWMAXLINE, Opcode::OP_ROWMAXLINE, ReduceInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ROWMINLINE, Opcode::OP_ROWMINLINE, ReduceInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ROWPRODLINE, Opcode::OP_ROWPRODLINE, ReduceInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ROWMAX_SINGLE, Opcode::OP_ROWMAX_SINGLE, ReduceInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ROWMIN_SINGLE, Opcode::OP_ROWMIN_SINGLE, ReduceInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ROWSUM_SINGLE, Opcode::OP_ROWSUM_SINGLE, ReduceInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ROWPROD_SINGLE, Opcode::OP_ROWPROD_SINGLE, ReduceInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ROWARGMAXWITHVALUE_SINGLE, Opcode::OP_ROWARGMAXWITHVALUE_SINGLE, ReduceInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ROWARGMINWITHVALUE_SINGLE, Opcode::OP_ROWARGMINWITHVALUE_SINGLE, ReduceInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ROWARGMAXWITHVALUE_LINE, Opcode::OP_ROWARGMAXWITHVALUE_LINE, ReduceInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ROWARGMINWITHVALUE_LINE, Opcode::OP_ROWARGMINWITHVALUE_LINE, ReduceInferFunc);

void OnlineSoftmaxInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    auto scoresValidShape = op->GetIOperands()[0]->GetDynValidShape();
    auto columnStatisticValidShape = scoresValidShape;
    if (!columnStatisticValidShape.empty()) {
        columnStatisticValidShape[0] = SymbolicScalar(1);
    }
    outValidShapes.push_back(scoresValidShape);
    outValidShapes.push_back(columnStatisticValidShape);
    outValidShapes.push_back(columnStatisticValidShape);
    outValidShapes.push_back(scoresValidShape);
    auto reduceWorkspaceValidShape = op->GetOOperands()[NUM_VALUE_4]->GetDynValidShape();
    if (reduceWorkspaceValidShape.empty()) {
        reduceWorkspaceValidShape = SymbolicScalar::FromConcrete(op->GetOOperands()[NUM_VALUE_4]->GetShape());
    }
    outValidShapes.push_back(reduceWorkspaceValidShape);
}

void OnlineSoftmaxUpdateInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    auto statisticValidShape = op->GetIOperands()[0]->GetDynValidShape();
    auto outputValidShape = op->GetIOperands()[NUM_VALUE_2]->GetDynValidShape();
    outValidShapes.push_back(statisticValidShape);
    outValidShapes.push_back(statisticValidShape);
    outValidShapes.push_back(outputValidShape);
    auto updateWorkspaceValidShape = op->GetOOperands()[NUM_VALUE_3]->GetDynValidShape();
    if (updateWorkspaceValidShape.empty()) {
        updateWorkspaceValidShape = SymbolicScalar::FromConcrete(op->GetOOperands()[NUM_VALUE_3]->GetShape());
    }
    outValidShapes.push_back(updateWorkspaceValidShape);
}

REGISTER_INFER_SHAPE_FUNC(OP_ONLINE_SOFTMAX, Opcode::OP_ONLINE_SOFTMAX, OnlineSoftmaxInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_ONLINE_SOFTMAX_UPDATE, Opcode::OP_ONLINE_SOFTMAX_UPDATE, OnlineSoftmaxUpdateInferFunc);

void WhereInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    ElewiseInferFunc(op, outValidShapes);
}
REGISTER_INFER_SHAPE_FUNC(OP_WHERE_TT, Opcode::OP_WHERE_TT, WhereInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_WHERE_TS, Opcode::OP_WHERE_TS, WhereInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_WHERE_ST, Opcode::OP_WHERE_ST, WhereInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_WHERE_SS, Opcode::OP_WHERE_SS, WhereInferFunc);

// Gather infer shape func
void InferFunc4Gather(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    auto iOperands = op->GetIOperands();
    ASSERT(VectorErrorCode::ERR_PARAM_COUNT_INVALID, iOperands.size() >= NUM2) << "iOperands.size() should be >= NUM2";
    int axis = op->GetIntAttribute(OP_ATTR_PREFIX + "axis");
    int src0Rank = iOperands[0]->GetShape().size();
    if (axis < 0) {
        axis = axis + src0Rank;
    }
    ASSERT(VectorErrorCode::ERR_PARAM_INVALID, axis >= 0 && axis < src0Rank) << "InferFunc4Gather, axis is invalid";

    std::vector<std::vector<SymbolicScalar>> inputValidShapes;
    for (auto inputTensor : iOperands) {
        inputValidShapes.push_back(inputTensor->GetDynValidShape());
    }
    // output shape: input0.shape[:aixs] + input1.shape + input0.shape[axis+1:]
    std::vector<SymbolicScalar> outValidShape = inputValidShapes[0];
    outValidShape.erase(outValidShape.begin() + axis);
    outValidShape.insert(outValidShape.begin() + axis, inputValidShapes[1].begin(), inputValidShapes[1].end());

    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_GATHER_FROM_UB, Opcode::OP_GATHER_FROM_UB, InferFunc4Gather);
REGISTER_INFER_SHAPE_FUNC(OP_GATHER, Opcode::OP_GATHER, InferFunc4Gather);

void InferFuncGatherInL1(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    auto iOperands = op->GetIOperands();
    ASSERT(MatmulErrorCode::ERR_PARAM_MISMATCH, iOperands.size() == NUM_VALUE_3) << "iOperands.size() should be 3";
    auto srcValidShape = iOperands[0]->GetDynValidShape();
    auto offsetValidShape = iOperands[1]->GetDynValidShape();
    auto srcStartColumnOffset = op->GetIntAttribute(OpAttributeKey::startOffset);
    ASSERT(MatmulErrorCode::ERR_PARAM_MISMATCH, op->GetOOperands().size() == 1)
        << "op->GetOOperands().size() should be 1";
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(
            {offsetValidShape[1], std::min(srcValidShape[1] - srcStartColumnOffset, output->GetShape()[1])});
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_GATHER_IN_L1, Opcode::OP_GATHER_IN_L1, InferFuncGatherInL1);
/**
 * 定制，
 * parma [a,b]
 * indices [1,c]
 * axis=-2
 * result [c,b]
 */
void InferFuncGatherInUB(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    auto iOperands = op->GetIOperands();
    ASSERT(VectorErrorCode::ERR_PARAM_COUNT_INVALID, iOperands.size() == NUM_VALUE_3) << "iOperands.size() should be 3";
    auto srcValidShape = iOperands[0]->GetDynValidShape();
    auto indicesValidShape = iOperands[1]->GetDynValidShape();
    ASSERT(VectorErrorCode::ERR_PARAM_COUNT_INVALID, op->GetOOperands().size() == 1)
        << "op->GetOOperands().size() should be 1";
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back({indicesValidShape[1], srcValidShape[1]});
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_GATHER_IN_UB, Opcode::OP_GATHER_IN_UB, InferFuncGatherInUB);

// matmul infer shape func
void MatmulInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    std::vector<SymbolicScalar> outValidShape;
    for (auto inputTensor : op->GetIOperands()) {
        auto inputValidShape = inputTensor->GetDynValidShape();
        if (inputTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0A) {
            outValidShape.push_back(inputValidShape[0]);
        } else if (inputTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0B) {
            outValidShape.push_back(inputValidShape[1]);
        }
    }

    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_A_MUL_B, Opcode::OP_A_MUL_B, MatmulInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_A_MUL_BT, Opcode::OP_A_MUL_BT, MatmulInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_AT_MUL_B, Opcode::OP_AT_MUL_B, MatmulInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_AT_MUL_BT, Opcode::OP_AT_MUL_BT, MatmulInferFunc);
// kirin9050 宏指令 matmul：输入/输出/partial sum 均驻留 L1
REGISTER_INFER_SHAPE_FUNC(OP_A_MUL_B_EPILOGUE, Opcode::OP_A_MUL_B_EPILOGUE, MatmulInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_A_MUL_BT_EPILOGUE, Opcode::OP_A_MUL_BT_EPILOGUE, MatmulInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_AT_MUL_B_EPILOGUE, Opcode::OP_AT_MUL_B_EPILOGUE, MatmulInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_AT_MUL_BT_EPILOGUE, Opcode::OP_AT_MUL_BT_EPILOGUE, MatmulInferFunc);

void LoadBTFBInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    for (auto output : op->GetOOperands()) {
        ASSERT(VectorErrorCode::ERR_PARAM_INVALID, !output->GetDynValidShape().empty())
            << "output->GetDynValidShape() should not be empty";
        outValidShapes.push_back(output->GetDynValidShape());
    }
}

REGISTER_INFER_SHAPE_FUNC(OP_L1_TO_BT, Opcode::OP_L1_TO_BT, LoadBTFBInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_L1_TO_FIX_QUANT_PRE, Opcode::OP_L1_TO_FIX_QUANT_PRE, LoadBTFBInferFunc);

void MatmulACCInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    std::vector<SymbolicScalar> outValidShape;
    for (auto inputTensor : op->GetIOperands()) {
        if (inputTensor->GetMemoryTypeOriginal() == MemoryType::MEM_L0C) {
            outValidShape = inputTensor->GetDynValidShape();
            break;
        }
    }

    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_A_MULACC_B, Opcode::OP_A_MULACC_B, MatmulACCInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_A_MULACC_BT, Opcode::OP_A_MULACC_BT, MatmulACCInferFunc);

void MatmulMacroACCInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    // kirin9050 宏指令 ACC：输入排布为 [a(L1), b(L0B), partialSum(L1)]（见 LinkAMulB）。
    // partial sum 与输出同 shape(M,N)；而 a 的 shape 是 (M,K)——必须取 partial sum，
    // 若按“第一个 L1 输入”会取到 a，导致输出 valid shape 的 N 维被写成 K。
    std::vector<SymbolicScalar> outValidShape;
    const auto& inputs = op->GetIOperands();
    for (size_t i = 1; i < inputs.size(); ++i) { // i=0 是 a，跳过
        auto& t = inputs[i];
        if (t != nullptr && t->GetMemoryTypeOriginal() == MemoryType::MEM_L1 &&
            t->GetDynValidShape().size() == SHAPE_DIM2) {
            outValidShape = t->GetDynValidShape();
            break;
        }
    }
    if (outValidShape.empty() && !inputs.empty() && inputs[0] != nullptr) {
        outValidShape = inputs[0]->GetDynValidShape(); // 兜底：至少保证非空
    }

    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_A_MULACC_B_EPILOGUE, Opcode::OP_A_MULACC_B_EPILOGUE, MatmulMacroACCInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_A_MULACC_BT_EPILOGUE, Opcode::OP_A_MULACC_BT_EPILOGUE, MatmulMacroACCInferFunc);

void LoadL0C2L1InferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
    if (copyAttr != nullptr) {
        auto fromValidShape = op->GetIOperands()[0]->GetDynValidShape();
        copyAttr->SetFromDynValidShape(OpImmediate::Specified(fromValidShape));
    } else {
        VECTOR_LOGW("%s[%d] has no copy out attr, set output valid shape same as input.", op->GetOpcodeStr().c_str(),
                    op->GetOpMagic());
        outValidShapes.emplace_back(op->GetIOperands()[0]->GetDynValidShape());
        return;
    }
    auto offsets = copyAttr->GetToOffset();
    auto inputShapes = copyAttr->GetToDynValidShape();
    std::vector<SymbolicScalar> outDynShape = op->GetOOperands()[0]->GetDynValidShape();
    if (outDynShape.empty()) {
        outDynShape.resize(op->GetOOperands()[0]->GetShape().size(), SymbolicScalar(0));
    }
    std::vector<SymbolicScalar> outShape;
    for (size_t i = 0; i < inputShapes.size(); i++) {
        auto inputShape = inputShapes[i].GetSpecifiedValue();
        auto offset = offsets[i].GetSpecifiedValue();
        SymbolicScalar actualDim = std::max(outDynShape[i], (inputShape + offset) * (inputShape != 0));
        outShape.emplace_back(actualDim);
    }
    for (auto output : op->GetOOperands()) {
        outValidShapes.emplace_back(outShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_L0C_TO_L1, Opcode::OP_L0C_TO_L1, LoadL0C2L1InferFunc);

void Load2L1InferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    std::vector<std::vector<SymbolicScalar>> inputValidShapes;
    for (auto inputTensor : op->GetIOperands()) {
        inputValidShapes.push_back(inputTensor->GetDynValidShape());
    }
    if (inputValidShapes.empty()) {
        return;
    }

    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(inputValidShapes[0]);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_UB_COPY_ND2NZ, Opcode::OP_UB_COPY_ND2NZ, Load2L1InferFunc);

void Load2L1MXScaleInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, op != nullptr) << "op should not be nullptr";
    ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, !op->GetIOperands().empty() && op->GetIOperands()[0] != nullptr &&
                                                   op->GetIOperands()[0]->GetDynValidShape().size() == SHAPE_DIM3)
        << "op->GetIOperands() should not be empty and GetDynValidShape().size() should be SHAPE_DIM3";
    auto copyOpAttribute = std::dynamic_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
    if (op->GetOOperands()[0] != nullptr && !(op->GetOOperands()[0]->GetDynValidShape().empty())) {
        outValidShapes.push_back(op->GetOOperands()[0]->GetDynValidShape());
        if (copyOpAttribute != nullptr && (copyOpAttribute->GetToDynValidShape()).empty()) {
            auto toDynShape = OpImmediate::Specified(op->GetOOperands()[0]->GetDynValidShape());
            copyOpAttribute->SetToDynValidShape(toDynShape);
        }
        return;
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_L1_COPY_IN_B_SCALE, Opcode::OP_L1_COPY_IN_B_SCALE, Load2L1MXScaleInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_L1_COPY_IN_A_SCALE, Opcode::OP_L1_COPY_IN_A_SCALE, Load2L1MXScaleInferFunc);

// MTE infer shape func
template <bool isTrans = false>
void LoadL0InferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, op != nullptr) << "op should not be nullptr";
    if (op->HasAttr(Matrix::L1_TO_L0_OFFSET) && op->HasAttr(Matrix::L1_TO_L0_TILE)) {
        // 大包搬运分支，无法直接从srcValidShape推导至输出dstValidShape，需要获取offset、tile信息
        std::vector<SymbolicScalar> offset;
        std::vector<SymbolicScalar> tile;
        op->GetAttr(Matrix::L1_TO_L0_OFFSET, offset);
        op->GetAttr(Matrix::L1_TO_L0_TILE, tile);
        ASSERT(MatmulErrorCode::ERR_CONFIG_TILE, offset.size() == SHAPE_DIM2) << "offset.size() should be SHAPE_DIM2";
        ASSERT(MatmulErrorCode::ERR_CONFIG_TILE, tile.size() == SHAPE_DIM2) << "tile.size() should be SHAPE_DIM2";
        ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, !op->GetIOperands().empty() && op->GetIOperands()[0] != nullptr &&
                                                       op->GetIOperands()[0]->GetDynValidShape().size() == SHAPE_DIM2)
            << "op->GetIOperands() check failed or GetDynValidShape().size() should be SHAPE_DIM2";
        std::vector<SymbolicScalar> srcValidShape = op->GetIOperands()[0]->GetDynValidShape();
        std::vector<SymbolicScalar> dstValidShape = GetViewValidShape(
            srcValidShape, SymbolicScalar::Concrete(offset, 0), offset, SymbolicScalar::Concrete(tile, 0));
        ASSERT(MatmulErrorCode::ERR_CONFIG_TILE, dstValidShape.size() == SHAPE_DIM2)
            << "dstValidShape.size() should be SHAPE_DIM2";
        if constexpr (isTrans) {
            // L0A始终保持(M, K)，L0B始终保持(K, N)
            std::swap(dstValidShape[0], dstValidShape[1]);
        }
        for (auto output : op->GetOOperands()) {
            outValidShapes.push_back(dstValidShape);
        }
        return;
    }
    // 普通分支，srcValidShape与dstValidShape相同
    std::vector<std::vector<SymbolicScalar>> inputValidShapes;
    for (auto inputTensor : op->GetIOperands()) {
        ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, inputTensor != nullptr) << "inputTensor should not be nullptr";
        inputValidShapes.push_back(inputTensor->GetDynValidShape());
    }
    if (inputValidShapes.empty() || inputValidShapes[0].size() != SHAPE_DIM2) {
        return;
    }
    for (auto output : op->GetOOperands()) {
        if constexpr (isTrans) {
            outValidShapes.push_back({inputValidShapes[0][1], inputValidShapes[0][0]});
        } else {
            outValidShapes.push_back(inputValidShapes[0]);
        }
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_L1_TO_L0A, Opcode::OP_L1_TO_L0A, LoadL0InferFunc<false>);
REGISTER_INFER_SHAPE_FUNC(OP_L1_TO_L0B, Opcode::OP_L1_TO_L0B, LoadL0InferFunc<false>);
REGISTER_INFER_SHAPE_FUNC(OP_L1_TO_L0_AT, Opcode::OP_L1_TO_L0_AT, LoadL0InferFunc<true>);
REGISTER_INFER_SHAPE_FUNC(OP_L1_TO_L0_BT, Opcode::OP_L1_TO_L0_BT, LoadL0InferFunc<true>);

// MTE infer shape func
void LoadL0MXInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, op != nullptr) << "op should not be nullptr";
    // 大包搬运分支，无法直接从srcValidShape推导至输出dstValidShape，需要获取offset、tile信息
    std::vector<SymbolicScalar> offset;
    std::vector<SymbolicScalar> tile;
    op->GetAttr(Matrix::L1_TO_L0_OFFSET, offset);
    op->GetAttr(Matrix::L1_TO_L0_TILE, tile);
    ASSERT(MatmulErrorCode::ERR_CONFIG_TILE, offset.size() == SHAPE_DIM3) << "offset.size() should be SHAPE_DIM3";
    ASSERT(MatmulErrorCode::ERR_CONFIG_TILE, tile.size() == SHAPE_DIM3) << "tile.size() should be SHAPE_DIM3";
    ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, !op->GetIOperands().empty() && op->GetIOperands()[0] != nullptr &&
                                                   op->GetIOperands()[0]->GetDynValidShape().size() == SHAPE_DIM3)
        << "op->GetIOperands() check failed or GetDynValidShape().size() should be SHAPE_DIM3";
    std::vector<SymbolicScalar> srcValidShape = op->GetIOperands()[0]->GetDynValidShape();
    std::vector<SymbolicScalar> dstValidShape = GetViewValidShape(srcValidShape, SymbolicScalar::Concrete(offset, 0),
                                                                  offset, SymbolicScalar::Concrete(tile, 0));
    ASSERT(MatmulErrorCode::ERR_CONFIG_TILE, dstValidShape.size() == SHAPE_DIM3)
        << "dstValidShape.size() should be SHAPE_DIM3";
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(dstValidShape);
    }
    return;
}
REGISTER_INFER_SHAPE_FUNC(OP_L1_TO_L0A_SCALE, Opcode::OP_L1_TO_L0A_SCALE, LoadL0MXInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_L1_TO_L0B_SCALE, Opcode::OP_L1_TO_L0B_SCALE, LoadL0MXInferFunc);

// conv infer
namespace {
std::vector<SymbolicScalar> InferSrcGmValidShapeForFmap3D(const std::vector<SymbolicScalar>& l1TileShape,
                                                          const std::vector<SymbolicScalar>& gmInputValidShape,
                                                          const std::vector<SymbolicScalar>& fromOffset, int64_t c0)
{
    SymbolicScalar alignedCin = l1TileShape[2] * SymbolicScalar(c0);
    SymbolicScalar srcGmCin = alignedCin;
    // N(batch)/D 维需按 view 有效范围裁剪: 尾块平面 (n_sub/d_sub 超出有效 batch/d 数) 不得读 GM,
    // 否则 TLoadConv 以满幅窗口从越界偏移读取 (如 batch 偏移超出 tensor 末尾触发 aicore error)。
    SymbolicScalar srcGmN = l1TileShape[0];
    SymbolicScalar srcGmD = l1TileShape[1];
    SymbolicScalar srcGmH = l1TileShape[3];
    SymbolicScalar srcGmW = l1TileShape[4];
    if (gmInputValidShape.size() >= SHAPE_DIM4 && fromOffset.size() >= SHAPE_DIM4) {
        srcGmCin = (gmInputValidShape[1] - fromOffset[1]).Min(alignedCin);
        srcGmN = (gmInputValidShape[0] - fromOffset[0]).Min(l1TileShape[0]).Max(0);
        srcGmD = (gmInputValidShape[2] - fromOffset[2]).Min(l1TileShape[1]).Max(0);
        srcGmH = (gmInputValidShape[3] - fromOffset[3]).Min(l1TileShape[3]).Max(0);
        srcGmW = (gmInputValidShape[4] - fromOffset[4]).Min(l1TileShape[4]).Max(0);
    }
    return {srcGmN, srcGmCin, srcGmD, srcGmH, srcGmW};
}

std::vector<SymbolicScalar> InferSrcGmValidShapeForFmap2D(const std::vector<SymbolicScalar>& l1TileShape,
                                                          const std::vector<SymbolicScalar>& gmInputValidShape,
                                                          const std::vector<SymbolicScalar>& fromOffset, int64_t c0)
{
    SymbolicScalar alignedCin = l1TileShape[1] * SymbolicScalar(c0);
    SymbolicScalar srcGmCin = alignedCin;
    // N(batch) 维需按 view 有效范围裁剪, 避免 batch 尾块满幅越界读 GM。
    SymbolicScalar srcGmN = l1TileShape[0];
    SymbolicScalar srcGmH = l1TileShape[2];
    SymbolicScalar srcGmW = l1TileShape[3];
    if (gmInputValidShape.size() >= SHAPE_DIM4 && fromOffset.size() >= SHAPE_DIM4) {
        srcGmCin = (gmInputValidShape[1] - fromOffset[1]).Min(alignedCin);
        srcGmN = (gmInputValidShape[0] - fromOffset[0]).Min(l1TileShape[0]).Max(0);
        srcGmH = (gmInputValidShape[2] - fromOffset[2]).Min(l1TileShape[2]).Max(0);
        srcGmW = (gmInputValidShape[3] - fromOffset[3]).Min(l1TileShape[3]).Max(0);
    }
    return {srcGmN, srcGmCin, srcGmH, srcGmW};
}

std::vector<SymbolicScalar> InferSrcGmValidShapeForWeight(Operation* op, const std::vector<SymbolicScalar>& l1TileShape,
                                                          const std::vector<SymbolicScalar>& gmInputValidShape,
                                                          const std::vector<SymbolicScalar>& fromOffset, int64_t c0,
                                                          bool isConv3D)
{
    int64_t kh = 1;
    int64_t kw = 1;
    op->GetAttr(OpAttributeKey::filterH, kh);
    op->GetAttr(OpAttributeKey::filterW, kw);
    int64_t khxkw = (kh * kw == 0) ? 1 : kh * kw;
    int64_t dkL1Size = 1;
    if (isConv3D) {
        op->GetAttr("CONV_DK_L1_SIZE", dkL1Size);
    }
    int64_t dkxkhxkw = (dkL1Size * khxkw == 0) ? 1 : dkL1Size * khxkw;
    SymbolicScalar cin1 = l1TileShape[0] / SymbolicScalar(dkxkhxkw);
    SymbolicScalar alignedCin = cin1 * SymbolicScalar(c0);
    SymbolicScalar srcGmCin = alignedCin;
    if (gmInputValidShape.size() >= SHAPE_DIM4 && fromOffset.size() >= SHAPE_DIM4) {
        srcGmCin = (gmInputValidShape[1] - fromOffset[1]).Min(alignedCin);
    }
    if (isConv3D) {
        return {l1TileShape[1] * l1TileShape[2], srcGmCin, SymbolicScalar(dkL1Size), SymbolicScalar(kh),
                SymbolicScalar(kw)};
    }
    return {l1TileShape[1] * l1TileShape[2], srcGmCin, SymbolicScalar(kh), SymbolicScalar(kw)};
}

std::vector<SymbolicScalar> InferSrcGmValidShapeDN2NZ(Operation* op, const std::vector<SymbolicScalar>& l1TileShape,
                                                      const std::shared_ptr<CopyOpAttribute>& copyAttr, int64_t c0,
                                                      bool isFmap, bool isConv3D)
{
    auto gmInputValidShape = op->GetIOperands()[0]->GetDynValidShape();
    auto fromOffset = OpImmediate::ToSpecified(copyAttr->GetFromOffset());
    if (isConv3D && isFmap && l1TileShape.size() == SHAPE_DIM6) {
        return InferSrcGmValidShapeForFmap3D(l1TileShape, gmInputValidShape, fromOffset, c0);
    }
    if (isFmap && l1TileShape.size() == SHAPE_DIM5) {
        return InferSrcGmValidShapeForFmap2D(l1TileShape, gmInputValidShape, fromOffset, c0);
    }
    if (!isFmap && l1TileShape.size() == SHAPE_DIM4) {
        return InferSrcGmValidShapeForWeight(op, l1TileShape, gmInputValidShape, fromOffset, c0, isConv3D);
    }
    return {};
}
} // namespace

void L1CopyInConvBpInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    ASSERT(MatmulErrorCode::ERR_RUNTIME_NULLPTR, op != nullptr) << "op should not be nullptr";
    const std::string L1_TILE_SHAPE = "l1_tile_shape";
    const std::string IS_FMAP_FLAG = "IS_FMAP";
    ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, op->HasAttr(L1_TILE_SHAPE)) << "op should have L1_TILE_SHAPE attr";
    std::vector<SymbolicScalar> tile;
    op->GetAttr(L1_TILE_SHAPE, tile);
    ASSERT(MatmulErrorCode::ERR_PARAM_INVALID, op->HasAttr(IS_FMAP_FLAG)) << "op should have IS_FMAP_FLAG attr";
    bool isFmap = false;
    op->GetAttr(IS_FMAP_FLAG, isFmap);
    if (isFmap) {
        ASSERT(MatmulErrorCode::ERR_CONFIG_TILE, tile.size() == SHAPE_DIM5 || tile.size() == SHAPE_DIM6)
            << "tile.size() should be SHAPE_DIM5 or SHAPE_DIM6";
    } else {
        ASSERT(MatmulErrorCode::ERR_CONFIG_TILE, tile.size() == SHAPE_DIM4) << "tile.size() should be SHAPE_DIM4";
    }
    std::vector<SymbolicScalar> outShape;
    for (size_t i = 0; i < tile.size(); i++) {
        outShape.push_back(tile[i]);
    }
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outShape);
    }
}

void L1CopyInConvInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    ASSERT(ConvExpandFuncError::EXPANDFUNC_TENSOR_OP_NULLPTR, op != nullptr) << "op should not be nullptr";
    const std::string IS_FMAP_FLAG = "IS_FMAP";
    const std::string COPY_IN_MODE_KEY = "COPY_IN_MODE";
    auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
    ASSERT(ConvExpandFuncError::EXPANDFUNC_TENSOR_ATTR_GET_FAILED, copyAttr != nullptr)
        << "L1CopyInConv op should have CopyOpAttribute";
    auto l1TileShape = OpImmediate::ToSpecified(copyAttr->GetToDynValidShape());
    ASSERT(ConvExpandFuncError::EXPANDFUNC_TENSOR_ATTR_GET_FAILED, op->HasAttr(IS_FMAP_FLAG))
        << "op should have IS_FMAP_FLAG attr";

    bool isFmap = false;
    op->GetAttr(IS_FMAP_FLAG, isFmap);
    int64_t copyInMode = 0;
    op->GetAttr(COPY_IN_MODE_KEY, copyInMode);
    bool isConv3D = false;
    op->GetAttr("IS_CONV3D", isConv3D);

    if (isFmap) {
        ASSERT(ConvExpandFuncError::EXPANDFUNC_PARAMS_INVALID,
               l1TileShape.size() == SHAPE_DIM5 || l1TileShape.size() == SHAPE_DIM6)
            << "tile.size() should be SHAPE_DIM5 or SHAPE_DIM6";
    } else {
        ASSERT(ConvExpandFuncError::EXPANDFUNC_PARAMS_INVALID, l1TileShape.size() == SHAPE_DIM4)
            << "tile.size() should be SHAPE_DIM4";
    }

    std::vector<SymbolicScalar> outShape(l1TileShape.begin(), l1TileShape.end());
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outShape);
    }

    int64_t c0 = l1TileShape.empty() ? 16 : l1TileShape.back().Concrete();
    const int64_t COPY_MOD_NZ2NZ = 2;
    const int64_t COPY_MOD_DN2NZ = 3;
    std::vector<SymbolicScalar> srcGmValidShape;
    if (copyInMode == COPY_MOD_NZ2NZ) {
        srcGmValidShape = l1TileShape;
    } else if (copyInMode == COPY_MOD_DN2NZ) {
        srcGmValidShape = InferSrcGmValidShapeDN2NZ(op, l1TileShape, copyAttr, c0, isFmap, isConv3D);
    }
    if (!srcGmValidShape.empty()) {
        op->SetAttribute(OpAttributeKey::srcGmConvValidShape, srcGmValidShape);
        // 取交: 尾块越界平面的 tile valid 收缩为 0, TLOAD 依 dst tile 形状决定拷贝量, 从而不读 GM。
        if (isFmap && copyInMode == COPY_MOD_DN2NZ && srcGmValidShape.size() >= SHAPE_DIM4) {
            outShape[0] = srcGmValidShape[0].Min(outShape[0]);
            const bool maskDepth = isConv3D && l1TileShape.size() == SHAPE_DIM6 && srcGmValidShape.size() >= SHAPE_DIM5;
            if (maskDepth) {
                outShape[1] = srcGmValidShape[2].Min(outShape[1]);
            }
            for (auto& validShape : outValidShapes) {
                if (validShape.size() == outShape.size()) {
                    validShape[0] = outShape[0];
                    if (maskDepth && validShape.size() > 1) {
                        validShape[1] = outShape[1];
                    }
                }
            }
            // COA 的 valid 槽位取自 copyAttr 的 ToDynValidShape, 需同步掩码, 使 CCE 侧
            // tile 形状参数(GET_PARAM_VALID_SHAPE)逐平面收缩为 0, 阻断越界 GM 读。
            copyAttr->SetToDynValidShape(OpImmediate::Specified(outShape));
        }
    }
}

void L1ToL0ConvInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    ASSERT(ConvExpandFuncError::EXPANDFUNC_TENSOR_OP_NULLPTR, op != nullptr) << "op should not be nullptr";
    const std::string L0_TILE_SHAPE = "l0_tile_shape";
    ASSERT(ConvExpandFuncError::EXPANDFUNC_TENSOR_ATTR_GET_FAILED, op->HasAttr(L0_TILE_SHAPE))
        << "op should have L0_TILE_SHAPE attr";
    std::vector<SymbolicScalar> tile;
    op->GetAttr(L0_TILE_SHAPE, tile);
    std::vector<SymbolicScalar> outShape(tile.begin(), tile.end());
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outShape);
    }
}

void L0CCopyOutConvInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    ASSERT(ConvExpandFuncError::EXPANDFUNC_TENSOR_OP_NULLPTR, op != nullptr) << "op should not be nullptr";
    if (!(op->GetOOperands()[0]->GetDynValidShape().empty())) {
        outValidShapes.push_back(op->GetOOperands()[0]->GetDynValidShape());
    }
}

// L0C(NZ 二维) -> UB(NCHW/NCDHW tile) 直通搬运的输出 valid 推导:
// 由本 copy 的有效 (M, N) (l0cValidMN 属性, VIEW 路径已按窗口刷新) 与 tile shape 推导
//   c = min(M_valid, tile_c)
//   w = min(realCutW, tile_w)
//   行 = min(N_valid... M_valid, tile_h * tile_w)
//   h = 行 / w
// INSERT (小搬大, 多个 L0C 块以 toOffset 拼入同一 tile) 场景: 各 copy 的贡献为
// toOffset + 窗口范围, 与已有 valid 按维取 max 合并 (参考 CopyOutInferFunc)
void L0CCopyUBConvInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    ASSERT(ConvExpandFuncError::EXPANDFUNC_TENSOR_OP_NULLPTR, op != nullptr) << "op should not be nullptr";
    auto output = op->GetOOperands()[0];
    auto input = op->GetIOperands()[0];
    if (output == nullptr || input == nullptr || input->GetDynValidShape().size() != SHAPE_DIM2) {
        if (output != nullptr && !output->GetDynValidShape().empty()) {
            outValidShapes.push_back(output->GetDynValidShape());
        }
        return;
    }
    const auto& inValid = input->GetDynValidShape();
    const Shape& tileShape = output->GetShape();
    auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
    // 本 copy 的有效 (M, N): 优先取 l0cValidMN 属性 (GenerateMoveOp 已按 view 窗口刷新,
    // 携带 conv tile 的运行时有效行列; 动态场景下 tensor 的 dynValidShape 可能被重置回
    // 静态全量), 缺省回退到输入 tensor 的 valid
    SymbolicScalar copyM = (inValid.size() == SHAPE_DIM2) ? inValid[0] : SymbolicScalar(0);
    SymbolicScalar copyN = (inValid.size() == SHAPE_DIM2) ? inValid[1] : SymbolicScalar(0);
    std::vector<SymbolicScalar>* validMN = op->GetAttr<std::vector<SymbolicScalar>>(OpAttributeKey::l0cValidMN);
    if (validMN != nullptr && validMN->size() == SHAPE_DIM2) {
        copyM = (*validMN)[0];
        copyN = (*validMN)[1];
    }
    // conv tile 的运行时有效 W (REAL_CUT_W 属性), 缺省取 tile 的 w
    SymbolicScalar realCutW = SymbolicScalar(tileShape.back());
    SymbolicScalar realCutWAttr;
    if (op->GetAttr(Conv::LoadStoreConvOpAttributeKey::realCutW, realCutWAttr) && realCutWAttr.IsValid()) {
        realCutW = realCutWAttr;
    }
    SymbolicScalar validN = copyN.Min(tileShape[1]).Max(0);
    SymbolicScalar validW = realCutW.Min(tileShape.back());
    SymbolicScalar validRows = copyM.Min(tileShape[tileShape.size() - NUM2] * tileShape.back()).Max(0);
    SymbolicScalar validH = validRows / validW.Max(1);
    // INSERT (小搬大) 判定: 输出存在多个生产者, 或目的偏移非零 (多块拼入同一 tile)
    bool isInsertPath = output->GetProducers().size() > 1;
    if (!isInsertPath && copyAttr != nullptr) {
        for (const auto& off : copyAttr->GetToOffset()) {
            if (off.IsSpecified()) {
                const auto& val = off.GetSpecifiedValue();
                if (!val.ConcreteValid() || val.Concrete() != 0) {
                    isInsertPath = true;
                    break;
                }
            }
        }
    }
    std::vector<SymbolicScalar> outShape;
    if (!isInsertPath) {
        outShape.push_back(SymbolicScalar(tileShape[0])); // n: batch 维由外层 tiling 迭代, tile 内恒满
        outShape.push_back(validN);
        if (tileShape.size() == SHAPE_DIM5) { // NCDHW: d 维由外层 tiling 迭代
            outShape.push_back(SymbolicScalar(tileShape[2]));
        }
        outShape.push_back(validH);
        outShape.push_back(validW);
        // 大搬小场景需要更新op的RealM和RealCutW参数
        op->SetAttribute(Conv::LoadStoreConvOpAttributeKey::realCutW, validW);
        op->SetAttribute(OpAttributeKey::l0cValidMN, std::vector<SymbolicScalar>{validRows, validN});
    } else {
        // 本 copy 的窗口范围 (n=1, c, [d=1,] h, w), 以 toOffset 平移后与已有 valid 按维取 max
        std::vector<SymbolicScalar> extent;
        extent.push_back(SymbolicScalar(1));
        extent.push_back(validN);
        if (tileShape.size() == SHAPE_DIM5) {
            extent.push_back(SymbolicScalar(1));
        }
        extent.push_back(validH);
        extent.push_back(validW);
        const auto& toOffset = copyAttr->GetToOffset();
        const auto& existValid = output->GetDynValidShape();
        for (size_t i = 0; i < extent.size(); ++i) {
            SymbolicScalar exist = (i < existValid.size()) ? existValid[i] : SymbolicScalar(0);
            SymbolicScalar offset = (i < toOffset.size() && toOffset[i].IsSpecified()) ?
                                        toOffset[i].GetSpecifiedValue() :
                                        SymbolicScalar(0);
            SymbolicScalar merged = exist.Max((extent[i] + offset) * (extent[i] != 0));
            // infer 相互独立, 仅依赖当前 exist 会因 infer 顺序/覆盖丢失先序 copy 的贡献
            for (const auto& producer : output->GetProducers()) {
                if (producer == op || producer->GetOpcode() != Opcode::OP_L0C_COPY_UB_CONV) {
                    continue;
                }
                auto producerInput = producer->GetIOperands()[0];
                if (producerInput == nullptr || producerInput->GetDynValidShape().size() != SHAPE_DIM2) {
                    continue;
                }
                std::vector<SymbolicScalar>* producerValidMN = producer->GetAttr<std::vector<SymbolicScalar>>(
                    OpAttributeKey::l0cValidMN);
                if (producerValidMN == nullptr || producerValidMN->size() != SHAPE_DIM2) {
                    continue;
                }
                SymbolicScalar producerRealCutW = SymbolicScalar(tileShape.back());
                SymbolicScalar producerRealCutWAttr;
                if (producer->GetAttr(Conv::LoadStoreConvOpAttributeKey::realCutW, producerRealCutWAttr) &&
                    producerRealCutWAttr.IsValid()) {
                    producerRealCutW = producerRealCutWAttr;
                }
                SymbolicScalar producerValidN = (*producerValidMN)[1].Min(tileShape[1]).Max(0);
                SymbolicScalar producerValidW = producerRealCutW.Min(tileShape.back());
                SymbolicScalar producerValidRows = (*producerValidMN)[0]
                                                       .Min(tileShape[tileShape.size() - NUM2] * tileShape.back())
                                                       .Max(0);
                SymbolicScalar producerValidH = producerValidRows / producerValidW.Max(1);
                SymbolicScalar producerExtent = (i == 1)                    ? producerValidN :
                                                (i == extent.size() - NUM2) ? producerValidH :
                                                (i == extent.size() - 1)    ? producerValidW :
                                                                              SymbolicScalar(1);
                auto producerCopyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(producer->GetOpAttribute());
                SymbolicScalar producerOffset = SymbolicScalar(0);
                if (producerCopyAttr != nullptr && i < producerCopyAttr->GetToOffset().size() &&
                    producerCopyAttr->GetToOffset()[i].IsSpecified()) {
                    producerOffset = producerCopyAttr->GetToOffset()[i].GetSpecifiedValue();
                }
                merged = merged.Max((producerExtent + producerOffset) * (producerExtent != 0));
            }
            outShape.push_back(merged);
        }
    }
    for (auto outputTensor : op->GetOOperands()) {
        outValidShapes.push_back(outShape);
    }
}

REGISTER_INFER_SHAPE_FUNC(OP_L1_COPY_IN_CONV, Opcode::OP_L1_COPY_IN_CONV, L1CopyInConvInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_L1_COPY_IN_CONV_BP_DX_DY, Opcode::OP_L1_COPY_IN_CONV_BP_DX_DY, L1CopyInConvBpInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_L1_COPY_IN_CONV_BP, Opcode::OP_L1_COPY_IN_CONV_BP, L1CopyInConvBpInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_LOAD3D_CONV, Opcode::OP_LOAD3D_CONV, L1ToL0ConvInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_LOAD2D_CONV, Opcode::OP_LOAD2D_CONV, L1ToL0ConvInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_LOAD2DDX_CONV, Opcode::OP_LOAD2DDX_CONV, L1ToL0ConvInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_L0C_COPY_OUT_CONV, Opcode::OP_L0C_COPY_OUT_CONV, L0CCopyOutConvInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_L0C_COPY_UB_CONV, Opcode::OP_L0C_COPY_UB_CONV, L0CCopyUBConvInferFunc);

void TransDataInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    // Keep the valid shape carried by the output tensor itself; only fall back to the leading
    // transDataOffset slots (filled in trans_data.cpp) when the tensor does not carry one.
    const auto& outTensor = op->GetOOperands()[0];
    std::vector<SymbolicScalar> outValidShape = outTensor->GetDynValidShape();
    std::vector<SymbolicScalar> transDataAttr;
    op->GetAttr(OpAttributeKey::transDataOffset, transDataAttr);
    if (transDataAttr.size() >= outTensor->GetShape().size()) {
        outValidShape.assign(transDataAttr.begin(), transDataAttr.begin() + outTensor->GetShape().size());
    }
    outValidShapes.push_back(outValidShape);
    outValidShapes.push_back(SymbolicScalar::FromConcrete(op->GetOOperands()[1]->GetShape()));
}

REGISTER_INFER_SHAPE_FUNC(OP_NCHW2NC1HWC0, Opcode::OP_NCHW2NC1HWC0, TransDataInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_NCHW2Fractal_Z, Opcode::OP_NCHW2Fractal_Z, TransDataInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_NC1HWC02NCHW, Opcode::OP_NC1HWC02NCHW, TransDataInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_NCDHW2NDC1HWC0, Opcode::OP_NCDHW2NDC1HWC0, TransDataInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_NCDHW2FRACTAL_Z_3D, Opcode::OP_NCDHW2FRACTAL_Z_3D, TransDataInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_NDC1HWC02NCDHW, Opcode::OP_NDC1HWC02NCDHW, TransDataInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_FractalZ2NCHW, Opcode::OP_FractalZ2NCHW, TransDataInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_FractalZ3D2NCDHW, Opcode::OP_FractalZ3D2NCDHW, TransDataInferFunc);

void CopyInInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    auto copyOpAttribute = std::dynamic_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
    if (!(op->GetOOperands()[0]->GetDynValidShape().empty())) {
        outValidShapes.push_back(op->GetOOperands()[0]->GetDynValidShape());
        if (copyOpAttribute != nullptr && (copyOpAttribute->GetToDynValidShape()).empty()) {
            auto toDynShape = OpImmediate::Specified(op->GetOOperands()[0]->GetDynValidShape());
            copyOpAttribute->SetToDynValidShape(toDynShape);
        }
        return;
    }

    // 连接incast
    auto toValidShape = copyOpAttribute->GetToDynValidShape();
    std::vector<SymbolicScalar> toValidShapeSym(toValidShape.size());
    OpImmediate::NormalizeValue(toValidShapeSym, 0, toValidShape, 0, false);
    auto toValidShapeValue = SymbolicScalar::Concrete(toValidShapeSym, -1);
    auto tileShape = copyOpAttribute->GetShape();
    std::vector<SymbolicScalar> tileShapeSym(tileShape.size());
    OpImmediate::NormalizeValue(tileShapeSym, 0, tileShape, 0, false);
    if (!toValidShape.empty()) {
        for (auto output : op->GetOOperands()) {
            outValidShapes.push_back(toValidShapeSym);
        }
        return;
    }
    if (!(op->GetOOperands()[0]->GetDynValidShape().empty())) {
        outValidShapes.push_back(op->GetOOperands()[0]->GetDynValidShape());
        auto toDynShape = OpImmediate::Specified(op->GetOOperands()[0]->GetDynValidShape());
        copyOpAttribute->SetToDynValidShape(toDynShape);
        return;
    }
    // 临时空间，固定大小
    if (op->GetIOperands()[0]->GetProducers().empty()) {
        if (toValidShape.empty()) {
            std::vector<SymbolicScalar> toValidShapeVec;
            for (auto dim : copyOpAttribute->GetShape()) {
                toValidShapeVec.push_back(dim.GetSpecifiedValue());
            }
            for (auto output : op->GetOOperands()) {
                outValidShapes.push_back(toValidShapeVec);
            }
        }
        auto toDynShape = OpImmediate::Specified(outValidShapes[0]);
        copyOpAttribute->SetToDynValidShape(toDynShape);
        return;
    }
    // 子图边界, 需要重新推导
    std::vector<std::vector<SymbolicScalar>> inputShapes;
    for (auto inputTensor : op->GetIOperands()) {
        inputShapes.push_back(inputTensor->GetDynValidShape());
    }
    auto offset = copyOpAttribute->GetFromOffset();
    std::vector<SymbolicScalar> oriOffset;
    for (auto offsetValue : offset) {
        oriOffset.push_back(offsetValue.GetSpecifiedValue());
    }
    std::vector<SymbolicScalar> outputShape;
    for (size_t i = 0U; i < inputShapes[0].size(); i++) {
        SymbolicScalar actualDim = std::max(0, std::min((inputShapes[0][i] - oriOffset[i]), tileShapeSym[i]));
        outputShape.push_back(actualDim);
    }
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outputShape);
    }
    // 设置validshape到copyin的toDynvalidshape
    auto toDynShape = OpImmediate::Specified(outputShape);
    copyOpAttribute->SetToDynValidShape(toDynShape);
}
REGISTER_INFER_SHAPE_FUNC(OP_COPY_IN, Opcode::OP_COPY_IN, CopyInInferFunc);

void ReshapeCopyInInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    if (!op->GetIOperands().empty() && !op->GetIOperands()[0]->GetDynValidShape().empty()) {
        auto validShape = op->GetIOperands().front()->GetDynValidShape();
        outValidShapes.push_back(validShape);
        auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
        if (copyAttr == nullptr) {
            return;
        }
        copyAttr->SetFromDynValidShape(OpImmediate::Specified(validShape));
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_RESHAPE_COPY_IN, Opcode::OP_RESHAPE_COPY_IN, ReshapeCopyInInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_L1_RESHAPE_COPY_IN, Opcode::OP_L1_RESHAPE_COPY_IN, ReshapeCopyInInferFunc);

void ReshapeCopyOutInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    if (!op->GetOOperands().empty() && !op->GetOOperands()[0]->GetDynValidShape().empty()) {
        auto validshape = op->GetOOperands()[0]->GetDynValidShape();
        outValidShapes.push_back(validshape);
        auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
        if (copyAttr == nullptr) {
            return;
        }
        copyAttr->SetToDynValidShape(OpImmediate::Specified(validshape));
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_RESHAPE_COPY_OUT, Opcode::OP_RESHAPE_COPY_OUT, ReshapeCopyOutInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_L0C_RESHAPE_COPY_OUT, Opcode::OP_L0C_RESHAPE_COPY_OUT, ReshapeCopyOutInferFunc);

void ShmemPutInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    auto copyOpAttribute = std::dynamic_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
    if (op->iOperand[2]->GetDynOffset().size() != 0) {
        copyOpAttribute->SetFromOffset(OpImmediate::Specified(op->iOperand[2]->GetDynOffset()));
    } else {
        copyOpAttribute->SetFromOffset(OpImmediate::Specified(op->iOperand[2]->GetOffset()));
    }
    std::vector<SymbolicScalar> shmemPutDynValidShape(copyOpAttribute->GetFromDynValidShape().size());
    OpImmediate::NormalizeValue(shmemPutDynValidShape, 0, copyOpAttribute->GetFromDynValidShape(), 0, false);
    for (size_t i = 0; i < op->GetOOperands().size(); i++) {
        if (i == 0) {
            outValidShapes.push_back(shmemPutDynValidShape);
        } else {
            outValidShapes.push_back(SymbolicScalar::FromConcrete(op->GetOOperands()[i]->GetShape()));
        }
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_SHMEM_PUT, Opcode::OP_SHMEM_PUT, ShmemPutInferFunc);

void ShmemGetInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    auto copyOpAttribute = std::dynamic_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
    std::vector<SymbolicScalar> shmemGetDynValidShape(copyOpAttribute->GetFromDynValidShape().size());
    OpImmediate::NormalizeValue(shmemGetDynValidShape, 0, copyOpAttribute->GetFromDynValidShape(), 0, false);
    for (size_t i = 0; i < op->GetOOperands().size(); i++) {
        if (i == 0) {
            outValidShapes.push_back(shmemGetDynValidShape);
        } else {
            outValidShapes.push_back(SymbolicScalar::FromConcrete(op->GetOOperands()[i]->GetShape()));
        }
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_SHMEM_GET, Opcode::OP_SHMEM_GET, ShmemGetInferFunc);

void ShmemStoreInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    auto copyOpAttribute = std::dynamic_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
    if (op->iOperand[1]->GetDynOffset().size() != 0) {
        copyOpAttribute->SetFromOffset(OpImmediate::Specified(op->iOperand[1]->GetDynOffset()));
    } else {
        copyOpAttribute->SetFromOffset(OpImmediate::Specified(op->iOperand[1]->GetOffset()));
    }
    std::vector<SymbolicScalar> shmemStoreDynValidShape(copyOpAttribute->GetFromDynValidShape().size());
    OpImmediate::NormalizeValue(shmemStoreDynValidShape, 0, copyOpAttribute->GetFromDynValidShape(), 0, false);
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(shmemStoreDynValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_SHMEM_STORE, Opcode::OP_SHMEM_STORE, ShmemStoreInferFunc);

void ShmemLoadInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    auto copyOpAttribute = std::dynamic_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
    if (op->iOperand[1]->GetDynOffset().size() != 0) {
        copyOpAttribute->SetFromOffset(OpImmediate::Specified(op->iOperand[1]->GetDynOffset()));
    } else {
        copyOpAttribute->SetFromOffset(OpImmediate::Specified(op->iOperand[1]->GetOffset()));
    }
    std::vector<SymbolicScalar> shmemLoadDynValidShape(copyOpAttribute->GetToDynValidShape().size());
    OpImmediate::NormalizeValue(shmemLoadDynValidShape, 0, copyOpAttribute->GetToDynValidShape(), 0, false);
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(shmemLoadDynValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_SHMEM_LOAD, Opcode::OP_SHMEM_LOAD, ShmemLoadInferFunc);

void CopyOutInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    auto copyOpAttribute = std::dynamic_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
    if (copyOpAttribute != nullptr) {
        copyOpAttribute->SetFromDynValidShape(OpImmediate::Specified(op->GetIOperands()[0]->GetDynValidShape()));
    } else {
        VECTOR_LOGW("Copyout [%d] has no copy out attr.", op->GetOpMagic());
        outValidShapes.push_back(op->GetIOperands()[0]->GetDynValidShape());
        return;
    }

    // 多个tile块copyout到同一个tensor时， 每一个tile都需要推导
    bool needInferShape = false;
    if (!(op->GetOOperands()[0]->GetDynValidShape().empty()) &&
        !op->GetOOperands()[0]->GetAttr(COPY_OUT_FORCE_INFER_SHAPE, needInferShape)) {
        outValidShapes.push_back(op->GetOOperands()[0]->GetDynValidShape());
        return;
    }

    op->GetOOperands()[0]->SetAttr(COPY_OUT_FORCE_INFER_SHAPE, true);

    auto offset = copyOpAttribute->GetToOffset();
    std::vector<SymbolicScalar> oriOffset;
    for (auto offsetValue : offset) {
        oriOffset.push_back(offsetValue.GetSpecifiedValue());
    }

    std::vector<std::vector<SymbolicScalar>> inputShapes;
    std::vector<std::vector<int64_t>> staticInputShapes;

    for (auto inputTensor : op->GetIOperands()) {
        inputShapes.push_back(inputTensor->GetDynValidShape());
        staticInputShapes.push_back(inputTensor->GetShape());
    }
    std::vector<SymbolicScalar> outDynShape = op->GetOOperands()[0]->GetDynValidShape();
    if (outDynShape.empty()) {
        for (size_t i = 0; i < op->GetOOperands()[0]->GetShape().size(); ++i) {
            outDynShape.push_back(SymbolicScalar(0));
        }
    }
    std::vector<SymbolicScalar> outShape;
    for (size_t i = 0U; i < inputShapes[0].size(); i++) {
        SymbolicScalar actualDim;
        if (staticInputShapes[0][i] ==
            op->GetOOperands()[0]->GetShape()[i]) { // src的该维度没有被切分，assmble后该维度大小不变
            actualDim = std::max(SymbolicScalar(0), inputShapes[0][i] + oriOffset[i]);
        } else {
            actualDim = std::max(outDynShape[i], (inputShapes[0][i] + oriOffset[i]) * (inputShapes[0][i] != 0));
        }
        outShape.push_back(actualDim);
    }
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_COPY_OUT, Opcode::OP_COPY_OUT, CopyOutInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_UB_COPY_L1, Opcode::OP_UB_COPY_L1, CopyOutInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_L0C_COPY_UB, Opcode::OP_L0C_COPY_UB, CopyOutInferFunc);

// MTE infer shape func
void TransposeInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    std::vector<std::vector<SymbolicScalar>> inputValidShapes;
    for (auto inputTensor : op->GetIOperands()) {
        inputValidShapes.push_back(inputTensor->GetDynValidShape());
    }
    if (inputValidShapes.empty()) {
        return;
    }
    for (auto output : op->GetOOperands()) {
        std::vector<SymbolicScalar> res;
        res.insert(res.end(), inputValidShapes[0].begin(), inputValidShapes[0].end());
        auto axises = op->GetVectorIntAttribute<int>(OP_ATTR_PREFIX + "shape");
        size_t index0 = axises[0];
        size_t index1 = axises[1];
        if (index0 < res.size() && index1 < res.size()) {
            std::swap(res[index0], res[index1]);
        }
        outValidShapes.push_back(res);
    }
    if (op->GetOpcode() == Opcode::OP_TRANSPOSE_MOVEIN) {
        auto copyOpAttribute = std::dynamic_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
        if (copyOpAttribute != nullptr) {
            copyOpAttribute->SetToDynValidShape(OpImmediate::Specified(outValidShapes[0]));
        }
    }
    if (op->GetOpcode() == Opcode::OP_TRANSPOSE_MOVEOUT) {
        auto copyOpAttribute = std::dynamic_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
        if (copyOpAttribute != nullptr) {
            copyOpAttribute->SetFromDynValidShape(OpImmediate::Specified(outValidShapes[0]));
            std::vector<SymbolicScalar> outDynShape = op->GetOOperands()[0]->GetDynValidShape();
            if (!outDynShape.empty()) {
                auto dynOffset = copyOpAttribute->GetToOffset();
                std::vector<SymbolicScalar> outShape;
                for (size_t i = 0U; i < dynOffset.size(); i++) {
                    SymbolicScalar actualDim = std::max(
                        outDynShape[i],
                        (outValidShapes[0][i] + dynOffset[i].GetSpecifiedValue()) * (outValidShapes[0][i] != 0));
                    outShape.push_back(actualDim);
                }
                outValidShapes[0] = outShape;
            }
        }
    }
}

REGISTER_INFER_SHAPE_FUNC(OP_TRANSPOSE_VNCHWCONV, Opcode::OP_TRANSPOSE_VNCHWCONV, TransposeInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_TRANSPOSE_MOVEIN, Opcode::OP_TRANSPOSE_MOVEIN, TransposeInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_TRANSPOSE_MOVEOUT, Opcode::OP_TRANSPOSE_MOVEOUT, TransposeInferFunc);

void PermuteInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    std::vector<SymbolicScalar> validShape;
    const auto& oOperands = op->GetOOperands();

    std::vector<SymbolicScalar> inputValidShape;
    if (!op->GetIOperands().empty()) {
        inputValidShape = op->GetIOperands()[0]->GetDynValidShape();
    }
    if (inputValidShape.empty()) {
        return;
    }
    std::vector<int> perm = op->GetVectorIntAttribute<int>(OpAttributeKey::perm);
    std::vector<SymbolicScalar> resultValidShape;
    resultValidShape.reserve(perm.size());
    for (int axis : perm) {
        resultValidShape.push_back(inputValidShape[axis]);
    }
    if (op->GetAttr(OP_ATTR_PREFIX + "validShape", validShape) && !validShape.empty()) {
        outValidShapes.push_back(validShape);
    } else if (!resultValidShape.empty()) {
        outValidShapes.push_back(resultValidShape);
    }
    for (size_t idx = 1; idx < oOperands.size(); ++idx) {
        auto outputValidShape = oOperands[idx]->GetDynValidShape();
        if (outputValidShape.empty()) {
            outputValidShape = resultValidShape;
        }
        outValidShapes.push_back(outputValidShape);
    }
}

REGISTER_INFER_SHAPE_FUNC(OP_PERMUTE, Opcode::OP_PERMUTE, PermuteInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_PERMUTE_ELEMENT, Opcode::OP_PERMUTE_ELEMENT, PermuteInferFunc);

void ViewInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    auto viewOpAttribute = std::dynamic_pointer_cast<ViewOpAttribute>(op->GetOpAttribute());
    if (viewOpAttribute == nullptr) {
        VECTOR_LOGW("View [%d] has no view attr.", op->GetOpMagic());
        outValidShapes.push_back(op->GetIOperands()[0]->GetDynValidShape());
        return;
    }
    // view的toDynValidShape是前端已经预设好，直接使用即可
    auto toValidShape = viewOpAttribute->GetToDynValidShape();
    if (!toValidShape.empty()) {
        for (auto output : op->GetOOperands()) {
            outValidShapes.push_back(toValidShape);
        }
    } else {
        auto inputValidShape = op->GetIOperands()[0]->GetDynValidShape();
        if (inputValidShape.empty()) {
            auto shapeImm = OpImmediate::Specified(op->GetIOperands()[0]->GetShape());
            inputValidShape.resize(shapeImm.size());
            OpImmediate::NormalizeValue(inputValidShape, 0, shapeImm, 0, false);
        }

        auto newDynValidShape = GetViewValidShape(inputValidShape, viewOpAttribute->GetFromOffset(),
                                                  viewOpAttribute->GetFromDynOffset(), op->GetOOperands()[0]->oriShape);

        for (auto output : op->GetOOperands()) {
            outValidShapes.push_back(newDynValidShape);
        }
        viewOpAttribute->SetToDynValidShape(newDynValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_VIEW, Opcode::OP_VIEW, ViewInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_SLICE, Opcode::OP_SLICE, ViewInferFunc);

void AssembleInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    if (op == nullptr || op->GetIOperands().empty() || op->GetOOperands().empty()) {
        VECTOR_LOGW("AssembleInferFunc: op or its operands are invalid, skip infer shape.");
        return;
    }
    auto assembleOpAttribute = std::dynamic_pointer_cast<AssembleOpAttribute>(op->GetOpAttribute());
    if (assembleOpAttribute != nullptr) {
        auto fromValidShape = op->GetIOperands()[0]->GetDynValidShape();
        assembleOpAttribute->SetFromDynValidShape(fromValidShape);
    } else {
        VECTOR_LOGW("Copyout [%d] has no copy out attr.", op->GetOpMagic());
        outValidShapes.push_back(op->GetIOperands()[0]->GetDynValidShape());
        return;
    }
    auto offset = assembleOpAttribute->GetToOffset();
    auto inputShapes = op->GetIOperands()[0]->GetDynValidShape();
    std::vector<SymbolicScalar> outDynShape = op->GetOOperands()[0]->GetDynValidShape();
    if (outDynShape.empty()) {
        for (size_t i = 0; i < op->GetOOperands()[0]->GetShape().size(); ++i) {
            outDynShape.push_back(SymbolicScalar(0));
        }
    }
    std::vector<SymbolicScalar> outShape;
    for (size_t i = 0U; i < inputShapes.size(); i++) {
        SymbolicScalar actualDim;
        // offset/outDynShape 维度可能小于 inputShapes（pass 改写后畸形 op），越界时按 0 处理避免崩溃
        int64_t curOffset = (i < offset.size()) ? offset[i] : 0;
        SymbolicScalar curOutDim = (i < outDynShape.size()) ? outDynShape[i] : SymbolicScalar(0);
        if (curOffset == 0) {
            actualDim = std::max(curOutDim, inputShapes[i]);
        } else {
            actualDim = std::max(curOutDim, (inputShapes[i] + curOffset) * (inputShapes[i] != 0));
        }
        outShape.push_back(actualDim);
    }
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_ASSEMBLE, Opcode::OP_ASSEMBLE, AssembleInferFunc);
REGISTER_INFER_SHAPE_FUNC(OP_CONTRACT, Opcode::OP_CONTRACT, AssembleInferFunc);

const std::string TOPK_AXIS = OP_ATTR_PREFIX + "axis";
const std::string TOPK_ORDER = OP_ATTR_PREFIX + "order";
const std::string TOPK_ALGO = OP_ATTR_PREFIX + "algo";
const std::string TOPK_KVALUE = OP_ATTR_PREFIX + "kvalue";
const std::string EXTRACT_MASKMODE = OP_ATTR_PREFIX + "makeMode";
const std::string SORT_AXIS = OP_ATTR_PREFIX + "axis";
constexpr int32_t blockSize = NUM_VALUE_32;
constexpr int32_t kFactorSize = NUM_VALUE_4;
constexpr int32_t kBlockFpNum = NUM_VALUE_8;

// m,n -> m,4*n align32
void BitSortFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    std::vector<std::vector<SymbolicScalar>> inputValidShapes;
    for (auto inputTensor : op->GetIOperands()) {
        inputValidShapes.push_back(inputTensor->GetDynValidShape());
    }
    if (inputValidShapes.empty()) {
        return;
    }
    std::vector<SymbolicScalar> res(inputValidShapes[0]);
    auto topk_axis = op->GetIntAttribute(TOPK_AXIS);
    res[topk_axis] = res[topk_axis] * NUM2;
    outValidShapes.emplace_back(res);
    if (inputValidShapes[0].size() == 1) {
        outValidShapes.emplace_back(std::vector<SymbolicScalar>{res[topk_axis]});
    } else {
        outValidShapes.emplace_back(std::vector<SymbolicScalar>{1, res[topk_axis]});
    }
}

REGISTER_INFER_SHAPE_FUNC(OP_BITSORT, Opcode::OP_BITSORT, BitSortFunc);

// m,4 *n align32byte -> m, 2 * k align8
void MrgSortFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    std::vector<std::vector<SymbolicScalar>> inputValidShapes;
    for (auto inputTensor : op->GetIOperands()) {
        inputValidShapes.push_back(inputTensor->GetDynValidShape());
    }
    if (inputValidShapes.empty()) {
        return;
    }
    std::vector<SymbolicScalar> res(inputValidShapes[0]);
    auto topk_axis = op->GetIntAttribute(TOPK_AXIS);
    auto topk_kvalue = op->GetIntAttribute(TOPK_KVALUE);
    res[topk_axis] = std::min(res[topk_axis], (topk_kvalue + kBlockFpNum - 1) / kBlockFpNum * kBlockFpNum * NUM2);
    outValidShapes.emplace_back(res);
    if (inputValidShapes[0].size() == 1) {
        outValidShapes.emplace_back(std::vector<SymbolicScalar>{res[topk_axis]});
    } else {
        outValidShapes.emplace_back(std::vector<SymbolicScalar>{1, res[topk_axis]});
    }
}

REGISTER_INFER_SHAPE_FUNC(OP_MRGSORT, Opcode::OP_MRGSORT, MrgSortFunc);

void TiledMrgSortFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    std::vector<std::vector<SymbolicScalar>> inputValidShapes;
    for (auto inputTensor : op->GetIOperands()) {
        inputValidShapes.push_back(inputTensor->GetDynValidShape());
    }
    if (inputValidShapes.empty()) {
        return;
    }
    auto outValidShape = inputValidShapes[0];
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}

REGISTER_INFER_SHAPE_FUNC(OP_TILEDMRGSORT, Opcode::OP_TILEDMRGSORT, TiledMrgSortFunc);

// m, 2 * k align8 -> m, k
void ExtractFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    std::vector<std::vector<SymbolicScalar>> inputValidShapes;
    for (auto inputTensor : op->GetIOperands()) {
        inputValidShapes.push_back(inputTensor->GetDynValidShape());
    }
    if (inputValidShapes.empty()) {
        return;
    }
    std::vector<SymbolicScalar> res(inputValidShapes[0]);
    res.back() = op->GetIntAttribute(TOPK_KVALUE);
    outValidShapes.push_back(res);
}

REGISTER_INFER_SHAPE_FUNC(OP_EXTRACT, Opcode::OP_EXTRACT, ExtractFunc);

void QuantMXInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    constexpr int64_t kQuantMXGroupSize = NUM_VALUE_32;
    constexpr int64_t kQuantMXScaleGroupSize = NUM_VALUE_64;
    constexpr int64_t kQuantMXScalePairSize = NUM_VALUE_2;
    constexpr int64_t kQuantMXFp32ScalingFactor = NUM_VALUE_2;
    std::vector<std::vector<SymbolicScalar>> inputValidShapes;
    for (auto inputTensor : op->GetIOperands()) {
        inputValidShapes.push_back(inputTensor->GetDynValidShape());
    }
    if (inputValidShapes.empty()) {
        return;
    }
    const auto& inputValidShape = inputValidShapes[0];
    int64_t axis = op->GetIntAttribute(OpAttributeKey::mxQuantAxis);
    if (axis < 0) {
        axis += static_cast<int64_t>(inputValidShape.size());
    }
    const bool isDnAxis = inputValidShape.size() >= NUM_VALUE_2 &&
                          axis == static_cast<int64_t>(inputValidShape.size()) - NUM_VALUE_2;
    if (isDnAxis) {
        auto groupedValidShape = inputValidShape;
        groupedValidShape[groupedValidShape.size() -
                          NUM_VALUE_2] = groupedValidShape[groupedValidShape.size() - NUM_VALUE_2] / kQuantMXGroupSize;
        auto expValidShape = inputValidShape;
        expValidShape[expValidShape.size() - NUM_VALUE_2] = expValidShape[expValidShape.size() - NUM_VALUE_2] /
                                                            kQuantMXScaleGroupSize;
        expValidShape.back() = expValidShape.back() * kQuantMXScalePairSize;
        outValidShapes.push_back(inputValidShape);
        outValidShapes.push_back(expValidShape);
        outValidShapes.push_back(groupedValidShape);
        outValidShapes.push_back(groupedValidShape);
        return;
    }
    std::vector<SymbolicScalar> groupedValidShape;
    if (inputValidShape.size() == 1) {
        groupedValidShape = {(inputValidShape[0] + kQuantMXGroupSize - 1) / kQuantMXGroupSize};
    } else {
        groupedValidShape.reserve(inputValidShape.size() - 1);
        for (size_t i = 0; i + NUM_VALUE_2 < inputValidShape.size(); ++i) {
            groupedValidShape.push_back(inputValidShape[i]);
        }
        groupedValidShape.push_back(inputValidShape[inputValidShape.size() - NUM_VALUE_2] *
                                    ((inputValidShape.back() + kQuantMXGroupSize - 1) / kQuantMXGroupSize));
    }
    std::vector<SymbolicScalar> expValidShape = groupedValidShape;
    if (op->GetIntAttribute(OpAttributeKey::mxQuantPerformanceMode) == 0) {
        expValidShape = inputValidShape;
        expValidShape.back() = (expValidShape.back() + kQuantMXGroupSize - 1) / kQuantMXGroupSize;
    }
    std::vector<SymbolicScalar> scalingValidShape = groupedValidShape;
    if (op->GetIOperands()[0]->Datatype() == DataType::DT_FP32) {
        scalingValidShape.back() = scalingValidShape.back() * kQuantMXFp32ScalingFactor;
    }
    outValidShapes.push_back(inputValidShape);
    outValidShapes.push_back(expValidShape);
    outValidShapes.push_back(groupedValidShape);
    outValidShapes.push_back(scalingValidShape);
}

REGISTER_INFER_SHAPE_FUNC(OP_QUANT_MX, Opcode::OP_QUANT_MX, QuantMXInferFunc);

void RadixSelectFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    auto input = op->GetIOperands()[0];
    std::vector<SymbolicScalar> res(input->GetDynValidShape());
    res.back() = op->GetIntAttribute(TOPK_KVALUE);
    outValidShapes.push_back(res);
    outValidShapes.push_back(res);
    auto output = op->GetOOperands()[NUM_VALUE_2];
    if (output->GetDynValidShape().empty()) {
        auto immShape = OpImmediate::Specified(output->GetShape());
        std::vector<SymbolicScalar> validShape;
        validShape.push_back(immShape[0].GetSpecifiedValue());
        outValidShapes.push_back(validShape);
    } else {
        outValidShapes.push_back(output->GetDynValidShape());
    }
}

REGISTER_INFER_SHAPE_FUNC(OP_RADIX_SELECT, Opcode::OP_RADIX_SELECT, RadixSelectFunc);

void VecDupInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& validShapes)
{
    std::vector<SymbolicScalar> validShape;
    op->GetAttr(OP_ATTR_PREFIX + "validShape", validShape);
    validShapes.push_back(validShape);
}
REGISTER_INFER_SHAPE_FUNC(OP_VEC_DUP, Opcode::OP_VEC_DUP, VecDupInferFunc);

void ReshapeInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& validShapes)
{
    for (const auto& output : op->GetOOperands()) {
        std::vector<SymbolicScalar> validShape;
        if (op->GetAttr(OP_ATTR_PREFIX + "validShape", validShape) && !validShape.empty()) {
            validShapes.push_back(validShape);
            continue;
        }
        const auto& dynValid = output->GetDynValidShape();
        if (!dynValid.empty()) {
            validShapes.push_back(dynValid);
        } else {
            validShapes.push_back(SymbolicScalar::FromConcrete(output->GetShape()));
        }
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_RESHAPE, Opcode::OP_RESHAPE, ReshapeInferFunc);

void BrcbInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    auto dimSize = op->GetIOperands()[0]->GetDynValidShape().size();
    std::vector<SymbolicScalar> outValidShape;
    for (size_t i = 0; i < dimSize - 1; i++) {
        outValidShape.push_back(op->GetIOperands()[0]->GetDynValidShape()[i]);
    }
    int64_t lastDimShape = blockSize / BytesOf(op->GetIOperands()[0]->Datatype());
    outValidShape.push_back(lastDimShape);
    for (auto output : op->GetOOperands()) {
        outValidShapes.push_back(outValidShape);
    }
}
REGISTER_INFER_SHAPE_FUNC(OP_BRCB, Opcode::OP_BRCB, BrcbInferFunc);

void TwoTileMrgSortFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    std::vector<std::vector<SymbolicScalar>> inputValidShapes;
    for (auto inputTensor : op->GetIOperands()) {
        inputValidShapes.push_back(inputTensor->GetDynValidShape());
    }
    if (inputValidShapes.empty()) {
        return;
    }

    std::vector<SymbolicScalar> res(inputValidShapes[0]);
    outValidShapes.push_back(res);
}
REGISTER_INFER_SHAPE_FUNC(OP_TWOTILEMRGSORT, Opcode::OP_TWOTILEMRGSORT, TwoTileMrgSortFunc);

void ExtractSingleFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    std::vector<std::vector<SymbolicScalar>> inputValidShapes;
    for (auto inputTensor : op->GetIOperands()) {
        inputValidShapes.push_back(inputTensor->GetDynValidShape());
    }
    if (inputValidShapes.empty()) {
        return;
    }
    std::vector<SymbolicScalar> res(inputValidShapes[0]);
    res.back() = res.back() / NUM_VALUE_2;
    outValidShapes.push_back(res);
}
REGISTER_INFER_SHAPE_FUNC(OP_EXTRACT_SINGLE, Opcode::OP_EXTRACT_SINGLE, ExtractSingleFunc);

void PReLUInferFunc(Operation* op, std::vector<std::vector<SymbolicScalar>>& outValidShapes)
{
    ASSERT(VectorErrorCode::ERR_PARAM_COUNT_INVALID, op->GetIOperands().size() == NUM_VALUE_2)
        << "PReLU input operand size should be 2";
    ASSERT(VectorErrorCode::ERR_PARAM_COUNT_INVALID, op->GetOOperands().size() == NUM_VALUE_2)
        << "PReLU output operand size should be 2";

    auto input0 = op->GetIOperands()[0];

    std::vector<SymbolicScalar> output0ValidShape = input0->GetDynValidShape();

    std::vector<SymbolicScalar> output1ValidShape;
    auto input0ShapeDim = input0->GetDynValidShape().size();

    if (input0ShapeDim == NUM_VALUE_2) {
        output1ValidShape.emplace_back(input0->GetDynValidShape().back());
    } else {
        constexpr int64_t ALIGN_SIZE = NUM_VALUE_32;
        int64_t elementCount = ALIGN_SIZE / BytesOf(input0->Datatype());
        output1ValidShape.emplace_back(elementCount);
    }

    outValidShapes.emplace_back(std::move(output0ValidShape));
    outValidShapes.emplace_back(std::move(output1ValidShape));
}
REGISTER_INFER_SHAPE_FUNC(OP_PRELU, Opcode::OP_PRELU, PReLUInferFunc);
} // namespace npu::tile_fwk
