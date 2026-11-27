/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file Kirin9050CostCoeffs.h
 * \brief Kirin9050 cost rules (expression + coefficients).
 */

#pragma once

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "cost_model/simulation/common/ISA.h"
#include "cost_model/simulation/arch/PipeMachineImpl.h"
#include "cost_model/simulation/expr/CostExpr.h"

namespace CostModel {

inline int KirinShapeProduct(const std::vector<int>& shape, size_t begin, size_t end, int fill)
{
    int product = 1;
    end = std::min(end, shape.size());
    for (size_t i = begin; i < end; ++i) {
        product *= shape[i] > 0 ? shape[i] : fill;
    }
    return product;
}

inline int KirinMinShapeProduct(const std::vector<std::vector<int>>& shape, size_t begin, size_t end, int fill,
                                size_t count = 0)
{
    if (count == 0 || count > shape.size()) {
        count = shape.size();
    }
    int minProduct = INT_MAX;
    for (size_t i = 0; i < count; ++i) {
        minProduct = std::min(minProduct, KirinShapeProduct(shape[i], begin, end, fill));
    }
    return minProduct == INT_MAX ? 0 : minProduct;
}

struct RuleInput {
    const TileOpPtr* tileOp = nullptr;
    const std::vector<std::vector<int>>* inShapes = nullptr;
    const std::vector<std::vector<int>>* outShapes = nullptr;
    std::vector<DataType> inDtypes;
    std::vector<DataType> outDtypes;
};

inline bool IsCopyInOpcode(const std::string& opcode)
{
    return opcode == "COPY_IN" || opcode == "UB_COPY_IN" || opcode == "L1_COPY_IN";
}

inline bool IsIndexPutCopyIn(const TileOp& tileOp)
{
    if (tileOp.funcPtr == nullptr || !IsCopyInOpcode(tileOp.opcode)) {
        return false;
    }
    const auto& ops = tileOp.funcPtr->tileOps;
    bool found = false;
    for (const auto& op : ops) {
        if (op == nullptr) {
            continue;
        }
        if (!found) {
            found = op->magic == tileOp.magic;
            continue;
        }
        if (IsCopyInOpcode(op->opcode)) {
            continue;
        }
        return op->opcode == "INDEX_PUT";
    }
    return false;
}

inline bool SetCommonCtx(FeatureContext& ctx, const RuleInput& input)
{
    if (input.inShapes == nullptr || input.inShapes->empty() || input.inShapes->front().empty()) {
        return false;
    }
    std::vector<std::vector<int>> shape = *input.inShapes;
    if (input.outShapes != nullptr) {
        shape.insert(shape.end(), input.outShapes->begin(), input.outShapes->end());
    }
    const size_t inputCount = input.inShapes->size();
    const DataType dataType = !input.inDtypes.empty() ?
                                  input.inDtypes[0] :
                                  (!input.outDtypes.empty() ? input.outDtypes[0] : DataType::DT_FP16);
    bool copyInOp = false;
    bool broadcastOp = false;
    bool broadcastW = false;
    bool broadcastH = false;
    if (input.tileOp != nullptr && *input.tileOp != nullptr) {
        const std::string& opcode = (*input.tileOp)->opcode;
        copyInOp = opcode == "COPY_IN" || opcode == "UB_COPY_IN" || opcode == "L1_COPY_IN";
        broadcastOp = opcode == "ADD" || opcode == "SUB" || opcode == "MUL" || opcode == "DIV" || opcode == "MAXIMUM" ||
                      opcode == "MINIMUM" || opcode == "CMP" || opcode == "CMPS";
    }
    if (broadcastOp && input.inShapes->size() >= 2 && input.outShapes != nullptr && !input.outShapes->empty()) {
        const auto& b = (*input.inShapes)[1];
        const auto& out = input.outShapes->front();
        if (b.size() >= 2 && out.size() >= 2) {
            broadcastW = b.back() == out.back() && b[b.size() - 2] == 1 && out[out.size() - 2] > 1;
            broadcastH = b[b.size() - 2] == out[out.size() - 2] && b.back() == 1 && out.back() > 1;
        }
    }
    const int inputSize = KirinMinShapeProduct(shape, 0, shape[0].size(), 1, inputCount);
    int outputSize = 0;
    if (input.outShapes != nullptr && !input.outShapes->empty()) {
        const auto& logicalOut = input.outShapes->front();
        if (!logicalOut.empty()) {
            outputSize = KirinShapeProduct(logicalOut, 0, logicalOut.size(), 1);
        }
    }
    const int totalSize = copyInOp ? KirinMinShapeProduct(shape, 0, shape[0].size(), 1) :
                                     (broadcastOp ? std::max(inputSize, outputSize) : inputSize);
    const size_t dtypeSize = BytesOf(dataType);
    const double dtypeBytes = static_cast<double>(dtypeSize);
    int dtypeId = 0;
    switch (dataType) {
        case DataType::DT_FP16:
            dtypeId = 0;
            break;
        case DataType::DT_FP32:
            dtypeId = 1;
            break;
        case DataType::DT_INT16:
            dtypeId = 2;
            break;
        case DataType::DT_INT32:
            dtypeId = 3;
            break;
        default:
            break;
    }
    const int cols = shape[0].back() > 0 ? shape[0].back() : 1;
    const double elems = static_cast<double>(totalSize);
    const double bytes = dtypeBytes * elems;
    const bool indexPutCopyIn = copyInOp && input.tileOp != nullptr && *input.tileOp != nullptr &&
                                IsIndexPutCopyIn(**input.tileOp);
    const double rows = elems / static_cast<double>(cols);
    const double r64 = std::ceil(bytes / 64.0);
    ctx.Set("BYTES", bytes);
    ctx.Set("R64", r64);
    ctx.Set("R256", std::ceil(bytes / 256.0));
    ctx.Set("DTYPE_BYTES", dtypeBytes);
    ctx.Set("INPUT_DTYPE", static_cast<double>(dtypeId));
    ctx.Set("EPR64", 64.0 / dtypeBytes);
    ctx.Set("EPR64_F32", 16.0);
    ctx.Set("IS_F16", dtypeSize == 2 ? 1.0 : 0.0);
    ctx.Set("IS_INT32", dataType == DataType::DT_INT32 ? 1.0 : 0.0);
    ctx.Set("COLS", static_cast<double>(cols));
    ctx.Set("ROWS", rows);
    ctx.Set("BCAST_W", broadcastW ? 1.0 : 0.0);
    ctx.Set("BCAST_H", broadcastH ? 1.0 : 0.0);
    ctx.Set("COPY_FLOOR", indexPutCopyIn ? 186.0 : 0.0);
    double outerLoop = 1.0;
    if (shape[0].size() > 2) {
        outerLoop = static_cast<double>(KirinShapeProduct(shape[0], 0, shape[0].size() - 2, 1));
    }
    ctx.Set("OUTER_LOOP", outerLoop);
    ctx.Set("ROWS_PER_OUTER", rows / outerLoop);
    ctx.Set("R64_PER_OUTER", std::ceil(r64 / outerLoop));
    return true;
}

inline bool SetWhereCtx(FeatureContext& ctx, const RuleInput& input)
{
    if (input.tileOp == nullptr || *input.tileOp == nullptr || input.inShapes == nullptr || input.inShapes->empty()) {
        return false;
    }
    const DataType dtype = !input.outDtypes.empty() ? input.outDtypes[0] :
                                                      (!input.inDtypes.empty() ? input.inDtypes[0] : DataType::DT_FP16);
    RuleInput commonInput = input;
    if (!input.outDtypes.empty()) {
        commonInput.inDtypes = input.outDtypes;
    }
    if (!SetCommonCtx(ctx, commonInput)) {
        return false;
    }
    const auto& inShape = input.inShapes->front();
    const auto& outShape = (input.outShapes != nullptr && !input.outShapes->empty()) ? input.outShapes->front() :
                                                                                       inShape;
    const double dtypeBytes = static_cast<double>(BytesOf(dtype));
    const int inElems = KirinShapeProduct(inShape, 0, inShape.size(), 1);
    const int outElems = KirinShapeProduct(outShape, 0, outShape.size(), 1);
    const double whCols = (!outShape.empty() && outShape.back() > 0) ? outShape.back() : 0.0;
    const double whRows = (whCols > 0.0) ? static_cast<double>(outElems) / whCols : 0.0;
    const double whFull = std::floor(whCols / 1024.0);
    const double whTail = whCols - whFull * 1024.0;
    const double whRTail = std::ceil(whTail * dtypeBytes / 64.0);
    ctx.Set("WH_ROWS", whRows);
    ctx.Set("WH_FULL", whFull);
    ctx.Set("WH_TAIL", whTail);
    ctx.Set("WH_R_TAIL", whRTail);
    ctx.Set("WHERE_BOOL", (inElems >= outElems) ? 1.0 : 0.0);
    return true;
}

inline bool SetCmpCtx(FeatureContext& ctx, const RuleInput& input)
{
    if (input.tileOp == nullptr || *input.tileOp == nullptr || input.inShapes == nullptr || input.inShapes->empty()) {
        return false;
    }
    if (!SetCommonCtx(ctx, input)) {
        return false;
    }
    const auto& inShape = input.inShapes->front();
    const auto& outShape = (input.outShapes != nullptr && !input.outShapes->empty()) ? input.outShapes->back() :
                                                                                       inShape;
    const int inElems = KirinShapeProduct(inShape, 0, inShape.size(), 1);
    const int outElems = KirinShapeProduct(outShape, 0, outShape.size(), 1);
    ctx.Set("IS_BIT_MODE", (outElems * 8 <= inElems) ? 1.0 : 0.0);
    return true;
}

inline bool SetConcatCtx(FeatureContext& ctx, const RuleInput& input)
{
    if (input.tileOp == nullptr || *input.tileOp == nullptr || input.inShapes == nullptr || input.inShapes->empty()) {
        return false;
    }
    if (!SetCommonCtx(ctx, input)) {
        return false;
    }
    const auto& inShapes = *input.inShapes;
    const auto& outShape = (input.outShapes != nullptr && !input.outShapes->empty()) ? input.outShapes->back() :
                                                                                       inShapes.back();
    ctx.Set("COLS0", (!inShapes[0].empty() && inShapes[0].back() > 0) ? static_cast<double>(inShapes[0].back()) : 0.0);
    const double cols1 = (inShapes.size() > 1 && !inShapes[1].empty()) ? static_cast<double>(inShapes[1].back()) : 0.0;
    ctx.Set("COLS1", cols1);
    double outRows = 1.0;
    for (size_t i = 0; i + 1 < outShape.size(); ++i) {
        outRows *= outShape[i] > 0 ? outShape[i] : 1;
    }
    ctx.Set("OUT_ROWS", outRows);
    return true;
}

inline bool SetIndexPutCtx(FeatureContext& ctx, const RuleInput& input)
{
    if (input.tileOp == nullptr || *input.tileOp == nullptr) {
        return false;
    }
    const TileOpPtr& tileOp = *input.tileOp;
    const auto shape = [&tileOp]() {
        std::vector<std::vector<int>> value;
        for (const auto& tile : tileOp->iOperand) {
            value.emplace_back(tile->shape);
        }
        for (const auto& tile : tileOp->oOperand) {
            value.emplace_back(tile->shape);
        }
        return value;
    }();
    const size_t inputCount = tileOp->iOperand.size();
    if (shape.empty() || shape[0].empty() || shape.size() < 2 || inputCount < 2) {
        SetCommonCtx(ctx, input);
        return false;
    }
    if (!SetCommonCtx(ctx, input)) {
        return false;
    }
    const size_t count = std::min(inputCount, shape.size());
    if (count < 2) {
        return false;
    }
    double indices = 0.0;
    bool hasIndices = false;
    size_t valuesIndex = 1;
    double valuesElements = 0.0;
    size_t valuesRank = 0;
    for (size_t i = 1; i < count; ++i) {
        const double product = static_cast<double>(KirinShapeProduct(shape[i], 0, shape[i].size(), 1));
        if (!hasIndices || product < indices) {
            indices = product;
            hasIndices = true;
        }
        const size_t rank = shape[i].size();
        if (rank > valuesRank || (rank == valuesRank && product > valuesElements)) {
            valuesIndex = i;
            valuesElements = product;
            valuesRank = rank;
        }
    }
    if (valuesIndex >= inputCount || valuesIndex >= shape.size()) {
        return false;
    }
    const TilePtr& valuesTile = tileOp->iOperand[valuesIndex];
    const DataType dtype = valuesTile != nullptr ? valuesTile->dataType : DataType::DT_FP16;
    const auto& valuesShape = shape[valuesIndex];
    const double dtypeBytes = static_cast<double>(BytesOf(dtype));
    double bursts = 1.0;
    double burstBytes = dtypeBytes;
    double blockFlag = 0.0;
    if (valuesShape.size() == 2) {
        burstBytes = static_cast<double>(valuesShape[1]) * dtypeBytes;
    } else if (valuesShape.size() == 3) {
        bursts = static_cast<double>(valuesShape[1]);
        burstBytes = static_cast<double>(valuesShape[2]) * dtypeBytes;
    } else if (valuesShape.size() >= 4) {
        double trailingElements = 1.0;
        for (size_t i = 1; i < valuesShape.size(); ++i) {
            trailingElements *= static_cast<double>(valuesShape[i] > 0 ? valuesShape[i] : 1);
        }
        burstBytes = trailingElements * dtypeBytes;
        blockFlag = 1.0;
    }
    ctx.Set("INDICES", indices);
    ctx.Set("INDEX_DIM", count >= 2 ? static_cast<double>(count - 2) : 1.0);
    ctx.Set("INDEX_BURSTS", bursts);
    ctx.Set("INDEX_BURST_BYTES", burstBytes);
    ctx.Set("INDEX_BLOCK_FLAG", blockFlag);
    return true;
}

inline bool SetVecDupCtx(FeatureContext& ctx, const RuleInput& input)
{
    if (input.tileOp == nullptr || *input.tileOp == nullptr || input.outShapes == nullptr || input.outShapes->empty() ||
        input.outShapes->front().empty()) {
        return false;
    }
    RuleInput commonInput = input;
    commonInput.inShapes = input.outShapes;
    commonInput.inDtypes = input.outDtypes;
    commonInput.outShapes = nullptr;
    commonInput.outDtypes.clear();
    return SetCommonCtx(ctx, commonInput);
}

inline bool SetCastCtx(FeatureContext& ctx, const RuleInput& input)
{
    if (input.tileOp == nullptr || *input.tileOp == nullptr) {
        return false;
    }
    const TileOpPtr& tileOp = *input.tileOp;
    if (!SetCommonCtx(ctx, input)) {
        return false;
    }
    if (tileOp->iOperand.empty() || tileOp->oOperand.empty() || tileOp->iOperand[0] == nullptr ||
        tileOp->oOperand[0] == nullptr) {
        return false;
    }
    const DataType inDtype = tileOp->iOperand[0]->dataType;
    const DataType outDtype = tileOp->oOperand[0]->dataType;
    const int inBytes = BytesOf(inDtype);
    const int outBytes = BytesOf(outDtype);
    double castType = -1.0;
    if (inBytes == 2 && outBytes == 4) {
        castType = 0.0;
    } else if (inBytes == 4 && outBytes == 2) {
        castType = 1.0;
    } else if (inBytes == 4 && outBytes == 4) {
        castType = 2.0;
    } else if (inBytes == 2 && outBytes == 2) {
        castType = 3.0;
    } else if (inBytes == 2 && outBytes == 1) {
        castType = 4.0;
    } else if (inBytes == 1 && outBytes == 2) {
        castType = 5.0;
    }
    if (castType >= 0.0) {
        ctx.Set("CAST_TYPE", castType);
    }
    uint32_t modeFlags = tileOp->opModeFlag;
    if (tileOp->operation != nullptr) {
        modeFlags = 0;
        int64_t satMode = 0;
        if (tileOp->operation->HasAttr("op_attr_satmode")) {
            satMode = tileOp->operation->GetIntAttribute("op_attr_satmode");
        }
        const bool slowPair = (inDtype == DataType::DT_FP16 && outDtype == DataType::DT_INT16) ||
                              (inDtype == DataType::DT_FP32 && outDtype == DataType::DT_INT16) ||
                              (inDtype == DataType::DT_FP16 && outDtype == DataType::DT_INT8);
        if (satMode != 0 && slowPair) {
            modeFlags |= 1u;
        }
    }
    ctx.Set("CAST_TMP", (modeFlags & 1u) ? 1.0 : 0.0);
    return true;
}

inline bool SetDivCtx(FeatureContext& ctx, const RuleInput& input)
{
    if (!SetCommonCtx(ctx, input)) {
        return false;
    }
    bool highPrecision = false;
    if (input.tileOp != nullptr && *input.tileOp != nullptr) {
        auto* operation = (*input.tileOp)->operation;
        if (operation != nullptr && operation->HasAttr(npu::tile_fwk::OpAttributeKey::precisionType)) {
            highPrecision = operation->GetIntAttribute(npu::tile_fwk::OpAttributeKey::precisionType) ==
                            static_cast<int64_t>(npu::tile_fwk::PrecisionType::HIGH_PRECISION);
        }
    }
    ctx.Set("DIV_HIGH_PRECISION", highPrecision ? 1.0 : 0.0);
    return true;
}

inline bool SetPairCtx(FeatureContext& ctx, const RuleInput& input)
{
    if (input.tileOp == nullptr || *input.tileOp == nullptr) {
        return false;
    }
    const TileOpPtr& tileOp = *input.tileOp;
    if (!SetCommonCtx(ctx, input)) {
        return false;
    }
    if (tileOp->opcode.rfind("PAIR", 0) != 0 || tileOp->iOperand.size() < 2 || tileOp->oOperand.empty()) {
        return false;
    }
    auto getShape = [](const TilePtr& tile) {
        if (tile == nullptr) {
            return std::vector<int>{};
        }
        return tile->validShape.size() == tile->shape.size() ? tile->validShape : tile->shape;
    };
    const auto a = getShape(tileOp->iOperand[0]);
    const auto b = getShape(tileOp->iOperand[1]);
    const auto d = getShape(tileOp->oOperand[0]);
    if (a.empty() || b.empty() || d.empty()) {
        return false;
    }
    auto rows = [](const std::vector<int>& value) {
        int result = 1;
        for (size_t i = 0; i + 1 < value.size(); ++i) {
            result *= value[i] > 0 ? value[i] : 1;
        }
        return result;
    };
    auto cols = [](const std::vector<int>& value) { return (!value.empty() && value.back() > 0) ? value.back() : 1; };
    ctx.Set("PAIR_H", static_cast<double>(a.size() >= 2 && a[0] > 0 ? a[0] : 1));
    const int aRows = rows(a), bRows = rows(b), dRows = rows(d);
    const int aCols = cols(a), bCols = cols(b), dCols = cols(d);
    const bool same = aRows == dRows && bRows == dRows && aCols == dCols && bCols == dCols;
    const int nSmall = std::min(aRows, bRows);
    const int nTail = std::max(dRows - nSmall, 0);
    const double epr = ctx.Get("EPR64");
    if (epr <= 0.0) {
        return false;
    }
    const int rSmall = static_cast<int>(std::ceil(std::min(aCols, bCols) / epr));
    const int rDst = static_cast<int>(std::ceil(dCols / epr));
    ctx.Set("PAIR_MODE", same ? 0.0 : 1.0);
    ctx.Set("PAIR_N_SMALL", static_cast<double>(nSmall));
    ctx.Set("PAIR_N_TAIL", static_cast<double>(nTail));
    ctx.Set("PAIR_R_SMALL", static_cast<double>(rSmall));
    ctx.Set("PAIR_R_DST", static_cast<double>(rDst));
    ctx.Set("PAIR_R_COPY", static_cast<double>(std::max(rDst - rSmall, 0)));
    return true;
}

inline bool SetMatmulEpilogueCtx(FeatureContext& ctx, const RuleInput& input)
{
    if (input.tileOp == nullptr || *input.tileOp == nullptr) {
        return false;
    }
    if (!SetCommonCtx(ctx, input)) {
        return false;
    }
    if (input.inShapes == nullptr || input.inShapes->size() < 2 || input.inShapes->front().size() < 2 ||
        (*input.inShapes)[1].empty()) {
        return false;
    }
    const auto& aShape = (*input.inShapes)[0];
    const auto& bShape = (*input.inShapes)[1];
    const int64_t m = aShape[aShape.size() - 2];
    const int64_t k = aShape[aShape.size() - 1];
    const int64_t n = bShape.back();
    int64_t batch = 1;
    for (size_t i = 0; i + 2 < aShape.size(); ++i) {
        batch *= aShape[i] > 0 ? aShape[i] : 1;
    }
    const int cubeM = 16;
    const int cubeN = 16;
    const DataType dtype = !input.inDtypes.empty() ? input.inDtypes[0] : DataType::DT_FP16;
    const int cubeK = dtype == DataType::DT_INT8 ? 32 : 8;
    ctx.Set("BATCH", static_cast<double>(batch));
    ctx.Set("TILE_M", static_cast<double>(m));
    ctx.Set("TILE_N", static_cast<double>(n));
    ctx.Set("TILE_K", static_cast<double>(k));
    ctx.Set("CUBE_M", static_cast<double>(cubeM));
    ctx.Set("CUBE_N", static_cast<double>(cubeN));
    ctx.Set("CUBE_K", static_cast<double>(cubeK));
    ctx.Set("DTYPE_INDEX", dtype == DataType::DT_INT8 ? 0.0 : 1.0);
    return true;
}

using RuleSetupCtx = bool (*)(FeatureContext&, const RuleInput&);

struct CostRule {
    std::string expr;
    std::vector<double> coeffs;
    RuleSetupCtx setupCtx;
};

namespace Kirin9050 {

inline const std::unordered_map<std::string, CostRule>& TileOpRules()
{
    static const std::unordered_map<std::string, CostRule> rules = {
        {"ABS",
         {"("
          "floor(16.9 + (R64 - 1) * 0.9987 + 0.5)"
          ")",
          {},
          &SetCommonCtx}},
        {"MAXIMUM",
         {"("
          "floor(0.9940 * (R64 - 1) + 16.0 + BCAST_W * 10.0 * ROWS + BCAST_H * 3.0 * ROWS + 0.5)"
          ")",
          {},
          &SetCommonCtx}},
        {"MINIMUM",
         {"("
          "floor(0.9873 * (R64 - 1) + 16.0 + BCAST_W * 10.0 * ROWS + BCAST_H * 3.0 * ROWS + 0.5)"
          ")",
          {},
          &SetCommonCtx}},
        // 64B repeat throughput rule.
        {"MUL",
         {"("
          "floor(1.5500 * (R64 - 1) + 18.0 + BCAST_W * 10.0 * ROWS + BCAST_H * 3.0 * ROWS + 0.5)"
          ")",
          {},
          &SetCommonCtx}},
        {"ADD",
         {"("
          "floor(1.0750 * (R64 - 1) + 17.0 + BCAST_W * 10.0 * ROWS + BCAST_H * 3.0 * ROWS + 0.5)"
          ")",
          {},
          &SetCommonCtx}},
        {"SUB",
         {"("
          "floor(1.0750 * (R64 - 1) + 17.0 + BCAST_W * 10.0 * ROWS + BCAST_H * 3.0 * ROWS + 0.5)"
          ")",
          {},
          &SetCommonCtx}},
        // Each outer iteration executes the full inner pipeline loop.
        // Inner issue interval = max(VLD, VOP, VST); outer calls do not overlap.
        {"ADDS",
         {"("
          "floor(switch(INPUT_DTYPE, {"
          "OUTER_LOOP * (31.0 + (R64_PER_OUTER - 1) * 1.0500) - 14.0,"
          "OUTER_LOOP * (31.0 + (R64_PER_OUTER - 1) * 1.0550) - 14.0,"
          "OUTER_LOOP * (31.0 + (R64_PER_OUTER - 1) * 1.1250) - 14.0,"
          "OUTER_LOOP * (31.0 + (R64_PER_OUTER - 1) * 1.1250) - 14.0"
          "}) + 0.5)"
          ")",
          {},
          &SetCommonCtx}},
        {"SUBS",
         {"("
          "floor(switch(INPUT_DTYPE, {"
          "OUTER_LOOP * (31.0 + (R64_PER_OUTER - 1) * 1.0500) - 14.0,"
          "OUTER_LOOP * (31.0 + (R64_PER_OUTER - 1) * 1.0600) - 14.0,"
          "OUTER_LOOP * (31.0 + (R64_PER_OUTER - 1) * 1.1250) - 14.0,"
          "OUTER_LOOP * (31.0 + (R64_PER_OUTER - 1) * 1.1250) - 14.0"
          "}) + 0.5)"
          ")",
          {},
          &SetCommonCtx}},
        {"MULS",
         {"("
          "floor(switch(INPUT_DTYPE, {"
          "OUTER_LOOP * (32.0 + (R64_PER_OUTER - 1) * 1.1200) - 14.0,"
          "OUTER_LOOP * (32.0 + (R64_PER_OUTER - 1) * 1.1200) - 14.0,"
          "OUTER_LOOP * (32.0 + (R64_PER_OUTER - 1) * 1.1670) - 14.0,"
          "OUTER_LOOP * (32.0 + (R64_PER_OUTER - 1) * 1.1670) - 14.0"
          "}) + 0.5)"
          ")",
          {},
          &SetCommonCtx}},
        {"MAXS",
         {"("
          "floor(OUTER_LOOP * (29.0 + R64_PER_OUTER) - 14.0 + 0.5)"
          ")",
          {},
          &SetCommonCtx}},
        {"MINS",
         {"("
          "floor(OUTER_LOOP * (29.0 + R64_PER_OUTER) - 14.0 + 0.5)"
          ")",
          {},
          &SetCommonCtx}},
        {"DIV",
         {"("
          "floor(select(DIV_HIGH_PRECISION, "
          "select(IS_F16, x4 * R64 + x5, x6 * R64), "
          "select(IS_F16, x1, x2) * R64 + x3) + 0.5)"
          ")",
          {1.6085f, 2.0120f, 23.0f, 219.2f, 567.0f, 93.7f},
          &SetDivCtx}},
        {"DIVS",
         {"("
          "floor(switch(IS_F16, {x4, x1}) + "
          "OUTER_LOOP * (switch(IS_F16, {x5, x2}) + "
          "R64_PER_OUTER * switch(IS_F16, {x6, x3})) + 0.5)"
          ")",
          {11.16f, 35.58f, 220.0f, -15.54f, 40.14f, 2.0f},
          &SetCommonCtx}},
        {"EXP",
         {"("
          "floor(select(IS_F16, "
          "16.6 + (R64 - 1) * 1.6375, "
          "25.0 + (R64 - 1) * 2.0000) + 0.5)"
          ")",
          {},
          &SetCommonCtx}},
        {"SQRT",
         {"("
          "floor(select(IS_F16, "
          "29.0 + (R64 - 1) * 4.0000, "
          "27.0 + (R64 - 1) * 2.0000) + 0.5)"
          ")",
          {},
          &SetCommonCtx}},
        {"PAIRMAX",
         {"("
          "floor(switch(PAIR_MODE, {"
          "1.0561*R64 + 16 + 3.86*PAIR_H,"
          "24.179 + OUTER_LOOP * ("
          "PAIR_N_SMALL * (12.272 + "
          "(PAIR_R_SMALL - 1) * 1.4) + "
          "PAIR_N_SMALL * select(PAIR_R_COPY, "
          "(-6.728 + (PAIR_R_COPY - 1) * 2.981), 0) + "
          "PAIR_N_TAIL * select(PAIR_N_TAIL, "
          "(13.0 + (PAIR_R_DST - 1)), 0)"
          ")}) + 0.5)"
          ")",
          {},
          &SetPairCtx}},
        {"PAIRMIN",
         {"("
          "floor(switch(PAIR_MODE, {"
          "1.0031*R64 + 16 + 3.99*PAIR_H,"
          "24.179 + OUTER_LOOP * ("
          "PAIR_N_SMALL * (12.272 + "
          "(PAIR_R_SMALL - 1) * 1.4) + "
          "PAIR_N_SMALL * select(PAIR_R_COPY, "
          "(-6.728 + (PAIR_R_COPY - 1) * 2.981), 0) + "
          "PAIR_N_TAIL * select(PAIR_N_TAIL, "
          "(13.0 + (PAIR_R_DST - 1)), 0)"
          ")}) + 0.5)"
          ")",
          {},
          &SetPairCtx}},
        {"PAIRSUM",
         {"("
          "floor(switch(PAIR_MODE, {"
          "1.0629*R64 + 17 + 3.87*PAIR_H,"
          "24.179 + OUTER_LOOP * ("
          "PAIR_N_SMALL * (13.272 + "
          "(PAIR_R_SMALL - 1) * 1.4) + "
          "PAIR_N_SMALL * select(PAIR_R_COPY, "
          "(-6.728 + (PAIR_R_COPY - 1) * 2.981), 0) + "
          "PAIR_N_TAIL * select(PAIR_N_TAIL, "
          "(13.0 + (PAIR_R_DST - 1)), 0)"
          ")}) + 0.5)"
          ")",
          {},
          &SetPairCtx}},
        // 64B repeat rules.
        {"NEG",
         {"("
          "floor(x1 * R64 + x2 + 0.5)"
          ")",
          {1.125f, 12.0f},
          &SetCommonCtx}},
        {"RECIPROCAL",
         {"("
          "floor(x1 * R64 + x2 + 0.5)"
          ")",
          {1.0f, 12.0f},
          &SetCommonCtx}},
        {"RSQRT",
         {"("
          "floor(x1 * R64 + x2 + 0.5)"
          ")",
          {5.6375f, 12.0f},
          &SetCommonCtx}},
        {"VEC_DUP",
         {"("
          "floor(select(2 - DTYPE_BYTES, "
          "OUTER_LOOP * (16.965 + ROWS_PER_OUTER * "
          "(4.991 + (ceil(COLS / EPR64) - 1) * 0.9920)), "
          "select(4 - DTYPE_BYTES, "
          "OUTER_LOOP * (16.910 + ROWS_PER_OUTER * "
          "(4.995 + (ceil(COLS / EPR64) - 1) * 0.9900)), "
          "OUTER_LOOP * (30.482 + ROWS_PER_OUTER * "
          "(4.371 + (ceil(COLS / EPR64) - 1) * 0.9140)))) + 0.5)"
          ")",
          {},
          &SetVecDupCtx}},
        // ROWSUM: f16 uses TRowSum_Fp32Acc, int32/f32 use separate branches.
        {"ROWSUM_SINGLE",
         {"max(20, ("
          "floor(select(IS_F16, "
          "x1 + OUTER_LOOP * (x2 + ROWS_PER_OUTER * (x3 + x4 * ceil(COLS / EPR64_F32))), "
          "select(IS_INT32, "
          "x5 + OUTER_LOOP * (x6 + ROWS_PER_OUTER * (x7 + x8 * ceil(COLS / EPR64))), "
          "x9 + OUTER_LOOP * (x10 + ROWS_PER_OUTER * (x11 + x12 * ceil(COLS / EPR64))))) + 0.5)"
          "))",
          {-3.3535f, 47.8417f, 1.2502f, 3.1261f, 7.2288f, 3.0857f, 1.7316f, 3.1734f, -102.5403f, 60.1217f, 0.8975f,
           2.6481f},
          &SetCommonCtx}},
        {"ROWMAX_SINGLE",
         {"max(20, ("
          "floor(select(IS_F16, "
          "x1 + OUTER_LOOP * (x2 + ROWS_PER_OUTER * (x3 + x4 * ceil(COLS / EPR64))), "
          "x5 + OUTER_LOOP * (x6 + ROWS_PER_OUTER * (x7 + x8 * ceil(COLS / EPR64)))) + 0.5)"
          "))",
          {-13.6574f, 35.5917f, 0.0f, 2.8588f, -0.1669f, 11.0941f, 1.25f, 2.7908f},
          &SetCommonCtx}},
        {"ROWMIN_SINGLE",
         {"max(20, ("
          "floor(select(IS_F16, "
          "x1 + OUTER_LOOP * (x2 + ROWS_PER_OUTER * (x3 + x4 * ceil(COLS / EPR64))), "
          "x5 + OUTER_LOOP * (x6 + ROWS_PER_OUTER * (x7 + x8 * ceil(COLS / EPR64)))) + 0.5)"
          "))",
          {-13.2151f, 35.6533f, 0.0f, 2.8584f, -0.5644f, 11.0804f, 1.25f, 2.7913f},
          &SetCommonCtx}},
        // ROW*LINE -> TCOL*.
        {"ROWSUMLINE",
         {"max(20, ("
          "floor(OUTER_LOOP * (ceil(COLS / 16.0) * "
          "select(IS_F16, "
          "-12.1422 + 1.9020*(ROWS_PER_OUTER - 1), "
          "-58.1838 + 2.080224*(ROWS_PER_OUTER - 2) + "
          "42.3112*(1.0 + log(2, ROWS_PER_OUTER))) "
          "+ select(IS_F16, 51.7464, -20.1797)) + 0.5)"
          "))",
          {},
          &SetCommonCtx}},
        {"ROWMAXLINE",
         {"max(20, ("
          "floor(select(IS_F16, "
          "x1 + OUTER_LOOP * (x2 + R64_PER_OUTER * x3), "
          "x4 + OUTER_LOOP * (x5 + R64_PER_OUTER * x6)) + 0.5)"
          "))",
          {-45.9954f, 41.0871f, 0.96645f, -25.6077f, 35.1169f, 0.80906f},
          &SetCommonCtx}},
        {"ROWMINLINE",
         {"max(20, ("
          "floor(select(IS_F16, "
          "x1 + OUTER_LOOP * (x2 + R64_PER_OUTER * x3), "
          "x4 + OUTER_LOOP * (x5 + R64_PER_OUTER * x6)) + 0.5)"
          "))",
          {-43.0167f, 40.8539f, 0.96192f, -26.1741f, 35.2105f, 0.80960f},
          &SetCommonCtx}},
        // WHERE row/chunk features.
        {"WHERE_TT",
         {"("
          "floor(select(WHERE_BOOL, "
          "select(IS_F16, "
          "x1 + WH_ROWS * (x2 + x3 * WH_FULL + "
          "x4 * select(WH_TAIL, 1, 0) + "
          "x5 * select(WH_TAIL, max(WH_R_TAIL - 1, 0), 0)), "
          "x6 + WH_ROWS * (x7 + x8 * WH_FULL + "
          "x9 * select(WH_TAIL, 1, 0) + "
          "x10 * select(WH_TAIL, max(WH_R_TAIL - 1, 0), 0))), "
          "select(IS_F16, "
          "x11 + WH_ROWS * (x12 + x13 * WH_FULL + "
          "x14 * select(WH_TAIL, 1, 0) + "
          "x15 * select(WH_TAIL, max(WH_R_TAIL - 1, 0), 0)), "
          "x16 + WH_ROWS * (x17 + x18 * WH_FULL + "
          "x19 * select(WH_TAIL, 1, 0) + "
          "x20 * select(WH_TAIL, max(WH_R_TAIL - 1, 0), 0)))) + 0.5)"
          ")",
          {-18.68f, -1.52f, 275.81f, 88.76f, 8.79f, -22.89f, -0.38f, 403.17f, 90.69f, 4.75f,
           -23.48f, 29.71f, 64.4f,   0.33f,  2.3f,  11.27f,  1.16f,  199.21f, 47.2f,  2.03f},
          &SetWhereCtx}},
        {"WHERE_TS",
         {"("
          "floor(select(WHERE_BOOL, "
          "select(IS_F16, "
          "x1 + WH_ROWS * (x2 + x3 * WH_FULL + "
          "x4 * select(WH_TAIL, 1, 0) + "
          "x5 * select(WH_TAIL, max(WH_R_TAIL - 1, 0), 0)), "
          "x6 + WH_ROWS * (x7 + x8 * WH_FULL + "
          "x9 * select(WH_TAIL, 1, 0) + "
          "x10 * select(WH_TAIL, max(WH_R_TAIL - 1, 0), 0))), "
          "select(IS_F16, "
          "x11 + WH_ROWS * (x12 + x13 * WH_FULL + "
          "x14 * select(WH_TAIL, 1, 0) + "
          "x15 * select(WH_TAIL, max(WH_R_TAIL - 1, 0), 0)), "
          "x16 + WH_ROWS * (x17 + x18 * WH_FULL + "
          "x19 * select(WH_TAIL, 1, 0) + "
          "x20 * select(WH_TAIL, max(WH_R_TAIL - 1, 0), 0)))) + 0.5)"
          ")",
          {47.76f, 2.57f,  266.5f, 84.42f, 8.31f, 76.31f, -11.91f, 406.03f, 102.17f, 4.75f,
           45.5f,  -1.62f, 83.82f, 0.88f,  6.18f, 75.39f, -6.19f,  227.89f, 48.89f,  2.4f},
          &SetWhereCtx}},
        {"WHERE_ST",
         {"("
          "floor(select(WHERE_BOOL, "
          "select(IS_F16, "
          "x1 + WH_ROWS * (x2 + x3 * WH_FULL + "
          "x4 * select(WH_TAIL, 1, 0) + "
          "x5 * select(WH_TAIL, max(WH_R_TAIL - 1, 0), 0)), "
          "x6 + WH_ROWS * (x7 + x8 * WH_FULL + "
          "x9 * select(WH_TAIL, 1, 0) + "
          "x10 * select(WH_TAIL, max(WH_R_TAIL - 1, 0), 0))), "
          "select(IS_F16, "
          "x11 + WH_ROWS * (x12 + x13 * WH_FULL + "
          "x14 * select(WH_TAIL, 1, 0) + "
          "x15 * select(WH_TAIL, max(WH_R_TAIL - 1, 0), 0)), "
          "x16 + WH_ROWS * (x17 + x18 * WH_FULL + "
          "x19 * select(WH_TAIL, 1, 0) + "
          "x20 * select(WH_TAIL, max(WH_R_TAIL - 1, 0), 0)))) + 0.5)"
          ")",
          {49.44f, -1.78f, 267.0f, 88.76f, 8.31f, 75.03f, -12.03f, 406.43f, 102.56f, 4.5f,
           44.08f, 0.34f,  83.51f, 0.84f,  5.9f,  75.95f, -0.51f,  225.78f, 43.05f,  2.41f},
          &SetWhereCtx}},
        {"WHERE_SS",
         {"("
          "floor(select(WHERE_BOOL, "
          "select(IS_F16, "
          "x1 + WH_ROWS * (x2 + x3 * WH_FULL + "
          "x4 * select(WH_TAIL, 1, 0) + "
          "x5 * select(WH_TAIL, max(WH_R_TAIL - 1, 0), 0)), "
          "x6 + WH_ROWS * (x7 + x8 * WH_FULL + "
          "x9 * select(WH_TAIL, 1, 0) + "
          "x10 * select(WH_TAIL, max(WH_R_TAIL - 1, 0), 0))), "
          "select(IS_F16, "
          "x11 + WH_ROWS * (x12 + x13 * WH_FULL + "
          "x14 * select(WH_TAIL, 1, 0) + "
          "x15 * select(WH_TAIL, max(WH_R_TAIL - 1, 0), 0)), "
          "x16 + WH_ROWS * (x17 + x18 * WH_FULL + "
          "x19 * select(WH_TAIL, 1, 0) + "
          "x20 * select(WH_TAIL, max(WH_R_TAIL - 1, 0), 0)))) + 0.5)"
          ")",
          {87.64f, -1.24f, 277.16f, 88.26f, 9.06f, 148.83f, -14.16f, 406.57f, 104.69f, 4.5f,
           85.93f, -2.75f, 94.97f,  0.98f,  6.84f, 197.36f, -51.51f, 226.23f, 118.81f, 0.65f},
          &SetWhereCtx}},
        // CMP/CMPS: bit/byte x f16/f32 feature set; setup is SetCmpCtx.
        {"CMP",
         {"("
          "floor(select(IS_BIT_MODE, "
          "switch(IS_F16, {x5 + x6 * ROWS * ceil(COLS / EPR64) + x7 * ROWS * ceil(COLS / 1024) + "
          "x8 * COLS, "
          "x1 + x2 * ROWS * ceil(COLS / EPR64) + x3 * ROWS * ceil(COLS / 1024) + x4 * COLS}), "
          "switch(IS_F16, {x13 + x14 * ROWS * ceil(COLS / EPR64) + x15 * ROWS * ceil(COLS / 1024) + "
          "x16 * COLS, "
          "x9 + x10 * ROWS * ceil(COLS / EPR64) + x11 * ROWS * ceil(COLS / 1024) + "
          "x12 * COLS})) + 0.5)"
          ")",
          {-24.365f, 1.7012f, 26.4230f, 0.0012f, -22.753f, 2.6906f, 37.9994f, 0.0149f, -19.559f, 8.0079f, 95.9829f,
           -0.0337f, -29.868f, 6.3562f, 145.5730f, 0.2059f},
          &SetCmpCtx}},
        {"CMPS",
         {"("
          "floor(select(IS_BIT_MODE, "
          "switch(IS_F16, {x5 + x6 * ROWS * ceil(COLS / EPR64) + x7 * ROWS * ceil(COLS / 1024) + "
          "x8 * COLS, "
          "x1 + x2 * ROWS * ceil(COLS / EPR64) + x3 * ROWS * ceil(COLS / 1024) + x4 * COLS}), "
          "switch(IS_F16, {x13 + x14 * ROWS * ceil(COLS / EPR64) + x15 * ROWS * ceil(COLS / 1024) + "
          "x16 * COLS, "
          "x9 + x10 * ROWS * ceil(COLS / EPR64) + x11 * ROWS * ceil(COLS / 1024) + "
          "x12 * COLS})) + 0.5)"
          ")",
          {-13.336f, 2.0122f, 21.9191f, -0.0092f, -25.772f, 2.6160f, 35.3561f, 0.0200f, -19.717f, 8.0068f, 94.9881f,
           -0.0375f, -31.088f, 5.4472f, 145.1269f, 0.2597f},
          &SetCmpCtx}},
        {"TRANSPOSE_VNCHWCONV",
         {"("
          "floor(select(COLS - ROWS_PER_OUTER, "
          "OUTER_LOOP * (12.0 + ROWS_PER_OUTER * ceil(COLS / EPR64) * 57.5 * "
          "select(2 - DTYPE_BYTES, 0.5000, "
          "select(4 - DTYPE_BYTES, "
          "select(ROWS_PER_OUTER - 32.0 * floor(ROWS_PER_OUTER / 32.0) - 0.5, 0.3146, 0.6035), "
          "0.3271))), "
          "OUTER_LOOP * (12.0 + COLS * ceil(ROWS_PER_OUTER / EPR64) * 66.5 * "
          "select(2 - DTYPE_BYTES, 0.5000, select(4 - DTYPE_BYTES, 0.5046, 0.2599)))) + 0.5)"
          ")",
          {},
          &SetCommonCtx}},
        // CONCAT: copy + scatter per row.
        {"CONCAT",
         {"("
          "floor(x1 + x2 * OUT_ROWS * (x3 * ceil(COLS0 / EPR64) + x4 * ceil(COLS1 / EPR64)) + 0.5)"
          ")",
          {12.0f, 1.0f, 1.5f, 3.5f},
          &SetConcatCtx}},
        {"CAST",
         {"("
          "floor(switch(CAST_TYPE, {"
          "OUTER_LOOP*(27.8545 + 1.302268*(ceil((ROWS_PER_OUTER*COLS)/16.0) - 1)), "
          "OUTER_LOOP*select(CAST_TMP, "
          "36.1862 + 1.983800*(ceil((ROWS_PER_OUTER*COLS)/16.0) - 1), "
          "31.0258 + 1.070682*(ceil((ROWS_PER_OUTER*COLS)/16.0) - 1)), "
          "OUTER_LOOP*(32.2380 + 1.059780*(ceil((ROWS_PER_OUTER*COLS)/16.0) - 1)), "
          "OUTER_LOOP*select(CAST_TMP, "
          "35.9704 + 1.980400*(ceil((ROWS_PER_OUTER*COLS)/16.0) - 1), "
          "32.4749 + 1.006192*(ceil((ROWS_PER_OUTER*COLS)/32.0) - 1)), "
          "OUTER_LOOP*select(CAST_TMP, "
          "44.1560 + 2.956600*(ceil((ROWS_PER_OUTER*COLS)/32.0) - 1), "
          "32.2898 + 1.076986*(ceil((ROWS_PER_OUTER*COLS)/32.0) - 1)), "
          "OUTER_LOOP*(30.5840 + 1.195494*(ceil((ROWS_PER_OUTER*COLS)/32.0) - 1))"
          "}) + 0.5)"
          ")",
          {},
          &SetCastCtx}},
        // COPY_IN: bandwidth-only model.
        {"COPY_IN",
         {"max(("
          "floor(x1 + BYTES / x2 + 0.5)"
          "), COPY_FLOOR)",
          {12.0f, 13.141f},
          &SetCommonCtx}},
        // COPY_OUT: two-segment bandwidth.
        {"COPY_OUT",
         {"("
          "floor(select(BYTES - x3, x1 + x3 / x2 + (BYTES - x3) / x4, x1 + BYTES / x2) + 0.5)"
          ")",
          {18.0f, 92.37f, 26214.0f, 12.793270845f},
          &SetCommonCtx}},
        // Operation path remaps COPY_IN/COPY_OUT by buffer type.
        {"UB_COPY_IN",
         {"max(("
          "floor(x1 + BYTES / x2 + 0.5)"
          "), COPY_FLOOR)",
          {12.0f, 13.141f},
          &SetCommonCtx}},
        {"UB_COPY_OUT",
         {"("
          "floor(select(BYTES - x3, x1 + x3 / x2 + (BYTES - x3) / x4, x1 + BYTES / x2) + 0.5)"
          ")",
          {18.0f, 92.37f, 26214.0f, 12.793270845f},
          &SetCommonCtx}},
        {"TRANSPOSE_MOVEIN",
         {"("
          "floor(1.0750 * ROWS * ceil(COLS * DTYPE_BYTES / 64) + x2 + 0.5)"
          ")",
          {1.0f, 0.0f},
          &SetCommonCtx}},
        {"TRANSPOSE_MOVEOUT",
         {"("
          "floor(1.0750 * ROWS * ceil(COLS * DTYPE_BYTES / 128) + x2 + 0.5)"
          ")",
          {1.0f, 0.0f},
          &SetCommonCtx}},
        // INDEX_PUT: one random TSTORE plus MTE3 synchronization per index.
        {"INDEX_PUT",
         {"("
          "floor(INDICES * (x1 + x2 * (INDEX_DIM - 1) + "
          "x3 * INDEX_BURSTS * ceil(INDEX_BURST_BYTES / 128) + "
          "x4 * INDEX_BLOCK_FLAG) + 0.5)"
          ")",
          {187.0f, 0.5f, 1.0f, 17.0f},
          &SetIndexPutCtx}},
        {"L1_TO_L0B",
         {"("
          "floor(1.0750 * R256 + x2 + 0.5)"
          ")",
          {1.0f, 14.0f},
          &SetCommonCtx}},
        {"A_MUL_B_EPILOGUE",
         {"("
          "floor(x1 + BATCH * (x2 + x3 * "
          "switch(DTYPE_INDEX, {1.190741272, 1.0}) * "
          "ceil(TILE_M / CUBE_M) * ceil(TILE_N / CUBE_N) * ceil(TILE_K / CUBE_K)) + 0.5)"
          ")",
          {34.510315f, 19.032306f, 0.977401f},
          &SetMatmulEpilogueCtx}},
        {"A_MULACC_B_EPILOGUE",
         {"("
          "floor(x1 + BATCH * (x2 + x3 * "
          "switch(DTYPE_INDEX, {1.190741272, 1.0}) * "
          "ceil(TILE_M / CUBE_M) * ceil(TILE_N / CUBE_N) * ceil(TILE_K / CUBE_K)) + 0.5)"
          ")",
          {35.903529f, 18.849308f, 1.029904f},
          &SetMatmulEpilogueCtx}},
    };
    return rules;
}

inline const CostRule* FindTileOpRule(const std::string& opcode)
{
    const auto& rules = TileOpRules();
    auto iter = rules.find(opcode);
    return iter == rules.end() ? nullptr : &iter->second;
}

} // namespace Kirin9050
} // namespace CostModel
