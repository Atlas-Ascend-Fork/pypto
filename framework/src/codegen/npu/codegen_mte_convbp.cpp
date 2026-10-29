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
 * \file codegen_mte_convbp.cpp
 * \brief
 */
#include <iterator>
#include <string>

#include "codegen_op_npu.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/utils/codegen_utils.h"
#include "tilefwk/error_code.h"
#include "securec.h"

namespace npu::tile_fwk {

std::string CodeGenOpNPU::GenMemL1ToL0Load2DConvBpDx() const
{
    std::vector<std::variant<std::string, uint8_t, uint16_t, int, int64_t>> paramList;

    auto tileOpParams = GetTileOpParamsByOrder();
    paramList.insert(paramList.end(), tileOpParams.begin(), tileOpParams.end());

    int64_t hkw = 0;
    int64_t kL0Size = 0;
    int64_t nL0Size = 0;
    int64_t k0Idx = 0;
    int64_t n0Idx = 0;
    GetOpAttr(OpAttributeKey::hwk, hkw);
    GetOpAttr(OpAttributeKey::kL0Size, kL0Size);
    GetOpAttr(OpAttributeKey::nL0Size, nL0Size);
    GetOpAttr(OpAttributeKey::k0Idx, k0Idx);
    GetOpAttr(OpAttributeKey::n0Idx, n0Idx);

    paramList.emplace_back(kL0Size);
    paramList.emplace_back(nL0Size);
    paramList.emplace_back(hkw);
    paramList.emplace_back(k0Idx);
    paramList.emplace_back(n0Idx);

    std::ostringstream oss;
    oss << tileOpName;
    oss << WrapParamByParentheses(paramList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenMemL1ToL0AConvDw() const
{
    std::vector<std::variant<std::string, uint16_t, int, int64_t>> paramList;

    auto tileOpParams = GetTileOpParamsByOrder();
    paramList.insert(paramList.end(), tileOpParams.begin(), tileOpParams.end());

    int64_t postM = 0;
    int64_t postK = 0;
    GetOpAttr(OpAttributeKey::postM, postM);
    GetOpAttr(OpAttributeKey::postK, postK);
    paramList.emplace_back(postM);
    paramList.emplace_back(postK);

    std::ostringstream oss;
    oss << tileOpName;
    oss << WrapParamByParentheses(paramList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenMemL1CopyInConvBp() const
{
    auto tileOpParams = GetTileOpParamsByOrder();

    int64_t strideH = 1;
    int64_t strideW = 1;
    int64_t skipH = 0;
    int64_t skipW = 0;
    GetOpAttr(OpAttributeKey::strideH, strideH);
    GetOpAttr(OpAttributeKey::strideW, strideW);
    GetOpAttr(OpAttributeKey::skipH, skipH);
    GetOpAttr(OpAttributeKey::skipW, skipW);

    bool isConv3D = false;
    GetOpAttr(Conv::LoadStoreConvOpAttributeKey::isConv3D, isConv3D);

    auto dynOffset = GetOffsetFromAttr(ToUnderlying(MISOIdx::SRC0_IDX));
    size_t expectedDim = isConv3D ? SHAPE_DIM5 : SHAPE_DIM4;
    ASSERT(ConvCodenGenError::CODEGEN_CHECK_DIM_INVALID, dynOffset.size() == expectedDim + 1)
        << "GenMemL1CopyInConvBp offset should be " << (expectedDim + 1) << "-dim!";
    std::vector<SymbolicScalar> srcShapeVec;
    GetOpAttr(OpAttributeKey::srcGmConvValidShape, srcShapeVec);
    ASSERT(ConvCodenGenError::CODEGEN_CHECK_DIM_INVALID, srcShapeVec.size() == expectedDim + 1)
        << "GenMemL1CopyInConvBp shape should be " << (expectedDim + 1) << "-dim!";

    std::vector<std::string> offsetExpr;
    std::vector<std::string> shapeExpr;
    for (size_t i = 0; i < expectedDim; i++) {
        offsetExpr.emplace_back(ToStringHelper(dynOffset[i]));
        shapeExpr.emplace_back(ToStringHelper(srcShapeVec[i]));
    }
    if (!isConv3D) {
        offsetExpr.insert(offsetExpr.begin() + 1, "0");
        shapeExpr.insert(shapeExpr.begin() + 1, "1");
    }

    std::vector<std::string> paramList;
    paramList.emplace_back(tileOpParams[ToUnderlying(MISOIdx::DST_IDX)]);
    paramList.emplace_back(tileOpParams[ToUnderlying(MISOIdx::SRC0_IDX)]);
    paramList.insert(paramList.end(), offsetExpr.begin(), offsetExpr.end());
    paramList.insert(paramList.end(), shapeExpr.begin(), shapeExpr.end());
    paramList.emplace_back(std::to_string(strideH));
    paramList.emplace_back(std::to_string(strideW));
    paramList.emplace_back(std::to_string(skipH));
    paramList.emplace_back(std::to_string(skipW));

    std::ostringstream oss;
    oss << tileOpName << WrapParamByAngleBrackets({std::to_string(isConv3D)});
    oss << WrapParamByParentheses(paramList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenMemL1CopyInConvBpDwDy() const
{
    auto tileOpParams = GetTileOpParamsByOrder();

    bool isConv3D = false;
    GetOpAttr(Conv::LoadStoreConvOpAttributeKey::isConv3D, isConv3D);

    auto dynOffset = GetOffsetFromAttr(ToUnderlying(MISOIdx::SRC0_IDX));
    size_t expectedDim = isConv3D ? SHAPE_DIM5 : SHAPE_DIM4;
    ASSERT(ConvCodenGenError::CODEGEN_CHECK_DIM_INVALID, dynOffset.size() == expectedDim + 1)
        << "GenMemL1CopyInConvBpDwDy offset should be " << (expectedDim + 1) << "-dim!";
    std::vector<SymbolicScalar> srcShapeVec;
    GetOpAttr(OpAttributeKey::srcGmConvValidShape, srcShapeVec);
    ASSERT(ConvCodenGenError::CODEGEN_CHECK_DIM_INVALID, srcShapeVec.size() == expectedDim + 1)
        << "GenMemL1CopyInConvBpDwDy shape should be " << (expectedDim + 1) << "-dim!";

    std::vector<std::string> offsetExpr;
    std::vector<std::string> shapeExpr;
    for (size_t i = 0; i < expectedDim; i++) {
        offsetExpr.emplace_back(ToStringHelper(dynOffset[i]));
        shapeExpr.emplace_back(ToStringHelper(srcShapeVec[i]));
    }
    if (!isConv3D) {
        offsetExpr.insert(offsetExpr.begin() + 1, "0");
        shapeExpr.insert(shapeExpr.begin() + 1, "1");
    }

    std::vector<std::string> paramList;
    paramList.emplace_back(tileOpParams[ToUnderlying(MISOIdx::DST_IDX)]);
    paramList.emplace_back(tileOpParams[ToUnderlying(MISOIdx::SRC0_IDX)]);
    paramList.insert(paramList.end(), offsetExpr.begin(), offsetExpr.end());
    paramList.insert(paramList.end(), shapeExpr.begin(), shapeExpr.end());

    std::ostringstream oss;
    oss << tileOpName << WrapParamByAngleBrackets({std::to_string(isConv3D)});
    oss << WrapParamByParentheses(paramList) << STMT_END;
    return oss.str();
}

std::string CodeGenOpNPU::GenMemL1CopyInConvBpNZ() const
{
    auto tileOpParams = GetTileOpParamsByOrder();

    std::vector<std::string> paramList;
    paramList.emplace_back(tileOpParams[ToUnderlying(MISOIdx::DST_IDX)]);
    paramList.emplace_back(tileOpParams[ToUnderlying(MISOIdx::SRC0_IDX)]);

    auto dynOffset = GetOffsetFromAttr(ToUnderlying(MISOIdx::SRC0_IDX));

    std::vector<std::string> gmOffsetExpr = GenSymbolicArgument(dynOffset);
    for (size_t i = 0; i < SHAPE_DIM4; i++) {
        paramList.emplace_back(gmOffsetExpr[i]);
    }

    std::ostringstream oss;
    oss << tileOpName;
    oss << WrapParamByParentheses(paramList) << STMT_END;
    return oss.str();
}

std::vector<std::string> CodeGenOpNPU::BuildCopyOutConvBpDwParamList(
    const std::string& dstTensor, const std::string& srcTensor, const std::vector<std::string>& gmOffsetExpr,
    const std::vector<int64_t>& staticOffsets, const std::string& realM, const std::string& realN) const
{
    std::vector<std::string> tileOpCopyOutParamList;
    tileOpCopyOutParamList.emplace_back(dstTensor);
    tileOpCopyOutParamList.emplace_back(srcTensor);

    // offset 统一为 SHAPE_DIM5 维：DW 的 4 维 FRACTAL_Z 视图 offset 补 0 至 5 维
    if (functionType == FunctionType::STATIC) {
        for (size_t i = 0; i < SHAPE_DIM5; i++) {
            tileOpCopyOutParamList.emplace_back(std::to_string(staticOffsets[i]));
        }
    } else {
        for (size_t i = 0; i < SHAPE_DIM5; i++) {
            tileOpCopyOutParamList.emplace_back(gmOffsetExpr[i]);
        }
    }

    tileOpCopyOutParamList.emplace_back(realM);
    tileOpCopyOutParamList.emplace_back(realN);
    return tileOpCopyOutParamList;
}

std::string CodeGenOpNPU::GenMemL0CCopyOutConvBpDw() const
{
    std::vector<std::string> tileOpParams = GetTileOpParamsByOrder();
    std::string copyOutModeStr = GetConvCopyOutMode();

    bool isConv3D = false;
    GetOpAttr(Conv::LoadStoreConvOpAttributeKey::isConv3D, isConv3D);

    // L0C有效形状{M, N}：M方向为Cout，N方向为Cin*Kh*Kw
    std::vector<SymbolicScalar> srcShapeVec;
    GetOpAttr(OpAttributeKey::l0cValidMN, srcShapeVec);
    ASSERT(ConvCodenGenError::CODEGEN_CHECK_DIM_INVALID, srcShapeVec.size() == SHAPE_DIM2)
        << "GenMemL0CCopyOutConvBpDw valid shape should be 2-dim!";
    std::string realM = SymbolicExpressionTable::BuildExpression(srcShapeVec[ID0]);
    std::string realN = SymbolicExpressionTable::BuildExpression(srcShapeVec[ID1]);

    // GM res为FRACTAL_Z (Cin1*Hk*Wk, Cout1, Cout0, Cin0)，按(1, Cin1*Hk*Wk, Cout1, Cout0, Cin0)视图搬出：
    // L0C的N方向(Cin*Kh*Kw)按Cin0分块对应Cin1*Hk*Wk轴，M方向(Cout)按Cout0分块对应Cout1轴
    auto dynOffset = GetOffsetFromAttr(ToUnderlying(MISOIdx::DST_IDX));
    size_t expectedDim = isConv3D ? SHAPE_DIM5 : SHAPE_DIM4;
    ASSERT(ConvCodenGenError::CODEGEN_CHECK_DIM_INVALID, dynOffset.size() >= expectedDim)
        << "GenMemL0CCopyOutConvBpDw offset should be at least " << expectedDim << "-dim!";

    std::vector<int64_t> staticOffsets;
    std::vector<std::string> gmOffsetExpr;
    GetDynamicOffsetExpr(dynOffset, isConv3D, gmOffsetExpr, staticOffsets);

    std::vector<std::string> tileOpParamList = BuildCopyOutConvBpDwParamList(
        tileOpParams[ToUnderlying(MISOIdx::DST_IDX)], tileOpParams[ToUnderlying(MISOIdx::SRC0_IDX)], gmOffsetExpr,
        staticOffsets, realM, realN);

    std::ostringstream oss;
    oss << tileOpName << WrapParamByAngleBrackets({copyOutModeStr, std::to_string(isConv3D)});
    oss << WrapParamByParentheses(tileOpParamList) << STMT_END;
    return oss.str();
}

} // namespace npu::tile_fwk
