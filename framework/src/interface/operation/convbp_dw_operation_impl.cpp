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
 * \file convbp_dw_operation_impl.cpp
 * \brief ConvBackwardFilter operation implementation
 */

#include "interface/configs/config_manager.h"
#include "interface/inner/pre_def.h"
#include "interface/operation/operation.h"
#include "interface/operation/operation_common.h"
#include "interface/program/program.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/utils/common.h"
#include "interface/utils/operator_tracer.h"
#include "tilefwk/error_code.h"
#include "operation_impl.h"
#include "tilefwk/data_type.h"
#include "tilefwk/tile_shape.h"
#include "tilefwk/platform.h"

namespace npu {
namespace tile_fwk {
namespace ConvBp {

void CheckDimensionRange(const std::vector<int64_t>& vec, const std::string& name, int minVal, int maxVal);

void CheckValueRange(int64_t value, const std::string& name, int64_t min, int64_t max, const std::string& formula = "");

void CheckAlignment(int64_t value, int64_t alignment, const std::string& valueName);

int64_t ConvAlignB(int64_t a, int64_t b);

void ConstructDwInputL1Tile(Function& function, const ConvBpGraphNodes& tensorGraphNodes,
                            const ConvBpTileInfo& convBpTileInfo, ConvBpIterInfo& convBpIterInfo,
                            LogicalTensorPtr& dstBL1TensorPtr, const ConvBpAttrParam& convBpAttrParam)
{
    int64_t khKw = convBpTileInfo.orgKh * convBpTileInfo.orgKw;
    int64_t cin1L1Size = CeilDiv(convBpIterInfo.nL1Size / khKw, convBpTileInfo.cin0);
    std::vector<int64_t> dstBL1Shape = std::vector<int64_t>{1, cin1L1Size, convBpIterInfo.hinL1Size,
                                                            convBpIterInfo.winL1Size, convBpTileInfo.cin0};
    dstBL1TensorPtr = std::make_shared<LogicalTensor>(function, tensorGraphNodes.inputTensorPtr->Datatype(),
                                                      dstBL1Shape, SymbolicScalar::FromConcrete(dstBL1Shape),
                                                      tensorGraphNodes.inputTensorPtr->Format(), "bL1Tensor");
    dstBL1TensorPtr->UpdateDynValidShape(SymbolicScalar::FromConcrete(dstBL1Shape));
    auto& copyInBL1Op = function.AddOperation(Opcode::OP_L1_COPY_IN_CONV, {tensorGraphNodes.inputTensorPtr},
                                              {dstBL1TensorPtr});

    int64_t src_c1_offset = convBpIterInfo.nL1Offset / (khKw * convBpTileInfo.cin0);
    int64_t src_hin_offset = convBpIterInfo.hinL1Offset > 0 ? convBpIterInfo.hinL1Offset : 0;
    int64_t src_win_offset = convBpIterInfo.winL1Offset > 0 ? convBpIterInfo.winL1Offset : 0;
    std::vector<int64_t> srcGmOffset = {convBpIterInfo.batchOffset, src_c1_offset, src_hin_offset, src_win_offset, 0};
    std::vector<int64_t> srcGmShape = {1, cin1L1Size, convBpIterInfo.hinL1Size, convBpIterInfo.winL1Size,
                                       convBpTileInfo.cin0};
    auto copyAttr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(srcGmOffset), MemoryType::MEM_L1, OpImmediate::Specified(srcGmShape),
        OpImmediate::Specified(tensorGraphNodes.inputTensorPtr->tensor->GetDynRawShape()),
        OpImmediate::Specified(dstBL1Shape));
    copyInBL1Op.SetOpAttribute(copyAttr);
    copyInBL1Op.SetAttribute(OpAttributeKey::srcGmConvValidShape, SymbolicScalar::FromConcrete(srcGmShape));
    copyInBL1Op.SetAttribute("l1_tile_shape", SymbolicScalar::FromConcrete(dstBL1Shape));
    copyInBL1Op.SetAttribute(OpAttributeKey::isConv, true);
    copyInBL1Op.SetAttribute(Conv::LoadStoreConvOpAttributeKey::isFmap, true);
    copyInBL1Op.SetAttribute(Conv::LoadStoreConvOpAttributeKey::isConv3D, convBpAttrParam.isConv3D);
    copyInBL1Op.SetAttribute(Conv::LoadStoreConvOpAttributeKey::copyInMode,
                             static_cast<int64_t>(Conv::CopyInMode::COPY_MOD_NZ2NZ));
    convBpIterInfo.bL1UpadateFlag = false;
}

LogicalTensorPtr ConstructDwInputTile(Function& function, const ConvBpGraphNodes& tensorGraphNodes,
                                      const ConvBpTileInfo& convBpTileInfo, ConvBpIterInfo& iterInfo,
                                      LogicalTensorPtr& dstBL1TensorPtr, const ConvBpAttrParam& convBpAttrParam)
{
    // L1层级 input 展开
    if (iterInfo.bL1UpadateFlag) {
        ConstructDwInputL1Tile(function, tensorGraphNodes, convBpTileInfo, iterInfo, dstBL1TensorPtr, convBpAttrParam);
    }

    std::vector<int64_t> dstBL0Shape = std::vector<int64_t>{ConvAlignB(iterInfo.kL0Size, Conv::MKN_N_VALUE),
                                                            ConvAlignB(iterInfo.nL0Size, Conv::MKN_N_VALUE)};
    std::vector<SymbolicScalar> dstBL0ValidShape = SymbolicScalar::FromConcrete({iterInfo.kL0Size, iterInfo.nL0Size});
    LogicalTensorPtr dstBL0TensorPtr = std::make_shared<LogicalTensor>(
        function, tensorGraphNodes.inputTensorPtr->Datatype(), dstBL0Shape, dstBL0ValidShape,
        tensorGraphNodes.inputTensorPtr->Format(), "bL0Tensor");
    dstBL0TensorPtr->UpdateDynValidShape(dstBL0ValidShape);

    auto& load3dOpBl0 = function.AddOperation(Opcode::OP_LOAD3D_CONV_DW, {dstBL1TensorPtr}, {dstBL0TensorPtr});
    load3dOpBl0.SetAttribute("l0_tile_shape", SymbolicScalar::FromConcrete(dstBL0Shape));

    load3dOpBl0.SetAttribute(OpAttributeKey::postM, iterInfo.kL0Offset - iterInfo.kL1Offset);
    load3dOpBl0.SetAttribute(OpAttributeKey::postK, iterInfo.nL0Offset);
    load3dOpBl0.SetAttribute(OpAttributeKey::isConv, true);
    load3dOpBl0.SetAttribute(OpAttributeKey::filterH, convBpTileInfo.orgKh);
    load3dOpBl0.SetAttribute(OpAttributeKey::filterW, convBpTileInfo.orgKw);
    load3dOpBl0.SetAttribute(OpAttributeKey::strideH, convBpAttrParam.strides[1]);
    load3dOpBl0.SetAttribute(OpAttributeKey::strideW, convBpAttrParam.strides[2]);
    load3dOpBl0.SetAttribute(OpAttributeKey::dilationH, convBpAttrParam.dilations[1]);
    load3dOpBl0.SetAttribute(OpAttributeKey::dilationW, convBpAttrParam.dilations[2]);
    CONV_LOGW("postM=%ld, postK=%ld, filterH=%ld, filterW=%ld, strideH=%ld, strideW=%ld, dilationH=%ld, dilationW=%ld",
              (iterInfo.kL0Offset - iterInfo.kL1Offset), iterInfo.nL0Offset, convBpTileInfo.orgKh, convBpTileInfo.orgKw,
              convBpAttrParam.strides[1], convBpAttrParam.strides[2], convBpAttrParam.dilations[1],
              convBpAttrParam.dilations[2]);

    // im2col 的 pad 填 0 边界按 tile 相对计算（L1 B 只搬有效 input 区间）
    int64_t khDilated = (convBpTileInfo.orgKh - 1) * convBpAttrParam.dilations[1] + 1;
    int64_t kwDilated = (convBpTileInfo.orgKw - 1) * convBpAttrParam.dilations[2] + 1;

    int64_t endHin = iterInfo.hinL1Offset + (iterInfo.houtL1Size - 1) * convBpAttrParam.strides[1] + khDilated;
    load3dOpBl0.SetAttribute(OpAttributeKey::paddingTop, iterInfo.hinL1Offset < 0 ? 0 - iterInfo.hinL1Offset : 0);
    load3dOpBl0.SetAttribute(OpAttributeKey::paddingBottom,
                             endHin > convBpTileInfo.orgHin ? endHin - convBpTileInfo.orgHin : 0);
    CONV_LOGW("hinL1Offset=%ld, endHin=%ld", iterInfo.hinL1Offset, endHin);

    int64_t endWin = iterInfo.winL1Offset + (iterInfo.woutL1Size - 1) * convBpAttrParam.strides[2] + kwDilated;
    load3dOpBl0.SetAttribute(OpAttributeKey::paddingLeft, iterInfo.winL1Offset < 0 ? 0 - iterInfo.winL1Offset : 0);
    load3dOpBl0.SetAttribute(OpAttributeKey::paddingRight,
                             endWin > convBpTileInfo.orgWin ? endWin - convBpTileInfo.orgWin : 0);
    CONV_LOGW("winL1Offset=%ld, endWin=%ld", iterInfo.winL1Offset, endWin);

    load3dOpBl0.SetAttribute(OpAttributeKey::padValue, 0);
    load3dOpBl0.SetAttribute(OpAttributeKey::repeatStride, iterInfo.repeatStride);
    load3dOpBl0.SetAttribute(OpAttributeKey::repeatTime, iterInfo.repeatTime);
    load3dOpBl0.SetAttribute(OpAttributeKey::wStride, iterInfo.wStride);
    CONV_LOGW("repeatStride=%ld, repeatTime=%ld, wStride=%ld", iterInfo.repeatStride, iterInfo.repeatTime,
              iterInfo.wStride);
    return dstBL0TensorPtr;
}

static void ConstructDwGradOutputL1Tile(Function& function, const ConvBpGraphNodes& tensorGraphNodes,
                                        const ConvBpTileInfo& convBpTileInfo, ConvBpIterInfo& convBpIterInfo,
                                        LogicalTensorPtr& dstAL1TensorPtr, const ConvBpAttrParam& convBpAttrParam)
{
    int64_t cout1L1Size = CeilDiv(convBpIterInfo.mL1Size, convBpTileInfo.cout0);
    int64_t houtWoutDiv16 = CeilDiv(convBpIterInfo.houtL1Size * convBpIterInfo.woutL1Size, 16);
    std::vector<int64_t> dstAL1Shape = {1, cout1L1Size, houtWoutDiv16, 16, convBpTileInfo.cout0};
    dstAL1TensorPtr = std::make_shared<LogicalTensor>(function, tensorGraphNodes.gradOutputTensorPtr->Datatype(),
                                                      dstAL1Shape, tensorGraphNodes.gradOutputTensorPtr->Format(),
                                                      "aL1Tensor");
    dstAL1TensorPtr->UpdateDynValidShape(SymbolicScalar::FromConcrete(dstAL1Shape));
    auto& copyInAL1Op = function.AddOperation(Opcode::OP_L1_COPY_IN_CONV_BP_DW_DY,
                                              {tensorGraphNodes.gradOutputTensorPtr}, {dstAL1TensorPtr});

    int64_t src_c1_offset = convBpIterInfo.mL1Offset / convBpTileInfo.cout0;
    std::vector<int64_t> srcGmOffset = {convBpIterInfo.batchOffset, src_c1_offset, convBpIterInfo.houtL1Offset,
                                        convBpIterInfo.woutL1Offset, 0};
    std::vector<int64_t> srcGmShape = {1, cout1L1Size, convBpIterInfo.houtL1Size, convBpIterInfo.woutL1Size,
                                       convBpTileInfo.cout0};
    auto copyAttr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(srcGmOffset), MemoryType::MEM_L1, OpImmediate::Specified(srcGmShape),
        OpImmediate::Specified(tensorGraphNodes.gradOutputTensorPtr->tensor->GetDynRawShape()),
        OpImmediate::Specified(dstAL1Shape));
    copyInAL1Op.SetOpAttribute(copyAttr);
    copyInAL1Op.SetAttribute(OpAttributeKey::srcGmConvValidShape, SymbolicScalar::FromConcrete(srcGmShape));
    copyInAL1Op.SetAttribute(Conv::LoadStoreConvOpAttributeKey::isFmap, true);
    copyInAL1Op.SetAttribute("l1_tile_shape", SymbolicScalar::FromConcrete(dstAL1Shape));
    copyInAL1Op.SetAttribute(Conv::LoadStoreConvOpAttributeKey::copyInMode,
                             static_cast<int64_t>(Conv::CopyInMode::COPY_MOD_NZ2NZ));
    copyInAL1Op.SetAttribute(Conv::LoadStoreConvOpAttributeKey::isConv3D, convBpAttrParam.isConv3D);
    convBpIterInfo.aL1UpadateFlag = false;
}

LogicalTensorPtr ConstructDwGradOutputTile(Function& function, const ConvBpGraphNodes& tensorGraphNodes,
                                           const ConvBpTileInfo& convBpTileInfo, ConvBpIterInfo& iterInfo,
                                           LogicalTensorPtr& dstAL1TensorPtr, const ConvBpAttrParam& convBpAttrParam)
{
    // L1层级 gradoutput 展开
    if (iterInfo.aL1UpadateFlag) {
        ConstructDwGradOutputL1Tile(function, tensorGraphNodes, convBpTileInfo, iterInfo, dstAL1TensorPtr,
                                    convBpAttrParam);
    }

    std::vector<int64_t> dstAL0Shape = std::vector<int64_t>{ConvAlignB(iterInfo.mL0Size, Conv::MKN_M_VALUE),
                                                            ConvAlignB(iterInfo.kL0Size, Conv::MKN_N_VALUE)};
    std::vector<SymbolicScalar> dstAL0ValidShape = SymbolicScalar::FromConcrete({iterInfo.mL0Size, iterInfo.kL0Size});
    LogicalTensorPtr dstAL0TensorPtr = std::make_shared<LogicalTensor>(
        function, tensorGraphNodes.gradOutputTensorPtr->Datatype(), dstAL0Shape, dstAL0ValidShape,
        tensorGraphNodes.gradOutputTensorPtr->Format(), "aL0Tensor");
    dstAL0TensorPtr->UpdateDynValidShape(dstAL0ValidShape);

    auto& loadToL0aOp = function.AddOperation(Opcode::OP_L1_TO_L0A_CONV_DW, {dstAL1TensorPtr}, {dstAL0TensorPtr});
    loadToL0aOp.SetAttribute(OpAttributeKey::postM, iterInfo.mL0Offset);
    loadToL0aOp.SetAttribute(OpAttributeKey::postK, iterInfo.kL0Offset - iterInfo.kL1Offset);
    loadToL0aOp.SetAttribute(OpAttributeKey::isConv, true);
    loadToL0aOp.SetAttribute("l0_tile_shape", SymbolicScalar::FromConcrete(dstAL0Shape));

    return dstAL0TensorPtr;
}

void BpSetDwAMulBAttr(const ConvBpGraphNodes& tensorGraphNodes, const ConvBpTileInfo& convBpTileInfo, Operation& op)
{
    ASSERT(ConvExpandFuncError::EXPANDFUNC_TILE_OP_NULLPTR, tensorGraphNodes.gradOutputTensorPtr != nullptr &&
                                                                tensorGraphNodes.inputTensorPtr != nullptr &&
                                                                tensorGraphNodes.resTensorPtr != nullptr)
        << "Expected gradOutputTensorPtr, inputTensorPtr, and resTensorPtr to be non-nullptr.";

    int64_t nzAttr = (static_cast<int64_t>(tensorGraphNodes.gradOutputTensorPtr->Format())) |
                     (static_cast<int64_t>(tensorGraphNodes.inputTensorPtr->Format()) << 1) |
                     (static_cast<int64_t>(tensorGraphNodes.resTensorPtr->Format()) << 2);
    op.SetAttribute(OpAttributeKey::isConv, true);
    op.SetAttribute(Conv::MATMUL_NZ_ATTR, nzAttr);
    op.SetAttribute(Conv::A_MUL_B_ACT_M, convBpTileInfo.mL0);
    op.SetAttribute(Conv::A_MUL_B_ACT_K, convBpTileInfo.kL0);
    op.SetAttribute(Conv::A_MUL_B_ACT_N, convBpTileInfo.nL0);
}

LogicalTensorPtr DoDwMmad(Function& function, const ConvBpGraphNodes& tensorGraphNodes,
                          ConvBpGraphNodes& tileGraphNodes, const ConvBpTileInfo& convBpTileInfo,
                          const ConvBpIterInfo& iterInfo)
{
    std::vector<SymbolicScalar> dstCL0DynValidShape = std::vector<SymbolicScalar>{iterInfo.mL0Size, iterInfo.nL0Size};
    ASSERT(ConvExpandFuncError::EXPANDFUNC_TILE_OP_NULLPTR, tileGraphNodes.gradOutputTensorPtr != nullptr &&
                                                                tileGraphNodes.inputTensorPtr != nullptr &&
                                                                tileGraphNodes.resTensorPtr != nullptr)
        << "Inputs and res must be non-nullptr.";
    // MMAD node add
    std::vector<LogicalTensorPtr> mmadInputs;
    std::vector<LogicalTensorPtr> mmadOutputs;
    const std::string MmadOpStr = iterInfo.isFirstK ? "TILE_A_MUL_B" : "TILE_A_MULACC_B";
    if (iterInfo.isFirstK) {
        mmadInputs = {tileGraphNodes.gradOutputTensorPtr, tileGraphNodes.inputTensorPtr};
    } else {
        mmadInputs = {tileGraphNodes.gradOutputTensorPtr, tileGraphNodes.inputTensorPtr,
                      tileGraphNodes.cL0PartialSumPtr};
    }

    if (iterInfo.isLastK) {
        mmadOutputs = {tileGraphNodes.resTensorPtr};
    } else {
        std::vector<int64_t> cL0PartialSumShape = {ConvAlignB(iterInfo.mL0Size, Conv::MKN_M_VALUE),
                                                   ConvAlignB(iterInfo.nL0Size, Conv::MKN_N_VALUE)};
        tileGraphNodes.cL0PartialSumPtr = std::make_shared<LogicalTensor>(
            function, DataType::DT_FP32, cL0PartialSumShape, dstCL0DynValidShape, TileOpFormat::TILEOP_NZ,
            "cL0PartialSumTensor");
        tileGraphNodes.cL0PartialSumPtr->UpdateDynValidShape(dstCL0DynValidShape);
        mmadOutputs = {tileGraphNodes.cL0PartialSumPtr};
    }
    auto& aMulBOp = function.AddOperation(MmadOpStr, mmadInputs, mmadOutputs);
    BpSetDwAMulBAttr(tensorGraphNodes, convBpTileInfo, aMulBOp);

    return mmadOutputs[0];
}

void ConstructDwCopyOutTile(Function& function, const ConvBpAttrParam& convBpAttrParam,
                            const ConvBpGraphNodes& tensorGraphNodes, const ConvBpTileInfo& convBpTileInfo,
                            const ConvBpIterInfo& convBpIterInfo, const LogicalTensorPtr& resCl0TensorPtr)
{
    std::vector<SymbolicScalar> dstCL0DynValidShape = std::vector<SymbolicScalar>{convBpIterInfo.mL0Size,
                                                                                  convBpIterInfo.nL0Size};
    auto& fixpipeOpRes = function.AddOperation(Opcode::OP_L0C_COPY_OUT_CONV_DW, {resCl0TensorPtr},
                                               {tensorGraphNodes.resTensorPtr});
    fixpipeOpRes.SetAttribute(OpAttributeKey::isConv, true);
    fixpipeOpRes.SetAttribute(Conv::LoadStoreConvOpAttributeKey::isConv3D, convBpAttrParam.isConv3D);
    resCl0TensorPtr->UpdateDynValidShape(dstCL0DynValidShape);
    fixpipeOpRes.SetAttribute(Conv::LoadStoreConvOpAttributeKey::copyOutMode,
                              static_cast<int64_t>(Conv::CopyOutMode::COPY_MOD_NZ2NZ));
    fixpipeOpRes.SetAttribute(OpAttributeKey::l0cValidMN, dstCL0DynValidShape);

    int64_t dstMOffset = convBpIterInfo.mL1Offset + convBpIterInfo.mL0Offset;
    int64_t dstNOffset = convBpIterInfo.nL1Offset + convBpIterInfo.nL0Offset;

    // GM res为FRACTAL_Z (Cin1*Hk*Wk, Cout1, Cout0, Cin0)，按该视图搬出：
    // L0C的N方向(Cin*Kh*Kw)按Cin0分块对应Cin1*Hk*Wk轴，M方向(Cout)按Cout0分块对应Cout1轴
    std::vector<int64_t> dstResGmOffset = {dstNOffset / convBpTileInfo.cin0, dstMOffset / convBpTileInfo.cout0, 0, 0};
    auto copyAttr = std::make_shared<CopyOpAttribute>(
        MemoryType::MEM_L0C, OpImmediate::Specified(dstResGmOffset),
        OpImmediate::Specified(tensorGraphNodes.resTensorPtr->tensor->GetRawShape()),
        OpImmediate::Specified(tensorGraphNodes.resTensorPtr->tensor->GetDynRawShape()),
        OpImmediate::Specified(dstCL0DynValidShape));
    fixpipeOpRes.SetOpAttribute(copyAttr);
}

void UpdateDwKL1IterInfo(const ConvBpTileInfo& convBpTileInfo, ConvBpIterInfo& convBpIterInfo,
                         const ConvBpAttrParam& convBpAttrParam)
{
    if (convBpTileInfo.kL1 < convBpTileInfo.orgWout) {
        // kL1小于wout场景下保证不跨行
        int64_t woutRemain = convBpTileInfo.orgWout - (convBpIterInfo.kL1Offset % convBpTileInfo.orgWout);
        convBpIterInfo.kL1Size = std::min(woutRemain, convBpTileInfo.kL1);
    } else {
        convBpIterInfo.kL1Size = std::min(convBpTileInfo.orgHoutWout - convBpIterInfo.kL1Offset, convBpTileInfo.kL1);
    }
    CONV_LOGW("kL1Size=%ld", convBpIterInfo.kL1Size);

    convBpIterInfo.houtL1Offset = convBpIterInfo.kL1Offset / convBpTileInfo.orgWout;
    convBpIterInfo.houtL1Size = CeilDiv(convBpIterInfo.kL1Size, convBpTileInfo.orgWout);
    convBpIterInfo.hinL1Offset = convBpIterInfo.houtL1Offset * convBpAttrParam.strides[1] - convBpAttrParam.paddings[2];
    int64_t endHinOffset = convBpIterInfo.hinL1Offset + (convBpIterInfo.houtL1Size - 1) * convBpAttrParam.strides[1] +
                           (convBpTileInfo.orgKh - 1) * convBpAttrParam.dilations[1] + 1;
    if (convBpIterInfo.hinL1Offset < convBpTileInfo.orgHin && endHinOffset > 0) {
        int64_t realStartHin = convBpIterInfo.hinL1Offset < 0 ? 0 : convBpIterInfo.hinL1Offset;
        int64_t realEndHin = endHinOffset > convBpTileInfo.orgHin ? convBpTileInfo.orgHin : endHinOffset;
        convBpIterInfo.hinL1Size = realEndHin - realStartHin;
    } else {
        convBpIterInfo.hinL1Size = 0;
    }
    CONV_LOGW("hinL1Offset=%ld, endHinOffset=%ld, hinL1Size=%ld", convBpIterInfo.hinL1Offset, endHinOffset,
              convBpIterInfo.hinL1Size);

    convBpIterInfo.woutL1Offset = convBpIterInfo.kL1Offset % convBpTileInfo.orgWout;
    convBpIterInfo.woutL1Size = std::min(convBpIterInfo.kL1Size, convBpTileInfo.orgWout);
    convBpIterInfo.winL1Offset = convBpIterInfo.woutL1Offset * convBpAttrParam.strides[2] - convBpAttrParam.paddings[4];
    int64_t endWinOffset = convBpIterInfo.winL1Offset + (convBpIterInfo.woutL1Size - 1) * convBpAttrParam.strides[2] +
                           (convBpTileInfo.orgKw - 1) * convBpAttrParam.dilations[2] + 1;
    if (convBpIterInfo.winL1Offset < convBpTileInfo.orgWin && endWinOffset > 0) {
        int64_t realStartWin = convBpIterInfo.winL1Offset < 0 ? 0 : convBpIterInfo.winL1Offset;
        int64_t realEndWin = endWinOffset > convBpTileInfo.orgWin ? convBpTileInfo.orgWin : endWinOffset;
        convBpIterInfo.winL1Size = realEndWin - realStartWin;
    } else {
        convBpIterInfo.winL1Size = 0;
    }

    CONV_LOGW("winL1Offset=%ld, endWinOffset=%ld, winL1Size=%ld", convBpIterInfo.winL1Offset, endWinOffset,
              convBpIterInfo.winL1Size);
}

void IterKExpandFunc(Function& function, ConvBpIterInfo& convBpIterInfo, ConvBpTileInfo& convBpTileInfo,
                     const ConvBpAttrParam& convBpAttrParam, const ConvBpGraphNodes& tensorGraphNodes,
                     ConvBpGraphNodes& tileGraphNodes, LogicalTensorPtr& gradOutputL1TensorPtr,
                     LogicalTensorPtr& inputL1TensorPtr, LogicalTensorPtr& resCl0TensorPtr)
{
    bool isFirstKOverall = true;
    for (convBpIterInfo.batchDoutOffset = 0; convBpIterInfo.batchDoutOffset < convBpTileInfo.orgBatchDout;
         convBpIterInfo.batchDoutOffset += 1) {
        convBpIterInfo.batchOffset = convBpIterInfo.batchDoutOffset / convBpTileInfo.orgDout;
        for (convBpIterInfo.kL1Offset = 0; convBpIterInfo.kL1Offset < convBpTileInfo.orgHoutWout;) {
            UpdateDwKL1IterInfo(convBpTileInfo, convBpIterInfo, convBpAttrParam);
            std::vector<int64_t> dstCL0Shape = {ConvAlignB(convBpIterInfo.mL0Size, Conv::MKN_M_VALUE),
                                                ConvAlignB(convBpIterInfo.nL0Size, Conv::MKN_N_VALUE)};
            std::vector<SymbolicScalar> dstCL0DynValidShape = {convBpIterInfo.mL0Size, convBpIterInfo.nL0Size};

            tileGraphNodes.resTensorPtr = std::make_shared<LogicalTensor>(
                function, tensorGraphNodes.gradOutputTensorPtr->Datatype(), dstCL0Shape, dstCL0DynValidShape,
                tensorGraphNodes.gradOutputTensorPtr->Format(), "cL0Tensor");

            convBpIterInfo.aL1UpadateFlag = true;
            convBpIterInfo.bL1UpadateFlag = true;
            int64_t kLimit = convBpIterInfo.kL1Size + convBpIterInfo.kL1Offset;
            for (convBpIterInfo.kL0Offset = convBpIterInfo.kL1Offset; convBpIterInfo.kL0Offset < kLimit;) {
                if (convBpTileInfo.kL0 < convBpTileInfo.orgWout) {
                    int64_t curWoutIdx = convBpIterInfo.kL0Offset % convBpTileInfo.orgWout;
                    convBpIterInfo.kL0Size = std::min(
                        {convBpTileInfo.kL0, convBpTileInfo.orgWout - curWoutIdx, kLimit - convBpIterInfo.kL0Offset});
                } else {
                    convBpIterInfo.kL0Size = std::min(kLimit - convBpIterInfo.kL0Offset, convBpTileInfo.kL0);
                }
                CONV_LOGW("kL0Size=%ld", convBpIterInfo.kL0Size);

                bool isLastK = (convBpIterInfo.batchDoutOffset == convBpTileInfo.orgBatchDout - 1) &&
                               (convBpIterInfo.kL0Offset + convBpIterInfo.kL0Size >= convBpTileInfo.orgHoutWout);
                convBpIterInfo.isFirstK = isFirstKOverall;
                convBpIterInfo.isLastK = isLastK;
                isFirstKOverall = false;

                convBpIterInfo.repeatStride = 0;
                convBpIterInfo.repeatTime = 1;
                convBpIterInfo.wStride = ConvAlignB(convBpIterInfo.kL0Size, Conv::MKN_M_VALUE);

                tileGraphNodes.gradOutputTensorPtr = ConstructDwGradOutputTile(
                    function, tensorGraphNodes, convBpTileInfo, convBpIterInfo, gradOutputL1TensorPtr, convBpAttrParam);
                tileGraphNodes.inputTensorPtr = ConstructDwInputTile(function, tensorGraphNodes, convBpTileInfo,
                                                                     convBpIterInfo, inputL1TensorPtr, convBpAttrParam);

                resCl0TensorPtr = DoDwMmad(function, tensorGraphNodes, tileGraphNodes, convBpTileInfo, convBpIterInfo);
                convBpIterInfo.kL0Offset += convBpIterInfo.kL0Size;
            }
            convBpIterInfo.kL1Offset += convBpIterInfo.kL1Size;
        }
    }
}

void IterDwL0ExpandFunc(Function& function, ConvBpIterInfo& convBpIterInfo, ConvBpTileInfo& convBpTileInfo,
                        const ConvBpAttrParam& convBpAttrParam, const ConvBpGraphNodes& tensorGraphNodes,
                        ConvBpGraphNodes& tileGraphNodes)
{
    LogicalTensorPtr gradOutputL1TensorPtr = nullptr;
    LogicalTensorPtr inputL1TensorPtr = nullptr;
    LogicalTensorPtr resCl0TensorPtr = nullptr;
    for (convBpIterInfo.nL0Offset = 0; convBpIterInfo.nL0Offset < convBpIterInfo.nL1Size;
         convBpIterInfo.nL0Offset += convBpTileInfo.nL0) {
        convBpIterInfo.nL0Size = std::min(convBpIterInfo.nL1Size - convBpIterInfo.nL0Offset, convBpTileInfo.nL0);

        for (convBpIterInfo.mL0Offset = 0; convBpIterInfo.mL0Offset < convBpIterInfo.mL1Size;
             convBpIterInfo.mL0Offset += convBpTileInfo.mL0) {
            convBpIterInfo.mL0Size = std::min(convBpIterInfo.mL1Size - convBpIterInfo.mL0Offset, convBpTileInfo.mL0);
            IterKExpandFunc(function, convBpIterInfo, convBpTileInfo, convBpAttrParam, tensorGraphNodes, tileGraphNodes,
                            gradOutputL1TensorPtr, inputL1TensorPtr, resCl0TensorPtr);
            ConstructDwCopyOutTile(function, convBpAttrParam, tensorGraphNodes, convBpTileInfo, convBpIterInfo,
                                   resCl0TensorPtr);
        }
    }
}

void IterDwL1ExpandFunc(Function& function, ConvBpIterInfo& convBpIterInfo, ConvBpTileInfo& convBpTileInfo,
                        const ConvBpAttrParam& convBpAttrParam, const ConvBpGraphNodes& tensorGraphNodes,
                        ConvBpGraphNodes& tileGraphNodes)
{
    for (convBpIterInfo.nL1Offset = 0; convBpIterInfo.nL1Offset < convBpTileInfo.cin1KdKhKwCin0;
         convBpIterInfo.nL1Offset += convBpTileInfo.nL1) {
        convBpIterInfo.nL1Size = std::min(convBpTileInfo.cin1KdKhKwCin0 - convBpIterInfo.nL1Offset, convBpTileInfo.nL1);
        for (convBpIterInfo.mL1Offset = 0; convBpIterInfo.mL1Offset < convBpTileInfo.coutPerGroup;
             convBpIterInfo.mL1Offset += convBpTileInfo.mL1) {
            convBpIterInfo.mL1Size = std::min(convBpTileInfo.coutPerGroup - convBpIterInfo.mL1Offset,
                                              convBpTileInfo.mL1);
            IterDwL0ExpandFunc(function, convBpIterInfo, convBpTileInfo, convBpAttrParam, tensorGraphNodes,
                               tileGraphNodes);
        }
    }
}

void SetDwConvBpShapeInfo(const TileShape& tileShape, const ConvBpAttrParam& convBpAttrParam,
                          ConvBpTileInfo& convBpTileInfo)
{
    // set org shape
    // 2D: oriResShape/oriGradOutputShape/oriInputShape are NCHW (4D)
    // 3D: oriResShape/oriGradOutputShape/oriInputShape are NCDHW (5D)
    if (convBpAttrParam.isConv3D) {
        convBpTileInfo.orgBatch = convBpAttrParam.oriInputShape[NCDHW_N_IDX];
        convBpTileInfo.orgCin = convBpAttrParam.oriInputShape[NCDHW_C_IDX];
        convBpTileInfo.orgDin = convBpAttrParam.oriInputShape[NCDHW_D_IDX];
        convBpTileInfo.orgHin = convBpAttrParam.oriInputShape[NCDHW_H_IDX];
        convBpTileInfo.orgWin = convBpAttrParam.oriInputShape[NCDHW_W_IDX];
        convBpTileInfo.orgCout = convBpAttrParam.oriGradOutputShape[NCDHW_C_IDX];
        convBpTileInfo.orgDout = convBpAttrParam.oriGradOutputShape[NCDHW_D_IDX];
        convBpTileInfo.orgHout = convBpAttrParam.oriGradOutputShape[NCDHW_H_IDX];
        convBpTileInfo.orgWout = convBpAttrParam.oriGradOutputShape[NCDHW_W_IDX];
        convBpTileInfo.orgKh = convBpAttrParam.oriResShape[NCDHW_H_IDX];
        convBpTileInfo.orgKw = convBpAttrParam.oriResShape[NCDHW_W_IDX];
        convBpTileInfo.orgKd = convBpAttrParam.oriResShape[NCDHW_D_IDX];
    } else {
        convBpTileInfo.orgBatch = convBpAttrParam.oriInputShape[NCHW_N_IDX];
        convBpTileInfo.orgCin = convBpAttrParam.oriInputShape[NCHW_C_IDX];
        convBpTileInfo.orgDin = 1;
        convBpTileInfo.orgHin = convBpAttrParam.oriInputShape[NCHW_H_IDX];
        convBpTileInfo.orgWin = convBpAttrParam.oriInputShape[NCHW_W_IDX];
        convBpTileInfo.orgCout = convBpAttrParam.oriGradOutputShape[NCHW_C_IDX];
        convBpTileInfo.orgDout = 1;
        convBpTileInfo.orgHout = convBpAttrParam.oriGradOutputShape[NCHW_H_IDX];
        convBpTileInfo.orgWout = convBpAttrParam.oriGradOutputShape[NCHW_W_IDX];
        convBpTileInfo.orgKh = convBpAttrParam.oriResShape[NCHW_H_IDX];
        convBpTileInfo.orgKw = convBpAttrParam.oriResShape[NCHW_W_IDX];
        convBpTileInfo.orgKd = 1;
    }

    convBpTileInfo.orgHoutWout = convBpTileInfo.orgHout * convBpTileInfo.orgWout;
    convBpTileInfo.orgBatchDout = convBpTileInfo.orgBatch * convBpTileInfo.orgDout;

    convBpTileInfo.cin0 = ALIGN_SIZE_16;
    convBpTileInfo.cout0 = Conv::MKN_N_VALUE;
    convBpTileInfo.cinPerGroup = convBpTileInfo.orgCin / convBpAttrParam.groups;
    convBpTileInfo.cin1PerGroup = CeilDiv(convBpTileInfo.cinPerGroup, convBpTileInfo.cin0);
    convBpTileInfo.coutPerGroup = convBpTileInfo.orgCout / convBpAttrParam.groups;
    convBpTileInfo.cout1PerGroup = CeilDiv(convBpTileInfo.coutPerGroup, convBpTileInfo.cout0);
    convBpTileInfo.cin1KdKhKwCin0 = convBpTileInfo.cin1PerGroup * convBpTileInfo.orgKd * convBpTileInfo.orgKh *
                                    convBpTileInfo.orgKw * convBpTileInfo.cin0;

    // set tileshape info
    auto& convBpTile = tileShape.GetConvBpFilterTile();
    convBpTileInfo.mL1 = convBpTile.tileL1Info.tileML1;
    convBpTileInfo.kL1 = convBpTile.tileL1Info.tileKL1;
    convBpTileInfo.nL1 = convBpTile.tileL1Info.tileNL1;
    convBpTileInfo.mL0 = convBpTile.tileL0Info.tileML0;
    convBpTileInfo.kL0 = convBpTile.tileL0Info.tileKL0;
    convBpTileInfo.nL0 = convBpTile.tileL0Info.tileNL0;
}

void SetDwTensorGraphNodes(const std::vector<LogicalTensorPtr>& operandVec, const LogicalTensorPtr& cTensorPtr,
                           ConvBpGraphNodes& tensorGraphNodes)
{
    tensorGraphNodes.gradOutputTensorPtr = operandVec[INPUT_GRAD_OUTPUT_IDX];
    tensorGraphNodes.inputTensorPtr = operandVec[INPUT_INPUT_IDX];
    ASSERT(ConvExpandFuncError::EXPANDFUNC_TILE_OP_NULLPTR,
           tensorGraphNodes.gradOutputTensorPtr != nullptr && tensorGraphNodes.inputTensorPtr != nullptr)
        << "Expected aTensorPtr and bTensorPtr to be non-nullptr.";
    ASSERT(ConvExpandFuncError::EXPANDFUNC_TILE_OP_NULLPTR, cTensorPtr != nullptr) << "cTensorPtr is nullptr.";
    tensorGraphNodes.resTensorPtr = cTensorPtr;
}

void SetDwConvBpAttrParam(const Operation& op, ConvBpAttrParam& convBpAttrParam)
{
    convBpAttrParam.isConv3D = (op.HasAttr(CONV_3D_FLAG)) ? op.GetBoolAttribute(CONV_3D_FLAG) : false;
    convBpAttrParam.paddings = (op.HasAttr(CONV_PADDINGS_ATTR)) ? op.GetVectorIntAttribute(CONV_PADDINGS_ATTR) :
                                                                  CONV3D_PAD_ATTR_DEFAULT_LIST;
    convBpAttrParam.strides = (op.HasAttr(CONV_STRIDES_ATTR)) ? op.GetVectorIntAttribute(CONV_STRIDES_ATTR) :
                                                                CONV3D_ATTR_DEFAULT_LIST;
    convBpAttrParam.dilations = (op.HasAttr(CONV_DILATIONS_ATTR)) ? op.GetVectorIntAttribute(CONV_DILATIONS_ATTR) :
                                                                    CONV3D_ATTR_DEFAULT_LIST;
    convBpAttrParam.groups = (op.HasAttr(CONV_GROUPS_ATTR)) ? op.GetIntAttribute(CONV_GROUPS_ATTR) : 1;

    convBpAttrParam.isInOutTensorNZ = true;

    ASSERT(ConvExpandFuncError::EXPANDFUNC_TENSOR_ATTR_GET_FAILED, op.HasAttr(CONV_ORI_GRAD_OUTPUT_SHAPE_ATTR))
        << "Conv Backward ori gradOutput shape should be set when InOut Tensor NZ mode.";
    ASSERT(ConvExpandFuncError::EXPANDFUNC_TENSOR_ATTR_GET_FAILED, op.HasAttr(CONV_ORI_INPUT_SHAPE_ATTR))
        << "Conv Backward ori input shape should be set when InOut Tensor NZ mode.";
    convBpAttrParam.oriGradOutputShape = op.GetVectorIntAttribute(CONV_ORI_GRAD_OUTPUT_SHAPE_ATTR);
    convBpAttrParam.oriInputShape = op.GetVectorIntAttribute(CONV_ORI_INPUT_SHAPE_ATTR);
    convBpAttrParam.oriResShape = op.GetVectorIntAttribute(CONV_ORI_RES_SHAPE_ATTR);
    convBpAttrParam.dynValidResShape = op.GetVectorSymbolicScalarAttribute("dynamicResValidShape");
}

void ConstructDwTileGraph(Function& function, const TileShape& tileShape,
                          const std::vector<LogicalTensorPtr>& operandVec, const LogicalTensorPtr& cTensorPtr,
                          const Operation& op)
{
    // op attr set
    ConvBpAttrParam convBpAttrParam;
    SetDwConvBpAttrParam(op, convBpAttrParam);
    // set tensor graph node info
    ConvBpGraphNodes tensorGraphNodes;
    SetDwTensorGraphNodes(operandVec, cTensorPtr, tensorGraphNodes);
    // save tile info
    ConvBpTileInfo convBpTileInfo;
    SetDwConvBpShapeInfo(tileShape, convBpAttrParam, convBpTileInfo);
    // save iter info
    ConvBpIterInfo convBpIterInfo;
    // set tile graph node info
    ConvBpGraphNodes tileGraphNodes;

    for (convBpIterInfo.groupOffset = 0; convBpIterInfo.groupOffset < convBpAttrParam.groups;
         convBpIterInfo.groupOffset += 1) {
        IterDwL1ExpandFunc(function, convBpIterInfo, convBpTileInfo, convBpAttrParam, tensorGraphNodes, tileGraphNodes);
    }
}

void FilterCheckOriginShape(const Tensor& gradOutputTensor, const Tensor& inputTensor,
                            const std::vector<int64_t>& filterSize)
{
    ConvBp::CheckDimensionRange(gradOutputTensor.GetShape(), "grad_output", NUM1, MAX_SIZE);
    ConvBp::CheckDimensionRange(inputTensor.GetShape(), "input", NUM1, MAX_SIZE);
    ConvBp::CheckDimensionRange(filterSize, "filter_size", NUM1, MAX_SIZE);
}

void FilterCheckOutputShape(const Tensor& gradOutputTensor, const Tensor& inputTensor,
                            const std::vector<int64_t>& filterSize, const ConvBpAttrParam& attrParam)
{
    int64_t cinPerGroup = filterSize[1];
    int64_t kh = filterSize[2];
    int64_t kw = filterSize[3];
    int64_t groups = attrParam.groups;

    CHECK(ExternalError::INVALID_VAL, gradOutputTensor.GetShape()[NCHW_N_IDX] == inputTensor.GetShape()[NCHW_N_IDX])
        << "Input illegal n size mismatch: grad_output n:" << gradOutputTensor.GetShape()[NCHW_N_IDX]
        << ", input n:" << inputTensor.GetShape()[NCHW_N_IDX] << ".";

    CHECK(ExternalError::INVALID_VAL, inputTensor.GetShape()[NCHW_C_IDX] == cinPerGroup * groups)
        << "Cin of input (" << inputTensor.GetShape()[NCHW_C_IDX]
        << ") is not equal to filterSize CinPerGroup * groups (" << cinPerGroup * groups << ").";

    CHECK(ExternalError::INVALID_VAL,
          gradOutputTensor.GetShape()[NCHW_H_IDX] == (inputTensor.GetShape()[NCHW_H_IDX] + attrParam.paddings[2] +
                                                      attrParam.paddings[3] - attrParam.dilations[1] * (kh - 1) - 1) /
                                                             attrParam.strides[1] +
                                                         1)
        << "Input illegal hout size:" << gradOutputTensor.GetShape()[NCHW_H_IDX] << ".";

    CHECK(ExternalError::INVALID_VAL,
          gradOutputTensor.GetShape()[NCHW_W_IDX] == (inputTensor.GetShape()[NCHW_W_IDX] + attrParam.paddings[4] +
                                                      attrParam.paddings[5] - attrParam.dilations[2] * (kw - 1) - 1) /
                                                             attrParam.strides[2] +
                                                         1)
        << "Input illegal wout size:" << gradOutputTensor.GetShape()[NCHW_W_IDX] << ".";
}

void FilterCheckLoad3dShape(DataType outType, const std::vector<int64_t>& filterSize, const ConvBpAttrParam& attrParam)
{
    ConvBp::CheckDimensionRange(attrParam.paddings, "paddings", 0, MAX_PAD_KERNEL);
    ConvBp::CheckDimensionRange(attrParam.dilations, "dilations", NUM1, MAX_DILATION_STRIDE);
    ConvBp::CheckDimensionRange(attrParam.strides, "strides", NUM1, MAX_DILATION_STRIDE);

    int64_t kh = filterSize[2];
    int64_t kw = filterSize[3];
    CHECK(ExternalError::OUT_OF_RANGE, kh <= MAX_PAD_KERNEL && kw <= MAX_PAD_KERNEL)
        << "Filter shapes do not satisfy Load3D's limits: kh=" << kh << ", kw=" << kw
        << ", which must be <= " << MAX_PAD_KERNEL << ".";

    int64_t k0 = ALIGN_SIZE_32 / BytesOf(outType); // 反向N轴=cin*kh*kw，K0(cin0)=32B/dtype
    CHECK(ExternalError::OUT_OF_RANGE, kh * kw * k0 <= SHAPE_INNER_AXIS_MAX_SIZE)
        << "Filter shapes do not satisfy Load3D's limits: kh*kw*k0=" << kh * kw * k0
        << "(k0 = 32B/dtypesize, cin0 block size), which must be <=" << SHAPE_INNER_AXIS_MAX_SIZE << ".";
}

void FilterCheckAttrShape(DataType outType, const Tensor& gradOutputTensor, const std::vector<int64_t>& filterSize,
                          const ConvBpAttrParam& attrParam)
{
    int64_t groups = attrParam.groups;
    int64_t coutGradOutput = gradOutputTensor.GetShape()[NCHW_C_IDX];
    int64_t coutFilter = filterSize[0];

    const std::vector<std::string> dimNames = {"H", "W"};
    for (size_t i = 0; i < filterSize.size() - NUM2; ++i) {
        int64_t filterVal = filterSize[filterSize.size() - i - 1];
        int64_t paddingLeft = attrParam.paddings[4 - i * NUM2];
        int64_t paddingRight = attrParam.paddings[5 - i * NUM2];
        int64_t dilationVal = attrParam.dilations[attrParam.dilations.size() - i - 1];
        int64_t dilatedFilterVal = (filterVal - 1) * dilationVal + 1;
        CHECK(ExternalError::INVALID_VAL, paddingLeft < dilatedFilterVal && paddingRight < dilatedFilterVal)
            << "The value of the " << dimNames[dimNames.size() - i - 1]
            << " dimension of filter must be > padding. Current filter value after dilation: " << dilatedFilterVal
            << ", padding value: " << paddingLeft << " and " << paddingRight << ".";
    }

    ConvBp::CheckValueRange(groups, "groups", NUM1, SHAPE_INNER_AXIS_MAX_SIZE);
    CHECK(ExternalError::INVALID_VAL, coutGradOutput == coutFilter)
        << "Cout of gradOutput (" << coutGradOutput << ") is not equal to Cout of filter (" << coutFilter << ").";
    CHECK(ExternalError::INVALID_VAL, coutGradOutput % groups == 0)
        << "Cout (" << coutGradOutput << ") is not divisible by groups (" << groups << ").";

    FilterCheckLoad3dShape(outType, filterSize, attrParam);
}

void FilterCheckTileTiling(DataType outType, const Tensor& gradOutputTensor, const std::vector<int64_t>& filterSize,
                           const ConvBpAttrParam& attrParam)
{
    auto& convTile = TileShape::Current().GetConvBpFilterTile();
    int64_t tileML0 = convTile.tileL0Info.tileML0;
    int64_t tileNL0 = convTile.tileL0Info.tileNL0;
    int64_t tileKL0 = convTile.tileL0Info.tileKL0;
    int64_t tileML1 = convTile.tileL1Info.tileML1;
    int64_t tileNL1 = convTile.tileL1Info.tileNL1;
    int64_t tileKL1 = convTile.tileL1Info.tileKL1;
    int64_t k0 = Conv::MKN_N_VALUE; // MMAD M/N/K block size = 16
    int64_t kh = filterSize[2];
    int64_t kw = filterSize[3];
    int64_t cin0 = ALIGN_SIZE_32 / BytesOf(outType); // N 轴(Cin*Kh*Kw)内块 = 32B/dtype
    int64_t wout = gradOutputTensor.GetShape(NCHW_W_IDX);

    ConvBp::CheckAlignment(tileML1, Conv::MKN_M_VALUE, "tileML1");
    ConvBp::CheckAlignment(tileNL1, cin0 * kh * kw, "tileNL1");
    // ConvBp::CheckAlignment(tileKL1, k0, "tileKL1");
    // K 轴按 Hout*Wout 切分，tileKL1 需为 wout 整数倍或小于 wout，保证 input 的 Load3D(im2col) 展开不跨 W 行
    CHECK(ExternalError::INVALID_VAL, (tileKL1 < wout) || (tileKL1 % wout == 0))
        << "tileKL1(" << tileKL1 << ") should be less than wout(" << wout << ") or be divisible by wout";

    ConvBp::CheckValueRange(tileML0, "tileML0", NUM1, ConvAlignB(tileML1, Conv::MKN_N_VALUE));
    ConvBp::CheckValueRange(tileKL0, "tileKL0", NUM1, ConvAlignB(tileKL1, k0));
    ConvBp::CheckValueRange(tileNL0, "tileNL0", NUM1, ConvAlignB(tileNL1, Conv::MKN_N_VALUE));
    ConvBp::CheckAlignment(tileML0, Conv::MKN_M_VALUE, "tileML0");
    ConvBp::CheckAlignment(tileKL0, k0, "tileKL0");
    ConvBp::CheckAlignment(tileNL0, Conv::MKN_N_VALUE, "tileNL0");

    // 循环块数上限校验（对齐 BpCheckL0TileTiling）
    int64_t cout = filterSize[0];
    int64_t cinPerGroup = filterSize[1];
    int64_t batch = gradOutputTensor.GetShape(NCHW_N_IDX);
    int64_t hout = gradOutputTensor.GetShape(NCHW_H_IDX);
    int64_t coutPerGroup = cout / attrParam.groups;
    int64_t numTileL0 = batch * attrParam.groups * CeilDiv(coutPerGroup, tileML0) *
                        CeilDiv(cinPerGroup * kh * kw, tileNL0);
    numTileL0 *= CeilDiv(hout * wout, tileKL0);
    if (numTileL0 > Conv::MAX_LOOP) {
        CONV_LOGW("Suggestion: Consider increasing tile size to reduce compilation time.");
    }

    Platform& platform = Platform::Instance();
    size_t l0aSize = platform.GetAICCore().GetMemorySize(MemoryType::MEM_L0A);
    size_t l0bSize = platform.GetAICCore().GetMemorySize(MemoryType::MEM_L0B);
    size_t l0cSize = platform.GetAICCore().GetMemorySize(MemoryType::MEM_L0C);
    ASSERT(ConvOperationError::OVER_BUFFER_LIMIT, tileML0 * tileKL0 * BytesOf(outType) <= l0aSize)
        << "Shape does not satisfy L0A load constraints, tileML0(" << tileML0 << ") × " << "tileKL0(" << tileKL0
        << ") × dtypesize ≤ L0A Size(" << l0aSize << ").";
    ASSERT(ConvOperationError::OVER_BUFFER_LIMIT, tileKL0 * tileNL0 * BytesOf(outType) <= l0bSize)
        << "Shape does not satisfy L0B load constraints, tileKL0(" << tileKL0 << ") × " << "tileNL0(" << tileNL0
        << ") × dtypesize ≤ L0B Size(" << l0bSize << ").";
    ASSERT(ConvOperationError::OVER_BUFFER_LIMIT, tileML0 * tileNL0 * BytesOf(DataType::DT_FP32) <= l0cSize)
        << "Shape does not satisfy L0C load constraints, tileML0(" << tileML0 << ") × " << "tileNL0(" << tileNL0
        << ") × dtypesize ≤ L0C Size(" << l0cSize << ").";
}

void FilterCheckL1SizeTiling(DataType outType, const Tensor& gradOutputTensor, const std::vector<int64_t>& filterSize,
                             const ConvBpAttrParam& attrParam)
{
    auto convTile = TileShape::Current().GetConvBpFilterTile();
    Platform& platform = Platform::Instance();
    uint64_t l1Size = platform.GetAIVCore().GetMemorySize(MemoryType::MEM_L1);

    uint64_t kh = filterSize[2];
    uint64_t kw = filterSize[3];
    uint64_t wout = gradOutputTensor.GetShape(NCHW_W_IDX);

    uint64_t tileML1 = convTile.tileL1Info.tileML1; // M = Cout
    uint64_t tileNL1 = convTile.tileL1Info.tileNL1; // N = Cin*Kh*Kw
    int64_t tileKL1 = convTile.tileL1Info.tileKL1;  // K = Hout*Wout

    uint64_t dilationH = attrParam.dilations[1];
    uint64_t dilationW = attrParam.dilations[2];
    uint64_t strideH = attrParam.strides[1];
    uint64_t strideW = attrParam.strides[2];

    // A operand (grad_output): [Cout, Hout*Wout]，无空间扩展
    uint64_t gradOutputL1Size = ConvAlignB(tileML1 * tileKL1 * BytesOf(outType), ALIGN_SIZE_32);

    // B operand (input, im2col 源): 空间按 stride/dilation 正向扩展
    uint64_t houtL1 = 1;
    uint64_t woutL1 = wout;
    if (tileKL1 % wout == 0) {
        houtL1 = tileKL1 / wout;
    } else {
        woutL1 = tileKL1;
    }
    uint64_t khDilated = (kh - 1) * dilationH + 1;
    uint64_t kwDilated = (kw - 1) * dilationW + 1;
    uint64_t hinL1 = (houtL1 - 1) * strideH + khDilated;
    uint64_t winL1 = (woutL1 - 1) * strideW + kwDilated;
    uint64_t cin1Cin0L1 = tileNL1 / (kh * kw);
    uint64_t inputL1Size = ConvAlignB(cin1Cin0L1 * hinL1 * winL1 * BytesOf(outType), ALIGN_SIZE_32);

    uint64_t minL1LoadSize = gradOutputL1Size + inputL1Size;
    ASSERT(ConvOperationError::OVER_BUFFER_LIMIT, minL1LoadSize <= l1Size)
        << "MinL1LoadSize > L1size, current MinL1LoadSize: " << minL1LoadSize << ", L1size: " << l1Size << ".";
}

void FilterCheckConvOperands(DataType outType, const Tensor& gradOutputTensor, const Tensor& inputTensor,
                             const std::vector<int64_t>& filterSize, const ConvBpAttrParam& attrParam)
{
    CHECK(ExternalError::INVALID_TYPE,
          outType == DataType::DT_FP32 || outType == DataType::DT_FP16 || outType == DataType::DT_BF16)
        << "Unsupported output data type. Only DT_FP32, DT_FP16, DT_BF16 are supported.";
    FilterCheckOriginShape(gradOutputTensor, inputTensor, filterSize);
    FilterCheckOutputShape(gradOutputTensor, inputTensor, filterSize, attrParam);
    FilterCheckAttrShape(outType, gradOutputTensor, filterSize, attrParam);
    FilterCheckTileTiling(outType, gradOutputTensor, filterSize, attrParam);
    FilterCheckL1SizeTiling(outType, gradOutputTensor, filterSize, attrParam);
}

std::vector<int64_t> FilterGetResTensorShape(DataType outType, const std::vector<int64_t>& filterSize,
                                             const ConvBpAttrParam& convAttrParam)
{
    int64_t cout = filterSize[0];
    int64_t cinPerGroup = filterSize[1];
    int64_t kh = filterSize[2];
    int64_t kw = filterSize[3];
    int64_t cout0 = Conv::MKN_N_VALUE;
    int64_t cin0 = ALIGN_SIZE_32 / BytesOf(outType);
    int64_t coutPerGroup = cout / convAttrParam.groups;
    int64_t cout1PerGroup = CeilDiv(coutPerGroup, cout0);
    int64_t cin1PerGroup = CeilDiv(cinPerGroup, cin0);

    return {convAttrParam.groups * cin1PerGroup * kh * kw, cout1PerGroup, cout0, cin0};
}

std::vector<LogicalTensorPtr> FilterGetOperandVecIn(std::vector<LogicalTensorPtr> operandVecIn,
                                                    const ConvBpAttrParam& convAttrParam,
                                                    const std::vector<int64_t>& filterSize)
{
    int64_t cin0 = ALIGN_SIZE_32 / BytesOf(operandVecIn[FILTER_INPUT_IDX]->Datatype());
    int64_t batch = operandVecIn[FILTER_GRAD_OUTPUT_IDX]->GetShape()[NCHW_N_IDX];
    int64_t ho = operandVecIn[FILTER_GRAD_OUTPUT_IDX]->GetShape()[NCHW_H_IDX];
    int64_t wo = operandVecIn[FILTER_GRAD_OUTPUT_IDX]->GetShape()[NCHW_W_IDX];
    int64_t hi = operandVecIn[FILTER_INPUT_IDX]->GetShape()[NCHW_H_IDX];
    int64_t wi = operandVecIn[FILTER_INPUT_IDX]->GetShape()[NCHW_W_IDX];
    int64_t cout = filterSize[0];
    int64_t cinPerGroup = filterSize[1];
    int64_t cout0 = Conv::MKN_N_VALUE;
    int64_t coutPerGroup = cout / convAttrParam.groups;
    int64_t cout1PerGroup = CeilDiv(coutPerGroup, cout0);
    int64_t cin1PerGroup = CeilDiv(cinPerGroup, cin0);

    std::vector<int64_t> gradoutputNzShape = {batch, convAttrParam.groups * cout1PerGroup, ho, wo, cout0};
    std::vector<int64_t> inputNzShape = {batch, convAttrParam.groups * cin1PerGroup, hi, wi, cin0};

    Tensor gradoutputNzTensor(operandVecIn[FILTER_GRAD_OUTPUT_IDX]->Datatype(), gradoutputNzShape, "TensorGradoutputNz",
                              TileOpFormat::TILEOP_NC1HWC0);
    Tensor inputNzTensor(operandVecIn[FILTER_INPUT_IDX]->Datatype(), inputNzShape, "TensorInputNz",
                         TileOpFormat::TILEOP_NC1HWC0);
    return {gradoutputNzTensor.GetStorage(), inputNzTensor.GetStorage()};
}

Tensor FilterGetFinalResTensorFZ2NCHW(const Tensor& resTensor, const std::vector<int64_t>& filterSize)
{
    // FRACTAL_Z -> NCHW: dW 与权重同形 [Cout, CinPerGroup, Kh, Kw] = filterSize
    std::vector<int64_t> orgOutShape = filterSize;
    Tensor finalResTensor(resTensor.GetStorage()->Datatype(), orgOutShape, "TensorGradWeight");
    finalResTensor.GetStorage()->UpdateDynValidShape(SymbolicScalar::FromConcrete(orgOutShape));
    return finalResTensor;
}

void FilterSetTensorOpAttr(Operation& op, const LogicalTensorPtr& gradOutputTensor, const LogicalTensorPtr& inputTensor,
                           const LogicalTensorPtr& resTensor, const ConvBpAttrParam& convAttrParam)
{
    op.SetAttribute(CONV_GROUPS_ATTR, convAttrParam.groups);
    op.SetAttribute(CONV_PADDINGS_ATTR, convAttrParam.paddings);
    op.SetAttribute(CONV_STRIDES_ATTR, convAttrParam.strides);
    op.SetAttribute(CONV_DILATIONS_ATTR, convAttrParam.dilations);
    op.SetAttribute(CONV_3D_FLAG, convAttrParam.isConv3D);
    op.SetAttribute(CONV_ORI_GRAD_OUTPUT_SHAPE_ATTR, gradOutputTensor->GetShape());
    op.SetAttribute(CONV_ORI_INPUT_SHAPE_ATTR, inputTensor->GetShape());
    op.SetAttribute(CONV_ORI_RES_SHAPE_ATTR, resTensor->GetShape());
    op.SetAttribute("dynamicResValidShape", resTensor->GetDynValidShape());
}

Tensor FilterConstructTensorGraphNZ2NZ(Function* functionPtr, std::vector<LogicalTensorPtr> operandVecIn,
                                       const Tensor& resTensor, const ConvBpAttrParam& convAttrParam,
                                       const std::vector<int64_t>& filterSize)
{
    std::vector<LogicalTensorPtr> operandVecOut = {resTensor.GetStorage()};
    std::vector<LogicalTensorPtr> operandVecInNZ = FilterGetOperandVecIn(operandVecIn, convAttrParam, filterSize);

    auto& gradoutputTransOp = functionPtr->AddOperation(Opcode::OP_FAKE_TRANS, {operandVecIn[FILTER_GRAD_OUTPUT_IDX]},
                                                        {operandVecInNZ[FILTER_GRAD_OUTPUT_IDX]});
    gradoutputTransOp.SetAttribute(FAKE_TRANS_IN_FORMAT_ATTR,
                                   static_cast<int64_t>(operandVecIn[FILTER_GRAD_OUTPUT_IDX]->Format()));
    gradoutputTransOp.SetAttribute(FAKE_TRANS_OUT_FORMAT_ATTR,
                                   static_cast<int64_t>(operandVecInNZ[FILTER_GRAD_OUTPUT_IDX]->Format()));

    auto& inputTransOp = functionPtr->AddOperation(Opcode::OP_FAKE_TRANS, {operandVecIn[FILTER_INPUT_IDX]},
                                                   {operandVecInNZ[FILTER_INPUT_IDX]});
    inputTransOp.SetAttribute(FAKE_TRANS_IN_FORMAT_ATTR,
                              static_cast<int64_t>(operandVecIn[FILTER_INPUT_IDX]->Format()));
    inputTransOp.SetAttribute(FAKE_TRANS_OUT_FORMAT_ATTR,
                              static_cast<int64_t>(operandVecInNZ[FILTER_INPUT_IDX]->Format()));

    auto& op = functionPtr->AddOperation(Opcode::OP_CONV_BP_FILTER_2D, operandVecInNZ, operandVecOut);

    Tensor finalResTensor = FilterGetFinalResTensorFZ2NCHW(resTensor, filterSize);
    auto& orgResOp = functionPtr->AddOperation(Opcode::OP_FAKE_TRANS, operandVecOut, {finalResTensor.GetStorage()});
    orgResOp.SetAttribute(FAKE_TRANS_IN_FORMAT_ATTR, static_cast<int64_t>(resTensor.Format()));
    orgResOp.SetAttribute(FAKE_TRANS_OUT_FORMAT_ATTR, static_cast<int64_t>(finalResTensor.Format()));
    FilterSetTensorOpAttr(op, operandVecIn[FILTER_GRAD_OUTPUT_IDX], operandVecIn[FILTER_INPUT_IDX],
                          finalResTensor.GetStorage(), convAttrParam);

    return finalResTensor;
}

Tensor FilterConstructTensorGraph(const Tensor& gradOutputTensor, const Tensor& inputTensor, const Tensor& resTensor,
                                  ConvBpAttrParam& convAttrParam, const std::vector<int64_t>& filterSize)
{
    Function* functionPtr = Program::GetInstance().GetCurrentFunction();
    ASSERT(ConvExpandFuncError::EXPANDFUNC_TILE_OP_NULLPTR, functionPtr != nullptr) << "functionPtr is nullptr.";
    std::vector<LogicalTensorPtr> operandVecIn = {gradOutputTensor.GetStorage(), inputTensor.GetStorage()};
    return FilterConstructTensorGraphNZ2NZ(functionPtr, operandVecIn, resTensor, convAttrParam, filterSize);
}

Tensor ConvBackwardFilter(DataType outType, const Tensor& gradOutputTensor, const Tensor& inputTensor,
                          const std::vector<int64_t>& filterSize, const std::vector<int64_t>& strides,
                          const std::vector<int64_t>& paddings, const std::vector<int64_t>& dilations,
                          const int64_t groups)
{
    CHECK(FwkErr::PLATFORM_NOT_SUPPORTED, Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_2201)
        << "Only support A2/A3 Platform";
    // 首版仅支持 2D 卷积
    CHECK(ExternalError::INVALID_VAL, gradOutputTensor.Dim() == 4 && inputTensor.Dim() == 4 && filterSize.size() == 4)
        << "Input illegal dim, ConvBackwardFilter only supports 2D conv: grad_output dim:" << gradOutputTensor.Dim()
        << ", input dim:" << inputTensor.Dim() << ", filterSize size:" << filterSize.size() << ".";
    CHECK(ExternalError::INVALID_VAL, dilations.size() == 2 && strides.size() == 2 && paddings.size() == 4)
        << "Input illegal dilations size:" << dilations.size() << ", strides size:" << strides.size()
        << ", paddings size:" << paddings.size() << ", which must be 2, 2, 4 for 2D conv.";

    // 归一化为 3D 语义 (前补 D 维)，与 ConvBackwardInput 保持一致
    std::vector<int64_t> finalPaddings = paddings;
    std::vector<int64_t> finalDilations = dilations;
    std::vector<int64_t> finalStrides = strides;
    finalStrides.insert(finalStrides.begin(), 1, 1);
    finalDilations.insert(finalDilations.begin(), 1, 1);
    finalPaddings.insert(finalPaddings.begin(), 2, 0);

    ConvBpAttrParam convAttrParam(finalPaddings, finalStrides, finalDilations, groups);
    convAttrParam.isConv1D = false;
    convAttrParam.isConv3D = false;

    FilterCheckConvOperands(outType, gradOutputTensor, inputTensor, filterSize, convAttrParam);
    std::vector<int64_t> resTensorShape = FilterGetResTensorShape(outType, filterSize, convAttrParam);

    Tensor resTensor(outType, resTensorShape, "TensorW", TileOpFormat::TILEOP_FRACTAL_Z);
    resTensor.GetStorage()->UpdateDynValidShape(SymbolicScalar::FromConcrete(resTensorShape));
    return FilterConstructTensorGraph(gradOutputTensor, inputTensor, resTensor, convAttrParam, filterSize);
}

} // namespace ConvBp
} // namespace tile_fwk
} // namespace npu
