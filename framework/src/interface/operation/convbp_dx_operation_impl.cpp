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
 * \file convbp_dx_operation_impl.cpp
 * \brief ConvBackwardInput operation implementation
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

const std::string LoadStoreConvBpOpAttributeKey::copyInMode = "COPY_IN_MODE";
const std::string LoadStoreConvBpOpAttributeKey::copyOutMode = "COPY_OUT_MODE";
const std::string LoadStoreConvBpOpAttributeKey::isGradOutput = "IS_GRAD_OUTPUT";

SymbolicScalar BpGetValidInputChannels(const Tensor& weightTensor, const int64_t groups)
{
    SymbolicScalar cinPerGroup = weightTensor.GetValidShape()[NCHW_C_IDX];
    return cinPerGroup * groups;
}

void CheckDimensionRange(const std::vector<int64_t>& vec, const std::string& name, int minVal, int maxVal)
{
    for (size_t i = 0; i < vec.size(); ++i) {
        CHECK(ExternalError::OUT_OF_RANGE, vec[i] >= minVal && vec[i] <= maxVal)
            << "The value of the " << i << "-th dimension of " << name << " must be in the range [" << minVal << ","
            << maxVal << "]. Current value: " << vec[i] << ".";
    }
}

void CheckValueRange(int64_t value, const std::string& name, int64_t min, int64_t max, const std::string& formula = "")
{
    std::ostringstream oss;
    oss << "Invalid " << name << ":" << value << ", expected range [" << min << "," << max << "].";
    if (!formula.empty()) {
        oss << "Formula: " << formula;
    }
    CHECK(ExternalError::OUT_OF_RANGE, value >= min && value <= max) << oss.str();
}

void CheckAlignment(int64_t value, int64_t alignment, const std::string& valueName)
{
    CHECK(ExternalError::INVALID_VAL, alignment != 0) << "Error in alignment check for " << valueName << ".";
    CHECK(ExternalError::INVALID_VAL, value % alignment == 0)
        << "Invalid " << valueName << ":" << value << ", requires " << alignment << "-element alignment.";
}

int64_t ConvAlignB(int64_t a, int64_t b)
{
    if (b == 0) {
        return 0;
    }
    return ((a + b - 1) / b) * b;
}

void CheckLoad3dShape(const Tensor& weightTensor, const ConvBpAttrParam& attrParam)
{
    CheckDimensionRange(attrParam.paddings, "paddings", 0, MAX_PAD_KERNEL);
    CheckDimensionRange(attrParam.dilations, "dilations", NUM1, MAX_DILATION_STRIDE);
    CheckDimensionRange(attrParam.strides, "strides", NUM1, MAX_DILATION_STRIDE);

    uint32_t indexH = attrParam.isConv3D ? NCDHW_H_IDX : NCHW_H_IDX;
    uint32_t indexW = attrParam.isConv3D ? NCDHW_W_IDX : (attrParam.isConv1D ? NCL_L_IDX : NCHW_W_IDX);
    int64_t kw = weightTensor.GetShape()[indexW];
    int64_t kh = attrParam.isConv1D ? 1 : weightTensor.GetShape()[indexH];
    CHECK(ExternalError::OUT_OF_RANGE, kh <= MAX_PAD_KERNEL && kw <= MAX_PAD_KERNEL)
        << "Weight shapes do not satisfy Load3D's" << (attrParam.isConv1D ? " limit: kw=" : " limits: kh=")
        << (attrParam.isConv1D ? kw : kh) << (attrParam.isConv1D ? "" : ", kw=" + std::to_string(kw))
        << ", which must be <= " << MAX_PAD_KERNEL << ".";

    int64_t k0 = Conv::MKN_N_VALUE; // 反向K轴=cout*kh*kw，K0(cout0)=16
    CHECK(ExternalError::OUT_OF_RANGE, kh * kw * k0 <= SHAPE_INNER_AXIS_MAX_SIZE)
        << "Weight shapes do not satisfy Load3D's limits: kh*kw*k0=" << kh * kw * k0
        << "(k0 = 16, mmad K block size), which must be <=" << SHAPE_INNER_AXIS_MAX_SIZE << ".";
}

void BpCheckOriginShape(const Tensor& gradOutputTensor, const std::vector<int64_t>& inputSize,
                        const Tensor& weightTensor, const Tensor& biasTensor)
{
    CheckDimensionRange(gradOutputTensor.GetShape(), "grad_output", NUM1, MAX_SIZE);
    CheckDimensionRange(weightTensor.GetShape(), "weight", NUM1, MAX_SIZE);
    CheckDimensionRange(inputSize, "input", NUM1, MAX_SIZE);

    if (biasTensor.IsEmpty()) {
        return;
    }
    int64_t cin = weightTensor.GetShape(1); // NCL/NCHW/NCDHW
    CHECK(ExternalError::INVALID_VAL, biasTensor.GetShape(0) == cin)
        << "Input illegal bias shape:" << biasTensor.GetShape(0) << ", which must equal Cin:" << cin << ".";
}

void BpCheckOutputShape(const Tensor& gradOutputTensor, const std::vector<int64_t>& inputSize,
                        const Tensor& weightTensor, const ConvBpAttrParam& attrParam)
{
    int64_t dimSize = inputSize.size();

    CHECK(ExternalError::INVALID_VAL, gradOutputTensor.GetShape()[0] == inputSize[0])
        << "Input illegal n size:" << inputSize[0] << ".";

    CHECK(ExternalError::INVALID_VAL, gradOutputTensor.GetShape()[dimSize - 1] ==
                                          (inputSize[dimSize - 1] + attrParam.paddings[4] + attrParam.paddings[5] -
                                           attrParam.dilations[2] * (weightTensor.GetShape()[dimSize - 1] - 1) - 1) /
                                                  attrParam.strides[2] +
                                              1)
        << "Input illegal win size:" << inputSize[dimSize - 1] << ".";

    if (attrParam.isConv3D) {
        CHECK(ExternalError::INVALID_VAL,
              gradOutputTensor.GetShape()[2] == (inputSize[2] + attrParam.paddings[0] + attrParam.paddings[1] -
                                                 attrParam.dilations[0] * (weightTensor.GetShape()[2] - 1) - 1) /
                                                        attrParam.strides[0] +
                                                    1)
            << "Input illegal din size:" << inputSize[2] << ".";
    }
    if (!attrParam.isConv1D) {
        CHECK(ExternalError::INVALID_VAL, gradOutputTensor.GetShape()[dimSize - 2] ==
                                              (inputSize[dimSize - 2] + attrParam.paddings[2] + attrParam.paddings[3] -
                                               attrParam.dilations[1] * (weightTensor.GetShape()[dimSize - 2] - 1) -
                                               1) / attrParam.strides[1] +
                                                  1)
            << "Input illegal hin size:" << inputSize[dimSize - 2] << ".";
    }
}

void BpCheckAttrShape(const Tensor& gradOutputTensor, const Tensor& weightTensor, const ConvBpAttrParam& attrParam)
{
    int64_t groups = attrParam.groups;
    int64_t coutGradOutput = gradOutputTensor.GetShape()[1];
    int64_t coutWeight = weightTensor.GetShape()[0];

    const std::vector<std::string> dimNames = attrParam.isConv1D ? std::vector<std::string>{"L"} :
                                              attrParam.isConv3D ? std::vector<std::string>{"D", "H", "W"} :
                                                                   std::vector<std::string>{"H", "W"};
    for (size_t i = 0; i < weightTensor.Dim() - NUM2; ++i) {
        int64_t weightVal = weightTensor.GetShape(weightTensor.Dim() - i - 1);
        int64_t paddingLeft = attrParam.paddings[4 - i * NUM2];
        int64_t paddingRight = attrParam.paddings[5 - i * NUM2];
        int64_t dilationVal = attrParam.dilations[attrParam.dilations.size() - i - 1];
        int64_t dilatedWeightVal = (weightVal - 1) * dilationVal + 1;
        CHECK(ExternalError::INVALID_VAL, paddingLeft < dilatedWeightVal && paddingRight < dilatedWeightVal)
            << "The value of the " << dimNames[dimNames.size() - i - 1]
            << " dimension of weight must be > padding. Current weight value after dilation: " << dilatedWeightVal
            << ", padding value: " << paddingLeft << " and " << paddingRight << ".";
    }
    // weight [cout, cin/groups, Dk, Hk, Wk]
    CheckValueRange(groups, "groups", NUM1, SHAPE_INNER_AXIS_MAX_SIZE);
    CHECK(ExternalError::INVALID_VAL, coutGradOutput == coutWeight)
        << "Cout of gradOutput (" << coutGradOutput << ") is not equal to Cout of weight (" << coutWeight << ").";
    CHECK(ExternalError::INVALID_VAL, coutGradOutput % groups == 0)
        << "Cout (" << coutGradOutput << ") is not divisible by groups (" << groups << ").";

    CheckLoad3dShape(weightTensor, attrParam);
}

void BpCheckL0TileTiling(DataType outType, const ConvBpAttrParam& attrParam, const Tensor& weightTensor,
                         const Tensor& gradOutputTensor, const std::vector<int64_t>& inputSize)
{
    auto& convTile = TileShape::Current().GetConvBpInputTile();
    int64_t tileML0 = convTile.tileL0Info.tileML0;
    int64_t tileNL0 = convTile.tileL0Info.tileNL0;
    int64_t tileKL0 = convTile.tileL0Info.tileKL0;
    int64_t tileML1 = convTile.tileL1Info.tileML1;
    int64_t tileNL1 = convTile.tileL1Info.tileNL1;
    int64_t k0 = Conv::MKN_N_VALUE; // 反向K轴=cout*kh*kw，K0=16(MMK的K块大小)
    int64_t tileKL1 = convTile.tileL1Info.tileKL1;

    uint32_t indexH = attrParam.isConv3D ? NCDHW_H_IDX : NCHW_H_IDX;
    uint32_t indexW = attrParam.isConv3D ? NCDHW_W_IDX : (attrParam.isConv1D ? NCL_L_IDX : NCHW_W_IDX);
    int64_t kh = attrParam.isConv1D ? 1 : weightTensor.GetShape()[indexH];
    int64_t kw = weightTensor.GetShape()[indexW];
    int64_t cout = gradOutputTensor.GetShape(1);
    int64_t cinPerGroup = weightTensor.GetShape(1);
    int64_t kAlignL1 = ConvAlignB(tileKL1, k0);
    int64_t oriK = ConvAlignB(cout, k0) * kh * kw;
    int64_t batch = gradOutputTensor.GetShape(0);
    int64_t groups = attrParam.groups;
    int64_t hin = attrParam.isConv1D ? 1 : inputSize[indexH];
    int64_t win = inputSize[indexW];
    int64_t numTileL0 = batch * groups * CeilDiv(cinPerGroup, tileNL0) * CeilDiv(hin * win, tileML0);
    if (attrParam.isConv3D) {
        int64_t kd = weightTensor.GetShape(NCDHW_D_IDX);
        int64_t din = inputSize[NCDHW_D_IDX];
        numTileL0 *= din;
        kAlignL1 *= kd;
        oriK *= kd;
    }
    if (numTileL0 * CeilDiv(oriK, tileKL0) > Conv::MAX_LOOP) {
        CONV_LOGW("Suggestion: Consider increasing tile size to reduce compilation time.");
    }

    CheckValueRange(tileML0, "tileML0", NUM1, ConvAlignB(tileML1, Conv::MKN_N_VALUE));
    CheckValueRange(tileKL0, "tileKL0", NUM1, kAlignL1);
    CheckValueRange(tileNL0, "tileNL0", NUM1, ConvAlignB(tileNL1, Conv::MKN_N_VALUE));
    CheckAlignment(tileML0, Conv::MKN_M_VALUE, "tileML0");
    CheckAlignment(tileKL0, k0, "tileKL0");
    CheckAlignment(tileNL0, Conv::MKN_N_VALUE, "tileNL0");
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

void BpCheckTileTiling(DataType outType, const Tensor& gradOutputTensor, const Tensor& weightTensor,
                       const ConvBpAttrParam& attrParam, const std::vector<int64_t>& inputSize)
{
    auto convTile = TileShape::Current().GetConvBpInputTile();
    int64_t tileML1 = convTile.tileL1Info.tileML1;
    int64_t tileNL1 = convTile.tileL1Info.tileNL1;
    int64_t tileKL1 = convTile.tileL1Info.tileKL1;
    CheckAlignment(tileNL1, Conv::MKN_N_VALUE, "tileNL1");
    int64_t k0 = Conv::MKN_N_VALUE; // 反向K轴=cout*kh*kw，K0=16
    CheckAlignment(tileKL1, k0, "tileKL1");
    uint32_t indexW = attrParam.isConv3D ? NCDHW_W_IDX : (attrParam.isConv1D ? NCL_L_IDX : NCHW_W_IDX);
    uint32_t indexH = attrParam.isConv3D ? NCDHW_H_IDX : NCHW_H_IDX;
    int64_t win = inputSize[indexW];
    int64_t kh = attrParam.isConv1D ? 1 : weightTensor.GetShape()[indexH];
    int64_t kw = weightTensor.GetShape()[indexW];

    CHECK(ExternalError::INVALID_VAL, (tileML1 < win) || (tileML1 % win == 0))
        << "tileML1(" << tileML1 << ") should be less than win(" << win << ") or be divisible by win";
    CHECK(ExternalError::INVALID_VAL, tileKL1 % (k0 * kh * kw) == 0)
        << "tileKL1(" << tileKL1 << ") should be divisible by Cout0 * Hk * Wk";

    BpCheckL0TileTiling(outType, attrParam, weightTensor, gradOutputTensor, inputSize);
}

void BpCheckL1SizeTiling(DataType outType, const Tensor& gradOutputTensor, const Tensor& weightTensor,
                         const Tensor& biasTensor, const ConvBpAttrParam& attrParam,
                         const std::vector<int64_t>& inputSize)
{
    auto convTile = TileShape::Current().GetConvBpInputTile();
    Platform& platform = Platform::Instance();
    uint64_t l1Size = platform.GetAIVCore().GetMemorySize(MemoryType::MEM_L1); // todo check AIV MEM L1?
    uint32_t indexH = attrParam.isConv3D ? NCDHW_H_IDX : NCHW_H_IDX;
    uint32_t indexW = attrParam.isConv3D ? NCDHW_W_IDX : (attrParam.isConv1D ? NCL_L_IDX : NCHW_W_IDX);

    uint64_t kh = attrParam.isConv1D ? 1 : weightTensor.GetShape(indexH);
    uint64_t kw = weightTensor.GetShape(indexW);
    uint64_t hout = attrParam.isConv1D ? 1 : gradOutputTensor.GetShape(indexH);
    uint64_t wout = gradOutputTensor.GetShape(indexW);
    uint64_t k0 = Conv::MKN_N_VALUE; // 反向K轴=cout*kh*kw，K0=16
    int64_t win = inputSize[indexW];

    uint64_t biasL1Size = 0;
    uint64_t tileNL1 = convTile.tileL1Info.tileNL1;
    if (!biasTensor.IsEmpty()) {
        biasL1Size = ConvAlignB(tileNL1 * BytesOf(outType), ALIGN_SIZE_32);
    }

    int64_t tileKL1 = convTile.tileL1Info.tileKL1;
    uint64_t kBL1 = ConvAlignB(tileKL1, k0);
    uint64_t weightL1Size = ConvAlignB(kBL1 * tileNL1 * BytesOf(outType), ALIGN_SIZE_32);

    uint64_t dilationH = attrParam.dilations[1];
    uint64_t dilationW = attrParam.dilations[2];
    uint64_t strideH = attrParam.strides[1];
    uint64_t strideW = attrParam.strides[2];
    uint64_t gradOutputL1Size = 0;
    uint64_t tileML1 = convTile.tileL1Info.tileML1;
    uint64_t hinL1 = 1;
    uint64_t winL1 = win;
    if (tileML1 % win == 0) {
        hinL1 = tileML1 / win;
    } else { // BpCheckTileTiling已校验过tileML1是win的倍数或小于win
        winL1 = tileML1;
    }
    uint64_t khDilated = (kh - 1) * dilationH + 1;
    uint64_t houtExpandL1 = std::min(hinL1 + khDilated - 1, (hout - 1) * strideH + 1);
    uint64_t kwDilated = (kw - 1) * dilationW + 1;
    uint64_t woutExpandL1 = std::min(winL1 + kwDilated - 1, (wout - 1) * strideW + 1);

    uint64_t gradOutputL1KSize = static_cast<uint64_t>(tileKL1) / (kh * kw);
    gradOutputL1Size = ConvAlignB(houtExpandL1 * woutExpandL1 * gradOutputL1KSize * BytesOf(outType), ALIGN_SIZE_32);
    uint64_t minL1LoadSize = biasL1Size + gradOutputL1Size + weightL1Size;
    ASSERT(ConvOperationError::OVER_BUFFER_LIMIT, minL1LoadSize <= l1Size)
        << "MinL1LoadSize > L1size, current MinL1LoadSize: " << minL1LoadSize << ", L1size: " << l1Size << ".";
}

void BpCheckConvOperands(DataType outType, const Tensor& gradOutputTensor, const std::vector<int64_t>& inputSize,
                         const Tensor& weightTensor, const Tensor& biasTensor, ConvBpAttrParam& attrParam)
{
    CHECK(ExternalError::INVALID_TYPE,
          outType == DataType::DT_FP32 || outType == DataType::DT_FP16 || outType == DataType::DT_BF16)
        << "Unsupported output data type. Only DT_FP32, DT_FP16, DT_BF16 are supported.";
    BpCheckOriginShape(gradOutputTensor, inputSize, weightTensor, biasTensor);
    BpCheckAttrShape(gradOutputTensor, weightTensor, attrParam);
    BpCheckOutputShape(gradOutputTensor, inputSize, weightTensor, attrParam);
    BpCheckTileTiling(outType, gradOutputTensor, weightTensor, attrParam, inputSize);
    BpCheckL1SizeTiling(outType, gradOutputTensor, weightTensor, biasTensor, attrParam, inputSize);
}

std::vector<int64_t> BpGetResTensorShape(DataType outType, const std::vector<int64_t>& inputSize,
                                         const ConvBpAttrParam& convAttrParam)
{
    int64_t batch = inputSize[0];
    int64_t cin = inputSize[1];
    int64_t di = 1;
    int64_t hi = 1;
    int64_t wi = inputSize.back();
    int64_t cin0 = ALIGN_SIZE_32 / BytesOf(outType);
    int64_t cin1 = convAttrParam.groups * CeilDiv(cin / convAttrParam.groups, cin0);

    std::vector<int64_t> resTensorShape;
    if (!convAttrParam.isConv1D) {
        hi = inputSize[inputSize.size() - 2];
    }

    resTensorShape = {batch, cin1, hi, wi, cin0};
    if (convAttrParam.isConv3D) {
        di = inputSize[2];
        resTensorShape = {batch, di, cin1, hi, wi, cin0};
    }

    return resTensorShape;
}

void BpSetTensorOpAttr(Operation& op, const LogicalTensorPtr& gradOutpuTensor, const LogicalTensorPtr& weightTensor,
                       const LogicalTensorPtr& resTensor, const ConvBpAttrParam& convAttrParam)
{
    op.SetAttribute(CONV_BIAS_ATTR, convAttrParam.hasBias);
    op.SetAttribute(CONV_GROUPS_ATTR, convAttrParam.groups);
    op.SetAttribute(CONV_PADDINGS_ATTR, convAttrParam.paddings);
    op.SetAttribute(CONV_STRIDES_ATTR, convAttrParam.strides);
    op.SetAttribute(CONV_DILATIONS_ATTR, convAttrParam.dilations);
    op.SetAttribute(CONV_3D_FLAG, convAttrParam.isConv3D);
    op.SetAttribute(CONV_ORI_GRAD_OUTPUT_SHAPE_ATTR, gradOutpuTensor->GetShape());
    op.SetAttribute(CONV_ORI_WEIGHT_SHAPE_ATTR, weightTensor->GetShape());
    op.SetAttribute(CONV_ORI_RES_SHAPE_ATTR, resTensor->GetShape());
    op.SetAttribute("dynamicResValidShape", resTensor->GetDynValidShape());
}

std::vector<LogicalTensorPtr> BpGetOperandVecIn(std::vector<LogicalTensorPtr> operandVecIn,
                                                const ConvBpAttrParam& convAttrParam)
{
    // operandVecIn Conv1D/2D NCHW, Conv3D NCDHW
    int64_t cin0 = ALIGN_SIZE_32 / BytesOf(operandVecIn[INPUT_GRAD_OUTPUT_IDX]->Datatype());
    int64_t batch = operandVecIn[INPUT_GRAD_OUTPUT_IDX]->GetShape()[NCDHW_N_IDX];
    int64_t ho = convAttrParam.isConv3D ? operandVecIn[INPUT_GRAD_OUTPUT_IDX]->GetShape()[NCDHW_H_IDX] :
                                          operandVecIn[INPUT_GRAD_OUTPUT_IDX]->GetShape()[NCHW_H_IDX];
    int64_t wo = convAttrParam.isConv3D ? operandVecIn[INPUT_GRAD_OUTPUT_IDX]->GetShape()[NCDHW_W_IDX] :
                                          operandVecIn[INPUT_GRAD_OUTPUT_IDX]->GetShape()[NCHW_W_IDX];
    int64_t cout = operandVecIn[INPUT_WEIGHT_IDX]->GetShape()[NCDHW_N_IDX];
    int64_t kh = convAttrParam.isConv3D ? operandVecIn[INPUT_WEIGHT_IDX]->GetShape()[NCDHW_H_IDX] :
                                          operandVecIn[INPUT_WEIGHT_IDX]->GetShape()[NCHW_H_IDX];
    int64_t kw = convAttrParam.isConv3D ? operandVecIn[INPUT_WEIGHT_IDX]->GetShape()[NCDHW_W_IDX] :
                                          operandVecIn[INPUT_WEIGHT_IDX]->GetShape()[NCHW_W_IDX];
    int64_t cinPerGroup = operandVecIn[INPUT_WEIGHT_IDX]->GetShape()[NCDHW_C_IDX];
    int64_t cin1PerGroup = CeilDiv(cinPerGroup, cin0);
    int64_t cout1PerGroup = CeilDiv(cout / convAttrParam.groups, Conv::MKN_N_VALUE);

    TileOpFormat gradoutputNzFormat = TileOpFormat::TILEOP_NC1HWC0;
    TileOpFormat weightFzFormat = TileOpFormat::TILEOP_FRACTAL_Z;
    std::vector<int64_t> gradoutputNzShape = {batch, convAttrParam.groups * cout1PerGroup, ho, wo, Conv::MKN_N_VALUE};
    std::vector<int64_t> weightFzShape = {convAttrParam.groups * cin1PerGroup * kh * kw, cout1PerGroup,
                                          Conv::MKN_N_VALUE, cin0};
    if (convAttrParam.isConv3D) {
        gradoutputNzFormat = TileOpFormat::TILEOP_NDC1HWC0;
        weightFzFormat = TileOpFormat::TILEOP_FRACTAL_Z_3D;
        int64_t dout = operandVecIn[INPUT_GRAD_OUTPUT_IDX]->GetShape()[NCDHW_D_IDX];
        int64_t kd = operandVecIn[INPUT_WEIGHT_IDX]->GetShape()[NCDHW_D_IDX];
        gradoutputNzShape = {batch, dout, convAttrParam.groups * cout1PerGroup, ho, wo, Conv::MKN_N_VALUE};
        weightFzShape = {convAttrParam.groups * kd * cin1PerGroup * kh * kw, cout1PerGroup, Conv::MKN_N_VALUE, cin0};
    }

    Tensor gradoutputNzTensor(operandVecIn[INPUT_GRAD_OUTPUT_IDX]->Datatype(), gradoutputNzShape, "TensorGradoutputNz",
                              gradoutputNzFormat);
    Tensor weightFzTensor(operandVecIn[INPUT_WEIGHT_IDX]->Datatype(), weightFzShape, "TensorWeightFz", weightFzFormat);
    return {gradoutputNzTensor.GetStorage(), weightFzTensor.GetStorage()};
}

Tensor BpGetFinalResTensorNZ2NZ(std::vector<LogicalTensorPtr> operandVecIn, const Tensor& resTensor,
                                const ConvBpAttrParam& convAttrParam)
{
    // {batch, (Din,) Cin1, Hin, Win, Cin0}  >  {batch, Cin, (Din,) Hin, Win}
    std::vector<int64_t> resShape = resTensor.GetShape();

    std::vector<int64_t> orgOutShape = {resShape[NC1HWC0_N_IDX],
                                        operandVecIn[INPUT_WEIGHT_IDX]->GetShape()[NCHW_C_IDX] * convAttrParam.groups,
                                        resShape[NC1HWC0_H_IDX], resShape[NC1HWC0_W_IDX]};
    if (convAttrParam.isConv3D) {
        orgOutShape = {resShape[NDC1HWC0_N_IDX],
                       operandVecIn[INPUT_WEIGHT_IDX]->GetShape()[NCHW_C_IDX] * convAttrParam.groups,
                       resShape[NDC1HWC0_D_IDX], resShape[NDC1HWC0_H_IDX], resShape[NDC1HWC0_W_IDX]};
    }

    Tensor finalResTensor(resTensor.GetStorage()->Datatype(), orgOutShape, "TensorOut");

    std::vector<SymbolicScalar> resValidShape = resTensor.GetValidShape();
    SymbolicScalar validCIn = operandVecIn[INPUT_WEIGHT_IDX]->GetDynValidShape()[NCHW_C_IDX] * convAttrParam.groups;
    std::vector<SymbolicScalar> finalResValidShape = {resValidShape[NC1HWC0_N_IDX], validCIn,
                                                      resValidShape[NC1HWC0_H_IDX], resValidShape[NC1HWC0_W_IDX]};
    if (convAttrParam.isConv3D) {
        finalResValidShape = {resValidShape[NDC1HWC0_N_IDX], validCIn, resValidShape[NDC1HWC0_D_IDX],
                              resValidShape[NDC1HWC0_H_IDX], resValidShape[NDC1HWC0_W_IDX]};
    }
    finalResTensor.GetStorage()->UpdateDynValidShape(finalResValidShape);
    return finalResTensor;
}

Tensor BpConstructTensorGraphNZ2NZ(Function* functionPtr, std::vector<LogicalTensorPtr> operandVecIn,
                                   const Tensor& resTensor, const ConvBpAttrParam& convAttrParam)
{
    std::vector<LogicalTensorPtr> operandVecOut = {resTensor.GetStorage()};
    std::vector<LogicalTensorPtr> operandVecInNZ = BpGetOperandVecIn(operandVecIn, convAttrParam);

    auto& gradoutputTransOp = functionPtr->AddOperation(Opcode::OP_FAKE_TRANS, {operandVecIn[INPUT_GRAD_OUTPUT_IDX]},
                                                        {operandVecInNZ[INPUT_GRAD_OUTPUT_IDX]});
    gradoutputTransOp.SetAttribute(FAKE_TRANS_IN_FORMAT_ATTR,
                                   static_cast<int64_t>(operandVecIn[INPUT_GRAD_OUTPUT_IDX]->Format()));
    gradoutputTransOp.SetAttribute(FAKE_TRANS_OUT_FORMAT_ATTR,
                                   static_cast<int64_t>(operandVecInNZ[INPUT_GRAD_OUTPUT_IDX]->Format()));

    auto& weightTransOp = functionPtr->AddOperation(Opcode::OP_FAKE_TRANS, {operandVecIn[INPUT_WEIGHT_IDX]},
                                                    {operandVecInNZ[INPUT_WEIGHT_IDX]});
    weightTransOp.SetAttribute(FAKE_TRANS_IN_FORMAT_ATTR,
                               static_cast<int64_t>(operandVecIn[INPUT_WEIGHT_IDX]->Format()));
    weightTransOp.SetAttribute(FAKE_TRANS_OUT_FORMAT_ATTR,
                               static_cast<int64_t>(operandVecInNZ[INPUT_WEIGHT_IDX]->Format()));

    if (convAttrParam.hasBias) {
        operandVecInNZ.push_back(operandVecIn[INPUT_BIAS_IDX]);
    }

    Opcode conBpOpCode = convAttrParam.isConv3D ? Opcode::OP_CONV_BP_INPUT_3D : Opcode::OP_CONV_BP_INPUT_2D;
    auto& op = functionPtr->AddOperation(conBpOpCode, operandVecInNZ, operandVecOut);
    // resTensor  {batch, (di,) cin1, hi, wi, cin0}
    // finalResTensor是最终的计算结果，format应该和原始一致, {batch, cin, (di,) hi, wi}
    Tensor finalResTensor = BpGetFinalResTensorNZ2NZ(operandVecIn, resTensor, convAttrParam);
    auto& orgResOp = functionPtr->AddOperation(Opcode::OP_FAKE_TRANS, operandVecOut, {finalResTensor.GetStorage()});
    orgResOp.SetAttribute(FAKE_TRANS_IN_FORMAT_ATTR, static_cast<int64_t>(resTensor.Format()));
    orgResOp.SetAttribute(FAKE_TRANS_OUT_FORMAT_ATTR, static_cast<int64_t>(finalResTensor.Format()));
    BpSetTensorOpAttr(op, operandVecIn[INPUT_GRAD_OUTPUT_IDX], operandVecIn[INPUT_WEIGHT_IDX],
                      finalResTensor.GetStorage(), convAttrParam);

    // 3D的resTensor要转换成1D、2D返回 插入reshape
    if (convAttrParam.isConv1D) {
        // {batch, cin, di, hi, wi} > {batch, cin, wi} / {batch, cin, hi, wi}
        std::vector<int64_t> finalShape = finalResTensor.GetShape();
        std::vector<int64_t> finalRealShape = std::vector<int64_t>{finalShape[NCHW_N_IDX], finalShape[NCHW_C_IDX],
                                                                   finalShape[NCHW_W_IDX]};
        Tensor finalRealDimTensor(resTensor.GetStorage()->Datatype(), finalRealShape, "TensorOutRealDim");
        // dynamic shape
        std::vector<SymbolicScalar> finalValidShape = finalResTensor.GetValidShape();
        std::vector<SymbolicScalar> finalRealValidShape = std::vector<SymbolicScalar>{
            finalValidShape[NCHW_N_IDX], finalValidShape[NCHW_C_IDX], finalValidShape[NCHW_W_IDX]};
        finalRealDimTensor.GetStorage()->UpdateDynValidShape(finalRealValidShape);

        auto& reshapeResOp = functionPtr->AddOperation(Opcode::OP_RESHAPE, {finalResTensor.GetStorage()},
                                                       {finalRealDimTensor.GetStorage()});
        reshapeResOp.SetAttribute(OpAttributeKey::isConv, true);
        return finalRealDimTensor;
    }

    return finalResTensor;
}

Tensor BpConstructTensorGraph(const Tensor& gradOutputTensor, const Tensor& weightTensor, const Tensor& biasTensor,
                              const Tensor& resTensor, ConvBpAttrParam& convAttrParam)
{
    Function* functionPtr = Program::GetInstance().GetCurrentFunction();
    ASSERT(ConvExpandFuncError::EXPANDFUNC_TILE_OP_NULLPTR, functionPtr != nullptr) << "functionPtr is nullptr.";
    std::vector<LogicalTensorPtr> operandVecIn = {gradOutputTensor.GetStorage(), weightTensor.GetStorage()};
    std::vector<LogicalTensorPtr> operandVecOut = {resTensor.GetStorage()};
    if (convAttrParam.isConv1D) {
        // unsqueeze input N,C,L -> N,C,1,L
        int64_t addDim = 1;
        std::vector<int64_t> gradoutput4DimShape = gradOutputTensor.GetShape();
        gradoutput4DimShape.insert(gradoutput4DimShape.begin() + 2, addDim, 1);
        Tensor gradOutput4DimTensor(gradOutputTensor.GetStorage()->Datatype(), gradoutput4DimShape, "",
                                    gradOutputTensor.Format());

        std::vector<int64_t> weight4DimShape = weightTensor.GetShape();
        weight4DimShape.insert(weight4DimShape.begin() + 2, addDim, 1);
        Tensor weight4DimTensor(weightTensor.GetStorage()->Datatype(), weight4DimShape, "", weightTensor.Format());
        std::vector<SymbolicScalar> weightValidShape = weightTensor.GetValidShape();
        weight4DimTensor.GetStorage()->UpdateDynValidShape(
            {weightValidShape[NCL_N_IDX], weightValidShape[NCL_C_IDX], 1, weightValidShape[NCL_L_IDX]});

        // 插入reshape
        auto& reshapeGradoutputOp = functionPtr->AddOperation(Opcode::OP_RESHAPE, {gradOutputTensor.GetStorage()},
                                                              {gradOutput4DimTensor.GetStorage()});
        auto& reshapeWeightOp = functionPtr->AddOperation(Opcode::OP_RESHAPE, {weightTensor.GetStorage()},
                                                          {weight4DimTensor.GetStorage()});
        reshapeGradoutputOp.SetAttribute(OpAttributeKey::isConv, true);
        reshapeWeightOp.SetAttribute(OpAttributeKey::isConv, true);
        operandVecIn = {gradOutput4DimTensor.GetStorage(), weight4DimTensor.GetStorage()};

        // squeeze output from NCDWH to origin format
    }

    // bias
    if (!biasTensor.IsEmpty()) {
        convAttrParam.hasBias = true;
        std::vector<int64_t> bias2DimShape{1, biasTensor.GetShape(0)};
        Tensor bias2DimTensor(biasTensor.GetStorage()->Datatype(), bias2DimShape, "", biasTensor.Format());
        auto& reshapeBiasOp = functionPtr->AddOperation(Opcode::OP_RESHAPE, {biasTensor.GetStorage()},
                                                        {bias2DimTensor.GetStorage()});
        reshapeBiasOp.SetAttribute(OpAttributeKey::isConv, true);
        operandVecIn.push_back(bias2DimTensor.GetStorage());
    }

    return BpConstructTensorGraphNZ2NZ(functionPtr, operandVecIn, resTensor, convAttrParam);
}

Tensor ConvBackwardInput(DataType outType, const Tensor& gradOutputTensor, const std::vector<int64_t>& inputSize,
                         const Tensor& weightTensor, const std::vector<int64_t>& strides,
                         const std::vector<int64_t>& paddings, const std::vector<int64_t>& dilations,
                         const ConvBpExtendParam& extendParam, const int64_t groups)
{
    CHECK(FwkErr::PLATFORM_NOT_SUPPORTED, Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_2201)
        << "Only support A2/A3 Platform";
    // 输入format是NC(D)HW/NCL
    uint64_t gradOutputDim = gradOutputTensor.Dim();
    std::unordered_set<uint64_t> allowDims = {3, 4, 5};
    CHECK(ExternalError::INVALID_VAL, allowDims.find(gradOutputDim) != allowDims.end())
        << "Input illegal grad_output dim:" << gradOutputDim << ", which must be 3 or 4 or 5.";
    CHECK(ExternalError::INVALID_VAL, gradOutputDim == weightTensor.Dim() && gradOutputDim == inputSize.size())
        << "Input illegal weight or input dim:" << weightTensor.Dim() << ", which must be equal to grad_output dim.";
    CHECK(ExternalError::INVALID_VAL, gradOutputDim - 2 == dilations.size() && gradOutputDim - 2 == strides.size() &&
                                          (gradOutputDim - 2) * 2 == paddings.size())
        << "Input illegal dilations size:" << dilations.size() << ", strides size:" << strides.size()
        << ", paddings size:" << paddings.size() << ", which must correspond to grad_output dim:" << gradOutputDim
        << ".";
    std::vector<int64_t> finalPaddings = paddings;
    std::vector<int64_t> finalDilations = dilations;
    std::vector<int64_t> finalStrides = strides;
    if (gradOutputDim != 5) {
        int64_t addDim = 5 - gradOutputDim;
        finalStrides.insert(finalStrides.begin(), addDim, 1);
        finalDilations.insert(finalDilations.begin(), addDim, 1);
        finalPaddings.insert(finalPaddings.begin(), addDim * 2, 0);
    }
    const Tensor& biasTensor = extendParam.biasTensor;
    ConvBpAttrParam convAttrParam(finalPaddings, finalStrides, finalDilations, groups);
    convAttrParam.isConv1D = gradOutputDim == Conv::CONV1D_INPUT_DIM;
    convAttrParam.isConv3D = gradOutputDim == Conv::CONV3D_INPUT_DIM;
    BpCheckConvOperands(outType, gradOutputTensor, inputSize, weightTensor, biasTensor, convAttrParam);
    std::vector<int64_t> resTensorShape = BpGetResTensorShape(outType, inputSize, convAttrParam);
    // dynamic Shape
    std::vector<SymbolicScalar> resTensorDynValidShape = SymbolicScalar::FromConcrete(resTensorShape);

    TileOpFormat outFormat = convAttrParam.isConv3D ? TileOpFormat::TILEOP_NDC1HWC0 : TileOpFormat::TILEOP_NC1HWC0;
    Tensor resTensor(outType, resTensorShape, "TensorC", outFormat);
    // dynamic shape
    resTensor.GetStorage()->UpdateDynValidShape(resTensorDynValidShape);
    return BpConstructTensorGraph(gradOutputTensor, weightTensor, biasTensor, resTensor, convAttrParam);
}

void SetImg2ColAttr(Operation& load3dOpAl0, const ConvBpAttrParam& convAttrParam, ConvBpIterInfo& iterInfo,
                    const ConvBpTileInfo& convBpTileInfo)
{
    int64_t dilationH = convAttrParam.dilations[1];
    int64_t dilationW = convAttrParam.dilations[2];
    int64_t dilatedKernelH = (convBpTileInfo.orgKh - 1) * dilationH + 1;
    int64_t dilatedKernelW = (convBpTileInfo.orgKw - 1) * dilationW + 1;
    load3dOpAl0.SetAttribute(OpAttributeKey::strideH, 1);
    load3dOpAl0.SetAttribute(OpAttributeKey::strideW, 1);
    load3dOpAl0.SetAttribute(OpAttributeKey::dilationH, dilationH);
    load3dOpAl0.SetAttribute(OpAttributeKey::dilationW, dilationW);
    load3dOpAl0.SetAttribute(OpAttributeKey::filterH, convBpTileInfo.orgKh);
    load3dOpAl0.SetAttribute(OpAttributeKey::filterW, convBpTileInfo.orgKw);
    // cal H padding
    if (iterInfo.houtL1Offset >= 0) {
        load3dOpAl0.SetAttribute(OpAttributeKey::paddingTop, 0);
    } else {
        load3dOpAl0.SetAttribute(OpAttributeKey::paddingTop, 0 - iterInfo.houtL1Offset);
    }
    int64_t houtAL1ExpandUsed = iterInfo.hinL1Size - 1 + dilatedKernelH;
    int64_t houtEndOffset = iterInfo.houtL1Offset + houtAL1ExpandUsed;
    if (houtEndOffset > convBpTileInfo.expandHout) {
        load3dOpAl0.SetAttribute(OpAttributeKey::paddingBottom, houtEndOffset - convBpTileInfo.expandHout);
    } else {
        load3dOpAl0.SetAttribute(OpAttributeKey::paddingBottom, 0);
    }
    // cal W padding
    if (iterInfo.woutL1Offset >= 0) {
        load3dOpAl0.SetAttribute(OpAttributeKey::paddingLeft, 0);
    } else {
        load3dOpAl0.SetAttribute(OpAttributeKey::paddingLeft, 0 - iterInfo.woutL1Offset);
    }
    int64_t woutAL1ExpandUsed = iterInfo.winL1Size - 1 + dilatedKernelW;
    int64_t woutEndOffset = iterInfo.woutL1Offset + woutAL1ExpandUsed;
    if (woutEndOffset > convBpTileInfo.expandWout) {
        load3dOpAl0.SetAttribute(OpAttributeKey::paddingRight, woutEndOffset - convBpTileInfo.expandWout);
    } else {
        load3dOpAl0.SetAttribute(OpAttributeKey::paddingRight, 0);
    }
    // cal postm postk
    int64_t mStartPt = iterInfo.mL0Offset;
    int64_t kStartPt = iterInfo.kL0Offset - iterInfo.kL1Offset;
    load3dOpAl0.SetAttribute(OpAttributeKey::postM, mStartPt);
    load3dOpAl0.SetAttribute(OpAttributeKey::postK, kStartPt);
    // set pad value
    load3dOpAl0.SetAttribute(OpAttributeKey::padValue, 0);
    // set load3dv2 params
    load3dOpAl0.SetAttribute(OpAttributeKey::repeatStride, iterInfo.repeatStride);
    load3dOpAl0.SetAttribute(OpAttributeKey::repeatTime, iterInfo.repeatTime);
    load3dOpAl0.SetAttribute(OpAttributeKey::wStride, iterInfo.wStride);
    // set conv/conv3d flag: backward always uses 3D format (N, D, C1, H, W, C0)
    load3dOpAl0.SetAttribute(OpAttributeKey::isConv, true);
    load3dOpAl0.SetAttribute(Conv::LoadStoreConvOpAttributeKey::isConv3D, false);
}

void SetCopyInAL1Op(Operation& copyInAL1Op, const ConvBpGraphNodes& tensorGraphNodes,
                    const ConvBpTileInfo& convBpTileInfo, ConvBpIterInfo& iterInfo,
                    const ConvBpAttrParam& convBpAttrParam, const std::vector<int64_t>& dstAL1Shape,
                    const int64_t& cout1L1Size)
{
    int64_t strideH = convBpAttrParam.strides[1];
    int64_t strideW = convBpAttrParam.strides[2];
    int64_t src_n_offset = iterInfo.batchOffset;
    int64_t cout1L1Offset = (iterInfo.kL1Offset / convBpTileInfo.cout0) / (convBpTileInfo.orgKh * convBpTileInfo.orgKw);
    // 当前只支持group=1，后续group>1的情况下需要修改
    int64_t src_c1_offset = iterInfo.groupOffset * convBpTileInfo.cout1PerGroup + cout1L1Offset;
    int64_t src_d_offset = iterInfo.doutOffset;
    int64_t realHoutOffset = std::min(std::max<int64_t>(0, iterInfo.houtL1Offset), convBpTileInfo.expandHout - 1);
    int64_t realWoutOffset = std::min(std::max<int64_t>(0, iterInfo.woutL1Offset), convBpTileInfo.expandWout - 1);
    int64_t src_h_offset = CeilDiv(realHoutOffset, strideH);
    int64_t src_w_offset = CeilDiv(realWoutOffset, strideW);
    int64_t hL1Skip = 0;
    int64_t wL1Skip = 0;
    if (strideH > 1 && realHoutOffset % strideH) {
        hL1Skip = ConvAlignB(realHoutOffset, strideH) - realHoutOffset;
    }
    int64_t hLoadToL1 = CeilDiv(iterInfo.houtL1Size - hL1Skip, strideH);
    if (strideW > 1 && realWoutOffset % strideW) {
        wL1Skip = ConvAlignB(realWoutOffset, strideW) - realWoutOffset;
    }
    int64_t wLoadToL1 = CeilDiv(iterInfo.woutL1Size - wL1Skip, strideW);

    // 当前只支持A2、A3(IsArch32Platform为true)
    copyInAL1Op.SetAttribute(Conv::LoadStoreConvOpAttributeKey::copyInMode,
                             static_cast<int64_t>(Conv::CopyInMode::COPY_MOD_NZ2NZ));
    copyInAL1Op.SetAttribute(Conv::LoadStoreConvOpAttributeKey::isConv3D, convBpAttrParam.isConv3D);
    std::vector<int64_t> srcGmOffset;
    std::vector<int64_t> srcGmShape;
    if (convBpAttrParam.isConv3D) {
        srcGmOffset = {src_n_offset, src_d_offset, src_c1_offset, src_h_offset, src_w_offset, 0};
        srcGmShape = {1, 1, cout1L1Size, hLoadToL1, wLoadToL1, convBpTileInfo.cout0};
    } else {
        srcGmOffset = {src_n_offset, src_c1_offset, src_h_offset, src_w_offset, 0};
        srcGmShape = {1, cout1L1Size, hLoadToL1, wLoadToL1, convBpTileInfo.cout0};
    }

    copyInAL1Op.SetAttribute(OpAttributeKey::strideH, strideH);
    copyInAL1Op.SetAttribute(OpAttributeKey::strideW, strideW);
    copyInAL1Op.SetAttribute(OpAttributeKey::skipH, hL1Skip);
    copyInAL1Op.SetAttribute(OpAttributeKey::skipW, wL1Skip);
    auto copyAttr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(srcGmOffset), MemoryType::MEM_L1, OpImmediate::Specified(srcGmShape),
        OpImmediate::Specified(tensorGraphNodes.gradOutputTensorPtr->tensor->GetDynRawShape()),
        OpImmediate::Specified(dstAL1Shape));
    copyInAL1Op.SetOpAttribute(copyAttr);
    copyInAL1Op.SetAttribute(OpAttributeKey::srcGmConvValidShape, SymbolicScalar::FromConcrete(srcGmShape));
}

static void ConstructGradOutputL1Tile(Function& function, const ConvBpGraphNodes& tensorGraphNodes,
                                      const ConvBpTileInfo& convBpTileInfo, ConvBpIterInfo& iterInfo,
                                      LogicalTensorPtr& dstAL1TensorPtr, const ConvBpAttrParam& convBpAttrParam)
{
    int64_t cout1L1Size = (iterInfo.kL1Size / convBpTileInfo.cout0) / (convBpTileInfo.orgKh * convBpTileInfo.orgKw);
    std::vector<int64_t> dstAL1Shape = {1, cout1L1Size, iterInfo.houtL1Size, iterInfo.woutL1Size, convBpTileInfo.cout0};
    dstAL1TensorPtr = std::make_shared<LogicalTensor>(function, tensorGraphNodes.gradOutputTensorPtr->Datatype(),
                                                      dstAL1Shape, tensorGraphNodes.gradOutputTensorPtr->Format(),
                                                      "aL1Tensor");
    dstAL1TensorPtr->UpdateDynValidShape(SymbolicScalar::FromConcrete(dstAL1Shape));
    auto& copyInAL1Op = function.AddOperation(Opcode::OP_L1_COPY_IN_CONV_BP_DX_DY,
                                              {tensorGraphNodes.gradOutputTensorPtr}, {dstAL1TensorPtr});
    copyInAL1Op.SetAttribute(Conv::LoadStoreConvOpAttributeKey::isFmap, true);
    copyInAL1Op.SetAttribute("l1_tile_shape", SymbolicScalar::FromConcrete(dstAL1Shape));

    SetCopyInAL1Op(copyInAL1Op, tensorGraphNodes, convBpTileInfo, iterInfo, convBpAttrParam, dstAL1Shape, cout1L1Size);
    iterInfo.aL1UpadateFlag = false;
}

LogicalTensorPtr ConstructGradOutputTile(Function& function, const ConvBpGraphNodes& tensorGraphNodes,
                                         const ConvBpTileInfo& convBpTileInfo, ConvBpIterInfo& iterInfo,
                                         LogicalTensorPtr& dstAL1TensorPtr, const ConvBpAttrParam& convBpAttrParam)
{
    // L1层级 gradoutput 展开
    if (iterInfo.aL1UpadateFlag) {
        ConstructGradOutputL1Tile(function, tensorGraphNodes, convBpTileInfo, iterInfo, dstAL1TensorPtr,
                                  convBpAttrParam);
    }

    // 二层展开
    // load3dv2()
    std::vector<int64_t> dstAL0Shape = std::vector<int64_t>{ConvAlignB(iterInfo.mL0Size, Conv::MKN_M_VALUE),
                                                            iterInfo.kL0Size};
    // dynamic shape
    LogicalTensorPtr dstAL0TensorPtr = std::make_shared<LogicalTensor>(
        function, tensorGraphNodes.gradOutputTensorPtr->Datatype(), dstAL0Shape,
        SymbolicScalar::FromConcrete({iterInfo.mL0Size, iterInfo.kL0Size}),
        tensorGraphNodes.gradOutputTensorPtr->Format(), "aL0Tensor");
    dstAL0TensorPtr->UpdateDynValidShape(SymbolicScalar::FromConcrete(dstAL0Shape));

    auto& load3dOpAl0 = function.AddOperation(Opcode::OP_LOAD3D_CONV, {dstAL1TensorPtr}, {dstAL0TensorPtr});
    load3dOpAl0.SetAttribute("l0_tile_shape", SymbolicScalar::FromConcrete(dstAL0Shape));
    SetImg2ColAttr(load3dOpAl0, convBpAttrParam, iterInfo, convBpTileInfo);

    return dstAL0TensorPtr;
}

void ConstructWeightL1Tile(Function& function, const ConvBpGraphNodes& tensorGraphNodes,
                           const ConvBpTileInfo& convTileInfo, ConvBpIterInfo& iterInfo,
                           LogicalTensorPtr& dstBL1TensorPtr)
{
    int64_t khxkw = convTileInfo.orgKh * convTileInfo.orgKw;
    std::vector<int64_t> dstBL1Shape = std::vector<int64_t>{(iterInfo.nL1Size / convTileInfo.cin0) * khxkw,
                                                            iterInfo.kL1Size / 16 / khxkw, 16, convTileInfo.cin0};

    dstBL1TensorPtr = std::make_shared<LogicalTensor>(function, tensorGraphNodes.weightTensorPtr->Datatype(),
                                                      dstBL1Shape, SymbolicScalar::FromConcrete(dstBL1Shape),
                                                      tensorGraphNodes.weightTensorPtr->Format(), "bL1Tensor");
    dstBL1TensorPtr->UpdateDynValidShape(SymbolicScalar::FromConcrete(dstBL1Shape));

    auto& copyInOpBl1 = function.AddOperation(Opcode::OP_L1_COPY_IN_CONV_BP, {tensorGraphNodes.weightTensorPtr},
                                              {dstBL1TensorPtr});
    copyInOpBl1.SetAttribute(OpAttributeKey::isConv, true);
    copyInOpBl1.SetAttribute(Conv::LoadStoreConvOpAttributeKey::isFmap, false);

    int64_t cout1Offset = iterInfo.kL1Offset / khxkw / 16;
    int64_t cin1Offset = iterInfo.nL1Offset / convTileInfo.cin0;
    int64_t cin1 = CeilDiv(static_cast<int>(convTileInfo.orgCin), static_cast<int>(convTileInfo.cin0));

    std::vector<int64_t> srcGmOffset = {(iterInfo.dkOffset * cin1 + cin1Offset) * khxkw, cout1Offset, 0, 0};
    std::vector<int64_t> srcGmShape = {convTileInfo.orgKd * CeilDiv(convTileInfo.orgCin, convTileInfo.cin0) * khxkw,
                                       CeilDiv(convTileInfo.orgCout, 16), 16, convTileInfo.cin0};

    auto copyAttr = std::make_shared<CopyOpAttribute>(
        OpImmediate::Specified(srcGmOffset), MemoryType::MEM_L1, OpImmediate::Specified(srcGmShape),
        OpImmediate::Specified(tensorGraphNodes.weightTensorPtr->tensor->GetDynRawShape()),
        OpImmediate::Specified(dstBL1Shape));
    copyInOpBl1.SetOpAttribute(copyAttr);
    copyInOpBl1.SetAttribute("l1_tile_shape", SymbolicScalar::FromConcrete(dstBL1Shape));
    iterInfo.bL1UpadateFlag = false;
}

LogicalTensorPtr ConstructWeightTile(Function& function, const ConvBpGraphNodes& tensorGraphNodes,
                                     const ConvBpTileInfo& convTileInfo, ConvBpIterInfo& iterInfo,
                                     LogicalTensorPtr& dstBL1TensorPtr)
{
    // L1层级 Weight 展开
    if (iterInfo.bL1UpadateFlag) {
        ConstructWeightL1Tile(function, tensorGraphNodes, convTileInfo, iterInfo, dstBL1TensorPtr);
    }
    // load2d()
    std::vector<int64_t> dstBL0Shape = std::vector<int64_t>{iterInfo.kL0Size,
                                                            ConvAlignB(iterInfo.nL0Size, Conv::MKN_N_VALUE)};
    // dynamic shape
    LogicalTensorPtr dstBL0TensorPtr = std::make_shared<LogicalTensor>(
        function, tensorGraphNodes.weightTensorPtr->Datatype(), dstBL0Shape,
        SymbolicScalar::FromConcrete({iterInfo.kL0Size, iterInfo.nL0Size}), tensorGraphNodes.weightTensorPtr->Format(),
        "bL0Tensor");
    dstBL0TensorPtr->UpdateDynValidShape(SymbolicScalar::FromConcrete(dstBL0Shape));

    auto& load2dOpBl0 = function.AddOperation(Opcode::OP_LOAD2DDX_CONV, {dstBL1TensorPtr}, {dstBL0TensorPtr});
    load2dOpBl0.SetAttribute(OpAttributeKey::postK, iterInfo.kL0Offset - iterInfo.kL1Offset);
    load2dOpBl0.SetAttribute(OpAttributeKey::postN, iterInfo.nL0Offset);
    load2dOpBl0.SetAttribute(OpAttributeKey::kL0Size, iterInfo.kL0Size);
    load2dOpBl0.SetAttribute(OpAttributeKey::nL0Size, iterInfo.nL0Size);
    load2dOpBl0.SetAttribute(OpAttributeKey::hwk, convTileInfo.orgKh * convTileInfo.orgKw);
    load2dOpBl0.SetAttribute(OpAttributeKey::k0Idx, iterInfo.kL0Offset - iterInfo.kL1Offset);
    load2dOpBl0.SetAttribute(OpAttributeKey::n0Idx, iterInfo.nL0Offset);
    load2dOpBl0.SetAttribute("l0_tile_shape", SymbolicScalar::FromConcrete(dstBL0Shape));
    load2dOpBl0.SetAttribute(OpAttributeKey::isConv, true);
    return dstBL0TensorPtr;
}

void BpSetAMulBAttr(const ConvBpGraphNodes& tensorGraphNodes, const ConvBpTileInfo& convBpTileInfo, Operation& op)
{
    ASSERT(ConvExpandFuncError::EXPANDFUNC_TILE_OP_NULLPTR, tensorGraphNodes.gradOutputTensorPtr != nullptr &&
                                                                tensorGraphNodes.weightTensorPtr != nullptr &&
                                                                tensorGraphNodes.resTensorPtr != nullptr)
        << "Expected gradOutputTensorPtr, weightTensorPtr, and resTensorPtr to be non-nullptr.";

    int64_t nzAttr = (static_cast<int64_t>(tensorGraphNodes.gradOutputTensorPtr->Format())) |
                     (static_cast<int64_t>(tensorGraphNodes.weightTensorPtr->Format()) << 1) |
                     (static_cast<int64_t>(tensorGraphNodes.resTensorPtr->Format()) << 2);
    op.SetAttribute(OpAttributeKey::isConv, true);
    op.SetAttribute(Conv::MATMUL_NZ_ATTR, nzAttr);
    op.SetAttribute(Conv::A_MUL_B_ACT_M, convBpTileInfo.mL0);
    op.SetAttribute(Conv::A_MUL_B_ACT_K, convBpTileInfo.kL0);
    op.SetAttribute(Conv::A_MUL_B_ACT_N, convBpTileInfo.nL0);

    if (op.GetOpcode() == Opcode::OP_A_MUL_B) {
        op.SetAttribute(Conv::A_MUL_B_BIAS_ATTR, tensorGraphNodes.biasTensorPtr != nullptr);
    }
}

LogicalTensorPtr DoMmad(Function& function, const ConvBpAttrParam& convBpAttrParam,
                        const ConvBpGraphNodes& tensorGraphNodes, ConvBpGraphNodes& tileGraphNodes,
                        const ConvBpTileInfo& convBpTileInfo, const ConvBpIterInfo& iterInfo)
{
    std::vector<SymbolicScalar> dstCL0DynValidShape = std::vector<SymbolicScalar>{iterInfo.mL0Size, iterInfo.nL0Size};
    ASSERT(ConvExpandFuncError::EXPANDFUNC_TILE_OP_NULLPTR, tileGraphNodes.gradOutputTensorPtr != nullptr &&
                                                                tileGraphNodes.weightTensorPtr != nullptr &&
                                                                tileGraphNodes.resTensorPtr != nullptr)
        << "Inputs and res must be non-nullptr.";
    // MMAD node add
    std::vector<LogicalTensorPtr> mmadInputs;
    std::vector<LogicalTensorPtr> mmadOutputs;
    const std::string MmadOpStr = iterInfo.isFirstK ? "TILE_A_MUL_B" : "TILE_A_MULACC_B";
    if (iterInfo.isFirstK) {
        mmadInputs = {tileGraphNodes.gradOutputTensorPtr, tileGraphNodes.weightTensorPtr};
        if (convBpAttrParam.hasBias) {
            ASSERT(ConvExpandFuncError::EXPANDFUNC_TILE_OP_NULLPTR, tileGraphNodes.biasTensorPtr != nullptr)
                << "bias must be non-nullptr when hasBias Flag.";
            mmadInputs.push_back(tileGraphNodes.biasTensorPtr);
        }
    } else {
        mmadInputs = {tileGraphNodes.gradOutputTensorPtr, tileGraphNodes.weightTensorPtr,
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
    BpSetAMulBAttr(tensorGraphNodes, convBpTileInfo, aMulBOp);

    return mmadOutputs[0];
}

void ConstructCopyOutTile(Function& function, const ConvBpAttrParam& convAttrParam,
                          const ConvBpGraphNodes& tensorGraphNodes, const ConvBpTileInfo& convTileInfo,
                          const ConvBpIterInfo& iterInfo, const LogicalTensorPtr& resCl0TensorPtr)
{
    std::vector<SymbolicScalar> dstCL0DynValidShape = std::vector<SymbolicScalar>{iterInfo.mL0Size, iterInfo.nL0Size};
    auto& fixpipeOpRes = function.AddOperation(Opcode::OP_L0C_COPY_OUT_CONV, {resCl0TensorPtr},
                                               {tensorGraphNodes.resTensorPtr});
    fixpipeOpRes.SetAttribute(OpAttributeKey::isConv, true);
    fixpipeOpRes.SetAttribute(Conv::LoadStoreConvOpAttributeKey::isConv3D, convAttrParam.isConv3D);

    resCl0TensorPtr->UpdateDynValidShape(dstCL0DynValidShape);

    // 设置cutW参数：L0C M方向(hw合轴)的w大小
    // 反向mL0里的hiwi连续的，cutW=realCutW=wi，逐行搬出
    fixpipeOpRes.SetAttribute(Conv::LoadStoreConvOpAttributeKey::cutW, static_cast<int64_t>(iterInfo.mL0Size));
    fixpipeOpRes.SetAttribute(Conv::LoadStoreConvOpAttributeKey::realCutW, SymbolicScalar(iterInfo.mL0Size));

    fixpipeOpRes.SetAttribute(Conv::LoadStoreConvOpAttributeKey::copyOutMode,
                              static_cast<int64_t>(Conv::CopyOutMode::COPY_MOD_NZ2NZ));
    fixpipeOpRes.SetAttribute(OpAttributeKey::l0cValidMN, dstCL0DynValidShape);

    int64_t dstMOffset = iterInfo.mL1Offset + iterInfo.mL0Offset;
    int64_t dstNOffset = iterInfo.nL1Offset + iterInfo.nL0Offset;

    std::vector<int64_t> dstResGmOffset = {iterInfo.batchOffset, dstNOffset / convTileInfo.cin0,
                                           dstMOffset / convTileInfo.orgWin, dstMOffset % convTileInfo.orgWin, 0};
    if (convAttrParam.isConv3D) {
        dstResGmOffset.insert(dstResGmOffset.begin() + 1, iterInfo.dinOffset);
    }
    auto copyAttr = std::make_shared<CopyOpAttribute>(
        MemoryType::MEM_L0C, OpImmediate::Specified(dstResGmOffset),
        OpImmediate::Specified(tensorGraphNodes.resTensorPtr->tensor->GetRawShape()),
        OpImmediate::Specified(tensorGraphNodes.resTensorPtr->tensor->GetDynRawShape()),
        OpImmediate::Specified(dstCL0DynValidShape));
    fixpipeOpRes.SetOpAttribute(copyAttr);
}

void IterL0ExpandFunc(Function& function, ConvBpIterInfo& iterInfo, ConvBpTileInfo& convTileInfo,
                      const ConvBpAttrParam& attrParam, const ConvBpGraphNodes& tensorGraphNodes,
                      ConvBpGraphNodes& tileGraphNodes)
{
    LogicalTensorPtr gradOutputL1TensorPtr = nullptr;
    LogicalTensorPtr weightL1TensorPtr = nullptr;
    LogicalTensorPtr resCl0TensorPtr = nullptr;
    // nL1Size -> Cout L1, loop nL0  baseN
    for (iterInfo.nL0Offset = 0; iterInfo.nL0Offset < iterInfo.nL1Size; iterInfo.nL0Offset += convTileInfo.nL0) {
        iterInfo.nL0Size = std::min(iterInfo.nL1Size - iterInfo.nL0Offset, convTileInfo.nL0);
        // baseM
        for (iterInfo.mL0Offset = 0; iterInfo.mL0Offset < iterInfo.mL1Size;) {
            // mL0在HiWi方向上连续，mL1现在不会跨win行，mL0同样限制不跨win行
            if (convTileInfo.mL0 < convTileInfo.orgWin) {
                int64_t curWinIdx = (iterInfo.mL1Offset + iterInfo.mL0Offset) % convTileInfo.orgWin;
                iterInfo.mL0Size = std::min(
                    {convTileInfo.mL0, convTileInfo.orgWin - curWinIdx, iterInfo.mL1Size - iterInfo.mL0Offset});
            } else {
                iterInfo.mL0Size = std::min(convTileInfo.mL0, iterInfo.mL1Size - iterInfo.mL0Offset);
            }

            iterInfo.repeatTime = 1;
            iterInfo.wStride = ConvAlignB(iterInfo.mL0Size, Conv::MKN_M_VALUE);

            // set res tile
            std::vector<int64_t> dstCL0Shape = std::vector<int64_t>{ConvAlignB(iterInfo.mL0Size, Conv::MKN_M_VALUE),
                                                                    ConvAlignB(iterInfo.nL0Size, Conv::MKN_N_VALUE)};
            std::vector<SymbolicScalar> dstCL0DynValidShape = std::vector<SymbolicScalar>{iterInfo.mL0Size,
                                                                                          iterInfo.nL0Size};

            tileGraphNodes.resTensorPtr = std::make_shared<LogicalTensor>(
                function, tensorGraphNodes.gradOutputTensorPtr->Datatype(), dstCL0Shape, dstCL0DynValidShape,
                tensorGraphNodes.gradOutputTensorPtr->Format(), "cL0Tensor");
            for (iterInfo.kL1Offset = 0; iterInfo.kL1Offset < convTileInfo.kPerGroup;
                 iterInfo.kL1Offset += convTileInfo.kL1) {
                iterInfo.kL1Size = std::min(convTileInfo.kPerGroup - iterInfo.kL1Offset, convTileInfo.kL1);
                iterInfo.aL1UpadateFlag = true;
                iterInfo.bL1UpadateFlag = true;
                int64_t kLimit = iterInfo.kL1Size + iterInfo.kL1Offset;
                for (iterInfo.kL0Offset = iterInfo.kL1Offset; iterInfo.kL0Offset < kLimit;
                     iterInfo.kL0Offset += convTileInfo.kL0) {
                    iterInfo.kL0Size = std::min(kLimit - iterInfo.kL0Offset, convTileInfo.kL0);
                    iterInfo.isFirstK = iterInfo.kL0Offset == 0 ? true : false;
                    iterInfo.isLastK = iterInfo.kL0Offset + convTileInfo.kL0 >= convTileInfo.kPerGroup ? true : false;

                    tileGraphNodes.gradOutputTensorPtr = ConstructGradOutputTile(
                        function, tensorGraphNodes, convTileInfo, iterInfo, gradOutputL1TensorPtr, attrParam);
                    tileGraphNodes.weightTensorPtr = ConstructWeightTile(function, tensorGraphNodes, convTileInfo,
                                                                         iterInfo, weightL1TensorPtr);
                    // add mmad node
                    resCl0TensorPtr = DoMmad(function, attrParam, tensorGraphNodes, tileGraphNodes, convTileInfo,
                                             iterInfo);
                }
            }
            ConstructCopyOutTile(function, attrParam, tensorGraphNodes, convTileInfo, iterInfo, resCl0TensorPtr);
            iterInfo.mL0Offset += iterInfo.mL0Size;
        }
    }
}

void UpdateL1IterInfo(const ConvBpTileInfo& convBpTileInfo, ConvBpIterInfo& iterInfo,
                      const ConvBpAttrParam& convBpAttrParam)
{
    if (convBpTileInfo.mL1 < convBpTileInfo.orgWin) {
        int64_t rowRemain = convBpTileInfo.orgWin - (iterInfo.mL1Offset % convBpTileInfo.orgWin);
        iterInfo.mL1Size = std::min(rowRemain, convBpTileInfo.mL1);
    } else {
        iterInfo.mL1Size = std::min(convBpTileInfo.orgHinWin - iterInfo.mL1Offset, convBpTileInfo.mL1);
    }

    int64_t cin1c0 = convBpTileInfo.cin1PerGroup * convBpTileInfo.cin0;
    iterInfo.nL1Size = (iterInfo.nL1Offset + convBpTileInfo.nL1) > cin1c0 ? (cin1c0 - iterInfo.nL1Offset) :
                                                                            convBpTileInfo.nL1;

    iterInfo.hinL1Offset = iterInfo.mL1Offset / convBpTileInfo.orgWin;
    iterInfo.hinL1Size = CeilDiv(iterInfo.mL1Size, convBpTileInfo.orgWin);
    iterInfo.houtL1Offset = iterInfo.hinL1Offset - convBpTileInfo.bpPadUp;
    int64_t endHoutOffset = iterInfo.houtL1Offset + iterInfo.hinL1Size +
                            (convBpTileInfo.orgKh - 1) * convBpAttrParam.dilations[1];
    if (iterInfo.houtL1Offset < convBpTileInfo.expandHout && endHoutOffset > 0) {
        int64_t realStartHout = iterInfo.houtL1Offset < 0 ? 0 : iterInfo.houtL1Offset;
        int64_t realEndHout = endHoutOffset > convBpTileInfo.expandHout ? convBpTileInfo.expandHout : endHoutOffset;
        iterInfo.houtL1Size = realEndHout - realStartHout;
    } else {
        iterInfo.houtL1Size = 0;
    }

    iterInfo.winL1Offset = iterInfo.mL1Offset % convBpTileInfo.orgWin;
    iterInfo.winL1Size = std::min(iterInfo.mL1Size, convBpTileInfo.orgWin);
    iterInfo.woutL1Offset = iterInfo.winL1Offset - convBpTileInfo.bpPadLeft;
    int64_t endWoutOffset = iterInfo.woutL1Offset + iterInfo.winL1Size +
                            (convBpTileInfo.orgKw - 1) * convBpAttrParam.dilations[2];
    if (iterInfo.woutL1Offset < convBpTileInfo.expandWout && endWoutOffset > 0) {
        int64_t realStartWout = iterInfo.woutL1Offset < 0 ? 0 : iterInfo.woutL1Offset;
        int64_t realEndWout = endWoutOffset > convBpTileInfo.expandWout ? convBpTileInfo.expandWout : endWoutOffset;
        iterInfo.woutL1Size = realEndWout - realStartWout;
    } else {
        iterInfo.woutL1Size = 0;
    }
}

void IterL1ExpandFunc(Function& function, ConvBpIterInfo& iterInfo, ConvBpTileInfo& convBpTileInfo,
                      const ConvBpAttrParam& convBpAttrParam, const ConvBpGraphNodes& tensorGraphNodes,
                      ConvBpGraphNodes& tileGraphNodes)
{
    // L1 loop  不再手动分核 框架自动分配task  对于反向N是Cin
    for (iterInfo.nL1Offset = 0; iterInfo.nL1Offset < convBpTileInfo.cinPerGroup;
         iterInfo.nL1Offset += convBpTileInfo.nL1) {
        for (iterInfo.mL1Offset = 0; iterInfo.mL1Offset < convBpTileInfo.orgHinWin;) {
            UpdateL1IterInfo(convBpTileInfo, iterInfo, convBpAttrParam);
            if (iterInfo.houtL1Size == 0 || iterInfo.woutL1Size == 0) {
                iterInfo.mL1Offset += iterInfo.mL1Size;
                continue;
            }
            IterL0ExpandFunc(function, iterInfo, convBpTileInfo, convBpAttrParam, tensorGraphNodes, tileGraphNodes);
            iterInfo.mL1Offset += iterInfo.mL1Size;
        }
    }
}

void SetConvBpAttrParam(const Operation& op, ConvBpAttrParam& convBpAttrParam)
{
    convBpAttrParam.isConv3D = (op.HasAttr(CONV_3D_FLAG)) ? op.GetBoolAttribute(CONV_3D_FLAG) : false;
    convBpAttrParam.paddings = (op.HasAttr(CONV_PADDINGS_ATTR)) ? op.GetVectorIntAttribute(CONV_PADDINGS_ATTR) :
                                                                  CONV3D_PAD_ATTR_DEFAULT_LIST;
    convBpAttrParam.strides = (op.HasAttr(CONV_STRIDES_ATTR)) ? op.GetVectorIntAttribute(CONV_STRIDES_ATTR) :
                                                                CONV3D_ATTR_DEFAULT_LIST;
    convBpAttrParam.dilations = (op.HasAttr(CONV_DILATIONS_ATTR)) ? op.GetVectorIntAttribute(CONV_DILATIONS_ATTR) :
                                                                    CONV3D_ATTR_DEFAULT_LIST;
    convBpAttrParam.groups = (op.HasAttr(CONV_GROUPS_ATTR)) ? op.GetIntAttribute(CONV_GROUPS_ATTR) : 1;
    convBpAttrParam.hasBias = (op.HasAttr(CONV_BIAS_ATTR)) ? op.GetBoolAttribute(CONV_BIAS_ATTR) : false;
    convBpAttrParam.isInOutTensorNZ = true;

    ASSERT(ConvExpandFuncError::EXPANDFUNC_TENSOR_ATTR_GET_FAILED, op.HasAttr(CONV_ORI_GRAD_OUTPUT_SHAPE_ATTR))
        << "Conv Backward ori gradOutput shape should be set when InOut Tensor NZ mode.";
    ASSERT(ConvExpandFuncError::EXPANDFUNC_TENSOR_ATTR_GET_FAILED, op.HasAttr(CONV_ORI_WEIGHT_SHAPE_ATTR))
        << "Conv Backward ori weight shape should be set when InOut Tensor NZ mode.";
    convBpAttrParam.oriGradOutputShape = op.GetVectorIntAttribute(CONV_ORI_GRAD_OUTPUT_SHAPE_ATTR);
    convBpAttrParam.oriWeightShape = op.GetVectorIntAttribute(CONV_ORI_WEIGHT_SHAPE_ATTR);
    convBpAttrParam.oriResShape = op.GetVectorIntAttribute(CONV_ORI_RES_SHAPE_ATTR);
    convBpAttrParam.dynValidResShape = op.GetVectorSymbolicScalarAttribute("dynamicResValidShape");
}

void SetTensorGraphNodes(const std::vector<LogicalTensorPtr>& operandVec, const LogicalTensorPtr& cTensorPtr,
                         const ConvBpAttrParam& convBpAttrParam, ConvBpGraphNodes& tensorGraphNodes)
{
    // set tensor GraphNodes
    size_t operandVecSize = SHAPE_DIM2 + static_cast<size_t>(convBpAttrParam.hasBias);
    ASSERT(ConvExpandFuncError::EXPANDFUNC_PARAMS_INVALID, operandVec.size() == operandVecSize)
        << "Operand vector size mismatch: "
        << "Expected size: " << operandVecSize << ", actual size: " << operandVec.size()
        << ", without bias, Conv Backward Input size should be " << SHAPE_DIM2
        << ", hasBias: " << convBpAttrParam.hasBias;

    tensorGraphNodes.gradOutputTensorPtr = operandVec[INPUT_GRAD_OUTPUT_IDX];
    tensorGraphNodes.weightTensorPtr = operandVec[INPUT_WEIGHT_IDX];
    if (convBpAttrParam.hasBias) {
        tensorGraphNodes.biasTensorPtr = operandVec[INPUT_BIAS_IDX];
    }
    ASSERT(ConvExpandFuncError::EXPANDFUNC_TILE_OP_NULLPTR,
           tensorGraphNodes.gradOutputTensorPtr != nullptr && tensorGraphNodes.weightTensorPtr != nullptr)
        << "Expected aTensorPtr and bTensorPtr to be non-nullptr.";

    ASSERT(ConvExpandFuncError::EXPANDFUNC_TILE_OP_NULLPTR, cTensorPtr != nullptr) << "cTensorPtr is nullptr.";
    tensorGraphNodes.resTensorPtr = cTensorPtr;
}

void SetConvBpShapeInfo(const TileShape& tileShape, const ConvBpAttrParam& convBpAttrParam,
                        ConvBpTileInfo& convBpTileInfo)
{
    // set org shape
    // 2D: oriResShape/oriGradOutputShape/oriWeightShape are NCHW (4D)
    // 3D: oriResShape/oriGradOutputShape/oriWeightShape are NCDHW (5D)
    if (convBpAttrParam.isConv3D) {
        convBpTileInfo.orgBatch = convBpAttrParam.oriResShape[NCDHW_N_IDX];
        convBpTileInfo.orgCin = convBpAttrParam.oriResShape[NCDHW_C_IDX];
        convBpTileInfo.orgDin = convBpAttrParam.oriResShape[NCDHW_D_IDX];
        convBpTileInfo.orgHin = convBpAttrParam.oriResShape[NCDHW_H_IDX];
        convBpTileInfo.orgWin = convBpAttrParam.oriResShape[NCDHW_W_IDX];
        convBpTileInfo.orgCout = convBpAttrParam.oriGradOutputShape[NCDHW_C_IDX];
        convBpTileInfo.orgDout = convBpAttrParam.oriGradOutputShape[NCDHW_D_IDX];
        convBpTileInfo.orgHout = convBpAttrParam.oriGradOutputShape[NCDHW_H_IDX];
        convBpTileInfo.orgWout = convBpAttrParam.oriGradOutputShape[NCDHW_W_IDX];
        convBpTileInfo.orgKh = convBpAttrParam.oriWeightShape[NCDHW_H_IDX];
        convBpTileInfo.orgKw = convBpAttrParam.oriWeightShape[NCDHW_W_IDX];
        convBpTileInfo.orgKd = convBpAttrParam.oriWeightShape[NCDHW_D_IDX];
    } else {
        convBpTileInfo.orgBatch = convBpAttrParam.oriResShape[NCHW_N_IDX];
        convBpTileInfo.orgCin = convBpAttrParam.oriResShape[NCHW_C_IDX];
        convBpTileInfo.orgDin = 1;
        convBpTileInfo.orgHin = convBpAttrParam.oriResShape[NCHW_H_IDX];
        convBpTileInfo.orgWin = convBpAttrParam.oriResShape[NCHW_W_IDX];
        convBpTileInfo.orgCout = convBpAttrParam.oriGradOutputShape[NCHW_C_IDX];
        convBpTileInfo.orgDout = 1;
        convBpTileInfo.orgHout = convBpAttrParam.oriGradOutputShape[NCHW_H_IDX];
        convBpTileInfo.orgWout = convBpAttrParam.oriGradOutputShape[NCHW_W_IDX];
        convBpTileInfo.orgKh = convBpAttrParam.oriWeightShape[NCHW_H_IDX];
        convBpTileInfo.orgKw = convBpAttrParam.oriWeightShape[NCHW_W_IDX];
        convBpTileInfo.orgKd = 1;
    }
    convBpTileInfo.orgHinWin = convBpTileInfo.orgHin * convBpTileInfo.orgWin;

    convBpTileInfo.expandHout = (convBpTileInfo.orgHout - 1) * convBpAttrParam.strides[1] + 1;
    convBpTileInfo.expandWout = (convBpTileInfo.orgWout - 1) * convBpAttrParam.strides[2] + 1;
    convBpTileInfo.bpPadUp = (convBpTileInfo.orgKh - 1) * convBpAttrParam.dilations[1] - convBpAttrParam.paddings[2];
    convBpTileInfo.bpPadDown = convBpTileInfo.orgHin - convBpTileInfo.expandHout +
                               convBpAttrParam.dilations[1] * (convBpTileInfo.orgKh - 1) - convBpTileInfo.bpPadUp;
    convBpTileInfo.bpPadLeft = (convBpTileInfo.orgKw - 1) * convBpAttrParam.dilations[2] - convBpAttrParam.paddings[4];
    convBpTileInfo.bpPadRight = convBpTileInfo.orgWin - convBpTileInfo.expandWout +
                                convBpAttrParam.dilations[2] * (convBpTileInfo.orgKw - 1) - convBpTileInfo.bpPadLeft;

    convBpTileInfo.cin0 = ALIGN_SIZE_16;
    convBpTileInfo.cout0 = Conv::MKN_N_VALUE;
    convBpTileInfo.cinPerGroup = convBpTileInfo.orgCin / convBpAttrParam.groups;
    convBpTileInfo.cin1PerGroup = CeilDiv(convBpTileInfo.cinPerGroup, convBpTileInfo.cin0);
    convBpTileInfo.coutPerGroup = convBpTileInfo.orgCout / convBpAttrParam.groups;
    convBpTileInfo.cout1PerGroup = CeilDiv(convBpTileInfo.coutPerGroup, convBpTileInfo.cout0);
    convBpTileInfo.kPerGroup = ConvAlignB(convBpTileInfo.coutPerGroup, convBpTileInfo.cout0) * convBpTileInfo.orgKh *
                               convBpTileInfo.orgKw;

    // set tileshape info
    auto& convBpTile = tileShape.GetConvBpInputTile();
    convBpTileInfo.mL1 = convBpTile.tileL1Info.tileML1;
    convBpTileInfo.kL1 = convBpTile.tileL1Info.tileKL1;
    convBpTileInfo.nL1 = convBpTile.tileL1Info.tileNL1;
    convBpTileInfo.mL0 = convBpTile.tileL0Info.tileML0;
    convBpTileInfo.kL0 = convBpTile.tileL0Info.tileKL0;
    convBpTileInfo.nL0 = convBpTile.tileL0Info.tileNL0;
}

void ConstructTileGraph(Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& operandVec,
                        const LogicalTensorPtr& cTensorPtr, const Operation& op)
{
    // op attr set
    ConvBpAttrParam convBpAttrParam;
    SetConvBpAttrParam(op, convBpAttrParam);
    // set tensor graph node info
    ConvBpGraphNodes tensorGraphNodes;
    SetTensorGraphNodes(operandVec, cTensorPtr, convBpAttrParam, tensorGraphNodes);
    // save tile info
    ConvBpTileInfo convBpTileInfo;
    SetConvBpShapeInfo(tileShape, convBpAttrParam, convBpTileInfo);
    // save iter info
    ConvBpIterInfo iterInfo;
    // set tile graph node info
    ConvBpGraphNodes tileGraphNodes;

    for (iterInfo.groupOffset = 0; iterInfo.groupOffset < convBpAttrParam.groups; iterInfo.groupOffset += 1) {
        for (iterInfo.batchOffset = 0; iterInfo.batchOffset < convBpTileInfo.orgBatch; iterInfo.batchOffset += 1) {
            for (iterInfo.dinOffset = 0; iterInfo.dinOffset < convBpTileInfo.orgDin; iterInfo.dinOffset += 1) {
                IterL1ExpandFunc(function, iterInfo, convBpTileInfo, convBpAttrParam, tensorGraphNodes, tileGraphNodes);
            }
        }
    }
}

} // namespace ConvBp
} // namespace tile_fwk
} // namespace npu
