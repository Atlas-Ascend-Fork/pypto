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
 * \file trans_data.cpp
 * \brief
 */

#include "unary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "tilefwk/error_code.h"
#include "tilefwk/platform.h"

namespace npu::tile_fwk {

struct TransDataTileInfoPara {
    TileInfo inputTileInfo;
    TileInfo dstTileInfo;
};

struct TransDataPara {
    const LogicalTensorPtr& input;
    const LogicalTensorPtr& dstTensor;
    const std::vector<SymbolicScalar> tileParams;
    const int group;
    int groupIdx;
    int DIdx;
    int C1Idx;
    int HIdx;
    int WIdx;
};

std::shared_ptr<LogicalTensor> transDataPadNC1HWC0(Function& function, const std::shared_ptr<LogicalTensor>& inputTile,
                                                   int64_t C0)
{
    auto inputShape = inputTile->GetShape();
    int64_t N = inputShape[0];
    int64_t C = inputShape[1];
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, C0 > 0) << "The C0 is not valid !";
    int64_t C1 = (C + C0 - 1) / C0;
    int64_t padC = C1 * C0 - C;
    int64_t H = inputShape[2];
    int64_t W = inputShape[3];

    if (!padC) {
        return inputTile;
    }

    Shape resShape = Shape{N, C1 * C0, H, W};
    auto resValidShape = inputTile->dynValidShape_;
    auto resTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), resShape, resValidShape);
    std::shared_ptr<LogicalTensor> tmpResTile = resTile->View(function, inputTile->GetShape(), {0, 0, 0, 0});
    [[maybe_unused]] auto& copyOp1 = function.AddOperation(Opcode::OP_REGISTER_COPY, {inputTile}, {tmpResTile});
    Shape resultRemainShape = {N, C1 * C0 - C, H, W};
    std::shared_ptr<LogicalTensor> resRemainTile = resTile->View(function, resultRemainShape, {0, C, 0, 0});
    auto padTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), resultRemainShape,
                                                   SymbolicScalar::FromConcrete(resultRemainShape));
    auto& vecDupOp = function.AddOperation(Opcode::OP_VEC_DUP, {}, {padTile});
    vecDupOp.SetAttribute(OpAttributeKey::scalar, Element(inputTile->Datatype(), 0));
    vecDupOp.SetAttribute(OP_ATTR_PREFIX + "shape", resultRemainShape);
    vecDupOp.SetAttribute(OP_ATTR_PREFIX + "validShape", SymbolicScalar::FromConcrete(resultRemainShape));
    [[maybe_unused]] auto& copyOp2 = function.AddOperation(Opcode::OP_REGISTER_COPY, {padTile}, {resRemainTile});
    return resTile;
}

std::shared_ptr<LogicalTensor> transDataPadFractalZ(Function& function, const std::shared_ptr<LogicalTensor>& inputTile,
                                                    int64_t C0)
{
    auto inputShape = inputTile->GetShape();
    int64_t N = inputShape[0];
    int64_t C = inputShape[1];
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, C0 > 0) << "The C0 is not valid !";
    int64_t C1 = (C + C0 - 1) / C0;
    int64_t padC = C1 * C0 - C;
    int64_t H = inputShape[2];
    int64_t W = inputShape[3];
    int64_t N0 = 16;
    int64_t N1 = (N + N0 - 1) / N0;
    int64_t padN = N1 * N0 - N;

    if ((!padC) && (!padN)) {
        return inputTile;
    }

    Shape resShape = Shape{N1 * N0, C1 * C0, H, W};
    auto resValidShape = inputTile->dynValidShape_;
    auto resTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), resShape, resValidShape);
    std::shared_ptr<LogicalTensor> tmpResTile = resTile->View(function, inputTile->GetShape(), {0, 0, 0, 0});
    [[maybe_unused]] auto& copyOp1 = function.AddOperation(Opcode::OP_REGISTER_COPY, {inputTile}, {tmpResTile});

    if (padC) {
        Shape resultCRemainShape = {N, C1 * C0 - C, H, W};
        std::shared_ptr<LogicalTensor> resCRemainTile = resTile->View(function, resultCRemainShape, {0, C, 0, 0});
        auto padCTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), resultCRemainShape,
                                                        SymbolicScalar::FromConcrete(resultCRemainShape));
        auto& vecDupOp = function.AddOperation(Opcode::OP_VEC_DUP, {}, {padCTile});
        vecDupOp.SetAttribute(OpAttributeKey::scalar, Element(inputTile->Datatype(), 0));
        vecDupOp.SetAttribute(OP_ATTR_PREFIX + "shape", resultCRemainShape);
        vecDupOp.SetAttribute(OP_ATTR_PREFIX + "validShape", SymbolicScalar::FromConcrete(resultCRemainShape));
        [[maybe_unused]] auto& copyOp2 = function.AddOperation(Opcode::OP_REGISTER_COPY, {padCTile}, {resCRemainTile});
    }
    if (padN) {
        Shape resultNRemainShape = {N1 * N0 - N, C1 * C0, H, W};
        std::shared_ptr<LogicalTensor> resNRemainTile = resTile->View(function, resultNRemainShape, {N, 0, 0, 0});
        auto padNTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), resultNRemainShape,
                                                        SymbolicScalar::FromConcrete(resultNRemainShape));
        auto& vecDupOp = function.AddOperation(Opcode::OP_VEC_DUP, {}, {padNTile});
        vecDupOp.SetAttribute(OpAttributeKey::scalar, Element(inputTile->Datatype(), 0));
        vecDupOp.SetAttribute(OP_ATTR_PREFIX + "shape", resultNRemainShape);
        vecDupOp.SetAttribute(OP_ATTR_PREFIX + "validShape", SymbolicScalar::FromConcrete(resultNRemainShape));
        [[maybe_unused]] auto& copyOp2 = function.AddOperation(Opcode::OP_REGISTER_COPY, {padNTile}, {resNRemainTile});
    }
    return resTile;
}

std::shared_ptr<LogicalTensor> transDataPadFractalZ3D(Function& function,
                                                      const std::shared_ptr<LogicalTensor>& inputTile, int64_t C0)
{
    auto inputShape = inputTile->GetShape();
    int64_t N = inputShape[0];
    int64_t C = inputShape[1];
    int64_t D = inputShape[2];
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, C0 > 0) << "The C0 is not valid !";
    int64_t C1 = (C + C0 - 1) / C0;
    int64_t padC = C1 * C0 - C;
    int64_t H = inputShape[3];
    int64_t W = inputShape[4];
    int64_t N0 = 16;
    int64_t N1 = (N + N0 - 1) / N0;
    int64_t padN = N1 * N0 - N;

    if ((!padC) && (!padN)) {
        return inputTile;
    }

    Shape resShape = Shape{N1 * N0, C1 * C0, D, H, W};
    auto resValidShape = inputTile->dynValidShape_;
    auto resTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), resShape, resValidShape);
    std::shared_ptr<LogicalTensor> tmpResTile = resTile->View(function, inputTile->GetShape(), {0, 0, 0, 0, 0});
    [[maybe_unused]] auto& copyOp1 = function.AddOperation(Opcode::OP_REGISTER_COPY, {inputTile}, {tmpResTile});

    if (padC) {
        Shape resultCRemainShape = {N, C1 * C0 - C, D, H, W};
        std::shared_ptr<LogicalTensor> resCRemainTile = resTile->View(function, resultCRemainShape, {0, C, 0, 0, 0});
        auto padCTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), resultCRemainShape,
                                                        SymbolicScalar::FromConcrete(resultCRemainShape));
        auto& vecDupOp = function.AddOperation(Opcode::OP_VEC_DUP, {}, {padCTile});
        vecDupOp.SetAttribute(OpAttributeKey::scalar, Element(inputTile->Datatype(), 0));
        vecDupOp.SetAttribute(OP_ATTR_PREFIX + "shape", resultCRemainShape);
        vecDupOp.SetAttribute(OP_ATTR_PREFIX + "validShape", SymbolicScalar::FromConcrete(resultCRemainShape));
        [[maybe_unused]] auto& copyOp2 = function.AddOperation(Opcode::OP_REGISTER_COPY, {padCTile}, {resCRemainTile});
    }
    if (padN) {
        Shape resultNRemainShape = {N1 * N0 - N, C1 * C0, D, H, W};
        std::shared_ptr<LogicalTensor> resNRemainTile = resTile->View(function, resultNRemainShape, {N, 0, 0, 0, 0});
        auto padNTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), resultNRemainShape,
                                                        SymbolicScalar::FromConcrete(resultNRemainShape));
        auto& vecDupOp = function.AddOperation(Opcode::OP_VEC_DUP, {}, {padNTile});
        vecDupOp.SetAttribute(OpAttributeKey::scalar, Element(inputTile->Datatype(), 0));
        vecDupOp.SetAttribute(OP_ATTR_PREFIX + "shape", resultNRemainShape);
        vecDupOp.SetAttribute(OP_ATTR_PREFIX + "validShape", SymbolicScalar::FromConcrete(resultNRemainShape));
        [[maybe_unused]] auto& copyOp2 = function.AddOperation(Opcode::OP_REGISTER_COPY, {padNTile}, {resNRemainTile});
    }
    return resTile;
}

std::shared_ptr<LogicalTensor> transDataPadNDC1HWC0(Function& function, const std::shared_ptr<LogicalTensor>& inputTile,
                                                    int64_t C0)
{
    auto inputShape = inputTile->GetShape();
    int64_t N = inputShape[0];
    int64_t D = inputShape[1];
    int64_t C = inputShape[2];
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, C0 > 0) << "The C0 is not valid !";
    int64_t C1 = (C + C0 - 1) / C0;
    int64_t padC = C1 * C0 - C;
    int64_t H = inputShape[3];
    int64_t W = inputShape[4];

    if (!padC) {
        return inputTile;
    }

    Shape resShape = Shape{N, D, C1 * C0, H, W};
    auto resValidShape = inputTile->dynValidShape_;
    auto resTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), resShape, resValidShape);
    std::shared_ptr<LogicalTensor> tmpResTile = resTile->View(function, inputTile->GetShape(), {0, 0, 0, 0, 0});
    [[maybe_unused]] auto& copyOp1 = function.AddOperation(Opcode::OP_REGISTER_COPY, {inputTile}, {tmpResTile});
    Shape resultRemainShape = {N, D, C1 * C0 - C, H, W};
    std::shared_ptr<LogicalTensor> resRemainTile = resTile->View(function, resultRemainShape, {0, 0, C, 0, 0});
    auto padTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), resultRemainShape,
                                                   SymbolicScalar::FromConcrete(resultRemainShape));
    auto& vecDupOp = function.AddOperation(Opcode::OP_VEC_DUP, {}, {padTile});
    vecDupOp.SetAttribute(OpAttributeKey::scalar, Element(inputTile->Datatype(), 0));
    vecDupOp.SetAttribute(OP_ATTR_PREFIX + "shape", resultRemainShape);
    vecDupOp.SetAttribute(OP_ATTR_PREFIX + "validShape", SymbolicScalar::FromConcrete(resultRemainShape));

    [[maybe_unused]] auto& copyOp2 = function.AddOperation(Opcode::OP_REGISTER_COPY, {padTile}, {resRemainTile});
    return resTile;
}

template <TileOpFormat T>
std::shared_ptr<LogicalTensor> transDataPad(Function& function, const std::shared_ptr<LogicalTensor>& inputTile,
                                            int64_t C0)
{
    switch (T) {
        case TileOpFormat::TILEOP_NC1HWC0:
            return transDataPadNC1HWC0(function, inputTile, C0);
        case TileOpFormat::TILEOP_FRACTAL_Z:
            return transDataPadFractalZ(function, inputTile, C0);
        case TileOpFormat::TILEOP_NDC1HWC0:
            return transDataPadNDC1HWC0(function, inputTile, C0);
        case TileOpFormat::TILEOP_FRACTAL_Z_3D:
            return transDataPadFractalZ3D(function, inputTile, C0);
        default:
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, false) << "The transDataType is not supported";
    }
    return inputTile;
}

std::shared_ptr<LogicalTensor> GetNC1HWC0DstTile(Function& function, const LogicalTensorPtr& dstTensor,
                                                 const std::shared_ptr<LogicalTensor>& inputTile,
                                                 TransDataTileInfoPara& transDataTileInfoPara, int64_t C0)
{
    const auto& inputShape = inputTile->GetShape();
    Shape dstShape = {inputShape[0], inputShape[1] / C0, inputShape[2], inputShape[3], C0};
    Offset dstOffset = {
        transDataTileInfoPara.inputTileInfo.offset[0], transDataTileInfoPara.inputTileInfo.offset[1] / C0,
        transDataTileInfoPara.inputTileInfo.offset[2], transDataTileInfoPara.inputTileInfo.offset[3], 0};

    return dstTensor->View(function, dstShape, dstOffset);
}

void HandleNC1HWC0Format(Function& function, const LogicalTensorPtr& dstTensor, const LogicalTensorPtr& inputTile,
                         std::vector<SymbolicScalar>& tileParams, TransDataTileInfoPara& transDataTileInfoPara)
{
    int64_t C0 = BLOCK_SIZE / BytesOf(inputTile->Datatype());
    auto realInputTile = transDataPad<TileOpFormat::TILEOP_NC1HWC0>(function, inputTile, C0);
    auto realInputShape = realInputTile->GetShape();
    auto dstTensorTile = GetNC1HWC0DstTile(function, dstTensor, realInputTile, transDataTileInfoPara, C0);

    int64_t N = realInputShape[0];
    int64_t C = realInputShape[1];
    int64_t H = realInputShape[2];
    int64_t W = realInputShape[3];
    int64_t WPad = CeilDiv(W, C0) * C0;
    int64_t C1 = C / C0;
    int64_t yTileSizeElem = BytesOf(inputTile->Datatype()) == 1 ? 32 : 16;
    int64_t shape2 = H * WPad * ((C0 + yTileSizeElem - 1) / yTileSizeElem * yTileSizeElem);
    std::vector<int64_t> tmpShape = {shape2};
    auto tmpTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), tmpShape);
    auto tmpDstTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), Shape{N, C1, H, WPad, C0});
    auto inputValidShape = realInputTile->GetDynValidShape();
    inputValidShape[1] = (inputValidShape[1] + C0 - 1) / C0;
    inputValidShape.emplace_back(SymbolicScalar(C0));
    tmpDstTile->UpdateDynValidShape(inputValidShape);
    dstTensorTile->UpdateDynValidShape(inputValidShape);

    auto& op = function.AddOperation(Opcode::OP_NCHW2NC1HWC0, {realInputTile}, {tmpDstTile, tmpTile});
    for (int i = 0; i < SHAPE_DIM4; i++) {
        tileParams[i] = SymbolicScalar(transDataTileInfoPara.inputTileInfo.offset[i]);
    }
    op.SetAttribute(OpAttributeKey::transDataOffset, tileParams);
    std::shared_ptr<LogicalTensor> realDstTile = tmpDstTile->View(function, Shape{N, C1, H, W, C0},
                                                                  Offset{0, 0, 0, 0, 0});
    [[maybe_unused]] auto& copyOp = function.AddOperation(Opcode::OP_REGISTER_COPY, {realDstTile}, {dstTensorTile});
}

std::shared_ptr<LogicalTensor> GetFractalZDstTile(Function& function, const LogicalTensorPtr& dstTensor,
                                                  const std::shared_ptr<LogicalTensor>& inputTile,
                                                  const TransDataTileInfoPara& transDataTileInfoPara,
                                                  const TransDataPara& transDataPara, int64_t C0)
{
    const auto& inputShape = inputTile->GetShape();
    int64_t N0 = 16;
    int64_t N1 = inputShape[0] / N0;
    int64_t shape0 = inputShape[1] / C0 * inputShape[2] * inputShape[3];
    Shape dstShape = {shape0, N1, N0, C0};

    auto originInput = transDataPara.input;
    auto originN = originInput->GetShape()[0];
    auto originC = originInput->GetShape()[1];
    auto originH = originInput->GetShape()[2];
    auto originW = originInput->GetShape()[3];

    int64_t originPerGroupN = originN / transDataPara.group;
    int64_t originPerGroupPadN = (originPerGroupN + N0 - 1) / N0 * N0;
    int64_t originC1 = (originC + C0 - 1) / C0;
    int64_t inputOffsetN = transDataTileInfoPara.inputTileInfo.offset[0];
    int64_t inputOffsetC1 = transDataTileInfoPara.inputTileInfo.offset[1] / C0;
    int64_t inputOffsetH = transDataTileInfoPara.inputTileInfo.offset[2];
    int64_t inputOffsetW = transDataTileInfoPara.inputTileInfo.offset[3];
    int64_t offset0 = transDataPara.groupIdx * originC1 * originH * originW + inputOffsetC1 * originH * originW +
                      inputOffsetH * originW + inputOffsetW;
    int64_t offset1 = (inputOffsetN % originPerGroupPadN) / N0;
    Offset dstOffset = {offset0, offset1, 0, 0};

    return dstTensor->View(function, dstShape, dstOffset);
}

void HandleFractalZFormat(Function& function, const LogicalTensorPtr& dstTensor, const LogicalTensorPtr& inputTile,
                          std::vector<SymbolicScalar>& tileParams, const TransDataTileInfoPara& transDataTileInfoPara,
                          const TransDataPara& transDataPara)
{
    int64_t C0 = BLOCK_SIZE / BytesOf(inputTile->Datatype());
    auto realInputTile = transDataPad<TileOpFormat::TILEOP_FRACTAL_Z>(function, inputTile, C0);
    auto realInputShape = realInputTile->GetShape();
    auto dstTensorTile = GetFractalZDstTile(function, dstTensor, realInputTile, transDataTileInfoPara, transDataPara,
                                            C0);

    int64_t N = realInputShape[0];
    int64_t C = realInputShape[1];
    int64_t H = realInputShape[2];
    int64_t W = realInputShape[3];
    int64_t WPad = CeilDiv(W, C0) * C0;
    int64_t N0 = 16;
    int64_t N1 = N / N0;
    int64_t C1 = C / C0;
    int64_t shape1 = N * C1 * H * WPad * C0;
    int64_t yTileSizeElem = BytesOf(inputTile->Datatype()) == 1 ? 32 : 16;
    int64_t shape3 = H * WPad * ((C0 + yTileSizeElem - 1) / yTileSizeElem * yTileSizeElem);
    std::vector<int64_t> tmpShape = {shape1 + shape3};
    auto tmpTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), tmpShape);

    auto tmpDstTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(),
                                                      Shape{C1 * H * WPad, N1, N0, C0});
    auto inputValidShape = realInputTile->GetDynValidShape();
    auto inputValidShapeN1 = (inputValidShape[0] + N0 - 1) / N0;
    auto inputValidShapeC1 = (inputValidShape[1] + C0 - 1) / C0;
    auto inputValidShapeH = inputValidShape[2];
    auto inputValidShapeW = inputValidShape[3];

    std::vector<SymbolicScalar> dstValidShape{inputValidShapeC1 * inputValidShapeH * inputValidShapeW,
                                              inputValidShapeN1, SymbolicScalar(N0), SymbolicScalar(C0)};
    tmpDstTile->UpdateDynValidShape(dstValidShape);
    dstTensorTile->UpdateDynValidShape(dstValidShape);

    auto& op = function.AddOperation(Opcode::OP_NCHW2Fractal_Z, {realInputTile}, {tmpDstTile, tmpTile});
    for (int i = 0; i < SHAPE_DIM4; i++) {
        tileParams[i] = SymbolicScalar(transDataTileInfoPara.inputTileInfo.offset[i]);
    }
    tileParams[4] = transDataPara.groupIdx;
    tileParams[5] = transDataPara.group;
    op.SetAttribute(OpAttributeKey::transDataOffset, tileParams);
    std::shared_ptr<LogicalTensor> realDstTile = tmpDstTile->View(function, Shape{C1 * H * W, N1, N0, C0},
                                                                  Offset{0, 0, 0, 0});
    [[maybe_unused]] auto& copyOp = function.AddOperation(Opcode::OP_REGISTER_COPY, {realDstTile}, {dstTensorTile});
}

std::shared_ptr<LogicalTensor> GetNDC1HWC0DstTile(Function& function, const LogicalTensorPtr& dstTensor,
                                                  const std::shared_ptr<LogicalTensor>& inputTile,
                                                  TransDataTileInfoPara& transDataTileInfoPara, int64_t C0)
{
    const auto& inputShape = inputTile->GetShape();
    Shape dstShape = {inputShape[0], inputShape[1], inputShape[2] / C0, inputShape[3], inputShape[4], C0};
    Offset dstOffset = {
        transDataTileInfoPara.inputTileInfo.offset[0],      transDataTileInfoPara.inputTileInfo.offset[1],
        transDataTileInfoPara.inputTileInfo.offset[2] / C0, transDataTileInfoPara.inputTileInfo.offset[3],
        transDataTileInfoPara.inputTileInfo.offset[4],      0};

    return dstTensor->View(function, dstShape, dstOffset);
}

void HandleNDC1HWC0Format(Function& function, const LogicalTensorPtr& dstTensor, const LogicalTensorPtr& inputTile,
                          std::vector<SymbolicScalar>& tileParams, TransDataTileInfoPara& transDataTileInfoPara)
{
    int64_t C0 = BLOCK_SIZE / BytesOf(inputTile->Datatype());
    auto realInputTile = transDataPad<TileOpFormat::TILEOP_NDC1HWC0>(function, inputTile, C0);
    auto realInputShape = realInputTile->GetShape();
    auto dstTensorTile = GetNDC1HWC0DstTile(function, dstTensor, realInputTile, transDataTileInfoPara, C0);

    int64_t D = realInputShape[1];
    int64_t C = realInputShape[2];
    int64_t H = realInputShape[3];
    int64_t W = realInputShape[4];
    int64_t WPad = CeilDiv(W, C0) * C0;
    int64_t C1 = C / C0;
    int64_t yTileSizeElem = BytesOf(inputTile->Datatype()) == 1 ? 32 : 16;
    int64_t shape2 = H * WPad * ((C0 + yTileSizeElem - 1) / yTileSizeElem * yTileSizeElem);
    std::vector<int64_t> tmpShape = {shape2};
    auto tmpTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), tmpShape);
    auto tmpDstTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), Shape{D, C1, H, WPad, C0});
    auto inputValidShape = realInputTile->GetDynValidShape();
    auto vN = inputValidShape[0];
    auto vD = inputValidShape[1];
    auto vC = inputValidShape[2];
    auto vH = inputValidShape[3];
    auto vW = inputValidShape[4];
    auto vC1 = (vC + C0 - 1) / C0;

    tmpDstTile->UpdateDynValidShape({vN * vD, vC1, vH, vW, SymbolicScalar(C0)});
    dstTensorTile->UpdateDynValidShape({vN, vD, vC1, vH, vW, SymbolicScalar(C0)});

    auto& op = function.AddOperation(Opcode::OP_NCDHW2NDC1HWC0, {realInputTile}, {tmpDstTile, tmpTile});
    for (int i = 0; i < SHAPE_DIM5; i++) {
        tileParams[i] = SymbolicScalar(transDataTileInfoPara.inputTileInfo.offset[i]);
    }
    op.SetAttribute(OpAttributeKey::transDataOffset, tileParams);
    std::shared_ptr<LogicalTensor> realDstTile = tmpDstTile->View(function, Shape{D, C1, H, W, C0},
                                                                  Offset{0, 0, 0, 0, 0});

    [[maybe_unused]] auto& reshapeOp = function.AddOperation("TILE_RESHAPE", {realDstTile}, {dstTensorTile});
}

std::shared_ptr<LogicalTensor> GetNCHWDstTile(Function& function, const LogicalTensorPtr& dstTensor,
                                              const std::shared_ptr<LogicalTensor>& inputTile,
                                              const TransDataTileInfoPara& transDataTileInfoPara,
                                              const TransDataPara& transDataPara, int64_t C0)
{
    auto& inputShape = inputTile->GetShape();
    int64_t shape1 = inputShape[1] * C0;

    int dstPerGroupC = dstTensor->GetShape()[1] / transDataPara.group;
    int inputPerGroupC = (dstPerGroupC + C0 - 1) / C0 * C0;
    int offsetSuffix = (transDataTileInfoPara.inputTileInfo.offset[1] * C0) % inputPerGroupC;
    int dstCOffset = dstPerGroupC * transDataPara.groupIdx + offsetSuffix;
    if ((offsetSuffix + shape1) > dstPerGroupC) {
        shape1 = dstPerGroupC - offsetSuffix;
    }

    Shape dstShape = {inputShape[0], shape1, inputShape[2], inputShape[3]};
    Offset dstOffset = {transDataTileInfoPara.inputTileInfo.offset[0], dstCOffset,
                        transDataTileInfoPara.inputTileInfo.offset[2], transDataTileInfoPara.inputTileInfo.offset[3]};

    return dstTensor->View(function, dstShape, dstOffset);
}

void HandleNCHW5DimFormat(Function& function, const LogicalTensorPtr& dstTensor, const LogicalTensorPtr& inputTile,
                          std::vector<SymbolicScalar>& tileParams, TransDataTileInfoPara& transDataTileInfoPara,
                          int64_t inputPerGroup, const TransDataPara& transDataPara)
{
    int64_t N = inputTile->GetShape()[0];
    int64_t C1 = inputTile->GetShape()[1];
    int64_t H = inputTile->GetShape()[2];
    int64_t W = inputTile->GetShape()[3];
    int64_t C0 = BLOCK_SIZE / BytesOf(inputTile->Datatype());
    int64_t WPad = CeilDiv(W, C0) * C0;

    int64_t yTileSizeElem = BytesOf(inputTile->Datatype()) == 1 ? 32 : 16;
    int64_t shape2 = C0 * ((H * WPad + yTileSizeElem - 1) / yTileSizeElem * yTileSizeElem);
    std::vector<int64_t> tmpShape = {shape2};
    auto tmpTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), tmpShape);

    for (int i = 0; i < SHAPE_DIM5; i++) {
        tileParams[i] = SymbolicScalar(transDataTileInfoPara.inputTileInfo.offset[i]);
    }
    tileParams[5] = transDataPara.groupIdx;
    tileParams[6] = transDataPara.group;
    int64_t tmp = transDataTileInfoPara.inputTileInfo.offset[1] + transDataTileInfoPara.inputTileInfo.shape[1];
    if (tmp != (transDataPara.groupIdx + 1) * inputPerGroup) {
        tileParams[7] = 0;
    }

    auto dstTensorTile = GetNCHWDstTile(function, dstTensor, inputTile, transDataTileInfoPara, transDataPara, C0);
    auto tmpDstTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), Shape{N, C1 * C0, H, WPad});
    tmpDstTile->UpdateDynValidShape(dstTensorTile->GetDynValidShape());
    auto realInput = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), Shape{N, C1, H, WPad, C0});
    realInput->UpdateDynValidShape(inputTile->GetDynValidShape());
    auto realInputTile = realInput->View(function, inputTile->GetShape(), Offset{0, 0, 0, 0, 0});

    [[maybe_unused]] auto& copyOp = function.AddOperation(Opcode::OP_REGISTER_COPY, {inputTile}, {realInputTile});
    auto& op = function.AddOperation(Opcode::OP_NC1HWC02NCHW, {realInput}, {tmpDstTile, tmpTile});
    op.SetAttribute(OpAttributeKey::transDataOffset, tileParams);
    auto tmpDstValidTile = tmpDstTile->View(function, dstTensorTile->GetShape(), Offset{0, 0, 0, 0});
    [[maybe_unused]] auto& copyOp2 = function.AddOperation(Opcode::OP_REGISTER_COPY, {tmpDstValidTile},
                                                           {dstTensorTile});
}

std::shared_ptr<LogicalTensor> GetNCDHWDstTile(Function& function, const LogicalTensorPtr& dstTensor,
                                               const std::shared_ptr<LogicalTensor>& inputTile,
                                               const TransDataTileInfoPara& transDataTileInfoPara,
                                               const TransDataPara& transDataPara, int64_t C0)
{
    auto& inputShape = inputTile->GetShape();
    int64_t shape1 = inputShape[2] * C0;

    int dstPerGroupC = dstTensor->GetShape()[2] / transDataPara.group;
    int inputPerGroupC = (dstPerGroupC + C0 - 1) / C0 * C0;
    int offsetSuffix = (transDataTileInfoPara.inputTileInfo.offset[2] * C0) % inputPerGroupC;
    int dstCOffset = dstPerGroupC * transDataPara.groupIdx + offsetSuffix;
    if ((offsetSuffix + shape1) > dstPerGroupC) {
        shape1 = dstPerGroupC - offsetSuffix;
    }

    Shape dstShape = {inputShape[0], inputShape[1], shape1, inputShape[3], inputShape[4]};
    Offset dstOffset = {transDataTileInfoPara.inputTileInfo.offset[0], transDataTileInfoPara.inputTileInfo.offset[1],
                        dstCOffset, transDataTileInfoPara.inputTileInfo.offset[3],
                        transDataTileInfoPara.inputTileInfo.offset[4]};

    return dstTensor->View(function, dstShape, dstOffset);
}

void HandleNCDHW6DimFormat(Function& function, const LogicalTensorPtr& dstTensor, const LogicalTensorPtr& inputTile,
                           std::vector<SymbolicScalar>& tileParams, const TransDataTileInfoPara& transDataTileInfoPara,
                           int64_t inputPerGroup, const TransDataPara& transDataPara)
{
    int64_t D = inputTile->GetShape()[1];
    int64_t C1 = inputTile->GetShape()[2];
    int64_t H = inputTile->GetShape()[3];
    int64_t W = inputTile->GetShape()[4];
    int64_t C0 = BLOCK_SIZE / BytesOf(inputTile->Datatype());
    int64_t WPad = CeilDiv(W, C0) * C0;

    int64_t yTileSizeElem = BytesOf(inputTile->Datatype()) == 1 ? 32 : 16;
    int64_t shape2 = C0 * ((H * WPad + yTileSizeElem - 1) / yTileSizeElem * yTileSizeElem);
    std::vector<int64_t> tmpShape = {shape2};
    auto tmpTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), tmpShape);

    for (int i = 0; i < SHAPE_DIM6; i++) {
        tileParams[i] = SymbolicScalar(transDataTileInfoPara.inputTileInfo.offset[i]);
    }
    tileParams[6] = transDataPara.groupIdx;
    tileParams[7] = transDataPara.group;
    int64_t tmp = transDataTileInfoPara.inputTileInfo.offset[2] + transDataTileInfoPara.inputTileInfo.shape[2];
    if (tmp != (transDataPara.groupIdx + 1) * inputPerGroup) {
        tileParams[8] = 0;
    }

    auto dstTensorTile = GetNCDHWDstTile(function, dstTensor, inputTile, transDataTileInfoPara, transDataPara, C0);
    auto inputValidShape = inputTile->GetDynValidShape();
    auto vN = inputValidShape[0];
    auto vD = inputValidShape[1];
    auto vC1 = inputValidShape[2];
    auto vH = inputValidShape[3];
    auto vW = inputValidShape[4];
    auto vC0 = inputValidShape[5];
    std::vector<SymbolicScalar> realInputValidShape = {vN * vD, vC1, vH, vW, vC0};

    auto reshapedTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), Shape{D, C1, H, W, C0});
    [[maybe_unused]] auto& reshapeOp = function.AddOperation("TILE_RESHAPE", {inputTile}, {reshapedTile});
    auto realInput = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), Shape{D, C1, H, WPad, C0});
    reshapedTile->UpdateDynValidShape(realInputValidShape);
    realInput->UpdateDynValidShape(realInputValidShape);
    auto realInputTile = realInput->View(function, Shape{D, C1, H, W, C0}, Offset{0, 0, 0, 0, 0});
    [[maybe_unused]] auto& copyOp = function.AddOperation(Opcode::OP_REGISTER_COPY, {reshapedTile}, {realInputTile});
    auto tmpDstTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), Shape{1, D, C1 * C0, H, WPad});
    tmpDstTile->UpdateDynValidShape(dstTensorTile->GetDynValidShape());
    auto& op = function.AddOperation(Opcode::OP_NDC1HWC02NCDHW, {realInput}, {tmpDstTile, tmpTile});
    op.SetAttribute(OpAttributeKey::transDataOffset, tileParams);
    auto tmpDstValidTile = tmpDstTile->View(function, dstTensorTile->GetShape(), Offset{0, 0, 0, 0, 0});
    [[maybe_unused]] auto& copyOp2 = function.AddOperation(Opcode::OP_REGISTER_COPY, {tmpDstValidTile},
                                                           {dstTensorTile});
}

template <TileOpFormat T>
void InnerTransDataND(size_t cur, Function& function, const TileShape& tileShape, const TransDataPara& transDataPara,
                      TransDataTileInfoPara& transDataTileInfoPara)
{
    const LogicalTensorPtr& input = transDataPara.input;
    const LogicalTensorPtr& dstTensor = transDataPara.dstTensor;
    std::vector<SymbolicScalar> tileParams = transDataPara.tileParams;
    const int group = transDataPara.group;
    const int groupIdx = transDataPara.groupIdx;
    auto vecTile = tileShape.GetVecTile();
    int inputSize = input->GetShape().size();

    std::unordered_map<int64_t, int64_t> format2InputAxis = {{SHAPE_DIM5, 1}, {SHAPE_DIM6, 2}};
    int64_t inputGroupAxis = format2InputAxis[inputSize];
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, group > 0) << "The group is not valid !";
    int64_t inputPerGroup = input->GetShape()[inputGroupAxis] / group;

    if (cur == input->GetShape().size()) {
        int64_t offsetSuffix = transDataTileInfoPara.inputTileInfo.offset[inputGroupAxis] % inputPerGroup;
        transDataTileInfoPara.inputTileInfo.offset[inputGroupAxis] = groupIdx * inputPerGroup + offsetSuffix;
        std::shared_ptr<LogicalTensor> inputTile = input->View(function, transDataTileInfoPara.inputTileInfo.shape,
                                                               transDataTileInfoPara.inputTileInfo.offset);

        switch (inputSize) {
            case SHAPE_DIM5:
                HandleNCHW5DimFormat(function, dstTensor, inputTile, tileParams, transDataTileInfoPara, inputPerGroup,
                                     transDataPara);
                return;
            case SHAPE_DIM6:
                HandleNCDHW6DimFormat(function, dstTensor, inputTile, tileParams, transDataTileInfoPara, inputPerGroup,
                                      transDataPara);
                return;
            default:
                CHECK(VectorErrorCode::ERR_PARAM_INVALID, false) << "The transDataType is not supported";
        }
    }

    int64_t tmpTile = vecTile[cur];
    int64_t curShapeLen = cur == static_cast<size_t>(inputGroupAxis) ? inputPerGroup : input->GetShape()[cur];

    for (int i = 0; i < curShapeLen; i += tmpTile) {
        transDataTileInfoPara.inputTileInfo.offset[cur] = i;
        transDataTileInfoPara.inputTileInfo.shape[cur] = std::min(curShapeLen - i, tmpTile);
        InnerTransDataND<T>(cur + 1, function, tileShape, transDataPara, transDataTileInfoPara);
    }
}

std::shared_ptr<LogicalTensor> GetFractalZ3DDstTile(Function& function, const LogicalTensorPtr& dstTensor,
                                                    const std::shared_ptr<LogicalTensor>& inputTile,
                                                    const TransDataTileInfoPara& transDataTileInfoPara,
                                                    const TransDataPara& transDataPara, int64_t C0)
{
    const auto& inputShape = inputTile->GetShape();
    int64_t N0 = 16;
    int64_t N1 = inputShape[0] / N0;
    int64_t shape0 = inputShape[1] / C0 * inputShape[2] * inputShape[3] * inputShape[4];
    Shape dstShape = {shape0, N1, N0, C0};

    auto originInput = transDataPara.input;
    auto originN = originInput->GetShape()[0];
    auto originC = originInput->GetShape()[1];
    auto originD = originInput->GetShape()[2];
    auto originH = originInput->GetShape()[3];
    auto originW = originInput->GetShape()[4];

    int64_t originPerGroupN = originN / transDataPara.group;
    int64_t originPerGroupPadN = (originPerGroupN + N0 - 1) / N0 * N0;
    int64_t originC1 = (originC + C0 - 1) / C0;
    int64_t inputOffsetN = transDataTileInfoPara.inputTileInfo.offset[0];
    int64_t inputOffsetC1 = transDataTileInfoPara.inputTileInfo.offset[1] / C0;
    int64_t inputOffsetD = transDataTileInfoPara.inputTileInfo.offset[2];
    int64_t inputOffsetH = transDataTileInfoPara.inputTileInfo.offset[3];
    int64_t inputOffsetW = transDataTileInfoPara.inputTileInfo.offset[4];
    int64_t offset0 = transDataPara.groupIdx * originD * originC1 * originH * originW +
                      inputOffsetD * originC1 * originH * originW + inputOffsetC1 * originH * originW +
                      inputOffsetH * originW + inputOffsetW;
    int64_t offset1 = (inputOffsetN % originPerGroupPadN) / N0;
    Offset dstOffset = {offset0, offset1, 0, 0};

    return dstTensor->View(function, dstShape, dstOffset);
}

void HandleFractalZ3DFormat(Function& function, const LogicalTensorPtr& dstTensor, const LogicalTensorPtr& inputTile,
                            std::vector<SymbolicScalar>& tileParams, const TransDataTileInfoPara& transDataTileInfoPara,
                            const TransDataPara& transDataPara)
{
    int64_t C0 = BLOCK_SIZE / BytesOf(inputTile->Datatype());
    auto realInputTile = transDataPad<TileOpFormat::TILEOP_FRACTAL_Z_3D>(function, inputTile, C0);
    auto realInputShape = realInputTile->GetShape();
    auto dstTensorTile = GetFractalZ3DDstTile(function, dstTensor, realInputTile, transDataTileInfoPara, transDataPara,
                                              C0);

    int64_t N = realInputShape[0];
    int64_t C = realInputShape[1];
    int64_t D = realInputShape[2];
    int64_t H = realInputShape[3];
    int64_t W = realInputShape[4];
    int64_t WPad = CeilDiv(W, C0) * C0;
    int64_t N0 = 16;
    int64_t N1 = N / N0;
    int64_t C1 = C / C0;

    int64_t tmp1 = N * C1 * C0 * H * WPad;
    int64_t yTileSizeElem = BytesOf(inputTile->Datatype()) == 1 ? 32 : 16;
    int64_t tmp2 = H * WPad * ((C0 + yTileSizeElem - 1) / yTileSizeElem * yTileSizeElem);
    int64_t shape2 = tmp1 + std::max(tmp1, tmp2);
    std::vector<int64_t> tmpShape = {shape2};
    auto tmpTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), tmpShape);

    auto tmpDstTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(),
                                                      Shape{D * C1 * H * WPad, N1, N0, C0});
    auto inputValidShape = realInputTile->GetDynValidShape();
    auto inputValidShapeN1 = (inputValidShape[0] + N0 - 1) / N0;
    auto inputValidShapeC1 = (inputValidShape[1] + C0 - 1) / C0;
    auto inputValidShapeD = inputValidShape[2];
    auto inputValidShapeH = inputValidShape[3];
    auto inputValidShapeW = inputValidShape[4];

    std::vector<SymbolicScalar> dstValidShape{
        inputValidShapeD * inputValidShapeC1 * inputValidShapeH * inputValidShapeW, inputValidShapeN1,
        SymbolicScalar(N0), SymbolicScalar(C0)};
    tmpDstTile->UpdateDynValidShape(dstValidShape);
    dstTensorTile->UpdateDynValidShape(dstValidShape);

    auto& op = function.AddOperation(Opcode::OP_NCDHW2FRACTAL_Z_3D, {realInputTile}, {tmpDstTile, tmpTile});
    for (int i = 0; i < SHAPE_DIM5; i++) {
        tileParams[i] = SymbolicScalar(transDataTileInfoPara.inputTileInfo.offset[i]);
    }
    tileParams[5] = transDataPara.groupIdx;
    tileParams[6] = transDataPara.group;
    op.SetAttribute(OpAttributeKey::transDataOffset, tileParams);
    std::shared_ptr<LogicalTensor> realDstTile = tmpDstTile->View(function, Shape{D * C1 * H * W, N1, N0, C0},
                                                                  Offset{0, 0, 0, 0});
    [[maybe_unused]] auto& copyOp = function.AddOperation(Opcode::OP_REGISTER_COPY, {realDstTile}, {dstTensorTile});
}

template <TileOpFormat T>
void InnerTransData(size_t cur, Function& function, const TileShape& tileShape, const TransDataPara& transDataPara,
                    TransDataTileInfoPara& transDataTileInfoPara)
{
    const LogicalTensorPtr& input = transDataPara.input;
    const LogicalTensorPtr& dstTensor = transDataPara.dstTensor;
    std::vector<SymbolicScalar> tileParams = transDataPara.tileParams;
    const int group = transDataPara.group;
    const int groupIdx = transDataPara.groupIdx;
    auto vecTile = tileShape.GetVecTile();

    int64_t C0 = BLOCK_SIZE / BytesOf(input->Datatype());
    int64_t N0 = 16;
    std::unordered_map<TileOpFormat, int64_t> format2InputAxis = {{TileOpFormat::TILEOP_NC1HWC0, 1},
                                                                  {TileOpFormat::TILEOP_FRACTAL_Z, 0},
                                                                  {TileOpFormat::TILEOP_NDC1HWC0, 2},
                                                                  {TileOpFormat::TILEOP_FRACTAL_Z_3D, 0}};
    int64_t inputGroupAxis = format2InputAxis[T];
    std::unordered_map<TileOpFormat, int64_t> format2OutputAxis = {{TileOpFormat::TILEOP_NC1HWC0, 1},
                                                                   {TileOpFormat::TILEOP_FRACTAL_Z, 1},
                                                                   {TileOpFormat::TILEOP_NDC1HWC0, 2},
                                                                   {TileOpFormat::TILEOP_FRACTAL_Z_3D, 1}};
    int64_t outputGroupAxis = format2OutputAxis[T];
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, group > 0) << "The group is not valid !";
    int64_t inputPerGroup = input->GetShape()[inputGroupAxis] / group;
    int64_t factor = (T == TileOpFormat::TILEOP_FRACTAL_Z || T == TileOpFormat::TILEOP_FRACTAL_Z_3D) ? N0 : C0;
    bool isFractalZ = T == TileOpFormat::TILEOP_FRACTAL_Z || T == TileOpFormat::TILEOP_FRACTAL_Z_3D;
    int64_t dstPerGroup = isFractalZ ? dstTensor->GetShape()[outputGroupAxis] * factor :
                                       dstTensor->GetShape()[outputGroupAxis] / group * factor;

    if (cur == input->GetShape().size()) {
        int64_t offsetSuffix = transDataTileInfoPara.inputTileInfo.offset[inputGroupAxis] % dstPerGroup;
        transDataTileInfoPara.inputTileInfo.offset[inputGroupAxis] = groupIdx * inputPerGroup + offsetSuffix;
        std::shared_ptr<LogicalTensor> inputTile = input->View(function, transDataTileInfoPara.inputTileInfo.shape,
                                                               transDataTileInfoPara.inputTileInfo.offset);
        transDataTileInfoPara.inputTileInfo.offset[inputGroupAxis] = groupIdx * dstPerGroup + offsetSuffix;

        switch (T) {
            case TileOpFormat::TILEOP_NC1HWC0:
                HandleNC1HWC0Format(function, dstTensor, inputTile, tileParams, transDataTileInfoPara);
                return;
            case TileOpFormat::TILEOP_FRACTAL_Z:
                HandleFractalZFormat(function, dstTensor, inputTile, tileParams, transDataTileInfoPara, transDataPara);
                return;
            case TileOpFormat::TILEOP_NDC1HWC0:
                HandleNDC1HWC0Format(function, dstTensor, inputTile, tileParams, transDataTileInfoPara);
                return;
            case TileOpFormat::TILEOP_FRACTAL_Z_3D:
                HandleFractalZ3DFormat(function, dstTensor, inputTile, tileParams, transDataTileInfoPara,
                                       transDataPara);
                return;
            default:
                CHECK(VectorErrorCode::ERR_PARAM_INVALID, false) << "The transDataType is not supported";
        }
    }

    int64_t tmpTile = vecTile[cur];
    int64_t curShapeLen = cur == static_cast<size_t>(inputGroupAxis) ? inputPerGroup : input->GetShape()[cur];

    for (int i = 0; i < curShapeLen; i += tmpTile) {
        transDataTileInfoPara.inputTileInfo.offset[cur] = i;
        transDataTileInfoPara.inputTileInfo.shape[cur] = std::min(curShapeLen - i, tmpTile);
        InnerTransData<T>(cur + 1, function, tileShape, transDataPara, transDataTileInfoPara);
    }
}

std::shared_ptr<LogicalTensor> GetFzNCHWDstTile(Function& function, const LogicalTensorPtr& dstTensor,
                                                const std::shared_ptr<LogicalTensor>& inputTile,
                                                const TransDataTileInfoPara& transDataTileInfoPara,
                                                const TransDataPara& transDataPara, int64_t C0)
{
    auto& inputShape = inputTile->GetShape();
    int N0 = 16;
    int DSTN = dstTensor->GetShape()[0];
    int DSTC = dstTensor->GetShape()[1];
    int DSTW = dstTensor->GetShape()[3];
    int DSTPerGroupN = DSTN / transDataPara.group;

    int N1Offset = transDataTileInfoPara.inputTileInfo.offset[1];

    int64_t dstTileN = std::min(inputShape[1] * N0, static_cast<int64_t>(DSTPerGroupN - N1Offset * N0));
    int64_t dstTileC = C0;
    int64_t dstTileH = 1;
    int64_t dstTileW = inputShape[0];

    int totalC1HWOffset = transDataTileInfoPara.inputTileInfo.offset[0]; // 1 * 1 * w

    int suffixOffset = totalC1HWOffset % DSTW;
    int C1Offset = transDataPara.C1Idx;
    int dstCOffset = C1Offset * C0;
    int dstHOffset = transDataPara.HIdx;
    int dstWOffset = suffixOffset;

    int dstNOffset = transDataPara.groupIdx * DSTPerGroupN + N1Offset * N0;

    if ((dstCOffset + dstTileC) > DSTC) {
        dstTileC = DSTC - dstCOffset;
    }

    Shape dstShape = {dstTileN, dstTileC, dstTileH, dstTileW};
    Offset dstOffset = {dstNOffset, dstCOffset, dstHOffset, dstWOffset};

    return dstTensor->View(function, dstShape, dstOffset);
}

std::shared_ptr<LogicalTensor> GetFz3DNCDHWDstTile(Function& function, const LogicalTensorPtr& dstTensor,
                                                   const std::shared_ptr<LogicalTensor>& inputTile,
                                                   const TransDataTileInfoPara& transDataTileInfoPara,
                                                   const TransDataPara& transDataPara, int64_t C0)
{
    auto& inputShape = inputTile->GetShape();
    int N0 = 16;
    int DSTN = dstTensor->GetShape()[0];
    int DSTC = dstTensor->GetShape()[1];
    int DSTW = dstTensor->GetShape()[4];
    int DSTPerGroupN = DSTN / transDataPara.group;

    int N1Offset = transDataTileInfoPara.inputTileInfo.offset[1];

    int64_t dstTileN = std::min(inputShape[1] * N0, static_cast<int64_t>(DSTPerGroupN - N1Offset * N0));
    int64_t dstTileC = C0;
    int64_t dstTileD = 1;
    int64_t dstTileH = 1;
    int64_t dstTileW = inputShape[0];

    int totalDC1HWOffset = transDataTileInfoPara.inputTileInfo.offset[0];

    int suffixOffset = totalDC1HWOffset % DSTW;
    int dstDOffset = transDataPara.DIdx;
    int dstCOffset = transDataPara.C1Idx * C0;
    int dstHOffset = transDataPara.HIdx;
    int dstWOffset = suffixOffset;
    int dstNOffset = transDataPara.groupIdx * DSTPerGroupN + N1Offset * N0;

    if ((dstCOffset + dstTileC) > DSTC) {
        dstTileC = DSTC - dstCOffset;
    }

    Shape dstShape = {dstTileN, dstTileC, dstTileD, dstTileH, dstTileW};
    Offset dstOffset = {dstNOffset, dstCOffset, dstDOffset, dstHOffset, dstWOffset};

    return dstTensor->View(function, dstShape, dstOffset);
}

void HandleFractalZ2NCHW(Function& function, const LogicalTensorPtr& dstTensor, const LogicalTensorPtr& inputTile,
                         std::vector<SymbolicScalar>& tileParams, const TransDataTileInfoPara& transDataTileInfoPara,
                         [[maybe_unused]] int64_t dstW, const TransDataPara& transDataPara)
{
    int64_t C0 = BLOCK_SIZE / BytesOf(inputTile->Datatype());
    int64_t N0 = 16;
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, inputTile->GetShape()[2] == N0 && inputTile->GetShape()[3] == C0)
        << "The transdata NZ input shape must be [C1HW, N1, 16, C0]!";
    int64_t N1 = inputTile->GetShape()[1];
    int64_t C1HW = inputTile->GetShape()[0];
    int64_t H = 1;
    int64_t C1 = 1;
    int64_t WPad = CeilDiv(C1HW, C0) * C0;

    int64_t yTileSizeElem = BytesOf(inputTile->Datatype()) == 1 ? 32 : 16;
    int64_t shape1 = WPad * N1 * N0 * C0;
    int64_t shape2 = C0 * ((WPad + yTileSizeElem - 1) / yTileSizeElem * yTileSizeElem);
    std::vector<int64_t> tmpShape = {shape1 + shape2};
    auto tmpTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), tmpShape);

    for (int i = 0; i < SHAPE_DIM4; i++) {
        tileParams[i] = SymbolicScalar(transDataTileInfoPara.inputTileInfo.offset[i]);
    }
    tileParams[4] = transDataPara.groupIdx;
    tileParams[5] = transDataPara.group;

    auto dstTensorTile = GetFzNCHWDstTile(function, dstTensor, inputTile, transDataTileInfoPara, transDataPara, C0);
    auto tmpDstTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(),
                                                      Shape{N1 * N0, C1 * C0, H, WPad});
    tmpDstTile->UpdateDynValidShape(dstTensorTile->GetDynValidShape());
    auto realInput = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), Shape{WPad, N1, N0, C0});
    realInput->UpdateDynValidShape(inputTile->GetDynValidShape());
    auto realInputTile = realInput->View(function, inputTile->GetShape(), Offset{0, 0, 0, 0});

    [[maybe_unused]] auto& copyOp1 = function.AddOperation(Opcode::OP_REGISTER_COPY, {inputTile}, {realInputTile});
    auto& op = function.AddOperation(Opcode::OP_FractalZ2NCHW, {realInput}, {tmpDstTile, tmpTile});
    op.SetAttribute(OpAttributeKey::transDataOffset, tileParams);
    auto tmpDstValidTile = tmpDstTile->View(function, dstTensorTile->GetShape(), Offset{0, 0, 0, 0});
    [[maybe_unused]] auto& copyOp2 = function.AddOperation(Opcode::OP_REGISTER_COPY, {tmpDstValidTile},
                                                           {dstTensorTile});
}

void HandleFractalZ3D2NCDHW(Function& function, const LogicalTensorPtr& dstTensor, const LogicalTensorPtr& inputTile,
                            std::vector<SymbolicScalar>& tileParams, const TransDataTileInfoPara& transDataTileInfoPara,
                            [[maybe_unused]] int64_t dstW, const TransDataPara& transDataPara)
{
    int64_t C0 = BLOCK_SIZE / BytesOf(inputTile->Datatype());
    int64_t N0 = 16;
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, inputTile->GetShape()[2] == N0 && inputTile->GetShape()[3] == C0)
        << "The transdata NZ input shape must be [DC1HW, N1, 16, C0]!";

    auto inputShape = inputTile->GetShape();
    int64_t N1 = inputShape[1];
    int64_t N = N1 * N0;
    int64_t D = 1;
    int64_t H = 1;
    int64_t DC1HW = inputShape[0];
    int64_t WPad = CeilDiv(DC1HW, C0) * C0;
    int64_t yTileSizeElem = BytesOf(inputTile->Datatype()) == 1 ? 32 : 16;
    int64_t ncplaneSize = N * D * H * WPad * ((C0 + yTileSizeElem - 1) / yTileSizeElem * yTileSizeElem);
    int64_t tmpAreaSize = C0 * ((H * WPad + yTileSizeElem - 1) / yTileSizeElem * yTileSizeElem);
    int64_t scratchSize = 2 * ncplaneSize + tmpAreaSize;
    std::vector<int64_t> tmpShape = {scratchSize};
    auto tmpTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), tmpShape);

    auto realInputTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), Shape{WPad, N1, N0, C0});
    realInputTile->UpdateDynValidShape(inputTile->GetDynValidShape());
    auto realInput = realInputTile->View(function, inputTile->GetShape(), Offset{0, 0, 0, 0});

    auto dstTensorTile = GetFz3DNCDHWDstTile(function, dstTensor, inputTile, transDataTileInfoPara, transDataPara, C0);

    auto tmpDstTile = std::make_shared<LogicalTensor>(function, inputTile->Datatype(), Shape{N, C0, D, H, WPad});
    tmpDstTile->UpdateDynValidShape(dstTensorTile->GetDynValidShape());

    for (int i = 0; i < SHAPE_DIM4; i++) {
        tileParams[i] = SymbolicScalar(transDataTileInfoPara.inputTileInfo.offset[i]);
    }
    tileParams[4] = transDataPara.groupIdx;
    tileParams[5] = transDataPara.group;

    [[maybe_unused]] auto& copyOp = function.AddOperation(Opcode::OP_REGISTER_COPY, {inputTile}, {realInput});
    auto& op = function.AddOperation(Opcode::OP_FractalZ3D2NCDHW, {realInputTile}, {tmpDstTile, tmpTile});
    op.SetAttribute(OpAttributeKey::transDataOffset, tileParams);

    std::shared_ptr<LogicalTensor> realDstTile = tmpDstTile->View(function, dstTensorTile->GetShape(),
                                                                  Offset{0, 0, 0, 0, 0});
    [[maybe_unused]] auto& copyOp1 = function.AddOperation(Opcode::OP_REGISTER_COPY, {realDstTile}, {dstTensorTile});
}

template <TileOpFormat T>
void InnerTransDataFz2ND(size_t cur, Function& function, const TileShape& tileShape, const TransDataPara& transDataPara,
                         TransDataTileInfoPara& transDataTileInfoPara)
{
    const LogicalTensorPtr& input = transDataPara.input;
    const LogicalTensorPtr& dstTensor = transDataPara.dstTensor;
    std::vector<SymbolicScalar> tileParams = transDataPara.tileParams;
    int64_t C0 = BLOCK_SIZE / BytesOf(input->Datatype());
    const int dstC1 = (dstTensor->GetShape()[1] + C0 - 1) / C0;
    const int dstH = dstTensor->GetShape()[2];
    const int dstW = dstTensor->GetShape()[3];
    const int groupIdx = transDataPara.groupIdx;
    const int C1Idx = transDataPara.C1Idx;
    const int HIdx = transDataPara.HIdx;
    auto vecTile = tileShape.GetVecTile();
    int outputSize = dstTensor->GetShape().size();

    std::unordered_map<int64_t, int64_t> format2DstAxis = {{SHAPE_DIM4, 0}, {SHAPE_DIM5, 0}};
    int64_t outputAxis = format2DstAxis[outputSize];
    int64_t tmpOffset = groupIdx * dstC1 * dstH * dstW + C1Idx * dstH * dstW + HIdx * dstW;

    if (cur == input->GetShape().size()) {
        int64_t offsetSuffix = transDataTileInfoPara.inputTileInfo.offset[0] % dstW;
        transDataTileInfoPara.inputTileInfo.offset[0] = tmpOffset + offsetSuffix;
        std::shared_ptr<LogicalTensor> inputTile = input->View(function, transDataTileInfoPara.inputTileInfo.shape,
                                                               transDataTileInfoPara.inputTileInfo.offset);

        switch (outputSize) {
            case SHAPE_DIM4:
                HandleFractalZ2NCHW(function, dstTensor, inputTile, tileParams, transDataTileInfoPara, dstW,
                                    transDataPara);
                return;
            default:
                CHECK(VectorErrorCode::ERR_PARAM_INVALID, false) << "The transDataType is not supported";
        }
    }

    int64_t tmpTile = vecTile[cur];
    int64_t curShapeLen = cur == static_cast<size_t>(outputAxis) ? dstW : input->GetShape()[cur];

    for (int i = 0; i < curShapeLen; i += tmpTile) {
        transDataTileInfoPara.inputTileInfo.offset[cur] = i;
        transDataTileInfoPara.inputTileInfo.shape[cur] = std::min(curShapeLen - i, tmpTile);
        InnerTransDataFz2ND<T>(cur + 1, function, tileShape, transDataPara, transDataTileInfoPara);
    }
}

template <TileOpFormat T>
void InnerTransDataFz3D2ND(size_t cur, Function& function, const TileShape& tileShape,
                           const TransDataPara& transDataPara, TransDataTileInfoPara& transDataTileInfoPara)
{
    const LogicalTensorPtr& input = transDataPara.input;
    const LogicalTensorPtr& dstTensor = transDataPara.dstTensor;
    std::vector<SymbolicScalar> tileParams = transDataPara.tileParams;
    int64_t C0 = BLOCK_SIZE / BytesOf(input->Datatype());
    const int dstC1 = (dstTensor->GetShape()[1] + C0 - 1) / C0;
    const int dstD = dstTensor->GetShape()[2];
    const int dstH = dstTensor->GetShape()[3];
    const int dstW = dstTensor->GetShape()[4];
    const int groupIdx = transDataPara.groupIdx;
    const int DIdx = transDataPara.DIdx;
    const int C1Idx = transDataPara.C1Idx;
    const int HIdx = transDataPara.HIdx;
    auto vecTile = tileShape.GetVecTile();

    int64_t outputAxis = 0;
    int64_t tmpOffset = groupIdx * dstD * dstC1 * dstH * dstW + DIdx * dstC1 * dstH * dstW + C1Idx * dstH * dstW +
                        HIdx * dstW;

    if (cur == input->GetShape().size()) {
        int64_t offsetSuffix = transDataTileInfoPara.inputTileInfo.offset[0] % dstW;
        transDataTileInfoPara.inputTileInfo.offset[0] = tmpOffset + offsetSuffix;
        std::shared_ptr<LogicalTensor> inputTile = input->View(function, transDataTileInfoPara.inputTileInfo.shape,
                                                               transDataTileInfoPara.inputTileInfo.offset);
        HandleFractalZ3D2NCDHW(function, dstTensor, inputTile, tileParams, transDataTileInfoPara, dstW, transDataPara);
        return;
    }

    int64_t tmpTile = vecTile[cur];
    int64_t curShapeLen = cur == static_cast<size_t>(outputAxis) ? dstW : input->GetShape()[cur];

    for (int i = 0; i < curShapeLen; i += tmpTile) {
        transDataTileInfoPara.inputTileInfo.offset[cur] = i;
        transDataTileInfoPara.inputTileInfo.shape[cur] = std::min(curShapeLen - i, tmpTile);
        InnerTransDataFz3D2ND<T>(cur + 1, function, tileShape, transDataPara, transDataTileInfoPara);
    }
}

template <TileOpFormat T>
void TiledTransData(Function& function, const TileShape& tileShape, TransDataPara& transDataPara)
{
    int64_t C0 = BLOCK_SIZE / BytesOf(transDataPara.input->Datatype());
    TransDataTileInfoPara transDataTileInfoPara{
        TileInfo(transDataPara.input->GetShape().size(), transDataPara.input->GetOffset().size()),
        TileInfo(transDataPara.dstTensor->GetShape().size(), transDataPara.dstTensor->GetOffset().size())};
    int group = transDataPara.group;
    int inputShapeSize = transDataPara.input->GetShape().size();
    int outputShapeSize = transDataPara.dstTensor->GetShape().size();
    if (T == TileOpFormat::TILEOP_ND) {
        if ((inputShapeSize == SHAPE_DIM5 && outputShapeSize == SHAPE_DIM4) ||
            (inputShapeSize == SHAPE_DIM6 && outputShapeSize == SHAPE_DIM5)) {
            for (int i = 0; i < group; i++) {
                transDataPara.groupIdx = i;
                InnerTransDataND<T>(0, function, tileShape, transDataPara, transDataTileInfoPara);
            }
        } else if (inputShapeSize == SHAPE_DIM4 && outputShapeSize == SHAPE_DIM5) {
            int dstC1 = (transDataPara.dstTensor->GetShape()[1] + C0 - 1) / C0;
            int dstD = transDataPara.dstTensor->GetShape()[2];
            int dstH = transDataPara.dstTensor->GetShape()[3];
            for (int i = 0; i < group; i++) {
                transDataPara.groupIdx = i;
                for (int j = 0; j < dstD; j++) {
                    transDataPara.DIdx = j;
                    for (int k = 0; k < dstC1; k++) {
                        transDataPara.C1Idx = k;
                        for (int l = 0; l < dstH; l++) {
                            transDataPara.HIdx = l;
                            InnerTransDataFz3D2ND<T>(0, function, tileShape, transDataPara, transDataTileInfoPara);
                        }
                    }
                }
            }
        } else {
            int dstC1 = (transDataPara.dstTensor->GetShape()[1] + C0 - 1) / C0;
            int dstH = transDataPara.dstTensor->GetShape()[2];
            for (int i = 0; i < group; i++) {
                transDataPara.groupIdx = i;
                for (int j = 0; j < dstC1; j++) {
                    transDataPara.C1Idx = j;
                    for (int k = 0; k < dstH; k++) {
                        transDataPara.HIdx = k;
                        InnerTransDataFz2ND<T>(0, function, tileShape, transDataPara, transDataTileInfoPara);
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < group; i++) {
            transDataPara.groupIdx = i;
            InnerTransData<T>(0, function, tileShape, transDataPara, transDataTileInfoPara);
        }
    }
}

LogicalTensorPtr TransDataNCHW2NC1HWC0(Function& function, const LogicalTensorPtr& self, int group)
{
    Shape resultShape = self->GetShape();
    int64_t C = resultShape[1];
    int64_t C0 = BLOCK_SIZE / BytesOf(self->Datatype());
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, C0 > 0) << "The C0 is not valid !";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, group > 0) << "The group is not valid !";
    int perGroupC = C / group;
    int perGroupC1 = (perGroupC + C0 - 1) / C0;
    int totalC1 = perGroupC1 * group;
    resultShape[1] = totalC1;
    resultShape.push_back(C0);

    VecTile oriVectile = TileShape::Current().GetVecTile();
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, oriVectile.tile[1] % C0 == 0)
        << "The tileShape C should be an integer multiple of C0!";

    std::vector<SymbolicScalar> resultValidShape(self->GetDynValidShape());
    SymbolicScalar validShapeC = resultValidShape[1];
    SymbolicScalar perGroupValidShapeC = validShapeC / group;
    SymbolicScalar perGroupValidShapeC1 = (perGroupValidShapeC + C0 - 1) / C0;
    SymbolicScalar totalValidShapeC1 = perGroupValidShapeC1 * group;
    resultValidShape[1] = totalValidShapeC1;
    resultValidShape.push_back(SymbolicScalar(C0));
    auto result = std::make_shared<LogicalTensor>(function, self->Datatype(), resultShape, resultValidShape,
                                                  TileOpFormat::TILEOP_NC1HWC0);

    auto& op = function.AddOperation(Opcode::OP_NCHW2NC1HWC0, {self}, {result});
    std::vector<SymbolicScalar> tileParams = {};
    // n c h w N C H W
    for (auto j : self->GetShape()) {
        (void)j;
        tileParams.push_back(SymbolicScalar(0));
    }
    for (auto j : self->GetShape()) {
        tileParams.push_back(SymbolicScalar(j));
    }
    tileParams[5] = totalC1 * C0;
    op.SetAttribute(OpAttributeKey::transDataOffset, tileParams);
    op.SetAttribute(OP_ATTR_PREFIX + "group", group);
    return result;
}

LogicalTensorPtr TransDataNCHW2Fractal_Z(Function& function, const LogicalTensorPtr& self, int group)
{
    int64_t N = self->GetShape()[0];
    int64_t C = self->GetShape()[1];
    int64_t H = self->GetShape()[2];
    int64_t W = self->GetShape()[3];
    int64_t N0 = 16;
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, group > 0) << "The group is not valid !";
    int64_t perGroupN = N / group;
    int64_t perGroupN1 = (perGroupN + N0 - 1) / N0;
    int64_t C0 = BLOCK_SIZE / BytesOf(self->Datatype());
    int64_t C1 = (C + C0 - 1) / C0;
    Shape resultShape = {group * C1 * H * W, perGroupN1, N0, C0};
    SymbolicScalar validShapeN = self->GetDynValidShape()[0];
    SymbolicScalar validShapeC = self->GetDynValidShape()[1];
    SymbolicScalar validShapeH = self->GetDynValidShape()[2];
    SymbolicScalar validShapeW = self->GetDynValidShape()[3];
    SymbolicScalar validShapeC1 = (validShapeC + C0 - 1) / C0;
    SymbolicScalar vSPerGroupN1 = (validShapeN / group + N0 - 1) / N0;
    std::vector<SymbolicScalar> resultValidShape = {group * validShapeC1 * validShapeH * validShapeW, vSPerGroupN1, N0,
                                                    C0};

    VecTile oriVectile = TileShape::Current().GetVecTile();
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, oriVectile.tile[0] % N0 == 0)
        << "The tileShape N should be an integer multiple of N0(16)!";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, oriVectile.tile[1] == C0)
        << "The tileShape C should be equal to C0, actual is " << oriVectile.tile[1];
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, oriVectile.tile[2] == 1)
        << "The tileShape H should be equal to 1, actual is " << oriVectile.tile[2];

    auto result = std::make_shared<LogicalTensor>(function, self->Datatype(), resultShape, resultValidShape,
                                                  TileOpFormat::TILEOP_FRACTAL_Z);
    auto& op = function.AddOperation(Opcode::OP_NCHW2Fractal_Z, {self}, {result});
    std::vector<SymbolicScalar> tileParams = {};
    // n c h w idx group N C H W
    for (auto i : self->GetShape()) {
        (void)i;
        tileParams.push_back(SymbolicScalar(0));
    }
    tileParams.push_back(0);
    tileParams.push_back(0);
    for (auto i : self->GetShape()) {
        tileParams.push_back(SymbolicScalar(i));
    }
    tileParams[6] = perGroupN1 * N0;
    tileParams[7] = C1 * C0;
    op.SetAttribute(OpAttributeKey::transDataOffset, tileParams);
    op.SetAttribute(OP_ATTR_PREFIX + "group", group);
    return result;
}

LogicalTensorPtr TransDataNCDHW2NDC1HWC0(Function& function, const LogicalTensorPtr& self, int group)
{
    int64_t N = self->GetShape()[0];
    int64_t C = self->GetShape()[1];
    int64_t D = self->GetShape()[2];
    int64_t H = self->GetShape()[3];
    int64_t W = self->GetShape()[4];
    int64_t C0 = BLOCK_SIZE / BytesOf(self->Datatype());
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, C0 > 0) << "The C0 is not valid !";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, group > 0) << "The group is not valid !";
    int64_t perGroupC = C / group;
    int64_t perGroupC1 = (perGroupC + C0 - 1) / C0;
    int64_t totalC1 = perGroupC1 * group;
    int64_t totalC = totalC1 * C0;
    Shape resultShape = {N, D, totalC1, H, W, C0};

    SymbolicScalar validShapeN = self->GetDynValidShape()[0];
    SymbolicScalar validShapeC = self->GetDynValidShape()[1];
    SymbolicScalar validShapeD = self->GetDynValidShape()[2];
    SymbolicScalar validShapeH = self->GetDynValidShape()[3];
    SymbolicScalar validShapeW = self->GetDynValidShape()[4];
    SymbolicScalar validShapePerGroupC = validShapeC / group;
    SymbolicScalar validShapePerGroupC1 = (validShapePerGroupC + C0 - 1) / C0;
    SymbolicScalar validShapePerTotalC1 = validShapePerGroupC1 * group;

    std::vector<SymbolicScalar> resultValidShape = {validShapeN, validShapeD, validShapePerTotalC1,
                                                    validShapeH, validShapeW, C0};
    auto result = std::make_shared<LogicalTensor>(function, self->Datatype(), resultShape, resultValidShape,
                                                  TileOpFormat::TILEOP_NDC1HWC0);
    auto tmpInput = Permute(function, Tensor(self), {0, 2, 1, 3, 4});

    VecTile oriVectile = TileShape::Current().GetVecTile();
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, oriVectile.tile[0] == 1)
        << "The tileShape N should be equal to 1, actual is " << oriVectile.tile[0];
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, oriVectile.tile[1] % C0 == 0)
        << "The tileShape C should be an integer multiple of C0!";
    VecTile tmpVectile = TileShape::Current().GetVecTile();
    std::swap(tmpVectile.tile[1], tmpVectile.tile[2]);
    TileShape::Current().SetVecTile(tmpVectile);

    auto& op = function.AddOperation(Opcode::OP_NCDHW2NDC1HWC0, {tmpInput.GetStorage()}, {result});
    std::vector<SymbolicScalar> tileParams = {};
    // n d c h w N D C H W
    for (auto i : tmpInput.GetShape()) {
        (void)i;
        tileParams.push_back(SymbolicScalar(0));
    }
    for (auto i : tmpInput.GetShape()) {
        tileParams.push_back(SymbolicScalar(i));
    }
    tileParams[7] = totalC;
    op.SetAttribute(OpAttributeKey::transDataOffset, tileParams);
    op.SetAttribute(OP_ATTR_PREFIX + "group", group);
    TileShape::Current().SetVecTile(oriVectile);
    return result;
}

LogicalTensorPtr TransDataFRACTAL_Z_3D(Function& function, const LogicalTensorPtr& self, int group)
{
    int64_t N = self->GetShape()[0];
    int64_t C = self->GetShape()[1];
    int64_t D = self->GetShape()[2];
    int64_t H = self->GetShape()[3];
    int64_t W = self->GetShape()[4];
    int64_t N0 = 16;
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, group > 0) << "The group is not valid !";
    int64_t perGroupN = N / group;
    int64_t perGroupN1 = (perGroupN + N0 - 1) / N0;
    int64_t C0 = BLOCK_SIZE / BytesOf(self->Datatype());
    int64_t C1 = (C + C0 - 1) / C0;
    Shape resultShape = {group * D * C1 * H * W, perGroupN1, N0, C0};

    SymbolicScalar validShapeN = self->GetDynValidShape()[0];
    SymbolicScalar validShapeC = self->GetDynValidShape()[1];
    SymbolicScalar validShapeD = self->GetDynValidShape()[2];
    SymbolicScalar validShapeH = self->GetDynValidShape()[3];
    SymbolicScalar validShapeW = self->GetDynValidShape()[4];
    auto validShapeC1 = (validShapeC + C0 - 1) / C0;
    auto validShapePerGroupN = validShapeN / group;
    auto validShapePerGroupN1 = (validShapePerGroupN + N0 - 1) / N0;
    std::vector<SymbolicScalar> resultValidShape = {group * validShapeD * validShapeC1 * validShapeH * validShapeW,
                                                    validShapePerGroupN1, N0, C0};

    VecTile oriVectile = TileShape::Current().GetVecTile();
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, oriVectile.tile[0] % N0 == 0)
        << "The tileShape N should be an integer multiple of N0(16)!";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, oriVectile.tile[1] == C0)
        << "The tileShape C should be equal to C0, actual is " << oriVectile.tile[1];
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, oriVectile.tile[3] == 1)
        << "The tileShape H should be equal to 1, actual is " << oriVectile.tile[3];

    auto result = std::make_shared<LogicalTensor>(function, self->Datatype(), resultShape, resultValidShape,
                                                  TileOpFormat::TILEOP_FRACTAL_Z_3D);

    auto& op = function.AddOperation(Opcode::OP_NCDHW2FRACTAL_Z_3D, {self}, {result});
    std::vector<SymbolicScalar> tileParams = {};
    // n c d h w idx group N C D H W
    for (auto i : self->GetShape()) {
        (void)i;
        tileParams.push_back(SymbolicScalar(0));
    }
    tileParams.push_back(0);
    tileParams.push_back(0);
    for (auto i : self->GetShape()) {
        tileParams.push_back(SymbolicScalar(i));
    }
    tileParams[7] = perGroupN1 * N0;
    tileParams[8] = C1 * C0;
    op.SetAttribute(OpAttributeKey::transDataOffset, tileParams);
    op.SetAttribute(OP_ATTR_PREFIX + "group", group);
    return result;
}

LogicalTensorPtr TransDataNDC1HWC02NCDHW(Function& function, const LogicalTensorPtr& self,
                                         const LogicalTensorPtr& output, int group)
{
    SymbolicScalar dstValidC = output->GetDynValidShape()[1];
    SymbolicScalar inputValidC1 = self->GetDynValidShape()[2];
    int64_t C0 = BLOCK_SIZE / BytesOf(self->Datatype());
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, group > 0) << "The group is not valid !";
    SymbolicScalar padSize = (inputValidC1 * C0 - dstValidC) / group;

    Shape resultShape = output->GetShape();
    std::swap(resultShape[1], resultShape[2]);
    auto resultValidShape = output->dynValidShape_;
    auto tmpValidShape = resultValidShape[1];
    resultValidShape[1] = resultValidShape[2];
    resultValidShape[2] = tmpValidShape;

    VecTile oriVectile = TileShape::Current().GetVecTile();
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, C0 > 0) << "The C0 is not valid !";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, oriVectile.tile[0] == 1)
        << "The tileShape N should be equal to 1, actual is " << oriVectile.tile[0];
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, oriVectile.tile[5] == C0) << "The tileShape c0 should be equal to C0!";

    auto result = std::make_shared<LogicalTensor>(function, self->Datatype(), resultShape, resultValidShape,
                                                  self->Format());
    auto& op = function.AddOperation(Opcode::OP_NDC1HWC02NCDHW, {self}, {result});
    std::vector<SymbolicScalar> tileParams = {};
    // n d c1 h w c0 idx group padSize N D dstC H W
    for (auto i : self->GetShape()) {
        (void)i;
        tileParams.push_back(SymbolicScalar(0));
    }
    tileParams.push_back(0);
    tileParams.push_back(0);
    tileParams.push_back(padSize);
    for (auto i : resultShape) {
        tileParams.push_back(SymbolicScalar(i));
    }
    op.SetAttribute(OpAttributeKey::transDataOffset, tileParams);
    op.SetAttribute(OP_ATTR_PREFIX + "group", group);
    VecTile tmpVectile = TileShape::Current().GetVecTile();
    tmpVectile.tile[1] *= tmpVectile.tile[5];
    tmpVectile.tile.pop_back();
    TileShape::Current().SetVecTile(tmpVectile);
    auto tmpResult = Permute(function, Tensor(result), {0, 2, 1, 3, 4});
    TileShape::Current().SetVecTile(oriVectile);
    return tmpResult.GetStorage();
}

LogicalTensorPtr TransDataNC1HWC02NCHW(Function& function, const LogicalTensorPtr& self, const LogicalTensorPtr& output,
                                       int group)
{
    SymbolicScalar dstValidC = output->GetDynValidShape()[1];
    SymbolicScalar inputValidC1 = self->GetDynValidShape()[1];
    int64_t C0 = BLOCK_SIZE / BytesOf(self->Datatype());
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, group > 0) << "The group is not valid !";
    SymbolicScalar padSize = (inputValidC1 * C0 - dstValidC) / group;

    VecTile oriVectile = TileShape::Current().GetVecTile();
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, C0 > 0) << "The C0 is not valid !";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, oriVectile.tile[4] == C0) << "The tileShape c0 should be equal to C0!";

    auto result = std::make_shared<LogicalTensor>(function, self->Datatype(), output->GetShape(),
                                                  output->dynValidShape_, TileOpFormat::TILEOP_ND);

    auto& op = function.AddOperation(Opcode::OP_NC1HWC02NCHW, {self}, {result});
    std::vector<SymbolicScalar> tileParams = {};
    // 0-4 input offset; 8-11 dst shape
    // n c1 h W c0 idx group padSize N dstC H W
    for (auto i : self->GetShape()) {
        (void)i;
        tileParams.push_back(SymbolicScalar(0));
    }
    tileParams.push_back(0);
    tileParams.push_back(0);
    tileParams.push_back(padSize);
    for (auto i : output->GetShape()) {
        tileParams.push_back(SymbolicScalar(i));
    }
    op.SetAttribute(OpAttributeKey::transDataOffset, tileParams);
    op.SetAttribute(OP_ATTR_PREFIX + "group", group);
    return result;
}

LogicalTensorPtr TransDataFractalZ2NCHW(Function& function, const LogicalTensorPtr& self,
                                        const LogicalTensorPtr& output, int group)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, group > 0) << "The group is not valid !";
    SymbolicScalar dstValidPerN = output->GetDynValidShape()[0] / group;
    SymbolicScalar dstValidC = output->GetDynValidShape()[1];
    int64_t N0 = 16;
    int64_t C0 = BLOCK_SIZE / BytesOf(self->Datatype());
    SymbolicScalar padNSize = (dstValidPerN + N0 - 1) / N0 * N0 - dstValidPerN;
    SymbolicScalar padCSize = (dstValidC + C0 - 1) / C0 * C0 - dstValidC;

    VecTile oriVectile = TileShape::Current().GetVecTile();
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, C0 > 0) << "The C0 is not valid !";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, oriVectile.tile[2] == N0) << "The tileShape n0 should be equal to N0!";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, oriVectile.tile[3] == C0) << "The tileShape c0 should be equal to C0!";

    auto result = std::make_shared<LogicalTensor>(function, self->Datatype(), output->GetShape(),
                                                  output->dynValidShape_, TileOpFormat::TILEOP_ND);

    auto& op = function.AddOperation(Opcode::OP_FractalZ2NCHW, {self}, {result});
    std::vector<SymbolicScalar> tileParams = {};
    // 0-3 input offset;
    // c1hw n1 n0 c0 idx group padNSize padCSize N C H W
    for (auto i : self->GetShape()) {
        (void)i;
        tileParams.push_back(SymbolicScalar(0));
    }
    tileParams.push_back(0);
    tileParams.push_back(0);
    tileParams.push_back(padNSize);
    tileParams.push_back(padCSize);
    for (auto i : output->GetShape()) {
        tileParams.push_back(SymbolicScalar(i));
    }
    op.SetAttribute(OpAttributeKey::transDataOffset, tileParams);
    op.SetAttribute(OP_ATTR_PREFIX + "group", group);
    return result;
}

LogicalTensorPtr TransDataFractalZ3D2NCDHW(Function& function, const LogicalTensorPtr& self,
                                           const LogicalTensorPtr& output, int group)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, group > 0) << "The group is not valid !";
    SymbolicScalar dstValidPerN = output->GetDynValidShape()[0] / group;
    SymbolicScalar dstValidC = output->GetDynValidShape()[1];
    int64_t N0 = 16;
    int64_t C0 = BLOCK_SIZE / BytesOf(self->Datatype());
    SymbolicScalar padNSize = (dstValidPerN + N0 - 1) / N0 * N0 - dstValidPerN;
    SymbolicScalar padCSize = (dstValidC + C0 - 1) / C0 * C0 - dstValidC;

    VecTile oriVectile = TileShape::Current().GetVecTile();
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, C0 > 0) << "The C0 is not valid !";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, oriVectile.tile[2] == N0) << "The tileShape n0 should be equal to N0!";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, oriVectile.tile[3] == C0) << "The tileShape c0 should be equal to C0!";

    auto result = std::make_shared<LogicalTensor>(function, self->Datatype(), output->GetShape(),
                                                  output->dynValidShape_, TileOpFormat::TILEOP_ND);

    auto& op = function.AddOperation(Opcode::OP_FractalZ3D2NCDHW, {self}, {result});
    std::vector<SymbolicScalar> tileParams = {};
    // 0-3 input offset (DC1HW, N1, N0, C0)
    // 4: groupIdx, 5: group, 6: padNSize, 7: padCSize, 8-12: N C D H W
    for (auto i : self->GetShape()) {
        (void)i;
        tileParams.push_back(SymbolicScalar(0));
    }
    tileParams.push_back(0);
    tileParams.push_back(0);
    tileParams.push_back(padNSize);
    tileParams.push_back(padCSize);
    for (auto i : output->GetShape()) {
        tileParams.push_back(SymbolicScalar(i));
    }
    op.SetAttribute(OpAttributeKey::transDataOffset, tileParams);
    op.SetAttribute(OP_ATTR_PREFIX + "group", group);
    return result;
}

LogicalTensorPtr TransDataReverse(Function& function, const LogicalTensorPtr& self, const LogicalTensorPtr& output,
                                  int group)
{
    int inputShapeSize = self->GetShape().size();
    int outputShapeSize = output->GetShape().size();
    if (inputShapeSize == SHAPE_DIM5 && outputShapeSize == SHAPE_DIM4) {
        return TransDataNC1HWC02NCHW(function, self, output, group);
    } else if (inputShapeSize == SHAPE_DIM6 && outputShapeSize == SHAPE_DIM5) {
        return TransDataNDC1HWC02NCDHW(function, self, output, group);
    } else if (inputShapeSize == SHAPE_DIM4 && outputShapeSize == SHAPE_DIM4) {
        return TransDataFractalZ2NCHW(function, self, output, group);
    } else if (inputShapeSize == SHAPE_DIM4 && outputShapeSize == SHAPE_DIM5) {
        return TransDataFractalZ3D2NCDHW(function, self, output, group);
    } else {
        return TransDataFractalZ2NCHW(function, self, output, group);
    }
}

LogicalTensorPtr TensorTransData(Function& function, const LogicalTensorPtr& self, const LogicalTensorPtr& output,
                                 TileOpFormat transDataType, int group)
{
    switch (transDataType) {
        case TileOpFormat::TILEOP_NC1HWC0:
            return TransDataNCHW2NC1HWC0(function, self, group);
        case TileOpFormat::TILEOP_FRACTAL_Z:
            return TransDataNCHW2Fractal_Z(function, self, group);
        case TileOpFormat::TILEOP_NDC1HWC0:
            return TransDataNCDHW2NDC1HWC0(function, self, group);
        case TileOpFormat::TILEOP_FRACTAL_Z_3D:
            return TransDataFRACTAL_Z_3D(function, self, group);
        case TileOpFormat::TILEOP_ND:
            return TransDataReverse(function, self, output, group); // 反向
        default:
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, false) << "The transDataType is not supported";
    }
    return self;
}

LogicalTensorPtr TransData(Function& function, const LogicalTensorPtr& self, const LogicalTensorPtr& output,
                           TileOpFormat transDataType, int group)
{
    DECLARE_TRACER();
    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_INT16, DT_FP32, DT_INT32};
    CheckTensorDataType(self, supportedTypes, "TRANSDATA");
    CheckTensorShapeSize(self, "TRANSDATA");
    const auto& shape = self->GetShape();
    VecTile oriVectile = TileShape::Current().GetVecTile();
    const int64_t MAX_TILE_NUM = 5000;
    int64_t tileNum = 1;
    for (size_t i = 0; i < shape.size() && i < oriVectile.tile.size() && tileNum < MAX_TILE_NUM; i++) {
        if (oriVectile.tile[i] <= 0) {
            tileNum = MAX_TILE_NUM;
            break;
        }
        tileNum *= (shape[i] + oriVectile.tile[i] - 1) / oriVectile.tile[i];
    }
    if (tileNum >= MAX_TILE_NUM) {
        VECTOR_LOGW("TransData tileNum=%ld exceeds limit, shape may be too large. Adjust `view_shape` or `tile_shape`.",
                    tileNum);
    }
    switch (transDataType) {
        case TileOpFormat::TILEOP_NC1HWC0:
            CheckTensorDimRange(self, SHAPE_DIM4, SHAPE_DIM4, "TRANSDATA NC1HWC0");
            break;
        case TileOpFormat::TILEOP_FRACTAL_Z:
            CheckTensorDimRange(self, SHAPE_DIM4, SHAPE_DIM4, "TRANSDATA FRACTAL_Z");
            break;
        case TileOpFormat::TILEOP_NDC1HWC0:
            CheckTensorDimRange(self, SHAPE_DIM5, SHAPE_DIM5, "TRANSDATA NDC1HWC0");
            break;
        case TileOpFormat::TILEOP_FRACTAL_Z_3D:
            CheckTensorDimRange(self, SHAPE_DIM5, SHAPE_DIM5, "TRANSDATA FRACTAL_Z_3D");
            break;
        case TileOpFormat::TILEOP_ND:
            CheckTensorDimRange(self, SHAPE_DIM4, SHAPE_DIM6, "TRANSDATA ND");
            break;
        default:
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, false) << "The transDataType is not supported";
    }
    return TensorTransData(function, self, output, transDataType, group);
}

Tensor TransData(const Tensor& self, TileOpFormat transDataType, const std::vector<int64_t>& outputShape,
                 const std::vector<SymbolicScalar>& validShape, int group)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "TransData");

    auto& function = *Program::GetInstance().GetCurrentFunction();
    auto output = std::make_shared<LogicalTensor>(function, self.GetDataType(), outputShape, validShape, self.Format());
    auto tmpTensor = TransData(function, self.GetStorage(), output, transDataType, group);
    tmpTensor->tensor->format = TileOpFormat::TILEOP_ND;
    return Tensor(tmpTensor);
}

void TransDataTileFunc(Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
                       const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    std::vector<SymbolicScalar> tileParams = op.GetVectorSymbolicScalarAttribute(OpAttributeKey::transDataOffset);
    int group = op.GetIntAttribute(OP_ATTR_PREFIX + "group");
    TransDataPara transDataPara = TransDataPara{iOperand[0], oOperand[0], tileParams, group, 0, 0, 0, 0, 0};
    switch (op.GetOpcode()) {
        case Opcode::OP_NCHW2NC1HWC0:
            TiledTransData<TileOpFormat::TILEOP_NC1HWC0>(function, tileShape, transDataPara);
            break;
        case Opcode::OP_NCHW2Fractal_Z:
            TiledTransData<TileOpFormat::TILEOP_FRACTAL_Z>(function, tileShape, transDataPara);
            break;
        case Opcode::OP_NC1HWC02NCHW:
            TiledTransData<TileOpFormat::TILEOP_ND>(function, tileShape, transDataPara);
            break;
        case Opcode::OP_NCDHW2NDC1HWC0:
            TiledTransData<TileOpFormat::TILEOP_NDC1HWC0>(function, tileShape, transDataPara);
            break;
        case Opcode::OP_NCDHW2FRACTAL_Z_3D:
            TiledTransData<TileOpFormat::TILEOP_FRACTAL_Z_3D>(function, tileShape, transDataPara);
            break;
        case Opcode::OP_NDC1HWC02NCDHW:
            TiledTransData<TileOpFormat::TILEOP_ND>(function, tileShape, transDataPara);
            break;
        case Opcode::OP_FractalZ2NCHW:
            TiledTransData<TileOpFormat::TILEOP_ND>(function, tileShape, transDataPara);
            break;
        case Opcode::OP_FractalZ3D2NCDHW:
            TiledTransData<TileOpFormat::TILEOP_ND>(function, tileShape, transDataPara);
            break;
        default:
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, false) << "The transDataType is not supported";
    }
}

REGISTER_OPERATION_TILED_FUNC(OP_NCHW2NC1HWC0, Opcode::OP_NCHW2NC1HWC0, TransDataTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_NCHW2Fractal_Z, Opcode::OP_NCHW2Fractal_Z, TransDataTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_NC1HWC02NCHW, Opcode::OP_NC1HWC02NCHW, TransDataTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_NCDHW2NDC1HWC0, Opcode::OP_NCDHW2NDC1HWC0, TransDataTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_NCDHW2FRACTAL_Z_3D, Opcode::OP_NCDHW2FRACTAL_Z_3D, TransDataTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_NDC1HWC02NCDHW, Opcode::OP_NDC1HWC02NCDHW, TransDataTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_FractalZ2NCHW, Opcode::OP_FractalZ2NCHW, TransDataTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_FractalZ3D2NCDHW, Opcode::OP_FractalZ3D2NCDHW, TransDataTileFunc);

} // namespace npu::tile_fwk
