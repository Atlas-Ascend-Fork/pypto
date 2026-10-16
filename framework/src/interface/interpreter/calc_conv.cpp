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
 * \file calc_conv.cpp
 * \brief Conv family calc implementations for the precision tool (pass_verify).
 */

#include <set>

#include "interface/interpreter/function.h"
#include "interface/interpreter/operation.h"
#include "interface/operation/operation.h"
#include "interface/operation/operation_impl.h"
#include "interface/operation/opcode.h"
#include "interface/utils/common.h"
#include "tilefwk/error_code.h"

using namespace npu::tile_fwk::calc;

namespace npu {
namespace tile_fwk {
namespace {

constexpr int64_t FMT_ND = static_cast<int64_t>(TileOpFormat::TILEOP_ND);
constexpr int64_t FMT_NC1HWC0 = static_cast<int64_t>(TileOpFormat::TILEOP_NC1HWC0);
constexpr int64_t FMT_NDC1HWC0 = static_cast<int64_t>(TileOpFormat::TILEOP_NDC1HWC0);
constexpr int64_t FMT_FRACTAL_Z = static_cast<int64_t>(TileOpFormat::TILEOP_FRACTAL_Z);
constexpr int64_t FMT_FRACTAL_Z_3D = static_cast<int64_t>(TileOpFormat::TILEOP_FRACTAL_Z_3D);

bool GetBoolAttrOr(const Operation* op, const std::string& key, bool defaultValue = false)
{
    return op->HasAttr(key) ? op->GetBoolAttribute(key) : defaultValue;
}

int64_t GetIntAttrOr(const Operation* op, const std::string& key, int64_t defaultValue = 0)
{
    return op->HasAttr(key) ? op->GetIntAttribute(key) : defaultValue;
}

Operation* FindNeighborConvOp(const Operation* transOp, bool searchConsumers)
{
    if (transOp == nullptr) {
        return nullptr;
    }
    std::set<Operation*> visited;
    if (searchConsumers) {
        for (const auto& oop : transOp->GetOOperands()) {
            if (oop == nullptr) {
                continue;
            }
            for (auto* consumer : oop->GetConsumers()) {
                if (consumer != nullptr && !consumer->IsDeleted() && visited.insert(consumer).second &&
                    (consumer->GetOpcode() == Opcode::OP_CONV2D || consumer->GetOpcode() == Opcode::OP_CONV3D)) {
                    return consumer;
                }
            }
        }
    } else {
        for (const auto& iop : transOp->GetIOperands()) {
            if (iop == nullptr) {
                continue;
            }
            for (auto* producer : iop->GetProducers()) {
                if (producer != nullptr && !producer->IsDeleted() && visited.insert(producer).second &&
                    (producer->GetOpcode() == Opcode::OP_CONV2D || producer->GetOpcode() == Opcode::OP_CONV3D)) {
                    return producer;
                }
            }
        }
    }
    return nullptr;
}

int64_t GetConvGroupsAround(const Operation* transOp)
{
    Operation* convOp = FindNeighborConvOp(transOp, true);
    if (convOp == nullptr) {
        convOp = FindNeighborConvOp(transOp, false);
    }
    if (convOp != nullptr && convOp->HasAttr(CONV_GROUPS_ATTR)) {
        return convOp->GetIntAttribute(CONV_GROUPS_ATTR);
    }
    return 1;
}

void ExecuteOpConv(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_NULL, ctx != nullptr);
    ASSERT(ExecuteOperationScene::CTX_OP_NULL, ctx->op != nullptr);
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH,
           ctx->ioperandDataViewList->size() == 2 || ctx->ioperandDataViewList->size() == 3);
    auto ret = ctx->ooperandInplaceDataViewList->at(0);
    auto fmap = ctx->ioperandDataViewList->at(0);
    auto weight = ctx->ioperandDataViewList->at(1);
    auto bias = ctx->ioperandDataViewList->size() > 2 ? ctx->ioperandDataViewList->at(2) : nullptr;
    Operation* op = ctx->op;
    bool is3d = GetBoolAttrOr(op, Conv::CONV_3D_FLAG);
    ConvParam param;
    param.isConv3D = is3d ? 1 : 0;
    param.groups = GetIntAttrOr(op, CONV_GROUPS_ATTR, 1);
    param.reluType = GetIntAttrOr(op, Conv::CONV_RELU_ATTR, 0);
    param.strides = op->HasAttr(Conv::CONV_STRIDES_ATTR) ?
                        op->GetVectorIntAttribute(Conv::CONV_STRIDES_ATTR) :
                        (is3d ? std::vector<int64_t>{1, 1, 1} : std::vector<int64_t>{1, 1});
    param.dilations = op->HasAttr(Conv::CONV_DILATIONS_ATTR) ?
                          op->GetVectorIntAttribute(Conv::CONV_DILATIONS_ATTR) :
                          (is3d ? std::vector<int64_t>{1, 1, 1} : std::vector<int64_t>{1, 1});
    param.paddings = op->HasAttr(Conv::CONV_PADDINGS_ATTR) ?
                         op->GetVectorIntAttribute(Conv::CONV_PADDINGS_ATTR) :
                         (is3d ? Conv::CONV3D_PAD_ATTR_DEFAULT_LIST : Conv::CONV2D_PAD_ATTR_DEFAULT_LIST);
    if (op->HasAttr(Conv::CONV_ORI_WEIGHT_SHAPE_ATTR)) {
        auto weightShape = op->GetVectorIntAttribute(Conv::CONV_ORI_WEIGHT_SHAPE_ATTR);
        if (weightShape.size() == 3) {
            param.kernelSize = {1, weightShape[2]};
        } else if (weightShape.size() == 4) {
            param.kernelSize = {weightShape[2], weightShape[3]};
        } else if (weightShape.size() == 5) {
            param.kernelSize = {weightShape[2], weightShape[3], weightShape[4]};
        }
    }
    param.fmapFormat = static_cast<int64_t>(op->GetIOperands()[0]->Format());
    param.weightFormat = static_cast<int64_t>(op->GetIOperands()[1]->Format());
    param.outFormat = static_cast<int64_t>(op->GetOOperands()[0]->Format());
    calc::Conv(ret, fmap, weight, bias, param);
}

REGISTER_CALC_OP(OP_CONV2D, Opcode::OP_CONV2D, ExecuteOpConv);
REGISTER_CALC_OP(OP_CONV3D, Opcode::OP_CONV3D, ExecuteOpConv);

void ExecuteOpFakeTrans(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto ret = ctx->ooperandInplaceDataViewList->at(0);
    auto self = ctx->ioperandDataViewList->at(0);
    Operation* op = ctx->op;
    int64_t srcFmt = op->GetIntAttribute(FAKE_TRANS_IN_FORMAT_ATTR);
    int64_t dstFmt = op->GetIntAttribute(FAKE_TRANS_OUT_FORMAT_ATTR);
    int64_t group = GetConvGroupsAround(op);
    calc::FormatTransConv(ret, self, srcFmt, dstFmt, group);
}

REGISTER_CALC_OP(OP_FAKE_TRANS, Opcode::OP_FAKE_TRANS, ExecuteOpFakeTrans);

void ExecuteOpTransData(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH,
           ctx->ooperandInplaceDataViewList->size() == 1 || ctx->ooperandInplaceDataViewList->size() == 0x2);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto ret = ctx->ooperandInplaceDataViewList->at(0);
    auto self = ctx->ioperandDataViewList->at(0);
    Operation* op = ctx->op;
    Opcode opcode = op->GetOpcode();
    int64_t srcFmt = FMT_ND;
    int64_t dstFmt = FMT_ND;
    bool inputIsNdchw = false;
    bool outputIsNdchw = false;
    switch (opcode) {
        case Opcode::OP_NCHW2NC1HWC0:
            dstFmt = FMT_NC1HWC0;
            break;
        case Opcode::OP_NCHW2Fractal_Z:
            dstFmt = FMT_FRACTAL_Z;
            break;
        case Opcode::OP_NC1HWC02NCHW:
            outputIsNdchw = true;
            srcFmt = FMT_NC1HWC0;
            break;
        case Opcode::OP_FractalZ2NCHW:
            srcFmt = FMT_FRACTAL_Z;
            break;
        case Opcode::OP_NCDHW2NDC1HWC0:
            dstFmt = FMT_NDC1HWC0;
            inputIsNdchw = true;
            break;
        case Opcode::OP_NCDHW2FRACTAL_Z_3D:
            dstFmt = FMT_FRACTAL_Z_3D;
            break;
        case Opcode::OP_NDC1HWC02NCDHW:
            srcFmt = FMT_NDC1HWC0;
            outputIsNdchw = true;
            break;
        case Opcode::OP_FractalZ3D2NCDHW:
            srcFmt = FMT_FRACTAL_Z_3D;
            break;
        default:
            break;
    }
    int64_t group = GetIntAttrOr(op, OP_ATTR_PREFIX + "group", 1);
    if (inputIsNdchw) {
        auto ndchwShape = self->GetShape();
        std::swap(ndchwShape[1], ndchwShape[2]);
        auto ncdhw = LogicalTensorData::CreateEmpty(self->GetDataType(), ndchwShape, std::vector<int64_t>{},
                                                    ndchwShape);
        calc::Permute(ncdhw, self, {0, 2, 1, 3, 4});
        calc::FormatTransConv(ret, ncdhw, srcFmt, dstFmt, group);
    } else if (outputIsNdchw) {
        if (ret->GetShape().size() == 4) {
            calc::FormatTransConv(ret, self, srcFmt, FMT_ND, group);
        } else {
            auto ndchwShape = ret->GetShape();
            auto ncdhwShape = ndchwShape;
            std::swap(ncdhwShape[1], ncdhwShape[2]);
            auto ncdhw = LogicalTensorData::CreateEmpty(ret->GetDataType(), ncdhwShape, std::vector<int64_t>{},
                                                        ncdhwShape);
            calc::FormatTransConv(ncdhw, self, srcFmt, FMT_ND, group);
            calc::Permute(ret, ncdhw, {0, 2, 1, 3, 4});
        }
    } else {
        calc::FormatTransConv(ret, self, srcFmt, dstFmt, group);
    }
}

REGISTER_CALC_OP(OP_NCHW2NC1HWC0, Opcode::OP_NCHW2NC1HWC0, ExecuteOpTransData);
REGISTER_CALC_OP(OP_NC1HWC02NCHW, Opcode::OP_NC1HWC02NCHW, ExecuteOpTransData);
REGISTER_CALC_OP(OP_NCHW2Fractal_Z, Opcode::OP_NCHW2Fractal_Z, ExecuteOpTransData);
REGISTER_CALC_OP(OP_FractalZ2NCHW, Opcode::OP_FractalZ2NCHW, ExecuteOpTransData);
REGISTER_CALC_OP(OP_NCDHW2NDC1HWC0, Opcode::OP_NCDHW2NDC1HWC0, ExecuteOpTransData);
REGISTER_CALC_OP(OP_NDC1HWC02NCDHW, Opcode::OP_NDC1HWC02NCDHW, ExecuteOpTransData);
REGISTER_CALC_OP(OP_NCDHW2FRACTAL_Z_3D, Opcode::OP_NCDHW2FRACTAL_Z_3D, ExecuteOpTransData);
REGISTER_CALC_OP(OP_FractalZ3D2NCDHW, Opcode::OP_FractalZ3D2NCDHW, ExecuteOpTransData);

void ExecuteOpTransFormatL1(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto ret = ctx->ooperandInplaceDataViewList->at(0);
    auto self = ctx->ioperandDataViewList->at(0);
    Operation* op = ctx->op;
    bool isFmap = GetBoolAttrOr(op, Conv::LoadStoreConvOpAttributeKey::isFmap);
    int64_t is3d = GetBoolAttrOr(op, Conv::LoadStoreConvOpAttributeKey::isConv3D) ? 1 : 0;
    if (isFmap) {
        calc::ConvFmapND2NZ(ret, self, is3d);
    } else {
        calc::ConvWeightND2FZ(ret, self, is3d);
    }
}

REGISTER_CALC_OP(OP_TRANS_FORMAT_L1, Opcode::OP_TRANS_FORMAT_L1, ExecuteOpTransFormatL1);

void ExecuteOpLoad3DConv(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto ret = ctx->ooperandInplaceDataViewList->at(0);
    auto l1 = ctx->ioperandDataViewList->at(0);
    Operation* op = ctx->op;
    ConvTileParam param;
    param.isConv3D = GetBoolAttrOr(op, Conv::LoadStoreConvOpAttributeKey::isConv3D) ? 1 : 0;
    param.strideH = GetIntAttrOr(op, OpAttributeKey::strideH, 1);
    param.strideW = GetIntAttrOr(op, OpAttributeKey::strideW, 1);
    param.dilationH = GetIntAttrOr(op, OpAttributeKey::dilationH, 1);
    param.dilationW = GetIntAttrOr(op, OpAttributeKey::dilationW, 1);
    param.filterH = GetIntAttrOr(op, OpAttributeKey::filterH, 1);
    param.filterW = GetIntAttrOr(op, OpAttributeKey::filterW, 1);
    param.padValue = GetIntAttrOr(op, OpAttributeKey::padValue, 0);
    param.validM = ret->GetValidShape().empty() ? ret->GetShape()[0] : ret->GetValidShape()[0];
    param.l0CutW = GetIntAttrOr(op, "CONV_L0_CUT_W", 0);
    param.l0HOffset = GetIntAttrOr(op, "CONV_L0_H_OFFSET", 0);
    param.l0WOffset = GetIntAttrOr(op, "CONV_L0_W_OFFSET", 0);
    param.srcHDelta = GetIntAttrOr(op, "CONV_SRC_H_DELTA", 0);
    param.srcWDelta = GetIntAttrOr(op, "CONV_SRC_W_DELTA", 0);
    param.kStartPt = GetIntAttrOr(op, OpAttributeKey::postK, 0);
    param.padTop = GetIntAttrOr(op, OpAttributeKey::paddingTop, 0);
    param.padLeft = GetIntAttrOr(op, OpAttributeKey::paddingLeft, 0);
    calc::ConvLoad3D(ret, l1, param);
}

REGISTER_CALC_OP(OP_LOAD3D_CONV, Opcode::OP_LOAD3D_CONV, ExecuteOpLoad3DConv);

void ExecuteOpLoad2DConv(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto ret = ctx->ooperandInplaceDataViewList->at(0);
    auto l1 = ctx->ioperandDataViewList->at(0);
    Operation* op = ctx->op;
    int64_t postK = GetIntAttrOr(op, OpAttributeKey::postK, 0);
    int64_t postN = GetIntAttrOr(op, OpAttributeKey::postN, 0);
    int64_t is3d = GetBoolAttrOr(op, Conv::LoadStoreConvOpAttributeKey::isConv3D) ? 1 : 0;
    calc::ConvLoad2D(ret, l1, postK, postN, is3d);
}

REGISTER_CALC_OP(OP_LOAD2D_CONV, Opcode::OP_LOAD2D_CONV, ExecuteOpLoad2DConv);

void ExecuteOpTransFormatL0C(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto ret = ctx->ooperandInplaceDataViewList->at(0);
    auto l0c = ctx->ioperandDataViewList->at(0);
    Operation* op = ctx->op;
    ConvL0CParam param;
    param.isConv3D = GetBoolAttrOr(op, Conv::LoadStoreConvOpAttributeKey::isConv3D) ? 1 : 0;
    param.copyOutMode = GetIntAttrOr(op, Conv::LoadStoreConvOpAttributeKey::copyOutMode, 0);
    param.cutW = GetIntAttrOr(op, Conv::LoadStoreConvOpAttributeKey::cutW, 0);
    param.reluType = GetIntAttrOr(op, Conv::LoadStoreConvOpAttributeKey::reluType, 0);
    const auto& validShape = ret->GetValidShape();
    if (param.copyOutMode == static_cast<int64_t>(Conv::CopyOutMode::COPY_MOD_NZ2DN)) {
        param.validN = validShape.size() >= 2 ? validShape[1] : 1;
        param.validD = (param.isConv3D != 0 && validShape.size() >= 3) ? validShape[2] : 1;
        param.validH = validShape.size() >= 2 ? validShape[validShape.size() - 2] : 1;
        param.validW = validShape.size() >= 1 ? validShape[validShape.size() - 1] : 1;
        // Merged graph: the op writes the res tensor through a CopyOpAttribute "toOffset" view.
        // The dst region is one L0 copy-out tile: [1(batch), validN, (1(d),) m/w, w]; deriving it from the
        // outcast shape (ret) breaks dynamic batch (batch dim not narrowed per tile) and can overflow storage.
        auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
        if (copyAttr != nullptr) {
            auto toOffsetAttr = copyAttr->GetCopyOutAttr().second;
            auto toOffset = ctx->opInter->EvaluateOpImmediate(ctx->frame, toOffsetAttr);
            const auto& l0cShape = l0c->GetShape();
            int64_t m = l0cShape[0];
            int64_t n = l0cShape[1];
            std::vector<SymbolicScalar> l0cValidMN;
            if (op->GetAttr(OpAttributeKey::l0cValidMN, l0cValidMN) && l0cValidMN.size() == 0x2) {
                // per-tile valid M/N kept as an op attribute: it survives InferParamIndex clearing tensor
                // valid shapes, and bounds group-tail tiles to their own channels (an unclamped region
                // would let group 0's copy-out clobber the next group's channels in the outcast)
                m = ctx->opInter->EvaluateSymbolicScalar(l0cValidMN[0]);
                n = ctx->opInter->EvaluateSymbolicScalar(l0cValidMN[1]);
            }
            int64_t w = param.cutW > 0 ? param.cutW : l0cShape[l0cShape.size() - 1];
            std::vector<int64_t> regionShape;
            if (param.isConv3D != 0) {
                regionShape = {1, n, 1, m / w, w};
            } else {
                regionShape = {1, n, m / w, w};
            }
            // clamp each region dim by the remaining GM extent: InferParamIndex may clear the L0C valid
            // shape (aligned N), and writing past the outcast bounds would overflow storage
            const auto& rawShape = ret->GetData()->GetShape();
            for (size_t i = 0; i < regionShape.size() && i < rawShape.size(); i++) {
                regionShape[i] = std::max<int64_t>(std::min(regionShape[i], rawShape[i] - toOffset[i]), 0);
            }
            ret = std::make_shared<LogicalTensorData>(ret->GetData(), regionShape, toOffset);
        }
    } else {
        param.validD = (param.isConv3D != 0 && validShape.size() >= 4) ? validShape[validShape.size() - 4] : 1;
        param.validH = validShape.size() >= 3 ? validShape[validShape.size() - 3] : 1;
        param.validW = validShape.size() >= 2 ? validShape[validShape.size() - 2] : 1;
    }
    calc::ConvTransL0C(ret, l0c, param);
}

REGISTER_CALC_OP(OP_TRANS_FORMAT_L0C, Opcode::OP_TRANS_FORMAT_L0C, ExecuteOpTransFormatL0C);
REGISTER_CALC_OP(OP_L0C_COPY_OUT_CONV, Opcode::OP_L0C_COPY_OUT_CONV, ExecuteOpTransFormatL0C);

void ExecuteOpL1CopyInConv(ExecuteOperationContext* ctx)
{
    ASSERT(ExecuteOperationScene::CTX_OUTPUT_COUNT_MISMATCH, ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ExecuteOperationScene::CTX_INPUT_COUNT_MISMATCH, ctx->ioperandDataViewList->size() == 1);
    auto ret = ctx->ooperandInplaceDataViewList->at(0);
    auto gm = ctx->ioperandDataViewList->at(0);
    Operation* op = ctx->op;
    auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
    ASSERT(ExecuteOperationScene::CTX_OP_NULL, copyAttr != nullptr);
    std::vector<int64_t> fromOffset = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyAttr->GetFromOffset());
    auto isFmap = GetBoolAttrOr(op, Conv::LoadStoreConvOpAttributeKey::isFmap);
    int64_t is3d = GetBoolAttrOr(op, Conv::LoadStoreConvOpAttributeKey::isConv3D) ? 1 : 0;
    int64_t copyInMode = GetIntAttrOr(op, Conv::LoadStoreConvOpAttributeKey::copyInMode,
                                      static_cast<int64_t>(Conv::CopyInMode::COPY_MOD_DN2NZ));
    if (copyInMode == static_cast<int64_t>(Conv::CopyInMode::COPY_MOD_NZ2NZ) ||
        copyInMode == static_cast<int64_t>(Conv::CopyInMode::COPY_MOD_ND2ND)) {
        std::vector<int64_t> validShape = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyAttr->GetToDynValidShape());
        if (validShape.empty()) {
            validShape = ret->GetShape();
        }
        calc::Copy(ret, gm->View(validShape, fromOffset));
        return;
    }
    auto os = ret->GetShape();
    std::vector<int64_t> srcShape;
    if (isFmap) {
        auto gs = gm->GetShape();
        if (is3d != 0) {
            // dst L1 NDC1HWC0 [n, d, c1, h, w, c0] -> src GM NCDHW [n, cin, dk, h, w]
            int64_t alignedCin = os[0x2] * os[os.size() - 1];
            // N/D 维按 GM 剩余范围裁剪, 尾块越界平面 (offset 超出 tensor) 不读 GM
            int64_t validN = std::max(int64_t(0), std::min(gs[0] - fromOffset[0], os[0]));
            int64_t validD = std::max(int64_t(0), std::min(gs[2] - fromOffset[2], os[1]));
            srcShape = {validN, std::min(gs[1] - fromOffset[1], alignedCin), validD, os[3], os[4]};
        } else {
            // dst L1 NC1HWC0 [n, c1, h, w, c0] -> src GM NCHW [n, cin, h, w]
            int64_t alignedCin = os[1] * os[os.size() - 1];
            int64_t validN = std::max(int64_t(0), std::min(gs[0] - fromOffset[0], os[0]));
            srcShape = {validN, std::min(gs[1] - fromOffset[1], alignedCin), os[0x2], os[3]};
        }
    } else {
        // Weight ND view: [n, cin, (dk,) kh, kw], derived from the L1 NZ tile [k/cin0, nBlocks, n0, cin0] and
        // clamped by the remaining GM extent, mirroring InferSrcGmValidShapeForWeight.
        auto gs = gm->GetShape();
        int64_t filterH = GetIntAttrOr(op, OpAttributeKey::filterH, 1);
        int64_t filterW = GetIntAttrOr(op, OpAttributeKey::filterW, 1);
        int64_t dkL1Size = 1;
        if (is3d != 0) {
            dkL1Size = GetIntAttrOr(op, "CONV_DK_L1_SIZE", 1);
        }
        int64_t dkxkhxkw = dkL1Size * filterH * filterW;
        if (dkxkhxkw == 0) {
            dkxkhxkw = 1;
        }
        int64_t cin1 = os[0] / dkxkhxkw;
        int64_t alignedCin = cin1 * os[os.size() - 1];
        srcShape.push_back(std::min(gs[0] - fromOffset[0], os[1] * os[0x2]));
        srcShape.push_back(std::min(gs[1] - fromOffset[1], alignedCin));
        if (is3d != 0) {
            srcShape.push_back(dkL1Size);
        }
        srcShape.push_back(filterH);
        srcShape.push_back(filterW);
    }
    auto srcTile = gm->View(srcShape, fromOffset);
    if (isFmap) {
        calc::ConvFmapND2NZ(ret, srcTile, is3d);
    } else {
        calc::ConvWeightND2FZ(ret, srcTile, is3d);
    }
}

REGISTER_CALC_OP(OP_L1_COPY_IN_CONV, Opcode::OP_L1_COPY_IN_CONV, ExecuteOpL1CopyInConv);
} // namespace
} // namespace tile_fwk
} // namespace npu
