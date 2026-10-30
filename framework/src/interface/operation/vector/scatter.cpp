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
 * \\file scatter.cpp
 * \\brief
 */

#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/operation/operation_common.h"
#include "interface/operation/vector/gather_mask_common.h"
#include "tensor_transformation.h"
#include "tilefwk/error_code.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {

struct ScatterElementSPara {
    const LogicalTensorPtr& dstTensor;
    const LogicalTensorPtr& srcInput;
    const LogicalTensorPtr& idxInput;
    const Element& scalar;
    const int axis;
    const int scatterMode;
    const bool inplace = false;
};

struct ScatterElementSTileInfoPara {
    TileInfo srcTileInfo;
    TileInfo idxTileInfo;
    TileInfo dstTileInfo;
};

void InnerTiledScatterElementS(size_t cur, Function& function, const TileShape& tileShape,
                               const ScatterElementSPara& scatterPara, ScatterElementSTileInfoPara& scatterTileInfo)
{
    const LogicalTensorPtr& dstTensor = scatterPara.dstTensor;
    const LogicalTensorPtr& srcInput = scatterPara.srcInput;
    const LogicalTensorPtr& idxInput = scatterPara.idxInput;
    const Element& scalar = scatterPara.scalar;
    const int axis = scatterPara.axis;
    const int mode = scatterPara.scatterMode;
    const bool dynamicGm = scatterPara.inplace && std::any_of(srcInput->shape.begin(), srcInput->shape.end(),
                                                              [](int64_t dim) { return dim < 0; });

    if (cur == dstTensor->shape.size()) {
        auto srcTile = dynamicGm ? srcInput :
                                   srcInput->View(function, scatterTileInfo.srcTileInfo.shape,
                                                  scatterTileInfo.srcTileInfo.offset);
        auto idxTile = idxInput->View(function, scatterTileInfo.idxTileInfo.shape, scatterTileInfo.idxTileInfo.offset);
        auto dstTile = scatterPara.inplace && srcInput->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR ?
                           dstTensor :
                           dstTensor->View(function, scatterTileInfo.dstTileInfo.shape,
                                           scatterTileInfo.dstTileInfo.offset);
        auto opcode = scatterPara.inplace ? Opcode::OP_SCATTER_ELEMENT_INPLACE : Opcode::OP_SCATTER_ELEMENT;
        auto& op = function.AddOperation(opcode, {srcTile, idxTile}, {dstTile});
        if (dynamicGm) {
            op.SetAttribute(OP_ATTR_PREFIX + "scatter_gm_offset", scatterTileInfo.srcTileInfo.offset);
        }
        if (scatterPara.inplace) {
            op.SetAttribute(OpAttributeKey::inplaceIdx, 0);
            op.SetAttr(OpAttributeKey::inplaceInfo, std::map<int, int>{{0, 0}});
            if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) {
                op.SetAttribute(OpAttributeKey::dontTouch, true);
            }
        }
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        op.SetAttribute(OpAttributeKey::scalar, scalar);
        op.SetAttribute(OP_ATTR_PREFIX + "scatter_mode", mode);
        return;
    }

    // 按照dstShape进行切分
    auto& vecTile = tileShape.GetVecTile();
    CHECK(VectorErrorCode::ERR_CONFIG_TILE, dynamicGm || vecTile[axis] >= dstTensor->shape[axis])
        << "The axis is not supported for tile splitting";
    CHECK(VectorErrorCode::ERR_CONFIG_TILE, vecTile[axis] >= idxInput->shape[axis])
        << "The axis is not supported for tile splitting";
    int64_t tmpTile = vecTile[cur];
    if (static_cast<int>(cur) == axis) {
        tmpTile = std::max(dstTensor->shape[axis], idxInput->shape[axis]);
    }
    for (int i = 0; i < idxInput->shape[cur]; i += tmpTile) {
        if (static_cast<int>(cur) == axis) {
            scatterTileInfo.idxTileInfo.offset[cur] = 0;
            scatterTileInfo.idxTileInfo.shape[cur] = idxInput->shape[cur];
            scatterTileInfo.dstTileInfo.offset[cur] = 0;
            scatterTileInfo.dstTileInfo.shape[cur] = dstTensor->shape[cur];
            scatterTileInfo.srcTileInfo.offset[cur] = 0;
            scatterTileInfo.srcTileInfo.shape[cur] = srcInput->shape[cur];
        } else {
            scatterTileInfo.idxTileInfo.offset[cur] = i % idxInput->shape[cur];
            scatterTileInfo.idxTileInfo.shape[cur] = std::min(
                idxInput->shape[cur] - scatterTileInfo.idxTileInfo.offset[cur], tmpTile);
            scatterTileInfo.dstTileInfo.offset[cur] = i;
            scatterTileInfo.dstTileInfo.shape[cur] = std::min(
                idxInput->shape[cur] - scatterTileInfo.idxTileInfo.offset[cur], tmpTile);
            scatterTileInfo.srcTileInfo.offset[cur] = i;
            scatterTileInfo.srcTileInfo.shape[cur] = std::min(
                idxInput->shape[cur] - scatterTileInfo.idxTileInfo.offset[cur], tmpTile);
        }
        InnerTiledScatterElementS(cur + 1, function, tileShape, scatterPara, scatterTileInfo);
    }
}

void TiledScatterElementS(Function& function, const TileShape& tileShape, const ScatterElementSPara& scatterPara)
{
    if (scatterPara.inplace && std::any_of(scatterPara.srcInput->shape.begin(), scatterPara.srcInput->shape.end(),
                                           [](int64_t dim) { return dim < 0; })) {
        auto memoryType = scatterPara.srcInput->GetMemoryTypeOriginal();
        CHECK(VectorErrorCode::ERR_PARAM_INVALID,
              memoryType == MemoryType::MEM_UNKNOWN || memoryType == MemoryType::MEM_DEVICE_DDR)
            << "Dynamic Scatter self requires GM storage";
        scatterPara.srcInput->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
        scatterPara.dstTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    }
    // Check Operands Valid
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, scatterPara.srcInput->shape.size() == scatterPara.srcInput->offset.size())
        << "The size of srcInput shape and offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, scatterPara.idxInput->shape.size() == scatterPara.idxInput->offset.size())
        << "The size of idxInput shape and offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          scatterPara.dstTensor->shape.size() == scatterPara.dstTensor->offset.size())
        << "The size of dst shape and offset should be equal";

    ScatterElementSTileInfoPara scatterTileInfo{
        TileInfo(scatterPara.srcInput->shape.size(), scatterPara.srcInput->offset.size()),
        TileInfo(scatterPara.idxInput->shape.size(), scatterPara.idxInput->offset.size()),
        TileInfo(scatterPara.dstTensor->shape.size(), scatterPara.dstTensor->offset.size()),
    };
    InnerTiledScatterElementS(0, function, tileShape, scatterPara, scatterTileInfo);
}

void TensorScatterElementS(Function& function, const ScatterElementSPara& scatterPara)
{
    auto& op = GraphUtils::AddDynOperation(function, Opcode::OP_SCATTER_ELEMENT,
                                           {scatterPara.srcInput, scatterPara.idxInput}, {scatterPara.dstTensor});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", scatterPara.axis);
    op.SetAttribute(OpAttributeKey::scalar, scatterPara.scalar);
    op.SetAttribute(OP_ATTR_PREFIX + "scatter_mode", scatterPara.scatterMode);
    op.SetAttr(OpAttributeKey::inplaceInfo, std::map<int, int>{{0, 0}});
}

static void CheckScatterElementSParamsInvalid(const Tensor& self, const Tensor& indices, const Element& src, int axis,
                                              const ScatterMode reduce)
{
    const auto& combinations = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_SCATTER_ELEMENT);
    CheckTensorDataTypeCombination({self.GetDataType(), indices.GetDataType()}, combinations, "SCATTER");
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, src.GetDataType() == self.GetDataType())
        << "Scatter scalar dtype must match self dtype";
    std::vector<LogicalTensorPtr> tensors = {self.GetStorage(), indices.GetStorage()};
    CheckTensorsDimConsistency(tensors, "SCATTER");
    CheckTensorsFormatConsistency(self.GetStorage(), indices.GetStorage(), "SCATTER");
    CheckAxisRange(self, axis);
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, reduce <= ScatterMode::UNKNOWN)
        << "The ScatterMode of reduce should be less than UNKNOWN";
    for (size_t i = 0; i < self.GetShape().size(); i++) {
        if (static_cast<int>(i) == axis) {
            continue;
        }
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, indices.GetShape()[i] <= self.GetShape()[i])
            << "The shape of indices and self should be equal";
    }
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "SCATTER");
}

Tensor Scatter(const Tensor& self, const Tensor& indices, const Element& src, int axis, ScatterMode reduce)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Scatter");
    CheckVectorOperand(indices.GetStorage(), "Scatter");

    CheckScatterElementSParamsInvalid(self, indices, src, axis < 0 ? self.GetShape().size() + axis : axis, reduce);
    DataType orgDtype = self.GetDataType();
    auto operandCast = Tensor(DataType::DT_FP32, self.GetShape());
    if ((orgDtype == DataType::DT_FP16 || orgDtype == DataType::DT_BF16) &&
        (reduce == ScatterMode::ADD || reduce == ScatterMode::MULTIPLY)) {
        operandCast = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                           self.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    } else {
        operandCast = self;
    }
    axis = axis < 0 ? operandCast.GetShape().size() + axis : axis;
    Tensor result(operandCast.GetStorage()->Datatype(), operandCast.GetShape());
    result.GetStorage()->UpdateDynValidShape(operandCast.GetStorage()->GetDynValidShape());
    CALL(ScatterElementS, *Program::GetInstance().GetCurrentFunction(),
         {result.GetStorage(), operandCast.GetStorage(), indices.GetStorage(), src, axis, static_cast<int>(reduce)});

    if ((orgDtype == DataType::DT_FP16 || orgDtype == DataType::DT_BF16) &&
        (reduce == ScatterMode::ADD || reduce == ScatterMode::MULTIPLY)) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), result.GetStorage(),
                    orgDtype, CastMode::CAST_RINT);
    }
    return result;
}

struct ScatterPara {
    const LogicalTensorPtr& dstTensor;
    const LogicalTensorPtr& selfInput;
    const LogicalTensorPtr& idxInput;
    const LogicalTensorPtr& srcInput;
    const int axis;
    const int scatterMode;
    const bool inplace;
};

struct ScatterTileInfoPara {
    TileInfo srcInfo;
    TileInfo idxInfo;
    TileInfo dstInfo;
    TileInfo selfInfo;
};

// An unbounded self cannot reside in UB. Keep its runtime GM extent intact.
bool HasDynamicScatterSelf(const ScatterPara& scatterPara)
{
    const auto& shape = scatterPara.selfInput->GetShape();
    return scatterPara.inplace && std::any_of(shape.begin(), shape.end(), [](int64_t dim) { return dim < 0; });
}

void SetDynamicScatterMemory(const ScatterPara& scatterPara)
{
    if (!HasDynamicScatterSelf(scatterPara)) {
        return;
    }
    auto memoryType = scatterPara.selfInput->GetMemoryTypeOriginal();
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          memoryType == MemoryType::MEM_UNKNOWN || memoryType == MemoryType::MEM_DEVICE_DDR)
        << "Dynamic Scatter self requires GM storage";
    scatterPara.selfInput->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    scatterPara.dstTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
}

void InnerTiledScatter(size_t cur, Function& function, const TileShape& tileShape, const ScatterPara& scatterPara,
                       ScatterTileInfoPara& scatterTileInfo)
{
    const LogicalTensorPtr& dstTensor = scatterPara.dstTensor;
    const LogicalTensorPtr& selfInput = scatterPara.selfInput;
    const LogicalTensorPtr& idxInput = scatterPara.idxInput;
    const LogicalTensorPtr& srcInput = scatterPara.srcInput;
    const int axis = scatterPara.axis;
    const int mode = scatterPara.scatterMode;

    if (cur == dstTensor->shape.size()) {
        // add Operation
        const bool dynamicGm = HasDynamicScatterSelf(scatterPara);
        auto selfTile = dynamicGm ?
                            selfInput :
                            selfInput->View(function, scatterTileInfo.selfInfo.shape, scatterTileInfo.selfInfo.offset);
        auto idxTile = idxInput->View(function, scatterTileInfo.idxInfo.shape, scatterTileInfo.idxInfo.offset);
        auto srcTile = srcInput->View(function, scatterTileInfo.srcInfo.shape, scatterTileInfo.srcInfo.offset);
        const bool inplaceToGm = scatterPara.inplace &&
                                 selfInput->GetMemoryTypeOriginal() == MemoryType::MEM_DEVICE_DDR;
        auto dstTile = inplaceToGm ?
                           dstTensor :
                           dstTensor->View(function, scatterTileInfo.dstInfo.shape, scatterTileInfo.dstInfo.offset);
        Shape tmpShape({idxTile->GetShape()[idxTile->GetShape().size() - 1]});
        auto tmpBuffer = std::make_shared<LogicalTensor>(function, idxTile->Datatype(), tmpShape);
        auto opcode = scatterPara.inplace ? Opcode::OP_SCATTER_INPLACE : Opcode::OP_SCATTER;
        auto& op = function.AddOperation(opcode, {selfTile, idxTile, srcTile}, {dstTile, tmpBuffer});
        op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
        op.SetAttribute(OP_ATTR_PREFIX + "scatter_mode", mode);
        if (dynamicGm) {
            op.SetAttribute(OP_ATTR_PREFIX + "scatter_gm_offset", scatterTileInfo.selfInfo.offset);
        }
        if (scatterPara.inplace) {
            std::map<int, int> inplaceInfo = {{0, 0}};
            op.SetAttr(OpAttributeKey::inplaceInfo, inplaceInfo);
            op.SetAttribute(OpAttributeKey::inplaceIdx, 0);
            if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) {
                op.SetAttribute(OpAttributeKey::dontTouch, true);
            }
        }
        return;
    }

    // 按照dstShape进行切分
    auto& vecTile = tileShape.GetVecTile();
    CHECK(VectorErrorCode::ERR_CONFIG_TILE,
          HasDynamicScatterSelf(scatterPara) || vecTile[axis] >= dstTensor->shape[axis])
        << "The axis is not supported for tile splitting";
    CHECK(VectorErrorCode::ERR_CONFIG_TILE, vecTile[axis] >= idxInput->shape[axis])
        << "The axis is not supported for tile splitting";
    int64_t tmpTile = vecTile[cur];
    if (static_cast<int>(cur) == axis) {
        tmpTile = std::max(dstTensor->shape[axis], idxInput->shape[axis]);
    }
    for (int i = 0; i < idxInput->shape[cur]; i += tmpTile) {
        if (static_cast<int>(cur) == axis) {
            scatterTileInfo.idxInfo.offset[cur] = 0;
            scatterTileInfo.idxInfo.shape[cur] = idxInput->shape[cur];
            scatterTileInfo.dstInfo.offset[cur] = 0;
            scatterTileInfo.dstInfo.shape[cur] = dstTensor->shape[cur];
            scatterTileInfo.srcInfo.offset[cur] = 0;
            scatterTileInfo.srcInfo.shape[cur] = idxInput->shape[cur];
            scatterTileInfo.selfInfo.offset[cur] = 0;
            scatterTileInfo.selfInfo.shape[cur] = selfInput->shape[cur];
        } else {
            scatterTileInfo.idxInfo.offset[cur] = i % idxInput->shape[cur];
            scatterTileInfo.idxInfo.shape[cur] = std::min(idxInput->shape[cur] - scatterTileInfo.idxInfo.offset[cur],
                                                          tmpTile);
            scatterTileInfo.dstInfo.offset[cur] = i;
            scatterTileInfo.dstInfo.shape[cur] = std::min(idxInput->shape[cur] - scatterTileInfo.idxInfo.offset[cur],
                                                          tmpTile);
            scatterTileInfo.srcInfo.offset[cur] = i;
            scatterTileInfo.srcInfo.shape[cur] = std::min(idxInput->shape[cur] - scatterTileInfo.idxInfo.offset[cur],
                                                          tmpTile);
            scatterTileInfo.selfInfo.offset[cur] = i;
            scatterTileInfo.selfInfo.shape[cur] = std::min(idxInput->shape[cur] - scatterTileInfo.idxInfo.offset[cur],
                                                           tmpTile);
        }
        InnerTiledScatter(cur + 1, function, tileShape, scatterPara, scatterTileInfo);
    }
}

void TiledScatter(Function& function, const TileShape& tileShape, const ScatterPara& scatterPara)
{
    SetDynamicScatterMemory(scatterPara);
    // Check Operands Valid
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, scatterPara.srcInput->shape.size() == scatterPara.srcInput->offset.size())
        << "The shape size of srcInput and offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, scatterPara.idxInput->shape.size() == scatterPara.idxInput->offset.size())
        << "The shape size of idxInput and offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          scatterPara.dstTensor->shape.size() == scatterPara.dstTensor->offset.size())
        << "The shape size of dst and offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          scatterPara.selfInput->shape.size() == scatterPara.selfInput->offset.size())
        << "The shape size of selfInput and offset should be equal";

    ScatterTileInfoPara scatterTileInfo{
        TileInfo(scatterPara.srcInput->shape.size(), scatterPara.srcInput->offset.size()),
        TileInfo(scatterPara.idxInput->shape.size(), scatterPara.idxInput->offset.size()),
        TileInfo(scatterPara.dstTensor->shape.size(), scatterPara.dstTensor->offset.size()),
        TileInfo(scatterPara.selfInput->shape.size(), scatterPara.selfInput->offset.size()),
    };
    InnerTiledScatter(0, function, tileShape, scatterPara, scatterTileInfo);
}

void TensorScatter(Function& function, const ScatterPara& scatterPara)
{
    SetDynamicScatterMemory(scatterPara);
    auto opcode = scatterPara.inplace ? Opcode::OP_SCATTER_INPLACE : Opcode::OP_SCATTER;
    auto& op = GraphUtils::AddDynOperation(
        function, opcode, {scatterPara.selfInput, scatterPara.idxInput, scatterPara.srcInput}, {scatterPara.dstTensor});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", scatterPara.axis);
    op.SetAttribute(OP_ATTR_PREFIX + "scatter_mode", scatterPara.scatterMode);
    if (scatterPara.inplace) {
        std::map<int, int> inplaceInfo = {{0, 0}};
        op.SetAttr(OpAttributeKey::inplaceInfo, inplaceInfo);
        op.SetAttribute(OpAttributeKey::inplaceIdx, 0);
        if (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) {
            op.SetAttribute(OpAttributeKey::dontTouch, true);
        }
    } else {
        op.SetAttr(OpAttributeKey::inplaceInfo, std::map<int, int>{{0, 0}});
    }
}

static void CheckScatterParamsInvalid(const Tensor& self, const Tensor& indices, const Tensor& src, int axis,
                                      const ScatterMode reduce)
{
    const auto& combinations = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_SCATTER);
    CheckTensorDataTypeCombination({self.GetDataType(), indices.GetDataType(), src.GetDataType()}, combinations,
                                   "SCATTER");
    std::vector<LogicalTensorPtr> tensors = {self.GetStorage(), indices.GetStorage(), src.GetStorage()};
    CheckTensorsDimConsistency(tensors, "SCATTER");
    CheckTensorsFormatConsistency(self.GetStorage(), indices.GetStorage(), "SCATTER");
    CheckTensorsFormatConsistency(self.GetStorage(), src.GetStorage(), "SCATTER");
    CheckTensorsFormatConsistency(indices.GetStorage(), src.GetStorage(), "SCATTER");
    CheckAxisRange(self, axis);
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, reduce <= ScatterMode::UNKNOWN)
        << "The ScatterMode of reduce should be less than UNKNOWN";
    for (size_t i = 0; i < self.GetShape().size(); i++) {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, indices.GetShape()[i] <= src.GetShape()[i])
            << "The shape size of src and indices should be equal";
        if (static_cast<int>(i) == axis) {
            continue;
        }
        CHECK(VectorErrorCode::ERR_PARAM_INVALID,
              (Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510 && self.GetShape()[i] < 0) ||
                  indices.GetShape()[i] <= self.GetShape()[i])
            << "The shape size of src and indices should be equal";
    }
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "SCATTER");
}

Tensor Scatter(const Tensor& self, const Tensor& indices, const Tensor& src, int axis, ScatterMode reduce)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Scatter");
    CheckVectorOperand(indices.GetStorage(), "Scatter");
    CheckVectorOperand(src.GetStorage(), "Scatter");

    CheckScatterParamsInvalid(self, indices, src, axis < 0 ? self.GetShape().size() + axis : axis, reduce);
    DataType orgDtype = self.GetDataType();
    auto operandSelfCast = Tensor(DataType::DT_FP32, self.GetShape());
    auto operandSrcCast = Tensor(DataType::DT_FP32, src.GetShape());
    if ((orgDtype == DataType::DT_FP16 || orgDtype == DataType::DT_BF16) &&
        (reduce == ScatterMode::ADD || reduce == ScatterMode::MULTIPLY)) {
        operandSelfCast = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                               self.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
        operandSrcCast = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                              src.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    } else {
        operandSelfCast = self;
        operandSrcCast = src;
    }
    axis = axis < 0 ? operandSelfCast.GetShape().size() + axis : axis;
    Tensor result(operandSelfCast.GetStorage()->Datatype(), operandSelfCast.GetShape());
    result.GetStorage()->UpdateDynValidShape(operandSelfCast.GetStorage()->GetDynValidShape());
    CALL(Scatter, *Program::GetInstance().GetCurrentFunction(),
         {result.GetStorage(), operandSelfCast.GetStorage(), indices.GetStorage(), operandSrcCast.GetStorage(), axis,
          static_cast<int>(reduce), false});

    if ((orgDtype == DataType::DT_FP16 || orgDtype == DataType::DT_BF16) &&
        (reduce == ScatterMode::ADD || reduce == ScatterMode::MULTIPLY)) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), result.GetStorage(),
                    orgDtype, CastMode::CAST_RINT);
    }
    return result;
}

void Scatter_(Tensor& self, const Tensor& indices, const Element& src, int axis, ScatterMode reduce)
{
    const bool a5 = Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510;
    if (!a5 || reduce != ScatterMode::NONE) {
        self = Scatter(self, indices, src, axis, reduce);
        return;
    }
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Scatter_");
    CheckVectorOperand(indices.GetStorage(), "Scatter_");
    CheckScatterElementSParamsInvalid(self, indices, src, axis < 0 ? self.GetShape().size() + axis : axis, reduce);
    axis = axis < 0 ? self.GetShape().size() + axis : axis;
    Tensor dst(self.GetDataType(), self.GetShape());
    dst.GetStorage()->UpdateDynValidShape(self.GetStorage()->GetDynValidShape());
    dst.GetStorage()->tensor->UpdateDynRawShape(self.GetStorage()->tensor->GetDynRawShape());
    if (std::any_of(self.GetShape().begin(), self.GetShape().end(), [](int64_t dim) { return dim < 0; })) {
        auto memoryType = self.GetStorage()->GetMemoryTypeOriginal();
        CHECK(VectorErrorCode::ERR_PARAM_INVALID,
              memoryType == MemoryType::MEM_UNKNOWN || memoryType == MemoryType::MEM_DEVICE_DDR)
            << "Dynamic Scatter self requires GM storage";
        self.GetStorage()->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
        dst.GetStorage()->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    }
    auto& op = GraphUtils::AddDynOperation(*Program::GetInstance().GetCurrentFunction(),
                                           Opcode::OP_SCATTER_ELEMENT_INPLACE,
                                           {self.GetStorage(), indices.GetStorage()}, {dst.GetStorage()});
    op.SetAttribute(OP_ATTR_PREFIX + "axis", axis);
    op.SetAttribute(OpAttributeKey::scalar, src);
    op.SetAttribute(OP_ATTR_PREFIX + "scatter_mode", static_cast<int>(reduce));
    op.SetAttribute(OpAttributeKey::inplaceIdx, 0);
    op.SetAttr(OpAttributeKey::inplaceInfo, std::map<int, int>{{0, 0}});
    op.SetAttribute(OpAttributeKey::dontTouch, true);
    Program::GetInstance().GetCurrentFunction()->SetSameMemId(self.GetStorage(), dst.GetStorage());
    self = dst;
}

void Scatter_(Tensor& self, const Tensor& indices, const Tensor& src, int axis, ScatterMode reduce)
{
    const bool a5 = Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510;
    if (!a5 || reduce != ScatterMode::NONE) {
        self = Scatter(self, indices, src, axis, reduce);
        return;
    }

    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Scatter_");
    CheckVectorOperand(indices.GetStorage(), "Scatter_");
    CheckVectorOperand(src.GetStorage(), "Scatter_");
    CheckScatterParamsInvalid(self, indices, src, axis < 0 ? self.GetShape().size() + axis : axis, reduce);

    axis = axis < 0 ? self.GetShape().size() + axis : axis;
    Tensor dst(self.GetDataType(), self.GetShape());
    dst.GetStorage()->UpdateDynValidShape(self.GetStorage()->GetDynValidShape());
    dst.GetStorage()->tensor->UpdateDynRawShape(self.GetStorage()->tensor->GetDynRawShape());
    CALL(Scatter, *Program::GetInstance().GetCurrentFunction(),
         {dst.GetStorage(), self.GetStorage(), indices.GetStorage(), src.GetStorage(), axis, static_cast<int>(reduce),
          true});
    Program::GetInstance().GetCurrentFunction()->SetSameMemId(self.GetStorage(), dst.GetStorage());
    self = dst;
}

void ScatterElementSOperationTileFunc(Function& function, const TileShape& tileShape,
                                      const std::vector<LogicalTensorPtr>& iOperand,
                                      const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    int axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
    Element scalar = op.GetElementAttribute(OpAttributeKey::scalar);
    int scatterMode = op.GetIntAttribute(OP_ATTR_PREFIX + "scatter_mode");
    TiledScatterElementS(function, tileShape,
                         {oOperand[0], iOperand[0], iOperand[1], scalar, axis, scatterMode,
                          op.GetOpcode() == Opcode::OP_SCATTER_ELEMENT_INPLACE});
}

void ScatterOperationTileFunc(Function& function, const TileShape& tileShape,
                              const std::vector<LogicalTensorPtr>& iOperand,
                              const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    int axis = op.GetIntAttribute(OP_ATTR_PREFIX + "axis");
    int scatterMode = op.GetIntAttribute(OP_ATTR_PREFIX + "scatter_mode");
    TiledScatter(function, tileShape,
                 {oOperand[0], iOperand[0], iOperand[1], iOperand[NUM_VALUE_2], axis, scatterMode,
                  op.GetOpcode() == Opcode::OP_SCATTER_INPLACE});
}

REGISTER_OPERATION_TILED_FUNC(OP_SCATTER_ELEMENT, Opcode::OP_SCATTER_ELEMENT, ScatterElementSOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_SCATTER_ELEMENT_INPLACE, Opcode::OP_SCATTER_ELEMENT_INPLACE,
                              ScatterElementSOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_SCATTER, Opcode::OP_SCATTER, ScatterOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_SCATTER_INPLACE, Opcode::OP_SCATTER_INPLACE, ScatterOperationTileFunc);

} // namespace npu::tile_fwk
