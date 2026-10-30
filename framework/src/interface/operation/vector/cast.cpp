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
 * \file cast.cpp
 * \brief
 */

#include "unary.h"
#include <string>
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "tilefwk/error_code.h"
#include "tilefwk/platform.h"

namespace npu::tile_fwk {

static bool IsFp4Type(DataType dtype)
{
    return dtype == DT_FP4_E2M1 || dtype == DT_FP4_E1M2 || dtype == DT_FP4_E2M1X2 || dtype == DT_FP4_E1M2X2;
}

template <CastOpType T>
void TiledCastOperation(Function& function, const TileShape& tileShape, const int cur, Input& input,
                        const LogicalTensorPtr& result, const CastMode& mode, const SaturationMode& satmode)
{
    if (cur == static_cast<int>(input.tensor.GetShape().size())) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);

        DataType srcDtype = tile->Datatype();
        DataType dstDtype = resultTile->Datatype();

        bool needTmpBuffer = false;
        if (((srcDtype == DT_FP32 && dstDtype == DT_INT16) || (srcDtype == DT_FP16 && dstDtype == DT_INT16) ||
             (srcDtype == DT_FP16 && dstDtype == DT_INT8)) &&
            satmode == SaturationMode::OFF) {
            needTmpBuffer = true;
        }

        Operation* op = nullptr;
        if (needTmpBuffer) {
            size_t shapeSize = input.tileInfo.shape.size();
            int64_t shapeW = (shapeSize >= 1) ? input.tileInfo.shape[shapeSize - 1] : 1;
            shapeW = AlignUp(shapeW + ALIGN_SIZE_64, static_cast<int64_t>(BLOCK_SIZE / BytesOf(DT_INT32)));
            int64_t shapeH = (shapeSize >= NUM_VALUE_2) ? input.tileInfo.shape[shapeSize - NUM_VALUE_2] : 1;
            shapeH = std::min(shapeH, static_cast<int64_t>(MAX_REPEAT));
            std::vector<int64_t> tmpShape = {shapeH, shapeW};
            auto tmpTensor = std::make_shared<LogicalTensor>(function, DT_INT32, tmpShape);
            op = &function.AddOperation(GetCastOpName<T>(), {tile}, {resultTile, tmpTensor});
        } else {
            op = &function.AddOperation(GetCastOpName<T>(), {tile}, {resultTile});
        }
        op->SetAttribute(OP_ATTR_PREFIX + "mode", mode);
        op->SetAttribute(OP_ATTR_PREFIX + "satmode", static_cast<int64_t>(satmode));
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledCastOperation<T>(function, tileShape, cur + 1, input, result, mode, satmode);
    }
}

template <CastOpType T>
void TiledCastOperation(Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand,
                        const LogicalTensorPtr& result, const CastMode& mode, const SaturationMode& satmode)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, operand->shape.size() == operand->offset.size())
        << "The shape size of operand and offset should be equal";

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};
    TiledCastOperation<T>(function, tileShape, 0, input, result, mode, satmode);
}

void CheckCastTypeSupport(DataType srcType, DataType dstType, const std::string& opName)
{
    // 同类型转换始终支持
    if (srcType == dstType) {
        return;
    }

    auto arch = Platform::Instance().GetSoc().GetNPUArch();
    bool isA5Architecture = (arch == NPUArch::DAV_3510);

    if (isA5Architecture) {
        // A5 架构支持的转换
        std::unordered_map<DataType, std::unordered_set<DataType>> a5SupportedConversions = {
            {DT_FP32, {DT_FP16, DT_BF16, DT_INT16, DT_INT32, DT_INT64, DT_FP8E4M3, DT_FP8E5M2, DT_HF8}},
            {DT_FP16, {DT_FP32, DT_INT32, DT_INT16, DT_INT8, DT_UINT8, DT_HF8}},
            {DT_BF16, {DT_FP32, DT_INT32, DT_FP16, DT_FP4_E1M2, DT_FP4_E2M1}},
            {DT_UINT8, {DT_FP16, DT_UINT16}},
            {DT_INT8, {DT_FP16, DT_INT16, DT_INT32}},
            {DT_INT16, {DT_UINT8, DT_FP16, DT_FP32, DT_UINT32, DT_INT32}},
            {DT_INT32, {DT_FP32, DT_INT16, DT_UINT16, DT_INT64, DT_UINT8, DT_FP16}},
            {DT_UINT32, {DT_UINT8, DT_UINT16, DT_INT16}},
            {DT_INT64, {DT_FP32, DT_INT32}},
            {DT_FP8E4M3, {DT_FP32}},
            {DT_FP8E5M2, {DT_FP32}},
            {DT_HF8, {DT_FP32}},
            {DT_FP4_E1M2, {DT_BF16}},
            {DT_FP4_E2M1, {DT_BF16}}};

        if (a5SupportedConversions.count(srcType) == 0 || a5SupportedConversions[srcType].count(dstType) == 0) {
            CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, false)
                << "A5 architecture does not support cast from " << npu::tile_fwk::DataType2String(srcType) << " to "
                << npu::tile_fwk::DataType2String(dstType) << " in " << opName;
        }
    } else if (IsLiteNPU(arch)) {
        // Kirin9030/KirinX90 架构支持的转换
        std::unordered_map<DataType, std::unordered_set<DataType>> liteNPUSupportedConversions = {
            {DT_FP32, {DT_FP16, DT_INT32, DT_INT16}},
            {DT_FP16, {DT_FP32, DT_INT32, DT_INT16, DT_INT8, DT_UINT8}},
            {DT_INT32, {DT_FP32}},
            {DT_INT16, {DT_FP32, DT_FP16}},
            {DT_UINT8, {DT_FP16}},
            {DT_INT8, {DT_FP16}}};

        if (liteNPUSupportedConversions.count(srcType) == 0 ||
            liteNPUSupportedConversions[srcType].count(dstType) == 0) {
            CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, false)
                << "Kirin architecture does not support cast from " << npu::tile_fwk::DataType2String(srcType) << " to "
                << npu::tile_fwk::DataType2String(dstType) << " in " << opName;
        }
    } else {
        // A2A3 架构支持的转换（其他架构也按A2A3处理）
        std::unordered_map<DataType, std::unordered_set<DataType>> a2a3SupportedConversions = {
            {DT_FP16, {DT_FP32, DT_INT32, DT_INT16, DT_INT8, DT_UINT8, DT_INT4}},
            {DT_BF16, {DT_FP32, DT_INT32}},
            {DT_INT32, {DT_FP32, DT_INT16, DT_INT64, DT_FP16}},
            {DT_FP32, {DT_BF16, DT_FP16, DT_INT16, DT_INT32, DT_INT64}},
            {DT_UINT8, {DT_FP16}},
            {DT_INT8, {DT_FP16}},
            {DT_INT16, {DT_FP32, DT_FP16}},
            {DT_INT64, {DT_FP32, DT_INT32}},
            {DT_INT4, {DT_FP16}}};

        if (a2a3SupportedConversions.count(srcType) == 0 || a2a3SupportedConversions[srcType].count(dstType) == 0) {
            CHECK(VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED, false)
                << "A2A3 architecture does not support cast from " << npu::tile_fwk::DataType2String(srcType) << " to "
                << npu::tile_fwk::DataType2String(dstType) << " in " << opName;
        }
    }
}

Tensor Cast(const Tensor& self, DataType dstDataType, CastMode mode, SaturationMode satmode)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Cast");

    CheckCastTypeSupport(self.GetDataType(), dstDataType, "CAST");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "CAST");
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, self.GetShape().size() == self.GetStorage()->offset.size())
        << "The shape size of self and offset should be equal";
    // Cast to same dType with no mode will do nothing
    if (self.GetStorage()->tensor->datatype == dstDataType && (mode == CAST_NONE || mode == CAST_RINT)) {
        return self;
    }
    if (IsFp4Type(self.GetDataType()) || IsFp4Type(dstDataType)) {
        const auto innerSize = self.GetShape().back();
        CHECK(ExternalError::INVALID_SHAPE, innerSize == -1 || (innerSize & 1) == 0)
            << "CAST with FP4 dtype requires an even inner axis size, actual: " << innerSize;
        const auto& vecTile = TileShape::Current().GetVecTile();
        if (vecTile.valid()) {
            CHECK(VectorErrorCode::ERR_CONFIG_TILE, vecTile.size() == self.GetShape().size())
                << "CAST tile shape rank must equal input tensor rank";
            const auto tileSize = vecTile[self.GetShape().size() - 1];
            const auto effectiveTileSize = innerSize == -1 ? tileSize : std::min(innerSize, tileSize);
            CHECK(ExternalError::INVALID_SHAPE, (effectiveTileSize & 1) == 0)
                << "CAST with FP4 dtype requires an even effective tile inner axis size, actual: " << effectiveTileSize;
        }
    }
    RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(),
                dstDataType, mode, satmode);
}

inline void CastOperationOperandCheck(const std::vector<LogicalTensorPtr>& iOperand,
                                      const std::vector<LogicalTensorPtr>& oOperand)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, iOperand.size() == 1) << "The input operand size should be 1";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, oOperand.size() == 1) << "The output operand size should be 1";
}

void CastOperationTileFunc(Function& function, const TileShape& tileShape,
                           const std::vector<LogicalTensorPtr>& iOperand, const std::vector<LogicalTensorPtr>& oOperand,
                           const Operation& op)
{
    CastOperationOperandCheck(iOperand, oOperand);
    int64_t satmodeValue = 1;
    op.GetAttr(OP_ATTR_PREFIX + "satmode", satmodeValue);
    SaturationMode satmode = static_cast<SaturationMode>(satmodeValue);
    auto mode = op.GetCastModeAttribute(OP_ATTR_PREFIX + "mode");
    TiledCastOperation<CastOpType::CAST>(function, tileShape, iOperand[0], oOperand[0], mode, satmode);
}

REGISTER_OPERATION_TILED_FUNC(OP_CAST, Opcode::OP_CAST, CastOperationTileFunc);

} // namespace npu::tile_fwk
