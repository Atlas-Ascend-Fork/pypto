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
 * \file logarithmic.cpp
 * \brief
 */

#include "unary_tiled.h"
#include "binary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

Tensor Ln(const Tensor& operand, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckVectorOperand(operand.GetStorage(), "Ln");

    std::unordered_set<DataType> supportedTypes = {DT_FP16, DT_BF16, DT_FP32};
    CheckTensorDataType(operand.GetStorage(), supportedTypes, "Ln");

    auto [result, op] = TensorUnaryOperationWithOp<UnaryOpType::LN>(*Program::GetInstance().GetCurrentFunction(),
                                                                    operand.GetStorage());
    op->SetAttribute(OpAttributeKey::precisionType, static_cast<int64_t>(precisionType));
    return Tensor(result);
}

void LnOperationTileFunc(Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
                         const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    int64_t precisionType = static_cast<int64_t>(PrecisionType::INTRINSIC);
    if (op.HasAttr(OpAttributeKey::precisionType)) {
        precisionType = op.GetIntAttribute(OpAttributeKey::precisionType);
    }
    return TiledUnaryOperation<UnaryOpType::LN>(function, tileShape, iOperand[0], oOperand[0], 0, precisionType);
}

Tensor Log(const Tensor& self, LogBaseType base, PrecisionType precisionType)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "Log");

    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          base == LogBaseType::LOG_E || base == LogBaseType::LOG_2 || base == LogBaseType::LOG_10)
        << "base is incorrect";
    std::unordered_set<DataType> supportedTypes = {DT_BF16, DT_FP16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "LOG");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "LOG");

    auto operandCast = Tensor(DataType::DT_FP32, self.GetShape());
    if (self.GetStorage()->Datatype() == DataType::DT_FP16 || self.GetStorage()->Datatype() == DataType::DT_BF16) {
        operandCast = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                           self.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    } else {
        operandCast = self;
    }

    auto resTensor = Tensor(DataType::DT_FP32, self.GetShape());
    resTensor = Ln(operandCast, precisionType);

    auto resTensorBeforeCast = Tensor(DataType::DT_FP32, self.GetShape());
    if (base == LogBaseType::LOG_2) {
        resTensorBeforeCast = CALL(BinaryOperationScalar<BinaryOpType::DIV>,
                                   *Program::GetInstance().GetCurrentFunction(), resTensor.GetStorage(),
                                   Element(DataType::DT_FP32, std::log(static_cast<float>(NUM_VALUE_2))));
    } else if (base == LogBaseType::LOG_10) {
        resTensorBeforeCast = CALL(BinaryOperationScalar<BinaryOpType::DIV>,
                                   *Program::GetInstance().GetCurrentFunction(), resTensor.GetStorage(),
                                   Element(DataType::DT_FP32, std::log(static_cast<float>(NUM_VALUE_10))));
    } else {
        resTensorBeforeCast = resTensor;
    }

    if (self.GetStorage()->Datatype() == DataType::DT_FP16) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                    resTensorBeforeCast.GetStorage(), DataType::DT_FP16, CastMode::CAST_NONE);
    } else if (self.GetStorage()->Datatype() == DataType::DT_BF16) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                    resTensorBeforeCast.GetStorage(), DataType::DT_BF16, CastMode::CAST_NONE);
    }
    return resTensorBeforeCast;
}

Tensor Log1p(const Tensor& self)
{
    DECLARE_TRACER();
    auto& function = *Program::GetInstance().GetCurrentFunction();
    CheckVectorOperand(self.GetStorage(), "Log1p");

    std::unordered_set<DataType> supportedTypes = {DT_BF16, DT_FP16, DT_FP32};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "LOG1P");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "LOG1P");

    auto operandCast = self.GetStorage();
    if (self.GetStorage()->Datatype() == DataType::DT_FP16 || self.GetStorage()->Datatype() == DataType::DT_BF16) {
        operandCast = CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(),
                           self.GetStorage(), DataType::DT_FP32, CastMode::CAST_NONE);
    }

    auto resTensorBeforeCast = std::make_shared<LogicalTensor>(*Program::GetInstance().GetCurrentFunction(),
                                                               DataType::DT_FP32, self.GetShape(),
                                                               self.GetStorage()->GetDynValidShape());
    function.AddOperation(Opcode::OP_LOG1P, {operandCast}, {resTensorBeforeCast});

    if (self.GetStorage()->Datatype() == DataType::DT_FP16) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), resTensorBeforeCast,
                    DataType::DT_FP16, CastMode::CAST_NONE);
    } else if (self.GetStorage()->Datatype() == DataType::DT_BF16) {
        RETURN_CALL(CastOperation<CastOpType::CAST>, *Program::GetInstance().GetCurrentFunction(), resTensorBeforeCast,
                    DataType::DT_BF16, CastMode::CAST_NONE);
    }
    return Tensor(resTensorBeforeCast);
}

void TiledLog1pOperation(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                         const LogicalTensorPtr& result)
{
    if (cur == input.tensor.GetShape().size()) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultTile = result->View(function, input.tileInfo.shape, input.tileInfo.offset);
        const auto& vecTile = tileShape.GetVecTile().tile;
        auto tileShapeLen = vecTile.size();
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, SHAPE_DIM1 <= tileShapeLen && tileShapeLen <= SHAPE_DIM4)
            << "Length of tile shape only supports 1~4";

        // The tileop keeps four scratch blocks (tmp0/tmp1/tmp2/mask), each the size of the last
        // two tile axes (tileH * tileW). Align the last axis to 32 bytes; treat 1-D tiles as tileH == 1.
        constexpr size_t ALIGN_SIZE = NUM_VALUE_32;
        auto alignElems = ALIGN_SIZE / BytesOf(DT_FP32);
        int64_t alignedW = AlignUp(vecTile[tileShapeLen - 1], alignElems);
        int64_t tileH = (tileShapeLen >= SHAPE_DIM2) ? vecTile[tileShapeLen - NUM_VALUE_2] : 1;
        int64_t tmpSize = NUM_VALUE_4 * alignedW * tileH;
        std::vector<int64_t> tmpShape({tmpSize});
        auto tmpTensor = std::make_shared<LogicalTensor>(function, DT_FP32, tmpShape);
        function.AddOperation(Opcode::OP_LOG1P, {tile}, {resultTile, tmpTensor});
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, vecTile[cur]);
        input.tileInfo.offset[cur] = i;
        TiledLog1pOperation(function, tileShape, cur + 1, input, result);
    }
}

void TiledLog1pOperation(Function& function, const TileShape& tileShape, const LogicalTensorPtr& self,
                         const LogicalTensorPtr& result)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, self->shape.size() == self->offset.size())
        << "Shape size and offset size should be equal";

    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{self, tileInfo};
    TiledLog1pOperation(function, tileShape, 0, input, result);
}

void Log1pOperationTileFunc(Function& function, const TileShape& tileShape,
                            const std::vector<LogicalTensorPtr>& iOperand,
                            const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    UnaryOperationOperandCheck(iOperand, oOperand);
    TiledLog1pOperation(function, tileShape, iOperand[0], oOperand[0]);
}

REGISTER_OPERATION_TILED_FUNC(OP_LN, Opcode::OP_LN, LnOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_LOG1P, Opcode::OP_LOG1P, Log1pOperationTileFunc);

} // namespace npu::tile_fwk
