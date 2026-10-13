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
 * \file concat.cpp
 * \brief
 */

#include "unary.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "tilefwk/error_code.h"
#include "tilefwk/platform.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {

void TensorInnerConcatNew(Function& function, const LogicalTensorPtr& operand, const LogicalTensorPtr& result)
{
    result->UpdateDynValidShape(operand->GetDynValidShape());
    function.AddOperation(Opcode::OP_REGISTER_COPY, {operand}, {result});
}

void InnerConcatNew(Function& function, const LogicalTensorPtr& operand, const LogicalTensorPtr& result)
{
    CALL(InnerConcatNew, function, operand, result);
}

void CheckCat(const std::vector<Tensor>& tensors, int axis)
{
    const auto& combinations = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_CONCAT);
    CheckTensorDataTypeCombination({tensors[0].GetDataType()}, combinations, "CAT");
    CheckAxisRange(tensors[0], axis);
    std::vector<LogicalTensorPtr> tensorPtrs;
    for (auto tensor : tensors) {
        CheckTensorShapeSize(tensor.GetStorage(), "CAT");
        tensorPtrs.push_back(tensor.GetStorage());
    }
    CheckTensorsDimConsistency(tensorPtrs, "CAT");
    CheckTensorsDataTypeConsistency(tensorPtrs, "CAT");
    CheckTensorsFormatConsistency(tensorPtrs, "CAT");
    auto shape = tensors[0].GetShape();
    for (auto tensor : tensors) {
        for (int i = 0; static_cast<size_t>(i) < tensors[0].GetShape().size(); ++i) {
            if (i == axis) {
                continue;
            }
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, shape[i] == tensor.GetShape()[i])
                << "The shape of all tensors should be equal except at axis";
        }
    }
}

Tensor Cat(const std::vector<Tensor>& tensors, int axis)
{
    DECLARE_TRACER();
    for (const auto& tensor : tensors) {
        CheckTensorFormat(tensor.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Cat");
    }

    CheckCat(tensors, axis);
    CheckAxisRange(tensors[0], axis);

    int axisSize = 0;
    SymbolicScalar axisDynSize(0);
    for (auto tensor : tensors) {
        axisSize += tensor.GetShape(axis);
        axisDynSize = axisDynSize + tensor.GetValidShape(axis);
    }

    auto format = tensors[0].Format();
    auto resultShape = tensors[0].GetShape();
    resultShape[axis] = axisSize;
    Tensor result(tensors[0].GetDataType(), resultShape, "", format);

    auto validShape = tensors[0].GetValidShape();
    if (validShape.empty()) {
        validShape = SymbolicScalar::FromConcrete(resultShape);
    }
    validShape[axis] = axisDynSize.Simplify();
    result.GetStorage(false)->UpdateDynValidShape(validShape);

    std::vector<SymbolicScalar> offset(resultShape.size(), 0);
    for (auto tensor : tensors) {
        auto materialized = Assign(tensor);
        Assemble(materialized, offset, result);
        offset[axis] = offset[axis] + tensor.GetShape(axis);
    }

    return result;
}

} // namespace npu::tile_fwk
