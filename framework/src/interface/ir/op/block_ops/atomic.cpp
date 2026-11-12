/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file atomic.cpp
 * \brief Scalar atomic read-modify-write operations on GM Tensor elements.
 */

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/op_attr_types.h"
#include "ir/op_registry.h"
#include "ir/type.h"
#include "ir/type_inference.h"
#include "pypto_pro/error.h"

namespace pypto {
namespace ir {
using npu::tile_fwk::ExternalError;

namespace {

bool IsDtypeIn(const DataType& dtype, const std::vector<DataType>& supported)
{
    for (const auto& candidate : supported) {
        if (dtype == candidate) {
            return true;
        }
    }
    return false;
}

std::vector<DataType> AtomicRMWDtypes(AtomicRMWMode mode)
{
    switch (mode) {
        case AtomicRMWMode::ADD:
        case AtomicRMWMode::SUB:
        case AtomicRMWMode::EXCH:
        case AtomicRMWMode::MAX:
        case AtomicRMWMode::MIN:
            return {DataType::INT32, DataType::UINT32, DataType::FP32, DataType::INT64, DataType::UINT64};
        case AtomicRMWMode::AND:
        case AtomicRMWMode::OR:
        case AtomicRMWMode::XOR:
            return {DataType::INT32, DataType::UINT32, DataType::INT64, DataType::UINT64};
        case AtomicRMWMode::INC:
        case AtomicRMWMode::DEC:
            return {DataType::UINT32, DataType::UINT64};
        default:
            return {};
    }
}

void CheckAtomicTensor(const std::string& opName, const ExprPtr& tensor)
{
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<TensorType>(tensor->GetType()))
        << opName << " tensor must be a GM Tensor; UB Tile atomics are not supported by this API";
}

void CheckAtomicOffset(const std::string& opName, const ExprPtr& offset)
{
    auto offsetType = As<ScalarType>(offset->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, offsetType != nullptr && offsetType->dtype_ != DataType::BOOL &&
                                                  (offsetType->dtype_.IsInt() || offsetType->dtype_ == DataType::INDEX))
        << opName << " offset must be a non-bool integer scalar";
}

void CheckAtomicValue(const std::string& opName, const std::string& name, const ExprPtr& value,
                      const DataType& elementDtype)
{
    auto valueType = As<ScalarType>(value->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, valueType) << opName << " " << name << " must be a scalar";
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, valueType->dtype_ == elementDtype)
        << opName << " " << name << " dtype must match the element dtype " << elementDtype.ToString() << ", but got "
        << valueType->dtype_.ToString();
}

TypePtr DeduceAtomicRMWType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    constexpr const char* kOpName = "block.atomic_rmw";
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x3)
        << kOpName << " requires tensor, offset and value";
    const int kindValue = GetOpKwarg<int>(kwargs, "kind");
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT,
                 kindValue >= static_cast<int>(AtomicRMWMode::ADD) && kindValue <= static_cast<int>(AtomicRMWMode::XOR))
        << kOpName << " kind " << kindValue << " is not an AtomicRMWMode value";
    const auto kind = static_cast<AtomicRMWMode>(kindValue);

    CheckAtomicTensor(kOpName, args[0]);
    const auto elementDtype = As<TensorType>(args[0]->GetType())->dtype_;
    PRO_IR_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, IsDtypeIn(elementDtype, AtomicRMWDtypes(kind)))
        << kOpName << " kind " << EnumToString(kind) << " does not support dtype " << elementDtype.ToString();
    CheckAtomicOffset(kOpName, args[1]);
    CheckAtomicValue(kOpName, "value", args[2], elementDtype);

    return std::make_shared<ScalarType>(elementDtype);
}

TypePtr DeduceAtomicCasType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    constexpr const char* kOpName = "block.atomic_cas";
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x4)
        << kOpName << " requires tensor, offset, compare and value";
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, kwargs.empty()) << kOpName << " does not accept keyword arguments";

    CheckAtomicTensor(kOpName, args[0]);
    const auto elementDtype = As<TensorType>(args[0]->GetType())->dtype_;
    PRO_IR_CHECK(
        ExternalError::NOT_IMPLEMENTED_ERROR,
        IsDtypeIn(elementDtype, {DataType::INT32, DataType::UINT32, DataType::FP32, DataType::INT64, DataType::UINT64}))
        << kOpName << " does not support dtype " << elementDtype.ToString();
    CheckAtomicOffset(kOpName, args[1]);
    CheckAtomicValue(kOpName, "compare", args[2], elementDtype);
    CheckAtomicValue(kOpName, "value", args[3], elementDtype);

    return std::make_shared<ScalarType>(elementDtype);
}

} // namespace

REGISTER_OP("block.atomic_rmw")
    .set_op_category("BlockOp")
    .set_description("Scalar read-modify-write on one GM Tensor element")
    .add_argument("tensor", "Destination GM Tensor (TensorType)")
    .add_argument("offset", "Linear element offset (ScalarType with integer dtype)")
    .add_argument("value", "Scalar operand matching the element dtype")
    .set_attr<int>("kind")
    .f_deduce_type(DeduceAtomicRMWType);

REGISTER_OP("block.atomic_cas")
    .set_op_category("BlockOp")
    .set_description("Scalar compare-and-swap on one GM Tensor element")
    .add_argument("tensor", "Destination GM Tensor (TensorType)")
    .add_argument("offset", "Linear element offset (ScalarType with integer dtype)")
    .add_argument("compare", "Expected scalar matching the element dtype")
    .add_argument("value", "Replacement scalar matching the element dtype")
    .f_deduce_type(DeduceAtomicCasType);

} // namespace ir
} // namespace pypto
