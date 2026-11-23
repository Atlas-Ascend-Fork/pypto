/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ir/op_registry.h"

#include <any>
#include <memory>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "core/error.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/memref.h"
#include "ir/span.h"
#include "ir/type.h"
#include "pypto_pro/error.h"

namespace pypto {
namespace ir {
using npu::tile_fwk::ExternalError;

namespace {

std::any NormalizeKwargValue(const std::any& value)
{
    if (value.type() == typeid(DataType)) {
        return std::any(AnyCast<DataType>(value, "kwarg").ToCTypeString());
    }
    if (value.type() == typeid(MemorySpace)) {
        return std::any(MemorySpaceToString(AnyCast<MemorySpace>(value, "kwarg")));
    }
    if (value.type() == typeid(int)) {
        return std::any(static_cast<int64_t>(AnyCast<int>(value, "kwarg")));
    }
    if (value.type() == typeid(std::vector<int>)) {
        const auto& numbers = AnyCast<std::vector<int>>(value, "kwarg");
        return std::any(std::vector<int64_t>(numbers.begin(), numbers.end()));
    }
    return value;
}

std::vector<std::pair<std::string, std::any>> NormalizeKwargs(
    const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    std::vector<std::pair<std::string, std::any>> normalized;
    normalized.reserve(kwargs.size());
    for (const auto& [key, value] : kwargs) {
        normalized.emplace_back(key, NormalizeKwargValue(value));
    }
    return normalized;
}

} // namespace

void ValidateKwargs(const std::vector<std::pair<std::string, std::any>>& kwargs,
                    const std::unordered_map<std::string, std::type_index>& allowedKwargs, const std::string& opName)
{
    for (const auto& [key, value] : kwargs) {
        auto it = allowedKwargs.find(key);
        if (it == allowedKwargs.end()) {
            PRO_IR_THROW(::pypto::ir::ValueError, ExternalError::INVALID_ARGUMENT)
                << "Unknown kwarg '" << key << "' for operator '" << opName << "'";
        }

        if (it->second == std::type_index(typeid(DataType))) {
            PRO_IR_CHECK(ExternalError::INVALID_TYPE,
                         std::type_index(value.type()) == std::type_index(typeid(std::string)))
                << "Kwarg '" << key << "' for operator '" << opName
                << "' expects a dtype string, but got incompatible type";
        } else if (it->second == std::type_index(typeid(int))) {
            PRO_IR_CHECK(ExternalError::INVALID_TYPE, std::type_index(value.type()) == std::type_index(typeid(int64_t)))
                << "Kwarg '" << key << "' for operator '" << opName << "' expects int64, but got incompatible type";
        } else if (it->second == std::type_index(typeid(MemorySpace))) {
            PRO_IR_CHECK(ExternalError::INVALID_VAL,
                         std::type_index(value.type()) == std::type_index(typeid(std::string)))
                << "Kwarg '" << key << "' for operator '" << opName
                << "' expects a memory-space string, but got incompatible type";
        } else if (it->second == std::type_index(typeid(std::vector<int>))) {
            PRO_IR_CHECK(ExternalError::INVALID_TYPE,
                         std::type_index(value.type()) == std::type_index(typeid(std::vector<int64_t>)))
                << "Kwarg '" << key << "' for operator '" << opName
                << "' expects std::vector<int64_t>, but got incompatible type";
        } else if (it->second == std::type_index(typeid(std::vector<std::string>))) {
            PRO_IR_CHECK(ExternalError::INVALID_TYPE,
                         std::type_index(value.type()) == std::type_index(typeid(std::vector<std::string>)))
                << "Kwarg '" << key << "' for operator '" << opName
                << "' expects std::vector<std::string>, but got incompatible type";
        } else if (std::type_index(value.type()) != it->second) {
            PRO_IR_CHECK(ExternalError::INVALID_TYPE, false)
                << "Kwarg '" << key << "' for operator '" << opName << "' has incompatible type";
        }
    }
}

OpRegistry& OpRegistry::GetInstance()
{
    static OpRegistry instance;
    return instance;
}

OpRegistryEntry& OpRegistry::Register(const std::string& opName)
{
    // Check if operator is already registered
    PRO_IR_CHECK(ExternalError::INVALID_OPERATION, registry_.find(opName) == registry_.end())
        << "Operator '" + opName + "' is already registered";

    // Create and insert the entry into the registry
    auto result = registry_.emplace(opName, OpRegistryEntry());
    auto& entry = result.first->second;
    entry.set_name(opName);

    // Create the operator instance with the operator name
    entry.op_ = std::make_shared<Op>(opName);

    return entry;
}

// ============================================================================
// OpRegistry Implementation
// ============================================================================

CallPtr OpRegistry::Create(const std::string& opName, const std::vector<ExprPtr>& args, Span span) const
{
    // Call new version with empty kwargs for backward compatibility
    return Create(opName, args, {}, std::move(span));
}

CallPtr OpRegistry::Create(const std::string& opName, const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs, Span span) const
{
    // Publish the DSL location for the whole of Create(): ValidateKwargs and the
    // deduce callbacks take no span, so their errors rely on the ambient one.
    const pro::SpanScope spanScope(span);

    // Look up operator in registry
    auto it = registry_.find(opName);
    PRO_IR_CHECK(ExternalError::KEY_ERROR, it != registry_.end()) << "Operator '" + opName + "' not found in registry";

    const auto& entry = it->second;
    const auto normalized = NormalizeKwargs(kwargs);

    // Validate kwargs against allowed attributes (stored in Op)
    if (!normalized.empty()) {
        OpPtr op = entry.GetOp();
        const auto& allowedKwargs = op->GetAttrs();
        if (!allowedKwargs.empty()) {
            ValidateKwargs(normalized, allowedKwargs, opName);
        }
    }

    const auto& deduceTypeFn = entry.GetDeduceType();

    // Deduce result type (pass args and kwargs separately)
    TypePtr resultType = deduceTypeFn(args, normalized);
    PRO_IR_INTERNAL_CHECK(npu::tile_fwk::InternalError::COMMON_INNER_ERROR, resultType)
        << "Type deduction failed for '" + opName + "'";

    // Create Call with deduced type
    return std::make_shared<Call>(opName, args, normalized, resultType, std::move(span));
}

const OpRegistryEntry& OpRegistry::GetEntry(const std::string& opName) const
{
    auto it = registry_.find(opName);
    PRO_IR_CHECK(ExternalError::KEY_ERROR, it != registry_.end()) << "Operator '" + opName + "' not found in registry";
    return it->second;
}

OpPtr OpRegistry::GetOp(const std::string& opName) const
{
    auto it = registry_.find(opName);
    PRO_IR_CHECK(ExternalError::KEY_ERROR, it != registry_.end()) << "Operator '" + opName + "' not found in registry";
    return it->second.GetOp();
}

} // namespace ir
} // namespace pypto
