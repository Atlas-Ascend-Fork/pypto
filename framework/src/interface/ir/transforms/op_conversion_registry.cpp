/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ir/transforms/op_conversion_registry.h"

#include <any>
#include <string>
#include <utility>
#include <vector>

#include "ir/expr.h"
#include "ir/op_registry.h"
#include "ir/span.h"

namespace pypto {
namespace ir {

OpConversionRegistry& OpConversionRegistry::GetInstance()
{
    static OpConversionRegistry instance;
    return instance;
}

// No default conversions: the tensor.* operators these used to lower from were
// removed (they had no backend implementation, so a kernel using one only failed
// once it reached codegen). Conversions are registered by whoever needs them,
// through RegisterSimple / RegisterCustom or their `register_op_conversion`
// bindings.
OpConversionRegistry::OpConversionRegistry() = default;

void OpConversionRegistry::RegisterSimple(const std::string& fromOp, const std::string& toOp)
{
    // Capture to_op by value for the lambda
    conversions_[fromOp] = [toOp](const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs,
                                  const Span& span) -> ConversionResult {
        auto& reg = OpRegistry::GetInstance();
        CallPtr call;
        if (kwargs.empty()) {
            call = reg.Create(toOp, args, span);
        } else {
            call = reg.Create(toOp, args, kwargs, span);
        }
        return ConversionResult{call};
    };
}

void OpConversionRegistry::RegisterCustom(const std::string& fromOp, ConversionFunc func)
{
    conversions_[fromOp] = std::move(func);
}

const ConversionFunc* OpConversionRegistry::Lookup(const std::string& opName) const
{
    auto it = conversions_.find(opName);
    if (it == conversions_.end()) {
        return nullptr;
    }
    return &it->second;
}

bool OpConversionRegistry::HasConversion(const std::string& opName) const { return conversions_.count(opName) > 0; }

} // namespace ir
} // namespace pypto
