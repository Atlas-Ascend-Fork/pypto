/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ir/expr.h"

#include <memory>
#include <utility>
#include <vector>

#include "core/logging.h"
#include "ir/kind_traits.h"
#include "ir/scalar_expr.h"
#include "ir/span.h"
#include "ir/transforms/structural_comparison.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

MakeTuple::MakeTuple(std::vector<ExprPtr> elements, Span span) : Expr(std::move(span)), elements_(std::move(elements))
{
    std::vector<TypePtr> elementTypes;
    elementTypes.reserve(elements_.size());
    for (const auto& elem : elements_) {
        elementTypes.push_back(elem->GetType());
    }
    // Positional TupleType. Field names of named tuples / structs are carried by the
    // parser-populated IRDebugInfo side table, not on the expression or the type.
    type_ = std::make_shared<TupleType>(std::move(elementTypes));
}

GetItemExpr::GetItemExpr(ExprPtr value, ExprPtr slice, Span span)
    : Expr(std::move(span)), value_(std::move(value)), slice_(std::move(slice))
{
    CHECK(value_) << "GetItemExpr requires a non-null value";
    CHECK(slice_) << "GetItemExpr requires a non-null slice";

    auto valueType = value_->GetType();

    auto tupleType = As<TupleType>(valueType);
    CHECK(tupleType) << "GetItemExpr requires value to have TupleType, got " << valueType->TypeName();

    if (auto constIdx = As<ConstInt>(slice_)) {
        int index = static_cast<int>(constIdx->value_);
        CHECK(index >= 0 && index < static_cast<int>(tupleType->types_.size()))
            << "GetItemExpr index " << index << " out of bounds for tuple with " << tupleType->types_.size()
            << " elements";
        type_ = tupleType->types_[index];
    } else {
        CHECK(!tupleType->types_.empty()) << "GetItemExpr: cannot index into an empty tuple";
        const auto& first = tupleType->types_[0];
        for (size_t i = 1; i < tupleType->types_.size(); ++i) {
            CHECK(structural_equal(first, tupleType->types_[i]))
                << "GetItemExpr with dynamic index requires all tuple elements to have the same type, "
                << "but element 0 and element " << i << " differ";
        }
        type_ = first;
    }
}

} // namespace ir
} // namespace pypto
