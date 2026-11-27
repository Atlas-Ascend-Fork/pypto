/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 * You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file CostExprEval.h
 * \brief Single public entry (EvaluateCostExpr) for cost-expression evaluation.
 *
 * The engine only knows expressions, x1..xN coefficients and feature contexts; rules
 * and coefficients live in the per-arch coefficient tables and are passed through the
 * interface by callers.  Post-scaling of scheduled results also stays with the callers.
 */

#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cost_model/simulation/expr/CostExpr.h"

namespace CostModel {

namespace detail {

struct CompiledExpr {
    ExprPtr expr;
    std::set<std::string> keywords; // collected once at parse time
};

inline bool IsCoefficientName(const std::string& name)
{
    if (name.size() < 2 || name[0] != 'x') {
        return false;
    }
    for (size_t i = 1; i < name.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(name[i]))) {
            return false;
        }
    }
    return true;
}

// Parse failures are reported through `error` and not cached.
inline const CompiledExpr* GetCompiled(const std::string& text, std::string& error)
{
    static std::mutex mutex;
    static std::unordered_map<std::string, CompiledExpr> cache;
    std::lock_guard<std::mutex> lock(mutex);
    auto iter = cache.find(text);
    if (iter == cache.end()) {
        std::string parseError;
        ExprPtr expr = ExprParser::Parse(text, parseError);
        if (expr == nullptr) {
            error = parseError;
            return nullptr;
        }
        CompiledExpr compiled;
        compiled.expr = expr;
        expr->CollectKeywords(compiled.keywords);
        iter = cache.emplace(text, std::move(compiled)).first;
    }
    return &iter->second;
}

} // namespace detail

//! `value`/`cycles` are valid only when `ok`.
struct EvalResult {
    bool ok = false;
    double value = 0.0;
    uint64_t cycles = 0;
    std::string error;
};

//! Evaluate `expr` with coefficients x1..xN and the features in `ctx` (taken by value:
//! the engine writes coefficient defaults into it).  Expressions round themselves
//! (floor/ceil); the engine only performs the final double -> uint64 cast.
inline EvalResult EvaluateCostExpr(const std::string& expr, const std::vector<double>& coeffs, FeatureContext ctx)
{
    EvalResult result;
    std::string error;
    const detail::CompiledExpr* compiled = detail::GetCompiled(expr, error);
    if (compiled == nullptr) {
        result.error = error;
        return result;
    }
    for (size_t i = 0; i < coeffs.size(); ++i) {
        ctx.Set("x" + std::to_string(i + 1), coeffs[i]);
    }
    for (const auto& keyword : compiled->keywords) {
        if (!ctx.Has(keyword)) {
            result.error = (detail::IsCoefficientName(keyword) ? "missing coefficient '" : "missing feature '") +
                           keyword + "'";
            return result;
        }
    }
    const double value = compiled->expr->Eval(ctx);
    if (!std::isfinite(value) || value >= static_cast<double>(std::numeric_limits<uint64_t>::max())) {
        result.error = "expression result is not a finite uint64 value";
        return result;
    }
    // Negative net values are fit artifacts; clamp them to zero.
    result.value = std::max(value, 0.0);
    result.cycles = static_cast<uint64_t>(result.value);
    result.ok = true;
    return result;
}

} // namespace CostModel
