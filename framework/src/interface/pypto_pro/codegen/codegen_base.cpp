/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen/codegen_base.h"

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

#include "core/dtype.h"
#include "core/error.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/scalar_expr.h"
#include "ir/scalar_expr_ops.h"
#include "ir/type.h"

namespace pypto {
namespace codegen {

using namespace pypto::ir; // NOLINT(build/namespaces)

std::string FormatIntCLiteral(int64_t value, const ir::DataType& dtype)
{
    if (dtype.IsUnsignedInt() && value < 0) {
        // Folded uint64 payload: the plain signed spelling would be out of range for long long.
        return std::to_string(static_cast<uint64_t>(value)) + "uLL";
    }
    if (value == std::numeric_limits<int64_t>::min()) {
        // The positive magnitude of INT64_MIN is not a signed C++ integer literal.
        return "(-9223372036854775807LL - 1LL)";
    }
    return std::to_string(value);
}

namespace {
// Above this magnitude every double is integral anyway, and the fixed form would spell out hundreds of
// digits; the scientific form round-trips just as exactly and stays readable.
constexpr double kMaxFixedMagnitude = 1e16;

// Exact comparison of two finite doubles. -Wfloat-equal forbids the == operator; islessgreater is the
// same test without the warning.
bool ExactlyEqual(double lhs, double rhs) { return !std::islessgreater(lhs, rhs); }
} // namespace

std::string FormatFloatCLiteral(double value)
{
    std::ostringstream oss;
    if (!std::isfinite(value)) {
        oss << value;
        return oss.str();
    }
    if (ExactlyEqual(value, std::floor(value)) && std::fabs(value) < kMaxFixedMagnitude) {
        // An integral value keeps a ".0" so the literal still reads as a float.
        oss << std::fixed << std::setprecision(1) << value;
        return oss.str();
    }
    // Shortest form that still parses back to the same double, so the literal round-trips without
    // spelling out 17 digits for a value that needs far fewer.
    for (int precision = std::numeric_limits<double>::digits10; precision < std::numeric_limits<double>::max_digits10;
         ++precision) {
        std::ostringstream candidate;
        candidate << std::setprecision(precision) << value;
        double parsed = 0.0;
        std::istringstream(candidate.str()) >> parsed;
        if (ExactlyEqual(parsed, value)) {
            return candidate.str();
        }
    }
    oss << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
    return oss.str();
}

} // namespace codegen
} // namespace pypto
