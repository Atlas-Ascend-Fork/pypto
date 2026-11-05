/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen/cce/code_context.h"

#include <cctype>
#include <cstddef>
#include <string>

#include "core/logging.h"
#include "tilefwk/error.h"
#include "ir/expr.h"
#include "pypto_pro/error.h"

namespace pypto {

namespace codegen {

std::string CodeContext::GetVarName(const ir::VarPtr& var)
{
    PRO_CODEGEN_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, var != nullptr)
        << "Cannot get name for null variable";
    auto it = nameToCpp_.find(var->name_);
    if (it != nameToCpp_.end()) {
        return it->second;
    }
    // Fallback: when a Var references a pre-SSA base name (e.g. `_buf_tile_0`)
    // that was versioned by the SSA pass to `<name>_0` but the reference site
    // was not updated, look up the first versioned form.
    auto v0 = nameToCpp_.find(var->name_ + "_0");
    if (v0 != nameToCpp_.end()) {
        return v0->second;
    }
    // Fallback: for variables first defined inside a loop body, the post-loop
    // reference may use the raw base name while SSA registered a versioned form
    // (e.g. 'reg_dst' vs 'reg_dst_1'). Search for the highest-versioned match.
    std::string bestMatch;
    int bestVersion = -1;
    std::string prefix = var->name_ + "_";
    for (const auto& [key, value] : nameToCpp_) {
        if (key.size() > prefix.size() && key.compare(0, prefix.size(), prefix) == 0) {
            const char* suffix = key.c_str() + prefix.size();
            bool allDigits = true;
            for (const char* p = suffix; *p; ++p) {
                if (*p < '0' || *p > '9') {
                    allDigits = false;
                    break;
                }
            }
            if (allDigits && *suffix != '\0') {
                int ver = std::atoi(suffix);
                if (ver > bestVersion) {
                    bestVersion = ver;
                    bestMatch = value;
                }
            }
        }
    }
    if (!bestMatch.empty()) {
        nameToCpp_[var->name_] = bestMatch;
        return bestMatch;
    }
    // Auto-register: variable may originate from a different section (Cube/Vector)
    // that was cleared on section boundary. Use the sanitized IR name.
    std::string cppName = SanitizeName(var);
    nameToCpp_[var->name_] = cppName;
    autoRegistered_.insert(cppName);
    return cppName;
}

void CodeContext::RegisterVar(const ir::VarPtr& var, const std::string& cppName)
{
    PRO_CODEGEN_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, var != nullptr)
        << "Cannot register null variable";
    PRO_CODEGEN_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, !cppName.empty())
        << "Cannot register variable with empty name";

    // Check if this name is already registered (suppress for array access and alias optimizations)
    auto it = nameToCpp_.find(var->name_);
    if (it != nameToCpp_.end() && it->second != cppName && cppName.find('[') == std::string::npos) {
        IR_LOGW() << "Variable " << var->name_ << " re-registered with different C++ name: " << cppName << " vs "
                  << it->second;
    }

    // Register name-based mapping
    nameToCpp_[var->name_] = cppName;
}

void CodeContext::Clear() { nameToCpp_.clear(); }

std::string CodeContext::SanitizeName(const ir::VarPtr& var) const
{
    PRO_CODEGEN_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, var != nullptr)
        << "Cannot sanitize null variable";
    auto irName = var->name_;
    if (irName.empty()) {
        return "var";
    }

    std::string result;
    result.reserve(irName.size());

    // First character must be letter or underscore
    if (std::isalpha(static_cast<unsigned char>(irName[0])) || irName[0] == '_') {
        result += irName[0];
    } else {
        result += '_';
    }

    // Subsequent characters can be alphanumeric or underscore
    for (size_t i = 1; i < irName.size(); ++i) {
        char c = irName[i];
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
            result += c;
        } else {
            result += '_';
        }
    }

    return result;
}

} // namespace codegen

} // namespace pypto
