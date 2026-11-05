/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PYPTO_CODEGEN_CCE_CODE_CONTEXT_H_
#define PYPTO_CODEGEN_CCE_CODE_CONTEXT_H_

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "ir/expr.h"

namespace pypto {
namespace codegen {

/**
 * \brief Context for tracking IR variable to C++ variable name mapping
 *
 * CodeContext provides name mapping from IR variables to valid C++ identifiers,
 * handling name sanitization and caching for consistent code generation.
 */
class CodeContext {
public:
    CodeContext() = default;

    /**
     * \brief Sanitize an IR variable name for use in C++
     *
     * \param var The IR variable
     * \return A valid C++ identifier
     */
    [[nodiscard]] std::string SanitizeName(const ir::VarPtr& var) const;

    /**
     * \brief Get the C++ variable name for an IR variable
     *
     * If the variable has been seen before, returns the previously assigned name.
     * Otherwise, generates a new name based on the IR variable's name.
     *
     * \param var The IR variable
     * \return The C++ variable name to use
     */
    std::string GetVarName(const ir::VarPtr& var);

    /**
     * \brief Register a variable with a specific C++ name
     *
     * Overrides any previously registered or generated name for this variable.
     * Used to establish naming conventions (e.g., the GlobalTensor instance is named
     * after the tensor variable itself).
     *
     * \param var The IR variable
     * \param cppName The C++ name to associate with this variable
     */
    void RegisterVar(const ir::VarPtr& var, const std::string& cppName);

    /**
     * \brief Clear all state
     */
    void Clear();

    /**
     * \brief True when *name* is a known variable C++ name
     *
     * An expression that evaluates to a known variable name is a plain
     * reference (alias), not a value that must be materialized into a fresh
     * local declaration.
     */
    [[nodiscard]] bool HasVarName(const std::string& name) const
    {
        return autoRegistered_.count(name) > 0 || nameToCpp_.count(name) > 0;
    }

private:
    std::unordered_map<std::string, std::string> nameToCpp_; ///< Mapping from IR var name to C++ name
    std::unordered_set<std::string> autoRegistered_; ///< Variables auto-registered from cross-section SSA references
};

} // namespace codegen
} // namespace pypto

#endif // PYPTO_CODEGEN_CCE_CODE_CONTEXT_H_
