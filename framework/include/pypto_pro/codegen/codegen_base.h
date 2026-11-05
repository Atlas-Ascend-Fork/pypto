/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PYPTO_CODEGEN_CODEGEN_BASE_H_
#define PYPTO_CODEGEN_CODEGEN_BASE_H_

#include <cstdint>
#include <string>

#include "core/dtype.h"
#include "core/error.h"
#include "ir/expr.h"
#include "pypto_pro/error.h"
#include "ir/transforms/base/visitor.h"
#include "ir/type.h"

namespace pypto {
namespace codegen {

/**
 * \brief Base class for platform code generators (CCE, PTO, etc.)
 *
 * Provides a common API used by operator codegen callbacks (f_codegen_cce, f_codegen_pto).
 * Subclasses implement platform-specific code generation while sharing this contract.
 * Does not define Generate() as each platform has different signature (e.g. map vs string).
 */
class CodegenBase : public ir::IRVisitor {
protected:
    using ir::IRVisitor::VisitExpr_;
    using ir::IRVisitor::VisitStmt_;

public:
    ~CodegenBase() override = default;

    // ---------------------------------------------------------------------------
    // Common API for operator codegen callbacks (unified names and semantics)
    // ---------------------------------------------------------------------------

    /**
     * \brief Get the current result target for the active Call
     *
     * Where the result of the current operation should be written (C++ variable name
     * or MLIR SSA buffer name).
     *
     * \return Current result target name
     */
    [[nodiscard]] virtual std::string GetCurrentResultTarget() const = 0;

    /**
     * \brief Emit one line of platform code (C++ or MLIR)
     *
     * \param line Line of code to emit
     */
    virtual void Emit(const std::string& line) = 0;

    /**
     * \brief Get platform code for an expression
     *
     * Converts an IR expression to platform-usable code (C++ fragment or MLIR SSA name).
     *
     * \param expr Expression to convert
     * \return Platform code string for the expression
     */
    virtual std::string GetExprAsCode(const ir::ExprPtr& expr) = 0;

    /**
     * \brief Convert DataType to platform type string
     *
     * \param dtype Data type (e.g. FP32, INT32)
     * \return Platform type string (e.g. "float"/"f32", "int32_t"/"i32")
     */
    [[nodiscard]] virtual std::string GetTypeString(const ir::DataType& dtype) const = 0;

    /**
     * \brief Extract constant integer value from expression
     *
     * \param expr Expression (must be ConstInt)
     * \return Integer value
     */
    virtual int64_t GetConstIntValue(const ir::ExprPtr& expr) = 0;

    /**
     * \brief Get platform name for a Var
     *
     * \param var The IR Var
     * \return Platform variable name (C++ name or MLIR SSA name)
     */
    virtual std::string GetVarName(const ir::VarPtr& var) = 0;

protected:
    /**
     * \brief Throw when no codegen is registered for a Call
     *
     * Subclasses call this from VisitExpr_(Call) when the op has no platform codegen.
     *
     * \param opName IR operation name (e.g., "block.load")
     */
    [[noreturn]] void ThrowNoCodegenForCall(const std::string& opName) const
    {
        using npu::tile_fwk::ExternalError;
        PRO_CODEGEN_THROW(::pypto::ir::ValueError, ExternalError::KEY_ERROR)
            << "No codegen registered for operation: " << opName;
        // The ErrLogger temporary above throws from its destructor; the explicit
        // throw below is unreachable and only satisfies -Wreturn-type on
        // toolchains that do not honor [[noreturn]] destructors.
        throw ::pypto::ir::ValueError("No codegen registered for operation: " + opName);
    }

    /**
     * \brief Default VisitExpr_(Call): throws (subclasses must override)
     *
     * All actual codegen paths go through subclass overrides. This default ensures
     * any Call that reaches the base implementation results in a compile-time error.
     */
    void VisitExpr_(const ir::CallPtr& op) override { ThrowNoCodegenForCall(op->name_); }
};

/**
 * \brief Render a ConstInt payload as a C literal, honouring the dtype's signedness
 *
 * ConstInt stores an int64_t that doubles as the bit image of a uint64_t. Only a value above INT64_MAX
 * is stored folded, and only that one has to be spelled unsigned to stay in range; every other value
 * keeps the plain signed spelling so the surrounding integer promotions do not change.
 *
 * \param value Stored value
 * \param dtype Data type the value belongs to
 * \return C literal text
 */
std::string FormatIntCLiteral(int64_t value, const ir::DataType& dtype);

/**
 * \brief Render a double as a C literal that round-trips
 *
 * std::to_string is "%f": exactly six decimal places, so a small constant such as 1e-30 becomes
 * 0.000000 and compiles to the wrong value. This keeps a trailing ".0" on an integral value and
 * otherwise emits the shortest form that parses back to the same double.
 *
 * \param value Value to render
 * \return C literal text, without a type suffix
 */
std::string FormatFloatCLiteral(double value);

} // namespace codegen
} // namespace pypto

#endif // PYPTO_CODEGEN_CODEGEN_BASE_H_
