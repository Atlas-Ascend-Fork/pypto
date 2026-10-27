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
 * \file test_backend_cce_dynamic_input_dim.cpp
 * \brief Backend CCE codegen test for dynamic_input_dim
 */

#include "gtest/gtest.h"

#include <any>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "backend/backend_cce.h"
#include "codegen/cce/cce_codegen.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/memref.h"
#include "ir/op_attr_types.h"
#include "ir/pipe.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/type.h"

namespace pypto {
namespace backend {
namespace {

class TestableCCECodegen : public codegen::CCECodegen {
public:
    using codegen::CCECodegen::CCECodegen;
    void RegisterVarName(const ir::VarPtr& var, const std::string& name) { context_.RegisterVar(var, name); }
};

ir::VarPtr MakeVar(const std::string& name, const ir::TypePtr& type)
{
    return std::make_shared<const ir::Var>(name, type, ir::Span::Unknown());
}

} // namespace

// dynamic_input_dim lowers to a raw shape-word read (dim != 0 layout), guarded by 0 <= axis < dim.
TEST(BackendCceDynamicInputDimTest, EmitsWordReadWithTrapGuard)
{
    auto ptr_type = std::make_shared<const ir::PtrType>(ir::DataType::UINT8);
    auto i_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT64);
    auto result_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT64);
    auto x = MakeVar("x", ptr_type);
    auto i = MakeVar("i", i_type);
    auto axis = MakeVar("a", i_type);
    auto call = std::make_shared<const ir::Call>("dynamic_input_dim", std::vector<ir::ExprPtr>{x, i, axis}, result_type,
                                                 ir::Span::Unknown());

    const auto* info = BackendCCE::Instance().GetOpInfo("dynamic_input_dim");
    ASSERT_NE(info, nullptr);

    TestableCCECodegen codegen(ir::SectionKind::Vector);
    codegen.RegisterVarName(x, "x");
    codegen.RegisterVarName(i, "i");
    codegen.RegisterVarName(axis, "a");
    auto expr = info->codegen_func(call, codegen);
    EXPECT_NE(expr.find("reinterpret_cast<__gm__ int64_t*>"), std::string::npos);
    EXPECT_NE(expr.find("0 <= "), std::string::npos);  // a negative axis is rejected too, not only axis >= dim
    EXPECT_NE(expr.find("trap()"), std::string::npos); // out-of-range axis traps, not silent garbage
}

TEST(BackendCceDynamicInputDimTest, RegisteredWithBackend)
{
    EXPECT_NE(BackendCCE::Instance().GetOpInfo("dynamic_input_dim"), nullptr);
}

} // namespace backend
} // namespace pypto
