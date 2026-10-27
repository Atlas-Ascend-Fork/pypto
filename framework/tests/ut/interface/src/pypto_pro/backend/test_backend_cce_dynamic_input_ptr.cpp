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
 * \file test_backend_cce_dynamic_input_ptr.cpp
 * \brief Backend CCE codegen test for dynamic_input_ptr
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

// dynamic_input_ptr lowers to a direct descriptor read (no AscendC::ListTensorDesc, hence no extra
// include): ((__gm__ T*)(*(__gm__ uint64_t*)((__gm__ uint8_t*)x + word0) + i)).
TEST(BackendCceDynamicInputPtrTest, EmitsDirectDescriptorRead)
{
    auto x_type = std::make_shared<const ir::PtrType>(ir::DataType::UINT8);
    auto i_type = std::make_shared<const ir::ScalarType>(ir::DataType::INT64);
    auto result_type = std::make_shared<const ir::PtrType>(ir::DataType::FP16);
    auto x = MakeVar("x", x_type);
    auto i = MakeVar("i", i_type);
    auto call = std::make_shared<const ir::Call>("dynamic_input_ptr", std::vector<ir::ExprPtr>{x, i}, result_type,
                                                 ir::Span::Unknown());

    const auto* info = BackendCCE::Instance().GetOpInfo("dynamic_input_ptr");
    ASSERT_NE(info, nullptr);

    TestableCCECodegen codegen(ir::SectionKind::Vector);
    codegen.RegisterVarName(x, "x");
    codegen.RegisterVarName(i, "i");
    auto expr = info->codegen_func(call, codegen);
    // Reads the descriptor head word (byte offset) and indexes the address array with the element type.
    EXPECT_EQ(expr.find("ListTensorDesc"), std::string::npos);
    EXPECT_NE(expr.find("reinterpret_cast<__gm__ uint64_t*>"), std::string::npos);
    EXPECT_NE(expr.find("reinterpret_cast<__gm__ uint8_t*>(x)"), std::string::npos);
    EXPECT_NE(expr.find("reinterpret_cast<__gm__ half*>("), std::string::npos);
}

// The pointer's element type follows the descriptor dtype for every element width
// (64/32/16/8-bit) and for the packed 4-bit types. 4-bit elements hold two values per
// byte, so their C type is float4_e2m1x2_t; torch has no 4-bit dtype, which is why that
// width is checked here rather than in an on-board ST.
TEST(BackendCceDynamicInputPtrTest, EmitsElementTypePerWidth)
{
    struct Case {
        ir::DataType dt;
        const char* ctype;
    };
    const Case cases[] = {
        {ir::DataType::INT64, "reinterpret_cast<__gm__ int64_t*>("},
        {ir::DataType::INT32, "reinterpret_cast<__gm__ int32_t*>("},
        {ir::DataType::FP32, "reinterpret_cast<__gm__ float*>("},
        {ir::DataType::FP16, "reinterpret_cast<__gm__ half*>("},
        {ir::DataType::BF16, "reinterpret_cast<__gm__ bfloat16_t*>("},
        {ir::DataType::INT8, "reinterpret_cast<__gm__ int8_t*>("},
        {ir::DataType::UINT8, "reinterpret_cast<__gm__ uint8_t*>("},
        {ir::DataType::FP4E2M1, "reinterpret_cast<__gm__ float4_e2m1x2_t*>("},
    };
    const auto* info = BackendCCE::Instance().GetOpInfo("dynamic_input_ptr");
    ASSERT_NE(info, nullptr);

    for (const auto& c : cases) {
        auto x = MakeVar("x", std::make_shared<const ir::PtrType>(ir::DataType::UINT8));
        auto i = MakeVar("i", std::make_shared<const ir::ScalarType>(ir::DataType::INT64));
        auto call = std::make_shared<const ir::Call>("dynamic_input_ptr", std::vector<ir::ExprPtr>{x, i},
                                                     std::make_shared<const ir::PtrType>(c.dt), ir::Span::Unknown());
        TestableCCECodegen codegen(ir::SectionKind::Vector);
        codegen.RegisterVarName(x, "x");
        codegen.RegisterVarName(i, "i");
        auto expr = info->codegen_func(call, codegen);
        EXPECT_NE(expr.find(c.ctype), std::string::npos) << "missing " << c.ctype;
    }
}

// A descriptor dtype without a packed C type has nothing to emit; reject it loudly instead of
// generating `__gm__ unknown*`.
TEST(BackendCceDynamicInputPtrTest, UnmappableDescriptorDtypeThrows)
{
    const auto* info = BackendCCE::Instance().GetOpInfo("dynamic_input_ptr");
    ASSERT_NE(info, nullptr);

    for (const auto dt : {ir::DataType::INT4, ir::DataType::UINT4, ir::DataType::FP4, ir::DataType::HF4}) {
        auto x = MakeVar("x", std::make_shared<const ir::PtrType>(ir::DataType::UINT8));
        auto i = MakeVar("i", std::make_shared<const ir::ScalarType>(ir::DataType::INT64));
        auto call = std::make_shared<const ir::Call>("dynamic_input_ptr", std::vector<ir::ExprPtr>{x, i},
                                                     std::make_shared<const ir::PtrType>(dt), ir::Span::Unknown());
        TestableCCECodegen codegen(ir::SectionKind::Vector);
        codegen.RegisterVarName(x, "x");
        codegen.RegisterVarName(i, "i");
        EXPECT_THROW((void)info->codegen_func(call, codegen), npu::tile_fwk::Error) << "dtype " << dt.ToString();
    }
}

// The op is registered with the CCE backend.
TEST(BackendCceDynamicInputPtrTest, RegisteredWithBackend)
{
    EXPECT_NE(BackendCCE::Instance().GetOpInfo("dynamic_input_ptr"), nullptr);
}

} // namespace backend
} // namespace pypto
