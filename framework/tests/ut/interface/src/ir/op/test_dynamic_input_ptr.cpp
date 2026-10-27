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
 * \file test_dynamic_input_ptr.cpp
 * \brief Coverage tests for the block_ops type deduction of dynamic_input_ptr
 */

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <vector>

#include "core/dtype.h"
#include "core/error.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/op_attr_types.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"
#include "test_op_helpers.h"
#include "tilefwk/error.h"

namespace pypto {
namespace ir {

using namespace test_helpers;

class DynamicInputPtrTest : public testing::Test {};

// dynamic_input_ptr(x, i) -> PtrType(dtype of x).
TEST_F(DynamicInputPtrTest, DefaultDtypeFollowsDescriptor)
{
    auto& reg = OpRegistry::GetInstance();
    auto x = MakePtrVar("x", DataType::FP16);
    auto i = MakeScalarVar("i", DataType::INT64);

    auto call = reg.Create("dynamic_input_ptr", {x, i}, {}, Sp());
    ASSERT_NE(call, nullptr);
    auto rt = As<PtrType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

// Exactly two arguments are required.
TEST_F(DynamicInputPtrTest, WrongArgCountThrows)
{
    auto& reg = OpRegistry::GetInstance();
    auto x = MakePtrVar("x", DataType::UINT8);
    EXPECT_THROW((void)reg.Create("dynamic_input_ptr", {x}, {}, Sp()), npu::tile_fwk::Error);
}

// The returned pointer's dtype follows the first argument's element type, so that argument must
// be a pointer.
TEST_F(DynamicInputPtrTest, NonPtrTypeThrows)
{
    auto& reg = OpRegistry::GetInstance();
    auto x = MakeScalarVar("x", DataType::INT64);
    auto i = MakeScalarVar("i", DataType::INT64);
    EXPECT_THROW((void)reg.Create("dynamic_input_ptr", {x, i}, {}, Sp()), npu::tile_fwk::Error);
}

} // namespace ir
} // namespace pypto
