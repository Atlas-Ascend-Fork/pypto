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
 * \file test_dist_ops.cpp
 * \brief Coverage tests for dist_ops.cpp type deduction (channel.is_shared)
 */

#include "gtest/gtest.h"

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "core/error.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/op_registry.h"
#include "ir/type.h"
#include "test_op_helpers.h"
#include "tilefwk/error.h"

namespace pypto {
namespace ir {

using namespace test_helpers;

class DistChannelOpsTest : public testing::Test {};

TEST_F(DistChannelOpsTest, ChannelIsShared_ReturnsBoolScalar)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("channel.is_shared", {MakeScalarVar("ch", DataType::INT64)}, Sp());
    ASSERT_NE(call, nullptr);
    auto result_type = As<ScalarType>(call->GetType());
    ASSERT_NE(result_type, nullptr);
    EXPECT_EQ(result_type->dtype_, DataType::BOOL);
}

TEST_F(DistChannelOpsTest, ChannelIsShared_RejectsInvalidArguments)
{
    auto& reg = OpRegistry::GetInstance();
    auto channel = MakeScalarVar("ch", DataType::INT64);

    EXPECT_THROW((void)reg.Create("channel.is_shared", {}, Sp()), npu::tile_fwk::Error);
    EXPECT_THROW((void)reg.Create("channel.is_shared", {MakeScalarVar("ch", DataType::FP32)}, Sp()),
                 npu::tile_fwk::Error);

    std::vector<std::pair<std::string, std::any>> kwargs = {{"unexpected", 1}};
    EXPECT_THROW((void)reg.Create("channel.is_shared", {channel}, kwargs, Sp()), npu::tile_fwk::Error);
}

} // namespace ir
} // namespace pypto
