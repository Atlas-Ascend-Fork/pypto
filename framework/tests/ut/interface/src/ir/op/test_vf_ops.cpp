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
 * \file test_vf_ops.cpp
 * \brief Coverage tests for vf_ops.cpp type deduction
 */

#include "gtest/gtest.h"

#include <any>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"
#include "test_op_helpers.h"

namespace pypto {
namespace ir {

using namespace test_helpers;

class VFOpsTest : public testing::Test {};

// ============================================================================
// DeduceVFMaskType: vf.create_mask
// ============================================================================

TEST_F(VFOpsTest, CreateMask_NoArgs_ReturnsUINT16Scalar)
{
    auto& reg = OpRegistry::GetInstance();
    auto call = reg.Create("vf.create_mask", {}, Sp());
    auto rt = As<RegType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->regtype_, RegKind::PredReg);
    EXPECT_EQ(rt->dtype_, DataType::UINT16);
}

// ============================================================================
// DeduceVFFromDstArg: vf.full
// ============================================================================

TEST_F(VFOpsTest, Full_WithArgs_ReturnsRegTensorOfSrcDtype)
{
    auto& reg = OpRegistry::GetInstance();
    auto scalar = MakeScalarVar("scalar", DataType::FP32);
    auto mask = MakeScalarVar("mask", DataType::UINT16);
    auto call = reg.Create("vf.full", {scalar, mask}, Sp());
    auto rt = As<RegType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->regtype_, RegKind::VecReg);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

// ============================================================================
// DeduceVFFromDstArg: vf.muls, vf.add
// ============================================================================

TEST_F(VFOpsTest, Muls_WithArgs_ReturnsFirstSrcType)
{
    auto& reg = OpRegistry::GetInstance();
    auto src = MakeScalarVar("src", DataType::FP16);
    auto scalar = MakeScalarVar("scalar", DataType::FP16);
    auto mask = MakeScalarVar("mask", DataType::UINT16);
    auto call = reg.Create("vf.muls", {src, scalar, mask}, Sp());
    auto rt = As<RegType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->regtype_, RegKind::VecReg);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

TEST_F(VFOpsTest, Add_WithArgs_ReturnsFirstSrcType)
{
    auto& reg = OpRegistry::GetInstance();
    auto src0 = MakeScalarVar("src0", DataType::FP16);
    auto src1 = MakeScalarVar("src1", DataType::FP16);
    auto mask = MakeScalarVar("mask", DataType::UINT16);
    auto call = reg.Create("vf.add", {src0, src1, mask}, Sp());
    auto rt = As<RegType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->regtype_, RegKind::VecReg);
    EXPECT_EQ(rt->dtype_, DataType::FP16);
}

// ============================================================================
// DeduceVFTypeChangingRegType: vf.bit_cast
// ============================================================================

TEST_F(VFOpsTest, BitCast_WithDtypeKwarg_ReturnsTargetScalarType)
{
    auto& reg = OpRegistry::GetInstance();
    auto src = MakeScalarVar("src", DataType::FP8E4M3FN);
    auto call = reg.Create("vf.bit_cast", {src}, {{"dtype", DataType::FP32}}, Sp());
    auto rt = As<RegType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->regtype_, RegKind::VecReg);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

TEST_F(VFOpsTest, BitCast_NoDtypeKwarg_IsRejected)
{
    auto& reg = OpRegistry::GetInstance();
    auto src = MakeScalarVar("src", DataType::UINT16);
    // Type-changing ops require an explicit dtype kwarg: the dst type differs
    // from the src, so no default can be inferred.
    EXPECT_THROW((void)reg.Create("vf.bit_cast", {src}, Sp()), pypto::ir::TypeError);
}

TEST_F(VFOpsTest, BitCast_MaskSrcNoDtypeKwarg_IsRejected)
{
    auto& reg = OpRegistry::GetInstance();
    auto src = std::make_shared<Var>("mask", std::make_shared<RegType>(RegKind::PredReg, DataType::UINT16), Sp());
    // A MaskReg source does not provide a dst dtype either: the mask dtype is a
    // placeholder, not a type to keep.
    EXPECT_THROW((void)reg.Create("vf.bit_cast", {src}, Sp()), pypto::ir::TypeError);
}

// ============================================================================
// DeduceVFArangeType: vf.arange
// ============================================================================

TEST_F(VFOpsTest, Arange_WithDtypeKwarg_ReturnsRegTensorOfDtype)
{
    auto& reg = OpRegistry::GetInstance();
    auto start = MakeScalarVar("start", DataType::INT64);
    auto call = reg.Create("vf.arange", {start}, {{"dtype", DataType::FP32}}, Sp());
    auto rt = As<RegType>(call->GetType());
    ASSERT_NE(rt, nullptr);
    EXPECT_EQ(rt->regtype_, RegKind::VecReg);
    EXPECT_EQ(rt->dtype_, DataType::FP32);
}

TEST_F(VFOpsTest, Arange_NoDtypeKwarg_IsRejected)
{
    auto& reg = OpRegistry::GetInstance();
    auto start = MakeScalarVar("start", DataType::INT64);
    // The start literal's dtype cannot decide the sequence element type.
    EXPECT_THROW((void)reg.Create("vf.arange", {start}, Sp()), pypto::ir::TypeError);
}

} // namespace ir
} // namespace pypto
