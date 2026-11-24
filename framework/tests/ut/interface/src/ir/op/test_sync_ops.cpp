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
 * \file test_sync_ops.cpp
 * \brief Coverage tests for sync_ops/sync.cpp type deduction
 */

#include "gtest/gtest.h"

#include <any>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "core/error.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/op_attr_types.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"
#include "ir/transforms/core_placement.h"
#include "test_op_helpers.h"
#include "tilefwk/error.h"

namespace pypto {
namespace ir {

using namespace test_helpers;

// ============================================================================
// system.mutex_lock_dyn / system.mutex_unlock_dyn
// ============================================================================

TEST(SyncOpsMutexTest, DynamicMutexRejectsPerTileGroups)
{
    auto& reg = OpRegistry::GetInstance();
    auto id0 = MakeScalarVar("id0", DataType::INDEX);
    auto id1 = MakeScalarVar("id1", DataType::INDEX);
    auto firstIds = std::make_shared<const MakeTuple>(std::vector<ExprPtr>{id0}, Sp());
    auto secondIds = std::make_shared<const MakeTuple>(std::vector<ExprPtr>{id1}, Sp());
    auto mutexId = std::make_shared<const MakeTuple>(std::vector<ExprPtr>{firstIds, secondIds}, Sp());
    auto firstCandidates = MakeIntTuple({0});
    auto secondCandidates = MakeIntTuple({1});
    auto candidateGroups = std::make_shared<const MakeTuple>(std::vector<ExprPtr>{firstCandidates, secondCandidates},
                                                             Sp());
    std::vector<std::pair<std::string, std::any>> kwargs = {{"pipe", 5}};

    for (const char* op_name : {"system.mutex_lock_dyn", "system.mutex_unlock_dyn"}) {
        EXPECT_THROW((void)reg.Create(op_name, {mutexId, candidateGroups}, kwargs, Sp()), npu::tile_fwk::Error);
    }
}

TEST(SyncOpsMutexTest, DynamicMutexAcceptsManualScalarId)
{
    auto& reg = OpRegistry::GetInstance();
    auto id0 = MakeScalarVar("id0", DataType::INDEX);
    std::vector<std::pair<std::string, std::any>> kwargs = {{"pipe", 5}};

    for (const char* op_name : {"system.mutex_lock_dyn", "system.mutex_unlock_dyn"}) {
        auto call = reg.Create(op_name, {id0}, kwargs, Sp());
        ASSERT_NE(call, nullptr);
        ASSERT_EQ(call->args_.size(), 1);
        EXPECT_EQ(call->args_[0], id0);
    }
}

TEST(SyncOpsMutexTest, DynamicMutexRejectsEmptyCandidateGroups)
{
    auto& reg = OpRegistry::GetInstance();
    auto idGroup = MakeIntTuple({0});
    auto mutexId = std::make_shared<const MakeTuple>(std::vector<ExprPtr>{idGroup}, Sp());
    auto candidateGroups = std::make_shared<const MakeTuple>(std::vector<ExprPtr>{}, Sp());
    std::vector<std::pair<std::string, std::any>> kwargs = {{"pipe", 5}};

    EXPECT_THROW((void)reg.Create("system.mutex_lock_dyn", {mutexId, candidateGroups}, kwargs, Sp()),
                 npu::tile_fwk::Error);
}

TEST(SyncOpsMutexTest, DynamicMutexRejectsInvalidArgumentCounts)
{
    auto& reg = OpRegistry::GetInstance();
    auto id = MakeScalarVar("id", DataType::INDEX);
    std::vector<std::pair<std::string, std::any>> kwargs = {{"pipe", 5}};

    EXPECT_THROW((void)reg.Create("system.mutex_lock_dyn", {}, kwargs, Sp()), npu::tile_fwk::Error);
    EXPECT_THROW((void)reg.Create("system.mutex_lock_dyn", {id, id, id}, kwargs, Sp()), npu::tile_fwk::Error);
}

TEST(SyncOpsMutexTest, DynamicMutexRejectsMisalignedCandidateGroups)
{
    auto& reg = OpRegistry::GetInstance();
    auto firstIds = MakeIntTuple({0});
    auto secondIds = MakeIntTuple({1});
    auto mutexId = std::make_shared<const MakeTuple>(std::vector<ExprPtr>{firstIds, secondIds}, Sp());
    auto candidates = MakeIntTuple({0});
    auto candidateGroups = std::make_shared<const MakeTuple>(std::vector<ExprPtr>{candidates}, Sp());
    std::vector<std::pair<std::string, std::any>> kwargs = {{"pipe", 5}};

    EXPECT_THROW((void)reg.Create("system.mutex_lock_dyn", {mutexId, candidateGroups}, kwargs, Sp()),
                 npu::tile_fwk::Error);
}

// ============================================================================
// system.dcci
// ============================================================================

TEST(SyncOpsHF32Test, SwitchesReturnNoneAndRejectArguments)
{
    auto& reg = OpRegistry::GetInstance();
    auto mode = MakeScalarVar("mode", DataType::INT64);
    std::vector<std::pair<std::string, std::any>> kwargs = {{"mode", 0}};
    for (const char* name : {"system.enable_hf32", "system.disable_hf32"}) {
        auto call = reg.Create(name, {}, Sp());
        EXPECT_NE(As<NoneType>(call->GetType()), nullptr);
        EXPECT_THROW((void)reg.Create(name, {mode}, Sp()), npu::tile_fwk::Error);
        EXPECT_THROW((void)reg.Create(name, {}, kwargs, Sp()), npu::tile_fwk::Error);
    }
}

TEST(SyncOpsHF32Test, RoundingAcceptsBothConstantsAndRuntimeInteger)
{
    auto& reg = OpRegistry::GetInstance();
    for (int64_t value : {0, 1}) {
        auto mode = std::make_shared<const ConstInt>(value, DataType::INT64, Sp());
        auto call = reg.Create("system.set_hf32_round_mode", {mode}, Sp());
        EXPECT_NE(As<NoneType>(call->GetType()), nullptr);
    }
    auto mode = MakeScalarVar("mode", DataType::INT32);
    auto call = reg.Create("system.set_hf32_round_mode", {mode}, Sp());
    EXPECT_EQ(call->args_[0], mode);
}

TEST(SyncOpsHF32Test, RoundingRejectsInvalidCountsTypesAndEncodings)
{
    auto& reg = OpRegistry::GetInstance();
    auto mode = MakeScalarVar("mode", DataType::INT64);
    EXPECT_THROW((void)reg.Create("system.set_hf32_round_mode", {}, Sp()), npu::tile_fwk::Error);
    EXPECT_THROW((void)reg.Create("system.set_hf32_round_mode", {mode, mode}, Sp()), npu::tile_fwk::Error);
    std::vector<std::pair<std::string, std::any>> kwargs = {{"mode", 0}};
    EXPECT_THROW((void)reg.Create("system.set_hf32_round_mode", {mode}, kwargs, Sp()), npu::tile_fwk::Error);
    for (auto dtype : {DataType::FP32, DataType::BOOL}) {
        auto invalid = MakeScalarVar("invalid", dtype);
        EXPECT_THROW((void)reg.Create("system.set_hf32_round_mode", {invalid}, Sp()), npu::tile_fwk::Error);
    }
    for (int64_t value : {-1, 2}) {
        auto invalid = std::make_shared<const ConstInt>(value, DataType::INT64, Sp());
        EXPECT_THROW((void)reg.Create("system.set_hf32_round_mode", {invalid}, Sp()), npu::tile_fwk::Error);
    }
}

TEST(SyncOpsHF32Test, ConfigurationIsCubeOnly)
{
    auto& reg = OpRegistry::GetInstance();
    for (const char* name : {"system.enable_hf32", "system.disable_hf32", "system.set_hf32_round_mode"}) {
        std::vector<ExprPtr> args;
        if (std::string(name) == "system.set_hf32_round_mode") {
            args.push_back(MakeScalarVar("mode", DataType::INT32));
        }
        auto call = reg.Create(name, args, Sp());
        EXPECT_EQ(GetAllowedCores(call), CoreMask::Cube);
        EXPECT_NO_THROW(ValidateCorePlacement(call, GetAllowedCores(call), CoreMask::Cube));
    }
}

class SyncOpsDcciTest : public testing::Test {};

TEST(SyncOpsICacheTest, PreloadAcceptsIntegerScalar)
{
    auto& reg = OpRegistry::GetInstance();
    auto prefetchLen = MakeScalarVar("prefetch_len", DataType::INT64);
    auto call = reg.Create("system.icache_preload", {prefetchLen}, Sp());
    ASSERT_NE(call, nullptr);
    EXPECT_NE(As<NoneType>(call->GetType()), nullptr);
}

TEST(SyncOpsICacheTest, PreloadRejectsFloatingPointScalar)
{
    auto& reg = OpRegistry::GetInstance();
    auto prefetchLen = MakeScalarVar("prefetch_len", DataType::FP32);
    EXPECT_THROW((void)reg.Create("system.icache_preload", {prefetchLen}, Sp()), npu::tile_fwk::Error);
}

TEST(SyncOpsNopTest, NopAcceptsConstantCount)
{
    auto& reg = OpRegistry::GetInstance();
    auto count = std::make_shared<ConstInt>(4, DataType::INT64, Sp());
    auto call = reg.Create("system.nop", {count}, Sp());
    ASSERT_NE(call, nullptr);
    EXPECT_NE(As<NoneType>(call->GetType()), nullptr);
}

TEST(SyncOpsNopTest, NopRejectsInvalidArgumentCount)
{
    auto& reg = OpRegistry::GetInstance();
    auto count = std::make_shared<ConstInt>(4, DataType::INT64, Sp());
    EXPECT_THROW((void)reg.Create("system.nop", {}, Sp()), npu::tile_fwk::Error);
    EXPECT_THROW((void)reg.Create("system.nop", {count, count}, Sp()), npu::tile_fwk::Error);
}

TEST(SyncOpsNopTest, NopRejectsKeywordArguments)
{
    auto& reg = OpRegistry::GetInstance();
    auto count = std::make_shared<ConstInt>(4, DataType::INT64, Sp());
    std::vector<std::pair<std::string, std::any>> kwargs = {{"n", static_cast<int64_t>(4)}};
    EXPECT_THROW((void)reg.Create("system.nop", {count}, kwargs, Sp()), npu::tile_fwk::Error);
}

TEST(SyncOpsNopTest, NopRejectsFloatingPointScalar)
{
    auto& reg = OpRegistry::GetInstance();
    auto count = MakeScalarVar("n", DataType::FP32);
    EXPECT_THROW((void)reg.Create("system.nop", {count}, Sp()), npu::tile_fwk::Error);
}

TEST_F(SyncOpsDcciTest, Dcci_TensorTarget_TupleOffset_ReturnsUnknown)
{
    auto& reg = OpRegistry::GetInstance();
    auto tensor = MakeTensorVar("gm", {32, 64}, DataType::FP16);
    auto offset = MakeOffsetsTuple({0, 16});
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"cache_line", static_cast<int>(CacheLine::SINGLE_CACHE_LINE)}, {"dst", static_cast<int>(DcciDst::AUTO)}};
    auto call = reg.Create("system.dcci", {tensor, offset}, kwargs, Sp());
    ASSERT_NE(call, nullptr);
    EXPECT_NE(As<UnknownType>(call->GetType()), nullptr);
}

TEST_F(SyncOpsDcciTest, Dcci_TileTarget_ScalarOffset_ReturnsUnknown)
{
    auto& reg = OpRegistry::GetInstance();
    auto tile = MakeTileVar("ub", {16, 32}, DataType::FP16);
    auto offset = MakeScalarVar("off", DataType::INDEX);
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"cache_line", static_cast<int>(CacheLine::SINGLE_CACHE_LINE)},
        {"dst", static_cast<int>(DcciDst::CACHELINE_UB)}};
    auto call = reg.Create("system.dcci", {tile, offset}, kwargs, Sp());
    ASSERT_NE(call, nullptr);
    EXPECT_NE(As<UnknownType>(call->GetType()), nullptr);
}

TEST_F(SyncOpsDcciTest, Dcci_TileTarget_TupleOffset_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    auto tile = MakeTileVar("ub", {16, 32}, DataType::FP16);
    auto offset = MakeOffsetsTuple({0, 16});
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"cache_line", static_cast<int>(CacheLine::SINGLE_CACHE_LINE)}, {"dst", static_cast<int>(DcciDst::AUTO)}};
    EXPECT_THROW((void)reg.Create("system.dcci", {tile, offset}, kwargs, Sp()), npu::tile_fwk::Error);
}

TEST_F(SyncOpsDcciTest, Dcci_NonTensorNonTileTarget_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    auto scalar = MakeScalarVar("x", DataType::INT32);
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"cache_line", static_cast<int>(CacheLine::SINGLE_CACHE_LINE)}, {"dst", static_cast<int>(DcciDst::AUTO)}};
    EXPECT_THROW((void)reg.Create("system.dcci", {scalar}, kwargs, Sp()), npu::tile_fwk::Error);
}

TEST_F(SyncOpsDcciTest, Dcci_TensorTarget_NonIntOffset_Throws)
{
    auto& reg = OpRegistry::GetInstance();
    auto tensor = MakeTensorVar("gm", {64, 128}, DataType::FP16);
    auto bad_offset = MakeScalarVar("off", DataType::FP32);
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {"cache_line", static_cast<int>(CacheLine::SINGLE_CACHE_LINE)}, {"dst", static_cast<int>(DcciDst::AUTO)}};
    EXPECT_THROW((void)reg.Create("system.dcci", {tensor, bad_offset}, kwargs, Sp()), npu::tile_fwk::Error);
}

} // namespace ir
} // namespace pypto
