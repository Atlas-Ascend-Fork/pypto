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
#include "ir/pipe.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"
#include "test_op_helpers.h"
#include "tilefwk/error.h"

namespace pypto {
namespace ir {

using namespace test_helpers;

// ============================================================================
// system.mark_stamp
// ============================================================================

TEST(SyncOpsPmuStampTest, RegisteredAsSyncOp)
{
    const auto& entry = OpRegistry::GetInstance().GetEntry("system.mark_stamp");
    EXPECT_EQ(entry.GetName(), "system.mark_stamp");
    EXPECT_EQ(entry.GetOpCategory(), "SyncOp");
}

TEST(SyncOpsPmuStampTest, NoOperandsReturnsNoneAndPreservesAttributes)
{
    auto& reg = OpRegistry::GetInstance();
    for (auto pipe :
         {PipeType::S, PipeType::MTE2, PipeType::MTE1, PipeType::M, PipeType::FIX, PipeType::V, PipeType::MTE3}) {
        for (int id : {0, 4095}) {
            SCOPED_TRACE(std::to_string(static_cast<int>(pipe)) + ":" + std::to_string(id));
            std::vector<std::pair<std::string, std::any>> kwargs = {{"pipe", static_cast<int>(pipe)}, {"stamp_id", id}};
            auto call = reg.Create("system.mark_stamp", {}, kwargs, Sp());
            ASSERT_NE(call, nullptr);
            EXPECT_EQ(call->name_, "system.mark_stamp");
            EXPECT_TRUE(call->args_.empty());
            EXPECT_NE(As<NoneType>(call->GetType()), nullptr);
            ASSERT_EQ(call->kwargs_.size(), 2);
            EXPECT_EQ(call->GetKwarg<int>("pipe"), static_cast<int>(pipe));
            EXPECT_EQ(call->GetKwarg<int>("stamp_id"), id);
        }
    }
}

TEST(SyncOpsPmuStampTest, RejectsInvalidAttributeTypes)
{
    auto& reg = OpRegistry::GetInstance();
    for (const std::any& pipe : {std::any(std::string("S")), std::any(true)}) {
        std::vector<std::pair<std::string, std::any>> kwargs = {{"pipe", pipe}, {"stamp_id", 0}};
        EXPECT_THROW((void)reg.Create("system.mark_stamp", {}, kwargs, Sp()), npu::tile_fwk::Error);
    }
    for (const std::any& id : {std::any(true), std::any(1.5), std::any(std::string("1"))}) {
        std::vector<std::pair<std::string, std::any>> kwargs = {{"pipe", static_cast<int>(PipeType::S)},
                                                                {"stamp_id", id}};
        EXPECT_THROW((void)reg.Create("system.mark_stamp", {}, kwargs, Sp()), npu::tile_fwk::Error);
    }
}

TEST(SyncOpsPmuStampTest, RejectsUnknownAttributes)
{
    auto& reg = OpRegistry::GetInstance();
    for (const char* attribute : {"name", "stamp_nops", "id"}) {
        SCOPED_TRACE(attribute);
        std::vector<std::pair<std::string, std::any>> kwargs = {
            {"pipe", static_cast<int>(PipeType::S)}, {"stamp_id", 0}, {attribute, 1}};
        EXPECT_THROW((void)reg.Create("system.mark_stamp", {}, kwargs, Sp()), ValueError);
    }
}

// ============================================================================
// system.mutex_lock_dyn / system.mutex_unlock_dyn
// ============================================================================

TEST(SyncOpsMutexTest, DynamicMutexRejectsPerTileGroups)
{
    auto& reg = OpRegistry::GetInstance();
    auto id0 = MakeScalarVar("id0", DataType::INT64);
    auto id1 = MakeScalarVar("id1", DataType::INT64);
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
    auto id0 = MakeScalarVar("id0", DataType::INT64);
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
    auto id = MakeScalarVar("id", DataType::INT64);
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
    auto offset = MakeScalarVar("off", DataType::INT64);
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
