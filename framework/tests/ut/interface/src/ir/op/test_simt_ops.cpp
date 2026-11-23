/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_simt_ops.cpp
 * \brief Type-deduction tests for the SIMT operation registry.
 */

#include "gtest/gtest.h"

#include <any>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "ir/function.h"
#include "ir/kind_traits.h"
#include "ir/op_attr_types.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/type.h"
#include "ir/transforms/io_text.h"
#include "ir/transforms/printer.h"
#include "ir/transforms/structural_comparison.h"
#include "test_op_helpers.h"
#include "tilefwk/error.h"

namespace pypto {
namespace ir {

using namespace test_helpers;

using Kwargs = std::vector<std::pair<std::string, std::any>>;

TEST(SimtOpsTest, GroupTypeRoundTripComparisonAndHashPreserveKind)
{
    const std::vector<std::pair<SimtGroupKind, std::string>> cases = {
        {SimtGroupKind::ThreadBlock, "simt_group<0>"},
        {SimtGroupKind::ThreadBlockTile, "simt_group<1>"},
        {SimtGroupKind::CoalescedGroup, "simt_group<2>"},
        {SimtGroupKind::GridGroup, "simt_group<3>"},
    };
    for (const auto& [kind, text] : cases) {
        auto type = std::make_shared<SimtGroupType>(kind);
        auto same_kind = std::make_shared<SimtGroupType>(kind);
        EXPECT_EQ(TextDumpType(type), text);
        EXPECT_TRUE(structural_equal(type, same_kind));
        EXPECT_EQ(structural_hash(type), structural_hash(same_kind));
        std::string error;
        auto loaded = TextLoadType(text, error);
        ASSERT_NE(loaded, nullptr) << error;
        EXPECT_TRUE(error.empty());
        EXPECT_TRUE(structural_equal(type, loaded));
        EXPECT_EQ(structural_hash(type), structural_hash(loaded));
    }
    auto block = std::make_shared<SimtGroupType>(SimtGroupKind::ThreadBlock);
    auto tile = std::make_shared<SimtGroupType>(SimtGroupKind::ThreadBlockTile);
    auto active = std::make_shared<SimtGroupType>(SimtGroupKind::CoalescedGroup);
    EXPECT_FALSE(structural_equal(block, tile));
    EXPECT_FALSE(structural_equal(block, active));
    EXPECT_FALSE(structural_equal(tile, active));
    EXPECT_NE(structural_hash(block), structural_hash(tile));
    EXPECT_NE(structural_hash(block), structural_hash(active));
    EXPECT_NE(structural_hash(tile), structural_hash(active));
}

TEST(SimtOpsTest, GroupTypePythonPrintUsesSimtNamespace)
{
    const std::vector<std::pair<SimtGroupKind, std::string>> cases = {
        {SimtGroupKind::ThreadBlock, "ThreadBlock"},
        {SimtGroupKind::ThreadBlockTile, "ThreadBlockTile"},
        {SimtGroupKind::CoalescedGroup, "CoalescedGroup"},
        {SimtGroupKind::GridGroup, "GridGroup"},
    };
    for (const auto& [kind, name] : cases) {
        EXPECT_EQ(PythonPrint(std::make_shared<SimtGroupType>(kind), "pl"), "pl.simt." + name);
    }
}

TEST(SimtOpsTest, GroupTextLoaderRejectsMalformedKinds)
{
    for (const auto& text :
         {"simt_group<-1>", "simt_group<4>", "simt_group<256>", "simt_group<1.5>", "simt_group<>", "simt_group<1"}) {
        std::string error;
        EXPECT_EQ(TextLoadType(text, error), nullptr) << text;
        EXPECT_FALSE(error.empty()) << text;
    }
}

TEST(SimtOpsTest, GroupValuesTextRoundTripPreservesTypes)
{
    auto type = std::make_shared<SimtGroupType>(SimtGroupKind::CoalescedGroup);
    auto source = std::make_shared<Var>("source", type, Sp());
    auto copy = std::make_shared<Var>("copy", type, Sp());
    auto body = std::make_shared<SeqStmts>(
        std::vector<StmtPtr>{std::make_shared<AssignStmt>(std::vector<VarPtr>{copy}, source, Sp()),
                             std::make_shared<ReturnStmt>(std::vector<ExprPtr>{}, Sp())},
        Sp());
    auto function = std::make_shared<Function>("copy_group", std::vector<VarPtr>{source}, std::vector<TypePtr>{}, body,
                                               Sp());
    std::string error;
    auto text = TextDump(function);
    auto loaded = TextLoadFunction(text, error);
    ASSERT_NE(loaded, nullptr) << error;
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(TextDump(loaded), text);
    EXPECT_TRUE(structural_equal(function, loaded));
    EXPECT_EQ(structural_hash(function), structural_hash(loaded));
}

TEST(SimtOpsTest, SemanticGroupOperationsRequireTypedReceivers)
{
    auto& registry = OpRegistry::GetInstance();
    auto block = registry.Create("simt.group.this_thread_block", {}, Sp());
    auto size = std::make_shared<ConstInt>(8, DataType::INT32, Sp());
    auto tile = registry.Create("simt.group.tiled_partition", {block, size}, Sp());
    auto active = registry.Create("simt.group.coalesced_threads", {}, Sp());
    auto pred = std::make_shared<ConstBool>(true, Sp());
    auto split = registry.Create("simt.group.binary_partition", {tile, pred}, Sp());
    EXPECT_NO_THROW((void)structural_hash(split));
    EXPECT_EQ(tile->args_.size(), 2U);
    EXPECT_TRUE(tile->kwargs_.empty());
    EXPECT_EQ(split->args_.size(), 2U);
    EXPECT_TRUE(split->kwargs_.empty());
    EXPECT_EQ(As<SimtGroupType>(split->GetType())->kind_, SimtGroupKind::CoalescedGroup);
    EXPECT_EQ(As<SimtGroupType>(tile->GetType())->kind_, SimtGroupKind::ThreadBlockTile);
    for (const auto& group : {tile, active, split}) {
        const std::string prefix = group == tile ? "simt.thread_block_tile." : "simt.coalesced_group.";
        for (const auto& query : {"size", "thread_rank", "meta_group_rank", "meta_group_size"}) {
            auto call = registry.Create(prefix + query, {group}, Sp());
            EXPECT_EQ(As<ScalarType>(call->GetType())->dtype_, DataType::UINT32);
        }
        EXPECT_TRUE(As<NoneType>(registry.Create(prefix + "sync", {group}, Sp())->GetType()));
    }
    EXPECT_THROW((void)registry.Create("simt.thread_block.size", {pred}, Sp()), std::exception);
    EXPECT_THROW((void)registry.Create("simt.thread_block.size", {}, Sp()), std::exception);
    EXPECT_THROW((void)registry.Create("simt.group.this_thread_block", {pred}, Sp()), npu::tile_fwk::Error);
    EXPECT_THROW((void)registry.Create("simt.group.coalesced_threads", {pred}, Sp()), npu::tile_fwk::Error);
    EXPECT_THROW((void)registry.Create("simt.group.tiled_partition", {block}, Sp()), npu::tile_fwk::Error);
    EXPECT_THROW((void)registry.Create("simt.group.tiled_partition", {block, size, size}, Sp()), npu::tile_fwk::Error);
    EXPECT_THROW((void)registry.Create("simt.group.binary_partition", {tile}, Sp()), npu::tile_fwk::Error);
    EXPECT_THROW((void)registry.Create("simt.thread_block_tile.shfl_xor", {active, pred, pred}, Sp()), std::exception);
    EXPECT_THROW((void)registry.Create("simt.coalesced_group.ballot", {block, pred}, Sp()), std::exception);
    EXPECT_THROW((void)registry.Create("simt.group.binary_partition", {block, pred}, Sp()), std::exception);
    EXPECT_THROW((void)registry.Create("simt.group.tiled_partition", {active, size}, Sp()), std::exception);
}

TEST(SimtOpsTest, GroupMemberNamespacesRequireMatchingReceivers)
{
    auto& registry = OpRegistry::GetInstance();
    const std::vector<std::pair<std::string, SimtGroupKind>> groups = {
        {"simt.thread_block.", SimtGroupKind::ThreadBlock},
        {"simt.thread_block_tile.", SimtGroupKind::ThreadBlockTile},
        {"simt.coalesced_group.", SimtGroupKind::CoalescedGroup},
        {"simt.grid_group.", SimtGroupKind::GridGroup},
    };
    for (const auto& [prefix, expectedKind] : groups) {
        for (const auto& [otherPrefix, actualKind] : groups) {
            auto receiver = std::make_shared<Var>("group", std::make_shared<SimtGroupType>(actualKind), Sp());
            for (const auto& method : {"size", "thread_rank", "sync"}) {
                if (actualKind != expectedKind) {
                    EXPECT_THROW((void)registry.Create(prefix + method, {receiver}, Sp()), npu::tile_fwk::Error)
                        << prefix << method << " accepted receiver from " << otherPrefix;
                    continue;
                }
                auto result = registry.Create(prefix + method, {receiver}, Sp());
                if (std::string(method) == "sync") {
                    EXPECT_TRUE(As<NoneType>(result->GetType()));
                } else {
                    const auto dtype = expectedKind == SimtGroupKind::GridGroup ? DataType::UINT64 : DataType::UINT32;
                    EXPECT_EQ(As<ScalarType>(result->GetType())->dtype_, dtype);
                }
            }
        }
    }
}

TEST(SimtOpsTest, GridGroupQueriesAndReceiverContracts)
{
    auto& registry = OpRegistry::GetInstance();
    auto grid = registry.Create("simt.group.this_grid", {}, Sp());
    auto block = registry.Create("simt.group.this_thread_block", {}, Sp());
    ASSERT_EQ(As<SimtGroupType>(grid->GetType())->kind_, SimtGroupKind::GridGroup);
    for (const auto& name : {"size", "thread_rank", "block_rank", "num_blocks"}) {
        auto query = registry.Create(std::string("simt.grid_group.") + name, {grid}, Sp());
        EXPECT_EQ(As<ScalarType>(query->GetType())->dtype_, DataType::UINT64);
    }
    EXPECT_EQ(As<ScalarType>(registry.Create("simt.grid_group.is_valid", {grid}, Sp())->GetType())->dtype_,
              DataType::BOOL);
    EXPECT_TRUE(As<NoneType>(registry.Create("simt.grid_group.sync", {grid}, Sp())->GetType()));
    for (const auto& name : {"group_dim", "block_index"}) {
        for (int axis = 0; axis < 3; ++axis) {
            auto query = registry.Create(std::string("simt.grid_group.") + name, {grid}, {{"axis", axis}}, Sp());
            EXPECT_EQ(As<ScalarType>(query->GetType())->dtype_, DataType::UINT32);
        }
    }
    for (const auto& name : {"is_valid", "block_rank", "num_blocks", "block_index"}) {
        EXPECT_THROW((void)registry.Create(std::string("simt.grid_group.") + name, {block}, {{"axis", 0}}, Sp()),
                     npu::tile_fwk::Error);
    }
    for (const auto& name : {"simt.thread_block.group_index", "simt.thread_block.thread_index",
                             "simt.thread_block_tile.meta_group_rank", "simt.coalesced_group.meta_group_size"}) {
        EXPECT_THROW((void)registry.Create(name, {grid}, {{"axis", 0}}, Sp()), npu::tile_fwk::Error);
    }
    auto size = std::make_shared<ConstInt>(8, DataType::INT32, Sp());
    auto pred = std::make_shared<ConstBool>(true, Sp());
    EXPECT_THROW((void)registry.Create("simt.group.this_grid", {size}, Sp()), npu::tile_fwk::Error);
    EXPECT_THROW((void)registry.Create("simt.group.tiled_partition", {grid, size}, Sp()), npu::tile_fwk::Error);
    EXPECT_THROW((void)registry.Create("simt.group.binary_partition", {grid, pred}, Sp()), npu::tile_fwk::Error);
    EXPECT_THROW((void)registry.Create("simt.coalesced_group.ballot", {grid, pred}, Sp()), npu::tile_fwk::Error);
    EXPECT_THROW((void)registry.Create("simt.thread_block_tile.shfl", {grid, size, size}, Sp()), npu::tile_fwk::Error);
    EXPECT_THROW((void)registry.Create("simt.grid_group.group_dim", {grid}, {{"axis", 3}}, Sp()), npu::tile_fwk::Error);
}

TEST(SimtOpsTest, ThreadBlockCoordinateAxes)
{
    auto& registry = OpRegistry::GetInstance();
    auto block = registry.Create("simt.group.this_thread_block", {}, Sp());
    for (const auto& name : {"simt.thread_block.group_index", "simt.thread_block.thread_index"}) {
        SCOPED_TRACE(name);
        for (int axis : {0, 1, 2}) {
            auto query = registry.Create(name, {block}, {{"axis", axis}}, Sp());
            auto type = As<ScalarType>(query->GetType());
            ASSERT_NE(type, nullptr);
            EXPECT_EQ(type->dtype_, DataType::UINT32);
        }
        for (int axis : {-1, 3}) {
            EXPECT_THROW((void)registry.Create(name, {block}, {{"axis", axis}}, Sp()), npu::tile_fwk::Error);
        }
    }
}

TEST(SimtOpsTest, SemanticGroupDtypeAndShapeContracts)
{
    auto& registry = OpRegistry::GetInstance();
    auto block = registry.Create("simt.group.this_thread_block", {}, Sp());
    auto size = std::make_shared<ConstInt>(8, DataType::INT32, Sp());
    auto tile = registry.Create("simt.group.tiled_partition", {block, size}, Sp());
    auto active = registry.Create("simt.group.coalesced_threads", {}, Sp());
    auto index = MakeScalarVar("index", DataType::INT32);
    for (const auto& group : {tile, active}) {
        const std::string prefix = group == tile ? "simt.thread_block_tile." : "simt.coalesced_group.";
        for (auto dtype : {DataType::FP16, DataType::BF16, DataType::FP32, DataType::INT32, DataType::UINT32,
                           DataType::INT64, DataType::UINT64}) {
            auto value = MakeScalarVar("value", dtype);
            for (const auto& method : {"shfl", "shfl_up", "shfl_down"}) {
                auto result = registry.Create(prefix + method, {group, value, index}, Sp());
                EXPECT_EQ(As<ScalarType>(result->GetType())->dtype_, dtype);
            }
        }
        for (auto dtype : {DataType::BOOL, DataType::INT32, DataType::UINT32, DataType::INT64, DataType::UINT64}) {
            auto predicate = MakeScalarVar("predicate", dtype);
            for (const auto& method : {"all", "any", "ballot"}) {
                auto result = registry.Create(prefix + method, {group, predicate}, Sp());
                EXPECT_EQ(As<ScalarType>(result->GetType())->dtype_,
                          std::string(method) == "ballot" ? DataType::UINT32 : DataType::INT32);
            }
        }
        EXPECT_THROW((void)registry.Create(prefix + "ballot", {group, MakeScalarVar("f", DataType::FP32)}, Sp()),
                     npu::tile_fwk::Error);
        EXPECT_THROW((void)registry.Create(prefix + "shfl", {group, MakeScalarVar("b", DataType::BOOL), index}, Sp()),
                     npu::tile_fwk::Error);
        EXPECT_THROW((void)registry.Create(prefix + "shfl", {group, index}, Sp()), npu::tile_fwk::Error);
        EXPECT_THROW((void)registry.Create("simt.thread_block.group_dim", {group}, Kwargs{{"axis", 0}}, Sp()),
                     npu::tile_fwk::Error);
    }
    for (int invalid_size : {0, 3, 64}) {
        auto invalid = std::make_shared<ConstInt>(invalid_size, DataType::INT32, Sp());
        EXPECT_THROW((void)registry.Create("simt.group.tiled_partition", {block, invalid}, Sp()), npu::tile_fwk::Error);
    }
    auto runtime_size = MakeScalarVar("runtime_size", DataType::UINT32);
    EXPECT_NO_THROW((void)registry.Create("simt.group.tiled_partition", {block, runtime_size}, Sp()));
    EXPECT_NO_THROW((void)registry.Create("simt.group.tiled_partition", {tile, runtime_size}, Sp()));
    EXPECT_THROW((void)registry.Create("simt.group.tiled_partition", {block, MakeScalarVar("f", DataType::FP32)}, Sp()),
                 npu::tile_fwk::Error);
    EXPECT_THROW((void)registry.Create("simt.thread_block.group_dim", {block}, Kwargs{{"axis", 3}}, Sp()),
                 npu::tile_fwk::Error);
}

TEST(SimtOpsTest, ContextAndBuiltinScalarTypes)
{
    auto& registry = OpRegistry::GetInstance();
    const std::vector<std::string> context_ops = {"simt.thread_idx", "simt.block_dim", "simt.block_idx",
                                                  "simt.grid_dim"};

    for (std::size_t axis = 0; axis < context_ops.size(); ++axis) {
        auto call = registry.Create(context_ops[axis], {}, Kwargs{{"axis", static_cast<int>(axis % 3)}}, Sp());
        auto type = As<ScalarType>(call->GetType());
        ASSERT_NE(type, nullptr);
        EXPECT_EQ(type->dtype_, DataType::UINT32);
    }

    auto linear = registry.Create("simt.linear_thread_idx", {}, Sp());
    auto linear_type = As<ScalarType>(linear->GetType());
    ASSERT_NE(linear_type, nullptr);
    EXPECT_EQ(linear_type->dtype_, DataType::UINT32);

    auto warp = registry.Create("simt.warp_size", {}, Sp());
    auto warp_type = As<ScalarType>(warp->GetType());
    ASSERT_NE(warp_type, nullptr);
    EXPECT_EQ(warp_type->dtype_, DataType::INT32);
}

TEST(SimtOpsTest, WarpQueryAndVoteTypes)
{
    auto& registry = OpRegistry::GetInstance();
    for (const auto& op_name : {"simt.lane_id", "simt.lanemask_eq", "simt.lanemask_le", "simt.lanemask_lt",
                                "simt.lanemask_ge", "simt.lanemask_gt"}) {
        auto result_type = As<ScalarType>(registry.Create(op_name, {}, Sp())->GetType());
        ASSERT_NE(result_type, nullptr);
        EXPECT_EQ(result_type->dtype_, DataType::INT32);
    }

    auto bool_predicate = MakeScalarVar("bool_predicate", DataType::BOOL);
    auto int_predicate = MakeScalarVar("int_predicate", DataType::INT32);
    for (const auto& op_name : {"simt.warp_all", "simt.warp_any"}) {
        auto result_type = As<ScalarType>(registry.Create(op_name, {bool_predicate}, Sp())->GetType());
        ASSERT_NE(result_type, nullptr);
        EXPECT_EQ(result_type->dtype_, DataType::INT32);
    }
    auto ballot_type = As<ScalarType>(registry.Create("simt.warp_ballot", {int_predicate}, Sp())->GetType());
    ASSERT_NE(ballot_type, nullptr);
    EXPECT_EQ(ballot_type->dtype_, DataType::UINT32);

    auto active_mask_type = As<ScalarType>(registry.Create("simt.warp_active_mask", {}, Sp())->GetType());
    ASSERT_NE(active_mask_type, nullptr);
    EXPECT_EQ(active_mask_type->dtype_, DataType::UINT32);
}

TEST(SimtOpsTest, WarpShuffleAndReductionTypes)
{
    auto& registry = OpRegistry::GetInstance();
    auto fp16_value = MakeScalarVar("fp16_value", DataType::FP16);
    auto src_lane = std::make_shared<ConstInt>(3, DataType::INT32, Sp());
    auto delta = std::make_shared<ConstInt>(1, DataType::UINT32, Sp());
    auto width = std::make_shared<ConstInt>(16, DataType::INT32, Sp());

    for (const auto& dtype : {DataType::INT32, DataType::UINT32, DataType::INT64, DataType::UINT64, DataType::FP16,
                              DataType::BF16, DataType::FP32}) {
        auto value = MakeScalarVar("shuffle_value", dtype);
        auto result_type = As<ScalarType>(registry.Create("simt.warp_shfl", {value, src_lane, width}, Sp())->GetType());
        ASSERT_NE(result_type, nullptr);
        EXPECT_EQ(result_type->dtype_, dtype);
    }
    for (const auto& [op_name, control] :
         std::vector<std::pair<std::string, ExprPtr>>{{"simt.warp_shfl", src_lane},
                                                      {"simt.warp_shfl_up", delta},
                                                      {"simt.warp_shfl_down", delta},
                                                      {"simt.warp_shfl_xor", src_lane}}) {
        auto result_type = As<ScalarType>(registry.Create(op_name, {fp16_value, control, width}, Sp())->GetType());
        ASSERT_NE(result_type, nullptr);
        EXPECT_EQ(result_type->dtype_, DataType::FP16);
    }

    for (const auto& dtype : {DataType::INT32, DataType::UINT32, DataType::FP16, DataType::FP32}) {
        auto value = MakeScalarVar("reduce_value", dtype);
        for (const auto& op_name : {"simt.warp_reduce_add", "simt.warp_reduce_max", "simt.warp_reduce_min"}) {
            auto result_type = As<ScalarType>(registry.Create(op_name, {value}, Sp())->GetType());
            ASSERT_NE(result_type, nullptr);
            EXPECT_EQ(result_type->dtype_, dtype);
        }
    }
}

TEST(SimtOpsTest, RejectsInvalidWarpContracts)
{
    auto& registry = OpRegistry::GetInstance();
    auto fp32_value = MakeScalarVar("fp32_value", DataType::FP32);
    auto bf16_value = MakeScalarVar("bf16_value", DataType::BF16);
    auto i16_value = MakeScalarVar("i16_value", DataType::INT16);
    auto src_lane = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto wrong_control = std::make_shared<ConstInt>(1, DataType::UINT32, Sp());
    auto width_3 = std::make_shared<ConstInt>(3, DataType::INT32, Sp());
    auto width_8 = std::make_shared<ConstInt>(8, DataType::INT32, Sp());
    auto lane_mask_8 = std::make_shared<ConstInt>(8, DataType::INT32, Sp());

    EXPECT_THROW((void)registry.Create("simt.warp_ballot", {fp32_value}, Sp()), npu::tile_fwk::Error);
    EXPECT_THROW((void)registry.Create("simt.warp_shfl", {i16_value, src_lane, width_8}, Sp()), npu::tile_fwk::Error);
    EXPECT_THROW((void)registry.Create("simt.warp_shfl", {fp32_value, wrong_control, width_8}, Sp()),
                 npu::tile_fwk::Error);
    EXPECT_THROW((void)registry.Create("simt.warp_shfl", {fp32_value, src_lane, width_3}, Sp()), npu::tile_fwk::Error);
    EXPECT_THROW((void)registry.Create("simt.warp_shfl_xor", {fp32_value, lane_mask_8, width_8}, Sp()),
                 npu::tile_fwk::Error);
    EXPECT_THROW((void)registry.Create("simt.warp_reduce_add", {bf16_value}, Sp()), npu::tile_fwk::Error);
}

TEST(SimtOpsTest, LaunchType)
{
    auto& registry = OpRegistry::GetInstance();
    auto scalar = MakeScalarVar("count", DataType::UINT32);

    std::vector<ExprPtr> launch_args = {
        std::make_shared<ConstInt>(8, DataType::INT64, Sp()),
        std::make_shared<ConstInt>(4, DataType::INT64, Sp()),
        std::make_shared<ConstInt>(8, DataType::INT64, Sp()),
        scalar,
    };
    auto launch = registry.Create("simt.launch", launch_args,
                                  Kwargs{{"callee", std::string("simt_entry")}, {"max_threads", 256}}, Sp());

    EXPECT_EQ(launch->GetType(), GetUnknownType());
}

TEST(SimtOpsTest, CastTypeDeductionContracts)
{
    auto& registry = OpRegistry::GetInstance();
    auto make_cast = [&registry](DataType source_dtype, DataType target_dtype, RoundMode mode) {
        auto source = MakeScalarVar("value", source_dtype);
        return registry.Create("simt.cast", {source},
                               Kwargs{{"target_type", target_dtype}, {"mode", static_cast<int>(mode)}}, Sp());
    };

    auto rounded = make_cast(DataType::FP32, DataType::FP16, RoundMode::CAST_RINT);
    EXPECT_EQ(rounded->name_, "simt.cast");
    EXPECT_EQ(rounded->GetKwarg<DataType>("target_type"), DataType::FP16);
    EXPECT_EQ(rounded->GetKwarg<int>("mode"), static_cast<int>(RoundMode::CAST_RINT));
    auto rounded_type = As<ScalarType>(rounded->GetType());
    ASSERT_NE(rounded_type, nullptr);
    EXPECT_EQ(rounded_type->dtype_, DataType::FP16);

    struct CastCase {
        DataType source;
        DataType target;
        RoundMode mode;
    };
    const std::vector<CastCase> representative_cases = {
        {DataType::INT32, DataType::UINT64, RoundMode::CAST_NONE},
        {DataType::FP32, DataType::FP32, RoundMode::CAST_NONE},
        {DataType::BF16, DataType::FP32, RoundMode::CAST_NONE},
        {DataType::FP32, DataType::BF16, RoundMode::CAST_ROUND},
        {DataType::FP32, DataType::UINT64, RoundMode::CAST_FLOOR},
        {DataType::INT64, DataType::FP32, RoundMode::CAST_TRUNC},
        {DataType::FP32, DataType::FP16, RoundMode::CAST_ODD},
    };
    for (const auto& cast_case : representative_cases) {
        auto result = make_cast(cast_case.source, cast_case.target, cast_case.mode);
        auto result_type = As<ScalarType>(result->GetType());
        ASSERT_NE(result_type, nullptr);
        EXPECT_EQ(result_type->dtype_, cast_case.target);
    }
}

TEST(SimtOpsTest, CastRejectsUnsupportedDtypeAndModeCombinations)
{
    auto& registry = OpRegistry::GetInstance();
    auto expect_rejected = [&registry](DataType source_dtype, DataType target_dtype, RoundMode mode) {
        auto source = MakeScalarVar("value", source_dtype);
        EXPECT_THROW(
            (void)registry.Create("simt.cast", {source},
                                  Kwargs{{"target_type", target_dtype}, {"mode", static_cast<int>(mode)}}, Sp()),
            npu::tile_fwk::Error);
    };

    expect_rejected(DataType::FP16, DataType::INT32, RoundMode::CAST_NONE);
    expect_rejected(DataType::FP32, DataType::BF16, RoundMode::CAST_ODD);
    expect_rejected(DataType::INT32, DataType::INT64, RoundMode::CAST_RINT);
}

TEST(SimtOpsTest, CastSupportsExtendedLowPrecisionConversionMatrix)
{
    auto& registry = OpRegistry::GetInstance();
    auto expect_supported = [&registry](DataType source_dtype, DataType target_dtype, RoundMode mode) {
        auto source = MakeScalarVar("value", source_dtype);
        auto result = registry.Create("simt.cast", {source},
                                      Kwargs{{"target_type", target_dtype}, {"mode", static_cast<int>(mode)}}, Sp());
        auto result_type = As<ScalarType>(result->GetType());
        ASSERT_NE(result_type, nullptr);
        EXPECT_EQ(result_type->dtype_, target_dtype);
    };

    const std::vector<DataType> low_float_dtypes = {DataType::FP16, DataType::BF16};
    const std::vector<DataType> wide_integer_dtypes = {DataType::INT32, DataType::UINT32, DataType::INT64,
                                                       DataType::UINT64};
    const std::vector<DataType> int16_dtypes = {DataType::INT16, DataType::UINT16};
    const std::vector<DataType> int8_dtypes = {DataType::INT8, DataType::UINT8};
    const std::vector<RoundMode> standard_modes = {RoundMode::CAST_RINT, RoundMode::CAST_ROUND, RoundMode::CAST_FLOOR,
                                                   RoundMode::CAST_CEIL, RoundMode::CAST_TRUNC};
    const std::vector<RoundMode> int16_modes = {RoundMode::CAST_RINT, RoundMode::CAST_FLOOR, RoundMode::CAST_CEIL,
                                                RoundMode::CAST_TRUNC};

    for (auto mode : standard_modes) {
        expect_supported(DataType::FP16, DataType::BF16, mode);
        expect_supported(DataType::BF16, DataType::FP16, mode);
        for (auto low_float_dtype : low_float_dtypes) {
            for (auto wide_integer_dtype : wide_integer_dtypes) {
                expect_supported(low_float_dtype, wide_integer_dtype, mode);
                expect_supported(wide_integer_dtype, low_float_dtype, mode);
            }
        }
    }
    for (auto mode : int16_modes) {
        for (auto low_float_dtype : low_float_dtypes) {
            for (auto int16_dtype : int16_dtypes) {
                expect_supported(low_float_dtype, int16_dtype, mode);
                expect_supported(int16_dtype, low_float_dtype, mode);
            }
        }
    }
    for (auto low_float_dtype : low_float_dtypes) {
        for (auto int8_dtype : int8_dtypes) {
            expect_supported(low_float_dtype, int8_dtype, RoundMode::CAST_TRUNC);
        }
    }
}

TEST(SimtOpsTest, CastRejectsUnsupportedExtendedConversionModes)
{
    auto& registry = OpRegistry::GetInstance();
    auto expect_rejected = [&registry](DataType source_dtype, DataType target_dtype, RoundMode mode) {
        auto source = MakeScalarVar("value", source_dtype);
        EXPECT_THROW(
            (void)registry.Create("simt.cast", {source},
                                  Kwargs{{"target_type", target_dtype}, {"mode", static_cast<int>(mode)}}, Sp()),
            npu::tile_fwk::Error);
    };

    expect_rejected(DataType::FP16, DataType::BF16, RoundMode::CAST_NONE);
    expect_rejected(DataType::FP16, DataType::INT64, RoundMode::CAST_ODD);
    expect_rejected(DataType::INT16, DataType::BF16, RoundMode::CAST_ROUND);
    expect_rejected(DataType::BF16, DataType::INT8, RoundMode::CAST_RINT);
    expect_rejected(DataType::INT8, DataType::FP16, RoundMode::CAST_TRUNC);
}

TEST(SimtOpsTest, BitcastSupportsOnlyAscScalarPairs)
{
    auto& registry = OpRegistry::GetInstance();
    const std::vector<std::pair<DataType, DataType>> supported_pairs = {
        {DataType::INT16, DataType::FP16},  {DataType::UINT16, DataType::FP16}, {DataType::FP16, DataType::INT16},
        {DataType::FP16, DataType::UINT16}, {DataType::INT16, DataType::BF16},  {DataType::UINT16, DataType::BF16},
        {DataType::BF16, DataType::INT16},  {DataType::BF16, DataType::UINT16}, {DataType::INT32, DataType::FP32},
        {DataType::UINT32, DataType::FP32}, {DataType::FP32, DataType::INT32},  {DataType::FP32, DataType::UINT32},
    };

    for (const auto& [source_dtype, target_dtype] : supported_pairs) {
        auto source = MakeScalarVar("value", source_dtype);
        auto result = registry.Create("simt.bitcast", {source}, Kwargs{{"target_type", target_dtype}}, Sp());
        EXPECT_EQ(result->name_, "simt.bitcast");
        EXPECT_EQ(result->GetKwarg<DataType>("target_type"), target_dtype);
        auto result_type = As<ScalarType>(result->GetType());
        ASSERT_NE(result_type, nullptr);
        EXPECT_EQ(result_type->dtype_, target_dtype);
    }

    const std::vector<std::pair<DataType, DataType>> unsupported_pairs = {
        {DataType::FP16, DataType::BF16},
        {DataType::INT16, DataType::UINT16},
        {DataType::INT32, DataType::UINT32},
        {DataType::INT64, DataType::UINT64},
    };
    for (const auto& [source_dtype, target_dtype] : unsupported_pairs) {
        auto source = MakeScalarVar("value", source_dtype);
        EXPECT_THROW((void)registry.Create("simt.bitcast", {source}, Kwargs{{"target_type", target_dtype}}, Sp()),
                     npu::tile_fwk::Error);
    }
}

TEST(SimtOpsTest, BitcastRejectsMismatchedBitWidths)
{
    auto& registry = OpRegistry::GetInstance();
    auto source = MakeScalarVar("value", DataType::FP16);

    try {
        (void)registry.Create("simt.bitcast", {source}, Kwargs{{"target_type", DataType::UINT32}}, Sp());
        FAIL() << "Expected simt.bitcast to reject mismatched bit widths";
    } catch (const npu::tile_fwk::Error& error) {
        EXPECT_NE(std::string(error.what()).find("same bit width"), std::string::npos);
    }
}

TEST(SimtOpsTest, TranscendentalUnaryMathTypeContracts)
{
    auto& registry = OpRegistry::GetInstance();
    auto expect_unary_type = [&registry](const std::string& name, DataType input_dtype, DataType result_dtype) {
        auto value = MakeScalarVar("value", input_dtype);
        auto call = registry.Create(name, {value}, Sp());
        auto type = As<ScalarType>(call->GetType());
        ASSERT_NE(type, nullptr);
        EXPECT_EQ(type->dtype_, result_dtype);
    };

    for (const auto& name : {"simt.exp10", "simt.log10"}) {
        for (const auto& dtype : {DataType::FP16, DataType::BF16, DataType::FP32}) {
            expect_unary_type(name, dtype, dtype);
        }
    }
    for (const auto& dtype : {DataType::FP16, DataType::BF16}) {
        expect_unary_type("simt.rcp", dtype, dtype);
    }
    for (const auto& name : {"simt.tan", "simt.atan", "simt.expm1", "simt.logb", "simt.cosh", "simt.acos", "simt.sinh",
                             "simt.asin", "simt.cbrt"}) {
        expect_unary_type(name, DataType::FP32, DataType::FP32);
    }

    EXPECT_THROW((void)registry.Create("simt.rcp", {MakeScalarVar("value", DataType::FP32)}, Sp()),
                 npu::tile_fwk::Error);
    EXPECT_THROW((void)registry.Create("simt.tan", {MakeScalarVar("value", DataType::FP16)}, Sp()),
                 npu::tile_fwk::Error);
}

TEST(SimtOpsTest, PiHyperbolicAndReciprocalCubeRootTypeContracts)
{
    auto& registry = OpRegistry::GetInstance();
    auto expect_result = [&registry](const std::string& name, const std::vector<ExprPtr>& args, DataType result_dtype) {
        auto type = As<ScalarType>(registry.Create(name, args, Sp())->GetType());
        ASSERT_NE(type, nullptr) << name;
        EXPECT_EQ(type->dtype_, result_dtype) << name;
    };

    auto fp32 = MakeScalarVar("fp32", DataType::FP32);
    auto i32 = MakeScalarVar("i32", DataType::INT32);
    for (const auto& name : {
             "simt.tanpi",
             "simt.atanh",
             "simt.cospi",
             "simt.acosh",
             "simt.sinpi",
             "simt.asinh",
             "simt.rcbrt",
         }) {
        expect_result(name, {fp32}, DataType::FP32);
    }

    EXPECT_THROW((void)registry.Create("simt.tanpi", {i32}, Sp()), npu::tile_fwk::Error);
}

TEST(SimtOpsTest, SelectedNanBinaryAndExponentTypeDeductionContracts)
{
    auto& registry = OpRegistry::GetInstance();
    auto expect_result = [&registry](const std::string& name, const std::vector<ExprPtr>& args, DataType result_dtype) {
        auto type = As<ScalarType>(registry.Create(name, args, Sp())->GetType());
        ASSERT_NE(type, nullptr) << name;
        EXPECT_EQ(type->dtype_, result_dtype) << name;
    };

    auto fp16 = MakeScalarVar("fp16", DataType::FP16);
    auto bf16 = MakeScalarVar("bf16", DataType::BF16);
    auto fp32 = MakeScalarVar("fp32", DataType::FP32);
    for (const auto& name : {"simt.max_nan", "simt.min_nan"}) {
        expect_result(name, {fp16, fp16}, DataType::FP16);
        expect_result(name, {bf16, bf16}, DataType::BF16);
    }
    for (const auto& name : {"simt.atan2", "simt.copysign", "simt.nextafter"}) {
        expect_result(name, {fp32, fp32}, DataType::FP32);
    }
    expect_result("simt.ilogb", {fp32}, DataType::INT32);
    expect_result("simt.signbit", {fp32}, DataType::INT32);

    EXPECT_THROW((void)registry.Create("simt.max_nan", {fp32, fp32}, Sp()), npu::tile_fwk::Error);
    EXPECT_THROW((void)registry.Create("simt.atan2", {fp16, fp16}, Sp()), npu::tile_fwk::Error);
    EXPECT_THROW((void)registry.Create("simt.max_nan", {fp16, bf16}, Sp()), npu::tile_fwk::Error);
    EXPECT_THROW((void)registry.Create("simt.nextafter", {fp32, fp16}, Sp()), npu::tile_fwk::Error);
    EXPECT_THROW((void)registry.Create("simt.ilogb", {fp16}, Sp()), npu::tile_fwk::Error);
    EXPECT_THROW((void)registry.Create("simt.signbit", {bf16}, Sp()), npu::tile_fwk::Error);
    EXPECT_THROW((void)registry.Create("simt.atan2", {fp32}, Sp()), npu::tile_fwk::Error);
    EXPECT_THROW((void)registry.Create("simt.rcbrt", {fp32, fp32}, Sp()), npu::tile_fwk::Error);
}

TEST(SimtOpsTest, RejectsInvalidContextAxisAndLaunchBounds)
{
    auto& registry = OpRegistry::GetInstance();
    EXPECT_THROW((void)registry.Create("simt.thread_idx", {}, Kwargs{{"axis", 3}}, Sp()), npu::tile_fwk::Error);

    std::vector<ExprPtr> too_many_threads = {
        std::make_shared<ConstInt>(2048, DataType::INT64, Sp()),
        std::make_shared<ConstInt>(2, DataType::INT64, Sp()),
        std::make_shared<ConstInt>(1, DataType::INT64, Sp()),
    };
    EXPECT_THROW((void)registry.Create("simt.launch", too_many_threads,
                                       Kwargs{{"callee", std::string("simt_entry")}, {"max_threads", 2048}}, Sp()),
                 npu::tile_fwk::Error);
}

TEST(SimtOpsTest, RejectsInvalidLaunchArgumentAbi)
{
    auto& registry = OpRegistry::GetInstance();
    auto invalid_tensor = MakeTensorVar("tensor", {256}, DataType::FP32);
    std::vector<ExprPtr> tensor_args = {
        std::make_shared<ConstInt>(1, DataType::INT64, Sp()),
        std::make_shared<ConstInt>(1, DataType::INT64, Sp()),
        std::make_shared<ConstInt>(1, DataType::INT64, Sp()),
        invalid_tensor,
    };
    EXPECT_THROW((void)registry.Create("simt.launch", tensor_args,
                                       Kwargs{{"callee", std::string("simt_entry")}, {"max_threads", 1}}, Sp()),
                 npu::tile_fwk::Error);

    auto invalid_tile = MakeTileVar("tile", {1, 256}, DataType::FP32);
    std::vector<ExprPtr> tile_args = {tensor_args[0], tensor_args[1], tensor_args[2], invalid_tile};
    EXPECT_THROW((void)registry.Create("simt.launch", tile_args,
                                       Kwargs{{"callee", std::string("simt_entry")}, {"max_threads", 1}}, Sp()),
                 npu::tile_fwk::Error);
}
TEST(SimtOpsTest, IntegerBitAndArithmeticTypeContracts)
{
    auto& registry = OpRegistry::GetInstance();
    auto expect_result = [&registry](const std::string& name, const std::vector<ExprPtr>& args, DataType result_dtype) {
        auto call = registry.Create(name, args, Sp());
        auto type = As<ScalarType>(call->GetType());
        ASSERT_NE(type, nullptr);
        EXPECT_EQ(type->dtype_, result_dtype);
    };

    auto i32 = MakeScalarVar("i32", DataType::INT32);
    auto u32 = MakeScalarVar("u32", DataType::UINT32);
    auto i64 = MakeScalarVar("i64", DataType::INT64);
    auto u64 = MakeScalarVar("u64", DataType::UINT64);
    expect_result("simt.brev", {u32}, DataType::UINT32);
    expect_result("simt.brev", {u64}, DataType::UINT64);
    expect_result("simt.clz", {i32}, DataType::INT32);
    expect_result("simt.clz", {i64}, DataType::INT32);
    expect_result("simt.ffs", {i32}, DataType::INT32);
    expect_result("simt.ffs", {i64}, DataType::INT32);
    expect_result("simt.byte_perm", {u32, u32, u32}, DataType::UINT32);
    expect_result("simt.sad", {i32, i32, u32}, DataType::UINT32);
    expect_result("simt.sad", {u32, u32, u32}, DataType::UINT32);
    for (const auto& name : {"simt.mul24", "simt.hadd", "simt.rhadd"}) {
        expect_result(name, {i32, i32}, DataType::INT32);
        expect_result(name, {u32, u32}, DataType::UINT32);
    }

    EXPECT_THROW((void)registry.Create("simt.brev", {i32}, Sp()), npu::tile_fwk::Error);
    EXPECT_THROW((void)registry.Create("simt.sad", {i32, i32, i32}, Sp()), npu::tile_fwk::Error);
}

TEST(SimtOpsTest, LowPrecisionComparisonAndFp32BinaryTypeContracts)
{
    auto& registry = OpRegistry::GetInstance();
    auto expect_result = [&registry](const std::string& name, const std::vector<ExprPtr>& args, DataType result_dtype) {
        auto call = registry.Create(name, args, Sp());
        auto type = As<ScalarType>(call->GetType());
        ASSERT_NE(type, nullptr);
        EXPECT_EQ(type->dtype_, result_dtype);
    };

    auto fp16 = MakeScalarVar("fp16", DataType::FP16);
    auto bf16 = MakeScalarVar("bf16", DataType::BF16);
    auto fp32 = MakeScalarVar("fp32", DataType::FP32);
    expect_result("simt.fma_relu", {fp16, fp16, fp16}, DataType::FP16);
    expect_result("simt.fma_relu", {bf16, bf16, bf16}, DataType::BF16);
    for (const auto& name : {"simt.equ", "simt.leu", "simt.geu", "simt.ltu", "simt.gtu"}) {
        expect_result(name, {fp16, fp16}, DataType::BOOL);
        expect_result(name, {bf16, bf16}, DataType::BOOL);
    }
    for (const auto& name : {"simt.hypot", "simt.rhypot", "simt.fdim", "simt.remainder"}) {
        expect_result(name, {fp32, fp32}, DataType::FP32);
    }

    EXPECT_THROW((void)registry.Create("simt.fma_relu", {fp32, fp32, fp32}, Sp()), npu::tile_fwk::Error);
    EXPECT_THROW((void)registry.Create("simt.equ", {fp16, bf16}, Sp()), npu::tile_fwk::Error);
}

TEST(SimtOpsTest, RecognizedSpecialMathTypeDeductionContracts)
{
    auto& registry = OpRegistry::GetInstance();
    auto expect_result = [&registry](const std::string& name, const std::vector<ExprPtr>& args, DataType result_dtype) {
        auto type = As<ScalarType>(registry.Create(name, args, Sp())->GetType());
        ASSERT_NE(type, nullptr) << name;
        EXPECT_EQ(type->dtype_, result_dtype) << name;
    };

    auto fp32 = MakeScalarVar("fp32", DataType::FP32);
    auto i32 = MakeScalarVar("i32", DataType::INT32);
    auto i64 = MakeScalarVar("i64", DataType::INT64);
    for (const auto& name : {
             "simt.erf",
             "simt.erfc",
             "simt.erfinv",
             "simt.erfcinv",
             "simt.erfcx",
             "simt.tgamma",
             "simt.lgamma",
             "simt.normcdf",
             "simt.normcdfinv",
         }) {
        expect_result(name, {fp32}, DataType::FP32);
    }
    expect_result("simt.ldexp", {fp32, i32}, DataType::FP32);
    expect_result("simt.scalbn", {fp32, i32}, DataType::FP32);
    expect_result("simt.scalbn", {fp32, i64}, DataType::FP32);
    for (const auto& name : {"simt.norm3d", "simt.rnorm3d"}) {
        expect_result(name, {fp32, fp32, fp32}, DataType::FP32);
    }
    for (const auto& name : {"simt.norm4d", "simt.rnorm4d"}) {
        expect_result(name, {fp32, fp32, fp32, fp32}, DataType::FP32);
    }

    EXPECT_THROW((void)registry.Create("simt.scalbn", {fp32, fp32}, Sp()), npu::tile_fwk::Error);
    EXPECT_THROW((void)registry.Create("simt.norm4d", {fp32, fp32, fp32}, Sp()), npu::tile_fwk::Error);
}

TEST(SimtOpsTest, MulI32ToI64TypeDeductionContracts)
{
    auto& registry = OpRegistry::GetInstance();
    auto i32 = MakeScalarVar("i32", DataType::INT32);
    auto u32 = MakeScalarVar("u32", DataType::UINT32);

    auto signed_type = As<ScalarType>(registry.Create("simt.mul_i32_to_i64", {i32, i32}, Sp())->GetType());
    auto unsigned_type = As<ScalarType>(registry.Create("simt.mul_i32_to_i64", {u32, u32}, Sp())->GetType());
    ASSERT_NE(signed_type, nullptr);
    ASSERT_NE(unsigned_type, nullptr);
    EXPECT_EQ(signed_type->dtype_, DataType::INT64);
    EXPECT_EQ(unsigned_type->dtype_, DataType::UINT64);
    EXPECT_THROW((void)registry.Create("simt.mul_i32_to_i64", {i32, u32}, Sp()), npu::tile_fwk::Error);
}

TEST(SimtOpsTest, BesselRoundingAndSaturateTypeDeductionContracts)
{
    auto& registry = OpRegistry::GetInstance();
    auto expect_result = [&registry](const std::string& name, const std::vector<ExprPtr>& args, DataType result_dtype) {
        auto type = As<ScalarType>(registry.Create(name, args, Sp())->GetType());
        ASSERT_NE(type, nullptr) << name;
        EXPECT_EQ(type->dtype_, result_dtype) << name;
    };
    auto expect_rounding_result = [&registry](const std::string& name, DataType input_dtype, DataType result_dtype) {
        auto value = MakeScalarVar("value", input_dtype);
        auto type = As<ScalarType>(
            registry.Create(name, {value}, Kwargs{{"target_type", result_dtype}}, Sp())->GetType());
        ASSERT_NE(type, nullptr) << name;
        EXPECT_EQ(type->dtype_, result_dtype) << name;
    };

    auto fp32 = MakeScalarVar("fp32", DataType::FP32);
    auto i32 = MakeScalarVar("i32", DataType::INT32);
    for (const auto& name : {
             "simt.cyl_bessel_i0",
             "simt.cyl_bessel_i1",
             "simt.j0",
             "simt.j1",
             "simt.y0",
             "simt.y1",
             "simt.saturate",
         }) {
        expect_result(name, {fp32}, DataType::FP32);
    }
    expect_result("simt.jn", {i32, fp32}, DataType::FP32);
    expect_result("simt.yn", {i32, fp32}, DataType::FP32);
    for (const auto& name : {"simt.rint", "simt.round"}) {
        expect_rounding_result(name, DataType::FP16, DataType::FP16);
        expect_rounding_result(name, DataType::BF16, DataType::BF16);
        expect_rounding_result(name, DataType::FP32, DataType::FP32);
        expect_rounding_result(name, DataType::FP32, DataType::INT64);
    }

    EXPECT_THROW((void)registry.Create("simt.jn", {fp32, i32}, Sp()), npu::tile_fwk::Error);
}

TEST(SimtOpsTest, NopAcceptsConstantCount)
{
    auto& registry = OpRegistry::GetInstance();
    auto count = std::make_shared<ConstInt>(4, DataType::INT32, Sp());
    auto call = registry.Create("simt.nop", {count}, Sp());
    ASSERT_NE(call, nullptr);
    EXPECT_NE(As<NoneType>(call->GetType()), nullptr);
}

TEST(SimtOpsTest, NopRejectsInvalidArguments)
{
    auto& registry = OpRegistry::GetInstance();
    auto count = std::make_shared<ConstInt>(4, DataType::INT32, Sp());
    EXPECT_THROW((void)registry.Create("simt.nop", {}, Sp()), npu::tile_fwk::Error);
    EXPECT_THROW((void)registry.Create("simt.nop", {count, count}, Sp()), npu::tile_fwk::Error);
    EXPECT_THROW((void)registry.Create("simt.nop", {count}, Kwargs{{"n", 4}}, Sp()), npu::tile_fwk::Error);
}

} // namespace ir
} // namespace pypto
