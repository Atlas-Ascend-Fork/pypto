/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <tuple>
#include "interface/cache/function_cache.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/irbuilder.h"
#include "computational_graph_builder.h"
#include "passes/block_graph_pass/memory_reuse/global_memory_reuse.h"
#include "passes/block_graph_pass/memory_reuse/ordered_memory_reuse.h"

namespace npu::tile_fwk {
namespace {
auto GraphSnapshot(const OrderedReuseGraph& graph)
{
    return std::make_tuple(graph.operations_, graph.indices_, graph.indegrees_, graph.successors_, graph.ancestors_,
                           graph.calls_, graph.leftMatch_, graph.rightMatch_, graph.antichain_, graph.matched_);
}

// Independent oracle: Floyd closure and exhaustive enumeration of all CALL subsets.
size_t BruteWidth(size_t count, const std::vector<std::pair<size_t, size_t>>& edges)
{
    std::vector<std::vector<bool>> reaches(count, std::vector<bool>(count, false));
    for (const auto& [from, to] : edges) {
        reaches[from][to] = true;
    }
    for (size_t via = 0; via < count; ++via) {
        for (size_t from = 0; from < count; ++from) {
            for (size_t to = 0; to < count; ++to) {
                reaches[from][to] = reaches[from][to] || (reaches[from][via] && reaches[via][to]);
            }
        }
    }
    size_t width = 0;
    for (size_t mask = 0; mask < (size_t{1} << count); ++mask) {
        size_t size = 0;
        bool independent = true;
        for (size_t a = 0; a < count; ++a) {
            if ((mask & (size_t{1} << a)) == 0) {
                continue;
            }
            ++size;
            for (size_t b = 0; b < a; ++b) {
                if ((mask & (size_t{1} << b)) != 0 && (reaches[a][b] || reaches[b][a])) {
                    independent = false;
                }
            }
        }
        if (independent) {
            width = std::max(size, width);
        }
    }
    return width;
}

struct OrderedReuseTokenSnapshot {
    explicit OrderedReuseTokenSnapshot(Function& function) : dependencies(function.GetVarDependency())
    {
        for (auto& op : function.Operations(false)) {
            inputs.emplace(&op, op.tokens_);
            outputs.emplace(&op, op.result_token_);
        }
    }

    void ExpectExistingDependenciesUnchanged(Function& function) const
    {
        const auto& current = function.GetVarDependency();
        for (const auto& [token, entry] : dependencies.GetAllDependencies()) {
            EXPECT_TRUE(current.HasDependency(token));
            EXPECT_EQ(current.GetProducers(token), entry.producers);
            EXPECT_EQ(current.GetConsumers(token), entry.consumers);
        }
    }

    void ExpectUnchanged(Function& function) const
    {
        EXPECT_EQ(function.GetVarDependency().Size(), dependencies.Size());
        ExpectExistingDependenciesUnchanged(function);
        for (auto& op : function.Operations(false)) {
            EXPECT_EQ(op.tokens_, inputs.at(&op));
            EXPECT_EQ(op.result_token_, outputs.at(&op));
        }
    }

    VarDependency dependencies;
    std::unordered_map<Operation*, std::vector<ir::VarPtr>> inputs;
    std::unordered_map<Operation*, std::vector<ir::VarPtr>> outputs;
};

void ExpectNoOrderedReuseControlEdges(Function& function)
{
    for (auto& op : function.Operations(false)) {
        EXPECT_TRUE(op.GetInCtrlOperations().empty());
        EXPECT_TRUE(op.GetOutCtrlOperations().empty());
    }
}
} // namespace

class TestOrderedMemoryReuse : public ::testing::Test {
public:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }

    void BindCalls(ComputationalGraphBuilder& builder, Function* callee = nullptr)
    {
        auto* function = builder.GetFunction();
        function->rootFunc_ = function;
        ops = function->Operations(false).DuplicatedOpList();
        if (callee == nullptr) {
            callee = function;
        }
        const auto hash = callee->ComputeHash();
        for (Operation* op : ops) {
            if (op->GetOpcode() != Opcode::OP_CALL) {
                continue;
            }
            op->SetOpAttribute(std::make_shared<CallOpAttribute>(hash, std::vector<std::vector<SymbolicScalar>>{},
                                                                 callee->GetMagicName()));
        }
    }

    void Independent(ComputationalGraphBuilder& builder, size_t count)
    {
        for (size_t i = 0; i < count; ++i) {
            const std::string key = std::to_string(i);
            ASSERT_TRUE(builder.AddTensors(DataType::DT_FP32, {16, 16}, {"i" + key, "o" + key}));
            ASSERT_TRUE(builder.AddOp(Opcode::OP_CALL, {"i" + key}, {"o" + key}, key, true));
        }
        BindCalls(builder);
    }

    void TwoChains(ComputationalGraphBuilder& builder)
    {
        ASSERT_TRUE(builder.AddTensors(DataType::DT_FP32, {16, 16}, {"i0", "o0", "i1", "o1", "r0", "r1"}));
        ASSERT_TRUE(builder.AddOps({Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL},
                                   {{"i0"}, {"o0"}, {"i1"}, {"o1"}}, {{"o0"}, {"r0"}, {"o1"}, {"r1"}},
                                   {"0", "1", "2", "3"}, true));
        ASSERT_TRUE(builder.SetInCast({"i0", "i1"}));
        ASSERT_TRUE(builder.SetOutCast({"r0", "r1"}));
        BindCalls(builder);
    }

    TensorsDesc Desc(ComputationalGraphBuilder& builder, size_t index, const std::string& name = "")
    {
        TensorsDesc desc(builder.GetFunction());
        auto tensor = builder.GetTensor(name.empty() ? "o" + std::to_string(index) : name);
        tensor->storage_ = std::make_shared<Storage>(MemoryType::MEM_WORKSPACE, tensor->GetRawMagic(), 1024);
        desc.tensors.insert(tensor);
        desc.orderedReuse.producerOps = {ops[index]};
        desc.orderedReuse.lifetimeEndOps = {ops[index]};
        desc.consumerOpIdxs = {index};
        desc.orderedReuse.firstProducerOpIdx = desc.orderedReuse.lastUseOpIdx = index;
        return desc;
    }

    void ExpectOrderingRejected(ComputationalGraphBuilder& builder, const TensorsDesc& previous,
                                const TensorsDesc& next)
    {
        TensorBucket bucket;
        ASSERT_TRUE(bucket.AddTensorGroup(previous));
        bucket.orderedReuse_.Update(previous.orderedReuse);
        ASSERT_TRUE(bucket.orderedReuse_.HasNonOverlappingOrderedInterval(next.orderedReuse));
        OrderedMemoryReuse policy(1);
        auto& function = *builder.GetFunction();
        ASSERT_TRUE(policy.graph_.Init(function));
        const auto graphBefore = GraphSnapshot(policy.graph_);
        const OrderedReuseTokenSnapshot tokensBefore(function);
        EXPECT_FALSE(policy.TryAddDependencies(bucket, next));
        EXPECT_EQ(GraphSnapshot(policy.graph_), graphBefore);
        EXPECT_TRUE(policy.addedConsumers_.empty());
        tokensBefore.ExpectUnchanged(function);
    }

    std::vector<Operation*> ops;
};

TEST_F(TestOrderedMemoryReuse, ExhaustiveDagWidthAndIncrementalAcceptanceMatchIndependentOracle)
{
    OrderedReuseGraph graph;
    for (size_t count = 0; count <= 5; ++count) {
        std::vector<std::pair<size_t, size_t>> possible;
        for (size_t from = 0; from < count; ++from) {
            for (size_t to = from + 1; to < count; ++to) {
                possible.emplace_back(from, to);
            }
        }
        for (size_t mask = 0; mask < (size_t{1} << possible.size()); ++mask) {
            ComputationalGraphBuilder builder;
            Independent(builder, count);
            std::vector<std::pair<size_t, size_t>> edges;
            for (size_t i = 0; i < possible.size(); ++i) {
                if ((mask & (size_t{1} << i)) != 0) {
                    auto [from, to] = possible[i];
                    ops[to]->AddInCtrlOperation(*ops[from]);
                    ops[from]->AddOutCtrlOperation(*ops[to]);
                    edges.push_back(possible[i]);
                }
            }
            ASSERT_TRUE(graph.Init(*builder.GetFunction()));
            ASSERT_EQ(graph.Width(), BruteWidth(count, edges)) << "n=" << count << " mask=" << mask;
            if (count <= 4 && count > 1) {
                for (const auto& [from, to] : possible) {
                    auto trialEdges = edges;
                    trialEdges.emplace_back(from, to);
                    const size_t trialWidth = BruteWidth(count, trialEdges);
                    for (size_t floor = 1; floor <= count; ++floor) {
                        auto trial = graph;
                        const auto before = GraphSnapshot(trial);
                        ASSERT_EQ(trial.TryOrder({ops[from]}, {ops[to]}, floor), trialWidth >= floor);
                        if (trialWidth >= floor) {
                            EXPECT_EQ(trial.Width(), trialWidth);
                        } else {
                            EXPECT_EQ(GraphSnapshot(trial), before);
                        }
                    }
                }
            }
        }
    }
}

TEST_F(TestOrderedMemoryReuse, BatchedAndCumulativeOrderingPreservesFloorAndTransactionalState)
{
    ComputationalGraphBuilder builder;
    Independent(builder, 4);
    OrderedReuseGraph graph;
    ASSERT_TRUE(graph.Init(*builder.GetFunction()));
    const auto before = GraphSnapshot(graph);
    EXPECT_FALSE(graph.TryOrder({ops[0], ops[1]}, {ops[2], ops[3]}, 3));
    EXPECT_EQ(GraphSnapshot(graph), before);
    ASSERT_TRUE(graph.TryOrder({ops[0]}, {ops[1]}, 3));
    const auto first = GraphSnapshot(graph);
    EXPECT_FALSE(graph.TryOrder({ops[1]}, {ops[2]}, 3));
    EXPECT_EQ(GraphSnapshot(graph), first);
    EXPECT_FALSE(graph.TryOrder({ops[1]}, {ops[0]}, 1));
    EXPECT_EQ(GraphSnapshot(graph), first);

    // Force a valid old maximum witness broken by 0->2. {1,2,3} remains valid.
    graph.antichain_ = {0, 2, 3};
    ASSERT_TRUE(graph.TryOrder({ops[0]}, {ops[2]}, 3));
    EXPECT_EQ(graph.Width(), 3U);
    ASSERT_TRUE(graph.TryOrder({ops[0]}, {ops[3]}, 3));
    EXPECT_EQ(graph.Width(), 3U);
}

TEST_F(TestOrderedMemoryReuse, IncludesAllFourDependenciesAndNonCallBridges)
{
    ComputationalGraphBuilder builder;
    ASSERT_TRUE(builder.AddTensors(DataType::DT_FP32, {16}, {"a", "b", "c", "d", "e", "f"}));
    ASSERT_TRUE(builder.AddOps({Opcode::OP_CALL, Opcode::OP_EXP, Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL},
                               {{"a"}, {"b"}, {"c"}, {"a"}, {"a"}}, {{"b"}, {"c"}, {"d"}, {"e"}, {"f"}},
                               {"0", "bridge", "1", "2", "3"}, true));
    Function& function = *builder.GetFunction();
    auto token = IRBuilder().CreateTokenVar(builder.GetOp("1")->GetSpan());
    builder.GetOp("1")->result_token_ = {token};
    builder.GetOp("2")->tokens_ = {token};
    // The same edge can occur in multiple dependency kinds; count its indegree once.
    builder.GetOp("2")->AddInCtrlOperation(*builder.GetOp("1"));
    builder.GetOp("3")->AddDependOperand(builder.GetTensor("e"));
    function.RefreshVarDependency();
    OrderedReuseGraph graph;
    ASSERT_TRUE(graph.Init(function));
    EXPECT_EQ(graph.calls_.size(), 4U);
    EXPECT_EQ(graph.Width(), 1U);
    builder.GetOp("0")->AddInCtrlOperation(*builder.GetOp("3"));
    EXPECT_FALSE(graph.Init(function));
}

TEST_F(TestOrderedMemoryReuse, CrossedProducerEndPathsAreRejectedWithoutMutation)
{
    ComputationalGraphBuilder builder;
    Independent(builder, 4);
    // ends={0,1}, producers={2,3}; reverse paths cross the two sets.
    ops[1]->AddInCtrlOperation(*ops[2]);
    ops[2]->AddOutCtrlOperation(*ops[1]);
    ops[0]->AddInCtrlOperation(*ops[3]);
    ops[3]->AddOutCtrlOperation(*ops[0]);
    OrderedReuseGraph graph;
    ASSERT_TRUE(graph.Init(*builder.GetFunction()));
    ASSERT_EQ(graph.Width(), 2U);
    const auto before = GraphSnapshot(graph);
    const OrderedReuseTokenSnapshot tokensBefore(*builder.GetFunction());
    EXPECT_FALSE(graph.TryOrder({ops[0], ops[1]}, {ops[2], ops[3]}, 1));
    EXPECT_EQ(GraphSnapshot(graph), before);
    tokensBefore.ExpectUnchanged(*builder.GetFunction());
    // Rejection must not prevent a later acyclic order: 3 -> 0 -> 2 -> 1.
    ASSERT_TRUE(graph.TryOrder({ops[0]}, {ops[2]}, 1));
    EXPECT_EQ(graph.Width(), 1U);
    tokensBefore.ExpectUnchanged(*builder.GetFunction());
}

TEST_F(TestOrderedMemoryReuse, RejectsForeignAndSelfEndpointsWithoutMutation)
{
    ComputationalGraphBuilder builder;
    Independent(builder, 2);
    OrderedReuseGraph graph;
    ASSERT_TRUE(graph.Init(*builder.GetFunction()));
    const auto before = GraphSnapshot(graph);
    Function foreign(Program::GetInstance(), "foreign", "foreign", builder.GetFunction());
    ComputationalGraphBuilder other(&foreign);
    ASSERT_TRUE(other.AddTensors(DataType::DT_FP32, {16}, {"x", "y"}));
    ASSERT_TRUE(other.AddOp(Opcode::OP_CALL, {"x"}, {"y"}, "foreign", true));
    const OrderedReuseTokenSnapshot tokensBefore(*builder.GetFunction());
    const OrderedReuseTokenSnapshot foreignBefore(foreign);
    EXPECT_FALSE(graph.TryOrder({ops[0]}, {ops[0]}, 1));
    EXPECT_FALSE(graph.TryOrder({ops[0]}, {nullptr}, 1));
    EXPECT_FALSE(graph.TryOrder({ops[0]}, {other.GetOp("foreign")}, 1));
    EXPECT_FALSE(graph.TryOrder({other.GetOp("foreign")}, {ops[1]}, 1));
    EXPECT_EQ(GraphSnapshot(graph), before);
    tokensBefore.ExpectUnchanged(*builder.GetFunction());
    foreignBefore.ExpectUnchanged(foreign);
    ops[0]->inputCtrlOps.insert(other.GetOp("foreign"));
    EXPECT_FALSE(graph.Init(*builder.GetFunction()));
}

TEST_F(TestOrderedMemoryReuse, AllocatorEnforcesActualFloorAndSkipsAtOrAboveInitialWidth)
{
    for (int64_t requested : {-1, 0, 1, 2, 3, 7, 8, 16}) {
        SCOPED_TRACE(requested);
        ComputationalGraphBuilder builder;
        Independent(builder, 8);
        Allocator allocator(builder.GetFunction());
        allocator.orderedReuse_.minimumParallelism = requested;
        for (size_t i = 0; i < ops.size(); ++i) {
            allocator.storageNeedToAllocate_.push_back(Desc(builder, i));
        }
        const OrderedReuseTokenSnapshot before(*builder.GetFunction());
        ASSERT_EQ(allocator.Allocate(), SUCCESS);
        OrderedReuseGraph actual;
        ASSERT_TRUE(actual.Init(*builder.GetFunction()));
        const size_t expected = requested < 0 || requested >= 8 ? 8 : std::max<int64_t>(1, requested);
        EXPECT_EQ(actual.Width(), expected) << requested;
        EXPECT_EQ(allocator.size_, expected * 1024) << requested;
        if (requested < 0 || requested >= 8) {
            before.ExpectUnchanged(*builder.GetFunction());
        }
    }
}

TEST_F(TestOrderedMemoryReuse, FailedBestFitContinuesToOtherBucketWithoutLeakingItsEdges)
{
    ComputationalGraphBuilder builder;
    Independent(builder, 5);
    auto a = Desc(builder, 0);
    a.orderedReuse.lifetimeEndOps.insert(ops[1]);
    a.orderedReuse.lastUseOpIdx = 1;
    auto e = Desc(builder, 2);
    (*e.tensors.begin())->storage_->length_ = 2048;
    auto next = Desc(builder, 3);
    next.orderedReuse.producerOps.insert(ops[4]);
    TensorBucket first, second;
    ASSERT_TRUE(first.AddTensorGroup(a));
    ASSERT_TRUE(second.AddTensorGroup(e));
    first.orderedReuse_.Update(a.orderedReuse);
    second.orderedReuse_.Update(e.orderedReuse);
    Allocator allocator(builder.GetFunction());
    OrderedMemoryReuse policy(4);
    ASSERT_TRUE(policy.graph_.Init(*builder.GetFunction()));
    allocator.orderedReuse_.active = &policy;
    allocator.buckets_ = {first, second};
    allocator.bucketsIdxToSize_ = {{0, 1024}, {1, 2048}};
    auto& selected = allocator.GetBestFitBucket(next);
    EXPECT_EQ(&selected, &allocator.buckets_[1]);
    EXPECT_EQ(policy.graph_.Width(), 4U);
    EXPECT_FALSE(policy.graph_.AllReachable({ops[0]}, {ops[3]}));
    EXPECT_TRUE(policy.graph_.AllReachable({ops[2]}, {ops[3], ops[4]}));
    EXPECT_EQ(policy.addedConsumers_.size(), 1U);
    EXPECT_EQ(policy.addedConsumers_.at(ops[2]).size(), 2U);
}

TEST_F(TestOrderedMemoryReuse, InvalidCallPreflightPreservesStorageAndTokens)
{
    ComputationalGraphBuilder builder;
    Independent(builder, 3);
    Allocator allocator(builder.GetFunction());
    allocator.orderedReuse_.minimumParallelism = 1;
    for (size_t i = 0; i < ops.size(); ++i) {
        allocator.storageNeedToAllocate_.push_back(Desc(builder, i));
    }
    auto tensor = builder.GetTensor("o0");
    tensor->storage_->start_ = 12345;
    tensor->storage_->id_ = 17;
    ops[2]->GetOpAttribute().reset();
    const OrderedReuseTokenSnapshot before(*builder.GetFunction());
    ASSERT_EQ(allocator.Allocate(), FAILED);
    EXPECT_EQ(tensor->storage_->start_, 12345U);
    EXPECT_EQ(tensor->storage_->length_, 1024U);
    EXPECT_EQ(tensor->storage_->id_, 17);
    before.ExpectUnchanged(*builder.GetFunction());
}

TEST_F(TestOrderedMemoryReuse, InvalidStoragePreflightPreservesState)
{
    for (int invalidKind = 0; invalidKind < 3; ++invalidKind) {
        SCOPED_TRACE(invalidKind);
        ComputationalGraphBuilder builder;
        Independent(builder, 2);
        Allocator allocator(builder.GetFunction());
        allocator.orderedReuse_.minimumParallelism = 1;
        auto invalid = Desc(builder, 0);
        auto untouched = Desc(builder, 1);
        auto tensor = builder.GetTensor("o1");
        tensor->storage_->start_ = 12345;
        tensor->storage_->id_ = 17;
        if (invalidKind == 0) {
            invalid.tensors.clear();
        } else if (invalidKind == 1) {
            invalid.tensors = {nullptr};
        } else {
            builder.GetTensor("o0")->storage_.reset();
        }
        allocator.storageNeedToAllocate_ = {invalid, untouched};
        const OrderedReuseTokenSnapshot before(*builder.GetFunction());
        ASSERT_EQ(allocator.Allocate(), FAILED);
        EXPECT_TRUE(allocator.buckets_.empty());
        EXPECT_EQ(tensor->storage_->start_, 12345U);
        EXPECT_EQ(tensor->storage_->length_, 1024U);
        EXPECT_EQ(tensor->storage_->id_, 17);
        before.ExpectUnchanged(*builder.GetFunction());
    }
}

TEST_F(TestOrderedMemoryReuse, InvalidInvokeInfoPreflightPreservesStorageAndTokens)
{
    for (int invalidKind = 0; invalidKind < 3; ++invalidKind) {
        SCOPED_TRACE(invalidKind);
        ComputationalGraphBuilder builder;
        Independent(builder, 3);
        Allocator allocator(builder.GetFunction());
        allocator.orderedReuse_.minimumParallelism = 1;
        for (size_t i = 0; i < ops.size(); ++i) {
            allocator.storageNeedToAllocate_.push_back(Desc(builder, i));
        }
        auto tensor = builder.GetTensor("o0");
        tensor->storage_->start_ = 12345;
        tensor->storage_->id_ = 17;
        auto attr = std::dynamic_pointer_cast<CallOpAttribute>(ops[2]->GetOpAttribute());
        ASSERT_NE(attr, nullptr);
        ASSERT_NE(attr->invokeInfo_, nullptr);
        if (invalidKind == 0) {
            attr->invokeInfo_.reset();
        } else if (invalidKind == 1) {
            attr->invokeInfo_->incastTensorParamList_.resize(ops[2]->iOperand.size() + 1);
        } else {
            attr->invokeInfo_->outcastTensorParamList_.resize(ops[2]->oOperand.size() + 1);
        }
        const OrderedReuseTokenSnapshot before(*builder.GetFunction());
        ASSERT_EQ(allocator.Allocate(), FAILED);
        EXPECT_TRUE(allocator.buckets_.empty());
        EXPECT_EQ(tensor->storage_->start_, 12345U);
        EXPECT_EQ(tensor->storage_->length_, 1024U);
        EXPECT_EQ(tensor->storage_->id_, 17);
        before.ExpectUnchanged(*builder.GetFunction());
    }
}

TEST_F(TestOrderedMemoryReuse, EqualBytePlanDoesNotMaterializeOrdering)
{
    ComputationalGraphBuilder builder;
    Independent(builder, 2);
    Allocator allocator(builder.GetFunction());
    allocator.orderedReuse_.minimumParallelism = 1;
    auto first = Desc(builder, 0);
    auto second = Desc(builder, 1);
    // No new dependencies can make a zero-sized allocation strictly smaller.
    (*first.tensors.begin())->storage_->length_ = 0;
    (*second.tensors.begin())->storage_->length_ = 0;
    allocator.storageNeedToAllocate_ = {first, second};
    const OrderedReuseTokenSnapshot before(*builder.GetFunction());
    ASSERT_EQ(allocator.Allocate(), SUCCESS);
    before.ExpectUnchanged(*builder.GetFunction());
    EXPECT_EQ(allocator.buckets_.size(), 2U);
    const std::unordered_map<int, int> expectedBuckets = {{builder.GetTensor("o0")->GetMagic(), 0},
                                                          {builder.GetTensor("o1")->GetMagic(), 1}};
    EXPECT_EQ(allocator.tensorMagicToBucketIdx_, expectedBuckets);
    EXPECT_EQ(allocator.bucketsIdxToSize_, (std::unordered_map<int, int64_t>{{0, 1024}, {1, 1024}}));
}

TEST_F(TestOrderedMemoryReuse, CanReuseBucketOutsidePredecessorWalkByGlobalTopology)
{
    for (int64_t minimum : {-1, 1}) {
        SCOPED_TRACE(minimum);
        ComputationalGraphBuilder builder;
        ASSERT_TRUE(builder.AddTensors(DataType::DT_FP32, {16, 16}, {"input", "old", "bridge", "next"}));
        ASSERT_TRUE(builder.AddOps({Opcode::OP_CALL, Opcode::OP_CALL}, {{"input"}, {"bridge"}},
                                   {{"old", "bridge"}, {"next"}}, {"0", "1"}, true));
        BindCalls(builder);
        auto first = Desc(builder, 0, "old");
        auto next = Desc(builder, 1, "next");
        // DATA orders CALL0 before CALL1, but the walk via sibling output "bridge" never visits "old".
        next.connectionOpsBitmap.SetBit(0);
        Allocator allocator(builder.GetFunction());
        allocator.orderedReuse_.minimumParallelism = minimum;
        allocator.storageNeedToAllocate_ = {first, next};
        const OrderedReuseTokenSnapshot before(*builder.GetFunction());
        ASSERT_EQ(allocator.Allocate(), SUCCESS);
        EXPECT_EQ(allocator.buckets_.size(), minimum < 0 ? 2U : 1U);
        EXPECT_EQ(allocator.size_, minimum < 0 ? 2048U : 1024U);
        EXPECT_EQ(builder.GetTensor("next")->storage_->start_, minimum < 0 ? 1024U : 0U);
        before.ExpectUnchanged(*builder.GetFunction());
    }
}

TEST_F(TestOrderedMemoryReuse, NonCallEndpointsFollowConfiguredReuseMode)
{
    for (int64_t minimum : {-1, 1}) {
        SCOPED_TRACE(minimum);
        ComputationalGraphBuilder builder;
        ASSERT_TRUE(builder.AddTensors(DataType::DT_FP32, {16, 16}, {"input", "call", "noncall", "output"}));
        ASSERT_TRUE(builder.AddOps({Opcode::OP_CALL, Opcode::OP_EXP, Opcode::OP_CALL},
                                   {{"input"}, {"call"}, {"noncall"}}, {{"call"}, {"noncall"}, {"output"}},
                                   {"0", "1", "2"}, true));
        ASSERT_TRUE(builder.SetInCast({"input"}));
        ASSERT_TRUE(builder.SetOutCast({"output"}));
        BindCalls(builder);
        Allocator allocator(builder.GetFunction());
        config::SetPassOption<int64_t>(GM_ORDERED_REUSE_MIN_PARALLELISM, minimum);
        allocator.Init();
        ASSERT_EQ(allocator.storageNeedToAllocate_.size(), 2U);
        const auto& call = allocator.storageNeedToAllocate_[0];
        const auto& noncall = allocator.storageNeedToAllocate_[1];
        const bool enabled = minimum >= 0;
        EXPECT_EQ(call.consumerOpIdxs, enabled ? std::unordered_set<uint64_t>{1} : std::unordered_set<uint64_t>{});
        EXPECT_EQ(noncall.consumerOpIdxs, std::unordered_set<uint64_t>{2});
        LargeBitmap connections(ops.size());
        if (enabled) {
            connections.SetBit(0);
            EXPECT_EQ(call.orderedReuse.producerOps, std::unordered_set<Operation*>{ops[0]});
            EXPECT_EQ(call.orderedReuse.lifetimeEndOps, std::unordered_set<Operation*>({ops[0], ops[1]}));
            EXPECT_EQ(noncall.orderedReuse.producerOps, std::unordered_set<Operation*>{ops[1]});
            EXPECT_EQ(noncall.orderedReuse.lifetimeEndOps, std::unordered_set<Operation*>({ops[1], ops[2]}));
        } else {
            // Master ignores non-CALL producers, leaving the producer bitmap unchanged.
            connections.SetValues(0xFFFFFFFFFFFFFFFF);
            for (const auto& desc : allocator.storageNeedToAllocate_) {
                EXPECT_TRUE(desc.orderedReuse.producerOps.empty());
                EXPECT_TRUE(desc.orderedReuse.lifetimeEndOps.empty());
                EXPECT_EQ(desc.orderedReuse.firstProducerOpIdx, ConnectionMatrix::INVALID_INDEX);
                EXPECT_EQ(desc.orderedReuse.lastUseOpIdx, ConnectionMatrix::INVALID_INDEX);
            }
        }
        EXPECT_EQ(noncall.connectionOpsBitmap, connections);
        const OrderedReuseTokenSnapshot before(*builder.GetFunction());
        ASSERT_EQ(allocator.Allocate(), SUCCESS);
        EXPECT_EQ(allocator.buckets_.size(), enabled ? 2U : 1U);
        EXPECT_EQ(allocator.size_, enabled ? 2048U : 1024U);
        EXPECT_EQ(builder.GetTensor("noncall")->storage_->start_, enabled ? 1024U : 0U);
        before.ExpectUnchanged(*builder.GetFunction());
    }
}

TEST_F(TestOrderedMemoryReuse, GlobalTopologyReuseRequiresPreviousConsumersToFinish)
{
    TensorBucket bucket;
    bucket.consumerOpIdxs_ = {3, 5};
    LargeBitmap connections(6);
    connections.SetBit(3);
    EXPECT_FALSE(bucket.HasTopoDependency(connections));
    connections.SetBit(5);
    EXPECT_TRUE(bucket.HasTopoDependency(connections));
    EXPECT_TRUE(TensorBucket().HasTopoDependency(connections));
}

TEST_F(TestOrderedMemoryReuse, OrderedReuseRejectsTokenCycleWithoutAddingPartialEdges)
{
    ComputationalGraphBuilder builder;
    Independent(builder, 5);
    auto firstToken = IRContext::Get().MakeSemanticToken("first", ir::TokenKind::WRITE, ir::Span::Unknown());
    auto secondToken = IRContext::Get().MakeSemanticToken("second", ir::TokenKind::WRITE, ir::Span::Unknown());
    ops[4]->result_token_ = {firstToken};
    ops[2]->tokens_ = {firstToken};
    ops[2]->result_token_ = {secondToken};
    ops[1]->tokens_ = {secondToken};
    builder.GetFunction()->RefreshVarDependency();
    auto previous = Desc(builder, 0);
    previous.orderedReuse.lifetimeEndOps.insert(ops[1]);
    previous.orderedReuse.lastUseOpIdx = 1;
    auto next = Desc(builder, 3);
    next.orderedReuse.producerOps.insert(ops[4]);
    ExpectOrderingRejected(builder, previous, next);
    ExpectNoOrderedReuseControlEdges(*builder.GetFunction());
}

TEST_F(TestOrderedMemoryReuse, OrderedReuseRejectsControlAndDependOperandCycle)
{
    ComputationalGraphBuilder builder;
    Independent(builder, 3);
    ops[1]->AddDependOperand(builder.GetTensor("o2"));
    ops[0]->AddInCtrlOperation(*ops[1]);
    ops[1]->AddOutCtrlOperation(*ops[0]);
    ExpectOrderingRejected(builder, Desc(builder, 0), Desc(builder, 2));
    EXPECT_EQ(ops[0]->GetInCtrlOperations(), std::unordered_set<Operation*>{ops[1]});
    EXPECT_EQ(ops[1]->GetOutCtrlOperations(), std::unordered_set<Operation*>{ops[0]});
}

TEST_F(TestOrderedMemoryReuse, OrderedReuseAfterTopologyReuseKeepsTransitiveOrdering)
{
    ComputationalGraphBuilder builder;
    ASSERT_TRUE(builder.AddTensors(DataType::DT_FP32, {16, 16}, {"a", "old", "bridge", "middle", "out", "b", "next"}));
    ASSERT_TRUE(builder.AddOps({Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL},
                               {{"a"}, {"old"}, {"bridge"}, {"middle"}, {"b"}},
                               {{"old"}, {"bridge"}, {"middle"}, {"out"}, {"next"}}, {"0", "1", "2", "3", "4"}, true));
    BindCalls(builder);
    auto old = Desc(builder, 0, "old");
    old.consumerOpIdxs = {1};
    old.orderedReuse.lifetimeEndOps.insert(ops[1]);
    old.orderedReuse.lastUseOpIdx = 1;
    auto middle = Desc(builder, 2, "middle");
    middle.consumerOpIdxs = {3};
    middle.connectionOpsBitmap.SetBit(0);
    middle.connectionOpsBitmap.SetBit(1);
    middle.orderedReuse.lifetimeEndOps.insert(ops[3]);
    middle.orderedReuse.lastUseOpIdx = 3;
    auto next = Desc(builder, 4, "next");
    Allocator allocator(builder.GetFunction());
    allocator.orderedReuse_.minimumParallelism = 1;
    allocator.storageNeedToAllocate_ = {old, middle, next};
    ASSERT_EQ(allocator.Allocate(), SUCCESS);
    ASSERT_EQ(allocator.buckets_.size(), 1U);
    EXPECT_EQ(allocator.size_, 1024U);
    EXPECT_TRUE(ops[2]->ProducerOpsByToken().empty());
    EXPECT_EQ(ops[4]->ProducerOpsByToken(), std::unordered_set<Operation*>({ops[2], ops[3]}));
    EXPECT_EQ(builder.GetTensor("old")->storage_->id_, builder.GetTensor("next")->storage_->id_);
    EXPECT_EQ(builder.GetTensor("middle")->storage_->id_, builder.GetTensor("next")->storage_->id_);
    const std::unordered_map<int, int> expectedBuckets = {{builder.GetTensor("old")->GetMagic(), 0},
                                                          {builder.GetTensor("middle")->GetMagic(), 0},
                                                          {builder.GetTensor("next")->GetMagic(), 0}};
    EXPECT_EQ(allocator.tensorMagicToBucketIdx_, expectedBuckets);
    EXPECT_EQ(allocator.bucketsIdxToSize_, (std::unordered_map<int, int64_t>{{0, 1024}}));
    ExpectNoOrderedReuseControlEdges(*builder.GetFunction());
}

TEST_F(TestOrderedMemoryReuse, OrderedReuseDisabledForUnknownCallee)
{
    ComputationalGraphBuilder builder;
    TwoChains(builder);
    ASSERT_EQ(Program::GetInstance().TryHitCahce(ops[0]->GetCalleeHash()), std::nullopt);
    Allocator allocator(builder.GetFunction());
    config::SetPassOption<int64_t>(GM_ORDERED_REUSE_MIN_PARALLELISM, 1);
    allocator.Init();
    ASSERT_EQ(allocator.orderedReuse_.minimumParallelism, 1);
    ASSERT_EQ(allocator.storageNeedToAllocate_.size(), 2U);
    for (const auto& desc : allocator.storageNeedToAllocate_) {
        EXPECT_FALSE(desc.orderedReuse.supportsOrderedReuse);
    }
    const OrderedReuseTokenSnapshot before(*builder.GetFunction());
    ASSERT_EQ(allocator.Allocate(), SUCCESS);
    EXPECT_EQ(allocator.buckets_.size(), 2U);
    before.ExpectUnchanged(*builder.GetFunction());
}

TEST_F(TestOrderedMemoryReuse, MaterializationAndRollbackPreserveAllEndpointsAndExistingTokens)
{
    ComputationalGraphBuilder builder;
    Independent(builder, 6);
    auto& function = *builder.GetFunction();
    auto shared = IRBuilder().CreateTokenVar(ops[0]->GetSpan());
    auto read = IRContext::Get().MakeSemanticToken("read", ir::TokenKind::READ, ir::Span::Unknown());
    auto write = IRContext::Get().MakeSemanticToken("write", ir::TokenKind::WRITE, ir::Span::Unknown());
    ops[0]->result_token_ = {shared, read};
    ops[1]->result_token_ = {write};
    ops[2]->result_token_ = {shared};
    ops[5]->tokens_ = {shared, read, write};
    function.RefreshVarDependency();
    const OrderedReuseTokenSnapshot before(function);
    auto previous = Desc(builder, 0);
    previous.orderedReuse.lifetimeEndOps.insert(ops[1]);
    previous.orderedReuse.lastUseOpIdx = 1;
    auto next = Desc(builder, 3);
    next.orderedReuse.producerOps.insert(ops[4]);
    TensorBucket bucket;
    ASSERT_TRUE(bucket.AddTensorGroup(previous));
    bucket.orderedReuse_.Update(previous.orderedReuse);
    OrderedMemoryReuse policy(1);
    ASSERT_TRUE(policy.graph_.Init(function));
    ASSERT_TRUE(policy.TryAddDependencies(bucket, next));
    ASSERT_TRUE(policy.Materialize(function));

    const auto& dependencies = function.GetVarDependency();
    EXPECT_EQ(dependencies.Size(), before.dependencies.Size() + 2U);
    before.ExpectExistingDependenciesUnchanged(function);
    std::unordered_set<ir::VarPtr> reuseTokens;
    for (auto* end : previous.orderedReuse.lifetimeEndOps) {
        ASSERT_EQ(end->result_token_.size(), before.outputs.at(end).size() + 1U);
        auto token = end->result_token_.back();
        EXPECT_TRUE(reuseTokens.emplace(token).second);
        auto type = std::dynamic_pointer_cast<const ir::TokenType>(token->GetType());
        ASSERT_NE(type, nullptr);
        EXPECT_EQ(type->kind_, ir::TokenKind::NORMAL);
        auto expected = before.outputs.at(end);
        expected.push_back(token);
        EXPECT_EQ(end->result_token_, expected);
        ASSERT_EQ(dependencies.GetProducers(token).size(), 1U);
        EXPECT_TRUE(dependencies.HasProducer(token, end->shared_from_this()));
        EXPECT_EQ(dependencies.GetConsumers(token).size(), next.orderedReuse.producerOps.size());
        for (auto* producer : next.orderedReuse.producerOps) {
            EXPECT_TRUE(dependencies.HasConsumer(token, producer->shared_from_this()));
        }
    }
    ASSERT_EQ(reuseTokens.size(), 2U);
    for (auto* producer : next.orderedReuse.producerOps) {
        EXPECT_EQ(producer->ProducerOpsByToken(), previous.orderedReuse.lifetimeEndOps);
        EXPECT_EQ(producer->tokens_.size(), reuseTokens.size());
        EXPECT_EQ(std::unordered_set<ir::VarPtr>(producer->tokens_.begin(), producer->tokens_.end()), reuseTokens);
    }
    for (auto* op : ops) {
        if (next.orderedReuse.producerOps.count(op) == 0) {
            EXPECT_EQ(op->tokens_, before.inputs.at(op));
        }
        if (previous.orderedReuse.lifetimeEndOps.count(op) == 0) {
            EXPECT_EQ(op->result_token_, before.outputs.at(op));
        }
    }
    ExpectNoOrderedReuseControlEdges(function);

    policy.Rollback(function);
    before.ExpectUnchanged(function);
    // Re-emitting these tokens without their planned edges must fail actual-IR validation.
    ASSERT_TRUE(policy.graph_.Init(function));
    ASSERT_FALSE(policy.Materialize(function));
    policy.Rollback(function);
    before.ExpectUnchanged(function);
}

TEST_F(TestOrderedMemoryReuse, PredecessorWalkHonorsMinimumParallelism)
{
    for (int64_t minimum : {-1, 1}) {
        SCOPED_TRACE(minimum);
        ComputationalGraphBuilder builder;
        ASSERT_TRUE(builder.AddTensors(DataType::DT_FP32, {16, 16}, {"t0", "t1", "t2", "t3", "t4"}));
        ASSERT_TRUE(builder.AddOps({Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL, Opcode::OP_CALL},
                                   {{"t0"}, {"t1"}, {"t1"}, {"t2"}}, {{"t1"}, {"t2"}, {"t3"}, {"t4"}},
                                   {"0", "1", "2", "3"}, true));
        BindCalls(builder);
        auto previous = Desc(builder, 0, "t1");
        previous.consumerOpIdxs = {1, 2};
        previous.orderedReuse.lifetimeEndOps = {ops[0], ops[1], ops[2]};
        previous.orderedReuse.lastUseOpIdx = 2;
        auto next = Desc(builder, 3, "t4");
        next.connectionOpsBitmap.SetBit(0);
        next.connectionOpsBitmap.SetBit(1);
        Allocator allocator(builder.GetFunction());
        allocator.orderedReuse_.minimumParallelism = minimum;
        allocator.storageNeedToAllocate_ = {previous, next};
        ASSERT_EQ(allocator.Allocate(), SUCCESS);
        const bool enabled = minimum == 1;
        EXPECT_EQ(allocator.buckets_.size(), enabled ? 1U : 2U);
        EXPECT_EQ(allocator.size_, enabled ? 1024U : 2048U);
        // CALL1 already reaches CALL3 through DATA; only CALL2 needs a new token.
        EXPECT_EQ(ops[3]->ProducerOpsByToken().count(ops[1]), 0U);
        EXPECT_EQ(ops[3]->ProducerOpsByToken().count(ops[2]), enabled ? 1U : 0U);
        EXPECT_EQ(builder.GetTensor("t1")->storage_->start_, 0U);
        EXPECT_EQ(builder.GetTensor("t4")->storage_->start_, enabled ? 0U : 1024U);
        ExpectNoOrderedReuseControlEdges(*builder.GetFunction());
    }
}

TEST_F(TestOrderedMemoryReuse, PassOptionWiresMinimumParallelism)
{
    const std::optional<int64_t> options[] = {std::nullopt, -1, 1};
    for (const auto minimum : options) {
        SCOPED_TRACE(minimum.has_value() ? std::to_string(*minimum) : "default");
        ComputationalGraphBuilder builder;
        TwoChains(builder);
        auto* function = builder.GetFunction();
        Function leaf(Program::GetInstance(), "ordinaryLeaf", "ordinaryLeaf", function);
        leaf.SetGraphType(GraphType::BLOCK_GRAPH);
        Program::GetInstance().GetFunctionCache().Insert(leaf.ComputeHash(), leaf);
        BindCalls(builder, &leaf);
        const bool enabled = minimum.value_or(-1) >= 0;
        {
            ConfigManagerNg::JitScopeGuard jitScope("ordered_reuse");
            if (minimum.has_value()) {
                config::SetPassOption<int64_t>(GM_ORDERED_REUSE_MIN_PARALLELISM, *minimum);
            }
            GlobalMemoryReuse pass;
            ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);
        }
        EXPECT_EQ(config::GetPassOption<int64_t>(GM_ORDERED_REUSE_MIN_PARALLELISM), -1);
        const auto first = builder.GetTensor("o0")->storage_;
        const auto second = builder.GetTensor("o1")->storage_;
        ASSERT_NE(first, nullptr);
        ASSERT_NE(second, nullptr);
        EXPECT_EQ(ops[2]->ProducerOpsByToken().count(ops[1]), enabled ? 1U : 0U);
        EXPECT_EQ(first->start_, 0U);
        EXPECT_EQ(second->start_, enabled ? 0U : 1024U);
        EXPECT_EQ(first->id_ == second->id_, enabled);
        ExpectNoOrderedReuseControlEdges(*function);
    }
}

TEST_F(TestOrderedMemoryReuse, CachesCalleeClassificationAndDisablesOrderingForCommunication)
{
    for (const auto& [minimum, sharedMemory] :
         {std::pair<int64_t, bool>{-1, false}, {-1, true}, {1, false}, {1, true}}) {
        SCOPED_TRACE(minimum);
        SCOPED_TRACE(sharedMemory);
        ComputationalGraphBuilder builder;
        TwoChains(builder);
        auto* function = builder.GetFunction();
        function->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
        Function leaf(Program::GetInstance(), "commLeaf", "commLeaf", function);
        leaf.rootFunc_ = function;
        ComputationalGraphBuilder leafBuilder(&leaf);
        ASSERT_TRUE(leafBuilder.AddTensors(DataType::DT_FP32, {16, 16}, {"a", "b", "out"}));
        ASSERT_TRUE(
            leafBuilder.AddOp(sharedMemory ? Opcode::OP_SHMEM_PUT : Opcode::OP_ADD, {"a", "b"}, {"out"}, "leaf"));
        ASSERT_TRUE(leafBuilder.SetInCast({"a"}));
        ASSERT_TRUE(leafBuilder.SetOutCast({"out"}));
        function->programs_.emplace(1, &leaf);
        leaf.SetGraphType(GraphType::BLOCK_GRAPH);
        Program::GetInstance().GetFunctionCache().Insert(leaf.ComputeHash(), leaf);
        BindCalls(builder, &leaf);
        Allocator allocator(function);
        config::SetPassOption<int64_t>(GM_ORDERED_REUSE_MIN_PARALLELISM, minimum);
        allocator.Init();
        ASSERT_EQ(allocator.orderedReuse_.minimumParallelism, minimum);
        ASSERT_EQ(allocator.storageNeedToAllocate_.size(), 2U);
        const auto& communication = allocator.orderedReuse_.communicationLeafCache;
        EXPECT_EQ(communication.count(&leaf), minimum < 0 ? 0U : 1U);
        if (minimum >= 0) {
            EXPECT_EQ(communication.at(&leaf), sharedMemory);
            EXPECT_LT(allocator.storageNeedToAllocate_[0].orderedReuse.lastUseOpIdx,
                      allocator.storageNeedToAllocate_[1].orderedReuse.firstProducerOpIdx);
        }
        for (const auto& desc : allocator.storageNeedToAllocate_) {
            EXPECT_EQ(desc.orderedReuse.supportsOrderedReuse, minimum < 0 || !sharedMemory);
        }
        const OrderedReuseTokenSnapshot before(*function);
        ASSERT_EQ(allocator.Allocate(), SUCCESS);
        const bool ordered = minimum >= 0 && !sharedMemory;
        EXPECT_EQ(allocator.buckets_.size(), ordered ? 1U : 2U);
        EXPECT_EQ(ops[2]->ProducerOpsByToken().count(ops[1]), ordered ? 1U : 0U);
        if (ordered) {
            before.ExpectExistingDependenciesUnchanged(*function);
        } else {
            before.ExpectUnchanged(*function);
        }
    }
}
} // namespace npu::tile_fwk
