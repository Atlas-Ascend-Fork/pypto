/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen/npu/cce_split.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <stdexcept>
#include <regex>

namespace npu::tile_fwk {
namespace {
CceSplitRegion NonSyncRegion(int magic, size_t bytes = 100)
{
    CceSplitRegion region;
    region.effect = CceSplitEffect::NON_SYNC_OP;
    region.code = "// op:" + std::to_string(magic) + "\n// " + std::string(bytes, 'x') + "\nconsume(tile);\n";
    region.declarations = {{"tile", "Tile tile(0, shape);\n"}};
    return region;
}

CceSplitRegion Effect(CceSplitEffect effect)
{
    CceSplitRegion region;
    region.effect = effect;
    region.code = "// op:0\n// synchronization\n";
    return region;
}

CceOriginalFunctionBody Body()
{
    CceOriginalFunctionBody body;
    body.preambleDefinitions = {{"shape", "int shape = 32;\n"}};
    body.regions = {NonSyncRegion(10), Effect(CceSplitEffect::FULL_BARRIER), NonSyncRegion(20)};
    return body;
}

CceSplitPlan Plan(const CceOriginalFunctionBody& body, int64_t threshold = 1)
{
    return PlanCceSplit(body, "#include \"device.h\"\n", "leaf",
                        "(Params* param, long GMStackBase, "
                        "long* hcclContext, Stats* taskStat)",
                        "leaf_parts.h", threshold);
}

// Inspect emitted source instead of retaining production diagnostic metadata.
std::vector<int> Operations(const CceGeneratedSegment& part)
{
    std::vector<int> result;
    const std::regex marker("// op:([0-9]+)");
    for (auto it = std::sregex_iterator(part.source.begin(), part.source.end(), marker); it != std::sregex_iterator();
         ++it) {
        result.push_back(std::stoi((*it)[1]));
    }
    return result;
}

size_t PreambleBytes(const CceGeneratedSegment& part, const CceOriginalFunctionBody& body)
{
    size_t bytes = 0;
    for (const auto& [name, code] : body.preambleDefinitions) {
        if (part.source.find(code) != std::string::npos) {
            bytes += code.size();
        }
    }
    return bytes;
}
} // namespace

TEST(CceSplitPlanner, DisabledAndExactThresholdKeepOriginal)
{
    EXPECT_TRUE(Plan(Body(), 0).parts.empty());
    EXPECT_TRUE(Plan(Body(), 3).parts.empty());
    EXPECT_THROW(Plan(Body(), -1), std::invalid_argument);
}

TEST(CceSplitPlanner, TopLevelOperationAllowsSplitAndOversize)
{
    auto plan = Plan(Body());
    ASSERT_EQ(plan.parts.size(), 2U);
    EXPECT_GT(plan.parts[0].source.size(), 1U);
    EXPECT_EQ(Operations(plan.parts[0]), (std::vector<int>{10}));
    EXPECT_EQ(Operations(plan.parts[1]), (std::vector<int>{0, 20}));
    EXPECT_LT(plan.entry.find("leaf_part0("), plan.entry.find("leaf_part1("));
    EXPECT_EQ(plan.header.find("consume"), std::string::npos);
    EXPECT_EQ(plan.parts[0].source.find("leaf_part1("), std::string::npos);
}

TEST(CceSplitPlanner, InternalPartsAreHiddenAndEntryRemainsVisible)
{
    const auto plan = Plan(Body());
    ASSERT_EQ(plan.parts.size(), 2U);
    for (const auto& part : plan.parts) {
        const std::string declaration = "extern \"C\" __aicore__ __attribute__((visibility(\"hidden\"))) void " +
                                        part.name + "(";
        EXPECT_NE(plan.header.find(declaration), std::string::npos);
        EXPECT_NE(plan.header.find("asm(\".hidden " + part.name + "\");"), std::string::npos);
        EXPECT_NE(part.source.find(declaration), std::string::npos);
        EXPECT_NE(plan.entry.find(part.name + "(param, GMStackBase, hcclContext, taskStat);"), std::string::npos);
    }
    EXPECT_NE(plan.entry.find("extern \"C\" [aicore] void leaf("), std::string::npos);
    EXPECT_EQ(plan.entry.find("visibility"), std::string::npos);
}

TEST(CceSplitPlanner, OrdinaryBarrierDoesNotBlockSplitting)
{
    auto body = Body();
    body.regions[1].effect = CceSplitEffect::BARRIER;
    auto plan = Plan(body);
    ASSERT_EQ(plan.parts.size(), 2U);
    EXPECT_EQ(Operations(plan.parts[0]), (std::vector<int>{10}));
    EXPECT_EQ(Operations(plan.parts[1]), (std::vector<int>{0, 20}));
}

TEST(CceSplitPlanner, SplitsWithoutAnyBarrier)
{
    auto body = Body();
    body.regions.erase(body.regions.begin() + 1);
    auto plan = Plan(body);
    ASSERT_EQ(plan.parts.size(), 2U);
    EXPECT_EQ(Operations(plan.parts[0]), (std::vector<int>{10}));
    EXPECT_EQ(Operations(plan.parts[1]), (std::vector<int>{20}));
    for (const auto& part : plan.parts) {
        EXPECT_EQ(part.source.find("pipe_barrier"), std::string::npos);
    }
}

TEST(CceSplitPlanner, EmptyBodyKeepsOriginal)
{
    CceOriginalFunctionBody body;
    auto plan = Plan(body);
    EXPECT_TRUE(plan.parts.empty());
}

TEST(CceSplitPlanner, DefaultRegionParticipatesWithoutOpcodeWhitelist)
{
    auto body = Body();
    CceSplitRegion region;
    region.code = "other_operation();\n";
    body.regions.back() = region;
    auto plan = Plan(body);
    ASSERT_EQ(plan.parts.size(), 2U);
    EXPECT_NE(plan.parts[1].source.find(region.code), std::string::npos);
}

TEST(CceSplitPlanner, ExternalStateProtectsSuffixWithoutRejectingPrefix)
{
    auto body = Body();
    body.regions.push_back(Effect(CceSplitEffect::EXTERNAL_STATE));
    body.regions.push_back(Effect(CceSplitEffect::FULL_BARRIER));
    body.regions.push_back(NonSyncRegion(30));
    auto plan = Plan(body);
    ASSERT_EQ(plan.parts.size(), 3U);
    EXPECT_EQ(Operations(plan.parts[1]), (std::vector<int>{0, 20}));
    EXPECT_EQ(Operations(plan.parts[2]), (std::vector<int>{0, 0, 30}));
    body.regions.insert(body.regions.begin(), Effect(CceSplitEffect::EXTERNAL_STATE));
    plan = Plan(body);
    EXPECT_TRUE(plan.parts.empty());
}

TEST(CceSplitPlanner, LoopRegionsRemainTogetherWithoutBarrier)
{
    auto body = Body();
    body.regions.erase(body.regions.begin() + 1);
    for (auto& region : body.regions) {
        region.allowCutAfter = false;
    }
    auto rejected = Plan(body);
    EXPECT_TRUE(rejected.parts.empty());
    body.regions.push_back(NonSyncRegion(30));
    body.regions.push_back(NonSyncRegion(40));
    auto plan = Plan(body);
    ASSERT_EQ(plan.parts.size(), 2U);
    EXPECT_EQ(Operations(plan.parts[0]), (std::vector<int>{10, 20, 30}));
    EXPECT_EQ(Operations(plan.parts[1]), (std::vector<int>{40}));
}

TEST(CceSplitPlanner, UnsupportedModeKeepsOriginal)
{
    auto body = Body();
    body.supported = false;
    EXPECT_TRUE(Plan(body).parts.empty());
}

TEST(CceSplitPlanner, SetAndWaitCanBeInDifferentSegmentsWithoutBarrier)
{
    CceOriginalFunctionBody body;
    auto set = Effect(CceSplitEffect::SET);
    set.code = "set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);\n";
    auto wait = Effect(CceSplitEffect::WAIT);
    wait.code = "wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);\n";
    body.regions = {NonSyncRegion(10), set, NonSyncRegion(20), wait};
    auto plan = Plan(body);
    ASSERT_EQ(plan.parts.size(), 2U);
    EXPECT_NE(plan.parts[0].source.find(set.code), std::string::npos);
    EXPECT_EQ(plan.parts[0].source.find(wait.code), std::string::npos);
    EXPECT_NE(plan.parts[1].source.find(wait.code), std::string::npos);
    EXPECT_EQ(plan.parts[1].source.find(set.code), std::string::npos);
}

TEST(CceSplitPlanner, ExternalStateAndLoopKeepOriginal)
{
    auto body = Body();
    body.regions[1].allowCutAfter = false;
    body.regions.insert(body.regions.begin(), Effect(CceSplitEffect::SET));
    body.regions.insert(body.regions.begin(), Effect(CceSplitEffect::EXTERNAL_STATE));
    body.regions.push_back(Effect(CceSplitEffect::WAIT));
    auto plan = Plan(body);
    EXPECT_TRUE(plan.parts.empty());
}

TEST(CceSplitPlanner, SynchronizationBeforeOperationsDoesNotBlockSplitting)
{
    auto body = Body();
    body.regions.insert(body.regions.begin(), Effect(CceSplitEffect::SET));
    body.regions.insert(body.regions.begin() + 1, Effect(CceSplitEffect::WAIT));
    auto plan = Plan(body);
    ASSERT_EQ(plan.parts.size(), 2U);
}

TEST(CceSplitPlanner, RepeatedSetsDoNotBlockSplitting)
{
    auto body = Body();
    body.regions.insert(body.regions.begin(), Effect(CceSplitEffect::SET));
    body.regions.insert(body.regions.begin() + 1, Effect(CceSplitEffect::SET));
    auto plan = Plan(body);
    ASSERT_EQ(plan.parts.size(), 2U);
    EXPECT_EQ(Operations(plan.parts[0]), (std::vector<int>{0, 0, 10}));
    EXPECT_EQ(Operations(plan.parts[1]), (std::vector<int>{0, 20}));
    body.regions.insert(body.regions.begin() + 1, Effect(CceSplitEffect::WAIT));
    body.regions.insert(body.regions.begin() + 3, Effect(CceSplitEffect::WAIT));
    EXPECT_EQ(Plan(body).parts.size(), 2U);
}

TEST(CceSplitPlanner, OutstandingEventsAtReturnDoNotBlockSplitting)
{
    auto body = Body();
    body.regions.push_back(Effect(CceSplitEffect::SET));
    body.regions.push_back(Effect(CceSplitEffect::SET));
    auto plan = Plan(body);
    ASSERT_EQ(plan.parts.size(), 2U);
    EXPECT_EQ(Operations(plan.parts[1]), (std::vector<int>{0, 20, 0, 0}));
}

TEST(CceSplitPlanner, SynchronizationDoesNotExceedExactOperationThreshold)
{
    for (auto effect : {CceSplitEffect::SET, CceSplitEffect::WAIT, CceSplitEffect::CROSS_CORE_SYNC}) {
        SCOPED_TRACE(static_cast<int>(effect));
        auto body = Body();
        body.regions[0] = Effect(effect);
        auto plan = Plan(body);
        EXPECT_TRUE(plan.parts.empty());
        body.regions[0].allowCutAfter = false;
        body.regions[1].allowCutAfter = false;
        plan = Plan(body);
        EXPECT_TRUE(plan.parts.empty());
    }
}

TEST(CceSplitPlanner, CrossCoreNotificationsDoNotProtectFollowingSuffix)
{
    auto body = Body();
    body.regions[0] = Effect(CceSplitEffect::CROSS_CORE_SYNC);
    body.regions[0].code = "notify_other_core();\n";
    body.regions.push_back(Effect(CceSplitEffect::FULL_BARRIER));
    body.regions.push_back(NonSyncRegion(30));
    auto plan = Plan(body);
    ASSERT_EQ(plan.parts.size(), 2U);
    EXPECT_NE(plan.parts[0].source.find(body.regions[0].code), std::string::npos);
    EXPECT_EQ(Operations(plan.parts[0]), (std::vector<int>{0, 20}));
    EXPECT_EQ(Operations(plan.parts[1]), (std::vector<int>{0, 30}));
}

TEST(CceSplitPlanner, ReconstructsOnlyDescriptorsUsedInEachPart)
{
    auto body = Body();
    body.regions[0].declarations.push_back({"unused_in_part1", "Tile unused_in_part1(32, shape);\n"});
    auto plan = Plan(body);
    ASSERT_EQ(plan.parts.size(), 2U);
    EXPECT_NE(plan.parts[0].source.find("unused_in_part1"), std::string::npos);
    EXPECT_EQ(plan.parts[1].source.find("unused_in_part1"), std::string::npos);
    EXPECT_NE(plan.parts[1].source.find("Tile tile(0, shape);"), std::string::npos);
}

TEST(CceSplitPlanner, MutableDescriptorRejected)
{
    auto body = Body();
    body.regions.back().declarations[0].second = "Tile tile(99, shape);\n";
    auto plan = Plan(body);
    EXPECT_TRUE(plan.parts.empty());
}

TEST(CceSplitPlanner, OperationCountIgnoresSourceSizeAndDefinitions)
{
    auto body = Body();
    auto first = Plan(body, 1);
    ASSERT_EQ(first.parts.size(), 2U);
    body.preambleDefinitions[0].second += std::string(20000, ' ');
    body.regions[0].code += "// " + std::string(20000, 'x') + "\n";
    auto enlarged = Plan(body, 1);
    ASSERT_EQ(enlarged.parts.size(), first.parts.size());
    for (size_t i = 0; i < first.parts.size(); ++i) {
        EXPECT_EQ(Operations(enlarged.parts[i]), Operations(first.parts[i]));
    }
    EXPECT_TRUE(Plan(body, 3).parts.empty());
}

TEST(CceSplitPlanner, AllSynchronizationKindsAreExcludedFromThreshold)
{
    for (auto effect : {CceSplitEffect::SET, CceSplitEffect::WAIT, CceSplitEffect::BARRIER,
                        CceSplitEffect::FULL_BARRIER, CceSplitEffect::CROSS_CORE_SYNC}) {
        SCOPED_TRACE(static_cast<int>(effect));
        CceOriginalFunctionBody body;
        body.regions.assign(1001, Effect(effect));
        EXPECT_TRUE(Plan(body, 1).parts.empty());
        body.regions.push_back(NonSyncRegion(10));
        EXPECT_TRUE(Plan(body, 1).parts.empty());
        body.regions.push_back(NonSyncRegion(20));
        const auto plan = Plan(body, 1);
        ASSERT_EQ(plan.parts.size(), 2U);
        EXPECT_EQ(Operations(plan.parts[0]).size(), 1002U);
        EXPECT_EQ(Operations(plan.parts[0]).back(), 10);
        EXPECT_EQ(Operations(plan.parts[1]), (std::vector<int>{20}));
    }
}

TEST(CceSplitPlanner, OversizedRegionDoesNotCreateSynchronizationOnlyParts)
{
    for (auto effect : {CceSplitEffect::SET, CceSplitEffect::WAIT, CceSplitEffect::BARRIER,
                        CceSplitEffect::FULL_BARRIER, CceSplitEffect::CROSS_CORE_SYNC}) {
        SCOPED_TRACE(static_cast<int>(effect));
        CceOriginalFunctionBody body;
        body.regions.push_back(Effect(effect));
        std::vector<int> oversizedOperations{0};
        for (int i = 1; i <= 20; ++i) {
            auto region = NonSyncRegion(i);
            region.allowCutAfter = i == 20;
            body.regions.push_back(region);
            oversizedOperations.push_back(i);
        }
        body.regions.push_back(Effect(effect));
        oversizedOperations.push_back(0);
        // A pure synchronization prefix/suffix cannot justify splitting a single oversized region.
        EXPECT_TRUE(Plan(body, 10).parts.empty());

        body.regions.push_back(NonSyncRegion(21));
        auto plan = Plan(body, 10);
        ASSERT_EQ(plan.parts.size(), 2U);
        auto prefixOperations = oversizedOperations;
        prefixOperations.pop_back();
        EXPECT_EQ(Operations(plan.parts[0]), prefixOperations);
        EXPECT_EQ(Operations(plan.parts[1]), (std::vector<int>{0, 21}));

        body.regions.pop_back();
        body.regions.insert(body.regions.begin(), NonSyncRegion(21));
        plan = Plan(body, 10);
        ASSERT_EQ(plan.parts.size(), 2U);
        std::vector<int> allOperations;
        for (const auto& part : plan.parts) {
            const auto operations = Operations(part);
            EXPECT_TRUE(std::any_of(operations.begin(), operations.end(), [](int op) { return op != 0; }));
            allOperations.insert(allOperations.end(), operations.begin(), operations.end());
        }
        oversizedOperations.insert(oversizedOperations.begin(), 21);
        EXPECT_EQ(allOperations, oversizedOperations);
    }
}

TEST(CceSplitPlanner, PartNamesSortInOperationOrderAcrossDigitBoundaries)
{
    for (const auto& [count, width] :
         std::vector<std::pair<int, size_t>>{{9, 1}, {10, 2}, {99, 2}, {100, 3}, {999, 3}, {1000, 4}, {1001, 4}}) {
        SCOPED_TRACE(count);
        CceOriginalFunctionBody body;
        // Synchronization must not increase the naming width.
        body.regions.assign(1000, Effect(CceSplitEffect::SET));
        for (int i = 0; i < count; ++i) {
            body.regions.push_back(NonSyncRegion(i, 0));
        }
        const auto plan = Plan(body, 1);
        ASSERT_EQ(plan.parts.size(), static_cast<size_t>(count));
        size_t previousCall = 0;
        for (size_t i = 0; i < plan.parts.size(); ++i) {
            const auto& part = plan.parts[i];
            const auto digits = std::to_string(i);
            const auto expected = "leaf_part" + std::string(width - digits.size(), '0') + digits;
            EXPECT_EQ(part.name, expected);
            EXPECT_NE(part.source.find("void " + expected + "("), std::string::npos);
            EXPECT_NE(plan.header.find("void " + expected + "("), std::string::npos);
            const auto call = plan.entry.find(expected + "(");
            ASSERT_NE(call, std::string::npos);
            if (i != 0) {
                EXPECT_LT(plan.parts[i - 1].name + ".cpp", part.name + ".cpp");
                EXPECT_LT(previousCall, call);
            }
            previousCall = call;
        }
    }
}

TEST(CceSplitPlanner, NamingWidthUsesOperationCountInsteadOfPartCount)
{
    CceOriginalFunctionBody body;
    body.regions.assign(1000, NonSyncRegion(1, 0));
    const auto plan = Plan(body, 500);
    ASSERT_EQ(plan.parts.size(), 2U);
    EXPECT_EQ(plan.parts[0].name, "leaf_part0000");
    EXPECT_EQ(plan.parts[1].name, "leaf_part0001");
}

TEST(CceSplitPlanner, ThousandOperationThreshold)
{
    CceOriginalFunctionBody body;
    for (int i = 0; i < 1000; ++i) {
        body.regions.push_back(NonSyncRegion(i, 0));
    }
    EXPECT_TRUE(Plan(body, 1000).parts.empty());
    body.regions.push_back(NonSyncRegion(1000, 0));
    auto plan = Plan(body, 1000);
    ASSERT_EQ(plan.parts.size(), 2U);
    EXPECT_EQ(Operations(plan.parts[0]).size(), 1000U);
    EXPECT_EQ(Operations(plan.parts[1]), (std::vector<int>{1000}));
}

TEST(CceSplitPlanner, FinalBarrierAloneDoesNotIntroduceWrapper)
{
    auto body = Body();
    body.regions.erase(body.regions.begin() + 1);
    for (auto& region : body.regions) {
        region.allowCutAfter = false;
    }
    body.regions.push_back(Effect(CceSplitEffect::FULL_BARRIER));
    auto plan = Plan(body);
    EXPECT_TRUE(plan.parts.empty());
    EXPECT_TRUE(plan.entry.empty());
}

TEST(CceSplitPlanner, BarrierWithoutBodyOperationStaysWithFollowingOperation)
{
    auto body = Body();
    body.regions.insert(body.regions.begin(), Effect(CceSplitEffect::FULL_BARRIER));
    auto plan = Plan(body);
    ASSERT_EQ(plan.parts.size(), 2U);
    EXPECT_EQ(Operations(plan.parts[0]), (std::vector<int>{0, 10}));
    EXPECT_EQ(Operations(plan.parts[1]), (std::vector<int>{0, 20}));
}

TEST(CceSplitPlanner, RegeneratedSingleSegmentWithinThresholdKeepsOriginal)
{
    auto plan = Plan(Body(), 9000);
    EXPECT_TRUE(plan.parts.empty());
}

TEST(CceSplitPlanner, DeterministicAndPreservesOperationOrder)
{
    auto body = Body();
    body.regions.push_back(Effect(CceSplitEffect::FULL_BARRIER));
    body.regions.push_back(NonSyncRegion(30));
    auto a = Plan(body);
    auto b = Plan(body);
    ASSERT_EQ(a.parts.size(), 3U);
    EXPECT_EQ(a.entry, b.entry);
    for (size_t i = 0; i < a.parts.size(); ++i) {
        EXPECT_EQ(a.parts[i].source, b.parts[i].source);
        EXPECT_EQ(Operations(a.parts[i]).back(), 10 * static_cast<int>(i + 1));
    }
}
TEST(CceSplitPlanner, PreambleUsesDescriptorAndTransitiveDependenciesInOriginalOrder)
{
    CceOriginalFunctionBody body;
    body.preambleDefinitions = {{"base", "uint64_t base = 32;\n"},
                                {"unused", "uint64_t unused = 99;\n"},
                                {"shape", "uint64_t shape = base;\n"},
                                {"Alias", "using Alias = Tile<float>;\n"},
                                {"addr", "float *addr = (float *)get_imm(0x1000);\n"}};
    auto first = NonSyncRegion(1);
    first.declarations = {{"tile", "Alias tile(addr, shape);\n"}};
    CceSplitRegion second;
    second.code = "consume(base);\n";
    body.regions = {first, second};
    auto plan = Plan(body);
    ASSERT_EQ(plan.parts.size(), 2U);
    const auto& source = plan.parts[0].source;
    EXPECT_EQ(source.find("uint64_t unused"), source.npos);
    EXPECT_LT(source.find("uint64_t base"), source.find("uint64_t shape"));
    EXPECT_LT(source.find("uint64_t shape"), source.find("using Alias"));
    EXPECT_LT(source.find("float *addr"), source.find("Alias tile"));
    EXPECT_EQ(PreambleBytes(plan.parts[1], body), body.preambleDefinitions[0].second.size());
    EXPECT_EQ(plan.parts[1].source.find("using Alias"), source.npos);
    EXPECT_EQ(plan.parts[1].source.find("uint64_t shape"), source.npos);
}

TEST(CceSplitPlanner, PreambleMatchesWholeIdentifiersAndIgnoresCommentsAndLiterals)
{
    CceOriginalFunctionBody body;
    body.preambleDefinitions = {
        {"sym_1", "int sym_1 = 1;\n"}, {"sym_10", "int sym_10 = 10;\n"}, {"comment_only", "int comment_only = 0;\n"}};
    CceSplitRegion first;
    first.code = "consume(sym_10); /* sym_1 */ // comment_only\n"
                 "const char *text = \"sym_1 \\\" // comment_only\";\n"
                 "const char *raw = u8R\"tag(sym_1 \" /* comment_only)tag\";\n";
    CceSplitRegion second;
    second.code = "consume(sym_1);\n";
    body.regions = {first, second};
    auto plan = Plan(body);
    ASSERT_EQ(plan.parts.size(), 2U);
    EXPECT_EQ(PreambleBytes(plan.parts[0], body), body.preambleDefinitions[1].second.size());
    EXPECT_EQ(plan.parts[0].source.find("int sym_1 ="), std::string::npos);
    EXPECT_EQ(plan.parts[0].source.find("int comment_only ="), std::string::npos);
    EXPECT_NE(plan.parts[0].source.find(first.code), std::string::npos);
    EXPECT_EQ(PreambleBytes(plan.parts[1], body), body.preambleDefinitions[0].second.size());
}

TEST(CceSplitPlanner, OperationCountSelectionRollsBackDefinitionsFromRejectedCandidate)
{
    CceOriginalFunctionBody body;
    body.preambleDefinitions = {{"a", "int a = 1;\n"}, {"b", "int b = 2;\n"}, {"c", "int c = 3;\n"}};
    for (const auto& name : {"a", "b", "c"}) {
        CceSplitRegion region;
        region.code = "consume(" + std::string(name) + ");\n";
        body.regions.push_back(region);
    }
    const auto individual = Plan(body);
    ASSERT_EQ(individual.parts.size(), 3U);
    auto plan = Plan(body, 2);
    ASSERT_EQ(plan.parts.size(), 2U);
    EXPECT_NE(plan.parts[0].source.find("consume(a)"), std::string::npos);
    EXPECT_NE(plan.parts[0].source.find("consume(b)"), std::string::npos);
    EXPECT_EQ(plan.parts[0].source.find("int c ="), std::string::npos);
    EXPECT_EQ(plan.parts[0].source.find("consume(c)"), std::string::npos);
    EXPECT_EQ(plan.parts[1].source.find("int a ="), std::string::npos);
    EXPECT_EQ(PreambleBytes(plan.parts[1], body), body.preambleDefinitions[2].second.size());
}

TEST(CceSplitPlanner, LargePreambleDoesNotForceEveryOperationIntoOversizedPart)
{
    CceOriginalFunctionBody body;
    size_t originalPreambleBytes = 0;
    for (size_t i = 0; i < 5000; ++i) {
        const auto name = "shape_" + std::to_string(i);
        const auto code = "uint64_t " + name + " = 32;\n";
        body.preambleDefinitions.emplace_back(name, code);
        originalPreambleBytes += code.size();
    }
    ASSERT_GT(originalPreambleBytes, 100 * 1024U);
    for (int i = 0; i < 30; ++i) {
        CceSplitRegion region;
        region.code = "// op:" + std::to_string(i) + "\n// " + std::string(4000, 'x') + "\nconsume(shape_" +
                      std::to_string(i) + ");\n";
        body.regions.push_back(region);
    }
    const auto plan = Plan(body, 5);
    ASSERT_GT(plan.parts.size(), 1U);
    EXPECT_LT(plan.parts.size(), body.regions.size());
    std::vector<int> magics;
    for (const auto& part : plan.parts) {
        EXPECT_EQ(Operations(part).size(), 5U);
        EXPECT_LT(PreambleBytes(part, body), 1024U);
        EXPECT_EQ(part.source.find("shape_4999"), std::string::npos);
        const auto operations = Operations(part);
        magics.insert(magics.end(), operations.begin(), operations.end());
    }
    ASSERT_EQ(magics.size(), body.regions.size());
    for (size_t i = 0; i < magics.size(); ++i) {
        EXPECT_EQ(magics[i], static_cast<int>(i));
    }
}

TEST(CceSplitPlanner, RequiredLargeDefinitionStillAllowsOversize)
{
    CceOriginalFunctionBody body;
    body.preambleDefinitions = {{"value", "int value = 1; // " + std::string(1024, 'x') + "\n"}};
    CceSplitRegion first;
    first.code = "consume(value);\n";
    CceSplitRegion second;
    second.code = "finish();\n";
    body.regions = {first, second};
    const auto plan = Plan(body, 1);
    ASSERT_EQ(plan.parts.size(), 2U);
    EXPECT_GT(plan.parts[0].source.size(), 512U);
    EXPECT_EQ(PreambleBytes(plan.parts[0], body), body.preambleDefinitions[0].second.size());
    EXPECT_LE(plan.parts[1].source.size(), 512U);
    EXPECT_EQ(PreambleBytes(plan.parts[1], body), 0U);
}

TEST(CceSplitPlanner, DuplicatePreambleNameKeepsOriginal)
{
    auto body = Body();
    body.preambleDefinitions = {{"shape", "int shape = 1;\n"}, {"shape", "int shape = 2;\n"}};
    const auto plan = Plan(body);
    EXPECT_TRUE(plan.parts.empty());
}
} // namespace npu::tile_fwk
