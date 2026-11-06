/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "codegen/npu/cloudnpu/codegen_op_cloudnpu.h"
#include "interface/tensor/irbuilder.h"
#include "gtest/gtest.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <set>
#include <sstream>
#include <unistd.h>

namespace npu::tile_fwk {
namespace {
std::string splitDebugLog;

int32_t CheckSplitDebugLevel(int32_t, int32_t level, LogModule module)
{
    return level == DLOG_DEBUG && module == LogModule::CODEGEN;
}

void __attribute__((format(printf, 3, 4))) RecordSplitDebugLog(int32_t, int32_t level, const char* format, ...)
{
    EXPECT_EQ(level, DLOG_DEBUG);
    char message[2048];
    va_list args;
    va_start(args, format);
    vsnprintf(message, sizeof(message), format, args);
    va_end(args);
    splitDebugLog += message;
    splitDebugLog += '\n';
}

class ScopedSplitDebugLog {
public:
    ScopedSplitDebugLog() : checkLevel_(LogFuncInfo::Instance().checkLevel), record_(LogFuncInfo::Instance().record)
    {
        splitDebugLog.clear();
        LogFuncInfo::Instance().checkLevel = CheckSplitDebugLevel;
        LogFuncInfo::Instance().record = RecordSplitDebugLog;
    }
    ~ScopedSplitDebugLog()
    {
        LogFuncInfo::Instance().checkLevel = checkLevel_;
        LogFuncInfo::Instance().record = record_;
    }

private:
    decltype(LogFuncInfo::checkLevel) checkLevel_;
    decltype(LogFuncInfo::record) record_;
};
} // namespace

class SplitBodyCodegen : public CodeGenCloudNPU {
public:
    SplitBodyCodegen() : CodeGenCloudNPU({}) {}
    // Override only the split admission context; this does not simulate a device/toolchain.
    SplitBodyCodegen(NPUArch platform, bool mainBlock) : SplitBodyCodegen()
    {
        platform_ = platform;
        ctx.isMainBlock = mainBlock;
    }
    using CodeGenNPU::GenDynParamForExpr;
    using CodeGenNPU::GenFuncBody;
    using CodeGenNPU::GenFuncBodyForSplit;
    using CodeGenNPU::GenFuncHeader;
};

class CceSplitCodegen : public testing::Test {
protected:
    void SetUp() override
    {
        config::Reset();
        config::SetCodeGenOption<std::string>(PLATFORM_SOC_VERSION, "Ascend910B3");
    }
    void TearDown() override { config::Reset(); }

    LogicalTensorPtr CreateUB(Function& function, const std::vector<int64_t>& shape = {32, 32})
    {
        IRBuilder builder;
        auto tensor = builder.CreateTensorVar(function, DT_FP32, shape);
        tensor->SetMemoryTypeOriginal(MEM_UB);
        tensor->SetMemoryTypeToBe(MEM_UB);
        tensor->SetAttr(OpAttributeKey::needAlloc, true);
        tensor->memoryrange.memId = 0;
        tensor->memoryrange.start = 0;
        tensor->memoryrange.end = 4096;
        return tensor;
    }
};

class SplitTaskCodegen : public SplitBodyCodegen {
public:
    using CodeGenNPU::GenSplitTasks;

private:
    // Task generation is tested without requiring a device compiler installation.
    void BuildIncludes(std::ostringstream&) const override {}
};

TEST_F(CceSplitCodegen, EntryArgumentsMatchGeneratedFunctionParameters)
{
    Function function(Program::GetInstance(), "split_signature", "split_signature", nullptr);
    function.SetFunctionType(FunctionType::STATIC);
    CompileInfo info(function, {}, {0, &function}, false);
    SplitBodyCodegen codegen;
    codegen.GenFuncHeader(function, info);
    const auto declaration = info.GetFuncDeclare();
    const auto begin = declaration.find('(');
    const auto end = declaration.rfind(')');
    ASSERT_NE(begin, std::string::npos);
    ASSERT_NE(end, std::string::npos);
    ASSERT_LT(begin, end);
    const auto parameters = declaration.substr(begin, end - begin + 1);

    // Derive the expected arguments from the real signature, including their order.
    std::istringstream parameterStream(declaration.substr(begin + 1, end - begin - 1));
    std::string parameter;
    std::string arguments;
    while (std::getline(parameterStream, parameter, ',')) {
        const auto nameEnd = parameter.find_last_not_of(" \t\n");
        ASSERT_NE(nameEnd, std::string::npos);
        const auto nameBegin = parameter.find_last_of(" \t\n*&", nameEnd);
        ASSERT_NE(nameBegin, std::string::npos);
        if (!arguments.empty()) {
            arguments += ", ";
        }
        arguments += parameter.substr(nameBegin + 1, nameEnd - nameBegin);
    }
    ASSERT_FALSE(arguments.empty());

    CceOriginalFunctionBody body;
    CceSplitRegion region;
    region.code = "operation();\n";
    body.regions.assign(2, region);
    const auto plan = PlanCceSplit(body, "", info.GetKernelName(), parameters, "parts.h", 1);
    ASSERT_EQ(plan.parts.size(), 2U);
    std::string expectedCalls;
    for (const auto& part : plan.parts) {
        expectedCalls += part.name + "(" + arguments + ");\n";
    }
    const auto bodyBegin = plan.entry.find("{\n");
    ASSERT_NE(bodyBegin, std::string::npos);
    EXPECT_EQ(plan.entry.substr(bodyBegin + 2), expectedCalls + "}\n");
}

TEST_F(CceSplitCodegen, SplitTasksLogResultsWithoutDiagnosticFiles)
{
    char path[] = "/tmp/pypto-split-tasks-XXXXXX";
    ASSERT_NE(mkdtemp(path), nullptr);
    struct Cleanup {
        std::filesystem::path directory;
        ~Cleanup() { std::filesystem::remove_all(directory); }
    } cleanup{path};
    Function function(Program::GetInstance(), "split_tasks", "split_tasks", nullptr);
    function.SetFunctionType(FunctionType::STATIC);
    CompileInfo info(function, {}, {0, &function}, false);
    info.SetCCEAbsPath(std::string(path) + "/leaf.cpp");
    info.SetBinAbsPath(std::string(path) + "/leaf.o");
    info.SetKernelName("leaf");
    info.SetFuncDeclare("void leaf(Params* param, long GMStackBase, long* hcclContext, Stats* taskStat);");
    CceOriginalFunctionBody body;
    CceSplitRegion region;
    region.code = "operation();\n";
    body.regions.assign(10, region);
    SplitTaskCodegen codegen;
    ScopedSplitDebugLog capture;

    EXPECT_FALSE(codegen.GenSplitTasks(function, info, body, "original", 10000));
    EXPECT_TRUE(std::filesystem::is_empty(path));
    EXPECT_NE(splitDebugLog.find("CCE split: kernel=leaf, split=false"), std::string::npos);

    splitDebugLog.clear();
    EXPECT_TRUE(codegen.GenSplitTasks(function, info, body, "original", 1));
    EXPECT_NE(splitDebugLog.find("CCE split: kernel=leaf, split=true"), std::string::npos);
    EXPECT_EQ(splitDebugLog.find("CCE split part:"), std::string::npos);
    std::set<std::string> files;
    for (const auto& file : std::filesystem::directory_iterator(path)) {
        files.insert(file.path().filename().string());
    }
    std::set<std::string> expectedFiles{"leaf.cpp", "leaf_parts.h", "leaf_entry.cpp"};
    for (int i = 0; i < 10; ++i) {
        expectedFiles.insert("leaf_part0" + std::to_string(i) + ".cpp");
    }
    EXPECT_EQ(files, expectedFiles);
    splitDebugLog.clear();
    LogFuncInfo::Instance().checkLevel = [](int32_t, int32_t, LogModule) -> int32_t { return 0; };
    EXPECT_FALSE(codegen.GenSplitTasks(function, info, body, "original", 10000));
    EXPECT_TRUE(splitDebugLog.empty());
}

TEST_F(CceSplitCodegen, NativeConfigurationRejectsNegativeThreshold)
{
    EXPECT_EQ(config::GetCodeGenOption<int64_t>(CCE_SPLIT_THRESHOLD_TILE_OPS), 200);
    EXPECT_ANY_THROW(config::SetCodeGenOption<int64_t>(CCE_SPLIT_THRESHOLD_TILE_OPS, -1));
    EXPECT_EQ(config::GetCodeGenOption<int64_t>(CCE_SPLIT_THRESHOLD_TILE_OPS), 200);
}

TEST_F(CceSplitCodegen, SplitAdmissionAllowsRequestedArchitecturesCubeAndMainBlock)
{
    for (auto arch : {NPUArch::DAV_2201, NPUArch::DAV_3510, NPUArch::DAV_1001}) {
        for (bool cube : {false, true}) {
            for (bool mainBlock : {false, true}) {
                SCOPED_TRACE(static_cast<int>(arch));
                SCOPED_TRACE(cube);
                SCOPED_TRACE(mainBlock);
                Function function(Program::GetInstance(), "split_admission", "split_admission", nullptr);
                function.SetFunctionType(FunctionType::STATIC);
                auto& first = function.AddOperation(Opcode::OP_PHASE1, {}, {});
                first.SetAttribute(OpAttributeKey::isCube, cube);
                ASSERT_EQ(function.IsCube(), cube);
                auto& barrier = function.AddOperation(Opcode::OP_BAR_ALL, {}, {});
                barrier.syncQueue_.pipeId_ = PipeType::PIPE_ALL;
                function.AddOperation(Opcode::OP_PHASE2, {}, {});
                SplitBodyCodegen codegen(arch, mainBlock);
                CceOriginalFunctionBody body;
                std::ostringstream original;
                codegen.GenFuncBodyForSplit(function, function, original, body);
                ASSERT_TRUE(body.supported);
                ASSERT_EQ(body.regions.size(), 3U);
                auto plan = PlanCceSplit(body, "", "leaf",
                                         "(Params* param, long GMStackBase, long* hcclContext, Stats* taskStat)",
                                         "parts.h", 1);
                ASSERT_EQ(plan.parts.size(), 2U);
            }
        }
    }
}

TEST_F(CceSplitCodegen, OtherArchitecturesKeepOriginalBody)
{
    Function function(Program::GetInstance(), "split_other_arch", "split_other_arch", nullptr);
    function.AddOperation(Opcode::OP_PHASE1, {}, {});
    SplitBodyCodegen codegen(NPUArch::DAV_9201, false);
    CceOriginalFunctionBody body;
    std::ostringstream original;
    codegen.GenFuncBodyForSplit(function, function, original, body);
    EXPECT_FALSE(body.supported);
    EXPECT_TRUE(body.regions.empty());
    EXPECT_NE(original.str().find("SUBKERNEL_PHASE1"), std::string::npos);
}

TEST_F(CceSplitCodegen, AllBarrierOpcodesUseActualPipelineScope)
{
    for (auto opcode : {Opcode::OP_BAR_V, Opcode::OP_BAR_M, Opcode::OP_BAR_ALL}) {
        for (auto pipe : {PipeType::PIPE_V, PipeType::PIPE_M, PipeType::PIPE_ALL}) {
            SCOPED_TRACE(static_cast<int>(opcode));
            SCOPED_TRACE(static_cast<int>(pipe));
            Function function(Program::GetInstance(), "split_sync", "split_sync", nullptr);
            auto& op = function.AddOperation(opcode, {}, {});
            op.syncQueue_ = {pipe, pipe, CoreType::AIV, CoreType::AIV, -1, AIVCore::UNSPECIFIED, AIVCore::UNSPECIFIED};
            auto symbols = std::make_shared<SymbolManager>();
            CodeGenOpCloudNPU generator({symbols, function, function, op});
            const auto code = generator.GenOpCode();
            const bool full = pipe == PipeType::PIPE_ALL;
            EXPECT_EQ(code.find("pipe_barrier(PIPE_ALL)") != std::string::npos, full);
            EXPECT_EQ(generator.GetSplitRegion().effect, full ? CceSplitEffect::FULL_BARRIER : CceSplitEffect::BARRIER);
        }
    }
}

TEST_F(CceSplitCodegen, LocalSyncClassificationMatchesEmittedInstruction)
{
    Function function(Program::GetInstance(), "split_event", "split_event", nullptr);
    auto& op = function.AddOperation(Opcode::OP_SYNC_SRC, {}, {});
    op.syncQueue_ = {PipeType::PIPE_MTE2,  PipeType::PIPE_V,    CoreType::AIV, CoreType::AIV, 3,
                     AIVCore::UNSPECIFIED, AIVCore::UNSPECIFIED};
    auto symbols = std::make_shared<SymbolManager>();
    CodeGenOpCloudNPU generator({symbols, function, function, op});
    auto region = generator.GetSplitRegion();
    EXPECT_EQ(region.effect, CceSplitEffect::SET);
    op.syncQueue_.trigCoreType_ = CoreType::AIC;
    CodeGenOpCloudNPU annotated({symbols, function, function, op});
    EXPECT_NE(annotated.GenOpCode().find("set_flag"), std::string::npos);
    EXPECT_EQ(annotated.GetSplitRegion().effect, CceSplitEffect::SET);
    auto& wait = function.AddOperation(Opcode::OP_SYNC_DST, {}, {});
    wait.syncQueue_ = op.syncQueue_;
    CodeGenOpCloudNPU waiter({symbols, function, function, wait});
    EXPECT_EQ(waiter.GetSplitRegion().effect, CceSplitEffect::WAIT);
}

TEST_F(CceSplitCodegen, CrossCoreSynchronizationAllowsFollowingSplit)
{
    for (auto opcode :
         {Opcode::OP_CV_SYNC_SRC, Opcode::OP_CV_SYNC_DST, Opcode::OP_FFTS_CROSS_CORE_SYNC, Opcode::OP_WAIT_FLAG_DEV}) {
        SCOPED_TRACE(static_cast<int>(opcode));
        Function function(Program::GetInstance(), "split_remote", "split_remote", nullptr);
        auto& op = function.AddOperation(opcode, {}, {});
        auto symbols = std::make_shared<SymbolManager>();
        CodeGenOpCloudNPU generator({symbols, function, function, op});
        auto region = generator.GetSplitRegion();
        EXPECT_EQ(region.effect, CceSplitEffect::CROSS_CORE_SYNC);
        CceOriginalFunctionBody body;
        body.regions = {region, CceSplitRegion{}, CceSplitRegion{}};
        auto plan = PlanCceSplit(body, "", "leaf",
                                 "(Params* param, long GMStackBase, long* hcclContext, Stats* taskStat)", "parts.h", 1);
        ASSERT_EQ(plan.parts.size(), 2U);
    }
}

TEST_F(CceSplitCodegen, ExternalStatePreventsSplit)
{
    Function function(Program::GetInstance(), "split_external", "split_external", nullptr);
    auto& remote = function.AddOperation(Opcode::OP_AICPU_CALL_AIV, {}, {});
    auto& barrier = function.AddOperation(Opcode::OP_BAR_ALL, {}, {});
    barrier.syncQueue_.pipeId_ = PipeType::PIPE_ALL;
    auto& tail = function.AddOperation(Opcode::OP_PHASE2, {}, {});
    CceOriginalFunctionBody body;
    auto symbols = std::make_shared<SymbolManager>();
    for (auto* op : {&remote, &barrier, &tail}) {
        CodeGenOpCloudNPU generator({symbols, function, function, *op});
        body.regions.push_back(generator.GetSplitRegion());
    }
    auto plan = PlanCceSplit(body, "", "leaf", "(Params* param, long GMStackBase, long* hcclContext, Stats* taskStat)",
                             "parts.h", 1);
    ASSERT_TRUE(plan.parts.empty());
}

TEST_F(CceSplitCodegen, UnpairedEventsAllowSplit)
{
    CceOriginalFunctionBody body;
    CceSplitRegion operation;
    CceSplitRegion set;
    set.effect = CceSplitEffect::SET;
    auto wait = set;
    wait.effect = CceSplitEffect::WAIT;
    body.regions = {operation, set, wait};
    auto plan = PlanCceSplit(body, "", "leaf", "(Params* param, long GMStackBase, long* hcclContext, Stats* taskStat)",
                             "parts.h", 1);
    EXPECT_TRUE(plan.parts.empty());
    body.regions.pop_back();
    body.regions.push_back(operation);
    plan = PlanCceSplit(body, "", "leaf", "(Params* param, long GMStackBase, long* hcclContext, Stats* taskStat)",
                        "parts.h", 1);
    ASSERT_EQ(plan.parts.size(), 2U);
}

TEST_F(CceSplitCodegen, MutableDescriptorAndSmallRegeneratedBodyKeepOriginal)
{
    CceOriginalFunctionBody body;
    CceSplitRegion first;
    first.declarations = {{"tile", "Tile tile(0);\n"}};
    auto last = first;
    last.declarations[0].second = "Tile tile(32);\n";
    body.regions = {first, last};
    auto plan = PlanCceSplit(body, "", "leaf", "(Params* param, long GMStackBase, long* hcclContext, Stats* taskStat)",
                             "parts.h", 1);
    EXPECT_TRUE(plan.parts.empty());
    body.regions.back() = first;
    plan = PlanCceSplit(body, "", "leaf", "(Params* param, long GMStackBase, long* hcclContext, Stats* taskStat)",
                        "parts.h", 9000);
    EXPECT_TRUE(plan.parts.empty());
}

TEST_F(CceSplitCodegen, AddDescriptorsComeFromSymbolManager)
{
    Function function(Program::GetInstance(), "split_add", "split_add", nullptr);
    function.SetFunctionType(FunctionType::STATIC);
    auto lhs = CreateUB(function);
    auto rhs = CreateUB(function);
    auto dst = CreateUB(function);
    auto& op = function.AddOperation(Opcode::OP_ADD, {lhs, rhs}, {dst});
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    auto symbols = std::make_shared<SymbolManager>();
    CodeGenCloudNPU codegen({});
    codegen.GenAllocForLocalBuffer(op, symbols);
    CodeGenOpCloudNPU generator({symbols, function, function, op});
    auto region = generator.GetSplitRegion();
    EXPECT_EQ(region.effect, CceSplitEffect::NON_SYNC_OP);
    ASSERT_EQ(region.declarations.size(), 3U);
    for (const auto& [name, definition] : region.declarations) {
        EXPECT_FALSE(name.empty());
        EXPECT_NE(definition.find(name), std::string::npos);
        EXPECT_NE(generator.GenOpCode().find(name), std::string::npos);
    }

    auto& barrier = function.AddOperation(Opcode::OP_BAR_ALL, {}, {});
    barrier.syncQueue_ = {PipeType::PIPE_ALL,   PipeType::PIPE_ALL,  CoreType::AIV, CoreType::AIV, -1,
                          AIVCore::UNSPECIFIED, AIVCore::UNSPECIFIED};
    function.AddOperation(Opcode::OP_ADD, {lhs, dst}, {dst});
    SplitBodyCodegen splitCodegen;
    CceOriginalFunctionBody body;
    std::ostringstream original;
    splitCodegen.GenFuncBodyForSplit(function, function, original, body);
    ASSERT_TRUE(body.supported);
    ASSERT_EQ(body.regions.size(), 3U);
    auto plan = PlanCceSplit(body, "", "leaf", "(Params* param, long GMStackBase, long* hcclContext, Stats* taskStat)",
                             "parts.h", 1);
    ASSERT_EQ(plan.parts.size(), 2U);
    EXPECT_NE(plan.parts[1].source.find("TAdd"), std::string::npos);
}

TEST_F(CceSplitCodegen, DivisionAndExpParticipateInBothCodegenModes)
{
    for (bool tileTensor : {false, true}) {
        SCOPED_TRACE(tileTensor);
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, tileTensor);
        Function function(Program::GetInstance(), "split_div_exp", "split_div_exp", nullptr);
        function.SetFunctionType(FunctionType::STATIC);
        auto lhs = CreateUB(function);
        auto rhs = CreateUB(function);
        auto dst = CreateUB(function);
        function.AddOperation(Opcode::OP_DIV, {lhs, rhs}, {dst});
        function.AddOperation(Opcode::OP_EXP, {dst}, {dst});
        SplitBodyCodegen codegen;
        CceOriginalFunctionBody body;
        std::ostringstream original;
        codegen.GenFuncBodyForSplit(function, function, original, body);
        ASSERT_TRUE(body.supported);
        ASSERT_EQ(body.regions.size(), 2U);
        EXPECT_EQ(body.regions[0].effect, CceSplitEffect::NON_SYNC_OP);
        EXPECT_EQ(body.regions[1].effect, CceSplitEffect::NON_SYNC_OP);
        auto plan = PlanCceSplit(body, "", "leaf",
                                 "(Params* param, long GMStackBase, long* hcclContext, Stats* taskStat)", "parts.h", 1);
        ASSERT_EQ(plan.parts.size(), 2U);
        EXPECT_NE(plan.parts[0].source.find(body.regions[0].code), std::string::npos);
        EXPECT_NE(plan.parts[1].source.find(body.regions[1].code), std::string::npos);
        EXPECT_EQ(body.regions[0].declarations.empty(), !tileTensor);
    }
}

TEST_F(CceSplitCodegen, LoopKeepsBodyAndFullDimensionalAddressDependenciesTogether)
{
    for (bool mainBlock : {false, true}) {
        SCOPED_TRACE(mainBlock);
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
        Function function(Program::GetInstance(), "split_loop", "split_loop", nullptr);
        function.SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
        function.SetUnderDynamicFunction(true);
        const std::vector<int64_t> shape{2, 2, 2, 4, 8};
        auto lhs = CreateUB(function, shape);
        auto rhs = CreateUB(function, shape);
        auto dst = CreateUB(function, shape);
        for (const auto& tensor : {lhs, rhs, dst}) {
            tensor->UpdateDynValidShape(SymbolicScalar::FromConcrete(shape));
        }
        auto& first = function.AddOperation(Opcode::OP_DIV, {lhs, rhs}, {dst});
        const std::vector<SymbolicScalar> axes{2, 2, 2};
        if (mainBlock) {
            first.SetAttribute(OpAttributeKey::loopAxes, std::vector<int64_t>{2, 2, 2});
        } else {
            first.SetAttribute(OpAttributeKey::dynloopAxes, axes);
        }
        first.SetAttribute(mainBlock ? OpAttributeKey::loopGroupStart : OpAttributeKey::dynloopGroupStart, true);
        auto& inside = function.AddOperation(Opcode::OP_BAR_ALL, {}, {});
        inside.syncQueue_.pipeId_ = PipeType::PIPE_ALL;
        auto& last = function.AddOperation(Opcode::OP_EXP, {dst}, {dst});
        if (mainBlock) {
            last.SetAttribute(OpAttributeKey::loopAxes, std::vector<int64_t>{2, 2, 2});
        } else {
            last.SetAttribute(OpAttributeKey::dynloopAxes, axes);
        }
        last.SetAttribute(mainBlock ? OpAttributeKey::loopGroupEnd : OpAttributeKey::dynloopGroupEnd, true);
        auto& outside = function.AddOperation(Opcode::OP_BAR_ALL, {}, {});
        outside.syncQueue_.pipeId_ = PipeType::PIPE_ALL;
        function.AddOperation(Opcode::OP_ADD, {lhs, rhs}, {dst});
        SplitBodyCodegen codegen(NPUArch::DAV_2201, mainBlock);
        CceOriginalFunctionBody body;
        std::ostringstream original;
        codegen.GenFuncBodyForSplit(function, function, original, body);
        ASSERT_EQ(body.regions.size(), 5U);
        EXPECT_FALSE(body.regions[0].allowCutAfter);
        EXPECT_FALSE(body.regions[1].allowCutAfter);
        EXPECT_FALSE(body.regions[2].allowCutAfter);
        EXPECT_TRUE(body.regions[3].allowCutAfter);
        EXPECT_TRUE(body.regions[0].code.empty());
        EXPECT_TRUE(body.regions[1].code.empty());
        EXPECT_NE(body.regions[2].code.find("for ("), std::string::npos);
        EXPECT_NE(body.regions[2].code.find(".GetLinearAddr("), std::string::npos);
        EXPECT_EQ(body.regions[0].declarations.size(), 6U);
        auto plan = PlanCceSplit(body, "", "leaf",
                                 "(Params* param, long GMStackBase, long* hcclContext, Stats* taskStat)", "parts.h", 1);
        ASSERT_EQ(plan.parts.size(), 2U);
        for (const auto& [name, definition] : body.regions[0].declarations) {
            EXPECT_NE(body.regions[2].code.find(name), std::string::npos);
            EXPECT_NE(plan.parts[0].source.find(definition), std::string::npos);
        }
        EXPECT_EQ(plan.parts[1].source.find("_low2DimInLoop"), std::string::npos);
    }
}

TEST_F(CceSplitCodegen, SplitPreamblePrunesUnusedAddressesAndAliasesInBothModes)
{
    for (bool tileTensor : {false, true}) {
        SCOPED_TRACE(tileTensor);
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, tileTensor);
        Function function(Program::GetInstance(), "split_preamble", "split_preamble", nullptr);
        function.SetFunctionType(FunctionType::STATIC);
        auto first = CreateUB(function);
        auto second = CreateUB(function, {16, 32});
        second->memoryrange.start = 0x4000;
        second->memoryrange.end = 0x5000;
        function.AddOperation(Opcode::OP_ADD, {first, first}, {first});
        function.AddOperation(Opcode::OP_ADD, {second, second}, {second});
        SplitBodyCodegen codegen;
        CceOriginalFunctionBody body;
        std::ostringstream original;
        codegen.GenFuncBodyForSplit(function, function, original, body);
        ASSERT_FALSE(body.preambleDefinitions.empty());
        std::string preamble;
        for (const auto& [name, code] : body.preambleDefinitions) {
            EXPECT_FALSE(name.empty());
            preamble += code;
        }
        EXPECT_EQ(original.str().find(preamble), 0U);
        std::ostringstream unsplit;
        codegen.GenFuncBody(function, function, unsplit);
        EXPECT_EQ(original.str(), unsplit.str());
        auto plan = PlanCceSplit(body, "", "leaf",
                                 "(Params* param, long GMStackBase, long* hcclContext, Stats* taskStat)", "parts.h", 1);
        ASSERT_EQ(plan.parts.size(), 2U);
        EXPECT_EQ(plan.parts[0].source.find("get_imm(0x4000)"), std::string::npos);
        EXPECT_NE(plan.parts[1].source.find("get_imm(0x4000)"), std::string::npos);
        EXPECT_NE(plan.parts[0].source.find("get_imm(0x0)"), std::string::npos);
        EXPECT_EQ(plan.parts[1].source.find("get_imm(0x0)"), std::string::npos);
    }
}

TEST_F(CceSplitCodegen, DynamicPreambleKeepsReplacementDependencyAndOriginalEmission)
{
    Function function(Program::GetInstance(), "split_dynamic_preamble", "split_dynamic_preamble", nullptr);
    function.SetUnderDynamicFunction(true);
    DynParamInfo base{1, 0, 0, DynParamInfoType::VALID_SHAPE, 0, SymbolicScalar(32), false, ""};
    DynParamInfo alias = base;
    alias.replacedSymbol = "base";
    function.InsertDynParam("base", base);
    function.InsertDynParam("used", alias);
    function.InsertDynParam("unused", base);
    SplitBodyCodegen codegen;
    CceOriginalFunctionBody body;
    std::ostringstream withMetadata;
    codegen.GenDynParamForExpr(withMetadata, function, &body.preambleDefinitions);
    std::ostringstream withoutMetadata;
    codegen.GenDynParamForExpr(withoutMetadata, function);
    EXPECT_EQ(withMetadata.str(), withoutMetadata.str());
    ASSERT_EQ(body.preambleDefinitions.size(), 3U);
    CceSplitRegion first;
    first.code = "consume(used);\n";
    CceSplitRegion second;
    second.code = "consume(base);\n";
    body.regions = {first, second};
    auto plan = PlanCceSplit(body, "", "leaf", "(Params* param, long GMStackBase, long* hcclContext, Stats* taskStat)",
                             "parts.h", 1);
    ASSERT_EQ(plan.parts.size(), 2U);
    EXPECT_NE(plan.parts[0].source.find("uint64_t used = base;"), std::string::npos);
    EXPECT_LT(plan.parts[0].source.find("uint64_t base"), plan.parts[0].source.find("uint64_t used"));
    EXPECT_EQ(plan.parts[0].source.find("uint64_t unused"), std::string::npos);
    EXPECT_EQ(plan.parts[1].source.find("uint64_t used"), std::string::npos);
}

TEST_F(CceSplitCodegen, SpecialFloatPreamblePreservesBitsAndPrunesUnusedConstants)
{
    Function function(Program::GetInstance(), "split_float_preamble", "split_float_preamble", nullptr);
    auto& op = function.AddOperation(Opcode::OP_PHASE1, {}, {});
    op.SetAttribute(OpAttributeKey::vectorScalar,
                    std::vector<Element>{Element(DT_FP32, std::numeric_limits<float>::infinity()),
                                         Element(DT_FP32, std::numeric_limits<float>::quiet_NaN())});
    FloatSpecValMgr manager;
    manager.UpdateByOp(op);
    CceOriginalFunctionBody body;
    std::ostringstream original;
    manager.PrintFloatSpecVal(original, &body.preambleDefinitions);
    ASSERT_EQ(body.preambleDefinitions.size(), 2U);
    std::ostringstream withoutMetadata;
    manager.PrintFloatSpecVal(withoutMetadata);
    EXPECT_EQ(original.str(), withoutMetadata.str());
    for (const auto& [name, code] : body.preambleDefinitions) {
        CceSplitRegion region;
        region.code = "consume(" + name + ".f);\n";
        body.regions.push_back(region);
    }
    auto plan = PlanCceSplit(body, "", "leaf", "(Params* param, long GMStackBase, long* hcclContext, Stats* taskStat)",
                             "parts.h", 1);
    ASSERT_EQ(plan.parts.size(), 2U);
    for (size_t i = 0; i < plan.parts.size(); ++i) {
        EXPECT_NE(plan.parts[i].source.find(body.preambleDefinitions[i].second), std::string::npos);
        EXPECT_EQ(plan.parts[i].source.find(body.preambleDefinitions[1 - i].second), std::string::npos);
    }
}
} // namespace npu::tile_fwk
