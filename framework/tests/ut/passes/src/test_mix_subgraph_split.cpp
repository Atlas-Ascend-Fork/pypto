/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_mix_subgraph_split.cpp
 * \brief Unit test for mixSubgraphSplit
 * */
#include <gtest/gtest.h>
#include "symbolic_scalar_test_utils.h"
#include "interface/tensor/irbuilder.h"
#define private public
#define protected public
#include "passes/block_graph_pass/mix_subgraph_split.h"
#undef private
#undef protected
#include "computational_graph_builder.h"
#include "interface/tensor/irbuilder.h"

namespace npu {
namespace tile_fwk {
constexpr uint64_t programId = 100;
constexpr int MS_NUM16 = 16;
constexpr int MS_NUM3 = 3;
constexpr int MS_NUM10005 = 10005;

// 辅助函数声明
namespace test_utils {
void VerifyBasicChecks(Status status, Function& rootFunc);
void VerifyProgramProperties(Function& rootFunc);
void VerifyCallOpsAfterSplit(Function& rootFunc);
void VerifyScopeTypes(Function& rootFunc, int expectedCubeCount, int expectedVectorCount);
void VerifyCleanup(Function& rootFunc, Function* originalMixFunc, Operation* originalCallOp);
} // namespace test_utils

class MixSubgraphSplitTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }

    void TearDown() override {}

protected:
    // 单个辅助函数：构建Mix子图的所有内容
    std::shared_ptr<Function> BuildMixFunction(Function* rootFunc, std::vector<int64_t>& tensorShape)
    {
        auto mixFuncPtr = std::make_shared<Function>(Program::GetInstance(), "mix_func_illegal", "mix_func_illegal",
                                                     rootFunc);
        mixFuncPtr->SetGraphType(GraphType::BLOCK_GRAPH);
        mixFuncPtr->SetFunctionType(FunctionType::STATIC);

        // 创建tensors
        auto inputTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, tensorShape,
                                                                      CreateTestConstIntVector(tensorShape));
        auto outputTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, tensorShape,
                                                                       CreateTestConstIntVector(tensorShape));
        auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, tensorShape,
                                                                  CreateTestConstIntVector(tensorShape));
        auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, tensorShape,
                                                                  CreateTestConstIntVector(tensorShape));

        // 设置边界tensor
        mixFuncPtr->inCasts_.push_back(inputTensor);
        mixFuncPtr->inCasts_.push_back(tensor1);
        mixFuncPtr->inCasts_.push_back(tensor2);
        mixFuncPtr->outCasts_.push_back(outputTensor);
        mixFuncPtr->outCasts_.push_back(tensor1);
        mixFuncPtr->outCasts_.push_back(tensor2);

        // 构建内部结构
        auto shapeImme = OpImmediate::Specified(tensorShape);
        std::vector<int64_t> offsetVec = {0, 0};
        auto offsetImme = OpImmediate::Specified(offsetVec);
        std::vector<OpImmediate> emptyVec;

        // Component 1 (CUBE)
        auto& copyout1 = IRBuilder().CreateTensorOpStmt(*mixFuncPtr, Opcode::OP_COPY_OUT, {inputTensor}, {tensor1});
        copyout1.SetOpAttribute(
            std::make_shared<CopyOpAttribute>(MemoryType::MEM_UB, offsetImme, shapeImme, shapeImme, emptyVec));
        copyout1.SetOOpAtt(0, 0);
        copyout1.UpdateInternalSubgraphID(1);
        copyout1.SetAttr(OpAttributeKey::isCube, true);

        auto& copyin3 = IRBuilder().CreateTensorOpStmt(*mixFuncPtr, Opcode::OP_COPY_IN, {tensor2}, {outputTensor});
        copyin3.SetOpAttribute(
            std::make_shared<CopyOpAttribute>(offsetImme, MemoryType::MEM_UB, shapeImme, shapeImme, emptyVec));
        copyin3.SetIOpAtt(0, 0);
        copyin3.UpdateInternalSubgraphID(1);
        copyin3.SetAttr(OpAttributeKey::isCube, true);

        // Component 0 (VECTOR)
        auto& copyin2 = IRBuilder().CreateTensorOpStmt(*mixFuncPtr, Opcode::OP_COPY_IN, {tensor1}, {tensor2});
        copyin2.SetOpAttribute(
            std::make_shared<CopyOpAttribute>(offsetImme, MemoryType::MEM_UB, shapeImme, shapeImme, emptyVec));
        copyin2.SetIOpAtt(0, 0);
        copyin2.UpdateInternalSubgraphID(0);
        copyin2.SetAIVCore(AIVCore::AIV0);

        return mixFuncPtr;
    }
};

// 辅助函数实现
namespace test_utils {
void VerifyBasicChecks(Status status, Function& rootFunc)
{
    // 状态验证
    ASSERT_EQ(status, SUCCESS) << "MixSubgraphSplit should succeed";
    // program数量验证
    auto& programs = rootFunc.programs_;
    EXPECT_EQ(programs.size(), 2) << "Should have 2 programs after split (originally 1 Mix, split to 2 leaves)";
    // ID连续性验证
    EXPECT_NE(programs.find(0), programs.end()) << "Should have program ID 0";
    EXPECT_NE(programs.find(1), programs.end()) << "Should have program ID 1";
}

void VerifyProgramProperties(Function& rootFunc)
{
    auto& programs = rootFunc.programs_;
    for (const auto& [progId, func] : programs) {
        ASSERT_NE(func, nullptr) << "Function should not be null";
        // 验证名称包含leaf后缀
        std::string funcName = func->GetRawName();
        EXPECT_NE(funcName.find("leaf"), std::string::npos)
            << "Function name should contain 'leaf' suffix: " << funcName;
        // 验证类型设置
        EXPECT_EQ(func->GetFunctionType(), FunctionType::STATIC);
        EXPECT_EQ(func->GetGraphType(), GraphType::BLOCK_GRAPH);
        // 验证programID一致性
        EXPECT_EQ(func->GetProgramId(), progId) << "Function's program ID should match map key";
        // 验证LeafFuncAttribute
        auto leafAttr = func->GetLeafFuncAttribute();
        ASSERT_NE(leafAttr, nullptr) << "LeafFuncAttribute should be set";
        EXPECT_NE(leafAttr->mixId, static_cast<uint64_t>(-1)) << "mixId should be assigned";
        EXPECT_NE(leafAttr->mixResourceType, MixResourceType::UNKNOWN) << "mixResourceType should be set";
    }
}

void VerifyCallOpsAfterSplit(Function& rootFunc)
{
    // callOp数量验证
    auto newCallOps = rootFunc.GetCallopList();
    EXPECT_EQ(newCallOps.size(), 2) << "Should have 2 call ops after split (1 original * 2 components)";
    // callOp属性验证
    auto& programs = rootFunc.programs_;
    for (auto* newCallOp : newCallOps) {
        ASSERT_NE(newCallOp, nullptr) << "CallOp should not be null";
        EXPECT_FALSE(newCallOp->IsDeleted()) << "CallOp should not be deleted";

        auto newCallAttr = dynamic_cast<CallOpAttribute*>(newCallOp->GetOpAttribute().get());
        ASSERT_NE(newCallAttr, nullptr) << "CallOpAttribute should exist";

        if (newCallAttr && newCallAttr->invokeInfo_) {
            uint64_t progId = newCallAttr->invokeInfo_->GetProgramId();
            EXPECT_TRUE(progId == 0 || progId == 1) << "CallOp's program ID should be 0 or 1, got: " << progId;

            // 验证对应函数存在
            auto it = programs.find(progId);
            EXPECT_NE(it, programs.end()) << "CallOp references non-existent program ID: " << progId;

            // 验证wrapId设置
            EXPECT_NE(newCallAttr->wrapId, static_cast<uint64_t>(-1)) << "wrapId should be set";
        }
    }
}

void VerifyScopeTypes(Function& rootFunc, int expectedCubeCount, int expectedVectorCount)
{
    // scope类型验证
    auto& programs = rootFunc.programs_;
    int cubeCount = 0;
    int vectorCount = 0;

    for (const auto& [progId, func] : programs) {
        (void)progId;
        auto leafAttr = func->GetLeafFuncAttribute();
        if (leafAttr) {
            if (leafAttr->aivCore == AIVCore::UNSPECIFIED) {
                cubeCount++;
            } else if (leafAttr->aivCore == AIVCore::AIV0 || leafAttr->aivCore == AIVCore::AIV1) {
                vectorCount++;
            }
        }
    }

    EXPECT_EQ(cubeCount, expectedCubeCount) << "Cube component count mismatch";
    EXPECT_EQ(vectorCount, expectedVectorCount) << "Vector component count mismatch";
}

void VerifyCleanup(Function& rootFunc, Function* originalMixFunc, Operation* originalCallOp)
{
    auto& programs = rootFunc.programs_;
    // 原始Mix子图验证
    bool originalMixFuncStillExists = false;
    for (const auto& [progId, func] : programs) {
        (void)progId;
        if (func == originalMixFunc) {
            originalMixFuncStillExists = true;
            break;
        }
    }
    EXPECT_FALSE(originalMixFuncStillExists) << "Original mix function should be removed";

    // 原始callOp清理验证
    auto newCallOps = rootFunc.GetCallopList();
    bool originalCallOpStillExists = false;
    for (auto* callOpPtr : newCallOps) {
        if (callOpPtr == originalCallOp) {
            originalCallOpStillExists = true;
            break;
        }
    }
    EXPECT_FALSE(originalCallOpStillExists) << "Original callOp should be deleted";
    // 验证programs映射正确性
    std::set<uint64_t> programIdsFromCallOps;
    for (auto* callOpPtr : newCallOps) {
        auto newCallOpAttr = dynamic_cast<CallOpAttribute*>(callOpPtr->GetOpAttribute().get());
        if (newCallOpAttr && newCallOpAttr->invokeInfo_) {
            programIdsFromCallOps.insert(newCallOpAttr->invokeInfo_->GetProgramId());
        }
    }
    for (const auto& [progId, func] : programs) {
        (void)func;
        EXPECT_NE(programIdsFromCallOps.find(progId), programIdsFromCallOps.end())
            << "Program ID " << progId << " should have corresponding callOp";
    }
}
} // namespace test_utils

// 在测试类定义之后，测试用例之前，先定义所有辅助函数

// 辅助函数1：验证结果
void VerifyBasicSplitResult(Status status, Function& rootFunc, Function* originalMixFunc, Operation* originalCallOp)
{
    using namespace test_utils;
    VerifyBasicChecks(status, rootFunc);
    VerifyProgramProperties(rootFunc);
    VerifyCallOpsAfterSplit(rootFunc);
    VerifyScopeTypes(rootFunc, 1, 1);
    VerifyCleanup(rootFunc, originalMixFunc, originalCallOp);
}

// 辅助函数2：创建CallOp
Operation& CreateCallOp(std::shared_ptr<Function>& rootFuncPtr, const uint64_t mixProgramId,
                        const FunctionHash& mixFuncHash)
{
    std::vector<int64_t> tensorShape = {MS_NUM16, MS_NUM16};

    auto callInTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, tensorShape,
                                                                    CreateTestConstIntVector(tensorShape));
    auto callInTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, tensorShape,
                                                                    CreateTestConstIntVector(tensorShape));
    auto callInTensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, tensorShape,
                                                                    CreateTestConstIntVector(tensorShape));
    auto callOutTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, tensorShape,
                                                                    CreateTestConstIntVector(tensorShape));

    auto& callOp = IRBuilder().CreateTensorOpStmt(*rootFuncPtr, Opcode::OP_CALL,
                                                  {callInTensor1, callInTensor2, callInTensor3}, {callOutTensor});

    auto callAttr = std::make_shared<CallOpAttribute>();
    auto invokeInfo = std::make_shared<SubfuncInvokeInfoTy>();
    invokeInfo->UpdateProgramSubgraphId(mixProgramId);
    callAttr->SetCalleeHash(mixFuncHash);
    callAttr->invokeInfo_ = invokeInfo;

    std::vector<SymbolicScalar> linearArgs;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 9; j++) {
            linearArgs.push_back(IRBuilder().CreateConstInt(static_cast<int64_t>(j + i * 10)));
        }
    }
    callAttr->linearArgList_ = linearArgs;
    callOp.SetOpAttribute(callAttr);
    callOp.UpdateSubgraphID(mixProgramId);

    return callOp;
}

// 辅助函数3：创建Vector scope
void CreateVectorScope(std::shared_ptr<Function>& mixFuncPtr, std::shared_ptr<LogicalTensor>& incast3,
                       std::shared_ptr<LogicalTensor>& outcast1)
{
    std::vector<int64_t> tensorShape = {MS_NUM16, MS_NUM16};

    // 获取cubeTensor3（最后一个cube操作的输出）
    auto operations = mixFuncPtr->Operations(false);
    auto cubeTensor3 = operations[operations.size() - 1].GetOOperands()[0];

    auto vectorTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, tensorShape,
                                                                    CreateTestConstIntVector(tensorShape));
    auto vectorTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, tensorShape,
                                                                    CreateTestConstIntVector(tensorShape));

    auto shapeImme = OpImmediate::Specified(tensorShape);
    std::vector<int64_t> offsetVec = {0, 0};
    auto offsetImme = OpImmediate::Specified(offsetVec);
    std::vector<OpImmediate> emptyVec;

    // Vector scope op（internalSubgraphID=1）
    auto& vectorAdd = IRBuilder().CreateTensorOpStmt(*mixFuncPtr, Opcode::OP_ADD, {cubeTensor3, incast3},
                                                     {vectorTensor1});
    vectorAdd.SetIOpAtt(1, 5);
    vectorAdd.UpdateInternalSubgraphID(1);
    vectorAdd.SetAIVCore(AIVCore::AIV0);

    auto& vectorSqrt = IRBuilder().CreateTensorOpStmt(*mixFuncPtr, Opcode::OP_SQRT, {vectorTensor1}, {vectorTensor2});
    vectorSqrt.UpdateInternalSubgraphID(1);
    vectorSqrt.SetAIVCore(AIVCore::AIV0);

    auto& vectorCopyOut = IRBuilder().CreateTensorOpStmt(*mixFuncPtr, Opcode::OP_COPY_OUT, {vectorTensor2}, {outcast1});
    vectorCopyOut.SetOpAttribute(
        std::make_shared<CopyOpAttribute>(MemoryType::MEM_UB, offsetImme, shapeImme, shapeImme, emptyVec));
    vectorCopyOut.SetOOpAtt(0, 0);
    vectorCopyOut.UpdateInternalSubgraphID(1);
    vectorCopyOut.SetAIVCore(AIVCore::AIV0);
}

// 辅助函数4：创建Cube scope
void CreateCubeScope(std::shared_ptr<Function>& mixFuncPtr, std::shared_ptr<LogicalTensor>& incast1,
                     std::shared_ptr<LogicalTensor>& incast2)
{
    std::vector<int64_t> tensorShape = {MS_NUM16, MS_NUM16};
    auto cubeTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, tensorShape,
                                                                  CreateTestConstIntVector(tensorShape));
    auto cubeTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, tensorShape,
                                                                  CreateTestConstIntVector(tensorShape));
    auto cubeTensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, tensorShape,
                                                                  CreateTestConstIntVector(tensorShape));

    auto shapeImme = OpImmediate::Specified(tensorShape);
    std::vector<int64_t> offsetVec = {0, 0};
    auto offsetImme = OpImmediate::Specified(offsetVec);
    std::vector<OpImmediate> emptyVec;

    // Cube scope op（internalSubgraphID=0）
    auto& cubeCopyIn1 = IRBuilder().CreateTensorOpStmt(*mixFuncPtr, Opcode::OP_COPY_IN, {incast1}, {cubeTensor1});
    cubeCopyIn1.SetOpAttribute(
        std::make_shared<CopyOpAttribute>(offsetImme, MemoryType::MEM_UB, shapeImme, shapeImme, emptyVec));
    cubeCopyIn1.SetIOpAtt(0, 0);
    cubeCopyIn1.UpdateInternalSubgraphID(0);
    cubeCopyIn1.SetAttr(OpAttributeKey::isCube, true);

    auto& cubeCopyIn2 = IRBuilder().CreateTensorOpStmt(*mixFuncPtr, Opcode::OP_COPY_IN, {incast2}, {cubeTensor2});
    cubeCopyIn2.SetOpAttribute(
        std::make_shared<CopyOpAttribute>(offsetImme, MemoryType::MEM_UB, shapeImme, shapeImme, emptyVec));
    cubeCopyIn2.SetIOpAtt(0, 0);
    cubeCopyIn2.UpdateInternalSubgraphID(0);
    cubeCopyIn2.SetAttr(OpAttributeKey::isCube, true);

    auto& cubeMul = IRBuilder().CreateTensorOpStmt(*mixFuncPtr, Opcode::OP_A_MUL_B, {cubeTensor1, cubeTensor2},
                                                   {cubeTensor3});
    cubeMul.UpdateInternalSubgraphID(0);
    cubeMul.SetAttr(OpAttributeKey::isCube, true);
}

// 辅助函数5：设置Mix子图结构
void SetupMixSubgraphStructure(std::shared_ptr<Function>& mixFuncPtr, FunctionHash& mixFuncHash)
{
    std::vector<int64_t> tensorShape = {MS_NUM16, MS_NUM16};

    // 创建incast和outcast
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, tensorShape,
                                                              CreateTestConstIntVector(tensorShape));
    auto incast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, tensorShape,
                                                              CreateTestConstIntVector(tensorShape));
    auto incast3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, tensorShape,
                                                              CreateTestConstIntVector(tensorShape));
    auto outcast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, tensorShape,
                                                               CreateTestConstIntVector(tensorShape));

    mixFuncPtr->inCasts_.push_back(incast1);
    mixFuncPtr->inCasts_.push_back(incast2);
    mixFuncPtr->inCasts_.push_back(incast3);
    mixFuncPtr->outCasts_.push_back(outcast1);

    // 创建Cube scope
    CreateCubeScope(mixFuncPtr, incast1, incast2);

    // 创建Vector scope
    CreateVectorScope(mixFuncPtr, incast3, outcast1);

    // 计算哈希并注册
    mixFuncPtr->ComputeHash();
    mixFuncHash = mixFuncPtr->GetFunctionHash();
    Program::GetInstance().GetFunctionCache().Insert(mixFuncHash, *mixFuncPtr);
}

// 辅助函数6：创建Mix子图
std::shared_ptr<Function> CreateMixSubgraph(std::shared_ptr<Function>& rootFuncPtr, FunctionHash& mixFuncHash)
{
    auto mixFuncPtr = std::make_shared<Function>(Program::GetInstance(), "test_mix_func", "test_mix_func",
                                                 rootFuncPtr.get());
    mixFuncPtr->SetGraphType(GraphType::BLOCK_GRAPH);
    mixFuncPtr->SetFunctionType(FunctionType::STATIC);

    // 添加到programs，使用programId=100
    const uint64_t mixProgramId = 100;
    rootFuncPtr->programs_[mixProgramId] = mixFuncPtr.get();

    // 创建Mix子图内部结构
    SetupMixSubgraphStructure(mixFuncPtr, mixFuncHash);

    return mixFuncPtr;
}

// 辅助函数7：创建根函数
std::shared_ptr<Function> CreateTestRootFunction()
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "test_root", "test_root", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    return rootFuncPtr;
}

TEST_F(MixSubgraphSplitTest, TestSingleMixSubgraphBasicSplit)
{
    // 1. 创建测试场景
    auto rootFuncPtr = CreateTestRootFunction();

    FunctionHash mixFuncHash;
    auto mixFuncPtr = CreateMixSubgraph(rootFuncPtr, mixFuncHash);

    // 2. 创建callOp
    const uint64_t mixProgramId = 100;
    auto& callOp = CreateCallOp(rootFuncPtr, mixProgramId, mixFuncHash);

    // 3. 执行拆分
    MixSubgraphSplit splitter;
    Status status = splitter.RunOnFunction(*rootFuncPtr);

    // 4. 验证结果
    VerifyBasicSplitResult(status, *rootFuncPtr, mixFuncPtr.get(), &callOp);
}

// 辅助函数1：为非Mix子图创建callOp（这个函数会被CreateNonMixFunctions调用，所以要放在前面）
void CreateCallOpForNonMix(std::shared_ptr<Function>& rootFuncPtr, uint64_t programIdx, FunctionHash hash,
                           const std::vector<int64_t>& shape)
{
    auto callInTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto callOutTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& callOp = IRBuilder().CreateTensorOpStmt(*rootFuncPtr, Opcode::OP_CALL, {callInTensor}, {callOutTensor});
    auto callAttr = std::make_shared<CallOpAttribute>();
    auto invokeInfo = std::make_shared<SubfuncInvokeInfoTy>();
    invokeInfo->UpdateProgramSubgraphId(programIdx);
    callAttr->SetCalleeHash(hash);
    callAttr->invokeInfo_ = invokeInfo;

    std::vector<SymbolicScalar> linearArgs;
    for (int argIdx = 0; argIdx < 2; argIdx++) {
        for (int j = 0; j < 9; j++) {
            linearArgs.push_back(IRBuilder().CreateConstInt(static_cast<int64_t>(j + argIdx * 10)));
        }
    }
    callAttr->linearArgList_ = linearArgs;
    callOp.SetOpAttribute(callAttr);
    callOp.UpdateSubgraphID(programIdx);
}

// 辅助函数2：创建非Mix子图（这个函数调用CreateCallOpForNonMix）
void CreateNonMixFunctions(std::shared_ptr<Function>& rootFuncPtr,
                           std::vector<std::shared_ptr<Function>>& nonMixFunctions,
                           const std::vector<uint64_t>& nonMixProgramIds)
{
    for (int i = 0; i < 2; i++) {
        auto nonMixFunc = std::make_shared<Function>(Program::GetInstance(), "test_non_mix_" + std::to_string(i),
                                                     "test_non_mix_" + std::to_string(i), rootFuncPtr.get());
        nonMixFunc->SetGraphType(GraphType::BLOCK_GRAPH);
        nonMixFunc->SetFunctionType(FunctionType::STATIC);

        std::vector<int64_t> shape = {8, 8};
        auto incastTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
        auto outcastTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape,
                                                                        CreateTestConstIntVector(shape));

        nonMixFunc->inCasts_.push_back(incastTensor);
        nonMixFunc->outCasts_.push_back(outcastTensor);

        auto internalTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape,
                                                                          CreateTestConstIntVector(shape));
        auto& copyInOp = IRBuilder().CreateTensorOpStmt(*nonMixFunc, Opcode::OP_COPY_IN, {incastTensor},
                                                        {internalTensor1});
        copyInOp.SetIOpAtt(0, 0);

        auto shapeImme = OpImmediate::Specified(shape);
        std::vector<int64_t> offsetVec = {0, 0};
        auto offsetImme = OpImmediate::Specified(offsetVec);
        std::vector<OpImmediate> emptyVec;
        auto copyInAttr = std::make_shared<CopyOpAttribute>(offsetImme, MemoryType::MEM_UB, shapeImme, shapeImme,
                                                            emptyVec);
        copyInOp.SetOpAttribute(copyInAttr);

        auto internalTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape,
                                                                          CreateTestConstIntVector(shape));
        auto& expOp = IRBuilder().CreateTensorOpStmt(*nonMixFunc, Opcode::OP_EXP, {internalTensor1}, {internalTensor2});
        (void)expOp;
        auto& copyOutOp = IRBuilder().CreateTensorOpStmt(*nonMixFunc, Opcode::OP_COPY_OUT, {internalTensor2},
                                                         {outcastTensor});
        copyOutOp.SetOOpAtt(0, 0);

        auto copyOutAttr = std::make_shared<CopyOpAttribute>(MemoryType::MEM_UB, offsetImme, shapeImme, shapeImme,
                                                             emptyVec);
        copyOutOp.SetOpAttribute(copyOutAttr);

        nonMixFunc->ComputeHash();
        FunctionHash hash = nonMixFunc->GetFunctionHash();
        Program::GetInstance().GetFunctionCache().Insert(hash, *nonMixFunc);
        rootFuncPtr->programs_[nonMixProgramIds[i]] = nonMixFunc.get();
        nonMixFunctions.push_back(nonMixFunc);

        // 调用CreateCallOpForNonMix创建对应的callOp
        CreateCallOpForNonMix(rootFuncPtr, nonMixProgramIds[i], hash, shape);
    }
}

void CreateCallOpsForMixFunction(std::shared_ptr<Function>& rootFuncPtr, uint64_t programIdx, FunctionHash hash,
                                 const std::vector<int64_t>& shape, int mixIdx)
{
    int callOpCount = (mixIdx % 2 == 0) ? 1 : 2;

    for (int callIdx = 0; callIdx < callOpCount; callIdx++) {
        auto callInTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape,
                                                                        CreateTestConstIntVector(shape));
        auto callInTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape,
                                                                        CreateTestConstIntVector(shape));
        auto callOutTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape,
                                                                        CreateTestConstIntVector(shape));

        auto& callOp = IRBuilder().CreateTensorOpStmt(*rootFuncPtr, Opcode::OP_CALL, {callInTensor1, callInTensor2},
                                                      {callOutTensor});

        auto callAttr = std::make_shared<CallOpAttribute>();
        auto invokeInfo = std::make_shared<SubfuncInvokeInfoTy>();
        invokeInfo->UpdateProgramSubgraphId(programIdx);
        callAttr->SetCalleeHash(hash);
        callAttr->invokeInfo_ = invokeInfo;

        std::vector<SymbolicScalar> linearArgs;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 9; j++) {
                linearArgs.push_back(IRBuilder().CreateConstInt(static_cast<int64_t>(j + i * 10)));
            }
        }
        callAttr->linearArgList_ = linearArgs;
        callOp.SetOpAttribute(callAttr);
        callOp.UpdateSubgraphID(programIdx);
    }
}

// 辅助函数4：创建额外的scope（这个函数会被CreateMixFunctions调用）
void CreateAdditionalScopes(std::shared_ptr<Function>& mixFunc, int componentCount, const std::vector<int64_t>& shape)
{
    // 获取最后一个tensor
    auto operations = mixFunc->Operations(false);
    auto lastTensor = operations.back().GetOOperands()[0];

    for (int compIdx = 1; compIdx < componentCount; compIdx++) {
        auto newTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
        Opcode opcode = (compIdx % 2 == 0) ? Opcode::OP_NEG : Opcode::OP_SQRT;

        if (compIdx % 2 == 0) {
            auto& cubeOp = IRBuilder().CreateTensorOpStmt(*mixFunc, opcode, {lastTensor}, {newTensor});
            cubeOp.UpdateInternalSubgraphID(compIdx);
            cubeOp.SetAttr(OpAttributeKey::isCube, true);
        } else {
            auto& vectorOp = IRBuilder().CreateTensorOpStmt(*mixFunc, opcode, {lastTensor}, {newTensor});
            vectorOp.UpdateInternalSubgraphID(compIdx);
            vectorOp.SetAIVCore(AIVCore::AIV0);
        }
        lastTensor = newTensor;
    }

    // 创建COPY_OUT op
    auto& copyOut = IRBuilder().CreateTensorOpStmt(*mixFunc, Opcode::OP_COPY_OUT, {lastTensor},
                                                   {mixFunc->outCasts_[0]});
    copyOut.UpdateInternalSubgraphID(componentCount - 1);
    copyOut.SetOOpAtt(0, 0);

    auto shapeImme = OpImmediate::Specified(shape);
    std::vector<int64_t> offsetVec = {0, 0};
    auto offsetImme = OpImmediate::Specified(offsetVec);
    std::vector<OpImmediate> emptyVec;
    auto copyOutAttr = std::make_shared<CopyOpAttribute>(MemoryType::MEM_UB, offsetImme, shapeImme, shapeImme,
                                                         emptyVec);
    copyOut.SetOpAttribute(copyOutAttr);

    if ((componentCount - 1) % 2 == 0) {
        copyOut.SetAttr(OpAttributeKey::isCube, true);
    } else {
        copyOut.SetAIVCore(AIVCore::AIV0);
    }
}

void CreateMixFunctions(std::shared_ptr<Function>& rootFuncPtr, std::vector<std::shared_ptr<Function>>& mixFunctions,
                        const std::vector<uint64_t>& mixProgramIds, const std::vector<int>& componentCounts)
{
    for (int mixIdx = 0; mixIdx < 3; mixIdx++) {
        auto mixFunc = std::make_shared<Function>(Program::GetInstance(), "test_mix_" + std::to_string(mixIdx),
                                                  "test_mix_" + std::to_string(mixIdx), rootFuncPtr.get());
        mixFunc->SetGraphType(GraphType::BLOCK_GRAPH);
        mixFunc->SetFunctionType(FunctionType::STATIC);

        std::vector<int64_t> shape = {16, 16};
        auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
        auto incast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
        auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

        mixFunc->inCasts_.push_back(incast1);
        mixFunc->inCasts_.push_back(incast2);
        mixFunc->outCasts_.push_back(outcast);

        // 创建Cube scope
        auto cubeTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
        auto cubeTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
        auto cubeTensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
        auto cubeOutput = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

        auto shapeImme = OpImmediate::Specified(shape);
        std::vector<int64_t> offsetVec = {0, 0};
        auto offsetImme = OpImmediate::Specified(offsetVec);
        std::vector<OpImmediate> emptyVec;

        auto& copyIn1 = IRBuilder().CreateTensorOpStmt(*mixFunc, Opcode::OP_COPY_IN, {incast1}, {cubeTensor1});
        copyIn1.UpdateInternalSubgraphID(0);
        copyIn1.SetIOpAtt(0, 0);
        copyIn1.SetAttr(OpAttributeKey::isCube, true);
        auto copyIn1Attr = std::make_shared<CopyOpAttribute>(offsetImme, MemoryType::MEM_UB, shapeImme, shapeImme,
                                                             emptyVec);
        copyIn1.SetOpAttribute(copyIn1Attr);

        auto& copyIn2 = IRBuilder().CreateTensorOpStmt(*mixFunc, Opcode::OP_COPY_IN, {incast2}, {cubeTensor2});
        copyIn2.UpdateInternalSubgraphID(0);
        copyIn2.SetIOpAtt(0, 0);
        copyIn2.SetAttr(OpAttributeKey::isCube, true);
        auto copyIn2Attr = std::make_shared<CopyOpAttribute>(offsetImme, MemoryType::MEM_UB, shapeImme, shapeImme,
                                                             emptyVec);
        copyIn2.SetOpAttribute(copyIn2Attr);

        auto& cubeMul = IRBuilder().CreateTensorOpStmt(*mixFunc, Opcode::OP_A_MUL_B, {cubeTensor1, cubeTensor2},
                                                       {cubeTensor3});
        cubeMul.UpdateInternalSubgraphID(0);
        cubeMul.SetAttr(OpAttributeKey::isCube, true);

        auto& cubeExp = IRBuilder().CreateTensorOpStmt(*mixFunc, Opcode::OP_EXP, {cubeTensor3}, {cubeOutput});
        cubeExp.UpdateInternalSubgraphID(0);
        cubeExp.SetAttr(OpAttributeKey::isCube, true);

        // 调用CreateAdditionalScopes创建后续scope
        CreateAdditionalScopes(mixFunc, componentCounts[mixIdx], shape);

        mixFunc->ComputeHash();
        FunctionHash hash = mixFunc->GetFunctionHash();
        Program::GetInstance().GetFunctionCache().Insert(hash, *mixFunc);
        rootFuncPtr->programs_[mixProgramIds[mixIdx]] = mixFunc.get();
        mixFunctions.push_back(mixFunc);

        // 调用CreateCallOpsForMixFunction为Mix子图创建callOp
        CreateCallOpsForMixFunction(rootFuncPtr, mixProgramIds[mixIdx], hash, shape, mixIdx);
    }
}

// 辅助函数6：验证多个Mix子图拆分结果（最后定义，因为它不调用其他辅助函数）
void VerifyMultipleMixSplitResults(std::shared_ptr<Function>& rootFuncPtr, Status status,
                                   const std::vector<std::shared_ptr<Function>>& mixFunctions,
                                   const std::vector<std::shared_ptr<Function>>& nonMixFunctions,
                                   const std::vector<int>& componentCounts)
{
    EXPECT_EQ(status, SUCCESS) << "Multiple mix subgraphs split should succeed";

    // 计算预期的新子图总数
    size_t expectedNewProgramCount = 2; // 2个非Mix子图保留
    for (int count : componentCounts) {
        expectedNewProgramCount += count; // 每个Mix子图的scope数
    }

    auto& programs = rootFuncPtr->programs_;
    EXPECT_EQ(programs.size(), expectedNewProgramCount)
        << "Program count mismatch. Expected: " << expectedNewProgramCount << ", Actual: " << programs.size();

    // ID连续性验证
    uint64_t expectedMaxId = expectedNewProgramCount - 1;
    for (uint64_t i = 0; i <= expectedMaxId; i++) {
        EXPECT_NE(programs.find(i), programs.end()) << "Missing continuous program ID: " << i;
    }

    // 非Mix子图ID重映射验证
    for (size_t i = 0; i < nonMixFunctions.size(); i++) {
        bool found = false;
        for (const auto& [progId, func] : programs) {
            if (func == nonMixFunctions[i].get()) {
                EXPECT_LT(progId, 2) << "Non-mix function should have ID < 2";
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "Non-mix function " << i << " not found after split";
    }

    // Mix子图scope验证
    int totalSplitFunctions = 0;
    for (const auto& [progId, func] : programs) {
        (void)progId;
        auto leafAttr = func->GetLeafFuncAttribute();
        if (leafAttr && leafAttr->mixId != -1) {
            totalSplitFunctions++;
        }
    }

    EXPECT_EQ(totalSplitFunctions, 2 + 3 + 4)
        << "Should have " << (2 + 3 + 4) << " split functions from 3 mix subgraphs";

    // callOp数量验证
    auto newCallOps = rootFuncPtr->GetCallopList();
    size_t expectedNewCallOpCount = 2 * 1 + 1 * 2 + 2 * 3 + 1 * 4;
    EXPECT_EQ(newCallOps.size(), expectedNewCallOpCount)
        << "CallOp count mismatch. Expected: " << expectedNewCallOpCount << ", Actual: " << newCallOps.size();

    // 资源清理验证
    for (const auto& mixFunc : mixFunctions) {
        bool stillExists = false;
        for (const auto& [progId, func] : programs) {
            (void)progId;
            if (func == mixFunc.get()) {
                stillExists = true;
                break;
            }
        }
        EXPECT_FALSE(stillExists) << "Original mix function should be removed";
    }
}

/**
 * 测试多个Mix子图拆分处理
 */
TEST_F(MixSubgraphSplitTest, TestMultipleMixSubgraphsSplit)
{
    // 1. 创建rootFunction
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "test_root_multi", "test_root_multi",
                                                  nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();

    // 2. 创建3个Mix子图和2个非Mix子图
    std::vector<std::shared_ptr<Function>> mixFunctions;
    std::vector<std::shared_ptr<Function>> nonMixFunctions;
    std::vector<uint64_t> mixProgramIds = {100, 101, 102};
    std::vector<uint64_t> nonMixProgramIds = {200, 201};

    // 2.1 创建2个非Mix子图
    CreateNonMixFunctions(rootFuncPtr, nonMixFunctions, nonMixProgramIds);

    // 2.2 创建3个Mix子图，scope数量分别为2,3,4
    std::vector<int> componentCounts = {2, 3, 4};
    CreateMixFunctions(rootFuncPtr, mixFunctions, mixProgramIds, componentCounts);

    // 执行拆分
    MixSubgraphSplit splitter;
    Status status = splitter.RunOnFunction(*rootFuncPtr);

    // 验证结果
    VerifyMultipleMixSplitResults(rootFuncPtr, status, mixFunctions, nonMixFunctions, componentCounts);
}

/**
 * 测试rootFunction无Mix子图时的处理逻辑
 */
TEST_F(MixSubgraphSplitTest, TestNoMixSubgraphScenario)
{
    // 1. 创建仅包含非Mix子图的rootFunction
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "test_root_no_mix", "test_root_no_mix",
                                                  nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    // 2. 创建3个普通（非Mix）子图
    std::vector<std::shared_ptr<Function>> nonMixFunctions;
    std::vector<uint64_t> programIds = {10, 20, 30};
    for (int i = 0; i < 3; i++) {
        auto func = std::make_shared<Function>(Program::GetInstance(), "test_func_" + std::to_string(i),
                                               "test_func_" + std::to_string(i), rootFuncPtr.get());
        func->SetGraphType(GraphType::BLOCK_GRAPH);
        func->SetFunctionType(FunctionType::STATIC);
        // 创建简单op（无internalSubgraphID标记）
        std::vector<int64_t> shape = {8, 8};
        auto inputTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
        auto outputTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
        func->inCasts_.push_back(inputTensor);
        func->outCasts_.push_back(outputTensor);
        auto& expOp = IRBuilder().CreateTensorOpStmt(*func, Opcode::OP_EXP, {inputTensor}, {outputTensor});
        (void)expOp;
        func->ComputeHash();
        FunctionHash hash = func->GetFunctionHash();
        Program::GetInstance().GetFunctionCache().Insert(hash, *func);

        rootFuncPtr->programs_[programIds[i]] = func.get();
        nonMixFunctions.push_back(func);
        // 创建callOp
        auto& callOp = IRBuilder().CreateTensorOpStmt(*rootFuncPtr, Opcode::OP_CALL, {}, {});
        auto callAttr = std::make_shared<CallOpAttribute>();
        auto invokeInfo = std::make_shared<SubfuncInvokeInfoTy>();
        invokeInfo->UpdateProgramSubgraphId(programIds[i]);
        callAttr->SetCalleeHash(hash);
        callAttr->invokeInfo_ = invokeInfo;
        callOp.SetOpAttribute(callAttr);
    }
    // 3. 记录原始状态
    auto originalPrograms = rootFuncPtr->programs_;
    auto originalCallOps = rootFuncPtr->GetCallopList();
    size_t originalProgramCount = originalPrograms.size(); // 应该为3
    size_t originalCallOpCount = originalCallOps.size();   // 应该为3
    // 4. 执行拆分（应该提前退出）
    MixSubgraphSplit splitter;
    Status status = splitter.RunOnFunction(*rootFuncPtr);
    // 5. 验证结果
    EXPECT_EQ(status, SUCCESS) << "Should succeed even with no mix subgraphs";
    // 5.1 验证IsMixSubgraph返回false
    for (const auto& func : nonMixFunctions) {
        bool isMix = splitter.IsMixSubgraph(*func);
        EXPECT_FALSE(isMix) << "Non-mix function should not be identified as mix";
    }
    // 5.2 验证programs保持不变
    auto& newPrograms = rootFuncPtr->programs_;
    EXPECT_EQ(newPrograms.size(), originalProgramCount) << "Program count should not change when no mix subgraphs";
    // 5.3 验证callOps数量不变
    auto newCallOps = rootFuncPtr->GetCallopList();
    EXPECT_EQ(newCallOps.size(), originalCallOpCount) << "CallOp count should not change when no mix subgraphs";
}

// 辅助函数：创建外部Mix子图
void CreateExternalMixFunction(std::shared_ptr<Function>& externalRootFuncPtr,
                               std::shared_ptr<Function>& externalMixFuncPtr)
{
    // 创建外部root function
    externalRootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "external_root", "external_root", nullptr);
    externalRootFuncPtr->rootFunc_ = externalRootFuncPtr.get();

    // 创建Mix子图
    const uint64_t externalMixProgramId = 999;
    externalMixFuncPtr = std::make_shared<Function>(Program::GetInstance(), "external_mix", "external_mix",
                                                    externalRootFuncPtr.get());
    externalMixFuncPtr->SetGraphType(GraphType::BLOCK_GRAPH);
    externalMixFuncPtr->SetFunctionType(FunctionType::STATIC);

    // 添加到programs
    externalRootFuncPtr->programs_[externalMixProgramId] = externalMixFuncPtr.get();

    // 创建Mix子图内部结构
    std::vector<int64_t> shape = {16, 16};
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto incast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outcast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    externalMixFuncPtr->inCasts_.push_back(incast1);
    externalMixFuncPtr->inCasts_.push_back(incast2);
    externalMixFuncPtr->outCasts_.push_back(outcast1);

    // 创建3个scope的op
    for (int compIdx = 0; compIdx < 3; compIdx++) {
        auto inputTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
        auto outputTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
        Opcode opcode = Opcode::OP_EXP;
        auto& op = IRBuilder().CreateTensorOpStmt(*externalMixFuncPtr, opcode, {inputTensor}, {outputTensor});
        op.UpdateInternalSubgraphID(compIdx);

        // 交替设置Cube和Vector
        if (compIdx % 2 == 0) {
            op.SetAttr(OpAttributeKey::isCube, true);
        } else {
            op.SetAIVCore((compIdx == 1) ? AIVCore::AIV0 : AIVCore::AIV1);
        }
    }

    // 计算哈希并注册到缓存
    externalMixFuncPtr->ComputeHash();
    FunctionHash mixFuncHash = externalMixFuncPtr->GetFunctionHash();
    Program::GetInstance().GetFunctionCache().Insert(mixFuncHash, *externalMixFuncPtr);
}

// 辅助函数：验证跨function调用结果
void VerifyCrossFunctionResults(MixSubgraphSplit& splitter, std::shared_ptr<Function>& rootFuncPtr,
                                std::shared_ptr<Function>& externalRootFuncPtr,
                                std::shared_ptr<Function>& externalMixFuncPtr, Operation* crossCallOp, Status status)
{
    EXPECT_EQ(status, FAILED) << "Fresh external mix function without preceding processing should fail";

    // 9.1 验证Mix子图识别
    bool isMix = splitter.IsMixSubgraph(*externalMixFuncPtr);
    EXPECT_TRUE(isMix) << "External mix function should be identified as mix";

    // 9.2 验证当前rootFunction的programs
    auto& programs = rootFuncPtr->programs_;
    // 跨function场景：新创建的leafFunction不加入当前rootFunction的programs
    EXPECT_EQ(programs.size(), 0);

    // 9.3 验证callOp创建
    auto newCallOps = rootFuncPtr->GetCallopList();
    // 应该只有原始的callOp，没有新的callOp被创建
    EXPECT_EQ(newCallOps.size(), 1) << "Should only have original call op on failure";

    // 9.4 验证原始外部Mix子图状态不变
    // 外部Mix子图不应该被修改
    EXPECT_EQ(externalRootFuncPtr->programs_.size(), 1) << "External root function's programs should remain unchanged";

    auto it = externalRootFuncPtr->programs_.find(999);
    EXPECT_NE(it, externalRootFuncPtr->programs_.end()) << "External mix function should still exist in external root";
    EXPECT_EQ(it->second, externalMixFuncPtr.get()) << "External mix function pointer should be unchanged";

    // 9.5 验证原始的callOp仍然存在
    bool originalCallOpExists = false;
    for (auto* callOp : newCallOps) {
        if (callOp == crossCallOp) {
            originalCallOpExists = true;
            break;
        }
    }
    EXPECT_TRUE(originalCallOpExists) << "Original cross-function callOp should still exist on failure";
}

/**
 * 测试Mix子图为跨function调用时的特殊处理
 */
TEST_F(MixSubgraphSplitTest, TestCrossFunctionMixSubgraph)
{
    // 1. 创建场景：Mix子图不在当前rootFunc的programs中
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "test_root_cross", "test_root_cross",
                                                  nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();

    // 2. 创建外部Mix子图（保持为shared_ptr，避免析构）
    std::shared_ptr<Function> externalRootFuncPtr;
    std::shared_ptr<Function> externalMixFuncPtr;
    CreateExternalMixFunction(externalRootFuncPtr, externalMixFuncPtr);

    // 3. 获取Mix函数hash
    FunctionHash mixFuncHash = externalMixFuncPtr->GetFunctionHash();

    // 4. 创建指向外部Mix子图的callOp
    auto& crossCallOp = IRBuilder().CreateTensorOpStmt(*rootFuncPtr, Opcode::OP_CALL, {}, {});
    auto crossCallAttr = std::make_shared<CallOpAttribute>();
    auto invokeInfo = std::make_shared<SubfuncInvokeInfoTy>();
    crossCallAttr->SetCalleeHash(mixFuncHash);
    crossCallAttr->invokeInfo_ = invokeInfo;
    crossCallOp.SetOpAttribute(crossCallAttr);

    // 5. 执行拆分
    MixSubgraphSplit splitter;
    Status status = splitter.RunOnFunction(*rootFuncPtr);

    // 6. 验证结果
    VerifyCrossFunctionResults(splitter, rootFuncPtr, externalRootFuncPtr, externalMixFuncPtr, &crossCallOp, status);
}

TEST_F(MixSubgraphSplitTest, TestDependOperand)
{
    // Build Graph
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
                                           MemoryType::MEM_UB,         MemoryType::MEM_UB,         MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
                                Opcode::OP_COPY_IN,  Opcode::OP_COPY_IN,  Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {"t1"}, {"t2"}, {"t4", "t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t3", "t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Copyin1", "Copyin2", "Add1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function* function = subGraph.GetFunction();

    // Add and check depend operand
    Operation* copyin2 = subGraph.GetOp("Copyin2");
    std::shared_ptr<LogicalTensor> tensor4 = subGraph.GetTensor("t4");
    copyin2->AddDependOperand(tensor4);
    Operation* add1 = subGraph.GetOp("Add1");
    std::shared_ptr<LogicalTensor> tensor3 = subGraph.GetTensor("t3");
    add1->AddDependOperand(tensor3);
    EXPECT_EQ(copyin2->GetDependOperands().front()->GetMagic(), MS_NUM3);
    EXPECT_EQ(copyin2->GetDependOperandSize(), 1);

    // Check depend
    tensor4->AddDependOp(copyin2);
    tensor4->AddDependOp(copyin2);
    EXPECT_EQ(tensor4->GetDependOps().size(), 1);
    tensor3->AddDependOp(add1);
    auto dependOp = *(tensor4->GetDependOps().begin());
    EXPECT_EQ(dependOp->GetOpMagic(), MS_NUM10005);
    EXPECT_EQ(tensor4->HasDependOp(copyin2), true);

    // Sort Operations
    function->SortOperations();
    Operation* alloc2 = subGraph.GetOp("Alloc2");
    auto sortedOpList = function->Operations().DuplicatedOpList();
    auto alloc2Iter = std::find(sortedOpList.begin(), sortedOpList.end(), alloc2);
    auto copyin2Iter = std::find(sortedOpList.begin(), sortedOpList.end(), copyin2);
    EXPECT_EQ(alloc2Iter - sortedOpList.begin() < copyin2Iter - sortedOpList.begin(), true);

    // Erase operands and depend Ops
    copyin2->EraseDependTensor(tensor4);
    add1->EraseDependTensor(tensor3);
    tensor4->RemoveDependOp(copyin2);
    tensor3->RemoveDependOp(add1);

    // Sort Operations
    function->SortOperations();
    auto sortedOpList2 = function->Operations().DuplicatedOpList();
    auto alloc2Iter2 = std::find(sortedOpList2.begin(), sortedOpList2.end(), alloc2);
    auto copyin2Iter2 = std::find(sortedOpList2.begin(), sortedOpList2.end(), copyin2);
    EXPECT_EQ(alloc2Iter2 - sortedOpList2.begin() > copyin2Iter2 - sortedOpList2.begin(), true);

    // Erase Operations
    copyin2->AddDependOperand(tensor4);
    tensor4->AddDependOp(copyin2);
    copyin2->SetAsDeleted();
    function->EraseOperations();
    EXPECT_EQ(tensor4->GetDependOps().size(), 0);
}

namespace {
// 构造 RUNTIME_GetTensorDataInt32Dim1(index, ioType, ioTypeIndex, address, offset...) 形式的调用表达式，
// 使 GetTensorDataUsage 能按 (ioType, ioTypeIndex) 提取引用（ioType=0 即 INCAST）
SymbolicScalar CreateGetTensorDataCall(IRBuilder& builder, int ioType, int ioTypeIndex)
{
    SymbolicScalar handler(AddRuntimePrefix("GetTensorDataInt32Dim1"));
    SymbolicScalar addr = SymbolicScalar(AddRuntimeCoaPrefix("GET_PARAM_ADDR"))(builder.CreateConstInt(-1),
                                                                                builder.CreateConstInt(110));
    return handler(builder.CreateConstInt(0), builder.CreateConstInt(ioType), builder.CreateConstInt(ioTypeIndex), addr,
                   builder.CreateConstInt(0));
}

// 未锚定变体：address为immediate占位符(未经Pass_30 depend链回填)，下标不在函数IO空间
SymbolicScalar CreateUnanchoredGetTensorDataCall(IRBuilder& builder, int ioType, int ioTypeIndex)
{
    SymbolicScalar handler(AddRuntimePrefix("GetTensorDataInt32Dim1"));
    return handler(builder.CreateConstInt(0), builder.CreateConstInt(ioType), builder.CreateConstInt(ioTypeIndex),
                   builder.CreateConstInt(0), builder.CreateConstInt(0));
}
} // namespace

// CollectGetTensorDataIncasts：MixSubgraphSplit值依赖补回的看护用例
class MixDependencyAnalyzerTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        mixFunc = std::make_shared<Function>(Program::GetInstance(), "test_mix_func", "test_mix_func", nullptr);
        mixFunc->SetGraphType(GraphType::BLOCK_GRAPH);
        mixFunc->SetFunctionType(FunctionType::STATIC);
        analyzer = std::make_unique<MixDependencyAnalyzer>();
    }

    void TearDown() override
    {
        analyzer.reset();
        mixFunc.reset();
    }

    // 创建incast并登记到原Mix函数：incastIdx即期望被补回的引用下标
    std::shared_ptr<LogicalTensor> CreateAndRegisterIncast(const std::vector<int64_t>& shape)
    {
        auto tensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_INT32, shape, CreateTestConstIntVector(shape));
        mixFunc->inCasts_.push_back(tensor);
        return tensor;
    }

    // 在原Mix函数中创建携带 GetTensorData 引用的op（借 OP_VEC_DUP 的 dynScalar 动态属性通道）
    Operation& CreateOpWithGetTensorDataRef(const std::shared_ptr<LogicalTensor>& input,
                                            const std::shared_ptr<LogicalTensor>& output, int ioType, int ioTypeIndex)
    {
        auto& op = IRBuilder().CreateTensorOpStmt(*mixFunc, Opcode::OP_VEC_DUP, {input}, {output});
        op.SetAttribute(OpAttributeKey::dynScalar, CreateGetTensorDataCall(builder, ioType, ioTypeIndex));
        return op;
    }

protected:
    std::shared_ptr<Function> mixFunc;
    std::unique_ptr<MixDependencyAnalyzer> analyzer;
    IRBuilder builder;
};

// 正常补回：引用的incast被加入每个component的allIncasts
TEST_F(MixDependencyAnalyzerTest, TestCollectGetTensorDataIncastsNormal)
{
    const std::vector<int64_t> shape = {16, 16};
    auto incast = CreateAndRegisterIncast(shape);
    auto internal = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    CreateOpWithGetTensorDataRef(internal, output, GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST, 0);

    std::vector<InternalComponentInfo> components = {{0, "comp_0"}, {1, "comp_1"}};

    analyzer->CollectGetTensorDataIncasts(components, mixFunc.get());

    EXPECT_EQ(analyzer->allIncasts.size(), 2) << "Both components should receive the incast";
    for (size_t comp = 0; comp < components.size(); comp++) {
        const auto& incasts = analyzer->allIncasts[comp];
        ASSERT_EQ(incasts.size(), 1) << "Component " << comp << " should have exactly one incast";
        EXPECT_EQ(incasts[0].tensor, incast) << "Component " << comp << " should reference the registered incast";
    }
}

// idx越界跳过：引用的incast下标超出原函数incast列表
TEST_F(MixDependencyAnalyzerTest, TestCollectGetTensorDataIncastsIndexOutOfRange)
{
    const std::vector<int64_t> shape = {16, 16};
    auto internal = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    // incast列表为空，ioTypeIndex=0必然越界
    CreateOpWithGetTensorDataRef(internal, output, GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST, 0);

    std::vector<InternalComponentInfo> components = {{0, "comp_0"}};

    analyzer->CollectGetTensorDataIncasts(components, mixFunc.get());

    EXPECT_TRUE(analyzer->allIncasts.empty()) << "Out-of-range incast index should be skipped";
}

// incast已存在去重：component的allIncasts中已有该tensor时不重复添加
TEST_F(MixDependencyAnalyzerTest, TestCollectGetTensorDataIncastsDeduplicate)
{
    const std::vector<int64_t> shape = {16, 16};
    auto incast = CreateAndRegisterIncast(shape);
    auto internal = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    CreateOpWithGetTensorDataRef(internal, output, GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST, 0);

    std::vector<InternalComponentInfo> components = {{0, "comp_0"}};
    // 预置该incast已存在于component 0
    analyzer->allIncasts[0].emplace_back(incast, -1, 0);

    analyzer->CollectGetTensorDataIncasts(components, mixFunc.get());

    ASSERT_EQ(analyzer->allIncasts[0].size(), 1) << "Existing incast should not be duplicated";
    EXPECT_EQ(analyzer->allIncasts[0][0].tensor, incast);
}

// originalMixFunc为null：直接返回，不产生任何依赖
TEST_F(MixDependencyAnalyzerTest, TestCollectGetTensorDataIncastsNullFunction)
{
    std::vector<InternalComponentInfo> components = {{0, "comp_0"}};

    analyzer->CollectGetTensorDataIncasts(components, nullptr);

    EXPECT_TRUE(analyzer->allIncasts.empty()) << "Null originalMixFunc should be skipped";
}

// 拆分后依赖传播语义：incast正向传播保留（V leaf持有C传播来的incast）；
// C已有incast，依赖已建立，不再反向传播outcast给它
TEST_F(MixSubgraphSplitTest, TestPerLeafIncastOutcastAfterSplit)
{
    // 复用基础拆分场景：C读incast1/incast2，V读incast3并写出outcast1
    auto rootFuncPtr = CreateTestRootFunction();

    FunctionHash mixFuncHash;
    auto mixFuncPtr = CreateMixSubgraph(rootFuncPtr, mixFuncHash);

    const uint64_t mixProgramId = 100;
    auto& callOp = CreateCallOp(rootFuncPtr, mixProgramId, mixFuncHash);
    (void)callOp;

    MixSubgraphSplit splitter;
    Status status = splitter.RunOnFunction(*rootFuncPtr);
    ASSERT_EQ(status, SUCCESS) << "MixSubgraphSplit should succeed";

    // 按aivCore区分C leaf与V leaf
    Function* cubeLeaf = nullptr;
    Function* vectorLeaf = nullptr;
    for (const auto& [progId, func] : rootFuncPtr->programs_) {
        (void)progId;
        auto leafAttr = func->GetLeafFuncAttribute();
        ASSERT_NE(leafAttr, nullptr);
        if (leafAttr->aivCore == AIVCore::UNSPECIFIED) {
            cubeLeaf = func;
        } else {
            vectorLeaf = func;
        }
    }
    ASSERT_NE(cubeLeaf, nullptr) << "Cube leaf should exist";
    ASSERT_NE(vectorLeaf, nullptr) << "Vector leaf should exist";

    // 原Mix子图的边界tensor
    const auto& originalIncasts = mixFuncPtr->GetIncast();
    const auto& originalOutcasts = mixFuncPtr->GetOutcast();
    ASSERT_EQ(originalIncasts.size(), 3);
    ASSERT_EQ(originalOutcasts.size(), 1);

    auto leafContains = [](const std::vector<std::shared_ptr<LogicalTensor>>& tensors,
                           const std::shared_ptr<LogicalTensor>& target) {
        return std::find(tensors.begin(), tensors.end(), target) != tensors.end();
    };

    // C leaf：持有自己读取的incast1/incast2，不持有incast3；
    // C已有incast（依赖已建立），outcast不反向传播给它
    const auto& cubeIncasts = cubeLeaf->GetIncast();
    const auto& cubeOutcasts = cubeLeaf->GetOutcast();
    EXPECT_TRUE(leafContains(cubeIncasts, originalIncasts[0])) << "Cube leaf should own incast1";
    EXPECT_TRUE(leafContains(cubeIncasts, originalIncasts[1])) << "Cube leaf should own incast2";
    EXPECT_FALSE(leafContains(cubeIncasts, originalIncasts[2])) << "Cube leaf should not own incast3";
    EXPECT_FALSE(leafContains(cubeOutcasts, originalOutcasts[0])) << "Cube leaf should not own outcast1";

    // V leaf：持有incast3和outcast1；incast正向传播保留，同时持有C传播来的incast1/incast2
    const auto& vectorIncasts = vectorLeaf->GetIncast();
    const auto& vectorOutcasts = vectorLeaf->GetOutcast();
    EXPECT_TRUE(leafContains(vectorIncasts, originalIncasts[0])) << "Vector leaf should hold propagated incast1";
    EXPECT_TRUE(leafContains(vectorIncasts, originalIncasts[1])) << "Vector leaf should hold propagated incast2";
    EXPECT_TRUE(leafContains(vectorIncasts, originalIncasts[2])) << "Vector leaf should own incast3";
    EXPECT_TRUE(leafContains(vectorOutcasts, originalOutcasts[0])) << "Vector leaf should own outcast1";
}

namespace {
// 拆分前场景：基础mix子图(C/V两组件) + 追加的符号源incast + 携带(INCAST,N)引用的VEC_DUP op + call op
struct GtdSplitScene {
    std::shared_ptr<Function> rootFunc;
    std::shared_ptr<Function> mixFunc;
    std::shared_ptr<LogicalTensor> gtdIncast;
    int oldIncastIdx = -1;
    Operation* gtdOp = nullptr;
};

SymbolicScalar BuildAnchoredIncastGtdCall(IRBuilder& builder, int incastIdx)
{
    return CreateGetTensorDataCall(builder, GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST, incastIdx);
}

GtdSplitScene CreateGtdSplitScene(SymbolicScalar (*buildCall)(IRBuilder&, int) = BuildAnchoredIncastGtdCall)
{
    GtdSplitScene scene;
    scene.rootFunc = CreateTestRootFunction();
    FunctionHash mixFuncHash;
    scene.mixFunc = CreateMixSubgraph(scene.rootFunc, mixFuncHash);
    // 追加第4个incast，并在C scope放置携带GetTensorData(INCAST,N)引用的VEC_DUP op
    const std::vector<int64_t> shape = {MS_NUM16, MS_NUM16};
    scene.gtdIncast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    scene.oldIncastIdx = static_cast<int>(scene.mixFunc->inCasts_.size());
    scene.mixFunc->inCasts_.push_back(scene.gtdIncast);
    auto internal = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    IRBuilder builder;
    auto& gtdOp = IRBuilder().CreateTensorOpStmt(*scene.mixFunc, Opcode::OP_VEC_DUP, {internal}, {output});
    gtdOp.SetAttribute(OpAttributeKey::dynScalar, buildCall(builder, scene.oldIncastIdx));
    gtdOp.UpdateInternalSubgraphID(0);
    gtdOp.SetAttr(OpAttributeKey::isCube, true);
    scene.gtdOp = &gtdOp;
    const uint64_t mixProgramId = 100;
    auto& callOp = CreateCallOp(scene.rootFunc, mixProgramId, mixFuncHash);
    (void)callOp;
    return scene;
}

// 取op符号中首个GetTensorData调用的address操作数
RawSymbolicScalarPtr GetFirstGtdAddressOperand(Operation& op)
{
    for (const auto& [index, callList] : GetTensorDataDict(op.GetDynamicAttributeList())) {
        (void)index;
        for (const auto& call : callList) {
            return call->GetExpressionOperandList()[GET_TENSOR_DATA_OPERAND_INDEX_ADDRESS];
        }
    }
    return nullptr;
}

// 定位包含指定op的leaf（op指针在拆分迁移中保持不变）
Function* FindLeafHostingOp(Function& rootFunc, Operation* op)
{
    for (const auto& [progId, func] : rootFunc.programs_) {
        (void)progId;
        for (auto& curOp : func->Operations(false)) {
            if (&curOp == op) {
                return func;
            }
        }
    }
    return nullptr;
}
} // namespace

// 拆分后op中的(INCAST,N)引用须重映射为所在leaf的incast下标且address不变
TEST_F(MixSubgraphSplitTest, TestGetTensorDataIORemappedAfterSplit)
{
    GtdSplitScene scene = CreateGtdSplitScene();

    // 拆分前：引用指向原Mix函数的incast下标
    auto usageBefore = GetTensorDataUsage(scene.gtdOp->GetDynamicAttributeList());
    ASSERT_EQ(usageBefore.size(), 1) << "Sanity: op should carry exactly one GetTensorData usage before split";
    EXPECT_EQ(usageBefore.count({GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST, scene.oldIncastIdx}), 1);
    auto addressBefore = GetFirstGtdAddressOperand(*scene.gtdOp);
    ASSERT_NE(addressBefore, nullptr);

    MixSubgraphSplit splitter;
    ASSERT_EQ(splitter.RunOnFunction(*scene.rootFunc), SUCCESS) << "MixSubgraphSplit should succeed";

    // 定位包含该op的leaf，其incast持有被引用张量且下标与原Mix函数不同
    Function* hostLeaf = FindLeafHostingOp(*scene.rootFunc, scene.gtdOp);
    ASSERT_NE(hostLeaf, nullptr) << "The op with GetTensorData reference should be moved into a new leaf";
    const auto& leafIncasts = hostLeaf->GetIncast();
    auto incastIt = std::find(leafIncasts.begin(), leafIncasts.end(), scene.gtdIncast);
    ASSERT_NE(incastIt, leafIncasts.end()) << "Leaf should own the GetTensorData-referenced incast";
    const int newIncastIdx = static_cast<int>(std::distance(leafIncasts.begin(), incastIt));
    ASSERT_NE(newIncastIdx, scene.oldIncastIdx) << "Test setup: new index should differ from original mix-func index";

    // 重映射后：op符号指向该leaf自己的incast下标，address操作数保持不变（设备端按COA address取数）
    auto usageAfter = GetTensorDataUsage(scene.gtdOp->GetDynamicAttributeList());
    ASSERT_EQ(usageAfter.size(), 1) << "Remap should not change the usage count";
    EXPECT_EQ(usageAfter.count({GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST, newIncastIdx}), 1)
        << "GetTensorData incast index should be remapped to the owning leaf's incast list";
    auto addressAfter = GetFirstGtdAddressOperand(*scene.gtdOp);
    ASSERT_NE(addressAfter, nullptr);
    EXPECT_EQ(addressAfter, addressBefore) << "Address operand should be preserved by the remap";
}

// 未锚定符号(address为immediate占位符)：跳过不改写，拆分正常完成
TEST_F(MixSubgraphSplitTest, TestGetTensorDataRemapSkipsUnanchoredSymbol)
{
    GtdSplitScene scene = CreateGtdSplitScene([](IRBuilder& builder, int incastIdx) {
        return CreateUnanchoredGetTensorDataCall(builder, GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST, incastIdx);
    });

    MixSubgraphSplit splitter;
    ASSERT_EQ(splitter.RunOnFunction(*scene.rootFunc), SUCCESS);

    // 符号随op迁入leaf后保持原样(下标未被改写)
    auto usage = GetTensorDataUsage(scene.gtdOp->GetDynamicAttributeList());
    EXPECT_EQ(usage.count({GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST, scene.oldIncastIdx}), 1);
}

// OUTCAST属未锚定残留绑定：跳过不改写
TEST_F(MixSubgraphSplitTest, TestGetTensorDataRemapSkipsOutcastSymbol)
{
    GtdSplitScene scene = CreateGtdSplitScene([](IRBuilder& builder, int incastIdx) {
        return CreateGetTensorDataCall(builder, GET_TENSOR_DATA_OPERAND_IOTYPE_OUTCAST, incastIdx);
    });

    MixSubgraphSplit splitter;
    ASSERT_EQ(splitter.RunOnFunction(*scene.rootFunc), SUCCESS);

    auto usage = GetTensorDataUsage(scene.gtdOp->GetDynamicAttributeList());
    EXPECT_EQ(usage.count({GET_TENSOR_DATA_OPERAND_IOTYPE_OUTCAST, scene.oldIncastIdx}), 1);
}

// 已锚定符号旧下标越界：下标语义未知，须编译期报错而非静默保留
TEST_F(MixSubgraphSplitTest, TestGetTensorDataRemapFailsOnOutOfRangeIndex)
{
    GtdSplitScene scene = CreateGtdSplitScene([](IRBuilder& builder, int incastIdx) {
        (void)incastIdx;
        // 指向原Mix函数incast列表(4项)之外的固定下标
        return CreateGetTensorDataCall(builder, GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST, 99);
    });

    MixSubgraphSplit splitter;
    ASSERT_EQ(splitter.RunOnFunction(*scene.rootFunc), FAILED);
}

// 符号仅由tensor dvs承载(analyzer的attr扫描不可见)，源incast不进新leaf：解析失败须报错
TEST_F(MixSubgraphSplitTest, TestGetTensorDataRemapFailsWhenLeafMissesSourceIncast)
{
    auto rootFuncPtr = CreateTestRootFunction();
    FunctionHash mixFuncHash;
    auto mixFuncPtr = CreateMixSubgraph(rootFuncPtr, mixFuncHash);
    const std::vector<int64_t> shape = {MS_NUM16, MS_NUM16};

    // 追加不被任何op消费的源incast；符号放进carrier的dvs，carrier由vector scope的普通op消费
    auto gtdIncast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    const int oldIncastIdx = static_cast<int>(mixFuncPtr->inCasts_.size());
    mixFuncPtr->inCasts_.push_back(gtdIncast);
    IRBuilder builder;
    auto dvsCarrier = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    dvsCarrier->UpdateDynValidShape(
        {CreateGetTensorDataCall(builder, GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST, oldIncastIdx)});
    auto carrierOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto& carrierOp = IRBuilder().CreateTensorOpStmt(*mixFuncPtr, Opcode::OP_ABS, {dvsCarrier}, {carrierOut});
    carrierOp.UpdateInternalSubgraphID(1);

    const uint64_t mixProgramId = 100;
    auto& callOp = CreateCallOp(rootFuncPtr, mixProgramId, mixFuncHash);
    (void)callOp;

    MixSubgraphSplit splitter;
    ASSERT_EQ(splitter.RunOnFunction(*rootFuncPtr), FAILED);
}

// 共享张量场景：两leaf共享的carrier dvs带符号引用，改写后下标仍须解析回源张量
TEST_F(MixSubgraphSplitTest, TestGetTensorDataSharedTensorDvsRemappedToSource)
{
    auto rootFuncPtr = CreateTestRootFunction();
    FunctionHash mixFuncHash;
    auto mixFuncPtr = CreateMixSubgraph(rootFuncPtr, mixFuncHash);
    const std::vector<int64_t> shape = {MS_NUM16, MS_NUM16};

    auto gtdIncast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    const int oldIncastIdx = static_cast<int>(mixFuncPtr->inCasts_.size());
    mixFuncPtr->inCasts_.push_back(gtdIncast);

    // carrier的dvs携带锚定的(INCAST, oldIncastIdx)引用；carrier与gtdIncast均被两个组件消费
    IRBuilder builder;
    auto carrier = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    carrier->UpdateDynValidShape(
        {CreateGetTensorDataCall(builder, GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST, oldIncastIdx)});
    for (int componentId = 0; componentId < 2; componentId++) {
        auto tmp = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
        auto& consumeOp = IRBuilder().CreateTensorOpStmt(*mixFuncPtr, Opcode::OP_ABS, {carrier, gtdIncast}, {tmp});
        consumeOp.UpdateInternalSubgraphID(componentId);
        consumeOp.SetAttr(OpAttributeKey::isCube, componentId == 0);
    }

    const uint64_t mixProgramId = 100;
    auto& callOp = CreateCallOp(rootFuncPtr, mixProgramId, mixFuncHash);
    (void)callOp;

    MixSubgraphSplit splitter;
    ASSERT_EQ(splitter.RunOnFunction(*rootFuncPtr), SUCCESS);

    // 改写后carrier dvs中的下标必须仍能解析到某个新leaf中的gtdIncast
    auto usage = GetTensorDataUsage(std::vector<std::reference_wrapper<SymbolicScalar>>(
        carrier->GetDynValidShape().begin(), carrier->GetDynValidShape().end()));
    ASSERT_EQ(usage.size(), 1);
    const auto& finalBinding = *usage.begin();
    EXPECT_EQ(finalBinding.first, GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST);
    bool resolvesToSource = false;
    for (const auto& [progId, func] : rootFuncPtr->programs_) {
        (void)progId;
        if (func == nullptr) {
            continue;
        }
        std::shared_ptr<LogicalTensor> gtdCopy = gtdIncast;
        if (func->GetIncastIndex(gtdCopy) == finalBinding.second) {
            resolvesToSource = true;
            break;
        }
    }
    EXPECT_TRUE(resolvesToSource) << "remapped index " << finalBinding.second
                                  << " should resolve to the source incast in a split leaf";
}

// DynParamTable表内条目随拆分克隆进新leaf，须重映射到所在leaf的incast下标
TEST_F(MixSubgraphSplitTest, TestGetTensorDataDynParamTableRemappedAfterSplit)
{
    GtdSplitScene scene = CreateGtdSplitScene();
    // 在原Mix函数的DynParamTable登记携带(INCAST, oldIncastIdx)引用的符号维
    IRBuilder builder;
    scene.mixFunc->GetMutableDynParam("dim_0").dim = CreateGetTensorDataCall(
        builder, GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST, scene.oldIncastIdx);

    MixSubgraphSplit splitter;
    ASSERT_EQ(splitter.RunOnFunction(*scene.rootFunc), SUCCESS);

    // 定位符号携带op所在leaf, 其表内条目应指向该leaf自己的incast下标
    Function* hostLeaf = FindLeafHostingOp(*scene.rootFunc, scene.gtdOp);
    ASSERT_NE(hostLeaf, nullptr);
    const auto& leafIncasts = hostLeaf->GetIncast();
    auto incastIt = std::find(leafIncasts.begin(), leafIncasts.end(), scene.gtdIncast);
    ASSERT_NE(incastIt, leafIncasts.end()) << "Leaf should own the GetTensorData-referenced incast";
    const int newIncastIdx = static_cast<int>(std::distance(leafIncasts.begin(), incastIt));
    ASSERT_NE(newIncastIdx, scene.oldIncastIdx) << "Test setup: new index should differ from mix-func index";

    const auto& leafTable = hostLeaf->GetDynParamTable();
    auto entryIt = leafTable.find("dim_0");
    ASSERT_NE(entryIt, leafTable.end()) << "Cloned leaf should carry the dyn param table entry";
    ASSERT_TRUE(entryIt->second.dim.IsValid());
    bool foundRemapped = false;
    for (const auto& [index, callList] : GetTensorDataDict(entryIt->second.dim)) {
        (void)index;
        for (const auto& call : callList) {
            const auto& operandList = call->GetExpressionOperandList();
            if (!operandList[GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE]->IsImmediate() ||
                !operandList[GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE_INDEX]->IsImmediate()) {
                continue;
            }
            EXPECT_EQ(static_cast<int>(operandList[GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE]->GetImmediateValue()),
                      GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST);
            EXPECT_EQ(static_cast<int>(operandList[GET_TENSOR_DATA_OPERAND_INDEX_IOTYPE_INDEX]->GetImmediateValue()),
                      newIncastIdx);
            foundRemapped = true;
            break;
        }
        break;
    }
    EXPECT_TRUE(foundRemapped) << "dyn param table entry should carry the remapped (ioType, ioTypeIndex)";
}

// outcast反向传播条件：source已有incast（依赖已建立）时不再传播；
// 仅无incast的source才接收反向传播，incast正向传播保留
TEST_F(MixDependencyAnalyzerTest, TestPropagateOutcastSkippedWhenIncastExists)
{
    const std::vector<int64_t> shape = {16, 16};
    auto incast = CreateAndRegisterIncast(shape);
    auto outcastA = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outcastB = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    // component 0持有incast；component 1无incast；component 2的outcast待反向传播
    analyzer->allIncasts[0].emplace_back(incast, -1, 0);
    analyzer->allOutcasts[2].emplace_back(outcastA, -1, 0);
    analyzer->allOutcasts[2].emplace_back(outcastB, -1, 0);

    // 依赖闭包：0 -> {2}，1 -> {2}
    std::unordered_map<int, std::set<int>> closure;
    closure[0] = {2};
    closure[1] = {2};

    analyzer->PropagateExternalDependenciesWithClosure(closure);

    auto contains = [](const std::vector<SimpleTensorParam>& params, const LogicalTensorPtr& tensor) {
        return std::any_of(params.begin(), params.end(),
                           [&tensor](const SimpleTensorParam& param) { return param.tensor == tensor; });
    };
    // 已有incast的component 0：不接收反向传播的outcast
    EXPECT_FALSE(contains(analyzer->allOutcasts[0], outcastA));
    EXPECT_FALSE(contains(analyzer->allOutcasts[0], outcastB));
    // 无incast的component 1：正常接收反向传播的outcast
    EXPECT_TRUE(contains(analyzer->allOutcasts[1], outcastA));
    EXPECT_TRUE(contains(analyzer->allOutcasts[1], outcastB));
    // incast正向传播保留：component 2获得component 0的incast
    EXPECT_TRUE(contains(analyzer->allIncasts[2], incast));
}

// ReattributeInplaceOutcasts：outcast挂在inplace视图算子（RESHAPE）上时，
// 重归属到数据生产者所在组件；源组件invokeInfo记录同步移除，目标组件invokeInfo不添加
TEST_F(MixDependencyAnalyzerTest, TestReattributeInplaceOutcastsToProducerComponent)
{
    const std::vector<int64_t> shape = {16, 16};
    auto producerInput = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto gmTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outcastTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& producerOp = IRBuilder().CreateTensorOpStmt(*mixFunc, Opcode::OP_VEC_DUP, {producerInput}, {gmTensor});
    auto& reshapeOp = IRBuilder().CreateTensorOpStmt(*mixFunc, Opcode::OP_RESHAPE, {gmTensor}, {outcastTensor});

    std::vector<InternalComponentInfo> components = {{0, "comp_0"}, {1, "comp_1"}};
    components[0].operations = {&producerOp};
    components[1].operations = {&reshapeOp};

    analyzer->InitSubgraphToFunction(components);
    analyzer->allOutcasts[1].emplace_back(outcastTensor, reshapeOp.GetOpMagic(), 0);
    auto& sourceInvokeInfo = analyzer->subgraphToFunction.subFuncInvokeInfos[1];
    sourceInvokeInfo.tensorParamList_.emplace_back(0, outcastTensor->GetRawMagic(), std::vector<int64_t>{}, shape,
                                                   shape, DT_FP32, true, outcastTensor, reshapeOp.GetOpMagic(), 0);

    ASSERT_EQ(analyzer->ReattributeInplaceOutcasts(mixFunc.get(), components), SUCCESS);

    auto contains = [](const std::vector<SimpleTensorParam>& params, const LogicalTensorPtr& tensor) {
        return std::any_of(params.begin(), params.end(),
                           [&tensor](const SimpleTensorParam& param) { return param.tensor == tensor; });
    };
    EXPECT_TRUE(contains(analyzer->allOutcasts[0], outcastTensor)) << "Outcast should move to producer component";
    EXPECT_FALSE(contains(analyzer->allOutcasts[1], outcastTensor)) << "Outcast should leave the anchor component";
    EXPECT_TRUE(sourceInvokeInfo.tensorParamList_.empty()) << "Source invokeInfo record should be removed";
    EXPECT_TRUE(analyzer->subgraphToFunction.subFuncInvokeInfos[0].tensorParamList_.empty())
        << "Target invokeInfo should not receive the record";
}

// ReattributeInplaceOutcasts：视图输入由多个组件共同生产时，outcast分别挂到每个生产者组件
TEST_F(MixDependencyAnalyzerTest, TestReattributeInplaceOutcastsMultiProducer)
{
    const std::vector<int64_t> shape = {16, 16};
    auto inputA = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto inputB = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto sharedTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outcastTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& producerA = IRBuilder().CreateTensorOpStmt(*mixFunc, Opcode::OP_VEC_DUP, {inputA}, {sharedTensor});
    auto& producerB = IRBuilder().CreateTensorOpStmt(*mixFunc, Opcode::OP_VEC_DUP, {inputB}, {sharedTensor});
    auto& reshapeOp = IRBuilder().CreateTensorOpStmt(*mixFunc, Opcode::OP_RESHAPE, {sharedTensor}, {outcastTensor});

    std::vector<InternalComponentInfo> components = {{0, "comp_0"}, {1, "comp_1"}, {2, "comp_2"}};
    components[0].operations = {&producerA};
    components[1].operations = {&reshapeOp};
    components[2].operations = {&producerB};

    analyzer->InitSubgraphToFunction(components);
    analyzer->allOutcasts[1].emplace_back(outcastTensor, reshapeOp.GetOpMagic(), 0);

    ASSERT_EQ(analyzer->ReattributeInplaceOutcasts(mixFunc.get(), components), SUCCESS);

    auto contains = [](const std::vector<SimpleTensorParam>& params, const LogicalTensorPtr& tensor) {
        return std::any_of(params.begin(), params.end(),
                           [&tensor](const SimpleTensorParam& param) { return param.tensor == tensor; });
    };
    EXPECT_TRUE(contains(analyzer->allOutcasts[0], outcastTensor)) << "Outcast should hang on producer A component";
    EXPECT_TRUE(contains(analyzer->allOutcasts[2], outcastTensor)) << "Outcast should hang on producer B component";
    EXPECT_FALSE(contains(analyzer->allOutcasts[1], outcastTensor)) << "Outcast should leave the anchor component";
}

// ReattributeInplaceOutcasts：多级视图链（RESHAPE←VIEW←非inplace生产者）递归穿透；
// 源组件invokeInfo的outcastTensorParamList_与tensorParamList_记录同步清理
TEST_F(MixDependencyAnalyzerTest, TestReattributeInplaceOutcastsChainedViews)
{
    const std::vector<int64_t> shape = {16, 16};
    auto producerInput = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto gmTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto viewTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outcastTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& producerOp = IRBuilder().CreateTensorOpStmt(*mixFunc, Opcode::OP_VEC_DUP, {producerInput}, {gmTensor});
    auto& viewOp = IRBuilder().CreateTensorOpStmt(*mixFunc, Opcode::OP_VIEW, {gmTensor}, {viewTensor});
    auto& reshapeOp = IRBuilder().CreateTensorOpStmt(*mixFunc, Opcode::OP_RESHAPE, {viewTensor}, {outcastTensor});

    std::vector<InternalComponentInfo> components = {{0, "comp_0"}, {1, "comp_1"}};
    components[0].operations = {&producerOp};
    components[1].operations = {&viewOp, &reshapeOp};

    analyzer->InitSubgraphToFunction(components);
    analyzer->allOutcasts[1].emplace_back(outcastTensor, reshapeOp.GetOpMagic(), 0);
    auto& sourceInvokeInfo = analyzer->subgraphToFunction.subFuncInvokeInfos[1];
    sourceInvokeInfo.outcastTensorParamList_.emplace_back(0, outcastTensor->GetRawMagic(), 0, shape, shape,
                                                          std::vector<int64_t>{}, DT_FP32, outcastTensor,
                                                          reshapeOp.GetOpMagic(), 0);
    sourceInvokeInfo.tensorParamList_.emplace_back(0, outcastTensor->GetRawMagic(), std::vector<int64_t>{}, shape,
                                                   shape, DT_FP32, true, outcastTensor, reshapeOp.GetOpMagic(), 0);

    ASSERT_EQ(analyzer->ReattributeInplaceOutcasts(mixFunc.get(), components), SUCCESS);

    auto contains = [](const std::vector<SimpleTensorParam>& params, const LogicalTensorPtr& tensor) {
        return std::any_of(params.begin(), params.end(),
                           [&tensor](const SimpleTensorParam& param) { return param.tensor == tensor; });
    };
    EXPECT_TRUE(contains(analyzer->allOutcasts[0], outcastTensor))
        << "Outcast should reach producer through view chain";
    EXPECT_FALSE(contains(analyzer->allOutcasts[1], outcastTensor)) << "Outcast should leave the anchor component";
    EXPECT_TRUE(sourceInvokeInfo.outcastTensorParamList_.empty()) << "Outcast pack should be removed from invokeInfo";
    EXPECT_TRUE(sourceInvokeInfo.tensorParamList_.empty()) << "GM tensor pack should be removed from invokeInfo";
}

// ReattributeInplaceOutcasts：纯透传（输入无函数内生产者）保持原归属
TEST_F(MixDependencyAnalyzerTest, TestReattributeInplaceOutcastsPassthroughKept)
{
    const std::vector<int64_t> shape = {16, 16};
    auto incastTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outcastTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& reshapeOp = IRBuilder().CreateTensorOpStmt(*mixFunc, Opcode::OP_RESHAPE, {incastTensor}, {outcastTensor});

    std::vector<InternalComponentInfo> components = {{0, "comp_0"}, {1, "comp_1"}};
    components[1].operations = {&reshapeOp};

    analyzer->InitSubgraphToFunction(components);
    analyzer->allOutcasts[1].emplace_back(outcastTensor, reshapeOp.GetOpMagic(), 0);

    ASSERT_EQ(analyzer->ReattributeInplaceOutcasts(mixFunc.get(), components), SUCCESS);

    auto contains = [](const std::vector<SimpleTensorParam>& params, const LogicalTensorPtr& tensor) {
        return std::any_of(params.begin(), params.end(),
                           [&tensor](const SimpleTensorParam& param) { return param.tensor == tensor; });
    };
    EXPECT_TRUE(contains(analyzer->allOutcasts[1], outcastTensor)) << "Passthrough outcast should keep its component";
    EXPECT_FALSE(contains(analyzer->allOutcasts[0], outcastTensor)) << "No producer exists, nothing to reattribute";
}

// ReattributeInplaceOutcasts：本组件也是生产者之一（producerComps含compId）时保留原归属，
// 其余生产者组件由pending补挂，源组件invokeInfo记录不动
TEST_F(MixDependencyAnalyzerTest, TestReattributeInplaceOutcastsSelfProducerKept)
{
    const std::vector<int64_t> shape = {16, 16};
    auto inputA = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto inputB = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto sharedTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outcastTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& producerA = IRBuilder().CreateTensorOpStmt(*mixFunc, Opcode::OP_VEC_DUP, {inputA}, {sharedTensor});
    auto& producerB = IRBuilder().CreateTensorOpStmt(*mixFunc, Opcode::OP_VEC_DUP, {inputB}, {sharedTensor});
    auto& reshapeOp = IRBuilder().CreateTensorOpStmt(*mixFunc, Opcode::OP_RESHAPE, {sharedTensor}, {outcastTensor});

    std::vector<InternalComponentInfo> components = {{0, "comp_0"}, {1, "comp_1"}};
    components[0].operations = {&producerA};
    components[1].operations = {&producerB, &reshapeOp};

    analyzer->InitSubgraphToFunction(components);
    analyzer->allOutcasts[1].emplace_back(outcastTensor, reshapeOp.GetOpMagic(), 0);
    auto& sourceInvokeInfo = analyzer->subgraphToFunction.subFuncInvokeInfos[1];
    sourceInvokeInfo.tensorParamList_.emplace_back(0, outcastTensor->GetRawMagic(), std::vector<int64_t>{}, shape,
                                                   shape, DT_FP32, true, outcastTensor, reshapeOp.GetOpMagic(), 0);

    ASSERT_EQ(analyzer->ReattributeInplaceOutcasts(mixFunc.get(), components), SUCCESS);

    auto contains = [](const std::vector<SimpleTensorParam>& params, const LogicalTensorPtr& tensor) {
        return std::any_of(params.begin(), params.end(),
                           [&tensor](const SimpleTensorParam& param) { return param.tensor == tensor; });
    };
    EXPECT_TRUE(contains(analyzer->allOutcasts[1], outcastTensor)) << "Self producer component keeps the outcast";
    EXPECT_TRUE(contains(analyzer->allOutcasts[0], outcastTensor)) << "Other producer component gets the outcast";
    EXPECT_FALSE(sourceInvokeInfo.tensorParamList_.empty()) << "Self producer invokeInfo record should be kept";
}

// ReattributeInplaceOutcasts：回溯遇环（RESHAPE互相为生产者）报错返回FAILED
TEST_F(MixDependencyAnalyzerTest, TestReattributeInplaceOutcastsCycleDetected)
{
    const std::vector<int64_t> shape = {16, 16};
    auto tensorA = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensorB = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outcastTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& reshapeA = IRBuilder().CreateTensorOpStmt(*mixFunc, Opcode::OP_RESHAPE, {tensorB}, {tensorA});
    auto& reshapeB = IRBuilder().CreateTensorOpStmt(*mixFunc, Opcode::OP_RESHAPE, {tensorA}, {tensorB});
    auto& reshapeOp = IRBuilder().CreateTensorOpStmt(*mixFunc, Opcode::OP_RESHAPE, {tensorA}, {outcastTensor});

    std::vector<InternalComponentInfo> components = {{0, "comp_0"}, {1, "comp_1"}};
    components[1].operations = {&reshapeA, &reshapeB, &reshapeOp};

    analyzer->InitSubgraphToFunction(components);
    analyzer->allOutcasts[1].emplace_back(outcastTensor, reshapeOp.GetOpMagic(), 0);

    EXPECT_EQ(analyzer->ReattributeInplaceOutcasts(mixFunc.get(), components), FAILED)
        << "Cycle in producer chain should be reported as an error";
}

// OUTCAST类型引用不补回（当前仅处理INCAST）
TEST_F(MixDependencyAnalyzerTest, TestCollectGetTensorDataIncastsSkipsOutcast)
{
    const std::vector<int64_t> shape = {16, 16};
    auto incast = CreateAndRegisterIncast(shape);
    auto internal = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto output = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    CreateOpWithGetTensorDataRef(internal, output, GET_TENSOR_DATA_OPERAND_IOTYPE_OUTCAST, 0);

    std::vector<InternalComponentInfo> components = {{0, "comp_0"}};

    analyzer->CollectGetTensorDataIncasts(components, mixFunc.get());

    EXPECT_TRUE(analyzer->allIncasts.empty()) << "OUTCAST references should not be collected as incasts";
}
} // namespace tile_fwk
} // namespace npu
