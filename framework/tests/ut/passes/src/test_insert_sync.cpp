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
 * \file test_insert_sync.cpp
 * \brief Unit test for InsertSync.
 */
#include <gtest/gtest.h>
#include <functional>
#include <tuple>
#include "symbolic_scalar_test_utils.h"
#include "computational_graph_builder.h"
#include "pass_test_utils.h"
#include "tilefwk/platform.h"
#include "interface/operation/attribute.h"
#include "interface/tensor/irbuilder.h"
#define private public
#include "passes/block_graph_pass/insert_sync.h"
#include "passes/block_graph_pass/tune_sync_for_vf.h"
#include "passes/pass_utils/pass_common_defs.h"
#include "ut_json/ut_json_tool.h"

namespace npu {
namespace tile_fwk {
constexpr int VF_SCOPE_A = VF_CLUSTER_ID_START;
constexpr int VF_SCOPE_B = VF_CLUSTER_ID_START + 1;
constexpr int IS_NUM1 = 1;
constexpr int IS_NUM2 = 2;
constexpr int IS_NUM3 = 3;
constexpr int IS_NUM4 = 4;
constexpr int IS_NUM5 = 5;
constexpr int IS_NUM6 = 6;
constexpr int IS_NUM7 = 7;
constexpr int IS_NUM8 = 8;
constexpr int IS_NUM9 = 9;
constexpr int IS_NUM10 = 10;
constexpr int IS_NUM11 = 11;
constexpr int IS_NUM12 = 12;
constexpr int IS_NUM16 = 16;
constexpr int IS_NUM18 = 18;
constexpr int IS_NUM19 = 19;
constexpr int IS_NUM20 = 20;
constexpr int IS_NUM29 = 29;
constexpr int IS_NUM30 = 30;
constexpr int IS_NUM32 = 32;
constexpr int IS_NUM39 = 39;
constexpr int IS_NUM40 = 40;
constexpr int IS_NUM49 = 49;
constexpr int IS_NUM50 = 50;
constexpr int IS_NUM59 = 59;
constexpr int IS_NUM60 = 60;
constexpr int IS_NUM69 = 69;
constexpr int IS_NUM70 = 70;
constexpr int IS_NUM79 = 79;
constexpr int IS_NUM80 = 80;
constexpr int IS_NUM89 = 89;
constexpr int IS_NUM90 = 90;
constexpr int IS_NUM99 = 99;
constexpr int IS_NUM100 = 100;
constexpr int IS_NUM101 = 101;
constexpr int IS_NUM109 = 109;
constexpr int IS_NUM110 = 110;
constexpr int IS_NUM119 = 119;
constexpr int IS_NUM120 = 120;
constexpr int IS_NUM129 = 129;
constexpr int IS_NUM130 = 130;
constexpr int IS_NUM139 = 139;
constexpr int IS_NUM140 = 140;
constexpr int IS_NUM149 = 149;
constexpr int IS_NUM150 = 150;
constexpr int IS_NUM159 = 159;
constexpr int IS_NUM160 = 160;
constexpr int IS_NUM169 = 169;
constexpr int IS_NUM170 = 170;
constexpr int IS_NUM179 = 179;
constexpr int IS_NUM180 = 180;
constexpr int IS_NUM189 = 189;
constexpr int IS_NUM190 = 190;
constexpr int IS_NUM199 = 199;
constexpr int IS_NUM200 = 200;
constexpr int IS_NUM209 = 209;
constexpr int IS_NUM210 = 210;
constexpr int IS_NUM219 = 219;
constexpr int IS_NUM220 = 220;
constexpr int IS_NUM229 = 229;
constexpr int IS_NUM230 = 230;
constexpr int IS_NUM239 = 239;
constexpr int IS_NUM240 = 240;
constexpr int IS_NUM249 = 249;
constexpr int IS_NUM250 = 250;
constexpr int IS_NUM259 = 259;
constexpr int IS_NUM260 = 260;
constexpr int IS_NUM269 = 269;
constexpr int IS_NUM270 = 270;
constexpr int IS_NUM279 = 279;
constexpr int IS_NUM280 = 280;
constexpr int IS_NUM289 = 289;
constexpr int IS_NUM290 = 290;
constexpr int IS_NUM299 = 299;
constexpr int IS_NUM300 = 300;
constexpr int IS_NUM499 = 499;
constexpr int IS_NUM500 = 500;
constexpr int IS_NUM600 = 600;
constexpr int IS_NUM699 = 699;
constexpr int IS_NUM700 = 700;
constexpr int IS_NUM800 = 800;
constexpr int IS_NUM900 = 900;
constexpr int IS_NUM1000 = 1000;
constexpr int IS_NUM1100 = 1100;
class InsertSyncTest : public ::testing::Test {
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

    void AdjustCopyOpTileCfg(Operation& op, TileOpCfg& opcfg)
    {
        opcfg.aivCore_ = AIVCore::AIV0;
        if (op.GetOpcode() == Opcode::OP_COPY_IN) {
            opcfg.pipeIdStart_ = PipeType::PIPE_MTE2;
            opcfg.pipeIdEnd_ = PipeType::PIPE_MTE2;
            opcfg.coreType_ = CoreType::AIV;
        } else if (op.GetOpcode() == Opcode::OP_COPY_OUT) {
            opcfg.pipeIdStart_ = PipeType::PIPE_MTE3;
            opcfg.pipeIdEnd_ = PipeType::PIPE_MTE3;
            opcfg.coreType_ = CoreType::AIV;
        }
    }

    void BuildDeps(PipeSync& ps, DataDependencySearcher& dataDependencySearcher, std::vector<Operation*>& opLogPtr,
                   std::vector<IndexOp>& synced)
    {
        for (size_t i = 0; i < opLogPtr.size(); i++) {
            auto opcfg = OpcodeManager::Inst().GetTileOpCfg(opLogPtr[i]->GetOpcode());
            AdjustCopyOpTileCfg(*opLogPtr[i], opcfg);
            PipeSync::DepOp op(i, {opcfg.pipeIdStart_, opcfg.pipeIdEnd_, opcfg.coreType_, opcfg.aivCore_});
            PipeSync::DepOp& currOp = ps.depOps_.emplace_back(op);
            auto dataDependencySet = dataDependencySearcher.Find(opLogPtr[i]);
            for (auto it = dataDependencySet.rbegin(); it != dataDependencySet.rend(); it++) {
                size_t k = *it;
                PipeSync::DepOp& prevOp = ps.depOps_[k];
                if (ps.HasDataDependency(*opLogPtr[k], *opLogPtr[i])) {
                    ps.UpdateDep(currOp, prevOp);
                }
            }
            dataDependencySearcher.Insert(opLogPtr[i], i);
            ps.EnqueueOp(currOp, opLogPtr, synced);
        }
    }
};

TEST_F(InsertSyncTest, TestEnableDebug)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestParams", "TestParams", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestAddParams", "TestAddParams",
                                                      rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());

    // Prepare the graph
    std::vector<int64_t> shape = {IS_NUM8, IS_NUM16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto incast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto ubTensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));

    auto& copy_op1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    (void)copy_op1;
    auto& copy_op2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {incast2}, {ubTensor2});
    (void)copy_op2;
    auto& add_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_ADD, {ubTensor1, ubTensor2},
                                                  {ubTensor3});
    (void)add_op;
    auto& copy_out_op = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_OUT, {ubTensor3}, {outCast});
    (void)copy_out_op;
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outCast);
    InsertSync syncPass;
    syncPass.SetEnableDebug(true);
    syncPass.RunOnFunction(*rootFuncPtr);
    EXPECT_TRUE(true);
}

std::vector<std::shared_ptr<LogicalTensor>> AddOpForTestFindDep(std::vector<Operation*>& opLogPtr,
                                                                std::shared_ptr<Function> currFunctionPtr)
{
    // Build graph
    std::vector<int64_t> shape1 = {IS_NUM16, IS_NUM16};
    std::vector<int64_t> shape2 = {IS_NUM8, IS_NUM16};
    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    tensor1->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor1->memoryrange.start = 0;
    tensor1->memoryrange.end = IS_NUM100;
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    tensor2->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor2->memoryrange.start = 0;
    tensor2->memoryrange.end = IS_NUM200;
    auto tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    tensor3->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor3->memoryrange.start = IS_NUM300;
    tensor3->memoryrange.end = IS_NUM499;
    auto tensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    tensor4->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor4->memoryrange.start = IS_NUM500;
    tensor4->memoryrange.end = IS_NUM699;
    auto tensor5 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    tensor5->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor5->memoryrange.start = IS_NUM700;
    tensor5->memoryrange.end = IS_NUM900;
    auto tensor6 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    tensor6->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor6->memoryrange.start = IS_NUM150;
    tensor6->memoryrange.end = IS_NUM200;
    auto& expend = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_EXPAND, {tensor1}, {tensor2});
    opLogPtr.emplace_back(&expend);
    auto& copyin1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor2}, {tensor3});
    opLogPtr.emplace_back(&copyin1);
    auto& copyin2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor2}, {tensor4});
    opLogPtr.emplace_back(&copyin2);
    auto& copyin3 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor2}, {tensor5});
    opLogPtr.emplace_back(&copyin3);
    auto& exp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_EXP, {tensor3}, {tensor6});
    opLogPtr.emplace_back(&exp);
    return {tensor1, tensor2, tensor3, tensor4, tensor5, tensor6};
}

void CheckDependencyForTestFindDep(PipeSync& ps, std::set<int> dataDependencySet, std::vector<Operation*>& opLogPtr,
                                   size_t i)
{
    for (auto it = dataDependencySet.rbegin(); it != dataDependencySet.rend(); it++) {
        size_t k = *it;
        // start tests
        if (i == IS_NUM1 && k == 0) {
            EXPECT_EQ(ps.CheckRawDependency(*opLogPtr[k], *opLogPtr[i]), true);
            EXPECT_EQ(ps.CheckWarDependency(*opLogPtr[k], *opLogPtr[i]), false);
            EXPECT_EQ(ps.CheckWawDependency(*opLogPtr[k], *opLogPtr[i]), false);
        }
        if (i == IS_NUM4 && k == 0) {
            EXPECT_EQ(ps.CheckRawDependency(*opLogPtr[k], *opLogPtr[i]), false);
            EXPECT_EQ(ps.CheckWarDependency(*opLogPtr[k], *opLogPtr[i]), false);
            EXPECT_EQ(ps.CheckWawDependency(*opLogPtr[k], *opLogPtr[i]), true);
        }
        if (i == IS_NUM4 && k == IS_NUM1) {
            EXPECT_EQ(ps.CheckRawDependency(*opLogPtr[k], *opLogPtr[i]), true);
            EXPECT_EQ(ps.CheckWarDependency(*opLogPtr[k], *opLogPtr[i]), true);
            EXPECT_EQ(ps.CheckWawDependency(*opLogPtr[k], *opLogPtr[i]), false);
        }
        // end tests
    }
}

void ProcessOpList(PipeSync& ps, DataDependencySearcher& dataDependencySearcher, std::vector<Operation*>& opLogPtr)
{
    ps.oriOpList_ = opLogPtr;
    for (auto& op : opLogPtr) {
        bool isCubeComponent = op->HasAttr(OpAttributeKey::isCube) && op->GetAttr<bool>(OpAttributeKey::isCube);
        if (!isCubeComponent) {
            op->SetAIVCore(AIVCore::AIV0);
        }
        ps.BuildTensorRangeMap(op);
    }
    dataDependencySearcher.ubTensorRangeMap = ps.ubTensorRangeMap;
}

TEST_F(InsertSyncTest, TestFindDep)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestFindDep", "TestFindDep", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestFindDepLeaf", "TestFindDepLeaf",
                                                      rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());

    std::vector<Operation*> opLogPtr;
    auto tensors = AddOpForTestFindDep(opLogPtr, currFunctionPtr);
    PipeSync ps;
    DataDependencySearcher dataDependencySearcher;
    ProcessOpList(ps, dataDependencySearcher, opLogPtr);
    for (size_t i = 0; i < opLogPtr.size(); i++) {
        auto opcfg = OpcodeManager::Inst().GetTileOpCfg(opLogPtr[i]->GetOpcode());
        AdjustCopyOpTileCfg(*opLogPtr[i], opcfg);
        PipeSync::DepOp op(i, {opcfg.pipeIdStart_, opcfg.pipeIdEnd_, opcfg.coreType_, opcfg.aivCore_});
        ps.depOps_.emplace_back(op);
        auto dataDependencySet = dataDependencySearcher.Find(opLogPtr[i]);
        // start tests
        if (i == IS_NUM1 || i == IS_NUM2 || i == IS_NUM3) {
            std::set<int> res = {0};
            EXPECT_EQ(dataDependencySet, res);
        }
        if (i == IS_NUM4) {
            std::set<int> res = {0, IS_NUM1, IS_NUM2, IS_NUM3};
            EXPECT_EQ(dataDependencySet, res);
        }
        // end tests
        CheckDependencyForTestFindDep(ps, dataDependencySet, opLogPtr, i);
        dataDependencySearcher.Insert(opLogPtr[i], i);
    }

    // test ignorable intra pipe dep
    EXPECT_EQ(ps.IgnorableIntraPipeDep(0, IS_NUM4, opLogPtr), false);
    tensors[1]->shape = {IS_NUM16, IS_NUM32, IS_NUM32};
    EXPECT_EQ(ps.IgnorableIntraPipeDep(0, IS_NUM4, opLogPtr), false);

    // test AdjustOpCfg
    auto opcfg1 = OpcodeManager::Inst().GetTileOpCfg(opLogPtr[1]->GetOpcode());
    EXPECT_EQ(ps.AdjustOpCfg(opcfg1, *opLogPtr[1]), FAILED);

    std::vector<int64_t> shape = {IS_NUM8, IS_NUM16};
    auto tensor7 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor7->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor7->memoryrange.start = IS_NUM1000;
    tensor7->memoryrange.end = IS_NUM1100;
    auto& copyout = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_OUT, {tensors[4]}, {tensor7});
    opLogPtr.emplace_back(&copyout);
    auto opcfg2 = OpcodeManager::Inst().GetTileOpCfg(opLogPtr[IS_NUM1]->GetOpcode());
    EXPECT_EQ(ps.AdjustOpCfg(opcfg2, *opLogPtr[IS_NUM5]), FAILED);
}

TEST_F(InsertSyncTest, TestTokenOnlyDependencyGeneratesSync)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "TestTokenOnlyDependencyGeneratesSync",
                                               "TestTokenOnlyDependencyGeneratesSync", nullptr);
    std::vector<int64_t> shape = {IS_NUM8, IS_NUM16};
    auto shapeImmediate = OpImmediate::Specified(shape);
    IRBuilder builder;
    auto gmInput = builder.CreateTensorVar(*function, DT_FP32, shape);
    auto producerOutput = builder.CreateTensorVar(*function, DT_FP32, shape);
    auto consumerInput = builder.CreateTensorVar(*function, DT_FP32, shape);
    auto consumerOutput = builder.CreateTensorVar(*function, DT_FP32, shape);
    gmInput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    producerOutput->SetMemoryTypeBoth(MemoryType::MEM_UB);
    consumerInput->SetMemoryTypeBoth(MemoryType::MEM_UB);
    consumerOutput->SetMemoryTypeBoth(MemoryType::MEM_UB);
    gmInput->memoryrange = TileRange(0, IS_NUM100, IS_NUM1);
    producerOutput->memoryrange = TileRange(0, IS_NUM100, IS_NUM2);
    consumerInput->memoryrange = TileRange(IS_NUM200, IS_NUM300, IS_NUM3);
    consumerOutput->memoryrange = TileRange(IS_NUM300, IS_NUM499, IS_NUM4);

    auto& producer = builder.CreateTensorOpStmt(*function, Opcode::OP_COPY_IN, {gmInput}, {producerOutput});
    producer.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MemoryType::MEM_UB,
                                                              shapeImmediate, shapeImmediate));
    producer.SetAIVCore(AIVCore::AIV0);
    auto& consumer = builder.CreateTensorOpStmt(*function, Opcode::OP_CAST, {consumerInput}, {consumerOutput});
    consumer.SetAIVCore(AIVCore::AIV0);

    PipeSync ps;
    EXPECT_FALSE(ps.HasDataDependency(producer, consumer));
    auto token = builder.CreateTokenVar(ir::Span::Unknown());
    producer.result_token_ = {token};
    consumer.tokens_.push_back(token);
    auto& dependency = function->GetVarDependency();
    dependency.AddProducer(token, std::static_pointer_cast<const ir::Stmt>(function->operations_[0]));
    dependency.AddConsumer(token, std::static_pointer_cast<const ir::Stmt>(function->operations_[IS_NUM1]));
    EXPECT_TRUE(ps.HasDataDependency(producer, consumer));

    std::vector<Operation*> syncedOps;
    ASSERT_EQ(ps.InsertSync(*function, syncedOps), SUCCESS);
    bool hasSyncSrc = false;
    bool hasSyncDst = false;
    for (auto* op : syncedOps) {
        hasSyncSrc = hasSyncSrc || op->GetOpcode() == Opcode::OP_SYNC_SRC;
        hasSyncDst = hasSyncDst || op->GetOpcode() == Opcode::OP_SYNC_DST;
    }
    EXPECT_TRUE(hasSyncSrc);
    EXPECT_TRUE(hasSyncDst);
}

TEST_F(InsertSyncTest, TestTokenOnlyCrossCoreDependencyGeneratesCvSync)
{
    auto function = std::make_shared<Function>(Program::GetInstance(),
                                               "TestTokenOnlyCrossCoreDependencyGeneratesCvSync",
                                               "TestTokenOnlyCrossCoreDependencyGeneratesCvSync", nullptr);
    std::vector<int64_t> shape = {IS_NUM8, IS_NUM16};
    IRBuilder builder;
    auto l1Input = builder.CreateTensorVar(*function, DT_FP32, shape);
    auto l0aOutput = builder.CreateTensorVar(*function, DT_FP32, shape);
    auto ubInput = builder.CreateTensorVar(*function, DT_FP32, shape);
    auto ubOutput = builder.CreateTensorVar(*function, DT_FP32, shape);
    l1Input->SetMemoryTypeBoth(MemoryType::MEM_L1);
    l0aOutput->SetMemoryTypeBoth(MemoryType::MEM_L0A);
    ubInput->SetMemoryTypeBoth(MemoryType::MEM_UB);
    ubOutput->SetMemoryTypeBoth(MemoryType::MEM_UB);
    l1Input->memoryrange = TileRange(0, IS_NUM100, IS_NUM1);
    l0aOutput->memoryrange = TileRange(0, IS_NUM100, IS_NUM2);
    ubInput->memoryrange = TileRange(IS_NUM200, IS_NUM300, IS_NUM3);
    ubOutput->memoryrange = TileRange(IS_NUM300, IS_NUM499, IS_NUM4);

    auto& producer = builder.CreateTensorOpStmt(*function, Opcode::OP_L1_TO_L0A, {l1Input}, {l0aOutput});
    auto& consumer = builder.CreateTensorOpStmt(*function, Opcode::OP_CAST, {ubInput}, {ubOutput});
    consumer.SetAIVCore(AIVCore::AIV0);
    auto token = builder.CreateTokenVar(ir::Span::Unknown());
    producer.result_token_ = {token};
    consumer.tokens_.push_back(token);
    auto& dependency = function->GetVarDependency();
    dependency.AddProducer(token, std::static_pointer_cast<const ir::Stmt>(function->operations_[0]));
    dependency.AddConsumer(token, std::static_pointer_cast<const ir::Stmt>(function->operations_[IS_NUM1]));

    PipeSync ps;
    EXPECT_TRUE(ps.HasDataDependency(producer, consumer));
    std::vector<Operation*> syncedOps;
    ASSERT_EQ(ps.InsertSync(*function, syncedOps), SUCCESS);
    bool hasCvSyncSrc = false;
    bool hasCvSyncDst = false;
    for (auto* op : syncedOps) {
        hasCvSyncSrc = hasCvSyncSrc || op->GetOpcode() == Opcode::OP_CV_SYNC_SRC;
        hasCvSyncDst = hasCvSyncDst || op->GetOpcode() == Opcode::OP_CV_SYNC_DST;
    }
    EXPECT_TRUE(hasCvSyncSrc);
    EXPECT_TRUE(hasCvSyncDst);
}

TEST_F(InsertSyncTest, TestPhaseKernelProcess)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestPhaseKernelProcess",
                                                  "TestPhaseKernelProcess", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPhaseKernelProcessLeaf",
                                                      "TestPhaseKernelProcessLeaf", rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());

    std::vector<int64_t> shape = {IS_NUM16, IS_NUM16};
    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor5 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    auto tensor6 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    std::vector<Operation*> opLogPtr;
    auto& copyin1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor1}, {tensor3});
    opLogPtr.emplace_back(&copyin1);
    auto& copyin2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor2}, {tensor4});
    opLogPtr.emplace_back(&copyin2);
    auto& add = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_ADD, {tensor3, tensor4}, {tensor5});
    opLogPtr.emplace_back(&add);
    auto& copyout = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_OUT, {tensor5}, {tensor6});
    opLogPtr.emplace_back(&copyout);

    PipeSync ps;
    std::vector<Operation*> resLogPtr;
    ps.PhaseKernelProcess(*currFunctionPtr, opLogPtr, resLogPtr);
    EXPECT_EQ(resLogPtr[0]->GetOpcode(), Opcode::OP_PHASE1);
    EXPECT_EQ(resLogPtr[IS_NUM3]->GetOpcode(), Opcode::OP_PHASE2);
}

void AddOpForTestUpdateDep(std::vector<Operation*>& opLogPtr, std::shared_ptr<Function> currFunctionPtr)
{
    std::vector<int64_t> shape = {IS_NUM16, IS_NUM16};
    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor1->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor1->memoryrange.start = 0;
    tensor1->memoryrange.end = IS_NUM100;
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor2->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor2->memoryrange.start = IS_NUM200;
    tensor2->memoryrange.end = IS_NUM300;
    auto tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor3->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor3->memoryrange.start = IS_NUM200;
    tensor3->memoryrange.end = IS_NUM300;
    auto tensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor4->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor4->memoryrange.start = IS_NUM200;
    tensor4->memoryrange.end = IS_NUM300;
    auto tensor5 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor5->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor5->memoryrange.start = IS_NUM500;
    tensor5->memoryrange.end = IS_NUM600;
    auto tensor6 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor6->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor6->memoryrange.start = IS_NUM500;
    tensor6->memoryrange.end = IS_NUM600;
    auto tensor7 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor7->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor7->memoryrange.start = 0;
    tensor7->memoryrange.end = IS_NUM100;
    auto tensor8 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor8->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor8->memoryrange.start = IS_NUM101;
    tensor8->memoryrange.end = IS_NUM199;
    auto& copyin1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor1}, {tensor8});
    opLogPtr.emplace_back(&copyin1);
    auto& copyin2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor2}, {tensor3});
    opLogPtr.emplace_back(&copyin2);
    auto& cast1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor3}, {tensor4});
    opLogPtr.emplace_back(&cast1);
    auto& cast2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor5}, {tensor6});
    opLogPtr.emplace_back(&cast2);
    auto& copyout = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_OUT, {tensor6}, {tensor7});
    opLogPtr.emplace_back(&copyout);
}

TEST_F(InsertSyncTest, TestUpdateDep)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestUpdateDep", "TestUpdateDep", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestUpdateDepLeaf", "TestUpdateDepLeaf",
                                                      rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());

    std::vector<Operation*> opLogPtr;
    AddOpForTestUpdateDep(opLogPtr, currFunctionPtr);

    PipeSync ps;
    DataDependencySearcher dataDependencySearcher;
    ProcessOpList(ps, dataDependencySearcher, opLogPtr);
    for (size_t i = 0; i < opLogPtr.size(); i++) {
        auto opcfg = OpcodeManager::Inst().GetTileOpCfg(opLogPtr[i]->GetOpcode());
        AdjustCopyOpTileCfg(*opLogPtr[i], opcfg);
        PipeSync::DepOp op(i, {opcfg.pipeIdStart_, opcfg.pipeIdEnd_, opcfg.coreType_, opcfg.aivCore_});
        auto& currOp = ps.depOps_.emplace_back(op);
        auto dataDependencySet = dataDependencySearcher.Find(opLogPtr[i]);
        for (auto it = dataDependencySet.rbegin(); it != dataDependencySet.rend(); it++) {
            size_t k = *it;
            auto& prevOp = ps.depOps_[k];
            if (ps.HasDataDependency(*opLogPtr[k], *opLogPtr[i])) {
                // start tests
                ps.UpdateDep(currOp, prevOp);
                PipeSync::PipeCoreRealEx pcCurr(PipeType::PIPE_MTE3, CoreType::AIV, AIVCore::AIV0);
                PipeSync::PipeCoreRealEx pcSet1(PipeType::PIPE_V, CoreType::AIV, AIVCore::AIV0);
                PipeSync::PipeCoreRealEx pcSet2(PipeType::PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
                auto setPipeIdx1 = ps.latestPipeDep_[pcCurr].setPipes[pcSet1];
                auto setPipeIdx2 = ps.latestPipeDep_[pcCurr].setPipes[pcSet2];
                ps.latestPipeDep_[pcCurr].DumpPipeDepInfo();
                if (i == IS_NUM4 && k == IS_NUM3) {
                    EXPECT_EQ(setPipeIdx1, static_cast<size_t>(IS_NUM3));
                    EXPECT_EQ(setPipeIdx2, static_cast<size_t>(IS_NUM1));
                }
                if (i == IS_NUM4 && k == 0) {
                    EXPECT_EQ(setPipeIdx1, static_cast<size_t>(IS_NUM3));
                    EXPECT_EQ(setPipeIdx2, static_cast<size_t>(IS_NUM1));
                }
            }
        }
        dataDependencySearcher.Insert(opLogPtr[i], i);
    }
}

void AddOpForTestHandleEventID(std::vector<Operation*>& opLogPtr, std::shared_ptr<Function> currFunctionPtr)
{
    std::vector<int64_t> shape = {IS_NUM16, IS_NUM16};
    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor1->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor1->memoryrange.start = 0;
    tensor1->memoryrange.end = IS_NUM99;
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor2->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor2->memoryrange.start = IS_NUM100;
    tensor2->memoryrange.end = IS_NUM199;
    auto tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor3->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor3->memoryrange.start = IS_NUM200;
    tensor3->memoryrange.end = IS_NUM300;
    auto tensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor4->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor4->memoryrange.start = IS_NUM500;
    tensor4->memoryrange.end = IS_NUM600;
    auto tensor5 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor5->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor5->memoryrange.start = IS_NUM700;
    tensor5->memoryrange.end = IS_NUM800;
    auto& copyin1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor1}, {tensor2});
    opLogPtr.emplace_back(&copyin1); // index 0
    auto& copyin2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor4}, {tensor5});
    opLogPtr.emplace_back(&copyin2); // index 1 - moved before cast to satisfy AdjustOpDep conditions
    auto& cast = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor2}, {tensor3});
    opLogPtr.emplace_back(&cast); // index 2
}

TEST_F(InsertSyncTest, TestHandleEventID)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestHandleEventID", "TestHandleEventID",
                                                  nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestHandleEventIDLeaf",
                                                      "TestHandleEventIDLeaf", rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());

    std::vector<Operation*> opLogPtr;
    AddOpForTestHandleEventID(opLogPtr, currFunctionPtr);

    PipeSync ps;
    ps.InitIssueQueue(); // Initialize issueState_ before BuildDeps
    DataDependencySearcher dataDependencySearcher;
    ProcessOpList(ps, dataDependencySearcher, opLogPtr);
    std::vector<IndexOp> synced;
    BuildDeps(ps, dataDependencySearcher, opLogPtr, synced);
    // After reorder: copyin1(0), copyin2(1), cast(2)
    EXPECT_EQ(ps.depOps_[0].setPipe[0], static_cast<size_t>(IS_NUM2));  // copyin1 sets for cast
    EXPECT_EQ(ps.depOps_[IS_NUM2].waitPipe[0], static_cast<size_t>(0)); // cast waits for copyin1

    // HandleEventID - Test AdjustOpDep path
    // Set up conditions to trigger AdjustOpDep:
    // 1. currIssueNum >= maxIssueNum (both set to 8)
    // 2. deadlock = true
    // 3. issueQ has enough elements for nextOp
    bool eventIdDeadlock = true;
    bool res = false;
    PipeSync::IssueNum issuenum;
    // issueState_[4] is AIV0_MTE2 queue, contains copyin1(0) and copyin2(1)
    PipeSync::IssueQueue& issueQ = ps.issueState_[IS_NUM4];
    PipeSync::DepOp& handleOp = ps.depOps_[0];    // copyin1
    PipeSync::DepOp& eleOp = ps.depOps_[IS_NUM2]; // cast
    PipeSync::PipeCoreRealEx currPipeCoreEx(handleOp.selfPipeCore.pipeEnd, handleOp.selfPipeCore.core,
                                            handleOp.selfPipeCore.aivCore);
    PipeSync::PipeCoreRealEx elePipeCoreEx(eleOp.selfPipeCore.pipeStart, eleOp.selfPipeCore.core,
                                           eleOp.selfPipeCore.aivCore);
    PipeSync::PipePairEx pp{currPipeCoreEx, elePipeCoreEx};

    // Pre-set issuenum to trigger AdjustOpDep: currIssueNum >= maxIssueNum
    issuenum.maxIssueNum[pp] = IS_NUM8;
    issuenum.currIssueNum[pp] = IS_NUM8;

    // Verify initial state before HandleEventID
    EXPECT_EQ(ps.depOps_[IS_NUM2].waitPipe[0], static_cast<size_t>(0)); // cast waits for copyin1
    EXPECT_EQ(ps.depOps_[0].setPipe[0], static_cast<size_t>(IS_NUM2));  // copyin1 sets for cast

    ps.HandleEventID(handleOp, issueQ, issuenum, eventIdDeadlock, res, synced);
    issueQ.DumpIssueQueue(ps.oriOpList_);
    ps.DumpLatestPipeDepMap();

    // After HandleEventID with AdjustOpDep:
    // RemoveOpDep: copyin1's setPipe removes cast, cast's waitPipe removes copyin1
    // AddOpDep: copyin2's setPipe adds cast, cast's waitPipe adds copyin2
    EXPECT_EQ(ps.depOps_[IS_NUM2].waitPipe[0], static_cast<size_t>(IS_NUM1)); // cast now waits for copyin2
    EXPECT_EQ(ps.depOps_[IS_NUM1].setPipe[0], static_cast<size_t>(IS_NUM2));  // copyin2 sets for cast
    EXPECT_EQ(ps.depOps_[0].setPipe.size(), static_cast<size_t>(0));          // copyin1 has no setPipe

    // InitCVEventIdQ
    ps.InitCVEventIdQ();
    PipeSync::PipeCoreRealEx pcAIC(PIPE_S, CoreType::AIC, AIVCore::UNSPECIFIED);
    PipeSync::PipeCoreRealEx pcAIV1(PIPE_V, CoreType::AIV, AIVCore::AIV1);
    EXPECT_EQ(ps.crossCoreFreeEventId_[pcAIC][0].size(), static_cast<size_t>(IS_NUM16));
    EXPECT_EQ(ps.crossCoreFreeEventId_[pcAIV1][0].size(), static_cast<size_t>(IS_NUM16));
}

TEST_F(InsertSyncTest, TestRelaxFakeDataDep)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestRelaxFakeDataDep",
                                                  "TestRelaxFakeDataDep", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestRelaxFakeDataDepLeaf",
                                                      "TestRelaxFakeDataDepLeaf", rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());

    std::vector<int64_t> shape = {IS_NUM8, IS_NUM8};
    auto tensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor1->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor1->memoryrange.start = IS_NUM100;
    tensor1->memoryrange.end = IS_NUM109;
    auto tensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor2->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor2->memoryrange.start = 0;
    tensor2->memoryrange.end = IS_NUM9;
    auto tensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor3->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor3->memoryrange.start = IS_NUM10;
    tensor3->memoryrange.end = IS_NUM19;
    auto tensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor4->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor4->memoryrange.start = IS_NUM20;
    tensor4->memoryrange.end = IS_NUM29;
    auto tensor5 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor5->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor5->memoryrange.start = IS_NUM30;
    tensor5->memoryrange.end = IS_NUM39;
    auto tensor6 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor6->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor6->memoryrange.start = IS_NUM40;
    tensor6->memoryrange.end = IS_NUM49;
    auto tensor7 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor7->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor7->memoryrange.start = IS_NUM50;
    tensor7->memoryrange.end = IS_NUM59;
    auto tensor8 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor8->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor8->memoryrange.start = IS_NUM60;
    tensor8->memoryrange.end = IS_NUM69;
    auto tensor9 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor9->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor9->memoryrange.start = IS_NUM70;
    tensor9->memoryrange.end = IS_NUM79;
    auto tensor10 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor10->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor10->memoryrange.start = IS_NUM80;
    tensor10->memoryrange.end = IS_NUM89;
    auto tensor11 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor11->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor11->memoryrange.start = IS_NUM90;
    tensor11->memoryrange.end = IS_NUM99;
    auto tensor12 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor12->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor12->memoryrange.start = IS_NUM200;
    tensor12->memoryrange.end = IS_NUM209;
    auto tensor13 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor13->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor13->memoryrange.start = IS_NUM210;
    tensor13->memoryrange.end = IS_NUM219;
    auto tensor14 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor14->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor14->memoryrange.start = IS_NUM220;
    tensor14->memoryrange.end = IS_NUM229;
    auto tensor15 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor15->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor15->memoryrange.start = IS_NUM230;
    tensor15->memoryrange.end = IS_NUM239;
    auto tensor16 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor16->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor16->memoryrange.start = IS_NUM240;
    tensor16->memoryrange.end = IS_NUM249;
    auto tensor17 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor17->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor17->memoryrange.start = IS_NUM250;
    tensor17->memoryrange.end = IS_NUM259;
    auto tensor18 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor18->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor18->memoryrange.start = IS_NUM260;
    tensor18->memoryrange.end = IS_NUM269;
    auto tensor19 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor19->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor19->memoryrange.start = IS_NUM270;
    tensor19->memoryrange.end = IS_NUM279;
    auto tensor20 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor20->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor20->memoryrange.start = IS_NUM280;
    tensor20->memoryrange.end = IS_NUM289;
    auto tensor21 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor21->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor21->memoryrange.start = IS_NUM290;
    tensor21->memoryrange.end = IS_NUM299;
    auto tensor22 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor22->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor22->memoryrange.start = IS_NUM110;
    tensor22->memoryrange.end = IS_NUM119;
    auto tensor23 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor23->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor23->memoryrange.start = IS_NUM120;
    tensor23->memoryrange.end = IS_NUM129;
    auto tensor24 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor24->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor24->memoryrange.start = IS_NUM130;
    tensor24->memoryrange.end = IS_NUM139;
    auto tensor25 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor25->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor25->memoryrange.start = IS_NUM140;
    tensor25->memoryrange.end = IS_NUM149;
    auto tensor26 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor26->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor26->memoryrange.start = IS_NUM150;
    tensor26->memoryrange.end = IS_NUM159;
    auto tensor27 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor27->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor27->memoryrange.start = IS_NUM160;
    tensor27->memoryrange.end = IS_NUM169;
    auto tensor28 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor28->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor28->memoryrange.start = IS_NUM170;
    tensor28->memoryrange.end = IS_NUM179;
    auto tensor29 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor29->SetMemoryTypeBoth(MemoryType::MEM_UB);
    tensor29->memoryrange.start = IS_NUM180;
    tensor29->memoryrange.end = IS_NUM189;
    std::vector<Operation*> opLogPtr;
    auto& copyin1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor1}, {tensor2});
    opLogPtr.emplace_back(&copyin1);
    auto& copyin2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor1}, {tensor3});
    opLogPtr.emplace_back(&copyin2);
    auto& copyin3 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor1}, {tensor4});
    opLogPtr.emplace_back(&copyin3);
    auto& copyin4 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor1}, {tensor5});
    opLogPtr.emplace_back(&copyin4);
    auto& copyin5 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor1}, {tensor6});
    opLogPtr.emplace_back(&copyin5);
    auto& copyin6 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor1}, {tensor7});
    opLogPtr.emplace_back(&copyin6);
    auto& copyin7 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor1}, {tensor8});
    opLogPtr.emplace_back(&copyin7);
    auto& copyin8 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor1}, {tensor9});
    opLogPtr.emplace_back(&copyin8);
    auto& copyin9 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor1}, {tensor10});
    opLogPtr.emplace_back(&copyin9);
    auto& copyin10 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_COPY_IN, {tensor1}, {tensor11});
    opLogPtr.emplace_back(&copyin10);
    auto& cast1 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor1}, {tensor22});
    opLogPtr.emplace_back(&cast1);
    auto& cast2 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor1}, {tensor23});
    opLogPtr.emplace_back(&cast2);
    auto& cast3 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor1}, {tensor24});
    opLogPtr.emplace_back(&cast3);
    auto& cast4 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor1}, {tensor25});
    opLogPtr.emplace_back(&cast4);
    auto& cast5 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor1}, {tensor26});
    opLogPtr.emplace_back(&cast5);
    auto& cast6 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor1}, {tensor27});
    opLogPtr.emplace_back(&cast6);
    auto& cast7 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor1}, {tensor28});
    opLogPtr.emplace_back(&cast7);
    auto& cast8 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor1}, {tensor29});
    opLogPtr.emplace_back(&cast8);
    auto& cast9 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor2}, {tensor12});
    opLogPtr.emplace_back(&cast9);
    auto& cast10 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor3}, {tensor13});
    opLogPtr.emplace_back(&cast10);
    auto& cast11 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor4}, {tensor14});
    opLogPtr.emplace_back(&cast11);
    auto& cast12 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor5}, {tensor15});
    opLogPtr.emplace_back(&cast12);
    auto& cast13 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor6}, {tensor16});
    opLogPtr.emplace_back(&cast13);
    auto& cast14 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor7}, {tensor17});
    opLogPtr.emplace_back(&cast14);
    auto& cast15 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor8}, {tensor18});
    opLogPtr.emplace_back(&cast15);
    auto& cast16 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor9}, {tensor19});
    opLogPtr.emplace_back(&cast16);
    auto& cast17 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor10}, {tensor20});
    opLogPtr.emplace_back(&cast17);
    auto& cast18 = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_CAST, {tensor11}, {tensor21});
    opLogPtr.emplace_back(&cast18);

    PipeSync ps;
    DataDependencySearcher dataDependencySearcher;
    ProcessOpList(ps, dataDependencySearcher, opLogPtr);
    std::vector<IndexOp> synced;
    size_t index = UINT64_MAX;
    EXPECT_EQ(ps.InjectSync(*currFunctionPtr, opLogPtr, index, synced), FAILED);
    BuildDeps(ps, dataDependencySearcher, opLogPtr, synced);

    // Issue op
    size_t totalIssued = 0;
    size_t allIssued = 0;
    for (int i = 0; i < static_cast<int>(PipeSeq::PIPE_END); i++) {
        allIssued += ps.issueState_[i].ops.size();
    }
    bool eventIdDeadlock = false;
    uint64_t eventIdDeadlockEnterTimes = 0;

    while (totalIssued < allIssued) {
        size_t issued = 0;
        size_t issuedTest = IS_NUM100;
        for (int i = 0; i < static_cast<int>(PipeSeq::PIPE_END); i++) {
            std::vector<size_t> issuedOps;
            ps.PopFromQueue(ps.issueState_[i], issuedOps, eventIdDeadlock, synced);
            issued += issuedOps.size();
            if (i == IS_NUM4) {
                issuedTest = issuedOps.size();
            }
            for (auto idx : issuedOps) {
                ps.InjectSync(*currFunctionPtr, opLogPtr, idx, synced);
            }
            if (issuedTest == static_cast<size_t>(0)) {
                break;
            }
        }
        totalIssued += issued;
        // eventIdDeadlockEnterTimes eventIdDeadlock syncedOpLog
        if (issuedTest == static_cast<size_t>(0)) {
            EXPECT_EQ(ps.depOps_[0].setPipe[0], static_cast<size_t>(IS_NUM18));
            EXPECT_EQ(ps.depOps_[IS_NUM1].setPipe[0], static_cast<size_t>(IS_NUM19));
            ps.ProcessDeadLock(eventIdDeadlockEnterTimes, eventIdDeadlock, synced);
            EXPECT_EQ(ps.depOps_[IS_NUM1].setPipe[0], static_cast<size_t>(IS_NUM18));
            continue;
        }
        eventIdDeadlock = false;
        eventIdDeadlockEnterTimes = static_cast<size_t>(0);
        break;
    }
}

TEST_F(InsertSyncTest, TestGetDepInfoWithNoPendingEvents)
{
    PipeSync ps;
    std::vector<IndexOp> emptySyncedOpLog;
    auto pipePair = PipeSync::dataDepPair[0];
    PipeSync::DataDepInfo depInfo;
    EXPECT_EQ(ps.GetDepInfo(emptySyncedOpLog, pipePair, depInfo), SUCCESS);
    EXPECT_TRUE(depInfo.opDepList.empty());
    int overlapIdx = -1;
    EXPECT_FALSE(ps.FindMaxOverlap(depInfo, overlapIdx));
    EXPECT_EQ(ps.PipeSeqName(PipeSeq::AIV0_MTE2), "AIV0_MTE2");
    EXPECT_EQ(ps.PipeSeqName(PipeSeq::AIV1_MTE2), "AIV1_MTE2");
    EXPECT_EQ(ps.PipeSeqName(PipeSeq::AIV0_V), "AIV0_V");
    EXPECT_EQ(ps.PipeSeqName(PipeSeq::AIV1_V), "AIV1_V");
    EXPECT_EQ(ps.PipeSeqName(PipeSeq::AIV0_MTE3), "AIV0_MTE3");
    EXPECT_EQ(ps.PipeSeqName(PipeSeq::AIV1_MTE3), "AIV1_MTE3");
    EXPECT_EQ(ps.PipeSeqName(PipeSeq::AIV0_S), "AIV0_S");
    EXPECT_EQ(ps.PipeSeqName(PipeSeq::AIV1_S), "AIV1_S");
}

TEST_F(InsertSyncTest, TestRecycleCrossCoreEventIds)
{
    PipeSync ps;
    ps.InitCVEventIdQ();

    // 场景1: currPipeRealEx 不在 crossCoreFreeEventId_ 中 -> 返回 FAILED
    PipeSync::PipeCoreRealEx notExistPipe(PIPE_V, CoreType::AIV, AIVCore::UNSPECIFIED);
    EXPECT_EQ(ps.RecycleCrossCoreEventIds(notExistPipe), FAILED);

    using CoreDetail = PipeSync::CoreTypeDetail;
    auto makeEr = [](int id, CoreDetail src, CoreDetail dst) {
        return PipeSync::EventResource{id, src, dst, PIPE_S, PIPE_V};
    };
    CoreDetail aicCore{CoreType::AIC, AIVCore::UNSPECIFIED};
    CoreDetail aiv0Core{CoreType::AIV, AIVCore::AIV0};
    CoreDetail aiv1Core{CoreType::AIV, AIVCore::AIV1};

    // 场景2: AIC core - 覆盖 srcCore==dstCore(continue) + eventId<16(queue0) + eventId>=16(queue1)
    PipeSync::PipeCoreRealEx pcAIC(PIPE_S, CoreType::AIC, AIVCore::UNSPECIFIED);
    ps.syncArriveStatus[pcAIC].clear();
    ps.crossCoreFreeEventId_[pcAIC][0].clear();
    ps.crossCoreFreeEventId_[pcAIC][1].clear();
    ps.syncArriveStatus[pcAIC].insert(makeEr(IS_NUM5, aicCore, aicCore));   // srcCore==dstCore -> continue
    ps.syncArriveStatus[pcAIC].insert(makeEr(IS_NUM3, aicCore, aiv0Core));  // eventId<16 -> queue0
    ps.syncArriveStatus[pcAIC].insert(makeEr(IS_NUM20, aicCore, aiv1Core)); // eventId>=16 -> queue1
    EXPECT_EQ(ps.RecycleCrossCoreEventIds(pcAIC), SUCCESS);
    EXPECT_EQ(ps.crossCoreFreeEventId_[pcAIC][0].size(), static_cast<size_t>(IS_NUM1));
    EXPECT_EQ(ps.crossCoreFreeEventId_[pcAIC][0].front(), IS_NUM3);
    EXPECT_EQ(ps.crossCoreFreeEventId_[pcAIC][1].size(), static_cast<size_t>(IS_NUM1));
    EXPECT_EQ(ps.crossCoreFreeEventId_[pcAIC][1].front(), IS_NUM20);

    // 场景3: AIV0 core - 覆盖 eventId<16(queue0) + eventId>=16(跳过)
    PipeSync::PipeCoreRealEx pcAIV0(PIPE_S, CoreType::AIV, AIVCore::AIV0);
    ps.syncArriveStatus[pcAIV0].clear();
    ps.crossCoreFreeEventId_[pcAIV0][0].clear();
    ps.syncArriveStatus[pcAIV0].insert(makeEr(IS_NUM9, aicCore, aiv0Core));  // eventId<16 -> queue0
    ps.syncArriveStatus[pcAIV0].insert(makeEr(IS_NUM29, aicCore, aiv0Core)); // eventId>=16 -> 跳过
    EXPECT_EQ(ps.RecycleCrossCoreEventIds(pcAIV0), SUCCESS);
    EXPECT_EQ(ps.crossCoreFreeEventId_[pcAIV0][0].size(), static_cast<size_t>(IS_NUM1));
    EXPECT_EQ(ps.crossCoreFreeEventId_[pcAIV0][0].front(), IS_NUM9);

    // 场景4: AIV1 core - 覆盖 eventId>=16(queue0) + eventId<16(跳过)
    PipeSync::PipeCoreRealEx pcAIV1(PIPE_S, CoreType::AIV, AIVCore::AIV1);
    ps.syncArriveStatus[pcAIV1].clear();
    ps.crossCoreFreeEventId_[pcAIV1][0].clear();
    ps.syncArriveStatus[pcAIV1].insert(makeEr(IS_NUM30, aicCore, aiv1Core)); // eventId>=16 -> queue0
    ps.syncArriveStatus[pcAIV1].insert(makeEr(IS_NUM10, aicCore, aiv1Core)); // eventId<16 -> 跳过
    EXPECT_EQ(ps.RecycleCrossCoreEventIds(pcAIV1), SUCCESS);
    EXPECT_EQ(ps.crossCoreFreeEventId_[pcAIV1][0].size(), static_cast<size_t>(IS_NUM1));
    EXPECT_EQ(ps.crossCoreFreeEventId_[pcAIV1][0].front(), IS_NUM30);
}

// Helper: create root + leaf Function pair for enableDebug tests
static std::pair<std::shared_ptr<Function>, std::shared_ptr<Function>> SetupDebugFunc(const std::string& name)
{
    auto root = std::make_shared<Function>(Program::GetInstance(), name, name, nullptr);
    root->rootFunc_ = root.get();
    auto leaf = std::make_shared<Function>(Program::GetInstance(), name + "Leaf", name + "Leaf", root.get());
    root->rootFunc_->programs_.emplace(leaf->GetFuncMagic(), leaf.get());
    return {root, leaf};
}

TEST_F(InsertSyncTest, IntraCoreWaitDoesNotImmediatelyReleaseEventId)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("IntraCoreWaitDoesNotImmediatelyReleaseEventId");
    auto& function = *funcs.second;
    PipeSync ps;
    const PipeSync::PipeCoreRealEx source(PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCoreRealEx destination(PIPE_V, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipePairEx pair{source, destination};
    // A single available slot makes premature reuse observable without depending
    // on the allocator's choice among the other six IDs.
    ps.freeEventId_[pair] = {IS_NUM1};
    ps.depOps_.emplace_back(0, PipeSync::PipeCore{PIPE_MTE2, PIPE_MTE2, CoreType::AIV, AIVCore::AIV0});
    ps.depOps_.emplace_back(1, PipeSync::PipeCore{PIPE_V, PIPE_V, CoreType::AIV, AIVCore::AIV0});
    ps.depOps_[0].setPipe = {1};
    ps.depOps_[1].waitPipe = {0};
    ps.oriOpList_.push_back(&IRBuilder().CreateTensorOpStmt(function, Opcode::OP_VEC_DUP, {}, {}));
    ps.oriOpList_.push_back(&IRBuilder().CreateTensorOpStmt(function, Opcode::OP_VEC_DUP, {}, {}));
    std::vector<IndexOp> synced;

    ASSERT_EQ(ps.InjectSetFlag(function, 0, synced), SUCCESS);
    ASSERT_EQ(ps.InjectWaitFlag(function, 1, synced), SUCCESS);
    ASSERT_EQ(synced.size(), 2U);
    EXPECT_EQ(synced[0].second.get().syncQueue_.eventId_, IS_NUM1);
    EXPECT_EQ(synced[1].second.get().syncQueue_.eventId_, IS_NUM1);

    // Inserting the wait is not a runtime ordering edge back to MTE2.
    // The next set must be refused, rather than emitting set/set/wait/wait.
    int eventId = -1;
    size_t waitIdx = 3;
    EXPECT_EQ(ps.GetEventId(pair, eventId, 2, waitIdx, synced, function), FAILED);
    EXPECT_EQ(synced.size(), 2U);
}

TEST_F(InsertSyncTest, IntraCoreEventIdPoolsAreDirectionalAndCoreLocal)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("IntraCoreEventIdPoolsAreDirectionalAndCoreLocal");
    PipeSync ps;
    const PipeSync::PipeCoreRealEx source(PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCoreRealEx destination(PIPE_V, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipePairEx forward{source, destination};
    const PipeSync::PipePairEx reverse{destination, source};
    const PipeSync::PipePairEx otherCore{{PIPE_MTE2, CoreType::AIV, AIVCore::AIV1},
                                         {PIPE_V, CoreType::AIV, AIVCore::AIV1}};
    ps.freeEventId_[forward] = {IS_NUM1};
    ps.freeEventId_[reverse] = {IS_NUM1};
    ps.freeEventId_[otherCore] = {IS_NUM1};
    std::vector<IndexOp> synced;
    size_t waitIdx = 1;
    int eventId = -1;
    ASSERT_EQ(ps.GetEventId(forward, eventId, 0, waitIdx, synced, *funcs.second), SUCCESS);
    EXPECT_EQ(eventId, IS_NUM1);
    EXPECT_EQ(ps.GetEventId(forward, eventId, 0, waitIdx, synced, *funcs.second), FAILED);
    waitIdx = 3;
    ASSERT_EQ(ps.GetEventId(reverse, eventId, 2, waitIdx, synced, *funcs.second), SUCCESS);
    EXPECT_EQ(eventId, IS_NUM1);
    waitIdx = 5;
    ASSERT_EQ(ps.GetEventId(otherCore, eventId, 4, waitIdx, synced, *funcs.second), SUCCESS);
    EXPECT_EQ(eventId, IS_NUM1);
    EXPECT_TRUE(synced.empty());
}

static std::pair<size_t, size_t> AddEventTestDependency(PipeSync& ps, Function& function,
                                                        const PipeSync::PipePairEx& pipes)
{
    const auto setIdx = ps.depOps_.size();
    const auto waitIdx = setIdx + 1;
    ps.depOps_.emplace_back(
        setIdx, PipeSync::PipeCore{pipes.first.pipe, pipes.first.pipe, pipes.first.core, pipes.first.aivCore});
    ps.depOps_.emplace_back(
        waitIdx, PipeSync::PipeCore{pipes.second.pipe, pipes.second.pipe, pipes.second.core, pipes.second.aivCore});
    ps.depOps_[setIdx].setPipe = {waitIdx};
    ps.depOps_[waitIdx].waitPipe = {setIdx};
    ps.oriOpList_.push_back(&IRBuilder().CreateTensorOpStmt(function, Opcode::OP_VEC_DUP, {}, {}));
    ps.oriOpList_.push_back(&IRBuilder().CreateTensorOpStmt(function, Opcode::OP_VEC_DUP, {}, {}));
    return {setIdx, waitIdx};
}

// Independent single-slot event model: unlike the interpreter's counter model,
// a second set before consumption loses an event. Enumerate every enabled
// interleaving of the actual per-pipe queues, including stalled waits.
static bool AllIntraEventSchedulesAreSafe(const std::vector<Operation*>& ops)
{
    using Endpoint = std::tuple<int, int, int>;
    using Resource = std::tuple<Endpoint, Endpoint, int>;
    using Instruction = std::pair<bool, Resource>;
    std::map<Endpoint, std::vector<Instruction>> queues;
    for (const auto* op : ops) {
        const auto& sync = op->syncQueue_;
        const Endpoint source{static_cast<int>(sync.coreType_), static_cast<int>(sync.setAivCore_),
                              static_cast<int>(sync.pipeId_)};
        const Endpoint destination{static_cast<int>(sync.trigCoreType_), static_cast<int>(sync.waitAivCore_),
                                   static_cast<int>(sync.trigPipeId_)};
        const bool isSet = op->GetOpcode() == Opcode::OP_SYNC_SRC;
        queues[isSet ? source : destination].push_back({isSet, {source, destination, sync.eventId_}});
    }
    std::vector<std::vector<Instruction>> pipes;
    for (const auto& entry : queues) {
        pipes.push_back(entry.second);
    }
    std::function<bool(std::vector<size_t>, std::map<Resource, bool>)> explore;
    explore = [&](std::vector<size_t> positions, std::map<Resource, bool> pending) {
        bool finished = true;
        bool progressed = false;
        for (size_t i = 0; i < pipes.size(); ++i) {
            if (positions[i] == pipes[i].size()) {
                continue;
            }
            finished = false;
            const auto& instruction = pipes[i][positions[i]];
            if (!instruction.first && !pending[instruction.second]) {
                continue;
            }
            if (instruction.first && pending[instruction.second]) {
                return false;
            }
            progressed = true;
            auto nextPositions = positions;
            auto nextPending = pending;
            ++nextPositions[i];
            nextPending[instruction.second] = instruction.first;
            if (!explore(nextPositions, nextPending)) {
                return false;
            }
        }
        return finished || progressed;
    };
    return explore(std::vector<size_t>(pipes.size(), 0), {});
}

TEST_F(InsertSyncTest, IntraCoreReuseRequiresCurrentGenerationToReturnToSource)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("IntraCoreReuseRequiresCurrentGenerationToReturnToSource");
    auto& function = *funcs.second;
    PipeSync ps;
    const PipeSync::PipeCoreRealEx a(PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCoreRealEx b(PIPE_V, CoreType::AIV, AIVCore::AIV0);
    ps.freeEventId_[{a, b}] = {1};
    ps.freeEventId_[{b, a}] = {1};
    const auto first = AddEventTestDependency(ps, function, {a, b});
    const auto back = AddEventTestDependency(ps, function, {b, a});
    const auto second = AddEventTestDependency(ps, function, {a, b});
    std::vector<IndexOp> synced;
    ASSERT_EQ(ps.InjectSetFlag(function, first.first, synced), SUCCESS);
    ASSERT_EQ(ps.InjectWaitFlag(function, first.second, synced), SUCCESS);
    EXPECT_TRUE(ps.GetAvailableIntraEventIds({a, b}, ps.eventState_.Known(a)).empty());
    ASSERT_EQ(ps.InjectSetFlag(function, back.first, synced), SUCCESS);
    // The returning wait is visible to the op's preflight, but has not committed.
    const auto projected = ps.ProjectConsumedInstances(ps.depOps_[back.second], a);
    EXPECT_EQ(ps.GetAvailableIntraEventIds({a, b}, projected), std::vector<int>({1}));
    EXPECT_TRUE(ps.GetAvailableIntraEventIds({a, b}, ps.eventState_.Known(a)).empty());
    ASSERT_EQ(ps.InjectWaitFlag(function, back.second, synced), SUCCESS);
    ASSERT_EQ(ps.InjectSetFlag(function, second.first, synced), SUCCESS);
    ASSERT_EQ(ps.InjectWaitFlag(function, second.second, synced), SUCCESS);
    EXPECT_EQ(ps.eventState_.Instances()[0].state, PipeSync::InstanceState::Retired);
    EXPECT_EQ(ps.eventState_.Instances().back().id.generation, 2U);
    // The generation-one proof still at A cannot reclaim generation two.
    EXPECT_TRUE(ps.GetAvailableIntraEventIds({a, b}, ps.eventState_.Known(a)).empty());
    std::vector<Operation*> ordered;
    for (const auto& op : synced) {
        ordered.push_back(&op.second.get());
    }
    EXPECT_EQ(ps.ValidateEventReuse(ordered), SUCCESS);
    EXPECT_TRUE(AllIntraEventSchedulesAreSafe(ordered));
    // Moving the second set ahead of the return wait destroys the proof even
    // though the serialized list still contains both set/wait pairs.
    std::swap(ordered[3], ordered[4]);
    EXPECT_EQ(ps.ValidateEventReuse(ordered), FAILED);
    EXPECT_FALSE(AllIntraEventSchedulesAreSafe(ordered));
}

TEST_F(InsertSyncTest, LiteCrossLogicalCoreReusesIntraEventOnlyAfterConsumptionReturns)
{
    NpuArchScope archScope(NPUArch::DAV_3113);
    auto funcs = SetupDebugFunc("LiteCrossLogicalCoreEventReuse");
    auto& function = *funcs.second;
    PipeSync ps;
    // On Lite, AIC MTE2 and AIV V share a physical core but keep their
    // canonical logical endpoints. This exercises IsIntraCoreEvent's Lite branch.
    const PipeSync::PipeCoreRealEx a(PIPE_MTE2, CoreType::AIC, AIVCore::UNSPECIFIED);
    const PipeSync::PipeCoreRealEx b(PIPE_V, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipePairEx forward{a, b}, reverse{b, a};
    ASSERT_TRUE(ps.IsIntraCoreEvent(forward));
    ASSERT_TRUE(ps.IsIntraCoreEvent(reverse));
    ps.freeEventId_[forward] = {1};
    ps.freeEventId_[reverse] = {1};
    ps.InitCVEventIdQ();
    const auto crossQueues = ps.crossCoreFreeEventId_;
    const auto first = AddEventTestDependency(ps, function, forward);
    const auto back = AddEventTestDependency(ps, function, reverse);
    const auto second = AddEventTestDependency(ps, function, forward);
    std::vector<IndexOp> synced;
    ASSERT_EQ(ps.InjectSetFlag(function, first.first, synced), SUCCESS);
    ASSERT_EQ(ps.InjectWaitFlag(function, first.second, synced), SUCCESS);
    const auto* old = ps.eventState_.Find(first.first, first.second);
    ASSERT_NE(old, nullptr);
    EXPECT_TRUE(old->intraCore);
    const auto blocked = ps.AnalyzeIntraEventCapacity(ps.depOps_[second.first]);
    ASSERT_TRUE(blocked.has_value());
    EXPECT_EQ(blocked->pipes, forward);
    EXPECT_EQ(blocked->required, 1U);
    EXPECT_EQ(blocked->available, 0U);
    size_t waitIdx = second.second;
    int eventId = -1;
    EXPECT_EQ(ps.GetEventId(forward, eventId, second.first, waitIdx, synced, function), FAILED);
    ASSERT_EQ(ps.InjectSetFlag(function, back.first, synced), SUCCESS);
    ASSERT_EQ(ps.InjectWaitFlag(function, back.second, synced), SUCCESS);
    EXPECT_FALSE(ps.AnalyzeIntraEventCapacity(ps.depOps_[second.first]).has_value());
    ASSERT_EQ(ps.InjectSetFlag(function, second.first, synced), SUCCESS);
    ASSERT_EQ(ps.InjectWaitFlag(function, second.second, synced), SUCCESS);
    const auto* current = ps.eventState_.Find(second.first, second.second);
    ASSERT_NE(current, nullptr);
    EXPECT_TRUE(current->intraCore);
    EXPECT_EQ(current->id.resource.eventId, 1);
    EXPECT_EQ(current->id.generation, 2U);
    EXPECT_EQ(ps.crossCoreFreeEventId_, crossQueues);
    EXPECT_TRUE(ps.forceSyncOps_.empty());
    std::vector<Operation*> ordered;
    for (const auto& entry : synced) {
        const auto* op = &entry.second.get();
        EXPECT_TRUE(op->GetOpcode() == Opcode::OP_SYNC_SRC || op->GetOpcode() == Opcode::OP_SYNC_DST);
        ordered.push_back(&entry.second.get());
    }
    EXPECT_EQ(ps.ValidateEventReuse(ordered), SUCCESS);
    EXPECT_TRUE(AllIntraEventSchedulesAreSafe(ordered));
    std::swap(ordered[3], ordered[4]);
    EXPECT_EQ(ps.ValidateEventReuse(ordered), FAILED);
    EXPECT_FALSE(AllIntraEventSchedulesAreSafe(ordered));
}

TEST_F(InsertSyncTest, NonLiteCrossCoreDependencyUsesCrossCoreAllocator)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("NonLiteCrossCoreAllocator");
    PipeSync ps;
    const PipeSync::PipePairEx pair{{PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED},
                                    {PIPE_V, CoreType::AIV, AIVCore::AIV0}};
    EXPECT_FALSE(ps.IsIntraCoreEvent(pair));
    ps.InitCVEventIdQ();
    const auto dep = AddEventTestDependency(ps, *funcs.second, pair);
    std::vector<IndexOp> synced;
    EXPECT_FALSE(ps.AnalyzeIntraEventCapacity(ps.depOps_[dep.first]).has_value());
    ASSERT_EQ(ps.InjectSetFlag(*funcs.second, dep.first, synced), SUCCESS);
    const auto* instance = ps.eventState_.Find(dep.first, dep.second);
    ASSERT_NE(instance, nullptr);
    EXPECT_FALSE(instance->intraCore);
    EXPECT_FALSE(ps.eventState_.HasActive(instance->id.resource));
    ASSERT_EQ(synced.size(), 1U);
    EXPECT_EQ(synced.front().second.get().GetOpcode(), Opcode::OP_CV_SYNC_SRC);
}

TEST_F(InsertSyncTest, EventWaitUsesFrozenSetSnapshot)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("EventWaitUsesFrozenSetSnapshot");
    auto& function = *funcs.second;
    PipeSync ps;
    const PipeSync::PipeCoreRealEx a(PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCoreRealEx b(PIPE_V, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCoreRealEx c(PIPE_MTE3, CoreType::AIV, AIVCore::AIV0);
    const auto ab = AddEventTestDependency(ps, function, {a, b});
    const auto bc = AddEventTestDependency(ps, function, {b, c});
    std::vector<IndexOp> synced;
    ASSERT_EQ(ps.InjectSetFlag(function, ab.first, synced), SUCCESS);
    const auto oldId = ps.eventState_.Instances()[0].id;
    ASSERT_EQ(ps.InjectSetFlag(function, bc.first, synced), SUCCESS);
    ASSERT_EQ(ps.InjectWaitFlag(function, ab.second, synced), SUCCESS);
    ASSERT_EQ(ps.eventState_.Known(b).count(oldId), 1U);
    ASSERT_EQ(ps.InjectWaitFlag(function, bc.second, synced), SUCCESS);
    EXPECT_EQ(ps.eventState_.Known(c).count(oldId), 0U);
    EXPECT_EQ(ps.InjectWaitFlag(function, bc.second, synced), FAILED);
}

TEST_F(InsertSyncTest, IntraCoreStaleFreeQueueEntryCannotBypassActiveInstance)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("IntraCoreStaleFreeQueueEntryCannotBypassActiveInstance");
    auto& function = *funcs.second;
    PipeSync ps;
    const PipeSync::PipePairEx pipes{{PIPE_MTE2, CoreType::AIV, AIVCore::AIV0}, {PIPE_V, CoreType::AIV, AIVCore::AIV0}};
    ps.freeEventId_[pipes] = {1};
    const auto dep = AddEventTestDependency(ps, function, pipes);
    std::vector<IndexOp> synced;
    ASSERT_EQ(ps.InjectSetFlag(function, dep.first, synced), SUCCESS);
    ps.freeEventId_[pipes].push_back(1);
    EXPECT_TRUE(ps.GetAvailableIntraEventIds(pipes, {}).empty());
    int eventId = -1;
    size_t waitIdx = 3;
    EXPECT_EQ(ps.GetEventId(pipes, eventId, 2, waitIdx, synced, function), FAILED);
    EXPECT_EQ(ps.eventState_.Instances().size(), 1U);
}

TEST_F(InsertSyncTest, IntraCoreExhaustionDoesNotCommitWaitOrForce)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("IntraCoreExhaustionDoesNotCommitWaitOrForce");
    auto& function = *funcs.second;
    PipeSync ps;
    const PipeSync::PipeCoreRealEx a(PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCoreRealEx b(PIPE_V, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCoreRealEx c(PIPE_MTE3, CoreType::AIV, AIVCore::AIV0);
    ps.freeEventId_[{b, c}] = {1};
    const auto pending = AddEventTestDependency(ps, function, {a, b});
    const auto occupied = AddEventTestDependency(ps, function, {b, c});
    const auto next = AddEventTestDependency(ps, function, {b, c});
    ps.RemoveOpDep(ps.depOps_[next.first], ps.depOps_[next.second]);
    ASSERT_EQ(ps.AddOpDep(ps.depOps_[pending.second], ps.depOps_[next.second]), SUCCESS);
    std::vector<IndexOp> synced;
    ASSERT_EQ(ps.InjectSetFlag(function, pending.first, synced), SUCCESS);
    ASSERT_EQ(ps.InjectSetFlag(function, occupied.first, synced), SUCCESS);
    ASSERT_EQ(ps.InjectWaitFlag(function, occupied.second, synced), SUCCESS);
    const auto before = synced.size();
    EXPECT_EQ(ps.InjectSync(function, ps.oriOpList_, pending.second, synced), FAILED);
    EXPECT_EQ(synced.size(), before);
    EXPECT_FALSE(ps.depOps_[pending.second].issued);
    EXPECT_EQ(ps.eventState_.Instances()[0].state, PipeSync::InstanceState::Allocated);
    EXPECT_EQ(ps.eventState_.Known(b).count(ps.eventState_.Instances()[0].id), 0U);
    EXPECT_TRUE(ps.forceSyncOps_.empty());
    EXPECT_TRUE(ps.crossCoreFreeEventId_.empty());
}

TEST_F(InsertSyncTest, CurrentOpWaitEnablesOutgoingIntraCoreReuse)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("CurrentOpWaitEnablesOutgoingIntraCoreReuse");
    auto& function = *funcs.second;
    PipeSync ps;
    const PipeSync::PipeCoreRealEx a(PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCoreRealEx b(PIPE_V, CoreType::AIV, AIVCore::AIV0);
    ps.freeEventId_[{a, b}] = {1};
    const auto first = AddEventTestDependency(ps, function, {a, b});
    const auto back = AddEventTestDependency(ps, function, {b, a});
    const auto next = AddEventTestDependency(ps, function, {a, b});
    ps.RemoveOpDep(ps.depOps_[next.first], ps.depOps_[next.second]);
    ASSERT_EQ(ps.AddOpDep(ps.depOps_[back.second], ps.depOps_[next.second]), SUCCESS);
    std::vector<IndexOp> synced;
    ASSERT_EQ(ps.InjectSetFlag(function, first.first, synced), SUCCESS);
    ASSERT_EQ(ps.InjectWaitFlag(function, first.second, synced), SUCCESS);
    ASSERT_EQ(ps.InjectSetFlag(function, back.first, synced), SUCCESS);
    ps.depOps_[back.first].issued = true;
    ps.orderedOpList_.push(back.second);
    PipeSync::IssueQueue queue(a);
    queue.ops = {back.second};
    std::vector<size_t> popped;
    bool deadlock = false;
    ASSERT_EQ(ps.PopFromQueue(queue, popped, deadlock, synced), SUCCESS);
    ASSERT_EQ(popped, std::vector<size_t>({back.second}));
    ASSERT_EQ(ps.InjectSync(function, ps.oriOpList_, back.second, synced), SUCCESS);
    EXPECT_EQ(ps.setWaitPairMap_.at({back.second, next.second}), 1);
    EXPECT_EQ(ps.eventState_.Instances().back().id.generation, 2U);
    ASSERT_EQ(ps.InjectWaitFlag(function, next.second, synced), SUCCESS);
    std::vector<Operation*> ordered;
    for (const auto& op : synced) {
        ordered.push_back(&op.second.get());
    }
    EXPECT_EQ(ps.ValidateEventReuse(ordered), SUCCESS);
}

TEST_F(InsertSyncTest, ConsumedIntraCorePoolExhaustionDoesNotInsertForce)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("ConsumedIntraCorePoolExhaustionDoesNotInsertForce");
    auto& function = *funcs.second;
    PipeSync ps;
    const PipeSync::PipePairEx pipes{{PIPE_MTE2, CoreType::AIV, AIVCore::AIV0}, {PIPE_V, CoreType::AIV, AIVCore::AIV0}};
    ps.freeEventId_[pipes] = {1};
    const auto first = AddEventTestDependency(ps, function, pipes);
    std::vector<IndexOp> synced;
    ASSERT_EQ(ps.InjectSetFlag(function, first.first, synced), SUCCESS);
    ASSERT_EQ(ps.InjectWaitFlag(function, first.second, synced), SUCCESS);
    PipeSync::DataDepInfo depInfo;
    ASSERT_EQ(ps.GetDepInfo(synced, pipes, depInfo), SUCCESS);
    EXPECT_TRUE(depInfo.opDepList.empty());
    uint64_t attempts = 0;
    bool deadlock = false;
    for (uint64_t i = 1; i < EVENTID_DEADLOCK_ENTER_TIME; ++i) {
        EXPECT_EQ(ps.ProcessDeadLock(attempts, deadlock, synced), SUCCESS);
    }
    EXPECT_EQ(ps.ProcessDeadLock(attempts, deadlock, synced), FAILED);
    EXPECT_EQ(synced.size(), 2U);
    EXPECT_EQ(ps.eventState_.Instances().size(), 1U);
    EXPECT_TRUE(ps.forceSyncOps_.empty());
}

// Exercise dispatch, dependency discovery, scheduling, exhaustion recovery and
// final validation together, without injecting allocator state or dependencies.
static Status InsertOneWayPipeline(Function& function, size_t stages, bool grouped, std::vector<Operation*>& synced)
{
    IRBuilder builder;
    const std::vector<int64_t> shape{8, 16};
    const auto shapeImmediate = OpImmediate::Specified(shape);
    std::vector<std::shared_ptr<LogicalTensor>> inputs;
    std::vector<std::shared_ptr<LogicalTensor>> loaded;
    std::vector<std::shared_ptr<LogicalTensor>> outputs;
    for (size_t i = 0; i < stages; ++i) {
        auto input = builder.CreateTensorVar(function, DT_FP32, shape);
        auto load = builder.CreateTensorVar(function, DT_FP32, shape);
        auto output = builder.CreateTensorVar(function, DT_FP32, shape);
        input->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
        load->SetMemoryTypeBoth(MemoryType::MEM_UB);
        output->SetMemoryTypeBoth(MemoryType::MEM_UB);
        // Each stage owns distinct storage, so V never signals MTE2 for reuse.
        input->memoryrange = TileRange(i * 2048, i * 2048 + 512, static_cast<int>(i * 3 + 1));
        load->memoryrange = TileRange(i * 2048, i * 2048 + 512, static_cast<int>(i * 3 + 2));
        output->memoryrange = TileRange(i * 2048 + 1024, i * 2048 + 1536, static_cast<int>(i * 3 + 3));
        inputs.push_back(input);
        loaded.push_back(load);
        outputs.push_back(output);
    }
    auto addCopy = [&](size_t i) {
        auto& copy = builder.CreateTensorOpStmt(function, Opcode::OP_COPY_IN, {inputs[i]}, {loaded[i]});
        copy.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MemoryType::MEM_UB,
                                                              shapeImmediate, shapeImmediate));
        copy.SetAIVCore(AIVCore::AIV0);
    };
    auto addCast = [&](size_t i) {
        auto& cast = builder.CreateTensorOpStmt(function, Opcode::OP_CAST, {loaded[i]}, {outputs[i]});
        cast.SetAIVCore(AIVCore::AIV0);
    };
    for (size_t i = 0; i < stages; ++i) {
        addCopy(i);
        if (!grouped) {
            addCast(i);
        }
    }
    if (grouped) {
        for (size_t i = 0; i < stages; ++i) {
            addCast(i);
        }
    }
    PipeSync sync;
    return sync.InsertSync(function, synced);
}

TEST_F(InsertSyncTest, OneWayPipelineReturnsProofWithoutDraining)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto seven = SetupDebugFunc("OneWayPipelineSevenEvents");
    std::vector<Operation*> synced;
    ASSERT_EQ(InsertOneWayPipeline(*seven.second, 7, false, synced), SUCCESS);
    std::vector<Operation*> flags;
    for (auto* op : synced) {
        if (op->GetOpcode() == Opcode::OP_SYNC_SRC || op->GetOpcode() == Opcode::OP_SYNC_DST) {
            flags.push_back(op);
        }
    }
    EXPECT_EQ(flags.size(), 14U);
    EXPECT_TRUE(AllIntraEventSchedulesAreSafe(flags));
    auto eight = SetupDebugFunc("OneWayPipelineEightEvents");
    synced.clear();
    ASSERT_EQ(InsertOneWayPipeline(*eight.second, 8, false, synced), SUCCESS);
    flags.clear();
    size_t barriers = 0;
    for (auto* op : synced) {
        if (op->GetOpcode() == Opcode::OP_BAR_ALL) {
            EXPECT_TRUE(AllIntraEventSchedulesAreSafe(flags));
            flags.clear();
            ++barriers;
        }
        EXPECT_NE(op->GetOpcode(), Opcode::OP_FFTS_CROSS_CORE_SYNC);
        if (op->GetOpcode() == Opcode::OP_SYNC_SRC || op->GetOpcode() == Opcode::OP_SYNC_DST) {
            flags.push_back(op);
        }
    }
    EXPECT_EQ(flags.size(), 18U);
    EXPECT_EQ(barriers, 0U);
    EXPECT_TRUE(AllIntraEventSchedulesAreSafe(flags));
}

TEST_F(InsertSyncTest, YieldRejectsDependencyCycleAfterReplay)
{
    PipeSync ps;
    for (auto pipe : {PIPE_MTE2, PIPE_MTE2, PIPE_V, PIPE_MTE1}) {
        ps.depOps_.emplace_back(ps.depOps_.size(), PipeSync::PipeCore{pipe, pipe, CoreType::AIV, AIVCore::AIV0});
    }
    ps.depOps_[0].setPipe = {3};
    ps.depOps_[3].waitPipe = {0};
    // Replay may place the wait before the next producer, despite its larger index.
    ps.depOps_[3].setPipe = {2};
    ps.depOps_[2].waitPipe = {3};
    ps.depOps_[2].setPipe = {1};
    ps.depOps_[1].waitPipe = {2};
    PipeSync::IssueQueue queue({PIPE_MTE2, CoreType::AIV, AIVCore::AIV0});
    queue.ops = {0, 1};
    bool failed = false;
    std::vector<size_t> order;
    ASSERT_TRUE(ps.BuildReplayIssueOrder(ps.depOps_, order));
    ASSERT_EQ(ps.AdjustOpDep(ps.depOps_[0], 3, queue, failed), SUCCESS);
    EXPECT_TRUE(failed);
    EXPECT_EQ(ps.depOps_[0].setPipe, std::vector<size_t>({3}));
    EXPECT_EQ(ps.depOps_[3].waitPipe, std::vector<size_t>({0}));
    EXPECT_TRUE(ps.depOps_[1].setPipe.empty());
    EXPECT_TRUE(ps.BuildReplayIssueOrder(ps.depOps_, order));

    ps.depOps_[3].setPipe.clear();
    ps.depOps_[2].waitPipe.clear();
    failed = false;
    ASSERT_EQ(ps.AdjustOpDep(ps.depOps_[0], 3, queue, failed), SUCCESS);
    EXPECT_FALSE(failed);
    EXPECT_TRUE(ps.depOps_[0].setPipe.empty());
    EXPECT_EQ(ps.depOps_[1].setPipe, std::vector<size_t>({3}));
    EXPECT_EQ(ps.depOps_[3].waitPipe, std::vector<size_t>({1}));
    EXPECT_TRUE(ps.BuildReplayIssueOrder(ps.depOps_, order));
}

TEST_F(InsertSyncTest, ReplayMergeRejectsBackwardEmissionEvenWithoutCycle)
{
    PipeSync ps;
    const PipeSync::PipePairEx pipes{{PIPE_MTE2, CoreType::AIV, AIVCore::AIV0}, {PIPE_V, CoreType::AIV, AIVCore::AIV0}};
    std::vector<PipeSync::DepOp> plan;
    for (size_t i = 0; i < 4; ++i) {
        const auto& pipe = i % 2 == 0 ? pipes.first : pipes.second;
        plan.emplace_back(i, PipeSync::PipeCore{pipe.pipe, pipe.pipe, pipe.core, pipe.aivCore});
    }
    plan[0].setPipe = {1};
    plan[1].waitPipe = {0};
    plan[2].setPipe = {3};
    plan[3].waitPipe = {2};
    std::vector<size_t> order;
    EXPECT_FALSE(ps.MergeDependenciesForReplay(plan, pipes, order));
    EXPECT_EQ(plan[0].setPipe, std::vector<size_t>({1}));
    EXPECT_EQ(plan[2].setPipe, std::vector<size_t>({3}));
}

TEST_F(InsertSyncTest, EventValidationRejectsWaitBeforeMatchingSet)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("EventValidationRejectsWaitBeforeMatchingSet");
    PipeSync ps;
    const PipeSync::PipePairEx pipes{{PIPE_MTE2, CoreType::AIV, AIVCore::AIV0}, {PIPE_V, CoreType::AIV, AIVCore::AIV0}};
    const auto dep = AddEventTestDependency(ps, *funcs.second, pipes);
    std::vector<IndexOp> synced;
    ASSERT_EQ(ps.InjectSetFlag(*funcs.second, dep.first, synced), SUCCESS);
    ASSERT_EQ(ps.InjectWaitFlag(*funcs.second, dep.second, synced), SUCCESS);
    ASSERT_EQ(synced.size(), 2U);
    std::vector<Operation*> ordered{&synced[0].second.get(), &synced[1].second.get()};
    EXPECT_EQ(ps.ValidateEventReuse(ordered), SUCCESS);
    std::reverse(ordered.begin(), ordered.end());
    EXPECT_EQ(ps.ValidateEventReuse(ordered), FAILED);
}

TEST_F(InsertSyncTest, ReplayMergePreservesFifoAndRejectsReverseDependencyCycles)
{
    PipeSync ps;
    const PipeSync::PipePairEx pipes{{PIPE_MTE2, CoreType::AIV, AIVCore::AIV0}, {PIPE_V, CoreType::AIV, AIVCore::AIV0}};
    std::vector<PipeSync::DepOp> plan;
    for (size_t i = 0; i < 4; ++i) {
        const auto& pipe = i % 2 == 0 ? pipes.first : pipes.second;
        plan.emplace_back(i, PipeSync::PipeCore{pipe.pipe, pipe.pipe, pipe.core, pipe.aivCore});
    }
    plan[0].setPipe = {1};
    plan[1].waitPipe = {0};
    plan[2].setPipe = {3};
    plan[3].waitPipe = {2};
    // The reverse data dependency makes delaying the first wait to set2 unsafe.
    plan[1].setPipe = {2};
    plan[2].waitPipe = {1};
    std::vector<size_t> order;
    EXPECT_FALSE(ps.MergeDependenciesForReplay(plan, pipes, order));
    EXPECT_EQ(plan[0].setPipe, std::vector<size_t>({1}));
    EXPECT_EQ(plan[2].waitPipe, std::vector<size_t>({1}));
    plan[1].setPipe.clear();
    plan[2].waitPipe.clear();
    // Removing the reverse edge makes the graph acyclic, but the merged set
    // would still follow its wait in the original-order output.
    EXPECT_FALSE(ps.MergeDependenciesForReplay(plan, pipes, order));
    EXPECT_EQ(plan[0].setPipe, std::vector<size_t>({1}));
    EXPECT_EQ(plan[2].setPipe, std::vector<size_t>({3}));
}

TEST_F(InsertSyncTest, OverlapSelectionSkipsTouchingDependencyIntervals)
{
    PipeSync ps;
    PipeSync::DataDepInfo info;
    info.opDepList = {{0, 2}, {2, 4}};
    int candidate = -1;
    EXPECT_FALSE(ps.FindMaxOverlap(info, candidate));
    info.opDepList.push_back({3, 5});
    ASSERT_TRUE(ps.FindMaxOverlap(info, candidate));
    EXPECT_EQ(candidate, 1);
}

TEST_F(InsertSyncTest, OverlapSelectionSkipsNestedWaits)
{
    PipeSync ps;
    PipeSync::DataDepInfo info;
    info.opDepList = {{26, 58}, {31, 94}, {36, 102}, {41, 63}, {46, 109}, {51, 112}};
    int candidate = -1;
    ASSERT_TRUE(ps.FindMaxOverlap(info, candidate));
    EXPECT_EQ(candidate, 1);
}

TEST_F(InsertSyncTest, OverlapSelectionRejectsReversedAndEqualEndpoints)
{
    PipeSync ps;
    PipeSync::DataDepInfo info;
    for (const auto& pairs : std::vector<std::vector<std::pair<int, int>>>{
             {{5, 10}, {4, 12}}, {{4, 10}, {4, 12}}, {{2, 12}, {4, 10}}, {{2, 10}, {4, 10}}}) {
        info.opDepList = pairs;
        int candidate = -1;
        EXPECT_FALSE(ps.FindMaxOverlap(info, candidate));
        EXPECT_EQ(candidate, -1);
    }
}

TEST_F(InsertSyncTest, ReplayMergeSearchesPastBackwardCandidate)
{
    PipeSync ps;
    const PipeSync::PipePairEx pipes{{PIPE_MTE1, CoreType::AIC, AIVCore::UNSPECIFIED},
                                     {PIPE_M, CoreType::AIC, AIVCore::UNSPECIFIED}};
    std::vector<PipeSync::DepOp> plan;
    for (size_t i = 0; i < 6; ++i) {
        const auto& pipe = (i == 0 || i == 2 || i == 3) ? pipes.first : pipes.second;
        plan.emplace_back(i, PipeSync::PipeCore{pipe.pipe, pipe.pipe, pipe.core, pipe.aivCore});
    }
    plan[0].setPipe = {1};
    plan[1].waitPipe = {0};
    plan[2].setPipe = {4};
    plan[4].waitPipe = {2};
    plan[3].setPipe = {5};
    plan[5].waitPipe = {3};
    std::vector<size_t> order;
    ASSERT_TRUE(ps.MergeDependenciesForReplay(plan, pipes, order));
    EXPECT_EQ(plan[0].setPipe, std::vector<size_t>({1}));
    EXPECT_EQ(plan[3].setPipe, std::vector<size_t>({4}));
    EXPECT_EQ(plan[4].waitPipe, std::vector<size_t>({3}));
    EXPECT_TRUE(plan[2].setPipe.empty());
}

TEST_F(InsertSyncTest, ReplayMergeSkipsDependenciesAfterBlockedProducer)
{
    PipeSync ps;
    const PipeSync::PipePairEx pipes{{PIPE_MTE1, CoreType::AIC, AIVCore::UNSPECIFIED},
                                     {PIPE_M, CoreType::AIC, AIVCore::UNSPECIFIED}};
    std::vector<PipeSync::DepOp> plan;
    for (size_t i = 0; i < 6; ++i) {
        const auto& pipe = (i == 0 || i == 2 || i == 3) ? pipes.first : pipes.second;
        plan.emplace_back(i, PipeSync::PipeCore{pipe.pipe, pipe.pipe, pipe.core, pipe.aivCore});
    }
    plan[0].setPipe = {1};
    plan[1].waitPipe = {0};
    plan[2].setPipe = {4};
    plan[4].waitPipe = {2};
    plan[3].setPipe = {5};
    plan[5].waitPipe = {3};
    ps.orderedOpList_.push(0);
    std::vector<size_t> order;
    EXPECT_FALSE(ps.MergeDependenciesForReplay(plan, pipes, order));
    EXPECT_EQ(plan[2].setPipe, std::vector<size_t>({4}));
    EXPECT_EQ(plan[3].setPipe, std::vector<size_t>({5}));
    ps.orderedOpList_.pop();
    ps.orderedOpList_.push(2);
    ASSERT_TRUE(ps.MergeDependenciesForReplay(plan, pipes, order));
    EXPECT_TRUE(plan[2].setPipe.empty());
    EXPECT_EQ(plan[3].setPipe, std::vector<size_t>({4}));
}

TEST_F(InsertSyncTest, ReplayMergeDoesNotCancelRetiredInstance)
{
    auto funcs = SetupDebugFunc("ReplayMergeDoesNotCancelRetiredInstance");
    auto& function = *funcs.second;
    PipeSync ps;
    const PipeSync::PipePairEx pipes{{PIPE_MTE1, CoreType::AIC, AIVCore::UNSPECIFIED},
                                     {PIPE_M, CoreType::AIC, AIVCore::UNSPECIFIED}};
    std::vector<PipeSync::DepOp> plan;
    for (size_t i = 0; i < 4; ++i) {
        const auto& pipe = i < 2 ? pipes.first : pipes.second;
        plan.emplace_back(i, PipeSync::PipeCore{pipe.pipe, pipe.pipe, pipe.core, pipe.aivCore});
    }
    plan[0].setPipe = {2};
    plan[2].waitPipe = {0};
    plan[1].setPipe = {3};
    plan[3].waitPipe = {1};
    const auto resource = ps.GetEventResource(pipes, 1);
    auto& set = IRBuilder().CreateTensorOpStmt(function, Opcode::OP_SYNC_SRC, {}, {});
    auto& wait = IRBuilder().CreateTensorOpStmt(function, Opcode::OP_SYNC_DST, {}, {});
    ASSERT_EQ(ps.eventState_.Allocate(resource, pipes, 0, 2), SUCCESS);
    ASSERT_EQ(ps.eventState_.RecordSet(resource, pipes, 0, 2, set, true), SUCCESS);
    ASSERT_EQ(ps.eventState_.Consume(0, 2, wait), SUCCESS);
    ASSERT_EQ(ps.eventState_.RetireConsumed(0, 2), SUCCESS);
    ps.orderedOpList_.push(1);
    std::vector<size_t> order;
    EXPECT_FALSE(ps.MergeDependenciesForReplay(plan, pipes, order));
    EXPECT_EQ(plan[0].setPipe, std::vector<size_t>({2}));
    EXPECT_EQ(plan[1].setPipe, std::vector<size_t>({3}));
}

TEST_F(InsertSyncTest, ReplayResetDiscardsConsumedProofsAndRestoresQueuePositions)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("ReplayResetDiscardsConsumedProofs");
    PipeSync ps;
    const PipeSync::PipePairEx pipes{{PIPE_MTE2, CoreType::AIV, AIVCore::AIV0}, {PIPE_V, CoreType::AIV, AIVCore::AIV0}};
    const auto dep = AddEventTestDependency(ps, *funcs.second, pipes);
    const auto plan = ps.depOps_;
    const auto queues = ps.issueState_;
    std::vector<IndexOp> synced;
    ASSERT_EQ(ps.InjectSetFlag(*funcs.second, dep.first, synced), SUCCESS);
    ASSERT_EQ(ps.InjectWaitFlag(*funcs.second, dep.second, synced), SUCCESS);
    ASSERT_FALSE(ps.eventState_.Known(pipes.second).empty());
    ps.depOps_[dep.first].issued = true;
    ps.issueState_[0].currOp = 1;
    ps.ResetForReplay(plan, queues, {dep.first, dep.second});
    EXPECT_TRUE(ps.eventState_.Instances().empty());
    EXPECT_TRUE(ps.eventState_.Known(pipes.second).empty());
    EXPECT_TRUE(ps.freeEventId_.empty());
    EXPECT_TRUE(ps.setWaitPairMap_.empty());
    EXPECT_TRUE(ps.setOpMap.empty());
    EXPECT_TRUE(ps.waitOpMap.empty());
    EXPECT_FALSE(ps.depOps_[dep.first].issued);
    EXPECT_EQ(ps.issueState_[0].currOp, 0U);
}

TEST_F(InsertSyncTest, ReplayMergeDoesNotCrossLegacyCrossCoreForceBoundary)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    PipeSync ps;
    const PipeSync::PipePairEx pipes{{PIPE_MTE1, CoreType::AIC, AIVCore::UNSPECIFIED},
                                     {PIPE_M, CoreType::AIC, AIVCore::UNSPECIFIED}};
    std::vector<PipeSync::DepOp> plan;
    for (size_t i = 0; i < 4; ++i) {
        const auto& pipe = i % 2 == 0 ? pipes.first : pipes.second;
        plan.emplace_back(i, PipeSync::PipeCore{pipe.pipe, pipe.pipe, pipe.core, pipe.aivCore});
    }
    plan.emplace_back(4, PipeSync::PipeCore{PIPE_V, PIPE_V, CoreType::AIV, AIVCore::AIV0});
    plan[0].setPipe = {1};
    plan[1].waitPipe = {0};
    plan[2].setPipe = {3};
    plan[3].waitPipe = {2};
    plan[1].setPipe = {4};
    plan[4].waitPipe = {1};
    std::vector<size_t> order;
    // set2 cannot be issued before the cross-core endpoint at original op1.
    EXPECT_FALSE(ps.MergeDependenciesForReplay(plan, pipes, order));
    EXPECT_EQ(plan[0].setPipe, std::vector<size_t>({1}));
    EXPECT_EQ(plan[2].setPipe, std::vector<size_t>({3}));
}

TEST_F(InsertSyncTest, LongOneWayPipelineReturnsProofWithoutDraining)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("LongOneWayPipelineReplays");
    std::vector<Operation*> synced;
    ASSERT_EQ(InsertOneWayPipeline(*funcs.second, 12, false, synced), SUCCESS);
    std::vector<Operation*> flags;
    size_t barriers = 0;
    for (auto* op : synced) {
        if (op->GetOpcode() == Opcode::OP_BAR_ALL) {
            EXPECT_TRUE(AllIntraEventSchedulesAreSafe(flags));
            flags.clear();
            ++barriers;
        }
        EXPECT_NE(op->GetOpcode(), Opcode::OP_FFTS_CROSS_CORE_SYNC);
        if (op->GetOpcode() == Opcode::OP_SYNC_SRC || op->GetOpcode() == Opcode::OP_SYNC_DST) {
            flags.push_back(op);
        }
    }
    EXPECT_EQ(flags.size(), 26U);
    EXPECT_EQ(barriers, 0U);
    EXPECT_TRUE(AllIntraEventSchedulesAreSafe(flags));
}

TEST_F(InsertSyncTest, CubeMte1ToMatrixExhaustionReturnsProofWithoutDraining)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("CubeMte1ToMatrixExhaustionReplay");
    auto& function = *funcs.second;
    IRBuilder builder;
    const std::vector<int64_t> shape{16, 16};
    for (size_t i = 0; i < 8; ++i) {
        auto l1 = builder.CreateTensorVar(function, DT_FP16, shape);
        auto a = builder.CreateTensorVar(function, DT_FP16, shape);
        auto b = builder.CreateTensorVar(function, DT_FP16, shape);
        auto output = builder.CreateTensorVar(function, DT_FP32, shape);
        l1->SetMemoryTypeBoth(MemoryType::MEM_L1);
        a->SetMemoryTypeBoth(MemoryType::MEM_L0A);
        b->SetMemoryTypeBoth(MemoryType::MEM_L0B);
        output->SetMemoryTypeBoth(MemoryType::MEM_L0C);
        for (auto tensor : {l1, a, b, output}) {
            tensor->memoryrange = TileRange(i * 2048, i * 2048 + 512, tensor->GetMagic());
        }
        builder.CreateTensorOpStmt(function, Opcode::OP_L1_TO_L0A, {l1}, {a});
        builder.CreateTensorOpStmt(function, Opcode::OP_A_MUL_B, {a, b}, {output});
    }
    const auto original = function.Operations(false).DuplicatedOpList();
    PipeSync ps;
    std::vector<Operation*> synced;
    ASSERT_EQ(ps.InsertSync(function, synced), SUCCESS);
    std::vector<Operation*> business;
    std::vector<Operation*> flags;
    size_t barriers = 0;
    for (auto* op : synced) {
        if (op->GetOpcode() == Opcode::OP_BAR_ALL) {
            EXPECT_TRUE(AllIntraEventSchedulesAreSafe(flags));
            flags.clear();
            ++barriers;
            continue;
        }
        EXPECT_NE(op->GetOpcode(), Opcode::OP_FFTS_CROSS_CORE_SYNC);
        if (op->GetOpcode() == Opcode::OP_SYNC_SRC || op->GetOpcode() == Opcode::OP_SYNC_DST) {
            flags.push_back(op);
        } else {
            business.push_back(op);
        }
    }
    EXPECT_EQ(business, original);
    EXPECT_EQ(flags.size(), 18U);
    EXPECT_EQ(barriers, 0U);
    EXPECT_TRUE(AllIntraEventSchedulesAreSafe(flags));
}

TEST_F(InsertSyncTest, ConsumedPoolReturnsThroughDirectedDependencyBeforeDraining)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("ConsumedPoolReturnsThroughDirectedDependencyBeforeDraining");
    auto& function = *funcs.second;
    PipeSync ps;
    const PipeSync::PipePairEx pipes{{PIPE_MTE2, CoreType::AIV, AIVCore::AIV0}, {PIPE_V, CoreType::AIV, AIVCore::AIV0}};
    for (size_t i = 0; i < 16; ++i) {
        const auto& pipe = i % 2 == 0 ? pipes.first : pipes.second;
        ps.depOps_.emplace_back(i, PipeSync::PipeCore{pipe.pipe, pipe.pipe, pipe.core, pipe.aivCore});
        ps.oriOpList_.push_back(&IRBuilder().CreateTensorOpStmt(function, Opcode::OP_VEC_DUP, {}, {}));
    }
    for (size_t i = 0; i < 8; ++i) {
        ps.depOps_[i * 2].setPipe = {i * 2 + 1};
        ps.depOps_[i * 2 + 1].waitPipe = {i * 2};
    }
    const auto original = ps.oriOpList_;
    std::vector<IndexOp> synced;
    for (auto& op : ps.depOps_) {
        ps.EnqueueOp(op, original, synced);
    }
    ASSERT_EQ(ps.IssueOp(function, original, synced), SUCCESS);
    EXPECT_TRUE(ps.reclaimBarriers_.empty());
    std::sort(synced.begin(), synced.end(), [](const IndexOp& a, const IndexOp& b) { return a.first < b.first; });
    std::vector<Operation*> ordered;
    std::vector<Operation*> flags;
    size_t forwardSets = 0;
    size_t returnSets = 0;
    for (const auto& entry : synced) {
        auto* op = &entry.second.get();
        ordered.push_back(op);
        EXPECT_NE(op->GetOpcode(), Opcode::OP_BAR_ALL);
        if (op->GetOpcode() == Opcode::OP_SYNC_SRC || op->GetOpcode() == Opcode::OP_SYNC_DST) {
            flags.push_back(op);
        }
        if (op->GetOpcode() == Opcode::OP_SYNC_SRC) {
            forwardSets += op->syncQueue_.pipeId_ == PIPE_MTE2 ? 1 : 0;
            returnSets += op->syncQueue_.pipeId_ == PIPE_V ? 1 : 0;
        }
    }
    EXPECT_EQ(forwardSets, 8U);
    EXPECT_EQ(returnSets, 1U);
    EXPECT_EQ(ps.ValidateEventReuse(ordered), SUCCESS);
    EXPECT_TRUE(AllIntraEventSchedulesAreSafe(flags));
}

TEST_F(InsertSyncTest, ConsumedPoolReplaysMergeBeforeDraining)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("ConsumedPoolReplaysMergeBeforeDraining");
    auto& function = *funcs.second;
    PipeSync ps;
    const PipeSync::PipePairEx pipes{{PIPE_MTE2, CoreType::AIV, AIVCore::AIV0},
                                     {PIPE_MTE3, CoreType::AIV, AIVCore::AIV0}};
    // Only the first two intervals overlap. Once all seven IDs have been
    // consumed, merging that prefix frees capacity without a full-core drain.
    for (size_t i = 0; i < 16; ++i) {
        const auto& pipe = (i < 2 || (i >= 4 && i % 2 == 0)) ? pipes.first : pipes.second;
        ps.depOps_.emplace_back(i, PipeSync::PipeCore{pipe.pipe, pipe.pipe, pipe.core, pipe.aivCore});
        ps.oriOpList_.push_back(&IRBuilder().CreateTensorOpStmt(function, Opcode::OP_VEC_DUP, {}, {}));
    }
    for (size_t i = 0; i < 8; ++i) {
        const size_t set = i < 2 ? i : i * 2;
        const size_t wait = i < 2 ? i + 2 : i * 2 + 1;
        ps.depOps_[set].setPipe = {wait};
        ps.depOps_[wait].waitPipe = {set};
    }
    const auto original = ps.oriOpList_;
    std::vector<IndexOp> synced;
    for (auto& op : ps.depOps_) {
        ps.EnqueueOp(op, original, synced);
    }
    ASSERT_EQ(ps.IssueOp(function, original, synced), SUCCESS);
    EXPECT_TRUE(ps.reclaimBarriers_.empty());
    std::sort(synced.begin(), synced.end(), [](const IndexOp& a, const IndexOp& b) { return a.first < b.first; });
    std::vector<Operation*> ordered;
    std::vector<Operation*> flags;
    for (const auto& entry : synced) {
        auto* op = &entry.second.get();
        ordered.push_back(op);
        EXPECT_NE(op->GetOpcode(), Opcode::OP_BAR_ALL);
        if (op->GetOpcode() == Opcode::OP_SYNC_SRC || op->GetOpcode() == Opcode::OP_SYNC_DST) {
            flags.push_back(op);
        }
    }
    EXPECT_EQ(flags.size(), 14U);
    EXPECT_EQ(ps.ValidateEventReuse(ordered), SUCCESS);
    EXPECT_TRUE(AllIntraEventSchedulesAreSafe(flags));
}

TEST_F(InsertSyncTest, ConsumedPoolAdvancesReturningWaitBeforeDraining)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("ConsumedPoolAdvancesReturningWaitBeforeDraining");
    auto& function = *funcs.second;
    PipeSync ps;
    const PipeSync::PipePairEx pipes{{PIPE_MTE2, CoreType::AIV, AIVCore::AIV0},
                                     {PIPE_MTE3, CoreType::AIV, AIVCore::AIV0}};
    for (size_t i = 0; i < 18; ++i) {
        const auto& pipe = i % 2 == 0 ? pipes.first : pipes.second;
        ps.depOps_.emplace_back(i, PipeSync::PipeCore{pipe.pipe, pipe.pipe, pipe.core, pipe.aivCore});
        ps.oriOpList_.push_back(&IRBuilder().CreateTensorOpStmt(function, Opcode::OP_VEC_DUP, {}, {}));
    }
    for (size_t i = 0; i < 8; ++i) {
        ps.depOps_[i * 2].setPipe = {i * 2 + 1};
        ps.depOps_[i * 2 + 1].waitPipe = {i * 2};
    }
    // The first consumed slot has a return path, but its wait is after the
    // eighth producer. Waiting earlier on that same pipe avoids a full drain.
    ps.depOps_[1].setPipe = {16};
    ps.depOps_[16].waitPipe = {1};
    const auto original = ps.oriOpList_;
    std::vector<IndexOp> synced;
    for (auto& op : ps.depOps_) {
        ps.EnqueueOp(op, original, synced);
    }
    ASSERT_EQ(ps.IssueOp(function, original, synced), SUCCESS);
    EXPECT_TRUE(ps.reclaimBarriers_.empty());
    EXPECT_NE(ps.eventState_.Find(1, 14), nullptr);
    EXPECT_EQ(ps.eventState_.Find(1, 16), nullptr);
    std::sort(synced.begin(), synced.end(), [](const IndexOp& a, const IndexOp& b) { return a.first < b.first; });
    std::vector<Operation*> ordered;
    std::vector<Operation*> flags;
    for (const auto& entry : synced) {
        auto* op = &entry.second.get();
        ordered.push_back(op);
        EXPECT_NE(op->GetOpcode(), Opcode::OP_BAR_ALL);
        if (op->GetOpcode() == Opcode::OP_SYNC_SRC || op->GetOpcode() == Opcode::OP_SYNC_DST) {
            flags.push_back(op);
        }
    }
    EXPECT_EQ(ps.ValidateEventReuse(ordered), SUCCESS);
    EXPECT_TRUE(AllIntraEventSchedulesAreSafe(flags));
}

TEST_F(InsertSyncTest, ConsumedPoolReturnsProofBeforeMergingIssuedDependencies)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("ConsumedPoolReturnsProofBeforeMergingIssuedDependencies");
    auto& function = *funcs.second;
    PipeSync ps;
    const PipeSync::PipePairEx pipes{{PIPE_MTE2, CoreType::AIV, AIVCore::AIV0},
                                     {PIPE_MTE3, CoreType::AIV, AIVCore::AIV0}};
    for (size_t i = 0; i < 20; ++i) {
        const auto& pipe = i % 2 == 0 ? pipes.first : pipes.second;
        ps.depOps_.emplace_back(i, PipeSync::PipeCore{pipe.pipe, pipe.pipe, pipe.core, pipe.aivCore});
        ps.oriOpList_.push_back(&IRBuilder().CreateTensorOpStmt(function, Opcode::OP_VEC_DUP, {}, {}));
    }
    const std::vector<std::pair<size_t, size_t>> dependencies{{0, 3},   {2, 5},   {6, 7},   {8, 9}, {10, 11},
                                                              {12, 13}, {14, 15}, {16, 17}, {3, 18}};
    for (const auto& [set, wait] : dependencies) {
        ps.depOps_[set].setPipe.push_back(wait);
        ps.depOps_[wait].waitPipe.push_back(set);
    }
    // Both replay merging and a returning proof can free the exhausted pool.
    // Preserve issued dependencies by consuming the existing return first.
    const auto original = ps.oriOpList_;
    std::vector<IndexOp> synced;
    for (auto& op : ps.depOps_) {
        ps.EnqueueOp(op, original, synced);
    }
    ASSERT_EQ(ps.IssueOp(function, original, synced), SUCCESS);
    EXPECT_TRUE(ps.reclaimBarriers_.empty());
    EXPECT_NE(ps.eventState_.Find(0, 3), nullptr);
    EXPECT_NE(ps.eventState_.Find(2, 5), nullptr);
    EXPECT_NE(ps.eventState_.Find(3, 16), nullptr);
    std::sort(synced.begin(), synced.end(), [](const IndexOp& a, const IndexOp& b) { return a.first < b.first; });
    std::vector<Operation*> ordered;
    std::vector<Operation*> flags;
    for (const auto& entry : synced) {
        auto* op = &entry.second.get();
        ordered.push_back(op);
        if (op->GetOpcode() == Opcode::OP_SYNC_SRC || op->GetOpcode() == Opcode::OP_SYNC_DST) {
            flags.push_back(op);
        }
    }
    EXPECT_EQ(ps.ValidateEventReuse(ordered), SUCCESS);
    EXPECT_TRUE(AllIntraEventSchedulesAreSafe(flags));
}

TEST_F(InsertSyncTest, DirectedReturnRejectsCyclesAndMultiPipeConsumers)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("DirectedReturnRejectsCyclesAndMultiPipeConsumers");
    auto& function = *funcs.second;
    PipeSync ps;
    const PipeSync::PipePairEx pipes{{PIPE_MTE2, CoreType::AIV, AIVCore::AIV0}, {PIPE_V, CoreType::AIV, AIVCore::AIV0}};
    for (size_t i = 0; i < 3; ++i) {
        const auto& pipe = i == 1 ? pipes.second : pipes.first;
        ps.depOps_.emplace_back(i, PipeSync::PipeCore{pipe.pipe, pipe.pipe, pipe.core, pipe.aivCore});
    }
    auto& set = IRBuilder().CreateTensorOpStmt(function, Opcode::OP_SYNC_SRC, {}, {});
    auto& wait = IRBuilder().CreateTensorOpStmt(function, Opcode::OP_SYNC_DST, {}, {});
    const auto resource = ps.GetEventResource(pipes, 1);
    ASSERT_EQ(ps.eventState_.Allocate(resource, pipes, 0, 1), SUCCESS);
    ASSERT_EQ(ps.eventState_.RecordSet(resource, pipes, 0, 1, set, true), SUCCESS);
    ASSERT_EQ(ps.eventState_.Consume(0, 1, wait), SUCCESS);
    auto plan = ps.depOps_;
    plan[1].selfPipeCore.pipeEnd = PIPE_MTE3;
    std::vector<size_t> order;
    EXPECT_FALSE(ps.AddReturningDependencyForReplay(plan, {2, 3, pipes, 1, 0}, order));
    EXPECT_TRUE(plan[2].waitPipe.empty());
    plan[1].selfPipeCore.pipeEnd = PIPE_V;
    plan[2].setPipe = {1};
    plan[1].waitPipe = {2};
    EXPECT_FALSE(ps.AddReturningDependencyForReplay(plan, {2, 3, pipes, 1, 0}, order));
    EXPECT_TRUE(plan[2].waitPipe.empty());
    EXPECT_TRUE(plan[1].setPipe.empty());
}

TEST_F(InsertSyncTest, DirectedReturnAdvancesExistingWaitWithoutAddingFanout)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("DirectedReturnAdvancesExistingWaitWithoutAddingFanout");
    auto& function = *funcs.second;
    const PipeSync::PipePairEx pipes{{PIPE_MTE2, CoreType::AIV, AIVCore::AIV0}, {PIPE_V, CoreType::AIV, AIVCore::AIV0}};
    for (size_t blocked : {2U, 3U}) {
        PipeSync ps;
        for (size_t i = 0; i < 4; ++i) {
            const auto& pipe = i == 1 ? pipes.second : pipes.first;
            ps.depOps_.emplace_back(i, PipeSync::PipeCore{pipe.pipe, pipe.pipe, pipe.core, pipe.aivCore});
        }
        const size_t oldWait = blocked == 2 ? 3 : 2;
        ps.depOps_[1].setPipe = {oldWait};
        ps.depOps_[oldWait].waitPipe = {1};
        auto& set = IRBuilder().CreateTensorOpStmt(function, Opcode::OP_SYNC_SRC, {}, {});
        auto& wait = IRBuilder().CreateTensorOpStmt(function, Opcode::OP_SYNC_DST, {}, {});
        const auto resource = ps.GetEventResource(pipes, 1);
        ASSERT_EQ(ps.eventState_.Allocate(resource, pipes, 0, 1), SUCCESS);
        ASSERT_EQ(ps.eventState_.RecordSet(resource, pipes, 0, 1, set, true), SUCCESS);
        ASSERT_EQ(ps.eventState_.Consume(0, 1, wait), SUCCESS);
        auto plan = ps.depOps_;
        std::vector<size_t> order;
        EXPECT_EQ(ps.AddReturningDependencyForReplay(plan, {blocked, blocked + 1, pipes, 1, 0}, order),
                  blocked < oldWait);
        EXPECT_EQ(plan[1].setPipe, std::vector<size_t>({std::min(blocked, oldWait)}));
        if (blocked < oldWait) {
            EXPECT_EQ(plan[blocked].waitPipe, std::vector<size_t>({1}));
            EXPECT_TRUE(plan[oldWait].waitPipe.empty());
        } else {
            EXPECT_TRUE(plan[blocked].waitPipe.empty());
            EXPECT_EQ(plan[oldWait].waitPipe, std::vector<size_t>({1}));
        }
        EXPECT_EQ(ps.eventState_.Instances().size(), 1U);
        EXPECT_EQ(ps.depOps_[1].setPipe, std::vector<size_t>({oldWait}));
    }
}

TEST_F(InsertSyncTest, ReturningWaitRequiresProofFrozenAtItsSet)
{
    auto funcs = SetupDebugFunc("ReturningWaitRequiresProofFrozenAtItsSet");
    auto& function = *funcs.second;
    PipeSync ps;
    const PipeSync::PipePairEx pipes{{PIPE_MTE2, CoreType::AIV, AIVCore::AIV0},
                                     {PIPE_MTE3, CoreType::AIV, AIVCore::AIV0}};
    for (size_t i = 0; i < 6; ++i) {
        const auto& pipe = i % 2 == 0 ? pipes.first : pipes.second;
        ps.depOps_.emplace_back(i, PipeSync::PipeCore{pipe.pipe, pipe.pipe, pipe.core, pipe.aivCore});
    }
    ps.depOps_[1].setPipe = {4};
    ps.depOps_[4].waitPipe = {1};
    auto& set = IRBuilder().CreateTensorOpStmt(function, Opcode::OP_SYNC_SRC, {}, {});
    auto& wait = IRBuilder().CreateTensorOpStmt(function, Opcode::OP_SYNC_DST, {}, {});
    const auto resource = ps.GetEventResource(pipes, 1);
    ASSERT_EQ(ps.eventState_.Allocate(resource, pipes, 0, 1), SUCCESS);
    ASSERT_EQ(ps.eventState_.RecordSet(resource, pipes, 0, 1, set, true), SUCCESS);
    const PipeSync::PipePairEx reverse{pipes.second, pipes.first};
    const auto returning = ps.GetEventResource(reverse, 1);
    ASSERT_EQ(ps.eventState_.Allocate(returning, reverse, 1, 4), SUCCESS);
    ASSERT_EQ(ps.eventState_.RecordSet(returning, reverse, 1, 4, set, true), SUCCESS);
    // Knowledge learned after the returning set must not enter its snapshot.
    ASSERT_EQ(ps.eventState_.Consume(0, 1, wait), SUCCESS);
    auto plan = ps.depOps_;
    std::vector<size_t> order;
    EXPECT_FALSE(ps.AdvanceReturningWaitForReplay(plan, {2, 3, pipes, 1, 0}, order));
    EXPECT_EQ(plan[1].setPipe, std::vector<size_t>({4}));
    EXPECT_TRUE(plan[2].waitPipe.empty());
}

TEST_F(InsertSyncTest, OneWayPendingPipelineMergesDependenciesWhenSevenEventsExhausted)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("OneWayPendingPipelineMergesDependencies");
    std::vector<Operation*> synced;
    ASSERT_EQ(InsertOneWayPipeline(*funcs.second, 10, true, synced), SUCCESS);
    std::vector<Operation*> flags;
    for (auto* op : synced) {
        EXPECT_NE(op->GetOpcode(), Opcode::OP_FFTS_CROSS_CORE_SYNC);
        if (op->GetOpcode() == Opcode::OP_SYNC_SRC || op->GetOpcode() == Opcode::OP_SYNC_DST) {
            flags.push_back(op);
        }
    }
    ASSERT_FALSE(flags.empty());
    EXPECT_LE(flags.size(), 14U);
    EXPECT_TRUE(AllIntraEventSchedulesAreSafe(flags));
}

TEST_F(InsertSyncTest, EventTrackerRejectsCancellingAnotherGenerationSlot)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("EventTrackerRejectsCancellingAnotherGenerationSlot");
    PipeSync ps;
    const PipeSync::PipePairEx pipes{{PIPE_MTE2, CoreType::AIV, AIVCore::AIV0}, {PIPE_V, CoreType::AIV, AIVCore::AIV0}};
    const auto dep = AddEventTestDependency(ps, *funcs.second, pipes);
    std::vector<IndexOp> synced;
    ASSERT_EQ(ps.InjectSetFlag(*funcs.second, dep.first, synced), SUCCESS);
    const auto original = ps.eventState_.Instances().front();
    auto newer = original;
    ++newer.id.generation;
    ps.eventState_.eventInstances_.push_back(newer);
    ps.eventState_.activeInstance_[original.id.resource] = 1;
    EXPECT_EQ(ps.eventState_.CancelPending(dep.first, dep.second), FAILED);
    ASSERT_NE(ps.eventState_.Find(dep.first, dep.second), nullptr);
    EXPECT_EQ(ps.eventState_.Find(dep.first, dep.second)->state, PipeSync::InstanceState::Allocated);
    EXPECT_EQ(ps.eventState_.activeInstance_.at(original.id.resource), 1U);
    EXPECT_FALSE(ps.eventState_.CanAllocate(original.id.resource, {}));
}

TEST_F(InsertSyncTest, EventCapacityGateDefersWholeFanoutBeforeInjection)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    PipeSync ps;
    const PipeSync::PipeCoreRealEx source(PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCoreRealEx target(PIPE_V, CoreType::AIV, AIVCore::AIV0);
    ps.freeEventId_[{source, target}] = {1};
    ps.depOps_.emplace_back(0, PipeSync::PipeCore{PIPE_MTE2, PIPE_MTE2, source.core, source.aivCore});
    for (size_t i = 1; i <= 2; ++i) {
        ps.depOps_.emplace_back(i, PipeSync::PipeCore{PIPE_V, PIPE_V, target.core, target.aivCore});
        ps.depOps_[0].setPipe.push_back(i);
        ps.depOps_[i].waitPipe.push_back(0);
    }
    PipeSync::IssueQueue queue(source);
    queue.ops = {0};
    ps.orderedOpList_.push(0);
    bool deadlock = false;
    std::vector<size_t> popped;
    std::vector<IndexOp> synced;
    ASSERT_EQ(ps.PopFromQueue(queue, popped, deadlock, synced), SUCCESS);
    EXPECT_TRUE(popped.empty());
    EXPECT_FALSE(ps.depOps_[0].issued);
    EXPECT_EQ(queue.currOp, 0U);
    ASSERT_FALSE(ps.orderedOpList_.empty());
    EXPECT_EQ(ps.orderedOpList_.front(), 0U);
    EXPECT_TRUE(synced.empty());
    EXPECT_TRUE(ps.eventState_.Instances().empty());
    EXPECT_TRUE(ps.AnalyzeIntraEventCapacity(ps.depOps_[0]).has_value());
    ps.freeEventId_[{source, target}] = {1, 2};
    ASSERT_EQ(ps.PopFromQueue(queue, popped, deadlock, synced), SUCCESS);
    EXPECT_EQ(popped, std::vector<size_t>({0}));
    EXPECT_TRUE(ps.depOps_[0].issued);
    EXPECT_EQ(queue.currOp, 1U);
    EXPECT_FALSE(ps.AnalyzeIntraEventCapacity(ps.depOps_[0]).has_value());
}

TEST_F(InsertSyncTest, EventTrackerRejectsCancellingConsumedInstance)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("EventTrackerRejectsCancellingConsumedInstance");
    PipeSync ps;
    const PipeSync::PipePairEx pipes{{PIPE_MTE2, CoreType::AIV, AIVCore::AIV0}, {PIPE_V, CoreType::AIV, AIVCore::AIV0}};
    const auto dep = AddEventTestDependency(ps, *funcs.second, pipes);
    std::vector<IndexOp> synced;
    ASSERT_EQ(ps.InjectSetFlag(*funcs.second, dep.first, synced), SUCCESS);
    ASSERT_EQ(ps.InjectWaitFlag(*funcs.second, dep.second, synced), SUCCESS);
    EXPECT_EQ(ps.eventState_.CancelPending(dep.first, dep.second), FAILED);
    const auto* instance = ps.eventState_.Find(dep.first, dep.second);
    ASSERT_NE(instance, nullptr);
    EXPECT_EQ(instance->state, PipeSync::InstanceState::WaitInserted);
    EXPECT_TRUE(ps.eventState_.HasActive(instance->id.resource));
    EXPECT_EQ(ps.eventState_.Known(pipes.first).count(instance->id), 0U);
}

TEST_F(InsertSyncTest, InvalidWaitGroupDoesNotConsumeEarlierWait)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("InvalidWaitGroupDoesNotConsumeEarlierWait");
    auto& function = *funcs.second;
    PipeSync ps;
    const PipeSync::PipeCoreRealEx a(PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCoreRealEx b(PIPE_V, CoreType::AIV, AIVCore::AIV0);
    const auto first = AddEventTestDependency(ps, function, {a, b});
    const auto missing = AddEventTestDependency(ps, function, {a, b});
    ps.depOps_[first.second].waitPipe.push_back(missing.first);
    std::vector<IndexOp> synced;
    ASSERT_EQ(ps.InjectSetFlag(function, first.first, synced), SUCCESS);
    EXPECT_EQ(ps.InjectWaitFlag(function, first.second, synced), FAILED);
    EXPECT_EQ(ps.eventState_.Instances()[0].state, PipeSync::InstanceState::Allocated);
    EXPECT_TRUE(ps.eventState_.Known(b).empty());
    EXPECT_EQ(synced.size(), 1U);
    EXPECT_TRUE(ps.waitOpMap.empty());
}

TEST_F(InsertSyncTest, EventTrackerFiltersRetiredProofFromDelayedSnapshot)
{
    auto funcs = SetupDebugFunc("EventTrackerFiltersRetiredProofFromDelayedSnapshot");
    PipeSync ps;
    auto& tracker = ps.eventState_;
    const PipeSync::PipeCoreRealEx a(PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCoreRealEx b(PIPE_V, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCoreRealEx c(PIPE_MTE3, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipePairEx ab{a, b}, ba{b, a}, bc{b, c};
    auto& function = *funcs.second;
    auto makeOp = [&]() -> Operation& { return IRBuilder().CreateTensorOpStmt(function, Opcode::OP_VEC_DUP, {}, {}); };
    auto emit = [&](const PipeSync::PipePairEx& pair, size_t setIdx, size_t waitIdx) -> Status {
        const auto resource = ps.GetEventResource(pair, 1);
        if (tracker.Allocate(resource, pair, setIdx, waitIdx) != SUCCESS) {
            return FAILED;
        }
        return tracker.RecordSet(resource, pair, setIdx, waitIdx, makeOp(), true);
    };
    ASSERT_EQ(emit(ab, 0, 1), SUCCESS);
    ASSERT_EQ(tracker.Consume(0, 1, makeOp()), SUCCESS);
    const auto old = tracker.Find(0, 1)->id;
    ASSERT_EQ(emit(bc, 2, 7), SUCCESS); // This snapshot contains the first AB generation.
    ASSERT_EQ(emit(ba, 3, 4), SUCCESS);
    ASSERT_EQ(tracker.Consume(3, 4, makeOp()), SUCCESS);
    ASSERT_EQ(emit(ab, 5, 6), SUCCESS); // Retire the old generation before BC's wait consumes its snapshot.
    ASSERT_EQ(tracker.Consume(2, 7, makeOp()), SUCCESS);
    EXPECT_EQ(tracker.Known(c).count(old), 0U);
    EXPECT_FALSE(tracker.CanAllocate(ps.GetEventResource(ab, 1), tracker.Known(c)));
    const auto* current = tracker.Find(5, 6);
    ASSERT_NE(current, nullptr);
    EXPECT_EQ(current->id.generation, 2U);
    EXPECT_EQ(current->state, PipeSync::InstanceState::Allocated);
}

TEST_F(InsertSyncTest, CrossAivWaitFailsBeforeConsumingAnyInstance)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("CrossAivWaitFailsBeforeConsumingAnyInstance");
    PipeSync ps;
    const PipeSync::PipePairEx pipes{{PIPE_MTE2, CoreType::AIV, AIVCore::AIV0}, {PIPE_V, CoreType::AIV, AIVCore::AIV1}};
    const auto dep = AddEventTestDependency(ps, *funcs.second, pipes);
    const auto failure = ps.AnalyzeIntraEventCapacity(ps.depOps_[dep.first]);
    ASSERT_TRUE(failure.has_value());
    EXPECT_EQ(failure->pipes, pipes);
    EXPECT_EQ(failure->setIdx, dep.first);
    EXPECT_EQ(failure->waitIdx, dep.second);
    std::vector<IndexOp> synced;
    EXPECT_EQ(ps.InjectWaitFlag(*funcs.second, dep.second, synced), FAILED);
    EXPECT_TRUE(synced.empty());
    EXPECT_TRUE(ps.eventState_.Instances().empty());
}

TEST_F(InsertSyncTest, IntraCoreFailedYieldRestoresAllDependencies)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    PipeSync ps;
    const PipeSync::PipeCoreRealEx a(PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCoreRealEx b(PIPE_V, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCoreRealEx c(PIPE_MTE3, CoreType::AIV, AIVCore::AIV0);
    for (size_t i = 0; i < 7; ++i) {
        const auto pipe = i == 2 ? c : (i >= 5 ? b : a);
        ps.depOps_.emplace_back(i, PipeSync::PipeCore{pipe.pipe, pipe.pipe, pipe.core, pipe.aivCore});
    }
    ps.depOps_[0].setPipe = {5, 2};
    ps.depOps_[5].waitPipe = {0};
    ps.depOps_[2].waitPipe = {0};
    ps.depOps_[3].setPipe = {6};
    ps.depOps_[6].waitPipe = {3};
    ps.freeEventId_[{a, b}] = {};
    ps.freeEventId_[{a, c}] = {};
    PipeSync::IssueQueue queue(a);
    queue.ops = {0, 3};
    PipeSync::IssueNum issueNum;
    std::vector<IndexOp> synced;
    bool deadlock = true;
    bool canIssue = true;
    ASSERT_EQ(ps.HandleEventID(ps.depOps_[0], queue, issueNum, deadlock, canIssue, synced), SUCCESS);
    EXPECT_FALSE(canIssue);
    EXPECT_TRUE(deadlock);
    EXPECT_EQ(ps.depOps_[0].setPipe, std::vector<size_t>({5, 2}));
    EXPECT_EQ(ps.depOps_[3].setPipe, std::vector<size_t>({6}));
    EXPECT_EQ(ps.depOps_[6].waitPipe, std::vector<size_t>({3}));
    EXPECT_EQ(ps.depOps_[5].waitPipe, std::vector<size_t>({0}));
    EXPECT_EQ(ps.depOps_[2].waitPipe, std::vector<size_t>({0}));
    EXPECT_TRUE(ps.eventState_.Instances().empty());
    EXPECT_TRUE(synced.empty());
}

TEST_F(InsertSyncTest, IntraCoreMergeCancelsOnlyPendingInstanceAndRebindsWait)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("IntraCoreMergeCancelsOnlyPendingInstanceAndRebindsWait");
    auto& function = *funcs.second;
    PipeSync ps;
    const PipeSync::PipeCoreRealEx a(PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCoreRealEx b(PIPE_V, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipePairEx pipes{a, b};
    ps.freeEventId_[pipes] = {1, 2};
    for (size_t i = 0; i < 5; ++i) {
        const auto pipe = i < 2 ? a : b;
        ps.depOps_.emplace_back(i, PipeSync::PipeCore{pipe.pipe, pipe.pipe, pipe.core, pipe.aivCore});
        ps.oriOpList_.push_back(&IRBuilder().CreateTensorOpStmt(function, Opcode::OP_VEC_DUP, {}, {}));
    }
    ASSERT_EQ(ps.AddOpDep(ps.depOps_[0], ps.depOps_[3]), SUCCESS);
    ASSERT_EQ(ps.AddOpDep(ps.depOps_[1], ps.depOps_[4]), SUCCESS);
    std::vector<IndexOp> synced;
    ASSERT_EQ(ps.InjectSetFlag(function, 0, synced), SUCCESS);
    ASSERT_EQ(ps.InjectSetFlag(function, 1, synced), SUCCESS);
    auto* cancelledSet = ps.eventState_.Instances()[0].setOp;
    auto* retainedSet = ps.eventState_.Instances()[1].setOp;
    PipeSync::DataDepInfo depInfo;
    ASSERT_EQ(ps.GetDepInfo(synced, pipes, depInfo), SUCCESS);
    ASSERT_EQ(depInfo.opDepList.size(), 2U);
    ASSERT_EQ(ps.SynDependency(0, depInfo, pipes, synced), SUCCESS);
    EXPECT_TRUE(cancelledSet->IsDeleted());
    EXPECT_EQ(ps.eventState_.Instances()[0].state, PipeSync::InstanceState::Cancelled);
    EXPECT_EQ(ps.eventState_.Instances()[1].waitIdx, 3U);
    EXPECT_EQ(ps.setOpMap.at(retainedSet), ps.oriOpList_[3]);
    EXPECT_EQ(ps.setWaitPairMap_.count({0, 3}), 0U);
    EXPECT_EQ(ps.setWaitPairMap_.count({1, 4}), 0U);
    ASSERT_EQ(ps.InjectWaitFlag(function, 3, synced), SUCCESS);
    EXPECT_EQ(ps.eventState_.Known(b).count(ps.eventState_.Instances()[0].id), 0U);
    EXPECT_EQ(ps.eventState_.Known(b).count(ps.eventState_.Instances()[1].id), 1U);
    EXPECT_EQ(ps.GetAvailableIntraEventIds(pipes, ps.eventState_.Known(a)), std::vector<int>({1}));
    ASSERT_EQ(ps.GetDepInfo(synced, pipes, depInfo), SUCCESS);
    EXPECT_TRUE(depInfo.opDepList.empty());
    std::vector<Operation*> ordered;
    for (const auto& op : synced) {
        ordered.push_back(&op.second.get());
    }
    EXPECT_EQ(ps.ValidateEventReuse(ordered), SUCCESS);
    EXPECT_TRUE(AllIntraEventSchedulesAreSafe(ordered));
}

TEST_F(InsertSyncTest, IntraCoreProofReturnsAcrossThreePipes)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("IntraCoreProofReturnsAcrossThreePipes");
    auto& function = *funcs.second;
    PipeSync ps;
    const PipeSync::PipeCoreRealEx a(PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCoreRealEx b(PIPE_V, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCoreRealEx c(PIPE_MTE3, CoreType::AIV, AIVCore::AIV0);
    ps.freeEventId_[{a, b}] = {1};
    const std::vector<PipeSync::PipePairEx> path{{a, b}, {b, c}, {c, a}, {a, b}};
    std::vector<IndexOp> synced;
    for (const auto& pipes : path) {
        const auto dep = AddEventTestDependency(ps, function, pipes);
        ASSERT_EQ(ps.InjectSetFlag(function, dep.first, synced), SUCCESS);
        ASSERT_EQ(ps.InjectWaitFlag(function, dep.second, synced), SUCCESS);
    }
    EXPECT_EQ(ps.eventState_.Instances().back().id.generation, 2U);
    std::vector<Operation*> ordered;
    for (const auto& op : synced) {
        ordered.push_back(&op.second.get());
    }
    EXPECT_EQ(ps.ValidateEventReuse(ordered), SUCCESS);
    EXPECT_TRUE(AllIntraEventSchedulesAreSafe(ordered));
}

TEST_F(InsertSyncTest, CrossCoreSyncCarriesIntraProofWithoutReclaimingCrossCoreId)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("CrossCoreSyncCarriesIntraProofWithoutReclaimingCrossCoreId");
    auto& function = *funcs.second;
    PipeSync ps;
    ps.InitCVEventIdQ();
    const PipeSync::PipeCoreRealEx a(PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCoreRealEx b(PIPE_V, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCoreRealEx c(PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED);
    ps.freeEventId_[{a, b}] = {1};
    std::vector<IndexOp> synced;
    for (const auto& pipes : std::vector<PipeSync::PipePairEx>{{a, b}, {b, c}, {c, a}, {a, b}}) {
        const auto dep = AddEventTestDependency(ps, function, pipes);
        ASSERT_EQ(ps.InjectSetFlag(function, dep.first, synced), SUCCESS);
        ASSERT_EQ(ps.InjectWaitFlag(function, dep.second, synced), SUCCESS);
    }
    EXPECT_EQ(ps.eventState_.Instances().back().id.generation, 2U);
    EXPECT_EQ(ps.GetCrossCoreEventIdQPtr({b, c})->size(), 14U);
    EXPECT_TRUE(ps.forceSyncOps_.empty());
    // A5 lowers the wait's source field to PIPE_V; its canonical source must
    // remain FIX for proof propagation and graph validation.
    EXPECT_EQ(ps.eventState_.Instances()[2].pipes.first, c);
    EXPECT_EQ(ps.eventState_.Instances()[2].waitOp->syncQueue_.pipeId_, PIPE_V);
    std::vector<Operation*> ordered;
    for (const auto& op : synced) {
        ordered.push_back(&op.second.get());
    }
    EXPECT_EQ(ps.ValidateEventReuse(ordered), SUCCESS);
}

TEST_F(InsertSyncTest, MultiPipeOpDoesNotInventConsumptionEdge)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("MultiPipeOpDoesNotInventConsumptionEdge");
    auto& function = *funcs.second;
    PipeSync ps;
    const PipeSync::PipeCoreRealEx a(PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCoreRealEx b(PIPE_V, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCoreRealEx c(PIPE_MTE3, CoreType::AIV, AIVCore::AIV0);
    const auto forward = AddEventTestDependency(ps, function, {a, b});
    const auto back = AddEventTestDependency(ps, function, {b, a});
    std::vector<IndexOp> synced;
    ASSERT_EQ(ps.InjectSetFlag(function, forward.first, synced), SUCCESS);
    ASSERT_EQ(ps.InjectWaitFlag(function, forward.second, synced), SUCCESS);
    ASSERT_EQ(ps.InjectSetFlag(function, back.first, synced), SUCCESS);
    ps.depOps_[back.second].selfPipeCore.pipeEnd = PIPE_MTE3;
    const auto oldId = ps.eventState_.Instances()[0].id;
    EXPECT_EQ(ps.ProjectConsumedInstances(ps.depOps_[back.second], a).count(oldId), 1U);
    EXPECT_EQ(ps.ProjectConsumedInstances(ps.depOps_[back.second], c).count(oldId), 0U);
    ASSERT_EQ(ps.InjectWaitFlag(function, back.second, synced), SUCCESS);
    EXPECT_EQ(ps.eventState_.Known(c).count(oldId), 0U);
}

TEST_F(InsertSyncTest, EventGenerationOverflowDoesNotAllocate)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("EventGenerationOverflowDoesNotAllocate");
    PipeSync ps;
    const PipeSync::PipePairEx pipes{{PIPE_MTE2, CoreType::AIV, AIVCore::AIV0}, {PIPE_V, CoreType::AIV, AIVCore::AIV0}};
    ps.freeEventId_[pipes] = {1};
    ps.eventState_.eventGeneration_[ps.GetEventResource(pipes, 1)] = std::numeric_limits<uint64_t>::max();
    std::vector<IndexOp> synced;
    int eventId = -1;
    size_t waitIdx = 1;
    EXPECT_EQ(ps.GetEventId(pipes, eventId, 0, waitIdx, synced, *funcs.second), FAILED);
    EXPECT_EQ(eventId, -1);
    EXPECT_EQ(ps.freeEventId_[pipes], std::deque<int>({1}));
    EXPECT_FALSE(ps.eventState_.HasActive(ps.GetEventResource(pipes, 1)));
    EXPECT_TRUE(ps.eventState_.Instances().empty());
}

TEST_F(InsertSyncTest, IntraCoreReusePreservesWaitQueueOrder)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("IntraCoreReusePreservesWaitQueueOrder");
    auto& function = *funcs.second;
    PipeSync ps;
    const PipeSync::PipeCoreRealEx a(PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCoreRealEx b(PIPE_V, CoreType::AIV, AIVCore::AIV0);
    ps.freeEventId_[{a, b}] = {1, 2};
    const auto first = AddEventTestDependency(ps, function, {a, b});
    const auto second = AddEventTestDependency(ps, function, {a, b});
    const auto back = AddEventTestDependency(ps, function, {b, a});
    std::vector<IndexOp> synced;
    ASSERT_EQ(ps.InjectSetFlag(function, first.first, synced), SUCCESS);
    ASSERT_EQ(ps.InjectSetFlag(function, second.first, synced), SUCCESS);
    ASSERT_EQ(ps.InjectWaitFlag(function, second.second, synced), SUCCESS);
    ASSERT_EQ(ps.InjectWaitFlag(function, first.second, synced), SUCCESS);
    EXPECT_TRUE(ps.GetAvailableIntraEventIds({a, b}, ps.eventState_.Known(a)).empty());
    ASSERT_EQ(ps.InjectSetFlag(function, back.first, synced), SUCCESS);
    ASSERT_EQ(ps.InjectWaitFlag(function, back.second, synced), SUCCESS);
    EXPECT_EQ(ps.GetAvailableIntraEventIds({a, b}, ps.eventState_.Known(a)), std::vector<int>({2, 1}));
}

TEST_F(InsertSyncTest, VFTuningPreservesEventReuseProofOrder)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("VFTuningPreservesEventReuseProofOrder");
    auto& function = *funcs.second;
    PipeSync ps;
    const PipeSync::PipeCoreRealEx a(PIPE_V, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCoreRealEx b(PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    ps.freeEventId_[{a, b}] = {1};
    std::vector<IndexOp> synced;
    for (const auto& pipes : std::vector<PipeSync::PipePairEx>{{a, b}, {b, a}, {a, b}}) {
        const auto dep = AddEventTestDependency(ps, function, pipes);
        ASSERT_EQ(ps.InjectSetFlag(function, dep.first, synced), SUCCESS);
        ASSERT_EQ(ps.InjectWaitFlag(function, dep.second, synced), SUCCESS);
    }
    std::vector<Operation*> ordered;
    for (const auto& op : synced) {
        ordered.push_back(&op.second.get());
    }
    ASSERT_EQ(ps.ValidateEventReuse(ordered), SUCCESS);
    auto& first = IRBuilder().CreateTensorOpStmt(function, Opcode::OP_VEC_DUP, {}, {});
    auto& second = IRBuilder().CreateTensorOpStmt(function, Opcode::OP_VEC_DUP, {}, {});
    TuneSyncForVF tune;
    tune.opList_ = {ordered[0], ordered[1], ordered[2], &first, ordered[3], ordered[4], &second, ordered[5]};
    tune.mergedOps = {{&first, &second}};
    std::vector<Operation*> sets;
    std::vector<Operation*> waits;
    ASSERT_TRUE(tune.IsMergeable(3, 6, sets, waits));
    EXPECT_EQ(tune.MoveOpsForMerge(3, 6, 0, sets, waits), 2UL);
    EXPECT_EQ(ps.ValidateEventReuse(tune.opList_), SUCCESS);
}

// Helper: batch-create N FP32 tensors with given shape
static std::vector<std::shared_ptr<LogicalTensor>> MakeTensors(int n, const std::vector<int64_t>& shape)
{
    std::vector<std::shared_ptr<LogicalTensor>> ts;
    for (int i = 0; i < n; i++) {
        ts.push_back(npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape)));
    }
    return ts;
}

// Helper: verify one sync op's opcode and AIVCore
static void VerifySyncOp(const Operation* op, Opcode expectedOp, AIVCore expectedCore)
{
    EXPECT_EQ(op->GetOpcode(), expectedOp);
    EXPECT_EQ(op->GetAIVCore(), expectedCore);
}

// Helper: verify BAR_ALL inserted between same-coreType ops
static void VerifyBarAll(const std::vector<Operation*>& ops, size_t idx)
{
    VerifySyncOp(ops[idx], Opcode::OP_BAR_ALL, AIVCore::UNSPECIFIED);
}

// 平台架构切换统一使用公共工具 NpuArchScope(见 pass_test_utils.h)

static Operation& CreateAtomicScopeSyncOp(Function& func, Opcode opcode)
{
    std::vector<std::shared_ptr<LogicalTensor>> input;
    std::vector<std::shared_ptr<LogicalTensor>> output;
    Operation& syncOp = IRBuilder().CreateTensorOpStmt(func, opcode, input, output);
    syncOp.syncQueue_ = {PipeType::PIPE_V, PipeType::PIPE_V, CoreType::AIV, CoreType::AIV,
                         IS_NUM1,          AIVCore::AIV0,    AIVCore::AIV0};
    syncOp.SetAIVCore(AIVCore::AIV0);
    return syncOp;
}

static void ExpectOpListEq(const std::vector<Operation*>& actual, const std::vector<Operation*>& expected)
{
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(actual[i], expected[i]) << "Mismatch at index " << i;
    }
}

// Helper: verify full InsertCvSyncOps 12-op sequence, return next index
static size_t VerifyCvSyncOps(const std::vector<Operation*>& ops, size_t i)
{
    // UNSPECIFIED group (ops 1-4)
    VerifySyncOp(ops[i], Opcode::OP_BAR_ALL, AIVCore::UNSPECIFIED);
    VerifySyncOp(ops[i + IS_NUM1], Opcode::OP_FFTS_CROSS_CORE_SYNC, AIVCore::UNSPECIFIED);
    VerifySyncOp(ops[i + IS_NUM2], Opcode::OP_WAIT_FLAG_DEV, AIVCore::UNSPECIFIED);
    VerifySyncOp(ops[i + IS_NUM3], Opcode::OP_BAR_ALL, AIVCore::UNSPECIFIED);
    // AIV0 group (ops 5-8)
    VerifySyncOp(ops[i + IS_NUM4], Opcode::OP_BAR_ALL, AIVCore::AIV0);
    VerifySyncOp(ops[i + IS_NUM5], Opcode::OP_WAIT_FLAG_DEV, AIVCore::AIV0);
    VerifySyncOp(ops[i + IS_NUM6], Opcode::OP_FFTS_CROSS_CORE_SYNC, AIVCore::AIV0);
    VerifySyncOp(ops[i + IS_NUM7], Opcode::OP_BAR_ALL, AIVCore::AIV0);
    // AIV1 group (ops 9-12)
    VerifySyncOp(ops[i + IS_NUM8], Opcode::OP_BAR_ALL, AIVCore::AIV1);
    VerifySyncOp(ops[i + IS_NUM9], Opcode::OP_WAIT_FLAG_DEV, AIVCore::AIV1);
    VerifySyncOp(ops[i + IS_NUM10], Opcode::OP_FFTS_CROSS_CORE_SYNC, AIVCore::AIV1);
    VerifySyncOp(ops[i + IS_NUM11], Opcode::OP_BAR_ALL, AIVCore::AIV1);
    return i + IS_NUM12;
}

// 覆盖 InsertCvPipeAll 全部 3 条分支：
//   ADD(AIV)→CAST(AIV): 同coreType → BAR_ALL
//   CAST(AIV)→A_MUL_B(AIC): 异coreType → InsertCvSyncOps
//   A_MUL_B(AIC)→EXP(AIV): 异coreType → InsertCvSyncOps
//   EXP 是最后一个 op → skip
TEST_F(InsertSyncTest, TestEnableDebugAllPaths)
{
    auto [root, leaf] = SetupDebugFunc("TestDbgAll");
    auto ts = MakeTensors(IS_NUM7, {IS_NUM8, IS_NUM16});
    (void)IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0], ts[1]}, {ts[2]});
    (void)IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CAST, {ts[2]}, {ts[3]});
    (void)IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_A_MUL_B, {ts[3], ts[4], ts[5]}, {ts[6]});
    (void)IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_EXP, {ts[6]}, {ts[0]});
    InsertSync syncPass;
    syncPass.SetEnableDebug(true);
    syncPass.RunOnFunction(*root);
    // 验证 oriOpList
    EXPECT_EQ(leaf->oriOpList.size(), static_cast<size_t>(IS_NUM4));
    // 验证新 op 列表: 4原始 + 1 BAR_ALL + 12 CvSync + 12 CvSync = 29
    auto ops = leaf->Operations(false).DuplicatedOpList();
    EXPECT_EQ(ops.size(), static_cast<size_t>(IS_NUM29));
    size_t idx = 0;
    EXPECT_EQ(ops[idx++]->GetOpcode(), Opcode::OP_ADD);
    VerifyBarAll(ops, idx++);
    EXPECT_EQ(ops[idx++]->GetOpcode(), Opcode::OP_CAST);
    idx = VerifyCvSyncOps(ops, idx);
    EXPECT_EQ(ops[idx++]->GetOpcode(), Opcode::OP_A_MUL_B);
    idx = VerifyCvSyncOps(ops, idx);
    EXPECT_EQ(ops[idx++]->GetOpcode(), Opcode::OP_EXP);
    EXPECT_EQ(idx, static_cast<size_t>(IS_NUM29));
}

// 边界: 单 op 时 i==oriOpList.size()-1 直接 skip，不插入任何 sync op
TEST_F(InsertSyncTest, TestEnableDebugSingleOp)
{
    auto [root, leaf] = SetupDebugFunc("TestDbgSingle");
    auto ts = MakeTensors(IS_NUM2, {IS_NUM8, IS_NUM16});
    (void)IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0]}, {ts[1]});
    InsertSync syncPass;
    syncPass.SetEnableDebug(true);
    syncPass.RunOnFunction(*root);
    EXPECT_EQ(leaf->oriOpList.size(), static_cast<size_t>(IS_NUM1));
    auto ops = leaf->Operations(false).DuplicatedOpList();
    EXPECT_EQ(ops.size(), static_cast<size_t>(IS_NUM1));
    EXPECT_EQ(ops[0]->GetOpcode(), Opcode::OP_ADD);
}

// 全同 coreType(AIV): 每对相邻 op 之间都插入 BAR_ALL
TEST_F(InsertSyncTest, TestEnableDebugAllSameCoreType)
{
    auto [root, leaf] = SetupDebugFunc("TestDbgSame");
    auto ts = MakeTensors(IS_NUM4, {IS_NUM8, IS_NUM16});
    (void)IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0], ts[1]}, {ts[2]});
    (void)IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CAST, {ts[2]}, {ts[3]});
    (void)IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_EXP, {ts[3]}, {ts[0]});
    InsertSync syncPass;
    syncPass.SetEnableDebug(true);
    syncPass.RunOnFunction(*root);
    EXPECT_EQ(leaf->oriOpList.size(), static_cast<size_t>(IS_NUM3));
    // 3 原始 + 2 BAR_ALL = 5
    auto ops = leaf->Operations(false).DuplicatedOpList();
    EXPECT_EQ(ops.size(), static_cast<size_t>(IS_NUM5));
    EXPECT_EQ(ops[0]->GetOpcode(), Opcode::OP_ADD);
    VerifyBarAll(ops, IS_NUM1);
    EXPECT_EQ(ops[IS_NUM2]->GetOpcode(), Opcode::OP_CAST);
    VerifyBarAll(ops, IS_NUM3);
    EXPECT_EQ(ops[IS_NUM4]->GetOpcode(), Opcode::OP_EXP);
}

// 按真实场景 mla_prolog_quant Unroll128 的 kr_cache 量化写回构造计算图
// (dump: op[38965/38968/38971/38974]):
//   incast b0/b1 --COPY_IN--> cin_i --ADD--> add_i(各路量化结果src)
//   incast kr_idx --COPY_IN--> idx_i(各路散写索引)
//   incast kr_cache 直连 INDEX_OUTCAST 第3输入, 4 路共享并共写 kr_cache_out(outcast, WAW)
// kr_cache_out 与 kr_cache 原地读改写(对照真实 dump: 3901/3943 的 mem_id 同为 3636)
static void BuildKrCacheWritebackGraph(ComputationalGraphBuilder& subGraph)
{
    std::vector<std::string> tensorNames{"b0",   "b1",   "cin0",   "cin1",     "cin2",        "cin3",
                                         "add0", "add1", "add2",   "add3",     "idx0",        "idx1",
                                         "idx2", "idx3", "kr_idx", "kr_cache", "kr_cache_out"};
    std::vector<MemoryType> memTypes{
        MemoryType::MEM_DEVICE_DDR, // b0 (incast)
        MemoryType::MEM_DEVICE_DDR, // b1 (incast)
        MemoryType::MEM_UB,         // cin0
        MemoryType::MEM_UB,         // cin1
        MemoryType::MEM_UB,         // cin2
        MemoryType::MEM_UB,         // cin3
        MemoryType::MEM_UB,         // add0
        MemoryType::MEM_UB,         // add1
        MemoryType::MEM_UB,         // add2
        MemoryType::MEM_UB,         // add3
        MemoryType::MEM_UB,         // idx0
        MemoryType::MEM_UB,         // idx1
        MemoryType::MEM_UB,         // idx2
        MemoryType::MEM_UB,         // idx3
        MemoryType::MEM_DEVICE_DDR, // kr_idx (incast)
        MemoryType::MEM_DEVICE_DDR, // kr_cache (incast)
        MemoryType::MEM_DEVICE_DDR, // kr_cache_out (outcast)
    };
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {256, 256}, memTypes, tensorNames, 0), true);
    std::vector<Opcode> opCodes{
        Opcode::OP_COPY_IN,       Opcode::OP_COPY_IN,       Opcode::OP_COPY_IN,       Opcode::OP_COPY_IN,
        Opcode::OP_ADD,           Opcode::OP_ADD,           Opcode::OP_ADD,           Opcode::OP_ADD,
        Opcode::OP_COPY_IN,       Opcode::OP_COPY_IN,       Opcode::OP_COPY_IN,       Opcode::OP_COPY_IN,
        Opcode::OP_INDEX_OUTCAST, Opcode::OP_INDEX_OUTCAST, Opcode::OP_INDEX_OUTCAST, Opcode::OP_INDEX_OUTCAST};
    std::vector<std::vector<std::string>> ioperands{{"b0"},
                                                    {"b1"},
                                                    {"b0"},
                                                    {"b1"},
                                                    {"cin0", "cin1"},
                                                    {"cin1", "cin2"},
                                                    {"cin2", "cin3"},
                                                    {"cin3", "cin0"},
                                                    {"kr_idx"},
                                                    {"kr_idx"},
                                                    {"kr_idx"},
                                                    {"kr_idx"},
                                                    {"add0", "idx0", "kr_cache"},
                                                    {"add1", "idx1", "kr_cache"},
                                                    {"add2", "idx2", "kr_cache"},
                                                    {"add3", "idx3", "kr_cache"}};
    std::vector<std::vector<std::string>> ooperands{
        {"cin0"}, {"cin1"}, {"cin2"}, {"cin3"}, {"add0"},         {"add1"},         {"add2"},         {"add3"},
        {"idx0"}, {"idx1"}, {"idx2"}, {"idx3"}, {"kr_cache_out"}, {"kr_cache_out"}, {"kr_cache_out"}, {"kr_cache_out"}};
    std::vector<std::string> opNames{"cpin0",    "cpin1",    "cpin2",    "cpin3",   "add0",  "add1",
                                     "add2",     "add3",     "cpin4",    "cpin5",   "cpin6", "cpin7",
                                     "outcast0", "outcast1", "outcast2", "outcast3"};
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    auto krCache = subGraph.GetTensor("kr_cache");
    auto krCacheOut = subGraph.GetTensor("kr_cache_out");
    ASSERT_NE(krCache, nullptr);
    ASSERT_NE(krCacheOut, nullptr);
    krCacheOut->memoryrange.memId = krCache->memoryrange.memId;
    EXPECT_EQ(subGraph.SetInCast({"b0", "b1", "kr_idx", "kr_cache"}), true);
    EXPECT_EQ(subGraph.SetOutCast({"kr_cache_out"}), true);
}

// 收集 opLogPtr 中全部 INDEX_OUTCAST 的索引
static std::vector<size_t> CollectOutcastIdxs(const std::vector<Operation*>& opLogPtr)
{
    std::vector<size_t> outcastIdxs;
    for (size_t j = 0; j < opLogPtr.size(); j++) {
        if (opLogPtr[j]->GetOpcode() == Opcode::OP_INDEX_OUTCAST) {
            outcastIdxs.push_back(j);
        }
    }
    return outcastIdxs;
}

// 禁止 AIV0↔AIV1 同步: 按 BuildKrCacheWritebackGraph 的 kr_cache 写回形态构造,
// 相邻路经 kr_cache(WAR, DDR同mem_id) 与 kr_cache_out(WAW) 建立数据依赖后, 被交替分到
// AIV0/AIV1 即触发 InjectWaitFlag 的 AIV0/AIV1 同步拦截, InjectSync 必须传播 FAILED 且不标记 issued
TEST_F(InsertSyncTest, TestInjectSyncFailOnAiv0Aiv1Sync)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestAiv0Aiv1SyncFail",
                                                  "TestAiv0Aiv1SyncFail", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestAiv0Aiv1SyncFailLeaf",
                                                      "TestAiv0Aiv1SyncFailLeaf", rootFuncPtr.get());
    ASSERT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());

    ComputationalGraphBuilder subGraph;
    BuildKrCacheWritebackGraph(subGraph);
    Function* function = subGraph.GetFunction();
    ASSERT_NE(function, nullptr);

    auto opLogPtr = function->Operations(false).DuplicatedOpList();
    ASSERT_EQ(opLogPtr.size(), static_cast<size_t>(IS_NUM16));
    // 模拟 OoOSchedule 对 4 路 INDEX_OUTCAST 的交替分核结果(AIV0/AIV0/AIV1/AIV1, 对照真实 dump)
    for (size_t i = 0; i < IS_NUM4; i++) {
        auto outcast = subGraph.GetOp("outcast" + std::to_string(i));
        ASSERT_NE(outcast, nullptr);
        outcast->SetAIVCore(i < IS_NUM2 ? AIVCore::AIV0 : AIVCore::AIV1);
    }

    PipeSync ps;
    DataDependencySearcher dataDependencySearcher;
    ProcessOpList(ps, dataDependencySearcher, opLogPtr);
    std::vector<IndexOp> synced;
    BuildDeps(ps, dataDependencySearcher, opLogPtr, synced);
    // 相邻路共写 kr_cache_out(WAW) 且经 kr_cache 构成 WAR(DDR同mem_id), 必须已建立依赖
    auto outcastIdxs = CollectOutcastIdxs(opLogPtr);
    ASSERT_EQ(outcastIdxs.size(), static_cast<size_t>(IS_NUM4));
    EXPECT_GT(ps.depOps_[outcastIdxs[1]].waitPipe.size(), static_cast<size_t>(0));
    // fixture 的 AdjustCopyOpTileCfg 将所有 op 的 aivCore 固定为 AIV0, 此处按模拟的 OoO 分核结果
    // 修正 4 路 INDEX_OUTCAST 在 depOps_ 中的 aivCore(AIV0/AIV0/AIV1/AIV1, 对照真实 dump)
    for (size_t i = 0; i < IS_NUM4; i++) {
        ps.depOps_[outcastIdxs[i]].selfPipeCore.aivCore = (i < IS_NUM2) ? AIVCore::AIV0 : AIVCore::AIV1;
    }
    // outcast2(AIV1) 与其等待的 outcast1(AIV0) 构成 AIV0↔AIV1 跨核依赖, 必须被拦截
    EXPECT_EQ(ps.InjectSync(*currFunctionPtr, opLogPtr, outcastIdxs[2], synced), FAILED);
    EXPECT_FALSE(ps.depOps_[outcastIdxs[2]].issued);
}

// 全同 coreType(AIC): AIC→AIC 同核心场景，BAR_ALL 的 AIVCore 取自 nextOp(AIC 算子)，其值为 UNSPECIFIED
TEST_F(InsertSyncTest, TestEnableDebugAllAic)
{
    auto [root, leaf] = SetupDebugFunc("TestDbgAic");
    auto ts = MakeTensors(IS_NUM10, {IS_NUM8, IS_NUM16});
    (void)IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_A_MUL_B, {ts[0], ts[1], ts[2]}, {ts[3]});
    (void)IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_A_MULACC_B, {ts[3], ts[4], ts[5]}, {ts[6]});
    (void)IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_A_MUL_BT, {ts[6], ts[7], ts[8]}, {ts[9]});
    InsertSync syncPass;
    syncPass.SetEnableDebug(true);
    syncPass.RunOnFunction(*root);
    EXPECT_EQ(leaf->oriOpList.size(), static_cast<size_t>(IS_NUM3));
    // 3 原始 + 2 BAR_ALL = 5
    auto ops = leaf->Operations(false).DuplicatedOpList();
    EXPECT_EQ(ops.size(), static_cast<size_t>(IS_NUM5));
    EXPECT_EQ(ops[0]->GetOpcode(), Opcode::OP_A_MUL_B);
    VerifyBarAll(ops, IS_NUM1);
    EXPECT_EQ(ops[IS_NUM2]->GetOpcode(), Opcode::OP_A_MULACC_B);
    VerifyBarAll(ops, IS_NUM3);
    EXPECT_EQ(ops[IS_NUM4]->GetOpcode(), Opcode::OP_A_MUL_BT);
}

// =========================================================================
// AdjustSyncByAtomicScope 测试用例
// 后处理逻辑：A5 平台上，cluster 内的 wait-like sync 移到 cluster 前，
// set-like sync 移到 cluster 后。非 A5 平台不做处理。
// =========================================================================

// 单条 wait-like 同步（OP_SYNC_DST）被移动到 cluster 前
TEST_F(InsertSyncTest, AdjustSyncByAtomicScopeMovesSingleWaitBeforeCluster)
{
    NpuArchScope scopedArch(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("TestMoveWaitBeforeCluster");
    (void)root;
    auto ts = MakeTensors(IS_NUM4, {IS_NUM8, IS_NUM16});
    auto& add = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0], ts[1]}, {ts[2]});
    auto& cast = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CAST, {ts[2]}, {ts[3]});
    auto& wait = CreateAtomicScopeSyncOp(*leaf, Opcode::OP_SYNC_DST);
    add.SetAtomicScopeId(VF_SCOPE_A);
    cast.SetAtomicScopeId(VF_SCOPE_A);
    add.SetAIVCore(AIVCore::AIV0);
    cast.SetAIVCore(AIVCore::AIV0);
    std::vector<Operation*> opList = {&add, &wait, &cast};

    InsertSync syncPass;
    EXPECT_EQ(syncPass.AdjustSyncByAtomicScope(opList, {}), SUCCESS);

    ExpectOpListEq(opList, {&wait, &add, &cast});
}

// 单条 set-like 同步（OP_SYNC_SRC）被移动到 cluster 后
TEST_F(InsertSyncTest, AdjustSyncByAtomicScopeMovesSingleSetAfterCluster)
{
    NpuArchScope scopedArch(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("TestMoveSetAfterCluster");
    (void)root;
    auto ts = MakeTensors(IS_NUM4, {IS_NUM8, IS_NUM16});
    auto& add = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0], ts[1]}, {ts[2]});
    auto& cast = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CAST, {ts[2]}, {ts[3]});
    auto& set = CreateAtomicScopeSyncOp(*leaf, Opcode::OP_SYNC_SRC);
    add.SetAtomicScopeId(VF_SCOPE_A);
    cast.SetAtomicScopeId(VF_SCOPE_A);
    add.SetAIVCore(AIVCore::AIV0);
    cast.SetAIVCore(AIVCore::AIV0);
    std::vector<Operation*> opList = {&add, &set, &cast};

    InsertSync syncPass;
    EXPECT_EQ(syncPass.AdjustSyncByAtomicScope(opList, {}), SUCCESS);

    ExpectOpListEq(opList, {&add, &cast, &set});
}

// OP_BAR_V 不可移动，出现在 cluster 内时失败
TEST_F(InsertSyncTest, AdjustSyncByAtomicScopeRejectsBarVInCluster)
{
    NpuArchScope scopedArch(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("TestRejectBarV");
    (void)root;
    auto ts = MakeTensors(IS_NUM4, {IS_NUM8, IS_NUM16});
    auto& add = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0], ts[1]}, {ts[2]});
    auto& cast = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CAST, {ts[2]}, {ts[3]});
    auto& barV = CreateAtomicScopeSyncOp(*leaf, Opcode::OP_BAR_V);
    add.SetAtomicScopeId(VF_SCOPE_A);
    cast.SetAtomicScopeId(VF_SCOPE_A);
    add.SetAIVCore(AIVCore::AIV0);
    cast.SetAIVCore(AIVCore::AIV0);
    std::vector<Operation*> opList = {&add, &barV, &cast};

    InsertSync syncPass;
    EXPECT_EQ(syncPass.AdjustSyncByAtomicScope(opList, {}), FAILED);
}

// CV eventId 耗尽兜底 force 序列（BAR_ALL 等）落在 cluster 内时，该 cluster 整体不再调整 op 顺序
TEST_F(InsertSyncTest, AdjustSyncByAtomicScopeSkipsForceSyncOps)
{
    NpuArchScope scopedArch(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("TestSkipForceSyncOps");
    (void)root;
    auto ts = MakeTensors(IS_NUM4, {IS_NUM8, IS_NUM16});
    auto& add = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0], ts[1]}, {ts[2]});
    auto& cast = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CAST, {ts[2]}, {ts[3]});
    // force 序列前的 set（先被收集，遇到 force op 后整体清空）与 force 序列后的 wait 均保持原位
    Operation& cvSet = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CV_SYNC_SRC, {}, {});
    cvSet.syncQueue_ = {PipeType::PIPE_M, PipeType::PIPE_V, CoreType::AIC, CoreType::AIV,
                        IS_NUM2,          AIVCore::AIV0,    AIVCore::AIV1};
    cvSet.SetAIVCore(AIVCore::AIV0);
    auto& barAll = CreateAtomicScopeSyncOp(*leaf, Opcode::OP_BAR_ALL);
    auto& wait = CreateAtomicScopeSyncOp(*leaf, Opcode::OP_SYNC_DST);
    add.SetAtomicScopeId(VF_SCOPE_A);
    cast.SetAtomicScopeId(VF_SCOPE_A);
    add.SetAIVCore(AIVCore::AIV0);
    cast.SetAIVCore(AIVCore::AIV0);
    std::vector<Operation*> opList = {&add, &cvSet, &barAll, &wait, &cast};

    InsertSync syncPass;
    std::unordered_set<Operation*> forceSyncOps = {&barAll};
    EXPECT_EQ(syncPass.AdjustSyncByAtomicScope(opList, forceSyncOps), SUCCESS);

    // cluster 整体不调整：所有 op 保持原位
    ExpectOpListEq(opList, {&add, &cvSet, &barAll, &wait, &cast});
}

// 非 force 序列的 immovable sync（BAR_ALL）落在同核 cluster 内时报错
TEST_F(InsertSyncTest, AdjustSyncByAtomicScopeRejectsBarAllInCluster)
{
    NpuArchScope scopedArch(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("TestRejectBarAll");
    (void)root;
    auto ts = MakeTensors(IS_NUM4, {IS_NUM8, IS_NUM16});
    auto& add = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0], ts[1]}, {ts[2]});
    auto& cast = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CAST, {ts[2]}, {ts[3]});
    auto& barAll = CreateAtomicScopeSyncOp(*leaf, Opcode::OP_BAR_ALL);
    add.SetAtomicScopeId(VF_SCOPE_A);
    cast.SetAtomicScopeId(VF_SCOPE_A);
    add.SetAIVCore(AIVCore::AIV0);
    cast.SetAIVCore(AIVCore::AIV0);
    std::vector<Operation*> opList = {&add, &barAll, &cast};

    InsertSync syncPass;
    EXPECT_EQ(syncPass.AdjustSyncByAtomicScope(opList, {}), FAILED);
}

// cluster 内连续 sync op（不配对）允许移动，各自移到对应边界
TEST_F(InsertSyncTest, AdjustSyncByAtomicScopeMovesConsecutiveUnpairedSyncOps)
{
    NpuArchScope scopedArch(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("TestConsecutiveSync");
    (void)root;
    auto ts = MakeTensors(IS_NUM5, {IS_NUM8, IS_NUM16});
    auto& add = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0], ts[1]}, {ts[2]});
    auto& midExp = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_EXP, {ts[2]}, {ts[3]});
    auto& cast = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CAST, {ts[3]}, {ts[4]});
    // wait 和 set 连续出现（中间无 cluster op 隔开），但 eventId 不同，不配对
    std::vector<std::shared_ptr<LogicalTensor>> emptyInput;
    std::vector<std::shared_ptr<LogicalTensor>> emptyOutput;
    Operation& wait = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_SYNC_DST, emptyInput, emptyOutput);
    wait.syncQueue_ = {PipeType::PIPE_MTE2, PipeType::PIPE_V, CoreType::AIV, CoreType::AIV, IS_NUM1,
                       AIVCore::AIV0,       AIVCore::AIV0};
    wait.SetAIVCore(AIVCore::AIV0);
    Operation& set = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_SYNC_SRC, emptyInput, emptyOutput);
    set.syncQueue_ = {PipeType::PIPE_V, PipeType::PIPE_MTE3, CoreType::AIV, CoreType::AIV,
                      IS_NUM2,          AIVCore::AIV0,       AIVCore::AIV0};
    set.SetAIVCore(AIVCore::AIV0);
    add.SetAtomicScopeId(VF_SCOPE_A);
    midExp.SetAtomicScopeId(VF_SCOPE_A);
    cast.SetAtomicScopeId(VF_SCOPE_A);
    add.SetAIVCore(AIVCore::AIV0);
    midExp.SetAIVCore(AIVCore::AIV0);
    cast.SetAIVCore(AIVCore::AIV0);
    // wait 和 set 连续，但 eventId 不同（1 vs 2），pipe 也不配对
    std::vector<Operation*> opList = {&add, &wait, &set, &midExp, &cast};

    // wait 移到 cluster 前，set 移到 cluster 后
    InsertSync syncPass;
    EXPECT_EQ(syncPass.AdjustSyncByAtomicScope(opList, {}), SUCCESS);
    ExpectOpListEq(opList, {&wait, &add, &midExp, &cast, &set});
}

// 反向 pipe 对复用同一 eventId（V->MTE2 set 与 MTE2->V wait）是两个无关事件，不构成配对，允许调整
TEST_F(InsertSyncTest, AdjustSyncByAtomicScopeAllowsOppositeDirectionSameEventId)
{
    NpuArchScope scopedArch(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("TestAllowOppositeDirection");
    (void)root;
    auto ts = MakeTensors(IS_NUM5, {IS_NUM8, IS_NUM16});
    auto& add = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0], ts[1]}, {ts[2]});
    auto& midExp = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_EXP, {ts[2]}, {ts[3]});
    auto& cast = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CAST, {ts[3]}, {ts[4]});
    // set: V->MTE2, eventId=6（V 产出通知 MTE2 搬运）
    // wait: MTE2->V, eventId=6（V 等 MTE2 搬运就绪）—— 反向对独立分配 eventid，数值相同只是复用
    std::vector<std::shared_ptr<LogicalTensor>> emptyInput;
    std::vector<std::shared_ptr<LogicalTensor>> emptyOutput;
    Operation& set = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_SYNC_SRC, emptyInput, emptyOutput);
    set.syncQueue_ = {PipeType::PIPE_V, PipeType::PIPE_MTE2, CoreType::AIV, CoreType::AIV,
                      IS_NUM6,          AIVCore::AIV0,       AIVCore::AIV0};
    set.SetAIVCore(AIVCore::AIV0);
    Operation& wait = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_SYNC_DST, emptyInput, emptyOutput);
    wait.syncQueue_ = {PipeType::PIPE_MTE2, PipeType::PIPE_V, CoreType::AIV, CoreType::AIV, IS_NUM6,
                       AIVCore::AIV0,       AIVCore::AIV0};
    wait.SetAIVCore(AIVCore::AIV0);
    add.SetAtomicScopeId(VF_SCOPE_A);
    midExp.SetAtomicScopeId(VF_SCOPE_A);
    cast.SetAtomicScopeId(VF_SCOPE_A);
    add.SetAIVCore(AIVCore::AIV0);
    midExp.SetAIVCore(AIVCore::AIV0);
    cast.SetAIVCore(AIVCore::AIV0);
    std::vector<Operation*> opList = {&add, &set, &midExp, &wait, &cast};

    InsertSync syncPass;
    EXPECT_EQ(syncPass.AdjustSyncByAtomicScope(opList, {}), SUCCESS);
    ExpectOpListEq(opList, {&wait, &add, &midExp, &cast, &set});
}

// cluster 内的核间同步 set（OP_CV_SYNC_SRC）可移动，移到 cluster 后
TEST_F(InsertSyncTest, AdjustSyncByAtomicScopeMovesCvSetAfterCluster)
{
    NpuArchScope scopedArch(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("TestMoveCvSetAfterCluster");
    (void)root;
    auto ts = MakeTensors(IS_NUM4, {IS_NUM8, IS_NUM16});
    auto& add = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0], ts[1]}, {ts[2]});
    auto& cast = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CAST, {ts[2]}, {ts[3]});
    // 跨核 CV set: AIV0 上执行 set(AIC PIPE_M 数据就绪) 通知 AIV1 wait(PIPE_V), eventId=1
    std::vector<std::shared_ptr<LogicalTensor>> emptyInput;
    std::vector<std::shared_ptr<LogicalTensor>> emptyOutput;
    Operation& cvSet = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CV_SYNC_SRC, emptyInput, emptyOutput);
    cvSet.syncQueue_ = {PipeType::PIPE_M, PipeType::PIPE_V, CoreType::AIC, CoreType::AIV,
                        IS_NUM1,          AIVCore::AIV0,    AIVCore::AIV1};
    cvSet.SetAIVCore(AIVCore::AIV0);
    add.SetAtomicScopeId(VF_SCOPE_A);
    cast.SetAtomicScopeId(VF_SCOPE_A);
    add.SetAIVCore(AIVCore::AIV0);
    cast.SetAIVCore(AIVCore::AIV0);
    std::vector<Operation*> opList = {&add, &cvSet, &cast};

    InsertSync syncPass;
    EXPECT_EQ(syncPass.AdjustSyncByAtomicScope(opList, {}), SUCCESS);

    ExpectOpListEq(opList, {&add, &cast, &cvSet});
}

// cluster 内的核间同步 wait（OP_CV_SYNC_DST）可移动，移到 cluster 前
TEST_F(InsertSyncTest, AdjustSyncByAtomicScopeMovesCvWaitBeforeCluster)
{
    NpuArchScope scopedArch(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("TestMoveCvWaitBeforeCluster");
    (void)root;
    auto ts = MakeTensors(IS_NUM4, {IS_NUM8, IS_NUM16});
    auto& add = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0], ts[1]}, {ts[2]});
    auto& cast = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CAST, {ts[2]}, {ts[3]});
    // 跨核 CV wait: AIV0 上等待 AIC 的 set, eventId=1
    std::vector<std::shared_ptr<LogicalTensor>> emptyInput;
    std::vector<std::shared_ptr<LogicalTensor>> emptyOutput;
    Operation& cvWait = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CV_SYNC_DST, emptyInput, emptyOutput);
    cvWait.syncQueue_ = {PipeType::PIPE_M, PipeType::PIPE_V,     CoreType::AIC, CoreType::AIV,
                         IS_NUM1,          AIVCore::UNSPECIFIED, AIVCore::AIV0};
    cvWait.SetAIVCore(AIVCore::AIV0);
    add.SetAtomicScopeId(VF_SCOPE_A);
    cast.SetAtomicScopeId(VF_SCOPE_A);
    add.SetAIVCore(AIVCore::AIV0);
    cast.SetAIVCore(AIVCore::AIV0);
    std::vector<Operation*> opList = {&add, &cvWait, &cast};

    InsertSync syncPass;
    EXPECT_EQ(syncPass.AdjustSyncByAtomicScope(opList, {}), SUCCESS);

    ExpectOpListEq(opList, {&cvWait, &add, &cast});
}

// 同一 cluster 内收集到相同 eventId 的 CV set 与 CV wait（AIV0->AIC 的 set 与 AIC->AIV0 的
// wait，双向握手共享 eventId 数值空间且复用后数值相同，两端均在本核执行）：若按边界挪动
// （wait 移前、set 移后），与窗口内原位的异核配对端 op 形成顺序颠倒（V set C wait C set V wait
// -> V wait C wait C set V set）互等死锁，须报错拒绝
TEST_F(InsertSyncTest, AdjustSyncByAtomicScopeRejectsPairedCvSetWaitInSameCluster)
{
    NpuArchScope scopedArch(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("TestRejectPairedCvSetWait");
    (void)root;
    auto ts = MakeTensors(IS_NUM5, {IS_NUM8, IS_NUM16});
    auto& add = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0], ts[1]}, {ts[2]});
    auto& midExp = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_EXP, {ts[2]}, {ts[3]});
    auto& cast = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CAST, {ts[3]}, {ts[4]});
    // CV set: AIV0->AIC 握手，AIV0 执行, eventId=3
    // CV wait: AIC->AIV0 握手，AIV0 执行, eventId=3 —— 双向握手 eventId 数值撞车
    std::vector<std::shared_ptr<LogicalTensor>> emptyInput;
    std::vector<std::shared_ptr<LogicalTensor>> emptyOutput;
    Operation& cvSet = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CV_SYNC_SRC, emptyInput, emptyOutput);
    cvSet.syncQueue_ = {PipeType::PIPE_V, PipeType::PIPE_FIX, CoreType::AIV,       CoreType::AIC,
                        IS_NUM3,          AIVCore::AIV0,      AIVCore::UNSPECIFIED};
    cvSet.SetAIVCore(AIVCore::AIV0);
    Operation& cvWait = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CV_SYNC_DST, emptyInput, emptyOutput);
    cvWait.syncQueue_ = {PipeType::PIPE_FIX,   PipeType::PIPE_V, CoreType::AIC, CoreType::AIV, IS_NUM3,
                         AIVCore::UNSPECIFIED, AIVCore::AIV0};
    cvWait.SetAIVCore(AIVCore::AIV0);
    add.SetAtomicScopeId(VF_SCOPE_A);
    midExp.SetAtomicScopeId(VF_SCOPE_A);
    cast.SetAtomicScopeId(VF_SCOPE_A);
    add.SetAIVCore(AIVCore::AIV0);
    midExp.SetAIVCore(AIVCore::AIV0);
    cast.SetAIVCore(AIVCore::AIV0);
    std::vector<Operation*> opList = {&add, &cvSet, &midExp, &cvWait, &cast};

    InsertSync syncPass;
    EXPECT_EQ(syncPass.AdjustSyncByAtomicScope(opList, {}), FAILED);
}

// 同一 cluster 内先 wait 后 set 的 CV 同步（相同 eventId）不报错，wait 移前 set 移后，相对顺序保持
TEST_F(InsertSyncTest, AdjustSyncByAtomicScopeMovesCvWaitBeforeSetInSameCluster)
{
    NpuArchScope scopedArch(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("TestMoveCvWaitBeforeSet");
    (void)root;
    auto ts = MakeTensors(IS_NUM5, {IS_NUM8, IS_NUM16});
    auto& add = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0], ts[1]}, {ts[2]});
    auto& midExp = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_EXP, {ts[2]}, {ts[3]});
    auto& cast = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CAST, {ts[3]}, {ts[4]});
    // CV wait 与 CV set 相同 eventId=3，wait 在前，移动后仍保持 wait 在 set 前
    std::vector<std::shared_ptr<LogicalTensor>> emptyInput;
    std::vector<std::shared_ptr<LogicalTensor>> emptyOutput;
    Operation& cvWait = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CV_SYNC_DST, emptyInput, emptyOutput);
    cvWait.syncQueue_ = {PipeType::PIPE_M, PipeType::PIPE_V, CoreType::AIC, CoreType::AIV,
                         IS_NUM3,          AIVCore::AIV0,    AIVCore::AIV1};
    cvWait.SetAIVCore(AIVCore::AIV0);
    Operation& cvSet = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CV_SYNC_SRC, emptyInput, emptyOutput);
    cvSet.syncQueue_ = {PipeType::PIPE_M, PipeType::PIPE_V, CoreType::AIC, CoreType::AIV,
                        IS_NUM3,          AIVCore::AIV0,    AIVCore::AIV1};
    cvSet.SetAIVCore(AIVCore::AIV0);
    add.SetAtomicScopeId(VF_SCOPE_A);
    midExp.SetAtomicScopeId(VF_SCOPE_A);
    cast.SetAtomicScopeId(VF_SCOPE_A);
    add.SetAIVCore(AIVCore::AIV0);
    midExp.SetAIVCore(AIVCore::AIV0);
    cast.SetAIVCore(AIVCore::AIV0);
    std::vector<Operation*> opList = {&add, &cvWait, &midExp, &cvSet, &cast};

    InsertSync syncPass;
    EXPECT_EQ(syncPass.AdjustSyncByAtomicScope(opList, {}), SUCCESS);

    ExpectOpListEq(opList, {&cvWait, &add, &midExp, &cast, &cvSet});
}

// cluster 内 CV set/wait eventId 不同，不配对，各自移动
TEST_F(InsertSyncTest, AdjustSyncByAtomicScopeMovesUnpairedCvSetAndWaitIndependently)
{
    NpuArchScope scopedArch(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("TestMoveUnpairedCvSetWait");
    (void)root;
    auto ts = MakeTensors(IS_NUM5, {IS_NUM8, IS_NUM16});
    auto& add = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0], ts[1]}, {ts[2]});
    auto& midExp = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_EXP, {ts[2]}, {ts[3]});
    auto& cast = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CAST, {ts[3]}, {ts[4]});
    // CV set eventId=3, CV wait eventId=5，各自独立移动
    std::vector<std::shared_ptr<LogicalTensor>> emptyInput;
    std::vector<std::shared_ptr<LogicalTensor>> emptyOutput;
    Operation& cvSet = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CV_SYNC_SRC, emptyInput, emptyOutput);
    cvSet.syncQueue_ = {PipeType::PIPE_M, PipeType::PIPE_V, CoreType::AIC, CoreType::AIV,
                        IS_NUM3,          AIVCore::AIV0,    AIVCore::AIV1};
    cvSet.SetAIVCore(AIVCore::AIV0);
    Operation& cvWait = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CV_SYNC_DST, emptyInput, emptyOutput);
    cvWait.syncQueue_ = {PipeType::PIPE_M, PipeType::PIPE_V, CoreType::AIC, CoreType::AIV,
                         IS_NUM5,          AIVCore::AIV0,    AIVCore::AIV1};
    cvWait.SetAIVCore(AIVCore::AIV0);
    add.SetAtomicScopeId(VF_SCOPE_A);
    midExp.SetAtomicScopeId(VF_SCOPE_A);
    cast.SetAtomicScopeId(VF_SCOPE_A);
    add.SetAIVCore(AIVCore::AIV0);
    midExp.SetAIVCore(AIVCore::AIV0);
    cast.SetAIVCore(AIVCore::AIV0);
    std::vector<Operation*> opList = {&add, &cvSet, &midExp, &cvWait, &cast};

    InsertSync syncPass;
    EXPECT_EQ(syncPass.AdjustSyncByAtomicScope(opList, {}), SUCCESS);

    ExpectOpListEq(opList, {&cvWait, &add, &midExp, &cast, &cvSet});
}

// CV 同步与普通同步 eventId 相同时不跨域配对，各自移动
TEST_F(InsertSyncTest, AdjustSyncByAtomicScopeCvAndRegularSyncSameEventIdNoPairing)
{
    NpuArchScope scopedArch(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("TestCvRegularSameEventId");
    (void)root;
    auto ts = MakeTensors(IS_NUM5, {IS_NUM8, IS_NUM16});
    auto& add = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0], ts[1]}, {ts[2]});
    auto& midExp = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_EXP, {ts[2]}, {ts[3]});
    auto& cast = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CAST, {ts[3]}, {ts[4]});
    // CV set eventId=3 与普通 wait(OP_SYNC_DST) eventId=3，跨域不配对
    std::vector<std::shared_ptr<LogicalTensor>> emptyInput;
    std::vector<std::shared_ptr<LogicalTensor>> emptyOutput;
    Operation& cvSet = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CV_SYNC_SRC, emptyInput, emptyOutput);
    cvSet.syncQueue_ = {PipeType::PIPE_M, PipeType::PIPE_V, CoreType::AIC, CoreType::AIV,
                        IS_NUM3,          AIVCore::AIV0,    AIVCore::AIV1};
    cvSet.SetAIVCore(AIVCore::AIV0);
    Operation& wait = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_SYNC_DST, emptyInput, emptyOutput);
    wait.syncQueue_ = {PipeType::PIPE_MTE2, PipeType::PIPE_V, CoreType::AIV, CoreType::AIV, IS_NUM3,
                       AIVCore::AIV0,       AIVCore::AIV0};
    wait.SetAIVCore(AIVCore::AIV0);
    add.SetAtomicScopeId(VF_SCOPE_A);
    midExp.SetAtomicScopeId(VF_SCOPE_A);
    cast.SetAtomicScopeId(VF_SCOPE_A);
    add.SetAIVCore(AIVCore::AIV0);
    midExp.SetAIVCore(AIVCore::AIV0);
    cast.SetAIVCore(AIVCore::AIV0);
    std::vector<Operation*> opList = {&add, &cvSet, &midExp, &wait, &cast};

    InsertSync syncPass;
    EXPECT_EQ(syncPass.AdjustSyncByAtomicScope(opList, {}), SUCCESS);

    ExpectOpListEq(opList, {&wait, &add, &midExp, &cast, &cvSet});
}

// 非 A5 平台不做后处理
TEST_F(InsertSyncTest, AdjustSyncByAtomicScopeNoopsOnNonA5)
{
    NpuArchScope scopedArch(NPUArch::DAV_2201);
    auto [root, leaf] = SetupDebugFunc("TestNoopNonA5");
    (void)root;
    auto ts = MakeTensors(IS_NUM4, {IS_NUM8, IS_NUM16});
    auto& add = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0], ts[1]}, {ts[2]});
    auto& cast = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CAST, {ts[2]}, {ts[3]});
    auto& wait = CreateAtomicScopeSyncOp(*leaf, Opcode::OP_SYNC_DST);
    add.SetAtomicScopeId(VF_SCOPE_A);
    cast.SetAtomicScopeId(VF_SCOPE_A);
    std::vector<Operation*> opList = {&add, &wait, &cast};

    InsertSync syncPass;
    EXPECT_EQ(syncPass.AdjustSyncByAtomicScope(opList, {}), SUCCESS);

    ExpectOpListEq(opList, {&add, &wait, &cast});
}

// 同一 cluster 内出现配对的 set/wait（同一 (setPipe, waitPipe, eventId) 三元组）时失败
TEST_F(InsertSyncTest, AdjustSyncByAtomicScopeRejectsPairedSetWaitInSameCluster)
{
    NpuArchScope scopedArch(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("TestRejectPairedSetWait");
    (void)root;
    auto ts = MakeTensors(IS_NUM5, {IS_NUM8, IS_NUM16});
    auto& add = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0], ts[1]}, {ts[2]});
    auto& midExp = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_EXP, {ts[2]}, {ts[3]});
    auto& cast = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CAST, {ts[3]}, {ts[4]});
    // 同一对 set/wait: 两端 syncQueue_ 均按 {set.pipe, wait.pipe} 构造 (GenSyncOp),
    // codegen 生成 set_flag(V, MTE2, 3) / wait_flag(V, MTE2, 3), 三元组完全一致
    std::vector<std::shared_ptr<LogicalTensor>> emptyInput;
    std::vector<std::shared_ptr<LogicalTensor>> emptyOutput;
    Operation& setOp = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_SYNC_SRC, emptyInput, emptyOutput);
    setOp.syncQueue_ = {PipeType::PIPE_V, PipeType::PIPE_MTE2, CoreType::AIV, CoreType::AIV,
                        IS_NUM3,          AIVCore::AIV0,       AIVCore::AIV0};
    setOp.SetAIVCore(AIVCore::AIV0);
    Operation& waitOp = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_SYNC_DST, emptyInput, emptyOutput);
    waitOp.syncQueue_ = {PipeType::PIPE_V, PipeType::PIPE_MTE2, CoreType::AIV, CoreType::AIV,
                         IS_NUM3,          AIVCore::AIV0,       AIVCore::AIV0};
    waitOp.SetAIVCore(AIVCore::AIV0);
    add.SetAtomicScopeId(VF_SCOPE_A);
    midExp.SetAtomicScopeId(VF_SCOPE_A);
    cast.SetAtomicScopeId(VF_SCOPE_A);
    add.SetAIVCore(AIVCore::AIV0);
    midExp.SetAIVCore(AIVCore::AIV0);
    cast.SetAIVCore(AIVCore::AIV0);
    // set 和 wait 都在 cluster 内部，中间有 midExp 隔开，各自 run length = 1
    std::vector<Operation*> opList = {&add, &setOp, &midExp, &waitOp, &cast};

    InsertSync syncPass;
    EXPECT_EQ(syncPass.AdjustSyncByAtomicScope(opList, {}), FAILED);
}

// cluster 内同时出现不配对的 set 和 wait（不同 eventId），各自移动到两侧，不报错
TEST_F(InsertSyncTest, AdjustSyncByAtomicScopeMovesUnpairedSetAndWaitIndependently)
{
    NpuArchScope scopedArch(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("TestMoveUnpairedSetWait");
    (void)root;
    auto ts = MakeTensors(IS_NUM5, {IS_NUM8, IS_NUM16});
    auto& add = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0], ts[1]}, {ts[2]});
    auto& midExp = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_EXP, {ts[2]}, {ts[3]});
    auto& cast = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CAST, {ts[3]}, {ts[4]});
    // set: eventId=3, pipeId_=PIPE_V, trigPipeId_=PIPE_MTE2
    // wait: eventId=5, pipeId_=PIPE_MTE3, trigPipeId_=PIPE_V
    // eventId 不同 → 不配对 → 各自移动
    std::vector<std::shared_ptr<LogicalTensor>> emptyInput;
    std::vector<std::shared_ptr<LogicalTensor>> emptyOutput;
    Operation& setOp = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_SYNC_SRC, emptyInput, emptyOutput);
    setOp.syncQueue_ = {PipeType::PIPE_V, PipeType::PIPE_MTE2, CoreType::AIV, CoreType::AIV,
                        IS_NUM3,          AIVCore::AIV0,       AIVCore::AIV0};
    setOp.SetAIVCore(AIVCore::AIV0);
    Operation& waitOp = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_SYNC_DST, emptyInput, emptyOutput);
    waitOp.syncQueue_ = {PipeType::PIPE_MTE3, PipeType::PIPE_V, CoreType::AIV, CoreType::AIV, IS_NUM5,
                         AIVCore::AIV0,       AIVCore::AIV0};
    waitOp.SetAIVCore(AIVCore::AIV0);
    add.SetAtomicScopeId(VF_SCOPE_A);
    midExp.SetAtomicScopeId(VF_SCOPE_A);
    cast.SetAtomicScopeId(VF_SCOPE_A);
    add.SetAIVCore(AIVCore::AIV0);
    midExp.SetAIVCore(AIVCore::AIV0);
    cast.SetAIVCore(AIVCore::AIV0);
    // set 和 wait 都在 cluster 内部，中间有 midExp 隔开
    std::vector<Operation*> opList = {&add, &setOp, &midExp, &waitOp, &cast};

    InsertSync syncPass;
    EXPECT_EQ(syncPass.AdjustSyncByAtomicScope(opList, {}), SUCCESS);

    // wait 移到 cluster 前，set 移到 cluster 后
    ExpectOpListEq(opList, {&waitOp, &add, &midExp, &cast, &setOp});
}

// 多个 cluster 各自独立处理，互不干扰
TEST_F(InsertSyncTest, AdjustSyncByAtomicScopeHandlesMultipleClustersIndependently)
{
    NpuArchScope scopedArch(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("TestMultiCluster");
    (void)root;
    auto ts = MakeTensors(IS_NUM8, {IS_NUM8, IS_NUM16});
    // cluster A: add, cast (atomicScopeId=7)
    // cluster B: exp, sqrt (atomicScopeId=8)
    // 中间有一个非 cluster op: neg
    auto& add = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0], ts[1]}, {ts[2]});
    auto& waitA = CreateAtomicScopeSyncOp(*leaf, Opcode::OP_SYNC_DST);
    auto& cast = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CAST, {ts[2]}, {ts[3]});
    auto& neg = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_NEG, {ts[3]}, {ts[4]});
    auto& setB = CreateAtomicScopeSyncOp(*leaf, Opcode::OP_SYNC_SRC);
    auto& exp = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_EXP, {ts[4]}, {ts[5]});
    auto& sqrt = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_SQRT, {ts[5]}, {ts[6]});
    add.SetAtomicScopeId(VF_SCOPE_A);
    cast.SetAtomicScopeId(VF_SCOPE_A);
    exp.SetAtomicScopeId(VF_SCOPE_B);
    sqrt.SetAtomicScopeId(VF_SCOPE_B);
    add.SetAIVCore(AIVCore::AIV0);
    cast.SetAIVCore(AIVCore::AIV0);
    exp.SetAIVCore(AIVCore::AIV0);
    sqrt.SetAIVCore(AIVCore::AIV0);
    std::vector<Operation*> opList = {&add, &waitA, &cast, &neg, &exp, &setB, &sqrt};

    InsertSync syncPass;
    EXPECT_EQ(syncPass.AdjustSyncByAtomicScope(opList, {}), SUCCESS);

    // cluster A: waitA 移到 add 前
    // cluster B: setB 移到 sqrt 后
    ExpectOpListEq(opList, {&waitA, &add, &cast, &neg, &exp, &sqrt, &setB});
}

// 无 atomic scope op 时不做任何处理
TEST_F(InsertSyncTest, AdjustSyncByAtomicScopeNoopsWithoutAtomicScope)
{
    NpuArchScope scopedArch(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("TestNoopNoScope");
    (void)root;
    auto ts = MakeTensors(IS_NUM4, {IS_NUM8, IS_NUM16});
    auto& add = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0], ts[1]}, {ts[2]});
    auto& cast = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CAST, {ts[2]}, {ts[3]});
    auto& wait = CreateAtomicScopeSyncOp(*leaf, Opcode::OP_SYNC_DST);
    // 不设置 atomicScopeId，默认 -1
    std::vector<Operation*> opList = {&add, &wait, &cast};

    InsertSync syncPass;
    EXPECT_EQ(syncPass.AdjustSyncByAtomicScope(opList, {}), SUCCESS);

    ExpectOpListEq(opList, {&add, &wait, &cast});
}

// 两个相邻 cluster（中间无非 cluster op），各自 set/wait sync 移到各自 cluster 边界
TEST_F(InsertSyncTest, AdjustSyncByAtomicScopeHandlesAdjacentClusters)
{
    NpuArchScope scopedArch(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("TestAdjacentClusters");
    (void)root;
    auto ts = MakeTensors(IS_NUM8, {IS_NUM8, IS_NUM16});
    // cluster A: add, cast (atomicScopeId=7)，内部有 waitA 和 setA
    // cluster B: exp, sqrt (atomicScopeId=8)，内部有 waitB 和 setB
    // 两个 cluster 直接相邻，中间无非 cluster op
    // 使用不同 eventId 避免跨 cluster 配对检查误触发
    std::vector<std::shared_ptr<LogicalTensor>> emptyInput;
    std::vector<std::shared_ptr<LogicalTensor>> emptyOutput;
    auto& add = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0], ts[1]}, {ts[2]});
    Operation& waitA = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_SYNC_DST, emptyInput, emptyOutput);
    waitA.syncQueue_ = {PipeType::PIPE_MTE2, PipeType::PIPE_V, CoreType::AIV, CoreType::AIV, IS_NUM1,
                        AIVCore::AIV0,       AIVCore::AIV0};
    waitA.SetAIVCore(AIVCore::AIV0);
    auto& cast = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CAST, {ts[2]}, {ts[3]});
    Operation& setA = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_SYNC_SRC, emptyInput, emptyOutput);
    setA.syncQueue_ = {PipeType::PIPE_V, PipeType::PIPE_MTE2, CoreType::AIV, CoreType::AIV,
                       IS_NUM2,          AIVCore::AIV0,       AIVCore::AIV0};
    setA.SetAIVCore(AIVCore::AIV0);
    auto& exp = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_EXP, {ts[3]}, {ts[4]});
    Operation& waitB = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_SYNC_DST, emptyInput, emptyOutput);
    waitB.syncQueue_ = {PipeType::PIPE_MTE3, PipeType::PIPE_V, CoreType::AIV, CoreType::AIV, IS_NUM3,
                        AIVCore::AIV0,       AIVCore::AIV0};
    waitB.SetAIVCore(AIVCore::AIV0);
    auto& sqrt = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_SQRT, {ts[4]}, {ts[5]});
    Operation& setB = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_SYNC_SRC, emptyInput, emptyOutput);
    setB.syncQueue_ = {PipeType::PIPE_V, PipeType::PIPE_MTE3, CoreType::AIV, CoreType::AIV,
                       IS_NUM4,          AIVCore::AIV0,       AIVCore::AIV0};
    setB.SetAIVCore(AIVCore::AIV0);
    add.SetAtomicScopeId(VF_SCOPE_A);
    cast.SetAtomicScopeId(VF_SCOPE_A);
    exp.SetAtomicScopeId(VF_SCOPE_B);
    sqrt.SetAtomicScopeId(VF_SCOPE_B);
    add.SetAIVCore(AIVCore::AIV0);
    cast.SetAIVCore(AIVCore::AIV0);
    exp.SetAIVCore(AIVCore::AIV0);
    sqrt.SetAIVCore(AIVCore::AIV0);
    // 原始顺序: add, waitA, cast, setA, exp, waitB, sqrt, setB
    std::vector<Operation*> opList = {&add, &waitA, &cast, &setA, &exp, &waitB, &sqrt, &setB};

    InsertSync syncPass;
    EXPECT_EQ(syncPass.AdjustSyncByAtomicScope(opList, {}), SUCCESS);

    // cluster A: waitA 移到 add 前，setA 移到 cast 后
    // cluster B: waitB 移到 exp 前，setB 移到 sqrt 后
    // 两个 cluster 相邻，但各自的 sync 仍然移到各自边界
    ExpectOpListEq(opList, {&waitA, &add, &cast, &setA, &waitB, &exp, &sqrt, &setB});
}

// 同一 cluster 内多条 sync 指令，移动后保持原始相对顺序
TEST_F(InsertSyncTest, AdjustSyncByAtomicScopePreservesSyncOrderWithinCluster)
{
    NpuArchScope scopedArch(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("TestPreserveSyncOrder");
    (void)root;
    auto ts = MakeTensors(IS_NUM6, {IS_NUM8, IS_NUM16});
    // cluster: add, exp, cast, neg, sqrt (atomicScopeId=7)
    // 内部有 2 条 wait-like 和 2 条 set-like，每条都被 cluster op 隔开（run length = 1）
    // 使用不同 eventId 避免配对检查失败
    auto& add = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0], ts[1]}, {ts[2]});
    auto& wait1 = CreateAtomicScopeSyncOp(*leaf, Opcode::OP_SYNC_DST);
    auto& exp = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_EXP, {ts[2]}, {ts[3]});
    std::vector<std::shared_ptr<LogicalTensor>> emptyInput;
    std::vector<std::shared_ptr<LogicalTensor>> emptyOutput;
    Operation& set1 = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_SYNC_SRC, emptyInput, emptyOutput);
    set1.syncQueue_ = {PipeType::PIPE_V, PipeType::PIPE_MTE2, CoreType::AIV, CoreType::AIV,
                       IS_NUM2,          AIVCore::AIV0,       AIVCore::AIV0};
    set1.SetAIVCore(AIVCore::AIV0);
    auto& cast = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CAST, {ts[3]}, {ts[4]});
    Operation& wait2 = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_SYNC_DST, emptyInput, emptyOutput);
    wait2.syncQueue_ = {PipeType::PIPE_MTE3, PipeType::PIPE_V, CoreType::AIV, CoreType::AIV, IS_NUM4,
                        AIVCore::AIV0,       AIVCore::AIV0};
    wait2.SetAIVCore(AIVCore::AIV0);
    auto& neg = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_NEG, {ts[4]}, {ts[5]});
    Operation& set2 = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_SYNC_SRC, emptyInput, emptyOutput);
    set2.syncQueue_ = {PipeType::PIPE_V, PipeType::PIPE_MTE2, CoreType::AIV, CoreType::AIV,
                       IS_NUM5,          AIVCore::AIV0,       AIVCore::AIV0};
    set2.SetAIVCore(AIVCore::AIV0);
    auto& sqrt = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_SQRT, {ts[5]}, {ts[0]});
    add.SetAtomicScopeId(VF_SCOPE_A);
    exp.SetAtomicScopeId(VF_SCOPE_A);
    cast.SetAtomicScopeId(VF_SCOPE_A);
    neg.SetAtomicScopeId(VF_SCOPE_A);
    sqrt.SetAtomicScopeId(VF_SCOPE_A);
    add.SetAIVCore(AIVCore::AIV0);
    exp.SetAIVCore(AIVCore::AIV0);
    cast.SetAIVCore(AIVCore::AIV0);
    neg.SetAIVCore(AIVCore::AIV0);
    sqrt.SetAIVCore(AIVCore::AIV0);
    // 原始顺序: add, wait1, exp, set1, cast, wait2, neg, set2, sqrt
    // 每条 sync 都被 cluster op 隔开，run length = 1
    // eventId 各不相同，不会触发配对检查
    std::vector<Operation*> opList = {&add, &wait1, &exp, &set1, &cast, &wait2, &neg, &set2, &sqrt};

    InsertSync syncPass;
    EXPECT_EQ(syncPass.AdjustSyncByAtomicScope(opList, {}), SUCCESS);

    // wait1, wait2 移到 cluster 前（保持原始出现顺序: wait1 在 wait2 前）
    // set1, set2 移到 cluster 后（保持原始出现顺序: set1 在 set2 前）
    ExpectOpListEq(opList, {&wait1, &wait2, &add, &exp, &cast, &neg, &sqrt, &set1, &set2});
}

// cluster 区间内交错另一核（AIV1）的 op（无 scope 辅助 op 与另一核 cluster op），跳过挪动判定不报错，
// 本核 sync 正常挪动，外来 op 保持原位
TEST_F(InsertSyncTest, AdjustSyncByAtomicScopeSkipsForeignCoreOps)
{
    NpuArchScope scopedArch(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("TestSkipForeignCoreOps");
    (void)root;
    auto ts = MakeTensors(IS_NUM6, {IS_NUM8, IS_NUM16});
    auto& add = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0], ts[1]}, {ts[2]});
    std::vector<std::shared_ptr<LogicalTensor>> emptyInput;
    std::vector<std::shared_ptr<LogicalTensor>> emptyOutput;
    Operation& cvSet = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CV_SYNC_SRC, emptyInput, emptyOutput);
    cvSet.syncQueue_ = {PipeType::PIPE_M, PipeType::PIPE_V, CoreType::AIC, CoreType::AIV,
                        IS_NUM1,          AIVCore::AIV0,    AIVCore::AIV1};
    cvSet.SetAIVCore(AIVCore::AIV0);
    // 模拟 OOO 交错发射的另一核 op: AIV1 无 scope 辅助 op + AIV1 另一 cluster 的 calc op
    auto& foreignMove = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_UB_COPY_ND2NZ, {ts[2]}, {ts[3]});
    auto& foreignCalc = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_EXP, {ts[3]}, {ts[4]});
    auto& cast = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CAST, {ts[4]}, {ts[5]});
    add.SetAtomicScopeId(VF_SCOPE_A);
    cast.SetAtomicScopeId(VF_SCOPE_A);
    foreignCalc.SetAtomicScopeId(VF_SCOPE_B);
    add.SetAIVCore(AIVCore::AIV0);
    cast.SetAIVCore(AIVCore::AIV0);
    foreignMove.SetAIVCore(AIVCore::AIV1);
    foreignCalc.SetAIVCore(AIVCore::AIV1);
    std::vector<Operation*> opList = {&add, &cvSet, &foreignMove, &foreignCalc, &cast};

    InsertSync syncPass;
    EXPECT_EQ(syncPass.AdjustSyncByAtomicScope(opList, {}), SUCCESS);

    // 默认模式：本核 cvSet 移到 cluster 后，另一核 op 保持原位
    ExpectOpListEq(opList, {&add, &foreignMove, &foreignCalc, &cast, &cvSet});
}

// 窗口内夹带同核 scope=-1 op（如用户 scope cluster 内未压实的搬运）：整 cluster 跳过调整而非报错
TEST_F(InsertSyncTest, AdjustSyncByAtomicScopeSkipsClusterWithUnscopedOpInside)
{
    NpuArchScope scopedArch(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("TestUnscopedInside");
    (void)root;
    auto ts = MakeTensors(IS_NUM16, {IS_NUM8, IS_NUM16});
    auto& add = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0], ts[1]}, {ts[2]});
    auto& wait = CreateAtomicScopeSyncOp(*leaf, Opcode::OP_SYNC_DST);
    // 同核但无 scope 的搬运 op，夹在 cluster 窗口内
    auto& copy = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_UB_COPY_ND2NZ, {ts[2]}, {ts[3]});
    auto& cast = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CAST, {ts[3]}, {ts[4]});
    add.SetAtomicScopeId(VF_SCOPE_A);
    cast.SetAtomicScopeId(VF_SCOPE_A);
    add.SetAIVCore(AIVCore::AIV0);
    copy.SetAIVCore(AIVCore::AIV0);
    cast.SetAIVCore(AIVCore::AIV0);
    std::vector<Operation*> opList = {&add, &wait, &copy, &cast};

    InsertSync syncPass;
    EXPECT_EQ(syncPass.AdjustSyncByAtomicScope(opList, {}), SUCCESS);

    // cluster 被跳过：wait 保持原位
    ExpectOpListEq(opList, {&add, &wait, &copy, &cast});
}

// 用户设置的 atomic_scope（scopeId < VF_CLUSTER_ID_START）：前序处理直接跳过，不收集不检测，VF 簇正常调整
TEST_F(InsertSyncTest, AdjustSyncByAtomicScopeSkipsUserScopedClusters)
{
    NpuArchScope scopedArch(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("TestSkipUserScope");
    (void)root;
    auto ts = MakeTensors(IS_NUM6, {IS_NUM8, IS_NUM16});
    // 用户 scope 簇：wait 保持原位不挪动
    auto& userAdd = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0], ts[1]}, {ts[2]});
    auto& userWait = CreateAtomicScopeSyncOp(*leaf, Opcode::OP_SYNC_DST);
    auto& userCast = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CAST, {ts[2]}, {ts[3]});
    // VF 簇：set 正常移到 cluster 后
    auto& vfAdd = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_EXP, {ts[3]}, {ts[4]});
    auto& vfSet = CreateAtomicScopeSyncOp(*leaf, Opcode::OP_SYNC_SRC);
    auto& vfCast = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_SUB, {ts[4]}, {ts[5]});
    constexpr int userScopeId = 7 * 10000 + 3;
    userAdd.SetAtomicScopeId(userScopeId);
    userCast.SetAtomicScopeId(userScopeId);
    vfAdd.SetAtomicScopeId(VF_SCOPE_A);
    vfCast.SetAtomicScopeId(VF_SCOPE_A);
    userAdd.SetAIVCore(AIVCore::AIV0);
    userCast.SetAIVCore(AIVCore::AIV0);
    vfAdd.SetAIVCore(AIVCore::AIV0);
    vfCast.SetAIVCore(AIVCore::AIV0);
    std::vector<Operation*> opList = {&userAdd, &userWait, &userCast, &vfAdd, &vfSet, &vfCast};

    InsertSync syncPass;
    EXPECT_EQ(syncPass.AdjustSyncByAtomicScope(opList, {}), SUCCESS);

    // 用户 scope 簇：wait 原位；VF 簇：set 移到 cluster 后
    ExpectOpListEq(opList, {&userAdd, &userWait, &userCast, &vfAdd, &vfCast, &vfSet});
}

// cluster 区间内交错另一核（AIV1）的 sync op，不收集不挪动、保持原位，本核 sync 正常挪动
TEST_F(InsertSyncTest, AdjustSyncByAtomicScopeSkipsForeignCoreSyncs)
{
    NpuArchScope scopedArch(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("TestSkipForeignCoreSyncs");
    (void)root;
    auto ts = MakeTensors(IS_NUM4, {IS_NUM8, IS_NUM16});
    auto& add = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0], ts[1]}, {ts[2]});
    std::vector<std::shared_ptr<LogicalTensor>> emptyInput;
    std::vector<std::shared_ptr<LogicalTensor>> emptyOutput;
    Operation& wait = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_SYNC_DST, emptyInput, emptyOutput);
    wait.syncQueue_ = {PipeType::PIPE_MTE2, PipeType::PIPE_V, CoreType::AIV, CoreType::AIV, IS_NUM1,
                       AIVCore::AIV0,       AIVCore::AIV0};
    wait.SetAIVCore(AIVCore::AIV0);
    // 模拟 OOO 交错发射的另一核（AIV1）sync
    Operation& foreignSyncSrc = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_SYNC_SRC, emptyInput, emptyOutput);
    foreignSyncSrc.syncQueue_ = {PipeType::PIPE_V, PipeType::PIPE_MTE3, CoreType::AIV, CoreType::AIV,
                                 IS_NUM2,          AIVCore::AIV1,       AIVCore::AIV1};
    foreignSyncSrc.SetAIVCore(AIVCore::AIV1);
    Operation& foreignSyncDst = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_SYNC_DST, emptyInput, emptyOutput);
    foreignSyncDst.syncQueue_ = {PipeType::PIPE_MTE3, PipeType::PIPE_V, CoreType::AIV, CoreType::AIV, IS_NUM3,
                                 AIVCore::AIV1,       AIVCore::AIV1};
    foreignSyncDst.SetAIVCore(AIVCore::AIV1);
    auto& cast = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CAST, {ts[2]}, {ts[3]});
    add.SetAtomicScopeId(VF_SCOPE_A);
    cast.SetAtomicScopeId(VF_SCOPE_A);
    add.SetAIVCore(AIVCore::AIV0);
    cast.SetAIVCore(AIVCore::AIV0);
    std::vector<Operation*> opList = {&add, &wait, &foreignSyncSrc, &foreignSyncDst, &cast};

    InsertSync syncPass;
    EXPECT_EQ(syncPass.AdjustSyncByAtomicScope(opList, {}), SUCCESS);

    // 默认模式：本核 wait 移到 cluster 前，另一核 sync 保持原位
    ExpectOpListEq(opList, {&wait, &add, &foreignSyncSrc, &foreignSyncDst, &cast});
}

// =========================================================================
// 端到端测试：通过 RunOnFunction 走完整 normal 路径
// PipeSync::InsertSync → PhaseKernelProcess → AdjustSyncByAtomicScope
// =========================================================================

// Helper: 创建带 UB memory range 的 tensor
static std::shared_ptr<LogicalTensor> MakeUbTensor(int start, int end)
{
    std::vector<int64_t> shape = {IS_NUM8, IS_NUM16};
    auto t = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    t->SetMemoryTypeBoth(MemoryType::MEM_UB);
    t->memoryrange.start = start;
    t->memoryrange.end = end;
    return t;
}

// Helper: 创建 GM tensor
static std::shared_ptr<LogicalTensor> MakeGmTensor()
{
    std::vector<int64_t> shape = {IS_NUM8, IS_NUM16};
    return npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
}

// Helper: 创建 copy_in op (GM → UB)，带 CopyOpAttribute
static Operation& MakeCopyInOp(Function& func, std::shared_ptr<LogicalTensor> gm, std::shared_ptr<LogicalTensor> ub)
{
    std::vector<int64_t> shape = {IS_NUM8, IS_NUM16};
    auto& op = IRBuilder().CreateTensorOpStmt(func, Opcode::OP_COPY_IN, {gm}, {ub});
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(std::vector<OpImmediate>{}, MemoryType::MEM_UB,
                                                        std::vector<OpImmediate>{OpImmediate::Specified(shape)},
                                                        std::vector<OpImmediate>{OpImmediate::Specified(shape)}));
    op.SetAIVCore(AIVCore::AIV0);
    return op;
}

// Helper: 创建 copy_out op (UB → GM)，带 CopyOpAttribute
static Operation& MakeCopyOutOp(Function& func, std::shared_ptr<LogicalTensor> ub, std::shared_ptr<LogicalTensor> gm)
{
    std::vector<int64_t> shape = {IS_NUM8, IS_NUM16};
    auto& op = IRBuilder().CreateTensorOpStmt(func, Opcode::OP_COPY_OUT, {ub}, {gm});
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(MemoryType::MEM_UB, std::vector<OpImmediate>{},
                                                        std::vector<OpImmediate>{OpImmediate::Specified(shape)},
                                                        std::vector<OpImmediate>{OpImmediate::Specified(shape)}));
    op.SetAIVCore(AIVCore::AIV0);
    return op;
}

// Helper: 创建 root + leaf function pair
static std::pair<std::shared_ptr<Function>, std::shared_ptr<Function>> CreateRootAndLeaf(const std::string& name)
{
    auto root = std::make_shared<Function>(Program::GetInstance(), name, name, nullptr);
    root->rootFunc_ = root.get();
    auto leaf = std::make_shared<Function>(Program::GetInstance(), name + "Leaf", name + "Leaf", root.get());
    root->rootFunc_->programs_.emplace(leaf->GetFuncMagic(), leaf.get());
    return {root, leaf};
}

// Helper: 验证 firstOp 和 lastOp 之间没有 sync op
static void VerifyNoSyncBetween(const std::vector<Operation*>& ops, Operation* firstOp, Operation* lastOp)
{
    bool foundFirst = false;
    bool foundLast = false;
    bool hasSync = false;
    for (auto* op : ops) {
        if (op == firstOp) {
            foundFirst = true;
            continue;
        }
        if (op == lastOp) {
            foundLast = true;
            continue;
        }
        if (foundFirst && !foundLast &&
            (op->GetOpcode() == Opcode::OP_SYNC_SRC || op->GetOpcode() == Opcode::OP_SYNC_DST)) {
            hasSync = true;
        }
    }
    EXPECT_TRUE(foundFirst);
    EXPECT_TRUE(foundLast);
    EXPECT_FALSE(hasSync) << "sync op found inside cluster";
}

// Helper: 验证 targetOp 之前有指定 sync opcode
static void VerifyHasSyncBefore(const std::vector<Operation*>& ops, Operation* targetOp, Opcode syncOpcode)
{
    bool found = false;
    for (auto* op : ops) {
        if (op == targetOp) {
            break;
        }
        if (op->GetOpcode() == syncOpcode) {
            found = true;
        }
    }
    EXPECT_TRUE(found) << "no sync before target op";
}

// Helper: 验证 targetOp 之后有指定 sync opcode
static void VerifyHasSyncAfter(const std::vector<Operation*>& ops, Operation* targetOp, Opcode syncOpcode)
{
    bool afterTarget = false;
    bool found = false;
    for (auto* op : ops) {
        if (op == targetOp) {
            afterTarget = true;
            continue;
        }
        if (afterTarget && op->GetOpcode() == syncOpcode) {
            found = true;
        }
    }
    EXPECT_TRUE(found) << "no sync after target op";
}

// Helper: 验证两个 op 之间有指定 sync opcode
static void VerifyHasSyncBetween(const std::vector<Operation*>& ops, Operation* firstOp, Operation* lastOp,
                                 Opcode syncOpcode)
{
    bool afterFirst = false;
    bool found = false;
    for (auto* op : ops) {
        if (op == firstOp) {
            afterFirst = true;
            continue;
        }
        if (op == lastOp) {
            break;
        }
        if (afterFirst && op->GetOpcode() == syncOpcode) {
            found = true;
        }
    }
    EXPECT_TRUE(found) << "no sync between ops";
}

// 端到端：cluster 内部有跨 pipe 依赖产生的 sync，验证 sync 被移到 cluster 边界
// 构图: copy_in(a) copy_in(b) → exp(a→c) add(c,b→d) sub(d→e) → copy_out(d) copy_out(e)
// exp/add/sub 设为同一 cluster，copy_in(b)→add 的 sync 会落在 cluster 内部
TEST_F(InsertSyncTest, EndToEndSyncMovedToClusterBoundary)
{
    NpuArchScope scopedArch(NPUArch::DAV_3510);
    auto [rootFuncPtr, leafFuncPtr] = CreateRootAndLeaf("E2EBoundary");

    auto ubA = MakeUbTensor(0, IS_NUM99);
    auto ubB = MakeUbTensor(IS_NUM100, IS_NUM199);
    auto ubC = MakeUbTensor(IS_NUM200, IS_NUM299);
    auto ubD = MakeUbTensor(IS_NUM300, IS_NUM300 + IS_NUM99);
    auto ubE = MakeUbTensor(IS_NUM500, IS_NUM500 + IS_NUM99);
    MakeCopyInOp(*leafFuncPtr, MakeGmTensor(), ubA);
    MakeCopyInOp(*leafFuncPtr, MakeGmTensor(), ubB);
    auto& exp = IRBuilder().CreateTensorOpStmt(*leafFuncPtr, Opcode::OP_EXP, {ubA}, {ubC});
    exp.SetAIVCore(AIVCore::AIV0);
    auto& add = IRBuilder().CreateTensorOpStmt(*leafFuncPtr, Opcode::OP_ADD, {ubC, ubB}, {ubD});
    add.SetAIVCore(AIVCore::AIV0);
    auto& sub = IRBuilder().CreateTensorOpStmt(*leafFuncPtr, Opcode::OP_SUB, {ubD}, {ubE});
    sub.SetAIVCore(AIVCore::AIV0);
    MakeCopyOutOp(*leafFuncPtr, ubD, MakeGmTensor());
    MakeCopyOutOp(*leafFuncPtr, ubE, MakeGmTensor());

    constexpr int clusterId = 200000000;
    exp.SetAtomicScopeId(clusterId);
    add.SetAtomicScopeId(clusterId);
    sub.SetAtomicScopeId(clusterId);

    InsertSync syncPass;
    ASSERT_EQ(syncPass.RunOnFunction(*rootFuncPtr), SUCCESS);
    auto ops = leafFuncPtr->Operations(false).DuplicatedOpList();

    VerifyNoSyncBetween(ops, &exp, &sub);
    VerifyHasSyncBefore(ops, &exp, Opcode::OP_SYNC_DST);
    VerifyHasSyncAfter(ops, &sub, Opcode::OP_SYNC_SRC);
}

// 端到端：两个相邻 cluster，各自内部有跨 pipe sync，验证互不干扰
// 构图: copy_in(a) copy_in(b) copy_in(c2) → exp add | sqrt cast → copy_out(d) copy_out(f)
// cluster A: exp,add  cluster B: sqrt,cast  相邻
TEST_F(InsertSyncTest, EndToEndAdjacentClustersSyncMovedIndependently)
{
    NpuArchScope scopedArch(NPUArch::DAV_3510);
    auto [rootFuncPtr, leafFuncPtr] = CreateRootAndLeaf("E2EAdj");

    auto ubA = MakeUbTensor(0, IS_NUM99);
    auto ubB = MakeUbTensor(IS_NUM100, IS_NUM199);
    auto ubC = MakeUbTensor(IS_NUM200, IS_NUM299);
    auto ubD = MakeUbTensor(IS_NUM300, IS_NUM300 + IS_NUM99);
    auto ubE = MakeUbTensor(IS_NUM500, IS_NUM500 + IS_NUM99);
    auto ubF = MakeUbTensor(IS_NUM600, IS_NUM699);
    auto ubC2 = MakeUbTensor(IS_NUM700, IS_NUM800);
    MakeCopyInOp(*leafFuncPtr, MakeGmTensor(), ubA);
    MakeCopyInOp(*leafFuncPtr, MakeGmTensor(), ubB);
    MakeCopyInOp(*leafFuncPtr, MakeGmTensor(), ubC2);
    auto& exp = IRBuilder().CreateTensorOpStmt(*leafFuncPtr, Opcode::OP_EXP, {ubA}, {ubC});
    exp.SetAIVCore(AIVCore::AIV0);
    auto& add = IRBuilder().CreateTensorOpStmt(*leafFuncPtr, Opcode::OP_ADD, {ubC, ubB}, {ubD});
    add.SetAIVCore(AIVCore::AIV0);
    auto& sqrt = IRBuilder().CreateTensorOpStmt(*leafFuncPtr, Opcode::OP_SQRT, {ubD}, {ubE});
    sqrt.SetAIVCore(AIVCore::AIV0);
    auto& cast = IRBuilder().CreateTensorOpStmt(*leafFuncPtr, Opcode::OP_CAST, {ubE, ubC2}, {ubF});
    cast.SetAIVCore(AIVCore::AIV0);
    MakeCopyOutOp(*leafFuncPtr, ubD, MakeGmTensor());
    MakeCopyOutOp(*leafFuncPtr, ubF, MakeGmTensor());

    constexpr int clusterA = 200000000;
    constexpr int clusterB = 200000001;
    exp.SetAtomicScopeId(clusterA);
    add.SetAtomicScopeId(clusterA);
    sqrt.SetAtomicScopeId(clusterB);
    cast.SetAtomicScopeId(clusterB);

    InsertSync syncPass;
    ASSERT_EQ(syncPass.RunOnFunction(*rootFuncPtr), SUCCESS);
    auto ops = leafFuncPtr->Operations(false).DuplicatedOpList();

    VerifyNoSyncBetween(ops, &exp, &add);
    VerifyNoSyncBetween(ops, &sqrt, &cast);
    VerifyHasSyncBefore(ops, &exp, Opcode::OP_SYNC_DST);
    VerifyHasSyncBetween(ops, &add, &sqrt, Opcode::OP_SYNC_DST);
    VerifyHasSyncAfter(ops, &cast, Opcode::OP_SYNC_SRC);
}

// 3510: AIV 同流水(PIPE_MTE3, assemble/COPY_OUT 场景)数据依赖必须生成 OP_BAR_M, 不能丢弃
TEST_F(InsertSyncTest, TestGenSyncOpAivSameMte3PipeOn3510GeneratesBarM)
{
    NpuArchScope scopedArch(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("TestGenSyncAivMte3");
    auto ts = MakeTensors(IS_NUM2, {IS_NUM8, IS_NUM16});
    Operation& op = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0]}, {ts[1]});
    PipeSync ps;
    PipeSync::PipeCoreRealEx set(PipeType::PIPE_MTE3, CoreType::AIV, AIVCore::AIV0);
    PipeSync::PipeCoreRealEx wait(PipeType::PIPE_MTE3, CoreType::AIV, AIVCore::AIV0);
    EXPECT_TRUE(ps.GenSyncOp(set, wait, IS_NUM1, false, op));
    EXPECT_EQ(op.GetOpcode(), Opcode::OP_BAR_M);
    EXPECT_EQ(op.syncQueue_.pipeId_, PipeType::PIPE_MTE3);
    EXPECT_EQ(op.syncQueue_.trigPipeId_, PipeType::PIPE_MTE3);
    EXPECT_EQ(op.syncQueue_.coreType_, CoreType::AIV);
    EXPECT_EQ(op.syncQueue_.trigCoreType_, CoreType::AIV);
    EXPECT_EQ(op.syncQueue_.eventId_, IS_NUM1);
}

// lite(3113): AIV 同流水(PIPE_MTE3)数据依赖同样必须生成 OP_BAR_M, 不能丢弃
TEST_F(InsertSyncTest, TestGenSyncOpAivSameMte3PipeOnLiteGeneratesBarM)
{
    NpuArchScope scopedArch(NPUArch::DAV_3113);
    auto [root, leaf] = SetupDebugFunc("TestGenSyncAivMte3Lite");
    auto ts = MakeTensors(IS_NUM2, {IS_NUM8, IS_NUM16});
    Operation& op = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0]}, {ts[1]});
    PipeSync ps;
    PipeSync::PipeCoreRealEx set(PipeType::PIPE_MTE3, CoreType::AIV, AIVCore::AIV0);
    PipeSync::PipeCoreRealEx wait(PipeType::PIPE_MTE3, CoreType::AIV, AIVCore::AIV0);
    EXPECT_TRUE(ps.GenSyncOp(set, wait, IS_NUM1, false, op));
    EXPECT_EQ(op.GetOpcode(), Opcode::OP_BAR_M);
}

// 3510: AIV V->V 同步维持原行为, 不生成同步 op
TEST_F(InsertSyncTest, TestGenSyncOpAivVVPipeOn3510KeepsSkip)
{
    NpuArchScope scopedArch(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("TestGenSyncAivV3510");
    auto ts = MakeTensors(IS_NUM2, {IS_NUM8, IS_NUM16});
    Operation& op = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0]}, {ts[1]});
    PipeSync ps;
    PipeSync::PipeCoreRealEx set(PipeType::PIPE_V, CoreType::AIV, AIVCore::AIV0);
    PipeSync::PipeCoreRealEx wait(PipeType::PIPE_V, CoreType::AIV, AIVCore::AIV0);
    EXPECT_FALSE(ps.GenSyncOp(set, wait, IS_NUM1, false, op));
}

// 非 3510 非 lite(2201): AIV V->V 仍生成 OP_BAR_V(原有行为)
TEST_F(InsertSyncTest, TestGenSyncOpAivVVPipeOn2201GeneratesBarV)
{
    NpuArchScope scopedArch(NPUArch::DAV_2201);
    auto [root, leaf] = SetupDebugFunc("TestGenSyncAivV2201");
    auto ts = MakeTensors(IS_NUM2, {IS_NUM8, IS_NUM16});
    Operation& op = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0]}, {ts[1]});
    PipeSync ps;
    PipeSync::PipeCoreRealEx set(PipeType::PIPE_V, CoreType::AIV, AIVCore::AIV0);
    PipeSync::PipeCoreRealEx wait(PipeType::PIPE_V, CoreType::AIV, AIVCore::AIV0);
    EXPECT_TRUE(ps.GenSyncOp(set, wait, IS_NUM1, false, op));
    EXPECT_EQ(op.GetOpcode(), Opcode::OP_BAR_V);
}

// isSet=true: 源头侧不生成同步 op(原有行为)
TEST_F(InsertSyncTest, TestGenSyncOpAivSamePipeSetSideNoop)
{
    NpuArchScope scopedArch(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("TestGenSyncAivSetSide");
    auto ts = MakeTensors(IS_NUM2, {IS_NUM8, IS_NUM16});
    Operation& op = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0]}, {ts[1]});
    PipeSync ps;
    PipeSync::PipeCoreRealEx set(PipeType::PIPE_MTE3, CoreType::AIV, AIVCore::AIV0);
    PipeSync::PipeCoreRealEx wait(PipeType::PIPE_MTE3, CoreType::AIV, AIVCore::AIV0);
    EXPECT_FALSE(ps.GenSyncOp(set, wait, IS_NUM1, true, op));
}

// cvHardSync debug mode：无CV边界（纯vec函数）正常完成，不插入跨核硬同步
TEST_F(InsertSyncTest, TestCvHardSyncNoBoundary)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestCvHardSyncVec", "TestCvHardSyncVec",
                                                      nullptr);

    auto incast1 = MakeGmTensor();
    auto incast2 = MakeGmTensor();
    auto ubTensor1 = MakeUbTensor(0, IS_NUM100);
    auto ubTensor2 = MakeUbTensor(IS_NUM200, IS_NUM300);
    auto ubTensor3 = MakeUbTensor(IS_NUM300, IS_NUM499);
    auto outCast = MakeGmTensor();

    MakeCopyInOp(*currFunctionPtr, incast1, ubTensor1);
    MakeCopyInOp(*currFunctionPtr, incast2, ubTensor2);
    auto& addOp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_ADD, {ubTensor1, ubTensor2}, {ubTensor3});
    (void)addOp;
    addOp.SetAIVCore(AIVCore::AIV0);
    MakeCopyOutOp(*currFunctionPtr, ubTensor3, outCast);
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outCast);

    InsertSync syncPass;
    syncPass.SetCvHardSync(true);
    EXPECT_EQ(syncPass.InsertSyncMainLoop(currFunctionPtr.get()), SUCCESS);
    for (auto& op : currFunctionPtr->Operations()) {
        EXPECT_NE(op.GetOpcode(), Opcode::OP_FFTS_CROSS_CORE_SYNC) << "No CV boundary, hard sync should not appear";
    }
}

// cvHardSync debug mode：vec→cube→vec混合函数在每个核切换处插入1C2V硬同步
TEST_F(InsertSyncTest, TestCvHardSyncWithBoundary)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestCvHardSyncMix", "TestCvHardSyncMix",
                                                      nullptr);

    auto incast = MakeGmTensor();
    auto ubTensor1 = MakeUbTensor(0, IS_NUM100);
    auto ubTensor2 = MakeUbTensor(IS_NUM200, IS_NUM300);
    auto ubTensor3 = MakeUbTensor(IS_NUM300, IS_NUM499);
    auto outCast = MakeGmTensor();

    MakeCopyInOp(*currFunctionPtr, incast, ubTensor1);
    auto& cubeOp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_A_MUL_B, {ubTensor1, ubTensor1},
                                                  {ubTensor2});
    (void)cubeOp;
    auto& addOp = IRBuilder().CreateTensorOpStmt(*currFunctionPtr, Opcode::OP_ADD, {ubTensor2}, {ubTensor3});
    (void)addOp;
    addOp.SetAIVCore(AIVCore::AIV0);
    MakeCopyOutOp(*currFunctionPtr, ubTensor3, outCast);
    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outCast);

    InsertSync syncPass;
    syncPass.SetCvHardSync(true);
    EXPECT_EQ(syncPass.InsertSyncMainLoop(currFunctionPtr.get()), SUCCESS);
    bool hasHardSync = false;
    for (auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_FFTS_CROSS_CORE_SYNC) {
            hasHardSync = true;
        }
    }
    EXPECT_TRUE(hasHardSync) << "CV boundary exists, 1C2V hard sync should be inserted";
}

// ENABLE_PIPE_ALL 全同步调试开关：InsertSyncMainLoop 直接走 InsertPipeAll，
// 在相邻非 RESHAPE/VIEW/VIEW_TYPE/ASSEMBLE op 之间插入 OP_BAR_ALL。
TEST_F(InsertSyncTest, TestInsertPipeAll)
{
    config::SetPassGlobalConfig(KEY_ENABLE_PIPE_ALL, true);
    auto [root, leaf] = SetupDebugFunc("TestPipeAll");
    auto ts = MakeTensors(IS_NUM7, {IS_NUM8, IS_NUM16});
    (void)IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0], ts[1]}, {ts[2]});
    (void)IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_RESHAPE, {ts[2]}, {ts[3]});
    (void)IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_VIEW, {ts[3]}, {ts[4]});
    (void)IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_CAST, {ts[4]}, {ts[5]});
    (void)IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_EXP, {ts[5]}, {ts[6]});
    InsertSync syncPass;
    syncPass.RunOnFunction(*root);
    // 原始 op 列表保持不变
    EXPECT_EQ(leaf->oriOpList.size(), static_cast<size_t>(IS_NUM5));
    // ADD BAR_ALL RESHAPE VIEW CAST BAR_ALL EXP
    auto ops = leaf->Operations(false).DuplicatedOpList();
    EXPECT_EQ(ops.size(), static_cast<size_t>(IS_NUM7));
    EXPECT_EQ(ops[0]->GetOpcode(), Opcode::OP_ADD);
    EXPECT_EQ(ops[IS_NUM1]->GetOpcode(), Opcode::OP_BAR_ALL);
    EXPECT_EQ(ops[IS_NUM2]->GetOpcode(), Opcode::OP_RESHAPE);
    EXPECT_EQ(ops[IS_NUM3]->GetOpcode(), Opcode::OP_VIEW);
    EXPECT_EQ(ops[IS_NUM4]->GetOpcode(), Opcode::OP_CAST);
    EXPECT_EQ(ops[IS_NUM5]->GetOpcode(), Opcode::OP_BAR_ALL);
    EXPECT_EQ(ops[IS_NUM6]->GetOpcode(), Opcode::OP_EXP);
}

// ENABLE_PIPE_ALL 边界：单 op 无相邻 op，不插入任何 OP_BAR_ALL。
TEST_F(InsertSyncTest, TestInsertPipeAllSingleOp)
{
    config::SetPassGlobalConfig(KEY_ENABLE_PIPE_ALL, true);
    auto [root, leaf] = SetupDebugFunc("TestPipeAllSingle");
    auto ts = MakeTensors(IS_NUM2, {IS_NUM8, IS_NUM16});
    (void)IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {ts[0]}, {ts[1]});
    InsertSync syncPass;
    syncPass.RunOnFunction(*root);
    EXPECT_EQ(leaf->oriOpList.size(), static_cast<size_t>(IS_NUM1));
    auto ops = leaf->Operations(false).DuplicatedOpList();
    EXPECT_EQ(ops.size(), static_cast<size_t>(IS_NUM1));
    EXPECT_EQ(ops[0]->GetOpcode(), Opcode::OP_ADD);
}

TEST_F(InsertSyncTest, IntraDrainUsesCompletedPrefixAfterReplay)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("IntraDrainReplayPrefix");
    auto& function = *funcs.second;
    PipeSync ps;
    const PipeSync::PipePairEx pipes{{PIPE_MTE1, CoreType::AIC, AIVCore::UNSPECIFIED},
                                     {PIPE_M, CoreType::AIC, AIVCore::UNSPECIFIED}};
    std::vector<IndexOp> synced;
    for (size_t i = 0; i < 7; ++i) {
        const auto dep = AddEventTestDependency(ps, function, pipes);
        ASSERT_EQ(ps.InjectSetFlag(function, dep.first, synced), SUCCESS);
        ASSERT_EQ(ps.InjectWaitFlag(function, dep.second, synced), SUCCESS);
    }
    const auto boundary = ps.depOps_.size();
    for (size_t i = 0; i < boundary; ++i) {
        ps.depOps_[i].issued = true;
        synced.emplace_back(i * SEQUENCE_IDX, std::ref(*ps.oriOpList_[i]));
    }
    // Replay can issue a later producer before an earlier consumer. The
    // blocked producer is not itself a complete original-order drain boundary.
    const auto blocked = AddEventTestDependency(ps, function, {pipes.second, pipes.first});
    ps.depOps_[blocked.first].setPipe.clear();
    ps.depOps_[blocked.second].waitPipe.clear();
    ps.depOps_[blocked.second].setPipe = {blocked.first};
    ps.depOps_[blocked.first].waitPipe = {blocked.second};
    const PipeSync::PipePairEx suffixPipes{{PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED}, pipes.second};
    const auto suffix = AddEventTestDependency(ps, function, suffixPipes);
    ASSERT_EQ(ps.InjectSetFlag(function, suffix.first, synced), SUCCESS);
    ASSERT_EQ(ps.InjectWaitFlag(function, suffix.second, synced), SUCCESS);
    const auto suffixId = ps.eventState_.Find(suffix.first, suffix.second)->id;
    for (auto i : {suffix.first, suffix.second}) {
        ps.depOps_[i].issued = true;
        synced.emplace_back(i * SEQUENCE_IDX, std::ref(*ps.oriOpList_[i]));
    }
    ASSERT_EQ(ps.ReclaimIntraCoreEvents(function, blocked.second, pipes, synced), SUCCESS);
    ASSERT_EQ(ps.reclaimBarriers_.size(), 1U);
    const auto* barrier = ps.reclaimBarriers_.front().ops.front();
    const auto entry = std::find_if(synced.begin(), synced.end(),
                                    [barrier](const IndexOp& op) { return &op.second.get() == barrier; });
    ASSERT_NE(entry, synced.end());
    EXPECT_LT(entry->first, boundary * SEQUENCE_IDX - HALF_SEQUENCE_IDX);
    EXPECT_GT(entry->first, (boundary - 1) * SEQUENCE_IDX);
    EXPECT_EQ(ps.GetAvailableIntraEventIds(pipes, {}).size(), 7U);
    EXPECT_TRUE(ps.eventState_.HasActive(suffixId.resource));
    EXPECT_EQ(ps.eventState_.Find(suffix.first, suffix.second)->state, PipeSync::InstanceState::WaitInserted);
    // The synthetic backward dependency above only exercises boundary selection.
    // It cannot be emitted in original order: its wait would precede its set.
    std::sort(synced.begin(), synced.end(), [](const IndexOp& a, const IndexOp& b) { return a.first < b.first; });
    std::vector<Operation*> ordered;
    for (const auto& op : synced) {
        ordered.push_back(&op.second.get());
    }
    EXPECT_EQ(ps.ValidateEventReuse(ordered), SUCCESS);
    ordered.erase(std::remove(ordered.begin(), ordered.end(), barrier), ordered.end());
    EXPECT_EQ(ps.ValidateEventReuse(ordered), FAILED);
}

TEST_F(InsertSyncTest, LiteDrainCoversBothLogicalCoreLabels)
{
    NpuArchScope archScope(NPUArch::DAV_3113);
    auto funcs = SetupDebugFunc("LiteDrainLogicalLabels");
    auto& function = *funcs.second;
    PipeSync ps;
    const PipeSync::PipePairEx pipes{{PIPE_MTE2, CoreType::AIC, AIVCore::UNSPECIFIED},
                                     {PIPE_V, CoreType::AIV, AIVCore::AIV0}};
    std::vector<IndexOp> synced;
    for (size_t i = 0; i < 7; ++i) {
        const auto dep = AddEventTestDependency(ps, function, pipes);
        ASSERT_EQ(ps.InjectSetFlag(function, dep.first, synced), SUCCESS);
        ASSERT_EQ(ps.InjectWaitFlag(function, dep.second, synced), SUCCESS);
        for (auto index : {dep.first, dep.second}) {
            ps.depOps_[index].issued = true;
            synced.emplace_back(index * SEQUENCE_IDX, std::ref(*ps.oriOpList_[index]));
        }
    }
    const auto next = AddEventTestDependency(ps, function, pipes);
    ASSERT_TRUE(ps.GetAvailableIntraEventIds(pipes, {}).empty());
    ASSERT_EQ(ps.ReclaimIntraCoreEvents(function, next.first, pipes, synced), SUCCESS);
    ASSERT_EQ(ps.GetAvailableIntraEventIds(pipes, {}).size(), 7U);
    ASSERT_EQ(ps.InjectSetFlag(function, next.first, synced), SUCCESS);
    ASSERT_EQ(ps.InjectWaitFlag(function, next.second, synced), SUCCESS);
    std::sort(synced.begin(), synced.end(), [](const IndexOp& a, const IndexOp& b) { return a.first < b.first; });
    std::vector<Operation*> ordered;
    for (const auto& entry : synced) {
        ordered.push_back(&entry.second.get());
    }
    EXPECT_EQ(ps.ValidateEventReuse(ordered), SUCCESS);
    auto* barrier = ps.reclaimBarriers_.front().ops.front();
    ordered.erase(std::remove(ordered.begin(), ordered.end(), barrier), ordered.end());
    EXPECT_EQ(ps.ValidateEventReuse(ordered), FAILED);
}

TEST_F(InsertSyncTest, IntraDrainReclaimsConsumedEventsWithoutResettingPendingOrOtherCore)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("IntraDrainConsumedPrefix");
    auto& function = *funcs.second;
    PipeSync ps;
    const PipeSync::PipePairEx pipes{{PIPE_MTE2, CoreType::AIV, AIVCore::AIV0},
                                     {PIPE_MTE3, CoreType::AIV, AIVCore::AIV0}};
    const PipeSync::PipePairEx other{{PIPE_MTE2, CoreType::AIV, AIVCore::AIV1},
                                     {PIPE_MTE3, CoreType::AIV, AIVCore::AIV1}};
    std::vector<IndexOp> synced;
    const auto otherDep = AddEventTestDependency(ps, function, other);
    ASSERT_EQ(ps.InjectSetFlag(function, otherDep.first, synced), SUCCESS);
    ASSERT_EQ(ps.InjectWaitFlag(function, otherDep.second, synced), SUCCESS);
    const auto otherId = ps.eventState_.Find(otherDep.first, otherDep.second)->id;
    for (size_t i = 0; i < 7; ++i) {
        const auto dep = AddEventTestDependency(ps, function, pipes);
        ASSERT_EQ(ps.InjectSetFlag(function, dep.first, synced), SUCCESS);
        ASSERT_EQ(ps.InjectWaitFlag(function, dep.second, synced), SUCCESS);
    }
    const PipeSync::PipePairEx pendingPipes{{PIPE_V, CoreType::AIV, AIVCore::AIV0}, pipes.second};
    const auto pending = AddEventTestDependency(ps, function, pendingPipes);
    ASSERT_EQ(ps.InjectSetFlag(function, pending.first, synced), SUCCESS);
    const auto pendingId = ps.eventState_.Find(pending.first, pending.second)->id;
    const auto count = synced.size();
    EXPECT_EQ(ps.ReclaimIntraCoreEvents(function, pending.second, pipes, synced), FAILED);
    EXPECT_EQ(synced.size(), count);
    EXPECT_TRUE(ps.reclaimBarriers_.empty());
    for (size_t i = 0; i < pending.second; ++i) {
        ps.depOps_[i].issued = true;
        synced.emplace_back(i * SEQUENCE_IDX, std::ref(*ps.oriOpList_[i]));
    }
    ps.depOps_[pending.second].setPipe.push_back(2);
    ps.depOps_[2].waitPipe.push_back(pending.second);
    EXPECT_EQ(ps.ReclaimIntraCoreEvents(function, pending.second, pipes, synced), FAILED);
    EXPECT_TRUE(ps.reclaimBarriers_.empty());
    EXPECT_TRUE(ps.GetAvailableIntraEventIds(pipes, {}).empty());
    ps.depOps_[pending.second].setPipe.pop_back();
    ps.depOps_[2].waitPipe.pop_back();
    ASSERT_EQ(ps.ReclaimIntraCoreEvents(function, pending.second, pipes, synced), SUCCESS);
    ASSERT_EQ(ps.reclaimBarriers_.size(), 1U);
    ASSERT_EQ(ps.reclaimBarriers_.front().ops.size(), 1U);
    auto* barrier = ps.reclaimBarriers_.front().ops.front();
    EXPECT_EQ(barrier->GetOpcode(), Opcode::OP_BAR_ALL);
    EXPECT_EQ(barrier->syncQueue_.coreType_, CoreType::AIV);
    EXPECT_EQ(barrier->syncQueue_.setAivCore_, AIVCore::AIV0);
    EXPECT_EQ(ps.GetAvailableIntraEventIds(pipes, {}).size(), 7U);
    EXPECT_TRUE(ps.eventState_.HasActive(otherId.resource));
    EXPECT_TRUE(ps.eventState_.HasActive(pendingId.resource));
    EXPECT_EQ(ps.eventState_.Find(pending.first, pending.second)->state, PipeSync::InstanceState::Allocated);
    EXPECT_EQ(ps.depOps_[pending.second].waitPipe, std::vector<size_t>({pending.first}));
    ASSERT_EQ(ps.InjectWaitFlag(function, pending.second, synced), SUCCESS);
    const auto next = AddEventTestDependency(ps, function, pipes);
    ASSERT_EQ(ps.InjectSetFlag(function, next.first, synced), SUCCESS);
    ASSERT_EQ(ps.InjectWaitFlag(function, next.second, synced), SUCCESS);
    EXPECT_EQ(ps.eventState_.Find(next.first, next.second)->id.generation, 2U);
    std::sort(synced.begin(), synced.end(), [](const IndexOp& a, const IndexOp& b) { return a.first < b.first; });
    std::vector<Operation*> ordered;
    for (const auto& entry : synced) {
        ordered.push_back(&entry.second.get());
    }
    EXPECT_EQ(ps.ValidateEventReuse(ordered), SUCCESS);
    ordered.erase(std::remove(ordered.begin(), ordered.end(), barrier), ordered.end());
    EXPECT_EQ(ps.ValidateEventReuse(ordered), FAILED);
}

TEST_F(InsertSyncTest, IntraExhaustionWithUnmergeableCrossCoreBoundariesReturnsProofAndContinues)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto funcs = SetupDebugFunc("IntraExhaustionUnmergeablePrefix");
    auto& function = *funcs.second;
    PipeSync ps;
    std::vector<IndexOp> synced;
    // Each store is a CV boundary, so replay cannot move the next load
    // before that store. No consumption proof ever returns to AIV.MTE2.
    // This isolates the scheduler's exhaustion path without relying on UB
    // address assignment or the Python c2v2 kernel's graph construction.
    for (size_t stage = 0; stage < 8; ++stage) {
        const auto base = ps.depOps_.size();
        for (const auto& pipe : {PipeSync::PipeCoreRealEx{PIPE_MTE2, CoreType::AIV, AIVCore::AIV0},
                                 PipeSync::PipeCoreRealEx{PIPE_MTE3, CoreType::AIV, AIVCore::AIV0},
                                 PipeSync::PipeCoreRealEx{PIPE_MTE1, CoreType::AIC, AIVCore::UNSPECIFIED}}) {
            const auto idx = ps.depOps_.size();
            ps.depOps_.emplace_back(idx, PipeSync::PipeCore{pipe.pipe, pipe.pipe, pipe.core, pipe.aivCore});
            ps.oriOpList_.push_back(&IRBuilder().CreateTensorOpStmt(function, Opcode::OP_VEC_DUP, {}, {}));
        }
        ps.depOps_[base].setPipe = {base + 1};
        ps.depOps_[base + 1].waitPipe = {base};
        ps.depOps_[base + 1].setPipe = {base + 2};
        ps.depOps_[base + 2].waitPipe = {base + 1};
    }
    // EnqueueOp places non-executable views in the output log before any
    // scheduling. Such suffix entries are not issued hardware instructions.
    const auto viewIdx = ps.depOps_.size();
    ps.depOps_.emplace_back(viewIdx, PipeSync::PipeCore{PIPE_S, PIPE_S, CoreType::AIV, AIVCore::AIV0});
    ps.oriOpList_.push_back(&IRBuilder().CreateTensorOpStmt(function, Opcode::OP_VIEW, {}, {}));
    const auto original = ps.oriOpList_;
    for (auto& op : ps.depOps_) {
        ps.EnqueueOp(op, original, synced);
    }
    const PipeSync::PipePairEx pipes{{PIPE_MTE2, CoreType::AIV, AIVCore::AIV0},
                                     {PIPE_MTE3, CoreType::AIV, AIVCore::AIV0}};
    auto plan = ps.depOps_;
    std::vector<size_t> order;
    EXPECT_FALSE(ps.MergeDependenciesForReplay(plan, pipes, order));
    ASSERT_EQ(ps.IssueOp(function, original, synced), SUCCESS);
    EXPECT_TRUE(ps.reclaimBarriers_.empty());
    for (const auto& queue : ps.issueState_) {
        EXPECT_EQ(queue.currOp, queue.ops.size());
    }
    size_t crossCoreInstances = 0;
    for (const auto& instance : ps.eventState_.Instances()) {
        if (!instance.intraCore) {
            ++crossCoreInstances;
            EXPECT_EQ(instance.state, PipeSync::InstanceState::WaitInserted);
            EXPECT_FALSE(ps.eventState_.HasActive(instance.id.resource));
        }
    }
    EXPECT_EQ(crossCoreInstances, 8U);
    std::sort(synced.begin(), synced.end(), [](const IndexOp& a, const IndexOp& b) { return a.first < b.first; });
    std::vector<Operation*> ordered;
    std::vector<Operation*> business;
    for (const auto& entry : synced) {
        auto* op = &entry.second.get();
        ordered.push_back(op);
        if (op->GetOpcode() == Opcode::OP_VEC_DUP || op->GetOpcode() == Opcode::OP_VIEW) {
            business.push_back(op);
        }
        EXPECT_NE(op->GetOpcode(), Opcode::OP_FFTS_CROSS_CORE_SYNC);
    }
    EXPECT_EQ(business, original);
    EXPECT_EQ(ps.ValidateEventReuse(ordered), SUCCESS);
}
TEST_F(InsertSyncTest, A5ExhaustionSynchronizesOnlyTheSelectedCorePair)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    for (auto vector : {AIVCore::AIV0, AIVCore::AIV1}) {
        for (bool cubeFirst : {true, false}) {
            auto [root, leaf] = SetupDebugFunc("A5LocalBarrier");
            PipeSync ps;
            ps.InitCVEventIdQ();
            const PipeSync::PipeCore source = cubeFirst ?
                                                  PipeSync::PipeCore(PIPE_FIX, PIPE_FIX, CoreType::AIC,
                                                                     AIVCore::UNSPECIFIED) :
                                                  PipeSync::PipeCore(PIPE_MTE3, PIPE_MTE3, CoreType::AIV, vector);
            const PipeSync::PipeCore target = cubeFirst ?
                                                  PipeSync::PipeCore(PIPE_MTE2, PIPE_MTE2, CoreType::AIV, vector) :
                                                  PipeSync::PipeCore(PIPE_MTE1, PIPE_MTE1, CoreType::AIC,
                                                                     AIVCore::UNSPECIFIED);
            constexpr size_t boundary = 16;
            for (size_t i = 0; i < boundary * 2 + 2; ++i) {
                auto& op = IRBuilder().CreateTensorOpStmt(*leaf, Opcode::OP_ADD, {}, {});
                const auto& pipe = i <= boundary ? source : target;
                op.SetAIVCore(pipe.aivCore);
                ps.oriOpList_.push_back(&op);
                ps.depOps_.emplace_back(i, pipe);
            }
            for (size_t i = 0; i <= boundary; ++i) {
                ASSERT_EQ(ps.AddOpDep(ps.depOps_[i], ps.depOps_[boundary + 1 + i]), SUCCESS);
            }
            std::vector<IndexOp> synced;
            for (size_t i = 0; i <= boundary; ++i) {
                ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, i, synced), SUCCESS);
                if (i < boundary) {
                    EXPECT_TRUE(ps.forceSyncOps_.empty());
                }
            }
            size_t bars = 0;
            size_t cvSets = 0;
            size_t cvWaits = 0;
            std::vector<Operation*> forceSyncs;
            for (const auto& entry : synced) {
                const auto& op = entry.second.get();
                EXPECT_NE(op.GetOpcode(), Opcode::OP_FFTS_CROSS_CORE_SYNC);
                EXPECT_NE(op.GetOpcode(), Opcode::OP_WAIT_FLAG_DEV);
                bars += op.GetOpcode() == Opcode::OP_BAR_ALL;
                cvSets += op.GetOpcode() == Opcode::OP_CV_SYNC_SRC;
                cvWaits += op.GetOpcode() == Opcode::OP_CV_SYNC_DST;
                if (ps.forceSyncOps_.count(&entry.second.get()) != 0) {
                    forceSyncs.push_back(&entry.second.get());
                    EXPECT_TRUE(op.GetAIVCore() == vector || op.GetAIVCore() == AIVCore::UNSPECIFIED);
                    EXPECT_LT(entry.first, boundary * SEQUENCE_IDX - HALF_SEQUENCE_IDX);
                }
            }
            EXPECT_EQ(bars, 2U);
            EXPECT_EQ(cvSets, 3U); // Two handshake sets and the current operation's set.
            EXPECT_EQ(cvWaits, 2U);
            EXPECT_EQ(ps.NoWaitCVPairs_.size(), 1U);
            EXPECT_EQ(ps.setWaitPairMap_.size(), 1U);
            const PipeSync::PipeCoreRealEx cube(PIPE_FIX, CoreType::AIC);
            const auto otherVector = vector == AIVCore::AIV0 ? AIVCore::AIV1 : AIVCore::AIV0;
            const size_t otherQueue = otherVector == AIVCore::AIV0 ? 0 : 1;
            EXPECT_EQ(ps.crossCoreFreeEventId_[cube][otherQueue].size(), boundary);
            ASSERT_EQ(forceSyncs.size(), 6U);
            EXPECT_EQ(forceSyncs[0]->GetOpcode(), Opcode::OP_BAR_ALL);
            EXPECT_EQ(forceSyncs[0]->syncQueue_.coreType_, CoreType::AIC);
            EXPECT_EQ(forceSyncs[1]->GetOpcode(), Opcode::OP_BAR_ALL);
            EXPECT_EQ(forceSyncs[1]->syncQueue_.coreType_, CoreType::AIV);
            const auto reverseSender = cubeFirst ? CoreType::AIV : CoreType::AIC;
            const auto oldSender = cubeFirst ? CoreType::AIC : CoreType::AIV;
            for (size_t i = 2; i < forceSyncs.size(); ++i) {
                const auto& sync = forceSyncs[i]->syncQueue_;
                EXPECT_EQ(sync.eventId_, vector == AIVCore::AIV0 ? 0 : 16);
                EXPECT_EQ(sync.coreType_, i < 4 ? reverseSender : oldSender);
                if (i % 2 == 0) {
                    EXPECT_EQ(forceSyncs[i]->GetOpcode(), Opcode::OP_CV_SYNC_SRC);
                    EXPECT_EQ(sync.pipeId_, sync.coreType_ == CoreType::AIC ? PIPE_FIX : PIPE_MTE3);
                } else {
                    EXPECT_EQ(forceSyncs[i]->GetOpcode(), Opcode::OP_CV_SYNC_DST);
                    EXPECT_EQ(sync.trigPipeId_, PIPE_S);
                }
            }
            const int handshakeId = vector == AIVCore::AIV0 ? 0 : 16;
            const auto finalReceiver = cubeFirst ? CoreType::AIV : CoreType::AIC;
            for (const auto& pipe : ps.cvPipeCoreEx) {
                if (pipe.core != CoreType::AIC && pipe.aivCore != vector) {
                    continue;
                }
                const auto& queue = ps.crossCoreFreeEventId_[pipe][pipe.core == CoreType::AIC ? 1 - otherQueue : 0];
                EXPECT_EQ(std::count(queue.begin(), queue.end(), handshakeId),
                          pipe.core == finalReceiver && pipe.pipe == PIPE_S ? 1 : 0);
            }
        }
    }
}

static size_t AppendLocalBarrierTestOp(PipeSync& ps, Function& function, const PipeSync::PipeCore& pipe)
{
    auto& op = IRBuilder().CreateTensorOpStmt(function, Opcode::OP_ADD, {}, {});
    op.SetAIVCore(pipe.aivCore);
    const size_t idx = ps.oriOpList_.size();
    ps.oriOpList_.push_back(&op);
    ps.depOps_.emplace_back(idx, pipe);
    return idx;
}

TEST_F(InsertSyncTest, A5BarrierPreservesConsumedEdgesOtherPoolAndIntraCoreState)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("A5BarrierPreservesState");
    PipeSync ps;
    ps.InitCVEventIdQ();
    const PipeSync::PipeCore cube(PIPE_FIX, PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED);
    const PipeSync::PipeCore vector0(PIPE_MTE2, PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCore vector1(PIPE_MTE2, PIPE_MTE2, CoreType::AIV, AIVCore::AIV1);
    AppendLocalBarrierTestOp(ps, *leaf, cube);
    AppendLocalBarrierTestOp(ps, *leaf, vector0); // Consume ID 0 before exhaustion.
    for (size_t i = 0; i < 16; ++i) {
        AppendLocalBarrierTestOp(ps, *leaf, cube);
    }
    for (size_t i = 0; i < 16; ++i) {
        AppendLocalBarrierTestOp(ps, *leaf, vector0);
    }
    const size_t otherConsumer = AppendLocalBarrierTestOp(ps, *leaf, vector1);
    ASSERT_EQ(ps.AddOpDep(ps.depOps_[0], ps.depOps_[1]), SUCCESS);
    ASSERT_EQ(ps.AddOpDep(ps.depOps_[0], ps.depOps_[otherConsumer]), SUCCESS);
    for (size_t i = 2; i < 18; ++i) {
        ASSERT_EQ(ps.AddOpDep(ps.depOps_[i], ps.depOps_[i + 16]), SUCCESS);
    }
    const PipeSync::PipePairEx intra{{PIPE_MTE2, CoreType::AIV, AIVCore::AIV1}, {PIPE_V, CoreType::AIV, AIVCore::AIV1}};
    ps.freeEventId_[intra] = {2, 4, 6};
    std::vector<IndexOp> synced;
    ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, 0, synced), SUCCESS);
    ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, 1, synced), SUCCESS);
    Operation* consumedSet = nullptr;
    Operation* consumedWait = nullptr;
    Operation* otherSet = nullptr;
    for (auto& entry : synced) {
        auto& op = entry.second.get();
        if (op.GetOpcode() == Opcode::OP_CV_SYNC_DST) {
            consumedWait = &op;
        } else if (op.GetOpcode() == Opcode::OP_CV_SYNC_SRC) {
            (op.syncQueue_.eventId_ == 0 ? consumedSet : otherSet) = &op;
        }
    }
    ASSERT_NE(consumedSet, nullptr);
    ASSERT_NE(consumedWait, nullptr);
    ASSERT_NE(otherSet, nullptr);
    const PipeSync::EventResource intraArrival{
        1, {CoreType::AIV, AIVCore::AIV1}, {CoreType::AIV, AIVCore::AIV1}, PIPE_MTE2, PIPE_V};
    ps.syncArriveStatus[intra.second].insert(intraArrival);
    const PipeSync::PipeCoreRealEx cubePipe(PIPE_FIX, CoreType::AIC);
    ps.crossCoreConsumed_[cubePipe][16] = 1;
    for (size_t i = 2; i < 18; ++i) {
        ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, i, synced), SUCCESS);
    }
    EXPECT_FALSE(consumedSet->IsDeleted());
    EXPECT_FALSE(consumedWait->IsDeleted());
    EXPECT_FALSE(otherSet->IsDeleted());
    EXPECT_EQ(ps.waitOpMap.at(consumedWait), ps.oriOpList_[0]);
    EXPECT_EQ(ps.setWaitPairMap_.at({0, otherConsumer}), 16);
    EXPECT_EQ(ps.NoWaitCVPairs_.size(), 2U);
    EXPECT_EQ(ps.freeEventId_.at(intra), (std::deque<int>{2, 4, 6}));
    EXPECT_EQ(ps.syncArriveStatus[intra.second].count(intraArrival), 1U);
    EXPECT_EQ(ps.crossCoreConsumed_[cubePipe][16], 1U);
    EXPECT_EQ(ps.crossCoreGenerations_[0], 3U);
    EXPECT_FALSE(ps.CanReuseCrossCoreEventId(0, ps.crossCoreConsumed_[cubePipe]));
    const PipeSync::PipeCoreRealEx receivingS(PIPE_S, CoreType::AIV, AIVCore::AIV0);
    EXPECT_TRUE(ps.CanReuseCrossCoreEventId(0, ps.crossCoreConsumed_[receivingS]));
    EXPECT_EQ(ps.crossCoreSnapshots_.at({0, otherConsumer}).consumed[0], 0U);
    for (const auto& [pipe, events] : ps.syncArriveStatus) {
        (void)pipe;
        for (const auto& event : events) {
            EXPECT_FALSE(event.srcCore.first != event.dstCore.first && event.eventId < 16);
        }
    }
}

TEST_F(InsertSyncTest, A5ConsecutiveBarriersKeepCurrentOutputSets)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("A5ConsecutiveBarriers");
    PipeSync ps;
    ps.InitCVEventIdQ();
    const PipeSync::PipeCore cube(PIPE_FIX, PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED);
    const PipeSync::PipeCore vector(PIPE_MTE2, PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    constexpr size_t producerCount = 32;
    for (size_t i = 0; i < producerCount * 2; ++i) {
        AppendLocalBarrierTestOp(ps, *leaf, i < producerCount ? cube : vector);
    }
    for (size_t i = 0; i < producerCount; ++i) {
        ASSERT_EQ(ps.AddOpDep(ps.depOps_[i], ps.depOps_[i + producerCount]), SUCCESS);
    }
    std::vector<IndexOp> synced;
    for (size_t i = 0; i < producerCount; ++i) {
        ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, i, synced), SUCCESS);
    }
    EXPECT_EQ(ps.forceSyncOps_.size(), 12U);
    ASSERT_EQ(ps.NoWaitCVPairs_.size(), 1U);
    EXPECT_EQ(ps.NoWaitCVPairs_[0].first, std::make_pair(producerCount - 1, producerCount * 2 - 1));
    EXPECT_FALSE(ps.NoWaitCVPairs_[0].second->IsDeleted());
    for (size_t i = producerCount; i < producerCount * 2; ++i) {
        ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, i, synced), SUCCESS);
    }
    EXPECT_TRUE(ps.NoWaitCVPairs_.empty());
}

TEST_F(InsertSyncTest, A5ReservesAllCurrentOutputsBeforeItsInputWaits)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("A5WholeOperationCapacity");
    PipeSync ps;
    ps.InitCVEventIdQ();
    const PipeSync::PipeCore cube(PIPE_FIX, PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED);
    const PipeSync::PipeCore vector(PIPE_MTE2, PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    for (size_t i = 0; i < 16; ++i) {
        AppendLocalBarrierTestOp(ps, *leaf, cube);
    }
    ps.depOps_[15].selfPipeCore.pipeStart = PIPE_MTE1;
    for (size_t i = 0; i < 16; ++i) {
        AppendLocalBarrierTestOp(ps, *leaf, vector);
    }
    AppendLocalBarrierTestOp(ps, *leaf, {PIPE_V, PIPE_V, CoreType::AIV, AIVCore::AIV0});
    ASSERT_EQ(ps.AddOpDep(ps.depOps_[0], ps.depOps_[15]), SUCCESS);
    for (size_t i = 0; i < 16; ++i) {
        ASSERT_EQ(ps.AddOpDep(ps.depOps_[i], ps.depOps_[i + 16]), SUCCESS);
    }
    ASSERT_EQ(ps.AddOpDep(ps.depOps_[15], ps.depOps_[32]), SUCCESS);
    std::vector<IndexOp> synced;
    for (size_t i = 0; i < 15; ++i) {
        ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, i, synced), SUCCESS);
    }
    EXPECT_TRUE(ps.forceSyncOps_.empty());
    ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, 15, synced), SUCCESS);
    EXPECT_EQ(ps.forceSyncOps_.size(), 6U);
    EXPECT_EQ(ps.NoWaitCVPairs_.size(), 2U);
    EXPECT_EQ(ps.depOps_[15].setPipe.size(), 2U);
    EXPECT_EQ(ps.setWaitPairMap_.at({15, 31}), 1);
    EXPECT_EQ(ps.setWaitPairMap_.at({15, 32}), 2);
    bool hasInputWait = false;
    for (const auto& entry : synced) {
        const auto& op = entry.second.get();
        if (op.GetOpcode() == Opcode::OP_SYNC_DST) {
            hasInputWait = true;
            EXPECT_GT(entry.first, 15 * SEQUENCE_IDX - HALF_SEQUENCE_IDX);
        }
        if (ps.forceSyncOps_.count(&entry.second.get()) != 0) {
            EXPECT_LT(entry.first, 15 * SEQUENCE_IDX - HALF_SEQUENCE_IDX);
        }
    }
    EXPECT_TRUE(hasInputWait);
}

TEST_F(InsertSyncTest, A5BothRequestedPoolsExhaustedUseOneGlobalBarrier)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("A5BothPoolsExhausted");
    PipeSync ps;
    ps.InitCVEventIdQ();
    const PipeSync::PipeCore cube(PIPE_FIX, PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED);
    for (size_t i = 0; i < 17; ++i) {
        AppendLocalBarrierTestOp(ps, *leaf, cube);
    }
    for (auto vector : {AIVCore::AIV0, AIVCore::AIV1}) {
        for (size_t i = 0; i < 17; ++i) {
            const auto consumer = AppendLocalBarrierTestOp(ps, *leaf, {PIPE_MTE2, PIPE_MTE2, CoreType::AIV, vector});
            ASSERT_EQ(ps.AddOpDep(ps.depOps_[i], ps.depOps_[consumer]), SUCCESS);
        }
    }
    std::vector<IndexOp> synced;
    for (size_t i = 0; i < 17; ++i) {
        ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, i, synced), SUCCESS);
        if (i < 16) {
            EXPECT_TRUE(ps.forceSyncOps_.empty());
        }
    }
    EXPECT_EQ(ps.forceSyncOps_.size(), 12U);
    EXPECT_EQ(ps.NoWaitCVPairs_.size(), 2U);
    EXPECT_EQ(ps.setWaitPairMap_.at({16, 33}), 0);
    EXPECT_EQ(ps.setWaitPairMap_.at({16, 50}), 16);
    size_t globalSets = 0;
    size_t globalWaits = 0;
    for (auto* op : ps.forceSyncOps_) {
        globalSets += op->GetOpcode() == Opcode::OP_FFTS_CROSS_CORE_SYNC;
        globalWaits += op->GetOpcode() == Opcode::OP_WAIT_FLAG_DEV;
    }
    EXPECT_EQ(globalSets, 3U);
    EXPECT_EQ(globalWaits, 3U);
    for (size_t i = 17; i < ps.oriOpList_.size(); ++i) {
        ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, i, synced), SUCCESS);
    }
    EXPECT_TRUE(ps.NoWaitCVPairs_.empty());
}

TEST_F(InsertSyncTest, A5ExhaustedUnrequestedPoolDoesNotForceBarrier)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    for (auto exhaustedVector : {AIVCore::AIV0, AIVCore::AIV1}) {
        auto [root, leaf] = SetupDebugFunc("A5UnrequestedPool");
        PipeSync ps;
        ps.InitCVEventIdQ();
        const PipeSync::PipeCore cube(PIPE_FIX, PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED);
        for (size_t i = 0; i < 17; ++i) {
            AppendLocalBarrierTestOp(ps, *leaf, cube);
        }
        const auto otherVector = exhaustedVector == AIVCore::AIV0 ? AIVCore::AIV1 : AIVCore::AIV0;
        for (size_t i = 0; i < 17; ++i) {
            const auto vector = i < 16 ? exhaustedVector : otherVector;
            const auto consumer = AppendLocalBarrierTestOp(ps, *leaf, {PIPE_MTE2, PIPE_MTE2, CoreType::AIV, vector});
            ASSERT_EQ(ps.AddOpDep(ps.depOps_[i], ps.depOps_[consumer]), SUCCESS);
        }
        std::vector<IndexOp> synced;
        for (size_t i = 0; i < ps.oriOpList_.size(); ++i) {
            ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, i, synced), SUCCESS);
            EXPECT_TRUE(ps.forceSyncOps_.empty());
        }
        EXPECT_TRUE(ps.NoWaitCVPairs_.empty());
    }
}

TEST_F(InsertSyncTest, A5RejectsDemandBeyondOneCrossCorePool)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("A5CrossCoreDemandExceedsCapacity");
    PipeSync ps;
    ps.InitCVEventIdQ();
    const auto producer = AppendLocalBarrierTestOp(ps, *leaf,
                                                   {PIPE_FIX, PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED});
    PipeSync::CrossCoreBarrierPlan plan;
    EXPECT_EQ(ps.PlanCrossCoreBarriers(producer, {PipeSync::CROSS_CORE_EVENT_NUM + 1, 0}, plan), FAILED);
    EXPECT_EQ(plan.localCount, 0U);
    EXPECT_FALSE(plan.globalBarrier);
}

TEST_F(InsertSyncTest, A5RejectsBarrierWithIncompleteEmittedPrefix)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("A5IncompleteCrossCoreBarrierPrefix");
    PipeSync ps;
    ps.InitCVEventIdQ();
    const size_t boundary = PipeSync::CROSS_CORE_EVENT_NUM;
    for (size_t i = 0; i <= boundary; ++i) {
        AppendLocalBarrierTestOp(ps, *leaf, {PIPE_FIX, PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED});
    }
    for (size_t i = 0; i <= boundary; ++i) {
        const auto consumer = AppendLocalBarrierTestOp(ps, *leaf, {PIPE_MTE2, PIPE_MTE2, CoreType::AIV, AIVCore::AIV0});
        ASSERT_EQ(ps.AddOpDep(ps.depOps_[i], ps.depOps_[consumer]), SUCCESS);
    }
    std::vector<IndexOp> synced;
    for (size_t i = 0; i < boundary; ++i) {
        ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, i, synced), SUCCESS);
    }
    ASSERT_FALSE(synced.empty());
    ASSERT_EQ(&synced.front().second.get(), ps.oriOpList_.front());
    synced.erase(synced.begin());
    const auto prefixSize = synced.size();
    EXPECT_EQ(ps.EnsureCrossCoreCapacity(*leaf, boundary, synced), FAILED);
    EXPECT_EQ(synced.size(), prefixSize);
    EXPECT_EQ(ps.NoWaitCVPairs_.size(), boundary);
    EXPECT_TRUE(ps.forceSyncOps_.empty());
}

TEST_F(InsertSyncTest, A5RejectsEmptyCrossCoreDrain)
{
    auto [root, leaf] = SetupDebugFunc("A5EmptyCrossCoreDrain");
    PipeSync ps;
    AppendLocalBarrierTestOp(ps, *leaf, {PIPE_FIX, PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED});
    const std::vector<IndexOp> synced{{0, std::ref(*ps.oriOpList_.front())}};
    EXPECT_EQ(ps.RecordCrossCoreDrain(1, synced, {}), FAILED);
    EXPECT_TRUE(ps.reclaimBarriers_.empty());
}

TEST_F(InsertSyncTest, A5ExhaustionWithBatchedIssuingAndFuturePassiveOperations)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    for (auto vector : {AIVCore::AIV0, AIVCore::AIV1}) {
        auto [root, leaf] = SetupDebugFunc("A5ProductionExhaustion");
        IRBuilder builder;
        std::vector<Operation*> producers;
        for (size_t i = 0; i < 17; ++i) {
            producers.push_back(&builder.CreateTensorOpStmt(*leaf, Opcode::OP_L1_TO_L0A, {}, {}));
        }
        auto& futureNop = builder.CreateTensorOpStmt(*leaf, Opcode::OP_NOP, {}, {});
        auto& futureView = builder.CreateTensorOpStmt(*leaf, Opcode::OP_VIEW, {}, {});
        for (size_t i = 0; i < producers.size(); ++i) {
            auto& consumer = builder.CreateTensorOpStmt(*leaf, Opcode::OP_CAST, {}, {});
            consumer.SetAIVCore(vector);
            auto token = builder.CreateTokenVar(ir::Span::Unknown());
            producers[i]->result_token_ = {token};
            consumer.tokens_.push_back(token);
            auto& dependencies = leaf->GetVarDependency();
            dependencies.AddProducer(token, std::static_pointer_cast<const ir::Stmt>(leaf->operations_[i]));
            dependencies.AddConsumer(token, std::static_pointer_cast<const ir::Stmt>(leaf->operations_.back()));
        }
        PipeSync ps;
        std::vector<Operation*> synced;
        ASSERT_EQ(ps.InsertSync(*leaf, synced), SUCCESS);
        EXPECT_EQ(ps.forceSyncOps_.size(), 6U);
        EXPECT_EQ(std::count(synced.begin(), synced.end(), &futureNop), 1);
        EXPECT_EQ(std::count(synced.begin(), synced.end(), &futureView), 1);
        EXPECT_TRUE(ps.NoWaitCVPairs_.empty());
        auto boundary = std::find(synced.begin(), synced.end(), producers.back());
        ASSERT_NE(boundary, synced.end());
        for (auto* force : ps.forceSyncOps_) {
            EXPECT_LT(std::find(synced.begin(), synced.end(), force), boundary);
        }
    }
}

TEST_F(InsertSyncTest, A5LegacyFallbackPrecedesCurrentInputWaits)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("A5LegacyBoundary");
    PipeSync ps;
    ps.InitCVEventIdQ();
    const PipeSync::PipeCore cube(PIPE_FIX, PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED);
    const PipeSync::PipeCore vector(PIPE_MTE2, PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    AppendLocalBarrierTestOp(ps, *leaf, cube);
    AppendLocalBarrierTestOp(ps, *leaf, {PIPE_MTE1, PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED});
    AppendLocalBarrierTestOp(ps, *leaf, vector);
    ASSERT_EQ(ps.AddOpDep(ps.depOps_[0], ps.depOps_[1]), SUCCESS);
    ASSERT_EQ(ps.AddOpDep(ps.depOps_[1], ps.depOps_[2]), SUCCESS);
    std::vector<IndexOp> synced;
    ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, 0, synced), SUCCESS);
    // No occupied slot has a known sender, so the local protocol cannot use an existing ID.
    ps.crossCoreFreeEventId_[{PIPE_FIX, CoreType::AIC}][0].clear();
    ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, 1, synced), SUCCESS);
    EXPECT_EQ(ps.forceSyncOps_.size(), 12U);
    size_t legacySets = 0;
    for (const auto& entry : synced) {
        auto& op = entry.second.get();
        legacySets += op.GetOpcode() == Opcode::OP_FFTS_CROSS_CORE_SYNC;
        if (ps.forceSyncOps_.count(&op) != 0) {
            EXPECT_LT(entry.first, SEQUENCE_IDX - HALF_SEQUENCE_IDX);
        } else if (op.GetOpcode() == Opcode::OP_SYNC_DST) {
            EXPECT_GT(entry.first, SEQUENCE_IDX - HALF_SEQUENCE_IDX);
        }
    }
    EXPECT_EQ(legacySets, 3U);
    EXPECT_EQ(ps.NoWaitCVPairs_.size(), 1U);
}

TEST_F(InsertSyncTest, A5AllocationFailureDoesNotDrainCommittedOutputs)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("A5ReservationFailure");
    PipeSync ps;
    ps.InitCVEventIdQ();
    const PipeSync::PipeCore cube(PIPE_FIX, PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED);
    const PipeSync::PipeCore vector(PIPE_MTE2, PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    for (size_t i = 0; i < 16; ++i) {
        AppendLocalBarrierTestOp(ps, *leaf, cube);
    }
    for (size_t i = 0; i < 16; ++i) {
        const auto consumer = AppendLocalBarrierTestOp(ps, *leaf, vector);
        ASSERT_EQ(ps.AddOpDep(ps.depOps_[i], ps.depOps_[consumer]), SUCCESS);
    }
    std::vector<IndexOp> synced;
    for (size_t i = 0; i < 16; ++i) {
        ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, i, synced), SUCCESS);
    }
    const auto emitted = synced.size();
    const auto pending = ps.NoWaitCVPairs_;
    const auto pairMap = ps.setWaitPairMap_;
    // Simulate a broken reservation after emission; allocation must not cancel prior outputs.
    int eventId = -1;
    EXPECT_EQ(ps.GetEventId({{PIPE_FIX, CoreType::AIC}, {PIPE_MTE2, CoreType::AIV, AIVCore::AIV0}}, eventId, 15, 31,
                            synced, *leaf),
              FAILED);
    EXPECT_EQ(synced.size(), emitted);
    EXPECT_EQ(ps.NoWaitCVPairs_, pending);
    EXPECT_EQ(ps.setWaitPairMap_, pairMap);
    EXPECT_TRUE(ps.forceSyncOps_.empty());
}

TEST_F(InsertSyncTest, A5CrossCoreReuseFollowsConsumedBidirectionalChain)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    for (auto vector : {AIVCore::AIV0, AIVCore::AIV1}) {
        auto [root, leaf] = SetupDebugFunc("A5ConsumedCrossCoreChain");
        PipeSync ps;
        ps.InitCVEventIdQ();
        const PipeSync::PipeCore cube(PIPE_FIX, PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED);
        const PipeSync::PipeCore vec(PIPE_MTE2, PIPE_MTE2, CoreType::AIV, vector);
        constexpr size_t count = 40;
        for (size_t i = 0; i < count; ++i) {
            AppendLocalBarrierTestOp(ps, *leaf, i % 2 == 0 ? cube : vec);
            if (i != 0) {
                ASSERT_EQ(ps.AddOpDep(ps.depOps_[i - 1], ps.depOps_[i]), SUCCESS);
            }
        }
        std::vector<IndexOp> synced;
        for (size_t i = 0; i < count; ++i) {
            ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, i, synced), SUCCESS);
        }
        EXPECT_TRUE(ps.forceSyncOps_.empty());
        EXPECT_TRUE(ps.NoWaitCVPairs_.empty());
        EXPECT_EQ(std::count_if(
                      synced.begin(), synced.end(),
                      [](const IndexOp& entry) { return entry.second.get().GetOpcode() == Opcode::OP_CV_SYNC_SRC; }),
                  count - 1);
        // A physical slot is shared by both directions; reuse needs its previous wait.
        EXPECT_EQ(ps.setWaitPairMap_.at({0, 1}), ps.setWaitPairMap_.at({16, 17}));
    }
}

TEST_F(InsertSyncTest, A5CrossCoreSnapshotDoesNotObserveLaterSenderWaits)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("A5FrozenCrossCoreProof");
    PipeSync ps;
    ps.InitCVEventIdQ();
    const PipeSync::PipeCore cube(PIPE_FIX, PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED);
    const PipeSync::PipeCore vec(PIPE_MTE2, PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    for (const auto& pipe : {vec, cube, vec, cube}) {
        AppendLocalBarrierTestOp(ps, *leaf, pipe);
    }
    ASSERT_EQ(ps.AddOpDep(ps.depOps_[0], ps.depOps_[3]), SUCCESS);
    ASSERT_EQ(ps.AddOpDep(ps.depOps_[1], ps.depOps_[2]), SUCCESS);
    std::vector<IndexOp> synced;
    for (size_t i = 0; i < 3; ++i) {
        ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, i, synced), SUCCESS);
    }
    const auto laterId = ps.setWaitPairMap_.at({1, 2});
    const PipeSync::PipeCoreRealEx vectorPipe(PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCoreRealEx cubePipe(PIPE_FIX, CoreType::AIC);
    EXPECT_EQ(ps.crossCoreConsumed_[vectorPipe][laterId], 1U);
    ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, 3, synced), SUCCESS);
    EXPECT_EQ(ps.crossCoreConsumed_[cubePipe][laterId], 0U);
    EXPECT_TRUE(ps.crossCoreSnapshots_.empty());
}

TEST_F(InsertSyncTest, A5CrossCoreOldGenerationAndOtherPipesCannotAuthorizeReuse)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("A5CrossCoreGeneration");
    PipeSync ps;
    ps.InitCVEventIdQ();
    const PipeSync::PipeCore cube(PIPE_FIX, PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED);
    const PipeSync::PipeCore vec(PIPE_MTE2, PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCore vectorAlu(PIPE_V, PIPE_V, CoreType::AIV, AIVCore::AIV0);
    for (const auto& pipe : {cube, vec, vectorAlu, cube}) {
        AppendLocalBarrierTestOp(ps, *leaf, pipe);
    }
    ASSERT_EQ(ps.AddOpDep(ps.depOps_[0], ps.depOps_[1]), SUCCESS);
    ASSERT_EQ(ps.AddOpDep(ps.depOps_[1], ps.depOps_[2]), SUCCESS);
    ASSERT_EQ(ps.AddOpDep(ps.depOps_[1], ps.depOps_[3]), SUCCESS);
    const PipeSync::PipeCoreRealEx vectorPipe(PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCoreRealEx aluPipe(PIPE_V, CoreType::AIV, AIVCore::AIV0);
    ps.freeEventId_[{vectorPipe, aluPipe}] = {1};
    std::vector<IndexOp> synced;
    ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, 0, synced), SUCCESS);
    // Select the consumed slot so that the earlier intra-core set carries its old generation.
    ps.crossCoreFreeEventId_[vectorPipe][0] = {0};
    ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, 1, synced), SUCCESS);
    EXPECT_EQ(ps.setWaitPairMap_.at({1, 3}), 0);
    EXPECT_EQ(ps.crossCoreGenerations_[0], 2U);
    ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, 2, synced), SUCCESS);
    EXPECT_EQ(ps.crossCoreConsumed_[aluPipe][0], 1U);
    EXPECT_FALSE(ps.CanReuseCrossCoreEventId(0, ps.crossCoreConsumed_[aluPipe]));
    EXPECT_FALSE(ps.CanReuseCrossCoreEventId(0, ps.crossCoreConsumed_[vectorPipe]));
    const PipeSync::PipeCoreRealEx otherVector(PIPE_MTE2, CoreType::AIV, AIVCore::AIV1);
    EXPECT_FALSE(ps.CanReuseCrossCoreEventId(0, ps.crossCoreConsumed_[otherVector]));
    ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, 3, synced), SUCCESS);
    EXPECT_TRUE(ps.CanReuseCrossCoreEventId(0, ps.crossCoreConsumed_[{PIPE_FIX, CoreType::AIC}]));
    EXPECT_TRUE(ps.crossCoreSnapshots_.empty());
}

TEST_F(InsertSyncTest, A5CrossCoreInputProofDoesNotCrossMultiPipeOperation)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("A5CrossCoreMultiPipeProof");
    PipeSync ps;
    ps.InitCVEventIdQ();
    const PipeSync::PipeCore cube(PIPE_FIX, PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED);
    const PipeSync::PipeCore vec(PIPE_MTE2, PIPE_MTE3, CoreType::AIV, AIVCore::AIV0);
    AppendLocalBarrierTestOp(ps, *leaf, cube);
    AppendLocalBarrierTestOp(ps, *leaf, vec);
    ASSERT_EQ(ps.AddOpDep(ps.depOps_[0], ps.depOps_[1]), SUCCESS);
    std::vector<IndexOp> synced;
    ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, 0, synced), SUCCESS);
    EXPECT_EQ(ps.ProjectCrossCoreConsumption(1)[0], 0U);
    ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, 1, synced), SUCCESS);
    const PipeSync::PipeCoreRealEx entry(PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    EXPECT_EQ(ps.crossCoreConsumed_[entry][0], 1U);
    EXPECT_EQ(ps.ProjectCrossCoreConsumption(1)[0], 0U);
}

TEST_F(InsertSyncTest, A5CrossCoreConsumptionPropagatesThroughRealIntraCoreSync)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("A5CrossCoreIntraTransport");
    PipeSync ps;
    ps.InitCVEventIdQ();
    const PipeSync::PipeCore cube(PIPE_FIX, PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED);
    const PipeSync::PipeCore vec(PIPE_MTE2, PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCore alu(PIPE_V, PIPE_V, CoreType::AIV, AIVCore::AIV0);
    for (const auto& pipe : {cube, vec, alu, cube}) {
        const auto i = AppendLocalBarrierTestOp(ps, *leaf, pipe);
        if (i != 0) {
            ASSERT_EQ(ps.AddOpDep(ps.depOps_[i - 1], ps.depOps_[i]), SUCCESS);
        }
    }
    const PipeSync::PipeCoreRealEx entry(PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCoreRealEx source(PIPE_V, CoreType::AIV, AIVCore::AIV0);
    ps.freeEventId_[{entry, source}] = {1};
    std::vector<IndexOp> synced;
    ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, 0, synced), SUCCESS);
    ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, 1, synced), SUCCESS);
    ps.crossCoreFreeEventId_[source][0] = {0};
    ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, 2, synced), SUCCESS);
    EXPECT_EQ(ps.setWaitPairMap_.at({2, 3}), ps.setWaitPairMap_.at({0, 1}));
    EXPECT_EQ(ps.crossCoreGenerations_[0], 2U);
    EXPECT_TRUE(ps.forceSyncOps_.empty());
    ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, 3, synced), SUCCESS);
    EXPECT_TRUE(ps.crossCoreSnapshots_.empty());
}

TEST_F(InsertSyncTest, A5CrossCoreProofIsNotRecreatedAfterDependencyMerge)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("A5MergedCrossCoreProof");
    PipeSync ps;
    const PipeSync::PipeCore producer(PIPE_MTE2, PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCore consumer(PIPE_V, PIPE_V, CoreType::AIV, AIVCore::AIV0);
    for (const auto& pipe : {producer, producer, consumer, consumer}) {
        AppendLocalBarrierTestOp(ps, *leaf, pipe);
    }
    ASSERT_EQ(ps.AddOpDep(ps.depOps_[0], ps.depOps_[2]), SUCCESS);
    ASSERT_EQ(ps.AddOpDep(ps.depOps_[1], ps.depOps_[3]), SUCCESS);
    const PipeSync::PipeCoreRealEx source(PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCoreRealEx target(PIPE_V, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipePairEx pair{source, target};
    ps.freeEventId_[pair] = {1, 2};
    ps.crossCoreConsumed_[source][0] = 1;
    std::vector<IndexOp> synced;
    ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, 0, synced), SUCCESS);
    ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, 1, synced), SUCCESS);
    PipeSync::DataDepInfo dependencies;
    dependencies.opDepList = {{0, 2}, {1, 3}};
    dependencies.setOpEventIdList = {1, 2};
    for (size_t i = 0; i < synced.size(); ++i) {
        if (synced[i].second.get().GetOpcode() == Opcode::OP_SYNC_SRC) {
            dependencies.setOpIdList.push_back(i);
        }
    }
    ASSERT_EQ(dependencies.setOpIdList.size(), 2U);
    ASSERT_EQ(ps.SynDependency(0, dependencies, pair, synced), SUCCESS);
    EXPECT_TRUE(ps.crossCoreSnapshots_.empty());
    ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, 2, synced), SUCCESS);
    EXPECT_EQ(ps.crossCoreConsumed_[target][0], 0U);
}

TEST_F(InsertSyncTest, A5CrossCoreBarrierRechecksProofLostByCancelledInputWait)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("A5CrossCoreCancelledProjection");
    PipeSync ps;
    ps.InitCVEventIdQ();
    const PipeSync::PipeCore cube(PIPE_FIX, PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED);
    const PipeSync::PipeCore current(PIPE_M, PIPE_M, CoreType::AIC, AIVCore::UNSPECIFIED);
    const PipeSync::PipeCore vec0(PIPE_MTE2, PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCore vec1(PIPE_MTE2, PIPE_MTE2, CoreType::AIV, AIVCore::AIV1);
    for (const auto& pipe : {cube, vec1, cube, vec0}) {
        const auto i = AppendLocalBarrierTestOp(ps, *leaf, pipe);
        if (i != 0) {
            ASSERT_EQ(ps.AddOpDep(ps.depOps_[i - 1], ps.depOps_[i]), SUCCESS);
        }
    }
    for (size_t i = 0; i < 15; ++i) {
        AppendLocalBarrierTestOp(ps, *leaf, current);
    }
    constexpr size_t boundary = 18;
    ASSERT_EQ(ps.AddOpDep(ps.depOps_[3], ps.depOps_[boundary]), SUCCESS);
    for (size_t i = 4; i < boundary; ++i) {
        for (const auto& vector : {vec0, vec1}) {
            const auto consumer = AppendLocalBarrierTestOp(ps, *leaf, vector);
            ASSERT_EQ(ps.AddOpDep(ps.depOps_[i], ps.depOps_[consumer]), SUCCESS);
        }
    }
    // The input wait projects two reusable IDs per pool. Only pool 0 is short initially.
    const PipeSync::PipeCore vecAlu(PIPE_V, PIPE_V, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCore vecStore(PIPE_MTE3, PIPE_MTE3, CoreType::AIV, AIVCore::AIV0);
    for (const auto& vector : {vec0, vecAlu, vecStore, vec1}) {
        const auto consumer = AppendLocalBarrierTestOp(ps, *leaf, vector);
        ASSERT_EQ(ps.AddOpDep(ps.depOps_[boundary], ps.depOps_[consumer]), SUCCESS);
    }
    std::vector<IndexOp> synced;
    for (size_t i = 0; i < boundary; ++i) {
        ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, i, synced), SUCCESS);
    }
    EXPECT_TRUE(ps.forceSyncOps_.empty());
    const auto projected = ps.ProjectCrossCoreConsumption(boundary);
    EXPECT_EQ(projected[16], 1U);
    EXPECT_EQ(projected[17], 1U);
    ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, boundary, synced), SUCCESS);
    EXPECT_EQ(ps.forceSyncOps_.size(), 12U);
    EXPECT_TRUE(ps.depOps_[boundary].waitPipe.empty());
    EXPECT_EQ(ps.NoWaitCVPairs_.size(), 4U);
    for (auto* op : ps.forceSyncOps_) {
        EXPECT_NE(op->GetOpcode(), Opcode::OP_FFTS_CROSS_CORE_SYNC);
    }
}

TEST_F(InsertSyncTest, A5CrossCoreReuseSurvivesAtomicScopeAndVFTuning)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("A5CrossCoreFinalSyncOrder");
    PipeSync ps;
    ps.InitCVEventIdQ();
    const PipeSync::PipeCore cube(PIPE_MTE1, PIPE_MTE1, CoreType::AIC, AIVCore::UNSPECIFIED);
    const PipeSync::PipeCore vector(PIPE_V, PIPE_V, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCore load(PIPE_MTE2, PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCoreRealEx loadPipe(PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
    const PipeSync::PipeCoreRealEx vectorPipe(PIPE_V, CoreType::AIV, AIVCore::AIV0);
    ps.freeEventId_[{loadPipe, vectorPipe}] = {1, 2, 3, 4, 5, 6, 7};
    constexpr size_t groups = 20;
    for (size_t group = 0; group < groups; ++group) {
        const auto producer = AppendLocalBarrierTestOp(ps, *leaf, cube);
        ps.oriOpList_[producer]->SetOpCode(Opcode::OP_L1_TO_L0A);
        const auto transport = AppendLocalBarrierTestOp(ps, *leaf, load);
        ps.oriOpList_[transport]->SetOpCode(Opcode::OP_TRANSPOSE_MOVEIN);
        const auto consumer = AppendLocalBarrierTestOp(ps, *leaf, vector);
        const auto nextVector = AppendLocalBarrierTestOp(ps, *leaf, vector);
        ps.oriOpList_[consumer]->SetAtomicScopeId(VF_SCOPE_A + group);
        ps.oriOpList_[nextVector]->SetAtomicScopeId(VF_SCOPE_A + group);
        ASSERT_EQ(ps.AddOpDep(ps.depOps_[producer], ps.depOps_[transport]), SUCCESS);
        ASSERT_EQ(ps.AddOpDep(ps.depOps_[transport], ps.depOps_[consumer]), SUCCESS);
        if (group != 0) {
            ASSERT_EQ(ps.AddOpDep(ps.depOps_[consumer - 4], ps.depOps_[producer]), SUCCESS);
        }
    }
    std::vector<IndexOp> synced;
    for (size_t i = 0; i < ps.oriOpList_.size(); ++i) {
        ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, i, synced), SUCCESS);
    }
    EXPECT_TRUE(ps.forceSyncOps_.empty());
    EXPECT_GT(ps.crossCoreGenerations_[0], 1U);
    std::sort(synced.begin(), synced.end(), [](const IndexOp& a, const IndexOp& b) { return a.first < b.first; });
    std::vector<Operation*> finalOps;
    for (const auto& entry : synced) {
        finalOps.push_back(&entry.second.get());
    }
    InsertSync syncPass;
    ASSERT_EQ(syncPass.AdjustSyncByAtomicScope(finalOps, ps.forceSyncOps_), SUCCESS);
    leaf->setOpMap = ps.setOpMap;
    leaf->waitOpMap = ps.waitOpMap;
    leaf->oriOpList = ps.oriOpList_;
    TuneSyncForVF tune;
    tune.opList_ = finalOps;
    tune.GenPipeOpMap(leaf.get());
    ASSERT_EQ(tune.ChangeOpSeq(leaf.get(), false), SUCCESS);
    std::array<bool, 32> active{};
    size_t sets = 0;
    for (auto* op : tune.opList_) {
        if (op->GetOpcode() == Opcode::OP_CV_SYNC_SRC) {
            EXPECT_FALSE(active[op->syncQueue_.eventId_]);
            active[op->syncQueue_.eventId_] = true;
            ++sets;
        } else if (op->GetOpcode() == Opcode::OP_CV_SYNC_DST) {
            EXPECT_TRUE(active[op->syncQueue_.eventId_]);
            active[op->syncQueue_.eventId_] = false;
        }
    }
    EXPECT_EQ(sets, groups * 2 - 1);
    EXPECT_TRUE(std::none_of(active.begin(), active.end(), [](bool value) { return value; }));
    for (size_t group = 0; group < groups; ++group) {
        Operation* transportSet = nullptr;
        Operation* transportWait = nullptr;
        Operation* crossSet = nullptr;
        for (auto* op : tune.opList_) {
            if (op->GetOpcode() == Opcode::OP_SYNC_SRC && ps.setOpMap.at(op) == ps.oriOpList_[group * 4 + 2]) {
                transportSet = op;
            } else if (op->GetOpcode() == Opcode::OP_SYNC_DST && ps.waitOpMap.at(op) == ps.oriOpList_[group * 4 + 1]) {
                transportWait = op;
            } else if (group + 1 < groups && op->GetOpcode() == Opcode::OP_CV_SYNC_SRC &&
                       ps.setOpMap.at(op) == ps.oriOpList_[(group + 1) * 4]) {
                crossSet = op;
            }
        }
        ASSERT_NE(transportSet, nullptr);
        ASSERT_NE(transportWait, nullptr);
        const auto setPosition = std::find(tune.opList_.begin(), tune.opList_.end(), transportSet);
        const auto waitPosition = std::find(tune.opList_.begin(), tune.opList_.end(), transportWait);
        EXPECT_LT(setPosition, waitPosition);
        if (group + 1 < groups) {
            ASSERT_NE(crossSet, nullptr);
            EXPECT_LT(waitPosition, std::find(tune.opList_.begin(), tune.opList_.end(), crossSet));
        }
    }
}

TEST_F(InsertSyncTest, A5CrossCoreBarrierPreservesIntraCapacityAfterCancellingInput)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    for (bool globalFallback : {false, true}) {
        auto [root, leaf] = SetupDebugFunc("A5CrossCoreBarrierIntraCapacity");
        PipeSync ps;
        ps.InitCVEventIdQ();
        const PipeSync::PipeCore cube(PIPE_FIX, PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED);
        const PipeSync::PipeCore math(PIPE_M, PIPE_M, CoreType::AIC, AIVCore::UNSPECIFIED);
        const PipeSync::PipeCore vector(PIPE_MTE2, PIPE_MTE2, CoreType::AIV, AIVCore::AIV0);
        const PipeSync::PipePairEx intra{{PIPE_M, CoreType::AIC}, {PIPE_FIX, CoreType::AIC}};
        for (size_t i = 0; i < 7; ++i) {
            AddEventTestDependency(ps, *leaf, intra);
        }
        const auto transport = AppendLocalBarrierTestOp(ps, *leaf, cube);
        const auto returnSet = AppendLocalBarrierTestOp(ps, *leaf, vector);
        ASSERT_EQ(ps.AddOpDep(ps.depOps_[transport], ps.depOps_[returnSet]), SUCCESS);
        std::vector<size_t> pending;
        for (size_t i = 0; i < 14; ++i) {
            pending.push_back(AppendLocalBarrierTestOp(ps, *leaf, cube));
        }
        const auto boundary = AppendLocalBarrierTestOp(ps, *leaf, math);
        ASSERT_EQ(ps.AddOpDep(ps.depOps_[returnSet], ps.depOps_[boundary]), SUCCESS);
        for (auto producer : pending) {
            const auto consumer = AppendLocalBarrierTestOp(ps, *leaf, vector);
            ASSERT_EQ(ps.AddOpDep(ps.depOps_[producer], ps.depOps_[consumer]), SUCCESS);
        }
        for (const auto& target : {cube, vector, PipeSync::PipeCore(PIPE_V, PIPE_V, CoreType::AIV, AIVCore::AIV0),
                                   PipeSync::PipeCore(PIPE_MTE3, PIPE_MTE3, CoreType::AIV, AIVCore::AIV0)}) {
            const auto consumer = AppendLocalBarrierTestOp(ps, *leaf, target);
            ASSERT_EQ(ps.AddOpDep(ps.depOps_[boundary], ps.depOps_[consumer]), SUCCESS);
        }
        std::vector<IndexOp> synced;
        for (size_t i = 0; i < boundary; ++i) {
            ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, i, synced), SUCCESS);
        }
        EXPECT_FALSE(ps.AnalyzeIntraEventCapacity(ps.depOps_[boundary]).has_value());
        EXPECT_TRUE(ps.forceSyncOps_.empty());
        if (globalFallback) {
            // A handshake requires two fresh generations; force the legacy fallback.
            ps.crossCoreGenerations_.fill(std::numeric_limits<uint64_t>::max() - 1);
        }
        ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, boundary, synced), SUCCESS);
        EXPECT_TRUE(ps.depOps_[boundary].waitPipe.empty());
        EXPECT_EQ(ps.forceSyncOps_.size(), globalFallback ? 12U : 6U);
        EXPECT_EQ(ps.eventState_.Find(returnSet, boundary), nullptr);
        for (size_t i = boundary + 1; i < ps.oriOpList_.size(); ++i) {
            ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, i, synced), SUCCESS);
        }
        std::sort(synced.begin(), synced.end(), [](const IndexOp& a, const IndexOp& b) { return a.first < b.first; });
        std::vector<Operation*> ordered;
        for (const auto& entry : synced) {
            ordered.push_back(&entry.second.get());
        }
        EXPECT_EQ(ps.ValidateEventReuse(ordered), SUCCESS);
    }
}

TEST_F(InsertSyncTest, A5ReplayDropsCrossCoreProofsAndSlotOccupancy)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("A5CrossCoreReplayReset");
    PipeSync ps;
    ps.InitCVEventIdQ();
    AppendLocalBarrierTestOp(ps, *leaf, {PIPE_FIX, PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED});
    AppendLocalBarrierTestOp(ps, *leaf, {PIPE_MTE2, PIPE_MTE2, CoreType::AIV, AIVCore::AIV0});
    ASSERT_EQ(ps.AddOpDep(ps.depOps_[0], ps.depOps_[1]), SUCCESS);
    const auto plan = ps.depOps_;
    std::vector<IndexOp> synced;
    ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, 0, synced), SUCCESS);
    ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, 1, synced), SUCCESS);
    ASSERT_TRUE(ps.crossCoreEventSenders_[0].has_value());
    ASSERT_FALSE(ps.crossCoreConsumed_.empty());
    ps.ResetForReplay(plan, {}, {});
    EXPECT_TRUE(ps.crossCoreConsumed_.empty());
    EXPECT_TRUE(ps.crossCoreSnapshots_.empty());
    EXPECT_TRUE(std::all_of(ps.crossCoreGenerations_.begin(), ps.crossCoreGenerations_.end(),
                            [](uint64_t generation) { return generation == 0; }));
    EXPECT_TRUE(std::none_of(ps.crossCoreEventSenders_.begin(), ps.crossCoreEventSenders_.end(),
                             [](const auto& sender) { return sender.has_value(); }));
}

TEST_F(InsertSyncTest, A5CrossCoreDrainPreservesPendingIntraAndOtherCore)
{
    NpuArchScope archScope(NPUArch::DAV_3510);
    auto [root, leaf] = SetupDebugFunc("A5CrossCoreDrainScope");
    PipeSync ps;
    ps.InitCVEventIdQ();
    const auto pending = AppendLocalBarrierTestOp(ps, *leaf,
                                                  {PIPE_MTE1, PIPE_MTE1, CoreType::AIC, AIVCore::UNSPECIFIED});
    const auto other = AddEventTestDependency(
        ps, *leaf, {{PIPE_MTE2, CoreType::AIV, AIVCore::AIV1}, {PIPE_V, CoreType::AIV, AIVCore::AIV1}});
    std::vector<size_t> producers;
    for (size_t i = 0; i < 17; ++i) {
        producers.push_back(
            AppendLocalBarrierTestOp(ps, *leaf, {PIPE_FIX, PIPE_FIX, CoreType::AIC, AIVCore::UNSPECIFIED}));
    }
    const auto consumer = AppendLocalBarrierTestOp(ps, *leaf, {PIPE_M, PIPE_M, CoreType::AIC, AIVCore::UNSPECIFIED});
    ASSERT_EQ(ps.AddOpDep(ps.depOps_[pending], ps.depOps_[consumer]), SUCCESS);
    for (auto producer : producers) {
        const auto target = AppendLocalBarrierTestOp(ps, *leaf, {PIPE_MTE2, PIPE_MTE2, CoreType::AIV, AIVCore::AIV0});
        ASSERT_EQ(ps.AddOpDep(ps.depOps_[producer], ps.depOps_[target]), SUCCESS);
    }
    std::vector<IndexOp> synced;
    for (size_t i = 0; i < consumer; ++i) {
        ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, i, synced), SUCCESS);
    }
    EXPECT_EQ(ps.forceSyncOps_.size(), 6U);
    ASSERT_NE(ps.eventState_.Find(pending, consumer), nullptr);
    EXPECT_EQ(ps.eventState_.Find(pending, consumer)->state, PipeSync::InstanceState::Allocated);
    ASSERT_NE(ps.eventState_.Find(other.first, other.second), nullptr);
    EXPECT_EQ(ps.eventState_.Find(other.first, other.second)->state, PipeSync::InstanceState::WaitInserted);
    for (size_t i = consumer; i < ps.oriOpList_.size(); ++i) {
        ASSERT_EQ(ps.InjectSync(*leaf, ps.oriOpList_, i, synced), SUCCESS);
    }
    std::sort(synced.begin(), synced.end(), [](const IndexOp& a, const IndexOp& b) { return a.first < b.first; });
    std::vector<Operation*> ordered;
    for (const auto& entry : synced) {
        ordered.push_back(&entry.second.get());
    }
    EXPECT_EQ(ps.ValidateEventReuse(ordered), SUCCESS);
}

#undef private

} // namespace tile_fwk
} // namespace npu
