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
 * \file test_expand_function.cpp
 * \brief Unit test for ExpandFunction pass.
 */

#include <gtest/gtest.h>
#include "symbolic_scalar_test_utils.h"
#include <vector>
#include <string>
#include "tilefwk/tilefwk.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "passes/pass_mgr/pass_manager.h"

#include "interface/tensor/irbuilder.h"
#include "passes/pass_utils/pass_operation_utils.h"
#define private public
#include "passes/block_graph_pass/loopaxes_proc.h"

namespace npu {
namespace tile_fwk {
static const int kKeepOut = -1;
static const int kNum0 = 0;
static const int kNum1 = 1;
static const int kNum2 = 2;
static const int kNum3 = 3;
static const int kNum4 = 4;
static const int kNum16 = 2;
static const std::vector<int64_t> shape1 = {kNum2, kNum16};
static const std::vector<int64_t> shape2 = {kNum2, kNum2, kNum2, kNum4};
static const std::vector<int64_t> shape3 = {kNum4, kNum2, kNum4};
static const std::vector<int64_t> shape4 = {kNum3, kNum2, kNum2, kNum4};
static const std::vector<SymbolicScalar> symShape1 = {kNum2, kNum16};
static const std::vector<SymbolicScalar> symShape2 = {kNum2, kNum2, kNum2, kNum4};
static const std::vector<SymbolicScalar> symShape3 = {kNum4, kNum2, kNum4};
static const std::vector<SymbolicScalar> symShape4 = {kNum3, kNum2, kNum2, kNum4};
static const std::vector<SymbolicScalar> expectedLoopAxis1 = {kNum2, kNum2};
static const std::vector<SymbolicScalar> expectedLoopAxis2 = {kNum3, kNum2};

class TestLoopaxesProcPass : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPassGlobalConfig(KEY_VF_OPT_MARK_FOR, true);
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "ExpandFunctionTestStrategy");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {}
};

bool EqualSymShape(const std::vector<SymbolicScalar>& A, const std::vector<SymbolicScalar>& B)
{
    if (A.size() != B.size()) {
        return false;
    }
    for (size_t i = 0; i < A.size(); ++i) {
        if (A[i].Dump() != B[i].Dump()) {
            return false;
        }
    }
    return true;
}

static constexpr int kAtomicScopeId = 200000000;
static const std::vector<int64_t> shapeReduced = {kNum2, kNum1, kNum2, kNum4};
static const std::vector<SymbolicScalar> symShapeReduced = {kNum2, kNum1, kNum2, kNum4};

static LogicalTensorPtr MakeUbTensor(const std::vector<int64_t>& shape, const std::vector<SymbolicScalar>& dynShape)
{
    auto tensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, CreateTestConstIntVector(shape));
    tensor->UpdateDynValidShape(dynShape);
    return tensor;
}

static Operation& MakeVfOp(Function& func, Opcode opcode, const LogicalTensors& inputs, const LogicalTensors& outputs,
                           int atomicScopeId = -1)
{
    auto& op = PassOperationUtils::AddOperation(func, opcode, inputs, outputs);
    if (atomicScopeId > 0) {
        op.SetAtomicScopeId(atomicScopeId);
    }
    return op;
}

static void SetMemRange(const LogicalTensorPtr& tensor, size_t start, size_t end)
{
    tensor->memoryrange.start = start;
    tensor->memoryrange.end = end;
}

struct DynLoopFuncPair {
    std::shared_ptr<Function> root;
    std::shared_ptr<Function> leaf;
};

static DynLoopFuncPair MakeDynLoopFuncPair(const std::string& name)
{
    auto root = std::make_shared<Function>(Program::GetInstance(), name, name, nullptr);
    root->rootFunc_ = root.get();
    auto leaf = std::make_shared<Function>(Program::GetInstance(), name + "Leaf", name + "Leaf", root.get());
    Program::GetInstance().InsertFuncToFunctionMap(leaf->GetMagicName(), leaf);
    root->rootFunc_->programs_.emplace(leaf->GetFuncMagic(), leaf.get());
    root->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    root->SetGraphType(GraphType::EXECUTE_GRAPH);
    leaf->SetGraphType(GraphType::TILE_GRAPH);
    leaf->SetFunctionType(FunctionType::STATIC);
    root->SetUnderDynamicFunction(true);
    return {root, leaf};
}

static void MakeCallOp(Function& root, Function& leaf, size_t numInputs, size_t numOutputs)
{
    LogicalTensors rootInputs, rootOutputs;
    for (size_t i = 0; i < numInputs; i++) {
        rootInputs.push_back(MakeUbTensor(shape2, symShape2));
    }
    for (size_t i = 0; i < numOutputs; i++) {
        rootOutputs.push_back(MakeUbTensor(shape2, symShape2));
    }
    auto& callOp = IRBuilder().CreateTensorOpStmt(root, Opcode::OP_CALL, rootInputs, rootOutputs);
    std::vector<std::vector<SymbolicScalar>> argList;
    std::map<int, SymbolicScalar> outIndexToExpr;
    callOp.SetOpAttribute(leaf.CreateCallOpAttribute(argList, outIndexToExpr));
    for (size_t i = 0; i < numInputs; i++)
        callOp.SetIOpAtt(i, 0);
    for (size_t i = 0; i < numOutputs; i++)
        callOp.SetOOpAtt(i, 0);
}

static void VerifyDynLoopGroup(const Operation& op, int64_t expectedGroup)
{
    EXPECT_TRUE(op.HasAttr(OpAttributeKey::dynloopGroup));
    EXPECT_EQ(op.GetIntAttribute(OpAttributeKey::dynloopGroup), expectedGroup);
}

static void VerifyDynLoopGroupStart(const Operation& op)
{
    EXPECT_TRUE(op.HasAttr(OpAttributeKey::dynloopGroupStart));
    EXPECT_TRUE(op.GetBoolAttribute(OpAttributeKey::dynloopGroupStart));
}

static void VerifyDynLoopGroupEnd(const Operation& op)
{
    EXPECT_TRUE(op.HasAttr(OpAttributeKey::dynloopGroupEnd));
    EXPECT_TRUE(op.GetBoolAttribute(OpAttributeKey::dynloopGroupEnd));
}

TEST_F(TestLoopaxesProcPass, LoopaxesProcUTest1)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestLoopaxesProcPass",
                                                  "TestLoopaxesProcPass", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestLoopaxesProcPassLeaf",
                                                      "TestLoopaxesProcPassLeaf", rootFuncPtr.get());
    // Register the leaf function in Program's functionmap_ so GetFunctionByMagicName can find it
    Program::GetInstance().InsertFuncToFunctionMap(currFunctionPtr->GetMagicName(), currFunctionPtr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());
    rootFuncPtr->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    rootFuncPtr->SetGraphType(GraphType::EXECUTE_GRAPH);
    currFunctionPtr->SetGraphType(GraphType::TILE_GRAPH);
    currFunctionPtr->SetFunctionType(FunctionType::STATIC);
    rootFuncPtr->SetUnderDynamicFunction(true);

    auto inCast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    inCast1->UpdateDynValidShape(symShape1);
    auto inCast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    inCast2->UpdateDynValidShape(symShape2);
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    ubTensor1->UpdateDynValidShape(symShape1);
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    ubTensor2->UpdateDynValidShape(symShape2);
    auto ubTensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    ubTensor3->UpdateDynValidShape(symShape2);
    auto ubTensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    ubTensor4->UpdateDynValidShape(symShape2);
    auto ubTensor5 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    ubTensor5->UpdateDynValidShape(symShape2);
    auto ubTensor6 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape4, CreateTestConstIntVector(shape4));
    ubTensor6->UpdateDynValidShape(symShape4);
    auto ubTensor7 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape4, CreateTestConstIntVector(shape4));
    ubTensor7->UpdateDynValidShape(symShape4);
    auto ubTensor8 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape4, CreateTestConstIntVector(shape4));
    ubTensor8->UpdateDynValidShape(symShape4);
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape3, CreateTestConstIntVector(shape3));
    outCast->UpdateDynValidShape(symShape3);

    auto& expand = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXPAND, {inCast2}, {ubTensor2});
    expand.SetAttribute(OpAttributeKey::expandDims, std::vector<int>{kNum3});
    PassOperationUtils::AddOperation(*currFunctionPtr, npu::tile_fwk::Opcode::OP_BAR_ALL, {inCast1}, {ubTensor2});
    auto& add = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ADD, {ubTensor2, ubTensor3}, {ubTensor4});
    auto& mul = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_MUL, {ubTensor2, ubTensor4}, {ubTensor5});
    auto& sub = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_SUB, {ubTensor6, ubTensor7}, {ubTensor8});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {ubTensor5}, {outCast});
    currFunctionPtr->inCasts_.push_back(inCast1);
    currFunctionPtr->inCasts_.push_back(inCast2);
    currFunctionPtr->outCasts_.push_back(outCast);

    // Create a call operation in rootFunc to connect to currFunctionPtr
    auto rootInCast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, CreateTestConstIntVector(shape1));
    auto rootInCast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, CreateTestConstIntVector(shape2));
    auto rootOutCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape3, CreateTestConstIntVector(shape3));
    auto& callOp = IRBuilder().CreateTensorOpStmt(*rootFuncPtr, Opcode::OP_CALL, {rootInCast1, rootInCast2},
                                                  {rootOutCast});
    std::vector<std::vector<SymbolicScalar>> argList;
    std::map<int, SymbolicScalar> outIndexToExpr;
    callOp.SetOpAttribute(currFunctionPtr->CreateCallOpAttribute(argList, outIndexToExpr));
    callOp.SetIOpAtt(0, 0);
    callOp.SetIOpAtt(1, 0);
    callOp.SetOOpAtt(0, 0);

    LoopaxesProc loopaxesprocpass;
    EXPECT_EQ(loopaxesprocpass.RunOnFunction(*rootFuncPtr), SUCCESS);

    EXPECT_TRUE(expand.HasAttr(OpAttributeKey::dynloopGroup));
    EXPECT_EQ(expand.GetIntAttribute(OpAttributeKey::dynloopGroup), kNum0);
    EXPECT_TRUE(expand.HasAttr(OpAttributeKey::dynloopAxes));
    EXPECT_TRUE(EqualSymShape(expand.GetVectorSymbolicScalarAttribute(OpAttributeKey::dynloopAxes), expectedLoopAxis1));

    EXPECT_TRUE(add.HasAttr(OpAttributeKey::dynloopGroup));
    EXPECT_EQ(add.GetIntAttribute(OpAttributeKey::dynloopGroup), kNum1);
    EXPECT_TRUE(add.HasAttr(OpAttributeKey::dynloopAxes));
    EXPECT_TRUE(EqualSymShape(add.GetVectorSymbolicScalarAttribute(OpAttributeKey::dynloopAxes), expectedLoopAxis1));

    EXPECT_TRUE(mul.HasAttr(OpAttributeKey::dynloopGroup));
    EXPECT_EQ(mul.GetIntAttribute(OpAttributeKey::dynloopGroup), kNum1);
    EXPECT_TRUE(mul.HasAttr(OpAttributeKey::dynloopAxes));
    EXPECT_TRUE(EqualSymShape(mul.GetVectorSymbolicScalarAttribute(OpAttributeKey::dynloopAxes), expectedLoopAxis1));

    EXPECT_TRUE(sub.HasAttr(OpAttributeKey::dynloopGroup));
    EXPECT_EQ(sub.GetIntAttribute(OpAttributeKey::dynloopGroup), kNum2);
    EXPECT_TRUE(sub.HasAttr(OpAttributeKey::dynloopAxes));
    EXPECT_TRUE(EqualSymShape(sub.GetVectorSymbolicScalarAttribute(OpAttributeKey::dynloopAxes), expectedLoopAxis2));
}

TEST_F(TestLoopaxesProcPass, RejectsMismatchedOutputValidShapeRank)
{
    auto pair = MakeDynLoopFuncPair("bad_rank");
    auto input = MakeUbTensor(shape2, symShape2);
    auto output = MakeUbTensor(shape2, symShape2);
    auto& add = MakeVfOp(*pair.leaf, Opcode::OP_ADD, {input, input}, {output}, kAtomicScopeId);
    output->UpdateDynValidShape({2});
    LoopaxesProc pass;
    EXPECT_EQ(pass.UpdateOpLoopAxes(add, *pair.leaf), FAILED);
}

TEST_F(TestLoopaxesProcPass, LoopaxesProcSubProgramNullptr)
{
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "LoopaxesProcNullTest",
                                                  "LoopaxesProcNullTest", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    rootFuncPtr->SetFunctionType(FunctionType::DYNAMIC_LOOP_PATH);
    rootFuncPtr->SetGraphType(GraphType::EXECUTE_GRAPH);
    rootFuncPtr->programs_[0] = nullptr;
    rootFuncPtr->programs_[1] = rootFuncPtr.get();

    LoopaxesProc loopaxesprocpass;
    EXPECT_EQ(loopaxesprocpass.RunOnFunction(*rootFuncPtr), SUCCESS);
}

// T1: atomicScope 内 loopAxes 全相同 → 正常形成 group
TEST_F(TestLoopaxesProcPass, AtomicScopeSameLoopAxesFormsGroup)
{
    auto pair = MakeDynLoopFuncPair("T1");
    auto b = MakeUbTensor(shape2, symShape2);
    auto c = MakeUbTensor(shape2, symShape2);
    auto d = MakeUbTensor(shape2, symShape2);
    auto e = MakeUbTensor(shape2, symShape2);
    auto f = MakeUbTensor(shape2, symShape2);
    auto g = MakeUbTensor(shape2, symShape2);
    auto h = MakeUbTensor(shape2, symShape2);
    auto& add = MakeVfOp(*pair.leaf, Opcode::OP_ADD, {b, c}, {d}, kAtomicScopeId);
    auto& mul = MakeVfOp(*pair.leaf, Opcode::OP_MUL, {d, e}, {f}, kAtomicScopeId);
    auto& sub = MakeVfOp(*pair.leaf, Opcode::OP_SUB, {f, g}, {h}, kAtomicScopeId);
    pair.leaf->inCasts_.push_back(b);
    pair.leaf->outCasts_.push_back(h);
    MakeCallOp(*pair.root, *pair.leaf, 1, 1);
    LoopaxesProc pass;
    EXPECT_EQ(pass.RunOnFunction(*pair.root), SUCCESS);
    VerifyDynLoopGroup(add, 0);
    VerifyDynLoopGroupStart(add);
    VerifyDynLoopGroup(mul, 0);
    VerifyDynLoopGroup(sub, 0);
    VerifyDynLoopGroupEnd(sub);
}

// T6: 非 atomicScope 地址冲突 → 正常 cut
TEST_F(TestLoopaxesProcPass, NonAtomicScopeAddrConflictNormalCut)
{
    auto pair = MakeDynLoopFuncPair("T6");
    auto b = MakeUbTensor(shape2, symShape2);
    auto c = MakeUbTensor(shape2, symShape2);
    auto d = MakeUbTensor(shape2, symShape2);
    auto e = MakeUbTensor(shape2, symShape2);
    auto f = MakeUbTensor(shape2, symShape2);
    auto g = MakeUbTensor(shape2, symShape2);
    auto h = MakeUbTensor(shape2, symShape2);
    SetMemRange(b, 0, 128);
    SetMemRange(d, 256, 384);
    SetMemRange(f, 512, 640);
    SetMemRange(h, 768, 896);
    auto& add = MakeVfOp(*pair.leaf, Opcode::OP_ADD, {b, c}, {d});
    auto& mul = MakeVfOp(*pair.leaf, Opcode::OP_MUL, {d, e}, {f});
    MakeVfOp(*pair.leaf, Opcode::OP_SUB, {f, g}, {h});
    pair.leaf->inCasts_.push_back(b);
    pair.leaf->outCasts_.push_back(h);
    MakeCallOp(*pair.root, *pair.leaf, 1, 1);
    LoopaxesProc pass;
    EXPECT_EQ(pass.RunOnFunction(*pair.root), SUCCESS);
    VerifyDynLoopGroupEnd(add);
    VerifyDynLoopGroupStart(mul);
}

// T6b: 真实非重叠 memoryrange，仅存在自冲突噪声 → 不切分，返回 SUCCESS（与主干行为一致）
TEST_F(TestLoopaxesProcPass, RealAddrNoCrossConflictReturnsSuccess)
{
    auto pair = MakeDynLoopFuncPair("T6b");
    auto b = MakeUbTensor(shape2, symShape2);
    auto c = MakeUbTensor(shape2, symShape2);
    auto d = MakeUbTensor(shape2, symShape2);
    auto f = MakeUbTensor(shape2, symShape2);
    auto g = MakeUbTensor(shape2, symShape2);
    auto h = MakeUbTensor(shape2, symShape2);
    SetMemRange(b, 0, 128);
    SetMemRange(c, 1000, 1128);
    SetMemRange(d, 256, 384);
    SetMemRange(f, 5000, 5128);
    SetMemRange(g, 6000, 6128);
    SetMemRange(h, 7000, 7128);
    MakeVfOp(*pair.leaf, Opcode::OP_ADD, {b, c}, {d});
    MakeVfOp(*pair.leaf, Opcode::OP_MUL, {f, g}, {h});
    pair.leaf->inCasts_.push_back(b);
    pair.leaf->outCasts_.push_back(h);
    MakeCallOp(*pair.root, *pair.leaf, 1, 1);
    LoopaxesProc pass;
    EXPECT_EQ(pass.RunOnFunction(*pair.root), SUCCESS);
}

// T7: atomicScope 与非 atomicScope 混合，loopAxes 全相同 → 正常分组，校验通过
TEST_F(TestLoopaxesProcPass, AtomicScopeMixedWithNonAtomic)
{
    auto pair = MakeDynLoopFuncPair("T7");
    auto inCast = MakeUbTensor(shape2, symShape2);
    auto b = MakeUbTensor(shape2, symShape2);
    auto c = MakeUbTensor(shape2, symShape2);
    auto d = MakeUbTensor(shape2, symShape2);
    auto e = MakeUbTensor(shape2, symShape2);
    auto f = MakeUbTensor(shape2, symShape2);
    auto g = MakeUbTensor(shape2, symShape2);
    auto h = MakeUbTensor(shape2, symShape2);
    auto out = MakeUbTensor(shape2, symShape2);
    auto& expand = MakeVfOp(*pair.leaf, Opcode::OP_EXPAND, {inCast}, {b});
    expand.SetAttribute(OpAttributeKey::expandDims, std::vector<int>{kNum3});
    auto& add = MakeVfOp(*pair.leaf, Opcode::OP_ADD, {b, c}, {d}, kAtomicScopeId);
    auto& mul = MakeVfOp(*pair.leaf, Opcode::OP_MUL, {d, e}, {f}, kAtomicScopeId);
    auto& sub = MakeVfOp(*pair.leaf, Opcode::OP_SUB, {f, g}, {h}, kAtomicScopeId);
    auto& cast = MakeVfOp(*pair.leaf, Opcode::OP_CAST, {h}, {out});
    cast.SetAttribute(OP_ATTR_PREFIX + "mode", static_cast<int64_t>(0));
    cast.SetAttribute(OP_ATTR_PREFIX + "satmode", static_cast<int64_t>(0));
    pair.leaf->inCasts_.push_back(inCast);
    pair.leaf->outCasts_.push_back(out);
    MakeCallOp(*pair.root, *pair.leaf, 1, 1);
    LoopaxesProc pass;
    EXPECT_EQ(pass.RunOnFunction(*pair.root), SUCCESS);
    VerifyDynLoopGroup(expand, 0);
    VerifyDynLoopGroupStart(expand);
    VerifyDynLoopGroup(add, 0);
    VerifyDynLoopGroup(mul, 0);
    VerifyDynLoopGroup(sub, 0);
    VerifyDynLoopGroup(cast, 0);
    VerifyDynLoopGroupEnd(cast);
}

// T9: 动态 loopAxes 全相同 → 正常形成 group
TEST_F(TestLoopaxesProcPass, AtomicScopeDynLoopAxesConsistent)
{
    auto pair = MakeDynLoopFuncPair("T9");
    auto b = MakeUbTensor(shape2, symShape2);
    auto c = MakeUbTensor(shape2, symShape2);
    auto d = MakeUbTensor(shape2, symShape2);
    auto e = MakeUbTensor(shape2, symShape2);
    auto f = MakeUbTensor(shape2, symShape2);
    auto& add = MakeVfOp(*pair.leaf, Opcode::OP_ADD, {b, c}, {d}, kAtomicScopeId);
    auto& mul = MakeVfOp(*pair.leaf, Opcode::OP_MUL, {d, e}, {f}, kAtomicScopeId);
    pair.leaf->inCasts_.push_back(b);
    pair.leaf->outCasts_.push_back(f);
    MakeCallOp(*pair.root, *pair.leaf, 1, 1);
    LoopaxesProc pass;
    EXPECT_EQ(pass.RunOnFunction(*pair.root), SUCCESS);
    VerifyDynLoopGroup(add, 0);
    VerifyDynLoopGroupStart(add);
    VerifyDynLoopGroup(mul, 0);
    VerifyDynLoopGroupEnd(mul);
}

// 跨 leaf function 状态不泄漏——上一份 leaf 的末组 axes/地址记录不得并入下一份 leaf 的首个组
// （否则后续 leaf 首个 op 沿用 previousLoopAxes 得到 INVALID 组号，或旧 op 被重复切分）
TEST_F(TestLoopaxesProcPass, NoStateLeakAcrossLeafFunctions)
{
    // leaf 1：两个 op 真实地址冲突，触发切分
    auto pair1 = MakeDynLoopFuncPair("T14Leaf1");
    auto b1 = MakeUbTensor(shape2, symShape2);
    auto c1 = MakeUbTensor(shape2, symShape2);
    auto d1 = MakeUbTensor(shape2, symShape2);
    auto b2 = MakeUbTensor(shape2, symShape2);
    auto c2 = MakeUbTensor(shape2, symShape2);
    auto d2 = MakeUbTensor(shape2, symShape2);
    SetMemRange(b1, 0, 128);
    SetMemRange(c1, 1000, 1128);
    SetMemRange(d1, 2000, 2128);
    SetMemRange(b2, 64, 192);
    SetMemRange(c2, 1000, 1128);
    SetMemRange(d2, 2000, 2128);
    auto& opA = MakeVfOp(*pair1.leaf, Opcode::OP_ADD, {b1, c1}, {d1});
    auto& opB = MakeVfOp(*pair1.leaf, Opcode::OP_MUL, {b2, c2}, {d2});
    pair1.leaf->inCasts_.push_back(b1);
    pair1.leaf->outCasts_.push_back(d2);
    MakeCallOp(*pair1.root, *pair1.leaf, 1, 1);

    // leaf 2：与 leaf 1 相同 loopAxes，无地址冲突
    auto pair2 = MakeDynLoopFuncPair("T14Leaf2");
    auto e1 = MakeUbTensor(shape2, symShape2);
    auto f1 = MakeUbTensor(shape2, symShape2);
    auto g1 = MakeUbTensor(shape2, symShape2);
    auto e2 = MakeUbTensor(shape2, symShape2);
    auto f2 = MakeUbTensor(shape2, symShape2);
    auto g2 = MakeUbTensor(shape2, symShape2);
    SetMemRange(e1, 300000, 300128);
    SetMemRange(f1, 301000, 301128);
    SetMemRange(g1, 302000, 302128);
    SetMemRange(e2, 303000, 303128);
    SetMemRange(f2, 304000, 304128);
    SetMemRange(g2, 305000, 305128);
    auto& opC = MakeVfOp(*pair2.leaf, Opcode::OP_ADD, {e1, f1}, {g1});
    auto& opD = MakeVfOp(*pair2.leaf, Opcode::OP_MUL, {e2, f2}, {g2});
    pair2.leaf->inCasts_.push_back(e1);
    pair2.leaf->outCasts_.push_back(g2);
    MakeCallOp(*pair2.root, *pair2.leaf, 1, 1);

    // 同一 pass 实例先后处理两份 leaf（模拟 UpdateFuncLoopAxes 的 leaf 轮次间 ResetGroupState）
    LoopaxesProc pass;
    EXPECT_EQ(pass.RunOnFunction(*pair1.root), SUCCESS);
    EXPECT_EQ(pass.RunOnFunction(*pair2.root), SUCCESS);
    // leaf 2 首个 op 必须开启新组并拿到合法组号（0），而不是沿用泄漏的 previousLoopAxes 得到 INVALID
    VerifyDynLoopGroupStart(opC);
    VerifyDynLoopGroup(opC, kNum0);
    VerifyDynLoopGroup(opD, kNum0);
    VerifyDynLoopGroupEnd(opD);
    // leaf 1 的切分结果不被 leaf 2 改写
    VerifyDynLoopGroupEnd(opA);
    VerifyDynLoopGroupStart(opB);
    VerifyDynLoopGroup(opB, kNum1);
}
} // namespace tile_fwk
} // namespace npu
