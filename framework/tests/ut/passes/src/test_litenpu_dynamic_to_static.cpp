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
 * \file test_litenpu_dynamic_to_static.cpp
 * \brief Unit tests for the LitenpuDynamicToStatic pass.
 */

#include <gtest/gtest.h>

#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/program/program.h"
#include "interface/tensor/irbuilder.h"
#include "interface/tensor/logical_tensor.h"
#include "ir/kind_traits.h"
#include "ir/stmt.h"
#include "ir/transforms/litenpu_dynamic_to_static_pass.h"
#include "tilefwk/platform.h"
#include "tilefwk/tilefwk.h"

namespace npu {
namespace tile_fwk {

class LitenpuDynamicToStaticTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        // The pass is gated on IsLiteNPU(arch); fix the arch to a Lite NPU SoC so the
        // converter actually runs (otherwise Apply returns the input unchanged).
        savedArch_ = Platform::Instance().GetSoc().GetNPUArch();
        Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3113);
    }

    void TearDown() override { Platform::Instance().GetSoc().SetNPUArch(savedArch_); }

private:
    NPUArch savedArch_{NPUArch::DAV_UNKNOWN};
};

namespace {

// Non-shared clone resets oriShape to shape, corrupting downstream valid-shape; assert it survives.
// The second statement keeps SeqStmts::Flatten from unwrapping the body into a bare stmt (null deref).
TEST_F(LitenpuDynamicToStaticTest, ClonePreservesOriShapeForNonSharedResult)
{
    IRBuilder builder;

    // Function parameter (loop-carried input).
    auto in = builder.CreateTensorVar(DT_FP32, std::vector<int64_t>{8}, TileOpFormat::TILEOP_ND, "in");

    // Two fillpad-style results: padded static shape 8, original valid shape 5.
    auto out0 = builder.CreateTensorVar(DT_FP32, std::vector<int64_t>{8}, TileOpFormat::TILEOP_ND, "out0");
    out0->oriShape = std::vector<int64_t>{5};
    auto out1 = builder.CreateTensorVar(DT_FP32, std::vector<int64_t>{8}, TileOpFormat::TILEOP_ND, "out1");
    out1->oriShape = std::vector<int64_t>{5};

    // Fillpad tensor-ops whose results materialize fresh storage (non-shared clone path).
    auto fillpad0 = builder.CreateTensorOpStmt({out0}, nullptr, "OP_FILLPAD", {in}, {}, {}, ir::Span::Unknown());
    auto fillpad1 = builder.CreateTensorOpStmt({out1}, nullptr, "OP_FILLPAD", {in}, {}, {}, ir::Span::Unknown());

    auto loopVar = builder.CreateScalarVar("idx").AsVar();
    auto start = SymbolicScalar(0).AsExpr();
    auto stop = SymbolicScalar(1).AsExpr();
    auto step = SymbolicScalar(1).AsExpr();
    auto body = builder.CreateSeqStmts({fillpad0, fillpad1}, ir::Span::Unknown());
    auto forStmt = builder.CreateForStmt(loopVar, start, stop, step, {}, body, {}, ir::Span::Unknown());
    auto funcBody = builder.CreateSeqStmts({forStmt}, ir::Span::Unknown());

    std::vector<ir::VarPtr> params{in};
    auto result = ir::RunLitenpuDynamicToStaticPass(funcBody, params);
    ASSERT_NE(result, nullptr);

    // A single-iteration loop with a two-statement body unrolls to a flat SeqStmts holding the
    // two cloned fillpad ops.
    ASSERT_EQ(result->stmts_.size(), 2U);
    auto clonedOp = std::dynamic_pointer_cast<const ir::TensorOpStmt>(result->stmts_[0]);
    ASSERT_NE(clonedOp, nullptr);
    ASSERT_EQ(clonedOp->result_.size(), 1U);

    auto clonedOut = std::dynamic_pointer_cast<const LogicalTensor>(clonedOp->result_[0]);
    ASSERT_NE(clonedOut, nullptr);

    // The core regression guard: oriShape must survive the non-shared clone.
    EXPECT_EQ(clonedOut->oriShape, std::vector<int64_t>({5}));
    // The static (padded) shape itself is untouched.
    EXPECT_EQ(clonedOut->shape, std::vector<int64_t>({8}));
    // A fresh-storage result must not alias the source buffer: the clone gets its own RawTensor
    // (different rawmagic). The aliasing branch is guarded by CloneSharesRawTensorForAliasingResult.
    EXPECT_NE(clonedOut->GetRawTensor().get(), out0->GetRawTensor().get());
    EXPECT_NE(clonedOut->GetRawMagic(), out0->GetRawMagic());
}

// Aliasing results (dassemble dst / inplace dst) write into pre-existing storage, so the unrolled
// clone must keep the destination's RawTensor (same rawmagic) or the write lands on a different
// buffer than the one later read. Guards the WritesExistingStorage predicate (assemble-like and
// inplace branches): ClonePreservesOriShapeForNonSharedResult above only exercises the
// fresh-storage branch, so a flipped predicate would silently swap both branches without failing.
TEST_F(LitenpuDynamicToStaticTest, CloneSharesRawTensorForAliasingResult)
{
    IRBuilder builder;

    // A real Function so AddRawOperation yields Operation nodes; the pass's WritesExistingStorage
    // only recognizes Operation (a bare TensorOpStmt dynamic-casts to null and falls back to the
    // non-shared branch).
    auto func = std::make_shared<Function>(Program::GetInstance(), "CloneSharesRawTensorForAliasingResult_magic",
                                           "CloneSharesRawTensorForAliasingResult", nullptr);
    func->SetFunctionType(FunctionType::DYNAMIC);
    func->SetGraphType(GraphType::TENSOR_GRAPH);

    auto src = builder.CreateTensorVar(*func, DT_FP32, std::vector<int64_t>{8}, TileOpFormat::TILEOP_ND, "src");
    auto dst0 = builder.CreateTensorVar(*func, DT_FP32, std::vector<int64_t>{8}, TileOpFormat::TILEOP_ND, "dst0");
    auto dst1 = builder.CreateTensorVar(*func, DT_FP32, std::vector<int64_t>{8}, TileOpFormat::TILEOP_ND, "dst1");

    // assemble-like: OP_ASSEMBLE carrying the dassemble attr writes into dst0's storage.
    auto& assemble = func->AddRawOperation(Opcode::OP_ASSEMBLE, {src}, {dst0}, ir::Span::Unknown());
    assemble.SetAttribute("dassemble", true);

    // inplace: result 0 aliases input 0 (inplaceIdx), writing into dst1's storage.
    auto& inplace = func->AddRawOperation(Opcode::OP_ADD, {src, src}, {dst1}, ir::Span::Unknown());
    inplace.SetAttribute(OpAttributeKey::inplaceIdx, 0);

    auto loopVar = builder.CreateScalarVar("idx").AsVar();
    auto start = SymbolicScalar(0).AsExpr();
    auto stop = SymbolicScalar(1).AsExpr();
    auto step = SymbolicScalar(1).AsExpr();
    // Two statements so SeqStmts::Flatten keeps the body as a SeqStmts (see non-shared test).
    auto body = builder.CreateSeqStmts({std::static_pointer_cast<const ir::Stmt>(assemble.shared_from_this()),
                                        std::static_pointer_cast<const ir::Stmt>(inplace.shared_from_this())},
                                       ir::Span::Unknown());
    auto forStmt = builder.CreateForStmt(loopVar, start, stop, step, {}, body, {}, ir::Span::Unknown());
    auto funcBody = builder.CreateSeqStmts({forStmt}, ir::Span::Unknown());

    std::vector<ir::VarPtr> params{};
    auto result = ir::RunLitenpuDynamicToStaticPass(funcBody, params);
    ASSERT_NE(result, nullptr);

    // A single-iteration loop unrolls the two aliasing statements to a flat SeqStmts.
    ASSERT_EQ(result->stmts_.size(), 2U);

    auto assertSharesRaw = [](const ir::StmtPtr& stmt, const LogicalTensorPtr& dst) {
        auto op = std::dynamic_pointer_cast<const ir::TensorOpStmt>(stmt);
        ASSERT_NE(op, nullptr);
        ASSERT_EQ(op->result_.size(), 1U);
        auto clonedOut = std::dynamic_pointer_cast<const LogicalTensor>(op->result_[0]);
        ASSERT_NE(clonedOut, nullptr);
        // The clone must reuse dst's RawTensor (same pointer, same rawmagic).
        EXPECT_EQ(clonedOut->GetRawTensor().get(), dst->GetRawTensor().get());
        EXPECT_EQ(clonedOut->GetRawMagic(), dst->GetRawMagic());
    };
    assertSharesRaw(result->stmts_[0], dst0);
    assertSharesRaw(result->stmts_[1], dst1);
}

} // namespace
} // namespace tile_fwk
} // namespace npu
