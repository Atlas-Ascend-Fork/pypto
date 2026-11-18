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
 * \file test_infer_shape.cpp
 * \brief
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "interface/program/program.h"
#include "tilefwk/tilefwk.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_utils/pass_operation_utils.h"
#include "symbolic_scalar_test_utils.h"
#include "interface/inner/tilefwk.h"
#include "interface/operation/op_infer_shape_impl.h"
#include "interface/operation/operation_impl.h"
#include "passes/tile_graph_pass/graph_constraint/infer_dyn_shape.h"
#include "interface/operation/attribute.h"
#include "interface/tensor/irbuilder.h"

namespace npu {
namespace tile_fwk {
class InferShapeTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
    }

    void TearDown() override {}
};

namespace {
std::shared_ptr<CopyOpAttribute> CreateCopyInAttribute(const std::vector<OpImmediate>& offset, MemoryType memoryType,
                                                       const std::vector<OpImmediate>& fromShape,
                                                       const std::vector<OpImmediate>& toShape,
                                                       std::initializer_list<const char*> dimNames = {})
{
    auto attr = std::make_shared<CopyOpAttribute>(offset, memoryType, fromShape, toShape, std::vector<OpImmediate>());
    if (dimNames.size() != 0) {
        std::vector<OpImmediate> dynValidShape;
        dynValidShape.reserve(dimNames.size());
        for (const char* dimName : dimNames) {
            dynValidShape.emplace_back(CreateTestScalarVar(dimName));
        }
        attr->SetToDynValidShape(dynValidShape);
    }
    return attr;
}

std::shared_ptr<CopyOpAttribute> CreateCopyOutAttribute(MemoryType memoryType, const std::vector<OpImmediate>& offset,
                                                        const std::vector<OpImmediate>& fromShape,
                                                        const std::vector<OpImmediate>& toShape)
{
    return std::make_shared<CopyOpAttribute>(memoryType, offset, fromShape, toShape, std::vector<OpImmediate>());
}

void AddCopyOp(const std::shared_ptr<Function>& currFunctionPtr, Opcode opcode,
               const std::shared_ptr<LogicalTensor>& input, const std::shared_ptr<LogicalTensor>& output,
               const std::shared_ptr<CopyOpAttribute>& attr)
{
    PassOperationUtils::AddOperation(*currFunctionPtr, opcode, {input}, {output},
                                     [&attr](Operation& op) { op.SetOpAttribute(attr); });
}

void AppendFunctionIO(const std::shared_ptr<Function>& currFunctionPtr,
                      const std::vector<std::shared_ptr<LogicalTensor>>& inputs,
                      const std::vector<std::shared_ptr<LogicalTensor>>& outputs)
{
    currFunctionPtr->inCasts_.insert(currFunctionPtr->inCasts_.end(), inputs.begin(), inputs.end());
    currFunctionPtr->outCasts_.insert(currFunctionPtr->outCasts_.end(), outputs.begin(), outputs.end());
}

void RunInferShapeAndExpect(const std::shared_ptr<Function>& currFunctionPtr, Status expected)
{
    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), expected);
}

} // namespace

TEST_F(InferShapeTest, AssumeDivisible_SimplifiesNestedViewTileInChildFunction)
{
    auto root = std::make_shared<Function>(Program::GetInstance(), "assume_root", "assume_root", nullptr);
    auto child = std::make_shared<Function>(Program::GetInstance(), "assume_child", "assume_child", root.get());
    SymbolicScalar q("q");
    SymbolicScalar tileIndex("tile_index");

    // These are the two 64-wide child views of a 128-wide parent tile.
    const auto firstValid = std::min(std::max(std::min(q - tileIndex * 128, 128), 0), 64);
    const auto secondValid = std::min(std::max(std::min(q - tileIndex * 128, 128) - 64, 0), 64);
    const auto validWithoutMax = std::min(q - tileIndex * 128, 64);
    Program::GetInstance().RegisterDivisibleAssumption(q, 128);

    InferDynShape pass;
    const auto firstResult = pass.SimplifyValidShapeWithAssumptions(*child, firstValid);
    const auto secondResult = pass.SimplifyValidShapeWithAssumptions(*child, secondValid);
    const auto noMaxResult = pass.SimplifyValidShapeWithAssumptions(*child, validWithoutMax);
    EXPECT_TRUE(firstResult.ConcreteValid());
    EXPECT_TRUE(secondResult.ConcreteValid());
    EXPECT_TRUE(noMaxResult.ConcreteValid());
    EXPECT_EQ(firstResult.Concrete(), 64);
    EXPECT_EQ(secondResult.Concrete(), 64);
    EXPECT_EQ(noMaxResult.Concrete(), 64);
    EXPECT_TRUE(Program::GetInstance().IsKnownDivisible(q, 64));
    EXPECT_TRUE(Program::GetInstance().IsKnownDivisible(q, 64));
}

TEST_F(InferShapeTest, AssumeDivisible_SimplifiesObservedFlashAttentionValidShapes)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "assume_fa_dump", "assume_fa_dump", nullptr);
    SymbolicScalar qEnd("q_end");
    SymbolicScalar qStart("q_start");
    SymbolicScalar kEnd("k_end");
    SymbolicScalar kStart("k_start");
    SymbolicScalar loopIdx3("loop_idx_3");
    SymbolicScalar loopIdx4("loop_idx_4");

    // Q/K mirror cu_seqlens_[loop_idx_1 + 1] - cu_seqlens_[loop_idx_1].
    const SymbolicScalar q = qEnd - qStart;
    const SymbolicScalar k = kEnd - kStart;
    const SymbolicScalar yq = q - loopIdx3 * 128;
    const SymbolicScalar yk = k - loopIdx4 * 128;
    Program::GetInstance().RegisterDivisibleAssumption(q, 128);
    Program::GetInstance().RegisterDivisibleAssumption(k, 128);

    // 5.1 - 5.4: observed K-side 128 tile valid shapes.
    const auto kTile0 = std::min(std::max(std::min(yk, 128), 0), 128);
    const auto kTile1 = std::min(std::max(std::min(yk, 256) + -128, 0), 128);
    const auto kTile2 = std::min(std::max(std::min(yk, 384) + -256, 0), 128);
    const auto kTile3 = std::min(std::max(std::min(yk, 512) + -384, 0), 128);
    const auto kTile4 = std::min(std::max(std::min(yk, 640) + -512, 0), 128);

    // 5.5: observed 128 -> 64 Q-side child Views.
    const auto qChild1 = std::min(std::max(std::min(yq, 128) + -64, 0), 64);
    const auto qChild0 = std::min(std::max(std::min(yq, 128), 0), 64);

    InferDynShape pass;
    const std::vector<SymbolicScalar> expected128 = {kTile0, kTile1, kTile2, kTile3, kTile4};
    for (const auto& validShape : expected128) {
        const auto result = pass.SimplifyValidShapeWithAssumptions(*function, validShape);
        ASSERT_TRUE(result.ConcreteValid());
        EXPECT_EQ(result.Concrete(), 128);
    }
    for (const auto& validShape : {qChild0, qChild1}) {
        const auto result = pass.SimplifyValidShapeWithAssumptions(*function, validShape);
        ASSERT_TRUE(result.ConcreteValid());
        EXPECT_EQ(result.Concrete(), 64);
    }
}

TEST_F(InferShapeTest, AssumeDivisible_SimplifiesMultiLevelViewTile)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "assume_multi_level", "assume_multi_level",
                                               nullptr);
    SymbolicScalar q("q");
    SymbolicScalar tileIndex("tile_index");
    Program::GetInstance().RegisterDivisibleAssumption(q, 128);

    // A 128 -> 64 -> 32 View chain. The expression intentionally retains
    // nested min caps and offsets instead of relying on prior algebraic
    // normalization.
    const auto valid = std::min(std::max(std::min(std::min(q - tileIndex * 128, 256) - 64, 128) - 32, 0), 32);

    InferDynShape pass;
    const auto result = pass.SimplifyValidShapeWithAssumptions(*function, valid);
    ASSERT_TRUE(result.ConcreteValid());
    EXPECT_EQ(result.Concrete(), 32);
}

TEST_F(InferShapeTest, AssumeDivisible_DoesNotSimplifyWithoutAssumption)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "assume_none", "assume_none", nullptr);
    SymbolicScalar q("q");
    SymbolicScalar tileIndex("tile_index");
    const auto valid = std::min(std::max(std::min(q - tileIndex * 128, 128) - 64, 0), 64);

    InferDynShape pass;
    const auto result = pass.SimplifyValidShapeWithAssumptions(*function, valid);
    EXPECT_FALSE(result.ConcreteValid());
}

TEST_F(InferShapeTest, AssumeDivisible_DoesNotSimplifyNonAlignedRemainingCap)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "assume_non_aligned", "assume_non_aligned",
                                               nullptr);
    SymbolicScalar q("q");
    Program::GetInstance().RegisterDivisibleAssumption(q, 64);

    // P - D is 66, which cannot represent a sequence of complete 64 tiles.
    const auto valid = std::min(std::max(std::min(q, 130) - 64, 0), 64);

    InferDynShape pass;
    const auto result = pass.SimplifyValidShapeWithAssumptions(*function, valid);
    EXPECT_FALSE(result.ConcreteValid());
}

TEST_F(InferShapeTest, AssumeDivisible_SimplifiesExplicitViewValidShapeAttribute)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "assume_explicit_view", "assume_explicit_view",
                                               nullptr);
    const std::vector<int64_t> shape = {64, 64};
    auto input = IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto output = IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    SymbolicScalar q("q");
    SymbolicScalar n("n");
    SymbolicScalar tileIndex("tile_index");
    Program::GetInstance().RegisterDivisibleAssumption(q, 128);

    const auto validM = std::min(std::max(std::min(q - tileIndex * 128, 128) - 64, 0), 64);
    auto viewAttr = std::make_shared<ViewOpAttribute>(
        std::vector<int64_t>{0, 0}, MEM_UNKNOWN, std::vector<SymbolicScalar>{}, std::vector<SymbolicScalar>{validM, n});
    PassOperationUtils::AddOperation(*function, Opcode::OP_VIEW, {input}, {output},
                                     [&viewAttr](Operation& op) { op.SetOpAttribute(viewAttr); });
    function->inCasts_.push_back(input);
    function->outCasts_.push_back(output);

    InferDynShape pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);
    const auto& attrValidShape = viewAttr->GetToDynValidShape();
    ASSERT_EQ(attrValidShape.size(), 2);
    EXPECT_TRUE(attrValidShape[0].ConcreteValid());
    EXPECT_EQ(attrValidShape[0].Concrete(), 64);
    EXPECT_EQ(attrValidShape[1].Dump(), n.Dump());
    EXPECT_TRUE(output->GetDynValidShape()[0].ConcreteValid());
    EXPECT_EQ(output->GetDynValidShape()[0].Concrete(), 64);
}

TEST_F(InferShapeTest, AssumeDivisible_SimplifiesCopyInValidShapeAttribute)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "assume_copyin", "assume_copyin", nullptr);
    const std::vector<int64_t> shape = {64, 64};
    const auto shapeImmediate = OpImmediate::Specified(shape);
    SymbolicScalar vm("vm");
    SymbolicScalar n("n");
    SymbolicScalar tileIndex("tile_index");
    Program::GetInstance().RegisterDivisibleAssumption(vm, 128);

    const auto validM = std::min(std::max(std::min(vm - tileIndex * 128, 128) + -64, 0), 64);
    auto input = IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto output = IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{validM, n});
    auto copyInAttr = CreateCopyInAttribute(OpImmediate::Specified({0, 0}), MEM_UB, shapeImmediate, shapeImmediate);
    copyInAttr->SetToDynValidShape(OpImmediate::Specified(std::vector<SymbolicScalar>{validM, n}));
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_COPY_IN, {input}, {output},
        [&copyInAttr](Operation& op) { op.SetOpAttribute(copyInAttr); }, ir::Span::Unknown(), false);
    function->inCasts_.push_back(input);
    function->outCasts_.push_back(output);

    InferDynShape pass;
    pass.SimplifyAllValidShapes(*function);

    const auto attrValidShape = OpImmediate::ToSpecified(copyInAttr->GetToDynValidShape());
    ASSERT_EQ(attrValidShape.size(), 2);
    EXPECT_TRUE(attrValidShape[0].ConcreteValid());
    EXPECT_EQ(attrValidShape[0].Concrete(), 64);
    EXPECT_EQ(attrValidShape[1].Dump(), n.Dump());
    EXPECT_TRUE(output->GetDynValidShape()[0].ConcreteValid());
    EXPECT_EQ(output->GetDynValidShape()[0].Concrete(), 64);
}

TEST_F(InferShapeTest, AssumeDivisible_ViewCopyInL0CCopyUB_CoexistSimplifyAndInferShape)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "assume_coexist", "assume_coexist", nullptr);
    const std::vector<int64_t> shape = {64};
    const auto shapeImmediate = OpImmediate::Specified(shape);
    SymbolicScalar vm("vm");
    SymbolicScalar tileIndex("tile_index");
    Program::GetInstance().RegisterDivisibleAssumption(vm, 128);

    const auto dynamicValid = std::min(std::max(std::min(vm - tileIndex * 128, 128) + -64, 0), 64);

    auto copyInInput = IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto copyInOutput = IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{dynamicValid});
    auto copyInAttr = CreateCopyInAttribute(OpImmediate::Specified({0}), MEM_UB, shapeImmediate, shapeImmediate);
    copyInAttr->SetToDynValidShape(OpImmediate::Specified(std::vector<SymbolicScalar>{dynamicValid}));
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_COPY_IN, {copyInInput}, {copyInOutput},
        [&copyInAttr](Operation& op) { op.SetOpAttribute(copyInAttr); }, ir::Span::Unknown(), false);

    auto viewInput = IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{dynamicValid});
    auto viewOutput = IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{dynamicValid});
    auto viewAttr = std::make_shared<ViewOpAttribute>(
        std::vector<int64_t>{0}, MEM_UNKNOWN, std::vector<SymbolicScalar>{}, std::vector<SymbolicScalar>{dynamicValid});
    PassOperationUtils::AddOperation(*function, Opcode::OP_VIEW, {viewInput}, {viewOutput},
                                     [&viewAttr](Operation& op) { op.SetOpAttribute(viewAttr); });

    auto l0cInput = IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{dynamicValid});
    auto l0cOutput = IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{dynamicValid});
    auto l0cAttr = CreateCopyOutAttribute(MEM_UB, OpImmediate::Specified({0}), shapeImmediate, shapeImmediate);
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_L0C_COPY_UB, {l0cInput}, {l0cOutput},
        [&l0cAttr](Operation& op) { op.SetOpAttribute(l0cAttr); }, ir::Span::Unknown(), false);

    function->inCasts_.push_back(copyInInput);
    function->inCasts_.push_back(viewInput);
    function->inCasts_.push_back(l0cInput);
    function->outCasts_.push_back(copyInOutput);
    function->outCasts_.push_back(viewOutput);
    function->outCasts_.push_back(l0cOutput);

    InferDynShape pass;
    pass.SimplifyAllValidShapes(*function);

    const auto copyInAttrValid = OpImmediate::ToSpecified(copyInAttr->GetToDynValidShape());
    ASSERT_EQ(copyInAttrValid.size(), 1);
    EXPECT_TRUE(copyInAttrValid[0].ConcreteValid());
    EXPECT_EQ(copyInAttrValid[0].Concrete(), 64);

    const auto& viewAttrValid = viewAttr->GetToDynValidShape();
    ASSERT_EQ(viewAttrValid.size(), 1);
    EXPECT_TRUE(viewAttrValid[0].ConcreteValid());
    EXPECT_EQ(viewAttrValid[0].Concrete(), 64);

    EXPECT_TRUE(copyInOutput->GetDynValidShape()[0].ConcreteValid());
    EXPECT_EQ(copyInOutput->GetDynValidShape()[0].Concrete(), 64);
    EXPECT_TRUE(viewOutput->GetDynValidShape()[0].ConcreteValid());
    EXPECT_EQ(viewOutput->GetDynValidShape()[0].Concrete(), 64);
}

TEST_F(InferShapeTest, AssumeDivisible_RegistersOnRootFunctionAndStoresOwnAssumption)
{
    auto root = std::make_shared<Function>(Program::GetInstance(), "assume_owner_root", "assume_owner_root", nullptr);
    root->SetFunctionType(FunctionType::DYNAMIC);
    auto loop = std::make_shared<Function>(Program::GetInstance(), "assume_owner_loop", "assume_owner_loop",
                                           root.get());
    loop->SetFunctionType(FunctionType::DYNAMIC_LOOP);
    auto child = std::make_shared<Function>(Program::GetInstance(), "assume_owner_child", "assume_owner_child",
                                            loop.get());
    child->SetFunctionType(FunctionType::STATIC);
    child->SetGraphType(GraphType::BLOCK_GRAPH);
    SymbolicScalar vm("vm");

    Program::GetInstance().RegisterDivisibleAssumption(vm, 128);

    const auto& rootAssumptions = Program::GetInstance().GetDivisibleAssumptions();
    const auto it = rootAssumptions.find(vm.Simplify().Dump());
    ASSERT_NE(it, rootAssumptions.end());
    EXPECT_EQ(it->second.expression.Dump(), vm.Dump());
    EXPECT_EQ(it->second.divisors, std::set<int64_t>({128}));
    EXPECT_TRUE(Program::GetInstance().IsKnownDivisible(vm, 64));
}

TEST_F(InferShapeTest, AssumeDivisible_StoresNormalizedExpressionIndependentOfRegistrationOrder)
{
    auto derivedFirst = std::make_shared<Function>(Program::GetInstance(), "assume_derived_first",
                                                   "assume_derived_first", nullptr);
    auto symbolFirst = std::make_shared<Function>(Program::GetInstance(), "assume_symbol_first", "assume_symbol_first",
                                                  nullptr);
    SymbolicScalar vm("vm");
    SymbolicScalar offset("offset");
    const auto equivalentExpr = (vm - offset) + offset;
    const auto key = vm.Simplify().Dump();

    Program::GetInstance().RegisterDivisibleAssumption(equivalentExpr, 64);
    Program::GetInstance().RegisterDivisibleAssumption(vm, 128);
    Program::GetInstance().RegisterDivisibleAssumption(vm, 128);
    Program::GetInstance().RegisterDivisibleAssumption(equivalentExpr, 64);

    for (size_t idx = 0; idx < 2U; idx++) {
        const auto& assumptions = Program::GetInstance().GetDivisibleAssumptions();
        const auto it = assumptions.find(key);
        ASSERT_NE(it, assumptions.end());
        EXPECT_TRUE(it->second.expression.IsSymbol());
        EXPECT_EQ(it->second.expression.Dump(), vm.Dump());
        EXPECT_EQ(it->second.divisors, std::set<int64_t>({64, 128}));
    }
}

TEST_F(InferShapeTest, AssumeDivisible_RunOnFunctionSavesSimplifiedValidShapeInStaticSnapshot)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "assume_snapshot", "assume_snapshot", nullptr);
    const std::vector<int64_t> shape = {64};
    const auto shapeImmediate = OpImmediate::Specified(shape);
    SymbolicScalar vm("vm");
    SymbolicScalar tileIndex("tile_index");
    const auto dynamicValid = std::min(std::max(vm - tileIndex * 128, 0), 64);
    auto input = IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{dynamicValid});
    auto output = IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{dynamicValid});
    auto copyAttr = CreateCopyOutAttribute(MEM_UB, OpImmediate::Specified({0}), shapeImmediate, shapeImmediate);
    auto& copyOp = PassOperationUtils::AddOperation(
        *function, Opcode::OP_L0C_COPY_UB, {input}, {output},
        [&copyAttr](Operation& op) { op.SetOpAttribute(copyAttr); }, ir::Span::Unknown(), false);
    function->inCasts_.push_back(input);
    function->outCasts_.push_back(output);
    Program::GetInstance().RegisterDivisibleAssumption(vm, 128);

    InferDynShape pass;
    ASSERT_EQ(pass.RunOnFunction(*function), SUCCESS);

    ASSERT_TRUE(copyOp.HasAttribute(OpAttributeKey::staticValidShape));
    const auto staticValidShape = copyOp.GetVectorIntAttribute<int64_t>(OpAttributeKey::staticValidShape);
    ASSERT_EQ(staticValidShape.size(), 1U);
    EXPECT_EQ(staticValidShape[0], 64);
}

TEST_F(InferShapeTest, AssumeDivisible_ChildQueriesRootAssumption)
{
    auto root = std::make_shared<Function>(Program::GetInstance(), "assume_root", "assume_root", nullptr);
    SymbolicScalar q("q");

    Program::GetInstance().RegisterDivisibleAssumption(q, 128);
    auto child = std::make_shared<Function>(Program::GetInstance(), "assume_child", "assume_child", root.get());

    EXPECT_TRUE(Program::GetInstance().IsKnownDivisible(q, 128));
    EXPECT_TRUE(Program::GetInstance().IsKnownDivisible(q, 64));
}

TEST_F(InferShapeTest, AssumeDivisible_RegistrationIsSharedThroughRootFunction)
{
    auto root = std::make_shared<Function>(Program::GetInstance(), "assume_root_sibling", "assume_root_sibling",
                                           nullptr);
    SymbolicScalar q("q_sibling");
    auto child = std::make_shared<Function>(Program::GetInstance(), "assume_child_sibling", "assume_child_sibling",
                                            root.get());
    auto sibling = std::make_shared<Function>(Program::GetInstance(), "assume_sibling", "assume_sibling", root.get());

    Program::GetInstance().RegisterDivisibleAssumption(q, 128);

    EXPECT_TRUE(Program::GetInstance().IsKnownDivisible(q, 128));
    EXPECT_FALSE(Program::GetInstance().GetDivisibleAssumptions().empty());
}

TEST_F(InferShapeTest, TestAdd)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestAddInferShape", "TestAddInferShape",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto incast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto ubTensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    outCast->UpdateDynValidShape({CreateTestScalarVar("output_0_Dim_0"), CreateTestScalarVar("output_0_Dim_1")});

    auto copyin1Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme,
                                                         std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {OpImmediate(CreateTestScalarVar("Input_0_Dim_0")),
                                                            OpImmediate(CreateTestScalarVar("Input_0_Dim_1"))};
    copyin1Attr->SetToDynValidShape(toValidShape);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast1}, {ubTensor1},
                                     [&copyin1Attr](Operation& op) { op.SetOpAttribute(copyin1Attr); });

    auto copyin2Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme,
                                                         std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape1 = {OpImmediate(CreateTestScalarVar("Input_1_Dim_0")),
                                                             OpImmediate(CreateTestScalarVar("Input_1_Dim_1"))};
    copyin2Attr->SetToDynValidShape(toValidShape1);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast2}, {ubTensor2},
                                     [&copyin2Attr](Operation& op) { op.SetOpAttribute(copyin2Attr); });

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ADD, {ubTensor1, ubTensor2}, {ubTensor3});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {ubTensor3}, {outCast});

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outCast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestAddAlignCase)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestAddInferShape", "TestAddInferShape",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto incast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto ubTensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});

    auto copyin1Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme,
                                                         std::vector<npu::tile_fwk::OpImmediate>());
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast1}, {ubTensor1},
                                     [&copyin1Attr](Operation& op) { op.SetOpAttribute(copyin1Attr); });

    auto copyin2Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme,
                                                         std::vector<npu::tile_fwk::OpImmediate>());
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast2}, {ubTensor2},
                                     [&copyin2Attr](Operation& op) { op.SetOpAttribute(copyin2Attr); });

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ADD, {ubTensor1, ubTensor2}, {ubTensor3});
    auto copyoutAttr = std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified({0, 0}), shapeImme, shapeImme,
                                                         std::vector<npu::tile_fwk::OpImmediate>());
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {ubTensor3}, {outCast},
                                     [&copyoutAttr](Operation& op) { op.SetOpAttribute(copyoutAttr); });

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outCast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestAddExp)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestAddInferShape", "TestAddInferShape",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto incast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto ubTensor1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto ubTensor2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto ubTensor3 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto outCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    outCast->UpdateDynValidShape({CreateTestScalarVar("output_0_Dim_0"), CreateTestScalarVar("output_0_Dim_1")});

    AddCopyOp(currFunctionPtr, Opcode::OP_COPY_IN, incast1, ubTensor1,
              CreateCopyInAttribute(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme,
                                    {"Input_0_Dim_0", "Input_0_Dim_1"}));
    AddCopyOp(currFunctionPtr, Opcode::OP_COPY_IN, incast2, ubTensor2,
              CreateCopyInAttribute(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme,
                                    {"Input_1_Dim_0", "Input_1_Dim_1"}));

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ADD, {ubTensor1, ubTensor2}, {ubTensor3});
    auto tmpCast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    AddCopyOp(currFunctionPtr, Opcode::OP_COPY_OUT, ubTensor3, tmpCast,
              CreateCopyOutAttribute(MEM_UB, OpImmediate::Specified({0, 0}), shapeImme, shapeImme));

    auto ubTensor4 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    AddCopyOp(currFunctionPtr, Opcode::OP_COPY_IN, tmpCast, ubTensor4,
              CreateCopyInAttribute(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme));

    auto ubTensor5 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {ubTensor4}, {ubTensor5});

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {ubTensor5}, {outCast});

    AppendFunctionIO(currFunctionPtr, {incast1, incast2}, {outCast});
    RunInferShapeAndExpect(currFunctionPtr, SUCCESS);
}

TEST_F(InferShapeTest, TestReduce)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReduceInferShape",
                                                      "TestReduceInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {4, 8, 16};
    std::vector<int64_t> outshape = {4, 8, 8};
    auto shapeImme = OpImmediate::Specified(inshape);
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, std::vector<SymbolicScalar>{});
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});
    auto inTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, std::vector<SymbolicScalar>{});
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});

    auto copyin_Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme,
                                                         std::vector<OpImmediate>());
    std::vector<OpImmediate> toValidShape = {OpImmediate(CreateTestScalarVar("Input_0_Dim_0")),
                                             OpImmediate(CreateTestScalarVar("Input_0_Dim_1")),
                                             OpImmediate(CreateTestScalarVar("Input_0_Dim_2"))};
    copyin_Attr->SetToDynValidShape(toValidShape);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {inTensor},
                                     [&copyin_Attr](Operation& op) { op.SetOpAttribute(copyin_Attr); });

    auto axis = inshape.size() - 1;
    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_ROWMAX_SINGLE, {inTensor}, {outTensor},
        [&axis](Operation& op) { op.SetAttribute(OP_ATTR_PREFIX + "AXIS", static_cast<int>(axis)); });

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {outTensor}, {outcast});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestView)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TesViewInferShape", "TesViewInferShape",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});

    incast->UpdateDynValidShape({CreateTestScalarVar("input_0_Dim_0"), CreateTestScalarVar("input_0_Dim_1")});
    auto view_Attr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>(), MEM_UNKNOWN,
                                                       std::vector<SymbolicScalar>(), std::vector<SymbolicScalar>());
    view_Attr->SetFromOffset(std::vector<int64_t>(),
                             {CreateTestScalarVar("Offset_0_Dim_0"), CreateTestScalarVar("Offset_0_Dim_1")});
    auto& view_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast}, {outcast},
                                                     [&view_Attr](Operation& op) { op.SetOpAttribute(view_Attr); });

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    std::cout << view_op.GetOOperands()[0]->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestViewAlign)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TesViewInferShape", "TesViewInferShape",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    std::vector<int64_t> offset = {2, 0};
    std::vector<int64_t> viewshape = {8, 4};
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, viewshape, std::vector<SymbolicScalar>{});

    auto view_Attr = std::make_shared<ViewOpAttribute>(std::vector<int64_t>(), MEM_UNKNOWN,
                                                       std::vector<SymbolicScalar>(), std::vector<SymbolicScalar>());
    view_Attr->SetFromOffset(offset, std::vector<SymbolicScalar>());
    auto& view_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast}, {outcast},
                                                     [&view_Attr](Operation& op) { op.SetOpAttribute(view_Attr); });

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    std::cout << view_op.GetOOperands()[0]->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestAssemble)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestAssembleInferShape",
                                                      "TestAssembleInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {8, 16};
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});

    incast->UpdateDynValidShape({CreateTestScalarVar("input_0_Dim_0"), CreateTestScalarVar("input_0_Dim_1")});
    outcast->UpdateDynValidShape({CreateTestScalarVar("output_0_Dim_0"), CreateTestScalarVar("output_0_Dim_1")});
    auto assemble_Attr = std::make_shared<AssembleOpAttribute>(
        MEM_UNKNOWN, std::vector<int64_t>(), std::vector<SymbolicScalar>(), std::vector<SymbolicScalar>());

    auto dynOffset = {CreateTestScalarVar("DynOffset_0_Dim_0"), CreateTestScalarVar("DynOffset_0_Dim_1")};
    assemble_Attr->SetToOffset({2, 2}, dynOffset);
    auto& assemble_op = PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_ASSEMBLE, {incast}, {outcast},
        [&assemble_Attr](Operation& op) { op.SetOpAttribute(assemble_Attr); });

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    std::cout << assemble_op.GetOOperands()[0]->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestFailCopyOut)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestCopyOutInferShape",
                                                      "TestCopyOutInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {8, 16};
    std::vector<int64_t> outshape = {8, 8};
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, std::vector<SymbolicScalar>{});
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {incast}, {outcast});
    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), FAILED);
}

TEST_F(InferShapeTest, TestCopyOut)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestCopyOutInferShape",
                                                      "TestCopyOutInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {8, 16};
    std::vector<int64_t> outshape = {8, 16};
    auto toOffsetImme = OpImmediate::Specified({4, 4});
    auto inshapeImme = OpImmediate::Specified(inshape);
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, std::vector<SymbolicScalar>{});
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});
    incast->UpdateDynValidShape({CreateTestScalarVar("input_0_Dim_0"), CreateTestScalarVar("input_0_Dim_1")});

    auto copyout_Attr = std::make_shared<CopyOpAttribute>(MEM_DEVICE_DDR, toOffsetImme, inshapeImme, inshapeImme,
                                                          std::vector<OpImmediate>());
    auto& copyout_op = PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_COPY_OUT, {incast}, {outcast},
        [&copyout_Attr](Operation& op) { op.SetOpAttribute(copyout_Attr); });

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    std::cout << copyout_op.GetOOperands()[0]->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestCopyIn)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestCopyInInferShape",
                                                      "TestCopyInInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {8, 16};
    std::vector<int64_t> outshape = {8, 8};
    auto fromOffsetImme = OpImmediate::Specified({4, 4});
    auto inshapeImme = OpImmediate::Specified(inshape);
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, std::vector<SymbolicScalar>{});
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});
    incast->UpdateDynValidShape({CreateTestScalarVar("input_0_Dim_0"), CreateTestScalarVar("input_0_Dim_1")});

    auto copyin_Attr = std::make_shared<CopyOpAttribute>(fromOffsetImme, MEM_UNKNOWN, inshapeImme, inshapeImme,
                                                         OpImmediate::Specified({4, 4}));
    auto& copyin_op = PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {outcast},
        [&copyin_Attr](Operation& op) { op.SetOpAttribute(copyin_Attr); });

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    std::cout << copyin_op.GetOOperands()[0]->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestReshape)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestReshapeInferShape",
                                                      "TestReshapeInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {8, 16};
    std::vector<int64_t> outshape = {4, 4};
    auto shapeImme = OpImmediate::Specified(inshape);
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, std::vector<SymbolicScalar>{});
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});
    auto inTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, std::vector<SymbolicScalar>{});
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});
    incast->UpdateDynValidShape({CreateTestScalarVar("input_0_Dim_0"), CreateTestScalarVar("input_0_Dim_1")});
    outcast->UpdateDynValidShape({CreateTestScalarVar("output_0_Dim_0"), CreateTestScalarVar("output_0_Dim_1")});

    auto copyin_Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UNKNOWN, shapeImme,
                                                         shapeImme, std::vector<OpImmediate>());

    std::vector<OpImmediate> toValidShape = {OpImmediate(CreateTestScalarVar("Input_1_Dim_0")),
                                             OpImmediate(CreateTestScalarVar("Input_1_Dim_1"))};
    copyin_Attr->SetToDynValidShape(toValidShape);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {inTensor},
                                     [&copyin_Attr](Operation& op) { op.SetOpAttribute(copyin_Attr); });

    auto& reshape_op = PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_RESHAPE, {inTensor}, {outTensor});

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {outTensor}, {outcast});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    std::cout << reshape_op.GetOOperands()[0]->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestSHMEM_LOAD)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestSHMEM_LOAD", "TestSHMEM_LOAD",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> inshape0 = {1, 1};
    std::vector<int64_t> inshape1 = {1, 1, 8, 16};
    std::vector<int64_t> outshape = {8, 16};
    auto shapeImme = OpImmediate::Specified(outshape);

    auto incast0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape0, std::vector<SymbolicScalar>{});
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape1, std::vector<SymbolicScalar>{});
    auto shmemLoadOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});

    auto shmemLoad_Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme,
                                                            shapeImme, std::vector<OpImmediate>());

    std::vector<OpImmediate> toValidShape = {OpImmediate(CreateTestScalarVar("Input_0_Dim_0")),
                                             OpImmediate(CreateTestScalarVar("Input_0_Dim_1"))};
    shmemLoad_Attr->SetToDynValidShape(toValidShape);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_SHMEM_LOAD, {incast0, incast1}, {shmemLoadOut},
                                     [&shmemLoad_Attr](Operation& op) { op.SetOpAttribute(shmemLoad_Attr); });
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {shmemLoadOut}, {outcast});

    currFunctionPtr->inCasts_.push_back(incast0);
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_NE(shmemLoadOut->GetDynValidShape().size(), 0);
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestPad)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPadInferShape", "TestPadInferShape",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {2, 2};
    std::vector<int64_t> outshape = {3, 4};
    auto shapeImme = OpImmediate::Specified(inshape);
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, std::vector<SymbolicScalar>{});
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});
    auto inTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, std::vector<SymbolicScalar>{});
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});

    auto copyin_Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme,
                                                         std::vector<OpImmediate>());
    std::vector<OpImmediate> toValidShape = {OpImmediate(CreateTestScalarVar("Input_0_Dim_0")),
                                             OpImmediate(CreateTestScalarVar("Input_0_Dim_1"))};
    copyin_Attr->SetToDynValidShape(toValidShape);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {inTensor},
                                     [&copyin_Attr](Operation& op) { op.SetOpAttribute(copyin_Attr); });

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_PAD, {inTensor}, {outTensor}, [](Operation& op) {
        op.SetAttribute(OP_ATTR_PREFIX + "pad_right", 2);
        op.SetAttribute(OP_ATTR_PREFIX + "pad_bottom", 1);
        op.SetAttribute(OpAttributeKey::scalar, Element(DT_FP32, 0.0f));
    });

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {outTensor}, {outcast});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestFillPad)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestFillPadInferShape",
                                                      "TestFillPadInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> inshape = {3, 4};
    std::vector<int64_t> outshape = {3, 4};
    auto shapeImme = OpImmediate::Specified(inshape);
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, std::vector<SymbolicScalar>{});
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});
    auto inTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, std::vector<SymbolicScalar>{});
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});

    auto copyin_Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme,
                                                         std::vector<OpImmediate>());
    std::vector<OpImmediate> toValidShape = {OpImmediate(CreateTestScalarVar("Input_0_Dim_0")),
                                             OpImmediate(CreateTestScalarVar("Input_0_Dim_1"))};
    copyin_Attr->SetToDynValidShape(toValidShape);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {inTensor},
                                     [&copyin_Attr](Operation& op) { op.SetOpAttribute(copyin_Attr); });

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_FILLPAD, {inTensor}, {outTensor}, [](Operation& op) {
        op.SetAttribute(OpAttributeKey::scalar, Element(DT_FP32, 0.0f));
    });

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {outTensor}, {outcast});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestIndexOutCast)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestIndexOutCast", "TestIndexOutCast",
                                                      nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> inshape0 = {1, 1};
    std::vector<int64_t> inshape1 = {2, 2};
    std::vector<int64_t> inshape2 = {4, 4};
    std::vector<int64_t> outshape = {4, 4};

    auto incast0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape0, std::vector<SymbolicScalar>{});
    incast0->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto view0 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape0, std::vector<SymbolicScalar>{});
    view0->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape1, std::vector<SymbolicScalar>{});
    incast1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto view1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape1, std::vector<SymbolicScalar>{});
    view1->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto incast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape2, std::vector<SymbolicScalar>{});
    std::vector<SymbolicScalar> validShape = {CreateTestScalarVar("Input_0_Dim_0"),
                                              CreateTestScalarVar("Input_0_Dim_1")};
    incast2->UpdateDynValidShape(validShape);
    incast2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});
    outcast->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    Offset offsets = {0, 0};
    auto viewOpAttribute0 = std::make_shared<ViewOpAttribute>(offsets);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast0}, {view0},
                                     [&viewOpAttribute0](Operation& op) { op.SetOpAttribute(viewOpAttribute0); });
    auto viewOpAttribute1 = std::make_shared<ViewOpAttribute>(offsets);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_VIEW, {incast1}, {view1},
                                     [&viewOpAttribute1](Operation& op) { op.SetOpAttribute(viewOpAttribute1); });
    auto indexoutcastOpAttr = std::make_shared<CopyOpAttribute>(
        MemoryType::MEM_DEVICE_DDR, OpImmediate::Specified(offsets), OpImmediate::Specified(inshape1),
        OpImmediate::Specified(incast2->tensor->GetDynRawShape()));
    auto& indexoutcastOp = PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_INDEX_OUTCAST, {view0, view1, incast2}, {outcast},
        [&indexoutcastOpAttr](Operation& op) { op.SetOpAttribute(indexoutcastOpAttr); });

    AppendFunctionIO(currFunctionPtr, {incast0, incast1, incast2}, {outcast});
    RunInferShapeAndExpect(currFunctionPtr, SUCCESS);
    EXPECT_NE(outcast->GetDynValidShape().size(), 0);
    auto indexOutCastOpAttribute = std::dynamic_pointer_cast<CopyOpAttribute>(indexoutcastOp.GetOpAttribute());
    const auto& fromDynValidShape = indexOutCastOpAttribute->GetFromDynValidShape();
    EXPECT_NE(fromDynValidShape.size(), 0U);
}

TEST_F(InferShapeTest, TestPermute)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPermuteInferShape",
                                                      "TestPermuteInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> inshape = {2, 3, 4};
    std::vector<int64_t> outshape = {3, 2, 4};
    auto shapeImme = OpImmediate::Specified(inshape);
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, std::vector<SymbolicScalar>{});
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});
    auto inTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, std::vector<SymbolicScalar>{});
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});

    auto copyin_Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0, 0}), MEM_UB, shapeImme,
                                                         shapeImme, std::vector<OpImmediate>());
    std::vector<OpImmediate> toValidShape = {OpImmediate(CreateTestScalarVar("Input_0_Dim_0")),
                                             OpImmediate(CreateTestScalarVar("Input_0_Dim_1")),
                                             OpImmediate(CreateTestScalarVar("Input_0_Dim_2"))};
    copyin_Attr->SetToDynValidShape(toValidShape);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {inTensor},
                                     [&copyin_Attr](Operation& op) { op.SetOpAttribute(copyin_Attr); });

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_PERMUTE, {inTensor}, {outTensor}, [](Operation& op) {
        op.SetAttribute(OpAttributeKey::perm, std::vector<int>{1, 0, 2});
    });

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {outTensor}, {outcast});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestPermuteElement)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestPermuteElemInferShape",
                                                      "TestPermuteElemInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> inshape = {2, 3, 4};
    std::vector<int64_t> outshape = {2, 4, 3};
    auto shapeImme = OpImmediate::Specified(inshape);
    auto incast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, std::vector<SymbolicScalar>{});
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});
    auto inTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, inshape, std::vector<SymbolicScalar>{});
    auto outTensor = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, outshape, std::vector<SymbolicScalar>{});

    auto copyin_Attr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0, 0}), MEM_UB, shapeImme,
                                                         shapeImme, std::vector<OpImmediate>());
    std::vector<OpImmediate> toValidShape = {OpImmediate(CreateTestScalarVar("Input_0_Dim_0")),
                                             OpImmediate(CreateTestScalarVar("Input_0_Dim_1")),
                                             OpImmediate(CreateTestScalarVar("Input_0_Dim_2"))};
    copyin_Attr->SetToDynValidShape(toValidShape);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {inTensor},
                                     [&copyin_Attr](Operation& op) { op.SetOpAttribute(copyin_Attr); });

    PassOperationUtils::AddOperation(
        *currFunctionPtr, Opcode::OP_PERMUTE_ELEMENT, {inTensor}, {outTensor},
        [](Operation& op) { op.SetAttribute(OpAttributeKey::perm, std::vector<int>{0, 2, 1}); });

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {outTensor}, {outcast});

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outcast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    std::cout << currFunctionPtr->Dump() << std::endl;
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

TEST_F(InferShapeTest, TestErfcUnary)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestErfcInferShape",
                                                      "TestErfcInferShape", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensorOut = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    outCast->UpdateDynValidShape({CreateTestScalarVar("output_0_Dim_0"), CreateTestScalarVar("output_0_Dim_1")});

    auto copyinAttr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme,
                                                        std::vector<npu::tile_fwk::OpImmediate>());
    std::vector<npu::tile_fwk::OpImmediate> toValidShape = {OpImmediate(CreateTestScalarVar("Input_0_Dim_0")),
                                                            OpImmediate(CreateTestScalarVar("Input_0_Dim_1"))};
    copyinAttr->SetToDynValidShape(toValidShape);
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_IN, {incast}, {ubTensor},
                                     [&copyinAttr](Operation& op) { op.SetOpAttribute(copyinAttr); });

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ERFC, {ubTensor}, {ubTensorOut});

    auto copyoutAttr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_UB, shapeImme, shapeImme,
                                                         std::vector<npu::tile_fwk::OpImmediate>());
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {ubTensorOut}, {outCast},
                                     [&copyoutAttr](Operation& op) { op.SetOpAttribute(copyoutAttr); });

    currFunctionPtr->inCasts_.push_back(incast);
    currFunctionPtr->outCasts_.push_back(outCast);

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

// BRCB op output whose DynValidShape exceeds tensor shape should be skipped and return SUCCESS.
// Graph:
//   incast1[16,16] --copy_in--> ub1[16,16] ------------------ add --> addOut[16,16] --copy_out--> outcast[16,16]
//                                                            /
//   incast2[16,1]  --copy_in--> ub2[16,1]  --brcb--> brcbOut[16,8](validShape=[16,16] > shape, skipped)
TEST_F(InferShapeTest, BRCBSkipsValidShapeValidation)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "BRCBSkipsValidShapeValidation",
                                                      "BRCBSkipsValidShapeValidation", nullptr);
    ASSERT_TRUE(currFunctionPtr != nullptr);

    std::vector<int64_t> shape1 = {16, 16};
    std::vector<int64_t> shape2 = {16, 1};
    std::vector<int64_t> brcbShape = {16, 8};
    auto shape1Imme = OpImmediate::Specified(shape1);
    auto shape2Imme = OpImmediate::Specified(shape2);

    auto incast1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, std::vector<SymbolicScalar>{});
    auto incast2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, std::vector<SymbolicScalar>{});
    auto ub1 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, std::vector<SymbolicScalar>{});
    auto ub2 = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape2, std::vector<SymbolicScalar>{});
    auto brcbOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, brcbShape, std::vector<SymbolicScalar>{});
    auto addOut = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, std::vector<SymbolicScalar>{});
    auto outcast = npu::tile_fwk::IRBuilder().CreateTensorVar(DT_FP32, shape1, std::vector<SymbolicScalar>{});

    auto copyin1Attr = CreateCopyInAttribute(OpImmediate::Specified({0, 0}), MEM_UB, shape1Imme, shape1Imme,
                                             {"Input_0_Dim_0", "Input_0_Dim_1"});
    AddCopyOp(currFunctionPtr, Opcode::OP_COPY_IN, incast1, ub1, copyin1Attr);

    auto copyin2Attr = CreateCopyInAttribute(OpImmediate::Specified({0, 0}), MEM_UB, shape2Imme, shape2Imme,
                                             {"Input_1_Dim_0", "Input_1_Dim_1"});
    AddCopyOp(currFunctionPtr, Opcode::OP_COPY_IN, incast2, ub2, copyin2Attr);

    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_BRCB, {ub2}, {brcbOut});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_ADD, {ub1, brcbOut}, {addOut});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_COPY_OUT, {addOut}, {outcast});

    ub1->UpdateDynValidShape(CreateTestConstIntVector(shape1));
    ub2->UpdateDynValidShape(CreateTestConstIntVector(shape2));
    brcbOut->UpdateDynValidShape({IRBuilder().CreateConstInt(16), IRBuilder().CreateConstInt(16)});
    addOut->UpdateDynValidShape(CreateTestConstIntVector(shape1));
    outcast->UpdateDynValidShape(CreateTestConstIntVector(shape1));

    AppendFunctionIO(currFunctionPtr, {incast1, incast2}, {outcast});

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), SUCCESS);
}

// A concrete DynValidShape larger than its tensor shape must fail post-check.
TEST_F(InferShapeTest, CheckDynValidShapeFailed)
{
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "CheckDynValidShapeFailed",
                                                      "CheckDynValidShapeFailed", nullptr);
    ASSERT_TRUE(currFunctionPtr != nullptr);

    const std::vector<int64_t> shape = {16, 16};
    auto input = IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    auto output = IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
    output->UpdateDynValidShape({IRBuilder().CreateConstInt(16), IRBuilder().CreateConstInt(17)});
    PassOperationUtils::AddOperation(*currFunctionPtr, Opcode::OP_EXP, {input}, {output});
    AppendFunctionIO(currFunctionPtr, {input}, {output});

    InferDynShape inferShapeTest;
    inferShapeTest.RunOnFunction(*currFunctionPtr);
    EXPECT_EQ(inferShapeTest.PostCheck(*currFunctionPtr), FAILED);
}

// L0C(NZ) -> UB tile 直通搬运: 输出 valid 由 l0cValidMN 与 realCutW 属性推导 (VIEW 路径)
TEST_F(InferShapeTest, L0CCopyUBConvDerivesValidShapeFromL0cAttrs)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "l0c_conv_view_infer", "l0c_conv_view_infer",
                                               nullptr);
    const std::vector<int64_t> l0cShape = {56, 48};
    const std::vector<int64_t> ubShape = {1, 64, 8, 8};
    auto l0cInput = IRBuilder().CreateTensorVar(DT_FP16, l0cShape, SymbolicScalar::FromConcrete(l0cShape));
    auto ubOutput = IRBuilder().CreateTensorVar(DT_FP16, ubShape, SymbolicScalar::FromConcrete(ubShape));
    auto copyAttr = CreateCopyOutAttribute(MEM_UB, OpImmediate::Specified({0, 0, 0, 0}),
                                           OpImmediate::Specified(ubShape), OpImmediate::Specified(ubShape));
    auto& convCopyOp = PassOperationUtils::AddOperation(
        *function, Opcode::OP_L0C_COPY_UB_CONV, {l0cInput}, {ubOutput},
        [&copyAttr](Operation& op) {
            op.SetOpAttribute(copyAttr);
            op.SetAttribute(OpAttributeKey::l0cValidMN,
                            std::vector<SymbolicScalar>{SymbolicScalar(56), SymbolicScalar(48)});
            op.SetAttribute(Conv::LoadStoreConvOpAttributeKey::realCutW, SymbolicScalar(6));
        },
        ir::Span::Unknown(), false);
    AppendFunctionIO(function, {l0cInput}, {ubOutput});

    InferShapeRegistry::GetInstance().CallInferShapeFunc(&convCopyOp);

    const auto& outValid = ubOutput->GetDynValidShape();
    ASSERT_EQ(outValid.size(), 4U);
    EXPECT_EQ(outValid[0].Concrete(), 1);
    EXPECT_EQ(outValid[1].Concrete(), 48);
    EXPECT_EQ(outValid[2].Concrete(), 9);
    EXPECT_EQ(outValid[3].Concrete(), 6);
}

// INSERT (小搬大) 路径: 多个 L0C copy 拼入同一 UB tile, 各窗口贡献按维取 max 合并
TEST_F(InferShapeTest, L0CCopyUBConvInsertPathMergesProducerWindow)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "l0c_conv_insert_infer", "l0c_conv_insert_infer",
                                               nullptr);
    const std::vector<int64_t> l0cShape = {128, 64};
    const std::vector<int64_t> ubShape = {1, 64, 8, 8};
    auto producerInput = IRBuilder().CreateTensorVar(DT_FP16, l0cShape, SymbolicScalar::FromConcrete(l0cShape));
    auto insertInput = IRBuilder().CreateTensorVar(DT_FP16, l0cShape, SymbolicScalar::FromConcrete(l0cShape));
    // INSERT 路径下输出 valid 是各 copy 窗口逐维 max 的累计结果, 初始必须为空 (未承载数据)
    auto ubOutput = IRBuilder().CreateTensorVar(DT_FP16, ubShape, std::vector<SymbolicScalar>{});
    auto producerAttr = CreateCopyOutAttribute(MEM_UB, OpImmediate::Specified({0, 0, 0, 0}),
                                               OpImmediate::Specified(ubShape), OpImmediate::Specified(ubShape));
    auto insertAttr = CreateCopyOutAttribute(MEM_UB, OpImmediate::Specified({0, 0, 0, 0}),
                                             OpImmediate::Specified(ubShape), OpImmediate::Specified(ubShape));
    PassOperationUtils::AddOperation(
        *function, Opcode::OP_L0C_COPY_UB_CONV, {producerInput}, {ubOutput},
        [&producerAttr](Operation& op) {
            op.SetOpAttribute(producerAttr);
            op.SetAttribute(OpAttributeKey::l0cValidMN,
                            std::vector<SymbolicScalar>{SymbolicScalar(56), SymbolicScalar(48)});
            op.SetAttribute(Conv::LoadStoreConvOpAttributeKey::realCutW, SymbolicScalar(6));
        },
        ir::Span::Unknown(), false);
    auto& insertOp = PassOperationUtils::AddOperation(
        *function, Opcode::OP_L0C_COPY_UB_CONV, {insertInput}, {ubOutput},
        [&insertAttr](Operation& op) {
            op.SetOpAttribute(insertAttr);
            op.SetAttribute(OpAttributeKey::l0cValidMN,
                            std::vector<SymbolicScalar>{SymbolicScalar(32), SymbolicScalar(16)});
            op.SetAttribute(Conv::LoadStoreConvOpAttributeKey::realCutW, SymbolicScalar(4));
        },
        ir::Span::Unknown(), false);
    AppendFunctionIO(function, {producerInput, insertInput}, {ubOutput});

    InferShapeRegistry::GetInstance().CallInferShapeFunc(&insertOp);

    const auto& outValid = ubOutput->GetDynValidShape();
    ASSERT_EQ(outValid.size(), 4U);
    EXPECT_EQ(outValid[0].Concrete(), 1);
    EXPECT_EQ(outValid[1].Concrete(), 48);
    EXPECT_EQ(outValid[2].Concrete(), 9);
    EXPECT_EQ(outValid[3].Concrete(), 6);
}

TEST_F(InferShapeTest, TestDivModInferShape)
{
    struct DivModCase {
        DataType dtype;
        NPUArch arch;
    };
    const std::vector<DivModCase> cases = {
        {DT_INT8, NPUArch::DAV_2201},   {DT_UINT8, NPUArch::DAV_2201},  {DT_INT16, NPUArch::DAV_2201},
        {DT_INT32, NPUArch::DAV_2201},  {DT_UINT16, NPUArch::DAV_2201}, {DT_INT8, NPUArch::DAV_3510},
        {DT_UINT8, NPUArch::DAV_3510},  {DT_INT16, NPUArch::DAV_3510},  {DT_INT32, NPUArch::DAV_3510},
        {DT_INT64, NPUArch::DAV_3510},  {DT_UINT16, NPUArch::DAV_3510}, {DT_UINT32, NPUArch::DAV_3510},
        {DT_UINT64, NPUArch::DAV_3510},
    };
    for (const auto& c : cases) {
        Program::GetInstance().Reset();
        config::Reset();
        Platform::Instance().GetSoc().SetNPUArch(c.arch);

        auto func = std::make_shared<Function>(Program::GetInstance(), "TestDivModInferShape", "TestDivModInferShape",
                                               nullptr);
        std::vector<int64_t> shape = {8, 16};
        auto in0 = IRBuilder().CreateTensorVar(c.dtype, shape, std::vector<SymbolicScalar>{});
        auto in1 = IRBuilder().CreateTensorVar(c.dtype, shape, std::vector<SymbolicScalar>{});
        auto quotient = IRBuilder().CreateTensorVar(c.dtype, shape, std::vector<SymbolicScalar>{});
        auto remainder = IRBuilder().CreateTensorVar(c.dtype, shape, std::vector<SymbolicScalar>{});
        auto tmp = IRBuilder().CreateTensorVar(DT_FP32, shape, std::vector<SymbolicScalar>{});
        auto validShape = std::vector<SymbolicScalar>{CreateTestScalarVar("divmod_dim0"),
                                                      CreateTestScalarVar("divmod_dim1")};
        in0->UpdateDynValidShape(validShape);
        in1->UpdateDynValidShape(validShape);

        auto& op = func->AddOperation(Opcode::OP_DIVMOD, {in0, in1}, {quotient, remainder, tmp});
        InferShapeRegistry::GetInstance().CallInferShapeFunc(&op);
        EXPECT_EQ(quotient->GetDynValidShape().size(), validShape.size());
        EXPECT_EQ(remainder->GetDynValidShape().size(), validShape.size());
        EXPECT_EQ(tmp->GetDynValidShape().size(), validShape.size());
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

// VIEW (大搬小) 路径 InferFunc 只是 l0cValidMN/realCutW 的消费者: 按维裁剪推导本窗口的 NCHW valid,
// 不得把裁剪结果回写 op 属性 (属性刷新由 GenerateMoveOp 按view窗口负责)。
// 若恢复回写 (旧实现写 l0cValidMN={validRows,validN}、realCutW=validW), 矩阵级 (M,N) 与切割宽
// 会被 UB tile 形状裁剪, 后续 copy 循环与 producer 窗口合并将读到被破坏的源数据。
TEST_F(InferShapeTest, L0CCopyUBConvViewPathDoesNotRewriteL0cAttrs)
{
    auto function = std::make_shared<Function>(Program::GetInstance(), "l0c_conv_view_keep_attrs",
                                               "l0c_conv_view_keep_attrs", nullptr);
    // L0C 源块 (M=256, N=32) 远大于 UB tile 窗口; realCutW=12 为切割级宽度, 跨越本 tile 的 W=8
    const std::vector<int64_t> l0cShape = {256, 32};
    const std::vector<int64_t> ubShape = {1, 16, 8, 8};
    auto l0cInput = IRBuilder().CreateTensorVar(DT_FP16, l0cShape, SymbolicScalar::FromConcrete(l0cShape));
    auto ubOutput = IRBuilder().CreateTensorVar(DT_FP16, ubShape, SymbolicScalar::FromConcrete(ubShape));
    auto copyAttr = CreateCopyOutAttribute(MEM_UB, OpImmediate::Specified({0, 0, 0, 0}),
                                           OpImmediate::Specified(ubShape), OpImmediate::Specified(ubShape));
    auto& convCopyOp = PassOperationUtils::AddOperation(
        *function, Opcode::OP_L0C_COPY_UB_CONV, {l0cInput}, {ubOutput},
        [&copyAttr](Operation& op) {
            op.SetOpAttribute(copyAttr);
            op.SetAttribute(OpAttributeKey::l0cValidMN,
                            std::vector<SymbolicScalar>{SymbolicScalar(256), SymbolicScalar(32)});
            op.SetAttribute(Conv::LoadStoreConvOpAttributeKey::realCutW, SymbolicScalar(12));
        },
        ir::Span::Unknown(), false);
    AppendFunctionIO(function, {l0cInput}, {ubOutput});

    InferShapeRegistry::GetInstance().CallInferShapeFunc(&convCopyOp);

    // 窗口 NCHW valid 按维裁剪推导: n=1, c=min(32,16)=16, h=min(256,8*8)/min(12,8)=64/8=8, w=min(12,8)=8
    const auto& outValid = ubOutput->GetDynValidShape();
    ASSERT_EQ(outValid.size(), 4U);
    EXPECT_EQ(outValid[0].Concrete(), 1);
    EXPECT_EQ(outValid[1].Concrete(), 16);
    EXPECT_EQ(outValid[2].Concrete(), 8);
    EXPECT_EQ(outValid[3].Concrete(), 8);

    // 属性保持矩阵级源值不变: l0cValidMN 仍为 (256,32) 而非裁剪后的 (64,16)
    auto validMN = convCopyOp.GetAttr<std::vector<SymbolicScalar>>(OpAttributeKey::l0cValidMN);
    ASSERT_NE(validMN, nullptr);
    ASSERT_EQ(validMN->size(), 2U);
    EXPECT_TRUE((*validMN)[0].ConcreteValid());
    EXPECT_EQ((*validMN)[0].Concrete(), 256);
    EXPECT_TRUE((*validMN)[1].ConcreteValid());
    EXPECT_EQ((*validMN)[1].Concrete(), 32);

    // realCutW 仍为切割级 12, 而非裁剪后的 8
    SymbolicScalar realCutWAttr;
    ASSERT_TRUE(convCopyOp.GetAttr(Conv::LoadStoreConvOpAttributeKey::realCutW, realCutWAttr));
    EXPECT_TRUE(realCutWAttr.ConcreteValid());
    EXPECT_EQ(realCutWAttr.Concrete(), 12);
}

} // namespace tile_fwk
} // namespace npu
