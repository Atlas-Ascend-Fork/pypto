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
 * \file test_assign_memory_type_token.cpp
 * \brief Unit test for assign_memory_type token (RAW scenario) adaptation.
 */

#include <gtest/gtest.h>
#include <vector>
#include "interface/function/function.h"
#include "interface/tensor/irbuilder.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#define private public
#define protected public
#include "passes/tile_graph_pass/data_path/assign_memory_type.h"
#undef private
#undef protected
#include "passes/tile_graph_pass/data_path/assign_memory_type.h"

using namespace npu::tile_fwk;

class AssignMemoryTypeTokenTest : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetPlatformConfig(KEY_TEST_IS_TIG, true);
    }
    void TearDown() override
    {
        Program::GetInstance().Reset();
        Program::GetInstance().lastFunc_ = nullptr;
        Program::GetInstance().currentDynamicFunctionPtr_ = nullptr;
        config::SetBuildStatic(false);
        config::SetHostOption(COMPILE_STAGE, CS_ALL_COMPLETE);
    }
};

TEST_F(AssignMemoryTypeTokenTest, GetLogicalTensorsByRawTensorBasic)
{
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("AmtTokenBasic")
    {
        Tensor a = Exp(input);
        output = Add(a, Element(DT_FP32, 1.0));
    }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_AmtTokenBasic");
    ASSERT_NE(func, nullptr);

    AssignMemoryType amt;
    auto opList = func->Operations(false).DuplicatedOpList();
    ASSERT_GE(opList.size(), 2);

    auto tensors = amt.GetLogicalTensorsByRawTensor(*func, opList[0]->oOperand.front());
    EXPECT_FALSE(tensors.empty());

    auto inputTensors = amt.GetLogicalTensorsByRawTensor(*func, opList[0]->iOperand.front());
    EXPECT_FALSE(inputTensors.empty());
}

TEST_F(AssignMemoryTypeTokenTest, GetLogicalTensorsByRawTensorNullInput)
{
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("AmtTokenNull") { output = Exp(input); }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_AmtTokenNull");
    ASSERT_NE(func, nullptr);

    AssignMemoryType amt;
    auto result = amt.GetLogicalTensorsByRawTensor(*func, nullptr);
    EXPECT_TRUE(result.empty());
}

TEST_F(AssignMemoryTypeTokenTest, ResolveInconsistentRawTensorMemoryTypesFallback)
{
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("AmtTokenInconsistent") { output = Exp(input); }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_AmtTokenInconsistent");
    ASSERT_NE(func, nullptr);

    auto opList = func->Operations(false).DuplicatedOpList();
    ASSERT_GE(opList.size(), 2);

    auto rawTensor = std::make_shared<RawTensor>(DT_FP32, std::vector<int64_t>{32, 32}, TileOpFormat::TILEOP_ND,
                                                 "sharedRaw");

    IRBuilder builder;
    auto srcTile = builder.CreateTensorVar(DT_FP32, std::vector<int64_t>{16, 32}, TileOpFormat::TILEOP_ND, "srcTile");
    auto lt1 = builder.CreateTensorVar(rawTensor, std::vector<int64_t>{0, 0}, std::vector<int64_t>{16, 32}, {});
    auto lt2 = builder.CreateTensorVar(rawTensor, std::vector<int64_t>{0, 16}, std::vector<int64_t>{16, 32}, {});

    (void)func->AddRawOperation(Opcode::OP_ASSEMBLE, {srcTile}, {lt1});
    (void)func->AddRawOperation(Opcode::OP_ASSEMBLE, {srcTile}, {lt2});

    lt1->SetMemoryTypeOriginal(MemoryType::MEM_L1, true);
    lt2->SetMemoryTypeOriginal(MemoryType::MEM_L0A, true);

    AssignMemoryType amt;
    EXPECT_EQ(amt.ResolveInconsistentRawTensorMemoryTypes(*func), SUCCESS);

    EXPECT_EQ(lt1->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
    EXPECT_EQ(lt2->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
}

TEST_F(AssignMemoryTypeTokenTest, ResolveInconsistentRawTensorMemoryTypesConsistentNoop)
{
    TileShape::Current().SetVecTile({32, 32});
    std::vector<int64_t> shape{32, 32};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("AmtTokenConsistent") { output = Exp(input); }
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_AmtTokenConsistent");
    ASSERT_NE(func, nullptr);

    auto rawTensor = std::make_shared<RawTensor>(DT_FP32, std::vector<int64_t>{32, 32}, TileOpFormat::TILEOP_ND,
                                                 "sharedRaw2");

    IRBuilder builder;
    auto srcTile = builder.CreateTensorVar(DT_FP32, std::vector<int64_t>{16, 32}, TileOpFormat::TILEOP_ND, "srcTile2");
    auto lt1 = builder.CreateTensorVar(rawTensor, std::vector<int64_t>{0, 0}, std::vector<int64_t>{16, 32}, {});
    auto lt2 = builder.CreateTensorVar(rawTensor, std::vector<int64_t>{0, 16}, std::vector<int64_t>{16, 32}, {});

    (void)func->AddRawOperation(Opcode::OP_ASSEMBLE, {srcTile}, {lt1});
    (void)func->AddRawOperation(Opcode::OP_ASSEMBLE, {srcTile}, {lt2});

    lt1->SetMemoryTypeOriginal(MemoryType::MEM_L1, true);
    lt2->SetMemoryTypeOriginal(MemoryType::MEM_L1, true);

    AssignMemoryType amt;
    EXPECT_EQ(amt.ResolveInconsistentRawTensorMemoryTypes(*func), SUCCESS);

    EXPECT_EQ(lt1->GetMemoryTypeOriginal(), MemoryType::MEM_L1);
    EXPECT_EQ(lt2->GetMemoryTypeOriginal(), MemoryType::MEM_L1);
}

void CheckReshapeStorage(bool inplace)
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "WritableReshape", "WritableReshape", nullptr);
    IRBuilder builder;
    auto base = builder.CreateTensorVar(DT_FP32, {32, 16});
    base->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
    auto alias = builder.CreateTensorVar(DT_FP32, {32, 1, 16});
    alias->SetMemoryTypeBoth(MemoryType::MEM_UB);
    auto& reshape = func->AddRawOperation(Opcode::OP_RESHAPE, {base}, {alias});
    reshape.SetAttribute("op_attr_isInplace", inplace);
    auto patch = builder.CreateTensorVar(DT_FP32, {8, 1, 16});
    auto& write = func->AddRawOperation(Opcode::OP_CONTRACT, {patch}, {alias});
    write.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{8, 0, 0}));

    AssignMemoryType pass;
    pass.inserter.UpdateTensorTobeMap(base, reshape, MemoryType::MEM_UB, "test reshape consumer requirement");
    ASSERT_EQ(pass.InferReshapeMemoryType(reshape), SUCCESS);
    EXPECT_EQ(base->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
    auto expected = inplace ? MemoryType::MEM_DEVICE_DDR : MemoryType::MEM_UB;
    EXPECT_EQ(pass.inserter.GetRequirementOrUnknown(base, reshape), expected);
    EXPECT_EQ(alias->GetMemoryTypeOriginal(), expected);
}

TEST_F(AssignMemoryTypeTokenTest, WritableInplaceReshapeKeepsSharedDdrStorage) { CheckReshapeStorage(true); }

TEST_F(AssignMemoryTypeTokenTest, WritableCopyReshapeKeepsIndependentUbStorage) { CheckReshapeStorage(false); }

void CheckWritableReshapeStorage(MemoryType original, MemoryType expected, const Shape& shape = {32, 16},
                                 const Offset& offset = {8, 0, 0})
{
    auto func = std::make_shared<Function>(Program::GetInstance(), "WritableStorage", "WritableStorage", nullptr);
    IRBuilder builder;
    auto base = builder.CreateTensorVar(DT_FP32, shape);
    base->SetMemoryTypeBoth(original);
    auto alias = builder.CreateTensorVar(DT_FP32, {shape[0], 1, shape[1]});
    alias->SetMemoryTypeBoth(original == MemoryType::MEM_UB ? MemoryType::MEM_DEVICE_DDR : MemoryType::MEM_UB);
    auto& reshape = func->AddRawOperation(Opcode::OP_RESHAPE, {base}, {alias});
    reshape.SetAttribute("op_attr_isInplace", true);
    auto patch = builder.CreateTensorVar(DT_FP32, {8, 1, 8});
    auto patchType = original == MemoryType::MEM_L1 ? MemoryType::MEM_L1 : MemoryType::MEM_UB;
    patch->SetMemoryTypeBoth(patchType);
    auto& write = func->AddRawOperation(Opcode::OP_CONTRACT, {patch}, {alias});
    write.SetOpAttribute(std::make_shared<AssembleOpAttribute>(patchType, offset));

    AssignMemoryType pass;
    pass.inserter.UpdateTensorTobeMap(base, reshape, MemoryType::MEM_DEVICE_DDR, "test previous requirement");
    ASSERT_EQ(pass.InferReshapeMemoryType(reshape), SUCCESS);
    auto inferred = original == MemoryType::MEM_UNKNOWN ? MemoryType::MEM_DEVICE_DDR : original;
    EXPECT_EQ(base->GetMemoryTypeOriginal(), inferred);
    EXPECT_EQ(pass.inserter.GetRequirementOrUnknown(base, reshape), inferred);
    EXPECT_EQ(alias->GetMemoryTypeOriginal(), inferred);
    ASSERT_EQ(pass.ResolveWritableReshapeMemoryTypes(*func), SUCCESS);
    EXPECT_EQ(base->GetMemoryTypeOriginal(), expected);
    EXPECT_EQ(pass.inserter.GetRequirementOrUnknown(base, reshape), expected);
    EXPECT_EQ(alias->GetMemoryTypeOriginal(), expected);
}

TEST_F(AssignMemoryTypeTokenTest, WritableInplaceReshapePreservesUbStorage)
{
    CheckWritableReshapeStorage(MemoryType::MEM_UB, MemoryType::MEM_UB);
}

TEST_F(AssignMemoryTypeTokenTest, WritableInplaceReshapePreservesL1Storage)
{
    CheckWritableReshapeStorage(MemoryType::MEM_L1, MemoryType::MEM_L1);
}

TEST_F(AssignMemoryTypeTokenTest, WritableInplaceReshapeUnknownStorageFallsBackToDdr)
{
    CheckWritableReshapeStorage(MemoryType::MEM_UNKNOWN, MemoryType::MEM_DEVICE_DDR);
}

TEST_F(AssignMemoryTypeTokenTest, WritableInplaceReshapeOversizedUbFallsBackToDdr)
{
    CheckWritableReshapeStorage(MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR, {1024, 1024});
}

TEST_F(AssignMemoryTypeTokenTest, WritableInplaceReshapeUnalignedWriteFallsBackToDdr)
{
    CheckWritableReshapeStorage(MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR, {32, 16}, {8, 0, 1});
}

TEST_F(AssignMemoryTypeTokenTest, WritableInplaceReshapeUnalignedL1WriteFallsBackToDdr)
{
    CheckWritableReshapeStorage(MemoryType::MEM_L1, MemoryType::MEM_DEVICE_DDR, {32, 16}, {8, 0, 1});
}

TEST_F(AssignMemoryTypeTokenTest, WritableInplaceReshapeAdvancedL0cToUbKeepsStorage)
{
    struct ArchGuard {
        NPUArch original = Platform::Instance().GetSoc().GetNPUArch();
        ~ArchGuard() { Platform::Instance().GetSoc().SetNPUArch(original); }
    } archGuard;
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    auto func = std::make_shared<Function>(Program::GetInstance(), "AdvancedWritableReshape", "AdvancedWritableReshape",
                                           nullptr);
    IRBuilder builder;
    auto base = builder.CreateTensorVar(DT_FP32, {32, 16});
    auto alias = builder.CreateTensorVar(DT_FP32, {32, 1, 16});
    base->SetMemoryTypeBoth(MemoryType::MEM_UB);
    alias->SetMemoryTypeBoth(MemoryType::MEM_UB);
    auto& reshape = func->AddRawOperation(Opcode::OP_RESHAPE, {base}, {alias});
    reshape.SetAttribute("op_attr_isInplace", true);
    auto patch = builder.CreateTensorVar(DT_FP32, {8, 1, 8});
    patch->SetMemoryTypeBoth(MemoryType::MEM_L0C);
    auto& write = func->AddRawOperation(Opcode::OP_CONTRACT, {patch}, {alias});
    write.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_L0C, std::vector<int64_t>{8, 0, 1}));

    AssignMemoryType pass;
    bool aligned = true;
    ASSERT_EQ(pass.IsAssembleToOffsetAligned(write, alias, aligned), SUCCESS);
    ASSERT_FALSE(aligned);
    ASSERT_EQ(pass.InferReshapeMemoryType(reshape), SUCCESS);
    ASSERT_EQ(pass.ResolveWritableReshapeMemoryTypes(*func), SUCCESS);
    EXPECT_EQ(base->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(alias->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
    EXPECT_EQ(pass.inserter.GetRequirementOrUnknown(base, reshape), MemoryType::MEM_UB);
}

TEST_F(AssignMemoryTypeTokenTest, WritableReshapeChainFollowsRawVersionDdrFallback)
{
    for (bool aliasVersion : {false, true}) {
        auto func = std::make_shared<Function>(Program::GetInstance(), "WritableChain", "WritableChain", nullptr);
        IRBuilder builder;
        auto base = builder.CreateTensorVar(DT_FP32, {32, 16});
        auto alias = builder.CreateTensorVar(DT_FP32, {32, 1, 16});
        auto tail = builder.CreateTensorVar(DT_FP32, {32, 16});
        for (const auto& tensor : {base, alias, tail}) {
            tensor->SetMemoryTypeBoth(MemoryType::MEM_UB);
        }
        auto basePatch = builder.CreateTensorVar(DT_FP32, {32, 16});
        basePatch->SetMemoryTypeBoth(MemoryType::MEM_UB);
        auto& baseWrite = func->AddRawOperation(Opcode::OP_CONTRACT, {basePatch}, {base});
        baseWrite.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_UB, std::vector<int64_t>{0, 0}));
        auto& reshape = func->AddRawOperation(Opcode::OP_RESHAPE, {base}, {alias});
        reshape.SetAttribute("op_attr_isInplace", true);
        auto& nextReshape = func->AddRawOperation(Opcode::OP_RESHAPE, {alias}, {tail});
        nextReshape.SetAttribute("op_attr_isInplace", true);
        auto patch = builder.CreateTensorVar(DT_FP32, {8, 1, 16});
        patch->SetMemoryTypeBoth(MemoryType::MEM_UB);
        auto& write = func->AddRawOperation(Opcode::OP_CONTRACT, {patch}, {alias});
        write.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_UB, std::vector<int64_t>{8, 0, 0}));
        auto shared = aliasVersion ? alias : base;
        auto version = builder.CreateTensorVar(shared->GetRawTensor(), shared->GetOffset(), shared->GetShape(), {});
        version->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR);
        auto& versionWrite = func->AddRawOperation(Opcode::OP_CONTRACT, {aliasVersion ? patch : basePatch}, {version});
        versionWrite.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_UB, shared->GetOffset()));

        AssignMemoryType pass;
        ASSERT_EQ(pass.InferReshapeMemoryType(reshape), SUCCESS);
        ASSERT_EQ(pass.InferReshapeMemoryType(nextReshape), SUCCESS);
        ASSERT_EQ(base->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
        ASSERT_EQ(alias->GetMemoryTypeOriginal(), MemoryType::MEM_UB);
        ASSERT_EQ(pass.ResolveInconsistentRawTensorMemoryTypes(*func), SUCCESS);
        EXPECT_EQ(shared->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
        ASSERT_EQ(pass.ResolveWritableReshapeMemoryTypes(*func), SUCCESS);
        for (const auto& tensor : {base, alias, tail, version}) {
            EXPECT_EQ(tensor->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR);
        }
        EXPECT_EQ(pass.inserter.GetRequirementOrUnknown(base, reshape), MemoryType::MEM_DEVICE_DDR);
        EXPECT_EQ(pass.inserter.GetRequirementOrUnknown(alias, nextReshape), MemoryType::MEM_DEVICE_DDR);
    }
}
