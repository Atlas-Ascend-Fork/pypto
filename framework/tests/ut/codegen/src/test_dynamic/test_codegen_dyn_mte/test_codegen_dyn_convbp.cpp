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
 * \file test_codegen_dyn_convbp.cpp
 * \brief Unit test for convbp codegen.
 */

#include "gtest/gtest.h"

#include "interface/operation/opcode.h"
#include "tilefwk/data_type.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/npu/cloudnpu/codegen_op_cloudnpu.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {

constexpr int64_t N0 = 16;

class TestCodegenDynConvBp : public CodegenTestBase {
public:
    TestCodegenDynConvBp()
        : CodegenTestBase({.compileStage = CS_CODEGEN_INSTRUCTION,
                           .setTileTensor = true,
                           .tileTensorValue = true,
                           .setIdGen = true,
                           .resetTileTensorOnTearDown = true})
    {}
};

Function* GetFunctionConvBp(const std::string& funcName)
{
    const std::vector<int64_t> shape = {64, 64};
    ConvBp::ConvBpTileL1Info l1TileShape(16, 144, 16);
    ConvBp::ConvBpTileL0Info l0TileShape(16, 144, 16);
    TileShape::Current().SetConvBpInputTile(l1TileShape, l0TileShape);

    auto function = GenMockFuncDyn(funcName, shape);
    return function;
}

void SetConvBpDxDyCopyInOpAttr(Operation& op, const std::vector<int64_t>& offset, const std::vector<int64_t>& gmShape,
                               const std::vector<int64_t>& dstL1Shape)
{
    auto copyAttr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified(offset), MemoryType::MEM_L1,
                                                      OpImmediate::Specified(gmShape), OpImmediate::Specified(gmShape),
                                                      OpImmediate::Specified(dstL1Shape));
    op.SetOpAttribute(copyAttr);
    op.SetAttribute(OpAttributeKey::srcGmConvValidShape, SymbolicScalar::FromConcrete(gmShape));
}

std::string TestConvBpDxDyCopyInBody(const std::string& funcName, bool isConv3D = false,
                                     DataType dtype = DataType::DT_FP16)
{
    auto function = GetFunctionConvBp(funcName);

    std::vector<int64_t> gmShape;
    std::vector<int64_t> dstL1Shape;
    std::vector<int64_t> offset;
    if (isConv3D) {
        gmShape = {1, 1, 2, 14, 14, 16};
        dstL1Shape = {1, 1, 2, 14, 14, 16};
        offset = {0, 0, 0, 0, 0, 0};
    } else {
        gmShape = {1, 1, 14, 14, 16};
        dstL1Shape = {1, 1, 14, 14, 16};
        offset = {0, 0, 0, 0, 0};
    }

    auto gmTensor = CreateConvTensor(*function, dtype, gmShape, MemoryType::MEM_DEVICE_DDR);
    auto localTensor = CreateConvTensor(*function, dtype, dstL1Shape, MemoryType::MEM_L1);

    auto& op = function->AddOperation(Opcode::OP_L1_COPY_IN_CONV_BP_DX_DY, {gmTensor}, {localTensor});
    op.SetAttribute(Conv::LoadStoreConvOpAttributeKey::isConv3D, isConv3D);
    op.SetAttribute(OpAttributeKey::strideH, (int64_t)2);
    op.SetAttribute(OpAttributeKey::strideW, (int64_t)2);
    op.SetAttribute(OpAttributeKey::skipH, (int64_t)0);
    op.SetAttribute(OpAttributeKey::skipW, (int64_t)0);
    op.SetAttribute(OpAttributeKey::gmTensorParamIdxInCall, 0);
    SetConvBpDxDyCopyInOpAttr(op, offset, gmShape, dstL1Shape);

    return GenOpCodeFromOp(*function, op);
}

TEST_F(TestCodegenDynConvBp, L1CopyInConvBpDxDy2D)
{
    std::string res = TestConvBpDxDyCopyInBody("L1CopyInConvBpDxDy2D", false);
    std::string expect =
        R"!!!(TLoadConvBpDxDedy<0>(l1Tensor_0, gmTensor_1, 0, 0, 0, 0, 0, 1, 1, 1, 14, 14, 2, 2, 0, 0);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynConvBp, L1CopyInConvBpDxDy3D)
{
    std::string res = TestConvBpDxDyCopyInBody("L1CopyInConvBpDxDy3D", true);
    std::string expect =
        R"!!!(TLoadConvBpDxDedy<1>(l1Tensor_0, gmTensor_1, 0, 0, 0, 0, 0, 1, 1, 2, 14, 14, 2, 2, 0, 0);
)!!!";
    EXPECT_EQ(res, expect);
}

std::string TestConvBpNZCopyInBody(const std::string& funcName, DataType dtype = DataType::DT_FP16)
{
    auto function = GetFunctionConvBp(funcName);

    std::vector<int64_t> gmShape = {1, 1, 16, 16};
    std::vector<int64_t> dstL1Shape = {1, 1, 16, 16};
    std::vector<int64_t> offset = {0, 0, 0, 0};

    auto gmTensor = CreateConvTensor(*function, dtype, gmShape, MemoryType::MEM_DEVICE_DDR);
    auto localTensor = CreateConvTensor(*function, dtype, dstL1Shape, MemoryType::MEM_L1);

    auto& op = function->AddOperation(Opcode::OP_L1_COPY_IN_CONV_BP, {gmTensor}, {localTensor});
    op.SetAttribute(OpAttributeKey::gmTensorParamIdxInCall, 0);
    auto copyAttr = std::make_shared<CopyOpAttribute>(OpImmediate::Specified(offset), MemoryType::MEM_L1,
                                                      OpImmediate::Specified(gmShape), OpImmediate::Specified(gmShape),
                                                      OpImmediate::Specified(dstL1Shape));
    op.SetOpAttribute(copyAttr);

    return GenOpCodeFromOp(*function, op);
}

TEST_F(TestCodegenDynConvBp, L1CopyInConvBpNZ)
{
    std::string res = TestConvBpNZCopyInBody("L1CopyInConvBpNZ");
    std::string expect =
        R"!!!(TLoadConvBPNZ(l1Tensor_0, gmTensor_1, 0, 0, 0, 0);
)!!!";
    EXPECT_EQ(res, expect);
}

void SetConvBpLoad2DDxAttributes(Operation& op)
{
    op.SetAttribute(OpAttributeKey::postK, (int64_t)0);
    op.SetAttribute(OpAttributeKey::postN, (int64_t)0);
    op.SetAttribute(OpAttributeKey::hwk, (int64_t)9);
    op.SetAttribute(OpAttributeKey::kL0Size, (int64_t)144);
    op.SetAttribute(OpAttributeKey::nL0Size, (int64_t)16);
    op.SetAttribute(OpAttributeKey::k0Idx, (int64_t)0);
    op.SetAttribute(OpAttributeKey::n0Idx, (int64_t)0);
}

std::string TestConvBpLoad2DDxBody(const std::string& funcName)
{
    auto function = GetFunctionConvBp(funcName);

    const std::vector<SymbolicScalar> dynValidShape = {64, 64};
    auto l1Tensor = CreateLogicalTensor({*function, DataType::DT_FP16, MemoryType::MEM_L1, {16, 16}, dynValidShape});
    auto l0Tensor = CreateLogicalTensor({*function, DataType::DT_FP16, MemoryType::MEM_L0B, {16, 16}, dynValidShape});

    std::vector<int64_t> offset = {0, 0};
    std::vector<SymbolicScalar> dynoffset = {0, 0};
    l1Tensor->UpdateOffset(TensorOffset(offset, dynoffset));

    auto& op = function->AddOperation(Opcode::OP_LOAD2DDX_CONV, {l1Tensor}, {l0Tensor});
    SetConvBpLoad2DDxAttributes(op);

    return GenOpCodeFromOp(*function, op);
}

TEST_F(TestCodegenDynConvBp, Load2DDxConv)
{
    std::string res = TestConvBpLoad2DDxBody("Load2DDxConv");
    std::string expect =
        R"!!!(TLoad2DDX(l0bTensor_0, l1Tensor_1, 144, 16, 9, 0, 0);
)!!!";
    EXPECT_EQ(res, expect);
}

std::string TestConvBpDwL0COutBody(const std::string& funcName, const std::vector<int64_t>& l0cShape,
                                   const std::vector<SymbolicScalar>& l0cValidShape,
                                   const std::vector<int64_t>& gmShape, const std::vector<int64_t>& offset)
{
    auto function = GetFunctionConvBp(funcName);
    auto gmTensor = CreateConvTensor(*function, DataType::DT_FP16, gmShape, MemoryType::MEM_DEVICE_DDR, false);
    auto l0cTensor = CreateConvTensor(*function, DataType::DT_FP32, l0cShape, MemoryType::MEM_L0C, false);

    auto& op = function->AddOperation(Opcode::OP_L0C_COPY_OUT_CONV_DW, {l0cTensor}, {gmTensor});
    op.SetAttribute("COPY_OUT_MODE", (int64_t)1); // CopyOutMode::NZ2NZ
    op.SetAttribute("IS_CONV3D", false);
    op.SetAttribute(OpAttributeKey::l0cValidMN, l0cValidShape);
    op.SetAttribute(OpAttributeKey::gmTensorParamIdxInCall, 0);
    op.SetOpAttribute(std::make_shared<CopyOpAttribute>(
        MemoryType::MEM_L0C, OpImmediate::Specified(offset), OpImmediate::Specified(gmShape),
        OpImmediate::Specified(gmShape), OpImmediate::Specified(l0cValidShape)));

    return GenOpCodeFromOp(*function, op);
}

TEST_F(TestCodegenDynConvBp, L0COutConvBpDw)
{
    std::vector<int64_t> l0cShape = {16, 64};
    std::vector<int64_t> gmShape = {4, 1, 16, 16};
    std::vector<int64_t> offset = {2, 1, 0, 0};
    std::string res = TestConvBpDwL0COutBody("L0COutConvBpDw", l0cShape, SymbolicScalar::FromConcrete(l0cShape),
                                             gmShape, offset);
    std::string expect =
        R"!!!(TStoreConvDW<CopyOutMode::NZ2NZ, 0>(gmTensor_0, l0cTensor_1, 2, 1, 0, 0, 0, 16, 64);
)!!!";
    EXPECT_EQ(res, expect);
}

TEST_F(TestCodegenDynConvBp, L0COutConvBpDwTailTensor)
{
    std::vector<int64_t> l0cShape = {16, 64};
    std::vector<SymbolicScalar> l0cValidShape = {16, 33};
    std::vector<int64_t> gmShape = {4, 1, 16, 16};
    std::vector<int64_t> offset = {0, 0, 0, 0};
    std::string res = TestConvBpDwL0COutBody("L0COutConvBpDwTailTensor", l0cShape, l0cValidShape, gmShape, offset);
    std::string expect =
        R"!!!(TStoreConvDW<CopyOutMode::NZ2NZ, 0>(gmTensor_0, l0cTensor_1, 0, 0, 0, 0, 0, 16, 33);
)!!!";
    EXPECT_EQ(res, expect);
}

} // namespace npu::tile_fwk
