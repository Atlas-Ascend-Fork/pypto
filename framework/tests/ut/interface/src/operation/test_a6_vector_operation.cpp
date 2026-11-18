/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation_common.h"

using namespace npu::tile_fwk;

class A6VectorOperationTest : public testing::Test {
protected:
    void SetUp() override
    {
        auto& soc = Platform::Instance().GetSoc();
        originalArch_ = soc.GetNPUArch();
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        soc.SetNPUArch(NPUArch::DAV_9201);
    }

    void TearDown() override { Platform::Instance().GetSoc().SetNPUArch(originalArch_); }

private:
    NPUArch originalArch_ = NPUArch::DAV_UNKNOWN;
};

TEST_F(A6VectorOperationTest, DtypesMatchA5)
{
    auto& soc = Platform::Instance().GetSoc();
    auto& configManager = ConfigManager::Instance();
    size_t supportedOps = 0;
    for (int code = 0; code < static_cast<int>(Opcode::OP_UNKNOWN); ++code) {
        auto opcode = static_cast<Opcode>(code);
        soc.SetNPUArch(NPUArch::DAV_3510);
        const auto& a5 = configManager.GetOpSupportedInputDtypes(opcode);
        soc.SetNPUArch(NPUArch::DAV_9201);
        const auto& a6 = configManager.GetOpSupportedInputDtypes(opcode);
        EXPECT_EQ(a6, a5) << "opcode=" << code;
        supportedOps += !a5.empty();
    }
    EXPECT_GT(supportedOps, 0U);
}

TEST_F(A6VectorOperationTest, CastClipAndRange)
{
    TileShape::Current().SetVecTile(8, 64);
    Tensor input(DT_INT64, {8, 64});
    Tensor fp32Input(DT_FP32, {8, 64});
    FUNCTION("A6CastClipAndRange", {input, fp32Input})
    {
        EXPECT_EQ(Add(input, input).GetDataType(), DT_INT64);
        EXPECT_EQ(Clip(input, Element(DT_INT64, 0), Element(DT_INT64, 10)).GetDataType(), DT_INT64);
        EXPECT_EQ(Clip(input, input, input).GetDataType(), DT_INT64);
        EXPECT_EQ(Cast(fp32Input, DT_FP8E4M3).GetDataType(), DT_FP8E4M3);
        EXPECT_EQ(Range(Element(DT_INT64, 0), Element(DT_INT64, 64), Element(DT_INT64, 1)).GetDataType(), DT_INT64);
        EXPECT_EQ(Add(fp32Input, fp32Input).GetDataType(), DT_FP32);
    }
}

TEST_F(A6VectorOperationTest, QuantMX)
{
    TileShape::Current().SetVecTile(8, 128);
    Tensor fp32Input(DT_FP32, {8, 128});
    Tensor fp16Input(DT_FP16, {8, 128});
    FUNCTION("A6QuantMX", {fp32Input, fp16Input})
    {
        auto fp8 = QuantMX(fp32Input);
        EXPECT_EQ(std::get<0>(fp8).GetDataType(), DT_FP8E4M3);
        EXPECT_EQ(std::get<1>(fp8).GetDataType(), DT_FP8E8M0);
        auto fp4 = QuantMX(fp16Input, DT_FP4_E2M1, DequantScaleRoundingMode::ROUND_UP, -1, true);
        EXPECT_EQ(std::get<0>(fp4).GetDataType(), DT_FP4_E2M1);
        EXPECT_EQ(std::get<1>(fp4).GetDataType(), DT_FP8E8M0);
        EXPECT_THROW(QuantMX(fp32Input, DT_FP4_E2M1), std::exception);
    }
}

TEST_F(A6VectorOperationTest, QuantMXAxisSupportByArch)
{
    config::SetHostOption(COMPILE_STAGE, CS_TILE_GRAPH);
    for (auto arch : {NPUArch::DAV_3510, NPUArch::DAV_9201}) {
        for (const auto& shape : std::vector<Shape>{{64, 128}, {2, 64, 128}, {2, 2, 64, 128}}) {
            Program::GetInstance().Reset();
            Platform::Instance().GetSoc().SetNPUArch(arch);
            TileShape::Current().SetVecTile(shape);
            Tensor input(DT_FP16, shape);
            const int64_t rank = static_cast<int64_t>(shape.size());
            FUNCTION("QuantMXAxisSupportByArch", {input})
            {
                for (auto dtype : {DT_FP8E4M3, DT_FP4_E2M1}) {
                    for (bool performanceMode : {false, true}) {
                        for (auto axis : {int64_t{-2}, rank - 2, int64_t{-1}, rank - 1}) {
                            SCOPED_TRACE(NPUArchToString(arch) + " rank=" + std::to_string(rank) +
                                         " axis=" + std::to_string(axis) + " dtype=" + DataType2String(dtype) +
                                         " performanceMode=" + std::to_string(performanceMode));
                            EXPECT_NO_THROW(
                                QuantMX(input, dtype, DequantScaleRoundingMode::ROUND_DOWN, axis, performanceMode));
                        }
                    }
                }
            }
        }
    }
}

TEST_F(A6VectorOperationTest, Fp8GatherAndPermute)
{
    TileShape::Current().SetVecTile(8, 64);
    Tensor input(DT_FP8E4M3, {8, 64});
    Tensor indices(DT_INT32, {4});
    FUNCTION("A6Fp8GatherAndPermute", {input, indices})
    {
        EXPECT_EQ(Gather(input, indices, 0).GetDataType(), DT_FP8E4M3);
        EXPECT_EQ(Permute(input, {1, 0}).GetDataType(), DT_FP8E4M3);
    }
}

TEST_F(A6VectorOperationTest, InterleaveAndUniform)
{
    TileShape::Current().SetVecTile(8, 64);
    Tensor input(DT_FP16, {8, 64});
    FUNCTION("A6Interleave", {input})
    {
        auto interleaved = Interleave(input, input);
        auto deinterleaved = DeInterleave(std::get<0>(interleaved), std::get<1>(interleaved));
        EXPECT_EQ(std::get<0>(deinterleaved).GetShape(), input.GetShape());
    }
    TileShape::Current().SetVecTile(64);
    FUNCTION("A6Uniform")
    {
        auto output = Uniform(Element(DT_UINT64, 1), SymbolicScalar(0), Element(DT_UINT64, 0), {64},
                              Element(DT_UINT16, 10), DT_FP32);
        EXPECT_EQ(output.GetDataType(), DT_FP32);
    }
}

TEST_F(A6VectorOperationTest, GcdSimtSelectionByArch)
{
    for (auto arch : {NPUArch::DAV_2201, NPUArch::DAV_3510, NPUArch::DAV_9201}) {
        for (auto dtype : {DT_INT8, DT_UINT8, DT_INT16, DT_INT32}) {
            SCOPED_TRACE(NPUArchToString(arch) + " " + DataType2String(dtype));
            Platform::Instance().GetSoc().SetNPUArch(arch);
            Program::GetInstance().Reset();
            TileShape::Current().SetVecTile(4, 32);
            Tensor input(dtype, {4, 32});
            Tensor output, scalarOutput;
            Function* function = nullptr;
            FUNCTION("GcdSimtSelection", {input})
            {
                function = Program::GetInstance().GetCurrentFunction();
                output = Gcd(input, input);
                scalarOutput = Gcd(input, Element(dtype, 12));
            }
            ASSERT_NE(function, nullptr);
            size_t gcdOps = 0;
            for (const auto& op : function->Operations()) {
                if (op.GetOpcode() != Opcode::OP_GCD && op.GetOpcode() != Opcode::OP_GCDS) {
                    continue;
                }
                ++gcdOps;
                bool simt = op.HasAttr(OP_ATTR_PREFIX + "requires_simt") &&
                            op.GetBoolAttribute(OP_ATTR_PREFIX + "requires_simt");
                EXPECT_EQ(simt, arch != NPUArch::DAV_2201);
                ASSERT_EQ(op.GetOOperands().size(), arch != NPUArch::DAV_2201 ? 1U : 2U);
                if (arch == NPUArch::DAV_2201) {
                    EXPECT_EQ(op.GetOOperands()[1]->Datatype(), DT_UINT8);
                    EXPECT_GT(op.GetOOperands()[1]->GetShape().back(), 0);
                }
            }
            EXPECT_EQ(gcdOps, 2U);
        }
    }
}

TEST_F(A6VectorOperationTest, OnlineSoftmaxAndRadixSelect)
{
    TileShape::Current().SetVecTile(8, 64);
    Tensor input(DT_FP32, {8, 64});
    FUNCTION("A6SoftmaxAndRadixSelect", {input})
    {
        auto softmax = OnlineSoftmax(input, 1.0f);
        EXPECT_EQ(std::get<0>(softmax).GetDataType(), DT_BF16);
        auto topk = TopK(input, 8, -1, true, TopKAlgo::RADIX_SELECT);
        EXPECT_EQ(std::get<0>(topk).GetShape(), std::vector<int64_t>({8, 8}));
        EXPECT_EQ(std::get<1>(topk).GetDataType(), DT_INT32);
    }
}

TEST_F(A6VectorOperationTest, IndexOpsSimtByArch)
{
    for (auto arch : {NPUArch::DAV_2201, NPUArch::DAV_3510, NPUArch::DAV_9201}) {
        SCOPED_TRACE(NPUArchToString(arch));
        Platform::Instance().GetSoc().SetNPUArch(arch);
        Program::GetInstance().Reset();
        TileShape::Current().SetVecTile(4, 32);
        Tensor self(DT_FP32, {4, 64});
        Tensor src(DT_FP32, {4, 32});
        Tensor indices(DT_INT32, {32});
        Tensor values(DT_FP32, {32});
        Function* function = nullptr;
        FUNCTION("A6IndexOps", {self, src, indices, values})
        {
            function = Program::GetInstance().GetCurrentFunction();
            IndexAdd_(self, src, indices, 1, Element(DT_FP32, 1.0f));
            IndexPut_(self, {indices, indices}, values, true);
        }
        ASSERT_NE(function, nullptr);
        size_t indexAddOps = 0;
        size_t indexPutOps = 0;
        for (const auto& op : function->Operations()) {
            if (op.GetOpcode() == Opcode::OP_INDEX_ADD || op.GetOpcode() == Opcode::OP_INDEX_PUT) {
                EXPECT_EQ(op.HasAttr(OP_ATTR_PREFIX + "requires_simt") &&
                              op.GetBoolAttribute(OP_ATTR_PREFIX + "requires_simt"),
                          arch != NPUArch::DAV_2201);
                if (op.GetOpcode() == Opcode::OP_INDEX_ADD) {
                    ++indexAddOps;
                    ASSERT_EQ(op.GetOOperands().size(), 2U);
                    EXPECT_EQ(op.GetOOperands()[1]->GetShape().front(), arch != NPUArch::DAV_2201 ? 2 : 1);
                } else {
                    ++indexPutOps;
                    EXPECT_EQ(op.GetOOperands().size(), arch != NPUArch::DAV_2201 ? 2U : 1U);
                }
            }
        }
        EXPECT_GT(indexAddOps, 0U);
        EXPECT_GT(indexPutOps, 0U);
    }
}

TEST_F(A6VectorOperationTest, ScatterInplaceSelectionByArch)
{
    config::SetHostOption(COMPILE_STAGE, CS_TILE_GRAPH);
    for (auto arch : {NPUArch::DAV_2201, NPUArch::DAV_3510, NPUArch::DAV_9201}) {
        for (bool scalar : {false, true}) {
            for (auto reduce : {ScatterMode::NONE, ScatterMode::ADD}) {
                SCOPED_TRACE(NPUArchToString(arch) + " scalar=" + std::to_string(scalar) +
                             " reduce=" + std::to_string(static_cast<int>(reduce)));
                Platform::Instance().GetSoc().SetNPUArch(arch);
                Program::GetInstance().Reset();
                TileShape::Current().SetVecTile(4, 32);
                Tensor self(DT_FP32, {4, 32});
                Tensor indices(DT_INT32, {4, 32});
                Tensor src(DT_FP32, {4, 32});
                Function* function = nullptr;
                FUNCTION("ScatterInplaceSelectionByArch", {self, indices, src})
                {
                    function = Program::GetInstance().GetCurrentFunction();
                    if (scalar) {
                        Scatter_(self, indices, Element(DT_FP32, 2), 1, reduce);
                    } else {
                        Scatter_(self, indices, src, 1, reduce);
                    }
                }
                ASSERT_NE(function, nullptr);
                bool inplace = arch != NPUArch::DAV_2201 && reduce == ScatterMode::NONE;
                auto expectedOpcode = scalar ?
                                          (inplace ? Opcode::OP_SCATTER_ELEMENT_INPLACE : Opcode::OP_SCATTER_ELEMENT) :
                                          (inplace ? Opcode::OP_SCATTER_INPLACE : Opcode::OP_SCATTER);
                size_t scatterOps = 0;
                for (const auto& op : function->Operations()) {
                    if (op.GetOpcode() != Opcode::OP_SCATTER && op.GetOpcode() != Opcode::OP_SCATTER_ELEMENT &&
                        op.GetOpcode() != Opcode::OP_SCATTER_INPLACE &&
                        op.GetOpcode() != Opcode::OP_SCATTER_ELEMENT_INPLACE) {
                        continue;
                    }
                    ++scatterOps;
                    EXPECT_EQ(op.GetOpcode(), expectedOpcode);
                    EXPECT_EQ(op.HasAttr(OpAttributeKey::dontTouch) && op.GetBoolAttribute(OpAttributeKey::dontTouch),
                              inplace);
                }
                EXPECT_EQ(scatterOps, 1U);
            }
        }
    }
}

TEST_F(A6VectorOperationTest, RemainderComputeTypeByArch)
{
    for (auto arch : {NPUArch::DAV_2201, NPUArch::DAV_3510, NPUArch::DAV_9201}) {
        for (auto dtype : {DT_INT16, DT_INT32, DT_FP16, DT_FP32, DT_BF16}) {
            SCOPED_TRACE(NPUArchToString(arch) + " " + DataType2String(dtype));
            Program::GetInstance().Reset();
            Platform::Instance().GetSoc().SetNPUArch(arch);
            TileShape::Current().SetVecTile(4, 32);
            Tensor input(dtype, {4, 32});
            Tensor tensorResult, scalarResult, reverseResult;
            Function* function = nullptr;
            FUNCTION("RemainderComputeTypeByArch", {input})
            {
                function = Program::GetInstance().GetCurrentFunction();
                tensorResult = Remainder(input, input);
                scalarResult = Remainder(input, Element(dtype, 3));
                reverseResult = Remainder(Element(dtype, 3), input);
            }
            EXPECT_EQ(tensorResult.GetDataType(), dtype);
            EXPECT_EQ(scalarResult.GetDataType(), dtype);
            EXPECT_EQ(reverseResult.GetDataType(), dtype);
            auto expectedType = dtype;
            // The existing AutoCast pass also promotes bf16 Remainder to fp32.
            if (dtype == DT_FP16 || dtype == DT_BF16 || (arch == NPUArch::DAV_2201 && dtype == DT_INT16)) {
                expectedType = DT_FP32;
            }
            size_t remainderOps = 0;
            for (const auto& op : function->Operations()) {
                if (op.GetOpcode() == Opcode::OP_REM || op.GetOpcode() == Opcode::OP_REMS ||
                    op.GetOpcode() == Opcode::OP_REMRS) {
                    ++remainderOps;
                    EXPECT_EQ(op.GetIOperands().front()->Datatype(), expectedType);
                }
            }
            EXPECT_EQ(remainderOps, 3U);
        }
    }
}

TEST_F(A6VectorOperationTest, RemainderUnsupportedArchitectures)
{
    for (auto arch :
         {NPUArch::DAV_1001, NPUArch::DAV_3003, NPUArch::DAV_3113, NPUArch::DAV_5101, NPUArch::DAV_UNKNOWN}) {
        SCOPED_TRACE(NPUArchToString(arch));
        Program::GetInstance().Reset();
        Platform::Instance().GetSoc().SetNPUArch(arch);
        TileShape::Current().SetVecTile(4, 32);
        Tensor input(DT_FP32, {4, 32});
        FUNCTION("RemainderUnsupportedArchitectures", {input})
        {
            EXPECT_THROW(Remainder(input, input), std::exception);
            EXPECT_THROW(Remainder(input, Element(DT_FP32, 3.0f)), std::exception);
            EXPECT_THROW(Remainder(Element(DT_FP32, 3.0f), input), std::exception);
        }
    }
}

TEST_F(A6VectorOperationTest, DtypeValidationByArch)
{
    for (auto arch : {NPUArch::DAV_2201, NPUArch::DAV_3510, NPUArch::DAV_9201}) {
        SCOPED_TRACE(NPUArchToString(arch));
        Program::GetInstance().Reset();
        Platform::Instance().GetSoc().SetNPUArch(arch);
        TileShape::Current().SetVecTile(8, 64);
        Tensor fp32Input(DT_FP32, {8, 64});
        Tensor int64Input(DT_INT64, {8, 64});
        Tensor fp8Input(DT_FP8E4M3, {8, 64});
        Tensor indices(DT_INT32, {4});
        FUNCTION("DtypeValidationByArch", {fp32Input, int64Input, fp8Input, indices})
        {
            EXPECT_EQ(Cast(fp32Input, DT_FP16).GetDataType(), DT_FP16);
            if (arch == NPUArch::DAV_2201) {
                EXPECT_THROW(Cast(fp32Input, DT_FP8E4M3), std::exception);
                EXPECT_THROW(Clip(int64Input, int64Input, int64Input), std::exception);
                EXPECT_THROW(Remainder(int64Input, int64Input), std::exception);
                EXPECT_THROW(Gather(fp8Input, indices, 0), std::exception);
                EXPECT_THROW(Permute(fp8Input, {1, 0}), std::exception);
            } else if (arch == NPUArch::DAV_9201) {
                EXPECT_EQ(Cast(fp32Input, DT_FP8E4M3).GetDataType(), DT_FP8E4M3);
                EXPECT_EQ(Clip(int64Input, int64Input, int64Input).GetDataType(), DT_INT64);
                EXPECT_EQ(Remainder(int64Input, int64Input).GetDataType(), DT_INT64);
                EXPECT_EQ(Gather(fp8Input, indices, 0).GetDataType(), DT_FP8E4M3);
                EXPECT_EQ(Permute(fp8Input, {1, 0}).GetDataType(), DT_FP8E4M3);
            } else {
                EXPECT_EQ(Cast(fp32Input, DT_FP8E4M3).GetDataType(), DT_FP8E4M3);
                EXPECT_EQ(Clip(int64Input, int64Input, int64Input).GetDataType(), DT_INT64);
                EXPECT_EQ(Remainder(int64Input, int64Input).GetDataType(), DT_INT64);
                EXPECT_EQ(Gather(fp8Input, indices, 0).GetDataType(), DT_FP8E4M3);
                EXPECT_EQ(Permute(fp8Input, {1, 0}).GetDataType(), DT_FP8E4M3);
            }
        }
    }
}

TEST_F(A6VectorOperationTest, ExplicitArchitectureDtypeValidation)
{
    for (auto arch : {NPUArch::DAV_2201, NPUArch::DAV_3510, NPUArch::DAV_9201, NPUArch::DAV_3003, NPUArch::DAV_3113,
                      NPUArch::DAV_5101, NPUArch::DAV_1001, NPUArch::DAV_UNKNOWN, static_cast<NPUArch>(9999)}) {
        SCOPED_TRACE(NPUArchToString(arch));
        Program::GetInstance().Reset();
        Platform::Instance().GetSoc().SetNPUArch(arch);
        TileShape::Current().SetVecTile(4, 32);
        Tensor input(DT_FP32, {4, 32});
        FUNCTION("ExplicitArchitectureDtypeValidation", {input})
        {
            if (arch == NPUArch::DAV_2201 || arch == NPUArch::DAV_3510 || arch == NPUArch::DAV_9201) {
                EXPECT_NO_THROW(Cast(input, DT_FP16));
                EXPECT_NO_THROW(Clip(input, input, input));
                EXPECT_NO_THROW(Permute(input, {1, 0}));
            } else {
                if (arch == NPUArch::DAV_3003 || arch == NPUArch::DAV_3113 || arch == NPUArch::DAV_5101) {
                    EXPECT_NO_THROW(Cast(input, DT_FP16));
                    EXPECT_THROW(Cast(input, DT_BF16), std::exception);
                } else {
                    EXPECT_NO_THROW(Cast(input, DT_FP16));
                }
            }
        }
    }
}

TEST_F(A6VectorOperationTest, RangeInt64ByArch)
{
    for (auto arch : {NPUArch::DAV_2201, NPUArch::DAV_3510, NPUArch::DAV_9201}) {
        SCOPED_TRACE(NPUArchToString(arch));
        Program::GetInstance().Reset();
        Platform::Instance().GetSoc().SetNPUArch(arch);
        TileShape::Current().SetVecTile(64);
        FUNCTION("RangeInt64ByArch")
        {
            auto result = Range(Element(DT_INT64, 0), Element(DT_INT64, 64), Element(DT_INT64, 1));
            EXPECT_EQ(result.GetDataType(), arch == NPUArch::DAV_2201 ? DT_INT32 : DT_INT64);
            int64_t start = int64_t{1} << 32;
            if (arch == NPUArch::DAV_2201) {
                EXPECT_THROW(Range(Element(DT_INT64, start), Element(DT_INT64, start + 64), Element(DT_INT64, 1)),
                             std::exception);
            } else {
                auto wide = Range(Element(DT_INT64, start), Element(DT_INT64, start + 64), Element(DT_INT64, 1));
                EXPECT_EQ(wide.GetDataType(), DT_INT64);
                EXPECT_EQ(wide.GetShape(), std::vector<int64_t>({64}));
            }
        }
    }
}

TEST_F(A6VectorOperationTest, IsaPendingOperationsFollowDeclaredCapabilities)
{
    config::SetHostOption(COMPILE_STAGE, CS_TILE_GRAPH);
    for (auto arch : {NPUArch::DAV_3510, NPUArch::DAV_9201}) {
        Program::GetInstance().Reset();
        Platform::Instance().GetSoc().SetNPUArch(arch);
        TileShape::Current().SetVecTile(4, 128);
        Tensor bf16(DT_BF16, {4, 128}), wide(DT_INT64, {4, 128}), small(DT_INT8, {4, 128});
        Tensor fp32(DT_FP32, {4, 128}), condition(DT_BOOL, {4, 128});
        FUNCTION("IsaPendingOperationsFollowDeclaredCapabilities", {bf16, wide, small, fp32, condition})
        {
            auto check = [](const std::function<void()>& call) { EXPECT_NO_THROW(call()); };
            EXPECT_NO_THROW(Add(bf16, bf16));
            check([&]() { Add(wide, Element(DT_INT64, 1)); });
            check([&]() { Compare(wide, wide, OpType::EQ, OutType::BOOL); });
            check([&]() { Where(condition, wide, wide); });
            check([&]() { FloorDiv(small, small); });
            EXPECT_NO_THROW(Tan(bf16));
            EXPECT_NO_THROW(Ceil(bf16));
            EXPECT_NO_THROW(Floor(bf16));
            EXPECT_NO_THROW(Trunc(bf16));
            EXPECT_NO_THROW(TriL(bf16, SymbolicScalar(0)));
            EXPECT_NO_THROW(Remainder(bf16, bf16));
            EXPECT_NO_THROW(Remainder(bf16, Element(DT_BF16, 3)));
            EXPECT_NO_THROW(Remainder(Element(DT_BF16, 3), bf16));
            EXPECT_NO_THROW(Cast(fp32, DT_FP8E4M3));
            EXPECT_NO_THROW(Cast(bf16, DT_FP32));
            // These paths compiled against the baseline ISA; do not ban a dtype globally.
            EXPECT_NO_THROW(IsFinite(bf16));
            EXPECT_NO_THROW(Signbit(bf16));
            EXPECT_NO_THROW(Compare(fp32, fp32, OpType::EQ, OutType::BOOL));
            EXPECT_NO_THROW(Cast(fp32, DT_BF16));
            EXPECT_NO_THROW(Add(fp32, fp32));
        }
    }
}
