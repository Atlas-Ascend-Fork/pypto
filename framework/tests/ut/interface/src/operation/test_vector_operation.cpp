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
 * \file test_vector_operation.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/operation/operation.h"
#include "interface/operation/operation_common.h"
#include "interface/tensor/irbuilder.h"
#include "passes/block_graph_pass/schedule_ooo/pre_schedule/add_alloc.h"
#include "passes/tensor_graph_pass/expand_function.h"
#include "passes/tile_graph_pass/data_path/convert_op_inserter.h"
#include "passes/tile_graph_pass/data_path/memory_path_utils.h"
#include <tuple>
#include <limits>
#include <cmath>

using namespace npu::tile_fwk;

class VectorOperationTest : public testing::Test {
public:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    }

    void TearDown() override {}
};

TEST_F(VectorOperationTest, test_CumSum_dim2_1)
{
    int axis = 1;
    TileShape::Current().SetVecTile(9, 9);
    Tensor input(DT_FP32, {13, 8}, "input");
    Tensor result;
    FUNCTION("TestCumSum") { result = CumSum(input, axis); }
}

TEST_F(VectorOperationTest, test_CumSum_dim2_0)
{
    int axis = 0;
    TileShape::Current().SetVecTile(4, 3);
    Tensor input(DT_INT16, {11, 7}, "input");
    Tensor result;
    FUNCTION("TestCumSum") { result = CumSum(input, axis); }
}

TEST_F(VectorOperationTest, test_CumSum_dim1)
{
    int axis = 0;
    TileShape::Current().SetVecTile(5);
    Tensor input(DT_INT32, {13}, "input");
    Tensor result;
    FUNCTION("TestCumSum") { result = CumSum(input, axis); }
}

TEST_F(VectorOperationTest, test_CumSum_dim3)
{
    int axis = 0;
    TileShape::Current().SetVecTile(4, 5, 3);
    Tensor input(DT_FP32, {8, 8, 8}, "input");
    Tensor result;
    FUNCTION("TestCumSum") { result = CumSum(input, axis); }
}

TEST_F(VectorOperationTest, test_CumSum_dim4)
{
    int axis = 0;
    TileShape::Current().SetVecTile(4, 5, 5, 3);
    Tensor input(DT_FP32, {7, 7, 7, 7}, "input");
    Tensor result;
    FUNCTION("TestCumSum") { result = CumSum(input, axis); }
}

TEST_F(VectorOperationTest, test_IsFinite_fp32)
{
    TileShape::Current().SetVecTile(4, 32);
    Tensor input(DT_FP32, {11, 32}, "input");
    Tensor result;
    FUNCTION("TestIsFinite") { result = IsFinite(input); }
}

TEST_F(VectorOperationTest, test_IsFinite_int32)
{
    TileShape::Current().SetVecTile(4, 32);
    Tensor input(DT_INT32, {11, 32}, "input");
    Tensor result;
    FUNCTION("TestIsFinite") { result = IsFinite(input); }
}

TEST_F(VectorOperationTest, test_CeilDivs_int32)
{
    TileShape::Current().SetVecTile(4, 32);
    Tensor self(DT_INT32, {11, 32}, "self");
    Element other(DT_INT32, 2);
    Tensor result;
    FUNCTION("TestCeilDiv") { result = CeilDiv(self, other); }
}

TEST_F(VectorOperationTest, test_CeilDiv_int32)
{
    TileShape::Current().SetVecTile(4, 32);
    Tensor self(DT_INT32, {11, 32}, "self");
    Tensor other(DT_INT32, {11, 32}, "other");
    Tensor result;
    FUNCTION("TestCeilDiv") { result = CeilDiv(self, other); }
}

TEST_F(VectorOperationTest, test_Compare_BOOL)
{
    TileShape::Current().SetVecTile({4, 4});
    Tensor operand1(DT_FP32, {8, 8}, "operand1");
    Tensor operand2(DT_FP32, {8, 8}, "operand2");
    std::vector<int64_t> dstShape = {8, 8};
    Tensor result;
    FUNCTION("TestCompare") { result = Compare(operand1, operand2, OpType::EQ, OutType::BOOL); }
}

TEST_F(VectorOperationTest, test_Sort)
{
    TileShape::Current().SetVecTile({6, 64});
    Tensor operand1(DT_FP32, {6, 64}, "operand1");
    Tensor sort32Result;
    Tensor mrgsortResult;
    FUNCTION("TestSort")
    {
        sort32Result = Sort32(operand1, 0);
        mrgsortResult = MrgSort(sort32Result, 32);
    }
}

TEST_F(VectorOperationTest, test_Compare_BIT)
{
    TileShape::Current().SetVecTile({8, 8});
    Tensor operand1(DT_FP16, {16, 16}, "operand1");
    Tensor operand2(DT_FP16, {16, 16}, "operand2");
    std::vector<int64_t> dstShape = {16, 2};
    Tensor result;
    FUNCTION("TestCompare") { result = Compare(operand1, operand2, OpType::EQ, OutType::BIT); }
}

TEST_F(VectorOperationTest, test_Cmps_BOOL)
{
    TileShape::Current().SetVecTile({4, 4});
    Tensor operand1(DT_FP32, {8, 8}, "operand1");
    float scalar = 10.0;
    Element operand2(DT_FP32, scalar);
    std::vector<int64_t> dstShape = {8, 8};
    Tensor result;
    FUNCTION("TestCompare") { result = Compare(operand1, operand2, OpType::EQ, OutType::BOOL); }
}

TEST_F(VectorOperationTest, test_Cmps_BIT)
{
    TileShape::Current().SetVecTile({8, 8});
    Tensor operand1(DT_FP16, {16, 16}, "operand1");
    float scalar = 10.0;
    Element operand2(DT_FP16, scalar);
    std::vector<int64_t> dstShape = {16, 2};
    Tensor result;
    FUNCTION("TestCompare") { result = Compare(operand1, operand2, OpType::EQ, OutType::BIT); }
}

TEST_F(VectorOperationTest, test_Hypot_FP32)
{
    TileShape::Current().SetVecTile({4, 4});
    Tensor operand1(DT_FP32, {8, 8}, "operand1");
    Tensor operand2(DT_FP32, {8, 8}, "operand2");
    std::vector<int64_t> dstShape = {8, 8};
    Tensor result;
    FUNCTION("TestHypot") { result = Hypot(operand1, operand2); }
}

TEST_F(VectorOperationTest, test_Hypot_FP16)
{
    TileShape::Current().SetVecTile({4, 4});
    Tensor operand1(DT_FP16, {8, 8}, "operand1");
    Tensor operand2(DT_FP16, {8, 8}, "operand2");
    std::vector<int64_t> dstShape = {8, 8};
    Tensor result;
    FUNCTION("TestHypot") { result = Hypot(operand1, operand2); }
}

TEST_F(VectorOperationTest, test_PReLU_FP32)
{
    TileShape::Current().SetVecTile({4, 4});
    Tensor operand1(DT_FP32, {8, 8}, "operand1");
    Tensor weight(DT_FP32, {8}, "weight");
    std::vector<int64_t> dstShape = {8, 8};
    Tensor result;
    FUNCTION("TestPReLU") { result = PReLU(operand1, weight); }
}

TEST_F(VectorOperationTest, test_PReLU_FP16)
{
    TileShape::Current().SetVecTile({4, 4});
    Tensor operand1(DT_FP16, {8, 8}, "operand1");
    Tensor weight(DT_FP16, {8}, "weight");
    std::vector<int64_t> dstShape = {8, 8};
    Tensor result;
    FUNCTION("TestPReLU") { result = PReLU(operand1, weight); }
}

TEST_F(VectorOperationTest, Test_IndexAddUB_BF16)
{
    float scalar = 1.2f;
    int axis = 0;

    TileShape::Current().SetVecTile({8, 16});
    Tensor self(DT_BF16, {10, 16}, "operand0");
    Tensor src(DT_BF16, {8, 16}, "operand1");
    Tensor index(DT_INT32, {8}, "operand2");
    Element alpha(DT_BF16, scalar);
    Tensor result;
    FUNCTION("TestIndxAdd") { result = IndexAddUB(self, src, index, axis, alpha); }
}

TEST_F(VectorOperationTest, Test_IndexAddUB_INT16)
{
    int scalar = 2;
    int axis = 1;

    TileShape::Current().SetVecTile({8, 16});
    Tensor self(DT_INT16, {10, 5}, "operand0");
    Tensor src(DT_INT16, {10, 2}, "operand1");
    Tensor index(DT_INT64, {2}, "operand2");
    Element alpha(DT_INT16, scalar);
    Tensor result;
    FUNCTION("TestIndxAdd") { result = IndexAddUB(self, src, index, axis, alpha); }
}

TEST_F(VectorOperationTest, Test_IndexAdd_FP32)
{
    float scalar = 1.2f;
    int axis = 0;

    TileShape::Current().SetVecTile({8, 8, 16});
    Tensor self(DT_FP32, {10, 10, 16}, "operand0");
    Tensor src(DT_FP32, {15, 10, 16}, "operand1");
    Tensor index(DT_INT32, {15}, "operand2");
    Element alpha(DT_FP32, scalar);
    FUNCTION("TestIndxAdd") { IndexAdd_(self, src, index, axis, alpha); }
}

TEST_F(VectorOperationTest, Test_IndexAddUB_FP16)
{
    float scalar = 1.0f;
    int axis = 0;

    TileShape::Current().SetVecTile({8, 8, 8, 16});
    Tensor self(DT_FP16, {10, 10, 10, 16}, "operand0");
    Tensor src(DT_FP16, {8, 10, 10, 16}, "operand1");
    Tensor index(DT_INT64, {8}, "operand2");
    Element alpha(DT_FP16, scalar);
    Tensor result;
    FUNCTION("TestIndxAdd") { result = IndexAddUB(self, src, index, axis, alpha); }
}

void TestPow(DataType selfType, DataType otherType, DataType resultType)
{
    std::vector<int64_t> shape = {32, 32};
    PROGRAM("POW")
    {
        TileShape::Current().SetVecTile({32, 32});
        Tensor input_a(selfType, shape, "input_a");
        Tensor input_b(otherType, shape, "input_b");
        auto output = Tensor(resultType, shape, "res");
        FUNCTION("POW_FUC") { output = Pow(input_a, input_b); }
    }
}

TEST_F(VectorOperationTest, Test_Pow)
{
    TestPow(DataType::DT_FP32, DataType::DT_FP32, DataType::DT_FP32);
    TestPow(DataType::DT_FP16, DataType::DT_FP16, DataType::DT_FP16);
    TestPow(DataType::DT_BF16, DataType::DT_FP16, DataType::DT_FP32);
    TestPow(DataType::DT_FP16, DataType::DT_BF16, DataType::DT_FP32);
    TestPow(DataType::DT_INT32, DataType::DT_FP32, DataType::DT_FP32);
    TestPow(DataType::DT_FP32, DataType::DT_INT32, DataType::DT_FP32);
    TestPow(DataType::DT_INT32, DataType::DT_INT32, DataType::DT_INT32);
}

TEST_F(VectorOperationTest, Test_Pow_FP32_Broadcast)
{
    PROGRAM("POW")
    {
        TileShape::Current().SetVecTile({32, 32});
        Tensor input_a(DataType::DT_FP32, {1, 32}, "input_a");
        Tensor input_b(DataType::DT_FP32, {32, 32}, "input_b");
        auto output = Tensor(DataType::DT_FP32, {32, 32}, "res");
        FUNCTION("POW_FUC") { output = Pow(input_a, input_b); }
    }
}

void TestPows(DataType dataType, double exponent)
{
    std::vector<int64_t> shape = {32, 32};
    PROGRAM("POWS")
    {
        TileShape::Current().SetVecTile({32, 32});
        Tensor input_a(dataType, shape, "input");
        auto output = Tensor(dataType, shape, "res");
        FUNCTION("POWS_FUC") { output = Pow(input_a, Element(dataType, exponent)); }
    }
}

TEST_F(VectorOperationTest, Test_Pows_0)
{
    constexpr double EXP0 = 0;
    TestPows(DataType::DT_FP32, EXP0);
    constexpr double EXP_1_5 = -1.5;
    TestPows(DataType::DT_FP32, EXP_1_5);
    constexpr double EXP1_5 = 1.5;
    TestPows(DataType::DT_FP16, EXP1_5);
    constexpr double EXP2 = 2;
    TestPows(DataType::DT_FP32, EXP2);
    constexpr double EXP3 = 3;
    TestPows(DataType::DT_FP32, EXP3);
}

TEST_F(VectorOperationTest, Test_Pows_1)
{
    constexpr double EXP1_5 = 1.5;
    std::vector<int64_t> shape = {32, 32};
    PROGRAM("POWS")
    {
        TileShape::Current().SetVecTile({32, 32});
        Tensor input_a(DataType::DT_INT32, shape, "input");
        auto output = Tensor(DataType::DT_FP32, shape, "res");
        FUNCTION("POWS_FUC") { output = Pow(input_a, Element(DataType::DT_FP32, EXP1_5)); }
    }
}

TEST_F(VectorOperationTest, Test_LogicalNot_BF16)
{
    PROGRAM("LogicalNot")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_BF16, shape, "A");
        auto output = Tensor(DT_BOOL, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("LogicalNot_BF16") { output = LogicalNot(input_a); }
    }
}

TEST_F(VectorOperationTest, Test_Expm1_FP16)
{
    PROGRAM("Expm1")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP16, shape, "operand1");
        auto output = Tensor(DT_FP16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Expm1_FP16") { output = Expm1(input_a); }
    }
}

TEST_F(VectorOperationTest, Test_Expm1_FP32)
{
    PROGRAM("Expm1")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP32, shape, "operand1");
        auto output = Tensor(DT_FP32, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Expm1_FP32") { output = Expm1(input_a); }
    }
}

TEST_F(VectorOperationTest, Test_Sinh_FP16)
{
    PROGRAM("Sinh")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP16, shape, "operand1");
        auto output = Tensor(DT_FP16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Sinh_FP16") { output = Sinh(input_a); }
    }
}

TEST_F(VectorOperationTest, Test_Sinh_FP32)
{
    PROGRAM("Sinh")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP32, shape, "operand1");
        auto output = Tensor(DT_FP32, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Sinh_FP32") { output = Sinh(input_a); }
    }
}

TEST_F(VectorOperationTest, Test_Cosh_FP16)
{
    PROGRAM("Cosh")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP16, shape, "operand1");
        auto output = Tensor(DT_FP16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Cosh_FP16") { output = Cosh(input_a); }
    }
}

TEST_F(VectorOperationTest, Test_Cosh_FP32)
{
    PROGRAM("Cosh")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP32, shape, "operand1");
        auto output = Tensor(DT_FP32, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Cosh_FP32") { output = Cosh(input_a); }
    }
}

TEST_F(VectorOperationTest, Test_ASinh_FP16)
{
    PROGRAM("ASinh")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP16, shape, "operand1");
        auto output = Tensor(DT_FP16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("ASinh_FP16") { output = ASinh(input_a); }
    }
}

TEST_F(VectorOperationTest, Test_ASinh_FP32)
{
    PROGRAM("ASinh")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP32, shape, "operand1");
        auto output = Tensor(DT_FP32, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("ASinh_FP32") { output = ASinh(input_a); }
    }
}

TEST_F(VectorOperationTest, Test_ACosh_FP16)
{
    PROGRAM("ACosh")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP16, shape, "operand1");
        auto output = Tensor(DT_FP16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("ACosh_FP16") { output = ACosh(input_a); }
    }
}

TEST_F(VectorOperationTest, Test_ACosh_FP32)
{
    PROGRAM("ACosh")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP32, shape, "operand1");
        auto output = Tensor(DT_FP32, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("ACosh_FP32") { output = ACosh(input_a); }
    }
}

TEST_F(VectorOperationTest, Test_Atanh_FP16)
{
    PROGRAM("Atanh")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP16, shape, "operand1");
        auto output = Tensor(DT_FP16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Atanh_FP16") { output = Atanh(input_a); }
    }
}

TEST_F(VectorOperationTest, Test_Atanh_FP32)
{
    PROGRAM("Atanh")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP32, shape, "operand1");
        auto output = Tensor(DT_FP32, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Atanh_FP32") { output = Atanh(input_a); }
    }
}

TEST_F(VectorOperationTest, Test_Atanh_BF16)
{
    PROGRAM("Atanh")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_BF16, shape, "operand1");
        auto output = Tensor(DT_BF16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Atanh_BF16") { output = Atanh(input_a); }
    }
}

TEST_F(VectorOperationTest, Test_Sign_FP16)
{
    PROGRAM("Sign")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP16, shape, "A");
        auto output = Tensor(DT_FP16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Sign_FP16") { output = Sign(input_a); }
    }
}

TEST_F(VectorOperationTest, Test_Sign_FP32)
{
    PROGRAM("Sign")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP32, shape, "A");
        auto output = Tensor(DT_FP32, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Sign_FP32") { output = Sign(input_a); }
    }
}

TEST_F(VectorOperationTest, Test_Sign_INT16)
{
    PROGRAM("Sign")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP16, shape, "A");
        auto output = Tensor(DT_INT16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Sign_INT16") { output = Sign(input_a); }
    }
}

TEST_F(VectorOperationTest, Test_Signbit_FP16)
{
    PROGRAM("Signbit")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP16, shape, "A");
        auto output = Tensor(DT_BOOL, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Signbit_FP16") { output = Signbit(input_a); }
    }
}

TEST_F(VectorOperationTest, Test_Signbit_FP32)
{
    PROGRAM("Signbit")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP32, shape, "A");
        auto output = Tensor(DT_BOOL, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Signbit_FP32") { output = Signbit(input_a); }
    }
}

TEST_F(VectorOperationTest, Test_Tan_FP16)
{
    PROGRAM("Tan")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP16, shape, "A");
        auto output = Tensor(DT_FP16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Tan_FP16") { output = Tan(input_a); }
    }
}

TEST_F(VectorOperationTest, Test_Tan_FP32)
{
    PROGRAM("Tan")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP32, shape, "A");
        auto output = Tensor(DT_FP32, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Tan_FP32") { output = Tan(input_a); }
    }
}

TEST_F(VectorOperationTest, Test_Tan_BF16)
{
    PROGRAM("Tan")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_BF16, shape, "A");
        auto output = Tensor(DT_BF16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Tan_BF16") { output = Tan(input_a); }
    }
}

TEST_F(VectorOperationTest, Test_Tanh_FP16)
{
    PROGRAM("Tanh")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP16, shape, "A");
        auto output = Tensor(DT_FP16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Tanh_FP16") { output = Tanh(input_a); }
    }
}

TEST_F(VectorOperationTest, Test_Tanh_FP32)
{
    PROGRAM("Tanh")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP32, shape, "A");
        auto output = Tensor(DT_FP32, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Tanh_FP32") { output = Tanh(input_a); }
    }
}

TEST_F(VectorOperationTest, Test_Log1p_FP16)
{
    PROGRAM("Log1p")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP16, shape, "A");
        auto output = Tensor(DT_FP16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Log1p_FP16") { output = Log1p(input_a); }
    }
}

TEST_F(VectorOperationTest, Test_Log1p_FP32)
{
    PROGRAM("Log1p")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP32, shape, "A");
        auto output = Tensor(DT_FP32, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Log1p_FP32") { output = Log1p(input_a); }
    }
}

TEST_F(VectorOperationTest, Test_Log1p_BF16)
{
    PROGRAM("Log1p")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_BF16, shape, "A");
        auto output = Tensor(DT_BF16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Log1p_BF16") { output = Log1p(input_a); }
    }
}

TEST_F(VectorOperationTest, Test_WhereTT_BF16)
{
    PROGRAM("Where")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_BF16, shape, "A");
        Tensor input_b(DT_BF16, shape, "B");
        Tensor input_c(DT_BOOL, shape, "C");
        auto output = Tensor(DT_BOOL, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Where_BF6") { output = Where(input_c, input_a, input_b); }
    }
}

TEST_F(VectorOperationTest, Test_WhereTS_BF16)
{
    PROGRAM("Where")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_BF16, shape, "A");
        float scalar = 10.0;
        Element operand2(DT_BF16, scalar);
        Tensor input_c(DT_BOOL, shape, "C");
        auto output = Tensor(DT_BOOL, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Where_BF6") { output = Where(input_c, input_a, operand2); }
    }
}

TEST_F(VectorOperationTest, Test_WhereSS_BF16)
{
    PROGRAM("Where")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        float scalar = 10.0;
        Element operand1(DT_BF16, scalar);
        Element operand2(DT_BF16, scalar);
        Tensor input_c(DT_BOOL, shape, "C");
        auto output = Tensor(DT_BOOL, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Where_BF6") { output = Where(input_c, operand1, operand2); }
    }
}

TEST_F(VectorOperationTest, Test_WhereST_BF16)
{
    PROGRAM("Where")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        float scalar = 10.0;
        Element operand1(DT_BF16, scalar);
        Tensor input_b(DT_BF16, shape, "B");
        Tensor input_c(DT_BOOL, shape, "C");
        auto output = Tensor(DT_BOOL, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Where_BF6") { output = Where(input_c, operand1, input_b); }
    }
}

template <DataType inputType, DataType outputType, bool IsANZ = false, bool IsBNZ = false, bool isTransB = false>
void TestNZFormatBatch(int bs, int m, int k, int n)
{
    std::vector<int64_t> batch_shape_a = {bs * m, k};
    auto nLen = isTransB ? bs * n : bs * k;
    auto kLen = isTransB ? k : n;
    std::vector<int64_t> batch_shape_b = {nLen, kLen};
    std::vector<int64_t> batch_shape_c = {bs * m, n};
    PROGRAM("BATCHMATMUL")
    {
        config::Reset();
        TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
        auto afmt = IsANZ ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
        auto bfmt = IsBNZ ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
        Tensor matA(inputType, batch_shape_a, "MatA", afmt);
        Tensor matB(inputType, batch_shape_b, "MatB", bfmt);
        Tensor matC(outputType, batch_shape_c, "MatC");
        std::vector<Tensor> matrixVec;
        config::SetBuildStatic(true);
        FUNCTION("BATCHMATMUL", {matA, matB, matC})
        {
            std::vector<std::pair<Tensor, std::vector<int64_t>>> assembleVec;
            for (size_t index = 0; index < (size_t)bs; ++index) {
                auto inputA = View(matA, {m, k}, {(int)index * m, 0});
                auto inputB = isTransB ? View(matB, {n, k}, {(int)index * n, 0}) :
                                         View(matB, {k, n}, {(int)index * k, 0});
                TileShape::Current().SetMatrixSize({m, k, n});
                auto outTensor = npu::tile_fwk::Matrix::Matmul(outputType, inputA, inputB, false, isTransB);
                std::vector<int64_t> pairSecond = {(int)index * m, 0};
                auto pair = std::make_pair(outTensor, pairSecond);
                assembleVec.emplace_back(pair);
            }
            matC = Assemble(assembleVec);
        }
    }
}

TEST_F(VectorOperationTest, test_Range_FP16)
{
    float startValue = (float)1.0;
    float endValue = (float)10.0;
    float stepValue = (float)1.1;
    int NUM_Eight = 8;
    Element start(DT_FP16, startValue);
    Element end(DT_FP16, endValue);
    Element step(DT_FP16, stepValue);
    TileShape::Current().SetVecTile(NUM_Eight);
    Tensor result;
    FUNCTION("TestRange") { result = Range(start, end, step); }
}

TEST_F(VectorOperationTest, test_Range_BF16)
{
    float startValue = (float)1.0;
    float endValue = (float)10.0;
    float stepValue = (float)1.1;
    int NUM_Eight = 8;
    Element start(DT_BF16, startValue);
    Element end(DT_BF16, endValue);
    Element step(DT_BF16, stepValue);
    TileShape::Current().SetVecTile(NUM_Eight);
    Tensor result;
    FUNCTION("TestRange") { result = Range(start, end, step); }
}

TEST_F(VectorOperationTest, test_Range_FP32)
{
    float startValue = (float)1.0;
    float endValue = (float)10.0;
    float stepValue = (float)1.1;
    int NUM_Eight = 8;
    Element start(DT_FP32, startValue);
    Element end(DT_FP32, endValue);
    Element step(DT_FP32, stepValue);
    TileShape::Current().SetVecTile(NUM_Eight);
    Tensor result;
    FUNCTION("TestRange") { result = Range(start, end, step); }
}

TEST_F(VectorOperationTest, test_Range_INT32)
{
    int startValue = 1;
    int endValue = 10;
    int stepValue = 3;
    int NUM_Eight = 8;
    Element start(DT_INT32, startValue);
    Element end(DT_INT32, endValue);
    Element step(DT_INT32, stepValue);
    TileShape::Current().SetVecTile(NUM_Eight);
    Tensor result;
    FUNCTION("TestRange") { result = Range(start, end, step); }
}

TEST_F(VectorOperationTest, Test_Uniform_UINT32)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    PROGRAM("Uniform")
    {
        std::vector<int64_t> shape = {128};
        TileShape::Current().SetVecTile({128});
        uint64_t key = 12345678901234;
        uint64_t counter0 = 0;
        uint64_t counter1 = 0;
        Tensor output(DT_FP32, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Uniform_UINT32")
        {
            output = Uniform(Element(DT_UINT64, key), SymbolicScalar(static_cast<int64_t>(counter0)),
                             Element(DT_UINT64, counter1), shape, Element(DT_UINT16, static_cast<uint16_t>(10)),
                             DT_FP32);
        }
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(VectorOperationTest, Test_Uniform_FP16)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    PROGRAM("Uniform")
    {
        std::vector<int64_t> shape = {128};
        TileShape::Current().SetVecTile({128});
        uint64_t key = 12345678901234;
        uint64_t counter0 = 0;
        uint64_t counter1 = 0;
        Tensor output(DT_FP16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Uniform_FP16")
        {
            output = Uniform(Element(DT_UINT64, key), SymbolicScalar(static_cast<int64_t>(counter0)),
                             Element(DT_UINT64, counter1), shape, Element(DT_UINT16, static_cast<uint16_t>(10)),
                             DT_FP16);
        }
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(VectorOperationTest, Test_Uniform_BF16)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    PROGRAM("Uniform")
    {
        std::vector<int64_t> shape = {128};
        TileShape::Current().SetVecTile({128});
        uint64_t key = 12345678901234;
        uint64_t counter0 = 0;
        uint64_t counter1 = 0;
        Tensor output(DT_BF16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Uniform_BF16")
        {
            output = Uniform(Element(DT_UINT64, key), SymbolicScalar(static_cast<int64_t>(counter0)),
                             Element(DT_UINT64, counter1), shape, Element(DT_UINT16, static_cast<uint16_t>(10)),
                             DT_BF16);
        }
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(VectorOperationTest, Test_Uniform_Rounds7)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    PROGRAM("Uniform")
    {
        std::vector<int64_t> shape = {128};
        TileShape::Current().SetVecTile({128});
        uint64_t key = 12345678901234;
        uint64_t counter0 = 0;
        uint64_t counter1 = 0;
        Tensor output(DT_FP32, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Uniform_Rounds7")
        {
            output = Uniform(Element(DT_UINT64, key), SymbolicScalar(static_cast<int64_t>(counter0)),
                             Element(DT_UINT64, counter1), shape, Element(DT_UINT16, static_cast<uint16_t>(7)),
                             DT_FP32);
        }
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(VectorOperationTest, Test_Uniform_FP16_Rounds7)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    PROGRAM("Uniform")
    {
        std::vector<int64_t> shape = {128};
        TileShape::Current().SetVecTile({128});
        uint64_t key = 12345678901234;
        uint64_t counter0 = 0;
        uint64_t counter1 = 0;
        Tensor output(DT_FP16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Uniform_FP16_Rounds7")
        {
            output = Uniform(Element(DT_UINT64, key), SymbolicScalar(static_cast<int64_t>(counter0)),
                             Element(DT_UINT64, counter1), shape, Element(DT_UINT16, static_cast<uint16_t>(7)),
                             DT_FP16);
        }
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(VectorOperationTest, Test_Uniform_BF16_Rounds7)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    PROGRAM("Uniform")
    {
        std::vector<int64_t> shape = {128};
        TileShape::Current().SetVecTile({128});
        uint64_t key = 12345678901234;
        uint64_t counter0 = 0;
        uint64_t counter1 = 0;
        Tensor output(DT_BF16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Uniform_BF16_Rounds7")
        {
            output = Uniform(Element(DT_UINT64, key), SymbolicScalar(static_cast<int64_t>(counter0)),
                             Element(DT_UINT64, counter1), shape, Element(DT_UINT16, static_cast<uint16_t>(7)),
                             DT_BF16);
        }
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(VectorOperationTest, Test_Uniform_LargeShape)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    PROGRAM("Uniform")
    {
        std::vector<int64_t> shape = {256};
        TileShape::Current().SetVecTile({128});
        uint64_t key = 9876543210;
        uint64_t counter0 = 100;
        uint64_t counter1 = 200;
        Tensor output(DT_FP32, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Uniform_LargeShape")
        {
            output = Uniform(Element(DT_UINT64, key), SymbolicScalar(static_cast<int64_t>(counter0)),
                             Element(DT_UINT64, counter1), shape, Element(DT_UINT16, static_cast<uint16_t>(10)),
                             DT_FP32);
        }
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(VectorOperationTest, Test_Exp2_FP16)
{
    PROGRAM("Exp2")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP16, shape, "operand1");
        auto output = Tensor(DT_FP16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Exp2_FP16") { output = Exp2(input_a); }
    }
}

TEST_F(VectorOperationTest, Test_Exp2_FP32)
{
    PROGRAM("Exp2")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP32, shape, "operand1");
        auto output = Tensor(DT_FP32, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Exp2_FP32") { output = Exp2(input_a); }
    }
}

TEST_F(VectorOperationTest, Test_Round_FP16)
{
    PROGRAM("Round")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP16, shape, "operand1");
        auto output = Tensor(DT_FP16, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Round_FP16") { output = Round(input_a, 1); }
    }
}

TEST_F(VectorOperationTest, Test_Round_FP32)
{
    PROGRAM("Round")
    {
        std::vector<int64_t> shape = {128, 32};
        TileShape::Current().SetVecTile({128, 32});
        Tensor input_a(DT_FP32, shape, "operand1");
        auto output = Tensor(DT_FP32, shape, "res");
        config::SetBuildStatic(true);
        FUNCTION("Round_FP32") { output = Round(input_a, 1); }
    }
}

TEST_F(VectorOperationTest, test_Rsqrt_FP16)
{
    constexpr int TILE_SHAPE = 32;
    constexpr int SHAPE = 128;
    TileShape::Current().SetVecTile(TILE_SHAPE, TILE_SHAPE);
    Tensor operand1(DT_FP16, {SHAPE, SHAPE}, "operand1");
    Tensor result;
    FUNCTION("TestRsqrt") { result = Rsqrt(operand1); }
}

TEST_F(VectorOperationTest, test_Rsqrt_FP32)
{
    constexpr int TILE_SHAPE = 32;
    constexpr int SHAPE = 128;
    TileShape::Current().SetVecTile(TILE_SHAPE, TILE_SHAPE);
    Tensor operand1(DT_FP32, {SHAPE, SHAPE}, "operand1");
    Tensor result;
    FUNCTION("TestRsqrt") { result = Rsqrt(operand1); }
}

TEST_F(VectorOperationTest, test_Ceil_FP32)
{
    constexpr int TILE_SHAPE = 32;
    constexpr int SHAPE = 128;
    TileShape::Current().SetVecTile(TILE_SHAPE, TILE_SHAPE);
    Tensor operand1(DT_FP32, {SHAPE, SHAPE}, "operand1");
    Tensor result;
    FUNCTION("TestCeil") { result = Ceil(operand1); }
}

TEST_F(VectorOperationTest, test_Floor_FP32)
{
    constexpr int TILE_SHAPE = 32;
    constexpr int SHAPE = 128;
    TileShape::Current().SetVecTile(TILE_SHAPE, TILE_SHAPE);
    Tensor operand1(DT_FP32, {SHAPE, SHAPE}, "operand1");
    Tensor result;
    FUNCTION("TestFloor") { result = Floor(operand1); }
}

TEST_F(VectorOperationTest, test_Trunc_FP32)
{
    constexpr int TILE_SHAPE = 32;
    constexpr int SHAPE = 128;
    TileShape::Current().SetVecTile(TILE_SHAPE, TILE_SHAPE);
    Tensor operand1(DT_FP32, {SHAPE, SHAPE}, "operand1");
    Tensor result;
    FUNCTION("TestTrunc") { result = Trunc(operand1); }
}

TEST_F(VectorOperationTest, test_FloorDiv_int32)
{
    TileShape::Current().SetVecTile(16, 32);
    Tensor self(DT_INT32, {64, 64}, "self");
    Tensor other(DT_INT32, {64, 64}, "other");
    Tensor result;
    FUNCTION("TestFloorDiv") { result = FloorDiv(self, other); }
}

TEST_F(VectorOperationTest, test_FloorDivs_int32)
{
    TileShape::Current().SetVecTile(16, 32);
    Tensor self(DT_INT32, {64, 64}, "self");
    Element other(DT_INT32, 2);
    Tensor result;
    FUNCTION("TestFloorDivs") { result = FloorDiv(self, other); }
}

TEST_F(VectorOperationTest, test_Reciprocal_FP32)
{
    constexpr int TILE_SHAPE = 32;
    constexpr int SHAPE = 128;
    TileShape::Current().SetVecTile(TILE_SHAPE, TILE_SHAPE);
    Tensor operand1(DT_FP32, {SHAPE, SHAPE}, "operand1");
    Tensor result;
    FUNCTION("TestReciprocal") { result = Reciprocal(operand1); }
}

TEST_F(VectorOperationTest, test_Relu_FP32)
{
    constexpr int TILE_SHAPE = 32;
    constexpr int SHAPE = 128;
    TileShape::Current().SetVecTile(TILE_SHAPE, TILE_SHAPE);
    Tensor operand1(DT_FP32, {SHAPE, SHAPE}, "operand1");
    Tensor result;
    FUNCTION("TestRelu") { result = Relu(operand1); }
}

TEST_F(VectorOperationTest, TestIndexPut_)
{
    constexpr int TILE_SHAPE = 8;
    TileShape::Current().SetVecTile(TILE_SHAPE);
    Shape shapeSelf({128, 8, 8});
    Shape shapeValues({128, 8});
    Shape shapeIndices({128});
    Tensor self(DT_INT32, shapeSelf, "self");
    Tensor values(DT_INT32, shapeValues, "values");
    Tensor indices0(DT_INT32, shapeIndices, "indices0");
    Tensor indices1(DT_INT32, shapeIndices, "indices1");
    std::vector<Tensor> indices{indices0, indices1};
    bool accumulate = false;
    Tensor result;
    FUNCTION("TestIndexPut_") { IndexPut_(self, indices, values, accumulate); }
}

TEST_F(VectorOperationTest, test_Expand_8_1_to_8_8)
{
    TileShape::Current().SetVecTile({4, 4});

    Tensor operand1(DT_FP32, {8, 1}, "operand1");
    std::vector<int64_t> dstShape = {8, 8};
    Tensor result;
    FUNCTION("TestExpand") { result = Expand(operand1, dstShape); }
}

TEST_F(VectorOperationTest, test_Expand_8_1_to_8_8_dyn)
{
    TileShape::Current().SetVecTile({3, 3});

    Tensor operand1(DT_FP32, {8, 1}, "operand1");
    std::vector<int64_t> dstShape = {8, 8};
    Tensor result;
    FUNCTION("TestExpand") { result = Expand(operand1, dstShape); }
}

TEST_F(VectorOperationTest, test_Clip_FP16)
{
    float minValue = 1.0, maxValue = 10.0;
    TileShape::Current().SetVecTile(8, 8, 8);

    Tensor src(DT_FP16, {8, 16, 16}, "src");
    Element min(DT_FP16, minValue);
    Element max(DT_FP16, maxValue);

    Tensor result;
    FUNCTION("TestClip") { result = Clip(src, min, max); }
}

TEST_F(VectorOperationTest, test_Clip_FP32_VS)
{
    float minValue = 1.0, maxValue = 10.0;
    TileShape::Current().SetVecTile(8, 8, 8);

    Tensor src(DT_FP32, {8, 16, 16}, "src");
    Element min(DT_FP32, minValue);
    Element max(DT_FP32, maxValue);

    Tensor result;
    FUNCTION("TestClip") { result = Clip(src, min, max); }
}

TEST_F(VectorOperationTest, test_Clip_FP16_VS)
{
    float minValue = 1.0, maxValue = 10.0;
    TileShape::Current().SetVecTile(8, 8, 8);

    Tensor src(DT_FP16, {8, 16, 16}, "src");
    Element min(DT_FP16, minValue);
    Element max(DT_FP16, maxValue);

    Tensor result;
    FUNCTION("TestClip") { result = Clip(src, min, max); }
}

TEST_F(VectorOperationTest, test_Clip_FP32_VV)
{
    TileShape::Current().SetVecTile(8, 8, 8);

    Tensor src(DT_FP32, {8, 16, 16}, "src");
    Tensor min(DT_FP32, {8, 16, 16}, "min");
    Tensor max(DT_FP32, {8, 16, 16}, "max");

    Tensor result;
    FUNCTION("TestClip") { result = Clip(src, min, max); }
}

TEST_F(VectorOperationTest, test_Clip_FP32_VV_BRC)
{
    TileShape::Current().SetVecTile(8, 8, 8);

    Tensor src(DT_FP32, {8, 16, 16}, "src");
    Tensor min(DT_FP32, {8, 1, 16}, "min");
    Tensor max(DT_FP32, {1, 16, 16}, "max");

    Tensor result;
    FUNCTION("TestClip") { result = Clip(src, min, max); }
}

TEST_F(VectorOperationTest, Test_Amax)
{
    TileShape::Current().SetVecTile(8, 8);
    Tensor operand(DT_FP32, {16, 16}, "operand");
    Tensor result;
    FUNCTION("TestAmax") { result = Amax(operand, -1, true); }
}

TEST_F(VectorOperationTest, Test_Amax_NegativeAxisOutOfRange)
{
    Tensor operand(DT_INT16, {4, 5120}, "operand");
    Tensor result;
    auto runAmax = [&]() {
        FUNCTION("TestAmaxNegativeAxisOutOfRange")
        {
            auto viewTensor = View(operand, {4, 138}, {4, 138}, {0, 0});
            TileShape::Current().SetVecTile(4, 16);
            result = Amax(viewTensor, -3, true);
        }
    };
    EXPECT_THROW(runAmax(), Error);
}

TEST_F(VectorOperationTest, Test_Amin)
{
    TileShape::Current().SetVecTile(8, 8);
    Tensor operand(DT_FP32, {16, 16}, "operand");
    Tensor result;
    FUNCTION("TestAmin") { result = Amin(operand, -1, true); }
}

TEST_F(VectorOperationTest, Test_Amin_NegativeAxisOutOfRange)
{
    Tensor operand(DT_INT16, {4, 5120}, "operand");
    Tensor result;
    auto runAmin = [&]() {
        FUNCTION("TestAminNegativeAxisOutOfRange")
        {
            auto viewTensor = View(operand, {4, 138}, {4, 138}, {0, 0});
            TileShape::Current().SetVecTile(4, 16);
            result = Amin(viewTensor, -3, true);
        }
    };
    EXPECT_THROW(runAmin(), Error);
}

TEST_F(VectorOperationTest, test_Atan)
{
    TileShape::Current().SetVecTile(16, 16);
    Tensor input(DT_FP16, {16, 16}, "input");
    Tensor result;
    FUNCTION("TestAtan") { result = Atan(input); }
}

TEST_F(VectorOperationTest, test_Atan2)
{
    TileShape::Current().SetVecTile(16, 16);
    Tensor input1(DT_FP16, {16, 16}, "input1");
    Tensor input2(DT_FP16, {16, 16}, "input2");
    Tensor result;
    FUNCTION("TestAtan2") { result = Atan2(input1, input2); }
}

TEST_F(VectorOperationTest, test_Gather)
{
    TileShape::Current().SetVecTile(8, 8, 8);
    Tensor operand1(DT_FP16, {8, 16}, "operand1");
    Tensor operand2(DT_INT32, {8, 16}, "operand1");
    Tensor result;
    FUNCTION("TestGather") { result = Gather(operand1, operand2, -1); }
}

TEST_F(VectorOperationTest, test_GatherMask_1)
{
    TileShape::Current().SetVecTile(8, 8);
    Tensor operand1(DT_FP16, {8, 16}, "operand1");
    Tensor result;
    FUNCTION("TestGatherMask") { result = GatherMask(operand1, 1); }
}

TEST_F(VectorOperationTest, test_GatherMask_3)
{
    TileShape::Current().SetVecTile(8, 8);
    Tensor operand1(DT_FP16, {8, 16}, "operand1");
    Tensor result;
    FUNCTION("TestGatherMask") { result = GatherMask(operand1, 3); }
}

TEST_F(VectorOperationTest, test_GatherMask_ViewSplitLastAxis_ExpectError)
{
    TileShape::Current().SetVecTile(8, 8);
    Tensor operand1(DT_FP16, {8, 32}, "operand1");
    Tensor result;
    auto runGatherMask = [&]() {
        FUNCTION("TestGatherMaskViewSplit")
        {
            auto viewTensor = View(operand1, {8, 16}, {8, 16}, {0, 0});
            result = GatherMask(viewTensor, 1);
        }
    };
    EXPECT_THROW(runGatherMask(), Error);
}

TEST_F(VectorOperationTest, test_GatherMask_ViewSplitLastAxis_Mode3_ExpectError)
{
    TileShape::Current().SetVecTile(8, 8);
    Tensor operand1(DT_FP16, {8, 32}, "operand1");
    Tensor result;
    auto runGatherMask = [&]() {
        FUNCTION("TestGatherMaskViewSplitMode3")
        {
            auto viewTensor = View(operand1, {8, 16}, {8, 16}, {0, 0});
            result = GatherMask(viewTensor, 3);
        }
    };
    EXPECT_THROW(runGatherMask(), Error);
}

TEST_F(VectorOperationTest, test_GatherMask_ViewSplitLastAxis_4D_ExpectError)
{
    TileShape::Current().SetVecTile(2, 3, 5, 16);
    Tensor operand1(DT_FP32, {4, 8, 8, 32}, "operand1");
    Tensor result;
    auto runGatherMask = [&]() {
        FUNCTION("TestGatherMaskViewSplit4D")
        {
            auto viewTensor = View(operand1, {4, 8, 8, 16}, {4, 8, 8, 16}, {0, 0, 0, 0});
            result = GatherMask(viewTensor, 2);
        }
    };
    EXPECT_THROW(runGatherMask(), Error);
}

TEST_F(VectorOperationTest, test_GatherMask_ViewEqualInputShape_Pass)
{
    TileShape::Current().SetVecTile(8, 16);
    Tensor operand1(DT_FP16, {8, 16}, "operand1");
    Tensor result;
    FUNCTION("TestGatherMaskViewEqual")
    {
        auto viewTensor = View(operand1, {8, 16}, {8, 16}, {0, 0});
        result = GatherMask(viewTensor, 1);
    }
}

TEST_F(VectorOperationTest, test_Scatter_FP16)
{
    TileShape::Current().SetVecTile(8, 16);
    Tensor operand1(DT_FP16, {8, 16}, "operand1");
    Tensor operand2(DT_INT64, {2, 16}, "operand2");
    Element operand3(DT_FP16, 1.0);
    Tensor result;
    FUNCTION("TestScatter") { result = Scatter(operand1, operand2, operand3, 0); }
}

TEST_F(VectorOperationTest, test_ScatterTensor_FP16)
{
    TileShape::Current().SetVecTile(8, 16);
    Tensor operand1(DT_FP16, {8, 16}, "operand1");
    Tensor operand2(DT_INT64, {2, 16}, "operand2");
    Tensor operand3(DT_FP16, {2, 16}, "operand3");
    Tensor result;
    FUNCTION("TestScatter") { result = Scatter(operand1, operand2, operand3, 0); }
}

TEST_F(VectorOperationTest, test_Var_FP16)
{
    TileShape::Current().SetVecTile(8, 16);
    Tensor operand1(DT_FP16, {8, 16}, "operand1");
    Tensor result;
    FUNCTION("TestVar") { result = Var(operand1); }
}

TEST_F(VectorOperationTest, test_Where)
{
    TileShape::Current().SetVecTile(8, 8);
    Tensor condition(DT_UINT8, {8, 2}, "condition");
    Tensor input(DT_FP32, {8, 16}, "input");
    Tensor other(DT_FP32, {8, 16}, "other");
    Tensor result;
    FUNCTION("TestWhere") { result = Where(condition, input, other); }
}

TEST_F(VectorOperationTest, test_Add_Brcb)
{
    TileShape::Current().SetVecTile(16, 16);
    Tensor input0(DT_FP32, {16, 16}, "input0");
    Tensor input1(DT_FP32, {16, 1}, "input0");
    Tensor result;
    config::SetOperationOption(KEY_COMBINE_AXIS, true);
    FUNCTION("TestAddBrcb") { result = Add(input0, input1); }
}

TEST_F(VectorOperationTest, test_Fmod)
{
    TileShape::Current().SetVecTile(16, 16);
    Tensor input0(DT_FP32, {16, 16}, "input0");
    Tensor input1(DT_FP32, {16, 16}, "input1");
    Tensor result;
    config::SetOperationOption(KEY_COMBINE_AXIS, false);
    FUNCTION("TestFmod") { result = Fmod(input0, input1); }
}

TEST_F(VectorOperationTest, test_Fmod_Brcb)
{
    TileShape::Current().SetVecTile(16, 16);
    Tensor input0(DT_FP32, {16, 16}, "input0");
    Tensor input1(DT_FP32, {16, 1}, "input1");
    Tensor result;
    config::SetOperationOption(KEY_COMBINE_AXIS, true);
    FUNCTION("TestFmodBrcb") { result = Fmod(input0, input1); }
}

TEST_F(VectorOperationTest, test_FmodS)
{
    TileShape::Current().SetVecTile({4, 4});
    Tensor input0(DT_FP32, {8, 8}, "input0");
    float scalar = 10.0;
    Element input1(DT_FP32, scalar);
    Tensor result;
    FUNCTION("TestFmodS") { result = Fmod(input0, input1); }
}

TEST_F(VectorOperationTest, test_LReLU)
{
    TileShape::Current().SetVecTile({4, 4});
    Tensor input0(DT_FP32, {8, 8}, "input0");
    float scalar = 0.01f;
    Element input1(DT_FP32, scalar);
    Tensor result;
    FUNCTION("TestLReLU") { result = LReLU(input0, input1); }
}

TEST_F(VectorOperationTest, Test_TopK_01)
{
    std::vector<int64_t> inputShape = {1, 16384};
    std::vector<int64_t> outputShape = {1, 2048};
    TileShape::Current().SetVecTile({1, 8192});
    Tensor input_a(DT_FP32, inputShape, "A");
    auto output = std::make_tuple(Tensor(DT_FP32, outputShape, "res"), Tensor(DT_FP32, outputShape, "resDics"));
    FUNCTION("TOPK_T") { output = TopK(input_a, 2048, -1); }
}

TEST_F(VectorOperationTest, Test_TopK_02)
{
    std::vector<int64_t> inputShape = {1, 24576};
    std::vector<int64_t> outputShape = {1, 2048};
    TileShape::Current().SetVecTile({1, 8192});
    Tensor input_a(DT_FP32, inputShape, "A");
    auto output = std::make_tuple(Tensor(DT_FP32, outputShape, "res"), Tensor(DT_FP32, outputShape, "resDics"));
    FUNCTION("TOPK_T") { output = TopK(input_a, 2048, -1); }
}

TEST_F(VectorOperationTest, Test_TopK_03)
{
    std::vector<int64_t> inputShape = {1, 49152};
    std::vector<int64_t> outputShape = {1, 2048};
    TileShape::Current().SetVecTile({1, 8192});
    Tensor input_a(DT_FP32, inputShape, "A");
    auto output = std::make_tuple(Tensor(DT_FP32, outputShape, "res"), Tensor(DT_FP32, outputShape, "resDics"));
    FUNCTION("TOPK_T") { output = TopK(input_a, 2048, -1); }
}

TEST_F(VectorOperationTest, Test_TopK_04)
{
    std::vector<int64_t> inputShape = {1, 40960};
    std::vector<int64_t> outputShape = {1, 2048};
    TileShape::Current().SetVecTile({1, 8192});
    Tensor input_a(DT_FP32, inputShape, "A");
    auto output = std::make_tuple(Tensor(DT_FP32, outputShape, "res"), Tensor(DT_FP32, outputShape, "resDics"));
    FUNCTION("TOPK_T") { output = TopK(input_a, 2048, -1); }
}

TEST_F(VectorOperationTest, Test_ArgSort_01)
{
    std::vector<int64_t> inputShape = {16, 128};
    std::vector<int64_t> outputShape = {16, 128};
    TileShape::Current().SetVecTile({4, 32});
    Tensor input_a(DT_FP32, inputShape, "A");
    Tensor output(DT_INT32, outputShape, "res");
    FUNCTION("ArgSort_T") { output = ArgSort(input_a, 1, true); }
}

TEST_F(VectorOperationTest, Test_ArgSort_02)
{
    std::vector<int64_t> inputShape = {1, 9000};
    std::vector<int64_t> outputShape = {1, 9000};
    TileShape::Current().SetVecTile({1, 5024});
    Tensor input_a(DT_FP32, inputShape, "A");
    Tensor output(DT_INT32, outputShape, "res");
    FUNCTION("ArgSort_T") { output = ArgSort(input_a, 1, false); }
}

TEST_F(VectorOperationTest, Test_BitwiseRightShift)
{
    TileShape::Current().SetVecTile({16, 16});
    Tensor self(DT_INT16, {16, 16}, "self");
    Tensor other(DT_INT16, {16, 16}, "other");
    Tensor result;
    FUNCTION("TestBitwiseRightShift") { result = BitwiseRightShift(self, other); }
}

TEST_F(VectorOperationTest, Test_BitwiseRightShift_brc)
{
    TileShape::Current().SetVecTile({16, 16});
    Tensor self(DT_INT16, {16, 16}, "self");
    Tensor other(DT_INT16, {1, 16}, "other");
    Tensor result;
    FUNCTION("TestBitwiseRightShift") { result = BitwiseRightShift(self, other); }
}

TEST_F(VectorOperationTest, Test_BitwiseRightShifts)
{
    TileShape::Current().SetVecTile({16, 16});
    Tensor self(DT_INT16, {16, 16}, "self");
    int scalar = 1;
    Element other(DT_INT16, scalar);
    Tensor result;
    FUNCTION("TestBitwiseRightShift") { result = BitwiseRightShift(self, other); }
}

TEST_F(VectorOperationTest, Test_SBitwiseRightShift)
{
    TileShape::Current().SetVecTile({16, 16});
    int scalar = 1;
    Element self(DT_INT16, scalar);
    Tensor other(DT_INT16, {16, 16}, "self");
    Tensor result;
    FUNCTION("TestBitwiseRightShift") { result = BitwiseRightShift(self, other); }
}

TEST_F(VectorOperationTest, Test_BitwiseLeftShift)
{
    TileShape::Current().SetVecTile({16, 16});
    Tensor self(DT_INT16, {16, 16}, "self");
    Tensor other(DT_INT16, {16, 16}, "other");
    Tensor result;
    FUNCTION("TestBitwiseLeftShift") { result = BitwiseLeftShift(self, other); }
}

TEST_F(VectorOperationTest, Test_BitwiseLeftShift_brc)
{
    TileShape::Current().SetVecTile({16, 16});
    Tensor self(DT_INT16, {1, 16}, "self");
    Tensor other(DT_INT16, {16, 16}, "other");
    Tensor result;
    FUNCTION("TestBitwiseLeftShift") { result = BitwiseLeftShift(self, other); }
}

TEST_F(VectorOperationTest, Test_BitwiseLeftShifts)
{
    TileShape::Current().SetVecTile({16, 16});
    Tensor self(DT_INT16, {16, 16}, "self");
    int scalar = 1;
    Element other(DT_INT16, scalar);
    Tensor result;
    FUNCTION("TestBitwiseLeftShift") { result = BitwiseLeftShift(self, other); }
}

TEST_F(VectorOperationTest, Test_SBitwiseLeftShift)
{
    TileShape::Current().SetVecTile({16, 16});
    int scalar = 1;
    Element self(DT_INT16, scalar);
    Tensor other(DT_INT16, {16, 16}, "self");
    Tensor result;
    FUNCTION("TestBitwiseLeftShift") { result = BitwiseLeftShift(self, other); }
}

TEST_F(VectorOperationTest, Test_CopySign)
{
    TileShape::Current().SetVecTile({16, 16});
    Tensor self(DT_FP32, {16, 16}, "self");
    Tensor other(DT_FP32, {16, 16}, "other");
    Tensor result;
    FUNCTION("TestBitwiseRightShift") { result = CopySign(self, other); }
}

TEST_F(VectorOperationTest, Test_CopySign_int)
{
    TileShape::Current().SetVecTile({16, 16});
    Tensor self(DT_INT32, {16, 16}, "self");
    Tensor other(DT_INT32, {16, 16}, "other");
    Tensor result;
    FUNCTION("TestBitwiseRightShift") { result = CopySign(self, other); }
}

TEST_F(VectorOperationTest, test_Pad_1D)
{
    TileShape::Current().SetVecTile(8);
    Tensor input(DT_FP32, {10}, "input");
    Tensor result;
    FUNCTION("TestPad1D") { result = Pad(input, {0, 36}, "constant", Element(DT_FP32, 0.0f)); }
}

TEST_F(VectorOperationTest, test_Pad_2D)
{
    TileShape::Current().SetVecTile(4, 4);
    Tensor input(DT_FP32, {6, 6}, "input");
    Tensor result;
    FUNCTION("TestPad2D") { result = Pad(input, {0, 12, 0, 12}, "constant", Element(DT_FP32, 0.0f)); }
}

TEST_F(VectorOperationTest, test_FillPad_1D)
{
    TileShape::Current().SetVecTile(8);
    Tensor input(DT_FP32, {10}, "input");
    Tensor result;
    FUNCTION("TestFillPad1D") { result = FillPad(input, "constant", Element(DT_FP32, 0.0f)); }
}

TEST_F(VectorOperationTest, test_FillPad_2D)
{
    TileShape::Current().SetVecTile(4, 4);
    Tensor input(DT_FP32, {6, 6}, "input");
    Tensor result;
    FUNCTION("TestFillPad2D") { result = FillPad(input, "constant", Element(DT_FP32, 0.0f)); }
}

TEST_F(VectorOperationTest, Test_Permute_3D_FP32)
{
    TileShape::Current().SetVecTile(4, 8, 16);
    Tensor input(DT_FP32, {4, 8, 16}, "input");
    Tensor result;
    FUNCTION("TestPermute3D") { result = Permute(input, {1, 0, 2}); }
}

TEST_F(VectorOperationTest, Test_Permute_3D_FP16)
{
    TileShape::Current().SetVecTile(4, 8, 16);
    Tensor input(DT_FP16, {4, 8, 16}, "input");
    Tensor result;
    FUNCTION("TestPermute3D_FP16") { result = Permute(input, {0, 2, 1}); }
}

TEST_F(VectorOperationTest, Test_Permute_4D_FP32)
{
    TileShape::Current().SetVecTile(4, 4, 8, 8);
    Tensor input(DT_FP32, {4, 4, 8, 8}, "input");
    Tensor result;
    FUNCTION("TestPermute4D") { result = Permute(input, {0, 2, 3, 1}); }
}

TEST_F(VectorOperationTest, Test_Permute_5D_FP32)
{
    TileShape::Current().SetVecTile(2, 2, 4, 4, 4);
    Tensor input(DT_FP32, {2, 2, 4, 4, 4}, "input");
    Tensor result;
    FUNCTION("TestPermute5D") { result = Permute(input, {4, 3, 2, 1, 0}); }
}

TEST_F(VectorOperationTest, Test_Permute_Identity)
{
    TileShape::Current().SetVecTile(4, 8);
    Tensor input(DT_FP32, {4, 8}, "input");
    Tensor result;
    FUNCTION("TestPermuteIdentity") { result = Permute(input, {0, 1}); }
}

TEST_F(VectorOperationTest, Test_Permute_1D)
{
    TileShape::Current().SetVecTile(32);
    Tensor input(DT_FP32, {32}, "input");
    Tensor result;
    FUNCTION("TestPermute1D") { result = Permute(input, {0}); }
}

TEST_F(VectorOperationTest, test_Erfc_fp16)
{
    TileShape::Current().SetVecTile(8, 128);
    Tensor input(DT_FP16, {8, 4608}, "input");
    Tensor result;
    FUNCTION("TestErfc") { result = Erfc(input); }
}

TEST_F(VectorOperationTest, test_Erfc_fp32)
{
    TileShape::Current().SetVecTile(8, 128);
    Tensor input(DT_FP32, {8, 4608}, "input");
    Tensor result;
    FUNCTION("TestErfc") { result = Erfc(input); }
}

class TestNanWorkspaceGraph : public ::testing::Test {
protected:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        function = std::make_shared<Function>(Program::GetInstance(), "NanWorkspace", "NanWorkspace", nullptr);
        function->paramConfigs_.combineAxis = false;
    }

    LogicalTensorPtr Tensor(DataType dtype, const Shape& shape)
    {
        auto tensor = IRBuilder().CreateTensorVar(*function, dtype, shape, SymbolicScalar::FromConcrete(shape));
        tensor->SetMemoryTypeBoth(MEM_UB);
        return tensor;
    }

    std::shared_ptr<Function> function;
};

TEST_F(TestNanWorkspaceGraph, OptionalScratchRegistrationAllocatesOnlyActualOutputs)
{
    EXPECT_EQ(OpcodeManager::Inst().GetOutputsMemType(Opcode::OP_ADDS).size(), 1U);
    for (auto opcode : {Opcode::OP_MAXIMUM, Opcode::OP_MINIMUM, Opcode::OP_MAXS, Opcode::OP_MINS, Opcode::OP_PAIRMAX,
                        Opcode::OP_PAIRMIN, Opcode::OP_PAIRARGMAX, Opcode::OP_PAIRARGMIN, Opcode::OP_ROWMAXLINE,
                        Opcode::OP_ROWMINLINE}) {
        const bool pairArg = opcode == Opcode::OP_PAIRARGMAX || opcode == Opcode::OP_PAIRARGMIN;
        const bool binary = opcode == Opcode::OP_MAXIMUM || opcode == Opcode::OP_MINIMUM ||
                            opcode == Opcode::OP_PAIRMAX || opcode == Opcode::OP_PAIRMIN;
        const size_t nativeOutputs = pairArg ? 2 : 1;
        ASSERT_EQ(OpcodeManager::Inst().GetOutputsMemType(opcode).size(), nativeOutputs + 1);
        for (bool scratch : {false, true}) {
            LogicalTensors inputs;
            const size_t inputCount = pairArg ? 4 : binary ? 2 : 1;
            for (size_t i = 0; i < inputCount; ++i)
                inputs.push_back(Tensor(pairArg && i % 2 == 1 ? DT_INT32 : DT_FP32, {4, 32}));
            LogicalTensors outputs{Tensor(DT_FP32, {4, 32})};
            if (pairArg)
                outputs.push_back(Tensor(DT_INT32, {4, 32}));
            if (scratch)
                outputs.push_back(Tensor(DT_UINT8, {1, 32}));
            auto& op = IRBuilder().CreateTensorOpStmt(*function, opcode, inputs, outputs);
            ConvertInserter inserter(AssignMemoryTypeMode::VIEW_ASSEMBLE);
            ASSERT_EQ(MemoryPathUtils::AssignOpcodeDefinedMemoryTypes(inserter, op), SUCCESS);
            ASSERT_EQ(op.GetOOperands().size(), nativeOutputs + scratch);
            for (size_t i = 0; i < outputs.size(); ++i)
                outputs[i]->memoryrange.memId = static_cast<int>(i);
            std::unordered_map<int, TensorAllocMsg> allocations;
            ASSERT_EQ(AddAlloc().SetTensorAllocMsg(op, allocations), SUCCESS);
            EXPECT_EQ(allocations.size(), nativeOutputs + scratch);
        }
    }
}

TEST_F(TestNanWorkspaceGraph, OnlyNanPathsAppendScratch)
{
    for (auto dtype : {DT_FP32, DT_INT32}) {
        for (bool propagate : {false, true}) {
            auto x = Tensor(dtype, {4, 32});
            auto& root = function->AddOperation(Opcode::OP_MAXIMUM, {x, x}, {Tensor(dtype, {4, 32})});
            root.SetAttribute(OP_ATTR_PREFIX + "propagate_nan", propagate);
            ExpandFunction::CurrentTileOpGuard guard(root);
            for (auto opcode : {Opcode::OP_MAXIMUM, Opcode::OP_MINIMUM, Opcode::OP_MAXS, Opcode::OP_MINS}) {
                const bool binary = opcode == Opcode::OP_MAXIMUM || opcode == Opcode::OP_MINIMUM;
                auto& op = AddNanSmallOperation(*function, opcode, binary ? LogicalTensors{x, x} : LogicalTensors{x},
                                                {Tensor(dtype, {4, 32})});
                EXPECT_EQ(op.GetOOperands().size(), dtype == DT_FP32 && !propagate ? 2U : 1U);
            }
        }
    }
    auto x = Tensor(DT_FP32, {4, 32});
    auto& max = AddNanSmallOperation(*function, Opcode::OP_MAXS, {x}, {Tensor(DT_FP32, {4, 32})});
    EXPECT_EQ(max.GetOOperands().size(), 1U);
}

TEST_F(TestNanWorkspaceGraph, NativeArgReductionKeepsOriginalOperands)
{
    auto x = Tensor(DT_FP32, {4, 32});
    auto value = Tensor(DT_FP32, {4, 1});
    auto index = Tensor(DT_INT32, {4, 1});
    auto nativeScratch = Tensor(DT_FP32, {4, 32});
    auto& root = function->AddOperation(Opcode::OP_ROWARGMAX_SINGLE, {x}, {index});
    ExpandFunction::CurrentTileOpGuard guard(root);
    auto& local = AddNanReduceOperation(*function, "TILE_ROWARGMAXWITHVALUE_SINGLE", {x}, {value, index, nativeScratch},
                                        1);
    EXPECT_FALSE(local.GetBoolAttribute(OP_ATTR_PREFIX + "nan_tile"));
    EXPECT_EQ(local.GetOutputOperand(2), nativeScratch);
    auto& merge = AddNanSmallOperation(*function, Opcode::OP_PAIRARGMAX, {value, index, value, index}, {value, index});
    EXPECT_EQ(merge.GetOOperands().size(), 2U);
    auto& finalize = function->AddOperation(Opcode::OP_ADDS, {index}, {Tensor(DT_INT32, {4, 1})});
    EXPECT_EQ(finalize.GetOOperands().size(), 1U);
}

TEST_F(TestNanWorkspaceGraph, IgnoreNanArgAppliesOffsetsAndFinalizesOnlyAtRoot)
{
    for (const bool split : {false, true}) {
        function->ResetOperations();
        auto input = Tensor(DT_FP32, {4, 65});
        auto result = Tensor(DT_INT32, {4, 1});
        const auto opcode = Opcode::OP_ROWARGMAX_SINGLE;
        auto& root = function->AddOperation(opcode, {input}, {result});
        root.SetAttribute(OP_ATTR_PREFIX + "AXIS", int64_t{1});
        root.SetAttribute(OP_ATTR_PREFIX + "propagate_nan", false);
        ExpandFunction::CurrentTileOpGuard guard(root);
        TileShape tile;
        tile.SetVecTile(4, split ? 32 : 65);
        auto expand = TiledFuncRegistry::GetInstance().GetTiledFunc(opcode);
        ASSERT_TRUE(expand);
        expand(*function, tile, {input}, {result}, root);

        std::vector<int64_t> offsets;
        std::vector<bool> mergeFinalize;
        for (auto& op : function->Operations(false)) {
            EXPECT_NE(op.GetOpcode(), Opcode::OP_ADDS);
            if (op.GetOpcode() == Opcode::OP_ROWARGMAXWITHVALUE_SINGLE) {
                offsets.push_back(op.GetIntAttribute(OP_ATTR_PREFIX + "nan_index_offset"));
                EXPECT_EQ(op.GetBoolAttribute(OP_ATTR_PREFIX + "nan_finalize"), !split);
                EXPECT_EQ(op.GetOOperands().size(), 3U);
            } else if (op.GetOpcode() == Opcode::OP_PAIRARGMAX) {
                mergeFinalize.push_back(op.GetBoolAttribute(OP_ATTR_PREFIX + "nan_finalize"));
                EXPECT_EQ(op.GetOOperands().size(), 3U);
            }
        }
        EXPECT_EQ(offsets, (split ? std::vector<int64_t>{0, 32, 64} : std::vector<int64_t>{0}));
        EXPECT_EQ(mergeFinalize, (split ? std::vector<bool>{false, true} : std::vector<bool>{}));
    }
}

TEST_F(TestNanWorkspaceGraph, RowScratchUsesPaddedWidth)
{
    for (const auto& [width, dtype, bytes] :
         {std::make_tuple(255, DT_FP32, 6560), std::make_tuple(511, DT_FP16, 6688)}) {
        auto x = Tensor(dtype, {4, width});
        auto result = Tensor(dtype, {4, 1});
        auto& root = function->AddOperation(Opcode::OP_ROWMAX_SINGLE, {x}, {result});
        root.SetAttribute(OP_ATTR_PREFIX + "propagate_nan", false);
        ExpandFunction::CurrentTileOpGuard guard(root);
        auto& op = AddNanReduceOperation(*function, "TILE_ROWMAX_SINGLE", {x}, {result}, 1);
        EXPECT_EQ(op.GetOutputOperand(1)->shape, (Shape{1, bytes}));
    }
}

TEST_F(TestNanWorkspaceGraph, PairScratchRemainsExplicitWhenNativeOperandsFillUb)
{
    const auto ub = Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB);
    constexpr int64_t width = 32;
    const Shape shape{static_cast<int64_t>(ub / (3 * width * sizeof(float))), width};
    auto x = Tensor(DT_FP32, shape);
    auto y = Tensor(DT_FP32, shape);
    auto result = Tensor(DT_FP32, shape);
    auto& root = function->AddOperation(Opcode::OP_MAXIMUM, {x, y}, {result});
    root.SetAttribute(OP_ATTR_PREFIX + "propagate_nan", false);
    ExpandFunction::CurrentTileOpGuard guard(root);
    auto& op = AddNanSmallOperation(*function, Opcode::OP_PAIRMAX, {x, y}, {result});
    ASSERT_EQ(op.GetOutputOperandSize(), 2U);
    EXPECT_EQ(op.GetOutputOperand(1)->shape.back(), (32 + 256) * shape.front() + 32);
}

TEST_F(TestNanWorkspaceGraph, NanCompareScratchCoversFullRepeats)
{
    for (auto dtype : {DT_FP16, DT_FP32}) {
        const int64_t elementsPerRepeat = 256 / BytesOf(dtype);
        for (int64_t width : {elementsPerRepeat / 8, elementsPerRepeat, elementsPerRepeat + elementsPerRepeat / 8}) {
            const Shape shape{3, width};
            auto x = Tensor(dtype, shape);
            auto& root = function->AddOperation(Opcode::OP_MAXIMUM, {x, x}, {Tensor(dtype, shape)});
            root.SetAttribute(OP_ATTR_PREFIX + "propagate_nan", false);
            ExpandFunction::CurrentTileOpGuard guard(root);
            for (auto code : {Opcode::OP_MAXIMUM, Opcode::OP_MINIMUM, Opcode::OP_MAXS, Opcode::OP_MINS,
                              Opcode::OP_PAIRMAX, Opcode::OP_PAIRMIN}) {
                const bool scalar = code == Opcode::OP_MAXS || code == Opcode::OP_MINS;
                auto& op = AddNanSmallOperation(*function, code, scalar ? LogicalTensors{x} : LogicalTensors{x, x},
                                                {Tensor(dtype, shape)});
                const int64_t copyBytes = width > elementsPerRepeat ? 3 * 512 : 3 * 256;
                EXPECT_EQ(op.GetOutputOperand(1)->shape, (Shape{1, 3 * 32 + 32 + copyBytes}));
            }
        }
    }
}

TEST_F(TestNanWorkspaceGraph, CastColumnScratchIncludesCompactInput)
{
    auto x = Tensor(DT_FP16, {3, 4, 1});
    auto converted = Tensor(DT_FP32, {3, 4, 1});
    IRBuilder().CreateTensorOpStmt(*function, Opcode::OP_CAST, {x}, {converted});
    auto value = Tensor(DT_FP32, {1, 4, 1});
    auto index = Tensor(DT_INT32, {1, 4, 1});
    auto& root = function->AddOperation(Opcode::OP_ROWARGMIN_SINGLE, {converted}, {index});
    root.SetAttribute(OP_ATTR_PREFIX + "propagate_nan", false);
    ExpandFunction::CurrentTileOpGuard guard(root);
    auto& op = AddNanReduceOperation(*function, "TILE_ROWARGMINWITHVALUE_LINE", {converted}, {value, index}, 0);
    EXPECT_EQ(op.GetOutputOperand(2)->shape, (Shape{1, 1, 1920}));
}

TEST_F(TestNanWorkspaceGraph, PairArgScratchUsesLogicalSliceAndGuardsWideTail)
{
    for (const auto& [shape, scratch] :
         {std::make_pair(Shape{16, 50, 1}, Shape{1, 1, 3232}), std::make_pair(Shape{4, 96}, Shape{1, 5024})}) {
        auto value = Tensor(DT_FP32, shape);
        auto index = Tensor(DT_INT32, shape);
        auto& root = function->AddOperation(Opcode::OP_ROWARGMAX_SINGLE, {value}, {index});
        root.SetAttribute(OP_ATTR_PREFIX + "propagate_nan", false);
        ExpandFunction::CurrentTileOpGuard guard(root);
        auto& op = AddNanSmallOperation(*function, Opcode::OP_PAIRARGMAX, {value, index, value, index}, {value, index});
        EXPECT_EQ(op.GetOutputOperand(2)->shape, scratch);
        std::vector<bool> rowPad;
        op.GetAttr(OpAttributeKey::rowPad, rowPad);
        EXPECT_EQ(rowPad, (std::vector<bool>{true, true, true, true}));
    }
}

TEST_F(TestNanWorkspaceGraph, NanModeChangesOperationHash)
{
    for (const auto code : {Opcode::OP_MAXIMUM, Opcode::OP_MINIMUM, Opcode::OP_MAXS, Opcode::OP_MINS,
                            Opcode::OP_ROWMAX_SINGLE, Opcode::OP_ROWMIN_SINGLE}) {
        auto input = Tensor(DT_FP32, {4, 32});
        LogicalTensors inputs{input};
        const bool binary = code == Opcode::OP_MAXIMUM || code == Opcode::OP_MINIMUM || code == Opcode::OP_PAIRMAX ||
                            code == Opcode::OP_PAIRMIN;
        if (binary) {
            inputs.push_back(input);
        }
        auto& op = function->AddOperation(code, inputs, {Tensor(DT_FP32, {4, 32})});
        op.SetAttribute(OP_ATTR_PREFIX + "propagate_nan", true);
        const auto nativeHash = op.ComputeHashOrderless();
        op.SetAttribute(OP_ATTR_PREFIX + "propagate_nan", false);
        EXPECT_NE(op.ComputeHashOrderless(), nativeHash) << static_cast<int>(code);
    }
    for (const auto code : {Opcode::OP_MAXIMUM, Opcode::OP_MINIMUM, Opcode::OP_MAXS, Opcode::OP_MINS,
                            Opcode::OP_PAIRMAX, Opcode::OP_PAIRMIN, Opcode::OP_ROWMAX_SINGLE, Opcode::OP_ROWMIN_SINGLE,
                            Opcode::OP_ROWMAXLINE, Opcode::OP_ROWMINLINE}) {
        auto input = Tensor(DT_FP32, {4, 32});
        LogicalTensors inputs{input};
        const bool binary = code == Opcode::OP_MAXIMUM || code == Opcode::OP_MINIMUM || code == Opcode::OP_PAIRMAX ||
                            code == Opcode::OP_PAIRMIN;
        if (binary) {
            inputs.push_back(input);
        }
        auto& op = function->AddOperation(code, inputs, {Tensor(DT_FP32, {4, 32})});
        const auto nativeHash = op.ComputeHashOrderless();
        op.SetAttribute(OP_ATTR_PREFIX + "nan_tile", true);
        EXPECT_NE(op.ComputeHashOrderless(), nativeHash) << static_cast<int>(code);
    }
}

TEST_F(VectorOperationTest, NanBroadcastCreatesOnlyConsumedExtrema)
{
    TileShape::Current().SetVecTile(4, 32);
    for (bool maximum : {false, true}) {
        for (const auto& shape : {Shape{1, 8}, Shape{8}}) {
            Tensor x(DT_FP32, {3, 8}, "input");
            Tensor y(DT_FP32, shape, "broadcast");
            FUNCTION("NanBroadcast")
            {
                auto result = maximum ? Maximum(x, y, false) : Minimum(x, y, false);
                size_t extremaCount = 0;
                size_t expandCount = 0;
                for (const auto& op : Program::GetInstance().GetCurrentFunction()->Operations()) {
                    extremaCount += op.GetOpcode() == Opcode::OP_MAXIMUM || op.GetOpcode() == Opcode::OP_MINIMUM;
                    expandCount += op.GetOpcode() == Opcode::OP_EXPAND;
                }
                EXPECT_EQ(extremaCount, 1U);
                EXPECT_EQ(expandCount, 1U);
                EXPECT_EQ(result.GetShape(), (Shape{3, 8}));
            }
        }
    }
}

TEST_F(TestNanWorkspaceGraph, ArgScratchIncludesCompactInputAndCompareGuard)
{
    for (const bool maximum : {false, true}) {
        for (const auto& [shape, axis, bytes] :
             {std::make_tuple(Shape{2, 2, 8}, 0, 1376), std::make_tuple(Shape{2, 4, 16}, 0, 2144),
              std::make_tuple(Shape{2, 17}, 1, 960), std::make_tuple(Shape{2, 64}, 1, 1440)}) {
            auto x = Tensor(DT_FP32, shape);
            auto resultShape = shape;
            resultShape[axis] = 1;
            auto value = Tensor(DT_FP32, resultShape);
            auto index = Tensor(DT_INT32, resultShape);
            auto& root = function->AddOperation(Opcode::OP_ROWARGMAX_SINGLE, {x}, {index});
            root.SetAttribute(OP_ATTR_PREFIX + "propagate_nan", false);
            ExpandFunction::CurrentTileOpGuard guard(root);
            const auto name = maximum ? "TILE_ROWARGMAXWITHVALUE_" : "TILE_ROWARGMINWITHVALUE_";
            auto& op = AddNanReduceOperation(
                *function, std::string(name) + (static_cast<size_t>(axis) + 1 == shape.size() ? "SINGLE" : "LINE"), {x},
                {value, index}, axis);
            EXPECT_EQ(op.GetOutputOperand(2)->shape.back(), bytes);
        }
    }
}

TEST_F(TestNanWorkspaceGraph, ColumnScratchExcludesOutputValueAndIndex)
{
    for (const bool arg : {false, true}) {
        for (const bool maximum : {false, true}) {
            for (const auto& [width, bytes] :
                 {std::make_pair(17, arg ? 1728 : 800), std::make_pair(64, arg ? 2688 : 1152),
                  std::make_pair(65, arg ? 3136 : 1696)}) {
                auto input = Tensor(DT_FP32, {3, width});
                auto value = Tensor(DT_FP32, {1, width});
                auto index = Tensor(DT_INT32, {1, width});
                auto& root = function->AddOperation(Opcode::OP_ROWMAX_SINGLE, {input}, {value});
                root.SetAttribute(OP_ATTR_PREFIX + "propagate_nan", false);
                ExpandFunction::CurrentTileOpGuard guard(root);
                const std::string name = arg ? (maximum ? "TILE_ROWARGMAXWITHVALUE_LINE" :
                                                          "TILE_ROWARGMINWITHVALUE_LINE") :
                                               (maximum ? "TILE_ROWMAXLINE" : "TILE_ROWMINLINE");
                auto& op = AddNanReduceOperation(*function, name, {input},
                                                 arg ? LogicalTensors{value, index} : LogicalTensors{value}, 0);
                EXPECT_EQ(op.GetOOperands().back()->shape, (Shape{1, bytes}));
            }
        }
    }
}

TEST_F(VectorOperationTest, NanExtremaApiModesPreserveShapeAndDtype)
{
    for (const auto dtype : {DT_FP16, DT_FP32, DT_BF16}) {
        for (int mode = 0; mode < 3; ++mode) {
            for (const auto& [axis, keepDim] : {std::make_pair(0, false), std::make_pair(0, true),
                                                std::make_pair(1, false), std::make_pair(1, true)}) {
                SCOPED_TRACE(std::to_string(dtype) + ":" + std::to_string(mode) + ":" + std::to_string(axis) + ":" +
                             std::to_string(keepDim));
                TileShape::Current().SetVecTile(4, 32);
                Tensor input(dtype, {3, 64}, "input");
                const bool propagate = mode != 2;
                FUNCTION("NanExtremaApiModes")
                {
                    const std::vector<Tensor> values = {
                        mode == 0 ? Amin(input, axis, keepDim) : Amin(input, axis, keepDim, propagate),
                        mode == 0 ? Amax(input, axis, keepDim) : Amax(input, axis, keepDim, propagate)};
                    const std::vector<Tensor> indices = {
                        mode == 0 ? ArgMin(input, axis, keepDim) : ArgMin(input, axis, keepDim, propagate),
                        mode == 0 ? ArgMax(input, axis, keepDim) : ArgMax(input, axis, keepDim, propagate)};
                    const std::vector<Tensor> binaries = {
                        mode == 0 ? Minimum(input, input) : Minimum(input, input, propagate),
                        mode == 0 ? Maximum(input, input) : Maximum(input, input, propagate)};
                    const Shape resultShape = keepDim ? (axis == 0 ? Shape{1, 64} : Shape{3, 1}) :
                                                        (axis == 0 ? Shape{64} : Shape{3});
                    for (const auto& value : values) {
                        EXPECT_EQ(value.GetShape(), resultShape);
                        EXPECT_EQ(value.GetDataType(), dtype);
                    }
                    for (const auto& index : indices) {
                        EXPECT_EQ(index.GetShape(), resultShape);
                        EXPECT_EQ(index.GetDataType(), DT_INT32);
                    }
                    for (const auto& binary : binaries) {
                        EXPECT_EQ(binary.GetShape(), input.GetShape());
                        EXPECT_EQ(binary.GetDataType(), dtype);
                    }
                    size_t extrema = 0;
                    for (const auto& op : Program::GetInstance().GetCurrentFunction()->Operations()) {
                        const auto code = op.GetOpcode();
                        if (code == Opcode::OP_ROWMAX_SINGLE || code == Opcode::OP_ROWMIN_SINGLE ||
                            code == Opcode::OP_ROWARGMAX_SINGLE || code == Opcode::OP_ROWARGMIN_SINGLE ||
                            code == Opcode::OP_MAXIMUM || code == Opcode::OP_MINIMUM) {
                            ++extrema;
                            const auto attribute = OP_ATTR_PREFIX + "propagate_nan";
                            EXPECT_EQ(!op.HasAttr(attribute) || op.GetBoolAttribute(attribute), propagate);
                        }
                    }
                    EXPECT_EQ(extrema, 6U);
                }
            }
        }
    }
}

TEST_F(VectorOperationTest, NanScalarModesAndConstants)
{
    for (const auto dtype : {DT_FP16, DT_FP32, DT_BF16, DT_INT32}) {
        for (const bool propagate : {false, true}) {
            for (const float scalar : {2.0f, std::numeric_limits<float>::quiet_NaN()}) {
                if (dtype == DT_INT32 && std::isnan(scalar)) {
                    continue;
                }
                TileShape::Current().SetVecTile(4, 32);
                Tensor input(dtype, {3, 64}, "input");
                FUNCTION("NanScalarModes")
                {
                    Element value = dtype == DT_INT32 ? Element(DT_INT32, 2) : Element(DT_FP32, scalar);
                    auto lo = Minimum(input, value, propagate);
                    auto hi = Maximum(input, value, propagate);
                    EXPECT_EQ(lo.GetShape(), input.GetShape());
                    EXPECT_EQ(hi.GetShape(), input.GetShape());
                    EXPECT_EQ(lo.GetDataType(), dtype);
                    EXPECT_EQ(hi.GetDataType(), dtype);
                    if (!propagate && std::isnan(scalar)) {
                        EXPECT_EQ(lo.GetStorage(), input.GetStorage());
                        EXPECT_EQ(hi.GetStorage(), input.GetStorage());
                    } else {
                        size_t extrema = 0;
                        for (const auto& op : Program::GetInstance().GetCurrentFunction()->Operations()) {
                            if (op.GetOpcode() == Opcode::OP_MAXS || op.GetOpcode() == Opcode::OP_MINS) {
                                ++extrema;
                                if (dtype != DT_INT32 && !propagate) {
                                    EXPECT_FALSE(op.GetBoolAttribute(OP_ATTR_PREFIX + "propagate_nan"));
                                }
                            }
                        }
                        EXPECT_EQ(extrema, 2U);
                    }
                }
            }
        }
    }
}
