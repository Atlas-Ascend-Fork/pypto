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
 * \file test_flow_verifier.cpp
 * \brief Unit tests for FlowVerifier::VerifyResult (FP8*, DT_HF8 decode path, padded raw views).
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>

#include "interface/interpreter/flow_verifier.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "tilefwk/tensor.h"

namespace npu::tile_fwk {

namespace {

LogicalTensorDataPtr MakeFp8E4m3Scalar(uint8_t byte)
{
    Tensor t(DT_FP8E4M3, {1});
    auto raw = RawTensorData::CreateConstantTensor(t, byte);
    return std::make_shared<LogicalTensorData>(raw);
}

LogicalTensorDataPtr MakeFp8E5m2Scalar(uint8_t byte)
{
    Tensor t(DT_FP8E5M2, {1});
    auto raw = RawTensorData::CreateConstantTensor(t, byte);
    return std::make_shared<LogicalTensorData>(raw);
}

LogicalTensorDataPtr MakeFp8E8m0Scalar(uint8_t byte)
{
    Tensor t(DT_FP8E8M0, {1});
    auto raw = RawTensorData::CreateConstantTensor(t, byte);
    return std::make_shared<LogicalTensorData>(raw);
}

LogicalTensorDataPtr MakeHf8Scalar(uint8_t byte)
{
    Tensor t(DT_HF8, {1});
    auto raw = RawTensorData::CreateConstantTensor(t, byte);
    return std::make_shared<LogicalTensorData>(raw);
}

constexpr float kVerifyRtol = 1e-3f;
constexpr float kVerifyAtol = 1e-2f;

LogicalTensorDataPtr MakeInt8ViewOnPaddedRaw(const std::vector<int8_t>& rawVals, const std::vector<int64_t>& rawShape,
                                             const std::vector<int64_t>& viewShape, const std::vector<int64_t>& offset)
{
    Tensor t(DT_INT8, rawShape);
    auto raw = RawTensorData::CreateTensor<int8_t>(t, rawVals);
    return std::make_shared<LogicalTensorData>(raw, viewShape, viewShape, offset);
}

// FP4打包张量比对: 一字节两nibble, validShape为逻辑值个数
LogicalTensorDataPtr MakeFp4PackedTensor(DataType dtype, const std::vector<int64_t>& logicalShape,
                                         const std::vector<uint8_t>& packedBytes)
{
    Tensor t(dtype, logicalShape);
    auto raw = RawTensorData::CreateTensor<uint8_t>(t, packedBytes);
    return std::make_shared<LogicalTensorData>(raw);
}

} // namespace

// FP4_E2M1: 相邻编码(0x01=高nibble 0/低nibble 1 => 值[0, 0.5])解码比对, 存储字节不同但解码值相同则PASS
TEST(FlowVerifierFp4Test, E2M1SameEncodingPackedPass)
{
    // 打包1字节(2个逻辑值): golden与output用相同字节
    auto golden = MakeFp4PackedTensor(DT_FP4_E2M1, {2}, {0x21}); // [1.0, 0.5]
    auto output = MakeFp4PackedTensor(DT_FP4_E2M1, {2}, {0x21});
    auto result = FlowVerifier::VerifyResult(golden, output, kVerifyRtol, kVerifyAtol);
    EXPECT_TRUE(result.Check()) << "FP4_E2M1相同编码应PASS";
}

// 相邻编码小差异在容差内PASS: 0x21([1.0, 0.5]) vs 0x22([1.0, 1.5])
TEST(FlowVerifierFp4Test, E2M1AdjacentNibbleDecodePassBytesDiffer)
{
    auto golden = MakeFp4PackedTensor(DT_FP4_E2M1, {2}, {0x21}); // [1.0, 0.5]
    auto output = MakeFp4PackedTensor(DT_FP4_E2M1, {2}, {0x22}); // [1.0, 1.5]
    // 0.5 vs 1.5 差1.0 > atol(0.001)应FAIL
    auto result = FlowVerifier::VerifyResult(golden, output, kVerifyRtol, kVerifyAtol);
    EXPECT_FALSE(result.Check()) << "FP4_E2M1值差1.0超出容差应FAIL";
}

// nibble错位检测: 高低nibble互换(0x12=[0.5,1.0] vs 0x21=[1.0,0.5])应FAIL
TEST(FlowVerifierFp4Test, E2M1NibbleSwapFails)
{
    auto golden = MakeFp4PackedTensor(DT_FP4_E2M1, {2}, {0x21}); // [1.0, 0.5]
    auto output = MakeFp4PackedTensor(DT_FP4_E2M1, {2}, {0x12}); // [0.5, 1.0]
    auto result = FlowVerifier::VerifyResult(golden, output, kVerifyRtol, kVerifyAtol);
    EXPECT_FALSE(result.Check()) << "nibble错位应FAIL";
}

// E1M2/HF4: 相同编码PASS覆盖两个分支
TEST(FlowVerifierFp4Test, E1M2AndHf4SameEncodingPass)
{
    auto g1 = MakeFp4PackedTensor(DT_FP4_E1M2, {2}, {0x23}); // [0.75, 0.5]
    auto o1 = MakeFp4PackedTensor(DT_FP4_E1M2, {2}, {0x23});
    EXPECT_TRUE(FlowVerifier::VerifyResult(g1, o1, kVerifyRtol, kVerifyAtol).Check());

    auto g2 = MakeFp4PackedTensor(DT_HF4, {2}, {0x23}); // [0.75, 0.5]
    auto o2 = MakeFp4PackedTensor(DT_HF4, {2}, {0x23});
    EXPECT_TRUE(FlowVerifier::VerifyResult(g2, o2, kVerifyRtol, kVerifyAtol).Check());
}

// HF4: 值差超容差FAIL(0x23=[0.75,0.5] vs 0x24=[0.75,1.0], 差0.5)
TEST(FlowVerifierFp4Test, Hf4ValueDiffFails)
{
    auto golden = MakeFp4PackedTensor(DT_HF4, {2}, {0x23});
    auto output = MakeFp4PackedTensor(DT_HF4, {2}, {0x24});
    auto result = FlowVerifier::VerifyResult(golden, output, kVerifyRtol, kVerifyAtol);
    EXPECT_FALSE(result.Check()) << "HF4值差0.5超出容差应FAIL";
}

// Golden 0x0E and output 0x0F are adjacent E4M3 codes (~0.0273 vs ~0.0293). Raw uint8 differs, but decode
// comparison with moderate atol passes (see flow_verifier FP8 decode path).
TEST(FlowVerifierFp8Test, E4M3AdjacentCodesDecodePassBytesDiffer)
{
    constexpr uint8_t kGoldenByte = 0x0E;
    constexpr uint8_t kOutputByte = 0x0F;
    const float kApproxGolden = std::ldexp(1.0f + 6.0f / 8.0f, 1 - 7); // exp=1, mant=6
    const float kApproxOutput = std::ldexp(1.0f + 7.0f / 8.0f, 1 - 7); // exp=1, mant=7

    auto golden = MakeFp8E4m3Scalar(kGoldenByte);
    auto output = MakeFp8E4m3Scalar(kOutputByte);

    ASSERT_EQ(golden->GetDataType(), DT_FP8E4M3);
    ASSERT_EQ(output->GetDataType(), DT_FP8E4M3);
    EXPECT_NE(golden->Get<uint8_t>(0), output->Get<uint8_t>(0));
    EXPECT_GT(std::fabs(kApproxOutput - kApproxGolden), 0.0f);
    EXPECT_LT(std::fabs(kApproxOutput - kApproxGolden), kVerifyAtol);

    auto result = FlowVerifier::VerifyResult(golden, output, kVerifyRtol, kVerifyAtol);
    EXPECT_TRUE(result.Check()) << "E4M3 0x0E vs 0x0F: decode+tol should pass while storage bytes differ";
}

// E5M2 [S][EEEEE][MM]: 0x14 = exp=5 mant=0, 0x15 = exp=5 mant=1 -> 2^-10 vs 2^-10*1.25
TEST(FlowVerifierFp8Test, E5M2AdjacentCodesDecodePassBytesDiffer)
{
    constexpr uint8_t kGoldenByte = 0x14;
    constexpr uint8_t kOutputByte = 0x15;
    const float kApproxGolden = std::ldexp(1.0f + 0.0f / 4.0f, 5 - 15);
    const float kApproxOutput = std::ldexp(1.0f + 1.0f / 4.0f, 5 - 15);

    auto golden = MakeFp8E5m2Scalar(kGoldenByte);
    auto output = MakeFp8E5m2Scalar(kOutputByte);

    ASSERT_EQ(golden->GetDataType(), DT_FP8E5M2);
    ASSERT_EQ(output->GetDataType(), DT_FP8E5M2);
    EXPECT_NE(golden->Get<uint8_t>(0), output->Get<uint8_t>(0));
    EXPECT_GT(std::fabs(kApproxOutput - kApproxGolden), 0.0f);
    EXPECT_LT(std::fabs(kApproxOutput - kApproxGolden), kVerifyAtol);

    auto result = FlowVerifier::VerifyResult(golden, output, kVerifyRtol, kVerifyAtol);
    EXPECT_TRUE(result.Check()) << "E5M2 0x14 vs 0x15: decode+tol should pass while storage bytes differ";
}

// E8M0 [S][EEEEEEE]: 0x32 vs 0x33 -> exp 50 vs 51 (positive), 2^(50-63) vs 2^(51-63)
// e8m0: unsigned exponent with bias 127. 0x32 = 2^-77, 0x33 = 2^-76 (adjacent, ratio 2).
TEST(FlowVerifierFp8Test, E8M0AdjacentCodesDecodePassBytesDiffer)
{
    constexpr uint8_t kGoldenByte = 0x32;
    constexpr uint8_t kOutputByte = 0x33;
    const float kApproxGolden = std::ldexp(1.0f, 50 - 127);
    const float kApproxOutput = std::ldexp(1.0f, 51 - 127);

    auto golden = MakeFp8E8m0Scalar(kGoldenByte);
    auto output = MakeFp8E8m0Scalar(kOutputByte);

    ASSERT_EQ(golden->GetDataType(), DT_FP8E8M0);
    ASSERT_EQ(output->GetDataType(), DT_FP8E8M0);
    EXPECT_NE(golden->Get<uint8_t>(0), output->Get<uint8_t>(0));
    EXPECT_GT(std::fabs(kApproxOutput - kApproxGolden), 0.0f);
    EXPECT_LT(std::fabs(kApproxOutput - kApproxGolden), kVerifyAtol);

    auto result = FlowVerifier::VerifyResult(golden, output, kVerifyRtol, kVerifyAtol);
    EXPECT_TRUE(result.Check()) << "E8M0 0x32 vs 0x33: decode+tol should pass while storage bytes differ";
}

// e8m0 byte 0 decodes to exact zero (MX zero-scale practice); 0x90 decodes to 2^17 (not negative).
// Under the old sign+bias63 decode, 0x00 vs 0x90 differ by ~2^-47 (within atol, silently passes);
// under the corrected unsigned+bias127 decode, |0 - 2^17| far exceeds atol and must FAIL.
TEST(FlowVerifierFp8Test, E8M0ZeroVsHighByteMustFail)
{
    auto golden = MakeFp8E8m0Scalar(0x00);
    auto output = MakeFp8E8m0Scalar(0x90);
    auto result = FlowVerifier::VerifyResult(golden, output, kVerifyRtol, kVerifyAtol);
    EXPECT_FALSE(result.Check()) << "0x00(zero) vs 0x90(2^17): must fail; old sign+bias63 decode silently passed";
}

// e8m0 >=0x80 decodes as positive powers of two (unsigned exponent, bias 127).
// 0x83 vs 0x84 = 2^4 vs 2^5 (difference 16, must FAIL); old sign+bias63 decode yields two negative
// tiny values differing by ~2^-59 and silently passes within atol.
TEST(FlowVerifierFp8Test, E8M0AdjacentHighBytesMustFail)
{
    auto golden = MakeFp8E8m0Scalar(0x83);
    auto output = MakeFp8E8m0Scalar(0x84);
    auto result = FlowVerifier::VerifyResult(golden, output, kVerifyRtol, kVerifyAtol);
    EXPECT_FALSE(result.Check()) << "0x83(16) vs 0x84(32): must fail; old sign+bias63 decode silently passed";

    auto golden2 = MakeFp8E8m0Scalar(0x87);
    auto output2 = MakeFp8E8m0Scalar(0x88);
    auto result2 = FlowVerifier::VerifyResult(golden2, output2, kVerifyRtol, kVerifyAtol);
    EXPECT_FALSE(result2.Check()) << "0x87(256) vs 0x88(512): must fail under unsigned bias-127 decode";
}

// DT_HF8: VerifyResult decodes uint8 storage (HF8.md) then applies rtol/atol like FP8.
TEST(FlowVerifierHf8Test, SameEncodingPasses)
{
    constexpr uint8_t kByte = 0x2F;
    auto golden = MakeHf8Scalar(kByte);
    auto output = MakeHf8Scalar(kByte);
    ASSERT_EQ(golden->GetDataType(), DT_HF8);
    ASSERT_EQ(output->GetDataType(), DT_HF8);
    auto result = FlowVerifier::VerifyResult(golden, output, 0.0f, 0.0f);
    EXPECT_TRUE(result.Check());
}

// Subnormal adjacent codes: M_v=6 vs 7 -> 2^-17 vs 2^-16, small decode gap; raw bytes differ.
TEST(FlowVerifierHf8Test, SubnormalAdjacentCodesDecodePassBytesDiffer)
{
    constexpr uint8_t kGoldenByte = 0x06;
    constexpr uint8_t kOutputByte = 0x07;
    const float kGolden = std::exp2(static_cast<float>(6 - 23));
    const float kOutput = std::exp2(static_cast<float>(7 - 23));

    auto golden = MakeHf8Scalar(kGoldenByte);
    auto output = MakeHf8Scalar(kOutputByte);
    ASSERT_EQ(golden->GetDataType(), DT_HF8);
    ASSERT_EQ(output->GetDataType(), DT_HF8);
    EXPECT_NE(golden->Get<uint8_t>(0), output->Get<uint8_t>(0));
    EXPECT_GT(std::fabs(kOutput - kGolden), 0.0f);
    EXPECT_LT(std::fabs(kOutput - kGolden), kVerifyAtol);

    auto result = FlowVerifier::VerifyResult(golden, output, kVerifyRtol, kVerifyAtol);
    EXPECT_TRUE(result.Check()) << "HF8 subnormal 0x06 vs 0x07: decode+tol should pass while storage bytes differ";
}

// D=0001 normal: 0x08 -> 1.0, 0x09 -> 1.125; decode gap exceeds zero tolerance -> verify fails.
TEST(FlowVerifierHf8Test, NormalAdjacentMismatchFailsWithZeroTol)
{
    auto golden = MakeHf8Scalar(0x08);
    auto output = MakeHf8Scalar(0x09);
    auto result = FlowVerifier::VerifyResult(golden, output, 0.0f, 0.0f);
    EXPECT_FALSE(result.Check());
}

// Regression guard for packed/physical stride usage in verifier recursion:
// view shape is [2,2] on top of raw [2,4] with offset [0,1].
// Correct row stepping must follow raw stride(0)=4 instead of logical stride(0)=2.
TEST(FlowVerifierStrideTest, PaddedRawViewUsesPhysicalStride)
{
    const std::vector<int64_t> rawShape = {2, 4};
    const std::vector<int64_t> viewShape = {2, 2};
    const std::vector<int64_t> offset = {0, 1};

    // raw index mapping for view:
    // row0 -> [1,2], row1 -> [5,6]
    const std::vector<int8_t> goldenRaw = {
        99, 10, 11, 98, // row0
        97, 12, 13, 96  // row1
    };
    const std::vector<int8_t> outputRaw = {-1, 10, 11, -2, // padding bytes intentionally differ
                                           -3, 12, 13, -4};

    auto golden = MakeInt8ViewOnPaddedRaw(goldenRaw, rawShape, viewShape, offset);
    auto output = MakeInt8ViewOnPaddedRaw(outputRaw, rawShape, viewShape, offset);

    auto result = FlowVerifier::VerifyResult(golden, output, 0.0f, 0.0f);
    EXPECT_TRUE(result.Check()) << "Verifier should compare valid view elements only with physical raw stride stepping";
}

} // namespace npu::tile_fwk
