/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <limits>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>

#include "interface/interpreter/calculator/calc_api.h"

namespace npu::tile_fwk {
namespace {

TEST(NanScalarTest, PreservesInfiniteScalar)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    std::vector<float> input = {nan, -inf, 1.0f, inf};
    std::vector<float> output(input.size());
    TensorData src{input.data(), {4}, {4}, {1}, 0, DT_FP32};
    TensorData dst{output.data(), {4}, {4}, {1}, 0, DT_FP32};
    for (bool maximum : {false, true}) {
        for (float scalar : {-inf, inf}) {
            SCOPED_TRACE(maximum);
            SCOPED_TRACE(scalar);
            GetCalcOps()->NanScalar(dst, src, Element(DT_FP32, scalar), maximum);
            for (size_t i = 0; i < input.size(); ++i) {
                EXPECT_EQ(output[i], maximum ? std::fmax(input[i], scalar) : std::fmin(input[i], scalar));
            }
        }
    }
}

class NanReduceTest : public testing::TestWithParam<std::tuple<DataType, int, bool>> {};

TEST_P(NanReduceTest, SignedZeroAndNan)
{
    const auto [dtype, axis, maximum] = GetParam();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    // Both mixed-zero orders, same-sign zeros, NaNs, finite values and infinities.
    std::vector<float> input = {-0.0f, 0.0f, nan,   0.0f, -0.0f, nan,   -0.0f, -0.0f, nan,   0.0f,
                                0.0f,  nan,  nan,   nan,  nan,   -2.0f, 3.0f,  nan,   -inf,  inf,
                                nan,   nan,  -0.0f, nan,  -2.0f, 0.0f,  nan,   2.0f,  -0.0f, nan};
    constexpr int64_t rows = 10;
    constexpr int64_t cols = 3;
    std::vector<uint8_t> storage(input.size() * BytesOf(dtype));
    TensorData fp32{input.data(), {rows, cols}, {rows, cols}, {cols, 1}, 0, DT_FP32};
    TensorData src{storage.data(), {rows, cols}, {rows, cols}, {cols, 1}, 0, dtype};
    auto* ops = GetCalcOps();
    ops->Cast(src, fp32, CastMode::CAST_NONE);
    if (axis == 0) {
        // Reduce a strided column view of the same data.
        src.shape = {cols, rows};
        src.stride = {1, cols};
    }
    const std::vector<int64_t> shape = axis == 0 ? std::vector<int64_t>{1, rows} : std::vector<int64_t>{rows, 1};
    const std::vector<int64_t> stride = axis == 0 ? std::vector<int64_t>{rows, 1} : std::vector<int64_t>{1, 1};
    std::vector<uint8_t> output(rows * BytesOf(dtype));
    TensorData dst{output.data(), shape, shape, stride, 0, dtype};
    ops->NanReduce(dst, {}, src, axis, maximum, 0, true);
    std::vector<float> actual(rows);
    TensorData result{actual.data(), shape, shape, stride, 0, DT_FP32};
    ops->Cast(result, dst, CastMode::CAST_NONE);
    const std::vector<float> expected = {maximum ? 0.0f : -0.0f,
                                         maximum ? 0.0f : -0.0f,
                                         -0.0f,
                                         0.0f,
                                         nan,
                                         maximum ? 3.0f : -2.0f,
                                         maximum ? inf : -inf,
                                         -0.0f,
                                         maximum ? 0.0f : -2.0f,
                                         maximum ? 2.0f : -0.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        SCOPED_TRACE(i);
        if (std::isnan(expected[i])) {
            EXPECT_TRUE(std::isnan(actual[i]));
        } else {
            EXPECT_EQ(actual[i], expected[i]);
            EXPECT_EQ(std::signbit(actual[i]), std::signbit(expected[i]));
        }
    }
    // Signed-zero value correction must not change the first equal arg index.
    std::vector<int32_t> indices(rows);
    TensorData index{indices.data(), shape, shape, stride, 0, DT_INT32};
    ops->NanReduce({}, index, src, axis, maximum, 0, true);
    const std::vector<int32_t> expectedIndices = {
        0, 0, 0, 0, 0, maximum ? 1 : 0, maximum ? 1 : 0, 1, maximum ? 1 : 0, maximum ? 0 : 1};
    EXPECT_EQ(indices, expectedIndices);
}

INSTANTIATE_TEST_SUITE_P(Extrema, NanReduceTest,
                         testing::Combine(testing::Values(DT_FP16, DT_FP32, DT_BF16), testing::Values(0, 1),
                                          testing::Bool()));

TEST(NanBinaryTest, BroadcastAndUnequalPairTails)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    std::vector<float> a = {nan, -inf, 3, nan, 4, inf, nan, nan};
    std::vector<float> b = {2, nan, 5, nan};
    std::vector<float> output(10, -777);
    TensorData lhs{a.data(), {2, 4}, {2, 4}, {4, 1}, 0, DT_FP32};
    TensorData rhs{b.data(), {1, 4}, {1, 4}, {4, 1}, 0, DT_FP32};
    TensorData dst{output.data() + 1, {2, 4}, {2, 4}, {4, 1}, 0, DT_FP32};
    for (const bool maximum : {false, true}) {
        for (const bool pair : {false, true}) {
            for (const bool reverse : {false, true}) {
                GetCalcOps()->NanBinary(dst, reverse ? rhs : lhs, reverse ? lhs : rhs, maximum, pair);
                for (size_t i = 0; i < a.size(); ++i) {
                    const float expected = pair && i >= b.size() ?
                                               a[i] :
                                               (maximum ? std::fmax(a[i], b[i % 4]) : std::fmin(a[i], b[i % 4]));
                    if (std::isnan(expected)) {
                        EXPECT_TRUE(std::isnan(output[i + 1]));
                    } else {
                        EXPECT_EQ(output[i + 1], expected);
                    }
                }
                EXPECT_EQ(output.front(), -777);
                EXPECT_EQ(output.back(), -777);
            }
        }
    }
}

TEST(NanPairArgTest, InvalidPartialsAndFirstEqualIndex)
{
    const float inf = std::numeric_limits<float>::infinity();
    const auto invalid = std::numeric_limits<int32_t>::max();
    std::vector<float> a = {5, 1, 9, 2, -inf, inf}, b = {5, 3, 2, 100, inf, -inf}, values(6);
    std::vector<int32_t> ai = {7, 9, invalid, 3, invalid, 7}, bi = {2, 4, 8, invalid, invalid, invalid}, indices(6);
    TensorData lhs{a.data(), {1, 6}, {1, 6}, {6, 1}, 0, DT_FP32};
    TensorData rhs{b.data(), {1, 6}, {1, 6}, {6, 1}, 0, DT_FP32};
    TensorData leftIndex{ai.data(), {1, 6}, {1, 6}, {6, 1}, 0, DT_INT32};
    TensorData rightIndex{bi.data(), {1, 6}, {1, 6}, {6, 1}, 0, DT_INT32};
    TensorData value{values.data(), {1, 6}, {1, 6}, {6, 1}, 0, DT_FP32};
    TensorData index{indices.data(), {1, 6}, {1, 6}, {6, 1}, 0, DT_INT32};
    for (const bool maximum : {false, true}) {
        for (const bool finalize : {false, true}) {
            GetCalcOps()->NanPairArg(value, index, lhs, leftIndex, rhs, rightIndex, maximum, finalize);
            EXPECT_EQ(indices, (std::vector<int32_t>{2, maximum ? 4 : 9, 8, 3, finalize ? 0 : invalid, 7}));
            EXPECT_EQ(values, (std::vector<float>{5, maximum ? 3.0f : 1.0f, 2, 2, -inf, inf}));
        }
    }
}

TEST(NanReduceBoundaryTest, EmptyPartialAndIndexOffset)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    float unused = 0;
    std::vector<float> values(2);
    std::vector<int32_t> indices(2);
    TensorData empty{&unused, {2, 0}, {2, 0}, {1, 1}, 0, DT_FP32};
    TensorData value{values.data(), {2, 1}, {2, 1}, {1, 1}, 0, DT_FP32};
    TensorData index{indices.data(), {2, 1}, {2, 1}, {1, 1}, 0, DT_INT32};
    for (const bool maximum : {false, true}) {
        GetCalcOps()->NanReduce(value, {}, empty, 1, maximum, 32, true);
        EXPECT_TRUE(std::isnan(values[0]));
        EXPECT_TRUE(std::isnan(values[1]));
        for (const bool finalize : {false, true}) {
            GetCalcOps()->NanReduce(value, index, empty, 1, maximum, 32, finalize);
            EXPECT_EQ(indices, (std::vector<int32_t>(2, finalize ? 0 : std::numeric_limits<int32_t>::max())));
            EXPECT_EQ(values, (std::vector<float>(2, maximum ? -inf : inf)));
            std::vector<float> data = {nan, maximum ? -inf : inf, maximum ? -inf : inf, nan, nan, nan};
            TensorData src{data.data(), {2, 3}, {2, 3}, {3, 1}, 0, DT_FP32};
            GetCalcOps()->NanReduce(value, index, src, 1, maximum, 32, finalize);
            EXPECT_EQ(indices[0], 33);
            EXPECT_EQ(indices[1], finalize ? 0 : std::numeric_limits<int32_t>::max());
        }
    }
}

TEST(NanScalarTest, IntegerScalarIgnoresNan)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> input = {nan, 1, 3}, output(3);
    TensorData src{input.data(), {3}, {3}, {1}, 0, DT_FP32};
    TensorData dst{output.data(), {3}, {3}, {1}, 0, DT_FP32};
    for (const bool maximum : {false, true}) {
        GetCalcOps()->NanScalar(dst, src, Element(DT_INT32, 2), maximum);
        EXPECT_EQ(output, (maximum ? std::vector<float>{2, 2, 3} : std::vector<float>{2, 1, 2}));
    }
}

} // namespace
} // namespace npu::tile_fwk
