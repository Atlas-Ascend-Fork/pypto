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
 * \file test_interp_calc_conv.cpp
 * \brief Interpreter calc unit tests (conv golden primitives and conv format conversions).
 */

#include <vector>

#include "test_interp_calc_utils.h"

namespace npu::tile_fwk {
namespace {

constexpr int64_t FMT_ND = 0;
constexpr int64_t FMT_NC1HWC0 = 2;
constexpr int64_t FMT_FRACTAL_Z = 4;

ConvParam MakeConv2DParam(const std::vector<int64_t>& strides, const std::vector<int64_t>& paddings,
                          const std::vector<int64_t>& dilations, const std::vector<int64_t>& kernelSize,
                          int64_t fmapFormat = FMT_ND, int64_t weightFormat = FMT_ND, int64_t outFormat = FMT_ND,
                          int64_t groups = 1)
{
    ConvParam param;
    param.strides = strides;
    param.paddings = paddings;
    param.dilations = dilations;
    param.kernelSize = kernelSize;
    param.groups = groups;
    param.fmapFormat = fmapFormat;
    param.weightFormat = weightFormat;
    param.outFormat = outFormat;
    return param;
}

} // namespace

TEST_F(TorchAdaptorTest, ConvND)
{
    // fmap [1, 4, 8, 8] all 1.0, weight [8, 4, 1, 1] all 1.0 -> out = 4.0
    auto fmap = makeTensorData(DT_FP32, {1, 4, 8, 8}, 1.0f);
    auto weight = makeTensorData(DT_FP32, {8, 4, 1, 1}, 1.0f);
    auto out = makeTensorData(DT_FP32, {1, 8, 8, 8}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {1, 8, 8, 8}, 4.0f);
    calc::Conv(out, fmap, weight, nullptr, MakeConv2DParam({1, 1}, {0, 0, 0, 0}, {1, 1}, {1, 1}));
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, ConvNDKernel3)
{
    // fmap [1, 2, 8, 8] all 1.0, weight [4, 2, 3, 3] all 1.0, stride 1 -> out = 2 * 9 = 18
    auto fmap = makeTensorData(DT_FP32, {1, 2, 8, 8}, 1.0f);
    auto weight = makeTensorData(DT_FP32, {4, 2, 3, 3}, 1.0f);
    auto out = makeTensorData(DT_FP32, {1, 4, 6, 6}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {1, 4, 6, 6}, 18.0f);
    calc::Conv(out, fmap, weight, nullptr, MakeConv2DParam({1, 1}, {0, 0, 0, 0}, {1, 1}, {3, 3}));
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, ConvNDStrideDilation)
{
    // stride 2 dilation 2 kernel 3: out = (8 - 2*2 - 1) / 2 + 1 = 2, window covers 9 elements -> out = 18
    auto fmap = makeTensorData(DT_FP32, {1, 2, 8, 8}, 1.0f);
    auto weight = makeTensorData(DT_FP32, {4, 2, 3, 3}, 1.0f);
    auto out = makeTensorData(DT_FP32, {1, 4, 2, 2}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {1, 4, 2, 2}, 18.0f);
    calc::Conv(out, fmap, weight, nullptr, MakeConv2DParam({2, 2}, {0, 0, 0, 0}, {2, 2}, {3, 3}));
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, ConvNDPadding)
{
    // padding 1 stride 1 kernel 3 on 8x8: interior windows have 9 valid elements -> out = 18
    // corner windows have 4 valid elements -> out = 8
    auto fmap = makeTensorData(DT_FP32, {1, 2, 8, 8}, 1.0f);
    auto weight = makeTensorData(DT_FP32, {4, 2, 3, 3}, 1.0f);
    auto out = makeTensorData(DT_FP32, {1, 4, 8, 8}, 0.0f);
    std::vector<float> gdata(4 * 8 * 8, 18.0f);
    for (int c = 0; c < 4; c++) {
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                int validH = std::min(y + 1, 7) - std::max(y - 1, 0) + 1;
                int validW = std::min(x + 1, 7) - std::max(x - 1, 0) + 1;
                gdata[c * 64 + y * 8 + x] = 2.0f * validH * validW;
            }
        }
    }
    auto golden = makeTensorData(DT_FP32, {1, 4, 8, 8}, gdata);
    calc::Conv(out, fmap, weight, nullptr, MakeConv2DParam({1, 1}, {1, 1, 1, 1}, {1, 1}, {3, 3}));
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, ConvNDBias)
{
    auto fmap = makeTensorData(DT_FP32, {1, 4, 8, 8}, 1.0f);
    auto weight = makeTensorData(DT_FP32, {8, 4, 1, 1}, 1.0f);
    std::vector<float> biasData(8, 1.5f);
    auto bias = makeTensorData(DT_FP32, {1, 8}, biasData);
    auto out = makeTensorData(DT_FP32, {1, 8, 8, 8}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {1, 8, 8, 8}, 5.5f);
    calc::Conv(out, fmap, weight, bias, MakeConv2DParam({1, 1}, {0, 0, 0, 0}, {1, 1}, {1, 1}));
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, ConvNDGroups)
{
    // groups=2: each output channel sums 2 input channels -> out = 2
    auto fmap = makeTensorData(DT_FP32, {1, 4, 8, 8}, 1.0f);
    auto weight = makeTensorData(DT_FP32, {8, 2, 1, 1}, 1.0f);
    auto out = makeTensorData(DT_FP32, {1, 8, 8, 8}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {1, 8, 8, 8}, 2.0f);
    calc::Conv(out, fmap, weight, nullptr,
               MakeConv2DParam({1, 1}, {0, 0, 0, 0}, {1, 1}, {1, 1}, FMT_ND, FMT_ND, FMT_ND, 2));
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, ConvNDFp16)
{
    auto fmap = makeTensorData(DT_FP16, {1, 4, 8, 8}, float16(1.0));
    auto weight = makeTensorData(DT_FP16, {8, 4, 1, 1}, float16(1.0));
    auto out = makeTensorData(DT_FP16, {1, 8, 8, 8}, float16(0.0));
    auto golden = makeTensorData(DT_FP16, {1, 8, 8, 8}, float16(4.0));
    calc::Conv(out, fmap, weight, nullptr, MakeConv2DParam({1, 1}, {0, 0, 0, 0}, {1, 1}, {1, 1}));
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, ConvNDRelu)
{
    // fmap value -1.0, weight 1.0 -> raw out = -4, relu -> 0
    auto fmap = makeTensorData(DT_FP32, {1, 4, 8, 8}, -1.0f);
    auto weight = makeTensorData(DT_FP32, {8, 4, 1, 1}, 1.0f);
    auto out = makeTensorData(DT_FP32, {1, 8, 8, 8}, 1.0f);
    auto golden = makeTensorData(DT_FP32, {1, 8, 8, 8}, 0.0f);
    ConvParam param = MakeConv2DParam({1, 1}, {0, 0, 0, 0}, {1, 1}, {1, 1});
    param.reluType = 1;
    calc::Conv(out, fmap, weight, nullptr, param);
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, ConvNC1HWC0)
{
    // A2A3 style: fmap NC1HWC0 [1, 1, 2, 2, 16], weight FZ [1, 1, 16, 16], out NC1HWC0.
    // fmap channels: ch0 = [[1, 2], [3, 4]], ch1 = [[5, 6], [7, 8]] at c0 index 0 / 1.
    // weight: ch0 -> 1.0, ch1 -> 2.0. out = 1*ch0 + 2*ch1 = [[11, 14], [17, 20]].
    std::vector<float> fmapData(1 * 1 * 2 * 2 * 16, 0.0f);
    fmapData[0 * 1 * 4 * 16 + 0 * 4 * 16 + 0] = 1.0f; // [0, 0, 0, 0, 0]
    fmapData[1] = 5.0f;                               // [0, 0, 0, 0, 1]
    fmapData[16 + 0] = 2.0f;                          // [0, 0, 0, 1, 0]
    fmapData[16 + 1] = 6.0f;                          // [0, 0, 0, 1, 1]
    fmapData[32 + 0] = 3.0f;                          // [0, 0, 1, 0, 0]
    fmapData[32 + 1] = 7.0f;                          // [0, 0, 1, 0, 1]
    fmapData[48 + 0] = 4.0f;                          // [0, 0, 1, 1, 0]
    fmapData[48 + 1] = 8.0f;                          // [0, 0, 1, 1, 1]
    auto fmap = makeTensorData(DT_FP32, {1, 1, 2, 2, 16}, fmapData);
    std::vector<float> weightData(16 * 16, 0.0f);
    weightData[0] = 1.0f; // fz[0, 0, 0, 0] -> ch0 weight
    weightData[1] = 2.0f; // fz[0, 0, 0, 1] -> ch1 weight
    auto weight = makeTensorData(DT_FP32, {1, 1, 16, 16}, weightData);
    auto out = makeTensorData(DT_FP32, {1, 1, 2, 2, 16}, std::vector<float>(64, 0.0f));
    std::vector<float> gdata(64, 0.0f);
    gdata[0] = 11.0f;
    gdata[16] = 14.0f;
    gdata[32] = 17.0f;
    gdata[48] = 20.0f;
    auto golden = makeTensorData(DT_FP32, {1, 1, 2, 2, 16}, gdata);
    calc::Conv(out, fmap, weight, nullptr,
               MakeConv2DParam({1, 1}, {0, 0, 0, 0}, {1, 1}, {1, 1}, FMT_NC1HWC0, FMT_FRACTAL_Z, FMT_NC1HWC0));
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, FormatTransConvNc1hwc0)
{
    // NCHW [1, 2, 2, 2] -> NC1HWC0 -> NCHW roundtrip with distinct values.
    std::vector<float> sdata = {1, 2, 3, 4, 5, 6, 7, 8};
    auto self = makeTensorData(DT_FP32, {1, 2, 2, 2}, sdata);
    auto out5d = makeTensorData(DT_FP32, {1, 1, 2, 2, 16}, std::vector<float>(64, 0.0f));
    calc::FormatTransConv(out5d, self, FMT_ND, FMT_NC1HWC0, 1);
    // out[n=0, c1=0, h, w, c0] layout: element [n, c, h, w] at ((c1 * H + h) * W + w) * C0 + (c % C0)
    ASSERT_EQ(out5d->Get<float>(0), 1.0f); // c0 -> ch0 [0, 0]
    ASSERT_EQ(out5d->Get<float>(1), 5.0f); // c1 -> ch1 [0, 0]
    ASSERT_EQ(out5d->Get<float>(16), 2.0f);
    ASSERT_EQ(out5d->Get<float>(17), 6.0f);
    ASSERT_EQ(out5d->Get<float>(48), 4.0f);
    ASSERT_EQ(out5d->Get<float>(49), 8.0f);
    ASSERT_EQ(out5d->Get<float>(2), 0.0f); // padding channel
    auto back = makeTensorData(DT_FP32, {1, 2, 2, 2}, std::vector<float>(8, 0.0f));
    calc::FormatTransConv(back, out5d, FMT_NC1HWC0, FMT_ND, 1);
    auto golden = makeTensorData(DT_FP32, {1, 2, 2, 2}, sdata);
    ASSERT_ALLCLOSE(back, golden);
}

TEST_F(TorchAdaptorTest, FormatTransConvFractalZ)
{
    // weight NCHW [4, 2, 2, 2] -> FZ [c1 * kh * kw, n1, 16, 16] with group=1.
    std::vector<float> sdata(4 * 2 * 2 * 2);
    for (int i = 0; i < static_cast<int>(sdata.size()); i++) {
        sdata[i] = static_cast<float>(i + 1);
    }
    auto self = makeTensorData(DT_FP32, {4, 2, 2, 2}, sdata);
    auto outFz = makeTensorData(DT_FP32, {4, 1, 16, 16}, std::vector<float>(4 * 16 * 16, 0.0f));
    calc::FormatTransConv(outFz, self, FMT_ND, FMT_FRACTAL_Z, 1);
    // fz[(c1 * kh + khI) * kw + kwI, n / 16, n % 16, c % 16] = w[n, c, khI, kwI]
    // w[0, 0, 0, 0] = 1 -> fz[0, 0, 0, 0]; w[3, 1, 1, 1] = value at n=3,c=1,khI=1,kwI=1
    float expect33 = sdata[3 * (2 * 2 * 2) + 1 * (2 * 2) + 1 * 2 + 1]; // w[3,1,1,1] = 32
    ASSERT_EQ(outFz->Get<float>(0), 1.0f);
    ASSERT_EQ(outFz->Get<float>(1), sdata[4]);                    // n=0 c=1
    ASSERT_EQ(outFz->Get<float>(16), sdata[8]);                   // n=1 c=0
    ASSERT_EQ(outFz->Get<float>(3 * 256 + 3 * 16 + 1), expect33); // w[3,1,1,1] -> fz[row=3, n0=3, c0=1]
    auto back = makeTensorData(DT_FP32, {4, 2, 2, 2}, std::vector<float>(32, 0.0f));
    calc::FormatTransConv(back, outFz, FMT_FRACTAL_Z, FMT_ND, 1);
    auto golden = makeTensorData(DT_FP32, {4, 2, 2, 2}, sdata);
    ASSERT_ALLCLOSE(back, golden);
}

TEST_F(TorchAdaptorTest, ConvLoad2D)
{
    // FZ [1, 1, 16, 16] all i -> B [16, 16]: B[k, n] = fz[k / 16, n / 16, n % 16, k % 16]
    std::vector<float> l1Data(16 * 16);
    for (int i = 0; i < 256; i++) {
        l1Data[i] = static_cast<float>(i);
    }
    auto l1 = makeTensorData(DT_FP32, {1, 1, 16, 16}, l1Data);
    auto out = makeTensorData(DT_FP32, {16, 16}, std::vector<float>(256, 0.0f));
    calc::ConvLoad2D(out, l1, 0, 0, 0);
    ASSERT_EQ(out->Get<float>(0), 0.0f);
    ASSERT_EQ(out->Get<float>(1), 16.0f);  // k=0, n=1 -> fz[0, 0, 1, 0] = 0*256? no: flat idx 1*16=16
    ASSERT_EQ(out->Get<float>(16), 1.0f);  // k=1, n=0 -> fz[0, 0, 0, 1] = 1
    ASSERT_EQ(out->Get<float>(17), 17.0f); // k=1, n=1 -> fz[0, 0, 1, 1] = 1*16+1
    ASSERT_EQ(out->Get<float>(255), 255.0f);
    // postK / postN slicing
    auto out2 = makeTensorData(DT_FP32, {4, 8}, std::vector<float>(32, 0.0f));
    calc::ConvLoad2D(out2, l1, 8, 4, 0);
    // B2[k, n] = fz[(8 + k) / 16, (4 + n) / 16, (4 + n) % 16, (8 + k) % 16]
    ASSERT_EQ(out2->Get<float>(0), 72.0f);   // k=0, n=0 -> fz[0, 0, 4, 8] = 4*16 + 8
    ASSERT_EQ(out2->Get<float>(31), 187.0f); // k=3, n=7 -> fz[0, 0, 11, 11] = 11*16 + 11
}

TEST_F(TorchAdaptorTest, ConvLoad3D)
{
    // L1 fmap [1, 1, 3, 3, 16] value = h * 10 + w; load 2x2 window with 1x1 filter stride 1.
    std::vector<float> l1Data(1 * 2 * 3 * 3 * 16, 0.0f);
    for (int h = 0; h < 3; h++) {
        for (int w = 0; w < 3; w++) {
            l1Data[(h * 3 + w) * 16] = static_cast<float>(h * 10 + w);
        }
    }
    auto l1 = makeTensorData(DT_FP32, {1, 2, 3, 3, 16}, l1Data);
    auto out = makeTensorData(DT_FP32, {16, 16}, std::vector<float>(256, 0.0f));
    ConvTileParam param;
    param.strideH = 1;
    param.strideW = 1;
    param.filterH = 1;
    param.filterW = 1;
    param.validM = 4;
    param.l0CutW = 2;
    calc::ConvLoad3D(out, l1, param);
    ASSERT_EQ(out->Get<float>(0), 0.0f); // m=0 (h0,w0), k=0
    ASSERT_EQ(out->Get<float>(1), 0.0f); // m=0, k=1 -> c0i=1 -> channel 1 = 0
    ASSERT_EQ(out->Get<float>(2), 0.0f); // k=2 -> channel 2 = 0
    // flat idx 16 = (m=1, k=0) -> (h0, w1) -> l1[0, 0, 0, 1, 0] = 1
    ASSERT_EQ(out->Get<float>(16), 1.0f);
    // m=3 (h1, w1): value l1[0, 0, 1, 1, 0] = 11
    ASSERT_EQ(out->Get<float>(3 * 16), 11.0f);
    // m=2 (h1, w0): l1[0, 0, 1, 0, 0] = 10
    ASSERT_EQ(out->Get<float>(2 * 16), 10.0f);
}

TEST_F(TorchAdaptorTest, ConvLoad3DPad)
{
    // padTop shifts the window up: row = hLocal - padTop, col = wLocal
    std::vector<float> l1Data(1 * 1 * 2 * 2 * 16, 0.0f);
    for (int h = 0; h < 2; h++) {
        for (int w = 0; w < 2; w++) {
            l1Data[(h * 2 + w) * 16] = static_cast<float>(h * 10 + w);
        }
    }
    l1Data[0] = 5.0f;
    auto l1 = makeTensorData(DT_FP32, {1, 1, 2, 2, 16}, l1Data);
    auto out = makeTensorData(DT_FP32, {16, 16}, std::vector<float>(256, 0.0f));
    ConvTileParam param;
    param.strideH = 1;
    param.strideW = 1;
    param.filterH = 1;
    param.filterW = 1;
    param.validM = 4;
    param.l0CutW = 2;
    param.l0HOffset = 0;
    param.l0WOffset = 0;
    param.padTop = 1;
    param.padLeft = 0;
    calc::ConvLoad3D(out, l1, param);
    // m=0: (h0,w0) -> row -1 (pad) -> 0
    ASSERT_EQ(out->Get<float>(0), 0.0f);
    // m=1: (h0,w1) -> row -1 (pad) -> 0
    ASSERT_EQ(out->Get<float>(16), 0.0f);
    // m=2: (h1,w0) -> row 0, col 0 -> l1[0, 0, 0, 0, 0] = 5
    ASSERT_EQ(out->Get<float>(2 * 16), 5.0f);
    // m=3: (h1,w1) -> row 0, col 1 -> l1[0, 0, 0, 1, 0] = 1
    ASSERT_EQ(out->Get<float>(3 * 16), 1.0f);
}

TEST_F(TorchAdaptorTest, ConvTransL0C)
{
    // l0c tile holds valid data only: [4, 8], cutW = 2, NZ2DN -> out [1, 8, 2, 2]
    std::vector<float> cData(4 * 8, 0.0f);
    for (int m = 0; m < 4; m++) {
        for (int n = 0; n < 8; n++) {
            cData[m * 8 + n] = static_cast<float>(m * 10 + n);
        }
    }
    auto l0c = makeTensorData(DT_FP32, {4, 8}, cData);
    auto out = makeTensorData(DT_FP32, {1, 8, 2, 2}, std::vector<float>(32, 0.0f));
    ConvL0CParam param;
    param.copyOutMode = 3;
    param.cutW = 2;
    calc::ConvTransL0C(out, l0c, param);
    // out[0, n, h, w] = l0c[h * cutW + w, n]
    ASSERT_EQ(out->Get<float>(0), 0.0f);  // n=0 h=0 w=0 -> m=0
    ASSERT_EQ(out->Get<float>(1), 10.0f); // n=0 h=0 w=1 -> m=1
    ASSERT_EQ(out->Get<float>(4), 1.0f);  // n=1 h=0 w=0 -> l0c[0, 1]
    ASSERT_EQ(out->Get<float>(8), 2.0f);  // n=2 h=0 w=0 -> l0c[0, 2]
    ASSERT_EQ(out->Get<float>(2), 20.0f); // n=0 h=1 w=0 -> m=2
    ASSERT_EQ(out->Get<float>(16), 4.0f); // n=4 h=0 w=0 -> l0c[0, 4]
}

TEST_F(TorchAdaptorTest, ConvTransL0CNz2Nz)
{
    // l0c [16, 16] valid {4, 4}, cutW = 2, NZ2NZ -> out [1, 1, 2, 2, 16]
    std::vector<float> cData(16 * 32, 0.0f);
    for (int m = 0; m < 4; m++) {
        for (int n = 0; n < 4; n++) {
            cData[m * 32 + n] = static_cast<float>(m * 10 + n);
        }
    }
    auto l0c = makeTensorData(DT_FP32, {16, 32}, cData);
    auto out = makeTensorData(DT_FP32, {1, 2, 2, 2, 16}, std::vector<float>(128, 0.0f));
    ConvL0CParam param;
    param.copyOutMode = 1;
    param.cutW = 2;
    param.validN = 4;
    param.validH = 2;
    param.validW = 2;
    calc::ConvTransL0C(out, l0c, param);
    // out[0, c1, h, w, c0] = l0c[h * cutW + w, c1 * 16 + c0]
    ASSERT_EQ(out->Get<float>(0), 0.0f);   // c1=0 c0=0 h=0 w=0 -> m=0 n=0
    ASSERT_EQ(out->Get<float>(1), 1.0f);   // c0=1 -> n=1
    ASSERT_EQ(out->Get<float>(16), 10.0f); // h=0 w=1 -> m=1
    ASSERT_EQ(out->Get<float>(3), 3.0f);   // n=3 valid -> l0c[0, 3] = 3
    // channel block 1 (n >= 16) -> 0; w=1 -> m=1
    ASSERT_EQ(out->Get<float>(64), 0.0f);
    ASSERT_EQ(out->Get<float>(16), 10.0f);
    ASSERT_EQ(out->Get<float>(32), 20.0f);
    ASSERT_EQ(out->Get<float>(48), 30.0f);
}

TEST_F(TorchAdaptorTest, ConvTransL0CNz2NzWTailTile)
{
    // W tail tile (validW < cutW): l0c raw [6, 4] with M = h(2) * cutW(3), valid rows are
    // h * cutW + w for w < 2 (non-contiguous: 0, 1, 3, 4). A squashed validM view ([4, 4])
    // both misplaces rows and overflows the h * cutW + w mapping (row 4 >= 4).
    std::vector<float> cData(6 * 4, 0.0f);
    for (int m = 0; m < 6; m++) {
        for (int n = 0; n < 4; n++) {
            cData[m * 4 + n] = static_cast<float>(m * 10 + n);
        }
    }
    auto l0c = makeTensorData(DT_FP32, {6, 4}, cData);
    auto out = makeTensorData(DT_FP32, {1, 1, 2, 2, 4}, std::vector<float>(16, 0.0f));
    ConvL0CParam param;
    param.copyOutMode = 1;
    param.cutW = 3;
    param.validN = 4;
    param.validH = 2;
    param.validW = 2;
    calc::ConvTransL0C(out, l0c, param);
    // out[0, c1, h, w, c0] = l0c[h * cutW + w, c1 * 4 + c0] = (h * 3 + w) * 10 + c0
    ASSERT_EQ(out->Get<float>(0), 0.0f);   // h=0 w=0 -> m=0
    ASSERT_EQ(out->Get<float>(1), 1.0f);   // c0=1 -> n=1
    ASSERT_EQ(out->Get<float>(4), 10.0f);  // h=0 w=1 -> m=1
    ASSERT_EQ(out->Get<float>(8), 30.0f);  // h=1 w=0 -> m=3 (raw row 1 * 3 + 0, not 2)
    ASSERT_EQ(out->Get<float>(9), 31.0f);  // c0=1 -> n=1
    ASSERT_EQ(out->Get<float>(12), 40.0f); // h=1 w=1 -> m=4
}

TEST_F(TorchAdaptorTest, ConvTransL0CNz2NzValidNPad)
{
    // validN < aligned N: columns at or beyond validN are aligned padding holding stale
    // storage data (filled with markers here) and must read as zeros
    std::vector<float> cData(4 * 8, 0.0f);
    for (int m = 0; m < 4; m++) {
        for (int n = 0; n < 8; n++) {
            cData[m * 8 + n] = static_cast<float>(m * 10 + n);
        }
    }
    auto l0c = makeTensorData(DT_FP32, {4, 8}, cData);
    auto out = makeTensorData(DT_FP32, {1, 2, 2, 2, 4}, std::vector<float>(32, 0.0f));
    ConvL0CParam param;
    param.copyOutMode = 1;
    param.cutW = 2;
    param.validN = 5;
    param.validH = 2;
    param.validW = 2;
    calc::ConvTransL0C(out, l0c, param);
    // out[0, c1, h, w, c0] = l0c[h * 2 + w, c1 * 4 + c0], zero for n >= validN(5)
    ASSERT_EQ(out->Get<float>(0), 0.0f);   // h=0 w=0 c1=0 c0=0 -> m=0 n=0
    ASSERT_EQ(out->Get<float>(4), 10.0f);  // h=0 w=1 -> m=1
    ASSERT_EQ(out->Get<float>(8), 20.0f);  // h=1 w=0 -> m=2
    ASSERT_EQ(out->Get<float>(16), 4.0f);  // c1=1 c0=0 -> n=4 < validN
    ASSERT_EQ(out->Get<float>(20), 14.0f); // h=0 w=1 -> m=1 n=4
    ASSERT_EQ(out->Get<float>(17), 0.0f);  // n=5 >= validN -> 0 (raw holds 5)
    ASSERT_EQ(out->Get<float>(19), 0.0f);  // n=7 -> 0
    ASSERT_EQ(out->Get<float>(31), 0.0f);  // n=7 -> 0
}

TEST_F(TorchAdaptorTest, ConvTransL0CNz2DNWTailTile)
{
    // NZ2DN W tail tile (REAL_CUT_W < cutW): l0c raw [8, 8] with M = h(2) * cutW(4), valid rows
    // are h * cutW + w for w < validW(3) (non-contiguous: 0, 1, 2, 4, 5, 6). The dst view is the
    // valid extent [1, 8, 2, 3] (W = 3 < cutW = 4): the copy width must shrink to the dst extent,
    // otherwise copy_ fails with "The size of tensor a (3) must match the size of tensor b (4)".
    std::vector<float> cData(8 * 8, 0.0f);
    for (int m = 0; m < 8; m++) {
        for (int n = 0; n < 8; n++) {
            if (n >= 6) {
                cData[m * 8 + n] = 99.0f; // aligned N pad columns (stale storage data)
            } else if ((m % 4) >= 3) {
                cData[m * 8 + n] = 77.0f; // invalid w rows (w >= validW), never read
            } else {
                cData[m * 8 + n] = static_cast<float>(m * 10 + n);
            }
        }
    }
    auto l0c = makeTensorData(DT_FP32, {8, 8}, cData);
    auto out = makeTensorData(DT_FP32, {1, 8, 2, 3}, std::vector<float>(48, 0.0f));
    ConvL0CParam param;
    param.copyOutMode = 3;
    param.cutW = 4;
    param.validN = 6;
    param.validH = 2;
    // note: validW is not consumed by the NZ2DN path; the copy width is cutW shrunk to the
    // dst view extent (W = 3 here), which is what this test locks in
    calc::ConvTransL0C(out, l0c, param);
    // out[0, n, h, w] = l0c[h * cutW + w, n] = (h * 4 + w) * 10 + n, zero for n >= validN(6);
    // flat idx = n * 6 + h * 3 + w
    ASSERT_EQ(out->Get<float>(0), 0.0f);   // n=0 h=0 w=0 -> m=0
    ASSERT_EQ(out->Get<float>(1), 10.0f);  // n=0 h=0 w=1 -> m=1
    ASSERT_EQ(out->Get<float>(2), 20.0f);  // n=0 h=0 w=2 -> m=2
    ASSERT_EQ(out->Get<float>(3), 40.0f);  // n=0 h=1 w=0 -> m=4 (row stride cutW=4, not validW=3)
    ASSERT_EQ(out->Get<float>(4), 50.0f);  // n=0 h=1 w=1 -> m=5
    ASSERT_EQ(out->Get<float>(5), 60.0f);  // n=0 h=1 w=2 -> m=6
    ASSERT_EQ(out->Get<float>(6), 1.0f);   // n=1 h=0 w=0 -> l0c[0, 1]
    ASSERT_EQ(out->Get<float>(25), 14.0f); // n=4 h=0 w=1 -> l0c[1, 4]
    ASSERT_EQ(out->Get<float>(36), 0.0f);  // n=6 >= validN -> 0 (raw holds 99)
    ASSERT_EQ(out->Get<float>(47), 0.0f);  // n=7 h=1 w=2 -> 0 (raw holds 99)
}

TEST_F(TorchAdaptorTest, ConvFmapND2NzWTail)
{
    // W tail fmap tile (DN2NZ copy-in): src GM NCHW [1, 16, 1, 4] (valid W = 4, clamped by the
    // remaining GM extent), dst L1 NZ tile [1, 1, 1, 16, 16]. The src W gap must be zero-padded
    // to the dst extent (hardware pads out-of-range GM reads with 0); before the fix the
    // internal view() failed on the shape mismatch.
    std::vector<float> srcData(16 * 4, 0.0f);
    for (int c = 0; c < 16; c++) {
        for (int w = 0; w < 4; w++) {
            srcData[c * 4 + w] = static_cast<float>(c * 10 + w);
        }
    }
    auto src = makeTensorData(DT_FP32, {1, 16, 1, 4}, srcData);
    auto out = makeTensorData(DT_FP32, {1, 1, 1, 16, 16}, std::vector<float>(256, 0.0f));
    calc::ConvFmapND2NZ(out, src, 0);
    // out[0, 0, 0, w, c0] = src[0, c0, 0, w] = c0 * 10 + w for w < 4, zero for w >= 4;
    // flat idx = w * 16 + c0
    ASSERT_EQ(out->Get<float>(0), 0.0f);    // w=0 c0=0
    ASSERT_EQ(out->Get<float>(1), 10.0f);   // w=0 c0=1
    ASSERT_EQ(out->Get<float>(15), 150.0f); // w=0 c0=15
    ASSERT_EQ(out->Get<float>(16), 1.0f);   // w=1 c0=0
    ASSERT_EQ(out->Get<float>(18), 21.0f);  // w=1 c0=2
    ASSERT_EQ(out->Get<float>(63), 153.0f); // w=3 c0=15
    ASSERT_EQ(out->Get<float>(64), 0.0f);   // w=4 >= valid W -> 0 (pad)
    ASSERT_EQ(out->Get<float>(255), 0.0f);  // w=15 c0=15 -> 0 (pad)
}

TEST_F(TorchAdaptorTest, ConvFmapND2NzBatchTail)
{
    // Batch tail + W tail combined (DN2NZ copy-in): src GM NCHW [13, 16, 1, 4] where the
    // runtime batch (13) is smaller than the aligned dst tile N (16) and W is a tail (4 < 16).
    // Both gaps must be zero-padded to the dst extent before view(); before the N pad was added
    // the internal view() failed with an element-count mismatch (same FB200F family as issue 3637).
    std::vector<float> srcData(13 * 16 * 4, 0.0f);
    for (int n = 0; n < 13; n++) {
        for (int c = 0; c < 16; c++) {
            for (int w = 0; w < 4; w++) {
                srcData[(n * 16 + c) * 4 + w] = static_cast<float>(c * 10 + w);
            }
        }
    }
    auto src = makeTensorData(DT_FP32, {13, 16, 1, 4}, srcData);
    auto out = makeTensorData(DT_FP32, {16, 1, 1, 16, 16}, std::vector<float>(4096, 0.0f));
    calc::ConvFmapND2NZ(out, src, 0);
    // out[n, 0, 0, w, c0] = src[n, c0, 0, w] = c0 * 10 + w for n < 13 and w < 4, zero otherwise;
    // flat idx = n * 256 + w * 16 + c0
    ASSERT_EQ(out->Get<float>(0), 0.0f);                      // n=0 w=0 c0=0
    ASSERT_EQ(out->Get<float>(1), 10.0f);                     // n=0 w=0 c0=1
    ASSERT_EQ(out->Get<float>(3), 30.0f);                     // n=0 w=0 c0=3
    ASSERT_EQ(out->Get<float>(16), 1.0f);                     // n=0 w=1 c0=0
    ASSERT_EQ(out->Get<float>(12 * 256 + 3 * 16 + 5), 53.0f); // n=12 w=3 c0=5
    ASSERT_EQ(out->Get<float>(12 * 256 + 15), 150.0f);        // n=12 w=0 c0=15
    ASSERT_EQ(out->Get<float>(13 * 256), 0.0f);               // n=13 >= runtime batch -> 0 (pad)
    ASSERT_EQ(out->Get<float>(15 * 256 + 255), 0.0f);         // n=15 w=15 c0=15 -> 0 (pad)
    ASSERT_EQ(out->Get<float>(5 * 256 + 5 * 16), 0.0f);       // n=5 w=5 >= valid W -> 0 (pad)
}

TEST_F(TorchAdaptorTest, ConvND3D)
{
    // conv3d: fmap [1, 2, 2, 4, 4] all 1.0, weight [4, 2, 1, 1, 1] all 1.0 -> out = 2
    auto fmap = makeTensorData(DT_FP32, {1, 2, 2, 4, 4}, 1.0f);
    auto weight = makeTensorData(DT_FP32, {4, 2, 1, 1, 1}, 1.0f);
    auto out = makeTensorData(DT_FP32, {1, 4, 2, 4, 4}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {1, 4, 2, 4, 4}, 2.0f);
    ConvParam param;
    param.strides = {1, 1, 1};
    param.paddings = {0, 0, 0, 0, 0, 0};
    param.dilations = {1, 1, 1};
    param.kernelSize = {1, 1, 1};
    param.groups = 1;
    param.isConv3D = 1;
    calc::Conv(out, fmap, weight, nullptr, param);
    ASSERT_ALLCLOSE(out, golden);
}

TEST_F(TorchAdaptorTest, ConvBiasPadDebug)
{
    // A5 A_MUL_B contract: bias cout must equal matmul N (N0-aligned): a[128,16] fp16 x b[16,16] fp16 + bias fp32
    // [1,16] -> out fp32 [128,16]
    auto a = makeTensorData(DT_FP16, {128, 16}, float16(1.0));
    auto b = makeTensorData(DT_FP16, {16, 16}, float16(1.0));
    auto bias = makeTensorData(DT_FP32, {1, 16}, 1.0f);
    auto out = makeTensorData(DT_FP32, {128, 16}, 0.0f);
    auto golden = makeTensorData(DT_FP32, {128, 16}, 17.0f);
    auto biasData = calc::Trans(bias);
    CalcOps* ops = calc::GetCalcOps();
    auto aData = calc::Trans(a);
    auto bData = calc::Trans(b);
    auto outData = calc::Trans(out);
    MatMulParam param2{};
    param2.aTrans = false;
    param2.bTrans = false;
    param2.biasPtr = &biasData;
    ops->MatMul(outData, aData, bData, nullptr, param2);
    ASSERT_ALLCLOSE(out, golden);
}
} // namespace npu::tile_fwk
