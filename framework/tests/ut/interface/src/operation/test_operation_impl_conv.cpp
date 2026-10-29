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
 * \file test_operation_impl_conv.cpp
 * \brief Migrated conv-related test cases from test_operation_impl.cpp
 */

#include "gtest/gtest.h"
#include "interface/interpreter/calc.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/platform.h"
#include "interface/inner/tilefwk.h"
using namespace npu::tile_fwk;

class OperationImplConvTest : public testing::Test {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase() {}

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    }

    void TearDown() override {}
};

TEST_F(OperationImplConvTest, Test_Conv2d_FP16_A2A3)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_2201);
    Conv::TileL1Info l1TileShape(2, 2, 64, 64, 16, 16, 16, 1);
    Conv::TileL0Info l0TileShape(2, 64, 16, 16);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);
    TileShape::Current().SetVecTile({16, 16, 2, 16});
    Tensor fmap(DT_FP16, {1, 16, 2, 64}, "fmap");
    Tensor weight(DT_FP16, {32, 16, 3, 3}, "weight");
    Tensor result;
    Conv::ConvExtendParam convExtendParam;
    FUNCTION("TestConvA2A3")
    {
        result = npu::tile_fwk::Conv::Conv(DT_FP16, fmap, weight, {1, 1}, {1, 1, 1, 1}, {1, 1}, convExtendParam, 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplConvTest, Test_Conv2d_FP32_A2A3)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_2201);
    Conv::TileL1Info l1TileShape(2, 2, 64, 64, 8, 8, 16, 1);
    Conv::TileL0Info l0TileShape(2, 64, 8, 16);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);
    TileShape::Current().SetVecTile({16, 16, 2, 16});
    Tensor fmap(DT_FP32, {1, 8, 2, 64}, "fmap");
    Tensor weight(DT_FP32, {32, 8, 3, 3}, "weight");
    Tensor result;
    Conv::ConvExtendParam convExtendParam;
    FUNCTION("TestConvA2A3")
    {
        result = npu::tile_fwk::Conv::Conv(DT_FP32, fmap, weight, {1, 1}, {1, 1, 1, 1}, {1, 1}, convExtendParam, 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplConvTest, Test_Conv2d_BF16_Groups_A2A3)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_2201);
    Conv::TileL1Info l1TileShape(2, 2, 64, 64, 16, 16, 16, 1);
    Conv::TileL0Info l0TileShape(2, 64, 16, 16);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);
    TileShape::Current().SetVecTile({16, 16, 2, 16});
    Tensor fmap(DT_BF16, {1, 32, 2, 64}, "fmap");
    Tensor weight(DT_BF16, {32, 16, 3, 3}, "weight");
    Tensor result;
    Conv::ConvExtendParam convExtendParam;
    FUNCTION("TestConvA2A3")
    {
        result = npu::tile_fwk::Conv::Conv(DT_BF16, fmap, weight, {1, 1}, {1, 1, 1, 1}, {1, 1}, convExtendParam, 2);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplConvTest, Test_Conv1d_FP16_Bias_A2A3)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_2201);
    Conv::TileL1Info l1TileShape(1, 1, 64, 64, 16, 16, 16, 1);
    Conv::TileL0Info l0TileShape(1, 64, 16, 16);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);
    TileShape::Current().SetVecTile({16, 16, 16});
    Tensor fmap(DT_FP16, {1, 32, 64}, "fmap");
    Tensor weight(DT_FP16, {32, 32, 3}, "weight");
    Tensor bias(DT_FP16, {32}, "bias");
    Tensor result;
    Conv::ConvExtendParam convExtendParam;
    convExtendParam.biasTensor = bias;
    FUNCTION("TestConvA2A3")
    {
        result = npu::tile_fwk::Conv::Conv(DT_FP16, fmap, weight, {1}, {1, 1}, {1}, convExtendParam, 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplConvTest, Test_Conv2d_FP16_A5)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Conv::TileL1Info l1TileShape(2, 2, 64, 64, 16, 16, 16, 1);
    Conv::TileL0Info l0TileShape(2, 64, 16, 16);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);
    TileShape::Current().SetVecTile({16, 16, 2, 16});
    Tensor fmap(DT_FP16, {1, 16, 2, 64}, "fmap");
    Tensor weight(DT_FP16, {32, 16, 3, 3}, "weight");
    Tensor result;
    Conv::ConvExtendParam convExtendParam;
    FUNCTION("TestConvA5")
    {
        result = npu::tile_fwk::Conv::Conv(DT_FP16, fmap, weight, {1, 1}, {1, 1, 1, 1}, {1, 1}, convExtendParam, 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplConvTest, Test_Conv2d_FP32_A5)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Conv::TileL1Info l1TileShape(2, 2, 64, 64, 8, 8, 16, 1);
    Conv::TileL0Info l0TileShape(2, 64, 8, 16);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);
    TileShape::Current().SetVecTile({16, 16, 2, 16});
    Tensor fmap(DT_FP32, {1, 8, 2, 64}, "fmap");
    Tensor weight(DT_FP32, {32, 8, 3, 3}, "weight");
    Tensor result;
    Conv::ConvExtendParam convExtendParam;
    FUNCTION("TestConvA5")
    {
        result = npu::tile_fwk::Conv::Conv(DT_FP32, fmap, weight, {1, 1}, {1, 1, 1, 1}, {1, 1}, convExtendParam, 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplConvTest, Test_Conv2d_BF16_Groups_A5)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Conv::TileL1Info l1TileShape(2, 2, 64, 64, 16, 16, 16, 1);
    Conv::TileL0Info l0TileShape(2, 64, 16, 16);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);
    TileShape::Current().SetVecTile({16, 16, 2, 16});
    Tensor fmap(DT_BF16, {1, 32, 2, 64}, "fmap");
    Tensor weight(DT_BF16, {32, 16, 3, 3}, "weight");
    Tensor result;
    Conv::ConvExtendParam convExtendParam;
    FUNCTION("TestConvA5")
    {
        result = npu::tile_fwk::Conv::Conv(DT_BF16, fmap, weight, {1, 1}, {1, 1, 1, 1}, {1, 1}, convExtendParam, 2);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplConvTest, Test_Conv1d_FP16_Bias_A5)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Conv::TileL1Info l1TileShape(1, 1, 64, 64, 16, 16, 16, 1);
    Conv::TileL0Info l0TileShape(1, 64, 16, 16);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);
    TileShape::Current().SetVecTile({16, 16, 16});
    Tensor fmap(DT_FP16, {1, 32, 64}, "fmap");
    Tensor weight(DT_FP16, {32, 32, 3}, "weight");
    Tensor bias(DT_FP16, {32}, "bias");
    Tensor result;
    Conv::ConvExtendParam convExtendParam;
    convExtendParam.biasTensor = bias;
    FUNCTION("TestConvA5")
    {
        result = npu::tile_fwk::Conv::Conv(DT_FP16, fmap, weight, {1}, {1, 1}, {1}, convExtendParam, 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplConvTest, Test_Conv2d_FP16_SmallTile_A2A3)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_2201);
    Conv::TileL1Info l1TileShape(1, 1, 16, 16, 16, 16, 16, 1);
    Conv::TileL0Info l0TileShape(1, 16, 16, 16);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);
    TileShape::Current().SetVecTile({16, 16, 1, 16});
    Tensor fmap(DT_FP16, {1, 16, 16, 16}, "fmap");
    Tensor weight(DT_FP16, {16, 16, 3, 3}, "weight");
    Tensor result;
    Conv::ConvExtendParam convExtendParam;
    FUNCTION("TestConvA2A3")
    {
        result = npu::tile_fwk::Conv::Conv(DT_FP16, fmap, weight, {2, 2}, {0, 0, 0, 0}, {1, 1}, convExtendParam, 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplConvTest, Test_Conv2d_BF16_Stride2_A2A3)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_2201);
    Conv::TileL1Info l1TileShape(1, 1, 16, 16, 16, 16, 32, 1);
    Conv::TileL0Info l0TileShape(1, 16, 16, 32);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);
    TileShape::Current().SetVecTile({16, 16, 1, 16});
    Tensor fmap(DT_BF16, {1, 16, 16, 16}, "fmap");
    Tensor weight(DT_BF16, {32, 16, 3, 3}, "weight");
    Tensor result;
    Conv::ConvExtendParam convExtendParam;
    FUNCTION("TestConvA2A3")
    {
        result = npu::tile_fwk::Conv::Conv(DT_BF16, fmap, weight, {2, 2}, {1, 1, 1, 1}, {1, 1}, convExtendParam, 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplConvTest, Test_Conv1d_FP32_A2A3)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_2201);
    Conv::TileL1Info l1TileShape(1, 1, 16, 16, 8, 8, 16, 1);
    Conv::TileL0Info l0TileShape(1, 16, 8, 16);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);
    TileShape::Current().SetVecTile({16, 16, 16});
    Tensor fmap(DT_FP32, {1, 8, 32}, "fmap");
    Tensor weight(DT_FP32, {16, 8, 3}, "weight");
    Tensor result;
    Conv::ConvExtendParam convExtendParam;
    FUNCTION("TestConvA2A3")
    {
        result = npu::tile_fwk::Conv::Conv(DT_FP32, fmap, weight, {2}, {1, 1}, {1}, convExtendParam, 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplConvTest, Test_Conv3d_FP16_A2A3)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_2201);
    Conv::TileL1Info l1TileShape(1, 1, 16, 16, 16, 16, 16, 1);
    Conv::TileL0Info l0TileShape(1, 16, 16, 16);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);
    TileShape::Current().SetVecTile({16, 16, 1, 4, 16});
    Tensor fmap(DT_FP16, {1, 16, 2, 2, 16}, "fmap");
    Tensor weight(DT_FP16, {16, 16, 2, 3, 3}, "weight");
    Tensor result;
    Conv::ConvExtendParam convExtendParam;
    FUNCTION("TestConvA2A3")
    {
        result = npu::tile_fwk::Conv::Conv(DT_FP16, fmap, weight, {1, 1, 1}, {0, 0, 1, 1, 1, 1}, {1, 1, 1},
                                           convExtendParam, 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplConvTest, Test_Conv2d_FP16_SmallTile_A5)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Conv::TileL1Info l1TileShape(1, 1, 16, 16, 16, 16, 16, 1);
    Conv::TileL0Info l0TileShape(1, 16, 16, 16);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);
    TileShape::Current().SetVecTile({16, 16, 1, 16});
    Tensor fmap(DT_FP16, {1, 16, 16, 16}, "fmap");
    Tensor weight(DT_FP16, {16, 16, 3, 3}, "weight");
    Tensor result;
    Conv::ConvExtendParam convExtendParam;
    FUNCTION("TestConvA5")
    {
        result = npu::tile_fwk::Conv::Conv(DT_FP16, fmap, weight, {2, 2}, {0, 0, 0, 0}, {1, 1}, convExtendParam, 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplConvTest, Test_Conv2d_BF16_Stride2_A5)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Conv::TileL1Info l1TileShape(1, 1, 16, 16, 16, 16, 32, 1);
    Conv::TileL0Info l0TileShape(1, 16, 16, 32);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);
    TileShape::Current().SetVecTile({16, 16, 1, 16});
    Tensor fmap(DT_BF16, {1, 16, 16, 16}, "fmap");
    Tensor weight(DT_BF16, {32, 16, 3, 3}, "weight");
    Tensor result;
    Conv::ConvExtendParam convExtendParam;
    FUNCTION("TestConvA5")
    {
        result = npu::tile_fwk::Conv::Conv(DT_BF16, fmap, weight, {2, 2}, {1, 1, 1, 1}, {1, 1}, convExtendParam, 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplConvTest, Test_Conv1d_FP32_A5)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Conv::TileL1Info l1TileShape(1, 1, 16, 16, 8, 8, 16, 1);
    Conv::TileL0Info l0TileShape(1, 16, 8, 16);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);
    TileShape::Current().SetVecTile({16, 16, 16});
    Tensor fmap(DT_FP32, {1, 8, 32}, "fmap");
    Tensor weight(DT_FP32, {16, 8, 3}, "weight");
    Tensor result;
    Conv::ConvExtendParam convExtendParam;
    FUNCTION("TestConvA5")
    {
        result = npu::tile_fwk::Conv::Conv(DT_FP32, fmap, weight, {2}, {1, 1}, {1}, convExtendParam, 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplConvTest, Test_Conv3d_FP16_A5)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    Conv::TileL1Info l1TileShape(1, 1, 16, 16, 16, 16, 16, 1);
    Conv::TileL0Info l0TileShape(1, 16, 16, 16);
    TileShape::Current().SetConvTile(l1TileShape, l0TileShape, true);
    TileShape::Current().SetVecTile({16, 16, 1, 4, 16});
    Tensor fmap(DT_FP16, {1, 16, 2, 2, 16}, "fmap");
    Tensor weight(DT_FP16, {16, 16, 2, 3, 3}, "weight");
    Tensor result;
    Conv::ConvExtendParam convExtendParam;
    FUNCTION("TestConvA5")
    {
        result = npu::tile_fwk::Conv::Conv(DT_FP16, fmap, weight, {1, 1, 1}, {0, 0, 1, 1, 1, 1}, {1, 1, 1},
                                           convExtendParam, 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplConvTest, Test_ConvBp2d_FP16)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_2201);
    ConvBp::ConvBpTileL1Info l1TileShape(16, 144, 16);
    ConvBp::ConvBpTileL0Info l0TileShape(16, 144, 16);
    TileShape::Current().SetConvBpInputTile(l1TileShape, l0TileShape);
    TileShape::Current().SetVecTile({16, 16, 16, 16});
    Tensor gradOutput(DT_FP16, {1, 16, 14, 14}, "gradOutput");
    Tensor weight(DT_FP16, {16, 16, 3, 3}, "weight");
    Tensor result;
    ConvBp::ConvBpExtendParam convBpExtendParam;
    FUNCTION("TestConvBp")
    {
        result = ConvBp::ConvBackwardInput(DT_FP16, gradOutput, {1, 16, 32, 32}, weight, {2, 2}, {1, 1, 1, 1}, {3, 3},
                                           convBpExtendParam, 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplConvTest, Test_ConvBp2d_BF16)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_2201);
    ConvBp::ConvBpTileL1Info l1TileShape(16, 144, 16);
    ConvBp::ConvBpTileL0Info l0TileShape(16, 144, 16);
    TileShape::Current().SetConvBpInputTile(l1TileShape, l0TileShape);
    TileShape::Current().SetVecTile({16, 16, 16, 16});
    Tensor gradOutput(DT_BF16, {1, 16, 14, 14}, "gradOutput");
    Tensor weight(DT_BF16, {16, 16, 3, 3}, "weight");
    Tensor result;
    ConvBp::ConvBpExtendParam convBpExtendParam;
    FUNCTION("TestConvBp")
    {
        result = ConvBp::ConvBackwardInput(DT_BF16, gradOutput, {1, 16, 32, 32}, weight, {2, 2}, {1, 1, 1, 1}, {3, 3},
                                           convBpExtendParam, 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplConvTest, Test_ConvBp2d_FP16_Stride1)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_2201);
    ConvBp::ConvBpTileL1Info l1TileShape(16, 144, 16);
    ConvBp::ConvBpTileL0Info l0TileShape(16, 144, 16);
    TileShape::Current().SetConvBpInputTile(l1TileShape, l0TileShape);
    TileShape::Current().SetVecTile({16, 16, 16, 16});
    Tensor gradOutput(DT_FP16, {1, 16, 32, 32}, "gradOutput");
    Tensor weight(DT_FP16, {16, 16, 3, 3}, "weight");
    Tensor result;
    ConvBp::ConvBpExtendParam convBpExtendParam;
    FUNCTION("TestConvBp")
    {
        result = ConvBp::ConvBackwardInput(DT_FP16, gradOutput, {1, 16, 32, 32}, weight, {1, 1}, {1, 1, 1, 1}, {1, 1},
                                           convBpExtendParam, 1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplConvTest, Test_ConvBpFilter2d_FP16)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_2201);
    ConvBp::ConvBpTileL1Info l1TileShape(16, 80, 64);
    ConvBp::ConvBpTileL0Info l0TileShape(16, 80, 64);
    TileShape::Current().SetConvBpFilterTile(l1TileShape, l0TileShape);
    TileShape::Current().SetVecTile({16, 16, 16, 16});
    Tensor gradOutput(DT_FP16, {1, 16, 5, 16}, "gradOutput");
    Tensor input(DT_FP16, {1, 16, 4, 15}, "input");
    Tensor result;
    FUNCTION("TestConvBpFilter")
    {
        result = ConvBp::ConvBackwardFilter(DT_FP16, gradOutput, input, {16, 16, 2, 2}, {1, 1}, {1, 1, 1, 1}, {1, 1},
                                            1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplConvTest, Test_ConvBpFilter2d_BF16)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_2201);
    ConvBp::ConvBpTileL1Info l1TileShape(16, 80, 64);
    ConvBp::ConvBpTileL0Info l0TileShape(16, 80, 64);
    TileShape::Current().SetConvBpFilterTile(l1TileShape, l0TileShape);
    TileShape::Current().SetVecTile({16, 16, 16, 16});
    Tensor gradOutput(DT_BF16, {1, 16, 5, 16}, "gradOutput");
    Tensor input(DT_BF16, {1, 16, 4, 15}, "input");
    Tensor result;
    FUNCTION("TestConvBpFilter")
    {
        result = ConvBp::ConvBackwardFilter(DT_BF16, gradOutput, input, {16, 16, 2, 2}, {1, 1}, {1, 1, 1, 1}, {1, 1},
                                            1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationImplConvTest, Test_ConvBpFilter2d_FP16_Stride2)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_2201);
    ConvBp::ConvBpTileL1Info l1TileShape(16, 16, 144);
    ConvBp::ConvBpTileL0Info l0TileShape(16, 16, 16);
    TileShape::Current().SetConvBpFilterTile(l1TileShape, l0TileShape);
    TileShape::Current().SetVecTile({16, 16, 16, 16});
    Tensor gradOutput(DT_FP16, {1, 16, 4, 4}, "gradOutput");
    Tensor input(DT_FP16, {1, 16, 8, 8}, "input");
    Tensor result;
    FUNCTION("TestConvBpFilter")
    {
        result = ConvBp::ConvBackwardFilter(DT_FP16, gradOutput, input, {16, 16, 3, 3}, {2, 2}, {1, 1, 1, 1}, {1, 1},
                                            1);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}
