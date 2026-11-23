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
 * \file test_operation.cpp
 */

#include "gtest/gtest.h"
#include <algorithm>
#include <memory>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "interface/operation/operation_impl.h"
#include "passes/tensor_graph_pass/expand_function.h"
#include "interface/program/program.h"
#include "tilefwk/hif4_scratch.h"
#include "tilefwk/data_type.h"
#include "tilefwk/platform.h"

using namespace npu::tile_fwk;

class OperationOpsTest : public testing::Test {
public:
    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
    }
    void TearDown() override {}
};

class OperationTilingTest : public OperationOpsTest {
protected:
    void SetUp() override
    {
        OperationOpsTest::SetUp();
        config::SetHostOption(COMPILE_STAGE, CS_TENSOR_GRAPH);
    }

    std::shared_ptr<Function> Expand(Operation& root, const TileShape& tileShape)
    {
        auto tiled = std::make_shared<Function>(Program::GetInstance(), "TiledOperation", "TiledOperation", nullptr);
        tiled->paramConfigs_.combineAxis = false;
        // The temporary graph must not attach operations to the enclosing FUNCTION's tensors.
        LogicalTensors inputs;
        for (const auto& input : root.GetIOperands()) {
            auto clone = std::make_shared<LogicalTensor>(*tiled, input->Datatype(), input->shape,
                                                         input->GetDynValidShape(), input->Format());
            inputs.push_back(clone);
            tiled->AddOriginIncast(clone);
        }
        LogicalTensors outputs;
        for (const auto& output : root.GetOOperands()) {
            auto clone = std::make_shared<LogicalTensor>(*tiled, output->Datatype(), output->shape,
                                                         output->GetDynValidShape(), output->Format());
            outputs.push_back(clone);
            tiled->AddOriginOutcast(clone);
        }
        ExpandFunction::CurrentTileOpGuard guard(root);
        ExpandOperationInto(*tiled, tileShape, root.GetOpcode(), inputs, outputs, root);
        for (const auto& input : root.GetIOperands()) {
            for (const auto* consumer : input->GetConsumers()) {
                EXPECT_NE(consumer->BelongTo(), tiled.get());
            }
        }
        for (const auto& output : root.GetOOperands()) {
            for (const auto* producer : output->GetProducers()) {
                EXPECT_NE(producer->BelongTo(), tiled.get());
            }
        }
        return tiled;
    }

    // AddOperation moves parent coordinates into View/Assemble attributes and resets the local Tile offset.
    void ExpectViewOffset(const LogicalTensorPtr& tile, const Offset& expected)
    {
        const auto& producers = tile->GetProducers();
        ASSERT_EQ(producers.size(), 1U);
        auto attr = std::dynamic_pointer_cast<ViewOpAttribute>((*producers.begin())->GetOpAttribute());
        ASSERT_NE(attr, nullptr);
        EXPECT_EQ(attr->GetFromOffset(), expected);
    }

    void ExpectAssembleOffset(const LogicalTensorPtr& tile, const Offset& expected)
    {
        const auto& consumers = tile->GetConsumers();
        ASSERT_EQ(consumers.size(), 1U);
        auto attr = std::dynamic_pointer_cast<AssembleOpAttribute>((*consumers.begin())->GetOpAttribute());
        ASSERT_NE(attr, nullptr);
        EXPECT_EQ(attr->GetToOffset(), expected);
    }
};

TEST_F(OperationTilingTest, MaxpoolPreservesPaddingAndTailTiles)
{
    Tensor input(DT_FP16, {1, 1, 5, 7, 16}, "input");
    FUNCTION("MaxpoolPaddingAndTail", {input})
    {
        auto output = Maxpool(input, {2, 2}, {1, 1}, {1, 0, 0, 1});
        EXPECT_EQ(output.GetShape(), (Shape{1, 1, 5, 7, 16}));
        EXPECT_EQ(output.GetDataType(), DT_FP16);
        auto& root = Program::GetInstance().GetCurrentFunction()->Operations(false).back();
        ASSERT_EQ(root.GetOpcode(), Opcode::OP_MAX_POOL);
        EXPECT_EQ(root.GetIntAttribute(ConvOpAttributeKey::paddingLeft), 1);
        EXPECT_EQ(root.GetIntAttribute(ConvOpAttributeKey::paddingTop), 0);
        EXPECT_EQ(root.GetIntAttribute(ConvOpAttributeKey::paddingRight), 0);
        EXPECT_EQ(root.GetIntAttribute(ConvOpAttributeKey::paddingBottom), 1);
        EXPECT_EQ(root.GetIntAttribute(ConvOpAttributeKey::strideh), 1);
        EXPECT_EQ(root.GetIntAttribute(ConvOpAttributeKey::stridew), 1);
        EXPECT_EQ(root.GetIntAttribute(PoolOpAttributeKey::poolh), 2);
        EXPECT_EQ(root.GetIntAttribute(PoolOpAttributeKey::poolw), 2);

        TileShape tile;
        tile.SetVecTile(3, 4);
        auto tiled = Expand(root, tile);
        size_t poolCount = 0;
        for (const auto& op : tiled->Operations(false)) {
            if (op.GetOpcode() != Opcode::OP_MAX_POOL) {
                continue;
            }
            const int64_t row = static_cast<int64_t>(poolCount / 2) * 3;
            const int64_t col = static_cast<int64_t>(poolCount % 2) * 4;
            ExpectAssembleOffset(op.GetOutputOperand(0), (Offset{0, 0, row, col, 0}));
            EXPECT_EQ(op.GetOutputOperand(0)->shape,
                      (Shape{1, 1, std::min<int64_t>(3, 5 - row), std::min<int64_t>(4, 7 - col), 16}));
            ExpectViewOffset(op.GetInputOperand(0), (Offset{0, 0, row, col == 0 ? 0 : col - 1, 0}));
            EXPECT_EQ(op.GetInputOperand(0)->shape, (Shape{1, 1, row == 0 ? 4 : 2, 4, 16}));
            EXPECT_EQ(op.GetIntAttribute(ConvOpAttributeKey::paddingLeft), col == 0 ? 1 : 0);
            EXPECT_EQ(op.GetIntAttribute(ConvOpAttributeKey::paddingBottom), row == 0 ? 0 : 1);
            EXPECT_EQ(op.GetIntAttribute(PoolOpAttributeKey::poolh), 2);
            EXPECT_EQ(op.GetIntAttribute(PoolOpAttributeKey::poolw), 2);
            ++poolCount;
        }
        EXPECT_EQ(poolCount, 4U);
    }
}

TEST_F(OperationTilingTest, UnitMaxpoolLowersToCopies)
{
    Tensor input(DT_FP16, {1, 1, 3, 5, 16}, "input");
    FUNCTION("UnitMaxpoolCopies", {input})
    {
        auto output = Maxpool(input, {1, 1}, {1, 1}, {0, 0, 0, 0});
        EXPECT_EQ(output.GetShape(), input.GetShape());
        auto& root = Program::GetInstance().GetCurrentFunction()->Operations(false).back();
        TileShape tile;
        tile.SetVecTile(2, 3);
        auto tiled = Expand(root, tile);
        size_t copyCount = 0;
        for (const auto& op : tiled->Operations(false)) {
            EXPECT_NE(op.GetOpcode(), Opcode::OP_MAX_POOL);
            if (op.GetOpcode() == Opcode::OP_COPY_UB_TO_UB) {
                const int64_t row = static_cast<int64_t>(copyCount / 2) * 2;
                const int64_t col = static_cast<int64_t>(copyCount % 2) * 3;
                EXPECT_EQ(op.GetInputOperand(0)->shape, op.GetOutputOperand(0)->shape);
                ExpectViewOffset(op.GetInputOperand(0), (Offset{0, 0, row, col, 0}));
                ExpectAssembleOffset(op.GetOutputOperand(0), (Offset{0, 0, row, col, 0}));
                ++copyCount;
            }
        }
        EXPECT_EQ(copyCount, 4U);
    }
}

TEST_F(OperationTilingTest, ScatterUpdatePaBsnd2DTilesRowsAndColumns)
{
    Tensor src(DT_FP32, {6, 16}, "src");
    Tensor index(DT_INT32, {2, 3}, "index");
    Tensor dst(DT_FP32, {8, 16}, "dst");
    FUNCTION("ScatterUpdatePaBsnd2D", {src, index, dst})
    {
        auto output = ScatterUpdate(dst, index, src, -2, "PA_BSND", 2);
        EXPECT_EQ(output.GetShape(), (Shape{8, 16}));
        auto& root = Program::GetInstance().GetCurrentFunction()->Operations(false).back();
        ASSERT_EQ(root.GetOpcode(), Opcode::OP_INDEX_OUTCAST);
        TileShape tile;
        tile.SetVecTile(3, 8);
        auto tiled = Expand(root, tile);
        size_t scatterCount = 0;
        for (const auto& op : tiled->Operations(false)) {
            if (op.GetOpcode() != Opcode::OP_INDEX_OUTCAST) {
                continue;
            }
            const int64_t batch = static_cast<int64_t>(scatterCount / 2);
            const int64_t col = static_cast<int64_t>(scatterCount % 2) * 8;
            EXPECT_EQ(op.GetInputOperand(0)->shape, (Shape{3, 8}));
            ExpectViewOffset(op.GetInputOperand(0), (Offset{batch * 3, col}));
            EXPECT_EQ(op.GetInputOperand(1)->shape, (Shape{1, 3}));
            ExpectViewOffset(op.GetInputOperand(1), (Offset{batch, 0}));
            EXPECT_EQ(op.GetInputOperand(2)->shape, (Shape{8, 8}));
            ExpectViewOffset(op.GetInputOperand(2), (Offset{0, col}));
            EXPECT_EQ(op.GetOutputOperand(0), tiled->GetOriginOutcast().at(0));
            EXPECT_EQ(op.GetOutputOperand(0)->shape, output.GetShape());
            EXPECT_EQ(op.GetIntAttribute("axis"), 0);
            EXPECT_EQ(op.GetStringAttribute(OpAttributeKey::cacheMode), "PA_BSND");
            EXPECT_EQ(op.GetIntAttribute(OpAttributeKey::panzBlockSize), 2);
            ++scatterCount;
        }
        EXPECT_EQ(scatterCount, 4U);
    }
}

TEST_F(OperationTilingTest, ScatterUpdatePaBsnd4DPreservesSequenceTail)
{
    Tensor src(DT_FP32, {2, 3, 2, 16}, "src");
    Tensor index(DT_INT32, {2, 3}, "index");
    Tensor dst(DT_FP32, {4, 2, 2, 16}, "dst");
    FUNCTION("ScatterUpdatePaBsnd4D", {src, index, dst})
    {
        auto output = ScatterUpdate(dst, index, src, -2, "PA_BSND", 2);
        EXPECT_EQ(output.GetShape(), (Shape{4, 2, 2, 16}));
        auto& root = Program::GetInstance().GetCurrentFunction()->Operations(false).back();
        ASSERT_EQ(root.GetOpcode(), Opcode::OP_INDEX_OUTCAST);
        TileShape tile;
        tile.SetVecTile(1, 2, 1, 8);
        auto tiled = Expand(root, tile);
        size_t scatterCount = 0;
        for (const auto& op : tiled->Operations(false)) {
            if (op.GetOpcode() != Opcode::OP_INDEX_OUTCAST) {
                continue;
            }
            const int64_t batch = static_cast<int64_t>(scatterCount / 8);
            const int64_t sequence = static_cast<int64_t>(scatterCount / 4 % 2) * 2;
            const int64_t head = static_cast<int64_t>(scatterCount / 2 % 2);
            const int64_t col = static_cast<int64_t>(scatterCount % 2) * 8;
            const int64_t sequenceSize = std::min<int64_t>(2, 3 - sequence);
            EXPECT_EQ(op.GetInputOperand(0)->shape, (Shape{1, sequenceSize, 1, 8}));
            ExpectViewOffset(op.GetInputOperand(0), (Offset{batch, sequence, head, col}));
            EXPECT_EQ(op.GetInputOperand(1)->shape, (Shape{1, sequenceSize}));
            ExpectViewOffset(op.GetInputOperand(1), (Offset{batch, sequence}));
            EXPECT_EQ(op.GetInputOperand(2)->shape, (Shape{4, 2, 2, 8}));
            ExpectViewOffset(op.GetInputOperand(2), (Offset{0, 0, 0, col}));
            EXPECT_EQ(op.GetOutputOperand(0), tiled->GetOriginOutcast().at(0));
            EXPECT_EQ(op.GetOutputOperand(0)->shape, output.GetShape());
            EXPECT_EQ(op.GetIntAttribute("axis"), 2);
            EXPECT_EQ(op.GetStringAttribute(OpAttributeKey::cacheMode), "PA_BSND");
            EXPECT_EQ(op.GetIntAttribute(OpAttributeKey::panzBlockSize), 2);
            ++scatterCount;
        }
        EXPECT_EQ(scatterCount, 16U);
    }
}

TEST_F(OperationOpsTest, CheckIndexAddParamsInvalid_FP16_Overflow)
{
    std::vector<int64_t> selfShape = {10, 10};
    std::vector<int64_t> srcShape = {5, 10};
    std::vector<int64_t> indicesShape = {5};
    int axis = 0;

    Tensor self(DT_FP16, selfShape);
    Tensor src(DT_FP16, srcShape);
    Tensor indices(DT_INT32, indicesShape);
    Element alpha(DT_FP16, 65505.0f);

    EXPECT_THROW(IndexAdd_(self, src, indices, axis, alpha), std::exception);
}

TEST_F(OperationOpsTest, ScatterUpdate_InconsistentDataType)
{
    Tensor dst(DT_FP32, {8, 8});
    Tensor index(DT_INT32, {1, 2});
    Tensor src(DT_FP16, {2, 8});

    EXPECT_THROW(ScatterUpdate(dst, index, src, -2), std::exception);
}

TEST_F(OperationOpsTest, ScatterUpdate_UnsupportedDataType)
{
    Tensor dst(DT_UINT8, {8, 8});
    Tensor index(DT_INT32, {1, 2});
    Tensor src(DT_UINT8, {2, 8});

    EXPECT_THROW(ScatterUpdate(dst, index, src, -2), std::exception);
}

TEST_F(OperationOpsTest, Range_UnsupportedStartDataType)
{
    Element start(DT_INT8, 0);
    Element end(DT_INT32, 10);
    Element step(DT_INT32, 1);

    EXPECT_THROW(Range(start, end, step), std::exception);
}

TEST_F(OperationOpsTest, Range_UnsupportedEndDataType)
{
    Element start(DT_INT32, 0);
    Element end(DT_INT8, 10);
    Element step(DT_INT32, 1);

    EXPECT_THROW(Range(start, end, step), std::exception);
}

TEST_F(OperationOpsTest, Range_UnsupportedStepDataType)
{
    Element start(DT_INT32, 0);
    Element end(DT_INT32, 10);
    Element step(DT_INT8, 1);

    EXPECT_THROW(Range(start, end, step), std::exception);
}

TEST_F(OperationOpsTest, Range_UnsupportedOutputDataType)
{
    Element start(DT_INT64, 0);
    Element end(DT_INT64, INT64_MAX);
    Element step(DT_INT64, 1);

    EXPECT_THROW(Range(start, end, step), std::exception);
}

TEST_F(OperationOpsTest, Gcd_UnsupportedScalarDataType)
{
    Tensor self(DT_INT32, {4, 8});
    Element scalar(DT_FP32, 1.5f);

    EXPECT_THROW(Gcd(self, scalar), std::exception);
}

TEST_F(OperationOpsTest, BitwiseRightShift_UnsupportedScalarDataType)
{
    Tensor self(DT_INT16, {4, 8});
    Element scalar(DT_FP32, 1.0f);

    EXPECT_THROW(BitwiseRightShift(self, scalar), std::exception);
}

TEST_F(OperationOpsTest, SBitwiseRightShift_UnsupportedScalarDataType)
{
    Element scalar(DT_FP32, 1.0f);
    Tensor other(DT_INT16, {4, 8});

    EXPECT_THROW(BitwiseRightShift(scalar, other), std::exception);
}

TEST_F(OperationOpsTest, BitwiseLeftShift_UnsupportedScalarDataType)
{
    Tensor self(DT_INT16, {4, 8});
    Element scalar(DT_FP32, 1.0f);

    EXPECT_THROW(BitwiseLeftShift(self, scalar), std::exception);
}

TEST_F(OperationOpsTest, SBitwiseLeftShift_UnsupportedScalarDataType)
{
    Element scalar(DT_FP32, 1.0f);
    Tensor other(DT_INT16, {4, 8});

    EXPECT_THROW(BitwiseLeftShift(scalar, other), std::exception);
}

TEST_F(OperationOpsTest, LogicalNot_UnsupportedDataType)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_2201);
    std::vector<int64_t> shape = {10, 10};
    Tensor input(DT_INT32, shape);

    EXPECT_THROW(LogicalNot(input), std::exception);
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationOpsTest, LogicalNot_A5SupportedDataType)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    TileShape::Current().SetVecTile(10, 10);
    Tensor input(DT_UINT32, {10, 10});

    FUNCTION("LogicalNotA5SupportedDataType", {input}) { EXPECT_NO_THROW((void)LogicalNot(input)); }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationOpsTest, Cast_Fp4TileInnerAxis)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    FUNCTION("CastFp4TileInnerAxis")
    {
        auto& function = *Program::GetInstance().GetCurrentFunction();
        for (auto dtype : {DT_FP32, DT_FP4_E2M1, DT_FP4_E1M2}) {
            for (bool fp4Input : {false, true}) {
                for (int64_t innerSize : {9, 10}) {
                    const std::vector<int64_t> shape = {2, innerSize};
                    Tensor input(std::make_shared<LogicalTensor>(function, fp4Input ? dtype : DT_BF16, shape));
                    const auto dstType = fp4Input ? DT_BF16 : dtype;
                    for (int64_t tileSize : {3, 4, 11}) {
                        SCOPED_TRACE(testing::Message() << "dtype=" << dtype << ", fp4Input=" << fp4Input
                                                        << ", innerSize=" << innerSize << ", tileSize=" << tileSize);
                        TileShape::Current().SetVecTile(2, tileSize);
                        if (dtype != DT_FP32 && (innerSize == 9 || tileSize == 3)) {
                            EXPECT_THROW(Cast(input, dstType), std::exception);
                        } else {
                            EXPECT_NO_THROW((void)Cast(input, dstType));
                        }
                    }
                }
            }
        }
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationOpsTest, QuantMX_DefaultRoundDownFp8Output)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    TileShape::Current().SetVecTile(8, 64);
    Tensor input(DT_FP32, {8, 64});

    FUNCTION("QuantMXDefaultFp8", {input})
    {
        auto defaultRes = QuantMX(input);
        EXPECT_EQ(std::get<0>(defaultRes).GetDataType(), DT_FP8E4M3);
        EXPECT_EQ(std::get<1>(defaultRes).GetDataType(), DT_FP8E8M0);
        EXPECT_EQ(std::get<1>(defaultRes).GetShape(), std::vector<int64_t>({8, 1, 2}));
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationOpsTest, QuantMX_RoundUpFp8AndFp4Output)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    TileShape::Current().SetVecTile(8, 128);
    Tensor fp16Input(DT_FP16, {8, 128});
    Tensor fp32Input(DT_FP32, {8, 128});

    FUNCTION("QuantMXRoundUp", {fp16Input, fp32Input})
    {
        auto fp8Res = QuantMX(fp32Input, DT_FP8E4M3, DequantScaleRoundingMode::ROUND_UP, -1, true);
        EXPECT_EQ(std::get<0>(fp8Res).GetDataType(), DT_FP8E4M3);
        EXPECT_EQ(std::get<1>(fp8Res).GetDataType(), DT_FP8E8M0);

        auto fp4Res = QuantMX(fp16Input, DT_FP4_E2M1, DequantScaleRoundingMode::ROUND_UP, -1, true);
        EXPECT_EQ(std::get<0>(fp4Res).GetDataType(), DT_FP4_E2M1);
        EXPECT_EQ(std::get<0>(fp4Res).GetShape(), std::vector<int64_t>({8, 128}));
        EXPECT_EQ(DataSizeOf(8 * 128, std::get<0>(fp4Res).GetDataType()), 8 * 128 / 2);
        EXPECT_EQ(std::get<1>(fp4Res).GetDataType(), DT_FP8E8M0);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationOpsTest, QuantMX_Fp32ToFp4Unsupported)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    TileShape::Current().SetVecTile(8, 64);
    Tensor input(DT_FP32, {8, 64});

    FUNCTION("QuantMXFp32ToFp4Unsupported", {input})
    {
        EXPECT_THROW(QuantMX(input, DT_FP4_E2M1, DequantScaleRoundingMode::ROUND_DOWN, -1, true), std::exception);
        EXPECT_THROW(QuantMX(input, DT_FP4_E2M1, DequantScaleRoundingMode::ROUND_UP, -1, true), std::exception);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationOpsTest, QuantMX_NonPerformanceTileQuantizationAxisAlignment)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    const std::vector<std::pair<DataType, DataType>> dtypePairs = {{DT_FP32, DT_FP8E4M3},
                                                                   {DT_FP16, DT_FP8E4M3},
                                                                   {DT_BF16, DT_FP8E4M3},
                                                                   {DT_FP16, DT_FP4_E2M1},
                                                                   {DT_BF16, DT_FP4_E2M1}};
    FUNCTION("QuantMXNonPerformanceTileAlignment")
    {
        auto& function = *Program::GetInstance().GetCurrentFunction();
        for (const auto& [inputDtype, quantDtype] : dtypePairs) {
            Tensor input(std::make_shared<LogicalTensor>(function, inputDtype, std::vector<int64_t>{128, 128}));
            for (int64_t axis : {-2, -1, 0, 1}) {
                for (int64_t tileDim : {32, 64, 96, 128}) {
                    SCOPED_TRACE(testing::Message() << "input dtype=" << inputDtype << ", quant dtype=" << quantDtype
                                                    << ", axis=" << axis << ", tile dim=" << tileDim);
                    const auto normalizedAxis = axis < 0 ? axis + 2 : axis;
                    std::vector<int64_t> tileShape = {128, 128};
                    tileShape[normalizedAxis] = tileDim;
                    TileShape::Current().SetVecTile(tileShape);
                    if (tileDim % 64 == 0) {
                        EXPECT_NO_THROW(
                            (void)QuantMX(input, quantDtype, DequantScaleRoundingMode::ROUND_DOWN, axis, false));
                    } else {
                        EXPECT_THROW(QuantMX(input, quantDtype, DequantScaleRoundingMode::ROUND_DOWN, axis, false),
                                     std::exception);
                    }
                }
            }
        }
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationOpsTest, DivMod_OutputShapeAndDtype)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    TileShape::Current().SetVecTile(8, 64);
    Tensor dividend(DT_INT32, {8, 64});
    Tensor divisor(DT_INT32, {8, 64});

    FUNCTION("DivModInt32", {dividend, divisor})
    {
        auto [quotient, remainder] = DivMod(dividend, divisor);
        EXPECT_EQ(quotient.GetDataType(), DT_INT32);
        EXPECT_EQ(remainder.GetDataType(), DT_INT32);
        EXPECT_EQ(quotient.GetShape(), std::vector<int64_t>({8, 64}));
        EXPECT_EQ(remainder.GetShape(), std::vector<int64_t>({8, 64}));
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationOpsTest, DivMod_Broadcast)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    TileShape::Current().SetVecTile(8, 64);
    Tensor dividend(DT_INT32, {8, 1});
    Tensor divisor(DT_INT32, {8, 64});

    FUNCTION("DivModBroadcast", {dividend, divisor})
    {
        auto [quotient, remainder] = DivMod(dividend, divisor);
        EXPECT_EQ(quotient.GetDataType(), DT_INT32);
        EXPECT_EQ(remainder.GetDataType(), DT_INT32);
        EXPECT_EQ(quotient.GetShape(), std::vector<int64_t>({8, 64}));
        EXPECT_EQ(remainder.GetShape(), std::vector<int64_t>({8, 64}));
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationOpsTest, DivMod_A5WideDtypes)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    TileShape::Current().SetVecTile(8, 64);
    Tensor dividend64(DT_INT64, {8, 64});
    Tensor divisor64(DT_INT64, {8, 64});
    Tensor dividendU16(DT_UINT16, {8, 64});
    Tensor divisorU16(DT_UINT16, {8, 64});
    Tensor dividendU32(DT_UINT32, {8, 64});
    Tensor divisorU32(DT_UINT32, {8, 64});
    Tensor dividendU64(DT_UINT64, {8, 64});
    Tensor divisorU64(DT_UINT64, {8, 64});

    FUNCTION("DivModA5Wide",
             {dividend64, divisor64, dividendU16, divisorU16, dividendU32, divisorU32, dividendU64, divisorU64})
    {
        auto res64 = DivMod(dividend64, divisor64);
        EXPECT_EQ(std::get<0>(res64).GetDataType(), DT_INT64);
        auto resU16 = DivMod(dividendU16, divisorU16);
        EXPECT_EQ(std::get<0>(resU16).GetDataType(), DT_UINT16);
        auto resU32 = DivMod(dividendU32, divisorU32);
        EXPECT_EQ(std::get<0>(resU32).GetDataType(), DT_UINT32);
        auto resU64 = DivMod(dividendU64, divisorU64);
        EXPECT_EQ(std::get<0>(resU64).GetDataType(), DT_UINT64);
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationOpsTest, DivMod_Uint16OnA2A3)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_2201);
    TileShape::Current().SetVecTile(8, 64);
    Tensor dividend(DT_UINT16, {8, 64});
    Tensor divisor(DT_UINT16, {8, 64});

    FUNCTION("DivModUint16A2A3", {dividend, divisor})
    {
        auto [quotient, remainder] = DivMod(dividend, divisor);
        EXPECT_EQ(quotient.GetDataType(), DT_UINT16);
        EXPECT_EQ(remainder.GetDataType(), DT_UINT16);
        EXPECT_EQ(quotient.GetShape(), std::vector<int64_t>({8, 64}));
        EXPECT_EQ(remainder.GetShape(), std::vector<int64_t>({8, 64}));
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationOpsTest, QuantHiF4Bf16)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_9201);
    TileShape::Current().SetVecTile(8, 128);
    Tensor input(DT_BF16, {8, 128});

    FUNCTION("QuantHiF4Bf16", {input})
    {
        auto defaultRes = QuantHiF4(input);
        EXPECT_EQ(std::get<0>(defaultRes).GetDataType(), DT_HF4);
        EXPECT_EQ(std::get<1>(defaultRes).GetDataType(), DT_UINT32);
        EXPECT_EQ(std::get<1>(defaultRes).GetShape(), std::vector<int64_t>({8, 2}));
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationOpsTest, QuantHiF4MultiTile)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_9201);
    // [8, 512] split into four tiles of [2, 512].
    TileShape::Current().SetVecTile(2, 512);
    Tensor input(DT_BF16, {8, 512});

    FUNCTION("QuantHiF4MultiTile", {input})
    {
        auto res = QuantHiF4(input);
        EXPECT_EQ(std::get<0>(res).GetDataType(), DT_HF4);
        EXPECT_EQ(std::get<0>(res).GetShape(), std::vector<int64_t>({8, 512}));
        EXPECT_EQ(std::get<1>(res).GetDataType(), DT_UINT32);
        EXPECT_EQ(std::get<1>(res).GetShape(), std::vector<int64_t>({8, 8}));
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationOpsTest, QuantHiF4TailTiles)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_9201);
    TileShape::Current().SetVecTile(4, 128);
    Tensor input(DT_BF16, {6, 192});
    FUNCTION("QuantHiF4TailTiles", {input})
    {
        auto res = QuantHiF4(input, DT_HF4, 1);
        EXPECT_EQ(std::get<1>(res).GetShape(), std::vector<int64_t>({6, 3}));
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationOpsTest, QuantHiF4PartialGroups)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_9201);
    TileShape::Current().SetVecTile(4, 512);
    Tensor input(DT_BF16, {6, 704});
    input.GetStorage()->UpdateDynValidShape({6, 650});
    FUNCTION("QuantHiF4PartialGroups", {input})
    {
        auto result = QuantHiF4(input);
        EXPECT_EQ(std::get<0>(result).GetShape(), std::vector<int64_t>({6, 704}));
        EXPECT_EQ(std::get<1>(result).GetShape(), std::vector<int64_t>({6, 11}));
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationOpsTest, QuantHiF4RejectsUnalignedTileCapacity)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_9201);
    TileShape::Current().SetVecTile(3, 145);
    Tensor input(DT_BF16, {5, 576});
    FUNCTION("QuantHiF4UnalignedTileCapacity", {input}) { EXPECT_THROW(QuantHiF4(input), std::exception); }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationOpsTest, QuantHiF4RejectsUnalignedView)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_9201);
    TileShape::Current().SetVecTile(4, 704);
    Tensor input(DT_BF16, {6, 650});
    FUNCTION("QuantHiF4UnalignedView", {input}) { EXPECT_THROW(QuantHiF4(input), std::exception); }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationOpsTest, QuantHiF4WholeTileScratchBounds)
{
    EXPECT_EQ(hif4::ScratchLayout(1, 512).bytes, 736u);
    EXPECT_EQ(hif4::ScratchLayout(32, 1024).bytes, 47104u);
    EXPECT_EQ(hif4::ScratchLayout(3, 192).bytes, 928u);
    for (size_t capacityCols = 64; capacityCols <= 2048; capacityCols += 64) {
        const hif4::ScratchLayout layout(7, capacityCols);
        EXPECT_EQ(layout.max8Offset, layout.scaleBytes);
        EXPECT_EQ(layout.eaOffset, layout.max8Offset + layout.max8Bytes);
        EXPECT_GE(layout.bytes, layout.ecOffset + layout.metaBytes);
        EXPECT_GE(layout.bytes, layout.elements);
        for (size_t validCols = 2; validCols <= capacityCols; validCols += 2) {
            EXPECT_LE((validCols + 63) / 64, layout.cols / 64);
        }
        {
            const auto elements = layout.elements;
            for (size_t base = 0; base < elements; base += 512) {
                const auto count = std::min<size_t>(512, elements - base);
                EXPECT_LE((base + count) / 4 * 2, layout.scaleBytes); // initial max4
                EXPECT_LE(base + count, layout.scaleBytes);           // expanded inverse scales
                EXPECT_LE((base + count) / 8 * 2, layout.max8Bytes);
                EXPECT_LE(base / 512 * 32 + 32, layout.metaBytes);
                // Reverse expansion never touches max4 of a still unread lower batch.
                EXPECT_GE(base, base / 4 * 2);
            }
        }
    }
}

TEST_F(OperationOpsTest, QuantHiF4ScratchPerMatrix)
{
    Program::GetInstance().Reset();
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_9201);
    TileShape::Current().SetVecTile(2, 3, 5, 192);
    Tensor input(DT_BF16, {2, 3, 5, 192}, "input");
    FUNCTION("QuantHiF4ScratchPerMatrix", {input})
    {
        auto result = QuantHiF4(input);
        (void)result;
        const auto& op = Program::GetInstance().GetCurrentFunction()->Operations(false).back();
        EXPECT_EQ(op.GetOOperands().at(2)->GetShape(),
                  std::vector<int64_t>({static_cast<int64_t>(hif4::ScratchLayout(5, 192).bytes)}));
    }
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
}

TEST_F(OperationOpsTest, MoveUbToUb_SameShapeOutput)
{
    TileShape::Current().SetVecTile(8, 64);
    Tensor input(DT_FP32, {8, 64});
    Tensor out(DT_FP32, {8, 64});

    FUNCTION("MoveUbToUbSameShape", {input, out})
    {
        auto srcTile = View(input, {8, 64}, {0, 0});
        Tensor dstTile(DT_FP32, {8, 64});
        MoveUbToUb(srcTile, dstTile);
        Assemble(dstTile, {0, 0}, out);
    }
    EXPECT_EQ(out.GetDataType(), DT_FP32);
    EXPECT_EQ(out.GetShape(), std::vector<int64_t>({8, 64}));
}

TEST_F(OperationOpsTest, MoveUbToUb_UnsupportedRank)
{
    TileShape::Current().SetVecTile(1, 8, 8, 8, 8);
    Tensor input(DT_FP32, {2, 2, 2, 2, 2, 2});
    Tensor out(DT_FP32, {2, 2, 2, 2, 2, 2});

    FUNCTION("MoveUbToUbUnsupportedRank", {input}) { EXPECT_THROW(MoveUbToUb(input, out), std::exception); }
}

TEST_F(OperationOpsTest, DumpAttr_PrintsInplaceInfoMap)
{
    Program::GetInstance().Reset();
    Tensor self(DT_FP32, {2, 5});
    Tensor indices(DT_INT64, {1, 4});

    FUNCTION("DumpAttrScatterInplace", {self, indices})
    {
        TileShape::Current().SetVecTile(8, 8);
        auto out = Scatter(self, indices, Element(DT_FP32, 2.0f), 0, ScatterMode::ADD);
        (void)out;
        auto* func = Program::GetInstance().GetCurrentFunction();
        ASSERT_NE(func, nullptr);
        ASSERT_EQ(func->GetOperationSize(), 1);
        auto opDump = func->Operations(false).back().DumpAttr();
        EXPECT_NE(opDump.find("#INPLACE_INFO{0:0}"), std::string::npos);
        EXPECT_EQ(opDump.find("unsupported type"), std::string::npos);

        auto opAttrJson = func->Operations(false).back().DumpAttrJson(OpAttributeKey::inplaceInfo);
        ASSERT_TRUE(opAttrJson.is_object());
        ASSERT_TRUE(opAttrJson.contains("0"));
        EXPECT_EQ(opAttrJson["0"], 0);
    }
}

TEST_F(OperationOpsTest, Axpy_UnsupportedSelfDim)
{
    std::vector<int64_t> shape5d = {2, 4, 8, 16, 32};
    Tensor self(DT_FP32, shape5d);
    Tensor other(DT_FP32, shape5d);

    FUNCTION("AxpyUnsupportedSelfDim", {self, other}) { EXPECT_THROW(Axpy(self, other, 1.0f), std::exception); }
}

TEST_F(OperationOpsTest, Add_Int64BroadcastRejected)
{
    Tensor self(DT_INT64, {4, 8});
    Tensor other(DT_INT64, {1, 8}); // broadcast on axis 0
    FUNCTION("AddInt64Broadcast", {self, other}) { EXPECT_THROW(Add(self, other), std::exception); }
}

TEST_F(OperationOpsTest, Clip_Int64BroadcastRejected)
{
    Tensor input(DT_INT64, {4, 8});
    Tensor minT(DT_INT64, {4, 1}); // broadcast on axis 0
    Tensor maxT(DT_INT64, {1, 8}); // broadcast on axis 1
    FUNCTION("ClipInt64Broadcast", {input, minT, maxT}) { EXPECT_THROW(Clip(input, minT, maxT), std::exception); }
}

TEST_F(OperationOpsTest, Axpy_UnsupportedOtherDim)
{
    std::vector<int64_t> shape4d = {4, 8, 16, 32};
    std::vector<int64_t> shape5d = {2, 4, 8, 16, 32};
    Tensor self(DT_FP32, shape4d);
    Tensor other(DT_FP32, shape5d);

    FUNCTION("AxpyUnsupportedOtherDim", {self, other}) { EXPECT_THROW(Axpy(self, other, 1.0f), std::exception); }
}

TEST_F(OperationOpsTest, Squeeze_DimensionExceedsMax)
{
    std::vector<int64_t> shape5d = {1, 2, 3, 4, 5};
    Tensor input(DT_FP32, shape5d);
    std::vector<int> dim = {0};

    FUNCTION("SqueezeDimExceedsMax", {input}) { EXPECT_THROW(Squeeze(input, dim), std::exception); }
}

TEST_F(OperationOpsTest, Squeeze_DuplicateDimensions)
{
    std::vector<int64_t> shape4d = {1, 2, 1, 4};
    Tensor input(DT_FP32, shape4d);
    std::vector<int> dim = {0, 0};

    FUNCTION("SqueezeDuplicateDims", {input}) { EXPECT_THROW(Squeeze(input, dim), std::exception); }
}
