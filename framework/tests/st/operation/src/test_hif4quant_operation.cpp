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
 * \file test_hif4quant_operation.cpp
 * \brief ST test for QuantHiF4 (BF16 -> HiFloat4 with three-level hierarchical scaling).
 */

#include "test_operation.h"

using namespace tile_fwk::test_operation;
namespace {
constexpr int64_t QUANT_HIF4_BLOCK_SIZE = 64;

struct QuantHiF4OpFuncArgs : public OpFuncArgs {
    QuantHiF4OpFuncArgs(const std::vector<int64_t>& viewShape, const std::vector<int64_t>& tileShape,
                        DataType quantDtype, int64_t scaleAlg, int64_t axis)
        : viewShape_(viewShape), tileShape_(tileShape), quantDtype_(quantDtype), scaleAlg_(scaleAlg), axis_(axis)
    {}

    std::vector<int64_t> viewShape_;
    std::vector<int64_t> tileShape_;
    DataType quantDtype_;
    int64_t scaleAlg_;
    int64_t axis_;
};

struct QuantHiF4OpMetaData {
    explicit QuantHiF4OpMetaData(const OpFunc& opFunc, const nlohmann::json& test_data)
        : opFunc_(opFunc), test_data_(test_data)
    {}

    OpFunc opFunc_;
    nlohmann::json test_data_;
};

// The public scale tensor is [*view[:-1], ceil(view[-1] / 64)], i.e. it keeps the input
// rank, so a view offset maps to the scale offset by dividing the quantization axis only.
// HiF4 is ND-only, hence there is no axis=-2 branch here.
std::vector<SymbolicScalar> BuildQuantHiF4ScaleOffset(const std::vector<SymbolicScalar>& offset)
{
    auto scaleOffset = offset;
    scaleOffset.back() = scaleOffset.back() / QUANT_HIF4_BLOCK_SIZE;
    return scaleOffset;
}

static void QuantHiF4OperationExeFunc1D(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                        const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0], outputs[1]})
    {
        auto args = static_cast<const QuantHiF4OpFuncArgs*>(opArgs);
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        const int firstViewShape = args->viewShape_[0];
        const int firstLoop = CeilDiv(firstDim, firstViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, firstLoop, 1))
        {
            std::vector<SymbolicScalar> offset = {bIdx * firstViewShape};
            auto viewTensor = View(inputs[0], args->viewShape_,
                                   {std::min(firstDim - bIdx * firstViewShape, firstViewShape)}, offset);
            TileShape::Current().SetVecTile(args->tileShape_);
            auto res = QuantHiF4(viewTensor, args->quantDtype_, args->scaleAlg_, args->axis_);
            auto scaleOffset = BuildQuantHiF4ScaleOffset(offset);
            Assemble(std::get<0>(res), offset, outputs[0]);
            Assemble(std::get<1>(res), scaleOffset, outputs[1]);
        }
    }
}

static void QuantHiF4OperationExeFunc2D(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                        const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0], outputs[1]})
    {
        auto args = static_cast<const QuantHiF4OpFuncArgs*>(opArgs);
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int firstLoop = CeilDiv(firstDim, firstViewShape);
        const int secondLoop = CeilDiv(secondDim, secondViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, firstLoop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, secondLoop, 1))
            {
                std::vector<SymbolicScalar> offset = {bIdx * firstViewShape, sIdx * secondViewShape};
                auto viewTensor = View(inputs[0], args->viewShape_,
                                       {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                        std::min(secondDim - sIdx * secondViewShape, secondViewShape)},
                                       offset);
                TileShape::Current().SetVecTile(args->tileShape_);
                auto res = QuantHiF4(viewTensor, args->quantDtype_, args->scaleAlg_, args->axis_);
                auto scaleOffset = BuildQuantHiF4ScaleOffset(offset);
                Assemble(std::get<0>(res), offset, outputs[0]);
                Assemble(std::get<1>(res), scaleOffset, outputs[1]);
            }
        }
    }
}

static void QuantHiF4OperationExeFunc3D(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                        const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0], outputs[1]})
    {
        auto args = static_cast<const QuantHiF4OpFuncArgs*>(opArgs);
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        const int firstLoop = CeilDiv(firstDim, firstViewShape);
        const int secondLoop = CeilDiv(secondDim, secondViewShape);
        const int thirdLoop = CeilDiv(thirdDim, thirdViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, firstLoop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, secondLoop, 1))
            {
                LOOP("LOOP_L2_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, thirdLoop, 1))
                {
                    std::vector<SymbolicScalar> offset = {bIdx * firstViewShape, sIdx * secondViewShape,
                                                          nIdx * thirdViewShape};
                    auto viewTensor = View(inputs[0], args->viewShape_,
                                           {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                            std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                            std::min(thirdDim - nIdx * thirdViewShape, thirdViewShape)},
                                           offset);
                    TileShape::Current().SetVecTile(args->tileShape_);
                    auto res = QuantHiF4(viewTensor, args->quantDtype_, args->scaleAlg_, args->axis_);
                    auto scaleOffset = BuildQuantHiF4ScaleOffset(offset);
                    Assemble(std::get<0>(res), offset, outputs[0]);
                    Assemble(std::get<1>(res), scaleOffset, outputs[1]);
                }
            }
        }
    }
}

static void QuantHiF4OperationExeFunc4D(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                                        const OpFuncArgs* opArgs)
{
    FUNCTION("main", {inputs[0]}, {outputs[0], outputs[1]})
    {
        auto args = static_cast<const QuantHiF4OpFuncArgs*>(opArgs);
        SymbolicScalar firstDim = inputs[0].GetShape()[0];
        SymbolicScalar secondDim = inputs[0].GetShape()[1];
        SymbolicScalar thirdDim = inputs[0].GetShape()[2];
        SymbolicScalar fourthDim = inputs[0].GetShape()[3];
        const int firstViewShape = args->viewShape_[0];
        const int secondViewShape = args->viewShape_[1];
        const int thirdViewShape = args->viewShape_[2];
        const int fourthViewShape = args->viewShape_[3];
        const int firstLoop = CeilDiv(firstDim, firstViewShape);
        const int secondLoop = CeilDiv(secondDim, secondViewShape);
        const int thirdLoop = CeilDiv(thirdDim, thirdViewShape);
        const int fourthLoop = CeilDiv(fourthDim, fourthViewShape);

        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, firstLoop, 1))
        {
            LOOP("LOOP_L1_sIdx", FunctionType::DYNAMIC_LOOP, sIdx, LoopRange(0, secondLoop, 1))
            {
                LOOP("LOOP_L2_mIdx", FunctionType::DYNAMIC_LOOP, mIdx, LoopRange(0, thirdLoop, 1))
                {
                    LOOP("LOOP_L3_nIdx", FunctionType::DYNAMIC_LOOP, nIdx, LoopRange(0, fourthLoop, 1))
                    {
                        std::vector<SymbolicScalar> offset = {bIdx * firstViewShape, sIdx * secondViewShape,
                                                              mIdx * thirdViewShape, nIdx * fourthViewShape};
                        auto viewTensor = View(inputs[0], args->viewShape_,
                                               {std::min(firstDim - bIdx * firstViewShape, firstViewShape),
                                                std::min(secondDim - sIdx * secondViewShape, secondViewShape),
                                                std::min(thirdDim - mIdx * thirdViewShape, thirdViewShape),
                                                std::min(fourthDim - nIdx * fourthViewShape, fourthViewShape)},
                                               offset);
                        TileShape::Current().SetVecTile(args->tileShape_);
                        auto res = QuantHiF4(viewTensor, args->quantDtype_, args->scaleAlg_, args->axis_);
                        auto scaleOffset = BuildQuantHiF4ScaleOffset(offset);
                        Assemble(std::get<0>(res), offset, outputs[0]);
                        Assemble(std::get<1>(res), scaleOffset, outputs[1]);
                    }
                }
            }
        }
    }
}

class QuantHiF4OperationTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac_param<QuantHiF4OpMetaData> {};

INSTANTIATE_TEST_SUITE_P(TestQuantHiF4, QuantHiF4OperationTest,
                         ::testing::ValuesIn(GetOpMetaData<QuantHiF4OpMetaData, 1>(
                             {QuantHiF4OperationExeFunc1D, QuantHiF4OperationExeFunc2D, QuantHiF4OperationExeFunc3D,
                              QuantHiF4OperationExeFunc4D},
                             "QuantHiF4")));

TEST_P(QuantHiF4OperationTest, TestQuantHiF4)
{
    auto test_data = GetParam().test_data_;
    auto quantDtype = GetDataType(test_data.at("output_tensors")[0].at("dtype"));
    int64_t scaleAlg = 0;
    if (test_data.at("params").contains("scale_alg")) {
        scaleAlg = GetValueByName<int64_t>(test_data, "scale_alg");
    }
    if (scaleAlg != 0 && scaleAlg != 1) {
        throw std::invalid_argument("Unsupported QuantHiF4 scale_alg: " + std::to_string(scaleAlg) +
                                    ". Expected 0 or 1.");
    }
    int64_t axis = -1;
    if (test_data.at("params").contains("axis")) {
        axis = GetValueByName<int64_t>(test_data, "axis");
    }
    auto args = QuantHiF4OpFuncArgs(GetViewShape(test_data), GetTileShape(test_data), quantDtype, scaleAlg, axis);
    auto testCase = CreateTestCaseDesc<QuantHiF4OpMetaData>(GetParam(), &args);
    TestExecutor::runTest(testCase);
}
} // namespace
