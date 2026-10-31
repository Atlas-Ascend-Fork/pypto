/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_codegen_dyn_tri.cpp
 * \brief Unit test for codegen.
 */

#include "gtest/gtest.h"
#include <fstream>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/npu/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_common.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {

namespace {
std::string GenAllCodeByFunction(Function& function)
{
    CodeGenCtx ctx;
    CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(function);

    std::string results;
    for (const auto& subFunc : function.rootFunc_->programs_) {
        auto leafFuncAttr = subFunc.second->GetLeafFuncAttribute();
        ASSERT(FwkErr::INVALID_FUNCTION, leafFuncAttr != nullptr) << "can not find leaf func attribute";
        const auto& binPath = leafFuncAttr->binPath;
        if (binPath.empty()) {
            continue;
        }
        std::string cppFile = binPath.substr(0, binPath.rfind('.')) + ".cpp";
        std::ifstream ifs(cppFile);
        results.append(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
        results.push_back('\n');
    }
    return results;
}
} // namespace

class TestCodegenDynTransData : public CodegenTestBase {
public:
    TestCodegenDynTransData() : CodegenTestBase({.compileStage = CS_CODEGEN_INSTRUCTION}) {}

    static void TearDownTestCase() {}

    struct TransDataTestCase {
        std::string caseName;
        std::vector<int64_t> inputShape;
        std::vector<int64_t> vecTile;
        DataType dtype;
        std::vector<int64_t> outputShape;
        int tileOpFormat;
        std::vector<SymbolicScalar> validShape;
        std::string expect;
        bool checkAllSubKernels = false;
    };

    void RunTransDataTest(const TransDataTestCase& tc)
    {
        TileShape::Current().SetVecTile(tc.vecTile);

        Tensor inputSrc(tc.dtype, tc.inputShape, "input");
        Tensor output(tc.dtype, tc.outputShape, "output");
        ConfigManager::Instance();
        FUNCTION(tc.caseName, {inputSrc, output})
        {
            LOOP(tc.caseName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
            {
                (void)i;
                output = TransData(inputSrc, TileOpFormat(tc.tileOpFormat), tc.outputShape, tc.validShape, 1);
            }
        }
        auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + tc.caseName + SUB_FUNC_SUFFIX +
                                                                    HIDDEN_FUNC_SUFFIX);
        std::string res = tc.checkAllSubKernels ? GenAllCodeByFunction(*function) : GenCodeByFunction(*function);
        CheckStringExist(tc.expect, res);
    }
};

TEST_F(TestCodegenDynTransData, TestTransData2)
{
    RunTransDataTest({.caseName = "TestTransData2",
                      .inputShape = {1, 7, 1, 8},
                      .vecTile = {1, 8, 1, 8},
                      .dtype = DT_FP32,
                      .outputShape = {1, 1, 1, 8, 8},
                      .tileOpFormat = 2,
                      .validShape = {1, 1, 1, 8, 8},
                      .expect = R"!!!(TTransDataNCHW2NC1HWC0<(int)(1), (int)(8), (int)(1), (int)(8)>(ubTensor_)!!!"});
}

TEST_F(TestCodegenDynTransData, TestTransData4)
{
    RunTransDataTest(
        {.caseName = "TestTransData4",
         .inputShape = {15, 15, 2, 16},
         .vecTile = {16, 16, 1, 16},
         .dtype = DT_FP16,
         .outputShape = {32, 1, 16, 16},
         .tileOpFormat = 4,
         .validShape = {32, 1, 16, 16},
         .expect = R"!!!(TTransDataNCHW2Fractal_Z<(int)(16), (int)(16), (int)(2), (int)(16)>(ubTensor_)!!!"});
}

TEST_F(TestCodegenDynTransData, TestTransData3)
{
    RunTransDataTest(
        {.caseName = "TestTransData3",
         .inputShape = {1, 7, 1, 1, 8},
         .vecTile = {1, 8, 1, 1, 8},
         .dtype = DT_FP32,
         .outputShape = {1, 1, 1, 1, 8, 8},
         .tileOpFormat = 3,
         .validShape = {1, 1, 1, 1, 8, 8},
         .expect = R"!!!(TTransDataNCDHW2NDC1HWC0<(int)(1), (int)(1), (int)(8), (int)(1), (int)(8)>(ubTensor_)!!!"});
}

TEST_F(TestCodegenDynTransData, TestTransData5)
{
    RunTransDataTest(
        {.caseName = "TestTransData5",
         .inputShape = {15, 7, 1, 1, 8},
         .vecTile = {16, 8, 1, 1, 8},
         .dtype = DT_FP32,
         .outputShape = {8, 1, 16, 8},
         .tileOpFormat = 5,
         .validShape = {8, 1, 16, 8},
         .expect = R"!!!(TTransDataNCDHW2FRACTAL_Z_3D<(int)(16), (int)(8), (int)(1), (int)(1), (int)(8)>(ubTensor_)!!!",
         .checkAllSubKernels = true});
}

TEST_F(TestCodegenDynTransData, TestTransDataTmpFzPartialW)
{
    RunTransDataTest({.caseName = "TestTransDataTmpFzPartialW",
                      .inputShape = {16, 16, 1, 40},
                      .vecTile = {16, 16, 1, 32},
                      .dtype = DT_FP16,
                      .outputShape = {40, 1, 16, 16},
                      .tileOpFormat = 4,
                      .validShape = {40, 1, 16, 16},
                      .expect = R"!!!(TTransDataNCHW2Fractal_Z)!!!"});
}

TEST_F(TestCodegenDynTransData, TestTransData5D2)
{
    RunTransDataTest(
        {.caseName = "TestTransData5D2",
         .inputShape = {16, 7, 2, 1, 8},
         .vecTile = {16, 8, 2, 1, 8},
         .dtype = DT_FP32,
         .outputShape = {16, 1, 16, 8},
         .tileOpFormat = 5,
         .validShape = {16, 1, 16, 8},
         .expect = R"!!!(TTransDataNCDHW2FRACTAL_Z_3D<(int)(16), (int)(8), (int)(2), (int)(1), (int)(8)>)!!!"});
}

TEST_F(TestCodegenDynTransData, TestTransData0_3)
{
    RunTransDataTest(
        {.caseName = "TestTransData0_3",
         .inputShape = {1, 1, 1, 8, 8},
         .vecTile = {1, 1, 1, 8, 8},
         .dtype = DT_FP32,
         .outputShape = {1, 7, 1, 8},
         .tileOpFormat = 0,
         .validShape = {1, 7, 1, 8},
         .expect = R"!!!(TTransDataNC1HWC02NCHW<(int)(1), (int)(1), (int)(1), (int)(8), (int)(8)>(ubTensor_)!!!"});
}

TEST_F(TestCodegenDynTransData, TestTransData0_6)
{
    RunTransDataTest(
        {.caseName = "TestTransData0_6",
         .inputShape = {1, 1, 1, 1, 8, 8},
         .vecTile = {1, 1, 1, 1, 8, 8},
         .dtype = DT_FP32,
         .outputShape = {1, 7, 1, 1, 8},
         .tileOpFormat = 0,
         .validShape = {1, 7, 1, 1, 8},
         .expect = R"!!!(TTransDataNDC1HWC02NCDHW<(int)(1), (int)(1), (int)(1), (int)(8), (int)(8)>(ubTensor_)!!!"});
}

TEST_F(TestCodegenDynTransData, TestTransData0_4)
{
    RunTransDataTest({.caseName = "TestTransData0_4",
                      .inputShape = {8, 1, 16, 8},
                      .vecTile = {8, 1, 16, 8},
                      .dtype = DT_FP32,
                      .outputShape = {16, 8, 1, 8},
                      .tileOpFormat = 0,
                      .validShape = {16, 8, 1, 8},
                      .expect = R"!!!(TTransDataFractalZ2NCHW(ubTensor_)!!!"});
}

TEST_F(TestCodegenDynTransData, TestTransData0_5)
{
    RunTransDataTest({.caseName = "TestTransData0_5",
                      .inputShape = {8, 1, 16, 8},
                      .vecTile = {8, 1, 16, 8},
                      .dtype = DT_FP32,
                      .outputShape = {16, 8, 1, 1, 8},
                      .tileOpFormat = 0,
                      .validShape = {16, 8, 1, 1, 8},
                      .expect = R"!!!(TTransDataFractalZ3D2NCDHW(ubTensor_)!!!"});
}
} // namespace npu::tile_fwk
