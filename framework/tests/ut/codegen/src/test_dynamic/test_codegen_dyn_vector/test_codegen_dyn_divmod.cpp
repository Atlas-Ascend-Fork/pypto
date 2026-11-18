#include "gtest/gtest.h"

#include <algorithm>
#include <string>

#include "tilefwk/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/inner/tilefwk.h"
#include "interface/operation/operation.h"
#include "interface/program/program.h"
#include "tilefwk/data_type.h"
#include "tilefwk/platform.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {
class TestCodegenDynDivMod : public CodegenTestBase {
public:
    TestCodegenDynDivMod()
        : CodegenTestBase(
              {.compileStage = CS_EXECUTE_GRAPH, .setTileTensor = true, .tileTensorValue = true, .setIdGen = true})
    {}

    void TearDown() override
    {
        Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
        CodegenTestBase::TearDown();
    }

protected:
    static size_t CountDivModParams(const std::string& code)
    {
        const std::string marker = "TDivMod(";
        auto begin = code.find(marker);
        if (begin == std::string::npos) {
            return 0;
        }
        begin += marker.size();
        auto end = code.find(");", begin);
        if (end == std::string::npos) {
            return 0;
        }
        std::string params = code.substr(begin, end - begin);
        if (params.empty()) {
            return 0;
        }
        return 1 + static_cast<size_t>(std::count(params.begin(), params.end(), ','));
    }

    static void CheckDivModCodegen(const std::string& funcName)
    {
        auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX +
                                                                    HIDDEN_FUNC_SUFFIX);
        ASSERT_NE(function, nullptr);
        // Two inputs (dividend/divisor) and two outputs (quotient/remainder).
        EXPECT_EQ(function->GetIncast().size(), 2U);
        EXPECT_EQ(function->GetOutcast().size(), 2U);
        const std::string code = GenCodeByFunction(*function);
        // DivMod is a fused tile op: quotient, remainder, src0, src1 and the
        // temporary buffer are all passed to a single TDivMod call.
        CheckStringExist(R"!!!(TDivMod()!!!", code);
        EXPECT_EQ(CountDivModParams(code), 5U);
    }

    static void RunDivModCodegenCase(DataType dtype, NPUArch arch, const std::string& funcName)
    {
        Platform::Instance().GetSoc().SetNPUArch(arch);
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
        config::SetHostOption(COMPILE_STAGE, CS_CODEGEN_INSTRUCTION);
        TileShape::Current().SetVecTile(8, 64);

        Tensor dividend(dtype, {8, 64}, "dividend");
        Tensor divisor(dtype, {8, 64}, "divisor");
        Tensor quotient(dtype, {8, 64}, "quotient");
        Tensor remainder(dtype, {8, 64}, "remainder");

        FUNCTION(funcName, {dividend, divisor}, {quotient, remainder})
        {
            LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
            {
                (void)i;
                auto result = DivMod(dividend, divisor);
                quotient = std::get<0>(result);
                remainder = std::get<1>(result);
            }
        }
        CheckDivModCodegen(funcName);
    }
};

TEST_F(TestCodegenDynDivMod, TestDivModInt32)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    config::SetHostOption(COMPILE_STAGE, CS_CODEGEN_INSTRUCTION);
    TileShape::Current().SetVecTile(8, 64);

    Tensor dividend(DT_INT32, {8, 64}, "dividend");
    Tensor divisor(DT_INT32, {8, 64}, "divisor");
    Tensor quotient(DT_INT32, {8, 64}, "quotient");
    Tensor remainder(DT_INT32, {8, 64}, "remainder");

    const std::string funcName = "DivModInt32";
    FUNCTION(funcName, {dividend, divisor}, {quotient, remainder})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            auto result = DivMod(dividend, divisor);
            quotient = std::get<0>(result);
            remainder = std::get<1>(result);
        }
    }

    CheckDivModCodegen(funcName);
}

TEST_F(TestCodegenDynDivMod, TestDivModInt16)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_2201);
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    config::SetHostOption(COMPILE_STAGE, CS_CODEGEN_INSTRUCTION);
    TileShape::Current().SetVecTile(8, 64);

    Tensor dividend(DT_INT16, {8, 64}, "dividend");
    Tensor divisor(DT_INT16, {8, 64}, "divisor");
    Tensor quotient(DT_INT16, {8, 64}, "quotient");
    Tensor remainder(DT_INT16, {8, 64}, "remainder");

    const std::string funcName = "DivModInt16";
    FUNCTION(funcName, {dividend, divisor}, {quotient, remainder})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            auto result = DivMod(dividend, divisor);
            quotient = std::get<0>(result);
            remainder = std::get<1>(result);
        }
    }

    CheckDivModCodegen(funcName);
}

TEST_F(TestCodegenDynDivMod, TestDivModUint32NativeOnA5)
{
    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    config::SetHostOption(COMPILE_STAGE, CS_CODEGEN_INSTRUCTION);
    TileShape::Current().SetVecTile(8, 64);

    Tensor dividend(DT_UINT32, {8, 64}, "dividend");
    Tensor divisor(DT_UINT32, {8, 64}, "divisor");
    Tensor quotient(DT_UINT32, {8, 64}, "quotient");
    Tensor remainder(DT_UINT32, {8, 64}, "remainder");

    const std::string funcName = "DivModUint32";
    FUNCTION(funcName, {dividend, divisor}, {quotient, remainder})
    {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            auto result = DivMod(dividend, divisor);
            quotient = std::get<0>(result);
            remainder = std::get<1>(result);
        }
    }

    CheckDivModCodegen(funcName);
}

TEST_F(TestCodegenDynDivMod, TestDivModInt8) { RunDivModCodegenCase(DT_INT8, NPUArch::DAV_2201, "DivModInt8"); }

TEST_F(TestCodegenDynDivMod, TestDivModUint8) { RunDivModCodegenCase(DT_UINT8, NPUArch::DAV_2201, "DivModUint8"); }

TEST_F(TestCodegenDynDivMod, TestDivModUint16) { RunDivModCodegenCase(DT_UINT16, NPUArch::DAV_2201, "DivModUint16"); }

TEST_F(TestCodegenDynDivMod, TestDivModInt64) { RunDivModCodegenCase(DT_INT64, NPUArch::DAV_3510, "DivModInt64"); }

TEST_F(TestCodegenDynDivMod, TestDivModUint64) { RunDivModCodegenCase(DT_UINT64, NPUArch::DAV_3510, "DivModUint64"); }

} // namespace npu::tile_fwk
