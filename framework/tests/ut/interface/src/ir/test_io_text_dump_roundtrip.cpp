/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 * \file test_text_roundtrip.cpp
 * \brief Round-trip tests for IRTextDumper / IRTextLoader.
 */

#include "gtest/gtest.h"

#include <any>
#include <cstdint>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

#include <map>

#include "core/dtype.h"
#include "core/error.h"
#include "tilefwk/tile_shape.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/memref.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/span.h"
#include "ir/stmt.h"
#include "ir/transforms/io_text.h"
#include "interface/configs/config_manager_ng.h"
#include "interface/tensor/logical_tensor.h"
#include "ir/type.h"
#include "tilefwk/symbolic_scalar.h"
#include "test_ir.h"

namespace pypto {
namespace ir {

static TypePtr Scalar(DataType dt) { return std::make_shared<ScalarType>(dt); }
static Span Sp() { return Span("test", 1, 1); }

static VarPtr Var_(const std::string& name, DataType dt = DataType::INT32)
{
    return std::make_shared<Var>(name, Scalar(dt), Sp());
}

class IoTextDumpRoundTripTest : public testing::Test {};

TEST_F(IoTextDumpRoundTripTest, MutexTileTypePreservesCount)
{
    for (uint32_t count : {0U, 2U}) {
        auto type = std::make_shared<MutexTileType>(
            TileType(std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, Sp()),
                                          std::make_shared<ConstInt>(8, DataType::INT64, Sp())},
                     DataType::FP32),
            count);
        std::string error;
        const auto text = TextDumpType(type);
        auto loaded = TextLoadType(text, error);
        ASSERT_TRUE(error.empty()) << error;
        ASSERT_NE(loaded, nullptr);
        auto mutex_type = std::dynamic_pointer_cast<const MutexTileType>(loaded);
        ASSERT_NE(mutex_type, nullptr);
        EXPECT_EQ(mutex_type->mutexCount_, count);
        EXPECT_EQ(TextDumpType(loaded), text);
    }
}

// ---- Expression round-trip ----
TEST_F(IoTextDumpRoundTripTest, TestExprRoundTrip)
{
    std::string error;
    auto a = std::make_shared<ConstInt>(42, DataType::INT32, Sp());
    EXPECT_EQ(TextDump(a), "42");
    EXPECT_EQ(TextDump(TextLoadExpr("42", error)), "42");
    EXPECT_TRUE(error.empty());

    auto b = std::make_shared<ConstFloat>(3.5, DataType::FP32, Sp());
    EXPECT_EQ(TextDump(b), "3.5");

    EXPECT_EQ(TextDump(std::make_shared<ConstBool>(true, Sp())), "true");
    EXPECT_EQ(TextDump(std::make_shared<ConstBool>(false, Sp())), "false");

    // Binary: (1 add 2)
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto two = std::make_shared<ConstInt>(2, DataType::INT32, Sp());
    auto add = std::make_shared<Add>(one, two, DataType::INT32, Sp());
    EXPECT_EQ(TextDump(add), "(1 add 2)");
    auto loaded = TextLoadExpr("(1 add 2)", error);
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(TextDump(loaded), "(1 add 2)");

    // Unary: (neg 5)
    auto five = std::make_shared<ConstInt>(5, DataType::INT32, Sp());
    auto neg = std::make_shared<Neg>(five, DataType::INT32, Sp());
    EXPECT_EQ(TextDump(neg), "(neg 5)");
    EXPECT_EQ(TextDump(TextLoadExpr("(neg 5)", error)), "(neg 5)");
    EXPECT_TRUE(error.empty());

    // Cast: (cast float 1)
    auto cast = std::make_shared<Cast>(one, DataType::FP32, Sp());
    EXPECT_EQ(TextDump(cast), "(cast float 1)");
    EXPECT_EQ(TextDump(TextLoadExpr("(cast float 1)", error)), "(cast float 1)");
    EXPECT_TRUE(error.empty());

    // MakeTuple: tuple(1, 2)
    auto tup = std::make_shared<MakeTuple>(std::vector<ExprPtr>{one, two}, Sp());
    EXPECT_EQ(TextDump(tup), "tuple(1, 2)");
    EXPECT_EQ(TextDump(TextLoadExpr("tuple(1, 2)", error)), "tuple(1, 2)");
    EXPECT_TRUE(error.empty());

    // GetItemExpr: getitem(tuple(1, 2), 0)
    auto idx = std::make_shared<ConstInt>(0, DataType::INDEX, Sp());
    auto gi = std::make_shared<GetItemExpr>(tup, idx, Sp());
    EXPECT_EQ(TextDump(gi), "getitem(tuple(1, 2), 0)");
}

// ---- Type round-trip ----
TEST_F(IoTextDumpRoundTripTest, TestTypeRoundTrip)
{
    std::string error;
    EXPECT_EQ(TextDumpType(Scalar(DataType::FP32)), "float");
    EXPECT_EQ(TextDumpType(TextLoadType("float", error)), "float");
    EXPECT_TRUE(error.empty());

    EXPECT_EQ(TextDumpType(GetUnknownType()), "unknown");
    EXPECT_EQ(TextDumpType(GetTokenType()), "token");
    EXPECT_EQ(TextDumpType(GetNoneType()), "none");

    EXPECT_EQ(TextDumpType(std::make_shared<PtrType>(DataType::FP16)), "ptr<half>");
    EXPECT_EQ(TextDumpType(TextLoadType("ptr<half>", error)), "ptr<half>");
    EXPECT_TRUE(error.empty());

    auto tensor = std::make_shared<TensorType>(
        std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, Sp()),
                             std::make_shared<ConstInt>(8, DataType::INT64, Sp())},
        DataType::FP32);
    EXPECT_EQ(TextDumpType(tensor), "tensor<4 x 8, float, tensor_view<>>");
    EXPECT_EQ(TextDumpType(TextLoadType("tensor<4 x 8, float, tensor_view<>>", error)),
              "tensor<4 x 8, float, tensor_view<>>");
    EXPECT_TRUE(error.empty());
}

// ---- Statement round-trip ----
TEST_F(IoTextDumpRoundTripTest, TestStmtRoundTrip)
{
    std::string error;
    // AssignStmt: int32_t %x = 42;
    auto x = Var_("x");
    auto assign = std::make_shared<AssignStmt>(x, std::make_shared<ConstInt>(42, DataType::INT32, Sp()), Sp());
    std::string dumped = TextDump(assign);
    EXPECT_NE(dumped.find("int32_t %x = 42;"), std::string::npos);

    // Round-trip
    auto loaded = TextLoadStmt(dumped, error);
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(TextDump(loaded), dumped);

    // ReturnStmt
    auto ret = std::make_shared<ReturnStmt>(std::vector<ExprPtr>{std::make_shared<ConstInt>(1, DataType::INT32, Sp())},
                                            Sp());
    std::string retDumped = TextDump(ret);
    EXPECT_NE(retDumped.find("return"), std::string::npos);
    auto retLoaded = TextLoadStmt(retDumped, error);
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(TextDump(retLoaded), retDumped);

    // YieldStmt
    auto yld = std::make_shared<YieldStmt>(std::vector<ExprPtr>{std::make_shared<ConstInt>(1, DataType::INT32, Sp())},
                                           Sp());
    std::string yldDumped = TextDump(yld);
    EXPECT_NE(yldDumped.find("yield"), std::string::npos);

    // BreakStmt / ContinueStmt
    EXPECT_EQ(TextDump(std::make_shared<BreakStmt>(Sp())), "break;");
    EXPECT_EQ(TextDump(std::make_shared<ContinueStmt>(Sp())), "continue;");
    EXPECT_EQ(TextDump(TextLoadStmt("break;", error)), "break;");
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(TextDump(TextLoadStmt("continue;", error)), "continue;");
    EXPECT_TRUE(error.empty());
}

// ---- For loop round-trip ----
TEST_F(IoTextDumpRoundTripTest, TestForLoopRoundTrip)
{
    std::string error;
    auto i = Var_("i");
    auto zero = std::make_shared<ConstInt>(0, DataType::INT32, Sp());
    auto ten = std::make_shared<ConstInt>(10, DataType::INT32, Sp());
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, Sp());
    auto body = std::make_shared<AssignStmt>(Var_("x"), one, Sp());
    auto forStmt = std::make_shared<ForStmt>(i, zero, ten, one, std::vector<IterArgPtr>{}, body, std::vector<VarPtr>{},
                                             Sp());
    std::string dumped = TextDump(forStmt);
    EXPECT_NE(dumped.find("for"), std::string::npos);
    EXPECT_NE(dumped.find("inrange"), std::string::npos);

    auto loaded = TextLoadStmt(dumped, error);
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(TextDump(loaded), dumped);
}

namespace {

const std::any* FindAttr(const TensorOpStmt& op, const std::string& key)
{
    for (const auto& item : op.attrs_) {
        if (item.first == key) {
            return &item.second;
        }
    }
    return nullptr;
}

std::string ConfigText(const std::string& key, const std::string& value)
{
    return "OP_CONFIG() token() #" + key + "(" + value + ");";
}
} // namespace

TEST_F(IoTextDumpRoundTripTest, TestStringAttrRoundTrip)
{
    const std::vector<std::pair<std::string, std::string>> cases = {
        {R"("")", ""},         {R"("hello world")", "hello world"},
        {R"("true")", "true"}, {R"("false")", "false"},
        {R"("3.14")", "3.14"}, {R"("#x([1]);")", "#x([1]);"},
        {R"("\\n")", "\\n"},   {R"("x\0y")", std::string("x\0y", 3)},
        {R"("123")", "123"},   {R"("if")", "if"},
        {R"("中文")", "中文"}, {R"("a\"b\\c\n\r\t\0")", std::string("a\"b\\c\n\r\t\0", 9)},
        {"DEFAULT", "DEFAULT"}};
    for (const auto& [literal, expected] : cases) {
        SCOPED_TRACE(literal);
        std::string error;
        const auto text = ConfigText("label", literal);
        auto original = std::make_shared<TensorOpStmt>(
            std::vector<VarPtr>{}, std::vector<VarPtr>{}, "OP_CONFIG", std::vector<ExprPtr>{}, std::vector<VarPtr>{},
            std::vector<std::pair<std::string, std::any>>{{"label", expected}}, Sp());
        EXPECT_EQ(TextDump(original), text);
        auto loaded = TextLoadStmt(text, error);
        ASSERT_TRUE(error.empty()) << error;
        auto op = std::dynamic_pointer_cast<const TensorOpStmt>(loaded);
        ASSERT_NE(op, nullptr);
        const auto* value = FindAttr(*op, "label");
        ASSERT_NE(value, nullptr);
        ASSERT_EQ(value->type(), typeid(std::string));
        EXPECT_EQ(std::any_cast<std::string>(*value), expected);
        EXPECT_EQ(TextDump(loaded), text);
    }
    auto config = std::make_shared<TensorOpStmt>(
        std::vector<VarPtr>{}, std::vector<VarPtr>{}, "OP_CONFIG", std::vector<ExprPtr>{}, std::vector<VarPtr>{},
        std::vector<std::pair<std::string, std::any>>{{"labels", std::vector<std::string>{"", "a b", "true"}}}, Sp());
    const auto text = ConfigText("labels", R"(["", "a b", "true"])");
    EXPECT_EQ(TextDump(config), text);
    std::string error;
    auto loaded = TextLoadStmt(text, error);
    ASSERT_NE(loaded, nullptr) << error;
    EXPECT_EQ(TextDump(loaded), text);
}

TEST_F(IoTextDumpRoundTripTest, TestStringMapEscapedKeys)
{
    const std::map<std::string, int64_t> values{{"", 1}, {"true", 2}, {"a\"b\\c", 3}, {std::string("x\0y", 3), 4}};
    auto config = std::make_shared<TensorOpStmt>(
        std::vector<VarPtr>{}, std::vector<VarPtr>{}, "OP_CONFIG", std::vector<ExprPtr>{}, std::vector<VarPtr>{},
        std::vector<std::pair<std::string, std::any>>{{"values", values}}, Sp());
    const auto text = TextDump(config);
    EXPECT_EQ(text, ConfigText("values", R"([["", 1], ["a\"b\\c", 3], ["true", 2], ["x\0y", 4]])"));
    std::string error;
    auto loaded = std::dynamic_pointer_cast<const TensorOpStmt>(TextLoadStmt(text, error));
    ASSERT_NE(loaded, nullptr) << error;
    const auto* attr = FindAttr(*loaded, "values");
    ASSERT_NE(attr, nullptr);
    const auto* decoded = std::any_cast<std::map<std::string, int64_t>>(attr);
    ASSERT_NE(decoded, nullptr);
    EXPECT_EQ(*decoded, values);
}

TEST_F(IoTextDumpRoundTripTest, TestInvalidStringAttr)
{
    for (const auto& literal :
         {R"("unterminated)", R"("bad\q")", R"("bad\x41")", R"("trailing\)", "\"raw\nnewline\"", "\"raw\rreturn\""}) {
        std::string error;
        EXPECT_EQ(TextLoadStmt(ConfigText("label", literal), error), nullptr);
        EXPECT_FALSE(error.empty());
    }
}

TEST_F(IoTextDumpRoundTripTest, TestConfigOpRoundTrip)
{
    std::vector<std::pair<std::string, std::any>> attrs{
        {"vec_tile_shapes", std::vector<int64_t>{16, 32}},
        {"host.compile_timeout", int64_t{1000}},
        {"pass.cube_nbuffer_setting", std::map<int64_t, int64_t>{{-1, 2}, {3, 4}}},
        {"pass.cube_nbuffer_setting_by_func", std::map<std::string, int64_t>{{"DEFAULT", 4}, {"func8_0", 1}}}};
    auto config = std::make_shared<TensorOpStmt>(std::vector<VarPtr>{}, std::vector<VarPtr>{}, "OP_CONFIG",
                                                 std::vector<ExprPtr>{}, std::vector<VarPtr>{}, attrs, Sp());
    std::string dumped = TextDump(config);
    EXPECT_EQ(dumped.find("OP_CONFIG() token()"), 0);
    EXPECT_NE(dumped.find("#pass.cube_nbuffer_setting([[-1, 2], [3, 4]])"), std::string::npos);
    EXPECT_NE(dumped.find("#pass.cube_nbuffer_setting_by_func([[\"DEFAULT\", 4], [\"func8_0\", 1]])"),
              std::string::npos);
    std::string error;
    auto loaded = TextLoadStmt(dumped, error);
    ASSERT_TRUE(error.empty()) << error;
    auto op = std::dynamic_pointer_cast<const TensorOpStmt>(loaded);
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->opcode_, "OP_CONFIG");
    EXPECT_TRUE(op->args_.empty());
    EXPECT_TRUE(op->result_.empty());
    EXPECT_TRUE(op->tokens_.empty());
    EXPECT_TRUE(op->result_token_.empty());
    const std::any* intSetting = FindAttr(*op, "pass.cube_nbuffer_setting");
    const std::any* stringSetting = FindAttr(*op, "pass.cube_nbuffer_setting_by_func");
    ASSERT_NE(intSetting, nullptr);
    ASSERT_NE(stringSetting, nullptr);
    EXPECT_EQ(intSetting->type(), typeid(std::map<int64_t, int64_t>));
    EXPECT_EQ(stringSetting->type(), typeid(std::map<std::string, int64_t>));
    EXPECT_EQ(TextDump(loaded), dumped);
}

TEST_F(IoTextDumpRoundTripTest, TestConfigMapDuplicateKeys)
{
    std::string error;
    const std::string text = "OP_CONFIG() token() #ints([[3, 4], [-1, 2], [3, 8]]) "
                             "#strings([[\"func8_0\", 1], [\"DEFAULT\", 4], [\"func8_0\", 2]]);";
    auto loaded = TextLoadStmt(text, error);
    ASSERT_TRUE(error.empty()) << error;
    auto op = std::dynamic_pointer_cast<const TensorOpStmt>(loaded);
    ASSERT_NE(op, nullptr);
    const auto* ints = FindAttr(*op, "ints");
    const auto* strings = FindAttr(*op, "strings");
    ASSERT_NE(ints, nullptr);
    ASSERT_NE(strings, nullptr);
    using IntMap = std::map<int64_t, int64_t>;
    using StringMap = std::map<std::string, int64_t>;
    const auto* intMap = std::any_cast<IntMap>(ints);
    const auto* stringMap = std::any_cast<StringMap>(strings);
    ASSERT_NE(intMap, nullptr);
    ASSERT_NE(stringMap, nullptr);
    EXPECT_EQ(*intMap, (IntMap{{-1, 2}, {3, 8}}));
    EXPECT_EQ(*stringMap, (StringMap{{"DEFAULT", 4}, {"func8_0", 2}}));
    EXPECT_EQ(TextDump(loaded), "OP_CONFIG() token() #ints([[-1, 2], [3, 8]]) "
                                "#strings([[\"DEFAULT\", 4], [\"func8_0\", 2]]);");
}

TEST_F(IoTextDumpRoundTripTest, TestConfigListsRemainVectors)
{
    // Empty, flat, malformed-pair and non-integer-pair lists must remain vectors.
    for (const auto& value :
         {"[]", "[16, 32]", "[[1, 2], [3]]", "[[1, true]]", "[[\"a b\", true]]", "[[\"a b\", 1], [2, 3]]"}) {
        SCOPED_TRACE(value);
        std::string error;
        const auto text = ConfigText("values", value);
        auto loaded = TextLoadStmt(text, error);
        ASSERT_TRUE(error.empty()) << error;
        auto op = std::dynamic_pointer_cast<const TensorOpStmt>(loaded);
        ASSERT_NE(op, nullptr);
        const auto* attr = FindAttr(*op, "values");
        ASSERT_NE(attr, nullptr);
        ASSERT_EQ(attr->type(), typeid(std::vector<std::any>));
        EXPECT_EQ(TextDump(loaded), text);
    }
}

TEST_F(IoTextDumpRoundTripTest, TestRejectsNestedAttrValues)
{
    for (const auto& value : {"4", "true", "[1, 2]"}) {
        SCOPED_TRACE(value);
        std::string error;
        auto loaded = TextLoadStmt(ConfigText("settings", "#DEFAULT(" + std::string(value) + ")"), error);
        EXPECT_EQ(loaded, nullptr);
        EXPECT_FALSE(error.empty());
    }
}

TEST_F(IoTextDumpRoundTripTest, TestTileConfigOpRoundTrip)
{
    CubeTile cube;
    cube.m = {128, 128};
    cube.k = {64, 64, 32};
    cube.n = {128, 128};
    cube.enableSplitK = true;

    ConvTile conv;
    conv.tileL1Info = {1, 1, 64, 64, 8, 8, 32, 1};
    conv.setL0Tile = false;

    ConvBpTile convBp;
    convBp.tileL1Info = {16, 32, 64};
    convBp.tileL0Info = {8, 16, 32};

    DistTile dist;
    dist.row = {1, 2, 3};
    dist.col = {4, 5, 6};
    dist.rank = {7, 8, 9};
    dist.rankId = 0;

    std::vector<std::pair<std::string, std::any>> attrs{
        {"cube_tile_shapes", cube},
        {"conv_tile_shapes", conv},
        {"convbp_tile_shapes", convBp},
        {"dist_tile_shapes", dist},
    };
    auto config = std::make_shared<TensorOpStmt>(std::vector<VarPtr>{}, std::vector<VarPtr>{}, "OP_CONFIG",
                                                 std::vector<ExprPtr>{}, std::vector<VarPtr>{}, attrs, Sp());
    const std::string dumped = TextDump(config);
    EXPECT_NE(dumped.find("#cube_tile_shapes([128, 128, 64, 64, 32, 128, 128, 1])"), std::string::npos);
    EXPECT_NE(dumped.find("#conv_tile_shapes([1, 1, 64, 64, 8, 8, 32, 1, 0, 0, 0, 0, 0])"), std::string::npos);
    EXPECT_NE(dumped.find("#convbp_tile_shapes([16, 32, 64, 8, 16, 32])"), std::string::npos);
    EXPECT_NE(dumped.find("#dist_tile_shapes([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])"), std::string::npos);
    EXPECT_EQ(dumped.find("CubeTile:"), std::string::npos);
    EXPECT_EQ(dumped.find("ConvTile:"), std::string::npos);
    EXPECT_EQ(dumped.find("ConvBpTile:"), std::string::npos);
    EXPECT_EQ(dumped.find("DistTile:"), std::string::npos);

    std::string error;
    auto loaded = TextLoadStmt(dumped, error);
    ASSERT_TRUE(error.empty()) << error;
    EXPECT_EQ(TextDump(loaded), dumped);
    auto op = std::dynamic_pointer_cast<const TensorOpStmt>(loaded);
    ASSERT_NE(op, nullptr);

    // Field layout is tested by IoTextTileTest; here verify typed IR restoration.
    const auto* cubeValue = FindAttr(*op, "cube_tile_shapes");
    const auto* convValue = FindAttr(*op, "conv_tile_shapes");
    const auto* convBpValue = FindAttr(*op, "convbp_tile_shapes");
    const auto* distValue = FindAttr(*op, "dist_tile_shapes");
    ASSERT_NE(cubeValue, nullptr);
    ASSERT_NE(convValue, nullptr);
    ASSERT_NE(convBpValue, nullptr);
    ASSERT_NE(distValue, nullptr);
    EXPECT_EQ(cubeValue->type(), typeid(CubeTile));
    EXPECT_EQ(convValue->type(), typeid(ConvTile));
    EXPECT_EQ(convBpValue->type(), typeid(ConvBpTile));
    EXPECT_EQ(distValue->type(), typeid(DistTile));
}

TEST_F(IoTextDumpRoundTripTest, TestTileDecodeErrors)
{
    std::string error;
    auto missing = TextLoadStmt(ConfigText("cube_tile_shapes", "[1, 2, 3, 4, 5, 6, 7]"), error);
    EXPECT_EQ(missing, nullptr);
    EXPECT_NE(error.find("cube_tile_shapes: CubeTile: expected 8 int64 values, got 7"), std::string::npos);
    EXPECT_EQ(error.find("cube_tile_shapes: cube_tile_shapes:"), std::string::npos);

    error.clear();
    auto badBool = TextLoadStmt(ConfigText("cube_tile_shapes", "[1, 2, 3, 4, 5, 6, 7, 2]"), error);
    EXPECT_EQ(badBool, nullptr);
    EXPECT_NE(error.find("cube_tile_shapes: CubeTile: enableSplitK must be 0 or 1, got 2"), std::string::npos);

    error.clear();
    auto badL0Flag = TextLoadStmt(ConfigText("conv_tile_shapes", "[1, 1, 64, 64, 8, 8, 32, 1, 0, 0, 0, 0, 2]"), error);
    EXPECT_EQ(badL0Flag, nullptr);
    EXPECT_NE(error.find("conv_tile_shapes: ConvTile: setL0Tile must be 0 or 1, got 2"), std::string::npos);

    error.clear();
    auto nonInteger = TextLoadStmt(ConfigText("cube_tile_shapes", "[1, 1, 1, 1, 1, 1, 1, true]"), error);
    EXPECT_EQ(nonInteger, nullptr);
    EXPECT_NE(error.find("cube_tile_shapes: expected int64 at index 7"), std::string::npos);
    error.clear();
    EXPECT_EQ(TextLoadStmt(ConfigText("conv_tile_shapes", "true"), error), nullptr);
    EXPECT_NE(error.find("conv_tile_shapes: expected an integer list"), std::string::npos);

    error.clear();
    EXPECT_EQ(TextLoadProgram("program #entry() #vec_tile(16) {\n}\n", error), nullptr);
    EXPECT_NE(error.find("vec_tile: expected a list of int64 values"), std::string::npos) << error;

    error.clear();
    EXPECT_EQ(TextLoadProgram("program #entry() #cube_tile([1, 2, 3]) {\n}\n", error), nullptr);
    EXPECT_NE(error.find("CubeTile: expected 8 int64 values, got 3"), std::string::npos) << error;
}

// ---- Function round-trip ----
TEST_F(IoTextDumpRoundTripTest, TestFunctionRoundTrip)
{
    std::string error;
    auto x = Var_("x");
    auto body = std::make_shared<ReturnStmt>(
        std::vector<ExprPtr>{std::make_shared<ConstInt>(42, DataType::INT32, Sp())}, Sp());
    auto func = std::make_shared<Function>("f", std::vector<VarPtr>{x}, std::vector<TypePtr>{Scalar(DataType::INT32)},
                                           body, Sp());
    std::string dumped = TextDump(func);
    EXPECT_NE(dumped.find("function f"), std::string::npos);
    EXPECT_NE(dumped.find("incast("), std::string::npos);
    EXPECT_NE(dumped.find("outcast("), std::string::npos);

    auto loaded = TextLoadFunction(dumped, error);
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(TextDump(loaded), dumped);
}

// ---- Integer literal in a scalar position keeps its RawSymbolic* identity ----
TEST_F(IoTextDumpRoundTripTest, TestIntLiteralInScalarPositionKeepsRawSymbolicIdentity)
{
    std::string error;
    auto expr = TextLoadExpr("42", error);
    EXPECT_TRUE(error.empty());
    ASSERT_NE(expr, nullptr);
    EXPECT_NE(std::dynamic_pointer_cast<const npu::tile_fwk::RawSymbolicImmediate>(expr), nullptr);
}

// ---- For-stmt attrs keep the element types that consumers read with GetAttr ----
TEST_F(IoTextDumpRoundTripTest, TestForLoopAttrsKeepDeclaredTypes)
{
    std::string error;
    auto stmt = TextLoadStmt("for %i inrange 0, 4, 1 iter { } #_config_scope(Unsupported) "
                             "#_loop_conds([v0ge(i, 0), v0lt(i, 4)]) #unroll_times(2) { }",
                             error);
    EXPECT_TRUE(error.empty()) << error;
    auto forStmt = std::dynamic_pointer_cast<const ForStmt>(stmt);
    ASSERT_NE(forStmt, nullptr);
    EXPECT_NE(std::dynamic_pointer_cast<const npu::tile_fwk::RawSymbolicSymbol>(forStmt->loopVar_), nullptr);
    EXPECT_EQ(forStmt->GetAttr<std::vector<npu::tile_fwk::SymbolicScalar>>("_loop_conds").size(), 2u);
    EXPECT_EQ(forStmt->GetAttr<int>("unroll_times"), 2);
    EXPECT_NE(forStmt->GetAttr<std::shared_ptr<npu::tile_fwk::ConfigScope>>("_config_scope"), nullptr);
}

// ---- The same name in two functions must not be bound to the same tensor ----
TEST_F(IoTextDumpRoundTripTest, TestSameNamedTensorInTwoFunctionsIsNotShared)
{
    std::string error;
    // The symbol table is per function: %x@0 of g must not be the %x@0 of f (same name and same rawmagic).
    const std::string
        text = "program #entry(f) {\n"
               "    function f incast(v0_logical_tensor %x@0 #dtype(float) #dynvalidshape([4, 8]) #offset([0, 0]) "
               "#shape([4, 8])) outcast() #entry(true) #type(Opaque) {\n"
               "        return %x@0;\n"
               "    }\n"
               "    function g incast(v0_logical_tensor %x@0 #dtype(int32_t) #dynvalidshape([2, 2]) #offset([0, 0]) "
               "#shape([2, 2])) outcast() #entry(false) #type(Opaque) {\n"
               "        return %x@0;\n"
               "    }\n"
               "}";
    auto program = TextLoadProgram(text, error);
    ASSERT_TRUE(error.empty()) << error;
    ASSERT_NE(program, nullptr);
    EXPECT_EQ(TextDump(program), text);

    auto paramF = std::dynamic_pointer_cast<const npu::tile_fwk::LogicalTensor>(
        program->functions_.at("f")->params_.front());
    auto paramG = std::dynamic_pointer_cast<const npu::tile_fwk::LogicalTensor>(
        program->functions_.at("g")->params_.front());
    ASSERT_NE(paramF, nullptr);
    ASSERT_NE(paramG, nullptr);
    EXPECT_NE(paramF.get(), paramG.get());
    EXPECT_EQ(paramF->tensor->rawmagic, 0);
    EXPECT_EQ(paramG->tensor->rawmagic, 0);
    EXPECT_NE(paramF->tensor.get(), paramG->tensor.get());
}

// ---- Loop iter-args: a tensor iter-arg keeps its tensor, and `%none` resolves on reference ----
TEST_F(IoTextDumpRoundTripTest, TestLoopIterArgKeepsTensorIdentity)
{
    std::string error;
    const std::string
        text = "program #entry() {\n"
               "    function foo incast(v0_logical_tensor %x@0 #dtype(float) #dynvalidshape([32, 16]) #offset([0, 0]) "
               "#shape([32, 16]), v0_logical_tensor %out@1 #dtype(float) #dynvalidshape([32, 16]) #offset([0, 0]) "
               "#shape([32, 16])) outcast() #entry(false) #type(Opaque) {\n"
               "        int64_t %i_0, v0_logical_tensor %xv_0@2 #dtype(float) #dynvalidshape([v0min(v0max(v0sub(32, "
               "v0mul(loop_idx_10, 16)), 0), 16), 16]) #offset([0, 0]) #shape([16, 16]) = for %loop_idx_10 inrange 0, "
               "2, 1 iter { none %i = %none; v0_logical_tensor %xv@2 #dtype(float) #dynvalidshape([16, 16]) "
               "#offset([0, 0]) #shape([16, 16]) = %x@0; } #_config_scope(Unsupported) #_loop_conds([v0ge(loop_idx_10, "
               "0), v0lt(loop_idx_10, 2)]) #parallel(false) #submit_before_loop(false) #unroll_times(1) {\n"
               "            v0_logical_tensor %View_x@2 #dtype(float) #dynvalidshape([v0min(v0max(v0sub(32, "
               "v0mul(loop_idx_10, 16)), 0), 16), 16]) #offset([0, 0]) #shape([16, 16]) = !10000 VIEW(%x@0) token() "
               "#fromDynOffset([v0mul(loop_idx_10, 16), 0]) #fromOffset([0, 0]) #toValidShape([v0min(v0max(v0sub(32, "
               "v0mul(loop_idx_10, 16)), 0), 16), 16]);\n"
               "            v0_logical_tensor %out@1 #dtype(float) #dynvalidshape([32, 16]) #offset([0, 0]) "
               "#shape([32, 16]) = !10001 ASSEMBLE(%View_x@2) token() #toDynOffset([v0mul(loop_idx_10, 16), 0]) "
               "#toOffset([0, 0]) #dassemble(true) #parallel(false);\n"
               "            continue %loop_idx_10, %View_x@2;\n"
               "        }\n"
               "        return %x@0, %out@1;\n"
               "    }\n"
               "}";
    auto program = TextLoadProgram(text, error);
    ASSERT_TRUE(error.empty()) << error;
    ASSERT_NE(program, nullptr);
    EXPECT_EQ(TextDump(program), text);

    auto func = program->functions_.at("foo");
    auto forStmt = std::dynamic_pointer_cast<const ForStmt>(func->body_->stmts_.front());
    ASSERT_NE(forStmt, nullptr);
    ASSERT_EQ(forStmt->iterArgs_.size(), 2U);
    EXPECT_EQ(forStmt->returnVars_.size(), 2U);

    // `none %i = %none`: the None sentinel has no var_def and is materialized when referenced.
    auto noneIterAsTensor = std::dynamic_pointer_cast<const npu::tile_fwk::LogicalTensor>(
        forStmt->iterArgs_[0]->iterVar_);
    EXPECT_EQ(noneIterAsTensor, nullptr);
    EXPECT_EQ(forStmt->iterArgs_[0]->iterVar_->name_, "i");
    auto noneInit = std::dynamic_pointer_cast<const Var>(forStmt->iterArgs_[0]->initValue_);
    ASSERT_NE(noneInit, nullptr);
    EXPECT_EQ(noneInit->name_, "none");

    // `v0_logical_tensor %xv@2 = %x@0`: the iter-arg keeps the @2 tensor, its init is the @0 param.
    auto tensorIterVar = std::dynamic_pointer_cast<const npu::tile_fwk::LogicalTensor>(forStmt->iterArgs_[1]->iterVar_);
    ASSERT_NE(tensorIterVar, nullptr);
    ASSERT_NE(tensorIterVar->tensor, nullptr);
    EXPECT_EQ(tensorIterVar->tensor->rawmagic, 2);
    auto tensorInit = std::dynamic_pointer_cast<const npu::tile_fwk::LogicalTensor>(forStmt->iterArgs_[1]->initValue_);
    ASSERT_NE(tensorInit, nullptr);
    ASSERT_NE(tensorInit->tensor, nullptr);
    EXPECT_EQ(tensorInit->tensor->rawmagic, 0);
}

TEST_F(IoTextDumpRoundTripTest, FailedLoadRestoresAmbientTile)
{
    auto scope = npu::tile_fwk::ConfigManagerNg::CurrentScope();
    auto currentVec = [&scope]() { return scope->GenerateTileShape().GetVecTile().tile; };
    const auto saved = currentVec();
    scope->AddValue("vec_tile_shapes", std::vector<int64_t>{32, 32});

    std::string error;
    EXPECT_EQ(TextLoadProgram("program #entry() #vec_tile([16, 16]) not_a_body {\n", error), nullptr);
    EXPECT_EQ(currentVec(), (std::vector<int64_t>{32, 32}));

    scope->AddValue("vec_tile_shapes", saved);
}

TEST_F(IoTextDumpRoundTripTest, ProgramHeaderRestoresCubeAndDistTile)
{
    auto scope = npu::tile_fwk::ConfigManagerNg::CurrentScope();
    const auto saved = scope->GenerateTileShape();

    std::string error;
    auto program = TextLoadProgram(
        "program #entry() #cube_tile([1, 2, 3, 4, 5, 6, 7, 1]) #dist_tile([1, 2, 3, 4, 5, 6, 7, 8, 9, 2]) {\n}\n",
        error);
    EXPECT_NE(program, nullptr) << error;

    const auto tileShape = scope->GenerateTileShape();
    const auto& cubeTile = tileShape.GetCubeTile();
    EXPECT_EQ(cubeTile.m[0], 1);
    EXPECT_EQ(cubeTile.m[1], 2);
    EXPECT_EQ(cubeTile.k[0], 3);
    EXPECT_EQ(cubeTile.k[2], 5);
    EXPECT_EQ(cubeTile.n[0], 6);
    EXPECT_EQ(cubeTile.n[1], 7);
    EXPECT_TRUE(cubeTile.enableSplitK);

    const auto& distTile = tileShape.GetDistTile();
    EXPECT_EQ(distTile.row[0], 1);
    EXPECT_EQ(distTile.col[0], 4);
    EXPECT_EQ(distTile.rank[0], 7);
    EXPECT_EQ(distTile.rankId, 2);

    scope->AddValue("vec_tile_shapes", saved.GetVecTile().tile);
    scope->AddValue("cube_tile_shapes", saved.GetCubeTile());
    scope->AddValue("dist_tile_shapes", saved.GetDistTile());
}

} // namespace ir
} // namespace pypto
