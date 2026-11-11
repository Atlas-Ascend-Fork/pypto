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
 * \file test_dev_encode_uselist_topo.cpp
 * \brief Pred-count sort can reverse outcast dependencies. useList emits them in dependency order.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include "machine/utils/dynamic/dev_encode_program.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "interface/configs/config_manager_ng.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/data_type.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

namespace {

constexpr int kTile = 32;

struct OpMark {
    int idx{-1};
    char kind{'?'};
};

std::string Dump(const std::vector<OpMark>& ops)
{
    std::string text;
    for (const auto& op : ops) {
        if (!text.empty()) {
            text += ",";
        }
        text += op.kind;
        text += std::to_string(op.idx);
    }
    return text;
}

std::vector<OpMark> Collect(DevAscendFunction* func, const DevLocalVector<DevAscendFunctionCallOperandUse>& list)
{
    std::vector<OpMark> ops;
    ops.reserve(list.size());
    for (size_t i = 0; i < list.size(); ++i) {
        const auto& use = func->At(list, i);
        ops.push_back(OpMark{use.operationIdx, use.opType == CellMatchOpType::READ ? 'R' : 'W'});
    }
    return ops;
}

} // namespace

// 首次写依赖两个 Add；Cube 读取它的计算结果并写 out 的另一个 tile。
// callList 按前驱数量排序，首次写排在后面；useList 必须按依赖顺序记录两次写和读。
TEST(UseListTopoEmitOrderTest, PredCountSortPutsReaderBeforeWriter)
{
    Program::GetInstance().Reset();
    config::Reset();
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    config::SetRuntimeOption(STITCH_FUNCTION_MAX_NUM, 64);
    TileShape::Current().SetVecTile(kTile, kTile);
    TileShape::Current().SetCubeTile({kTile, kTile}, {kTile, kTile}, {kTile, kTile});

    Tensor a(DT_FP16, {kTile * 2, kTile}, "a");
    Tensor b(DT_FP16, {kTile * 2, kTile}, "b");
    Tensor weight(DT_FP16, {kTile, kTile}, "weight");
    Tensor out(DT_FP16, {kTile * 2, kTile}, "out");
    Tensor scaled(DT_FP16, {kTile * 2, kTile}, "scaled");
    FUNCTION("pred_count_reader_first", {a, b, weight}, {out, scaled})
    {
        LOOP("PV_W", FunctionType::DYNAMIC_LOOP, i, LoopRange(1), std::set<int>{1})
        {
            auto tileA = View(a, {kTile, kTile}, {i * kTile, 0});
            auto tileB = View(b, {kTile, kTile}, {i * kTile, 0});
            auto x = Add(tileA, tileA);
            auto y = Add(tileB, tileB);
            auto sum = Add(x, y);
            Assemble(sum, {i * kTile, 0}, out);
            // The Cube writer depends on the first writer through sum, without reading back out.
            Assemble(Matrix::Matmul(DT_FP16, sum, weight), {i * kTile + kTile, 0}, out);
            auto written = View(out, {kTile, kTile}, {i * kTile, 0});
            Assemble(Mul(written, Element(DT_FP16, 1.5f)), {i * kTile, 0}, scaled);
        }
    }

    Function* func = Program::GetInstance().GetLastFunction();
    ASSERT_NE(func, nullptr);
    auto dynAttr = func->GetDyndevAttribute();
    ASSERT_NE(dynAttr, nullptr);
    ASSERT_FALSE(dynAttr->devProgBinary.empty());
    auto* devProg = reinterpret_cast<DevAscendProgram*>(dynAttr->devProgBinary.data());
    ASSERT_NE(devProg, nullptr);
    devProg->RelocProgram(0, reinterpret_cast<uint64_t>(devProg));

    // producerConsumerList 的填写顺序就是 useList 拓扑序。同时含写和读的表只该有一条。
    DevAscendFunction* devFunc = nullptr;
    std::vector<OpMark> useListOrder;
    int matchCount = 0;
    for (int fi = 0; fi < static_cast<int>(devProg->GetFunctionSize()); ++fi) {
        DevAscendFunction* cur = devProg->GetFunction(fi);
        if (cur == nullptr) {
            continue;
        }
        for (size_t oi = 0; oi < cur->GetOutcastSize(); ++oi) {
            auto& outcast = cur->GetOutcast(static_cast<int>(oi));
            std::vector<OpMark> ops = Collect(cur, outcast.producerConsumerList);
            bool hasW = false;
            bool hasR = false;
            for (const auto& op : ops) {
                hasW = hasW || op.kind == 'W';
                hasR = hasR || op.kind == 'R';
            }
            if (hasW && hasR) {
                ++matchCount;
                devFunc = cur;
                useListOrder = std::move(ops);
            }
        }
    }
    ASSERT_EQ(matchCount, 1) << "expected exactly one outcast used across subgraphs for both writes and reads";

    // operationIdx 是 callList 下标；useList 按拓扑序填写时不修改这些下标。
    std::vector<OpMark> callListOrder = useListOrder;
    std::sort(callListOrder.begin(), callListOrder.end(),
              [](const OpMark& lhs, const OpMark& rhs) { return lhs.idx < rhs.idx; });
    constexpr int kReadOp = 2;
    constexpr int kSecondWriteOp = 3;
    constexpr int kFirstWriteOp = 4;
    EXPECT_EQ(Dump(callListOrder), "R2,W3,W4");
    EXPECT_EQ(Dump(useListOrder), "W4,W3,R2");
    EXPECT_LT(devFunc->GetOperationDepGraphPredCount(kReadOp), devFunc->GetOperationDepGraphPredCount(kFirstWriteOp));
    EXPECT_LT(devFunc->GetOperationDepGraphPredCount(kSecondWriteOp),
              devFunc->GetOperationDepGraphPredCount(kFirstWriteOp));

    auto hasSuccessor = [devFunc](int opIdx, int successorIdx) {
        const auto& successors = devFunc->GetOperationDepGraphSuccList(opIdx);
        for (size_t i = 0; i < successors.size(); ++i) {
            if (static_cast<int>(devFunc->At(successors, i)) == successorIdx) {
                return true;
            }
        }
        return false;
    };
    EXPECT_TRUE(hasSuccessor(kFirstWriteOp, kSecondWriteOp));
    EXPECT_TRUE(hasSuccessor(kSecondWriteOp, kReadOp));
}
