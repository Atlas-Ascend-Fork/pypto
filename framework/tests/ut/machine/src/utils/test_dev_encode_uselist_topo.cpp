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
 * \brief Pred-count sort can place a reader before its writer. useList topo order puts the write first.
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

// 写依赖两个 add，pred 更大；读只依赖这个写，pred 更小。
// callList 按 pred 个数排序，读会排到写前面。useList 按拓扑序填写，写回到读前面。
TEST(UseListTopoEmitOrderTest, PredCountSortPutsReaderBeforeWriter)
{
    Program::GetInstance().Reset();
    config::Reset();
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    config::SetRuntimeOption(STITCH_FUNCTION_MAX_NUM, 64);
    TileShape::Current().SetVecTile(kTile, kTile);

    Tensor a(DT_FP32, {kTile * 2, kTile}, "a");
    Tensor b(DT_FP32, {kTile * 2, kTile}, "b");
    Tensor out(DT_FP32, {kTile * 2, kTile}, "out");
    Tensor scaled(DT_FP32, {kTile * 2, kTile}, "scaled");
    FUNCTION("pred_count_reader_first", {a, b}, {out, scaled})
    {
        LOOP("PV_W", FunctionType::DYNAMIC_LOOP, i, LoopRange(1), std::set<int>{1})
        {
            auto tileA = View(a, {kTile, kTile}, {i * kTile, 0});
            auto tileB = View(b, {kTile, kTile}, {i * kTile, 0});
            auto x = Add(tileA, tileA);
            auto y = Add(tileB, tileB);
            Assemble(Add(x, y), {i * kTile, 0}, out);
            auto written = View(out, {kTile, kTile}, {i * kTile, 0});
            Assemble(Mul(written, Element(DT_FP32, 1.5f)), {i * kTile, 0}, scaled);
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
    ASSERT_EQ(matchCount, 1);

    // operationIdx 是 callList 下标。按下标排回去，就是拓扑排序之前的顺序。
    std::vector<OpMark> callListOrder = useListOrder;
    std::sort(callListOrder.begin(), callListOrder.end(),
              [](const OpMark& lhs, const OpMark& rhs) { return lhs.idx < rhs.idx; });
    constexpr int kReadOp = 2;
    constexpr int kWriteOp = 3;
    // callList：pred count 把读排在写前面。
    EXPECT_EQ(Dump(callListOrder), "R2,W3");
    // useList：拓扑序把写放回读前面。
    EXPECT_EQ(Dump(useListOrder), "W3,R2");
    // 读的 pred 更小，所以上面的乱序来自 pred count。
    EXPECT_LT(devFunc->GetOperationDepGraphPredCount(kReadOp), devFunc->GetOperationDepGraphPredCount(kWriteOp));

    bool writeReachesRead = false;
    const auto& succList = devFunc->GetOperationDepGraphSuccList(kWriteOp);
    for (size_t i = 0; i < succList.size(); ++i) {
        writeReachesRead = writeReachesRead || static_cast<int>(devFunc->At(succList, i)) == kReadOp;
    }
    // W3 是 R2 的前驱，拓扑序必须先写后读。
    EXPECT_TRUE(writeReachesRead);
}
