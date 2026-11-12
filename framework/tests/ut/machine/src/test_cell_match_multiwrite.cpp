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
 * \file test_cell_match_multiwrite.cpp
 * \brief Multi-writer cell match judged from encoded operator kernels.
 */

#include <vector>
#include <gtest/gtest.h>
#include "interface/configs/config_manager.h"
#include "interface/configs/config_manager_ng.h"
#include "interface/program/program.h"
#include "machine/utils/dynamic/dev_encode_program.h"
#include "machine/utils/dynamic/dev_stitch_dependency_enhanced.h"
#include "tilefwk/data_type.h"
#include "tilefwk/tilefwk_op.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;
using npu::tile_fwk::MakeTaskID;

namespace {

constexpr uint64_t kTag = 7;
constexpr int kHidden = 32;

struct EncodedOutcast {
    DevAscendProgram* prog{nullptr};
    DevAscendProgramUpdate* update{nullptr};
    DevAscendFunction* func{nullptr};
    uint32_t funcIdx{0};
    DevAscendFunctionOutcast* outcast{nullptr};
};

struct ReadAccess {
    uint32_t funcIdx{0};
    uint64_t offset[DEV_SHAPE_DIM_MAX]{};
    uint64_t shape[DEV_SHAPE_DIM_MAX]{};
};

void PrepareTailKernel(int tileRow, int tileCol)
{
    Program::GetInstance().Reset();
    config::Reset();
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    config::SetRuntimeOption(STITCH_FUNCTION_MAX_NUM, 64);
    ConfigManagerNg::SetGlobalConfig("compute_determinism_level", static_cast<int64_t>(0));
    TileShape::Current().SetVecTile(tileRow, tileCol);
}

bool BindOutcast(DevAscendProgram* prog, DevAscendProgramUpdate* update, EncodedOutcast* slot)
{
    for (uint32_t funcIdx = 0; funcIdx < prog->GetFunctionSize(); ++funcIdx) {
        DevAscendFunction* func = prog->GetFunction(static_cast<int>(funcIdx));
        int outcastIdx = func->LookupOutcastBySlotIndex(update->slotIndex);
        if (outcastIdx == INVALID_INDEX) {
            continue;
        }
        DevAscendFunctionOutcast& outcast = func->GetOutcast(outcastIdx);
        if (outcast.producerConsumerList.size() == 0) {
            continue;
        }
        if (slot->outcast != nullptr &&
            outcast.producerConsumerList.size() <= slot->outcast->producerConsumerList.size()) {
            continue;
        }
        slot->prog = prog;
        slot->update = update;
        slot->func = func;
        slot->funcIdx = funcIdx;
        slot->outcast = &outcast;
    }
    return slot->outcast != nullptr;
}

bool FindEncoded(int cellDim0, uint32_t normalWriteCap, uint32_t atomicWriteCap, EncodedOutcast* slot)
{
    Function* func = Program::GetInstance().GetLastFunction();
    if (func == nullptr || func->GetDyndevAttribute() == nullptr) {
        return false;
    }
    auto* prog = reinterpret_cast<DevAscendProgram*>(func->GetDyndevAttribute()->devProgBinary.data());
    if (prog == nullptr) {
        return false;
    }
    prog->RelocProgram(0, reinterpret_cast<uint64_t>(prog));
    *slot = EncodedOutcast{};
    for (size_t i = 0; i < prog->updateList.size(); ++i) {
        auto& update = prog->At(prog->updateList, i);
        if (!update.isPartial || update.stitchCtrlBitMask == STITCH_CTRL_NONE) {
            continue;
        }
        const DevCellMatchTableDesc& desc = update.cellMatchTableDesc;
        if (desc.GetDimensionSize() < 1 || desc.GetCellShape(0) != cellDim0) {
            continue;
        }
        if (desc.GetCacheOpMaxCount(CELL_MATCH_OP_TYPE_NORMAL_WRITE) != normalWriteCap) {
            continue;
        }
        if (desc.GetCacheOpMaxCount(CELL_MATCH_OP_TYPE_ATOMIC_WRITE) != atomicWriteCap) {
            continue;
        }
        EncodedOutcast candidate;
        if (!BindOutcast(prog, &update, &candidate)) {
            continue;
        }
        if (slot->outcast == nullptr ||
            candidate.outcast->producerConsumerList.size() > slot->outcast->producerConsumerList.size()) {
            *slot = candidate;
        }
    }
    return slot->outcast != nullptr;
}

bool FillProducers(EncodedOutcast& slot, std::vector<uint64_t>* table)
{
    const DevCellMatchTableDesc& desc = slot.update->cellMatchTableDesc;
    size_t cells = desc.GetStride(0);
    *table = std::vector<uint64_t>(cells * desc.cellUint64Size, 0);
    auto& list = slot.outcast->producerConsumerList;
    uint64_t* expr = slot.func->GetExpressionSize() == 0 ? nullptr : slot.func->GetExpressionAddr();
    uint32_t err = CellMatchFillIncastOutcast(slot.func, &slot.func->At(list, 0), list.size(), expr, desc,
                                              table->data(), kTag, slot.funcIdx);
    return err == 0;
}

uint32_t WriterCount(const std::vector<uint64_t>& table, const DevCellMatchTableDesc& desc, int cell)
{
    uint64_t base = CellMatchCellIndexToMemBase(static_cast<uint64_t>(cell), desc);
    uint64_t meta = table[base];
    if (CellMatchGetCurrentOpType(meta) != CELL_MATCH_OP_TYPE_NORMAL_WRITE) {
        return 0;
    }
    return CellMatchGetCurrentOpCount(meta);
}

bool HasWriteRange(const std::vector<uint64_t>& table, const DevCellMatchTableDesc& desc, int cell, uint64_t off0,
                   uint64_t off1, uint64_t shp0, uint64_t shp1)
{
    uint64_t base = CellMatchCellIndexToMemBase(static_cast<uint64_t>(cell), desc);
    uint32_t count = WriterCount(table, desc, cell);
    for (uint32_t i = 0; i < count; ++i) {
        const uint64_t* pairs = CellMatchNormalWriteOpRangeBase(const_cast<uint64_t*>(table.data()), base, i, desc);
        if (pairs[0] == off0 && pairs[2] == off1 && pairs[1] == shp0 && pairs[3] == shp1) {
            return true;
        }
    }
    return false;
}

int CellWithWriterCount(const std::vector<uint64_t>& table, const DevCellMatchTableDesc& desc, uint32_t want)
{
    int found = -1;
    int cells = static_cast<int>(desc.GetStride(0));
    for (int cell = 0; cell < cells; ++cell) {
        if (WriterCount(table, desc, cell) == want) {
            EXPECT_EQ(found, -1);
            found = cell;
        }
    }
    return found;
}

void CollectReads(const EncodedOutcast& slot, std::vector<ReadAccess>* reads)
{
    const DevCellMatchTableDesc& desc = slot.update->cellMatchTableDesc;
    int dims = desc.GetDimensionSize();
    for (uint32_t funcIdx = 0; funcIdx < slot.prog->GetFunctionSize(); ++funcIdx) {
        DevAscendFunction* func = slot.prog->GetFunction(static_cast<int>(funcIdx));
        int incastIdx = func->LookupIncastBySlotIndex(slot.update->slotIndex);
        if (incastIdx == INVALID_INDEX) {
            continue;
        }
        DevAscendFunctionIncast& incast = func->GetIncast(incastIdx);
        uint64_t* expr = func->GetExpressionSize() == 0 ? nullptr : func->GetExpressionAddr();
        auto append = [&](DevLocalVector<DevAscendFunctionCallOperandUse>& list) {
            for (size_t i = 0; i < list.size(); ++i) {
                auto& use = func->At(list, i);
                ReadAccess access;
                access.funcIdx = funcIdx;
                GetTensorOffsetAndValidShape(func, access.offset, access.shape, expr, desc, dims, use.offsetAttrIdx,
                                             use.cachedAttrBase, use.operationIdx);
                reads->push_back(access);
            }
        };
        append(incast.consumerList);
        append(incast.stitchPolicyFullCoverConsumerList);
    }
}

const ReadAccess* FindRead(const std::vector<ReadAccess>& reads, uint64_t shape0)
{
    for (const auto& access : reads) {
        if (access.shape[0] == shape0) {
            return &access;
        }
    }
    return nullptr;
}

int CountOverlap(std::vector<uint64_t>& table, const DevCellMatchTableDesc& desc, int cell, const ReadAccess& access)
{
    int dims = desc.GetDimensionSize();
    uint64_t base = CellMatchCellIndexToMemBase(static_cast<uint64_t>(cell), desc);
    uint32_t count = WriterCount(table, desc, cell);
    int hit = 0;
    for (uint32_t i = 0; i < count; ++i) {
        uint64_t opId = CellMatchGetOpId(table.data(), base, CELL_MATCH_OP_TYPE_NORMAL_WRITE, i, desc);
        const uint64_t* storedPairs = CellMatchNormalWriteOpRangeBase(table.data(), base, i, desc);
        if (CellMatchNormalWriteOpRangesOverlap(access.offset, access.shape, storedPairs, dims) &&
            FuncID(static_cast<uint32_t>(opId)) != access.funcIdx) {
            ++hit;
        }
    }
    return hit;
}

void ExpectReaderMask(const DevAscendProgramUpdate& update)
{
    EXPECT_EQ(update.stitchCtrlBitMask & (STITCH_CTRL_WAR | STITCH_CTRL_RAW), STITCH_CTRL_WAR | STITCH_CTRL_RAW);
}

} // namespace

TEST(CellMatchMultiWrite, TailMultipleOfCellUsesOneWriter)
{
    PrepareTailKernel(64, kHidden);
    Tensor in(DT_FP32, {80, kHidden}, "in");
    Tensor out(DT_FP32, {80, kHidden}, "out");
    Tensor result(DT_FP32, {80, kHidden}, "result");
    FUNCTION("tail_multiple", {in}, {out, result})
    {
        LOOP("TM_W", FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            auto head0 = View(in, {32, kHidden}, {0, 0});
            auto head1 = View(in, {32, kHidden}, {32, 0});
            auto tail = View(in, {16, kHidden}, {64, 0});
            Assemble(Add(head0, head0), {0, 0}, out);
            Assemble(Add(head1, head1), {32, 0}, out);
            Assemble(Add(tail, tail), {64, 0}, out);
        }
        LOOP("TM_R", FunctionType::DYNAMIC_LOOP, j, LoopRange(1))
        {
            (void)j;
            Assemble(View(out, {32, kHidden}, {0, 0}), {0, 0}, result);
            Assemble(View(out, {32, kHidden}, {32, 0}), {32, 0}, result);
            Assemble(View(out, {16, kHidden}, {64, 0}), {64, 0}, result);
        }
    }

    EncodedOutcast slot;
    ASSERT_TRUE(FindEncoded(16, 1u, 0u, &slot));
    const DevCellMatchTableDesc& desc = slot.update->cellMatchTableDesc;
    EXPECT_EQ(desc.GetDimensionSize(), 2);
    EXPECT_EQ(desc.GetCellShape(1), kHidden);
    EXPECT_EQ(desc.GetStrideShape(0), 5);
    EXPECT_EQ(desc.GetStrideShape(1), 1);
    EXPECT_EQ(desc.normalWriteOpRangeBase, 0u);
    EXPECT_EQ(desc.cellUint64Size, 2u);
    ExpectReaderMask(*slot.update);

    std::vector<uint64_t> table;
    ASSERT_TRUE(FillProducers(slot, &table));
    ASSERT_EQ(desc.GetStride(0), 5u);
    for (int cell = 0; cell < 5; ++cell) {
        EXPECT_EQ(WriterCount(table, desc, cell), 1u) << cell;
    }
}

TEST(CellMatchMultiWrite, ReadAtLeastMaxWriteForcesOneWriter)
{
    PrepareTailKernel(64, kHidden);
    Tensor in(DT_FP32, {144, kHidden}, "in");
    Tensor out(DT_FP32, {144, kHidden}, "out");
    Tensor result(DT_FP32, {128, kHidden}, "result");
    Tensor sideOut(DT_FP32, {64, kHidden}, "side_out");
    FUNCTION("read_covers_max_write", {in}, {out, result, sideOut})
    {
        LOOP("FC_W", FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            // Writes keep distinct shapes so cell/gcd leave divides=false (gate under test).
            auto head = View(in, {64, kHidden}, {0, 0});
            auto tail = View(in, {48, kHidden}, {64, 0});
            Assemble(Add(head, head), {0, 0}, out);
            Assemble(Add(tail, tail), {64, 0}, out);
        }
        LOOP("FC_R", FunctionType::DYNAMIC_LOOP, j, LoopRange(1))
        {
            (void)j;
            // All reads COA shape 64 → min_read 64 >= max_write 64 → factor 1.
            Assemble(View(out, {64, kHidden}, {0, 0}), {0, 0}, result);
            Assemble(View(out, {64, kHidden}, {64, 0}), {64, 0}, result);
            Assemble(View(out, {64, kHidden}, {96, 0}), {0, 0}, sideOut);
        }
    }

    EncodedOutcast slot;
    ASSERT_TRUE(FindEncoded(48, 1u, 0u, &slot));
    const DevCellMatchTableDesc& desc = slot.update->cellMatchTableDesc;
    EXPECT_EQ(desc.normalWriteOpRangeBase, 0u);
    EXPECT_EQ(desc.cellUint64Size, 2u);
    ExpectReaderMask(*slot.update);
}

TEST(CellMatchMultiWrite, TailNotMultipleOfCellKeepsTwoWriters)
{
    PrepareTailKernel(64, kHidden);
    Tensor in(DT_FP32, {144, kHidden}, "in");
    Tensor out(DT_FP32, {144, kHidden}, "out");
    Tensor result(DT_FP32, {96, kHidden}, "result");
    FUNCTION("tail_not_multiple", {in}, {out, result})
    {
        LOOP("TN_W", FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            auto head = View(in, {64, kHidden}, {0, 0});
            auto tail = View(in, {48, kHidden}, {64, 0});
            Assemble(Add(head, head), {0, 0}, out);
            Assemble(Add(tail, tail), {64, 0}, out);
        }
        LOOP("TN_R", FunctionType::DYNAMIC_LOOP, j, LoopRange(1))
        {
            (void)j;
            // min read 48 < max write 64 → keep outcast factor 2.
            Assemble(View(out, {48, kHidden}, {0, 0}), {0, 0}, result);
            Assemble(View(out, {48, kHidden}, {48, 0}), {48, 0}, result);
        }
    }

    EncodedOutcast slot;
    ASSERT_TRUE(FindEncoded(48, 2u, 0u, &slot));
    const DevCellMatchTableDesc& desc = slot.update->cellMatchTableDesc;
    EXPECT_EQ(desc.GetDimensionSize(), 2);
    EXPECT_EQ(desc.GetCellShape(1), kHidden);
    EXPECT_EQ(desc.GetStrideShape(0), 3);
    EXPECT_EQ(desc.GetStrideShape(1), 1);
    EXPECT_EQ(desc.normalWriteOpRangeBase, 3u);
    EXPECT_EQ(desc.cellUint64Size, 11u);
    ExpectReaderMask(*slot.update);

    std::vector<uint64_t> table;
    ASSERT_TRUE(FillProducers(slot, &table));
    EXPECT_EQ(WriterCount(table, desc, 0), 1u);
    EXPECT_EQ(WriterCount(table, desc, 1), 2u);
    EXPECT_EQ(WriterCount(table, desc, 2), 1u);
    EXPECT_TRUE(HasWriteRange(table, desc, 1, 0, 0, 64, kHidden));
    EXPECT_TRUE(HasWriteRange(table, desc, 1, 64, 0, 48, kHidden));
}

TEST(CellMatchMultiWrite, TwoDimCornerKeepsFourWriters)
{
    PrepareTailKernel(128, 128);
    Tensor in(DT_FP32, {75, 75}, "in");
    Tensor out(DT_FP32, {75, 75}, "out");
    Tensor result(DT_FP32, {15, 15}, "result");
    FUNCTION("two_dim_corner", {in}, {out, result})
    {
        LOOP("TD_W", FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            auto a = View(in, {46, 46}, {0, 0});
            auto b = View(in, {15, 46}, {46, 0});
            auto c = View(in, {46, 15}, {0, 46});
            auto d = View(in, {15, 15}, {46, 46});
            Assemble(Add(a, a), {0, 0}, out);
            Assemble(Add(b, b), {46, 0}, out);
            Assemble(Add(c, c), {0, 46}, out);
            Assemble(Add(d, d), {46, 46}, out);
        }
        LOOP("TD_R", FunctionType::DYNAMIC_LOOP, j, LoopRange(1))
        {
            (void)j;
            // min read 15 < max write 46 → keep factor 4.
            Assemble(View(out, {15, 15}, {0, 0}), {0, 0}, result);
        }
    }

    EncodedOutcast slot;
    ASSERT_TRUE(FindEncoded(15, 4u, 0u, &slot));
    const DevCellMatchTableDesc& desc = slot.update->cellMatchTableDesc;
    EXPECT_EQ(desc.GetCellShape(1), 15);
    EXPECT_EQ(desc.GetStrideShape(0), 5);
    EXPECT_EQ(desc.GetStrideShape(1), 5);
    EXPECT_EQ(desc.normalWriteOpRangeBase, 5u);
    EXPECT_EQ(desc.cellUint64Size, 21u);
    ExpectReaderMask(*slot.update);

    std::vector<uint64_t> table;
    ASSERT_TRUE(FillProducers(slot, &table));
    int corner = CellWithWriterCount(table, desc, 4);
    ASSERT_GE(corner, 0);
    EXPECT_TRUE(HasWriteRange(table, desc, corner, 0, 0, 46, 46));
    EXPECT_TRUE(HasWriteRange(table, desc, corner, 46, 0, 15, 46));
    EXPECT_TRUE(HasWriteRange(table, desc, corner, 0, 46, 46, 15));
    EXPECT_TRUE(HasWriteRange(table, desc, corner, 46, 46, 15, 15));
}

TEST(CellMatchMultiWrite, CrossRootOverlapSelectsWriters)
{
    PrepareTailKernel(64, kHidden);
    Tensor in(DT_FP32, {144, kHidden}, "in");
    Tensor out(DT_FP32, {144, kHidden}, "out");
    Tensor narrow(DT_FP32, {16, kHidden}, "narrow");
    Tensor wide(DT_FP32, {48, kHidden}, "wide");
    FUNCTION("cross_root_overlap", {in}, {out, narrow, wide})
    {
        LOOP("CR_W", FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            auto head = View(in, {64, kHidden}, {0, 0});
            auto tail = View(in, {48, kHidden}, {64, 0});
            Assemble(Add(head, head), {0, 0}, out);
            Assemble(Add(tail, tail), {64, 0}, out);
        }
        LOOP("CR_R", FunctionType::DYNAMIC_LOOP, j, LoopRange(1))
        {
            (void)j;
            Assemble(View(out, {16, kHidden}, {48, 0}), {0, 0}, narrow);
            Assemble(View(out, {48, kHidden}, {48, 0}), {0, 0}, wide);
        }
    }

    EncodedOutcast slot;
    ASSERT_TRUE(FindEncoded(48, 2u, 0u, &slot));
    const DevCellMatchTableDesc& desc = slot.update->cellMatchTableDesc;
    EXPECT_NE(desc.normalWriteOpRangeBase, 0u);
    ExpectReaderMask(*slot.update);

    std::vector<uint64_t> table;
    ASSERT_TRUE(FillProducers(slot, &table));
    int shared = CellWithWriterCount(table, desc, 2);
    ASSERT_GE(shared, 0);

    std::vector<ReadAccess> reads;
    CollectReads(slot, &reads);
    const ReadAccess* narrowRead = FindRead(reads, 16);
    const ReadAccess* wideRead = FindRead(reads, 48);
    ASSERT_NE(narrowRead, nullptr);
    ASSERT_NE(wideRead, nullptr);
    EXPECT_EQ(narrowRead->offset[0], 48u);
    EXPECT_EQ(wideRead->offset[0], 48u);
    EXPECT_NE(narrowRead->funcIdx, slot.funcIdx);

    EXPECT_EQ(CountOverlap(table, desc, shared, *narrowRead), 1);
    EXPECT_EQ(CountOverlap(table, desc, shared, *wideRead), 2);
    ReadAccess sameRoot = *narrowRead;
    sameRoot.funcIdx = slot.funcIdx;
    EXPECT_EQ(CountOverlap(table, desc, shared, sameRoot), 0);
}

TEST(CellMatchMultiWrite, MiddleShapeNotMultipleOfCellUsesTwoWriters)
{
    PrepareTailKernel(32, kHidden);
    Tensor in(DT_FP32, {60, kHidden}, "in");
    Tensor out(DT_FP32, {60, kHidden}, "out");
    Tensor result(DT_FP32, {12, kHidden}, "result");
    FUNCTION("middle_shape", {in}, {out, result})
    {
        LOOP("MS_W", FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            auto wide = View(in, {24, kHidden}, {0, 0});
            auto mid = View(in, {18, kHidden}, {24, 0});
            auto narrow = View(in, {12, kHidden}, {42, 0});
            Assemble(Add(wide, wide), {0, 0}, out);
            Assemble(Add(mid, mid), {24, 0}, out);
            Assemble(Add(narrow, narrow), {42, 0}, out);
        }
        LOOP("MS_R", FunctionType::DYNAMIC_LOOP, j, LoopRange(1))
        {
            (void)j;
            // min read 12 < max write 24 → keep factor 2.
            Assemble(View(out, {12, kHidden}, {0, 0}), {0, 0}, result);
        }
    }

    EncodedOutcast slot;
    ASSERT_TRUE(FindEncoded(12, 2u, 0u, &slot));
    EXPECT_NE(slot.update->cellMatchTableDesc.normalWriteOpRangeBase, 0u);
    ExpectReaderMask(*slot.update);
}

TEST(CellMatchMultiWrite, AtomicAddsAppendInsteadOfReplace)
{
    PrepareTailKernel(64, kHidden);
    Tensor a(DT_FP32, {kHidden, kHidden}, "a");
    Tensor b(DT_FP32, {kHidden, kHidden}, "b");
    Tensor out(DT_FP32, {64, kHidden}, "out");
    Tensor result(DT_FP32, {kHidden, kHidden}, "result");
    FUNCTION("atomic_append", {a, b}, {out, result})
    {
        LOOP("AA_W", FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            AtomicRMW(Add(a, b), {0, 0}, out, AtomicRMWMode::ADD);
            AtomicRMW(Add(a, a), {16, 0}, out, AtomicRMWMode::ADD);
        }
        LOOP("AA_R", FunctionType::DYNAMIC_LOOP, j, LoopRange(1))
        {
            (void)j;
            Assemble(View(out, {kHidden, kHidden}, {0, 0}), {0, 0}, result);
        }
    }

    EncodedOutcast slot;
    ASSERT_TRUE(FindEncoded(kHidden, 1u, CELL_MATCH_MAX_ATOMIC_WRITE_COUNT, &slot));
    const DevCellMatchTableDesc& desc = slot.update->cellMatchTableDesc;
    EXPECT_EQ(desc.normalWriteOpRangeBase, 0u);
    EXPECT_EQ(desc.cellUint64Size, 1u + 1u + CELL_MATCH_MAX_ATOMIC_WRITE_COUNT);
    ExpectReaderMask(*slot.update);

    std::vector<uint64_t> table;
    ASSERT_TRUE(FillProducers(slot, &table));
    uint64_t base = CellMatchCellIndexToMemBase(0, desc);
    uint64_t meta = table[base];
    EXPECT_EQ(CellMatchGetCurrentOpType(meta), CELL_MATCH_OP_TYPE_ATOMIC_WRITE);
    EXPECT_EQ(CellMatchGetCurrentOpCount(meta), 2u);

    std::vector<int> atomicOps;
    for (size_t i = 0; i < slot.outcast->producerConsumerList.size(); ++i) {
        const auto& use = slot.func->At(slot.outcast->producerConsumerList, i);
        if (use.opType == CellMatchOpType::ATOMIC_WRITE) {
            atomicOps.push_back(use.operationIdx);
        }
    }
    ASSERT_EQ(atomicOps.size(), 2u);
    for (size_t i = 0; i < atomicOps.size(); ++i) {
        uint64_t expect = (kTag << CELL_MATCH_META_TAGID_SHIFT32) | MakeTaskID(slot.funcIdx, atomicOps[i]);
        EXPECT_EQ(CellMatchGetOpId(table.data(), base, CELL_MATCH_OP_TYPE_ATOMIC_WRITE, static_cast<uint32_t>(i), desc),
                  expect);
    }
}

// Narrow View({16}) pulls min_read below max_write → keep factor 2.
TEST(CellMatchMultiWrite, MidNarrowReadKeepsTwoWriters)
{
    PrepareTailKernel(64, kHidden);
    Tensor in(DT_FP32, {112, kHidden}, "in");
    Tensor out(DT_FP32, {112, kHidden}, "out");
    Tensor result(DT_FP32, {112, kHidden}, "result");
    Tensor narrow(DT_FP32, {16, kHidden}, "narrow");
    FUNCTION("mid_narrow_read", {in}, {out, result, narrow})
    {
        LOOP("MN_W", FunctionType::DYNAMIC_LOOP, i, LoopRange(1))
        {
            (void)i;
            Assemble(Add(View(in, {64, kHidden}, {0, 0}), View(in, {64, kHidden}, {0, 0})), {0, 0}, out);
            Assemble(Add(View(in, {48, kHidden}, {64, 0}), View(in, {48, kHidden}, {64, 0})), {64, 0}, out);
        }
        LOOP("MN_R", FunctionType::DYNAMIC_LOOP, j, LoopRange(1))
        {
            (void)j;
            Assemble(View(out, {64, kHidden}, {0, 0}), {0, 0}, result);
            Assemble(View(out, {16, kHidden}, {48, 0}), {0, 0}, narrow);
        }
    }

    EncodedOutcast slot;
    ASSERT_TRUE(FindEncoded(48, 2u, 0u, &slot));
    ExpectReaderMask(*slot.update);
}
