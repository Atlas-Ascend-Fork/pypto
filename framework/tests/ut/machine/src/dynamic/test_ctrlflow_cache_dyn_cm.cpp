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
 * \file test_ctrlflow_cache_dyn_cm.cpp
 * \brief Dyn-CM CF-cache continue: Data==null → need-alloc flag → TryAllocate.
 */
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#define private public
#define protected public
#include "machine/utils/dynamic/dev_workspace.h"
#include "machine/utils/dynamic/dev_encode_program.h"
#include "machine/utils/dynamic/dev_encode_program_ctrlflow_cache.h"
#include "machine/utils/dynamic/dev_encode_function_stitch.h"
#include "machine/utils/dynamic/dev_start_args.h"
#include "machine/utils/dynamic/item_pool.h"
#include "machine/utils/dynamic/runtime_outcast_tensor.h"
#include "machine/device/dynamic/context/device_execute_context.h"
#include "machine/device/dynamic/context/device_slot_context.h"
#undef private
#undef protected

using namespace npu::tile_fwk::dynamic;

namespace {

void FillMinimalDynCmProg(DevAscendProgram& prog)
{
    prog.memBudget.metadata.general = 1u << 18;
    prog.memBudget.metadata.stitchPool = 1u << 16;
    prog.memBudget.metadata.duppedStitchPool = 1u << 14;
    prog.memBudget.metadata.generalSlabSize = 1u << 15;
    prog.memBudget.metadata.stitchSlabSize = 1u << 14;
    prog.memBudget.metadata.duppedStitchSlabSize = 1u << 12;
    prog.memBudget.metadata.dynamicCellMatchSlotNum = 2;
    prog.memBudget.metadata.maxDynamicCellMatchTableMem = 256;
    prog.memBudget.metadata.dynamicCellMatch = 2 * 256;
    prog.memBudget.tensor.runtimeOutcastPoolSize = 4;
    prog.devArgs.nrAic = 1;
    prog.devArgs.nrAiv = 1;
}

void BindWorkspaceAddrs(DevStartArgs& args, DevAscendProgram& prog, std::vector<uint8_t>& workspace)
{
    args.deviceRuntimeDataDesc.generalAddr = reinterpret_cast<uint64_t>(workspace.data());
    args.deviceRuntimeDataDesc.stitchPoolAddr = args.deviceRuntimeDataDesc.generalAddr +
                                                prog.memBudget.metadata.general;
    args.deviceRuntimeDataDesc.duppedStitchPoolAddr = args.deviceRuntimeDataDesc.stitchPoolAddr +
                                                      prog.memBudget.metadata.stitchPool;
    args.deviceRuntimeDataDesc.dynamicCellMatchAddr = args.deviceRuntimeDataDesc.duppedStitchPoolAddr +
                                                      prog.memBudget.metadata.duppedStitchPool;
    args.InitWorkspace(&prog, workspace.data());
}

} // namespace

// flag true → Allocate once and clear; second call / flag false → no-op.
TEST(CtrlFlowCacheDynCmUt, TryAllocate_OnlyWhenNeedAllocFlag)
{
    std::vector<uint8_t> workspace(1u << 20, 0);
    DevAscendProgram prog{};
    FillMinimalDynCmProg(prog);
    DevStartArgs args{};
    BindWorkspaceAddrs(args, prog, workspace);

    DeviceWorkspaceAllocator ws;
    ws.Init(&args);
    ASSERT_TRUE(ws.HasDynamicCellMatchSlots());
    const size_t avail = ws.metadataAllocators_.dynamicCellMatch.AvailableSlots();

    ItemPoolIter outIter = ws.MakeRuntimeOutcastTensor(WsAllocation(0xAAull, 0), RuntimeTensorMemProperty::EXTERNAL);
    ASSERT_NE(outIter, ITEM_POOL_INVALID_INDEX);

    DevAscendProgramUpdate partial{};
    partial.cellMatchTableDesc.SetCellShape({4});
    partial.cellMatchTableDesc.SetStrideShape({4});
    partial.cellMatchTableDesc.cellUint64Size = 2;
    partial.cellMatchRuntimePartialUpdateTable = DevRelocVector<uint64_t>(0, nullptr);

    DeviceExecuteSlot slot{};
    slot.partialUpdate = &partial;
    slot.rtOutcastIter = outIter;

    slot.isDynamicCellMatchNeedAlloc = false;
    ws.TryAllocateDynamicCellMatchForAssembleSlot(slot);
    EXPECT_EQ(ws.metadataAllocators_.dynamicCellMatch.AvailableSlots(), avail);
    EXPECT_EQ(partial.cellMatchRuntimePartialUpdateTable.Data(), nullptr);

    slot.isDynamicCellMatchNeedAlloc = true;
    ws.TryAllocateDynamicCellMatchForAssembleSlot(slot);
    EXPECT_NE(partial.cellMatchRuntimePartialUpdateTable.Data(), nullptr);
    EXPECT_NE(ws.GetRuntimeOutcastTensor(outIter).dynamicCellMatchAllocation.ptr, 0u);
    EXPECT_EQ(ws.metadataAllocators_.dynamicCellMatch.AvailableSlots(), avail - 1);
    EXPECT_FALSE(slot.isDynamicCellMatchNeedAlloc);

    auto* data = partial.cellMatchRuntimePartialUpdateTable.Data();
    ws.TryAllocateDynamicCellMatchForAssembleSlot(slot);
    EXPECT_EQ(partial.cellMatchRuntimePartialUpdateTable.Data(), data);
    EXPECT_EQ(ws.metadataAllocators_.dynamicCellMatch.AvailableSlots(), avail - 1);
}

static void RunContinueRootStitch(DeviceExecuteContext& ctx, DevAscendProgram& prog)
{
    std::vector<DeviceExecuteSlot> backupSlots(static_cast<size_t>(prog.slotSize));
    std::vector<ItemPool<RuntimeOutcastTensor>::ItemBlock> backupOutcast(
        static_cast<size_t>(prog.memBudget.tensor.runtimeOutcastPoolSize));
    auto& cache = prog.controlFlowCache;
    cache.runtimeBackup.slotContext.slotList.HostAssignDataSize(reinterpret_cast<uint64_t>(backupSlots.data()),
                                                                backupSlots.size());
    cache.runtimeBackup.workspace.runtimeOutcastTensorPool.HostAssignDataSize(
        reinterpret_cast<uint64_t>(backupOutcast.data()), backupOutcast.size());
    cache.RuntimeAddrBackup(ctx.slotContext.GetSlotList(), ctx.workspace.GetRuntimeOutcastTensorPool(), prog.slotSize,
                            prog.memBudget.tensor.runtimeOutcastPoolSize, ctx.workspace.GetTensorAllocator(),
                            prog.GetParallelism());
    ctx.controlFlowCacheActivated = true;
    ctx.duppedRootCount = 0;
    (void)DeviceExecuteContext::DeviceExecuteRuntimeCallRootStitch(&ctx, /*non-special*/ 0x1234ULL);
}

// CF continue: Data empty → need-alloc (CM.ptr left for TryAllocate to overwrite).
TEST(CtrlFlowCacheDynCmUt, Continue_MarksNeedAllocWhenDataNull)
{
    std::vector<uint8_t> workspace(1u << 20, 0);
    DevAscendProgram prog{};
    FillMinimalDynCmProg(prog);
    prog.memBudget.tensor.parallelism = 0;
    prog.slotSize = 1;
    prog.ctrlFlowCacheAnchor = &prog.controlFlowCache;
    prog.controlFlowCache.rootTaskCount = 1;

    DevStartArgs args{};
    args.devProg = &prog;
    BindWorkspaceAddrs(args, prog, workspace);
    prog.devArgs.dynamicCellMatchAddr = args.deviceRuntimeDataDesc.dynamicCellMatchAddr;

    DeviceExecuteContext ctx(&args);
    ctx.args = &args;
    ctx.devProg = &prog;
    ctx.workspace.Init(&args);
    ctx.slotContext.InitAllocator(ctx.workspace, prog.slotSize);

    ItemPoolIter outIter = ctx.workspace.MakeRuntimeOutcastTensor(WsAllocation(0xAAull, 0),
                                                                  RuntimeTensorMemProperty::EXTERNAL);
    ASSERT_NE(outIter, ITEM_POOL_INVALID_INDEX);
    ctx.workspace.GetRuntimeOutcastTensor(outIter).dynamicCellMatchAllocation.ptr = 0xCAFEBABEULL;

    DevAscendProgramUpdate partial{};
    partial.isPartial = true;
    partial.stitchCtrlBitMask = STITCH_CTRL_WAW;
    partial.cellMatchRuntimePartialUpdateTable = DevRelocVector<uint64_t>(0, nullptr);
    DeviceExecuteSlot* slots = ctx.slotContext.GetSlotList();
    slots[0].rtOutcastIter = outIter;
    slots[0].isPartialUpdateStitch = true;
    slots[0].isDynamicCellMatchNeedAlloc = false;
    slots[0].partialUpdate = &partial;
    slots[0].stitchCtrlBitMask = STITCH_CTRL_WAW;

    RunContinueRootStitch(ctx, prog);

    EXPECT_TRUE(ctx.slotContext.GetSlotList()[0].isDynamicCellMatchNeedAlloc);
}

// Never Host-allocated dyn (CM.ptr==0, Data empty): must still mark — not static.
TEST(CtrlFlowCacheDynCmUt, Continue_MarksWhenDataNullEvenIfCmPtrZero)
{
    std::vector<uint8_t> workspace(1u << 20, 0);
    DevAscendProgram prog{};
    FillMinimalDynCmProg(prog);
    prog.memBudget.tensor.parallelism = 0;
    prog.slotSize = 1;
    prog.ctrlFlowCacheAnchor = &prog.controlFlowCache;
    prog.controlFlowCache.rootTaskCount = 1;

    DevStartArgs args{};
    args.devProg = &prog;
    BindWorkspaceAddrs(args, prog, workspace);

    DeviceExecuteContext ctx(&args);
    ctx.args = &args;
    ctx.devProg = &prog;
    ctx.workspace.Init(&args);
    ctx.slotContext.InitAllocator(ctx.workspace, prog.slotSize);

    DevAscendProgramUpdate partial{};
    partial.isPartial = true;
    partial.stitchCtrlBitMask = STITCH_CTRL_WAW;
    partial.cellMatchRuntimePartialUpdateTable = DevRelocVector<uint64_t>(0, nullptr);
    DeviceExecuteSlot* slots = ctx.slotContext.GetSlotList();
    slots[0].rtOutcastIter = ITEM_POOL_INVALID_INDEX;
    slots[0].isPartialUpdateStitch = true;
    slots[0].isDynamicCellMatchNeedAlloc = false;
    slots[0].partialUpdate = &partial;
    slots[0].stitchCtrlBitMask = STITCH_CTRL_WAW;

    RunContinueRootStitch(ctx, prog);

    EXPECT_TRUE(ctx.slotContext.GetSlotList()[0].isDynamicCellMatchNeedAlloc);
}

// Static prog-embedded table (Data non-null): continue must not set need-alloc.
TEST(CtrlFlowCacheDynCmUt, Continue_NoMarkWhenDataNonNull)
{
    std::vector<uint8_t> workspace(1u << 20, 0);
    DevAscendProgram prog{};
    FillMinimalDynCmProg(prog);
    prog.memBudget.tensor.parallelism = 0;
    prog.slotSize = 1;
    prog.ctrlFlowCacheAnchor = &prog.controlFlowCache;
    prog.controlFlowCache.rootTaskCount = 1;

    DevStartArgs args{};
    args.devProg = &prog;
    BindWorkspaceAddrs(args, prog, workspace);

    DeviceExecuteContext ctx(&args);
    ctx.args = &args;
    ctx.devProg = &prog;
    ctx.workspace.Init(&args);
    ctx.slotContext.InitAllocator(ctx.workspace, prog.slotSize);

    uint64_t staticTable[4] = {};
    DevAscendProgramUpdate partial{};
    partial.isPartial = true;
    partial.stitchCtrlBitMask = STITCH_CTRL_WAW;
    partial.cellMatchRuntimePartialUpdateTable = DevRelocVector<uint64_t>(4, staticTable);
    DeviceExecuteSlot* slots = ctx.slotContext.GetSlotList();
    slots[0].rtOutcastIter = ITEM_POOL_INVALID_INDEX;
    slots[0].isPartialUpdateStitch = true;
    slots[0].isDynamicCellMatchNeedAlloc = false;
    slots[0].partialUpdate = &partial;
    slots[0].stitchCtrlBitMask = STITCH_CTRL_WAW;

    RunContinueRootStitch(ctx, prog);

    EXPECT_FALSE(ctx.slotContext.GetSlotList()[0].isDynamicCellMatchNeedAlloc);
}
