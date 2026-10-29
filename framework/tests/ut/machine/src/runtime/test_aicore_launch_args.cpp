/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

#include "interface/machine/device/tilefwk/aikernel_tensor.h"
#include "machine/runtime/runner/aicore_launch_args.h"
#include "machine/utils/dynamic/dev_encode_program.h"
#include "machine/utils/dynamic/dev_encode_program_ctrlflow_cache.h"
#include "machine/utils/dynamic/dev_encode_tensor.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

TEST(AicoreLaunchArgsTest, LayoutBytes)
{
    EXPECT_EQ(AicoreLaunchSlotsBytes(), kAicoreLaunchArgSlots * sizeof(void*));
    EXPECT_EQ(AicoreLaunchArgsBytes(0), AicoreLaunchSlotsBytes());
    EXPECT_EQ(AicoreLaunchArgsBytes(8), AicoreLaunchSlotsBytes() + 8);
    EXPECT_EQ(AicoreLaunchArgsBytes(1) % 8, 0u);
}

TEST(AicoreLaunchArgsTest, InitAicoreHostInputInfo)
{
    RtHostInputInfo info{};
    InitAicoreHostInputInfo(info);
    EXPECT_EQ(info.addrOffset, static_cast<uint32_t>(kAicoreInputsSlot * sizeof(void*)));
    EXPECT_EQ(info.dataOffset, static_cast<uint32_t>(AicoreLaunchSlotsBytes()));
}

TEST(AicoreLaunchArgsTest, PrepareNullArgsIsNoOp)
{
    std::vector<uint8_t> buf;
    RtArgsEx rtArgs{};
    rtArgs.argsSize = 123;
    PrepareAicoreLaunchArgsBuffer(buf, nullptr, "op", 0, rtArgs);
    EXPECT_EQ(rtArgs.args, nullptr);
    EXPECT_EQ(rtArgs.argsSize, 0u);
    EXPECT_EQ(rtArgs.hostInputInfoNum, 0u);
}

TEST(AicoreLaunchArgsTest, PrepareFillsSevenSlots)
{
    alignas(AiCpuArgs) uint8_t raw[sizeof(AiCpuArgs) + 2 * sizeof(int64_t) + sizeof(DevTensorData)] = {};
    auto* aicpuArgs = new (raw) AiCpuArgs();
    aicpuArgs->kArgs.cfgdata = reinterpret_cast<int64_t*>(0x1000);
    int64_t* inputp = reinterpret_cast<int64_t*>(aicpuArgs + 1);
    inputp[0] = 1;
    inputp[1] = 0;

    std::vector<uint8_t> buf;
    RtArgsEx rtArgs{};
    PrepareAicoreLaunchArgsBuffer(buf, aicpuArgs, "demo_op", 0, rtArgs);

    ASSERT_EQ(buf.size(), AicoreLaunchSlotsBytes());
    ASSERT_EQ(rtArgs.argsSize, static_cast<uint32_t>(AicoreLaunchSlotsBytes()));
    ASSERT_EQ(rtArgs.hostInputInfoNum, 0u);
    auto* slots = reinterpret_cast<void**>(buf.data());
    EXPECT_STREQ(static_cast<const char*>(slots[0]), "demo_op");
    EXPECT_EQ(slots[1], nullptr);
    EXPECT_EQ(slots[4], inputp);
    EXPECT_EQ(slots[5], aicpuArgs->kArgs.cfgdata);
    EXPECT_EQ(slots[6], reinterpret_cast<DevTensorData*>(inputp + DEV_TENSOR_DATA_OFFSET));
}

TEST(AicoreLaunchArgsTest, PrepareGrowsForArenaThenFillNoCrashOnNullHostCache)
{
    alignas(AiCpuArgs) uint8_t raw[sizeof(AiCpuArgs) + 2 * sizeof(int64_t)] = {};
    auto* aicpuArgs = new (raw) AiCpuArgs();
    aicpuArgs->kArgs.workspace = reinterpret_cast<int64_t*>(0x2000);

    std::vector<uint8_t> buf;
    RtArgsEx rtArgs{};
    RtHostInputInfo hostInfo{};
    constexpr size_t kArenaBytes = 32;
    PrepareAicoreLaunchArgsBuffer(buf, aicpuArgs, "op", kArenaBytes, rtArgs);
    ASSERT_EQ(buf.size(), AicoreLaunchArgsBytes(kArenaBytes));

    // Null hostCache must not enable host-input.
    FillAicoreIoAddressArenaBuffer(buf, aicpuArgs, nullptr, 1, 0, rtArgs, hostInfo);
    EXPECT_EQ(rtArgs.hostInputInfoNum, 0u);
}

TEST(AicoreLaunchArgsTest, EnsureHostCtrlFlowCacheRebasedNullSafe)
{
    EnsureHostCtrlFlowCacheRebased(nullptr, nullptr);
    DevAscendProgram prog{};
    EnsureHostCtrlFlowCacheRebased(nullptr, &prog);
}
