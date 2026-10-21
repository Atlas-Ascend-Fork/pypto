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
 * \file test_device_perf.cpp
 * \brief UT for machine/runtime/runner/fed_prof.cpp
 */

#include <gtest/gtest.h>
#define private public
#include "machine/runtime/runner/fed_prof_manager.h"
#undef private
#include "interface/configs/config_manager.h"
#include "tilefwk/aicpu_common.h"

using namespace npu::tile_fwk;

TEST(DevicePerfTest, AllPerfOps_NoCrash)
{
    {
        FedProfManager perf;
    }

    FedProfManager perf;
    FedProfDeviceArgs args;
    args.nrAic = 2;
    args.nrAiv = 4;
    perf.args_ = args;
    EXPECT_EQ(perf.GetCoreCount(), 2u + 4u + 1u);

    perf.PerCoreProfSync(false);
    perf.PerCoreProfResetMetadata(false);
    perf.PerCoreProfResetMetadata(true);
    perf.PerCoreProfReleaseData();
    EXPECT_EQ(perf.PerCoreProfDevAddr(), 0u);

    perf.AicpuProfDumpThreadStop();
    perf.AicpuProfDumpThreadStart();
    EXPECT_FALSE(perf.dumpThread_.joinable());
}

TEST(DevicePerfTest, ResetMetadata_EmptyPerfData)
{
    FedProfManager perf;
    perf.PerCoreProfResetMetadata(false);
}

TEST(DevicePerfTest, ResetMetadata_MultipleLaunch)
{
    setenv("DUMP_DEVICE_PERF", "true", 1);
    FedProfManager perf;
    perf.DeviceDataAllocate(FED_PROF_PER_CORE_PROF);
    perf.PerCoreProfResetMetadata(true);
    unsetenv("DUMP_DEVICE_PERF");
}

TEST(DevicePerfTest, SyncData_DebugDisabled)
{
    FedProfManager perf;
    perf.PerCoreProfSync(false);
}

TEST(DevicePerfTest, ReleaseData_NullPtrs)
{
    FedProfManager perf;
    perf.PerCoreProfReleaseData();
    EXPECT_EQ(perf.PerCoreProfDevAddr(), 0u);
}

TEST(DevicePerfTest, GetCoreCount_Default)
{
    FedProfManager perf;
    FedProfDeviceArgs args{};
    args.nrAic = 0;
    args.nrAiv = 0;
    perf.args_ = args;
    EXPECT_EQ(perf.GetCoreCount(), 1u);
}

TEST(DevicePerfTest, StartMachinePerfTraceDumpThread_ZeroAddr)
{
    FedProfManager perf;
    perf.AicpuProfDumpThreadStart();
    EXPECT_FALSE(perf.dumpThread_.joinable());
}

TEST(DevicePerfTest, ResetData_WithPerfData)
{
    setenv("DUMP_DEVICE_PERF", "true", 1);
    FedProfManager perf;
    FedProfDeviceArgs args{};
    args.nrAic = 1;
    args.nrAiv = 0;
    perf.args_ = args;
    perf.DeviceDataAllocate(FED_PROF_PER_CORE_PROF);
    perf.PerCoreProfResetData();
    unsetenv("DUMP_DEVICE_PERF");
}

TEST(DevicePerfTest, ResetMetadata_WithAicpuPerfAddr)
{
    setenv("DUMP_DEVICE_PERF", "true", 1);
    FedProfManager perf;
    FedProfDeviceArgs args{};
    args.nrAic = 1;
    args.nrAiv = 0;
    perf.args_ = args;
    perf.DeviceDataAllocate(FED_PROF_PER_CORE_PROF);
    perf.DeviceDataAllocate(FED_PROF_AICPU_PROF);
    perf.PerCoreProfResetMetadata(true);
    unsetenv("DUMP_DEVICE_PERF");
}

TEST(DevicePerfTest, ResetMetadata_WithoutAicpuPerfAddr)
{
    setenv("DUMP_DEVICE_PERF", "true", 1);
    FedProfManager perf;
    FedProfDeviceArgs args{};
    args.nrAic = 1;
    args.nrAiv = 0;
    perf.args_ = args;
    perf.DeviceDataAllocate(FED_PROF_PER_CORE_PROF);
    perf.PerCoreProfResetMetadata(false);
    unsetenv("DUMP_DEVICE_PERF");
}
