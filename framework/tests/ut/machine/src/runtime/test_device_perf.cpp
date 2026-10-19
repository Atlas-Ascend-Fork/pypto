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
#include "machine/runtime/runner/fed_prof.h"
#undef private
#include "interface/configs/config_manager.h"
#include "tilefwk/aicpu_common.h"

using namespace npu::tile_fwk;

TEST(DevicePerfTest, AllPerfOps_NoCrash)
{
    {
        FedProf perf;
    }

    FedProf perf;
    FedProfDeviceArgs args;
    args.nrAic = 2;
    args.nrAiv = 4;
    perf.args_ = args;
    EXPECT_EQ(perf.GetCoreCount(), 2u + 4u + 1u);

    perf.PerCoreProfSync(false);
    perf.PerCoreProfResetMetadata(0);
    perf.PerCoreProfResetMetadata(999);
    perf.PerCoreProfReleaseData();
    EXPECT_EQ(perf.perCoreProfDevAddr_, 0u);

    perf.AicpuProfDumpThreadStop();
    perf.AicpuProfDumpThreadStart();
    EXPECT_FALSE(perf.dumpThread_.joinable());
}

TEST(DevicePerfTest, ResetMetadata_EmptyPerfData)
{
    FedProf perf;
    perf.PerCoreProfResetMetadata(0);
}

TEST(DevicePerfTest, ResetMetadata_IndexOutOfRange)
{
    FedProf perf;
    perf.perCoreProfDevAddr_ = 0x1000;
    perf.PerCoreProfResetMetadata(5);
    perf.perCoreProfDevAddr_ = 0;
}

TEST(DevicePerfTest, SyncData_DebugDisabled)
{
    FedProf perf;
    perf.PerCoreProfSync(false);
}

TEST(DevicePerfTest, ReleaseData_NullPtrs)
{
    FedProf perf;
    perf.perCoreProfDevAddr_ = 0;
    perf.PerCoreProfReleaseData();
    EXPECT_EQ(perf.perCoreProfDevAddr_, 0u);
}

TEST(DevicePerfTest, GetCoreCount_Default)
{
    FedProf perf;
    FedProfDeviceArgs args{};
    args.nrAic = 0;
    args.nrAiv = 0;
    perf.args_ = args;
    EXPECT_EQ(perf.GetCoreCount(), 1u);
}

TEST(DevicePerfTest, StartMachinePerfTraceDumpThread_ZeroAddr)
{
    FedProf perf;
    perf.aicpuProfDevAddr_ = 0;
    perf.AicpuProfDumpThreadStart();
    EXPECT_FALSE(perf.dumpThread_.joinable());
}

TEST(DevicePerfTest, ResetData_WithPerfData)
{
    FedProf perf;
    FedProfDeviceArgs args{};
    args.nrAic = 1;
    args.nrAiv = 0;
    perf.args_ = args;
    perf.perCoreProfDevAddr_ = 0x1000;
    perf.PerCoreProfResetData();
    perf.perCoreProfDevAddr_ = 0;
}

TEST(DevicePerfTest, ResetMetadata_WithAicpuPerfAddr)
{
    FedProf perf;
    FedProfDeviceArgs args{};
    args.nrAic = 1;
    args.nrAiv = 0;
    perf.args_ = args;
    perf.aicpuProfDevAddr_ = 0x2000;
    perf.perCoreProfDevAddr_ = 0x1000;
    perf.PerCoreProfResetMetadata(0);
    EXPECT_TRUE(perf.perCoreProfDataInited_);
    perf.perCoreProfDevAddr_ = 0;
}

TEST(DevicePerfTest, ResetMetadata_WithAicpuPerfAddr_AlreadyInited)
{
    FedProf perf;
    FedProfDeviceArgs args{};
    args.nrAic = 1;
    args.nrAiv = 0;
    perf.args_ = args;
    perf.aicpuProfDevAddr_ = 0x2000;
    perf.perCoreProfDevAddr_ = 0x1000;
    perf.perCoreProfDataInited_ = true;
    perf.PerCoreProfResetMetadata(0);
    perf.perCoreProfDevAddr_ = 0;
}

TEST(DevicePerfTest, ResetMetadata_WithoutAicpuPerfAddr)
{
    FedProf perf;
    FedProfDeviceArgs args{};
    args.nrAic = 1;
    args.nrAiv = 0;
    perf.args_ = args;
    perf.perCoreProfDevAddr_ = 0x1000;
    perf.PerCoreProfResetMetadata(0);
    perf.perCoreProfDevAddr_ = 0;
}
