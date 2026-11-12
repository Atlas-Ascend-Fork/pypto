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

#include "adapter/api/runtime_define.h"
#include "machine/runtime/runner/device_error_tracking.h"
#include "tilefwk/device_error_code.h"
#include "tilefwk/error_manager.h"

using namespace npu::tile_fwk;

class DeviceErrorTrackingTest : public testing::Test {
protected:
    void SetUp() override { ErrorManager::Instance().OutputErrorMessage(true); }
    void TearDown() override { ErrorManager::Instance().OutputErrorMessage(true); }
};

TEST_F(DeviceErrorTrackingTest, AicpuNamesMatchPrefix)
{
    RtExceptionInfo info = {};
    info.expandInfo.type = RtExceptionExpandType::AICPU;
    EXPECT_FALSE(IsPyPTOAicpuException(&info));
    for (const char* name : {"DynTileFwkKernelServer", "DynTileFwkKernelServerInit", "DynTileFwkKernelServerOther",
                             "DynPyptoKernelServer", "DynPyptoKernelServerInit", "DynPyptoKernelServerOther"}) {
        info.expandInfo.u.aicpuInfo.functionName = name;
        EXPECT_TRUE(IsPyPTOAicpuException(&info));
    }
    for (const char* name : {"", "OtherKernel", "DynTileFwkKernel", "DynPyptoKernel"}) {
        info.expandInfo.u.aicpuInfo.functionName = name;
        EXPECT_FALSE(IsPyPTOAicpuException(&info));
    }
}

TEST_F(DeviceErrorTrackingTest, UnrelatedExceptionsAreIgnored)
{
    RtExceptionInfo info = {};
    testing::internal::CaptureStderr();
    PyPTOExceptionInfoCallBack(nullptr);
    info.expandInfo.type = RtExceptionExpandType::FFTS_PLUS;
    PyPTOExceptionInfoCallBack(&info);
    info.expandInfo.type = RtExceptionExpandType::AICORE;
    info.expandInfo.u.aicoreInfo.exceptionArgs.exceptionKernelInfo.kernelName = "OtherKernel";
    PyPTOExceptionInfoCallBack(&info);
    info = {};
    info.expandInfo.type = RtExceptionExpandType::AICPU;
    info.expandInfo.u.aicpuInfo.functionName = "OtherKernel";
    PyPTOExceptionInfoCallBack(&info);
    EXPECT_TRUE(testing::internal::GetCapturedStderr().empty());
}

TEST_F(DeviceErrorTrackingTest, AicpuExceptionIsOutput)
{
    RtExceptionInfo info = {};
    info.expandInfo.type = RtExceptionExpandType::AICPU;
    info.expandInfo.u.aicpuInfo.functionName = "DynTileFwkKernelServerInit";
    info.retcode = PYPTO_DEVICE_ERROR_AICPU_EXCEPTION;
    testing::internal::CaptureStderr();
    PyPTOExceptionInfoCallBack(&info);
    EXPECT_FALSE(testing::internal::GetCapturedStderr().empty());
}

TEST_F(DeviceErrorTrackingTest, AicoreExceptionIsOutput)
{
    RtExceptionInfo info = {};
    info.expandInfo.type = RtExceptionExpandType::AICORE;
    info.expandInfo.u.aicoreInfo.exceptionArgs.exceptionKernelInfo.kernelName = "PyPTO_test_kernel";
    info.retcode = PYPTO_DEVICE_ERROR_AICORE_EXCEPTION;
    testing::internal::CaptureStderr();
    PyPTOExceptionInfoCallBack(&info);
    EXPECT_FALSE(testing::internal::GetCapturedStderr().empty());
}
