/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aicpu_interface.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <dlfcn.h>

#define private public
#include "machine/device/machine_interface/pypto_aicpu_interface.h"
#undef private

using namespace npu::tile_fwk;

extern "C" uint32_t DynPyptoKernelServerNull(void* args);

TEST(PyptoAicpuInterfaceTest, NullInitArgReturnsError)
{
    EXPECT_EQ(DynPyptoKernelServerNull(nullptr), static_cast<uint32_t>(DevCommonErr::NULLPTR));
}

TEST(PyptoAicpuInterfaceTest, ExecuteBeforeLoadReturnsError)
{
    constexpr uint32_t execFailed = static_cast<uint32_t>(ServerKernelErr::KERNEL_EXEC_FAILED);
    EXPECT_EQ(DynPyptoKernelServer(nullptr), execFailed);
    EXPECT_EQ(DynPyptoKernelServerInit(nullptr), execFailed);
}

namespace {
int MockKernelFunc(void*) { return 0; }
} // namespace

class BackendServerHandleManagerTest : public testing::Test {};

TEST_F(BackendServerHandleManagerTest, SaveSoFile_ShortLenAndWriteFail)
{
    BackendServerHandleManager mgr;
    EXPECT_EQ(mgr.SaveSoFile(nullptr, 0), true);
    char data[] = "test_so_data";
    EXPECT_EQ(mgr.SaveSoFile(data, sizeof(data) - 1), true);
    EXPECT_EQ(mgr.firstCreatSo_, true);
    EXPECT_GE(mgr.fileFd_, 0);
    EXPECT_EQ(mgr.pyptoServerSoName_.rfind("/proc/self/fd/", 0), 0U);

    BackendServerHandleManager mgrFail;
    EXPECT_EQ(mgrFail.SaveSoFile(reinterpret_cast<char*>(1), 8192), false);
    EXPECT_EQ(mgrFail.fileFd_, -1);
}

TEST_F(BackendServerHandleManagerTest, SetTileFwkKernelMap_AlreadyLoaded)
{
    BackendServerHandleManager mgrLoaded;
    mgrLoaded.firstLoadSo_ = true;
    mgrLoaded.SetTileFwkKernelMap();
    EXPECT_EQ(mgrLoaded.soHandle_, nullptr);
}

TEST_F(BackendServerHandleManagerTest, LoadTileFwkKernelFunc_DlsymSuccessAndDuplicate)
{
    BackendServerHandleManager mgr;
    mgr.soHandle_ = dlopen(nullptr, RTLD_LAZY);
    ASSERT_NE(mgr.soHandle_, nullptr);
    mgr.LoadTileFwkKernelFunc(dynServerKernelFun);
    EXPECT_NE(mgr.kernelKey2FuncHandle_.count(dyExecFuncKey), 0U);
    mgr.LoadTileFwkKernelFunc(dynServerKernelFun);
    EXPECT_EQ(mgr.kernelKey2FuncHandle_.size(), 1U);
}

TEST_F(BackendServerHandleManagerTest, Destructor_WithAndWithoutHandle)
{
    {
        BackendServerHandleManager mgr;
    }
    void* handle = dlopen(nullptr, RTLD_LAZY);
    ASSERT_NE(handle, nullptr);
    {
        BackendServerHandleManager mgr;
        mgr.soHandle_ = handle;
    }
    SUCCEED();
}

TEST_F(BackendServerHandleManagerTest, GetTileFwkKernelFunc_HitAndMiss)
{
    BackendServerHandleManager mgr;
    mgr.kernelKey2FuncHandle_[dyExecFuncKey] = MockKernelFunc;
    EXPECT_NE(mgr.GetTileFwkKernelFunc(dyExecFuncKey), nullptr);
    EXPECT_EQ(mgr.GetTileFwkKernelFunc(dyInitFuncKey), nullptr);
}
