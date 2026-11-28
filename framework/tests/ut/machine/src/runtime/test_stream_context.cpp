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
 * \file test_stream_context.cpp
 * \brief UT for machine/runtime/context/stream_context: the instance must stay process-wide.
 */

#include <gtest/gtest.h>

#include <thread>

#include "machine/runtime/context/stream_context.h"

using namespace npu::tile_fwk;

TEST(StreamContextTest, InstanceIsProcessWide)
{
    StreamContext* fromMain = &GetStreamContext();
    StreamContext* fromWorker = nullptr;

    std::thread worker([&fromWorker]() { fromWorker = &GetStreamContext(); });
    worker.join();

    ASSERT_NE(fromWorker, nullptr);
    EXPECT_EQ(fromMain, fromWorker);
    EXPECT_EQ(fromMain->GetScheStream(), fromWorker->GetScheStream());
    EXPECT_EQ(fromMain->GetCtrlStream(), fromWorker->GetCtrlStream());
}
