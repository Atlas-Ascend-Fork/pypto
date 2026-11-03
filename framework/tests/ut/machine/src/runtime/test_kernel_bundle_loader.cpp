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

#include <array>
#include <cstdint>
#include <new>

#include "machine/runtime/bundle/kernel_bundle_loader.h"
#include "tilefwk/aikernel_root_function.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::bundle;

TEST(KernelBundleLoaderTest, HostCachePreservesTaggedStitchPointers)
{
    // Keep allocations alive and vary their sizes so ordinary malloc alignment cannot hide the regression.
    std::array<LoadedBundle, 16> bundles;
    for (size_t i = 0; i < bundles.size(); ++i) {
        SCOPED_TRACE(i);
        auto& cache = bundles[i].ctrlFlowCache;
        cache.resize(2 * DUPPED_STITCH_NODE_ALIGN + i * sizeof(uint64_t));
        ASSERT_EQ(reinterpret_cast<uintptr_t>(cache.data()) % DUPPED_STITCH_NODE_ALIGN, 0u);

        auto* head = new (cache.data()) DevAscendFunctionDuppedStitchNode{};
        auto* tail = new (cache.data() + DUPPED_STITCH_NODE_ALIGN) DevAscendFunctionDuppedStitchNode{};
        // The low six bits hold a successor count; decoding must preserve the actual node address.
        constexpr uintptr_t remainCount = 1;
        head->NextRaw() = reinterpret_cast<DevAscendFunctionDuppedStitchNode*>(reinterpret_cast<uintptr_t>(tail) |
                                                                               remainCount);
        EXPECT_EQ(head->Next(), tail);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(head->NextRaw()) & DUPPED_STITCH_NODE_REMAIN_COUNT_MASK, remainCount);
    }
}
