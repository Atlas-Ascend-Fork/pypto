/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details.
 * You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>

namespace pypto::dist {
constexpr int kNumMaxRanks = 256;         // max ranks per window (remote_ptrs array size)
constexpr int64_t kL2CacheLineSize = 512; // L2 cache line size, one signal slot per line

constexpr int32_t kPolicyUbMem = 1; // direct peer-VA memory-semantic tier
constexpr int32_t kPolicyUrma = 2;  // jetty / WQE remote-memory-access tier

struct RegionEntry {
    int64_t remote_ptrs[kNumMaxRanks]; // per-rank UB_MEM base; 0 if the region is not on that plane
    int64_t size;                      // bytes; identical on every rank
    int32_t policy;                    // kPolicyUbMem | kPolicyUrma bitmask
    int32_t urma_index;                // position in the channel remote-handle list; -1 = not URMA
};

struct Window {
    int32_t rank;
    int32_t nrank;
    int32_t num_jetties;  // local jetties: multi-issuer = AIV count; per-peer = nrank - 1
    int32_t multi_issuer; // 0: one jetty per peer 1: one jetty per AIV
    int32_t num_regions;  // control region included; == 1 + user regions
    int32_t reserved;     // keep the descriptor 8-byte aligned
    int64_t region_table; // GM address of RegionEntry[num_regions]
    int64_t jetty_ptrs;   // GM address of the channel table
};

// Control-region layout (relative to region 0's base): the Barrier is the first 512B
// slot (offset 0); signal slot i follows at (i + 1) * 512B.
struct Barrier {
    alignas(kL2CacheLineSize) int64_t barrier_counter; // offset 0
    int32_t barrier_signals[2];                        // offsets 8, 12
};
} // namespace pypto::dist
