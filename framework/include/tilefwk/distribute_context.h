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

namespace dist {
constexpr int kNumMaxRanks = 256;         // max ranks per window (remote_ptrs array size)
constexpr int64_t kL2CacheLineSize = 512; // L2 cache line size, one signal slot per line

struct Window {
    int rank;
    int nrank;
    int num_signals;  // reserved signal-slot count (win.num_signals())
    int num_jetties;  // local jetties: multi-issuer = AIV count; per-peer = nrank - 1
    int multi_issuer; // 0: one jetty per peer 1: one jetty per AIV
    int64_t size;
    int64_t jetty_ptrs;
    int64_t buffer_ptr;                // this window's data region base
    int64_t signal_offset;             // signal region offset from remote_ptrs[rank], 512B aligned
    int64_t barrier_offset;            // barrier region offset from remote_ptrs[rank] (Barrier layout below)
    int64_t remote_ptrs[kNumMaxRanks]; // peer window base addresses, indexed by rank
};

struct Barrier {
    alignas(kL2CacheLineSize) int64_t barrier_counter; // offset 0
    int32_t barrier_signals[2];                        // offsets 8, 12
};
} // namespace dist
