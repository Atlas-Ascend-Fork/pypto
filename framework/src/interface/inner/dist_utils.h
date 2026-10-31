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
// Scalar field index -> Window member name (nullptr = not a scalar field).
constexpr const char* WindowScalarFieldName(int32_t idx)
{
    switch (idx) {
        case 0:
            return "rank";
        case 1:
            return "nrank";
        case 2:
            return "num_signals";
        case 3:
            return "size";
        case 4:
            return "jetty_ptrs";
        case 5:
            return "buffer_ptr";
        case 6:
            return "signal_offset";
        case 7:
            return "barrier_offset";
        default:
            return nullptr; // 8: remote_ptrs is an inline array, not a scalar field
    }
}

// Scalar fields 0-2 (rank, nrank, num_signals) are int32; every other member is int64.
constexpr bool WindowFieldIsInt32(int32_t idx) { return idx >= 0 && idx <= 2; }
} // namespace dist
