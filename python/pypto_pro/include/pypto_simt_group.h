/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

namespace pypto::simt {

// Thread block tile
struct ThreadBlockTile {
    unsigned int size;            // Tile thread count
    unsigned int lane_mask;       // Member lane mask
    unsigned int meta_group_rank; // Tile index in the parent group
    unsigned int meta_group_size; // Tile count in the parent group
};

// Coalesced group
struct CoalescedGroup {
    unsigned int lane_mask;       // Captured member lane mask
    unsigned int meta_group_rank; // Subgroup index (0 for coalesced_threads())
    unsigned int meta_group_size; // Subgroup count (1 for coalesced_threads())
};

} // namespace pypto::simt
