/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TILEOP_VECTOR_NAN_COMPARE_H
#define TILEOP_VECTOR_NAN_COMPARE_H

#include "utils/sync.h"

// TCMP reads complete 256-byte repeats on A2/A3, including invalid tail lanes.
// Keep that read inside initialized scratch instead of padding a shared input.
template <int H, int W, typename M, typename X>
TILEOP void NanCompareMask(M& mask, X& src, uint64_t address)
{
    using E = typename X::DType;
    constexpr int paddedWidth = (W * sizeof(E) + 255) / 256 * 256 / sizeof(E);
    constexpr int alignedWidth = (W * sizeof(E) + 31) / 32 * 32 / sizeof(E);
    constexpr int copyWidth = alignedWidth < X::Cols ? alignedWidth : X::Cols;
    constexpr int copyRows = H < X::Rows ? H : X::Rows;
    using Full = pto::Tile<pto::TileType::Vec, E, H, paddedWidth>;
    using Valid = pto::Tile<pto::TileType::Vec, E, copyRows, copyWidth, pto::BLayout::RowMajor, -1, -1>;
    Full full;
    Valid valid(src.GetValidRow(), src.GetValidCol());
    pto::TASSIGN(full, address);
    pto::TASSIGN(valid, address);
    pto::TEXPANDS(full, static_cast<E>(0));
    SyncV();
    pto::TEXTRACT(valid, src, 0, 0);
    SyncV();
    pto::TCMP(mask, valid, valid, pto::CmpMode::EQ);
    SyncV();
}

#endif // TILEOP_VECTOR_NAN_COMPARE_H
