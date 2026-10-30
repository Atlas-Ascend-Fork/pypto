/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TILEFWK_HIF4_SCRATCH_H
#define TILEFWK_HIF4_SCRATCH_H
#include <cstddef>
#ifdef __CCE_AICORE__
#define HIF4_CONSTEXPR [ host, aicore ] constexpr
#else
#define HIF4_CONSTEXPR constexpr
#endif
namespace hif4 {
HIF4_CONSTEXPR size_t Align32(size_t bytes) { return (bytes + 31) / 32 * 32; }
HIF4_CONSTEXPR size_t Align64(size_t elements) { return (elements + 63) / 64 * 64; }
struct ScratchLayout {
    size_t cols, elements, scaleBytes, max8Bytes, metaBytes;
    size_t max8Offset, eaOffset, ebOffset, ecOffset, bytes;
    HIF4_CONSTEXPR ScratchLayout(size_t rows, size_t capacityCols)
        : cols(Align64(capacityCols)),
          elements(rows * cols),
          scaleBytes(Align32(elements)),
          max8Bytes(Align32(((elements + 7) / 8) * 2)),
          metaBytes(((elements + 511) / 512) * 32),
          max8Offset(scaleBytes),
          eaOffset(max8Offset + max8Bytes),
          ebOffset(eaOffset + metaBytes),
          ecOffset(ebOffset + metaBytes),
          bytes(ecOffset + metaBytes)
    {}
};

} // namespace hif4
#undef HIF4_CONSTEXPR
#endif
