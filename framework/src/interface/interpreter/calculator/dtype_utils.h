/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <array>
#include <cstdint>

#include "tilefwk/data_type.h"

namespace npu::tile_fwk {

inline bool IsFp8Dtype(DataType t)
{
    return t == DT_FP8 || t == DT_HF8 || t == DT_FP8E4M3 || t == DT_FP8E5M2 || t == DT_FP8E8M0 || t == DT_FP8E6M2;
}

inline bool IsFp4PackedDtype(DataType t) { return t == DT_FP4_E2M1 || t == DT_FP4_E1M2 || t == DT_HF4; }

// FP4E2M1 nibble = [S][E2][M1], value = (-1)^S * (1 + M/2) * 2^(E-1), codes 0..7 >= 0 and 8..15 < 0.
inline constexpr std::array<float, 16> FP4_E2M1_DECODE_TABLE = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// FP4E1M2 nibble = [S][E1][M2], value = (-1)^S * (1 + M/4) * 2^E, range [-1.75, 1.75] in steps of 0.25.
inline constexpr std::array<float, 16> FP4_E1M2_DECODE_TABLE = {
    0.0f, 0.25f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 1.75f, -0.0f, -0.25f, -0.5f, -0.75f, -1.0f, -1.25f, -1.5f, -1.75f,
};

// HF4 nibble = [S][I][F1][F2], value = (-1)^S * (I + F1/2 + F2/4), range [-1.75, 1.75] in steps of 0.25.
inline constexpr std::array<float, 16> HF4_DECODE_TABLE = {
    0.0f, 0.25f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 1.75f, -0.0f, -0.25f, -0.5f, -0.75f, -1.0f, -1.25f, -1.5f, -1.75f,
};

inline float DecodeFp4E2M1Nibble(uint8_t nib) { return FP4_E2M1_DECODE_TABLE[static_cast<size_t>(nib & 0x0F)]; }

inline float DecodeFp4E1M2Nibble(uint8_t nib) { return FP4_E1M2_DECODE_TABLE[static_cast<size_t>(nib & 0x0F)]; }

inline float DecodeHf4Nibble(uint8_t nib) { return HF4_DECODE_TABLE[static_cast<size_t>(nib & 0x0F)]; }

// Decode one 4-bit nibble of a packed FP4 storage byte; non-FP4 types fall back to the raw nibble value.
inline float DecodeFp4Nibble(uint8_t nib, DataType actualType)
{
    switch (actualType) {
        case DT_FP4_E2M1:
            return DecodeFp4E2M1Nibble(nib);
        case DT_FP4_E1M2:
            return DecodeFp4E1M2Nibble(nib);
        case DT_HF4:
            return DecodeHf4Nibble(nib);
        default:
            return static_cast<float>(nib);
    }
}

} // namespace npu::tile_fwk
