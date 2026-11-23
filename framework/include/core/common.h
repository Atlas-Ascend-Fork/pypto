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
 * \file common.h
 * \brief Common macros, constants, and utility definitions
 *
 * This header provides commonly used macros and constants that are shared
 * across the PyPTO codebase, including:
 * - Compiler hints and attributes
 * - Utility macros for code generation
 * - Build configuration constants
 * - nanobind module configuration
 */

#pragma once
#include <cstdint>

namespace npu::tile_fwk {

constexpr int32_t NUM_VALUE_2 = 2;
constexpr int32_t NUM_VALUE_3 = 3;
constexpr int32_t NUM_VALUE_4 = 4;
constexpr int32_t NUM_VALUE_5 = 5;
constexpr int32_t NUM_VALUE_6 = 6;
constexpr int32_t NUM_VALUE_7 = 7;
constexpr int32_t NUM_VALUE_8 = 8;
constexpr int32_t NUM_VALUE_9 = 9;
constexpr int32_t NUM_VALUE_10 = 10;
constexpr int32_t NUM_VALUE_12 = 12;
constexpr int32_t NUM_VALUE_16 = 16;
constexpr int32_t NUM_VALUE_21 = 21;
constexpr int32_t NUM_VALUE_22 = 22;
constexpr int32_t NUM_VALUE_26 = 26;
constexpr int32_t NUM_VALUE_31 = 31;
constexpr int32_t NUM_VALUE_32 = 32;
constexpr int32_t NUM_VALUE_34 = 34;
constexpr int32_t NUM_VALUE_64 = 64;
constexpr int32_t NUM_VALUE_128 = 128;
constexpr int32_t NUM_VALUE_256 = 256;
constexpr double NUM_VALUE_0_5 = 0.5;
constexpr double NUM_VALUE_EPS = 1e-9;

} // namespace npu::tile_fwk

namespace pypto {

using npu::tile_fwk::NUM_VALUE_0_5;
using npu::tile_fwk::NUM_VALUE_10;
using npu::tile_fwk::NUM_VALUE_12;
using npu::tile_fwk::NUM_VALUE_128;
using npu::tile_fwk::NUM_VALUE_16;
using npu::tile_fwk::NUM_VALUE_2;
using npu::tile_fwk::NUM_VALUE_21;
using npu::tile_fwk::NUM_VALUE_22;
using npu::tile_fwk::NUM_VALUE_256;
using npu::tile_fwk::NUM_VALUE_26;
using npu::tile_fwk::NUM_VALUE_3;
using npu::tile_fwk::NUM_VALUE_31;
using npu::tile_fwk::NUM_VALUE_32;
using npu::tile_fwk::NUM_VALUE_34;
using npu::tile_fwk::NUM_VALUE_4;
using npu::tile_fwk::NUM_VALUE_5;
using npu::tile_fwk::NUM_VALUE_6;
using npu::tile_fwk::NUM_VALUE_64;
using npu::tile_fwk::NUM_VALUE_7;
using npu::tile_fwk::NUM_VALUE_8;
using npu::tile_fwk::NUM_VALUE_9;
using npu::tile_fwk::NUM_VALUE_EPS;

// ============================================================================
// IR Constants
// ============================================================================

// Dynamic dimension constant for tensor/tile shapes
// Use -1 to represent dimensions that are unknown at compile time
constexpr int64_t kDynamicDim = -1;

// ============================================================================
// Compiler Hints and Attributes
// ============================================================================

#define PYPTO_ALWAYS_INLINE __attribute__((always_inline))
#define PYPTO_UNUSED __attribute__((unused))
#define PYPTO_STR_CONCAT_IMPL(__x, __y) __x##__y
#define PYPTO_STR_CONCAT(__x, __y) PYPTO_STR_CONCAT_IMPL(__x, __y)
} // namespace pypto
