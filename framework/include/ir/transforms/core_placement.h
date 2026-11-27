/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#pragma once

#include <cstdint>

#include "ir/expr.h"

namespace pypto::ir {
enum class CoreMask : uint8_t {
    None = 0b000,
    Cube = 0b001,
    Vector = 0b010,
    CV = 0b011,    // both AI cores (Cube + Vector)
    Aicpu = 0b100, // the control CPU alone
    All = 0b111,   // any execution unit: both AI cores and the control CPU
};

// Unclassified calls allow both cores. None means no core satisfies the constraints.
CoreMask GetAllowedCores(const CallPtr& call);
void ValidateCorePlacement(const CallPtr& call, CoreMask allowed, CoreMask required);
} // namespace pypto::ir
