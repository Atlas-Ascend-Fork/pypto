/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#pragma once

#include <cstdint>

#include "ir/expr.h"

namespace pypto::ir {
enum class CoreMask : uint8_t { None = 0b00, Cube = 0b01, Vector = 0b10, Both = 0b11 };

// Unclassified calls allow both cores. None means no core satisfies the constraints.
CoreMask GetAllowedCores(const CallPtr& call);
void ValidateCorePlacement(const CallPtr& call, CoreMask allowed, CoreMask required);
} // namespace pypto::ir
