/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file common_types.h
 * \brief PyPTO-vendored subset: AscendC::Hardware for dump position tags.
 */
#ifndef IMPL_UTILS_COMMON_TYPES_H
#define IMPL_UTILS_COMMON_TYPES_H

#include <cstdint>
#include "sys_macros_impl.h"

namespace AscendC {
enum class Hardware : uint8_t { GM, UB, L1, L0A, L0B, L0C, BIAS, FIXBUF, MAX };
} // namespace AscendC

#endif // IMPL_UTILS_COMMON_TYPES_H
