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
 * \file pass_attr_defs.h
 * \brief Pass attribute name definitions.
 */

#pragma once

namespace npu::tile_fwk {
inline constexpr const char* OP_ATTR_VALID_SHAPE = "op_attr_validShape";
inline constexpr const char* ATOMIC_FROM_REDUCE_ACC_ATTR = "op_attr_atomic_from_reduce_acc";
inline constexpr const char* ATOMIC_FROM_EXPLICIT_RMW_ATTR = "op_attr_atomic_from_explicit_rmw";
inline constexpr const char* SKIP_REMOVE_REDUNDANT_OP_VIEW_CHECK_ATTR = "op_attr_skip_remove_redundant_op_view_check";
} // namespace npu::tile_fwk
