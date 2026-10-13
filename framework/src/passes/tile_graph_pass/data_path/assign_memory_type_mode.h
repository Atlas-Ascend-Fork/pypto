/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TILE_FWK_ASSIGN_MEMORY_TYPE_MODE_H
#define TILE_FWK_ASSIGN_MEMORY_TYPE_MODE_H

#include "interface/operation/opcode.h"

namespace npu::tile_fwk {

enum class AssignMemoryTypeMode {
    VIEW_ASSEMBLE,
    SLICE_CONTRACT,
};

inline Opcode GetMoveInOpcode(AssignMemoryTypeMode mode)
{
    return mode == AssignMemoryTypeMode::SLICE_CONTRACT ? Opcode::OP_SLICE : Opcode::OP_VIEW;
}

inline Opcode GetMoveOutOpcode(AssignMemoryTypeMode mode)
{
    return mode == AssignMemoryTypeMode::SLICE_CONTRACT ? Opcode::OP_CONTRACT : Opcode::OP_ASSEMBLE;
}

inline bool IsModeMoveIn(AssignMemoryTypeMode mode, Opcode opcode) { return opcode == GetMoveInOpcode(mode); }

inline bool IsModeMoveOut(AssignMemoryTypeMode mode, Opcode opcode) { return opcode == GetMoveOutOpcode(mode); }

inline bool IsModeMoveOp(AssignMemoryTypeMode mode, Opcode opcode)
{
    return IsModeMoveIn(mode, opcode) || IsModeMoveOut(mode, opcode);
}

} // namespace npu::tile_fwk

#endif // TILE_FWK_ASSIGN_MEMORY_TYPE_MODE_H
