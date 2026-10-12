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

#include <cstdint>
#include <memory>

#include "interface/function/function.h"
#include "tilefwk/platform.h"

namespace npu {
namespace tile_fwk {

// RAII 平台架构切换: 作用域内临时切换 NPUArch, 作用域结束恢复进入前架构。
// 仅切换 arch 字段, 派生 SoC 状态(核数/内存上限等)保持环境值, 适用于只读 arch 分支的用例。
class NpuArchScope {
public:
    explicit NpuArchScope(NPUArch arch) : savedArch_(Platform::Instance().GetSoc().GetNPUArch())
    {
        Platform::Instance().GetSoc().SetNPUArch(arch);
    }
    ~NpuArchScope() { Platform::Instance().GetSoc().SetNPUArch(savedArch_); }
    NpuArchScope(const NpuArchScope&) = delete;
    NpuArchScope& operator=(const NpuArchScope&) = delete;

private:
    NPUArch savedArch_;
};

inline uint32_t CountOpcode(const std::shared_ptr<Function>& currFunctionPtr, Opcode opcode)
{
    uint32_t count = 0u;
    for (const auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == opcode) {
            ++count;
        }
    }
    return count;
}

inline const Operation* FindSingleOp(const std::shared_ptr<Function>& currFunctionPtr, Opcode opcode)
{
    for (const auto& op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == opcode) {
            return &op;
        }
    }
    return nullptr;
}

} // namespace tile_fwk
} // namespace npu
