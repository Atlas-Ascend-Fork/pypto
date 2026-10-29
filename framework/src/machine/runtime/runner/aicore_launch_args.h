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
 * \file aicore_launch_args.h
 * \brief Shared AICore host-args layout + IO-arena Materialize helpers for KernelBinary and kernel-bundle.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "adapter/api/runtime_define.h"
#include "machine/utils/machine_ws_intf.h"

namespace npu::tile_fwk::dynamic {

class DevControlFlowCache;
struct DevAscendProgram;

inline constexpr size_t kAicoreLaunchArgSlots = 7;
inline constexpr size_t kAicoreInputsSlot = 1; // KernelEntry `inputs`

inline size_t AicoreLaunchSlotsBytes() { return kAicoreLaunchArgSlots * sizeof(void*); }

inline size_t AicoreLaunchArgsBytes(size_t arenaBytes)
{
    const size_t slotsBytes = AicoreLaunchSlotsBytes();
    if (arenaBytes == 0) {
        return slotsBytes;
    }
    return (slotsBytes + arenaBytes + 7ULL) & ~static_cast<size_t>(7ULL);
}

void InitAicoreHostInputInfo(RtHostInputInfo& info);

/* Size buffer, fill 7 launch slots, wire rtAicoreArgs (host-input disabled until Fill). */
void PrepareAicoreLaunchArgsBuffer(std::vector<uint8_t>& aicoreArgBuf, AiCpuArgs* aicpuArgs, const char* displayName,
                                   size_t arenaBytes, RtArgsEx& rtAicoreArgs);

/* Rebase host CF cache mirror once so Materialize can dereference pointers. */
void EnsureHostCtrlFlowCacheRebased(DevControlFlowCache* hostCache, DevAscendProgram* devProg);

/* Write absolute IO addrs into the arena region and enable RtHostInputInfo for `inputs`.
 * inputTensorSize/outputTensorSize must match the recording ABI used by RelocDescFromCache. */
void FillAicoreIoAddressArenaBuffer(std::vector<uint8_t>& aicoreArgBuf, AiCpuArgs* aicpuArgs,
                                    DevControlFlowCache* hostCache, uint64_t inputTensorSize, uint64_t outputTensorSize,
                                    RtArgsEx& rtAicoreArgs, RtHostInputInfo& aicoreHostInfo);

} // namespace npu::tile_fwk::dynamic
