/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "machine/runtime/runner/aicore_launch_args.h"

#include <algorithm>
#include <cstring>

#include "securec.h"
#include "machine/utils/dynamic/dev_encode_program.h"
#include "machine/utils/dynamic/dev_encode_program_ctrlflow_cache.h"
#include "machine/utils/dynamic/dev_encode_tensor.h"

namespace npu::tile_fwk::dynamic {

void InitAicoreHostInputInfo(RtHostInputInfo& info)
{
    info.addrOffset = static_cast<uint32_t>(kAicoreInputsSlot * sizeof(void*));
    info.dataOffset = static_cast<uint32_t>(AicoreLaunchSlotsBytes());
}

void PrepareAicoreLaunchArgsBuffer(std::vector<uint8_t>& aicoreArgBuf, AiCpuArgs* aicpuArgs, const char* displayName,
                                   size_t arenaBytes, RtArgsEx& rtAicoreArgs)
{
    if (aicpuArgs == nullptr || displayName == nullptr) {
        rtAicoreArgs.args = nullptr;
        rtAicoreArgs.argsSize = 0;
        rtAicoreArgs.hostInputInfoNum = 0;
        rtAicoreArgs.hostInputInfoPtr = nullptr;
        return;
    }
    const size_t needTotal = AicoreLaunchArgsBytes(arenaBytes);
    if (aicoreArgBuf.capacity() < needTotal) {
        const size_t growTo = std::max(needTotal, aicoreArgBuf.capacity() + (aicoreArgBuf.capacity() / 2));
        aicoreArgBuf.reserve(growTo);
    }
    if (aicoreArgBuf.size() != needTotal) {
        aicoreArgBuf.resize(needTotal);
    }

    auto* slots = reinterpret_cast<void**>(aicoreArgBuf.data());
    int64_t* inputp = reinterpret_cast<int64_t*>(aicpuArgs + 1);
    slots[0] = const_cast<char*>(displayName);
    // slots[1] is RT-patched to the arena device addr when hostInputInfoNum==1; must be null otherwise.
    slots[1] = nullptr;
    slots[2] = nullptr;
    slots[3] = nullptr;
    slots[4] = inputp;
    slots[5] = aicpuArgs->kArgs.cfgdata;
    slots[6] = reinterpret_cast<DevTensorData*>(inputp + DEV_TENSOR_DATA_OFFSET);

    rtAicoreArgs.args = aicoreArgBuf.data();
    rtAicoreArgs.argsSize = static_cast<uint32_t>(needTotal);
    rtAicoreArgs.isNoNeedH2DCopy = 0;
    rtAicoreArgs.hostInputInfoNum = 0;
    rtAicoreArgs.hostInputInfoPtr = nullptr;
}

void EnsureHostCtrlFlowCacheRebased(DevControlFlowCache* hostCache, DevAscendProgram* devProg)
{
    if (hostCache == nullptr || devProg == nullptr) {
        return;
    }
    const uint64_t hostBase = reinterpret_cast<uint64_t>(hostCache);
    if (!hostCache->isRelocMetaDev) {
        hostCache->RelocMetaCache(0, hostBase);
        hostCache->isRelocMetaDev = true;
    }
    if (!hostCache->isRelocDataDev) {
        hostCache->TaskAddrRelocProgramAndCtrlCache(0, 0, reinterpret_cast<uint64_t>(devProg), hostBase);
        hostCache->isRelocDataDev = true;
    }
}

void FillAicoreIoAddressArenaBuffer(std::vector<uint8_t>& aicoreArgBuf, AiCpuArgs* aicpuArgs,
                                    DevControlFlowCache* hostCache, uint64_t inputTensorSize, uint64_t outputTensorSize,
                                    RtArgsEx& rtAicoreArgs, RtHostInputInfo& aicoreHostInfo)
{
    if (hostCache == nullptr || aicpuArgs == nullptr) {
        return;
    }
    int64_t* inputp = reinterpret_cast<int64_t*>(aicpuArgs + 1);
    auto* arena = reinterpret_cast<uint64_t*>(aicoreArgBuf.data() + AicoreLaunchSlotsBytes());
    DevStartArgsBase startBase{};
    startBase.devTensorList = reinterpret_cast<DevTensorData*>(inputp + DEV_TENSOR_DATA_OFFSET);
    startBase.inputTensorSize = inputTensorSize;
    startBase.outputTensorSize = outputTensorSize;
    hostCache->MaterializeIncastOutcastAddrs(0, reinterpret_cast<uint64_t>(aicpuArgs->kArgs.workspace), &startBase,
                                             arena);

    InitAicoreHostInputInfo(aicoreHostInfo);
    rtAicoreArgs.hostInputInfoPtr = &aicoreHostInfo;
    rtAicoreArgs.hostInputInfoNum = 1;
}

} // namespace npu::tile_fwk::dynamic
