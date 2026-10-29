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
 * \file kernel_bundle_launch.cpp
 * \brief Self-contained offline bundle launch (see header). Kept out of device_launcher.cpp on purpose.
 */

#include "machine/runtime/bundle/kernel_bundle_launch.h"

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "securec.h"
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"
#include "adapter/api/acl_api.h"
#include "adapter/api/msprof_api.h"
#include "interface/utils/op_info_manager.h"
#include "machine/runtime/bundle/kernel_bundle_dev_cache.h"
#include "machine/runtime/context/stream_context.h"
#include "machine/runtime/context/device_launcher_context.h"
#include "machine/runtime/runner/runtime_utils.h"
#include "machine/runtime/runner/load_aicpu_op.h"
#include "machine/runtime/runner/host_prof.h"
#include "machine/runtime/runner/aicore_launch_args.h"
#include "machine/runtime/launcher/device_launcher.h"
#include "machine/runtime/launcher/cell_match_dynamic.h"
#include "machine/runtime/memory_utils/device_memory_utils.h"
#include "machine/utils/dynamic/dev_encode_program_ctrlflow_cache.h"
#include "machine/utils/dynamic/dev_encode_tensor.h"

namespace npu::tile_fwk::dynamic {

namespace {
int BundleLaunchAicpu(AicpuLaunchDesc& launchDesc, DevAscendProgram* devProg)
{
    auto ctrlStream = GetStreamContext().GetCtrlStream();
    auto schedStream = GetStreamContext().GetScheStream();
    int ret = 0;
    auto args = launchDesc.args;
    const int nrAicpu = static_cast<int>(devProg->devArgs.nrAicpu);
    const bool launchSchedSameCluster = static_cast<int>(devProg->devArgs.launchSchedSameCluster);
    if (launchSchedSameCluster) {
        MACHINE_LOGW("When available AICPUs are insufficient, execute export PYPTO_LAUNCH_SCHED_SAME_CLUSTER=false "
                     "to disable the constraint that forces scheduling threads onto the same cluster.");
    }
    args->kArgs.parameter.ctrlBlockNum = static_cast<int>(devProg->ctrlBlockDim);
    auto startTime = MspfSysCycleTime();
    args->kArgs.parameter.runMode = RUN_SPLITTED_STREAM_CTRL;
    launchDesc.stream = ctrlStream;
    launchDesc.blockDim = 1U;
    ret = LaunchPyptoRunWithHostArgs(launchDesc);
    HostProf::GetInstance().ReportHostProfInfo(ctrlStream, startTime, 1, MSPF_GE_TASK_TYPE_AI_CPU, false);
    if (ret != RT_SUCCESS) {
        return ret;
    }
    if (devProg->devArgs.enableAicoreResolve) {
        return ret;
    }
    args->kArgs.parameter.runMode = RUN_SPLITTED_STREAM_SCHE;
    startTime = MspfSysCycleTime();
    const int scheCpuNum = static_cast<int>(devProg->devArgs.scheCpuNum);
    launchDesc.stream = schedStream;
    launchDesc.blockDim = static_cast<uint32_t>(nrAicpu);
    ret = LaunchPyptoRunWithHostArgs(launchDesc);
    HostProf::GetInstance().ReportHostProfInfo(schedStream, startTime, scheCpuNum, MSPF_GE_TASK_TYPE_AI_CPU, false);
    return ret;
}

int BundleLaunchAicore(AclRtStream aicoreStream, void* kernel, RtArgsEx& rtArgs, RtTaskCfgInfo& rtTaskCfg,
                       DevAscendProgram* devProg)
{
    auto tilingKey = OpInfoManager::GetInstance().GetOpTilingKey();
    int blockDim = static_cast<int>(devProg->ctrlBlockDim);
    if (blockDim == 0) {
        blockDim = static_cast<int>(devProg->devArgs.nrValidAic);
    }
    auto startTime = MspfSysCycleTime();
    auto ret = RuntimeKernelLaunchWithHandleV2(kernel, tilingKey, blockDim, &rtArgs, nullptr, aicoreStream, &rtTaskCfg);
    HostProf::GetInstance().ReportHostProfInfo(aicoreStream, startTime, blockDim, MSPF_GE_TASK_TYPE_MIX_AIC, true);
    return ret;
}

std::string MakeAicpuOpName(const std::string& displayName, uint64_t bundleKey)
{
    constexpr size_t maxNameSize = sizeof(AiCpuArgs::opName) - 1;
    if (displayName.size() <= maxNameSize) {
        return displayName;
    }
    char keySuffix[18] = {};
    const int ret = snprintf_s(keySuffix, sizeof(keySuffix), sizeof(keySuffix) - 1, "_%016llx",
                               static_cast<unsigned long long>(bundleKey));
    MACHINE_ASSERT(ret >= 0) << "[kernel-bundle] failed to format AICPU op name suffix";
    const size_t prefixSize = maxNameSize - std::strlen(keySuffix);
    return displayName.substr(0, prefixSize) + keySuffix;
}

size_t BundleAicpuArgsBytes(size_t tensorCount, size_t cellMatchPatchCount)
{
    return sizeof(AiCpuArgs) + 2 * sizeof(int64_t) + tensorCount * sizeof(DevTensorData) + sizeof(uint64_t) +
           cellMatchPatchCount * sizeof(DevDynamicCellMatchStridePatch);
}

int BundleInitRuntime()
{
    int rc = AclInit(nullptr);
    if (rc != 0 && rc != ACLRT_ERROR_REPEAT_INITIALIZE) {
        return rc;
    }
    DeviceLauncher::CheckAscendDriverVersionOnboard();
    CheckDeviceId();
    return 0;
}

AiCpuArgs* BundleBuildAicpuArgs(std::vector<int64_t>& aicpuArgBuf, size_t tensorCount, size_t cellMatchPatchCount,
                                const std::string& aicpuOpName)
{
    const size_t argSize = BundleAicpuArgsBytes(tensorCount, cellMatchPatchCount);
    aicpuArgBuf.assign((argSize + sizeof(int64_t) - 1) / sizeof(int64_t), 0);
    auto* aicpuArgs = new (aicpuArgBuf.data()) AiCpuArgs();
    aicpuArgs->kArgs.inputs = nullptr;
    aicpuArgs->kArgs.outputs = nullptr;
    if (strcpy_s(aicpuArgs->opName, sizeof(aicpuArgs->opName), aicpuOpName.c_str()) != 0) {
        MACHINE_LOGE(DevCommonErr::PARAM_CHECK_FAILED, "[kernel-bundle] invalid AICPU op name: %s",
                     aicpuOpName.c_str());
        return nullptr;
    }
    return aicpuArgs;
}

void BundleFillTensorList(AiCpuArgs* aicpuArgs, const std::vector<DeviceTensorData>& tensorList,
                          const std::vector<DevDynamicCellMatchStridePatch>& cellMatchStridePatches)
{
    int64_t* inputp = reinterpret_cast<int64_t*>(aicpuArgs + 1);
    inputp[0] = static_cast<int64_t>(tensorList.size());
    inputp[1] = 0; // nOut: outputs are folded into tensorList
    auto* tensorData = reinterpret_cast<DevTensorData*>(inputp + DEV_TENSOR_DATA_OFFSET);
    for (const auto& t : tensorList) {
        tensorData->address = reinterpret_cast<uint64_t>(t.GetAddr());
        tensorData->dataType = t.GetDataType();
        const auto& shape = t.GetShape();
        tensorData->shape.dimSize = static_cast<int>(shape.size());
        for (int j = 0; j < tensorData->shape.dimSize; ++j) {
            tensorData->shape.dim[j] = shape[j];
        }
        tensorData++;
    }
    WriteDynamicCellMatchStridePatchesToLaunchArgs(inputp, cellMatchStridePatches);
}

void BundleFillPerLaunchKArgs(AiCpuArgs* aicpuArgs, DevAscendProgram* devProg, uint8_t* devCtrlCache,
                              void* workspaceAddr, uint64_t cacheKey)
{
    static std::atomic<int64_t> bundleSequence{0};
    aicpuArgs->kArgs.ctrlFlowCache = reinterpret_cast<int64_t*>(devCtrlCache);
    aicpuArgs->kArgs.workspace = reinterpret_cast<int64_t*>(workspaceAddr);
    aicpuArgs->kArgs.parameter.globalRound = ++bundleSequence;
    aicpuArgs->kArgs.maxDynamicAssembleOutcastMem = devProg->memBudget.tensor.maxDynamicAssembleOutcastMem;
    aicpuArgs->kArgs.maxDynamicCellMatchTableMem = devProg->memBudget.metadata.maxDynamicCellMatchTableMem;

    const uint64_t dynamicCellMatchBytes = devProg->memBudget.metadata.dynamicCellMatch;
    uint8_t* cellMatchAddr = bundle::KernelBundleDevCache::Instance().GetOrAllocCellMatch(cacheKey,
                                                                                          dynamicCellMatchBytes);
    aicpuArgs->kArgs.runtimeDynamicCellMatchAddr = reinterpret_cast<uint64_t>(cellMatchAddr);
    aicpuArgs->kArgs.runtimeDynamicCellMatchCapacity = cellMatchAddr != nullptr ? dynamicCellMatchBytes : 0;
}

struct BundleAicpuLaunchState {
    std::vector<int64_t> argBuf;
    AiCpuArgs* args{nullptr};
    AicpuLaunchDesc launchDesc{};
    RtHostInputInfo hostInfo{};
};

bool BundlePrepareAicpuLaunch(BundleAicpuLaunchState& state, const std::vector<uint8_t>& devProgBinary,
                              DevAscendProgram* devProg, const std::vector<DeviceTensorData>& tensorList,
                              const std::vector<DevDynamicCellMatchStridePatch>& cellMatchStridePatches,
                              uint8_t* devCtrlCache, void* workspaceAddr, uint64_t cacheKey,
                              const std::string& aicpuOpName)
{
    state.args = BundleBuildAicpuArgs(state.argBuf, tensorList.size(), cellMatchStridePatches.size(), aicpuOpName);
    if (state.args == nullptr) {
        return false;
    }
    DeviceMemoryUtils devMem;
    DeviceLauncher::FillDeviceKernelArgs(devMem, const_cast<std::vector<uint8_t>&>(devProgBinary), state.args->kArgs,
                                         std::vector<std::string>{});
    BundleFillTensorList(state.args, tensorList, cellMatchStridePatches);
    BundleFillPerLaunchKArgs(state.args, devProg, devCtrlCache, workspaceAddr, cacheKey);

    state.hostInfo.addrOffset = offsetof(AiCpuArgs, kArgs.inputs);
    state.hostInfo.dataOffset = sizeof(AiCpuArgs);
    state.launchDesc = {};
    state.launchDesc.hostInputs = &state.hostInfo;
    state.launchDesc.hostInputNum = 1;
    state.launchDesc.timeout = AICPU_EXECUTE_TIMEOUT;
    state.launchDesc.args = state.args;
    state.launchDesc.argsSize = static_cast<uint32_t>(state.argBuf.size() * sizeof(int64_t));
    return true;
}

struct BundleAicoreLaunchState {
    std::vector<uint8_t> argBuf;
    RtArgsEx rtArgs{};
    RtHostInputInfo hostInfo{};
    RtTaskCfgInfo taskCfg{};
    DevControlFlowCache* hostCache{nullptr};
    size_t arenaBytes{0};
};

void BundlePrepareAicoreArgs(BundleAicoreLaunchState& state, AiCpuArgs* aicpuArgs, DevAscendProgram* devProg,
                             std::vector<uint8_t>& hostCtrlFlowCache, const std::string& displayName)
{
    if (!hostCtrlFlowCache.empty()) {
        state.hostCache = reinterpret_cast<DevControlFlowCache*>(hostCtrlFlowCache.data());
        state.arenaBytes = static_cast<size_t>(state.hostCache->ioAddressArenaBytes);
    }
    // Size/slots only here. Materialize stays after AICPU launch (same order as DeviceLauncher::LaunchKernel).
    PrepareAicoreLaunchArgsBuffer(state.argBuf, aicpuArgs, displayName.c_str(), state.arenaBytes, state.rtArgs);

    (void)memset_s(&state.taskCfg, sizeof(RtTaskCfgInfo), 0, sizeof(RtTaskCfgInfo));
    state.taskCfg.schemMode = static_cast<uint8_t>(RtSchemModeType::BATCH);
    state.taskCfg.localMemorySize = GetAicoreLocalMemorySize(static_cast<NPUArch>(devProg->devArgs.archInfo),
                                                             devProg->requiresSimt);
}

void BundleMaterializeAicoreIoArena(BundleAicoreLaunchState& state, AiCpuArgs* aicpuArgs, DevAscendProgram* devProg)
{
    if (state.arenaBytes == 0 || state.hostCache == nullptr) {
        return;
    }
    EnsureHostCtrlFlowCacheRebased(state.hostCache, devProg);
    // Packed ABI: nIn/nOut come from the recording cache, not the folded [nTensors,0] launch list.
    FillAicoreIoAddressArenaBuffer(state.argBuf, aicpuArgs, state.hostCache,
                                   state.hostCache->inputTensorDataList.size(),
                                   state.hostCache->outputTensorDataList.size(), state.rtArgs, state.hostInfo);
}

int BundleDispatchKernels(AclRtStream aicoreStream, void* binHandle, AicpuLaunchDesc& aicpuLaunchDesc,
                          AiCpuArgs* aicpuArgs, DevAscendProgram* devProg, BundleAicoreLaunchState& aicoreState,
                          bool streamSynchronize)
{
    const bool isCaptureMode = DeviceLauncher::IsCaptureMode();
    int rc = DeviceLauncher::LaunchSyncTask(aicoreStream, isCaptureMode, /*launchEarlyMode=*/0);
    if (rc != RT_SUCCESS) {
        return rc;
    }
    rc = DeviceLauncher::SetDevPerfAddr(false, isCaptureMode, devProg->devArgs.toSubMachineConfig);
    if (rc != RT_SUCCESS) {
        return rc;
    }

    rc = BundleLaunchAicpu(aicpuLaunchDesc, devProg);
    if (rc != RT_SUCCESS) {
        MACHINE_LOGE(HostLauncherErr::LAUNCH_BUILTIN_OP_NULL_FAILED, "[kernel-bundle] launch aicpu failed: %d", rc);
        return rc;
    }

    BundleMaterializeAicoreIoArena(aicoreState, aicpuArgs, devProg);

    rc = BundleLaunchAicore(aicoreStream, static_cast<RtBinHandle>(binHandle), aicoreState.rtArgs, aicoreState.taskCfg,
                            devProg);
    if (rc != RT_SUCCESS) {
        MACHINE_LOGE(HostLauncherErr::REGISTER_KERNEL_FAILED, "[kernel-bundle] launch aicore failed: %d", rc);
        return rc;
    }

    if (streamSynchronize) {
        rc = DeviceLauncher::DynamicLaunchSynchronize(GetContextScheStream(), GetContextCtrlStream(), aicoreStream);
    }
    return rc;
}
} // namespace

int LaunchBundleKernelOnce(const std::vector<uint8_t>& devProgBinary, void* binHandle, uint64_t cacheKey,
                           const std::vector<DeviceTensorData>& tensorList, std::vector<uint8_t>& hostCtrlFlowCache,
                           const std::string& displayName,
                           const std::vector<DevDynamicCellMatchStridePatch>& cellMatchStridePatches,
                           void* workspaceAddr, RtStream aicoreStream, bool streamSynchronize,
                           const DeviceLauncherConfig& config)
{
    (void)config;
    MACHINE_LOGI("Kernel Launch (bundle)");
    if (binHandle == nullptr) {
        MACHINE_LOGE(HostLauncherErr::REGISTER_KERNEL_FAILED, "[kernel-bundle] null kernel bin handle.");
        return -1;
    }
    aicoreStream = aicoreStream == nullptr ? GetContextAiCoreStream() : aicoreStream;
    int rc = BundleInitRuntime();
    if (rc != 0) {
        return rc;
    }

    auto* devProg = reinterpret_cast<DevAscendProgram*>(const_cast<uint8_t*>(devProgBinary.data()));
    // H2D before host rebase so Materialize cannot leak absolute host pointers into device cache.
    uint8_t* devCtrlCache = bundle::KernelBundleDevCache::Instance().GetOrCopy(cacheKey, hostCtrlFlowCache);

    BundleAicpuLaunchState aicpuState;
    if (!BundlePrepareAicpuLaunch(aicpuState, devProgBinary, devProg, tensorList, cellMatchStridePatches, devCtrlCache,
                                  workspaceAddr, cacheKey, MakeAicpuOpName(displayName, cacheKey))) {
        return -1;
    }

    BundleAicoreLaunchState aicoreState;
    BundlePrepareAicoreArgs(aicoreState, aicpuState.args, devProg, hostCtrlFlowCache, displayName);

    rc = BundleDispatchKernels(aicoreStream, binHandle, aicpuState.launchDesc, aicpuState.args, devProg, aicoreState,
                               streamSynchronize);
    MACHINE_LOGI("finish Kernel Launch (bundle).");
    return rc;
}

} // namespace npu::tile_fwk::dynamic
