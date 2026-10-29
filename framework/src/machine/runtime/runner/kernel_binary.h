/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_binary.h
 * \brief KernelBinary class for managing compiled kernel binary and its runtime resources.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "interface/machine/device/tilefwk/aicpu_common.h"
#include "interface/function/function.h"
#include "machine/runtime/launcher/ctrl_flow_cache_manager.h"
#include "machine/runtime/launcher/device_launcher_types.h"
#include "machine/runtime/runner/load_aicpu_op.h"
#include "machine/utils/dynamic/dev_encode_tensor.h"
#include "machine/utils/machine_ws_intf.h"
#include "adapter/api/runtime_define.h"
#include "adapter/api/acl_define.h"
#include "machine/runtime/runner/host_prof.h"
#include "machine/runtime/runner/fed_prof_manager.h"

namespace npu::tile_fwk::dynamic {

class KernelBinary {
public:
    KernelBinary(std::shared_ptr<Function> func, std::vector<std::shared_ptr<Function>> pinnedGraph = {});
    ~KernelBinary();

    static void SetAiCpuSoOverride(std::vector<uint8_t> soBytes);
    static void InitMetaData(DeviceArgs& devArgs);
    static bool GetEnableDumpDevPref();

    ToSubMachineConfig& GetMachineConfig();

    uint8_t* FindCtrlFlowCache(std::vector<std::vector<int64_t>>& inputs, bool isOriginShape);
    uint8_t* FindCtrlFlowCache(std::vector<DeviceTensorData>& inputs, bool isOriginShape);
    /* storeToCache=false commits the value-depend slot (hash + device ptr + host mirror) in one write. */
    uint8_t* BuildControlFlowCache(std::vector<DeviceTensorData>& inputs, bool isOriginShape, bool storeToCache = true,
                                   uint64_t valueDependHash = 0);

    int64_t GetWorkspaceSize(const std::vector<DeviceTensorData>& tensors);

    std::pair<AiCpuArgs*, int64_t> BuildKernelArgs(const std::vector<DeviceTensorData>& tensors);
    /* After workspace is set: materialize IO addrs into AICore launch host-args arena.
     * Uses activeHostCtrlFlowCache_ bound by FindOrBuildDevCache (no second hash lookup). */
    void MaterializeIoAddressArenaForAicore(AiCpuArgs* aicpuArgs);

    bool CheckArgs(const std::vector<DeviceTensorData>& tensors) const;

    void* GetKernelBin();
    auto& GetArgTypes() { return argTypes; }
    Function* GetFunction();
    const std::string& GetKernelname() const;
    bool DisableHostCtrlFlowCacheBuild() const;
    bool HasValueDepend() const { return devProg != nullptr && devProg->hasValueDepend != 0; }
    const std::vector<size_t>& GetValueDependInputIndices() const;
    uint64_t GetCachedCtrlFlowHash() const { return valueDependCache_.hash; }
    uint8_t* GetValueDependDevCache() const { return valueDependCache_.devCache; }
    void FreeAndClearValueDependCache();
    void BindValueDependActiveHost();
    uint64_t GetMaxDynamicAssembleOutcastMem() const;
    uint64_t GetMaxDynamicCellMatchTableMem() const;
    uint64_t GetRuntimeDynamicCellMatchAddr() const;
    uint64_t GetRuntimeDynamicCellMatchCapacity() const;
    void SetCtrlFlowCacheReplay(bool replay)
    {
        ctrlFlowCacheReplay_ = replay;
        if (!replay) {
            activeHostCtrlFlowCache_ = nullptr;
        }
    }
    bool IsCtrlFlowCacheReplay() const { return ctrlFlowCacheReplay_; }
    DevControlFlowCache* GetActiveHostCtrlFlowCache() const { return activeHostCtrlFlowCache_; }
    /* Emulation path: copy out host mirror by shape hash; D2H attach when missing. */
    DevControlFlowCache* FindHostCtrlFlowCache(const std::vector<DeviceTensorData>& tensors,
                                               CtrlFlowCacheBlob& hostCacheOut);
    void AttachHostCtrlFlowCache(const std::vector<DeviceTensorData>& tensors, CtrlFlowCacheBlob&& hostCache);

    // Eager and aclgraph cannot share one record/wait pair. epoch=0 is eager;
    // a non-zero epoch is the capture model. Switching epoch restarts sequence
    // and (on capture) allocates a fresh graph event pool.
    // Hot path after warm-up: epoch unchanged + events already created + one ++sequence.
    bool BeginRingLaunch(uintptr_t epoch, bool isCapture, int64_t& outSequence);
    AclRtEvent RingEvent(bool isCapture, int64_t idx) const
    {
        return isCapture ? graphEvents_[idx] : eagerEvents_[idx];
    }

    static constexpr int64_t kRingPingPongCount = 2;

    void SetSyncMode(uint8_t syncModel);
    uint8_t GetSyncMode();

    void ResetRuntimeDynamicCellMatchPool(bool useHostMirror) const;

    void PatchHostDynamicCellMatchAddr(DevAscendProgram* hostProg);

    AicpuLaunchDesc& GetAicpuLaunchDesc() { return aicpuLaunchDesc_; }
    RtArgsEx& GetRtAicoreArgs() { return rtAicoreArgs_; }
    RtTaskCfgInfo& GetRtTaskCfg() { return rtTaskCfg_; }

private:
    void InitCachedArgs();
    void InitLaunchArgs();
    void RefreshRuntimeDynamicCellMatchMeta(uint64_t needBytes);

    void BeginRingEpoch(uintptr_t epoch, bool isCapture);
    bool EnsureRingEventsCreated(bool isCapture);

    static void InitDeviceArgs();
    static int InitDeviceArgs(DeviceArgs& args);
    static int InitDeviceArgsCore(DeviceArgs& args);
    static void GetAicoreRegs(const ArchInfo archInfo, std::vector<int64_t>& regs, std::vector<int64_t>& regsPmu);
    static void InitAiCpuSoBin(DeviceArgs& devArgs);

    std::shared_ptr<Function> dynFunc;
    std::vector<std::shared_ptr<Function>> pinnedGraph_;
    DyndevFunctionAttribute* dynAttr{nullptr};
    DevAscendProgram* devProg{nullptr};
    void* kernelBin{nullptr};
    int64_t workspaceSize{0}; // static workspace size
    std::vector<ControlFlowCache> inferShapeCaches;
    std::vector<ControlFlowCache> originShapeCaches;
    /* Single value-depend slot: data+shape hash, device cache, and host mirror. */
    struct ValueDependCtrlFlowCache {
        uint64_t hash{0};
        uint8_t* devCache{nullptr};
        CtrlFlowCacheBlob hostCache;
    };

    /* Bound when Find/Build hits; Materialize reads this without re-hashing. */
    DevControlFlowCache* activeHostCtrlFlowCache_{nullptr};
    ValueDependCtrlFlowCache valueDependCache_;
    bool ctrlFlowCacheReplay_{false};
    int64_t ringSequence_{0};
    uintptr_t ringEpoch_{0};
    bool ringEpochInited_{false};
    AclRtEvent eagerEvents_[kRingPingPongCount]{};
    AclRtEvent graphEvents_[kRingPingPongCount]{};
    bool eagerEventsCreated_{false};
    bool graphEventsCreated_{false};

    std::vector<int64_t> aicpuArgBuf;
    uint64_t l2Offset{0};
    std::vector<DeviceTensorData> argTypes;
    std::vector<DevDynamicCellMatchStridePatch> dynamicCellMatchDescPatches_;
    uint64_t lastPreparedDynamicCellMatchBytes_{0};
    uint64_t runtimeDynamicCellMatchAddr_{0};
    uint64_t runtimeDynamicCellMatchHostAddr_{0};
    uint64_t runtimeDynamicCellMatchCapacity_{0};
    bool runtimeDynamicCellMatchOwned_{false};
    bool runtimeDynamicCellMatchHostOwned_{false};
    std::string kernelName_;
    ToSubMachineConfig toSubMachineConfig_;
    uint8_t scheSyncModel_{0};

    AicpuLaunchDesc aicpuLaunchDesc_;
    RtArgsEx rtAicoreArgs_;
    RtTaskCfgInfo rtTaskCfg_;
    /* Contiguous AICore launch buffer: 7 void* slots + optional IO arena. */
    std::vector<uint8_t> aicoreArgBuf_;
    RtHostInputInfo hostInfo_{};
    RtHostInputInfo aicoreHostInfo_{};
    static std::vector<size_t> emptyIndices_;
};

} // namespace npu::tile_fwk::dynamic
