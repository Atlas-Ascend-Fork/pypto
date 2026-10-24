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
 * \file fed_prof_manager.h
 * \brief
 */

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include "machine/runtime/runner/fed_prof_holder.h"

struct DeviceArgs;
struct ToSubMachineConfig;

namespace npu::tile_fwk {

class FedProfManager {
public:
    static FedProfManager& GetInstance();
    FedProfManager();
    ~FedProfManager();
    void InitDeviceArgs(const DeviceArgs& args);
    void AicpuProfDumpThreadStart();
    uint64_t DeviceDataAllocate(FedProfKind kind);
    void PerCoreProfResetData() const;
    void PerCoreProfSync(bool debugEnable);
    int SetDebugEnable(const ToSubMachineConfig& profLevelConfig);

    // PGO (Profiling Guided Optimization) support.
    // PGO 侧只负责"记录"：在算子作用域内同步一次选项（launch 期间调用，可重复调用）。
    //   experimental_pgo_update=true 时，把 experimental_pgo_path 记为导出目标（只有一个字符串）。
    // 不区分同一进程内不同算子，也不对数据做算子维度的切分——PGO 数据的读取与使用由使用方
    // 自行按其作用域里的选项处理。
    void PgoUpdatePath();
    // PGO: 是否已有算子要求导出（experimental_pgo_update=true 且路径非空）。
    bool IsPgoDumpRequested() const { return !pgoPath_.empty(); }
    // PGO: 枚举所有 prof 数据，逐类 DumpJson 落盘到 <pgoPath_>/<kind 名称>.json。
    void PgoDump();

private:
    uint32_t GetCoreCount() const;
    uint64_t PerCoreProfDevAddr() const
    {
        return profList_[FED_PROF_PER_CORE_PROF] != nullptr ? profList_[FED_PROF_PER_CORE_PROF]->GetDevAddr() : 0;
    }
    uint64_t AicpuProfDevAddr() const
    {
        return profList_[FED_PROF_AICPU_PROF] != nullptr ? profList_[FED_PROF_AICPU_PROF]->GetDevAddr() : 0;
    }
    void PerCoreProfReleaseData();
    void AicpuProfReleaseData();
    void PerCoreProfResetMetadata(bool multipleLaunch);

    void PerCoreProfDumpSwimlane();
    void PerCoreProfDumpSwimlaneDraw(const json& perCoreProfObjList);

    void AicpuProfDumpThreadStop();
    void AicpuProfDumpThreadEntry();
    void AicpuProfMaybeDumpSwimlane(bool isLast);
    void AicpuProfDumpSwimlane(bool isLast);
    void AicpuProfDumpSwimlaneDraw(const json& launchObjList, uint32_t currLaunchRound);
    int64_t AicpuProfDumpSwimlaneCheck(bool isLast);

    FedProfDeviceArgs args_;

    std::unique_ptr<FedProfHolderBase> profList_[FED_PROF_KIND_MAX];

    uint32_t lastDumpRound{0};
    bool perCoreProfDataInited_{false};
    std::string pgoPath_;
    std::thread dumpThread_;
    std::atomic<bool> dumpThreadStopFlag_{false};
};
} // namespace npu::tile_fwk
