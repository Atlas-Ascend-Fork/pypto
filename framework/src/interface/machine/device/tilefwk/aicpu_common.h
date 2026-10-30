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
 * \file aicpu_common.h
 * \brief
 */

#ifndef RUNTIME_COMMON_DEF_H
#define RUNTIME_COMMON_DEF_H

#include <cstddef>
#include <cstdint>
#include "aikernel_profile_aicpu.h"
#include "aikernel_data.h"
#include "aikernel_profile.h"
#include "aikernel_launch.h"

using ArchInfo = npu::tile_fwk::ArchInfo;

const uint64_t AICORE_TASK_INIT = 0xFFFFFFFF;
const uint64_t AICORE_TASK_STOP = 0xF0000001;
const uint64_t AICORE_FUNC_STOP = 0xF0000002;
const uint64_t AICORE_TASK_ABNORMAL_STOP = 0xF0000004;
const uint64_t AICORE_TASK_FETCH_CONFLICT = 0xF0000008;
const uint64_t AICORE_TASK_NO_INCOME = 0xF0000009;
const uint64_t AICORE_TASK_ALL_FINISH = 0xF000000a;
const uint64_t AICORE_FIN_MASK = 0x80000000;
const uint64_t AICORE_TASK_MAX = 0x70000000;

const uint64_t AICORE_SAY_HELLO = 0x80000000;
const uint64_t AICORE_SAY_ACK = 0x80000000;
const uint64_t AICORE_SAY_GOODBYE = 0x88888888;
const int64_t PRO_LEVEL1 = 2;
const int64_t PRO_LEVEL2 = 3;

constexpr int REG_LOW_TASK_PING = 0;
constexpr int REG_LOW_TASK_PONG = 1;

// AICPU dispatch/sync 阶段超时时间
#define AICPU_DISPATCH_TIMEOUT_A2A3 50000000ULL * 10
#define AICPU_DISPATCH_TIMEOUT_A5 1000000000ULL * 10

constexpr int SHAK_BUF_PRINT_BUFFER_INDEX = 5;

constexpr int CPU_TO_CORE_SHAK_BUF_COREFUNC_DATA_INDEX = 0;
constexpr int CPU_TO_CORE_SHAK_BUF_GOODBYE_INDEX = 0;

constexpr int FUNC_ID_BATCH = 0x7FF;

const uint64_t PMU_BUFFER_SIZE = 4096;
const uint64_t DEVICE_QUEUE_SIZE = 512;
const uint64_t PRINT_BUFFER_SIZE = 131072; // 128KB

constexpr const int DEV_SHAPE_DIM_NUM_2 = 2;
constexpr const int DEV_SHAPE_DIM_NUM_3 = 3;
constexpr const int DEV_SHAPE_DIM_NUM_4 = 4;
constexpr const int DEV_SHAPE_DIM_NUM_5 = 5;

#define DEVICE_TASK_STOP 0x7FFFFFFE

#define DEVICE_TASK_TYPE_STATIC 0
#define DEVICE_TASK_TYPE_DYN 1
#define DEVICE_TASK_TYPE_INVALID 0xf

struct ToSubMachineConfig {
    FedProfFlag profConfig{FedProfFlag::OFF};
};

enum DeviceKernelRunMode : uint32_t {
    RUN_INVALID = 0,
    RUN_SPLITTED_STREAM_CTRL = 1,
    RUN_SPLITTED_STREAM_SCHE = 2,
};

struct DeviceKernelArgsParameter {
    uint32_t runMode{RUN_INVALID};
    uint32_t p1;
    uint64_t globalRound{0};
    uint64_t ctrlBlockNum{0};
};
static_assert(sizeof(DeviceKernelArgsParameter) == sizeof(uint64_t) * 0x3, "Invalid parameter size");

struct DeviceRuntimeOffset {
    uint64_t startArgsOffset{0};
    uint64_t taskCtrlPoolOffset{0};
    uint64_t taskQueueOffset{0};
    uint64_t deviceTaskReadyQueueOffset{0};
    uint64_t generalOffset{0};
    uint64_t stitchPoolOffset{0};
    uint64_t duppedStitchPoolOffset{0};
    uint64_t size{0};
    uint64_t count{0};
};

struct DeviceArgs {
    uint32_t nrAic{0};
    uint32_t nrAiv{0};
    uint32_t nrAicpu{0};
    uint32_t nrValidAic{0};
    uint64_t opaque{0};         // store device global data, must be init with zero
    uint64_t devQueueAddr;      // pcie/XLink mem, used between host and device, `DEVICE_QUEUE_SIZE`
    uint64_t sharedBuffer;      // SHARED_BUFFER_SIZE per core, aics first
    uint64_t coreRegAddr;       // core reg addr, uint64_t per core, aic first
    uint64_t corePmuRegAddr;    // pmu reg addr, uint64_t per core, aic first
    uint64_t corePmuAddr;       // pmu data addr, PAGE_SIZE per core, aic first
    uint64_t pmuEventAddr;      // pmu event addr
    uint64_t taskType : 4;      // initial task type
    uint64_t machineConfig : 8; // machine config
    uint64_t taskId : 52;       // initial task id
    uint64_t taskData;          // initial task data
    uint64_t taskWastTime{0};
    uint64_t aicpuSoBin{0};                // server so Bin
    uint64_t aicpuSoLen{0};                // server so len
    uint64_t deviceId{0};                  // for device copy fileName
    uint64_t runtimeDataRingBufferAddr{0}; // DevStartArgs addr
    // DRCO devTask 就绪队列在 runtime data 槽内的布局偏移（per-program 常量，随 DeviceArgs下发）：AICore 入口以"当前槽
    // + 偏移"自行推导队列指针。
    uint64_t deviceTaskReadyQueueOffset{0};
    uint64_t dynamicCellMatchAddr{0};              // standalone metadata pool for dynamic cell match
    uint64_t dynamicCellMatchCapacity{0};          // allocated bytes for dynamicCellMatchAddr
    uint32_t hostPid{0};                           // for dump tensor
    uint32_t scheCpuNum{0};                        // sche cpu num calc by host
    uint32_t enableCtrl : 2;                       // if enable builtin ctrl
    uint32_t validGetPgMask : 30;                  // mark pgmask is invalid
    uint64_t fedProfAddrList[FED_PROF_KIND_MAX]{}; // FedProf Gm addr list, indexed by FedProfKind
    uint64_t devDfxArgAddr{0};                     // devDfx
    uint64_t GetBlockNum() const { return nrValidAic * (nrAiv / nrAic + 1); }
    uint32_t die0MaxCpuid{0};
    uint32_t launchSchedAicpuNum{7};    // user configures the number of launch schedules for aicpu
    bool launchSchedSameCluster = true; // user configures whether enables binding aicpu within the same cluster
    bool enableVFFusion = false;
    bool enableEslModel = false;
    bool hasAicpuTask = false;
    bool enableAicoreResolve = false; // runtime switch, driven by env ENABLE_AICORE_RESOLVE=true
    // 并行（DRCO devTask 并行窗口）：launch 级单一事实源——ctrl 挂链状态机与 AICore 入口选路同源读本标志。on：ctrl
    // 链头入队+挂链、设备走窗口；
    bool parallelMode = false;
    ArchInfo archInfo{ArchInfo::DAV_2201};
    ToSubMachineConfig toSubMachineConfig;
};

#define TO_ENTRY_IMPL(name, sep1, key1, sep2, key2, type) (name##sep1##key1##sep2##key2##type)
#define TO_ENTRY(name, key1, key2, type) TO_ENTRY_IMPL(name, _, key1, _, key2, type)

#ifdef __MIX__
#ifdef __AIV__
#define KERNEL_ENTRY(x, y, z) TO_ENTRY(x, y, z, _mix_aiv)
#else
#define KERNEL_ENTRY(x, y, z) TO_ENTRY(x, y, z, _mix_aic)
#endif
#else
#define KERNEL_ENTRY(x, y, z) x
#endif

const uint64_t AICORE_REG_SAY_HELLO = 0xF000000080000000;
constexpr uint32_t REG_HIGH_DTASKID_SHIFT = 32;
enum class TASK_POS : size_t { LOW_REG = 0, HIGH_REG = 1, ALL_REG = 2, REG_POS_BUTT = 3 };
constexpr uint32_t
    MAX_SYNC_EVENT_NUM = 48; // the max set/wait insts in a mix subgraph leaffunction is 48, can set larger manually

struct DevDfxArgs {
    int32_t logLevel{-1};
    int32_t isOpenPerfTrace{0};
    uint32_t deviceId{0};
    uint8_t profLevel{0};
};

constexpr uint32_t INVALID_DEV_TASK_ID = 0xFFFFFFFF;

#endif
