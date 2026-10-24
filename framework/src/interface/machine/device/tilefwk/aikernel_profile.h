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
 * \file aikernel_profile.h
 * \brief
 */

#ifndef AIKERNEL_PROFILE_H
#define AIKERNEL_PROFILE_H

#include <cstdint>
#include "aikernel_profile_aicpu.h"

enum FedProfKind {
    FED_PROF_PER_CORE_PROF,
    FED_PROF_AICPU_PROF,
    FED_PROF_KIND_MAX,
};

constexpr const char* fed_prof_name_list[FED_PROF_KIND_MAX] = {"per_core", "aicpu"};

struct FedProfDataBase {};

template <typename DerivedType, typename UnderlyingType>
class BitFlag {
public:
    using underlying_type = UnderlyingType;
    underlying_type value{0};
    constexpr BitFlag(underlying_type v = 0) : value(v) {}
    constexpr bool Empty() const { return value == 0; }
    constexpr bool Contains(underlying_type mask) const { return (value & mask) == mask; }
    constexpr bool Overlaps(underlying_type mask) const { return (value & mask) != 0; }
    constexpr void Add(underlying_type mask) { value |= mask; }
    constexpr void Remove(underlying_type mask) { value &= ~mask; }
    friend constexpr DerivedType operator|(DerivedType lhs, DerivedType rhs)
    {
        return DerivedType(lhs.value | rhs.value);
    }
    friend constexpr DerivedType operator&(DerivedType lhs, DerivedType rhs)
    {
        return DerivedType(lhs.value & rhs.value);
    }
    friend constexpr DerivedType operator^(DerivedType lhs, DerivedType rhs)
    {
        return DerivedType(lhs.value ^ rhs.value);
    }
    friend constexpr DerivedType operator~(DerivedType lhs) { return DerivedType(~lhs.value); }
    constexpr operator underlying_type() const { return value; }
};

struct FedProfFlag : public BitFlag<FedProfFlag, uint32_t> {
    using BitFlag::BitFlag;
    enum : underlying_type {
        OFF = 0x0,
        AICPU_FUNC = 0x1 << 0,
        AICORE_TIME = 0x1 << 1,
        AICORE_PMU = 0x1 << 2,
    };
};

struct FedProfLeafTaskProf {
    int16_t devTaskIdx;
    int16_t leafFuncIdx;
    int32_t taskId;
    int64_t timestampStart;
    int64_t timestampEnd;
    uint64_t perfDataBaseAddr;
    uint64_t setEventAddr;
    uint64_t waitEventAddr;
    int16_t setEventNum;
    int16_t waitEventNum;
};

enum FedProfPerfTraceKind {
    PERF_TRACE_CORE_BEGIN = 0,
    PERF_TRACE_CORE_INIT,
    PERF_TRACE_CORE_DEV_TASK_RCV_MODEL,
    PERF_TRACE_CORE_DEV_TASK_WAIT_RCV_FIRST_LEAF_TASK,
    PERF_TRACE_CORE_DEV_TASK_LEAF_TASK_EXEC,
    PERF_TRACE_CORE_DEV_TASK_WAIT_SYNC_STOP_NOTIFY,
    PERF_TRACE_CORE_WAIT_ALL_DEV_TASK_LEAF_TASK_EXEC_FINISH,
    PERF_TRACE_CORE_WAIT_EXIT_NOTIFY,
    PERF_TRACE_CORE_KIND_MAX
};

constexpr uint32_t PERF_TRACE_CORE_PER_KIND_MAX_COUNT = 200;

inline const char* FedProfPerfTraceKindName[] = {"BEGIN",
                                                 "INIT",
                                                 "DEV_TASK_RCV_MODEL",
                                                 "DEV_TASK_WAIT_RCV_FIRST_LEAF_TASK",
                                                 "DEV_TASK_ALL_LEAF_TASK_EXEC",
                                                 "DEV_TASK_WAIT_SYNC_STOP_NOTIFY",
                                                 "WAIT_ALL_DEV_TASK_LEAF_TASK_EXEC_FINISH",
                                                 "WAIT_EXIT_NOTIFY"};

struct FedProfDevTaskCoreEvent {
    uint64_t type = 0;
    uint64_t devTaskIdx = 0;
    uint64_t timestamp = 0;
    FedProfDevTaskCoreEvent() = default;
    FedProfDevTaskCoreEvent(const FedProfDevTaskCoreEvent&) = default;
    FedProfDevTaskCoreEvent& operator=(const FedProfDevTaskCoreEvent&) = default;
};

constexpr const uint32_t FED_PROF_DEV_TASK_MAX_COUNT = 200;
constexpr const uint32_t FED_PROF_LEAF_TASK_MAX_COUNT = 10000;

struct FedProfDevTaskProf {
    uint64_t cnt = 0;
    FedProfDevTaskCoreEvent aicoreEveryDevTypeTimeStamp[PERF_TRACE_CORE_KIND_MAX * PERF_TRACE_CORE_PER_KIND_MAX_COUNT];
    FedProfDevTaskProf() = default;
    FedProfDevTaskProf(const FedProfDevTaskProf&) = default;
    FedProfDevTaskProf& operator=(const FedProfDevTaskProf&) = default;
};

struct FedProfPerCoreStatus {
    int16_t isProfFinish;
    int16_t scheCpuIdx;
    int32_t coreType = -1;
    int64_t leafTaskCount;
    int64_t currLaunchRound;
};
constexpr uint32_t FED_PROF_PER_CORE_STATUS_SIZE = sizeof(FedProfPerCoreStatus);

struct FedProfPerCoreProf : public FedProfDataBase {
    FedProfPerCoreStatus status;
    FedProfDevTaskProf aicoreDevTaskInfo[FED_PROF_DEV_TASK_MAX_COUNT];
    FedProfLeafTaskProf leafTaskProfList[];
};

struct FedProfLaunchDevTaskProf {
    uint64_t cnt{0};
    uint64_t timeStamp[npu::tile_fwk::dynamic::PERF_TRACE_COUNT_DEVTASK_MAX_NUM] = {0};
};

struct FedProfAicpuProf : public FedProfDataBase {
    uint64_t perfAicpuTrace[npu::tile_fwk::dynamic::MAX_USED_AICPU_NUM][npu::tile_fwk::dynamic::PERF_TRACE_MAX] = {{0}};
    FedProfLaunchDevTaskProf devTaskPerfs[npu::tile_fwk::dynamic::MAX_USED_AICPU_NUM]
                                         [npu::tile_fwk::dynamic::DEVTASK_PERF_TYPE_NUM];
};

constexpr uint32_t FED_PROF_LEAF_TASK_MIX_SYNC_EVENT_AVG_COUNT = 16;
constexpr uint32_t FED_PROF_DEV_TASK_MIX_SYNC_EVENT_COUNT = FED_PROF_LEAF_TASK_MAX_COUNT *
                                                            FED_PROF_LEAF_TASK_MIX_SYNC_EVENT_AVG_COUNT *
                                                            sizeof(uint64_t);
constexpr uint32_t FED_PROF_PER_CORE_MIX_SYNC_EVENT_PROF_SIZE = FED_PROF_DEV_TASK_MIX_SYNC_EVENT_COUNT * 2;
constexpr uint32_t FED_PROF_PER_CORE_TASK_PROF_SIZE = sizeof(FedProfPerCoreProf) +
                                                      FED_PROF_LEAF_TASK_MAX_COUNT * sizeof(FedProfLeafTaskProf);
constexpr uint32_t FED_PROF_PER_CORE_PROF_SIZE = FED_PROF_PER_CORE_TASK_PROF_SIZE +
                                                 FED_PROF_PER_CORE_MIX_SYNC_EVENT_PROF_SIZE;
constexpr uint32_t FED_PROF_PER_CORE_PROF_ALIGN = 512;
constexpr uint32_t FED_PROF_PER_CORE_PROF_ALIGNED_SIZE = (FED_PROF_PER_CORE_PROF_SIZE + FED_PROF_PER_CORE_PROF_ALIGN -
                                                          1) /
                                                         FED_PROF_PER_CORE_PROF_ALIGN * FED_PROF_PER_CORE_PROF_ALIGN;

#endif
