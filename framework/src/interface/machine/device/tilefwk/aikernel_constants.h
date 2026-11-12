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
 * \file aikernel_constants.h
 */

#pragma once

#include <cstdint>

namespace npu::tile_fwk {
enum class ArchInfo : uint32_t {
    DAV_1001 = 1001,
    DAV_2201 = 2201,
    DAV_3510 = 3510,
    DAV_3003 = 3003,
    DAV_3113 = 3113,
    DAV_5101 = 5101,
    DAV_9201 = 9201,
    DAV_UNKNOWN
};

constexpr bool IsA5A6Architecture(ArchInfo arch) { return arch == ArchInfo::DAV_3510 || arch == ArchInfo::DAV_9201; }
} // namespace npu::tile_fwk

namespace npu::tile_fwk::dynamic {
#ifndef PAGE_SIZE
#define PAGE_SIZE 4096
#endif

constexpr uint32_t MAX_AICORE_NUM = 140;
constexpr uint32_t NAX_AIV_TOTAL_NUM = 72;
constexpr uint32_t MAX_MANAGER_AIV_NUM = 72;
constexpr uint32_t MAX_RESOLVE_TASK_NUM = MAX_MANAGER_AIV_NUM * 2;
const uint8_t INVALID_COREIDX_POSITION = 0xff;

const uint32_t REG_SPR_FAST_PATH_ENABLE = 0x18;
const uint64_t REG_SPR_FAST_PATH_OPEN = 0xE;
const uint64_t REG_SPR_FAST_PATH_CLOSE = 0xF;

constexpr uint32_t NUM_ONE = 1;
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t NUM_THREE = 3;
constexpr uint32_t NUM_FOUR = 4;
constexpr uint32_t NUM_FIVE = 5;
constexpr uint32_t NUM_THIRTY_TWO = 32;

constexpr int32_t AICORE_COREID_MASK = 0x0FFF;
constexpr int32_t AICORE_BLOCKID_MASK = 0x0FFF;

struct ArchCfg {
    uint32_t maxCore{0};
    uint32_t subCorePerAiCore{0};
    uint32_t aivNumPerAiCore{0};
    uint32_t readyRegOffset{0};
    uint32_t finishRegOffset{0};
    int32_t maxAicoreNum{0};
    int32_t maxAicpuNum{0};
    bool needWriteRegForFastPath{false};
    uint32_t aicorePerDie{0};
    uint32_t aivAicRatio{0};
};

template <ArchInfo ARCH>
struct ArchTraits;

template <>
struct ArchTraits<ArchInfo::DAV_2201> {
    static constexpr ArchCfg kCfg{25, 3, 2, 0xA0, 0x4C8, 75, 5, true, 0, 2};
};

template <>
struct ArchTraits<ArchInfo::DAV_3510> {
    static constexpr ArchCfg kCfg{36, 3, 2, 0xD0, 0x5108, 108, 7, false, 18, 2};
};

template <>
struct ArchTraits<ArchInfo::DAV_9201> {
    static constexpr ArchCfg kCfg{70, 2, 1, 0x5500, 0x5520, 140, 11, false, 35, 1};
};

template <>
struct ArchTraits<ArchInfo::DAV_UNKNOWN> {
    static constexpr ArchCfg kCfg{0, 0, 0, 0, 0, 0, 5, false, 0, 2};
};

template <ArchInfo ARCH>
constexpr const ArchCfg& GetArchCfg()
{
    return ArchTraits<ARCH>::kCfg;
}

constexpr const ArchCfg& GetArchCfg(ArchInfo arch)
{
    switch (arch) {
        case ArchInfo::DAV_2201:
            return GetArchCfg<ArchInfo::DAV_2201>();
        case ArchInfo::DAV_3510:
            return GetArchCfg<ArchInfo::DAV_3510>();
        case ArchInfo::DAV_9201:
            return GetArchCfg<ArchInfo::DAV_9201>();
        default:
            return GetArchCfg<ArchInfo::DAV_UNKNOWN>();
    }
}

constexpr uint32_t GetMaxCore(ArchInfo arch) { return GetArchCfg(arch).maxCore; }

constexpr uint32_t GetSubCorePerAiCore(ArchInfo arch) { return GetArchCfg(arch).subCorePerAiCore; }

constexpr uint32_t GetAicorePerDie(ArchInfo arch) { return GetArchCfg(arch).aicorePerDie; }

constexpr uint32_t GetAivNumPerCore(ArchInfo arch) { return GetArchCfg(arch).aivNumPerAiCore; }

constexpr uint32_t GetMaxAicoreNum(ArchInfo arch) { return static_cast<uint32_t>(GetArchCfg(arch).maxAicoreNum); }

constexpr uint32_t GetReadyRegOffset(ArchInfo arch) { return GetArchCfg(arch).readyRegOffset; }

constexpr uint32_t GetFinishRegOffset(ArchInfo arch) { return GetArchCfg(arch).finishRegOffset; }

constexpr bool GetIsNeedWriteRegForFastPath(ArchInfo arch) { return GetArchCfg(arch).needWriteRegForFastPath; }

constexpr uint32_t GetMaxAicpuNum(ArchInfo arch) { return static_cast<uint32_t>(GetArchCfg(arch).maxAicpuNum); }

constexpr uint32_t GetAivAicRatio(ArchInfo arch) { return GetArchCfg(arch).aivAicRatio; }
} // namespace npu::tile_fwk::dynamic
