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
 * \file aikernel_launch.h
 * \brief
 */

#ifndef AIKERNEL_LAUNCH_H
#define AIKERNEL_LAUNCH_H

#include <cstdint>
#include "aikernel_data.h"

const uint64_t SHARED_BUFFER_SIZE = 512;

// use ring buffer to control parallel multi devtask
struct ParallelDevTask {
    uint32_t front{0};
    uint32_t rear{0};
    uint32_t version;
    uint32_t reserver;
    uint32_t idElements[npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM]; // device task id
    int64_t ptrElements[npu::tile_fwk::SCH_DEVTASK_MAX_PARALLELISM]; // device task ptr
};

struct TaskEntry {
    int32_t subGraphId;
    int32_t taskId;
    int64_t funcAddr;
    int64_t tensorAddrs;
    int64_t gmStackSize;
    int64_t gmStackBase;
    int64_t reserved2[2];
    uint32_t tensorSize;
    uint32_t reserved[1];
};

struct KernelArgs {
    int64_t shakeBuffer[8];
    int64_t shakeBufferCpuToCore[8];
    int64_t waveBufferCpuToCore[8];
    int64_t dfxBuffer[8];
    struct ParallelDevTask parallelDevTask;
    TaskEntry taskEntry;
};
static_assert(sizeof(KernelArgs) < SHARED_BUFFER_SIZE);

union KernelSharedBuffer {
    struct KernelArgs args;
    uint8_t sharedBuffer[SHARED_BUFFER_SIZE];
    KernelSharedBuffer() {};
};

#endif
