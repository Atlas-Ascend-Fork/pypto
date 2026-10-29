/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details.
 * You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>

#include "adapter/api/hcomm_define.h"

#if defined(BUILD_WITH_CANN)
#include "hccl/hccl_channel.h"
#include "hccl/hccl_rank_graph.h"
#else
typedef void* EndpointHandle;
typedef void* HcommMemHandle;
typedef void* HcclMemHandle;
typedef void* HcommChannelConfig;
typedef uint64_t ChannelHandle;

struct EndpointDesc;     // layout in hcomm_res_defs.h
struct CommMem;          // layout in hcomm_res_defs.h
struct HcommChannelDesc; // layout in hcomm_channel.h
struct HcclChannelDesc;  // layout in hccl_channel.h
struct CommLink;         // layout in hccl_rank_graph.h

enum CommEngine {
    COMM_ENGINE_RESERVED = -1, ///< 保留的通信引擎
    COMM_ENGINE_CPU = 0,       ///< HOST CPU引擎
    COMM_ENGINE_CPU_TS = 1,    ///< HOST CPU TS引擎
    COMM_ENGINE_AICPU = 2,     ///< AICPU引擎
    COMM_ENGINE_AICPU_TS = 3,  ///< AICPU TS引擎
    COMM_ENGINE_AIV = 4,       ///< AIV引擎
    COMM_ENGINE_CCU = 5        ///< CCU引擎
};

enum HcommChannelConfigType {
    HCOMM_CHANNEL_CONFIG_TYPE_INVALID = -1,       ///< 无效值
    HCOMM_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE = 0 ///< 多channel共享一个Jetty(仅UB语义协议)
};
#endif
