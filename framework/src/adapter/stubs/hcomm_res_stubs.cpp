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

/*!
 * \file hcomm_res_stubs.cpp
 * \brief
 */

#include "adapter/stubs/hcomm_res_stubs.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk {
HcommResult StubEndpointCreate(const EndpointDesc* endpoint, EndpointHandle* endpointHandle)
{
    ADAPTER_LOGD("Enter stub function of EndpointCreate.");
    (void)endpoint;
    (void)endpointHandle;
    return HCOMM_SUCCESS;
}

HcommResult StubEndpointDestroy(EndpointHandle endpointHandle)
{
    ADAPTER_LOGD("Enter stub function of EndpointDestroy.");
    (void)endpointHandle;
    return HCOMM_SUCCESS;
}

HcommResult StubEndpointGetListenPort(EndpointHandle endpointHandle, uint32_t* port)
{
    ADAPTER_LOGD("Enter stub function of EndpointGetListenPort.");
    (void)endpointHandle;
    (void)port;
    return HCOMM_SUCCESS;
}

HcommResult StubMemReg(EndpointHandle endpointHandle, const char* memTag, const CommMem* mem, HcommMemHandle* memHandle)
{
    ADAPTER_LOGD("Enter stub function of MemReg.");
    (void)endpointHandle;
    (void)memTag;
    (void)mem;
    (void)memHandle;
    return HCOMM_SUCCESS;
}

HcommResult StubMemUnreg(EndpointHandle endpointHandle, HcommMemHandle memHandle)
{
    ADAPTER_LOGD("Enter stub function of MemUnreg.");
    (void)endpointHandle;
    (void)memHandle;
    return HCOMM_SUCCESS;
}

HcommResult StubChannelConfigCreate(HcommChannelConfig* config)
{
    ADAPTER_LOGD("Enter stub function of ChannelConfigCreate.");
    (void)config;
    return HCOMM_SUCCESS;
}

HcommResult StubChannelConfigDestroy(HcommChannelConfig config)
{
    ADAPTER_LOGD("Enter stub function of ChannelConfigDestroy.");
    (void)config;
    return HCOMM_SUCCESS;
}

HcommResult StubChannelConfigSetInt(HcommChannelConfig config, HcommChannelConfigType type, uint32_t value)
{
    ADAPTER_LOGD("Enter stub function of ChannelConfigSetInt.");
    (void)config;
    (void)type;
    (void)value;
    return HCOMM_SUCCESS;
}

HcommResult StubChannelCreateWithConfig(EndpointHandle endpointHandle, CommEngine engine,
                                        HcommChannelDesc* channelDescs, uint32_t channelNum, HcommChannelConfig config,
                                        ChannelHandle* channels)
{
    ADAPTER_LOGD("Enter stub function of ChannelCreateWithConfig.");
    (void)endpointHandle;
    (void)engine;
    (void)channelDescs;
    (void)channelNum;
    (void)config;
    (void)channels;
    return HCOMM_SUCCESS;
}

HcommResult StubChannelGetStatus(const ChannelHandle* channelList, uint32_t listNum, int32_t* statusList)
{
    ADAPTER_LOGD("Enter stub function of ChannelGetStatus.");
    (void)channelList;
    (void)listNum;
    (void)statusList;
    return HCOMM_SUCCESS;
}

HcommResult StubChannelDestroy(const ChannelHandle* channels, uint32_t channelNum)
{
    ADAPTER_LOGD("Enter stub function of ChannelDestroy.");
    (void)channels;
    (void)channelNum;
    return HCOMM_SUCCESS;
}
} // namespace npu::tile_fwk
