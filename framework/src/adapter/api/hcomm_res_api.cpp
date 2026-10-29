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
 * \file hcomm_res_api.cpp
 * \brief
 */

#include "adapter/api/hcomm_res_api.h"

#if defined(BUILD_WITH_CANN)
#include "adapter/manager/adapter_manager.h"
#include "hcomm_channel.h"
#endif
#include "adapter/stubs/hcomm_res_stubs.h"

namespace npu::tile_fwk {
HcommResult HcommEndpointCreate(const EndpointDesc* endpoint, EndpointHandle* endpointHandle)
{
#if defined(BUILD_WITH_CANN)
    void* func = AdapterManager::Instance().GetHcommResAdapter().GetFunction(HcommResFunc::EndpointCreate);
    if (func != nullptr) {
        int32_t (*hcommFunc)(
            const ::EndpointDesc*,
            ::EndpointHandle*) = reinterpret_cast<int32_t (*)(const ::EndpointDesc*, ::EndpointHandle*)>(func);
        return static_cast<HcommResult>(hcommFunc(reinterpret_cast<const ::EndpointDesc*>(endpoint), endpointHandle));
    }
#endif
    return StubEndpointCreate(endpoint, endpointHandle);
}

HcommResult HcommEndpointDestroy(EndpointHandle endpointHandle)
{
#if defined(BUILD_WITH_CANN)
    void* func = AdapterManager::Instance().GetHcommResAdapter().GetFunction(HcommResFunc::EndpointDestroy);
    if (func != nullptr) {
        int32_t (*hcommFunc)(::EndpointHandle) = reinterpret_cast<int32_t (*)(::EndpointHandle)>(func);
        return static_cast<HcommResult>(hcommFunc(endpointHandle));
    }
#endif
    return StubEndpointDestroy(endpointHandle);
}

HcommResult HcommEndpointGetListenPort(EndpointHandle endpointHandle, uint32_t* port)
{
#if defined(BUILD_WITH_CANN)
    void* func = AdapterManager::Instance().GetHcommResAdapter().GetFunction(HcommResFunc::EndpointGetListenPort);
    if (func != nullptr) {
        int32_t (*hcommFunc)(::EndpointHandle,
                             uint32_t*) = reinterpret_cast<int32_t (*)(::EndpointHandle, uint32_t*)>(func);
        return static_cast<HcommResult>(hcommFunc(endpointHandle, port));
    }
#endif
    return StubEndpointGetListenPort(endpointHandle, port);
}

HcommResult HcommMemReg(EndpointHandle endpointHandle, const char* memTag, const CommMem* mem,
                        HcommMemHandle* memHandle)
{
#if defined(BUILD_WITH_CANN)
    void* func = AdapterManager::Instance().GetHcommResAdapter().GetFunction(HcommResFunc::ResMemReg);
    if (func != nullptr) {
        int32_t (*hcommFunc)(::EndpointHandle, const char*, const ::CommMem*, ::HcommMemHandle*) = reinterpret_cast<
            int32_t (*)(::EndpointHandle, const char*, const ::CommMem*, ::HcommMemHandle*)>(func);
        return static_cast<HcommResult>(
            hcommFunc(endpointHandle, memTag, reinterpret_cast<const ::CommMem*>(mem), memHandle));
    }
#endif
    return StubMemReg(endpointHandle, memTag, mem, memHandle);
}

HcommResult HcommMemUnreg(EndpointHandle endpointHandle, HcommMemHandle memHandle)
{
#if defined(BUILD_WITH_CANN)
    void* func = AdapterManager::Instance().GetHcommResAdapter().GetFunction(HcommResFunc::ResMemUnreg);
    if (func != nullptr) {
        int32_t (*hcommFunc)(
            ::EndpointHandle,
            ::HcommMemHandle) = reinterpret_cast<int32_t (*)(::EndpointHandle, ::HcommMemHandle)>(func);
        return static_cast<HcommResult>(hcommFunc(endpointHandle, memHandle));
    }
#endif
    return StubMemUnreg(endpointHandle, memHandle);
}

HcommResult HcommChannelConfigCreate(HcommChannelConfig* config)
{
#if defined(BUILD_WITH_CANN)
    void* func = AdapterManager::Instance().GetHcommResAdapter().GetFunction(HcommResFunc::ChannelConfigCreate);
    if (func != nullptr) {
        int32_t (*hcommFunc)(::HcommChannelConfig*) = reinterpret_cast<int32_t (*)(::HcommChannelConfig*)>(func);
        return static_cast<HcommResult>(hcommFunc(config));
    }
#endif
    return StubChannelConfigCreate(config);
}

HcommResult HcommChannelConfigDestroy(HcommChannelConfig config)
{
#if defined(BUILD_WITH_CANN)
    void* func = AdapterManager::Instance().GetHcommResAdapter().GetFunction(HcommResFunc::ChannelConfigDestroy);
    if (func != nullptr) {
        int32_t (*hcommFunc)(::HcommChannelConfig) = reinterpret_cast<int32_t (*)(::HcommChannelConfig)>(func);
        return static_cast<HcommResult>(hcommFunc(config));
    }
#endif
    return StubChannelConfigDestroy(config);
}

HcommResult HcommChannelConfigSetInt(HcommChannelConfig config, HcommChannelConfigType type, uint32_t value)
{
#if defined(BUILD_WITH_CANN)
    void* func = AdapterManager::Instance().GetHcommResAdapter().GetFunction(HcommResFunc::ChannelConfigSetInt);
    if (func != nullptr) {
        int32_t (*hcommFunc)(
            ::HcommChannelConfig, ::HcommChannelConfigType,
            uint32_t) = reinterpret_cast<int32_t (*)(::HcommChannelConfig, ::HcommChannelConfigType, uint32_t)>(func);
        return static_cast<HcommResult>(hcommFunc(config, static_cast<::HcommChannelConfigType>(type), value));
    }
#endif
    return StubChannelConfigSetInt(config, type, value);
}

HcommResult HcommChannelCreateWithConfig(EndpointHandle endpointHandle, CommEngine engine,
                                         HcommChannelDesc* channelDescs, uint32_t channelNum, HcommChannelConfig config,
                                         ChannelHandle* channels)
{
#if defined(BUILD_WITH_CANN)
    void* func = AdapterManager::Instance().GetHcommResAdapter().GetFunction(HcommResFunc::ChannelCreateWithConfig);
    if (func != nullptr) {
        int32_t (*hcommFunc)(
            ::EndpointHandle, ::CommEngine, ::HcommChannelDesc*, uint32_t, ::HcommChannelConfig,
            ::ChannelHandle*) = reinterpret_cast<int32_t (*)(::EndpointHandle, ::CommEngine, ::HcommChannelDesc*,
                                                             uint32_t, ::HcommChannelConfig, ::ChannelHandle*)>(func);
        return static_cast<HcommResult>(hcommFunc(endpointHandle, static_cast<::CommEngine>(engine),
                                                  reinterpret_cast<::HcommChannelDesc*>(channelDescs), channelNum,
                                                  config, channels));
    }
#endif
    return StubChannelCreateWithConfig(endpointHandle, engine, channelDescs, channelNum, config, channels);
}

HcommResult HcommChannelGetStatus(const ChannelHandle* channelList, uint32_t listNum, int32_t* statusList)
{
#if defined(BUILD_WITH_CANN)
    void* func = AdapterManager::Instance().GetHcommResAdapter().GetFunction(HcommResFunc::ChannelGetStatus);
    if (func != nullptr) {
        int32_t (*hcommFunc)(
            const ::ChannelHandle*, uint32_t,
            int32_t*) = reinterpret_cast<int32_t (*)(const ::ChannelHandle*, uint32_t, int32_t*)>(func);
        return static_cast<HcommResult>(hcommFunc(channelList, listNum, statusList));
    }
#endif
    return StubChannelGetStatus(channelList, listNum, statusList);
}

HcommResult HcommChannelDestroy(const ChannelHandle* channels, uint32_t channelNum)
{
#if defined(BUILD_WITH_CANN)
    void* func = AdapterManager::Instance().GetHcommResAdapter().GetFunction(HcommResFunc::ChannelDestroy);
    if (func != nullptr) {
        int32_t (*hcommFunc)(const ::ChannelHandle*,
                             uint32_t) = reinterpret_cast<int32_t (*)(const ::ChannelHandle*, uint32_t)>(func);
        return static_cast<HcommResult>(hcommFunc(channels, channelNum));
    }
#endif
    return StubChannelDestroy(channels, channelNum);
}
} // namespace npu::tile_fwk
