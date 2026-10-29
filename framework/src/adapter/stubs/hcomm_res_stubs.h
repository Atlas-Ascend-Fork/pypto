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
 * \file hcomm_res_stubs.h
 * \brief
 */

#pragma once

#include "adapter/api/hcomm_res_define.h"

namespace npu::tile_fwk {
HcommResult StubEndpointCreate(const EndpointDesc* endpoint, EndpointHandle* endpointHandle);

HcommResult StubEndpointDestroy(EndpointHandle endpointHandle);

HcommResult StubEndpointGetListenPort(EndpointHandle endpointHandle, uint32_t* port);

HcommResult StubMemReg(EndpointHandle endpointHandle, const char* memTag, const CommMem* mem,
                       HcommMemHandle* memHandle);

HcommResult StubMemUnreg(EndpointHandle endpointHandle, HcommMemHandle memHandle);

HcommResult StubChannelConfigCreate(HcommChannelConfig* config);

HcommResult StubChannelConfigDestroy(HcommChannelConfig config);

HcommResult StubChannelConfigSetInt(HcommChannelConfig config, HcommChannelConfigType type, uint32_t value);

HcommResult StubChannelCreateWithConfig(EndpointHandle endpointHandle, CommEngine engine,
                                        HcommChannelDesc* channelDescs, uint32_t channelNum, HcommChannelConfig config,
                                        ChannelHandle* channels);

HcommResult StubChannelGetStatus(const ChannelHandle* channelList, uint32_t listNum, int32_t* statusList);

HcommResult StubChannelDestroy(const ChannelHandle* channels, uint32_t channelNum);
} // namespace npu::tile_fwk
