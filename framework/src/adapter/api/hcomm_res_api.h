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
 * \file hcomm_res_api.h
 * \brief Wrappers of the HCOMM resource library (libhcomm.so): endpoint / jetty / channel
 * management for one-sided communication windows. Signatures use the local mirror types
 * (hcomm_res_define.h), never the vendor headers.
 */

#pragma once

#include "adapter/api/hcomm_res_define.h"

namespace npu::tile_fwk {
HcommResult HcommEndpointCreate(const EndpointDesc* endpoint, EndpointHandle* endpointHandle);

HcommResult HcommEndpointDestroy(EndpointHandle endpointHandle);

HcommResult HcommEndpointGetListenPort(EndpointHandle endpointHandle, uint32_t* port);

HcommResult HcommMemReg(EndpointHandle endpointHandle, const char* memTag, const CommMem* mem,
                        HcommMemHandle* memHandle);

HcommResult HcommMemUnreg(EndpointHandle endpointHandle, HcommMemHandle memHandle);

HcommResult HcommChannelConfigCreate(HcommChannelConfig* config);

HcommResult HcommChannelConfigDestroy(HcommChannelConfig config);

HcommResult HcommChannelConfigSetInt(HcommChannelConfig config, HcommChannelConfigType type, uint32_t value);

HcommResult HcommChannelCreateWithConfig(EndpointHandle endpointHandle, CommEngine engine,
                                         HcommChannelDesc* channelDescs, uint32_t channelNum, HcommChannelConfig config,
                                         ChannelHandle* channels);

HcommResult HcommChannelGetStatus(const ChannelHandle* channelList, uint32_t listNum, int32_t* statusList);

HcommResult HcommChannelDestroy(const ChannelHandle* channels, uint32_t channelNum);
} // namespace npu::tile_fwk
