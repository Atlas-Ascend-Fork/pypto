/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hcomm_res_adapter_types.h
 * \brief Symbol table of the HCOMM resource library (libhcomm.so): endpoint / jetty management.
 */

#pragma once

#include <string>
#include <map>

namespace npu::tile_fwk {
enum class HcommResFunc {
    EndpointCreate = 0,
    EndpointDestroy,
    EndpointGetListenPort,
    ResMemReg,
    ResMemUnreg,
    ChannelConfigCreate,
    ChannelConfigDestroy,
    ChannelConfigSetInt,
    ChannelCreateWithConfig,
    ChannelGetStatus,
    ChannelDestroy,
    Bottom
};

const std::string kHcommResLibName = "libhcomm.so";
const std::map<HcommResFunc, std::string> kHcommResFuncStrMap{
    {HcommResFunc::EndpointCreate, "HcommEndpointCreate"},
    {HcommResFunc::EndpointDestroy, "HcommEndpointDestroy"},
    {HcommResFunc::EndpointGetListenPort, "HcommEndpointGetListenPort"},
    {HcommResFunc::ResMemReg, "HcommMemReg"},
    {HcommResFunc::ResMemUnreg, "HcommMemUnreg"},
    {HcommResFunc::ChannelConfigCreate, "HcommChannelConfigCreate"},
    {HcommResFunc::ChannelConfigDestroy, "HcommChannelConfigDestroy"},
    {HcommResFunc::ChannelConfigSetInt, "HcommChannelConfigSetInt"},
    {HcommResFunc::ChannelCreateWithConfig, "HcommChannelCreateWithConfig"},
    {HcommResFunc::ChannelGetStatus, "HcommChannelGetStatus"},
    {HcommResFunc::ChannelDestroy, "HcommChannelDestroy"}};
} // namespace npu::tile_fwk
