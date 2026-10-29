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
 * \file hcomm_stubs.cpp
 * \brief
 */

#include "adapter/stubs/hcomm_stubs.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk {
HcommResult StubGetCommName(HcommHandle comm, char* commName)
{
    ADAPTER_LOGD("Enter stub function of GetCommName.");
    (void)comm;
    (void)commName;
    return HCOMM_SUCCESS;
}

HcommResult StubGetL0TopoTypeEx(const char* group, HCommTopo* topoType, uint32_t flag)
{
    ADAPTER_LOGD("Enter stub function of GetL0TopoTypeEx.");
    (void)group;
    (void)topoType;
    (void)flag;
    return HCOMM_SUCCESS;
}

HcommResult StubGetCommHandleByGroup(const char* group, HcommHandle* commHandle)
{
    ADAPTER_LOGD("Enter stub function of GetCommHandleByGroup.");
    (void)group;
    (void)commHandle;
    return HCOMM_SUCCESS;
}

HcommResult StubGetRootInfo(HcommRootInfo* rootInfo)
{
    ADAPTER_LOGD("Enter stub function of GetRootInfo.");
    (void)rootInfo;
    return HCOMM_SUCCESS;
}

HcommResult StubCommInitRootInfo(uint32_t nRanks, const HcommRootInfo* rootInfo, uint32_t rank, HcommHandle* comm)
{
    ADAPTER_LOGD("Enter stub function of CommInitRootInfo.");
    (void)nRanks;
    (void)rootInfo;
    (void)rank;
    (void)comm;
    return HCOMM_SUCCESS;
}

HcommResult StubCommDestroy(HcommHandle comm)
{
    ADAPTER_LOGD("Enter stub function of CommDestroy.");
    (void)comm;
    return HCOMM_SUCCESS;
}

HcommResult StubAllocComResourceByTiling(HcommHandle comm, void* stream, void* Mc2Tiling, void** commContext)
{
    ADAPTER_LOGD("Enter stub function of AllocComResourceByTiling.");
    (void)comm;
    (void)stream;
    (void)Mc2Tiling;
    (void)commContext;
    return HCOMM_SUCCESS;
}

HcommResult StubCommGetHandleWithName(const char* commName, HcommHandle* comm)
{
    ADAPTER_LOGD("Enter stub function of CommGetHandleWithName.");
    (void)commName;
    (void)comm;
    return HCOMM_SUCCESS;
}

HcommResult StubCommMemReg(HcommHandle comm, const char* memTag, const CommMem* mem, HcclMemHandle* memHandle)
{
    ADAPTER_LOGD("Enter stub function of CommMemReg.");
    (void)comm;
    (void)memTag;
    (void)mem;
    (void)memHandle;
    return HCOMM_SUCCESS;
}

HcommResult StubRankGraphGetLayers(HcommHandle comm, uint32_t** netLayers, uint32_t* netLayerNum)
{
    ADAPTER_LOGD("Enter stub function of RankGraphGetLayers.");
    (void)comm;
    (void)netLayers;
    (void)netLayerNum;
    return HCOMM_SUCCESS;
}

HcommResult StubRankGraphGetLinks(HcommHandle comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank,
                                  CommLink** links, uint32_t* linkNum)
{
    ADAPTER_LOGD("Enter stub function of RankGraphGetLinks.");
    (void)comm;
    (void)netLayer;
    (void)srcRank;
    (void)dstRank;
    (void)links;
    (void)linkNum;
    return HCOMM_SUCCESS;
}

HcommResult StubChannelAcquire(HcommHandle comm, CommEngine engine, const HcclChannelDesc* channelDescs,
                               uint32_t channelNum, ChannelHandle* channels)
{
    ADAPTER_LOGD("Enter stub function of ChannelAcquire.");
    (void)comm;
    (void)engine;
    (void)channelDescs;
    (void)channelNum;
    (void)channels;
    return HCOMM_SUCCESS;
}

HcommResult StubChannelGetRemoteMems(HcommHandle comm, ChannelHandle channel, uint32_t* memNum, CommMem** remoteMems,
                                     char*** memTags)
{
    ADAPTER_LOGD("Enter stub function of ChannelGetRemoteMems.");
    (void)comm;
    (void)channel;
    (void)memNum;
    (void)remoteMems;
    (void)memTags;
    return HCOMM_SUCCESS;
}

HcommResult StubAllGather(void* sendBuf, void* recvBuf, uint64_t sendCount, HcommDataType dataType, HcommHandle comm,
                          void* stream)
{
    ADAPTER_LOGD("Enter stub function of AllGather.");
    (void)sendBuf;
    (void)recvBuf;
    (void)sendCount;
    (void)dataType;
    (void)comm;
    (void)stream;
    return HCOMM_SUCCESS;
}
} // namespace npu::tile_fwk
