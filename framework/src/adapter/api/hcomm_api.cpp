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
 * \file hcomm_api.cpp
 * \brief
 */

#include "adapter/api/hcomm_api.h"

#if defined(BUILD_WITH_CANN)
#include "adapter/manager/adapter_manager.h"
#include "acl/acl_base.h"
#include "hccl/hccl_channel.h"
#include "hccl/hccl_rank_graph.h"
#include "hccl/hccl_types.h"
#endif
#include "adapter/stubs/hcomm_stubs.h"

namespace npu::tile_fwk {
HcommResult HcommGetCommName(HcommHandle comm, char* commName)
{
#if defined(BUILD_WITH_CANN)
    void* func = AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::GetCommName);
    if (func != nullptr) {
        HcclResult (*hcommFunc)(HcclComm, char*) = reinterpret_cast<HcclResult (*)(HcclComm, char*)>(func);
        return static_cast<HcommResult>(hcommFunc(comm, commName));
    }
#endif
    return StubGetCommName(comm, commName);
}

HcommResult HcommGetL0TopoTypeEx(const char* group, HCommTopo* topoType, uint32_t flag)
{
#if defined(BUILD_WITH_CANN)
    void* func = AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::GetL0TopoTypeEx);
    if (func != nullptr) {
        HcclResult (*hcommFunc)(const char*, CommTopo*,
                                uint32_t) = reinterpret_cast<HcclResult (*)(const char*, CommTopo*, uint32_t)>(func);
        return static_cast<HcommResult>(hcommFunc(group, reinterpret_cast<CommTopo*>(topoType), flag));
    }
#endif
    return StubGetL0TopoTypeEx(group, topoType, flag);
}

HcommResult HcommGetCommHandleByGroup(const char* group, HcommHandle* commHandle)
{
#if defined(BUILD_WITH_CANN)
    void* func = AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::GetCommHandleByGroup);
    if (func != nullptr) {
        HcclResult (*hcommFunc)(const char*,
                                HcclComm*) = reinterpret_cast<HcclResult (*)(const char*, HcclComm*)>(func);
        return static_cast<HcommResult>(hcommFunc(group, commHandle));
    }
#endif
    return StubGetCommHandleByGroup(group, commHandle);
}

HcommResult HcommGetRootInfo(HcommRootInfo* rootInfo)
{
#if defined(BUILD_WITH_CANN)
    void* func = AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::GetRootInfo);
    if (func != nullptr) {
        HcclResult (*hcommFunc)(HcclRootInfo*) = reinterpret_cast<HcclResult (*)(HcclRootInfo*)>(func);
        return static_cast<HcommResult>(hcommFunc(reinterpret_cast<HcclRootInfo*>(rootInfo)));
    }
#endif
    return StubGetRootInfo(rootInfo);
}

HcommResult HcommCommInitRootInfo(uint32_t nRanks, const HcommRootInfo* rootInfo, uint32_t rank, HcommHandle* comm)
{
#if defined(BUILD_WITH_CANN)
    void* func = AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::CommInitRootInfo);
    if (func != nullptr) {
        HcclResult (*hcommFunc)(
            uint32_t, const HcclRootInfo*, uint32_t,
            HcclComm*) = reinterpret_cast<HcclResult (*)(uint32_t, const HcclRootInfo*, uint32_t, HcclComm*)>(func);
        return static_cast<HcommResult>(hcommFunc(nRanks, reinterpret_cast<const HcclRootInfo*>(rootInfo), rank, comm));
    }
#endif
    return StubCommInitRootInfo(nRanks, rootInfo, rank, comm);
}

HcommResult HcommCommDestroy(HcommHandle comm)
{
#if defined(BUILD_WITH_CANN)
    void* func = AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::CommDestroy);
    if (func != nullptr) {
        HcclResult (*hcommFunc)(HcclComm) = reinterpret_cast<HcclResult (*)(HcclComm)>(func);
        return static_cast<HcommResult>(hcommFunc(comm));
    }
#endif
    return StubCommDestroy(comm);
}

HcommResult HcommAllocComResourceByTiling(HcommHandle comm, void* stream, void* Mc2Tiling, void** commContext)
{
#if defined(BUILD_WITH_CANN)
    void* func = AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::AllocComResourceByTiling);
    if (func != nullptr) {
        HcclResult (*hcommFunc)(HcclComm, void*, void*,
                                void**) = reinterpret_cast<HcclResult (*)(HcclComm, void*, void*, void**)>(func);
        return static_cast<HcommResult>(hcommFunc(comm, stream, Mc2Tiling, commContext));
    }
#endif
    return StubAllocComResourceByTiling(comm, stream, Mc2Tiling, commContext);
}
HcommResult HcommCommGetHandleWithName(const char* commName, HcommHandle* comm)
{
#if defined(BUILD_WITH_CANN)
    void* func = AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::CommGetHandleWithName);
    if (func != nullptr) {
        HcclResult (*hcommFunc)(const char*,
                                HcclComm*) = reinterpret_cast<HcclResult (*)(const char*, HcclComm*)>(func);
        return static_cast<HcommResult>(hcommFunc(commName, comm));
    }
#endif
    return StubCommGetHandleWithName(commName, comm);
}

HcommResult HcommCommMemReg(HcommHandle comm, const char* memTag, const CommMem* mem, HcclMemHandle* memHandle)
{
#if defined(BUILD_WITH_CANN)
    void* func = AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::CommMemReg);
    if (func != nullptr) {
        HcclResult (*hcommFunc)(HcclComm, const char*, const ::CommMem*, ::HcclMemHandle*) = reinterpret_cast<
            HcclResult (*)(HcclComm, const char*, const ::CommMem*, ::HcclMemHandle*)>(func);
        return static_cast<HcommResult>(hcommFunc(comm, memTag, reinterpret_cast<const ::CommMem*>(mem), memHandle));
    }
#endif
    return StubCommMemReg(comm, memTag, mem, memHandle);
}

HcommResult HcommRankGraphGetLayers(HcommHandle comm, uint32_t** netLayers, uint32_t* netLayerNum)
{
#if defined(BUILD_WITH_CANN)
    void* func = AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::RankGraphGetLayers);
    if (func != nullptr) {
        HcclResult (*hcommFunc)(HcclComm, uint32_t**,
                                uint32_t*) = reinterpret_cast<HcclResult (*)(HcclComm, uint32_t**, uint32_t*)>(func);
        return static_cast<HcommResult>(hcommFunc(comm, netLayers, netLayerNum));
    }
#endif
    return StubRankGraphGetLayers(comm, netLayers, netLayerNum);
}

HcommResult HcommRankGraphGetLinks(HcommHandle comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank,
                                   CommLink** links, uint32_t* linkNum)
{
#if defined(BUILD_WITH_CANN)
    void* func = AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::RankGraphGetLinks);
    if (func != nullptr) {
        HcclResult (*hcommFunc)(HcclComm, uint32_t, uint32_t, uint32_t, ::CommLink**,
                                uint32_t*) = reinterpret_cast<HcclResult (*)(HcclComm, uint32_t, uint32_t, uint32_t,
                                                                             ::CommLink**, uint32_t*)>(func);
        return static_cast<HcommResult>(
            hcommFunc(comm, netLayer, srcRank, dstRank, reinterpret_cast<::CommLink**>(links), linkNum));
    }
#endif
    return StubRankGraphGetLinks(comm, netLayer, srcRank, dstRank, links, linkNum);
}

HcommResult HcommChannelAcquire(HcommHandle comm, CommEngine engine, const HcclChannelDesc* channelDescs,
                                uint32_t channelNum, ChannelHandle* channels)
{
#if defined(BUILD_WITH_CANN)
    void* func = AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::ChannelAcquire);
    if (func != nullptr) {
        HcclResult (*hcommFunc)(
            HcclComm, ::CommEngine, const ::HcclChannelDesc*, uint32_t,
            ::ChannelHandle*) = reinterpret_cast<HcclResult (*)(HcclComm, ::CommEngine, const ::HcclChannelDesc*,
                                                                uint32_t, ::ChannelHandle*)>(func);
        return static_cast<HcommResult>(hcommFunc(comm, static_cast<::CommEngine>(engine),
                                                  reinterpret_cast<const ::HcclChannelDesc*>(channelDescs), channelNum,
                                                  channels));
    }
#endif
    return StubChannelAcquire(comm, engine, channelDescs, channelNum, channels);
}

HcommResult HcommChannelGetRemoteMems(HcommHandle comm, ChannelHandle channel, uint32_t* memNum, CommMem** remoteMems,
                                      char*** memTags)
{
#if defined(BUILD_WITH_CANN)
    void* func = AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::ChannelGetRemoteMems);
    if (func != nullptr) {
        HcclResult (*hcommFunc)(HcclComm, ::ChannelHandle, uint32_t*, ::CommMem**,
                                char***) = reinterpret_cast<HcclResult (*)(HcclComm, ::ChannelHandle, uint32_t*,
                                                                           ::CommMem**, char***)>(func);
        return static_cast<HcommResult>(
            hcommFunc(comm, channel, memNum, reinterpret_cast<::CommMem**>(remoteMems), memTags));
    }
#endif
    return StubChannelGetRemoteMems(comm, channel, memNum, remoteMems, memTags);
}

HcommResult HcommAllGather(void* sendBuf, void* recvBuf, uint64_t sendCount, HcommDataType dataType, HcommHandle comm,
                           void* stream)
{
#if defined(BUILD_WITH_CANN)
    void* func = AdapterManager::Instance().GetHcclAdapter().GetFunction(HcclFunc::AllGather);
    if (func != nullptr) {
        HcclResult (*hcommFunc)(void*, void*, uint64_t, HcclDataType, HcclComm,
                                aclrtStream) = reinterpret_cast<HcclResult (*)(void*, void*, uint64_t, HcclDataType,
                                                                               HcclComm, aclrtStream)>(func);
        return static_cast<HcommResult>(
            hcommFunc(sendBuf, recvBuf, sendCount, static_cast<HcclDataType>(dataType), comm, stream));
    }
#endif
    return StubAllGather(sendBuf, recvBuf, sendCount, dataType, comm, stream);
}

#if defined(BUILD_WITH_CANN)
static_assert(static_cast<int32_t>(HCOMM_SUCCESS) == static_cast<int32_t>(HCCL_SUCCESS));
static_assert(sizeof(HcommResult) == sizeof(HcclResult));
static_assert(sizeof(HCommTopo) == sizeof(CommTopo));
static_assert(sizeof(HcommHandle) == sizeof(HcclComm));
static_assert(sizeof(HcommRootInfo) == sizeof(HcclRootInfo));
#endif
} // namespace npu::tile_fwk
