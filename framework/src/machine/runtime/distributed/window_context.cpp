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

#include "window_context.h"

#include <algorithm>
#include <arpa/inet.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <utility>

#include "adapter/api/acl_api.h"
#include "adapter/api/hcomm_api.h"
#include "adapter/api/hcomm_res_api.h"
#include "tilefwk/distribute_context.h"
#include "tilefwk/pypto_fwk_log.h"

#ifdef BUILD_WITH_CANN // needs vendor struct layouts, see hcomm_res_define.h

namespace npu::tile_fwk::Distributed {
namespace {
constexpr uint32_t kNotifyNum = 3;                // notify slots per channel, same as DeepEP/HCCL
constexpr int32_t kCpuTimeoutSecs = 300;          // channel-connect deadline
constexpr int32_t kAclDevAttrVectorCoreNum = 201; // aclrtDevAttr::ACL_DEV_ATTR_VECTOR_CORE_NUM
constexpr int32_t kChannelReady = 0;
constexpr int32_t kChannelConnecting = 1;
constexpr int64_t kMinRegionBytes = 32; // engine write granularity; region sizes must be a multiple

bool IsSameEndpoint(const EndpointDesc& lhs, const EndpointDesc& rhs)
{
    return lhs.protocol == rhs.protocol && lhs.commAddr.type == rhs.commAddr.type &&
           std::memcmp(lhs.commAddr.raws, rhs.commAddr.raws, sizeof(lhs.commAddr.raws)) == 0 &&
           lhs.loc.locType == rhs.loc.locType && std::memcmp(lhs.loc.raws, rhs.loc.raws, sizeof(lhs.loc.raws)) == 0;
}

// The vendor returns a library-managed list that repeated calls on the same comm may
// invalidate, so copy it out immediately and never free it.
std::vector<CommLink> GetPeerLinks(void* comm, uint32_t netLayer, int32_t rank, int32_t peer)
{
    CommLink* links = nullptr;
    uint32_t numLinks = 0;
    auto ret = HcommRankGraphGetLinks(comm, netLayer, static_cast<uint32_t>(rank), static_cast<uint32_t>(peer), &links,
                                      &numLinks);
    DIST_CHECK(ret == HCOMM_SUCCESS) << "HcclRankGraphGetLinks(peer " << peer << ") failed, ret " << ret;
    return std::vector<CommLink>(links, links + numLinks);
}

std::vector<HcclChannelDesc> BuildPeerDescs(void* comm, uint32_t netLayer, int32_t rank, int32_t nrank,
                                            CommProtocol protocol, const std::vector<HcclMemHandle>& memHandles)
{
    std::vector<HcclChannelDesc> descs;
    descs.reserve(static_cast<size_t>(nrank - 1));
    for (int32_t peer = 0; peer < nrank; ++peer) {
        if (peer == rank) {
            continue;
        }
        const auto links = GetPeerLinks(comm, netLayer, rank, peer);
        bool found = false;
        for (const auto& link : links) {
            if (link.linkAttr.linkProtocol != protocol) {
                continue;
            }
            HcclChannelDesc desc;
            HcclChannelDescInit(&desc, 1);
            desc.remoteRank = static_cast<uint32_t>(peer);
            desc.channelProtocol = protocol;
            desc.localEndpoint = link.srcEndpointDesc;
            desc.remoteEndpoint = link.dstEndpointDesc;
            desc.notifyNum = kNotifyNum;
            desc.memHandles = const_cast<HcclMemHandle*>(memHandles.data());
            desc.memHandleNum = static_cast<uint32_t>(memHandles.size());
            descs.push_back(desc);
            found = true;
            break;
        }
        DIST_CHECK(found) << " link to peer " << peer << " on net layer " << netLayer;
    }
    return descs;
}

// True when all peers share one local UBC_CTP endpoint (DeepEP's topology). That is the
// only case where one jetty can reach every peer, so the write engine can give each AIV a
// private jetty. Otherwise every peer gets its own jetty on its own link endpoint and only
// that peer's issuer AIV may post.
bool AllPeersShareEndpoint(const std::vector<HcclChannelDesc>& peerDescs)
{
    for (const auto& desc : peerDescs) {
        if (!IsSameEndpoint(desc.localEndpoint, peerDescs.front().localEndpoint)) {
            return false;
        }
    }
    return true;
}

struct ChannelPlan {
    int32_t remoteRank; // peer this channel connects to
    int32_t pairIdx;    // 0: peer channel; 1..E: extra channel on a pair/triangle edge
    int32_t jettyIdx;   // local jetty (owned by that AIV) carrying the channel
};

std::vector<int32_t> ExtraChannelPartners(int32_t rank, int32_t nrank)
{
    // Ordinary pairs: (0, 1), (2, 3), ...
    if ((nrank % 2 == 0) || rank < nrank - 3) {
        return {rank ^ 1};
    }
    // the last three ranks connect to each other.
    if (rank == nrank - 3) {
        return {nrank - 2, nrank - 1};
    } else if (rank == nrank - 2) {
        return {nrank - 3, nrank - 1};
    } else {
        return {nrank - 2, nrank - 3};
    }
}

// Per-peer mode: one channel per peer, and channel i owns jetty i. The vector order is
// also the device table order.
std::vector<ChannelPlan> PlanPerPeerChannels(int32_t rank, int32_t nrank)
{
    const int32_t numPeers = nrank - 1;
    std::vector<ChannelPlan> plans;
    plans.reserve(static_cast<size_t>(numPeers));
    int32_t peerIdx = 0;
    for (int32_t peer = 0; peer < nrank; ++peer) {
        if (peer != rank) {
            plans.push_back(ChannelPlan{peer, 0, peerIdx});
            ++peerIdx;
        }
    }
    return plans;
}

// Multi-issuer mode table layout, where E = max(0, numJetties - numPeers) jetties have no
// dedicated peer:
//   [0, numPeers)    one channel per peer (channel i instantiates jetty i % numJetties)
//   [numPeers, +E)   primary extra channel per extra jetty (device-indexed)
//   [numPeers+E, +E) secondary extras of a triangle rank (setup only, not indexed)
// Extra channel k rides jetty numPeers + k - 1. The vector order is also the device table
// order.
std::vector<ChannelPlan> PlanMultiIssuerChannels(int32_t rank, int32_t nrank, int32_t numJetties)
{
    const int32_t numPeers = nrank - 1;
    const int32_t numExtraJetties = std::max(int32_t{0}, numJetties - numPeers);

    std::vector<int32_t> extraPartners;
    if (numExtraJetties > 0) {
        extraPartners = ExtraChannelPartners(rank, nrank);
    }

    std::vector<ChannelPlan> plans;
    int32_t peerChannelIdx = 0;
    for (int32_t peer = 0; peer < nrank; ++peer) {
        if (peer != rank) {
            plans.push_back(ChannelPlan{peer, 0, peerChannelIdx % numJetties});
            ++peerChannelIdx;
        }
    }

    for (int32_t partner : extraPartners) {
        for (int32_t extraIdx = 0; extraIdx < numExtraJetties; ++extraIdx) {
            plans.push_back(ChannelPlan{partner, extraIdx + 1, numPeers + extraIdx});
        }
    }

    return plans;
}

std::vector<uint8_t> AllGatherLocal(const void* local, uint64_t count, size_t elemBytes, HcommDataType type,
                                    int32_t nrank, void* comm, void* stream)
{
    auto numBytes = elemBytes * static_cast<size_t>(count);
    auto totalBytes = numBytes * static_cast<size_t>(nrank) + numBytes;
    void* devBuf = nullptr;
    DIST_CHECK(AclRtMalloc(&devBuf, totalBytes, AclRtMemMallocPolicy::NORMAL_ONLY) == 0)
        << "alloc allgather buffer of " << totalBytes << " bytes failed";
    auto* sendSlot = static_cast<uint8_t*>(devBuf) + numBytes * static_cast<size_t>(nrank);
    DIST_CHECK(AclRtMemcpy(sendSlot, numBytes, local, numBytes, AclRtMemcpyKind::HOST_TO_DEVICE) == 0)
        << "copy allgather payload failed";
    auto ret = HcommAllGather(sendSlot, devBuf, count, type, comm, stream);
    DIST_CHECK(ret == HCOMM_SUCCESS) << "HcclAllGather failed, ret " << ret;
    DIST_CHECK(AclRtSynchronizeStream(stream) == 0) << "sync ranks failed";
    std::vector<uint8_t> out(numBytes * static_cast<size_t>(nrank));
    DIST_CHECK(AclRtMemcpy(out.data(), out.size(), devBuf, out.size(), AclRtMemcpyKind::DEVICE_TO_HOST) == 0)
        << "gather allgather result failed";
    AclRtFree(devBuf);
    return out;
}

void SynchronizeRanks(int32_t nrank, void* comm, void* stream)
{
    uint8_t local = 1;
    (void)AllGatherLocal(&local, 1, sizeof(local), HCOMM_DATA_TYPE_INT8, nrank, comm, stream);
}

std::vector<uint32_t> GetNetLayers(void* comm)
{
    uint32_t* layers = nullptr;
    uint32_t numLayers = 0;
    auto ret = HcommRankGraphGetLayers(comm, &layers, &numLayers);
    DIST_CHECK(ret == HCOMM_SUCCESS) << "HcclRankGraphGetLayers failed, ret " << ret;
    DIST_CHECK(numLayers > 0) << "rank graph has no net layer";
    return std::vector<uint32_t>(layers, layers + numLayers);
}

// layer < 0 auto-selects (the top net layer); an explicit layer must exist in the rank graph.
uint32_t SelectNetLayer(void* comm, int32_t layer)
{
    const auto layers = GetNetLayers(comm);
    if (layer < 0) {
        return layers.back();
    }
    DIST_CHECK(std::find(layers.begin(), layers.end(), static_cast<uint32_t>(layer)) != layers.end())
        << "net layer " << layer << " not in rank graph";
    return static_cast<uint32_t>(layer);
}

// Tolerant variant of GetPeerLinks for topology dumps: a peer with no link on this layer
// (e.g. a cross-server peer at layer 0) is reported by the caller instead of aborting.
bool TryGetPeerLinks(void* comm, uint32_t netLayer, int32_t rank, int32_t peer, std::vector<CommLink>& links)
{
    CommLink* rawLinks = nullptr;
    uint32_t numLinks = 0;
    auto ret = HcommRankGraphGetLinks(comm, netLayer, static_cast<uint32_t>(rank), static_cast<uint32_t>(peer),
                                      &rawLinks, &numLinks);
    if (ret != HCOMM_SUCCESS) {
        return false;
    }
    links.assign(rawLinks, rawLinks + numLinks);
    return true;
}

const char* TopoName(HCommTopo topo)
{
    switch (topo) {
        case HCOMM_TOPO_CLOS:
            return "CLOS";
        case HCOMM_TOPO_1DMESH:
            return "1DMesh";
        case HCOMM_TOPO_CUSTOM:
            return "Custom";
        default:
            return "Unknown";
    }
}

const char* ProtocolName(CommProtocol protocol)
{
    switch (protocol) {
        case COMM_PROTOCOL_ROCE:
            return "ROCE";
        case COMM_PROTOCOL_PCIE:
            return "PCIE";
        case COMM_PROTOCOL_UB_CTP:
            return "UB_CTP";
        case COMM_PROTOCOL_UBC_TP:
            return "UBC_TP";
        case COMM_PROTOCOL_UB_MEM:
            return "UB_MEM";
        case COMM_PROTOCOL_UBOE:
            return "UBOE";
        case COMM_PROTOCOL_UB_RTP:
            return "UB_RTP";
        default:
            return "unknown";
    }
}

std::string FormatCommAddr(const CommAddr& addr)
{
    char buf[INET6_ADDRSTRLEN] = {0};
    switch (addr.type) {
        case COMM_ADDR_TYPE_IP_V4:
            if (inet_ntop(AF_INET, &addr.addr, buf, sizeof(buf)) != nullptr) {
                return buf;
            }
            return "ipv4 ?";
        case COMM_ADDR_TYPE_IP_V6:
            if (inet_ntop(AF_INET6, &addr.addr6, buf, sizeof(buf)) != nullptr) {
                return buf;
            }
            return "ipv6 ?";
        case COMM_ADDR_TYPE_ID:
            return "id " + std::to_string(addr.id);
        case COMM_ADDR_TYPE_EID: {
            std::ostringstream hex;
            for (size_t i = 0; i < sizeof(addr.eid); ++i) {
                hex << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(addr.eid[i]);
            }
            return "eid " + hex.str();
        }
        default:
            return "addr ?";
    }
}

std::string FormatLocDevice(const EndpointLoc& loc)
{
    if (loc.locType == ENDPOINT_LOC_TYPE_DEVICE) {
        return std::to_string(loc.device.devPhyId);
    }
    if (loc.locType == ENDPOINT_LOC_TYPE_HOST) {
        return "host " + std::to_string(loc.host.id);
    }
    return "?";
}

void WaitChannelsReady(const std::vector<ChannelHandle>& channels)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(kCpuTimeoutSecs);
    std::vector<int32_t> statuses(channels.size(), kChannelConnecting);
    while (true) {
        const auto ret = tile_fwk::HcommChannelGetStatus(channels.data(), static_cast<uint32_t>(channels.size()),
                                                         statuses.data());
        if (ret != static_cast<HcommResult>(HCOMM_E_AGAIN)) {
            DIST_CHECK(ret == HCOMM_SUCCESS) << "HcclChannelGetStatus failed, ret " << ret;
        }
        if (ret == HCOMM_SUCCESS) {
            bool allReady = true;
            for (const auto status : statuses) {
                DIST_CHECK(status == kChannelReady || status == kChannelConnecting)
                    << "channel connect failed, status " << status;
                if (status != kChannelReady) {
                    allReady = false;
                }
            }
            if (allReady) {
                break;
            }
        }
        DIST_CHECK(std::chrono::steady_clock::now() < deadline) << "channel connect timeout";
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}
} // namespace

struct WindowContext::Impl {
    struct MemoryRegion {
        std::string name;   // engine memTag (the caller-provided region name)
        int64_t ptr{};      // local base address
        int64_t size{};     // bytes; identical on every rank per name (validated at connect)
        int32_t protocol{}; // kPolicyUbMem | kPolicyUrma bitmask
        HcclMemHandle commHandle{nullptr};
        std::vector<HcommMemHandle> endpointHandles; // per jetty endpoint, filled at connect
    };

    enum class State { Created, Connected, Closed };

    Impl(int32_t rank, int32_t nrank, void* stream, const std::string& hcclGroupName, const std::string& name,
         int32_t layer);
    ~Impl();

    void RegisterMemory(int64_t tensorPtr, int64_t size, const std::string& regionName, int32_t protocol);
    void Connect();
    void AcquireUlbmemChannels(); // resolve every peer's region bases (regionBases_)
    void AcquireJettyChannels();  // multi-issuer: per-AIV jetties; else one jetty per peer
    void WriteDeviceWindow();     // fill region table + pypto::dist::Window and copy them to device
    void Close();                 // all-rank collective teardown; best-effort, never throws

    std::string ToString() const;

    int32_t rank_;
    int32_t nrank_;
    void* stream_{nullptr};
    std::string hcclGroupName_;
    std::string name_;
    State state_{State::Created};

    std::vector<MemoryRegion> regions_;             // regions_[0] = control; index == region id
    std::vector<int32_t> ubmemRegionIds_;           // regions_ indices on the UB_MEM plane
    std::vector<int32_t> urmaRegionIds_;            // regions_ indices on the URMA plane
    std::vector<int32_t> urmaIndex_;                // per region: position in urmaRegionIds_, -1 = absent
    std::vector<std::vector<int64_t>> regionBases_; // [region][rank]; [r][rank_] = local base

    void* comm_{nullptr};
    std::vector<ChannelHandle> ubmemChannels_; // UBMEM channel per peer, for teardown in Close()
    int32_t numJetties_{0};                    // one private jetty per AIV in multi-issuer mode
    bool multiIssuer_{false};                  // one local endpoint covers every peer
    std::vector<EndpointHandle> jettyEndpoints_;
    std::vector<ChannelHandle> channelHandles_;
    std::vector<std::string> channelNames_;               // HCOMM shallow-copies the pointer; keep alive
    std::vector<pypto::dist::RegionEntry> regionEntries_; // host copy of the region table
    void* windowDevPtr_{nullptr};                         // device copy of the pypto::dist::Window descriptor
    void* regionTableDevPtr_{nullptr};                    // device copy of RegionEntry[num_regions]
    void* jettyTableDevPtr_{nullptr};                     // device copy of the channel table (Window::jetty_ptrs)
    int32_t netLayer_{-1};                                // requested layer at create; resolved at connect
};

WindowContext::Impl::Impl(int32_t rank, int32_t nrank, void* stream, const std::string& hcclGroupName,
                          const std::string& name, int32_t layer)
    : rank_(rank), nrank_(nrank), stream_(stream), hcclGroupName_(hcclGroupName), name_(name), netLayer_(layer)
{
    DIST_CHECK(nrank > 1 && rank >= 0 && rank < nrank) << "invalid rank/nrank: " << rank << "/" << nrank;
    DIST_CHECK(nrank <= pypto::dist::kNumMaxRanks)
        << "nrank " << nrank << " exceeds supported max " << pypto::dist::kNumMaxRanks << " (remote_ptrs table size)";
    DIST_CHECK(!hcclGroupName_.empty()) << "hccl group name is empty";
}

void WindowContext::Impl::RegisterMemory(int64_t tensorPtr, int64_t size, const std::string& regionName,
                                         int32_t protocol)
{
    DIST_CHECK(state_ == State::Created) << "register_memory is only allowed before connect()";
    DIST_CHECK(!regionName.empty()) << "region name is empty";
    DIST_CHECK(protocol != 0 && (protocol & ~(pypto::dist::kPolicyUbMem | pypto::dist::kPolicyUrma)) == 0)
        << "region '" << regionName << "' has invalid protocol mask " << protocol
        << " (expected UB_MEM (1), URMA (2), or both (3))";
    DIST_CHECK(tensorPtr != 0) << "region '" << regionName << "' pointer is null";
    DIST_CHECK(size > 0 && size % kMinRegionBytes == 0)
        << "region '" << regionName << "' size " << size << " must be positive and a multiple of " << kMinRegionBytes;
    for (const auto& reg : regions_) {
        DIST_CHECK(reg.name != regionName) << "region '" << regionName << "' already registered on this window";
    }

    MemoryRegion reg;
    reg.name = regionName;
    reg.ptr = tensorPtr;
    reg.size = size;
    reg.protocol = protocol;
    regions_.push_back(reg);
}

std::shared_ptr<WindowContext> CreateWindow(int32_t rank, int32_t nrank, int64_t streamAddr,
                                            const std::string& hcclGroupName, const std::string& name, int32_t layer)
{
    return std::make_shared<WindowContext>(rank, nrank, reinterpret_cast<void*>(streamAddr), hcclGroupName, name,
                                           layer);
}

std::string GetGroupTopology(const std::string& hcclGroupName, int32_t rank, int32_t nrank, int32_t layer)
{
    void* comm = nullptr;
    auto ret = HcommCommGetHandleWithName(hcclGroupName.c_str(), &comm);
    DIST_CHECK(ret == HCOMM_SUCCESS) << "HcclCommGetHandleWithName(" << hcclGroupName << ") failed, ret " << ret;
    DIST_CHECK(comm != nullptr) << "HCCL comm of group " << hcclGroupName
                                << " unavailable (not a CANN build, libhccl not loaded, or no such comm)";

    const auto layers = GetNetLayers(comm);
    DIST_CHECK(layer < 0 || std::find(layers.begin(), layers.end(), static_cast<uint32_t>(layer)) != layers.end())
        << "net layer " << layer << " not in rank graph of group " << hcclGroupName;

    std::ostringstream out;
    for (const auto l : layers) {
        if (layer >= 0 && static_cast<uint32_t>(layer) != l) {
            continue;
        }
        HCommTopo topo = HCOMM_TOPO_RESERVED;
        (void)HcommRankGraphGetTopoTypeByLayer(comm, l, &topo); // best-effort; "Unknown" on failure

        // Ranks reachable from this rank at layer l (copy out immediately: library-managed
        // list that repeated calls may invalidate). Falls back to the whole comm.
        std::vector<uint32_t> peers;
        uint32_t* ranks = nullptr;
        uint32_t numRanks = 0;
        if (HcommRankGraphGetRanksByLayer(comm, l, &ranks, &numRanks) == HCOMM_SUCCESS) {
            peers.assign(ranks, ranks + numRanks);
        } else {
            peers.reserve(static_cast<size_t>(nrank));
            for (int32_t r = 0; r < nrank; ++r) {
                peers.push_back(static_cast<uint32_t>(r));
            }
        }

        const std::string separator(88, '-');
        out << "HCCL Topology\n" << separator << "\n";
        out << "Local Rank   : " << rank << " / " << nrank << "\n";
        out << "Network Layer: " << l << "\n";
        out << "Topology     : " << TopoName(topo) << "\n";
        out << "Reachable    : " << peers.size() << " / " << nrank << " ranks\n";
        out << separator << "\n\n";

        // Fixed-width columns (%-Ns): an address longer than its column simply extends it.
        out << "Peer  Transport  Hops  LocalDevice  LocalAddress        RemoteDevice  RemoteAddress\n";
        out << "----  ---------  ----  -----------  ------------------  ------------  ------------------\n";
        std::vector<uint32_t> unlinked;
        for (const auto peer : peers) {
            // Only comm ranks can be queried; a rank id outside [0, nrank) means the vendor
            // reported world ranks of a subgroup — query what maps, skip the rest.
            if (static_cast<int32_t>(peer) == rank || peer >= static_cast<uint32_t>(nrank)) {
                continue;
            }
            std::vector<CommLink> links;
            if (!TryGetPeerLinks(comm, l, rank, static_cast<int32_t>(peer), links) || links.empty()) {
                unlinked.push_back(peer);
                continue;
            }
            for (const auto& link : links) {
                char row[512];
                (void)snprintf(row, sizeof(row), "%-4s  %-9s  %-4s  %-11s  %-18s  %-12s  %s\n",
                               std::to_string(peer).c_str(), ProtocolName(link.linkAttr.linkProtocol),
                               std::to_string(static_cast<int>(link.linkAttr.hop)).c_str(),
                               FormatLocDevice(link.srcEndpointDesc.loc).c_str(),
                               FormatCommAddr(link.srcEndpointDesc.commAddr).c_str(),
                               FormatLocDevice(link.dstEndpointDesc.loc).c_str(),
                               FormatCommAddr(link.dstEndpointDesc.commAddr).c_str());
                out << row;
            }
        }
        for (const auto peer : unlinked) {
            out << "peer " << peer << ": no link\n";
        }
        out << "\n";
    }
    return out.str();
}

void WindowContext::Impl::Connect()
{
    DIST_CHECK(state_ == State::Created) << "connect() called on an already-connected or closed window";
    DIST_CHECK(!regions_.empty()) << "no memory registered; register the control region before connect()";

    auto ret = HcommCommGetHandleWithName(hcclGroupName_.c_str(), &comm_);
    DIST_CHECK(ret == HCOMM_SUCCESS) << "HcclCommGetHandleWithName(" << hcclGroupName_ << ") failed, ret " << ret;
    DIST_CHECK(comm_ != nullptr) << "HCCL comm of group " << hcclGroupName_
                                 << " unavailable (not a CANN build, libhccl not loaded, or no such comm)";

    netLayer_ = static_cast<int32_t>(SelectNetLayer(comm_, netLayer_));

    regionBases_.assign(regions_.size(), std::vector<int64_t>(static_cast<size_t>(nrank_), 0));
    urmaIndex_.assign(regions_.size(), -1);
    for (size_t id = 0; id < regions_.size(); ++id) {
        regionBases_[id][rank_] = regions_[id].ptr;
        if ((regions_[id].protocol & pypto::dist::kPolicyUbMem) != 0) {
            ubmemRegionIds_.push_back(static_cast<int32_t>(id));
        }
        if ((regions_[id].protocol & pypto::dist::kPolicyUrma) != 0) {
            urmaIndex_[id] = static_cast<int32_t>(urmaRegionIds_.size());
            urmaRegionIds_.push_back(static_cast<int32_t>(id));
        }
    }

    for (const auto id : ubmemRegionIds_) {
        auto& reg = regions_[id];
        CommMem mem{};
        mem.type = COMM_MEM_TYPE_DEVICE;
        mem.addr = reinterpret_cast<void*>(reg.ptr);
        mem.size = static_cast<uint64_t>(reg.size);
        ret = HcommCommMemReg(comm_, reg.name.c_str(), &mem, &reg.commHandle);
        DIST_CHECK(ret == HCOMM_SUCCESS) << "HcclCommMemReg(" << reg.name << ") failed, ret " << ret;
    }

    SynchronizeRanks(nrank_, comm_, stream_);
    AcquireUlbmemChannels();
    AcquireJettyChannels();
    WaitChannelsReady(channelHandles_);
    WriteDeviceWindow();
    state_ = State::Connected;
    DISTRIBUTED_LOGI("Window %s connected on rank %d/%d, regions %zu, control %lldB, topology %s, jetties %zu, "
                     "channels %zu",
                     name_.empty() ? hcclGroupName_.c_str() : name_.c_str(), rank_, nrank_, regions_.size(),
                     static_cast<long long>(regions_.front().size), multiIssuer_ ? "multi-issuer" : "per-endpoint",
                     jettyEndpoints_.size(), channelHandles_.size());
}

void WindowContext::Impl::AcquireUlbmemChannels()
{
    std::vector<HcclMemHandle> handles;
    handles.reserve(ubmemRegionIds_.size());
    for (const auto id : ubmemRegionIds_) {
        handles.push_back(regions_[id].commHandle);
    }
    auto descs = BuildPeerDescs(comm_, netLayer_, rank_, nrank_, COMM_PROTOCOL_UB_MEM, handles);
    std::vector<ChannelHandle> channels(descs.size(), 0);
    auto ret = HcommChannelAcquire(comm_, COMM_ENGINE_AIV, descs.data(), static_cast<uint32_t>(descs.size()),
                                   channels.data());
    DIST_CHECK(ret == HCOMM_SUCCESS) << "HcclChannelAcquire(UB_MEM) failed, ret " << ret;

    ubmemChannels_ = std::move(channels); // retained for teardown in Close()
    for (size_t i = 0; i < ubmemChannels_.size(); ++i) {
        const auto peer = static_cast<int32_t>(descs[i].remoteRank);
        uint32_t numMems = 0;
        CommMem* remoteMems = nullptr;
        char** memTags = nullptr;
        auto memRet = HcommChannelGetRemoteMems(comm_, ubmemChannels_[i], &numMems, &remoteMems, &memTags);
        DIST_CHECK(memRet == HCOMM_SUCCESS) << "HcclChannelGetRemoteMems(peer " << peer << ") failed, ret " << memRet;

        // Region names are the engine memTags: match each local UB_MEM region by name.
        std::unordered_map<std::string, const CommMem*> peerRegions;
        for (uint32_t m = 0; m < numMems; ++m) {
            if (memTags == nullptr || memTags[m] == nullptr) {
                continue;
            }
            peerRegions.emplace(memTags[m], &remoteMems[m]);
        }

        // Every local region must exist on the peer with the same size.
        for (const auto id : ubmemRegionIds_) {
            const auto& reg = regions_[id];
            const auto it = peerRegions.find(reg.name);
            DIST_CHECK(it != peerRegions.end()) << "peer " << peer << " did not register region '" << reg.name << "'";
            DIST_CHECK(it->second->size == static_cast<uint64_t>(reg.size))
                << "region '" << reg.name << "' size mismatch with peer " << peer << ": local " << reg.size
                << ", remote " << it->second->size;
            regionBases_[id][peer] = reinterpret_cast<int64_t>(it->second->addr);
        }
    }
}

void WindowContext::Impl::AcquireJettyChannels()
{
    int32_t deviceId = -1;
    int64_t aivNum = 0;
    DIST_CHECK(AclRtGetDevice(&deviceId) == 0 && deviceId >= 0) << "get current device failed";
    DIST_CHECK(AclRtGetDeviceInfo(deviceId, kAclDevAttrVectorCoreNum, &aivNum) == 0 && aivNum > 0)
        << "query vector core count of device " << deviceId << " failed";
    numJetties_ = static_cast<int32_t>(aivNum); // one jetty per AIV (DeepEP: num_jetties = num_vec_cores)

    std::vector<HcclMemHandle> commHandles;
    commHandles.reserve(ubmemRegionIds_.size());
    for (const auto id : ubmemRegionIds_) {
        commHandles.push_back(regions_[id].commHandle);
    }

    auto ubcDescs = BuildPeerDescs(comm_, netLayer_, rank_, nrank_, COMM_PROTOCOL_UBC_CTP, commHandles);
    multiIssuer_ = AllPeersShareEndpoint(ubcDescs);

    // Multi-issuer: one private jetty per AIV, all on the common local endpoint.
    // Per-peer: one jetty per peer on that peer's link endpoint; only its issuer AIV posts.
    const auto numEndpoints = multiIssuer_ ? static_cast<size_t>(numJetties_) : ubcDescs.size();
    jettyEndpoints_.resize(numEndpoints);
    for (auto& reg : regions_) {
        reg.endpointHandles.assign(numEndpoints, nullptr);
    }
    for (size_t j = 0; j < numEndpoints; ++j) {
        const auto& endpointDesc = multiIssuer_ ? ubcDescs.front().localEndpoint : ubcDescs[j].localEndpoint;
        auto ret = tile_fwk::HcommEndpointCreate(&endpointDesc, &jettyEndpoints_[j]);
        DIST_CHECK(ret == HCOMM_SUCCESS) << "HcclEndpointCreate(jetty " << j << ") failed, ret " << ret;
        // Every URMA-plane region is registered on every jetty endpoint.
        for (const auto id : urmaRegionIds_) {
            auto& reg = regions_[id];
            CommMem jettyMem{};
            jettyMem.type = COMM_MEM_TYPE_DEVICE;
            jettyMem.addr = reinterpret_cast<void*>(reg.ptr);
            jettyMem.size = static_cast<uint64_t>(reg.size);
            ret = tile_fwk::HcommMemReg(jettyEndpoints_[j], reg.name.c_str(), &jettyMem, &reg.endpointHandles[j]);
            DIST_CHECK(ret == HCOMM_SUCCESS) << "HcommMemReg(jetty " << j << ", region " << reg.name << ") failed, "
                                             << "ret " << ret;
        }
    }

    // Exchange listen ports once per window: AllGather one uint16 per rank over the comm.
    uint32_t listenPort = 0;
    auto ret = tile_fwk::HcommEndpointGetListenPort(jettyEndpoints_.front(), &listenPort);
    DIST_CHECK(ret == HCOMM_SUCCESS) << "HcclEndpointGetListenPort failed, ret " << ret;
    DIST_CHECK(listenPort > 0 && listenPort <= std::numeric_limits<uint16_t>::max()) << "invalid listen port";
    const auto localPort = static_cast<uint16_t>(listenPort);
    const auto ports = AllGatherLocal(&localPort, 1, sizeof(localPort), HCOMM_DATA_TYPE_UINT16, nrank_, comm_, stream_);
    const auto* portList = reinterpret_cast<const uint16_t*>(ports.data());
    const std::vector<uint16_t> listenPorts(portList, portList + nrank_);
    for (const auto port : listenPorts) {
        DIST_CHECK(port != 0) << "peer reported an invalid listen port";
    }

    const auto plans = multiIssuer_ ? PlanMultiIssuerChannels(rank_, nrank_, numJetties_) :
                                      PlanPerPeerChannels(rank_, nrank_);
    const auto numChannels = static_cast<int32_t>(plans.size());

    HcommChannelConfig config = nullptr;
    ret = tile_fwk::HcommChannelConfigCreate(&config);
    DIST_CHECK(ret == HCOMM_SUCCESS) << "HcommChannelConfigCreate failed, ret " << ret;
    ret = tile_fwk::HcommChannelConfigSetInt(config, HCOMM_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE, 1);
    DIST_CHECK(ret == HCOMM_SUCCESS) << "HcommChannelConfigSetInt(IS_SHARED_QUEUE) failed, ret " << ret;

    channelHandles_.assign(static_cast<size_t>(numChannels), 0);
    channelNames_.resize(static_cast<size_t>(numChannels));
    for (int32_t channelIdx = 0; channelIdx < numChannels; ++channelIdx) {
        const auto& plan = plans[channelIdx];
        const auto peerRank = plan.remoteRank;
        auto& name = channelNames_[channelIdx];
        name = (name_.empty() ? hcclGroupName_ : name_) + "/" + std::to_string(std::min(rank_, peerRank)) + "-" +
               std::to_string(std::max(rank_, peerRank)) + "/" + std::to_string(plan.pairIdx);
        DIST_CHECK(name.size() <= HCOMM_CHANNEL_NAME_MAX_LEN) << "channel name exceeds HCOMM_CHANNEL_NAME_MAX_LEN";
        // Attach the URMA-plane regions of this jetty's endpoint in urma-index order, so
        // the peer's remoteBufferAddr[urma_index] is the region's remote range.
        HcommChannelDesc desc;
        HcommChannelDescInit(&desc, 1);
        desc.remoteEndpoint = ubcDescs[peerRank < rank_ ? peerRank : peerRank - 1].remoteEndpoint;
        desc.notifyNum = kNotifyNum;
        std::vector<HcommMemHandle> endpointRegionHandles;
        endpointRegionHandles.reserve(urmaRegionIds_.size());
        for (const auto id : urmaRegionIds_) {
            endpointRegionHandles.push_back(regions_[id].endpointHandles[plan.jettyIdx]);
        }
        desc.memHandles = endpointRegionHandles.data();
        desc.memHandleNum = static_cast<uint32_t>(endpointRegionHandles.size());
        // The lower rank listens; the higher rank connects.
        desc.role = rank_ < peerRank ? HCOMM_SOCKET_ROLE_SERVER : HCOMM_SOCKET_ROLE_CLIENT;
        desc.port = listenPorts[std::min(rank_, peerRank)];
        desc.channelName = name.c_str();
        ret = tile_fwk::HcommChannelCreateWithConfig(jettyEndpoints_[plan.jettyIdx], COMM_ENGINE_AIV, &desc, 1, config,
                                                     &channelHandles_[channelIdx]);
        DIST_CHECK(ret == HCOMM_SUCCESS) << "HcommChannelCreateWithConfig(" << name << ") failed, ret " << ret;
    }
    ret = tile_fwk::HcommChannelConfigDestroy(config);
    DIST_CHECK(ret == HCOMM_SUCCESS) << "HcommChannelConfigDestroy failed, ret " << ret;
}

void WindowContext::Impl::WriteDeviceWindow()
{
    pypto::dist::Window window{};
    window.rank = rank_;
    window.nrank = nrank_;
    window.num_jetties = multiIssuer_ ? numJetties_ : static_cast<int32_t>(jettyEndpoints_.size());
    window.multi_issuer = multiIssuer_ ? 1 : 0;
    window.num_regions = static_cast<int32_t>(regions_.size());

    DIST_CHECK(regionBases_.size() == regions_.size()) << "region bases incomplete";
    regionEntries_.assign(regions_.size(), pypto::dist::RegionEntry{});
    for (size_t id = 0; id < regions_.size(); ++id) {
        auto& entry = regionEntries_[id];
        const auto& reg = regions_[id];
        entry.size = reg.size;
        entry.policy = reg.protocol;
        entry.urma_index = urmaIndex_[id];
        // The local base identifies the region on device (region_of) on every plane.
        entry.remote_ptrs[rank_] = regionBases_[id][rank_];
        if ((reg.protocol & pypto::dist::kPolicyUbMem) != 0) {
            for (int32_t r = 0; r < nrank_; ++r) {
                DIST_CHECK(regionBases_[id][r] != 0)
                    << "region '" << reg.name << "' base of rank " << r << " unresolved";
                entry.remote_ptrs[r] = regionBases_[id][r];
            }
        }
    }
    const auto regionBytes = regionEntries_.size() * sizeof(pypto::dist::RegionEntry);
    DIST_CHECK(AclRtMalloc(&regionTableDevPtr_, regionBytes, AclRtMemMallocPolicy::HUGE_FIRST) == 0)
        << "alloc region table of " << regionBytes << " bytes failed";
    DIST_CHECK(AclRtMemcpy(regionTableDevPtr_, regionBytes, regionEntries_.data(), regionBytes,
                           AclRtMemcpyKind::HOST_TO_DEVICE) == 0)
        << "copy region table failed";
    window.region_table = reinterpret_cast<int64_t>(regionTableDevPtr_);

    const auto tableBytes = channelHandles_.size() * sizeof(ChannelHandle);
    DIST_CHECK(AclRtMalloc(&jettyTableDevPtr_, tableBytes, AclRtMemMallocPolicy::HUGE_FIRST) == 0)
        << "alloc jetty table of " << tableBytes << " bytes failed";
    DIST_CHECK(AclRtMemcpy(jettyTableDevPtr_, tableBytes, channelHandles_.data(), tableBytes,
                           AclRtMemcpyKind::HOST_TO_DEVICE) == 0)
        << "copy jetty table failed";
    window.jetty_ptrs = reinterpret_cast<int64_t>(jettyTableDevPtr_);

    DIST_CHECK(AclRtMalloc(&windowDevPtr_, sizeof(window), AclRtMemMallocPolicy::HUGE_FIRST) == 0)
        << "alloc window descriptor failed";
    DIST_CHECK(AclRtMemcpy(windowDevPtr_, sizeof(window), &window, sizeof(window), AclRtMemcpyKind::HOST_TO_DEVICE) ==
               0)
        << "copy window descriptor failed";
}

void WindowContext::Impl::Close()
{
    if (state_ == State::Closed) {
        return;
    }
    const auto windowName = name_.empty() ? hcclGroupName_ : name_;
    state_ = State::Closed;
    if (windowDevPtr_ != nullptr) {
        (void)AclRtFree(windowDevPtr_);
        windowDevPtr_ = nullptr;
    }
    if (regionTableDevPtr_ != nullptr) {
        (void)AclRtFree(regionTableDevPtr_);
        regionTableDevPtr_ = nullptr;
    }
    if (jettyTableDevPtr_ != nullptr) {
        (void)AclRtFree(jettyTableDevPtr_);
        jettyTableDevPtr_ = nullptr;
    }
    if (!channelHandles_.empty()) {
        auto ret = tile_fwk::HcommChannelDestroy(channelHandles_.data(), static_cast<uint32_t>(channelHandles_.size()));
        if (ret != HCOMM_SUCCESS) {
            DISTRIBUTED_LOGW("Window %s: HcommChannelDestroy failed, ret %d", windowName.c_str(), ret);
        }
        channelHandles_.clear();
    }
    if (!ubmemChannels_.empty()) {
        auto ret = tile_fwk::HcommChannelDestroy(ubmemChannels_.data(), static_cast<uint32_t>(ubmemChannels_.size()));
        if (ret != HCOMM_SUCCESS) {
            DISTRIBUTED_LOGW("Window %s: HcommChannelDestroy(UB_MEM) failed, ret %d", windowName.c_str(), ret);
        }
        ubmemChannels_.clear();
    }
    // The comm-level HcclCommMemReg handles have no unreg wrapper in the vendor API; they
    // are released with the comm itself (DeepEP carries the same TODO).
    for (size_t j = 0; j < jettyEndpoints_.size(); ++j) {
        for (auto& reg : regions_) {
            if (reg.endpointHandles.size() != jettyEndpoints_.size() || reg.endpointHandles[j] == nullptr) {
                continue;
            }
            auto unregRet = tile_fwk::HcommMemUnreg(jettyEndpoints_[j], reg.endpointHandles[j]);
            if (unregRet != HCOMM_SUCCESS) {
                DISTRIBUTED_LOGW("Window %s: HcommMemUnreg(jetty %zu, region %s) failed, ret %d", windowName.c_str(), j,
                                 reg.name.c_str(), unregRet);
            }
            reg.endpointHandles[j] = nullptr;
        }
        auto destroyRet = tile_fwk::HcommEndpointDestroy(jettyEndpoints_[j]);
        if (destroyRet != HCOMM_SUCCESS) {
            DISTRIBUTED_LOGW("Window %s: HcommEndpointDestroy(jetty %zu) failed, ret %d", windowName.c_str(), j,
                             destroyRet);
        }
    }
    jettyEndpoints_.clear();
}

WindowContext::Impl::~Impl() { Close(); }

std::string WindowContext::Impl::ToString() const
{
    std::ostringstream info;
    info << "Window(name=" << (name_.empty() ? hcclGroupName_ : name_) << ")\n";
    info << "  rank " << rank_ << "/" << nrank_ << ", regions " << regions_.size() << ", state "
         << (state_ == State::Created ? "created" : (state_ == State::Connected ? "connected" : "closed"));
    info << "\n  connect: layer " << netLayer_ << ", topology " << (multiIssuer_ ? "multi-issuer" : "per-endpoint")
         << ", jetties " << jettyEndpoints_.size() << ", channels " << channelHandles_.size() << ", window 0x"
         << std::hex << reinterpret_cast<uintptr_t>(windowDevPtr_);
    // Dump the region table exactly as it was shipped to the device.
    for (size_t id = 0; id < regions_.size(); ++id) {
        const auto& reg = regions_[id];
        const auto& entry = id < regionEntries_.size() ? regionEntries_[id] : pypto::dist::RegionEntry{};
        info << "\n  region[" << id << "] '" << reg.name << "' size " << reg.size << " policy "
             << ((reg.protocol & pypto::dist::kPolicyUbMem) != 0 ? "UB_MEM" : "");
        if ((reg.protocol & pypto::dist::kPolicyUrma) != 0) {
            info << (reg.protocol & pypto::dist::kPolicyUbMem ? "|URMA" : "URMA");
        }
        info << " local 0x" << std::hex << static_cast<uint64_t>(reg.ptr) << std::dec << " urma_index "
             << (id < urmaIndex_.size() ? urmaIndex_[id] : -1) << ":";
        for (int32_t r = 0; r < nrank_; ++r) {
            info << " [r" << r << "] 0x" << std::hex << entry.remote_ptrs[r];
        }
    }
    info << "\n  jetty_ptrs (table 0x" << std::hex << reinterpret_cast<uintptr_t>(jettyTableDevPtr_) << "):";
    for (size_t j = 0; j < channelHandles_.size(); ++j) {
        info << " [" << std::dec << j << "] 0x" << std::hex << channelHandles_[j];
    }
    return info.str();
}
// The public class is a thin shell over Impl; the definitions above carry the real logic.

WindowContext::WindowContext(int32_t rank, int32_t nrank, void* stream, const std::string& hcclGroupName,
                             const std::string& name, int32_t layer)
    : impl_(std::make_unique<Impl>(rank, nrank, stream, hcclGroupName, name, layer))
{}

WindowContext::~WindowContext() = default;

int32_t WindowContext::Rank() const { return impl_->rank_; }
int32_t WindowContext::NRank() const { return impl_->nrank_; }
int64_t WindowContext::DataPtr() const { return reinterpret_cast<int64_t>(impl_->windowDevPtr_); }
bool WindowContext::SupportsMultiIssuer() const { return impl_->multiIssuer_; }
bool WindowContext::Connected() const { return impl_->state_ == Impl::State::Connected; }

void WindowContext::RegisterMemory(int64_t tensorPtr, int64_t size, const std::string& name, int32_t protocol)
{
    impl_->RegisterMemory(tensorPtr, size, name, protocol);
}

void WindowContext::Connect() { impl_->Connect(); }

std::string WindowContext::ToString() const { return impl_->ToString(); }
void WindowContext::Close() { impl_->Close(); }
} // namespace npu::tile_fwk::Distributed

#endif // BUILD_WITH_CANN
