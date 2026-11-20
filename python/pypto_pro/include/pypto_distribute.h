/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 * You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include <cstdint>
#include <type_traits>

// The CANN device headers must come before pypto_assert.h: both CANN and PyPTO's
// vendored debug headers define the same debug globals, and only the PyPTO copy defers
// to the CANN include guards, so CANN has to be seen first.
#if defined(__DAV_VEC__) && defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "kernel_operator_sys_var_intf.h"
#include "jetty/hcomm_jetty.h"
#define PYPTO_DIST_HAVE_JETTY 1
#endif

#include <pypto_assert.h>
// The descriptor is a plain POD shared with the host writer; the helpers below run on
// the vector core only.
#include <tilefwk/distribute_context.h>

#if defined(__DAV_VEC__)

namespace pypto::dist {
constexpr int32_t kControlRegion = 0;

__aicore__ inline __gm__ RegionEntry* GetRegionEntry(__gm__ Window* win, int32_t region)
{
    pypto_assert(region >= 0 && region < win->num_regions, "dist: region %d out of range [0, %d)\n", region,
                 win->num_regions);
    return reinterpret_cast<__gm__ RegionEntry*>(win->region_table) + region;
}

// Region containing `local_ptr`, with its byte offset from the region base. The pointer
// may sit anywhere inside a region, not only at its base; region sizes match across
// ranks, so the same offset locates the byte in a peer's copy.
__aicore__ inline int32_t RegionOfPtr(__gm__ Window* win, int64_t local_ptr, int64_t* offset)
{
    __gm__ RegionEntry* table = reinterpret_cast<__gm__ RegionEntry*>(win->region_table);
    for (int32_t i = 0; i < win->num_regions; ++i) {
        const int64_t base = table[i].remote_ptrs[win->rank];
        if (base != 0 && local_ptr >= base && local_ptr - base < table[i].size) {
            *offset = local_ptr - base;
            return i;
        }
    }
    pypto_panic("dist: memory 0x%llx is not inside a registered region of this window\n",
                static_cast<unsigned long long>(local_ptr));
    return -1;
}

// Byte pointer to `local_ptr`'s byte in the peer's UB_MEM copy of its region. The region
// and in-region offset are resolved from the pointer at runtime; only the UB_MEM plane has
// a direct peer-VA suitable for pl.load/pl.store.
__aicore__ inline __gm__ uint8_t* RemotePtr(__gm__ Window* win, int64_t local_ptr, int32_t peer)
{
    int64_t offset = 0;
    const int32_t region = RegionOfPtr(win, local_ptr, &offset);
    __gm__ RegionEntry* entry = GetRegionEntry(win, region);
    pypto_assert(peer >= 0 && peer < win->nrank, "dist: peer %d out of range [0, %d)\n", peer, win->nrank);
    const int64_t base = entry->remote_ptrs[peer];
    if (base == 0) {
        pypto_panic("dist.remote_ptr: region %d has no UB_MEM address for peer %d "
                    "(register it with Protocol.UB_MEM)\n",
                    region, peer);
    }
    return reinterpret_cast<__gm__ uint8_t*>(base + offset);
}

__aicore__ inline __gm__ uint8_t* RemoteControlPtr(__gm__ Window* win, int32_t peer)
{
    const int64_t localBase = GetRegionEntry(win, kControlRegion)->remote_ptrs[win->rank];
    return RemotePtr(win, localBase, peer);
}

__simt_vf__ __launch_bounds__(kNumMaxRanks) inline void BarrierSignal(__gm__ Window* win, int64_t slot_off,
                                                                      int32_t delta)
{
    int32_t peer = static_cast<int32_t>(threadIdx.x);
    if (peer < win->nrank) {
        __gm__ RegionEntry* entry = reinterpret_cast<__gm__ RegionEntry*>(win->region_table) + kControlRegion;
        atomicAdd(reinterpret_cast<__gm__ int32_t*>(entry->remote_ptrs[peer] + slot_off), delta);
    }
}

__aicore__ inline void CpuRelax()
{
#pragma unroll 1
    for (int32_t i = 0; i < 300; ++i) {
        asm volatile("nop");
    }
}

// Typed bypass-cache load/store of a device cell, addressed either by its GM byte
// address or by a typed __gm__ pointer.
template <typename T>
__aicore__ inline T LoadDev(uint64_t addr)
{
    return static_cast<T>(ld_dev(reinterpret_cast<__gm__ T*>(addr), 0));
}

template <typename T>
__aicore__ inline T LoadDev(__gm__ T* addr)
{
    return static_cast<T>(ld_dev(addr, 0));
}

template <typename T>
__aicore__ inline void StoreDev(uint64_t addr, T value)
{
    st_dev(value, reinterpret_cast<__gm__ T*>(addr), 0);
}

template <typename T>
__aicore__ inline void StoreDev(__gm__ T* addr, T value)
{
    st_dev(value, addr, 0);
}

template <int64_t kTimeoutCycles, bool kSyncLocal = true, bool kFlushStore = true>
__aicore__ inline void BarrierSync(__gm__ Window* win)
{
    if constexpr (kFlushStore) {
        // Store flush: drain this AIV's outstanding MTE3 stores (e.g. a preceding pl.store
        // TSTORE) into the vector pipe before the arrival exchange. Event id 0 is hardcoded
        // — the barrier owns the MTE3->V event-0 slot, so a kernel must not keep its own
        // MTE3->V event 0 in flight across a barrier call.
        set_flag(PIPE_MTE3, PIPE_V, 0);
        wait_flag(PIPE_MTE3, PIPE_V, 0);
    }

    if constexpr (kSyncLocal) {
        // Local AIV-core exchange only — never between ranks; the cross-rank ordering
        // comes solely from the software signal exchange in the leader section below.
        AscendC::SyncAll();
    }

    if (get_block_idx() * get_subblockdim() + get_subblockid() == 0) {
        __gm__ Barrier* bar = reinterpret_cast<__gm__ Barrier*>(RemoteControlPtr(win, win->rank));
        __gm__ uint64_t* counter = reinterpret_cast<__gm__ uint64_t*>(&bar->barrier_counter);
        const int64_t cnt = static_cast<int64_t>(LoadDev(counter));
        int32_t phase = static_cast<int32_t>(cnt & 1);
        int32_t sign = static_cast<int32_t>((cnt >> 1) & 1);
        int32_t delta = sign ? -1 : 1;
        int32_t target = sign ? 0 : win->nrank;
        StoreDev(counter, static_cast<uint64_t>(cnt + 1));

        __gm__ uint32_t* sig_addr = reinterpret_cast<__gm__ uint32_t*>(&bar->barrier_signals[phase]);
        const int64_t slot_off = reinterpret_cast<int64_t>(sig_addr) - reinterpret_cast<int64_t>(bar);
        cce::async_invoke<BarrierSignal>(cce::dim3{kNumMaxRanks, 1, 1}, win, slot_off, delta);

        const int64_t t0 = get_sys_cnt();
        while (true) {
            int32_t sig = static_cast<int32_t>(LoadDev(sig_addr));
            if (sig == target) {
                break;
            }
            if (kTimeoutCycles > 0 && get_sys_cnt() - t0 >= kTimeoutCycles) {
                pypto_panic("dist.barrier timeout: nrank %d, Signal %d, target %d, phase %d, counter %lld\n",
                            win->nrank, sig, target, phase, static_cast<long long>(cnt));
            }
            CpuRelax();
        }
    }

    if constexpr (kSyncLocal) {
        AscendC::SyncAll();
    }
}

#if defined(PYPTO_DIST_HAVE_JETTY)
// SQE wire layout: one 64B WQEBB holds a 48B HcommUrmaSqeCtx header (words 0-5)
// followed by one 16B HcommUrmaSgeCtx (words 6-7). Every indexed row is one such
// single-SGE WQEBB.
constexpr uint32_t kDistWqebbBytes = 64U;
constexpr uint64_t kDistMaxTransferBytes = 256ULL * 1024ULL * 1024ULL;

// SQE word0 flag byte (bits 16-23), mirroring hcomm's UrmaWqeEntry. The scalar write
// forms stage the caller's UrmaOption here; write_indexed's batch body uses odr=1 with
// no fence/completion and its last SQE carries 0x2D (URMA_DEFAULT_CFG), so one AdvanceSq
// commits the whole batch and accounts for exactly one CQE. RingDoorbell publishes the
// committed prefix.
constexpr uint32_t kDistSqeFlagShift = 16U;
constexpr uint32_t kDistDefaultSqeOption = 0x2DU; // URMA_DEFAULT_CFG
constexpr uint64_t kDistSqeBodyFlags = 1ULL << kDistSqeFlagShift;
constexpr uint64_t kDistSqeFinalFlags = static_cast<uint64_t>(kDistDefaultSqeOption) << kDistSqeFlagShift;
constexpr uint64_t kDistSqeFixedBits = (1ULL << 28U) | (1ULL << 29U); // tokenEn, rmtJettyType
constexpr uint64_t kDistSqeBase = kDistSqeBodyFlags | kDistSqeFixedBits;
// Word-0 fields outside the flag byte; the reduce descriptor lives in word 4.
constexpr uint32_t kDistSqeOpcodeShift = 40U;
constexpr uint32_t kDistSqeSgeNumShift = 24U;
constexpr uint32_t kDistSqeOwnerBit = 31U;
constexpr uint32_t kDistSqeSeqMask = 0xFFFFU;
constexpr uint32_t kDistSqeInlineSizeShift = 54U;
constexpr uint64_t kDistSqeInlineFlag = 0x40ULL; // inlineEn, bit 6 of the flag byte
constexpr uint32_t kDistWqe4ReduceDataTypeShift = 40U;
constexpr uint32_t kDistWqe4ReduceOpShift = 44U;
constexpr uint64_t kDistSqePublishFlags = 0x28ULL << kDistSqeFlagShift;

enum class UrmaOpCode : uint32_t {
    SEND = 0,
    SEND_WITH_IMM = 1,
    SEND_WITH_INVAL = 2,
    WRITE = 3,
    WRITE_WITH_IMM = 4,
    WRITE_WITH_NOTIFY = 5,
    READ = 6,
    ATOMIC_CAS = 7,
    ATOMIC_SWAP = 8,
    ATOMIC_STORE = 9,
    ATOMIC_LOAD = 10,
    ATOMIC_FADD = 11,
    ATOMIC_FSUB = 12,
    ATOMIC_FAND = 13,
    ATOMIC_FOR = 14,
    ATOMIC_FXOR = 15,
    NOP = 0x11,
};

// write_reduce rides the WRITE opcode with the UDF flag bit set and the reduce
// fields in SQE word 4 above the remote token value (bits [43:40] / [47:44], the
// vendor HcommUrmaSqeCtx layout): standalone-final flags 0x2D | 0x80 = 0xAD.
constexpr uint64_t kDistSqeUdfFlag = 0x80ULL << 16U;

template <uint32_t kSqeOption>
constexpr uint64_t kSqeFlags = static_cast<uint64_t>(kSqeOption) << kDistSqeFlagShift;

template <typename T>
constexpr uint64_t kInlineValueBits = (kDistSqeInlineFlag << kDistSqeFlagShift) |
                                      (static_cast<uint64_t>(sizeof(T)) << kDistSqeInlineSizeShift);

// Engine reduce data-type codes (vendor HCOMM_URMA_REDUCE_DATA_TYPE).
enum class ReduceDataType : uint32_t {
    INT8 = 0x0,
    INT16 = 0x1,
    INT32 = 0x2,
    UINT32 = 0x5,
    FP16 = 0x6,
    FP32 = 0x7,
    BF16 = 0x8,
    INVALID = 0xF,
};

template <typename T>
__aicore__ constexpr ReduceDataType GetReduceDataType()
{
    if constexpr (std::is_same_v<T, int8_t>) {
        return ReduceDataType::INT8;
    } else if constexpr (std::is_same_v<T, int16_t>) {
        return ReduceDataType::INT16;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return ReduceDataType::INT32;
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return ReduceDataType::UINT32;
    } else if constexpr (std::is_same_v<T, half>) {
        return ReduceDataType::FP16;
    } else if constexpr (std::is_same_v<T, float>) {
        return ReduceDataType::FP32;
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        return ReduceDataType::BF16;
    } else {
        return ReduceDataType::INVALID;
    }
}

template <typename T, uint32_t kReduceOp>
__aicore__ constexpr uint64_t ReduceDescriptorBits()
{
    return (static_cast<uint64_t>(GetReduceDataType<T>()) << kDistWqe4ReduceDataTypeShift) |
           (static_cast<uint64_t>(kReduceOp) << kDistWqe4ReduceOpShift);
}

// write_indexed builds one single-SGE 64B WQEBB per row directly in the jetty SQ
// in GM. One lane owns one SQE word, so eight lanes cover one WQEBB; each warp
// therefore writes four contiguous WQEBBs.
constexpr uint32_t kDistSimtWarps = 8U;
constexpr uint32_t kDistSimtThreads = 32U * kDistSimtWarps;
constexpr uint32_t kDistWqebbWords = kDistWqebbBytes / sizeof(uint64_t);
constexpr uint32_t kDistSimtRowsPerPass = kDistSimtThreads / kDistWqebbWords;
constexpr uint32_t kDistSimtMaxChunkWqebbs = 1024U;
constexpr uint32_t kDistNearFullThreshold = 10U;

// Cross-AIV channel lock word values (the vendor ChannelEntity protocol: free = 0,
// held = 1) and the contended-retry backoff.
constexpr uint32_t kDistLockFree = 0U;
constexpr uint32_t kDistLockHeld = 1U;
constexpr uint32_t kDistLockBackoff = 800U;

// --- WQE option and operand checks -------------------------------------------
// Flag byte: odr[2:0], fence[3], se[4], cqe[5]. A standalone post must carry cqe
// (drain accounts one completion per post); a staged append must not (the batch
// publish adds fence+cqe to the last staged WQE header).
template <uint32_t kSqeOption>
__aicore__ inline void CheckSqeFlagBits()
{
    static_assert((kSqeOption & ~0x2FU) == 0U, "WQE option sets unsupported flag bits");
    static_assert((kSqeOption & 0x3U) != 0x3U, "WQE option cannot set the relaxed and strong order bits together");
}

template <uint32_t kSqeOption>
__aicore__ inline void CheckSqeOption()
{
    CheckSqeFlagBits<kSqeOption>();
    static_assert((kSqeOption & 0x20U) != 0U, "dist.channel option must include cqe");
}

template <uint32_t kSqeOption>
__aicore__ inline void CheckStagedSqeOption()
{
    CheckSqeFlagBits<kSqeOption>();
    static_assert((kSqeOption & 0x20U) == 0U, "dist.buffer_channel option must not set cqe (submit owns completion)");
}

// Atomics move one 32/64-bit integer word; reductions accept the element types the
// wire's reduceDataType field can describe.
template <typename T>
__aicore__ inline void CheckAtomicWord()
{
    static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, int64_t> ||
                      std::is_same_v<T, uint64_t>,
                  "atomic operations support 32/64-bit integer words only");
}

template <typename T, uint32_t kReduceOp>
__aicore__ inline void CheckReduceType()
{
    static_assert(GetReduceDataType<T>() != ReduceDataType::INVALID,
                  "write_reduce supports int8/int16/int32/uint32/half/float/bfloat16 only");
    static_assert(kReduceOp == 0x8U || kReduceOp == 0x9U || kReduceOp == 0xAU,
                  "write_reduce supports ReduceOp.MAX/MIN/SUM only");
}

// A single WQE carries at most kDistMaxTransferBytes.
__aicore__ inline void CheckWqeMaxBytes(int64_t num_bytes)
{
    pypto_assert(num_bytes > 0 && num_bytes <= static_cast<int64_t>(kDistMaxTransferBytes),
                 "dist transfer: num_bytes %lld exceeds the 256 MiB single-WQE limit\n",
                 static_cast<long long>(num_bytes));
}

// Immutable per-channel peer metadata cache.
struct PeerInfo {
    uint32_t sqDepth;
    uint64_t peerWord1;
    uint64_t peerEid0;
    uint64_t peerEid1;
    uint64_t peerWord4;
};

// A notify descriptor occupies four words immediately before the data SGE.
__aicore__ inline void FillNotifyDescriptor(__ubuf__ uint64_t* words, const PeerInfo& peer, GM_ADDR addr,
                                            uint64_t value)
{
    words[0] = ((peer.peerWord1 >> 32U) & 0xFFFFFU) | ((peer.peerWord4 & 0xFFFFFFFFULL) << 32U);
    words[1] = reinterpret_cast<uint64_t>(addr);
    words[2] = value;
    words[3] = 0U;
}

// Words 0..5 of a one-SGE WQE from the channel's peer snapshot.
template <UrmaOpCode kOpCode, uint64_t kFlags, uint64_t kExtraWord0, bool kWithSeq, uint64_t kExtraWord4 = 0U>
__aicore__ inline void FillSqeWords(__ubuf__ uint64_t* w, uint32_t seq, const PeerInfo& peer, uint32_t sgeNum,
                                    GM_ADDR remote)
{
    uint64_t word0 = kFlags | kExtraWord0 | kDistSqeFixedBits | (static_cast<uint64_t>(kOpCode) << kDistSqeOpcodeShift);
    if constexpr (kWithSeq) {
        word0 |= static_cast<uint64_t>((seq + 1U) & kDistSqeSeqMask);
    }
    if ((seq & peer.sqDepth) == 0U) {
        word0 |= 1ULL << kDistSqeOwnerBit;
    }
    w[0] = word0;
    w[1] = peer.peerWord1 | (static_cast<uint64_t>(sgeNum) << kDistSqeSgeNumShift);
    w[2] = peer.peerEid0;
    w[3] = peer.peerEid1;
    w[4] = peer.peerWord4 | kExtraWord4;
    w[5] = reinterpret_cast<uint64_t>(remote);
}

// Scalar-prepared SQE words 0..4; the worker ORs the per-row owner bit and the
// final-batch flags into word 0 and fills word 7 (SGE source VA).
struct Wqebb {
    uint64_t words[kDistWqebbWords];
};

struct WriteIndexedParams {
    uint64_t src_base;
    uint64_t dst_base;
    uint64_t row_bytes;
    uint64_t src_stride;
    uint64_t dst_stride;
    uint32_t count;
    uint32_t sq_depth;
    uint32_t sq_seq;
};

static_assert(sizeof(Wqebb) <= kDistWqebbBytes, "SQE header must fit one WQEBB");

template <typename IndexT, bool kHasSrcIndex, bool kHasDstIndex>
__simt_vf__ __launch_bounds__(kDistSimtThreads) inline void WriteIndexedSimt(__gm__ uint64_t* sq,
                                                                             __ubuf__ const Wqebb* hdr,
                                                                             __ubuf__ const IndexT* src_indices,
                                                                             __ubuf__ const IndexT* dst_indices,
                                                                             WriteIndexedParams args)
{
    uint32_t tid = threadIdx.y * 32U + threadIdx.x;
    uint32_t lane = tid & (kDistWqebbWords - 1U);
    uint32_t group = tid / kDistWqebbWords;
    const uint64_t header = lane < 5U ? hdr->words[lane] : 0ULL;

    for (uint32_t row = group; row < args.count; row += kDistSimtRowsPerPass) {
        uint32_t seq = args.sq_seq + row;
        uint32_t phys = seq & (args.sq_depth - 1U);
        uint64_t word = header;

        if (lane == 0U) {
            // Header word 0 must have owner and final-only flags cleared.
            word |= static_cast<uint64_t>((seq + 1U) & kDistSqeSeqMask);
            word |= (seq & args.sq_depth) == 0U ? (1ULL << kDistSqeOwnerBit) : 0ULL;
            word |= row + 1U == args.count ? kDistSqeFinalFlags : 0ULL;
        } else if (lane == 5U) {
            uint64_t index = static_cast<uint64_t>(row);
            if constexpr (kHasDstIndex) {
                index = static_cast<uint64_t>(dst_indices[row]);
            }
            word = args.dst_base + index * args.dst_stride;
        } else if (lane == 6U) {
            word = args.row_bytes;
        } else if (lane == 7U) {
            uint64_t index = static_cast<uint64_t>(row);
            if constexpr (kHasSrcIndex) {
                index = static_cast<uint64_t>(src_indices[row]);
            }
            word = args.src_base + index * args.src_stride;
        }
        __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(
            sq + static_cast<uint64_t>(phys) * kDistWqebbWords + lane, word);
    }

    __threadfence();
}

struct Channel {
    __gm__ Window* win_;
    uint32_t jettyIdx_;
    __ubuf__ AscendC::HcommJettyInfo* info_;
    __ubuf__ uint8_t* stage_;
    AscendC::HcommJetty jetty_;
    AscendC::HcommPeer peer_;
    PeerInfo peerInfo_;

    __aicore__ inline Channel(__gm__ Window* w, __ubuf__ uint8_t* ubuf, int32_t peer)
        : win_(w),
          jettyIdx_(JettyOf(w, peer)),
          info_(JettyInfo(ubuf)),
          stage_(UbufSqe(ubuf)),
          jetty_(info_, stage_, reinterpret_cast<GM_ADDR>(w->jetty_ptrs), jettyIdx_),
          peer_(JettyPeerInfo(ubuf), reinterpret_cast<GM_ADDR>(win_->jetty_ptrs), jettyIdx_)
    {
        pypto_assert(peer != -1 || win_->multi_issuer != 0,
                     "dist.channel: window has no per-AIV jetty (multi_issuer == 0); pass peer=\n");
        if (info_->numWqebbBytes != kDistWqebbBytes || info_->numCqeBytes == 0U || info_->sqDepth == 0U ||
            (info_->sqDepth & (info_->sqDepth - 1U)) != 0U || info_->cqDepth <= kDistNearFullThreshold) {
            pypto_panic("dist.channel: unsupported jetty (wqebb %u, cqe %u, sqDepth %u, cqDepth %u)\n",
                        info_->numWqebbBytes, info_->numCqeBytes, info_->sqDepth, info_->cqDepth);
        }
        auto* peer_info = peer_.peerInfo;
        peerInfo_.sqDepth = info_->sqDepth;
        peerInfo_.peerWord1 = peer_info->tpIdNumSgesRemoteTokenId;
        peerInfo_.peerEid0 = peer_info->remoteEid[0];
        peerInfo_.peerEid1 = peer_info->remoteEid[1];
        peerInfo_.peerWord4 = peer_info->remoteTokenValueUdf;
    }

    // URMA-plane byte pointer to the peer's copy of the byte at `local_ptr` (anywhere inside
    // a region registered on this channel). The channel entity carries the exchanged local
    // and remote buffer descriptors in the same order, so the local base identifies the
    // region without a window-side index. Resolve once, then pass the result to Write/... .
    template <typename T>
    __aicore__ inline __gm__ T* RemoteAddr(int64_t local_ptr) const
    {
        return reinterpret_cast<__gm__ T*>(RemoteAddrOf(local_ptr));
    }

private:
    // SQE words 0..5 into stage_: the opcode, the standalone-final flag byte and the
    // remote address come from the caller; word 0 carries the per-post sequence number
    // and owner bit (same fields the write_indexed worker ORs per row), word 1 keeps
    // one SGE. Words 6/7 (SGE length, local VA) and any opcode payload stay the
    // caller's per-op fill. Declared before its template users: the CCE frontend
    // binds non-template names in member-template bodies at first parse.
    template <UrmaOpCode kOpCode, uint64_t kFinalFlags>
    __aicore__ inline void FillSqeHeader(GM_ADDR remote)
    {
        FillSqeWords<kOpCode, kFinalFlags, 0U, true>(reinterpret_cast<__ubuf__ uint64_t*>(stage_),
                                                     static_cast<uint32_t>(info_->packedHead), peerInfo_, 1U, remote);
    }

    // Wrap-aware two-segment UB->SQ copy (the vendor CopyWqeToSq shape). The caller
    // owns the head: the publish paths differ in when they advance it, but the copy
    // and its PIPE_S<->PIPE_MTE3 handshake are common. kWqebbBytes == 0 reads the
    // per-channel size, non-zero keeps the caller's compile-time constant.
    template <uint32_t kWqebbBytes>
    __aicore__ inline void CopyWqesToSq(uint32_t head, uint32_t num_wqebbs, __ubuf__ uint8_t* src)
    {
        const uint32_t wqebb_bytes = kWqebbBytes != 0U ? kWqebbBytes : info_->numWqebbBytes;
        const uint32_t sq_idx = head & (info_->sqDepth - 1U);
        const uint32_t first = num_wqebbs < info_->sqDepth - sq_idx ? num_wqebbs : info_->sqDepth - sq_idx;
        set_flag(PIPE_S, PIPE_MTE3, 0);
        wait_flag(PIPE_S, PIPE_MTE3, 0);
        asc_copy_ub2gm_align(
            reinterpret_cast<__gm__ uint8_t*>(info_->sqBaseAddr + static_cast<uint64_t>(wqebb_bytes) * sq_idx), src, 1,
            wqebb_bytes * first, asc_store_l2_cache_mode::NOTALLOC_CLEAN, 0, 0);
        if (first < num_wqebbs) {
            asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t*>(info_->sqBaseAddr),
                                 src + static_cast<uint64_t>(wqebb_bytes) * first, 1,
                                 wqebb_bytes * (num_wqebbs - first), asc_store_l2_cache_mode::NOTALLOC_CLEAN, 0, 0);
        }
        set_flag(PIPE_MTE3, PIPE_S, 0);
        wait_flag(PIPE_MTE3, PIPE_S, 0);
    }

    // The channel's cross-AIV lock word lives at the default CQ context's ubJfc.headAddr.
    // Resolve it the way hcomm does: cqContextAddr shares a cache line with the channel
    // counters, so read it bypassing the DCache — a cached entity field would hand back a
    // stale lock address before the lock is even acquired.
    __aicore__ inline __gm__ uint32_t* LockWord() const
    {
        const uint64_t entry = reinterpret_cast<__gm__ uint64_t*>(win_->jetty_ptrs)[jettyIdx_];
        pypto_assert(entry != 0U, "dist.channel: jetty table entry %u is null\n", jettyIdx_);
        auto* entity = reinterpret_cast<__gm__ AscendC::ChannelEntity*>(entry);
        __gm__ uint64_t* cqContextsAddrField = reinterpret_cast<__gm__ uint64_t*>(
            reinterpret_cast<__gm__ uint8_t*>(entity) + offsetof(AscendC::ChannelEntity, cqContextAddr));
        const uint64_t cqContextsAddr = AscendC::ReadGmBypassDCache<uint64_t>(cqContextsAddrField);
        auto* cqContext = reinterpret_cast<__gm__ AscendC::CqContext*>(cqContextsAddr);
        return reinterpret_cast<__gm__ uint32_t*>(cqContext->contextInfo.ubJfc.headAddr);
    }

    // Post a staged WQE to the SQ, optionally ringing the doorbell.
    // ReserveOrWait ensures SQ/CQ capacity (one CQE per WQE, regardless of WQEBBs)
    // and submits pending WQEs when space must be reclaimed, preventing deadlock
    // with submit=false. Copy the WQE from UB to SQ before advancing the head,
    // using PIPE_S <-> PIPE_MTE3 synchronization to ensure DMA completion.
    template <int64_t kTimeoutCycles, bool kSubmit>
    __aicore__ inline void PostSqe(uint32_t numWqebbs)
    {
        ReserveOrWait<kTimeoutCycles>(numWqebbs, 1U);
        const uint32_t head = static_cast<uint32_t>(info_->packedHead);
        CopySqeToSq(head, numWqebbs, stage_);
        jetty_.AdvanceSq(numWqebbs);
        if constexpr (kSubmit) {
            jetty_.RingDoorbell();
        }
    }

public:
    template <int64_t kTimeoutCycles, bool kSubmit, uint32_t kSqeOption>
    __aicore__ inline void Write(GM_ADDR dst, __gm__ uint8_t* src, int64_t num_bytes)
    {
        CheckWqeMaxBytes(num_bytes);
        BuildWriteSqe<kSqeOption, false>(dst, src, static_cast<uint64_t>(num_bytes), nullptr, 0);
        PostSqe<kTimeoutCycles, kSubmit>(1U);
    }

    template <int64_t kTimeoutCycles, typename T, bool kSubmit, uint32_t kSqeOption>
    __aicore__ inline void WriteValue(GM_ADDR dst, T value)
    {
        BuildWriteValueSqe<kSqeOption, T>(dst, value);
        PostSqe<kTimeoutCycles, kSubmit>(1U);
    }

    // Engine read: pull the remote source into the local GM destination (at most
    // 256 MiB; larger transfers must be split by the caller). The data is
    // engine-written, so it is readable only after Drain().
    template <int64_t kTimeoutCycles, bool kSubmit, uint32_t kSqeOption>
    __aicore__ inline void Read(GM_ADDR src, __gm__ uint8_t* dst, int64_t num_bytes)
    {
        CheckSqeOption<kSqeOption>();
        CheckWqeMaxBytes(num_bytes);
        FillSqeHeader<UrmaOpCode::READ, kSqeFlags<kSqeOption>>(src);
        auto* w = reinterpret_cast<__ubuf__ uint64_t*>(stage_);
        w[6] = static_cast<uint64_t>(num_bytes); // SGE length
        w[7] = reinterpret_cast<uint64_t>(dst);  // SGE local VA
        PostSqe<kTimeoutCycles, kSubmit>(1U);
    }

    // Engine-side reduction of the local GM source into the remote destination,
    // element-wise in T (at most 256 MiB; larger transfers must be split by the
    // caller). Concurrent reduces from several ranks into one target are
    // hardware-serialized; the result is not ordered against other engine ops —
    // Drain() first.
    template <int64_t kTimeoutCycles, typename T, uint32_t kReduceOp, bool kSubmit, uint32_t kSqeOption>
    __aicore__ inline void WriteReduce(GM_ADDR dst, __gm__ uint8_t* src, int64_t num_bytes)
    {
        CheckSqeOption<kSqeOption>();
        CheckReduceType<T, kReduceOp>();
        CheckWqeMaxBytes(num_bytes);
        // udfEn rides above the caller's option: the reduce fields in SQE word 4
        // only take effect when the UDF flag byte bit is set.
        FillSqeHeader<UrmaOpCode::WRITE, kDistSqeUdfFlag | kSqeFlags<kSqeOption>>(dst);
        auto* w = reinterpret_cast<__ubuf__ uint64_t*>(stage_);
        w[4] |= ReduceDescriptorBits<T, kReduceOp>();
        w[6] = static_cast<uint64_t>(num_bytes); // SGE length (bytes)
        w[7] = reinterpret_cast<uint64_t>(src);  // SGE local VA
        PostSqe<kTimeoutCycles, kSubmit>(1U);
    }

    template <int64_t kTimeoutCycles, uint32_t kSqeOption, bool kSubmit = true>
    __aicore__ inline void WriteNotify(GM_ADDR dst, __gm__ uint8_t* src, int64_t num_bytes, __gm__ uint8_t* notify_addr,
                                       uint64_t val)
    {
        CheckWqeMaxBytes(num_bytes);
        BuildWriteSqe<kSqeOption, true>(dst, src, static_cast<uint64_t>(num_bytes), notify_addr, val);
        PostSqe<kTimeoutCycles, kSubmit>(2U);
    }

    // Remote fetch-and-add of one aligned 32/64-bit integer word: the engine adds
    // `value` and writes the PREVIOUS remote value into the local GM `fetch` word
    // (two WQEBBs — the immediate rides the second; readable after Drain()).
    template <int64_t kTimeoutCycles, typename T, bool kSubmit, uint32_t kSqeOption>
    __aicore__ inline void AtomicFetchAdd(GM_ADDR dst, __gm__ uint8_t* fetch, T value)
    {
        CheckSqeOption<kSqeOption>();
        CheckAtomicWord<T>();
        if (reinterpret_cast<uint64_t>(dst) % sizeof(T) != 0U) {
            pypto_panic("dist.channel.atomic_fetch_add: destination 0x%llx is not %u-byte aligned\n",
                        static_cast<unsigned long long>(reinterpret_cast<uint64_t>(dst)),
                        static_cast<unsigned>(sizeof(T)));
        }
        FillSqeHeader<UrmaOpCode::ATOMIC_FADD, kSqeFlags<kSqeOption>>(dst);
        auto* w = reinterpret_cast<__ubuf__ uint64_t*>(stage_);
        w[6] = sizeof(T);                         // SGE length
        w[7] = reinterpret_cast<uint64_t>(fetch); // SGE local VA (fetch destination)
        *reinterpret_cast<__ubuf__ T*>(stage_ + kDistWqebbBytes) = value;
        for (uint32_t i = kDistWqebbBytes + sizeof(T); i < 2U * kDistWqebbBytes; i += 4U) {
            *reinterpret_cast<__ubuf__ uint32_t*>(stage_ + i) = 0U;
        }
        PostSqe<kTimeoutCycles, kSubmit>(2U);
    }

    // Remote compare-and-swap of one aligned word: swaps `desired` in when the
    // remote word equals `expected` and writes the PREVIOUS value into `fetch`
    // either way — success iff fetch == expected. Two WQEBBs; vendor payload order:
    // swap value at byte 64, compare value right after it.
    template <int64_t kTimeoutCycles, typename T, bool kSubmit, uint32_t kSqeOption>
    __aicore__ inline void AtomicCas(GM_ADDR dst, __gm__ uint8_t* fetch, T expected, T desired)
    {
        CheckSqeOption<kSqeOption>();
        CheckAtomicWord<T>();
        if (reinterpret_cast<uint64_t>(dst) % sizeof(T) != 0U) {
            pypto_panic("dist.channel.atomic_cas: destination 0x%llx is not %u-byte aligned\n",
                        static_cast<unsigned long long>(reinterpret_cast<uint64_t>(dst)),
                        static_cast<unsigned>(sizeof(T)));
        }
        FillSqeHeader<UrmaOpCode::ATOMIC_CAS, kSqeFlags<kSqeOption>>(dst);
        auto* w = reinterpret_cast<__ubuf__ uint64_t*>(stage_);
        w[6] = sizeof(T);                         // SGE length
        w[7] = reinterpret_cast<uint64_t>(fetch); // SGE local VA (fetch destination)
        *reinterpret_cast<__ubuf__ T*>(stage_ + kDistWqebbBytes) = desired;
        *reinterpret_cast<__ubuf__ T*>(stage_ + kDistWqebbBytes + sizeof(T)) = expected;
        for (uint32_t i = kDistWqebbBytes + 2U * sizeof(T); i < 2U * kDistWqebbBytes; i += 4U) {
            *reinterpret_cast<__ubuf__ uint32_t*>(stage_ + i) = 0U;
        }
        PostSqe<kTimeoutCycles, kSubmit>(2U);
    }

    template <typename IndexT, bool kHasSrcIndex, bool kHasDstIndex, int64_t kTimeoutCycles, bool kSubmit>
    __aicore__ inline void WriteIndexed(GM_ADDR dst_base, GM_ADDR src_base, __ubuf__ IndexT* src_indices,
                                        __ubuf__ IndexT* dst_indices, uint64_t row_bytes, uint64_t src_stride,
                                        uint64_t dst_stride, uint64_t count)
    {
        static_assert(std::is_same_v<IndexT, int32_t> || std::is_same_v<IndexT, int64_t> ||
                          std::is_same_v<IndexT, uint32_t> || std::is_same_v<IndexT, uint64_t>,
                      "write_indexed indices must be a 32/64-bit integer type");
        // count == 0 is a complete no-op: no WQE, no CQE, no doorbell.
        if (count == 0ULL) {
            return;
        }
        if (count > 0x7FFFFFFFULL) {
            pypto_panic("dist.channel.write_indexed count %llu exceeds INT32_MAX\n",
                        static_cast<unsigned long long>(count));
        }
        // The SIMT encoder publishes at most kDistSimtMaxChunkWqebbs rows per
        // AdvanceSq, so the ring must hold one full chunk.
        if (info_->sqDepth < kDistSimtMaxChunkWqebbs) {
            pypto_panic("dist.channel.write_indexed requires sqDepth >= %u WQEBBs, got %u\n", kDistSimtMaxChunkWqebbs,
                        info_->sqDepth);
        }

        if (row_bytes == 0ULL || (row_bytes % 32ULL) != 0ULL || row_bytes > kDistMaxTransferBytes) {
            pypto_panic("dist.channel.write_indexed invalid row_bytes %llu (must be a positive multiple of 32 "
                        "bytes, <= 256 MiB)\n",
                        static_cast<unsigned long long>(row_bytes));
        }
        if (src_stride < row_bytes || dst_stride < row_bytes) {
            pypto_panic("dist.channel.write_indexed strides (%llu, %llu) must be >= row_bytes %llu\n",
                        static_cast<unsigned long long>(src_stride), static_cast<unsigned long long>(dst_stride),
                        static_cast<unsigned long long>(row_bytes));
        }

        if constexpr (!kHasSrcIndex && !kHasDstIndex) {
            if (src_stride == row_bytes && dst_stride == row_bytes) {
                Write<kTimeoutCycles, kSubmit, kDistDefaultSqeOption>(dst_base, src_base,
                                                                      static_cast<int64_t>(count * row_bytes));
                return;
            }
        }

        PostIndexedChunks<IndexT, kHasSrcIndex, kHasDstIndex, kTimeoutCycles, kSubmit>(
            src_base, dst_base, src_indices, dst_indices, row_bytes, src_stride, dst_stride,
            static_cast<uint32_t>(count));
    }

    template <int64_t kTimeoutCycles>
    __aicore__ inline void Drain()
    {
        PollCqUntil<kTimeoutCycles>(static_cast<uint32_t>(info_->packedHead >> 32U));
    }

    // Cross-AIV exclusion over the channel's shared queue state. A multi-issuer channel
    // rides the calling AIV's own jetty, so there is nothing to exclude and both calls are
    // free. Otherwise the GM counters are authoritative while info_->packedHead is only a
    // per-AIV cache: Lock refreshes it after acquiring, Unlock publishes it before
    // releasing (the vendor ChannelEntity Lock/Unlock protocol).
    __aicore__ inline void Lock()
    {
        if (IsExclusive()) {
            return;
        }
        __gm__ uint32_t* lock = LockWord();
        while (atomicCAS(lock, kDistLockFree, kDistLockHeld) != kDistLockFree) {
            AscendC::Nop<kDistLockBackoff>(); // vendor backoff: reduce atomic contention
        }
        // from GM rather than from this AIV's stale copy (the vendor does the same).
        info_->packedHead = AscendC::ReadGmBypassDCache(reinterpret_cast<__gm__ uint64_t*>(info_->sqHeadAddr));
    }

    __aicore__ inline void Unlock()
    {
        if (IsExclusive()) {
            return;
        }
        // Hand the cached producer state (head + expected CQEs) over before the next AIV
        // can read it. The store bypasses the DCache, which is what the vendor's
        // CacheWriteThrough flush achieves for its cached counters; no extra fence.
        AscendC::WriteGmBypassDCache(reinterpret_cast<__gm__ uint64_t*>(info_->sqHeadAddr), info_->packedHead);
        (void)atomicCAS(LockWord(), kDistLockHeld, kDistLockFree);
    }

    __aicore__ inline bool IsExclusive() const { return win_->multi_issuer != 0; }

    // Narrow view for BufferChannel, which composes on this channel: it needs the head,
    // the SQ depth, the peer metadata and the head advance, not the transport objects.
    __aicore__ inline uint32_t PackedHead() const { return static_cast<uint32_t>(info_->packedHead); }
    __aicore__ inline void AdvanceSq(uint32_t num_wqebbs) { jetty_.AdvanceSq(num_wqebbs); }

private:
    // Stage the inline-value WQE (one WQEBB) with the caller's flag byte; inlineEn is
    // inherent to the form, so it is always set.
    template <uint32_t kSqeOption, typename T>
    __aicore__ inline void BuildWriteValueSqe(GM_ADDR dst, T value)
    {
        CheckSqeOption<kSqeOption>();
        auto* words = reinterpret_cast<__ubuf__ uint64_t*>(stage_);
        FillSqeWords<UrmaOpCode::WRITE, kSqeFlags<kSqeOption>, kInlineValueBits<T>, false>(
            words, static_cast<uint32_t>(info_->packedHead), peerInfo_, 0U, dst);
        *reinterpret_cast<__ubuf__ T*>(words + 6U) = value;
    }

    // Stage one write WQE (two WQEBBs with notify) in `stage_`, mirroring hcomm's
    // BuildWriteSqe but with the caller's flag byte instead of URMA_DEFAULT_CFG.
    template <uint32_t kSqeOption, bool kWithNotify>
    __aicore__ inline void BuildWriteSqe(GM_ADDR dst, __gm__ uint8_t* src, uint64_t num_bytes, GM_ADDR notify_addr,
                                         uint64_t notify_value)
    {
        CheckSqeOption<kSqeOption>();
        constexpr UrmaOpCode kOpCode = kWithNotify ? UrmaOpCode::WRITE_WITH_NOTIFY : UrmaOpCode::WRITE;

        __ubuf__ uint64_t* words = reinterpret_cast<__ubuf__ uint64_t*>(stage_);
        // SQE header: flag byte, fixed bits 28/29, owner, opcode; inlineMsgLen stays 0.
        FillSqeWords<kOpCode, kSqeFlags<kSqeOption>, 0U, false>(words, static_cast<uint32_t>(info_->packedHead),
                                                                peerInfo_, 1U, dst);
        __ubuf__ uint64_t* sge = words + 6U;
        if constexpr (kWithNotify) {
            FillNotifyDescriptor(sge, peerInfo_, notify_addr, notify_value);
            sge += 4U;
            // Clear the trailer of the second WQEBB
            for (uint32_t i = 12U; i < 2U * kDistWqebbWords; ++i) {
                words[i] = 0U;
            }
        }
        sge[0] = static_cast<uint32_t>(num_bytes);
        sge[1] = reinterpret_cast<uint64_t>(src);
    }

    // SQ/CQ publish and capacity primitives, shared with BufferChannel (which
    // composes on this channel instead of copying it).
public:
    // Wrap-aware two-segment UB -> SQ copy (the vendor CopyWqeToSq shape); the source
    // is the channel's single-WQE staging area or a buffer channel's staging buffer.
    __aicore__ inline void CopySqeToSq(uint32_t head, uint32_t num_wqebbs, __ubuf__ uint8_t* src)
    {
        CopyWqesToSq<0U>(head, num_wqebbs, src);
    }

    template <int64_t kTimeoutCycles>
    __aicore__ inline void PollCqUntil(uint32_t target)
    {
        uint32_t tail = LoadDev<uint32_t>(info_->cqTailAddr);
        uint32_t released = LoadDev<uint32_t>(info_->completionTailAddr);

        while (tail != target) {
            auto* cqe = reinterpret_cast<__gm__ uint32_t*>(
                info_->cqBaseAddr + static_cast<uint64_t>(tail & (info_->cqDepth - 1U)) * info_->numCqeBytes);

            uint32_t phase = (tail / info_->cqDepth) & 1U;
            int64_t start = get_sys_cnt();
            uint32_t header;

            do {
                header = LoadDev(cqe);
                if (((header >> 2U) & 1U) != phase) {
                    break;
                }
                if constexpr (kTimeoutCycles > 0) {
                    if (get_sys_cnt() - start >= kTimeoutCycles) {
                        pypto_panic("dist.channel PollCq timeout, tail %u, target %u\n", tail, target);
                    }
                }
                CpuRelax();
            } while (true);

            if ((header >> 16U) != 0U) {
                pypto_panic("dist.channel CQE failed, status %u, substatus %u\n", header >> 24U,
                            (header >> 16U) & 0xFFU);
            }

            // CQE word 1 carries the completed SQ WQEBB index; rebuild the free-running
            // release count from it, mirroring Hcomm SIMT PollCqEntry.
            uint32_t completed = LoadDev(cqe + 1) & (info_->sqDepth - 1U);
            released += (completed + info_->sqDepth - (released & (info_->sqDepth - 1U))) & (info_->sqDepth - 1U);
            ++released;
            ++tail;
        }

        StoreDev<uint32_t>(info_->cqTailAddr, tail);
        StoreDev<uint32_t>(info_->cqDoorbellAddr, tail & 0xFFFFFFU);
        StoreDev<uint32_t>(info_->completionTailAddr, released);
        dsb(DSB_DDR);
    }

    // Reserve SQ/CQ capacity for one indexed chunk.
    template <int64_t kTimeoutCycles>
    __aicore__ inline void ReserveOrWait(uint32_t wqebbs, uint32_t completions)
    {
        uint32_t sqDepth = info_->sqDepth;
        uint32_t cqLimit = info_->cqDepth - kDistNearFullThreshold - completions;

        uint32_t head = static_cast<uint32_t>(info_->packedHead);
        uint32_t expected = static_cast<uint32_t>(info_->packedHead >> 32U);
        int64_t start = get_sys_cnt();
        bool published = false;

        while (true) {
            uint32_t sqTail = LoadDev<uint32_t>(info_->completionTailAddr);
            uint32_t cqTail = LoadDev<uint32_t>(info_->cqTailAddr);
            uint32_t usedSq = head - sqTail;
            uint32_t usedCq = expected - cqTail;

            if (usedSq <= sqDepth && wqebbs <= sqDepth - usedSq && usedCq <= cqLimit) {
                return;
            }

            // Capacity pressure: publish earlier pending writes.
            if (!published) {
                jetty_.RingDoorbell();
                published = true;
            }
            // No completion left to reclaim (equality), or the CQ counters disagree.
            pypto_assert(cqTail < expected, "dist.channel: SQ/CQ accounting inconsistent\n");

            if constexpr (kTimeoutCycles > 0) {
                if (get_sys_cnt() - start >= kTimeoutCycles) {
                    pypto_panic("dist.channel ReserveOrWait timeout, wqebbs %u, usedSq %u, usedCq %u\n", wqebbs, usedSq,
                                usedCq);
                }
            }

            // Reclaim one completion, then recheck both capacities.
            PollCqUntil<kTimeoutCycles>(cqTail + 1U);
        }
    }

private:
    __aicore__ inline static uint32_t JettyOf(__gm__ Window* win, int32_t peer)
    {
        uint32_t idx = 0;
        if (peer >= 0) {
            pypto_assert(peer < win->nrank, "dist.channel: peer %d out of range [0, %d)\n", peer, win->nrank);
            idx = static_cast<uint32_t>(peer < win->rank ? peer : peer - 1);
        } else {
            idx = static_cast<uint32_t>(get_block_idx() * get_subblockdim() + get_subblockid());
        }
        pypto_assert(idx < static_cast<uint32_t>(win->num_jetties), "dist.channel: jetty %u out of range [0, %d)\n",
                     idx, win->num_jetties);
        return idx;
    }

    // URMA-plane address of the byte at `local_ptr`: find the channel buffer containing the
    // local address, then apply the same offset to its exchanged remote address.
    __aicore__ inline GM_ADDR RemoteAddrOf(int64_t local_ptr) const
    {
        uint64_t entry = reinterpret_cast<__gm__ uint64_t*>(win_->jetty_ptrs)[jettyIdx_];
        if (entry == 0) {
            pypto_panic("dist.channel: jetty table entry %u is null\n", jettyIdx_);
        }
        auto* channel = reinterpret_cast<__gm__ AscendC::ChannelEntity*>(entry);
        if (channel->localBufferAddr == nullptr || channel->remoteBufferAddr == nullptr) {
            pypto_panic("dist.channel.remote_ptr: channel has no buffer descriptors\n");
        }
        const uint32_t numBuffers = channel->localBufferNum < channel->remoteBufferNum ? channel->localBufferNum :
                                                                                         channel->remoteBufferNum;
        for (uint32_t i = 0; i < numBuffers; ++i) {
            const int64_t base = static_cast<int64_t>(channel->localBufferAddr[i].bufferInfo.rma.addr);
            const int64_t size = static_cast<int64_t>(channel->localBufferAddr[i].bufferInfo.rma.size);
            if (base != 0 && local_ptr >= base && local_ptr - base < size) {
                return reinterpret_cast<GM_ADDR>(channel->remoteBufferAddr[i].bufferInfo.rma.addr +
                                                 static_cast<uint64_t>(local_ptr - base));
            }
        }
        pypto_panic("dist.channel.remote_ptr: pointer 0x%llx is not in a region registered on this channel\n",
                    static_cast<unsigned long long>(local_ptr));
        return nullptr;
    }

    // Every index form: the worker writes the WQEs into the GM SQ, the scalar side
    // only fences and publishes one chunk (one AdvanceSq, one CQE) at a time. Each
    // row is a single-SGE 64B WQEBB; the worker applies the ring
    // mask per row, so a chunk may cross the physical wrap (but never exceeds sqDepth,
    // which would overwrite a slot twice inside one chunk).
    template <typename IndexT, bool kHasSrcIndex, bool kHasDstIndex, int64_t kTimeoutCycles, bool kSubmit>
    __aicore__ inline void PostIndexedChunks(GM_ADDR src_base, GM_ADDR dst_base, __ubuf__ IndexT* src_indices,
                                             __ubuf__ IndexT* dst_indices, uint64_t row_bytes, uint64_t src_stride,
                                             uint64_t dst_stride, uint32_t count)
    {
        auto* hdr = reinterpret_cast<__ubuf__ Wqebb*>(stage_);
        WriteIndexedParams args{};
        args.row_bytes = row_bytes;
        args.src_stride = src_stride;
        args.dst_stride = dst_stride;
        args.sq_depth = info_->sqDepth;
        uint64_t src_base_addr = reinterpret_cast<uint64_t>(src_base);
        uint64_t dst_base_addr = reinterpret_cast<uint64_t>(dst_base);
        __gm__ uint64_t* sq_base = reinterpret_cast<__gm__ uint64_t*>(info_->sqBaseAddr);

        if constexpr (kHasSrcIndex || kHasDstIndex) {
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        }

        hdr->words[0] = kDistSqeBase | (static_cast<uint64_t>(UrmaOpCode::WRITE) << kDistSqeOpcodeShift);
        hdr->words[1] = peer_.peerInfo->tpIdNumSgesRemoteTokenId | (1ULL << kDistSqeSgeNumShift);
        hdr->words[2] = peer_.peerInfo->remoteEid[0];
        hdr->words[3] = peer_.peerInfo->remoteEid[1];
        hdr->words[4] = peer_.peerInfo->remoteTokenValueUdf;
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);

        uint32_t done = 0;
        uint32_t chunk = kDistSimtMaxChunkWqebbs;
        while (done < count) {
            if (count - done < chunk) {
                chunk = count - done;
            }
            ReserveOrWait<kTimeoutCycles>(chunk, 1U);
            args.count = chunk;
            args.sq_seq = static_cast<uint32_t>(info_->packedHead);
            args.src_base = src_base_addr + (kHasSrcIndex ? 0ULL : static_cast<uint64_t>(done) * src_stride);
            args.dst_base = dst_base_addr + (kHasDstIndex ? 0ULL : static_cast<uint64_t>(done) * dst_stride);
            __ubuf__ IndexT* src_chunk = kHasSrcIndex ? src_indices + done : nullptr;
            __ubuf__ IndexT* dst_chunk = kHasDstIndex ? dst_indices + done : nullptr;

            cce::async_invoke<pypto::dist::WriteIndexedSimt<IndexT, kHasSrcIndex, kHasDstIndex>>(
                cce::dim3{32, kDistSimtWarps}, sq_base, hdr, src_chunk, dst_chunk, args);
            // The V-pipe marker is queued behind the worker, so this Wait fires when
            // every lane store has completed.
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

            jetty_.AdvanceSq(chunk);
            if constexpr (kSubmit) {
                jetty_.RingDoorbell();
            }
            done += chunk;
        }
    }

private:
    __aicore__ inline static __ubuf__ AscendC::HcommJettyPeerInfo* JettyPeerInfo(__ubuf__ uint8_t* base)
    {
        return reinterpret_cast<__ubuf__ AscendC::HcommJettyPeerInfo*>(base);
    }

    __aicore__ inline static __ubuf__ AscendC::HcommJettyInfo* JettyInfo(__ubuf__ uint8_t* base)
    {
        return reinterpret_cast<__ubuf__ AscendC::HcommJettyInfo*>(JettyPeerInfo(base) + 1);
    }

    __aicore__ inline static __ubuf__ uint8_t* UbufSqe(__ubuf__ uint8_t* base)
    {
        return reinterpret_cast<__ubuf__ uint8_t*>(JettyInfo(base) + 1);
    }
};

struct BufferChannel {
    Channel* ch_;
    __ubuf__ uint8_t* buf_;
    uint32_t slots_;    // capacity in WQEBBs (buf_bytes / 64)
    uint32_t pending_;  // staged WQEBBs since the last Submit
    uint32_t base_seq_; // free-running SQ seq of buf_ slot 0, checked at submit
    uint32_t last_hdr_; // slot of the last staged op's WQE header (atomics span 2 slots)

    // Capacity is a trap, not a flush: size the ring for the largest batch.
    __aicore__ inline BufferChannel(Channel* ch, __ubuf__ uint8_t* buf, uint32_t buf_bytes)
        : ch_(ch), buf_(buf), slots_(buf_bytes / kDistWqebbBytes), pending_(0U), base_seq_(0U), last_hdr_(0U)
    {
        if (ch == nullptr || buf == nullptr || buf_bytes < 2U * kDistWqebbBytes ||
            (buf_bytes % kDistWqebbBytes) != 0U) {
            pypto_panic("dist.buffer_channel: buf_bytes %u must be a 64-byte multiple of at least 128\n", buf_bytes);
        }
        if (slots_ > ch->info_->sqDepth) {
            pypto_panic("dist.buffer_channel: buf_bytes %u needs %u WQEBBs but the jetty SQ holds only %u\n", buf_bytes,
                        slots_, ch->info_->sqDepth);
        }
    }

private:
    // Capture the batch sequence once. Flush patches the last header, even when
    // a two-slot operation ends the batch with a payload.
    template <UrmaOpCode kOpCode, uint32_t kSqeOption, uint64_t kExtraWord0 = 0U, uint64_t kExtraWord4 = 0U>
    __aicore__ inline __ubuf__ uint64_t* BeginSqe(GM_ADDR remote, uint32_t extra_wqebbs, uint32_t sge_num = 1U)
    {
        CheckStagedSqeOption<kSqeOption>();
        // Keep the capacity trap inline: a pypto_panic call in this per-append path
        // costs several cycles. The message is for sanitizer builds only.
        pypto_assert(pending_ + extra_wqebbs <= slots_, "dist.buffer_channel: ring is full; submit first\n");
        if (pending_ + extra_wqebbs > slots_) {
            trap();
        }
        if (pending_ == 0U) {
            base_seq_ = ch_->PackedHead();
        }
        last_hdr_ = pending_;
        auto* words = reinterpret_cast<__ubuf__ uint64_t*>(buf_) + static_cast<uint64_t>(pending_) * kDistWqebbWords;
        FillSqeWords<kOpCode, kSqeFlags<kSqeOption>, kExtraWord0, true, kExtraWord4>(words, base_seq_ + pending_,
                                                                                     ch_->peerInfo_, sge_num, remote);
        return words;
    }

public:
    template <uint32_t kSqeOption>
    __aicore__ inline void Write(GM_ADDR dst, __gm__ uint8_t* src, int64_t num_bytes)
    {
        CheckWqeMaxBytes(num_bytes);
        auto* w = BeginSqe<UrmaOpCode::WRITE, kSqeOption>(dst, 1U);
        w[6] = static_cast<uint64_t>(num_bytes);
        w[7] = reinterpret_cast<uint64_t>(src);
        ++pending_;
    }

    template <uint32_t kSqeOption, typename T>
    __aicore__ inline void WriteValue(GM_ADDR dst, T value)
    {
        auto* w = BeginSqe<UrmaOpCode::WRITE, kSqeOption, kInlineValueBits<T>>(dst, 1U, 0U);
        *reinterpret_cast<__ubuf__ T*>(w + 6U) = value;
        ++pending_;
    }

    template <uint32_t kSqeOption>
    __aicore__ inline void WriteNotify(GM_ADDR dst, __gm__ uint8_t* src, int64_t num_bytes, __gm__ uint8_t* notify_addr,
                                       uint64_t val)
    {
        CheckWqeMaxBytes(num_bytes);
        auto* w = BeginSqe<UrmaOpCode::WRITE_WITH_NOTIFY, kSqeOption>(dst, 2U);
        FillNotifyDescriptor(w + 6U, ch_->peerInfo_, notify_addr, val);
        w[10] = static_cast<uint32_t>(num_bytes);
        w[11] = reinterpret_cast<uint64_t>(src);
        for (uint32_t i = 12U; i < 2U * kDistWqebbWords; ++i) {
            w[i] = 0U;
        }
        pending_ += 2U;
    }

    // Mark the final header for completion and copy the batch to the SQ. Submit rings the doorbell.
    template <int64_t kTimeoutCycles>
    __aicore__ inline void Flush()
    {
        if (pending_ == 0U) {
            return;
        }
        // Posting on the base channel would invalidate the staged sequence numbers.
        const uint32_t head = ch_->PackedHead();
        if (head != base_seq_) {
            pypto_panic("dist.buffer_channel: the base channel posted while %u WQEBBs were staged "
                        "(head %u, expected %u); do not interleave ch ops with staged appends\n",
                        pending_, head, base_seq_);
        }
        __ubuf__ uint64_t* last = reinterpret_cast<__ubuf__ uint64_t*>(buf_) +
                                  static_cast<uint64_t>(last_hdr_) * kDistWqebbWords;
        last[0] |= kDistSqePublishFlags;
        ch_->template ReserveOrWait<kTimeoutCycles>(pending_, 1U);
        ch_->CopySqeToSq(head, pending_, buf_);
        ch_->AdvanceSq(pending_);
        pending_ = 0U;
    }

    // Publish the staged batch — the vendor BatchCommit shape: one Flush (one SQ
    // reservation, one wrap-aware DMA, one AdvanceSq) plus one doorbell. Always rings,
    // so it also publishes an engine op posted earlier with submit=false.
    template <int64_t kTimeoutCycles>
    __aicore__ inline void Submit()
    {
        Flush<kTimeoutCycles>();
        ch_->jetty_.RingDoorbell();
    }

    // Same scope as the base channel's; traps if WQEBBs were staged and never submitted.
    template <int64_t kTimeoutCycles>
    __aicore__ inline void Drain()
    {
        if (pending_ != 0U) {
            pypto_panic("dist.buffer_channel: %u staged WQEBBs were never submitted before drain\n", pending_);
        }
        ch_->template Drain<kTimeoutCycles>();
    }

    // Engine-written read and atomic results are available after Drain.
    template <uint32_t kSqeOption>
    __aicore__ inline void ReadSqe(GM_ADDR src, __gm__ uint8_t* dst, uint64_t num_bytes)
    {
        auto* w = BeginSqe<UrmaOpCode::READ, kSqeOption>(src, 1U);
        w[6] = num_bytes;
        w[7] = reinterpret_cast<uint64_t>(dst);
        ++pending_;
    }

    // Stage one write-reduce WQEBB: the UDF flag byte rides above the caller's option,
    // the reduce fields sit in word 4 above the remote token (the vendor layout).
    template <uint32_t kSqeOption, typename T, uint32_t kReduceOp>
    __aicore__ inline void WriteReduceSqe(GM_ADDR dst, __gm__ uint8_t* src, uint64_t num_bytes)
    {
        CheckReduceType<T, kReduceOp>();
        auto* w = BeginSqe<UrmaOpCode::WRITE, kSqeOption, kDistSqeUdfFlag, ReduceDescriptorBits<T, kReduceOp>()>(dst,
                                                                                                                 1U);
        w[6] = num_bytes;
        w[7] = reinterpret_cast<uint64_t>(src);
        ++pending_;
    }

    // Stage one fetch-add (two WQEBBs: the header + the immediate payload the engine
    // writes the previous remote value into, zero-padded like the plain channel).
    template <uint32_t kSqeOption, typename T>
    __aicore__ inline void AtomicFetchAddSqe(GM_ADDR dst, __gm__ uint8_t* fetch, T value)
    {
        CheckAtomicWord<T>();
        if (reinterpret_cast<uint64_t>(dst) % sizeof(T) != 0U) {
            pypto_panic("dist.channel.atomic_fetch_add: destination 0x%llx is not %u-byte aligned\n",
                        static_cast<unsigned long long>(reinterpret_cast<uint64_t>(dst)),
                        static_cast<unsigned>(sizeof(T)));
        }
        auto* w = BeginSqe<UrmaOpCode::ATOMIC_FADD, kSqeOption>(dst, 2U);
        w[6] = sizeof(T);
        w[7] = reinterpret_cast<uint64_t>(fetch);
        auto* base = reinterpret_cast<__ubuf__ uint8_t*>(w);
        *reinterpret_cast<__ubuf__ T*>(base + kDistWqebbBytes) = value;
        uint32_t offset = kDistWqebbBytes + sizeof(T);
        if constexpr ((kDistWqebbBytes + sizeof(T)) % sizeof(uint64_t) != 0U) {
            *reinterpret_cast<__ubuf__ uint32_t*>(base + offset) = 0U;
            offset += sizeof(uint32_t);
        }
        for (; offset < 2U * kDistWqebbBytes; offset += sizeof(uint64_t)) {
            *reinterpret_cast<__ubuf__ uint64_t*>(base + offset) = 0U;
        }
        pending_ += 2U;
    }

    // Stage one compare-and-swap (two WQEBBs; vendor payload order: swap value at
    // byte 64, compare value right after it).
    template <uint32_t kSqeOption, typename T>
    __aicore__ inline void AtomicCasSqe(GM_ADDR dst, __gm__ uint8_t* fetch, T expected, T desired)
    {
        CheckAtomicWord<T>();
        if (reinterpret_cast<uint64_t>(dst) % sizeof(T) != 0U) {
            pypto_panic("dist.channel.atomic_cas: destination 0x%llx is not %u-byte aligned\n",
                        static_cast<unsigned long long>(reinterpret_cast<uint64_t>(dst)),
                        static_cast<unsigned>(sizeof(T)));
        }
        auto* w = BeginSqe<UrmaOpCode::ATOMIC_CAS, kSqeOption>(dst, 2U);
        w[6] = sizeof(T);
        w[7] = reinterpret_cast<uint64_t>(fetch);
        auto* base = reinterpret_cast<__ubuf__ uint8_t*>(w);
        *reinterpret_cast<__ubuf__ T*>(base + kDistWqebbBytes) = desired;
        *reinterpret_cast<__ubuf__ T*>(base + kDistWqebbBytes + sizeof(T)) = expected;
        uint32_t offset = kDistWqebbBytes + 2U * sizeof(T);
        if constexpr ((kDistWqebbBytes + 2U * sizeof(T)) % sizeof(uint64_t) != 0U) {
            *reinterpret_cast<__ubuf__ uint32_t*>(base + offset) = 0U;
            offset += sizeof(uint32_t);
        }
        for (; offset < 2U * kDistWqebbBytes; offset += sizeof(uint64_t)) {
            *reinterpret_cast<__ubuf__ uint64_t*>(base + offset) = 0U;
        }
        pending_ += 2U;
    }

    // Append an engine read (one WQEBB, at most 256 MiB) — staged like Write; the
    // fetched bytes are engine-written and readable only after Drain.
    template <uint32_t kSqeOption>
    __aicore__ inline void Read(GM_ADDR src, __gm__ uint8_t* dst, int64_t num_bytes)
    {
        CheckWqeMaxBytes(num_bytes);
        ReadSqe<kSqeOption>(src, dst, static_cast<uint64_t>(num_bytes));
    }

    // Append an engine-side reduction (one WQEBB, at most 256 MiB).
    template <uint32_t kSqeOption, typename T, uint32_t kReduceOp>
    __aicore__ inline void WriteReduce(GM_ADDR dst, __gm__ uint8_t* src, int64_t num_bytes)
    {
        CheckWqeMaxBytes(num_bytes);
        WriteReduceSqe<kSqeOption, T, kReduceOp>(dst, src, static_cast<uint64_t>(num_bytes));
    }

    template <uint32_t kSqeOption, typename T>
    __aicore__ inline void AtomicFetchAdd(GM_ADDR dst, __gm__ uint8_t* fetch, T value)
    {
        AtomicFetchAddSqe<kSqeOption, T>(dst, fetch, value);
    }

    template <uint32_t kSqeOption, typename T>
    __aicore__ inline void AtomicCas(GM_ADDR dst, __gm__ uint8_t* fetch, T expected, T desired)
    {
        AtomicCasSqe<kSqeOption, T>(dst, fetch, expected, desired);
    }
};
#endif // defined(PYPTO_DIST_HAVE_JETTY)

} // namespace pypto::dist

#endif // defined(__DAV_VEC__)
