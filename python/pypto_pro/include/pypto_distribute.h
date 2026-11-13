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
#if defined(__DAV_VEC__)
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "kernel_operator_sys_var_intf.h"
#include "jetty/hcomm_jetty.h"
#define PYPTO_DIST_HAVE_JETTY 1
#endif
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

template <typename T, int32_t kNotifyOp, bool kFlushStore = true>
__aicore__ inline void Notify(__gm__ T* addr, int64_t val)
{
    static_assert(std::is_integral<T>::value && (sizeof(T) == 4 || sizeof(T) == 8),
                  "signal words must be 32/64-bit integers");
    static_assert(kNotifyOp == 0 || kNotifyOp == 1, "notify_op must be 0 (SET) or 1 (ATOMIC_ADD)");
    if constexpr (kFlushStore) {
        // Drain this AIV's outstanding MTE3 stores into the scalar pipe before the
        // signal write. Event id 0 is hardcoded — do not keep an MTE3 event 0 in flight.
        set_flag(PIPE_MTE3, PIPE_S, 0);
        wait_flag(PIPE_MTE3, PIPE_S, 0);
    }

    // Flush the store buffer so prior stores are visible before the signal write.
    dsb(DSB_DDR);

    if constexpr (kNotifyOp != 0) {
        (void)atomicAdd(addr, static_cast<T>(val));
    } else {
        (void)atomicExch(addr, static_cast<T>(val));
    }
}

template <int32_t kCmp>
__aicore__ inline constexpr bool Cmp(int64_t sig, int64_t val)
{
    static_assert(kCmp >= 0 && kCmp <= 5, "Cmp must be a CmpMode value (0..5)");
    if constexpr (kCmp == 0) {
        return sig == val;
    } else if constexpr (kCmp == 1) {
        return sig != val;
    } else if constexpr (kCmp == 2) {
        return sig < val;
    } else if constexpr (kCmp == 3) {
        return sig <= val;
    } else if constexpr (kCmp == 4) {
        return sig > val;
    } else {
        return sig >= val;
    }
}

template <typename T, int32_t kCmp>
__aicore__ inline void Wait(__gm__ T* addr, int64_t val, int64_t timeout_cycles)
{
    static_assert(std::is_integral<T>::value && (sizeof(T) == 4 || sizeof(T) == 8),
                  "signal words must be 32/64-bit integers");
    static_assert(kCmp >= 0 && kCmp <= 5, "Cmp must be a CmpMode value (0..5)");
    const int64_t t0 = get_sys_cnt();
    while (true) {
        const int64_t sig = static_cast<int64_t>(LoadDev(addr));
        if (Cmp<kCmp>(sig, val)) {
            break;
        }
        if (timeout_cycles > 0 && get_sys_cnt() - t0 >= timeout_cycles) {
            pypto_panic("dist.Wait timeout: signal %lld, val %lld, Cmp %d, elapsed %lld cycles, timeout "
                        "%lld cycles\n",
                        static_cast<long long>(sig), static_cast<long long>(val), kCmp,
                        static_cast<long long>(get_sys_cnt() - t0), static_cast<long long>(timeout_cycles));
        }
        CpuRelax();
    }
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

// SQE word0 flag byte (bits 16-23). A batch body uses odr=1 with no fence/cqe;
// the last SQE carries odr=5, fence=1, cqe=1 (0b00101101 = 0x2D = URMA_DEFAULT_CFG),
// so one AdvanceSq commits the whole batch and accounts for exactly one CQE.
// RingDoorbell publishes the committed prefix.
constexpr uint64_t kDistSqeBodyFlags = 1ULL << 16U;
constexpr uint64_t kDistSqeFinalFlags = 0x2DULL << 16U;
constexpr uint64_t kDistSqeBase = kDistSqeBodyFlags | (1ULL << 28U) | (1ULL << 29U);
constexpr uint32_t kDistOpWrite = static_cast<uint32_t>(AscendC::HcommUrmaOpCode::WRITE);

// write_indexed builds one single-SGE 64B WQEBB per row directly in the jetty SQ
// in GM. One lane owns one SQE word, so eight lanes cover one WQEBB; each warp
// therefore writes four contiguous WQEBBs.
constexpr uint32_t kDistSimtWarps = 8U;
constexpr uint32_t kDistSimtThreads = 32U * kDistSimtWarps;
constexpr uint32_t kDistWqebbWords = kDistWqebbBytes / sizeof(uint64_t);
constexpr uint32_t kDistSimtRowsPerPass = kDistSimtThreads / kDistWqebbWords;
constexpr uint32_t kDistSimtMaxChunkWqebbs = 1024U;
constexpr uint32_t kDistNearFullThreshold = 10U;

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
            word |= static_cast<uint64_t>((seq + 1U) & 0xFFFFU);
            word |= (seq & args.sq_depth) == 0U ? (1ULL << 31U) : 0ULL;
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
    uint32_t jetty_idx_;
    __ubuf__ AscendC::HcommJettyInfo* info_;
    __ubuf__ uint8_t* stage_;
    AscendC::HcommJetty jetty_;
    AscendC::HcommPeer peer_;

    __aicore__ inline Channel(__gm__ Window* w, __ubuf__ uint8_t* ubuf, int32_t peer)
        : win_(w),
          jetty_idx_(JettyOf(w, peer)),
          info_(JettyInfo(ubuf)),
          stage_(UbufSqe(ubuf)),
          jetty_(info_, stage_, reinterpret_cast<GM_ADDR>(w->jetty_ptrs), jetty_idx_),
          peer_(JettyPeerInfo(ubuf), reinterpret_cast<GM_ADDR>(win_->jetty_ptrs), jetty_idx_)

    {
        pypto_assert(peer != -1 || win_->multi_issuer != 0,
                     "dist.channel: window has no per-AIV jetty (multi_issuer == 0); pass peer=\n");
        if (info_->numWqebbBytes != kDistWqebbBytes || info_->numCqeBytes == 0U || info_->sqDepth == 0U ||
            (info_->sqDepth & (info_->sqDepth - 1U)) != 0U || info_->cqDepth <= kDistNearFullThreshold) {
            pypto_panic("dist.channel: unsupported jetty (wqebb %u, cqe %u, sqDepth %u, cqDepth %u)\n",
                        info_->numWqebbBytes, info_->numCqeBytes, info_->sqDepth, info_->cqDepth);
        }
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

    template <int64_t kTimeoutCycles, bool kSubmit>
    __aicore__ inline void Write(GM_ADDR dst, __gm__ uint8_t* src, int64_t num_bytes)
    {
        constexpr int64_t kMaxChunkBytes = static_cast<int64_t>(kDistMaxTransferBytes);

        for (int64_t offset = 0; offset < num_bytes;) {
            int64_t bytes = num_bytes - offset;
            if (bytes > kMaxChunkBytes) {
                bytes = kMaxChunkBytes;
            }
            int32_t ret = jetty_.Write<kTimeoutCycles, false>(peer_, dst + offset, src + offset,
                                                              static_cast<uint64_t>(bytes));
            if (ret != AscendC::HCOMM_SUCCESS) {
                pypto_panic("dist.channel.write failed, ret %d\n", ret);
            }
            offset += bytes;
        }
        if constexpr (kSubmit) {
            Submit();
        }
    }

    template <int64_t kTimeoutCycles, typename T, bool kSubmit>
    __aicore__ inline void WriteValue(GM_ADDR dst, T value)
    {
        int32_t ret = jetty_.WriteValue<kTimeoutCycles, T, false>(peer_, dst, value);
        if (ret != AscendC::HCOMM_SUCCESS) {
            pypto_panic("dist.channel.write_value failed, ret %d\n", ret);
        }
        if constexpr (kSubmit) {
            Submit();
        }
    }

    template <int64_t kTimeoutCycles>
    __aicore__ inline void WriteNotify(GM_ADDR dst, __gm__ uint8_t* src, int64_t num_bytes, __gm__ uint8_t* notify_addr,
                                       uint64_t val)
    {
        int32_t ret = jetty_.WriteWithNotify<kTimeoutCycles>(peer_, dst, src, static_cast<uint64_t>(num_bytes),
                                                             notify_addr, val);
        if (ret != AscendC::HCOMM_SUCCESS) {
            pypto_panic("dist.channel.write_notify failed, ret %d\n", ret);
        }
        dsb(DSB_DDR);
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
                Write<kTimeoutCycles, kSubmit>(dst_base, src_base, static_cast<int64_t>(count * row_bytes));
                return;
            }
        }

        WriteIndexedSimt<IndexT, kHasSrcIndex, kHasDstIndex, kTimeoutCycles, kSubmit>(
            src_base, dst_base, src_indices, dst_indices, row_bytes, src_stride, dst_stride,
            static_cast<uint32_t>(count));
    }

    // Publish is a release fence: the doorbell store must be visible before the next op.
    __aicore__ inline void Submit()
    {
        jetty_.RingDoorbell();
        dsb(DSB_DDR);
    }

    template <int64_t kTimeoutCycles>
    __aicore__ inline void Drain()
    {
        PollCqUntil<kTimeoutCycles>(static_cast<uint32_t>(info_->packedHead >> 32U));
    }

    __aicore__ inline bool IsExclusive() const { return win_->multi_issuer != 0; }

private:
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
    __aicore__ inline bool TryReserve(uint32_t wqebbs, uint32_t completions)
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
                return true;
            }

            // Capacity pressure: publish earlier pending writes.
            if (!published) {
                Submit();
                published = true;
            }
            // No completion left to reclaim (equality), or the CQ counters disagree.
            pypto_assert(cqTail < expected, "dist.channel: SQ/CQ accounting inconsistent\n");

            if constexpr (kTimeoutCycles > 0) {
                if (get_sys_cnt() - start >= kTimeoutCycles) {
                    pypto_panic("dist.channel TryReserve timeout, wqebbs %u, usedSq %u, usedCq %u\n", wqebbs, usedSq,
                                usedCq);
                }
            }

            // Reclaim one completion, then recheck both capacities.
            PollCqUntil<kTimeoutCycles>(cqTail + 1U);
        }
    }

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
        uint64_t entry = reinterpret_cast<__gm__ uint64_t*>(win_->jetty_ptrs)[jetty_idx_];
        if (entry == 0) {
            pypto_panic("dist.channel: jetty table entry %u is null\n", jetty_idx_);
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
    __aicore__ inline void WriteIndexedSimt(GM_ADDR src_base, GM_ADDR dst_base, __ubuf__ IndexT* src_indices,
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

        hdr->words[0] = kDistSqeBase | (static_cast<uint64_t>(kDistOpWrite) << 40U);
        hdr->words[1] = peer_.peerInfo->tpIdNumSgesRemoteTokenId | (1ULL << 24U);
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
            if (!TryReserve<kTimeoutCycles>(chunk, 1U)) {
                pypto_panic("dist.channel.write_indexed: cannot reserve %u SQ WQEBBs\n", chunk);
            }
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
                Submit();
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
#endif // defined(PYPTO_DIST_HAVE_JETTY)

} // namespace pypto::dist

#endif // defined(__DAV_VEC__)
