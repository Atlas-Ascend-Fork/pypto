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
#include <pypto_assert.h>
// The descriptor is a plain POD shared with the host writer; the helpers below run on
// the vector core only.
#include <tilefwk/distribute_context.h>

#if defined(__DAV_VEC__)

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "kernel_operator_sys_var_intf.h"
#include "jetty/hcomm_jetty.h"
#define PYPTO_DIST_HAVE_JETTY 1
#endif

namespace pypto::dist {
constexpr int32_t kControlRegion = 0;

__aicore__ inline __gm__ RegionEntry* region_entry(__gm__ Window* win, int32_t region)
{
    pypto_assert(region >= 0 && region < win->num_regions, "dist: region %d out of range [0, %d)\n", region,
                 win->num_regions);
    return reinterpret_cast<__gm__ RegionEntry*>(win->region_table) + region;
}

__aicore__ inline int32_t region_of(__gm__ Window* win, int64_t local_base)
{
    __gm__ RegionEntry* table = reinterpret_cast<__gm__ RegionEntry*>(win->region_table);
    for (int32_t i = 0; i < win->num_regions; ++i) {
        if (table[i].remote_ptrs[win->rank] == local_base) {
            return i;
        }
    }
    pypto_panic("dist: memory 0x%llx is not a registered region of this window\n",
                static_cast<unsigned long long>(local_base));
    return -1;
}

template <typename T>
__aicore__ inline __gm__ T* remote_ptr(__gm__ Window* win, int32_t region, int32_t peer)
{
    pypto_assert(peer >= 0 && peer < win->nrank, "dist: peer %d out of range [0, %d)\n", peer, win->nrank);
    return reinterpret_cast<__gm__ T*>(region_entry(win, region)->remote_ptrs[peer]);
}

__aicore__ inline __gm__ uint8_t* remote_control_ptr(__gm__ Window* win, int32_t peer)
{
    return remote_ptr<uint8_t>(win, kControlRegion, peer);
}

__aicore__ inline int64_t signal_slots(__gm__ Window* win)
{
    return region_entry(win, kControlRegion)->size / kL2CacheLineSize - 1;
}

// Address of signal slot `slot` on `peer`; slot 0 of the control region is the Barrier.
__aicore__ inline __gm__ uint64_t* signal_addr(__gm__ Window* win, int32_t peer, int64_t slot)
{
    return reinterpret_cast<__gm__ uint64_t*>(remote_control_ptr(win, peer) + (slot + 1) * kL2CacheLineSize);
}

__simt_vf__ __launch_bounds__(kNumMaxRanks) inline void signal(__gm__ Window* win, int64_t slot_off, int32_t delta)
{
    int32_t peer = static_cast<int32_t>(threadIdx.x);
    if (peer < win->nrank) {
        __gm__ RegionEntry* entry = reinterpret_cast<__gm__ RegionEntry*>(win->region_table) + kControlRegion;
        atomicAdd(reinterpret_cast<__gm__ int32_t*>(entry->remote_ptrs[peer] + slot_off), delta);
    }
}

template <bool kAtomicAdd>
__simt_vf__ __launch_bounds__(kNumMaxRanks) inline void notify_all_signal(__gm__ Window* win, int64_t slot_off,
                                                                          int64_t val)
{
    int32_t peer = static_cast<int32_t>(threadIdx.x);
    if (peer < win->nrank) {
        __gm__ RegionEntry* entry = reinterpret_cast<__gm__ RegionEntry*>(win->region_table) + kControlRegion;
        auto* addr = reinterpret_cast<__gm__ uint64_t*>(entry->remote_ptrs[peer] + slot_off);
        if constexpr (kAtomicAdd) {
            (void)atomicAdd(reinterpret_cast<__gm__ int64_t*>(addr), val);
        } else {
            (void)atomicExch(addr, static_cast<uint64_t>(val));
        }
    }
}

__aicore__ inline void cpu_relax()
{
#pragma unroll 1
    for (int32_t i = 0; i < 300; ++i) {
        asm volatile("nop");
    }
}

template <int32_t kNotifyOp, bool kFlushStore = true>
__aicore__ inline void notify(__gm__ Window* win, int64_t slot, int64_t val, int32_t peer)
{
    static_assert(kNotifyOp == 0 || kNotifyOp == 1, "notify_op must be 0 (SET) or 1 (ATOMIC_ADD)");
    pypto_assert(slot >= 0 && slot < signal_slots(win), "dist.notify: slot %lld out of range\n",
                 static_cast<long long>(slot));
    if constexpr (kFlushStore) {
        // Drain this AIV's outstanding MTE3 stores into the scalar pipe before the
        // signal write. Event id 0 is hardcoded — do not keep an MTE3 event 0 in flight.
        set_flag(PIPE_MTE3, PIPE_S, 0);
        wait_flag(PIPE_MTE3, PIPE_S, 0);
    }

    // Flush the store buffer so prior stores are visible before the signal write.
    dsb(DSB_DDR);

    __gm__ uint64_t* addr = signal_addr(win, peer, slot);
    if constexpr (kNotifyOp != 0) {
        (void)atomicAdd(reinterpret_cast<__gm__ int64_t*>(addr), val);
    } else {
        (void)atomicExch(addr, static_cast<uint64_t>(val));
    }
}

template <int32_t kCmp>
__aicore__ inline constexpr bool cmp(int64_t sig, int64_t val)
{
    static_assert(kCmp >= 0 && kCmp <= 5, "cmp must be a CmpMode value (0..5)");
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

// The wait always polls the LOCAL window's slot (LD_DEV bypasses the DCache).
template <int32_t kCmp>
__aicore__ inline void wait(__gm__ Window* win, int64_t slot, int64_t val, int64_t timeout_cycles)
{
    static_assert(kCmp >= 0 && kCmp <= 5, "cmp must be a CmpMode value (0..5)");
    pypto_assert(slot >= 0 && slot < signal_slots(win), "dist.wait: slot %lld out of range\n",
                 static_cast<long long>(slot));
    __gm__ uint64_t* addr = signal_addr(win, win->rank, slot);
    const int64_t t0 = get_sys_cnt();
    while (true) {
        const int64_t sig = static_cast<int64_t>(ld_dev(addr, 0));
        if (cmp<kCmp>(sig, val)) {
            break;
        }
        if (timeout_cycles > 0 && get_sys_cnt() - t0 >= timeout_cycles) {
            pypto_panic("dist.wait timeout: slot %lld, signal %lld, val %lld, cmp %d, elapsed %lld cycles, timeout "
                        "%lld cycles\n",
                        static_cast<long long>(slot), static_cast<long long>(sig), static_cast<long long>(val), kCmp,
                        static_cast<long long>(get_sys_cnt() - t0), static_cast<long long>(timeout_cycles));
        }
        cpu_relax();
    }

    // Enforce ordering between the wait and subsequent instructions.
    dsb(DSB_DDR);
}

template <int32_t kNotifyOp, bool kFlushStore = true>
__aicore__ inline void notify_all(__gm__ Window* win, int64_t slot, int64_t val)
{
    static_assert(kNotifyOp == 0 || kNotifyOp == 1, "notify_op must be 0 (SET) or 1 (ATOMIC_ADD)");
    pypto_assert(slot >= 0 && slot < signal_slots(win), "dist.notify_all: slot %lld out of range\n",
                 static_cast<long long>(slot));
    if constexpr (kFlushStore) {
        // Drain this AIV's outstanding MTE3 stores into the vector pipe before the
        // fan-out. Event id 0 is hardcoded — do not keep an MTE3->V event 0 in flight.
        set_flag(PIPE_MTE3, PIPE_V, 0);
        wait_flag(PIPE_MTE3, PIPE_V, 0);
    }

    // Flush the store buffer so prior stores are visible before the signals.
    dsb(DSB_DDR);

    const int64_t slot_off = (slot + 1) * kL2CacheLineSize;
    cce::async_invoke<notify_all_signal<kNotifyOp == 1>>(cce::dim3{kNumMaxRanks, 1, 1}, win, slot_off, val);
}

template <int64_t kTimeoutCycles, bool kSyncLocal = true, bool kFlushStore = true>
__aicore__ inline void barrier(__gm__ Window* win)
{
    if constexpr (kFlushStore) {
        // Store flush: drain this AIV's outstanding MTE3 stores (e.g. a preceding dist.put
        // TSTORE) into the vector pipe before the arrival exchange. Event id 0 is hardcoded
        // — the barrier owns the MTE3->V event-0 slot, so a kernel must not keep its own
        // MTE3->V event 0 in flight across a barrier call.
        set_flag(PIPE_MTE3, PIPE_V, 0);
        wait_flag(PIPE_MTE3, PIPE_V, 0);
    }

    if constexpr (kSyncLocal) {
        // Local AIV-core exchange only — never between ranks; the cross-rank ordering
        // comes solely from the software signal exchange in the leader section below.
        SYNCALL<SyncCoreType::AIVOnly>();
    }

    if (get_block_idx() * get_subblockdim() + get_subblockid() == 0) {
        __gm__ Barrier* bar = reinterpret_cast<__gm__ Barrier*>(remote_control_ptr(win, win->rank));
        __gm__ uint64_t* counter = reinterpret_cast<__gm__ uint64_t*>(&bar->barrier_counter);
        const int64_t cnt = static_cast<int64_t>(ld_dev(counter, 0));
        int32_t phase = static_cast<int32_t>(cnt & 1);
        int32_t sign = static_cast<int32_t>((cnt >> 1) & 1);
        int32_t delta = sign ? -1 : 1;
        int32_t target = sign ? 0 : win->nrank;
        st_dev(static_cast<uint64_t>(cnt + 1), counter, 0);

        __gm__ uint32_t* sig_addr = reinterpret_cast<__gm__ uint32_t*>(&bar->barrier_signals[phase]);
        const int64_t slot_off = reinterpret_cast<int64_t>(sig_addr) - reinterpret_cast<int64_t>(bar);
        cce::async_invoke<signal>(cce::dim3{kNumMaxRanks, 1, 1}, win, slot_off, delta);

        const int64_t t0 = get_sys_cnt();
        while (true) {
            int32_t sig = static_cast<int32_t>(ld_dev(sig_addr, 0));
            if (sig == target) {
                break;
            }
            if (kTimeoutCycles > 0 && get_sys_cnt() - t0 >= kTimeoutCycles) {
                pypto_panic("dist.barrier timeout: nrank %d, signal %d, target %d, phase %d, counter %lld\n",
                            win->nrank, sig, target, phase, static_cast<long long>(cnt));
            }
            cpu_relax();
        }
    }

    if constexpr (kSyncLocal) {
        SYNCALL<SyncCoreType::AIVOnly>();
    }
}

template <typename T>
__aicore__ inline constexpr bool atomic_supported()
{
    return std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, float> ||
           std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>;
}

template <typename T>
__aicore__ inline __gm__ T* atomic_addr(__gm__ Window* win, int32_t region, int32_t peer, int64_t base_bytes,
                                        int64_t elem_offset)
{
    static_assert(atomic_supported<T>(), "dist atomics support int32_t/uint32_t/float/int64_t/uint64_t only");
    auto* base = reinterpret_cast<__gm__ uint8_t*>(remote_ptr<T>(win, region, peer)) + base_bytes;
    return reinterpret_cast<__gm__ T*>(base) + elem_offset;
}

template <typename T>
__aicore__ inline T atomic_add(__gm__ Window* win, int32_t region, int32_t peer, int64_t base_bytes,
                               int64_t elem_offset, T val)
{
    return atomicAdd(atomic_addr<T>(win, region, peer, base_bytes, elem_offset), val);
}

template <typename T>
__aicore__ inline T atomic_exchange(__gm__ Window* win, int32_t region, int32_t peer, int64_t base_bytes,
                                    int64_t elem_offset, T val)
{
    return atomicExch(atomic_addr<T>(win, region, peer, base_bytes, elem_offset), val);
}

template <typename T>
__aicore__ inline T atomic_cas(__gm__ Window* win, int32_t region, int32_t peer, int64_t base_bytes,
                               int64_t elem_offset, T expect, T desired)
{
    return atomicCAS(atomic_addr<T>(win, region, peer, base_bytes, elem_offset), expect, desired);
}

#if defined(PYPTO_DIST_HAVE_JETTY)
struct Channel {
    __gm__ Window* win_;
    uint32_t jetty_idx_;
    AscendC::HcommJetty jetty_;
    AscendC::HcommPeer peer_;

    __aicore__ inline Channel(__gm__ Window* w, __ubuf__ uint8_t* ubuf, int32_t peer)
        : win_(w),
          jetty_idx_(JettyOf(w, peer)),
          jetty_(JettyInfo(ubuf), UbufSqe(ubuf), reinterpret_cast<GM_ADDR>(w->jetty_ptrs), jetty_idx_),
          peer_(JettyPeerInfo(ubuf), reinterpret_cast<GM_ADDR>(win_->jetty_ptrs), jetty_idx_)

    {
        pypto_assert(peer != -1 || win_->multi_issuer != 0,
                     "dist.channel: window has no per-AIV jetty (multi_issuer == 0); pass peer=\n");
    }

    template <int64_t kTimeoutCycles, bool kSubmit>
    __aicore__ inline void Write(int32_t region, int64_t base_bytes, __gm__ uint8_t* src, int64_t num_bytes)
    {
        GM_ADDR dst = RemoteOf(region) + base_bytes;
        int32_t ret = jetty_.Write<kTimeoutCycles, false>(peer_, dst, src, static_cast<uint64_t>(num_bytes));
        pypto_assert(ret == AscendC::HCOMM_SUCCESS, "dist.channel.write failed, ret %d\n", ret);
        if constexpr (kSubmit) {
            Submit();
        }
    }

    template <int64_t kTimeoutCycles, typename T, bool kSubmit>
    __aicore__ inline void WriteValue(int32_t region, int64_t base_bytes, T value)
    {
        GM_ADDR dst = RemoteOf(region) + base_bytes;
        int32_t ret = jetty_.WriteValue<kTimeoutCycles, T, false>(peer_, dst, value);
        pypto_assert(ret == AscendC::HCOMM_SUCCESS, "dist.channel.write_value failed, ret %d\n", ret);
        if constexpr (kSubmit) {
            Submit();
        }
    }

    template <int64_t kTimeoutCycles>
    __aicore__ inline void WriteNotify(int32_t region, int64_t base_bytes, __gm__ uint8_t* src, int64_t num_bytes,
                                       int64_t signal_slot, int64_t val)
    {
        pypto_assert(signal_slot >= 0 && signal_slot < signal_slots(win_),
                     "dist.channel.write_notify: slot %lld out of range\n", static_cast<long long>(signal_slot));

        GM_ADDR dst = RemoteOf(region) + base_bytes;
        GM_ADDR notify_addr = RemoteOf(kControlRegion) + (signal_slot + 1) * kL2CacheLineSize;
        int32_t ret = jetty_.WriteWithNotify<kTimeoutCycles>(peer_, dst, src, static_cast<uint64_t>(num_bytes),
                                                             notify_addr, static_cast<uint64_t>(val));
        pypto_assert(ret == AscendC::HCOMM_SUCCESS, "dist.channel.write_notify failed, ret %d\n", ret);
        dsb(DSB_DDR);
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
        int32_t ret = jetty_.Drain<kTimeoutCycles>();
        pypto_assert(ret == AscendC::HCOMM_SUCCESS, "dist.channel.drain failed, ret %d\n", ret);
    }

    __aicore__ inline bool IsExclusive() const { return win_->multi_issuer != 0; }

private:
    __aicore__ inline static int32_t JettyOf(__gm__ Window* win, int32_t peer)
    {
        if (peer >= 0) {
            return peer < win->rank ? peer : peer - 1;
        } else {
            return get_block_idx() * get_subblockdim() + get_subblockid();
        }
    }

    __aicore__ inline static GM_ADDR RemoteOf(__gm__ Window* win, int32_t peer, int32_t region)
    {
        uint64_t entry = reinterpret_cast<__gm__ uint64_t*>(win->jetty_ptrs)[peer];
        pypto_assert(entry != 0, "dist.channel: jetty table entry %d is null\n", peer);
        auto* channel = reinterpret_cast<__gm__ AscendC::ChannelEntity*>(entry);
        int32_t urma_index = region_entry(win, region)->urma_index;
        pypto_assert(urma_index >= 0 && urma_index < (int32_t)channel->remoteBufferNum,
                     "dist.channel: urma_index %d out of range\n", urma_index);
        auto* remote_buf = channel->remoteBufferAddr + urma_index;
        return reinterpret_cast<GM_ADDR>(remote_buf->bufferInfo.rma.addr);
    }

    __aicore__ inline GM_ADDR RemoteOf(int32_t region) const { return RemoteOf(win_, jetty_idx_, region); }

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
