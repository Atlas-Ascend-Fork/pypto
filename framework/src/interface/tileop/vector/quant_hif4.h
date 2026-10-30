/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TILEOP_VECTOR_QUANT_HIF4_H
#define TILEOP_VECTOR_QUANT_HIF4_H
#include "tilefwk/hif4_scratch.h"
#include <type_traits>

namespace hif4 {
// Arithmetic traverses the valid rows of one contiguous 2D tile linearly.
struct VfPlan {
    unsigned rows, cols, computeCols, fullBatches, tailElements;
    unsigned fullCols, tailCols, tailScaleLow, tailScaleHigh;
    __aicore__ inline VfPlan(unsigned validCols, const ScratchLayout& storage)
        : rows(storage.elements / storage.cols),
          cols(validCols),
          computeCols(storage.cols),
          fullBatches(storage.elements / 512),
          tailElements(storage.elements % 512),
          fullCols(storage.elements / 128),
          tailCols(storage.elements % 128),
          tailScaleLow(tailElements / 2 < 128 ? tailElements / 2 : 128),
          tailScaleHigh(tailElements > 256 ? tailElements / 2 - 128 : 0)
    {}
};

// Only valid rows are visited; invalid outer rows are never modified.
template <bool PAD>
__aicore__ inline __attribute__((always_inline)) void Pad(__ubuf__ bfloat16_t* src, const VfPlan& plan)
{
    if constexpr (PAD) {
        vector_u16 zero;
        vbr(zero, (uint16_t)0);
        unsigned count = plan.computeCols - plan.cols;
        for (uint16_t row = 0; row < (uint16_t)plan.rows; ++row) {
            for (uint16_t part = 0; part < (uint16_t)((count + 127) / 128); ++part) {
                unsigned remaining = count - (unsigned)part * 128;
                unsigned length = remaining < 128 ? remaining : 128;
                vector_align out;
                auto write = (__ubuf__ uint16_t*)(src + (unsigned)row * plan.computeCols + plan.cols +
                                                  (unsigned)part * 128);
                vstus(out, length, zero, write, POST_UPDATE);
                vstas(out, write, 0, POST_UPDATE);
            }
        }
        mem_bar(VST_VLD);
    }
}

// Fixed-size load/reduction paths: no shape tests survive template instantiation.
template <unsigned ELEMENTS, bool IN_PLACE, bool FULL = true>
__aicore__ inline __attribute__((always_inline)) void Max4Block(__ubuf__ bfloat16_t* src, __ubuf__ bfloat16_t* dst,
                                                                vector_u16 lane, vector_u16 absMask, vector_bool all,
                                                                unsigned n = ELEMENTS)
{
    unsigned active = n / 4;
    auto mask = plt_b16(active, POST_UPDATE);
    vector_bf16 x0, x1, x2, x3, maximum;
    if constexpr (ELEMENTS >= 256) {
        vlds(x0, x1, src, 0, DINTLV_B32);
        vand((vector_u16&)x0, (vector_u16&)x0, absMask, all);
        vand((vector_u16&)x1, (vector_u16&)x1, absMask, all);
        vmax(x0, x0, x1, all, MODE_ZEROING);
        if constexpr (ELEMENTS == 512) {
            vlds(x2, x3, src, 256, DINTLV_B32);
            vand((vector_u16&)x2, (vector_u16&)x2, absMask, all);
            vand((vector_u16&)x3, (vector_u16&)x3, absMask, all);
            vmax(x2, x2, x3, all, MODE_ZEROING);
            vdintlv(x0, x1, x0, x2);
        } else {
            vdintlv(x0, x1, x0, x0);
        }
        vmax(maximum, x0, x1, mask, MODE_ZEROING);
    } else if constexpr (ELEMENTS == 128) {
        vlds(x0, src, 0, NORM);
        vand((vector_u16&)x0, (vector_u16&)x0, absMask, all);
        vdintlv(x0, x1, x0, x0);
        vmax(x0, x0, x1, all, MODE_ZEROING);
        vdintlv(x0, x1, x0, x0);
        vmax(maximum, x0, x1, mask, MODE_ZEROING);
    } else {
        vector_u16 index, index1;
        vshls(index, lane, (uint16_t)2, all, MODE_ZEROING);
        vgather2(x0, src, index, mask);
        vadds(index1, index, (uint16_t)1, all, MODE_ZEROING);
        vgather2(x1, src, index1, mask);
        vadds(index1, index, (uint16_t)2, all, MODE_ZEROING);
        vgather2(x2, src, index1, mask);
        vadds(index1, index, (uint16_t)3, all, MODE_ZEROING);
        vgather2(x3, src, index1, mask);
        vand((vector_u16&)x0, (vector_u16&)x0, absMask, all);
        vand((vector_u16&)x1, (vector_u16&)x1, absMask, all);
        vand((vector_u16&)x2, (vector_u16&)x2, absMask, all);
        vand((vector_u16&)x3, (vector_u16&)x3, absMask, all);
        vmax(maximum, x0, x1, mask, MODE_ZEROING);
        vmax(x2, x2, x3, mask, MODE_ZEROING);
        vmax(maximum, maximum, x2, mask, MODE_ZEROING);
    }
    if constexpr (ELEMENTS == 512 && FULL) {
        vsts(maximum, dst, 0, NORM_B16, mask);
    } else {
        vector_align out;
        auto write = (__ubuf__ uint16_t*)dst;
        vstus(out, n / 4, (vector_u16&)maximum, write, POST_UPDATE);
        vstas(out, write, 0, POST_UPDATE);
    }
    if constexpr (IN_PLACE) {
        mem_bar(VST_VLD);
    }
}

template <unsigned TAIL>
__aicore__ inline __attribute__((always_inline)) void Max4(__ubuf__ bfloat16_t* src, __ubuf__ bfloat16_t* dst,
                                                           const VfPlan& plan, vector_u16 lane, vector_u16 absMask,
                                                           vector_bool all)
{
    for (uint16_t part = 0; part < (uint16_t)plan.fullBatches; ++part) {
        Max4Block<512, false>(src + (unsigned)part * 512, dst + (unsigned)part * 128, lane, absMask, all);
    }
    if constexpr (TAIL != 0) {
        unsigned base = plan.fullBatches * 512;
        // UB has read headroom. Mask only complete valid groups; never write headroom.
        Max4Block<512, false, false>(src + base, dst + base / 4, lane, absMask, all, TAIL);
    }
    mem_bar(VST_VLD);
}

template <bool FULL>
__aicore__ inline __attribute__((always_inline)) void Max8EaBlock(__ubuf__ uint8_t* scratch,
                                                                  const ScratchLayout& storage, unsigned batch,
                                                                  unsigned n, vector_u16 lane)
{
    unsigned base = batch * 512;
    unsigned micro = n / 4, sub = n / 8, groups = n / 64;
    auto pm = plt_b16(micro, POST_UPDATE);
    auto ps = plt_b16(sub, POST_UPDATE);
    auto pg = plt_b16(groups, POST_UPDATE);
    auto max4 = (__ubuf__ bfloat16_t*)scratch + base / 4;
    vector_bf16 m4, m8, ma, x0, x1;
    vlds(m4, max4, 0, NORM);
    vdintlv(x0, x1, m4, m4);
    vmax(m8, x0, x1, ps, MODE_ZEROING);
    vector_align maxOut;
    auto maxWrite = (__ubuf__ uint16_t*)(scratch + storage.max8Offset) + base / 8;
    vstus(maxOut, n / 8, (vector_u16&)m8, maxWrite, POST_UPDATE);
    vstas(maxOut, maxWrite, 0, POST_UPDATE);
    vcgmax((vector_u16&)ma, (vector_u16&)m4, pm);
    vmuls(ma, ma, (bfloat16_t)0.142578125f, pg, MODE_ZEROING);
    vector_u8 eaReg;
    vcvt_bf162e6m2((vector_f8e6m2&)eaReg, ma, pg, ROUND_R, PART_EVEN);
    vector_align eaOut;
    auto eaWrite = (__ubuf__ uint16_t*)(scratch + storage.eaOffset + batch * 32);
    vstus(eaOut, n / 64, (vector_u16&)eaReg, eaWrite, POST_UPDATE);
    vstas(eaOut, eaWrite, 0, POST_UPDATE);
}

template <int SCALE_ALG, bool FULL, bool HIGH = false>
__aicore__ inline __attribute__((always_inline)) void ScaleBlock(__ubuf__ uint8_t* scratch,
                                                                 const ScratchLayout& storage, unsigned batch,
                                                                 unsigned n, const VfPlan& plan, vector_u16 lane,
                                                                 vector_bool all, vector_bf16 half, vector_bf16 one)
{
    unsigned base = batch * 512;
    unsigned micro = n / 4;
    auto pm = plt_b16(micro, POST_UPDATE);
    auto max4 = (__ubuf__ bfloat16_t*)scratch + base / 4;
    auto max8 = (__ubuf__ bfloat16_t*)(scratch + storage.max8Offset) + base / 8;
    auto ea = (__ubuf__ bfloat16_t*)(scratch + storage.eaOffset + batch * 32);
    vector_bf16 m4, m8, eaBits;
    vlds(m4, max4, 0, NORM);
    vlds(m8, max8, 0, US_B16);
    vlds(eaBits, ea, 0, E2B_B16);
    vector_bf16 eaRec, a, b, ebRec, ecRec, invScale;
    vcvt_rcpe6m22bf16(eaRec, (vector_f8e6m2&)eaBits, pm, PART_EVEN);
    vmul(a, m8, eaRec, pm, MODE_ZEROING);
    vector_bool pb, pc;
    vcmps_ge(pb, a, (bfloat16_t)(SCALE_ALG == 1 ? 3.75f : 4.0f), pm);
    vsel(ebRec, half, one, pb);
    vmul(b, m4, eaRec, pm, MODE_ZEROING);
    vmul(b, b, ebRec, pm, MODE_ZEROING);
    vcmps_ge(pc, b, (bfloat16_t)(SCALE_ALG == 1 ? 1.875f : 2.0f), pm);
    vsel(ecRec, half, one, pc);
    vmul(invScale, ebRec, ecRec, pm, MODE_ZEROING);
    vmul(invScale, invScale, eaRec, pm, MODE_ZEROING);
    if constexpr (FULL) {
        vsts(invScale, invScale, (__ubuf__ bfloat16_t*)(scratch + base), 0, INTLV_B16, pm);
    } else {
        vector_u16 low, high;
        vintlv(low, high, (vector_u16&)invScale, (vector_u16&)invScale);
        vector_align out;
        auto write = (__ubuf__ uint16_t*)(scratch + base);
        vstus(out, plan.tailScaleLow, low, write, POST_UPDATE);
        if constexpr (HIGH) {
            vstus(out, plan.tailScaleHigh, high, write, POST_UPDATE);
        }
        vstas(out, write, 0, POST_UPDATE);
    }
    vector_align ebOut;
    auto ebWrite = (__ubuf__ uint32_t*)(scratch + storage.ebOffset + batch * 32);
    pstu(ebOut, pb, ebWrite);
    vstas(ebOut, ebWrite, 0, POST_UPDATE);
    auto ecWrite = (__ubuf__ uint32_t*)(scratch + storage.ecOffset + batch * 32);
    psts(pc, ecWrite, 16, PK, POST_UPDATE);
    // Reverse traversal preserves unread max4 during expansion to two-copy scales.
    mem_bar(VST_VLD);
}

__aicore__ inline __attribute__((always_inline)) void ExpBlock(__ubuf__ uint32_t* dst, __ubuf__ uint8_t* scratch,
                                                               const ScratchLayout& storage, unsigned group,
                                                               unsigned count, vector_u16 lane, vector_bool all)
{
    unsigned active = count;
    auto pg = plt_b16(active, POST_UPDATE);
    unsigned slot = group / 8 * 32, within = group % 8;
    auto ea = scratch + storage.eaOffset + slot + within * 2;
    auto eb = scratch + storage.ebOffset + slot + within;
    auto ec = (__ubuf__ uint16_t*)(scratch + storage.ecOffset + slot) + within;
    vector_u16 index, lo, ebReg, hi, words0, words1;
    vshls(index, lane, (uint16_t)1, all, MODE_ZEROING);
    vgather2(lo, ea, index, pg);
    vgather2(ebReg, eb, lane, pg);
    vshls(ebReg, ebReg, (uint16_t)8, pg, MODE_ZEROING);
    vor(lo, lo, ebReg, pg);
    vgather2(hi, ec, lane, pg);
    vintlv(words0, words1, lo, hi);
    vector_align out;
    auto write = dst;
    vstus(out, count, (vector_u32&)words0, write, POST_UPDATE);
    vstas(out, write, 0, POST_UPDATE);
}

// The 2D exponent tile keeps its own physical row stride.
template <unsigned MODE, unsigned TAIL>
__aicore__ inline __attribute__((always_inline)) void WriteExp(__ubuf__ uint32_t* exp, __ubuf__ uint8_t* scratch,
                                                               const ScratchLayout& storage, unsigned expStride,
                                                               vector_u16 lane, vector_bool all)
{
    if constexpr (MODE == 0) {
        for (uint16_t batch = 0; batch < (uint16_t)(storage.elements / 512); ++batch) {
            ExpBlock(exp + (unsigned)batch * 8, scratch, storage, (unsigned)batch * 8, 8, lane, all);
        }
        if constexpr (TAIL != 0) {
            unsigned group = storage.elements / 512 * 8;
            ExpBlock(exp + group, scratch, storage, group, TAIL / 64, lane, all);
        }
    } else {
        unsigned groups = storage.cols / 64;
        for (uint16_t row = 0; row < (uint16_t)(storage.elements / storage.cols); ++row) {
            unsigned group = (unsigned)row * groups;
            auto dst = exp + (unsigned)row * expStride;
            if constexpr (MODE == 1) {
                ExpBlock(dst, scratch, storage, group, groups, lane, all);
            } else {
                unsigned col = 0;
                unsigned pieces = (group % 8 + groups + 7) / 8;
                for (uint16_t piece = 0; piece < (uint16_t)pieces; ++piece) {
                    unsigned remaining = groups - col, slotRemaining = 8 - (group + col) % 8;
                    unsigned count = remaining < slotRemaining ? remaining : slotRemaining;
                    ExpBlock(dst + col, scratch, storage, group + col, count, lane, all);
                    col += count;
                }
            }
        }
    }
    mem_bar(VST_VLD);
}

template <bool FULL>
__aicore__ inline __attribute__((always_inline)) void QuantBlock(__ubuf__ bfloat16_t* src, __ubuf__ uint8_t* dst,
                                                                 __ubuf__ bfloat16_t* scales, unsigned n,
                                                                 vector_u16 lane, vector_bool all)
{
    unsigned active = n;
    auto mask = plt_b16(active, POST_UPDATE);
    vector_bf16 input, scale;
    vlds(input, src, 0, NORM);
    vlds(scale, scales, 0, US_B16);
    vmul(input, input, scale, mask, MODE_ZEROING);
    vector_u8 packed;
    vcvt((vector_f4e1m2x2&)packed, input, mask, ROUND_R, PART_P0);
    vsts(packed, dst, 0, PK4_B32, mask);
}

template <bool BATCH_TAIL>
__aicore__ inline __attribute__((always_inline)) void Max8Ea(__ubuf__ uint8_t* scratch, const ScratchLayout& storage,
                                                             const VfPlan& plan, vector_u16 lane)
{
    for (uint16_t batch = 0; batch < (uint16_t)plan.fullBatches; ++batch) {
        Max8EaBlock<true>(scratch, storage, batch, 512, lane);
    }
    if constexpr (BATCH_TAIL) {
        Max8EaBlock<false>(scratch, storage, plan.fullBatches, plan.tailElements, lane);
    }
    mem_bar(VST_VLD);
}

template <int SCALE_ALG, unsigned TAIL_KIND>
__aicore__ inline __attribute__((always_inline)) void Scale(__ubuf__ uint8_t* scratch, const ScratchLayout& storage,
                                                            const VfPlan& plan, vector_u16 lane, vector_bool all,
                                                            vector_bf16 half, vector_bf16 one)
{
    if constexpr (TAIL_KIND != 0) {
        ScaleBlock<SCALE_ALG, false, TAIL_KIND == 2>(scratch, storage, plan.fullBatches, plan.tailElements, plan, lane,
                                                     all, half, one);
    }
    for (uint16_t reverse = 0; reverse < (uint16_t)plan.fullBatches; ++reverse) {
        unsigned batch = plan.fullBatches - 1 - (unsigned)reverse;
        ScaleBlock<SCALE_ALG, true>(scratch, storage, batch, 512, plan, lane, all, half, one);
    }
}

template <unsigned TAIL>
__aicore__ inline __attribute__((always_inline)) void Quant(__ubuf__ bfloat16_t* src, __ubuf__ uint8_t* dst,
                                                            __ubuf__ uint8_t* scratch, const VfPlan& plan,
                                                            vector_u16 lane, vector_bool all)
{
    for (uint16_t part = 0; part < (uint16_t)plan.fullCols; ++part) {
        QuantBlock<true>(src + (unsigned)part * 128, dst + (unsigned)part * 64,
                         (__ubuf__ bfloat16_t*)scratch + (unsigned)part * 64, 128, lane, all);
    }
    if constexpr ((TAIL & 64) != 0) {
        QuantBlock<false>(src + plan.fullCols * 128, dst + plan.fullCols * 64,
                          (__ubuf__ bfloat16_t*)scratch + plan.fullCols * 64, 64, lane, all);
    }
    mem_bar(VST_VLD);
}

template <int ALG, bool PAD, unsigned TAIL, unsigned EXP_MODE>
__aicore__ inline void Run(__ubuf__ uint8_t* dst, __ubuf__ uint32_t* exp, __ubuf__ uint8_t* scratch,
                           __ubuf__ bfloat16_t* src, unsigned cols, unsigned expStride, const ScratchLayout& storage)
{
    const VfPlan plan(cols, storage);
    __VEC_SCOPE__
    {
        vector_u16 lane, absMask;
        vci((vector_s16&)lane, (int16_t)0, INC_ORDER);
        auto all = pset_b16(PAT_ALL);
        vbr(absMask, 0x7fff);
        vector_bf16 half, one;
        vbr(half, 0.5f);
        vbr(one, 1.0f);
        Pad<PAD>(src, plan);
        Max4<TAIL>(src, (__ubuf__ bfloat16_t*)scratch, plan, lane, absMask, all);
        Max8Ea<TAIL != 0>(scratch, storage, plan, lane);
        Scale<ALG, TAIL == 0 ? 0 : (TAIL <= 256 ? 1 : 2)>(scratch, storage, plan, lane, all, half, one);
        WriteExp<EXP_MODE, TAIL>(exp, scratch, storage, expStride, lane, all);
        Quant<TAIL>(src, dst, scratch, plan, lane, all);
    }
}

template <int ALG, bool PAD, unsigned MODE>
__aicore__ inline void DispatchTail(__ubuf__ uint8_t* dst, __ubuf__ uint32_t* exp, __ubuf__ uint8_t* scratch,
                                    __ubuf__ bfloat16_t* src, unsigned cols, unsigned expStride,
                                    const ScratchLayout& storage)
{
#define HIF4_TAIL_CASE(N)                                                         \
    case N:                                                                       \
        Run<ALG, PAD, N, MODE>(dst, exp, scratch, src, cols, expStride, storage); \
        break
    switch (storage.elements % 512) {
        HIF4_TAIL_CASE(0);
        HIF4_TAIL_CASE(64);
        HIF4_TAIL_CASE(128);
        HIF4_TAIL_CASE(192);
        HIF4_TAIL_CASE(256);
        HIF4_TAIL_CASE(320);
        HIF4_TAIL_CASE(384);
        HIF4_TAIL_CASE(448);
    }
#undef HIF4_TAIL_CASE
}

template <int ALG, bool PAD>
__aicore__ inline void DispatchExp(__ubuf__ uint8_t* dst, __ubuf__ uint32_t* exp, __ubuf__ uint8_t* scratch,
                                   __ubuf__ bfloat16_t* src, unsigned cols, unsigned expStride,
                                   const ScratchLayout& storage)
{
    unsigned groups = storage.cols / 64;
    if (expStride == groups || storage.elements == storage.cols) {
        DispatchTail<ALG, PAD, 0>(dst, exp, scratch, src, cols, expStride, storage);
    } else if (groups <= 8 && 8 % groups == 0) {
        DispatchTail<ALG, PAD, 1>(dst, exp, scratch, src, cols, expStride, storage);
    } else {
        DispatchTail<ALG, PAD, 2>(dst, exp, scratch, src, cols, expStride, storage);
    }
}
} // namespace hif4

// Only 2D layout is visible here. Higher dimensions belong to the scalar caller.
template <int SCALE_ALG = 0>
__aicore__ inline void QuantHiF4Kernel(__ubuf__ uint8_t* dst, __ubuf__ uint32_t* exp, __ubuf__ uint8_t* scratch,
                                       __ubuf__ bfloat16_t* src, unsigned rows, unsigned cols, unsigned capacityCols,
                                       unsigned expStride)
{
    if (rows == 0 || cols == 0) {
        return;
    }
    hif4::ScratchLayout storage(rows, capacityCols);
    if (cols != capacityCols) {
        hif4::DispatchExp<SCALE_ALG, true>(dst, exp, scratch, src, cols, expStride, storage);
    } else {
        hif4::DispatchExp<SCALE_ALG, false>(dst, exp, scratch, src, cols, expStride, storage);
    }
}

namespace hif4 {
// Local implementation with ISA-style 2D tile arguments. No ISA arithmetic call.
template <int SCALE_ALG = 0, typename DstTile, typename SrcTile, typename ExpTile, typename ScratchTile>
__aicore__ inline void TQUANT(DstTile& dst, SrcTile& src, ExpTile& exp, ScratchTile& scratch)
{
    static_assert(SCALE_ALG == 0 || SCALE_ALG == 1, "HiF4 scale algorithm must be 0 or 1.");
    static_assert(SrcTile::isRowMajor && DstTile::isRowMajor && ExpTile::isRowMajor && !SrcTile::isBoxedLayout &&
                      !DstTile::isBoxedLayout && !ExpTile::isBoxedLayout,
                  "HiF4 requires unboxed row-major 2D tiles.");
    static_assert(std::is_same_v<typename SrcTile::DType, bfloat16_t>, "HiF4 input must be BF16.");
    static_assert(std::is_same_v<typename DstTile::DType, float4_e1m2x2_t>, "HiF4 output must be packed HF4.");
    static_assert(std::is_same_v<typename ExpTile::DType, uint32_t>, "HiF4 exponent must be uint32.");
    static_assert(std::is_same_v<typename ScratchTile::DType, uint8_t>, "HiF4 scratch must be bytes.");
    static_assert(SrcTile::Cols > 0 && SrcTile::Cols % 64 == 0, "HiF4 tile columns must be 64-aligned.");
    static_assert(DstTile::Cols == SrcTile::Cols && DstTile::Rows >= SrcTile::Rows,
                  "HiF4 source and destination must have matching 2D capacity.");
    static_assert(ExpTile::Rows >= SrcTile::Rows && ExpTile::Cols >= SrcTile::Cols / 64,
                  "HiF4 exponent capacity must cover every source group.");
    constexpr ScratchLayout allocation(SrcTile::Rows, SrcTile::Cols);
    static_assert(ScratchTile::Rows * ScratchTile::Cols >= allocation.bytes,
                  "HiF4 scratch must cover one complete 2D tile.");
    QuantHiF4Kernel<SCALE_ALG>((__ubuf__ uint8_t*)dst.data(), (__ubuf__ uint32_t*)exp.data(),
                               (__ubuf__ uint8_t*)scratch.data(), (__ubuf__ bfloat16_t*)src.data(), src.GetValidRow(),
                               src.GetValidCol(), SrcTile::Cols, ExpTile::Cols);
}
} // namespace hif4
#endif
