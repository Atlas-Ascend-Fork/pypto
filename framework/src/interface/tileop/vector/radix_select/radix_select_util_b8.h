/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file radix_select_util_b8.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_RADIX_SELECT_UTIL_B8__H
#define TILEOP_TILE_OPERATOR_RADIX_SELECT_UTIL_B8__H
#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6)

#include "radix_select_util.h"

namespace RadixSelectUtil {

constexpr size_t B8_TO_B4_COL_RATIO = sizeof(uint64_t) / sizeof(uint32_t);

template <typename T>
struct B8ToB4 {
    using U = typename std::conditional_t<std::is_same_v<T, int64_t>, int32_t, uint32_t>;
};

template <typename T>
struct RSB8ToB4Tile {
    using U = typename pto::Tile<T::Loc, typename B8ToB4<typename T::DType>::U, T::Rows, T::Cols * B8_TO_B4_COL_RATIO,
                                 T::BFractal, -1, -1>;
};

template <typename DST, typename SRC>
TILEOP void RadixSelectMoveB8(DST dst, SRC src)
{
    typename RSB8ToB4Tile<DST>::U dst32(dst.GetValidRow(), dst.GetValidCol() * B8_TO_B4_COL_RATIO);
    typename RSB8ToB4Tile<SRC>::U src32(src.GetValidRow(), src.GetValidCol() * B8_TO_B4_COL_RATIO);
    pto::TASSIGN(dst32, (uint64_t)dst.data());
    pto::TASSIGN(src32, (uint64_t)src.data());
    pto::TMOV(dst32, src32);
}

template <typename DST, typename SRC>
TILEOP void RadixSelectNotB8(DST dst, SRC src)
{
    typename RSB8ToB4Tile<DST>::U dst32(dst.GetValidRow(), dst.GetValidCol() * B8_TO_B4_COL_RATIO);
    typename RSB8ToB4Tile<SRC>::U src32(src.GetValidRow(), src.GetValidCol() * B8_TO_B4_COL_RATIO);
    pto::TASSIGN(dst32, (uint64_t)dst.data());
    pto::TASSIGN(src32, (uint64_t)src.data());
    pto::TNOT(dst32, src32);
}

template <bool isLargest, bool in, bool isUInt, typename SrcDType, typename TWI, typename SRC, typename TMP>
TILEOP void RadixSelectTwiddleB8(TWI twi, SRC src, TMP tmp)
{
    if constexpr (!isLargest && !in) {
        RadixSelectNotB8(src, src);
    }
    if constexpr (isUInt) {
        RadixSelectMoveB8(twi, src);
    } else {
        constexpr auto SIGN = SignByType<SrcDType>::value;
        pto::TADDS(tmp, src, SIGN);
        RadixSelectMoveB8(twi, tmp);
    }
    if constexpr (!isLargest && in) {
        RadixSelectNotB8(twi, twi);
    }
}

template <size_t where>
TILEOP constexpr pto::MaskPattern RadixSelectGetPatternB8()
{
    if constexpr (where < RADIX_BITS_PER_PASS) {
        return pto::MaskPattern::P1000;
    } else if constexpr (where < RADIX_BITS_PER_PASS * 2) {
        return pto::MaskPattern::P0100;
    } else if constexpr (where < RADIX_BITS_PER_PASS * 3) {
        return pto::MaskPattern::P0010;
    } else {
        return pto::MaskPattern::P0001;
    }
}

template <size_t where, typename DST, typename SRC>
TILEOP void RadixSelect16BitGetB8(DST dst, SRC src)
{
    constexpr pto::MaskPattern pattern = RadixSelectGetPatternB8<where>();
    pto::TGATHER<DST, SRC, pattern>(dst, src);
    if constexpr (where % RADIX_BITS_PER_PASS == 0) {
        pto::TSHRS(dst, dst, BITS_PER_BYTE);
    } else {
        pto::TANDS(dst, dst, UINT8_VALUE_MASK);
    }
}

template <size_t bit, typename DST16U, typename TMP16U, typename SRC16UF, typename MASK, typename USELESS>
TILEOP void RadixSelectHistogramPrepareB8(DST16U dst16U, TMP16U tmp16U, SRC16UF src16UF, MASK mask, USELESS useless)
{
    if constexpr (bit == 0) {
        RadixSelect16BitGetB8<0>(dst16U, src16UF);
    } else {
        pto::TNOT(tmp16U, dst16U);
        RadixSelectSel(dst16U, mask, dst16U, tmp16U, useless);
        pto::TSHLS(dst16U, dst16U, BITS_PER_BYTE);
        RadixSelect16BitGetB8<bit / BITS_PER_BYTE>(tmp16U, src16UF);
        pto::TADD(dst16U, dst16U, tmp16U);
    }
}

template <typename HIS, typename HIS_TMP32, typename HIS_TMP32U, typename HIGH, typename RMK, typename CMP,
          typename TCI, typename USELESS>
TILEOP void RadixSelectCalcKTHBitB8(HIS his, HIS_TMP32 hisTmp32, HIS_TMP32U hisTmp32U, HIGH high, RMK rmk, CMP cmp,
                                    TCI tci, USELESS useless)
{
    RadixSelectCmps<pto::CmpMode::GT>(cmp, his, rmk);
    pto::TMOV(hisTmp32U, tci);
    pto::TSELS(hisTmp32U, cmp, hisTmp32U, useless, HISTOGRAM_SENTINEL);
    pto::TEXPANDS(high, 0);
    pto::TROWMIN(high, hisTmp32, useless);
}

template <size_t bit, typename KTH, typename SRC, typename TCI, typename TMP>
TILEOP void RadixSelectUpdateKTHValueB8(KTH kth, SRC src, TCI tci, TMP tmp)
{
    RadixSelectTCI(tci);
    if constexpr (bit >= B32_SORT_BITS) {
        pto::TEXPANDS(tmp, 1);
        pto::TSUB(tci, tmp, tci);
    }
    constexpr size_t realBit = bit % B32_SORT_BITS;
    if constexpr (realBit == B32_SORT_BITS - BITS_PER_BYTE) {
        pto::TROWEXPAND(tmp, src);
    } else {
        pto::TSHLS(tmp, src, B32_SORT_BITS - BITS_PER_BYTE - realBit);
        pto::TROWEXPAND(tmp, tmp);
    }
    pto::TMUL(tmp, tmp, tci);
    pto::TADD(kth, kth, tmp);
}

template <typename SRC, typename HIGH, typename MASK, typename CMP>
TILEOP void RadixSelectUpdateMaskB8(SRC src, HIGH high, MASK mask, CMP cmp)
{
    pto::TSHLS(src, src, BITS_PER_BYTE);
    pto::TSHRS(src, src, BITS_PER_BYTE);
    RadixSelectCmps<pto::CmpMode::EQ>(cmp, src, high);
    pto::TAND(mask, mask, cmp);
    pto::TROWEXPAND(src, high);
}

template <typename RMK, typename HIS, typename HIGH, typename GATHER, typename IDX, typename CMP, typename USELESS>
TILEOP void RadixSelectUpdateRemaindKB8(RMK rmk, HIS his, HIGH high, GATHER gather, IDX idx, CMP cmp, USELESS useless)
{
    pto::TCMPS(cmp, high, 0, pto::CmpMode::NE);
    pto::TADDS(idx, high, BYTE_LOW_BIT_MASK);
    RadixSelectGather(gather, his, idx, useless);
    pto::TSELS(gather, cmp, gather, useless, 0);
    pto::TSUB(rmk, rmk, gather);
}

template <pto::CmpMode cmpMode, typename SELECT, typename SRC, typename K, typename TMP1, typename TMP2, typename COUNT,
          typename CMP, typename USELESS>
TILEOP void RadixSelectGatherB8(SELECT select, SRC src, K kth, TMP1 tmp1, TMP2 tmp2, COUNT count, CMP cmp,
                                USELESS useless)
{
    pto::TEXPANDS(tmp1, UINT8_VALUE_MASK);
    pto::TEXPANDS(tmp2, UINT8_VALUE_MASK);
    auto validRow = src.GetValidRow();
    auto validCol = src.GetValidCol();
    constexpr size_t elementPerLoop = 64;
    size_t loop = validCol / elementPerLoop;
    typename RSExchangeTile<CMP>::U cmpOneRow(1, elementPerLoop / BITS_PER_BYTE);
    typename RSExchangeTile<SRC>::U srcOneRow(1, elementPerLoop);
    typename RSExchangeTile<K>::U kthOneRow(1, elementPerLoop);
    typename RSExchangeTile<TMP1>::U tmp1OneRow(1, elementPerLoop);
    typename RSExchangeTile<TMP2>::U tmp2OneRow(1, elementPerLoop);
    for (LoopVar i = 0; i < validRow; ++i) {
        for (LoopVar j = 0; j < loop; ++j) {
            pto::TASSIGN(cmpOneRow, (int64_t)(cmp.data() + i * CMP::RowStride));
            pto::TASSIGN(srcOneRow, (int64_t)(src.data() + i * SRC::RowStride + j * elementPerLoop));
            pto::TASSIGN(kthOneRow, (int64_t)(kth.data() + i * K::RowStride + j * elementPerLoop));
            pto::TASSIGN(tmp1OneRow, (int64_t)(tmp1.data() + i * TMP1::RowStride + j * elementPerLoop));
            pto::TASSIGN(tmp2OneRow, (int64_t)(tmp2.data() + i * TMP2::RowStride + j * elementPerLoop));
            pto::TCMP(cmpOneRow, srcOneRow, kthOneRow, cmpMode);
            pto::TSELS(tmp1OneRow, cmpOneRow, tmp1OneRow, useless, ~UINT8_VALUE_MASK);
        }
    }
    RadixSelectGather<pto::CmpMode::EQ>(select, tmp1, tmp2, count, useless);
}

template <typename GT, typename EQ, typename SRC, typename K, typename TMP1, typename TMP2, typename CGT, typename CEQ,
          typename IDX, typename CMP, typename USELESS>
TILEOP void RadixSelectFinalSelectB8(GT selectGT, EQ selectEQ, SRC src, K kth, TMP1 tmp1, TMP2 tmp2, CGT countGT,
                                     CEQ countEQ, IDX idx, CMP cmp, USELESS useless)
{
    RadixSelectGatherB8<pto::CmpMode::GT>(selectGT, src, kth, tmp1, tmp2, countGT, cmp, useless);
    RadixSelectGatherB8<pto::CmpMode::EQ>(selectEQ, src, kth, tmp1, tmp2, countEQ, cmp, useless);
    pto::TCONCAT(idx, selectGT, selectEQ, countGT, countEQ);
}

/*
Memory Usage:
srcMaskAddr       | srcMaskShape    uint64 |
sortTmpAddr       | srcMaskShape    uint64 |
number0Addr       | 16              uint16 |
number1Addr       | 16              uint16 |
number2Addr       | 16              uint16 |
number3Addr       | 16              uint16 |
cnt1Addr          | 8               uint32 |
cnt2Addr          | 8               uint32 |
cnt3Addr          | 8               uint32 |
select1Addr       | srcMaskShape    uint32 |
select2Addr       | srcMaskShape    uint32 | and sortTmp2Addr
select3Addr       | srcMaskShape    uint32 |
indexAddr         | srcMaskShape    uint32 |
*/
template <int64_t bit, typename SRCK, typename IDX, typename TMPK, typename TMP16, typename SELECT, typename COUNT,
          typename NUM, typename USELESS>
TILEOP void RadixSelectSortTwoBitCalcB8(SRCK srck, IDX index, TMPK tmpk, TMP16 tmp16, USELESS useless, SELECT select1,
                                        SELECT select2, SELECT select3, COUNT count1, COUNT count2, COUNT count3,
                                        NUM num0, NUM num1, NUM num2, NUM num3)
{
    RadixSelectGather<pto::CmpMode::EQ>(select2, tmp16, num3, count2, useless);
    RadixSelectGather<pto::CmpMode::EQ>(select1, tmp16, num2, count1, useless);
    pto::TADD(count3, count1, count2);
    pto::TCONCAT(select3, select2, select1, count2, count1);
    RadixSelectGather<pto::CmpMode::EQ>(select1, tmp16, num1, count1, useless);
    pto::TADD(count2, count1, count3);
    pto::TCONCAT(select2, select3, select1, count3, count1);
    if constexpr (bit == 0) {
        RadixSelectGather<pto::CmpMode::EQ>(select3, tmp16, num0, count3, useless);
        pto::TCONCAT(select1, select2, select3, count2, count3);
        RadixSelectGather(tmpk, srck, select1, useless);
    } else {
        RadixSelectGather<pto::CmpMode::EQ>(select1, tmp16, num0, count1, useless);
        pto::TCONCAT(select3, select2, select1, count2, count1);
        RadixSelectGather(select1, index, select3, useless);
        RadixSelectGather(tmpk, srck, select3, useless);
    }
}

template <int64_t bit, typename SRC, typename TMP32F, typename TMP16K, typename TMP2U16, typename TMP2U16K>
TILEOP void RadixSelectSortTwoBitPrepareB8(SRC src, TMP32F tmp32F, TMP16K tmp16K, TMP2U16 tmp2U16, TMP2U16K tmp2U16K)
{
    RadixSelect16BitGetB8<(B64_SORT_BITS - RADIX_BITS_PER_PASS - bit) / BITS_PER_BYTE>(tmp2U16, src);
    pto::TSHRS(tmp2U16, tmp2U16, bit % BITS_PER_BYTE);
    pto::TANDS(tmp2U16, tmp2U16, RADIX_PASS_MASK);
    pto::TEXPANDS(tmp32F, PACKED_INT16_MAX);
    pto::TMOV(tmp16K, tmp2U16K);
}

template <int64_t bit, int64_t lastBit, typename SRC16UF, typename SRCK, typename SRC32F, typename SRC16,
          typename SRC16K, typename IDX, typename TMP16UF, typename TMPK, typename TMP32F, typename TMP16,
          typename TMP16K, typename TMP2U16, typename TMP2U16K, typename SELECT, typename COUNT, typename NUM,
          typename USELESS>
TILEOP void RadixSelectSortTwoBitB8(SRC16UF src16UF, SRCK srck, SRC32F src32F, SRC16 src16, SRC16K src16K, IDX index,
                                    TMP16UF tmp16UF, TMPK tmpk, TMP32F tmp32F, TMP16 tmp16, TMP16K tmp16K,
                                    TMP2U16 tmp2U16, TMP2U16K tmp2U16K, USELESS useless, SELECT select1, SELECT select2,
                                    SELECT select3, COUNT count1, COUNT count2, COUNT count3, NUM num0, NUM num1,
                                    NUM num2, NUM num3)
{
    if constexpr (bit < lastBit) {
        RadixSelectSortTwoBitPrepareB8<bit>(src16UF, tmp32F, tmp16K, tmp2U16, tmp2U16K);
        RadixSelectSortTwoBitCalcB8<bit>(srck, index, tmpk, tmp16, useless, select1, select2, select3, count1, count2,
                                         count3, num0, num1, num2, num3);
        RadixSelectSortTwoBitB8<bit + RADIX_BITS_PER_PASS, lastBit>(
            tmp16UF, tmpk, tmp32F, tmp16, tmp16K, select1, src16UF, srck, src32F, src16, src16K, tmp2U16, tmp2U16K,
            useless, index, select2, select3, count1, count2, count3, num0, num1, num2, num3);
    }
}

} // namespace RadixSelectUtil

#endif // defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6)
#endif // TILEOP_TILE_OPERATOR_RADIX_SELECT_UTIL_B8__H
