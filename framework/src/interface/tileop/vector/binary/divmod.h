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
 * \file divmod.h
 * \brief Fused integer divmod tile operation (integer-only, no dtype promotion).
 */

#ifndef TILEOP_TILE_OPERATOR_VEC_BINARY_DIVMOD_H
#define TILEOP_TILE_OPERATOR_VEC_BINARY_DIVMOD_H

#include "utils/sync.h"
#include "basic.h"
#include <limits>

constexpr int16_t INT16_MAX_DIVMOD_ZERO_DIVISOR_RESULT = 0x7FFF;
constexpr int16_t INT16_MIN_DIVMOD_ZERO_DIVISOR_RESULT = static_cast<int16_t>(0x8000U);
constexpr int32_t INT32_MAX_DIVMOD_ZERO_DIVISOR_RESULT = 0x7FFFFFFF;
constexpr int32_t INT32_MIN_DIVMOD_ZERO_DIVISOR_RESULT = static_cast<int32_t>(0x80000000U);
constexpr int64_t INT64_MAX_DIVMOD_ZERO_DIVISOR_RESULT = 0x7FFFFFFFFFFFFFFF;
constexpr int64_t INT64_MIN_DIVMOD_ZERO_DIVISOR_RESULT = static_cast<int64_t>(0x8000000000000000ULL);
constexpr int16_t UINT8_DIVMOD_ZERO_DIVISOR_RESULT = 0xFF;
constexpr int16_t UINT16_DIVMOD_ZERO_DIVISOR_RESULT = static_cast<int16_t>(0xFFFFU);
constexpr float UINT16_DIVMOD_NORMALIZE_OFFSET = 65536.0f;
constexpr size_t DIVMOD_MASK_WIDTH_SCALE_32BIT = 4;
constexpr size_t DIVMOD_SCRATCH_TAIL_BYTES = 32;

template <typename T>
TILEOP constexpr size_t DivModTmpPlaneCount()
{
#ifdef __DAV_V220
    if constexpr (std::is_same_v<T, int16_t>) {
        return TileOp::NUM_VALUE_3;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return TileOp::NUM_VALUE_6;
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return TileOp::NUM_VALUE_6;
    } else {
        return TileOp::NUM_VALUE_5;
    }
#else
    if constexpr (std::is_same_v<T, uint16_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>) {
        return TileOp::NUM_VALUE_2;
    } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
        return TileOp::NUM_VALUE_5;
    } else if constexpr (std::is_same_v<T, int16_t>) {
        return TileOp::NUM_VALUE_3;
    } else {
        return TileOp::NUM_VALUE_4;
    }
#endif
}

// Physical mask tile width (uint8 columns). PTO requires RowMajor mask tiles to be 32-byte
// aligned, and the ValidCol (= tileW, one byte per column) must not exceed Cols. Masks are
// bit-packed (1 bit/element), so AlignUp(tileW, 32) is both valid and enough bits, and it
// still fits the existing tmp planes (plane bytes >= 4 * AlignUp(tileW, 8)).
template <auto TileW>
TILEOP constexpr size_t DivModMaskCols()
{
    constexpr size_t kMaskAlign = 32;
    constexpr size_t raw = static_cast<size_t>(TileW);
    return ((raw + kMaskAlign - 1) / kMaskAlign) * kMaskAlign;
}

template <auto pos, auto neg, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
          typename T7>
TILEOP void DivModNonV220Int(T0 dst, T1 src0, T2 src1, T3 tmp0, T4 tmp1, T5 tmp2, T6 tmp3, T7 tmp4)
{
    pto::TCMPS(tmp3, src0, 0, pto::CmpMode::LT);
    pto::TSELS(tmp2, tmp3, src0, tmp2, pos);
    pto::TCMPS(tmp3, src0, 0, pto::CmpMode::GE);
    pto::TSELS(tmp2, tmp3, tmp2, tmp2, neg);
    pto::TCMPS(tmp3, src1, 0, pto::CmpMode::NE);
    pto::TPARTSEL(tmp0, tmp3, src0, tmp2, tmp2);
    pto::TSELS(tmp1, tmp3, src1, tmp2, 1);

    pto::TCMPS(tmp3, tmp0, 0, pto::CmpMode::LT);
    pto::TCMPS(tmp4, tmp1, 0, pto::CmpMode::LT);
    pto::TXOR(tmp3, tmp3, tmp4, tmp4);

    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(dst, tmp0, tmp1);

    // A5 TREM does not use the tmp argument.
    pto::TREM(tmp0, tmp0, tmp1, tmp0);

    pto::TCMPS(tmp4, tmp0, 0, pto::CmpMode::NE);
    pto::TAND(tmp3, tmp3, tmp4);
    pto::TADDS(tmp0, dst, -1);
    pto::TPARTSEL(dst, tmp3, tmp0, dst, tmp2);
}

template <typename TmpTensor>
TILEOP uint64_t DivModTmpAddr(TmpTensor tmp, size_t byteOffset, size_t tileShapeSize, size_t tileIndex,
                              size_t elementSize)
{
    return (uint64_t)((__ubuf__ uint8_t*)tmp.GetAddr() + tileIndex * tileShapeSize * elementSize + byteOffset);
}

template <auto tileH, auto tileW, typename T4, typename Src0Tile, typename Src1Tile, typename DstTile>
TILEOP void DivModV220Int8Compute(DstTile qTile, DstTile rTile, Src0Tile src0Tile, Src1Tile src1Tile, T4 tmp,
                                  size_t offset, size_t dstShape3, size_t dstShape4, size_t tileShapeSize)
{
    using T = typename DstTile::DType;
    using HalfTileDefine = pto::Tile<pto::TileType::Vec, half, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using Fp32TileDefine = pto::Tile<pto::TileType::Vec, float, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using MaskTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, DIVMOD_MASK_WIDTH_SCALE_32BIT * tileW,
                                     pto::BLayout::RowMajor, -1, -1>;
    HalfTileDefine tmp0Tile(dstShape3, dstShape4);
    HalfTileDefine tmp1Tile(dstShape3, dstShape4);
    HalfTileDefine qTileHalf(dstShape3, dstShape4);
    HalfTileDefine rTileHalf(dstShape3, dstShape4);
    HalfTileDefine qSentHalf(dstShape3, dstShape4);
    HalfTileDefine scratchTile(dstShape3, dstShape4);
    Fp32TileDefine tmp2Tile(dstShape3, dstShape4);
    Fp32TileDefine tmp3Tile(dstShape3, dstShape4);
    MaskTileDefine zeroMask(dstShape3, dstShape4);
    MaskTileDefine aZeroMask(dstShape3, dstShape4);
    pto::TASSIGN(tmp0Tile, DivModTmpAddr(tmp, offset, tileShapeSize, 0, sizeof(float)));
    pto::TASSIGN(tmp1Tile, DivModTmpAddr(tmp, offset, tileShapeSize, 1, sizeof(float)));
    pto::TASSIGN(tmp2Tile, DivModTmpAddr(tmp, offset, tileShapeSize, 2, sizeof(float)));
    pto::TASSIGN(tmp3Tile, DivModTmpAddr(tmp, offset, tileShapeSize, 3, sizeof(float)));
    pto::TASSIGN(qTileHalf, DivModTmpAddr(tmp, offset, tileShapeSize, 2, sizeof(float)));
    pto::TASSIGN(rTileHalf, DivModTmpAddr(tmp, offset, tileShapeSize, 3, sizeof(float)));
    pto::TASSIGN(qSentHalf, DivModTmpAddr(tmp, offset, tileShapeSize, 3, sizeof(float)));
    pto::TASSIGN(zeroMask, DivModTmpAddr(tmp, offset, tileShapeSize, 4, sizeof(float)));
    pto::TASSIGN(aZeroMask, DivModTmpAddr(tmp, offset, tileShapeSize, 3, sizeof(float)));
    pto::TASSIGN(scratchTile,
                 DivModTmpAddr(tmp, offset + DIVMOD_MASK_WIDTH_SCALE_32BIT * tileW - DIVMOD_SCRATCH_TAIL_BYTES,
                               tileShapeSize, 4, sizeof(float)));

    pto::TCVT(tmp0Tile, src0Tile, pto::RoundMode::CAST_NONE);
    SyncV();
    pto::TCVT(tmp1Tile, src1Tile, pto::RoundMode::CAST_NONE);
    SyncV();
    pto::TCVT(tmp2Tile, tmp0Tile, pto::RoundMode::CAST_NONE);
    SyncV();
    pto::TCVT(tmp3Tile, tmp1Tile, pto::RoundMode::CAST_NONE);
    SyncV();
    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(tmp2Tile, tmp2Tile, tmp3Tile);
    SyncV();
    pto::TCVT(qTileHalf, tmp2Tile, pto::RoundMode::CAST_FLOOR);
    SyncV();

    // b == 0: q = (a < 0) ? dtype MIN : dtype MAX, r = a
    constexpr half kZeroMax = static_cast<half>(
        static_cast<float>(static_cast<int32_t>(std::numeric_limits<T>::max())));
    constexpr half kZeroMin = static_cast<half>(
        static_cast<float>(static_cast<int32_t>(std::numeric_limits<T>::min())));
    if constexpr (std::is_signed_v<T>) {
        pto::TCMPS(zeroMask, tmp0Tile, static_cast<half>(0.0f), pto::CmpMode::LT);
        SyncV();
    }
    pto::TSUB(qSentHalf, tmp0Tile, tmp0Tile);
    SyncV();
    pto::TADDS(qSentHalf, qSentHalf, kZeroMin);
    SyncV();
    if constexpr (std::is_signed_v<T>) {
        pto::TSELS(qSentHalf, zeroMask, qSentHalf, scratchTile, kZeroMax);
        SyncV();
    } else {
        pto::TADDS(qSentHalf, qSentHalf, kZeroMax);
        SyncV();
    }
    pto::TCMPS(zeroMask, tmp1Tile, static_cast<half>(0.0f), pto::CmpMode::EQ);
    SyncV();
    pto::TSEL(qTileHalf, zeroMask, qSentHalf, qTileHalf, scratchTile);
    SyncV();
    pto::TCMPS(aZeroMask, tmp0Tile, static_cast<half>(0.0f), pto::CmpMode::EQ);
    SyncV();
    pto::TAND(aZeroMask, aZeroMask, zeroMask);
    SyncV();
    pto::TNOT(zeroMask, aZeroMask);
    SyncV();
    pto::TSELS(qTileHalf, zeroMask, qTileHalf, scratchTile, static_cast<half>(0.0f));
    SyncV();

    pto::TCVT(qTile, qTileHalf, pto::RoundMode::CAST_FLOOR, pto::SaturationMode::OFF);
    SyncV();
    pto::TCVT(qTileHalf, qTile, pto::RoundMode::CAST_NONE);
    SyncV();
    pto::TMUL(rTileHalf, qTileHalf, tmp1Tile);
    SyncV();
    pto::TSUB(rTileHalf, tmp0Tile, rTileHalf);
    SyncV();
    pto::TSELS(rTileHalf, zeroMask, rTileHalf, scratchTile, kZeroMax);
    SyncV();
    pto::TCVT(rTile, rTileHalf, pto::RoundMode::CAST_RINT);
    SyncV();
}

template <auto tileH, auto tileW, typename T4, typename Src0Tile, typename Src1Tile, typename DstTile>
TILEOP void DivModNonV220Int8Compute(DstTile qTile, DstTile rTile, Src0Tile src0Tile, Src1Tile src1Tile, T4 tmp,
                                     size_t offset, size_t dstShape3, size_t dstShape4, size_t tileShapeSize)
{
    using T = typename DstTile::DType;
    using HalfTileDefine = pto::Tile<pto::TileType::Vec, half, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using Fp32TileDefine = pto::Tile<pto::TileType::Vec, float, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using MaskTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, DivModMaskCols<tileW>(),
                                     pto::BLayout::RowMajor, -1, -1>;
    HalfTileDefine tmp0Tile(dstShape3, dstShape4);
    HalfTileDefine tmp1Tile(dstShape3, dstShape4);
    HalfTileDefine qTileHalf(dstShape3, dstShape4);
    HalfTileDefine rTileHalf(dstShape3, dstShape4);
    HalfTileDefine qSentHalf(dstShape3, dstShape4);
    HalfTileDefine scratchTile(dstShape3, dstShape4);
    Fp32TileDefine tmp2Tile(dstShape3, dstShape4);
    Fp32TileDefine tmp3Tile(dstShape3, dstShape4);
    MaskTileDefine zeroMask(dstShape3, dstShape4);
    MaskTileDefine aZeroMask(dstShape3, dstShape4);
    pto::TASSIGN(tmp0Tile, DivModTmpAddr(tmp, offset, tileShapeSize, 0, sizeof(float)));
    pto::TASSIGN(tmp1Tile, DivModTmpAddr(tmp, offset, tileShapeSize, 1, sizeof(float)));
    pto::TASSIGN(tmp2Tile, DivModTmpAddr(tmp, offset, tileShapeSize, 2, sizeof(float)));
    pto::TASSIGN(tmp3Tile, DivModTmpAddr(tmp, offset, tileShapeSize, 3, sizeof(float)));
    pto::TASSIGN(qTileHalf, DivModTmpAddr(tmp, offset, tileShapeSize, 2, sizeof(float)));
    pto::TASSIGN(rTileHalf, DivModTmpAddr(tmp, offset, tileShapeSize, 3, sizeof(float)));
    pto::TASSIGN(qSentHalf, DivModTmpAddr(tmp, offset, tileShapeSize, 3, sizeof(float)));
    pto::TASSIGN(zeroMask, DivModTmpAddr(tmp, offset, tileShapeSize, 4, sizeof(float)));
    pto::TASSIGN(aZeroMask, DivModTmpAddr(tmp, offset, tileShapeSize, 3, sizeof(float)));
    pto::TASSIGN(scratchTile, DivModTmpAddr(tmp, offset + DivModMaskCols<tileW>() - DIVMOD_SCRATCH_TAIL_BYTES,
                                            tileShapeSize, 4, sizeof(float)));

    pto::TCVT(tmp0Tile, src0Tile, pto::RoundMode::CAST_NONE);
    pto::TCVT(tmp1Tile, src1Tile, pto::RoundMode::CAST_NONE);
    pto::TCVT(tmp2Tile, tmp0Tile, pto::RoundMode::CAST_NONE);
    pto::TCVT(tmp3Tile, tmp1Tile, pto::RoundMode::CAST_NONE);
    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(tmp2Tile, tmp2Tile, tmp3Tile);
    pto::TCVT(qTileHalf, tmp2Tile, pto::RoundMode::CAST_FLOOR);

    // b == 0: q = (a < 0) ? dtype MIN : dtype MAX, r = a
    constexpr half kZeroMax = static_cast<half>(
        static_cast<float>(static_cast<int32_t>(std::numeric_limits<T>::max())));
    constexpr half kZeroMin = static_cast<half>(
        static_cast<float>(static_cast<int32_t>(std::numeric_limits<T>::min())));
    pto::TCMPS(zeroMask, tmp0Tile, static_cast<half>(0.0f), pto::CmpMode::LT);
    pto::TSUB(qSentHalf, tmp0Tile, tmp0Tile);
    pto::TADDS(qSentHalf, qSentHalf, kZeroMin);
    pto::TSELS(qSentHalf, zeroMask, qSentHalf, scratchTile, kZeroMax);
    pto::TCMPS(zeroMask, tmp1Tile, static_cast<half>(0.0f), pto::CmpMode::EQ);
    pto::TSEL(qTileHalf, zeroMask, qSentHalf, qTileHalf, scratchTile);
    pto::TCMPS(aZeroMask, tmp0Tile, static_cast<half>(0.0f), pto::CmpMode::EQ);
    pto::TAND(aZeroMask, aZeroMask, zeroMask);
    pto::TNOT(zeroMask, aZeroMask);
    pto::TSELS(qTileHalf, zeroMask, qTileHalf, scratchTile, static_cast<half>(0.0f));

    pto::TCVT(qTile, qTileHalf, pto::RoundMode::CAST_FLOOR, pto::SaturationMode::OFF);
    pto::TCVT(qTileHalf, qTile,
              pto::RoundMode::CAST_NONE); // This is need, because qTileHalf must be int/uint, for calculate  rTile
    pto::TMUL(rTileHalf, qTileHalf, tmp1Tile);
    pto::TSUB(rTileHalf, tmp0Tile, rTileHalf);
    pto::TSELS(rTileHalf, zeroMask, rTileHalf, scratchTile, kZeroMax);
    pto::TCVT(rTile, rTileHalf, pto::RoundMode::CAST_RINT);
}

template <auto tileH, auto tileW, typename T4, typename Src0Tile, typename Src1Tile, typename DstTile>
TILEOP void DivModInt8Compute(DstTile qTile, DstTile rTile, Src0Tile src0Tile, Src1Tile src1Tile, T4 tmp, size_t offset,
                              size_t dstShape3, size_t dstShape4, size_t tileShapeSize)
{
#ifdef __DAV_V220
    DivModV220Int8Compute<tileH, tileW>(qTile, rTile, src0Tile, src1Tile, tmp, offset, dstShape3, dstShape4,
                                        tileShapeSize);
#else
    DivModNonV220Int8Compute<tileH, tileW>(qTile, rTile, src0Tile, src1Tile, tmp, offset, dstShape3, dstShape4,
                                           tileShapeSize);
#endif
}

template <auto tileH, auto tileW, typename T4, typename Src0Tile, typename Src1Tile, typename DstTile>
TILEOP void DivModNonV220Uint8Compute(DstTile qTile, DstTile rTile, Src0Tile src0Tile, Src1Tile src1Tile, T4 tmp,
                                      size_t offset, size_t dstShape3, size_t dstShape4, size_t tileShapeSize)
{
    using HalfTileDefine = pto::Tile<pto::TileType::Vec, half, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using Int16TileDefine = pto::Tile<pto::TileType::Vec, int16_t, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using MaskTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, DivModMaskCols<tileW>(),
                                     pto::BLayout::RowMajor, -1, -1>;

    HalfTileDefine tmp0Tile(dstShape3, dstShape4);
    HalfTileDefine tmp1Tile(dstShape3, dstShape4);
    HalfTileDefine qTileHalf(dstShape3, dstShape4);
    HalfTileDefine rTileHalf(dstShape3, dstShape4);
    HalfTileDefine scratchTile(dstShape3, dstShape4);
    Int16TileDefine tmp2Tile(dstShape3, dstShape4);
    Int16TileDefine tmp3Tile(dstShape3, dstShape4);
    MaskTileDefine zeroMask(dstShape3, dstShape4);
    MaskTileDefine aZeroMask(dstShape3, dstShape4);
    pto::TASSIGN(tmp0Tile, DivModTmpAddr(tmp, offset, tileShapeSize, 0, sizeof(float)));
    pto::TASSIGN(tmp1Tile, DivModTmpAddr(tmp, offset, tileShapeSize, 1, sizeof(float)));
    pto::TASSIGN(tmp2Tile, DivModTmpAddr(tmp, offset, tileShapeSize, 2, sizeof(float)));
    pto::TASSIGN(tmp3Tile, DivModTmpAddr(tmp, offset, tileShapeSize, 3, sizeof(float)));
    pto::TASSIGN(qTileHalf, DivModTmpAddr(tmp, offset, tileShapeSize, 2, sizeof(float)));
    pto::TASSIGN(rTileHalf, DivModTmpAddr(tmp, offset, tileShapeSize, 3, sizeof(float)));
    pto::TASSIGN(zeroMask, DivModTmpAddr(tmp, offset, tileShapeSize, 4, sizeof(float)));
    pto::TASSIGN(aZeroMask, DivModTmpAddr(tmp, offset, tileShapeSize, 3, sizeof(float)));
    pto::TASSIGN(scratchTile, DivModTmpAddr(tmp, offset + DivModMaskCols<tileW>() - DIVMOD_SCRATCH_TAIL_BYTES,
                                            tileShapeSize, 4, sizeof(float)));

    pto::TCVT(tmp0Tile, src0Tile, pto::RoundMode::CAST_NONE);
    pto::TCVT(tmp1Tile, src1Tile, pto::RoundMode::CAST_NONE);
    pto::TCVT(tmp2Tile, tmp0Tile, pto::RoundMode::CAST_NONE);
    pto::TCVT(tmp3Tile, tmp1Tile, pto::RoundMode::CAST_NONE);
    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(tmp2Tile, tmp2Tile, tmp3Tile);

    // b == 0: q = dtype MAX, r = a
    pto::TCMPS(zeroMask, tmp1Tile, static_cast<half>(0.0f), pto::CmpMode::NE);
    pto::TSELS(tmp2Tile, zeroMask, tmp2Tile, scratchTile, UINT8_DIVMOD_ZERO_DIVISOR_RESULT);
    pto::TCMPS(aZeroMask, tmp0Tile, static_cast<half>(0.0f), pto::CmpMode::EQ);
    pto::TNOT(zeroMask, zeroMask);
    pto::TAND(aZeroMask, aZeroMask, zeroMask);
    pto::TNOT(zeroMask, aZeroMask);
    pto::TSELS(tmp2Tile, zeroMask, tmp2Tile, scratchTile, static_cast<int16_t>(0));

    pto::TCVT(qTile, tmp2Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
    pto::TCVT(qTileHalf, qTile, pto::RoundMode::CAST_NONE);
    pto::TMUL(rTileHalf, qTileHalf, tmp1Tile);
    pto::TSUB(rTileHalf, tmp0Tile, rTileHalf);
    pto::TSELS(rTileHalf, zeroMask, rTileHalf, scratchTile,
               static_cast<half>(static_cast<float>(UINT8_DIVMOD_ZERO_DIVISOR_RESULT)));
    pto::TCVT(rTile, rTileHalf, pto::RoundMode::CAST_RINT);
}

template <auto tileH, auto tileW, typename T4, typename Src0Tile, typename Src1Tile, typename DstTile>
TILEOP void DivModUint8Compute(DstTile qTile, DstTile rTile, Src0Tile src0Tile, Src1Tile src1Tile, T4 tmp,
                               size_t offset, size_t dstShape3, size_t dstShape4, size_t tileShapeSize)
{
#ifdef __DAV_V220
    DivModV220Int8Compute<tileH, tileW>(qTile, rTile, src0Tile, src1Tile, tmp, offset, dstShape3, dstShape4,
                                        tileShapeSize);
#else
    DivModNonV220Uint8Compute<tileH, tileW>(qTile, rTile, src0Tile, src1Tile, tmp, offset, dstShape3, dstShape4,
                                            tileShapeSize);
#endif
}

template <auto tileH, auto tileW, typename T4, typename Src0Tile, typename Src1Tile, typename DstTile>
TILEOP void DivModV220Int16Compute(DstTile qTile, DstTile rTile, Src0Tile src0Tile, Src1Tile src1Tile, T4 tmp,
                                   size_t offset, size_t dstShape3, size_t dstShape4, size_t tileShapeSize)
{
    using Fp32TileDefine = pto::Tile<pto::TileType::Vec, float, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using MaskTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, DIVMOD_MASK_WIDTH_SCALE_32BIT * tileW,
                                     pto::BLayout::RowMajor, -1, -1>;
    using I16TileDefine = pto::Tile<pto::TileType::Vec, int16_t, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    Fp32TileDefine tmp0Tile(dstShape3, dstShape4);
    Fp32TileDefine tmp1Tile(dstShape3, dstShape4);
    MaskTileDefine maskTile(dstShape3, dstShape4);
    MaskTileDefine aZeroMask(dstShape3, dstShape4);
    I16TileDefine src0I16(dstShape3, dstShape4);
    I16TileDefine src1I16(dstShape3, dstShape4);
    pto::TASSIGN(tmp0Tile, DivModTmpAddr(tmp, offset, tileShapeSize, 0, sizeof(float)));
    pto::TASSIGN(tmp1Tile, DivModTmpAddr(tmp, offset, tileShapeSize, 1, sizeof(float)));
    pto::TASSIGN(maskTile, DivModTmpAddr(tmp, offset, tileShapeSize, 2, sizeof(float)));
    pto::TASSIGN(aZeroMask, DivModTmpAddr(tmp, offset, tileShapeSize, 1, sizeof(float)));
    pto::TASSIGN(src0I16, (uint64_t)src0Tile.data());
    pto::TASSIGN(src1I16, (uint64_t)src1Tile.data());
    // qSentinel = (a < 0) ? INT16_MIN : INT16_MAX
    pto::TCVT(tmp0Tile, src0Tile, pto::RoundMode::CAST_NONE);
    SyncV();
    pto::TCMPS(maskTile, tmp0Tile, 0.0f, pto::CmpMode::LT);
    SyncV();
    pto::TSUB(qTile, src0Tile, src0Tile);
    SyncV();
    pto::TADDS(qTile, qTile, INT16_MIN_DIVMOD_ZERO_DIVISOR_RESULT);
    SyncV();
    pto::TSELS(qTile, maskTile, qTile, tmp0Tile, INT16_MAX_DIVMOD_ZERO_DIVISOR_RESULT);
    SyncV();

    // zeroMask = (b == 0)
    pto::TCVT(tmp0Tile, src1Tile, pto::RoundMode::CAST_NONE);
    SyncV();
    pto::TCMPS(maskTile, tmp0Tile, 0.0f, pto::CmpMode::EQ);
    SyncV();

    // safeA = zero ? qSentinel : a; safeB = zero ? 1 : b
    pto::TSEL(qTile, maskTile, qTile, src0I16, tmp0Tile);
    SyncV();
    pto::TSUB(rTile, src1Tile, src1Tile);
    SyncV();
    pto::TADDS(rTile, rTile, 1);
    SyncV();
    pto::TSEL(rTile, maskTile, rTile, src1I16, tmp0Tile);
    SyncV();

    // q = floor(float32(safeA) / float32(safeB))
    pto::TCVT(tmp0Tile, qTile, pto::RoundMode::CAST_NONE);
    SyncV();
    pto::TCVT(tmp1Tile, rTile, pto::RoundMode::CAST_NONE);
    SyncV();
    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(tmp0Tile, tmp0Tile, tmp1Tile);
    SyncV();
    pto::TCVT(qTile, tmp0Tile, pto::RoundMode::CAST_FLOOR, pto::SaturationMode::OFF);
    SyncV();

    // r = a - q*b, using the original a/b (b == 0 -> r = a)
    pto::TMUL(rTile, qTile, src1Tile);
    SyncV();
    pto::TSUB(rTile, src0Tile, rTile);
    SyncV();
    pto::TCVT(tmp0Tile, src0Tile, pto::RoundMode::CAST_NONE);
    SyncV();
    pto::TCMPS(aZeroMask, tmp0Tile, 0.0f, pto::CmpMode::EQ);
    SyncV();
    pto::TAND(aZeroMask, aZeroMask, maskTile);
    SyncV();
    pto::TNOT(maskTile, aZeroMask);
    SyncV();
    pto::TSELS(qTile, maskTile, qTile, tmp0Tile, static_cast<int16_t>(0));
    SyncV();
    pto::TSELS(rTile, maskTile, rTile, tmp0Tile, INT16_MAX_DIVMOD_ZERO_DIVISOR_RESULT);
    SyncV();
}

template <auto tileH, auto tileW, typename T4, typename Src0Tile, typename Src1Tile, typename DstTile>
TILEOP void DivModNonV220Int16Compute(DstTile qTile, DstTile rTile, Src0Tile src0Tile, Src1Tile src1Tile, T4 tmp,
                                      size_t offset, size_t dstShape3, size_t dstShape4, size_t tileShapeSize)
{
    using Fp32TileDefine = pto::Tile<pto::TileType::Vec, float, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using MaskTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, DivModMaskCols<tileW>(),
                                     pto::BLayout::RowMajor, -1, -1>;
    using I16TileDefine = pto::Tile<pto::TileType::Vec, int16_t, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    Fp32TileDefine tmp0Tile(dstShape3, dstShape4);
    Fp32TileDefine tmp1Tile(dstShape3, dstShape4);
    MaskTileDefine maskTile(dstShape3, dstShape4);
    MaskTileDefine aZeroMask(dstShape3, dstShape4);
    I16TileDefine src0I16(dstShape3, dstShape4);
    I16TileDefine src1I16(dstShape3, dstShape4);
    pto::TASSIGN(tmp0Tile, DivModTmpAddr(tmp, offset, tileShapeSize, 0, sizeof(float)));
    pto::TASSIGN(tmp1Tile, DivModTmpAddr(tmp, offset, tileShapeSize, 1, sizeof(float)));
    pto::TASSIGN(maskTile, DivModTmpAddr(tmp, offset, tileShapeSize, 2, sizeof(float)));
    pto::TASSIGN(aZeroMask, DivModTmpAddr(tmp, offset, tileShapeSize, 1, sizeof(float)));
    pto::TASSIGN(src0I16, (uint64_t)src0Tile.data());
    pto::TASSIGN(src1I16, (uint64_t)src1Tile.data());
    // qSentinel = (a < 0) ? INT16_MIN : INT16_MAX
    pto::TCVT(tmp0Tile, src0Tile, pto::RoundMode::CAST_NONE);
    pto::TCMPS(maskTile, tmp0Tile, 0.0f, pto::CmpMode::LT);
    pto::TSUB(qTile, src0Tile, src0Tile);
    pto::TADDS(qTile, qTile, INT16_MIN_DIVMOD_ZERO_DIVISOR_RESULT);
    pto::TSELS(qTile, maskTile, qTile, tmp0Tile, INT16_MAX_DIVMOD_ZERO_DIVISOR_RESULT);

    // zeroMask = (b == 0)
    pto::TCVT(tmp0Tile, src1Tile, pto::RoundMode::CAST_NONE);
    pto::TCMPS(maskTile, tmp0Tile, 0.0f, pto::CmpMode::EQ);

    // safeA = zero ? qSentinel : a; safeB = zero ? 1 : b
    pto::TSEL(qTile, maskTile, qTile, src0I16, tmp0Tile);
    pto::TSUB(rTile, src1Tile, src1Tile);
    pto::TADDS(rTile, rTile, 1);
    pto::TSEL(rTile, maskTile, rTile, src1I16, tmp0Tile);

    // q = floor(float32(safeA) / float32(safeB))
    pto::TCVT(tmp0Tile, qTile, pto::RoundMode::CAST_NONE);
    pto::TCVT(tmp1Tile, rTile, pto::RoundMode::CAST_NONE);
    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(tmp0Tile, tmp0Tile, tmp1Tile);
    pto::TCVT(qTile, tmp0Tile, pto::RoundMode::CAST_FLOOR, pto::SaturationMode::OFF);

    // r = a - q*b, using the original a/b (b == 0 -> r = a)
    pto::TMUL(rTile, qTile, src1Tile);
    pto::TSUB(rTile, src0Tile, rTile);
    pto::TCVT(tmp0Tile, src0Tile, pto::RoundMode::CAST_NONE);
    pto::TCMPS(aZeroMask, tmp0Tile, 0.0f, pto::CmpMode::EQ);
    pto::TAND(aZeroMask, aZeroMask, maskTile);
    pto::TNOT(maskTile, aZeroMask);
    pto::TSELS(qTile, maskTile, qTile, tmp0Tile, static_cast<int16_t>(0));
    pto::TSELS(rTile, maskTile, rTile, tmp0Tile, INT16_MAX_DIVMOD_ZERO_DIVISOR_RESULT);
}

template <auto tileH, auto tileW, typename T4, typename Src0Tile, typename Src1Tile, typename DstTile>
TILEOP void DivModInt16Compute(DstTile qTile, DstTile rTile, Src0Tile src0Tile, Src1Tile src1Tile, T4 tmp,
                               size_t offset, size_t dstShape3, size_t dstShape4, size_t tileShapeSize)
{
#ifdef __DAV_V220
    DivModV220Int16Compute<tileH, tileW>(qTile, rTile, src0Tile, src1Tile, tmp, offset, dstShape3, dstShape4,
                                         tileShapeSize);
#else
    DivModNonV220Int16Compute<tileH, tileW>(qTile, rTile, src0Tile, src1Tile, tmp, offset, dstShape3, dstShape4,
                                            tileShapeSize);
#endif
}

template <typename DstTile, typename SrcTile, typename ScratchTile, typename MaskTile, typename TmpTile>
TILEOP void DivModUint16Normalize(DstTile dst, SrcTile src, ScratchTile scratch, MaskTile mask, TmpTile tmp)
{
    pto::TCVT(dst, src, pto::RoundMode::CAST_NONE);
    SyncV();
    pto::TCMPS(mask, dst, 0.0f, pto::CmpMode::LT);
    SyncV();
    pto::TADDS(scratch, dst, UINT16_DIVMOD_NORMALIZE_OFFSET);
    SyncV();
    pto::TSEL(dst, mask, scratch, dst, tmp);
    SyncV();
}

template <auto tileH, auto tileW, typename T4, typename Src0Tile, typename Src1Tile, typename DstTile>
TILEOP void DivModV220Uint16Compute(DstTile qTile, DstTile rTile, Src0Tile src0I16, Src1Tile src1I16, T4 tmp,
                                    size_t offset, size_t dstShape3, size_t dstShape4, size_t tileShapeSize)
{
    using Fp32TileDefine = pto::Tile<pto::TileType::Vec, float, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using I16TileDefine = pto::Tile<pto::TileType::Vec, int16_t, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using MaskTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, DIVMOD_MASK_WIDTH_SCALE_32BIT * tileW,
                                     pto::BLayout::RowMajor, -1, -1>;
    Fp32TileDefine src0Fp32(dstShape3, dstShape4);
    Fp32TileDefine src1Fp32(dstShape3, dstShape4);
    Fp32TileDefine qFp32(dstShape3, dstShape4);
    Fp32TileDefine rFp32(dstShape3, dstShape4);
    Fp32TileDefine scratchFp32(dstShape3, dstShape4);
    MaskTileDefine zeroMask(dstShape3, dstShape4);
    MaskTileDefine aZeroMask(dstShape3, dstShape4);
    I16TileDefine qI16(dstShape3, dstShape4);
    I16TileDefine rI16(dstShape3, dstShape4);
    pto::TASSIGN(src0Fp32, DivModTmpAddr(tmp, offset, tileShapeSize, 0, sizeof(float)));
    pto::TASSIGN(src1Fp32, DivModTmpAddr(tmp, offset, tileShapeSize, 1, sizeof(float)));
    pto::TASSIGN(qFp32, DivModTmpAddr(tmp, offset, tileShapeSize, 2, sizeof(float)));
    pto::TASSIGN(rFp32, DivModTmpAddr(tmp, offset, tileShapeSize, 3, sizeof(float)));
    pto::TASSIGN(scratchFp32, DivModTmpAddr(tmp, offset, tileShapeSize, 4, sizeof(float)));
    pto::TASSIGN(zeroMask, DivModTmpAddr(tmp, offset, tileShapeSize, 5, sizeof(float)));
    pto::TASSIGN(aZeroMask, DivModTmpAddr(tmp, offset, tileShapeSize, 4, sizeof(float)));
    pto::TASSIGN(qI16, (uint64_t)qTile.data());
    pto::TASSIGN(rI16, (uint64_t)rTile.data());

    DivModUint16Normalize(src0Fp32, src0I16, rFp32, zeroMask, qFp32);
    DivModUint16Normalize(src1Fp32, src1I16, rFp32, zeroMask, qFp32);

    // zeroMask = (b == 0)
    pto::TCMPS(zeroMask, src1Fp32, 0.0f, pto::CmpMode::EQ);
    SyncV();

    // qSentinel = 0xFFFF; safeA = zero ? 0xFFFF : a; safeB = zero ? 1 : b
    pto::TSUB(qI16, src0I16, src0I16);
    SyncV();
    pto::TADDS(qI16, qI16, UINT16_DIVMOD_ZERO_DIVISOR_RESULT);
    SyncV();
    pto::TSEL(qI16, zeroMask, qI16, src0I16, scratchFp32);
    SyncV();
    pto::TSUB(rI16, src1I16, src1I16);
    SyncV();
    pto::TADDS(rI16, rI16, static_cast<int16_t>(1));
    SyncV();
    pto::TSEL(rI16, zeroMask, rI16, src1I16, scratchFp32);
    SyncV();

    DivModUint16Normalize(src0Fp32, qI16, rFp32, zeroMask, qFp32);
    DivModUint16Normalize(src1Fp32, rI16, rFp32, zeroMask, qFp32);

    // q = floor(a / b)
    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(qFp32, src0Fp32, src1Fp32);
    SyncV();
    pto::TCVT(qFp32, qFp32, pto::RoundMode::CAST_FLOOR);
    SyncV();

    // r_safe = a - q*b; corr = floor(r_safe / b); q += corr
    pto::TMUL(rFp32, qFp32, src1Fp32);
    SyncV();
    pto::TSUB(rFp32, src0Fp32, rFp32);
    SyncV();
    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(scratchFp32, rFp32, src1Fp32);
    SyncV();
    pto::TCVT(scratchFp32, scratchFp32, pto::RoundMode::CAST_FLOOR);
    SyncV();
    pto::TADD(qFp32, qFp32, scratchFp32);
    SyncV();

    // Final r uses the original a/b: r = a - q*b, so b == 0 gives r = a.
    DivModUint16Normalize(src0Fp32, src0I16, rFp32, zeroMask, scratchFp32);
    DivModUint16Normalize(src1Fp32, src1I16, rFp32, zeroMask, scratchFp32);
    pto::TMUL(rFp32, qFp32, src1Fp32);
    SyncV();
    pto::TSUB(rFp32, src0Fp32, rFp32);
    SyncV();
    pto::TCVT(qI16, qFp32, pto::RoundMode::CAST_RINT, pto::SaturationMode::OFF);
    SyncV();
    pto::TCVT(rI16, rFp32, pto::RoundMode::CAST_RINT, pto::SaturationMode::OFF);
    SyncV();
    pto::TCMPS(zeroMask, src1Fp32, 0.0f, pto::CmpMode::EQ);
    SyncV();
    pto::TCMPS(aZeroMask, src0Fp32, 0.0f, pto::CmpMode::EQ);
    SyncV();
    pto::TAND(aZeroMask, aZeroMask, zeroMask);
    SyncV();
    pto::TNOT(zeroMask, aZeroMask);
    SyncV();
    pto::TSELS(qI16, zeroMask, qI16, qFp32, static_cast<int16_t>(0));
    SyncV();
    pto::TSELS(rI16, zeroMask, rI16, qFp32, UINT16_DIVMOD_ZERO_DIVISOR_RESULT);
    SyncV();
}

template <auto tileH, auto tileW, typename T4, typename Src0Tile, typename Src1Tile, typename DstTile>
TILEOP void DivModNonV220UnsignedCompute(DstTile qTile, DstTile rTile, Src0Tile src0Tile, Src1Tile src1Tile, T4 tmp,
                                         size_t offset, size_t dstShape3, size_t dstShape4, size_t tileShapeSize)
{
    using MaskTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, DivModMaskCols<tileW>(),
                                     pto::BLayout::RowMajor, -1, -1>;
    MaskTileDefine zeroMask(dstShape3, dstShape4);
    MaskTileDefine aZeroMask(dstShape3, dstShape4);
    MaskTileDefine scratchMask(dstShape3, dstShape4);
    pto::TASSIGN(zeroMask, DivModTmpAddr(tmp, offset, tileShapeSize, 0, sizeof(float)));
    pto::TASSIGN(aZeroMask, DivModTmpAddr(tmp, offset, tileShapeSize, 1, sizeof(float)));
    pto::TASSIGN(scratchMask, DivModTmpAddr(tmp, offset + DivModMaskCols<tileW>() - DIVMOD_SCRATCH_TAIL_BYTES,
                                            tileShapeSize, 0, sizeof(float)));
    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(qTile, src0Tile, src1Tile);
    pto::TMUL(rTile, qTile, src1Tile);
    pto::TSUB(rTile, src0Tile, rTile);
    pto::TCMPS(zeroMask, src1Tile, 0, pto::CmpMode::EQ);
    pto::TCMPS(aZeroMask, src0Tile, 0, pto::CmpMode::EQ);
    pto::TAND(aZeroMask, aZeroMask, zeroMask);
    pto::TNOT(zeroMask, zeroMask);
    pto::TSELS(qTile, zeroMask, qTile, scratchMask, std::numeric_limits<typename DstTile::DType>::max());
    pto::TNOT(zeroMask, aZeroMask);
    pto::TSELS(qTile, zeroMask, qTile, scratchMask, static_cast<typename DstTile::DType>(0));
    pto::TSELS(rTile, zeroMask, rTile, scratchMask, std::numeric_limits<typename DstTile::DType>::max());
}

template <auto tileH, auto tileW, typename T4, typename Src0Tile, typename Src1Tile, typename DstTile>
TILEOP void DivModUint16Compute(DstTile qTile, DstTile rTile, Src0Tile src0Tile, Src1Tile src1Tile, T4 tmp,
                                size_t offset, size_t dstShape3, size_t dstShape4, size_t tileShapeSize)
{
#ifdef __DAV_V220
    using I16TileDefine = pto::Tile<pto::TileType::Vec, int16_t, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    I16TileDefine src0I16(dstShape3, dstShape4);
    I16TileDefine src1I16(dstShape3, dstShape4);
    pto::TASSIGN(src0I16, (uint64_t)src0Tile.data());
    pto::TASSIGN(src1I16, (uint64_t)src1Tile.data());
    DivModV220Uint16Compute<tileH, tileW>(qTile, rTile, src0I16, src1I16, tmp, offset, dstShape3, dstShape4,
                                          tileShapeSize);
#else
    DivModNonV220UnsignedCompute<tileH, tileW>(qTile, rTile, src0Tile, src1Tile, tmp, offset, dstShape3, dstShape4,
                                               tileShapeSize);
#endif
}

template <auto tileH, auto tileW, typename T4, typename Src0Tile, typename Src1Tile, typename DstTile>
TILEOP void DivModV220Int32Compute(DstTile qTile, DstTile rTile, Src0Tile src0Tile, Src1Tile src1Tile, T4 tmp,
                                   size_t offset, size_t dstShape3, size_t dstShape4, size_t tileShapeSize)
{
    using Fp32TileDefine = pto::Tile<pto::TileType::Vec, float, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using Int32TileDefine = pto::Tile<pto::TileType::Vec, int32_t, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using MaskTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, DIVMOD_MASK_WIDTH_SCALE_32BIT * tileW,
                                     pto::BLayout::RowMajor, -1, -1>;

    Fp32TileDefine tmp0Fp32Tile(dstShape3, dstShape4);
    Fp32TileDefine tmp1Fp32Tile(dstShape3, dstShape4);
    Fp32TileDefine tmp2Fp32Tile(dstShape3, dstShape4);
    Fp32TileDefine tmp3Fp32Tile(dstShape3, dstShape4);
    Fp32TileDefine tmp5Fp32Tile(dstShape3, dstShape4);
    Int32TileDefine tmp0I32Tile(dstShape3, dstShape4);
    Int32TileDefine tmp2I32Tile(dstShape3, dstShape4);
    Int32TileDefine tmp3I32Tile(dstShape3, dstShape4);
    Int32TileDefine tmp4I32Tile(dstShape3, dstShape4);
    Int32TileDefine tmp5I32Tile(dstShape3, dstShape4);
    MaskTileDefine tmp1MaskTile(dstShape3, dstShape4);
    MaskTileDefine tmp3MaskTile(dstShape3, dstShape4);
    Int32TileDefine src0I32(dstShape3, dstShape4);
    Int32TileDefine src1I32(dstShape3, dstShape4);

    pto::TASSIGN(tmp0Fp32Tile, DivModTmpAddr(tmp, offset, tileShapeSize, 0, sizeof(int32_t)));
    pto::TASSIGN(tmp0I32Tile, DivModTmpAddr(tmp, offset, tileShapeSize, 0, sizeof(int32_t)));
    pto::TASSIGN(tmp1Fp32Tile, DivModTmpAddr(tmp, offset, tileShapeSize, 1, sizeof(int32_t)));
    pto::TASSIGN(tmp1MaskTile, DivModTmpAddr(tmp, offset, tileShapeSize, 1, sizeof(int32_t)));
    pto::TASSIGN(tmp2Fp32Tile, DivModTmpAddr(tmp, offset, tileShapeSize, 2, sizeof(int32_t)));
    pto::TASSIGN(tmp2I32Tile, DivModTmpAddr(tmp, offset, tileShapeSize, 2, sizeof(int32_t)));
    pto::TASSIGN(tmp3Fp32Tile, DivModTmpAddr(tmp, offset, tileShapeSize, 3, sizeof(int32_t)));
    pto::TASSIGN(tmp3I32Tile, DivModTmpAddr(tmp, offset, tileShapeSize, 3, sizeof(int32_t)));
    pto::TASSIGN(tmp3MaskTile, DivModTmpAddr(tmp, offset, tileShapeSize, 3, sizeof(int32_t)));
    pto::TASSIGN(tmp4I32Tile, DivModTmpAddr(tmp, offset, tileShapeSize, 4, sizeof(int32_t)));
    pto::TASSIGN(tmp5Fp32Tile, DivModTmpAddr(tmp, offset, tileShapeSize, 5, sizeof(int32_t)));
    pto::TASSIGN(tmp5I32Tile, DivModTmpAddr(tmp, offset, tileShapeSize, 5, sizeof(int32_t)));
    pto::TASSIGN(src0I32, (uint64_t)src0Tile.data());
    pto::TASSIGN(src1I32, (uint64_t)src1Tile.data());

    // b == 0: safeA = (a >= 0) ? INT32_MAX : INT32_MIN, safeB = 1.
    // Non-zero lanes keep the original a/b.
    pto::TCVT(tmp2Fp32Tile, src1Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
    SyncV();
    pto::TCMPS(tmp1MaskTile, tmp2Fp32Tile, 0.0f, pto::CmpMode::EQ);
    SyncV();
    pto::TCVT(tmp2Fp32Tile, src0Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
    SyncV();
    pto::TCMPS(tmp3MaskTile, tmp2Fp32Tile, 0.0f, pto::CmpMode::LT);
    SyncV();
    pto::TSUB(tmp4I32Tile, src0Tile, src0Tile);
    SyncV();
    pto::TADDS(tmp4I32Tile, tmp4I32Tile, INT32_MAX_DIVMOD_ZERO_DIVISOR_RESULT);
    SyncV();
    pto::TSUB(tmp0I32Tile, src0Tile, src0Tile);
    SyncV();
    pto::TADDS(tmp0I32Tile, tmp0I32Tile, INT32_MIN_DIVMOD_ZERO_DIVISOR_RESULT);
    SyncV();
    pto::TSEL(tmp0I32Tile, tmp3MaskTile, tmp0I32Tile, tmp4I32Tile, tmp5I32Tile);
    SyncV();
    pto::TSEL(qTile, tmp1MaskTile, tmp0I32Tile, src0I32, tmp5I32Tile);
    SyncV();
    pto::TSUB(tmp4I32Tile, src0Tile, src0Tile);
    SyncV();
    pto::TADDS(tmp4I32Tile, tmp4I32Tile, 1);
    SyncV();
    pto::TSEL(rTile, tmp1MaskTile, tmp4I32Tile, src1I32, tmp5I32Tile);
    SyncV();

    // Step 1: qTemp = floor(float32(safeA) / float32(safeB)).
    pto::TCVT(tmp2Fp32Tile, qTile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
    SyncV();
    pto::TCVT(tmp1Fp32Tile, rTile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
    SyncV();
    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(tmp2Fp32Tile, tmp2Fp32Tile, tmp1Fp32Tile);
    SyncV();
    pto::TCVT(tmp0I32Tile, tmp2Fp32Tile, pto::RoundMode::CAST_FLOOR);
    SyncV();

    // Step 2: r = safeA - qTemp * safeB.
    pto::TMUL(tmp2I32Tile, tmp0I32Tile, rTile);
    SyncV();
    pto::TSUB(tmp2I32Tile, qTile, tmp2I32Tile);
    SyncV();

    // Step 3: refine qTemp with floor(float32(r) / float32(safeB)).
    pto::TCVT(tmp3Fp32Tile, tmp2I32Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
    SyncV();
    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(tmp3Fp32Tile, tmp3Fp32Tile, tmp1Fp32Tile);
    SyncV();
    pto::TCVT(tmp2I32Tile, tmp3Fp32Tile, pto::RoundMode::CAST_FLOOR);
    SyncV();
    pto::TADD(tmp0I32Tile, tmp0I32Tile, tmp2I32Tile);
    SyncV();

    // Step 5: r2 = safeA - qTemp * safeB.
    pto::TMUL(tmp2I32Tile, tmp0I32Tile, rTile);
    SyncV();
    pto::TSUB(tmp2I32Tile, qTile, tmp2I32Tile);
    SyncV();

    // Step 6: final +/-1 correction, using safeB as the divisor.
    pto::TCVT(tmp5Fp32Tile, rTile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
    SyncV();
    pto::TCMPS(tmp1MaskTile, tmp5Fp32Tile, 0.0f, pto::CmpMode::LT);
    SyncV();

    pto::TMULS(tmp3I32Tile, tmp2I32Tile, -1);
    SyncV();
    pto::TSEL(tmp2I32Tile, tmp1MaskTile, tmp3I32Tile, tmp2I32Tile, tmp5I32Tile);
    SyncV();

    pto::TMULS(tmp3I32Tile, rTile, -1);
    SyncV();
    pto::TSEL(tmp3I32Tile, tmp1MaskTile, tmp3I32Tile, rTile, tmp5I32Tile);
    SyncV();
    pto::TSUB(tmp4I32Tile, tmp3I32Tile, tmp3I32Tile);
    SyncV();

    pto::TCVT(tmp5Fp32Tile, tmp2I32Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
    SyncV();
    pto::TCMPS(tmp1MaskTile, tmp5Fp32Tile, 0.0f, pto::CmpMode::LT);
    SyncV();

    pto::TSEL(rTile, tmp1MaskTile, tmp4I32Tile, tmp2I32Tile, tmp5I32Tile);
    SyncV();
    pto::TSUB(tmp5I32Tile, rTile, tmp3I32Tile);
    SyncV();

    pto::TADDS(tmp3I32Tile, tmp4I32Tile, -1);
    SyncV();
    pto::TSEL(rTile, tmp1MaskTile, tmp3I32Tile, tmp4I32Tile, tmp2I32Tile);
    SyncV();

    pto::TCVT(tmp2Fp32Tile, tmp5I32Tile, pto::RoundMode::CAST_NONE, pto::SaturationMode::OFF);
    SyncV();
    pto::TCMPS(tmp1MaskTile, tmp2Fp32Tile, 0.0f, pto::CmpMode::GE);
    SyncV();
    pto::TADDS(tmp3I32Tile, tmp4I32Tile, 1);
    SyncV();
    pto::TSEL(rTile, tmp1MaskTile, tmp3I32Tile, rTile, tmp2I32Tile);
    SyncV();

    pto::TADD(tmp0I32Tile, tmp0I32Tile, rTile);
    SyncV();
    pto::TADD(qTile, tmp0I32Tile, tmp4I32Tile);
    SyncV();
}

template <auto tileH, auto tileW, typename T4, typename Src0Tile, typename Src1Tile, typename DstTile>
TILEOP void DivModNonV220Int32Compute(DstTile dstTile, Src0Tile src0Tile, Src1Tile src1Tile, T4 tmp, size_t offset,
                                      size_t dstShape3, size_t dstShape4, size_t tileShapeSize)
{
    using Int32TileDefine = pto::Tile<pto::TileType::Vec, int32_t, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using Uint8TileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, DivModMaskCols<tileW>(),
                                      pto::BLayout::RowMajor, -1, -1>;
    Int32TileDefine tmp0DataTile(dstShape3, dstShape4);
    Int32TileDefine tmp1DataTile(dstShape3, dstShape4);
    Int32TileDefine tmp2DataTile(dstShape3, dstShape4);
    Uint8TileDefine tmp3MaskTile(dstShape3, dstShape4);
    Uint8TileDefine tmp4MaskTile(dstShape3, dstShape4);
    pto::TASSIGN(tmp0DataTile, DivModTmpAddr(tmp, offset, tileShapeSize, 0, sizeof(int32_t)));
    pto::TASSIGN(tmp1DataTile, DivModTmpAddr(tmp, offset, tileShapeSize, 1, sizeof(int32_t)));
    pto::TASSIGN(tmp2DataTile, DivModTmpAddr(tmp, offset, tileShapeSize, 3, sizeof(int32_t)));
    pto::TASSIGN(tmp3MaskTile, DivModTmpAddr(tmp, offset, tileShapeSize, 2, sizeof(int32_t)));
    pto::TASSIGN(tmp4MaskTile, DivModTmpAddr(tmp, offset, tileShapeSize, 3, sizeof(int32_t)));

    DivModNonV220Int<INT32_MAX_DIVMOD_ZERO_DIVISOR_RESULT, INT32_MIN_DIVMOD_ZERO_DIVISOR_RESULT>(
        dstTile, src0Tile, src1Tile, tmp0DataTile, tmp1DataTile, tmp2DataTile, tmp3MaskTile, tmp4MaskTile);
}

template <auto tileH, auto tileW, typename T4, typename Src0Tile, typename Src1Tile, typename DstTile>
TILEOP void DivModInt32Compute(DstTile qTile, DstTile rTile, Src0Tile src0Tile, Src1Tile src1Tile, T4 tmp,
                               size_t offset, size_t dstShape3, size_t dstShape4, size_t tileShapeSize)
{
#ifdef __DAV_V220
    DivModV220Int32Compute<tileH, tileW>(qTile, rTile, src0Tile, src1Tile, tmp, offset, dstShape3, dstShape4,
                                         tileShapeSize);
#else
    DivModNonV220Int32Compute<tileH, tileW>(qTile, src0Tile, src1Tile, tmp, offset, dstShape3, dstShape4,
                                            tileShapeSize);
#endif
    SyncV();
    pto::TMUL(rTile, qTile, src1Tile);
    SyncV();
    pto::TSUB(rTile, src0Tile, rTile);
    SyncV();

#ifdef __DAV_V220
    constexpr auto kMaskCols = DIVMOD_MASK_WIDTH_SCALE_32BIT * tileW;
#else
    constexpr auto kMaskCols = DivModMaskCols<tileW>();
#endif
    using MaskTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, kMaskCols, pto::BLayout::RowMajor, -1, -1>;
    using Int32TileDefine = pto::Tile<pto::TileType::Vec, int32_t, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    MaskTileDefine maskB(dstShape3, dstShape4);
    MaskTileDefine maskA(dstShape3, dstShape4);
    Int32TileDefine scratchTile(dstShape3, dstShape4);
    pto::TASSIGN(maskB, DivModTmpAddr(tmp, offset, tileShapeSize, 0, sizeof(int32_t)));
    pto::TASSIGN(maskA, DivModTmpAddr(tmp, offset, tileShapeSize, 1, sizeof(int32_t)));
    pto::TASSIGN(scratchTile, DivModTmpAddr(tmp, offset, tileShapeSize, 2, sizeof(int32_t)));
    pto::TCMPS(maskB, src1Tile, 0, pto::CmpMode::EQ);
    SyncV();
    pto::TCMPS(maskA, src0Tile, 0, pto::CmpMode::EQ);
    SyncV();
    pto::TAND(maskA, maskA, maskB);
    SyncV();
    pto::TNOT(maskB, maskA);
    SyncV();
    pto::TSELS(qTile, maskB, qTile, scratchTile, static_cast<int32_t>(0));
    SyncV();
    pto::TSELS(rTile, maskB, rTile, scratchTile, INT32_MAX_DIVMOD_ZERO_DIVISOR_RESULT);
    SyncV();
}

template <auto tileH, auto tileW, typename T4, typename Src0Tile, typename Src1Tile, typename DstTile>
TILEOP void DivModNonV220Int64MinDivisor(DstTile qTile, Src0Tile src0Tile, Src1Tile src1Tile, T4 tmp, size_t offset,
                                         size_t dstShape3, size_t dstShape4, size_t tileShapeSize)
{
    using Int64TileDefine = pto::Tile<pto::TileType::Vec, int64_t, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using MaskTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, DivModMaskCols<tileW>(),
                                     pto::BLayout::RowMajor, -1, -1>;
    Int64TileDefine qPatch(dstShape3, dstShape4);
    Int64TileDefine constTile(dstShape3, dstShape4);
    MaskTileDefine maskMinDivisor(dstShape3, dstShape4);
    MaskTileDefine maskDividend(dstShape3, dstShape4);

    pto::TASSIGN(qPatch, DivModTmpAddr(tmp, offset, tileShapeSize, 0, sizeof(int64_t)));
    pto::TASSIGN(constTile, DivModTmpAddr(tmp, offset, tileShapeSize, 1, sizeof(int64_t)));
    pto::TASSIGN(maskMinDivisor, DivModTmpAddr(tmp, offset, tileShapeSize, 2, sizeof(int64_t)));
    pto::TASSIGN(maskDividend, DivModTmpAddr(tmp, offset, tileShapeSize, 3, sizeof(int64_t)));

    pto::TCMPS(maskMinDivisor, src1Tile, INT64_MIN_DIVMOD_ZERO_DIVISOR_RESULT, pto::CmpMode::EQ);
    pto::TSUB(qPatch, src0Tile, src0Tile);
    pto::TADDS(constTile, qPatch, static_cast<int64_t>(1));
    pto::TCMPS(maskDividend, src0Tile, INT64_MIN_DIVMOD_ZERO_DIVISOR_RESULT, pto::CmpMode::EQ);
    pto::TAND(maskDividend, maskDividend, maskMinDivisor);
    pto::TSEL(qPatch, maskDividend, constTile, qPatch, qPatch);

    pto::TADDS(constTile, constTile, static_cast<int64_t>(-2));
    pto::TCMPS(maskDividend, src0Tile, static_cast<int64_t>(0), pto::CmpMode::GT);
    pto::TAND(maskDividend, maskDividend, maskMinDivisor);
    pto::TSEL(qPatch, maskDividend, constTile, qPatch, qPatch);
    pto::TSEL(qTile, maskMinDivisor, qPatch, qTile, qPatch);
}

template <auto tileH, auto tileW, typename T4, typename Src0Tile, typename Src1Tile, typename DstTile>
TILEOP void DivModNonV220Int64Compute(DstTile qTile, DstTile rTile, Src0Tile src0Tile, Src1Tile src1Tile, T4 tmp,
                                      size_t offset, size_t dstShape3, size_t dstShape4, size_t tileShapeSize)
{
    using Int64TileDefine = pto::Tile<pto::TileType::Vec, int64_t, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using MaskTileDefine = pto::Tile<pto::TileType::Vec, uint8_t, tileH, DivModMaskCols<tileW>(),
                                     pto::BLayout::RowMajor, -1, -1>;
    Int64TileDefine tmp0DataTile(dstShape3, dstShape4);
    Int64TileDefine tmp1DataTile(dstShape3, dstShape4);
    Int64TileDefine tmp2DataTile(dstShape3, dstShape4);
    MaskTileDefine tmp3MaskTile(dstShape3, dstShape4);
    MaskTileDefine tmp4MaskTile(dstShape3, dstShape4);
    pto::TASSIGN(tmp0DataTile, DivModTmpAddr(tmp, offset, tileShapeSize, 0, sizeof(int64_t)));
    pto::TASSIGN(tmp1DataTile, DivModTmpAddr(tmp, offset, tileShapeSize, 1, sizeof(int64_t)));
    pto::TASSIGN(tmp2DataTile, DivModTmpAddr(tmp, offset, tileShapeSize, 3, sizeof(int64_t)));
    pto::TASSIGN(tmp3MaskTile, DivModTmpAddr(tmp, offset, tileShapeSize, 2, sizeof(int64_t)));
    pto::TASSIGN(tmp4MaskTile, DivModTmpAddr(tmp, offset, tileShapeSize, 3, sizeof(int64_t)));

    DivModNonV220Int<INT64_MAX_DIVMOD_ZERO_DIVISOR_RESULT, INT64_MIN_DIVMOD_ZERO_DIVISOR_RESULT>(
        qTile, src0Tile, src1Tile, tmp0DataTile, tmp1DataTile, tmp2DataTile, tmp3MaskTile, tmp4MaskTile);
    DivModNonV220Int64MinDivisor<tileH, tileW>(qTile, src0Tile, src1Tile, tmp, offset, dstShape3, dstShape4,
                                               tileShapeSize);
    pto::TMUL(rTile, qTile, src1Tile);
    pto::TSUB(rTile, src0Tile, rTile);

    // a == 0 && b == 0: q = 0, r = dtype MAX; other lanes keep the computed q/r.
    pto::TCMPS(tmp3MaskTile, src1Tile, 0, pto::CmpMode::EQ);
    pto::TCMPS(tmp4MaskTile, src0Tile, 0, pto::CmpMode::EQ);
    pto::TAND(tmp3MaskTile, tmp3MaskTile, tmp4MaskTile);
    pto::TSUB(tmp0DataTile, src0Tile, src0Tile);
    pto::TADDS(tmp1DataTile, tmp0DataTile, INT64_MAX_DIVMOD_ZERO_DIVISOR_RESULT);
    pto::TSEL(qTile, tmp3MaskTile, tmp0DataTile, qTile, tmp4MaskTile);
    pto::TSEL(rTile, tmp3MaskTile, tmp1DataTile, rTile, tmp4MaskTile);
}

#define OP_TILE_OP_DIVMOD TDivMod
template <typename T0, typename T1, typename T2, typename T3, typename T4>
TILEOP void TDivMod(T0 dst0Tile, T1 dst1Tile, T2 src0Tile, T3 src1Tile, T4 tmp)
{
    constexpr size_t DIVMOD_TMP_PLANE_COUNT = DivModTmpPlaneCount<typename T0::Type>();
    const auto dstLayout = dst0Tile.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    constexpr auto tileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto tmpPlaneSize = TileOp::GetAnyAxisMergeResult<DIM_1ST, Std::tuple_size<typename T4::TileShape>::value,
                                                                typename T4::TileShape>() /
                                  DIVMOD_TMP_PLANE_COUNT;

    using DataTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, tileH, tileW, pto::BLayout::RowMajor, -1,
                                     -1>;
    auto src0ExecTile = MakeElementwiseOperandExecTile(dst0Tile, src0Tile);
    auto src1ExecTile = MakeElementwiseOperandExecTile(dst0Tile, src1Tile);
    DataTileDefine qutotientTile(dstShape3, dstShape4);
    DataTileDefine remainderTile(dstShape3, dstShape4);

    for (LoopVar n0Index = 0; n0Index < dstShape0; n0Index++) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; n1Index++) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; n2Index++) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto dstByteoffset = TileOp::GetPackedByteOffset<typename T0::Type>(
                    GenTileOffset(dst0Tile, tileOffsets));
                auto tmpByteOffset = TileOp::GetPackedByteOffset<typename T4::Type>(GenTileOffset(tmp, tileOffsets));
                AssignElementwiseOperandExecTile(src0ExecTile, src0Tile, tileOffsets);
                AssignElementwiseOperandExecTile(src1ExecTile, src1Tile, tileOffsets);
                pto::TASSIGN(qutotientTile, (uint64_t)(dst0Tile.GetAddr() + dstByteoffset));
                pto::TASSIGN(remainderTile, (uint64_t)(dst1Tile.GetAddr() + dstByteoffset));

                if constexpr (std::is_same_v<typename T0::Type, int8_t>) {
                    DivModInt8Compute<tileH, tileW>(qutotientTile, remainderTile, src0ExecTile, src1ExecTile, tmp,
                                                    tmpByteOffset, dstShape3, dstShape4, tmpPlaneSize);
                } else if constexpr (std::is_same_v<typename T0::Type, uint8_t>) {
                    DivModUint8Compute<tileH, tileW>(qutotientTile, remainderTile, src0ExecTile, src1ExecTile, tmp,
                                                     tmpByteOffset, dstShape3, dstShape4, tmpPlaneSize);
                } else if constexpr (std::is_same_v<typename T0::Type, int16_t>) {
                    DivModInt16Compute<tileH, tileW>(qutotientTile, remainderTile, src0ExecTile, src1ExecTile, tmp,
                                                     tmpByteOffset, dstShape3, dstShape4, tmpPlaneSize);
                } else if constexpr (std::is_same_v<typename T0::Type, uint16_t>) {
                    DivModUint16Compute<tileH, tileW>(qutotientTile, remainderTile, src0ExecTile, src1ExecTile, tmp,
                                                      tmpByteOffset, dstShape3, dstShape4, tmpPlaneSize);
                } else if constexpr (std::is_same_v<typename T0::Type, int32_t>) {
                    DivModInt32Compute<tileH, tileW>(qutotientTile, remainderTile, src0ExecTile, src1ExecTile, tmp,
                                                     tmpByteOffset, dstShape3, dstShape4, tmpPlaneSize);
                } else if constexpr (std::is_same_v<typename T0::Type, uint32_t> ||
                                     std::is_same_v<typename T0::Type, uint64_t>) {
                    DivModNonV220UnsignedCompute<tileH, tileW>(qutotientTile, remainderTile, src0ExecTile, src1ExecTile,
                                                               tmp, tmpByteOffset, dstShape3, dstShape4, tmpPlaneSize);
                } else {
                    DivModNonV220Int64Compute<tileH, tileW>(qutotientTile, remainderTile, src0ExecTile, src1ExecTile,
                                                            tmp, tmpByteOffset, dstShape3, dstShape4, tmpPlaneSize);
                }
            }
        }
    }
}

#endif // TILEOP_TILE_OPERATOR_VEC_BINARY_DIVMOD_H
