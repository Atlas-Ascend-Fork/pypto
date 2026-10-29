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
 * \file uniform.h
 * \brief Uniform random number generator implementation
 */

#ifndef TILEOP_TILE_OPERATOR_UNIFORM__H
#define TILEOP_TILE_OPERATOR_UNIFORM__H

#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#if defined(PTO_NPU_ARCH_A5) || defined(__CPU_SIM)
#define OP_TILE_OP_UNIFORM TUniform
template <int Size, typename TDst, typename TTmp>
TILEOP void TUniform(TDst dst, TTmp tmpbuf, uint64_t key, uint64_t counter0, uint64_t counter1, uint16_t rounds)
{
    constexpr uint16_t REDUCED_ROUNDS = 7;
    constexpr uint16_t DEFAULT_ROUNDS = 10;
    constexpr size_t COUNTER_WORDS = 2;
    constexpr uint64_t UINT32_LOW_MASK = 0xFFFFFFFFULL;
    constexpr uint32_t UINT32_BIT_WIDTH = sizeof(uint32_t) * TileOp::BITS_PER_BYTE;
    constexpr uint32_t FP32_MANTISSA_MASK = 0x7FFFFF;
    constexpr uint32_t FP32_ONE_BITS = 0x3F800000;
    constexpr uint32_t UINT16_LOW_MASK = 0xFFFF;
    constexpr uint16_t FP16_MANTISSA_MASK = 0x03FF;
    constexpr uint16_t FP16_ONE_BITS = 0x3C00;
    constexpr uint16_t BF16_MANTISSA_MASK = 0x007F;
    constexpr uint16_t BF16_ONE_BITS = 0x3F80;
    const auto validSize = dst.GetLayout().template GetShapeDim<DIM_5TH, MAX_DIMS>();
    constexpr int alignElems = static_cast<int>(TileOp::BLOCK_SIZE / sizeof(uint32_t));
    constexpr int tileW = (Size + alignElems - 1) / alignElems * alignElems;

    uint64_t tileCounter[COUNTER_WORDS] = {counter0, counter1};

    uint64_t tmpbufAddr = tmpbuf.GetAddr();
    __ubuf__ uint32_t* uint32Buffer = reinterpret_cast<__ubuf__ uint32_t*>(tmpbufAddr);

    using TileUint32 = pto::Tile<pto::TileType::Vec, uint32_t, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
    TileUint32 uint32Tile(1, validSize);
    TileUint32 dstUint32Tile(1, validSize);
    pto::TASSIGN(uint32Tile, (uint64_t)uint32Buffer);

    pto::TRandomKey uniformKey = {static_cast<uint32_t>(key & UINT32_LOW_MASK),
                                  static_cast<uint32_t>(key >> UINT32_BIT_WIDTH)};
    pto::TRandomCounter uniformCounter = {static_cast<uint32_t>(tileCounter[0] & UINT32_LOW_MASK),
                                          static_cast<uint32_t>(tileCounter[0] >> UINT32_BIT_WIDTH),
                                          static_cast<uint32_t>(tileCounter[1] & UINT32_LOW_MASK),
                                          static_cast<uint32_t>(tileCounter[1] >> UINT32_BIT_WIDTH)};

    if (rounds == REDUCED_ROUNDS) {
        pto::TRANDOM<REDUCED_ROUNDS>(uint32Tile, uniformKey, uniformCounter);
    } else {
        pto::TRANDOM<DEFAULT_ROUNDS>(uint32Tile, uniformKey, uniformCounter);
    }

    using DstType = typename TDst::Type;
    constexpr bool isFloat = std::is_same_v<DstType, float>;
    constexpr bool isHalf = std::is_same_v<DstType, half>;
    constexpr bool isBfloat16 = std::is_same_v<DstType, bfloat16_t>;

    if constexpr (isFloat) {
        pto::TASSIGN(dstUint32Tile, (uint64_t)dst.GetAddr());

        pto::TANDS(dstUint32Tile, uint32Tile, FP32_MANTISSA_MASK);
        pto::TORS(uint32Tile, dstUint32Tile, FP32_ONE_BITS);

        using TileFloat = pto::Tile<pto::TileType::Vec, float, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
        TileFloat floatTile(1, validSize);
        pto::TASSIGN(floatTile, (uint64_t)uint32Buffer);

        using TileDst = pto::Tile<pto::TileType::Vec, DstType, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
        TileDst dstTile(1, validSize);
        pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr()));
        pto::TSUBS(dstTile, floatTile, 1.0f);
    } else if constexpr (isHalf || isBfloat16) {
        constexpr int64_t uint32BufferBytes = ((Size * sizeof(uint32_t) + TileOp::BLOCK_SIZE - 1) /
                                               TileOp::BLOCK_SIZE) *
                                              TileOp::BLOCK_SIZE;
        __ubuf__ uint32_t* uint32BufferLow = reinterpret_cast<__ubuf__ uint32_t*>(tmpbufAddr + uint32BufferBytes);

        TileUint32 uint32TileLow(1, validSize);
        pto::TASSIGN(uint32TileLow, (uint64_t)uint32BufferLow);

        pto::TANDS(uint32TileLow, uint32Tile, UINT16_LOW_MASK);

        __ubuf__ uint16_t* uint16Buffer = reinterpret_cast<__ubuf__ uint16_t*>(tmpbufAddr);

        using TileUint16 = pto::Tile<pto::TileType::Vec, uint16_t, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
        TileUint16 uint16Tile(1, validSize);
        TileUint16 dstUint16Tile(1, validSize);
        pto::TASSIGN(uint16Tile, (uint64_t)uint16Buffer);
        pto::TASSIGN(dstUint16Tile, (uint64_t)dst.GetAddr());

        pto::TCVT(uint16Tile, uint32TileLow, pto::RoundMode::CAST_NONE);

        if constexpr (isHalf) {
            pto::TANDS(dstUint16Tile, uint16Tile, FP16_MANTISSA_MASK);
            pto::TORS(uint16Tile, dstUint16Tile, FP16_ONE_BITS);
        } else {
            pto::TANDS(dstUint16Tile, uint16Tile, BF16_MANTISSA_MASK);
            pto::TORS(uint16Tile, dstUint16Tile, BF16_ONE_BITS);
        }

        __ubuf__ DstType* resultBuffer = reinterpret_cast<__ubuf__ DstType*>(uint16Buffer);

        using TileResult = pto::Tile<pto::TileType::Vec, DstType, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
        TileResult resultTile(1, validSize);
        pto::TASSIGN(resultTile, (uint64_t)resultBuffer);

        using TileDst = pto::Tile<pto::TileType::Vec, DstType, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
        TileDst dstTile(1, validSize);
        pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr()));

        if constexpr (isHalf) {
            pto::TSUBS(dstTile, resultTile, static_cast<half>(1.0));
        } else {
            pto::TSUBS(dstTile, resultTile, static_cast<bfloat16_t>(1.0));
        }
    }
}
#endif

#endif
