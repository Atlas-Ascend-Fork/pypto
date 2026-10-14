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
 * \file digamma.h
 * \brief Unary tile operation implementations.
 */

#ifndef TILEOP_TILE_OPERATOR_VEC_UNARY_DIGAMMA_H
#define TILEOP_TILE_OPERATOR_VEC_UNARY_DIGAMMA_H

#include "basic.h"
#include "tileop_common.h"
#include "utils/sync.h"
#include <cmath>

constexpr float MIN_NEG_WITH_FLOAT = -8388608.0f;
constexpr float ZERO = 0.0f;
constexpr float ONE = 1.0f;
constexpr float NEG_ONE = -1.0f;
constexpr float TWO = 2.0f;
constexpr float HALF = 0.5f;
constexpr float RECURRENCE_SHIFT = 10.0f;
constexpr float PSI_10 = 2.251752589066721f;
constexpr float DIGAMMA_CORRECTION_MAX = 0x1.0p+57f;
constexpr float SPLIT_FACTOR = 4097.0f; // 2^12 + 1
constexpr float REDUCE_BOUND = 1.3984375f;
constexpr float ASYM_INIT = 1.0f / 12.0f;
// Two-float constants: value ~= main + residual; the last two fields support compensated multiplication.
constexpr float DIGAMMA_CONSTANTS[][TileOp::NUM_VALUE_4] = {
    {0x1.921fb60000000p+1f, -0x1.777a5c0000000p-24f, 0x1.9220000000000p+1f, -0x1.2800000000000p-17f},
    {0x1.0000000000000p+0f, 0x0.0p+0f, 0x1.0000000000000p+0f, 0x0.0p+0f},
    {0x1.0000000000000p-1f, 0x0.0p+0f, 0x1.0000000000000p-1f, 0x0.0p+0f},
    {-0x1.0000000000000p+0f, 0x0.0p+0f, -0x1.0000000000000p+0f, 0x0.0p+0f},
    {0x1.62e4300000000p-1f, -0x1.05c6100000000p-29f, 0x1.62e0000000000p-1f, 0x1.0c00000000000p-15f},
    {-0x1.00ea340000000p-2f, -0x1.1710660000000p-27f, -0x1.00e0000000000p-2f, -0x1.4680000000000p-15f},
    {0x1.5575b00000000p-2f, 0x1.7c016e0000000p-27f, 0x1.5580000000000p-2f, -0x1.4a00000000000p-15f},
    {-0x1.fffff00000000p-2f, 0x1.beb7dc0000000p-27f, -0x1.0000000000000p-1f, 0x1.0000000000000p-22f},
};

constexpr float DIGAMMA_SIN_COEFFICIENTS[][TileOp::NUM_VALUE_4] = {
    {0x1.0000000000000p+0f, 0x0.0p+0f, 0x1.0000000000000p+0f, 0x0.0p+0f},
    {-0x1.5555560000000p-3f, 0x1.5555560000000p-28f, -0x1.5560000000000p-3f, 0x1.5540000000000p-16f},
    {0x1.1111120000000p-7f, -0x1.ddddde0000000p-32f, 0x1.1120000000000p-7f, -0x1.ddc0000000000p-20f},
    {-0x1.a01a020000000p-13f, 0x1.7f97fa0000000p-39f, -0x1.a020000000000p-13f, 0x1.7f80000000000p-27f},
    {0x1.71de3a0000000p-19f, 0x1.55b1cc0000000p-45f, 0x1.71e0000000000p-19f, -0x1.c600000000000p-35f},
    {-0x1.ae64560000000p-26f, -0x1.fd51380000000p-52f, -0x1.ae60000000000p-26f, -0x1.1580000000000p-40f},
    {0x1.6124620000000p-33f, -0x1.8af25e0000000p-58f, 0x1.6120000000000p-33f, 0x1.1880000000000p-47f},
    {-0x1.ae7f3e0000000p-41f, -0x1.ccee080000000p-67f, -0x1.ae80000000000p-41f, 0x1.8400000000000p-58f},
    {0x1.952c780000000p-49f, -0x1.f9ea560000000p-74f, 0x1.9520000000000p-49f, 0x1.8f00000000000p-62f},
    {-0x1.2f49b40000000p-57f, -0x1.a050560000000p-83f, -0x1.2f40000000000p-57f, -0x1.3680000000000p-70f},
};

constexpr float DIGAMMA_COS_COEFFICIENTS[][TileOp::NUM_VALUE_4] = {
    {0x1.0000000000000p+0f, 0x0.0p+0f, 0x1.0000000000000p+0f, 0x0.0p+0f},
    {-0x1.0000000000000p-1f, 0x0.0p+0f, -0x1.0000000000000p-1f, 0x0.0p+0f},
    {0x1.5555560000000p-5f, -0x1.5555560000000p-30f, 0x1.5560000000000p-5f, -0x1.5540000000000p-18f},
    {-0x1.6c16c20000000p-10f, 0x1.27d27e0000000p-35f, -0x1.6c20000000000p-10f, 0x1.27c0000000000p-23f},
    {0x1.a01a020000000p-16f, -0x1.7f97fa0000000p-42f, 0x1.a020000000000p-16f, -0x1.7f80000000000p-30f},
    {-0x1.27e4fc0000000p-22f, 0x1.10ec140000000p-47f, -0x1.27e0000000000p-22f, -0x1.3f00000000000p-36f},
    {0x1.1eed8e0000000p-29f, 0x1.ff1b140000000p-54f, 0x1.1ee0000000000p-29f, 0x1.b1c0000000000p-42f},
    {-0x1.93974a0000000p-37f, -0x1.180f940000000p-62f, -0x1.93a0000000000p-37f, 0x1.16c0000000000p-50f},
    {0x1.ae7f3e0000000p-45f, 0x1.ccee080000000p-71f, 0x1.ae80000000000p-45f, -0x1.8400000000000p-62f},
    {-0x1.6827860000000p-53f, -0x1.dcbecc0000000p-80f, -0x1.6820000000000p-53f, -0x1.e180000000000p-67f},
    {0x1.e542ba0000000p-62f, 0x1.00808a0000000p-88f, 0x1.e540000000000p-62f, 0x1.5d00000000000p-77f},
};

constexpr float DIGAMMA_INVERSE_C_TABLE[][TileOp::NUM_VALUE_4] = {
    {0x1.661ec80000000p+0f, -0x1.81c3100000000p-26f, 0x1.6620000000000p+0f, -0x1.3800000000000p-16f},
    {0x1.571ed40000000p+0f, 0x1.55f1080000000p-25f, 0x1.5720000000000p+0f, -0x1.2c00000000000p-16f},
    {0x1.4953a00000000p+0f, -0x1.e1fdea0000000p-25f, 0x1.4960000000000p+0f, -0x1.8c00000000000p-13f},
    {0x1.3c995c0000000p+0f, -0x1.e8ff900000000p-25f, 0x1.3ca0000000000p+0f, -0x1.a900000000000p-14f},
    {0x1.30d1900000000p+0f, 0x1.910c940000000p-25f, 0x1.30e0000000000p+0f, -0x1.ce00000000000p-13f},
    {0x1.25e2280000000p+0f, -0x1.3d1c580000000p-26f, 0x1.25e0000000000p+0f, 0x1.1400000000000p-15f},
    {0x1.1bb4a40000000p+0f, 0x1.4346880000000p-25f, 0x1.1bc0000000000p+0f, -0x1.6b80000000000p-13f},
    {0x1.1235900000000p+0f, -0x1.eea3480000000p-25f, 0x1.1240000000000p+0f, -0x1.4e00000000000p-13f},
    {0x1.0953f40000000p+0f, 0x1.9900a80000000p-28f, 0x1.0960000000000p+0f, -0x1.8180000000000p-13f},
    {0x1.0000000000000p+0f, 0x0.0p+0f, 0x1.0000000000000p+0f, 0x0.0p+0f},
    {0x1.e608d00000000p-1f, -0x1.32dc2a0000000p-28f, 0x1.e600000000000p-1f, 0x1.1a00000000000p-14f},
    {0x1.ca4b320000000p-1f, -0x1.fb2ac00000000p-30f, 0x1.ca40000000000p-1f, 0x1.6640000000000p-14f},
    {0x1.b203660000000p-1f, -0x1.12a0640000000p-26f, 0x1.b200000000000p-1f, 0x1.b300000000000p-16f},
    {0x1.9c2d160000000p-1f, 0x1.d0d5160000000p-28f, 0x1.9c2000000000p-1f, 0x1.a2c0000000000p-14f},
    {0x1.886e600000000p-1f, 0x1.bc20f60000000p-28f, 0x1.8860000000000p-1f, 0x1.cc00000000000p-14f},
    {0x1.767dd00000000p-1f, -0x1.5596f40000000p-26f, 0x1.7680000000000p-1f, -0x1.1800000000000p-16f},
};

constexpr float DIGAMMA_LOG_C_TABLE[][TileOp::NUM_VALUE_4] = {
    {-0x1.57bf780000000p-2f, -0x1.1955bc0000000p-31f, -0x1.57c0000000000p-2f, 0x1.1000000000000p-19f},
    {-0x1.2bef0a0000000p-2f, -0x1.f01b760000000p-28f, -0x1.2be0000000000p-2f, -0x1.e140000000000p-15f},
    {-0x1.01eae80000000p-2f, 0x1.5d8b320000000p-31f, -0x1.01e0000000000p-2f, -0x1.5d00000000000p-15f},
    {-0x1.b31d8a0000000p-3f, -0x1.a0893a0000000p-29f, -0x1.b320000000000p-3f, 0x1.3b00000000000p-18f},
    {-0x1.6574f00000000p-3f, -0x1.580eec0000000p-28f, -0x1.6580000000000p-3f, 0x1.6200000000000p-16f},
    {-0x1.1aa2bc0000000p-3f, -0x1.e720400000000p-29f, -0x1.1aa0000000000p-3f, -0x1.5e00000000000p-18f},
    {-0x1.a4e76c0000000p-4f, -0x1.d181cc0000000p-29f, -0x1.a4e0000000000p-4f, -0x1.db00000000000p-18f},
    {-0x1.1973c60000000p-4f, 0x1.67b8cc0000000p-30f, -0x1.1980000000000p-4f, 0x1.8740000000000p-17f},
    {-0x1.252f440000000p-5f, 0x1.c7bcf80000000p-31f, -0x1.2520000000000p-5f, -0x1.e880000000000p-18f},
    {0x0.0p+0f, 0x0.0p+0f, 0x0.0p+0f, 0x0.0p+0f},
    {0x1.aa5aa60000000p-5f, -0x1.06d33e0000000p-32f, 0x1.aa6000000000p-5f, -0x1.5680000000000p-19f},
    {0x1.c5e53a0000000p-4f, 0x1.46c5d60000000p-29f, 0x1.c5e0000000000p-4f, 0x1.4e80000000000p-18f},
    {0x1.526e580000000p-3f, -0x1.1be4a00000000p-28f, 0x1.5260000000000p-3f, 0x1.cb00000000000p-16f},
    {0x1.bc28600000000p-3f, 0x1.a448ee0000000p-28f, 0x1.bc2000000000p-3f, 0x1.0c00000000000p-16f},
    {0x1.1058bc0000000p-2f, 0x1.140fdc0000000p-27f, 0x1.1060000000000p-2f, -0x1.d100000000000p-16f},
    {0x1.4043060000000p-2f, -0x1.09223e0000000p-27f, 0x1.4040000000000p-2f, 0x1.8300000000000p-17f},
};

constexpr float DIGAMMA_LOOKUP_BOUNDARIES[] = {
    0x1.6600000000000p-1f, 0x1.7600000000000p-1f, 0x1.8600000000000p-1f, 0x1.9600000000000p-1f,
    0x1.a600000000000p-1f, 0x1.b600000000000p-1f, 0x1.c600000000000p-1f, 0x1.d600000000000p-1f,
    0x1.e600000000000p-1f, 0x1.f600000000000p-1f, 0x1.0600000000000p+0f, 0x1.1600000000000p+0f,
    0x1.2600000000000p+0f, 0x1.3600000000000p+0f, 0x1.4600000000000p+0f, 0x1.5600000000000p+0f,
};
constexpr float DIGAMMA_REDUCE_BOUNDARIES[] = {
    0x1.0p+64f, 0x1.0p+32f, 0x1.0p+16f, 0x1.0p+8f, 0x1.0p+4f, 0x1.0p+2f, 0x1.0p+1f,
};
constexpr float DIGAMMA_REDUCE_SCALES[] = {
    0x1.0p-64f, 0x1.0p-32f, 0x1.0p-16f, 0x1.0p-8f, 0x1.0p-4f, 0x1.0p-2f, 0x1.0p-1f,
};
constexpr float DIGAMMA_REDUCE_EXPONENTS[] = {
    64.0f, 32.0f, 16.0f, 8.0f, 4.0f, 2.0f, 1.0f,
};
constexpr float DIGAMMA_ASYMPTOTIC_COEFFICIENTS[] = {
    -691.0f / 32760.0f, 1.0f / 132.0f, -1.0f / 240.0f, 1.0f / 252.0f, -1.0f / 120.0f, 1.0f / 12.0f,
};
template <typename TileType>
TILEOP void DigammaSet(TileType outHi, TileType outLo, const float* value)
{
    // Load a two-float constant.
    pto::TEXPANDS(outHi, value[TileOp::NUM_VALUE_0]);
    SyncV();
    pto::TEXPANDS(outLo, value[TileOp::NUM_VALUE_1]);
    SyncV();
}

template <typename TileType>
TILEOP void DigammaLift(TileType outHi, TileType outLo, TileType src)
{
    // Lift fp32 to (src, 0).
    pto::TMULS(outHi, src, ONE);
    SyncV();
    pto::TEXPANDS(outLo, ZERO);
    SyncV();
}

template <typename TileType>
TILEOP void DigammaNormalize(TileType outHi, TileType outLo, TileType value, TileType error, TileType tmp6,
                             TileType tmp7)
{
    // Normalize: outHi + outLo ~= value + error.
    pto::TADD(outHi, value, error);
    SyncV();
    pto::TSUB(tmp6, outHi, value);
    SyncV();
    pto::TSUB(tmp7, outHi, tmp6);
    SyncV();
    pto::TSUB(tmp7, value, tmp7);
    SyncV();
    pto::TSUB(tmp6, error, tmp6);
    SyncV();
    pto::TADD(outLo, tmp7, tmp6);
    SyncV();
}

template <typename TileType>
TILEOP void DigammaAdd(TileType outHi, TileType outLo, TileType lhsHi, TileType lhsLo, TileType rhsHi, TileType rhsLo,
                       TileType tmp0, TileType tmp1, TileType tmp2, TileType tmp6, TileType tmp7)
{
    // Compensated two-float addition.
    pto::TADD(tmp0, lhsHi, rhsHi);
    SyncV();
    pto::TSUB(tmp1, tmp0, lhsHi);
    SyncV();
    pto::TSUB(tmp2, tmp0, tmp1);
    SyncV();
    pto::TSUB(tmp2, lhsHi, tmp2);
    SyncV();
    pto::TSUB(tmp1, rhsHi, tmp1);
    SyncV();
    pto::TADD(tmp2, tmp2, tmp1);
    SyncV();
    pto::TADD(tmp2, tmp2, lhsLo);
    SyncV();
    pto::TADD(tmp2, tmp2, rhsLo);
    SyncV();
    DigammaNormalize(outHi, outLo, tmp0, tmp2, tmp6, tmp7);
}

template <typename TileType>
TILEOP void DigammaAdd(TileType outHi, TileType outLo, TileType lhsHi, TileType lhsLo, const float* rhs, TileType tmp0,
                       TileType tmp1, TileType tmp2, TileType tmp3, TileType tmp6, TileType tmp7)
{
    // Add a two-float constant.
    pto::TADDS(tmp0, lhsHi, rhs[TileOp::NUM_VALUE_0]);
    SyncV();
    pto::TSUB(tmp1, tmp0, lhsHi);
    SyncV();
    pto::TSUB(tmp2, tmp0, tmp1);
    SyncV();
    pto::TSUB(tmp2, lhsHi, tmp2);
    SyncV();
    pto::TEXPANDS(tmp3, rhs[TileOp::NUM_VALUE_0]);
    SyncV();
    pto::TSUB(tmp1, tmp3, tmp1);
    SyncV();
    pto::TADD(tmp2, tmp2, tmp1);
    SyncV();
    pto::TADD(tmp2, tmp2, lhsLo);
    SyncV();
    pto::TADDS(tmp2, tmp2, rhs[TileOp::NUM_VALUE_1]);
    SyncV();
    DigammaNormalize(outHi, outLo, tmp0, tmp2, tmp6, tmp7);
}

template <typename TileType>
TILEOP void DigammaSplit(TileType outHi, TileType outLo, TileType src, TileType tmp6, TileType tmp7)
{
    // Dekker split: src = outHi + outLo, using 4097 = 2^12 + 1.
    pto::TMULS(tmp7, src, SPLIT_FACTOR);
    SyncV();
    pto::TSUB(tmp6, tmp7, src);
    SyncV();
    pto::TSUB(outHi, tmp7, tmp6);
    SyncV();
    pto::TSUB(outLo, src, outHi);
    SyncV();
}

template <typename TileType>
TILEOP void DigammaMul(TileType outHi, TileType outLo, TileType lhsHi, TileType lhsLo, TileType rhsHi, TileType rhsLo,
                       TileType tmp0, TileType tmp1, TileType tmp2, TileType tmp3, TileType tmp4, TileType tmp5,
                       TileType tmp6, TileType tmp7)
{
    // Compensated two-float multiplication:
    // result ~= lhsHi*rhsHi + lhsHi*rhsLo + lhsLo*rhsHi.
    // Recover the lhsHi*rhsHi rounding error with Dekker splits.
    pto::TMUL(tmp0, lhsHi, rhsHi);
    SyncV();
    DigammaSplit(tmp1, tmp2, lhsHi, tmp6, tmp7);
    DigammaSplit(tmp3, tmp4, rhsHi, tmp6, tmp7);
    pto::TMUL(tmp5, tmp1, tmp3);
    SyncV();
    pto::TSUB(tmp5, tmp5, tmp0);
    SyncV();
    pto::TMUL(tmp6, tmp1, tmp4);
    SyncV();
    pto::TADD(tmp5, tmp5, tmp6);
    SyncV();
    pto::TMUL(tmp6, tmp2, tmp3);
    SyncV();
    pto::TADD(tmp5, tmp5, tmp6);
    SyncV();
    pto::TMUL(tmp6, tmp2, tmp4);
    SyncV();
    pto::TADD(tmp5, tmp5, tmp6);
    SyncV();
    pto::TMUL(tmp6, lhsHi, rhsLo);
    SyncV();
    pto::TADD(tmp5, tmp5, tmp6);
    SyncV();
    pto::TMUL(tmp6, lhsLo, rhsHi);
    SyncV();
    pto::TADD(tmp5, tmp5, tmp6);
    SyncV();
    DigammaNormalize(outHi, outLo, tmp0, tmp5, tmp6, tmp7);
}

template <typename TileType>
TILEOP void DigammaMul(TileType outHi, TileType outLo, TileType lhsHi, TileType lhsLo, const float* rhs, TileType tmp0,
                       TileType tmp1, TileType tmp2, TileType tmp5, TileType tmp6, TileType tmp7)
{
    // Multiply by a two-float constant.
    pto::TMULS(tmp0, lhsHi, rhs[TileOp::NUM_VALUE_0]);
    SyncV();
    DigammaSplit(tmp1, tmp2, lhsHi, tmp6, tmp7);
    pto::TMULS(tmp5, tmp1, rhs[TileOp::NUM_VALUE_2]);
    SyncV();
    pto::TSUB(tmp5, tmp5, tmp0);
    SyncV();
    pto::TMULS(tmp6, tmp1, rhs[TileOp::NUM_VALUE_3]);
    SyncV();
    pto::TADD(tmp5, tmp5, tmp6);
    SyncV();
    pto::TMULS(tmp6, tmp2, rhs[TileOp::NUM_VALUE_2]);
    SyncV();
    pto::TADD(tmp5, tmp5, tmp6);
    SyncV();
    pto::TMULS(tmp6, tmp2, rhs[TileOp::NUM_VALUE_3]);
    SyncV();
    pto::TADD(tmp5, tmp5, tmp6);
    SyncV();
    pto::TMULS(tmp6, lhsHi, rhs[TileOp::NUM_VALUE_1]);
    SyncV();
    pto::TADD(tmp5, tmp5, tmp6);
    SyncV();
    pto::TMULS(tmp6, lhsLo, rhs[TileOp::NUM_VALUE_0]);
    SyncV();
    pto::TADD(tmp5, tmp5, tmp6);
    SyncV();
    DigammaNormalize(outHi, outLo, tmp0, tmp5, tmp6, tmp7);
}

template <typename TileType>
TILEOP void DigammaDiv(TileType outHi, TileType outLo, TileType lhsHi, TileType lhsLo, TileType rhsHi, TileType rhsLo,
                       TileType tmp0, TileType tmp1, TileType tmp2, TileType tmp3, TileType tmp4, TileType tmp5,
                       TileType tmp6, TileType tmp7, TileType tmp8, TileType tmp9, TileType tmp10)
{
    // Two-float division with one residual correction:
    // q0 = lhsHi / rhsHi
    // r  = lhs - rhs*q0
    // q1 = r / rhsHi
    // result ~= q0 + q1
    pto::TDIV(tmp8, lhsHi, rhsHi);
    SyncV();
    DigammaLift(tmp9, tmp10, tmp8);
    DigammaMul(tmp9, tmp10, rhsHi, rhsLo, tmp9, tmp10, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7);
    pto::TMULS(tmp9, tmp9, NEG_ONE);
    SyncV();
    pto::TMULS(tmp10, tmp10, NEG_ONE);
    SyncV();
    DigammaAdd(tmp9, tmp10, lhsHi, lhsLo, tmp9, tmp10, tmp0, tmp1, tmp2, tmp6, tmp7);
    pto::TADD(tmp0, tmp9, tmp10);
    SyncV();
    pto::TDIV(tmp10, tmp0, rhsHi);
    SyncV();
    DigammaNormalize(outHi, outLo, tmp8, tmp10, tmp6, tmp7);
}

template <typename TileType, typename MaskTile>
TILEOP void DigammaSelect(TileType out, TileType src0, TileType src1, TileType tmp7, TileType tmp10, MaskTile mask)
{
    // Select src0 when mask is true; tmp7 is the output and tmp10 is TSEL scratch.
    pto::TSEL(tmp7, mask, src0, src1, tmp10);
    SyncV();
    pto::TMULS(out, tmp7, ONE);
    SyncV();
}

template <typename TileType, typename MaskTile>
TILEOP void DigammaReduce(TileType outHi, TileType outLo, TileType src, TileType tmp1, TileType tmp2, TileType tmp7,
                          TileType tmp10, MaskTile mask)
{
    // Reduce src = 2^k * z and store k in outLo.
    // The 64/32/.../1 stages cover all finite positive fp32 values.
    DigammaLift(outHi, outLo, src);
    for (int i = 0; i < TileOp::NUM_VALUE_7; ++i) {
        pto::TCMPS(mask, outHi, DIGAMMA_REDUCE_BOUNDARIES[i], pto::CmpMode::GE);
        SyncV();
        pto::TADDS(tmp1, outLo, DIGAMMA_REDUCE_EXPONENTS[i]);
        SyncV();
        pto::TMULS(tmp2, outHi, DIGAMMA_REDUCE_SCALES[i]);
        SyncV();
        DigammaSelect(outLo, tmp1, outLo, tmp7, tmp10, mask);
        DigammaSelect(outHi, tmp2, outHi, tmp7, tmp10, mask);
    }
    // Apply the final reduction for the log lookup interval.
    pto::TCMPS(mask, outHi, REDUCE_BOUND, pto::CmpMode::GE);
    SyncV();
    pto::TADDS(tmp1, outLo, ONE);
    SyncV();
    pto::TMULS(tmp2, outHi, HALF);
    SyncV();
    DigammaSelect(outLo, tmp1, outLo, tmp7, tmp10, mask);
    DigammaSelect(outHi, tmp2, outHi, tmp7, tmp10, mask);
}

template <typename TileType, typename MaskTile>
TILEOP void DigammaLookup(TileType outHi, TileType outLo, TileType src, const float table[][TileOp::NUM_VALUE_4],
                          TileType tmp0, TileType tmp7, TileType tmp10, MaskTile mask)
{
    // Select a two-float table entry for src.
    DigammaSet(outHi, outLo, table[0]);
    for (int i = 1; i < TileOp::NUM_VALUE_16; ++i) {
        pto::TCMPS(mask, src, DIGAMMA_LOOKUP_BOUNDARIES[i], pto::CmpMode::GE);
        SyncV();
        pto::TEXPANDS(tmp0, table[i][TileOp::NUM_VALUE_0]);
        SyncV();
        DigammaSelect(outHi, tmp0, outHi, tmp7, tmp10, mask);
        pto::TEXPANDS(tmp0, table[i][TileOp::NUM_VALUE_1]);
        SyncV();
        DigammaSelect(outLo, tmp0, outLo, tmp7, tmp10, mask);
    }
}

template <typename TileType, typename SrcTile, typename MaskTile>
TILEOP void DigammaReflection(TileType cot, SrcTile src, TileType tmp0, TileType tmp1, TileType tmp2, TileType tmp3,
                              TileType tmp4, TileType tmp5, TileType tmp6, TileType tmp7, TileType tmp8, TileType tmp9,
                              TileType tmp10, TileType tmp11, TileType tmp12, TileType tmp13, TileType tmp14,
                              TileType tmp15, TileType tmp16, TileType tmp17, TileType tmp18, MaskTile tmp20)
{
    // Reflection: psi(x) = psi(1-x) - pi*cot(pi*x).
    // Reduce with k = round(2*x), r = (2*x-k)/2, and t = pi*r.
    // Then t is in [-pi/4, pi/4].
    pto::TMULS(tmp17, src, TWO);
    SyncV();
    pto::TCVT(tmp18, tmp17, pto::RoundMode::CAST_RINT);
    SyncV();
    pto::TSUB(tmp11, tmp17, tmp18);
    SyncV();
    pto::TMULS(tmp11, tmp11, HALF);
    SyncV();
    pto::TEXPANDS(tmp12, ZERO);
    SyncV();

    // parity = k - 2*floor(k/2), either 0 or 1.
    pto::TMULS(tmp0, tmp18, HALF);
    SyncV();
    pto::TCVT(tmp1, tmp0, pto::RoundMode::CAST_FLOOR);
    SyncV();
    pto::TMULS(tmp0, tmp1, TWO);
    SyncV();
    pto::TSUB(cot, tmp18, tmp0);
    SyncV();

    // t = pi*r, square = t*t.
    DigammaMul(tmp11, tmp12, tmp11, tmp12, DIGAMMA_CONSTANTS[TileOp::NUM_VALUE_0], tmp0, tmp1, tmp2, tmp5, tmp6, tmp7);
    DigammaMul(tmp13, tmp14, tmp11, tmp12, tmp11, tmp12, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7);

    // Horner evaluation: sin(t) ~= t * P(t^2).
    DigammaSet(tmp15, tmp16, DIGAMMA_SIN_COEFFICIENTS[TileOp::NUM_VALUE_9]);
    for (int j = TileOp::NUM_VALUE_8; j >= 0; --j) {
        DigammaMul(tmp15, tmp16, tmp15, tmp16, tmp13, tmp14, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7);
        DigammaAdd(tmp15, tmp16, tmp15, tmp16, DIGAMMA_SIN_COEFFICIENTS[j], tmp0, tmp1, tmp2, tmp3, tmp6, tmp7);
    }
    DigammaMul(tmp15, tmp16, tmp15, tmp16, tmp11, tmp12, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7);

    // Horner evaluation: cos(t) ~= Q(t^2).
    DigammaSet(tmp11, tmp12, DIGAMMA_COS_COEFFICIENTS[TileOp::NUM_VALUE_10]);
    for (int j = TileOp::NUM_VALUE_9; j >= 0; --j) {
        DigammaMul(tmp11, tmp12, tmp11, tmp12, tmp13, tmp14, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7);
        DigammaAdd(tmp11, tmp12, tmp11, tmp12, DIGAMMA_COS_COEFFICIENTS[j], tmp0, tmp1, tmp2, tmp3, tmp6, tmp7);
    }

    // Even k: -pi*cos(t)/sin(t); odd k: pi*sin(t)/cos(t).
    pto::TCMPS(tmp20, cot, HALF, pto::CmpMode::GE);
    SyncV();
    pto::TMULS(tmp8, tmp11, NEG_ONE);
    SyncV();
    pto::TMULS(tmp9, tmp12, NEG_ONE);
    SyncV();
    DigammaSelect(tmp13, tmp15, tmp8, tmp7, tmp10, tmp20);
    DigammaSelect(tmp14, tmp16, tmp9, tmp7, tmp10, tmp20);
    DigammaSelect(tmp8, tmp11, tmp15, tmp7, tmp10, tmp20);
    DigammaSelect(tmp9, tmp12, tmp16, tmp7, tmp10, tmp20);
    pto::TMULS(tmp11, tmp8, ONE);
    SyncV();
    pto::TMULS(tmp12, tmp9, ONE);
    SyncV();
    DigammaDiv(tmp15, tmp16, tmp13, tmp14, tmp11, tmp12, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9,
               tmp10);
    DigammaMul(tmp15, tmp16, tmp15, tmp16, DIGAMMA_CONSTANTS[TileOp::NUM_VALUE_0], tmp0, tmp1, tmp2, tmp5, tmp6, tmp7);
    pto::TADD(cot, tmp15, tmp16);
    SyncV();
}

template <typename TileType, typename MaskTile>
TILEOP void DigammaPositive(TileType out, TileType tmp0, TileType tmp1, TileType tmp2, TileType tmp3, TileType tmp4,
                            TileType tmp5, TileType tmp6, TileType tmp7, TileType tmp8, TileType tmp9, TileType tmp10,
                            TileType tmp11, TileType tmp12, TileType tmp13, TileType tmp14, TileType tmp15,
                            TileType tmp16, TileType tmp17, TileType tmp18, MaskTile tmp20, MaskTile tmp21)
{
    // Shift positive x to w >= 10:
    // psi(w+1) = psi(w) + 1/w
    // psi(x) = psi(w) - sum(1/(x+i)).
    pto::TEXPANDS(tmp18, ZERO);
    SyncV();
    for (int j = 0; j < TileOp::NUM_VALUE_10; ++j) {
        // Use 1 as the inactive denominator to avoid invalid compensated division.
        pto::TCMPS(tmp21, tmp17, RECURRENCE_SHIFT, pto::CmpMode::LT);
        SyncV();
        pto::TEXPANDS(tmp2, ONE);
        SyncV();
        DigammaSelect(tmp2, tmp17, tmp2, tmp7, tmp10, tmp21);
        DigammaSet(tmp11, tmp12, DIGAMMA_CONSTANTS[TileOp::NUM_VALUE_1]);
        DigammaLift(tmp13, tmp14, tmp2);
        DigammaDiv(tmp11, tmp12, tmp11, tmp12, tmp13, tmp14, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9,
                   tmp10);
        pto::TADD(tmp0, tmp11, tmp12);
        SyncV();
        pto::TSUB(tmp1, tmp18, tmp0);
        SyncV();
        pto::TADDS(tmp2, tmp17, ONE);
        SyncV();
        DigammaSelect(tmp18, tmp1, tmp18, tmp7, tmp10, tmp21);
        DigammaSelect(tmp17, tmp2, tmp17, tmp7, tmp10, tmp21);
    }
    // Save the exact w == 10 recurrence result.
    pto::TADDS(out, tmp18, PSI_10);
    SyncV();

    // Compute log(w):
    // w = 2^k*z
    // log(w) = k*ln(2) + log(c) + log(z/c).
    // Look up 1/c and log(c) from the interval containing z.
    DigammaReduce(tmp13, tmp14, tmp17, tmp1, tmp2, tmp7, tmp10, tmp20);
    DigammaLookup(tmp11, tmp12, tmp13, DIGAMMA_INVERSE_C_TABLE, tmp0, tmp7, tmp10, tmp20);
    pto::TEXPANDS(tmp14, ZERO);
    SyncV();
    // Approximate log(1+r), where r = z/c - 1.
    DigammaMul(tmp11, tmp12, tmp13, tmp14, tmp11, tmp12, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7);
    DigammaAdd(tmp11, tmp12, tmp11, tmp12, DIGAMMA_CONSTANTS[TileOp::NUM_VALUE_3], tmp0, tmp1, tmp2, tmp3, tmp6, tmp7);
    DigammaMul(tmp13, tmp14, tmp11, tmp12, tmp11, tmp12, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7);
    DigammaMul(tmp15, tmp16, tmp11, tmp12, DIGAMMA_CONSTANTS[TileOp::NUM_VALUE_6], tmp0, tmp1, tmp2, tmp5, tmp6, tmp7);
    DigammaAdd(tmp15, tmp16, tmp15, tmp16, DIGAMMA_CONSTANTS[TileOp::NUM_VALUE_7], tmp0, tmp1, tmp2, tmp3, tmp6, tmp7);
    DigammaMul(tmp8, tmp9, tmp13, tmp14, DIGAMMA_CONSTANTS[TileOp::NUM_VALUE_5], tmp0, tmp1, tmp2, tmp5, tmp6, tmp7);
    DigammaAdd(tmp15, tmp16, tmp8, tmp9, tmp15, tmp16, tmp0, tmp1, tmp2, tmp6, tmp7);
    DigammaMul(tmp15, tmp16, tmp15, tmp16, tmp13, tmp14, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7);

    // Reduce w again and assemble log(w).
    DigammaReduce(tmp13, tmp14, tmp17, tmp1, tmp2, tmp7, tmp10, tmp20);
    DigammaLookup(tmp8, tmp9, tmp13, DIGAMMA_LOG_C_TABLE, tmp0, tmp7, tmp10, tmp20);
    pto::TMULS(tmp13, tmp14, ONE);
    SyncV();
    pto::TEXPANDS(tmp14, ZERO);
    SyncV();
    DigammaMul(tmp13, tmp14, tmp13, tmp14, DIGAMMA_CONSTANTS[TileOp::NUM_VALUE_4], tmp0, tmp1, tmp2, tmp5, tmp6, tmp7);
    DigammaAdd(tmp13, tmp14, tmp8, tmp9, tmp13, tmp14, tmp0, tmp1, tmp2, tmp6, tmp7);
    DigammaAdd(tmp13, tmp14, tmp13, tmp14, tmp11, tmp12, tmp0, tmp1, tmp2, tmp6, tmp7);
    DigammaAdd(tmp15, tmp16, tmp15, tmp16, tmp13, tmp14, tmp0, tmp1, tmp2, tmp6, tmp7);
    pto::TADD(tmp0, tmp15, tmp16);
    SyncV();
    pto::TADD(tmp18, tmp18, tmp0);
    SyncV();

    // Use the original w for log and min(w, 2^57) for safe asymptotic corrections.
    // Above 2^57, the omitted correction is below fp32 resolution.
    pto::TCMPS(tmp20, tmp17, DIGAMMA_CORRECTION_MAX, pto::CmpMode::GT);
    SyncV();
    pto::TEXPANDS(tmp2, DIGAMMA_CORRECTION_MAX);
    SyncV();
    DigammaSelect(tmp17, tmp2, tmp17, tmp7, tmp10, tmp20);

    // Add the asymptotic term -1/(2*w).
    DigammaSet(tmp11, tmp12, DIGAMMA_CONSTANTS[TileOp::NUM_VALUE_2]);
    DigammaLift(tmp13, tmp14, tmp17);
    DigammaDiv(tmp11, tmp12, tmp11, tmp12, tmp13, tmp14, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9,
               tmp10);
    pto::TADD(tmp0, tmp11, tmp12);
    SyncV();
    pto::TSUB(tmp18, tmp18, tmp0);
    SyncV();

    // Bernoulli correction with z = 1/(w*w):
    // psi(w) ~= log(w) - 1/(2*w)
    //           - z*(1/12 - z*(1/120 - z*(1/252 - ...))).
    pto::TMUL(tmp0, tmp17, tmp17);
    SyncV();
    DigammaLift(tmp13, tmp14, tmp0);
    DigammaSet(tmp11, tmp12, DIGAMMA_CONSTANTS[TileOp::NUM_VALUE_1]);
    DigammaDiv(tmp11, tmp12, tmp11, tmp12, tmp13, tmp14, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9,
               tmp10);
    pto::TADD(tmp11, tmp11, tmp12);
    SyncV();
    pto::TEXPANDS(tmp12, ASYM_INIT);
    SyncV();
    for (int j = 0; j < TileOp::NUM_VALUE_6; ++j) {
        pto::TMUL(tmp12, tmp12, tmp11);
        SyncV();
        pto::TADDS(tmp12, tmp12, DIGAMMA_ASYMPTOTIC_COEFFICIENTS[j]);
        SyncV();
    }
    pto::TMUL(tmp0, tmp12, tmp11);
    SyncV();
    pto::TSUB(tmp18, tmp18, tmp0);
    SyncV();

    // Use the saved recurrence result for w == 10; otherwise use the asymptotic result.
    pto::TCMPS(tmp20, tmp17, RECURRENCE_SHIFT, pto::CmpMode::EQ);
    SyncV();
    DigammaSelect(tmp1, out, tmp18, tmp7, tmp10, tmp20);
    pto::TMULS(out, tmp1, ONE);
    SyncV();
}

template <bool CanonicalNan, typename DstTile, typename SrcTile, typename TileType, typename MaskTile>
TILEOP void DigammaCompute(DstTile dst, SrcTile src, TileType tmp0, TileType tmp1, TileType tmp2, TileType tmp3,
                           TileType tmp4, TileType tmp5, TileType tmp6, TileType tmp7, TileType tmp8, TileType tmp9,
                           TileType tmp10, TileType tmp11, TileType tmp12, TileType tmp13, TileType tmp14,
                           TileType tmp15, TileType tmp16, TileType tmp17, TileType tmp18, TileType tmp19,
                           MaskTile tmp20, MaskTile tmp21)
{
    // Compute the negative candidate: psi(1-x) - pi*cot(pi*x).
    DigammaReflection(tmp19, src, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12,
                      tmp13, tmp14, tmp15, tmp16, tmp17, tmp18, tmp20);
    pto::TEXPANDS(tmp0, ONE);
    SyncV();
    pto::TSUB(tmp17, tmp0, src);
    SyncV();
    DigammaPositive(dst, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14,
                    tmp15, tmp16, tmp17, tmp18, tmp20, tmp21);
    pto::TADD(dst, dst, tmp19);
    SyncV();

    // Compute the positive candidate and select by x >= 0.
    pto::TADDS(tmp17, src, ZERO);
    SyncV();
    DigammaPositive(tmp19, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12, tmp13,
                    tmp14, tmp15, tmp16, tmp17, tmp18, tmp20, tmp21);
    pto::TCMPS(tmp20, src, ZERO, pto::CmpMode::GE);
    SyncV();
    DigammaSelect(dst, tmp19, dst, tmp7, tmp10, tmp20);

    // Return NaN for x <= -2^23, where fp32 cannot distinguish adjacent integers.
    pto::TEXPANDS(tmp0, NAN);
    SyncV();
    pto::TCMPS(tmp20, src, MIN_NEG_WITH_FLOAT, pto::CmpMode::LE);
    SyncV();
    pto::TSEL(dst, tmp20, tmp0, dst, tmp1);
    SyncV();

    // Return NaN at negative integer poles.
    pto::TCMPS(tmp20, src, ZERO, pto::CmpMode::LT);
    SyncV();
    pto::TCMPS(tmp21, src, MIN_NEG_WITH_FLOAT, pto::CmpMode::GT);
    SyncV();
    pto::TAND(tmp20, tmp20, tmp21);
    SyncV();
    pto::TCVT(tmp0, src, pto::RoundMode::CAST_RINT);
    SyncV();
    pto::TCMP(tmp21, src, tmp0, pto::CmpMode::EQ);
    SyncV();
    pto::TAND(tmp20, tmp20, tmp21);
    SyncV();
    pto::TEXPANDS(tmp0, NAN);
    SyncV();
    pto::TSEL(dst, tmp20, tmp0, dst, tmp1);
    SyncV();

    // Preserve signed zero:
    // -1/(+0) = -inf, -1/(-0) = +inf.
    pto::TEXPANDS(tmp0, NEG_ONE);
    SyncV();
    pto::TDIV(tmp1, tmp0, src);
    SyncV();
    pto::TCMPS(tmp20, src, ZERO, pto::CmpMode::EQ);
    SyncV();
    pto::TSEL(dst, tmp20, tmp1, dst, tmp2);
    SyncV();

    // psi(+inf) = +inf.
    pto::TCMPS(tmp20, src, INFINITY, pto::CmpMode::EQ);
    SyncV();
    pto::TPARTSEL(dst, tmp20, src, dst, tmp2);
    SyncV();

    pto::TCMP(tmp20, src, src, pto::CmpMode::NE);
    SyncV();
    if constexpr (CanonicalNan) {
        pto::TEXPANDS(tmp0, NAN);
        SyncV();
        pto::TSEL(dst, tmp20, tmp0, dst, tmp2);
    } else {
        pto::TSEL(dst, tmp20, tmp0, dst, tmp2);
    }
    SyncV();
}

#define OP_TILE_OP_DIGAMMA TDigamma
template <typename T0, typename T1, typename T2>
TILEOP void TDigamma(T0 dst, T1 tmp, T2 src)
{
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto shape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto shape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    if (shape0 == 0 || shape1 == 0 || shape2 == 0 || shape3 == 0 || shape4 == 0) {
        return;
    }

    constexpr auto tileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    auto tileSize = shape3 * tileW * sizeof(float);
    using DataTile = pto::Tile<pto::TileType::Vec, float, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using TmpMaskTile = pto::Tile<pto::TileType::Vec, uint8_t, tileH, tileW * TileOp::NUM_VALUE_4,
                                  pto::BLayout::RowMajor, -1, -1>;

    DataTile dstTile(shape3, shape4);
    auto srcTile = MakeElementwiseOperandExecTile(dst, src);

    DataTile tmp0Tile(shape3, shape4);
    DataTile tmp1Tile(shape3, shape4);
    DataTile tmp2Tile(shape3, shape4);
    DataTile tmp3Tile(shape3, shape4);
    DataTile tmp4Tile(shape3, shape4);
    DataTile tmp5Tile(shape3, shape4);
    DataTile tmp6Tile(shape3, shape4);
    DataTile tmp7Tile(shape3, shape4);
    DataTile tmp8Tile(shape3, shape4);
    DataTile tmp9Tile(shape3, shape4);
    DataTile tmp10Tile(shape3, shape4);
    DataTile tmp11Tile(shape3, shape4);
    DataTile tmp12Tile(shape3, shape4);
    DataTile tmp13Tile(shape3, shape4);
    DataTile tmp14Tile(shape3, shape4);
    DataTile tmp15Tile(shape3, shape4);
    DataTile tmp16Tile(shape3, shape4);
    DataTile tmp17Tile(shape3, shape4);
    DataTile tmp18Tile(shape3, shape4);
    DataTile tmp19Tile(shape3, shape4);
    TmpMaskTile tmp20Tile(shape3, shape4);
    TmpMaskTile tmp21Tile(shape3, shape4);

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + GenTileOffset(dst, tileOffsets) * sizeof(float)));
                AssignElementwiseOperandExecTile(srcTile, src, tileOffsets);

                pto::TASSIGN(tmp0Tile, (uint64_t)(tmp.GetAddr()));
                pto::TASSIGN(tmp1Tile, (uint64_t)(tmp.GetAddr() + TileOp::NUM_VALUE_1 * tileSize));
                pto::TASSIGN(tmp2Tile, (uint64_t)(tmp.GetAddr() + TileOp::NUM_VALUE_2 * tileSize));
                pto::TASSIGN(tmp3Tile, (uint64_t)(tmp.GetAddr() + TileOp::NUM_VALUE_3 * tileSize));
                pto::TASSIGN(tmp4Tile, (uint64_t)(tmp.GetAddr() + TileOp::NUM_VALUE_4 * tileSize));
                pto::TASSIGN(tmp5Tile, (uint64_t)(tmp.GetAddr() + TileOp::NUM_VALUE_5 * tileSize));
                pto::TASSIGN(tmp6Tile, (uint64_t)(tmp.GetAddr() + TileOp::NUM_VALUE_6 * tileSize));
                pto::TASSIGN(tmp7Tile, (uint64_t)(tmp.GetAddr() + TileOp::NUM_VALUE_7 * tileSize));
                pto::TASSIGN(tmp8Tile, (uint64_t)(tmp.GetAddr() + TileOp::NUM_VALUE_8 * tileSize));
                pto::TASSIGN(tmp9Tile, (uint64_t)(tmp.GetAddr() + TileOp::NUM_VALUE_9 * tileSize));
                pto::TASSIGN(tmp10Tile, (uint64_t)(tmp.GetAddr() + TileOp::NUM_VALUE_10 * tileSize));
                // Keep tmp19 off the last data slot required by A2/A3 TCMP/TCMPS src0.
                pto::TASSIGN(tmp19Tile, (uint64_t)(tmp.GetAddr() + TileOp::NUM_VALUE_11 * tileSize));
                pto::TASSIGN(tmp11Tile, (uint64_t)(tmp.GetAddr() + TileOp::NUM_VALUE_12 * tileSize));
                pto::TASSIGN(tmp12Tile, (uint64_t)(tmp.GetAddr() + TileOp::NUM_VALUE_13 * tileSize));
                pto::TASSIGN(tmp13Tile, (uint64_t)(tmp.GetAddr() + TileOp::NUM_VALUE_14 * tileSize));
                pto::TASSIGN(tmp14Tile, (uint64_t)(tmp.GetAddr() + TileOp::NUM_VALUE_15 * tileSize));
                pto::TASSIGN(tmp15Tile, (uint64_t)(tmp.GetAddr() + TileOp::NUM_VALUE_16 * tileSize));
                pto::TASSIGN(tmp16Tile, (uint64_t)(tmp.GetAddr() + TileOp::NUM_VALUE_17 * tileSize));
                pto::TASSIGN(tmp17Tile, (uint64_t)(tmp.GetAddr() + TileOp::NUM_VALUE_18 * tileSize));
                // tmp18 is not a TCMP/TCMPS src0, so it may use the last data slot.
                pto::TASSIGN(tmp18Tile, (uint64_t)(tmp.GetAddr() + TileOp::NUM_VALUE_19 * tileSize));
                // tmp20 and dead tmp4 share slot 4 between compare and select.
                pto::TASSIGN(tmp20Tile, (uint64_t)(tmp.GetAddr() + TileOp::NUM_VALUE_4 * tileSize));
                // tmp21 stays live across DigammaDiv and needs a dedicated slot.
                pto::TASSIGN(tmp21Tile, (uint64_t)(tmp.GetAddr() + TileOp::NUM_VALUE_20 * tileSize));

#ifdef __DAV_V220
                DigammaCompute<false>(dstTile, srcTile, tmp0Tile, tmp1Tile, tmp2Tile, tmp3Tile, tmp4Tile, tmp5Tile,
                                      tmp6Tile, tmp7Tile, tmp8Tile, tmp9Tile, tmp10Tile, tmp11Tile, tmp12Tile,
                                      tmp13Tile, tmp14Tile, tmp15Tile, tmp16Tile, tmp17Tile, tmp18Tile, tmp19Tile,
                                      tmp20Tile, tmp21Tile);
#else
                DigammaCompute<true>(dstTile, srcTile, tmp0Tile, tmp1Tile, tmp2Tile, tmp3Tile, tmp4Tile, tmp5Tile,
                                     tmp6Tile, tmp7Tile, tmp8Tile, tmp9Tile, tmp10Tile, tmp11Tile, tmp12Tile, tmp13Tile,
                                     tmp14Tile, tmp15Tile, tmp16Tile, tmp17Tile, tmp18Tile, tmp19Tile, tmp20Tile,
                                     tmp21Tile);
#endif
            }
        }
    }
}

#undef SyncV

#endif // TILEOP_TILE_OPERATOR_VEC_UNARY_DIGAMMA_H
