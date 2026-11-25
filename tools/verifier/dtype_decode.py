#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Dtype decode utilities: decode packed low-precision dtypes (INT4/FP8/HF8/BF16) to numeric values."""

import numpy as np


def _build_fp8_e4m3fn_lut():
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        sign = -1.0 if i & 0x80 else 1.0
        exp = (i >> 3) & 0xF
        mant = i & 0x7
        if exp == 0:
            val = mant * 2.0**-9
        elif exp == 15 and mant == 7:
            val = float("nan")
        else:
            val = (1.0 + mant / 8.0) * 2.0 ** (exp - 7)
        lut[i] = sign * val
    return lut


_FP8_E4M3FN_LUT = _build_fp8_e4m3fn_lut()


def _decode_fp8e5m2_value(bits):
    """Decode one FP8 E5M2 byte to float. Python port of C++ DecodeFp8E5M2 in aicore_print_base.h."""
    sign = -1.0 if (bits >> 7) & 0x1 else 1.0
    exp = (bits >> 2) & 0x1F
    mant = bits & 0x3
    if exp == 0:
        if mant == 0:
            return sign * 0.0
        # subnormal: mant/4 * 2^(1-15) = mant * 2^-16
        return sign * mant * 2.0**-16
    if exp == 0x1F:  # E5M2 has infinity
        if mant == 0:
            return sign * float("inf")
        return float("nan")
    return sign * (1.0 + mant / 4.0) * 2.0 ** (exp - 15)


def _build_fp8e5m2_lut():
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        lut[i] = _decode_fp8e5m2_value(i)
    return lut


_FP8E5M2_LUT = _build_fp8e5m2_lut()


def _decode_hf8_value(bits):
    """Decode one HF8 (HiFloat8) byte to float. Python port of C++ DecodeHf8 in aicore_print_base.h."""
    sign_bit = (bits >> 7) & 0x1
    sign = -1.0 if sign_bit else 1.0
    lower7 = bits & 0x7F
    top4 = lower7 >> 3
    if top4 == 0:  # Tiny (D=0000): mv in [0,7], value = 2^(mv-23); mv=0: sign=0 -> +0, sign=1 -> NaN
        mv = lower7 & 0x7
        if mv == 0:
            if sign_bit:
                return float("nan")
            return 0.0
        return sign * 2.0 ** (mv - 23)
    if top4 == 1:  # Small (D=0001): ev=0, mv in [0,7], value = (1 + mv/8)
        mv = lower7 & 0x7
        return sign * (1.0 + mv / 8.0)
    top3 = lower7 >> 4
    if top3 == 1:  # Medium (D=001): 1-bit eb, ev = +1/-1, mv in [0,7]
        eb = (lower7 >> 3) & 0x1
        ev = -1 if eb else 1
        mv = lower7 & 0x7
        return sign * 2.0**ev * (1.0 + mv / 8.0)
    top2 = lower7 >> 5
    if top2 == 1:  # Large (D=01): 2-bit eb, ev = ±[2,3], mv in [0,7]
        eb = (lower7 >> 3) & 0x3
        ev_sign = (eb >> 1) & 0x1
        ev_abs = 2 + (eb & 0x1)
        ev = -ev_abs if ev_sign else ev_abs
        mv = lower7 & 0x7
        return sign * 2.0**ev * (1.0 + mv / 8.0)
    if top2 == 2:  # Huge (D=10): 3-bit eb, ev = ±[4,7], mv in [0,3]
        eb = (lower7 >> 2) & 0x7
        ev_sign = (eb >> 2) & 0x1
        ev_abs = 4 + (eb & 0x3)
        ev = -ev_abs if ev_sign else ev_abs
        mv = lower7 & 0x3
        return sign * 2.0**ev * (1.0 + mv / 4.0)
    # Max (D=11): 4-bit eb, ev = ±[8,15], mv in [0,1]; S_11_0111_1 (ev=15, mv=1) is infinity
    eb = (lower7 >> 1) & 0xF
    ev_sign = (eb >> 3) & 0x1
    ev_abs = 8 + (eb & 0x7)
    ev = -ev_abs if ev_sign else ev_abs
    mv = lower7 & 0x1
    if ev == 15 and mv == 1:
        return sign * float("inf")
    return sign * 2.0**ev * (1.0 + mv / 2.0)


def _build_hf8_lut():
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        lut[i] = _decode_hf8_value(i)
    return lut


_HF8_LUT = _build_hf8_lut()


def _decode_int4(raw):
    v = raw.astype(np.int16) & 0xF
    return ((v ^ 0x8) - 0x8).astype(np.int8)


def _decode_fp8(raw):
    return _FP8_E4M3FN_LUT[raw.astype(np.uint8)]


def _decode_fp8e5m2(raw):
    return _FP8E5M2_LUT[raw.astype(np.uint8)]


def _decode_hf8(raw):
    return _HF8_LUT[raw.astype(np.uint8)]


def _decode_bf16(raw):
    return (raw.astype(np.uint32) << 16).view(np.float32)


def _decode_fp8e8m0_value(bits):
    """Decode one FP8 E8M0 byte, aligned with calculator Fp8E8M0ToFloat32:
    unsigned exponent, bias 127; 0 means zero-scale; 0xFF is NaN per MX spec."""
    if bits == 0:
        return 0.0
    if bits == 0xFF:
        return float("nan")
    return 2.0 ** (bits - 127)


def _build_fp8e8m0_lut():
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        lut[i] = _decode_fp8e8m0_value(i)
    return lut


_FP8E8M0_LUT = _build_fp8e8m0_lut()


def _decode_fp8e6m2_value(bits):
    """Decode one unsigned FP8 E6M2 byte. Python port of C++ DecodeFp8E6M2 in flow_verifier.cpp."""
    if bits == 0:
        return 0.0
    if bits == 0xFF:
        return float("nan")
    exp = (bits >> 2) & 0x3F
    mant = bits & 0x3
    return (1.0 + mant / 4.0) * 2.0 ** (exp - 48)


def _build_fp8e6m2_lut():
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        lut[i] = _decode_fp8e6m2_value(i)
    return lut


_FP8E6M2_LUT = _build_fp8e6m2_lut()


def _decode_fp8e8m0(raw):
    return _FP8E8M0_LUT[raw.astype(np.uint8)]


def _decode_fp8e6m2(raw):
    return _FP8E6M2_LUT[raw.astype(np.uint8)]


# Nibble decode tables, Python port of dtype_utils.h (FP4_E2M1/FP4_E1M2/HF4_DECODE_TABLE).
_FP4_E2M1_LUT = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32
)
_FP4_E1M2_LUT = np.array(
    [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, -0.0, -0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -1.75],
    dtype=np.float32,
)
_HF4_LUT = _FP4_E1M2_LUT


def _make_fp4_unpacker(lut):
    # One storage byte holds two values, high nibble first (see Fp4PackedToFloat32 in fp_convert.cpp).
    def unpack(raw):
        b = raw.astype(np.uint8)
        out = np.empty(b.size * 2, dtype=np.float32)
        out[0::2] = lut[(b >> 4) & 0xF]
        out[1::2] = lut[b & 0xF]
        return out

    return unpack


_dtype_decoders = {
    "INT4": _decode_int4,
    "FP8": _decode_fp8,
    "FP8E4M3": _decode_fp8,
    "FP8E5M2": _decode_fp8e5m2,
    "FP8E8M0": _decode_fp8e8m0,
    "FP8E6M2": _decode_fp8e6m2,
    "HF8": _decode_hf8,
    "BF16": _decode_bf16,
    "FP4_E2M1": _make_fp4_unpacker(_FP4_E2M1_LUT),
    "FP4_E1M2": _make_fp4_unpacker(_FP4_E1M2_LUT),
    "HF4": _make_fp4_unpacker(_HF4_LUT),
}

# Packed 4-bit dtypes: the dump holds two values per byte, so decoding doubles the element count and the
# result matches the logical shape recorded by the verifier.
PACKED_NIBBLE_DTYPES = frozenset({"FP4_E2M1", "FP4_E1M2", "HF4"})


def is_packed_nibble_dtype(dtype_name):
    return dtype_name in PACKED_NIBBLE_DTYPES


def decode_tensor_data(data, dtype_name):
    """Decode raw storage bytes (uint8/uint16) to numeric values for packed dtypes.

    INT4/FP8*/HF8/BF16 are read as raw bytes from files and decoded one value per storage element.
    FP4_E2M1/FP4_E1M2/HF4 pack two values per byte (high nibble first), so decoding doubles the element
    count and the result can be reshaped directly to the recorded logical shape.
    Other dtypes are returned unchanged.
    """
    decoder = _dtype_decoders.get(dtype_name)
    if decoder is None:
        return data
    return decoder(data)
