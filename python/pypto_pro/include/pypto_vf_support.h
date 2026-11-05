/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

// Support helpers for generated __simd_vf__ / __simd_callee__ code, shipped as
// a framework header (included by the Vector target codegen) instead of being
// spliced into each generated translation unit.
//
// RtWrap: a __simd_callee__-compatible wrapper around RegTensor.
//
// RegTensor's default constructor is declared PTO_INTERNAL (i.e. [aicore]),
// and bisheng forbids a __simd_vf__ function from calling anything that is not
// __simd_callee__, so the plain `RegTensor<T> x;` declaration emitted by VF
// codegen fails to compile inside a __simd_vf__ body. The union below carries a
// RegTensor member whose construction is skipped by the empty __simd_callee__
// union constructor (C++ leaves the non-trivial variant member unconstructed),
// and the two conversion operators recover both uses codegen generates:
//   - operator RegType&()  : passes the underlying register to vbr/vlds/vmax
//                            builtins (single user-defined conversion);
//   - operator RegTensor<T>&() : serves the (RegTensor<T>&)x casts emitted by
//                            the B64 interleave/deinterleave helpers.
// The wrapper sits in the global namespace next to `pto` (the generated file
// does `using namespace pto;`, so RegTensor resolves correctly).
template <typename T>
struct RtWrap {
    using RegType = typename RegTensor<T>::RegType;
    // A union rather than a plain RegTensor member: RegTensor's default
    // constructor is [aicore]-attributed, so `RegTensor<T> r;` would trigger a
    // forbidden non-__simd_callee__ call inside a __simd_vf__ body. The union
    // defines its own empty __simd_callee__ constructor instead; C++ leaves the
    // non-trivial member `r` unconstructed, and it is only ever reached by
    // reference through the conversion operators below -- no RegTensor
    // construction ever runs in the generated VF code.
    union U {
        RegTensor<T> r;
        __simd_callee__ inline U() {}
        __simd_callee__ inline ~U() {}
    };
    U u;
    __simd_callee__ inline RtWrap() {}
    __simd_callee__ inline operator RegType&() { return u.r.reg; }
    __simd_callee__ inline operator RegTensor<T>&() { return u.r; }
    // RegTensor -> wrapper assignment: emitted casts such as
    // `dst = (RegTensor<int32_t>&)src;` (vf.bit_cast) target the RtWrap-named
    // registers, so the assignment must copy the register view in.
    __simd_callee__ inline RtWrap<T>& operator=(const RegTensor<T>& o)
    {
        u.r = o;
        return *this;
    }
};

// _vag_bXX_N wrappers: the vag_b8/b16/b32 address-generation intrinsics are
// [aicore]-attributed and cannot be called directly from a __simd_vf__ function
// (bisheng: "vag should be written in the vf"). Each wrapper is a single
// inline __simd_callee__ stub forwarding to the matching vag_* intrinsic, so
// bisheng inlines it and sees vag_* at the call site. All variants are
// declared; unused inline functions do not emit code.
__simd_callee__ inline AddrReg _vag_b8_1(uint32_t s0) { return vag_b8(s0); }
__simd_callee__ inline AddrReg _vag_b8_2(uint32_t s0, uint32_t s1) { return vag_b8(s0, s1); }
__simd_callee__ inline AddrReg _vag_b8_3(uint32_t s0, uint32_t s1, uint32_t s2) { return vag_b8(s0, s1, s2); }
__simd_callee__ inline AddrReg _vag_b8_4(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3)
{
    return vag_b8(s0, s1, s2, s3);
}
__simd_callee__ inline AddrReg _vag_b16_1(uint32_t s0) { return vag_b16(s0); }
__simd_callee__ inline AddrReg _vag_b16_2(uint32_t s0, uint32_t s1) { return vag_b16(s0, s1); }
__simd_callee__ inline AddrReg _vag_b16_3(uint32_t s0, uint32_t s1, uint32_t s2) { return vag_b16(s0, s1, s2); }
__simd_callee__ inline AddrReg _vag_b16_4(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3)
{
    return vag_b16(s0, s1, s2, s3);
}
__simd_callee__ inline AddrReg _vag_b32_1(uint32_t s0) { return vag_b32(s0); }
__simd_callee__ inline AddrReg _vag_b32_2(uint32_t s0, uint32_t s1) { return vag_b32(s0, s1); }
__simd_callee__ inline AddrReg _vag_b32_3(uint32_t s0, uint32_t s1, uint32_t s2) { return vag_b32(s0, s1, s2); }
__simd_callee__ inline AddrReg _vag_b32_4(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3)
{
    return vag_b32(s0, s1, s2, s3);
}
