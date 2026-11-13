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
"""Per-API VF wrappers and dispatch kernel.

All operands are UB tiles. ``counter`` is a UINT32 tile used as an invocation
counter; every wrapper increments it once. Data inputs are loaded from UB, the
API is invoked, and the result is stored back to the output UB tile(s).
"""

import difflib
import logging
import os
from pathlib import Path
import re

import numpy as np
from pypto_pro import ir
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
import pytest
import torch

M = 64
N = 1
TILE_SIZE = N * M * 4  # 256 bytes

VA_INDEX = 0
VA_COUNTER = VA_INDEX + TILE_SIZE
VA_IN0 = VA_COUNTER + TILE_SIZE
VA_IN1 = VA_IN0 + TILE_SIZE
VA_OUT0 = VA_IN1 + TILE_SIZE
VA_OUT1 = VA_OUT0 + TILE_SIZE
VA_OUT2 = VA_OUT1 + TILE_SIZE
VA_OUT3 = VA_OUT2 + TILE_SIZE
VA_OUT4 = VA_OUT3 + TILE_SIZE

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


@pl.vector_function
def vf_counter_inc(counter):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    reg = vf.load_align(counter, 0, dtype=pl.DT_UINT32)
    reg = vf.adds(reg, 1, preg)
    vf.store_align(counter, reg, preg)

@pl.vector_function
def vf_full_api(counter, src, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    result = vf.full(reg_src, preg, dtype=pl.DT_FP32, mode=pl.MergeMode.ZEROING, pos=pl.DuplicatePos.LOWEST)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_move_api(counter, src, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    result = vf.move(reg_src)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_gather_api(counter, src, index, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    reg_index = vf.load_align(index, 0, dtype=pl.DT_UINT32)
    result = vf.gather(reg_src, reg_index)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_bit_cast_api(counter, src, dst):
    vf_counter_inc(counter)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    result = vf.bit_cast(reg_src, dtype=pl.DT_UINT32)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_arange_api(counter, dst):
    vf_counter_inc(counter)
    result = vf.arange(0, dtype=pl.DT_UINT32)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_interleave_api(counter, src0, src1, dst0, dst1):
    vf_counter_inc(counter)
    reg_src0 = vf.load_align(src0, 0, dtype=pl.DT_FP32)
    reg_src1 = vf.load_align(src1, 0, dtype=pl.DT_FP32)
    result = vf.interleave(reg_src0, reg_src1, dtype=pl.DT_FP32)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    vf.store_align(dst0, result[0], preg)
    vf.store_align(dst1, result[1], preg)

@pl.vector_function
def vf_de_interleave_api(counter, src0, src1, dst0, dst1):
    vf_counter_inc(counter)
    reg_src0 = vf.load_align(src0, 0, dtype=pl.DT_FP32)
    reg_src1 = vf.load_align(src1, 0, dtype=pl.DT_FP32)
    result = vf.de_interleave(reg_src0, reg_src1, dtype=pl.DT_FP32)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    vf.store_align(dst0, result[0], preg)
    vf.store_align(dst1, result[1], preg)

@pl.vector_function
def vf_pack_api(counter, src, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_UINT32)
    reg_u16 = vf.pack(reg_src, part=pl.PackPart.LOWER, dtype=pl.DT_UINT16)
    result = vf.unpack(reg_u16, part=pl.PackPart.LOWER, dtype=pl.DT_UINT32)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_unpack_api(counter, src, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_UINT16)
    result = vf.unpack(reg_src, part=pl.PackPart.LOWER, dtype=pl.DT_UINT32)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_load_api(counter, tile, dst):
    vf_counter_inc(counter)
    result = vf.load(tile, 0)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_load_align_api(counter, tile, dst):
    vf_counter_inc(counter)
    result = vf.load_align(tile, 0, dtype=pl.DT_FP32, post_update=False, is_reg=True)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_store_api(counter, src, tile):
    vf_counter_inc(counter)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    vf.store(tile, reg_src, 64)

@pl.vector_function
def vf_store_align_api(counter, src, tile):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    vf.store_align(tile, reg_src, preg, 0, post_update=False)

@pl.vector_function
def vf_scatter_api(counter, src, index, tile):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    reg_index = vf.arange(0, dtype=pl.DT_UINT32)
    vf.scatter(tile, reg_src, reg_index, preg)

@pl.vector_function
def vf_mask_gen_with_reg_tensor_api(counter, src, preg_out):
    vf_counter_inc(counter)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    result = vf.mask_gen_with_reg_tensor(reg_src, offset=0)
    vf.store_align(preg_out, result)

@pl.vector_function
def vf_move_merg_api(counter, dst, src, dst_out):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_dst = vf.load_align(dst, 0, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    result = vf.move_merg(reg_dst, reg_src, preg, mode=pl.MergeMode.MERGING)
    vf.store_align(dst_out, result, preg)

@pl.vector_function
def vf_add_api(counter, src0, src1, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src0 = vf.load_align(src0, 0, dtype=pl.DT_FP32)
    reg_src1 = vf.load_align(src1, 0, dtype=pl.DT_FP32)
    result = vf.add(reg_src0, reg_src1, preg, mode=pl.MergeMode.ZEROING)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_sub_api(counter, src0, src1, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src0 = vf.load_align(src0, 0, dtype=pl.DT_FP32)
    reg_src1 = vf.load_align(src1, 0, dtype=pl.DT_FP32)
    result = vf.sub(reg_src0, reg_src1, preg, mode=pl.MergeMode.ZEROING)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_mul_api(counter, src0, src1, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src0 = vf.load_align(src0, 0, dtype=pl.DT_FP32)
    reg_src1 = vf.load_align(src1, 0, dtype=pl.DT_FP32)
    result = vf.mul(reg_src0, reg_src1, preg, mode=pl.MergeMode.ZEROING)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_div_api(counter, src0, src1, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src0 = vf.load_align(src0, 0, dtype=pl.DT_FP32)
    reg_src1 = vf.load_align(src1, 0, dtype=pl.DT_FP32)
    result = vf.div(reg_src0, reg_src1, preg, mode=pl.MergeMode.ZEROING, precision=False)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_max_api(counter, src0, src1, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src0 = vf.load_align(src0, 0, dtype=pl.DT_FP32)
    reg_src1 = vf.load_align(src1, 0, dtype=pl.DT_FP32)
    result = vf.max(reg_src0, reg_src1, preg, mode=pl.MergeMode.ZEROING)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_min_api(counter, src0, src1, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src0 = vf.load_align(src0, 0, dtype=pl.DT_FP32)
    reg_src1 = vf.load_align(src1, 0, dtype=pl.DT_FP32)
    result = vf.min(reg_src0, reg_src1, preg, mode=pl.MergeMode.ZEROING)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_muls_api(counter, src, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    result = vf.muls(reg_src, 1.0, preg, mode=pl.MergeMode.ZEROING)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_adds_api(counter, src, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    result = vf.adds(reg_src, 1.0, preg, mode=pl.MergeMode.ZEROING)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_mins_api(counter, src, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    result = vf.mins(reg_src, 1.0, preg, mode=pl.MergeMode.ZEROING)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_maxs_api(counter, src, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    result = vf.maxs(reg_src, 1.0, preg, mode=pl.MergeMode.ZEROING)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_neg_api(counter, src, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    result = vf.neg(reg_src, preg, mode=pl.MergeMode.ZEROING)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_abs_api(counter, src, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    result = vf.abs(reg_src, preg, mode=pl.MergeMode.ZEROING)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_sqrt_api(counter, src, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    result = vf.sqrt(reg_src, preg, mode=pl.MergeMode.ZEROING, precision=False)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_relu_api(counter, src, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    result = vf.relu(reg_src, preg, mode=pl.MergeMode.ZEROING)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_leaky_relu_api(counter, src, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    result = vf.leaky_relu(reg_src, 1.0, preg, mode=pl.MergeMode.ZEROING)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_prelu_api(counter, src, slope, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    reg_slope = vf.load_align(slope, 0, dtype=pl.DT_FP32)
    result = vf.prelu(reg_src, reg_slope, preg, mode=pl.MergeMode.ZEROING)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_ln_api(counter, src, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    result = vf.ln(reg_src, preg, mode=pl.MergeMode.ZEROING, precision=False)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_log_api(counter, src, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    result = vf.log(reg_src, preg, mode=pl.MergeMode.ZEROING, precision=False)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_log2_api(counter, src, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    result = vf.log2(reg_src, preg, mode=pl.MergeMode.ZEROING, precision=False)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_log10_api(counter, src, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    result = vf.log10(reg_src, preg, mode=pl.MergeMode.ZEROING, precision=False)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_exp_api(counter, src, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    result = vf.exp(reg_src, preg, mode=pl.MergeMode.ZEROING, precision=False)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_mull_api(counter, src0, src1, lo, hi):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    reg_src0 = vf.load_align(src0, 0, dtype=pl.DT_UINT32)
    reg_src1 = vf.load_align(src1, 0, dtype=pl.DT_UINT32)
    result = vf.mull(reg_src0, reg_src1, preg)
    vf.store_align(lo, result[0], preg)
    vf.store_align(hi, result[1], preg)

@pl.vector_function
def vf_addc_api(counter, src0, src1, carry, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    carry_src = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    reg_src0 = vf.load_align(src0, 0, dtype=pl.DT_UINT32)
    reg_src1 = vf.load_align(src1, 0, dtype=pl.DT_UINT32)
    result = vf.addc(reg_src0, reg_src1, carry_src, preg)
    vf.store_align(carry, result[0])
    vf.store_align(dst, result[1], preg)

@pl.vector_function
def vf_subc_api(counter, src0, src1, borrow, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    borrow_src = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    reg_src0 = vf.load_align(src0, 0, dtype=pl.DT_UINT32)
    reg_src1 = vf.load_align(src1, 0, dtype=pl.DT_UINT32)
    result = vf.subc(reg_src0, reg_src1, borrow_src, preg)
    vf.store_align(borrow, result[0])
    vf.store_align(dst, result[1], preg)

@pl.vector_function
def vf_and__api(counter, src0, src1, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    reg_src0 = vf.load_align(src0, 0, dtype=pl.DT_UINT32)
    reg_src1 = vf.load_align(src1, 0, dtype=pl.DT_UINT32)
    result = vf.and_(reg_src0, reg_src1, preg, mode=pl.MergeMode.ZEROING)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_or__api(counter, src0, src1, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    reg_src0 = vf.load_align(src0, 0, dtype=pl.DT_UINT32)
    reg_src1 = vf.load_align(src1, 0, dtype=pl.DT_UINT32)
    result = vf.or_(reg_src0, reg_src1, preg, mode=pl.MergeMode.ZEROING)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_xor_api(counter, src0, src1, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    reg_src0 = vf.load_align(src0, 0, dtype=pl.DT_UINT32)
    reg_src1 = vf.load_align(src1, 0, dtype=pl.DT_UINT32)
    result = vf.xor(reg_src0, reg_src1, preg, mode=pl.MergeMode.ZEROING)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_not__api(counter, src, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_UINT32)
    result = vf.not_(reg_src, preg, mode=pl.MergeMode.ZEROING)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_shift_left_api(counter, src, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_UINT32)
    result = vf.shift_left(reg_src, 1, preg, mode=pl.MergeMode.ZEROING)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_shift_right_api(counter, src, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_UINT32)
    result = vf.shift_right(reg_src, 1, preg, mode=pl.MergeMode.ZEROING)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_eq_api(counter, src0, src1, preg_out):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src0 = vf.load_align(src0, 0, dtype=pl.DT_FP32)
    reg_src1 = vf.load_align(src1, 0, dtype=pl.DT_FP32)
    result = vf.eq(reg_src0, reg_src1, preg)
    vf.store_align(preg_out, result)

@pl.vector_function
def vf_ne_api(counter, src0, src1, preg_out):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src0 = vf.load_align(src0, 0, dtype=pl.DT_FP32)
    reg_src1 = vf.load_align(src1, 0, dtype=pl.DT_FP32)
    result = vf.ne(reg_src0, reg_src1, preg)
    vf.store_align(preg_out, result)

@pl.vector_function
def vf_lt_api(counter, src0, src1, preg_out):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src0 = vf.load_align(src0, 0, dtype=pl.DT_FP32)
    reg_src1 = vf.load_align(src1, 0, dtype=pl.DT_FP32)
    result = vf.lt(reg_src0, reg_src1, preg)
    vf.store_align(preg_out, result)

@pl.vector_function
def vf_gt_api(counter, src0, src1, preg_out):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src0 = vf.load_align(src0, 0, dtype=pl.DT_FP32)
    reg_src1 = vf.load_align(src1, 0, dtype=pl.DT_FP32)
    result = vf.gt(reg_src0, reg_src1, preg)
    vf.store_align(preg_out, result)

@pl.vector_function
def vf_le_api(counter, src0, src1, preg_out):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src0 = vf.load_align(src0, 0, dtype=pl.DT_FP32)
    reg_src1 = vf.load_align(src1, 0, dtype=pl.DT_FP32)
    result = vf.le(reg_src0, reg_src1, preg)
    vf.store_align(preg_out, result)

@pl.vector_function
def vf_ge_api(counter, src0, src1, preg_out):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src0 = vf.load_align(src0, 0, dtype=pl.DT_FP32)
    reg_src1 = vf.load_align(src1, 0, dtype=pl.DT_FP32)
    result = vf.ge(reg_src0, reg_src1, preg)
    vf.store_align(preg_out, result)

@pl.vector_function
def vf_select_api(counter, src0, src1, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src0 = vf.load_align(src0, 0, dtype=pl.DT_FP32)
    reg_src1 = vf.load_align(src1, 0, dtype=pl.DT_FP32)
    result = vf.select(reg_src0, reg_src1, preg)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_squeeze_api(counter, src, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    result = vf.squeeze(reg_src, preg)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_astype_api(counter, src, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    result_s32 = vf.astype(reg_src, preg, round_mode=pl.VFRoundMode.CAST_TRUNC, dtype=pl.DT_INT32)
    result = vf.bit_cast(result_s32, dtype=pl.DT_UINT32)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_truncate_api(counter, src, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    result = vf.truncate(reg_src, preg, round_mode=pl.VFRoundMode.CAST_TRUNC, mode=pl.MergeMode.ZEROING)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_reduce_sum_api(counter, src, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    result = vf.reduce_sum(reg_src, preg, datablock=False)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_reduce_max_api(counter, src, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    result = vf.reduce_max(reg_src, preg, datablock=False)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_reduce_min_api(counter, src, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    result = vf.reduce_min(reg_src, preg, datablock=False)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_pair_reduce_sum_api(counter, src, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    result = vf.pair_reduce_sum(reg_src, preg, mode=pl.MergeMode.ZEROING)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_mul_add_dst_api(counter, src0, src1, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src0 = vf.load_align(src0, 0, dtype=pl.DT_FP32)
    reg_src1 = vf.load_align(src1, 0, dtype=pl.DT_FP32)
    result = vf.mul_add_dst(reg_src0, reg_src1, preg, mode=pl.MergeMode.ZEROING)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_mul_add_dst_acc_api(counter, dst, src0, src1, dst_out):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_dst = vf.load_align(dst, 0, dtype=pl.DT_FP32)
    reg_src0 = vf.load_align(src0, 0, dtype=pl.DT_FP32)
    reg_src1 = vf.load_align(src1, 0, dtype=pl.DT_FP32)
    result = vf.mul_add_dst_acc(reg_dst, reg_src0, reg_src1, preg, mode=pl.MergeMode.ZEROING)
    vf.store_align(dst_out, result, preg)

@pl.vector_function
def vf_mul_dst_add_api(counter, src0, src1, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src0 = vf.load_align(src0, 0, dtype=pl.DT_FP32)
    reg_src1 = vf.load_align(src1, 0, dtype=pl.DT_FP32)
    result = vf.mul_dst_add(reg_src0, reg_src1, preg, mode=pl.MergeMode.ZEROING)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_mul_dst_add_acc_api(counter, dst, src0, src1, dst_out):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_dst = vf.load_align(dst, 0, dtype=pl.DT_FP32)
    reg_src0 = vf.load_align(src0, 0, dtype=pl.DT_FP32)
    reg_src1 = vf.load_align(src1, 0, dtype=pl.DT_FP32)
    result = vf.mul_dst_add_acc(reg_dst, reg_src0, reg_src1, preg, mode=pl.MergeMode.ZEROING)
    vf.store_align(dst_out, result, preg)

@pl.vector_function
def vf_axpy_api(counter, src, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    result = vf.axpy(reg_src, 1.0, preg, mode=pl.MergeMode.ZEROING)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_axpy_acc_api(counter, dst, src, dst_out):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_dst = vf.load_align(dst, 0, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    result = vf.axpy_acc(reg_dst, reg_src, 1.0, preg, mode=pl.MergeMode.ZEROING)
    vf.store_align(dst_out, result, preg)

@pl.vector_function
def vf_abs_sub_api(counter, src0, src1, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src0 = vf.load_align(src0, 0, dtype=pl.DT_FP32)
    reg_src1 = vf.load_align(src1, 0, dtype=pl.DT_FP32)
    result = vf.abs_sub(reg_src0, reg_src1, preg, mode=pl.MergeMode.ZEROING)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_muls_cast_api(counter, src, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    result = vf.muls_cast(reg_src, 1.0, preg, dtype=pl.DT_FP16)
    result_f32 = vf.astype(result, preg, dtype=pl.DT_FP32)
    vf.store_align(dst, result_f32, preg)

@pl.vector_function
def vf_histograms_api(counter, src, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT8)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_UINT8)
    result = vf.histograms(reg_src, preg)  # noqa: F841

@pl.vector_function
def vf_histogram_acc_api(counter, dst, src, dst_out):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT8)
    reg_dst = vf.load_align(dst, 0, dtype=pl.DT_UINT16)
    reg_src = vf.load_align(src, 0, dtype=pl.DT_UINT8)
    result = vf.histogram_acc(reg_dst, reg_src, preg)
    vf.store_align(dst_out, result, preg)

@pl.vector_function
def vf_unsqueeze_api(counter, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    result = vf.unsqueeze(preg)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_exp_sub_api(counter, src0, src1, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_src0 = vf.load_align(src0, 0, dtype=pl.DT_FP32)
    reg_src1 = vf.load_align(src1, 0, dtype=pl.DT_FP32)
    result = vf.exp_sub(reg_src0, reg_src1, preg, dtype=pl.DT_FP32)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_create_addr_reg_api(counter):
    vf_counter_inc(counter)
    for _ in pl.range(0, 1):
        ar = vf.create_addr_reg(0, 0, 0, 0, dtype=pl.DT_UINT32)  # noqa: F841

@pl.vector_function
def vf_load_unalign_init_api(counter):
    vf_counter_inc(counter)
    align_reg = vf.load_unalign_init()  # noqa: F841

@pl.vector_function
def vf_load_unalign_pre_api(counter, tile):
    vf_counter_inc(counter)
    align_reg = vf.load_unalign_init()
    vf.load_unalign_pre(align_reg, tile)

@pl.vector_function
def vf_load_unalign_api(counter, tile, dst):
    vf_counter_inc(counter)
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    align_reg = vf.load_unalign_init()
    vf.load_unalign_pre(align_reg, tile)
    result = vf.load_unalign(align_reg, tile)
    vf.store_align(dst, result, preg)

@pl.vector_function
def vf_store_unalign_api(counter, src, tile):
    vf_counter_inc(counter)
    align_reg = vf.unalign_reg_for_store()
    reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
    vf.store_unalign(tile, reg_src, align_reg, 0, post_update=True)

@pl.vector_function
def vf_store_unalign_post_api(counter, tile):
    vf_counter_inc(counter)
    align_reg = vf.unalign_reg_for_store()
    vf.store_unalign_post(tile, align_reg, 0, post_update=True)

@pl.vector_function
def vf_squeeze_store_unalign_api(counter, src, tile, count):
    vf_counter_inc(counter)
    if count > 1000:
        align_reg = vf.unalign_reg_for_store()
        reg_src = vf.load_align(src, 0, dtype=pl.DT_FP32)
        vf.squeeze_store_unalign(tile, reg_src, align_reg)

@pl.vector_function
def vf_squeeze_store_unalign_post_api(counter, tile, count):
    vf_counter_inc(counter)
    if count > 1000:
        align_reg = vf.unalign_reg_for_store()
        vf.squeeze_store_unalign_post(tile, align_reg)

@pl.vector_function
def vf_unalign_reg_for_store_api(counter):
    vf_counter_inc(counter)
    align_reg = vf.unalign_reg_for_store()  # noqa: F841

@pl.vector_function
def vf_create_mask_api(counter, preg_out):
    vf_counter_inc(counter)
    result = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    vf.store_align(preg_out, result)

@pl.vector_function
def vf_update_mask_api(counter, preg_out):
    vf_counter_inc(counter)
    result = vf.update_mask(1.0, dtype=pl.DT_FP32)
    vf.store_align(preg_out, result)

@pl.vector_function
def vf_mem_bar_api(counter):
    vf_counter_inc(counter)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)

@pl.vector_function
def vf_clear_spr_api(counter):
    vf_counter_inc(counter)
    vf.clear_spr()

@pl.vector_function
def vf_get_mask_spr_api(counter, preg_out):
    vf_counter_inc(counter)
    result = vf.get_mask_spr(width=pl.MaskWidth.B32)
    vf.store_align(preg_out, result)


_VF_OP_NAMES = (
    "full",
    "move",
    "gather",
    "bit_cast",
    "arange",
    "interleave",
    "de_interleave",
    "pack",
    "unpack",
    "load",
    "load_align",
    "store",
    "store_align",
    "scatter",
    "mask_gen_with_reg_tensor",
    "move_merg",
    "add",
    "sub",
    "mul",
    "div",
    "max",
    "min",
    "muls",
    "adds",
    "mins",
    "maxs",
    "neg",
    "abs",
    "sqrt",
    "relu",
    "leaky_relu",
    "prelu",
    "ln",
    "log",
    "log2",
    "log10",
    "exp",
    "mull",
    "addc",
    "subc",
    "and_",
    "or_",
    "xor",
    "not_",
    "shift_left",
    "shift_right",
    "eq",
    "ne",
    "lt",
    "gt",
    "le",
    "ge",
    "select",
    "squeeze",
    "astype",
    "truncate",
    "reduce_sum",
    "reduce_max",
    "reduce_min",
    "pair_reduce_sum",
    "mul_add_dst",
    "mul_add_dst_acc",
    "mul_dst_add",
    "mul_dst_add_acc",
    "axpy",
    "axpy_acc",
    "abs_sub",
    "muls_cast",
    "histograms",
    "histogram_acc",
    "unsqueeze",
    "exp_sub",
    "create_addr_reg",
    "load_unalign_init",
    "load_unalign_pre",
    "load_unalign",
    "store_unalign",
    "store_unalign_post",
    "squeeze_store_unalign",
    "squeeze_store_unalign_post",
    "unalign_reg_for_store",
    "create_mask",
    "update_mask",
    "mem_bar",
    "clear_spr",
    "get_mask_spr",
)


@pl.jit()
def test_vf_ops_kernel(
    index: pl.Tensor[[1, 64], pl.DT_UINT32],
    counter: pl.Tensor[[1, 64], pl.DT_UINT32],
    in0_f32: pl.Tensor[[1, 64], pl.DT_FP32],
    in1_f32: pl.Tensor[[1, 64], pl.DT_FP32],
    out0_f32: pl.Tensor[[1, 64], pl.DT_FP32],
    out1_f32: pl.Tensor[[1, 64], pl.DT_FP32],
    out2_u32: pl.Tensor[[1, 64], pl.DT_UINT32],
    out3_u32: pl.Tensor[[1, 64], pl.DT_UINT32],
    out4_u32: pl.Tensor[[1, 64], pl.DT_UINT32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tu = pl.TileType(shape=[1, 64], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    t_index = pl.make_tile(tu, addr=VA_INDEX)
    t_counter = pl.make_tile(tu, addr=VA_COUNTER)
    t_in0 = pl.make_tile(tf, addr=VA_IN0)
    t_in1 = pl.make_tile(tf, addr=VA_IN1)
    t_out0 = pl.make_tile(tf, addr=VA_OUT0)
    t_out1 = pl.make_tile(tf, addr=VA_OUT1)
    t_out2 = pl.make_tile(tu, addr=VA_OUT2)
    t_out3 = pl.make_tile(tu, addr=VA_OUT3)
    t_out4 = pl.make_tile(tu, addr=VA_OUT4)
    with pl.section_vector():
        pl.load(t_index, index, [0, 0])
        pl.load(t_counter, counter, [0, 0])
        pl.load(t_in0, in0_f32, [0, 0])
        pl.load(t_in1, in1_f32, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        idx = pl.getval(index, 0)
        if idx == 0:
            vf_full_api(t_counter, t_in0, t_out0)
        elif idx == 1:
            vf_move_api(t_counter, t_in0, t_out0)
        elif idx == 2:
            vf_gather_api(t_counter, t_in0, t_in1, t_out0)
        elif idx == 3:
            vf_bit_cast_api(t_counter, t_in0, t_out2)
        elif idx == 4:
            vf_arange_api(t_counter, t_out2)
        elif idx == 5:
            vf_interleave_api(t_counter, t_in0, t_in1, t_out0, t_out1)
        elif idx == 6:
            vf_de_interleave_api(t_counter, t_in0, t_in1, t_out0, t_out1)
        elif idx == 7:
            vf_pack_api(t_counter, t_in0, t_out2)
        elif idx == 8:
            vf_unpack_api(t_counter, t_in0, t_out2)
        elif idx == 9:
            vf_load_api(t_counter, t_in0, t_out0)
        elif idx == 10:
            vf_load_align_api(t_counter, t_in0, t_out0)
        elif idx == 11:
            vf_store_api(t_counter, t_in0, t_out0)
        elif idx == 12:
            vf_store_align_api(t_counter, t_in0, t_out0)
        elif idx == 13:
            vf_scatter_api(t_counter, t_in0, t_in1, t_out0)
        elif idx == 14:
            vf_mask_gen_with_reg_tensor_api(t_counter, t_in0, t_out2)
        elif idx == 15:
            vf_move_merg_api(t_counter, t_in0, t_in1, t_out0)
        elif idx == 16:
            vf_add_api(t_counter, t_in0, t_in1, t_out0)
        elif idx == 17:
            vf_sub_api(t_counter, t_in0, t_in1, t_out0)
        elif idx == 18:
            vf_mul_api(t_counter, t_in0, t_in1, t_out0)
        elif idx == 19:
            vf_div_api(t_counter, t_in0, t_in1, t_out0)
        elif idx == 20:
            vf_max_api(t_counter, t_in0, t_in1, t_out0)
        elif idx == 21:
            vf_min_api(t_counter, t_in0, t_in1, t_out0)
        elif idx == 22:
            vf_muls_api(t_counter, t_in0, t_out0)
        elif idx == 23:
            vf_adds_api(t_counter, t_in0, t_out0)
        elif idx == 24:
            vf_mins_api(t_counter, t_in0, t_out0)
        elif idx == 25:
            vf_maxs_api(t_counter, t_in0, t_out0)
        elif idx == 26:
            vf_neg_api(t_counter, t_in0, t_out0)
        elif idx == 27:
            vf_abs_api(t_counter, t_in0, t_out0)
        elif idx == 28:
            vf_sqrt_api(t_counter, t_in0, t_out0)
        elif idx == 29:
            vf_relu_api(t_counter, t_in0, t_out0)
        elif idx == 30:
            vf_leaky_relu_api(t_counter, t_in0, t_out0)
        elif idx == 31:
            vf_prelu_api(t_counter, t_in0, t_in1, t_out0)
        elif idx == 32:
            vf_ln_api(t_counter, t_in0, t_out0)
        elif idx == 33:
            vf_log_api(t_counter, t_in0, t_out0)
        elif idx == 34:
            vf_log2_api(t_counter, t_in0, t_out0)
        elif idx == 35:
            vf_log10_api(t_counter, t_in0, t_out0)
        elif idx == 36:
            vf_exp_api(t_counter, t_in0, t_out0)
        elif idx == 37:
            vf_mull_api(t_counter, t_in0, t_in1, t_out2, t_out3)
        elif idx == 38:
            vf_addc_api(t_counter, t_in0, t_in1, t_out2, t_out2)
        elif idx == 39:
            vf_subc_api(t_counter, t_in0, t_in1, t_out2, t_out2)
        elif idx == 40:
            vf_and__api(t_counter, t_in0, t_in1, t_out2)
        elif idx == 41:
            vf_or__api(t_counter, t_in0, t_in1, t_out2)
        elif idx == 42:
            vf_xor_api(t_counter, t_in0, t_in1, t_out2)
        elif idx == 43:
            vf_not__api(t_counter, t_in0, t_out2)
        elif idx == 44:
            vf_shift_left_api(t_counter, t_in0, t_out2)
        elif idx == 45:
            vf_shift_right_api(t_counter, t_in0, t_out2)
        elif idx == 46:
            vf_eq_api(t_counter, t_in0, t_in1, t_out2)
        elif idx == 47:
            vf_ne_api(t_counter, t_in0, t_in1, t_out2)
        elif idx == 48:
            vf_lt_api(t_counter, t_in0, t_in1, t_out2)
        elif idx == 49:
            vf_gt_api(t_counter, t_in0, t_in1, t_out2)
        elif idx == 50:
            vf_le_api(t_counter, t_in0, t_in1, t_out2)
        elif idx == 51:
            vf_ge_api(t_counter, t_in0, t_in1, t_out2)
        elif idx == 52:
            vf_select_api(t_counter, t_in0, t_in1, t_out0)
        elif idx == 53:
            vf_squeeze_api(t_counter, t_in0, t_out0)
        elif idx == 54:
            vf_astype_api(t_counter, t_in0, t_out2)
        elif idx == 55:
            vf_truncate_api(t_counter, t_in0, t_out0)
        elif idx == 56:
            vf_reduce_sum_api(t_counter, t_in0, t_out0)
        elif idx == 57:
            vf_reduce_max_api(t_counter, t_in0, t_out0)
        elif idx == 58:
            vf_reduce_min_api(t_counter, t_in0, t_out0)
        elif idx == 59:
            vf_pair_reduce_sum_api(t_counter, t_in0, t_out0)
        elif idx == 60:
            vf_mul_add_dst_api(t_counter, t_in0, t_in1, t_out0)
        elif idx == 61:
            vf_mul_add_dst_acc_api(t_counter, t_in0, t_in1, t_in0, t_out0)
        elif idx == 62:
            vf_mul_dst_add_api(t_counter, t_in0, t_in1, t_out0)
        elif idx == 63:
            vf_mul_dst_add_acc_api(t_counter, t_in0, t_in1, t_in0, t_out0)
        elif idx == 64:
            vf_axpy_api(t_counter, t_in0, t_out0)
        elif idx == 65:
            vf_axpy_acc_api(t_counter, t_in0, t_in1, t_out0)
        elif idx == 66:
            vf_abs_sub_api(t_counter, t_in0, t_in1, t_out0)
        elif idx == 67:
            vf_muls_cast_api(t_counter, t_in0, t_out0)
        elif idx == 68:
            vf_histograms_api(t_counter, t_in0, t_out2)
        elif idx == 69:
            vf_histogram_acc_api(t_counter, t_in0, t_in1, t_out2)
        elif idx == 70:
            vf_unsqueeze_api(t_counter, t_out0)
        elif idx == 71:
            vf_exp_sub_api(t_counter, t_in0, t_in1, t_out0)
        elif idx == 72:
            vf_create_addr_reg_api(t_counter)
        elif idx == 73:
            vf_load_unalign_init_api(t_counter)
        elif idx == 74:
            vf_load_unalign_pre_api(t_counter, t_in0)
        elif idx == 75:
            vf_load_unalign_api(t_counter, t_in0, t_out0)
        elif idx == 76:
            vf_store_unalign_api(t_counter, t_in0, t_out0)
        elif idx == 77:
            vf_store_unalign_post_api(t_counter, t_out0)
        elif idx == 78:
            vf_squeeze_store_unalign_api(t_counter, t_in0, t_out0, idx)
        elif idx == 79:
            vf_squeeze_store_unalign_post_api(t_counter, t_out0, idx)
        elif idx == 80:
            vf_unalign_reg_for_store_api(t_counter)
        elif idx == 81:
            vf_create_mask_api(t_counter, t_out2)
        elif idx == 82:
            vf_update_mask_api(t_counter, t_out2)
        elif idx == 83:
            vf_mem_bar_api(t_counter)
        elif idx == 84:
            vf_clear_spr_api(t_counter)
        elif idx == 85:
            vf_get_mask_spr_api(t_counter, t_out2)
        else:
            pass
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(counter, t_counter, [0, 0])
        pl.store(out0_f32, t_out0, [0, 0])
        pl.store(out1_f32, t_out1, [0, 0])
        pl.store(out2_u32, t_out2, [0, 0])
        pl.store(out3_u32, t_out3, [0, 0])
        pl.store(out4_u32, t_out4, [0, 0])

def _to_cpu_outputs(outputs):
    return [out.detach().cpu().clone() for out in outputs]


def _float_to_uint32_bits(tensor):
    return tensor.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF


def _torch_golden(idx, in0, in1):
    """PyTorch reference for each vf API, returning outputs [o0, o1, o2, o3, o4]."""
    zf = torch.zeros([1, 64], dtype=torch.float32)
    o0 = None
    o1 = None
    o2 = None
    o3 = None
    o4 = None

    if idx == 0:  # full, pos=LOWEST
        o0 = torch.full_like(in0, in0.flatten()[0])
    elif idx == 1:  # move
        o0 = in0.clone()
    elif idx == 2:  # gather with index from in1 bits
        index = (_float_to_uint32_bits(in1) % 64).long().reshape(-1)
        o0 = in0.flatten()[index].reshape(1, 64)
    elif idx == 3:  # bit_cast to uint32
        o2 = _float_to_uint32_bits(in0).to(torch.uint32)
    elif idx == 4:  # arange
        o2 = torch.arange(64, dtype=torch.int64).to(torch.uint32).reshape(1, 64)
    elif idx == 5:  # interleave: (o0, o1) = concat halves of interleaved pairs
        o0 = zf.clone()
        o1 = zf.clone()
        for i in range(32):
            o0[0, 2 * i] = in0[0, i]
            o0[0, 2 * i + 1] = in1[0, i]
            o1[0, 2 * i] = in0[0, 32 + i]
            o1[0, 2 * i + 1] = in1[0, 32 + i]
    elif idx == 6:  # de_interleave: even/odd elements of concatenated src0/src1
        o0 = zf.clone()
        o1 = zf.clone()
        for i in range(32):
            o0[0, i] = in0[0, 2 * i]
            o1[0, i] = in0[0, 2 * i + 1]
            o0[0, 32 + i] = in1[0, 2 * i]
            o1[0, 32 + i] = in1[0, 2 * i + 1]
    elif idx == 13:  # scatter with sequential indices
        o0 = in0.clone()
    elif idx in (7, 8, 14, 38, 39, 46, 47, 48, 49, 50, 51, 68, 69, 70, 81, 82, 85):
        return [None, None, None, None, None]
    elif idx in (9, 10, 11, 12, 75, 76, 78):  # load/store identity paths
        o0 = in0.clone()
    elif idx == 15:  # move_merg, src=in1
        o0 = in1.clone()
    elif idx == 16:
        o0 = in0 + in1
    elif idx == 17:
        o0 = in0 - in1
    elif idx == 18:
        o0 = in0 * in1
    elif idx == 19:
        o0 = in0 / in1
    elif idx == 20:
        o0 = torch.maximum(in0, in1)
    elif idx == 21:
        o0 = torch.minimum(in0, in1)
    elif idx == 22:
        o0 = in0 * 1.0
    elif idx == 23:
        o0 = in0 + 1.0
    elif idx == 24:
        o0 = torch.minimum(in0, torch.tensor(1.0))
    elif idx == 25:
        o0 = torch.maximum(in0, torch.tensor(1.0))
    elif idx == 26:
        o0 = -in0
    elif idx == 27:
        o0 = torch.abs(in0)
    elif idx == 28:
        o0 = torch.sqrt(in0)
    elif idx == 29:
        o0 = torch.relu(in0)
    elif idx == 30:  # leaky_relu with scalar slope 1.0
        o0 = in0.clone()
    elif idx == 31:  # prelu slope=in1
        o0 = torch.where(in0 >= 0, in0, in0 * in1)
    elif idx == 32:
        o0 = torch.log(in0)
    elif idx == 33:
        o0 = torch.log(in0)
    elif idx == 34:
        o0 = torch.log2(in0)
    elif idx == 35:
        o0 = torch.log10(in0)
    elif idx == 36:
        o0 = torch.exp(in0)
    elif idx == 37:  # mull
        product = _float_to_uint32_bits(in0) * _float_to_uint32_bits(in1)
        o2 = (product & 0xFFFFFFFF).to(torch.uint32)
        o3 = ((product >> 32) & 0xFFFFFFFF).to(torch.uint32)
    elif idx == 40:
        o2 = (_float_to_uint32_bits(in0) & _float_to_uint32_bits(in1)).to(torch.uint32)
    elif idx == 41:
        o2 = (_float_to_uint32_bits(in0) | _float_to_uint32_bits(in1)).to(torch.uint32)
    elif idx == 42:
        o2 = (_float_to_uint32_bits(in0) ^ _float_to_uint32_bits(in1)).to(torch.uint32)
    elif idx == 43:
        o2 = ((~_float_to_uint32_bits(in0)) & 0xFFFFFFFF).to(torch.uint32)
    elif idx == 44:
        o2 = ((_float_to_uint32_bits(in0) << 1) & 0xFFFFFFFF).to(torch.uint32)
    elif idx == 45:
        o2 = ((_float_to_uint32_bits(in0) >> 1) & 0xFFFFFFFF).to(torch.uint32)
    elif idx == 52:  # select, preg=ALL
        o0 = in0.clone()
    elif idx == 53:  # squeeze, preg=ALL
        o0 = in0.clone()
    elif idx == 54:  # astype FP32 -> INT32 (trunc), bit_cast to UINT32
        o2 = torch.from_numpy(in0.numpy().astype(np.int32).view(np.uint32).copy())
    elif idx == 55:
        o0 = torch.trunc(in0)
    elif idx == 56:
        o0 = zf.clone()
        o0[0, 0] = in0.sum()
    elif idx == 57:
        o0 = zf.clone()
        o0[0, 0] = in0.max()
    elif idx == 58:
        o0 = zf.clone()
        o0[0, 0] = in0.min()
    elif idx == 59:
        o0 = zf.clone()
        for i in range(32):
            o0[0, i] = in0[0, 2 * i] + in0[0, 2 * i + 1]
    elif idx == 61:  # mul_add_dst_acc: dst=in0, src0=in1, src1=in0
        o0 = in1 * in0 + in0
    elif idx == 63:  # mul_dst_add_acc: dst=in0, src0=in1, src1=in0
        o0 = in0 * in1 + in0
    elif idx == 65:  # axpy_acc: dst=in0, src=in1, scalar=1.0
        o0 = in1 + in0
    elif idx == 66:
        o0 = torch.abs(in0 - in1)
    elif idx == 67:  # muls_cast FP32 -> FP16 -> back to FP32 for storage
        o0 = (in0 * 1.0).to(torch.float16).to(torch.float32)
    elif idx == 71:
        o0 = torch.exp(in0 - in1)
    elif idx in (72, 73, 74, 77, 79, 80, 83, 84):
        return [None, None, None, None, None]
    else:
        return [None, None, None, None, None]
    return [o0, o1, o2, o3, o4]


def _compare_outputs(actual, expected):
    assert len(actual) == len(expected)
    for index, (actual_tensor, expected_tensor) in enumerate(zip(actual, expected)):
        if expected_tensor is None:
            continue
        assert actual_tensor.shape == expected_tensor.shape
        assert actual_tensor.dtype == expected_tensor.dtype
        if actual_tensor.dtype.is_floating_point:
            try:
                torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-5, atol=1e-5, equal_nan=True)
            except AssertionError:
                logging.info(
                    "DEBUG output[%d] actual=%s expected=%s",
                    index,
                    actual_tensor.flatten()[:8].tolist(),
                    expected_tensor.flatten()[:8].tolist(),
                )
                raise
        else:
            assert torch.equal(actual_tensor, expected_tensor), f"output[{index}] mismatch"


@pytest.mark.soc("950", "960")
def test_vf_ops():
    logging.info("Setting device %s", ST_DEVICE)
    torch.npu.set_device(ST_DEVICE)
    logging.info("Setting device %s done.", ST_DEVICE)
    torch.manual_seed(2026)

    index = torch.zeros([1, 64], dtype=torch.uint32).to(ST_DEVICE)
    counter = torch.zeros([1, 64], dtype=torch.uint32).to(ST_DEVICE)
    in0_f32 = torch.rand([1, 64], dtype=torch.float32, device=ST_DEVICE) * 2.0 + 0.1
    in1_f32 = torch.rand([1, 64], dtype=torch.float32, device=ST_DEVICE) * 2.0 + 0.1
    outputs = [
        torch.zeros([1, 64], dtype=torch.float32, device=ST_DEVICE),
        torch.zeros([1, 64], dtype=torch.float32, device=ST_DEVICE),
        torch.zeros([1, 64], dtype=torch.uint32).to(ST_DEVICE),
        torch.zeros([1, 64], dtype=torch.uint32).to(ST_DEVICE),
        torch.zeros([1, 64], dtype=torch.uint32).to(ST_DEVICE),
    ]

    in0_cpu = in0_f32.detach().cpu()
    in1_cpu = in1_f32.detach().cpu()
    logging.info("Testing start")
    for idx in range(86):
        index = torch.full([1, 64], idx, dtype=torch.uint32).to(ST_DEVICE)
        torch.npu.synchronize()
        logging.info("Running vf op %d (%s)", idx, _VF_OP_NAMES[idx])
        test_vf_ops_kernel[1](index, counter, in0_f32, in1_f32, *outputs)
        logging.info("Running vf op %d (%s) done", idx, _VF_OP_NAMES[idx])
        torch.npu.synchronize()

        expected_counter = torch.full([1, 64], idx + 1, dtype=torch.uint32)
        _counter_cpu = counter.detach().cpu()
        assert torch.equal(_counter_cpu, expected_counter), (
            f"counter mismatch at idx={idx}: actual={_counter_cpu.flatten()[:8].tolist()}"
        )

        cpu_outputs = _to_cpu_outputs(outputs)
        golden = _torch_golden(idx, in0_cpu, in1_cpu)
        _compare_outputs(cpu_outputs, golden)


_IR_GOLDEN_PATH = Path(__file__).with_suffix(".pypto")


@pytest.mark.soc("950", "960")
def test_vf_ops_ir_dump():
    """The Vector-target IR of every vf API matches the sibling .pypto dump."""
    program, matched = test_vf_ops_kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
    assert matched
    # Memref ids are numbered process-wide, so they are compared as memref_id=<id>.
    actual = re.sub(r"memref_id=\d+", "memref_id=<id>", ir.text_dump(program))
    if os.environ.get("PYPTO_RENDER_IR"):
        _IR_GOLDEN_PATH.write_text(actual.strip() + "\n", encoding="utf-8")
        logging.info("Updated: %s", _IR_GOLDEN_PATH)
        return
    golden = _IR_GOLDEN_PATH.read_text(encoding="utf-8")
    if actual.strip() != golden.strip():
        diff = "".join(
            difflib.unified_diff(golden.splitlines(keepends=True), actual.splitlines(keepends=True))
        )
        raise AssertionError(f"IR dump mismatch in {_IR_GOLDEN_PATH}:\n{diff}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    test_vf_ops()
