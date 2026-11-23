# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import pypto
from pypto.pil.compile_pipeline import compile_new_ir

from .test_common import npuarch


def c2v2_kernel(
    input_tensor_a: pypto.Tensor[[pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC]],
    input_tensor_b: pypto.Tensor[[pypto.DYNAMIC, pypto.STATIC, pypto.DYNAMIC]],
    input_tensor_w: pypto.Tensor[[pypto.STATIC, pypto.STATIC]],
    output_tensor1: pypto.Tensor[[pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC]],
    output_tensor2: pypto.Tensor[[pypto.DYNAMIC, pypto.STATIC]],
    output_tensor3: pypto.Tensor[[pypto.DYNAMIC, pypto.DYNAMIC, pypto.STATIC]],
):
    tile_b = 32
    tile_s = 2
    batch_size, seq_len = input_tensor_a.shape[:2]
    hidden = input_tensor_a.shape[2]
    b_loop = (batch_size + tile_b - 1) // tile_b
    s_loop = (seq_len + tile_s - 1) // tile_s
    dtype = input_tensor_b.dtype

    out2_buf = pypto.tensor([128, 9, 2 * 64], dtype=dtype)

    for b_idx in pypto.loop(b_loop, name="Loop_B", idx_name="b_idx"):
        for s_idx in pypto.loop(s_loop, name="Loop_S", idx_name="s_idx"):
            b_valid = pypto.min(tile_b, batch_size - b_idx * tile_b)
            s_valid = pypto.min(tile_s, seq_len - s_idx * tile_s)
            bs_valid = b_valid * s_valid
            pypto.set_vec_tile_shapes(64, 64, 64)
            pypto.set_cube_tile_shapes([32, 32], [64, 64], [32, 32])

            input_a_view = input_tensor_a[b_idx * tile_b:(b_idx + 1) * tile_b,
                                          s_idx * tile_s:(s_idx + 1) * tile_s, :]
            input_b_view = input_tensor_b[b_idx * tile_b:(b_idx + 1) * tile_b, :,
                                          s_idx * tile_s:(s_idx + 1) * tile_s]

            # ================= scope1: bmm -> bmm -> v1 -> v2 =================
            # C1: bmm(input_a_view[M,K,N], input_b_view[M,N,K]) = [tile_b, tile_s, tile_s]
            # C2: bmm(C1[tile_b,tile_s,tile_s], input_a_view[tile_b,tile_s,N]) = [tile_b, tile_s, N]
            # V1: add (binary) — input_a_view + C2 — 3D
            # V2: mul (binary) — input_a_view + V1 — 3D
            # BMM Mix场景L0C2UB需要使能多核切K，UB2L1正在适配中
            pypto.set_pass_options(sg_set_scope=(1, True, False))
            c1 = pypto.matmul(input_a_view, input_b_view, out_dtype=dtype)
            c2 = pypto.matmul(c1, input_a_view, out_dtype=dtype)
            v1 = pypto.add(input_a_view, c2)
            v2 = pypto.mul(input_a_view, v1)
            pypto.set_pass_options(sg_set_scope=-1)
            # scope1 end

            output_tensor1[b_idx * tile_b:(b_idx + 1) * tile_b,
                           s_idx * tile_s:(s_idx + 1) * tile_s,
                           :] = v2

            # ================= scope2: C3->V3->C4->T->V4 =================
            # V2 reshape to 2D: [tile_b*tile_s, hidden]
            # C3: matmul(V2_2d, Incast(input_tensor_w)) — 2D
            # V3: sub (binary) — C3 + Incast(V2_2d view) — 2D
            # C4: matmul(V3, Incast(input_tensor_w)) — 2D
            # T: out_tmp tensor, assemble(C3, V3, C4) then view with valid_shape
            # V4: exp (unary) — T view + Incast view
            # pypto.set_pass_options(sg_set_scope=(2, True, False))
            v2_2d = pypto.reshape(v2, [tile_b * tile_s, hidden],
                                  valid_shape=[bs_valid, hidden])
            incast_v2_view = pypto.view(v2_2d, [tile_b * tile_s, hidden], [0, 0],
                                        valid_shape=[bs_valid, hidden])
            pypto.set_pass_options(sg_set_scope=(2, True, False))
            pypto.set_cube_tile_shapes([32, 32], [64, 64], [32, 32])
            c3 = pypto.matmul(incast_v2_view, input_tensor_w, out_dtype=dtype)
            pypto.set_vec_tile_shapes(64, 64)
            v3 = pypto.sub(c3, incast_v2_view)
            pypto.set_cube_tile_shapes([32, 32], [64, 64], [32, 32])
            c4 = pypto.matmul(v3, input_tensor_w, out_dtype=dtype)
            # pypto.set_pass_options(sg_set_scope=-1)

            out_tmp = pypto.tensor([tile_b * tile_s, c3.shape[-1] + v3.shape[-1] + c4.shape[-1]], dtype=dtype)
            pypto.assemble(c3, [0, 0], out_tmp)
            pypto.assemble(v3, [0, c3.shape[-1]], out_tmp)
            pypto.assemble(c4, [0, c3.shape[-1] + v3.shape[-1]], out_tmp)
            t_view = pypto.view(out_tmp, [tile_b * tile_s, c3.shape[-1] + v3.shape[-1] + c4.shape[-1]], [0, 0],
                                valid_shape=[bs_valid, c3.shape[-1] + v3.shape[-1] + c4.shape[-1]])
            t_view_h = pypto.view(t_view, [tile_b * tile_s, 2 * c3.shape[-1]], [0, 0],
                                  valid_shape=[bs_valid, 2 * c3.shape[-1]])
            v4 = pypto.exp(t_view_h)
            pypto.set_pass_options(sg_set_scope=-1)
            # scope2 end

            v4_3d = pypto.reshape(v4, [tile_b, tile_s, 2 * 64],
                                  valid_shape=[b_valid, s_valid, 2 * 64])
            out2_buf[b_idx * tile_b:(b_idx + 1) * tile_b,
                     s_idx * tile_s:(s_idx + 1) * tile_s, :] = v4_3d

            # ================= scope3: vector组合运算 =================
            # exp(v4_3d first half) + rsqrt(sum(v2*v2, dim=-1, keepdim))
            # concat with exp(input_a_view) and exp(input_b_view first half)
            # output to output_tensor3 [B, S, hidden]
            pypto.set_pass_options(sg_set_scope=(3, True, False))
            pypto.set_vec_tile_shapes(64, 64, 64)
            v4_h = pypto.view(v4_3d, [tile_b, tile_s, 64], [0, 0, 0],
                              valid_shape=[b_valid, s_valid, 64])
            exp_v4 = pypto.exp(v4_h)
            sq_v2 = v2 * v2
            sum_v2 = pypto.sum(sq_v2, dim=-1, keepdim=True)
            rsqrt_v2 = pypto.rsqrt(sum_v2)
            exp_a = pypto.exp(input_a_view)
            b_view_reshaped = pypto.transpose(input_b_view, 1, 2)
            b_view_h = pypto.view(b_view_reshaped, [tile_b, tile_s, 64], [0, 0, 0],
                                  valid_shape=[b_valid, s_valid, 64])
            exp_b = pypto.exp(b_view_h)
            cat_v = pypto.concat([exp_v4, rsqrt_v2, exp_a, exp_b], dim=-1)
            cat_v_h = pypto.view(cat_v, [tile_b, tile_s, 64], [0, 0, 0],
                                 valid_shape=[b_valid, s_valid, 64])
            output_tensor3[b_idx * tile_b:(b_idx + 1) * tile_b,
                           s_idx * tile_s:(s_idx + 1) * tile_s, :] = cat_v_h
            pypto.set_pass_options(sg_set_scope=-1)
            # scope3 end

    out2_2d = pypto.reshape(out2_buf, [128 * 9, 2 * 64])
    output_tensor2[:, :] = out2_2d


def test_c2v2_kernel():
    b = pypto.symbolic_scalar('b')
    s = pypto.symbolic_scalar('s')
    with npuarch("DAV_3510"):
        compile_new_ir(c2v2_kernel,
            input_tensor_a=pypto.tensor([b, s, 64], pypto.DT_FP32),
            input_tensor_b=pypto.tensor([b, 64, s], pypto.DT_FP32),
            input_tensor_w=pypto.tensor([64, 64], pypto.DT_FP32),
            output_tensor1=pypto.tensor([b, s, 64], pypto.DT_FP32),
            output_tensor2=pypto.tensor([b*s, 128], pypto.DT_FP32),
            output_tensor3=pypto.tensor([b, s, 64], pypto.DT_FP32),
            create_new_logical_tensor=True,
        )
