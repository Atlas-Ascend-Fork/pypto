# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# LiteNPU matmul quant 参数转换工具（kirin9030 / KirinX90 / kirin9050 共用）。
# 供量化用例生成 quant_pre / requant 的 scale（fp19 打包，可选 sign 位）与 golden 用 scale（fp19 解码），
# 测试用例应调用本工具的接口，不要在用例内联做打包/解码位运算。

import torch

FP19_LOW_MANTISSA_MASK = 0x1FFF  # 低 13 位尾数


def trans_quant_param(scale: torch.Tensor, offset: int = 0, is_signed: bool = False) -> torch.Tensor:
    """CPU 转换 quant scale 为硬件格式的 int64 打包 tensor。

    低 32 位存放 fp19 scale（1 位符号 + 8 位指数 + 10 位尾数），kirin9050 matmul
    宏指令 pre-stage quant/dequant 的 per-channel scale 使用该格式：把 fp32（23 位尾数）
    逐元素截断低 13 位尾数（mantissa 从左到右、高位到低位保留 10 位）。

    kirin9030 / KirinX90 FIXP VREQ8 requant（int32 acc→int8，L0C_COPY_OUT 的 FixPipe）
    在 int64 高比特位额外打包 offset 与 sign 位：
      - bits[37,46)  offset（9 位，round 前加在 acc*scale 上，requant 恒取 0）
      - bit46        sign（1=有符号 [-128,127]，0=无符号 [0,255]）

    宏指令只读低 32 位的 fp19，offset/sign 高比特位对其无影响，故 requant 与宏指令
    可共用本接口：s8 有符号 requant 传 is_signed=True，其余量化路径用默认值即可。

    用法：
        scale_fp32 = torch.full((1, n), 0.2)
        quant_pre = trans_quant_param(scale_fp32)                     # 纯 fp19（宏指令）
        requant_pre = trans_quant_param(scale_fp32, is_signed=True)   # VREQ8，sign=1
        extend_params = {'scale_tensor': quant_pre}
        pypto.matmul(a, b, pypto.DT_FP16, extend_params=extend_params)

    注意：
        - 输入须为 fp32 tensor，per-channel 场景 shape 为 (1, n)，n 与 mat2 的 N 维度一致；
        - 返回值直接作为 extend_params 的 scale_tensor 传入，勿再做 dtype 转换。
    """
    if scale.dtype != torch.float32:
        raise ValueError(f"scale must be fp32 tensor, got {scale.dtype}")
    bits = scale.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    bits = bits & ~FP19_LOW_MANTISSA_MASK
    return bits | ((offset & 0x1FF) << 37) | ((1 if is_signed else 0) << 46)


def decode_quant_param(packed: torch.Tensor) -> torch.Tensor:
    """把 trans_quant_param 打包的 int64 scale 按位解码回 fp32 近似值。

    fp19 打包值即截断尾数后的 fp32 bit pattern（低 32 位），可直接按位解码回 fp32，
    用于量化用例的 golden 计算（与硬件侧生效的 fp19 scale 保持一致）。

    用法：
        packed = trans_quant_param(scale_fp32)   # int64, shape (1, n)
        scale_golden = decode_quant_param(packed)  # fp32, shape (1, n)
    """
    return (packed & 0xFFFFFFFF).to(torch.int32).view(torch.float32)
