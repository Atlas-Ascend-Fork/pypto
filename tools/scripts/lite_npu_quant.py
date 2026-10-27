# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# LiteNPU(kirin9050) matmul 宏指令 pre-stage quant 参数转换工具。
# 供 kirin9050 量化用例生成 quant_pre（fp19 打包）与 golden 用 scale（fp19 解码），
# 测试用例应调用本工具的接口，不要在用例内联做打包/解码位运算。

import torch

FP19_LOW_MANTISSA_MASK = 0x1FFF  # 低 13 位尾数


def trans_quant_param(scale: torch.Tensor) -> torch.Tensor:
    """CPU 转换 quant_pre scale 为 kirin9050 matmul 宏指令的硬件格式。

    kirin9050 宏指令 pre-stage quant/dequant 的 per-channel scale 使用 fp19 格式
    （1 位符号 + 8 位指数 + 10 位尾数）。本接口将用户传入的 fp32（23 位尾数）
    tensor 逐元素截断低 13 位尾数（mantissa 从左到右、高位到低位保留 10 位），
    返回 int64 打包 tensor。

    用法：
        scale_fp32 = torch.full((1, n), 0.2)
        quant_pre = trans_quant_param(scale_fp32)  # int64, shape (1, n)
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
    return bits


def decode_quant_param(packed: torch.Tensor) -> torch.Tensor:
    """把 trans_quant_param 打包的 int64 scale 按位解码回 fp32 近似值。

    fp19 打包值即截断尾数后的 fp32 bit pattern，可直接按位解码回 fp32，
    用于量化用例的 golden 计算（与硬件侧生效的 fp19 scale 保持一致）。

    用法：
        packed = trans_quant_param(scale_fp32)   # int64, shape (1, n)
        scale_golden = decode_quant_param(packed)  # fp32, shape (1, n)
    """
    return (packed & 0xFFFFFFFF).to(torch.int32).view(torch.float32)
