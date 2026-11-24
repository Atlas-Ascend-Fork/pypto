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
 * \file quantize.h
 * \brief INT8 对称/非对称量化 Tile 算子
 *
 * - INT8_SYM:  FP32 -> INT8,  范围 [-128, 127]
 * - INT8_ASYM: FP32 -> UINT8, 范围 [0, 255]
 *
 * 支持逐行量化 (axis=-1, scale 行广播) 与逐列量化 (axis=-2, scale 列广播)，
 * 由模板参数 quantAxis 编译期分派，无需在 Operation 层做 Transpose。
 */

#ifndef TILEOP_TILE_OPERATOR_QUANTIZE__H
#define TILEOP_TILE_OPERATOR_QUANTIZE__H

#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/sync.h"
#include "utils/tile_tensor.h"
#include <type_traits>

/// Tile 内存对齐字节数
constexpr size_t TILE_ALIGNMENT_BYTES = TileOp::BLOCK_SIZE;

/// 向上取整到对齐边界
#ifndef PTO_CEIL
#define PTO_CEIL(x, y) ((((x) + (y) - 1) / (y)) * (y))
#endif

#if !defined(PTO_NPU_ARCH_A5) && !defined(PTO_NPU_ARCH_A6)
template <typename Dst, typename Src>
TILEOP void QuantizeS32ToF16(Dst& dst, Src& src)
{
    // VCONV.DEQ uses 8-bit repeat strides in 32-byte blocks.
    constexpr unsigned srcRowBytes = Src::RowStride * sizeof(int32_t);
    constexpr unsigned dstRowBytes = Dst::RowStride * sizeof(half);
    if constexpr (srcRowBytes / TILE_ALIGNMENT_BYTES > pto::REPEAT_STRIDE_MAX ||
                  dstRowBytes / TILE_ALIGNMENT_BYTES > pto::REPEAT_STRIDE_MAX) {
        constexpr unsigned elemsPerRepeat = pto::REPEAT_BYTE / sizeof(int32_t);
        const auto validCols = dst.GetValidCol();
        const auto validRows = dst.GetValidRow();
        if (validRows > 1 && validCols % elemsPerRepeat != 0) {
            using SrcRow = pto::Tile<pto::TileType::Vec, int32_t, 1, Src::Cols, pto::BLayout::RowMajor, -1, -1>;
            using DstRow = pto::Tile<pto::TileType::Vec, half, 1, Dst::Cols, pto::BLayout::RowMajor, -1, -1>;
            SrcRow srcRow(1, validCols);
            DstRow dstRow(1, validCols);
            const auto srcAddr = reinterpret_cast<uint64_t>(src.data());
            const auto dstAddr = reinterpret_cast<uint64_t>(dst.data());
            for (LoopVar row = 0; row < validRows; ++row) {
                pto::TASSIGN(srcRow, srcAddr + row * srcRowBytes);
                pto::TASSIGN(dstRow, dstAddr + row * dstRowBytes);
                pto::TCVT(dstRow, srcRow, pto::RoundMode::CAST_RINT);
            }
            return;
        }
    }
    pto::TCVT(dst, src, pto::RoundMode::CAST_RINT);
}
#endif

// =============================================================================
// INT8 对称量化
// =============================================================================
#define OP_TILE_OP_TQUANT_INT8_SYM TQuantInt8Sym

/**
 * @brief INT8 对称量化
 * @param dst   输出 INT8 张量, 形状 [..., H, W]
 * @param src   输入 FP32 张量, 形状 [..., H, W]
 * @param scale 缩放因子: axis=-1 形状 [..., H] (行广播), axis=-2 形状 [..., W] (列广播)
 * @param tmp   临时缓冲区
 * @tparam quantAxis 量化轴: -1 逐行量化 / -2 逐列量化
 *
 * ISA 指令拼接 (不依赖 TQUANT 组合指令的内部 tmp 布局):
 *   scaled = src * broadcast(scale) -> s32(RINT) -> f16(RINT) -> int8(RINT, 饱和)
 * tmp 分段 (按 float/int32 元素计, 与 host 侧 workspaceSize 公式逐字一致):
 *   [scaled/f16 平面 (ceil8(H) x ceil8(W))][s32 平面 (同尺寸)][axis=-1 行广播 vbrcb 缓冲 (ceil8(H) x 8)]
 */
template <pto::QuantType quantType, int quantAxis, typename T0, typename T1, typename T2, typename T3>
TILEOP void TQuantInt8Sym(T0 dst, T1 src, T2 scale, T3 tmp)
{
    static_assert(quantType == pto::QuantType::INT8_SYM, "TQuantInt8Sym only supports INT8_SYM.");
    static_assert(quantAxis == -1 || quantAxis == -2, "TQuantInt8Sym only supports axis=-1/-2.");

    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    const auto scaleLayout = scale.GetLayout();

    // 获取形状
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    if (dstShape3 == 0 || dstShape4 == 0) {
        return;
    }

    auto srcShape3 = srcLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto srcShape4 = srcLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    // 获取步长
    auto dstStride0 = dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto dstStride1 = dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto dstStride2 = dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();

    auto srcStride0 = srcLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto srcStride1 = srcLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto srcStride2 = srcLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();

    auto scaleStride1 = scaleLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto scaleStride2 = scaleLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto scaleStride3 = scaleLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();

    // 获取 Tile 形状并计算对齐 (UB 局部 buffer 末维按 32B 对齐)
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    constexpr int paddedCol_dst = PTO_CEIL(dstTileW, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(int8_t)));

    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, DIM_4TH, MAX_DIMS>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, DIM_5TH, MAX_DIMS>();
    constexpr int paddedCol_src = PTO_CEIL(srcTileW, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(float)));

    constexpr auto scaleTileW = TileOp::GetTensorTileShapeDim<T2, DIM_5TH, MAX_DIMS>();
    constexpr int paddedRow_scale = PTO_CEIL(scaleTileW, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(float)));

    // 数据类型
    using DstDtype = typename T0::Type;
    using SrcDtype = typename T1::Type;
    using ScaleDtype = typename T2::Type;

    // 输出/输入/中转 tile (两轴同几何: tile 内紧凑, 无切片/共享块)
    using DstTileDefine = pto::Tile<pto::TileType::Vec, DstDtype, dstTileH, paddedCol_dst, pto::BLayout::RowMajor, -1,
                                    -1>;
    using SrcTileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, srcTileH, paddedCol_src, pto::BLayout::RowMajor, -1,
                                    -1>;
    using ScaledTileDefine = pto::Tile<pto::TileType::Vec, float, srcTileH, paddedCol_src, pto::BLayout::RowMajor, -1,
                                       -1>;
    using S32TileDefine = pto::Tile<pto::TileType::Vec, int32_t, srcTileH, paddedCol_src, pto::BLayout::RowMajor, -1,
                                    -1>;
    // fp16 视图复用 scaled 平面: 列宽按 half 计为 fp32 区的 2 倍 (字节等宽, 行跨步一致)
    constexpr int paddedCol_f16 = paddedCol_src * 2;
    using F16TileDefine = pto::Tile<pto::TileType::Vec, half, srcTileH, paddedCol_f16, pto::BLayout::RowMajor, -1, -1>;

    // scale tile 按量化轴选择: -1 行广播 ColMajor 列向量 / -2 列广播 RowMajor 行向量
    using ScaleRowTileDefine = pto::Tile<pto::TileType::Vec, ScaleDtype, paddedRow_scale, 1, pto::BLayout::ColMajor, -1,
                                         -1>;
    using ScaleColTileDefine = pto::Tile<pto::TileType::Vec, ScaleDtype, 1, paddedRow_scale, pto::BLayout::RowMajor, -1,
                                         -1>;

#if !defined(PTO_NPU_ARCH_A5) && !defined(PTO_NPU_ARCH_A6)
    // A2/A3 行广播 vbrcb 缓冲 (仅 axis=-1 使用), 实际写入 ceil(H/8)*256B
    constexpr int bcastRows = PTO_CEIL(srcTileH < 256 ? srcTileH : 256, 8);
    using BcastTileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, bcastRows, 64, pto::BLayout::RowMajor, -1, -1>;
#endif

    // tmp 分段 (元素数, 与 host 侧公式逐字一致): [scaled/f16][s32] (A2/A3 另有 [bcast])
    constexpr int planeElems = PTO_CEIL(srcTileH, 8) * paddedCol_src;
    constexpr int scaledOff = 0;
    constexpr int s32Off = planeElems;
#if !defined(PTO_NPU_ARCH_A5) && !defined(PTO_NPU_ARCH_A6)
    constexpr int bcastOff = 2 * planeElems;
#endif

    // 遍历所有 Tile
    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                DstTileDefine dstTile(dstShape3, dstShape4);
                SrcTileDefine srcTile(srcShape3, srcShape4);
                ScaledTileDefine scaledTile(srcShape3, srcShape4);
                S32TileDefine s32Tile(srcShape3, srcShape4);
                F16TileDefine f16Tile(srcShape3, srcShape4);

                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                auto scaleOffset = n0Index * scaleStride1 + n1Index * scaleStride2 + n2Index * scaleStride3;

                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * sizeof(DstDtype)));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * sizeof(SrcDtype)));

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6)
                (void)tmp;
                if constexpr (quantAxis == -2) {
                    // A5/A6 列广播: TCOLEXPANDMUL + 寄存器级 TCVT 链。
                    // A5/A6 的 tile 级 TCVT 不支持 int32->half, 需与原生链一致: fp32 -> s32 -> fp32 -> f16 -> int8。
                    // 平面复用: scaled 平面先存 scaled fp32, s32 转换后再接回 s32->fp32; f16 复用 s32 平面。
                    ScaleColTileDefine scaleTile(1, srcShape4);
                    pto::TASSIGN(scaleTile, (uint64_t)(scale.GetAddr() + scaleOffset * sizeof(ScaleDtype)));
                    pto::TASSIGN(scaledTile, (uint64_t)(tmp.GetAddr()) + scaledOff * sizeof(int32_t));
                    pto::TASSIGN(s32Tile, (uint64_t)(tmp.GetAddr()) + s32Off * sizeof(int32_t));
                    pto::TASSIGN(f16Tile, (uint64_t)(tmp.GetAddr()) + s32Off * sizeof(int32_t));
                    SyncV();
                    pto::TCOLEXPANDMUL(scaledTile, srcTile, scaleTile);
                    SyncV();
                    pto::TCVT(s32Tile, scaledTile, pto::RoundMode::CAST_RINT); // fp32 -> s32
                    SyncV();
                    pto::TCVT(scaledTile, s32Tile, pto::RoundMode::CAST_RINT); // s32 -> fp32
                    SyncV();
                    pto::TCVT(f16Tile, scaledTile, pto::RoundMode::CAST_RINT); // fp32 -> f16
                    SyncV();
                    pto::TCVT(dstTile, f16Tile, pto::RoundMode::CAST_RINT, pto::SaturationMode::ON);
                } else {
                    // A5/A6 原生 INT8 量化: 行 scale 由 vlds BRC_B32 广播, 整条转换链在寄存器内完成, 不使用 tmp
                    ScaleRowTileDefine scaleTile(srcShape3, 1);
                    pto::TASSIGN(scaleTile, (uint64_t)(scale.GetAddr() + scaleOffset * sizeof(ScaleDtype)));
                    pto::TQUANT<quantType>(dstTile, srcTile, scaleTile);
                }
#else
                pto::TASSIGN(scaledTile, (uint64_t)(tmp.GetAddr()) + scaledOff * sizeof(int32_t));
                pto::TASSIGN(f16Tile, (uint64_t)(tmp.GetAddr()) + scaledOff * sizeof(int32_t));
                pto::TASSIGN(s32Tile, (uint64_t)(tmp.GetAddr()) + s32Off * sizeof(int32_t));

                if constexpr (quantAxis == -2) {
                    // 列广播: scale[c] 沿 H 方向广播
                    ScaleColTileDefine scaleTile(1, srcShape4);
                    pto::TASSIGN(scaleTile, (uint64_t)(scale.GetAddr() + scaleOffset * sizeof(ScaleDtype)));
                    pipe_barrier(PIPE_V);
                    pto::TCOLEXPANDMUL(scaledTile, srcTile, scaleTile);
                    pipe_barrier(PIPE_V);
                } else {
                    // 行广播: scale[r] 沿 W 方向广播
                    ScaleRowTileDefine scaleTile(srcShape3, 1);
                    BcastTileDefine bcastTile;
                    pto::TASSIGN(scaleTile, (uint64_t)(scale.GetAddr() + scaleOffset * sizeof(ScaleDtype)));
                    pto::TASSIGN(bcastTile, (uint64_t)(tmp.GetAddr()) + bcastOff * sizeof(int32_t));
                    pipe_barrier(PIPE_V);
                    pto::TROWEXPANDMUL(scaledTile, srcTile, scaleTile, bcastTile);
                    pipe_barrier(PIPE_V);
                }

                // fp32 -> s32 -> fp16 -> int8: 与 golden 的 round+饱和语义一致
                pto::TCVT(s32Tile, scaledTile, pto::RoundMode::CAST_RINT);
                pipe_barrier(PIPE_V);
                QuantizeS32ToF16(f16Tile, s32Tile);
                pipe_barrier(PIPE_V);
                pto::TCVT(dstTile, f16Tile, pto::RoundMode::CAST_RINT, pto::SaturationMode::ON);
                pipe_barrier(PIPE_V);
#endif
            }
        }
    }
}

// =============================================================================
// INT8 非对称量化
// =============================================================================
#define OP_TILE_OP_TQUANT_INT8_ASYM TQuantInt8Asym

/**
 * @brief INT8 非对称量化
 * @param dst    输出 UINT8 张量, 形状 [..., H, W]
 * @param src    输入 FP32 张量, 形状 [..., H, W]
 * @param scale  缩放因子: axis=-1 形状 [..., H] (行广播), axis=-2 形状 [..., W] (列广播)
 * @param offset 零点偏移, 形状与 scale 相同
 * @param tmp    临时缓冲区
 * @tparam quantAxis 量化轴: -1 逐行量化 / -2 逐列量化
 *
 * ISA 指令拼接, 与 TQuantInt8Sym 同构, 仅多一步 offset 广播加:
 *   rounded = src * broadcast(scale) + broadcast(offset) -> s32(RINT) -> f16(RINT) -> uint8(RINT, 饱和)
 * tmp 分段与 TQuantInt8Sym 完全一致。
 */
template <pto::QuantType quantType, int quantAxis, typename T0, typename T1, typename T2, typename T3, typename T4>
TILEOP void TQuantInt8Asym(T0 dst, T1 src, T2 scale, T3 offset, T4 tmp)
{
    static_assert(quantType == pto::QuantType::INT8_ASYM, "TQuantInt8Asym only supports INT8_ASYM.");
    static_assert(quantAxis == -1 || quantAxis == -2, "TQuantInt8Asym only supports axis=-1/-2.");

    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    const auto scaleLayout = scale.GetLayout();
    const auto offsetLayout = offset.GetLayout();

    // 获取形状
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    if (dstShape3 == 0 || dstShape4 == 0) {
        return;
    }

    auto srcShape3 = srcLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto srcShape4 = srcLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    // 获取步长
    auto dstStride0 = dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto dstStride1 = dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto dstStride2 = dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();

    auto srcStride0 = srcLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto srcStride1 = srcLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto srcStride2 = srcLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();

    auto scaleStride1 = scaleLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto scaleStride2 = scaleLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto scaleStride3 = scaleLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();

    auto offsetStride1 = offsetLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto offsetStride2 = offsetLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    auto offsetStride3 = offsetLayout.template GetStrideDim<DIM_4TH, MAX_DIMS>();

    // 获取 Tile 形状并计算对齐 (UB 局部 buffer 末维按 32B 对齐)
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    constexpr int paddedCol_dst = PTO_CEIL(dstTileW, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(uint8_t)));

    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, DIM_4TH, MAX_DIMS>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, DIM_5TH, MAX_DIMS>();
    constexpr int paddedCol_src = PTO_CEIL(srcTileW, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(float)));

    constexpr auto scaleTileW = TileOp::GetTensorTileShapeDim<T2, DIM_5TH, MAX_DIMS>();
    constexpr int paddedRow_scale = PTO_CEIL(scaleTileW, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(float)));

    // 数据类型
    using DstDtype = typename T0::Type;
    using SrcDtype = typename T1::Type;
    using ScaleDtype = typename T2::Type;
    using OffsetDtype = typename T3::Type;

    // 输出/输入/中转 tile (两轴同几何: tile 内紧凑, 无切片/共享块)
    using DstTileDefine = pto::Tile<pto::TileType::Vec, DstDtype, dstTileH, paddedCol_dst, pto::BLayout::RowMajor, -1,
                                    -1>;
    using SrcTileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, srcTileH, paddedCol_src, pto::BLayout::RowMajor, -1,
                                    -1>;
    using ScaledTileDefine = pto::Tile<pto::TileType::Vec, float, srcTileH, paddedCol_src, pto::BLayout::RowMajor, -1,
                                       -1>;
    using S32TileDefine = pto::Tile<pto::TileType::Vec, int32_t, srcTileH, paddedCol_src, pto::BLayout::RowMajor, -1,
                                    -1>;
    // fp16 视图复用 scaled 平面: 列宽按 half 计为 fp32 区的 2 倍 (字节等宽, 行跨步一致)
    constexpr int paddedCol_f16 = paddedCol_src * 2;
    using F16TileDefine = pto::Tile<pto::TileType::Vec, half, srcTileH, paddedCol_f16, pto::BLayout::RowMajor, -1, -1>;

    // scale/offset tile 按量化轴选择: -1 行广播 ColMajor 列向量 / -2 列广播 RowMajor 行向量。
    // offset tile 与 scale 同样按 ScaleDtype 实例化 (行/列广播指令要求操作数与 dst 同 dtype;
    // 与旧 TQUANT 路径的 ParaTileDefine 行为保持一致, 地址仍按 OffsetDtype 元素大小计算)。
    using ScaleRowTileDefine = pto::Tile<pto::TileType::Vec, ScaleDtype, paddedRow_scale, 1, pto::BLayout::ColMajor, -1,
                                         -1>;
    using ScaleColTileDefine = pto::Tile<pto::TileType::Vec, ScaleDtype, 1, paddedRow_scale, pto::BLayout::RowMajor, -1,
                                         -1>;
    using OffsetRowTileDefine = pto::Tile<pto::TileType::Vec, ScaleDtype, paddedRow_scale, 1, pto::BLayout::ColMajor,
                                          -1, -1>;
    using OffsetColTileDefine = pto::Tile<pto::TileType::Vec, ScaleDtype, 1, paddedRow_scale, pto::BLayout::RowMajor,
                                          -1, -1>;

#if !defined(PTO_NPU_ARCH_A5) && !defined(PTO_NPU_ARCH_A6)
    // A2/A3 行广播 vbrcb 缓冲 (仅 axis=-1 使用), 实际写入 ceil(H/8)*256B
    constexpr int bcastRows = PTO_CEIL(srcTileH < 256 ? srcTileH : 256, 8);
    using BcastTileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, bcastRows, 64, pto::BLayout::RowMajor, -1, -1>;
#endif

    // tmp 分段 (元素数, 与 host 侧公式逐字一致): [scaled/f16][s32] (A2/A3 另有 [bcast])
    constexpr int planeElems = PTO_CEIL(srcTileH, 8) * paddedCol_src;
    constexpr int scaledOff = 0;
    constexpr int s32Off = planeElems;
#if !defined(PTO_NPU_ARCH_A5) && !defined(PTO_NPU_ARCH_A6)
    constexpr int bcastOff = 2 * planeElems;
#endif

    // 遍历所有 Tile
    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                DstTileDefine dstTile(dstShape3, dstShape4);
                SrcTileDefine srcTile(srcShape3, srcShape4);
                ScaledTileDefine scaledTile(srcShape3, srcShape4);
                S32TileDefine s32Tile(srcShape3, srcShape4);
                F16TileDefine f16Tile(srcShape3, srcShape4);

                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                auto scaleOffset = n0Index * scaleStride1 + n1Index * scaleStride2 + n2Index * scaleStride3;
                auto offsetOffset = n0Index * offsetStride1 + n1Index * offsetStride2 + n2Index * offsetStride3;

                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * sizeof(DstDtype)));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * sizeof(SrcDtype)));

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6)
                (void)tmp;
                if constexpr (quantAxis == -2) {
                    // A5/A6 列广播: TCOLEXPANDMUL/ADD + 寄存器级 TCVT 链。
                    // A5/A6 的 tile 级 TCVT 不支持 int32->half, 需与原生链一致: fp32 -> s32 -> fp32 -> f16 -> uint8。
                    // 平面复用: scaled 平面先存 scaled fp32, s32 转换后再接回 s32->fp32; f16 复用 s32 平面。
                    ScaleColTileDefine scaleTile(1, srcShape4);
                    OffsetColTileDefine offsetTile(1, srcShape4);
                    pto::TASSIGN(scaleTile, (uint64_t)(scale.GetAddr() + scaleOffset * sizeof(ScaleDtype)));
                    pto::TASSIGN(offsetTile, (uint64_t)(offset.GetAddr() + offsetOffset * sizeof(OffsetDtype)));
                    pto::TASSIGN(scaledTile, (uint64_t)(tmp.GetAddr()) + scaledOff * sizeof(int32_t));
                    pto::TASSIGN(s32Tile, (uint64_t)(tmp.GetAddr()) + s32Off * sizeof(int32_t));
                    pto::TASSIGN(f16Tile, (uint64_t)(tmp.GetAddr()) + s32Off * sizeof(int32_t));
                    SyncV();
                    pto::TCOLEXPANDMUL(scaledTile, srcTile, scaleTile);
                    SyncV();
                    pto::TCOLEXPANDADD(scaledTile, scaledTile, offsetTile);
                    SyncV();
                    pto::TCVT(s32Tile, scaledTile, pto::RoundMode::CAST_RINT); // fp32 -> s32
                    SyncV();
                    pto::TCVT(scaledTile, s32Tile, pto::RoundMode::CAST_RINT); // s32 -> fp32
                    SyncV();
                    pto::TCVT(f16Tile, scaledTile, pto::RoundMode::CAST_RINT); // fp32 -> f16
                    SyncV();
                    pto::TCVT(dstTile, f16Tile, pto::RoundMode::CAST_RINT, pto::SaturationMode::ON);
                } else {
                    // A5/A6 原生 INT8 非对称量化: 行 scale/offset 由 vlds BRC_B32 广播, 不使用 tmp
                    ScaleRowTileDefine scaleTile(srcShape3, 1);
                    OffsetRowTileDefine offsetTile(srcShape3, 1);
                    pto::TASSIGN(scaleTile, (uint64_t)(scale.GetAddr() + scaleOffset * sizeof(ScaleDtype)));
                    pto::TASSIGN(offsetTile, (uint64_t)(offset.GetAddr() + offsetOffset * sizeof(OffsetDtype)));
                    pto::TQUANT<quantType>(dstTile, srcTile, scaleTile, &offsetTile);
                }
#else
                pto::TASSIGN(scaledTile, (uint64_t)(tmp.GetAddr()) + scaledOff * sizeof(int32_t));
                pto::TASSIGN(f16Tile, (uint64_t)(tmp.GetAddr()) + scaledOff * sizeof(int32_t));
                pto::TASSIGN(s32Tile, (uint64_t)(tmp.GetAddr()) + s32Off * sizeof(int32_t));

                if constexpr (quantAxis == -2) {
                    // 列广播: scale[c]/offset[c] 沿 H 方向广播
                    ScaleColTileDefine scaleTile(1, srcShape4);
                    OffsetColTileDefine offsetTile(1, srcShape4);
                    pto::TASSIGN(scaleTile, (uint64_t)(scale.GetAddr() + scaleOffset * sizeof(ScaleDtype)));
                    pto::TASSIGN(offsetTile, (uint64_t)(offset.GetAddr() + offsetOffset * sizeof(OffsetDtype)));
                    pipe_barrier(PIPE_V);
                    pto::TCOLEXPANDMUL(scaledTile, srcTile, scaleTile);
                    pipe_barrier(PIPE_V);
                    pto::TCOLEXPANDADD(scaledTile, scaledTile, offsetTile);
                    pipe_barrier(PIPE_V);
                } else {
                    // 行广播: scale[r]/offset[r] 沿 W 方向广播
                    ScaleRowTileDefine scaleTile(srcShape3, 1);
                    OffsetRowTileDefine offsetTile(srcShape3, 1);
                    BcastTileDefine bcastTile;
                    pto::TASSIGN(scaleTile, (uint64_t)(scale.GetAddr() + scaleOffset * sizeof(ScaleDtype)));
                    pto::TASSIGN(offsetTile, (uint64_t)(offset.GetAddr() + offsetOffset * sizeof(OffsetDtype)));
                    pto::TASSIGN(bcastTile, (uint64_t)(tmp.GetAddr()) + bcastOff * sizeof(int32_t));
                    pipe_barrier(PIPE_V);
                    pto::TROWEXPANDMUL(scaledTile, srcTile, scaleTile, bcastTile);
                    pipe_barrier(PIPE_V);
                    pto::TROWEXPANDADD(scaledTile, scaledTile, offsetTile, bcastTile);
                    pipe_barrier(PIPE_V);
                }

                // fp32 -> s32 -> fp16 -> uint8: 与 golden 的 round+饱和语义一致
                pto::TCVT(s32Tile, scaledTile, pto::RoundMode::CAST_RINT);
                pipe_barrier(PIPE_V);
                QuantizeS32ToF16(f16Tile, s32Tile);
                pipe_barrier(PIPE_V);
                pto::TCVT(dstTile, f16Tile, pto::RoundMode::CAST_RINT, pto::SaturationMode::ON);
                pipe_barrier(PIPE_V);
#endif
            }
        }
    }
}

// =============================================================================
// 统一量化接口
// =============================================================================
#define OP_TILE_OP_TQUANT TQuant

/**
 * @brief 统一量化接口
 * @tparam quantType INT8_SYM 或 INT8_ASYM
 */
template <pto::QuantType quantType, int quantAxis, typename T0, typename T1, typename T2, typename T3>
TILEOP void TQuant(T0 dst, T1 src, T2 scale, T3 tmp)
{
    static_assert(quantType == pto::QuantType::INT8_SYM,
                  "TQuant with 4 parameters only supports INT8_SYM. "
                  "TQuant only supports INT8_SYM(4 parameters) and INT8_ASYM(5 parameters).");
    TQuantInt8Sym<quantType, quantAxis>(dst, src, scale, tmp);
}

template <pto::QuantType quantType, int quantAxis, typename T0, typename T1, typename T2, typename T3, typename T4>
TILEOP void TQuant(T0 dst, T1 src, T2 scale, T3 offset, T4 tmp)
{
    static_assert(quantType == pto::QuantType::INT8_ASYM,
                  "TQuant with 5 parameters only supports INT8_ASYM."
                  "TQuant only supports INT8_SYM(4 parameters) and INT8_ASYM(5 parameters).");
    TQuantInt8Asym<quantType, quantAxis>(dst, src, scale, offset, tmp);
}

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6)
// =============================================================================
// MX 量化
// =============================================================================
#define OP_TILE_OP_QUANT_MX TQuantMX
constexpr int kDequantScaleRoundingModeRoundUp = 0;
constexpr int kDequantScaleRoundingModeRoundDown = 1;
constexpr int kQuantMXPerformanceModeOn = 1;

template <int DEQUANT_SCALE_ROUNDING_MODE, typename DstTile, typename SrcTile, typename ExpTile, typename MaxTile,
          typename ScalingTile>
__aicore__ inline void QuantMXDispatch(DstTile& dstTile, SrcTile& srcTile, ExpTile& expTile, MaxTile& maxTile,
                                       ScalingTile& scalingTile)
{
    if constexpr (std::is_same_v<typename DstTile::DType, float4_e2m1x2_t>) {
        if constexpr (DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundUp) {
            pto::TQUANT<pto::QuantType::MXFP4_E2M1, DstTile, SrcTile, ExpTile, MaxTile, ScalingTile,
                        pto::QuantScaleAlg::NV>(dstTile, srcTile, &expTile, &maxTile, &scalingTile);
        } else {
            pto::TQUANT<pto::QuantType::MXFP4_E2M1, DstTile, SrcTile, ExpTile, MaxTile, ScalingTile,
                        pto::QuantScaleAlg::OCP>(dstTile, srcTile, &expTile, &maxTile, &scalingTile);
        }
    } else {
        if constexpr (DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundUp) {
            pto::TQUANT<pto::QuantType::MXFP8, DstTile, SrcTile, ExpTile, MaxTile, ScalingTile, pto::QuantScaleAlg::NV>(
                dstTile, srcTile, &expTile, &maxTile, &scalingTile);
        } else {
            pto::TQUANT<pto::QuantType::MXFP8, DstTile, SrcTile, ExpTile, MaxTile, ScalingTile,
                        pto::QuantScaleAlg::OCP>(dstTile, srcTile, &expTile, &maxTile, &scalingTile);
        }
    }
}

template <int DEQUANT_SCALE_ROUNDING_MODE, typename DstTile, typename SrcTile, typename ExpTile, typename MaxTile,
          typename ScalingTile>
__aicore__ inline void QuantMXDispatchDn(DstTile& dstTile, SrcTile& srcTile, ExpTile& expTile, MaxTile& maxTile,
                                         ScalingTile& scalingTile)
{
    if constexpr (std::is_same_v<typename DstTile::DType, float4_e2m1x2_t>) {
        if constexpr (DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundUp) {
            pto::TQUANT<0, pto::MxQuantAlg::NvMxFp4E2M1, true>(dstTile, srcTile, &expTile, &maxTile, &scalingTile);
        } else {
            pto::TQUANT<0, pto::MxQuantAlg::OcpMxFp4E2M1, true>(dstTile, srcTile, &expTile, &maxTile, &scalingTile);
        }
    } else {
        if constexpr (DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundUp) {
            pto::TQUANT<0, pto::MxQuantAlg::NvMxFp8E4M3, true>(dstTile, srcTile, &expTile, &maxTile, &scalingTile);
        } else {
            pto::TQUANT<0, pto::MxQuantAlg::OcpMxFp8E4M3, true>(dstTile, srcTile, &expTile, &maxTile, &scalingTile);
        }
    }
}

template <typename T, typename Layout>
__aicore__ inline size_t GetQuantMXPerformanceGroupedOffset(const Layout& layout, LoopVar n0Index, LoopVar n1Index,
                                                            LoopVar n2Index)
{
    (void)n0Index;
    constexpr auto srcRank = Std::tuple_size<typename T::Shape>::value;
    static_assert(srcRank >= 1 && srcRank <= 4, "TQuantMX only supports 1D to 4D input.");
    if constexpr (srcRank <= 2) {
        return 0;
    } else if constexpr (srcRank == 3) {
        return n2Index * layout.template GetStrideDim<DIM_4TH, MAX_DIMS>();
    } else {
        return n1Index * layout.template GetStrideDim<DIM_3RD, MAX_DIMS>() +
               n2Index * layout.template GetStrideDim<DIM_4TH, MAX_DIMS>();
    }
}

template <int DEQUANT_SCALE_ROUNDING_MODE = kDequantScaleRoundingModeRoundDown, int AXIS = -1, typename T0, typename T1,
          typename T2, typename T3, typename T4>
TILEOP void TQuantMXGeneral(T0 dst, T1 exp, T2 maxScratch, T3 scalingScratch, T4 src)
{
    (void)AXIS;
    const auto dstLayout = dst.GetLayout();
    const auto expLayout = exp.GetLayout();
    const auto maxLayout = maxScratch.GetLayout();
    const auto scalingLayout = scalingScratch.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto expStride0 = expLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto expStride1 = expLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto expStride2 = expLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();

    constexpr auto expTileH = TileOp::GetTensorTileShapeDim<T1, DIM_4TH, MAX_DIMS>();
    constexpr auto expTileW = TileOp::GetTensorTileShapeDim<T1, DIM_5TH, MAX_DIMS>();
    using ExpByteTile = pto::Tile<pto::TileType::Vec, uint8_t, expTileH, expTileW, pto::BLayout::RowMajor, -1, -1>;

    auto dstTile = PtoTile<T0>(dst);
    auto maxTile = PtoTile<T2>(maxScratch);
    auto scalingTile = PtoTile<T3>(scalingScratch);
    auto srcTile = PtoTile<T4>(src);
    ExpByteTile expByteTile(expLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>(),
                            expLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>());

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto expTileOffset = n0Index * expStride0 + n1Index * expStride1 + n2Index * expStride2;
                auto maxTileOffset = GetQuantMXPerformanceGroupedOffset<T4>(maxLayout, n0Index, n1Index, n2Index);
                auto scalingTileOffset = GetQuantMXPerformanceGroupedOffset<T4>(scalingLayout, n0Index, n1Index,
                                                                                n2Index);
                auto srcTileAddr = (uint64_t)(src.GetAddr() +
                                              GenTileOffset(src, tileOffsets) * sizeof(typename T4::Type));
                dstTile.Assign(dst, tileOffsets);
                maxTile.Assign((uint64_t)(maxScratch.GetAddr() + maxTileOffset * sizeof(typename T2::Type)));
                scalingTile.Assign(
                    (uint64_t)(scalingScratch.GetAddr() + scalingTileOffset * sizeof(typename T3::Type)));
                srcTile.Assign(srcTileAddr);
                pto::TASSIGN(expByteTile, (uint64_t)(exp.GetAddr() + expTileOffset * sizeof(typename T1::Type)));
                if constexpr (DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundDown ||
                              DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundUp) {
                    QuantMXDispatch<DEQUANT_SCALE_ROUNDING_MODE>(dstTile.Data(), srcTile.Data(), expByteTile,
                                                                 maxTile.Data(), scalingTile.Data());
                } else {
                    static_assert(DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundDown ||
                                      DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundUp,
                                  "TQuantMX only supports ROUND_DOWN (OCP) and ROUND_UP (NV) modes currently.");
                }
            }
        }
    }
}

template <int DEQUANT_SCALE_ROUNDING_MODE = kDequantScaleRoundingModeRoundDown, int AXIS = -1, typename T0, typename T1,
          typename T2, typename T3, typename T4>
TILEOP void TQuantMXDn(T0 dst, T1 exp, T2 maxScratch, T3 scalingScratch, T4 src)
{
    (void)AXIS;
    const auto dstLayout = dst.GetLayout();
    const auto expLayout = exp.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto expStride0 = expLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto expStride1 = expLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto expStride2 = expLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();

    constexpr auto expTileH = TileOp::GetTensorTileShapeDim<T1, DIM_4TH, MAX_DIMS>();
    constexpr auto expTileW = TileOp::GetTensorTileShapeDim<T1, DIM_5TH, MAX_DIMS>();
    using ExpByteTile = pto::Tile<pto::TileType::Vec, uint8_t, expTileH, expTileW, pto::BLayout::RowMajor, -1, -1>;

    auto dstTile = PtoTile<T0>(dst);
    auto maxTile = PtoTile<T2>(maxScratch);
    auto scalingTile = PtoTile<T3>(scalingScratch);
    // Other UB consumers may pad the plane height. Quantize only complete 64-row groups at the original row pitch.
    constexpr auto srcCols = TileOp::GetTensorTileShapeDim<T4, DIM_5TH, MAX_DIMS>();
    using SrcTile = pto::Tile<pto::TileType::Vec, typename T4::Type, expTileH * 64, srcCols, pto::BLayout::RowMajor, -1,
                              -1>;
    SrcTile srcTile(src.GetLayout().template GetShapeDim<DIM_4TH, MAX_DIMS>(),
                    src.GetLayout().template GetShapeDim<DIM_5TH, MAX_DIMS>());
    ExpByteTile expByteTile(expLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>(),
                            expLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>());

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto expTileOffset = n0Index * expStride0 + n1Index * expStride1 + n2Index * expStride2;
                auto srcTileAddr = (uint64_t)(src.GetAddr() +
                                              GenTileOffset(src, tileOffsets) * sizeof(typename T4::Type));
                dstTile.Assign(dst, tileOffsets);
                maxTile.Assign(maxScratch, tileOffsets);
                scalingTile.Assign(scalingScratch, tileOffsets);
                pto::TASSIGN(srcTile, srcTileAddr);
                pto::TASSIGN(expByteTile, (uint64_t)(exp.GetAddr() + expTileOffset * sizeof(typename T1::Type)));
                if constexpr (DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundDown ||
                              DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundUp) {
                    QuantMXDispatchDn<DEQUANT_SCALE_ROUNDING_MODE>(dstTile.Data(), srcTile, expByteTile, maxTile.Data(),
                                                                   scalingTile.Data());
                } else {
                    static_assert(DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundDown ||
                                      DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundUp,
                                  "TQuantMX only supports ROUND_DOWN (OCP) and ROUND_UP (NV) modes currently.");
                }
            }
        }
    }
}

template <int DEQUANT_SCALE_ROUNDING_MODE = kDequantScaleRoundingModeRoundDown, int AXIS = -1, typename T0, typename T1,
          typename T2, typename T3, typename T4>
TILEOP void TQuantMXPerformance(T0 dst, T1 exp, T2 maxScratch, T3 scalingScratch, T4 src)
{
    (void)AXIS;
    const auto dstLayout = dst.GetLayout();
    const auto maxLayout = maxScratch.GetLayout();
    const auto scalingLayout = scalingScratch.GetLayout();
    const auto srcLayout = src.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto scalingTile = PtoTile<T3>(scalingScratch);
    auto srcTile = PtoTile<T4>(src);
    constexpr int kMxQuantGroupSize = 32;
    constexpr int kExpTileAlignElems = 32;
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T4, DIM_4TH, MAX_DIMS>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T4, DIM_5TH, MAX_DIMS>();
    constexpr auto expFlatTileW = srcTileH * ((srcTileW + kMxQuantGroupSize - 1) / kMxQuantGroupSize);
    constexpr auto expFlatTileWAligned = ((expFlatTileW + kExpTileAlignElems - 1) / kExpTileAlignElems) *
                                         kExpTileAlignElems;
    constexpr auto maxTileH = TileOp::GetTensorTileShapeDim<T2, DIM_4TH, MAX_DIMS>();
    constexpr auto maxTileW = TileOp::GetTensorTileShapeDim<T2, DIM_5TH, MAX_DIMS>();
    using ExpByteTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, expFlatTileWAligned, pto::BLayout::RowMajor, -1, -1>;
    using MaxDtype = std::conditional_t<std::is_same_v<typename T2::Type, bool>, uint8_t, typename T2::Type>;
    using MaxTile = pto::Tile<pto::TileType::Vec, MaxDtype, maxTileH, maxTileW, pto::BLayout::RowMajor, -1, -1>;
    ExpByteTile expByteTile(
        1, srcLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>() *
               ((srcLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>() + kMxQuantGroupSize - 1) / kMxQuantGroupSize));
    MaxTile maxTile(maxLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>(),
                    maxLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>());

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto expTileOffset = GenTileOffset(exp, tileOffsets);
                auto maxTileOffset = GetQuantMXPerformanceGroupedOffset<T4>(maxLayout, n0Index, n1Index, n2Index);
                auto scalingTileOffset = GetQuantMXPerformanceGroupedOffset<T4>(scalingLayout, n0Index, n1Index,
                                                                                n2Index);
                auto srcTileAddr = (uint64_t)(src.GetAddr() +
                                              GenTileOffset(src, tileOffsets) * sizeof(typename T4::Type));
                dstTile.Assign(dst, tileOffsets);
                scalingTile.Assign(
                    (uint64_t)(scalingScratch.GetAddr() + scalingTileOffset * sizeof(typename T3::Type)));
                srcTile.Assign(srcTileAddr);
                pto::TASSIGN(expByteTile, (uint64_t)(exp.GetAddr() + expTileOffset * sizeof(typename T1::Type)));
                pto::TASSIGN(maxTile, (uint64_t)(maxScratch.GetAddr() + maxTileOffset * sizeof(typename T2::Type)));
                if constexpr (DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundDown ||
                              DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundUp) {
                    QuantMXDispatch<DEQUANT_SCALE_ROUNDING_MODE>(dstTile.Data(), srcTile.Data(), expByteTile, maxTile,
                                                                 scalingTile.Data());
                } else {
                    static_assert(DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundDown ||
                                      DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundUp,
                                  "TQuantMX only supports ROUND_DOWN (OCP) and ROUND_UP (NV) modes currently.");
                }
            }
        }
    }
}

template <int DEQUANT_SCALE_ROUNDING_MODE = kDequantScaleRoundingModeRoundDown, int AXIS = -1, int PERFORMANCE_MODE = 0,
          typename T0, typename T1, typename T2, typename T3, typename T4>
TILEOP void TQuantMX(T0 dst, T1 exp, T2 maxScratch, T3 scalingScratch, T4 src)
{
    auto scale = MakeMxScaleTensor<T1>(exp);
    constexpr auto srcRank = Std::tuple_size<typename T4::Shape>::value;
    if constexpr (srcRank >= 2 && AXIS == static_cast<int>(srcRank) - 2) {
        TQuantMXDn<DEQUANT_SCALE_ROUNDING_MODE, AXIS>(dst, scale, maxScratch, scalingScratch, src);
    } else if constexpr (PERFORMANCE_MODE == kQuantMXPerformanceModeOn) {
        constexpr auto rowWidth = TileOp::GetTensorTileShapeDim<decltype(scale), DIM_5TH, MAX_DIMS>();
        const auto groups = (src.GetLayout().template GetShapeDim<DIM_5TH, MAX_DIMS>() + 31) / 32;
        // The flat performance tile requires contiguous scale rows.
        if (rowWidth != groups) {
            TQuantMXGeneral<DEQUANT_SCALE_ROUNDING_MODE, AXIS>(dst, scale, maxScratch, scalingScratch, src);
            return;
        }
        TQuantMXPerformance<DEQUANT_SCALE_ROUNDING_MODE, AXIS>(dst, scale, maxScratch, scalingScratch, src);
    } else {
        TQuantMXGeneral<DEQUANT_SCALE_ROUNDING_MODE, AXIS>(dst, scale, maxScratch, scalingScratch, src);
    }
}

template <int MODE, int AXIS, int PERFORMANCE_MODE, typename T0, typename T1, typename T2, typename T3, typename T4>
TILEOP void TQuantMXPadded(T0 dst, T1 exp, T2 maxScratch, T3 scalingScratch, T4 src)
{
    constexpr auto pairsPerBlock = 32 / sizeof(uint16_t);
    constexpr auto rank = Std::tuple_size<typename T4::Shape>::value;
    constexpr bool dn = rank >= 2 && AXIS == static_cast<int>(rank) - 2;
    constexpr auto srcCols = TileOp::GetTensorTileShapeDim<T4, DIM_5TH, MAX_DIMS>();
    // The ISA scale tile follows the quantization input, independently of the consumer's UB padding.
    constexpr auto rowPairs = dn ? srcCols : (srcCols + 63) / 64;
    constexpr auto tileRows = TileOp::GetTensorTileShapeDim<T1, DIM_3RD, MAX_DIMS>();
    using PackedLayout = LocalLayout5Dim<TileOp::GetTensorTileShapeDim<T1, DIM_1ST, MAX_DIMS>(),
                                         TileOp::GetTensorTileShapeDim<T1, DIM_2ND, MAX_DIMS>(), tileRows,
                                         (rowPairs + pairsPerBlock - 1) / pairsPerBlock * pairsPerBlock, 2>;
    const auto layout = exp.GetLayout();
    auto batches = layout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto planes = layout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto rows = layout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto pairs = layout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    constexpr auto packedPairs = (rowPairs + pairsPerBlock - 1) / pairsPerBlock * pairsPerBlock;
    constexpr auto dstRowBytes = TileOp::GetTensorTileShapeDim<T1, DIM_4TH, MAX_DIMS>() *
                                 TileOp::GetTensorTileShapeDim<T1, DIM_5TH, MAX_DIMS>();
    static_assert(packedPairs * sizeof(uint16_t) <= dstRowBytes);
    TileTensor<typename T1::Type, PackedLayout> packed(exp.GetAddr(), Shape5Dim(batches, planes, rows, pairs, 2));
    TQuantMX<MODE, AXIS, PERFORMANCE_MODE>(dst, packed, maxScratch, scalingScratch, src);

    constexpr unsigned lanes = CCE_VL / sizeof(uint16_t);
    constexpr unsigned dstStride = TileOp::GetTensorStrideDim<T1, DIM_4TH, MAX_DIMS>() / sizeof(uint16_t);
    using SrcRow = pto::Tile<pto::TileType::Vec, uint16_t, 1, lanes, pto::BLayout::RowMajor, -1, -1>;
    using DstCol = pto::Tile<pto::TileType::Vec, uint16_t, lanes, dstStride, pto::BLayout::RowMajor, -1, -1>;
    constexpr auto maxBytes = GetAllAxisTileProduct<T2>() * sizeof(typename T2::Type);
    constexpr auto scalingBytes = GetAllAxisTileProduct<T3>() * sizeof(typename T3::Type);
    // TMOV loads a full vector even for the last partial chunk of a packed row.
    constexpr auto readBytes = (packedPairs + lanes - 1) / lanes * CCE_VL;
    constexpr bool useTranspose = dstRowBytes >= readBytes && (maxBytes >= CCE_VL || scalingBytes >= CCE_VL);
    if constexpr (!useTranspose) {
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    }
    // Expand backwards so a destination row or chunk cannot overwrite unread packed pairs.
    for (auto b = batches; b > 0; --b) {
        for (auto p = planes; p > 0; --p) {
            for (auto r = rows; r > 0; --r) {
                const auto row = TileOffset(b - 1, p - 1, r - 1);
                if constexpr (useTranspose) {
                    const auto scratchAddr = maxBytes >= CCE_VL ? maxScratch.GetAddr() : scalingScratch.GetAddr();
                    for (auto end = pairs; end > 0;) {
                        const auto start = (end - 1) / lanes * lanes;
                        const auto count = end - start;
                        SrcRow packedRow(1, count);
                        SrcRow srcRow(1, count);
                        DstCol dstCol(count, 1);
                        pto::TASSIGN(packedRow, packed.GetLinearAddr(row) + start * sizeof(uint16_t));
                        pto::TASSIGN(srcRow, scratchAddr);
                        pto::TASSIGN(dstCol, exp.GetLinearAddr(row) + start * dstStride * sizeof(uint16_t));
                        pto::TMOV(srcRow, packedRow);
                        pto::TTRANS(dstCol, srcRow, srcRow);
                        end = start;
                    }
                } else {
                    auto* srcPairs = reinterpret_cast<volatile __ubuf__ uint16_t*>(packed.GetLinearAddr(row));
                    auto* dstPairs = reinterpret_cast<volatile __ubuf__ uint16_t*>(exp.GetLinearAddr(row));
                    for (auto i = pairs; i > 0; --i) {
                        dstPairs[(i - 1) * dstStride] = srcPairs[i - 1];
                    }
                }
            }
        }
    }
    if constexpr (!useTranspose) {
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    }
}
#endif // defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6)

#if __NPU_ARCH__ == 9201
#include "quant_hif4_tile.h"
#endif
#endif // TILEOP_TILE_OPERATOR_QUANTIZE__H
