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
 * \file gcd.h
 * \brief Greatest common divisor tile operation implementation.
 */

#ifndef TILEOP_TILE_OPERATOR_VEC_BINARY_GCD_H
#define TILEOP_TILE_OPERATOR_VEC_BINARY_GCD_H

#include "utils/sync.h"
#include "../pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"
#include "../binary_brcinline.h"

#define OP_TILE_OP_GCD TGcd
template <int... BrcOperands, typename T0, typename T1, typename T2, typename T3>
TILEOP void TGcd(T0 dst, T1 src0, T2 src1, T3 temp)
{
    // 计算位宽按输入类型选择：int8/uint8 输入时中间值 <= 510（q*b <= a+b），int16 槽不溢出；
    // int16/int32 输入时 q*b 可达 a+b <= 65534/2e8，需 int32 槽。
    using CalcType = typename std::conditional<Std::is_same_v<typename T0::Type, int8_t> ||
                                                   Std::is_same_v<typename T0::Type, uint8_t>,
                                               int16_t, int32_t>::type;
    constexpr auto tileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto tileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();
    using CalcTile = pto::Tile<pto::TileType::Vec, CalcType, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using TTile = pto::Tile<pto::TileType::Vec, typename T0::Type, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using FTile = pto::Tile<pto::TileType::Vec, float, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using F16Tile = pto::Tile<pto::TileType::Vec, half, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    // TCMPS/TCMP/TSEL 的掩码按位打包存放：每行 ceil(tileW/32)*4 字节，按 32B 对齐
    constexpr auto BITS_PER_BYTE = TileOp::BITS_PER_BYTE;
    constexpr auto MASK_ALIGNMENT = TileOp::BLOCK_SIZE;
    constexpr auto maskCols = ((tileW + BITS_PER_BYTE - 1) / BITS_PER_BYTE + MASK_ALIGNMENT - 1) / MASK_ALIGNMENT *
                              MASK_ALIGNMENT;
    using MaskTile = pto::Tile<pto::TileType::Vec, uint8_t, tileH, maskCols, pto::BLayout::RowMajor, -1, -1>;

    const auto dstLayout = dst.GetLayout();
    const auto src0Layout = src0.GetLayout();
    const auto src1Layout = src1.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto shape3 = dstLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto shape4 = dstLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    if (shape0 == 0 || shape1 == 0 || shape2 == 0) {
        return;
    }
    using Src0TileInfo = TensorTileInfo<T1>;
    using Src1TileInfo = TensorTileInfo<T2>;
    constexpr BrcMode brcmode = GetBrcMode<BrcOperands...>();

    // temp 布局（与 BinaryOperationTileFunc 中 GCD 的临时空间分配保持一致）：
    //   slot0: a = |src0|  slot1: b = |src1|  slot2: r = a % b  slot3: q
    //   slot4: 暂存，顺序复用为 q*b / q修正 / r修正（生命周期互不重叠）
    //   f/fb: 两个 f32 槽（近似商与 refine 除法）
    //   之后依次为：冻结掩码、修正掩码、TSEL tmp（256B+64B 保护带）
    // 计算槽按 CalcType 位宽（int8/uint8 输入为 int16），f32 槽与掩码固定大小
    auto calcStride = shape3 * tileW * sizeof(CalcType);
    auto floatBytes = shape3 * tileW * sizeof(float);
    auto maskBytes = shape3 * maskCols;
    constexpr size_t R_SLOT = 2;
    constexpr size_t Q_SLOT = 3;
    constexpr size_t SCRATCH_SLOT = 4;
    constexpr size_t AUX_SLOT = 5;
    constexpr size_t FLOAT_SLOT = 6;
    auto fBase = FLOAT_SLOT * calcStride;
    auto fbBase = fBase + floatBytes;
    auto maskBase = fbBase + floatBytes;
    auto bNegBase = maskBase + maskBytes;
    auto oppBase = bNegBase + maskBytes;
    auto fixMaskBase = oppBase + maskBytes;
    auto selTmpBase = fixMaskBase + maskBytes;

    CalcTile aTile(shape3, shape4);
    CalcTile bTile(shape3, shape4);
    CalcTile rTile(shape3, shape4);
    CalcTile qTile(shape3, shape4);
    CalcTile scratchTile(shape3, shape4);
    CalcTile auxTile(shape3, shape4);
    FTile fTile(shape3, shape4);
    FTile fbTile(shape3, shape4);
    MaskTile maskTile(shape3, shape4);
    MaskTile bNegTile(shape3, shape4);
    MaskTile oppTile(shape3, shape4);
    MaskTile fixMaskTile(shape3, shape4);
    CalcTile selTmpTile(shape3, shape4);
    TTile src0Tile(shape3, shape4);
    TTile src1Tile(shape3, shape4);
    TTile dstTile(shape3, shape4);

    pto::TASSIGN(aTile, temp.GetAddr());
    pto::TASSIGN(bTile, temp.GetAddr() + calcStride);
    pto::TASSIGN(rTile, temp.GetAddr() + R_SLOT * calcStride);
    pto::TASSIGN(qTile, temp.GetAddr() + Q_SLOT * calcStride);
    pto::TASSIGN(scratchTile, temp.GetAddr() + SCRATCH_SLOT * calcStride);
    pto::TASSIGN(auxTile, temp.GetAddr() + AUX_SLOT * calcStride);
    pto::TASSIGN(fTile, temp.GetAddr() + fBase);
    pto::TASSIGN(fbTile, temp.GetAddr() + fbBase);
    pto::TASSIGN(maskTile, temp.GetAddr() + maskBase);
    pto::TASSIGN(bNegTile, temp.GetAddr() + bNegBase);
    pto::TASSIGN(oppTile, temp.GetAddr() + oppBase);
    pto::TASSIGN(fixMaskTile, temp.GetAddr() + fixMaskBase);
    pto::TASSIGN(selTmpTile, temp.GetAddr() + selTmpBase);

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto dstOffset = GenTileOffset(dst, tileOffsets);
                auto src0tileOffsets = TileOffset(
                    (Src0TileInfo::tile0 == 1 || GetBrcOperandAt<DIM_1ST, BrcOperands...>() == BRC_LEFT) ? 0 : n0Index,
                    (Src0TileInfo::tile1 == 1 || GetBrcOperandAt<DIM_2ND, BrcOperands...>() == BRC_LEFT) ? 0 : n1Index,
                    (Src0TileInfo::tile2 == 1 || GetBrcOperandAt<DIM_3RD, BrcOperands...>() == BRC_LEFT) ? 0 : n2Index);
                auto src1tileOffsets = TileOffset(
                    (Src1TileInfo::tile0 == 1 || GetBrcOperandAt<DIM_1ST, BrcOperands...>() == BRC_RIGHT) ? 0 : n0Index,
                    (Src1TileInfo::tile1 == 1 || GetBrcOperandAt<DIM_2ND, BrcOperands...>() == BRC_RIGHT) ? 0 : n1Index,
                    (Src1TileInfo::tile2 == 1 || GetBrcOperandAt<DIM_3RD, BrcOperands...>() == BRC_RIGHT) ? 0 :
                                                                                                            n2Index);
                auto src0Offset = GenTileOffset(src0, src0tileOffsets);
                auto src1Offset = GenTileOffset(src1, src1tileOffsets);
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * sizeof(typename T0::Type)));
                pto::TASSIGN(src0Tile, (uint64_t)(src0.GetAddr() + src0Offset * sizeof(typename T1::Type)));
                pto::TASSIGN(src1Tile, (uint64_t)(src1.GetAddr() + src1Offset * sizeof(typename T2::Type)));

                // 统一转换到计算位宽。a2/a3 的 TCVT 没有 int8/uint8/int16 到 int32 的
                // 直接转换，int16 经 f32、int8/uint8 经 f16 中转（小整数转换无精度损失）
                if constexpr (Std::is_same_v<typename T0::Type, int32_t>) {
                    pto::TMOV(aTile, src0Tile);
                    pto::TMOV(bTile, src1Tile);
                } else if constexpr (Std::is_same_v<typename T0::Type, int16_t>) {
                    pto::TCVT(fTile, src0Tile, pto::RoundMode::CAST_NONE);
                    pto::TCVT(fbTile, src1Tile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TCVT(aTile, fTile, pto::RoundMode::CAST_RINT);
                    pto::TCVT(bTile, fbTile, pto::RoundMode::CAST_RINT);
                } else {
                    // int8/uint8: s8/u8 -> f16 -> s16（f16 中转视图复用 fbTile 的地址）
                    F16Tile f16Tile(shape3, shape4);
                    pto::TASSIGN(f16Tile, temp.GetAddr() + fbBase);
                    pto::TCVT(f16Tile, src0Tile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TCVT(aTile, f16Tile, pto::RoundMode::CAST_RINT);
                    SyncV();
                    pto::TCVT(f16Tile, src1Tile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TCVT(bTile, f16Tile, pto::RoundMode::CAST_RINT);
                }
                SyncV();

                // |x| = x * clamp(x, -1, 1)
                pto::TMINS(rTile, aTile, 1);
                SyncV();
                pto::TMAXS(rTile, rTile, -1);
                SyncV();
                pto::TMUL(aTile, aTile, rTile);
                SyncV();
                pto::TMINS(rTile, bTile, 1);
                SyncV();
                pto::TMAXS(rTile, rTile, -1);
                SyncV();
                pto::TMUL(bTile, bTile, rTile);
                SyncV();
                if constexpr (Std::is_same_v<typename T0::Type, int16_t>) {
                    // int16 回绕: -32768 的绝对值 32768 回绕为 -32768（与 torch.gcd 一致）
                    pto::TCVT(fTile, aTile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    constexpr float INT16_ABS_LIMIT = 32768.0f;
                    constexpr int32_t UINT16_RANGE = 65536;
                    pto::TCMPS(maskTile, fTile, INT16_ABS_LIMIT, pto::CmpMode::GE);
                    SyncV();
                    pto::TADDS(scratchTile, aTile, -UINT16_RANGE);
                    SyncV();
                    pto::TSEL(aTile, maskTile, scratchTile, aTile, selTmpTile);
                    SyncV();
                    pto::TCVT(fTile, bTile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TCMPS(maskTile, fTile, INT16_ABS_LIMIT, pto::CmpMode::GE);
                    SyncV();
                    pto::TADDS(scratchTile, bTile, -UINT16_RANGE);
                    SyncV();
                    pto::TSEL(bTile, maskTile, scratchTile, bTile, selTmpTile);
                    SyncV();
                } else if constexpr (Std::is_same_v<typename T0::Type, int8_t>) {
                    // int8 回绕: -128 的绝对值 128 回绕为 -128（与 torch.gcd 一致）
                    pto::TCVT(fTile, aTile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    constexpr float INT8_ABS_LIMIT = 128.0f;
                    constexpr int16_t UINT8_RANGE = 256;
                    pto::TCMPS(maskTile, fTile, INT8_ABS_LIMIT, pto::CmpMode::GE);
                    SyncV();
                    pto::TADDS(scratchTile, aTile, -UINT8_RANGE);
                    SyncV();
                    pto::TSEL(aTile, maskTile, scratchTile, aTile, selTmpTile);
                    SyncV();
                    pto::TCVT(fTile, bTile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TCMPS(maskTile, fTile, INT8_ABS_LIMIT, pto::CmpMode::GE);
                    SyncV();
                    pto::TADDS(scratchTile, bTile, -UINT8_RANGE);
                    SyncV();
                    pto::TSEL(bTile, maskTile, scratchTile, bTile, selTmpTile);
                    SyncV();
                }

                // Euclid 迭代：固定迭代次数，避免仿真器上 while + GetValue 的标量同步竞态（卡死）。
                // b == 0 的 lane 用掩码冻结，收敛后保持 (gcd, 0) 不变；除零时 vdiv 结果
                // 为 inf/0，商饱和后与 0 相乘仍得 0，余数保持 a，冻结逻辑不受影响。
                // 最大迭代次数：int32 <= 1e8 时 Euclid 步数 <= 40（Fibonacci 界），64 封顶；
                // int16 <= 32767 时 <= 23，32 封顶；int8/uint8 <= 255 时 <= 13，16 封顶。
                constexpr int INT32_MAX_ITER = 64;
                constexpr int INT16_MAX_ITER = 32;
                constexpr int INT8_MAX_ITER = 16;
                constexpr int kMaxIter = Std::is_same_v<typename T0::Type, int32_t> ?
                                             INT32_MAX_ITER :
                                             (Std::is_same_v<typename T0::Type, int16_t> ? INT16_MAX_ITER :
                                                                                           INT8_MAX_ITER);
                for (LoopVar k = 0; k < kMaxIter; ++k) {
                    // 冻结 b == 0 的 lane，避免除零
                    pto::TCMPS(maskTile, bTile, 0, pto::CmpMode::EQ);
                    SyncV();
                    // 符号归一化副本（a/b 原值不动，链更新用原始符号值）：
                    // aDiv = bNeg ? -a : a（aux）；bDiv = bNeg ? -b : b（rTile，bDiv > 0）
                    pto::TCVT(fbTile, bTile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TCMPS(bNegTile, fbTile, 0.0f, pto::CmpMode::LT);
                    SyncV();
                    pto::TMULS(auxTile, aTile, -1);
                    SyncV();
                    pto::TSEL(auxTile, bNegTile, auxTile, aTile, selTmpTile);
                    SyncV();
                    pto::TMULS(rTile, bTile, -1);
                    SyncV();
                    pto::TSEL(rTile, bNegTile, rTile, bTile, selTmpTile);
                    SyncV();
                    // floored 精确余数（bDiv > 0，aDiv 任意符号，结果 r_f = aux ∈ [0, bDiv)）
                    pto::TCVT(fbTile, rTile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TCVT(fTile, auxTile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(fTile, fTile, fbTile);
                    SyncV();
                    pto::TCVT(qTile, fTile, pto::RoundMode::CAST_FLOOR);
                    SyncV();
                    pto::TMUL(scratchTile, qTile, rTile);
                    SyncV();
                    pto::TSUB(auxTile, auxTile, scratchTile); // aux = r = aDiv - q*bDiv
                    SyncV();
                    pto::TCVT(fTile, auxTile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TDIV<pto::DivAlgorithm::HIGH_PRECISION>(fTile, fTile, fbTile);
                    SyncV();
                    pto::TCVT(scratchTile, fTile, pto::RoundMode::CAST_FLOOR); // scratch = qc
                    SyncV();
                    // r_new = r_old - qc * bDiv（aux 已是 r_old，不能从 aDiv 重算）
                    pto::TMUL(scratchTile, scratchTile, rTile);
                    SyncV();
                    pto::TSUB(auxTile, auxTile, scratchTile);
                    SyncV();
                    // floored +/-1 修正（f32 符号桥接，bDiv > 0）
                    pto::TSUB(scratchTile, auxTile, rTile);
                    SyncV();
                    pto::TCVT(fTile, scratchTile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TCMPS(fixMaskTile, fTile, 0.0f, pto::CmpMode::GE);
                    SyncV();
                    pto::TSEL(auxTile, fixMaskTile, scratchTile, auxTile, selTmpTile);
                    SyncV();
                    pto::TADD(scratchTile, auxTile, rTile);
                    SyncV();
                    pto::TCVT(fTile, auxTile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TCMPS(fixMaskTile, fTile, 0.0f, pto::CmpMode::LT);
                    SyncV();
                    pto::TSEL(auxTile, fixMaskTile, scratchTile, auxTile, selTmpTile);
                    SyncV();
                    // r_f == 0 标记（守卫用）
                    pto::TCVT(fTile, auxTile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TCMPS(fixMaskTile, fTile, 0.0f, pto::CmpMode::EQ);
                    SyncV();
                    // 转换为 C 截断余数（用原始 a/b 的符号）：
                    //   opp = (f32(a) * f32(b) < 0)（原始异号，乘积符号精确）
                    //   r_trunc = opp ? (bDiv - r_f) : (bNeg ? -r_f : r_f)
                    pto::TCVT(fTile, aTile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TCVT(fbTile, bTile, pto::RoundMode::CAST_NONE);
                    SyncV();
                    pto::TMUL(fTile, fTile, fbTile);
                    SyncV();
                    pto::TCMPS(oppTile, fTile, 0.0f, pto::CmpMode::LT);
                    SyncV();
                    pto::TMULS(qTile, auxTile, -1); // qTile = -r_f（q 已死）
                    SyncV();
                    pto::TSUB(scratchTile, rTile, auxTile); // bDiv - r_f（bNeg 分支）
                    SyncV();
                    pto::TSUB(rTile, auxTile, rTile); // r_f - bDiv（非 bNeg 分支, bDiv 已死）
                    SyncV();
                    pto::TSEL(scratchTile, bNegTile, scratchTile, rTile, selTmpTile); // rOpp
                    SyncV();
                    pto::TSEL(qTile, bNegTile, qTile, auxTile, selTmpTile); // rBase
                    SyncV();
                    pto::TSEL(auxTile, oppTile, scratchTile, qTile, selTmpTile); // r = opp ? rOpp : rBase
                    SyncV();
                    // r_f == 0 时保持 0（±b 毛刺会翻转最终符号; 零 tile 用自减生成）
                    pto::TSUB(scratchTile, auxTile, auxTile); // 0
                    SyncV();
                    pto::TSEL(scratchTile, fixMaskTile, scratchTile, auxTile, selTmpTile);
                    SyncV();
                    pto::TMOV(auxTile, scratchTile);
                    SyncV();
                    // 冻结 lane：a, b 保持不变（原始符号值），其余 a = b, b = a % b
                    pto::TSEL(aTile, maskTile, aTile, bTile, selTmpTile);
                    SyncV();
                    pto::TSEL(bTile, maskTile, bTile, auxTile, selTmpTile);
                    SyncV();
                } // 结果取非负 gcd（与 torch.gcd 语义一致）
                if constexpr (Std::is_same_v<typename T0::Type, int32_t>) {
                    pto::TMOV(dstTile, aTile);
                } else if constexpr (Std::is_same_v<typename T0::Type, int16_t>) {
                    pto::TCVT(dstTile, aTile, pto::RoundMode::CAST_RINT);
                } else {
                    // int8/uint8: s16 -> f32 -> f16 -> s8/u8（a2/a3 无直接转换）
                    F16Tile f16Tile(shape3, shape4);
                    pto::TASSIGN(f16Tile, temp.GetAddr() + fbBase);
                    pto::TCVT(fTile, aTile, pto::RoundMode::CAST_RINT);
                    SyncV();
                    pto::TCVT(f16Tile, fTile, pto::RoundMode::CAST_RINT);
                    SyncV();
                    pto::TCVT(dstTile, f16Tile, pto::RoundMode::CAST_RINT);
                }
                SyncV();
            }
        }
    }
}

#endif // TILEOP_TILE_OPERATOR_VEC_BINARY_GCD_H
