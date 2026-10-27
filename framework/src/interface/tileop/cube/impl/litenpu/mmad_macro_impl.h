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
 * \file mmad_macro_impl.h
 * \brief kirin9050 matmul 宏指令 tileop 实现（pre-stage quant/dequant、bias、GEMV、多 K 片 int32 累加）
 *
 * 本文件只在 __DAV_L510__ 下由 impl/mmad_impl.h（共用入口）包含。宏指令 tileop 与非宏指令的
 * TMatmul/TGEMV 解耦：codegen 侧（CodeGenOpLiteNPU）按 opcode/量化标志选择对应 tileop 名，
 * 本文件不做架构宏区分——这些模板仅在 kirin9050 codegen 显式调用/实例化。
 */

#ifndef TILEOP_TILE_OPERATOR_KIRIN9050_MMAD_MACRO_IMPL__H
#define TILEOP_TILE_OPERATOR_KIRIN9050_MMAD_MACRO_IMPL__H

#include "../cube_utils.h"

// kirin9050 宏指令多 K 片：K 中间片的 partial sum 必须是 int32 累加器（指令 biasAddr/acc 契约为
// int32；末片再以 out tile 形式读 int32 acc -> out dtype）。对齐 pto-isa 参考实现
// tests/npu/kirin9050/src/st/testcase/tmatmul_new_full/RunTMATMUL_SPLIT_K。
template <bool initMatrixC, bool isGemv = false, typename TileAcc, typename TileLeft, typename TileRight>
INLINE void TMatmulMacroAccImpl(TileAcc& c, TileLeft& a, TileRight& b)
{
    int64_t validM = GetShape<0>(a);
    int64_t validN = GetShape<1>(b);
    int64_t validK = GetShape<1>(a);
    if (validM == 0 || validN == 0 || (validK == 0 && !initMatrixC)) {
        return;
    }
    constexpr uint64_t shapeSizeA = Std::tuple_size<typename TileLeft::Shape>::value;
    constexpr uint64_t shapeSizeB = Std::tuple_size<typename TileRight::Shape>::value;
    constexpr uint64_t shapeSizeC = Std::tuple_size<typename TileAcc::Shape>::value;
    constexpr auto staticL0AH = Std::tuple_element<shapeSizeA - SHAPE_DIM2, typename TileLeft::TileShape>::type::value;
    constexpr auto staticL0AW = Std::tuple_element<shapeSizeA - 1, typename TileLeft::TileShape>::type::value;
    constexpr auto staticL0BH = Std::tuple_element<shapeSizeB - SHAPE_DIM2, typename TileRight::TileShape>::type::value;
    constexpr auto staticL0BW = Std::tuple_element<shapeSizeB - 1, typename TileRight::TileShape>::type::value;
    constexpr auto staticL0CH = Std::tuple_element<shapeSizeC - SHAPE_DIM2, typename TileAcc::TileShape>::type::value;
    constexpr auto staticL0CW = Std::tuple_element<shapeSizeC - 1, typename TileAcc::TileShape>::type::value;
    if (validK == 0 && initMatrixC) {
        validK = staticL0AW;
    }
    using tileL0ATensor = pto::TileLeft<typename TileLeft::Type, staticL0AH, staticL0AW, -1, -1>;
    using tileL0BTensor = pto::TileRight<typename TileRight::Type, staticL0BH, staticL0BW, -1, -1>;
    using tileL0CTensor = pto::TileAcc<typename TileAcc::Type, staticL0CH, staticL0CW, -1, -1>;
    if constexpr (isGemv) {
        // GEMV M=1，不打 pad 到 16。但宏 acc 的累加（bias=C）只按 m=16 形态覆盖整块 C：
        // int8 GEMV 若 m=1，N>16 的第 2 个 N fractal 不参与累加（只保留首片值）。
        // 对齐作者 RunTMATMUL_SPLIT_K 的 GEMV 用法（AccTile M=16 + gemvCtrl）→ int8 用 m=16。
        validM = std::is_same<typename TileLeft::Type, int8_t>::value ? BLOCK_CUBE_M_N : 1;
    } else {
        validM = (validM + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    }
    tileL0ATensor l0a(validM, validK);
    tileL0BTensor l0b(validK, validN);
    tileL0CTensor l0c(validM, validN);
    pto::TASSIGN(l0a, static_cast<uint64_t>(a.GetAddr()));
    pto::TASSIGN(l0b, static_cast<uint64_t>(b.GetAddr()));
    pto::TASSIGN(l0c, static_cast<uint64_t>(c.GetAddr()));
    pto::MatmulMacroConfig cfg;
    cfg.gemvCtrl = isGemv;
    if constexpr (initMatrixC) {
        pto::TMATMUL_MACRO_ACC_IMPL<pto::AccPhase::Unspecified, tileL0CTensor, tileL0ATensor, tileL0BTensor, void,
                                    true>(l0c, l0a, l0b, nullptr, cfg);
    } else {
        pto::TMATMUL_MACRO_ACC_IMPL<pto::AccPhase::Unspecified, tileL0CTensor, tileL0ATensor, tileL0BTensor, void,
                                    false>(l0c, l0a, l0b, nullptr, cfg);
    }
}

// 首片带 bias 的 int32 partial sum（bias 经 BT 下发）：C(int32 acc) = A*B + bias
template <bool isGemv = false, typename TileAcc, typename TileLeft, typename TileRight, typename TileBias>
INLINE void TMatmulMacroAccBiasImpl(TileAcc& c, TileLeft& a, TileRight& b, TileBias& bias)
{
    int64_t validM = GetShape<0>(a);
    int64_t validN = GetShape<1>(b);
    int64_t validK = GetShape<1>(a);
    if (validM == 0 || validN == 0 || validK == 0) {
        return;
    }
    constexpr uint64_t shapeSizeA = Std::tuple_size<typename TileLeft::Shape>::value;
    constexpr uint64_t shapeSizeB = Std::tuple_size<typename TileRight::Shape>::value;
    constexpr uint64_t shapeSizeC = Std::tuple_size<typename TileAcc::Shape>::value;
    constexpr auto staticL0AH = Std::tuple_element<shapeSizeA - SHAPE_DIM2, typename TileLeft::TileShape>::type::value;
    constexpr auto staticL0AW = Std::tuple_element<shapeSizeA - 1, typename TileLeft::TileShape>::type::value;
    constexpr auto staticL0BH = Std::tuple_element<shapeSizeB - SHAPE_DIM2, typename TileRight::TileShape>::type::value;
    constexpr auto staticL0BW = Std::tuple_element<shapeSizeB - 1, typename TileRight::TileShape>::type::value;
    constexpr auto staticL0CH = Std::tuple_element<shapeSizeC - SHAPE_DIM2, typename TileAcc::TileShape>::type::value;
    constexpr auto staticL0CW = Std::tuple_element<shapeSizeC - 1, typename TileAcc::TileShape>::type::value;
    using tileL0ATensor = pto::TileLeft<typename TileLeft::Type, staticL0AH, staticL0AW, -1, -1>;
    using tileL0BTensor = pto::TileRight<typename TileRight::Type, staticL0BH, staticL0BW, -1, -1>;
    using tileL0CTensor = pto::TileAcc<typename TileAcc::Type, staticL0CH, staticL0CW, -1, -1>;
    using tileBiasTensor = pto::Tile<pto::TileType::Bias, typename TileBias::Type, 1, staticL0BW,
                                     pto::BLayout::RowMajor, -1, -1>;
    if constexpr (isGemv) {
        validM = 1; // GEMV M=1，不打 pad 到 16
    } else {
        validM = (validM + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    }
    tileL0ATensor l0a(validM, validK);
    tileL0BTensor l0b(validK, validN);
    tileL0CTensor l0c(validM, validN);
    tileBiasTensor biasT(1, validN);
    pto::TASSIGN(l0a, static_cast<uint64_t>(a.GetAddr()));
    pto::TASSIGN(l0b, static_cast<uint64_t>(b.GetAddr()));
    pto::TASSIGN(l0c, static_cast<uint64_t>(c.GetAddr()));
    pto::TASSIGN(biasT, static_cast<uint64_t>(bias.GetAddr()));
    pto::MatmulMacroConfig cfg;
    cfg.gemvCtrl = isGemv;
    pto::TMATMUL_MACRO_ACC_IMPL<pto::AccPhase::Unspecified, tileL0CTensor, tileL0ATensor, tileL0BTensor,
                                tileBiasTensor>(l0c, l0a, l0b, &biasT, cfg);
}

// 末片：读 int32 partial sum 累加器 -> 转 out dtype 写出（acc 与输出分离）
// withQuantPre=true 时反量化走向量档：quant_pre 为 FIXBUF 中的 per-channel scale 向量，
// 地址经 MatmulMacroConfig.preQuantTileAddr 写入 m_fpc（与单 K 片 TMatmulQuant 的向量档一致）。
template <bool isGemv = false, bool withQuantPre = false, typename TileAcc, typename TileLeft, typename TileRight,
          typename TileAccIn>
INLINE void TMatmulMacroAccFromImpl(TileAcc& c, TileLeft& a, TileRight& b, TileAccIn& cIn, uint64_t quantPreAddr = 0)
{
    int64_t validM = GetShape<0>(a);
    int64_t validN = GetShape<1>(b);
    int64_t validK = GetShape<1>(a);
    if (validM == 0 || validN == 0 || validK == 0) {
        return;
    }
    constexpr uint64_t shapeSizeA = Std::tuple_size<typename TileLeft::Shape>::value;
    constexpr uint64_t shapeSizeB = Std::tuple_size<typename TileRight::Shape>::value;
    constexpr uint64_t shapeSizeC = Std::tuple_size<typename TileAcc::Shape>::value;
    constexpr uint64_t shapeSizeCin = Std::tuple_size<typename TileAccIn::Shape>::value;
    constexpr auto staticL0AH = Std::tuple_element<shapeSizeA - SHAPE_DIM2, typename TileLeft::TileShape>::type::value;
    constexpr auto staticL0AW = Std::tuple_element<shapeSizeA - 1, typename TileLeft::TileShape>::type::value;
    constexpr auto staticL0BH = Std::tuple_element<shapeSizeB - SHAPE_DIM2, typename TileRight::TileShape>::type::value;
    constexpr auto staticL0BW = Std::tuple_element<shapeSizeB - 1, typename TileRight::TileShape>::type::value;
    constexpr auto staticL0CH = Std::tuple_element<shapeSizeC - SHAPE_DIM2, typename TileAcc::TileShape>::type::value;
    constexpr auto staticL0CW = Std::tuple_element<shapeSizeC - 1, typename TileAcc::TileShape>::type::value;
    constexpr auto
        staticInH = Std::tuple_element<shapeSizeCin - SHAPE_DIM2, typename TileAccIn::TileShape>::type::value;
    constexpr auto staticInW = Std::tuple_element<shapeSizeCin - 1, typename TileAccIn::TileShape>::type::value;
    using tileL0ATensor = pto::TileLeft<typename TileLeft::Type, staticL0AH, staticL0AW, -1, -1>;
    using tileL0BTensor = pto::TileRight<typename TileRight::Type, staticL0BH, staticL0BW, -1, -1>;
    using tileL0CTensor = pto::TileAcc<typename TileAcc::Type, staticL0CH, staticL0CW, -1, -1>;
    using tileAccInTensor = pto::TileAcc<int32_t, staticInH, staticInW, -1, -1>;
    if constexpr (isGemv) {
        validM = 1; // GEMV M=1，不打 pad 到 16
    } else {
        validM = (validM + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    }
    tileL0ATensor l0a(validM, validK);
    tileL0BTensor l0b(validK, validN);
    tileL0CTensor l0c(validM, validN);
    tileAccInTensor accIn(validM, validN);
    pto::TASSIGN(l0a, static_cast<uint64_t>(a.GetAddr()));
    pto::TASSIGN(l0b, static_cast<uint64_t>(b.GetAddr()));
    pto::TASSIGN(l0c, static_cast<uint64_t>(c.GetAddr()));
    pto::TASSIGN(accIn, static_cast<uint64_t>(cIn.GetAddr()));
    __cbuf__ int32_t* accPtr = (__cbuf__ int32_t*)pto::GetCbufTilePtr(accIn.data());
    // 末片把 int32 累加器转 out dtype：走标量反量化（DeqF16Scalar 等），与单 K 片 fp16 输出一致。
    // 不能用 TMatmulMacro<...isAcc=true>：它对 half 输出选 InferPreQuant=DeqF16Vector，
    // 会按 preQuantTileAddr 读 per-channel scale 向量；非量化 matmul 未下发该向量（地址 0），
    // 反量化结果为全 0。此处 acc 作为 bias 输入、initCtrl=false，等价作者 RunTMATMUL_SPLIT_K
    // 的末片语义但取标量档。
    // 带 quant_pre 的多 K 末片例外：per-channel scale 已随操作数下发到 FIXBUF，此时走向量档，
    // 由 m_fpc 里的向量地址取 scale（与单 K 片 TMatmulQuant 同档）。
    using OutType = typename TileAcc::Type;
    constexpr auto preQuantScalar = std::is_same<OutType, half>::value    ? pto::MatmulPreQuant::DeqF16Scalar :
                                    std::is_same<OutType, int8_t>::value  ? pto::MatmulPreQuant::ReqS8Scalar :
                                    std::is_same<OutType, int16_t>::value ? pto::MatmulPreQuant::DeqS16Scalar :
                                                                            pto::MatmulPreQuant::NoQuant;
    constexpr auto preQuantVector = std::is_same<OutType, half>::value    ? pto::MatmulPreQuant::DeqF16Vector :
                                    std::is_same<OutType, int8_t>::value  ? pto::MatmulPreQuant::ReqS8Vector :
                                    std::is_same<OutType, int16_t>::value ? pto::MatmulPreQuant::DeqS16Vector :
                                                                            pto::MatmulPreQuant::NoQuant;
    constexpr auto preQuant = withQuantPre ? preQuantVector : preQuantScalar;
    if constexpr (withQuantPre) {
        pto::MatmulMacroConfig cfg;
        cfg.preQuantTileAddr = quantPreAddr;
        cfg.quantMode = pto::MatmulQuantMode::VectorQuant;
        cfg.gemvCtrl = isGemv;
        pto::SetMFpcFromConfig<typename TileLeft::Type, preQuant>(pto::MatmulPreRelu::NoRelu, cfg);
    } else {
        pto::SetMFpcForFp16<OutType>();
    }
    pto::SetMQuantPre<OutType>(0.0f);
    pto::MatmulConfig matmulCfg;
    matmulCfg.initCtrl = false;
    matmulCfg.preQuant = preQuant;
    matmulCfg.gemvCtrl = isGemv;
    matmulCfg.sizeN = static_cast<uint16_t>(validN);
    pto::TMatmulCore<tileL0CTensor, tileL0ATensor, tileL0BTensor>(l0c.data(), l0a.data(), l0b.data(), accPtr, 0, validM,
                                                                  validK, validN, matmulCfg.Build());
}

// kirin9050 matmul 宏指令（pre-stage quant/dequant、bias、GEMV）：
// 对接 pto-isa kirin9050 TMATMUL/TMATMUL_ACC 的 MatmulMacroConfig 重载。
// quant_pre 为 per-channel scale 向量（shape (1,n)，cbuf 驻留），地址经 MatmulMacroConfig.preQuantTileAddr
// 由 SetMFpcFromConfig 写入 m_fpc。
template <bool initMatrixC, bool isGemv = false, typename TileAcc, typename TileLeft, typename TileRight,
          typename TileQuantPre>
INLINE void TMatmulMacroInstrImpl(TileAcc& c, TileLeft& a, TileRight& b, TileQuantPre& quantPre)
{
    int64_t validM = GetShape<0>(a);
    int64_t validN = GetShape<1>(b);
    int64_t validK = GetShape<1>(a);
    if (validM == 0 || validN == 0 || (validK == 0 && !initMatrixC)) {
        return;
    }
    constexpr uint64_t shapeSizeA = Std::tuple_size<typename TileLeft::Shape>::value;
    constexpr uint64_t shapeSizeB = Std::tuple_size<typename TileRight::Shape>::value;
    constexpr uint64_t shapeSizeC = Std::tuple_size<typename TileAcc::Shape>::value;
    constexpr auto staticL0AH = Std::tuple_element<shapeSizeA - SHAPE_DIM2, typename TileLeft::TileShape>::type::value;
    constexpr auto staticL0AW = Std::tuple_element<shapeSizeA - 1, typename TileLeft::TileShape>::type::value;
    constexpr auto staticL0BH = Std::tuple_element<shapeSizeB - SHAPE_DIM2, typename TileRight::TileShape>::type::value;
    constexpr auto staticL0BW = Std::tuple_element<shapeSizeB - 1, typename TileRight::TileShape>::type::value;
    constexpr auto staticL0CH = Std::tuple_element<shapeSizeC - SHAPE_DIM2, typename TileAcc::TileShape>::type::value;
    constexpr auto staticL0CW = Std::tuple_element<shapeSizeC - 1, typename TileAcc::TileShape>::type::value;
    if (validK == 0 && initMatrixC) {
        validK = staticL0AW;
    }
    using tileL0ATensor = pto::TileLeft<typename TileLeft::Type, staticL0AH, staticL0AW, -1, -1>;
    using tileL0BTensor = pto::TileRight<typename TileRight::Type, staticL0BH, staticL0BW, -1, -1>;
    using tileL0CTensor = pto::TileAcc<typename TileAcc::Type, staticL0CH, staticL0CW, -1, -1>;
    if constexpr (isGemv) {
        validM = 1; // GEMV M=1，不打 pad 到 16
    } else {
        validM = (validM + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    }
    tileL0ATensor l0a(validM, validK);
    tileL0BTensor l0b(validK, validN);
    tileL0CTensor l0c(validM, validN);
    pto::TASSIGN(l0a, static_cast<uint64_t>(a.GetAddr()));
    pto::TASSIGN(l0b, static_cast<uint64_t>(b.GetAddr()));
    pto::TASSIGN(l0c, static_cast<uint64_t>(c.GetAddr()));
    pto::MatmulMacroConfig cfg;
    cfg.preQuantTileAddr = static_cast<uint64_t>(quantPre.GetAddr());
    cfg.quantMode = pto::MatmulQuantMode::VectorQuant;
    cfg.gemvCtrl = isGemv;
    if constexpr (initMatrixC) {
        pto::TMATMUL(l0c, l0a, l0b, cfg);
    } else {
        // 宏指令 partial sum 与输出同地址 inplace 累加
        pto::TMATMUL<pto::AccPhase::Unspecified, tileL0CTensor, tileL0ATensor, tileL0BTensor, true>(l0c, l0a, l0b, cfg);
    }
}

template <bool initMatrixC, bool isGemv = false, typename TileAcc, typename TileLeft, typename TileRight,
          typename TileBias, typename TileQuantPre>
INLINE void TMatmulMacroInstrBiasImpl(TileAcc& c, TileLeft& a, TileRight& b, TileBias& bias, TileQuantPre& quantPre)
{
    int64_t validM = GetShape<0>(a);
    int64_t validN = GetShape<1>(b);
    int64_t validK = GetShape<1>(a);
    if (validM == 0 || validN == 0 || validK == 0) {
        return;
    }
    constexpr uint64_t shapeSizeA = Std::tuple_size<typename TileLeft::Shape>::value;
    constexpr uint64_t shapeSizeC = Std::tuple_size<typename TileAcc::Shape>::value;
    constexpr uint64_t shapeSizeB = Std::tuple_size<typename TileRight::Shape>::value;
    constexpr auto staticL0AW = Std::tuple_element<shapeSizeA - 1, typename TileLeft::TileShape>::type::value;
    constexpr auto staticL0AH = Std::tuple_element<shapeSizeA - SHAPE_DIM2, typename TileLeft::TileShape>::type::value;
    constexpr auto staticL0BH = Std::tuple_element<shapeSizeB - SHAPE_DIM2, typename TileRight::TileShape>::type::value;
    constexpr auto staticL0BW = Std::tuple_element<shapeSizeB - 1, typename TileRight::TileShape>::type::value;
    constexpr auto staticL0CW = Std::tuple_element<shapeSizeC - 1, typename TileAcc::TileShape>::type::value;
    constexpr auto staticL0CH = Std::tuple_element<shapeSizeC - SHAPE_DIM2, typename TileAcc::TileShape>::type::value;
    using tileL0ATensor = pto::TileLeft<typename TileLeft::Type, staticL0AH, staticL0AW, -1, -1>;
    using tileL0BTensor = pto::TileRight<typename TileRight::Type, staticL0BH, staticL0BW, -1, -1>;
    using tileL0CTensor = pto::TileAcc<typename TileAcc::Type, staticL0CH, staticL0CW, -1, -1>;
    using tileBiasTensor = pto::Tile<pto::TileType::Bias, typename TileBias::Type, 1, staticL0BW,
                                     pto::BLayout::RowMajor, -1, -1>;
    if constexpr (isGemv) {
        validM = 1; // GEMV M=1，不打 pad 到 16
    } else {
        validM = (validM + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    }
    tileL0ATensor l0a(validM, validK);
    tileL0BTensor l0b(validK, validN);
    tileL0CTensor l0c(validM, validN);
    tileBiasTensor biasT(1, validN);
    pto::TASSIGN(l0a, static_cast<uint64_t>(a.GetAddr()));
    pto::TASSIGN(l0b, static_cast<uint64_t>(b.GetAddr()));
    pto::TASSIGN(l0c, static_cast<uint64_t>(c.GetAddr()));
    pto::TASSIGN(biasT, static_cast<uint64_t>(bias.GetAddr()));
    pto::MatmulMacroConfig cfg;
    cfg.preQuantTileAddr = static_cast<uint64_t>(quantPre.GetAddr());
    cfg.quantMode = pto::MatmulQuantMode::VectorQuant;
    cfg.gemvCtrl = isGemv;
    // bias 仅在首 K 片下发（构图保证），initMatrixC=true 时为 C=A*B+bias 零启动
    pto::TMATMUL(l0c, l0a, l0b, biasT, cfg);
}

// ---- kirin9050 宏指令 tileop 入口（与非宏指令 TMatmul/TGEMV 解耦，由 codegen 显式调用） ----

// 多 K 片 int32 累加（initMatrixC=true 为首片零启动，false 为中间片 inplace 累加）
template <bool initMatrixC, typename TileAcc, typename TileLeft, typename TileRight>
TILEOP void TMatmulMacroAcc(TileAcc& c, TileLeft& a, TileRight& b)
{
    TMatmulMacroAccImpl<initMatrixC, false>(c, a, b);
}

template <bool initMatrixC, typename TileAcc, typename TileLeft, typename TileRight>
TILEOP void TGemvMacroAcc(TileAcc& c, TileLeft& a, TileRight& b)
{
    TMatmulMacroAccImpl<initMatrixC, true>(c, a, b);
}

// 首片带 bias（bias 经 BT 下发，int32 累加域）
template <typename TileAcc, typename TileLeft, typename TileRight, typename TileBias>
TILEOP void TMatmulMacroAccBias(TileAcc& c, TileLeft& a, TileRight& b, TileBias& bias)
{
    TMatmulMacroAccBiasImpl<false>(c, a, b, bias);
}

template <typename TileAcc, typename TileLeft, typename TileRight, typename TileBias>
TILEOP void TGemvMacroAccBias(TileAcc& c, TileLeft& a, TileRight& b, TileBias& bias)
{
    TMatmulMacroAccBiasImpl<true>(c, a, b, bias);
}

// 末片：读 int32 partial sum 累加器 -> 转 out dtype 写出（acc 与输出分离）
template <typename TileAcc, typename TileLeft, typename TileRight, typename TileAccIn>
TILEOP void TMatmulMacroAccFrom(TileAcc& c, TileLeft& a, TileRight& b, TileAccIn& cIn)
{
    TMatmulMacroAccFromImpl<false>(c, a, b, cIn);
}

template <typename TileAcc, typename TileLeft, typename TileRight, typename TileAccIn>
TILEOP void TGemvMacroAccFrom(TileAcc& c, TileLeft& a, TileRight& b, TileAccIn& cIn)
{
    TMatmulMacroAccFromImpl<true>(c, a, b, cIn);
}

// 末片（带 per-channel quant_pre）：读 int32 partial sum 累加器 -> 按 per-channel scale 反量化 -> 转 out dtype
template <typename TileAcc, typename TileLeft, typename TileRight, typename TileAccIn, typename TileQuantPre>
TILEOP void TMatmulMacroAccFromQuant(TileAcc& c, TileLeft& a, TileRight& b, TileAccIn& cIn, TileQuantPre& quantPre)
{
    TMatmulMacroAccFromImpl<false, true>(c, a, b, cIn, static_cast<uint64_t>(quantPre.GetAddr()));
}

template <typename TileAcc, typename TileLeft, typename TileRight, typename TileAccIn, typename TileQuantPre>
TILEOP void TGemvMacroAccFromQuant(TileAcc& c, TileLeft& a, TileRight& b, TileAccIn& cIn, TileQuantPre& quantPre)
{
    TMatmulMacroAccFromImpl<true, true>(c, a, b, cIn, static_cast<uint64_t>(quantPre.GetAddr()));
}

// 带 quant_pre 的量化接口（pre-stage quant/dequant）。
// 删除 kirin 不需要的 transMode/kAlignFlag 模板参数，内部对接 pto-isa 的 MatmulMacroConfig 重载。
template <bool initMatrixC, typename TileAcc, typename TileLeft, typename TileRight, typename TileQuantPre>
TILEOP void TMatmulQuant(TileAcc& c, TileLeft& a, TileRight& b, TileQuantPre& quantPre)
{
    constexpr uint64_t shapeSizeA = Std::tuple_size<typename TileLeft::Shape>::value;
    constexpr uint64_t shapeSizeB = Std::tuple_size<typename TileRight::Shape>::value;
    constexpr uint64_t shapeSizeC = Std::tuple_size<typename TileAcc::Shape>::value;
    static_assert(shapeSizeA == SHAPE_DIM2 && shapeSizeB == SHAPE_DIM2 && shapeSizeC == SHAPE_DIM2,
                  "[TMatmulQuant ERROR]: Shape dim size should be 2");
    TMatmulMacroInstrImpl<initMatrixC, false>(c, a, b, quantPre);
}

template <bool initMatrixC, typename TileAcc, typename TileLeft, typename TileRight, typename TileBias,
          typename TileQuantPre>
TILEOP void TMatmulQuant(TileAcc& c, TileLeft& a, TileRight& b, TileBias& bias, TileQuantPre& quantPre)
{
    constexpr uint64_t shapeSizeA = Std::tuple_size<typename TileLeft::Shape>::value;
    constexpr uint64_t shapeSizeB = Std::tuple_size<typename TileRight::Shape>::value;
    constexpr uint64_t shapeSizeC = Std::tuple_size<typename TileAcc::Shape>::value;
    static_assert(shapeSizeA == SHAPE_DIM2 && shapeSizeB == SHAPE_DIM2 && shapeSizeC == SHAPE_DIM2,
                  "[TMatmulQuant ERROR]: Shape dim size should be 2");
    TMatmulMacroInstrBiasImpl<initMatrixC, false>(c, a, b, bias, quantPre);
}

template <bool initMatrixC, typename TileAcc, typename TileLeft, typename TileRight, typename TileQuantPre>
TILEOP void TGemvQuant(TileAcc& c, TileLeft& a, TileRight& b, TileQuantPre& quantPre)
{
    constexpr uint64_t shapeSizeA = Std::tuple_size<typename TileLeft::Shape>::value;
    constexpr uint64_t shapeSizeB = Std::tuple_size<typename TileRight::Shape>::value;
    constexpr uint64_t shapeSizeC = Std::tuple_size<typename TileAcc::Shape>::value;
    static_assert(shapeSizeA == SHAPE_DIM2 && shapeSizeB == SHAPE_DIM2 && shapeSizeC == SHAPE_DIM2,
                  "[TGemvQuant ERROR]: Shape dim size should be 2");
    TMatmulMacroInstrImpl<initMatrixC, true>(c, a, b, quantPre);
}

template <bool initMatrixC, typename TileAcc, typename TileLeft, typename TileRight, typename TileBias,
          typename TileQuantPre>
TILEOP void TGemvQuant(TileAcc& c, TileLeft& a, TileRight& b, TileBias& bias, TileQuantPre& quantPre)
{
    constexpr uint64_t shapeSizeA = Std::tuple_size<typename TileLeft::Shape>::value;
    constexpr uint64_t shapeSizeB = Std::tuple_size<typename TileRight::Shape>::value;
    constexpr uint64_t shapeSizeC = Std::tuple_size<typename TileAcc::Shape>::value;
    static_assert(shapeSizeA == SHAPE_DIM2 && shapeSizeB == SHAPE_DIM2 && shapeSizeC == SHAPE_DIM2,
                  "[TGemvQuant ERROR]: Shape dim size should be 2");
    TMatmulMacroInstrBiasImpl<initMatrixC, true>(c, a, b, bias, quantPre);
}

#endif // TILEOP_TILE_OPERATOR_KIRIN9050_MMAD_MACRO_IMPL__H
