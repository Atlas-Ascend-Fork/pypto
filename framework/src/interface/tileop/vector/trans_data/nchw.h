/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file nchw.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_TRANS_DATA_NCHW_H
#define TILEOP_TILE_OPERATOR_TRANS_DATA_NCHW_H
#include "utils/sync.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#define OP_TILE_OP_TRANSDATA_NCHW2NC1HWC0 TTransDataNCHW2NC1HWC0
template <int tileN, int tileC, int tileH, int tmpTileW, typename DST, typename TMP, typename INPUT>
__aicore__ inline void TTransDataNCHW2NC1HWC0(DST dst, TMP tmpTensor, INPUT input)
{
    constexpr auto inputTypeSize = sizeof(typename INPUT::Type);
    constexpr auto C0 = TileOp::BLOCK_SIZE / inputTypeSize;
    constexpr auto tileC1 = tileC / C0;
    constexpr auto tileW = (tmpTileW + C0 - 1) / C0 * C0;

    constexpr auto inputTileN = Std::tuple_element<DIM_1ST, typename INPUT::TileShape>::type::value;
    constexpr auto inputTileC = Std::tuple_element<DIM_2ND, typename INPUT::TileShape>::type::value;
    constexpr auto inputTileH = Std::tuple_element<DIM_3RD, typename INPUT::TileShape>::type::value;
    constexpr auto inputTileW = Std::tuple_element<DIM_4TH, typename INPUT::TileShape>::type::value;

    constexpr auto dstTileN = Std::tuple_element<DIM_1ST, typename DST::TileShape>::type::value;
    constexpr auto dstTileC1 = Std::tuple_element<DIM_2ND, typename DST::TileShape>::type::value;
    constexpr auto dstTileH = Std::tuple_element<DIM_3RD, typename DST::TileShape>::type::value;
    constexpr auto dstTileW = Std::tuple_element<DIM_4TH, typename DST::TileShape>::type::value;
    constexpr auto dstTileC0 = Std::tuple_element<DIM_5TH, typename DST::TileShape>::type::value;

    const auto inputLayout = input.GetLayout();
    auto inputN = inputLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto inputC = inputLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto inputH = inputLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto inputW = inputLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto inputPadC = (inputC + C0 - 1) / C0 * C0;
    auto padCSize = inputPadC - inputC;

    if (inputN == 0 || inputC == 0 || inputH == 0 || inputW == 0) {
        return;
    }

    Sync2_VS();
    if (padCSize != 0) {
        using TileDefine = pto::Tile<pto::TileType::Vec, typename INPUT::Type, inputTileC, inputTileH * inputTileW,
                                     pto::BLayout::RowMajor, -1, -1>;
        TileDefine tmpInputTile(padCSize, inputTileH * inputTileW);
        for (LoopVar i = 0; i < inputN; i++) {
            pto::TASSIGN(tmpInputTile, (uint64_t)(input.GetAddr() +
                                                  (i * inputTileC + inputC) * inputTileH * inputTileW * inputTypeSize));
            pto::TEXPANDS(tmpInputTile, static_cast<typename INPUT::Type>(0));
        }
        pipe_barrier(PIPE_V);
    }

    constexpr int inputElementSize = inputTileN * inputTileC * inputTileH * inputTileW;
    constexpr int inputBufferSize = inputElementSize * inputTypeSize;
    constexpr int dstElementSize = dstTileN * dstTileC1 * dstTileH * dstTileW * dstTileC0;
    constexpr int dstBufferSize = dstElementSize * inputTypeSize;

    using inputTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, inputBufferSize, pto::Layout::NCHW,
                                        pto::ConvTileShape<inputTileN, inputTileC, inputTileH, inputTileW>,
                                        pto::ConvTileShape<tileN, tileC, tileH, tileW>>;
    using tmpDstTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, dstBufferSize, pto::Layout::NC1HWC0,
                                         pto::ConvTileShape<dstTileN, dstTileC1, dstTileH, dstTileW, dstTileC0>,
                                         pto::ConvTileShape<tileN, tileC1, tileH, tileW, C0>>;
    using tmpTileData = pto::Tile<pto::TileType::Vec, typename INPUT::Type, 1, C0, pto::BLayout::RowMajor, 1, C0>;
    inputTileData convInput;
    tmpDstTileData convTmpDst;
    tmpTileData tmpAreaTile;
    auto tmpDstAddr = (__ubuf__ typename INPUT::Type*)((uint64_t)(dst.GetAddr()));
    auto tmpAreaAddr = (__ubuf__ typename INPUT::Type*)((uint64_t)(tmpTensor.GetAddr()));

    pto::TASSIGN(convInput, (uint64_t)input.GetAddr());
    pto::TASSIGN(convTmpDst, (uint64_t)tmpDstAddr);
    pto::TASSIGN(tmpAreaTile, (uint64_t)tmpAreaAddr);
    pto::TTRANS(convTmpDst, convInput, tmpAreaTile);
}

#define OP_TILE_OP_TRANSDATA_NCHW2Fractal_Z TTransDataNCHW2Fractal_Z
template <int tileN, int tileC, int tileH, int tmpTileW, typename DST, typename TMP, typename INPUT>
__aicore__ inline void TTransDataNCHW2Fractal_Z(DST dst, TMP tmpTensor, INPUT input)
{
    constexpr auto inputTypeSize = sizeof(typename INPUT::Type);
    constexpr auto C0 = TileOp::BLOCK_SIZE / inputTypeSize;
    constexpr auto N0 = 16;
    constexpr auto tileC1 = tileC / C0;
    constexpr auto tileN1 = tileN / N0;
    constexpr int elementSize = tileN * tileC * tileH * tmpTileW;
    constexpr int bufferSize = elementSize * inputTypeSize;

    const auto inputLayout = input.GetLayout();
    auto inputN = inputLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto inputCHW = inputLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto inputC = inputCHW / (tileH * tmpTileW);

    if (inputN == 0 || inputCHW == 0) {
        return;
    }

    using inputTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::NCHW,
                                        pto::ConvTileShape<tileN, tileC, tileH, tmpTileW>>;
    using tmpDst1TileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::NC1HWC0,
                                          pto::ConvTileShape<tileN, tileC1, tileH, tmpTileW, C0>>;
    using tmp1TileData = pto::Tile<pto::TileType::Vec, typename INPUT::Type, tileH * tmpTileW, C0,
                                   pto::BLayout::RowMajor, tileH * tmpTileW, C0>;
    inputTileData convInput;
    tmpDst1TileData convTmpDstNC1HWC0;
    tmp1TileData tmpTile;

    auto tmpDstNC1HWC0Addr = (__ubuf__ typename INPUT::Type*)((uint64_t)(tmpTensor.GetAddr()));
    auto tmpAreaTileAddr = tmpDstNC1HWC0Addr + elementSize;
    pto::TASSIGN(convInput, (uint64_t)input.GetAddr());
    pto::TASSIGN(convTmpDstNC1HWC0, (uint64_t)tmpDstNC1HWC0Addr);
    pto::TASSIGN(tmpTile, (uint64_t)tmpAreaTileAddr);

    Sync2_VS();
    if constexpr (inputTypeSize == 2 && tileN % N0 == 0 && tileC % C0 == 0 && C0 == N0) {
        if (inputN <= tileN && inputC <= tileC) {
            constexpr int HW = tileH * tmpTileW;
            constexpr int CHW = tileC * HW;
            auto src = (__ubuf__ typename INPUT::Type*)((uint64_t)input.GetAddr());
            auto tmp = tmpDstNC1HWC0Addr;
            auto out = (__ubuf__ typename INPUT::Type*)((uint64_t)dst.GetAddr());
            if (inputC < tileC) {
                using PadCTile = pto::Tile<pto::TileType::Vec, typename INPUT::Type, 1, CHW, pto::BLayout::RowMajor, -1,
                                           -1>;
                PadCTile padCTile(1, (tileC - inputC) * HW);
                for (LoopVar n = 0; n < inputN; ++n) {
                    pto::TASSIGN(padCTile, (uint64_t)(src + n * CHW + inputC * HW));
                    pto::TEXPANDS(padCTile, static_cast<typename INPUT::Type>(0));
                }
            }
            if (inputN < tileN) {
                using PadNTile = pto::Tile<pto::TileType::Vec, typename INPUT::Type, tileN, CHW, pto::BLayout::RowMajor,
                                           -1, -1>;
                PadNTile padNTile(tileN - inputN, CHW);
                pto::TASSIGN(padNTile, (uint64_t)(src + inputN * CHW));
                pto::TEXPANDS(padNTile, static_cast<typename INPUT::Type>(0));
            }
            if (inputN < tileN || inputC < tileC) {
                pipe_barrier(PIPE_V);
            }
            uint64_t dstAddr[16];
            uint64_t srcAddr[16];
            // [N, C*HW] -> [C*HW, N]
            for (int n = 0; n < tileN; n += N0) {
                for (int chw = 0; chw < CHW; chw += N0) {
                    for (int i = 0; i < N0; ++i) {
                        srcAddr[i] = (uint64_t)(src + (n + i) * CHW + chw);
                        dstAddr[i] = (uint64_t)(tmp + (chw + i) * tileN + n);
                    }
                    set_va_reg_sb(VA0, dstAddr);
                    set_va_reg_sb(VA1, dstAddr + 8);
                    set_va_reg_sb(VA2, srcAddr);
                    set_va_reg_sb(VA3, srcAddr + 8);
                    // A single 16x16 block uses zero repeat strides.
                    scatter_vnchwconv_b16(VA0, VA2, 1, 0, 0);
                }
            }
            pipe_barrier(PIPE_V);
            // [C, HW, N] -> [C1*HW, N, C0]
            for (int c1 = 0; c1 < tileC1; ++c1) {
                for (int hw = 0; hw < HW; ++hw) {
                    for (int n = 0; n < tileN; n += N0) {
                        for (int i = 0; i < C0; ++i) {
                            srcAddr[i] = (uint64_t)(tmp + ((c1 * C0 + i) * HW + hw) * tileN + n);
                            dstAddr[i] = (uint64_t)(out + ((c1 * HW + hw) * tileN + n + i) * C0);
                        }
                        set_va_reg_sb(VA0, dstAddr);
                        set_va_reg_sb(VA1, dstAddr + 8);
                        set_va_reg_sb(VA2, srcAddr);
                        set_va_reg_sb(VA3, srcAddr + 8);
                        scatter_vnchwconv_b16(VA0, VA2, 1, 0, 0);
                    }
                }
            }
            return;
        }
    }
    pto::TTRANS(convTmpDstNC1HWC0, convInput, tmpTile);
    pipe_barrier(PIPE_V);
    using tmpDst2TileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::FRACTAL_Z,
                                          pto::ConvTileShape<tileC1 * tileH * tmpTileW, tileN1, N0, C0>>;
    tmpDst2TileData dstFractalZ;
    pto::TASSIGN(dstFractalZ, (uint64_t)dst.GetAddr());
    pto::TTRANS(dstFractalZ, convTmpDstNC1HWC0, tmpTile);
}

#define OP_TILE_OP_TRANSDATA_NC1HWC02NCHW TTransDataNC1HWC02NCHW
template <int tileN, int tileC1, int tileH, int tileW, int C0, typename DST, typename TMP, typename INPUT>
__aicore__ inline void TTransDataNC1HWC02NCHW(DST dst, TMP tmpTensor, INPUT input)
{
    constexpr auto inputTypeSize = sizeof(typename INPUT::Type);

    constexpr auto inputTileN = Std::tuple_element<DIM_1ST, typename INPUT::TileShape>::type::value;
    constexpr auto inputTileC1 = Std::tuple_element<DIM_2ND, typename INPUT::TileShape>::type::value;
    constexpr auto inputTileH = Std::tuple_element<DIM_3RD, typename INPUT::TileShape>::type::value;
    constexpr auto inputTileW = Std::tuple_element<DIM_4TH, typename INPUT::TileShape>::type::value;

    constexpr auto dstTileN = Std::tuple_element<DIM_1ST, typename DST::TileShape>::type::value;
    constexpr auto dstTileC = Std::tuple_element<DIM_2ND, typename DST::TileShape>::type::value;
    constexpr auto dstTileH = Std::tuple_element<DIM_3RD, typename DST::TileShape>::type::value;
    constexpr auto dstTileW = Std::tuple_element<DIM_4TH, typename DST::TileShape>::type::value;

    auto inputLayout = input.GetLayout();
    auto inputN = inputLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto inputC1 = inputLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto inputH = inputLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto inputW = inputLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    if (inputN == 0 || inputC1 == 0 || inputH == 0 || inputW == 0) {
        return;
    }

    constexpr int inputElementSize = inputTileN * inputTileC1 * inputTileH * inputTileW * C0;
    constexpr int inputBufferSize = inputElementSize * inputTypeSize;
    constexpr int dstElementSize = dstTileN * dstTileC * dstTileH * dstTileW;
    constexpr int dstBufferSize = dstElementSize * inputTypeSize;
    using inputTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, inputBufferSize, pto::Layout::NC1HWC0,
                                        pto::ConvTileShape<inputTileN, inputTileC1, inputTileH, inputTileW, C0>,
                                        pto::ConvTileShape<tileN, tileC1, tileH, tileW, C0>>;
    using tmpDstTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, dstBufferSize, pto::Layout::NCHW,
                                         pto::ConvTileShape<dstTileN, dstTileC, dstTileH, dstTileW>,
                                         pto::ConvTileShape<tileN, tileC1 * C0, tileH, tileW>>;
    using tmpTileData = pto::Tile<pto::TileType::Vec, typename INPUT::Type, 1, C0, pto::BLayout::RowMajor, 1, C0>;
    inputTileData convInput;
    tmpDstTileData convTmpDst;
    tmpTileData tmpAreaTile;

    pto::TASSIGN(convInput, (uint64_t)input.GetAddr());
    pto::TASSIGN(convTmpDst, (uint64_t)dst.GetAddr());
    pto::TASSIGN(tmpAreaTile, (uint64_t)tmpTensor.GetAddr());
    pto::TTRANS(convTmpDst, convInput, tmpAreaTile);
}

#define OP_TILE_OP_TRANSDATA_FractalZ2NCHW TTransDataFractalZ2NCHW
template <typename DST, typename TMP, typename INPUT>
__aicore__ inline void TTransDataFractalZ2NCHW(DST dst, TMP tmpTensor, INPUT input)
{
    constexpr auto inputTypeSize = sizeof(typename INPUT::Type);
    constexpr auto C0 = TileOp::BLOCK_SIZE / inputTypeSize;
    constexpr auto N0 = 16;
    constexpr auto TileC1HW = Std::tuple_element<DIM_1ST, typename INPUT::TileShape>::type::value;
    constexpr auto TileN1 = Std::tuple_element<DIM_2ND, typename INPUT::TileShape>::type::value;
    constexpr auto dstTileH = Std::tuple_element<DIM_3RD, typename DST::TileShape>::type::value;
    constexpr auto dstTileW = Std::tuple_element<DIM_4TH, typename DST::TileShape>::type::value;
    constexpr auto TileC1 = TileC1HW / dstTileH / dstTileW;
    constexpr int elementSize = TileC1HW * TileN1 * N0 * C0;
    constexpr int bufferSize = elementSize * inputTypeSize;
    constexpr int TileN = N0 * TileN1;

    const auto inputLayout = input.GetLayout();
    auto inputC1HW = inputLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto inputN1 = inputLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto inputN0 = inputLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto inputC0 = inputLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    using Tdst = typename INPUT::Type;
    using Tsrc = typename INPUT::Type;
    using Ttmp = typename INPUT::Type;

    __ubuf__ Tsrc* inputAddr = (__ubuf__ Tsrc*)((uint64_t)(input.GetAddr()));
    __ubuf__ Tdst* dstAddr = (__ubuf__ Tdst*)((uint64_t)(dst.GetAddr()));
    __ubuf__ Ttmp* tmpAddr = (__ubuf__ Ttmp*)((uint64_t)(tmpTensor.GetAddr()));

    if (inputC1HW == 0 || inputN1 == 0 || inputN0 == 0 || inputC0 == 0) {
        return;
    }

    uint32_t burstNum = TileN;
    uint32_t lenBurst = (C0 * inputTypeSize + TileOp::BLOCK_SIZE - 1) / TileOp::BLOCK_SIZE;
    uint32_t srcGap = 0;
    uint32_t dstGap = (TileC1HW * C0 * inputTypeSize + TileOp::BLOCK_SIZE - 1) / TileOp::BLOCK_SIZE - lenBurst;
    for (int i = 0; i < TileC1HW; i++) {
        __ubuf__ Tsrc* srcPtr = inputAddr + i * TileN * C0;
        __ubuf__ Tdst* tmpPtr = tmpAddr + i * C0;
        pto::pto_copy_ubuf_to_ubuf(tmpPtr, srcPtr, burstNum, lenBurst, srcGap, dstGap);
    }
    pipe_barrier(PIPE_V);

    __ubuf__ Ttmp* tmpAreaTileAddr = tmpAddr + elementSize;
    using inputTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::NC1HWC0,
                                        pto::ConvTileShape<TileN, TileC1, dstTileH, dstTileW, C0>>;
    using tmpDstTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::NCHW,
                                         pto::ConvTileShape<TileN, TileC1 * C0, dstTileH, dstTileW>>;
    using tmpTileData = pto::Tile<pto::TileType::Vec, typename INPUT::Type, dstTileH * dstTileW, C0,
                                  pto::BLayout::RowMajor, dstTileH * dstTileW, C0>;
    inputTileData convInput;
    tmpDstTileData convTmpDst;
    tmpTileData tmpAreaTile;

    pto::TASSIGN(convInput, (uint64_t)(tmpAddr));
    pto::TASSIGN(convTmpDst, (uint64_t)(dstAddr));
    pto::TASSIGN(tmpAreaTile, (uint64_t)(tmpAreaTileAddr));
    pto::TTRANS(convTmpDst, convInput, tmpAreaTile);
}

#endif
