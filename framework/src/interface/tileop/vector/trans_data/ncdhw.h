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
 * \file ncdhw.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_TRANS_DATA_NCDHW_H
#define TILEOP_TILE_OPERATOR_TRANS_DATA_NCDHW_H
#include "utils/sync.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#define OP_TILE_OP_TRANSDATA_NCDHW2NDC1HWC0 TTransDataNCDHW2NDC1HWC0
template <int tileN, int tileD, int tileC, int tileH, int tmpTileW, typename DST, typename TMP, typename INPUT>
__aicore__ inline void TTransDataNCDHW2NDC1HWC0(DST dst, TMP tmpTensor, INPUT input)
{
    constexpr auto inputTypeSize = sizeof(typename INPUT::Type);
    constexpr auto C0 = TileOp::BLOCK_SIZE / inputTypeSize;
    constexpr auto tileC1 = tileC / C0;
    constexpr auto tileW = (tmpTileW + C0 - 1) / C0 * C0;

    constexpr auto inputTileN = Std::tuple_element<DIM_1ST, typename INPUT::TileShape>::type::value;
    constexpr auto inputTileD = Std::tuple_element<DIM_2ND, typename INPUT::TileShape>::type::value;
    constexpr auto inputTileC = Std::tuple_element<DIM_3RD, typename INPUT::TileShape>::type::value;
    constexpr auto inputTileH = Std::tuple_element<DIM_4TH, typename INPUT::TileShape>::type::value;
    constexpr auto inputTileW = Std::tuple_element<DIM_5TH, typename INPUT::TileShape>::type::value;

    constexpr auto dstTileD = Std::tuple_element<DIM_1ST, typename DST::TileShape>::type::value;
    constexpr auto dstTileC1 = Std::tuple_element<DIM_2ND, typename DST::TileShape>::type::value;
    constexpr auto dstTileH = Std::tuple_element<DIM_3RD, typename DST::TileShape>::type::value;
    constexpr auto dstTileW = Std::tuple_element<DIM_4TH, typename DST::TileShape>::type::value;
    constexpr auto dstTileC0 = Std::tuple_element<DIM_5TH, typename DST::TileShape>::type::value;

    const auto inputLayout = input.GetLayout();
    auto inputN = inputLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto inputD = inputLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto inputC = inputLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto inputH = inputLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto inputW = inputLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    auto inputPadC = (inputC + C0 - 1) / C0 * C0;
    auto padCSize = inputPadC - inputC;

    if (inputN == 0 || inputD == 0 || inputC == 0 || inputH == 0 || inputW == 0) {
        return;
    }

    Sync23_VS();
    if (padCSize != 0) {
        using TileDefine = pto::Tile<pto::TileType::Vec, typename INPUT::Type, inputTileC, inputTileH * inputTileW,
                                     pto::BLayout::RowMajor, -1, -1>;
        TileDefine tmpInputTile(padCSize, inputTileH * inputTileW);
        for (LoopVar i = 0; i < inputN; i++) {
            for (LoopVar j = 0; j < inputD; j++) {
                pto::TASSIGN(
                    tmpInputTile,
                    (uint64_t)(input.GetAddr() + i * inputTileD * inputTileC * inputTileH * inputTileW * inputTypeSize +
                               (j * inputTileC + inputC) * inputTileH * inputTileW * inputTypeSize));
                pto::TEXPANDS(tmpInputTile, static_cast<typename INPUT::Type>(0));
            }
        }
        pipe_barrier(PIPE_V);
    }

    constexpr int inputElementSize = inputTileN * inputTileD * inputTileC * inputTileH * inputTileW;
    constexpr int inputBufferSize = inputElementSize * inputTypeSize;
    constexpr int dstElementSize = dstTileD * dstTileC1 * dstTileH * dstTileW * dstTileC0;
    constexpr int dstBufferSize = dstElementSize * inputTypeSize;

    using inputTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, inputBufferSize, pto::Layout::GNCHW,
                                        pto::ConvTileShape<inputTileN, inputTileD, inputTileC, inputTileH, inputTileW>,
                                        pto::ConvTileShape<tileN, tileD, tileC, tileH, tileW>>;
    using tmpDstTileData = pto::ConvTile<
        pto::TileType::Vec, typename INPUT::Type, dstBufferSize, pto::Layout::GNC1HWC0,
        pto::ConvTileShape<inputTileN, dstTileD, dstTileC1, dstTileH, dstTileW, dstTileC0>,
        pto::ConvTileShape<tileN, tileD, tileC1, tileH, tileW, C0>>;
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

#define OP_TILE_OP_TRANSDATA_NDC1HWC02NCDHW TTransDataNDC1HWC02NCDHW
template <int tileD, int tileC1, int tileH, int tileW, int C0, typename DST, typename TMP, typename INPUT>
__aicore__ inline void TTransDataNDC1HWC02NCDHW(DST dst, TMP tmpTensor, INPUT input)
{
    constexpr auto inputTypeSize = sizeof(typename INPUT::Type);

    constexpr auto inputTileD = Std::tuple_element<DIM_1ST, typename INPUT::TileShape>::type::value;
    constexpr auto inputTileC1 = Std::tuple_element<DIM_2ND, typename INPUT::TileShape>::type::value;
    constexpr auto inputTileH = Std::tuple_element<DIM_3RD, typename INPUT::TileShape>::type::value;
    constexpr auto inputTileW = Std::tuple_element<DIM_4TH, typename INPUT::TileShape>::type::value;

    constexpr auto dstTileD = Std::tuple_element<DIM_2ND, typename DST::TileShape>::type::value;
    constexpr auto dstTileC = Std::tuple_element<DIM_3RD, typename DST::TileShape>::type::value;
    constexpr auto dstTileH = Std::tuple_element<DIM_4TH, typename DST::TileShape>::type::value;
    constexpr auto dstTileW = Std::tuple_element<DIM_5TH, typename DST::TileShape>::type::value;

    auto inputLayout = input.GetLayout();
    auto inputD = inputLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto inputC1 = inputLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto inputH = inputLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto inputW = inputLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    if (inputD == 0 || inputC1 == 0 || inputH == 0 || inputW == 0) {
        return;
    }

    constexpr int inputElementSize = inputTileD * inputTileC1 * inputTileH * inputTileW * C0;
    constexpr int inputBufferSize = inputElementSize * inputTypeSize;
    constexpr int dstElementSize = dstTileD * dstTileC * dstTileH * dstTileW;
    constexpr int dstBufferSize = dstElementSize * inputTypeSize;
    using inputTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, inputBufferSize, pto::Layout::NC1HWC0,
                                        pto::ConvTileShape<inputTileD, inputTileC1, inputTileH, inputTileW, C0>,
                                        pto::ConvTileShape<tileD, tileC1, tileH, tileW, C0>>;
    using tmpDstTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, dstBufferSize, pto::Layout::NCHW,
                                         pto::ConvTileShape<dstTileD, dstTileC, dstTileH, dstTileW>,
                                         pto::ConvTileShape<tileD, tileC1 * C0, tileH, tileW>>;
    using tmpTileData = pto::Tile<pto::TileType::Vec, typename INPUT::Type, 1, C0, pto::BLayout::RowMajor, 1, C0>;
    inputTileData convInput;
    tmpDstTileData convTmpDst;
    tmpTileData tmpAreaTile;

    pto::TASSIGN(convInput, (uint64_t)input.GetAddr());
    pto::TASSIGN(convTmpDst, (uint64_t)dst.GetAddr());
    pto::TASSIGN(tmpAreaTile, (uint64_t)tmpTensor.GetAddr());
    pto::TTRANS(convTmpDst, convInput, tmpAreaTile);
}

#define OP_TILE_OP_TRANSDATA_NCDHW2FRACTAL_Z_3D TTransDataNCDHW2FRACTAL_Z_3D
template <int tileN, int tileC, int tileD, int tileH, int tmpTileW, typename DST, typename TMP, typename INPUT>
__aicore__ inline void TTransDataNCDHW2FRACTAL_Z_3D(DST dst, TMP tmpTensor, INPUT input)
{
    constexpr auto inputTypeSize = sizeof(typename INPUT::Type);
    constexpr auto C0 = TileOp::BLOCK_SIZE / inputTypeSize;
    constexpr auto N0 = 16;
    constexpr auto tileC1 = tileC / C0;
    constexpr auto tileN1 = tileN / N0;
    constexpr int elementSize = tileN * tileC * tileD * tileH * tmpTileW;
    constexpr int bufferSize = elementSize * inputTypeSize;

    const auto inputLayout = input.GetLayout();
    auto inputN = inputLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto inputCDHW = inputLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();

    if (inputN == 0 || inputCDHW == 0) {
        return;
    }

    using inputTileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize, pto::Layout::NCDHW,
                                        pto::ConvTileShape<tileN, tileC, tileD, tileH, tmpTileW>>;
    using tmpDst1TileData = pto::ConvTile<pto::TileType::Vec, typename INPUT::Type, bufferSize,
                                          pto::Layout::FRACTAL_Z_3D,
                                          pto::ConvTileShape<tileD * tileC1 * tileH * tmpTileW, tileN1, N0, C0>>;
    using tmp1TileData = pto::Tile<pto::TileType::Vec, typename INPUT::Type, tileH * tmpTileW, C0,
                                   pto::BLayout::RowMajor, tileH * tmpTileW, C0>;
    inputTileData convInput;
    tmpDst1TileData ConvDst;
    tmp1TileData tmpTile;

    pto::TASSIGN(convInput, (uint64_t)input.GetAddr());
    pto::TASSIGN(ConvDst, (uint64_t)dst.GetAddr());
    pto::TASSIGN(tmpTile, (uint64_t)tmpTensor.GetAddr());

    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    pto::TTRANS(ConvDst, convInput, tmpTile);
}

template <typename T>
__aicore__ void ConvNCHWPlane2NCDHW(__ubuf__ T* dst, __ubuf__ T* src, unsigned dstN, unsigned dstC, unsigned dstD,
                                    unsigned dstH, unsigned dstW, unsigned paddedC, unsigned d)
{
    unsigned hw = dstH * dstW;
    unsigned ncStride = dstD * hw;
    unsigned srcNStride = paddedC * hw;
    unsigned dstNStride = paddedC * ncStride;

    uint32_t lenBurst = hw * sizeof(T) / pto::BLOCK_BYTE_SIZE;
    uint32_t srcGap = 0;
    uint32_t dstGap = (ncStride - hw) * sizeof(T) / pto::BLOCK_BYTE_SIZE;
    for (unsigned n = 0; n < dstN; n++) {
        __ubuf__ T* sBase = src + n * srcNStride;
        __ubuf__ T* tBase = dst + n * dstNStride + d * hw;
        copy_ubuf_to_ubuf(tBase, sBase, 0, (uint32_t)dstC, (uint32_t)lenBurst, srcGap, dstGap);
    }
}

#define OP_TILE_OP_TRANSDATA_FractalZ3D2NCDHW TTransDataFractalZ3D2NCDHW
template <typename DST, typename TMP, typename INPUT>
__aicore__ inline void TTransDataFractalZ3D2NCDHW(DST dst, TMP tmpTensor, INPUT input)
{
    using Tdst = typename DST::Type;
    using Tsrc = typename INPUT::Type;
    using Ttmp = typename TMP::Type;

    constexpr auto inputTypeSize = sizeof(Tsrc);
    constexpr auto C0 = TileOp::BLOCK_SIZE / inputTypeSize;
    constexpr auto N0 = 16;

    constexpr auto dstTileN = Std::tuple_element<DIM_1ST, typename DST::TileShape>::type::value;
    constexpr auto dstTileC = Std::tuple_element<DIM_2ND, typename DST::TileShape>::type::value;
    constexpr auto dstTileD = Std::tuple_element<DIM_3RD, typename DST::TileShape>::type::value;
    constexpr auto dstTileH = Std::tuple_element<DIM_4TH, typename DST::TileShape>::type::value;
    constexpr auto dstTileW = Std::tuple_element<DIM_5TH, typename DST::TileShape>::type::value;

    constexpr unsigned C1_dstTile = dstTileC / C0;
    constexpr unsigned paddedC_dstTile = dstTileC;
    constexpr unsigned paddedN_dstTile = dstTileN;
    constexpr unsigned hw_dstTile = dstTileH * dstTileW;
    constexpr unsigned c1hw_dstTile = C1_dstTile * hw_dstTile;
    constexpr unsigned ncplaneSize = paddedN_dstTile * paddedC_dstTile * hw_dstTile;
    constexpr unsigned srcSliceSize = c1hw_dstTile * paddedN_dstTile * C0;

    const auto inputLayout = input.GetLayout();
    auto inputDC1HW = inputLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto inputN1 = inputLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto inputN0 = inputLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>();
    auto inputC0 = inputLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>();
    if (inputDC1HW == 0 || inputN1 == 0 || inputN0 == 0 || inputC0 == 0) {
        return;
    }

    __ubuf__ Tsrc* dstPtrOrig = (__ubuf__ Tsrc*)((uint64_t)(dst.GetAddr()));
    __ubuf__ Tdst* srcPtrOrig = (__ubuf__ Tdst*)((uint64_t)(input.GetAddr()));
    __ubuf__ Ttmp* tmpPtrOrig = (__ubuf__ Ttmp*)((uint64_t)(tmpTensor.GetAddr()));

    __ubuf__ Tsrc* nc1hwc0Ptr = (__ubuf__ Tsrc*)tmpPtrOrig;
    __ubuf__ Tdst* nchwPtr = nc1hwc0Ptr + ncplaneSize;
    __ubuf__ Ttmp* tmpAreaPtr = tmpPtrOrig + 2 * ncplaneSize;

    constexpr int elementSize2 = dstTileN * dstTileC * dstTileH * dstTileW;
    constexpr int bufferSize2 = elementSize2 * inputTypeSize;

    using nc1hwc0TileData = pto::ConvTile<pto::TileType::Vec, Tsrc, bufferSize2, pto::Layout::NC1HWC0,
                                          pto::ConvTileShape<dstTileN, dstTileC / C0, dstTileH, dstTileW, C0>>;
    using nchwTileData = pto::ConvTile<pto::TileType::Vec, Tdst, bufferSize2, pto::Layout::NCHW,
                                       pto::ConvTileShape<dstTileN, dstTileC, dstTileH, dstTileW>>;
    using tmpAreaTileData = pto::Tile<pto::TileType::Vec, Ttmp, dstTileH * dstTileW, C0, pto::BLayout::RowMajor,
                                      dstTileH * dstTileW, C0>;
    Sync2_VS();
    for (unsigned d = 0; d < dstTileD; d++) {
        __ubuf__ Tsrc* fzSlicePtr = srcPtrOrig + d * srcSliceSize;
        uint32_t burstNum = c1hw_dstTile;
        uint32_t lenBurst = (C0 * sizeof(Tsrc) + pto::BLOCK_BYTE_SIZE - 1) / pto::BLOCK_BYTE_SIZE;
        uint32_t srcGap = (paddedN_dstTile * C0 * sizeof(Tsrc) + pto::BLOCK_BYTE_SIZE - 1) / pto::BLOCK_BYTE_SIZE -
                          lenBurst;
        uint32_t dstGap = 0;
        unsigned nStride = c1hw_dstTile * C0;
        for (unsigned i = 0; i < paddedN_dstTile; i++) {
            __ubuf__ Tsrc* innerSrc = fzSlicePtr + i * C0;
            __ubuf__ Tsrc* innerDst = nc1hwc0Ptr + i * nStride;
            copy_ubuf_to_ubuf(innerDst, innerSrc, 0, burstNum, lenBurst, srcGap, dstGap);
        }

        pipe_barrier(PIPE_V);

        nc1hwc0TileData convNc1hwc0;
        nchwTileData convNchw;
        tmpAreaTileData tmpAreaTile;
        pto::TASSIGN(convNc1hwc0, (uint64_t)nc1hwc0Ptr);
        pto::TASSIGN(convNchw, (uint64_t)nchwPtr);
        pto::TASSIGN(tmpAreaTile, (uint64_t)tmpAreaPtr);
        pto::TTRANS(convNchw, convNc1hwc0, tmpAreaTile);

        pipe_barrier(PIPE_V);

        ConvNCHWPlane2NCDHW<Tdst>(dstPtrOrig, nchwPtr, dstTileN, dstTileC, dstTileD, dstTileH, dstTileW, dstTileC, d);
    }
}

#endif
