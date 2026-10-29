/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv_bp_pto.h
 * \brief Conv Backward TileOp Interface Definition
 */

#ifndef TILEOP_TILE_OPERATOR_CONV_BP_DX_LOAD2D_LOAD3D_PTO__H
#define TILEOP_TILE_OPERATOR_CONV_BP_DX_LOAD2D_LOAD3D_PTO__H

#include "impl/conv_utils.h"

template <bool isConv3D, typename T, typename U>
TILEOP void TLoadConvBpDxDedy(T& dst, U& src, const int64_t& offset0, const int64_t& offset1, const int64_t& offset2,
                              const int64_t& offset3, const int64_t& offset4, const int64_t& shape0,
                              const int64_t& shape1, const int64_t& shape2, const int64_t& shape3,
                              const int64_t& shape4, const int64_t& strideH, const int64_t& strideW,
                              const int64_t& skipH, const int64_t& skipW)
{
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename U::Type);
    // dstAL1 永远是 NC1HWC0 (5维): H=idx2, W=idx3
    constexpr auto stcDstH = Std::tuple_element<CONV_IDX_2, typename T::TileShape>::type::value;
    constexpr auto stcDstW = Std::tuple_element<CONV_IDX_3, typename T::TileShape>::type::value;
    int64_t dstH = GetConvShape<CONV_IDX_2>(dst);
    int64_t dstW = GetConvShape<CONV_IDX_3>(dst);
    constexpr int64_t rowSize = (stcDstH * stcDstW + 16 - 1) / 16 * 16;

    // 2D: GM stride [N, C1, H, W], D 轴补 stride=0; 3D: GM stride [N, D, C1, H, W]
    int64_t srcStride0 = GetConvStride<CONV_IDX_0>(src);
    int64_t srcStride1 = isConv3D ? GetConvStride<CONV_IDX_1>(src) : 0;
    int64_t srcStride2 = GetConvStride<CONV_IDX_1 + (isConv3D ? 1 : 0)>(src);
    int64_t srcStride3 = GetConvStride<CONV_IDX_2 + (isConv3D ? 1 : 0)>(src);
    int64_t srcStride4 = GetConvStride<CONV_IDX_3 + (isConv3D ? 1 : 0)>(src);

    using shapeDim = pto::Shape<-1, -1, -1, -1, -1>;
    using strideDim = pto::Stride<-1, -1, -1, -1, -1>;
    using globalData = pto::GlobalTensor<typename U::Type, shapeDim, strideDim, pto::Layout::NDC1HWC0>;
    constexpr auto bufferSize = stcDstW * BLOCK_ALIGN_BYTE;
    using tileData = pto::ConvTile<pto::TileType::Mat, typename T::Type, bufferSize, pto::Layout::NDC1HWC0,
                                   pto::ConvTileShape<1, 1, 1, 1, 1, c0Size>>;

    constexpr auto stcDstC1 = Std::tuple_element<CONV_IDX_1, typename T::TileShape>::type::value;
    using zeroTileData = pto::Tile<pto::TileType::Mat, typename T::Type, rowSize, stcDstC1 * c0Size,
                                   pto::BLayout::ColMajor, -1, -1, pto::SLayout::RowMajor>;
    zeroTileData zeroTile(0, 0);
    pto::TASSIGN(zeroTile, (uint64_t)dst.GetAddr());
    pto::TFILLPAD(zeroTile, zeroTile);
    pipe_barrier(PIPE_MTE2);

    for (int64_t srcCout1 = 0; srcCout1 < shape2; srcCout1++) {
        for (int64_t srcH = 0; srcH < shape3; srcH++) {
            for (int64_t srcW = 0; srcW < shape4; srcW++) {
                int64_t gmOffset = offset0 * srcStride0 + offset1 * srcStride1 + (offset2 + srcCout1) * srcStride2 +
                                   (offset3 + srcH) * srcStride3 + (offset4 + srcW) * srcStride4;
                int64_t dstStride = srcCout1 * dstH * dstW + strideH * dstW * srcH + strideW * srcW;
                globalData srcGlobal((__gm__ typename U::Type*)(src.GetAddr() + gmOffset), shapeDim(1, 1, 1, 1, 1),
                                     strideDim(srcStride0, srcStride1, srcStride2, srcStride3, srcStride4));
                tileData dstL1;
                pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr() +
                                        (skipH * dstW + skipW + dstStride) * c0Size * sizeof(typename T::Type));
                pto::TLOAD(dstL1, srcGlobal);
            }
        }
    }
}

/**
 * Convbackprop filter(arch32):Copy gradOutput data from Gm to L1
 *
 * dst: AL1[1, C1, HW/16, 16, C0]
 * src: GM(NDC1HWC0)
 * offset0: src_N_offset   shape0: src_N_shape
 * offset1: src_D_offset   shape1: src_D_shape
 * offset2: src_C1_offset  shape2: src_C1_shape
 * offset3: src_H_offset   shape3: src_H_shape
 * offset4: src_W_offset   shape4: src_W_shape
 */
template <bool isConv3D, typename T, typename U>
INLINE void TLoadConvBPDwDedy(T& dst, U& src, const int64_t& offset0, const int64_t& offset1, const int64_t& offset2,
                              const int64_t& offset3, const int64_t& offset4, const int64_t& shape0,
                              const int64_t& shape1, const int64_t& shape2, const int64_t& shape3,
                              const int64_t& shape4)
{
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename U::Type);
    // dstAL1:{1, cout1, howo/16, 16, cout0} stcDstShape3:16
    constexpr auto stcDstShape1 = Std::tuple_element<CONV_IDX_1, typename T::TileShape>::type::value;
    constexpr auto stcDstShape2 = Std::tuple_element<CONV_IDX_2, typename T::TileShape>::type::value;
    constexpr auto stcDstShape3 = Std::tuple_element<CONV_IDX_3, typename T::TileShape>::type::value;
    int64_t dstShape1 = GetConvShape<CONV_IDX_1>(dst);
    int64_t dstShape2 = GetConvShape<CONV_IDX_2>(dst);
    int64_t dstShape3 = GetConvShape<CONV_IDX_3>(dst);
    // 2D: GM [N, C1, H, W, C0], D 轴补 stride=0; 3D: GM stride [N, D, C1, H, W, C0]
    int64_t srcStride0 = GetConvStride<CONV_IDX_0>(src);
    int64_t srcStride1 = isConv3D ? GetConvStride<CONV_IDX_1>(src) : 0;
    int64_t srcStride2 = GetConvStride<CONV_IDX_1 + (isConv3D ? 1 : 0)>(src);
    int64_t srcStride3 = GetConvStride<CONV_IDX_2 + (isConv3D ? 1 : 0)>(src);
    int64_t srcStride4 = GetConvStride<CONV_IDX_3 + (isConv3D ? 1 : 0)>(src);

    using shapeDim = pto::Shape<-1, -1, -1, -1, -1>;
    using strideDim = pto::Stride<-1, -1, -1, -1, -1>;
    using globalData = pto::GlobalTensor<typename U::Type, shapeDim, strideDim, pto::Layout::NDC1HWC0>;

    constexpr auto bufferSize = stcDstShape1 * stcDstShape2 * stcDstShape3 * BLOCK_ALIGN_BYTE;
    using tileData = pto::ConvTile<pto::TileType::Mat, typename T::Type, bufferSize, pto::Layout::NDC1HWC0,
                                   pto::ConvTileShape<1, 1, 1, -1, -1, c0Size>>;

    using zeroTileData = pto::Tile<pto::TileType::Mat, typename T::Type, stcDstShape2 * stcDstShape3,
                                   stcDstShape1 * c0Size, pto::BLayout::ColMajor, -1, -1, pto::SLayout::RowMajor>;
    zeroTileData zeroTile(0, 0);
    pto::TASSIGN(zeroTile, (uint64_t)dst.GetAddr());
    pto::TFILLPAD(zeroTile, zeroTile);
    pipe_barrier(PIPE_MTE2);
    // 跨cout1 offset搬运
    for (int64_t srcCout1 = 0; srcCout1 < shape2; srcCout1++) {
        int64_t gmOffset = offset0 * srcStride0 + offset1 * srcStride1 + (offset2 + srcCout1) * srcStride2 +
                           offset3 * srcStride3 + offset4 * srcStride4;
        int64_t dstStride = srcCout1 * dstShape2 * dstShape3 * BLOCK_ALIGN_BYTE;
        globalData srcGlobal((__gm__ typename U::Type*)(src.GetAddr() + gmOffset), shapeDim(1, 1, 1, shape3, shape4),
                             strideDim(srcStride0, srcStride1, srcStride2, srcStride3, srcStride4));
        tileData dstL1(shape3, shape4);
        pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr() + dstStride);
        pto::TLOAD(dstL1, srcGlobal);
    }
}

template <typename T, typename U>
INLINE void TLoadConvBPNZ(T& dst, U& src, int64_t offset0, int64_t offset1, int64_t offset2, int64_t offset3)
{
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename U::Type);
    constexpr auto stcDstBlockCol = Std::tuple_element<CONV_IDX_0, typename T::TileShape>::type::value;
    constexpr auto stcDstBlockRow = Std::tuple_element<CONV_IDX_1, typename T::TileShape>::type::value;
    constexpr auto stcDstFractalRow = Std::tuple_element<CONV_IDX_2, typename T::TileShape>::type::value;

    int64_t srcStride0 = GetConvStride<CONV_IDX_0>(src);
    int64_t srcStride1 = GetConvStride<CONV_IDX_1>(src);

    int64_t blockCol = GetConvShape<CONV_IDX_0>(dst);
    int64_t blockRow = GetConvShape<CONV_IDX_1>(dst);
    int64_t fractalRow = GetConvShape<CONV_IDX_2>(dst);

    using shapeDim = pto::Shape<1, -1, -1, 16, c0Size>;
    using strideDim = pto::Stride<1, -1, c0Size * 16, c0Size, 1>;
    using globalData = pto::GlobalTensor<typename U::Type, shapeDim, strideDim, pto::Layout::NZ>;
    int64_t gmOffset = offset0 * srcStride0 + offset1 * srcStride1;

    globalData srcGlobal((__gm__ typename U::Type*)(src.GetAddr()) + gmOffset, shapeDim(blockCol, blockRow),
                         strideDim(srcStride0));

    // [blockCol(dst0),blockRow(dst1),fractalRow(dst2),c0]
    // 如果是FZ格式，则为[C1HW,N1,N0,C0]
    constexpr auto stcRow = stcDstBlockRow * stcDstFractalRow;
    constexpr auto stcCol = stcDstBlockCol * c0Size;
    using tileData = pto::Tile<pto::TileType::Mat, typename T::Type, stcRow, stcCol, pto::BLayout::ColMajor, -1, -1,
                               pto::SLayout::RowMajor>;

    tileData dstL1(blockRow * fractalRow, blockCol * c0Size);
    pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
    pto::TLOAD(dstL1, srcGlobal);
}

template <typename T, typename U>
TILEOP void TLoad2DDX(T& dst, U& src, const int64_t& kL0Size, const int64_t& nL0Size, const int64_t& hwk,
                      const int64_t& k0Idx, const int64_t& n0Idx)
{
    // L1: FRACTAL_Z  [Cin1*kh*kw, Cout1, Cout0, Cin0] = [c1hw, n1, n0, c0]
    //     cbuf: c1hw groups × (n1 × c0 columns of n0=16 elem each, ColMajor)
    // L0:  TileRight [K = Cout1*kh*kw*Cout0, N = Cin1*Cin0]
    constexpr auto staticN0 = Std::tuple_element<CONV_IDX_2, typename U::TileShape>::type::value;
    constexpr auto staticC0 = Std::tuple_element<CONV_IDX_3, typename U::TileShape>::type::value;
    int64_t c1hw = GetConvShape<CONV_IDX_0>(src); // Cin1*kh*kw
    int64_t n1 = GetConvShape<CONV_IDX_1>(src);   // Cout1
    int64_t n0 = GetConvShape<CONV_IDX_2>(src);   // Cout0 = 16
    int64_t c0 = GetConvShape<CONV_IDX_3>(src);   // Cin0   32 / dtype
    int64_t kL0 = GetConvShape<CONV_IDX_0>(dst);
    int64_t nL0 = GetConvShape<CONV_IDX_1>(dst);

    int64_t kCout0Num = k0Idx / n0; // L1B 在K方向上小块[n0,c0]的起始Idx
    int64_t kRepeat = kL0Size / n0;
    int64_t nCin0Num = n0Idx / c0;
    int64_t nRepeat = nL0Size / c0;

    int64_t srcB1Offset;
    int64_t dstB0Offset;

    using srcL1 = pto::Tile<pto::TileType::Mat, typename U::Type, staticN0, staticC0, pto::BLayout::ColMajor, -1, -1,
                            pto::SLayout::RowMajor>;

    using dstL0 = pto::TileRight<typename T::Type, staticN0, staticC0, -1, -1>;

    for (int64_t j = 0; j < nRepeat; j++) {
        for (int64_t i = 0; i < kRepeat; i++) {
            int64_t iRev = kCout0Num + i;  // 计算在k方向上是第几个[n0,c0]小块
            int64_t hwRevIdx = iRev % hwk; // 计算在第cout1Idx个N1块上hw个[n0,co]小块中的第几个
            int64_t hwSrcIdx = hwk - hwRevIdx - 1;
            int64_t n1Idx = iRev / hwk;
            srcB1Offset = (nCin0Num + j) * n0 * c0 * n1 * hwk + hwSrcIdx * n0 * c0 * n1 + n1Idx * n0 * c0;
            dstB0Offset = i * n0 * nL0 + j * n0 * c0;
            srcL1 l1(n0, c0);
            dstL0 l0(n0, c0);
            pto::TASSIGN(l1, static_cast<uint64_t>(src.GetAddr()) + srcB1Offset * sizeof(typename U::Type));
            pto::TASSIGN(l0, static_cast<uint64_t>(dst.GetAddr()) + dstB0Offset * sizeof(typename T::Type));

            pto::TEXTRACT<dstL0, srcL1>(l0, l1, 0, 0);
        }
    }
}

template <typename T, typename U>
TILEOP void TLoadDwA(T& dst, U& src, const int64_t& postM, const int64_t& postK)
{
    constexpr auto staticC1 = Std::tuple_element<CONV_IDX_1, typename U::TileShape>::type::value;
    constexpr auto statichw16 = Std::tuple_element<CONV_IDX_2, typename U::TileShape>::type::value;
    constexpr auto staticW = Std::tuple_element<CONV_IDX_3, typename U::TileShape>::type::value;  // 16
    constexpr auto staticC0 = Std::tuple_element<CONV_IDX_4, typename U::TileShape>::type::value; // Cout0
    constexpr auto bufferSize = staticC1 * statichw16 * staticW * staticC0;
    int64_t c1 = GetConvShape<CONV_IDX_1>(src);   // Cout1
    int64_t hw16 = GetConvShape<CONV_IDX_2>(src); // Hout*Wout/16
    int64_t w = GetConvShape<CONV_IDX_3>(src);    // 16
    int64_t c0 = GetConvShape<CONV_IDX_4>(src);   // Cout0

    constexpr auto staticML0 = Std::tuple_element<CONV_IDX_0, typename T::TileShape>::type::value;
    constexpr auto staticKL0 = Std::tuple_element<CONV_IDX_1, typename T::TileShape>::type::value;
    int64_t mL0 = GetConvShape<CONV_IDX_0>(dst);
    int64_t kL0 = GetConvShape<CONV_IDX_1>(dst);

    constexpr auto stcRow = staticC1 * staticC0;
    constexpr auto stcCol = statichw16 * staticW;
    using srcL1 = pto::Tile<pto::TileType::Mat, typename U::Type, stcRow, stcCol, pto::BLayout::RowMajor, -1, -1,
                            pto::SLayout::ColMajor>;
    using dstL0 = pto::TileLeft<typename T::Type, staticML0, staticKL0, -1, -1>;

    srcL1 l1(c1 * c0, hw16 * w);
    dstL0 l0(mL0, kL0);
    pto::TASSIGN(l1, static_cast<uint64_t>(src.GetAddr()));
    pto::TASSIGN(l0, static_cast<uint64_t>(dst.GetAddr()));
    pto::TEXTRACT<dstL0, srcL1>(l0, l1, static_cast<uint16_t>(postM), static_cast<uint16_t>(postK));
}

template <typename DstTensor, typename SrcTensor>
__tf__ INLINE void TImg2ColToL0B(DstTensor& dst, SrcTensor& src, const uint16_t& posM, const uint16_t& posK)
{
    // 1) fmatrix_b 寄存器：fmapH(高16bit)/fmapW(低16bit) + padList(每个 8bit)
    constexpr uint16_t fmapShiftBit = 16;
    constexpr uint16_t padListShiftBit = 8;
    constexpr uint16_t padListShiftBase = 32;
    constexpr uint16_t padNumber = 4;
    uint64_t regFmatrix = 0;
    regFmatrix |= static_cast<uint64_t>(src.GetFmapW() & 0xFFFF);
    regFmatrix |= static_cast<uint64_t>(src.GetFmapH() & 0xFFFF) << fmapShiftBit;
    const uint8_t* padList = src.GetPadListArray();
    for (uint16_t i = 0; i < padNumber; i++) {
        regFmatrix |= static_cast<uint64_t>(padList[i] & 0xFF) << (padListShiftBase + i * padListShiftBit);
    }
    set_fmatrix_b(regFmatrix);

    // 2) l3d_rpt 寄存器
    constexpr uint16_t repeatTimeShiftBit = 16;
    constexpr uint16_t repeatModeShiftBit = 24;
    uint64_t rptConfig = 0;
    rptConfig |= static_cast<uint64_t>(src.GetRepeatStride());
    rptConfig |= static_cast<uint64_t>(src.GetRepeatTime()) << repeatTimeShiftBit;
    rptConfig |= static_cast<uint64_t>(src.GetRepeatMode()) << repeatModeShiftBit;
    set_l3d_rpt(rptConfig);

    // 3) padding 值寄存器
    using SrcDType = typename SrcTensor::DType;
    SrcDType padValue = src.GetPadValue();
    constexpr uint16_t padValueShiftBit = 8;
    uint32_t paddingValue = 0;
    if constexpr (sizeof(SrcDType) == 1) {
        uint8_t u8Value = *reinterpret_cast<const uint8_t*>(&padValue);
        paddingValue = (static_cast<uint16_t>(u8Value) << padValueShiftBit) | u8Value;
    } else if constexpr (sizeof(SrcDType) == 2) {
        paddingValue = *reinterpret_cast<const uint16_t*>(&padValue);
    } else if constexpr (sizeof(SrcDType) == 4) {
        paddingValue = *reinterpret_cast<const uint32_t*>(&padValue);
    }
    set_padding(paddingValue);

    // 4) stepM/stepK
    constexpr int32_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename DstTensor::DType);
    uint16_t stepM = static_cast<uint16_t>(dst.GetValidRow());
    uint16_t stepK = static_cast<uint16_t>((dst.GetValidCol() + c0Size - 1) / c0Size * c0Size);

    // 5) filter 高低位拆分
    uint16_t filterW = src.GetFilterW();
    uint16_t filterH = src.GetFilterH();
    uint8_t lowFilterW = filterW & 0xFF;
    uint8_t lowFilterH = filterH & 0xFF;
    bool highFilterW = (filterW > 255);
    bool highFilterH = (filterH > 255);
    constexpr bool fmatrixCtrl = true; // FMATRIX_B_AUTO

    // 6) 地址 + 下发指令 (注意 stepK/stepM、posK/posM 的顺序与 TIMG2COL 接口相反)
    using SrcType = std::conditional_t<(sizeof(SrcDType) == 2), half, SrcDType>;
    using DstType = std::conditional_t<(sizeof(typename DstTensor::DType) == 2), half, typename DstTensor::DType>;
    __cbuf__ SrcType* srcAddr = (__cbuf__ SrcType*)__cce_get_tile_ptr(src.data());
    __cb__ DstType* dstAddr = (__cb__ DstType*)__cce_get_tile_ptr(dst.data());
    img2colv2_cbuf_to_cb(dstAddr, srcAddr, stepK, stepM, posK, posM, src.GetStrideW(), src.GetStrideH(), lowFilterW,
                         lowFilterH, src.GetDilationW(), src.GetDilationH(), highFilterW, highFilterH,
                         src.GetTranspose(), fmatrixCtrl, src.GetChannelSize());
}

template <bool isConv3D, typename T, typename U>
TILEOP void TLoad3DDW(T& dst, U& src, const int64_t& mPos, const int64_t& kPos, const int64_t& padLeft,
                      const int64_t& padRight, const int64_t& padTop, const int64_t& padBottom, const int64_t& padValue,
                      const int64_t& filterH, const int64_t& filterW, const int64_t& dilationH,
                      const int64_t& dilationW, const int64_t& strideH, const int64_t& strideW,
                      const int64_t& repeatStride, const int64_t& repeatTime, const int64_t& dstStride)
{
    // 2D： n c1 h w c0 ; 3D： n d c1 h w c0
    constexpr auto static0 = Std::tuple_element<CONV_IDX_0, typename U::TileShape>::type::value;
    constexpr auto static1 = Std::tuple_element<CONV_IDX_1, typename U::TileShape>::type::value;
    constexpr auto static2 = Std::tuple_element<CONV_IDX_2, typename U::TileShape>::type::value;
    constexpr auto static3 = Std::tuple_element<CONV_IDX_3, typename U::TileShape>::type::value;
    constexpr auto static4 = Std::tuple_element<CONV_IDX_4, typename U::TileShape>::type::value;
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename U::Type);
    constexpr auto elements = static0 * static1 * static2 * static3 * static4;
    int64_t shape0 = GetConvShape<CONV_IDX_0>(src);
    int64_t shape1 = GetConvShape<CONV_IDX_1>(src);
    int64_t shape2 = GetConvShape<CONV_IDX_2>(src);
    int64_t shape3 = GetConvShape<CONV_IDX_3>(src);
    int64_t shape4 = GetConvShape<CONV_IDX_4>(src);
    using srcTensor = select_srcTensor<isConv3D, U, elements, c0Size>;
    srcTensor l1(shape0, shape1, shape2, shape3, shape4);

    constexpr auto staticML0 = Std::tuple_element<CONV_IDX_0, typename T::TileShape>::type::value;
    constexpr auto staticKL0 = Std::tuple_element<CONV_IDX_1, typename T::TileShape>::type::value;
    int64_t mL0 = GetConvShape<CONV_IDX_0>(dst);
    int64_t kL0 = GetConvShape<CONV_IDX_1>(dst);
    using dstTensor = pto::TileRight<typename T::Type, staticML0, staticKL0, -1, -1>;
    dstTensor l0(dstStride, kL0);

    uint8_t values[4] = {static_cast<uint8_t>(padLeft), static_cast<uint8_t>(padRight), static_cast<uint8_t>(padTop),
                         static_cast<uint8_t>(padBottom)};
    SetConvTileParams<isConv3D>(l1, values, filterH, filterW, dilationH, dilationW, strideH, strideW, padValue, mL0,
                                kL0, shape1, shape2, shape3, shape4, c0Size, repeatStride, repeatTime, dstStride);

    pto::TASSIGN(l1, static_cast<uint64_t>(src.GetAddr()));
    pto::TASSIGN(l0, static_cast<uint64_t>(dst.GetAddr()));
    TImg2ColToL0B(l0, l1, static_cast<uint16_t>(mPos), static_cast<uint16_t>(kPos));
}

template <CopyOutMode mode, bool isConv3D, typename T, typename U>
TILEOP void TStoreConvDW(T& dst, U& src, const int64_t& offset0, const int64_t& offset1, const int64_t& offset2,
                         const int64_t& offset3, const int64_t& offset4, const int64_t& realM, const int64_t& realN)
{
    constexpr auto srcM = Std::tuple_element<CONV_IDX_0, typename U::TileShape>::type::value;
    constexpr auto srcN = Std::tuple_element<CONV_IDX_1, typename U::TileShape>::type::value;
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename T::Type);
    (void)offset2;
    (void)offset3;
    (void)offset4;

    int64_t dstStride0 = GetConvStride<CONV_IDX_0>(dst); // Cin1*Kh*Kw
    int64_t dstStride1 = GetConvStride<CONV_IDX_1>(dst); // Cout1
    int64_t blockCol = GetConvShape<CONV_IDX_0>(dst);    // Cin1*Kh*Kw
    int64_t blockRow = GetConvShape<CONV_IDX_1>(dst);    // Cout1

    using shapeDim = pto::Shape<1, -1, -1, 16, c0Size>;
    using strideDim = pto::Stride<1, -1, c0Size * 16, c0Size, 1>;
    using globalData = pto::GlobalTensor<typename T::Type, shapeDim, strideDim, pto::Layout::NZ>;

    int64_t gmOffset = offset0 * dstStride0 + offset1 * dstStride1;
    globalData dstGlobal((__gm__ typename T::Type*)(dst.GetAddr() + gmOffset), shapeDim(blockCol, blockRow),
                         strideDim(dstStride0));

    using tileData = pto::Tile<pto::TileType::Acc, typename U::Type, srcM, srcN, pto::BLayout::ColMajor, -1, -1,
                               pto::SLayout::RowMajor, pto::TileConfig::fractalCSize, pto::PadValue::Null,
                               pto::CompactMode::Null>;
    tileData srcL0C(realM, realN);
    pto::TASSIGN(srcL0C, static_cast<uint64_t>(src.GetAddr()));
    // DW搬出不带激活，固定NoRelu
    pto::TSTORE<tileData, globalData, pto::AtomicType::AtomicNone, pto::ReluPreMode::NoRelu>(dstGlobal, srcL0C);
}
#endif // TILEOP_TILE_OPERATOR_CONV_BP_DX_LOAD2D_LOAD3D_PTO__H
