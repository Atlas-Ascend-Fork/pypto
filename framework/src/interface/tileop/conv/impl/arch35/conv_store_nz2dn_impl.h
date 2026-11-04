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
 * \file conv_store_nz2dn_impl.h
 * \brief Copy data from L0C to DDR/UB with NZ2DN, NZ -> NCHW/NCDHW (Ascend 950PR/Ascend 950DT)
 */

#ifndef TILEOP_TILE_OPERATOR_CONV_STORE_NZ2DN_IMPL__H
#define TILEOP_TILE_OPERATOR_CONV_STORE_NZ2DN_IMPL__H
#include "conv_offset_utils.h"

/**
 * Copy data from L0C to DDR with NZ -> NCHW format.
 * dst: GM(NCHW)
 * src: l0c(NZ)
 * offset0: dst_n_offset
 * offset1: dst_c_offset
 * offset2: dst_h_offset (conv2d 参数顺序调整：n, c, h, w, 0)
 * offset3: dst_w_offset
 * offset4: 0 (占位)
 */
template <int64_t reluType, typename T, typename U>
INLINE void TStoreConv2DNZ2DN(T& dst, U& src, const OffsetInfo& offsetInfo, const int64_t& realM, const int64_t& realN,
                              const int64_t& realCutW, const int64_t& cutW)
{
    constexpr auto srcM = Std::tuple_element<CONV_IDX_0, typename U::TileShape>::type::value;
    constexpr auto srcN = Std::tuple_element<CONV_IDX_1, typename U::TileShape>::type::value;
    int64_t dstN = GetConvShape<CONV_IDX_0>(dst);
    int64_t dstC = GetConvShape<CONV_IDX_1>(dst);
    int64_t dstH = GetConvShape<CONV_IDX_2>(dst);
    int64_t dstW = GetConvShape<CONV_IDX_3>(dst);
    int64_t dstStrideN = GetConvStride<CONV_IDX_0>(dst);
    int64_t dstStrideC = GetConvStride<CONV_IDX_1>(dst);
    int64_t dstStrideH = GetConvStride<CONV_IDX_2>(dst);
    int64_t dstStrideW = GetConvStride<CONV_IDX_3>(dst);
    int64_t gmOffset = offsetInfo.offset0 * dstStrideN + offsetInfo.offset1 * dstStrideC + offsetInfo.offset2 * dstW +
                       offsetInfo.offset3;

    using shapeDim4 = pto::Shape<1, -1, -1, -1, -1>;
    using strideDim4 = pto::Stride<1, -1, -1, -1, -1>;
    using tileData = pto::Tile<pto::TileType::Acc, typename U::Type, srcM, srcN, pto::BLayout::ColMajor, -1, -1,
                               pto::SLayout::RowMajor, pto::TileConfig::fractalCSize, pto::PadValue::Null,
                               pto::CompactMode::Null>;
    using globalData = pto::GlobalTensor<typename T::Type, shapeDim4, strideDim4, pto::Layout::NCHW>;
    // 分块搬出，每次搬出cutW大小的数据
    for (int64_t loopH = 0; realCutW > 0 && loopH < (realM / realCutW); loopH++) {
        globalData dstGlobal((__gm__ typename T::Type*)(dst.GetAddr() + gmOffset), shapeDim4(dstN, dstC, dstH, dstW),
                             strideDim4(dstStrideN, dstStrideC, dstStrideH, dstStrideW));
        tileData srcL0C(realCutW, realN);
        pto::TASSIGN(srcL0C, (uint64_t)src.GetAddr() + loopH * cutW * BLOCK_CUBE_M_N * sizeof(typename U::Type));
        pto::TSTORE<tileData, globalData, pto::AtomicType::AtomicNone,
                    reluType == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(dstGlobal, srcL0C);
        gmOffset += dstW;
    }
}

/**
 * Copy data from L0C to DDR with NZ -> NCDHW format.
 * dst: GM(NCDHW)
 * src: l0c(NZ)
 * offset0: dst_n_offset
 * offset1: dst_c_offset
 * offset2: dst_d_offset
 * offset3: dst_h_offset
 * offset4: dst_w_offset
 */
template <int64_t reluType, typename T, typename U>
INLINE void TStoreConv3DNZ2DN(T& dst, U& src, const OffsetInfo& offsetInfo, const int64_t& realM, const int64_t& realN,
                              const int64_t& realCutW, const int64_t& cutW)
{
    constexpr auto srcM = Std::tuple_element<CONV_IDX_0, typename U::TileShape>::type::value;
    constexpr auto srcN = Std::tuple_element<CONV_IDX_1, typename U::TileShape>::type::value;
    int64_t dstN = GetConvShape<CONV_IDX_0>(dst);
    int64_t dstC = GetConvShape<CONV_IDX_1>(dst);
    int64_t dstD = GetConvShape<CONV_IDX_2>(dst);
    int64_t dstH = GetConvShape<CONV_IDX_3>(dst);
    int64_t dstW = GetConvShape<CONV_IDX_4>(dst);
    int64_t dstStrideN = GetConvStride<CONV_IDX_0>(dst);
    int64_t dstStrideC = GetConvStride<CONV_IDX_1>(dst);
    int64_t dstStrideD = GetConvStride<CONV_IDX_2>(dst);
    int64_t dstStrideH = GetConvStride<CONV_IDX_3>(dst);
    int64_t dstStrideW = GetConvStride<CONV_IDX_4>(dst);
    ShapeInfo shapeInfo{dstC, dstD, dstH, dstW};
    using shapeDim5 = pto::Shape<1, -1, -1, -1, -1>;
    using strideDim5 = pto::Stride<-1, -1, -1, -1, -1>;
    using tileData = pto::Tile<pto::TileType::Acc, typename U::Type, srcM, srcN, pto::BLayout::ColMajor, -1, -1,
                               pto::SLayout::RowMajor, pto::TileConfig::fractalCSize, pto::PadValue::Null,
                               pto::CompactMode::Null>;
    using globalData = pto::GlobalTensor<typename T::Type, shapeDim5, strideDim5, pto::Layout::NCDHW>;
    // 分块搬出，每次搬出cutW大小的数据
    for (int64_t loopH = 0; realCutW > 0 && loopH < (realM / realCutW); loopH++) {
        int64_t gmOffset = CalStoreOffsetNCDHW(shapeInfo, offsetInfo, loopH);
        globalData dstGlobal((__gm__ typename T::Type*)(dst.GetAddr() + gmOffset),
                             shapeDim5(dstN, dstC, dstD, dstH, dstW),
                             strideDim5(dstStrideN, dstStrideC, dstStrideD, dstStrideH, dstStrideW));
        tileData srcL0C(realCutW, realN);
        pto::TASSIGN(srcL0C, (uint64_t)src.GetAddr() + loopH * cutW * BLOCK_CUBE_M_N * sizeof(typename U::Type));
        pto::TSTORE<tileData, globalData, pto::AtomicType::AtomicNone,
                    reluType == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(dstGlobal, srcL0C);
    }
}

template <bool isConv3D, int64_t reluType, typename T, typename U>
INLINE void TStoreConvNZ2DN(T& dst, U& src, const OffsetInfo& offsetInfo, const int64_t& realM, const int64_t& realN,
                            const int64_t& realCutW, const int64_t& cutW)
{
    if constexpr (isConv3D) {
        TStoreConv3DNZ2DN<reluType>(dst, src, offsetInfo, realM, realN, realCutW, cutW);
    } else {
        TStoreConv2DNZ2DN<reluType>(dst, src, offsetInfo, realM, realN, realCutW, cutW);
    }
}

#if defined(PTO_NPU_ARCH_A5)
// fixpipe L0C(NZ) -> UB(NCHW/NCDHW) 散列搬运所需的量化模式，语义与 GM 侧 TStore 一致
template <typename SrcType, typename DstType>
INLINE constexpr uint64_t GetConvL0C2UBQuantPre()
{
    if constexpr (std::is_same<SrcType, float>::value) {
        if constexpr (std::is_same<DstType, half>::value) {
            return QuantMode_t::F322F16;
        } else if constexpr (std::is_same<DstType, bfloat16_t>::value) {
            return QuantMode_t::F322BF16;
        } else {
            return QuantMode_t::NoQuant;
        }
    } else {
        return QuantMode_t::NoQuant;
    }
}

// nz2dn 散列模式的 loop3/channel 寄存器配置，与 GM 侧 TStoreAccNCHW/TMov DN 路径一致
INLINE void SetConvL0C2UBNdPara()
{
    constexpr uint16_t ndNum = 1;
    constexpr uint16_t dstNdStride = 0;
    constexpr uint16_t srcNdStride = 0;
    constexpr uint64_t loop3Para = static_cast<uint64_t>(dstNdStride) << 32 | static_cast<uint64_t>(srcNdStride) << 16 |
                                   static_cast<uint64_t>(ndNum);
    set_loop3_para(loop3Para);
    // CHANNEL_PARA[63:48]: loop0 源间隔，NZ 源为 1 个 C0 单位
    constexpr uint64_t channelPara = static_cast<uint64_t>(1) << 48;
    set_channel_para(channelPara);
}

/**
 * L0C NZ 分形布局 ([N/16][M/16][16][16]) 下源侧 (m, n) 坐标的线性元素偏移:
 * m 行 -> m*16 (行内一个 C0); n 列 -> (n/16)*srcM*16 (整列分形块, srcM 为 tile 静态 M) + n%16 (块内列偏移)
 */
template <typename U>
INLINE int64_t GetConvL0CSrcOffsetElems(const int64_t& srcOffsetM, const int64_t& srcOffsetN)
{
    constexpr auto srcM = Std::tuple_element<CONV_IDX_0, typename U::TileShape>::type::value;
    return srcOffsetM * BLOCK_CUBE_M_N + srcOffsetN / BLOCK_CUBE_M_N * srcM * BLOCK_CUBE_M_N +
           srcOffsetN % BLOCK_CUBE_M_N;
}

/**
 * Copy data from L0C to UB with NZ -> NCHW format, dst 布局与 TStoreConv2DNZ2DN 的 GM 目的端一致.
 * dst: UB(NCHW)
 * src: l0c(NZ)
 * offset0: dst_n_offset
 * offset1: dst_c_offset
 * offset2: dst_h_offset
 * offset3: dst_w_offset
 * offset4: 0 (占位)
 * srcOffsetM/srcOffsetN: L0C 源侧 (m, n) 偏移
 */
template <int64_t reluType, typename T, typename U>
INLINE void TCopyL0C2UBConv2DNZ2DN(T& dst, U& src, const OffsetInfo& offsetInfo, const int64_t& realM,
                                   const int64_t& realN, const int64_t& realCutW, const int64_t& cutW,
                                   const int64_t& srcOffsetM, const int64_t& srcOffsetN, const int64_t& subBlockId)
{
    constexpr auto srcM = Std::tuple_element<CONV_IDX_0, typename U::TileShape>::type::value;
    constexpr auto srcN = Std::tuple_element<CONV_IDX_1, typename U::TileShape>::type::value;
    constexpr auto dstN = Std::tuple_element<CONV_IDX_1, typename T::TileShape>::type::value;
    constexpr auto dstW = Std::tuple_element<CONV_IDX_3, typename T::TileShape>::type::value;
    int64_t dstStrideN = GetConvStride<CONV_IDX_0>(dst);
    int64_t dstStrideC = GetConvStride<CONV_IDX_1>(dst);
    // 行距取布局 stride (静态) 而非 shape 的 W (动态 valid): 动态 shape 仅作界,
    // copy 行基址按静态行距 32B 对齐; 若用 runtime shape W 做行进位, W 尾块时基址未对齐
    int64_t dstStrideH = GetConvStride<CONV_IDX_2>(dst);
    int64_t ubOffset = offsetInfo.offset0 * dstStrideN + offsetInfo.offset1 * dstStrideC +
                       offsetInfo.offset2 * dstStrideH + offsetInfo.offset3;
    constexpr uint16_t srcStride = static_cast<uint16_t>(srcM);
    // n 方向搬运大小: 大搬小 (srcN > dstN) 按 dst 的 N 搬出, 防止越界写相邻 UB buffer;
    // 小搬大 (srcN <= dstN) 按搬运需要的源大小 srcN 搬出; 两者均为 C0 对齐值,
    // 非 C0 倍数的 realN (cout 尾块) 不能直接使用
    constexpr uint16_t nSize = static_cast<uint16_t>(srcN < dstN ? srcN : dstN);
    // m 方向 (W) 同理: 大搬小 (cutW > dstW) 按 dst 的 W 搬出, 防止溢出到后续行/相邻
    // UB buffer; 小搬大 (cutW <= dstW) 按搬运需要的源行宽 cutW 搬出; mix 路径已保证
    // dstW 16 对齐 (末维对齐校验), cutW 亦为 16 的倍数, min 结果保持 32B 行对齐;
    // cutW 为运行时参数, mSize 不能为 constexpr
    const uint16_t mSize = static_cast<uint16_t>(cutW < dstW ? cutW : dstW);
    uint32_t dstStride = static_cast<uint32_t>(dstStrideC);
    uint64_t quantPre = GetConvL0C2UBQuantPre<typename U::Type, typename T::Type>();
    uint8_t reluPre = reluType == 0 ? static_cast<uint8_t>(pto::ReluPreMode::NoRelu) :
                                      static_cast<uint8_t>(pto::ReluPreMode::NormalRelu);
    SetConvL0C2UBNdPara();
    int64_t srcBaseOffset = GetConvL0CSrcOffsetElems<U>(srcOffsetM, srcOffsetN);

    // 分块搬出，每次搬出cutW大小的数据；循环次数为有效窗口行数 (realM/realCutW=validH)，
    // 防止大搬小 (一个 L0C 块跨多个 UB tile) 时越界；单次搬出按整分形宽 (nSize×cutW)，
    // 保证 32B 对齐写入 (realCutW<16 的部分行写在跨核可见性上有问题)
    for (int64_t loopH = 0; realCutW > 0 && loopH < (realM / realCutW); loopH++) {
        __ubuf__ typename T::Type* dstAddr = (__ubuf__
                                              typename T::Type*)(dst.GetAddr() + ubOffset * sizeof(typename T::Type));
        __cc__ typename U::Type* srcAddr = (__cc__ typename U::Type*)(src.GetAddr() +
                                                                      (srcBaseOffset + loopH * cutW * BLOCK_CUBE_M_N) *
                                                                          sizeof(typename U::Type));
        copy_matrix_cc_to_ub(dstAddr, srcAddr, 0, nSize, mSize, dstStride, srcStride, 0, subBlockId, 0, 0, quantPre,
                             reluPre, false, false, 0, 0, false, false, 0, false, false, false, false, false, true);
        ubOffset += dstStrideH;
    }
}

/**
 * Copy data from L0C to UB with NZ -> NCDHW format, dst 布局与 TStoreConv3DNZ2DN 的 GM 目的端一致.
 * dst: UB(NCDHW)
 * src: l0c(NZ)
 * offset0: dst_n_offset
 * offset1: dst_c_offset
 * offset2: dst_d_offset
 * offset3: dst_h_offset
 * offset4: dst_w_offset
 * srcOffsetM/srcOffsetN: L0C 源侧 (m, n) 偏移
 */
template <int64_t reluType, typename T, typename U>
INLINE void TCopyL0C2UBConv3DNZ2DN(T& dst, U& src, const OffsetInfo& offsetInfo, const int64_t& realM,
                                   const int64_t& realN, const int64_t& realCutW, const int64_t& cutW,
                                   const int64_t& srcOffsetM, const int64_t& srcOffsetN, const int64_t& subBlockId)
{
    constexpr auto srcM = Std::tuple_element<CONV_IDX_0, typename U::TileShape>::type::value;
    constexpr auto srcN = Std::tuple_element<CONV_IDX_1, typename U::TileShape>::type::value;
    constexpr auto dstN = Std::tuple_element<CONV_IDX_1, typename T::TileShape>::type::value;
    constexpr auto dstTileW = Std::tuple_element<CONV_IDX_4, typename T::TileShape>::type::value;
    int64_t dstStrideC = GetConvStride<CONV_IDX_1>(dst);
    constexpr uint16_t srcStride = static_cast<uint16_t>(srcM);
    // n 方向搬运大小: 大搬小 (srcN > dstN) 按 dst 的 N 搬出, 防止越界写相邻 UB buffer;
    // 小搬大 (srcN <= dstN) 按搬运需要的源大小 srcN 搬出; 两者均为 C0 对齐值,
    // 非 C0 倍数的 realN (cout 尾块) 不能直接使用
    constexpr uint16_t nSize = static_cast<uint16_t>(srcN < dstN ? srcN : dstN);
    // m 方向 (W) 同理: 大搬小按 dst tile 静态 W, 小搬大按源行宽 cutW; 取静态 TileShape
    // 的 W (运行时 dstW 可能携带 valid 值), cutW 为运行时参数, mSize 不能为 constexpr
    const uint16_t mSize = static_cast<uint16_t>(cutW < dstTileW ? cutW : dstTileW);
    uint32_t dstStride = static_cast<uint32_t>(dstStrideC);
    uint64_t quantPre = GetConvL0C2UBQuantPre<typename U::Type, typename T::Type>();
    uint8_t reluPre = reluType == 0 ? static_cast<uint8_t>(pto::ReluPreMode::NoRelu) :
                                      static_cast<uint8_t>(pto::ReluPreMode::NormalRelu);
    SetConvL0C2UBNdPara();
    int64_t srcBaseOffset = GetConvL0CSrcOffsetElems<U>(srcOffsetM, srcOffsetN);
    // UB 目的偏移按静态布局 stride 计算 (与 UB tile 物理布局及向量侧读取一致), 与 2D 路径
    // 同策略: 运行时 shape 可能携带 valid 值 (W 尾块), 用作行进位会把行距压缩为 valid W,
    // 与传给 copy_matrix_cc_to_ub 的静态 dstStride 不自洽; GM 侧 (TStoreConv3DNZ2DN) 用
    // shape 当 stride 正确 (GM 稠密布局 stride 即 shape), UB 侧须用静态 stride
    int64_t dstStrideN = GetConvStride<CONV_IDX_0>(dst);
    int64_t dstStrideD = GetConvStride<CONV_IDX_2>(dst);
    int64_t dstStrideH = GetConvStride<CONV_IDX_3>(dst);
    int64_t ubOffset = offsetInfo.offset0 * dstStrideN + offsetInfo.offset1 * dstStrideC +
                       offsetInfo.offset2 * dstStrideD + offsetInfo.offset3 * dstStrideH + offsetInfo.offset4;

    // 分块搬出，每次搬出cutW大小的数据；循环次数为有效窗口行数 (realM/realCutW=validH)，
    // 防止大搬小 (一个 L0C 块跨多个 UB tile) 时越界；单次搬出按整分形宽 (nSize×cutW)，
    // 保证 32B 对齐写入 (realCutW<16 的部分行写在跨核可见性上有问题)
    for (int64_t loopH = 0; realCutW > 0 && loopH < (realM / realCutW); loopH++) {
        __ubuf__ typename T::Type* dstAddr = (__ubuf__
                                              typename T::Type*)(dst.GetAddr() + ubOffset * sizeof(typename T::Type));
        __cc__ typename U::Type* srcAddr = (__cc__ typename U::Type*)(src.GetAddr() +
                                                                      (srcBaseOffset + loopH * cutW * BLOCK_CUBE_M_N) *
                                                                          sizeof(typename U::Type));
        copy_matrix_cc_to_ub(dstAddr, srcAddr, 0, nSize, mSize, dstStride, srcStride, 0, subBlockId, 0, 0, quantPre,
                             reluPre, false, false, 0, 0, false, false, 0, false, false, false, false, false, true);
        ubOffset += dstStrideH;
    }
}

template <bool isConv3D, int64_t reluType, typename T, typename U>
INLINE void TCopyL0C2UBConvNZ2DN(T& dst, U& src, const OffsetInfo& offsetInfo, const int64_t& realM,
                                 const int64_t& realN, const int64_t& realCutW, const int64_t& cutW,
                                 const int64_t& srcOffsetM, const int64_t& srcOffsetN, const int64_t& subBlockId)
{
    if constexpr (isConv3D) {
        TCopyL0C2UBConv3DNZ2DN<reluType>(dst, src, offsetInfo, realM, realN, realCutW, cutW, srcOffsetM, srcOffsetN,
                                         subBlockId);
    } else {
        TCopyL0C2UBConv2DNZ2DN<reluType>(dst, src, offsetInfo, realM, realN, realCutW, cutW, srcOffsetM, srcOffsetN,
                                         subBlockId);
    }
}
#endif // defined(PTO_NPU_ARCH_A5)

#endif // TILEOP_TILE_OPERATOR_CONV_STORE_NZ2DN_IMPL__H
