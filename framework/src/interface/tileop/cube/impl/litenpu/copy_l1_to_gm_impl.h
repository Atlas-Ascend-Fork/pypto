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
 * \file copy_l1_to_gm_impl.h
 * \brief L1 to DDR Data Movement Interface Implementation
 */

#ifndef TILEOP_TILE_OPERATOR_COPY_L1_TO_GM_IMPL__H
#define TILEOP_TILE_OPERATOR_COPY_L1_TO_GM_IMPL__H

#include "../cube_utils.h"

// kirin9050(DAV_5101) 无 L0C：cube 输出驻留 L1（NZ Mat）。直接 L1(NZ) -> GM(ND) 落盘
// （pto-isa TSTORE(Mat) 直连，省去 L1->UB->GM 两步搬运）。
// 仅支持 2D ND global：Shape<1,1,1,validRow,validCol>，validRow/Col 取自源 tile 的有效形状。
template <typename config, typename GlobalData, typename TileData>
INLINE void TStoreL1MatNz2Nd(GlobalData& dst, TileData& src, const int64_t& offset0, const int64_t& offset1)
{
    static_assert(config::kMode == CopyOutMode::NZ2ND, "kirin9050 L1->GM only supports NZ2ND.");
    constexpr uint64_t shapeSize = Std::tuple_size<typename TileData::Shape>::value;
    constexpr auto staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename TileData::TileShape>::type::value;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - 1, typename TileData::TileShape>::type::value;
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    int64_t dstStride0 = GetStride<0>(dst);
    int64_t dstStride1 = GetStride<1>(dst);
    using shapeDim2 = pto::Shape<1, 1, 1, -1, -1>;
    using strideDim2 = pto::Stride<1, 1, 1, -1, -1>;
    int64_t gmOffset = offset1 + offset0 * dstStride0;
    using globalData = pto::GlobalTensor<typename GlobalData::Type, shapeDim2, strideDim2, pto::Layout::ND>;
    // 源为 L1 中 mmad 直接写出的 NZ Mat（无 L0C，TSTORE(Mat) 要求 NZ 分形）。
    // 分形宽度与硬件 C 布局一致（内列 = max(16, 32B/sizeof)）：1B 类型用 AB 分形（32），
    // 2B/4B 用 C 分形（16）。指令侧按 TileData::InnerCols 取宽度。
    constexpr int kSFractalSize = (sizeof(typename TileData::Type) == 1) ? pto::TileConfig::fractalABSize :
                                                                           pto::TileConfig::fractalCSize;
    using tileData = pto::Tile<pto::TileType::Mat, typename TileData::Type, staticL1H, staticL1W,
                               pto::BLayout::ColMajor, -1, -1, pto::SLayout::RowMajor, kSFractalSize>;
    globalData dstGlobal((__gm__ typename GlobalData::Type*)(dst.GetAddr() + gmOffset), shapeDim2(srcShape0, srcShape1),
                         strideDim2(dstStride0, dstStride1));
    tileData srcL1(srcShape0, srcShape1);
    pto::TASSIGN(srcL1, static_cast<uint64_t>(src.GetAddr()));
    pto::TSTORE<tileData, globalData>(dstGlobal, srcL1);
}

#endif // TILEOP_TILE_OPERATOR_COPY_L1_TO_GM_IMPL__H
