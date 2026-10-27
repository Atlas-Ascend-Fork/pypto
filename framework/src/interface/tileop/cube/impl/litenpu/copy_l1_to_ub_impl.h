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
 * \file copy_l1_to_ub_impl.h
 * \brief L1 to UB Data Transfer Interface Implementation (LiteNPU)
 */

#ifndef TILEOP_TILE_OPERATOR_LITENPU_COPY_L1_TO_UB_IMPL__H
#define TILEOP_TILE_OPERATOR_LITENPU_COPY_L1_TO_UB_IMPL__H

#include "../cube_utils.h"

template <typename ubTileData, typename l1TileData>
TILEOP void TExtractL1ToUB(ubTileData& ubTile, l1TileData& l1Tile, uint16_t l1Offset0, uint16_t l1Offset1)
{
    pto::TEXTRACT(ubTile, l1Tile, l1Offset0, l1Offset1);
}

template <typename ubTileData, typename l1TileData>
TILEOP void TInsertL1ToUB(ubTileData& ubTile, l1TileData& l1Tile, uint16_t ubOffset0, uint16_t ubOffset1)
{
    pto::TINSERT(ubTile, l1Tile, ubOffset0, ubOffset1);
}

// Copy data from L1 to UB
template <CopyMode mode, typename Coord, typename DstTileTensor, typename SrcTileTensor>
INLINE void TCopyL12UBImpl(DstTileTensor& dst, SrcTileTensor& src, const Coord& dstCoord, const Coord& srcCoord)
{
    static_assert(mode != CopyMode::UNKNOWN,
                  "[TCopyL12UB Error]: Current CopyMode is UNKNOWN. CopyMode only support EXTRACT, INSERT and MOVE");
    constexpr uint64_t shapeSize = Std::tuple_size<typename DstTileTensor::Shape>::value;
    int64_t dstOffset0 = TileOp::GetTupleElement<Coord, 0, SHAPE_DIM2, 0>(dstCoord);
    int64_t dstOffset1 = TileOp::GetTupleElement<Coord, 1, SHAPE_DIM2, 0>(dstCoord);
    int64_t srcOffset0 = TileOp::GetTupleElement<Coord, 0, SHAPE_DIM2, 0>(srcCoord);
    int64_t srcOffset1 = TileOp::GetTupleElement<Coord, 1, SHAPE_DIM2, 0>(srcCoord);
    constexpr auto
        staticUBH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename DstTileTensor::TileShape>::type::value;
    constexpr auto staticUBW = Std::tuple_element<shapeSize - 1, typename DstTileTensor::TileShape>::type::value;
    constexpr auto
        staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename SrcTileTensor::TileShape>::type::value;
    constexpr auto staticL1W = Std::tuple_element<shapeSize - 1, typename SrcTileTensor::TileShape>::type::value;
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    // L1->UB 仅支持 NZ2ND：L1 源为 mmad 直接写出的 NZ Mat，UB 目的为 ND
    using ubTileData = pto::Tile<pto::TileType::Vec, typename DstTileTensor::Type, staticUBH, staticUBW,
                                 pto::BLayout::RowMajor, -1, -1, pto::SLayout::NoneBox>;
    constexpr int kSFractalSize = (sizeof(typename SrcTileTensor::Type) == 1) ? pto::TileConfig::fractalABSize :
                                                                                pto::TileConfig::fractalCSize;
    using l1TileData = pto::Tile<pto::TileType::Mat, typename SrcTileTensor::Type, staticL1H, staticL1W,
                                 pto::BLayout::ColMajor, -1, -1, pto::SLayout::RowMajor, kSFractalSize>;
    ubTileData ubTile(dstShape0, dstShape1);
    l1TileData l1Tile(srcShape0, srcShape1);
    pto::TASSIGN(ubTile, (uint64_t)dst.GetAddr());
    pto::TASSIGN(l1Tile, (uint64_t)src.GetAddr());
    if constexpr (mode == CopyMode::EXTRACT || mode == CopyMode::MOVE) {
        TExtractL1ToUB<ubTileData, l1TileData>(ubTile, l1Tile, srcOffset0, srcOffset1);
    } else if (mode == CopyMode::INSERT) {
        TInsertL1ToUB<ubTileData, l1TileData>(ubTile, l1Tile, dstOffset0, dstOffset1);
    }
}
#endif // TILEOP_TILE_OPERATOR_LITENPU_COPY_L1_TO_UB_IMPL__H
