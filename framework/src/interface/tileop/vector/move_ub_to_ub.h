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
 * \file move_ub_to_ub.h
 * \brief UB 到 UB 数据搬运 Tile 算子
 *
 * 通过 pto-isa 的 TMOV（TMovVecToVec）在 UB 内完成数据拷贝。
 * TMovVecToVec 仅支持 2 维（Rows x Cols）Tile，因此外层按照 5 维默认表示法
 * （n0/n1/n2 为外层维度，n3/n4 为 Tile 高宽）逐块搬运；当输入不足 5 维时，
 * 外层缺失维度自动视为 1。
 */

#ifndef TILEOP_TILE_OPERATOR_MOVE_UB_TO_UB__H
#define TILEOP_TILE_OPERATOR_MOVE_UB_TO_UB__H

#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#define OP_TILE_OP_MOVE_UB_TO_UB TMoveUbToUb

template <typename T0, typename T1>
TILEOP void TMoveUbToUb(T0 dst, T1 src)
{
    using DstTileData = typename PtoTile<T0>::Type;
    using SrcTileData = typename PtoTile<T1>::Type;

    static_assert(DstTileData::Rows == SrcTileData::Rows && DstTileData::Cols == SrcTileData::Cols,
                  "MoveUbToUb requires source and destination tiles to have identical static shapes.");
    static_assert(std::is_same_v<typename DstTileData::DType, typename SrcTileData::DType>,
                  "MoveUbToUb requires source and destination tiles to have identical data types.");

    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto srcTile = PtoTile<T1>(src);
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                srcTile.Assign(src, tileOffsets);
                pto::TMOV(dstTile.Data(), srcTile.Data());
            }
        }
    }
}

#endif // TILEOP_TILE_OPERATOR_MOVE_UB_TO_UB__H
