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
 * \\file scatter_update.cpp
 * \\brief
 */

#include "interface/utils/operator_tracer.h"
#include "passes/pass_utils/graph_utils.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/operation/operation_common.h"
#include "interface/operation/vector/gather_mask_common.h"
#include "tensor_transformation.h"
#include "tilefwk/error_code.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {

void TiledScatterUpdate(size_t cur, Function& function, const TileShape& tileShape, Input& srcInput, Input& indexInput,
                        Input& dstInput, int axis, const LogicalTensorPtr& dst, TileInfo& dstTileInfo,
                        std::string cacheMode, int blockSize)
{
    if (cur == dst->shape.size()) {
        // add Operation
        auto srcTile = srcInput.tensor.GetStorage()->View(function, srcInput.tileInfo.shape, srcInput.tileInfo.offset);
        auto dstTile = dstInput.tensor.GetStorage()->View(function, dstTileInfo.shape, dstTileInfo.offset);
        auto resultTile = dst->View(function, dstTileInfo.shape, dstTileInfo.offset);
        auto indexTile = indexInput.tensor.GetStorage()->View(function, indexInput.tileInfo.shape,
                                                              indexInput.tileInfo.offset);
        auto& op = function.AddOperation("TILE_INDEX_OUTCAST", {srcTile, indexTile, dstTile}, {resultTile});
        op.SetAttribute("axis", axis);
        op.SetAttribute(OpAttributeKey::panzBlockSize, blockSize);
        op.SetAttribute(OpAttributeKey::cacheMode, cacheMode);
        return;
    }

    // 按照dstShape进行切分
    auto& vecTile = tileShape.GetVecTile();
    int64_t tmpTile = vecTile[cur];
    if (static_cast<int>(cur) == axis) {
        tmpTile = dst->shape[cur];
    }
    for (int i = 0; i < dst->shape[cur]; i += tmpTile) {
        if (static_cast<int>(cur) == axis) {
            srcInput.tileInfo.offset[cur] = 0;
            srcInput.tileInfo.shape[cur] = srcInput.tensor.GetShape()[cur];
            if (cur <= 1) {
                indexInput.tileInfo.offset[cur] = 0;
                indexInput.tileInfo.shape[cur] = indexInput.tensor.GetShape()[cur];
            }
            dstTileInfo.offset[cur] = 0;
            dstTileInfo.shape[cur] = dst->shape[cur];
        } else {
            srcInput.tileInfo.offset[cur] = i % srcInput.tensor.GetShape()[cur];
            srcInput.tileInfo.shape[cur] = std::min(srcInput.tensor.GetShape()[cur] - srcInput.tileInfo.offset[cur],
                                                    tmpTile);
            if (cur == 0) { // only cut index first axis
                indexInput.tileInfo.offset[cur] = i % indexInput.tensor.GetShape()[cur];
                indexInput.tileInfo.shape[cur] = std::min(
                    indexInput.tensor.GetShape()[cur] - indexInput.tileInfo.offset[cur], tmpTile);
            } else {
                indexInput.tileInfo.offset[1] = 0;
                indexInput.tileInfo.shape[1] = indexInput.tensor.GetShape()[1];
            }
            dstTileInfo.offset[cur] = i;
            dstTileInfo.shape[cur] = std::min(dst->shape[cur] - dstTileInfo.offset[cur], tmpTile);
        }
        TiledScatterUpdate(cur + 1, function, tileShape, srcInput, indexInput, dstInput, axis, dst, dstTileInfo,
                           cacheMode, blockSize);
    }
}

void TiledIndexScatterUpdate(size_t cur, Function& function, const TileShape& tileShape, Input& srcInput,
                             Input& indexInput, Input& dstInput, int axis, const std::shared_ptr<LogicalTensor>& dst,
                             TileInfo& dstTileInfo, std::string cacheMode, int blockSize)
{
    if (cur == dst->shape.size()) {
        // add Operation
        auto srcTile = srcInput.tensor.GetStorage()->View(function, srcInput.tileInfo.shape, srcInput.tileInfo.offset);
        auto dstTile = dstInput.tensor.GetStorage()->View(function, dstTileInfo.shape, dstTileInfo.offset);
        auto indexTile = indexInput.tensor.GetStorage()->View(function, indexInput.tileInfo.shape,
                                                              indexInput.tileInfo.offset);
        auto& op = function.AddOperation("TILE_INDEX_OUTCAST", {srcTile, indexTile, dstTile}, {dst});
        op.SetAttribute("axis", axis);
        op.SetAttribute(OpAttributeKey::panzBlockSize, blockSize);
        op.SetAttribute(OpAttributeKey::cacheMode, cacheMode);
        return;
    }

    // 按照srcShape进行切分
    auto& vecTile = tileShape.GetVecTile();
    int64_t tmpTile = vecTile[cur];
    if (static_cast<int>(cur) == axis) {
        tmpTile = srcInput.tensor.GetShape()[cur];
    }

    for (int i = 0; i < srcInput.tensor.GetShape()[cur]; i += tmpTile) {
        if (static_cast<int>(cur) == axis) { // asis == 1
            srcInput.tileInfo.offset[cur] = 0;
            srcInput.tileInfo.shape[cur] = srcInput.tensor.GetShape()[cur];

            int64_t indexTileLen = vecTile[0];
            indexInput.tileInfo.offset[cur] = 0;
            indexInput.tileInfo.shape[cur] = std::min(indexInput.tensor.GetShape()[cur] - indexInput.tileInfo.offset[0],
                                                      indexTileLen);

            // indextileinfo need trans : [16,0] -> [0,16]
            indexInput.tileInfo.offset[cur] = indexInput.tileInfo.offset[0];
            indexInput.tileInfo.offset[0] = 0;

            dstTileInfo.offset[cur] = 0;
            dstTileInfo.shape[cur] = dst->shape[cur];
        } else {
            srcInput.tileInfo.offset[cur] = i % srcInput.tensor.GetShape()[cur];
            srcInput.tileInfo.shape[cur] = std::min(srcInput.tensor.GetShape()[cur] - srcInput.tileInfo.offset[cur],
                                                    tmpTile);

            indexInput.tileInfo.offset[0] = i % indexInput.tensor.GetShape()[1];
            indexInput.tileInfo.shape[0] = indexInput.tensor.GetShape()[0]; // index axis 0

            dstTileInfo.offset[cur] = i;
            dstTileInfo.shape[cur] = tmpTile;
        }
        TiledIndexScatterUpdate(cur + 1, function, tileShape, srcInput, indexInput, dstInput, axis, dst, dstTileInfo,
                                cacheMode, blockSize);
    }
}

void TiledScatterUpdateFor2Dims(Function& function, const TileShape& tileShape, const LogicalTensorPtr& result,
                                const LogicalTensorPtr& src, const LogicalTensorPtr& index, const LogicalTensorPtr& dst,
                                int axis, std::string cacheMode, int blockSize)
{
    auto& vecTile = tileShape.GetVecTile();
    int64_t tileBS = vecTile[NUM_VALUE_0];
    int64_t tileD = vecTile[NUM_VALUE_1];
    int64_t s = index->shape[1];
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, s != 0) << "1 dim of index is zero.";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, tileBS != 0) << "0 dim of tileshape is zero.";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, tileD != 0) << "1 dim of tileshape is zero.";
    CHECK(VectorErrorCode::ERR_CONFIG_TILE, (tileBS <= s && s % tileBS == 0) || (tileBS > s && tileBS % s == 0))
        << "tileshape 0 is invalid, tileshape(" << tileBS << ", " << tileD << ")";
    int64_t tileB = CeilDiv(tileBS, s);
    int64_t tileS = tileBS < s ? tileBS : s;
    int64_t bsOffset = 0;
    for (int64_t bIdx = 0; bIdx < index->shape[0]; bIdx += tileB) {
        for (int64_t sIdx = 0; sIdx < index->shape[1]; sIdx += tileS) {
            auto indexTile = index->View(
                function, {std::min(index->shape[0] - bIdx, tileB), std::min(index->shape[1] - sIdx, tileS)},
                {bIdx, sIdx});
            for (int64_t j = 0; j < src->shape[1]; j += tileD) {
                auto srcTile = src->View(
                    function, {std::min(src->shape[0] - bsOffset, tileBS), std::min(src->shape[1] - j, tileD)},
                    {bsOffset, j});
                auto dstTile = dst->View(function, {dst->shape[0], std::min(dst->shape[1] - j, tileD)}, {0, j});
                auto& op = function.AddOperation("TILE_INDEX_OUTCAST", {srcTile, indexTile, dstTile}, {result});
                op.SetAttribute("axis", axis);
                op.SetAttribute(OpAttributeKey::panzBlockSize, blockSize);
                op.SetAttribute(OpAttributeKey::cacheMode, cacheMode);
            }
            bsOffset += tileBS;
        }
    }
}

void TiledScatterUpdateFor4Dims(Function& function, const TileShape& tileShape, const LogicalTensorPtr& result,
                                const LogicalTensorPtr& src, const LogicalTensorPtr& index, const LogicalTensorPtr& dst,
                                int axis, std::string cacheMode, int blockSize)
{
    auto& vecTile = tileShape.GetVecTile();
    int64_t tileB = vecTile[NUM_VALUE_0];
    int64_t tileS = vecTile[NUM_VALUE_1];
    int64_t tileN = vecTile[NUM_VALUE_2];
    int64_t tileD = vecTile[NUM_VALUE_3];
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, tileB != 0) << "0 dim of tileshape is zero.";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, tileS != 0) << "1 dim of tileshape is zero.";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, tileN != 0) << "2 dim of tileshape is zero.";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, tileD != 0) << "3 dim of tileshape is zero.";
    for (int64_t i = 0; i < src->shape[0]; i += tileB) {
        for (int64_t j = 0; j < src->shape[1]; j += tileS) {
            auto indexTile = index->View(
                function, {std::min(index->shape[0] - i, tileB), std::min(index->shape[1] - j, tileS)}, {i, j});
            for (int64_t n = 0; n < src->shape[NUM_VALUE_2]; n += tileN) {
                for (int64_t d = 0; d < src->shape[NUM_VALUE_3]; d += tileD) {
                    auto srcTile = src->View(
                        function,
                        {std::min(src->shape[0] - i, tileB), std::min(src->shape[1] - j, tileS),
                         std::min(src->shape[NUM_VALUE_2] - n, tileN), std::min(src->shape[NUM_VALUE_3] - d, tileD)},
                        {i, j, n, d});
                    auto dstTile = dst->View(function,
                                             {dst->shape[0], dst->shape[1], dst->shape[NUM_VALUE_2],
                                              std::min(dst->shape[NUM_VALUE_3] - d, tileD)},
                                             {0, 0, 0, d});
                    auto& op = function.AddOperation("TILE_INDEX_OUTCAST", {srcTile, indexTile, dstTile}, {result});
                    op.SetAttribute("axis", axis);
                    op.SetAttribute(OpAttributeKey::panzBlockSize, blockSize);
                    op.SetAttribute(OpAttributeKey::cacheMode, cacheMode);
                }
            }
        }
    }
}

void TiledScatterUpdate(Function& function, const TileShape& tileShape, const LogicalTensorPtr& result,
                        const LogicalTensorPtr& src, const LogicalTensorPtr& index, const LogicalTensorPtr& dst,
                        int axis, std::string cacheMode, int blockSize)
{
    if (cacheMode == "PA_BSND") {
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, src->shape.size() == NUM_VALUE_2 || src->shape.size() == NUM_VALUE_4)
            << "shape must be 2 or 4";
        if (src->shape.size() == NUM_VALUE_2) {
            TiledScatterUpdateFor2Dims(function, tileShape, result, src, index, dst, axis, cacheMode, blockSize);
        } else if (src->shape.size() == NUM_VALUE_4) {
            TiledScatterUpdateFor4Dims(function, tileShape, result, src, index, dst, axis, cacheMode, blockSize);
        }
        return;
    }
    // Check Operands Valid
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, result->shape.size() == result->offset.size())
        << "The shape of result and offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, src->shape.size() == src->offset.size())
        << "The shape of src and offset should be equal";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, index->shape.size() == index->offset.size())
        << "The shape of index and offset should be equal";

    TileInfo srcTileInfo(src->shape.size(), src->offset.size());
    TileInfo indexTileInfo(index->shape.size(), index->offset.size());
    TileInfo dstTileInfo(dst->shape.size(), dst->offset.size());
    TileInfo resultTileInfo(result->shape.size(), result->offset.size());

    auto srcInput = Input{src, srcTileInfo};
    auto indexInput = Input{index, indexTileInfo};
    auto dstInput = Input{dst, dstTileInfo};
    auto& vecTile = tileShape.GetVecTile();
    if (axis == 1 && src->shape.size() == NUM_VALUE_2 && vecTile[1] == src->shape[1]) { // 2维切index场景
        TiledIndexScatterUpdate(0, function, tileShape, srcInput, indexInput, dstInput, axis, result, resultTileInfo,
                                cacheMode, blockSize);
    } else {
        TiledScatterUpdate(0, function, tileShape, srcInput, indexInput, dstInput, axis, result, resultTileInfo,
                           cacheMode, blockSize);
    }
}

void TensorScatterUpdate(Function& function, const LogicalTensorPtr& result, const LogicalTensorPtr& dst,
                         const LogicalTensorPtr& index, const LogicalTensorPtr& src, int axis, std::string cacheMode,
                         int blockSize)
{
    // src: ub
    // index: ub
    // dst: gm
    // result: gm
    auto& op = function.AddOperation(Opcode::OP_INDEX_OUTCAST, {src, index, dst}, {result});
    op.SetAttribute("axis", axis);
    op.SetAttribute(OpAttributeKey::panzBlockSize, blockSize);
    op.SetAttribute(OpAttributeKey::cacheMode, cacheMode);
}

static void CheckScatterUpdateInput(const Tensor& input)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          (input.GetShape().size() == NUM_VALUE_2 &&
           (input.GetShape(NUM_VALUE_0) != NUM_VALUE_0 && input.GetShape(NUM_VALUE_1) != NUM_VALUE_0)) ||
              (input.GetShape().size() == NUM_VALUE_4 &&
               (input.GetShape(NUM_VALUE_0) != NUM_VALUE_0 && input.GetShape(NUM_VALUE_1) != NUM_VALUE_0 &&
                input.GetShape(NUM_VALUE_2) != NUM_VALUE_0 && input.GetShape(NUM_VALUE_3) != NUM_VALUE_0)))
        << "The shape of input is invalid";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          input.GetShape().size() == NUM_VALUE_2 || input.GetShape().size() == NUM_VALUE_4)
        << "The shape size of input is invalid";
    CheckTensorDimRange(input.GetStorage(), NUM_VALUE_2, NUM_VALUE_4, "SCATTERUPDATE");
}

static void CheckScatterUpdateIndex(const Tensor& index)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, index.GetShape().size() == NUM_VALUE_2 &&
                                                  index.GetShape(NUM_VALUE_0) != NUM_VALUE_0 &&
                                                  index.GetShape(NUM_VALUE_1) != NUM_VALUE_0)
        << "The shape of index is invalid";
}

static void CheckScatterUpdateInvalid(const Tensor& dst, const Tensor& index, const Tensor& src)
{
    const auto& combinations = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_SCATTER_UPDATE);
    CheckTensorDataTypeCombination({dst.GetDataType(), index.GetDataType(), src.GetDataType()}, combinations,
                                   "SCATTERUPDATE");
    std::vector<LogicalTensorPtr> tensors = {dst.GetStorage(), src.GetStorage()};
    CheckTensorsDimConsistency(tensors, "SCATTERUPDATE");
    CheckScatterUpdateIndex(index);
    CheckScatterUpdateInput(src);
    CheckScatterUpdateInput(dst);
}

Tensor ScatterUpdate(Tensor& dst, const Tensor& index, const Tensor& src, int axis, std::string cacheMode,
                     int chunkSize)
{
    DECLARE_TRACER();
    CheckTensorFormat(dst.GetStorage(), {TileOpFormat::TILEOP_NZ}, "ScatterUpdate");
    CheckTensorFormat(index.GetStorage(), {TileOpFormat::TILEOP_NZ}, "ScatterUpdate");
    CheckTensorFormat(src.GetStorage(), {TileOpFormat::TILEOP_NZ}, "ScatterUpdate");

    CheckScatterUpdateInvalid(dst, index, src);
    CheckAxisRange(dst, axis);

    Tensor result(dst.GetStorage()->Datatype(), dst.GetStorage()->GetShape(), "", dst.Format());
    if (std::find(dst.GetStorage()->GetShape().begin(), dst.GetStorage()->GetShape().end(), -1) !=
        dst.GetStorage()->GetShape().end()) {
        Tensor resTmp(dst.GetStorage()->Datatype(), dst.GetStorage()->GetDynValidShape(), "", dst.Format());
        result = resTmp;
    }

    if (cacheMode == "PA_NZ") {
        axis = 1;
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, src.GetShape().size() == NUM_VALUE_2)
            << "Only 2D input is supported"; // only 2D input is supported

        Tensor newIndex = Reshape(index, {1, index.GetShape()[0] * index.GetShape()[1]});
        CALL(ScatterUpdate, *Program::GetInstance().GetCurrentFunction(), result.GetStorage(), dst.GetStorage(),
             newIndex.GetStorage(), src.GetStorage(), axis, cacheMode, chunkSize);
    } else {
        CALL(ScatterUpdate, *Program::GetInstance().GetCurrentFunction(), result.GetStorage(), dst.GetStorage(),
             index.GetStorage(), src.GetStorage(), axis, cacheMode, chunkSize);
    }

    dst = result;
    return result;
}

void IndexOutcastOperationTileFunc(Function& function, const TileShape& tileShape,
                                   const std::vector<LogicalTensorPtr>& iOperand,
                                   const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    int axis = op.GetIntAttribute("axis");
    int blockSize = op.GetIntAttribute(OpAttributeKey::panzBlockSize);
    std::string cacheMode = op.GetStringAttribute(OpAttributeKey::cacheMode);
    TiledScatterUpdate(function, tileShape, oOperand[0], iOperand[0], iOperand[1], iOperand[NUM_VALUE_2], axis,
                       cacheMode, blockSize);
}

REGISTER_OPERATION_TILED_FUNC(OP_INDEX_OUTCAST, Opcode::OP_INDEX_OUTCAST, IndexOutcastOperationTileFunc);

} // namespace npu::tile_fwk
