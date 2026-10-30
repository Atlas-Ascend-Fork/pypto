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
 * \file sort.cpp
 * \brief
 */

#include <string>
#include <queue>
#include "interface/utils/common.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation_common.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "interface/utils/operator_tracer.h"
#include "tensor_transformation.h"
#include "passes/pass_utils/graph_utils.h"
#include "tilefwk/platform.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

const std::string EXTRACT_MASKMODE = OP_ATTR_PREFIX + "makeMode";

const std::string SORT_AXIS = OP_ATTR_PREFIX + "axis";
const std::string SORT_GMSTRIDE = OP_ATTR_PREFIX + "gmstride";
const std::string SORT_KVALUE = OP_ATTR_PREFIX + "kvalue";
const std::string SORT_MERGE_SIZE = OP_ATTR_PREFIX + "mergeSize";
const std::string SORT_ORDER = OP_ATTR_PREFIX + "order";
const std::string SORT_OFFSET = OP_ATTR_PREFIX + "offset";
const std::string SORT_FIRSTSHAPE = OP_ATTR_PREFIX + "firstShape";

constexpr int32_t kBlockSize = NUM_VALUE_32;
constexpr int32_t kBlockFpNum = NUM_VALUE_8;

void TiledArgSort(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                  const LogicalTensorPtr& resultDices, TileInfo& resultDicesTileInfo, int axis, int isLargest)
{
    if (cur == input.tensor.GetShape().size()) {
        auto inputTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto resultDicesTile = resultDices->View(function, resultDicesTileInfo.shape, resultDicesTileInfo.offset);
        function.AddOperation(Opcode::OP_ARGSORT, {inputTile}, {resultDicesTile});
        return;
    }
    if (cur == static_cast<size_t>(axis)) {
        input.tileInfo.offset[cur] = 0;
        input.tileInfo.shape[cur] = input.tensor.GetShape()[cur];
        TiledArgSort(function, tileShape, cur + 1, input, resultDices, resultDicesTileInfo, axis, isLargest);
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        // update input && result && resultDices shape and offset info
        input.tileInfo.offset[cur] = i % input.tensor.GetShape()[cur];
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - input.tileInfo.offset[cur], vecTile[cur]);

        resultDicesTileInfo.offset[cur] = i;
        resultDicesTileInfo.shape[cur] = std::min(resultDices->shape[cur] - resultDicesTileInfo.offset[cur],
                                                  vecTile[cur]);
        TiledArgSort(function, tileShape, cur + 1, input, resultDices, resultDicesTileInfo, axis, isLargest);
    }
}

void TiledArgSort(Function& function, const TileShape& tileShape, const LogicalTensorPtr operand,
                  const LogicalTensorPtr resDicesOperand, int axis, int isLargest)
{
    // Build Init tile info
    TileInfo tileInfo(operand->shape.size(), operand->offset.size());
    TileInfo resultDicesTileInfo(resDicesOperand->shape.size(), resDicesOperand->offset.size());
    auto input = Input{operand, tileInfo};
    TiledArgSort(function, tileShape, 0, input, resDicesOperand, resultDicesTileInfo, axis, isLargest);
}

bool checkIsExceedUB(const std::vector<int64_t>& tileShape, const std::vector<int64_t>& shape, int axis,
                     int blockSize = NUM_VALUE_32)
{
    int64_t UBSize = 196608;

    // check shape is out of UB size
    int64_t tileRowShapeSize = 1; // tileShape[0] * tileShape[1] * ... * rawShape[-1]
    for (const auto& num : tileShape) {
        tileRowShapeSize *= num;
    }
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, tileShape[axis] > 0) << "tileShape in axis must greater than 0.";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, blockSize > 0) << "blockSize must greater than 0.";
    tileRowShapeSize = tileRowShapeSize / tileShape[axis] * ((shape[axis] + blockSize - 1) / blockSize * blockSize);
    int64_t maxShapeSize = tileRowShapeSize * NUM_VALUE_2 * NUM_VALUE_4 * NUM_VALUE_4; // every element is 8B
    bool isInGM = maxShapeSize >= UBSize ? true : false;
    return isInGM;
}

void TiledSortInUb(Function& function, const LogicalTensorPtr& source, const LogicalTensorPtr& valueResult,
                   const LogicalTensorPtr& indexResult, TileInfo& resultTileInfo, int axis, int descending)
{
    // 每32个元素进行排序
    std::vector<int64_t> bitSortOutputShape = source->shape;
    bitSortOutputShape[axis] = (bitSortOutputShape[axis] + kBlockSize - 1) / kBlockSize * kBlockSize;
    bitSortOutputShape[axis] = bitSortOutputShape[axis] * NUM_VALUE_2;
    auto bitSortOutputTensor = std::make_shared<LogicalTensor>(function, source->Datatype(), bitSortOutputShape);
    std::vector<int64_t> tmpShape;
    if (bitSortOutputShape.size() == 1) {
        tmpShape = {bitSortOutputShape[axis]};
    } else {
        tmpShape = {1, bitSortOutputShape[axis]};
    }
    auto tempTensor = std::make_shared<LogicalTensor>(function, source->Datatype(), tmpShape);
    auto& bitSortOp = function.AddOperation(Opcode::OP_BITSORT, {source}, {bitSortOutputTensor, tempTensor});
    bitSortOp.SetAttribute(SORT_AXIS, axis);
    bitSortOp.SetAttribute(SORT_ORDER, static_cast<int>(descending));
    bitSortOp.SetAttribute(SORT_OFFSET, static_cast<int>(0));
    std::vector<SymbolicScalar> bitSortDynValidShape(source->GetDynValidShape());
    bitSortDynValidShape[axis] = bitSortDynValidShape[axis] * NUM_VALUE_2;
    bitSortOutputTensor->UpdateDynValidShape(bitSortDynValidShape);
    if (bitSortOutputShape.size() == 1) {
        tempTensor->UpdateDynValidShape({bitSortDynValidShape[axis]});
    } else {
        tempTensor->UpdateDynValidShape({1, bitSortDynValidShape[axis]});
    }

    // 32个元素组成的block之间进行归并
    std::vector<int64_t> mrgSortOutputShape = source->shape;
    mrgSortOutputShape[axis] = (mrgSortOutputShape[axis] + kBlockFpNum - 1) / kBlockFpNum * kBlockFpNum * NUM_VALUE_2;
    auto mrgSortOutputTensor = std::make_shared<LogicalTensor>(function, source->Datatype(), mrgSortOutputShape);
    auto& mrgSortOp = function.AddOperation(Opcode::OP_MRGSORT, {bitSortOutputTensor},
                                            {mrgSortOutputTensor, tempTensor});
    mrgSortOp.SetAttribute(SORT_AXIS, axis);
    mrgSortOp.SetAttribute(SORT_KVALUE, source->shape[axis]);
    mrgSortOp.SetAttribute(SORT_MERGE_SIZE, NUM_VALUE_32);
    std::vector<SymbolicScalar> mrgSortDynValidShape(source->GetDynValidShape());
    mrgSortDynValidShape[axis] = source->GetDynValidShape()[axis] * NUM_VALUE_2;
    mrgSortOutputTensor->UpdateDynValidShape(mrgSortDynValidShape);
    if (bitSortOutputShape.size() == 1) {
        tempTensor->UpdateDynValidShape({bitSortDynValidShape[axis]});
    } else {
        tempTensor->UpdateDynValidShape({1, bitSortDynValidShape[axis]});
    }

    // 提取value和index
    auto valueTile = valueResult->View(function, resultTileInfo.shape, resultTileInfo.offset);
    auto& valueOp = function.AddOperation(Opcode::OP_EXTRACT, {mrgSortOutputTensor}, {valueTile});
    valueOp.SetAttribute(EXTRACT_MASKMODE, 0);
    valueOp.SetAttribute(SORT_KVALUE, source->shape[axis]);
    valueOp.SetAttribute(SORT_ORDER, descending);
    valueTile->UpdateDynValidShape(source->GetDynValidShape());

    auto indexTile = indexResult->View(function, resultTileInfo.shape, resultTileInfo.offset);
    auto& indexOp = function.AddOperation(Opcode::OP_EXTRACT, {mrgSortOutputTensor}, {indexTile});
    indexOp.SetAttribute(EXTRACT_MASKMODE, 1);
    indexOp.SetAttribute(SORT_KVALUE, source->shape[axis]);
    indexOp.SetAttribute(SORT_ORDER, descending);
    indexTile->UpdateDynValidShape(source->GetDynValidShape());
    return;
}

void TiledSortInGm(Function& function, const VecTile& vecTile, const Input& input, const LogicalTensorPtr& source,
                   const LogicalTensorPtr& valueResult, const LogicalTensorPtr& indexResult, TileInfo& resultTileInfo,
                   int axis, int descending)
{
    std::vector<int64_t> vecTileAlign = vecTile.tile; // tile shape after align axis
    vecTileAlign[axis] = (vecTileAlign[axis] + kBlockSize - 1) / kBlockSize * kBlockSize;

    std::vector<int64_t> tileSourceShape = source->shape;
    std::vector<int64_t> tileSourceOffset(tileSourceShape.size(), 0);
    std::vector<int64_t> tileBitSortShape = source->shape;

    // 创建一个2倍source的GM上的空间sortOutputTensor, 用于存储source排序后的结果
    std::vector<int64_t> sortOutputShape = source->shape;
    auto sortOutputValidShape = source->GetDynValidShape();
    // 元素个数k和8对齐，extract中二维的vreduce才能正常转换，因为UB中32B对齐，k*4B和32B对齐，则k与8对齐
    sortOutputShape[axis] = (sortOutputShape[axis] + kBlockFpNum - 1) / kBlockFpNum * kBlockFpNum * NUM_VALUE_2;
    sortOutputValidShape[axis] = sortOutputValidShape[axis] * NUM_VALUE_2;
    auto sortOutputTensor = std::make_shared<LogicalTensor>(function, source->Datatype(), sortOutputShape,
                                                            sortOutputValidShape);
    std::vector<int64_t> tileOutputShape = sortOutputShape;
    std::vector<int64_t> tileOutputOffset(sortOutputShape.size(), 0);

    for (int64_t i = 0; i < input.tensor.GetShape()[axis]; i += vecTileAlign[axis]) {
        tileSourceShape[axis] = std::min(vecTileAlign[axis], source->shape[axis] - i);
        tileSourceOffset[axis] = i;
        auto inputTile = source->View(function, tileSourceShape, tileSourceOffset);
        tileBitSortShape[axis] = (tileSourceShape[axis] + kBlockSize - 1) / kBlockSize * kBlockSize * NUM_VALUE_2;
        auto bitSortTile = std::make_shared<LogicalTensor>(function, source->Datatype(), tileBitSortShape);
        std::vector<int64_t> tmpShape = {1, tileBitSortShape[axis]};
        if (tileBitSortShape.size() == 1) {
            tmpShape = {tileBitSortShape[axis]};
        } else {
            tmpShape = {1, tileBitSortShape[axis]};
        }
        auto tempTensor = std::make_shared<LogicalTensor>(function, source->Datatype(), tmpShape);
        auto& bitSortOp = function.AddOperation(Opcode::OP_BITSORT, {inputTile}, {bitSortTile, tempTensor});
        bitSortOp.SetAttribute(SORT_AXIS, axis);
        bitSortOp.SetAttribute(SORT_ORDER, descending);
        bitSortOp.SetAttribute(SORT_OFFSET, i);
        std::vector<SymbolicScalar> bitSortDynValidShape(inputTile->GetDynValidShape());
        bitSortDynValidShape[axis] = bitSortDynValidShape[axis] * NUM_VALUE_2;
        bitSortTile->UpdateDynValidShape(bitSortDynValidShape);
        if (tileBitSortShape.size() == 1) {
            tempTensor->UpdateDynValidShape({bitSortDynValidShape[axis]});
        } else {
            tempTensor->UpdateDynValidShape({1, bitSortDynValidShape[axis]});
        }

        tileOutputShape[axis] = (tileSourceShape[axis] + kBlockFpNum - 1) / kBlockFpNum * kBlockFpNum *
                                NUM_VALUE_2; // UB 32B对齐，兼顾了DynMrgSort中的k向8对齐
        tileOutputOffset[axis] = i * NUM_VALUE_2;
        auto tmp = std::make_shared<LogicalTensor>(function, source->Datatype(), tileOutputShape);
        auto& mrgSortOp = function.AddOperation(Opcode::OP_MRGSORT, {bitSortTile}, {tmp, tempTensor});
        mrgSortOp.SetAttribute(SORT_AXIS, axis);
        mrgSortOp.SetAttribute(SORT_KVALUE, static_cast<int>(tileSourceShape[axis]));
        mrgSortOp.SetAttribute(SORT_MERGE_SIZE, NUM_VALUE_32);
        std::vector<SymbolicScalar> mrgSortDynValidShape(inputTile->GetDynValidShape());
        mrgSortDynValidShape[axis] = mrgSortDynValidShape[axis] * NUM_VALUE_2;
        tmp->UpdateDynValidShape(mrgSortDynValidShape);
        if (tileBitSortShape.size() == 1) {
            tempTensor->UpdateDynValidShape({mrgSortDynValidShape[axis]});
        } else {
            tempTensor->UpdateDynValidShape({1, mrgSortDynValidShape[axis]});
        }

        auto& assembleOp = function.AddOperation(config::GetContractOpcode(), {tmp}, {sortOutputTensor});
        assembleOp.iOperand[0]->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
        assembleOp.oOperand[0]->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
        assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(
            MemoryType::MEM_UB, tileOutputOffset,
            std::vector<SymbolicScalar>(tileOutputOffset.begin(), tileOutputOffset.end()), tmp->GetDynValidShape()));
    }

    vecTileAlign[axis] = vecTileAlign[axis] * NUM_VALUE_2;
    int64_t tileNum = (sortOutputShape[axis] + vecTileAlign[axis] - 1) / vecTileAlign[axis]; // 计算有多少Tile块

    int64_t roundNum = tileNum;
    std::queue<LogicalTensorPtr> q;
    q.push(sortOutputTensor);
    bool flag = true; // 判断当前是偶数还是奇数阶段
    for (int64_t round = 1; round <= roundNum; round++) {
        auto roundInputTensor = q.front();
        q.pop();
        unsigned firstShape = vecTileAlign[axis];
        auto roundOutputTensor = std::make_shared<LogicalTensor>(function, source->Datatype(), sortOutputShape,
                                                                 sortOutputValidShape);
        std::vector<SymbolicScalar> curValidShape = sortOutputValidShape;
        for (int64_t i = 0; i < sortOutputShape[axis];) {
            tileOutputOffset[axis] = i;
            if (i + vecTileAlign[axis] >= sortOutputShape[axis]) { // 尾块
                tileOutputShape[axis] = sortOutputShape[axis] - i;
            } else if (!flag && i == 0) { // 奇数阶段的头块
                tileOutputShape[axis] = vecTileAlign[axis];
            } else { // 两块
                tileOutputShape[axis] = std::min(NUM_VALUE_2 * vecTileAlign[axis], sortOutputShape[axis] - i);
            }
            i += tileOutputShape[axis];

            auto src = std::make_shared<LogicalTensor>(function, source->Datatype(), tileOutputShape);
            auto& viewOp = function.AddOperation(config::GetSliceOpcode(), {roundInputTensor}, {src});
            curValidShape[axis] = std::max(
                0, std::min(sortOutputValidShape[axis] - tileOutputOffset[axis], tileOutputShape[axis]));
            viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(
                tileOutputOffset, MemoryType::MEM_UB,
                std::vector<SymbolicScalar>(tileOutputOffset.begin(), tileOutputOffset.end()), curValidShape));
            src->UpdateDynValidShape(curValidShape);

            auto outputInUB = std::make_shared<LogicalTensor>(function, src->Datatype(), tileOutputShape);
            auto& twoTileMrgSortOp = function.AddOperation(Opcode::OP_TWOTILEMRGSORT, {src}, {outputInUB});
            twoTileMrgSortOp.SetAttribute(SORT_FIRSTSHAPE, static_cast<int>(firstShape));
            std::vector<SymbolicScalar> tileMrgSortDynValidShape(src->GetDynValidShape());
            outputInUB->UpdateDynValidShape(tileMrgSortDynValidShape);

            auto& assembleOp = function.AddOperation(config::GetContractOpcode(), {outputInUB}, {roundOutputTensor});
            assembleOp.iOperand[0]->SetMemoryTypeOriginal(MemoryType::MEM_UB, true);
            assembleOp.oOperand[0]->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
            assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(
                MemoryType::MEM_UB, tileOutputOffset,
                std::vector<SymbolicScalar>(tileOutputOffset.begin(), tileOutputOffset.end()),
                outputInUB->GetDynValidShape()));
        }
        q.push(roundOutputTensor);
        flag = !flag;
    }

    auto extractInputTensor = q.front();
    q.pop();
    for (int i = 0; i < sortOutputShape[axis]; i += vecTileAlign[axis]) {
        tileOutputShape[axis] = std::min(vecTileAlign[axis], sortOutputShape[axis] - i);
        tileOutputOffset[axis] = i;
        auto src = extractInputTensor->View(function, tileOutputShape, tileOutputOffset);
        resultTileInfo.shape[axis] = std::min(tileOutputShape[axis] / NUM_VALUE_2,
                                              source->shape[axis] - i / NUM_VALUE_2);
        resultTileInfo.offset[axis] = i / NUM_VALUE_2;

        auto valueTile = valueResult->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto& valueOp = function.AddOperation(Opcode::OP_EXTRACT_SINGLE, {src}, {valueTile});
        valueOp.SetAttribute(SORT_ORDER, descending);
        valueOp.SetAttribute(EXTRACT_MASKMODE, 0);
        std::vector<SymbolicScalar> extractDynValidShape(src->GetDynValidShape());
        extractDynValidShape[axis] = extractDynValidShape[axis] / NUM_VALUE_2;
        valueTile->UpdateDynValidShape(extractDynValidShape);

        auto indexTile = indexResult->View(function, resultTileInfo.shape, resultTileInfo.offset);
        auto& indexOp = function.AddOperation(Opcode::OP_EXTRACT_SINGLE, {src}, {indexTile});
        indexOp.SetAttribute(SORT_ORDER, descending);
        indexOp.SetAttribute(EXTRACT_MASKMODE, 1);
        indexTile->UpdateDynValidShape(extractDynValidShape);
    }
    return;
}

void TiledSort(Function& function, const TileShape& tileShape, size_t cur, Input& input,
               const LogicalTensorPtr& valueResult, const LogicalTensorPtr& indexResult, TileInfo& resultTileInfo,
               int axis, int descending)
{
    auto& vecTile = tileShape.GetVecTile();
    if (static_cast<int>(cur) == axis) {
        input.tileInfo.offset[axis] = 0;
        auto source = input.tensor.GetStorage()->View(
            function, input.tileInfo.shape, input.tileInfo.offset); // input.tensor是viewTensor, source是tileTensor
        bool isInGM = checkIsExceedUB(vecTile.tile, source->shape, axis, kBlockSize);
        if (isInGM) {
            TiledSortInGm(function, vecTile, input, source, valueResult, indexResult, resultTileInfo, axis, descending);
        } else {
            TiledSortInUb(function, source, valueResult, indexResult, resultTileInfo, axis, descending);
        }
        return;
    }

    for (int i = 0; i < input.tensor.GetShape()[cur]; i += vecTile[cur]) {
        input.tileInfo.offset[cur] = i % input.tensor.GetShape()[cur];
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - input.tileInfo.offset[cur], vecTile[cur]);
        resultTileInfo.offset[cur] = i;
        resultTileInfo.shape[cur] = std::min(valueResult->shape[cur] - resultTileInfo.offset[cur], vecTile[cur]);
        TiledSort(function, tileShape, cur + 1, input, valueResult, indexResult, resultTileInfo, axis, descending);
    }
}

void TiledSort(Function& function, const TileShape& tileShape, const LogicalTensorPtr operand,
               const LogicalTensorPtr valueResult, const LogicalTensorPtr indexResult, int axis, int descending)
{
    TileInfo tileInfo(operand->shape, operand->offset);
    TileInfo resultTileInfo(valueResult->shape, valueResult->offset);
    auto input = Input{operand, tileInfo};
    TiledSort(function, tileShape, 0, input, valueResult, indexResult, resultTileInfo, axis, descending);
}

void TensorSort(Function& function, const LogicalTensorPtr& self, LogicalTensorPtr& valueResult,
                LogicalTensorPtr& indexResult, int axis, bool descending)
{
    auto validShape = self->GetDynValidShape();
    auto& op = GraphUtils::AddDynOperation(function, Opcode::OP_SORT_UB, {self}, {valueResult, indexResult},
                                           {validShape, validShape});
    op.SetAttribute(SORT_AXIS, static_cast<int>(axis));
    op.SetAttribute(SORT_ORDER, static_cast<int>(descending));
    return;
}

std::tuple<Tensor, Tensor> sort(const Tensor& self, int axis = -1, bool descending = false)
{
    DECLARE_TRACER();
    CheckVectorOperand(self.GetStorage(), "sort");

    std::unordered_set<DataType> supportedTypes = {DT_FP32, DT_FP16};
    CheckTensorDataType(self.GetStorage(), supportedTypes, "SORT");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_4, "SORT");
    auto len = static_cast<int>(self.GetShape().size());
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, len >= NUM_VALUE_1 && len <= NUM_VALUE_4)
        << "Only 1D to 4D input is supported.\n";

    axis = axis >= 0 ? axis : axis + len;
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, axis >= 0 && axis < len)
        << "Invalid axis value: " << axis << ". Expected range: [-" << len << "," << len - 1 << "]\n";

    CHECK(VectorErrorCode::ERR_PARAM_INVALID, len != NUM_VALUE_4 || axis != 0)
        << "Sort not support the 0th axis of 4D input.\n";

    auto validShape = self.GetStorage()->GetDynValidShape();
    auto vecTileShape = TileShape::Current().GetVecTile();
    CHECK(VectorErrorCode::ERR_CONFIG_ALIGNMENT, vecTileShape[axis] % NUM_VALUE_32 == 0)
        << "The size of the tile shape along axis " << axis << " must be a multiple of 32. Got " << vecTileShape[axis]
        << ".\n";

    if (checkIsExceedUB(vecTileShape.tile, self.GetShape(), axis, NUM_VALUE_32)) {
        int64_t tileNum = (self.GetShape()[axis] + vecTileShape[axis] - 1) / vecTileShape[axis];
        CHECK(VectorErrorCode::ERR_PARAM_INVALID, tileNum < NUM_VALUE_128)
            << "For Large Shape in GM, the number of tile on sort axis must be less than 128.";
    }

    auto transposeSelf = Transpose(self, {axis, len - 1});
    std::swap(validShape[axis], validShape[len - 1]);
    transposeSelf.GetStorage()->UpdateDynValidShape(validShape);
    std::swap(vecTileShape[axis], vecTileShape[len - 1]);
    TileShape::Current().SetVecTile(vecTileShape);

    auto castSelf = Cast(transposeSelf, DataType::DT_FP32, CastMode::CAST_NONE);
    castSelf.GetStorage()->UpdateDynValidShape(transposeSelf.GetStorage()->GetDynValidShape());

    auto outShape = castSelf.GetShape();
    auto valueResult = Tensor(DataType::DT_FP32, outShape);
    auto indexResult = Tensor(DataType::DT_INT32, outShape);
    CALL(Sort, *Program::GetInstance().GetCurrentFunction(), castSelf.GetStorage(), valueResult.GetStorage(),
         indexResult.GetStorage(), len - 1, descending);

    auto castValueResult = Cast(valueResult, self.GetDataType(), CastMode::CAST_NONE);
    castValueResult.GetStorage()->UpdateDynValidShape(valueResult.GetStorage()->GetDynValidShape());

    TileShape::Current().SetVecTile(vecTileShape);
    auto transposeValueResult = Transpose(castValueResult, {axis, len - 1});
    auto transposeIndexResult = Transpose(indexResult, {axis, len - 1});
    std::swap(validShape[axis], validShape[len - 1]);
    transposeValueResult.GetStorage()->UpdateDynValidShape(validShape);
    transposeIndexResult.GetStorage()->UpdateDynValidShape(validShape);
    std::swap(vecTileShape[axis], vecTileShape[len - 1]);
    TileShape::Current().SetVecTile(vecTileShape);
    return std::tie(transposeValueResult, transposeIndexResult);
}

Tensor ArgSort(const Tensor& self, int axis, bool descending) { return std::get<1>(sort(self, axis, descending)); }

void ArgSortOperationTileFunc(Function& function, const TileShape& tileShape,
                              const std::vector<LogicalTensorPtr>& iOperand,
                              const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    int axis = op.GetIntAttribute("axis");
    int isLargest = op.GetIntAttribute("order");
    TiledArgSort(function, tileShape, iOperand[0], oOperand[0], axis, isLargest);
}

void SortOperationTileFunc(Function& function, const TileShape& tileShape,
                           const std::vector<LogicalTensorPtr>& iOperand, const std::vector<LogicalTensorPtr>& oOperand,
                           [[maybe_unused]] const Operation& op)
{
    int axis = op.GetIntAttribute(SORT_AXIS);
    int descending = op.GetIntAttribute(SORT_ORDER);
    TiledSort(function, tileShape, iOperand[0], oOperand[0], oOperand[1], axis, descending);
}

REGISTER_OPERATION_TILED_FUNC(OP_ARGSORT, Opcode::OP_ARGSORT, ArgSortOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_SORT_UB, Opcode::OP_SORT_UB, SortOperationTileFunc);

} // namespace npu::tile_fwk
