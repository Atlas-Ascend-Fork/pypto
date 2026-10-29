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
 * \file move_ub_to_ub.cpp
 * \brief UB 到 UB 数据搬运操作实现
 *
 * 在 UB 内将输入 Tensor 搬运到输出 Tensor，输入输出 shape、valid shape、分形（ND）保持一致。
 * TileShape 切分方式参考 Add 算子：沿各维度按照 vec_tile 步长递归切分。
 */

#include "interface/utils/operator_tracer.h"
#include "interface/utils/common.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation_common.h"
#include "interface/function/function.h"
#include "interface/program/program.h"

#include <string>

namespace npu::tile_fwk {

namespace {
constexpr size_t MOVE_UB_TO_UB_MIN_RANK = 1;
constexpr size_t MOVE_UB_TO_UB_MAX_RANK = NUM_VALUE_5;
const std::vector<NPUArch> MOVE_UB_TO_UB_SUPPORTED_ARCHITECTURES = {NPUArch::DAV_2201, NPUArch::DAV_3510};

std::string SymbolicShapesToStr(const std::vector<SymbolicScalar>& shapes)
{
    std::string result = "[";
    for (size_t i = 0; i < shapes.size(); ++i) {
        result += shapes[i].Dump();
        if (i + 1 < shapes.size()) {
            result += ", ";
        }
    }
    result += "]";
    return result;
}

void CheckMoveUbToUbShapeConsistency(const LogicalTensorPtr& src, const LogicalTensorPtr& dst)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, src->GetShape() == dst->GetShape())
        << "MoveUbToUb requires input and output to have the same shape. Input shape: " << IntVecToStr(src->GetShape())
        << ", output shape: " << IntVecToStr(dst->GetShape());
    const auto& srcValidShape = src->GetDynValidShape();
    const auto& dstValidShape = dst->GetDynValidShape();
    bool validShapeConsistent = srcValidShape.size() == dstValidShape.size();
    if (validShapeConsistent) {
        for (size_t i = 0; i < srcValidShape.size(); ++i) {
            if (srcValidShape[i].Simplify().Dump() != dstValidShape[i].Simplify().Dump()) {
                validShapeConsistent = false;
                break;
            }
        }
    }
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, validShapeConsistent)
        << "MoveUbToUb requires input and output to have the same valid shape. Input valid shape: "
        << SymbolicShapesToStr(srcValidShape) << ", output valid shape: " << SymbolicShapesToStr(dstValidShape);
}

void CheckMoveUbToUbInput(const Tensor& input)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, input.Format() == TileOpFormat::TILEOP_ND)
        << "MoveUbToUb only supports TILEOP_ND input.";
    CHECK(VectorErrorCode::ERR_PARAM_SHAPE_DIM_UNSUPPORTED,
          MOVE_UB_TO_UB_MIN_RANK <= input.GetShape().size() && input.GetShape().size() <= MOVE_UB_TO_UB_MAX_RANK)
        << "MoveUbToUb only supports 1D to 5D input. Current input rank: " << input.GetShape().size();
}

void CheckMoveUbToUbTileShape(const LogicalTensorPtr& input, const VecTile& vecTile)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, vecTile.size() == input->GetShape().size())
        << "MoveUbToUb tile shape rank must match input rank. Input shape: " << IntVecToStr(input->GetShape())
        << ", tile shape: " << IntVecToStr(vecTile.tile);
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, vecTile[vecTile.size() - 1] > 0)
        << "MoveUbToUb tile shape last dim must be positive. Current last tile dim: " << vecTile[vecTile.size() - 1];
}

void TiledMoveUbToUbOperation(Function& function, const TileShape& tileShape, size_t cur, Input& input,
                              const LogicalTensorPtr& dst)
{
    if (cur == input.tensor.GetShape().size()) {
        auto srcTile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        auto dstTile = dst->View(function, input.tileInfo.shape, input.tileInfo.offset);
        function.AddOperation(Opcode::OP_MOVE_UB_TO_UB, {srcTile}, {dstTile});
        return;
    }

    const auto& vecTile = tileShape.GetVecTile();
    int64_t step = std::max<int64_t>(1, std::min<int64_t>(vecTile[cur], input.tensor.GetShape()[cur]));
    for (int64_t i = 0; i < input.tensor.GetShape()[cur]; i += step) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, step);
        input.tileInfo.offset[cur] = i;
        TiledMoveUbToUbOperation(function, tileShape, cur + 1, input, dst);
    }
}

void MoveUbToUbTileFunc(Function& function, const TileShape& tileShape, const std::vector<LogicalTensorPtr>& iOperand,
                        const std::vector<LogicalTensorPtr>& oOperand, [[maybe_unused]] const Operation& op)
{
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, iOperand.size() == NUM_VALUE_1) << "MoveUbToUb expects 1 input tensor.";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, oOperand.size() == NUM_VALUE_1) << "MoveUbToUb expects 1 output tensor.";

    const auto& src = iOperand[0];
    const auto& dst = oOperand[0];
    CheckMoveUbToUbShapeConsistency(src, dst);
    CheckTensorsDataTypeConsistency(src, dst, "MoveUbToUb");
    CheckMoveUbToUbTileShape(src, tileShape.GetVecTile());
    TileInfo inputTileInfo(src->shape.size(), src->offset.size());
    auto input = Input{Tensor(src), inputTileInfo};
    TiledMoveUbToUbOperation(function, tileShape, 0, input, dst);
}
} // namespace

void MoveUbToUb(const Tensor& src, Tensor& dst)
{
    DECLARE_TRACER();
    CheckSupportedNPUArch(MOVE_UB_TO_UB_SUPPORTED_ARCHITECTURES, "MoveUbToUb");
    CheckTensorFormat(src.GetStorage(), {TileOpFormat::TILEOP_NZ}, "MoveUbToUb");
    CheckTensorFormat(dst.GetStorage(), {TileOpFormat::TILEOP_NZ}, "MoveUbToUb");
    CheckMoveUbToUbInput(src);
    CheckMoveUbToUbInput(dst);
    CheckMoveUbToUbShapeConsistency(src.GetStorage(), dst.GetStorage());
    CheckTensorsDataTypeConsistency(src.GetStorage(), dst.GetStorage(), "MoveUbToUb");

    auto& op = Program::GetInstance().GetCurrentFunction()->AddOperation(Opcode::OP_MOVE_UB_TO_UB, {src.GetStorage()},
                                                                         {dst.GetStorage()});
    (void)op;
}

REGISTER_OPERATION_TILED_FUNC(MoveUbToUb, Opcode::OP_MOVE_UB_TO_UB, MoveUbToUbTileFunc);

} // namespace npu::tile_fwk
