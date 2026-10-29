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
 * \file transpose.cpp
 * \brief
 */

#include "unary.h"
#include "interface/operation/operation_utils.h"
#include "tensor_transformation.h"
#include "interface/utils/operator_tracer.h"
#include "tilefwk/error_code.h"
#include "tilefwk/platform.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk {

// transpose 的三条执行路径：
//   degenerate  参与转置的某一维长度为 1（或两维等长）：等价于一次 reshape；
//   merged      把 [.., T1, .., T2, ..] 的多段 batch 维压成一段，做一次 2D/ND 转置；
//   multi-level 合轴不适用（视图不连续、UB 放不下、维数过高）时，用 MOVEIN/VNCHWCONV/
//               MOVEOUT 三级 tile 转置逐块处理。

// UB 容量随 SoC 变化（DAV_2201 为 192KB-256B，1001/2002 为 256KB，3002 为 248KB），
// 下面几个 tile 预算都按平台查询值取比例，不再假定某个固定容量；查不到时回退到框架支持的
// 最小 UB（192KB）。
static int64_t UbBytes()
{
    static const int64_t ubBytes = [] {
        int64_t ub = static_cast<int64_t>(Platform::Instance().GetDie().GetMemoryLimit(MemoryType::MEM_UB));
        return ub >= 128 * 1024 ? ub : 192 * 1024;
    }();
    return ubBytes;
}

// 单个 trans 指令的 tile 需要整体驻留 UB，留出安全余量后取 UB 的 5/6（≈160KB@192KB）。
static int64_t OpUbBudgetBytes() { return UbBytes() * 5 / 6; }

// 转置输入 tile 的合并预算，取 UB 的 1/4（≈48KB@192KB）。
static int64_t MergedTileBudgetBytes() { return UbBytes() / 4; }

// 2D 合轴后切 3D 尾块做最后一次转置时，单块预算取 UB 的 1/3（≈64KB@192KB）。
static int64_t Merged2DTailBudgetBytes() { return UbBytes() / 3; }

// 2D 特化合轴：被转置维小于它时合并前段维更划算（形状启发式，与 UB 容量无关）。
constexpr int64_t MERGE_2D_SMALL_LAST_DIM = 4096;
// b16 且行数不足重复因子时，列数超过 VNCHWCONV_B16_LARGE_COL_DIM 的大 tile 每列最多搬
// VNCHWCONV_B16_MAX_COL_STEP 个元素（形状启发式，与 UB 容量无关）。
constexpr int64_t VNCHWCONV_B16_LARGE_COL_DIM = 4096;
constexpr int64_t VNCHWCONV_B16_MAX_COL_STEP = 1024;

enum class TransposeOpType {
    TRANSPOSE_MOVEIN,
    TRANSPOSE_MOVEOUT,
    TRANSPOSE_VNCHWCONV,
};

template <TransposeOpType T>
Opcode GetTransposeOpName()
{
#define CASE(X)              \
    case TransposeOpType::X: \
        return Opcode::OP_##X
    switch (T) {
        CASE(TRANSPOSE_MOVEOUT);
        CASE(TRANSPOSE_MOVEIN);
        CASE(TRANSPOSE_VNCHWCONV);
        default:
            CHECK(VectorErrorCode::ERR_PARAM_INVALID, false) << "unknown transpose op type";
    }
#undef CASE
}

inline void UnalignPadTmpBufTile(std::vector<int64_t>& shape, int blockElem, DataType dtype)
{
    // tmpbuf按16 8对齐
    auto size = shape.size();
    if (size >= NUM_VALUE_2) {
        int64_t alignSize = VNCHWCONV_REPEAT;
        if (BytesOf(dtype) == 1) {
            alignSize = BLOCK_SIZE;
        }
        shape[size - NUM_VALUE_2] = AlignUp(shape[size - NUM_VALUE_2], alignSize);
        shape[size - 1] = AlignUp(shape[size - 1], blockElem);
    }
}

// 行方向：沿被转置的某一维一次取多块 tile，摊薄每条指令的固定开销。
// VNCHWCONV 额外要求行方向按 16 元素对齐，否则先按 tile 行数取整、再回退到安全步长。
static int64_t ComputeMergeStep(TransposeOpType type, int cur, const std::vector<int>& transDims,
                                const std::vector<int64_t>& tensorShape, const VecTile& vecTile, int64_t maxMerge)
{
    int64_t extent = tensorShape[cur];
    int64_t tile = vecTile[cur];
    bool canMerge = (cur == transDims[0]) || (cur == transDims[1] && tensorShape.size() >= NUM_VALUE_3);
    if (!canMerge || maxMerge <= 1) {
        return tile;
    }
    int64_t tilesInRow = (extent + tile - 1) / tile;
    int64_t step = tile * std::min<int64_t>(maxMerge, tilesInRow);
    if (cur != transDims[0] || type != TransposeOpType::TRANSPOSE_VNCHWCONV) {
        return step;
    }
    int64_t step16 = (step / VNCHWCONV_REPEAT) * VNCHWCONV_REPEAT;
    if (step16 >= tile) {
        return step16;
    }
    if (tensorShape.size() >= NUM_VALUE_3) {
        return std::max(step, std::min<int64_t>(VNCHWCONV_REPEAT, tilesInRow * tile));
    }
    return (tilesInRow * tile >= VNCHWCONV_REPEAT) ? VNCHWCONV_REPEAT : tile;
}

// 列方向：VNCHWCONV 的 tmp 占用随 batch 元素数与行数增长，按 UB 预算限制单次列步长；
// b16 且行数不足 16 时对超宽 tile 再收紧一档，避免一条指令的 tmp 越界。
static int64_t ClampStepByUbBudget(const Input& input, const std::vector<int>& transDims, int cur, int64_t step)
{
    int64_t rowTile = input.tileInfo.shape[transDims[0]];
    int64_t dtypeBytes = BytesOf(input.tensor.GetDataType());
    int64_t blockElem = BLOCK_SIZE / dtypeBytes;
    int64_t padRows = AlignUp(rowTile, blockElem);
    int64_t batchElems = 1;
    for (int d = 0; d < cur; ++d) {
        if (d != transDims[0]) {
            batchElems *= input.tileInfo.shape[d];
        }
    }
    // tmp 容量与 TTrans 的 tmpShape 分配保持一致（1 字节按 32 元素对齐，其它按 16 元素对齐）。
    int64_t tmpCoef = AlignUp(rowTile, dtypeBytes == 1 ? BLOCK_SIZE : VNCHWCONV_REPEAT);
    int64_t bytesPerCol = batchElems * (rowTile + padRows + tmpCoef) * dtypeBytes;
    int64_t colStepMax = OpUbBudgetBytes() / bytesPerCol;
    colStepMax = (colStepMax / blockElem) * blockElem;
    if (dtypeBytes == NUM_VALUE_2 && rowTile < VNCHWCONV_REPEAT &&
        input.tensor.GetShape()[cur] >= VNCHWCONV_B16_LARGE_COL_DIM) {
        colStepMax = std::min<int64_t>(colStepMax, VNCHWCONV_B16_MAX_COL_STEP);
    }
    return std::min(step, std::max<int64_t>(colStepMax, blockElem));
}

template <TransposeOpType T>
void TiledInnerTranspose(Function& function, const TileShape& tileShape, const int cur, Input& input,
                         const LogicalTensorPtr& result, const std::vector<int>& shape, int64_t maxMerge,
                         int64_t baseBytes)
{
    int shapeSize = input.tensor.GetShape().size();
    if (cur == shapeSize) {
        auto tile = input.tensor.GetStorage()->View(function, input.tileInfo.shape, input.tileInfo.offset);
        std::vector<int64_t> resultTileShape(input.tileInfo.shape);
        std::swap(resultTileShape[shape[0]], resultTileShape[shape[1]]);
        std::vector<int64_t> resultTileOfs(input.tileInfo.offset);
        std::swap(resultTileOfs[shape[0]], resultTileOfs[shape[1]]);
        auto resultTile = result->View(function, resultTileShape, resultTileOfs);
        if (T == TransposeOpType::TRANSPOSE_MOVEOUT || T == TransposeOpType::TRANSPOSE_MOVEIN) {
            auto& op = function.AddOperation(GetTransposeOpName<T>(), {tile}, {resultTile});
            op.SetAttribute(OP_ATTR_PREFIX + "shape", shape);
        } else {
            int64_t blockElem = BLOCK_SIZE / static_cast<int>(BytesOf(tile->Datatype()));
            // TTrans 会把整个转置 tile 暂存到 tmp，容量需覆盖
            // AlignUp(rows,16) * AlignUp(cols,blockElem)，不能按固定 32 列收缩，
            // 否则列数超过 32 的 tile 会写越界（UB OOB）。
            std::vector<int64_t> tmpShape = input.tileInfo.shape;
            UnalignPadTmpBufTile(tmpShape, blockElem, tile->Datatype());
            auto tempTensor = std::make_shared<LogicalTensor>(function, tile->Datatype(), tmpShape);
            tempTensor->dynValidShape_ = SymbolicScalar::FromConcrete(tmpShape);
            auto& op = function.AddOperation(GetTransposeOpName<T>(), {tile}, {resultTile, tempTensor});
            op.SetAttribute(OP_ATTR_PREFIX + "shape", shape);
        }
        return;
    }
    auto& vecTile = tileShape.GetVecTile();
    int64_t step = ComputeMergeStep(T, cur, shape, input.tensor.GetShape(), vecTile, maxMerge);
    if (cur == shape[1] && T == TransposeOpType::TRANSPOSE_VNCHWCONV) {
        step = ClampStepByUbBudget(input, shape, cur, step);
    }
    for (int i = 0; i < input.tensor.GetShape()[cur]; i += step) {
        input.tileInfo.shape[cur] = std::min(input.tensor.GetShape()[cur] - i, step);
        input.tileInfo.offset[cur] = i;
        TiledInnerTranspose<T>(function, tileShape, cur + 1, input, result, shape, maxMerge, baseBytes);
    }
}

template <TransposeOpType T>
void TiledInnerTranspose(Function& function, const TileShape& tileShape, const LogicalTensorPtr& operand,
                         const LogicalTensorPtr& result, const std::vector<int>& shape)
{
    TileInfo tileInfo(result->shape.size(), result->offset.size());
    auto input = Input{operand, tileInfo};
    const auto& vecTile = tileShape.GetVecTile();
    // UB tensor 按 32B 对齐行存储，最内维实际占用需向上取整到 blockElem；
    // 按逻辑元素数计算会低估（如 int32 的 1 列只占 4B 却按 32B 分配）。
    int64_t dtypeBytes = BytesOf(operand->Datatype());
    int64_t blockElem = BLOCK_SIZE / dtypeBytes;
    int64_t baseBytes = dtypeBytes;
    for (int i = 0; i < (int)operand->shape.size(); ++i) {
        int64_t ext = std::min<int64_t>(operand->shape[i], vecTile[i]);
        if (i == (int)operand->shape.size() - 1) {
            ext = AlignUp(ext, blockElem);
        }
        baseBytes *= ext;
    }
    int64_t maxMerge = baseBytes > 0 ? MergedTileBudgetBytes() / baseBytes : 1;
    if (maxMerge > 1) {
        auto dimMergeFactor = [&](int d) {
            return std::min<int64_t>(maxMerge, (operand->shape[d] + vecTile[d] - 1) / vecTile[d]);
        };
        while (maxMerge > 1 &&
               baseBytes * dimMergeFactor(shape[0]) * dimMergeFactor(shape[1]) > MergedTileBudgetBytes()) {
            --maxMerge;
        }
    }
    TiledInnerTranspose<T>(function, tileShape, 0, input, result, shape, maxMerge, baseBytes);
}

void TensorInnerTranspose(Function& function, const LogicalTensorPtr& self, const LogicalTensorPtr& result,
                          std::vector<int> perm)
{
    if (perm[0] != (int)self->shape.size() - 1 && perm[1] != (int)self->shape.size() - 1) {
        auto& operation = function.AddOperation(Opcode::OP_TRANSPOSE_MOVEOUT, {self}, {result});
        operation.SetAttribute(OP_ATTR_PREFIX + "shape", perm);
        return;
    }

    if (perm[0] == (int)self->shape.size() - NUM_VALUE_2 && // last 2 dims transpose
        perm[1] == (int)self->shape.size() - 1) {
        auto& operation = function.AddOperation(Opcode::OP_TRANSPOSE_VNCHWCONV, {self}, {result});
        operation.SetAttribute(OP_ATTR_PREFIX + "shape", perm);
        return;
    }

    CHECK(VectorErrorCode::ERR_PARAM_INVALID,
          self->shape.size() == NUM_VALUE_3 || self->shape.size() == NUM_VALUE_4) // input should be 3 or 4 dims
        << "Transpose shape should be [A1,T1,A2,T2] or [T1,A2,T2]";

    // [A1,T1,A2,T2] to [A1,A2,T1,T2] or [T1,A2,T2] to [A2,T1,T2]
    auto oldVecTileShapes = TileShape::Current().GetVecTile();
    auto newVecTileShape = oldVecTileShapes;
    std::vector<int64_t> tmpShape(self->shape);
    int dim1 = (tmpShape.size() == NUM_VALUE_3) ? 0 : 1;           // if input is 3 dims, dim1 = 0, otherwise dim1 = 1
    int dim2 = (tmpShape.size() == NUM_VALUE_3) ? 1 : NUM_VALUE_2; // if input is 3 dims, dim2 = 1, otherwise dim2 = 2
    std::swap(tmpShape[dim1], tmpShape[dim2]);
    std::swap(newVecTileShape[dim1], newVecTileShape[dim2]);
    auto outValidShapes = self->GetDynValidShape();
    std::swap(outValidShapes[dim1], outValidShapes[dim2]);
    auto moveInResult = std::make_shared<LogicalTensor>(function, self->Datatype(), tmpShape, outValidShapes);
    auto& inOp = function.AddOperation(Opcode::OP_TRANSPOSE_MOVEIN, {self}, {moveInResult});
    inOp.SetAttribute(OP_ATTR_PREFIX + "shape", std::vector<int>{dim1, dim2});
    TileShape::Current().SetVecTile(newVecTileShape);

    // [A1,A2,T1,T2] to [A1,A2,T2,T1] or [A2,T1,T2] to [A2,T2,T1]
    tmpShape = moveInResult->shape;
    dim1 = (tmpShape.size() == NUM_VALUE_3) ? 1 : NUM_VALUE_2; // if input is 3 dims, dim1 = 1, otherwise dim1 = 2
    dim2 = (tmpShape.size() == NUM_VALUE_3) ? NUM_VALUE_2 : NUM_VALUE_3;
    // if input is 3 dims, dim2 = 2, otherwise dim2 = 3
    std::swap(tmpShape[dim1], tmpShape[dim2]);
    std::swap(newVecTileShape[dim1], newVecTileShape[dim2]);
    std::swap(outValidShapes[dim1], outValidShapes[dim2]);
    auto vnchwconvResult = std::make_shared<LogicalTensor>(function, self->Datatype(), tmpShape, outValidShapes);
    auto& convOp = function.AddOperation(Opcode::OP_TRANSPOSE_VNCHWCONV, {moveInResult}, {vnchwconvResult});
    convOp.SetAttribute(OP_ATTR_PREFIX + "shape", std::vector<int>{dim1, dim2});
    TileShape::Current().SetVecTile(newVecTileShape);

    // [A1,A2,T2,T1] to [A1,T2,A2,T1] or [A2,T2,T1] to [T2,A2,T1]
    tmpShape = vnchwconvResult->shape;
    dim1 = (tmpShape.size() == NUM_VALUE_3) ? 0 : 1;           // if input is 3 dims, dim1 = 0, otherwise dim1 = 1
    dim2 = (tmpShape.size() == NUM_VALUE_3) ? 1 : NUM_VALUE_2; // if input is 3 dims, dim2 = 1, otherwise dim2 = 2
    std::swap(tmpShape[dim1], tmpShape[dim2]);
    auto& outOp = function.AddOperation(Opcode::OP_TRANSPOSE_MOVEOUT, {vnchwconvResult}, {result});
    outOp.SetAttribute(OP_ATTR_PREFIX + "shape", std::vector<int>{dim1, dim2});
    TileShape::Current().SetVecTile(oldVecTileShapes);
}

// ---------------------------------------------------------------------------------------------
// 合轴（merged）路径
//
// 代价模型（为什么要合轴 / 什么时候不合轴）：
//   * 多级转置（MOVEIN/VNCHWCONV/MOVEOUT）每条指令只搬一个 tile，固定开销大，指令数随 tile 数
//     线性增长 —— 所以优先合轴，把多次 tile 转置压成一次。
//   * 合轴要插入一次 reshape，而 reshape 会物化中间 tensor：动态函数的 valid shape 可能远小于
//     shape，物化后的有效形状与实参对不上（实测出错值）—— 所以动态函数一律不做合轴。
//   * 2D 合轴把前段维并进 VNCHWCONV 的行方向，行方向越满（>=16 行）单条指令的搬运效率越高。
//   * 但末维很大时列方向本来就会切成很多 tile，再合并行方向的收益有限（4096 是经验阈值）。
// 一句话：这里所有形状判断都是在估算"指令数"，不是正确性前提；正确性前提只有 3 条
// （2D 形状可成立、内存连续、静态函数），其余都是收益权衡。
// ---------------------------------------------------------------------------------------------

// 合轴方式：None 表示走多级转置；TwoD/ND 表示压成 2D / N 维后做一次转置。
enum class MergeKind { None, TwoD, ND };

// 合轴方案：输入先 reshape 成 inputShape，再做 transposeDims 这一次转置，最后 reshape 回目标形状。
struct MergedTransposePlan {
    std::vector<int64_t> inputShape;
    std::vector<int64_t> vecTile;
    std::vector<SymbolicScalar> validShape;
    std::vector<int> transposeDims;
};

// 以 perm 的两维为界，把输入切成 pre / mid / after 三段（维序保持原样）。
struct AxisSegments {
    int64_t preElems = 1;
    int64_t midElems = 1;
    int64_t afterElems = 1;
    int64_t preTile = 1;
    int64_t midTile = 1;
    int64_t afterTile = 1;
    SymbolicScalar preValid = 1;
    SymbolicScalar midValid = 1;
    SymbolicScalar afterValid = 1;
    int preDims = 0;
    int midDims = 0;
    int afterDims = 0;
};

static AxisSegments SplitAxisSegments(const Tensor& operand, const std::vector<int>& perm,
                                      const std::vector<SymbolicScalar>& srcValid)
{
    AxisSegments seg;
    const auto& shape = operand.GetShape();
    const auto& vecTile = TileShape::Current().GetVecTile();
    for (int i = 0; i < (int)shape.size(); ++i) {
        if (i < perm[0]) {
            seg.preElems *= shape[i];
            seg.preTile *= vecTile[i];
            seg.preValid = seg.preValid * srcValid[i];
            seg.preDims++;
        } else if (i > perm[0] && i < perm[1]) {
            seg.midElems *= shape[i];
            seg.midTile *= vecTile[i];
            seg.midValid = seg.midValid * srcValid[i];
            seg.midDims++;
        } else if (i > perm[1]) {
            seg.afterElems *= shape[i];
            seg.afterTile *= vecTile[i];
            seg.afterValid = seg.afterValid * srcValid[i];
            seg.afterDims++;
        }
    }
    return seg;
}

// ---- 2D 合轴的门槛 ----------------------------------------------------------------
// 只有"转置最后两维"的形状，才存在 (A*T1, T2) 这种 2D 视图。
static bool IsLastTwoDimsTranspose(int shapeSize, const std::vector<int>& perm)
{
    return perm[0] == shapeSize - NUM_VALUE_2 && perm[1] == shapeSize - 1;
}

// 动态函数下的 b16：合并后的 tile 与 VNCHWCONV 的 16 元素行约束对不上，走多级更稳。
static bool IsUnsupported2DMergeDtype(const Tensor& operand)
{
    return !IsStaticFunction() && BytesOf(operand.GetStorage()->Datatype()) == NUM_VALUE_2;
}

// 被转置的首维 T1 要窄（2..15）：T1 自己就能填满 VNCHWCONV 的行块时，不必再并前段维。
static bool IsNarrowTransposedDim(int64_t extent) { return extent > 1 && extent < VNCHWCONV_REPEAT; }

// 合并收益：并进前段维后行数能填满一个 VNCHWCONV 行块；否则只在末维较小时才划算
// （末维 >= 4096 时列方向已切得足够细，合并行方向带来的指令数收益有限，4096 为经验阈值）。
static bool IsMergeProfitable(int64_t mergedRows, int64_t lastDim)
{
    return mergedRows >= VNCHWCONV_REPEAT || lastDim < MERGE_2D_SMALL_LAST_DIM;
}

// 2D 视图成立的前提：转置维之前恰好一段、之后没有维度，且末维 tile 至少一个 VNCHWCONV 行宽。
static bool IsSimple2DMergeShape(const AxisSegments& seg, int64_t lastDimTile)
{
    return seg.preDims == 1 && seg.afterDims == 0 && lastDimTile >= VNCHWCONV_REPEAT;
}

static bool TryBuild2DMergedPlan(const Tensor& operand, const AxisSegments& seg, const std::vector<int>& perm,
                                 const std::vector<SymbolicScalar>& srcValid, MergedTransposePlan& plan)
{
    const auto& shape = operand.GetShape();
    const auto& vecTile = TileShape::Current().GetVecTile();
    int dim1 = perm[0];
    int dim2 = perm[1];
    if (!IsLastTwoDimsTranspose((int)shape.size(), perm) || IsUnsupported2DMergeDtype(operand) ||
        !IsNarrowTransposedDim(shape[dim1]) || !IsMergeProfitable(seg.preElems * shape[dim1], shape[dim2]) ||
        !IsSimple2DMergeShape(seg, vecTile[dim2])) {
        return false;
    }
    plan.inputShape = {seg.preElems * shape[dim1], shape[dim2]};
    plan.vecTile = {seg.preTile * vecTile[dim1], vecTile[dim2]};
    plan.validShape = {seg.preValid * srcValid[dim1], srcValid[dim2]};
    plan.transposeDims = {0, 1};
    return true;
}

// ---- 通用合轴的门槛 ---------------------------------------------------------------
static bool HasMergeableBatchDims(const AxisSegments& seg)
{
    return seg.preDims > 1 || seg.midDims > 1 || seg.afterDims > 1;
}

static bool IsContiguousForMerge(const Tensor& operand) { return !TensorNeedsContiguousCopy(operand); }

// 通用合轴：[A1,T1,A2,T2,A3] -> [A1, T1, A2, T2, A3]，把相邻的 batch 维并成一段。
// 调用前已确认门槛全部满足，这里只负责填充方案。
static void FillNDPlan(const Tensor& operand, const std::vector<int>& perm, const std::vector<SymbolicScalar>& srcValid,
                       const AxisSegments& seg, MergedTransposePlan& plan)
{
    const auto& shape = operand.GetShape();
    const auto& vecTile = TileShape::Current().GetVecTile();
    plan.inputShape.clear();
    plan.vecTile.clear();
    plan.validShape.clear();
    if (seg.preDims > 0) {
        plan.inputShape.push_back(seg.preElems);
        plan.vecTile.push_back(seg.preTile);
        plan.validShape.push_back(seg.preValid);
        plan.transposeDims[0] -= (seg.preDims - 1);
        plan.transposeDims[1] -= (seg.preDims - 1);
    }
    plan.inputShape.push_back(shape[perm[0]]);
    plan.vecTile.push_back(vecTile[perm[0]]);
    plan.validShape.push_back(srcValid[perm[0]]);
    if (seg.midDims > 0) {
        plan.inputShape.push_back(seg.midElems);
        plan.vecTile.push_back(seg.midTile);
        plan.validShape.push_back(seg.midValid);
        plan.transposeDims[1] -= (seg.midDims - 1);
    }
    plan.inputShape.push_back(shape[perm[1]]);
    plan.vecTile.push_back(vecTile[perm[1]]);
    plan.validShape.push_back(srcValid[perm[1]]);
    if (seg.afterDims > 0) {
        plan.inputShape.push_back(seg.afterElems);
        plan.vecTile.push_back(seg.afterTile);
        plan.validShape.push_back(seg.afterValid);
    }
}

// 唯一的决策点：先试 2D（最省），再试通用合轴，都不满足就走多级转置。
static MergeKind SelectMergeKind(const Tensor& operand, const std::vector<int>& perm,
                                 const std::vector<SymbolicScalar>& srcValid, MergedTransposePlan& plan)
{
    AxisSegments seg = SplitAxisSegments(operand, perm, srcValid);
    plan.transposeDims = perm;
    if (TryBuild2DMergedPlan(operand, seg, perm, srcValid, plan)) {
        return MergeKind::TwoD;
    }
    if (!HasMergeableBatchDims(seg) || !IsContiguousForMerge(operand) || !IsStaticFunction()) {
        return MergeKind::None;
    }
    FillNDPlan(operand, perm, srcValid, seg, plan);
    return MergeKind::ND;
}

// 动态函数下 2D 合轴：2D 转置后切出 3D 尾块再做一次转置，避免输入拷贝被按列拆成多段。
static Tensor TransposeMerged2DDynamic(const Tensor& self, const MergedTransposePlan& plan,
                                       const std::vector<int>& perm, const std::vector<SymbolicScalar>& srcValid,
                                       const std::vector<SymbolicScalar>& resultValid,
                                       const std::vector<int64_t>& resultShape, const VecTile& outerVecTile,
                                       const Tensor& tmpOutput)
{
    // 注意用调用方原始的 perm（而非合轴后重写过的 plan.transposeDims）取行列。
    int64_t rows = self.GetShape()[perm[0]];
    int64_t cols = self.GetShape()[perm[1]];
    int64_t pre = plan.inputShape[0] / rows;
    std::vector<int64_t> tmp3DShape = {cols, pre, rows};
    // 3D 临时 tensor 必须沿用源 tensor 的动态 valid shape；用具体 full shape 会让非整除尾块按
    // 全量范围参与转置，尾部数据错乱。
    std::vector<SymbolicScalar> tmp3DValid = SymbolicScalar::FromConcrete(tmp3DShape);
    if (perm[0] > 0) {
        tmp3DValid = {srcValid[perm[1]], srcValid[perm[0] - 1], srcValid[perm[0]]};
    }
    auto tmp3D = Reshape(tmpOutput, tmp3DShape, tmp3DValid);
    Tensor result(self.GetStorage()->Datatype(), resultShape);
    result.GetStorage()->UpdateDynValidShape(resultValid);
    // 尾块步长：按"每列（一次搬 pre*rows 个元素）占用的字节"反推一次能搬多少列，
    // 让单块落在 UB 预算内；行方向 b16 需要 16 元素对齐，其它类型 4 元素即可。
    int64_t dtypeBytes = BytesOf(self.GetStorage()->Datatype());
    int64_t alignRows = dtypeBytes == NUM_VALUE_2 ? VNCHWCONV_REPEAT : 4;
    int64_t rowSliceBytes = pre * AlignUp(rows, alignRows) * dtypeBytes;
    int64_t colTile = Merged2DTailBudgetBytes() / std::max<int64_t>(1, rowSliceBytes);
    colTile = std::max<int64_t>(1, std::min(cols, colTile) / VNCHWCONV_REPEAT * VNCHWCONV_REPEAT);
    TileShape::Current().SetVecTile({colTile, pre, rows});
    CALL(InnerTranspose, *Program::GetInstance().GetCurrentFunction(), tmp3D.GetStorage(), result.GetStorage(),
         std::vector<int>{0, 1});
    TileShape::Current().SetVecTile(outerVecTile);
    return result;
}

// 合轴路径：reshape -> 一次转置 -> reshape 回目标形状。
static Tensor TransposeViaMerge(const Tensor& self, const MergedTransposePlan& plan, const std::vector<int>& perm,
                                const std::vector<SymbolicScalar>& srcValid,
                                const std::vector<SymbolicScalar>& resultValid, const std::vector<int64_t>& resultShape,
                                const VecTile& outerVecTile, bool mergedAs2D)
{
    auto tmpInput = Reshape(self, plan.inputShape, plan.validShape);
    // 动态函数下合轴得到的 2D 转置：列 tile 一次覆盖完整 T2，否则输入拷贝会被按列拆成多段，
    // 非整除尾块的 assemble valid shape 会被放大到全量 extent。
    bool is2DDynamic = mergedAs2D && !IsStaticFunction();
    auto vecTile = plan.vecTile;
    if (is2DDynamic) {
        vecTile.back() = plan.inputShape.back();
    }
    TileShape::Current().SetVecTile(vecTile);
    auto tmpOutput = Transpose(tmpInput, plan.transposeDims);
    TileShape::Current().SetVecTile(outerVecTile);
    if (is2DDynamic) {
        return TransposeMerged2DDynamic(self, plan, perm, srcValid, resultValid, resultShape, outerVecTile, tmpOutput);
    }
    return Reshape(tmpOutput, resultShape, resultValid);
}

// ---------------------------------------------------------------------------------------------
// 入口
// ---------------------------------------------------------------------------------------------

// 退化路径：两维等长时转置是恒等变换；某一维长度为 1 且 batch 维被 tile 完整覆盖时等价于 reshape。
static bool TryReshapeDegenerateTranspose(const Tensor& self, const std::vector<int>& perm, Tensor& out)
{
    const auto& shape = self.GetShape();
    int dim0 = perm[0];
    int dim1 = perm[1];
    if (shape[dim0] != 1 && shape[dim1] != 1 && dim0 != dim1) {
        return false;
    }
    if (shape[dim0] == shape[dim1]) {
        out = self;
        return true;
    }
    const auto& vecTile = TileShape::Current().GetVecTile();
    bool tileCoversBatch = true;
    for (int d = 0; d < (int)shape.size(); ++d) {
        if (d != dim0 && d != dim1 && vecTile[d] < shape[d]) {
            tileCoversBatch = false;
            break;
        }
    }
    int otherDim = (shape[dim0] == 1) ? dim1 : dim0;
    bool otherDimNotCovered = vecTile[otherDim] < shape[otherDim];
    if (!tileCoversBatch || !otherDimNotCovered) {
        return false;
    }
    std::vector<int64_t> resultShape(shape);
    std::swap(resultShape[dim0], resultShape[dim1]);
    auto validShapes = self.GetStorage()->GetDynValidShape();
    if (validShapes.empty()) {
        validShapes = SymbolicScalar::FromConcrete(shape);
    }
    std::swap(validShapes[dim0], validShapes[dim1]);
    out = Reshape(self, resultShape, validShapes);
    return true;
}

Tensor Transpose(const Tensor& self, std::vector<int> perm)
{
    DECLARE_TRACER();
    CheckTensorFormat(self.GetStorage(), {TileOpFormat::TILEOP_NZ}, "Transpose");

    const auto& combinations = ConfigManager::Instance().GetOpSupportedInputDtypes(Opcode::OP_TRANSPOSE_VNCHWCONV);
    CheckTensorDataTypeCombination({self.GetDataType()}, combinations, "TRANSPOSE");
    CheckTensorDimRange(self.GetStorage(), 1, NUM_VALUE_5, "TRANSPOSE");
    CheckTensorShapeSize(self.GetStorage(), "TRANSPOSE");
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, perm.size() == NUM_VALUE_2)
        << "Transpose dim num should be 2."; // perm should be 2 dims
    int shapeSize = self.GetShape().size();
    if (perm[0] < 0) {
        perm[0] += shapeSize;
    }
    if (perm[1] < 0) {
        perm[1] += shapeSize;
    }
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, perm[0] < shapeSize && perm[0] >= 0) << "Transpose dim 0 is invalid.";
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, perm[1] < shapeSize && perm[1] >= 0) << "Transpose dim 1 is invalid.";

    std::sort(perm.begin(), perm.end());
    Tensor degenerate;
    if (TryReshapeDegenerateTranspose(self, perm, degenerate)) {
        return degenerate;
    }

    auto outerVecTile = TileShape::Current().GetVecTile();
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, (int)outerVecTile.size() == shapeSize)
        << "TileShape dim num should be the same as input.";
    // srcValid 保持输入维序（缺失时退化为具体 shape），源/目的维序的 valid shape 都从它派生。
    auto srcValid = self.GetStorage()->GetDynValidShape();
    if (srcValid.empty()) {
        srcValid = SymbolicScalar::FromConcrete(self.GetShape());
    }
    CHECK(VectorErrorCode::ERR_PARAM_INVALID, (int)srcValid.size() == shapeSize)
        << "ValidShape dim num should be the same as input.";

    std::vector<int64_t> resultShape(self.GetShape());
    std::swap(resultShape[perm[0]], resultShape[perm[1]]);
    auto resultValid = srcValid;
    std::swap(resultValid[perm[0]], resultValid[perm[1]]);

    MergedTransposePlan plan;
    MergeKind kind = SelectMergeKind(self, perm, srcValid, plan);
    if (kind == MergeKind::None) {
        // 多级转置：MOVEIN/VNCHWCONV/MOVEOUT 逐 tile 处理，不受形状/视图限制。
        Tensor result(self.GetStorage()->Datatype(), resultShape);
        result.GetStorage()->UpdateDynValidShape(resultValid);
        CALL(InnerTranspose, *Program::GetInstance().GetCurrentFunction(), self.GetStorage(), result.GetStorage(),
             perm);
        return result;
    }
    return TransposeViaMerge(self, plan, perm, srcValid, resultValid, resultShape, outerVecTile,
                             kind == MergeKind::TwoD);
}

void MoveOutOperationTileFunc(Function& function, const TileShape& tileShape,
                              const std::vector<LogicalTensorPtr>& iOperand,
                              const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    auto shape = op.GetVectorIntAttribute<int>(OP_ATTR_PREFIX + "shape");
    TiledInnerTranspose<TransposeOpType::TRANSPOSE_MOVEOUT>(function, tileShape, iOperand[0], oOperand[0], shape);
}

void MoveInOperationTileFunc(Function& function, const TileShape& tileShape,
                             const std::vector<LogicalTensorPtr>& iOperand,
                             const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    auto shape = op.GetVectorIntAttribute<int>(OP_ATTR_PREFIX + "shape");
    TiledInnerTranspose<TransposeOpType::TRANSPOSE_MOVEIN>(function, tileShape, iOperand[0], oOperand[0], shape);
}

void VnchwconvOperationTileFunc(Function& function, const TileShape& tileShape,
                                const std::vector<LogicalTensorPtr>& iOperand,
                                const std::vector<LogicalTensorPtr>& oOperand, const Operation& op)
{
    auto shape = op.GetVectorIntAttribute<int>(OP_ATTR_PREFIX + "shape");
    TiledInnerTranspose<TransposeOpType::TRANSPOSE_VNCHWCONV>(function, tileShape, iOperand[0], oOperand[0], shape);
}

REGISTER_OPERATION_TILED_FUNC(OP_TRANSPOSE_MOVEOUT, Opcode::OP_TRANSPOSE_MOVEOUT, MoveOutOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_TRANSPOSE_MOVEIN, Opcode::OP_TRANSPOSE_MOVEIN, MoveInOperationTileFunc);
REGISTER_OPERATION_TILED_FUNC(OP_TRANSPOSE_VNCHWCONV, Opcode::OP_TRANSPOSE_VNCHWCONV, VnchwconvOperationTileFunc);

} // namespace npu::tile_fwk
