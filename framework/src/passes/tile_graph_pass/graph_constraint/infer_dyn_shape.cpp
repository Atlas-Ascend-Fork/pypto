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
 * \file infer_dyn_shape.cpp
 * \brief
 */

#include <queue>
#include <optional>
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/operation/attribute.h"
#include "infer_dyn_shape.h"
#include "passes/pass_check/infer_dyn_shape_checker.h"
#include "passes/pass_log/pass_log.h"
#include "passes/pass_utils/infer_shape_utils.h"

#define MODULE_NAME "InferDynShape"

namespace npu {
namespace tile_fwk {
using RawPtr = RawSymbolicScalarPtr;

// 二元表达式（加减乘、min/max）节点恰好包含的操作数个数。
static constexpr size_t BINARY_EXPR_OPERAND_COUNT = 2;

// 从 SymbolicScalar 节点中提取立即数值。
static std::optional<int64_t> GetImmediate(const RawPtr& raw)
{
    if (raw != nullptr && raw->IsImmediate()) {
        return raw->GetImmediateValue();
    }
    return std::nullopt;
}

// 判断表达式节点是否为指定的 SymbolicScalar 操作码。
static bool IsOpcode(const RawPtr& raw, SymbolicOpcode opcode)
{
    return raw != nullptr && raw->IsExpression() && raw->GetExpressionOpcode() == opcode;
}

// 从二元 min/max 表达式中拆出立即数和另一个操作数。
static bool SplitExtremaWithImmediate(const RawPtr& raw, SymbolicOpcode opcode, int64_t immediate, RawPtr& other)
{
    if (!IsOpcode(raw, opcode)) {
        return false;
    }
    const auto& operands = raw->GetExpressionOperandList();
    if (operands.size() != BINARY_EXPR_OPERAND_COUNT) {
        return false;
    }
    if (GetImmediate(operands[0]) == immediate) {
        other = operands[1];
        return true;
    }
    if (GetImmediate(operands[1]) == immediate) {
        other = operands[0];
        return true;
    }
    return false;
}

// 匹配 min(value, upper)，其中 upper 必须是立即数。
static bool ExtractMinWithUpper(const RawPtr& raw, RawPtr& value, int64_t& upper)
{
    if (!IsOpcode(raw, SymbolicOpcode::T_MOP_MIN)) {
        return false;
    }
    const auto& operands = raw->GetExpressionOperandList();
    if (operands.size() != BINARY_EXPR_OPERAND_COUNT) {
        return false;
    }
    if (auto immediate = GetImmediate(operands[0]); immediate.has_value()) {
        upper = *immediate;
        value = operands[1];
        return true;
    }
    if (auto immediate = GetImmediate(operands[1]); immediate.has_value()) {
        upper = *immediate;
        value = operands[0];
        return true;
    }
    return false;
}

// 化简正确性依赖以下三个前提，均由 assume_divisible API 契约和 Tile Graph 展开方式共同保证：
// 1. 用户通过 assume_divisible(vm, T) 显式声明 vm % T == 0；
// 2. 循环展开采用 ceil 除法 count = ceil(vm / T)，当 vm % T == 0 时 count = vm / T；
// 3. 循环条件 idx ∈ [0, count)，即 idx * T < vm，因此 vm - idx * T >= T > 0。
// unroll 展开不破坏该前提：展开范围向下取整到 unroll 倍数 (count / unroll * unroll)，
// 保证展开内每个 tile 对应的 idx + n < count，即 vm - (idx + n) * T >= T。
// 综上，每次执行的迭代中 vm - idx * T >= T，min(vm - idx * T, cap) = cap（cap <= T 的倍数），
// cap 恒等于实际值，化简为常量是安全的。
// 若用户未注册 assume_divisible，SimplifyAllValidShapes 的门禁会跳过化简，
// 避免 cap 不等于实际值时错误化简。

// 递归查询表达式是否能够被 divisor 整除。
// 该查询同时消费 Program 中登记的假设和加减乘的结构信息。
static constexpr size_t MAX_DIVISIBLE_RECURSION_DEPTH = 64;

static bool IsExprDivisible(const Function& function, const SymbolicScalar& expr, int64_t divisor, size_t depth = 0)
{
    if (depth >= MAX_DIVISIBLE_RECURSION_DEPTH) {
        return false;
    }
    if (Program::GetInstance().IsKnownDivisible(expr, divisor)) {
        return true;
    }
    const auto raw = expr.Raw();
    if (!raw || !raw->IsExpression()) {
        return false;
    }
    const auto opcode = raw->GetExpressionOpcode();
    const auto& operands = raw->GetExpressionOperandList();
    if ((opcode == SymbolicOpcode::T_BOP_ADD || opcode == SymbolicOpcode::T_BOP_SUB) &&
        operands.size() == BINARY_EXPR_OPERAND_COUNT) {
        return IsExprDivisible(function, SymbolicScalar(operands[0]), divisor, depth + 1) &&
               IsExprDivisible(function, SymbolicScalar(operands[1]), divisor, depth + 1);
    }
    if (opcode == SymbolicOpcode::T_BOP_MUL && operands.size() == BINARY_EXPR_OPERAND_COUNT) {
        return IsExprDivisible(function, SymbolicScalar(operands[0]), divisor, depth + 1) ||
               IsExprDivisible(function, SymbolicScalar(operands[1]), divisor, depth + 1);
    }
    return false;
}

// 证明 value - accumulatedOffset 至少包含一个完整 tile。
// View valid shape 表达式形如 min(max(min(vm - idx*T, cap) - offset, 0), tile)，
// 其中 cap = (n+1)*T，offset = n*T，n 为展开内第 n 个子 tile。
// 根据上述化简前提，循环条件保证 vm - idx*T >= T，因此：
//   - 展开内第 n 个子 tile 满足 vm - (idx+n)*T >= T，即 vm - idx*T >= (n+1)*T = cap；
//   - min(vm - idx*T, cap) = cap（cap 为上界且实际值 >= cap）；
//   - cap - accumulatedOffset = (n+1)*T - n*T = T >= tile。
// 沿表达式树递归下沉 offset，不假设当前 cap 属于哪一层 View。
// 递归深度上限 MAX_DIVISIBLE_RECURSION_DEPTH 防止深表达式栈溢出。
static bool IsFullTileAfterOffset(const Function& function, const RawPtr& raw, int64_t accumulatedOffset, int64_t tile,
                                  size_t depth = 0);

static std::optional<bool> TryTraverseOffsetExpression(const Function& function, const RawPtr& raw,
                                                       int64_t accumulatedOffset, int64_t tile, size_t depth)
{
    if (raw == nullptr || !raw->IsExpression()) {
        return std::nullopt;
    }
    const auto opcode = raw->GetExpressionOpcode();
    const auto& operands = raw->GetExpressionOperandList();
    if (operands.size() != BINARY_EXPR_OPERAND_COUNT) {
        return std::nullopt;
    }
    if (opcode == SymbolicOpcode::T_BOP_SUB) {
        const auto delta = GetImmediate(operands[1]);
        if (delta.has_value() && *delta >= 0) {
            return IsFullTileAfterOffset(function, operands[0], accumulatedOffset + *delta, tile, depth + 1);
        }
    }
    if (opcode == SymbolicOpcode::T_BOP_ADD) {
        for (size_t valueIndex = 0; valueIndex < operands.size(); ++valueIndex) {
            const auto delta = GetImmediate(operands[1 - valueIndex]);
            if (delta.has_value() && *delta <= 0) {
                return IsFullTileAfterOffset(function, operands[valueIndex], accumulatedOffset - *delta, tile,
                                             depth + 1);
            }
        }
    }
    return std::nullopt;
}

static std::optional<bool> TryValidateMinCap(const Function& function, const RawPtr& raw, int64_t accumulatedOffset,
                                             int64_t tile, size_t depth)
{
    RawPtr value;
    int64_t cap = 0;
    if (!ExtractMinWithUpper(raw, value, cap)) {
        return std::nullopt;
    }
    const int64_t remainingCap = cap - accumulatedOffset;
    if (remainingCap < tile || remainingCap % tile != 0) {
        APASS_LOG_DEBUG_F(Elements::Function,
                          "[AssumeDivisible][CapFail] func=%s expr=%s cap=%ld offset=%ld tile=%ld remaining=%ld",
                          function.GetMagicName().c_str(), SymbolicScalar(raw).Dump().c_str(), cap, accumulatedOffset,
                          tile, remainingCap);
        return false;
    }
    return IsFullTileAfterOffset(function, value, accumulatedOffset, tile, depth + 1);
}

static bool ValidateFullTileSource(const Function& function, const RawPtr& raw, int64_t accumulatedOffset, int64_t tile)
{
    const bool divisible = IsExprDivisible(function, SymbolicScalar(raw), tile);
    const bool aligned = accumulatedOffset % tile == 0;
    if (!divisible || !aligned) {
        APASS_LOG_DEBUG_F(Elements::Function,
                          "[AssumeDivisible][SourceFail] func=%s expr=%s tile=%ld divisible=%d offset=%ld aligned=%d",
                          function.GetMagicName().c_str(), SymbolicScalar(raw).Dump().c_str(), tile, divisible ? 1 : 0,
                          accumulatedOffset, aligned ? 1 : 0);
    }
    return divisible && aligned;
}

static bool IsFullTileAfterOffset(const Function& function, const RawPtr& raw, int64_t accumulatedOffset, int64_t tile,
                                  size_t depth)
{
    if (raw == nullptr || tile <= 0 || depth >= MAX_DIVISIBLE_RECURSION_DEPTH) {
        return false;
    }

    if (auto result = TryTraverseOffsetExpression(function, raw, accumulatedOffset, tile, depth); result.has_value()) {
        return *result;
    }
    if (auto result = TryValidateMinCap(function, raw, accumulatedOffset, tile, depth); result.has_value()) {
        return *result;
    }
    // 到达最内层源表达式（如 vm 或 vm - idx*T）后，由整除假设和累计 View offset 建立 tile 对齐的下界。
    return ValidateFullTileSource(function, raw, accumulatedOffset, tile);
}

// 将已经化简的 View/Slice output valid shape 同步回非空 attr，避免后续 InferShape 恢复旧表达式。
static void SyncViewValidShapeAttribute(Operation& op)
{
    if (op.GetOpcode() != Opcode::OP_VIEW && op.GetOpcode() != Opcode::OP_SLICE) {
        return;
    }
    auto viewAttr = std::dynamic_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
    if (viewAttr == nullptr) {
        return;
    }
    auto& validShape = viewAttr->GetToDynValidShape();
    if (validShape.empty() || op.GetOOperands().empty()) {
        return;
    }
    const auto& output = op.GetOOperands()[0];
    if (output == nullptr || output->GetDynValidShape().empty()) {
        return;
    }
    // View 构造时会将同一份 valid shape 同时写入 output 和 attr。
    // output 化简完成后同步更新 attr，避免 ViewInferFunc 恢复化简前的表达式。
    validShape = output->GetDynValidShape();
}

// CopyInInferFunc 优先使用 output tensor 的 valid shape，但仅在 toDynValidShape 为空时回填 attr。
// 主动覆盖非空旧 attr，保证后续直接读取 CopyOpAttribute 的 codegen/pass 与 tensor 看到同一结果。
static void SyncCopyInValidShapeAttribute(Operation& op)
{
    if (op.GetOpcode() != Opcode::OP_COPY_IN) {
        return;
    }
    auto copyAttr = std::dynamic_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
    if (copyAttr == nullptr || op.GetOOperands().empty()) {
        return;
    }
    const auto& output = op.GetOOperands()[0];
    if (output == nullptr || output->GetDynValidShape().empty()) {
        return;
    }
    copyAttr->SetToDynValidShape(OpImmediate::Specified(output->GetDynValidShape()));
}

SymbolicScalar InferDynShape::SimplifyValidShapeWithAssumptions(const Function& function,
                                                                const SymbolicScalar& shape) const
{
    const auto simplified = shape.Simplify();
    RawPtr clampedValue;
    int64_t tile = 0;
    if (!ExtractMinWithUpper(simplified.Raw(), clampedValue, tile) || tile <= 0) {
        return simplified;
    }
    RawPtr value = clampedValue;
    RawPtr nonNegativeValue;
    if (SplitExtremaWithImmediate(clampedValue, SymbolicOpcode::T_MOP_MAX, 0, nonNegativeValue)) {
        value = nonNegativeValue;
    }

    // 递归下沉各层 View 的立即数 offset，并检查嵌套 min cap。
    // 同时支持 min(max(E, 0), tile) 和显式的 min(E, tile)，
    // 覆盖直接 tile 以及父/子/孙 View。无法匹配的表达式保持不变。
    if (!IsFullTileAfterOffset(function, value, 0, tile)) {
        return simplified;
    }
    APASS_LOG_DEBUG_F(Elements::Function, "[AssumeDivisible][Simplified] func=%s expr=%s result=%ld",
                      function.GetMagicName().c_str(), simplified.Dump().c_str(), tile);
    return SymbolicScalar(tile);
}

// 只遍历 op 的 output tensor 做一次化简；相关 attr 只同步 output 的结果，不重复化简。
void InferDynShape::SimplifyAllValidShapes(Function& function)
{
    if (Program::GetInstance().GetDivisibleAssumptions().empty()) {
        return;
    }
    for (auto& op : function.Operations(false)) {
        for (const auto& outTensor : op.GetOOperands()) {
            if (outTensor == nullptr) {
                continue;
            }
            auto& validShape = outTensor->GetDynValidShape();
            if (validShape.empty()) {
                continue;
            }
            for (auto& dim : validShape) {
                dim = SimplifyValidShapeWithAssumptions(function, dim);
            }
        }
        SyncViewValidShapeAttribute(op);
        SyncCopyInValidShapeAttribute(op);
    }
}
Status InferDynShape::PostCheck(Function& function)
{
    InferDynShapeChecker checker;
    return checker.DoPostCheck(function);
}

Status InferDynShape::RunOnFunction(Function& function)
{
    // 遍历每一个op，调用对应的infershape函数
    // 遍历顺序，按照入度解依赖
    APASS_LOG_INFO_F(Elements::Function, "===> Start InferDynShape.");
    // InferShape 会把所有 shape 统一 normalize 成动态表达式 (即便原本是静态),
    // 之后下游 (如 OoOSchedule 的 dualdst 融合) 比较 validShape 时静态信息已丢失。
    // 在这一步之前先把 OP_L0C_COPY_UB 的 UB 输出 validShape 快照到 op 属性,
    // 供 dualdst_fuse 在融合候选判定阶段优先使用。本函数与 InferShape 完全解耦,
    // 不影响原有逻辑;若 op 输出 validShape 含动态成分则直接跳过 (回退 dyn 比较)。
    SimplifyAllValidShapes(function);
    RecordStaticValidShapeOnL0CCopyUB(function);
    if (InferShapeUtils::InferShape(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "InferShape failed; Please check the InferShape method.");
        return FAILED;
    }
    APASS_LOG_DEBUG_F(Elements::Function, "Dump: %s", function.Dump().c_str());
    APASS_LOG_INFO_F(Elements::Function, "===> End InferDynShape.");
    return SUCCESS;
}

// 在 InferDynShape::RunOnFunction 中, InferShape 转换前调用一次。
// 仅处理 OP_L0C_COPY_UB: 把其唯一输出 (UB tensor) 的 validShape (此时仍是静态)
// 以 vector<int64_t> 形式写到 op 属性 OpAttributeKey::staticValidShape。
// 任何一维含动态成分时整体跳过, 让下游回退到 GetDynValidShape。
void InferDynShape::RecordStaticValidShapeOnL0CCopyUB(Function& function)
{
    // 不强制排序: 仅做 op 属性快照, 与遍历顺序无关, 避免对 InferShape 的 op 顺序产生副作用。
    for (auto& op : function.Operations(false)) {
        if (op.GetOpcode() != Opcode::OP_L0C_COPY_UB)
            continue;
        if (op.GetOOperands().empty())
            continue;
        auto ubOut = op.GetOutputOperand(0);
        if (ubOut == nullptr)
            continue;
        const auto& valid = ubOut->GetDynValidShape();
        if (valid.empty())
            continue;
        std::vector<int64_t> staticVals;
        staticVals.reserve(valid.size());
        bool allConcrete = true;
        for (const auto& s : valid) {
            if (!s.ConcreteValid()) {
                allConcrete = false;
                break;
            }
            staticVals.push_back(s.Concrete());
        }
        if (!allConcrete)
            continue;
        op.SetAttribute(OpAttributeKey::staticValidShape, staticVals);
    }
}
} // namespace tile_fwk
} // namespace npu
