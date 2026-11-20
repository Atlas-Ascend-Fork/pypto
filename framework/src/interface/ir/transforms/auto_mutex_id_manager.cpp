/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ir/transforms/passes.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ir/debug_info.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/op_registry.h"
#include "ir/span.h"
#include "ir/transforms/base/mutator.h"
#include "ir/transforms/base/visitor.h"
#include "ir/transforms/structural_comparison.h"
#include "ir/type.h"
#include "pypto_pro/error.h"

namespace pypto {
namespace ir {

using npu::tile_fwk::ExternalError;

namespace {

// AUTO mutex-ID planning:
// collect Tile records -> build containment and SSA count constraints
// -> solve mutex-ID sets -> rewrite AUTO bindings and SSA types.
constexpr int kMutexIdCount = 32;
constexpr uint32_t kAllMutexIds = std::numeric_limits<uint32_t>::max();
constexpr size_t kBacktracksPerActiveNode = 32;

int MaskToIdCount(uint32_t mask)
{
    int count = 0;
    while (mask != 0) {
        mask &= mask - 1;
        ++count;
    }
    return count;
}

std::vector<int> MaskToIds(uint32_t mask)
{
    std::vector<int> result;
    for (int id = 0; mask != 0; ++id) {
        if ((mask & 1U) != 0) {
            result.push_back(id);
        }
        mask >>= 1U;
    }
    return result;
}

std::string FormatIds(const std::vector<int>& ids)
{
    std::ostringstream os;
    os << '[';
    for (size_t index = 0; index < ids.size(); ++index) {
        if (index != 0) {
            os << ", ";
        }
        os << ids[index];
    }
    os << ']';
    return os.str();
}

std::string FormatMask(uint32_t mask) { return FormatIds(MaskToIds(mask)); }

std::string HexRange(int64_t start, int64_t end)
{
    std::ostringstream os;
    os << "[0x" << std::hex << start << ", 0x" << end << ")";
    return os.str();
}

using SsaMutexCounts = std::unordered_map<const Var*, std::vector<uint32_t>>;

// Expanding an AUTO ID row changes descriptor types. Consume the already-solved
// counts to rebuild definitions and let IRMutator remap their uses, including
// yield/break/continue operands. No merge analysis is needed during rewriting.
class PlaceholderRewriter final : public IRMutator {
public:
    PlaceholderRewriter(const std::vector<std::pair<VarPtr, std::vector<int>>>& replacements,
                        const SsaMutexCounts& ssaMutexCounts, IRDebugInfoPtr debugInfo)
        : ssaMutexCounts_(ssaMutexCounts), debugInfo_(std::move(debugInfo))
    {
        for (const auto& [placeholder, values] : replacements) {
            placeholderValues_[placeholder.get()] = values;
        }
    }

protected:
    using IRMutator::VisitExpr_;
    using IRMutator::VisitStmt_;

    ExprPtr VisitExpr_(const MakeTuplePtr& op) override
    {
        // Bindings and constant-propagated uses may share one tuple. Keep that
        // identity because dynamic-index codegen keys backing arrays by MakeTuple*.
        auto cached = tupleRemap_.find(op.get());
        if (cached != tupleRemap_.end()) {
            return cached->second;
        }
        if (op->elements_.size() == 1) {
            if (auto placeholder = As<Var>(op->elements_.front())) {
                auto it = placeholderValues_.find(placeholder.get());
                if (it != placeholderValues_.end()) {
                    std::vector<ExprPtr> elements;
                    elements.reserve(it->second.size());
                    for (int mutexId : it->second) {
                        elements.push_back(
                            std::make_shared<const ConstInt>(mutexId, DataType::INT64, placeholder->span_));
                    }
                    auto result = std::make_shared<const MakeTuple>(std::move(elements), op->span_);
                    tupleRemap_.emplace(op.get(), result);
                    return result;
                }
            }
        }
        auto [elements, changed] = VisitExprList(op->elements_);
        if (!changed) {
            tupleRemap_.emplace(op.get(), op);
            return op;
        }
        auto result = std::make_shared<const MakeTuple>(std::move(elements), op->span_);
        CopyTupleTypeInfo(op->GetType(), result->GetType());
        tupleRemap_.emplace(op.get(), result);
        return result;
    }

    ExprPtr VisitExpr_(const CallPtr& op) override
    {
        auto [arguments, changed] = VisitExprList(op->args_);
        if (!changed) {
            return op;
        }

        if (op->name_ == "block.make_mutex_tile" || op->name_ == "block.reinterpret" || op->name_ == "block.subview") {
            // The default Call mutator preserves the old return type.
            return OpRegistry::GetInstance().Create(op->name_, arguments, op->kwargs_, op->span_);
        }
        return std::make_shared<const Call>(op->name_, std::move(arguments), op->kwargs_, op->GetType(), op->span_);
    }

    StmtPtr VisitStmt_(const AssignStmtPtr& op) override
    {
        auto newValue = VisitExpr(op->value_);
        auto targets = op->targets_;
        if (targets.size() == 1) {
            targets.front() = RemapDefinition(targets.front(), newValue->GetType());
        }
        if (targets != op->targets_ || newValue.get() != op->value_.get()) {
            return std::make_shared<const AssignStmt>(std::move(targets), std::move(newValue), op->span_);
        }
        return op;
    }

    StmtPtr VisitStmt_(const IfStmtPtr& op) override
    {
        auto condition = VisitExpr(op->condition_);
        auto thenBody = VisitStmt(op->thenBody_);
        std::optional<StmtPtr> elseBody;
        if (op->elseBody_.has_value()) {
            elseBody = VisitStmt(*op->elseBody_);
        }

        auto returnVars = RebuildReturnVars(op->returnVars_);
        bool changed = condition.get() != op->condition_.get() || thenBody.get() != op->thenBody_.get() ||
                       (op->elseBody_.has_value() && elseBody->get() != op->elseBody_->get()) ||
                       returnVars != op->returnVars_;
        if (!changed) {
            return op;
        }
        return std::make_shared<const IfStmt>(std::move(condition), std::move(thenBody), std::move(elseBody),
                                              std::move(returnVars), op->span_);
    }

    StmtPtr VisitStmt_(const ForStmtPtr& op) override
    {
        auto loopVar = As<Var>(VisitExpr(op->loopVar_));
        auto start = VisitExpr(op->start_);
        auto stop = VisitExpr(op->stop_);
        auto step = VisitExpr(op->step_);
        auto iterArgs = RebuildIterArgs(op->iterArgs_);
        auto body = VisitStmt(op->body_);
        EraseIterMappings(op->iterArgs_);
        auto returnVars = RebuildReturnVars(op->returnVars_);
        if (loopVar.get() == op->loopVar_.get() && start.get() == op->start_.get() && stop.get() == op->stop_.get() &&
            step.get() == op->step_.get() && iterArgs == op->iterArgs_ && body.get() == op->body_.get() &&
            returnVars == op->returnVars_) {
            return op;
        }
        return std::make_shared<const ForStmt>(std::move(loopVar), std::move(start), std::move(stop), std::move(step),
                                               std::move(iterArgs), std::move(body), std::move(returnVars), op->span_,
                                               op->attrs_);
    }

    StmtPtr VisitStmt_(const WhileStmtPtr& op) override
    {
        // The condition may use iterVars, so establish their mappings first.
        auto iterArgs = RebuildIterArgs(op->iterArgs_);
        auto condition = VisitExpr(op->condition_);
        auto body = VisitStmt(op->body_);
        EraseIterMappings(op->iterArgs_);
        // While exits use their own solved groups, independent of header types.
        auto returnVars = RebuildReturnVars(op->returnVars_);
        if (iterArgs == op->iterArgs_ && condition.get() == op->condition_.get() && body.get() == op->body_.get() &&
            returnVars == op->returnVars_) {
            return op;
        }
        return std::make_shared<const WhileStmt>(std::move(condition), std::move(iterArgs), std::move(body),
                                                 std::move(returnVars), op->span_);
    }

private:
    TypePtr RebuildMutexCounts(const TypePtr& type, const std::vector<uint32_t>& counts, size_t& index)
    {
        if (auto tuple = As<TupleType>(type)) {
            std::vector<TypePtr> elements;
            bool changed = false;
            for (const auto& element : tuple->types_) {
                auto rebuilt = RebuildMutexCounts(element, counts, index);
                changed |= rebuilt.get() != element.get();
                elements.push_back(std::move(rebuilt));
            }
            if (!changed) {
                return type;
            }
            auto result = std::make_shared<const TupleType>(std::move(elements));
            CopyTupleTypeInfo(type, result);
            return result;
        }
        if (auto tile = As<MutexTileType>(type); tile != nullptr && tile->mutexCount_ != 0) {
            uint32_t count = counts.at(index++);
            if (count != 0 && count != tile->mutexCount_) {
                // Equal counts do not imply equal dtype/shape, e.g. after reinterpret.
                return std::make_shared<const MutexTileType>(*tile, count);
            }
        }
        return type;
    }

    VarPtr RemapDefinition(const VarPtr& oldVar, TypePtr type)
    {
        if (type.get() == oldVar->GetType().get() || structural_equal(type, oldVar->GetType())) {
            return oldVar;
        }
        auto newVar = std::make_shared<const Var>(oldVar->name_, type, oldVar->span_);
        // SSA identity is the old Var pointer; equal names in nested scopes are unrelated.
        var_remap_[oldVar.get()] = newVar;
        return newVar;
    }

    VarPtr RemapControlFlowDefinition(const VarPtr& oldVar)
    {
        auto type = oldVar->GetType();
        auto counts = ssaMutexCounts_.find(oldVar.get());
        if (counts != ssaMutexCounts_.end()) {
            size_t index = 0;
            type = RebuildMutexCounts(type, counts->second, index);
        }
        return RemapDefinition(oldVar, std::move(type));
    }

    std::vector<IterArgPtr> RebuildIterArgs(const std::vector<IterArgPtr>& oldArgs)
    {
        std::vector<IterArgPtr> result;
        result.reserve(oldArgs.size());
        for (const auto& oldArg : oldArgs) {
            auto init = VisitExpr(oldArg->initValue_);
            // Rebuild the definition explicitly: the base mutator only rewrites initValue.
            auto var = RemapControlFlowDefinition(oldArg->iterVar_);
            if (init.get() == oldArg->initValue_.get() && var.get() == oldArg->iterVar_.get()) {
                result.push_back(oldArg);
            } else {
                result.push_back(std::make_shared<const IterArg>(std::move(var), std::move(init)));
            }
        }
        return result;
    }

    std::vector<VarPtr> RebuildReturnVars(const std::vector<VarPtr>& oldVars)
    {
        std::vector<VarPtr> result;
        result.reserve(oldVars.size());
        for (size_t index = 0; index < oldVars.size(); ++index) {
            result.push_back(RemapControlFlowDefinition(oldVars[index]));
        }
        return result;
    }

    void EraseIterMappings(const std::vector<IterArgPtr>& arguments)
    {
        for (const auto& argument : arguments) {
            var_remap_.erase(argument->iterVar_.get());
        }
    }

    void CopyTupleTypeInfo(const TypePtr& oldType, const TypePtr& newType)
    {
        if (debugInfo_ == nullptr) {
            return;
        }
        auto oldTuple = As<TupleType>(oldType);
        auto newTuple = As<TupleType>(newType);
        if (oldTuple == nullptr || newTuple == nullptr) {
            return;
        }
        const auto* info = debugInfo_->GetTupleTypeInfo(oldTuple.get());
        if (info != nullptr) {
            debugInfo_->RegisterTupleTypeInfo(newTuple, *info);
        }
    }

    std::unordered_map<const Var*, std::vector<int>> placeholderValues_;
    std::unordered_map<const MakeTuple*, MakeTuplePtr> tupleRemap_;
    const SsaMutexCounts& ssaMutexCounts_;
    IRDebugInfoPtr debugInfo_;
};

class DisjointSet {
public:
    explicit DisjointSet(size_t size) : parent_(size), rank_(size) { std::iota(parent_.begin(), parent_.end(), 0); }

    size_t Add()
    {
        size_t index = parent_.size();
        parent_.push_back(index);
        rank_.push_back(0);
        return index;
    }

    size_t Find(size_t value)
    {
        if (parent_[value] != value) {
            parent_[value] = Find(parent_[value]);
        }
        return parent_[value];
    }

    void Join(size_t left, size_t right)
    {
        left = Find(left);
        right = Find(right);
        if (left == right) {
            return;
        }
        if (rank_[left] < rank_[right]) {
            std::swap(left, right);
        }
        parent_[right] = left;
        if (rank_[left] == rank_[right]) {
            ++rank_[left];
        }
    }

private:
    std::vector<size_t> parent_;
    std::vector<size_t> rank_;
};

using MutexCountNodeId = size_t;
using MutexCountGroupId = size_t;

// Each nonzero-count MutexTileType leaf has one node, regardless of its ID count.
// Nodes in one group require equal counts, not identical mutex IDs. Tuple leaves
// are flattened in order; the input IR must have compatible positional layouts.
// One traversal records equality edges; union-find makes them transitive without
// repeatedly propagating candidate ID sets as insert_sync.cpp does.
class SsaMutexCountAnalysis final : public IRVisitor {
public:
    explicit SsaMutexCountAnalysis(const ProgramPtr& program) { VisitProgram(program); }

    MutexCountGroupId GetTileCountGroupId(const ExprPtr& tile)
    {
        // Each collected Tile has non-empty mutex IDs and contributes one count node.
        const auto& nodes = GetOrCreateCountNodes(tile);
        return countSets_.Find(nodes.front());
    }

    SsaMutexCounts BuildSsaMutexCounts(const std::unordered_map<MutexCountGroupId, uint32_t>& countsByGroup)
    {
        SsaMutexCounts result;
        for (const auto& [expr, nodes] : countNodesByExpr_) {
            // Only SSA definitions consume the counts during type rewriting.
            if (expr->GetKind() != ObjectKind::Var || nodes.empty()) {
                continue;
            }
            auto& counts = result[static_cast<const Var*>(expr)];
            counts.reserve(nodes.size());
            for (MutexCountNodeId node : nodes) {
                auto it = countsByGroup.find(countSets_.Find(node));
                // Zero means no allocation result: preserve the original count.
                counts.push_back(it == countsByGroup.end() ? 0 : it->second);
            }
        }
        return result;
    }

private:
    using IRVisitor::VisitExpr_;
    using IRVisitor::VisitStmt_;

    static size_t CountMutexTileSlots(const TypePtr& type)
    {
        if (auto tuple = As<TupleType>(type)) {
            size_t count = 0;
            for (const auto& element : tuple->types_) {
                count += CountMutexTileSlots(element);
            }
            return count;
        }
        auto tile = As<MutexTileType>(type);
        return tile != nullptr && tile->mutexCount_ != 0 ? 1 : 0;
    }

    const std::vector<MutexCountNodeId>& GetOrCreateCountNodes(const ExprPtr& expr)
    {
        auto [it, inserted] = countNodesByExpr_.try_emplace(expr.get());
        if (inserted) {
            size_t count = CountMutexTileSlots(expr->GetType());
            it->second.reserve(count);
            for (size_t i = 0; i < count; ++i) {
                it->second.push_back(countSets_.Add());
            }
        }
        return it->second;
    }

    void UnifyCountSlots(const std::vector<MutexCountNodeId>& left, const std::vector<MutexCountNodeId>& right,
                         size_t rightOffset, size_t count)
    {
        // Scalars, None/poison values and zero-count Tiles have no mutex slots.
        if (left.empty() || count == 0) {
            return;
        }
        for (size_t i = 0; i < count; ++i) {
            countSets_.Join(left[i], right[rightOffset + i]);
        }
    }

    void SetConstrainEqualCounts(const ExprPtr& left, const ExprPtr& right)
    {
        const auto& leftNodes = GetOrCreateCountNodes(left);
        const auto& rightNodes = GetOrCreateCountNodes(right);
        UnifyCountSlots(leftNodes, rightNodes, 0, rightNodes.size());
    }

    void MergeInputs(const std::vector<VarPtr>& targets, const std::vector<ExprPtr>& values)
    {
        for (size_t i = 0; i < targets.size(); ++i) {
            SetConstrainEqualCounts(targets[i], values[i]);
        }
    }

    void ConnectTupleElements(const MakeTuplePtr& tuple)
    {
        const auto& tupleNodes = GetOrCreateCountNodes(tuple);
        if (tupleNodes.empty()) {
            return;
        }
        size_t offset = 0;
        for (const auto& element : tuple->elements_) {
            const auto& elementNodes = GetOrCreateCountNodes(element);
            UnifyCountSlots(elementNodes, tupleNodes, offset, elementNodes.size());
            offset += elementNodes.size();
        }
    }

    void VisitExpr(const ExprPtr& op) override
    {
        // Creating count nodes does not imply that an expression was visited:
        // loop inputs can be connected before traversing their definitions.
        if (visitedExprs_.insert(op.get()).second) {
            IRVisitor::VisitExpr(op);
        }
    }

    void VisitExpr_(const MakeTuplePtr& op) override
    {
        IRVisitor::VisitExpr_(op);
        ConnectTupleElements(op);
    }

    void VisitExpr_(const GetItemExprPtr& op) override
    {
        IRVisitor::VisitExpr_(op);
        auto tuple = As<TupleType>(op->value_->GetType());
        if (tuple == nullptr) {
            return;
        }
        const auto& baseNodes = GetOrCreateCountNodes(op->value_);
        const auto& resultNodes = GetOrCreateCountNodes(op);
        auto index = As<ConstInt>(op->slice_);
        size_t offset = 0;
        for (size_t i = 0; i < tuple->types_.size(); ++i) {
            size_t count = CountMutexTileSlots(tuple->types_[i]);
            if (index == nullptr || index->value_ == static_cast<int64_t>(i)) {
                UnifyCountSlots(resultNodes, baseNodes, offset, count);
                if (index != nullptr) {
                    break;
                }
            }
            offset += count;
        }
    }

    void VisitExpr_(const CallPtr& op) override
    {
        IRVisitor::VisitExpr_(op);
        if (op->name_ == "block.reinterpret" || op->name_ == "block.subview") {
            SetConstrainEqualCounts(op, op->args_[0]);
        }
    }

    void VisitStmt_(const AssignStmtPtr& op) override
    {
        IRVisitor::VisitStmt_(op);
        // Parser tuple unpacking emits one GetItem assignment per target.
        if (op->targets_.size() == 1) {
            SetConstrainEqualCounts(op->targets_.front(), op->value_);
        }
    }

    void VisitStmt_(const IfStmtPtr& op) override
    {
        // Yield belongs to this if; break/continue still target the enclosing loop.
        auto saved = std::move(yieldTargets_);
        yieldTargets_ = op->returnVars_;
        IRVisitor::VisitStmt_(op);
        yieldTargets_ = std::move(saved);
    }

    template <class Loop>
    void VisitLoop(const Loop& op, bool isFor)
    {
        // Nested loops replace all jump targets and restore the outer context on exit.
        auto savedYield = std::move(yieldTargets_);
        auto savedContinue = std::move(continueTargets_);
        auto savedBreak = std::move(breakTargets_);
        continueTargets_.clear();
        for (const auto& arg : op->iterArgs_) {
            continueTargets_.push_back(arg->iterVar_);
            SetConstrainEqualCounts(arg->iterVar_, arg->initValue_);
        }
        yieldTargets_ = continueTargets_;
        // While exit phis are independent of the header/back-edge phis.
        // Natural while conditions are lowered to explicit break guards by the parser.
        breakTargets_ = isFor ? continueTargets_ : op->returnVars_;
        if (isFor) {
            for (size_t i = 0; i < op->returnVars_.size(); ++i) {
                SetConstrainEqualCounts(op->returnVars_[i], continueTargets_[i]);
            }
        }
        IRVisitor::VisitStmt_(op);
        yieldTargets_ = std::move(savedYield);
        continueTargets_ = std::move(savedContinue);
        breakTargets_ = std::move(savedBreak);
    }

    void VisitStmt_(const ForStmtPtr& op) override { VisitLoop(op, true); }
    void VisitStmt_(const WhileStmtPtr& op) override { VisitLoop(op, false); }

    void VisitStmt_(const YieldStmtPtr& op) override
    {
        IRVisitor::VisitStmt_(op);
        MergeInputs(yieldTargets_, op->value_);
    }

    void VisitStmt_(const ContinueStmtPtr& op) override
    {
        IRVisitor::VisitStmt_(op);
        MergeInputs(continueTargets_, op->value_);
    }

    void VisitStmt_(const BreakStmtPtr& op) override
    {
        IRVisitor::VisitStmt_(op);
        MergeInputs(breakTargets_, op->value_);
    }

    DisjointSet countSets_{0};
    // Stored IDs identify slots; Find(node) returns the current count group.
    std::unordered_map<const Expr*, std::vector<MutexCountNodeId>> countNodesByExpr_;
    std::unordered_set<const Expr*> visitedExprs_;
    // Only jump targets are scoped. Count groups survive scope exit, connecting
    // nested results to outer merges through the actual SSA values they carry.
    std::vector<VarPtr> yieldTargets_;
    std::vector<VarPtr> continueTargets_;
    std::vector<VarPtr> breakTargets_;
};

using Mask = uint32_t;

struct TileMutexId {
    VarPtr placeholder;
    std::vector<int> manualIds;

    bool IsAuto() const { return placeholder != nullptr; }
};

struct TileGroupMembership {
    size_t groupId;
    size_t memberIndex;
};

struct TileRecord {
    ExprPtr tile;
    MemorySpace memorySpace;
    int64_t start;
    int64_t end;
    std::vector<int> manualIds;
    VarPtr placeholder;
    TileGroupMembership membership;
    Span span;

    bool IsAuto() const { return placeholder != nullptr; }
};

struct OverlapComponent {
    std::vector<size_t> tiles;
    MemorySpace memorySpace;
    int64_t end;
    bool containsAuto = false;
};

struct ContainmentNode {
    ContainmentNode(MemorySpace memory, int64_t rangeStart, int64_t rangeEnd, std::vector<size_t> logicalMembers,
                    size_t componentIndex, bool isActive)
        : memorySpace(memory),
          start(rangeStart),
          end(rangeEnd),
          members(std::move(logicalMembers)),
          component(componentIndex),
          active(isActive)
    {}

    MemorySpace memorySpace;
    int64_t start;
    int64_t end;
    std::vector<size_t> members;
    size_t component = 0;
    bool active = false;
    std::optional<Mask> manualMask;
    std::optional<size_t> parent;
    std::vector<size_t> children;
    int depth = 0;
    Mask allowedIds = kAllMutexIds;
    size_t countEqSetId = 0;
    std::vector<size_t> softPeers;
};

struct ContainmentForest {
    std::vector<ContainmentNode> nodes;
    std::vector<size_t> tileToNode;
};

struct PropagationState {
    std::vector<Mask> requiredIds;
    std::vector<int> lower;
    std::vector<int> upper;
};

class PlannerDiagnostics {
public:
    [[noreturn]] void FailInvalidArgument(const std::string& message, const Span& span)
    {
        pro::SpanScope span_scope(span);
        PRO_IR_THROW(ValueError, ExternalError::INVALID_ARGUMENT) << message;
    }

    [[noreturn]] void FailInvalidOperation(const std::string& message, const Span& span)
    {
        pro::SpanScope span_scope(span);
        PRO_IR_THROW(ValueError, ExternalError::INVALID_OPERATION) << message;
    }
};

Mask IdsToMask(const std::vector<int>& ids)
{
    Mask mask = 0;
    for (int id : ids) {
        mask |= Mask{1} << static_cast<unsigned>(id);
    }
    return mask;
}

bool Contains(int64_t outerStart, int64_t outerEnd, int64_t innerStart, int64_t innerEnd)
{
    return outerStart <= innerStart && innerEnd <= outerEnd;
}

std::string TileLabel(const std::vector<TileRecord>& tiles, size_t tileIndex)
{
    const auto& tile = tiles[tileIndex];
    return MemorySpaceToString(tile.memorySpace) + " Tile at line " + std::to_string(tile.span.BeginLine());
}

// Collect logical Tiles and their manual/AUTO mutex bindings from IR declarations.
class MutexDeclarationCollector final : public IRVisitor {
public:
    void VisitProgram(const ProgramPtr& program) override
    {
        debugInfo_ = program->GetDebugInfo();
        IRVisitor::VisitProgram(program);
    }

    bool HasAuto() const { return hasAuto_; }
    const std::vector<TileRecord>& Tiles() const { return tiles_; }

private:
    using IRVisitor::VisitExpr_;
    using IRVisitor::VisitStmt_;

    ExprPtr ResolveValue(ExprPtr value) const
    {
        while (auto var = As<Var>(value)) {
            auto it = definitions_.find(var.get());
            if (it == definitions_.end()) {
                break;
            }
            value = it->second;
        }
        return value;
    }

    void VisitStmt_(const AssignStmtPtr& stmt) override
    {
        if (stmt->targets_.size() == 1) {
            definitions_[stmt->targets_.front().get()] = stmt->value_;
        }
        IRVisitor::VisitStmt_(stmt);
    }

    void VisitExpr_(const MakeTuplePtr& tuple) override
    {
        IRVisitor::VisitExpr_(tuple);

        auto type = As<TupleType>(tuple->GetType());
        const auto* info = debugInfo_ && type ? debugInfo_->GetTupleTypeInfo(type.get()) : nullptr;
        if (info == nullptr || info->kind != TupleTypeKind::NAMED_TUPLE || !info->name.has_value() ||
            *info->name != "tile_group") {
            return;
        }

        auto field = std::find(info->fields.begin(), info->fields.end(), "tiles");
        size_t index = static_cast<size_t>(field - info->fields.begin());
        auto tiles = As<MakeTuple>(ResolveValue(tuple->elements_[index]));
        std::vector<ExprPtr> rows;
        rows.reserve(tiles->elements_.size());
        for (const auto& tile : tiles->elements_) {
            auto call = As<Call>(ResolveValue(tile));
            // Reinterpreted groups reuse the original declarations' mutex bindings.
            if (call == nullptr || call->name_ != "block.make_mutex_tile") {
                return;
            }
            auto ids = As<MakeTuple>(call->args_[2]);
            if (ids->elements_.empty()) {
                return;
            }
            rows.push_back(ids);
        }
        auto mutexIds = std::make_shared<const MakeTuple>(std::move(rows), tuple->span_);
        CollectGroup(tiles->elements_, mutexIds);
    }

    void CollectGroup(const std::vector<ExprPtr>& tiles, const MakeTuplePtr& mutexIds)
    {
        PRO_PASS_INTERNAL_CHECK_WITH_SPAN(npu::tile_fwk::InternalError::PASS_INNER_ERROR,
                                          mutexIds->elements_.size() == tiles.size(), mutexIds->span_)
            << "mutex ID count must match TileGroup depth";

        size_t groupId = nextGroupId_++;
        for (size_t index = 0; index < tiles.size(); ++index) {
            auto tileMutexId = ParseTileMutexId(mutexIds->elements_[index]);
            CollectTile(tiles[index], tileMutexId, TileGroupMembership{groupId, index});
        }
    }

    TileMutexId ParseTileMutexId(const ExprPtr& value)
    {
        auto mutexIds = As<MakeTuple>(value);
        PRO_PASS_INTERNAL_CHECK_WITH_SPAN(npu::tile_fwk::InternalError::PASS_INNER_ERROR,
                                          mutexIds != nullptr && !mutexIds->elements_.empty(), value->span_)
            << "each Tile must provide a non-empty mutex-ID tuple";

        if (mutexIds->elements_.size() == 1) {
            auto placeholder = As<Var>(mutexIds->elements_.front());
            if (placeholder != nullptr) {
                return TileMutexId{placeholder, {}};
            }
        }

        std::vector<int> manualIds;
        manualIds.reserve(mutexIds->elements_.size());
        for (const auto& mutexIdExpr : mutexIds->elements_) {
            auto mutexId = As<ConstInt>(mutexIdExpr);
            PRO_PASS_INTERNAL_CHECK_WITH_SPAN(npu::tile_fwk::InternalError::PASS_INNER_ERROR, mutexId != nullptr,
                                              mutexIdExpr->span_)
                << "manual mutex ID must be a ConstInt";
            ValidateMutexId(mutexId->value_, mutexId->span_);
            manualIds.push_back(static_cast<int>(mutexId->value_));
        }
        return TileMutexId{nullptr, std::move(manualIds)};
    }

    void CollectTile(const ExprPtr& tile, const TileMutexId& tileMutexId, TileGroupMembership membership)
    {
        auto tileType = As<TileType>(tile->GetType());
        PRO_PASS_INTERNAL_CHECK_WITH_SPAN(
            npu::tile_fwk::InternalError::PASS_INNER_ERROR,
            tileType != nullptr && tileType->memref_.has_value() && *tileType->memref_ != nullptr, tile->span_)
            << "AUTO mutex-ID planning requires every Tile to have a MemRef";
        auto memref = *tileType->memref_;
        auto address = As<ConstInt>(memref->addr_);
        PRO_PASS_INTERNAL_CHECK_WITH_SPAN(npu::tile_fwk::InternalError::PASS_INNER_ERROR, address != nullptr,
                                          tile->span_)
            << "AUTO mutex-ID planning requires a constant Tile address";
        PRO_PASS_INTERNAL_CHECK_WITH_SPAN(
            npu::tile_fwk::InternalError::PASS_INNER_ERROR,
            memref->size_ <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) &&
                address->value_ <= std::numeric_limits<int64_t>::max() - static_cast<int64_t>(memref->size_),
            tile->span_)
            << "Tile address range must fit int64";

        if (collectedTiles_.count(tile.get()) != 0) {
            return;
        }

        if (tileMutexId.IsAuto()) {
            hasAuto_ = true;
            PRO_PASS_INTERNAL_CHECK_WITH_SPAN(npu::tile_fwk::InternalError::PASS_INNER_ERROR,
                                              placeholders_.insert(tileMutexId.placeholder.get()).second,
                                              tileMutexId.placeholder->span_)
                << "AUTO mutex placeholder was reused by multiple Tiles";
        }

        tiles_.push_back(TileRecord{tile, memref->memorySpace_, address->value_,
                                    address->value_ + static_cast<int64_t>(memref->size_), tileMutexId.manualIds,
                                    tileMutexId.placeholder, std::move(membership), tile->span_});
        collectedTiles_.insert(tile.get());
    }

    static void ValidateMutexId(int64_t mutexId, const Span& span)
    {
        PRO_PASS_INTERNAL_CHECK_WITH_SPAN(npu::tile_fwk::InternalError::PASS_INNER_ERROR,
                                          mutexId >= 0 && mutexId < kMutexIdCount, span)
            << "mutex ID must be in [0, 31]";
    }

    std::vector<TileRecord> tiles_;
    std::unordered_set<const Expr*> collectedTiles_;
    std::unordered_set<const Var*> placeholders_;
    size_t nextGroupId_ = 0;
    bool hasAuto_ = false;
    const IRDebugInfo* debugInfo_ = nullptr;
    std::unordered_map<const Var*, ExprPtr> definitions_;
};

// Planning phase: normalize active address components and build containment forests.
class ContainmentForestBuilder {
public:
    ContainmentForestBuilder(const std::vector<TileRecord>& tiles, PlannerDiagnostics& diagnostics)
        : tiles_(tiles), diagnostics_(diagnostics)
    {}

    ContainmentForest Build()
    {
        BuildNodes();
        ValidateCrossingRanges();
        BuildForest();
        return ContainmentForest{std::move(nodes_), std::move(tileToNode_)};
    }

private:
    std::vector<OverlapComponent> BuildOverlapComponents() const
    {
        std::vector<size_t> ordered(tiles_.size());
        std::iota(ordered.begin(), ordered.end(), 0);
        std::sort(ordered.begin(), ordered.end(), [this](size_t left, size_t right) {
            const auto& lhs = tiles_[left];
            const auto& rhs = tiles_[right];
            const auto& lhsMembership = lhs.membership;
            const auto& rhsMembership = rhs.membership;
            return std::tie(lhs.memorySpace, lhs.start, lhs.end, lhsMembership.groupId, lhsMembership.memberIndex) <
                   std::tie(rhs.memorySpace, rhs.start, rhs.end, rhsMembership.groupId, rhsMembership.memberIndex);
        });

        std::vector<OverlapComponent> components;
        for (size_t tileIndex : ordered) {
            const auto& tile = tiles_[tileIndex];
            bool beginsNew = components.empty() || components.back().memorySpace != tile.memorySpace ||
                             tile.start >= components.back().end;
            if (beginsNew) {
                components.push_back(OverlapComponent{{}, tile.memorySpace, tile.end, false});
            }
            auto& component = components.back();
            component.tiles.push_back(tileIndex);
            component.end = std::max(component.end, tile.end);
            component.containsAuto |= tile.IsAuto();
        }
        return components;
    }

    void BuildNodes()
    {
        nodes_.clear();
        tileToNode_.assign(tiles_.size(), std::numeric_limits<size_t>::max());
        auto components = BuildOverlapComponents();

        for (size_t componentIndex = 0; componentIndex < components.size(); ++componentIndex) {
            const auto& component = components[componentIndex];
            if (!component.containsAuto) {
                for (size_t tileIndex : component.tiles) {
                    const auto& tile = tiles_[tileIndex];
                    size_t nodeIndex = nodes_.size();
                    ContainmentNode node(tile.memorySpace, tile.start, tile.end, {tileIndex}, componentIndex, false);
                    node.manualMask = IdsToMask(tile.manualIds);
                    nodes_.push_back(std::move(node));
                    tileToNode_[tileIndex] = nodeIndex;
                }
                continue;
            }

            using RangeKey = std::tuple<MemorySpace, int64_t, int64_t>;
            std::map<RangeKey, std::vector<size_t>> ranges;
            for (size_t tileIndex : component.tiles) {
                const auto& tile = tiles_[tileIndex];
                ranges[{tile.memorySpace, tile.start, tile.end}].push_back(tileIndex);
            }
            for (auto& [key, members] : ranges) {
                auto [memorySpace, start, end] = key;
                ContainmentNode node(memorySpace, start, end, std::move(members), componentIndex, true);
                std::optional<Mask> fixed;
                for (size_t tileIndex : node.members) {
                    const auto& tile = tiles_[tileIndex];
                    if (!tile.IsAuto()) {
                        Mask mask = IdsToMask(tile.manualIds);
                        if (fixed.has_value() && *fixed != mask) {
                            diagnostics_.FailInvalidOperation(
                                "Tiles with the same address range require identical manual mutex IDs when AUTO "
                                "allocation is enabled: " +
                                    TileLabel(tiles_, node.members.front()) + " uses " + FormatMask(*fixed) + "; " +
                                    TileLabel(tiles_, tileIndex) + " uses " + FormatMask(mask) +
                                    ". Assign identical mutex IDs to these Tiles or redesign the TileGroups",
                                tile.span);
                        }
                        fixed = mask;
                    }
                }
                node.manualMask = fixed;
                size_t nodeIndex = nodes_.size();
                nodes_.push_back(std::move(node));
                for (size_t tileIndex : nodes_.back().members) {
                    tileToNode_[tileIndex] = nodeIndex;
                }
            }
        }
    }

    void ValidateCrossingRanges()
    {
        for (size_t leftIndex = 0; leftIndex < nodes_.size(); ++leftIndex) {
            const auto& left = nodes_[leftIndex];
            if (!left.active) {
                continue;
            }
            for (size_t rightIndex = leftIndex + 1; rightIndex < nodes_.size(); ++rightIndex) {
                const auto& right = nodes_[rightIndex];
                if (!right.active || left.component != right.component) {
                    continue;
                }
                bool overlaps = left.start < right.end && right.start < left.end;
                bool nested = Contains(left.start, left.end, right.start, right.end) ||
                              Contains(right.start, right.end, left.start, left.end);
                if (overlaps && !nested) {
                    int64_t overlapStart = std::max(left.start, right.start);
                    int64_t overlapEnd = std::min(left.end, right.end);
                    size_t leftTile = left.members.front();
                    size_t rightTile = right.members.front();
                    diagnostics_.FailInvalidOperation(
                        "mutex_ids=\"auto\" does not support crossing Tile address ranges: " +
                            TileLabel(tiles_, leftTile) + " uses " + HexRange(left.start, left.end) + "; " +
                            TileLabel(tiles_, rightTile) + " uses " + HexRange(right.start, right.end) +
                            "; overlap is " + HexRange(overlapStart, overlapEnd) +
                            "; possible buffer address trampling",
                        tiles_[rightTile].span);
                }
            }
        }
    }

    void BuildForest()
    {
        for (size_t nodeIndex = 0; nodeIndex < nodes_.size(); ++nodeIndex) {
            auto& node = nodes_[nodeIndex];
            if (!node.active) {
                continue;
            }
            std::optional<size_t> bestParent;
            int64_t bestLength = std::numeric_limits<int64_t>::max();
            for (size_t candidateIndex = 0; candidateIndex < nodes_.size(); ++candidateIndex) {
                const auto& candidate = nodes_[candidateIndex];
                if (!candidate.active || candidate.component != node.component || candidateIndex == nodeIndex ||
                    !Contains(candidate.start, candidate.end, node.start, node.end)) {
                    continue;
                }
                int64_t length = candidate.end - candidate.start;
                if (length < bestLength) {
                    bestParent = candidateIndex;
                    bestLength = length;
                }
            }
            node.parent = bestParent;
        }
        for (size_t nodeIndex = 0; nodeIndex < nodes_.size(); ++nodeIndex) {
            if (nodes_[nodeIndex].parent.has_value()) {
                nodes_[*nodes_[nodeIndex].parent].children.push_back(nodeIndex);
            }
        }

        std::function<void(size_t, int)> setDepth = [this, &setDepth](size_t nodeIndex, int depth) {
            nodes_[nodeIndex].depth = depth;
            for (size_t child : nodes_[nodeIndex].children) {
                setDepth(child, depth + 1);
            }
        };
        for (size_t nodeIndex = 0; nodeIndex < nodes_.size(); ++nodeIndex) {
            if (nodes_[nodeIndex].active && !nodes_[nodeIndex].parent.has_value()) {
                setDepth(nodeIndex, 0);
            }
        }
        std::vector<size_t> activeNodes;
        for (size_t nodeIndex = 0; nodeIndex < nodes_.size(); ++nodeIndex) {
            if (nodes_[nodeIndex].active) {
                activeNodes.push_back(nodeIndex);
            }
        }
        std::sort(activeNodes.begin(), activeNodes.end(),
                  [this](size_t left, size_t right) { return nodes_[left].depth < nodes_[right].depth; });
        for (size_t nodeIndex : activeNodes) {
            auto& node = nodes_[nodeIndex];
            Mask inherited = node.parent.has_value() ? nodes_[*node.parent].allowedIds : kAllMutexIds;
            if (node.manualMask.has_value()) {
                if ((*node.manualMask & ~inherited) != 0) {
                    size_t childTile = node.members.front();
                    size_t parentTile = nodes_[*node.parent].members.front();
                    diagnostics_.FailInvalidArgument(
                        "Manual mutex IDs for " + TileLabel(tiles_, childTile) + " are incompatible with containing " +
                            TileLabel(tiles_, parentTile) +
                            ": the containing Tile must include every mutex ID used "
                            "by the contained Tile. Redesign the TileGroups, split the Tiles into separate "
                            "TileGroups, or assign compatible mutex IDs manually",
                        tiles_[childTile].span);
                }
                node.allowedIds = *node.manualMask;
            } else {
                node.allowedIds = inherited;
            }
        }
    }

    const std::vector<TileRecord>& tiles_;
    PlannerDiagnostics& diagnostics_;
    std::vector<ContainmentNode> nodes_;
    std::vector<size_t> tileToNode_;
};

// Solving phase: combine same-depth soft preferences, TileGroup count equalities, and hard set constraints.
class MutexConstraintSolver {
public:
    MutexConstraintSolver(const std::vector<TileRecord>& tiles, ContainmentForest& forest,
                          SsaMutexCountAnalysis& ssaCountAnalysis, PlannerDiagnostics& diagnostics)
        : tiles_(tiles),
          nodes_(forest.nodes),
          tileToNode_(forest.tileToNode),
          ssaCountAnalysis_(ssaCountAnalysis),
          diagnostics_(diagnostics)
    {}

    std::vector<Mask> Solve()
    {
        BuildTileGroupCountEqSets();
        BuildSsaCountEqSets();
        BuildSameTreeSameDepthAvoidance();
        FinalizeCountEqSets();

        return AllocateMutexIdSets();
    }

private:
    void BuildTileGroupCountEqSets()
    {
        countEqSets_ = std::make_unique<DisjointSet>(nodes_.size());
        std::unordered_map<size_t, size_t> firstNodeByGroup;
        for (size_t tileIndex = 0; tileIndex < tiles_.size(); ++tileIndex) {
            size_t nodeIndex = tileToNode_[tileIndex];
            auto [it, inserted] = firstNodeByGroup.emplace(tiles_[tileIndex].membership.groupId, nodeIndex);
            if (!inserted) {
                countEqSets_->Join(it->second, nodeIndex);
            }
        }
    }

    void BuildSsaCountEqSets()
    {
        std::unordered_map<MutexCountGroupId, size_t> firstNodeByGroup;
        for (size_t tileIndex = 0; tileIndex < tiles_.size(); ++tileIndex) {
            size_t nodeIndex = tileToNode_[tileIndex];
            auto [it, inserted] = firstNodeByGroup.emplace(
                ssaCountAnalysis_.GetTileCountGroupId(tiles_[tileIndex].tile), nodeIndex);
            if (!inserted) {
                countEqSets_->Join(it->second, nodeIndex);
            }
        }
    }

    void BuildSameTreeSameDepthAvoidance()
    {
        for (size_t rootIndex = 0; rootIndex < nodes_.size(); ++rootIndex) {
            const auto& root = nodes_[rootIndex];
            if (!root.active || root.parent.has_value()) {
                continue;
            }
            std::vector<size_t> layer{rootIndex};
            while (!layer.empty()) {
                std::vector<size_t> nextLayer;
                for (size_t left = 0; left < layer.size(); ++left) {
                    for (size_t right = left + 1; right < layer.size(); ++right) {
                        AddSoftEdge(layer[left], layer[right]);
                    }
                    const auto& children = nodes_[layer[left]].children;
                    nextLayer.insert(nextLayer.end(), children.begin(), children.end());
                }
                layer = std::move(nextLayer);
            }
        }
    }

    void AddSoftEdge(size_t left, size_t right)
    {
        if (left == right) {
            return;
        }
        nodes_[left].softPeers.push_back(right);
        nodes_[right].softPeers.push_back(left);
    }

    void FinalizeCountEqSets()
    {
        activeCountEqSets_.assign(nodes_.size(), false);
        for (size_t nodeIndex = 0; nodeIndex < nodes_.size(); ++nodeIndex) {
            nodes_[nodeIndex].countEqSetId = countEqSets_->Find(nodeIndex);
            if (nodes_[nodeIndex].active) {
                activeCountEqSets_[nodes_[nodeIndex].countEqSetId] = true;
            }
            auto& peers = nodes_[nodeIndex].softPeers;
            std::sort(peers.begin(), peers.end());
            peers.erase(std::unique(peers.begin(), peers.end()), peers.end());
        }
    }

    std::vector<Mask> AllocateMutexIdSets()
    {
        std::vector<Mask> assignedIds(nodes_.size(), 0);
        for (size_t nodeIndex = 0; nodeIndex < nodes_.size(); ++nodeIndex) {
            if (nodes_[nodeIndex].manualMask.has_value()) {
                assignedIds[nodeIndex] = *nodes_[nodeIndex].manualMask;
            }
        }
        PropagationState initialPropagation;
        std::optional<size_t> conflictingCountEqSetId;
        if (Propagate(assignedIds, initialPropagation, &conflictingCountEqSetId)) {
            backtrackCount_ = 0;
            backtrackLimit_ = CountActiveNodes() * kBacktracksPerActiveNode;
            if (Search(assignedIds, initialPropagation)) {
                return assignedIds;
            }
        } else if (conflictingCountEqSetId.has_value() && CountEqSetSize(*conflictingCountEqSetId) > 1) {
            FailTileGroupCountConflict(*conflictingCountEqSetId, initialPropagation);
        }

        size_t tileIndex = FirstAutoTile();
        diagnostics_.FailInvalidArgument(
            "Unable to allocate AUTO mutex IDs for " + TileLabel(tiles_, tileIndex) +
                " under the current Tile containment and TileGroup constraints. Redesign the TileGroups, split the "
                "Tiles into separate TileGroups, or assign mutex IDs manually",
            tiles_[tileIndex].span);
    }

    bool Propagate(const std::vector<Mask>& assignedIds, PropagationState& state,
                   std::optional<size_t>* conflictingCountEqSetId = nullptr) const
    {
        if (!PropagateRequiredIds(assignedIds, state.requiredIds)) {
            return false;
        }
        InitCountEqSetBounds(assignedIds, state);
        return PropagateContainmentCountBounds(state, conflictingCountEqSetId);
    }

    bool PropagateRequiredIds(const std::vector<Mask>& assignedIds, std::vector<Mask>& requiredIds) const
    {
        requiredIds.assign(nodes_.size(), 0);
        std::vector<size_t> bottomUp;
        for (size_t nodeIndex = 0; nodeIndex < nodes_.size(); ++nodeIndex) {
            if (nodes_[nodeIndex].active) {
                requiredIds[nodeIndex] = assignedIds[nodeIndex];
                bottomUp.push_back(nodeIndex);
            }
        }
        std::sort(bottomUp.begin(), bottomUp.end(),
                  [this](size_t left, size_t right) { return nodes_[left].depth > nodes_[right].depth; });
        for (size_t nodeIndex : bottomUp) {
            const auto& node = nodes_[nodeIndex];
            if ((requiredIds[nodeIndex] & ~node.allowedIds) != 0 ||
                (assignedIds[nodeIndex] != 0 && (requiredIds[nodeIndex] & ~assignedIds[nodeIndex]) != 0)) {
                return false;
            }
            if (node.parent.has_value()) {
                requiredIds[*node.parent] |= requiredIds[nodeIndex];
            }
        }
        return true;
    }

    void InitCountEqSetBounds(const std::vector<Mask>& assignedIds, PropagationState& state) const
    {
        state.lower.assign(nodes_.size(), 1);
        state.upper.assign(nodes_.size(), kMutexIdCount);
        for (size_t nodeIndex = 0; nodeIndex < nodes_.size(); ++nodeIndex) {
            const auto& node = nodes_[nodeIndex];
            size_t countEqSetId = node.countEqSetId;
            if (!activeCountEqSets_[countEqSetId]) {
                continue;
            }
            if (node.active) {
                state.lower[countEqSetId] = std::max(state.lower[countEqSetId],
                                                     MaskToIdCount(state.requiredIds[nodeIndex]));
                state.upper[countEqSetId] = std::min(state.upper[countEqSetId], MaskToIdCount(node.allowedIds));
            }
            if (assignedIds[nodeIndex] != 0) {
                int exact = MaskToIdCount(assignedIds[nodeIndex]);
                state.lower[countEqSetId] = std::max(state.lower[countEqSetId], exact);
                state.upper[countEqSetId] = std::min(state.upper[countEqSetId], exact);
            }
        }
    }

    bool PropagateContainmentCountBounds(PropagationState& state, std::optional<size_t>* conflictingCountEqSetId) const
    {
        bool changed = true;
        while (changed) {
            changed = false;
            for (size_t nodeIndex = 0; nodeIndex < nodes_.size(); ++nodeIndex) {
                const auto& node = nodes_[nodeIndex];
                if (!node.active || !node.parent.has_value()) {
                    continue;
                }
                size_t childCountEqSetId = node.countEqSetId;
                size_t parentCountEqSetId = nodes_[*node.parent].countEqSetId;
                int parentLower = std::max(state.lower[parentCountEqSetId], state.lower[childCountEqSetId]);
                int childUpper = std::min(state.upper[childCountEqSetId], state.upper[parentCountEqSetId]);
                changed |= parentLower != state.lower[parentCountEqSetId] ||
                           childUpper != state.upper[childCountEqSetId];
                state.lower[parentCountEqSetId] = parentLower;
                state.upper[childCountEqSetId] = childUpper;
            }
            for (size_t countEqSetId = 0; countEqSetId < nodes_.size(); ++countEqSetId) {
                if (activeCountEqSets_[countEqSetId] && state.lower[countEqSetId] > state.upper[countEqSetId]) {
                    if (conflictingCountEqSetId != nullptr) {
                        *conflictingCountEqSetId = countEqSetId;
                    }
                    return false;
                }
            }
        }
        return true;
    }

    bool Search(std::vector<Mask>& assignedIds, const PropagationState& propagation)
    {
        auto selectedNode = SelectNextNode(assignedIds, propagation);
        if (!selectedNode.has_value()) {
            return true;
        }
        auto freeIds = RankFreeIds(*selectedNode, assignedIds, propagation);
        return TryNodeAssignments(*selectedNode, freeIds, assignedIds, propagation);
    }

    std::optional<size_t> SelectNextNode(const std::vector<Mask>& assignedIds,
                                         const PropagationState& propagation) const
    {
        std::optional<size_t> selectedNode;
        std::tuple<int, int, int, int, int, int, MemorySpace, int64_t, int64_t> selectedPriority;
        for (size_t nodeIndex = 0; nodeIndex < nodes_.size(); ++nodeIndex) {
            const auto& node = nodes_[nodeIndex];
            if (!node.active || assignedIds[nodeIndex] != 0) {
                continue;
            }
            size_t countEqSetId = node.countEqSetId;
            int leafRank = node.children.empty() ? 0 : 1;
            int freeCount = MaskToIdCount(node.allowedIds & ~propagation.requiredIds[nodeIndex]);
            // Finish deeper nodes before ancestors fix sets that could force sibling ID reuse.
            auto priority = std::make_tuple(IsStandalone(nodeIndex) ? 1 : 0, leafRank, -node.depth,
                                            propagation.upper[countEqSetId] - propagation.lower[countEqSetId],
                                            freeCount, -static_cast<int>(node.softPeers.size()), node.memorySpace,
                                            node.start, node.end);
            if (!selectedNode.has_value() || priority < selectedPriority) {
                selectedNode = nodeIndex;
                selectedPriority = priority;
            }
        }
        return selectedNode;
    }

    std::vector<int> RankFreeIds(size_t nodeIndex, const std::vector<Mask>& assignedIds,
                                 const PropagationState& propagation) const
    {
        Mask freeMask = nodes_[nodeIndex].allowedIds & ~propagation.requiredIds[nodeIndex];
        auto parent = nodes_[nodeIndex].parent;
        while (parent.has_value()) {
            const auto& ancestor = nodes_[*parent];
            Mask requiredIds = propagation.requiredIds[*parent];
            // A saturated ancestor cannot admit IDs outside its required set.
            // A count-equal peer can saturate it even before the ancestor is assigned.
            if (MaskToIdCount(requiredIds) == propagation.upper[ancestor.countEqSetId]) {
                freeMask &= requiredIds;
            }
            parent = ancestor.parent;
        }
        Mask avoid = AvoidMask(nodeIndex, assignedIds, propagation.requiredIds);
        auto usage = IdUsage(assignedIds);
        auto freeIds = MaskToIds(freeMask);
        std::sort(freeIds.begin(), freeIds.end(), [&avoid, &usage](int left, int right) {
            auto score = [&avoid, &usage](int mutexId) {
                return std::make_tuple((avoid & (Mask{1} << static_cast<unsigned>(mutexId))) != 0, usage[mutexId],
                                       mutexId);
            };
            return score(left) < score(right);
        });
        return freeIds;
    }

    bool TryNodeAssignments(size_t nodeIndex, const std::vector<int>& freeIds, std::vector<Mask>& assignedIds,
                            const PropagationState& propagation)
    {
        size_t countEqSetId = nodes_[nodeIndex].countEqSetId;
        Mask base = propagation.requiredIds[nodeIndex];
        // Try smaller counts first within the propagated bounds.
        int baseCount = MaskToIdCount(base);
        for (int count = propagation.lower[countEqSetId]; count <= propagation.upper[countEqSetId]; ++count) {
            if (TryIdCombinations(nodeIndex, freeIds, 0, count - baseCount, base, assignedIds)) {
                return true;
            }
        }
        return false;
    }

    bool TryIdCombinations(size_t nodeIndex, const std::vector<int>& freeIds, size_t startIndex, int remainingCount,
                           Mask candidateMask, std::vector<Mask>& assignedIds)
    {
        if (remainingCount == 0) {
            return TryAssignment(nodeIndex, candidateMask, assignedIds);
        }
        if (freeIds.size() - startIndex < static_cast<size_t>(remainingCount)) {
            return false;
        }
        for (size_t index = startIndex; index < freeIds.size(); ++index) {
            Mask bit = Mask{1} << static_cast<unsigned>(freeIds[index]);
            if (TryIdCombinations(nodeIndex, freeIds, index + 1, remainingCount - 1, candidateMask | bit,
                                  assignedIds)) {
                return true;
            }
        }
        return false;
    }

    bool TryAssignment(size_t nodeIndex, Mask mask, std::vector<Mask>& assignedIds)
    {
        assignedIds[nodeIndex] = mask;

        PropagationState nextPropagation;
        if (Propagate(assignedIds, nextPropagation) && Search(assignedIds, nextPropagation)) {
            return true;
        }

        assignedIds[nodeIndex] = 0;
        RecordBacktrack();
        return false;
    }

    bool IsStandalone(size_t nodeIndex) const
    {
        const auto& node = nodes_[nodeIndex];
        return !node.parent.has_value() && node.children.empty() && node.members.size() == 1;
    }

    Mask AvoidMask(size_t nodeIndex, const std::vector<Mask>& assignedIds, const std::vector<Mask>& requiredIds) const
    {
        Mask avoid = OwnAvoidMask(nodeIndex, assignedIds, requiredIds);
        auto parent = nodes_[nodeIndex].parent;
        while (parent.has_value()) {
            avoid |= OwnAvoidMask(*parent, assignedIds, requiredIds);
            parent = nodes_[*parent].parent;
        }
        return avoid;
    }

    Mask OwnAvoidMask(size_t nodeIndex, const std::vector<Mask>& assignedIds,
                      const std::vector<Mask>& requiredIds) const
    {
        Mask avoid = 0;
        // Soft edges only connect active nodes.
        for (size_t peer : nodes_[nodeIndex].softPeers) {
            if (assignedIds[peer] != 0) {
                avoid |= assignedIds[peer];
            } else {
                avoid |= requiredIds[peer];
            }
        }
        return avoid;
    }

    std::array<int, kMutexIdCount> IdUsage(const std::vector<Mask>& assignedIds) const
    {
        std::array<int, kMutexIdCount> usage{};
        for (size_t nodeIndex = 0; nodeIndex < nodes_.size(); ++nodeIndex) {
            Mask mask = assignedIds[nodeIndex];
            if (mask == 0) {
                continue;
            }
            int weight = static_cast<int>(nodes_[nodeIndex].members.size());
            for (int mutexId : MaskToIds(mask)) {
                usage[mutexId] += weight;
            }
        }
        return usage;
    }

    size_t CountActiveNodes() const
    {
        return static_cast<size_t>(
            std::count_if(nodes_.begin(), nodes_.end(), [](const auto& node) { return node.active; }));
    }

    void RecordBacktrack()
    {
        if (++backtrackCount_ <= backtrackLimit_) {
            return;
        }
        size_t tileIndex = FirstAutoTile();
        diagnostics_.FailInvalidOperation(
            "Unable to allocate AUTO mutex IDs for " + TileLabel(tiles_, tileIndex) +
                " under the current Tile containment and TileGroup constraints. Redesign the TileGroups, split the "
                "Tiles into separate TileGroups, or assign mutex IDs manually",
            tiles_[tileIndex].span);
    }

    size_t CountEqSetSize(size_t countEqSetId) const
    {
        return static_cast<size_t>(std::count_if(nodes_.begin(), nodes_.end(), [countEqSetId](const auto& node) {
            return node.countEqSetId == countEqSetId;
        }));
    }

    [[noreturn]] void FailTileGroupCountConflict(size_t countEqSetId, const PropagationState& state)
    {
        std::ostringstream message;
        message << "Unable to assign the same mutex ID count to all Tiles in a TileGroup or SSA merge: ";
        bool first = true;
        size_t diagnosticTile = FirstAutoTile();
        for (size_t nodeIndex = 0; nodeIndex < nodes_.size(); ++nodeIndex) {
            const auto& node = nodes_[nodeIndex];
            if (node.countEqSetId != countEqSetId) {
                continue;
            }
            size_t tileIndex = node.members.front();
            if (first) {
                diagnosticTile = tileIndex;
            } else {
                message << "; ";
            }
            first = false;
            int lower = std::max(1, MaskToIdCount(state.requiredIds[nodeIndex]));
            int upper = MaskToIdCount(node.allowedIds);
            if (node.manualMask.has_value()) {
                lower = MaskToIdCount(*node.manualMask);
                upper = lower;
            }
            message << TileLabel(tiles_, tileIndex) << " requires [" << lower << ", " << upper << "] IDs";
        }
        message << ". The TileGroup or SSA merge requires at least " << state.lower[countEqSetId]
                << " IDs but allows at most " << state.upper[countEqSetId]
                << ". Split the Tiles into separate TileGroups or assign compatible mutex IDs manually";
        diagnostics_.FailInvalidArgument(message.str(), tiles_[diagnosticTile].span);
    }

    size_t FirstAutoTile() const
    {
        for (size_t tileIndex = 0; tileIndex < tiles_.size(); ++tileIndex) {
            if (tiles_[tileIndex].IsAuto()) {
                return tileIndex;
            }
        }
        return 0;
    }

    const std::vector<TileRecord>& tiles_;
    std::vector<ContainmentNode>& nodes_;
    const std::vector<size_t>& tileToNode_;
    SsaMutexCountAnalysis& ssaCountAnalysis_;
    PlannerDiagnostics& diagnostics_;
    std::unique_ptr<DisjointSet> countEqSets_;
    std::vector<bool> activeCountEqSets_;
    size_t backtrackCount_ = 0;
    size_t backtrackLimit_ = 0;
};

ProgramPtr AssignMutexIds(const ProgramPtr& program)
{
    MutexDeclarationCollector collector;
    collector.VisitProgram(program);
    if (!collector.HasAuto()) {
        return program;
    }

    const auto& tiles = collector.Tiles();
    PlannerDiagnostics diagnostics;
    ContainmentForestBuilder forestBuilder(tiles, diagnostics);
    auto forest = forestBuilder.Build();

    SsaMutexCountAnalysis ssaCountAnalysis(program);
    MutexConstraintSolver solver(tiles, forest, ssaCountAnalysis, diagnostics);
    auto assignedIds = solver.Solve();

    std::unordered_map<MutexCountGroupId, uint32_t> countBySsaGroup;
    std::vector<std::pair<VarPtr, std::vector<int>>> replacements;
    replacements.reserve(tiles.size());
    for (size_t tileIndex = 0; tileIndex < tiles.size(); ++tileIndex) {
        const auto& tile = tiles[tileIndex];
        size_t nodeIndex = forest.tileToNode[tileIndex];
        PRO_PASS_INTERNAL_CHECK_WITH_SPAN(npu::tile_fwk::InternalError::PASS_INNER_ERROR,
                                          nodeIndex < assignedIds.size(), tile.span)
            << "mutex Tile has no containment node";
        countBySsaGroup[ssaCountAnalysis.GetTileCountGroupId(tile.tile)] = static_cast<uint32_t>(
            MaskToIdCount(assignedIds[nodeIndex]));
        if (!tile.IsAuto()) {
            continue;
        }
        replacements.emplace_back(tile.placeholder, MaskToIds(assignedIds[nodeIndex]));
    }
    auto ssaMutexCounts = ssaCountAnalysis.BuildSsaMutexCounts(countBySsaGroup);
    PlaceholderRewriter rewriter(replacements, ssaMutexCounts, program->debugInfo_);
    return rewriter.VisitProgram(program);
}
} // namespace

Pass pass::AllocateMutexIds() { return CreateProgramPass(AssignMutexIds, "AllocateMutexIds"); }

} // namespace ir
} // namespace pypto
