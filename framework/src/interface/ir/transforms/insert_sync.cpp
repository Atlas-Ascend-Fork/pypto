/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include <algorithm>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "backend/backend_cce.h"
#include "ir/kind_traits.h"
#include "ir/op_attr_types.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr_ops.h"
#include "ir/transforms/base/mutator.h"
#include "ir/transforms/base/visitor.h"
#include "ir/transforms/passes.h"
#include "ir/transforms/structural_comparison.h"
#include "pypto_pro/error.h"

namespace pypto::ir {
namespace {
using npu::tile_fwk::ExternalError;

MemorySpace Memory(const ExprPtr& value)
{
    auto type = As<TileType>(value->GetType());
    return type && type->memref_ ? (*type->memref_)->memorySpace_ : MemorySpace::DDR;
}

bool DescriptorOnly(const std::string& name)
{
    return name == "block.make_mutex_tile" || name == "block.reinterpret" || name == "block.subview" ||
           name == "block.set_validshape" || name == "block.tile_valid_shape" || name == "block.tile_mutex_id";
}

std::optional<PipeType> Pipe(const CallPtr& call)
{
    const auto& name = call->name_;
    if ((name.rfind("block.", 0) != 0 && name != "simt.launch") || DescriptorOnly(name))
        return std::nullopt;
    auto info = backend::BackendCCE::Instance().GetOpInfo(name);
    if (info && info->pipe)
        return info->pipe;
    if ((name == "block.move" || name == "block.insert") && call->args_.size() >= 2) {
        auto dst = Memory(call->args_[0]);
        auto src = Memory(call->args_[1]);
        if (src == MemorySpace::Acc && (dst == MemorySpace::Vec || dst == MemorySpace::Mat))
            return PipeType::FIX;
        if (src == MemorySpace::Mat) {
            if (dst == MemorySpace::Left || dst == MemorySpace::Right || dst == MemorySpace::Bias ||
                dst == MemorySpace::ScaleLeft || dst == MemorySpace::ScaleRight)
                return PipeType::MTE1;
            return dst == MemorySpace::Vec ? PipeType::V : PipeType::FIX;
        }
        if (src == MemorySpace::Vec && dst == MemorySpace::Mat)
            return PipeType::MTE3;
        return PipeType::V;
    }
    if ((name == "block.store" || name == "block.store_tile") && call->args_.size() > 1)
        return Memory(call->args_[1]) == MemorySpace::Acc ? PipeType::FIX : PipeType::MTE3;
    if ((name == "block.expands" || name == "block.full") && !call->args_.empty())
        return Memory(call->args_[0]) == MemorySpace::Mat ? PipeType::MTE2 : PipeType::V;
    return std::nullopt;
}

bool SkipAcc(const CallPtr& call)
{
    static const std::set<std::string> phase_ops{"block.matmul",     "block.matmul_bias",   "block.matmul_acc",
                                                 "block.matmul_mx",  "block.matmul_mx_acc", "block.store",
                                                 "block.store_tile", "block.move",          "block.insert"};
    if (!phase_ops.count(call->name_))
        return false;
    int phase = call->GetKwarg<int>("phase", 0);
    return phase == static_cast<int>(AccPhase::Partial) || phase == static_cast<int>(AccPhase::Final);
}

int MutexCount(const TypePtr& type)
{
    auto tile = As<MutexTileType>(type);
    return tile ? static_cast<int>(tile->mutexCount_) : 0;
}

// Tuple slots are flattened in type order. No separate tuple layout or
// make_tile provenance needs to survive candidate propagation.
size_t MutexSlots(const TypePtr& type)
{
    if (auto tuple = As<TupleType>(type)) {
        size_t count = 0;
        for (const auto& element : tuple->types_)
            count += MutexSlots(element);
        return count;
    }
    return MutexCount(type);
}

// Propagate candidates directly over SSA definitions and control-flow merges.
// Revisit the function until aliases and loop-carried values stop changing.
class CandidateAnalysis : public IRVisitor {
public:
    using IRVisitor::VisitExpr_;
    using IRVisitor::VisitStmt_;
    using SlotCandidates = std::vector<std::set<int64_t>>;

    explicit CandidateAnalysis(const FunctionPtr& function)
    {
        for (const auto& parameter : function->params_) {
            auto& slots = candidates_[parameter];
            slots.resize(MutexSlots(parameter->GetType()));
            for (auto& slot : slots)
                for (int64_t id = 0; id < 32; ++id)
                    slot.insert(id);
        }
        do {
            changed_ = false;
            VisitStmt(function->body_);
        } while (changed_);
    }

    const SlotCandidates& GetCandidates(const ExprPtr& value) const { return candidates_.at(value); }

private:
    std::unordered_map<ExprPtr, SlotCandidates> candidates_;
    std::vector<VarPtr> yield_targets_;
    std::vector<VarPtr> continue_targets_;
    std::vector<VarPtr> break_targets_;
    bool changed_ = false;

    void VisitExpr(const ExprPtr& op) override
    {
        candidates_.try_emplace(op, MutexSlots(op->GetType()));
        IRVisitor::VisitExpr(op);
    }

    void SetCandidates(const ExprPtr& value, SlotCandidates candidates)
    {
        auto& current = candidates_.at(value);
        if (current != candidates) {
            current = std::move(candidates);
            changed_ = true;
        }
    }

    SlotCandidates TupleCandidates(const std::vector<ExprPtr>& elements) const
    {
        SlotCandidates result;
        for (const auto& element : elements) {
            const auto& slots = GetCandidates(element);
            result.insert(result.end(), slots.begin(), slots.end());
        }
        return result;
    }

    void VisitExpr_(const MakeTuplePtr& op) override
    {
        IRVisitor::VisitExpr_(op);
        SetCandidates(op, TupleCandidates(op->elements_));
    }

    void VisitExpr_(const GetItemExprPtr& op) override
    {
        IRVisitor::VisitExpr_(op);
        auto tuple_type = As<TupleType>(op->value_->GetType());
        if (!tuple_type)
            return;
        SlotCandidates result(MutexSlots(op->GetType()));
        const auto& base = GetCandidates(op->value_);
        auto index = As<ConstInt>(op->slice_);
        size_t offset = 0;
        for (size_t i = 0; i < tuple_type->types_.size(); ++i) {
            auto size = MutexSlots(tuple_type->types_[i]);
            if (!index || index->value_ == static_cast<int64_t>(i))
                MergeCandidates(result, SlotCandidates(base.begin() + offset, base.begin() + offset + size), op->span_);
            offset += size;
        }
        SetCandidates(op, std::move(result));
    }

    void VisitExpr_(const CallPtr& op) override
    {
        IRVisitor::VisitExpr_(op);
        if (op->name_ == "block.make_mutex_tile") {
            SlotCandidates result(MutexSlots(op->GetType()));
            const auto ids = As<MakeTuple>(op->args_[2]);
            for (size_t i = 0; i < ids->elements_.size(); ++i) {
                auto constant = As<ConstInt>(ids->elements_[i]);
                PRO_IR_IRCHECK(ExternalError::INVALID_ARGUMENT, constant) << "InsertSync requires allocated mutex IDs";
                result[i].insert(constant->value_);
            }
            SetCandidates(op, std::move(result));
        } else if (op->name_ == "struct.create") {
            SetCandidates(op, TupleCandidates(op->args_));
        } else if (op->name_ == "block.reinterpret" || op->name_ == "block.subview") {
            // Derived descriptors retain the source tile's mutex binding.
            SetCandidates(op, GetCandidates(op->args_[0]));
        }
    }

    static bool MergeCandidates(SlotCandidates& into, const SlotCandidates& from, const Span& span)
    {
        // Empty sources contribute nothing; empty targets have no slots to track.
        if (into.empty() || from.empty())
            return false;
        pro::SpanScope span_scope(span);
        PRO_IR_IRCHECK(ExternalError::INVALID_TYPE, into.size() == from.size())
            << "Inconsistent mutex ID counts in control flow or dynamic tuple selection";
        bool changed = false;
        for (size_t i = 0; i < into.size(); ++i) {
            auto size = into[i].size();
            into[i].insert(from[i].begin(), from[i].end());
            changed |= into[i].size() != size;
        }
        return changed;
    }

    void Merge(const VarPtr& target, const ExprPtr& value)
    {
        auto it = candidates_.try_emplace(target, MutexSlots(target->GetType())).first;
        changed_ |= MergeCandidates(it->second, GetCandidates(value), value->span_);
    }

    void Merge(const std::vector<VarPtr>& targets, const std::vector<ExprPtr>& values)
    {
        for (size_t i = 0; i < targets.size(); ++i)
            Merge(targets[i], values[i]);
    }

    void VisitStmt_(const AssignStmtPtr& op) override
    {
        IRVisitor::VisitStmt_(op);
        SetCandidates(op->var_, GetCandidates(op->value_));
    }

    void VisitStmt_(const IfStmtPtr& op) override
    {
        auto saved = std::move(yield_targets_);
        yield_targets_ = op->returnVars_;
        IRVisitor::VisitStmt_(op);
        yield_targets_ = std::move(saved);
    }

    template <class Loop>
    void LoopBody(const Loop& op, bool is_for)
    {
        auto old_continue = std::move(continue_targets_);
        auto old_break = std::move(break_targets_);
        continue_targets_.clear();
        for (const auto& arg : op->iterArgs_)
            continue_targets_.push_back(arg->iterVar_);
        // Match the parser: for merges every jump into each carried variable;
        // while keeps break results separate from initial/continue values.
        break_targets_ = is_for ? continue_targets_ : op->returnVars_;
        IRVisitor::VisitStmt_(op);
        for (const auto& arg : op->iterArgs_)
            Merge(arg->iterVar_, arg->initValue_);
        if (is_for)
            for (size_t i = 0; i < op->returnVars_.size(); ++i)
                Merge(op->returnVars_[i], continue_targets_[i]);
        continue_targets_ = std::move(old_continue);
        break_targets_ = std::move(old_break);
    }

    void VisitStmt_(const ForStmtPtr& op) override { LoopBody(op, true); }
    void VisitStmt_(const WhileStmtPtr& op) override { LoopBody(op, false); }
    void VisitStmt_(const YieldStmtPtr& op) override
    {
        IRVisitor::VisitStmt_(op);
        Merge(yield_targets_, op->value_);
    }
    void VisitStmt_(const ContinueStmtPtr& op) override
    {
        IRVisitor::VisitStmt_(op);
        Merge(continue_targets_, op->value_);
    }
    void VisitStmt_(const BreakStmtPtr& op) override
    {
        IRVisitor::VisitStmt_(op);
        Merge(break_targets_, op->value_);
    }
};

class SyncInserter : public IRMutator {
public:
    using IRMutator::VisitExpr_;
    using IRMutator::VisitStmt_;

    explicit SyncInserter(const FunctionPtr& function) : analysis_(function)
    {
        class Names : public IRVisitor {
        public:
            using IRVisitor::VisitExpr_;
            std::unordered_set<std::string> values;
            void VisitExpr_(const VarPtr& op) override { values.insert(op->name_); }
        } names;
        names.VisitStmt(function->body_);
        names_ = std::move(names.values);
    }

private:
    CandidateAnalysis analysis_;
    std::unordered_set<std::string> names_;
    size_t counter_ = 0;
    struct Mutex {
        ExprPtr tile;
        ExprPtr id;
        std::set<int64_t> candidates;
    };

    VarPtr Temp(const std::string& prefix, const TypePtr& type, const Span& span)
    {
        std::string name;
        do {
            name = "__sync_" + prefix + "_" + std::to_string(counter_++);
        } while (!names_.insert(name).second);
        return std::make_shared<Var>(name, type, span);
    }

    void AddTiles(const ExprPtr& value, std::vector<ExprPtr>& tiles)
    {
        if (As<TileType>(value->GetType())) {
            if (MutexCount(value->GetType()) > 0 && std::none_of(tiles.begin(), tiles.end(), [&](const ExprPtr& old) {
                    return structural_equal(old, value);
                }))
                tiles.push_back(value);
        } else if (auto tuple = As<MakeTuple>(value)) {
            for (const auto& element : tuple->elements_)
                AddTiles(element, tiles);
        }
    }

    StmtPtr Wrap(const StmtPtr& stmt, const std::vector<ExprPtr>& tiles, PipeType pipe)
    {
        const auto& span = stmt->span_;
        std::vector<StmtPtr> setup, locks, unlocks;
        std::vector<Mutex> prior;
        for (const auto& tile : tiles) {
            const auto& candidates = analysis_.GetCandidates(tile);
            for (int slot = 0; slot < MutexCount(tile->GetType()); ++slot) {
                const auto& effective = candidates[slot];
                ExprPtr id;
                if (effective.size() == 1)
                    id = std::make_shared<ConstInt>(*effective.begin(), DataType::INT64, span);
                ExprPtr guard;
                bool duplicate = false;
                for (const auto& old : prior) {
                    if (id && As<ConstInt>(old.id) && structural_equal(id, old.id)) {
                        duplicate = true;
                        break;
                    }
                }
                if (duplicate)
                    continue;
                if (!id) {
                    auto index = std::make_shared<ConstInt>(slot, DataType::INT64, span);
                    auto getter = OpRegistry::GetInstance().Create("block.tile_mutex_id", {tile, index}, span);
                    id = Temp("id", getter->GetType(), span);
                    setup.push_back(std::make_shared<AssignStmt>(As<Var>(id), getter, span));
                }
                for (const auto& old : prior) {
                    if (old.tile == tile)
                        continue; // Slots in one binding are unique.
                    bool overlap = std::any_of(effective.begin(), effective.end(),
                                               [&](int64_t value) { return old.candidates.count(value) != 0; });
                    if (!overlap)
                        continue;
                    auto compare = MakeNe(id, old.id, span);
                    guard = guard ? MakeAnd(guard, compare, span) : compare;
                }
                if (guard) {
                    auto var = Temp("needed", guard->GetType(), span);
                    setup.push_back(std::make_shared<AssignStmt>(var, guard, span));
                    guard = var;
                }
                for (bool lock : {true, false}) {
                    auto call = OpRegistry::GetInstance().Create(
                        lock ? "system.mutex_lock_dyn" : "system.mutex_unlock_dyn", {id},
                        {{"pipe", static_cast<int>(pipe)}}, span);
                    StmtPtr sync = std::make_shared<EvalStmt>(call, span);
                    if (guard) {
                        auto body = std::make_shared<SeqStmts>(
                            std::vector<StmtPtr>{sync, std::make_shared<YieldStmt>(span)}, span);
                        sync = std::make_shared<IfStmt>(guard, body, std::nullopt, std::vector<VarPtr>{}, span);
                    }
                    (lock ? locks : unlocks).push_back(sync);
                }
                prior.push_back({tile, id, effective});
            }
        }
        if (locks.empty())
            return stmt;
        setup.insert(setup.end(), locks.begin(), locks.end());
        setup.push_back(stmt);
        setup.insert(setup.end(), unlocks.begin(), unlocks.end());
        return SeqStmts::Flatten(std::move(setup), span);
    }

    StmtPtr Operation(const StmtPtr& stmt, const ExprPtr& expr)
    {
        auto call = As<Call>(expr);
        if (!call)
            return stmt;
        auto pipe = Pipe(call);
        if (!pipe)
            return stmt;
        std::vector<ExprPtr> tiles;
        for (const auto& arg : call->args_) {
            if (!SkipAcc(call) || Memory(arg) != MemorySpace::Acc)
                AddTiles(arg, tiles);
        }
        return Wrap(stmt, tiles, *pipe);
    }
    StmtPtr VisitStmt_(const AssignStmtPtr& op) override { return Operation(op, op->value_); }
    StmtPtr VisitStmt_(const EvalStmtPtr& op) override { return Operation(op, op->expr_); }
    StmtPtr VisitStmt_(const SectionStmtPtr& op) override
    {
        if (op->sectionKind_ != SectionKind::VF)
            return IRMutator::VisitStmt_(op);
        for (const auto& stmt : op->body_->stmts_) {
            auto eval = As<EvalStmt>(stmt);
            auto call = eval ? As<Call>(eval->expr_) : nullptr;
            if (call && call->name_ == "system.vf_scope_args") {
                std::vector<ExprPtr> tiles;
                for (const auto& arg : call->args_)
                    AddTiles(arg, tiles);
                // All mutex reads, guards and locks precede the VF scope.
                return Wrap(op, tiles, PipeType::V);
            }
        }
        return op;
    }
};
} // namespace

Pass pass::InsertSync()
{
    return CreateProgramPass(
        [](const ProgramPtr& program) -> ProgramPtr {
            auto functions = program->functions_;
            for (auto& entry : functions) {
                auto& function = entry.second;
                if (function->funcType_ == FunctionType::SIMT_VF || function->funcType_ == FunctionType::SIMT_CALLEE ||
                    function->funcType_ == FunctionType::ORCHESTRATION)
                    continue;
                SyncInserter inserter(function);
                auto body = inserter.VisitStmt(function->body_);
                if (body != function->body_)
                    function = std::make_shared<Function>(function->name_, function->params_, function->returnTypes_,
                                                          body, function->span_, function->funcType_, function->entry_,
                                                          function->attrs_);
            }
            return std::make_shared<Program>(std::move(functions), program->name_, program->span_, program->debugInfo_);
        },
        "InsertSync");
}
} // namespace pypto::ir
