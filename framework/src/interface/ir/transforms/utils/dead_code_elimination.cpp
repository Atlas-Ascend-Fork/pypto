/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "ir/transforms/utils/dead_code_elimination.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

#include "interface/tensor/ir.h"
#include "interface/tensor/ir_tensor_op_rebuild.h"
#include "interface/tensor/irbuilder.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/stmt.h"
#include "ir/transforms/base/visitor.h"
#include "ir/type.h"
#include "stmt_utils.h"

namespace pypto {
namespace ir {
namespace dce {

using utils::CollectStmtVarRefs;
using utils::CollectVarUses;
using utils::FlattenBody;
using utils::MakeSeqBody;
using utils::StmtsEqual;

void CollectAllAssignStmts(const std::vector<StmtPtr>& stmts, std::vector<std::shared_ptr<const AssignStmt>>& assigns)
{
    for (const auto& stmt : stmts) {
        if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
            assigns.push_back(assign);
        }
        if (auto forStmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
            CollectAllAssignStmts(FlattenBody(forStmt->body_), assigns);
        } else if (auto ifStmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
            CollectAllAssignStmts(FlattenBody(ifStmt->thenBody_), assigns);
            if (ifStmt->elseBody_.has_value()) {
                CollectAllAssignStmts(FlattenBody(ifStmt->elseBody_.value()), assigns);
            }
        } else if (auto whileStmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
            CollectAllAssignStmts(FlattenBody(whileStmt->body_), assigns);
        } else if (auto sectionStmt = std::dynamic_pointer_cast<const SectionStmt>(stmt)) {
            if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(sectionStmt->body_)) {
                CollectAllAssignStmts(seq->stmts_, assigns);
            } else if (sectionStmt->body_) {
                std::vector<StmtPtr> single = {sectionStmt->body_};
                CollectAllAssignStmts(single, assigns);
            }
        }
    }
}

namespace {

using RemovablePredicate = std::function<bool(const StmtPtr&)>;

/// Collect live-root variables.
/// A statement is a "live root" when it is NOT classified as a removal
/// candidate by `is_removable`. Its own Var references (expressions and
/// direct fields, not nested-body refs) are added to the live set; the
/// nested body, if any, is recursed into separately so its own candidate
/// assignments remain eligible for removal.
void FindLiveRootsRecursiveImpl(const std::vector<StmtPtr>& stmts, const RemovablePredicate& isRemovable,
                                std::unordered_set<const Var*>& live)
{
    auto collectExprRefs = [&](const ExprPtr& expr) {
        if (!expr)
            return;
        auto refs = CollectVarUses(expr);
        live.insert(refs.begin(), refs.end());
    };
    auto collectIterArgRefs = [&](const auto& loopStmt) {
        for (const auto& iterArg : loopStmt->iterArgs_) {
            collectExprRefs(iterArg->initValue_);
        }
    };

    for (const auto& stmt : stmts) {
        // Live-root: non-candidate leaf statements contribute their refs.
        // ContinueStmt/BreakStmt are always non-removable — their carried vars
        // must be considered live even when no other statement uses them.
        bool isLeaf = std::dynamic_pointer_cast<const AssignStmt>(stmt) ||
                      std::dynamic_pointer_cast<const EvalStmt>(stmt) ||
                      std::dynamic_pointer_cast<const ReturnStmt>(stmt) ||
                      std::dynamic_pointer_cast<const YieldStmt>(stmt) ||
                      std::dynamic_pointer_cast<const TensorOpStmt>(stmt) ||
                      std::dynamic_pointer_cast<const ScalarOpStmt>(stmt) ||
                      std::dynamic_pointer_cast<const ContinueStmt>(stmt) ||
                      std::dynamic_pointer_cast<const BreakStmt>(stmt);
        if (isLeaf && !isRemovable(stmt)) {
            auto allRefs = CollectStmtVarRefs(stmt);
            // Do not keep an unconsumed result token alive just because its TensorOp is non-removable.
            // When the assemble-new-logical-tensor mechanism is active, result tokens are inferred by
            // InferTokenPass and must survive DCE so that RemoveRedundantTokenPass can analyse them.
            if (!npu::tile_fwk::IRContext::Get().AssembleNewLogicalTensor()) {
                if (auto tensorOp = std::dynamic_pointer_cast<const TensorOpStmt>(stmt)) {
                    for (const auto& token : tensorOp->result_token_) {
                        allRefs.erase(token.get());
                    }
                }
            }
            live.insert(allRefs.begin(), allRefs.end());
        }

        // Control-flow headers: add direct-field refs (bounds, conditions,
        // iter-arg initializers) but defer body traversal to the recursive
        // call so nested candidate assignments remain eligible for removal.
        if (auto forStmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
            collectExprRefs(forStmt->start_);
            collectExprRefs(forStmt->stop_);
            collectExprRefs(forStmt->step_);
            collectIterArgRefs(forStmt);
            FindLiveRootsRecursiveImpl(FlattenBody(forStmt->body_), isRemovable, live);
        } else if (auto ifStmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
            collectExprRefs(ifStmt->condition_);
            FindLiveRootsRecursiveImpl(FlattenBody(ifStmt->thenBody_), isRemovable, live);
            if (ifStmt->elseBody_.has_value()) {
                FindLiveRootsRecursiveImpl(FlattenBody(ifStmt->elseBody_.value()), isRemovable, live);
            }
        } else if (auto whileStmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
            collectExprRefs(whileStmt->condition_);
            collectIterArgRefs(whileStmt);
            FindLiveRootsRecursiveImpl(FlattenBody(whileStmt->body_), isRemovable, live);
        } else if (auto sectionStmt = std::dynamic_pointer_cast<const SectionStmt>(stmt)) {
            if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(sectionStmt->body_)) {
                FindLiveRootsRecursiveImpl(seq->stmts_, isRemovable, live);
            } else if (sectionStmt->body_) {
                std::vector<StmtPtr> single = {sectionStmt->body_};
                FindLiveRootsRecursiveImpl(single, isRemovable, live);
            }
        }
    }
}

std::vector<StmtPtr> FilterDeadCodeImpl(const std::vector<StmtPtr>& stmts, const RemovablePredicate& isRemovable,
                                        const std::unordered_set<const Var*>& live)
{
    std::vector<StmtPtr> result;

    // Check if a body is a single break or continue statement (trivial loop body).
    auto isTrivialLoopBody = [](const std::vector<StmtPtr>& body) -> bool {
        return body.size() == 1 && (ir::As<BreakStmt>(body[0]) || ir::As<ContinueStmt>(body[0]));
    };

    // Check if all statements in a list are YieldStmt.
    auto allYield = [](const std::vector<StmtPtr>& body) -> bool {
        return body.empty() || (body.size() == 1 && ir::As<YieldStmt>(body[0]));
    };

    for (const auto& stmt : stmts) {
        if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
            if (isRemovable(stmt) && !live.count(assign->var_.get()))
                continue;
            result.push_back(stmt);
        } else if (auto tensorOp = std::dynamic_pointer_cast<const TensorOpStmt>(stmt)) {
            if (isRemovable(stmt)) {
                bool anyLive = false;
                for (const auto& r : tensorOp->result_) {
                    if (live.count(r.get())) {
                        anyLive = true;
                        break;
                    }
                }
                for (const auto& token : tensorOp->result_token_) {
                    if (live.count(token.get())) {
                        anyLive = true;
                        break;
                    }
                }
                if (!anyLive)
                    continue;
            }
            std::vector<VarPtr> liveTokens;
            if (npu::tile_fwk::IRContext::Get().AssembleNewLogicalTensor()) {
                // Preserve all result tokens so RemoveRedundantTokenPass can analyse the full
                // token dependency graph.  Only tensor results determine statement liveness.
                liveTokens = tensorOp->result_token_;
            } else {
                for (const auto& token : tensorOp->result_token_) {
                    if (live.count(token.get())) {
                        liveTokens.push_back(token);
                    }
                }
            }
            if (liveTokens.size() != tensorOp->result_token_.size()) {
                result.push_back(npu::tile_fwk::RebuildTensorOpStmt(tensorOp, tensorOp->result_, std::move(liveTokens),
                                                                    tensorOp->args_, tensorOp->tokens_,
                                                                    tensorOp->span_));
            } else {
                result.push_back(stmt);
            }
        } else if (auto scalarOp = std::dynamic_pointer_cast<const ScalarOpStmt>(stmt)) {
            if (isRemovable(stmt) && !live.count(scalarOp->result_.get()))
                continue;
            result.push_back(stmt);
        } else if (auto forStmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
            auto original = FlattenBody(forStmt->body_);
            auto filtered = FilterDeadCodeImpl(original, isRemovable, live);
            // Drop no-op loops: body is just break/continue, no return vars.
            if (forStmt->returnVars_.empty() && isTrivialLoopBody(filtered)) {
                continue;
            }
            if (StmtsEqual(filtered, original)) {
                result.push_back(stmt);
                continue;
            }
            auto newBody = MakeSeqBody(filtered, forStmt->span_);
            auto newFor = std::make_shared<const ForStmt>(forStmt->loopVar_, forStmt->start_, forStmt->stop_,
                                                          forStmt->step_, forStmt->iterArgs_, newBody,
                                                          forStmt->returnVars_, forStmt->span_, forStmt->attrs_);
            result.push_back(newFor);
        } else if (auto ifStmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
            auto thenOrig = FlattenBody(ifStmt->thenBody_);
            auto filteredThen = FilterDeadCodeImpl(thenOrig, isRemovable, live);
            bool thenUnchanged = StmtsEqual(filteredThen, thenOrig);
            bool elseUnchanged = true;
            std::optional<SeqStmtsPtr> filteredElse;
            if (ifStmt->elseBody_.has_value()) {
                auto elseOrig = FlattenBody(ifStmt->elseBody_.value());
                auto fe = FilterDeadCodeImpl(elseOrig, isRemovable, live);
                elseUnchanged = StmtsEqual(fe, elseOrig);
                filteredElse = MakeSeqBody(fe, ifStmt->span_);
            }
            // Drop no-op if-stmts: branches only yield, no return vars.
            bool thenAllYield = allYield(filteredThen);
            bool elseAllYield = !filteredElse.has_value() || allYield(FlattenBody(*filteredElse));
            if (ifStmt->returnVars_.empty() && thenAllYield && elseAllYield) {
                continue;
            }
            if (thenUnchanged && elseUnchanged) {
                result.push_back(stmt);
                continue;
            }
            std::optional<StmtPtr> elseBodyStmt;
            if (filteredElse) {
                elseBodyStmt = *filteredElse;
            }
            auto newIf = std::make_shared<const IfStmt>(
                ifStmt->condition_,
                filteredThen.empty() ? std::make_shared<const SeqStmts>(std::vector<StmtPtr>{}, ifStmt->span_) :
                                       MakeSeqBody(filteredThen, ifStmt->span_),
                elseBodyStmt, ifStmt->returnVars_, ifStmt->span_);
            result.push_back(newIf);
        } else if (auto whileStmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
            auto original = FlattenBody(whileStmt->body_);
            auto filtered = FilterDeadCodeImpl(original, isRemovable, live);
            // Drop no-op loops: body is just break/continue, no return vars.
            if (whileStmt->returnVars_.empty() && isTrivialLoopBody(filtered)) {
                continue;
            }
            if (StmtsEqual(filtered, original)) {
                result.push_back(stmt);
                continue;
            }
            auto newBody = MakeSeqBody(filtered, whileStmt->span_);
            auto newWhile = std::make_shared<const WhileStmt>(whileStmt->condition_, whileStmt->iterArgs_, newBody,
                                                              whileStmt->returnVars_, whileStmt->span_);
            result.push_back(newWhile);
        } else if (auto sectionStmt = std::dynamic_pointer_cast<const SectionStmt>(stmt)) {
            std::vector<StmtPtr> original;
            if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(sectionStmt->body_)) {
                original = seq->stmts_;
            } else if (sectionStmt->body_) {
                original = {sectionStmt->body_};
            }
            auto filtered = FilterDeadCodeImpl(original, isRemovable, live);
            if (StmtsEqual(filtered, original)) {
                result.push_back(stmt);
                continue;
            }
            auto newBody = filtered.empty() ?
                               std::make_shared<const SeqStmts>(std::vector<StmtPtr>{}, sectionStmt->span_) :
                               MakeSeqBody(filtered, sectionStmt->span_);
            auto newSection = std::make_shared<const SectionStmt>(sectionStmt->sectionKind_, newBody,
                                                                  sectionStmt->span_);
            result.push_back(newSection);
        } else {
            result.push_back(stmt);
        }
    }

    return result;
}

/// Predicate for the default `EliminateDeadCode`: any AssignStmt that is not
/// a known side-effect op is a removal candidate.
bool IsRemovableForDefaultDce(const StmtPtr& stmt)
{
    return std::dynamic_pointer_cast<const AssignStmt>(stmt) != nullptr;
}

/// Walk an expression tree and report whether any Call appears.
class CallFinder : public IRVisitor {
public:
    bool found = false;

private:
    using IRVisitor::VisitExpr_;
    void VisitExpr_(const CallPtr& op) override
    {
        found = true;
        IRVisitor::VisitExpr_(op);
    }
};

bool ExprContainsCall(const ExprPtr& expr)
{
    if (!expr)
        return false;
    CallFinder finder;
    finder.VisitExpr(expr);
    return finder.found;
}

/// Predicate for `EliminateDeadScalarAssignments`: an AssignStmt with a
/// scalar-typed LHS whose RHS contains no `Call` anywhere.
bool IsRemovableScalarAssign(const StmtPtr& stmt)
{
    auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt);
    if (!assign)
        return false;
    if (!As<ScalarType>(assign->var_->GetType()))
        return false;
    if (ExprContainsCall(assign->value_))
        return false;
    return true;
}

} // namespace

std::vector<StmtPtr> EliminateDeadCode(const std::vector<StmtPtr>& stmts, const RemovablePredicate& isRemovable)
{
    std::unordered_set<const Var*> live;
    FindLiveRootsRecursiveImpl(stmts, isRemovable, live);

    // Collect all definition sites (AssignStmt, TensorOpStmt, ScalarOpStmt)
    // and cache their def/use pairs for the fixed-point propagation.
    struct MemoryDef {
        const Var* var;
        int memoryId;
    };
    struct DefSite {
        std::vector<const Var*> defs;
        std::vector<MemoryDef> memoryDefs;
        std::unordered_set<const Var*> uses;
    };

    std::vector<DefSite> defSites;

    std::function<void(const std::vector<StmtPtr>&)> collectDefSites = [&](const std::vector<StmtPtr>& stmtsVec) {
        for (const auto& s : stmtsVec) {
            if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(s)) {
                defSites.push_back({{assign->var_.get()}, {}, CollectVarUses(assign->value_)});
            } else if (auto tensorOp = std::dynamic_pointer_cast<const TensorOpStmt>(s)) {
                std::vector<const Var*> defs;
                std::vector<MemoryDef> memoryDefs;
                for (const auto& r : tensorOp->result_) {
                    defs.push_back(r.get());
                    if (auto mid = GetVarMemoryId(r.get())) {
                        memoryDefs.push_back({r.get(), *mid});
                    }
                }
                for (const auto& token : tensorOp->result_token_) {
                    defs.push_back(token.get());
                }
                std::unordered_set<const Var*> uses;
                for (const auto& arg : tensorOp->args_) {
                    auto argUses = CollectVarUses(arg);
                    uses.insert(argUses.begin(), argUses.end());
                }
                for (const auto& r : tensorOp->tokens_) {
                    uses.insert(r.get());
                }
                defSites.push_back({std::move(defs), std::move(memoryDefs), std::move(uses)});
            } else if (auto scalarOp = std::dynamic_pointer_cast<const ScalarOpStmt>(s)) {
                std::unordered_set<const Var*> uses;
                for (const auto& arg : scalarOp->args_) {
                    auto argUses = CollectVarUses(arg);
                    uses.insert(argUses.begin(), argUses.end());
                }
                defSites.push_back({{scalarOp->result_.get()}, {}, std::move(uses)});
            }
            // Recurse into control-flow bodies
            if (auto forS = std::dynamic_pointer_cast<const ForStmt>(s)) {
                collectDefSites(FlattenBody(forS->body_));
            } else if (auto ifS = std::dynamic_pointer_cast<const IfStmt>(s)) {
                collectDefSites(FlattenBody(ifS->thenBody_));
                if (ifS->elseBody_.has_value()) {
                    collectDefSites(FlattenBody(ifS->elseBody_.value()));
                }
            } else if (auto whileS = std::dynamic_pointer_cast<const WhileStmt>(s)) {
                collectDefSites(FlattenBody(whileS->body_));
            } else if (auto secS = std::dynamic_pointer_cast<const SectionStmt>(s)) {
                if (auto seq = std::dynamic_pointer_cast<const SeqStmts>(secS->body_)) {
                    collectDefSites(seq->stmts_);
                } else if (secS->body_) {
                    collectDefSites({secS->body_});
                }
            }
        }
    };

    collectDefSites(stmts);

    // Fixed-point: propagate liveness from defs to their uses.
    bool changed = true;
    while (changed) {
        changed = false;
        auto liveMemoryIds = CollectMemoryIds(live);
        for (size_t i = defSites.size(); i-- > 0;) {
            bool anyDefLive = false;
            for (const auto* def : defSites[i].defs) {
                if (live.count(def)) {
                    anyDefLive = true;
                    break;
                }
            }
            if (!anyDefLive) {
                for (const auto& memoryDef : defSites[i].memoryDefs) {
                    if (liveMemoryIds.count(memoryDef.memoryId)) {
                        if (live.insert(memoryDef.var).second)
                            changed = true;
                        anyDefLive = true;
                    }
                }
            }
            if (!anyDefLive)
                continue;
            for (const Var* ref : defSites[i].uses) {
                if (live.insert(ref).second)
                    changed = true;
            }
        }
    }

    return FilterDeadCodeImpl(stmts, isRemovable, live);
}

std::vector<StmtPtr> EliminateDeadCode(const std::vector<StmtPtr>& stmts)
{
    return EliminateDeadCode(stmts, IsRemovableForDefaultDce);
}

std::vector<StmtPtr> EliminateDeadScalarAssignments(const std::vector<StmtPtr>& stmts)
{
    return EliminateDeadCode(stmts, IsRemovableScalarAssign);
}

} // namespace dce
} // namespace ir
} // namespace pypto
