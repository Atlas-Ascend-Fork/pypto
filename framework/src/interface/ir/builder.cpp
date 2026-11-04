/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ir/builder.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "core/error.h"
#include "core/logging.h"
#include "tilefwk/error.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/program.h"
#include "ir/scalar_expr_ops.h"
#include "ir/span.h"
#include "ir/stmt.h"
#include "ir/type.h"

namespace pypto {
namespace ir {
namespace {

StmtPtr MakeStmtBody(const std::vector<StmtPtr>& stmts, const Span& span)
{
    if (stmts.empty()) {
        return std::make_shared<SeqStmts>(std::vector<StmtPtr>(), span);
    }
    if (stmts.size() == 1) {
        return stmts[0];
    }
    return std::make_shared<SeqStmts>(stmts, span);
}

Span MakeCombinedSpan(const Span& beginSpan, const Span& endSpan)
{
    return Span(beginSpan.Filename(), beginSpan.BeginLine(), beginSpan.BeginColumn(), endSpan.EndLine(),
                endSpan.EndColumn());
}

/**
 * \brief Reconcile the branch value types of an if statement's SSA return vars.
 *
 * A ternary expression ``a if cond else b`` lowers to an if statement whose then/else
 * branches each yield their value into a phi return var. When the two branches produce
 * differing but same-category scalar dtypes (e.g. ``INT64`` vs ``UINT64``), promote
 * both yielded values according to the scalar promotion rules and
 * retype the phi return var to match, mirroring the promotion binary operators already
 * perform. Non-scalar or cross-category (int vs float) mismatches are left untouched so
 * the type verifier still reports them.
 *
 * \param thenStmts Statements of the then branch (last one is the YieldStmt); mutated in place.
 * \param elseStmts Statements of the else branch (last one is the YieldStmt); mutated in place.
 * \param returnVars Phi return vars; retyped in place when a branch pair is promoted.
 */
void ReconcileIfReturnVarTypes(std::vector<StmtPtr>& thenStmts, std::vector<StmtPtr>& elseStmts,
                               std::vector<VarPtr>& returnVars)
{
    if (returnVars.empty() || thenStmts.empty() || elseStmts.empty()) {
        return;
    }
    auto thenYield = std::dynamic_pointer_cast<const YieldStmt>(thenStmts.back());
    auto elseYield = std::dynamic_pointer_cast<const YieldStmt>(elseStmts.back());
    if (!thenYield || !elseYield) {
        return;
    }
    const size_t count = returnVars.size();
    if (thenYield->value_.size() != count || elseYield->value_.size() != count) {
        return;
    }

    std::vector<ExprPtr> thenValues = thenYield->value_;
    std::vector<ExprPtr> elseValues = elseYield->value_;
    bool changed = false;
    for (size_t i = 0; i < count; ++i) {
        const ExprPtr& thenValue = thenValues[i];
        const ExprPtr& elseValue = elseValues[i];
        if (!thenValue || !elseValue) {
            continue;
        }
        auto thenScalar = std::dynamic_pointer_cast<const ScalarType>(thenValue->GetType());
        auto elseScalar = std::dynamic_pointer_cast<const ScalarType>(elseValue->GetType());
        if (!thenScalar || !elseScalar) {
            continue;
        }
        DataType thenDtype = thenScalar->dtype_;
        DataType elseDtype = elseScalar->dtype_;
        if (thenDtype == elseDtype) {
            continue;
        }
        // Only reconcile a matching numeric category (both int or both float); bool and
        // int/float mismatches stay errors for the verifier to surface.
        bool bothInt = thenDtype.IsInt() && elseDtype.IsInt();
        bool bothFloat = thenDtype.IsFloat() && elseDtype.IsFloat();
        if (!bothInt && !bothFloat) {
            continue;
        }
        DataType promoted = PromoteSameCategoryDtype(thenDtype, elseDtype, "ternary", returnVars[i]->span_);
        thenValues[i] = MaybeCast(thenValue, promoted, thenYield->span_);
        elseValues[i] = MaybeCast(elseValue, promoted, elseYield->span_);
        returnVars[i] = std::make_shared<Var>(returnVars[i]->name_, std::make_shared<ScalarType>(promoted),
                                              returnVars[i]->span_);
        changed = true;
    }
    if (!changed) {
        return;
    }
    thenStmts.back() = std::make_shared<YieldStmt>(std::move(thenValues), thenYield->span_);
    elseStmts.back() = std::make_shared<YieldStmt>(std::move(elseValues), elseYield->span_);
}

std::string MakeLoopReturnVarsMismatchMessage(const char* loopKind, size_t iterArgCount, size_t returnVarCount)
{
    std::ostringstream oss;
    oss << loopKind << " loop has " << iterArgCount << " iteration arguments but " << returnVarCount
        << " return variables. They must match.";
    return oss.str();
}
} // namespace

// ========== IRBuilder Implementation ==========

IRBuilder::IRBuilder() = default;

// ========== Function Building ==========

void IRBuilder::BeginFunction(const std::string& name, const Span& span, FunctionType type)
{
    if (InFunction()) {
        throw RuntimeError("Cannot begin function '" + name + "': already inside function '" +
                           static_cast<FunctionContext*>(CurrentContext())->GetName() + "' at " +
                           CurrentContext()->GetBeginSpan().ToString());
    }

    contextStack_.push_back(std::make_shared<FunctionContext>(name, span, type));
}

VarPtr IRBuilder::FuncArg(const std::string& name, const TypePtr& type, const Span& span)
{
    ValidateInFunction("FuncArg");

    auto var = std::make_shared<ir::Var>(name, type, span);
    static_cast<FunctionContext*>(CurrentContext())->AddParam(var);
    return var;
}

void IRBuilder::ReturnType(const TypePtr& type)
{
    ValidateInFunction("ReturnType");
    static_cast<FunctionContext*>(CurrentContext())->AddReturnType(type);
}

FunctionPtr IRBuilder::EndFunction(const Span& endSpan)
{
    ValidateInFunction("EndFunction");

    auto* funcCtx = static_cast<FunctionContext*>(CurrentContext());

    // Build body from accumulated statements
    StmtPtr body;
    const auto& stmts = funcCtx->GetStmts();
    if (stmts.empty()) {
        // Empty body - create empty SeqStmts
        body = std::make_shared<SeqStmts>(std::vector<StmtPtr>(), endSpan);
    } else if (stmts.size() == 1) {
        body = stmts[0];
    } else {
        body = std::make_shared<SeqStmts>(stmts, endSpan);
    }

    // Combine begin and end spans
    auto combinedSpan = MakeCombinedSpan(funcCtx->GetBeginSpan(), endSpan);

    // Create function
    auto func = std::make_shared<Function>(funcCtx->GetName(), funcCtx->GetParams(), funcCtx->GetReturnTypes(), body,
                                           combinedSpan, funcCtx->GetFuncType(), false, funcCtx->GetAttrs());

    // Pop context
    contextStack_.pop_back();

    return func;
}

// ========== For Loop Building ==========

void IRBuilder::BeginForLoop(const VarPtr& loopVar, const ExprPtr& start, const ExprPtr& stop, const ExprPtr& step,
                             const Span& span)
{
    if (contextStack_.empty()) {
        throw RuntimeError("Cannot begin for loop: not inside a function or another valid context at " +
                           span.ToString());
    }

    contextStack_.push_back(std::make_shared<ForLoopContext>(loopVar, start, stop, step, span));
}

void IRBuilder::AddIterArg(const IterArgPtr& iterArg)
{
    ValidateInLoop("AddIterArg");
    static_cast<ForLoopContext*>(CurrentContext())->AddIterArg(iterArg);
}

void IRBuilder::AddReturnVar(const VarPtr& var)
{
    ValidateInLoop("AddReturnVar");
    static_cast<ForLoopContext*>(CurrentContext())->AddReturnVar(var);
}

StmtPtr IRBuilder::EndForLoop(const Span& endSpan)
{
    ValidateInLoop("EndForLoop");

    auto* loopCtx = static_cast<ForLoopContext*>(CurrentContext());

    // Validate iter_args and return_vars match
    if (loopCtx->GetIterArgs().size() != loopCtx->GetReturnVars().size()) {
        // Pop context before throwing to maintain stack consistency
        contextStack_.pop_back();

        throw RuntimeError(
            MakeLoopReturnVarsMismatchMessage("For", loopCtx->GetIterArgs().size(), loopCtx->GetReturnVars().size()));
    }

    auto body = MakeStmtBody(loopCtx->GetStmts(), endSpan);

    auto combinedSpan = MakeCombinedSpan(loopCtx->GetBeginSpan(), endSpan);

    // Create for statement
    auto forStmt = std::make_shared<ForStmt>(loopCtx->GetLoopVar(), loopCtx->GetStart(), loopCtx->GetStop(),
                                             loopCtx->GetStep(), loopCtx->GetIterArgs(), body, loopCtx->GetReturnVars(),
                                             combinedSpan);

    // Pop context
    contextStack_.pop_back();

    // Emit to parent context if it exists
    if (!contextStack_.empty()) {
        CurrentContext()->AddStmt(forStmt);
    }

    return forStmt;
}

// ========== While Loop Building ==========

void IRBuilder::BeginWhileLoop(const ExprPtr& condition, const Span& span)
{
    if (contextStack_.empty()) {
        throw RuntimeError("Cannot begin while loop: not inside a function or another valid context at " +
                           span.ToString());
    }

    contextStack_.push_back(std::make_shared<WhileLoopContext>(condition, span));
}

void IRBuilder::AddWhileIterArg(const IterArgPtr& iterArg)
{
    ValidateInWhileLoop("AddWhileIterArg");
    static_cast<WhileLoopContext*>(CurrentContext())->AddIterArg(iterArg);
}

void IRBuilder::AddWhileReturnVar(const VarPtr& var)
{
    ValidateInWhileLoop("AddWhileReturnVar");
    static_cast<WhileLoopContext*>(CurrentContext())->AddReturnVar(var);
}

void IRBuilder::SetWhileLoopCondition(const ExprPtr& condition)
{
    ValidateInWhileLoop("SetWhileLoopCondition");
    static_cast<WhileLoopContext*>(CurrentContext())->SetCondition(condition);
}

StmtPtr IRBuilder::EndWhileLoop(const Span& endSpan)
{
    ValidateInWhileLoop("EndWhileLoop");

    auto* loopCtx = static_cast<WhileLoopContext*>(CurrentContext());

    // Validate iter_args and return_vars match
    if (loopCtx->GetIterArgs().size() != loopCtx->GetReturnVars().size()) {
        // Pop context before throwing to maintain stack consistency
        contextStack_.pop_back();

        throw RuntimeError(
            MakeLoopReturnVarsMismatchMessage("While", loopCtx->GetIterArgs().size(), loopCtx->GetReturnVars().size()));
    }

    auto body = MakeStmtBody(loopCtx->GetStmts(), endSpan);
    auto combinedSpan = MakeCombinedSpan(loopCtx->GetBeginSpan(), endSpan);
    // Create while statement
    auto whileStmt = std::make_shared<WhileStmt>(loopCtx->GetCondition(), loopCtx->GetIterArgs(), body,
                                                 loopCtx->GetReturnVars(), combinedSpan);

    // Pop context
    contextStack_.pop_back();

    // Emit to parent context if it exists
    if (!contextStack_.empty()) {
        CurrentContext()->AddStmt(whileStmt);
    }

    return whileStmt;
}

// ========== If Statement Building ==========

void IRBuilder::BeginIf(const ExprPtr& condition, const Span& span)
{
    IRCHECK(!contextStack_.empty()) << "Cannot begin if statement: not inside a function or another valid context at "
                                    << span.ToString();
    contextStack_.push_back(std::make_shared<IfStmtContext>(condition, span));
}

void IRBuilder::BeginElse(const Span& span)
{
    ValidateInIf("BeginElse");

    auto* ifCtx = static_cast<IfStmtContext*>(CurrentContext());
    IRCHECK(!ifCtx->InElseBranch()) << "Cannot begin else branch: already in else branch at " << span.ToString();

    ifCtx->BeginElseBranch();
}

void IRBuilder::AddIfReturnVar(const VarPtr& var)
{
    ValidateInIf("AddIfReturnVar");
    static_cast<IfStmtContext*>(CurrentContext())->AddReturnVar(var);
}

StmtPtr IRBuilder::EndIf(const Span& endSpan)
{
    ValidateInIf("EndIf");

    auto* ifCtx = static_cast<IfStmtContext*>(CurrentContext());

    std::vector<StmtPtr> thenStmts = ifCtx->GetStmts();
    std::vector<StmtPtr> elseStmts = ifCtx->GetElseStmts();
    std::vector<VarPtr> returnVars = ifCtx->GetReturnVars();

    // Promote differing but same-category scalar branch types to a common dtype
    // before building the phi.
    if (ifCtx->InElseBranch()) {
        ReconcileIfReturnVarTypes(thenStmts, elseStmts, returnVars);
    }

    // Build then body
    StmtPtr thenBody = MakeStmtBody(thenStmts, endSpan);

    // Build else body (optional)
    std::optional<StmtPtr> elseBody;
    if (ifCtx->InElseBranch() && !elseStmts.empty()) {
        elseBody = MakeStmtBody(elseStmts, endSpan);
    }

    // Combine begin and end spans
    auto combinedSpan = MakeCombinedSpan(ifCtx->GetBeginSpan(), endSpan);

    // Create if statement
    auto ifStmt = std::make_shared<IfStmt>(ifCtx->GetCondition(), thenBody, elseBody, returnVars, combinedSpan);
    // Pop context
    contextStack_.pop_back();

    // Emit to parent context if it exists
    if (!contextStack_.empty()) {
        CurrentContext()->AddStmt(ifStmt);
    }

    return ifStmt;
}

// ========== Section Building ==========

void IRBuilder::BeginSection(SectionKind sectionKind, const Span& span)
{
    CHECK(!contextStack_.empty()) << "Cannot begin section: not inside a function or another valid context at "
                                  << span.ToString();
    contextStack_.push_back(std::make_shared<SectionContext>(sectionKind, span));
}

StmtPtr IRBuilder::EndSection(const Span& endSpan)
{
    CHECK(!contextStack_.empty() && CurrentContext()->GetType() == BuildContext::Type::SECTION)
        << "Cannot end section: not inside a section context at " << endSpan.ToString();

    auto* sectionCtx = static_cast<SectionContext*>(CurrentContext());

    // Build body
    StmtPtr body;
    const auto& stmts = sectionCtx->GetStmts();
    if (stmts.empty()) {
        body = std::make_shared<SeqStmts>(std::vector<StmtPtr>(), endSpan);
    } else if (stmts.size() == 1) {
        body = stmts[0];
    } else {
        body = std::make_shared<SeqStmts>(stmts, endSpan);
    }

    // Combine spans
    auto combinedSpan = MakeCombinedSpan(sectionCtx->GetBeginSpan(), endSpan);

    // Create section statement
    auto sectionStmt = std::make_shared<SectionStmt>(sectionCtx->GetSectionKind(), body, combinedSpan);

    // Pop context
    contextStack_.pop_back();

    // Emit to parent context if it exists
    if (!contextStack_.empty()) {
        CurrentContext()->AddStmt(sectionStmt);
    }

    return sectionStmt;
}

// ========== Program Building ==========

void IRBuilder::BeginProgram(const std::string& name, const Span& span)
{
    if (InProgram()) {
        throw RuntimeError("Cannot begin program '" + name + "': already inside program '" +
                           static_cast<ProgramContext*>(CurrentContext())->GetName() + "' at " +
                           CurrentContext()->GetBeginSpan().ToString());
    }

    contextStack_.push_back(std::make_shared<ProgramContext>(name, span));
}

void IRBuilder::AddFunction(const FunctionPtr& func)
{
    ValidateInProgram("AddFunction");
    static_cast<ProgramContext*>(CurrentContext())->AddFunction(func);
}

void IRBuilder::SetInsertPoint(std::shared_ptr<InsertPoint> ctx) { contextStack_.push_back(ctx); }

void IRBuilder::ClearInsertPoint() { contextStack_.pop_back(); }

ProgramPtr IRBuilder::EndProgram(const Span& endSpan)
{
    ValidateInProgram("EndProgram");

    auto* progCtx = static_cast<ProgramContext*>(CurrentContext());

    // Combine begin and end spans
    auto combinedSpan = MakeCombinedSpan(progCtx->GetBeginSpan(), endSpan);

    // Create program from functions vector
    auto program = std::make_shared<Program>(progCtx->GetFunctions(), progCtx->GetName(), combinedSpan);

    // Pop context
    contextStack_.pop_back();

    return program;
}

bool IRBuilder::InProgram() const
{
    for (const auto& ctx : contextStack_) {
        if (ctx->GetType() == BuildContext::Type::PROGRAM) {
            return true;
        }
    }
    return false;
}

std::vector<TypePtr> IRBuilder::GetFunctionReturnTypes(const std::string& funcName) const
{
    // Find the program context in the stack
    for (const auto& ctx : contextStack_) {
        if (ctx->GetType() == BuildContext::Type::PROGRAM) {
            auto* progCtx = static_cast<const ProgramContext*>(ctx.get());
            return progCtx->GetReturnTypes(funcName);
        }
    }
    return {};
}

// ========== Statement Recording ==========

void IRBuilder::Emit(const StmtPtr& stmt)
{
    if (contextStack_.empty()) {
        throw RuntimeError("Cannot emit statement: not inside any context");
    }

    auto* ctx = CurrentContext();
    ctx->AddStmt(stmt);
}

AssignStmtPtr IRBuilder::Assign(const VarPtr& var, const ExprPtr& value, const Span& span)
{
    auto assign = std::make_shared<AssignStmt>(var, value, span);
    Emit(assign);
    return assign;
}

VarPtr IRBuilder::Var(const std::string& name, const TypePtr& type, const Span& span)
{
    return std::make_shared<ir::Var>(name, type, span);
}

ReturnStmtPtr IRBuilder::Return(const std::vector<ExprPtr>& values, const Span& span)
{
    auto returnStmt = std::make_shared<ReturnStmt>(values, span);
    Emit(returnStmt);
    return returnStmt;
}

ReturnStmtPtr IRBuilder::Return(const Span& span)
{
    auto returnStmt = std::make_shared<ReturnStmt>(span);
    Emit(returnStmt);
    return returnStmt;
}

BreakStmtPtr IRBuilder::Break(const Span& span)
{
    auto breakStmt = std::make_shared<BreakStmt>(span);
    Emit(breakStmt);
    return breakStmt;
}

ContinueStmtPtr IRBuilder::Continue(const Span& span)
{
    auto continueStmt = std::make_shared<ContinueStmt>(span);
    Emit(continueStmt);
    return continueStmt;
}

void IRBuilder::UpdateJumpValues(const StmtPtr& jumpOp, std::vector<ExprPtr> values)
{
    IRCHECK(jumpOp != nullptr) << "Jump statement must not be null";
    switch (jumpOp->GetKind()) {
        case ObjectKind::YieldStmt:
            const_cast<YieldStmt*>(static_cast<const YieldStmt*>(jumpOp.get()))->value_ = std::move(values);
            return;
        case ObjectKind::BreakStmt:
            const_cast<BreakStmt*>(static_cast<const BreakStmt*>(jumpOp.get()))->value_ = std::move(values);
            return;
        case ObjectKind::ContinueStmt:
            const_cast<ContinueStmt*>(static_cast<const ContinueStmt*>(jumpOp.get()))->value_ = std::move(values);
            return;
        default:
            IRCHECK(false) << "Expected YieldStmt, BreakStmt, or ContinueStmt, got " << jumpOp->TypeName();
    }
}

// ========== Context State Queries ==========

BuildContext* IRBuilder::CurrentContext()
{
    if (contextStack_.empty()) {
        return nullptr;
    }
    return contextStack_.back().get();
}

bool IRBuilder::InFunction() const
{
    for (const auto& ctx : contextStack_) {
        if (ctx->GetType() == BuildContext::Type::FUNCTION) {
            return true;
        }
    }
    return false;
}

bool IRBuilder::InLoop() const
{
    if (contextStack_.empty()) {
        return false;
    }
    return contextStack_.back()->GetType() == BuildContext::Type::FOR_LOOP;
}

bool IRBuilder::InIf() const
{
    if (contextStack_.empty()) {
        return false;
    }
    return contextStack_.back()->GetType() == BuildContext::Type::IF_STMT;
}

bool IRBuilder::InWhileLoop() const
{
    if (contextStack_.empty()) {
        return false;
    }
    return contextStack_.back()->GetType() == BuildContext::Type::WHILE_LOOP;
}

// ========== Private Helpers ==========

template <typename T>
T* IRBuilder::GetCurrentContextAs()
{
    auto* ctx = CurrentContext();
    if (!ctx) {
        return nullptr;
    }
    return dynamic_cast<T*>(ctx);
}

void IRBuilder::ValidateInFunction(const std::string& operation)
{
    IRCHECK(InFunction()) << operation << " can only be called inside a function context";
    IRCHECK(CurrentContext()->GetType() == BuildContext::Type::FUNCTION)
        << operation << " must be called directly in function context, not nested";
}

void IRBuilder::ValidateInLoop(const std::string& operation)
{
    IRCHECK(InLoop()) << operation << " can only be called inside a for loop context";
}

void IRBuilder::ValidateInIf(const std::string& operation)
{
    IRCHECK(InIf()) << operation << " can only be called inside an if statement context";
}

void IRBuilder::ValidateInWhileLoop(const std::string& operation)
{
    IRCHECK(InWhileLoop()) << operation << " can only be called inside a while loop context";
}

void IRBuilder::ValidateInProgram(const std::string& operation)
{
    IRCHECK(InProgram()) << operation << " can only be called inside a program context";
}

// ========== ProgramContext Implementation ==========

void ProgramContext::AddFunction(const FunctionPtr& func)
{
    INTERNAL_CHECK(func) << "Cannot add null function to program";

    // Store return types for this function
    returnTypes_[func->name_] = func->returnTypes_;

    functions_.push_back(func);
}

std::vector<TypePtr> ProgramContext::GetReturnTypes(const std::string& funcName) const
{
    auto it = returnTypes_.find(funcName);
    if (it != returnTypes_.end()) {
        return it->second;
    }
    return {};
}

} // namespace ir
} // namespace pypto
