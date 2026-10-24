/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ir/transforms/litenpu_dynamic_to_static_pass.h"

#include "ir/transforms/base/mutator.h"
#include "ir/transforms/base/visitor.h"
#include "ir/transforms/passes.h"

#include "interface/operation/operation.h"
#include "interface/tensor/ir_tensor_op_rebuild.h"
#include "interface/tensor/logical_tensor.h"

#include "tilefwk/platform.h"
#include "tilefwk/symbolic_scalar.h"

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace pypto::ir {

namespace {

using npu::tile_fwk::AsLogicalTensor;
using npu::tile_fwk::LogicalTensor;
using npu::tile_fwk::LogicalTensorPtr;
using npu::tile_fwk::Operation;
using npu::tile_fwk::RawSymbolicExpression;
using npu::tile_fwk::RawSymbolicImmediate;
using npu::tile_fwk::RawSymbolicScalarPtr;
using npu::tile_fwk::RawSymbolicSymbol;
using npu::tile_fwk::SymbolicScalar;

struct ResidualDynamicDim {
    std::string tensorName;
    size_t dim{0};
    std::string dynDump;
};

class ResidualDynamicShapeCollector : public IRVisitor {
public:
    void VisitVarLike_(const VarPtr& op) override
    {
        Collect(AsLogicalTensor(op));
        IRVisitor::VisitVarLike_(op);
    }

    void Collect(const LogicalTensorPtr& lt)
    {
        if (lt == nullptr || !seen_.insert(lt.get()).second) {
            return;
        }
        for (size_t d = 0; d < lt->shape.size(); ++d) {
            if (lt->shape[d] != -1) {
                continue;
            }
            ResidualDynamicDim issue;
            issue.tensorName = lt->name_;
            issue.dim = d;
            issue.dynDump = (d < lt->dynValidShape_.size()) ? lt->dynValidShape_[d].Dump() :
                                                              std::string("<no dynValidShape>");
            issues_.push_back(std::move(issue));
        }
    }

    std::vector<ResidualDynamicDim> issues_;

private:
    std::unordered_set<const LogicalTensor*> seen_;
};

void CheckNoResidualDynamicShape(const SeqStmtsPtr& seq)
{
    ResidualDynamicShapeCollector collector;
    collector.VisitStmt(seq);
    if (collector.issues_.empty()) {
        return;
    }

    std::string msg = "LitenpuDynamicToStatic: failed to staticize dynamic shape, " +
                      std::to_string(collector.issues_.size()) + " residual -1 dim(s) remain:\n";
    for (const auto& issue : collector.issues_) {
        msg += "  - tensor '" + issue.tensorName + "' dim " + std::to_string(issue.dim) +
               ": dynValidShape = " + issue.dynDump + "\n";
    }
    msg += "Kirin requires a fully static shape (no -1).";
    PASS_ASSERT(collector.issues_.empty()) << msg;
}

/**
 * \brief Converts device-side dynamic expressions to static expressions.
 *
 * The main work is static loop unrolling: a ForStmt with concrete start/stop/step is
 * expanded into its body repeated once per iteration, with the loop variable (and the
 * loop-carried iterArgs) substituted by their concrete/current values. Dynamic shape and
 * offset expressions are folded to static values via SymbolicScalar substitution.
 */
class LitenpuDynamicToStaticConverter : public IRMutator {
public:
    using IRMutator::VisitExpr_;
    using IRMutator::VisitStmt_;

    SeqStmtsPtr Apply(const SeqStmtsPtr& seq, const std::vector<ir::VarPtr>& params)
    {
        if (!npu::tile_fwk::IsLiteNPU(npu::tile_fwk::Platform::Instance().GetSoc().GetNPUArch())) {
            return seq;
        }
        BuildStaticShapeByTensor(params);
        auto result = SeqStmts::AsMut(VisitStmt(seq));
        ResolveRemainingDynamicShapes(result);
        CheckNoResidualDynamicShape(result);
        return result;
    }

private:
    // Whether we are currently expanding a static loop body.
    bool unrolling_{false};
    // Per-iteration substitution: loop var and loop-carried iterArgs -> their current values.
    std::unordered_map<ir::VarPtr, ir::ExprPtr> subst_;
    // Scalar substitution used for folding shape/offset symbols (loop variables only).
    // A stack keeps every enclosing loop variable alive while unrolling a nested loop, so a
    // symbolic scalar that references both the outer and inner loop indices folds completely.
    std::vector<std::pair<ir::VarPtr, ir::ExprPtr>> loopVarStack_;
    // Input raw symbol -> parameter tensor, precomputed once for O(1) lookup in EvaluateShapeDim.
    // Keyed by the tensor's raw symbol (tensor->symbol), which is the name baked into the
    // ARG_<name> operand of RUNTIME_GetInputShapeDim; unlike ir::Var::name_ it is immune to the
    // GetVarName de-dup suffix, so it always matches the ARG operand exactly. The parameter tensor
    // carries the concrete static shape written back by register_static_shape_tensor.
    std::unordered_map<std::string, const LogicalTensor*> staticShapeTensorBySymbol_;

    void BuildStaticShapeByTensor(const std::vector<ir::VarPtr>& params)
    {
        staticShapeTensorBySymbol_.clear();
        for (const auto& param : params) {
            auto lt = AsLogicalTensor(param);
            if (lt == nullptr) {
                continue;
            }
            std::string name = lt->Symbol();
            if (name.empty()) {
                continue;
            }
            IRCHECK(staticShapeTensorBySymbol_.emplace(name, lt.get()).second)
                << "LitenpuDynamicToStatic: duplicate input symbol '" << name << "'; cannot disambiguate "
                << "RUNTIME_GetInputShapeDim(ARG_" << name << ", dim) by name.";
        }
    }

    SymbolicScalar Substitute(SymbolicScalar scalar) const
    {
        if (loopVarStack_.empty()) {
            return scalar;
        }
        // Only the scalar loop variables participate in symbolic folding; substituting
        // tensor/token iterArgs into a symbolic scalar is meaningless (and unsafe for
        // the raw-symbol cast performed by SubstituteVars). All enclosing loop variables
        // are applied so a nested loop folds both its outer and inner indices.
        std::unordered_map<ir::VarPtr, ir::ExprPtr> varMap;
        varMap.reserve(loopVarStack_.size());
        for (const auto& [var, value] : loopVarStack_) {
            varMap[var] = value;
        }
        scalar = scalar.SubstituteVars(varMap);
        return scalar.Simplify();
    }

    LogicalTensor* AsMutableTensor(const ir::ExprPtr& expr)
    {
        auto lt = std::dynamic_pointer_cast<const LogicalTensor>(expr);
        return lt ? const_cast<LogicalTensor*>(lt.get()) : nullptr;
    }

    // Evaluate a branch condition to a concrete boolean when statically resolvable.
    std::optional<bool> EvaluateCondition(const ExprPtr& condExpr)
    {
        if (!condExpr) {
            return std::nullopt;
        }
        if (auto c = As<ConstBool>(condExpr)) {
            return c->value_;
        }
        if (auto ci = As<ConstInt>(condExpr)) {
            return ci->value_ != 0;
        }
        SymbolicScalar scalar = SymbolicScalar::FromExpr(condExpr);
        if (!scalar.IsValid()) {
            return std::nullopt;
        }
        scalar = StaticizeScalar(scalar);
        if (scalar.ConcreteValid()) {
            return scalar.Concrete() != 0;
        }
        return std::nullopt;
    }

    // Resolve a single RUNTIME_GetInputShapeDim(ARG_<input>, dim) call node to its concrete
    // static shape value. Returns nullopt when the node is not such a call; fails fast (via
    // IRCHECK) when the call cannot be resolved to a function parameter tensor.
    std::optional<int64_t> ResolveShapeDimCall(const RawSymbolicScalarPtr& node) const
    {
        if (node == nullptr || !node->IsExpression()) {
            return std::nullopt;
        }
        auto expr = std::static_pointer_cast<RawSymbolicExpression>(node);
        if (!expr->IsExpressionCall("RUNTIME_GetInputShapeDim")) {
            return std::nullopt;
        }
        const auto& ops = expr->GetExpressionOperandList();
        if (ops.size() < 3 || !ops[1]->IsSymbol() || !ops[2]->IsImmediate()) {
            return std::nullopt;
        }
        std::string name = ops[1]->GetSymbolName();
        const std::string kArgPrefix(npu::tile_fwk::SPECIAL_SYMBOL_NAME_ARG_PREFIX);
        if (name.compare(0, kArgPrefix.size(), kArgPrefix) == 0) {
            name = name.substr(kArgPrefix.size());
        }
        const int64_t dim = std::static_pointer_cast<RawSymbolicImmediate>(ops[2])->Immediate();
        auto it = staticShapeTensorBySymbol_.find(name);
        IRCHECK(it != staticShapeTensorBySymbol_.end())
            << "LitenpuDynamicToStatic: cannot resolve input '" << name << "' in "
            << "RUNTIME_GetInputShapeDim(ARG_" << name << ", dim); no function parameter matches this "
            << "symbol. The -1 dim cannot be staticized.";
        const auto& staticShape = it->second->shape;
        IRCHECK(dim >= 0 && static_cast<size_t>(dim) < staticShape.size())
            << "LitenpuDynamicToStatic: dim " << dim << " out of range for input '" << name << "' (static shape size "
            << staticShape.size() << ").";
        return staticShape[static_cast<size_t>(dim)];
    }

    // Deep-substitute every RUNTIME_GetInputShapeDim call inside a scalar expression with its
    // concrete static shape value, mirroring the recursive walk in SymbolicScalar::Substitute.
    // Returns the folded scalar; the per-node resolution is delegated to ResolveShapeDimCall.
    SymbolicScalar SubstituteShapeDimCalls(const SymbolicScalar& scalar) const
    {
        std::function<RawSymbolicScalarPtr(const RawSymbolicScalarPtr&)> substitute =
            [&](const RawSymbolicScalarPtr& raw) -> RawSymbolicScalarPtr {
            if (raw == nullptr) {
                return raw;
            }
            if (auto concrete = ResolveShapeDimCall(raw); concrete.has_value()) {
                return RawSymbolicImmediate::Create(*concrete);
            }
            if (!raw->IsExpression()) {
                return raw;
            }
            auto expr = std::static_pointer_cast<RawSymbolicExpression>(raw);
            std::vector<RawSymbolicScalarPtr> operands;
            operands.reserve(expr->OperandList().size());
            bool changed = false;
            for (const auto& operand : expr->OperandList()) {
                auto resolved = substitute(operand);
                operands.push_back(resolved);
                changed = changed || resolved != operand;
            }
            if (!changed) {
                return raw;
            }
            return std::make_shared<RawSymbolicExpression>(expr->Opcode(), operands);
        };
        return SymbolicScalar(substitute(scalar.Raw()));
    }

    // Substitute the loop variable, resolve every RUNTIME_GetInputShapeDim call against the
    // registered static shape, then simplify. ConcreteValid() is true when the scalar is fully
    // staticized (used for shape dims, offsets and op dynamic attributes).
    SymbolicScalar StaticizeScalar(SymbolicScalar scalar) const
    {
        scalar = Substitute(scalar);
        scalar = SubstituteShapeDimCalls(scalar);
        return scalar.Simplify();
    }

    // Evaluate an Expr to a concrete integer, resolving any RUNTIME_GetInputShapeDim call against
    // the registered static shape, or nullopt when still not statically known.
    std::optional<int64_t> EvaluateInt(const ExprPtr& expr) const
    {
        if (!expr) {
            return std::nullopt;
        }
        if (auto c = As<ConstInt>(expr)) {
            return c->value_;
        }
        SymbolicScalar scalar = SymbolicScalar::FromExpr(expr);
        if (!scalar.IsValid()) {
            return std::nullopt;
        }
        scalar = SubstituteShapeDimCalls(scalar).Simplify();
        if (scalar.ConcreteValid()) {
            return scalar.Concrete();
        }
        return std::nullopt;
    }

    // Resolve a dynamic shape/offset scalar to a concrete integer when possible. Handles both a
    // bare RUNTIME_GetInputShapeDim call and composite expressions (e.g. min(...), arithmetic).
    std::optional<int64_t> EvaluateShapeDim(const SymbolicScalar& scalar) const
    {
        SymbolicScalar s = StaticizeScalar(scalar);
        if (s.ConcreteValid()) {
            return s.Concrete();
        }
        return std::nullopt;
    }

    // Replace -1 (dynamic) dims in a logical tensor's static shape with their concrete value,
    // derived from the tensor's dynamic valid shape.
    void ResolveShape(LogicalTensor* lt)
    {
        if (lt == nullptr) {
            return;
        }
        for (size_t d = 0; d < lt->shape.size(); ++d) {
            if (lt->shape[d] != -1 || d >= lt->dynValidShape_.size()) {
                continue;
            }
            auto concrete = EvaluateShapeDim(lt->dynValidShape_[d]);
            if (concrete.has_value()) {
                lt->shape[d] = *concrete;
            }
        }
    }

    // Resolve every residual -1 dim left after unrolling. A write-only output param (written via
    // assemble inside the loop) is only referenced by the function's return statement and by the
    // loop's returnVar, neither of which is covered by VisitExpr_(VarPtr) (reads) nor CloneTensor
    // (op results). Walk the final body once and resolve whatever -1 dims remain.
    void ResolveRemainingDynamicShapes(const SeqStmtsPtr& seq)
    {
        class Resolver : public IRVisitor {
        public:
            explicit Resolver(LitenpuDynamicToStaticConverter* converter) : converter_(converter) {}

            void VisitVarLike_(const VarPtr& op) override
            {
                converter_->ResolveShape(converter_->AsMutableTensor(op));
                IRVisitor::VisitVarLike_(op);
            }

        private:
            LitenpuDynamicToStaticConverter* converter_;
        };
        Resolver resolver(this);
        resolver.VisitStmt(seq);
    }

    // Clone a tensor result to a fresh SSA version and substitute its dynamic symbols.
    ir::VarPtr CloneTensor(const ir::VarPtr& res)
    {
        auto lt = std::dynamic_pointer_cast<const LogicalTensor>(res);
        if (lt == nullptr) {
            return res;
        }
        auto cloned = lt->Clone(true);
        auto* clonedTensor = AsMutableTensor(cloned);
        for (auto& dim : clonedTensor->GetDynValidShape()) {
            dim = StaticizeScalar(dim);
        }
        for (auto& off : clonedTensor->dynOffset_) {
            off = StaticizeScalar(off);
        }
        ResolveShape(clonedTensor);
        return cloned;
    }

    StmtPtr VisitStmt_(const ForStmtPtr& op) override
    {
        auto start = EvaluateInt(op->start_);
        auto stop = EvaluateInt(op->stop_);
        auto step = EvaluateInt(op->step_);
        if (!start || !stop || !step || *step == 0) {
            // Kirin STATIC codegen has no dynamic loop support: loop bounds that cannot be
            // folded to compile-time constants (or a zero step) would leave the loop variable
            // dangling, producing NaNs at runtime. Fail fast instead.
            auto dump = [](const ExprPtr& e) {
                SymbolicScalar s = SymbolicScalar::FromExpr(e);
                return s.IsValid() ? s.Dump() : std::string("?");
            };
            PASS_ASSERT(start && stop && step && *step != 0)
                << "LitenpuDynamicToStatic: cannot statically resolve loop bounds (start=" << dump(op->start_)
                << ", stop=" << dump(op->stop_) << ", step=" << dump(op->step_)
                << "). Kirin requires compile-time constant loop start/stop/step.";
        }
        return UnrollFor(op, *start, *stop, *step);
    }

    StmtPtr VisitStmt_(const IfStmtPtr& op) override
    {
        auto takeThen = EvaluateCondition(op->condition_);
        if (!takeThen) {
            // Kirin STATIC codegen has no dynamic branch support: a branch condition that cannot
            // be folded to a constant would leave a dynamic branch behind, which the static codegen
            // cannot lower. Fail fast instead.
            SymbolicScalar cond = SymbolicScalar::FromExpr(op->condition_);
            std::string condStr = cond.IsValid() ? cond.Dump() : "?";
            PASS_ASSERT(takeThen.has_value())
                << "LitenpuDynamicToStatic: cannot statically resolve if condition '" << condStr
                << "'. Kirin requires a compile-time constant branch condition.";
        }

        const SeqStmtsPtr& branch = *takeThen ? op->thenBody_ : *op->elseBody_;
        auto visited = SeqStmts::AsMut(VisitStmt(branch));

        // The branch yields its results; thread them to the if's return vars so
        // later references (e.g. the loop's continue) resolve to the selected branch.
        std::vector<ir::ExprPtr> yieldVals;
        ExtractTerminator(visited, yieldVals);
        for (size_t i = 0; i < op->returnVars_.size() && i < yieldVals.size(); ++i) {
            subst_[op->returnVars_[i]] = yieldVals[i];
        }
        return visited;
    }

    StmtPtr UnrollFor(const ForStmtPtr& op, int64_t start, int64_t stop, int64_t step)
    {
        const size_t n = op->iterArgs_.size();
        std::vector<ir::ExprPtr> current(n);
        for (size_t i = 0; i < n; ++i) {
            current[i] = op->iterArgs_[i]->initValue_;
        }

        bool prevUnrolling = unrolling_;
        unrolling_ = true;

        std::vector<StmtPtr> unrolled;
        for (int64_t idx = start; (step > 0 ? idx < stop : idx > stop); idx += step) {
            // Bind the loop variable and the current iteration args.
            auto loopValue = SymbolicScalar(idx).AsExpr();
            loopVarStack_.emplace_back(op->loopVar_, loopValue);
            subst_[op->loopVar_] = loopValue;
            for (size_t i = 0; i < n; ++i) {
                subst_[op->iterArgs_[i]->iterVar_] = current[i];
            }

            auto body = SeqStmts::AsMut(VisitStmt(op->body_));

            // The trailing continue/yield carries the next iteration values.
            std::vector<ir::ExprPtr> next;
            ExtractTerminator(body, next);

            for (auto& s : body->stmts_) {
                unrolled.push_back(std::move(s));
            }

            if (next.size() == n) {
                current = std::move(next);
            }

            // Reset per-iteration bindings.
            loopVarStack_.pop_back();
            subst_.erase(op->loopVar_);
            for (const auto& ia : op->iterArgs_) {
                subst_.erase(ia->iterVar_);
            }
        }

        unrolling_ = prevUnrolling;

        // Expose the final carried values through the return vars.
        for (size_t i = 0; i < n && i < op->returnVars_.size(); ++i) {
            var_remap_[op->returnVars_[i].get()] = current[i];
        }

        if (unrolled.empty()) {
            return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, op->span_);
        }
        return std::make_shared<SeqStmts>(std::move(unrolled), op->span_);
    }

    static void ExtractTerminator(const SeqStmtsPtr& body, std::vector<ir::ExprPtr>& out)
    {
        out.clear();
        if (body == nullptr || body->stmts_.empty()) {
            return;
        }
        const auto& last = body->stmts_.back();
        if (auto cont = As<ContinueStmt>(last)) {
            out = cont->value_;
            body->stmts_.pop_back();
        } else if (auto yld = As<YieldStmt>(last)) {
            out = yld->value_;
            body->stmts_.pop_back();
        }
    }

    ExprPtr VisitExpr_(const VarPtr& op) override
    {
        // Resolve any -1 (dynamic) dim in the tensor's static shape to its concrete value.
        ResolveShape(AsMutableTensor(op));

        auto it = subst_.find(op);
        if (it != subst_.end()) {
            return it->second;
        }
        // Follow var_remap_ transitively. Nested loops produced by unroll_list map an intermediate
        // return value to another intermediate before reaching the final carried value; a single
        // lookup stops one hop early and leaves the function return bound to a stale tensor.
        ExprPtr resolved = op;
        std::unordered_set<const Expr*> seen;
        seen.insert(op.get());
        while (true) {
            auto vit = var_remap_.find(resolved.get());
            if (vit == var_remap_.end() || !seen.insert(vit->second.get()).second) {
                break;
            }
            resolved = vit->second;
        }
        if (resolved.get() != op.get()) {
            return resolved;
        }
        return IRMutator::VisitExpr_(op);
    }

    StmtPtr VisitStmt_(const TensorOpStmtPtr& op) override
    {
        if (!unrolling_ || subst_.empty()) {
            return IRMutator::VisitStmt_(op);
        }

        auto newArgs = VisitExprList(op->args_).first;

        // Clone result tensors to fresh SSA versions and substitute their dynamic symbols.
        std::vector<ir::VarPtr> newResults;
        newResults.reserve(op->result_.size());
        for (const auto& res : op->result_) {
            auto cloned = CloneTensor(res);
            subst_[res] = cloned;
            newResults.push_back(cloned);
        }

        auto newResultTokens = VisitVarList(op->result_token_).first;
        auto newTokens = VisitVarList(op->tokens_).first;

        auto rebuilt = npu::tile_fwk::RebuildTensorOpStmt(op, std::move(newResults), std::move(newResultTokens),
                                                          std::move(newArgs), std::move(newTokens), op->span_);

        // Substitute the operation-level dynamic attributes (view/assemble offsets & shapes).
        auto rebuiltOp = std::dynamic_pointer_cast<const Operation>(rebuilt);
        if (rebuiltOp != nullptr) {
            auto mutableOp = std::const_pointer_cast<Operation>(rebuiltOp);
            for (auto& attr : mutableOp->GetDynamicAttributeList()) {
                attr.get() = StaticizeScalar(attr.get());
            }
        }
        return rebuilt;
    }
};

} // namespace

SeqStmtsPtr RunLitenpuDynamicToStaticPass(SeqStmtsPtr seq, const std::vector<ir::VarPtr>& params)
{
    LitenpuDynamicToStaticConverter converter;
    return converter.Apply(seq, params);
}

Pass pass::LitenpuDynamicToStatic()
{
    return pass::CreateFunctionPass(
        [](const FunctionPtr& func) -> FunctionPtr {
            if (!func || !func->body_) {
                return func;
            }
            auto body = RunLitenpuDynamicToStaticPass(SeqStmts::AsMut(func->body_), func->params_);
            if (body == func->body_) {
                return func;
            }
            return std::make_shared<const Function>(func->name_, func->params_, func->returnTypes_, body, func->span_,
                                                    func->funcType_, func->entry_, func->attrs_);
        },
        "LitenpuDynamicToStatic");
}

} // namespace pypto::ir
