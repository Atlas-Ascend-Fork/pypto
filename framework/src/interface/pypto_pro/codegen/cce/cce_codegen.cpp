/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen/cce/cce_codegen.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <functional>
#include <ios>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "backend/common/backend.h"
#include "backend/common/backend_utils.h"
#include "core/error.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/pipe.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/scalar_expr_ops.h"
#include "ir/stmt.h"
#include "ir/kind_traits.h"
#include "ir/transforms/base/mutator.h"
#include "ir/transforms/structural_comparison.h"
#include "ir/type.h"
#include "pypto_pro/error.h"
#include "tilefwk/error.h"

namespace pypto {
namespace codegen {
using ir::DataType;
using npu::tile_fwk::ExternalError;

namespace {

std::string TupleItemName(const std::string& tupleName, size_t index)
{
    return tupleName + "__item_" + std::to_string(index);
}

} // namespace

std::string BuildFP32FmodExpression(const std::string& lhs, const std::string& rhs)
{
    // Binary long division of normalized significands avoids quotient overflow
    // and cancellation. Integer shifts also preserve subnormals and signed zero.
    std::ostringstream code;
    code << "({"
         << "float __x = (" << lhs << "); float __y = (" << rhs << ");"
         << "uint32_t __ux = reinterpret_cast<uint32_t&>(__x);"
         << "uint32_t __uy = reinterpret_cast<uint32_t&>(__y);"
         << "uint32_t __sign = __ux & 0x80000000u;"
         << "uint32_t __ax = __ux & 0x7fffffffu, __ay = __uy & 0x7fffffffu;"
         << "uint32_t __result = __ux;"
         << "if (__ay == 0 || __ax >= 0x7f800000u || __ay > 0x7f800000u) {"
         << "__result = 0x7fc00000u;"
         << "} else if (__ax >= __ay) {"
         << "int __ex = (int)(__ax >> 23), __ey = (int)(__ay >> 23);"
         << "uint32_t __mx = __ax & 0x007fffffu, __my = __ay & 0x007fffffu;"
         << "if (__ex == 0) {"
         << "__ex = 1; while (__mx < 0x00800000u) { __mx <<= 1; --__ex; }"
         << "} else { __mx |= 0x00800000u; }"
         << "if (__ey == 0) {"
         << "__ey = 1; while (__my < 0x00800000u) { __my <<= 1; --__ey; }"
         << "} else { __my |= 0x00800000u; }"
         << "while (__ex > __ey) {"
         << "if (__mx >= __my) { __mx -= __my; }"
         << "__mx <<= 1; --__ex;"
         << "}"
         << "if (__mx >= __my) { __mx -= __my; }"
         << "if (__mx == 0) { __result = __sign; } else {"
         << "while (__mx < 0x00800000u) { __mx <<= 1; --__ex; }"
         << "if (__ex > 0) { __result = (__mx & 0x007fffffu) | ((uint32_t)__ex << 23); }"
         << "else { __result = __mx >> (1 - __ex); }"
         << "__result |= __sign;"
         << "}"
         << "}"
         << "reinterpret_cast<float&>(__result);"
         << "})";
    return code.str();
}

CCECodegen::CCECodegen(ir::SectionKind target) : backend_(backend::GetBackend()), target_(target)
{
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT,
                      target_ == ir::SectionKind::Cube || target_ == ir::SectionKind::Vector)
        << "CCE target must be Cube or Vector";
}

// ============================================================================

namespace {

/// Renames all variable definitions in a function body by adding a prefix,
/// and substitutes parameter references with call-site argument expressions.
class BodyRenamer : public ir::IRMutator {
public:
    using ir::IRMutator::VisitExpr_;
    using ir::IRMutator::VisitStmt_;
    BodyRenamer(const std::string& prefix, const std::vector<ir::VarPtr>& params, const std::vector<ir::ExprPtr>& args,
                const std::unordered_map<const ir::Expr*, ir::ExprPtr>& parentRemap = {})
        : prefix_(prefix)
    {
        // Seed with the enclosing inline's remap so free-variable references to an outer
        // helper's vars (e.g. an auto_mutex buf_id captured from the caller and carried into a
        // nested helper) are rewritten consistently. Params override any inherited entry.
        var_remap_ = parentRemap;
        // The Python->IR conversion gives each function its own Var objects, so a nested helper's
        // free reference to an outer helper's var is a *distinct* pointer with the same name. Build
        // a name-keyed fallback from the enclosing remap to bridge those (see VisitExpr_(Var)).
        for (const auto& [key, val] : parentRemap) {
            if (key != nullptr && key->GetKind() == ir::ObjectKind::Var) {
                nameRemap_[static_cast<const ir::Var*>(key)->name_] = val;
            }
        }
        for (size_t i = 0; i < params.size() && i < args.size(); ++i) {
            var_remap_[params[i].get()] = args[i];
        }
    }

    /// Final var mapping after visiting; used to seed nested helper inlines.
    const std::unordered_map<const ir::Expr*, ir::ExprPtr>& GetRemap() const { return var_remap_; }

private:
    std::string prefix_;
    // Name-keyed fallback for free variables inherited from an enclosing inline (see ctor).
    std::unordered_map<std::string, ir::ExprPtr> nameRemap_;

    ir::VarPtr RenameVar(const ir::VarPtr& var)
    {
        auto it = var_remap_.find(var.get());
        if (it != var_remap_.end()) {
            if (auto v = ir::As<ir::Var>(it->second))
                return v;
            return var;
        }
        auto renamed = std::make_shared<const ir::Var>(prefix_ + var->name_, var->GetType(), var->span_);
        var_remap_[var.get()] = renamed;
        return renamed;
    }

    ir::ExprPtr VisitExpr_(const ir::VarPtr& op) override
    {
        auto it = var_remap_.find(op.get());
        if (it != var_remap_.end())
            return it->second;
        // Pointer miss on a var whose definition is not in this body means it is a free variable
        // inherited from an enclosing inline; bridge it by name (its defining object lives in the
        // enclosing helper and was already renamed there). Helper-local uses always hit above,
        // since their defining statement is visited (and pointer-mapped) before any use.
        auto nameIt = nameRemap_.find(op->name_);
        if (nameIt != nameRemap_.end())
            return nameIt->second;
        return op;
    }

    ir::StmtPtr VisitStmt_(const ir::AssignStmtPtr& op) override
    {
        auto newValue = VisitExpr(op->value_);
        std::vector<ir::VarPtr> targets;
        bool changed = newValue.get() != op->value_.get();
        for (const auto& target : op->targets_) {
            auto renamed = RenameVar(target);
            changed |= renamed.get() != target.get();
            targets.push_back(std::move(renamed));
        }
        if (changed) {
            return std::make_shared<const ir::AssignStmt>(std::move(targets), std::move(newValue), op->span_);
        }
        return op;
    }

    std::vector<ir::IterArgPtr> RenameIterArgs(const std::vector<ir::IterArgPtr>& iterArgs)
    {
        std::vector<ir::IterArgPtr> newIterArgs;
        newIterArgs.reserve(iterArgs.size());
        for (const auto& ia : iterArgs) {
            auto newInit = VisitExpr(ia->initValue_);
            // Iter args may shadow helper params, so keep them isolated from call-site remaps.
            auto renamedVar = std::make_shared<const ir::Var>(prefix_ + ia->iterVar_->name_, ia->iterVar_->GetType(),
                                                              ia->iterVar_->span_);
            auto renamed = std::make_shared<const ir::IterArg>(renamedVar, newInit);
            var_remap_[ia->iterVar_.get()] = renamedVar;
            newIterArgs.push_back(renamed);
        }
        return newIterArgs;
    }

    ir::StmtPtr VisitStmt_(const ir::ForStmtPtr& op) override
    {
        auto newLoopVar = RenameVar(op->loopVar_);
        var_remap_[op->loopVar_.get()] = newLoopVar;
        auto newStart = VisitExpr(op->start_);
        auto newStop = VisitExpr(op->stop_);
        auto newStep = VisitExpr(op->step_);

        auto newIterArgs = RenameIterArgs(op->iterArgs_);
        auto newBody = VisitStmt(op->body_);

        std::vector<ir::VarPtr> newReturnVars;
        newReturnVars.reserve(op->returnVars_.size());
        for (const auto& rv : op->returnVars_) {
            newReturnVars.push_back(RenameVar(rv));
        }

        return std::make_shared<const ir::ForStmt>(newLoopVar, newStart, newStop, newStep, std::move(newIterArgs),
                                                   newBody, std::move(newReturnVars), op->span_);
    }

    ir::StmtPtr VisitStmt_(const ir::WhileStmtPtr& op) override
    {
        auto newIterArgs = RenameIterArgs(op->iterArgs_);
        auto newCond = VisitExpr(op->condition_);
        auto newBody = VisitStmt(op->body_);

        std::vector<ir::VarPtr> newReturnVars;
        newReturnVars.reserve(op->returnVars_.size());
        for (const auto& rv : op->returnVars_) {
            newReturnVars.push_back(RenameVar(rv));
        }

        return std::make_shared<const ir::WhileStmt>(newCond, std::move(newIterArgs), newBody, std::move(newReturnVars),
                                                     op->span_);
    }

    ir::StmtPtr VisitStmt_(const ir::IfStmtPtr& op) override
    {
        auto newCond = VisitExpr(op->condition_);
        auto newThen = VisitStmt(op->thenBody_);
        std::optional<ir::StmtPtr> newElse;
        if (op->elseBody_.has_value()) {
            newElse = VisitStmt(*op->elseBody_);
        }
        std::vector<ir::VarPtr> newRvs;
        newRvs.reserve(op->returnVars_.size());
        for (const auto& rv : op->returnVars_) {
            newRvs.push_back(RenameVar(rv));
        }
        return std::make_shared<const ir::IfStmt>(newCond, newThen, newElse, std::move(newRvs), op->span_);
    }

    bool RenameJumpValues(const std::vector<ir::ExprPtr>& values, std::vector<ir::ExprPtr>& newValues)
    {
        newValues.reserve(values.size());
        bool changed = false;
        for (const auto& value : values) {
            auto newValue = VisitExpr(value);
            changed = changed || (newValue.get() != value.get());
            newValues.push_back(std::move(newValue));
        }
        return changed;
    }

    // The parser snapshots each loop-carried value's live SSA version at a
    // break/continue. Rename those operands together with the inlined body so
    // their references cannot retain pre-inline names.
    ir::StmtPtr VisitStmt_(const ir::BreakStmtPtr& op) override
    {
        std::vector<ir::ExprPtr> newValues;
        if (!RenameJumpValues(op->value_, newValues))
            return op;
        return std::make_shared<const ir::BreakStmt>(std::move(newValues), op->span_);
    }

    ir::StmtPtr VisitStmt_(const ir::ContinueStmtPtr& op) override
    {
        std::vector<ir::ExprPtr> newValues;
        if (!RenameJumpValues(op->value_, newValues))
            return op;
        return std::make_shared<const ir::ContinueStmt>(std::move(newValues), op->span_);
    }
};

/// Replaces all ReturnStmt nodes in a body with an assignment to `target`.
///
/// The call site is always a single var (tuple unpacking is lowered to
/// `_tuple_tmp = helper(); a = _tuple_tmp[0]` by the parser), and a tuple
/// return is a single MakeTuple expression. So a return lowers to exactly
/// `target = value`; tuple resolution is left to backend codegen.
class ReturnReplacer : public ir::IRMutator {
public:
    using ir::IRMutator::VisitStmt_;
    explicit ReturnReplacer(ir::VarPtr target) : target_(std::move(target)) {}

    ir::StmtPtr VisitStmt_(const ir::ReturnStmtPtr& op) override
    {
        if (target_ && op->value_.empty()) {
            // The call result is used as a value, but this path is a void `return` (a
            // `return None` in the helper). Assign an empty-tuple sentinel: if this path is
            // dead it is dropped by const-folding; if it is the taken path the sentinel
            // survives and codegen reports a clear error at the use site (see
            // VisitExpr_(GetItemExprPtr)) instead of emitting an undeclared identifier that
            // only fails later inside the CCE compiler.
            return std::make_shared<const ir::AssignStmt>(
                std::vector<ir::VarPtr>{target_},
                std::make_shared<const ir::MakeTuple>(std::vector<ir::ExprPtr>{}, op->span_), op->span_);
        }
        if (!target_ || op->value_.empty()) {
            // Void call — replace return with empty SeqStmts
            return std::make_shared<const ir::SeqStmts>(std::vector<ir::StmtPtr>{}, op->span_);
        }
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->value_.size() == 1)
            << "helper return must carry a single value (tuple returns are a single MakeTuple)";
        return std::make_shared<const ir::AssignStmt>(std::vector<ir::VarPtr>{target_}, op->value_[0], op->span_);
    }

private:
    ir::VarPtr target_;
};

/// Build the inlined body for a helper call. Replaces ReturnStmt with an
/// assignment to `target` (null for result-discarding calls).
ir::StmtPtr BuildInlinedBody(const ir::FunctionPtr& helper, const std::vector<ir::ExprPtr>& callArgs,
                             const std::string& prefix, const ir::VarPtr& target,
                             const std::unordered_map<const ir::Expr*, ir::ExprPtr>& parentRemap = {},
                             std::unordered_map<const ir::Expr*, ir::ExprPtr>* outRemap = nullptr)
{
    BodyRenamer renamer(prefix, helper->params_, callArgs, parentRemap);
    auto renamedBody = renamer.VisitStmt(helper->body_);
    if (outRemap != nullptr) {
        *outRemap = renamer.GetRemap();
    }

    ReturnReplacer replacer(target);
    return replacer.VisitStmt(renamedBody);
}

/// Check if a Call targets a helper function by name.
bool IsHelperCall(const ir::CallPtr& call, const std::map<std::string, ir::FunctionPtr>& helpers)
{
    if (!call || call->name_.empty())
        return false;
    return helpers.count(call->name_) > 0;
}

/// Walks the kernel body, replacing helper calls with inlined bodies.
///
/// Tuple handling is intentionally left to backend codegen: a tuple-returning
/// helper inlines to `target = MakeTuple(...)` (via ReturnReplacer), and a
/// later `target[i]` is resolved by CCE/PTO codegen's tupleVarToMakeTuple_.
class CallInliner : public ir::IRMutator {
public:
    using ir::IRMutator::VisitStmt_;
    explicit CallInliner(const std::map<std::string, ir::FunctionPtr>& helpers) : helpers_(helpers) {}

    ir::StmtPtr VisitStmt_(const ir::AssignStmtPtr& op) override
    {
        auto call = ir::As<ir::Call>(op->value_);
        if (!IsHelperCall(call, helpers_)) {
            return IRMutator::VisitStmt_(op);
        }
        PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, op->targets_.size() <= 1)
            << "CCE codegen does not support AssignStmt with multiple targets";
        auto targetVar = op->targets_.empty() ? nullptr : op->targets_.front();
        std::string prefix = "_inl" + std::to_string(counter_++) + "_";
        const auto& helper = helpers_.at(call->name_);

        std::vector<ir::ExprPtr> args;
        args.reserve(call->args_.size());
        for (const auto& a : call->args_) {
            args.push_back(VisitExpr(a));
        }

        // Single target: tuple returns lower to `target = MakeTuple(...)`.
        std::unordered_map<const ir::Expr*, ir::ExprPtr> childRemap;
        auto inlined = BuildInlinedBody(helper, args, prefix, targetVar, ambientRemap_, &childRemap);
        auto result = VisitInlinedWithAmbient(inlined, std::move(childRemap)); // recurse for nested helper calls
        return UnwrapIfNestedVF(result);
    }

    /// Track whether we are inlining inside an enclosing VF section. A VF helper
    /// (@pl.vector_function) is compiled with its body wrapped in a SectionStmt(VF);
    /// when such a helper is called from another VF helper, naive inlining would
    /// nest two VF sections, which the codegen VF-section invariant forbids
    /// (see EmitSection: "VF section must be nested inside a Vector section").
    /// Flatten by tracking the current section kind here and unwrapping a nested
    /// VF section at the inline site (see UnwrapIfNestedVF).
    ir::StmtPtr VisitStmt_(const ir::SectionStmtPtr& op) override
    {
        bool prevInVf = inVfSection_;
        if (op->sectionKind_ == ir::SectionKind::VF) {
            inVfSection_ = true;
        }
        auto result = IRMutator::VisitStmt_(op);
        inVfSection_ = prevInVf;
        return result;
    }

private:
    /// If we are already inside a VF section and the freshly-inlined statement is
    /// itself a SectionStmt(VF), splice in its body instead of the nested section,
    /// collapsing multiple VF-helper calls into a single VF section.
    ir::StmtPtr UnwrapIfNestedVF(const ir::StmtPtr& inlined)
    {
        if (!inVfSection_) {
            return inlined;
        }
        auto section = ir::As<ir::SectionStmt>(inlined);
        if (section && section->sectionKind_ == ir::SectionKind::VF) {
            return section->body_;
        }
        return inlined;
    }

    /// Recurse into a freshly-inlined body with `child_remap` (this inline's var mapping) as the
    /// ambient remap, so a nested helper's free references to this helper's vars are rewritten
    /// consistently. Restores the previous ambient afterward.
    ir::StmtPtr VisitInlinedWithAmbient(const ir::StmtPtr& inlined,
                                        std::unordered_map<const ir::Expr*, ir::ExprPtr>&& childRemap)
    {
        auto saved = std::move(ambientRemap_);
        ambientRemap_ = std::move(childRemap);
        auto result = VisitStmt(inlined);
        ambientRemap_ = std::move(saved);
        return result;
    }

    const std::map<std::string, ir::FunctionPtr>& helpers_;
    int counter_ = 0;
    // Var mapping from the enclosing inline(s); seeds nested helper inlines so their free-variable
    // references (e.g. auto_mutex buf_ids captured from an outer scope) remap correctly.
    std::unordered_map<const ir::Expr*, ir::ExprPtr> ambientRemap_;
    // True while visiting inside a VF section, so a VF helper inlined into another
    // VF helper has its redundant nested VF section unwrapped (see UnwrapIfNestedVF).
    bool inVfSection_ = false;
};

} // namespace

// ============================================================================
// Single-program CCE generation (skip ptoas)
// ============================================================================

void CCECodegen::ResetFunctionGenerationState()
{
    context_.Clear();
    tensorToPointer_.clear();
    tensorDefs_.clear();
    flatTensorDeclCount_ = 0;

    currentTargetVar_.clear();
    currentExprValue_.clear();
    currentTuple_.reset();

    tupleVarToMakeTuple_.clear();
    tupleBackingArr_.clear();
    loopTargetStack_.clear();
    yieldBuffer_.clear();

    tileOffsetCounter_ = 0;
    currentFunctionType_ = ir::FunctionType::OPAQUE;
}

std::string CCECodegen::BuildVfFunctionSignature(const ir::FunctionPtr& func)
{
    const bool isSimt = IsSimtFuncType(func->funcType_);
    std::ostringstream sig;
    if (func->funcType_ == ir::FunctionType::SIMT_VF) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_VAL, func->HasAttr(ir::kMaxThreadsAttr))
            << "SIMT function '" << func->name_ << "' has no max_threads";
        sig << "__simt_vf__ __launch_bounds__(" << func->GetAttr<int>(ir::kMaxThreadsAttr) << ") inline void ";
    } else if (func->funcType_ == ir::FunctionType::SIMT_KERNEL) {
        sig << "__aicore__ inline void ";
    } else if (func->funcType_ == ir::FunctionType::SIMT_CALLEE) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, func->returnTypes_.size() <= 1)
            << "SIMT callee '" << func->name_ << "' has multiple return values";
        const char* qualifier = GetCurrentKernelType() == ir::FunctionType::SIMT_KERNEL ? "__aicore__" :
                                                                                          "__simt_callee__";
        if (func->returnTypes_.empty()) {
            sig << qualifier << " inline void ";
        } else {
            auto returnType = ir::As<ir::ScalarType>(func->returnTypes_[0]);
            PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, returnType != nullptr)
                << "SIMT callee '" << func->name_ << "' must return a scalar or void";
            sig << qualifier << " inline " << returnType->dtype_.ToCTypeString() << " ";
        }
    } else if (func->funcType_ == ir::FunctionType::SIMD_VF) {
        sig << "__simd_vf__ inline void ";
    } else {
        sig << "__simd_callee__ inline void ";
    }
    sig << func->name_ << (func->funcType_ == ir::FunctionType::SIMT_KERNEL ? "_impl" : "") << "(";
    bool first = true;
    for (const auto& param : func->params_) {
        if (!first) {
            sig << ", ";
        }
        first = false;
        std::string paramName = context_.SanitizeName(param);
        if (auto scalarType = ir::As<ir::ScalarType>(param->GetType())) {
            sig << scalarType->dtype_.ToCTypeString() << " " << paramName;
        } else if (auto ptrType = ir::As<ir::PtrType>(param->GetType())) {
            sig << GetGeneratedType(ptrType) << " " << paramName;
        } else if (auto tensorType = ir::As<ir::TensorType>(param->GetType())) {
            if (GetCurrentKernelType() == ir::FunctionType::SIMT_KERNEL) {
                sig << tensorType->dtype_.ToCTypeString() << "* " << paramName;
            } else {
                PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, isSimt)
                    << "SIMD VF function '" << func->name_ << "' supports only TileType and ScalarType parameters";
                sig << "__gm__ " << tensorType->dtype_.ToCTypeString() << "* " << paramName;
            }
            RegisterPointer(paramName, paramName);
        } else if (auto tileType = ir::As<ir::TileType>(param->GetType())) {
            sig << "__ubuf__ " << tileType->dtype_.ToCTypeString() << "* " << paramName;
            for (int axis = 0; axis < 2; ++axis) {
                if (!GetStaticTileValidDim(tileType, axis)) {
                    sig << ", uint32_t " << paramName << (axis == 0 ? "__valid_row" : "__valid_col");
                }
            }
        } else if (auto regType = ir::As<ir::RegType>(param->GetType())) {
            // SIMD VF/callee register parameters are passed by reference so the
            // callee can use the caller's register storage directly.
            PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, !isSimt)
                << "SIMT function '" << func->name_ << "' does not support register parameters";
            sig << GetGeneratedType(param->GetType()) << "& " << paramName;
        } else {
            PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, false)
                << (isSimt ? "SIMT function '" : "SIMD VF function '") << func->name_
                << (isSimt ? "' supports only TensorType, TileType, PtrType, and ScalarType parameters" :
                             "' supports only TensorType, TileType, ScalarType, PtrType, and RegType parameters");
        }
        context_.RegisterVar(param, paramName);
    }
    sig << ")";
    return sig.str();
}

void CCECodegen::GenerateVfFunction(const ir::FunctionPtr& func)
{
    PRO_CODEGEN_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR,
                      func != nullptr && (IsSimtFuncType(func->funcType_) || IsSimdFuncType(func->funcType_)))
        << "GenerateVfFunction expects a SIMT_VF/SIMT_CALLEE/SIMT_KERNEL/SIMD_VF/SIMD_CALLEE function";

    ResetFunctionGenerationState();
    currentFunctionType_ = func->funcType_;

    try {
        emitter_.EmitLine(BuildVfFunctionSignature(func));
        emitter_.EmitLine("{");
        emitter_.IncreaseIndent();
        GenerateBody(func);
    } catch (...) {
        currentFunctionType_ = ir::FunctionType::OPAQUE;
        throw;
    }
    currentFunctionType_ = ir::FunctionType::OPAQUE;
    emitter_.EmitLine("");
}

namespace {

class VfCalleeCallCollector : public ir::IRVisitor {
public:
    using ir::IRVisitor::VisitExpr_;

    std::vector<std::string> calls_;

    void VisitExpr_(const ir::CallPtr& op) override
    {
        calls_.push_back(op->name_);
        ir::IRVisitor::VisitExpr_(op);
    }
};

bool HasWindowParam(const ir::FunctionPtr& func)
{
    return std::any_of(func->params_.begin(), func->params_.end(),
                       [](const ir::VarPtr& param) { return ir::As<ir::WindowType>(param->GetType()) != nullptr; });
}

std::vector<ir::FunctionPtr> OrderVfCallees(const std::map<std::string, ir::FunctionPtr>& callees)
{
    std::unordered_set<std::string> active;
    std::unordered_set<std::string> orderedNames;
    std::vector<ir::FunctionPtr> ordered;

    std::function<void(const ir::FunctionPtr&)> visit = [&](const ir::FunctionPtr& func) {
        if (orderedNames.find(func->name_) != orderedNames.end()) {
            return;
        }
        PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, active.insert(func->name_).second)
            << "Recursive callee call involving '" << func->name_ << "' is not supported";
        VfCalleeCallCollector collector;
        collector.VisitStmt(func->body_);
        for (const auto& name : collector.calls_) {
            auto it = callees.find(name);
            if (it != callees.end()) {
                visit(it->second);
            }
        }
        active.erase(func->name_);
        orderedNames.insert(func->name_);
        ordered.push_back(func);
    };

    for (const auto& entry : callees) {
        visit(entry.second);
    }
    return ordered;
}

} // namespace

std::string CCECodegen::GenerateSingle(const ir::ProgramPtr& inputProgram, npu::tile_fwk::NPUArch arch)
{
    PRO_CODEGEN_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, inputProgram != nullptr)
        << "Cannot generate code for null program";
    auto program = LowerScalarExpressionsForCCE()(inputProgram);

    ResetFunctionGenerationState();
    needsSimtAtomicHeader_ = false;
    needsScalarHelpers_ = false;
    currentKernelType_ = ir::FunctionType::OPAQUE;
    arch_ = arch;
    PRO_CODEGEN_CHECK(ExternalError::INVALID_VAL, arch_ != npu::tile_fwk::NPUArch::DAV_UNKNOWN)
        << "unknown arch '" << npu::tile_fwk::NPUArchToString(arch_) << "'";
    PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, IsRegisterBasedArch())
        << "PyPTO Pro only supports arch '3510' or '9201', got '" << npu::tile_fwk::NPUArchToString(arch_) << "'";

    // Parser-produced Programs always carry the tuple metadata table created at
    // the top-level parse entry. Codegen relies on absence *within* that table to
    // distinguish a plain tuple from named tuples and structs.
    debugInfo_ = program->GetDebugInfo();
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, debugInfo_ != nullptr)
        << "CCE codegen requires Program IRDebugInfo";

    ir::FunctionPtr kernelFunc;
    std::vector<ir::FunctionPtr> simtFuncs;
    std::vector<ir::FunctionPtr> simdVfFuncs;
    simtCallees_.clear();
    simdCallees_.clear();
    for (const auto& funcEntry : program->functions_) {
        const auto& func = funcEntry.second;
        if (func->funcType_ == ir::FunctionType::SIMT_VF) {
            simtFuncs.push_back(func);
            continue;
        }
        if (func->funcType_ == ir::FunctionType::SIMT_CALLEE) {
            simtCallees_[func->name_] = func;
            continue;
        }
        if (func->funcType_ == ir::FunctionType::SIMD_VF) {
            simdVfFuncs.push_back(func);
            continue;
        }
        if (func->funcType_ == ir::FunctionType::SIMD_CALLEE) {
            simdCallees_[func->name_] = func;
            continue;
        }
        if (func->funcType_ == ir::FunctionType::ORCHESTRATION) {
            continue;
        }
        if (!kernelFunc) {
            kernelFunc = func;
        }
    }
    PRO_CODEGEN_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, kernelFunc != nullptr)
        << "No kernel function found in program";
    currentKernelType_ = kernelFunc->funcType_;
    if (currentKernelType_ == ir::FunctionType::SIMT_KERNEL || !simtFuncs.empty() || !simtCallees_.empty()) {
        PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, target_ == ir::SectionKind::Vector)
            << "SIMT functions can only be generated for the Vector target";
        PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, arch_ == npu::tile_fwk::NPUArch::DAV_3510)
            << "SIMT direct CCE generation currently requires arch='3510'";
    }
    if (!simdVfFuncs.empty() || !simdCallees_.empty()) {
        PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, target_ == ir::SectionKind::Vector)
            << "SIMD VF functions can only be generated for the Vector target";
    }

    emitter_.Clear();
    tilingHeaders_.clear();
    structDefinitions_.clear();

    // SIMT callee/vector functions are emitted once, before their caller: either
    // the SIMT kernel itself or a non-SIMT kernel that launches them via
    // simt.launch.
    for (const auto& simtCallee : OrderVfCallees(simtCallees_)) {
        GenerateVfFunction(simtCallee);
    }
    for (const auto& simtFunc : simtFuncs) {
        GenerateVfFunction(simtFunc);
    }

    if (kernelFunc->funcType_ == ir::FunctionType::SIMT_KERNEL) {
        GenerateVfFunction(kernelFunc);
        if (needsSimtAtomicHeader_) {
            return "#if defined(__DAV_VEC__)\n"
                   "#include \"__clang_cce_simt_atomic.h\"\n"
                   "#endif\n" +
                   emitter_.GetCode();
        }
        return emitter_.GetCode();
    }

    std::string simtFuncCode = emitter_.GetCode();
    emitter_.Clear();

    // SIMD VF functions are standalone __simd_vf__/__simd_callee__ definitions
    // emitted before the kernel that calls them. They are collected separately
    // because only they depend on the framework VF support header.
    for (const auto& simdCallee : OrderVfCallees(simdCallees_)) {
        GenerateVfFunction(simdCallee);
    }
    for (const auto& simdVfFunc : simdVfFuncs) {
        GenerateVfFunction(simdVfFunc);
    }
    std::string simdFuncCode = emitter_.GetCode();
    emitter_.Clear();

    ResetFunctionGenerationState();
    RegisterTilingStructTypes(kernelFunc);
    GenerateSinglePrologue(kernelFunc);
    GenerateBody(kernelFunc);

    std::string functionCode = emitter_.GetCode();
    emitter_.Clear();

    if (target_ == ir::SectionKind::Cube) {
        emitter_.EmitLine("#if defined(__DAV_CUBE__)");
    } else {
        emitter_.EmitLine("#if defined(__DAV_VEC__)");
    }
    emitter_.EmitLine("#ifndef PYPTO_SYNCED_TILE_DEFINED");
    emitter_.EmitLine("#define PYPTO_SYNCED_TILE_DEFINED");
    emitter_.EmitLine("#include <stdint.h>");
    emitter_.EmitLine("template<class Tile, uint32_t N> struct SyncedTile {");
    emitter_.EmitLine("    using PtoTile = Tile;");
    emitter_.EmitLine("    PtoTile tile;");
    emitter_.EmitLine("    int64_t mutex_ids[N];");
    emitter_.EmitLine("};");
    emitter_.EmitLine("#endif");
    EmitStructTypes();
    emitter_.EmitLine("#include <pypto_dump.h>");
    if (HasWindowParam(kernelFunc)) {
        emitter_.EmitLine("#include <pypto_distribute.h>");
    }
    if (needsScalarHelpers_) {
        emitter_.EmitLine("#include <pypto_scalar.h>");
    }
    // RtWrap and the _vag_bXX_N wrappers live in the framework support header;
    // only the SIMD VF/callee functions use them, so it is pulled in exactly
    // when such functions exist.
    if (target_ == ir::SectionKind::Vector && !simdFuncCode.empty()) {
        emitter_.EmitLine("#include <pypto_vf_support.h>");
    }
    // The vector-function definitions precede the impl function: its body calls
    // them, so the definitions must come first in the translation unit.
    if (!simtFuncCode.empty()) {
        emitter_.EmitRaw(simtFuncCode);
    }
    if (!simdFuncCode.empty()) {
        emitter_.EmitRaw(simdFuncCode);
    }
    emitter_.EmitRaw(functionCode);
    emitter_.EmitLine("#endif");

    return emitter_.GetCode();
}

namespace {
std::vector<ir::VarPtr> CollectDynamicDimVars(const ir::FunctionPtr& func)
{
    std::vector<ir::VarPtr> dynDimVars;
    std::set<std::string> seenDynNames;
    for (const auto& param : func->params_) {
        auto tensorType = std::dynamic_pointer_cast<const ir::TensorType>(param->GetType());
        if (!tensorType) {
            continue;
        }
        for (const auto& dimExpr : tensorType->shape_) {
            auto dimVar = std::dynamic_pointer_cast<const ir::Var>(dimExpr);
            if (dimVar && seenDynNames.insert(dimVar->name_).second) {
                dynDimVars.push_back(dimVar);
            }
        }
    }
    return dynDimVars;
}

} // namespace

// ========================================================================
// Phase 6 helpers: GenerateSinglePrologue sub-functions
// ========================================================================

void CCECodegen::EmitSingleFunctionSignature(const ir::FunctionPtr& func)
{
    // Collect dynamic dim variables from tensor shapes (first-occurrence order)
    std::vector<ir::VarPtr> dynDimVars = CollectDynamicDimVars(func);

    // Emit the kernel body as a device function `<name>_impl`; the `__global__` entry that
    // forwards to it is generated by the launcher (JIT call_kernel.cpp) or the binary-delivery
    // op cpp. Keeps the body identical while letting the entry layer add adaptation logic later.
    std::ostringstream sig;
    sig << "__aicore__ inline void " << func->name_ << "_impl";
    if (target_ == ir::SectionKind::Cube) {
        sig << "_cube";
    } else {
        sig << "_vector";
    }
    sig << "(";
    bool first = true;
    for (const auto& param : func->params_) {
        if (!first)
            sig << ", ";
        first = false;
        // Each parameter is registered here while building the signature (the single place
        // that walks all params): the C++ var name, plus the raw pointer for tensor params.
        if (auto tensorType = std::dynamic_pointer_cast<const ir::TensorType>(param->GetType())) {
            std::string paramName = context_.SanitizeName(param);
            std::string ptrName = paramName + "_ptr";
            // The view and its source pointer are both optional (the IR text grammar allows
            // `tensor_view<>`, and hand-built programs need not attach one). Registering the ptr
            // Var only gives body references to it the same C++ name; the signature always takes
            // `<name>_ptr`, so a view-less tensor param is emitted the same way.
            if (tensorType->tensor_view_.has_value() && tensorType->tensor_view_->ptr.has_value()) {
                auto ptrVar = std::const_pointer_cast<ir::Var>(ir::As<ir::Var>(*tensorType->tensor_view_->ptr));
                PRO_CODEGEN_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, ptrVar != nullptr)
                    << "Tensor parameter '" << param->name_ << "' has a tensor_view ptr that is not a Var";
                context_.RegisterVar(ptrVar, ptrName);
            }
            sig << GetGeneratedType(tensorType) << " " << ptrName;
            context_.RegisterVar(param, paramName);
            RegisterPointer(paramName, ptrName);
        } else if (auto scalarType = std::dynamic_pointer_cast<const ir::ScalarType>(param->GetType())) {
            std::string paramName = context_.SanitizeName(param);
            sig << GetGeneratedType(scalarType) << " " << paramName;
            context_.RegisterVar(param, paramName);
        } else if (auto ptrType = std::dynamic_pointer_cast<const ir::PtrType>(param->GetType())) {
            std::string paramName = context_.SanitizeName(param);
            sig << GetGeneratedType(ptrType) << " " << paramName;
            context_.RegisterVar(param, paramName);
        } else if (std::dynamic_pointer_cast<const ir::TupleType>(param->GetType())) {
            // Tiling struct param: pass a raw GM byte pointer `<name>_ptr`; the local
            // struct `<name>` (filled by EmitTilingStructCopy) takes the bare name,
            // so `GetItemExpr` reads emit `<name>.field`.
            std::string paramName = context_.SanitizeName(param);
            sig << "__gm__ uint8_t* " << paramName << "_ptr";
            context_.RegisterVar(param, paramName);
        } else if (std::dynamic_pointer_cast<const ir::WindowType>(param->GetType())) {
            // Window param: raw GM byte pointer to the device pypto::dist::Window
            // descriptor the host wrote at create_window. Field reads (dist.window_get)
            // cast it to the embedded pypto::dist::Window copy, so only the pointer name is
            // addressable in C++.
            std::string paramName = context_.SanitizeName(param);
            std::string ptrName = paramName + "_ptr";
            sig << "__gm__ uint8_t* " << ptrName;
            context_.RegisterVar(param, paramName);
            RegisterPointer(paramName, ptrName);
        }
    }
    // Append dynamic dim variables as int64_t parameters
    for (const auto& dynVar : dynDimVars) {
        if (!first)
            sig << ", ";
        first = false;
        sig << "int64_t " << context_.SanitizeName(dynVar);
    }
    sig << ")";

    emitter_.EmitLine(sig.str());
    emitter_.EmitLine("{");
    emitter_.IncreaseIndent();

    // Register dynamic dim parameters directly (no _local_ copy needed)
    for (const auto& dynVar : dynDimVars) {
        std::string paramName = context_.SanitizeName(dynVar);
        context_.RegisterVar(dynVar, paramName);
    }

    emitter_.EmitLine("");
}

void CCECodegen::EmitSingleTensorDeclarations(const ir::FunctionPtr& func)
{
    // Prescan all accessed tensor vars into tensorDefs_ (params + ptr.make_tensor views).
    // Parameter pointers are already registered while building the signature (see
    // EmitSingleFunctionSignature). Here we emit GlobalTensor declarations for the tensor
    // parameters only; ptr.make_tensor views are emitted in place at their op (see
    // MakeBlockMakeTensorCodegenCCE), where the source pointer's C++ code is already in scope.
    CollectTensorDefs(func);

    for (const auto& param : func->params_) {
        for (const TensorDef* def : GetTensorDefs(context_.SanitizeName(param))) {
            GenerateGlobalTensorTypeDeclaration(*def);
            emitter_.EmitLine("");
        }
    }
}

void CCECodegen::GenerateSinglePrologue(const ir::FunctionPtr& func)
{
    EmitSingleFunctionSignature(func);
    // Tiling struct variable definition + GM copy must come first: the very next decls
    // (tensors/tiles with dynamic shapes) and the body read `tiling.field`.
    EmitTilingStructCopy(func);
    EmitSingleTensorDeclarations(func);
}

void CCECodegen::VisitStmt_(const ir::SectionStmtPtr& op)
{
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op != nullptr)
        << "Internal error: null SectionStmt";

    // SIMD VF functions are standalone functions generated by GenerateVfFunction;
    // no SectionStmt can survive into CCE codegen.
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, false)
        << "SectionStmt must be eliminated or projected before CCE CodeGen";
}

// The two generic dispatch entries are the only gate the codegen walk passes, so
// publishing the DSL location here reaches every emitter below without adding a
// span parameter to each. SpanScope ignores an invalid span.
void CCECodegen::VisitStmt(const ir::StmtPtr& stmt)
{
    const pypto::pro::SpanScope spanScope(stmt ? stmt->span_ : ir::Span::Unknown());
    ir::IRVisitor::VisitStmt(stmt);
}

void CCECodegen::VisitExpr(const ir::ExprPtr& expr)
{
    const pypto::pro::SpanScope spanScope(expr ? expr->span_ : ir::Span::Unknown());
    ir::IRVisitor::VisitExpr(expr);
}

void CCECodegen::GenerateBody(const ir::FunctionPtr& func)
{
    if (func->body_) {
        VisitStmt(func->body_);
    }

    emitter_.DecreaseIndent();
    emitter_.EmitLine("}");
}

void CCECodegen::VisitStmt_(const ir::AssignStmtPtr& op)
{
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op != nullptr)
        << "Internal error: null AssignStmt";
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op->value_ != nullptr)
        << "Internal error: AssignStmt has null value";

    PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, op->targets_.size() <= 1)
        << "CCE codegen does not support AssignStmt with multiple targets";
    if (op->targets_.empty()) {
        currentTargetVar_ = "";
        currentTuple_ = nullptr;
        // Backend side-effect ops emit their statements while visiting the expression. A direct
        // VF callee/VF function call instead returns its call text, which must be emitted here.
        // Results from all other evaluation expressions are intentionally discarded.
        auto call = ir::As<ir::Call>(op->value_);
        bool isVfCall = call != nullptr && FindCallableCallee(call->name_) != nullptr;
        currentExprValue_ = "";
        VisitExpr(op->value_);
        if (isVfCall && !currentExprValue_.empty()) {
            emitter_.EmitLine(currentExprValue_ + ";");
        }
        currentExprValue_ = "";
        currentTuple_ = nullptr;
        return;
    }

    auto targetVar = op->targets_.front();
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, targetVar != nullptr)
        << "Internal error: AssignStmt has null target";
    std::string varName = context_.SanitizeName(targetVar);
    context_.RegisterVar(targetVar, varName);

    // Both outputs of the same visit are needed here -- the code to bind the name to, and the
    // underlying MakeTuple to register -- so this walks the expression directly instead of
    // going through GetExprAsCode/GetExprAsMakeTuple, which each surface only one of the two.
    currentTargetVar_ = varName;
    currentExprValue_ = "";
    currentTuple_ = nullptr;
    VisitExpr(op->value_);

    if (!currentExprValue_.empty()) {
        auto varType = targetVar->GetType();
        auto tileType = ir::As<ir::TileType>(varType);
        bool isTuple = ir::As<ir::TupleType>(varType) != nullptr;
        bool rhsIsVar = ir::As<ir::Var>(op->value_) != nullptr;
        if (tileType != nullptr && IsInSimdContext()) {
            emitter_.EmitLine(GetGeneratedType(varType) + " " + varName + " = " + currentExprValue_ + ";");
            context_.RegisterVar(targetVar, varName);
            EmitVFTileValidShape(targetVar, op->value_, true);
        } else if (tileType != nullptr || isTuple || rhsIsVar || context_.HasVarName(currentExprValue_)) {
            // Tile/tuple aliases, simple var-to-var copies, and tuple items that
            // are already-declared variables: register as alias (no code emitted).
            context_.RegisterVar(targetVar, currentExprValue_);
        } else {
            // Scalar non-trivial expression: emit a local variable declaration so the
            // expression (e.g. min(), %, *) is computed once and not re-expanded at
            // every use site (avoids VFLoop Cond containing function calls).
            emitter_.EmitLine("auto " + varName + " = " + currentExprValue_ + ";");
            context_.RegisterVar(targetVar, varName);
        }
    }
    if (currentTuple_) {
        tupleVarToMakeTuple_.emplace(varName, currentTuple_); // first-write wins

        // Materialize an array tuple only once. Constant propagation may expose
        // this same MakeTuple through multiple tuple Vars, but all aliases must
        // resolve through the tuple's unique backing array.
        auto makeTuple = currentTuple_;
        auto tt = ir::As<ir::TupleType>(targetVar->GetType());
        if (IsArrayTuple(tt) && tupleBackingArr_.count(makeTuple.get()) == 0) {
            auto elemNames = CollectTupleElemNames(targetVar);
            PRO_CODEGEN_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, !elemNames.empty())
                << "Array tuple assignment to '" << varName << "' has elements without a C++ name at "
                << op->span_.ToString();
            emitter_.EmitLine(BuildDynamicTupleArrayDecl(tt->types_[0], elemNames, varName));
            EmitVFTileArrayValidShape(varName, tt, makeTuple, true);
            tupleBackingArr_.emplace(makeTuple.get(), varName);
        }
    }

    currentTuple_ = nullptr;
    currentExprValue_ = "";
    currentTargetVar_ = "";
}

void CCECodegen::VisitStmt_(const ir::ReturnStmtPtr& op)
{
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op != nullptr)
        << "Internal error: null ReturnStmt";
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->value_.size() <= 1)
        << "ReturnStmt must contain at most one value";
    if (op->value_.empty()) {
        emitter_.EmitLine("return;");
    } else {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, ir::As<ir::ScalarType>(op->value_[0]->GetType()) != nullptr)
            << "ReturnStmt value must be a scalar";
        emitter_.EmitLine("return " + GetExprAsCode(op->value_[0]) + ";");
    }
    currentExprValue_.clear();
}

void CCECodegen::VisitStmt_(const ir::YieldStmtPtr& op)
{
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op != nullptr)
        << "Internal error: null YieldStmt";

    if (op->value_.empty()) {
        yieldBuffer_.clear();
        return; // No values to yield
    }

    yieldBuffer_ = op->value_;
    currentExprValue_ = "";
}

void CCECodegen::EmitYieldAssignments(const std::vector<ir::VarPtr>& returnVars)
{
    if (returnVars.empty() || yieldBuffer_.empty()) {
        yieldBuffer_.clear();
        return;
    }
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, returnVars.size() == yieldBuffer_.size())
        << "IfStmt yield values must match its return variables";
    for (size_t i = 0; i < returnVars.size(); ++i) {
        EmitVariable(returnVars[i], yieldBuffer_[i], false);
    }
    yieldBuffer_.clear();
}

// ========================================================================
// Dynamic tuple array declaration: visitExpr-driven resolution
// ========================================================================

std::vector<std::string> CCECodegen::CollectTupleElemNames(const ir::ExprPtr& tupleValue)
{
    // Step 1: resolve the tuple expression to its underlying MakeTuple.
    auto mt = GetExprAsMakeTuple(tupleValue);
    if (!mt)
        return {};

    // Step 2: name each element. For dynamic-GetItem the elements are tile/scalar Var or const
    // literals, so each one has a C++ name of its own; anything else cannot back an array.
    std::vector<std::string> names;
    names.reserve(mt->elements_.size());
    for (const auto& elem : mt->elements_) {
        std::string name = GetExprAsCode(elem);
        if (name.empty())
            return {};
        names.push_back(std::move(name));
    }
    return names;
}

std::string CCECodegen::GetGeneratedType(const ir::TypePtr& type) const
{
    if (auto scalarType = ir::As<ir::ScalarType>(type)) {
        return scalarType->dtype_.ToCTypeString();
    }
    if (auto regType = ir::As<ir::RegType>(type)) {
        switch (regType->regtype_) {
            case ir::RegKind::VecReg:
                if (regType->dtype_ == DataType::INT4) {
                    return "vector_s4x2";
                }
                if (regType->dtype_ == DataType::UINT4) {
                    return "vector_u4x2";
                }
                return "RtWrap<" + regType->dtype_.ToCTypeString() + ">";
            case ir::RegKind::PredReg:
                return "MaskReg";
            case ir::RegKind::AddrReg:
                return "AddrReg";
            case ir::RegKind::UnalignReg:
                return "UnalignReg";
            default:
                break;
        }
    }
    if (auto ptrType = ir::As<ir::PtrType>(type)) {
        return "__gm__ " + ptrType->dtype_.ToCTypeString() + "*";
    }
    if (auto tensorType = ir::As<ir::TensorType>(type)) {
        return "__gm__ " + tensorType->dtype_.ToCTypeString() + "*";
    }
    if (auto tileType = ir::As<ir::TileType>(type); tileType && (IsInSimtContext() || IsInSimdContext())) {
        return "__ubuf__ " + tileType->dtype_.ToCTypeString() + "*";
    }
    if (auto tileType = ir::As<ir::MutexTileType>(type)) {
        auto shapeDims = ExtractShapeDimensions(tileType->shape_);
        int64_t rows = shapeDims.size() >= 1 ? shapeDims[0] : 1;
        int64_t cols = shapeDims.size() >= 2 ? shapeDims[1] : 1;
        return typeConverter_.ConvertTileType(tileType, rows, cols);
    }
    if (auto tupleType = ir::As<ir::TupleType>(type)) {
        if (const std::string* structName = GetStructName(tupleType)) {
            return *structName;
        }
    }
    // Single-element basic types only. Aggregate tuples and arrays are flattened to their
    // elements before reaching here, so an unregistered composite type is a codegen bug.
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, false) << "Unsupported type for CCE codegen: " << type->TypeName();
    return "";
}

std::string CCECodegen::BuildDynamicTupleArrayDecl(const ir::TypePtr& elemType,
                                                   const std::vector<std::string>& elemNames,
                                                   const std::string& arrName) const
{
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, !elemNames.empty())
        << "Array tuple '" << arrName << "' has no elements";

    const auto elemCppType = GetGeneratedType(elemType);
    std::ostringstream init;
    for (size_t i = 0; i < elemNames.size(); ++i) {
        if (i > 0)
            init << ", ";
        init << "static_cast<" << elemCppType << ">(" << elemNames[i] << ")";
    }

    return elemCppType + " " + arrName + "[] = {" + init.str() + "};";
}

bool CCECodegen::IsHomogeneousTuple(const ir::TupleTypePtr& tt) const
{
    if (!tt || tt->types_.empty())
        return false;
    for (size_t i = 1; i < tt->types_.size(); ++i) {
        if (!ir::structural_equal(tt->types_[i], tt->types_[0]))
            return false;
    }
    return true;
}

bool CCECodegen::IsArrayTuple(const ir::TupleTypePtr& tt) const
{
    if (!tt || tt->types_.empty())
        return false;
    if (GetTupleTypeInfo(tt) != nullptr || !IsHomogeneousTuple(tt))
        return false;
    auto elemTuple = ir::As<ir::TupleType>(tt->types_[0]);
    return !elemTuple || GetStructName(elemTuple) != nullptr;
}

void CCECodegen::EmitFullPhiIf(const ir::IfStmtPtr& op)
{
    // A runtime scalar phi if/else inside a __simd_vf__ body is not supported:
    // bisheng's VF instruction selection cannot legalize a runtime scalar
    // condition branch, and there is no select rewrite for it. Reject the
    // writing style up front instead of emitting broken code.
    for (const auto& returnVar : op->returnVars_) {
        auto scalarTy = ir::As<ir::ScalarType>(returnVar->GetType());
        PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR,
                          !IsInSimdContext() || scalarTy == nullptr || IsBoolDtype(scalarTy->dtype_))
            << "if/else producing a runtime scalar phi is not supported inside a SIMD VF function";
        auto tileTy = ir::As<ir::TileType>(returnVar->GetType());
        PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, !IsInSimdContext() || tileTy == nullptr)
            << "if/else producing a Tile phi is not supported inside a SIMD VF function: "
               "merging Tile values across an if/else is not supported";
    }

    // Declare the phi slots before the `if`: a phi has no initial value, but its C++ shape
    // follows from its type alone, including a struct name, which codegen seeds from IRDebugInfo
    // at entry rather than discovering as it walks the branches.
    for (const auto& returnVar : op->returnVars_) {
        context_.RegisterVar(returnVar, context_.SanitizeName(returnVar));
        EmitVariable(returnVar, nullptr, true);
    }

    std::string condition = GetExprAsCode(op->condition_);

    emitter_.EmitLine("if (" + condition + ") {");
    emitter_.IncreaseIndent();
    yieldBuffer_.clear();
    VisitStmt(op->thenBody_);
    EmitYieldAssignments(op->returnVars_);
    emitter_.DecreaseIndent();

    if (op->elseBody_.has_value()) {
        emitter_.EmitLine("} else {");
        emitter_.IncreaseIndent();
        yieldBuffer_.clear();
        VisitStmt(*op->elseBody_);
        EmitYieldAssignments(op->returnVars_);
        emitter_.DecreaseIndent();
    }
    emitter_.EmitLine("}");
}

void CCECodegen::VisitStmt_(const ir::IfStmtPtr& op)
{
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op != nullptr)
        << "Internal error: null IfStmt";
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op->condition_ != nullptr)
        << "Internal error: IfStmt has null condition";
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op->thenBody_ != nullptr)
        << "Internal error: IfStmt has null then_body";

    EmitFullPhiIf(op);
}

// ========================================================================
// Phase 5 helpers: ForStmt iter-arg registration and carried assignments
// ========================================================================

bool CCECodegen::CheckEmitVariable(const ir::VarPtr& target, ir::ExprPtr& value, bool initialize)
{
    const auto& targetType = target->GetType();
    if (ir::As<ir::UnknownType>(targetType) || ir::As<ir::NoneType>(targetType)) {
        return false;
    }

    // UnknownType and NoneType sources are valueless. Initialization still
    // declares a valid target slot; a later write-back is skipped completely.
    if (value && (ir::As<ir::UnknownType>(value->GetType()) || ir::As<ir::NoneType>(value->GetType()))) {
        value = nullptr;
    }
    // A control-flow back edge carries a value the body never modified, so the yielded
    // expression *is* the slot's own variable. Writing it back would emit `x = x`.
    if (value == target) {
        value = nullptr;
    }
    // With no value left, only a pending declaration is still worth emitting.
    return initialize || value != nullptr;
}

void CCECodegen::EmitVariable(const ir::VarPtr& target, ir::ExprPtr value, bool initialize)
{
    if (!CheckEmitVariable(target, value, initialize)) {
        return;
    }

    const auto& type = target->GetType();
    std::string name = context_.SanitizeName(target);
    if (ir::As<ir::TensorType>(type)) {
        PRO_CODEGEN_THROW(::pypto::ir::RuntimeError, ExternalError::NOT_IMPLEMENTED_ERROR)
            << "Tensor variable code generation is not supported";
    }
    // A struct is a single C++ object with a type name of its own, so it is emitted like any
    // other scalar. Only a tuple without one needs splitting into an array or leaf slots.
    if (auto tupleType = ir::As<ir::TupleType>(type); tupleType && GetStructName(tupleType) == nullptr) {
        EmitTupleVariable(name, tupleType, value, initialize);
        return;
    }

    std::string source = value ? GetExprAsCode(value) : "";
    if (initialize) {
        emitter_.EmitLine(GetGeneratedType(type) + " " + name + (value ? " = " + source : "") + ";");
    } else {
        emitter_.EmitLine(name + " = " + source + ";");
    }
    if (ir::As<ir::TileType>(type) && IsInSimdContext()) {
        EmitVFTileValidShape(target, value, initialize);
    }
}

void CCECodegen::EmitTupleVariable(const std::string& name, const ir::TupleTypePtr& type, const ir::ExprPtr& value,
                                   bool initialize)
{
    ir::MakeTuplePtr source;
    ir::Span span = value ? value->span_ : ir::Span::Unknown();
    if (value) {
        source = GetExprAsMakeTuple(value);
        PRO_CODEGEN_INTERNAL_CHECK_WITH_SPAN(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, source, span)
            << "Tuple assignment source does not resolve to MakeTuple";
    }

    // The representation follows the type, not the source value, so a merge variable can be
    // declared from its type alone once its control-flow structure has been traversed.
    // `initialize` means this is a fresh slot: define it and take ownership of its storage.
    if (IsArrayTuple(type)) {
        const auto& elemType = type->types_[0];

        if (initialize) {
            // The declaration depends only on the type; only the initial value comes from source.
            auto targetTuple = std::make_shared<ir::MakeTuple>(source ? source->elements_ : std::vector<ir::ExprPtr>{},
                                                               span);
            tupleVarToMakeTuple_[name] = targetTuple;
            tupleBackingArr_[targetTuple.get()] = name;
            emitter_.EmitLine(GetGeneratedType(elemType) + " " + name + "[" + std::to_string(type->types_.size()) +
                              "];");
        }
        if (source) {
            auto sourceArray = tupleBackingArr_.find(source.get());
            PRO_CODEGEN_INTERNAL_CHECK_WITH_SPAN(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR,
                                                 sourceArray != tupleBackingArr_.end(), span)
                << "Array tuple assignment to '" << name << "' requires an array source";
            if (name != sourceArray->second) {
                for (size_t i = 0; i < type->types_.size(); ++i) {
                    std::string index = "[" + std::to_string(i) + "]";
                    emitter_.EmitLine(name + index + " = " + sourceArray->second + index + ";");
                }
            }
        }
        EmitVFTileArrayValidShape(name, type, source, initialize);
        return;
    }

    // Aggregate: flatten into one leaf slot per element, recursing into nested tuples.
    if (initialize) {
        std::vector<ir::ExprPtr> elements;
        elements.reserve(type->types_.size());
        for (size_t i = 0; i < type->types_.size(); ++i) {
            std::string elementName = TupleItemName(name, i);
            ir::ExprPtr elementValue = source ? source->elements_[i] : nullptr;
            // A struct is a leaf: it has a C++ type name of its own, so it is declared and
            // assigned whole. Only a tuple without one is taken apart further.
            if (auto elementType = ir::As<ir::TupleType>(type->types_[i]);
                elementType && GetStructName(elementType) == nullptr) {
                EmitTupleVariable(elementName, elementType, elementValue, true);
                elements.push_back(tupleVarToMakeTuple_.at(elementName));
            } else {
                auto element = std::make_shared<ir::Var>(elementName, type->types_[i], span);
                context_.RegisterVar(element, elementName);
                elements.push_back(element);
                EmitVariable(element, elementValue, true);
            }
        }
        tupleVarToMakeTuple_[name] = std::make_shared<ir::MakeTuple>(std::move(elements), span);
        return;
    }

    // Reuse the leaf slots recorded when this aggregate was defined.
    auto target = tupleVarToMakeTuple_.find(name);
    PRO_CODEGEN_INTERNAL_CHECK_WITH_SPAN(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR,
                                         target != tupleVarToMakeTuple_.end(), span)
        << "Assignment to undefined aggregate tuple '" << name << "'";
    for (size_t i = 0; i < type->types_.size(); ++i) {
        ir::ExprPtr elementValue = source ? source->elements_[i] : nullptr;
        if (auto elementType = ir::As<ir::TupleType>(type->types_[i]);
            elementType && GetStructName(elementType) == nullptr) {
            EmitTupleVariable(TupleItemName(name, i), elementType, elementValue, false);
            continue;
        }
        auto element = ir::As<ir::Var>(target->second->elements_[i]);
        PRO_CODEGEN_INTERNAL_CHECK_WITH_SPAN(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, element, span)
            << "Aggregate tuple '" << name << "' has no leaf slot at " << i;
        EmitVariable(element, elementValue, false);
    }
}

std::vector<std::string> CCECodegen::RegisterLoopIterArgs(const std::vector<ir::IterArgPtr>& iterArgs)
{
    // Registration only: EmitLoop emits each iter-arg declaration inline right
    // before the loop header (and its body uses), so the declarations are
    // never written here.
    std::vector<std::string> iterArgNames;
    iterArgNames.reserve(iterArgs.size());

    for (auto& iterArg : iterArgs) {
        const auto& var = iterArg->iterVar_;
        std::string iterArgName = context_.SanitizeName(var);

        context_.RegisterVar(var, iterArgName);
        iterArgNames.push_back(iterArgName);
    }
    return iterArgNames;
}

void CCECodegen::EmitCarriedAssignments(const std::vector<ir::VarPtr>& targets, const std::vector<ir::ExprPtr>& sources)
{
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, targets.size() == sources.size())
        << "Loop-carried write-back expects " << targets.size() << " values but got " << sources.size();

    // A jump transfers all SSA operands simultaneously. In C++ the carried slots
    // are mutable, and a source may alias another slot (e.g. a, b = b, a). Save
    // the outgoing values before overwriting any slot, including tuple storage.
    // Use EmitVariable rather than AssignStmt: the latter aliases Var sources
    // instead of copying them, which would leave the write-back hazard intact.
    std::vector<ir::ExprPtr> savedSources = sources;
    // One aggregate target can still contain several mutable leaf slots. In
    // particular, nested tuple fields may alias those slots without a scalar
    // AssignStmt to materialize their old values before a permutation.
    if (targets.size() > 1 || (!targets.empty() && ir::As<ir::TupleType>(targets.front()->GetType()))) {
        for (size_t i = 0; i < targets.size(); ++i) {
            auto& source = savedSources[i];
            if (!CheckEmitVariable(targets[i], source, false) || ir::As<ir::ConstInt>(source) ||
                ir::As<ir::ConstFloat>(source) || ir::As<ir::ConstBool>(source)) {
                continue;
            }
            auto saved = std::make_shared<const ir::Var>(context_.SanitizeName(targets[i]) + "__next",
                                                         targets[i]->GetType(), source->span_);
            context_.RegisterVar(saved, context_.SanitizeName(saved));
            EmitVariable(saved, source, true);
            source = saved;
        }
    }
    for (size_t i = 0; i < targets.size(); ++i) {
        EmitVariable(targets[i], savedSources[i], false);
    }
}

void CCECodegen::RegisterLoopReturnVars(const std::vector<ir::VarPtr>& returnVars,
                                        const std::vector<std::string>& iterArgNames)
{
    if (returnVars.empty()) {
        return;
    }
    for (size_t i = 0; i < returnVars.size(); ++i) {
        const auto& returnVar = returnVars[i];
        if (i < iterArgNames.size()) {
            context_.RegisterVar(returnVar, iterArgNames[i]);
            if (ir::As<ir::TensorType>(returnVar->GetType())) {
                RegisterPointer(context_.SanitizeName(returnVar), GetPointer(iterArgNames[i]));
            }
            if (ir::As<ir::TupleType>(returnVar->GetType())) {
                auto tupleIt = tupleVarToMakeTuple_.find(iterArgNames[i]);
                if (tupleIt != tupleVarToMakeTuple_.end()) {
                    tupleVarToMakeTuple_[context_.SanitizeName(returnVar)] = tupleIt->second;
                }
            }
        } else {
            PRO_CODEGEN_THROW(::pypto::ir::RuntimeError, ExternalError::INVALID_VAL)
                << "Loop return_var has no corresponding iter_arg";
        }
    }
}

void CCECodegen::VisitStmt_(const ir::ForStmtPtr& op)
{
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op != nullptr)
        << "Internal error: null ForStmt";
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op->loopVar_ != nullptr)
        << "Internal error: ForStmt has null loop_var";
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op->start_ != nullptr)
        << "Internal error: ForStmt has null start";
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op->stop_ != nullptr)
        << "Internal error: ForStmt has null stop";
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op->step_ != nullptr)
        << "Internal error: ForStmt has null step";
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op->body_ != nullptr)
        << "Internal error: ForStmt has null body";

    // Check consistency: iter_args and return_vars must have same size
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->iterArgs_.size() == op->returnVars_.size())
        << "ForStmt iter_args size (" << op->iterArgs_.size() << ") must equal return_vars size ("
        << op->returnVars_.size() << ")";

    auto stopCi = ir::As<ir::ConstInt>(op->stop_);

    // Register loop variable
    std::string loopVarName = context_.SanitizeName(op->loopVar_);
    context_.RegisterVar(op->loopVar_, loopVarName);

    std::vector<std::string> iterArgNames = RegisterLoopIterArgs(op->iterArgs_);

    // Evaluate loop range
    VisitExpr(op->start_);
    std::string start = currentExprValue_;
    currentExprValue_ = "";

    VisitExpr(op->stop_);
    std::string stop = currentExprValue_;
    currentExprValue_ = "";

    VisitExpr(op->step_);
    std::string step = currentExprValue_;
    currentExprValue_ = "";

    // --- Emit for-loop with hoisting ---
    // In a __simd_vf__ body, bisheng requires uint16_t loop variables AND the bound expression to
    // also resolve to uint16_t (otherwise -Wcce-compat -Werror fails the build), so cast inside a
    // SIMD VF function. SIMT bodies use int64_t loops like the kernel.
    std::string loopType = IsInSimdContext() ? "uint16_t" : "int64_t";
    std::string stopExpr = IsInSimdContext() ? ("(uint16_t)(" + stop + ")") : stop;
    // Perf: in a SIMD VF body, hoist a NON-constant loop bound into a `const` local
    // and use that as the trip count.  A literal-constant bound is already ideal,
    // so it is left inline.  Binding a variable bound to a const evaluates it once
    // and lets the bisheng VF compiler treat the trip count as loop-invariant
    // (enabling unrolling / vectorization it would otherwise skip).
    if (IsInSimdContext() && !stopCi) {
        std::string boundName = loopVarName + "_ub";
        emitter_.EmitLine("const uint16_t " + boundName + " = " + stopExpr + ";");
        stopExpr = boundName;
    }
    std::string header = "for (" + loopType + " " + loopVarName + " = " + start + "; " + loopVarName + " < " +
                         stopExpr + "; " + loopVarName + " += " + step + ") {";
    EmitLoop(header, op->body_, op->iterArgs_, op->returnVars_, iterArgNames, false);
}

void CCECodegen::EmitLoop(const std::string& header, const ir::StmtPtr& body,
                          const std::vector<ir::IterArgPtr>& iterArgs, const std::vector<ir::VarPtr>& returnVars,
                          const std::vector<std::string>& iterArgNames, bool separateWhileResults)
{
    // The iter-arg slots are declared before the loop. Loop carries only include
    // bindings defined before the loop, so every slot has an initial value.
    std::vector<ir::VarPtr> iterVars;
    iterVars.reserve(iterArgs.size());
    for (const auto& iterArg : iterArgs) {
        iterVars.push_back(iterArg->iterVar_);
        EmitVariable(iterArg->iterVar_, iterArg->initValue_, true);
    }
    if (separateWhileResults) {
        for (const auto& returnVar : returnVars) {
            context_.RegisterVar(returnVar, context_.SanitizeName(returnVar));
            EmitVariable(returnVar, nullptr, true);
        }
    }
    if (!iterArgs.empty() || (separateWhileResults && !returnVars.empty())) {
        emitter_.EmitLine("");
    }

    emitter_.EmitLine(header);
    emitter_.IncreaseIndent();

    // For loops share one carried slot set. While loops keep the back-edge slots and
    // break-result slots independent because their inferred types may differ.
    loopTargetStack_.push_back(LoopWritebackTargets{iterVars, separateWhileResults ? returnVars : iterVars});
    VisitStmt(body);
    loopTargetStack_.pop_back();

    emitter_.DecreaseIndent();
    emitter_.EmitLine("}");

    // For return variables capture the final iter-arg values through aliases. While
    // return variables were declared independently and are written only by break.
    if (!separateWhileResults) {
        RegisterLoopReturnVars(returnVars, iterArgNames);
    }
}

void CCECodegen::VisitStmt_(const ir::WhileStmtPtr& op)
{
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op != nullptr)
        << "Internal error: null WhileStmt";
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op->condition_ != nullptr)
        << "Internal error: WhileStmt has null condition";
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op->body_ != nullptr)
        << "Internal error: WhileStmt has null body";

    // iter_args and return_vars must agree in size (each loop-carried value has a phi).
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->iterArgs_.size() == op->returnVars_.size())
        << "WhileStmt iter_args size (" << op->iterArgs_.size() << ") must equal return_vars size ("
        << op->returnVars_.size() << ")";

    // Every loop path ends in parser-created BreakStmt/ContinueStmt values.
    // Continue updates iter_args; break updates the independent return_vars.
    std::vector<std::string> iterArgNames = RegisterLoopIterArgs(op->iterArgs_);

    // The condition is re-evaluated each iteration, so emit it inline in the while header.
    // Comparison/logical exprs lower to pure C++ expression strings (see IMPLEMENT_BINARY_OP),
    // so they are safe to place directly in `while (...)`.
    std::string condition = GetExprAsCode(op->condition_);

    EmitLoop("while (" + condition + ") {", op->body_, op->iterArgs_, op->returnVars_, iterArgNames, true);
}

// ========================================================================
// Native break/continue support
// ========================================================================

// Write a native jump's self-described carried values to the matching slots before jumping.
void CCECodegen::EmitJumpCarriedWriteback(const std::vector<ir::ExprPtr>& values, bool isBreak)
{
    if (values.empty() || loopTargetStack_.empty()) {
        return; // bare jump: loop carries nothing, or jump is outside a tracked loop
    }
    const auto& targets = loopTargetStack_.back();
    EmitCarriedAssignments(isBreak ? targets.breakVars : targets.continueVars, values);
}

void CCECodegen::VisitStmt_(const ir::BreakStmtPtr& op)
{
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op != nullptr)
        << "Internal error: null BreakStmt";
    EmitJumpCarriedWriteback(op->value_, true);
    emitter_.EmitLine("break;");
}

void CCECodegen::VisitStmt_(const ir::ContinueStmtPtr& op)
{
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op != nullptr)
        << "Internal error: null ContinueStmt";
    EmitJumpCarriedWriteback(op->value_, false);
    emitter_.EmitLine("continue;");
}

void CCECodegen::VisitStmt_(const ir::SeqStmtsPtr& op)
{
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op != nullptr)
        << "Internal error: null SeqStmts";
    for (const auto& stmt : op->stmts_) {
        VisitStmt(stmt);
        // A native break/continue/return ends this straight-line sequence; statements after it are
        // unreachable even when a conservative control-flow producer left a trailing terminator.
        if (ir::As<ir::BreakStmt>(stmt) || ir::As<ir::ContinueStmt>(stmt) || ir::As<ir::ReturnStmt>(stmt)) {
            break;
        }
    }
}

// ========================================================================
// Expression Visitor Methods - Dual-Mode Pattern
// ========================================================================
// - Statement-Emitting Mode (Call): Uses currentTargetVar_, emits instructions
// - Value-Returning Mode (others): Sets currentExprValue_ with inline C++ code

// ---- Leaf Nodes ----
void CCECodegen::VisitExpr_(const ir::VarPtr& op)
{
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op != nullptr)
        << "Internal error: null Var";
    std::string name = context_.SanitizeName(op);
    auto it = tupleVarToMakeTuple_.find(name);
    if (it == tupleVarToMakeTuple_.end()) {
        it = tupleVarToMakeTuple_.find(name + "_0");
    }
    // Positional tuple Var: produce the underlying MakeTuple in currentTuple_
    // (with _0 SSA fallback). Empty currentExprValue_ signals "no direct identifier".
    currentTuple_ = (it != tupleVarToMakeTuple_.end()) ? it->second : nullptr;

    currentExprValue_ = context_.GetVarName(op);
}

void CCECodegen::VisitExpr_(const ir::MakeTuplePtr& op)
{
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op != nullptr)
        << "Internal error: null MakeTuple";
    currentTuple_ = op;
    currentExprValue_ = "";
}

void CCECodegen::VisitExpr_(const ir::ConstIntPtr& op)
{
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op != nullptr)
        << "Internal error: null ConstInt";
    currentExprValue_ = FormatIntCLiteral(op->value_, op->dtype());
}

void CCECodegen::VisitExpr_(const ir::ConstFloatPtr& op)
{
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op != nullptr)
        << "Internal error: null ConstFloat";
    double val = op->value_;
    if (std::isnan(val)) {
        currentExprValue_ = "__builtin_nanf(\"\")";
    } else if (std::isinf(val)) {
        currentExprValue_ = val > 0 ? "__builtin_huge_valf()" : "(-__builtin_huge_valf())";
    } else {
        currentExprValue_ = FormatFloatCLiteral(val) + "f";
    }
    if (op->dtype() == DataType::BF16 || op->dtype() == DataType::FP16) {
        currentExprValue_ = ScalarCast(currentExprValue_, DataType::FP32, op->dtype());
    }
}

void CCECodegen::VisitExpr_(const ir::ConstBoolPtr& op)
{
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op != nullptr)
        << "Internal error: null ConstBool";
    currentExprValue_ = op->value_ ? "true" : "false";
}

std::string CCECodegen::GetOrCreateVFTilePtr(const ir::ExprPtr& expr, bool isPostUpdate)
{
    auto tileType = ir::As<ir::TileType>(expr->GetType());
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, tileType != nullptr)
        << "GetOrCreateVFTilePtr expects a tile expr, got " << expr->TypeName();
    std::string code = GetExprAsCode(expr);

    // POST_UPDATE advances the pointer operand in place, so it must be a
    // modifiable lvalue: a tile parameter or a tile-typed binding (both real
    // __ubuf__* variables in a SIMD body). An unbound offset expression is an
    // rvalue and cannot be advanced — bind it to a variable first.
    if (isPostUpdate) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, ir::As<ir::Var>(expr) != nullptr)
            << "POST_UPDATE requires a named tile variable (bind the expression first: base = tile + offset), got "
            << expr->TypeName();
    }
    return code;
}

std::optional<int64_t> CCECodegen::GetStaticTileValidDim(const ir::TileTypePtr& type, int axis)
{
    if (type->tileView_ && static_cast<size_t>(axis) < type->tileView_->validShape.size()) {
        auto dim = ir::As<ir::ConstInt>(type->tileView_->validShape[axis]);
        if (dim && dim->value_ >= 0) {
            return dim->value_;
        }
    }
    return std::nullopt;
}

std::string CCECodegen::GetVFTileValidShape(const ir::ExprPtr& expr, int axis)
{
    if (auto dim = GetStaticTileValidDim(ir::As<ir::TileType>(expr->GetType()), axis)) {
        return std::to_string(*dim);
    }
    if (auto call = ir::As<ir::Call>(expr); call && call->name_ == "block.subview") {
        // Linear pointer offsets preserve the tile metadata. A slice carries its
        // own valid extent, already computed by the frontend.
        if (!ir::As<ir::MakeTuple>(call->args_[1])) {
            return GetVFTileValidShape(call->args_[0], axis);
        }
        auto shape = ir::As<ir::MakeTuple>(call->args_[2]);
        return GetExprAsCode(shape->elements_[axis]);
    }
    if (auto item = ir::As<ir::GetItemExpr>(expr)) {
        auto tuple = GetExprAsMakeTuple(item->value_);
        if (tuple) {
            auto array = tupleBackingArr_.find(tuple.get());
            if (array != tupleBackingArr_.end()) {
                return array->second + (axis == 0 ? "__valid_row" : "__valid_col") + "[" + GetExprAsCode(item->slice_) +
                       "]";
            }
            auto index = ir::As<ir::ConstInt>(item->slice_);
            return GetVFTileValidShape(tuple->elements_[index->value_], axis);
        }
    }
    return GetExprAsCode(expr) + (axis == 0 ? "__valid_row" : "__valid_col");
}

void CCECodegen::EmitVFTileArrayValidShape(const std::string& name, const ir::TupleTypePtr& type,
                                           const ir::MakeTuplePtr& source, bool initialize)
{
    auto tile = ir::As<ir::TileType>(type->types_[0]);
    if (!IsInSimdContext() || !tile) {
        return;
    }
    auto sourceArray = source ? tupleBackingArr_.find(source.get()) : tupleBackingArr_.end();
    for (int axis = 0; axis < 2; ++axis) {
        if (GetStaticTileValidDim(tile, axis)) {
            continue;
        }
        const std::string suffix = axis == 0 ? "__valid_row" : "__valid_col";
        if (initialize) {
            emitter_.EmitLine("uint32_t " + name + suffix + "[" + std::to_string(type->types_.size()) + "];");
        }
        if (!source || (sourceArray != tupleBackingArr_.end() && name == sourceArray->second)) {
            continue;
        }
        for (size_t i = 0; i < type->types_.size(); ++i) {
            const std::string index = "[" + std::to_string(i) + "]";
            const std::string value = sourceArray != tupleBackingArr_.end() ?
                                          sourceArray->second + suffix + index :
                                          GetVFTileValidShape(source->elements_[i], axis);
            emitter_.EmitLine(name + suffix + index + " = " + value + ";");
        }
    }
}

void CCECodegen::EmitVFTileValidShape(const ir::VarPtr& target, const ir::ExprPtr& value, bool initialize)
{
    const std::string name = context_.SanitizeName(target);
    for (int axis = 0; axis < 2; ++axis) {
        if (GetStaticTileValidDim(ir::As<ir::TileType>(target->GetType()), axis)) {
            continue;
        }
        const std::string suffix = axis == 0 ? "__valid_row" : "__valid_col";
        const std::string source = value ? " = " + GetVFTileValidShape(value, axis) : "";
        emitter_.EmitLine((initialize ? "uint32_t " : "") + name + suffix + source + ";");
    }
}

void CCECodegen::VisitExpr_(const ir::GetItemExprPtr& op)
{
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op != nullptr)
        << "Internal error: null GetItemExpr";

    auto valueType = op->value_->GetType();
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, ir::As<ir::TupleType>(valueType))
        << "GetItemExpr requires value to have TupleType, got " << valueType->TypeName();

    std::string indexCode = GetExprAsCode(op->slice_);
    // Which of the three cases below applies depends on both outputs of one visit of the base,
    // so this walks it directly rather than through the single-output accessors.
    currentTuple_ = nullptr;
    currentExprValue_ = "";
    VisitExpr(op->value_);

    // Case 1: the value is already a materialized expression. Struct TupleTypes
    // require a constant field ordinal. Other TupleTypes represent indexable
    // expressions such as an array-valued struct field.
    if (currentTuple_ == nullptr) {
        std::string baseCode = currentExprValue_;
        auto tupleType = ir::As<ir::TupleType>(valueType);
        const ir::TupleTypeInfo* info = GetTupleTypeInfo(tupleType);
        if (info != nullptr && info->kind == ir::TupleTypeKind::STRUCT) {
            auto constIdx = ir::As<ir::ConstInt>(op->slice_);
            PRO_CODEGEN_CHECK(ExternalError::INVALID_VAL, constIdx != nullptr)
                << "GetItemExpr struct field requires a constant index at " << op->span_.ToString();
            int idx = static_cast<int>(constIdx->value_);
            PRO_CODEGEN_CHECK(ExternalError::OUT_OF_RANGE, idx >= 0 && idx < static_cast<int>(info->fields.size()))
                << "GetItemExpr struct field index " << idx << " out of bounds";
            currentExprValue_ = baseCode + "." + info->fields[idx];
        } else {
            currentExprValue_ = baseCode + "[" + indexCode + "]";
        }
        currentTuple_ = nullptr;
        return;
    }

    // Case 2: a homogeneous MakeTuple has a backing C++ array, so both constant
    // and dynamic indices must be emitted as array accesses.
    auto arrIt = tupleBackingArr_.find(currentTuple_.get());
    if (arrIt != tupleBackingArr_.end()) {
        currentTuple_ = nullptr;
        currentExprValue_ = arrIt->second + "[" + indexCode + "]";
        return;
    }

    // Case 3: a MakeTuple without a backing array can only be evaluated
    // statically. Dynamic indexing is valid only for homogeneous tuples, which
    // must have been materialized as a backing array above.
    auto constIdx = ir::As<ir::ConstInt>(op->slice_);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, constIdx != nullptr)
        << "Dynamic GetItemExpr requires a backing array at " << op->span_.ToString();
    int idx = static_cast<int>(constIdx->value_);
    PRO_CODEGEN_CHECK(ExternalError::OUT_OF_RANGE, idx >= 0 && idx < static_cast<int>(currentTuple_->elements_.size()))
        << "GetItemExpr index " << idx << " out of bounds";
    ir::ExprPtr elem = currentTuple_->elements_[idx];
    currentTuple_ = nullptr;
    currentExprValue_ = "";
    VisitExpr(elem);
    return;
}

// ========================================================================
// CodegenBase interface and CCE-specific helper methods
// ========================================================================

// Both accessors below clear currentExprValue_ and currentTuple_ on the way in and on the
// way out. Clearing on the way in is what makes the result describe this expression alone:
// only Var and MakeTuple write currentTuple_ on every visit, so a scalar expression would
// otherwise inherit whichever tuple the previous visit left behind. Clearing on the way out
// keeps that same leftover from reaching whatever is parsed next.

std::string CCECodegen::GetExprAsCode(const ir::ExprPtr& expr)
{
    currentExprValue_ = "";
    currentTuple_ = nullptr;
    VisitExpr(expr);
    auto result = currentExprValue_;
    currentExprValue_ = "";
    currentTuple_ = nullptr;
    return result;
}

ir::MakeTuplePtr CCECodegen::GetExprAsMakeTuple(const ir::ExprPtr& expr)
{
    currentExprValue_ = "";
    currentTuple_ = nullptr;
    VisitExpr(expr);
    auto result = currentTuple_;
    currentExprValue_ = "";
    currentTuple_ = nullptr;
    return result;
}

void CCECodegen::Emit(const std::string& line) { emitter_.EmitLine(line); }

std::string CCECodegen::GetTypeString(const ir::DataType& dtype) const { return dtype.ToCTypeString(); }

int64_t CCECodegen::GetConstIntValue(const ir::ExprPtr& expr) { return ExtractConstInt(expr); }

std::string CCECodegen::GetVarName(const ir::VarPtr& var) { return context_.GetVarName(var); }

void CCECodegen::RegisterPointer(const std::string& tensorVarName, const std::string& ptrName)
{
    PRO_CODEGEN_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, !tensorVarName.empty())
        << "Cannot register pointer with empty tensor var name";
    PRO_CODEGEN_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, !ptrName.empty())
        << "Cannot register pointer with empty pointer name";

    auto it = tensorToPointer_.find(tensorVarName);
    if (it != tensorToPointer_.end() && it->second != ptrName) {
        IR_LOGW() << "Pointer for tensor " << tensorVarName << " re-registered with: " << ptrName << " vs "
                  << it->second;
    }
    tensorToPointer_[tensorVarName] = ptrName;
}

std::string CCECodegen::GetPointer(const std::string& varName)
{
    auto it = tensorToPointer_.find(varName);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_VAL, it != tensorToPointer_.end())
        << "Pointer for tensor " << varName << " not found";
    return it->second;
}

bool CCECodegen::HasPointer(const std::string& varName) const
{
    return tensorToPointer_.find(varName) != tensorToPointer_.end();
}

std::vector<const TensorDef*> CCECodegen::GetTensorDefs(const std::string& name) const
{
    // Every variant of one tensor shares the key's first half, so they are one contiguous range.
    std::vector<const TensorDef*> defs;
    for (auto it = tensorDefs_.lower_bound({name, ""}); it != tensorDefs_.end() && it->first.first == name; ++it) {
        defs.push_back(&it->second);
    }
    return defs;
}

std::string CCECodegen::ComputeTensorOffset(const ir::TensorTypePtr& tensorType, const ir::MakeTuplePtr& offsets)
{
    const size_t ndim = tensorType->shape_.size();
    PRO_CODEGEN_CHECK(ExternalError::INVALID_SHAPE, offsets->elements_.size() == ndim)
        << "Offset dimensions (" << offsets->elements_.size() << ") != tensor dimensions (" << ndim << ")";

    if (backend::cce::IsNZTensorType(tensorType)) {
        return ComputeNZStrideOffset(tensorType, offsets);
    }
    return ComputeStrideOffset(tensorType, offsets);
}

std::string CCECodegen::ComputeAlignedShapeDimension(const ir::ExprPtr& dimension, int64_t alignment)
{
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, alignment > 0)
        << "Shape alignment must be positive";
    if (const auto value = ir::As<ir::ConstInt>(dimension)) {
        return std::to_string((value->value_ + alignment - 1) / alignment * alignment);
    }
    const std::string expression = GetExprAsCode(dimension);
    return "((" + expression + " + " + std::to_string(alignment - 1) + ") / " + std::to_string(alignment) + " * " +
           std::to_string(alignment) + ")";
}

std::string CCECodegen::ComputeStrideOffset(const ir::TensorTypePtr& tensorType, const ir::MakeTuplePtr& offsets)
{
    const size_t ndim = tensorType->shape_.size();
    const auto strideExprs = BuildTensorStrideExpressions(tensorType);

    std::ostringstream result;
    result << "(";
    bool first = true;
    for (size_t i = 0; i < ndim; ++i) {
        std::string offExpr = GetExprAsCode(offsets->elements_[i]);

        // TensorView stride is used when present; otherwise a contiguous stride is derived from the shape.
        // For a unit stride, add the offset directly.
        if (!first)
            result << " + ";
        first = false;
        result << offExpr;
        if (strideExprs[i] != "1") {
            result << " * (" << strideExprs[i] << ")";
        }
    }
    result << ")";
    // Shapes, strides and offsets all count elements. The pointer this is added to does not: a
    // sub-byte element type packs several elements per storage unit, so the element offset has
    // to be converted. Tile shapes are checked to be whole storage units, which is what keeps a
    // tile-aligned access landing on one.
    const size_t bits = tensorType->dtype_.GetBit();
    constexpr size_t kBitsPerByte = 8;
    if (bits < kBitsPerByte) {
        return "(" + result.str() + " / " + std::to_string(kBitsPerByte / bits) + ")";
    }
    return result.str();
}

std::string CCECodegen::ComputeNZStrideOffset(const ir::TensorTypePtr& tensorType, const ir::MakeTuplePtr& offsets)
{
    const size_t ndim = tensorType->shape_.size();
    PRO_CODEGEN_CHECK(ExternalError::INVALID_SHAPE, ndim >= 2)
        << "CCE NZ tensor lowering requires a tensor rank of at least 2";

    const size_t rowAxis = ndim - 2;
    const size_t colAxis = ndim - 1;
    const int64_t c0 = backend::cce::GetNZInnerCols(tensorType->dtype_);
    const std::string paddedRows = ComputeAlignedShapeDimension(tensorType->shape_[rowAxis], 16);
    const std::string paddedCols = ComputeAlignedShapeDimension(tensorType->shape_[colAxis], c0);

    std::ostringstream result;
    result << "(";
    bool first = true;
    for (size_t i = 0; i < rowAxis; ++i) {
        if (!first) {
            result << " + ";
        }
        first = false;
        result << GetExprAsCode(offsets->elements_[i]);
        for (size_t j = i + 1; j < rowAxis; ++j) {
            result << " * " << GetExprAsCode(tensorType->shape_[j]);
        }
        result << " * " << paddedRows << " * " << paddedCols;
    }

    if (!first) {
        result << " + ";
    }
    result << GetExprAsCode(offsets->elements_[colAxis]) << " * " << paddedRows << " + "
           << GetExprAsCode(offsets->elements_[rowAxis]) << " * " << c0 << ")";
    const std::string logicalOffset = result.str();
    // FP4 C++ pointer types pack two logical elements into one byte, so pointer arithmetic uses pairs.
    return tensorType->dtype_.GetBit() == 4 ? "(" + logicalOffset + " / 2)" : logicalOffset;
}

// ========================================================================
// Call Expression Visitor (uses operator registry codegen functions)
// ========================================================================

ir::FunctionPtr CCECodegen::FindCallableCallee(const std::string& name) const
{
    if (IsInSimtContext()) {
        auto simtCallee = simtCallees_.find(name);
        if (simtCallee != simtCallees_.end()) {
            return simtCallee->second;
        }
        return nullptr;
    }
    if (IsInSimdContext()) {
        auto simdCallee = simdCallees_.find(name);
        if (simdCallee != simdCallees_.end()) {
            return simdCallee->second;
        }
        return nullptr;
    }
    // Kernel-side VF calls use the registered "simd.call" op (analogous to
    // simt.launch); there is no direct-name VF call from the kernel body.
    return nullptr;
}

std::string CCECodegen::GenerateVfCalleeCall(const ir::CallPtr& op, const ir::FunctionPtr& callee)
{
    const bool isSimt = IsSimtFuncType(callee->funcType_);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == callee->params_.size())
        << (isSimt ? "SIMT callee '" : "SIMD callee '") << callee->name_ << "' expects " << callee->params_.size()
        << " arguments, got " << op->args_.size();

    std::ostringstream call;
    call << callee->name_ << "(";
    bool first = true;
    auto append = [&](const std::string& value) {
        if (!first) {
            call << ", ";
        }
        first = false;
        call << value;
    };

    for (size_t i = 0; i < op->args_.size(); ++i) {
        const auto& actual = op->args_[i];
        append(GetExprAsCode(actual));
        if (auto tileType = ir::As<ir::TileType>(actual->GetType())) {
            if (isSimt) {
                auto actualVar = ir::As<ir::Var>(actual);
                PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, actualVar != nullptr)
                    << "SIMT Tile callee arguments must be named Tile variables";
            }
            for (int axis = 0; axis < 2; ++axis) {
                if (!GetStaticTileValidDim(tileType, axis)) {
                    append(GetVFTileValidShape(actual, axis));
                }
            }
        }
    }
    call << ")";
    return call.str();
}

void CCECodegen::VisitExpr_(const ir::CallPtr& op)
{
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op != nullptr)
        << "Internal error: null Call";

    // SIMT callee call from a SIMT body, SIMD callee call from within a
    // __simd_vf__/__simd_callee__ body, or SIMD VF call from the kernel body.
    if (auto callee = FindCallableCallee(op->name_)) {
        currentExprValue_ = GenerateVfCalleeCall(op, callee);
        return;
    }

    PRO_CODEGEN_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, backend_ != nullptr)
        << "CCE backend must not be null";
    const auto* opInfo = backend_->GetOpInfo(op->name_);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, opInfo != nullptr)
        << "Unknown call '" << op->name_ << "' reached CCE codegen; helper calls must be inlined";

    std::string result = opInfo->codegenFunc(op, *this);
    currentExprValue_ = result;
}

// ---- Binary Operators ----

namespace {

std::string BuildPythonIntegerDivModExpression(const std::string& left, const std::string& right, const DataType& dtype,
                                               bool returnRemainder)
{
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, dtype.IsInt())
        << "Internal error: integer FloorDiv/FloorMod requires an integer dtype";
    const std::string cppType = dtype.ToCTypeString();
    if (dtype.IsSignedInt()) {
        std::ostringstream code;
        code << "({" << cppType << " __pypto_lhs = (" << left << "), __pypto_rhs = (" << right << ");";
        if (returnRemainder) {
            code << cppType << " __pypto_rem = __pypto_lhs % __pypto_rhs;"
                 << "__pypto_rem + "
                    "((__pypto_rem != 0 && ((__pypto_rem < 0) != (__pypto_rhs < 0))) ? __pypto_rhs : 0);})";
        } else {
            // C++ evaluates && left-to-right. The common same-sign index path therefore avoids computing a remainder.
            code << cppType << " __pypto_quot = __pypto_lhs / __pypto_rhs;"
                 << "__pypto_quot - (((__pypto_lhs < 0) != (__pypto_rhs < 0)) && "
                    "(__pypto_lhs % __pypto_rhs != 0));})";
        }
        return code.str();
    }
    return "(" + left + (returnRemainder ? " % " : " / ") + right + ")";
}

std::string BuildFP32FloorExpression(const std::string& value)
{
    // Clear the fractional significand bits directly because the CCE scalar floor intrinsic is unavailable.
    std::ostringstream code;
    code << "({float __pypto_floor_input = (" << value << ");"
         << "uint32_t __pypto_floor_bits = reinterpret_cast<uint32_t&>(__pypto_floor_input);"
         << "uint32_t __pypto_floor_abs = __pypto_floor_bits & 0x7fffffffu;"
         << "uint32_t __pypto_floor_exp = __pypto_floor_abs >> 23;"
         << "float __pypto_floor_result = __pypto_floor_input;"
         << "if (__pypto_floor_abs != 0 && __pypto_floor_exp < 127u) {"
         << "__pypto_floor_result = (__pypto_floor_bits & 0x80000000u) ? -1.0f : 0.0f;"
         << "} else if (__pypto_floor_exp >= 127u && __pypto_floor_exp < 150u) {"
         << "uint32_t __pypto_floor_frac_mask = (1u << (150u - __pypto_floor_exp)) - 1u;"
         << "if (__pypto_floor_abs & __pypto_floor_frac_mask) {"
         << "uint32_t __pypto_floor_trunc_bits = __pypto_floor_bits & ~__pypto_floor_frac_mask;"
         << "__pypto_floor_result = reinterpret_cast<float&>(__pypto_floor_trunc_bits);"
         << "if (__pypto_floor_bits & 0x80000000u) { __pypto_floor_result -= 1.0f; }"
         << "}"
         << "}"
         << "__pypto_floor_result;})";
    return code.str();
}

std::string BuildPythonFloatDivModExpression(const std::string& left, const std::string& right, const DataType& dtype,
                                             bool returnRemainder)
{
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, dtype.IsFloat())
        << "Internal error: floating-point FloorDiv/FloorMod requires a float dtype";
    const std::string cppType = dtype.ToCTypeString();
    std::ostringstream code;
    code << "({float __pypto_lhs = (float)(" << left << "), __pypto_rhs = (float)(" << right << ");"
         << "float __pypto_mod = " << BuildFP32FmodExpression("__pypto_lhs", "__pypto_rhs") << ";";
    if (returnRemainder) {
        code << "if (__pypto_mod == 0.0f) {"
             << "uint32_t __pypto_zero = reinterpret_cast<uint32_t&>(__pypto_rhs) & 0x80000000u;"
             << "__pypto_mod = reinterpret_cast<float&>(__pypto_zero);"
             << "} else if ((__pypto_rhs < 0.0f) != (__pypto_mod < 0.0f)) {"
             << "__pypto_mod += __pypto_rhs;"
             << "}";
        code << "(" << cppType << ")__pypto_mod;})";
        return code.str();
    }
    code << "float __pypto_div = (__pypto_lhs - __pypto_mod) / __pypto_rhs;"
         << "if (__pypto_mod != 0.0f && ((__pypto_rhs < 0.0f) != (__pypto_mod < 0.0f))) {"
         << "__pypto_div -= 1.0f;"
         << "}"
         << "float __pypto_floor_div;"
         << "if (__pypto_div != 0.0f) {"
         << "__pypto_floor_div = " << BuildFP32FloorExpression("__pypto_div") << ";"
         << "if (__pypto_div - __pypto_floor_div > 0.5f) { __pypto_floor_div += 1.0f; }"
         << "} else {"
         << "uint32_t __pypto_zero = (reinterpret_cast<uint32_t&>(__pypto_lhs) ^ "
            "reinterpret_cast<uint32_t&>(__pypto_rhs)) & 0x80000000u;"
         << "__pypto_floor_div = reinterpret_cast<float&>(__pypto_zero);"
         << "}"
         << "(" << cppType << ")__pypto_floor_div;})";
    return code.str();
}

std::string BuildPythonDivModExpression(const std::string& left, const std::string& right, const DataType& dtype,
                                        bool returnRemainder)
{
    if (dtype.IsFloat()) {
        return BuildPythonFloatDivModExpression(left, right, dtype, returnRemainder);
    }
    return BuildPythonIntegerDivModExpression(left, right, dtype, returnRemainder);
}

std::string BuildBFloat16MinMaxExpression(const std::string& left, const std::string& right, const std::string& op)
{
    // BF16 occupies the upper 16 bits of FP32. Keep the expanded bit patterns in scalar registers:
    // bisheng can spill integer/FP32 bitcasts around FP32 arithmetic, including __builtin_bit_cast.
    // The "l" constraint selects 64-bit scalar registers; MIN/MAX.f32 consumes their low 32 bits.
    std::ostringstream code;
    code << "({bfloat16_t __pypto_lhs = " << left << ";"
         << "bfloat16_t __pypto_rhs = " << right << ";"
         << "uint64_t __pypto_lhs_bits = (uint64_t)reinterpret_cast<uint16_t&>(__pypto_lhs) << 16;"
         << "uint64_t __pypto_rhs_bits = (uint64_t)reinterpret_cast<uint16_t&>(__pypto_rhs) << 16;"
         << "uint64_t __pypto_result;"
         << "asm(\"" << op << ".f32 %0, %1, %2\" : \"=l\"(__pypto_result) : "
         << "\"l\"(__pypto_lhs_bits), \"l\"(__pypto_rhs_bits));"
         << "uint16_t __pypto_result_bits = __pypto_result >> 16;"
         << "reinterpret_cast<bfloat16_t&>(__pypto_result_bits);})";
    return code.str();
}

} // namespace

std::string CCECodegen::ScalarCast(const std::string& value, const DataType& source, const DataType& target)
{
    if (source == target) {
        return value;
    }
    const std::string cppType = target.ToCTypeString();
    if (!IsInSimtContext() && (source == DataType::BF16 || target == DataType::BF16)) {
        needsScalarHelpers_ = true;
        return "pypto_scalar::Cast<" + cppType + ">(" + value + ")";
    }
    return "((" + cppType + ")" + value + ")";
}

#define IMPLEMENT_BINARY_OP(OpType, OpName, CppOp)                                                   \
    void CCECodegen::VisitExpr_(const ir::OpType##Ptr& op)                                           \
    {                                                                                                \
        PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op != nullptr) \
            << "Internal error: null " << (OpName);                                                  \
        std::string left = GetExprAsCode(op->left_);                                                 \
        std::string right = GetExprAsCode(op->right_);                                               \
        currentExprValue_ = "(" + left + " " + (CppOp) + " " + right + ")";                          \
    }

// Arithmetic operators
IMPLEMENT_BINARY_OP(Add, "Add", "+")
IMPLEMENT_BINARY_OP(Sub, "Sub", "-")
IMPLEMENT_BINARY_OP(Mul, "Mul", "*")
IMPLEMENT_BINARY_OP(FloatDiv, "FloatDiv", "/")

// Comparison operators
IMPLEMENT_BINARY_OP(Eq, "Eq", "==")
IMPLEMENT_BINARY_OP(Ne, "Ne", "!=")
IMPLEMENT_BINARY_OP(Lt, "Lt", "<")
IMPLEMENT_BINARY_OP(Le, "Le", "<=")
IMPLEMENT_BINARY_OP(Gt, "Gt", ">")
IMPLEMENT_BINARY_OP(Ge, "Ge", ">=")

// Logical operators
IMPLEMENT_BINARY_OP(And, "And", "&&")
IMPLEMENT_BINARY_OP(Or, "Or", "||")

// Bitwise operators
IMPLEMENT_BINARY_OP(BitAnd, "BitAnd", "&")
IMPLEMENT_BINARY_OP(BitOr, "BitOr", "|")
IMPLEMENT_BINARY_OP(BitXor, "BitXor", "^")
IMPLEMENT_BINARY_OP(BitShiftLeft, "BitShiftLeft", "<<")
IMPLEMENT_BINARY_OP(BitShiftRight, "BitShiftRight", ">>")

#undef IMPLEMENT_BINARY_OP

void CCECodegen::VisitExpr_(const ir::XorPtr& op)
{
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op != nullptr)
        << "Internal error: null Xor";
    const std::string left = GetExprAsCode(op->left_);
    const std::string right = GetExprAsCode(op->right_);
    currentExprValue_ = "((bool)(" + left + ") != (bool)(" + right + "))";
}

// C++ signed division truncates toward zero, and C++ has no floating-point modulo operator.
// Lower both integer and floating-point FloorDiv/FloorMod explicitly to match Python semantics.
void CCECodegen::VisitExpr_(const ir::FloorDivPtr& op)
{
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op != nullptr)
        << "Internal error: null FloorDiv";
    auto scalarType = ir::As<ir::ScalarType>(op->GetType());
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, scalarType != nullptr)
        << "Internal error: FloorDiv result must be a scalar";
    const std::string left = GetExprAsCode(op->left_);
    const std::string right = GetExprAsCode(op->right_);
    currentExprValue_ = BuildPythonDivModExpression(left, right, scalarType->dtype_, false);
}

void CCECodegen::VisitExpr_(const ir::FloorModPtr& op)
{
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op != nullptr)
        << "Internal error: null FloorMod";
    auto scalarType = ir::As<ir::ScalarType>(op->GetType());
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, scalarType != nullptr)
        << "Internal error: FloorMod result must be a scalar";
    const std::string left = GetExprAsCode(op->left_);
    const std::string right = GetExprAsCode(op->right_);
    currentExprValue_ = BuildPythonDivModExpression(left, right, scalarType->dtype_, true);
}

// Special binary operators (function calls)
void CCECodegen::VisitExpr_(const ir::MinPtr& op)
{
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op != nullptr)
        << "Internal error: null Min";
    auto scalarType = std::dynamic_pointer_cast<const ir::ScalarType>(op->GetType());
    std::string cppType = scalarType ? scalarType->dtype_.ToCTypeString() : "int64_t";
    VisitExpr(op->left_);
    std::string left = currentExprValue_;
    VisitExpr(op->right_);
    std::string right = currentExprValue_;
    if (scalarType && scalarType->dtype_ == DataType::BF16) {
        currentExprValue_ = IsInSimtContext() ? "__min(" + left + ", " + right + ")" :
                                                BuildBFloat16MinMaxExpression(left, right, "min");
        return;
    }
    currentExprValue_ = "min((" + cppType + ")(" + left + "), (" + cppType + ")(" + right + "))";
}

void CCECodegen::VisitExpr_(const ir::MaxPtr& op)
{
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op != nullptr)
        << "Internal error: null Max";
    auto scalarType = std::dynamic_pointer_cast<const ir::ScalarType>(op->GetType());
    std::string cppType = scalarType ? scalarType->dtype_.ToCTypeString() : "int64_t";
    VisitExpr(op->left_);
    std::string left = currentExprValue_;
    VisitExpr(op->right_);
    std::string right = currentExprValue_;
    if (scalarType && scalarType->dtype_ == DataType::BF16) {
        currentExprValue_ = IsInSimtContext() ? "__max(" + left + ", " + right + ")" :
                                                BuildBFloat16MinMaxExpression(left, right, "max");
        return;
    }
    currentExprValue_ = "max((" + cppType + ")(" + left + "), (" + cppType + ")(" + right + "))";
}

void CCECodegen::VisitExpr_(const ir::PowPtr& op)
{
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op != nullptr)
        << "Internal error: null Pow";
    VisitExpr(op->left_);
    std::string left = currentExprValue_;
    VisitExpr(op->right_);
    std::string right = currentExprValue_;
    currentExprValue_ = "pow(" + left + ", " + right + ")";
}

// ---- Unary Operators ----

#define IMPLEMENT_UNARY_OP(OpType, OpName, CppOp)                                                    \
    void CCECodegen::VisitExpr_(const ir::OpType##Ptr& op)                                           \
    {                                                                                                \
        PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op != nullptr) \
            << "Internal error: null " << (OpName);                                                  \
        const std::string operand = GetExprAsCode(op->operand_);                                     \
        currentExprValue_ = std::string("(") + (CppOp) + operand + ")";                              \
    }

IMPLEMENT_UNARY_OP(Not, "Not", "!")
IMPLEMENT_UNARY_OP(BitNot, "BitNot", "~")

#undef IMPLEMENT_UNARY_OP

void CCECodegen::VisitExpr_(const ir::NegPtr& op)
{
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op != nullptr)
        << "Internal error: null Neg";
    VisitExpr(op->operand_);
    if (ir::GetScalarDtype(op->operand_) == DataType::BF16 && !IsInSimtContext()) {
        needsScalarHelpers_ = true;
        currentExprValue_ = "pypto_scalar::ChangeSign<false>(" + currentExprValue_ + ")";
    } else {
        currentExprValue_ = "(-" + currentExprValue_ + ")";
    }
}

// Special unary operators
void CCECodegen::VisitExpr_(const ir::AbsPtr& op)
{
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op != nullptr)
        << "Internal error: null Abs";
    VisitExpr(op->operand_);
    std::string operand = currentExprValue_;
    currentExprValue_ = "abs(" + operand + ")";
}

void CCECodegen::VisitExpr_(const ir::CastPtr& op)
{
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, op != nullptr)
        << "Internal error: null Cast";
    const auto operand = GetExprAsCode(op->operand_);

    const auto scalarType = ir::As<ir::ScalarType>(op->GetType());
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, scalarType != nullptr) << "Cast target must be ScalarType";

    const auto source_type = op->operand_->GetType();
    if (auto source = ir::As<ir::ScalarType>(source_type)) {
        currentExprValue_ = ScalarCast(operand, source->dtype_, scalarType->dtype_);
        return;
    }

    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, ir::As<ir::PtrType>(source_type) != nullptr)
        << "Cast operand must be ScalarType or PtrType";
    // Optional-pointer checks cast the address, not the pointed-to element value.
    currentExprValue_ = "((" + scalarType->dtype_.ToCTypeString() + ")" + operand + ")";
}

// ========================================================================
// End of Expression Visitor Methods
// ========================================================================

int64_t CCECodegen::ExtractConstInt(const ir::ExprPtr& expr) const
{
    auto constInt = std::dynamic_pointer_cast<const ir::ConstInt>(expr);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_VAL, constInt != nullptr) << "Expected constant integer expression";
    return constInt->value_;
}

namespace {

// Access window shape: the tile operand's shape, which is what the access transfers.
std::optional<std::vector<ir::ExprPtr>> ResolveAccessShape(const ir::CallPtr& op,
                                                           const backend::cce::AccessArgIndices& indices)
{
    if (indices.tileArgIdx >= 0 && indices.tileArgIdx < static_cast<int>(op->args_.size())) {
        const auto& tileArg = op->args_[indices.tileArgIdx];
        if (auto tileType = std::dynamic_pointer_cast<const ir::TileType>(tileArg->GetType())) {
            return tileType->shape_;
        }
    }
    return std::nullopt;
}

// A single-column access walks the tensor down a column, which takes Layout::DN just as a
// transposed load does -- but without swapping the strides. The two are therefore distinct
// variants, not one.
bool IsColumnAccess(const ir::CallPtr& op)
{
    auto shape = ResolveAccessShape(op, backend::cce::ResolveAccessArgIndices(op->name_));
    if (!shape || shape->empty()) {
        return false;
    }
    auto last = std::dynamic_pointer_cast<const ir::ConstInt>(shape->back());
    return last != nullptr && last->value_ == 1;
}

/**
 * \brief Helper visitor for collecting tensor definitions from block.load/store
 *
 * Traverses the IR tree to find block.load/block.store calls and records, for each tensor
 * variable *and layout variant* (keyed by the variant's cce name), the access window shape,
 * tile_dims and DN flag. A tensor loaded both row-major and transposed needs one declaration
 * per layout, since Layout::DN is a GlobalTensor template argument.
 */
class TensorDefCollector : public ir::IRVisitor {
    using ir::IRVisitor::VisitExpr_;
    using ir::IRVisitor::VisitStmt_;

public:
    explicit TensorDefCollector(const CodeContext& ctx) : ctx_(ctx) {}

    std::map<std::pair<std::string, std::string>, TensorDef> defs_; ///< (tensor cce name, layout key) -> TensorDef
    std::map<std::string, int> variantCounts_;                      ///< tensor cce name -> layouts seen so far

    void VisitStmt_(const ir::AssignStmtPtr& op) override
    {
        // Keep aliases distinct for SSA/body codegen, but attribute their accesses to the
        // underlying parameter or view whose GlobalTensor declaration owns the base pointer.
        auto source = ir::As<ir::Var>(op->value_);
        if (op->targets_.size() == 1 && ir::As<ir::TensorType>(op->targets_.front()->GetType()) && source &&
            ir::As<ir::TensorType>(source->GetType())) {
            tensorAliases_[op->targets_.front().get()] = ResolveTensorAlias(source);
        }
        ir::IRVisitor::VisitStmt_(op);
    }

    void VisitExpr_(const ir::CallPtr& op) override
    {
        backend::cce::AccessArgIndices indices = backend::cce::ResolveAccessArgIndices(op->name_);
        if (indices.tensorArgIdx >= 0 && static_cast<int>(op->args_.size()) > indices.tensorArgIdx) {
            auto tensorVar = std::dynamic_pointer_cast<const ir::Var>(op->args_[indices.tensorArgIdx]);
            RecordTensorDef(op, ResolveTensorAlias(tensorVar), indices);
        }

        ir::IRVisitor::VisitExpr_(op);
    }

private:
    std::shared_ptr<const ir::Var> ResolveTensorAlias(std::shared_ptr<const ir::Var> var) const
    {
        while (var) {
            auto it = tensorAliases_.find(var.get());
            if (it == tensorAliases_.end()) {
                break;
            }
            var = it->second;
        }
        return var;
    }

    // One def per (tensor, layout variant): accesses that share a variant agree on every field
    // the declaration reads, so the first one to arrive fills it and the rest match by
    // construction. Accesses that disagree land in different variants instead of overwriting
    // each other -- which is what used to silently give one of them the wrong layout.
    void RecordTensorDef(const ir::CallPtr& op, const std::shared_ptr<const ir::Var>& tensorVar,
                         const backend::cce::AccessArgIndices& indices)
    {
        if (!tensorVar) {
            return;
        }
        const std::string baseName = ctx_.SanitizeName(tensorVar);
        auto [defIt, isNewVariant] = defs_.try_emplace({baseName, TensorLayoutVariantKey(op)});
        if (!isNewVariant) {
            return; // an earlier access already declared this layout; every field below matches
        }
        TensorDef& def = defIt->second;
        // The first layout keeps the tensor's plain name, so a tensor accessed one way
        // generates exactly what it did before variants existed.
        const int index = variantCounts_[baseName]++;
        def.cceName = index == 0 ? baseName : baseName + "__v" + std::to_string(index);
        def.var = tensorVar;
        if (auto shape = ResolveAccessShape(op, indices)) {
            def.accessShape = *shape;
        }
        def.layout = backend::cce::MXLoadLayoutName(op);
        if (op->HasKwarg("tile_dims")) {
            def.tileDims = op->GetKwarg<std::vector<int>>("tile_dims");
        }
        def.isTranspose = op->name_ == "block.load" && op->GetKwarg<bool>("is_transpose", false);
        def.isMerged = op->name_ == "block.load" && op->HasKwarg("nd_inner");
    }

    const CodeContext& ctx_; ///< for SanitizeName (pure: derives the cce var-name key)
    std::map<const ir::Var*, std::shared_ptr<const ir::Var>> tensorAliases_;
};

} // namespace

std::string TensorLayoutVariantKey(const ir::CallPtr& op)
{
    // Exactly the inputs GenerateGlobalTensorTypeDeclaration reads: two accesses with the same
    // key produce the same declaration and share it. NZ tensors are always single-variant --
    // their declaration is driven by the fractal layout, not by these kwargs.
    const backend::cce::AccessArgIndices indices = backend::cce::ResolveAccessArgIndices(op->name_);
    if (indices.tensorArgIdx < 0 || static_cast<int>(op->args_.size()) <= indices.tensorArgIdx) {
        return "";
    }
    auto tensorType = ir::As<ir::TensorType>(op->args_[indices.tensorArgIdx]->GetType());
    if (tensorType == nullptr || backend::cce::IsNZTensorType(tensorType)) {
        return "";
    }
    // An MX load names its layout outright, so it keys on that alone.
    const std::string mxLayout = backend::cce::MXLoadLayoutName(op);
    if (!mxLayout.empty()) {
        return mxLayout;
    }

    // Merged loads use a distinct 3D descriptor type. Like ordinary loads, they
    // share declarations only when the source axes and their initialized strides agree.
    const bool isMerged = op->name_ == "block.load" && op->HasKwarg("nd_inner");
    std::string key = isMerged ? "m" : "";
    // A transposed load swaps the row/col strides; a single-column access takes Layout::DN
    // without swapping them, so the two are distinct layouts rather than one.
    if (op->name_ == "block.load" && op->GetKwarg<bool>("is_transpose", false)) {
        key += "t";
    }

    if (IsColumnAccess(op)) {
        key += "c";
    }

    if (op->HasKwarg("tile_dims")) {
        for (int dim : op->GetKwarg<std::vector<int>>("tile_dims")) {
            key += "d" + std::to_string(dim);
        }
    }
    return key;
}

void CCECodegen::CollectTensorDefs(const ir::FunctionPtr& func)
{
    if (!func || !func->body_)
        return;

    TensorDefCollector collector(context_);
    collector.VisitStmt(func->body_);
    tensorDefs_ = std::move(collector.defs_);
}

std::vector<int64_t> CCECodegen::ExtractShapeDimensions(const std::vector<ir::ExprPtr>& shapeExprs) const
{
    std::vector<int64_t> dims;
    dims.reserve(shapeExprs.size());
    for (const auto& expr : shapeExprs) {
        dims.push_back(ExtractConstInt(expr));
    }
    return dims;
}

namespace {

// Map an IR field type to a best-effort C++ type string for struct member declaration.
// Scalar types resolve through DataType::ToCTypeString; an uncovered code is an upstream
// validation gap and fails loudly instead of emitting a mismatched struct definition.
std::string CppTypeForField(const ir::TypePtr& t)
{
    if (auto scalar = ir::As<ir::ScalarType>(t)) {
        std::string ctype = scalar->dtype_.ToCTypeString();
        PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, ctype != "unknown")
            << "Struct field uses unsupported scalar dtype: " << scalar->dtype_.ToString();
        return ctype;
    }
    if (ir::As<ir::TileType>(t)) {
        // Tile fields are stored as opaque tile references; v1 emits as auto-deduced.
        return "auto";
    }
    // Nested registered struct -> the field stores another C++ struct; look up its
    // type name. For v1 we fall back to int64_t and rely on CCE flat expansion if the
    // nested tuple type isn't already materialized; full nested support is a follow-up.
    return "int64_t";
}

// Build a C++ `struct Name { <type> <field>; ... };` definition from a struct type name,
// its field names, and per-field IR types. A TupleType field (Array[T, N]) becomes an
// array member `<T> name[N];`.
std::string BuildStructTypeDef(const std::string& name, const std::vector<std::string>& fields,
                               const std::vector<ir::TypePtr>& types, bool requiresVolatile)
{
    std::string def = "class " + name + " {\npublic:\n    ";
    const std::string qualifier = requiresVolatile ? "volatile " : "";
    for (size_t i = 0; i < fields.size(); ++i) {
        const auto& ft = types[i];
        if (auto arr = ir::As<ir::TupleType>(ft)) {
            def += qualifier + CppTypeForField(arr->types_[0]) + " " + fields[i] + "[" +
                   std::to_string(arr->types_.size()) + "]; ";
        } else {
            def += qualifier + CppTypeForField(ft) + " " + fields[i] + "; ";
        }
    }
    def += "\n};";
    return def;
}

std::string StructFieldTypeSignature(const ir::TypePtr& type)
{
    if (auto arrayType = ir::As<ir::TupleType>(type)) {
        PRO_CODEGEN_CHECK(ExternalError::INVALID_VAL, !arrayType->types_.empty())
            << "Struct array field type must not be empty";
        return CppTypeForField(arrayType->types_[0]) + "[" + std::to_string(arrayType->types_.size()) + "]";
    }
    return CppTypeForField(type);
}

} // namespace

std::string CCECodegen::GetStructFieldTypeString(const ir::TypePtr& type) const { return CppTypeForField(type); }

std::string CCECodegen::GenerateTilingHeader(const std::string& typeName, const std::vector<std::string>& fields,
                                             const std::vector<ir::TypePtr>& types, bool requiresVolatile)
{
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, fields.size() == types.size())
        << "Tiling struct field count mismatch for type '" << typeName << "'";
    return "#pragma once\n" + BuildStructTypeDef(typeName, fields, types, requiresVolatile) + "\n";
}

void CCECodegen::RegisterStructDefinition(const ir::TupleTypePtr& tupleType, const std::string& typeName,
                                          const std::vector<std::string>& fields, bool isTiling)
{
    PRO_CODEGEN_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, tupleType != nullptr)
        << "Cannot register a null TupleType for struct '" << typeName << "'";
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, fields.size() == tupleType->types_.size())
        << "Struct field count mismatch for type '" << typeName << "'";

    auto [it, inserted] = structDefinitions_.try_emplace(
        typeName, StructDefinition{typeName, fields, tupleType->types_, isTiling, false});
    if (!inserted) {
        const auto& existing = it->second;
        PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, existing.isTiling == isTiling)
            << "Struct type '" << typeName << "' cannot be both tiling and non-tiling";
        PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION,
                          existing.fields == fields && existing.types.size() == tupleType->types_.size())
            << "Conflicting definitions for struct type '" << typeName << "'";
        for (size_t i = 0; i < existing.types.size(); ++i) {
            PRO_CODEGEN_CHECK(ExternalError::INVALID_OPERATION, StructFieldTypeSignature(existing.types[i]) ==
                                                                    StructFieldTypeSignature(tupleType->types_[i]))
                << "Conflicting type for field '" << fields[i] << "' in struct '" << typeName << "'";
        }
    }
}

const ir::TupleTypeInfo* CCECodegen::GetTupleTypeInfo(const ir::TupleTypePtr& tupleType) const
{
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, debugInfo_ != nullptr)
        << "CCE codegen requires IRDebugInfo";
    return tupleType != nullptr ? debugInfo_->GetTupleTypeInfo(tupleType.get()) : nullptr;
}

const std::string* CCECodegen::GetStructName(const ir::TupleTypePtr& tupleType) const
{
    const ir::TupleTypeInfo* info = GetTupleTypeInfo(tupleType);
    if (info == nullptr || info->kind != ir::TupleTypeKind::STRUCT) {
        return nullptr;
    }
    PRO_CODEGEN_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, info->name.has_value())
        << "Struct TupleTypeInfo has no type name";
    return &info->name.value();
}

void CCECodegen::MarkStructVolatile(const ir::TypePtr& type)
{
    auto tupleType = ir::As<ir::TupleType>(type);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_TYPE, tupleType != nullptr)
        << "ssbuf_load/store requires a struct TupleType argument";
    const std::string* structName = GetStructName(tupleType);
    PRO_CODEGEN_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, structName != nullptr)
        << "ssbuf_load/store struct type is not registered for CCE codegen";
    structDefinitions_.at(*structName).requiresVolatile = true;
}

void CCECodegen::RegisterTilingStructTypes(const ir::FunctionPtr& func)
{
    for (const auto& param : func->params_) {
        auto tupleType = ir::As<ir::TupleType>(param->GetType());
        if (tupleType == nullptr)
            continue;
        const ir::TupleTypeInfo* info = GetTupleTypeInfo(tupleType);
        PRO_CODEGEN_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR,
                          info != nullptr && info->kind == ir::TupleTypeKind::STRUCT)
            << "Tiling param '" << param->name_ << "' has no struct TupleTypeInfo in IRDebugInfo";
        PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, info->fields.size() == tupleType->types_.size())
            << "Tiling struct field count mismatch for param '" << param->name_ << "'";
        PRO_CODEGEN_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, info->name.has_value())
            << "Tiling struct param '" << param->name_ << "' has no type name";
        RegisterStructDefinition(tupleType, info->name.value(), info->fields, true);
    }
}

void CCECodegen::EmitStructTypes()
{
    for (const auto& [typeName, info] : structDefinitions_) {
        if (!info.isTiling) {
            emitter_.EmitLine(BuildStructTypeDef(typeName, info.fields, info.types, info.requiresVolatile));
            continue;
        }
        const std::string headerName = typeName + "_tiling.h";
        tilingHeaders_[headerName] = GenerateTilingHeader(typeName, info.fields, info.types, info.requiresVolatile);
        emitter_.EmitLine("#include \"" + headerName + "\"");
    }
}

void CCECodegen::EmitTilingStructCopy(const ir::FunctionPtr& func)
{
    for (const auto& param : func->params_) {
        auto tupleType = ir::As<ir::TupleType>(param->GetType());
        if (tupleType == nullptr)
            continue;
        std::string name = context_.GetVarName(param);
        const std::string* registeredName = GetStructName(tupleType);
        PRO_CODEGEN_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, registeredName != nullptr)
            << "Tiling param '" << param->name_ << "' has no struct TupleTypeInfo in IRDebugInfo";
        const std::string& structName = *registeredName;
        // The static compiler force-includes the Host Tiling result and defines CONST_TILING;
        // keep a writable local copy because that entry passes nullptr for tiling_ptr. Emit the
        // block unconditionally so JIT and binary recompilation share one source text.
        emitter_.EmitLine("#ifdef CONST_TILING");
        emitter_.EmitLine("GET_TILING_DATA_WITH_STRUCT(" + structName + ", " + name + "_constant, " + name + "_ptr);");
        emitter_.EmitLine("PYPTO_ACLNN_CHECK_TILING_SIZE(" + structName + ", " + name + "_constant);");
        emitter_.EmitLine(structName + " " + name + " = " + name + "_constant;");
        emitter_.EmitLine("#else");
        emitter_.EmitLine("constexpr uint32_t " + name + "_all_bytes = sizeof(" + structName + ");");
        emitter_.EmitLine(structName + " " + name + ";");
        auto emitCubeCopy = [&]() {
            emitter_.EmitLine("copy_data_align64((uint8_t*)&" + name + ", (__gm__ uint8_t *)" + name + "_ptr, " + name +
                              "_all_bytes);");
        };
        auto emitVectorCopy = [&]() {
            emitter_.EmitLine("__ubuf__ uint8_t *" + name + "_in_ub = (__ubuf__ uint8_t *)get_imm(0);");
            emitter_.EmitLine("constexpr uint32_t " + name + "_len_burst = (" + name + "_all_bytes + 31) / 32;");
            if (GetArch() == npu::tile_fwk::NPUArch::DAV_9201) {
                emitter_.EmitLine("copy_gm_to_ubuf_align_v2((__ubuf__ uint8_t *)" + name +
                                  "_in_ub, (__gm__ uint8_t *)" + name + "_ptr, 0, 1, " + name +
                                  "_len_burst * 32, 0, 0, false, false, 0, 0, 0, false);");
            } else {
                emitter_.EmitLine("copy_gm_to_ubuf_align_v2((__ubuf__ uint8_t *)" + name +
                                  "_in_ub, (__gm__ uint8_t *)" + name + "_ptr, 0, 1, " + name +
                                  "_len_burst * 32, 0, 0, false, 0, 0, 0);");
            }
            emitter_.EmitLine("set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);");
            emitter_.EmitLine("wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);");
            emitter_.EmitLine("copy_data_align64((uint8_t*)&" + name + ", (__ubuf__ uint8_t *)" + name + "_in_ub, " +
                              name + "_all_bytes);");
        };
        if (target_ == ir::SectionKind::Cube) {
            emitCubeCopy();
        } else {
            emitVectorCopy();
        }
        emitter_.EmitLine("#endif");
        emitter_.EmitLine("");
    }
}

// ========================================================================
// Phase 7 helpers: Stride type generation and GlobalTensor instance emission
// ========================================================================

std::vector<std::string> CCECodegen::BuildTensorStrideExpressions(const ir::TensorTypePtr& tensorType)
{
    PRO_CODEGEN_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, tensorType != nullptr)
        << "Cannot build stride expressions for a null tensor type";
    const size_t ndim = tensorType->shape_.size();

    if (tensorType->tensor_view_.has_value() && !tensorType->tensor_view_->stride.empty()) {
        const auto& viewStride = tensorType->tensor_view_->stride;
        PRO_CODEGEN_CHECK(ExternalError::INVALID_SHAPE, viewStride.size() == ndim)
            << "TensorView stride rank (" << viewStride.size() << ") != tensor rank (" << ndim << ")";

        std::vector<std::string> strideExprs;
        strideExprs.reserve(ndim);
        for (const auto& stride : viewStride) {
            strideExprs.push_back(GetExprAsCode(stride));
        }
        return strideExprs;
    }

    std::vector<std::string> strideExprs(ndim, "1");
    for (size_t i = 0; i < ndim; ++i) {
        std::ostringstream stride;
        for (size_t j = i + 1; j < ndim; ++j) {
            if (j > i + 1) {
                stride << " * ";
            }
            stride << GetExprAsCode(tensorType->shape_[j]);
        }
        if (i + 1 < ndim) {
            strideExprs[i] = stride.str();
        }
    }
    return strideExprs;
}

std::string CCECodegen::BuildAccessStrideArgs(const ir::TensorTypePtr& tensorType,
                                              const std::optional<std::vector<int>>& tileDims, bool isTranspose,
                                              bool isMx, bool isMerged)
{
    auto strideExprs = BuildTensorStrideExpressions(tensorType);
    if (isMerged) {
        INTERNAL_CHECK(tileDims.has_value() && tileDims->size() == 3)
            << "Internal error: merged GlobalTensor requires three access axes";
        if (!tensorType->tensor_view_.has_value() || tensorType->tensor_view_->stride.empty()) {
            // Promote runtime dimensions before multiplying, as in the merged source offset.
            for (auto& stride : strideExprs) {
                stride = "uint64_t{1} * " + stride;
            }
        }
        const auto& matrixStride = strideExprs[static_cast<size_t>((*tileDims)[0])];
        const auto& rowStride = strideExprs[static_cast<size_t>((*tileDims)[1])];
        return "1, 1, " + matrixStride + (isTranspose ? ", 1, " + rowStride : ", " + rowStride + ", 1");
    }
    const size_t ndim = tensorType->shape_.size();
    std::string rowStrideExpr;
    std::string colStrideExpr;
    if (tileDims.has_value() && tileDims->size() == 2) {
        rowStrideExpr = strideExprs[static_cast<size_t>((*tileDims)[0])];
        colStrideExpr = strideExprs[static_cast<size_t>((*tileDims)[1])];
    } else if (ndim >= 2) {
        rowStrideExpr = strideExprs[ndim - 2];
        colStrideExpr = strideExprs[ndim - 1];
    } else {
        rowStrideExpr = strideExprs[0];
        colStrideExpr = "1";
    }
    // Same two strides throughout; the layouts differ only in which Stride slots carry them.
    // An MX shape puts the matrix axes in DIM_2/DIM_3 and pins DIM_4 to the fractal length,
    // every other layout leaves DIM_2 at 1 and carries them in DIM_3/DIM_4.
    if (isMx) {
        return "1, 1, " + rowStrideExpr + ", " + colStrideExpr + ", 1";
    }
    // A transposed access walks the same memory with the row and column strides exchanged.
    return isTranspose ? "1, 1, 1, " + colStrideExpr + ", " + rowStrideExpr :
                         "1, 1, 1, " + rowStrideExpr + ", " + colStrideExpr;
}

std::string CCECodegen::BindGlobalTensor(const ir::VarPtr& tensorVar, const ir::CallPtr& op,
                                         const std::string& pointerExpr, const std::string& validRows,
                                         const std::string& validCols, const std::string& matrixCount)
{
    // The declaration for this access's layout. An access the prescan never saw has no
    // declaration of its own and falls back to the tensor's plain name.
    auto it = tensorDefs_.find({GetVarName(tensorVar), TensorLayoutVariantKey(op)});
    const std::string declName = it != tensorDefs_.end() ? it->second.cceName : GetVarName(tensorVar);
    const bool isMerged = op->name_ == "block.load" && op->HasKwarg("nd_inner");
    INTERNAL_CHECK(isMerged == !matrixCount.empty())
        << "Internal error: matrix count is required only for merged GlobalTensor accesses";
    INTERNAL_CHECK(!isMerged || (!validRows.empty() && !validCols.empty()))
        << "Internal error: merged GlobalTensor binding requires descriptor rows and columns";
    if (!validRows.empty() && !validCols.empty()) {
        const auto tensorType = ir::As<ir::TensorType>(tensorVar->GetType());
        PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, tensorType != nullptr)
            << "Internal error: GlobalTensor binding requires a TensorType";
        const bool isNz = backend::cce::IsNZTensorType(tensorType);
        const bool isMx = it != tensorDefs_.end() && !it->second.layout.empty();
        const std::string outer = isNz ? "pto::GlobalTensorDim::DIM_1" :
                                         (isMx ? "pto::GlobalTensorDim::DIM_2" : "pto::GlobalTensorDim::DIM_3");
        const std::string inner = isNz ? "pto::GlobalTensorDim::DIM_2" :
                                         (isMx ? "pto::GlobalTensorDim::DIM_3" : "pto::GlobalTensorDim::DIM_4");

        // Layout helpers own the logical-to-physical mapping. Build a
        // temporary shape and read the two dimensions updated by SetShape
        // instead of duplicating that mapping here.
        std::string args = validRows + ", " + validCols;
        if (isNz || isMx) {
            const std::string shape = declName + "_shape_" + std::to_string(GetTileOffsetCounter());
            emitter_.EmitLine("const " + declName + "ShapeDim5 " + shape + "(" + args + ");");
            args = shape + ".shape[" + outer + "], " + shape + ".shape[" + inner + "]";
        }
        std::string dims = outer + ", " + inner;
        if (isMerged) {
            dims = "pto::GlobalTensorDim::DIM_2, " + dims;
            args = matrixCount + ", " + args;
        }
        emitter_.EmitLine(declName + ".SetShape<" + dims + ">(" + args + ");");
    }
    emitter_.EmitLine("TASSIGN(" + declName + ", " + pointerExpr + ");");
    return declName;
}

void CCECodegen::EmitGlobalTensorInstance(const std::string& instanceName, const std::string& declName,
                                          const std::string& pointerExpr, const std::string& shapeArgs,
                                          const std::string& strideArgs)
{
    std::ostringstream instance;
    instance << declName << "Type " << instanceName << "(" << pointerExpr;
    // Declarations with runtime shape or stride arguments construct those metadata objects here.
    if (!shapeArgs.empty() || !strideArgs.empty()) {
        instance << ", " << declName << "ShapeDim5(" << shapeArgs << "), " << declName << "StrideDim5(" << strideArgs
                 << ")";
    }
    instance << ");";
    emitter_.EmitLine(instance.str());
}

bool CCECodegen::IsDNAccessLayout(const std::vector<int64_t>& shapeDims, bool isTranspose)
{
    return *shapeDims.rbegin() == 1 || isTranspose;
}

void CCECodegen::GenerateGlobalTensorTypeDeclaration(const TensorDef& def)
{
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, def.var != nullptr)
        << "Internal error: TensorDef.var is null";
    const std::string& varName = def.cceName;
    // The base name owns the pointer, which every layout variant of the tensor shares;
    // var_name carries this variant's declaration, so one tensor read both row-major and
    // transposed gets one declaration of each.
    const std::string baseName = context_.SanitizeName(def.var);
    auto tensorType = ir::As<ir::TensorType>(def.var->GetType());
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, tensorType != nullptr)
        << "Internal error: TensorDef.var is not a tensor";
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, HasPointer(baseName))
        << "Internal error: tensor '" << baseName << "' has no base pointer";

    const std::string basePointer = GetPointer(baseName);
    std::string elementType = tensorType->dtype_.ToCTypeString();
    std::string shapeTypeName = varName + "ShapeDim5";
    std::string strideTypeName = varName + "StrideDim5";
    std::string globalTypeName = varName + "Type";

    const bool isNz = backend::cce::IsNZTensorType(tensorType);
    std::string shapeType;
    std::string layoutArg;
    std::string strideType;
    std::string strideArgs;
    if (def.isMerged) {
        // Keep Shape2 dynamic even for a single matrix: PTO's multi-matrix path
        // respects the runtime valid extent, including a partial final matrix.
        shapeType = "pto::Shape<1, 1, -1, -1, -1>";
        layoutArg = def.isTranspose ? "Layout::DN" : "Layout::ND";
        strideType = std::string("pto::Stride<1, 1, -1, ") + (def.isTranspose ? "1, -1>" : "-1, 1>");
        // Source strides do not depend on the valid window. PTO only advances the
        // matrix stride when loading a subsequent matrix, including a final tail.
        strideArgs = BuildAccessStrideArgs(tensorType, def.tileDims, def.isTranspose, false, true);
    } else if (isNz) {
        const size_t rowAxis = tensorType->shape_.size() - 2;
        const size_t colAxis = tensorType->shape_.size() - 1;
        const int64_t c0 = backend::cce::GetNZInnerCols(tensorType->dtype_);
        layoutArg = "Layout::NZ";
        const std::string paddedRows = ComputeAlignedShapeDimension(tensorType->shape_[rowAxis], 16);
        const std::string paddedCols = ComputeAlignedShapeDimension(tensorType->shape_[colAxis], c0);
        strideType = "pto::BaseShape2D<" + elementType + ", pto::DYNAMIC, pto::DYNAMIC, Layout::NZ>";
        strideArgs = paddedRows + ", " + paddedCols;
    } else {
        // Every non-NZ declaration is an access variant whose shape and order determine
        // its ND/DN or MX layout and the two tensor strides it walks.
        PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, !def.accessShape.empty())
            << "Internal error: tensor '" << varName << "' has no access_shape";
        const bool isTranspose = def.isTranspose;
        const bool isMx = !def.layout.empty();
        const std::vector<int64_t> shapeDims = ExtractShapeDimensions(def.accessShape);
        layoutArg = isMx ? def.layout : (IsDNAccessLayout(shapeDims, isTranspose) ? "Layout::DN" : "Layout::ND");
        strideType = isMx ? "pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>" :
                            "pto::Stride<-1, -1, -1, -1, -1>";
        strideArgs = BuildAccessStrideArgs(tensorType, def.tileDims, isTranspose, isMx);
    }

    if (shapeType.empty()) {
        shapeType = "pto::TileShape2D<" + elementType + ", pto::DYNAMIC, pto::DYNAMIC, " + layoutArg + ">";
    }
    emitter_.EmitLine("using " + shapeTypeName + " = " + shapeType + ";");
    emitter_.EmitLine("using " + strideTypeName + " = " + strideType + ";");
    emitter_.EmitLine("using " + globalTypeName + " = GlobalTensor<" + elementType + ", " + shapeTypeName + ", " +
                      strideTypeName + ", " + layoutArg + ">;");

    EmitGlobalTensorInstance(varName, varName, basePointer, "", strideArgs);
}

std::string CCECodegen::DeclareFlatGlobalTensor(const ir::VarPtr& tensorVar, const std::string& numelExpr)
{
    const std::string baseName = context_.SanitizeName(tensorVar);
    // Fresh declaration per call under a unique name (no dedup bookkeeping): the caller
    // (block.init_output) is its sole consumer and addresses GM through flat element
    // offsets, so the declaration is simply a [1, numel] ND view over the tensor --
    // shape and stride both carry the element-count product, which constant-folds when
    // every dim is static.
    const std::string varName = baseName + "__io" + std::to_string(flatTensorDeclCount_++);
    auto tensorType = ir::As<ir::TensorType>(tensorVar->GetType());
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, tensorType != nullptr)
        << "Internal error: DeclareFlatGlobalTensor requires a TensorType";
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, HasPointer(baseName))
        << "Internal error: tensor '" << baseName << "' has no base pointer";
    const std::string elementType = tensorType->dtype_.ToCTypeString();
    const std::string shapeType = varName + "ShapeDim5";
    const std::string strideType = varName + "StrideDim5";
    const std::string globalType = varName + "Type";
    emitter_.EmitLine("using " + shapeType + " = pto::TileShape2D<" + elementType +
                      ", pto::DYNAMIC, pto::DYNAMIC, Layout::ND>;");
    emitter_.EmitLine("using " + strideType + " = pto::Stride<-1, -1, -1, -1, -1>;");
    emitter_.EmitLine("using " + globalType + " = GlobalTensor<" + elementType + ", " + shapeType + ", " + strideType +
                      ", Layout::ND>;");
    emitter_.EmitLine(globalType + " " + varName + "(" + GetPointer(baseName) + ", " + shapeType + "(1, " + numelExpr +
                      "), " + strideType + "(1, 1, 1, " + numelExpr + ", 1));");
    emitter_.EmitLine("");
    return varName;
}

} // namespace codegen

} // namespace pypto
