/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ir/transforms/passes.h"
#include <algorithm>
#include <array>
#include "ir/transforms/base/mutator.h"
#include "interface/configs/config_manager_ng.h"
#include "interface/operation/operation.h"
#include "interface/tensor/ir_tensor_op_rebuild.h"

namespace pypto::ir {
namespace {
using npu::tile_fwk::ConfigManagerNg;
using npu::tile_fwk::ConfigScope;
using npu::tile_fwk::ConfigScopePtr;
using npu::tile_fwk::Operation;

constexpr std::array<const char*, 8> kOpConfigKeys = {
    "vec_tile_shapes",  "cube_tile_shapes", "conv_tile_shapes",  "convbp_tile_shapes",
    "dist_tile_shapes", "matrix_size",      "pass.sg_set_scope", "pass.sg_set_atomic_scope"};

// Text loading represents ordinary lists as vector<any>; configuration scopes use typed lists.
std::any ConfigValue(const std::any& value)
{
    const auto* list = std::any_cast<std::vector<std::any>>(&value);
    if (!list) {
        return value;
    }
    std::vector<int64_t> ints;
    for (const auto& element : *list) {
        const auto* number = std::any_cast<int64_t>(&element);
        if (!number) {
            return value;
        }
        ints.push_back(*number);
    }
    return ints;
}

TensorOpStmtPtr ApplyOpConfig(const TensorOpStmtPtr& stmt, const ConfigScopePtr& scope)
{
    auto result = std::const_pointer_cast<TensorOpStmt>(As<TensorOpStmt>(npu::tile_fwk::RebuildTensorOpStmt(
        stmt, stmt->result_, stmt->result_token_, stmt->args_, stmt->tokens_, stmt->span_)));
    // Keep effective values on generic IR too, including statements loaded from text.
    for (const auto* key : kOpConfigKeys) {
        if (!scope->HasConfig(key)) {
            continue;
        }
        const auto& value = scope->GetAnyConfig(key);
        auto it = std::find_if(result->attrs_.begin(), result->attrs_.end(),
                               [&](const auto& attr) { return attr.first == key; });
        if (it == result->attrs_.end()) {
            result->attrs_.emplace_back(key, value);
        } else {
            it->second = value;
        }
    }
    if (auto op = std::dynamic_pointer_cast<Operation>(result)) {
        op->UpdateTileShape(scope->GenerateTileShape());
        if (scope->HasConfig("pass.sg_set_scope")) {
            auto values = scope->GetPassConfig<std::vector<int64_t>>(npu::tile_fwk::SG_SET_SCOPE);
            if (values.size() == 3) {
                op->SetScopeInfo(Operation::ScopeInfo::FromConfig(values));
            } else if (values.size() == 1) {
                op->SetScopeId(static_cast<int>(values[0]));
            }
        }
        if (scope->HasConfig("pass.sg_set_atomic_scope")) {
            auto values = scope->GetPassConfig<std::vector<int64_t>>(npu::tile_fwk::SG_SET_ATOMIC_SCOPE);
            op->SetAtomicScopeId(values.empty() ? -1 : static_cast<int>(values[0]));
        }
    }
    return result;
}

class ConfigReplayer : public IRMutator {
public:
    using IRMutator::VisitStmt_;
    explicit ConfigReplayer(ConfigScopePtr scope) : scope_(std::move(scope)) {}

private:
    ConfigScopePtr scope_;

    void ApplyConfig(const TensorOpStmt& op)
    {
        auto next = std::make_shared<ConfigScope>(scope_);
        for (const auto& [key, value] : op.attrs_) {
            next->AddValue(key, ConfigValue(value));
        }
        scope_ = std::move(next);
    }

    void FlushSegment(std::vector<StmtPtr>& segment, std::vector<StmtPtr>& result, const Span& span)
    {
        if (segment.empty()) {
            return;
        }
        // Publish the final scope once per computation segment. Later writes create child scopes.
        result.push_back(std::make_shared<TensorOpStmt>(
            std::vector<VarPtr>{}, nullptr, "OP_CONFIG", std::vector<ExprPtr>{}, std::vector<VarPtr>{},
            std::vector<std::pair<std::string, std::any>>{{"_config_scope", scope_}}, span));
        result.insert(result.end(), segment.begin(), segment.end());
        segment.clear();
    }

    void ScanStatements(const std::vector<StmtPtr>& stmts, std::vector<StmtPtr>& segment, std::vector<StmtPtr>& result,
                        const Span& span)
    {
        for (const auto& stmt : stmts) {
            if (auto tensor = As<TensorOpStmt>(stmt)) {
                if (tensor->opcode_ == "OP_CONFIG") {
                    ApplyConfig(*tensor);
                } else {
                    segment.push_back(ApplyOpConfig(tensor, scope_));
                }
            } else if (auto nested = As<SeqStmts>(stmt)) {
                // Plain sequences introduce neither a config scope nor a segment boundary.
                ScanStatements(nested->stmts_, segment, result, span);
            } else {
                FlushSegment(segment, result, span);
                result.push_back(VisitStmt(stmt));
            }
        }
    }

    StmtPtr VisitStmt_(const SeqStmtsPtr& seq) override
    {
        const auto parentScope = scope_;
        std::vector<StmtPtr> result;
        std::vector<StmtPtr> segment;
        ScanStatements(seq->stmts_, segment, result, seq->span_);
        FlushSegment(segment, result, seq->span_);
        scope_ = parentScope;
        return std::make_shared<SeqStmts>(std::move(result), seq->span_);
    }
};
} // namespace

namespace pass {
Pass ApplyConfigOps()
{
    return CreateFunctionPass(
        [](const FunctionPtr& func) -> FunctionPtr {
            if (func->GetAttr<bool>("_config_applied", false)) {
                return func;
            }
            auto entry = func->GetAttr<ConfigScopePtr>("_config_scope");
            if (!entry) {
                entry = ConfigManagerNg::CurrentScope();
            }
            ConfigReplayer replayer(entry);
            auto body = replayer.VisitStmt(func->body_);
            auto attrs = func->attrs_;
            if (!func->GetAttr<ConfigScopePtr>("_config_scope")) {
                attrs.emplace_back("_config_scope", entry);
            }
            attrs.emplace_back("_config_applied", true);
            return std::make_shared<Function>(func->name_, func->params_, func->returnTypes_, body, func->span_,
                                              func->funcType_, func->entry_, std::move(attrs));
        },
        "ApplyConfigOps");
}
} // namespace pass
} // namespace pypto::ir
