/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "passes/pass_utils/token_utils.h"
#include "interface/tensor/irbuilder.h"
#include "passes/pass_log/pass_log.h"

#include <algorithm>
#include <unordered_map>
#include <vector>

#define MODULE_NAME "TokenUtils"

namespace npu::tile_fwk {

Status TokenUtils::RebuildTokenDependencies(Function& function)
{
    auto& varDependency = function.GetVarDependency();
    varDependency.Clear();
    for (const auto& op : function.operations_) {
        auto stmt = std::static_pointer_cast<const ir::Stmt>(op);
        for (const auto& token : op->result_token_) {
            if (token != nullptr) {
                varDependency.AddProducer(token, stmt);
            }
        }
        for (const auto& token : op->tokens_) {
            if (token != nullptr) {
                varDependency.AddConsumer(token, stmt);
            }
        }
    }
    return SUCCESS;
}

Status TokenUtils::SplitMultiProducerTokens(Function& function)
{
    using OpPtr = std::shared_ptr<Operation>;
    std::unordered_map<ir::VarPtr, std::vector<OpPtr>> tokenProducers;
    std::unordered_map<ir::VarPtr, std::vector<OpPtr>> tokenConsumers;

    for (const auto& op : function.operations_) {
        for (const auto& token : op->result_token_) {
            if (token != nullptr) {
                tokenProducers[token].push_back(op);
            }
        }
        for (const auto& token : op->tokens_) {
            if (token != nullptr) {
                tokenConsumers[token].push_back(op);
            }
        }
    }

    bool anySplit = false;

    for (auto& [token, producers] : tokenProducers) {
        if (producers.size() <= 1) {
            continue;
        }

        anySplit = true;
        APASS_LOG_DEBUG_F(Elements::Operation, "Token %s has %zu producers, splitting.", token->name_.c_str(),
                          producers.size());

        for (size_t i = 1; i < producers.size(); i++) {
            auto& producer = producers[i];
            auto newToken = IRBuilder().CreateTokenVar(producer->GetSpan());
            std::replace(producer->result_token_.begin(), producer->result_token_.end(), token, newToken);
            auto consumerIt = tokenConsumers.find(token);
            if (consumerIt != tokenConsumers.end()) {
                for (auto& consumer : consumerIt->second) {
                    consumer->tokens_.push_back(newToken);
                }
            }
        }
    }

    if (RebuildTokenDependencies(function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Function, "Failed to rebuild token dependencies for function %s.",
                          function.GetRawName().c_str());
        return FAILED;
    }

    if (anySplit) {
        APASS_LOG_INFO_F(Elements::Operation, "SplitMultiProducerTokens completed for function %s.",
                         function.GetRawName().c_str());
    }

    return SUCCESS;
}

Status TokenUtils::RemoveDanglingTokens(Function& function)
{
    struct TokenDependency {
        std::vector<Operation*> producers;
        std::vector<Operation*> consumers;
    };
    std::unordered_map<ir::VarPtr, TokenDependency> tokenDependencies;
    for (const auto& op : function.operations_) {
        if (op->IsDeleted()) {
            continue;
        }
        for (const auto& token : op->result_token_) {
            if (token != nullptr) {
                tokenDependencies[token].producers.push_back(op.get());
            }
        }
        for (const auto& token : op->tokens_) {
            if (token != nullptr) {
                tokenDependencies[token].consumers.push_back(op.get());
            }
        }
    }

    size_t danglingNum = 0;
    for (const auto& [token, dependency] : tokenDependencies) {
        // A token is dangling when either side of its dependency is missing.
        if (!dependency.producers.empty() && !dependency.consumers.empty()) {
            continue;
        }
        ++danglingNum;
        for (auto* producer : dependency.producers) {
            producer->result_token_.erase(
                std::remove(producer->result_token_.begin(), producer->result_token_.end(), token),
                producer->result_token_.end());
        }
        for (auto* consumer : dependency.consumers) {
            consumer->tokens_.erase(std::remove(consumer->tokens_.begin(), consumer->tokens_.end(), token),
                                    consumer->tokens_.end());
        }
    }
    function.RefreshVarDependency();

    APASS_LOG_INFO_F(Elements::Operation, "RemoveDanglingTokens: dropped %zu dangling token(s) on function [%s].",
                     danglingNum, function.GetRawName().c_str());

    return SUCCESS;
}

} // namespace npu::tile_fwk
