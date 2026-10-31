/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "interface/function/function.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {

/*!
 * \brief Token-related utility functions shared across passes.
 */
class TokenUtils {
public:
    /*!
     * \brief Rebuild Function-level token dependencies from current operations.
     *
     * Operation token fields are the canonical representation while passes are
     * rebuilding or moving operations.  This method discards stale statement
     * pointers in VarDependency and recreates the producer/consumer entries from
     * the operations currently owned by \p function. A consumer-only entry is
     * valid because hidden/leaf functions may consume a token produced by their
     * caller.
     */
    static Status RebuildTokenDependencies(Function& function);

    /*!
     * \brief Split tokens that have multiple producers into one-producer-per-token.
     *
     * After expansion passes (e.g. ExpandFunction), a single token Var may end up
     * in the \c result_token_ list of multiple operations.  This violates the
     * single-producer expectation and makes downstream token analysis unreliable.
     *
     * This method scans every operation in \p function, groups operations by
     * every token in their \c result_token_ list, and for each token that has
     * more than one producer:
     *   - The first producer keeps the original token Var.
     *   - Every subsequent producer receives a freshly-created token Var that
     *     replaces the original token in its \c result_token_ list.
     *   - All original consumers of the old token have the new token appended
     *     to their \c tokens_ list, preserving the dependency on every producer.
     *
     * Both the Operation-level fields (\c result_token_, \c tokens_) and the
     * Function-level \c VarDependency map are kept consistent.
     *
     * \param function  The function whose operations should be processed.
     * \return SUCCESS on completion, FAILED on internal error.
     */
    static Status SplitMultiProducerTokens(Function& function);

    /*!
     * \brief Remove dangling tokens: tokens whose dependency is incomplete.
     *
     * A token is dangling when either side of its dependency is missing within
     * \p function: it is consumed (in some op's \c tokens_ list) but has no
     * producer, or it is produced (in some op's \c result_token_ list) but has
     * no consumer.  Such tokens carry no usable dependency and would otherwise
     * be duplicated by later expansion passes.
     *
     * The live producer/consumer sets are derived directly from the operations'
     * token lists, so a stale \c VarDependency index cannot hide or invent a
     * dependency.  For each dangling token this removes it from the consuming
     * ops' \c tokens_ list and/or the producing ops' \c result_token_ list, and
     * rebuilds the Function-level \c VarDependency map from the remaining
     * operations.
     *
     * \param function  The function whose dangling tokens should be removed.
     * \return SUCCESS on completion.
     */
    static Status RemoveDanglingTokens(Function& function);
};

} // namespace npu::tile_fwk
