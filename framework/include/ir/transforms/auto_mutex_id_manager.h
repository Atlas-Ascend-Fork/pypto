/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PYPTO_IR_TRANSFORMS_AUTO_MUTEX_ID_MANAGER_H_
#define PYPTO_IR_TRANSFORMS_AUTO_MUTEX_ID_MANAGER_H_

#include <memory>
#include <string>
#include <vector>

#include "ir/expr.h"
#include "ir/program.h"
#include "ir/span.h"

namespace pypto {
namespace ir {

/**
 * \brief Collect TileGroup memory ranges, allocate AUTO mutex IDs, and resolve placeholders.
 *
 * One manager belongs to one target Program, so Cube and Vector keep independent
 * 0-31 mutex-ID pools. Logical Tiles are identified by their IR expression
 * pointers; physical aliasing is determined from the TileType MemRef range.
 */
class AutoMutexIdManager {
public:
    AutoMutexIdManager();
    ~AutoMutexIdManager();

    AutoMutexIdManager(const AutoMutexIdManager&) = delete;
    AutoMutexIdManager& operator=(const AutoMutexIdManager&) = delete;

    /** Collect a TileGroup using the mutex-ID expressions created by the frontend. */
    void CollectGroup(const std::vector<ExprPtr>& tiles, const MakeTuplePtr& mutexIds, const std::string& groupName);

    ProgramPtr AssignMutexIds(const ProgramPtr& program);

    /** Source span associated with the most recent user-facing planning failure. */
    [[nodiscard]] Span DiagnosticSpan() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ir
} // namespace pypto

#endif // PYPTO_IR_TRANSFORMS_AUTO_MUTEX_ID_MANAGER_H_
