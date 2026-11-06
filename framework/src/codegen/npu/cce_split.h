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

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "codegen/utils/codegen_preamble.h"

namespace npu::tile_fwk {
enum class CceSplitEffect { NON_SYNC_OP, SET, WAIT, BARRIER, FULL_BARRIER, CROSS_CORE_SYNC, EXTERNAL_STATE };

// One Operation participating in codegen; synchronization operations do not count toward the split threshold.
struct CceSplitRegion {
    std::string code;
    // Complete, side-effect-free definitions obtained from SymbolManager, not parsed C++.
    std::vector<std::pair<std::string, std::string>> declarations;
    CceSplitEffect effect{CceSplitEffect::NON_SYNC_OP};
    // Regions emitted inside a loop cannot be split into separate functions.
    bool allowCutAfter{true};
};

// Input: the entire original function body before splitting.
struct CceOriginalFunctionBody {
    PreambleDefinitions preambleDefinitions;
    std::vector<CceSplitRegion> regions;
    bool supported{true};
};

// Output: one generated segment with complete source for independent compilation.
struct CceGeneratedSegment {
    std::string name;
    std::string source;
};

struct CceSplitPlan {
    std::string header;
    std::string entry;
    std::vector<CceGeneratedSegment> parts;
};

// Parameters includes parentheses, without a trailing semicolon.
CceSplitPlan PlanCceSplit(const CceOriginalFunctionBody& body, const std::string& includes,
                          const std::string& kernelName, const std::string& parameters, const std::string& headerName,
                          int64_t threshold);

struct CompileTaskInfo {
    std::string outputPath;
    std::string inputPath;
    std::string compileCmd;
    std::vector<std::string> dependencies;
    std::string publishFrom;
};

std::string QuoteCceBuildArg(const std::string& value);
std::string RenderCceMakefile(std::vector<CompileTaskInfo> tasks);
} // namespace npu::tile_fwk
