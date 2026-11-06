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

#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace npu::tile_fwk {
// Definitions in their original emission order, with names supplied by codegen.
using PreambleDefinitions = std::vector<std::pair<std::string, std::string>>;

inline void EmitPreambleDefinition(std::ostream& out, const std::string& name, const std::string& code,
                                   PreambleDefinitions* definitions)
{
    out << code;
    if (definitions != nullptr) {
        definitions->emplace_back(name, code);
    }
}
} // namespace npu::tile_fwk
