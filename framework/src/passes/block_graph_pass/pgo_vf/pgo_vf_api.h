/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pgo_vf_api.h
 * \brief Public ABI of the pgo_vf module.
 */

#ifndef PGO_VF_API_H
#define PGO_VF_API_H

#include <cstdint>

#include "interface/plugin/plugin_name.h"
#include "ir/program.h"

namespace npu::tile_fwk {
enum Status : uint32_t;

constexpr const char* kPgoVfVersion = "1.0.0";

} // namespace npu::tile_fwk

#if defined(__GNUC__)
#define PGO_VF_API __attribute__((visibility("default")))
#else
#define PGO_VF_API
#endif

extern "C" {
// ABI version string; the host must verify it after dlopen.
PGO_VF_API const char* PLUGIN_GET_VERSION_NAME();

// Runs the function pipeline when function is non-null, otherwise runs the program pipeline.
// Writes the corresponding output only on success; the other output is unchanged.
PGO_VF_API npu::tile_fwk::Status PGOVFRun(const pypto::ir::ProgramPtr& program, const pypto::ir::FunctionPtr& function,
                                          pypto::ir::ProgramPtr& outProgram, pypto::ir::FunctionPtr& outFunction);
}

#endif // PGO_VF_API_H
