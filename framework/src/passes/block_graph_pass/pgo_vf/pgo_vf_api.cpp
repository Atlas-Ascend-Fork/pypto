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
 * \file pgo_vf_api.cpp
 * \brief Version and fallback implementations of the pgo_vf ABI.
 */

#include "passes/block_graph_pass/pgo_vf/pgo_vf_api.h"

#include "interface/utils/common.h"

extern "C" const char* PLUGIN_GET_VERSION_NAME() { return npu::tile_fwk::kPgoVfVersion; }

#ifdef PGO_VF_SKIP
extern "C" PGO_VF_API npu::tile_fwk::Status PGOVFRun(const pypto::ir::ProgramPtr&, const pypto::ir::FunctionPtr&,
                                                     pypto::ir::ProgramPtr&, pypto::ir::FunctionPtr&)
{
    CHECK(npu::tile_fwk::ExternalError::NOT_IMPLEMENTED_ERROR, false)
        << "PGOVFRun is not implemented: the PGO VF implementation is unavailable.";
    return npu::tile_fwk::FAILED;
}
#endif
