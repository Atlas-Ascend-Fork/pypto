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
 * \brief Temporary identity implementation of the pgo_vf ABI.
 */

#include "passes/block_graph_pass/pgo_vf/pgo_vf_api.h"

extern "C" const char* PLUGIN_GET_VERSION_NAME() { return npu::tile_fwk::kPgoVfVersion; }

#ifdef PGO_VF_SKIP
extern "C" PGO_VF_API npu::tile_fwk::Status PGOVFRunOnFunction(npu::tile_fwk::Function& function)
{
    (void)function;
    return static_cast<npu::tile_fwk::Status>(0);
}
#endif
