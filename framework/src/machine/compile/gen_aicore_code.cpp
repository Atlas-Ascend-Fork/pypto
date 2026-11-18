/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gen_aicore_code.cpp
 * \brief
 */

#include "machine/compile/gen_aicore_code.h"
#include "utils/file_utils.h"

namespace npu::tile_fwk {
namespace {
const std::string kAicoreSrcCode = R"!!!(
#include "tilefwk/aicore_entry.h"

#if REQUIRES_SIMT && (defined(__DAV_V310) || defined(__DAV_V920)) && defined(__AIV__)
#define SIMT_META_ENTRY_IMPL(name, sep1, key1, sep2, key2, type) name##sep1##key1##sep2##key2##type
#define SIMT_META_ENTRY(name, key1, key2, type) SIMT_META_ENTRY_IMPL(name, _, key1, _, key2, type)
namespace {
struct SimtMetaTlv {
    uint16_t type;
    uint16_t length;
    uint32_t value;
};

struct SimtKernelMeta {
    SimtMetaTlv aivType;
    SimtMetaTlv staticUbSize;
};

static const SimtKernelMeta g_simt_kernel_meta
    __attribute__((used,
        section(".ascend.meta." TO_STRING(SIMT_META_ENTRY(__OPTYPE__, __OPNAME__, __TILINGKEY__, _mix_aiv))))) = {
        {12U, sizeof(uint32_t), 4U},
        {7U, sizeof(uint32_t), 8U * 1024U},
};
} // namespace
#endif

extern "C" __global__ __aicore__ void KERNEL_ENTRY(__OPTYPE__, __OPNAME__, __TILINGKEY__)(int64_t ffts_addr, int64_t inputs,
        int64_t outputs, int64_t workspace, int64_t tilingdata, int64_t cfgdata) {
    return KernelEntry(ffts_addr, inputs, outputs, workspace, tilingdata, cfgdata);
}
)!!!";
} // namespace

bool GenAicoreSrcFile(const std::string& codeSrcPath)
{
    if (RealPath(codeSrcPath).empty()) {
        SaveFile(codeSrcPath, kAicoreSrcCode);
    }
    return !RealPath(codeSrcPath).empty();
}
} // namespace npu::tile_fwk
