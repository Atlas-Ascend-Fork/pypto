/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file codegen_mte_litenpu.cpp
 * \brief
 */

#include <iterator>
#include <string>

#include "codegen_op_litenpu.h"
#include "codegen/utils/codegen_utils.h"

namespace npu::tile_fwk {

std::string CodeGenOpLiteNPU::GenL1ToUBTileTensor() const
{
    int64_t copyMode = 0;
    GetOpAttr(OpAttributeKey::localCopyLocalMode, copyMode);
    std::string cpModeStr = CopyModeToString(static_cast<Matrix::CopyMode>(copyMode));
    std::string dstTensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::DST_IDX));
    std::string src0Tensor = QueryTileTensorNameByIdx(ToUnderlying(MISOIdx::SRC0_IDX));
    auto [coordDst, coordSrc] = PrintDstSrcCoordFromAttr();
    std::vector<std::string> tileOpParamList = {dstTensor, src0Tensor, coordDst, coordSrc};
    std::ostringstream oss;
    oss << tileOpName;
    oss << WrapParamByAngleBrackets({cpModeStr});
    oss << WrapParamByParentheses(tileOpParamList);
    oss << STMT_END;
    return oss.str();
}

} // namespace npu::tile_fwk
