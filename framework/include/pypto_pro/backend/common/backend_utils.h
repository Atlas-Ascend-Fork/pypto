/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PYPTO_PRO_BACKEND_COMMON_BACKEND_UTILS_H_
#define PYPTO_PRO_BACKEND_COMMON_BACKEND_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "ir/expr.h"
#include "ir/span.h"
#include "ir/type.h"

namespace pypto {

namespace codegen {
class CCECodegen;
}

namespace backend {

namespace round_mode {

int FindIndex(const std::string& mode, const std::string& opName);

} // namespace round_mode

namespace gather {

struct CompareAttrs {
    bool hasCmpMode = false;
    int cmpMode = 0;
    int offset = 0;
};

CompareAttrs GetCompareAttrs(const ir::CallPtr& op);

} // namespace gather

namespace cce {

/// Adapt an IR value to a data-instruction operand, extracting a tile's PTO descriptor.
/// Tile expressions must denote wrappers; SIMD/SIMT pointer operands use GetExprAsCode instead.
std::string GetOperandAsCode(codegen::CCECodegen& codegen, const ir::ExprPtr& expr);

/// Positions of the operands of a GM access op (tile <-> tensor transfer:
/// block.load / block.store). tensor/tile name the GM-side tensor and the tile
/// operand; offsets names the offset-tuple position.
/// Scalar-container ops (getval/setval) deliberately stay out: consumers
/// like the CCE tensor-declaration collector treat every contract op as a
/// tile-tensor transfer.
struct AccessArgIndices {
    int tensorArgIdx = -1;
    int tileArgIdx = -1;
    int offsetsArgIdx = -1;
};

AccessArgIndices ResolveAccessArgIndices(const std::string& opName);

/// Derive the Tile's ND/DN/NZ/ZN/NN/ZZ layout from its block/scatter layout pair.
std::optional<ir::TensorLayout> GetTileTensorLayout(const ir::HardwareInfo& hardwareInfo);
bool IsNZTensorType(const ir::TensorTypePtr& tensorType);
/// Whether a block.load is an MX scale load. Such a load declares its own GlobalTensor at the
/// op -- TileShape2D maps rows/cols into layout-specific dims that the shared per-layout
/// declaration cannot express -- so it never reads the prologue declaration.
bool IsMXLoad(const ir::TensorTypePtr& tensorType, const ir::TileTypePtr& tileType);
/// The Layout enum an MX load walks its tensor with: the A/B role comes from the destination
/// tile's block layout, ND vs DN from the access order. Empty when *op* is not an MX load.
std::string MXLoadLayoutName(const ir::CallPtr& op);
int64_t GetNZInnerCols(const ir::DataType& dtype);
void ValidateNZTransfer(const std::string& opName, const ir::CallPtr& op, const ir::ExprPtr& tileExpr,
                        const ir::MakeTuplePtr& offsets, const ir::TensorTypePtr& tensorType);

/// Emit *scalar_expr* as a truncated integer literal when *dtype* is an integer
/// (fraction dropped, value masked/sign-extended to the dtype width); return *original*
/// unchanged when *dtype* is not an integer or the scalar is not a float constant.
/// Shared by every backend that takes a scalar operand on an integer destination so a
/// float constant is truncated uniformly instead of emitted as a bare float literal
/// that trips -Wliteral-conversion.
std::string CoerceScalarToInt(const ir::ExprPtr& scalarExpr, const ir::DataType& dtype, const std::string& original);

} // namespace cce

namespace debug_printf {

struct PrintfSegment {
    std::string formatSegment;
    char conversion;
};

std::string EscapeStringLiteral(const std::string& text);
std::string QuoteMlirStringLiteral(const std::string& text);
std::string FormatDebugLocation(const ir::Span& span);
std::string FormatDebugLocationHeader(const ir::Span& span, const std::string& opName);
bool IsSupportedPrintfConversion(char conversion);
size_t FindPrintfConversionIndex(const std::string& formatSegment);
std::vector<PrintfSegment> ParsePrintfSegments(const std::string& format);

} // namespace debug_printf
} // namespace backend
} // namespace pypto

#endif // PYPTO_PRO_BACKEND_COMMON_BACKEND_UTILS_H_
