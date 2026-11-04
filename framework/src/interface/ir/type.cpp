/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ir/type.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "core/common.h"
#include "core/dtype.h"
#include "ir/identifier.h"
#include "ir/scalar_expr.h"
#include "ir/span.h"
#include "tilefwk/error.h"

namespace pypto {
namespace ir {

namespace {
TileView NormalizeTileView(std::optional<TileView> tileView, size_t rank)
{
    TileView result = tileView ? std::move(*tileView) : TileView{};
    if (result.validShape.empty()) {
        result.validShape.reserve(rank);
        for (size_t i = 0; i < rank; ++i) {
            result.validShape.push_back(std::make_shared<ConstInt>(kDynamicDim, DataType::INT64, Span::Unknown()));
        }
    }
    return result;
}
} // namespace

ShapedType::ShapedType(DataType dtype, const std::vector<int64_t>& shape, std::optional<MemRefPtr> memref)
    : dtype_(dtype), memref_(std::move(memref))
{
    for (int64_t dim : shape) {
        shape_.push_back(std::make_shared<ConstInt>(dim, DataType::INT64, Span::Unknown()));
    }
}

TileType::TileType(const std::vector<int64_t>& shape, DataType dtype, std::optional<MemRefPtr> memref,
                   std::optional<TileView> tileView, std::optional<HardwareInfo> hardwareInfo)
    : ShapedType(dtype, shape, std::move(memref)),
      tileView_(NormalizeTileView(std::move(tileView), shape_.size())),
      hardwareInfo_(std::move(hardwareInfo))
{}

TileType::TileType(std::vector<ExprPtr> shape, DataType dtype, std::optional<MemRefPtr> memref,
                   std::optional<TileView> tileView, std::optional<HardwareInfo> hardwareInfo)
    : ShapedType(dtype, std::move(shape), std::move(memref)),
      tileView_(NormalizeTileView(std::move(tileView), shape_.size())),
      hardwareInfo_(std::move(hardwareInfo))
{}

TupleType::TupleType(std::vector<TypePtr> types) : types_(std::move(types)) {}
} // namespace ir
} // namespace pypto
