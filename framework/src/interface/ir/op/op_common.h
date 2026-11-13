/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file op/op_common.h
 * \brief Shared argument validation helpers for IR operators.
 *
 * The block explicit-output op families (elementwise, reduction, matmul)
 * validate their tile arguments one by one via CheckTileArg, passing each
 * argument's allowed memory spaces as a list. Scalar-access operators
 * (getval/setval, SIMT atomics) validate their (container, index) pair via
 * ValidateScalarAccessIndex.
 */

#ifndef PYPTO_IR_OP_OP_COMMON_H_
#define PYPTO_IR_OP_OP_COMMON_H_

#include <algorithm>
#include <initializer_list>
#include <optional>
#include <string>
#include <vector>

#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/memory_space.h"
#include "ir/type.h"
#include "pypto_pro/error.h"

namespace pypto {
namespace ir {

// Report the memory space of a tile, or nullopt when the tile carries no
// memref yet (nothing to check in that case).
inline std::optional<MemorySpace> GetTileMemSpace(const TileTypePtr& tileType)
{
    if (tileType->memref_ == std::nullopt) {
        return std::nullopt;
    }
    return (*tileType->memref_)->memorySpace_;
}

// Format the allowed memory spaces of CheckTileArg for diagnostics, e.g. "Vec or Mat".
inline std::string MemorySpacesToString(std::initializer_list<MemorySpace> spaces)
{
    std::string result;
    for (MemorySpace space : spaces) {
        if (!result.empty()) {
            result += " or ";
        }
        result += MemorySpaceToString(space);
    }
    return result;
}

/**
 * \brief Validate that one operator argument is a TileType whose memory space is allowed
 *
 * Tile arguments are validated one by one: ops with several tile arguments
 * call this once per index, passing the allowed memory spaces of that
 * argument as a list (vec-pipe ops pass {Vec}, cube ops pass {Acc}/{Left}/
 * {Right}/..., and ops that accept several destinations list them all, e.g.
 * {Vec, Mat}). When the tile already carries a memref, its memory space must
 * be one of the allowed spaces; tiles without a memref are accepted (nothing
 * to check yet).
 *
 * \param args Operator argument expressions
 * \param idx Index of the argument to validate
 * \param opName Operator name for diagnostics
 * \param spaces Allowed memory spaces for the tile
 */
inline void CheckTileArg(const std::vector<ExprPtr>& args, size_t idx, const std::string& opName,
                         std::initializer_list<MemorySpace> spaces)
{
    using npu::tile_fwk::ExternalError; // Block-scoped: short enum names without leaking to includers
    auto tileType = As<TileType>(args[idx]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, tileType)
        << "The operator " << opName << " requires argument " << idx << " to be TileType, but got "
        << args[idx]->GetType()->TypeName();
    auto space = GetTileMemSpace(tileType);
    PRO_IR_CHECK(ExternalError::INVALID_OPERATION,
                 space == std::nullopt || std::find(spaces.begin(), spaces.end(), *space) != spaces.end())
        << "The operator " << opName << " requires argument " << idx << " to be a " << MemorySpacesToString(spaces)
        << " tile, but got " << MemorySpaceToString(space.value_or(MemorySpace::DDR));
}

/**
 * \brief Validate the (container, index) argument pair of a scalar-access operator
 *
 * The container must be a Tile or Tensor; the index must be an integer
 * scalar offset or a coordinate tuple whose element count matches the
 * container rank (getval/setval and SIMT atomics share this contract).
 *
 * \param container Container expression (Tile or Tensor)
 * \param index Scalar offset or coordinate tuple expression
 * \param opName Operator name for diagnostics
 */
inline void ValidateScalarAccessIndex(const ExprPtr& container, const ExprPtr& index, const std::string& opName)
{
    using npu::tile_fwk::ExternalError; // Block-scoped: short enum names without leaking to includers
    auto tileType = As<TileType>(container->GetType());
    auto tensorType = As<TensorType>(container->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, tileType || tensorType)
        << opName << " container must be a Tile or Tensor";
    auto checkInteger = [&](const ExprPtr& value) {
        auto scalar = As<ScalarType>(value->GetType());
        PRO_IR_CHECK(ExternalError::INVALID_TYPE, scalar && scalar->dtype_.IsInt())
            << opName << " index must be a non-bool integer scalar";
    };
    if (auto coordinates = As<MakeTuple>(index)) {
        const auto rank = tileType ? tileType->shape_.size() : tensorType->shape_.size();
        PRO_IR_CHECK(ExternalError::INVALID_SHAPE,
                     !coordinates->elements_.empty() && coordinates->elements_.size() == rank)
            << opName << " coordinate count must match container rank " << rank;
        for (const auto& coordinate : coordinates->elements_) {
            checkInteger(coordinate);
        }
    } else {
        checkInteger(index);
    }
}

} // namespace ir
} // namespace pypto

#endif // PYPTO_IR_OP_OP_COMMON_H_
