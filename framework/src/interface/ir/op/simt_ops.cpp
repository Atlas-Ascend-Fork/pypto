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
 * @file simt_ops.cpp
 * \brief SIMT context and launch operations used by PyPTO Pro's A5 direct-CCE path.
 */

#include <any>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/op_attr_types.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"
#include "ir/type_inference.h"
#include "pypto_pro/error.h"

namespace pypto {
namespace ir {
using npu::tile_fwk::ExternalError;

namespace {

TypePtr DeduceSimtContextComponentType(const std::vector<ExprPtr>& args,
                                       const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.empty())
        << "SIMT context operations do not accept positional arguments";
    int axis = GetOpKwarg<int>(kwargs, "axis", 0);
    PRO_IR_CHECK(ExternalError::INVALID_SHAPE, axis >= 0 && axis <= 2)
        << "SIMT context axis must be in [0, 2], got axis=" << axis;
    return std::make_shared<ScalarType>(DataType(DataType::UINT32));
}

TypePtr DeduceSimtLinearThreadIdxType(const std::vector<ExprPtr>& args,
                                      const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.empty())
        << "SIMT builtin scalar operations do not accept positional arguments";
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, kwargs.empty())
        << "SIMT builtin scalar operations do not accept keyword arguments";
    return std::make_shared<ScalarType>(DataType(DataType::UINT32));
}

TypePtr DeduceSimtWarpSizeType(const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.empty())
        << "simt.warp_size does not accept positional arguments";
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, kwargs.empty()) << "simt.warp_size does not accept keyword arguments";
    return std::make_shared<ScalarType>(DataType(DataType::INT32));
}

TypePtr DeduceSimtSyncType(const std::string& op_name, const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.empty()) << op_name << " does not accept positional arguments";
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, kwargs.empty()) << op_name << " does not accept keyword arguments";
    return GetNoneType();
}

bool IsSimtCastIntegerDtype(DataType dtype)
{
    return dtype == DataType::INT8 || dtype == DataType::INT16 || dtype == DataType::INT32 ||
           dtype == DataType::INT64 || dtype == DataType::UINT8 || dtype == DataType::UINT16 ||
           dtype == DataType::UINT32 || dtype == DataType::UINT64;
}

bool IsSimtCastSupportedDtype(DataType dtype)
{
    return dtype == DataType::INT8 || dtype == DataType::INT16 || dtype == DataType::INT32 ||
           dtype == DataType::INT64 || dtype == DataType::UINT8 || dtype == DataType::UINT16 ||
           dtype == DataType::UINT32 || dtype == DataType::UINT64 || dtype == DataType::FP16 ||
           dtype == DataType::BF16 || dtype == DataType::FP32;
}

bool IsSimtCastStandardRoundMode(RoundMode mode)
{
    return mode == RoundMode::CAST_NONE || mode == RoundMode::CAST_RINT || mode == RoundMode::CAST_ROUND ||
           mode == RoundMode::CAST_FLOOR || mode == RoundMode::CAST_CEIL || mode == RoundMode::CAST_TRUNC;
}

bool IsSimtCastFromInt8OrUint8Supported(DataType target_dtype, RoundMode mode)
{
    if (IsSimtCastIntegerDtype(target_dtype)) {
        return mode == RoundMode::CAST_NONE;
    }
    return false;
}

bool IsSimtCastFromInt16OrUint16Supported(DataType target_dtype, RoundMode mode)
{
    if (IsSimtCastIntegerDtype(target_dtype)) {
        return mode == RoundMode::CAST_NONE;
    }
    if (target_dtype == DataType::FP16 || target_dtype == DataType::BF16) {
        return mode == RoundMode::CAST_RINT || mode == RoundMode::CAST_FLOOR || mode == RoundMode::CAST_CEIL ||
               mode == RoundMode::CAST_TRUNC;
    }
    return false;
}

bool IsSimtCastFromWideIntegerSupported(DataType target_dtype, RoundMode mode)
{
    if (IsSimtCastIntegerDtype(target_dtype)) {
        return mode == RoundMode::CAST_NONE;
    }
    if (target_dtype == DataType::FP16 || target_dtype == DataType::BF16) {
        return mode == RoundMode::CAST_RINT || mode == RoundMode::CAST_ROUND || mode == RoundMode::CAST_FLOOR ||
               mode == RoundMode::CAST_CEIL || mode == RoundMode::CAST_TRUNC;
    }
    if (target_dtype == DataType::FP32) {
        return IsSimtCastStandardRoundMode(mode);
    }
    return false;
}

bool IsSimtCastFromFp16OrBf16Supported(DataType target_dtype, RoundMode mode)
{
    if (target_dtype == DataType::FP32) {
        return mode == RoundMode::CAST_NONE;
    }
    if (target_dtype == DataType::FP16 || target_dtype == DataType::BF16 || target_dtype == DataType::INT32 ||
        target_dtype == DataType::INT64 || target_dtype == DataType::UINT32 || target_dtype == DataType::UINT64) {
        return mode == RoundMode::CAST_RINT || mode == RoundMode::CAST_ROUND || mode == RoundMode::CAST_FLOOR ||
               mode == RoundMode::CAST_CEIL || mode == RoundMode::CAST_TRUNC;
    }
    if (target_dtype == DataType::INT8 || target_dtype == DataType::UINT8) {
        return mode == RoundMode::CAST_TRUNC;
    }
    if (target_dtype == DataType::INT16 || target_dtype == DataType::UINT16) {
        return mode == RoundMode::CAST_RINT || mode == RoundMode::CAST_FLOOR || mode == RoundMode::CAST_CEIL ||
               mode == RoundMode::CAST_TRUNC;
    }
    return false;
}

bool IsSimtCastFromFp32Supported(DataType target_dtype, RoundMode mode)
{
    if (target_dtype == DataType::FP16) {
        return mode == RoundMode::CAST_ODD || IsSimtCastStandardRoundMode(mode);
    }
    if (target_dtype == DataType::BF16 || target_dtype == DataType::INT32 || target_dtype == DataType::INT64 ||
        target_dtype == DataType::UINT32 || target_dtype == DataType::UINT64) {
        return IsSimtCastStandardRoundMode(mode);
    }
    return false;
}

bool IsSimtCastSupported(DataType source_dtype, DataType target_dtype, RoundMode mode)
{
    if (source_dtype == target_dtype) {
        if (mode != RoundMode::CAST_NONE) {
            return false;
        }
        return IsSimtCastSupportedDtype(source_dtype);
    }
    if (source_dtype == DataType::INT8 || source_dtype == DataType::UINT8) {
        return IsSimtCastFromInt8OrUint8Supported(target_dtype, mode);
    }
    if (source_dtype == DataType::INT16 || source_dtype == DataType::UINT16) {
        return IsSimtCastFromInt16OrUint16Supported(target_dtype, mode);
    }
    if (source_dtype == DataType::INT32 || source_dtype == DataType::UINT32 || source_dtype == DataType::INT64 ||
        source_dtype == DataType::UINT64) {
        return IsSimtCastFromWideIntegerSupported(target_dtype, mode);
    }
    if (source_dtype == DataType::FP16 || source_dtype == DataType::BF16) {
        return IsSimtCastFromFp16OrBf16Supported(target_dtype, mode);
    }
    if (source_dtype == DataType::FP32) {
        return IsSimtCastFromFp32Supported(target_dtype, mode);
    }
    return false;
}

TypePtr DeduceSimtCastType(const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 1) << "simt.cast requires one scalar argument";
    auto source_type = As<ScalarType>(args[0]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, source_type) << "simt.cast value must be a scalar";

    DataType target_dtype = GetOpKwarg<DataType>(kwargs, "target_type");
    auto mode = static_cast<RoundMode>(GetOpKwarg<int>(kwargs, "mode"));
    PRO_IR_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, IsSimtCastSupported(source_type->dtype_, target_dtype, mode))
        << "simt.cast does not support " << source_type->dtype_.ToString() << " -> " << target_dtype.ToString()
        << " with mode " << EnumToString(mode);
    return std::make_shared<ScalarType>(target_dtype);
}

bool IsSimtBitcastSupported(DataType source_dtype, DataType target_dtype)
{
    if (source_dtype == DataType::FP16 || source_dtype == DataType::BF16) {
        return target_dtype == DataType::INT16 || target_dtype == DataType::UINT16;
    }
    if (source_dtype == DataType::INT16 || source_dtype == DataType::UINT16) {
        return target_dtype == DataType::FP16 || target_dtype == DataType::BF16;
    }
    if (source_dtype == DataType::FP32) {
        return target_dtype == DataType::INT32 || target_dtype == DataType::UINT32;
    }
    if (source_dtype == DataType::INT32 || source_dtype == DataType::UINT32) {
        return target_dtype == DataType::FP32;
    }
    return false;
}

TypePtr DeduceSimtBitcastType(const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 1) << "simt.bitcast requires one scalar argument";
    auto source_type = As<ScalarType>(args[0]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, source_type) << "simt.bitcast value must be a scalar";

    DataType target_dtype = GetOpKwarg<DataType>(kwargs, "target_type");
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, source_type->dtype_.GetBit() == target_dtype.GetBit())
        << "simt.bitcast requires source and target dtypes to have the same bit width, got "
        << source_type->dtype_.ToString() << " (" << source_type->dtype_.GetBit() << " bits) -> "
        << target_dtype.ToString() << " (" << target_dtype.GetBit() << " bits)";
    PRO_IR_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, IsSimtBitcastSupported(source_type->dtype_, target_dtype))
        << "simt.bitcast does not support " << source_type->dtype_.ToString() << " -> " << target_dtype.ToString();
    return std::make_shared<ScalarType>(target_dtype);
}

bool IsDtypeInList(const DataType& dtype, std::initializer_list<DataType> supported_dtypes)
{
    for (const auto& supported_dtype : supported_dtypes) {
        if (dtype == supported_dtype) {
            return true;
        }
    }
    return false;
}

TypePtr DeduceSimtAtomicType(const std::string& op_name, size_t operand_count,
                             std::initializer_list<DataType> ub_dtypes, std::initializer_list<DataType> gm_dtypes,
                             const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs,
                             std::initializer_list<DataType> no_result_dtypes = {})
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == operand_count + 2)
        << op_name << " requires container, offset, and " << operand_count << " scalar operand(s)";
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, kwargs.empty()) << op_name << " does not accept keyword arguments";

    auto tile_type = As<TileType>(args[0]->GetType());
    auto tensor_type = As<TensorType>(args[0]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, tile_type || tensor_type)
        << op_name << " container must be a Tile or Tensor";
    DataType dtype = tile_type ? tile_type->dtype_ : tensor_type->dtype_;
    PRO_IR_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, IsDtypeInList(dtype, tile_type ? ub_dtypes : gm_dtypes))
        << op_name << " does not support dtype " << dtype.ToString() << " on " << (tile_type ? "UB Tile" : "GM Tensor");

    auto offset_type = As<ScalarType>(args[1]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE,
                 offset_type && offset_type->dtype_ != DataType::BOOL &&
                     (offset_type->dtype_.IsInt() || offset_type->dtype_ == DataType::INDEX))
        << op_name << " offset must be a non-bool integer scalar";

    for (size_t i = 0; i < operand_count; ++i) {
        auto operand_type = As<ScalarType>(args[i + 2]->GetType());
        PRO_IR_CHECK(ExternalError::INVALID_TYPE, operand_type) << op_name << " operand " << i << " must be a scalar";
        PRO_IR_CHECK(ExternalError::INVALID_TYPE, operand_type->dtype_ == dtype)
            << op_name << " operand " << i << " dtype must match target dtype " << dtype.ToString() << ", but got "
            << operand_type->dtype_.ToString();
    }
    if (IsDtypeInList(dtype, no_result_dtypes)) {
        return GetNoneType();
    }
    return std::make_shared<ScalarType>(dtype);
}

std::string FormatSupportedDtypes(std::initializer_list<DataType> dtypes)
{
    std::string result;
    for (const auto& dtype : dtypes) {
        if (!result.empty()) {
            result += ", ";
        }
        result += dtype.ToString();
    }
    return result;
}

TypePtr DeduceSimtWarpNoArgType(const std::string& op_name, DataType result_dtype, const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.empty()) << op_name << " does not accept positional arguments";
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, kwargs.empty()) << op_name << " does not accept keyword arguments";
    return std::make_shared<ScalarType>(result_dtype);
}

TypePtr DeduceSimtWarpVoteType(const std::string& op_name, DataType result_dtype, const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 1) << op_name << " requires exactly one predicate";
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, kwargs.empty()) << op_name << " does not accept keyword arguments";
    auto predicate_type = As<ScalarType>(args[0]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, predicate_type) << op_name << " predicate must be a scalar";
    PRO_IR_CHECK(ExternalError::INVALID_TYPE,
                 predicate_type->dtype_ == DataType::BOOL || predicate_type->dtype_ == DataType::INT32)
        << op_name << " predicate must have BOOL or INT32 dtype, got " << predicate_type->dtype_.ToString();
    return std::make_shared<ScalarType>(result_dtype);
}

ScalarTypePtr DeduceSimtWarpShuffleCommonType(const std::string& op_name, const std::string& control_name,
                                              const std::vector<ExprPtr>& args,
                                              const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 3)
        << op_name << " requires value, " << control_name << ", and width";
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, kwargs.empty()) << op_name << " does not accept keyword arguments";

    auto value_type = As<ScalarType>(args[0]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, value_type) << op_name << " value must be a scalar";
    PRO_IR_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR,
                 IsDtypeInList(value_type->dtype_, {DataType::INT32, DataType::UINT32, DataType::INT64,
                                                    DataType::UINT64, DataType::FP16, DataType::BF16, DataType::FP32}))
        << op_name << " supports only INT32, UINT32, INT64, UINT64, FP16, BF16, and FP32 values, got "
        << value_type->dtype_.ToString();

    auto width_type = As<ScalarType>(args[2]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, width_type && width_type->dtype_ == DataType::INT32)
        << op_name << " width must be an INT32 scalar";
    auto width = As<ConstInt>(args[2]);
    if (width) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT,
                     width->value_ >= 1 && width->value_ <= 32 && (width->value_ & (width->value_ - 1)) == 0)
            << op_name << " width must be one of 1, 2, 4, 8, 16, or 32";
    }
    return std::make_shared<ScalarType>(value_type->dtype_);
}

TypePtr DeduceSimtWarpShflType(const std::string& op_name, const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    auto value_type = DeduceSimtWarpShuffleCommonType(op_name, "src_lane", args, kwargs);
    auto src_lane_type = As<ScalarType>(args[1]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, src_lane_type) << op_name << " src_lane must be a scalar";
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, src_lane_type->dtype_ == DataType::INT32)
        << op_name << " src_lane must have " << DataType::INT32.ToString() << " dtype, got "
        << src_lane_type->dtype_.ToString();

    auto src_lane = As<ConstInt>(args[1]);
    if (src_lane) {
        PRO_IR_CHECK(ExternalError::OUT_OF_RANGE, src_lane->value_ >= 0 && src_lane->value_ < 32)
            << op_name << " src_lane must be in [0, 31]";
    }
    return value_type;
}

TypePtr DeduceSimtWarpShflUpDownType(const std::string& op_name, const std::vector<ExprPtr>& args,
                                     const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    auto value_type = DeduceSimtWarpShuffleCommonType(op_name, "delta", args, kwargs);
    auto delta_type = As<ScalarType>(args[1]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, delta_type) << op_name << " delta must be a scalar";
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, delta_type->dtype_ == DataType::UINT32)
        << op_name << " delta must have " << DataType::UINT32.ToString() << " dtype, got "
        << delta_type->dtype_.ToString();

    auto delta = As<ConstInt>(args[1]);
    if (delta) {
        PRO_IR_CHECK(ExternalError::OUT_OF_RANGE, delta->value_ >= 0 && delta->value_ < 32)
            << op_name << " delta must be in [0, 31]";
    }
    return value_type;
}

TypePtr DeduceSimtWarpShflXorType(const std::string& op_name, const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    auto value_type = DeduceSimtWarpShuffleCommonType(op_name, "lane_mask", args, kwargs);
    auto lane_mask_type = As<ScalarType>(args[1]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, lane_mask_type) << op_name << " lane_mask must be a scalar";
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, lane_mask_type->dtype_ == DataType::INT32)
        << op_name << " lane_mask must have " << DataType::INT32.ToString() << " dtype, got "
        << lane_mask_type->dtype_.ToString();

    auto lane_mask = As<ConstInt>(args[1]);
    if (lane_mask) {
        PRO_IR_CHECK(ExternalError::OUT_OF_RANGE, lane_mask->value_ >= 0 && lane_mask->value_ < 32)
            << op_name << " lane_mask must be in [0, 31]";
        auto width = As<ConstInt>(args[2]);
        PRO_IR_CHECK(ExternalError::OUT_OF_RANGE, !width || lane_mask->value_ < width->value_)
            << op_name << " lane_mask must be less than width";
    }
    return value_type;
}

TypePtr DeduceSimtMathType(const std::string& op_name, size_t operand_count,
                           std::initializer_list<DataType> supported_dtypes, const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs,
                           std::optional<DataType> result_dtype = std::nullopt)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == operand_count)
        << op_name << " requires exactly " << operand_count << " scalar operand(s)";
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, kwargs.empty()) << op_name << " does not accept keyword arguments";

    DataType dtype = DataType::BOOL;
    for (size_t i = 0; i < args.size(); ++i) {
        auto scalar_type = As<ScalarType>(args[i]->GetType());
        PRO_IR_CHECK(ExternalError::INVALID_TYPE, scalar_type) << op_name << " operand " << i << " must be a scalar";
        if (i == 0) {
            dtype = scalar_type->dtype_;
            PRO_IR_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, IsDtypeInList(dtype, supported_dtypes))
                << op_name << " supports only " << FormatSupportedDtypes(supported_dtypes) << ", got "
                << dtype.ToString();
        } else {
            PRO_IR_CHECK(ExternalError::INVALID_TYPE, scalar_type->dtype_ == dtype)
                << op_name << " requires operands with the same dtype, got " << dtype.ToString() << " and "
                << scalar_type->dtype_.ToString();
        }
    }
    return std::make_shared<ScalarType>(result_dtype.value_or(dtype));
}

TypePtr DeduceSimtLaunchType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_SHAPE, args.size() >= 3) << "simt.launch requires three launch dimensions";
    int64_t total_threads = 1;
    for (std::size_t i = 0; i < 3; ++i) {
        auto dim_type = As<ScalarType>(args[i]->GetType());
        PRO_IR_CHECK(ExternalError::INVALID_SHAPE,
                     dim_type && dim_type->dtype_.IsInt() && dim_type->dtype_ != DataType(DataType::BOOL))
            << "simt.launch dimensions must be non-bool integer scalars";
        auto dim = As<ConstInt>(args[i]);
        PRO_IR_CHECK(ExternalError::INVALID_SHAPE, dim && dim->value_ > 0 && dim->value_ <= 2048)
            << "simt.launch dimensions must be compile-time integers in [1, 2048], got dim[" << i
            << "]=" << (dim ? dim->value_ : -1);
        total_threads *= dim->value_;
    }
    PRO_IR_CHECK(ExternalError::OUT_OF_RANGE, total_threads <= 2048)
        << "simt.launch total thread count must not exceed 2048, got total_threads=" << total_threads;
    int max_threads = GetOpKwarg<int>(kwargs, "max_threads");
    PRO_IR_CHECK(ExternalError::OUT_OF_RANGE, max_threads >= 1 && max_threads <= 2048)
        << "simt.launch max_threads must be in [1, 2048]";
    PRO_IR_CHECK(ExternalError::OUT_OF_RANGE, total_threads <= max_threads)
        << "simt.launch threads " << total_threads << " exceed callee max_threads " << max_threads;
    for (std::size_t i = 3; i < args.size(); ++i) {
        auto arg_type = args[i]->GetType();
        if (auto tile_type = As<TileType>(arg_type)) {
            PRO_IR_CHECK(ExternalError::INVALID_SHAPE, tile_type->shape_.size() == 2)
                << "simt.launch Tile argument must have a two-dimensional shape";
            PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<Var>(args[i]) != nullptr)
                << "simt.launch Tile argument must be a Var";
            for (const auto& dim : tile_type->shape_) {
                PRO_IR_CHECK(ExternalError::DYNAMIC_SHAPE_COMPUTE_UNSUPPORTED, As<ConstInt>(dim) != nullptr)
                    << "simt.launch Tile argument must have a static shape";
            }
            PRO_IR_CHECK(ExternalError::INVALID_TYPE, tile_type->dtype_.GetBit() >= 8)
                << "simt.launch Tile argument has a sub-byte element dtype";
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, tile_type->memref_.has_value())
                << "simt.launch Tile argument has no memory reference";
            PRO_IR_CHECK(ExternalError::INVALID_TYPE, (*tile_type->memref_)->memorySpace_ == MemorySpace::Vec)
                << "simt.launch Tile argument must be a Vec-memory Tile";
            PRO_IR_CHECK(ExternalError::INVALID_FORMAT,
                         tile_type->hardwareInfo_.has_value() &&
                             tile_type->hardwareInfo_->blayout == TileLayout::row_major &&
                             tile_type->hardwareInfo_->slayout == TileLayout::none_box)
                << "simt.launch Tile argument must be an ND Vec Tile";
        } else if (auto tensor_type = As<TensorType>(arg_type)) {
            PRO_IR_CHECK(ExternalError::INVALID_SHAPE, !tensor_type->shape_.empty())
                << "simt.launch Tensor argument must have a non-empty shape";
            for (const auto& dim : tensor_type->shape_) {
                PRO_IR_CHECK(ExternalError::DYNAMIC_SHAPE_COMPUTE_UNSUPPORTED, As<ConstInt>(dim) != nullptr)
                    << "simt.launch Tensor argument must have a static shape";
            }
            PRO_IR_CHECK(ExternalError::INVALID_TYPE, tensor_type->dtype_.GetBit() >= 8)
                << "simt.launch Tensor argument has a sub-byte element dtype";
            PRO_IR_CHECK(ExternalError::INVALID_FORMAT,
                         tensor_type->tensor_view_.has_value() && tensor_type->tensor_view_->layout == TensorLayout::ND)
                << "simt.launch Tensor argument must use ND layout";
            PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<Var>(args[i]) != nullptr)
                << "simt.launch Tensor argument must be a Var";
        } else {
            PRO_IR_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, As<ScalarType>(arg_type))
                << "simt.launch supports only Tile, Tensor, and scalar arguments";
        }
    }
    return GetUnknownType();
}

} // namespace

REGISTER_OP("simt.thread_idx")
    .set_op_category("SimtOp")
    .set_description("Read one component of the native SIMT thread index")
    .no_argument()
    .set_attr<int>("axis")
    .f_deduce_type(DeduceSimtContextComponentType);

REGISTER_OP("simt.block_dim")
    .set_op_category("SimtOp")
    .set_description("Read one component of the native SIMT block dimensions")
    .no_argument()
    .set_attr<int>("axis")
    .f_deduce_type(DeduceSimtContextComponentType);

REGISTER_OP("simt.block_idx")
    .set_op_category("SimtOp")
    .set_description("Read one component of the native SIMT block index")
    .no_argument()
    .set_attr<int>("axis")
    .f_deduce_type(DeduceSimtContextComponentType);

REGISTER_OP("simt.grid_dim")
    .set_op_category("SimtOp")
    .set_description("Read one component of the native SIMT grid dimensions")
    .no_argument()
    .set_attr<int>("axis")
    .f_deduce_type(DeduceSimtContextComponentType);

REGISTER_OP("simt.linear_thread_idx")
    .set_op_category("SimtOp")
    .set_description("Read the x-major flattened thread index within the current block")
    .no_argument()
    .f_deduce_type(DeduceSimtLinearThreadIdxType);

REGISTER_OP("simt.warp_size")
    .set_op_category("SimtOp")
    .set_description("Read the native SIMT warp size")
    .no_argument()
    .f_deduce_type(DeduceSimtWarpSizeType);

REGISTER_OP("simt.syncthreads")
    .set_op_category("SimtOp")
    .set_description("Synchronize all SIMT threads in the current block")
    .no_argument()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtSyncType("simt.syncthreads", args, kwargs);
    });

REGISTER_OP("simt.threadfence_block")
    .set_op_category("SimtOp")
    .set_description("Order SIMT memory operations for threads in the current block")
    .no_argument()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtSyncType("simt.threadfence_block", args, kwargs);
    });

REGISTER_OP("simt.threadfence")
    .set_op_category("SimtOp")
    .set_description("Order SIMT memory operations with device-wide visibility")
    .no_argument()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtSyncType("simt.threadfence", args, kwargs);
    });

#define REGISTER_SIMT_LANE_OP(OpName, Description)                                                              \
    REGISTER_OP("simt." OpName)                                                                                 \
        .set_op_category("SimtOp")                                                                              \
        .set_description(Description)                                                                           \
        .no_argument()                                                                                          \
        .f_deduce_type(                                                                                         \
            [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) { \
                return DeduceSimtWarpNoArgType("simt." OpName, DataType::INT32, args, kwargs);                  \
            })

REGISTER_SIMT_LANE_OP("lane_id", "Read the current lane index within the warp");
REGISTER_SIMT_LANE_OP("lanemask_eq", "Return a mask containing only the current lane");
REGISTER_SIMT_LANE_OP("lanemask_le", "Return a mask containing lanes up to and including the current lane");
REGISTER_SIMT_LANE_OP("lanemask_lt", "Return a mask containing lanes below the current lane");
REGISTER_SIMT_LANE_OP("lanemask_ge", "Return a mask containing lanes at or above the current lane");
REGISTER_SIMT_LANE_OP("lanemask_gt", "Return a mask containing lanes above the current lane");

#undef REGISTER_SIMT_LANE_OP

#define REGISTER_SIMT_WARP_VOTE_OP(OpName, Description, ResultDtype)                                            \
    REGISTER_OP("simt." OpName)                                                                                 \
        .set_op_category("SimtOp")                                                                              \
        .set_description(Description)                                                                           \
        .add_argument("predicate", "BOOL or INT32 lane predicate")                                              \
        .f_deduce_type(                                                                                         \
            [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) { \
                return DeduceSimtWarpVoteType("simt." OpName, ResultDtype, args, kwargs);                       \
            })

REGISTER_SIMT_WARP_VOTE_OP("warp_all", "Test whether all active lanes satisfy a predicate", DataType::INT32);
REGISTER_SIMT_WARP_VOTE_OP("warp_any", "Test whether any active lane satisfies a predicate", DataType::INT32);
REGISTER_SIMT_WARP_VOTE_OP("warp_ballot", "Return the active-lane predicate bit mask", DataType::UINT32);

#undef REGISTER_SIMT_WARP_VOTE_OP

REGISTER_OP("simt.warp_active_mask")
    .set_op_category("SimtOp")
    .set_description("Return the currently active lane mask")
    .no_argument()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtWarpNoArgType("simt.warp_active_mask", DataType::UINT32, args, kwargs);
    });

REGISTER_OP("simt.warp_shfl")
    .set_op_category("SimtOp")
    .set_description("Read a scalar from one source lane")
    .add_argument("value", "Scalar value to exchange")
    .add_argument("src_lane", "Source lane index")
    .add_argument("width", "Logical warp subgroup width")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtWarpShflType("simt.warp_shfl", args, kwargs);
    });

REGISTER_OP("simt.warp_shfl_up")
    .set_op_category("SimtOp")
    .set_description("Read a scalar from a lower-numbered lane")
    .add_argument("value", "Scalar value to exchange")
    .add_argument("delta", "Unsigned lane delta")
    .add_argument("width", "Logical warp subgroup width")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtWarpShflUpDownType("simt.warp_shfl_up", args, kwargs);
    });

REGISTER_OP("simt.warp_shfl_down")
    .set_op_category("SimtOp")
    .set_description("Read a scalar from a higher-numbered lane")
    .add_argument("value", "Scalar value to exchange")
    .add_argument("delta", "Unsigned lane delta")
    .add_argument("width", "Logical warp subgroup width")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtWarpShflUpDownType("simt.warp_shfl_down", args, kwargs);
    });

REGISTER_OP("simt.warp_shfl_xor")
    .set_op_category("SimtOp")
    .set_description("Read a scalar from the lane selected by XOR")
    .add_argument("value", "Scalar value to exchange")
    .add_argument("lane_mask", "XOR lane mask")
    .add_argument("width", "Logical warp subgroup width")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtWarpShflXorType("simt.warp_shfl_xor", args, kwargs);
    });

#define REGISTER_SIMT_WARP_REDUCE_OP(OpName, Description)                                                              \
    REGISTER_OP("simt." OpName)                                                                                        \
        .set_op_category("SimtOp")                                                                                     \
        .set_description(Description)                                                                                  \
        .add_argument("value", "Scalar value to reduce")                                                               \
        .f_deduce_type([](const std::vector<ExprPtr>& args,                                                            \
                          const std::vector<std::pair<std::string, std::any>>& kwargs) {                               \
            return DeduceSimtMathType(                                                                                 \
                "simt." OpName, 1, {DataType::INT32, DataType::UINT32, DataType::FP16, DataType::FP32}, args, kwargs); \
        })

REGISTER_SIMT_WARP_REDUCE_OP("warp_reduce_add", "Reduce a scalar sum across active lanes");
REGISTER_SIMT_WARP_REDUCE_OP("warp_reduce_max", "Reduce a scalar maximum across active lanes");
REGISTER_SIMT_WARP_REDUCE_OP("warp_reduce_min", "Reduce a scalar minimum across active lanes");

#undef REGISTER_SIMT_WARP_REDUCE_OP

REGISTER_OP("simt.cast")
    .set_op_category("SimtOp")
    .set_description("Convert one SIMT scalar value to a supported target dtype")
    .add_argument("value", "Source scalar value")
    .set_attr<DataType>("target_type")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceSimtCastType);

REGISTER_OP("simt.bitcast")
    .set_op_category("SimtOp")
    .set_description("Reinterpret one SIMT scalar bit pattern as a supported equal-width dtype")
    .add_argument("value", "Source scalar value")
    .set_attr<DataType>("target_type")
    .f_deduce_type(DeduceSimtBitcastType);

#define REGISTER_SIMT_MATH_UNARY_OP(OpName, Description, ResultDtype, ...)                                      \
    REGISTER_OP("simt." OpName)                                                                                 \
        .set_op_category("SimtOp")                                                                              \
        .set_description(Description)                                                                           \
        .add_argument("value", "Scalar operand")                                                                \
        .f_deduce_type(                                                                                         \
            [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) { \
                return DeduceSimtMathType("simt." OpName, 1, {__VA_ARGS__}, args, kwargs, ResultDtype);         \
            })

REGISTER_SIMT_MATH_UNARY_OP("abs", "Compute the absolute value of one supported scalar", std::nullopt, DataType::FP16,
                            DataType::BF16, DataType::FP32, DataType::INT32, DataType::INT64);
REGISTER_SIMT_MATH_UNARY_OP("sqrt", "Compute the square root of one floating-point scalar", std::nullopt,
                            DataType::FP16, DataType::BF16, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("rsqrt", "Compute the reciprocal square root of one floating-point scalar", std::nullopt,
                            DataType::FP16, DataType::BF16, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("exp", "Compute the natural exponential of one floating-point scalar", std::nullopt,
                            DataType::FP16, DataType::BF16, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("exp2", "Compute the base-two exponential of one floating-point scalar", std::nullopt,
                            DataType::FP16, DataType::BF16, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("log", "Compute the natural logarithm of one floating-point scalar", std::nullopt,
                            DataType::FP16, DataType::BF16, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("log2", "Compute the base-two logarithm of one floating-point scalar", std::nullopt,
                            DataType::FP16, DataType::BF16, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("log1p", "Compute log(1 + value) for one FP32 scalar", std::nullopt, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("sin", "Compute the sine of one floating-point scalar", std::nullopt, DataType::FP16,
                            DataType::BF16, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("cos", "Compute the cosine of one floating-point scalar", std::nullopt, DataType::FP16,
                            DataType::BF16, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("tanh", "Compute the hyperbolic tangent of one floating-point scalar", std::nullopt,
                            DataType::FP16, DataType::BF16, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("rint", "Round one floating-point scalar to the nearest integer value", std::nullopt,
                            DataType::FP16, DataType::BF16, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("round", "Round one floating-point scalar halfway away from zero", std::nullopt,
                            DataType::FP16, DataType::BF16, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("floor", "Round one floating-point scalar downward", std::nullopt, DataType::FP16,
                            DataType::BF16, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("ceil", "Round one floating-point scalar upward", std::nullopt, DataType::FP16,
                            DataType::BF16, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("trunc", "Round one floating-point scalar toward zero", std::nullopt, DataType::FP16,
                            DataType::BF16, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("isnan", "Test whether one floating-point scalar is NaN", DataType::BOOL, DataType::FP16,
                            DataType::BF16, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("isinf", "Test whether one floating-point scalar is infinite", DataType::BOOL,
                            DataType::FP16, DataType::BF16, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("isfinite", "Test whether one FP16 or FP32 scalar is finite", DataType::BOOL,
                            DataType::FP16, DataType::FP32);

REGISTER_SIMT_MATH_UNARY_OP("exp10", "Compute the base-ten exponential of one floating-point scalar", std::nullopt,
                            DataType::FP16, DataType::BF16, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("log10", "Compute the base-ten logarithm of one floating-point scalar", std::nullopt,
                            DataType::FP16, DataType::BF16, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("rcp", "Compute the reciprocal of one FP16 or BF16 scalar", std::nullopt, DataType::FP16,
                            DataType::BF16);
REGISTER_SIMT_MATH_UNARY_OP("tan", "Compute the tangent of one FP32 scalar", std::nullopt, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("atan", "Compute the arc tangent of one FP32 scalar", std::nullopt, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("expm1", "Compute exp(value) - 1 for one FP32 scalar", std::nullopt, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("logb", "Extract the unbiased exponent of one FP32 scalar", std::nullopt, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("cosh", "Compute the hyperbolic cosine of one FP32 scalar", std::nullopt, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("acos", "Compute the arc cosine of one FP32 scalar", std::nullopt, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("sinh", "Compute the hyperbolic sine of one FP32 scalar", std::nullopt, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("asin", "Compute the arc sine of one FP32 scalar", std::nullopt, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("cbrt", "Compute the cube root of one FP32 scalar", std::nullopt, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("tanpi", "Compute the tangent of pi times an FP32 scalar", std::nullopt, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("atanh", "Compute the inverse hyperbolic tangent of an FP32 scalar", std::nullopt,
                            DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("cospi", "Compute the cosine of pi times an FP32 scalar", std::nullopt, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("acosh", "Compute the inverse hyperbolic cosine of an FP32 scalar", std::nullopt,
                            DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("sinpi", "Compute the sine of pi times an FP32 scalar", std::nullopt, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("asinh", "Compute the inverse hyperbolic sine of an FP32 scalar", std::nullopt,
                            DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("rcbrt", "Compute the reciprocal cube root of an FP32 scalar", std::nullopt,
                            DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("ilogb", "Return the unbiased exponent of an FP32 scalar", DataType::INT32, DataType::FP32);
REGISTER_SIMT_MATH_UNARY_OP("signbit", "Return the sign bit of an FP32 scalar", DataType::INT32, DataType::FP32);

#undef REGISTER_SIMT_MATH_UNARY_OP

REGISTER_OP("simt.popcount")
    .set_op_category("SimtOp")
    .set_description("Count set bits in a UINT32 or UINT64 scalar, returning INT32")
    .add_argument("value", "Unsigned scalar operand")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtMathType("simt.popcount", 1, {DataType::UINT32, DataType::UINT64}, args, kwargs,
                                  DataType::INT32);
    });

REGISTER_OP("simt.mul_hi")
    .set_op_category("SimtOp")
    .set_description("Compute the high half of the full product of two same-dtype integers")
    .add_argument("lhs", "Left scalar operand")
    .add_argument("rhs", "Right scalar operand")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtMathType("simt.mul_hi", 2,
                                  {DataType::INT32, DataType::UINT32, DataType::INT64, DataType::UINT64}, args, kwargs);
    });

REGISTER_OP("simt.max_nan")
    .set_op_category("SimtOp")
    .set_description("Select the maximum FP16 or BF16 scalar and propagate NaN")
    .add_argument("lhs", "Left scalar operand")
    .add_argument("rhs", "Right scalar operand")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtMathType("simt.max_nan", 2, {DataType::FP16, DataType::BF16}, args, kwargs);
    });

REGISTER_OP("simt.min_nan")
    .set_op_category("SimtOp")
    .set_description("Select the minimum FP16 or BF16 scalar and propagate NaN")
    .add_argument("lhs", "Left scalar operand")
    .add_argument("rhs", "Right scalar operand")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtMathType("simt.min_nan", 2, {DataType::FP16, DataType::BF16}, args, kwargs);
    });

REGISTER_OP("simt.atan2")
    .set_op_category("SimtOp")
    .set_description("Compute the quadrant-aware arc tangent of two FP32 scalars")
    .add_argument("lhs", "Left scalar operand")
    .add_argument("rhs", "Right scalar operand")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtMathType("simt.atan2", 2, {DataType::FP32}, args, kwargs);
    });

REGISTER_OP("simt.copysign")
    .set_op_category("SimtOp")
    .set_description("Select the magnitude and sign of two FP32 scalars")
    .add_argument("lhs", "Left scalar operand")
    .add_argument("rhs", "Right scalar operand")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtMathType("simt.copysign", 2, {DataType::FP32}, args, kwargs);
    });

REGISTER_OP("simt.nextafter")
    .set_op_category("SimtOp")
    .set_description("Return the next FP32 value toward another FP32 scalar")
    .add_argument("lhs", "Left scalar operand")
    .add_argument("rhs", "Right scalar operand")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtMathType("simt.nextafter", 2, {DataType::FP32}, args, kwargs);
    });

REGISTER_OP("simt.fmod")
    .set_op_category("SimtOp")
    .set_description("Compute an FP32 remainder with a quotient truncated toward zero")
    .add_argument("lhs", "Dividend")
    .add_argument("rhs", "Divisor")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtMathType("simt.fmod", 2, {DataType::FP32}, args, kwargs);
    });

REGISTER_OP("simt.min")
    .set_op_category("SimtOp")
    .set_description("Compute the minimum of two same-dtype supported scalars")
    .add_argument("lhs", "Left scalar operand")
    .add_argument("rhs", "Right scalar operand")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtMathType(
            "simt.min", 2,
            {DataType::FP16, DataType::BF16, DataType::FP32, DataType::INT8, DataType::INT16, DataType::INT32,
             DataType::INT64, DataType::UINT8, DataType::UINT16, DataType::UINT32, DataType::UINT64},
            args, kwargs);
    });

REGISTER_OP("simt.max")
    .set_op_category("SimtOp")
    .set_description("Compute the maximum of two same-dtype supported scalars")
    .add_argument("lhs", "Left scalar operand")
    .add_argument("rhs", "Right scalar operand")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtMathType(
            "simt.max", 2,
            {DataType::FP16, DataType::BF16, DataType::FP32, DataType::INT8, DataType::INT16, DataType::INT32,
             DataType::INT64, DataType::UINT8, DataType::UINT16, DataType::UINT32, DataType::UINT64},
            args, kwargs);
    });

REGISTER_OP("simt.fma")
    .set_op_category("SimtOp")
    .set_description("Compute a fused multiply-add of three same-dtype floating-point scalars")
    .add_argument("lhs", "Left multiplication operand")
    .add_argument("rhs", "Right multiplication operand")
    .add_argument("addend", "Scalar addend")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtMathType("simt.fma", 3, {DataType::FP16, DataType::BF16, DataType::FP32}, args, kwargs);
    });

REGISTER_OP("simt.atomic_add")
    .set_op_category("SimtOp")
    .set_description("Atomically add to one UB Tile or GM Tensor element; FP16/BF16 return no value")
    .add_argument("container", "Destination Tile or Tensor")
    .add_argument("offset", "Linear element offset")
    .add_argument("value", "Scalar addend")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtAtomicType("simt.atomic_add", 1,
                                    {DataType::INT32, DataType::UINT32, DataType::FP16, DataType::BF16, DataType::FP32},
                                    {DataType::INT32, DataType::UINT32, DataType::FP16, DataType::BF16, DataType::FP32,
                                     DataType::INT64, DataType::UINT64},
                                    args, kwargs, {DataType::FP16, DataType::BF16});
    });

REGISTER_OP("simt.atomic_sub")
    .set_op_category("SimtOp")
    .set_description("Atomically subtract from one UB Tile or GM Tensor element and return its old value")
    .add_argument("container", "Destination Tile or Tensor")
    .add_argument("offset", "Linear element offset")
    .add_argument("value", "Scalar subtrahend")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtAtomicType(
            "simt.atomic_sub", 1, {DataType::INT32, DataType::UINT32, DataType::FP32},
            {DataType::INT32, DataType::UINT32, DataType::FP32, DataType::INT64, DataType::UINT64}, args, kwargs);
    });

REGISTER_OP("simt.atomic_exch")
    .set_op_category("SimtOp")
    .set_description("Atomically exchange one UB Tile or GM Tensor element and return its old value")
    .add_argument("container", "Destination Tile or Tensor")
    .add_argument("offset", "Linear element offset")
    .add_argument("value", "Replacement scalar")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtAtomicType(
            "simt.atomic_exch", 1, {DataType::INT32, DataType::UINT32, DataType::FP32},
            {DataType::INT32, DataType::UINT32, DataType::FP32, DataType::INT64, DataType::UINT64}, args, kwargs);
    });

REGISTER_OP("simt.atomic_max")
    .set_op_category("SimtOp")
    .set_description("Atomically maximize one UB Tile or GM Tensor element; FP16/BF16 return no value")
    .add_argument("container", "Destination Tile or Tensor")
    .add_argument("offset", "Linear element offset")
    .add_argument("value", "Candidate scalar")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtAtomicType("simt.atomic_max", 1,
                                    {DataType::INT32, DataType::UINT32, DataType::FP16, DataType::BF16, DataType::FP32},
                                    {DataType::INT32, DataType::UINT32, DataType::FP16, DataType::BF16, DataType::FP32,
                                     DataType::INT64, DataType::UINT64},
                                    args, kwargs, {DataType::FP16, DataType::BF16});
    });

REGISTER_OP("simt.atomic_min")
    .set_op_category("SimtOp")
    .set_description("Atomically minimize one UB Tile or GM Tensor element; FP16/BF16 return no value")
    .add_argument("container", "Destination Tile or Tensor")
    .add_argument("offset", "Linear element offset")
    .add_argument("value", "Candidate scalar")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtAtomicType("simt.atomic_min", 1,
                                    {DataType::INT32, DataType::UINT32, DataType::FP16, DataType::BF16, DataType::FP32},
                                    {DataType::INT32, DataType::UINT32, DataType::FP16, DataType::BF16, DataType::FP32,
                                     DataType::INT64, DataType::UINT64},
                                    args, kwargs, {DataType::FP16, DataType::BF16});
    });

REGISTER_OP("simt.atomic_inc")
    .set_op_category("SimtOp")
    .set_description("Atomically increment one wrapping counter element and return its old value")
    .add_argument("container", "Destination Tile or Tensor")
    .add_argument("offset", "Linear element offset")
    .add_argument("limit", "Inclusive wrap limit")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtAtomicType("simt.atomic_inc", 1, {DataType::UINT32}, {DataType::UINT32, DataType::UINT64},
                                    args, kwargs);
    });

REGISTER_OP("simt.atomic_dec")
    .set_op_category("SimtOp")
    .set_description("Atomically decrement one wrapping counter element and return its old value")
    .add_argument("container", "Destination Tile or Tensor")
    .add_argument("offset", "Linear element offset")
    .add_argument("limit", "Inclusive wrap limit")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtAtomicType("simt.atomic_dec", 1, {DataType::UINT32}, {DataType::UINT32, DataType::UINT64},
                                    args, kwargs);
    });

REGISTER_OP("simt.atomic_cas")
    .set_op_category("SimtOp")
    .set_description("Atomically compare and exchange one UB Tile or GM Tensor element and return its old value")
    .add_argument("container", "Destination Tile or Tensor")
    .add_argument("offset", "Linear element offset")
    .add_argument("compare", "Expected scalar")
    .add_argument("value", "Replacement scalar")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtAtomicType(
            "simt.atomic_cas", 2, {DataType::INT32, DataType::UINT32, DataType::FP32},
            {DataType::INT32, DataType::UINT32, DataType::FP32, DataType::INT64, DataType::UINT64}, args, kwargs);
    });

REGISTER_OP("simt.atomic_and")
    .set_op_category("SimtOp")
    .set_description("Atomically apply bitwise AND to one UB Tile or GM Tensor element and return its old value")
    .add_argument("container", "Destination Tile or Tensor")
    .add_argument("offset", "Linear element offset")
    .add_argument("value", "Scalar bit mask")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtAtomicType("simt.atomic_and", 1, {DataType::INT32, DataType::UINT32},
                                    {DataType::INT32, DataType::UINT32, DataType::INT64, DataType::UINT64}, args,
                                    kwargs);
    });

REGISTER_OP("simt.atomic_or")
    .set_op_category("SimtOp")
    .set_description("Atomically apply bitwise OR to one UB Tile or GM Tensor element and return its old value")
    .add_argument("container", "Destination Tile or Tensor")
    .add_argument("offset", "Linear element offset")
    .add_argument("value", "Scalar bit mask")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtAtomicType("simt.atomic_or", 1, {DataType::INT32, DataType::UINT32},
                                    {DataType::INT32, DataType::UINT32, DataType::INT64, DataType::UINT64}, args,
                                    kwargs);
    });

REGISTER_OP("simt.atomic_xor")
    .set_op_category("SimtOp")
    .set_description("Atomically apply bitwise XOR to one UB Tile or GM Tensor element and return its old value")
    .add_argument("container", "Destination Tile or Tensor")
    .add_argument("offset", "Linear element offset")
    .add_argument("value", "Scalar bit mask")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtAtomicType("simt.atomic_xor", 1, {DataType::INT32, DataType::UINT32},
                                    {DataType::INT32, DataType::UINT32, DataType::INT64, DataType::UINT64}, args,
                                    kwargs);
    });

REGISTER_OP("simt.launch")
    .set_op_category("SimtOp")
    .set_description("Launch a SIMT vector function from an AIV kernel")
    .add_argument("threads_x", "Compile-time X dimension")
    .add_argument("threads_y", "Compile-time Y dimension")
    .add_argument("threads_z", "Compile-time Z dimension")
    .set_attr<std::string>("callee")
    .set_attr<int>("max_threads")
    .f_deduce_type(DeduceSimtLaunchType);

} // namespace ir
} // namespace pypto
