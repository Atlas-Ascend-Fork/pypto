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
#include "ir/op/op_common.h"
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

TypePtr DeduceSimtSyncType(const std::string& opName, const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.empty()) << opName << " does not accept positional arguments";
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, kwargs.empty()) << opName << " does not accept keyword arguments";
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

bool IsSimtCastFromInt8OrUint8Supported(DataType targetDtype, RoundMode mode)
{
    if (IsSimtCastIntegerDtype(targetDtype)) {
        return mode == RoundMode::CAST_NONE;
    }
    return false;
}

bool IsSimtCastFromInt16OrUint16Supported(DataType targetDtype, RoundMode mode)
{
    if (IsSimtCastIntegerDtype(targetDtype)) {
        return mode == RoundMode::CAST_NONE;
    }
    if (targetDtype == DataType::FP16 || targetDtype == DataType::BF16) {
        return mode == RoundMode::CAST_RINT || mode == RoundMode::CAST_FLOOR || mode == RoundMode::CAST_CEIL ||
               mode == RoundMode::CAST_TRUNC;
    }
    return false;
}

bool IsSimtCastFromWideIntegerSupported(DataType targetDtype, RoundMode mode)
{
    if (IsSimtCastIntegerDtype(targetDtype)) {
        return mode == RoundMode::CAST_NONE;
    }
    if (targetDtype == DataType::FP16 || targetDtype == DataType::BF16) {
        return mode == RoundMode::CAST_RINT || mode == RoundMode::CAST_ROUND || mode == RoundMode::CAST_FLOOR ||
               mode == RoundMode::CAST_CEIL || mode == RoundMode::CAST_TRUNC;
    }
    if (targetDtype == DataType::FP32) {
        return IsSimtCastStandardRoundMode(mode);
    }
    return false;
}

bool IsSimtCastFromFp16OrBf16Supported(DataType targetDtype, RoundMode mode)
{
    if (targetDtype == DataType::FP32) {
        return mode == RoundMode::CAST_NONE;
    }
    if (targetDtype == DataType::FP16 || targetDtype == DataType::BF16 || targetDtype == DataType::INT32 ||
        targetDtype == DataType::INT64 || targetDtype == DataType::UINT32 || targetDtype == DataType::UINT64) {
        return mode == RoundMode::CAST_RINT || mode == RoundMode::CAST_ROUND || mode == RoundMode::CAST_FLOOR ||
               mode == RoundMode::CAST_CEIL || mode == RoundMode::CAST_TRUNC;
    }
    if (targetDtype == DataType::INT8 || targetDtype == DataType::UINT8) {
        return mode == RoundMode::CAST_TRUNC;
    }
    if (targetDtype == DataType::INT16 || targetDtype == DataType::UINT16) {
        return mode == RoundMode::CAST_RINT || mode == RoundMode::CAST_FLOOR || mode == RoundMode::CAST_CEIL ||
               mode == RoundMode::CAST_TRUNC;
    }
    return false;
}

bool IsSimtCastFromFp32Supported(DataType targetDtype, RoundMode mode)
{
    if (targetDtype == DataType::FP16) {
        return mode == RoundMode::CAST_ODD || IsSimtCastStandardRoundMode(mode);
    }
    if (targetDtype == DataType::BF16 || targetDtype == DataType::INT32 || targetDtype == DataType::INT64 ||
        targetDtype == DataType::UINT32 || targetDtype == DataType::UINT64) {
        return IsSimtCastStandardRoundMode(mode);
    }
    return false;
}

bool IsSimtCastSupported(DataType sourceDtype, DataType targetDtype, RoundMode mode)
{
    if (sourceDtype == targetDtype) {
        if (mode != RoundMode::CAST_NONE) {
            return false;
        }
        return IsSimtCastSupportedDtype(sourceDtype);
    }
    if (sourceDtype == DataType::INT8 || sourceDtype == DataType::UINT8) {
        return IsSimtCastFromInt8OrUint8Supported(targetDtype, mode);
    }
    if (sourceDtype == DataType::INT16 || sourceDtype == DataType::UINT16) {
        return IsSimtCastFromInt16OrUint16Supported(targetDtype, mode);
    }
    if (sourceDtype == DataType::INT32 || sourceDtype == DataType::UINT32 || sourceDtype == DataType::INT64 ||
        sourceDtype == DataType::UINT64) {
        return IsSimtCastFromWideIntegerSupported(targetDtype, mode);
    }
    if (sourceDtype == DataType::FP16 || sourceDtype == DataType::BF16) {
        return IsSimtCastFromFp16OrBf16Supported(targetDtype, mode);
    }
    if (sourceDtype == DataType::FP32) {
        return IsSimtCastFromFp32Supported(targetDtype, mode);
    }
    return false;
}

TypePtr DeduceSimtCastType(const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 1) << "simt.cast requires one scalar argument";
    auto sourceType = As<ScalarType>(args[0]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, sourceType) << "simt.cast value must be a scalar";

    DataType targetDtype = GetOpKwarg<DataType>(kwargs, "target_type");
    auto mode = static_cast<RoundMode>(GetOpKwarg<int>(kwargs, "mode"));
    PRO_IR_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, IsSimtCastSupported(sourceType->dtype_, targetDtype, mode))
        << "simt.cast does not support " << sourceType->dtype_.ToString() << " -> " << targetDtype.ToString()
        << " with mode " << EnumToString(mode);
    return std::make_shared<ScalarType>(targetDtype);
}

bool IsSimtBitcastSupported(DataType sourceDtype, DataType targetDtype)
{
    if (sourceDtype == DataType::FP16 || sourceDtype == DataType::BF16) {
        return targetDtype == DataType::INT16 || targetDtype == DataType::UINT16;
    }
    if (sourceDtype == DataType::INT16 || sourceDtype == DataType::UINT16) {
        return targetDtype == DataType::FP16 || targetDtype == DataType::BF16;
    }
    if (sourceDtype == DataType::FP32) {
        return targetDtype == DataType::INT32 || targetDtype == DataType::UINT32;
    }
    if (sourceDtype == DataType::INT32 || sourceDtype == DataType::UINT32) {
        return targetDtype == DataType::FP32;
    }
    return false;
}

TypePtr DeduceSimtBitcastType(const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 1) << "simt.bitcast requires one scalar argument";
    auto sourceType = As<ScalarType>(args[0]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, sourceType) << "simt.bitcast value must be a scalar";

    DataType targetDtype = GetOpKwarg<DataType>(kwargs, "target_type");
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, sourceType->dtype_.GetBit() == targetDtype.GetBit())
        << "simt.bitcast requires source and target dtypes to have the same bit width, got "
        << sourceType->dtype_.ToString() << " (" << sourceType->dtype_.GetBit() << " bits) -> "
        << targetDtype.ToString() << " (" << targetDtype.GetBit() << " bits)";
    PRO_IR_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, IsSimtBitcastSupported(sourceType->dtype_, targetDtype))
        << "simt.bitcast does not support " << sourceType->dtype_.ToString() << " -> " << targetDtype.ToString();
    return std::make_shared<ScalarType>(targetDtype);
}

bool IsDtypeInList(const DataType& dtype, std::initializer_list<DataType> supportedDtypes)
{
    for (const auto& supportedDtype : supportedDtypes) {
        if (dtype == supportedDtype) {
            return true;
        }
    }
    return false;
}

TypePtr DeduceSimtAtomicType(const std::string& opName, size_t operandCount, std::initializer_list<DataType> ubDtypes,
                             std::initializer_list<DataType> gmDtypes, const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs,
                             std::initializer_list<DataType> noResultDtypes = {})
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == operandCount + 2)
        << opName << " requires container, offset, and " << operandCount << " scalar operand(s)";
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, kwargs.empty()) << opName << " does not accept keyword arguments";

    ValidateScalarAccessIndex(args[0], args[1], opName);
    auto tileType = As<TileType>(args[0]->GetType());
    auto tensorType = As<TensorType>(args[0]->GetType());
    DataType dtype = tileType ? tileType->dtype_ : tensorType->dtype_;
    PRO_IR_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, IsDtypeInList(dtype, tileType ? ubDtypes : gmDtypes))
        << opName << " does not support dtype " << dtype.ToString() << " on " << (tileType ? "UB Tile" : "GM Tensor");

    for (size_t i = 0; i < operandCount; ++i) {
        auto operandType = As<ScalarType>(args[i + 2]->GetType());
        PRO_IR_CHECK(ExternalError::INVALID_TYPE, operandType) << opName << " operand " << i << " must be a scalar";
        PRO_IR_CHECK(ExternalError::INVALID_TYPE, operandType->dtype_ == dtype)
            << opName << " operand " << i << " dtype must match target dtype " << dtype.ToString() << ", but got "
            << operandType->dtype_.ToString();
    }
    if (IsDtypeInList(dtype, noResultDtypes)) {
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

TypePtr DeduceSimtWarpNoArgType(const std::string& opName, DataType resultDtype, const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.empty()) << opName << " does not accept positional arguments";
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, kwargs.empty()) << opName << " does not accept keyword arguments";
    return std::make_shared<ScalarType>(resultDtype);
}

TypePtr DeduceSimtWarpVoteType(const std::string& opName, DataType resultDtype, const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 1) << opName << " requires exactly one predicate";
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, kwargs.empty()) << opName << " does not accept keyword arguments";
    auto predicateType = As<ScalarType>(args[0]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, predicateType) << opName << " predicate must be a scalar";
    PRO_IR_CHECK(ExternalError::INVALID_TYPE,
                 predicateType->dtype_ == DataType::BOOL || predicateType->dtype_ == DataType::INT32)
        << opName << " predicate must have BOOL or INT32 dtype, got " << predicateType->dtype_.ToString();
    return std::make_shared<ScalarType>(resultDtype);
}

ScalarTypePtr DeduceSimtWarpShuffleCommonType(const std::string& opName, const std::string& controlName,
                                              const std::vector<ExprPtr>& args,
                                              const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 3)
        << opName << " requires value, " << controlName << ", and width";
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, kwargs.empty()) << opName << " does not accept keyword arguments";

    auto valueType = As<ScalarType>(args[0]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, valueType) << opName << " value must be a scalar";
    PRO_IR_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR,
                 IsDtypeInList(valueType->dtype_, {DataType::INT32, DataType::UINT32, DataType::INT64, DataType::UINT64,
                                                   DataType::FP16, DataType::BF16, DataType::FP32}))
        << opName << " supports only INT32, UINT32, INT64, UINT64, FP16, BF16, and FP32 values, got "
        << valueType->dtype_.ToString();

    auto widthType = As<ScalarType>(args[2]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, widthType && widthType->dtype_ == DataType::INT32)
        << opName << " width must be an INT32 scalar";
    auto width = As<ConstInt>(args[2]);
    if (width) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT,
                     width->value_ >= 1 && width->value_ <= 32 && (width->value_ & (width->value_ - 1)) == 0)
            << opName << " width must be one of 1, 2, 4, 8, 16, or 32";
    }
    return std::make_shared<ScalarType>(valueType->dtype_);
}

TypePtr DeduceSimtWarpShflType(const std::string& opName, const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    auto valueType = DeduceSimtWarpShuffleCommonType(opName, "src_lane", args, kwargs);
    auto srcLaneType = As<ScalarType>(args[1]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, srcLaneType) << opName << " src_lane must be a scalar";
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, srcLaneType->dtype_ == DataType::INT32)
        << opName << " src_lane must have " << DataType::INT32.ToString() << " dtype, got "
        << srcLaneType->dtype_.ToString();

    auto srcLane = As<ConstInt>(args[1]);
    if (srcLane) {
        PRO_IR_CHECK(ExternalError::OUT_OF_RANGE, srcLane->value_ >= 0 && srcLane->value_ < 32)
            << opName << " src_lane must be in [0, 31]";
    }
    return valueType;
}

TypePtr DeduceSimtWarpShflUpDownType(const std::string& opName, const std::vector<ExprPtr>& args,
                                     const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    auto valueType = DeduceSimtWarpShuffleCommonType(opName, "delta", args, kwargs);
    auto deltaType = As<ScalarType>(args[1]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, deltaType) << opName << " delta must be a scalar";
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, deltaType->dtype_ == DataType::UINT32)
        << opName << " delta must have " << DataType::UINT32.ToString() << " dtype, got "
        << deltaType->dtype_.ToString();

    auto delta = As<ConstInt>(args[1]);
    if (delta) {
        PRO_IR_CHECK(ExternalError::OUT_OF_RANGE, delta->value_ >= 0 && delta->value_ < 32)
            << opName << " delta must be in [0, 31]";
    }
    return valueType;
}

TypePtr DeduceSimtWarpShflXorType(const std::string& opName, const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    auto valueType = DeduceSimtWarpShuffleCommonType(opName, "lane_mask", args, kwargs);
    auto laneMaskType = As<ScalarType>(args[1]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, laneMaskType) << opName << " lane_mask must be a scalar";
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, laneMaskType->dtype_ == DataType::INT32)
        << opName << " lane_mask must have " << DataType::INT32.ToString() << " dtype, got "
        << laneMaskType->dtype_.ToString();

    auto laneMask = As<ConstInt>(args[1]);
    if (laneMask) {
        PRO_IR_CHECK(ExternalError::OUT_OF_RANGE, laneMask->value_ >= 0 && laneMask->value_ < 32)
            << opName << " lane_mask must be in [0, 31]";
        auto width = As<ConstInt>(args[2]);
        PRO_IR_CHECK(ExternalError::OUT_OF_RANGE, !width || laneMask->value_ < width->value_)
            << opName << " lane_mask must be less than width";
    }
    return valueType;
}

TypePtr DeduceSimtMathType(const std::string& opName, size_t operandCount,
                           std::initializer_list<DataType> supportedDtypes, const std::vector<ExprPtr>& args,
                           std::optional<DataType> resultDtype = std::nullopt)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == operandCount)
        << opName << " requires exactly " << operandCount << " scalar operand(s)";

    DataType dtype = DataType::BOOL;
    for (size_t i = 0; i < args.size(); ++i) {
        auto scalarType = As<ScalarType>(args[i]->GetType());
        PRO_IR_CHECK(ExternalError::INVALID_TYPE, scalarType) << opName << " operand " << i << " must be a scalar";
        if (i == 0) {
            dtype = scalarType->dtype_;
            PRO_IR_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, IsDtypeInList(dtype, supportedDtypes))
                << opName << " supports only " << FormatSupportedDtypes(supportedDtypes) << ", got "
                << dtype.ToString();
        } else {
            PRO_IR_CHECK(ExternalError::INVALID_TYPE, scalarType->dtype_ == dtype)
                << opName << " requires operands with the same dtype, got " << dtype.ToString() << " and "
                << scalarType->dtype_.ToString();
        }
    }
    return std::make_shared<ScalarType>(resultDtype.value_or(dtype));
}

TypePtr DeduceSimtRintRoundType(const std::string& opName, const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 1) << opName << " requires exactly one scalar operand";
    auto sourceType = As<ScalarType>(args[0]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, sourceType) << opName << " value must be a scalar";
    PRO_IR_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR,
                 IsDtypeInList(sourceType->dtype_, {DataType::FP16, DataType::BF16, DataType::FP32}))
        << opName << " value supports only fp16, bf16, and fp32, got " << sourceType->dtype_.ToString();

    DataType targetDtype = GetOpKwarg<DataType>(kwargs, "target_type");
    if (targetDtype == DataType::INT64) {
        PRO_IR_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, sourceType->dtype_ == DataType::FP32)
            << opName << " supports dtype=int64 only for an fp32 value, got value=" << sourceType->dtype_.ToString();
        return std::make_shared<ScalarType>(DataType::INT64);
    }
    PRO_IR_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, targetDtype == sourceType->dtype_)
        << opName << " does not support value=" << sourceType->dtype_.ToString()
        << " with dtype=" << targetDtype.ToString();
    return std::make_shared<ScalarType>(targetDtype);
}

TypePtr DeduceSimtLaunchType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_SHAPE, args.size() >= 3) << "simt.launch requires three launch dimensions";
    int64_t totalThreads = 1;
    for (std::size_t i = 0; i < 3; ++i) {
        auto dimType = As<ScalarType>(args[i]->GetType());
        PRO_IR_CHECK(ExternalError::INVALID_SHAPE,
                     dimType && dimType->dtype_.IsInt() && dimType->dtype_ != DataType(DataType::BOOL))
            << "simt.launch dimensions must be non-bool integer scalars";
        auto dim = As<ConstInt>(args[i]);
        PRO_IR_CHECK(ExternalError::INVALID_SHAPE, dim && dim->value_ > 0 && dim->value_ <= 2048)
            << "simt.launch dimensions must be compile-time integers in [1, 2048], got dim[" << i
            << "]=" << (dim ? dim->value_ : -1);
        totalThreads *= dim->value_;
    }
    PRO_IR_CHECK(ExternalError::OUT_OF_RANGE, totalThreads <= 2048)
        << "simt.launch total thread count must not exceed 2048, got total_threads=" << totalThreads;
    int maxThreads = GetOpKwarg<int>(kwargs, "max_threads");
    PRO_IR_CHECK(ExternalError::OUT_OF_RANGE, maxThreads >= 1 && maxThreads <= 2048)
        << "simt.launch max_threads must be in [1, 2048]";
    PRO_IR_CHECK(ExternalError::OUT_OF_RANGE, totalThreads <= maxThreads)
        << "simt.launch threads " << totalThreads << " exceed callee max_threads " << maxThreads;
    for (std::size_t i = 3; i < args.size(); ++i) {
        auto argType = args[i]->GetType();
        if (auto tileType = As<TileType>(argType)) {
            PRO_IR_CHECK(ExternalError::INVALID_SHAPE, tileType->shape_.size() == 2)
                << "simt.launch Tile argument must have a two-dimensional shape";
            PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<Var>(args[i]) != nullptr)
                << "simt.launch Tile argument must be a Var";
            for (const auto& dim : tileType->shape_) {
                PRO_IR_CHECK(ExternalError::DYNAMIC_SHAPE_COMPUTE_UNSUPPORTED, As<ConstInt>(dim) != nullptr)
                    << "simt.launch Tile argument must have a static shape";
            }
            PRO_IR_CHECK(ExternalError::INVALID_TYPE, tileType->dtype_.GetBit() >= 8)
                << "simt.launch Tile argument has a sub-byte element dtype";
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, tileType->memref_.has_value())
                << "simt.launch Tile argument has no memory reference";
            PRO_IR_CHECK(ExternalError::INVALID_TYPE, (*tileType->memref_)->memorySpace_ == MemorySpace::Vec)
                << "simt.launch Tile argument must be a Vec-memory Tile";
            PRO_IR_CHECK(ExternalError::INVALID_FORMAT, tileType->hardwareInfo_.has_value() &&
                                                            tileType->hardwareInfo_->blayout == TileLayout::row_major &&
                                                            tileType->hardwareInfo_->slayout == TileLayout::none_box)
                << "simt.launch Tile argument must be an ND Vec Tile";
        } else if (auto tensorType = As<TensorType>(argType)) {
            PRO_IR_CHECK(ExternalError::INVALID_SHAPE, !tensorType->shape_.empty())
                << "simt.launch Tensor argument must have a non-empty shape";
            for (const auto& dim : tensorType->shape_) {
                PRO_IR_CHECK(ExternalError::DYNAMIC_SHAPE_COMPUTE_UNSUPPORTED, As<ConstInt>(dim) != nullptr)
                    << "simt.launch Tensor argument must have a static shape";
            }
            PRO_IR_CHECK(ExternalError::INVALID_TYPE, tensorType->dtype_.GetBit() >= 8)
                << "simt.launch Tensor argument has a sub-byte element dtype";
            PRO_IR_CHECK(ExternalError::INVALID_FORMAT,
                         tensorType->tensor_view_.has_value() && tensorType->tensor_view_->layout == Layout::ND)
                << "simt.launch Tensor argument must use ND layout";
            PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<Var>(args[i]) != nullptr)
                << "simt.launch Tensor argument must be a Var";
        } else {
            // Compiler passes may append internal GM pointers (e.g. sanitizer logs).
            PRO_IR_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, As<ScalarType>(argType) || As<PtrType>(argType))
                << "simt.launch supports only Tile, Tensor, pointer, and scalar arguments";
        }
    }
    return GetUnknownType();
}

} // namespace

TypePtr DeduceSimtSadType(const std::vector<ExprPtr>& args,
                          const std::vector<std::pair<std::string, std::any>>& /*kwargs*/)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 3) << "simt.sad requires exactly 3 scalar operands";
    auto lhsType = As<ScalarType>(args[0]->GetType());
    auto rhsType = As<ScalarType>(args[1]->GetType());
    auto addendType = As<ScalarType>(args[2]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, lhsType) << "simt.sad lhs must be a scalar";
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, rhsType) << "simt.sad rhs must be a scalar";
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, addendType) << "simt.sad addend must be a scalar";
    PRO_IR_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR,
                 lhsType->dtype_ == DataType::INT32 || lhsType->dtype_ == DataType::UINT32)
        << "simt.sad lhs supports only int32, uint32, got " << lhsType->dtype_.ToString();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, rhsType->dtype_ == lhsType->dtype_)
        << "simt.sad lhs and rhs must have the same dtype, got lhs=" << lhsType->dtype_.ToString()
        << ", rhs=" << rhsType->dtype_.ToString();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, addendType->dtype_ == DataType::UINT32)
        << "simt.sad addend must be uint32, got " << addendType->dtype_.ToString();
    return std::make_shared<ScalarType>(DataType::UINT32);
}

TypePtr DeduceSimtMathSignature(const std::string& opName, std::initializer_list<DataType> operandDtypes,
                                DataType resultDtype, const std::vector<ExprPtr>& args)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == operandDtypes.size())
        << opName << " requires exactly " << operandDtypes.size() << " scalar operand(s)";
    size_t index = 0;
    for (const auto& expectedDtype : operandDtypes) {
        auto scalarType = As<ScalarType>(args[index]->GetType());
        PRO_IR_CHECK(ExternalError::INVALID_TYPE, scalarType) << opName << " operand " << index << " must be a scalar";
        PRO_IR_CHECK(ExternalError::INVALID_TYPE, scalarType->dtype_ == expectedDtype)
            << opName << " operand " << index << " must be " << expectedDtype.ToString() << ", got "
            << scalarType->dtype_.ToString();
        ++index;
    }
    return std::make_shared<ScalarType>(resultDtype);
}

TypePtr DeduceSimtScalbnType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& /*kwargs*/)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 2) << "simt.scalbn requires exactly 2 scalar operands";
    auto valueType = As<ScalarType>(args[0]->GetType());
    auto exponentType = As<ScalarType>(args[1]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, valueType) << "simt.scalbn value must be a scalar";
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, exponentType) << "simt.scalbn exponent must be a scalar";
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, valueType->dtype_ == DataType::FP32)
        << "simt.scalbn value must be fp32, got " << valueType->dtype_.ToString();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE,
                 exponentType->dtype_ == DataType::INT32 || exponentType->dtype_ == DataType::INT64)
        << "simt.scalbn exponent must be int32 or int64, got " << exponentType->dtype_.ToString();
    return std::make_shared<ScalarType>(DataType::FP32);
}

TypePtr DeduceSimtMulI32ToI64Type(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& /*kwargs*/)
{
    DeduceSimtMathType("simt.mul_i32_to_i64", 2, {DataType::INT32, DataType::UINT32}, args);
    auto inputType = As<ScalarType>(args[0]->GetType());
    DataType resultDtype = inputType->dtype_ == DataType::INT32 ? DataType::INT64 : DataType::UINT64;
    return std::make_shared<ScalarType>(resultDtype);
}

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

#define REGISTER_SIMT_WARP_REDUCE_OP(OpName, Description)                                                             \
    REGISTER_OP("simt." OpName)                                                                                       \
        .set_op_category("SimtOp")                                                                                    \
        .set_description(Description)                                                                                 \
        .add_argument("value", "Scalar value to reduce")                                                              \
        .f_deduce_type(                                                                                               \
            [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {   \
                return DeduceSimtMathType("simt." OpName, 1,                                                          \
                                          {DataType::INT32, DataType::UINT32, DataType::FP16, DataType::FP32}, args); \
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

#define REGISTER_SIMT_MATH_UNARY_OP(OpName, Description, ResultDtype, ...)                                          \
    REGISTER_OP("simt." OpName)                                                                                     \
        .set_op_category("SimtOp")                                                                                  \
        .set_description(Description)                                                                               \
        .add_argument("value", "Scalar operand")                                                                    \
        .f_deduce_type(                                                                                             \
            [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) { \
                return DeduceSimtMathType("simt." OpName, 1, {__VA_ARGS__}, args, ResultDtype);                     \
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
REGISTER_SIMT_MATH_UNARY_OP("brev", "Reverse all bits of one UINT32 or UINT64 scalar", std::nullopt, DataType::UINT32,
                            DataType::UINT64);

REGISTER_SIMT_MATH_UNARY_OP("clz", "Count leading zeros in one INT32 or INT64 scalar", DataType::INT32, DataType::INT32,
                            DataType::INT64);

REGISTER_SIMT_MATH_UNARY_OP("ffs", "Find the first set bit in one INT32 or INT64 scalar", DataType::INT32,
                            DataType::INT32, DataType::INT64);

#undef REGISTER_SIMT_MATH_UNARY_OP

REGISTER_OP("simt.rint")
    .set_op_category("SimtOp")
    .set_description("Round one floating-point scalar to nearest-even with a selectable result dtype")
    .add_argument("value", "Floating-point scalar operand")
    .set_attr<DataType>("target_type")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtRintRoundType("simt.rint", args, kwargs);
    });

REGISTER_OP("simt.round")
    .set_op_category("SimtOp")
    .set_description("Round one floating-point scalar halfway away from zero with a selectable result dtype")
    .add_argument("value", "Floating-point scalar operand")
    .set_attr<DataType>("target_type")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtRintRoundType("simt.round", args, kwargs);
    });

REGISTER_OP("simt.popcount")
    .set_op_category("SimtOp")
    .set_description("Count set bits in a UINT32 or UINT64 scalar, returning INT32")
    .add_argument("value", "Unsigned scalar operand")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
        return DeduceSimtMathType("simt.popcount", 1, {DataType::UINT32, DataType::UINT64}, args, DataType::INT32);
    });

REGISTER_OP("simt.mul_hi")
    .set_op_category("SimtOp")
    .set_description("Compute the high half of the full product of two same-dtype integers")
    .add_argument("lhs", "Left scalar operand")
    .add_argument("rhs", "Right scalar operand")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
        return DeduceSimtMathType("simt.mul_hi", 2,
                                  {DataType::INT32, DataType::UINT32, DataType::INT64, DataType::UINT64}, args);
    });

REGISTER_OP("simt.max_nan")
    .set_op_category("SimtOp")
    .set_description("Select the maximum FP16 or BF16 scalar and propagate NaN")
    .add_argument("lhs", "Left scalar operand")
    .add_argument("rhs", "Right scalar operand")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
        return DeduceSimtMathType("simt.max_nan", 2, {DataType::FP16, DataType::BF16}, args);
    });

REGISTER_OP("simt.min_nan")
    .set_op_category("SimtOp")
    .set_description("Select the minimum FP16 or BF16 scalar and propagate NaN")
    .add_argument("lhs", "Left scalar operand")
    .add_argument("rhs", "Right scalar operand")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
        return DeduceSimtMathType("simt.min_nan", 2, {DataType::FP16, DataType::BF16}, args);
    });

REGISTER_OP("simt.atan2")
    .set_op_category("SimtOp")
    .set_description("Compute the quadrant-aware arc tangent of two FP32 scalars")
    .add_argument("y", "Y-coordinate scalar operand")
    .add_argument("x", "X-coordinate scalar operand")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
        return DeduceSimtMathType("simt.atan2", 2, {DataType::FP32}, args);
    });

REGISTER_OP("simt.copysign")
    .set_op_category("SimtOp")
    .set_description("Select the magnitude and sign of two FP32 scalars")
    .add_argument("magnitude", "Magnitude scalar operand")
    .add_argument("sign", "Sign scalar operand")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
        return DeduceSimtMathType("simt.copysign", 2, {DataType::FP32}, args);
    });

REGISTER_OP("simt.nextafter")
    .set_op_category("SimtOp")
    .set_description("Return the next FP32 value toward another FP32 scalar")
    .add_argument("value", "Starting input scalar value")
    .add_argument("direction", "Input scalar value to move toward")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
        return DeduceSimtMathType("simt.nextafter", 2, {DataType::FP32}, args);
    });

REGISTER_OP("simt.byte_perm")
    .set_op_category("SimtOp")
    .set_description("Select four bytes from two UINT32 scalars according to a UINT32 selector")
    .add_argument("lhs", "Low four source bytes")
    .add_argument("rhs", "High four source bytes")
    .add_argument("selector", "Byte selector")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
        return DeduceSimtMathType("simt.byte_perm", 3, {DataType::UINT32}, args);
    });

REGISTER_OP("simt.equ")
    .set_op_category("SimtOp")
    .set_description("Compare equal or unordered FP16 or BF16 scalars")
    .add_argument("lhs", "Left scalar operand")
    .add_argument("rhs", "Right scalar operand")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
        return DeduceSimtMathType("simt.equ", 2, {DataType::FP16, DataType::BF16}, args, DataType::BOOL);
    });

REGISTER_OP("simt.fma_relu")
    .set_op_category("SimtOp")
    .set_description("Fused-multiply-add three FP16 or BF16 scalars and clamp negative results to zero")
    .add_argument("lhs", "Left multiplication operand")
    .add_argument("rhs", "Right multiplication operand")
    .add_argument("addend", "Addition operand")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
        return DeduceSimtMathType("simt.fma_relu", 3, {DataType::FP16, DataType::BF16}, args);
    });

REGISTER_OP("simt.geu")
    .set_op_category("SimtOp")
    .set_description("Compare greater-or-equal or unordered FP16 or BF16 scalars")
    .add_argument("lhs", "Left scalar operand")
    .add_argument("rhs", "Right scalar operand")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
        return DeduceSimtMathType("simt.geu", 2, {DataType::FP16, DataType::BF16}, args, DataType::BOOL);
    });

REGISTER_OP("simt.gtu")
    .set_op_category("SimtOp")
    .set_description("Compare greater-than or unordered FP16 or BF16 scalars")
    .add_argument("lhs", "Left scalar operand")
    .add_argument("rhs", "Right scalar operand")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
        return DeduceSimtMathType("simt.gtu", 2, {DataType::FP16, DataType::BF16}, args, DataType::BOOL);
    });

REGISTER_OP("simt.hadd")
    .set_op_category("SimtOp")
    .set_description("Average two same-dtype 32-bit integers without intermediate overflow")
    .add_argument("lhs", "Left scalar operand")
    .add_argument("rhs", "Right scalar operand")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
        return DeduceSimtMathType("simt.hadd", 2, {DataType::INT32, DataType::UINT32}, args);
    });

REGISTER_OP("simt.jn")
    .set_op_category("SimtOp")
    .set_description("Compute an integer-order FP32 Bessel function of the first kind")
    .add_argument("order", "Nonnegative INT32 order")
    .add_argument("value", "FP32 scalar operand")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
        return DeduceSimtMathSignature("simt.jn", {DataType::INT32, DataType::FP32}, DataType::FP32, args);
    });

REGISTER_OP("simt.ldexp")
    .set_op_category("SimtOp")
    .set_description("Scale an FP32 scalar by two raised to an INT32 exponent")
    .add_argument("value", "FP32 scalar operand")
    .add_argument("exponent", "INT32 exponent")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
        return DeduceSimtMathSignature("simt.ldexp", {DataType::FP32, DataType::INT32}, DataType::FP32, args);
    });

REGISTER_OP("simt.leu")
    .set_op_category("SimtOp")
    .set_description("Compare less-or-equal or unordered FP16 or BF16 scalars")
    .add_argument("lhs", "Left scalar operand")
    .add_argument("rhs", "Right scalar operand")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
        return DeduceSimtMathType("simt.leu", 2, {DataType::FP16, DataType::BF16}, args, DataType::BOOL);
    });

REGISTER_OP("simt.ltu")
    .set_op_category("SimtOp")
    .set_description("Compare less-than or unordered FP16 or BF16 scalars")
    .add_argument("lhs", "Left scalar operand")
    .add_argument("rhs", "Right scalar operand")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
        return DeduceSimtMathType("simt.ltu", 2, {DataType::FP16, DataType::BF16}, args, DataType::BOOL);
    });

REGISTER_OP("simt.mul24")
    .set_op_category("SimtOp")
    .set_description("Multiply the low 24 bits of two same-dtype 32-bit integers")
    .add_argument("lhs", "Left scalar operand")
    .add_argument("rhs", "Right scalar operand")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
        return DeduceSimtMathType("simt.mul24", 2, {DataType::INT32, DataType::UINT32}, args);
    });

REGISTER_OP("simt.mul_i32_to_i64")
    .set_op_category("SimtOp")
    .set_description("Multiply two same-dtype 32-bit integers and return the full 64-bit product")
    .add_argument("lhs", "Left scalar operand")
    .add_argument("rhs", "Right scalar operand")
    .f_deduce_type(DeduceSimtMulI32ToI64Type);

REGISTER_OP("simt.norm3d")
    .set_op_category("SimtOp")
    .set_description("Compute the Euclidean norm of three FP32 scalars")
    .add_argument("x", "First FP32 component")
    .add_argument("y", "Second FP32 component")
    .add_argument("z", "Third FP32 component")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
        return DeduceSimtMathType("simt.norm3d", 3, {DataType::FP32}, args);
    });

REGISTER_OP("simt.norm4d")
    .set_op_category("SimtOp")
    .set_description("Compute the Euclidean norm of four FP32 scalars")
    .add_argument("x", "First FP32 component")
    .add_argument("y", "Second FP32 component")
    .add_argument("z", "Third FP32 component")
    .add_argument("w", "Fourth FP32 component")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
        return DeduceSimtMathType("simt.norm4d", 4, {DataType::FP32}, args);
    });

REGISTER_OP("simt.rhadd")
    .set_op_category("SimtOp")
    .set_description("Compute the upward-rounded average of two same-dtype 32-bit integers")
    .add_argument("lhs", "Left scalar operand")
    .add_argument("rhs", "Right scalar operand")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
        return DeduceSimtMathType("simt.rhadd", 2, {DataType::INT32, DataType::UINT32}, args);
    });

REGISTER_OP("simt.rnorm3d")
    .set_op_category("SimtOp")
    .set_description("Compute the reciprocal norm of three FP32 scalars")
    .add_argument("x", "First FP32 component")
    .add_argument("y", "Second FP32 component")
    .add_argument("z", "Third FP32 component")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
        return DeduceSimtMathType("simt.rnorm3d", 3, {DataType::FP32}, args);
    });

REGISTER_OP("simt.rnorm4d")
    .set_op_category("SimtOp")
    .set_description("Compute the reciprocal norm of four FP32 scalars")
    .add_argument("x", "First FP32 component")
    .add_argument("y", "Second FP32 component")
    .add_argument("z", "Third FP32 component")
    .add_argument("w", "Fourth FP32 component")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
        return DeduceSimtMathType("simt.rnorm4d", 4, {DataType::FP32}, args);
    });

REGISTER_OP("simt.sad")
    .set_op_category("SimtOp")
    .set_description("Add a UINT32 scalar to the absolute difference of two INT32 or UINT32 scalars")
    .add_argument("lhs", "Left difference operand")
    .add_argument("rhs", "Right difference operand")
    .add_argument("addend", "UINT32 addend")
    .f_deduce_type(DeduceSimtSadType);

REGISTER_OP("simt.scalbn")
    .set_op_category("SimtOp")
    .set_description("Scale an FP32 scalar by two raised to an INT32 or INT64 exponent")
    .add_argument("value", "FP32 scalar operand")
    .add_argument("exponent", "INT32 or INT64 exponent")
    .f_deduce_type(DeduceSimtScalbnType);

REGISTER_OP("simt.yn")
    .set_op_category("SimtOp")
    .set_description("Compute an integer-order FP32 Bessel function of the second kind")
    .add_argument("order", "Nonnegative INT32 order")
    .add_argument("value", "Positive FP32 scalar operand")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
        return DeduceSimtMathSignature("simt.yn", {DataType::INT32, DataType::FP32}, DataType::FP32, args);
    });

#define REGISTER_SIMT_FP32_UNARY_OP(OpName, Description)                                                            \
    REGISTER_OP("simt." OpName)                                                                                     \
        .set_op_category("SimtOp")                                                                                  \
        .set_description(Description)                                                                               \
        .add_argument("value", "FP32 scalar operand")                                                               \
        .f_deduce_type(                                                                                             \
            [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) { \
                return DeduceSimtMathType("simt." OpName, 1, {DataType::FP32}, args);                               \
            })

REGISTER_SIMT_FP32_UNARY_OP("cyl_bessel_i0", "Compute the order-zero modified cylindrical Bessel function");

REGISTER_SIMT_FP32_UNARY_OP("cyl_bessel_i1", "Compute the order-one modified cylindrical Bessel function");

REGISTER_SIMT_FP32_UNARY_OP("erf", "Compute the error function of an FP32 scalar");

REGISTER_SIMT_FP32_UNARY_OP("erfc", "Compute the complementary error function of an FP32 scalar");

REGISTER_SIMT_FP32_UNARY_OP("erfcinv", "Compute the inverse complementary error function of an FP32 scalar");

REGISTER_SIMT_FP32_UNARY_OP("erfcx", "Compute the scaled complementary error function of an FP32 scalar");

REGISTER_SIMT_FP32_UNARY_OP("erfinv", "Compute the inverse error function of an FP32 scalar");

REGISTER_SIMT_FP32_UNARY_OP("j0", "Compute the order-zero Bessel function of the first kind");

REGISTER_SIMT_FP32_UNARY_OP("j1", "Compute the order-one Bessel function of the first kind");

REGISTER_SIMT_FP32_UNARY_OP("lgamma", "Compute log absolute Gamma of an FP32 scalar");

REGISTER_SIMT_FP32_UNARY_OP("normcdf", "Compute the standard normal cumulative distribution of an FP32 scalar");

REGISTER_SIMT_FP32_UNARY_OP("normcdfinv", "Compute the inverse standard normal cumulative distribution");

REGISTER_SIMT_FP32_UNARY_OP("saturate", "Clamp an FP32 scalar to the unit interval and map NaN to zero");

REGISTER_SIMT_FP32_UNARY_OP("tgamma", "Compute the Gamma function of an FP32 scalar");

REGISTER_SIMT_FP32_UNARY_OP("y0", "Compute the order-zero Bessel function of the second kind");

REGISTER_SIMT_FP32_UNARY_OP("y1", "Compute the order-one Bessel function of the second kind");

#undef REGISTER_SIMT_FP32_UNARY_OP

#define REGISTER_SIMT_FP32_BINARY_OP(OpName, Description)                                                           \
    REGISTER_OP("simt." OpName)                                                                                     \
        .set_op_category("SimtOp")                                                                                  \
        .set_description(Description)                                                                               \
        .add_argument("lhs", "Left scalar operand")                                                                 \
        .add_argument("rhs", "Right scalar operand")                                                                \
        .f_deduce_type(                                                                                             \
            [](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) { \
                return DeduceSimtMathType("simt." OpName, 2, {DataType::FP32}, args);                               \
            })

REGISTER_SIMT_FP32_BINARY_OP("fdim", "Compute the positive difference of two FP32 scalars");

REGISTER_SIMT_FP32_BINARY_OP("hypot", "Compute an overflow-resistant FP32 hypotenuse");

REGISTER_SIMT_FP32_BINARY_OP("pow", "Raise an FP32 base to an FP32 exponent");

REGISTER_SIMT_FP32_BINARY_OP("remainder", "Compute the FP32 remainder using a nearest-even quotient");

REGISTER_SIMT_FP32_BINARY_OP("rhypot", "Compute the reciprocal of an FP32 hypotenuse");

#undef REGISTER_SIMT_FP32_BINARY_OP

REGISTER_OP("simt.fmod")
    .set_op_category("SimtOp")
    .set_description("Compute an FP32 remainder with a quotient truncated toward zero")
    .add_argument("lhs", "Dividend")
    .add_argument("rhs", "Divisor")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
        return DeduceSimtMathType("simt.fmod", 2, {DataType::FP32}, args);
    });

REGISTER_OP("simt.min")
    .set_op_category("SimtOp")
    .set_description("Compute the minimum of two same-dtype supported scalars")
    .add_argument("lhs", "Left scalar operand")
    .add_argument("rhs", "Right scalar operand")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
        return DeduceSimtMathType(
            "simt.min", 2,
            {DataType::FP16, DataType::BF16, DataType::FP32, DataType::INT8, DataType::INT16, DataType::INT32,
             DataType::INT64, DataType::UINT8, DataType::UINT16, DataType::UINT32, DataType::UINT64},
            args);
    });

REGISTER_OP("simt.max")
    .set_op_category("SimtOp")
    .set_description("Compute the maximum of two same-dtype supported scalars")
    .add_argument("lhs", "Left scalar operand")
    .add_argument("rhs", "Right scalar operand")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
        return DeduceSimtMathType(
            "simt.max", 2,
            {DataType::FP16, DataType::BF16, DataType::FP32, DataType::INT8, DataType::INT16, DataType::INT32,
             DataType::INT64, DataType::UINT8, DataType::UINT16, DataType::UINT32, DataType::UINT64},
            args);
    });

REGISTER_OP("simt.fma")
    .set_op_category("SimtOp")
    .set_description("Compute a fused multiply-add of three same-dtype floating-point scalars")
    .add_argument("lhs", "Left multiplication operand")
    .add_argument("rhs", "Right multiplication operand")
    .add_argument("addend", "Scalar addend")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& /*kwargs*/) {
        return DeduceSimtMathType("simt.fma", 3, {DataType::FP16, DataType::BF16, DataType::FP32}, args);
    });

REGISTER_OP("simt.atomic_add")
    .set_op_category("SimtOp")
    .set_description("Atomically add to one UB Tile or GM Tensor element; FP16/BF16 return no value")
    .add_argument("container", "Destination Tile or Tensor")
    .add_argument("offset", "Integer element offset or tuple of integer coordinates")
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
    .add_argument("offset", "Integer element offset or tuple of integer coordinates")
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
    .add_argument("offset", "Integer element offset or tuple of integer coordinates")
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
    .add_argument("offset", "Integer element offset or tuple of integer coordinates")
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
    .add_argument("offset", "Integer element offset or tuple of integer coordinates")
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
    .add_argument("offset", "Integer element offset or tuple of integer coordinates")
    .add_argument("limit", "Inclusive wrap limit")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtAtomicType("simt.atomic_inc", 1, {DataType::UINT32}, {DataType::UINT32, DataType::UINT64},
                                    args, kwargs);
    });

REGISTER_OP("simt.atomic_dec")
    .set_op_category("SimtOp")
    .set_description("Atomically decrement one wrapping counter element and return its old value")
    .add_argument("container", "Destination Tile or Tensor")
    .add_argument("offset", "Integer element offset or tuple of integer coordinates")
    .add_argument("limit", "Inclusive wrap limit")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSimtAtomicType("simt.atomic_dec", 1, {DataType::UINT32}, {DataType::UINT32, DataType::UINT64},
                                    args, kwargs);
    });

REGISTER_OP("simt.atomic_cas")
    .set_op_category("SimtOp")
    .set_description("Atomically compare and exchange one UB Tile or GM Tensor element and return its old value")
    .add_argument("container", "Destination Tile or Tensor")
    .add_argument("offset", "Integer element offset or tuple of integer coordinates")
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
    .add_argument("offset", "Integer element offset or tuple of integer coordinates")
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
    .add_argument("offset", "Integer element offset or tuple of integer coordinates")
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
    .add_argument("offset", "Integer element offset or tuple of integer coordinates")
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
