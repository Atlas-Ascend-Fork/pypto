/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <any>
#include <cctype>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/any_cast.h"
#include "core/dtype.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/memory_space.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"
#include "pypto_pro/error.h"

namespace pypto {
namespace ir {
using npu::tile_fwk::ExternalError;

namespace {

bool IsSupportedPrintfConversion(char conversion)
{
    return conversion == 'd' || conversion == 'i' || conversion == 'u' || conversion == 'x' || conversion == 'f' ||
           conversion == 'p';
}

std::vector<char> ParsePrintfConversions(const std::string& format)
{
    std::vector<char> conversions;
    size_t i = 0;
    while (i < format.size()) {
        if (format[i] != '%') {
            ++i;
            continue;
        }
        if (i + 1 < format.size() && format[i + 1] == '%') {
            PRO_IR_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, false) << "debug.printf does not support literal '%%'";
        }

        size_t j = i + 1;
        while (j < format.size()) {
            char c = format[j];
            if (c == '-' || c == '+' || c == ' ' || c == '#' || c == '0') {
                ++j;
            } else {
                break;
            }
        }
        while (j < format.size() && std::isdigit(static_cast<unsigned char>(format[j]))) {
            ++j;
        }
        if (j < format.size() && format[j] == '.') {
            ++j;
            PRO_IR_CHECK(ExternalError::INVALID_VAL,
                         j < format.size() && std::isdigit(static_cast<unsigned char>(format[j])))
                << "debug.printf precision must be followed by digits";
            while (j < format.size() && std::isdigit(static_cast<unsigned char>(format[j]))) {
                ++j;
            }
        }

        PRO_IR_CHECK(ExternalError::INVALID_VAL, j < format.size())
            << "debug.printf format ends with an incomplete conversion";
        char conversion = format[j];
        PRO_IR_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, IsSupportedPrintfConversion(conversion))
            << "debug.printf does not support conversion '%" << conversion << "'";
        conversions.push_back(conversion);
        i = j + 1;
    }

    return conversions;
}

[[maybe_unused]] bool IsPrintfIntegerType(const DataType& dtype)
{
    return dtype == DataType::INT8 || dtype == DataType::INT16 || dtype == DataType::INT32 ||
           dtype == DataType::INT64 || dtype == DataType::UINT8 || dtype == DataType::UINT16 ||
           dtype == DataType::UINT32 || dtype == DataType::UINT64;
}

bool IsPrintfSignedIntegerType(const DataType& dtype)
{
    return dtype == DataType::INT8 || dtype == DataType::INT16 || dtype == DataType::INT32 || dtype == DataType::INT64;
}

bool IsPrintfUnsignedIntegerType(const DataType& dtype)
{
    return dtype == DataType::UINT8 || dtype == DataType::UINT16 || dtype == DataType::UINT32 ||
           dtype == DataType::UINT64;
}

bool IsPrintfIndexType(const DataType& dtype) { return dtype == DataType::INDEX; }

bool IsPrintfBoolType(const DataType& dtype) { return dtype == DataType::BOOL; }

template <typename InputType>
TypePtr DeduceDebugDumpType(const std::vector<ExprPtr>& args, const std::string& op_name,
                            const std::string& input_type_name)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 1 || args.size() == 3)
        << op_name << " requires 1 argument (input) or 3 arguments (input, offsets, shapes), but got " << args.size();

    auto input_type = As<InputType>(args[0]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, input_type)
        << op_name << " requires first argument to be a " << input_type_name << ", but got "
        << args[0]->GetType()->TypeName();

    if (args.size() == 1) {
        return GetUnknownType();
    }

    auto offsets = As<MakeTuple>(args[1]);
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, offsets) << op_name << " requires offsets to be a MakeTuple";

    auto shapes = As<MakeTuple>(args[2]);
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, shapes) << op_name << " requires shapes to be a MakeTuple";

    const size_t rank = input_type->shape_.size();
    PRO_IR_CHECK(ExternalError::INVALID_SHAPE, offsets->elements_.size() == rank && shapes->elements_.size() == rank)
        << op_name << " offsets/shapes must match input rank " << rank << ", got " << offsets->elements_.size()
        << " offsets and " << shapes->elements_.size() << " shapes";

    return GetUnknownType();
}

TypePtr DeduceDebugPrintfType([[maybe_unused]] const std::vector<ExprPtr>& args,
                              [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    bool found_format = false;
    std::string format;
    for (const auto& [key, value] : kwargs) {
        if (key == "format") {
            format = AnyCast<std::string>(value, "kwarg key: format");
            found_format = true;
            break;
        }
    }
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, found_format) << "debug.printf requires 'format' kwarg";

    auto conversions = ParsePrintfConversions(format);
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, conversions.size() == args.size())
        << "debug.printf format expects " << conversions.size() << " scalar arguments, but got " << args.size();

    for (size_t i = 0; i < args.size(); ++i) {
        char conversion = conversions[i];
        if (conversion == 'p') {
            auto ptr_type = As<PtrType>(args[i]->GetType());
            PRO_IR_CHECK(ExternalError::INVALID_TYPE, ptr_type)
                << "debug.printf argument " << i << " with '%p' must be PtrType, but got "
                << args[i]->GetType()->TypeName();
            continue;
        }
        auto scalar_type = As<ScalarType>(args[i]->GetType());
        PRO_IR_CHECK(ExternalError::INVALID_TYPE, scalar_type)
            << "debug.printf argument " << i << " must be ScalarType, but got " << args[i]->GetType()->TypeName();

        const DataType& dtype = scalar_type->dtype_;
        if (conversion == 'f') {
            PRO_IR_CHECK(ExternalError::INVALID_TYPE, dtype == DataType::FP32)
                << "debug.printf conversion '%f' requires FP32 scalar, but got " << dtype.ToString();
        } else if (conversion == 'x') {
            PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsPrintfUnsignedIntegerType(dtype) || IsPrintfIndexType(dtype))
                << "debug.printf conversion '%" << conversion << "' requires unsigned integer or index scalar, but got "
                << dtype.ToString();
        } else if (conversion == 'u') {
            PRO_IR_CHECK(ExternalError::INVALID_TYPE,
                         IsPrintfUnsignedIntegerType(dtype) || IsPrintfBoolType(dtype) || IsPrintfIndexType(dtype))
                << "debug.printf conversion '%" << conversion
                << "' requires unsigned integer, bool, or index scalar, but got " << dtype.ToString();
        } else {
            PRO_IR_CHECK(ExternalError::INVALID_TYPE,
                         IsPrintfSignedIntegerType(dtype) || IsPrintfBoolType(dtype) || IsPrintfIndexType(dtype))
                << "debug.printf conversion '%" << conversion
                << "' requires signed integer, bool, or index scalar, but got " << dtype.ToString();
        }
    }

    return GetUnknownType();
}

TypePtr DeduceDebugAssertType([[maybe_unused]] const std::vector<ExprPtr>& args,
                              [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() >= 1)
        << "debug.assert requires at least 1 argument (condition), but got " << args.size();

    bool found_condition_text = false;
    bool found_format = false;
    std::string condition_text;
    std::string format;
    for (const auto& [key, value] : kwargs) {
        if (key == "condition_text") {
            condition_text = AnyCast<std::string>(value, "kwarg key: condition_text");
            found_condition_text = true;
        } else if (key == "format") {
            format = AnyCast<std::string>(value, "kwarg key: format");
            found_format = true;
        }
    }
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, found_condition_text)
        << "debug.assert requires 'condition_text' kwarg";
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, found_format) << "debug.assert requires 'format' kwarg";
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, !condition_text.empty())
        << "debug.assert requires non-empty condition_text";

    auto scalar_type = As<ScalarType>(args[0]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, scalar_type)
        << "debug.assert requires condition to be ScalarType, but got " << args[0]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, scalar_type->dtype_ == DataType::BOOL)
        << "debug.assert requires bool scalar condition, but got " << scalar_type->dtype_.ToString();

    auto conversions = ParsePrintfConversions(format);
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, conversions.size() == args.size() - 1)
        << "debug.assert format expects " << conversions.size() << " scalar arguments, but got " << (args.size() - 1);

    for (size_t i = 1; i < args.size(); ++i) {
        auto printf_arg = As<ScalarType>(args[i]->GetType());
        PRO_IR_CHECK(ExternalError::INVALID_TYPE, printf_arg)
            << "debug.assert argument " << (i - 1) << " must be ScalarType, but got " << args[i]->GetType()->TypeName();

        const DataType& dtype = printf_arg->dtype_;
        char conversion = conversions[i - 1];
        if (conversion == 'f') {
            PRO_IR_CHECK(ExternalError::INVALID_TYPE, dtype == DataType::FP32)
                << "debug.assert conversion '%f' requires FP32 scalar, but got " << dtype.ToString();
        } else if (conversion == 'x') {
            PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsPrintfUnsignedIntegerType(dtype) || IsPrintfIndexType(dtype))
                << "debug.assert conversion '%" << conversion << "' requires unsigned integer or index scalar, but got "
                << dtype.ToString();
        } else if (conversion == 'u') {
            PRO_IR_CHECK(ExternalError::INVALID_TYPE,
                         IsPrintfUnsignedIntegerType(dtype) || IsPrintfBoolType(dtype) || IsPrintfIndexType(dtype))
                << "debug.assert conversion '%" << conversion
                << "' requires unsigned integer, bool, or index scalar, but got " << dtype.ToString();
        } else {
            PRO_IR_CHECK(ExternalError::INVALID_TYPE,
                         IsPrintfSignedIntegerType(dtype) || IsPrintfBoolType(dtype) || IsPrintfIndexType(dtype))
                << "debug.assert conversion '%" << conversion
                << "' requires signed integer, bool, or index scalar, but got " << dtype.ToString();
        }
    }

    return GetUnknownType();
}

} // namespace

REGISTER_OP("debug.dump_tensor")
    .set_op_category("DebugOp")
    .set_description("Print a full tensor or a tensor window for debugging. Window dumps support dynamic "
                     "offsets/shapes and explicit Tensor strides.")
    .add_argument("tensor", "Input tensor (TensorType)")
    .add_argument("offsets", "Optional offsets per dimension (MakeTuple of integer scalars)")
    .add_argument("shapes", "Optional shape per dimension (MakeTuple of integer scalars)")
    .set_attr<bool>("show_location")
    .set_attr<std::string>("dump_flag")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDebugDumpType<TensorType>(args, "debug.dump_tensor", "TensorType");
    });

REGISTER_OP("debug.dump_tile")
    .set_op_category("DebugOp")
    .set_description("Print a tile or tile window for debugging. Full dumps support tiles with dynamic valid-shape; "
                     "window dumps support dynamic offsets/shapes. Tile windows are currently 2D-only.")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("offsets", "Optional offsets per dimension (MakeTuple of integer scalars)")
    .add_argument("shapes", "Optional shape per dimension (MakeTuple of integer scalars)")
    .set_attr<bool>("show_location")
    .set_attr<std::string>("dump_flag")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDebugDumpType<TileType>(args, "debug.dump_tile", "TileType");
    });

REGISTER_OP("debug.printf")
    .set_op_category("DebugOp")
    .set_description("Print scalar values using a compile-time format string")
    .add_argument("scalars", "Scalar arguments consumed by format conversions (variadic)")
    .set_attr<std::string>("format")
    .set_attr<bool>("show_location")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDebugPrintfType(args, kwargs);
    });

REGISTER_OP("debug.assert")
    .set_op_category("DebugOp")
    .set_description("Print an assertion failure message and abort execution when condition is false")
    .add_argument("condition", "Scalar boolean condition")
    .add_argument("scalars", "Optional scalar arguments consumed by assertion format conversions (variadic)")
    .set_attr<std::string>("condition_text")
    .set_attr<std::string>("format")
    .set_attr<bool>("show_location")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDebugAssertType(args, kwargs);
    });

REGISTER_OP("debug.trap")
    .set_op_category("DebugOp")
    .set_description("Abort execution by inserting a trap")
    .no_argument()
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        (void)args;
        (void)kwargs;
        return GetUnknownType();
    });

} // namespace ir
} // namespace pypto
