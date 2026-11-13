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
 * @file memory.cpp
 * \brief Memory block operations (get_block_idx, load, store)
 *
 * This file implements memory operations for block-level programming.
 * These operations handle data movement between tensors and unified buffers (tiles).
 */

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "core/error.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/op/op_common.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr.h"
#include "ir/span.h"
#include "ir/type.h"
#include "ir/transforms/sanitizer_layout.h"
#include "ir/type_inference.h"
#include "pypto_pro/error.h"

namespace pypto {
namespace ir {
using npu::tile_fwk::ExternalError;

namespace {

DataType GetValResultDtype(const DataType& elementDtype, bool preserveDtype)
{
    if (preserveDtype || !elementDtype.IsInt()) {
        return elementDtype;
    }
    return elementDtype == DataType::UINT64 ? DataType::UINT64 : DataType::INT64;
}

} // namespace

TypePtr DeduceBlockGetBlockIdxType([[maybe_unused]] const std::vector<ExprPtr>& args,
                                   [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs,
                                   const std::string& opName)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0)
        << "The operator " << opName << " requires no arguments, but got " << args.size();

    return std::make_shared<ScalarType>(DataType::INT64);
}

TypePtr DeduceBlockCreateMutexTileType([[maybe_unused]] const std::vector<ExprPtr>& args,
                                       [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs,
                                       const std::string& opName)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 3)
        << "The operator " << opName << " requires shape, valid_shape and mutex_ids";
    auto mutexIds = As<MakeTuple>(args[2]);
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, mutexIds != nullptr)
        << "make_mutex_tile mutex_ids must be a fixed-size tuple";

    // Extract dtype attribute
    DataType dtype = GetOpKwarg<DataType>(kwargs, "dtype");

    // First argument must be MakeTuple with static ConstInt elements
    auto shapeTuple = As<MakeTuple>(args[0]);
    PRO_IR_CHECK(ExternalError::DYNAMIC_SHAPE_COMPUTE_UNSUPPORTED, shapeTuple)
        << "The operator " << opName
        << " requires first argument to be a MakeTuple expression with static shape values, but got "
        << args[0]->TypeName();
    // Conv data layout (Layout ordinal) for a multi-dim pto::ConvTile; -1 = not a conv tile.
    int conv_layout = GetOpKwarg<int>(kwargs, "conv_layout", -1);
    if (conv_layout < 0) {
        // Every non-conv tile is rank-2.
        PRO_IR_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, shapeTuple->elements_.size() == 0x2)
            << "TileType only supports rank-2 shape; "
            << "for a 1-D tensor with shape [N], use TileType shape [N, 1] or [1, N], "
            << "and keep the same convention for all tile shapes of that tensor";
    }

    // Validate all elements are ConstInt (static compile-time constants)
    std::vector<ExprPtr> tileShape;
    tileShape.reserve(shapeTuple->elements_.size());

    for (size_t i = 0; i < shapeTuple->elements_.size(); ++i) {
        auto constInt = As<ConstInt>(shapeTuple->elements_[i]);
        PRO_IR_CHECK(ExternalError::DYNAMIC_SHAPE_COMPUTE_UNSUPPORTED, constInt)
            << "The operator " << opName << " shape element " << i
            << " must be a compile-time constant (ConstInt), but got " << shapeTuple->elements_[i]->TypeName();
        PRO_IR_CHECK(ExternalError::INVALID_SHAPE, constInt->value_ > 0)
            << "The operator " << opName << " shape element " << i << " must be positive, got " << constInt->value_;
        tileShape.push_back(shapeTuple->elements_[i]);
    }

    TileView tileView;

    auto validShapeTuple = As<MakeTuple>(args[1]);
    if (validShapeTuple)
        tileView.validShape = validShapeTuple->elements_;

    HardwareInfo hwInfo;

    int blayout = GetOpKwarg<int>(kwargs, "blayout", -1);
    if (blayout >= 0) {
        hwInfo.blayout = static_cast<TileLayout>(blayout);
    }

    int slayout = GetOpKwarg<int>(kwargs, "slayout", -1);
    if (slayout >= 0) {
        hwInfo.slayout = static_cast<TileLayout>(slayout);
    }

    int fractal = GetOpKwarg<int>(kwargs, "fractal", -1);
    if (fractal >= 0) {
        hwInfo.fractal = static_cast<uint64_t>(fractal);
    }

    int pad = GetOpKwarg<int>(kwargs, "pad", -1);
    if (pad >= 0) {
        hwInfo.pad = static_cast<TilePad>(pad);
    }

    int compact = GetOpKwarg<int>(kwargs, "compact", -1);
    if (compact >= 0) {
        hwInfo.compact = static_cast<CompactMode>(compact);
    }

    if (conv_layout >= 0) {
        hwInfo.convLayout = conv_layout;
    }
    // If explicit memref kwargs are provided (addr + size + id), attach a MemRef to the TileType.
    // This allows the PTO codegen to emit pto.alloc_tile with base_addr directly from the IR,
    // without requiring the init_memref pass.
    MemorySpace targetMemory = GetOpKwarg<MemorySpace>(kwargs, "target_memory",
                                                       std::optional<MemorySpace>(MemorySpace::Vec));

    bool hasMemref = false;
    for (const auto& kwarg : kwargs) {
        if (kwarg.first == "memref_id") {
            hasMemref = true;
            break;
        }
    }
    if (hasMemref) {
        int64_t addrVal = GetOpKwarg<int>(kwargs, "memref_addr");
        int64_t sizeVal = GetOpKwarg<int>(kwargs, "memref_size");
        uint64_t idVal = static_cast<uint64_t>(GetOpKwarg<int>(kwargs, "memref_id"));
        auto addrExpr = std::make_shared<ConstInt>(addrVal, DataType::INT64, Span::Unknown());
        MemRefPtr memref = std::make_shared<MemRef>(targetMemory, addrExpr, static_cast<uint64_t>(sizeVal), idVal);
        return std::make_shared<MutexTileType>(
            TileType(tileShape, dtype, std::optional<MemRefPtr>(memref), tileView, hwInfo),
            static_cast<uint32_t>(mutexIds->elements_.size()));
    }

    return std::make_shared<MutexTileType>(TileType(tileShape, dtype, std::nullopt, tileView, hwInfo),
                                           static_cast<uint32_t>(mutexIds->elements_.size()));
}

TypePtr DeduceGetValType([[maybe_unused]] const std::vector<ExprPtr>& args,
                         [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x2)
        << "getval requires exactly 2 arguments, but got " << args.size();

    auto firstType = args[0]->GetType();
    ValidateScalarAccessIndex(args[0], args[1], "getval");

    // SIMT element reads preserve their storage dtype; SIMD scalar reads retain integer widening.
    const bool preserveDtype = GetOpKwarg<bool>(kwargs, "preserve_dtype", false);
    if (auto tileType = As<TileType>(firstType)) {
        return std::make_shared<ScalarType>(GetValResultDtype(tileType->dtype_, preserveDtype));
    }
    auto tensorType = As<TensorType>(firstType);
    return std::make_shared<ScalarType>(GetValResultDtype(tensorType->dtype_, preserveDtype));
}

TypePtr DeduceTileValidShapeType(const std::vector<ExprPtr>& args,
                                 const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 1 && As<TileType>(args[0]->GetType()))
        << "block.tile_valid_shape requires one Tile argument";
    int axis = GetOpKwarg<int>(kwargs, "axis", 0);
    PRO_IR_CHECK(ExternalError::INVALID_SHAPE, axis >= 0 && axis <= 1)
        << "block.tile_valid_shape axis must be in [0, 1], got axis=" << axis;
    return std::make_shared<ScalarType>(DataType::INT64);
}

TypePtr DeduceTileAddrType(const std::vector<ExprPtr>& args,
                           [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 1)
        << "block.tile_addr requires exactly one argument (tile), but got " << args.size();
    auto tileType = As<TileType>(args[0]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, tileType)
        << "block.tile_addr requires a TileType argument, but got " << args[0]->GetType()->TypeName();
    return std::make_shared<ScalarType>(DataType::INT64);
}

TypePtr DeduceSetValType([[maybe_unused]] const std::vector<ExprPtr>& args,
                         [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x3)
        << "setval requires exactly 3 arguments, but got " << args.size();

    ValidateScalarAccessIndex(args[0], args[1], "setval");
    auto valueType = As<ScalarType>(args[2]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, valueType)
        << "setval requires value to be ScalarType, but got " << args[2]->GetType()->TypeName();

    return GetNoneType();
}

REGISTER_OP("block.sanitizer_log_region")
    .set_op_category("BlockOp")
    .set_description("Select the current core's region in a sanitizer log buffer")
    .add_argument("log", "Byte pointer to the whole log buffer")
    .add_argument("capacity", "Per-region payload capacity in uint32 words")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) -> TypePtr {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 2)
            << "block.sanitizer_log_region requires (log, capacity)";
        auto ptr = As<PtrType>(args[0]->GetType());
        PRO_IR_CHECK(ExternalError::INVALID_TYPE, ptr && ptr->dtype_ == DataType(DataType::INT8))
            << "block.sanitizer_log_region requires an int8 log pointer";
        auto capacity = As<ScalarType>(args[1]->GetType());
        PRO_IR_CHECK(ExternalError::INVALID_TYPE, capacity && capacity->dtype_.IsInt())
            << "block.sanitizer_log_region capacity must be an integer scalar";
        return std::make_shared<PtrType>(ptr->dtype_);
    });

// The variadic record payload is checked against the shared wire contract.
TypePtr DeduceSanitizerLogType(const std::vector<ExprPtr>& args,
                               [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() >= 4)
        << "block.sanitizer_log requires (record_head, fields..., log_ptr, capacity, line)";
    auto head = As<ConstInt>(args[0]);
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT,
                 head && head->value_ >= 0 && (static_cast<uint64_t>(head->value_) >> 16U) == kSanitizerRecordMagic)
        << "block.sanitizer_log requires a constant record head with sanitizer magic";
    const auto* layout = FindSanitizerRecordLayout(static_cast<uint32_t>(head->value_) & 0xffffU);
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, layout && args.size() == layout->wire_count + 3U)
        << "block.sanitizer_log record fields do not match the wire layout";
    const size_t log_index = args.size() - 3;
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<PtrType>(args[log_index]->GetType()))
        << "block.sanitizer_log log buffer must be a pointer";
    for (size_t i = 1; i < args.size(); ++i) {
        if (i == log_index) {
            continue;
        }
        auto scalar = As<ScalarType>(args[i]->GetType());
        PRO_IR_CHECK(ExternalError::INVALID_TYPE, scalar && scalar->dtype_.IsInt())
            << "block.sanitizer_log fields, capacity and line must be integer scalars";
    }
    return GetNoneType();
}

REGISTER_OP("block.sanitizer_log")
    .set_op_category("BlockOp")
    .set_description("Append a sanitizer record to the selected log region")
    .add_argument("record", "Record head, variadic integer fields, log pointer, capacity and source line")
    .f_deduce_type(DeduceSanitizerLogType);

// ============================================================================
// Registration Function for Block Memory Operations
// ============================================================================

REGISTER_OP("get_block_idx")
    .set_op_category("LanguageOp")
    .set_description("Get the current block index")
    .no_argument()
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockGetBlockIdxType(args, kwargs, "get_block_idx");
    });

REGISTER_OP("get_block_num")
    .set_op_category("LanguageOp")
    .set_description("Get the current block number")
    .no_argument()
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockGetBlockIdxType(args, kwargs, "get_block_num");
    });

REGISTER_OP("get_subblock_idx")
    .set_op_category("LanguageOp")
    .set_description("Get the current subblock index")
    .no_argument()
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockGetBlockIdxType(args, kwargs, "get_subblock_idx");
    });

REGISTER_OP("get_subblock_num")
    .set_op_category("LanguageOp")
    .set_description("Get the subblock count per AI Core (task ration)")
    .no_argument()
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockGetBlockIdxType(args, kwargs, "get_subblock_num");
    });

// Pointer to the i-th tensor inside a runtime-variadic (DYNAMIC) input descriptor. Returns a
// PtrType; build a typed view with pl.make_tensor(ptr, shape, stride, dtype=...).
TypePtr DeduceBlockDynamicInputPtrType(const std::vector<ExprPtr>& args, const std::string& opName)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 2)
        << "The operator " << opName << " requires 2 arguments (x, i), but got " << args.size();
    // Whether i is an integer index is settled by the frontend; reading x's dtype only
    // needs it to be a pointer.
    auto descType = As<PtrType>(args[0] ? args[0]->GetType() : nullptr);
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, descType != nullptr)
        << "The operator " << opName << " requires a PtrType as its first argument, got "
        << (args[0] ? args[0]->GetType()->TypeName() : std::string("null"));
    return std::make_shared<PtrType>(descType->dtype_);
}

REGISTER_OP("dynamic_input_ptr")
    .set_op_category("LanguageOp")
    .set_description("Pointer to the i-th tensor in a runtime-variadic (DYNAMIC) input descriptor")
    .add_argument("x", "Descriptor pointer of the dynamic input (PtrType)")
    .add_argument("i", "Tensor index within the descriptor")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockDynamicInputPtrType(args, "dynamic_input_ptr");
    });

// Shape entry of the i-th tensor inside a runtime-variadic (DYNAMIC) input descriptor
// (dim != 0 form). Returns an INT64 scalar (the length of that axis).
TypePtr DeduceBlockDynamicInputDimType(const std::vector<ExprPtr>& args, const std::string& opName)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 3)
        << "The operator " << opName << " requires 3 arguments (x, i, axis), but got " << args.size();
    return std::make_shared<ScalarType>(DataType::INT64);
}

REGISTER_OP("dynamic_input_dim")
    .set_op_category("LanguageOp")
    .set_description("Shape(axis) of the i-th tensor in a runtime-variadic (DYNAMIC) input descriptor")
    .add_argument("x", "Descriptor pointer of the dynamic input (PtrType)")
    .add_argument("i", "Tensor index within the descriptor")
    .add_argument("axis", "Shape axis to read")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockDynamicInputDimType(args, "dynamic_input_dim");
    });

REGISTER_OP("get_spr")
    .set_op_category("LanguageOp")
    .set_description("Read special purpose register value (get_ar instruction). "
                     "Currently only AR register is supported.")
    .no_argument()
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0) << "get_spr requires no arguments";
        return std::make_shared<ScalarType>(DataType::INT64);
    });

REGISTER_OP("get_cycle")
    .set_op_category("LanguageOp")
    .set_description("Read the device system cycle counter (get_sys_cnt instruction). "
                     "Use differences between two reads to measure elapsed cycles; "
                     "counter frequency differs by chip.")
    .no_argument()
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0) << "get_cycle requires no arguments";
        return std::make_shared<ScalarType>(DataType::INT64);
    });

REGISTER_OP("set_saturation_flag")
    .set_op_category("LanguageOp")
    .set_description("Set saturation flag in CTRL special purpose register.")
    .no_argument()
    .set_attr<int>("mode")
    .set_attr<bool>("enable")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return GetUnknownType();
    });

REGISTER_OP("get_saturation_flag")
    .set_op_category("LanguageOp")
    .set_description("Read saturation flag from CTRL special purpose register.")
    .no_argument()
    .set_attr<int>("mode")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return std::make_shared<ScalarType>(DataType::BOOL);
    });

REGISTER_OP("set_ctrl_spr")
    .set_op_category("LanguageOp")
    .set_description("Set a bit range in the CTRL special purpose register.")
    .add_argument("start_bit", "Start bit index (0-63)")
    .add_argument("end_bit", "End bit index (0-63)")
    .add_argument("value", "Value to write into the bit range")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 3)
            << "set_ctrl_spr requires 3 arguments (start_bit, end_bit, value), but got " << args.size();
        return GetUnknownType();
    });

REGISTER_OP("get_ctrl_spr")
    .set_op_category("LanguageOp")
    .set_description("Read a bit range from the CTRL special purpose register.")
    .add_argument("start_bit", "Start bit index (0-63)")
    .add_argument("end_bit", "End bit index (0-63)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 2)
            << "get_ctrl_spr requires 2 arguments (start_bit, end_bit), but got " << args.size();
        return std::make_shared<ScalarType>(DataType::INT64);
    });

REGISTER_OP("reset_ctrl_spr")
    .set_op_category("LanguageOp")
    .set_description("Reset a bit range in the CTRL register to default values.")
    .add_argument("start_bit", "Start bit index (0-63)")
    .add_argument("end_bit", "End bit index (0-63)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 2)
            << "reset_ctrl_spr requires 2 arguments (start_bit, end_bit), but got " << args.size();
        return GetUnknownType();
    });

REGISTER_OP("block.make_mutex_tile")
    .set_op_category("BlockOp")
    .set_description("Create a tile with a fixed-size mutex ID binding")
    .add_argument("shape", "Shape dimensions (TupleType of ScalarType(INT64))")
    .add_argument("valid_shape", "Valid shape dimensions (optional, TupleType)")
    .add_argument("mutex_ids", "Fixed-size tuple of bound mutex IDs")
    .set_attr<DataType>("dtype")
    .set_attr<MemorySpace>("target_memory")
    .set_attr<int>("memref_addr")
    .set_attr<int>("memref_size")
    .set_attr<int>("memref_id")
    .set_attr<int>("blayout")
    .set_attr<int>("slayout")
    .set_attr<int>("fractal")
    .set_attr<int>("pad")
    .set_attr<int>("compact")
    .set_attr<int>("conv_layout")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBlockCreateMutexTileType(args, kwargs, "block.make_mutex_tile");
    });

REGISTER_OP("block.getval")
    .set_op_category("BlockOp")
    .set_description("Read a scalar value from a tile or tensor at offset")
    .add_argument("container", "Input tile (TileType) or tensor (TensorType)")
    .add_argument("offset", "Integer element offset or tuple of integer coordinates")
    .set_attr<bool>("preserve_dtype")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceGetValType(args, kwargs);
    });

REGISTER_OP("block.tile_valid_shape")
    .set_op_category("BlockOp")
    .set_description("Read one runtime valid-shape dimension of a Tile")
    .add_argument("tile", "Input tile")
    .set_attr<int>("axis")
    .f_deduce_type(DeduceTileValidShapeType);

REGISTER_OP("block.tile_addr")
    .set_op_category("BlockOp")
    .set_description("Return the runtime base address of a Tile")
    .add_argument("tile", "Input tile (TileType or MutexTileType)")
    .f_deduce_type(DeduceTileAddrType);

REGISTER_OP("block.setval")
    .set_op_category("BlockOp")
    .set_description("Write a scalar value to a tile or tensor at offset")
    .add_argument("container", "Input tile (TileType) or tensor (TensorType)")
    .add_argument("offset", "Integer element offset or tuple of integer coordinates")
    .add_argument("value", "Scalar value to write (ScalarType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSetValType(args, kwargs);
    });

// ============================================================================
// block.subview — Tile/Tensor sub-view with offset and new shape
// ============================================================================

TypePtr DeduceSubViewType([[maybe_unused]] const std::vector<ExprPtr>& args,
                          [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    // args: [container, offset, valid_shape]
    // Type deduction uses the container's original shape (preserving row_stride).
    // The valid_shape argument is consumed by codegen to auto-emit SetValidShape.
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x3)
        << "block.subview requires exactly 3 arguments (container, offset, valid_shape), but got " << args.size();
    auto containerType = args[0]->GetType();
    if (auto tileType = As<TileType>(containerType)) {
        // A subview keeps the source Tile's physical storage description, but its
        // runtime SetValidShape must not be overridden by a source tile_view.
        auto result = std::make_shared<TileType>(tileType->shape_, tileType->dtype_, tileType->memref_, std::nullopt,
                                                 tileType->hardwareInfo_);
        if (auto mutexType = As<MutexTileType>(containerType))
            return std::make_shared<MutexTileType>(*result, mutexType->mutexCount_);
        return result;
    }

    PRO_IR_CHECK(ExternalError::INVALID_TYPE, false)
        << "block.subview requires first argument to be TileType, but got " << containerType->TypeName();
    return nullptr;
}

REGISTER_OP("block.subview")
    .set_op_category("BlockOp")
    .set_description("Create a sub-view of a tile with offset and valid_shape")
    .add_argument("container", "Input tile (TileType)")
    .add_argument("offset", "Linear element offset (ScalarType with integer dtype)")
    .add_argument("valid_shape", "Sub-window dimensions (MakeTuple)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceSubViewType(args, kwargs);
    });

REGISTER_OP("block.tile_mutex_id")
    .set_op_category("BlockOp")
    .set_description("Read one runtime mutex ID from a tile")
    .add_argument("tile", "Bound tile")
    .add_argument("index", "Compile-time mutex slot index")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>&) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 2) << "tile_mutex_id requires tile and index";
        auto index = As<ConstInt>(args[1]);
        PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<TileType>(args[0]->GetType()) && index && index->value_ >= 0)
            << "tile_mutex_id requires a tile and a non-negative constant index";
        auto type = As<MutexTileType>(args[0]->GetType());
        PRO_IR_IRCHECK(ExternalError::INVALID_ARGUMENT, type && index->value_ < type->mutexCount_)
            << "tile_mutex_id index is outside the tile's mutex binding";
        return std::make_shared<ScalarType>(DataType::INT64);
    });

REGISTER_OP("block.reinterpret")
    .set_op_category("BlockOp")
    .set_description("Create an independent descriptor of a source tile's runtime storage")
    .add_argument("source", "Source tile, including its mutex binding")
    .add_argument("shape", "Target static shape")
    .set_attr<DataType>("dtype")
    .set_attr<int>("blayout")
    .set_attr<int>("slayout")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) -> TypePtr {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 2) << "reinterpret requires source and shape";
        auto source = As<TileType>(args[0]->GetType());
        auto shape = As<MakeTuple>(args[1]);
        PRO_IR_CHECK(ExternalError::INVALID_TYPE, source && source->memref_ && shape)
            << "reinterpret requires a placed tile and static shape";
        PRO_IR_CHECK(ExternalError::INVALID_SHAPE, shape->elements_.size() == 2)
            << "reinterpret requires a rank-2 shape";
        for (const auto& dim : shape->elements_) {
            auto constant = As<ConstInt>(dim);
            PRO_IR_CHECK(ExternalError::INVALID_SHAPE, constant && constant->value_ > 0)
                << "reinterpret shape must contain positive compile-time integers";
        }
        auto hw = source->hardwareInfo_.value_or(HardwareInfo{});
        hw.blayout = static_cast<TileLayout>(GetOpKwarg<int>(kwargs, "blayout"));
        hw.slayout = static_cast<TileLayout>(GetOpKwarg<int>(kwargs, "slayout"));
        auto result = std::make_shared<TileType>(shape->elements_, GetOpKwarg<DataType>(kwargs, "dtype"),
                                                 source->memref_, std::nullopt, hw);
        if (auto mutexType = As<MutexTileType>(source))
            return std::make_shared<MutexTileType>(*result, mutexType->mutexCount_);
        return result;
    });

} // namespace ir
} // namespace pypto
