/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 * You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License in the root of the software repository for the full text of the License.
 */

/**
 * @file dist_ops.cpp
 * \brief Distributed-window operations
 */

#include <any>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "core/error.h"
#include "core/logging.h"
#include "interface/inner/dist_utils.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr.h"
#include "ir/span.h"
#include "ir/type.h"
#include "ir/type_inference.h"
#include "pypto_pro/error.h"

namespace pypto {
namespace ir {
using npu::tile_fwk::ExternalError;

namespace {
TypePtr DeduceWindowGetType(const std::vector<ExprPtr>& args,
                            [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x2)
        << "dist.window_get requires exactly 2 arguments, but got " << args.size();

    PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<WindowType>(args[0]->GetType()))
        << "dist.window_get requires window to be WindowType, but got " << args[0]->GetType()->TypeName();
    auto field = As<ConstInt>(args[1]);
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, field != nullptr)
        << "dist.window_get field index must be a compile-time constant";

    PRO_IR_CHECK(ExternalError::OUT_OF_RANGE, dist::WindowScalarFieldName(field->value_) != nullptr)
        << "dist.window_get field index " << field->value_
        << " is not a scalar field (remote_ptrs array reads are not supported yet)";
    return std::make_shared<ScalarType>(dist::WindowFieldIsInt32(field->value_) ? DataType::INT32 : DataType::INT64);
}

TypePtr DeduceBarrierType(const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x1)
        << "dist.barrier requires exactly 1 argument (the window), but got " << args.size();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<WindowType>(args[0]->GetType()))
        << "dist.barrier requires window to be WindowType, but got " << args[0]->GetType()->TypeName();

    int timeoutCycles = 0;
    for (const auto& [name, value] : kwargs) {
        if (name == "timeout_cycles") {
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, value.type() == typeid(int))
                << "dist.barrier 'timeout_cycles' must be a compile-time constant integer";
            timeoutCycles = std::any_cast<int>(value);
            continue;
        }
        if (name == "sync_local" || name == "flush_store") {
            // Static instruction attributes lowered to codegen template parameters.
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, value.type() == typeid(bool))
                << "dist.barrier '" << name << "' must be a compile-time constant bool";
            continue;
        }
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, false)
            << "dist.barrier does not accept keyword argument '" << name << "'";
    }
    PRO_IR_CHECK(ExternalError::INVALID_VAL, timeoutCycles >= 0)
        << "dist.barrier timeout_cycles must be >= 0 (0 = wait forever), but got " << timeoutCycles;
    return GetNoneType();
}
} // namespace

REGISTER_OP("dist.window_get")
    .set_op_category("DistOp")
    .set_description("Read one scalar field of the device pypto::dist::Window descriptor")
    .add_argument("window", "Window handle (WindowType)")
    .add_argument("field", "Field index (compile-time constant integer)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceWindowGetType(args, kwargs);
    });

REGISTER_OP("dist.barrier")
    .set_op_category("DistOp")
    .set_description("All-rank software barrier over the window's barrier region; "
                     "every AIV of every rank must execute it, timeout traps")
    .add_argument("window", "Window handle (WindowType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceBarrierType(args, kwargs);
    });

namespace {
const auto kIsIntScalar = [](const ExprPtr& expr) {
    auto scalar = As<ScalarType>(expr->GetType());
    return scalar != nullptr && (scalar->dtype_ == DataType::INT32 || scalar->dtype_ == DataType::INT64 ||
                                 scalar->dtype_ == DataType::UINT32 || scalar->dtype_ == DataType::UINT64);
};

TypePtr DeduceDistTransferType(const std::vector<ExprPtr>& args,
                               const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x7)
        << "requires 7 arguments [window, memory, base_bytes, offsets, shape, tile, peer], but got " << args.size();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<WindowType>(args[0]->GetType()))
        << "window must be WindowType, but got " << args[0]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<TensorType>(args[1]->GetType()))
        << "memory must be a tensor kernel parameter registered with Window.register_memory, but got "
        << args[1]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, kIsIntScalar(args[2]))
        << "base_bytes must be an integer scalar, but got " << args[2]->GetType()->TypeName();
    auto offsets = As<MakeTuple>(args[3]);
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, offsets != nullptr)
        << "offsets must be a tuple of element coordinates, but got " << args[3]->GetType()->TypeName();
    for (const auto& coord : offsets->elements_) {
        PRO_IR_CHECK(ExternalError::INVALID_TYPE, kIsIntScalar(coord)) << "offset coordinates must be integer scalars";
    }
    auto shape = As<MakeTuple>(args[4]);
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, shape != nullptr)
        << "shape must be a tuple of view dimensions, but got " << args[4]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_SHAPE, !shape->elements_.empty() && shape->elements_.size() <= 5)
        << "DTensor view shape must have 1-5 dimensions";
    for (const auto& dim : shape->elements_) {
        PRO_IR_CHECK(ExternalError::INVALID_TYPE, kIsIntScalar(dim))
            << "DTensor view shape dims must be integer scalars";
        if (const auto constant = As<ConstInt>(dim)) {
            PRO_IR_CHECK(ExternalError::INVALID_SHAPE, constant->value_ > 0)
                << "DTensor view shape dims must be positive";
        }
    }
    auto tileType = As<TileType>(args[5]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, tileType != nullptr)
        << "tile must be a TileType, but got " << args[5]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, kIsIntScalar(args[6]))
        << "peer must be an integer scalar, but got " << args[6]->GetType()->TypeName();

    bool hasDtype = false;
    for (const auto& [name, value] : kwargs) {
        if (name == "dtype") {
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, value.type() == typeid(DataType))
                << "'dtype' must be a DataType constant";
            PRO_IR_CHECK(ExternalError::INVALID_TYPE, std::any_cast<DataType>(value) == tileType->dtype_)
                << "tile dtype does not match the DTensor view dtype";
            hasDtype = true;
        } else if (name == "tile_dims" || name == "is_transpose") {
            // Resolved 'order' mapping — validated at parse time (block.load semantics).
        } else if (name == "atomic") {
            // Static instruction attribute: the put's MTE3 atomic mode (AtomicType).
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, value.type() == typeid(int))
                << "'atomic' must be a compile-time constant integer";
            PRO_IR_CHECK(ExternalError::INVALID_VAL,
                         std::any_cast<int>(value) == static_cast<int>(AtomicType::AtomicNone) ||
                             std::any_cast<int>(value) == static_cast<int>(AtomicType::AtomicAdd))
                << "'atomic' must be an AtomicType member (0 = none, 1 = add), but got " << std::any_cast<int>(value);
        } else {
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, false) << "does not accept keyword argument '" << name << "'";
        }
    }
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, hasDtype)
        << "requires the 'dtype' keyword (the DTensor view element type)";
    PRO_IR_CHECK(ExternalError::INVALID_SHAPE, offsets->elements_.size() == shape->elements_.size())
        << "offset coordinate count (" << offsets->elements_.size() << ") must equal the DTensor view dimension count ("
        << shape->elements_.size() << ")";
    return GetNoneType();
}
} // namespace

REGISTER_OP("dist.get")
    .set_op_category("DistOp")
    .set_description("One-sided tile load from a peer's window; element-coordinate addressed like block.load "
                     "")
    .add_argument("window", "Window handle (WindowType)")
    .add_argument("region_id", "Registered memory region tensor parameter (TensorType); selects the region on "
                               "device")
    .add_argument("base_bytes", "DTensor view start, scalar byte offset into the region (integer scalar)")
    .add_argument("offsets", "Per-axis element coordinates inside the DTensor view (tuple of integer scalars)")
    .add_argument("shape", "DTensor view dimensions, compile-time or runtime integer scalars (tuple)")
    .add_argument("tile", "Destination tile (TileType)")
    .add_argument("peer", "Source peer rank (integer scalar)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistTransferType(args, kwargs);
    });

namespace {
TypePtr DeduceNotifyType(const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x4)
        << "dist.notify requires 4 arguments [window, slot, val, peer], but got " << args.size();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<WindowType>(args[0]->GetType()))
        << "dist.notify requires window to be WindowType, but got " << args[0]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, kIsIntScalar(args[1]))
        << "dist.notify slot must be an integer scalar, but got " << args[1]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, kIsIntScalar(args[2]))
        << "dist.notify val must be an integer scalar, but got " << args[2]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, kIsIntScalar(args[3]))
        << "dist.notify peer must be an integer scalar, but got " << args[3]->GetType()->TypeName();

    int notifyOp = 0;
    for (const auto& [name, value] : kwargs) {
        if (name == "notify_op") {
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, value.type() == typeid(int))
                << "dist.notify 'notify_op' must be a compile-time constant integer";
            notifyOp = std::any_cast<int>(value);
            continue;
        }
        if (name == "flush_store") {
            // Static instruction attribute lowered to a codegen template parameter.
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, value.type() == typeid(bool))
                << "dist.notify 'flush_store' must be a compile-time constant bool";
            continue;
        }
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, false)
            << "dist.notify does not accept keyword argument '" << name << "'";
    }
    // 0 = NotifyOp.SET, 1 = NotifyOp.ATOMIC_ADD (python/pypto_pro/language/distribute.py).
    PRO_IR_CHECK(ExternalError::INVALID_VAL, notifyOp == 0 || notifyOp == 1)
        << "dist.notify notify_op must be 0 (SET) or 1 (ATOMIC_ADD), but got " << notifyOp;
    return GetNoneType();
}

TypePtr DeduceWaitType(const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x3)
        << "dist.wait requires 3 arguments [window, slot, val], but got " << args.size();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<WindowType>(args[0]->GetType()))
        << "dist.wait requires window to be WindowType, but got " << args[0]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, kIsIntScalar(args[1]))
        << "dist.wait slot must be an integer scalar, but got " << args[1]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, kIsIntScalar(args[2]))
        << "dist.wait val must be an integer scalar, but got " << args[2]->GetType()->TypeName();

    int cmp = 0;
    int timeoutCycles = 0;
    for (const auto& [name, value] : kwargs) {
        if (name == "cmp" || name == "timeout_cycles") {
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, value.type() == typeid(int))
                << "dist.wait '" << name << "' must be a compile-time constant integer";
            const int parsed = std::any_cast<int>(value);
            if (name == "cmp") {
                cmp = parsed;
            } else {
                timeoutCycles = parsed;
            }
            continue;
        }
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, false)
            << "dist.wait does not accept keyword argument '" << name << "'";
    }
    // CmpMode: EQ/NE/LT/LE/GT/GE = 0-5 (ir/op_attr_types.h); the device helper's case
    // order is locked to this enum.
    PRO_IR_CHECK(ExternalError::OUT_OF_RANGE, cmp >= 0 && cmp <= 5)
        << "dist.wait cmp must be a CmpMode member (0-5), but got " << cmp;
    PRO_IR_CHECK(ExternalError::INVALID_VAL, timeoutCycles >= 0)
        << "dist.wait timeout_cycles must be >= 0 (0 = wait forever), but got " << timeoutCycles;
    return GetNoneType();
}

TypePtr DeduceNotifyAllType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x3)
        << "dist.notify_all requires 3 arguments [window, slot, val], but got " << args.size();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<WindowType>(args[0]->GetType()))
        << "dist.notify_all requires window to be WindowType, but got " << args[0]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, kIsIntScalar(args[1]))
        << "dist.notify_all slot must be an integer scalar, but got " << args[1]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, kIsIntScalar(args[2]))
        << "dist.notify_all val must be an integer scalar, but got " << args[2]->GetType()->TypeName();

    int notifyOp = 0;
    for (const auto& [name, value] : kwargs) {
        if (name == "notify_op") {
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, value.type() == typeid(int))
                << "dist.notify_all 'notify_op' must be a compile-time constant integer";
            notifyOp = std::any_cast<int>(value);
            continue;
        }
        if (name == "flush_store") {
            // Static instruction attribute lowered to a codegen template parameter.
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, value.type() == typeid(bool))
                << "dist.notify_all 'flush_store' must be a compile-time constant bool";
            continue;
        }
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, false)
            << "dist.notify_all does not accept keyword argument '" << name << "'";
    }
    // 0 = NotifyOp.SET, 1 = NotifyOp.ATOMIC_ADD (python/pypto_pro/language/distribute.py).
    PRO_IR_CHECK(ExternalError::INVALID_VAL, notifyOp == 0 || notifyOp == 1)
        << "dist.notify_all notify_op must be 0 (SET) or 1 (ATOMIC_ADD), but got " << notifyOp;
    return GetNoneType();
}

} // namespace

REGISTER_OP("dist.notify")
    .set_op_category("DistOp")
    .set_description("Write a value into a signal slot of a peer's (or the local) window; flush_store= "
                     "optionally orders it after the issuing AIV's prior puts via an MTE3 flush")
    .add_argument("window", "Window handle (WindowType)")
    .add_argument("slot", "Signal slot index (integer scalar)")
    .add_argument("val", "Value to write (integer scalar)")
    .add_argument("peer", "Destination rank (integer scalar)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceNotifyType(args, kwargs);
    });

REGISTER_OP("dist.wait")
    .set_op_category("DistOp")
    .set_description("Spin the calling AIV until the local window's signal slot satisfies the comparison; "
                     "timeout traps")
    .add_argument("window", "Window handle (WindowType)")
    .add_argument("slot", "Signal slot index (integer scalar)")
    .add_argument("val", "Value to compare against (integer scalar)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceWaitType(args, kwargs);
    });

REGISTER_OP("dist.notify_all")
    .set_op_category("DistOp")
    .set_description("All-rank notify: every calling AIV signals every rank's window slot (local rank included) "
                     "with one SIMT fan-out; flush_store= optionally orders it after its own prior puts; async "
                     "like dist.notify")
    .add_argument("window", "Window handle (WindowType)")
    .add_argument("slot", "Signal slot index (integer scalar)")
    .add_argument("val", "Value to write (integer scalar)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceNotifyAllType(args, kwargs);
    });

REGISTER_OP("dist.put")
    .set_op_category("DistOp")
    .set_description("One-sided tile store into a peer's window; element-coordinate addressed like block.store "
                     "")
    .add_argument("window", "Window handle (WindowType)")
    .add_argument("region_id", "Registered memory region tensor parameter (TensorType); selects the region on "
                               "device")
    .add_argument("base_bytes", "DTensor view start, scalar byte offset into the region (integer scalar)")
    .add_argument("offsets", "Per-axis element coordinates inside the DTensor view (tuple of integer scalars)")
    .add_argument("shape", "DTensor view dimensions, compile-time or runtime integer scalars (tuple)")
    .add_argument("tile", "Source tile (TileType)")
    .add_argument("peer", "Destination peer rank (integer scalar)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistTransferType(args, kwargs);
    });

namespace {
bool IsSupportedScalarDType(const DataType& dtype)
{
    return dtype == DataType::INT32 || dtype == DataType::UINT32 || dtype == DataType::INT64 ||
           dtype == DataType::UINT64 || dtype == DataType::FP32;
}

bool IsSupportedIndexDType(const DataType& dtype)
{
    return dtype == DataType::INT32 || dtype == DataType::INT64 || dtype == DataType::UINT32 ||
           dtype == DataType::UINT64;
}

const auto kIsNumericScalar = [](const ExprPtr& expr) {
    auto scalar = As<ScalarType>(expr->GetType());
    return scalar != nullptr && (scalar->dtype_.IsInt() || scalar->dtype_.IsFloat());
};

TypePtr DeduceDistChannelType(const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x2)
        << "dist.channel requires 2 arguments [window, peer], but got " << args.size();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<WindowType>(args[0]->GetType()))
        << "dist.channel requires window to be WindowType, but got " << args[0]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, kIsIntScalar(args[1]))
        << "dist.channel peer must be an integer scalar (-1 selects the calling AIV's jetty), but got "
        << args[1]->GetType()->TypeName();

    bool hasUbAddr = false;
    int ubAddr = 0;
    for (const auto& [name, value] : kwargs) {
        if (name == "ub_addr") {
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, value.type() == typeid(int))
                << "dist.channel 'ub_addr' must be a compile-time constant integer";
            ubAddr = std::any_cast<int>(value);
            hasUbAddr = true;
            continue;
        }
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, false)
            << "dist.channel does not accept keyword argument '" << name << "'";
    }
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, hasUbAddr) << "dist.channel requires the 'ub_addr' keyword";
    PRO_IR_CHECK(ExternalError::INVALID_VAL, ubAddr >= 0 && ubAddr % 64 == 0)
        << "dist.channel ub_addr must be a non-negative 64-byte aligned address, but got " << ubAddr;
    // Opaque handle: an int64 the frontend guarantees is a channel token.
    return std::make_shared<ScalarType>(DataType::INT64);
}

void CheckChannelOperand(const std::vector<ExprPtr>& args, const char* opName)
{
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, kIsIntScalar(args[0]))
        << opName << " first argument must be a channel handle (int64 token), but got "
        << args[0]->GetType()->TypeName();
}

struct ChannelWriteKwargs {
    DataType dtype;
    bool submit = true;
    int timeoutCycles = 0;
};

ChannelWriteKwargs ParseChannelWriteKwargs(const std::vector<std::pair<std::string, std::any>>& kwargs,
                                           const ExprPtr& tensorArg, const char* opName, bool requireSubmit)
{
    ChannelWriteKwargs parsed;
    bool hasDtype = false;
    bool hasSubmit = false;
    for (const auto& [name, value] : kwargs) {
        if (name == "dtype") {
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, value.type() == typeid(DataType))
                << "'dtype' must be a DataType constant";
            parsed.dtype = std::any_cast<DataType>(value);
            if (tensorArg != nullptr) {
                auto tensorType = As<TensorType>(tensorArg->GetType());
                PRO_IR_CHECK(ExternalError::INVALID_TYPE, tensorType != nullptr)
                    << opName << " tensor must be a TensorType";
                PRO_IR_CHECK(ExternalError::INVALID_TYPE, parsed.dtype == tensorType->dtype_)
                    << opName << " dtype does not match the source tensor dtype";
            }
            hasDtype = true;
        } else if (name == "submit") {
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, value.type() == typeid(bool))
                << opName << " 'submit' must be a compile-time bool";
            parsed.submit = std::any_cast<bool>(value);
            hasSubmit = true;
        } else if (name == "timeout_cycles") {
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, value.type() == typeid(int))
                << opName << " 'timeout_cycles' must be a compile-time constant integer";
            parsed.timeoutCycles = std::any_cast<int>(value);
        } else {
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, false)
                << opName << " does not accept keyword argument '" << name << "'";
        }
    }
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, hasDtype) << opName << " requires the 'dtype' keyword";
    if (requireSubmit) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, hasSubmit) << opName << " requires the 'submit' keyword";
    }
    // 0 = wait forever at the op layer; the backend emits INT64_MAX for the engine template.
    PRO_IR_CHECK(ExternalError::INVALID_VAL, parsed.timeoutCycles >= 0)
        << opName << " timeout_cycles must be >= 0 (0 = wait forever), but got " << parsed.timeoutCycles;
    return parsed;
}

TypePtr DeduceDistChannelWriteType(const std::vector<ExprPtr>& args,
                                   const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x5)
        << "dist.channel_write requires 5 arguments [channel, memory, base_bytes, num_bytes, tensor], but got "
        << args.size();
    CheckChannelOperand(args, "dist.channel_write");
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<TensorType>(args[1]->GetType()))
        << "dist.channel_write memory must be a TensorType (registered region), but got "
        << args[1]->GetType()->TypeName();
    for (size_t i = 2; i <= 3; ++i) {
        PRO_IR_CHECK(ExternalError::INVALID_TYPE, kIsIntScalar(args[i]))
            << "dist.channel_write argument " << i << " must be an integer scalar, but got "
            << args[i]->GetType()->TypeName();
    }
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<TensorType>(args[4]->GetType()))
        << "dist.channel_write tensor must be a TensorType, but got " << args[4]->GetType()->TypeName();
    (void)ParseChannelWriteKwargs(kwargs, args[4], "dist.channel_write", true);
    return GetNoneType();
}

TypePtr DeduceDistChannelWriteIndexedType(const std::vector<ExprPtr>& args,
                                          const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x8)
        << "dist.channel_write_indexed requires 8 arguments [channel, region, base_bytes, tensor, count, "
           "row_bytes, src_indices, dst_indices], but got "
        << args.size();
    CheckChannelOperand(args, "dist.channel_write_indexed");
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<TensorType>(args[1]->GetType()))
        << "dist.channel_write_indexed region must be a TensorType (registered region), but got "
        << args[1]->GetType()->TypeName();
    for (size_t i : {size_t{2}, size_t{4}, size_t{5}}) {
        PRO_IR_CHECK(ExternalError::INVALID_TYPE, kIsIntScalar(args[i]))
            << "dist.channel_write_indexed argument " << i << " must be an integer scalar, but got "
            << args[i]->GetType()->TypeName();
    }
    auto tensorType = As<TensorType>(args[3]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, tensorType != nullptr)
        << "dist.channel_write_indexed tensor must be a TensorType, but got " << args[3]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_SHAPE, !tensorType->shape_.empty())
        << "dist.channel_write_indexed tensor must have a row axis";

    // An index side is a 1-D int32/int64 UB Tile, or the 0 sentinel when that side
    // uses the row number j directly. Presence is a compile-time property.
    bool hasSrcIndex = false;
    bool hasDstIndex = false;
    std::optional<DataType> srcIndexDType;
    std::optional<DataType> dstIndexDType;
    for (size_t i = 6; i <= 7; ++i) {
        const char* name = i == 6 ? "src_indices" : "dst_indices";
        auto tileType = As<TileType>(args[i]->GetType());
        if (tileType == nullptr) {
            const auto sentinel = As<ConstInt>(args[i]);
            PRO_IR_CHECK(ExternalError::INVALID_TYPE, sentinel != nullptr && sentinel->value_ == 0)
                << "dist.channel_write_indexed " << name << " must be a Tile or the compile-time 0 sentinel, but got "
                << args[i]->GetType()->TypeName();
            continue;
        }
        PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsSupportedIndexDType(tileType->dtype_))
            << "dist.channel_write_indexed " << name << " must be an INT32/INT64/UINT32/UINT64 Tile, but got "
            << tileType->dtype_.ToString();
        // A 1-D logical table is spelled [N] or, because tile allocation only takes
        // rank-2 shapes, [N, 1] / [1, N] with a static unit axis.
        bool isOneDimensional = tileType->shape_.size() == 1;
        if (tileType->shape_.size() == 2) {
            const auto first = As<ConstInt>(tileType->shape_[0]);
            const auto second = As<ConstInt>(tileType->shape_[1]);
            isOneDimensional = (first != nullptr && first->value_ == 1) || (second != nullptr && second->value_ == 1);
        }
        PRO_IR_CHECK(ExternalError::INVALID_SHAPE, isOneDimensional)
            << "dist.channel_write_indexed " << name
            << " must be a 1-D row-index Tile ([N] or the rank-2 [N, 1] / [1, N] spelling), but got rank "
            << tileType->shape_.size();
        PRO_IR_CHECK(ExternalError::INVALID_TYPE,
                     tileType->memref_.has_value() && (*tileType->memref_)->memorySpace_ == MemorySpace::Vec)
            << "dist.channel_write_indexed " << name << " must live in UB (MemorySpace::Vec)";
        if (i == 6) {
            hasSrcIndex = true;
            srcIndexDType = tileType->dtype_;
        } else {
            hasDstIndex = true;
            dstIndexDType = tileType->dtype_;
        }
    }
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, hasSrcIndex || hasDstIndex)
        << "dist.channel_write_indexed requires src_indices or dst_indices; use dist.channel_write for a "
           "contiguous copy";
    // The device helper has one IndexT, so both tables must be read at the same element size.
    PRO_IR_CHECK(ExternalError::INVALID_TYPE,
                 !srcIndexDType.has_value() || !dstIndexDType.has_value() || *srcIndexDType == *dstIndexDType)
        << "dist.channel_write_indexed src_indices and dst_indices must share one dtype, but got "
        << (srcIndexDType.has_value() ? srcIndexDType->ToString() : "none") << " and "
        << (dstIndexDType.has_value() ? dstIndexDType->ToString() : "none");

    if (const auto rowBytes = As<ConstInt>(args[5])) {
        PRO_IR_CHECK(ExternalError::INVALID_VAL, rowBytes->value_ > 0)
            << "dist.channel_write_indexed row_bytes must be positive, but got " << rowBytes->value_;
        PRO_IR_CHECK(ExternalError::INVALID_VAL, rowBytes->value_ % 32 == 0)
            << "dist.channel_write_indexed row_bytes must be a multiple of 32 bytes, but got " << rowBytes->value_;
        PRO_IR_CHECK(ExternalError::INVALID_VAL, rowBytes->value_ <= 256 * 1024 * 1024)
            << "dist.channel_write_indexed row_bytes must be <= 256 MiB, but got " << rowBytes->value_;
    }

    // Static capacity cross-check when both the count and the table length are known.
    auto indexCapacity = [](const ExprPtr& arg) -> std::optional<int64_t> {
        auto tileType = As<TileType>(arg->GetType());
        if (tileType == nullptr) {
            return std::nullopt;
        }
        const auto& shape = tileType->shape_;
        if (shape.size() == 1) {
            const auto length = As<ConstInt>(shape[0]);
            return length != nullptr ? std::optional<int64_t>(length->value_) : std::nullopt;
        }
        if (shape.size() == 2) {
            const auto first = As<ConstInt>(shape[0]);
            const auto second = As<ConstInt>(shape[1]);
            if (second != nullptr && second->value_ == 1 && first != nullptr) {
                return first->value_;
            }
            if (first != nullptr && first->value_ == 1 && second != nullptr) {
                return second->value_;
            }
        }
        return std::nullopt;
    };
    if (const auto count = As<ConstInt>(args[4])) {
        PRO_IR_CHECK(ExternalError::INVALID_VAL, count->value_ >= 0)
            << "dist.channel_write_indexed count must be >= 0, but got " << count->value_;
        auto checkCapacity = [&count](const std::optional<int64_t>& capacity, const char* subject) {
            PRO_IR_CHECK(ExternalError::INVALID_VAL, !capacity.has_value() || count->value_ <= *capacity)
                << "dist.channel_write_indexed count " << count->value_ << " exceeds the " << subject << " capacity "
                << (capacity.has_value() ? *capacity : 0);
        };
        if (hasSrcIndex) {
            checkCapacity(indexCapacity(args[6]), "src_indices");
        } else if (const auto rows = As<ConstInt>(tensorType->shape_[0])) {
            checkCapacity(rows->value_, "source row");
        }
        if (hasDstIndex) {
            checkCapacity(indexCapacity(args[7]), "dst_indices");
        }
    }
    (void)ParseChannelWriteKwargs(kwargs, args[3], "dist.channel_write_indexed", true);
    return GetNoneType();
}

TypePtr DeduceDistChannelSubmitType(const std::vector<ExprPtr>& args,
                                    const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x1)
        << "dist.channel_submit requires 1 argument (the channel), but got " << args.size();
    CheckChannelOperand(args, "dist.channel_submit");
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, kwargs.empty()) << "dist.channel_submit accepts no keyword arguments";
    return GetNoneType();
}

TypePtr DeduceDistChannelDrainType(const std::vector<ExprPtr>& args,
                                   const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x1)
        << "dist.channel_drain requires 1 argument (the channel), but got " << args.size();
    CheckChannelOperand(args, "dist.channel_drain");
    int timeoutCycles = 0;
    for (const auto& [name, value] : kwargs) {
        if (name == "timeout_cycles") {
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, value.type() == typeid(int))
                << "dist.channel_drain 'timeout_cycles' must be a compile-time constant integer";
            timeoutCycles = std::any_cast<int>(value);
            continue;
        }
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, false)
            << "dist.channel_drain does not accept keyword argument '" << name << "'";
    }
    PRO_IR_CHECK(ExternalError::INVALID_VAL, timeoutCycles >= 0)
        << "dist.channel_drain timeout_cycles must be >= 0 (0 = wait forever), but got " << timeoutCycles;
    return GetNoneType();
}

TypePtr DeduceDistChannelIsExclusiveType(const std::vector<ExprPtr>& args,
                                         const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x1)
        << "dist.channel_is_exclusive requires 1 argument (the channel), but got " << args.size();
    CheckChannelOperand(args, "dist.channel_is_exclusive");
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, kwargs.empty())
        << "dist.channel_is_exclusive accepts no keyword arguments";
    return std::make_shared<ScalarType>(DataType::BOOL);
}
} // namespace

REGISTER_OP("dist.channel")
    .set_op_category("DistOp")
    .set_description("Open the caller's write-engine channel (peer = -1: the calling AIV's jetty in "
                     "multi-issuer mode; a rank: that peer's jetty); body is a stack object, this op is "
                     "the opaque token that binds it")
    .add_argument("window", "Window handle (WindowType)")
    .add_argument("peer", "Destination peer rank, or -1 for the per-AIV jetty (integer scalar)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistChannelType(args, kwargs);
    });

REGISTER_OP("dist.channel_write")
    .set_op_category("DistOp")
    .set_description("Async GM tensor -> remote window write through the channel's URMA jetty "
                     "(destination bound at channel creation); submit=true publishes immediately, "
                     "submit=false stages the WQE")
    .add_argument("channel", "Channel handle (opaque integer token)")
    .add_argument("region_id", "Registered memory region tensor parameter (TensorType); selects the region on "
                               "device")
    .add_argument("base_bytes", "DTensor view start, scalar byte offset into the payload (integer scalar)")
    .add_argument("num_bytes", "Transfer length in bytes (integer scalar)")
    .add_argument("tensor", "Local GM source tensor (TensorType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistChannelWriteType(args, kwargs);
    });

REGISTER_OP("dist.channel_write_indexed")
    .set_op_category("DistOp")
    .set_description("Async indexed row writes through the channel's URMA jetty: row j copies "
                     "src[src_indices[j]] to dst[dst_indices[j]] (missing table -> j); index tables "
                     "are 1-D int32/int64/uint32/uint64 UB Tiles, count is a runtime scalar")
    .add_argument("channel", "Channel handle (opaque integer token)")
    .add_argument("base_bytes", "DTensor view start, scalar byte offset into the payload (integer scalar)")
    .add_argument("tensor", "Local GM source tensor (TensorType)")
    .add_argument("count", "Runtime number of valid rows (integer scalar)")
    .add_argument("row_bytes", "Bytes of one row payload, used as the row stride (integer scalar)")
    .add_argument("src_indices", "Row-index Tile (1-D int32/int64/uint32/uint64, UB) or the 0 sentinel")
    .add_argument("dst_indices", "Row-index Tile (1-D int32/int64/uint32/uint64, UB) or the 0 sentinel")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistChannelWriteIndexedType(args, kwargs);
    });

REGISTER_OP("dist.channel_submit")
    .set_op_category("DistOp")
    .set_description("Publish the channel's deferred write-engine WQEs (doorbell); ordering, not completion")
    .add_argument("channel", "Channel handle (opaque integer token)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistChannelSubmitType(args, kwargs);
    });

REGISTER_OP("dist.channel_drain")
    .set_op_category("DistOp")
    .set_description("Block until the channel's previously enqueued writes complete; timeout traps")
    .add_argument("channel", "Channel handle (opaque integer token)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistChannelDrainType(args, kwargs);
    });

REGISTER_OP("dist.channel_is_exclusive")
    .set_op_category("DistOp")
    .set_description("True when the channel rides the calling AIV's private queue (multi-issuer "
                     "topology); false for a per-peer jetty")
    .add_argument("channel", "Channel handle (opaque integer token)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistChannelIsExclusiveType(args, kwargs);
    });

namespace {
TypePtr DeduceDistChannelWriteValueType(const std::vector<ExprPtr>& args,
                                        const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x4)
        << "dist.channel_write_value requires 4 arguments [channel, memory, base_bytes, value], but got "
        << args.size();
    CheckChannelOperand(args, "dist.channel_write_value");
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<TensorType>(args[1]->GetType()))
        << "dist.channel_write_value memory must be a TensorType (registered region), but got "
        << args[1]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, kIsIntScalar(args[2]))
        << "dist.channel_write_value base_bytes must be an integer scalar, but got " << args[2]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, kIsNumericScalar(args[3]))
        << "dist.channel_write_value value must be a numeric scalar, but got " << args[3]->GetType()->TypeName();

    const auto parsed = ParseChannelWriteKwargs(kwargs, nullptr, "dist.channel_write_value", true);
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsSupportedScalarDType(parsed.dtype))
        << "dist.channel_write_value does not support dtype " << parsed.dtype.ToString()
        << "; supported: INT32/UINT32/INT64/UINT64/FP32";
    const auto valueType = As<ScalarType>(args[3]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, !(parsed.dtype.IsInt() && valueType->dtype_.IsFloat()))
        << "dist.channel_write_value integer destination requires an integer value, but got "
        << valueType->dtype_.ToString();
    return GetNoneType();
}

TypePtr DeduceDistChannelWriteNotifyType(const std::vector<ExprPtr>& args,
                                         const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x7)
        << "dist.channel_write_notify requires 7 arguments [channel, memory, base_bytes, num_bytes, tensor, slot, "
           "val], but got "
        << args.size();
    CheckChannelOperand(args, "dist.channel_write_notify");
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<TensorType>(args[1]->GetType()))
        << "dist.channel_write_notify memory must be a TensorType (registered region), but got "
        << args[1]->GetType()->TypeName();
    for (size_t i = 2; i <= 3; ++i) {
        PRO_IR_CHECK(ExternalError::INVALID_TYPE, kIsIntScalar(args[i]))
            << "dist.channel_write_notify argument " << i << " must be an integer scalar, but got "
            << args[i]->GetType()->TypeName();
    }
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<TensorType>(args[4]->GetType()))
        << "dist.channel_write_notify tensor must be a TensorType, but got " << args[4]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, kIsIntScalar(args[5]))
        << "dist.channel_write_notify slot must be an integer scalar, but got " << args[5]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, kIsIntScalar(args[6]))
        << "dist.channel_write_notify val must be an integer scalar, but got " << args[6]->GetType()->TypeName();
    (void)ParseChannelWriteKwargs(kwargs, args[4], "dist.channel_write_notify", false);
    return GetNoneType();
}

TypePtr DeduceAtomicType(const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x7 || args.size() == 0x8)
        << "dist.atomic_* requires 7 or 8 arguments [window, memory, base_bytes, offsets, shape, peer, value(s)], "
        << "but got " << args.size();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<WindowType>(args[0]->GetType()))
        << "dist.atomic_* requires window to be WindowType, but got " << args[0]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<TensorType>(args[1]->GetType()))
        << "dist.atomic_* memory must be a TensorType (registered region), but got " << args[1]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, kIsIntScalar(args[2]))
        << "dist.atomic_* base_bytes must be an integer scalar, but got " << args[2]->GetType()->TypeName();

    auto shape = As<MakeTuple>(args[4]);
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, shape != nullptr)
        << "dist.atomic_* shape must be a tuple of view dimensions, but got " << args[4]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_SHAPE, !shape->elements_.empty() && shape->elements_.size() <= 5)
        << "DTensor view shape must have 1-5 dimensions";
    for (const auto& dim : shape->elements_) {
        PRO_IR_CHECK(ExternalError::INVALID_TYPE, kIsIntScalar(dim))
            << "DTensor view shape dims must be integer scalars";
        if (const auto constant = As<ConstInt>(dim)) {
            PRO_IR_CHECK(ExternalError::INVALID_SHAPE, constant->value_ > 0)
                << "DTensor view shape dims must be positive";
        }
    }

    auto offsets = As<MakeTuple>(args[3]);
    if (offsets != nullptr) {
        for (const auto& coord : offsets->elements_) {
            PRO_IR_CHECK(ExternalError::INVALID_TYPE, kIsIntScalar(coord))
                << "offset coordinates must be integer scalars";
        }
        PRO_IR_CHECK(ExternalError::INVALID_SHAPE, offsets->elements_.size() == shape->elements_.size())
            << "offset coordinate count (" << offsets->elements_.size()
            << ") must equal the DTensor view dimension count (" << shape->elements_.size() << ")";
    } else {
        PRO_IR_CHECK(ExternalError::INVALID_TYPE, kIsIntScalar(args[3]))
            << "offset must be a tuple of element coordinates or an integer linear offset, but got "
            << args[3]->GetType()->TypeName();
    }

    PRO_IR_CHECK(ExternalError::INVALID_TYPE, kIsIntScalar(args[5]))
        << "dist.atomic_* peer must be an integer scalar, but got " << args[5]->GetType()->TypeName();
    for (size_t i = 6; i < args.size(); ++i) {
        PRO_IR_CHECK(ExternalError::INVALID_TYPE, kIsNumericScalar(args[i]))
            << "dist.atomic_* value operand " << (i - 6) << " must be a numeric scalar, but got "
            << args[i]->GetType()->TypeName();
    }

    bool hasDtype = false;
    DataType dtype;
    for (const auto& [name, value] : kwargs) {
        if (name == "dtype") {
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, value.type() == typeid(DataType))
                << "dist.atomic_* 'dtype' must be a DataType constant";
            dtype = std::any_cast<DataType>(value);
            hasDtype = true;
            continue;
        }
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, false)
            << "dist.atomic_* does not accept keyword argument '" << name << "'";
    }
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, hasDtype) << "dist.atomic_* requires the 'dtype' keyword";
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsSupportedScalarDType(dtype))
        << "dist.atomic_* does not support dtype " << dtype.ToString() << "; supported: INT32/UINT32/INT64/UINT64/FP32";
    return std::make_shared<ScalarType>(dtype);
}
} // namespace

REGISTER_OP("dist.channel_write_value")
    .set_op_category("DistOp")
    .set_description("Async scalar-word write into the peer's window through the channel's jetty "
                     "(inline WQE; destination bound at channel creation); the value is cast to the "
                     "DTensor dtype")
    .add_argument("channel", "Channel handle (opaque integer token)")
    .add_argument("region_id", "Registered memory region tensor parameter (TensorType); selects the region on "
                               "device")
    .add_argument("base_bytes", "DTensor view start, scalar byte offset into the payload (integer scalar)")
    .add_argument("value", "Scalar word to write (numeric scalar)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistChannelWriteValueType(args, kwargs);
    });

REGISTER_OP("dist.channel_write_notify")
    .set_op_category("DistOp")
    .set_description("Async GM tensor -> remote window write fused with a hardware notify on payload "
                     "completion (one WriteWithNotify WQE, born published; destination bound at channel "
                     "creation)")
    .add_argument("channel", "Channel handle (opaque integer token)")
    .add_argument("region_id", "Registered memory region tensor parameter (TensorType); selects the region on "
                               "device")
    .add_argument("base_bytes", "DTensor view start, scalar byte offset into the payload (integer scalar)")
    .add_argument("num_bytes", "Transfer length in bytes (integer scalar)")
    .add_argument("tensor", "Local GM source tensor (TensorType)")
    .add_argument("slot", "Signal slot index in the destination rank's window (integer scalar)")
    .add_argument("val", "64-bit value written to the slot on completion (integer scalar)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistChannelWriteNotifyType(args, kwargs);
    });

REGISTER_OP("dist.atomic_add")
    .set_op_category("DistOp")
    .set_description("Remote atomic fetch-add on one window word; returns the pre-op value")
    .add_argument("window", "Window handle (WindowType)")
    .add_argument("region_id", "Registered memory region tensor parameter (TensorType); selects the region on "
                               "device")
    .add_argument("base_bytes", "DTensor view start, scalar byte offset into the region (integer scalar)")
    .add_argument("offsets", "Per-axis element coordinates (tuple) or a scalar linear element offset")
    .add_argument("shape", "DTensor view dimensions (tuple of integer scalars)")
    .add_argument("peer", "Target rank (integer scalar)")
    .add_argument("val", "Value to add (numeric scalar)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceAtomicType(args, kwargs);
    });

REGISTER_OP("dist.atomic_exchange")
    .set_op_category("DistOp")
    .set_description("Remote atomic exchange on one window word; returns the pre-op value")
    .add_argument("window", "Window handle (WindowType)")
    .add_argument("region_id", "Registered memory region tensor parameter (TensorType); selects the region on "
                               "device")
    .add_argument("base_bytes", "DTensor view start, scalar byte offset into the region (integer scalar)")
    .add_argument("offsets", "Per-axis element coordinates (tuple) or a scalar linear element offset")
    .add_argument("shape", "DTensor view dimensions (tuple of integer scalars)")
    .add_argument("peer", "Target rank (integer scalar)")
    .add_argument("val", "Replacement value (numeric scalar)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceAtomicType(args, kwargs);
    });

REGISTER_OP("dist.atomic_cas")
    .set_op_category("DistOp")
    .set_description("Remote atomic compare-and-swap on one window word; returns the pre-op value")
    .add_argument("window", "Window handle (WindowType)")
    .add_argument("region_id", "Registered memory region tensor parameter (TensorType); selects the region on "
                               "device")
    .add_argument("base_bytes", "DTensor view start, scalar byte offset into the region (integer scalar)")
    .add_argument("offsets", "Per-axis element coordinates (tuple) or a scalar linear element offset")
    .add_argument("shape", "DTensor view dimensions (tuple of integer scalars)")
    .add_argument("peer", "Target rank (integer scalar)")
    .add_argument("expect", "Value the word must hold for the swap (numeric scalar)")
    .add_argument("desired", "Value stored on success (numeric scalar)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceAtomicType(args, kwargs);
    });

} // namespace ir
} // namespace pypto
