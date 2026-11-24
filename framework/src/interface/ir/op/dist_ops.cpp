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

#include "core/common.h"
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
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, value.type() == typeid(int64_t))
                << "dist.barrier 'timeout_cycles' must be a compile-time constant integer";
            timeoutCycles = AnyCastFromIR<int>(value, name);
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
bool IsIntScalar(const ExprPtr& expr)
{
    auto scalar = As<ScalarType>(expr->GetType());
    return scalar != nullptr && IsAtomicDType(scalar->dtype_);
}

bool IsAtomicIntPtr(const ExprPtr& expr, bool allow_32 = true)
{
    auto ptr = As<PtrType>(expr->GetType());
    if (ptr == nullptr || !IsAtomicDType(ptr->dtype_)) {
        return false;
    }
    return allow_32 || ptr->dtype_ == DataType::INT64 || ptr->dtype_ == DataType::UINT64;
}

} // namespace

namespace {
bool IsSupportedScalarDType(const DataType& dtype)
{
    return dtype == DataType::INT32 || dtype == DataType::UINT32 || dtype == DataType::INT64 ||
           dtype == DataType::UINT64 || dtype == DataType::FP32;
}

bool IsSupportedReduceDType(const DataType& dtype)
{
    return dtype == DataType::INT8 || dtype == DataType::INT16 || dtype == DataType::INT32 ||
           dtype == DataType::UINT32 || dtype == DataType::FP16 || dtype == DataType::FP32 || dtype == DataType::BF16;
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
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsIntScalar(args[1]))
        << "dist.channel peer must be an integer scalar (-1 selects the calling AIV's jetty on a shared "
           "window), but got "
        << args[1]->GetType()->TypeName();

    bool hasUbAddr = false;
    int ubAddr = 0;
    for (const auto& [name, value] : kwargs) {
        if (name == "ub_addr") {
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, value.type() == typeid(int64_t))
                << "dist.channel 'ub_addr' must be a compile-time constant integer";
            ubAddr = AnyCastFromIR<int>(value, name);
            hasUbAddr = true;
            continue;
        }
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, false)
            << "dist.channel does not accept keyword argument '" << name << "'";
    }
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, hasUbAddr) << "dist.channel requires the 'ub_addr' keyword";
    PRO_IR_CHECK(ExternalError::INVALID_VAL, ubAddr >= 0 && ubAddr % NUM_VALUE_64 == 0)
        << "dist.channel ub_addr must be a non-negative 64-byte aligned address, but got " << ubAddr;
    // Opaque handle: an int64 the frontend guarantees is a channel token.
    return std::make_shared<ScalarType>(DataType::INT64);
}

void CheckChannelOperand(const std::vector<ExprPtr>& args, const char* opName)
{
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsIntScalar(args[0]))
        << opName << " first argument must be a channel handle (int64 token), but got "
        << args[0]->GetType()->TypeName();
}

struct ChannelWriteKwargs {
    bool submit = true;
    int timeoutCycles = 0;
    int option = 0x2D;
};

ChannelWriteKwargs ParseChannelWriteKwargs(const std::vector<std::pair<std::string, std::any>>& kwargs,
                                           const char* opName, bool requireSubmit, bool allowCustomOption)
{
    ChannelWriteKwargs parsed;
    bool hasSubmit = false;
    for (const auto& [name, value] : kwargs) {
        if (name == "submit") {
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, value.type() == typeid(bool))
                << opName << " 'submit' must be a compile-time bool";
            parsed.submit = std::any_cast<bool>(value);
            hasSubmit = true;
        } else if (name == "timeout_cycles") {
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, value.type() == typeid(int64_t))
                << opName << " 'timeout_cycles' must be a compile-time constant integer";
            parsed.timeoutCycles = AnyCastFromIR<int>(value, name);
        } else if (name == "option") {
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, value.type() == typeid(int64_t))
                << opName << " 'option' must be a compile-time constant integer";
            parsed.option = AnyCastFromIR<int>(value, name);
        } else {
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, false)
                << opName << " does not accept keyword argument '" << name << "'";
        }
    }
    if (requireSubmit) {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, hasSubmit) << opName << " requires the 'submit' keyword";
    }
    // 0 = wait forever at the op layer; the backend emits INT64_MAX for the engine template.
    PRO_IR_CHECK(ExternalError::INVALID_VAL, parsed.timeoutCycles >= 0)
        << opName << " timeout_cycles must be >= 0 (0 = wait forever), but got " << parsed.timeoutCycles;
    // WQE flag byte: odr[2:0], fence[3], cqe[5]; cqe is required for drain accounting.
    PRO_IR_CHECK(ExternalError::INVALID_VAL, parsed.option >= 0 && (parsed.option & ~0x2F) == 0)
        << opName << " option sets unsupported WQE flag bits, got " << parsed.option;
    PRO_IR_CHECK(ExternalError::INVALID_VAL, (parsed.option & 0x20) != 0)
        << opName << " option must include cqe (drain waits for one completion per WQE), got " << parsed.option;
    PRO_IR_CHECK(ExternalError::INVALID_VAL, (parsed.option & 0x3) != 0x3)
        << opName << " option cannot set the relaxed and strong order bits together, got " << parsed.option;
    if (!allowCustomOption) {
        PRO_IR_CHECK(ExternalError::INVALID_VAL, parsed.option == 0x2D)
            << opName << " rides hcomm's fixed WQE config, so 'option' must be 0x2D (UrmaOption.DEFAULT), got "
            << parsed.option;
    }
    return parsed;
}

DataType ChannelDstPtrDType(const std::vector<ExprPtr>& args, size_t index, const char* opName)
{
    auto ptrType = As<PtrType>(args[index]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, ptrType != nullptr)
        << opName << " destination must be a pointer (PtrType), but got " << args[index]->GetType()->TypeName();
    return ptrType->dtype_;
}

TypePtr DeduceDistChannelWriteType(const std::vector<ExprPtr>& args,
                                   const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x4)
        << "channel.write requires 4 arguments [channel, dst_ptr, num_bytes, tensor], but got " << args.size();
    CheckChannelOperand(args, "channel.write");
    (void)ChannelDstPtrDType(args, 1, "channel.write");
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsIntScalar(args[2]))
        << "channel.write num_bytes must be an integer scalar, but got " << args[2]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<TensorType>(args[3]->GetType()))
        << "channel.write tensor must be a TensorType, but got " << args[3]->GetType()->TypeName();
    (void)ParseChannelWriteKwargs(kwargs, "channel.write", true, true);
    return GetNoneType();
}

// BODY flag byte of a staged append: the caller's UrmaOption minus completion —
// submit adds fence+cqe to the last WQEBB, so staged WQEBBs never carry their
// own cqe (a caller-set bit is ignored). Default: UrmaOption.DEFAULT, which lands as
// the fence-only body 0x0D (cqe stripped).
int ParseAppendOption(const std::vector<std::pair<std::string, std::any>>& kwargs, const char* opName,
                      std::initializer_list<const char*> alreadyParsed = {})
{
    constexpr int kDefaultAppendOption = 0x2D;
    int option = kDefaultAppendOption;
    for (const auto& [name, value] : kwargs) {
        if (name == "option") {
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, value.type() == typeid(int64_t))
                << opName << " 'option' must be a compile-time constant integer";
            option = AnyCastFromIR<int>(value, name);
            continue;
        }
        // Ops with extra compile-time keywords validate those first and list them here.
        bool handled = false;
        for (const char* parsed : alreadyParsed) {
            handled = handled || name == parsed;
        }
        if (handled) {
            continue;
        }
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, false)
            << opName << " does not accept keyword argument '" << name << "'";
    }
    PRO_IR_CHECK(ExternalError::INVALID_VAL, option >= 0 && (option & ~0x2F) == 0)
        << opName << " option sets unsupported WQE flag bits, got " << option;
    PRO_IR_CHECK(ExternalError::INVALID_VAL, (option & 0x3) != 0x3)
        << opName << " option cannot set the relaxed and strong order bits together, got " << option;
    return option & ~0x20; // cqe is the publish's: stripped from the body byte
}

TypePtr DeduceDistBufferChannelReadType(const std::vector<ExprPtr>& args,
                                        const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x4)
        << "buffer_channel.read requires 4 arguments [channel, src_ptr, num_bytes, tensor], but got " << args.size();
    CheckChannelOperand(args, "buffer_channel.read");
    (void)ChannelDstPtrDType(args, 1, "buffer_channel.read");
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsIntScalar(args[2]))
        << "buffer_channel.read num_bytes must be an integer scalar, but got " << args[2]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<TensorType>(args[3]->GetType()))
        << "buffer_channel.read destination must be a TensorType, but got " << args[3]->GetType()->TypeName();
    (void)ParseAppendOption(kwargs, "buffer_channel.read");
    return GetNoneType();
}

TypePtr DeduceDistBufferChannelWriteReduceType(const std::vector<ExprPtr>& args,
                                               const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x4)
        << "buffer_channel.write_reduce requires 4 arguments [channel, dst_ptr, num_bytes, tensor], but got "
        << args.size();
    CheckChannelOperand(args, "buffer_channel.write_reduce");
    (void)ChannelDstPtrDType(args, 1, "buffer_channel.write_reduce");
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsIntScalar(args[2]))
        << "buffer_channel.write_reduce num_bytes must be an integer scalar, but got "
        << args[2]->GetType()->TypeName();
    const auto tensorType = As<TensorType>(args[3]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, tensorType)
        << "buffer_channel.write_reduce source must be a TensorType, but got " << args[3]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsSupportedReduceDType(tensorType->dtype_))
        << "buffer_channel.write_reduce does not support dtype " << tensorType->dtype_.ToString()
        << "; supported: INT8/INT16/INT32/UINT32/FP16/FP32/BF16";
    bool hasReduceOp = false;
    int reduceOp = 0;
    for (const auto& [name, value] : kwargs) {
        if (name == "reduce_op") {
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, value.type() == typeid(int64_t))
                << "buffer_channel.write_reduce 'reduce_op' must be a compile-time ReduceOp";
            reduceOp = AnyCastFromIR<int>(value, name);
            hasReduceOp = true;
            continue;
        }
        // option is validated by ParseAppendOption below; anything else is rejected.
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, name == "option")
            << "buffer_channel.write_reduce does not accept keyword argument '" << name << "'";
    }
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, hasReduceOp)
        << "buffer_channel.write_reduce requires the 'reduce_op' keyword";
    PRO_IR_CHECK(ExternalError::INVALID_VAL, reduceOp == 0x8 || reduceOp == 0x9 || reduceOp == 0xA)
        << "buffer_channel.write_reduce reduce_op must be ReduceOp.MAX/MIN/SUM, got " << reduceOp;
    (void)ParseAppendOption(kwargs, "buffer_channel.write_reduce", {"reduce_op"});
    return GetNoneType();
}

TypePtr DeduceDistBufferChannelWriteNotifyType(const std::vector<ExprPtr>& args,
                                               const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x6)
        << "buffer_channel.write_notify requires 6 arguments [channel, dst_ptr, num_bytes, tensor, signal_ptr, val], "
           "but got "
        << args.size();
    CheckChannelOperand(args, "buffer_channel.write_notify");
    (void)ChannelDstPtrDType(args, 1, "buffer_channel.write_notify");
    for (size_t i : {size_t{2}, size_t{5}}) {
        PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsIntScalar(args[i]))
            << "buffer_channel.write_notify argument " << i << " must be an integer scalar, but got "
            << args[i]->GetType()->TypeName();
    }
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsAtomicIntPtr(args[4], /*allow_32=*/false))
        << "buffer_channel.write_notify signal address must be a 64-bit integer pointer "
           "(INT64/UINT64; the hardware notify value is 64-bit), but got "
        << args[4]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<TensorType>(args[3]->GetType()))
        << "buffer_channel.write_notify tensor must be a TensorType, but got " << args[3]->GetType()->TypeName();
    (void)ParseAppendOption(kwargs, "buffer_channel.write_notify");
    return GetNoneType();
}

TypePtr DeduceDistBufferChannelAtomicFetchAddType(const std::vector<ExprPtr>& args,
                                                  const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x4)
        << "buffer_channel.atomic_fetch_add requires 4 arguments [channel, dst_ptr, fetch, val], but got "
        << args.size();
    CheckChannelOperand(args, "buffer_channel.atomic_fetch_add");
    const DataType dstDType = ChannelDstPtrDType(args, 1, "buffer_channel.atomic_fetch_add");
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsAtomicDType(dstDType))
        << "buffer_channel.atomic_fetch_add destination must point at a 32/64-bit integer word, but got "
        << dstDType.ToString();
    const auto fetchType = As<TensorType>(args[2]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, fetchType != nullptr)
        << "buffer_channel.atomic_fetch_add fetch must be a TensorType, but got " << args[2]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, fetchType->dtype_ == dstDType)
        << "buffer_channel.atomic_fetch_add fetch dtype must match the destination pointer (" << dstDType.ToString()
        << "), but got " << fetchType->dtype_.ToString();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsIntScalar(args[3]))
        << "buffer_channel.atomic_fetch_add val must be an integer scalar, but got " << args[3]->GetType()->TypeName();
    (void)ParseAppendOption(kwargs, "buffer_channel.atomic_fetch_add");
    return GetNoneType();
}

TypePtr DeduceDistBufferChannelAtomicCasType(const std::vector<ExprPtr>& args,
                                             const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x5)
        << "buffer_channel.atomic_cas requires 5 arguments [channel, dst_ptr, fetch, expected, desired], "
           "but got "
        << args.size();
    CheckChannelOperand(args, "buffer_channel.atomic_cas");
    const DataType dstDType = ChannelDstPtrDType(args, 1, "buffer_channel.atomic_cas");
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsAtomicDType(dstDType))
        << "buffer_channel.atomic_cas destination must point at a 32/64-bit integer word, but got "
        << dstDType.ToString();
    const auto fetchType = As<TensorType>(args[2]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, fetchType != nullptr)
        << "buffer_channel.atomic_cas fetch must be a TensorType, but got " << args[2]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, fetchType->dtype_ == dstDType)
        << "buffer_channel.atomic_cas fetch dtype must match the destination pointer (" << dstDType.ToString()
        << "), but got " << fetchType->dtype_.ToString();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsIntScalar(args[3]) && IsIntScalar(args[4]))
        << "buffer_channel.atomic_cas expected/desired must be integer scalars, but got "
        << args[3]->GetType()->TypeName() << " / " << args[4]->GetType()->TypeName();
    (void)ParseAppendOption(kwargs, "buffer_channel.atomic_cas");
    return GetNoneType();
}

TypePtr DeduceDistChannelSubmitType(const std::vector<ExprPtr>& args,
                                    const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x1)
        << "channel.submit requires 1 argument (the channel), but got " << args.size();
    CheckChannelOperand(args, "channel.submit");
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, kwargs.empty()) << "channel.submit accepts no keyword arguments";
    return GetNoneType();
}

TypePtr DeduceDistChannelDrainType(const std::vector<ExprPtr>& args,
                                   const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x1)
        << "channel.drain requires 1 argument (the channel), but got " << args.size();
    CheckChannelOperand(args, "channel.drain");
    int timeoutCycles = 0;
    for (const auto& [name, value] : kwargs) {
        if (name == "timeout_cycles") {
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, value.type() == typeid(int64_t))
                << "channel.drain 'timeout_cycles' must be a compile-time constant integer";
            timeoutCycles = AnyCastFromIR<int>(value, name);
            continue;
        }
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, false)
            << "channel.drain does not accept keyword argument '" << name << "'";
    }
    PRO_IR_CHECK(ExternalError::INVALID_VAL, timeoutCycles >= 0)
        << "channel.drain timeout_cycles must be >= 0 (0 = wait forever), but got " << timeoutCycles;
    return GetNoneType();
}

TypePtr DeduceDistBufferChannelType(const std::vector<ExprPtr>& args,
                                    const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x1)
        << "dist.buffer_channel requires 1 argument [channel], but got " << args.size();
    CheckChannelOperand(args, "dist.buffer_channel");
    bool hasAddr = false;
    bool hasBytes = false;
    int64_t bufAddr = 0;
    int64_t bufBytes = 0;
    for (const auto& [name, value] : kwargs) {
        if (name == "buf_addr") {
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, value.type() == typeid(int64_t))
                << "dist.buffer_channel 'buf_addr' must be a compile-time constant integer";
            bufAddr = AnyCastFromIR<int>(value, name);
            hasAddr = true;
            continue;
        }
        if (name == "buf_bytes") {
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, value.type() == typeid(int64_t))
                << "dist.buffer_channel 'buf_bytes' must be a compile-time constant integer";
            bufBytes = AnyCastFromIR<int>(value, name);
            hasBytes = true;
            continue;
        }
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, false)
            << "dist.buffer_channel does not accept keyword argument '" << name << "'";
    }
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, hasAddr && hasBytes)
        << "dist.buffer_channel requires the 'buf_addr' and 'buf_bytes' keywords";
    PRO_IR_CHECK(ExternalError::INVALID_VAL, bufAddr >= 0 && bufAddr % 64 == 0)
        << "dist.buffer_channel buf_addr must be a non-negative 64-byte aligned address, but got " << bufAddr;
    PRO_IR_CHECK(ExternalError::INVALID_VAL, bufBytes >= 128 && bufBytes % 64 == 0)
        << "dist.buffer_channel buf_bytes must be a 64-byte multiple of at least 128, but got " << bufBytes;
    // Opaque handle: an int64 the frontend guarantees is a buffer-channel token.
    return std::make_shared<ScalarType>(DataType::INT64);
}

TypePtr DeduceDistBufferChannelAppendWriteType(const std::vector<ExprPtr>& args,
                                               const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x4)
        << "buffer_channel.write requires 4 arguments [channel, dst_ptr, num_bytes, tensor], but got " << args.size();
    CheckChannelOperand(args, "buffer_channel.write");
    (void)ChannelDstPtrDType(args, 1, "buffer_channel.write");
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsIntScalar(args[2]))
        << "buffer_channel.write num_bytes must be an integer scalar, but got " << args[2]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<TensorType>(args[3]->GetType()))
        << "buffer_channel.write tensor must be a TensorType, but got " << args[3]->GetType()->TypeName();
    (void)ParseAppendOption(kwargs, "buffer_channel.write");
    return GetNoneType();
}

TypePtr DeduceDistBufferChannelAppendWriteValueType(const std::vector<ExprPtr>& args,
                                                    const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x3)
        << "buffer_channel.write_value requires 3 arguments [channel, dst_ptr, value], but got " << args.size();
    CheckChannelOperand(args, "buffer_channel.write_value");
    const DataType dstDType = ChannelDstPtrDType(args, 1, "buffer_channel.write_value");
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, kIsNumericScalar(args[2]))
        << "buffer_channel.write_value value must be a numeric scalar, but got " << args[2]->GetType()->TypeName();

    (void)ParseAppendOption(kwargs, "buffer_channel.write_value");
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsSupportedScalarDType(dstDType))
        << "buffer_channel.write_value does not support dtype " << dstDType.ToString()
        << "; supported: INT32/UINT32/INT64/UINT64/FP32";
    const auto valueType = As<ScalarType>(args[2]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, !(dstDType.IsInt() && valueType->dtype_.IsFloat()))
        << "buffer_channel.write_value integer destination requires an integer value, but got "
        << valueType->dtype_.ToString();
    return GetNoneType();
}

TypePtr DeduceDistBufferChannelSubmitType(const std::vector<ExprPtr>& args,
                                          const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x1)
        << "buffer_channel.submit requires 1 argument (the channel), but got " << args.size();
    CheckChannelOperand(args, "buffer_channel.submit");
    int timeoutCycles = 0;
    for (const auto& [name, value] : kwargs) {
        if (name == "timeout_cycles") {
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, value.type() == typeid(int64_t))
                << "buffer_channel.submit 'timeout_cycles' must be a compile-time constant integer";
            timeoutCycles = AnyCastFromIR<int>(value, name);
            continue;
        }
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, false)
            << "buffer_channel.submit does not accept keyword argument '" << name << "'";
    }
    PRO_IR_CHECK(ExternalError::INVALID_VAL, timeoutCycles >= 0)
        << "buffer_channel.submit timeout_cycles must be >= 0 (0 = wait forever), but got " << timeoutCycles;
    return GetNoneType();
}

TypePtr DeduceDistChannelLockType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x1)
        << "channel.lock requires 1 argument (the channel), but got " << args.size();
    CheckChannelOperand(args, "channel.lock");
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, kwargs.empty()) << "channel.lock accepts no keyword arguments";
    return GetNoneType();
}

TypePtr DeduceDistChannelUnlockType(const std::vector<ExprPtr>& args,
                                    const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x1)
        << "channel.unlock requires 1 argument (the channel), but got " << args.size();
    CheckChannelOperand(args, "channel.unlock");
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, kwargs.empty()) << "channel.unlock accepts no keyword arguments";
    return GetNoneType();
}

TypePtr DeduceDistChannelIsSharedType(const std::vector<ExprPtr>& args,
                                      const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x1)
        << "channel.is_shared requires 1 argument (the channel), but got " << args.size();
    CheckChannelOperand(args, "channel.is_shared");
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, kwargs.empty()) << "channel.is_shared accepts no keyword arguments";
    return std::make_shared<ScalarType>(DataType::BOOL);
}

TypePtr DeduceDistRemotePtrType(const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x3)
        << "dist.remote_ptr requires 3 arguments [window, ptr, peer], but got " << args.size();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<WindowType>(args[0]->GetType()))
        << "dist.remote_ptr requires window to be WindowType, but got " << args[0]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<PtrType>(args[1]->GetType()))
        << "dist.remote_ptr requires a pointer into a registered region, but got " << args[1]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsIntScalar(args[2]))
        << "dist.remote_ptr peer must be an integer scalar, but got " << args[2]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, kwargs.empty()) << "dist.remote_ptr accepts no keyword arguments";
    return std::make_shared<PtrType>(As<PtrType>(args[1]->GetType())->dtype_);
}

TypePtr DeduceDistChannelRemotePtrType(const std::vector<ExprPtr>& args,
                                       const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x2)
        << "channel.remote_ptr requires 2 arguments [channel, ptr], but got " << args.size();
    CheckChannelOperand(args, "channel.remote_ptr");
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, kwargs.empty()) << "channel.remote_ptr accepts no keyword arguments";
    return std::make_shared<PtrType>(ChannelDstPtrDType(args, 1, "channel.remote_ptr"));
}
} // namespace

REGISTER_OP("dist.channel")
    .set_op_category("DistOp")
    .set_description("Open the caller's write-engine channel: omit peer (peer = -1) on a shared window to "
                     "ride the calling AIV's own jetty; on a per-endpoint window peer is the destination "
                     "rank whose jetty the channel rides, and omitting it traps; body is a stack object, "
                     "this op is the opaque token that binds it")
    .add_argument("window", "Window handle (WindowType)")
    .add_argument("peer", "Destination peer rank; -1 (omit it) selects the calling AIV's jetty on a shared window")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistChannelType(args, kwargs);
    });

REGISTER_OP("dist.buffer_channel")
    .set_op_category("DistOp")
    .set_description("Bind a caller-reserved UB staging buffer to an opened channel (the vendor "
                     "MakeBatchHandle workspace shape: linear WQEBB slots, one bulk publish per "
                     "submit); body is a stack object holding a pointer to the base channel, this "
                     "op is the opaque token that binds it")
    .add_argument("channel", "Base channel handle (opaque integer token)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistBufferChannelType(args, kwargs);
    });

REGISTER_OP("channel.write")
    .set_op_category("DistOp")
    .set_description("Async GM tensor -> remote window write through the channel's URMA jetty "
                     "(destination bound at channel creation); submit=true publishes immediately, "
                     "submit=false stages the WQE")
    .add_argument("channel", "Channel handle (opaque integer token)")
    .add_argument("dst_ptr", "Destination remote pointer from channel.remote_ptr (PtrType)")
    .add_argument("num_bytes", "Transfer length in bytes (integer scalar)")
    .add_argument("tensor", "Local GM source tensor (TensorType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistChannelWriteType(args, kwargs);
    });

REGISTER_OP("channel.submit")
    .set_op_category("DistOp")
    .set_description("Publish the channel's deferred write-engine WQEs (doorbell); ordering, not completion")
    .add_argument("channel", "Channel handle (opaque integer token)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistChannelSubmitType(args, kwargs);
    });

REGISTER_OP("buffer_channel.submit")
    .set_op_category("DistOp")
    .set_description("Publish the buffer channel's staged WQEBBs: one SQ reservation, one wrap-aware "
                     "bulk DMA, one head advance, one doorbell — one CQE for the whole batch; "
                     "empty batch is a no-op")
    .add_argument("channel", "Buffer channel handle (opaque integer token)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistBufferChannelSubmitType(args, kwargs);
    });

REGISTER_OP("buffer_channel.write")
    .set_op_category("DistOp")
    .set_description("Stage a GM tensor -> remote window write as WQEBBs in the buffer channel's UB "
                     "staging area: pure UB stores, nothing touches the SQ until buffer_channel.submit; "
                     "option is the staged body flag byte (a caller-set cqe is stripped)")
    .add_argument("channel", "Buffer channel handle (opaque integer token)")
    .add_argument("dst_ptr", "Destination remote pointer from channel.remote_ptr (PtrType)")
    .add_argument("num_bytes", "Transfer length in bytes (integer scalar)")
    .add_argument("tensor", "Local GM source tensor (TensorType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistBufferChannelAppendWriteType(args, kwargs);
    });

REGISTER_OP("buffer_channel.write_value")
    .set_op_category("DistOp")
    .set_description("Stage a scalar-word write (inline WQEBB) in the buffer channel's UB staging "
                     "area; option is the staged body flag byte (a caller-set cqe is stripped)")
    .add_argument("channel", "Buffer channel handle (opaque integer token)")
    .add_argument("dst_ptr", "Destination remote pointer from channel.remote_ptr (PtrType)")
    .add_argument("value", "Scalar word to write (numeric scalar)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistBufferChannelAppendWriteValueType(args, kwargs);
    });

REGISTER_OP("buffer_channel.read")
    .set_op_category("DistOp")
    .set_description("Stage an engine read as WQEBBs in the buffer channel's UB staging area: the "
                     "fetched bytes are engine-written and readable only after drain; option is the "
                     "staged body flag byte (a caller-set cqe is stripped)")
    .add_argument("channel", "Buffer channel handle (opaque integer token)")
    .add_argument("src_ptr", "Source remote pointer from channel.remote_ptr (PtrType)")
    .add_argument("num_bytes", "Transfer length in bytes (integer scalar)")
    .add_argument("tensor", "Local GM destination tensor (TensorType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistBufferChannelReadType(args, kwargs);
    });

REGISTER_OP("buffer_channel.write_reduce")
    .set_op_category("DistOp")
    .set_description("Stage an engine-side reduction as WQEBBs in the buffer channel's UB staging "
                     "area (UDF-flagged WRITE WQEs); reduce_op selects ReduceOp.MAX/MIN/SUM and "
                     "option is the staged body flag byte (a caller-set cqe is stripped)")
    .add_argument("channel", "Buffer channel handle (opaque integer token)")
    .add_argument("dst_ptr", "Destination remote pointer from channel.remote_ptr (PtrType)")
    .add_argument("num_bytes", "Transfer length in bytes (integer scalar)")
    .add_argument("tensor", "Local GM source tensor (TensorType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistBufferChannelWriteReduceType(args, kwargs);
    });

REGISTER_OP("buffer_channel.atomic_fetch_add")
    .set_op_category("DistOp")
    .set_description("Stage a remote fetch-and-add (two WQEBBs: header + immediate payload) in the "
                     "buffer channel's UB staging area; fetch holds the previous remote value after "
                     "drain")
    .add_argument("channel", "Buffer channel handle (opaque integer token)")
    .add_argument("dst_ptr", "Target remote pointer from channel.remote_ptr (PtrType)")
    .add_argument("fetch", "Local one-element GM tensor receiving the previous value (TensorType)")
    .add_argument("val", "Addend (integer scalar)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistBufferChannelAtomicFetchAddType(args, kwargs);
    });

REGISTER_OP("buffer_channel.write_notify")
    .set_op_category("DistOp")
    .set_description("Stage a write-with-notify WQE (two WQEBBs) in the buffer channel's UB staging "
                     "area; the signal fires when the batch is published")
    .add_argument("channel", "Buffer channel handle (opaque integer token)")
    .add_argument("dst_ptr", "Destination remote pointer from channel.remote_ptr (PtrType)")
    .add_argument("num_bytes", "Transfer size in bytes (integer scalar)")
    .add_argument("tensor", "GM source tensor (TensorType)")
    .add_argument("signal_ptr", "64-bit integer remote signal pointer (PtrType)")
    .add_argument("val", "Notify value written to the signal (integer scalar)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistBufferChannelWriteNotifyType(args, kwargs);
    });

REGISTER_OP("buffer_channel.atomic_cas")
    .set_op_category("DistOp")
    .set_description("Stage a remote compare-and-swap (two WQEBBs: header + immediate payload) in "
                     "the buffer channel's UB staging area; fetch holds the previous remote value "
                     "after drain; success iff fetch == expected")
    .add_argument("channel", "Buffer channel handle (opaque integer token)")
    .add_argument("dst_ptr", "Target remote pointer from channel.remote_ptr (PtrType)")
    .add_argument("fetch", "Local one-element GM tensor receiving the previous value (TensorType)")
    .add_argument("expected", "Compare value (integer scalar)")
    .add_argument("desired", "Swap value (integer scalar)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistBufferChannelAtomicCasType(args, kwargs);
    });

REGISTER_OP("channel.drain")
    .set_op_category("DistOp")
    .set_description("Block until the channel's previously enqueued writes complete; timeout traps")
    .add_argument("channel", "Channel handle (opaque integer token)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistChannelDrainType(args, kwargs);
    });

REGISTER_OP("channel.lock")
    .set_op_category("DistOp")
    .set_description("Take the channel's cross-AIV lock: no-op when the channel rides the calling "
                     "AIV's own jetty (shared window), otherwise spin on the channel lock word and "
                     "reload the cached producer head from the shared counter")
    .add_argument("channel", "Channel handle (opaque integer token)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistChannelLockType(args, kwargs);
    });

REGISTER_OP("channel.unlock")
    .set_op_category("DistOp")
    .set_description("Release the channel's cross-AIV lock, publishing the cached producer state "
                     "first; no-op on a shared window")
    .add_argument("channel", "Channel handle (opaque integer token)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistChannelUnlockType(args, kwargs);
    });

REGISTER_OP("channel.is_shared")
    .set_op_category("DistOp")
    .set_description("True when the window is shared by the peers, so the channel rides the "
                     "calling AIV's private queue; false for a per-peer jetty")
    .add_argument("channel", "Channel handle (opaque integer token)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistChannelIsSharedType(args, kwargs);
    });

REGISTER_OP("dist.remote_ptr")
    .set_op_category("DistOp")
    .set_description("Typed pointer to a registered region of a peer's window (UB_MEM plane), "
                     "resolved from a local pointer into that region; advance with ptr.addptr and use "
                     "through ptr.make_tensor")
    .add_argument("window", "Window handle (WindowType)")
    .add_argument("ptr", "Pointer into a registered memory region (PtrType); the region is resolved on device")
    .add_argument("peer", "Target rank (integer scalar)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistRemotePtrType(args, kwargs);
    });

REGISTER_OP("channel.remote_ptr")
    .set_op_category("DistOp")
    .set_description("Typed pointer to a registered region in the channel's peer (URMA plane), resolved "
                     "from a local pointer into that region; pass the result to ch.write*")
    .add_argument("channel", "Channel handle (opaque integer token)")
    .add_argument("ptr", "Pointer into a registered memory region (PtrType); the region is resolved on device")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistChannelRemotePtrType(args, kwargs);
    });

namespace {
TypePtr DeduceDistChannelWriteValueType(const std::vector<ExprPtr>& args,
                                        const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x3)
        << "channel.write_value requires 3 arguments [channel, dst_ptr, value], but got " << args.size();
    CheckChannelOperand(args, "channel.write_value");
    const DataType dstDType = ChannelDstPtrDType(args, 1, "channel.write_value");
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, kIsNumericScalar(args[2]))
        << "channel.write_value value must be a numeric scalar, but got " << args[2]->GetType()->TypeName();

    (void)ParseChannelWriteKwargs(kwargs, "channel.write_value", true, true);
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsSupportedScalarDType(dstDType))
        << "channel.write_value does not support dtype " << dstDType.ToString()
        << "; supported: INT32/UINT32/INT64/UINT64/FP32";
    const auto valueType = As<ScalarType>(args[2]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, !(dstDType.IsInt() && valueType->dtype_.IsFloat()))
        << "channel.write_value integer destination requires an integer value, but got "
        << valueType->dtype_.ToString();
    return GetNoneType();
}

TypePtr DeduceDistChannelReadType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x4)
        << "channel.read requires 4 arguments [channel, src_ptr, num_bytes, tensor], but got " << args.size();
    CheckChannelOperand(args, "channel.read");
    (void)ChannelDstPtrDType(args, 1, "channel.read");
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsIntScalar(args[2]))
        << "channel.read num_bytes must be an integer scalar, but got " << args[2]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<TensorType>(args[3]->GetType()))
        << "channel.read destination must be a TensorType, but got " << args[3]->GetType()->TypeName();
    (void)ParseChannelWriteKwargs(kwargs, "channel.read", true, true);
    return GetNoneType();
}

TypePtr DeduceDistChannelWriteReduceType(const std::vector<ExprPtr>& args,
                                         const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x4)
        << "channel.write_reduce requires 4 arguments [channel, dst_ptr, num_bytes, tensor], but got " << args.size();
    CheckChannelOperand(args, "channel.write_reduce");
    (void)ChannelDstPtrDType(args, 1, "channel.write_reduce");
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsIntScalar(args[2]))
        << "channel.write_reduce num_bytes must be an integer scalar, but got " << args[2]->GetType()->TypeName();
    const auto tensorType = As<TensorType>(args[3]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, tensorType)
        << "channel.write_reduce source must be a TensorType, but got " << args[3]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsSupportedReduceDType(tensorType->dtype_))
        << "channel.write_reduce does not support dtype " << tensorType->dtype_.ToString()
        << "; supported: INT8/INT16/INT32/UINT32/FP16/FP32/BF16";
    bool hasReduceOp = false;
    int reduceOp = 0;
    bool hasOption = false;
    int option = 0x2D;
    for (const auto& [name, value] : kwargs) {
        if (name == "reduce_op") {
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, value.type() == typeid(int64_t))
                << "channel.write_reduce 'reduce_op' must be a compile-time ReduceOp";
            reduceOp = AnyCastFromIR<int>(value, name);
            hasReduceOp = true;
        } else if (name == "submit") {
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, value.type() == typeid(bool))
                << "channel.write_reduce 'submit' must be a compile-time bool";
        } else if (name == "timeout_cycles") {
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, value.type() == typeid(int64_t))
                << "channel.write_reduce 'timeout_cycles' must be a compile-time constant integer";
        } else if (name == "option") {
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, value.type() == typeid(int64_t))
                << "channel.write_reduce 'option' must be a compile-time constant integer";
            option = AnyCastFromIR<int>(value, name);
            hasOption = true;
        } else {
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, false)
                << "channel.write_reduce does not accept keyword argument '" << name << "'";
        }
    }
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, hasReduceOp)
        << "channel.write_reduce requires the reduce_op keyword (pld.ReduceOp.SUM/MAX/MIN)";
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, reduceOp == 0x8 || reduceOp == 0x9 || reduceOp == 0xA)
        << "channel.write_reduce reduce_op must be a pld.ReduceOp member (SUM=10/MAX=8/MIN=9), got " << reduceOp;
    if (hasOption) {
        // Same WQE flag-byte validation ParseChannelWriteKwargs applies to the writes.
        PRO_IR_CHECK(ExternalError::INVALID_VAL, option >= 0 && (option & ~0x2F) == 0)
            << "channel.write_reduce option sets unsupported WQE flag bits, got " << option;
        PRO_IR_CHECK(ExternalError::INVALID_VAL, (option & 0x20) != 0)
            << "channel.write_reduce option must include cqe (drain waits for one completion per WQE), got " << option;
        PRO_IR_CHECK(ExternalError::INVALID_VAL, (option & 0x3) != 0x3)
            << "channel.write_reduce option cannot set the relaxed and strong order bits together, got " << option;
    }
    return GetNoneType();
}

TypePtr DeduceDistChannelWriteNotifyType(const std::vector<ExprPtr>& args,
                                         const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x6)
        << "channel.write_notify requires 6 arguments [channel, dst_ptr, num_bytes, tensor, signal_ptr, val], "
           "but got "
        << args.size();
    CheckChannelOperand(args, "channel.write_notify");
    (void)ChannelDstPtrDType(args, 1, "channel.write_notify");
    for (size_t i : {size_t{2}, size_t{5}}) {
        PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsIntScalar(args[i]))
            << "channel.write_notify argument " << i << " must be an integer scalar, but got "
            << args[i]->GetType()->TypeName();
    }
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsAtomicIntPtr(args[4], /*allow_32=*/false))
        << "channel.write_notify signal address must be a 64-bit integer pointer "
           "(INT64/UINT64; the hardware notify value is 64-bit), but got "
        << args[4]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, As<TensorType>(args[3]->GetType()))
        << "channel.write_notify tensor must be a TensorType, but got " << args[3]->GetType()->TypeName();
    (void)ParseChannelWriteKwargs(kwargs, "channel.write_notify", false, true);
    return GetNoneType();
}

TypePtr DeduceDistChannelAtomicFetchAddType(const std::vector<ExprPtr>& args,
                                            const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x4)
        << "channel.atomic_fetch_add requires 4 arguments [channel, dst_ptr, fetch_tensor, value], but got "
        << args.size();
    CheckChannelOperand(args, "channel.atomic_fetch_add");
    const DataType dstDType = ChannelDstPtrDType(args, 1, "channel.atomic_fetch_add");
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsAtomicDType(dstDType))
        << "channel.atomic_fetch_add destination must point at a 32/64-bit integer word, but got dtype "
        << dstDType.ToString();
    const auto fetchType = As<TensorType>(args[2]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, fetchType)
        << "channel.atomic_fetch_add fetch destination must be a TensorType, but got "
        << args[2]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, fetchType->dtype_ == dstDType)
        << "channel.atomic_fetch_add fetch dtype must match the destination pointer (" << dstDType.ToString()
        << "), but got " << fetchType->dtype_.ToString();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsIntScalar(args[3]))
        << "channel.atomic_fetch_add value must be an integer scalar, but got " << args[3]->GetType()->TypeName();
    (void)ParseChannelWriteKwargs(kwargs, "channel.atomic_fetch_add", true, true);
    return GetNoneType();
}

TypePtr DeduceDistChannelAtomicCasType(const std::vector<ExprPtr>& args,
                                       const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 0x5)
        << "channel.atomic_cas requires 5 arguments [channel, dst_ptr, fetch_tensor, expected, desired], but "
           "got "
        << args.size();
    CheckChannelOperand(args, "channel.atomic_cas");
    const DataType dstDType = ChannelDstPtrDType(args, 1, "channel.atomic_cas");
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsAtomicDType(dstDType))
        << "channel.atomic_cas destination must point at a 32/64-bit integer word, but got dtype "
        << dstDType.ToString();
    const auto fetchType = As<TensorType>(args[2]->GetType());
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, fetchType)
        << "channel.atomic_cas fetch destination must be a TensorType, but got " << args[2]->GetType()->TypeName();
    PRO_IR_CHECK(ExternalError::INVALID_TYPE, fetchType->dtype_ == dstDType)
        << "channel.atomic_cas fetch dtype must match the destination pointer (" << dstDType.ToString() << "), but got "
        << fetchType->dtype_.ToString();
    for (size_t i : {size_t{3}, size_t{4}}) {
        PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsIntScalar(args[i]))
            << "channel.atomic_cas argument " << i << " must be an integer scalar, but got "
            << args[i]->GetType()->TypeName();
    }
    (void)ParseChannelWriteKwargs(kwargs, "channel.atomic_cas", true, true);
    return GetNoneType();
}

} // namespace

REGISTER_OP("channel.write_value")
    .set_op_category("DistOp")
    .set_description("Async scalar-word write into the peer's window through the channel's jetty "
                     "(inline WQE; destination bound at channel creation); the value dtype is the "
                     "destination pointer's element type")
    .add_argument("channel", "Channel handle (opaque integer token)")
    .add_argument("dst_ptr", "Destination remote pointer from channel.remote_ptr (PtrType)")
    .add_argument("value", "Scalar word to write (numeric scalar)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistChannelWriteValueType(args, kwargs);
    });

REGISTER_OP("channel.write_notify")
    .set_op_category("DistOp")
    .set_description("Async GM tensor -> remote window write fused with a hardware notify on payload "
                     "completion (one WriteWithNotify WQE, born published; destination bound at channel "
                     "creation)")
    .add_argument("channel", "Channel handle (opaque integer token)")
    .add_argument("dst_ptr", "Destination remote pointer from channel.remote_ptr (PtrType)")
    .add_argument("num_bytes", "Transfer length in bytes (integer scalar)")
    .add_argument("tensor", "Local GM source tensor (TensorType)")
    .add_argument("signal_ptr", "Signal address, a 64-bit integer pointer (INT64/UINT64)")
    .add_argument("val", "64-bit value written on completion (integer scalar)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistChannelWriteNotifyType(args, kwargs);
    });

REGISTER_OP("channel.atomic_fetch_add")
    .set_op_category("DistOp")
    .set_description("Async remote fetch-and-add of one aligned 32/64-bit integer word (FAA WQE; the "
                     "PREVIOUS remote value lands in the local fetch tensor, readable after "
                     "channel.drain)")
    .add_argument("channel", "Channel handle (opaque integer token)")
    .add_argument("dst_ptr", "Target remote pointer from channel.remote_ptr (PtrType)")
    .add_argument("fetch_tensor", "Local one-element GM tensor receiving the previous value (TensorType)")
    .add_argument("value", "Addend (integer scalar)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistChannelAtomicFetchAddType(args, kwargs);
    });

REGISTER_OP("channel.atomic_cas")
    .set_op_category("DistOp")
    .set_description("Async remote compare-and-swap of one aligned 32/64-bit integer word (CAS WQE; the "
                     "PREVIOUS remote value lands in the local fetch tensor either way — success iff it "
                     "equals expected; readable after channel.drain)")
    .add_argument("channel", "Channel handle (opaque integer token)")
    .add_argument("dst_ptr", "Target remote pointer from channel.remote_ptr (PtrType)")
    .add_argument("fetch_tensor", "Local one-element GM tensor receiving the previous value (TensorType)")
    .add_argument("expected", "Compare value (integer scalar)")
    .add_argument("desired", "Swap value (integer scalar)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistChannelAtomicCasType(args, kwargs);
    });

REGISTER_OP("channel.read")
    .set_op_category("DistOp")
    .set_description("Async remote window -> local GM tensor read through the channel's jetty (one "
                     "READ WQE per 256 MiB chunk; destination bound at channel creation); the fetched "
                     "bytes are readable after channel.drain")
    .add_argument("channel", "Channel handle (opaque integer token)")
    .add_argument("src_ptr", "Source remote pointer from channel.remote_ptr (PtrType)")
    .add_argument("num_bytes", "Transfer length in bytes (integer scalar)")
    .add_argument("tensor", "Local GM destination tensor (TensorType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistChannelReadType(args, kwargs);
    });

REGISTER_OP("channel.write_reduce")
    .set_op_category("DistOp")
    .set_description("Async engine-side element-wise reduction of a local GM tensor into the peer's "
                     "window (one WRITE_WITH_REDUCE WQE per 256 MiB chunk; destination bound at channel "
                     "creation; concurrent reduces to one target are hardware-serialized)")
    .add_argument("channel", "Channel handle (opaque integer token)")
    .add_argument("dst_ptr", "Destination remote pointer from channel.remote_ptr (PtrType)")
    .add_argument("num_bytes", "Transfer length in bytes (integer scalar)")
    .add_argument("tensor", "Local GM source tensor (TensorType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        return DeduceDistChannelWriteReduceType(args, kwargs);
    });

} // namespace ir
} // namespace pypto
