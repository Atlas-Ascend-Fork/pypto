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
 * @file backend_cce_atomic_ops.cpp
 * \brief CCE codegen for the scalar atomic block ops.
 */

#include <string>

#include "backend/backend_cce.h"
#include "backend/common/backend.h"
#include "backend/common/backend_utils.h"
#include "codegen/cce/cce_codegen.h"
#include "codegen/codegen_base.h"
#include "core/logging.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/op_attr_types.h"
#include "ir/pipe.h"
#include "ir/type.h"
#include "pypto_pro/error.h"
#include "tilefwk/error.h"

namespace pypto {
namespace backend {
using npu::tile_fwk::ExternalError;

namespace {

const char* AtomicRMWIntrinsic(ir::AtomicRMWMode kind)
{
    switch (kind) {
        case ir::AtomicRMWMode::ADD:
            return "atomicAdd";
        case ir::AtomicRMWMode::SUB:
            return "atomicSub";
        case ir::AtomicRMWMode::EXCH:
            return "atomicExch";
        case ir::AtomicRMWMode::MAX:
            return "atomicMax";
        case ir::AtomicRMWMode::MIN:
            return "atomicMin";
        case ir::AtomicRMWMode::INC:
            return "atomicInc";
        case ir::AtomicRMWMode::DEC:
            return "atomicDec";
        case ir::AtomicRMWMode::AND:
            return "atomicAnd";
        case ir::AtomicRMWMode::OR:
            return "atomicOr";
        case ir::AtomicRMWMode::XOR:
            return "atomicXOr";
        default:
            return nullptr;
    }
}

std::string AtomicElementPointer(codegen::CCECodegen& codegen, const ir::CallPtr& op, ir::DataType* elementDtype)
{
    auto tensorVar = ir::As<ir::Var>(op->args_[0]);
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, tensorVar != nullptr)
        << op->name_ << " tensor must be a Tensor variable";
    auto tensorType = ir::As<ir::TensorType>(tensorVar->GetType());
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, tensorType != nullptr)
        << op->name_ << " tensor must have TensorType";
    *elementDtype = tensorType->dtype_;
    return codegen.GetPointer(codegen.GetVarName(tensorVar));
}

std::string AtomicOperand(codegen::CCECodegen& codegen, const ir::ExprPtr& operand, const ir::DataType& elementDtype)
{
    auto scalarType = ir::As<ir::ScalarType>(operand->GetType());
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, scalarType != nullptr)
        << "atomic operand must be ScalarType";
    return codegen.ScalarCast(codegen.GetExprAsCode(operand), scalarType->dtype_, elementDtype);
}

std::string EmitAtomicCall(codegen::CCECodegen& codegen, const ir::CallPtr& op, const std::string& call)
{
    // The intrinsic returns the element-typed pre-op value, and the IR result
    // type is that same element dtype, so the binding needs no extra cast.
    auto resultType = ir::As<ir::ScalarType>(op->GetType());
    PRO_CODEGEN_INTERNAL_CHECK(npu::tile_fwk::InternalError::CODEGEN_INNER_ERROR, resultType != nullptr)
        << op->name_ << " result must be ScalarType";
    std::string result = "__atomic_old_" + std::to_string(codegen.GetTileOffsetCounter());
    codegen.Emit("auto " + result + " = " + call + ";");
    return result;
}

std::string MakeBlockAtomicRMWCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 3)
        << "block.atomic_rmw requires tensor, offset and value";
    PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, codegen.GetTarget() == ir::SectionKind::Vector)
        << "block.atomic_rmw is only supported in a Vector section";

    const auto kind = static_cast<ir::AtomicRMWMode>(op->GetKwarg<int>("kind"));
    const char* intrinsic = AtomicRMWIntrinsic(kind);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, intrinsic != nullptr)
        << "block.atomic_rmw kind " << static_cast<int>(kind) << " has no device intrinsic";

    ir::DataType elementDtype;
    const std::string pointer = AtomicElementPointer(codegen, op, &elementDtype);
    const std::string offset = codegen.GetExprAsCode(op->args_[1]);
    const std::string value = AtomicOperand(codegen, op->args_[2], elementDtype);
    const std::string call = std::string(intrinsic) + "((__gm__ " + elementDtype.ToCTypeString() + "*)" + pointer +
                             " + (" + offset + "), " + value + ")";
    return EmitAtomicCall(codegen, op, call);
}

std::string MakeBlockAtomicCasCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegenBase)
{
    auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegenBase);
    PRO_CODEGEN_CHECK(ExternalError::INVALID_ARGUMENT, op->args_.size() == 4)
        << "block.atomic_cas requires tensor, offset, compare and value";
    PRO_CODEGEN_CHECK(ExternalError::NOT_IMPLEMENTED_ERROR, codegen.GetTarget() == ir::SectionKind::Vector)
        << "block.atomic_cas is only supported in a Vector section";

    ir::DataType elementDtype;
    const std::string pointer = AtomicElementPointer(codegen, op, &elementDtype);
    const std::string offset = codegen.GetExprAsCode(op->args_[1]);
    const std::string compare = AtomicOperand(codegen, op->args_[2], elementDtype);
    const std::string value = AtomicOperand(codegen, op->args_[3], elementDtype);
    const std::string call = "atomicCAS((__gm__ " + elementDtype.ToCTypeString() + "*)" + pointer + " + (" + offset +
                             "), " + compare + ", " + value + ")";
    return EmitAtomicCall(codegen, op, call);
}

} // namespace

REGISTER_BACKEND_OP(BackendCCE, "block.atomic_rmw")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeBlockAtomicRMWCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(BackendCCE, "block.atomic_cas")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
        return MakeBlockAtomicCasCodegenCCE(op, codegen);
    });

} // namespace backend
} // namespace pypto
