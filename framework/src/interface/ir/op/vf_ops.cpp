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
 * @file vf_ops.cpp
 * \brief VF API IR op registration.
 *
 * Registers VF ops with type deduction for the direct VF instruction path.
 * These ops are part of the merged shared IR surface and remain consumed by
 * the block VF frontend/backend path.
 */

#include <any>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/any_cast.h"
#include "core/error.h"
#include "core/logging.h"
#include "ir/kind_traits.h"
#include "ir/op_attr_types.h"
#include "ir/op_registry.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

namespace {

TypePtr DeduceVFUnknownType([[maybe_unused]] const std::vector<ExprPtr>& args,
                            [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    return GetUnknownType();
}

static DataType VfDeduceDataDtype(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs);

// MaskReg dtype is not a semantic attribute: instruction variants are selected
// from the call's kwargs by the backend, never from this type. Always carry the
// UINT16 placeholder (the historical mask-register default).
TypePtr DeduceVFMaskType([[maybe_unused]] const std::vector<ExprPtr>& args,
                         [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    return std::make_shared<RegType>(RegKind::PredReg, DataType::UINT16);
}

// Compare ops carry the compared register's dtype as a passive placeholder
// (instruction variants are selected from the call's kwargs, never from this
// type).
TypePtr DeduceVFMaskTypeFromSrc(const std::vector<ExprPtr>& args,
                                [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    if (!args.empty()) {
        if (IsRegTensor(args[0]->GetType())) {
            return std::make_shared<RegType>(RegKind::PredReg, As<RegType>(args[0]->GetType())->dtype_);
        }
        if (auto st = As<ScalarType>(args[0]->GetType())) {
            return std::make_shared<RegType>(RegKind::PredReg, st->dtype_);
        }
    }
    return std::make_shared<RegType>(RegKind::PredReg, DataType::UINT16);
}

TypePtr DeduceVFAddrRegType([[maybe_unused]] const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    for (const auto& [key, value] : kwargs) {
        if (key == "dtype") {
            return std::make_shared<RegType>(RegKind::AddrReg, AnyCast<DataType>(value, "kwarg: dtype"));
        }
    }
    return std::make_shared<RegType>(RegKind::AddrReg, DataType::UINT32);
}

TypePtr DeduceVFUnalignRegType([[maybe_unused]] const std::vector<ExprPtr>& args,
                               [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    return std::make_shared<RegType>(RegKind::UnalignReg, DataType::UINT32);
}

// ============================================================================
// Type deduction for value-form calls (the single deduce entry point)
//
// A value-form Call carries only its source args; f_deduce_type computes the
// destination register type(s) — RegTensor / MaskReg / a TupleType of them for
// dual-dst ops — from the args and kwargs alone (arg-complete, like block
// ops), which the codegen mirrors when synthesizing the dst declaration.
// ============================================================================

// dtype kwarg wins; otherwise the first data-carrying source's dtype decides.
// Single RegTensor dst.
TypePtr DeduceVFRegType(const std::vector<ExprPtr>& args, const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    return std::make_shared<RegType>(RegKind::VecReg, VfDeduceDataDtype(args, kwargs));
}

// Unified ops: the dst register kind follows the first source (MaskReg src →
// MaskReg dst).
TypePtr DeduceVFRegLikeType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    if (!args.empty() && IsMaskReg(args[0]->GetType())) {
        return std::make_shared<RegType>(RegKind::PredReg, VfDeduceDataDtype(args, kwargs));
    }
    return std::make_shared<RegType>(RegKind::VecReg, VfDeduceDataDtype(args, kwargs));
}

// Type-changing ops with fixed RegTensor dst: the dtype kwarg is mandatory
// because the dst type differs from the src and no default can be inferred.
TypePtr DeduceVFTypeChangingRegType([[maybe_unused]] const std::vector<ExprPtr>& args,
                                    const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    for (const auto& [key, value] : kwargs) {
        if (key == "dtype") {
            return std::make_shared<RegType>(RegKind::VecReg, AnyCast<DataType>(value, "kwarg: dtype"));
        }
    }
    throw TypeError("vf type-changing op requires an explicit dtype kwarg because the dst type "
                    "differs from the src. Example: vf.astype(src, mask, dtype=pl.DT_FP16)");
}

// Type-changing unified ops (pack/unpack): the dtype kwarg is mandatory for
// RegTensor sources; the kind follows the source (MaskReg src → MaskReg dst).
TypePtr DeduceVFTypeChangingLikeType(const std::vector<ExprPtr>& args,
                                     const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    for (const auto& [key, value] : kwargs) {
        if (key == "dtype") {
            auto dt = AnyCast<DataType>(value, "kwarg: dtype");
            if (!args.empty() && IsMaskReg(args[0]->GetType())) {
                return std::make_shared<RegType>(RegKind::PredReg, dt);
            }
            return std::make_shared<RegType>(RegKind::VecReg, dt);
        }
    }
    if (!args.empty()) {
        if (IsMaskReg(args[0]->GetType())) {
            return std::make_shared<RegType>(RegKind::PredReg, As<RegType>(args[0]->GetType())->dtype_);
        }
    }
    throw TypeError("vf type-changing op requires an explicit dtype kwarg because the dst type "
                    "differs from the src. Example: vf.pack(src, mask, dtype=pl.DT_FP16)");
}

// arange: the dtype kwarg is mandatory. The start value is a scalar literal
// whose dtype is only a placeholder, so it cannot decide the element type of
// the generated sequence.
TypePtr DeduceVFArangeType([[maybe_unused]] const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    for (const auto& [key, value] : kwargs) {
        if (key == "dtype") {
            return std::make_shared<RegType>(RegKind::VecReg, AnyCast<DataType>(value, "kwarg: dtype"));
        }
    }
    throw TypeError("vf.arange requires an explicit dtype kwarg because the destination element type "
                    "cannot be inferred from the start value. Example: vf.arange(0, dtype=pl.DT_INT32)");
}

// histograms: the hardware chistv2/dhistv2 counters are always uint16 bins.
TypePtr DeduceVFHistogramsType([[maybe_unused]] const std::vector<ExprPtr>& args,
                               [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    return std::make_shared<RegType>(RegKind::VecReg, DataType::UINT16);
}

// First data-carrying dtype: the dtype kwarg wins, otherwise the first
// RegTensor / Scalar / Shaped / MaskReg source's dtype.
static DataType VfDeduceDataDtype(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    for (const auto& [key, value] : kwargs) {
        if (key == "dtype") {
            return AnyCast<DataType>(value, "kwarg: dtype");
        }
    }
    for (const auto& arg : args) {
        auto type = arg->GetType();
        if (IsRegTensor(type)) {
            return As<RegType>(type)->dtype_;
        }
        if (auto st = As<ScalarType>(type)) {
            return st->dtype_;
        }
        if (auto sh = As<ShapedType>(type)) {
            return sh->dtype_;
        }
        if (IsMaskReg(type)) {
            return As<RegType>(type)->dtype_;
        }
    }
    throw TypeError("vf op requires an explicit dtype kwarg or a data-carrying source "
                    "operand to deduce the result dtype");
}

// load_align: the value-form call carries [tile, (offset | mask)]. The dst kind
// follows the dist kwarg: US/DS dists produce a MaskReg dst (plds), DINTLV_B*
// produces a pair of RegTensor dsts (the de-interleave load), every other dist
// a RegTensor (vlds). NORM is the only kind-ambiguous dist (plds/mask vs
// vlds/data): is_reg selects its kind (False -> MaskReg, True or absent ->
// RegTensor) and is ignored for every other dist.
TypePtr DeduceVFLoadAlignType(const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    bool hasIsReg = false;
    bool isMaskNorm = false;
    for (const auto& [key, value] : kwargs) {
        if (key == "is_reg") {
            hasIsReg = true;
            isMaskNorm = !AnyCast<bool>(value, "kwarg: is_reg");
        }
    }
    for (const auto& [key, value] : kwargs) {
        if (key == "dist") {
            auto dist = static_cast<LoadDist>(AnyCast<int>(value, "kwarg: dist"));
            switch (dist) {
                case LoadDist::DINTLV_B8:
                case LoadDist::DINTLV_B16:
                case LoadDist::DINTLV_B32: {
                    if (hasIsReg && isMaskNorm) {
                        throw TypeError("vf.load_align is_reg=False (MaskReg dst) does not support "
                                        "the DINTLV_B* de-interleave dist");
                    }
                    auto elem = std::make_shared<RegType>(RegKind::VecReg, VfDeduceDataDtype(args, kwargs));
                    return std::make_shared<TupleType>(std::vector<TypePtr>{elem, elem});
                }
                case LoadDist::US:
                case LoadDist::US_B8:
                case LoadDist::US_B16:
                case LoadDist::DS:
                case LoadDist::DS_B8:
                case LoadDist::DS_B16:
                    return std::make_shared<RegType>(RegKind::PredReg, VfDeduceDataDtype(args, kwargs));
                case LoadDist::NORM:
                    return std::make_shared<RegType>(isMaskNorm ? RegKind::PredReg : RegKind::VecReg,
                                                     VfDeduceDataDtype(args, kwargs));
                default:
                    return std::make_shared<RegType>(RegKind::VecReg, VfDeduceDataDtype(args, kwargs));
            }
        }
    }
    return std::make_shared<RegType>(isMaskNorm ? RegKind::PredReg : RegKind::VecReg, VfDeduceDataDtype(args, kwargs));
}

// mull: the 64-bit product is a pair of RegTensor halves.
TypePtr DeduceVFRegPairOfFirstSrc(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    auto elem = std::make_shared<RegType>(RegKind::VecReg, VfDeduceDataDtype(args, kwargs));
    return std::make_shared<TupleType>(std::vector<TypePtr>{elem, elem});
}

// interleave/de_interleave: the dst kind follows the first source
// (MaskReg src keeps the mask pair dialect).
TypePtr DeduceVFPairLikeFirstSrc(const std::vector<ExprPtr>& args,
                                 const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    bool maskKind = !args.empty() && IsMaskReg(args[0]->GetType());
    const DataType dt = VfDeduceDataDtype(args, kwargs);
    TypePtr elem;
    if (maskKind) {
        elem = std::make_shared<RegType>(RegKind::PredReg, dt);
    } else {
        elem = std::make_shared<RegType>(RegKind::VecReg, dt);
    }
    return std::make_shared<TupleType>(std::vector<TypePtr>{elem, elem});
}

// addc/subc: (carry-out MaskReg, dst RegTensor).
TypePtr DeduceVFCarryPairOfFirstSrc(const std::vector<ExprPtr>& args,
                                    const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    const DataType dt = VfDeduceDataDtype(args, kwargs);
    return std::make_shared<TupleType>(std::vector<TypePtr>{std::make_shared<RegType>(RegKind::PredReg, dt),
                                                            std::make_shared<RegType>(RegKind::VecReg, dt)});
}

} // namespace

// Mask
REGISTER_OP("vf.create_mask")
    .set_op_category("VFOp")
    .set_description("Declare and initialize a mask register")
    .no_argument()
    .set_attr<int>("pattern")
    .set_attr<DataType>("dtype")
    .f_deduce_type(DeduceVFMaskType);

// Initialization
REGISTER_OP("vf.full")
    .set_op_category("VFOp")
    .set_description("Scalar broadcast to register")
    .add_argument("dst", "Destination register")
    .add_argument("scalar", "Scalar value to broadcast")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .set_attr<int>("pos")
    .set_attr<DataType>("dtype")
    .f_deduce_type(DeduceVFRegType);

// Data transfer
REGISTER_OP("vf.load_align")
    .set_op_category("VFOp")
    .set_description("Unified vlds/plds load: MaskReg dst via US/DS dist (or is_reg=False with NORM), "
                     "RegTensor otherwise; "
                     "dual dst via dist=DINTLV_B* (lo, hi = vf.load_align(...))")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source UB pointer")
    .add_argument("offset", "Element offset or post-update stride")
    .set_attr<int>("data_copy_mode")
    .set_attr<int>("block_stride")
    .set_attr<int>("repeat_stride")
    .set_attr<bool>("post_update")
    .set_attr<int>("dist")
    .set_attr<bool>("is_reg")
    .set_attr<DataType>("dtype")
    .f_deduce_type(DeduceVFLoadAlignType);

REGISTER_OP("vf.store_align")
    .set_op_category("VFOp")
    .set_description(
        "Store aligned data from register to UB (MaskReg src via dist kwarg → psts/pst, RegTensor src → vsts/vst)")
    .add_argument("dst", "Destination UB pointer")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("dist")
    .set_attr<int>("data_copy_mode")
    .set_attr<int>("block_stride")
    .set_attr<int>("repeat_stride")
    .set_attr<bool>("post_update")
    .f_deduce_type(DeduceVFUnknownType);

// Compute
REGISTER_OP("vf.max")
    .set_op_category("VFOp")
    .set_description("Vector maximum")
    .add_argument("dst", "Destination register")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.add")
    .set_op_category("VFOp")
    .set_description("Vector addition")
    .add_argument("dst", "Destination register")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.sub")
    .set_op_category("VFOp")
    .set_description("Vector subtraction")
    .add_argument("dst", "Destination register")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.and_")
    .set_op_category("VFOp")
    .set_description("Bitwise AND")
    .add_argument("dst", "Destination register")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFRegLikeType);

REGISTER_OP("vf.xor")
    .set_op_category("VFOp")
    .set_description("Bitwise XOR")
    .add_argument("dst", "Destination register")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFRegLikeType);

REGISTER_OP("vf.or_")
    .set_op_category("VFOp")
    .set_description("Bitwise OR")
    .add_argument("dst", "Destination register")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFRegLikeType);

// Unified reduction: supports mode=SUM/MAX/MIN (new) and reduce_type=ADD/MAX (legacy)
REGISTER_OP("vf.reduce_sum")
    .set_op_category("VFOp")
    .set_description("Sum reduction across all lanes (vcadd/vcgadd)")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<bool>("datablock")
    .set_attr<int>("merge_mode")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.reduce_max")
    .set_op_category("VFOp")
    .set_description("Max reduction across all lanes (vcmax/vcgmax)")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<bool>("datablock")
    .set_attr<int>("merge_mode")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.reduce_min")
    .set_op_category("VFOp")
    .set_description("Min reduction across all lanes (vcmin/vcgmin)")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<bool>("datablock")
    .set_attr<int>("merge_mode")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.mul")
    .set_op_category("VFOp")
    .set_description("Vector multiplication")
    .add_argument("dst", "Destination register")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.mul_add_dst")
    .set_op_category("VFOp")
    .set_description("Fused multiply-add (FMA): dst = src0 * src1 + dst (hardware vmadd)")
    .add_argument("dst", "Destination/accumulator register (read+write)")
    .add_argument("src0", "Source register 0 (multiplicand)")
    .add_argument("src1", "Source register 1 (multiplicand)")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.mul_add_dst_acc")
    .set_op_category("VFOp")
    .set_description("Fused multiply-add with explicit dst: dst = src0 * src1 + old (vmula)")
    .add_argument("dst", "Destination/accumulator register (read+write)")
    .add_argument("src0", "Source register 0 (multiplicand)")
    .add_argument("src1", "Source register 1 (multiplicand)")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.div")
    .set_op_category("VFOp")
    .set_description("Vector division")
    .add_argument("dst", "Destination register")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .set_attr<bool>("precision")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.muls")
    .set_op_category("VFOp")
    .set_description("Scalar multiplication")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("scalar", "Scalar multiplier")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.ln")
    .set_op_category("VFOp")
    .set_description("Natural logarithm (vln instruction)")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .set_attr<bool>("precision")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.log")
    .set_op_category("VFOp")
    .set_description("Natural logarithm (alias of Ln, same vln instruction on A5)")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .set_attr<bool>("precision")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.min")
    .set_op_category("VFOp")
    .set_description("Vector minimum")
    .add_argument("dst", "Destination register")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.exp")
    .set_op_category("VFOp")
    .set_description("Exponential function")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .set_attr<bool>("precision")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.abs")
    .set_op_category("VFOp")
    .set_description("Absolute value")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.not_")
    .set_op_category("VFOp")
    .set_description("Bitwise NOT")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFRegLikeType);

REGISTER_OP("vf.sqrt")
    .set_op_category("VFOp")
    .set_description("Square root")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .set_attr<bool>("precision")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.relu")
    .set_op_category("VFOp")
    .set_description("ReLU activation")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.neg")
    .set_op_category("VFOp")
    .set_description("Negate")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.adds")
    .set_op_category("VFOp")
    .set_description("Scalar addition")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("scalar", "Scalar addend")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.mins")
    .set_op_category("VFOp")
    .set_description("Scalar minimum")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("scalar", "Scalar value")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.maxs")
    .set_op_category("VFOp")
    .set_description("Scalar maximum")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("scalar", "Scalar value")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.leaky_relu")
    .set_op_category("VFOp")
    .set_description("Leaky ReLU activation")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("alpha", "Negative slope scalar")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.interleave")
    .set_op_category("VFOp")
    .set_description("Interleave two registers (vintlv for RegTensor, pintlv for MaskReg)")
    .add_argument("dst0", "Destination register 0")
    .add_argument("dst1", "Destination register 1")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .set_attr<DataType>("dtype")
    .f_deduce_type(DeduceVFPairLikeFirstSrc);

REGISTER_OP("vf.pair_reduce_sum")
    .set_op_category("VFOp")
    .set_description("Pairwise reduction sum")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.abs_sub")
    .set_op_category("VFOp")
    .set_description("Absolute difference |src0 - src1|")
    .add_argument("dst", "Destination register")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.axpy")
    .set_op_category("VFOp")
    .set_description("Accumulate: dst = src * scalar + dst")
    .add_argument("dst", "Destination/accumulator register (read+write)")
    .add_argument("src", "Source register")
    .add_argument("scalar", "Scalar multiplier")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.axpy_acc")
    .set_op_category("VFOp")
    .set_description("Accumulate with explicit dst: dst = src * scalar + old (vaxpy)")
    .add_argument("dst", "Destination/accumulator register (read+write)")
    .add_argument("src", "Source register")
    .add_argument("scalar", "Scalar multiplier")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.bit_cast")
    .set_op_category("VFOp")
    .set_description("Bitwise type reinterpretation (no value conversion)")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .set_attr<DataType>("dtype")
    .f_deduce_type(DeduceVFTypeChangingRegType);

REGISTER_OP("vf.mul_dst_add")
    .set_op_category("VFOp")
    .set_description("Multiply-dst-add: dst = dst + src0 * src1 (vmadd)")
    .add_argument("dst", "Destination/accumulator register (read+write)")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.mul_dst_add_acc")
    .set_op_category("VFOp")
    .set_description("Multiply-add with an explicit dst operand: dst = dst * src0 + src1 (vmadd)")
    .add_argument("dst", "Destination/accumulator register (read+write)")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.pack")
    .set_op_category("VFOp")
    .set_description("Pack/narrow data type (e.g. u16->u8). vpack for RegTensor, ppack for MaskReg.")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .set_attr<DataType>("dtype")
    .set_attr<int>("part")
    .f_deduce_type(DeduceVFTypeChangingLikeType);

REGISTER_OP("vf.unpack")
    .set_op_category("VFOp")
    .set_description("Unpack/widen data type (e.g. u8->u16). vunpack for RegTensor, punpack for MaskReg.")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .set_attr<DataType>("dtype")
    .set_attr<int>("part")
    .f_deduce_type(DeduceVFTypeChangingLikeType);

REGISTER_OP("vf.prelu")
    .set_op_category("VFOp")
    .set_description("Parametric ReLU with per-element slope")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("slope", "Slope register (per-element)")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.shift_left")
    .set_op_category("VFOp")
    .set_description("Shift left by a scalar or per-lane amount")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("shift", "Integer scalar, or same-width 8/16/32-bit register (signed; also unsigned on DAV_9201)")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.shift_right")
    .set_op_category("VFOp")
    .set_description("Shift right by a scalar or per-lane amount")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("shift", "Integer scalar, or same-width 8/16/32-bit register (signed; also unsigned on DAV_9201)")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.mull")
    .set_op_category("VFOp")
    .set_description("Long multiply: 32x32->64, output split into lo/hi registers")
    .add_argument("dst_lo", "Low 32 bits of product")
    .add_argument("dst_hi", "High 32 bits of product")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("mask", "Mask register")
    .f_deduce_type(DeduceVFRegPairOfFirstSrc);

REGISTER_OP("vf.addc")
    .set_op_category("VFOp")
    .set_description("Add with carry: carry_out, dst = src0 + src1 + carry_in")
    .add_argument("carry_out", "Output carry flag register")
    .add_argument("dst", "Destination register (sum)")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("carry_in", "Input carry flag register")
    .add_argument("mask", "Mask register")
    .f_deduce_type(DeduceVFCarryPairOfFirstSrc);

REGISTER_OP("vf.subc")
    .set_op_category("VFOp")
    .set_description("Subtract with borrow: borrow_out, dst = src0 - src1 - borrow_in")
    .add_argument("borrow_out", "Output borrow flag register")
    .add_argument("dst", "Destination register (difference)")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .add_argument("borrow_in", "Input borrow flag register")
    .add_argument("mask", "Mask register")
    .f_deduce_type(DeduceVFCarryPairOfFirstSrc);

REGISTER_OP("vf.exp_sub")
    .set_op_category("VFOp")
    .set_description("Fused exp(src - max)")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("max", "Max register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("layout")
    .set_attr<DataType>("dtype")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.astype")
    .set_op_category("VFOp")
    .set_description("Type conversion")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<DataType>("dtype")
    .set_attr<int>("layout")
    .set_attr<int>("round_mode")
    .set_attr<int>("saturate")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFTypeChangingRegType);

REGISTER_OP("vf.de_interleave")
    .set_op_category("VFOp")
    .set_description("De-interleave (vdintlv for RegTensor, pdintlv for MaskReg)")
    .add_argument("dst0", "Destination register 0")
    .add_argument("dst1", "Destination register 1")
    .add_argument("src0", "Source register 0")
    .add_argument("src1", "Source register 1")
    .set_attr<DataType>("dtype")
    .f_deduce_type(DeduceVFPairLikeFirstSrc);

REGISTER_OP("vf.select")
    .set_op_category("VFOp")
    .set_description("Conditional select")
    .add_argument("dst", "Destination register")
    .add_argument("src_true", "True branch register")
    .add_argument("src_false", "False branch register")
    .add_argument("mask", "Mask register")
    .f_deduce_type(DeduceVFRegLikeType);

REGISTER_OP("vf.update_mask")
    .set_op_category("VFOp")
    .set_description("Update mask with scalar value")
    .add_argument("scalar", "Scalar value for mask")
    .set_attr<DataType>("dtype")
    .f_deduce_type(DeduceVFMaskType);

REGISTER_OP("vf.mem_bar")
    .set_op_category("VFOp")
    .set_description("Memory barrier")
    .no_argument()
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFUnknownType);

// TopK VF ops
REGISTER_OP("vf.histograms")
    .set_op_category("VFOp")
    .set_description("Histogram accumulation")
    .add_argument("dst", "Destination histogram register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("bin_type")
    .set_attr<int>("hist_type")
    .f_deduce_type(DeduceVFHistogramsType);

REGISTER_OP("vf.histogram_acc")
    .set_op_category("VFOp")
    .set_description("Histogram accumulation with explicit dst")
    .add_argument("dst", "Destination histogram register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("bin_type")
    .set_attr<int>("hist_type")
    .f_deduce_type(DeduceVFHistogramsType);

REGISTER_OP("vf.eq")
    .set_op_category("VFOp")
    .set_description("Equality compare and generate mask")
    .add_argument("dst", "Destination mask register")
    .add_argument("src0", "First source register")
    .add_argument("src1", "Second source register")
    .add_argument("mask_src", "Source mask register")
    .f_deduce_type(DeduceVFMaskTypeFromSrc);

REGISTER_OP("vf.ne")
    .set_op_category("VFOp")
    .set_description("Not-equal compare and generate mask")
    .add_argument("dst", "Destination mask register")
    .add_argument("src0", "First source register")
    .add_argument("src1", "Second source register")
    .add_argument("mask_src", "Source mask register")
    .f_deduce_type(DeduceVFMaskTypeFromSrc);

REGISTER_OP("vf.lt")
    .set_op_category("VFOp")
    .set_description("Less-than compare and generate mask")
    .add_argument("dst", "Destination mask register")
    .add_argument("src0", "First source register")
    .add_argument("src1", "Second source register")
    .add_argument("mask_src", "Source mask register")
    .f_deduce_type(DeduceVFMaskTypeFromSrc);

REGISTER_OP("vf.gt")
    .set_op_category("VFOp")
    .set_description("Greater-than compare and generate mask")
    .add_argument("dst", "Destination mask register")
    .add_argument("src0", "First source register")
    .add_argument("src1", "Second source register")
    .add_argument("mask_src", "Source mask register")
    .f_deduce_type(DeduceVFMaskTypeFromSrc);

REGISTER_OP("vf.le")
    .set_op_category("VFOp")
    .set_description("Less-or-equal compare and generate mask")
    .add_argument("dst", "Destination mask register")
    .add_argument("src0", "First source register")
    .add_argument("src1", "Second source register")
    .add_argument("mask_src", "Source mask register")
    .f_deduce_type(DeduceVFMaskTypeFromSrc);

REGISTER_OP("vf.ge")
    .set_op_category("VFOp")
    .set_description("Greater-or-equal compare and generate mask")
    .add_argument("dst", "Destination mask register")
    .add_argument("src0", "First source register")
    .add_argument("src1", "Second source register")
    .add_argument("mask_src", "Source mask register")
    .f_deduce_type(DeduceVFMaskTypeFromSrc);

REGISTER_OP("vf.squeeze")
    .set_op_category("VFOp")
    .set_description("Squeeze mask to indices")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("gather_mode")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.arange")
    .set_op_category("VFOp")
    .set_description("Generate index sequence (index_order=INCREASE_ORDER: vci INC_ORDER, DECREASE_ORDER: DEC_ORDER)")
    .add_argument("dst", "Destination register")
    .set_attr<int>("index_order")
    .set_attr<DataType>("dtype")
    .f_deduce_type(DeduceVFArangeType);

REGISTER_OP("vf.gather")
    .set_op_category("VFOp")
    .set_description("Gather elements by indices. Two forms: "
                     "(1) UB-to-Reg: gather(src_ub, indices, mask[, data_copy_mode]). "
                     "b8 sources are zero-extended to b16. "
                     "b16 sources support both uint16 and uint32 index. "
                     "(2) Reg-to-Reg: gather(src_reg, indices). "
                     "Source and destination must have the same type, supports INT8->INT8. "
                     "Backend dispatches based on src argument type (Tile vs RegTensor).")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source UB pointer or source register")
    .add_argument("indices", "Index register")
    .add_argument("mask", "Mask register (required for UB form, omitted for Reg form)")
    .set_attr<int>("data_copy_mode")
    .f_deduce_type(DeduceVFRegType);

// Two calling conventions (codegen branches on arg count):
//   store_unalign:      3 args [dst, src, align_reg] -> vstur (strideless)
//                       4 args [dst, vreg, ureg, stride] -> vstus (strided)
//   store_unalign_post: 2 args [dst, align_reg] -> vstar (strideless)
//                       3 args [dst, ureg, stride] -> vstas (strided)
// Declared at the strideless minimum; the extra strided arg is passed through.
REGISTER_OP("vf.store_unalign")
    .set_op_category("VFOp")
    .set_description("Store unaligned data with post update")
    .add_argument("dst_ptr", "Destination UB pointer")
    .add_argument("src", "Source register")
    .add_argument("align_reg", "Alignment register")
    .set_attr<bool>("post_update")
    .f_deduce_type(DeduceVFUnknownType);

REGISTER_OP("vf.store_unalign_post")
    .set_op_category("VFOp")
    .set_description("Complete unaligned store")
    .add_argument("dst_ptr", "Destination UB pointer")
    .add_argument("align_reg", "Alignment register")
    .set_attr<bool>("post_update")
    .f_deduce_type(DeduceVFUnknownType);

REGISTER_OP("vf.squeeze_store_unalign")
    .set_op_category("VFOp")
    .set_description("Squeeze-based unaligned store (reads AR register for byte count)")
    .add_argument("dst_ptr", "Destination UB pointer")
    .add_argument("src", "Source register")
    .add_argument("align_reg", "Alignment register")
    .f_deduce_type(DeduceVFUnknownType);

REGISTER_OP("vf.squeeze_store_unalign_post")
    .set_op_category("VFOp")
    .set_description("Complete squeeze-based unaligned store (reads AR register for remaining bytes)")
    .add_argument("dst_ptr", "Destination UB pointer")
    .add_argument("align_reg", "Alignment register")
    .f_deduce_type(DeduceVFUnknownType);

REGISTER_OP("vf.unalign_reg_for_store")
    .set_op_category("VFOp")
    .set_description("Declare unaligned register for store")
    .no_argument()
    .f_deduce_type(DeduceVFUnalignRegType);

REGISTER_OP("vf.clear_spr")
    .set_op_category("VFOp")
    .set_description("Clear special purpose register AR")
    .no_argument()
    .f_deduce_type(DeduceVFUnknownType);

REGISTER_OP("vf.load_unalign_init")
    .set_op_category("VFOp")
    .set_description("Declare unaligned register for load")
    .no_argument()
    .f_deduce_type(DeduceVFUnalignRegType);

REGISTER_OP("vf.load_unalign_pre")
    .set_op_category("VFOp")
    .set_description("Setup unaligned load (vldas)")
    .add_argument("ureg", "UnalignRegForLoad register")
    .add_argument("src_ptr", "Source UB pointer")
    .f_deduce_type(DeduceVFUnknownType);

REGISTER_OP("vf.load_unalign")
    .set_op_category("VFOp")
    .set_description("Load unaligned data from UB to register (vldus; passing stride selects POST_UPDATE mode)")
    .add_argument("dst", "Destination register")
    .add_argument("ureg", "UnalignRegForLoad register")
    .add_argument("src_ptr", "Source UB pointer")
    .add_argument("stride", "Post-update stride in bytes (passing it selects POST_UPDATE mode)")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.scatter")
    .set_op_category("VFOp")
    .set_description("Scatter store by indices (vscatter)")
    .add_argument("base_ptr", "Base UB pointer")
    .add_argument("src", "Source register")
    .add_argument("index", "Index register")
    .add_argument("mask", "Mask register")
    .f_deduce_type(DeduceVFUnknownType);

REGISTER_OP("vf.unsqueeze")
    .set_op_category("VFOp")
    .set_description("Unsqueeze mask bits to register (vusqz)")
    .add_argument("dst", "Destination register")
    .add_argument("mask", "Mask register to unsqueeze")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.truncate")
    .set_op_category("VFOp")
    .set_description(
        "Convert float to integer register (vtrc; rounding via round_mode kwarg, default CAST_RINT/ROUND_R)")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .set_attr<int>("round_mode")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.mask_gen_with_reg_tensor")
    .set_op_category("VFOp")
    .set_description("Generate MaskReg from RegTensor bit at offset (movvp)")
    .add_argument("src", "Source register (uint16/uint32)")
    .set_attr<int>("offset")
    .f_deduce_type(DeduceVFMaskType);

REGISTER_OP("vf.get_mask_spr")
    .set_op_category("VFOp")
    .set_description("Get mask from special purpose register (movp_b32/b16 via kwarg width)")
    .no_argument()
    .set_attr<int>("width")
    .f_deduce_type(DeduceVFMaskType);

REGISTER_OP("vf.log2")
    .set_op_category("VFOp")
    .set_description("Base-2 logarithm: dst = log2(src) = ln(src) * 1/ln(2)")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .set_attr<bool>("precision")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.log10")
    .set_op_category("VFOp")
    .set_description("Base-10 logarithm: dst = log10(src) = ln(src) * 1/ln(10)")
    .add_argument("dst", "Destination register")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .set_attr<bool>("precision")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.muls_cast")
    .set_op_category("VFOp")
    .set_description("Multiply by scalar then cast to half: dst(fp16) = cast(src(fp32) * scalar)")
    .add_argument("dst", "Destination register (fp16)")
    .add_argument("src", "Source register (fp32)")
    .add_argument("scalar", "Scalar multiplier")
    .add_argument("mask", "Mask register")
    .set_attr<DataType>("dtype")
    .set_attr<int>("layout")
    .f_deduce_type(DeduceVFTypeChangingRegType);

// mask_and/or/xor/not/mov/sel/pack/unpack/interleave/deinterleave have been
// merged into their unified counterparts (and_, or_, ..., move, select,
// pack, unpack, interleave, de_interleave). The Python parser redirects
// vf.mask_* calls to the unified IR op via _VF_OP_NAME_MAP.

// vf.mask_load/mask_store/mask_store_unalign IR registrations removed —
// the parser redirects these to vf.load_align/vf.store_align/vf.store_unalign
// which dispatch on the destination register's IR type (RegTensor vs MaskReg,
// matching AscendC's function-overloading model).

REGISTER_OP("vf.load")
    .set_op_category("VFOp")
    .set_description("Unified unaligned load (vldas+vldus, matches AscendC Load)")
    .add_argument("dst", "Destination register")
    .add_argument("src_ptr", "Source UB pointer")
    .add_argument("stride", "Post-update stride (optional, triggers POST_UPDATE mode)")
    .f_deduce_type(DeduceVFRegType);

REGISTER_OP("vf.store")
    .set_op_category("VFOp")
    .set_description("Unified unaligned store (vstus+vstas, matches AscendC Store)")
    .add_argument("dst_ptr", "Destination UB pointer")
    .add_argument("src", "Source register")
    .add_argument("count", "Element count (optional, defaults to 256/elem_bytes)")
    .f_deduce_type(DeduceVFUnknownType);

REGISTER_OP("vf.create_addr_reg")
    .set_op_category("VFOp")
    .set_description("Create an address offset register for aligned load/store")
    .add_argument("stride0", "Loop axis 0 stride (element count)")
    .add_argument("stride1", "Loop axis 1 stride (optional)")
    .add_argument("stride2", "Loop axis 2 stride (optional)")
    .add_argument("stride3", "Loop axis 3 stride (optional)")
    .set_attr<DataType>("dtype")
    .f_deduce_type(DeduceVFAddrRegType);

REGISTER_OP("vf.move")
    .set_op_category("VFOp")
    .set_description("Move/copy register elements (RegTensor or MaskReg). "
                     "RegTensor: vmov with MODE_MERGING (masked) or vmov (unmasked). "
                     "MaskReg: pmov (masked) or pmov (unmasked). "
                     "Unified op: dst register kind is inferred from source operands.")
    .add_argument("dst", "Destination register (RegTensor or MaskReg)")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register (optional)")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFRegLikeType);

REGISTER_OP("vf.move_acc")
    .set_op_category("VFOp")
    .set_description("Masked move with an explicit accumulate source: the old dst is an "
                     "explicit argument copy-fetched into the fresh SSA dst, then "
                     "vmov/pmov with MODE_MERGING (masked-out lanes keep the old value)")
    .add_argument("dst", "Destination/accumulator register (read+write) (RegTensor or MaskReg)")
    .add_argument("src", "Source register")
    .add_argument("mask", "Mask register")
    .set_attr<int>("mode")
    .f_deduce_type(DeduceVFRegLikeType);

// Kernel-side SIMD VF launch: carries the callee name and its runtime arguments;
// the called function is a separate SIMD_VF/SIMD_CALLEE IR Function.
TypePtr DeduceSimdCallType([[maybe_unused]] const std::vector<ExprPtr>& args,
                           [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs)
{
    return GetNoneType();
}

REGISTER_OP("simd.call")
    .set_op_category("VFOp")
    .set_description("Launch a SIMD vector function from an AIV kernel")
    .add_argument("args", "Runtime arguments passed to the callee (variadic)")
    .set_attr<std::string>("callee")
    .f_deduce_type(DeduceSimdCallType);

} // namespace ir
} // namespace pypto
