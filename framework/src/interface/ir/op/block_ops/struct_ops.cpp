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
 * @file block_ops/struct_ops.cpp
 * \brief Struct array operations for clean C++ codegen.
 *
 *  - struct.create: Expression op that returns a struct-classified TupleType. Lowered by
 *    the CCE backend to ``Name var = {.f0=v0, ...};``.
 *  - struct.create_array: Expression op that returns TupleType(N × struct TupleType).
 *    Lowered to ``Name var[N] = {};`` plus per-slot field-init lines.
 */

#include <any>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "core/logging.h"
#include "ir/expr.h"
#include "ir/identifier.h"
#include "ir/op_registry.h"
#include "ir/type.h"
#include "pypto_pro/error.h"
#include "tilefwk/error.h"

namespace pypto {
namespace ir {
using npu::tile_fwk::ExternalError;

REGISTER_OP("struct.create")
    .set_op_category("StructOp")
    .set_description("Materialize a TupleType into a C++ struct. args are field values; kwargs `name` "
                     "is the C++ struct type name and `fields` is the list of field names. Result type is "
                     "TupleType(field_types); the parser registers its struct metadata. Must be let-bound to a Var "
                     "by the producer.")
    .add_argument("...", "Field value expressions in declaration order")
    .set_attr<std::string>("name")
    .set_attr<std::vector<std::string>>("fields")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) -> TypePtr {
        std::string structName;
        std::vector<std::string> fields;
        bool hasName = false;
        bool hasFields = false;
        for (const auto& [key, value] : kwargs) {
            if (key == "name") {
                structName = std::any_cast<std::string>(value);
                hasName = true;
            } else if (key == "fields") {
                fields = std::any_cast<std::vector<std::string>>(value);
                hasFields = true;
            }
        }
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, hasName)
            << "struct.create requires kwarg 'name' (C++ struct type name)";
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, hasFields)
            << "struct.create requires kwarg 'fields' (list of field names)";
        PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsValidIdentifier(structName))
            << "struct.create 'name' must be a valid C identifier, got '" << structName << "'";
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, fields.size() == args.size())
            << "struct.create: fields count (" << fields.size() << ") must match args count (" << args.size() << ")";
        std::set<std::string> seen;
        for (const auto& f : fields) {
            PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, IsValidIdentifier(f))
                << "struct.create field name '" << f << "' is not a valid identifier";
            PRO_IR_CHECK(ExternalError::INVALID_OPERATION, seen.insert(f).second)
                << "struct.create field name '" << f << "' is duplicated";
        }
        std::vector<TypePtr> fieldTypes;
        fieldTypes.reserve(args.size());
        for (size_t i = 0; i < args.size(); ++i) {
            PRO_IR_CHECK(npu::tile_fwk::InternalError::COMMON_INNER_ERROR, args[i])
                << "struct.create arg #" << i << " is null";
            fieldTypes.push_back(args[i]->GetType());
        }
        // Positional TupleType: field names live in IRDebugInfo (registered by the
        // parser from the op's 'fields' kwarg), not on the type.
        (void)fields;
        return std::make_shared<TupleType>(std::move(fieldTypes));
    });

REGISTER_OP("struct.set")
    .set_op_category("StructOp")
    .set_description("Statement side-effect call: write one struct field. args are (base, value) for a scalar "
                     "field or (base, index, value) for an array field element; kwarg `field` is the C++ field "
                     "name (same naming convention as struct.create's `fields`). Lowered by the CCE backend to "
                     "`base.field = value;` or `base.field[index] = value;`. Must be used inside an EvalStmt, "
                     "not as an RHS value.")
    .add_argument("base", "Struct instance (struct-classified TupleType)")
    .add_argument("value", "New field value expression (or index for array field writes)")
    .set_attr<std::string>("field")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) -> TypePtr {
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, args.size() == 2 || args.size() == 3)
            << "struct.set requires 2 args (base, value) or 3 args (base, index, value), got " << args.size();
        PRO_IR_CHECK(npu::tile_fwk::InternalError::COMMON_INNER_ERROR, args[0]) << "struct.set: base arg is null";
        if (args.size() == 3) {
            PRO_IR_CHECK(npu::tile_fwk::InternalError::COMMON_INNER_ERROR, args[1]) << "struct.set: index arg is null";
        }
        PRO_IR_CHECK(npu::tile_fwk::InternalError::COMMON_INNER_ERROR, args.back()) << "struct.set: value arg is null";
        bool hasField = false;
        for (const auto& [key, value] : kwargs) {
            if (key == "field") {
                PRO_IR_CHECK(ExternalError::INVALID_TYPE, IsValidIdentifier(std::any_cast<std::string>(value)))
                    << "struct.set 'field' must be a valid identifier";
                hasField = true;
            }
        }
        PRO_IR_CHECK(ExternalError::INVALID_ARGUMENT, hasField)
            << "struct.set requires kwarg 'field' (the C++ field name to write)";
        PRO_IR_CHECK(ExternalError::INVALID_TYPE, args[0]->GetType()->GetKind() == ObjectKind::TupleType)
            << "struct.set: base must have TupleType";
        // Statement side effect; the discarded result type mirrors the struct being written.
        return args[0]->GetType();
    });

} // namespace ir
} // namespace pypto
