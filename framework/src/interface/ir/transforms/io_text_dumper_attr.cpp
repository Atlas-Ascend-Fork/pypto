/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 * \file io_text_dumper_attr.cpp
 * \brief IR text dumper — attribute printing.
 */

#include "ir/transforms/io_text.h"

#include <any>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/any_cast.h"
#include "core/dtype.h"
#include "ir/memory_space.h"

#include "interface/operation/attribute.h"
#include "interface/operation/operation.h"
#include "interface/tensor/logical_tensor.h"

namespace pypto {
namespace ir {

// ---------------------------------------------------------------------------
// Attr value printing
// ---------------------------------------------------------------------------
void IRTextDumper::PrintAttrValue(const std::string& key, const std::any& value)
{
    if (value.type() == typeid(int)) {
        stream_ << AnyCast<int>(value, key);
    } else if (value.type() == typeid(uint64_t)) {
        stream_ << (AnyCast<uint64_t>(value));
    } else if (value.type() == typeid(int64_t)) {
        stream_ << (AnyCast<int64_t>(value));
    } else if (value.type() == typeid(double)) {
        stream_ << AnyCast<double>(value, key);
    } else if (value.type() == typeid(float)) {
        stream_ << static_cast<double>(AnyCast<float>(value, key));
    } else if (value.type() == typeid(bool)) {
        stream_ << (AnyCast<bool>(value, key) ? IR_KW_TRUE : IR_KW_FALSE);
    } else if (value.type() == typeid(std::string)) {
        stream_ << (AnyCast<std::string>(value));
    } else if (value.type() == typeid(DataType)) {
        stream_ << (AnyCast<DataType>(value, key).ToCTypeString());
    } else if (value.type() == typeid(MemorySpace)) {
        stream_ << MemorySpaceToString(AnyCast<MemorySpace>(value, key));
    } else if (value.type() == typeid(std::vector<int>)) {
        const auto& vals = AnyCast<std::vector<int>>(value, key);
        PrintDataArray(std::vector<int64_t>(vals.begin(), vals.end()));
    } else if (value.type() == typeid(std::vector<int64_t>)) {
        const auto& vals = AnyCast<std::vector<int64_t>>(value, key);
        PrintDataArray(vals);
    } else if (value.type() == typeid(ExprPtr)) {
        VisitExpr(AnyCast<ExprPtr>(value, key));
    } else if (value.type() == typeid(std::vector<std::any>)) {
        const auto& vals = AnyCast<std::vector<std::any>>(value, key);
        stream_ << IR_PUN_LBRACKET;
        for (size_t i = 0; i < vals.size(); ++i) {
            if (i > 0) {
                stream_ << IR_PUN_COMMA << " ";
            }
            PrintAttrValue(key, vals[i]);
        }
        stream_ << IR_PUN_RBRACKET;
    } else if (value.type() == typeid(std::vector<npu::tile_fwk::SymbolicScalar>)) {
        const auto vals = AnyCast<std::vector<npu::tile_fwk::SymbolicScalar>>(value, key);
        stream_ << IR_PUN_LBRACKET;
        for (size_t i = 0; i < vals.size(); ++i) {
            if (i > 0) {
                stream_ << IR_PUN_COMMA << " ";
            }
            PrintScalarExpr(vals[i].Raw());
        }
        stream_ << IR_PUN_RBRACKET;
    } else {
        stream_ << "Unsupported";
    }
}

void IRTextDumper::PrintAttr(const std::string& key, const std::any& value)
{
    stream_ << " " << IR_PUN_ATTRNAME << key << IR_PUN_LPAREN;
    PrintAttrValue(key, value);
    stream_ << IR_PUN_RPAREN;
}

void IRTextDumper::PrintDataArray(const std::vector<int64_t>& vals)
{
    stream_ << IR_PUN_LBRACKET;
    for (size_t i = 0; i < vals.size(); ++i) {
        if (i > 0) {
            stream_ << IR_PUN_COMMA << " ";
        }
        stream_ << vals[i];
    }
    stream_ << IR_PUN_RBRACKET;
}

// ---------------------------------------------------------------------------
// Attr dict collection
//
// Per-node-type rules for which attrs are dumped.  Attr is the leaf layer of
// the traversal: the dumper hands each carrying node to its Get*AttrDict and
// prints the returned dict via PrintAttrDict.  For tensor-op offsets the
// static part is emitted under #toOffset/#fromOffset and the dynamic part
// under #toDynOffset/#fromDynOffset, so both survive a dump.
// ---------------------------------------------------------------------------
namespace {

// A dynamic offset list is dumped only when it really carries dynamic values:
// lists holding solely immediates duplicate the static offset.
bool HasDynamicValue(const std::vector<npu::tile_fwk::SymbolicScalar>& dynOffset)
{
    for (const auto& value : dynOffset) {
        if (!value.IsImmediate()) {
            return true;
        }
    }
    return false;
}

void CollectOffsetAttrs(const char* offsetKey, const std::vector<int64_t>& offset, const char* dynOffsetKey,
                        const std::vector<npu::tile_fwk::SymbolicScalar>& dynOffset,
                        std::map<std::string, std::any>& attrs)
{
    if (!offset.empty()) {
        attrs[offsetKey] = offset;
    }
    if (HasDynamicValue(dynOffset)) {
        attrs[dynOffsetKey] = dynOffset;
    }
}

void CollectViewOpAttrs(npu::tile_fwk::ViewOpAttribute& attr, std::map<std::string, std::any>& attrs)
{
    CollectOffsetAttrs("fromOffset", attr.GetFromOffset(), "fromDynOffset", attr.GetFromDynOffset(), attrs);
    if (!attr.GetToDynValidShape().empty()) {
        attrs["toValidShape"] = attr.GetToDynValidShape();
    }
}

void CollectAssembleOpAttrs(npu::tile_fwk::AssembleOpAttribute& attr, std::map<std::string, std::any>& attrs)
{
    CollectOffsetAttrs("toOffset", attr.GetToOffset(), "toDynOffset", attr.GetToDynOffset(), attrs);
    if (!attr.GetFromDynValidShape().empty()) {
        attrs["fromValidShape"] = attr.GetFromDynValidShape();
    }
}

} // namespace

std::map<std::string, std::any> GetOperationAttrDict(const TensorOpStmtPtr& top)
{
    std::map<std::string, std::any> attrs;
    auto op = std::dynamic_pointer_cast<npu::tile_fwk::Operation>(std::const_pointer_cast<TensorOpStmt>(top));
    if (op == nullptr) {
        // Generic tensor op stmt: attrs are stored verbatim on the stmt.
        for (const auto& [key, value] : top->attrs_) {
            attrs[key] = value;
        }
        return attrs;
    }
    auto& opAttr = op->GetOpAttribute();
    if (auto viewAttr = std::dynamic_pointer_cast<npu::tile_fwk::ViewOpAttribute>(opAttr)) {
        CollectViewOpAttrs(*viewAttr, attrs);
    } else if (auto assembleAttr = std::dynamic_pointer_cast<npu::tile_fwk::AssembleOpAttribute>(opAttr)) {
        CollectAssembleOpAttrs(*assembleAttr, attrs);
    }
    return attrs;
}

std::map<std::string, std::any> IRTextDumper::GetTensorAttrDict(const VarPtr& var)
{
    std::map<std::string, std::any> attrs;
    auto t = std::dynamic_pointer_cast<const npu::tile_fwk::LogicalTensor>(var);
    if (t == nullptr) {
        return attrs;
    }
    attrs["dtype"] = std::string(GetTileFwkDataTypeDict().Find(t->Datatype()));
    attrs["shape"] = t->shape;
    attrs["offset"] = t->offset;
    if (!t->dynValidShape_.empty()) {
        attrs["dynvalidshape"] = t->dynValidShape_;
    }
    if (!t->dynOffset_.empty()) {
        attrs["dynvalidoffset"] = t->dynOffset_;
    }
    return attrs;
}

std::map<std::string, std::any> IRTextDumper::GetFunctionAttrDict(const FunctionPtr& func)
{
    return {
        {IR_KW_TYPE, FunctionTypeToString(func->funcType_)},
        {IR_KW_ENTRY, func->entry_},
    };
}

} // namespace ir
} // namespace pypto
