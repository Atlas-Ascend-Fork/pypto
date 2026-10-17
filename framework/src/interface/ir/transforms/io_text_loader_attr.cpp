/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 * \file io_text_loader_attr.cpp
 * \brief IR text loader — attribute (attr_list / attr_value) parsing.
 */

#include "io_text_loader.h"

#include <any>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/any_cast.h"
#include "interface/operation/attribute.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation.h"
#include "interface/tensor/logical_tensor.h"

namespace pypto {
namespace ir {

// ---- Attr list:  #name(attr_value) { #name(attr_value) } ----
IRTextParserStringPtr IRTextParser::ParseAttrName()
{
    if (!CheckAttrName()) {
        return Error("expected attribute name, got '" + Peek().text + "'");
    }
    auto result = std::make_shared<IRTextParserData<std::string>>();
    result->data = tokens_[cur_++].text;
    return result;
}

IRTextParserAttrDictPtr IRTextParser::ParseAttrDict()
{
    auto result = std::make_shared<IRTextParserData<std::map<std::string, std::any>>>();
    while (CheckAttrName()) {
        MUST_VALID(key, ParseAttrName());
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunLParen));
        MUST_VALID(val, ParseAttrValueRaw());
        MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunRParen));
        result->data[key->data] = val->data;
    }
    return result;
}

namespace {

bool IsIntListAttr(const std::any& value)
{
    if (value.type() != typeid(std::vector<std::any>)) {
        return false;
    }
    const auto& list = std::any_cast<const std::vector<std::any>&>(value);
    for (const auto& elem : list) {
        if (elem.type() != typeid(int64_t)) {
            return false;
        }
    }
    return true;
}

bool IsSymbolicScalarListAttr(const std::any& value)
{
    if (value.type() != typeid(std::vector<std::any>)) {
        return false;
    }
    const auto& list = std::any_cast<const std::vector<std::any>&>(value);
    for (const auto& elem : list) {
        const bool supported = elem.type() == typeid(int64_t) || elem.type() == typeid(std::string) ||
                               elem.type() == typeid(ExprPtr);
        if (!supported) {
            return false;
        }
    }
    return true;
}

} // namespace

bool IRTextParser::FillSymbolicScalarListAttr(const std::string& key, const std::any& value,
                                              std::vector<npu::tile_fwk::SymbolicScalar>& out)
{
    if (!IsSymbolicScalarListAttr(value)) {
        Error("invalid attr value for #" + key);
        return false;
    }
    SetValue(out, value);
    return true;
}

bool IRTextParser::FillOffsetAttr(const std::string& key, const std::any& value, std::vector<int64_t>& offset,
                                  std::vector<npu::tile_fwk::SymbolicScalar>& dynOffset)
{
    if (IsIntListAttr(value)) {
        SetValue(offset, value);
        return true;
    }
    // Legacy text dumped the dynamic offset under the static offset key.
    return FillSymbolicScalarListAttr(key, value, dynOffset);
}

bool IRTextParser::SetViewOpAttrDict(npu::tile_fwk::Operation& op, const std::map<std::string, std::any>& attrDict)
{
    std::vector<int64_t> fromOffset;
    std::vector<npu::tile_fwk::SymbolicScalar> fromDynOffset;
    std::vector<npu::tile_fwk::SymbolicScalar> toDynValidShape;
    bool hasAttr = false;
    for (const auto& [key, value] : attrDict) {
        if (key == "fromOffset") {
            if (!FillOffsetAttr(key, value, fromOffset, fromDynOffset)) {
                return false;
            }
            hasAttr = true;
        } else if (key == "fromDynOffset") {
            if (!FillSymbolicScalarListAttr(key, value, fromDynOffset)) {
                return false;
            }
            hasAttr = true;
        } else if (key == "toValidShape") {
            if (!FillSymbolicScalarListAttr(key, value, toDynValidShape)) {
                return false;
            }
            hasAttr = true;
        }
    }
    if (hasAttr) {
        op.SetOpAttribute(std::make_shared<npu::tile_fwk::ViewOpAttribute>(fromOffset, fromDynOffset, toDynValidShape));
    }
    return true;
}

bool IRTextParser::SetAssembleOpAttrDict(npu::tile_fwk::Operation& op, const std::map<std::string, std::any>& attrDict)
{
    std::vector<int64_t> toOffset;
    std::vector<npu::tile_fwk::SymbolicScalar> toDynOffset;
    std::vector<npu::tile_fwk::SymbolicScalar> fromDynValidShape;
    bool hasAttr = false;
    for (const auto& [key, value] : attrDict) {
        if (key == "toOffset") {
            if (!FillOffsetAttr(key, value, toOffset, toDynOffset)) {
                return false;
            }
            hasAttr = true;
        } else if (key == "toDynOffset") {
            if (!FillSymbolicScalarListAttr(key, value, toDynOffset)) {
                return false;
            }
            hasAttr = true;
        } else if (key == "fromValidShape") {
            if (!FillSymbolicScalarListAttr(key, value, fromDynValidShape)) {
                return false;
            }
            hasAttr = true;
        }
    }
    if (hasAttr) {
        auto assembleAttr = std::make_shared<npu::tile_fwk::AssembleOpAttribute>(toOffset, toDynOffset);
        if (!fromDynValidShape.empty()) {
            assembleAttr->SetFromDynValidShape(fromDynValidShape);
        }
        op.SetOpAttribute(assembleAttr);
    }
    return true;
}

// ---- v0_logical_tensor var-def attrs ----
bool IRTextParser::SetTensorAttrDict(const VarPtr& var, const std::map<std::string, std::any>& attrDict)
{
    auto t = std::dynamic_pointer_cast<npu::tile_fwk::LogicalTensor>(std::const_pointer_cast<Var>(var));
    if (t == nullptr) {
        Error("SetTensorAttrDict: var is not a logical tensor");
        return false;
    }
    TensorVarAttrValues values;
    for (const auto& [key, value] : attrDict) {
        if (key == "dtype") {
            SetValue(values.dtype, value);
        } else if (key == "shape") {
            SetValue(values.shape, value);
        } else if (key == "offset") {
            SetValue(values.offset, value);
        } else if (key == "dynvalidshape") {
            SetValue(values.dynValidShape, value);
        } else if (key == "dynvalidoffset") {
            SetValue(values.dynOffset, value);
        } else {
            Error("unknown logical tensor attr: #" + key);
            return false;
        }
    }
    t->tensor->datatype = values.dtype;
    t->tensor->rawshape = values.shape;
    t->shape = values.shape;
    t->oriShape = values.shape;
    t->offset = values.offset.empty() ? npu::tile_fwk::Offset(values.shape.size(), 0) : values.offset;
    t->dynValidShape_ = values.dynValidShape;
    t->dynOffset_ = values.dynOffset;
    return true;
}

bool IRTextParser::SetOperationAttrDict(const TensorOpStmtPtr& op, const std::map<std::string, std::any>& attrDict)
{
    auto stmt = std::const_pointer_cast<TensorOpStmt>(op);
    auto operation = std::dynamic_pointer_cast<npu::tile_fwk::Operation>(stmt);
    if (operation == nullptr) {
        // Generic tensor op stmt: attrs are stored verbatim on the stmt.
        for (const auto& [key, value] : attrDict) {
            stmt->attrs_.emplace_back(key, value);
        }
        return true;
    }
    if (npu::tile_fwk::IsViewLike(operation->GetOpcode())) {
        return SetViewOpAttrDict(*operation, attrDict);
    }
    if (npu::tile_fwk::IsAssembleLike(operation->GetOpcode())) {
        return SetAssembleOpAttrDict(*operation, attrDict);
    }
    return true;
}

bool IRTextParser::SetFunctionAttrDict(const FunctionPtr& func, const std::map<std::string, std::any>& attrDict)
{
    auto f = std::const_pointer_cast<Function>(func);
    for (const auto& [key, value] : attrDict) {
        if (key == IR_KW_TYPE) {
            try {
                f->funcType_ = StringToFunctionType(AnyCast<std::string>(value));
            } catch (const std::invalid_argument& e) {
                Error(std::string("unknown function type: ") + e.what());
                return false;
            }
        } else if (key == IR_KW_ENTRY) {
            f->entry_ = AnyCast<bool>(value);
        }
    }
    return true;
}

IRTextParserAttrValuePtr IRTextParser::ParseAttrValueInt()
{
    const IRTextLexerToken& tok = Peek();
    ++cur_;
    std::any data = std::any(static_cast<int64_t>(tok.intVal));
    return std::make_shared<IRTextParserAttrValuePtr::element_type>(data);
}

IRTextParserAttrValuePtr IRTextParser::ParseAttrValueFloat()
{
    const IRTextLexerToken& tok = Peek();
    ++cur_;
    std::any data = std::any(tok.floatVal);
    return std::make_shared<IRTextParserAttrValuePtr::element_type>(data);
}

IRTextParserAttrValuePtr IRTextParser::ParseAttrValueTrue()
{
    MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwTrue));
    std::any data = true;
    return std::make_shared<IRTextParserAttrValuePtr::element_type>(data);
}

IRTextParserAttrValuePtr IRTextParser::ParseAttrValueFalse()
{
    MUST_MATCH(ParseToken(IRTextLexerTokenKind::KwFalse));
    std::any data = false;
    return std::make_shared<IRTextParserAttrValuePtr::element_type>(data);
}

IRTextParserAttrValuePtr IRTextParser::ParseAttrValueIdent()
{
    const IRTextLexerToken& tok = Peek();
    std::string s = tok.text;
    ++cur_;
    std::any data = std::any(s);
    return std::make_shared<IRTextParserAttrValuePtr::element_type>(data);
}

IRTextParserAttrValuePtr IRTextParser::ParseAttrValueList()
{
    // '[' [ attr_value { ',' attr_value } ] ']' -> std::vector<std::any>
    MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunLBracket));
    std::vector<std::any> vals;
    if (!CheckToken(IRTextLexerTokenKind::PunRBracket)) {
        do {
            MUST_VALID(val, ParseAttrValueRaw());
            vals.push_back(val->data);
        } while (CheckToken(IRTextLexerTokenKind::PunComma, true));
    }
    MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunRBracket));
    std::any data = std::any(vals);
    return std::make_shared<IRTextParserAttrValuePtr::element_type>(data);
}

IRTextParserAttrValuePtr IRTextParser::ParseAttrValueExpr()
{
    // expr fallback: keep the parsed expr as std::any(ExprPtr)
    MUST_VALID(expr, ParseExpr());
    std::any data = std::any(expr);
    return std::make_shared<IRTextParserAttrValuePtr::element_type>(data);
}

IRTextParserAttrValuePtr IRTextParser::ParseAttrValueRaw()
{
    static const std::unordered_map<IRTextLexerTokenKind, ParseAttrValueFn> dispatch = {
        {IRTextLexerTokenKind::TokInt, &IRTextParser::ParseAttrValueInt},
        {IRTextLexerTokenKind::TokFloat, &IRTextParser::ParseAttrValueFloat},
        {IRTextLexerTokenKind::TokIdent, &IRTextParser::ParseAttrValueIdent},
        // dtype keywords (int32_t / float / bfloat16_t ...) are treated as ident-shaped attr values
        {IRTextLexerTokenKind::KwTypeBool, &IRTextParser::ParseAttrValueIdent},
        {IRTextLexerTokenKind::KwTypeInt8, &IRTextParser::ParseAttrValueIdent},
        {IRTextLexerTokenKind::KwTypeInt16, &IRTextParser::ParseAttrValueIdent},
        {IRTextLexerTokenKind::KwTypeInt32, &IRTextParser::ParseAttrValueIdent},
        {IRTextLexerTokenKind::KwTypeInt64, &IRTextParser::ParseAttrValueIdent},
        {IRTextLexerTokenKind::KwTypeUint8, &IRTextParser::ParseAttrValueIdent},
        {IRTextLexerTokenKind::KwTypeUint16, &IRTextParser::ParseAttrValueIdent},
        {IRTextLexerTokenKind::KwTypeUint32, &IRTextParser::ParseAttrValueIdent},
        {IRTextLexerTokenKind::KwTypeUint64, &IRTextParser::ParseAttrValueIdent},
        {IRTextLexerTokenKind::KwTypeFp16, &IRTextParser::ParseAttrValueIdent},
        {IRTextLexerTokenKind::KwTypeFp32, &IRTextParser::ParseAttrValueIdent},
        {IRTextLexerTokenKind::KwTypeFp64, &IRTextParser::ParseAttrValueIdent},
        {IRTextLexerTokenKind::KwTypeBf16, &IRTextParser::ParseAttrValueIdent},
        {IRTextLexerTokenKind::KwTypeFp8e4m3fn, &IRTextParser::ParseAttrValueIdent},
        {IRTextLexerTokenKind::KwTypeFp8e5m2, &IRTextParser::ParseAttrValueIdent},
        {IRTextLexerTokenKind::KwTypeHf4, &IRTextParser::ParseAttrValueIdent},
        {IRTextLexerTokenKind::KwTypeHf8, &IRTextParser::ParseAttrValueIdent},
        {IRTextLexerTokenKind::KwTypeUnknown, &IRTextParser::ParseAttrValueIdent},
        {IRTextLexerTokenKind::KwTrue, &IRTextParser::ParseAttrValueTrue},
        {IRTextLexerTokenKind::KwFalse, &IRTextParser::ParseAttrValueFalse},
        {IRTextLexerTokenKind::PunLBracket, &IRTextParser::ParseAttrValueList},
        // expr-start tokens -> std::any(ExprPtr)
        {IRTextLexerTokenKind::TokVarName, &IRTextParser::ParseAttrValueExpr},
        {IRTextLexerTokenKind::PunLParen, &IRTextParser::ParseAttrValueExpr},
        {IRTextLexerTokenKind::KwTuple, &IRTextParser::ParseAttrValueExpr},
        {IRTextLexerTokenKind::KwGetItem, &IRTextParser::ParseAttrValueExpr},
        {IRTextLexerTokenKind::KwMemref, &IRTextParser::ParseAttrValueExpr},
        {IRTextLexerTokenKind::KwV0ScalarUOpPos, &IRTextParser::ParseAttrValueExpr},
        {IRTextLexerTokenKind::KwV0ScalarUOpNeg, &IRTextParser::ParseAttrValueExpr},
        {IRTextLexerTokenKind::KwV0ScalarUOpNot, &IRTextParser::ParseAttrValueExpr},
        {IRTextLexerTokenKind::KwV0ScalarBOpAdd, &IRTextParser::ParseAttrValueExpr},
        {IRTextLexerTokenKind::KwV0ScalarBOpSub, &IRTextParser::ParseAttrValueExpr},
        {IRTextLexerTokenKind::KwV0ScalarBOpMul, &IRTextParser::ParseAttrValueExpr},
        {IRTextLexerTokenKind::KwV0ScalarBOpDiv, &IRTextParser::ParseAttrValueExpr},
        {IRTextLexerTokenKind::KwV0ScalarBOpMod, &IRTextParser::ParseAttrValueExpr},
        {IRTextLexerTokenKind::KwV0ScalarBOpEq, &IRTextParser::ParseAttrValueExpr},
        {IRTextLexerTokenKind::KwV0ScalarBOpNe, &IRTextParser::ParseAttrValueExpr},
        {IRTextLexerTokenKind::KwV0ScalarBOpLt, &IRTextParser::ParseAttrValueExpr},
        {IRTextLexerTokenKind::KwV0ScalarBOpLe, &IRTextParser::ParseAttrValueExpr},
        {IRTextLexerTokenKind::KwV0ScalarBOpGt, &IRTextParser::ParseAttrValueExpr},
        {IRTextLexerTokenKind::KwV0ScalarBOpGe, &IRTextParser::ParseAttrValueExpr},
        {IRTextLexerTokenKind::KwV0ScalarMOpCall, &IRTextParser::ParseAttrValueExpr},
        {IRTextLexerTokenKind::KwV0ScalarMOpMin, &IRTextParser::ParseAttrValueExpr},
        {IRTextLexerTokenKind::KwV0ScalarMOpMax, &IRTextParser::ParseAttrValueExpr},
        {IRTextLexerTokenKind::KwV0ScalarMOpAnd, &IRTextParser::ParseAttrValueExpr},
        {IRTextLexerTokenKind::KwV0ScalarMOpOr, &IRTextParser::ParseAttrValueExpr},
    };
    auto it = dispatch.find(Peek().kind);
    if (it != dispatch.end()) {
        auto result = (this->*(it->second))();
        return result;
    }
    return Error("expected attr value");
}

} // namespace ir
} // namespace pypto
