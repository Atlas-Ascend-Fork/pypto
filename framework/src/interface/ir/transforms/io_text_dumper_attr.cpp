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
#include <regex>
#include <string>
#include <type_traits>
#include <vector>

#include "core/any_cast.h"
#include "core/dtype.h"
#include "ir/memory_space.h"

#include "interface/operation/attribute.h"
#include "interface/operation/operation.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tile_shape.h"

namespace pypto {
namespace ir {

// ---------------------------------------------------------------------------
// Attr value printing
// ---------------------------------------------------------------------------
namespace {
std::vector<int64_t> EncodeTile(const CubeTile& tile)
{
    return {tile.m[0], tile.m[1], tile.k[0], tile.k[1],
            tile.k[2], tile.n[0], tile.n[1], static_cast<int64_t>(tile.enableSplitK)};
}

std::vector<int64_t> EncodeTile(const ConvTile& tile)
{
    const auto& l1 = tile.tileL1Info;
    const auto& l0 = tile.tileL0Info;
    return {l1.tileHin,
            l1.tileHout,
            l1.tileWin,
            l1.tileWout,
            l1.tileCinFmap,
            l1.tileCinWeight,
            l1.tileN,
            l1.tileBatch,
            l0.tileH,
            l0.tileW,
            l0.tileK,
            l0.tileN,
            static_cast<int64_t>(tile.setL0Tile)};
}

std::vector<int64_t> EncodeTile(const ConvBpTile& tile)
{
    const auto& l1 = tile.tileL1Info;
    const auto& l0 = tile.tileL0Info;
    return {l1.tileML1, l1.tileKL1, l1.tileNL1, l0.tileML0, l0.tileKL0, l0.tileNL0};
}

std::vector<int64_t> EncodeTile(const DistTile& tile)
{
    return {tile.row[0], tile.row[1],  tile.row[2],  tile.col[0],  tile.col[1],
            tile.col[2], tile.rank[0], tile.rank[1], tile.rank[2], tile.rankId};
}

std::string EncodeString(const std::string& value, bool quoted = false)
{
    static const std::regex ident("[A-Za-z_][A-Za-z0-9_]*");
    const auto kind = IRTextGetLexerTokenDict().Find(value, IRTextLexerTokenKind::TokIdent);
    if (!quoted && std::regex_match(value, ident) &&
        (kind == IRTextLexerTokenKind::TokIdent ||
         (kind >= IRTextLexerTokenKind::KwTypeBegin && kind <= IRTextLexerTokenKind::KwTypeEnd))) {
        return value;
    }
    std::ostringstream stream;
    stream << '"';
    for (char c : value) {
        switch (c) {
            case '"':
                stream << "\\\"";
                break;
            case '\\':
                stream << "\\\\";
                break;
            case '\n':
                stream << "\\n";
                break;
            case '\r':
                stream << "\\r";
                break;
            case '\t':
                stream << "\\t";
                break;
            case '\0':
                stream << "\\0";
                break;
            default:
                stream << c;
                break;
        }
    }
    stream << '"';
    return stream.str();
}

template <typename T>
void PrintVector(std::ostringstream& stream, const std::vector<T>& values)
{
    stream << IR_PUN_LBRACKET;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            stream << IR_PUN_COMMA << " ";
        }
        if constexpr (std::is_same_v<T, std::pair<std::string, std::string>>) {
            stream << IR_PUN_LBRACKET;
            stream << EncodeString(values[i].first);
            stream << IR_PUN_COMMA << " ";
            stream << EncodeString(values[i].second);
            stream << IR_PUN_RBRACKET;
        } else if constexpr (std::is_same_v<T, std::string>) {
            stream << EncodeString(values[i]);
        } else {
            stream << values[i];
        }
    }
    stream << IR_PUN_RBRACKET;
}

template <typename Key>
void PrintMap(std::ostream& stream, const std::map<Key, int64_t>& values)
{
    stream << IR_PUN_LBRACKET;
    bool first = true;
    for (const auto& [key, value] : values) {
        if (!first) {
            stream << IR_PUN_COMMA << " ";
        }
        stream << IR_PUN_LBRACKET;
        if constexpr (std::is_same_v<Key, std::string>) {
            stream << EncodeString(key, true);
        } else {
            stream << key;
        }
        stream << IR_PUN_COMMA << " " << value << IR_PUN_RBRACKET;
        first = false;
    }
    stream << IR_PUN_RBRACKET;
}

} // namespace

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
        stream_ << EncodeString(AnyCast<std::string>(value));
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
    } else if (value.type() == typeid(std::vector<double>)) {
        const auto& vals = AnyCast<std::vector<double>>(value, key);
        PrintVector(stream_, vals);
    } else if (value.type() == typeid(std::vector<std::string>)) {
        const auto& vals = AnyCast<std::vector<std::string>>(value, key);
        PrintVector(stream_, vals);
    } else if (value.type() == typeid(std::vector<std::pair<std::string, std::string>>)) {
        const auto& vals = AnyCast<std::vector<std::pair<std::string, std::string>>>(value, key);
        PrintVector(stream_, vals);
    } else if (value.type() == typeid(std::map<int64_t, int64_t>)) {
        const auto& vals = AnyCast<std::map<int64_t, int64_t>>(value, key);
        PrintMap(stream_, vals);
    } else if (value.type() == typeid(std::map<std::string, int64_t>)) {
        const auto& vals = AnyCast<std::map<std::string, int64_t>>(value, key);
        PrintMap(stream_, vals);
    } else if (value.type() == typeid(CubeTile)) {
        PrintDataArray(EncodeTile(AnyCast<CubeTile>(value, key)));
    } else if (value.type() == typeid(ConvTile)) {
        PrintDataArray(EncodeTile(AnyCast<ConvTile>(value, key)));
    } else if (value.type() == typeid(ConvBpTile)) {
        PrintDataArray(EncodeTile(AnyCast<ConvBpTile>(value, key)));
    } else if (value.type() == typeid(DistTile)) {
        PrintDataArray(EncodeTile(AnyCast<DistTile>(value, key)));
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
