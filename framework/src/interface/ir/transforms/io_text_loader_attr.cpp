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
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/any_cast.h"
#include "tilefwk/element.h"
#include "interface/configs/config_manager_ng.h"
#include "interface/operation/attribute.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tile_shape.h"

namespace pypto {
namespace ir {

namespace {

std::string DecodeString(const std::string& text)
{
    std::string value;
    // The lexer has already validated the quotes and supported escape sequences.
    for (size_t i = 1; i + 1 < text.size(); ++i) {
        char c = text[i];
        if (c == '\\') {
            c = text[++i];
            switch (c) {
                case 'n':
                    c = '\n';
                    break;
                case 'r':
                    c = '\r';
                    break;
                case 't':
                    c = '\t';
                    break;
                case '0':
                    c = '\0';
                    break;
                default:
                    break;
            }
        }
        value.push_back(c);
    }
    return value;
}

// Flat IR-text layouts (fixed order; changing these changes the text format):
// Cube: m[0..1], k[0..2], n[0..1], enableSplitK.
// Conv: tileHin, tileHout, tileWin, tileWout, tileCinFmap, tileCinWeight,
//       tileN, tileBatch (L1), tileH, tileW, tileK, tileN (L0), setL0Tile.
// ConvBp: tileML1, tileKL1, tileNL1, tileML0, tileKL0, tileNL0.
// Dist: row[0..2], col[0..2], rank[0..2], rankId.
// Use int64_t to preserve dimension widths; bool fields are encoded as 0 or 1.
// Conversion preserves stored values, including zero/default tiles; it does not call valid().
constexpr std::size_t kCubeTileVectorSize = 8;
constexpr std::size_t kConvTileVectorSize = 13;
constexpr std::size_t kConvBpTileVectorSize = 6;
constexpr std::size_t kDistTileVectorSize = 10;

bool RequireSize(const std::vector<int64_t>& values, size_t expected, const char* typeName, std::string& error)
{
    if (values.size() == expected) {
        return true;
    }
    error = std::string(typeName) + ": expected " + std::to_string(expected) + " int64 values, got " +
            std::to_string(values.size());
    return false;
}

bool RequireBool01(int64_t value, const char* typeName, const char* field, bool& out, std::string& error)
{
    if (value != 0 && value != 1) {
        error = std::string(typeName) + ": " + field + " must be 0 or 1, got " + std::to_string(value);
        return false;
    }
    out = value == 1;
    return true;
}

bool RequireDistInt(int64_t value, const char* field, int& out, std::string& error)
{
    const int64_t kMin = std::numeric_limits<int>::min();
    const int64_t kMax = std::numeric_limits<int>::max();
    if (value < kMin || value > kMax) {
        error = std::string("DistTile: ") + field + " does not fit in int";
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

bool DecodeTile(const std::vector<int64_t>& values, CubeTile& out, std::string& error)
{
    if (!RequireSize(values, kCubeTileVectorSize, "CubeTile", error)) {
        return false;
    }
    bool enableSplitK = false;
    if (!RequireBool01(values[7], "CubeTile", "enableSplitK", enableSplitK, error)) {
        return false;
    }
    CubeTile local;
    local.m = {values[0], values[1]};
    local.k = {values[2], values[3], values[4]};
    local.n = {values[5], values[6]};
    local.enableSplitK = enableSplitK;
    out = local;
    return true;
}

bool DecodeTile(const std::vector<int64_t>& values, ConvTile& out, std::string& error)
{
    if (!RequireSize(values, kConvTileVectorSize, "ConvTile", error)) {
        return false;
    }
    bool setL0Tile = false;
    if (!RequireBool01(values[12], "ConvTile", "setL0Tile", setL0Tile, error)) {
        return false;
    }
    ConvTile local;
    local.tileL1Info = {values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7]};
    local.tileL0Info = {values[8], values[9], values[10], values[11]};
    local.setL0Tile = setL0Tile;
    out = local;
    return true;
}

bool DecodeTile(const std::vector<int64_t>& values, ConvBpTile& out, std::string& error)
{
    if (!RequireSize(values, kConvBpTileVectorSize, "ConvBpTile", error)) {
        return false;
    }
    ConvBpTile local;
    local.tileL1Info = {values[0], values[1], values[2]};
    local.tileL0Info = {values[3], values[4], values[5]};
    out = local;
    return true;
}

bool DecodeTile(const std::vector<int64_t>& values, DistTile& out, std::string& error)
{
    if (!RequireSize(values, kDistTileVectorSize, "DistTile", error)) {
        return false;
    }
    const char* fields[] = {"row[0]", "row[1]",  "row[2]",  "col[0]",  "col[1]",
                            "col[2]", "rank[0]", "rank[1]", "rank[2]", "rankId"};
    int decoded[kDistTileVectorSize];
    for (size_t i = 0; i < kDistTileVectorSize; ++i) {
        if (!RequireDistInt(values[i], fields[i], decoded[i], error)) {
            return false;
        }
    }
    DistTile local;
    for (size_t i = 0; i < MAX_DIST_DIM_SIZE; ++i) {
        local.row[i] = decoded[i];
        local.col[i] = decoded[MAX_DIST_DIM_SIZE + i];
        local.rank[i] = decoded[2 * MAX_DIST_DIM_SIZE + i];
    }
    local.rankId = decoded[kDistTileVectorSize - 1];
    out = local;
    return true;
}

template <typename Tile>
bool DecodeTypedTile(const std::vector<int64_t>& values, std::any& out, std::string& error)
{
    Tile tile;
    if (!DecodeTile(values, tile, error)) {
        return false;
    }
    out = tile;
    return true;
}

using TileDecodeFn = bool (*)(const std::vector<int64_t>&, std::any&, std::string&);

const std::unordered_map<std::string, TileDecodeFn> kTileAttrDecoders = {
    {"cube_tile_shapes", &DecodeTypedTile<CubeTile>},
    {"conv_tile_shapes", &DecodeTypedTile<ConvTile>},
    {"convbp_tile_shapes", &DecodeTypedTile<ConvBpTile>},
    {"dist_tile_shapes", &DecodeTypedTile<DistTile>},
};

// Leave other attributes unchanged. Tile attributes must be flat integer lists.
bool DecodeTileAttr(const std::string& key, std::any& value, std::string& error)
{
    const auto it = kTileAttrDecoders.find(key);
    if (it == kTileAttrDecoders.end()) {
        return true;
    }
    const auto* list = std::any_cast<std::vector<std::any>>(&value);
    if (list == nullptr) {
        error = key + ": expected an integer list";
        return false;
    }
    std::vector<int64_t> ints;
    ints.reserve(list->size());
    for (size_t i = 0; i < list->size(); ++i) {
        const auto* element = std::any_cast<int64_t>(&(*list)[i]);
        if (element == nullptr) {
            error = key + ": expected int64 at index " + std::to_string(i);
            return false;
        }
        ints.push_back(*element);
    }
    if (!it->second(ints, value, error)) {
        error = key + ": " + error;
        return false;
    }
    return true;
}

} // namespace

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
        std::any decoded = val->data;
        std::string decodeError;
        if (!DecodeTileAttr(key->data, decoded, decodeError)) {
            return Error(decodeError);
        }
        result->data[key->data] = std::move(decoded);
    }
    return result;
}

namespace {

// Nonempty lists of [key, integer] pairs become maps when all keys have the same type.
// An empty list stays a vector; [] is also the spelling of an empty int list.
template <typename Key>
bool AsKeyMap(const std::vector<std::any>& values, std::map<Key, int64_t>& out)
{
    if (values.empty()) {
        return false;
    }
    std::map<Key, int64_t> mapped;
    for (const auto& value : values) {
        const auto* pair = std::any_cast<std::vector<std::any>>(&value);
        if (pair == nullptr || pair->size() != 2) {
            return false;
        }
        const auto* key = std::any_cast<Key>(&(*pair)[0]);
        const auto* mappedValue = std::any_cast<int64_t>(&(*pair)[1]);
        if (key == nullptr || mappedValue == nullptr) {
            return false;
        }
        mapped[*key] = *mappedValue;
    }
    out = std::move(mapped);
    return true;
}

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

template <typename T>
std::any IRTextParser::ParseAs(const std::string&, const std::any& value)
{
    return static_cast<T>(AnyCast<int64_t>(value));
}

template <>
std::any IRTextParser::ParseAs<std::vector<npu::tile_fwk::SymbolicScalar>>(const std::string& key,
                                                                           const std::any& value)
{
    std::vector<npu::tile_fwk::SymbolicScalar> conds;
    if (!FillSymbolicScalarListAttr(key, value, conds)) {
        return {};
    }
    return conds;
}

template <>
std::any IRTextParser::ParseAs<std::shared_ptr<npu::tile_fwk::ConfigScope>>(const std::string&, const std::any&)
{
    return npu::tile_fwk::ConfigManagerNg::CurrentScope();
}

bool IRTextParser::FillForStmtAttrs(const std::map<std::string, std::any>& attrDict,
                                    std::vector<std::pair<std::string, std::any>>& attrs)
{
    using AttrParser = std::any (IRTextParser::*)(const std::string&, const std::any&);
    static const std::map<std::string, AttrParser> parsers = {
        {"_loop_conds", &IRTextParser::ParseAs<std::vector<npu::tile_fwk::SymbolicScalar>>},
        {"unroll_times", &IRTextParser::ParseAs<int>},
        {"_config_scope", &IRTextParser::ParseAs<std::shared_ptr<npu::tile_fwk::ConfigScope>>},
    };
    attrs.reserve(attrDict.size());
    for (const auto& [key, value] : attrDict) {
        const auto it = parsers.find(key);
        if (it == parsers.end()) {
            attrs.emplace_back(key, value);
            continue;
        }
        std::any rebuilt = (this->*(it->second))(key, value);
        if (!rebuilt.has_value()) {
            Error("invalid attr value for #" + key);
            return false;
        }
        attrs.emplace_back(key, rebuilt);
    }
    return true;
}

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
    // Loaded tensors start with an empty shape: rebuild the dyn shape from #shape (read later via GetDynRawShape).
    if (t->tensor != nullptr && t->tensor->GetDynRawShape().empty() && !values.shape.empty()) {
        t->tensor->UpdateDynRawShape(npu::tile_fwk::SymbolicScalar::FromConcrete(values.shape));
    }
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
    // Per-op tile is captured from the config scope at construction, so restore it from the text.
    static const std::unordered_set<std::string> kTileAttrs = {"vec_tile", "cube_tile", "dist_tile"};
    for (const auto& [key, value] : attrDict) {
        if (kTileAttrs.count(key) != 0) {
            SetOpTileShape(*operation, key, value);
        }
    }
    // Remaining generic op attrs (SCALAR/AXIS/mode) read back via GetAttr; keys with dedicated
    // handling (tile attrs above, view/assemble offsets and valid shapes below) are skipped here.
    static const std::unordered_set<std::string> kReservedAttrs = {"fromOffset",  "fromDynOffset", "toOffset",
                                                                   "toDynOffset", "toValidShape",  "fromValidShape"};
    for (const auto& [key, value] : attrDict) {
        if (kTileAttrs.count(key) == 0 && kReservedAttrs.count(key) == 0) {
            SetOpGenericAttr(*operation, key, value);
        }
    }
    if (npu::tile_fwk::IsViewLike(operation->GetOpcode())) {
        return SetViewOpAttrDict(*operation, attrDict);
    }
    if (npu::tile_fwk::IsAssembleLike(operation->GetOpcode())) {
        return SetAssembleOpAttrDict(*operation, attrDict);
    }
    return true;
}

void IRTextParser::SetOpTileShape(npu::tile_fwk::Operation& op, const std::string& key, const std::any& value)
{
    auto& tileShape = op.GetTileShapeForSetting();
    if (key == "vec_tile") {
        SetValue(tileShape.GetVecTile().tile, value);
    } else if (key == "cube_tile") {
        std::vector<int64_t> cube;
        SetValue(cube, value);
        if (cube.size() == 8) {
            auto& cubeTile = tileShape.GetCubeTile();
            cubeTile.m = {cube[0], cube[1]};
            cubeTile.k = {cube[2], cube[3], cube[4]};
            cubeTile.n = {cube[5], cube[6]};
            cubeTile.enableSplitK = (cube[7] != 0);
        }
    } else if (key == "dist_tile") {
        std::vector<int64_t> dist;
        SetValue(dist, value);
        if (dist.size() == 10) {
            auto& distTile = tileShape.GetDistTile();
            for (int i = 0; i < 3; i++) {
                distTile.row[i] = static_cast<int>(dist[i]);
                distTile.col[i] = static_cast<int>(dist[3 + i]);
                distTile.rank[i] = static_cast<int>(dist[6 + i]);
            }
            distTile.rankId = static_cast<int>(dist[9]);
        }
    }
}

void IRTextParser::SetOpGenericAttr(npu::tile_fwk::Operation& op, const std::string& key, const std::any& value)
{
    if (value.type() == typeid(int64_t)) {
        op.SetAttr(key, AnyCast<int64_t>(value));
    } else if (value.type() == typeid(bool)) {
        op.SetAttr(key, AnyCast<bool>(value));
    } else if (value.type() == typeid(std::string)) {
        op.SetAttr(key, AnyCast<std::string>(value));
    } else if (value.type() == typeid(double)) {
        op.SetAttr(key, AnyCast<double>(value));
    } else if (value.type() == typeid(npu::tile_fwk::Element)) {
        op.SetAttr(key, AnyCast<npu::tile_fwk::Element>(value));
    } else if (value.type() == typeid(npu::tile_fwk::SymbolicScalar)) {
        op.SetAttr(key, AnyCast<npu::tile_fwk::SymbolicScalar>(value));
    } else if (value.type() == typeid(std::vector<std::any>)) {
        std::vector<int64_t> vec;
        SetValue(vec, value);
        op.SetAttr(key, vec);
    }
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

IRTextParserAttrValuePtr IRTextParser::ParseAttrValueString()
{
    auto value = DecodeString(tokens_[cur_++].text);
    return std::make_shared<IRTextParserAttrValuePtr::element_type>(std::any(std::move(value)));
}

IRTextParserAttrValuePtr IRTextParser::ParseAttrValueList()
{
    // Nonempty lists of homogeneous key/integer pairs become maps; other lists remain vectors.
    MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunLBracket));
    std::vector<std::any> vals;
    if (!CheckToken(IRTextLexerTokenKind::PunRBracket)) {
        do {
            MUST_VALID(val, ParseAttrValueRaw());
            vals.push_back(val->data);
        } while (CheckToken(IRTextLexerTokenKind::PunComma, true));
    }
    MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunRBracket));
    std::map<int64_t, int64_t> intMap;
    if (AsKeyMap(vals, intMap)) {
        return std::make_shared<IRTextParserAttrValuePtr::element_type>(std::any(std::move(intMap)));
    }
    std::map<std::string, int64_t> stringMap;
    if (AsKeyMap(vals, stringMap)) {
        return std::make_shared<IRTextParserAttrValuePtr::element_type>(std::any(std::move(stringMap)));
    }
    return std::make_shared<IRTextParserAttrValuePtr::element_type>(std::any(std::move(vals)));
}

IRTextParserAttrValuePtr IRTextParser::ParseAttrValueExpr()
{
    // expr fallback: keep the parsed expr as std::any(ExprPtr)
    MUST_VALID(expr, ParseExpr());
    std::any data = std::any(expr);
    return std::make_shared<IRTextParserAttrValuePtr::element_type>(data);
}

IRTextParserAttrValuePtr IRTextParser::ParseAttrValueElem()
{
    ++cur_; // consume the 'v0elem' keyword
    MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunLParen));
    MUST_VALID(dtype, ParseAttrValueIdent());
    auto code = GetTileFwkDataTypeDict().Find(AnyCast<std::string>(dtype->data), npu::tile_fwk::DT_BOTTOM);
    if (code == npu::tile_fwk::DT_BOTTOM) {
        return Error("unknown dtype in v0elem");
    }
    MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunComma));
    MUST_VALID(data, ParseAttrValueRaw());
    MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunRParen));
    std::any elem = (data->data.type() == typeid(double)) ?
                        std::any(npu::tile_fwk::Element(code, AnyCast<double>(data->data))) :
                        std::any(npu::tile_fwk::Element(code, AnyCast<int64_t>(data->data)));
    return std::make_shared<IRTextParserAttrValuePtr::element_type>(elem);
}

IRTextParserAttrValuePtr IRTextParser::ParseAttrValueSym()
{
    ++cur_; // consume the 'v0sym' keyword
    MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunLParen));
    MUST_VALID(inner, ParseAttrValueRaw());
    MUST_MATCH(ParseToken(IRTextLexerTokenKind::PunRParen));
    const std::any& v = inner->data;
    npu::tile_fwk::SymbolicScalar sym;
    if (v.type() == typeid(int64_t)) {
        sym = npu::tile_fwk::SymbolicScalar(AnyCast<int64_t>(v));
    } else if (v.type() == typeid(std::string)) {
        sym = npu::tile_fwk::SymbolicScalar(AnyCast<std::string>(v));
    } else if (v.type() == typeid(ExprPtr)) {
        sym = npu::tile_fwk::SymbolicScalar(std::dynamic_pointer_cast<npu::tile_fwk::RawSymbolicScalar>(
            std::const_pointer_cast<ir::Expr>(AnyCast<ExprPtr>(v))));
    } else {
        return Error("unsupported value in v0sym");
    }
    return std::make_shared<IRTextParserAttrValuePtr::element_type>(std::any(sym));
}

IRTextParserAttrValuePtr IRTextParser::ParseAttrValueRaw()
{
    // Empty attribute value, e.g. `program #entry()` when there is no entry function.
    if (Peek().kind == IRTextLexerTokenKind::PunRParen) {
        std::any data = std::string("");
        return std::make_shared<IRTextParserAttrValuePtr::element_type>(data);
    }
    static const std::unordered_map<IRTextLexerTokenKind, ParseAttrValueFn> dispatch = {
        {IRTextLexerTokenKind::TokInt, &IRTextParser::ParseAttrValueInt},
        {IRTextLexerTokenKind::TokFloat, &IRTextParser::ParseAttrValueFloat},
        {IRTextLexerTokenKind::TokIdent, &IRTextParser::ParseAttrValueIdent},
        {IRTextLexerTokenKind::TokString, &IRTextParser::ParseAttrValueString},
        {IRTextLexerTokenKind::KwV0ScalarElement, &IRTextParser::ParseAttrValueElem},
        {IRTextLexerTokenKind::KwV0ScalarSymbolic, &IRTextParser::ParseAttrValueSym},
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
