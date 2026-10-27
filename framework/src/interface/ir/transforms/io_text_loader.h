/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#ifndef PYPTO_IR_TRANSFORMS_IO_TEXT_LOADER_H_
#define PYPTO_IR_TRANSFORMS_IO_TEXT_LOADER_H_

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "ir/core.h"
#include "ir/expr.h"
#include "ir/span.h"
#include "ir/stmt.h"
#include "ir/transforms/io_text.h"
#include "ir/type.h"

#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/tensor/symbolic_scalar.h"

namespace npu {
namespace tile_fwk {
class Operation;
} // namespace tile_fwk
} // namespace npu

namespace pypto {
namespace ir {

/// Token produced by the IR text lexer.
struct IRTextLexerToken {
    IRTextLexerTokenKind kind = IRTextLexerTokenKind::Invalid;
    std::string text; // ident text, punct char, or quoted string lexeme
    int64_t intVal = 0;
    double floatVal = 0.0;
    size_t begin = 0; // byte offset of token start in source
    size_t end = 0;   // byte offset of token end in source
};

#define IR_TEXT_PARSER_RESULT "IRTextParserResult"
#define IR_TEXT_PARSER_SUCCESS "IRTextParserSuccess"

struct IRTextParserResult : public IRNode {
    IRTextParserResult() : IRNode(Span::Unknown()) {}
    ObjectKind GetKind() const override { return ObjectKind::Invalid; }
    std::string TypeName() const override { return IR_TEXT_PARSER_RESULT; }
};

struct IRTextParserSuccess : public IRTextParserResult {
    IRTextParserSuccess() : IRTextParserResult() {}
    std::string TypeName() const override { return IR_TEXT_PARSER_SUCCESS; }
};
using IRTextParserSuccessPtr = std::shared_ptr<const IRTextParserSuccess>;

template <typename T>
struct IRTextParserData : public IRTextParserResult {
    IRTextParserData() : IRTextParserResult() {}
    IRTextParserData(const T& d) : data(d) {}
    T data;
    std::string TypeName() const override { return typeid(T).name(); }
};
template <typename T>
using IRTextParserDataPtr = std::shared_ptr<IRTextParserData<T>>;

struct VarDefList {
    std::vector<VarPtr> valueList;
    std::vector<VarPtr> tokenList;

    const std::vector<VarPtr>& GetValueList() const { return valueList; }
    const std::vector<VarPtr>& GetTokenList() const { return tokenList; }
    std::vector<VarPtr> GetVarDefList() const
    {
        std::vector<VarPtr> resultList;
        resultList.insert(resultList.end(), valueList.begin(), valueList.end());
        resultList.insert(resultList.end(), tokenList.begin(), tokenList.end());
        return resultList;
    }
};

using IRTextParserStringPtr = IRTextParserDataPtr<std::string>;
using IRTextParserAttrValuePtr = IRTextParserDataPtr<std::any>;
using IRTextParserExprListPtr = IRTextParserDataPtr<std::vector<ExprPtr>>;
using IRTextParserIterArgListPtr = IRTextParserDataPtr<std::vector<IterArgPtr>>;
using IRTextParserAttrDictPtr = IRTextParserDataPtr<std::map<std::string, std::any>>;
using IRTextParserOptionalTensorViewPtr = IRTextParserDataPtr<std::optional<TensorView>>;
using IRTextParserOptionalTileViewPtr = IRTextParserDataPtr<std::optional<TileView>>;
using IRTextParserOptionalHardwareInfoPtr = IRTextParserDataPtr<std::optional<HardwareInfo>>;
using IRTextParserVarDefListPtr = IRTextParserDataPtr<VarDefList>;

#define MUST_MATCH(call)    \
    do {                    \
        if (!(call))        \
            return nullptr; \
    } while (0)

#define MUST_VALID(name, call) \
    auto name = (call);        \
    do {                       \
        if (!(name))           \
            return nullptr;    \
    } while (0)

inline bool IsKw(IRTextLexerTokenKind k)
{
    return k >= IRTextLexerTokenKind::KwProgram && k <= IRTextLexerTokenKind::KwUOpCast;
}

inline bool IsDTypeKw(IRTextLexerTokenKind k)
{
    return k >= IRTextLexerTokenKind::KwTypeBegin && k <= IRTextLexerTokenKind::KwTypeEnd;
}

inline bool IsIdentOrKw(IRTextLexerTokenKind k)
{
    return k == IRTextLexerTokenKind::TokIdent || IsKw(k) || IsDTypeKw(k);
}

/// Recursive-descent parser for the IR text grammar produced by IRTextDumper.
///
/// Declared here (implementation split across io_text_loader.cpp and
/// io_text_loader_attr.cpp) so that the attribute grammar part can live in its
/// own translation unit, mirroring the dumper side.
class IRTextParser {
public:
    explicit IRTextParser(const std::string& text);

    // ---- Helpers ----
    const IRTextLexerToken& Peek() const { return tokens_[cur_]; }
    const IRTextLexerToken& PeekAt(size_t offset) const { return tokens_[std::min(cur_ + offset, tokens_.size() - 1)]; }
    bool CheckToken(IRTextLexerTokenKind kind, bool consumeIfPass = false);
    bool CheckVarName() const { return Peek().kind == IRTextLexerTokenKind::TokVarName; }
    std::nullptr_t Error(const std::string& msg);
    IRNodePtr ParseToken(IRTextLexerTokenKind kind);

    // ---- Generic std::any -> typed value setters ----
    void SetValue(npu::tile_fwk::DataType& dst, const std::any& src);
    void SetValue(std::vector<int64_t>& dst, const std::any& src);
    void SetValue(std::vector<npu::tile_fwk::SymbolicScalar>& dst, const std::any& src);
    void SetOpTileShape(npu::tile_fwk::Operation& op, const std::string& key, const std::any& value);
    void SetOpGenericAttr(npu::tile_fwk::Operation& op, const std::string& key, const std::any& value);

    IRTextParserStringPtr ParseIdent();
    IRTextParserStringPtr ParseVarName();
    IRTextParserDataPtr<int64_t> ParseOpMagic();
    IRTextParserDataPtr<DataType> ParseDType();
    IRTextParserExprListPtr ParseExprList(IRTextLexerTokenKind split, IRTextLexerTokenKind stop);
    IRTextParserIterArgListPtr ParseIterArgs();

    // ---- Var definition:  type %name ----
    VarPtr ParseVarDef();
    VarPtr ParseLogicalTensorVar(const std::string& varName);

    // ---- Shape parsing:  dim0 x dim1 x ... ----
    IRTextParserExprListPtr ParseShape();

    // ---- Operator helpers ----
    ExprPtr MakeUnary(const std::string& opName, const ExprPtr& operand, const TypePtr& type);
    ExprPtr MakeBinary(const std::string& opName, const ExprPtr& left, const ExprPtr& right);
    static bool IsUnaryOpName(const std::string& s)
    {
        return s == IR_KW_SCALAR_UOP_ABS || s == IR_KW_SCALAR_UOP_NEG || s == IR_KW_SCALAR_UOP_NOT ||
               s == IR_KW_SCALAR_UOP_INV || s == IR_KW_SCALAR_UOP_CAST;
    }

    // ---- Type parsing (per-branch, dispatched by token) ----
    TypePtr ParseTypeUnknown();
    TypePtr ParseTypeMemRefType();
    TypePtr ParseTypeToken();
    TypePtr ParseTypeNone();
    TypePtr ParseTypeLogicalTensor();
    TypePtr ParseTypePtr();
    TypePtr ParseTypeTuple();
    TypePtr ParseTypeTensor();
    TypePtr ParseTypeTile();
    TypePtr ParseTypeMutexTile();
    TypePtr ParseTypeScalar();
    IRTextParserOptionalTensorViewPtr ParserTensorTypeTensorView();
    TypePtr ParseTensorType();
    IRTextParserOptionalTileViewPtr ParseTileTypeTileView();
    IRTextParserOptionalHardwareInfoPtr ParseTileTypeHardwareInfo();
    TypePtr ParseTileType();
    TypePtr ParseType();

    // ---- Expression parsing ----
    ExprPtr ParseExprVarName();
    ExprPtr ParseExprInt();
    ExprPtr ParseExprFloat();
    ExprPtr ParseExprTrue();
    ExprPtr ParseExprFalse();
    ExprPtr ParseExprTuple();
    ExprPtr ParseExprGetItem();
    ExprPtr ParseExprMemref();
    ExprPtr ParseExprScalarExpr();
    ExprPtr ParseExprCall();
    ExprPtr ParseExprParen();
    ExprPtr ParseExpr();

    // ---- Statement parsing ----
    IRTextParserVarDefListPtr ParseVarDefList(IRTextLexerTokenKind split, IRTextLexerTokenKind stop);
    StmtPtr ParseStmtSeq(const VarDefList& = {});
    StmtPtr ParseStmtYield(const VarDefList&);
    StmtPtr ParseStmtReturn(const VarDefList&);
    StmtPtr ParseStmtIf(const VarDefList& defVarList);
    StmtPtr ParseStmtFor(const VarDefList& defVarList);
    StmtPtr ParseStmtWhile(const VarDefList& defVarList);
    StmtPtr ParseStmtSection(const VarDefList&);
    StmtPtr ParseStmtEval(const VarDefList&);
    StmtPtr ParseStmtBreak(const VarDefList&);
    StmtPtr ParseStmtContinue(const VarDefList&);
    StmtPtr ParseStmtTensorOrScalarOp(const VarDefList& defVarList);
    StmtPtr ParseStmtV0TensorOp(const VarDefList& defVarList);
    StmtPtr ParseStmtAssign(const VarDefList& defVarList);
    StmtPtr ParseStmtVarDef();
    StmtPtr ParseStmt();

    // ---- Function / Program parsing ----
    FunctionPtr ParseFunction();
    ProgramPtr ParseProgram();

    // ---- Attr dict:  #name(attr_value) { #name(attr_value) } ----
    IRTextParserAttrDictPtr ParseAttrDict();

protected:
    using ParseAttrValueFn = IRTextParserAttrValuePtr (IRTextParser::*)();

    bool CheckAttrName() const { return Peek().kind == IRTextLexerTokenKind::TokAttrName; }
    IRTextParserStringPtr ParseAttrName();

    // ---- Attr value parsing (per-branch, dispatched by token) ----
    IRTextParserAttrValuePtr ParseAttrValueInt();
    IRTextParserAttrValuePtr ParseAttrValueFloat();
    IRTextParserAttrValuePtr ParseAttrValueTrue();
    IRTextParserAttrValuePtr ParseAttrValueFalse();
    IRTextParserAttrValuePtr ParseAttrValueIdent();
    IRTextParserAttrValuePtr ParseAttrValueString();
    IRTextParserAttrValuePtr ParseAttrValueList();
    IRTextParserAttrValuePtr ParseAttrValueExpr();
    IRTextParserAttrValuePtr ParseAttrValueRaw();
    IRTextParserAttrValuePtr ParseAttrValueElem();
    IRTextParserAttrValuePtr ParseAttrValueSym();

    bool SetOperationAttrDict(const TensorOpStmtPtr& op, const std::map<std::string, std::any>& attrDict);
    bool SetViewOpAttrDict(npu::tile_fwk::Operation& op, const std::map<std::string, std::any>& attrDict);
    bool SetAssembleOpAttrDict(npu::tile_fwk::Operation& op, const std::map<std::string, std::any>& attrDict);
    bool FillOffsetAttr(const std::string& key, const std::any& value, std::vector<int64_t>& offset,
                        std::vector<npu::tile_fwk::SymbolicScalar>& dynOffset);
    bool FillSymbolicScalarListAttr(const std::string& key, const std::any& value,
                                    std::vector<npu::tile_fwk::SymbolicScalar>& out);

    // ---- v0_logical_tensor var-def attrs ----
    /// Parsed attr values of a v0_logical_tensor var-def.
    struct TensorVarAttrValues {
        npu::tile_fwk::DataType dtype = npu::tile_fwk::DT_INT64;
        std::vector<int64_t> shape;
        std::vector<int64_t> offset;
        std::vector<npu::tile_fwk::SymbolicScalar> dynValidShape;
        std::vector<npu::tile_fwk::SymbolicScalar> dynOffset;
    };
    bool SetTensorAttrDict(const VarPtr& var, const std::map<std::string, std::any>& attrDict);
    bool SetFunctionAttrDict(const FunctionPtr& func, const std::map<std::string, std::any>& attrDict);

private:
    using ParseStmtFn = StmtPtr (IRTextParser::*)(const VarDefList&);

    // ---- Member variables ----
    std::vector<IRTextLexerToken> tokens_;
    size_t cur_ = 0;
    std::unordered_map<std::string, VarPtr> symtab_;
    IRTextParserSuccessPtr success_;
    std::string errorMsg_;
    size_t errorPos_ = 0;
    std::unique_ptr<npu::tile_fwk::Program> tempProgram_;
    std::shared_ptr<npu::tile_fwk::Function> tempFunc_;

    friend class IRTextLoader;
};

} // namespace ir
} // namespace pypto

#endif // PYPTO_IR_TRANSFORMS_IO_TEXT_LOADER_H_
