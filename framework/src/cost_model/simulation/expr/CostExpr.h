/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file CostExpr.h
 * \brief Architecture-neutral expression evaluator for cost models.
 *
 * Grammar (intentionally small):
 *   expr    := term (('+' | '-') term)*
 *   term    := unary (('*' | '/') unary)*
 *   unary   := ('+' | '-') unary | primary
 *   primary := number | keyword | func '(' expr (',' expr)* ')' | '(' expr ')'
 *   func    := min | max | clamp | floor | ceil | select | switch | log
 *
 * The expression only knows features supplied by FeatureContext.  Coefficients are passed as
 * x1..xN (or any keyword) so the same rule can be fitted offline.  Missing features evaluate to
 * 0; callers that need validation should use Expr::CollectKeywords().
 */

#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace CostModel {

class FeatureContext {
public:
    void Set(const std::string& name, double value) { values_[name] = value; }

    double Get(const std::string& name) const
    {
        auto iter = values_.find(name);
        return iter == values_.end() ? 0.0 : iter->second;
    }

    bool Has(const std::string& name) const { return values_.find(name) != values_.end(); }

private:
    std::unordered_map<std::string, double> values_;
};

class Expr {
public:
    virtual ~Expr() = default;
    virtual double Eval(const FeatureContext& ctx) const = 0;
    virtual void CollectKeywords(std::set<std::string>& keywords) const = 0;
};

using ExprPtr = std::shared_ptr<const Expr>;

class ExprParser {
public:
    // Returns nullptr and fills `error` on a syntax/shape error.
    static ExprPtr Parse(const std::string& text, std::string& error);
};

namespace detail {

class NumberExpr final : public Expr {
public:
    explicit NumberExpr(double value) : value_(value) {}
    double Eval(const FeatureContext&) const override { return value_; }
    void CollectKeywords(std::set<std::string>&) const override {}

private:
    double value_;
};

class KeywordExpr final : public Expr {
public:
    explicit KeywordExpr(std::string name) : name_(std::move(name)) {}
    double Eval(const FeatureContext& ctx) const override { return ctx.Get(name_); }
    void CollectKeywords(std::set<std::string>& keywords) const override { keywords.insert(name_); }

private:
    std::string name_;
};

class UnaryExpr final : public Expr {
public:
    UnaryExpr(bool negate, ExprPtr operand) : negate_(negate), operand_(std::move(operand)) {}
    double Eval(const FeatureContext& ctx) const override
    {
        double value = operand_->Eval(ctx);
        return negate_ ? -value : value;
    }
    void CollectKeywords(std::set<std::string>& keywords) const override { operand_->CollectKeywords(keywords); }

private:
    bool negate_;
    ExprPtr operand_;
};

class BinaryExpr final : public Expr {
public:
    enum class Op { Add, Sub, Mul, Div };

    BinaryExpr(Op op, ExprPtr lhs, ExprPtr rhs) : op_(op), lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}

    double Eval(const FeatureContext& ctx) const override
    {
        const double left = lhs_->Eval(ctx);
        const double right = rhs_->Eval(ctx);
        switch (op_) {
            case Op::Add:
                return left + right;
            case Op::Sub:
                return left - right;
            case Op::Mul:
                return left * right;
            case Op::Div:
                return std::fabs(right) < std::numeric_limits<double>::min() ? std::numeric_limits<double>::infinity() :
                                                                               left / right;
            default:
                return 0.0;
        }
    }

    void CollectKeywords(std::set<std::string>& keywords) const override
    {
        lhs_->CollectKeywords(keywords);
        rhs_->CollectKeywords(keywords);
    }

private:
    Op op_;
    ExprPtr lhs_;
    ExprPtr rhs_;
};

class CallExpr final : public Expr {
public:
    enum class Func { Min, Max, Clamp, Floor, Ceil, Select, Log };

    CallExpr(Func func, std::vector<ExprPtr> args) : func_(func), args_(std::move(args)) {}

    double Eval(const FeatureContext& ctx) const override
    {
        switch (func_) {
            case Func::Min: {
                double value = args_[0]->Eval(ctx);
                for (size_t i = 1; i < args_.size(); ++i) {
                    value = std::min(value, args_[i]->Eval(ctx));
                }
                return value;
            }
            case Func::Max: {
                double value = args_[0]->Eval(ctx);
                for (size_t i = 1; i < args_.size(); ++i) {
                    value = std::max(value, args_[i]->Eval(ctx));
                }
                return value;
            }
            case Func::Clamp: {
                const double value = args_[0]->Eval(ctx);
                const double low = args_[1]->Eval(ctx);
                const double high = args_[2]->Eval(ctx);
                return std::min(std::max(value, low), high);
            }
            case Func::Floor:
                return std::floor(args_[0]->Eval(ctx));
            case Func::Ceil:
                return std::ceil(args_[0]->Eval(ctx));
            case Func::Select:
                // Returns one operand verbatim (no arithmetic), so calibrated factors
                // stay bit-exact in both branches.  The condition is a 0/1 flag.
                return args_[0]->Eval(ctx) > 0.0 ? args_[1]->Eval(ctx) : args_[2]->Eval(ctx);
            case Func::Log: {
                const double base = args_[0]->Eval(ctx);
                const double value = args_[1]->Eval(ctx);
                if (base <= 0.0 || std::fabs(base - 1.0) <= std::numeric_limits<double>::epsilon() || value <= 0.0) {
                    return std::numeric_limits<double>::quiet_NaN();
                }
                return std::log(value) / std::log(base);
            }
            default:
                return 0.0;
        }
    }

    void CollectKeywords(std::set<std::string>& keywords) const override
    {
        for (const auto& arg : args_) {
            arg->CollectKeywords(keywords);
        }
    }

private:
    Func func_;
    std::vector<ExprPtr> args_;
};

class SwitchExpr final : public Expr {
public:
    SwitchExpr(ExprPtr index, std::vector<ExprPtr> branches) : index_(std::move(index)), branches_(std::move(branches))
    {}

    double Eval(const FeatureContext& ctx) const override
    {
        const double raw = index_->Eval(ctx);
        if (!std::isfinite(raw) || raw < 0.0 || raw >= static_cast<double>(branches_.size())) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return branches_[static_cast<size_t>(raw)]->Eval(ctx);
    }

    void CollectKeywords(std::set<std::string>& keywords) const override
    {
        index_->CollectKeywords(keywords);
        for (const auto& branch : branches_) {
            branch->CollectKeywords(keywords);
        }
    }

private:
    ExprPtr index_;
    std::vector<ExprPtr> branches_;
};

class Parser {
public:
    Parser(const std::string& text, std::string& error) : text_(text), error_(error) { Next(); }

    ExprPtr Parse()
    {
        ExprPtr expr = ParseExpression();
        if (expr == nullptr) {
            return nullptr;
        }
        if (token_.type != TokenType::End) {
            Fail("unexpected token '" + token_.text + "'");
            return nullptr;
        }
        return expr;
    }

private:
    enum class TokenType { Number, Identifier, Plus, Minus, Star, Slash, LParen, RParen, LBrace, RBrace, Comma, End };

    struct Token {
        TokenType type = TokenType::End;
        std::string text;
        double number = 0.0;
        size_t pos = 0;
    };

    void Next()
    {
        while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) {
            ++pos_;
        }
        token_ = Token();
        token_.pos = pos_;
        if (pos_ >= text_.size()) {
            token_.type = TokenType::End;
            return;
        }
        const char ch = text_[pos_];
        if (std::isdigit(static_cast<unsigned char>(ch)) || ch == '.') {
            ParseNumber();
            return;
        }
        if (std::isalpha(static_cast<unsigned char>(ch)) || ch == '_') {
            size_t begin = pos_;
            while (pos_ < text_.size()) {
                const char c = text_[pos_];
                if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
                    break;
                }
                ++pos_;
            }
            token_.type = TokenType::Identifier;
            token_.text = text_.substr(begin, pos_ - begin);
            return;
        }
        ++pos_;
        switch (ch) {
            case '+':
                token_.type = TokenType::Plus;
                break;
            case '-':
                token_.type = TokenType::Minus;
                break;
            case '*':
                token_.type = TokenType::Star;
                break;
            case '/':
                token_.type = TokenType::Slash;
                break;
            case '(':
                token_.type = TokenType::LParen;
                break;
            case ')':
                token_.type = TokenType::RParen;
                break;
            case '{':
                token_.type = TokenType::LBrace;
                break;
            case '}':
                token_.type = TokenType::RBrace;
                break;
            case ',':
                token_.type = TokenType::Comma;
                break;
            default:
                Fail(std::string("unexpected character '") + ch + "'");
                token_.type = TokenType::End;
                break;
        }
    }

    void ParseNumber()
    {
        const char* begin = text_.c_str() + pos_;
        char* end = nullptr;
        const double value = std::strtod(begin, &end);
        if (end == begin) {
            Fail("invalid number");
            token_.type = TokenType::End;
            return;
        }
        pos_ += static_cast<size_t>(end - begin);
        token_.type = TokenType::Number;
        token_.number = value;
        token_.text = text_.substr(token_.pos, pos_ - token_.pos);
    }

    ExprPtr ParseExpression()
    {
        ExprPtr lhs = ParseTerm();
        while (lhs != nullptr && (token_.type == TokenType::Plus || token_.type == TokenType::Minus)) {
            const bool add = token_.type == TokenType::Plus;
            Next();
            ExprPtr rhs = ParseTerm();
            if (rhs == nullptr) {
                return nullptr;
            }
            lhs = std::make_shared<BinaryExpr>(add ? BinaryExpr::Op::Add : BinaryExpr::Op::Sub, lhs, rhs);
        }
        return lhs;
    }

    ExprPtr ParseTerm()
    {
        ExprPtr lhs = ParseUnary();
        while (lhs != nullptr && (token_.type == TokenType::Star || token_.type == TokenType::Slash)) {
            const bool mul = token_.type == TokenType::Star;
            Next();
            ExprPtr rhs = ParseUnary();
            if (rhs == nullptr) {
                return nullptr;
            }
            lhs = std::make_shared<BinaryExpr>(mul ? BinaryExpr::Op::Mul : BinaryExpr::Op::Div, lhs, rhs);
        }
        return lhs;
    }

    ExprPtr ParseUnary()
    {
        if (token_.type == TokenType::Plus || token_.type == TokenType::Minus) {
            const bool negate = token_.type == TokenType::Minus;
            Next();
            ExprPtr operand = ParseUnary();
            if (operand == nullptr) {
                return nullptr;
            }
            return std::make_shared<UnaryExpr>(negate, operand);
        }
        return ParsePrimary();
    }

    ExprPtr ParsePrimary()
    {
        if (token_.type == TokenType::Number) {
            const double value = token_.number;
            Next();
            return std::make_shared<NumberExpr>(value);
        }
        if (token_.type == TokenType::Identifier) {
            const std::string name = token_.text;
            Next();
            if (token_.type != TokenType::LParen) {
                return std::make_shared<KeywordExpr>(name);
            }
            Next();
            if (name == "switch") {
                return ParseSwitch();
            }
            std::vector<ExprPtr> args;
            if (token_.type != TokenType::RParen) {
                while (true) {
                    ExprPtr arg = ParseExpression();
                    if (arg == nullptr) {
                        return nullptr;
                    }
                    args.push_back(std::move(arg));
                    if (token_.type == TokenType::Comma) {
                        Next();
                        continue;
                    }
                    break;
                }
            }
            if (token_.type != TokenType::RParen) {
                Fail("expected ')'");
                return nullptr;
            }
            Next();
            return MakeCall(name, std::move(args));
        }
        if (token_.type == TokenType::LParen) {
            Next();
            ExprPtr expr = ParseExpression();
            if (expr == nullptr) {
                return nullptr;
            }
            if (token_.type != TokenType::RParen) {
                Fail("expected ')'");
                return nullptr;
            }
            Next();
            return expr;
        }
        Fail("expected number, keyword or '('");
        return nullptr;
    }

    ExprPtr ParseSwitch()
    {
        ExprPtr index = ParseExpression();
        if (index == nullptr) {
            return nullptr;
        }
        if (token_.type != TokenType::Comma) {
            Fail("switch expects ',' before '{'");
            return nullptr;
        }
        Next();
        if (token_.type != TokenType::LBrace) {
            Fail("switch expects '{'");
            return nullptr;
        }
        Next();
        std::vector<ExprPtr> branches;
        if (token_.type != TokenType::RBrace) {
            while (true) {
                ExprPtr branch = ParseExpression();
                if (branch == nullptr) {
                    return nullptr;
                }
                branches.push_back(std::move(branch));
                if (token_.type == TokenType::Comma) {
                    Next();
                    continue;
                }
                break;
            }
        }
        if (token_.type != TokenType::RBrace) {
            Fail("expected '}'");
            return nullptr;
        }
        Next();
        if (token_.type != TokenType::RParen) {
            Fail("expected ')'");
            return nullptr;
        }
        Next();
        if (branches.empty()) {
            Fail("switch expects at least one branch");
            return nullptr;
        }
        return std::make_shared<SwitchExpr>(std::move(index), std::move(branches));
    }

    ExprPtr MakeCall(const std::string& name, std::vector<ExprPtr> args)
    {
        CallExpr::Func func;
        size_t arity = 0;
        bool variadic = false;
        if (name == "min") {
            func = CallExpr::Func::Min;
            arity = 2;
            variadic = true;
        } else if (name == "max") {
            func = CallExpr::Func::Max;
            arity = 2;
            variadic = true;
        } else if (name == "clamp") {
            func = CallExpr::Func::Clamp;
            arity = 3;
        } else if (name == "floor") {
            func = CallExpr::Func::Floor;
            arity = 1;
        } else if (name == "ceil") {
            func = CallExpr::Func::Ceil;
            arity = 1;
        } else if (name == "select") {
            func = CallExpr::Func::Select;
            arity = 3;
        } else if (name == "log") {
            func = CallExpr::Func::Log;
            arity = 2;
        } else {
            Fail("unknown function '" + name + "'");
            return nullptr;
        }
        if (variadic) {
            if (args.size() < arity) {
                Fail("function '" + name + "' expects at least " + std::to_string(arity) + " arguments");
                return nullptr;
            }
        } else if (args.size() != arity) {
            Fail("function '" + name + "' expects " + std::to_string(arity) + " arguments");
            return nullptr;
        }
        return std::make_shared<CallExpr>(func, std::move(args));
    }

    void Fail(const std::string& message)
    {
        if (error_.empty()) {
            error_ = message + " at position " + std::to_string(token_.pos);
        }
    }

    const std::string& text_;
    std::string& error_;
    size_t pos_ = 0;
    Token token_;
};

} // namespace detail

inline ExprPtr ExprParser::Parse(const std::string& text, std::string& error)
{
    error.clear();
    detail::Parser parser(text, error);
    return parser.Parse();
}

} // namespace CostModel
