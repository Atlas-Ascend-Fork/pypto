/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <any>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/any_cast.h"
#include "core/dtype.h"
#include "tilefwk/error.h"
#include "ir/core.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/program.h"
#include "ir/reflection/field_visitor.h"
#include "ir/scalar_expr.h"
#include "ir/span.h"
#include "ir/stmt.h"
#include "ir/transforms/printer.h"
#include "ir/transforms/structural_comparison.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

namespace {

bool AreForSyntaxScalarDtypesEquivalent(const DataType& lhs, const DataType& rhs)
{
    return (lhs == DataType::INT64 && rhs == DataType::INDEX) || (lhs == DataType::INDEX && rhs == DataType::INT64);
}

bool AreDoublesEquivalent(double lhs, double rhs) { return std::abs(lhs - rhs) <= 1e-10; }

} // namespace

/**
 * \brief Unified structural equality checker for IR nodes
 *
 * Template parameter controls behavior on mismatch:
 * - AssertMode=false: Returns false (for structural_equal)
 * - AssertMode=true: Throws npu::tile_fwk::Error with detailed error message (for assert_structural_equal)
 *
 * This class is not part of the public API - use structural_equal() or assert_structural_equal().
 *
 * Implements the FieldIterator visitor interface for generic field-based comparison.
 * Uses the dual-node Visit overload which calls visitor methods with two field arguments.
 */
template <bool AssertMode>
class StructuralEqualImpl {
public:
    using ResultType = bool;

    explicit StructuralEqualImpl(bool enableAutoMapping) : enableAutoMapping_(enableAutoMapping) {}

    // Returns bool for structural_equal, throws for assert_structural_equal
    bool operator()(const IRNodePtr& lhs, const IRNodePtr& rhs)
    {
        if constexpr (AssertMode) {
            Equal(lhs, rhs);
            return true; // Only reached if no exception thrown
        } else {
            return Equal(lhs, rhs);
        }
    }

    bool operator()(const TypePtr& lhs, const TypePtr& rhs)
    {
        if constexpr (AssertMode) {
            EqualType(lhs, rhs);
            return true; // Only reached if no exception thrown
        } else {
            return EqualType(lhs, rhs);
        }
    }

    // FieldIterator visitor interface (dual-node version - methods receive two fields)
    [[nodiscard]] ResultType InitResult() const { return true; }

    template <typename IRNodePtrType>
    ResultType VisitIRNodeField(const IRNodePtrType& lhs, const IRNodePtrType& rhs)
    {
        INTERNAL_CHECK(lhs) << "structural_equal encountered null lhs IR node field";
        INTERNAL_CHECK_SPAN(rhs, lhs->span_) << "structural_equal encountered null rhs IR node field";
        return Equal(lhs, rhs);
    }

    // Specialization for std::optional<IRNodePtr>
    template <typename IRNodePtrType>
    ResultType VisitIRNodeField(const std::optional<IRNodePtrType>& lhs, const std::optional<IRNodePtrType>& rhs)
    {
        if (!lhs.has_value() && !rhs.has_value()) {
            return true;
        }
        if (!lhs.has_value() || !rhs.has_value()) {
            return IRAssert({lhs.has_value() ? *lhs : IRNodePtr(), rhs.has_value() ? *rhs : IRNodePtr(),
                             lhs.has_value() ? "has value" : "nullopt", rhs.has_value() ? "has value" : "nullopt"},
                            "Optional field presence mismatch");
        }
        if (!*lhs && !*rhs) {
            return true;
        }
        if (!*lhs || !*rhs) {
            return IRAssert({*lhs, *rhs, *lhs ? "has value" : "nullptr", *rhs ? "has value" : "nullptr"},
                            "Optional field nullptr mismatch");
        }
        return Equal(*lhs, *rhs);
    }

    template <typename IRNodePtrType>
    ResultType VisitIRNodeVectorField(const std::vector<IRNodePtrType>& lhs, const std::vector<IRNodePtrType>& rhs)
    {
        if (lhs.size() != rhs.size()) {
            return IRAssert({}, "Vector size mismatch (", lhs.size(), " items != ", rhs.size(), " items)");
        }
        for (size_t i = 0; i < lhs.size(); ++i) {
            INTERNAL_CHECK(lhs[i]) << "structural_equal encountered null lhs IR node in vector at index " << i;
            INTERNAL_CHECK_SPAN(rhs[i], lhs[i]->span_)
                << "structural_equal encountered null rhs IR node in vector at index " << i;

            if constexpr (AssertMode) {
                std::ostringstream indexStr;
                indexStr << "[" << i << "]";
                path_.emplace_back(indexStr.str());
            }

            if (!Equal(lhs[i], rhs[i])) {
                if constexpr (AssertMode) {
                    path_.pop_back();
                }
                return false;
            }

            if constexpr (AssertMode) {
                path_.pop_back();
            }
        }
        return true;
    }

    template <typename MapType, typename EntryChecker, typename KeyFormatter>
    ResultType VisitIRNodeMapFieldImpl(const MapType& lhs, const MapType& rhs, EntryChecker&& checkEntry,
                                       KeyFormatter&& formatKey)
    {
        if (lhs.size() != rhs.size()) {
            return IRAssert({}, "Map size mismatch (", lhs.size(), " items != ", rhs.size(), " items)");
        }

        auto lhsIt = lhs.begin();
        auto rhsIt = rhs.begin();
        while (lhsIt != lhs.end()) {
            checkEntry(lhsIt, rhsIt);

            std::string lhsKey = formatKey(lhsIt->first);
            std::string rhsKey = formatKey(rhsIt->first);
            if (lhsKey != rhsKey) {
                return IRAssert({}, "Map key mismatch ('", lhsKey, "' != '", rhsKey, "')");
            }

            if constexpr (AssertMode) {
                path_.emplace_back("['" + lhsKey + "']");
            }

            if (!Equal(lhsIt->second, rhsIt->second)) {
                if constexpr (AssertMode) {
                    path_.pop_back();
                }
                return false;
            }

            if constexpr (AssertMode) {
                path_.pop_back();
            }
            ++lhsIt;
            ++rhsIt;
        }
        return true;
    }

    template <typename KeyType, typename ValueType, typename Compare>
    ResultType VisitIRNodeMapField(const std::map<KeyType, ValueType, Compare>& lhs,
                                   const std::map<KeyType, ValueType, Compare>& rhs)
    {
        return VisitIRNodeMapFieldImpl(
            lhs, rhs,
            [](const auto& lhsIt, const auto& rhsIt) {
                INTERNAL_CHECK(lhsIt->first) << "structural_equal encountered null lhs key in map";
                INTERNAL_CHECK(lhsIt->second) << "structural_equal encountered null lhs value in map";
                INTERNAL_CHECK(rhsIt->first) << "structural_equal encountered null rhs key in map";
                INTERNAL_CHECK_SPAN(rhsIt->second, lhsIt->second->span_)
                    << "structural_equal encountered null rhs value in map";
            },
            [](const auto& key) { return key->name_; });
    }

    template <typename ValueType>
    ResultType VisitIRNodeMapField(const std::map<std::string, ValueType>& lhs,
                                   const std::map<std::string, ValueType>& rhs)
    {
        return VisitIRNodeMapFieldImpl(
            lhs, rhs,
            [](const auto& lhsIt, const auto& rhsIt) {
                INTERNAL_CHECK(lhsIt->second) << "structural_equal encountered null lhs value in map";
                INTERNAL_CHECK_SPAN(rhsIt->second, lhsIt->second->span_)
                    << "structural_equal encountered null rhs value in map";
            },
            [](const std::string& key) { return key; });
    }

    // Leaf field comparisons (dual-node version)
    ResultType VisitLeafField(const int& lhs, const int& rhs)
    {
        if (lhs != rhs) {
            return IRAssert({}, "Integer value mismatch (", lhs, " != ", rhs, ")");
        }
        return true;
    }

    ResultType VisitLeafField(const int64_t& lhs, const int64_t& rhs)
    {
        if (lhs != rhs) {
            return IRAssert({}, "int64_t value mismatch (", lhs, " != ", rhs, ")");
        }
        return true;
    }

    ResultType VisitLeafField(const uint64_t& lhs, const uint64_t& rhs)
    {
        if (lhs != rhs) {
            return IRAssert({}, "uint64_t value mismatch (", lhs, " != ", rhs, ")");
        }
        return true;
    }

    ResultType VisitLeafField(const double& lhs, const double& rhs)
    {
        if (!AreDoublesEquivalent(lhs, rhs)) {
            return IRAssert({}, "double value mismatch (", lhs, " != ", rhs, ")");
        }
        return true;
    }

    ResultType VisitLeafField(const std::string& lhs, const std::string& rhs)
    {
        if (lhs != rhs) {
            return IRAssert({}, "String value mismatch (\"", lhs, "\" != \"", rhs, "\")");
        }
        return true;
    }

    ResultType VisitLeafField(const DataType& lhs, const DataType& rhs)
    {
        if (lhs != rhs) {
            return IRAssert({}, "DataType mismatch (", lhs.ToString(), " != ", rhs.ToString(), ")");
        }
        return true;
    }

    ResultType VisitLeafField(const FunctionType& lhs, const FunctionType& rhs)
    {
        if (lhs != rhs) {
            return IRAssert({}, "FunctionType mismatch (", FunctionTypeToString(lhs), " != ", FunctionTypeToString(rhs),
                            ")");
        }
        return true;
    }

    [[nodiscard]] ResultType VisitLeafField(const SectionKind& lhs, const SectionKind& rhs)
    {
        if (lhs != rhs) {
            return IRAssert({}, "SectionKind mismatch (", SectionKindToString(lhs), " != ", SectionKindToString(rhs),
                            ")");
        }
        return true;
    }

    // Compare kwargs (vector of pairs to preserve order)
    ResultType VisitLeafField(const std::vector<std::pair<std::string, std::any>>& lhs,
                              const std::vector<std::pair<std::string, std::any>>& rhs)
    {
        if (lhs.size() != rhs.size()) {
            return IRAssert({}, "Kwargs size mismatch (", lhs.size(), " != ", rhs.size(), ")");
        }
        for (size_t i = 0; i < lhs.size(); ++i) {
            if (lhs[i].first != rhs[i].first) {
                return IRAssert({}, "Kwargs key mismatch at index ", i, " ('", lhs[i].first, "' != '", rhs[i].first,
                                "')");
            }
            // Compare std::any values by type and content
            const auto& lhsVal = lhs[i].second;
            const auto& rhsVal = rhs[i].second;
            if (lhsVal.type() != rhsVal.type()) {
                return IRAssert({}, "Kwargs value type mismatch for key '", lhs[i].first, "'");
            }
            // Type-specific comparison
            bool valuesEqual = true;
            if (lhsVal.type() == typeid(int)) {
                valuesEqual = (AnyCast<int>(lhsVal, "comparing kwarg: " + lhs[i].first) ==
                               AnyCast<int>(rhsVal, "comparing kwarg: " + lhs[i].first));
            } else if (lhsVal.type() == typeid(bool)) {
                valuesEqual = (AnyCast<bool>(lhsVal, "comparing kwarg: " + lhs[i].first) ==
                               AnyCast<bool>(rhsVal, "comparing kwarg: " + lhs[i].first));
            } else if (lhsVal.type() == typeid(std::string)) {
                valuesEqual = (AnyCast<std::string>(lhsVal, "comparing kwarg: " + lhs[i].first) ==
                               AnyCast<std::string>(rhsVal, "comparing kwarg: " + lhs[i].first));
            } else if (lhsVal.type() == typeid(double)) {
                valuesEqual = AreDoublesEquivalent(AnyCast<double>(lhsVal, "comparing kwarg: " + lhs[i].first),
                                                   AnyCast<double>(rhsVal, "comparing kwarg: " + lhs[i].first));
            } else if (lhsVal.type() == typeid(DataType)) {
                valuesEqual = (AnyCast<DataType>(lhsVal, "comparing kwarg: " + lhs[i].first) ==
                               AnyCast<DataType>(rhsVal, "comparing kwarg: " + lhs[i].first));
            } else if (lhsVal.type() == typeid(std::vector<int>)) {
                valuesEqual = (AnyCast<std::vector<int>>(lhsVal, "comparing kwarg: " + lhs[i].first) ==
                               AnyCast<std::vector<int>>(rhsVal, "comparing kwarg: " + lhs[i].first));
            }
            if (!valuesEqual) {
                return IRAssert({}, "Kwargs value mismatch for key '", lhs[i].first, "'");
            }
        }
        return true;
    }

    ResultType VisitLeafField(const MemorySpace& lhs, const MemorySpace& rhs)
    {
        if (lhs != rhs) {
            return IRAssert({}, "MemorySpace mismatch (", MemorySpaceToString(lhs), " != ", MemorySpaceToString(rhs),
                            ")");
        }
        return true;
    }

    ResultType VisitLeafField(const TypePtr& lhs, const TypePtr& rhs) { return EqualType(lhs, rhs); }

    ResultType VisitLeafField(const std::vector<TypePtr>& lhs, const std::vector<TypePtr>& rhs)
    {
        if (lhs.size() != rhs.size()) {
            return IRAssert({}, "Type vector size mismatch (", lhs.size(), " types != ", rhs.size(), " types)");
        }
        for (size_t i = 0; i < lhs.size(); ++i) {
            INTERNAL_CHECK(lhs[i]) << "structural_equal encountered null lhs TypePtr in vector at index " << i;
            INTERNAL_CHECK(rhs[i]) << "structural_equal encountered null rhs TypePtr in vector at index " << i;
            if (!EqualType(lhs[i], rhs[i]))
                return false;
        }
        return true;
    }

    ResultType VisitLeafField(const std::vector<std::string>& lhs, const std::vector<std::string>& rhs)
    {
        if (lhs.size() != rhs.size()) {
            return IRAssert({}, "String vector size mismatch (", lhs.size(), " != ", rhs.size(), ")");
        }
        for (size_t i = 0; i < lhs.size(); ++i) {
            if (lhs[i] != rhs[i]) {
                return IRAssert({IRNodePtr(), IRNodePtr(), lhs[i], rhs[i]}, "string vector element mismatch");
            }
        }
        return true;
    }

    [[nodiscard]] ResultType VisitLeafField(const Span& lhs, const Span&) const
    {
        INTERNAL_CHECK_SPAN(false, lhs) << "structural_equal should not visit Span field";
        return true; // Never reached
    }

    ResultType VisitLeafField(const std::vector<IterArgPtr>& lhs, const std::vector<IterArgPtr>& rhs)
    {
        if (lhs.size() != rhs.size()) {
            return IRAssert({}, "IterArg vector size mismatch (", lhs.size(), " != ", rhs.size(), ")");
        }
        for (size_t i = 0; i < lhs.size(); ++i) {
            INTERNAL_CHECK(lhs[i]) << "structural_equal encountered null lhs IterArgPtr in vector at index " << i;
            INTERNAL_CHECK(rhs[i]) << "structural_equal encountered null rhs IterArgPtr in vector at index " << i;
            if (!EqualIterArg(lhs[i], rhs[i])) {
                return false;
            }
        }
        return true;
    }

    // Field kind hooks
    template <typename FVisitOp>
    void VisitIgnoreField([[maybe_unused]] FVisitOp&& visitOp)
    {
        // Ignored fields are always considered equal
    }

    template <typename FVisitOp>
    void VisitDefField(FVisitOp&& visitOp)
    {
        bool enableAutoMapping = true;
        std::swap(enableAutoMapping, enableAutoMapping_);
        visitOp();
        std::swap(enableAutoMapping, enableAutoMapping_);
    }

    template <typename FVisitOp>
    void VisitUsualField(FVisitOp&& visitOp)
    {
        visitOp();
    }

    // Path tracking hooks called by FieldIterator::VisitFieldImpl for each field.
    // Transparent containers (Program, SeqStmts) suppress their own field names
    // so that vector/map element accessors ([i] / ['key']) attach directly to the
    // parent field name, producing paths like body[1] instead of body.stmts[1].
    void PushFieldName(const char* name)
    {
        if (transparentDepth_ == 0) {
            fieldNameStack_.emplace_back(name);
        }
        if constexpr (AssertMode) {
            if (transparentDepth_ == 0) {
                path_.emplace_back(name);
            }
        }
    }

    void PopFieldName()
    {
        if (transparentDepth_ == 0) {
            fieldNameStack_.pop_back();
        }
        if constexpr (AssertMode) {
            if (transparentDepth_ == 0) {
                path_.pop_back();
            }
        }
    }

    // Combine results (AND logic)
    template <typename Desc>
    void CombineResult(ResultType& accumulator, ResultType fieldResult, [[maybe_unused]] const Desc& desc)
    {
        accumulator = accumulator && fieldResult;
    }

private:
    bool Equal(const IRNodePtr& lhs, const IRNodePtr& rhs);
    bool EqualVar(const VarPtr& lhs, const VarPtr& rhs);
    bool EqualMemRef(const MemRefPtr& lhs, const MemRefPtr& rhs);
    bool EqualIterArg(const IterArgPtr& lhs, const IterArgPtr& rhs);
    bool EqualType(const TypePtr& lhs, const TypePtr& rhs);
    bool IsLoopVarFieldContext() const { return !fieldNameStack_.empty() && fieldNameStack_.back() == "loop_var"; }
    bool IsConstIntTypeContext() const
    {
        return !nodeTypeStack_.empty() && nodeTypeStack_.back() == "ConstInt" && !fieldNameStack_.empty() &&
               fieldNameStack_.back() == "type";
    }

    /**
     * \brief Generic field-based equality check for IR nodes using FieldIterator
     *
     * Uses the dual-node Visit overload which passes two fields to each visitor method.
     *
     * \tparam NodePtr Shared pointer type to the node
     * \param lhsOp Left-hand side node
     * \param rhsOp Right-hand side node
     * \return true if all fields are equal
     */
    template <typename NodePtr>
    bool EqualWithFields(const NodePtr& lhsOp, const NodePtr& rhsOp)
    {
        using NodeType = typename NodePtr::element_type;
        auto descriptors = NodeType::GetFieldDescriptors();

        return std::apply(
            [&](auto&&... descs) {
                return reflection::FieldIterator<NodeType, StructuralEqualImpl<AssertMode>, decltype(descs)...>::Visit(
                    *lhsOp, *rhsOp, *this, descs...);
            },
            descriptors);
    }

    struct IRAssertContext {
        IRNodePtr lhs{};
        IRNodePtr rhs{};
        std::string lhsDesc{};
        std::string rhsDesc{};
    };

    template <typename... ReasonArgs>
    ResultType IRAssert(const IRAssertContext& context, ReasonArgs&&... reasonArgs)
    {
        if constexpr (AssertMode) {
            std::ostringstream reason;
            ((reason << std::forward<ReasonArgs>(reasonArgs)), ...);

            std::ostringstream msg;
            msg << "Structural equality assertion failed";

            if (!path_.empty()) {
                msg << " at: ";
                for (size_t i = 0; i < path_.size(); ++i) {
                    msg << path_[i];
                    if (i < path_.size() - 1 && path_[i + 1][0] != '[') {
                        msg << ".";
                    }
                }
            }

            if (!nodeTypeStack_.empty()) {
                msg << "\nNode stack: ";
                for (size_t i = 0; i < nodeTypeStack_.size(); ++i) {
                    if (i > 0) {
                        msg << " > ";
                    }
                    msg << nodeTypeStack_[i];
                }
            }
            msg << "\n\n";

            if (context.lhs || context.rhs) {
                msg << "Left-hand side:\n";
                if (context.lhs) {
                    std::string lhsStr = PythonPrint(context.lhs, "pl");
                    std::istringstream iss(lhsStr);
                    std::string line;
                    while (std::getline(iss, line)) {
                        msg << "  " << line << "\n";
                    }
                } else {
                    msg << "  (null)\n";
                }

                msg << "\nRight-hand side:\n";
                if (context.rhs) {
                    std::string rhsStr = PythonPrint(context.rhs, "pl");
                    std::istringstream iss(rhsStr);
                    std::string line;
                    while (std::getline(iss, line)) {
                        msg << "  " << line << "\n";
                    }
                } else {
                    msg << "  (null)\n";
                }
                msg << "\n";
            } else if (!context.lhsDesc.empty() || !context.rhsDesc.empty()) {
                msg << "Left: " << context.lhsDesc << "\n";
                msg << "Right: " << context.rhsDesc << "\n\n";
            }

            msg << "Reason: " << reason.str();
            FE_ASSERT(npu::tile_fwk::FeError::INVALID_VAL, false) << msg.str();
        }
        return false;
    }

    bool enableAutoMapping_;
    std::unordered_map<VarPtr, VarPtr> lhsToRhsVarMap_;
    std::unordered_map<VarPtr, VarPtr> rhsToLhsVarMap_;
    std::vector<std::string> path_; // Only used in assert mode
    std::vector<std::string> fieldNameStack_;
    std::vector<std::string> nodeTypeStack_;
    int transparentDepth_ = 0;
};

// Type dispatch macro for generic field-based comparison.
// Saves and resets transparentDepth_ to 0 before entering EqualWithFields so that
// field names of this (non-transparent) node are always pushed into the path.
#define EQUAL_DISPATCH(Type)                                                 \
    if (auto lhs_##Type = As<Type>(lhs)) {                                   \
        auto rhs_##Type = As<Type>(rhs);                                     \
        nodeTypeStack_.emplace_back(#Type);                                  \
        int savedDepth = transparentDepth_;                                  \
        transparentDepth_ = 0;                                               \
        bool result = rhs_##Type && EqualWithFields(lhs_##Type, rhs_##Type); \
        transparentDepth_ = savedDepth;                                      \
        nodeTypeStack_.pop_back();                                           \
        return result;                                                       \
    }

// Dispatch macro for transparent container nodes (Program, SeqStmts).
// Increments transparentDepth_ so that their field names are suppressed in the path.
#define EQUAL_DISPATCH_TRANSPARENT(Type)                                     \
    if (auto lhs_##Type = As<Type>(lhs)) {                                   \
        transparentDepth_++;                                                 \
        auto rhs_##Type = As<Type>(rhs);                                     \
        nodeTypeStack_.emplace_back(#Type);                                  \
        bool result = rhs_##Type && EqualWithFields(lhs_##Type, rhs_##Type); \
        nodeTypeStack_.pop_back();                                           \
        transparentDepth_--;                                                 \
        return result;                                                       \
    }

template <bool AssertMode>
bool StructuralEqualImpl<AssertMode>::Equal(const IRNodePtr& lhs, const IRNodePtr& rhs)
{
    if (lhs.get() == rhs.get())
        return true;

    if (!lhs || !rhs) {
        return IRAssert({lhs, rhs}, "One node is null, the other is not");
    }

    if (lhs->TypeName() != rhs->TypeName()) {
        return IRAssert({lhs, rhs}, "Node type mismatch (", lhs->TypeName(), " != ", rhs->TypeName(), ")");
    }

    // Check MemRef before Var (MemRef inherits from Var)
    if (auto lhsMemref = As<MemRef>(lhs)) {
        nodeTypeStack_.emplace_back("MemRef");
        auto rhsMemref = std::static_pointer_cast<const MemRef>(rhs);
        bool result = rhsMemref && EqualMemRef(lhsMemref, rhsMemref);
        nodeTypeStack_.pop_back();
        return result;
    }

    if (auto lhsVar = As<Var>(lhs)) {
        nodeTypeStack_.emplace_back("Var");
        bool result = EqualVar(lhsVar, std::static_pointer_cast<const Var>(rhs));
        nodeTypeStack_.pop_back();
        return result;
    }

    // All other types use generic field-based comparison
    EQUAL_DISPATCH(ConstInt)
    EQUAL_DISPATCH(ConstFloat)
    EQUAL_DISPATCH(ConstBool)
    EQUAL_DISPATCH(Call)
    EQUAL_DISPATCH(MakeTuple)
    EQUAL_DISPATCH(GetItemExpr)

    // BinaryExpr and UnaryExpr are abstract base classes
    EQUAL_DISPATCH(BinaryExpr)
    EQUAL_DISPATCH(UnaryExpr)

    EQUAL_DISPATCH(AssignStmt)
    EQUAL_DISPATCH(IfStmt)
    EQUAL_DISPATCH(YieldStmt)
    EQUAL_DISPATCH(ReturnStmt)
    EQUAL_DISPATCH(ForStmt)
    EQUAL_DISPATCH(WhileStmt)
    EQUAL_DISPATCH(SectionStmt)
    EQUAL_DISPATCH_TRANSPARENT(SeqStmts)
    EQUAL_DISPATCH(EvalStmt)
    EQUAL_DISPATCH(BreakStmt)
    EQUAL_DISPATCH(ContinueStmt)
    EQUAL_DISPATCH(Function)
    EQUAL_DISPATCH_TRANSPARENT(Program)

    std::string typeErrMsg = "Unknown IR node type in StructuralEqualImpl::Equal: " + lhs->TypeName();
    FE_ASSERT(npu::tile_fwk::FeError::INVALID_TYPE, false) << typeErrMsg;
    return false;
}

#undef EQUAL_DISPATCH
#undef EQUAL_DISPATCH_TRANSPARENT

template <bool AssertMode>
bool StructuralEqualImpl<AssertMode>::EqualType(const TypePtr& lhs, const TypePtr& rhs)
{
    if (lhs->TypeName() != rhs->TypeName()) {
        return IRAssert({}, "Type name mismatch (", lhs->TypeName(), " != ", rhs->TypeName(), ")");
    }

    if (auto lhsScalar = As<ScalarType>(lhs)) {
        auto rhsScalar = As<ScalarType>(rhs);
        if (!rhsScalar) {
            return IRAssert({}, "Type cast failed for ScalarType");
        }
        if ((IsLoopVarFieldContext() || IsConstIntTypeContext()) &&
            AreForSyntaxScalarDtypesEquivalent(lhsScalar->dtype_, rhsScalar->dtype_)) {
            return true;
        }
        if (lhsScalar->dtype_ != rhsScalar->dtype_) {
            return IRAssert({}, "ScalarType dtype mismatch (", lhsScalar->dtype_.ToString(),
                            " != ", rhsScalar->dtype_.ToString(), ")");
        }
        return true;
    } else if (auto lhsReg = As<RegType>(lhs)) {
        auto rhsReg = As<RegType>(rhs);
        if (!rhsReg) {
            return IRAssert({}, "Type cast failed for RegType");
        }
        if (lhsReg->regtype_ != rhsReg->regtype_) {
            return IRAssert({}, "RegType regtype mismatch (", static_cast<unsigned>(lhsReg->regtype_),
                            " != ", static_cast<unsigned>(rhsReg->regtype_), ")");
        }
        if (lhsReg->regtype_ != RegKind::PredReg && lhsReg->dtype_ != rhsReg->dtype_) {
            return IRAssert({}, "RegType dtype mismatch (", lhsReg->dtype_.ToString(),
                            " != ", rhsReg->dtype_.ToString(), ")");
        }
        return true;
    } else if (auto lhsTensor = As<TensorType>(lhs)) {
        auto rhsTensor = As<TensorType>(rhs);
        if (!rhsTensor) {
            return IRAssert({}, "Type cast failed for TensorType");
        }
        if (lhsTensor->dtype_ != rhsTensor->dtype_) {
            return IRAssert({}, "TensorType dtype mismatch (", lhsTensor->dtype_.ToString(),
                            " != ", rhsTensor->dtype_.ToString(), ")");
        }
        if (lhsTensor->shape_.size() != rhsTensor->shape_.size()) {
            return IRAssert({}, "TensorType shape rank mismatch (", lhsTensor->shape_.size(),
                            " != ", rhsTensor->shape_.size(), ")");
        }
        for (size_t i = 0; i < lhsTensor->shape_.size(); ++i) {
            if (!Equal(lhsTensor->shape_[i], rhsTensor->shape_[i]))
                return false;
        }
        return true;
    } else if (auto lhsTile = As<TileType>(lhs)) {
        if (auto lhsMutexTile = As<MutexTileType>(lhs)) {
            auto rhsMutexTile = As<MutexTileType>(rhs);
            if (!rhsMutexTile) {
                return IRAssert({}, "Type cast failed for MutexTileType");
            }
            if (lhsMutexTile->mutexCount_ != rhsMutexTile->mutexCount_) {
                return IRAssert({}, "MutexTileType mutex count mismatch");
            }
        }
        auto rhsTile = As<TileType>(rhs);
        if (!rhsTile) {
            return IRAssert({}, "Type cast failed for TileType");
        }
        // Compare dtype
        if (lhsTile->dtype_ != rhsTile->dtype_) {
            return IRAssert({}, "TileType dtype mismatch (", lhsTile->dtype_.ToString(),
                            " != ", rhsTile->dtype_.ToString(), ")");
        }
        // Compare shape size and dimensions
        if (lhsTile->shape_.size() != rhsTile->shape_.size()) {
            return IRAssert({}, "TileType shape rank mismatch (", lhsTile->shape_.size(),
                            " != ", rhsTile->shape_.size(), ")");
        }
        for (size_t i = 0; i < lhsTile->shape_.size(); ++i) {
            if (!Equal(lhsTile->shape_[i], rhsTile->shape_[i]))
                return false;
        }
        // Buffer kind is part of a Tile's C++ type; address and size belong to the allocation.
        const auto& lhsMemref = lhsTile->memref_;
        const auto& rhsMemref = rhsTile->memref_;
        const bool lhsHasMemref = lhsMemref && *lhsMemref;
        const bool rhsHasMemref = rhsMemref && *rhsMemref;
        if (lhsHasMemref != rhsHasMemref) {
            return IRAssert({}, "TileType memory reference presence mismatch");
        }
        if (lhsHasMemref && (*lhsMemref)->memorySpace_ != (*rhsMemref)->memorySpace_) {
            return IRAssert({}, "TileType memory space mismatch (", MemorySpaceToString((*lhsMemref)->memorySpace_),
                            " != ", MemorySpaceToString((*rhsMemref)->memorySpace_), ")");
        }
        // Compare tile_view
        if (lhsTile->tileView_.has_value() != rhsTile->tileView_.has_value()) {
            return IRAssert({}, "TileType tile_view presence mismatch");
        }
        if (lhsTile->tileView_.has_value()) {
            const auto& lhsTv = lhsTile->tileView_.value();
            const auto& rhsTv = rhsTile->tileView_.value();
            // Compare valid_shape
            if (lhsTv.validShape.size() != rhsTv.validShape.size()) {
                return IRAssert({}, "TileView valid_shape size mismatch (", lhsTv.validShape.size(),
                                " != ", rhsTv.validShape.size(), ")");
            }
            for (size_t i = 0; i < lhsTv.validShape.size(); ++i) {
                if (!Equal(lhsTv.validShape[i], rhsTv.validShape[i]))
                    return false;
            }
        }
        // Compare hardware_info
        if (lhsTile->hardwareInfo_.has_value() != rhsTile->hardwareInfo_.has_value()) {
            return IRAssert({}, "TileType hardware_info presence mismatch");
        }
        if (lhsTile->hardwareInfo_.has_value()) {
            const auto& lhsHw = lhsTile->hardwareInfo_.value();
            const auto& rhsHw = rhsTile->hardwareInfo_.value();
            if (lhsHw.blayout != rhsHw.blayout) {
                return IRAssert({}, "HardwareInfo blayout mismatch");
            }
            if (lhsHw.slayout != rhsHw.slayout) {
                return IRAssert({}, "HardwareInfo slayout mismatch");
            }
            if (lhsHw.fractal != rhsHw.fractal) {
                return IRAssert({}, "HardwareInfo fractal mismatch");
            }
            if (lhsHw.pad != rhsHw.pad) {
                return IRAssert({}, "HardwareInfo pad mismatch");
            }
            if (lhsHw.compact != rhsHw.compact) {
                return IRAssert({}, "HardwareInfo compact mismatch");
            }
            if (lhsHw.convLayout != rhsHw.convLayout) {
                return IRAssert({}, "HardwareInfo convLayout mismatch");
            }
        }
        return true;
    } else if (auto lhsPtr = As<PtrType>(lhs)) {
        auto rhsPtr = As<PtrType>(rhs);
        if (!rhsPtr) {
            return IRAssert({}, "Type cast failed for PtrType");
        }
        if (lhsPtr->dtype_ != rhsPtr->dtype_) {
            return IRAssert({}, "PtrType dtype mismatch (", lhsPtr->dtype_.ToString(),
                            " != ", rhsPtr->dtype_.ToString(), ")");
        }
        return true;
    } else if (auto lhsTuple = As<TupleType>(lhs)) {
        auto rhsTuple = As<TupleType>(rhs);
        if (!rhsTuple) {
            return IRAssert({}, "Type cast failed for TupleType");
        }
        if (lhsTuple->types_.size() != rhsTuple->types_.size()) {
            return IRAssert({}, "TupleType size mismatch (", lhsTuple->types_.size(), " != ", rhsTuple->types_.size(),
                            ")");
        }
        for (size_t i = 0; i < lhsTuple->types_.size(); ++i) {
            if (!EqualType(lhsTuple->types_[i], rhsTuple->types_[i]))
                return false;
        }
        return true;
    } else if (auto lhsToken = As<TokenType>(lhs)) {
        auto rhsToken = As<TokenType>(rhs);
        if (!rhsToken) {
            return IRAssert({}, "Type cast failed for TokenType");
        }
        if (lhsToken->kind_ != rhsToken->kind_) {
            return IRAssert({}, "TokenType kind mismatch");
        }
        return true;
    } else if (IsFieldLessType(lhs)) {
        return true; // Singleton type, both being same type kind is sufficient
    } else if (IsA<LogicalTensorType>(lhs) && IsA<LogicalTensorType>(rhs)) {
        return true;
    }

    INTERNAL_UNREACHABLE << "EqualType encountered unhandled Type: " << lhs->TypeName();
    return false;
}

template <bool AssertMode>
bool StructuralEqualImpl<AssertMode>::EqualVar(const VarPtr& lhs, const VarPtr& rhs)
{
    if (!enableAutoMapping_) {
        auto lhsIt = lhsToRhsVarMap_.find(lhs);
        auto rhsIt = rhsToLhsVarMap_.find(rhs);
        // Case 1: already mapped to the same variable
        if (lhsIt != lhsToRhsVarMap_.end() && rhsIt != rhsToLhsVarMap_.end()) {
            if (lhsIt->second != rhs || rhsIt->second != lhs) {
                return IRAssert({std::static_pointer_cast<const IRNode>(lhs),
                                 std::static_pointer_cast<const IRNode>(rhs), "var " + lhs->name_, "var " + rhs->name_},
                                "Variable mapping inconsistent (without auto-mapping)");
            }
            return true;
        }
        // Case 2: different variables
        if (lhs.get() != rhs.get()) {
            return IRAssert({std::static_pointer_cast<const IRNode>(lhs), std::static_pointer_cast<const IRNode>(rhs),
                             "var " + lhs->name_, "var " + rhs->name_},
                            "Variable pointer mismatch (without auto-mapping)");
        }
        return true;
    }

    if (!EqualType(lhs->GetType(), rhs->GetType())) {
        return IRAssert({}, "Variable type mismatch (", lhs->GetType()->TypeName(), " != ", rhs->GetType()->TypeName(),
                        ")");
    }

    auto it = lhsToRhsVarMap_.find(lhs);
    if (it != lhsToRhsVarMap_.end()) {
        if (it->second != rhs) {
            return IRAssert({std::static_pointer_cast<const IRNode>(lhs), std::static_pointer_cast<const IRNode>(rhs)},
                            "Variable mapping inconsistent ('", lhs->name_, "' cannot map to both '", it->second->name_,
                            "' and '", rhs->name_, "')");
        }
        return true;
    }

    auto rhsIt = rhsToLhsVarMap_.find(rhs);
    if (rhsIt != rhsToLhsVarMap_.end() && rhsIt->second != lhs) {
        return IRAssert({std::static_pointer_cast<const IRNode>(lhs), std::static_pointer_cast<const IRNode>(rhs)},
                        "Variable mapping inconsistent ('", rhs->name_, "' is already mapped from '",
                        rhsIt->second->name_, "', cannot map from '", lhs->name_, "')");
    }

    lhsToRhsVarMap_[lhs] = rhs;
    rhsToLhsVarMap_[rhs] = lhs;
    return true;
}

template <bool AssertMode>
bool StructuralEqualImpl<AssertMode>::EqualMemRef(const MemRefPtr& lhs, const MemRefPtr& rhs)
{
    if (!MemRef::SameAllocation(lhs, rhs)) {
        return IRAssert({std::static_pointer_cast<const IRNode>(lhs), std::static_pointer_cast<const IRNode>(rhs)},
                        "MemRef base mismatch");
    }
    return true;
}

template <bool AssertMode>
bool StructuralEqualImpl<AssertMode>::EqualIterArg(const IterArgPtr& lhs, const IterArgPtr& rhs)
{
    // 1. Compare the variable (handles variable mapping)
    if (!EqualVar(lhs->iterVar_, rhs->iterVar_)) {
        return false;
    }

    // 2. Compare the initial value
    if (!Equal(lhs->initValue_, rhs->initValue_)) {
        return IRAssert({std::static_pointer_cast<const IRNode>(lhs->initValue_),
                         std::static_pointer_cast<const IRNode>(rhs->initValue_)},
                        "IterArg initValue mismatch");
    }

    return true;
}

// Explicit template instantiations
template class StructuralEqualImpl<false>; // For structural_equal
template class StructuralEqualImpl<true>;  // For assert_structural_equal

// Type aliases for cleaner code
using StructuralEqual = StructuralEqualImpl<false>;
using StructuralEqualAssert = StructuralEqualImpl<true>;

// Public API implementation
bool structural_equal(const IRNodePtr& lhs, const IRNodePtr& rhs, bool enableAutoMapping)
{
    StructuralEqual checker(enableAutoMapping);
    return checker(lhs, rhs);
}

bool structural_equal(const TypePtr& lhs, const TypePtr& rhs, bool enableAutoMapping)
{
    StructuralEqual checker(enableAutoMapping);
    return checker(lhs, rhs);
}

// Public assert API
void assert_structural_equal(const IRNodePtr& lhs, const IRNodePtr& rhs, bool enableAutoMapping)
{
    StructuralEqualAssert checker(enableAutoMapping);
    checker(lhs, rhs);
}

void assert_structural_equal(const TypePtr& lhs, const TypePtr& rhs, bool enableAutoMapping)
{
    StructuralEqualAssert checker(enableAutoMapping);
    checker(lhs, rhs);
}

} // namespace ir
} // namespace pypto
