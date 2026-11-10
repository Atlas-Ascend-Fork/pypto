/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/any_cast.h"
#include "core/dtype.h"
#include "core/error.h"
#include "core/logging.h"
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
#include "ir/type.h"

namespace pypto {
namespace ir {

namespace {

DataType CanonicalizeForSyntaxScalarDtype(const DataType& dtype)
{
    if (dtype == DataType::INT64 || dtype == DataType::INDEX) {
        return DataType::INDEX;
    }
    return dtype;
}

} // namespace

/**
 * \brief Hash combine using Boost-inspired algorithm
 */
inline uint64_t HashCombine(uint64_t seed, uint64_t value)
{
    return seed ^ (value + 0x9e3779b9 + (seed << 0x6) + (seed >> 0x2));
}

/**
 * \brief Structural hasher for IR nodes
 *
 * Computes hash based on IR node tree structure, ignoring Span (source location).
 * Also serves as a FieldVisitor for the reflection-based field iteration.
 */
class StructuralHasher {
public:
    using ResultType = uint64_t;

    explicit StructuralHasher(bool enableAutoMapping) : enableAutoMapping_(enableAutoMapping) {}

    ResultType operator()(const IRNodePtr& node) { return HashNode(node); }

    ResultType operator()(const TypePtr& type) { return HashType(type); }

    // FieldVisitor interface methods
    [[nodiscard]] ResultType InitResult() const { return 0; }

    template <typename IRNodePtrType>
    ResultType VisitIRNodeField(const IRNodePtrType& field)
    {
        INTERNAL_CHECK(field) << "structural_hash encountered null IR node field";
        return HashNode(field);
    }

    // Specialization for std::optional<IRNodePtr>
    template <typename IRNodePtrType>
    ResultType VisitIRNodeField(const std::optional<IRNodePtrType>& field)
    {
        if (field.has_value() && *field) {
            return HashNode(*field);
        } else {
            // Hash empty optional as 0
            return 0;
        }
    }

    template <typename IRNodePtrType>
    ResultType VisitIRNodeVectorField(const std::vector<IRNodePtrType>& fields)
    {
        ResultType h = 0;
        for (size_t i = 0; i < fields.size(); ++i) {
            INTERNAL_CHECK(fields[i]) << "structural_hash encountered null IR node in vector at index " << i;
            h = HashCombine(h, HashNode(fields[i]));
        }
        return h;
    }

    template <typename KeyType, typename ValueType, typename Compare>
    ResultType VisitIRNodeMapField(const std::map<KeyType, ValueType, Compare>& field)
    {
        ResultType h = 0;
        for (const auto& [key, value] : field) {
            INTERNAL_CHECK(key) << "structural_hash encountered null key in map";
            INTERNAL_CHECK(value) << "structural_hash encountered null value in map";
            // Hash key by name (keys are Op types, not IRNode)
            h = HashCombine(h, static_cast<ResultType>(std::hash<std::string>{}(key->name_)));
            // Hash value (values are IRNode types)
            h = HashCombine(h, HashNode(value));
        }
        return h;
    }

    template <typename ValueType>
    ResultType VisitIRNodeMapField(const std::map<std::string, ValueType>& field)
    {
        ResultType h = 0;
        for (const auto& [key, value] : field) {
            INTERNAL_CHECK(value) << "structural_hash encountered null value in map";
            // Hash key string
            h = HashCombine(h, static_cast<ResultType>(std::hash<std::string>{}(key)));
            // Hash value (values are IRNode types)
            h = HashCombine(h, HashNode(value));
        }
        return h;
    }

    template <typename FVisitOp>
    void VisitIgnoreField([[maybe_unused]] FVisitOp&& visitOp)
    {
        // Ignore field, do nothing
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

    void PushFieldName(const char* name)
    {
        if (transparentDepth_ == 0) {
            fieldNameStack_.emplace_back(name);
        }
    }
    void PopFieldName()
    {
        if (transparentDepth_ == 0) {
            fieldNameStack_.pop_back();
        }
    }

    ResultType VisitLeafField(const int& field) { return static_cast<ResultType>(std::hash<int>{}(field)); }

    ResultType VisitLeafField(const int64_t& field) { return static_cast<ResultType>(std::hash<int64_t>{}(field)); }

    ResultType VisitLeafField(const uint64_t& field) { return static_cast<ResultType>(std::hash<uint64_t>{}(field)); }

    ResultType VisitLeafField(const double& field) { return static_cast<ResultType>(std::hash<double>{}(field)); }

    ResultType VisitLeafField(const std::string& field)
    {
        return static_cast<ResultType>(std::hash<std::string>{}(field));
    }

    ResultType VisitLeafField(const DataType& field)
    {
        return static_cast<ResultType>(std::hash<uint8_t>{}(field.Code()));
    }

    ResultType VisitLeafField(const FunctionType& field)
    {
        return static_cast<ResultType>(std::hash<uint8_t>{}(static_cast<uint8_t>(field)));
    }

    ResultType VisitLeafField(const SectionKind& field)
    {
        return static_cast<ResultType>(std::hash<uint8_t>{}(static_cast<uint8_t>(field)));
    }

    ResultType VisitLeafField(const MemorySpace& field)
    {
        return static_cast<ResultType>(std::hash<int>{}(static_cast<int>(field)));
    }

    ResultType VisitLeafField(const TypePtr& field)
    {
        INTERNAL_CHECK(field) << "structural_hash encountered null TypePtr field";
        return HashType(field);
    }

    ResultType VisitLeafField(const std::vector<TypePtr>& fields)
    {
        ResultType h = 0;
        for (size_t i = 0; i < fields.size(); ++i) {
            INTERNAL_CHECK(fields[i]) << "structural_hash encountered null TypePtr in vector at index " << i;
            h = HashCombine(h, HashType(fields[i]));
        }
        return h;
    }

    ResultType VisitLeafField(const std::vector<std::string>& fields)
    {
        ResultType h = static_cast<ResultType>(fields.size());
        for (const auto& s : fields) {
            h = HashCombine(h, std::hash<std::string>{}(s));
        }
        return h;
    }

    // Hash kwargs (vector of pairs - order is preserved and matters)
    ResultType VisitLeafField(const std::vector<std::pair<std::string, std::any>>& kwargs)
    {
        ResultType h = 0;
        // Hash keys and values in order (no need to sort since order is preserved)
        for (const auto& [key, value] : kwargs) {
            h = HashCombine(h, std::hash<std::string>{}(key));

            // Hash value based on type
            if (value.type() == typeid(int)) {
                h = HashCombine(h, std::hash<int>{}(AnyCast<int>(value, "hashing kwarg: " + key)));
            } else if (value.type() == typeid(bool)) {
                h = HashCombine(h, std::hash<bool>{}(AnyCast<bool>(value, "hashing kwarg: " + key)));
            } else if (value.type() == typeid(std::string)) {
                h = HashCombine(h, std::hash<std::string>{}(AnyCast<std::string>(value, "hashing kwarg: " + key)));
            } else if (value.type() == typeid(double)) {
                h = HashCombine(h, std::hash<double>{}(AnyCast<double>(value, "hashing kwarg: " + key)));
            } else if (value.type() == typeid(float)) {
                h = HashCombine(h, std::hash<float>{}(AnyCast<float>(value, "hashing kwarg: " + key)));
            } else if (value.type() == typeid(DataType)) {
                h = HashCombine(h, std::hash<uint8_t>{}(AnyCast<DataType>(value, "hashing kwarg: " + key).Code()));
            } else if (value.type() == typeid(std::vector<int>)) {
                const auto& vec = AnyCast<std::vector<int>>(value, "hashing kwarg: " + key);
                for (int v : vec) {
                    h = HashCombine(h, std::hash<int>{}(v));
                }
            } else if (value.type() == typeid(std::vector<std::string>)) {
                const auto& vec = AnyCast<std::vector<std::string>>(value, "hashing kwarg: " + key);
                for (const auto& s : vec) {
                    h = HashCombine(h, std::hash<std::string>{}(s));
                }
            } else {
                CHECK(false) << "Invalid kwarg type for key: " << key
                             << ", expected int, bool, std::string, double, float, DataType, "
                                "or std::vector<int>, but got "
                             << DemangleTypeName(value.type().name());
            }
        }
        return h;
    }

    ResultType VisitLeafField(const Span& field)
    {
        INTERNAL_CHECK_SPAN(false, field) << "structural_hash should not visit Span field";
        return 0;
    }

    ResultType VisitLeafField(const std::vector<IterArgPtr>& fields)
    {
        ResultType h = 0;
        for (const auto& ia : fields) {
            INTERNAL_CHECK(ia) << "structural_hash encountered null IterArgPtr in vector";
            h = HashCombine(h, HashNode(std::static_pointer_cast<const IRNode>(ia->iterVar_)));
            h = HashCombine(h, HashNode(std::static_pointer_cast<const IRNode>(ia->initValue_)));
        }
        return h;
    }

    template <typename Desc>
    void CombineResult(ResultType& accumulator, ResultType fieldHash, [[maybe_unused]] const Desc& descriptor)
    {
        accumulator = HashCombine(accumulator, fieldHash);
    }

private:
    ResultType HashNode(const IRNodePtr& node);
    ResultType HashType(const TypePtr& type);
    bool IsLoopVarFieldContext() const { return !fieldNameStack_.empty() && fieldNameStack_.back() == "loop_var"; }
    bool IsConstIntTypeContext() const
    {
        return !nodeTypeStack_.empty() && nodeTypeStack_.back() == "ConstInt" && !fieldNameStack_.empty() &&
               fieldNameStack_.back() == "type";
    }

    template <typename NodePtr>
    ResultType HashNodeImpl(const NodePtr& node);

    bool enableAutoMapping_;
    std::unordered_map<IRNodePtr, ResultType> hashValueMap_;
    int64_t freeVarCounter_ = 0;
    std::vector<std::string> fieldNameStack_;
    std::vector<std::string> nodeTypeStack_;
    int transparentDepth_ = 0;
};

template <typename NodePtr>
StructuralHasher::ResultType StructuralHasher::HashNodeImpl(const NodePtr& node)
{
    using NodeType = typename NodePtr::element_type;

    // Start with type discriminator
    ResultType h = static_cast<ResultType>(std::hash<std::string>{}(node->TypeName()));
    nodeTypeStack_.emplace_back(node->TypeName());

    // Mirror EQUAL_DISPATCH / EQUAL_DISPATCH_TRANSPARENT from structural_equal.cpp:
    // - Transparent containers (Program, SeqStmts) suppress their own field names.
    // - Non-transparent nodes reset transparentDepth_ to 0 so their fields are always tracked.
    constexpr bool isTransparent = std::is_same_v<NodeType, Program> || std::is_same_v<NodeType, SeqStmts>;
    int savedDepth = transparentDepth_;
    if constexpr (isTransparent) {
        transparentDepth_++;
    } else {
        transparentDepth_ = 0;
    }

    // Visit all fields using reflection
    auto descriptors = NodeType::GetFieldDescriptors();

    ResultType fieldsHash = std::apply(
        [&](auto&&... descs) {
            return reflection::FieldIterator<NodeType, StructuralHasher, decltype(descs)...>::Visit(*node, *this,
                                                                                                    descs...);
        },
        descriptors);

    transparentDepth_ = savedDepth;
    nodeTypeStack_.pop_back();

    return HashCombine(h, fieldsHash);
}

StructuralHasher::ResultType StructuralHasher::HashType(const TypePtr& type)
{
    INTERNAL_CHECK(type) << "structural_hash encountered null TypePtr";
    ResultType h = static_cast<ResultType>(std::hash<std::string>{}(type->TypeName()));
    if (auto scalarType = As<ScalarType>(type)) {
        DataType dtype = scalarType->dtype_;
        if (IsLoopVarFieldContext() || IsConstIntTypeContext()) {
            dtype = CanonicalizeForSyntaxScalarDtype(dtype);
        }
        h = HashCombine(h, static_cast<ResultType>(std::hash<uint8_t>{}(dtype.Code())));
    } else if (auto regType = As<RegType>(type)) {
        h = HashCombine(h, static_cast<ResultType>(std::hash<uint8_t>{}(static_cast<uint8_t>(regType->regtype_))));
        if (regType->regtype_ != RegKind::PredReg) {
            h = HashCombine(h, static_cast<ResultType>(std::hash<uint8_t>{}(regType->dtype_.Code())));
        }
    } else if (auto tensorType = As<TensorType>(type)) {
        h = HashCombine(h, static_cast<ResultType>(std::hash<uint8_t>{}(tensorType->dtype_.Code())));
        h = HashCombine(h, static_cast<ResultType>(tensorType->shape_.size()));
        for (const auto& dim : tensorType->shape_) {
            INTERNAL_CHECK(dim) << "structural_hash encountered null shape dimension in TypePtr";
            h = HashCombine(h, HashNode(dim));
        }
    } else if (auto tileType = As<TileType>(type)) {
        if (auto mutexTile = As<MutexTileType>(type)) {
            h = HashCombine(h, static_cast<ResultType>(mutexTile->mutexCount_));
        }
        // Hash dtype
        h = HashCombine(h, static_cast<ResultType>(std::hash<uint8_t>{}(tileType->dtype_.Code())));
        // Hash shape size and dimensions
        h = HashCombine(h, static_cast<ResultType>(tileType->shape_.size()));
        for (const auto& dim : tileType->shape_) {
            INTERNAL_CHECK(dim) << "structural_hash encountered null shape dimension in TileType";
            h = HashCombine(h, HashNode(dim));
        }
        // Hash tile_view if present
        if (tileType->tileView_.has_value()) {
            const auto& tv = tileType->tileView_.value();
            h = HashCombine(h, static_cast<ResultType>(1)); // indicate presence
            h = HashCombine(h, static_cast<ResultType>(tv.validShape.size()));
            for (const auto& dim : tv.validShape) {
                INTERNAL_CHECK(dim) << "structural_hash encountered null valid_shape dimension in TileView";
                h = HashCombine(h, HashNode(dim));
            }
        } else {
            h = HashCombine(h, static_cast<ResultType>(0)); // indicate absence
        }
        // Hash hardware_info if present
        if (tileType->hardwareInfo_.has_value()) {
            const auto& hw = tileType->hardwareInfo_.value();
            h = HashCombine(h, static_cast<ResultType>(1));
            h = HashCombine(h, static_cast<ResultType>(hw.blayout));
            h = HashCombine(h, static_cast<ResultType>(hw.slayout));
            h = HashCombine(h, static_cast<ResultType>(hw.fractal));
            h = HashCombine(h, static_cast<ResultType>(hw.pad));
            h = HashCombine(h, static_cast<ResultType>(hw.compact));
            h = HashCombine(h, static_cast<ResultType>(hw.convLayout));
        } else {
            h = HashCombine(h, static_cast<ResultType>(0));
        }
    } else if (auto ptrType = As<PtrType>(type)) {
        h = HashCombine(h, static_cast<ResultType>(std::hash<uint8_t>{}(ptrType->dtype_.Code())));
    } else if (auto tupleType = As<TupleType>(type)) {
        h = HashCombine(h, static_cast<ResultType>(tupleType->types_.size()));
        for (const auto& t : tupleType->types_) {
            INTERNAL_CHECK(t) << "structural_hash encountered null type in TupleType";
            h = HashCombine(h, HashType(t));
        }
    } else if (auto tokenType = As<TokenType>(type)) {
        h = HashCombine(h, static_cast<ResultType>(tokenType->kind_));
    } else if (IsFieldLessType(type)) {
        // Field-less singletons have no fields; only the type name is hashed (already done above)
    } else {
        INTERNAL_CHECK(false) << "HashType encountered unhandled Type: " << type->TypeName();
    }
    return h;
}

// Type dispatch macro
#define HASH_DISPATCH(Type)                                                                                            \
    if (auto p = As<Type>(node)) {                                                                                     \
        INTERNAL_CHECK_SPAN(dispatched == false, node->span_) << "HashNodeImpl already dispatched for type " << #Type; \
        hashValue = HashNodeImpl(p);                                                                                   \
        dispatched = true;                                                                                             \
    }

StructuralHasher::ResultType StructuralHasher::HashNode(const IRNodePtr& node)
{
    INTERNAL_CHECK(node) << "structural_hash received null IR node";

    auto it = hashValueMap_.find(node);
    if (it != hashValueMap_.end()) {
        return it->second;
    }

    ResultType hashValue = 0;
    bool dispatched = false;

    // MemRef needs special handling: dispatch for fields, then add Var mapping
    HASH_DISPATCH(MemRef)
    // IterArg needs special handling: dispatch for fields, then add Var mapping
    HASH_DISPATCH(Var)
    HASH_DISPATCH(ConstInt)
    HASH_DISPATCH(ConstFloat)
    HASH_DISPATCH(ConstBool)
    HASH_DISPATCH(Call)
    HASH_DISPATCH(MakeTuple)
    HASH_DISPATCH(GetItemExpr)

    // BinaryExpr and UnaryExpr are abstract base classes
    HASH_DISPATCH(BinaryExpr)
    HASH_DISPATCH(UnaryExpr)

    HASH_DISPATCH(AssignStmt)
    HASH_DISPATCH(IfStmt)
    HASH_DISPATCH(YieldStmt)
    HASH_DISPATCH(ReturnStmt)
    HASH_DISPATCH(ForStmt)
    HASH_DISPATCH(WhileStmt)
    HASH_DISPATCH(SectionStmt)
    HASH_DISPATCH(SeqStmts)
    HASH_DISPATCH(EvalStmt)
    HASH_DISPATCH(BreakStmt)
    HASH_DISPATCH(ContinueStmt)
    HASH_DISPATCH(Function)
    HASH_DISPATCH(Program)

    // Free Var types (including MemRef and IterArg) that may be mapped to other free vars.
    // These have already been dispatched above for field hashing;
    // here we add the variable identity hash.
    auto hashVarIdentity = [&](const Var* var) {
        if (enableAutoMapping_) {
            hashValue = HashCombine(hashValue, freeVarCounter_++);
        } else {
            hashValue = HashCombine(hashValue, std::hash<std::string>{}(var->name_));
        }
    };

    auto kind = node->GetKind();
    if (kind == ObjectKind::MemRef || kind == ObjectKind::Var) {
        hashVarIdentity(static_cast<const Var*>(node.get()));
    }

    if (!dispatched) {
        INTERNAL_UNREACHABLE << "Unknown IR node type in StructuralHasher::HashNode";
    }

    hashValueMap_.emplace(node, hashValue);
    return hashValue;
}

#undef HASH_DISPATCH

// Public API
uint64_t structural_hash(const IRNodePtr& node, bool enableAutoMapping)
{
    StructuralHasher hasher(enableAutoMapping);
    return hasher(node);
}

uint64_t structural_hash(const TypePtr& type, bool enableAutoMapping)
{
    StructuralHasher hasher(enableAutoMapping);
    return hasher(type);
}

uint64_t structural_hash_with_var_identity(const IRNodePtr& node, bool enableAutoMapping)
{
    StructuralHasher hasher(enableAutoMapping);
    return hasher(node);
}

uint64_t structural_hash_with_var_identity(const TypePtr& type, bool enableAutoMapping)
{
    StructuralHasher hasher(enableAutoMapping);
    return hasher(type);
}

} // namespace ir
} // namespace pypto
