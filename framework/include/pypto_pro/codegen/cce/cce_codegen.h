/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PYPTO_CODEGEN_CCE_CCE_CODEGEN_H_
#define PYPTO_CODEGEN_CCE_CCE_CODEGEN_H_

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "backend/common/backend.h"
#include "codegen/cce/code_context.h"
#include "codegen/cce/code_emitter.h"
#include "codegen/cce/type_converter.h"
#include "codegen/codegen_base.h"
#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/pipe.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/scalar_expr_ops.h"
#include "ir/stmt.h"
#include "ir/type.h"
#include "ir/transforms/passes.h"
#include "tilefwk/platform.h"

namespace pypto {

namespace codegen {

/// Tensor definition collected from block.load/store access windows.
/// tensor_type and the base pointer are derived from `var` at emit time (not stored):
/// tensor_type = ir::As<ir::TensorType>(var->GetType()); the base pointer is the view's
/// source pointer (ptr.make_tensor) or "<name>_ptr" for a plain tensor parameter.
struct TensorDef {
    ir::VarPtr var;                           ///< tensor variable (parameter or ptr.make_tensor view)
    std::string cceName;                      ///< emitted declaration name: the tensor's plain cce name for the
                                              ///< first layout it is accessed with, "<name>__v<n>" for each later one
    std::vector<ir::ExprPtr> accessShape;     ///< access window shape (tile-derived) from load/store
    std::optional<std::vector<int>> tileDims; ///< tile_dims kwarg for strided views (if any)
    bool isTranspose = false;                 ///< loaded with is_transpose=true (needs Layout::DN)
    bool isMerged = false;                    ///< multi-matrix load with three dynamic shape dimensions
    std::string layout;                       ///< explicit Layout enum (MX loads); else derived from the above
};

/**
 * \brief Key identifying the layout variant one load/store access needs.
 *
 * Layout::DN and the row/col stride order are baked into the GlobalTensor *type*, so accesses
 * that walk one tensor differently cannot share a declaration. The key is a pure function of
 * the access (its is_transpose / tile_dims kwargs and its access shape), so the prescan and
 * the load/store codegen agree on which declaration an access belongs to. Accesses that share
 * a key share a declaration.
 */
[[nodiscard]] std::string TensorLayoutVariantKey(const ir::CallPtr& op);

/// Build a side-effect-free FP32 fmod expression for CCE device code.
[[nodiscard]] std::string BuildFP32FmodExpression(const std::string& lhs, const std::string& rhs);

/// Legalize scalar arithmetic and conditions before Pro CCE emission; preserve SIMT arithmetic.
[[nodiscard]] ir::Pass LowerScalarExpressionsForCCE();

/// Definition of one user-visible C++ class materialized from a TupleType.
struct StructDefinition {
    std::string name;
    std::vector<std::string> fields;
    std::vector<ir::TypePtr> types;
    bool isTiling = false;
    bool requiresVolatile = false;
};

/**
 * \brief CCE code generator for converting PyPTO IR to pto-isa C++ code
 *
 * CCECodegen traverses the IR using the visitor pattern and generates
 * compilable C++ code using pto-isa instructions. It handles:
 * - Function prologue (signature, argument unpacking, type definitions)
 * - Function body (block operations, sync operations, control flow)
 * - Type conversions and memory management
 */
class CCECodegen : public CodegenBase {
protected:
    using CodegenBase::VisitExpr_;
    using CodegenBase::VisitStmt_;

public:
    /// Per-axis element strides of a tensor, innermost first at "1". Public because the
    /// block.load backend needs them to drive a merged-row nd2nz directly.
    std::vector<std::string> BuildTensorStrideExpressions(const ir::TensorTypePtr& tensorType);

    /** \brief Construct a CCE code generator for one fixed Cube/Vector target. */
    explicit CCECodegen(ir::SectionKind target);

    /**
     * \brief Generate a standalone TilingData header from an ordered field definition.
     *
     * This is the same header format emitted during full kernel codegen, but does not
     * require a Program or run any IR passes.
     */
    [[nodiscard]] static std::string GenerateTilingHeader(const std::string& typeName,
                                                          const std::vector<std::string>& fields,
                                                          const std::vector<ir::TypePtr>& types,
                                                          bool requiresVolatile = false);

    /**
     * \brief Generate a single C++ file from a PyPTO IR Program
     *
     * Generates an ordinary kernel device implementation from SSA IR with:
     * - PTO-style function signature
     * - Target-specific function naming and DAV macro guards
     * - constexpr for compile-time constants
     * - FFTS support for cross-core sync
     *
     * SIMT kernels emit a __global__ entry directly.
     *
     * \param program The IR Program to generate code for
     * \return Generated C++ code as a single string
     */
    [[nodiscard]] std::string GenerateSingle(const ir::ProgramPtr& program, npu::tile_fwk::NPUArch arch);

    /**
     * \brief Tiling struct headers generated by the last GenerateSingle call.
     *
     * Maps header filename (e.g. "OpTiling_tiling.h") to its content (a `#pragma once`
     * guarded `struct <ClassName> { ... };`). The kernel.cpp `#include`s these by name;
     * the caller (jit.py) writes them next to kernel.cpp in the build dir.
     */
    [[nodiscard]] const std::map<std::string, std::string>& GetTilingHeaders() const { return tilingHeaders_; }

    // CodegenBase interface (unified API for operator codegen callbacks)
    [[nodiscard]] std::string GetCurrentResultTarget() const override { return currentTargetVar_; }
    void Emit(const std::string& line) override;
    bool needsSimtAtomicHeader_ = false;
    std::string GetExprAsCode(const ir::ExprPtr& expr) override;
    std::string ScalarCast(const std::string& value, const ir::DataType& source, const ir::DataType& target);
    std::string GetGeneratedType(const ir::TypePtr& type) const;

    /// Resolve a tuple-typed expression to the MakeTuple backing it (the tuple counterpart of
    /// GetExprAsCode, which yields a C++ string). Returns null when the expression is not a tuple.
    ir::MakeTuplePtr GetExprAsMakeTuple(const ir::ExprPtr& expr);
    [[nodiscard]] std::string GetTypeString(const ir::DataType& dtype) const override;
    /// True when currently generating code inside a VF section (CCE __VEC_SCOPE__).
    [[nodiscard]] bool IsInVFSection() const { return inVfSection_; }
    /// True when emitting a 3510 SIMT VF, callee, or SIMT kernel body.
    [[nodiscard]] bool IsInSimtContext() const
    {
        return currentFunctionType_ == ir::FunctionType::SIMT_VF ||
               currentFunctionType_ == ir::FunctionType::SIMT_CALLEE ||
               currentFunctionType_ == ir::FunctionType::SIMT_KERNEL;
    }
    [[nodiscard]] ir::FunctionType GetCurrentKernelType() const { return currentKernelType_; }
    int GetTileOffsetCounter() { return tileOffsetCounter_++; }
    /// Resolve a VF tile expr (plain tile or `tile[offset]`) to its `__ubuf__` pointer, typed as the
    /// tile element and hoisted before `__VEC_SCOPE__`. A `tile[offset]` load reuses the rvalue
    /// `(vf_tile_ptr_N + off)`; otherwise a `vf_tile_ptr_N` is declared once and cached. With
    /// `is_post_update`, the access gets its own hoisted variable, kept separate from a plain base
    /// pointer to the same tile (e.g. a POST_UPDATE store cursor that must not alias a base load),
    /// and even a `tile[offset]` materialises a real lvalue var rather than the shared rvalue.
    std::string GetOrCreateVFTilePtr(const ir::ExprPtr& expr, bool isPostUpdate = false);
    /// Mark a VF block.subview result as a pointer expression, including linear offsets.
    std::string RecordVFTileSubviewPointer(std::string code);
    int64_t GetConstIntValue(const ir::ExprPtr& expr) override;
    std::string GetVarName(const ir::VarPtr& var) override;

    const TypeConverter& GetTypeConverter() const { return typeConverter_; }

    /** \brief Get the target architecture */
    npu::tile_fwk::NPUArch GetArch() const { return arch_; }

    bool IsRegisterBasedArch() const
    {
        return arch_ == npu::tile_fwk::NPUArch::DAV_3510 || arch_ == npu::tile_fwk::NPUArch::DAV_9201;
    }

    /** \brief CV ratio feature: A5 has two Vector cores per Cube, A6 has one. */
    int GetVectorCoresPerCube() const { return arch_ == npu::tile_fwk::NPUArch::DAV_9201 ? 1 : 2; }

    /** \brief Get the fixed Cube/Vector target selected for this generation. */
    ir::SectionKind GetTarget() const { return target_; }

    /** \brief Register a VF RegTensor variable (called from EmitVFRegTensor). */
    void RegisterRegTensorVar(const std::string& cppName) { regTensorVars_.insert(cppName); }

    /** \brief Push a RegTensor declaration to the hoisting buffer (emitted at __VEC_SCOPE__ top). */
    void HoistRegTensorDecl(const std::string& decl) { vfRegHoistedDecls_.push_back(decl); }

    /** \brief Check if a C++ variable name was declared as a VF RegTensor. */
    bool IsRegTensorVar(const std::string& cppName) const { return regTensorVars_.count(cppName) > 0; }

    /** \brief Register a VF MaskReg variable (called from EmitVFCreateMask/Compare/etc). */
    void RegisterMaskRegVar(const std::string& cppName) { maskRegVars_.insert(cppName); }

    /** \brief Check if a C++ variable name was declared as a VF MaskReg. */
    bool IsMaskRegVar(const std::string& cppName) const { return maskRegVars_.count(cppName) > 0; }

    /** \brief Register a VF AddrReg variable (called from EmitVFCreateAddrReg). */
    void RegisterAddrRegVar(const std::string& cppName) { addrRegVars_.insert(cppName); }

    /** \brief Check if a C++ variable name was declared as a VF AddrReg. */
    bool IsAddrRegVar(const std::string& cppName) const { return addrRegVars_.count(cppName) > 0; }

    /** \brief Register a VF UnalignReg variable (called from EmitVFUnalignRegForStore/Load). */
    void RegisterUnalignRegVar(const std::string& cppName) { unalignRegVars_.insert(cppName); }

    /** \brief Check if a C++ variable name was declared as a VF UnalignReg. */
    bool IsUnalignRegVar(const std::string& cppName) const { return unalignRegVars_.count(cppName) > 0; }

    /** \brief Map logical tensor coordinates to a layout-aware physical element offset. */
    std::string ComputeTensorOffset(const ir::TensorTypePtr& tensorType, const ir::MakeTuplePtr& offsets);

    /** \brief Render a shape dimension rounded up to an alignment, folding constants when possible. */
    std::string ComputeAlignedShapeDimension(const ir::ExprPtr& dimension, int64_t alignment);

    const ir::Span& GetCurrentStmtSpan() const { return currentStmtSpan_; }
    const ir::Span& GetCurrentExprSpan() const { return currentExprSpan_; }

    /**
     * \brief Register a tensor variable's underlying raw pointer (CCE-specific).
     *
     * Associates a tensor variable (parameter, ptr.make_tensor view, or iter_arg) with its
     * raw pointer expression. Used for GlobalTensor construction and TASSIGN address
     * computation in block.load/store. Same-key re-registration with a different value warns.
     */
    void RegisterPointer(const std::string& tensorVarName, const std::string& ptrName);

    /**
     * \brief Get pointer name for a variable (CCE-specific)
     */
    std::string GetPointer(const std::string& varName);

    /**
     * \brief Check whether a raw pointer mapping has been registered (non-throwing).
     *
     * Distinguishes a tensor parameter / view (pointer already registered) from an
     * IfStmt phi return var (pointer not yet known).
     */
    [[nodiscard]] bool HasPointer(const std::string& varName) const;

    /**
     * \brief Register a struct TupleType definition for deferred class emission.
     */
    void RegisterStructDefinition(const ir::TupleTypePtr& tupleType, const std::string& typeName,
                                  const std::vector<std::string>& fields, bool isTiling);

    /** \brief Get the emitted C++ member type for a struct field. */
    [[nodiscard]] std::string GetStructFieldTypeString(const ir::TypePtr& type) const;

    /**
     * \brief Mark a registered TupleType as requiring volatile member declarations.
     */
    void MarkStructVolatile(const ir::TypePtr& type);

    /**
     * \brief Generate GlobalTensor type declaration and instance for a TensorDef.
     *
     * Emits shape type alias, stride type alias, GlobalTensor type alias, and instance
     * declaration, all named after def.cce_name -- this layout variant's name, as recorded by
     * CollectTensorDefs. The tensor type and the base pointer still come from def.var, which
     * every variant of one tensor shares; dn / tile_dims come from def. Public so ptr.make_tensor
     * codegen can emit a view in place.
     */
    void GenerateGlobalTensorTypeDeclaration(const TensorDef& def);

    /**
     * \brief Emit a fresh [1, numel] ND GlobalTensor declaration over *tensor_var*, in place.
     *
     * For GM-writing ops whose declaration has no consumer other than the op itself (e.g.
     * block.init_output): the declaration is a linearized [1, numel] view -- *numel_expr*
     * is the C++ element-count product of the tensor's shape dims (static dims fold to
     * literals, dynamic dims reference runtime scalars). Returns a unique instance name
     * ("<base>__io<n>"), so repeated calls each get their own declaration with no dedup.
     */
    std::string DeclareFlatGlobalTensor(const ir::VarPtr& tensorVar, const std::string& numelExpr);
    /**
     * \brief Look up every prescanned layout variant of a tensor by its cce variable name.
     *
     * Empty when the name was never accessed by a block.load/store (no declaration needed);
     * one def per layout the accesses need. Populated by CollectTensorDefs during the prologue.
     */
    [[nodiscard]] std::vector<const TensorDef*> GetTensorDefs(const std::string& name) const;

    /**
     * \brief Bind one access to its declaration and return the instance to read or write through.
     *
     * Selects the declaration matching this access's layout -- a tensor also read with a different
     * order has one per layout, and only the matching one walks it correctly -- then points its
     * hoisted instance at the access and resizes it in place. valid_rows/valid_cols are the logical
     * transfer shape; layout-specific physical dimensions are derived while binding.
     * Merged loads supply descriptor rows/cols and a matrix count. All accesses retain the
     * source strides initialized by their declaration.
     */
    [[nodiscard]] std::string BindGlobalTensor(const ir::VarPtr& tensorVar, const ir::CallPtr& op,
                                               const std::string& pointerExpr, const std::string& validRows,
                                               const std::string& validCols, const std::string& matrixCount = "");

    /**
     * \brief Emit a per-call-site GlobalTensor over a peer window payload (dist.get/dist.put).
     *
     * Window views have no prescan declaration (those are keyed to kernel tensor params), so
     * unlike BindGlobalTensor this declares a private alias set + instance at the call site.
     * *view_type* is the DTensor view as a synthetic contiguous tensor (offset flattening and
     * strides reuse the block.load/block.store helpers); *remote_base* is the peer payload
     * expression (element-typed pointer), *tile* the tile's C++ name for the valid window.
     * Returns the instance name to pass to TLOAD/TSTORE.
     */
    [[nodiscard]] std::string BindDistGlobalTensor(const ir::TensorTypePtr& viewType, const ir::MakeTuplePtr& offsets,
                                                   const std::optional<std::vector<int>>& tileDims, bool isTranspose,
                                                   const std::vector<ir::ExprPtr>& tileShape,
                                                   const std::string& remoteBase, const std::string& tile);

protected:
    // Override visitor methods for code generation - Statements
    void VisitStmt_(const ir::AssignStmtPtr& op) override;
    void VisitStmt_(const ir::EvalStmtPtr& op) override;
    void VisitStmt_(const ir::ReturnStmtPtr& op) override;
    void VisitStmt_(const ir::ForStmtPtr& op) override;
    void VisitStmt_(const ir::WhileStmtPtr& op) override;
    void VisitStmt_(const ir::IfStmtPtr& op) override;
    void VisitStmt_(const ir::YieldStmtPtr& op) override;
    void VisitStmt_(const ir::SectionStmtPtr& op) override;
    void VisitStmt_(const ir::SeqStmtsPtr& op) override;
    void VisitStmt_(const ir::BreakStmtPtr& op) override;
    void VisitStmt_(const ir::ContinueStmtPtr& op) override;

    void VisitStmt(const ir::StmtPtr& stmt) override;
    void VisitExpr(const ir::ExprPtr& expr) override;

    // Override visitor methods for code generation - Expressions
    // Leaf nodes
    void VisitExpr_(const ir::VarPtr& op) override;

    void VisitExpr_(const ir::ConstIntPtr& op) override;
    void VisitExpr_(const ir::ConstFloatPtr& op) override;
    void VisitExpr_(const ir::ConstBoolPtr& op) override;
    void VisitExpr_(const ir::CallPtr& op) override;
    void VisitExpr_(const ir::GetItemExprPtr& op) override;
    void VisitExpr_(const ir::MakeTuplePtr& op) override;

    // Binary operations
    void VisitExpr_(const ir::AddPtr& op) override;
    void VisitExpr_(const ir::SubPtr& op) override;
    void VisitExpr_(const ir::MulPtr& op) override;
    void VisitExpr_(const ir::FloorDivPtr& op) override;
    void VisitExpr_(const ir::FloorModPtr& op) override;
    void VisitExpr_(const ir::FloatDivPtr& op) override;
    void VisitExpr_(const ir::MinPtr& op) override;
    void VisitExpr_(const ir::MaxPtr& op) override;
    void VisitExpr_(const ir::PowPtr& op) override;
    void VisitExpr_(const ir::EqPtr& op) override;
    void VisitExpr_(const ir::NePtr& op) override;
    void VisitExpr_(const ir::LtPtr& op) override;
    void VisitExpr_(const ir::LePtr& op) override;
    void VisitExpr_(const ir::GtPtr& op) override;
    void VisitExpr_(const ir::GePtr& op) override;
    void VisitExpr_(const ir::AndPtr& op) override;
    void VisitExpr_(const ir::OrPtr& op) override;
    void VisitExpr_(const ir::XorPtr& op) override;
    void VisitExpr_(const ir::BitAndPtr& op) override;
    void VisitExpr_(const ir::BitOrPtr& op) override;
    void VisitExpr_(const ir::BitXorPtr& op) override;
    void VisitExpr_(const ir::BitShiftLeftPtr& op) override;
    void VisitExpr_(const ir::BitShiftRightPtr& op) override;

    // Unary operations
    void VisitExpr_(const ir::AbsPtr& op) override;
    void VisitExpr_(const ir::NegPtr& op) override;
    void VisitExpr_(const ir::NotPtr& op) override;
    void VisitExpr_(const ir::BitNotPtr& op) override;
    void VisitExpr_(const ir::CastPtr& op) override;

private:
    /**
     * \brief Emit yield-assignment code for if-stmt return variables.
     *
     * For each return variable, writes the buffered YieldStmt expression to the
     * corresponding return variable and clears yieldBuffer_ when done.
     */
    void EmitYieldAssignments(const std::vector<ir::VarPtr>& returnVars);

    /**
     * \brief Emit full phi-style IfStmt codegen.
     *
     * Declares return variables before the if, emits then and else bodies, and
     * writes yield-assignment code in each branch.
     */
    void EmitFullPhiIf(const ir::IfStmtPtr& op);

    /**
     * \brief Generate function body
     *
     * Visits the function body statement to generate the main code.
     *
     * \param func The function to generate body for
     */
    void GenerateBody(const ir::FunctionPtr& func);

    /**
     * \brief Extract constant integer value from expression
     *
     * \param expr The expression (must be ConstInt)
     * \return The integer value
     */
    int64_t ExtractConstInt(const ir::ExprPtr& expr) const;

    /**
     * \brief Collect all TileType variables from function body
     *
     * Recursively traverses the statement tree to find all variables
     * with TileType that need Tile declarations in the prologue.
     *
     * \param stmt The statement to scan (typically func->body_)
     * \return Vector of (Var, TileType) pairs
     */
    std::vector<std::pair<ir::VarPtr, ir::TileTypePtr>> CollectTileVariables(const ir::StmtPtr& stmt);

    /// Collect one TensorDef per accessed tensor var *and layout variant* from block.load/store
    /// access windows into tensorDefs_, keyed by (tensor cce name, layout key) (parameters and
    /// ptr.make_tensor views both collected; used for prologue param-tensor emission and
    /// in-place view emission at the make_tensor op).
    void CollectTensorDefs(const ir::FunctionPtr& func);

    /**
     * \brief Extract shape dimensions from shape expressions
     *
     * Converts a vector of shape expressions (assumed to be ConstInt)
     * into a vector of integer dimensions.
     *
     * \param shapeExprs Vector of shape expressions (ConstInt)
     * \return Vector of integer dimensions
     */
    std::vector<int64_t> ExtractShapeDimensions(const std::vector<ir::ExprPtr>& shapeExprs) const;

    /**
     * \brief Register each tiling TupleType parameter for deferred header emission.
     */
    void RegisterTilingStructTypes(const ir::FunctionPtr& func);

    /**
     * \brief Emit all registered class definitions and materialize tiling headers.
     */
    void EmitStructTypes();

    /**
     * \brief Emit the kernel-entry copy from the `<name>_ptr` GM pointer into the local
     *        struct `<name>` (cube: copy_data_align64 GM→struct; vector: copy GM→UB scratch,
     *        then copy_data_align64 UB→struct).
     */
    void EmitTilingStructCopy(const ir::FunctionPtr& func);

    /**
     * \brief Generate the ordinary kernel device-function signature and prologue
     *
     * Emits __aicore__ inline void <name>_impl_cube/vector with target setup
     * and tiling and Tensor declarations.
     */
    void GenerateSinglePrologue(const ir::FunctionPtr& func, bool hasCrossSync);
    std::string BuildSimtFunctionSignature(const ir::FunctionPtr& func);
    void GenerateSimtFunction(const ir::FunctionPtr& func);
    std::string GenerateSimtCalleeCall(const ir::CallPtr& op, const ir::FunctionPtr& callee);
    void ResetFunctionGenerationState();
    std::unordered_set<std::string> vfTileSubviewPointerCodes_;

    bool DetectCrossCoreSyncOps(const ir::StmtPtr& stmt);

    // --- Phase 5: ForStmt helpers ---

    /**
     * \brief Register loop iteration arguments and emit their initialization.
     *
     * Unknown init values use the uninitialized declaration path; all other init
     * values use the initialized declaration path. Returns the sanitized names
     * of each iter-arg for later yield assignment. Shared by ForStmt and WhileStmt.
     */
    std::vector<std::string> RegisterLoopIterArgs(const std::vector<ir::IterArgPtr>& iterArgs);

    /// Write source expressions to typed loop-carried slots, skipping self-assignments.
    /// TupleType is dispatched here by its CCE representation: tuples with a
    /// backing array are copied element-wise, while aggregates recurse into leaf slots.
    /// Shared by If/loop yields and native jumps.
    void EmitCarriedAssignments(const std::vector<ir::VarPtr>& targets, const std::vector<ir::ExprPtr>& sources);

    /// Generate `type name`, `type name = value`, or `name = value`. A null value
    /// omits assignment; initialize controls whether the generated statement includes the type.
    /// Shared entry validation for EmitVariable: skips UnknownType/NoneType targets, treats
    /// UnknownType/NoneType sources as valueless, and drops self-assignment back edges.
    bool CheckEmitVariable(const ir::VarPtr& target, ir::ExprPtr& value, bool initialize);
    void EmitVariable(const ir::VarPtr& target, ir::ExprPtr value, bool initialize);
    void EmitTupleVariable(const std::string& name, const ir::TupleTypePtr& type, const ir::ExprPtr& value,
                           bool initialize);
    void RegisterLoopReturnVars(const std::vector<ir::VarPtr>& returnVars,
                                const std::vector<std::string>& iterArgNames);

    /**
     * \brief Emit a C++ loop shared by both loop kinds.
     *
     * The caller supplies the fully-formed loop header line ("for (...) {" or "while (...) {");
     * For loops write both break and continue values to the iter-arg slots. While loops keep
     * continue values in iter-arg slots and break values in independent return-var slots.
     */
    void EmitLoop(const std::string& header, const ir::StmtPtr& body, const std::vector<ir::IterArgPtr>& iterArgs,
                  const std::vector<ir::VarPtr>& returnVars, const std::vector<std::string>& iterArgNames,
                  bool separateWhileResults);

    // --- Native break/continue support ---

    /// Write a native jump's self-described values to the innermost loop's matching slots.
    void EmitJumpCarriedWriteback(const std::vector<ir::ExprPtr>& values, bool isBreak);

    // --- Phase 6: GenerateSinglePrologue helpers ---

    /**
     * \brief Emit the __global__ AICORE function signature and opening boilerplate.
     *
     * Collects dynamic dim variables, builds the parameter list, emits the opening
     * brace, registers dynamic dims, and emits FFTS setup if needed.
     */
    void EmitSingleFunctionSignature(const ir::FunctionPtr& func, bool hasCrossSync);

    /**
     * \brief Emit GlobalTensor declarations for one target Program.
     *
     * Registers the C++ name (= tensor var name) and pointer ("<name>_ptr") for every
     * tensor parameter and collects TensorDefs.
     */
    void EmitSingleTensorDeclarations(const ir::FunctionPtr& func);

    // --- Offset computation helpers ---
    std::string ComputeStrideOffset(const ir::TensorTypePtr& tensorType, const ir::MakeTuplePtr& offsets);
    std::string ComputeNZStrideOffset(const ir::TensorTypePtr& tensorType, const ir::MakeTuplePtr& offsets);

    // --- Phase 7: GenerateGlobalTensorTypeDeclaration helpers ---

    /// Whether an access walks the tensor down columns: transposed, or a single-column window.
    static bool IsDNAccessLayout(const std::vector<int64_t>& shapeDims, bool isTranspose);

    /// The Stride ctor arguments an access walks the tensor with, row/col exchanged when transposed.
    std::string BuildAccessStrideArgs(const ir::TensorTypePtr& tensorType,
                                      const std::optional<std::vector<int>>& tileDims, bool isTranspose, bool isMx,
                                      bool isMerged = false);

    /**
     * \brief Emit one GlobalTensor instance over a declaration made by GenerateGlobalTensorTypeDeclaration.
     *
     * Renders `<decl>Type <instance>(<pointer>, <decl>ShapeDim5(<shape_args>), <decl>StrideDim5(<stride_args>));`.
     * When both metadata argument strings are empty, only the pointer is emitted.
     */
    void EmitGlobalTensorInstance(const std::string& instanceName, const std::string& declName,
                                  const std::string& pointerExpr, const std::string& shapeArgs,
                                  const std::string& strideArgs);

    // Dual-mode context for expression visitor pattern
    std::string currentTargetVar_;               ///< INPUT: Assignment target variable name (for Call expressions)
    std::string currentExprValue_;               ///< OUTPUT: Inline C++ value for scalar / tile expressions
    ir::MakeTuplePtr currentTuple_;              ///< OUTPUT: underlying MakeTuple for tuple-typed expressions
    std::vector<ir::ExprPtr> yieldBuffer_;       ///< If-branch Yield expressions awaiting phi assignment
    const ir::IRDebugInfo* debugInfo_ = nullptr; ///< Program tuple metadata, required at codegen entry
    std::map<std::string, std::string> tilingHeaders_;          ///< Tiling struct headers (filename -> content)
    std::map<std::string, StructDefinition> structDefinitions_; ///< Struct type name ->definition

    CodeEmitter emitter_;                                            ///< Code emitter for structured output
    CodeContext context_;                                            ///< Context for variable tracking
    TypeConverter typeConverter_;                                    ///< Type converter
    const backend::Backend* backend_;                                ///< CCE backend instance (for op info, core type)
    npu::tile_fwk::NPUArch arch_ = npu::tile_fwk::NPUArch::DAV_2201; ///< Target architecture
    const ir::SectionKind target_;                                   ///< Fixed Cube/Vector target for this generator
    bool inVfSection_ = false;                                       ///< True only while emitting a nested VF section
    bool needsScalarHelpers_ = false;
    /// Actual IR type of the SIMT function being emitted.
    ir::FunctionType currentFunctionType_ = ir::FunctionType::OPAQUE;
    /// Kernel entry type for GenerateSingle; preserved across per-function resets.
    ir::FunctionType currentKernelType_ = ir::FunctionType::OPAQUE;
    std::map<std::string, ir::FunctionPtr> simtCallees_;
    std::unordered_map<std::string, std::string> tensorToPointer_; ///< Tensor var name ->raw pointer expression
    /// Prescan: (tensor cce name, layout key) ->TensorDef. One entry per layout an access needs,
    /// so every variant of one tensor is a contiguous range (see GetTensorDefs).
    std::map<std::pair<std::string, std::string>, TensorDef> tensorDefs_;
    int flatTensorDeclCount_ = 0; ///< Monotonic suffix for DeclareFlatGlobalTensor names, reset per function

    std::map<std::pair<bool, std::string>, std::string>
        vfTilePtrs_; ///< (is_post_update, VF tile expr code) -> hoisted vf_tile_ptr_N var. A dedicated var (e.g. a
                     ///< POST_UPDATE store cursor) and a plain base pointer to the same tile are kept separate, so a
                     ///< cursor store does not corrupt a base load of the same tile.
    std::vector<std::string> sectionHoistedDecls_; ///< VF section decls hoisted before __VEC_SCOPE__ (pre mem_bar)

    struct LoopWritebackTargets {
        std::vector<ir::VarPtr> continueVars;
        std::vector<ir::VarPtr> breakVars;
    };

    /// Per active loop, keep separate continue and break write-back slots.
    std::vector<LoopWritebackTargets> loopTargetStack_;

    int tileOffsetCounter_ = 0; ///< Counter for unique tile-offset GetItemExpr temp names
    int vfTilePtrCounter_ = 0;  ///< Counter for unique VF base-ptr var names (vf_tile_ptr_N)

    /// Track VF RegTensor variable names (C++ names set by EmitVFRegTensor)
    std::set<std::string> regTensorVars_;

    /// RegTensor declarations to hoist to the top of __VEC_SCOPE__ (avoid loop-interior placement)
    std::vector<std::string> vfRegHoistedDecls_;

    /// Track VF MaskReg variable names (C++ names set by EmitVFCreateMask/Compare/etc)
    std::set<std::string> maskRegVars_;

    /// Track VF AddrReg variable names (C++ names set by EmitVFCreateAddrReg)
    std::set<std::string> addrRegVars_;

    /// Track VF UnalignReg variable names (C++ names set by EmitVFUnalignRegForStore/Load)
    std::set<std::string> unalignRegVars_;

    // Tuple Var C++ name -> the MakeTuple that was assigned to it.
    // Populated during body codegen by VisitStmt_(AssignStmtPtr) on tuple-typed lhs:
    // VisitExpr(rhs) drives currentTuple_ through MakeTuple literals, tuple-var aliases,
    // and chained static GetItem, all via the visitExpr chain. ForStmt iterArg/returnVar
    // are propagated in VisitStmt_(ForStmtPtr).
    std::map<std::string, ir::MakeTuplePtr> tupleVarToMakeTuple_;

    // Array-valued MakeTuple object -> its unique C++ backing array. Parser constant
    // propagation can replace tuple Vars with their MakeTuple, so array identity
    // follows the IR object rather than a particular Var name.
    std::map<const ir::MakeTuple*, std::string> tupleBackingArr_;

    /**
     * \brief Whether all element types of a tuple are structurally identical.
     *
     * Gate for emitting the dynamic-index array at the MakeTuple assignment: an array
     * is only valid when every element maps to the same C++ type. Empty tuple -> false.
     */
    bool IsHomogeneousTuple(const ir::TupleTypePtr& tt) const;

    /// Semantic metadata registered for one exact tuple type, or null for a plain tuple.
    const ir::TupleTypeInfo* GetTupleTypeInfo(const ir::TupleTypePtr& tupleType) const;

    /// The C++ struct type name registered for a tuple type, or null if it is not a struct.
    /// Reads the parser's side table, so the answer does not depend on how far codegen has
    /// walked the body.
    const std::string* GetStructName(const ir::TupleTypePtr& tupleType) const;

    /**
     * \brief Whether a tuple is represented as a C++ array (vs. flattened leaf slots).
     *
     * Pure function of the type: it reads only the TupleType and debugInfo_, which is
     * captured once at GenerateSingle entry, so it does not depend on how far codegen has
     * walked the body.
     *
     * A tuple gets an array only when it is homogeneous and has no TupleTypeInfo
     * registration. Named tuples and structs are emitted as aggregate leaf slots.
     */
    bool IsArrayTuple(const ir::TupleTypePtr& tt) const;

    /**
     * \brief Collect element C++ names from a tuple value expression.
     *
     * Body-codegen use. visitExpr-driven: VisitExpr(tuple_value) resolves the
     * underlying MakeTuple into currentTuple_, then each element is VisitExpr'd
     * to produce its C++ name in currentExprValue_. Dynamic-GetItem callers
     * guarantee all elements are tile/scalar Var or const literals, so each
     * element visit yields a non-empty currentExprValue_.
     * Returns empty vector on failure.
     */
    std::vector<std::string> CollectTupleElemNames(const ir::ExprPtr& tupleValue);

    std::string BuildDynamicTupleArrayDecl(const ir::TypePtr& elemType, const std::vector<std::string>& elemNames,
                                           const std::string& arrName) const;

    ir::Span currentStmtSpan_;
    ir::Span currentExprSpan_;
};

} // namespace codegen
} // namespace pypto

#endif // PYPTO_CODEGEN_CCE_CCE_CODEGEN_H_
