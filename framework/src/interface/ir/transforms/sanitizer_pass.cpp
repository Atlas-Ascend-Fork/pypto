/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <unordered_map>
#include <utility>

#include "backend/common/backend_utils.h"
#include "ir/kind_traits.h"
#include "ir/memory_space.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr.h"
#include "ir/scalar_expr_ops.h"
#include "ir/type.h"
#include "ir/transforms/base/mutator.h"
#include "ir/transforms/base/visitor.h"
#include "ir/transforms/passes.h"
#include "ir/transforms/sanitizer_layout.h"
#include "pypto_pro/error.h"

namespace pypto {
namespace ir {

namespace {

// det_id values live in ir/transforms/sanitizer_layout.h together with the
// record layouts -- the single wire-contract source shared with the pybind
// export and the Python replay.

// Phase record op kinds. Write side = matmul family (dst Acc tile), read side
// = L0C consumers (src Acc tile): store/store_tile/move/move_fp/insert.
// Values travel in the record; the replay classifies (write/read,
// overwrite/accumulate, bias) from the kind alone. Kind numbering is frozen
// (the wire discipline): 7/9 never travel today -- the DSL lowers store_tile
// to block.store and the frontend no longer emits move_fp -- but stay
// reserved instead of being renumbered.
enum class PhaseOpKind : uint32_t {
    Matmul = 1,
    MatmulAcc = 2,
    MatmulBias = 3,
    MatmulMx = 4,
    MatmulMxAcc = 5,
    Store = 6,
    StoreTile = 7, // reserved (never emitted; store_tile lowers to block.store)
    Move = 8,
    MoveFp = 9,  // reserved (never emitted; the frontend dropped move_fp)
    Insert = 10, // Acc->Mat insert: reads the Acc tile (src) with STPhase
};

/// Max logical tensor rank supported by the GM bounds check.
constexpr uint32_t kSanitizerMaxTensorDims = 5;
constexpr char kTileAddrOp[] = "block.tile_addr";

// Runtime-evaluated GetValidRow()/GetValidCol() of the tile.
ExprPtr TileDim(const ExprPtr& tile, int axis)
{
    return OpRegistry::GetInstance().Create("block.tile_valid_shape", {tile}, {{"axis", axis}}, tile->span_);
}

ExprPtr Int64Const(int64_t v) { return std::make_shared<ConstInt>(v, DataType::INT64, Span::Unknown()); }

// ---------------------------------------------------------------------------
// SanitizerInstrumenter injects the log records (an IRMutator).
//
// Two record handlers: RecordBounds (every memory-access op -- GM
// loads/stores, tile transfers at an offset, scalar access, set_validshape
// windows) and RecordMutex (lock/unlock pairing); the op name picks the
// operand layout inside the handler. Records carry values only -- the pass
// never decides, the replay computes the detections. Declarations (make_tile,
// ptr.make_tensor) get their own declaration records at the AssignStmt hook.
// ---------------------------------------------------------------------------

class SanitizerInstrumenter : public IRMutator {
public:
    using IRMutator::VisitExpr_;
    using IRMutator::VisitStmt_;

    SanitizerInstrumenter()
    {
        // Table entries name the check the record serves, not the op spell.
        // Same-entry ops differ only in how they collect the operands (that
        // part lives inside each builder).
        auto tensor_bounds = [this](const CallPtr& c) { return RecordGmAccess(c); };
        auto tile_transfer_bounds = [this](const CallPtr& c) {
            // move (TEXTRACT): the offset is the SOURCE extraction
            // start, the window the DESTINATION's valid shape.
            ExprPtr src_off_row = Int64Const(0);
            ExprPtr src_off_col = Int64Const(0);
            for (size_t i = 2; i < c->args_.size(); ++i) {
                if (auto off_tuple = As<MakeTuple>(c->args_[i])) {
                    if (!off_tuple->elements_.empty())
                        src_off_row = off_tuple->elements_[0];
                    if (off_tuple->elements_.size() > 1)
                        src_off_col = off_tuple->elements_[1];
                    break;
                }
            }
            ExprPtr win_row = TileDim(c->args_[0], 0);
            ExprPtr win_col = TileDim(c->args_[0], 1);
            auto src = RecordTileAccess(c, 1, src_off_row, src_off_col, win_row, win_col);
            auto dst = RecordTileAccess(c, 0, Int64Const(0), Int64Const(0), win_row, win_col);
            return SeqStmts::Flatten({src, dst}, c->span_);
        };
        auto tile_insert_bounds = [this](const CallPtr& c) {
            // insert (TINSERT): the offset is the DESTINATION write start,
            // the window the SOURCE's valid shape.
            ExprPtr win_row = TileDim(c->args_[1], 0);
            ExprPtr win_col = TileDim(c->args_[1], 1);
            auto src = RecordTileAccess(c, 1, Int64Const(0), Int64Const(0), win_row, win_col);
            auto dst = RecordTileAccess(c, 0, c->args_[2], c->args_[3], win_row, win_col);
            return SeqStmts::Flatten({src, dst}, c->span_);
        };
        auto scalar_bounds = [this](const CallPtr& c) { return RecordScalarAccess(c); };
        auto validshape_bounds = [this](const CallPtr& c) { return RecordValidShape(c); };
        auto mutex_pairing = [this](const CallPtr& c) { return RecordMutex(c); };
        auto tile_overlap_decl = [this](const CallPtr& c) { return CollectMakeTile(c); };
        auto view_source_decl = [this](const CallPtr& c) { return CollectMakeTensor(c); };
        // unit_flag phase records. The matmul family carries no other record
        // (bounds live on the load/move side); the read side composes with
        // the existing GM/tile-transfer records so one op emits both.
        auto phase_matmul = [this](const CallPtr& c) { return RecordPhase(c, PhaseOpKind::Matmul, 0); };
        auto phase_matmul_acc = [this](const CallPtr& c) { return RecordPhase(c, PhaseOpKind::MatmulAcc, 0, 1); };
        auto phase_matmul_bias = [this](const CallPtr& c) { return RecordPhase(c, PhaseOpKind::MatmulBias, 0); };
        auto phase_matmul_mx = [this](const CallPtr& c) { return RecordPhase(c, PhaseOpKind::MatmulMx, 0); };
        auto phase_matmul_mx_acc = [this](const CallPtr& c) { return RecordPhase(c, PhaseOpKind::MatmulMxAcc, 0, 1); };
        // sync_src/sync_dst pairing: one record per call (the flag event is a
        // one-way signal, not a resource), grouped by (region, pipes, event).
        auto sync_record = [this](const CallPtr& c) { return RecordSync(c); };
        auto gm_and_phase_store = [this, tensor_bounds](const CallPtr& c) {
            auto gm = tensor_bounds(c);
            auto phase = RecordPhase(c, PhaseOpKind::Store, 1);
            if (gm == nullptr)
                return phase;
            if (phase == nullptr)
                return gm;
            return SeqStmts::Flatten({gm, phase}, c->span_);
        };
        auto tile_and_phase_move = [this, tile_transfer_bounds](const CallPtr& c) {
            auto bounds = tile_transfer_bounds(c);
            auto phase = RecordPhase(c, PhaseOpKind::Move, 1);
            if (phase == nullptr)
                return bounds;
            return SeqStmts::Flatten({bounds, phase}, c->span_);
        };
        // insert reads the Acc tile (src, Acc->Mat) with STPhase: compose the
        // tile-bounds record with the phase record, like store/move.
        auto tile_and_phase_insert = [this, tile_insert_bounds](const CallPtr& c) {
            auto bounds = tile_insert_bounds(c);
            auto phase = RecordPhase(c, PhaseOpKind::Insert, 1);
            if (phase == nullptr)
                return bounds;
            return SeqStmts::Flatten({bounds, phase}, c->span_);
        };

        op_records_ = {
            {"block.make_mutex_tile", tile_overlap_decl},
            {"ptr.make_tensor", view_source_decl},
            {"block.load", tensor_bounds},
            {"block.store", gm_and_phase_store},
            {"block.move", tile_and_phase_move},
            {"block.insert", tile_and_phase_insert},
            {"block.getval", scalar_bounds},
            {"block.setval", scalar_bounds},
            {"block.set_validshape", validshape_bounds},
            {"system.mutex_lock_dyn", mutex_pairing},
            {"system.mutex_unlock_dyn", mutex_pairing},
            {"system.sync_src_dyn", sync_record},
            {"system.sync_dst_dyn", sync_record},
            {"block.matmul", phase_matmul},
            {"block.matmul_acc", phase_matmul_acc},
            {"block.matmul_bias", phase_matmul_bias},
            {"block.matmul_mx", phase_matmul_mx},
            {"block.matmul_mx_acc", phase_matmul_mx_acc},
        };
    }

    ProgramPtr Instrument(ProgramPtr program);

private:
    // -- Layer 1: traversal hooks -------------------------------------------
    StmtPtr VisitStmt_(const EvalStmtPtr& op) override;
    StmtPtr VisitStmt_(const AssignStmtPtr& op) override;
    // ------------------------------------------------------------------------

    // Record builders shared by the table entries: every entry produces its
    // records through these, varying only how it collects the operands.
    // RecordGmAccess: GM loads/stores (per-dim record). RecordTileAccess:
    // one tile record (dims from the access-site type, caller supplies the
    // offset and window expressions). RecordMutex: one record per mutex id.
    StmtPtr RecordGmAccess(const CallPtr& call);
    StmtPtr RecordTileAccess(const CallPtr& call, uint32_t tile_arg_idx, const ExprPtr& off_row, const ExprPtr& off_col,
                             const ExprPtr& win_row, const ExprPtr& win_col);
    StmtPtr RecordMutex(const CallPtr& call);
    // sync_src/sync_dst record: event_id (runtime expr from args[0]) +
    // set_pipe/wait_pipe kwargs + is_src flag. Frontend already validates
    // pipe combos and the [0,7] event range at compile time; the runtime
    // pairing/alternation is what this record feeds.
    StmtPtr RecordSync(const CallPtr& call);
    // unit_flag phase record: op_kind + phase kwarg (Unspecified when absent)
    // + the Acc tile's memref addr expressions as the runtime acc_addr /
    // acc_src_addr identity. acc_tile_idx picks the tile whose address lands
    // in acc_addr (0 = matmul dst, 1 = store/move src); acc_src_idx names
    // matmul_acc's accumulator source arg, or -1 when the op has no separate
    // accumulator source.
    StmtPtr RecordPhase(const CallPtr& call, PhaseOpKind kind, uint32_t acc_tile_idx, int32_t acc_src_idx = -1);
    StmtPtr RecordScalarAccess(const CallPtr& call);
    StmtPtr RecordValidShape(const CallPtr& call);
    // Op name -> record functions. Same-handler ops (load/store) share one
    // entry; per-op differences live in the collected operands only.
    using RecordFunc = std::function<StmtPtr(const CallPtr&)>;
    std::unordered_map<std::string, RecordFunc> op_records_;
    // Declaration records (table entries like every other op). The
    // make_tile builder reads the assigned variable's type from
    // target_type_ (set by the AssignStmt hook: the call's own args[0]
    // types as the TileType *constructor* expression, not a Tile).
    StmtPtr CollectMakeTile(const CallPtr& call);
    StmtPtr CollectMakeTensor(const CallPtr& call);
    // Table lookup for the op call hooks.
    StmtPtr RecordForCall(const CallPtr& call);
    // Keep the original Tile expression so the backend can read the runtime
    // Tile object data() address after normal TileGroup lowering.
    ExprPtr RuntimeTileAddr(const ExprPtr& tile) const;
    // Collection helpers shared by the record builders.
    // Declared tile dims (static by frontend contract); nullptr when the
    // expression is not a Tile.
    std::pair<int64_t, int64_t> TileDims(const ExprPtr& tile);
    // Whole-shape element count as an expression product (static and
    // dynamic dims alike travel in the record).
    ExprPtr ShapeProduct(const ShapedTypePtr& type, const Span& span);

    // Unified exit of every record function: assembles the block.sanitizer_log
    // call (det_id + fields + hidden log params + source line).
    StmtPtr MakeLogStmt(uint32_t det_id, std::vector<ExprPtr> fields, const Span& span);

    // Hidden log-buffer parameters, appended to the entry signature; passed
    // as Var references so codegen resolves their final (SSA-suffixed) names.
    ExprPtr log_param_;
    ExprPtr capacity_param_;
    // Type of the current AssignStmt target (set by the hook, read by the
    // make_tile builder; empty while visiting EvalStmts).
    TypePtr target_type_;
    // Frontend let/SSA bindings needed to resolve TileGroup tuple fields while
    // building sanitizer-only address expressions.
};

StmtPtr SanitizerInstrumenter::RecordForCall(const CallPtr& call)
{
    auto it = op_records_.find(call->name_);
    return (it == op_records_.end()) ? nullptr : it->second(call);
}

StmtPtr SanitizerInstrumenter::VisitStmt_(const EvalStmtPtr& op)
{
    auto new_stmt = IRMutator::VisitStmt_(op);
    auto call = As<Call>(op->expr_);
    if (call == nullptr)
        return new_stmt;
    auto record = RecordForCall(call);
    if (record == nullptr)
        return new_stmt;
    return SeqStmts::Flatten({record, new_stmt}, op->span_);
}

// AssignStmt runs the same op dispatch as EvalStmt: value-producing ops
// (getval...) and the declaration builders (make_tile / make_tensor) are
// all table entries; the hook knows nothing about names.
StmtPtr SanitizerInstrumenter::VisitStmt_(const AssignStmtPtr& op)
{
    auto new_stmt = IRMutator::VisitStmt_(op);
    target_type_ = op->var_->GetType();
    auto call = As<Call>(op->value_);
    if (call != nullptr) {
        auto record = RecordForCall(call);
        if (record != nullptr)
            new_stmt = SeqStmts::Flatten({record, new_stmt}, op->span_);
    }
    target_type_ = nullptr;
    return new_stmt;
}

ExprPtr SanitizerInstrumenter::RuntimeTileAddr(const ExprPtr& tile) const
{
    auto tile_type = As<TileType>(tile->GetType());
    if (tile_type == nullptr || !tile_type->memref_.has_value() || *tile_type->memref_ == nullptr) {
        return nullptr;
    }
    // Keep the tile expression intact. The CCE backend evaluates it after the
    // existing TileGroup/MakeTuple lowering and calls data() on that runtime
    // Tile object, so dynamic indices and control-flow-selected groups follow
    // the same path as the original operation.
    // Use the registry so block.tile_addr type deduction validates the Tile operand.
    return OpRegistry::GetInstance().Create(kTileAddrOp, {tile}, tile->span_);
}

// ---------------------------------------------------------------------------
// Record functions: collect the operands, hand fields to MakeLogStmt
// ---------------------------------------------------------------------------

// Collection helpers shared by the record builders.

std::pair<int64_t, int64_t> SanitizerInstrumenter::TileDims(const ExprPtr& tile)
{
    auto tile_type = As<TileType>(tile->GetType());
    if (tile_type == nullptr)
        return {-1, -1};
    auto dim0 = As<ConstInt>(tile_type->shape_[0]);
    auto dim1 = As<ConstInt>(tile_type->shape_[1]);
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, dim0 != nullptr && dim1 != nullptr)
        << "sanitizer: tile shape dims must be static";
    return {dim0->value_, dim1->value_};
}

ExprPtr SanitizerInstrumenter::ShapeProduct(const ShapedTypePtr& type, const Span& span)
{
    ExprPtr numel = Int64Const(1);
    for (const auto& d : type->shape_)
        numel = MakeMul(numel, d, span);
    return numel;
}

// Unified exit of every record function: assembles the block.sanitizer_log
// call (det_id + fields + hidden log params + source line). The field
// count is checked against the shared wire layout (sanitizer_layout.h) so
// an encoder/replay drift fails the build here, not as a decode error.
StmtPtr SanitizerInstrumenter::MakeLogStmt(uint32_t det_id, std::vector<ExprPtr> fields, const Span& span)
{
    const SanitizerRecordLayout* desc = FindSanitizerRecordLayout(det_id);
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR,
                            desc != nullptr && fields.size() + 1 == desc->wire_count)
        << "sanitizer: det " << det_id << " record fields (" << fields.size() << ") do not match the wire layout ("
        << (desc != nullptr ? desc->wire_count - 1 : 0) << " expected)";
    std::vector<ExprPtr> args;
    // Record head: magic in the high 16 bits, det_id in the low 16 -- the
    // decoder dispatches on the det_id half and self-verifies the magic.
    args.push_back(Int64Const(SanitizerRecordHead(det_id)));
    args.insert(args.end(), std::make_move_iterator(fields.begin()), std::make_move_iterator(fields.end()));
    args.push_back(log_param_);
    args.push_back(capacity_param_);
    // Source line for the replay report; the file comes from the Python
    // side, which knows the kernel's source path.
    args.push_back(Int64Const(static_cast<int64_t>(span.BeginLine())));
    auto call = std::make_shared<Call>("block.sanitizer_log", std::move(args), Span::Unknown());
    return std::make_shared<EvalStmt>(call, Span::Unknown());
}

// getval/setval: args = [container, index]. A Tensor container becomes a
// linear GM record (ndim = 0, acc_row = element count); a Tile container a
// tile-count record.
StmtPtr SanitizerInstrumenter::RecordScalarAccess(const CallPtr& call)
{
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, call->args_.size() >= 0x2)
        << "sanitizer: " << call->name_ << " requires (container, offset)";
    const auto& container = call->args_[0];
    auto [tile_dim0, tile_dim1] = TileDims(container);
    if (tile_dim0 >= 0) {
        std::vector<ExprPtr> fields;
        fields.push_back(Int64Const(tile_dim0));
        fields.push_back(Int64Const(tile_dim1));
        fields.push_back(call->args_[1]);
        return MakeLogStmt(static_cast<uint32_t>(SanitizerDetection::TileScalar), std::move(fields), call->span_);
    }
    auto tensor_type = As<TensorType>(container->GetType());
    if (tensor_type == nullptr)
        return nullptr;
    // Linear bound: the whole-tensor element count as an expression product
    // (static and dynamic dims alike travel in the record). Same length as
    // the per-dim GM record (one det owns one format): shape slots zero,
    // acc_row carries the count, no window axes (the scalar branch of the
    // replay never reads them).
    ExprPtr numel = ShapeProduct(tensor_type, call->span_);
    std::vector<ExprPtr> fields;
    fields.push_back(Int64Const(0)); // ndim = 0: linear-offset access
    fields.push_back(call->args_[1]);
    for (uint32_t i = 0; i < kSanitizerMaxTensorDims - 1; ++i)
        fields.push_back(Int64Const(0));
    for (uint32_t i = 0; i < kSanitizerMaxTensorDims; ++i)
        fields.push_back(Int64Const(0)); // shape slots (length only)
    fields.push_back(numel);
    fields.push_back(Int64Const(1));
    fields.push_back(Int64Const(-1)); // win_ax0: unused by the linear branch
    fields.push_back(Int64Const(-1)); // win_ax1
    return MakeLogStmt(static_cast<uint32_t>(SanitizerDetection::GmAccess), std::move(fields), call->span_);
}

// set_validshape: window being set vs the declared dims (constant windows
// included -- everything is judged by the replay).
StmtPtr SanitizerInstrumenter::RecordValidShape(const CallPtr& call)
{
    if (call->args_.size() != 0x3)
        return nullptr;
    auto [dim0, dim1] = TileDims(call->args_[0]);
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, dim0 >= 0)
        << "sanitizer: set_validshape target must be a Tile";
    std::vector<ExprPtr> fields;
    fields.push_back(Int64Const(dim0));
    fields.push_back(Int64Const(dim1));
    fields.push_back(Int64Const(0));
    fields.push_back(Int64Const(0));
    fields.push_back(call->args_[1]);
    fields.push_back(call->args_[2]);
    return MakeLogStmt(static_cast<uint32_t>(SanitizerDetection::TileAccess), std::move(fields), call->span_);
}

StmtPtr SanitizerInstrumenter::RecordGmAccess(const CallPtr& call)
{
    // Operand positions follow the shared access-op contract; the offsets are
    // always a MakeTuple (frontend normalizes, CCE codegen CHECKs the same).
    auto indices = backend::cce::ResolveAccessArgIndices(call->name_);
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR,
                            indices.tensor_arg_idx >= 0 && indices.tile_arg_idx >= 0 && indices.offsets_arg_idx >= 0)
        << "sanitizer: no access-arg contract for op " << call->name_;
    ExprPtr tensor = call->args_[indices.tensor_arg_idx];
    ExprPtr tile = call->args_[indices.tile_arg_idx];
    auto off_tuple = As<MakeTuple>(call->args_[indices.offsets_arg_idx]);
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, off_tuple != nullptr)
        << "sanitizer: " << call->name_ << " offsets must be a MakeTuple";

    // Fields: ndim, off[5] (padded), shape[5] (padded), acc_row/acc_col.
    // The tensor shape comes from the access-site TensorType -- static dims
    // are ConstInt, dynamic dims are Var expressions the codegen resolves to
    // the ABI scalars, so the record carries the runtime truth with no host
    // id<->shape mapping (aliases and dynamic dims included).
    auto tensor_type = As<TensorType>(tensor->GetType());
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, tensor_type != nullptr)
        << "sanitizer: " << call->name_ << " tensor operand must be a Tensor";
    uint32_t ndim = std::min<uint32_t>(static_cast<uint32_t>(off_tuple->elements_.size()), kSanitizerMaxTensorDims);
    std::vector<ExprPtr> fields;
    fields.push_back(Int64Const(ndim));
    // Offsets (up to 5, pad with 0).
    for (uint32_t i = 0; i < kSanitizerMaxTensorDims; ++i) {
        if (i < off_tuple->elements_.size()) {
            fields.push_back(off_tuple->elements_[i]);
        } else {
            fields.push_back(Int64Const(0));
        }
    }
    // Shape dims aligned with the offsets (same alignment as the replay's
    // trailing-dim windowing; dynamic dims travel as expressions).
    uint32_t rank = std::min<uint32_t>(static_cast<uint32_t>(tensor_type->shape_.size()), kSanitizerMaxTensorDims);
    for (uint32_t i = 0; i < kSanitizerMaxTensorDims; ++i) {
        uint32_t dim = (rank <= ndim) ? i : (ndim - rank) + i; // leading dims dropped
        fields.push_back(dim < rank ? tensor_type->shape_[dim] : Int64Const(0));
    }
    // The tile's two valid-shape windows land on the tensor axes named by
    // win_ax0/win_ax1 (-1: that tile dim maps to no axis). Axis resolution
    // delegates to the same single sources the backend consumes: the
    // frontend's order decomposition (ascending tile_dims + is_transpose,
    // reconstructed here), MXLoadTileDims for MX scale loads (phase-axis
    // default included), and the generic trailing-axes default.
    ExprPtr acc_row = TileDim(tile, 0);
    ExprPtr acc_col = TileDim(tile, 1);
    int32_t win_ax0 = -1;
    int32_t win_ax1 = -1;
    if (ndim == 1) {
        // Rank-1: the whole tile projects onto the single axis -- one tile
        // dim is statically 1, so the valid-shape product is the access
        // length whichever side is degenerate. acc_col carries no window of
        // its own (win_ax1 stays -1); write 1 so the report reads naturally.
        win_ax0 = 0;
        acc_row = MakeMul(acc_row, acc_col, call->span_);
        acc_col = Int64Const(1);
    } else {
        std::vector<int> tile_dims;
        if (backend::cce::IsMXLoadCall(call)) {
            tile_dims = backend::cce::MXLoadTileDims(call, tensor_type);
        } else if (call->HasKwarg("tile_dims")) {
            tile_dims = call->GetKwarg<std::vector<int>>("tile_dims");
        } else {
            tile_dims = {static_cast<int>(ndim) - 2, static_cast<int>(ndim) - 1};
        }
        if (tile_dims.size() != 0x2) {
            return nullptr;
        }
        // Rebuild the original order: tile_dims is ascending, is_transpose
        // swapped it, so the first tile dim owned the second axis.
        if (call->GetKwarg<bool>("is_transpose", false)) {
            win_ax0 = tile_dims[1];
            win_ax1 = tile_dims[0];
        } else {
            win_ax0 = tile_dims[0];
            win_ax1 = tile_dims[1];
        }
        // MX scale tiles store one physical phase pair per logical matrix
        // element. The backend lowering expands A's second dimension or B's
        // first dimension by two; replay must compare logical group counts
        // against the GM tensor and keep the phase pair out of that window.
        if (backend::cce::IsMXLoadCall(call)) {
            const auto layout = backend::cce::MXLoadLayoutName(call);
            if (layout.find("MX_A_") != std::string::npos) {
                acc_col = MakeFloorDiv(acc_col, Int64Const(2), call->span_);
            } else if (layout.find("MX_B_") != std::string::npos) {
                acc_row = MakeFloorDiv(acc_row, Int64Const(2), call->span_);
            }
        }
    }
    fields.push_back(std::move(acc_row));
    fields.push_back(std::move(acc_col));
    fields.push_back(Int64Const(win_ax0));
    fields.push_back(Int64Const(win_ax1));
    return MakeLogStmt(static_cast<uint32_t>(SanitizerDetection::GmAccess), std::move(fields), call->span_);
}

StmtPtr SanitizerInstrumenter::RecordTileAccess(const CallPtr& call, uint32_t tile_arg_idx, const ExprPtr& off_row,
                                                const ExprPtr& off_col, const ExprPtr& win_row, const ExprPtr& win_col)
{
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, tile_arg_idx < call->args_.size())
        << "sanitizer: missing tile operand at " << tile_arg_idx;
    // Dims come from the access-site tile type -- what the codegen
    // instantiates the transfer with, i.e. the runtime truth of the access.
    // This keeps the bound check independent of the tile table (overlap-scan
    // only) and lets group cursors record: unresolvable name, known type.
    auto [dim0, dim1] = TileDims(call->args_[tile_arg_idx]);
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, dim0 >= 0)
        << "sanitizer: tile operand must be a Tile";
    std::vector<ExprPtr> fields;
    fields.push_back(Int64Const(dim0));
    fields.push_back(Int64Const(dim1));
    fields.push_back(off_row);
    fields.push_back(off_col);
    fields.push_back(win_row);
    fields.push_back(win_col);
    return MakeLogStmt(static_cast<uint32_t>(SanitizerDetection::TileAccess), std::move(fields), call->span_);
}

StmtPtr SanitizerInstrumenter::RecordMutex(const CallPtr& call)
{
    bool is_lock = (call->name_ == "system.mutex_lock_dyn");
    int pipe = call->GetKwarg<int>("pipe", 0);
    std::vector<StmtPtr> logs;
    auto recordMutexId = [&](const ExprPtr& mutexId) {
        std::vector<ExprPtr> fields;
        fields.push_back(mutexId);
        fields.push_back(Int64Const(pipe));
        fields.push_back(Int64Const(is_lock ? 1 : 0));
        logs.push_back(
            MakeLogStmt(static_cast<uint32_t>(SanitizerDetection::MutexAccess), std::move(fields), call->span_));
    };

    recordMutexId(call->args_[0]);
    return SeqStmts::Flatten(std::move(logs), call->span_);
}

StmtPtr SanitizerInstrumenter::RecordPhase(const CallPtr& call, PhaseOpKind kind, uint32_t acc_tile_idx,
                                           int32_t acc_src_idx)
{
    // Phase records serve the unit_flag (AccPhase/STPhase) pairing check.
    // Read side (store/move/insert) records are emitted for EVERY call whose
    // source tile lives in Acc -- including phase-less ones, whose
    // Unspecified marker is what lets the replay separate "read without
    // phase" (mixed-sync A-1) from "no read at all" (dangling C-T1). Ops
    // whose operand is not an Acc tile are unrelated to unit_flag: no record.
    // The acc_addr field is the runtime Tile object data() expression, so
    // pairing groups use the true block identity for plain tiles and TileGroups.
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, acc_tile_idx < call->args_.size())
        << "sanitizer: " << call->name_ << " phase record tile index " << acc_tile_idx << " out of range [0, "
        << call->args_.size() << ")";
    auto tile_type = As<TileType>(call->args_[acc_tile_idx]->GetType());
    if (tile_type == nullptr)
        return nullptr;
    auto memref = tile_type->memref_.has_value() ? *tile_type->memref_ : nullptr;
    if (memref == nullptr)
        return nullptr;
    if (memref->memorySpace_ != MemorySpace::Acc)
        return nullptr;
    int phase = 0; // AccPhase/STPhase: Unspecified = 0, Partial = 1, Final = 2
    if (call->HasKwarg("phase")) {
        phase = call->GetKwarg<int>("phase");
    }
    auto acc_addr = RuntimeTileAddr(call->args_[acc_tile_idx]);
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, acc_addr != nullptr)
        << "sanitizer: Acc tile has no resolvable phase address";
    ExprPtr acc_src_addr = Int64Const(-1);
    if (acc_src_idx >= 0) {
        const auto src_idx = static_cast<uint32_t>(acc_src_idx);
        PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, src_idx < call->args_.size())
            << "sanitizer: " << call->name_ << " accumulator source index " << src_idx << " out of range [0, "
            << call->args_.size() << ")";
        auto acc_type = As<TileType>(call->args_[src_idx]->GetType());
        PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, acc_type != nullptr)
            << "sanitizer: " << call->name_ << " accumulator source is not a Tile";
        auto acc_memref = acc_type->memref_.has_value() ? *acc_type->memref_ : nullptr;
        PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, acc_memref != nullptr)
            << "sanitizer: " << call->name_ << " accumulator source has no memref";
        PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR,
                                acc_memref->memorySpace_ == MemorySpace::Acc)
            << "sanitizer: " << call->name_ << " accumulator source is not in Acc memory";
        acc_src_addr = RuntimeTileAddr(call->args_[src_idx]);
        PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, acc_src_addr != nullptr)
            << "sanitizer: accumulator source has no resolvable phase address";
    }
    std::vector<ExprPtr> fields;
    fields.push_back(Int64Const(static_cast<int64_t>(kind)));
    fields.push_back(Int64Const(phase));
    fields.push_back(std::move(acc_addr));
    fields.push_back(acc_src_addr);
    auto log = MakeLogStmt(static_cast<uint32_t>(SanitizerDetection::PhaseAccess), std::move(fields), call->span_);
    return log;
}

StmtPtr SanitizerInstrumenter::RecordSync(const CallPtr& call)
{
    // sync is a one-way signal dependency (src sets the flag once, dst waits
    // for it and clears it): the replay pairs records per
    // (region, set_pipe, wait_pipe, event_id) key with strict alternation.
    // All three key fields travel in the record -- the event id as the args[0]
    // runtime expression (the frontend allows an integer Scalar), the pipes as
    // compile-time enum values. Parameter-mismatched src/dst fall into
    // different groups and surface as unpaired endpoints, so no separate
    // cross-comparison is needed.
    bool is_src = (call->name_ == "system.sync_src_dyn");
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, !call->args_.empty())
        << "sanitizer: " << call->name_ << " requires the event_id argument";
    int set_pipe = call->GetKwarg<int>("set_pipe");
    int wait_pipe = call->GetKwarg<int>("wait_pipe");
    std::vector<ExprPtr> fields;
    fields.push_back(call->args_[0]); // event_id (runtime expression)
    fields.push_back(Int64Const(set_pipe));
    fields.push_back(Int64Const(wait_pipe));
    fields.push_back(Int64Const(is_src ? 1 : 0));
    return MakeLogStmt(static_cast<uint32_t>(SanitizerDetection::SyncAccess), std::move(fields), call->span_);
}

StmtPtr SanitizerInstrumenter::CollectMakeTile(const CallPtr& call)
{
    // make_tile: the assigned variable's type (target_type_, set by the
    // AssignStmt hook) carries the TileType; addr
    // comes from the resolved MemRef, whose size preserves explicit physical
    // spans and layout padding. Emits a
    // declaration record for the overlap scan; the replay dedups identical
    // re-registrations (the top-level statements are copied into both
    // programs of one kernel) by (addr, size, space, span) -- same-statement
    // copies share the span, distinct same-range tiles keep both records and
    // are exactly what the scan must report.
    auto tile_type = As<TileType>(target_type_);
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR,
                            tile_type != nullptr && tile_type->shape_.size() >= 0x2)
        << "sanitizer: make_tile value must be a 2-D TileType";
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR,
                            tile_type->memref_.has_value() && *tile_type->memref_ != nullptr)
        << "sanitizer: make_tile requires a resolved MemRef (frontend enforces addr)";
    int64_t sz = 1;
    for (const auto& d : tile_type->shape_) {
        auto c = As<ConstInt>(d);
        PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, c != nullptr)
            << "sanitizer: tile shape dims must be static";
        sz *= c->value_;
    }
    auto memref = *tile_type->memref_;
    auto addr_const = As<ConstInt>(memref->addr_);
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, addr_const != nullptr)
        << "sanitizer: make_tile memref address must be a constant integer";
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, memref->size_ > 0)
        << "sanitizer: make_tile memref size must be positive";
    // Use the resolved MemRef footprint rather than shape*dtype.  The MemRef
    // preserves explicit spans and physical NZ/fractal padding.
    std::vector<ExprPtr> fields;
    fields.push_back(Int64Const(addr_const->value_));
    fields.push_back(Int64Const(static_cast<int64_t>(memref->size_)));
    fields.push_back(Int64Const(static_cast<int64_t>(memref->memorySpace_)));
    fields.push_back(Int64Const(sz)); // element count (report display)
    return MakeLogStmt(static_cast<uint32_t>(SanitizerDetection::TileDecl), std::move(fields), call->span_);
}

StmtPtr SanitizerInstrumenter::CollectMakeTensor(const CallPtr& call)
{
    // ptr.make_tensor: args = [ptr_or_tensor, shape_tuple, stride_tuple].
    // The view shares its source's storage, so the op contract ("all accesses
    // must stay inside the source") is checked at declaration time with one
    // ViewShape record carrying expressions only (design principle: every
    // check value travels in the record): the declared dims, the byte
    // footprint sum((dim-1)*stride) + 1 scaled by the view dtype width, and
    // the source's byte capacity (a raw-pointer source records -1: no known
    // bound). Nothing is registered -- the replay needs no host-side view
    // table.
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, call->args_.size() >= 0x2)
        << "sanitizer: ptr.make_tensor requires (ptr, shape)";
    auto shape_tuple = As<MakeTuple>(call->args_[1]);
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, shape_tuple != nullptr)
        << "sanitizer: ptr.make_tensor shape must be a MakeTuple";
    auto stride_tuple = As<MakeTuple>(call->args_[2]);
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, stride_tuple != nullptr)
        << "sanitizer: ptr.make_tensor stride must be a MakeTuple";
    PRO_PASS_INTERNAL_CHECK(
        npu::tile_fwk::InternalError::PASS_INNER_ERROR,
        stride_tuple->elements_.empty() || stride_tuple->elements_.size() == shape_tuple->elements_.size())
        << "sanitizer: ptr.make_tensor stride rank must match the shape";

    // Byte footprint expression: (sum((dim-1)*stride) + 1) * elem_bytes, with
    // an omitted stride read as the compact row-major product of inner dims.
    DataType view_dtype;
    if (call->HasKwarg("dtype")) {
        view_dtype = call->GetKwarg<DataType>("dtype");
    } else if (auto src_shaped = As<ShapedType>(call->args_[0]->GetType())) {
        view_dtype = src_shaped->dtype_;
    } else {
        auto src_ptr = As<PtrType>(call->args_[0]->GetType());
        PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, src_ptr != nullptr)
            << "sanitizer: ptr.make_tensor source must be a Ptr or Tensor";
        view_dtype = src_ptr->dtype_;
    }
    int64_t elem_bytes = std::max<int64_t>(1, static_cast<int64_t>(view_dtype.GetBit()) / 8);
    auto stride_of = [&](size_t i) -> ExprPtr {
        if (i < stride_tuple->elements_.size())
            return stride_tuple->elements_[i];
        ExprPtr product = Int64Const(1);
        for (size_t j = i + 1; j < shape_tuple->elements_.size(); ++j)
            product = MakeMul(product, shape_tuple->elements_[j], call->span_);
        return product;
    };
    ExprPtr fp_elems = Int64Const(1);
    for (size_t i = 0; i < shape_tuple->elements_.size(); ++i) {
        ExprPtr term = MakeMul(MakeSub(shape_tuple->elements_[i], Int64Const(1), call->span_), stride_of(i),
                               call->span_);
        fp_elems = MakeAdd(fp_elems, term, call->span_);
    }
    ExprPtr fp_bytes = MakeMul(fp_elems, Int64Const(elem_bytes), call->span_);

    // Prefer the resolved physical MemRef span or an existing view stride.
    // Fall back to the logical tensor extent when no storage metadata exists.
    ExprPtr src_bytes = Int64Const(-1);
    auto src_tensor = As<TensorType>(call->args_[0]->GetType());
    if (src_tensor != nullptr) {
        int64_t src_elem = std::max<int64_t>(1, static_cast<int64_t>(src_tensor->dtype_.GetBit()) / 8);
        if (src_tensor->memref_.has_value() && *src_tensor->memref_ != nullptr) {
            src_bytes = Int64Const(static_cast<int64_t>((*src_tensor->memref_)->size_));
        } else if (src_tensor->tensor_view_.has_value() &&
                   src_tensor->tensor_view_->stride.size() == src_tensor->shape_.size() &&
                   !src_tensor->tensor_view_->stride.empty()) {
            ExprPtr src_fp_elems = Int64Const(1);
            const auto& strides = src_tensor->tensor_view_->stride;
            for (size_t i = 0; i < src_tensor->shape_.size() && i < strides.size(); ++i) {
                ExprPtr term = MakeMul(MakeSub(src_tensor->shape_[i], Int64Const(1), call->span_), strides[i],
                                       call->span_);
                src_fp_elems = MakeAdd(src_fp_elems, term, call->span_);
            }
            src_bytes = MakeMul(src_fp_elems, Int64Const(src_elem), call->span_);
        } else {
            src_bytes = MakeMul(ShapeProduct(src_tensor, call->span_), Int64Const(src_elem), call->span_);
        }
    }

    std::vector<ExprPtr> fields;
    for (uint32_t i = 0; i < kSanitizerMaxTensorDims; ++i) {
        fields.push_back(i < shape_tuple->elements_.size() ? shape_tuple->elements_[i] : Int64Const(0));
    }
    // Source dims for the report display; a raw-pointer source pads zeros
    // (its records skip the check via src_bytes = -1 anyway).
    for (uint32_t i = 0; i < kSanitizerMaxTensorDims; ++i) {
        fields.push_back((src_tensor != nullptr && i < src_tensor->shape_.size()) ? src_tensor->shape_[i] :
                                                                                    Int64Const(0));
    }
    fields.push_back(std::move(fp_bytes));
    fields.push_back(std::move(src_bytes));
    return MakeLogStmt(static_cast<uint32_t>(SanitizerDetection::ViewShape), std::move(fields), call->span_);
}

// ---------------------------------------------------------------------------
// Instrument
// ---------------------------------------------------------------------------

ProgramPtr SanitizerInstrumenter::Instrument(ProgramPtr program)
{
    // Instrument the entry function only, appending the hidden
    // sanitizer_log (ptr) + sanitizer_log_capacity (scalar) parameters.
    //    The entry is identified structurally, as the Opaque function the
    //    frontend builds for the kernel body (the same criterion the host
    //    side uses to locate the kernel entry), never by name: the program
    //    name (e.g. from @pl.jit(name=...)) may differ from the entry
    //    function name, and a name match would silently skip instrumentation
    //    then, breaking the launch ABI.
    //    SIMT functions (SIMT_VF / SIMT_CALLEE) are outside the detection
    //    scope: excluded by function type (the codegen's own criterion), not
    //    only by name -- a same-named SIMT function can never be mistaken
    //    for the entry.
    std::map<std::string, FunctionPtr> new_functions;
    uint32_t entry_count = 0;
    for (const auto& [name, func] : program->functions_) {
        if (func == nullptr || func->funcType_ != FunctionType::OPAQUE) {
            new_functions[name] = func;
            continue;
        }
        auto log_var = std::make_shared<Var>("sanitizer_log", std::make_shared<PtrType>(DataType(DataType::INT8)),
                                             Span::Unknown());
        auto capacity_var = std::make_shared<Var>("sanitizer_log_capacity",
                                                  std::make_shared<ScalarType>(DataType::INT64), Span::Unknown());
        log_param_ = log_var;
        capacity_param_ = capacity_var;

        auto new_body = VisitStmt(func->body_);
        StmtPtr body = (new_body != func->body_) ? new_body : func->body_;

        auto params = func->params_;
        params.push_back(log_var);
        params.push_back(capacity_var);
        new_functions[name] = std::make_shared<Function>(func->name_, std::move(params), func->returnTypes_, body,
                                                         func->span_, func->funcType_, func->entry_);
        ++entry_count;
    }
    PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, entry_count == 1)
        << "sanitizer: expected exactly one Opaque entry function, found " << entry_count;
    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_, program->debugInfo_);
}

} // namespace

Pass pass::Sanitizer()
{
    return CreateProgramPass(
        [](const ProgramPtr& program) -> ProgramPtr {
            PRO_PASS_INTERNAL_CHECK(npu::tile_fwk::InternalError::PASS_INNER_ERROR, program)
                << "Sanitizer pass cannot run on a null program";
            SanitizerInstrumenter instrumenter;
            return instrumenter.Instrument(program);
        },
        "Sanitizer");
}

} // namespace ir
} // namespace pypto
