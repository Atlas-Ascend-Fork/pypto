/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "ir/transforms/core_placement.h"

#include <optional>
#include <unordered_map>

#include "backend/backend_cce.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/op_attr_types.h"
#include "ir/pipe.h"
#include "ir/transforms/base/visitor.h"
#include "ir/transforms/passes.h"
#include "pypto_pro/error.h"

namespace pypto::ir {
namespace {
using npu::tile_fwk::ExternalError;

CoreMask Intersect(CoreMask lhs, CoreMask rhs)
{
    return static_cast<CoreMask>(static_cast<unsigned>(lhs) & static_cast<unsigned>(rhs));
}

const char* CoreName(CoreMask cores)
{
    switch (cores) {
        case CoreMask::Cube:
            return "Cube";
        case CoreMask::Vector:
            return "Vector";
        case CoreMask::CV:
            return "Cube and Vector";
        case CoreMask::Aicpu:
            return "AICPU";
        case CoreMask::All:
            return "Cube, Vector and AICPU";
        default:
            return "neither core";
    }
}

std::optional<MemorySpace> TileMemory(const TypePtr& type)
{
    auto tile = As<TileType>(type);
    if (!tile || !tile->memref_)
        return std::nullopt;
    return (*tile->memref_)->memorySpace_;
}

std::optional<CoreMask> LocalTileCore(const CallPtr& call, size_t localTile)
{
    if (localTile >= call->args_.size())
        return std::nullopt;
    auto memory = TileMemory(call->args_[localTile]->GetType());
    if (memory == MemorySpace::Vec)
        return CoreMask::Vector;
    return std::nullopt;
}

std::optional<CoreMask> ExclusivePipeCore(PipeType pipe)
{
    switch (pipe) {
        case PipeType::M:
        case PipeType::MTE1:
        case PipeType::FIX:
            return CoreMask::Cube;
        case PipeType::V:
            return CoreMask::Vector;
        default:
            return std::nullopt;
    }
}

CoreMask SyncPipeCores(const CallPtr& call, const char* key)
{
    auto pipe = static_cast<PipeType>(call->GetKwarg<int>(key, -1));
    return ExclusivePipeCore(pipe).value_or(CoreMask::CV);
}

std::optional<CoreMask> ScalarAccessCores(const CallPtr& call)
{
    const auto& type = call->args_[0]->GetType();
    if (As<TensorType>(type))
        return CoreMask::CV;
    if (TileMemory(type) == MemorySpace::Vec)
        return CoreMask::Vector;
    return std::nullopt;
}

// GM scalar reads/writes (block.getval/block.setval on a GM Tensor) are the one
// scalar surface the AICPU kernel shares with the AI cores.
std::optional<CoreMask> GmScalarAccessCores(const CallPtr& call)
{
    const auto& type = call->args_[0]->GetType();
    if (As<TensorType>(type))
        return CoreMask::All;
    auto memory = TileMemory(type);
    if (memory == MemorySpace::Vec)
        return CoreMask::Vector;
    return std::nullopt;
}

std::optional<CoreMask> RegisteredPipeCores(const std::string& name)
{
    const auto* info = backend::BackendCCE::Instance().GetOpInfo(name);
    if (info && info->pipe)
        return ExclusivePipeCore(*info->pipe);
    return std::nullopt;
}

std::optional<CoreMask> SyncOpCores(const CallPtr& call)
{
    const auto& name = call->name_;
    if (name == "system.sync_all") {
        switch (static_cast<SyncCoreType>(call->GetKwarg<int>("core_type", static_cast<int>(SyncCoreType::MIX)))) {
            case SyncCoreType::AIC_ONLY:
                return CoreMask::Cube;
            case SyncCoreType::AIV_ONLY:
                return CoreMask::Vector;
            case SyncCoreType::MIX:
                return CoreMask::CV;
            default:
                return std::nullopt;
        }
    }
    if (name == "system.sync_src_dyn" || name == "system.sync_dst_dyn") {
        auto set = SyncPipeCores(call, "set_pipe");
        auto wait = SyncPipeCores(call, "wait_pipe");
        return Intersect(set, wait);
    }
    if (name == "system.set_cross_core" || name == "system.wait_cross_core" || name == "system.set_cross_core_dyn" ||
        name == "system.wait_cross_core_dyn") {
        auto cores = SyncPipeCores(call, "pipe");
        if (call->GetKwarg<int>("sync_mode", -1) == static_cast<int>(CrossCoreSyncMode::INTER_SUBBLOCK))
            return Intersect(cores, CoreMask::Vector);
        return cores;
    }
    if (name == "system.mutex_lock_dyn" || name == "system.mutex_unlock_dyn")
        return SyncPipeCores(call, "pipe");
    return std::nullopt;
}

class TargetValidator final : public IRVisitor {
public:
    explicit TargetValidator(CoreMask target) : target_(target) {}

    void VisitFunction(const FunctionPtr& func) override
    {
        // VF definitions have their own execution context; validate their launch at the call site.
        if (target_ == CoreMask::Cube && IsVectorFunction(func->funcType_))
            return;
        IRVisitor::VisitFunction(func);
    }

private:
    using IRVisitor::VisitExpr_;
    void VisitExpr_(const CallPtr& call) override
    {
        ValidateCorePlacement(call, GetAllowedCores(call), target_);
        IRVisitor::VisitExpr_(call);
    }

    CoreMask target_;
};

std::optional<CoreMask> OpSpecificCores(const CallPtr& call)
{
    // The registered pipe alone cannot determine the allowed cores for these operations.
    static const std::unordered_map<std::string, CoreMask> fixedCores{
        // TCI fills a UB tile even though its registered scheduling pipe is S.
        {"block.fill_index", CoreMask::Vector},

        // AICPU kernel surface: pointer construction and struct creation/writes
        // are target-independent IR bookkeeping that also holds on the control CPU.
        {"ptr.make_ptr", CoreMask::All},
        {"ptr.make_tensor", CoreMask::All},
        {"struct.create", CoreMask::All},
        {"struct.set", CoreMask::All},

        // Barrier placement follows the pipe being fenced.
        {"system.bar_fix", CoreMask::Cube},
        {"system.bar_m", CoreMask::Cube},
        {"system.bar_mte1", CoreMask::Cube},
        {"system.reset_mask", CoreMask::Vector},
        {"system.set_mask_count", CoreMask::Vector},
        {"system.set_mask_norm", CoreMask::Vector},
        {"system.set_mm_layout_transform", CoreMask::Cube},
        {"system.set_vec_mask", CoreMask::Vector},
    };
    const auto& name = call->name_;
    if (auto known = fixedCores.find(name); known != fixedCores.end())
        return known->second;
    // Window accessors read scalar fields from GM on either core.
    if (name.rfind("dist.", 0) == 0 && name != "dist.window_get")
        return CoreMask::Vector;
    if (name == "block.load" || name == "block.expands" || name == "block.fillpad" || name == "debug.dump_tile")
        return LocalTileCore(call, 0);
    if (name == "block.store" || name == "block.move" || name == "block.insert")
        return LocalTileCore(call, 1);
    if (name == "block.getval" || name == "block.setval")
        return GmScalarAccessCores(call);
    if (name == "system.dcci")
        return ScalarAccessCores(call);
    if (name.rfind("simt.", 0) == 0)
        return CoreMask::Vector;
    return SyncOpCores(call);
}
} // namespace

CoreMask GetAllowedCores(const CallPtr& call)
{
    if (auto cores = RegisteredPipeCores(call->name_))
        return *cores;
    if (auto cores = OpSpecificCores(call))
        return *cores;
    return CoreMask::CV;
}

void ValidateCorePlacement(const CallPtr& call, CoreMask allowed, CoreMask required)
{
    auto missing = static_cast<CoreMask>(static_cast<unsigned>(required) & ~static_cast<unsigned>(allowed));
    if (missing == CoreMask::None)
        return;
    pro::SpanScope spanScope(call->span_);
    PRO_IR_THROW(ValueError, ExternalError::INVALID_OPERATION)
        << "This operation cannot execute on " << CoreName(missing)
        << ". Supported cores for this call: " << CoreName(allowed) << ".";
}

Pass pass::ValidateTargetOps(SectionKind target)
{
    if (target != SectionKind::Cube && target != SectionKind::Vector && target != SectionKind::Aicpu) {
        PRO_IR_THROW(ValueError, ExternalError::INVALID_ARGUMENT)
            << "ValidateTargetOps requires a Cube, Vector or Aicpu target";
    }
    return CreateProgramPass(
        [target](const ProgramPtr& program) {
            CoreMask required = CoreMask::Cube;
            if (target == SectionKind::Vector) {
                required = CoreMask::Vector;
            } else if (target == SectionKind::Aicpu) {
                required = CoreMask::Aicpu;
            }
            TargetValidator validator(required);
            validator.VisitProgram(program);
            return program;
        },
        "ValidateTargetOps");
}
} // namespace pypto::ir
