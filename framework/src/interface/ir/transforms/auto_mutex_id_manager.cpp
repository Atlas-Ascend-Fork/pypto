/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ir/transforms/auto_mutex_id_manager.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/transforms/base/mutator.h"
#include "ir/type.h"
#include "pypto_pro/error.h"

namespace pypto {
namespace ir {

using npu::tile_fwk::ExternalError;

namespace {

constexpr int kMutexIdCount = 32;

class PlaceholderResolver final : public IRMutator {
public:
    explicit PlaceholderResolver(const std::vector<std::pair<VarPtr, int>>& replacements)
    {
        for (const auto& [placeholder, value] : replacements) {
            placeholderValues_[placeholder.get()] = value;
        }
    }

protected:
    using IRMutator::VisitExpr_;

    ExprPtr VisitExpr_(const VarPtr& op) override
    {
        auto it = placeholderValues_.find(op.get());
        if (it == placeholderValues_.end()) {
            // Ordinary SSA Vars are not mutex placeholders and must remain unchanged.
            return IRMutator::VisitExpr_(op);
        }
        return std::make_shared<const ConstInt>(it->second, DataType::INT64, op->span_);
    }

private:
    std::unordered_map<const Var*, int> placeholderValues_;
};

std::string HexRange(int64_t start, int64_t end)
{
    std::ostringstream os;
    os << "[0x" << std::hex << start << ", 0x" << end << ")";
    return os.str();
}

} // namespace

class AutoMutexIdManager::Impl {
public:
    struct TileMutexId {
        VarPtr placeholder;
        std::vector<int> manualIds;

        bool IsAuto() const { return placeholder != nullptr; }
    };

    struct TileRecord {
        MemorySpace memorySpace;
        int64_t start;
        int64_t end;
        std::vector<int> manualIds;
        VarPtr placeholder;
        std::string groupName;
        Span span;

        bool IsAuto() const { return placeholder != nullptr; }
    };

    struct AliasComponent {
        std::vector<size_t> tiles;
        MemorySpace memorySpace;
        int64_t start;
        int64_t end;
    };

    struct MutexJob {
        size_t component;
        std::vector<size_t> autoTiles;
        std::vector<int> candidates;
    };

    struct AllocationState {
        std::vector<AliasComponent> components;
        std::array<int, kMutexIdCount> tileUsageById{};
        std::array<int, kMutexIdCount> componentUsageById{};
    };

    void CollectGroup(const std::vector<ExprPtr>& tiles, const MakeTuplePtr& mutexIds, const std::string& groupName)
    {
        Span span = tiles.empty() || tiles.front() == nullptr ? Span::Unknown() : tiles.front()->span_;
        PRO_PASS_INTERNAL_CHECK_WITH_SPAN(npu::tile_fwk::InternalError::PASS_INNER_ERROR,
                                          mutexIds->elements_.size() == tiles.size(), span)
            << "mutex ID count must match TileGroup depth";

        // The outer tuple follows TileGroup depth; each row describes the IDs for the Tile at
        // the same position.
        for (size_t index = 0; index < tiles.size(); ++index) {
            auto tileMutexId = ParseTileMutexId(mutexIds->elements_[index], span);
            CollectTile(tiles[index], tileMutexId, groupName);
        }
    }

    ProgramPtr AssignMutexIds(const ProgramPtr& program)
    {
        if (!hasAuto_) {
            return program;
        }
        // Allocation is completed before mutation; the resolver then performs only registered
        // AUTO-placeholder-to-ConstInt substitutions.
        auto assignedIdByTile = AllocateMutexIds();
        std::vector<std::pair<VarPtr, int>> replacements;
        replacements.reserve(placeholders_.size());
        for (size_t tileIndex = 0; tileIndex < tiles_.size(); ++tileIndex) {
            const auto& tile = tiles_[tileIndex];
            if (!tile.IsAuto()) {
                continue;
            }
            auto value = assignedIdByTile[tileIndex];
            PRO_PASS_INTERNAL_CHECK_WITH_SPAN(npu::tile_fwk::InternalError::PASS_INNER_ERROR, value.has_value(),
                                              tile.span)
                << "AUTO mutex Tile has no allocated ID";
            replacements.emplace_back(tile.placeholder, *value);
        }
        PlaceholderResolver resolver(replacements);
        return resolver.VisitProgram(program);
    }

    Span DiagnosticSpan() const { return diagnosticSpan_; }

private:
    TileMutexId ParseTileMutexId(const ExprPtr& value, const Span& fallbackSpan)
    {
        auto mutexIds = As<MakeTuple>(value);
        PRO_PASS_INTERNAL_CHECK_WITH_SPAN(npu::tile_fwk::InternalError::PASS_INNER_ERROR,
                                          mutexIds != nullptr && !mutexIds->elements_.empty(), fallbackSpan)
            << "each Tile must provide a non-empty mutex-ID tuple";

        // A one-element Var row is the AUTO marker. Every other valid row is a list of fixed
        // manual ConstInt IDs.
        if (mutexIds->elements_.size() == 1) {
            auto placeholder = As<Var>(mutexIds->elements_.front());
            if (placeholder != nullptr) {
                return TileMutexId{placeholder, {}};
            }
        }

        std::vector<int> manualIds;
        manualIds.reserve(mutexIds->elements_.size());
        for (const auto& mutexIdExpr : mutexIds->elements_) {
            auto mutexId = As<ConstInt>(mutexIdExpr);
            Span mutexIdSpan = mutexIdExpr == nullptr ? fallbackSpan : mutexIdExpr->span_;
            PRO_PASS_INTERNAL_CHECK_WITH_SPAN(npu::tile_fwk::InternalError::PASS_INNER_ERROR, mutexId != nullptr,
                                              mutexIdSpan)
                << "manual mutex ID must be a ConstInt";
            ValidateMutexId(mutexId->value_, mutexIdSpan);
            manualIds.push_back(static_cast<int>(mutexId->value_));
        }
        return TileMutexId{nullptr, std::move(manualIds)};
    }

    void CollectTile(const ExprPtr& tile, const TileMutexId& tileMutexId, const std::string& groupName)
    {
        auto tileType = As<TileType>(tile->GetType());
        PRO_PASS_INTERNAL_CHECK_WITH_SPAN(
            npu::tile_fwk::InternalError::PASS_INNER_ERROR,
            tileType != nullptr && tileType->memref_.has_value() && *tileType->memref_ != nullptr, tile->span_)
            << "AUTO mutex-ID planning requires every Tile to have a MemRef";
        auto memref = *tileType->memref_;
        auto address = As<ConstInt>(memref->addr_);
        PRO_PASS_INTERNAL_CHECK_WITH_SPAN(npu::tile_fwk::InternalError::PASS_INNER_ERROR, address != nullptr,
                                          tile->span_)
            << "AUTO mutex-ID planning requires a constant Tile address";
        PRO_PASS_INTERNAL_CHECK_WITH_SPAN(
            npu::tile_fwk::InternalError::PASS_INNER_ERROR,
            memref->size_ <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) &&
                address->value_ <= std::numeric_limits<int64_t>::max() - static_cast<int64_t>(memref->size_),
            tile->span_)
            << "Tile address range overflows int64";
        auto previous = tileToIndex_.find(tile.get());
        if (previous != tileToIndex_.end()) {
            // One Tile can reach several mutex ops; keep one record and reject inconsistent
            // metadata rather than silently changing its allocation mode.
            const auto& record = tiles_[previous->second];
            PRO_PASS_INTERNAL_CHECK_WITH_SPAN(
                npu::tile_fwk::InternalError::PASS_INNER_ERROR,
                record.placeholder.get() == tileMutexId.placeholder.get() && record.manualIds == tileMutexId.manualIds,
                tile->span_)
                << "the same Tile was collected with inconsistent mutex settings";
            return;
        }

        if (tileMutexId.IsAuto()) {
            hasAuto_ = true;
            PRO_PASS_INTERNAL_CHECK_WITH_SPAN(npu::tile_fwk::InternalError::PASS_INNER_ERROR,
                                              placeholders_.find(tileMutexId.placeholder.get()) == placeholders_.end(),
                                              tileMutexId.placeholder->span_)
                << "AUTO mutex placeholder was reused by multiple Tiles";
        }

        size_t tileIndex = tiles_.size();
        tiles_.push_back(TileRecord{memref->memorySpace_, address->value_,
                                    address->value_ + static_cast<int64_t>(memref->size_), tileMutexId.manualIds,
                                    tileMutexId.placeholder, groupName, tile->span_});
        tileToIndex_[tile.get()] = tileIndex;
        if (tileMutexId.IsAuto()) {
            placeholders_.insert(tileMutexId.placeholder.get());
        }
    }

    static void ValidateMutexId(int64_t mutexId, const Span& span)
    {
        PRO_PASS_INTERNAL_CHECK_WITH_SPAN(npu::tile_fwk::InternalError::PASS_INNER_ERROR,
                                          mutexId >= 0 && mutexId < kMutexIdCount, span)
            << "mutex ID must be in [0, 31]";
    }

    std::vector<AliasComponent> BuildAliasComponents() const
    {
        std::vector<size_t> ordered(tiles_.size());
        std::iota(ordered.begin(), ordered.end(), 0);
        std::sort(ordered.begin(), ordered.end(), [this](size_t left, size_t right) {
            const auto& lhs = tiles_[left];
            const auto& rhs = tiles_[right];
            return std::tie(lhs.memorySpace, lhs.start, lhs.end, left) <
                   std::tie(rhs.memorySpace, rhs.start, rhs.end, right);
        });

        // Sweep sorted half-open ranges. Direct and transitive overlaps become one component,
        // whose Tiles must ultimately share one mutex ID.
        std::vector<AliasComponent> components;
        for (size_t tileIndex : ordered) {
            const auto& tile = tiles_[tileIndex];
            bool beginsNew = components.empty() || components.back().memorySpace != tile.memorySpace ||
                             tile.start >= components.back().end;
            if (beginsNew) {
                components.push_back(AliasComponent{{}, tile.memorySpace, tile.start, tile.end});
            }
            auto& component = components.back();
            component.tiles.push_back(tileIndex);
            component.end = std::max(component.end, tile.end);
        }
        return components;
    }

    void ValidateCrossingRanges(const std::vector<AliasComponent>& components)
    {
        for (const auto& component : components) {
            bool containsAuto = std::any_of(component.tiles.begin(), component.tiles.end(),
                                            [this](size_t index) { return tiles_[index].IsAuto(); });
            if (!containsAuto) {
                continue;
            }
            // Equal or nested ranges have an unambiguous shared owner. Crossing ranges only
            // partially overlap and are unsupported because they can trample each other.
            for (size_t leftPos = 0; leftPos < component.tiles.size(); ++leftPos) {
                const auto& left = tiles_[component.tiles[leftPos]];
                for (size_t rightPos = leftPos + 1; rightPos < component.tiles.size(); ++rightPos) {
                    const auto& right = tiles_[component.tiles[rightPos]];
                    bool overlaps = left.start < right.end && right.start < left.end;
                    bool contains = (left.start <= right.start && right.end <= left.end) ||
                                    (right.start <= left.start && left.end <= right.end);
                    if (overlaps && !contains) {
                        int64_t overlapStart = std::max(left.start, right.start);
                        int64_t overlapEnd = std::min(left.end, right.end);
                        FailInvalidOperation(
                            std::string("mutex_ids=\"auto\" does not support crossing Tile address ranges: ") +
                                "TileGroup '" + left.groupName + "' at line " + std::to_string(left.span.BeginLine()) +
                                " uses " + HexRange(left.start, left.end) + "; TileGroup '" + right.groupName +
                                "' at line " + std::to_string(right.span.BeginLine()) + " uses " +
                                HexRange(right.start, right.end) + "; overlap is " +
                                HexRange(overlapStart, overlapEnd) + "; possible buffer address trampling",
                            right.span);
                    }
                }
            }
        }
    }

    std::vector<size_t> GetManualTileIndices(const AliasComponent& component) const
    {
        std::vector<size_t> manualTiles;
        for (size_t tileIndex : component.tiles) {
            if (!tiles_[tileIndex].IsAuto()) {
                manualTiles.push_back(tileIndex);
            }
        }
        return manualTiles;
    }

    void ValidateManualTiles(const AliasComponent& component, const std::vector<size_t>& manualTiles)
    {
        // A manual Tile pins its whole overlap component. Such a pin must be a single ID, and all
        // manual Tiles in that component must agree on it.
        for (size_t tileIndex : manualTiles) {
            const auto& tile = tiles_[tileIndex];
            if (tile.manualIds.size() != 1) {
                FailInvalidArgument(
                    std::string("when address-overlapping Tiles include a manually configured Tile, that Tile ") +
                        "must have exactly one manual mutex ID; TileGroup '" + tile.groupName + "' at line " +
                        std::to_string(tile.span.BeginLine()) +
                        " has an overlapping Tile with multiple mutex IDs: mutex_ids=" + FormatIds(tile.manualIds),
                    tile.span);
            }
        }

        const auto& expectedTile = tiles_[manualTiles.front()];
        int expected = expectedTile.manualIds.front();
        for (size_t tileIndex : manualTiles) {
            const auto& tile = tiles_[tileIndex];
            if (tile.manualIds.front() != expected) {
                FailInvalidOperation(
                    std::string("manual Tiles connected through overlapping address ranges must use the same ") +
                        "mutex ID: TileGroup '" + expectedTile.groupName + "' at line " +
                        std::to_string(expectedTile.span.BeginLine()) + " uses " +
                        HexRange(expectedTile.start, expectedTile.end) + " with mutex_id=" + std::to_string(expected) +
                        "; TileGroup '" + tile.groupName + "' at line " + std::to_string(tile.span.BeginLine()) +
                        " uses " + HexRange(tile.start, tile.end) +
                        " with mutex_id=" + std::to_string(tile.manualIds.front()) + "; address-overlap component is " +
                        HexRange(component.start, component.end),
                    tile.span);
            }
        }
    }

    std::vector<int> GetCandidateIdsForAutoComponent(const AliasComponent& component)
    {
        auto manualTiles = GetManualTileIndices(component);
        if (manualTiles.empty()) {
            // With no manual pin, an AUTO component may choose from the entire hardware ID pool.
            std::vector<int> all(kMutexIdCount);
            std::iota(all.begin(), all.end(), 0);
            return all;
        }

        ValidateManualTiles(component, manualTiles);
        return {tiles_[manualTiles.front()].manualIds.front()};
    }

    static std::string FormatIds(const std::vector<int>& ids)
    {
        std::ostringstream os;
        os << '[';
        for (size_t index = 0; index < ids.size(); ++index) {
            if (index != 0) {
                os << ", ";
            }
            os << ids[index];
        }
        os << ']';
        return os.str();
    }

    AllocationState BuildAllocationState()
    {
        AllocationState state;
        state.components = BuildAliasComponents();
        ValidateCrossingRanges(state.components);

        // Account for fixed IDs before AUTO allocation so the load-balancing score includes
        // already occupied manual IDs.
        for (const auto& tile : tiles_) {
            if (!tile.IsAuto()) {
                for (int mutexId : tile.manualIds) {
                    ++state.tileUsageById[mutexId];
                }
            }
        }

        return state;
    }

    std::vector<MutexJob> BuildAllocationJobs(const AllocationState& state)
    {
        std::vector<MutexJob> jobs;
        for (size_t componentIndex = 0; componentIndex < state.components.size(); ++componentIndex) {
            std::vector<size_t> autoTiles;
            for (size_t tileIndex : state.components[componentIndex].tiles) {
                if (tiles_[tileIndex].IsAuto()) {
                    autoTiles.push_back(tileIndex);
                }
            }
            if (!autoTiles.empty()) {
                // All AUTO Tiles in one overlap component are allocated together and receive the
                // same selected ID.
                jobs.push_back(MutexJob{componentIndex, std::move(autoTiles),
                                        GetCandidateIdsForAutoComponent(state.components[componentIndex])});
            }
        }
        return jobs;
    }

    static void SortAllocationJobs(std::vector<MutexJob>& jobs)
    {
        // Allocate manually pinned jobs first, then larger components.
        auto priority = [](const MutexJob& job) {
            return std::make_tuple(job.candidates.size(), -static_cast<int>(job.autoTiles.size()), job.component);
        };
        std::sort(jobs.begin(), jobs.end(), [&priority](const MutexJob& left, const MutexJob& right) {
            return priority(left) < priority(right);
        });
    }

    static int ChooseMutexId(const MutexJob& job, const AllocationState& state)
    {
        // Balance Tile/component usage, with the numeric ID as a stable tie-break.
        auto score = [&state](int mutexId) {
            return std::make_tuple(state.tileUsageById[mutexId], state.componentUsageById[mutexId], mutexId);
        };
        return *std::min_element(job.candidates.begin(), job.candidates.end(),
                                 [&score](int left, int right) { return score(left) < score(right); });
    }

    std::vector<std::optional<int>> AllocateJobs(AllocationState& state, const std::vector<MutexJob>& jobs) const
    {
        std::vector<std::optional<int>> assignedIdByTile(tiles_.size());
        for (const auto& job : jobs) {
            int chosen = ChooseMutexId(job, state);
            // Component members alias one address region and therefore share the chosen ID.
            for (size_t tileIndex : job.autoTiles) {
                assignedIdByTile[tileIndex] = chosen;
            }
            state.tileUsageById[chosen] += static_cast<int>(job.autoTiles.size());
            ++state.componentUsageById[chosen];
        }
        return assignedIdByTile;
    }

    std::vector<std::optional<int>> AllocateMutexIds()
    {
        auto state = BuildAllocationState();
        auto jobs = BuildAllocationJobs(state);
        SortAllocationJobs(jobs);
        return AllocateJobs(state, jobs);
    }

    void FailInvalidArgument(const std::string& message, const Span& span)
    {
        diagnosticSpan_ = span;
        PRO_IR_THROW(ValueError, ExternalError::INVALID_ARGUMENT) << message;
    }

    void FailInvalidOperation(const std::string& message, const Span& span)
    {
        diagnosticSpan_ = span;
        PRO_IR_THROW(ValueError, ExternalError::INVALID_OPERATION) << message;
    }

    std::vector<TileRecord> tiles_;
    std::unordered_map<const Expr*, size_t> tileToIndex_;
    std::unordered_set<const Var*> placeholders_;
    bool hasAuto_ = false;
    Span diagnosticSpan_ = Span::Unknown();
};

AutoMutexIdManager::AutoMutexIdManager() : impl_(std::make_unique<Impl>()) {}

AutoMutexIdManager::~AutoMutexIdManager() = default;

void AutoMutexIdManager::CollectGroup(const std::vector<ExprPtr>& tiles, const MakeTuplePtr& mutexIds,
                                      const std::string& groupName)
{
    impl_->CollectGroup(tiles, mutexIds, groupName);
}

ProgramPtr AutoMutexIdManager::AssignMutexIds(const ProgramPtr& program) { return impl_->AssignMutexIds(program); }

Span AutoMutexIdManager::DiagnosticSpan() const { return impl_->DiagnosticSpan(); }

} // namespace ir
} // namespace pypto
