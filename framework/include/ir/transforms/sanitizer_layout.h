/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PYPTO_IR_TRANSFORMS_SANITIZER_LAYOUT_H_
#define PYPTO_IR_TRANSFORMS_SANITIZER_LAYOUT_H_

#include <cstdint>

namespace pypto {
namespace ir {

/// Sanitizer record detection ids on the wire; each owns the one record
/// layout declared below. This header is the single source of the wire
/// contract, consumed by the encoder (sanitizer_pass.cpp checks every
/// MakeLogStmt call against it), the pybind export (bindings/ir/pass.cpp
/// hands it to Python) and the host decoder
/// (pypto_pro/runtime/sanitizer_replay decodes by it).
///
/// Version discipline: append new ids and new semantic fields at the tail of
/// their class only; never renumber or reorder existing ones (device, host and
/// report consumers may come from different builds).  The source-line field is
/// a fixed MakeLogStmt trailer and is intentionally kept last for every record.
enum class SanitizerDetection : uint32_t {
    GmAccess = 1,    // GM access: ndim, off[5], shape[5], acc_row, acc_col, win_ax[2]
    TileAccess = 2,  // tile access: dims, off_row/col, acc_row/col
    MutexAccess = 3, // mutex lock/unlock: mutex_id, pipe, is_lock
    ViewShape = 4,   // runtime view shape: dim[5], src_dim[5], fp/src bytes
    TileScalar = 5,  // tile scalar access: dims, linear offset
    TileDecl = 6,    // make_tile declaration: addr, size, space, numel
    PhaseAccess = 7, // unit_flag phase: op_kind, phase, acc_addr/acc_src_addr (Acc memref addrs), line
    SyncAccess = 8,  // sync_src/sync_dst: event_id, set_pipe, wait_pipe, is_src
};

/// One wire field: its name (the record/report vocabulary) and its decode
/// signedness ('u' unsigned u32, 's' signed i32). The device-side write is
/// a plain int32 either way; the kind only picks how the host decoder
/// reads the word back.
struct SanitizerFieldLayout {
    const char* name;
    char kind;
};

/// Record-head magic: the first word of every record packs the record magic in the
/// high 16 bits and the det_id in the low 16 bits. The decoder verifies the
/// magic word-by-word and rejects malformed or unknown records; it never
/// presents a partially decoded report as a clean result.

constexpr uint32_t kSanitizerRecordMagic = 0x5A5AU;

/// Pack the record-head word from a det id.
constexpr uint32_t SanitizerRecordHead(uint32_t detId) { return (kSanitizerRecordMagic << 16U) | detId; }

/// One record layout: the detection it belongs to, its vocabulary name,
/// and the full wire field list in the exact order the encoder writes
/// them. MakeLogStmt appends the trailing "line" field itself.
struct SanitizerRecordLayout {
    SanitizerDetection det;
    const char* name;
    const SanitizerFieldLayout* wire;
    uint32_t wire_count; // including the trailing line
};

constexpr SanitizerFieldLayout kGmAccessWire[] = {
    {"ndim", 'u'},    {"off0", 's'},    {"off1", 's'},    {"off2", 's'}, {"off3", 's'}, {"off4", 's'},
    {"dim0", 's'},    {"dim1", 's'},    {"dim2", 's'},    {"dim3", 's'}, {"dim4", 's'}, {"acc_row", 'u'},
    {"acc_col", 'u'}, {"win_ax0", 's'}, {"win_ax1", 's'}, {"line", 'u'},
};

constexpr SanitizerFieldLayout kTileAccessWire[] = {
    {"dim_row", 'u'}, {"dim_col", 'u'}, {"off_row", 's'}, {"off_col", 's'},
    {"acc_row", 'u'}, {"acc_col", 'u'}, {"line", 'u'},
};

constexpr SanitizerFieldLayout kMutexAccessWire[] = {
    {"mutex_id", 'u'},
    {"pipe", 'u'},
    {"is_lock", 'u'},
    {"line", 'u'},
};

constexpr SanitizerFieldLayout kViewShapeWire[] = {
    {"dim0", 's'},     {"dim1", 's'},      {"dim2", 's'},     {"dim3", 's'},     {"dim4", 's'},
    {"src_dim0", 's'}, {"src_dim1", 's'},  {"src_dim2", 's'}, {"src_dim3", 's'}, {"src_dim4", 's'},
    {"fp_bytes", 's'}, {"src_bytes", 's'}, {"line", 'u'},
};

constexpr SanitizerFieldLayout kTileScalarWire[] = {
    {"dim_row", 'u'},
    {"dim_col", 'u'},
    {"offset", 's'},
    {"line", 'u'},
};

constexpr SanitizerFieldLayout kTileDeclWire[] = {
    {"addr", 's'}, {"size", 's'}, {"space", 'u'}, {"numel", 'u'}, {"line", 'u'},
};

constexpr SanitizerFieldLayout kPhaseAccessWire[] = {
    {"op_kind", 'u'}, {"phase", 'u'}, {"acc_addr", 's'}, {"acc_src_addr", 's'}, {"line", 'u'},
};

constexpr SanitizerFieldLayout kSyncAccessWire[] = {
    {"event_id", 's'}, {"set_pipe", 'u'}, {"wait_pipe", 'u'}, {"is_src", 'u'}, {"line", 'u'},
};

constexpr SanitizerRecordLayout kSanitizerRecordLayouts[] = {
    {SanitizerDetection::GmAccess, "GmAccess", kGmAccessWire, sizeof(kGmAccessWire) / sizeof(kGmAccessWire[0])},
    {SanitizerDetection::TileAccess, "TileAccess", kTileAccessWire,
     sizeof(kTileAccessWire) / sizeof(kTileAccessWire[0])},
    {SanitizerDetection::MutexAccess, "MutexAccess", kMutexAccessWire,
     sizeof(kMutexAccessWire) / sizeof(kMutexAccessWire[0])},
    {SanitizerDetection::ViewShape, "ViewShape", kViewShapeWire, sizeof(kViewShapeWire) / sizeof(kViewShapeWire[0])},
    {SanitizerDetection::TileScalar, "TileScalar", kTileScalarWire,
     sizeof(kTileScalarWire) / sizeof(kTileScalarWire[0])},
    {SanitizerDetection::TileDecl, "TileDecl", kTileDeclWire, sizeof(kTileDeclWire) / sizeof(kTileDeclWire[0])},
    {SanitizerDetection::PhaseAccess, "PhaseAccess", kPhaseAccessWire,
     sizeof(kPhaseAccessWire) / sizeof(kPhaseAccessWire[0])},
    {SanitizerDetection::SyncAccess, "SyncAccess", kSyncAccessWire,
     sizeof(kSyncAccessWire) / sizeof(kSyncAccessWire[0])},
};

/// Layout lookup by wire det_id; nullptr when the id is unknown (a newer
/// device build than this host, or corrupt data).
inline const SanitizerRecordLayout* FindSanitizerRecordLayout(uint32_t det_id)
{
    for (const auto& desc : kSanitizerRecordLayouts) {
        if (static_cast<uint32_t>(desc.det) == det_id) {
            return &desc;
        }
    }
    return nullptr;
}

} // namespace ir
} // namespace pypto

#endif // PYPTO_IR_TRANSFORMS_SANITIZER_LAYOUT_H_
