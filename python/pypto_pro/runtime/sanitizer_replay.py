#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Host replay for the PyPTO Pro sanitizer: device log -> typed records -> findings.

One unique ``sanitizer_report.<launch_id>.bin`` per launch (written by the launch glue in
``runtime/jit.py`` from the GM log buffer) is decoded into records and
scanned by the registered checks. Every check value -- offsets, windows,
shapes, tile ranges, source lines -- travels in the records themselves;
the pass never decides, this file does.

The record types come from the C++ wire contract
(framework/include/ir/transforms/sanitizer_layout.h -- the single source
the encoder's MakeLogStmt field-count check also runs against), exported
to Python via pypto.pypto_impl.ir.sanitizer_record_layouts: each row is
one detection's wire fields in the exact order the C++ pass writes them,
plus the 'u'/'s' decode signedness. _record_type builds each frozen
record class from one row. One self-contained block per detection
follows: the check function that turns those records into findings. The
registries and the report-file I/O sit at the bottom. Adding a detection
= one C++ layout row + the instrumentation + one check function + one
DETECTIONS row.

Detections:
    1  GmAccess    GM_OUT_OF_BOUNDS    access window vs tensor shape
    2  TileAccess  TILE_OUT_OF_BOUNDS  transfer offsets vs tile dims
    3  MutexAccess MUTEX_*             lock/unlock pairing
    4  ViewShape   GM_OUT_OF_BOUNDS    make_tensor view vs source capacity
    5  TileScalar  TILE_OUT_OF_BOUNDS  getval/setval vs tile element count
    6  TileDecl    TILE_OVERLAP        tile address ranges must not intersect
    7  PhaseAccess PHASE_*             unit_flag state machine (K chains)
    8  SyncAccess  SYNC_*              sync_src/sync_dst alternation
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
import struct
from typing import Callable

from pypto.pypto_impl.ir import SANITIZER_RECORD_MAGIC as RECORD_MAGIC
from pypto.pypto_impl.ir import sanitizer_record_layouts

# ---------------------------------------------------------------------------
# Shared vocabulary (mirrors the C++ enums for readable reports)
# ---------------------------------------------------------------------------

# ir::PipeType order: MTE1=0 ... ALL=7.
PIPES = ("MTE1", "MTE2", "MTE3", "M", "V", "S", "FIX", "ALL")


def pipe_name(pipe: int) -> str:
    return PIPES[pipe] if 0 <= pipe < len(PIPES) else f"PIPE({pipe})"


# ir::MemorySpace values that appear in TileDecl records.
MEMORY_SPACES = {0: "DDR", 1: "Vec", 2: "Mat", 3: "Left", 4: "Right", 5: "Scaling",
                 6: "Acc", 7: "Bias", 8: "ScaleLeft", 9: "ScaleRight"}


def memory_space_name(space: int) -> str:
    return MEMORY_SPACES.get(space, f"space{space}")


# ---------------------------------------------------------------------------
# Findings and the replay context
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    kind: str
    message: str
    location: str = "<unknown>"
    hint: str = ""
    level: str = "ERROR"
    hits: int = 1  # how many records raised this exact diagnostic

    def __str__(self) -> str:
        loc = f" at {self.location}" if self.location else ""
        msg = f"{self.kind}: {self.message}{loc}"
        if self.hint:
            msg += f"\nHint: {self.hint}"
        return msg


class SanitizerReportError(ValueError):
    """The sanitizer report is truncated or otherwise cannot be trusted."""


class SanitizerReplayError(RuntimeError):
    """Aggregated sanitizer findings from one replay pass."""

    def __init__(self, findings: list[Finding]):
        self.findings = findings
        lines = [f"\nPyPTO Sanitizer Replay: {len(findings)} issue(s) detected\n"]
        for i, f in enumerate(findings, 1):
            lines.append(f"  [{i}/{len(findings)}] {f}")
            lines.append("")
        self.detail = "\n".join(lines)
        super().__init__(f"PyPTO Sanitizer Replay: {len(findings)} issue(s) detected")


class ReplayContext:
    """What the checks need beyond the records: the kernel's source file,
    so findings can name the offending line."""

    def __init__(self, source_file: str):
        self.source_file = source_file

    def location(self, line: int) -> str:
        return f"{self.source_file}:{line}" if line > 0 else "<unknown>"


# ---------------------------------------------------------------------------
# Wire records
#
# The wire contract (field names, order, signedness) lives in C++
# (framework/include/ir/transforms/sanitizer_layout.h) -- the single source
# the encoder's MakeLogStmt field-count check and this decoder both
# consume. It reaches Python through the pybind export
# pypto.pypto_impl.ir.sanitizer_record_layouts; _record_type builds each
# frozen namedtuple from one table row, appending the non-wire 'region'
# field (default 0, stamped by decode_regions, never written by the
# device).
# ---------------------------------------------------------------------------


def _record_type(det_id: int, name: str, fields, wire: str):
    defaults = (-1, 0, 0) if det_id == 7 else (0,)
    cls = collections.namedtuple(name, list(fields) + ["region"], defaults=defaults)
    cls.det_id = det_id
    cls.WIRE = wire
    cls.STRUCT = struct.Struct("<I" + "".join("I" if c == "u" else "i" for c in cls.WIRE))
    return cls


_LAYOUTS = sanitizer_record_layouts()
RECORD_TYPES: dict[int, type] = {
    det_id: _record_type(det_id, f"{name}Record", [f for f, _ in fields], "".join(k for _, k in fields))
    for det_id, (name, fields) in _LAYOUTS.items()
}
GmAccessRecord = RECORD_TYPES[1]
TileAccessRecord = RECORD_TYPES[2]
MutexAccessRecord = RECORD_TYPES[3]
ViewShapeRecord = RECORD_TYPES[4]
TileScalarRecord = RECORD_TYPES[5]
TileDeclRecord = RECORD_TYPES[6]
PhaseAccessRecord = RECORD_TYPES[7]
SyncAccessRecord = RECORD_TYPES[8]

# Field semantics, mirroring the C++ builders in sanitizer_pass.cpp:
#   1 GmAccess   RecordGmAccess: ndim = 0 marks a linear scalar access
#     (off0 = element index, acc_row = whole-tensor element count); for
#     ndim > 0 offsets and dims are axis-aligned, the tile's 2-D valid
#     window (acc_row/acc_col) applies to the tensor axes named by
#     win_ax0/win_ax1 (-1 = that tile dim maps to no axis), all other
#     dims are offset-only. Rank-1 access carries the whole-tile window
#     product in acc_row with win_ax0 = 0.
#   2 TileAccess RecordTileAccess: dims from the access-site tile type --
#     the runtime truth of the access, not the tile table.
#   3 MutexAccess RecordMutex: one record per mutex id of a deduped call;
#     region distinguishes independent sub-block mutex hardware slots.
#   4 ViewShape  CollectMakeTensor: the declared shape, its byte footprint
#     sum((dim-1)*stride)+1 scaled by the view dtype width, and the source
#     capacity (-1 = raw pointer: no known bound). All runtime values.
#   5 TileScalar RecordScalarAccess: linear element index vs dim_row*dim_col.
#   6 TileDecl   CollectMakeTile: numel is written by the device but only
#     addr/size/space matter here; it stays in the wire for compatibility.
#   7 PhaseAccess RecordPhase: op_kind 1-5 write side (matmul family,
#     dst Acc tile), 6-10 read side (store/move/insert, src Acc tile; 7/9
#     reserved and never emitted); phase 0/1/2 = Unspecified/Partial/Final
#     (AccPhase and STPhase share the values); acc_addr is the Acc tile's
#     memref addr (runtime value under TileGroup cursors) -- the pairing
#     group key together with the region; acc_src_addr is matmul_acc's
#     accumulator source tile addr (-1 for ops without one).
#   8 SyncAccess RecordSync: event_id is the runtime args[0] expression,
#     set_pipe/wait_pipe the enum kwargs, is_src marks the signal direction.


# ===========================================================================
# det 1 -- GM access bounds (GM_OUT_OF_BOUNDS)
# ===========================================================================


def check_gm_bounds(records: list[GmAccessRecord], ctx: ReplayContext) -> list[Finding]:
    findings: list[Finding] = []
    for rec in records:
        ndim = rec.ndim
        acc_row, acc_col = rec.acc_row, rec.acc_col
        if ndim == 0:
            # Linear scalar access: acc_row carries the element count.
            if acc_row <= 0:
                continue  # unknown (dynamic) bound
            if rec.off0 < 0 or rec.off0 + 1 > acc_row:
                findings.append(Finding(
                    "GM_OUT_OF_BOUNDS",
                    f"scalar access at linear offset {rec.off0} exceeds "
                    f"{acc_row} elements",
                    ctx.location(rec.line),
                    hint="getval/setval offsets must stay inside the tensor element count",
                ))
            continue
        off = (rec.off0, rec.off1, rec.off2, rec.off3, rec.off4)
        shape = (rec.dim0, rec.dim1, rec.dim2, rec.dim3, rec.dim4)
        details = []
        for i in range(ndim):
            dim_size = shape[i]
            if dim_size <= 0:
                continue  # unknown (dynamic) dim
            dim_off = off[i]
            if i == rec.win_ax0:
                dim_win = acc_row
            elif i == rec.win_ax1:
                dim_win = acc_col
            else:
                dim_win = 1
            if dim_off < 0:
                details.append(f"dim{i}: negative offset {dim_off}")
            elif dim_off + dim_win > dim_size:
                over = dim_off + dim_win - dim_size
                if dim_win == 1:
                    # Leading dimension: offset-only access (no tile window).
                    details.append(f"dim{i}: offset {dim_off} ≥ {dim_size} (over by {over})")
                else:
                    details.append(
                        f"dim{i}: offset {dim_off} + valid_shape {dim_win} = {dim_off + dim_win} "
                        f"> {dim_size} (over by {over})"
                    )
        if not details:
            continue
        access = f"offsets {list(off[:ndim])} + valid_shape [{acc_row}, {acc_col}]"
        msg = (f"{access} exceed tensor shape "
               f"{tuple(shape[:ndim])} — {'; '.join(details)}")
        findings.append(Finding(
            "GM_OUT_OF_BOUNDS",
            msg,
            ctx.location(rec.line),
            hint="The tile valid_shape plus the loop offsets must stay inside "
                 "the tensor logical shape on every dimension",
        ))
    return findings


# ===========================================================================
# det 2 -- tile transfer bounds (TILE_OUT_OF_BOUNDS)
# ===========================================================================


def check_tile_bounds(records: list[TileAccessRecord], ctx: ReplayContext) -> list[Finding]:
    findings: list[Finding] = []
    for rec in records:
        violations = []
        # A runtime offset is signed on the wire.  Check the lower bound before
        # the upper bound so a negative dynamic offset cannot hide a bad access.
        if rec.off_row < 0:
            violations.append(f"dim0: negative offset {rec.off_row}")
        if rec.off_col < 0:
            violations.append(f"dim1: negative offset {rec.off_col}")
        # Per-dimension logical bounds (0 = unknown, skipped).
        if rec.dim_row > 0 and rec.off_row + rec.acc_row > rec.dim_row:
            over = rec.off_row + rec.acc_row - rec.dim_row
            violations.append(f"dim0: offset {rec.off_row} + valid_shape {rec.acc_row} = "
                              f"{rec.off_row + rec.acc_row} > {rec.dim_row} (over by {over})")
        if rec.dim_col > 0 and rec.off_col + rec.acc_col > rec.dim_col:
            over = rec.off_col + rec.acc_col - rec.dim_col
            violations.append(f"dim1: offset {rec.off_col} + valid_shape {rec.acc_col} = "
                              f"{rec.off_col + rec.acc_col} > {rec.dim_col} (over by {over})")
        if not violations:
            continue
        findings.append(Finding(
            "TILE_OUT_OF_BOUNDS",
            f"offsets [{rec.off_row}, {rec.off_col}] + valid_shape [{rec.acc_row}, {rec.acc_col}] "
            f"exceed tile shape [{rec.dim_row}, {rec.dim_col}] — " + "; ".join(violations),
            ctx.location(rec.line),
            hint="Keep the tile's valid shape within the declared TileType dims",
        ))
    return findings


# ===========================================================================
# det 3 -- mutex pairing (MUTEX_*)
# ===========================================================================


def check_mutex_pairing(records: list[MutexAccessRecord], ctx: ReplayContext) -> list[Finding]:
    """Validate strict lock/unlock alternation per mutex execution key.

    A blocking mutex is not re-entrant: a second lock on the same pipe and
    mutex id stalls that pipe before the later unlock can execute.  Once that
    terminal error is found, later records for the same key are unreachable and
    are ignored to avoid cascading unlock findings.  Other keys continue to be
    checked normally.
    """
    findings: list[Finding] = []
    pending: dict[tuple, int] = {}
    blocked: set[tuple] = set()
    for rec in records:
        key = (rec.region, rec.mutex_id, rec.pipe)
        if key in blocked:
            continue
        if rec.is_lock:
            if key in pending:
                findings.append(Finding(
                    "MUTEX_DOUBLE_LOCK",
                    f"mutex_lock(mutex_id={rec.mutex_id}, pipe={pipe_name(rec.pipe)}, "
                    f"region={rec.region}) executes before the previous lock was "
                    "unlocked and will block this pipe permanently",
                    ctx.location(rec.line),
                    hint="A mutex key must alternate lock and unlock; do not lock "
                         "the same (region, mutex_id, pipe) twice",
                ))
                blocked.add(key)
                pending.pop(key, None)
            else:
                pending[key] = rec.line
            continue
        lock_line = pending.pop(key, None)
        if lock_line is None:
            findings.append(Finding(
                "MUTEX_UNLOCK_BEFORE_LOCK",
                f"mutex_unlock(mutex_id={rec.mutex_id}, pipe={pipe_name(rec.pipe)}, "
                f"region={rec.region}) without a preceding mutex_lock",
                ctx.location(rec.line),
                hint="Every mutex_unlock must be paired with a mutex_lock on the "
                     "same (region, mutex_id, pipe)",
            ))
    for key, lock_line in pending.items():
        region, mutex_id, pipe = key
        findings.append(Finding(
            "UNPAIRED_MUTEX_LOCK",
            f"mutex_lock(mutex_id={mutex_id}, pipe={pipe_name(pipe)}, region={region}) "
            "never unlocked",
            ctx.location(lock_line),
            hint="Every mutex_lock must be followed by a mutex_unlock on the same "
                 "(region, mutex_id, pipe)",
        ))
    return findings


# ===========================================================================
# det 4 -- view over source (GM_OUT_OF_BOUNDS, declaration-time)
# ===========================================================================


def check_view_over_source(records: list[ViewShapeRecord], ctx: ReplayContext) -> list[Finding]:
    """A view shares its source's storage, so an oversized declaration
    reads/writes past the source even for in-bounds accesses."""
    findings: list[Finding] = []
    for rec in records:
        if rec.src_bytes < 0 or rec.fp_bytes < 0:
            continue  # pointer source or unknown bound
        if rec.fp_bytes <= rec.src_bytes:
            continue
        dims = [rec.dim0, rec.dim1, rec.dim2, rec.dim3, rec.dim4]
        while dims and dims[-1] == 0:  # trailing pad zeros
            dims.pop()
        src_dims = [rec.src_dim0, rec.src_dim1, rec.src_dim2, rec.src_dim3, rec.src_dim4]
        while src_dims and src_dims[-1] == 0:
            src_dims.pop()
        view_desc = f"make_tensor view {dims}" if dims else "make_tensor view"
        src_desc = f"source tensor {src_dims}" if src_dims else "source tensor"
        # The covered bytes ride along: under wide strides the shape stays
        # innocent while the coverage explains the overflow.
        findings.append(Finding(
            "GM_OUT_OF_BOUNDS",
            f"{view_desc} covers {rec.fp_bytes} bytes, {src_desc} holds {rec.src_bytes} bytes",
            ctx.location(rec.line),
            hint="A view shares its source's storage; keep its shape/stride size within the source",
        ))
    return findings


# ===========================================================================
# det 5 -- tile scalar access (TILE_OUT_OF_BOUNDS)
# ===========================================================================


def check_tile_scalar_bounds(records: list[TileScalarRecord], ctx: ReplayContext) -> list[Finding]:
    findings: list[Finding] = []
    for rec in records:
        numel = rec.dim_row * rec.dim_col
        if rec.offset < 0 or rec.offset + 1 > numel:
            findings.append(Finding(
                "TILE_OUT_OF_BOUNDS",
                f"scalar tile access at linear offset {rec.offset} "
                f"exceeds {numel} elements (dims [{rec.dim_row},{rec.dim_col}])",
                ctx.location(rec.line),
                hint="getval/setval offsets must stay inside the tile element count",
            ))
    return findings


# ===========================================================================
# det 6 -- tile overlap (TILE_OVERLAP, declaration scan)
# ===========================================================================


def check_tile_overlap(records: list[TileDeclRecord], ctx: ReplayContext) -> list[Finding]:
    """Two make_tile declarations with intersecting address ranges in the
    same memory space. Identical re-registrations (the same statement
    copied into both programs of one kernel, or the same declaration seen
    from every region) share (addr, size, space, line) and collapse, while
    distinct statements with the same range keep both records -- exactly
    what the scan must report."""
    findings: list[Finding] = []
    decls = {}
    for rec in records:
        if rec.addr < 0 or rec.size <= 0:
            continue
        decls.setdefault((rec.addr, rec.size, rec.space, rec.line), rec)
    items = list(decls.values())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            ta, tb = items[i], items[j]
            if ta.space != tb.space:
                continue
            a0, sz_a = ta.addr, ta.size
            b0, sz_b = tb.addr, tb.size
            if not (a0 + sz_a <= b0 or b0 + sz_b <= a0):  # ranges intersect
                sp = memory_space_name(ta.space)
                findings.append(Finding(
                    "TILE_OVERLAP",
                    f"tile range [{sp} 0x{a0:x}, 0x{a0 + sz_a:x}) overlaps "
                    f"another tile [{sp} 0x{b0:x}, 0x{b0 + sz_b:x})",
                    ctx.location(ta.line),
                    hint="Two tiles must not share on-chip address space",
                ))
    return findings


# ===========================================================================
# det 7 -- unit_flag phase state machine (PHASE_*)
# ===========================================================================

# Phase record constants (mirror the C++ PhaseOpKind/AccPhase/STPhase).
PHASE_UNSPECIFIED, PHASE_PARTIAL, PHASE_FINAL = 0, 1, 2
PHASE_WRITE_KINDS = {1, 2, 3, 4, 5}   # matmul family (dst Acc tile)
PHASE_ACC_KINDS = {2, 5}              # matmul_acc / matmul_mx_acc: accumulate
PHASE_BIAS_KINDS = {3}                # matmul_bias: initializes L0C (chain head only)
PHASE_OVERWRITE_KINDS = {1, 4}        # matmul / matmul_mx: overwrite (chain head only)
PHASE_NO_ACC_SRC_ADDR = -1

_PHASE_MODE_UNKNOWN = 0
_PHASE_MODE_SOFTWARE = 1
_PHASE_MODE_HARDWARE = 2
_PHASE_EXPECT_WRITE = 0
_PHASE_EXPECT_READ = 1

@dataclass(frozen=True)
class _PhaseIssue:
    """A phase finding anchored to one exact record."""

    finding: Finding
    record_index: int


@dataclass(frozen=True)
class _PhaseCheckResult:
    """Result of a phase check that may make later records unreachable."""

    issue: _PhaseIssue | None = None
    fatal: bool = False


@dataclass
class _KChainState:
    round_has_write: bool = False
    chain_closed: bool = False


@dataclass(frozen=True)
class _PhaseToken:
    """One normalized record consumed by the unified phase scanner."""

    record_index: int
    record: PhaseAccessRecord
    is_write: bool
    is_phased: bool


@dataclass
class _PhaseAccState:
    """All state that belongs to one (region, acc_addr) timeline."""

    mode: int = _PHASE_MODE_UNKNOWN
    mode_mixed: bool = False
    seen_phased_write: bool = False
    seen_unspec_write: bool = False
    seen_unspec_read: bool = False
    flag: int = _PHASE_EXPECT_WRITE
    flag_set_by: tuple[int, PhaseAccessRecord] | None = None
    kchain: _KChainState = field(default_factory=_KChainState)
    blocked: bool = False


@dataclass
class _PhaseRegionState:
    """All phase timelines owned by one execution region."""

    acc_states: dict[int, _PhaseAccState] = field(default_factory=dict)


def _phase_issue(record_index: int, finding: Finding) -> _PhaseIssue:
    return _PhaseIssue(finding, record_index)


def check_phase_state_machine(records: list[PhaseAccessRecord], ctx: ReplayContext) -> list[Finding]:
    """Validate every phase record once, in source order.

    The scanner owns one independent state per ``(region, acc_addr)``.  The
    in-place matmul_acc contract means a different Acc address never seeds
    this timeline.  A unit_flag deadlock stops only the affected address;
    recoverable K-chain findings still update the physical state so later
    records remain meaningful.
    """
    issues: list[_PhaseIssue] = []
    regions: dict[int, _PhaseRegionState] = {}
    for record_index, rec in enumerate(records):
        token = _phase_token(record_index, rec)
        region_state = regions.setdefault(rec.region, _PhaseRegionState())
        acc_state = region_state.acc_states.setdefault(rec.acc_addr, _PhaseAccState())
        # The dst/acc identity rule is a local operation contract.  Check it
        # before reachability pruning so a later malformed matmul_acc is not
        # hidden by an earlier flag deadlock on the same address.
        contract_issue = _check_in_place_acc_contract(token, ctx)
        if contract_issue is not None:
            issues.append(contract_issue)
        if acc_state.blocked:
            continue

        mode_result = _check_mode_transition(acc_state, token, ctx)
        if mode_result.issue is not None:
            issues.append(mode_result.issue)
        if mode_result.fatal:
            acc_state.blocked = True
            continue

        # MIXED is a mode diagnostic, not a reason to discard hardware facts.
        # Phased records must keep checking and updating unit_flag after a
        # non-fatal mode boundary so a later flag violation remains visible.
        flag_result = _check_flag_permission(acc_state, token, ctx)
        if flag_result.issue is not None:
            issues.append(flag_result.issue)
        # A flag failure makes later records on this address unreachable, but
        # this record may violate the K-chain contract as well.  Evaluate the
        # remaining check before marking the timeline blocked so both findings
        # survive in the report, just as the device error record keeps all
        # diagnostics attached to one operation.
        k_issue = _check_k_shape(acc_state, token, ctx)
        if k_issue is not None:
            issues.append(k_issue)
        if not flag_result.fatal:
            _close_phase_round_if_complete(acc_state, token)
        if flag_result.fatal:
            acc_state.blocked = True
            continue

    for region_state in regions.values():
        for acc_state in region_state.acc_states.values():
            # A non-fatal mode boundary does not close the phased timeline.
            # Only an address-local fatal finding makes its EOF state
            # unreachable.
            if acc_state.blocked:
                continue
            terminal_issue = _terminal_flag_issue(acc_state, ctx)
            if terminal_issue is not None:
                issues.append(terminal_issue)
    return _merge_phase_issues(issues)


def _phase_token(record_index: int, rec: PhaseAccessRecord) -> _PhaseToken:
    return _PhaseToken(
        record_index=record_index,
        record=rec,
        is_write=rec.op_kind in PHASE_WRITE_KINDS,
        is_phased=rec.phase != PHASE_UNSPECIFIED,
    )


def _check_in_place_acc_contract(token: _PhaseToken, ctx: ReplayContext) -> _PhaseIssue | None:
    """Enforce the public matmul_acc/matmul_mx_acc in-place contract."""
    rec = token.record
    if rec.op_kind not in PHASE_ACC_KINDS:
        return None
    if rec.acc_src_addr == PHASE_NO_ACC_SRC_ADDR or rec.acc_src_addr == rec.acc_addr:
        return None
    tile_desc = f"region {rec.region}, Acc tile 0x{rec.acc_addr:x}"
    return _phase_issue(token.record_index, Finding(
        "PHASE_MISMATCH",
        f"matmul_acc writes {tile_desc} from accumulator tile 0x{rec.acc_src_addr:x}; "
        f"dst and acc must be the same Tile",
        ctx.location(rec.line),
        hint="Pass the same Acc Tile as both dst_tile and acc_tile",
    ))


def _mode_mix_write_issue(token: _PhaseToken, tile_desc: str,
                           ctx: ReplayContext) -> _PhaseIssue:
    rec = token.record
    return _phase_issue(token.record_index, Finding(
        "PHASE_MISMATCH",
        f"{tile_desc}: matmul ops mix phase and phase-less writes on one block -- the "
        f"hardware unit_flag handshake and the software sync do not order against "
        f"each other (silent wrong result)",
        ctx.location(rec.line),
        hint="A K chain must run under one sync regime: either every matmul block "
             "uses AccPhase, or none does",
    ))


def _mode_mix_read_issue(token: _PhaseToken, tile_desc: str,
                          ctx: ReplayContext) -> _PhaseIssue:
    rec = token.record
    return _phase_issue(token.record_index, Finding(
        "PHASE_MISMATCH",
        f"{tile_desc}: matmul uses phase (hardware unit_flag sync) while a store/move "
        f"on the same block omits STPhase (software sync) -- the two mechanisms mix "
        f"and Fixpipe may read data before the matmul finishes",
        ctx.location(rec.line),
        hint="When any matmul of a chain uses phase, the consuming store/move must use "
             "STPhase as well",
    ))


def _check_mode_transition(state: _PhaseAccState, token: _PhaseToken,
                           ctx: ReplayContext) -> _PhaseCheckResult:
    """Check a synchronization-mode transition within the current round."""
    rec = token.record
    issue: _PhaseIssue | None = None
    fatal = False
    tile_desc = f"region {rec.region}, Acc tile 0x{rec.acc_addr:x}"

    current_mode = _PHASE_MODE_HARDWARE if token.is_phased else _PHASE_MODE_SOFTWARE
    if not state.mode_mixed and state.mode not in (_PHASE_MODE_UNKNOWN, current_mode):
        if token.is_phased:
            # Keep the old candidate order: phased-write + phase-less-read
            # wins over write mixing when both begin at this record.
            if token.is_write and state.seen_unspec_read:
                issue = _mode_mix_read_issue(token, tile_desc, ctx)
            elif token.is_write and state.seen_unspec_write:
                issue = _mode_mix_write_issue(token, tile_desc, ctx)
            elif not token.is_write and state.seen_unspec_write:
                issue = _phase_issue(token.record_index, Finding(
                    "PHASE_DEADLOCK",
                    f"{tile_desc}: a phase-less matmul uses software sync, but a phased "
                    f"store/move waits for the hardware unit_flag that the matmul never sets",
                    ctx.location(rec.line),
                    hint="Use matching AccPhase/STPhase on both sides, or omit phase on both sides",
                ))
                fatal = True
        else:
            if token.is_write and state.seen_phased_write:
                issue = _mode_mix_write_issue(token, tile_desc, ctx)
            elif not token.is_write and state.seen_phased_write:
                issue = _mode_mix_read_issue(token, tile_desc, ctx)

    if token.is_phased:
        state.mode = _PHASE_MODE_HARDWARE
        if token.is_write:
            state.seen_phased_write = True
    else:
        state.mode = _PHASE_MODE_SOFTWARE
        if token.is_write:
            state.seen_unspec_write = True
        else:
            state.seen_unspec_read = True

    if issue is not None:
        state.mode_mixed = True
    return _PhaseCheckResult(issue, fatal)


def _check_flag_permission(state: _PhaseAccState, token: _PhaseToken,
                           ctx: ReplayContext) -> _PhaseCheckResult:
    """Validate the flag prefix and mark every violation as fatal."""
    if not token.is_phased:
        return _PhaseCheckResult()
    rec = token.record
    tile_desc = f"region {rec.region}, Acc tile 0x{rec.acc_addr:x}"
    if token.is_write:
        if state.flag != _PHASE_EXPECT_WRITE:
            return _PhaseCheckResult(
                _phase_issue(token.record_index, Finding(
                    "PHASE_DEADLOCK",
                    f"matmul write on {tile_desc} waits for the unit_flag to clear, but it is "
                    f"still set by an unconsumed Final write",
                    ctx.location(rec.line),
                    hint="Every Final write must be followed by a Final read before the next write",
                )),
                fatal=True,
            )
        if rec.phase == PHASE_FINAL:
            state.flag = _PHASE_EXPECT_READ
            state.flag_set_by = (token.record_index, rec)
        return _PhaseCheckResult()

    if state.flag != _PHASE_EXPECT_READ:
        return _PhaseCheckResult(
            _phase_issue(token.record_index, Finding(
                "PHASE_DEADLOCK",
                f"store/move read on {tile_desc} waits for the unit_flag to be set, but no "
                f"Final write has set it",
                ctx.location(rec.line),
                hint="The last matmul of the chain must use AccPhase.Final before an STPhase read",
            )),
            fatal=True,
        )
    if rec.phase == PHASE_FINAL:
        state.flag = _PHASE_EXPECT_WRITE
        state.flag_set_by = None
    return _PhaseCheckResult()


def _check_k_shape(state: _PhaseAccState, token: _PhaseToken,
                   ctx: ReplayContext) -> _PhaseIssue | None:
    """Check K roles and enforce the in-place matmul_acc contract."""
    rec = token.record
    kchain = state.kchain
    if not token.is_write:
        if rec.phase == PHASE_FINAL:
            kchain.chain_closed = True
        return None

    if kchain.chain_closed:
        kchain.chain_closed = False
        kchain.round_has_write = False

    if rec.phase == PHASE_UNSPECIFIED:
        kchain.round_has_write = True
        return None

    tile_desc = f"region {rec.region}, Acc tile 0x{rec.acc_addr:x}"
    issue = None
    if rec.op_kind in PHASE_ACC_KINDS:
        if not kchain.round_has_write:
            issue = _phase_issue(token.record_index, Finding(
                "PHASE_MISMATCH",
                f"matmul_acc is the first write of a chain on {tile_desc}: it accumulates "
                f"onto leftover or uninitialized L0C data instead of overwriting it",
                ctx.location(rec.line),
                hint="Start an in-place chain with matmul (overwrite), not matmul_acc",
            ))
    elif kchain.round_has_write and rec.op_kind in PHASE_BIAS_KINDS:
        issue = _phase_issue(token.record_index, Finding(
            "PHASE_MISMATCH",
            f"matmul_bias appears mid-chain on {tile_desc}: its L0C initialization "
            f"wipes the partial sums accumulated so far",
            ctx.location(rec.line),
            hint="matmul_bias is only valid as the first block of a K chain",
        ))
    elif kchain.round_has_write and rec.op_kind in PHASE_OVERWRITE_KINDS:
        issue = _phase_issue(token.record_index, Finding(
            "PHASE_MISMATCH",
            f"matmul appears mid-chain on {tile_desc}: its overwrite discards the "
            f"partial sums accumulated so far",
            ctx.location(rec.line),
            hint="Only the first block of a K chain may overwrite; use matmul_acc later",
        ))

    kchain.round_has_write = True
    return issue


def _close_phase_round_if_complete(state: _PhaseAccState, token: _PhaseToken) -> None:
    """Reset mode/K-chain facts only after a result has a valid consumer."""
    if token.is_write or state.flag != _PHASE_EXPECT_WRITE:
        return
    rec = token.record
    hardware_round_closed = token.is_phased and rec.phase == PHASE_FINAL
    software_round_closed = not token.is_phased and state.kchain.round_has_write
    if not (hardware_round_closed or software_round_closed):
        return
    state.mode = _PHASE_MODE_UNKNOWN
    state.mode_mixed = False
    state.seen_phased_write = False
    state.seen_unspec_write = False
    state.seen_unspec_read = False
    state.kchain = _KChainState()


def _terminal_flag_issue(state: _PhaseAccState, ctx: ReplayContext) -> _PhaseIssue | None:
    """Report a Final write whose flag remains set at end of the stream."""
    if state.flag_set_by is None:
        return None
    record_index, rec = state.flag_set_by
    tile_desc = f"region {rec.region}, Acc tile 0x{rec.acc_addr:x}"
    return _phase_issue(record_index, Finding(
        "PHASE_DEADLOCK",
        f"{tile_desc} ends with the unit_flag still set: the Final write has no Final read "
        f"to consume it",
        ctx.location(rec.line),
        hint="A Final write must be followed by a Final read (store/move) in the same chain",
    ))


def _merge_phase_issues(issues: list[_PhaseIssue]) -> list[Finding]:
    """Keep every diagnosis, in record and detector order."""
    ordered = sorted(enumerate(issues), key=lambda item: (item[1].record_index, item[0]))
    return [issue.finding for _, issue in ordered]

# ===========================================================================
# det 8 -- sync_src/sync_dst pairing (SYNC_*)
# ===========================================================================


def check_sync_pairing(records: list[SyncAccessRecord], ctx: ReplayContext) -> list[Finding]:
    """Strict alternation per (region, set_pipe, wait_pipe, event_id).

    sync is a one-way signal dependency -- src sets the flag, dst waits for
    it and clears it -- so within one key the records must strictly
    alternate src, dst, src, dst, ... (the same event id may be reused
    only after its dst has waited). The frontend already rejects invalid
    pipe combinations and out-of-range event ids at compile time; this
    check covers the runtime sequence errors:

    - SYNC_UNPAIRED_SRC: a src whose signal nobody waits for (data race:
      the consumer runs without ordering against the producer).
    - SYNC_UNPAIRED_DST: a dst with no preceding src (waits on a flag
      nobody sets -- deadlock on the device).
    - SYNC_EVENT_REUSE: a second src before the previous dst consumed it
      (overwrites the pending flag).
    - Parameter mismatches need no cross-comparison: a src and a dst with
      differing pipes/event fall into different keys and each surfaces as
      an unpaired endpoint.

    One pass over the records: the per-key state lives in a map instead of
    a group-then-iterate pair of phases. Within one key the state evolution
    depends only on that key's subsequence in order, so interleaved
    processing is equivalent to the grouped traversal. The terminal S-1
    sweep over the distinct keys is irreducible -- "nobody consumes the
    src" is only decidable at end of stream -- and points at the pending
    src's line, not at the stream head.
    """
    findings: list[Finding] = []
    # key -> [expecting_src, pending_src_line]
    state: dict[tuple, list] = {}
    for rec in records:
        key = (rec.region, rec.set_pipe, rec.wait_pipe, rec.event_id)
        st = state.get(key)
        if st is None:
            st = state[key] = [True, rec.line]
        if st[0]:  # expecting src
            if not rec.is_src:
                findings.append(Finding(
                    "SYNC_UNPAIRED_DST",
                    f"sync_dst on region {rec.region}, {rec.set_pipe} -> {rec.wait_pipe}, "
                    f"event_id {rec.event_id} waits for a flag no sync_src has set "
                    f"(deadlock: the wait never completes)",
                    ctx.location(rec.line),
                    hint="Every sync_dst must be preceded by a matching sync_src with the same "
                         "set_pipe, wait_pipe and event_id",
                ))
                # stay expecting src: this dst is still unmatched
            else:
                st[0] = False
                st[1] = rec.line
        else:  # expecting dst
            if rec.is_src:
                findings.append(Finding(
                    "SYNC_EVENT_REUSE",
                    f"sync_src on region {rec.region}, {rec.set_pipe} -> {rec.wait_pipe}, "
                    f"event_id {rec.event_id} fires again before the previous sync_dst "
                    f"consumed the event (the pending flag is overwritten)",
                    ctx.location(rec.line),
                    hint="The same event_id may only be reused after its sync_dst has "
                         "completed the wait",
                ))
                st[1] = rec.line  # the newest src is the pending one
            else:
                st[0] = True
    for (region, set_pipe, wait_pipe, event_id), (expecting_src, pending_src_line) in state.items():
        if not expecting_src:
            findings.append(Finding(
                "SYNC_UNPAIRED_SRC",
                f"sync_src on region {region}, {set_pipe} -> {wait_pipe}, "
                f"event_id {event_id} sets a flag no sync_dst waits for "
                f"(data race: the consumer runs unordered against the producer)",
                ctx.location(pending_src_line),
                hint="Every sync_src must be followed by a matching sync_dst with the same "
                     "set_pipe, wait_pipe and event_id",
            ))
    return findings


# ---------------------------------------------------------------------------
# Detection registry: the whole detection set in one glance, driving the
# scan (the record classes it names were built from _RECORD_SPECS above).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Detection:
    """One registered detection: the record class it consumes (which
    doubles as its det_id), the level every finding it raises is reported
    at, and the check itself."""
    record_cls: type
    level: str  # "ERROR" / "WARNING"
    check: Callable[[list, ReplayContext], list[Finding]]


DETECTIONS: list[Detection] = [
    Detection(GmAccessRecord, "ERROR", check_gm_bounds),
    Detection(TileAccessRecord, "ERROR", check_tile_bounds),
    Detection(TileScalarRecord, "ERROR", check_tile_scalar_bounds),
    Detection(ViewShapeRecord, "ERROR", check_view_over_source),
    Detection(MutexAccessRecord, "ERROR", check_mutex_pairing),
    Detection(TileDeclRecord, "WARNING", check_tile_overlap),
    Detection(PhaseAccessRecord, "ERROR", check_phase_state_machine),
    Detection(SyncAccessRecord, "ERROR", check_sync_pairing),
]


# ---------------------------------------------------------------------------
# Record decode
# ---------------------------------------------------------------------------


def decode_regions(region_chunks: list[bytes]) -> list:
    """Decode trusted, counter-bounded region chunks.

    A report that claims bytes which were not received, contains an unknown
    record id, or has a malformed record head is rejected.  Returning a clean
    result after silently skipping such data would turn transport corruption
    into a false "no issues" report.
    """
    records: list = []
    for region_idx, chunk in enumerate(region_chunks):
        if len(chunk) < 4:
            raise SanitizerReportError(
                f"region {region_idx}: missing 4-byte counter")
        (ctr,) = struct.unpack_from("<I", chunk, 0)
        available_words = (len(chunk) - 4) // 4
        if ctr > available_words:
            raise SanitizerReportError(
                f"region {region_idx}: counter claims {ctr} words, "
                f"but only {available_words} words were received")
        pos, end = 4, 4 + ctr * 4
        while pos < end:
            (word,) = struct.unpack_from("<I", chunk, pos)
            if (word >> 16) != RECORD_MAGIC:
                raise SanitizerReportError(
                    f"region {region_idx}: invalid record magic at byte {pos}")
            cls = RECORD_TYPES.get(word & 0xFFFF)
            if cls is None:
                raise SanitizerReportError(
                    f"region {region_idx}: unknown sanitizer record id {word & 0xFFFF}")
            if pos + cls.STRUCT.size > end:
                raise SanitizerReportError(
                    f"region {region_idx}: truncated record id {word & 0xFFFF}")
            rec = cls(*cls.STRUCT.unpack_from(chunk, pos)[1:])
            records.append(rec._replace(region=region_idx))
            pos += cls.STRUCT.size
    return records


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


def run_replay(records: list, ctx: ReplayContext) -> list[Finding]:
    """Run every registered check over the decoded records.

    Records are bucketed by type once (the type is the det_id), so each
    check receives only its own records. Repeats of the same diagnostic
    (kind, location, hint) collapse with a count, while dynamic addresses in
    the message do not turn one source defect into many reports. Different
    diagnostics raised by the same source record remain visible through their
    distinct hints.
    """
    buckets: dict[type, list] = {}
    for rec in records:
        buckets.setdefault(type(rec), []).append(rec)
    by_key: dict[tuple, Finding] = {}
    for det in DETECTIONS:
        for f in det.check(buckets.get(det.record_cls, []), ctx):
            f.level = det.level
            key = (f.kind, f.location, f.hint)
            if key in by_key:
                by_key[key].hits += 1
            else:
                f.hits = 1
                by_key[key] = f
    return [
        f if f.hits == 1
        else Finding(f.kind, f"{f.message}  (same defect hit {f.hits} times "
                                f"at this location)", f.location, f.hint, f.level,
                     f.hits)
        for f in by_key.values()
    ]


# ---------------------------------------------------------------------------
# Report file (sanitizer_report.<launch_id>.bin)
#
#   u32 region_count | per region: u32 chunk_bytes + chunk
#
# Each chunk is one device region trimmed to its written part. An
# immediate write-then-scan round trip inside one launch call, not a
# persistence contract -- no magic or version guard. Every check value
# travels in the records.
# ---------------------------------------------------------------------------


def build_report_file(region_chunks: list[bytes], path: str) -> None:
    """Write the report file: per-region written chunks."""
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(region_chunks)))
        for chunk in region_chunks:
            f.write(struct.pack("<I", len(chunk)))
            f.write(chunk)


def scan_report_file(path: str, source_file: str) -> list[Finding]:
    """Scan one report file and reject any truncated or trailing bytes."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 4:
        raise SanitizerReportError("report is missing the region count")
    (region_count,) = struct.unpack_from("<I", data, 0)
    pos = 4
    region_chunks = []
    for region_idx in range(region_count):
        if pos + 4 > len(data):
            raise SanitizerReportError(
                f"report is truncated before region {region_idx} length")
        (chunk_len,) = struct.unpack_from("<I", data, pos)
        pos += 4
        if chunk_len > len(data) - pos:
            raise SanitizerReportError(
                f"report region {region_idx} claims {chunk_len} bytes, "
                f"but only {len(data) - pos} remain")
        region_chunks.append(data[pos:pos + chunk_len])
        pos += chunk_len
    if pos != len(data):
        raise SanitizerReportError(
            f"report has {len(data) - pos} trailing bytes after {region_count} regions")
    return run_replay(decode_regions(region_chunks), ReplayContext(source_file))
