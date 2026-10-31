# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details.
# You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import enum
from typing import TYPE_CHECKING, NamedTuple, overload

from pypto import pypto_impl
from pypto.ir import AtomicType, CmpMode

from ._api import _api_decl

if TYPE_CHECKING:
    import torch.distributed as dist

__all__ = [
    "create_window",
    "dump_topology",
    "barrier",
    "make_dtensor",
    "get",
    "get_tile",
    "put",
    "put_tile",
    "signal",
    "notify",
    "wait",
    "notify_all",
    "channel",
    "atomic_add",
    "atomic_sub",
    "atomic_exchange",
    "atomic_cas",
    "Window",
    "DTensor",
    "Signal",
    "Channel",
    "CHANNEL_UB_BYTES",
    "NotifyOp",
    "Cmp",
]

Window = pypto_impl.dist.Window

# Signal comparison modes for pld.wait (alias of the IR CmpMode; the int values cross
# to the device helper whose case order is locked to this declaration order).
Cmp = CmpMode


class NotifyOp(enum.IntEnum):
    """Signal write mode of :func:`notify`.

    ``SET`` overwrites the slot with ``val``; ``ATOMIC_ADD`` fetch-adds ``val`` — the
    counter for ``wait`` (producers add 1, the consumer waits with ``Cmp.GE``).
    """

    SET = 0
    ATOMIC_ADD = 1


class DTensor:
    """Parse-time handle of a typed window-payload view (device-only).

    Built only by :func:`make_dtensor` inside a kernel; the transfer ops dissolve it
    into their IR ops, so it has no runtime representation.
    """

    __slots__ = ("window", "byte_offset", "shape", "dtype")

    def __init__(self, window, byte_offset, shape, dtype):
        self.window = window
        self.byte_offset = byte_offset
        self.shape = shape
        self.dtype = dtype


class Signal:
    """Parse-time handle of a window signal slot (device-only).

    Built only by :func:`signal` inside a kernel; ``notify`` / ``wait`` dissolve it
    into their IR ops, so it has no runtime representation.
    """

    __slots__ = ("window", "slot")

    def __init__(self, window, slot):
        self.window = window
        self.slot = slot


CHANNEL_UB_BYTES = 256  # one channel's UB scratch (peer descriptor + jetty bookkeeping + SQE)


class Channel(NamedTuple):
    """Parse-time handle of one write-engine channel (device-only).

    Built only by :func:`channel` inside a kernel. ``peer`` is ``None`` for the
    multi-issuer form. The parser intercepts the methods below; like every
    ``@_api_decl`` API they raise outside a kernel.
    """

    window: object  # kernel pld.Window parameter (ir.Var of WindowType)
    peer: object  # None (multi-issuer) or the bound peer rank (int / integer scalar expr)
    ub_addr: int  # UB scratch address of this channel (>= CHANNEL_UB_BYTES bytes, 64-B aligned)
    token: object  # IR Var (int64) produced by dist.channel

    # -- declarations only: the parser routes every call via dist_ops.parse_channel_method

    @overload
    def write(self, peer: int, dtensor: DTensor, src, /, *, submit: bool = True,
              timeout: int | None = None) -> None: ...
    @overload
    def write(self, dtensor: DTensor, src, /, *, submit: bool = True,
              timeout: int | None = None) -> None: ...
    @_api_decl
    def write(self, *args, submit: bool = True, timeout: int | None = None) -> None:
        """Async write into the peer's window (device-only).

        ``src`` is a local GM tensor (<= 256 MiB, byte length a multiple of 32) or one
        scalar word of the view dtype. ``submit=False`` stages the WQE for a later
        :meth:`submit`; ``True`` (default) publishes it. On a peer-less channel the
        destination leads: ``ch.write(peer, dtensor, src)``.
        """

    @overload
    def write_notify(self, peer: int, dtensor: DTensor, src, signal: Signal, val, /, *,
                     timeout: int | None = None) -> None: ...
    @overload
    def write_notify(self, dtensor: DTensor, src, signal: Signal, val, /, *,
                     timeout: int | None = None) -> None: ...
    @_api_decl
    def write_notify(self, *args, timeout: int | None = None) -> None:
        """Fused write whose completion writes the signal in hardware (device-only).

        Hardware writes ``val`` into ``signal``'s slot when the payload lands, so data
        is visible before the signal — never spell this as ``write`` + ``notify``.
        ``src`` must be a local GM tensor (no scalar form).
        """

    @_api_decl
    def submit(self) -> None:
        """Publish the staged WQEs (doorbell). Scope follows :meth:`is_exclusive`."""

    @_api_decl
    def drain(self, timeout: int | None = None) -> None:
        """Wait for engine completion of this channel's writes (device-only).

        Same scope as :meth:`submit`. ``timeout`` is a compile-time device-cycle
        bound; expiry traps the AIV. ``None`` waits forever.
        """

    @_api_decl
    def is_exclusive(self) -> bool:
        """True if the channel rides the calling AIV's own jetty (multi-issuer)."""

_L2_CACHE_LINE_SIZE = 512  # L2 cache line size


@_api_decl
def barrier(
    win: Window,
    timeout: int | None = None,
    *,
    sync_local: bool = True,
    flush_store: bool = True,
) -> None:
    """All-rank, all-AIV software barrier over ``win``'s group (device-only).

    Every rank must launch a kernel that reaches the barrier and every AIV must execute
    it (SPMD). Returns after the barrier completes on all ranks. The barrier does **not**
    drain the write engine — ``ch.drain()`` first for engine writes that must be visible
    to peers.

    Args:
        win: the window (carries the group and the barrier region).
        timeout: compile-time device-cycle bound (max 2^31-1); expiry traps the AIV.
            ``None`` waits forever.
        sync_local: include the local AIV-core exchange (default ``True``) — between
            this rank's AIV cores only, never between ranks. ``False`` leaves the
            arrival exchange to the leader AIV alone (single-AIV launches).
        flush_store: flush this AIV's outstanding MTE3 stores before the arrival
            exchange (default ``True``). The flush uses hardcoded event id 0 — do not
            keep MTE3->V event 0 in flight across the call. ``False`` skips the flush.
    """


@_api_decl
def make_dtensor(win: Window, offset, shape, dtype) -> DTensor:
    """Typed view of ``win``'s payload for one-sided transfers (device-only).

    Args:
        win: the owning window.
        offset: byte offset of the view start, relative to the payload base (naturally
            aligned for ``dtype``).
        shape: element shape, 1-5 dims; compile-time ints or in-kernel integer scalars.
        dtype: element dtype (must match the tile used in ``get``/``put``).
    """


@_api_decl
def get(tile, dtensor: DTensor, offset, peer: int, *, order=None) -> None:
    """Load a tile from a peer's window (device-only).

    Element-coordinate ``offset`` per view axis, same semantics as ``pl.load``.
    ``peer == win.rank()`` is a local GM copy.
    """


@_api_decl
def get_tile(tile, dtensor: DTensor, tile_offsets, peer: int, *, order=None) -> None:
    """Load a tile from a peer's window by tile-block index (device-only).

    One offset per view axis, like ``pl.load_tile``: mapped axes (``order``; trailing
    by default) take block indices, the rest element offsets.
    """


@_api_decl
def put(
    dtensor: DTensor,
    tile,
    offset,
    peer: int,
    *,
    order=None,
    atomic_mode: AtomicType = AtomicType.AtomicNone,
) -> None:
    """Store a tile into a peer's window (device-only).

    Element-coordinate ``offset`` per view axis, same semantics as ``pl.store``.
    ``peer == win.rank()`` is a local GM copy. Async MTE3: a following ``pld.barrier``
    or ``pld.notify`` is ordered after the data.

    Args:
        atomic_mode: ``pl.AtomicType.AtomicNone`` (default) overwrites; ``pl.AtomicType.AtomicAdd``
            atomically accumulates the tile into the peer's window instead. The natural
            collective: every rank puts its shard with ``AtomicAdd`` into every rank's
            window for an all-reduce sum. The destination must be pre-zeroed (host-side,
            before the launch). Supported dtypes: FP32/FP16/BF16/INT32/INT16/INT8.
    """


@_api_decl
def put_tile(
    dtensor: DTensor,
    tile,
    tile_offsets,
    peer: int,
    *,
    order=None,
    atomic_mode: AtomicType = AtomicType.AtomicNone,
) -> None:
    """Store a tile into a peer's window by tile-block index (device-only).

    One offset per view axis, like ``pl.store_tile``: mapped axes (``order``; trailing
    by default) take block indices, the rest element offsets.

    Args:
        atomic_mode: atomic write mode, as in :func:`put`.
    """


@_api_decl
def signal(win: Window, idx) -> Signal:
    """Build the handle of window signal slot ``idx`` (device-only).

    ``wait`` spins on the local window's slot; ``notify`` targets the same slot index
    on a peer's window. ``idx`` is a compile-time int or an in-kernel expression in
    ``[0, win.num_signals())`` (runtime bounds-checked).
    """


@_api_decl
def notify(signal: Signal, peer, val=None, *, op: NotifyOp = NotifyOp.SET) -> None:
    """Write a value into a signal slot (device-only).

    ``notify(signal, peer, val)`` writes the slot on ``peer``'s window;
    ``notify(signal, val)`` targets the local window's slot. ``op``: ``SET`` overwrites,
    ``ATOMIC_ADD`` fetch-adds. The internal MTE3 flush orders the signal after this
    AIV's prior ``put``\\ s.
    """


@_api_decl
def wait(signal: Signal, val, *, cmp: Cmp = Cmp.EQ, timeout: int | None = None) -> None:
    """Spin the calling AIV until the local signal slot satisfies ``cmp`` (device-only).

    Args:
        signal: slot handle built by :func:`signal` from the **local** window.
        val: int64 value to compare against.
        cmp: ``pld.Cmp.EQ / NE / LT / LE / GT / GE``.
        timeout: compile-time device-cycle bound (max 2^31-1); expiry traps the AIV.
            ``None`` waits forever.
    """


@_api_decl
def notify_all(signal: Signal, val, *, op: NotifyOp = NotifyOp.SET) -> None:
    """Signal ``signal``'s slot on **every** rank of the window group, the local one
    included (device-only).

    Parallel spelling of ``for r in range(win.nrank()): notify(signal, r, val)``: one
    SIMT fan-out writes all ranks' slots at once, one thread per rank, instead of a
    serial per-peer loop. Like :func:`notify`, the writes are async and ordered after
    this AIV's prior ``put``\\ s (an internal MTE3 flush, event id 0 — do not keep an
    MTE3->V event 0 in flight across the call); order explicitly (``barrier`` /
    ``wait``) before relying on a peer's arrival.

    Args:
        signal: slot handle built by :func:`signal` from the **local** window.
        val: int64 value to write.
        op: ``NotifyOp.SET`` overwrites, ``NotifyOp.ATOMIC_ADD`` fetch-adds.
    """


@_api_decl
def channel(win: Window, buffer: int, peer: int | None = None) -> Channel:
    """Open one write-engine channel of ``win``, backed by a UB scratch buffer (device-only).

    Must be the RHS of a simple assignment, outside control flow — the C++ object lives
    in the enclosing scope. Topology picks the jetty: without ``peer`` the channel rides
    this AIV's own jetty (multi-issuer; transfers lead with the destination,
    ``ch.write(peer, dtensor, src)``); with ``peer=k`` it rides peer ``k``'s jetty
    (per-peer; transfers omit the peer). A write is not ordered against ``notify`` —
    use ``drain`` first, or ``write_notify``.

    Args:
        win: the owning window.
        buffer: UB scratch address, >= ``pld.CHANNEL_UB_BYTES`` (256) bytes, 64-byte
            aligned, not overlapping any Tile or another channel's scratch. Compile-time.
        peer: destination rank for the per-peer form (compile-time or in-kernel integer).
            ``peer == win.rank()`` is rejected at runtime. Omit for multi-issuer.
    """


@_api_decl
def atomic_add(dtensor: DTensor, peer: int, offset, val) -> int:
    """Atomically add ``val`` to a word in a peer's window (device-only).

    Returns the word's pre-op value. ``offset`` is per-axis element coordinates (like
    :func:`get`) or a scalar linear element index. Dtypes: INT32/UINT32/INT64/UINT64/FP32,
    naturally aligned. Not ordered against get/put/jetty writes — order explicitly when
    data must be visible first.
    """


@_api_decl
def atomic_sub(dtensor: DTensor, peer: int, offset, val) -> int:
    """Atomically subtract ``val`` from a word in a peer's window (device-only).

    Wrapper for ``atomic_add(..., -val)`` — same contract; unsigned targets wrap
    modulo 2^N.
    """


@_api_decl
def atomic_exchange(dtensor: DTensor, peer: int, offset, val) -> int:
    """Atomically swap ``val`` into a word in a peer's window (device-only).

    Returns the pre-op value; same offset/dtype/alignment/ordering contract as
    :func:`atomic_add`.
    """


@_api_decl
def atomic_cas(dtensor: DTensor, peer: int, offset, expect, desired) -> int:
    """Atomic compare-and-swap on a word in a peer's window (device-only).

    Stores ``desired`` if the word equals ``expect``. Returns the pre-op value — compare
    it with ``expect`` to learn whether the swap happened; this is a single CAS, not a
    retry loop. Same offset/dtype/alignment/ordering contract as :func:`atomic_add`.
    """


def _align_up(x: int, alignment: int) -> int:
    return (x + alignment - 1) & ~(alignment - 1)


def _get_hccl_name(group: "dist.ProcessGroup") -> str:
    import torch
    import torch.distributed as dist

    # HCCL comm names are keyed by the GLOBAL rank, not the in-group rank (subgroup-safe).
    return group._get_backend(torch.device("npu")).get_hccl_comm_name(dist.get_rank())


def create_window(group: "dist.ProcessGroup", size: int, num_signals=15, name=None):
    """Create a symmetric window for communication between ranks in ``group``.

    All-rank collective: every rank must call it (same order across several windows).
    ``size`` is rounded up to a multiple of 512; the signal area is framework-managed
    and zeroed here.

    Args:
        group: ``torch.distributed`` process group with the ``hccl`` backend.
        size: payload size in bytes.
        num_signals: number of signal slots to reserve (default 15).
        name: optional window name, part of the registration uid.

    Returns:
        the :class:`Window`
    """
    import torch

    size = _align_up(size, _L2_CACHE_LINE_SIZE)  # C++ requires a 512B-aligned payload size
    signal_size = _L2_CACHE_LINE_SIZE * (num_signals + 1)  # extra space for signals
    buffer_size = _align_up(size + signal_size, 1 << 21)  # align to 2M
    device_buffer = torch.empty(buffer_size, dtype=torch.uint8, device='npu')

    rank = group.rank()
    win = pypto_impl.dist.create_window(
        rank,
        group.size(),
        size,
        buffer_size,
        num_signals,
        device_buffer.data_ptr(),
        torch.npu.current_stream().npu_stream,
        _get_hccl_name(group),
        name or "",
    )
    win.device_buffer = device_buffer
    return win


def dump_topology(group: "dist.ProcessGroup", layer: int = -1) -> str:
    """Dump the communication topology of ``group`` (host-side; not a collective).

    Queries the HCCL rank graph of the group's comm: per net layer the interconnect
    topology (e.g. ``1DMesh`` intra-server, ``CLOS`` inter-server), the ranks reachable
    from this rank on that layer, and the physical links to every peer (protocol, hop
    count, endpoint addresses).

    Args:
        group: ``torch.distributed`` process group with the ``hccl`` backend.
        layer: dump only this net layer (e.g. ``0`` for the intra-server mesh, ``1``
            for the inter-server network); ``-1`` (default) dumps every layer.
    """
    return pypto_impl.dist.dump_topology(_get_hccl_name(group), group.rank(), group.size(), layer)
