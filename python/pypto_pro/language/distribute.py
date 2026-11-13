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
from typing import TYPE_CHECKING, NamedTuple

from pypto import pypto_impl
from pypto.ir import CmpMode

from ._api import _api_decl
from .typing import Ptr

if TYPE_CHECKING:
    import torch.distributed as dist

__all__ = [
    "create_window",
    "register_memory",
    "dump_topology",
    "barrier",
    "notify",
    "wait",
    "channel",
    "remote_ptr",
    "Window",
    "Channel",
    "CHANNEL_UB_BYTES",
    "NotifyOp",
    "Protocol",
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


class Protocol(enum.IntFlag):
    """Engine plane selection for a registered memory region: UB_MEM (direct peer-VA),
    URMA (jetty/WQE). Combine with ``|`` to register on both planes."""

    UB_MEM = 1
    URMA = 2


CHANNEL_UB_BYTES = 256  # one channel's UB scratch (peer descriptor + jetty bookkeeping + SQE)


class Channel(NamedTuple):
    """Parse-time handle of one write-engine channel (device-only).

    Built only by :func:`channel` inside a kernel. The parser intercepts the
    methods below; like every ``@_api_decl`` API they raise outside a kernel.
    """

    window: object  # kernel pld.Window parameter (ir.Var of WindowType)
    ub_addr: int  # UB scratch address of this channel (>= CHANNEL_UB_BYTES bytes, 64-B aligned)
    token: object  # IR Var (int64) produced by dist.channel

    # -- declarations only: the parser routes every call via dist_ops.parse_channel_method

    @_api_decl
    def remote_ptr(self, tensor) -> Ptr:
        """URMA-plane address of a region registered on this channel (device-only).

        ``tensor`` is a kernel tensor parameter registered with
        ``Window.register_memory`` on the URMA plane, or a ``pl.Ptr`` inside such a
        tensor. The channel resolves the matching local buffer and returns the peer
        address of the same byte, typed like the source. Resolve once and reuse the
        result for :meth:`write` / :meth:`write_indexed`.

        The returned address is a peer VA for jetty WQEs; it is **not** dereferenceable
        by ``pl.load``/``pl.store``. Use ``pld.remote_ptr`` for the UB_MEM plane.
        """

    @_api_decl
    def write(self, dst, src, /, *, submit: bool = True,
              timeout: int | None = None) -> None:
        """Async write into the channel's peer window (device-only).

        The destination peer is fixed when the channel is opened
        (``pld.channel(..., peer=...)``); a peer-less channel rides the calling AIV's
        own jetty. ``dst`` is a remote pointer from :meth:`remote_ptr` (or advanced
        from one with pointer arithmetic). ``src`` is a local GM tensor (<= 256 MiB,
        byte length a multiple of 32) or one scalar word of the destination dtype.
        ``submit=False`` stages the WQE for a later :meth:`submit`; ``True`` (default)
        publishes it.
        """

    @_api_decl
    def write_indexed(self, dst, src, /, *, src_indices=None, dst_indices=None,
                      count: int, submit: bool = True, timeout: int | None = None) -> None:
        """Asynchronously write selected rows to a remote region.

        Copies ``count`` rows from ``src`` to the remote address ``dst`` using optional
        source and destination row indices. If an index table is omitted,
        sequential row indices ``[0, count)`` are used.

        Args:
            dst: Remote destination pointer from :meth:`remote_ptr`.
            src: Source tensor with the same dtype and row shape as the destination.
            src_indices: Optional 1-D UB Tile of source row indices.
            dst_indices: Optional 1-D UB Tile of destination row indices.
            count: Number of rows to transfer. Zero is a no-op.
            submit: If True, submit the generated WQEs immediately.
                Otherwise, queue them for a later ``submit()``.
            timeout: Optional timeout for queue operations.

        Notes:
            - At least one index table must be provided.
            - Indices are row numbers, not byte offsets. They must address rows
              of the corresponding tensor; an out-of-range index reads or writes
              outside the region.
            - Index tables must reside in UB (``pl.MemorySpace.Vec``) and be
              ready before the call. Load GM index tables into UB first.
            - Supported index dtypes: int32, int64, uint32, uint64. When both
              tables are present they must share one dtype.
            - Both tensors must be row-major and have contiguous rows. The row
              payload (axes after the row axis) must be a multiple of 32 bytes and
              at most 256 MiB.
            - Each row is one single-SGE WQEBB. A SIMT worker encodes the WQEs
              directly in the channel's GM send queue; only the 5-word SQE
              header is staged in UB. Rows are published in chunks of at most
              1024 (the jetty SQ depth must be at least 1024), one CQE per
              chunk; a chunk may wrap the SQ ring. Intermediate chunks are
              published when the queue is full even with ``submit=False``.
            - ``count`` must fit in int32; the device traps above ``INT32_MAX``.
            - The number of generated WQEs must fit the channel queue when
              ``submit=False``.
        """

    @_api_decl
    def write_notify(self, dst, src, signal, val, /, *,
                     timeout: int | None = None) -> None:
        """Fused write whose completion writes the signal in hardware (device-only).

        ``dst`` is a remote pointer from :meth:`remote_ptr`. ``signal`` is the
        peer's URMA address of a 64-bit signal word — resolve it with
        ``ch.remote_ptr(signal_tensor)`` (use an int64/uint64 tensor) and offset it
        with ``pl.addptr`` for a slot. The hardware notify value is a ``uint64_t``
        written as an 8-byte word, so the target must provide an aligned 64-bit
        word; a 32-bit pointer is rejected. The framework cannot verify that
        ``signal`` came from :meth:`remote_ptr`. Hardware writes ``val`` to that
        address when the payload lands, so data is visible before the signal —
        never spell this as ``write`` + ``notify``. ``src`` must be a local GM
        tensor (no scalar form).
        """

    @_api_decl
    def submit(self) -> None:
        """Publish the staged WQEs (doorbell). Scope follows :meth:`is_exclusive`."""

    @_api_decl
    def drain(self, timeout: int | None = None) -> None:
        """Wait for engine completion of this channel's writes (device-only).

        Same scope as :meth:`submit`. ``timeout`` is a compile-time device-cycle
        bound; expiry traps the AIV. ``None`` waits forever. WQEs staged with
        ``submit=False`` are not published implicitly.
        """

    @_api_decl
    def is_exclusive(self) -> bool:
        """True if the channel rides the calling AIV's own jetty (multi-issuer)."""

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
def remote_ptr(win: Window, tensor, peer: int) -> Ptr:
    """UB_MEM pointer to a registered region in a peer's window (device-only).

    Returns a typed pointer to the same byte of ``peer``'s copy of the region. The
    region and offset are resolved from the argument at run time, so it may point
    anywhere inside it, not only at its base::

        ptr = pld.remote_ptr(win, buf, peer)   # peer base
        view = pl.make_tensor(ptr, [m, n], dtype=pl.DT_FP32)
        pl.load(tile, view, [0, 0])
        pl.store(view, tile, [0, 0])               # or atomic=... for an accumulate

        ptr = pl.addptr(pl.make_ptr(buf), 4096)    # interior byte pointer
        off = pld.remote_ptr(win, ptr, peer)   # peer base + 4096

    The region must be registered on the UB_MEM plane; a URMA-only region traps at
    runtime. Use :meth:`Channel.remote_ptr` for the URMA (jetty write) plane.
    Pointer arithmetic is element-scaled; use a UINT8 pointer for byte offsets.

    Args:
        win: the owning window.
        tensor: a tensor kernel parameter the host registered on ``win``, or a
            ``pl.Ptr`` inside such a region.
        peer: destination rank (compile-time or in-kernel integer).
    """


@_api_decl
def notify(dst, val, *, op: NotifyOp = NotifyOp.SET, flush_store: bool = True) -> None:
    """Write one signal word at ``dst`` (device-only).

    Signals live in user-registered memory. Resolve the peer copy and apply the slot
    offset with pointer arithmetic. The word follows the pointer's element type
    (32/64-bit signed or unsigned)::

        sig = ...                                  # registered GM tensor, int64 slots
        remote = pld.remote_ptr(win, sig, peer)
        pld.notify(pl.addptr(remote, me), val=1)   # peer's copy, slot `me`

    ``op``: ``SET`` overwrites, ``ATOMIC_ADD`` fetch-adds. The write is async; order
    explicitly (``barrier`` / ``wait``) before relying on a peer's arrival.

    Args:
        dst: 64-bit integer pointer to the signal word (local or peer copy).
        val: value to write.
        op: ``NotifyOp.SET`` overwrites, ``NotifyOp.ATOMIC_ADD`` fetch-adds.
        flush_store: flush this AIV's outstanding MTE3 stores before writing the
            signal (default ``True``), so the signal is ordered after its prior
            ``put``\\ s. The flush uses hardcoded event id 0 — do not keep an MTE3
            event 0 in flight across the call. ``False`` skips it when the caller
            already ordered the data.
    """


@_api_decl
def wait(src, val, *, cmp: Cmp = Cmp.EQ, timeout: int | None = None) -> None:
    """Spin the calling AIV until the signal word at ``src`` satisfies ``cmp`` (device-only).

    ``src`` points at the **local** copy of the signal word; offset a slot with
    ``pl.addptr(pl.make_ptr(sig), slot)``.

    Args:
        src: 32/64-bit integer pointer to the signal word to poll.
        val: int64 value to compare against.
        cmp: ``pld.Cmp.EQ / NE / LT / LE / GT / GE``.
        timeout: compile-time device-cycle bound (max 2^31-1); expiry traps the AIV.
            ``None`` waits forever.
    """


@_api_decl
def channel(win: Window, buffer: int, /, *, peer: int | None = None) -> Channel:
    """Open one write-engine channel of ``win``, backed by a UB scratch buffer (device-only).

    Must be the RHS of a simple assignment, outside control flow — the C++ object lives
    in the enclosing scope. The window must have at least one URMA-registered region:
    the jetty is built from them, and constructing a channel on a window without any
    URMA region traps. The channel binds its destination when it is opened: with
    ``peer=k`` it rides peer ``k``'s jetty (per-peer); without ``peer`` it rides this
    AIV's own jetty (multi-issuer). :meth:`write` and :meth:`write_notify` take no
    peer. A write is not ordered against ``notify`` — use ``drain`` first, or
    ``write_notify``.

    Args:
        win: the owning window.
        buffer: UB scratch address, >= ``pld.CHANNEL_UB_BYTES`` (256) bytes, 64-byte
            aligned, not overlapping any Tile or another channel's scratch. The first
            128 bytes hold the peer/jetty headers; :meth:`write_indexed` keeps its SQE
            header above them. Compile-time.
        peer: destination rank for the per-peer form (compile-time or in-kernel integer).
            ``peer == win.rank()`` is rejected at runtime. Omit for multi-issuer.
    """


def _get_hccl_name(group: "dist.ProcessGroup") -> str:
    import torch
    import torch.distributed as dist

    # HCCL comm names are keyed by the GLOBAL rank, not the in-group rank (subgroup-safe).
    return group._get_backend(torch.device("npu")).get_hccl_comm_name(dist.get_rank())


def register_memory(win: Window, tensor, name, protocol: Protocol):
    """Register an existing NPU tensor's storage in place (zero-copy, local, before connect).

    Tensor form of :meth:`Window.register_memory`: validates the tensor and passes its
    address/size to the window. ``name`` is the engine memTag (every rank must register
    the same names); ``protocol`` selects the engine plane(s). The caller keeps owning
    the tensor and must keep it alive while the window is in use.
    """
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"tensor must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.device.type != "npu":
        raise ValueError(f"tensor must live on an NPU device, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError("tensor must be contiguous")

    size = tensor.numel() * tensor.element_size()
    win.register_memory(tensor.data_ptr(), size, name, int(protocol))


def create_window(group: "dist.ProcessGroup", name=None, layer: int = -1):
    """Create a communication window over ``group`` (local; not a collective).

    The window allocates only its framework-owned barrier region (one zeroed L2 line,
    registered as region 0). Payload memory is not allocated — register existing NPU
    tensors in place with :func:`register_memory` and keep them alive while the window
    is in use; signals are ordinary registered tensors too. Then call
    ``win.connect()`` (all-rank collective) before launching kernels.

    Region names are the engine memTags, so every rank must register the same names, and
    every window on a comm needs a distinct ``name``.

    Args:
        group: ``torch.distributed`` process group with the ``hccl`` backend.
        name: optional window name, used as the barrier-region tag prefix
            (``"<name>_barrier"``, or ``"<hccl_name>_barrier"`` when unnamed).
        layer: HCCL net layer for the window's channels; ``-1`` auto-selects the top
            layer, explicit ids are listed by :func:`dump_topology`.

    Returns:
        :class:`Window`: register payloads, then call :meth:`Window.connect`.
    """
    import torch

    return pypto_impl.dist.create_window(
        group.rank(),
        group.size(),
        torch.npu.current_stream().npu_stream,
        _get_hccl_name(group),
        name or "",
        layer,
    )


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
