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

from ._api import _api_decl
from .typing import Ptr

if TYPE_CHECKING:
    import torch.distributed as dist

__all__ = [
    "create_window",
    "register_memory",
    "dump_topology",
    "barrier",
    "channel",
    "buffer_channel",
    "remote_ptr",
    "remote_tensor",
    "Window",
    "Channel",
    "BufferChannel",
    "CHANNEL_UB_BYTES",
    "Protocol",
    "UrmaOption",
    "ReduceOp",
]

Window = pypto_impl.dist.Window


class ReduceOp(enum.IntEnum):
    """Element-wise reduction mode of :meth:`Channel.write_reduce`.

    The values are the engine's wire codes; the declaration order is locked to the
    vendor enum.
    """

    MAX = 0x8
    MIN = 0x9
    SUM = 0xA


class Protocol(enum.IntFlag):
    """Engine plane selection for a registered memory region: UB_MEM (direct peer-VA),
    URMA (jetty/WQE). Combine with ``|`` to register on both planes."""

    UB_MEM = 1
    URMA = 2


class UrmaOption(enum.IntFlag):
    """URMA WQE control flags (subset of ``WqeEntry`` in hcomm_common.h).

    The value is the packed WQE flag byte (``odr:3, fence:1, cqe:1``), keeping the hcomm
    bit layout; the ``se`` and ``inlineEn`` bits are not exposed. These flags matter only
    on the UB_CTP/URMA plane — on RoCE only ``FENCE`` and ``CQE`` take effect.

    Attributes:
        ODR_NONE: ``odr[1:0]=00``; no execution-order requirement between WQEs.
        ODR_RELAX: ``odr[1:0]=01``; relaxed execution order — a later WQE may start before
            an earlier one completes. Mutually exclusive with ``ODR_STRONG``.
        ODR_STRONG: ``odr[1:0]=10``; strong execution order — WQEs execute in issue order.
            Mutually exclusive with ``ODR_RELAX``.
        ODR_IN_ORDER_COMPLETION: ``odr[2]=1``; completion records (CQEs) are reported in
            issue order.
        FENCE: the WQE waits for all previously issued WQEs to complete before it runs.
        CQE: request one completion record. Required for standalone channel operations.
            Buffered appends ignore this bit and request one completion per batch at submit.
        DEFAULT: hcomm's ``URMA_DEFAULT_CFG`` (``0x2D``): relaxed execution order,
            in-order completion, fenced, one completion record.
    """

    ODR_NONE = 0  # odr[1:0]=00: WQEs may execute in any order
    ODR_RELAX = 1 << 0  # odr[1:0]=01: relaxed execution order
    ODR_STRONG = 1 << 1  # odr[1:0]=10: strong (issue) execution order
    ODR_IN_ORDER_COMPLETION = 1 << 2  # odr[2]=1: CQEs reported in issue order
    FENCE = 1 << 3  # wait for all previously issued WQEs to complete
    CQE = 1 << 5  # generate one completion record per WQE

    DEFAULT = ODR_RELAX | ODR_IN_ORDER_COMPLETION | FENCE | CQE  # 0x2D, URMA_DEFAULT_CFG


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
              timeout: int | None = None,
              option: UrmaOption = UrmaOption.DEFAULT) -> None:
        """Async write into the channel's peer window (device-only).

        The destination peer is fixed when the channel is opened
        (``pld.channel(..., peer=...)``); a peer-less channel rides the calling AIV's
        own jetty. ``dst`` is a remote pointer from :meth:`remote_ptr` (or advanced
        from one with pointer arithmetic). ``src`` is a local GM tensor (<= 256 MiB,
        byte length a multiple of 32; larger transfers must be split by the caller —
        the limit is asserted in sanitizer builds) or one scalar word of the
        destination dtype. ``submit=False`` stages the WQE for a later
        :meth:`submit`; ``True`` (default) publishes it.

        ``option`` controls the WQE control flags of the issued WQEs. The scalar
        form always sets ``inlineEn`` (no SGE), so ``option`` only contributes its
        order/fence/completion bits there.
        """

    @_api_decl
    def read(self, dst, src, /, *, submit: bool = True,
             timeout: int | None = None,
             option: UrmaOption = UrmaOption.DEFAULT) -> None:
        """Async read from the channel's peer window into a local tensor (device-only).

        ``src`` is a remote pointer from :meth:`remote_ptr` (or advanced from one
        with pointer arithmetic); ``dst`` is a local GM tensor receiving the bytes
        (<= 256 MiB, byte length a multiple of 32; larger transfers must be split by
        the caller — the limit is asserted in sanitizer builds). The engine writes
        ``dst``, so its contents are readable only after :meth:`drain`.
        ``submit=False`` stages the WQE for a later :meth:`submit`; ``True``
        (default) publishes it.
        ``option`` controls the WQE control flags of the issued WQEs.
        """

    @_api_decl
    def write_reduce(self, dst, src, /, *, op: ReduceOp, submit: bool = True,
                     timeout: int | None = None,
                     option: UrmaOption = UrmaOption.DEFAULT) -> None:
        """Async engine-side reduction of a local tensor into the peer's window (device-only).

        Element-wise reduces ``src`` (a local GM tensor) into the remote ``dst``
        pointer per ``op`` — ``pld.ReduceOp.SUM/MAX/MIN``. Supported dtypes:
        int8/int16/int32/uint32/float16/float32/bfloat16. At most 256 MiB per call
        (one WQE; larger transfers must be split by the caller — the limit is
        asserted in sanitizer builds). Concurrent reduces from several ranks into
        one target are hardware-serialized; the result is not ordered against other
        engine ops — :meth:`drain` first.
        ``submit=False`` stages the WQE for a later :meth:`submit`. ``option``
        controls the WQE control flags; the engine's UDF flag for the reduce fields
        is always set on top of it.
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
            - To keep all WQEs queued until an explicit ``submit()``, the channel
              must have enough free capacity for the whole transfer.
        """

    @_api_decl
    def atomic_fetch_add(self, dst, fetch, val, /, *, submit: bool = True,
                   timeout: int | None = None,
                   option: UrmaOption = UrmaOption.DEFAULT) -> None:
        """Remote fetch-and-add of one aligned integer word (device-only).

        ``dst`` is a remote pointer to the target word (int32/uint32/int64/uint64,
        naturally aligned — misaligned destinations trap). The engine adds ``val``
        and writes the **previous** remote value into ``fetch``, a local one-element
        GM tensor of the same dtype — readable only after :meth:`drain`.
        ``submit=False`` stages the WQE for a later :meth:`submit`. ``option``
        controls the WQE control flags of the issued WQEs.
        """

    @_api_decl
    def atomic_cas(self, dst, fetch, expected, desired, /, *, submit: bool = True,
                   timeout: int | None = None,
                   option: UrmaOption = UrmaOption.DEFAULT) -> None:
        """Remote compare-and-swap of one aligned integer word (device-only).

        Swaps ``desired`` into ``dst`` when the remote word equals ``expected``
        and writes the **previous** value into ``fetch`` (local one-element tensor,
        same dtype) either way — success iff ``fetch == expected``. The target must
        be naturally aligned; misaligned destinations trap. Readable only after
        :meth:`drain`. ``submit=False`` stages the WQE for a later :meth:`submit`.
        ``option`` controls the WQE control flags of the issued WQEs.
        """

    @_api_decl
    def write_notify(self, dst, src, signal, val, /, *,
                     submit: bool = True,
                     timeout: int | None = None,
                     option: UrmaOption = UrmaOption.DEFAULT) -> None:
        """Fused write whose completion writes the signal in hardware (device-only).

        ``dst`` is a remote pointer from :meth:`remote_ptr`. ``signal`` is the
        peer's URMA address of a 64-bit signal word — resolve it with
        ``ch.remote_ptr(signal_tensor)`` (use an int64/uint64 tensor) and offset it
        with ``pl.addptr`` for a slot. The hardware notify value is a ``uint64_t``
        written as an 8-byte word, so the target must provide an aligned 64-bit
        word; a 32-bit pointer is rejected. The framework cannot verify that
        ``signal`` came from :meth:`remote_ptr`. Hardware writes ``val`` to that
        address when the payload lands, so data is visible before the signal —
        never spell this as a write followed by a separate signal write. ``src`` must be a local GM
        tensor of at most 256 MiB (larger transfers must be split by the caller —
        the limit is asserted in sanitizer builds) and has no scalar form.
        ``submit=False`` stages the WQE for a later :meth:`submit`; ``True``
        (default) publishes it. ``option`` controls the WQE control flags of the
        fused write WQE.
        """

    @_api_decl
    def submit(self) -> None:
        """Publish the staged WQEs (doorbell). Scope follows :meth:`is_shared`.

        ``submit=False`` defers publication only while the SQ and CQ have room: a post
        that would overflow either ring first publishes the WQEs staged so far.
        """

    @_api_decl
    def drain(self, timeout: int | None = None) -> None:
        """Wait for engine completion of this channel's writes (device-only).

        Same scope as :meth:`submit`. ``timeout`` is a compile-time device-cycle
        bound; expiry traps the AIV. ``None`` waits forever. :meth:`drain` never
        publishes WQEs staged with ``submit=False``: they need :meth:`submit`, or a
        later post that runs out of queue capacity.
        """

    @_api_decl
    def lock(self) -> None:
        """Take the channel's cross-AIV lock (device-only).

        On a shared window the channel rides the calling AIV's own jetty, so this returns
        immediately. Otherwise the per-peer jetty is shared by the AIVs and it spins on the
        channel's lock word and then reloads the producer head from the shared counter —
        the per-AIV mirror would otherwise be stale. Wrap a whole post/submit/drain
        sequence, and the staged appends that publish with it, in one lock/unlock pair.
        """

    @_api_decl
    def unlock(self) -> None:
        """Release the channel's cross-AIV lock (device-only).

        Publishes the cached producer state before handing the channel over. No-op on a
        shared window.
        """

    @_api_decl
    def is_shared(self) -> bool:
        """True if the channel's window is shared by the peers, so it rides this AIV's jetty."""


class BufferChannel(NamedTuple):
    """Parse-time handle of one buffer channel (device-only).

    Built only by :func:`buffer_channel` inside a kernel; composes on an opened
    :class:`Channel` with a caller-reserved UB staging ring. WQE semantics per method
    are documented on :class:`Channel`; the buffer-channel differences are noted per
    method and in :func:`buffer_channel`.
    """

    ch: Channel  # the opened base channel (same jetty, peer and headers)
    buf_addr: int  # UB staging-ring address, 64-byte aligned
    buf_bytes: int  # UB staging-ring size in bytes, a 64-byte multiple of at least 128
    token: object  # IR Var (int64) produced by dist.buffer_channel

    # -- declarations only: the parser routes every call via dist_ops.parse_buffer_channel_method

    @_api_decl
    def write(self, dst, src, /, *, option: UrmaOption = UrmaOption.DEFAULT) -> None:
        """Append one WQE to the staging ring (device-only).

        Pure UB stores — nothing touches the SQ. The payload must be <= 256 MiB
        (larger transfers must be split by the caller — the limit is asserted in
        sanitizer builds). ``option`` is the staged WQEs' BODY flag byte (default
        ``pld.UrmaOption.DEFAULT``); a caller-set ``CQE`` bit is ignored —
        :meth:`submit` adds fence+cqe to the last WQEBB when it publishes, so one
        completion covers the batch. No ``submit``/``timeout``: everything stages,
        :meth:`submit` publishes. A full ring traps.
        """

    @_api_decl
    def write_notify(self, dst, src, signal, val, /, *, option: UrmaOption = UrmaOption.DEFAULT) -> None:
        """Append a write-with-notify to the staging ring (device-only, two WQEBBs).

        The payload must be <= 256 MiB (larger transfers must be split by the
        caller — the limit is asserted in sanitizer builds). The notify fires when
        the batch is published, in order behind the staged appends. ``signal`` must
        be a 64-bit integer pointer; ``option`` semantics match :meth:`write`.
        """

    @_api_decl
    def submit(self, timeout: int | None = None) -> None:
        """Publish the staged batch: one bulk DMA, one head advance, one doorbell.

        One CQE per published batch (the last staged WQEBB carries the final flags).
        The doorbell always rings, so an engine op posted earlier with ``submit=False``
        is published too. ``timeout`` bounds the SQ/CQ capacity wait; expiry traps
        the AIV.
        """

    @_api_decl
    def drain(self, timeout: int | None = None) -> None:
        """Wait for engine completion of every published batch (device-only).

        Traps if WQEs were staged and never submitted. Same ``timeout`` semantics as
        :meth:`Channel.drain`.
        """

    @_api_decl
    def read(self, dst, src, /, *, option: UrmaOption = UrmaOption.DEFAULT) -> None:
        """Append an engine read to the staging ring (device-only).

        The transfer must be <= 256 MiB (larger transfers must be split by the
        caller — the limit is asserted in sanitizer builds). The fetched bytes are
        engine-written: readable only after :meth:`drain`. ``option`` semantics
        match :meth:`write`.
        """

    @_api_decl
    def write_reduce(self, dst, src, /, *, op: ReduceOp,
                     option: UrmaOption = UrmaOption.DEFAULT) -> None:
        """Append an engine-side reduction to the staging ring (device-only).

        Element-wise reduces ``src`` into the remote ``dst`` per ``op``; at most
        256 MiB per call (larger transfers must be split by the caller — the limit
        is asserted in sanitizer builds). Concurrent reduces from several ranks into
        one target are hardware-serialized and not ordered against other engine ops
        — :meth:`drain` first. ``option`` semantics match :meth:`write`.
        """

    @_api_decl
    def atomic_fetch_add(self, dst, fetch, val, /, *, option: UrmaOption = UrmaOption.DEFAULT) -> None:
        """Append a remote fetch-and-add to the staging ring (device-only, two WQEBBs).

        ``fetch`` (a local one-element GM tensor) holds the previous remote value
        after :meth:`drain`. ``option`` semantics match :meth:`write`.
        """

    @_api_decl
    def atomic_cas(self, dst, fetch, expected, desired, /, *, option: UrmaOption = UrmaOption.DEFAULT) -> None:
        """Append a remote compare-and-swap to the staging ring (device-only, two WQEBBs).

        ``fetch`` holds the previous remote value after :meth:`drain`; success iff
        ``fetch == expected``. ``option`` semantics match :meth:`write`."""

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
def remote_tensor(win: Window, tensor, peer: int):
    """Tensor view over a peer's copy of a region registered on the UB_MEM plane.

    Sugar for ``pl.make_tensor(pld.remote_ptr(win, tensor, peer), tensor.shape)``: the
    view keeps the source tensor's shape, so element offsets match the local tensor and
    ``pl.atomic_add(view, idx, v)`` / ``pl.atomic_load`` address the peer's element
    directly. Device-only; the region must be registered on the UB_MEM plane.

    Args:
        win: the owning window.
        tensor: a kernel tensor parameter registered on ``win``.
        peer: destination rank (compile-time or in-kernel integer).
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

        ptr = pl.addptr(pl.make_ptr(buf), 4096)    # 4096 elements into buf
        off = pld.remote_ptr(win, ptr, peer)       # same element offset on the peer

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
def channel(win: Window, buffer: int, /, *, peer: int | None = None) -> Channel:
    """Open one write-engine channel of ``win``, backed by a UB scratch buffer (device-only).

    Must be the RHS of a simple assignment, outside control flow — the C++ object lives
    in the enclosing scope. The window must have at least one URMA-registered region:
    the jetty is built from them, and constructing a channel on a window without any
    URMA region traps. The channel binds its destination when it is opened: with
    ``peer=k`` it rides peer ``k``'s jetty (per-endpoint); without ``peer`` it rides this
    AIV's own jetty (shared window). Passing ``peer`` on a shared window traps at
    runtime, and so does omitting it on a per-endpoint window. :meth:`write` and
    :meth:`write_notify` take no peer. A write is not ordered against a separate signal
    write — use ``drain`` first, or ``write_notify``.

    Args:
        win: the owning window.
        buffer: UB scratch address, >= ``pld.CHANNEL_UB_BYTES`` (256) bytes, 64-byte
            aligned, not overlapping any Tile or another channel's scratch. The first
            128 bytes hold the peer/jetty headers; :meth:`write_indexed` keeps its SQE
            header above them. Compile-time.
        peer: destination rank for the per-peer form (compile-time or in-kernel integer).
            ``peer == win.rank()`` is rejected at runtime. Must be omitted when
            ``win.is_shared()`` is true (per-AIV jetty); passing it there traps.
    """


@_api_decl
def buffer_channel(ch: Channel, buf_addr: int, buf_bytes: int, /) -> BufferChannel:
    """Bind a caller-reserved UB staging buffer to an opened channel (device-only).

    Must be the RHS of a simple assignment, outside control flow — the C++ object
    lives in the enclosing scope. The result composes on ``ch`` (same jetty, peer and
    headers — the C++ side holds a ``Channel*``, not a copy); the buffer is pure WQEBB
    staging (the vendor ``MakeBatchHandle`` workspace shape). Batch semantics:

    - ``bch.write`` / scalar writes **append** to the buffer — pure UB stores;
      ``option=`` selects the staged WQEs' body flag byte (a caller-set ``CQE`` bit
      is ignored); no ``submit``/``timeout`` arguments.
    - ``bch.submit()`` publishes the staged batch with one SQ reservation, one bulk
      DMA, one head advance and one doorbell, adding fence+cqe to the last WQEBB at
      submit — **one CQE per batch**. A full buffer is a trap, not a flush.
    - ``bch.read`` / ``write_reduce`` / ``write_notify`` / the atomics also **stage**
      into the ring (reads and atomics are engine-written and their results are
      readable only after :meth:`drain`; the published final WQEBB's fence retires the
      batch in order). ``write_indexed`` is not available on a buffer channel — use
      the base channel for that.
    - ``bch.drain()`` traps if WQEBBs were staged and never submitted.

    Do not call ``ch`` posting methods while a batch is staged — submit checks the
    channel head and traps on interleaved posts. Between batches (after ``submit``)
    plain ``ch`` posts are harmless.

    Args:
        ch: the opened channel (``pld.channel`` handle) to build on.
        buf_addr: compile-time UB address, 64-byte aligned. The staging buffer must
            not overlap any Tile or another channel's scratch.
        buf_bytes: compile-time buffer size, at least 128 bytes and a multiple of 64.
            A notify or atomic WQE occupies two 64-byte slots.
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
