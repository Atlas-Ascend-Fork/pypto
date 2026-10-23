#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Python API declarations for the PyPTO Pro system namespace (``pl.system.*``).

These declarations exist so that:
- IDE "Go to Definition" works for every ``pl.system.xxx`` call
- Python catches typos at import time
- Type checkers can validate argument types
- Docstrings document the user-facing calling convention

None of these functions are meant to be called at runtime.  Inside a PyPTO
kernel the AST parser intercepts every ``pl.system.xxx`` call before Python
executes it.  Outside a kernel, calling a declaration raises ``InvalidOperation``.

The ``span`` parameter every implementation in ``ir/op/system_ops.py`` carries is
absent here: the parser captures the source location itself, so it is not part of
what a user writes.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

from pypto.ir import CacheLine, CrossCoreSyncMode, DcciDst, PipeType, SyncCoreType

from ._api import Tensor, Tile, _api_decl


class System:
    """System namespace (``pl.system.*``).

    Provides pipeline barriers, intra- and cross-core synchronization, mutex
    tokens, cache maintenance, and matmul layout control.
    """

    # -----------------------------------------------------------------
    # Pipeline barriers
    # -----------------------------------------------------------------

    @staticmethod
    @_api_decl
    def bar_all() -> None:
        """Global barrier synchronization."""

    @staticmethod
    @_api_decl
    def bar_m() -> None:
        """Matrix unit barrier."""

    @staticmethod
    @_api_decl
    def bar_fix() -> None:
        """FIX pipeline barrier."""

    @staticmethod
    @_api_decl
    def bar_mte1() -> None:
        """MTE1 pipeline barrier."""

    @staticmethod
    @_api_decl
    def bar_mte2() -> None:
        """MTE2 pipeline barrier."""

    @staticmethod
    @_api_decl
    def bar_mte3() -> None:
        """MTE3 pipeline barrier."""

    # -----------------------------------------------------------------
    # Intra-core synchronization (flag set / wait)
    # -----------------------------------------------------------------

    @staticmethod
    @_api_decl
    def sync_src(*, set_pipe: PipeType, wait_pipe: PipeType, event_id: int) -> None:
        """Send a synchronization signal (Set Flag).

        Args:
            set_pipe: Pipe that sets the flag
            wait_pipe: Pipe that will wait on the flag
            event_id: Event identifier
        """

    @staticmethod
    @_api_decl
    def sync_dst(*, set_pipe: PipeType, wait_pipe: PipeType, event_id: int) -> None:
        """Wait for a synchronization signal (Wait Flag).

        Args:
            set_pipe: Pipe that sets the flag
            wait_pipe: Pipe that waits on the flag
            event_id: Event identifier
        """

    # -----------------------------------------------------------------
    # Cross-core synchronization
    # -----------------------------------------------------------------

    @staticmethod
    @_api_decl
    def set_cross_core(
        *,
        pipe: PipeType,
        event_id: int,
        sync_mode: CrossCoreSyncMode = CrossCoreSyncMode.INTRA_BLOCK,
    ) -> None:
        """Set a synchronization signal (cross core).

        Args:
            pipe: Pipe that sets the flag. ``PipeType.S`` supports only
                ``CrossCoreSyncMode.INTRA_BLOCK`` and ``CrossCoreSyncMode.UNICAST_BLOCK``.
            event_id: Event identifier (int for static, Scalar expression for dynamic)
            sync_mode: Cross-core sync mode. ``pl.CrossCoreSyncMode.INTRA_BLOCK`` (default,
                mode 2) for AIC-AIV both subcores, ``UNICAST_BLOCK`` (mode 3) for AIC-AIV one
                subcore, ``INTER_BLOCK`` (mode 0) for inter-core, ``INTER_SUBBLOCK`` (mode 1)
                for AIV-to-AIV.
        """

    @staticmethod
    @_api_decl
    def wait_cross_core(
        *,
        pipe: PipeType,
        event_id: int,
        sync_mode: CrossCoreSyncMode = CrossCoreSyncMode.INTRA_BLOCK,
    ) -> None:
        """Wait for a synchronization signal (cross core).

        Args:
            pipe: Pipe that waits on the flag. ``PipeType.S`` is supported for all sync modes.
            event_id: Event identifier (int for static, Scalar expression for dynamic)
            sync_mode: Cross-core sync mode. Prefer the same mode as the paired
                ``set_cross_core``; ensure the participating cores and event IDs match.
        """

    @staticmethod
    @_api_decl
    def sync_all(*, core_type: SyncCoreType = SyncCoreType.MIX) -> None:
        """Global core synchronization (delegates to pto-isa SYNCALL).

        Uses the FFTS hardware signal and needs no workspace.

        Args:
            core_type: Which cores participate. ``pl.SyncCoreType.AIV_ONLY`` syncs
                vector cores only, ``MIX`` (default) syncs both AIC and AIV cores.
        """

    # -----------------------------------------------------------------
    # Mutex buffer-id tokens
    # -----------------------------------------------------------------

    @staticmethod
    @_api_decl
    def mutex_lock(*, pipe: PipeType, mutex_id: int) -> None:
        """Acquire a Mutex buffer-id token on ``pipe``.

        Blocks the ``pipe`` instruction queue until the previous holder of
        ``mutex_id`` releases it via ``mutex_unlock``.

        Args:
            pipe: PipeType for which to acquire the lock (e.g. ``pl.PipeType.MTE2``)
            mutex_id: MutexID 0-31, as an int or an integer scalar expression
        """

    @staticmethod
    @_api_decl
    def mutex_unlock(*, pipe: PipeType, mutex_id: int) -> None:
        """Release a Mutex buffer-id token previously acquired on ``pipe``.

        Must be paired with ``mutex_lock`` on the same ``pipe`` and ``mutex_id``.

        Args:
            pipe: PipeType for which to release the lock
            mutex_id: MutexID passed to the paired ``mutex_lock``
        """

    # -----------------------------------------------------------------
    # Cache maintenance and matmul layout
    # -----------------------------------------------------------------

    @staticmethod
    @_api_decl
    def dcci(
        target: Union[Tensor, Tile],
        offset: Optional[Union[Sequence[int], int]] = None,
        *,
        cache_line: CacheLine = CacheLine.ENTIRE_DATA_CACHE,
        dst: DcciDst = DcciDst.AUTO,
    ) -> None:
        """Data Cache Clean and Invalid for a GM Tensor or a UB Tile.

        Args:
            target: GM Tensor or UB Tile
            offset: A Tensor target takes per-dimension offsets or a scalar element
                offset; a Tile target takes a scalar element offset. Omitted, the
                target base address is used.
            cache_line: ``pl.CacheLine.SINGLE_CACHE_LINE`` or ``ENTIRE_DATA_CACHE``
            dst: DCCI destination. ``pl.DcciDst.AUTO`` maps a Tensor to CACHELINE_OUT
                and a Tile to CACHELINE_UB.
        """

    @staticmethod
    @_api_decl
    def set_mm_layout_transform(*, enabled: bool) -> None:
        """Set the matmul layout transform mode for the fixpipe drain direction.

        When enabled, fixpipe drains L0C in the N direction (column-first) instead of
        the M direction (row-first), so cube and fixpipe access L0C along orthogonal
        axes within one slot.

        Args:
            enabled: True to enable N-direction drain, False to restore M-direction
        """
