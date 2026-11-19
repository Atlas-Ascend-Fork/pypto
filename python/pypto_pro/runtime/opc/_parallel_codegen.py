# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bound OPC frontend parallelism and run codegen in isolated Python processes."""

from contextlib import contextmanager
from multiprocessing import Pipe
from multiprocessing.connection import wait
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

from pypto_pro._errors import RuntimeFailure

_CANCELLATION = threading.local()
_MAX_CODEGEN_JOBS = 32


class _CodegenCancellation:
    def __init__(self):
        self.depth = 0
        self.pending = None

    def __call__(self, signum, frame):
        if self.depth:
            if self.pending is None:
                self.pending = (signum, frame)
            return
        if signum == signal.SIGTERM:
            # A repeated termination must not interrupt resource cleanup.
            signal.signal(signum, signal.SIG_IGN)
            raise SystemExit(128 + signum)
        signal.default_int_handler(signum, frame)


@contextmanager
def _defer_cancellation():
    """Finish recording resource ownership before delivering a pending cancellation."""
    state = getattr(_CANCELLATION, "state", None)
    if state is None:
        yield
        return
    # Blocking signals in this thread alone is insufficient: CANN can have native
    # threads that receive a process signal and schedule its Python handler here.
    state.depth += 1
    try:
        yield
    finally:
        state.depth -= 1
        if not state.depth and state.pending is not None:
            pending, state.pending = state.pending, None
            state(*pending)


def _requested_jobs(task_count):
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    return max(1, min(_MAX_CODEGEN_JOBS, cpus, task_count))


class _CodegenBudget:
    """Keep Make tokens in the OPC process while the codegen pool changes size."""

    def __init__(self, limit, reader=None, writer=None):
        self.limit = limit
        self.reader = reader
        self.writer = writer
        self.tokens = bytearray()

    def acquire(self, remaining):
        target = min(self.limit, remaining)
        if self.reader is None:
            return target
        with _defer_cancellation():
            while len(self.tokens) < target - 1:
                try:
                    token = os.read(self.reader, 1)
                except BlockingIOError:
                    break
                if not token:
                    break
                self.tokens.extend(token)
        return min(target, 1 + len(self.tokens))

    def release_unused(self, remaining):
        with _defer_cancellation():
            while len(self.tokens) > max(0, remaining - 1):
                os.write(self.writer, bytes([self.tokens[0]]))
                del self.tokens[0]


def _open_jobserver(auth):
    """Open a private nonblocking reader without changing Make's descriptor flags."""
    if auth.startswith("fifo:"):
        if not stat.S_ISFIFO(os.stat(auth[5:]).st_mode):
            raise ValueError("invalid Make jobserver FIFO")
        reader = os.open(auth[5:], os.O_RDONLY | os.O_NONBLOCK)
        try:
            writer = os.open(auth[5:], os.O_WRONLY | os.O_NONBLOCK)
        except BaseException:
            os.close(reader)
            raise
        return reader, writer
    read_fd, write_fd = (int(fd) for fd in auth.split(","))
    read_stat, write_stat = os.fstat(read_fd), os.fstat(write_fd)
    if not stat.S_ISFIFO(read_stat.st_mode) or (read_stat.st_dev, read_stat.st_ino) != (
        write_stat.st_dev, write_stat.st_ino
    ):
        raise ValueError("invalid Make jobserver pipe")
    # dup() shares O_NONBLOCK with Make; reopening through /proc gives this
    # client its own flags. PyPTO OPC runs on Linux.
    reader = os.open(f"/proc/self/fd/{read_fd}", os.O_RDONLY | os.O_NONBLOCK)
    try:
        writer = os.dup(write_fd)
    except BaseException:
        os.close(reader)
        raise
    return reader, writer


@contextmanager
def codegen_budget(task_count):
    """Release workers and Make tokens on ordinary OPC termination as well as errors."""
    # SIGTERM normally bypasses Python finally blocks. Convert it to an exception
    # while we own frontend resources, without replacing an embedding application's
    # handler or attempting to install one from a non-main thread.
    if threading.current_thread() is not threading.main_thread() or getattr(_CANCELLATION, "state", None) is not None:
        with _jobserver_budget(task_count) as jobs:
            yield jobs
        return
    state = _CANCELLATION.state = _CodegenCancellation()
    previous = {}
    try:
        for signum, default in ((signal.SIGTERM, signal.SIG_DFL), (signal.SIGINT, signal.default_int_handler)):
            if signal.getsignal(signum) == default:
                previous[signum] = default
                signal.signal(signum, state)
        with _jobserver_budget(task_count) as jobs:
            yield jobs
    finally:
        try:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        finally:
            del _CANCELLATION.state


@contextmanager
def _jobserver_budget(task_count):
    """Open the shared budget; the caller borrows and returns tokens while codegen runs."""
    jobs = _requested_jobs(task_count)
    flags = os.environ.get("MAKEFLAGS", "")
    if jobs == 1 or (not flags and not os.environ.get("MAKELEVEL")):
        yield _CodegenBudget(jobs)
        return
    matches = re.findall(r"--jobserver-(?:auth|fds)=([^\s]+)", flags)
    if not matches:
        # A serial Make invocation owns only its implicit slot.
        yield _CodegenBudget(1)
        return
    try:
        reader, writer = _open_jobserver(matches[-1])
    except (OSError, ValueError):
        # E.g. a non-recursive recipe did not inherit the pipe descriptors.
        yield _CodegenBudget(1)
        return
    budget = _CodegenBudget(jobs, reader, writer)
    try:
        yield budget
    finally:
        with _defer_cancellation():
            try:
                budget.release_unused(0)
            finally:
                os.close(reader)
                os.close(writer)


def _start_worker(workers, index, request, task, log):
    worker = Path(__file__).with_name("_codegen_worker.py")
    control, child = Pipe()
    try:
        with child, _defer_cancellation():
            process = subprocess.Popen(
                [sys.executable, str(worker), str(child.fileno())],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(child.fileno(),),
            )
            workers[control] = (process, index)
        control.send((request, task))
    except BaseException:
        control.close()
        raise


def _run_workers(tasks, request, budget, log):
    # Each live process owns one slot, including during initialization and shutdown.
    workers = {}
    results = [None] * len(tasks)
    submitted = 0
    try:
        while submitted < len(tasks) or workers:
            if submitted < len(tasks):
                allowed = budget.acquire(len(workers) + len(tasks) - submitted)
                if submitted == 0:
                    print(f"PyPTO codegen: {len(tasks)} tilingkeys, {allowed} initial workers, "
                          f"up to {budget.limit}", file=log, flush=True)
                while len(workers) < allowed:
                    _start_worker(workers, submitted, request, tasks[submitted], log)
                    submitted += 1
            # A worker with no assigned index has received its stop message.
            active = [control for control, (_, index) in workers.items() if index is not None]
            for control in wait(active, timeout=0.05):
                process, index = workers[control]
                try:
                    results[index] = SimpleNamespace(**control.recv())
                except EOFError as exc:
                    raise ChildProcessError(f"worker exited while generating tilingkey {tasks[index][0]}") from exc
                if submitted < len(tasks):
                    control.send(tasks[submitted])
                    workers[control] = (process, submitted)
                    submitted += 1
                else:
                    control.send(None)
                    workers[control] = (process, None)
            for control, (process, index) in list(workers.items()):
                if process.poll() is not None:
                    if index is not None or process.returncode:
                        raise ChildProcessError(f"codegen worker exited with {process.returncode}")
                    control.close()
                    del workers[control]
                    budget.release_unused(len(workers))
    finally:
        with _defer_cancellation():
            for control, (process, _) in workers.items():
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                control.close()
            for process, _ in workers.values():
                process.wait()
    return results


def parallel_codegen(source, arch, dtype_consts, tasks, output_dir, budget):
    """Return results in input order; keep OPC globals and packaging in the caller."""
    request = {
        "source": str(Path(source).resolve()),
        "arch": str(arch),
        "dtype_consts": (
            {name: str(dtype) for name, dtype in dtype_consts.items()} if dtype_consts is not None else None
        ),
        "output_dir": str(Path(output_dir).resolve()),
        "python_path": [os.path.abspath(path) for path in sys.path],
        # Offline OPC wrappers stub these imports; embedded callers can use the real packages.
        "mock_modules": [name for name in ("torch", "torch_npu") if isinstance(sys.modules.get(name), Mock)],
    }
    log_path = Path(output_dir, "pypto_codegen.log")
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        try:
            results = _run_workers(tasks, request, budget, log)
        except (EOFError, BrokenPipeError, ConnectionResetError, ChildProcessError) as exc:
            detail = log_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeFailure(f"PyPTO parallel codegen failed ({exc}); see {log_path}\n{detail}") from exc
        print(f"PyPTO codegen completed in {time.perf_counter() - started:.3f}s", file=log)
    for result in results:
        result.entry_params = [tuple(param) for param in result.entry_params]
    return results
