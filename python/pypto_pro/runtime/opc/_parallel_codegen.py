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
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace

from pypto_pro._errors import RuntimeFailure

_CANCELLATION = threading.local()
_MAX_CODEGEN_JOBS = 8


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
    """Use one implicit Make slot plus currently available tokens; never wait for tokens."""
    jobs = _requested_jobs(task_count)
    flags = os.environ.get("MAKEFLAGS", "")
    if jobs == 1 or (not flags and not os.environ.get("MAKELEVEL")):
        yield jobs
        return
    matches = re.findall(r"--jobserver-(?:auth|fds)=([^\s]+)", flags)
    if not matches:
        # A serial Make invocation owns only its implicit slot.
        yield 1
        return
    try:
        reader, writer = _open_jobserver(matches[-1])
    except (OSError, ValueError):
        # E.g. a non-recursive recipe did not inherit the pipe descriptors.
        yield 1
        return
    tokens = bytearray()
    try:
        with _defer_cancellation():
            for _ in range(jobs - 1):
                try:
                    token = os.read(reader, 1)
                except BlockingIOError:
                    break
                if not token:
                    break
                tokens.extend(token)
        yield 1 + len(tokens)
    finally:
        with _defer_cancellation():
            try:
                for token in tokens:
                    os.write(writer, bytes([token]))
            finally:
                os.close(reader)
                os.close(writer)


def parallel_codegen(source, arch, dtype_consts, tasks, output_dir, workers):
    """Return results in input order; keep OPC globals and packaging in the caller."""
    output_dir = str(Path(output_dir).resolve())
    request = {
        "source": str(Path(source).resolve()),
        "arch": str(arch),
        "dtype_consts": (
            {name: str(dtype) for name, dtype in dtype_consts.items()} if dtype_consts is not None else None
        ),
        "tasks": tasks,
        "output_dir": output_dir,
        "workers": workers,
        "python_path": [os.path.abspath(path) for path in sys.path],
    }
    worker = Path(__file__).with_name("_codegen_worker.py")
    log_path = Path(output_dir, "pypto_codegen.log")
    with tempfile.TemporaryDirectory(prefix="_codegen_", dir=output_dir) as temporary:
        request_path, result_path = Path(temporary, "request.json"), Path(temporary, "result.json")
        request_path.write_text(json.dumps(request), encoding="utf-8")
        with log_path.open("w", encoding="utf-8") as log:
            process = None
            try:
                with _defer_cancellation():
                    process = subprocess.Popen(
                        [sys.executable, str(worker), str(request_path), str(result_path)],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                returncode = process.wait()
            except BaseException:
                # Include spawned workers when the calling build is interrupted.
                with _defer_cancellation():
                    if process is not None:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                raise
        if returncode:
            detail = log_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeFailure(f"PyPTO parallel codegen failed (exit {returncode}); see {log_path}\n{detail}")
        results = json.loads(result_path.read_text(encoding="utf-8"))
    for result in results:
        result["entry_params"] = [tuple(param) for param in result["entry_params"]]
    return [SimpleNamespace(**result) for result in results]
