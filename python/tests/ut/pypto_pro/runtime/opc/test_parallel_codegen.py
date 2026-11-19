# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise real spawned codegen and GNU Make token ownership without an NPU."""

import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
import threading
import time

from pypto_pro import DataType
from pypto_pro._errors import RuntimeFailure
from pypto_pro.runtime.opc import _parallel_codegen as parallel
from pypto_pro.runtime.opc._parallel_codegen import codegen_budget
from pypto_pro.runtime.opc.pypto_compile import _codegen_variants, _load_kernel
from pypto_pro.runtime.platform import NpuArch
import pytest


@pytest.fixture(autouse=True)
def standalone_environment(monkeypatch):
    monkeypatch.delenv("MAKEFLAGS", raising=False)
    monkeypatch.delenv("MAKELEVEL", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(8)))


@pytest.fixture
def kernel_file(tmp_path):
    source = tmp_path / "probe.py"
    source.write_text(textwrap.dedent('''\
        from dataclasses import dataclass
        import pypto_pro.language as pl
        from pypto_pro.runtime.tilingkey import TilingKeyField

        class Keys:
            operation = TilingKeyField(bits=1, values=[0, 1])

        @dataclass
        class Tiling:
            count: int

        @pl.jit(auto_mutex=False, tiling_key=Keys, datatype={"x": "input_dtype"})
        def probe(x: pl.Ptr[pl.DT_UINT8], tiling: Tiling):
            tile_type = pl.TileType(shape=[16, 16], dtype=input_dtype, target_memory=pl.MemorySpace.Vec)
            a = pl.make_tile(tile_type, addr=0)
            b = pl.make_tile(tile_type, addr=512)
            c = pl.make_tile(tile_type, addr=1024)
            with pl.section_cube():
                pl.system.bar_all()
            with pl.section_vector():
                if operation == 0:
                    pl.add(c, a, b)
                else:
                    pl.sub(c, a, b)
    '''), encoding="utf-8")
    return source


@pytest.mark.parametrize("dtype", [DataType.FP16, DataType.BF16])
def test_spawn_matches_serial_sources_headers_and_abi(kernel_file, tmp_path, monkeypatch, dtype, capsys):
    kernel = _load_kernel(str(kernel_file))
    kernel_name = f"probe_{dtype}"
    outputs = []
    for workers in (1, 2):
        monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(workers)))
        directory = tmp_path / str(workers)
        directory.mkdir()
        results = _codegen_variants(
            kernel, str(kernel_file), NpuArch.DAV_3510, {"input_dtype": dtype}, ["1", "0"], str(directory), kernel_name
        )
        stdout = capsys.readouterr().out
        assert f"kernel_name={kernel_name}, 2 tilingkeys, up to {workers} workers" in stdout
        assert f"kernel_name={kernel_name}, 2 tilingkeys completed in " in stdout
        outputs.append([
            (
                cg.kernel_name, cg.entry_params, cg.has_cube, cg.has_vector,
                {
                    file.name: file.read_bytes() for file in Path(cg.build_dir).iterdir()
                    if file.suffix in (".cpp", ".h")
                },
            ) for cg in results
        ])
        assert [Path(cg.build_dir).name for cg in results] == ["_cg_tk_1", "_cg_tk_0"]
        assert all(cg.has_cube and cg.has_vector for cg in results)
        assert not list(directory.glob("_codegen_*"))
    assert outputs[0] == outputs[1]
    # Both branches were specialized, rather than accidentally sharing a cached kernel.
    assert outputs[1][0][-1]["kernel.cpp"] != outputs[1][1][-1]["kernel.cpp"]


def test_worker_error_keeps_key_dtype_and_source(kernel_file, tmp_path):
    kernel_file.write_text(kernel_file.read_text().replace("pl.sub(c, a, b)", "missing_operation(c, a, b)"))
    kernel = _load_kernel(str(kernel_file))
    output = tmp_path / "failed"
    output.mkdir()
    with pytest.raises(RuntimeFailure) as error:
        _codegen_variants(
            kernel, str(kernel_file), NpuArch.DAV_3510, {"input_dtype": DataType.BF16}, ["1", "0"], str(output), "probe"
        )
    message = str(error.value)
    for expected in ("tilingkey 1", "input_dtype", str(DataType.BF16), "probe.py", "missing_operation"):
        assert expected in message
    assert (output / "pypto_codegen.log").is_file()
    assert not (output / "_cg_tk_1/kernel.cpp").exists()
    assert not list(output.glob("_codegen_*"))


def test_one_make_slot_reuses_worker_with_real_torch(kernel_file, tmp_path, monkeypatch):
    pytest.importorskip("torch.nn.functional")
    kernel_file.write_text(textwrap.dedent('''\
        import os
        from pathlib import Path
        import torch.nn.functional
        Path(__file__).with_name(f"{os.getpid()}.loaded").touch()
    ''') + kernel_file.read_text(), encoding="utf-8")
    reader, writer = os.pipe()
    try:
        monkeypatch.setenv("MAKEFLAGS", f"-j2 --jobserver-auth={reader},{writer}")
        with codegen_budget(2) as budget:
            results = parallel.parallel_codegen(
                kernel_file, "3510", {"input_dtype": DataType.FP16},
                [(0, {"operation": 0}), (1, {"operation": 1})], tmp_path, budget,
            )
        assert len(list(tmp_path.glob("*.loaded"))) == 1
        assert [Path(result.build_dir).name for result in results] == ["_cg_tk_0", "_cg_tk_1"]
    finally:
        os.close(reader)
        os.close(writer)


def test_failed_key_reaps_blocked_sibling(kernel_file, tmp_path, monkeypatch):
    kernel_file.write_text(textwrap.dedent('''\
        import importlib
        from pathlib import Path
        import time

        def fail_codegen(prog, arch, **kwargs):
            root = Path(__file__).parent
            if kwargs["tilingkey_packed"] == 0:
                deadline = time.monotonic() + 15
                while not (root / "sibling_started").exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                raise RuntimeError("failed first key")
            (root / "sibling_started").touch()
            time.sleep(10)
            (root / "sibling_finished").touch()

        importlib.import_module("pypto_pro.runtime.jit")._codegen = fail_codegen
    ''') + kernel_file.read_text(), encoding="utf-8")
    reader, writer = os.pipe()
    try:
        os.set_blocking(reader, False)
        os.write(writer, b"x")
        monkeypatch.setenv("MAKEFLAGS", f"-j4 --jobserver-auth={reader},{writer}")
        with pytest.raises(RuntimeFailure, match="failed first key"):
            with codegen_budget(4) as budget:
                parallel.parallel_codegen(
                    kernel_file, "3510", {"input_dtype": DataType.FP16},
                    [(key, {"operation": key % 2}) for key in range(4)], tmp_path, budget,
                )
        assert (tmp_path / "sibling_started").exists()
        assert not (tmp_path / "sibling_finished").exists()
        assert os.read(reader, 2) == b"x"
    finally:
        os.close(reader)
        os.close(writer)


@pytest.mark.parametrize("cpus,tasks,expected", [(64, 64, 32), (3, 20, 3), (8, 2, 2), (1, 20, 1)])
def test_worker_count_respects_fixed_cpu_and_task_limits(monkeypatch, cpus, tasks, expected):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(cpus)))
    with codegen_budget(tasks) as budget:
        assert budget.acquire(tasks) == expected


@pytest.mark.parametrize("flags", ["", "w", "-j4 --jobserver-auth=-1,-1"])
def test_make_without_accessible_jobserver_stays_serial(monkeypatch, flags):
    monkeypatch.setenv("MAKELEVEL", "1")
    monkeypatch.setenv("MAKEFLAGS", flags)
    with codegen_budget(10) as budget:
        assert budget.limit == 1
        assert budget.acquire(10) == 1


@pytest.mark.parametrize("fifo", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_jobserver_returns_exact_tokens_even_after_failure(monkeypatch, tmp_path, fifo, fail):
    if fifo:
        path = tmp_path / "jobserver"
        os.mkfifo(path)
        reader = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        writer = os.open(path, os.O_WRONLY)
        auth = f"fifo:{path}"
    else:
        reader, writer = os.pipe()
        auth = f"{reader},{writer}"
    try:
        os.write(writer, b"xy")
        original_flags = fcntl.fcntl(reader, fcntl.F_GETFL)
        monkeypatch.setenv("MAKEFLAGS", f"-j3 --jobserver-auth={auth}")
        monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(8)))
        try:
            with codegen_budget(20) as budget:
                assert budget.acquire(20) == 3
                assert fcntl.fcntl(reader, fcntl.F_GETFL) == original_flags
                # A second compiler can always use its implicit slot, without blocking.
                with codegen_budget(20) as second_budget:
                    assert second_budget.acquire(20) == 1
                if fail:
                    raise RuntimeError("worker failed")
        except RuntimeError as error:
            assert fail and str(error) == "worker failed"
        assert os.read(reader, 2) == b"xy"
    finally:
        os.close(reader)
        os.close(writer)


@pytest.mark.parametrize("fifo", [False, True])
def test_budget_expands_and_returns_unused_tokens(monkeypatch, tmp_path, fifo):
    if fifo:
        path = tmp_path / "jobserver"
        os.mkfifo(path)
        reader = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        writer = os.open(path, os.O_WRONLY)
        auth = f"fifo:{path}"
    else:
        reader, writer = os.pipe()
        os.set_blocking(reader, False)
        auth = f"{reader},{writer}"
    try:
        monkeypatch.setenv("MAKEFLAGS", f"-j4 --jobserver-auth={auth}")
        with codegen_budget(20) as budget:
            assert budget.acquire(20) == 1
            os.write(writer, b"xyz")
            assert budget.acquire(20) == 4
            budget.release_unused(2)
            assert os.read(reader, 2) == b"xy"
            assert budget.acquire(2) == 2
        assert os.read(reader, 1) == b"z"
    finally:
        os.close(reader)
        os.close(writer)


def test_running_codegen_uses_late_jobserver_tokens(kernel_file, tmp_path):
    # Keep every task running until all four start. New tokens must let the pool
    # expand before its first task finishes, rather than between fixed batches.
    kernel_file.write_text(textwrap.dedent('''\
        import importlib
        import os
        from pathlib import Path
        import time
        from types import SimpleNamespace

        def controlled_codegen(prog, arch, **kwargs):
            root = Path(__file__).parent
            key = kwargs["tilingkey_packed"]
            (root / f"started_{key}_{os.getpid()}").touch()
            deadline = time.monotonic() + 30
            gate = "release_last" if key == 0 else "release"
            while not (root / gate).exists():
                if time.monotonic() > deadline:
                    raise RuntimeError("late workers did not start")
                time.sleep(0.01)
            return SimpleNamespace(
                build_dir=kwargs["out_dir"], kernel_name="probe", entry_params=[],
                has_cube=True, has_vector=True,
            )

        importlib.import_module("pypto_pro.runtime.jit")._codegen = controlled_codegen
    ''') + kernel_file.read_text().replace("bits=1, values=[0, 1]", "bits=2, values=[0, 1, 2, 3]"), encoding="utf-8")
    driver = tmp_path / "expand.py"
    driver.write_text(textwrap.dedent(f'''\
        import json
        import os
        from pathlib import Path
        import sys
        from unittest.mock import MagicMock
        sys.modules["torch"] = MagicMock(name="torch")
        sys.modules["torch_npu"] = MagicMock(name="torch_npu")
        sys.path[:] = {sys.path!r}
        os.sched_getaffinity = lambda _: set(range(4))
        from pypto_pro.runtime.opc.pypto_compile import _codegen_variants, _load_kernel
        kernel = _load_kernel(sys.argv[1])
        results = _codegen_variants(kernel, sys.argv[1], "3510", None, ["3", "2", "1", "0"], sys.argv[2], "probe")
        Path(sys.argv[2], "order.json").write_text(json.dumps([result.build_dir for result in results]))
    '''), encoding="utf-8")
    output = tmp_path / "expanded"
    output.mkdir()
    reader, writer = os.pipe()
    os.set_blocking(reader, False)
    env = dict(os.environ, MAKEFLAGS=f"-j4 --jobserver-auth={reader},{writer}")
    process = None
    try:
        with (tmp_path / "expand.log").open("w") as log:
            process = subprocess.Popen(
                [sys.executable, str(driver), str(kernel_file), str(output)],
                env=env, stdout=log, stderr=subprocess.STDOUT, pass_fds=(reader, writer),
            )
            deadline = time.monotonic() + 30
            while not list(tmp_path.glob("started_*")):
                assert process.poll() is None, (tmp_path / "expand.log").read_text()
                assert time.monotonic() < deadline, "first codegen task did not start"
                time.sleep(0.05)
            assert len(list(tmp_path.glob("started_*"))) == 1
            os.write(writer, b"xyz")
            while len(list(tmp_path.glob("started_*"))) < 4:
                assert process.poll() is None, (tmp_path / "expand.log").read_text()
                assert time.monotonic() < deadline, "codegen did not acquire late tokens"
                time.sleep(0.05)
            pids = {int(path.name.rsplit("_", 1)[1]) for path in tmp_path.glob("started_*")}
            assert len(pids) == 4
            (tmp_path / "release").touch()
            # Three finished tasks must return all extra slots while the last
            # task still owns the OPC's implicit slot.
            returned = bytearray()
            while len(returned) < 3:
                assert process.poll() is None, (tmp_path / "expand.log").read_text()
                assert time.monotonic() < deadline, "finished tasks retained Make slots"
                try:
                    returned.extend(os.read(reader, 3 - len(returned)))
                except BlockingIOError:
                    time.sleep(0.05)
            assert sorted(returned) == sorted(b"xyz")
            last_pid = int(next(tmp_path.glob("started_0_*")).name.rsplit("_", 1)[1])
            assert all(not Path(f"/proc/{pid}").exists() for pid in pids - {last_pid})
            (tmp_path / "release_last").touch()
            assert process.wait(timeout=10) == 0, (tmp_path / "expand.log").read_text()
        assert [Path(path).name for path in json.loads((output / "order.json").read_text())] == [
            "_cg_tk_3", "_cg_tk_2", "_cg_tk_1", "_cg_tk_0",
        ]
        with pytest.raises(BlockingIOError):
            os.read(reader, 1)
    finally:
        (tmp_path / "release").touch()
        (tmp_path / "release_last").touch()
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        os.close(reader)
        os.close(writer)


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_cancellation_stops_workers_and_returns_tokens(kernel_file, tmp_path, signum):
    # Pause real spawned workers during import so cancellation cannot race normal
    # completion. Only the pool imports this source; the driver does not load it.
    kernel_file.write_text(textwrap.dedent('''\
        import os
        from pathlib import Path
        import time
        Path(__file__).with_name(f"{os.getpid()}.worker").write_text(str(os.getpgrp()))
        time.sleep(60)
    ''') + kernel_file.read_text(), encoding="utf-8")
    driver = tmp_path / "cancel.py"
    driver.write_text(textwrap.dedent(f'''\
        import sys
        from unittest.mock import MagicMock
        sys.modules["torch"] = MagicMock(name="torch")
        sys.modules["torch_npu"] = MagicMock(name="torch_npu")
        sys.path[:] = {sys.path!r}
        from pypto_pro import DataType
        from pypto_pro.runtime.opc._parallel_codegen import codegen_budget, parallel_codegen
        with codegen_budget(2) as budget:
            parallel_codegen(sys.argv[1], "3510", {{"input_dtype": DataType.FP16}},
                             [(0, {{"operation": 0}}), (1, {{"operation": 1}})], sys.argv[2], budget)
    '''), encoding="utf-8")
    output = tmp_path / "cancelled"
    output.mkdir()
    reader, writer = os.pipe()
    os.set_blocking(reader, False)
    os.write(writer, b"xy")
    env = dict(os.environ, MAKEFLAGS=f"-j3 --jobserver-auth={reader},{writer}")
    process = None
    log_path = tmp_path / "cancel.log"
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [sys.executable, str(driver), str(kernel_file), str(output)],
                env=env, stdout=log, stderr=subprocess.STDOUT,
                pass_fds=(reader, writer), start_new_session=True,
            )
            deadline = time.monotonic() + 30
            while len(list(tmp_path.glob("*.worker"))) < 2:
                assert process.poll() is None, log_path.read_text()
                assert time.monotonic() < deadline, "codegen workers did not start"
                time.sleep(0.05)
            process.send_signal(signum)
            assert process.wait(timeout=10) != 0

        def running_workers():
            states = [Path(f"/proc/{marker.stem}/stat") for marker in tmp_path.glob("*.worker")]
            # Killed grandchildren may briefly be zombies pending reaping by init.
            running = []
            for path in states:
                try:
                    if path.read_text().rsplit(")", 1)[1].split()[0] != "Z":
                        running.append(path)
                except FileNotFoundError:
                    pass
            return running

        deadline = time.monotonic() + 5
        while running_workers() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not running_workers(), "codegen workers survived cancellation"
        assert sorted(os.read(reader, 10)) == sorted(b"xy")
        assert not list(output.glob("_codegen_*"))
    finally:
        # This also cleans up the reproducer when running against a broken version.
        for group in {int(marker.read_text()) for marker in tmp_path.glob("*.worker")}:
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        os.close(reader)
        os.close(writer)


def test_budget_restores_default_and_preserves_custom_signal_handler():
    previous = signal.getsignal(signal.SIGTERM)
    try:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        with pytest.raises(SystemExit, match="143"):
            with codegen_budget(1):
                os.kill(os.getpid(), signal.SIGTERM)
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        with codegen_budget(1):
            assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
    finally:
        signal.signal(signal.SIGTERM, previous)


@pytest.mark.parametrize("native_thread", [False, True])
def test_cancellation_during_token_acquisition_returns_consumed_token(monkeypatch, native_thread):
    reader, writer = os.pipe()
    os.set_blocking(reader, False)
    os.write(writer, b"xy")
    monkeypatch.setenv("MAKEFLAGS", f"-j3 --jobserver-auth={reader},{writer}")
    original_read = os.read
    stop = threading.Event()
    thread = threading.Thread(target=stop.wait)
    thread.start()

    def interrupted_read(fd, count):
        token = original_read(fd, count)
        if native_thread:
            signal.pthread_kill(thread.ident, signal.SIGTERM)
            time.sleep(0.05)
        else:
            os.kill(os.getpid(), signal.SIGTERM)
        return token

    monkeypatch.setattr(os, "read", interrupted_read)
    try:
        with pytest.raises(SystemExit, match="143"):
            with codegen_budget(2) as budget:
                budget.acquire(2)
                pytest.fail("pending cancellation was not delivered")
        assert sorted(original_read(reader, 10)) == sorted(b"xy")
    finally:
        os.close(reader)
        os.close(writer)
        stop.set()
        thread.join()


def test_cancellation_during_process_creation_reaps_child(kernel_file, tmp_path, monkeypatch):
    original_popen = subprocess.Popen
    started = []

    def interrupted_popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        started.append(process)
        os.kill(os.getpid(), signal.SIGTERM)
        return process

    monkeypatch.setattr(subprocess, "Popen", interrupted_popen)
    with pytest.raises(SystemExit, match="143"):
        with codegen_budget(2) as budget:
            parallel.parallel_codegen(
                kernel_file, "3510", {"input_dtype": DataType.FP16},
                [(0, {"operation": 0}), (1, {"operation": 1})], tmp_path, budget,
            )
    assert len(started) == 1 and started[0].poll() is not None
    assert not list(tmp_path.glob("_codegen_*"))
