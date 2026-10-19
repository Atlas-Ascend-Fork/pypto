#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Test LiteNPU kernel entry-name uniqueness for Kirin9050.

The LiteNPU entry symbol (and the ``kernelName`` field of the JSON sidecar that carries it) is built from the
function magic name plus the function content hash. The magic name alone is ``<function name>_<id>``, where the
function is the generated top-level device function -- so its name reflects the lowering structure rather than
anything the kernel author wrote -- and the id comes from a process-global counter that ``Program::Reset()``
rewinds. Kernels of the same structural shape therefore share a magic name whenever they are built in separate
processes, which is the regime that matters: each operator of a multi-operator offline model is exported and
built by its own process, and the resulting artifacts are only combined afterwards. Colliding entry names make
one kernel's binary shadow the other's inside the combined model.

Two consequences drive the shape of this test:

* The builds must happen in separate interpreters. Inside one interpreter the id counter keeps advancing, so
  two same-named kernels already get distinct magic names and the content hash is not what separates them --
  an in-process test would pass whether or not the hash takes part in the entry name.
* A control case is required. The content hash must depend on the kernel body only. If it picked up anything
  process-varying, the uniqueness assertion would go green for the wrong reason while colliding entry names
  stayed possible. Building the identical kernel in two fresh interpreters and demanding an identical entry
  name is what rules that out.

The sidecar is emitted by ``CodeGenLiteNPU::GenConfigJson`` only in a CANN-enabled build with ASCEND_HOME_PATH
set, so this is a box test: it skips when no sidecar is produced.
"""

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

# The build driver always names its kernel add_kernel, so the source name cannot be what separates two
# builds, and takes the operation to use as its one argument.
_DRIVER_PATH = Path(__file__).with_name("kernel_entry_uniqueness_driver.py")

# Entry names have the form <function name>_<function id>_<content hash>_main. The magic-name part is captured
# separately from the hash part so a test can assert on each independently.
_ENTRY_NAME = re.compile(r"^(?P<magic>[A-Za-z_][A-Za-z0-9_]*?_\d+)_(?P<hash>\d+)_main$")

_BUILD_TIMEOUT_SECONDS = 600

_ENTRY_NAME_PREFIX = "ENTRY_NAME="
_CCE_SYMBOL_PREFIX = "CCE_SYMBOL="
# Key under which the fixture carries the per-build emitted symbols, kept out of the tag namespace.
_SYMBOLS_KEY = "__symbols__"


def _output_tail(stdout, stderr):
    """Render the tail of a build's streams for a failure or skip message."""

    def as_text(stream):
        if stream is None:
            return ""
        return stream.decode(errors="replace") if isinstance(stream, bytes) else stream

    return f"--- stdout ---\n{as_text(stdout)[-2000:]}\n--- stderr ---\n{as_text(stderr)[-2000:]}"


def _entry_name(base_dir, tag, op):
    """Build add_kernel with the given operation in a fresh interpreter and return its emitted entry name."""
    # Each build gets its own copy, so the three builds differ in script path as well as body: the control
    # case then also shows the content hash ignores where the source was read from.
    script = base_dir / f"build_{tag}.py"
    shutil.copyfile(_DRIVER_PATH, script)

    # Each build gets its own framework output directory so the two runs cannot read each other's sidecars.
    output_dir = base_dir / f"output_{tag}"
    output_dir.mkdir()
    env = dict(os.environ)
    env["TILE_FWK_OUTPUT_DIR"] = str(output_dir)

    try:
        proc = subprocess.run(
            [sys.executable, str(script), op],
            cwd=str(base_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=_BUILD_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as expired:
        raise AssertionError(
            f"{tag} build exceeded {_BUILD_TIMEOUT_SECONDS}s\n{_output_tail(expired.stdout, expired.stderr)}"
        ) from expired

    # Exit code 3 is the one environment-limited outcome: code generation ran but the CANN-gated sidecar writer
    # produced nothing. Every other non-zero code, including a missing kernel_aicore directory, is a breakage.
    if proc.returncode == 3:
        pytest.skip(f"no LiteNPU JSON sidecar emitted for the {tag} build\n{_output_tail(proc.stdout, proc.stderr)}")
    if proc.returncode != 0:
        raise AssertionError(
            f"{tag} build failed with exit code {proc.returncode}\n{_output_tail(proc.stdout, proc.stderr)}"
        )

    # Scan for the tagged line rather than taking the last one: pypto's native logging also writes to stdout.
    reported = [
        line[len(_ENTRY_NAME_PREFIX):].strip()
        for line in proc.stdout.splitlines()
        if line.startswith(_ENTRY_NAME_PREFIX)
    ]
    if len(reported) != 1 or not reported[0]:
        raise AssertionError(
            f"{tag} build did not report exactly one entry name\n{_output_tail(proc.stdout, proc.stderr)}"
        )

    symbols = [
        line[len(_CCE_SYMBOL_PREFIX):].strip()
        for line in proc.stdout.splitlines()
        if line.startswith(_CCE_SYMBOL_PREFIX)
    ]
    return reported[0], symbols


@pytest.fixture(scope="module")
def entry_names(tmp_path_factory):
    """Entry names of three separate builds: add_kernel twice, then the same-named kernel with a different body."""
    if not os.environ.get("ASCEND_HOME_PATH"):
        pytest.skip("ASCEND_HOME_PATH is unset, so no LiteNPU JSON sidecar is emitted")

    base_dir = tmp_path_factory.mktemp("kernel_entry_uniqueness")
    builds = {
        "add_first": _entry_name(base_dir, "add_first", "add"),
        "add_second": _entry_name(base_dir, "add_second", "add"),
        "other_body": _entry_name(base_dir, "other_body", "sub"),
    }
    return {tag: name for tag, (name, _symbols) in builds.items()} | {
        _SYMBOLS_KEY: {tag: symbols for tag, (_name, symbols) in builds.items()}
    }


def test_same_kernel_in_two_processes_has_one_entry_name(entry_names):
    """Control: an identical kernel body built twice yields one entry name, so the hash is body-determined."""
    assert entry_names["add_first"] == entry_names["add_second"], (
        f"identical kernel bodies produced different entry names: "
        f"{entry_names['add_first']} vs {entry_names['add_second']}"
    )


def test_different_kernels_sharing_a_name_get_distinct_entry_names(entry_names):
    """Guard: two kernels both named add_kernel but with different bodies get distinct entry names."""
    assert entry_names["add_first"] != entry_names["other_body"], (
        f"kernels with different bodies share the entry name {entry_names['add_first']}, "
        f"so combining their artifacts into one offline model loses one of them"
    )


def test_distinct_entry_names_differ_only_in_the_content_hash(entry_names):
    """The two kernels of the guard case collide on the magic name and are separated by the content hash alone.

    Asserting the collision is what gives the guard its force. The magic-name part is what an entry name built
    from the magic name alone would consist of, so equality there demonstrates that the hash suffix -- and only
    the hash suffix -- is what keeps these two kernels apart.
    """
    first = _ENTRY_NAME.match(entry_names["add_first"])
    other = _ENTRY_NAME.match(entry_names["other_body"])
    assert first is not None, f"unexpected entry name format: {entry_names['add_first']}"
    assert other is not None, f"unexpected entry name format: {entry_names['other_body']}"

    assert first.group("magic") == other.group("magic"), (
        f"the two kernels do not collide on the magic name ({first.group('magic')} vs {other.group('magic')}), "
        f"so this case does not exercise entry-name uniqueness"
    )
    assert first.group("hash") != other.group("hash"), f"the two kernels share the content hash {first.group('hash')}"


def test_sidecar_entry_name_matches_the_emitted_kernel_symbol(entry_names):
    """The sidecar and the generated kernel source name the kernel through two independent code paths.

    The sidecar writer is handed a name and appends the entry suffix itself; the kernel source gets its symbol
    from a template that spells that suffix inline. Nothing compares the two, so a change to either spelling
    would ship silently. This is the same class of divergence the shared entry-name helper removes between
    codegen and the launch path, on the one seam a test can actually reach.
    """
    symbols_by_tag = entry_names[_SYMBOLS_KEY]
    for tag in ("add_first", "add_second", "other_body"):
        symbols = symbols_by_tag[tag]
        assert symbols, f"the {tag} build emitted no kernel source symbol to compare against the sidecar"
        assert entry_names[tag] in symbols, (
            f"{tag}: the sidecar records kernelName {entry_names[tag]!r}, but the emitted kernel source "
            f"declares {symbols!r} -- the two spellings of the entry symbol have diverged"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
