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
"""``pl.jit(sk_cap=...)``: the merged superkernel entry/capability parameter.

``sk_cap`` replaces the former ``sk_bind``/``sk_cap`` pair — the capability mask only
means anything once the kernel is bound, so ``None`` means "no superkernel entry" and a
``pl.SkCap`` member (conventionally ``CAP_4_DISABLE_DCCI_ALL``) both enables and
configures it.
"""
from pypto_pro._errors import InvalidArgument
import pypto_pro.language as pl
from pypto_pro.runtime.compile_config import get_jit_compile_config
from pypto_pro.runtime.jit import (
    CodegenResult,
    _make_global_entry,
    _make_sk_bind_block,
    _make_sk_caller_parts,
    _resolve_sk_cap,
)
from pypto_pro.runtime.platform import NpuArch
import pytest


def _target():
    return get_jit_compile_config().resolve_kernel_target(NpuArch.DAV_3510, has_cube=False, has_vector=True)


def _sk_bind_line(sk_cap: int) -> str:
    source = _make_sk_bind_block("k", [], _target(), sk_cap=sk_cap)
    return next(line for line in source.splitlines() if line.startswith("SK_BIND"))


def _cg(**overrides) -> CodegenResult:
    return CodegenResult(
        build_dir="", content="", kernel_name="k", kernel_params=[],
        caller_cross_core_sync=False, bisheng_cross_sync=False, needs_print_debug=False,
        **overrides,
    )


def _entry_code(sk_cap: int | None, **cg_overrides) -> str:
    cg = _cg(**cg_overrides)
    prelude, sk_entry = (
        _make_sk_caller_parts(cg, _target(), sk_cap) if sk_cap is not None else ("", "")
    )
    return _make_global_entry("k", cg.entry_params, target=_target(), body_prelude=prelude) + sk_entry


# ---------------------------------------------------------------------------
# _resolve_sk_cap: None disables, a SkCap member (or combination) is the mask
# ---------------------------------------------------------------------------
def test_none_means_no_superkernel_entry():
    assert _resolve_sk_cap(None) is None


def test_members_match_the_official_sk_bind_bits():
    assert {member.name: int(member) for member in pl.SkCap} == {
        "CAP_1_EARLY_START_WAIT": 1,
        "CAP_2_EARLY_START_SET": 2,
        "CAP_4_DISABLE_DCCI_ALL": 4,
        "CAP_8_DISABLE_BATCHMODE_CHECK": 8,
        "CAP_16_BLOCKDIM_SCALE_UP": 16,
    }


@pytest.mark.parametrize(
    "sk_cap",
    [
        pl.SkCap.CAP_1_EARLY_START_WAIT,
        pl.SkCap.CAP_2_EARLY_START_SET,
        pl.SkCap.CAP_1_EARLY_START_WAIT | pl.SkCap.CAP_2_EARLY_START_SET,
        pl.SkCap.CAP_4_DISABLE_DCCI_ALL,
        pl.SkCap.CAP_1_EARLY_START_WAIT | pl.SkCap.CAP_4_DISABLE_DCCI_ALL,
        pl.SkCap.CAP_8_DISABLE_BATCHMODE_CHECK,
        pl.SkCap.CAP_16_BLOCKDIM_SCALE_UP,
        pl.SkCap.CAP_4_DISABLE_DCCI_ALL | pl.SkCap.CAP_16_BLOCKDIM_SCALE_UP,
        pl.SkCap(31),
    ],
)
def test_valid_masks_pass_through(sk_cap):
    assert _resolve_sk_cap(sk_cap) == int(sk_cap)


def test_zero_is_rejected():
    """0 requests the entry while declaring no capability; None already means no entry."""
    with pytest.raises(ValueError, match="sk_cap=0 declares no capability"):
        _resolve_sk_cap(pl.SkCap(0))


def test_unknown_bits_are_rejected():
    """IntFlag accepts unknown values, so the explicit mask check stays."""
    with pytest.raises(ValueError, match="unknown capability bits"):
        _resolve_sk_cap(pl.SkCap(32))


@pytest.mark.parametrize("sk_cap", [True, False, 0, 4, 4 | 16, 32, 4.0, "4"])
def test_raw_values_are_rejected(sk_cap):
    """The member name is the capability contract; raw ints (and bool/str) are refused."""
    with pytest.raises(TypeError, match="sk_cap must be a pl.SkCap member or None"):
        _resolve_sk_cap(sk_cap)


@pytest.mark.parametrize("value", [32, 33, 64, -1])
def test_out_of_range_bits_are_rejected(value):
    """IntFlag builds pseudo-members for unknown values; the mask check rejects them."""
    with pytest.raises(ValueError, match="unknown capability bits"):
        _resolve_sk_cap(pl.SkCap(value))


# ---------------------------------------------------------------------------
# The mask reaches the SK_BIND record
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sk_cap", [4, 4 | 16, 31])
def test_mask_is_written_into_sk_bind(sk_cap):
    assert _sk_bind_line(sk_cap) == f"SK_BIND(k, {sk_cap}, k_sk<0>, k_sk<1>, k_sk<2>, k_sk<3>);"


def test_bare_enable_is_the_callers_choice_of_mask():
    """There is no implicit default: ``None`` disables, the caller states the mask."""
    assert _sk_bind_line(4) == "SK_BIND(k, 4, k_sk<0>, k_sk<1>, k_sk<2>, k_sk<3>);"


# ---------------------------------------------------------------------------
# Sub-kernel block count normalization (MIX 1:2 AIV count vs logical block count)
# ---------------------------------------------------------------------------
def test_tile_block_num_is_normalized_from_aiv_granular_sk_count():
    """``skNumBlocks`` is AIV-granular; Tile ``get_block_num()`` must stay logical."""
    source = _make_sk_bind_block("k", [], _target(), sk_cap=4, uses_block_num=True)
    assert (
        "__pypto_seed_block_num((uint32_t)(sysArgs->skNumBlocks / AscendC::GetTaskRatio()));"
        in source
    )


def test_tile_block_idx_is_not_normalized():
    """``skBlockIdx`` is already AIV-granular, matching ``get_block_idx()``."""
    source = _make_sk_bind_block("k", [], _target(), sk_cap=4, uses_block_idx=True)
    assert "__pypto_seed_block_idx((uint32_t)sysArgs->skBlockIdx);" in source
    assert "__pypto_seed_block_num(" not in source


# ---------------------------------------------------------------------------
# global entry responsibility: signature + schedmode only, SK setup via prelude
# ---------------------------------------------------------------------------
def test_global_entry_takes_an_opaque_prelude_and_stays_sk_agnostic():
    source = _make_global_entry("k", [], target=_target(), schedmode=1, body_prelude="    seed();\n")
    assert "__schedmode__(1) __global__ __vector__ void k(" in source
    assert "    seed();\n    k_impl();" in source


def test_non_fused_entry_seeds_block_meta_records():
    """The shared impl reads the pypto_block_meta.h records, so the non-fused entry must
    seed them (the seed helpers exist in the SK build only)."""
    source = _entry_code(4, uses_block_num=True, uses_block_idx=True)
    assert "__pypto_seed_block_num((uint32_t)__pypto_raw_block_num());" in source
    assert "__pypto_seed_block_idx((uint32_t)__pypto_raw_block_idx());" in source


def test_non_sk_build_does_not_reference_the_seed_helpers():
    """The seed helpers exist in the __PYPTO_SK_BLOCK__ header branch only: without
    sk_cap the entry must not reference them, even if the usage flags are set."""
    source = _entry_code(None, uses_block_num=True, uses_block_idx=True)
    assert "__pypto_seed_block_num(" not in source
    assert "__pypto_seed_block_idx(" not in source
    # The entry itself is still rendered (the launcher always needs it).
    assert "__global__" in source


def test_non_fused_entry_clears_stale_early_start_config():
    """Early-start capable binaries always zero the block-local config on a plain launch.

    The V3 SetNextTaskStart / WaitPreTaskEnd implementations read
    ``g_super_kernel_early_start_config``; a stale non-zero value left by a previous
    superkernel task on the same core would emit an unpaired sync and stall the launch.
    The reset is unconditional for SK builds carrying cap bits 1/2: the header that
    declares the symbol is included for every SK build, so no content detection is
    needed.
    """
    reset = "g_super_kernel_early_start_config = 0;"
    # Cap bits 1/2: always reset, whether or not this kernel uses the sync ops.
    source = _entry_code(2)
    assert reset in source
    # No early-start cap bit: the V3 config is never read, no reset.
    source = _entry_code(4)
    assert reset not in source
    # Non-SK build: no SK entry and no reset.
    source = _entry_code(None)
    assert reset not in source


# ---------------------------------------------------------------------------
# pl.jit surface
# ---------------------------------------------------------------------------
def test_jit_accepts_cap_and_combination():
    @pl.jit(auto_mutex=False, sk_cap=pl.SkCap.CAP_4_DISABLE_DCCI_ALL)
    def one(x: pl.Tensor[[64], pl.DT_FP32]):
        pl.system.bar_all()

    @pl.jit(
        auto_mutex=False,
        sk_cap=pl.SkCap.CAP_4_DISABLE_DCCI_ALL | pl.SkCap.CAP_16_BLOCKDIM_SCALE_UP,
    )
    def combined(x: pl.Tensor[[64], pl.DT_FP32]):
        pl.system.bar_all()

    assert one._sk_cap == 4
    assert combined._sk_cap == 4 | 16


def test_jit_defaults_to_no_superkernel_entry():
    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
        pl.system.bar_all()

    assert kernel._sk_cap is None


def test_removed_sk_bind_parameter_is_rejected():
    with pytest.raises(InvalidArgument, match="unexpected keyword argument\\(s\\): sk_bind"):

        @pl.jit(auto_mutex=False, sk_bind=True)
        def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
            pl.system.bar_all()


@pytest.mark.parametrize("sk_cap", [True, 0, 4, 4 | 16, 32, "4"])
def test_jit_rejects_raw_caps(sk_cap):
    """Raw ints (the previous API) are no longer accepted."""
    with pytest.raises(TypeError, match="pl.SkCap"):

        @pl.jit(auto_mutex=False, sk_cap=sk_cap)
        def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
            pl.system.bar_all()


@pytest.mark.parametrize("sk_cap", [pl.SkCap(0), pl.SkCap(32)])
def test_jit_rejects_invalid_cap_values(sk_cap):
    with pytest.raises(ValueError):

        @pl.jit(auto_mutex=False, sk_cap=sk_cap)
        def kernel(x: pl.Tensor[[64], pl.DT_FP32]):
            pl.system.bar_all()
