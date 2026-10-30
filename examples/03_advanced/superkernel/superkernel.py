#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
SuperKernel (SK) example: SIMD (Tile) elementwise exp with ``sk_cap=pl.SkCap.CAP_4_DISABLE_DCCI_ALL``.

This example demonstrates the SuperKernel adaptation of a PyPTO Pro JIT
SIMD (Tile) kernel:

- The kernel is written with tile-group programming (``pl.load`` / ``pl.exp`` /
  ``pl.store``) inside ``with pl.section_vector()``.
- ``@pl.jit(auto_mutex=True, sk_cap=pl.SkCap.CAP_4_DISABLE_DCCI_ALL)`` enables the SK binding with capability
  bit 4 (disable DCCI). The generated ``call_kernel.cpp`` contains an ``__sk__``
  entry plus a ``SK_BIND`` record next to the normal ``__global__`` entry, so the
  kernel can be fused by the SuperKernel optimization.
- The kernel is wrapped as a torch custom op and compiled together with a native
  torch op (``+ 1.0``) through the ``npugraph_ex`` backend with
  ``super_kernel_optimize`` enabled. The two sub-kernels are fused into one
  superkernel inside the captured ACLGraph.
- The result is compared with eager ``torch.exp(x) + 1``.

Only the SK capability ``pl.SkCap.CAP_4_DISABLE_DCCI_ALL`` is used: no early-start
(``wait_pre_task_end`` / ``set_next_task_start``), no ``schedmode`` and no narrow
scalar arguments.
"""

import argparse
import inspect
import os
import sys

# This sample targets PyPTO Pro (Python 3.10+, and a ``pypto_pro`` build that supports
# ``pl.jit(sk_cap=...)``). Smoke runners may execute with an older interpreter or the
# CANN-bundled older package, so skip cleanly instead of failing the examples job.
if sys.version_info < (3, 10):
    print(
        f"SKIP: {os.path.basename(__file__)} requires Python 3.10+ (pypto_pro); "
        f"current interpreter is {sys.version.split()[0]}"
    )
    raise SystemExit(0)

try:
    import pypto_pro.language as pl
except ImportError as exc:
    print(f"SKIP: pypto_pro is not available in this environment ({exc})")
    raise SystemExit(0)

try:
    supports_sk_cap = "sk_cap" in inspect.signature(pl.jit).parameters
except (TypeError, ValueError):
    supports_sk_cap = False
if not supports_sk_cap:
    print("SKIP: the installed pypto_pro does not support pl.jit(sk_cap=...); "
          "install the matching PyPTO Pro build to run this sample")
    raise SystemExit(0)

import torch
import torch_npu

SHAPE = (1024, 512)
TILE_ROWS = 16
MAX_N = 512
SLOT_BYTES = TILE_ROWS * MAX_N * 4
BLOCK_DIM = 32

# UB addresses: double-buffered input / output tile-group slots.
VA_IN0 = 0
VA_IN1 = VA_IN0 + SLOT_BYTES
VA_OUT0 = VA_IN1 + SLOT_BYTES
VA_OUT1 = VA_OUT0 + SLOT_BYTES


def get_device_id():
    """
    Get and validate TILE_FWK_DEVICE_ID from environment variable.

    Returns:
        int: The device ID if valid, None otherwise.
    """
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("Please set the environment variable TILE_FWK_DEVICE_ID before running:")
        print("  export TILE_FWK_DEVICE_ID=0")
        return None

    try:
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        return device_id
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


@pl.jit(auto_mutex=True, sk_cap=pl.SkCap.CAP_4_DISABLE_DCCI_ALL)
def exp_tile_kernel(
    input_tensor: pl.Tensor[[SHAPE[0], SHAPE[1]], pl.DT_FP32],
    output_tensor: pl.Tensor[[SHAPE[0], SHAPE[1]], pl.DT_FP32],
):
    """
    Elementwise exp computed with a PyPTO Pro Tile/SIMD kernel.

    Each core strides over the row-tile grid with ``pl.get_block_num()`` /
    ``pl.get_block_idx()``. With ``pl.SkCap.CAP_4_DISABLE_DCCI_ALL`` the JIT emits an ``__sk__`` entry
    beside the normal ``__global__`` entry, so a fused sub-kernel takes its own
    block count/index from ``sysArgs`` instead of the container-wide grid.
    """
    tile_type = pl.TileType(
        shape=[TILE_ROWS, MAX_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
    )
    in_group = pl.make_tile_group(type=tile_type, addrs=[VA_IN0, VA_IN1], mutex_ids=[0, 1])
    out_group = pl.make_tile_group(type=tile_type, addrs=[VA_OUT0, VA_OUT1], mutex_ids=[2, 3])

    with pl.section_vector():
        rows = input_tensor.shape[0]
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx()
        num_tiles = (rows + TILE_ROWS - 1) // TILE_ROWS

        for tile_id in pl.range(core_id, num_tiles, num_cores):
            row_off = tile_id * TILE_ROWS
            valid_rows = pl.min(TILE_ROWS, rows - row_off)

            in_slot = in_group.next()
            pl.set_validshape(in_slot, [valid_rows, MAX_N])
            pl.load(in_slot, input_tensor, [row_off, 0])

            out_slot = out_group.next()
            pl.set_validshape(out_slot, [valid_rows, MAX_N])
            pl.exp(out_slot, in_slot)
            pl.store(output_tensor, out_slot, [row_off, 0])


@torch.library.custom_op("pypto_sk::exp", mutates_args=())
def _exp(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    exp_tile_kernel[None, BLOCK_DIM](x, output)
    return output


@_exp.register_fake
def _exp_fake(x):
    return torch.empty_like(x)


class ExpAddOne(torch.nn.Module):
    """``exp(x) + 1``: one PyPTO Pro sub-kernel plus one torch native sub-kernel."""

    def forward(self, x):
        return _exp(x) + 1.0


def test_superkernel(device_id=None) -> None:
    if not device_id:
        device_id = torch_npu.npu.current_device()
    else:
        torch_npu.npu.set_device(device_id)

    x = torch.randn(SHAPE, dtype=torch.float, device=f'npu:{device_id}')

    # Warm the PyPTO Pro kernel outside capture (the first call JIT-compiles it).
    _exp(x)
    torch_npu.npu.synchronize()

    # Compile with npugraph_ex and super_kernel_optimize: the exp sub-kernel and
    # the torch add sub-kernel are fused into one superkernel in the static graph.
    model = torch.compile(
        ExpAddOne(),
        backend="npugraph_ex",
        dynamic=False,
        fullgraph=True,
        options={"static_kernel_compile": True, "super_kernel_optimize": True, "clone_input": False},
    )

    y = model(x)
    torch_npu.npu.synchronize()

    golden = torch.exp(x.cpu()) + 1.0
    torch.testing.assert_close(y.cpu(), golden, rtol=1e-4, atol=1e-4)

    print(f"Input shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    print("PASS: SuperKernel example passed (sk_cap=CAP_4_DISABLE_DCCI_ALL, SIMD Tile exp + torch add)")


def main():
    """Run the SuperKernel example.

    Usage:
        python3 superkernel.py          # Run example
        python3 superkernel.py --list   # List available examples
    """
    parser = argparse.ArgumentParser(
        description="PyPTO SuperKernel example (SIMD Tile exp, sk_cap=CAP_4_DISABLE_DCCI_ALL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run the example
  %(prog)s --list       List all available examples
        """,
    )
    parser.add_argument(
        'example_id', type=str, nargs='?', help='Example ID to run. If not specified, the example will run.'
    )
    parser.add_argument('--list', action='store_true', help='List all available examples and exit')

    args = parser.parse_args()

    examples = {
        "superkernel::test_superkernel": {
            'name': 'SuperKernel',
            'description': 'SIMD Tile exp with SK binding (cap=disable DCCI), fused with a torch add',
            'function': test_superkernel,
            'requires_npu': True,
        }
    }

    if args.list:
        print("\n" + "=" * 60)
        print("Available Examples")
        print("=" * 60 + "\n")
        for ex_id, ex_info in sorted(examples.items()):
            npu_req = " (Requires NPU)" if ex_info['requires_npu'] else " (No NPU required)"
            print(f"  {ex_id}. {ex_info['name']}{npu_req}")
            print(f"     {ex_info['description']}\n")
        return

    if args.example_id is not None and args.example_id not in examples:
        print(f"ERROR: Invalid example ID: {args.example_id}")
        print(f"Valid example IDs are: {', '.join(map(str, sorted(examples.keys())))}")
        print("\nUse --list to see all available examples.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("PyPTO SuperKernel Example (SIMD Tile exp, sk_cap=CAP_4_DISABLE_DCCI_ALL)")
    print("=" * 60 + "\n")

    examples_to_run = (
        [(args.example_id, examples[args.example_id])] if args.example_id is not None else list(examples.items())
    )

    device_id = get_device_id()
    if device_id is None:
        return
    torch_npu.npu.set_device(device_id)

    try:
        for ex_id, ex_info in examples_to_run:
            print(f"Running Example {ex_id}: {ex_info['name']}")
            ex_info['function'](device_id)
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
