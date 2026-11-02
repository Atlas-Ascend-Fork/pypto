# pypto_pro.language.matmul_mx_acc

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T12:57:51.704Z pushedAt=2026-10-28T11:57:28.689Z nl=2 -->

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Not supported
<!-- end id3 -->

## Description

Accumulates a matrix multiplication result with grouped quantization coefficients onto the existing accumulator value:

$$
C_{M \times N} = \left(\mathrm{scale\_a}_{M \times (K/32)} \otimes A_{M \times K}\right) \times \left(\mathrm{scale\_b}_{(K/32) \times N} \otimes B_{K \times N}\right) + C_{M \times N}
$$

Here, ⊗ denotes broadcast multiplication. Along the K dimension, every 32 consecutive elements of A and B share one quantization coefficient from scale_a and scale_b, respectively.

## Prototype

```python
pypto_pro.language.matmul_mx_acc(
    dst_tile: Tile,
    acc_tile: Tile,
    lhs_tile: Tile,
    rhs_tile: Tile,
    /,
    scale_a: Tile,
    scale_b: Tile,
    *,
    phase: Optional[AccPhase] = None,
) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| dst_tile | Output | Destination operand of Tile type. Storage space is L0C Buffer, shape is [M, N], data type is DT_FP32, and layout must be NZ. |
| acc_tile | Input | Source operand (existing accumulation result) of Tile type. Storage space is L0C Buffer, data type is DT_FP32, layout must be NZ, and shape must be identical to dst_tile. It usually shares the same Tile as dst_tile for in-place accumulation. Its content must be initialized by a preceding matmul_mx or matmul_mx_acc. |
| lhs_tile | Input | Source operand (A, left matrix) of Tile type. Storage space is L0A Buffer, shape is [M, K]. Supported data types are DT_FP8E4M3FN, DT_FP8E5M2, DT_FP4E2M1, and DT_FP4E1M2. Layout must be NZ, and K must match the K dimension of rhs_tile and be a multiple of 64. |
| rhs_tile | Input | Source operand (B, right matrix) of Tile type. Storage space is L0B Buffer, shape is [K, N]. Supported data types are DT_FP8E4M3FN, DT_FP8E5M2, DT_FP4E2M1, and DT_FP4E1M2. Its data type may differ from that of lhs_tile, but both must be selected from DT_FP8E4M3FN and DT_FP8E5M2, or both from DT_FP4E2M1 and DT_FP4E1M2. Layout must be ZN, and K must match the K dimension of lhs_tile and be a multiple of 64. |
| scale_a | Input | Source operand (left quantization coefficient matrix) of Tile type, located in L0A_MX Buffer. Data type is DT_FP8E8M0, shape is [M, K/32]. Layout is ZZ by default and only ZZ is supported, and fractal is 32 by default and only 32 is supported. Each quantization coefficient corresponds to 32 consecutive elements of matrix A along the K dimension. |
| scale_b | Input | Source operand (right quantization coefficient matrix) of Tile type, located in L0B_MX Buffer. Data type is DT_FP8E8M0, shape is [K/32, N]. Layout is NN by default and only NN is supported, and fractal is 32 by default and only 32 is supported. Each quantization coefficient corresponds to 32 consecutive elements of matrix B along the K dimension. |
| phase | Input | K-dimension block accumulation phase, of [pypto_pro.language.AccPhase](../basic_data_structures/AccPhase.md) type, optional. It controls the UnitFlag synchronization between matrix computation and L0C Buffer data movement. For how it works with [pypto_pro.language.STPhase](../basic_data_structures/STPhase.md), see [Advanced Cube Computation](../../../../guide/programming_guide/pro/advanced_programming/cube_computation_advanced_usage.md). |

## Constraints

- The quantization coefficient Tiles and the A/B matrix Tiles must satisfy the following hardware address relationship:

  ```text
  addr(scale_a) = addr(lhs_tile) >> 4
  addr(scale_b) = addr(rhs_tile) >> 4
  ```

  The hardware locates scale_a/scale_b based on the base address of the A/B matrices. When multiple groups of L0A/L0B Tiles are used, the addresses of each group must satisfy the above relationship.

- During matrix multiply-accumulate operations, call [matmul_mx](matmul_mx.md) to establish the initial result for acc_tile, and then call this API for subsequent accumulation operations.

## Return Value

None

## Examples

Divide the K dimension into two blocks of 128 elements. Call matmul_mx for the first block and matmul_mx_acc for the last block. The complete K dimension of both A and B contains eight quantization coefficient groups, so the physical shapes of the scale_a/scale_b tensors in GM are [128, 4, 2] and [4, 128, 2], respectively.

```python
import os
import pypto_pro.language as pl
import torch

M_SIZE = 128
K_SIZE = 256
N_SIZE = 128
TILE_K = 128
SCALE_K = TILE_K // 32


@pl.jit(auto_mutex=True)
def mxfp8_matmul_acc_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP8E4M3FN],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP8E5M2],
    scale_a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 2], pl.DT_FP8E8M0],
    scale_b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 2], pl.DT_FP8E8M0],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M_SIZE, TILE_K], dtype=pl.DT_FP8E4M3FN,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x00000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_K, N_SIZE], dtype=pl.DT_FP8E5M2,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x10000, mutex_ids=[1])
    scale_a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M_SIZE, SCALE_K], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZZ),
        addrs=0x20000, mutex_ids=[2])
    scale_b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[SCALE_K, N_SIZE], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NN),
        addrs=0x21000, mutex_ids=[3])
    a_l0 = pl.make_tile_group(
        type=pl.TileType(shape=[M_SIZE, TILE_K], dtype=pl.DT_FP8E4M3FN,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0000, mutex_ids=[4])
    b_l0 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_K, N_SIZE], dtype=pl.DT_FP8E5M2,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0000, mutex_ids=[5])
    scale_a_l0 = pl.make_tile_group(
        type=pl.TileType(shape=[M_SIZE, SCALE_K], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.ScaleLeft, layout=pl.ZZ),
        addrs=0x0000, mutex_ids=[6])
    scale_b_l0 = pl.make_tile_group(
        type=pl.TileType(shape=[SCALE_K, N_SIZE], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.ScaleRight, layout=pl.NN),
        addrs=0x0000, mutex_ids=[7])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[M_SIZE, N_SIZE], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
        addrs=0x0000, mutex_ids=[8])

    with pl.section_cube():
        al1, bl1 = a_l1.current(), b_l1.current()
        sal1, sbl1 = scale_a_l1.current(), scale_b_l1.current()
        al0, bl0 = a_l0.current(), b_l0.current()
        sal0, sbl0 = scale_a_l0.current(), scale_b_l0.current()
        ac = acc.current()

        for ki in pl.range(0, K_SIZE, TILE_K):
            pl.load(al1, a, [0, ki])
            pl.load(bl1, b, [ki, 0])
            pl.load(sal1, scale_a, [0, ki // 64, 0], order=[0, 1])
            pl.load(sbl1, scale_b, [ki // 64, 0, 0], order=[0, 1])
            pl.move(al0, al1)
            pl.move(bl0, bl1)
            pl.move(sal0, sal1)
            pl.move(sbl0, sbl1)

            if ki == 0:
                pl.matmul_mx(ac, al0, bl0, sal0, sbl0, phase=pl.AccPhase.Partial)
            else:
                pl.matmul_mx_acc(ac, ac, al0, bl0, sal0, sbl0, phase=pl.AccPhase.Final)

        pl.store(out, ac, [0, 0], phase=pl.STPhase.Final)


if __name__ == "__main__":
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)

    # E4M3 encoding 0x38 and E5M2 encoding 0x3C both represent 1.0; E8M0 encoding 127 represents a scaling value of 1.0.
    a = torch.full([M_SIZE, K_SIZE], 0x38, dtype=torch.uint8).view(torch.float8_e4m3fn).to(device)
    b = torch.full([K_SIZE, N_SIZE], 0x3C, dtype=torch.uint8).view(torch.float8_e5m2).to(device)
    scale_a = torch.full([M_SIZE, K_SIZE // 64, 2], 127, device=device, dtype=torch.uint8)
    scale_b = torch.full([K_SIZE // 64, N_SIZE, 2], 127, device=device, dtype=torch.uint8)
    out = torch.zeros([M_SIZE, N_SIZE], device=device, dtype=torch.float32)

    mxfp8_matmul_acc_kernel(a, b, scale_a, scale_b, out)
    torch.npu.synchronize()

    ref = torch.full([M_SIZE, N_SIZE], float(K_SIZE), device=device, dtype=torch.float32)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
    print(f"max diff = {(out - ref).abs().max().item()}")
```
