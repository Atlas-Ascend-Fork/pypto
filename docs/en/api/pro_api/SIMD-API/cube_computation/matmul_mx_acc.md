# pypto_pro.language.matmul_mx_acc

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

In-place accumulates the matrix multiplication result with grouped quantization coefficients onto the existing accumulator value of dst_tile:

$$
C_{M \times N} = \left(\mathrm{scale\_a}_{M \times (K/32)} \otimes A_{M \times K}\right) \times \left(\mathrm{scale\_b}_{(K/32) \times N} \otimes B_{K \times N}\right) + C_{M \times N}
$$

Here, C denotes dst_tile, A and B denote lhs_tile and rhs_tile, respectively, and ⊗ denotes broadcast multiplication. Every 32 consecutive elements of A and B along the K dimension share one quantization coefficient from scale_a and scale_b, respectively.

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

| Parameter| Input/Output| Description|
|---|---|---|
| dst_tile | Input/Output| Accumulator operand (in-place accumulation result), of the Tile type, stored in the L0C Buffer, with a shape of [M, N]. The data type is DT_FP32, and the layout must be NZ.|
| acc_tile | Input| Accumulator parameter, of the Tile type, stored in the L0C Buffer, with a shape of [M, N]. The data type is DT_FP32, and the layout must be NZ.|
| lhs_tile | Input| Source operand (A, left matrix), of the Tile type, stored in the L0A Buffer, with a shape of [M, K]. The data type can be DT_FP8E4M3FN, DT_FP8E5M2, DT_FP4E2M1, or DT_FP4E1M2. The layout must be NZ. The K dimension must match that of rhs_tile and be a multiple of 64.|
| rhs_tile | Input| Source operand (B, right matrix), of the Tile type, stored in the L0B Buffer, with a shape of [K, N]. The data type can be DT_FP8E4M3FN, DT_FP8E5M2, DT_FP4E2M1, or DT_FP4E1M2. The data type may differ from that of lhs_tile, but both must be selected from DT_FP8E4M3FN and DT_FP8E5M2, or both from DT_FP4E2M1 and DT_FP4E1M2. The layout must be ZN, and the K dimension must match that of lhs_tile and be a multiple of 64.|
| scale_a | Input| Source operand (left quantization coefficient matrix), of the Tile type, stored in the L0A_MX Buffer, with a data type of DT_FP8E8M0 and a shape of [M, K/32]. The layout defaults to and only supports ZZ, and the fractal defaults to and only supports 32. Each quantization coefficient corresponds to 32 consecutive elements of matrix A along the K direction.|
| scale_b | Input| Source operand (right quantization coefficient matrix), of the Tile type, stored in the L0B_MX Buffer, with a data type of DT_FP8E8M0 and a shape of [K/32, N]. The layout defaults to and only supports NN, and the fractal defaults to and only supports 32. Each quantization coefficient corresponds to 32 consecutive elements of matrix B along the K direction.|
| phase | Input| K-dimensional tile accumulation phase, of the [pypto_pro.language.AccPhase](../basic_data_structures/AccPhase.md) type, optional. It is used to control the UnitFlag synchronization between matrix computation and L0C Buffer data movement. For how it works with [pypto_pro.language.STPhase](../basic_data_structures/STPhase.md), see [Advanced Cube Computation](../../../../guide/programming_guide/pro/advanced_programming/cube_computation_advanced_usage.md).|

## Constraints

- dst_tile and acc_tile must be the same tile.

- The quantization coefficient tile and the A/B matrix tile must satisfy the following hardware address relationship:

  ```text
  addr(scale_a) = addr(lhs_tile) >> 4
  addr(scale_b) = addr(rhs_tile) >> 4
  ```

  The hardware locates scale_a/scale_b based on the base addresses of the A/B matrices. When multiple groups of L0A/L0B Buffer tiles are used, each group of addresses must satisfy the above relationship.

- Before the matrix multiply-accumulate operation, call [pypto_pro.language.matmul_mx](matmul_mx.md) to initialize dst_tile, and then call this API to perform in-place accumulation.

## Return Value

None

## Examples

Divide the K direction into two 128-element tiles. Call matmul_mx for the first tile and matmul_mx_acc for the final tile. The complete K direction of A and B contains eight quantization coefficient groups, so the physical shapes of the scale_a/scale_b tensors in GM are [128, 4, 2] and [4, 128, 2], respectively.

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
