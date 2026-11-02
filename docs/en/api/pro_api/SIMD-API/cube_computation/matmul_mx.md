# pypto_pro.language.matmul_mx

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T12:57:12.825Z pushedAt=2026-10-28T11:57:28.686Z nl=2 -->

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

Performs matrix multiplication with quantization coefficients:

$$
C_{M \times N} = \left(\mathrm{scale\_a}_{M \times (K/32)} \otimes A_{M \times K}\right) \times \left(\mathrm{scale\_b}_{(K/32) \times N} \otimes B_{K \times N}\right)
$$

Here, ⊗ denotes broadcast multiplication. Every 32 consecutive elements of A and B along the K dimension share one quantization coefficient from scale_a and scale_b, respectively.

## Prototype

```python
pypto_pro.language.matmul_mx(
    dst_tile: Tile,
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
| dst_tile | Output | Destination operand of Tile type, stored in L0C Buffer, with shape [M, N], data type DT_FP32, and layout NZ. |
| lhs_tile | Input | Source operand (A, the left matrix) of Tile type, stored in L0A Buffer, with shape [M, K]. Supported data types are DT_FP8E4M3FN, DT_FP8E5M2, DT_FP4E2M1, and DT_FP4E1M2. The layout must be NZ, and K must match the K dimension of rhs_tile and be a multiple of 64. |
| rhs_tile | Input | Source operand (B, the right matrix) of Tile type, stored in L0B Buffer, with shape [K, N]. Supported data types are DT_FP8E4M3FN, DT_FP8E5M2, DT_FP4E2M1, and DT_FP4E1M2. The data type may differ from that of lhs_tile, but both must be selected from DT_FP8E4M3FN and DT_FP8E5M2, or both from DT_FP4E2M1 and DT_FP4E1M2. The layout must be ZN, and K must match the K dimension of lhs_tile and be a multiple of 64. |
| scale_a | Input | Source operand (left quantization coefficient matrix) of Tile type, stored in L0A_MX Buffer, with data type DT_FP8E8M0 and shape [M, K/32]. The layout is default and only ZZ, and the fractal is default and only 32. Each quantization coefficient corresponds to 32 consecutive elements of matrix A along the K direction. |
| scale_b | Input | Source operand (right quantization coefficient matrix) of Tile type, stored in L0B_MX Buffer, with data type DT_FP8E8M0 and shape [K/32, N]. The layout is default and only NN, and the fractal is default and only 32. Each quantization coefficient corresponds to 32 consecutive elements of matrix B along the K direction. |
| phase | Input | K-dimension block accumulation phase, of [pypto_pro.language.AccPhase](../basic_data_structures/AccPhase.md) type. Optional. Controls the UnitFlag synchronization between matrix computation and L0C Buffer data movement. For how it works with [pypto_pro.language.STPhase](../basic_data_structures/STPhase.md), see [Advanced Cube Computation](../../../../guide/programming_guide/pro/advanced_programming/cube_computation_advanced_usage.md). |

## Constraints

- The quantization coefficient Tiles and the A/B matrix Tiles must satisfy the following hardware address relationship:

  ```text
  addr(scale_a) = addr(lhs_tile) >> 4
  addr(scale_b) = addr(rhs_tile) >> 4
  ```

  The hardware locates scale_a/scale_b based on the base addresses of the A/B matrices. If the addresses do not satisfy the above relationship, the computation result may be incorrect.

## Return Value

None

## Examples

### MXFP8 Matrix Multiplication

The following computes C[128, 128] = A[128, 128] @ B[128, 128]. The K dimension of both A and B contains four quantization coefficient groups, so the logical shapes of the scale_a/scale_b tiles are [128, 4] and [4, 128], respectively, and the physical shapes of the tensors in GM are [128, 2, 2] and [2, 128, 2], respectively.

```python
import os
import pypto_pro.language as pl
import torch

TILE = 128
SCALE_K = TILE // 32


@pl.jit(auto_mutex=True)
def mxfp8_matmul_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP8E4M3FN],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP8E5M2],
    scale_a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 2], pl.DT_FP8E8M0],
    scale_b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 2], pl.DT_FP8E8M0],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP8E4M3FN,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x00000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP8E5M2,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x10000, mutex_ids=[1])
    scale_a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, SCALE_K], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZZ),
        addrs=0x20000, mutex_ids=[2])
    scale_b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[SCALE_K, TILE], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NN),
        addrs=0x21000, mutex_ids=[3])
    a_l0 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP8E4M3FN,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0000, mutex_ids=[4])
    b_l0 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP8E5M2,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0000, mutex_ids=[5])
    scale_a_l0 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, SCALE_K], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.ScaleLeft, layout=pl.ZZ),
        addrs=0x0000, mutex_ids=[6])
    scale_b_l0 = pl.make_tile_group(
        type=pl.TileType(shape=[SCALE_K, TILE], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.ScaleRight, layout=pl.NN),
        addrs=0x0000, mutex_ids=[7])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
        addrs=0x0000, mutex_ids=[8])

    with pl.section_cube():
        al1, bl1 = a_l1.current(), b_l1.current()
        sal1, sbl1 = scale_a_l1.current(), scale_b_l1.current()
        al0, bl0 = a_l0.current(), b_l0.current()
        sal0, sbl0 = scale_a_l0.current(), scale_b_l0.current()
        ac = acc.current()
        pl.load(al1, a, [0, 0])
        pl.load(bl1, b, [0, 0])
        pl.load(sal1, scale_a, [0, 0, 0], order=[0, 1])
        pl.load(sbl1, scale_b, [0, 0, 0], order=[0, 1])
        pl.move(al0, al1)
        pl.move(bl0, bl1)
        pl.move(sal0, sal1)
        pl.move(sbl0, sbl1)
        pl.matmul_mx(ac, al0, bl0, sal0, sbl0)
        pl.store(out, ac, [0, 0])


if __name__ == "__main__":
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)

    # E4M3 encoding 0x38 and E5M2 encoding 0x3C both represent 1.0; E8M0 encoding 127 represents a scaling value of 1.0.
    a = torch.full([TILE, TILE], 0x38, dtype=torch.uint8).view(torch.float8_e4m3fn).to(device)
    b = torch.full([TILE, TILE], 0x3C, dtype=torch.uint8).view(torch.float8_e5m2).to(device)
    scale_a = torch.full([TILE, SCALE_K // 2, 2], 127, device=device, dtype=torch.uint8)
    scale_b = torch.full([SCALE_K // 2, TILE, 2], 127, device=device, dtype=torch.uint8)
    out = torch.zeros([TILE, TILE], device=device, dtype=torch.float32)

    mxfp8_matmul_kernel(a, b, scale_a, scale_b, out)
    torch.npu.synchronize()

    ref = torch.full([TILE, TILE], float(TILE), device=device, dtype=torch.float32)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
    print(f"max diff = {(out - ref).abs().max().item()}")
```

### MXFP4 Matrix Multiplication

The following computes C[128, 128] = A[128, 128] @ B[128, 128]. Both A and B contain four quantization coefficient groups along the K dimension, so the logical shapes of the scale_a/scale_b tiles are [128, 4] and [4, 128], respectively, and the physical shapes of the tensors in GM are [128, 2, 2] and [2, 128, 2], respectively. FP4 matrices are stored as uint8 at runtime, with two FP4 elements packed into each byte.

```python
import os
import pypto_pro.language as pl
import torch

TILE = 128
SCALE_K = TILE // 32


@pl.jit(auto_mutex=True)
def mxfp4_matmul_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP4E2M1],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP4E1M2],
    scale_a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 2], pl.DT_FP8E8M0],
    scale_b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, 2], pl.DT_FP8E8M0],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP4E2M1,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x00000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP4E1M2,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x10000, mutex_ids=[1])
    scale_a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, SCALE_K], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZZ),
        addrs=0x20000, mutex_ids=[2])
    scale_b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[SCALE_K, TILE], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NN),
        addrs=0x21000, mutex_ids=[3])
    a_l0 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP4E2M1,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0000, mutex_ids=[4])
    b_l0 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP4E1M2,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0000, mutex_ids=[5])
    scale_a_l0 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, SCALE_K], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.ScaleLeft, layout=pl.ZZ),
        addrs=0x0000, mutex_ids=[6])
    scale_b_l0 = pl.make_tile_group(
        type=pl.TileType(shape=[SCALE_K, TILE], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.ScaleRight, layout=pl.NN),
        addrs=0x0000, mutex_ids=[7])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
        addrs=0x0000, mutex_ids=[8])

    with pl.section_cube():
        al1, bl1 = a_l1.current(), b_l1.current()
        sal1, sbl1 = scale_a_l1.current(), scale_b_l1.current()
        al0, bl0 = a_l0.current(), b_l0.current()
        sal0, sbl0 = scale_a_l0.current(), scale_b_l0.current()
        ac = acc.current()
        pl.load(al1, a, [0, 0])
        pl.load(bl1, b, [0, 0])
        pl.load(sal1, scale_a, [0, 0, 0], order=[0, 1])
        pl.load(sbl1, scale_b, [0, 0, 0], order=[0, 1])
        pl.move(al0, al1)
        pl.move(bl0, bl1)
        pl.move(sal0, sal1)
        pl.move(sbl0, sbl1)
        pl.matmul_mx(ac, al0, bl0, sal0, sbl0)
        pl.store(out, ac, [0, 0])


def pack_fp4(codes: torch.Tensor) -> torch.Tensor:
    if codes.shape[-1] % 2 != 0:
        raise ValueError("FP4 packing requires an even last dimension")
    low = codes[..., 0::2] & 0x0F
    high = (codes[..., 1::2] & 0x0F) << 4
    return (low | high).contiguous()


if __name__ == "__main__":
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)

    # E2M1 encoding 0x2 and E1M2 encoding 0x4 both represent 1.0; E8M0 encoding 127 represents a scaling value of 1.0.
    a_codes = torch.full([TILE, TILE], 0x2, dtype=torch.uint8)
    b_codes = torch.full([TILE, TILE], 0x4, dtype=torch.uint8)
    a = pack_fp4(a_codes).to(device)
    b = pack_fp4(b_codes).to(device)
    scale_a = torch.full([TILE, SCALE_K // 2, 2], 127, device=device, dtype=torch.uint8)
    scale_b = torch.full([SCALE_K // 2, TILE, 2], 127, device=device, dtype=torch.uint8)
    out = torch.zeros([TILE, TILE], device=device, dtype=torch.float32)

    mxfp4_matmul_kernel(a, b, scale_a, scale_b, out)
    torch.npu.synchronize()

    ref = torch.full([TILE, TILE], float(TILE), device=device, dtype=torch.float32)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
    print(f"max diff = {(out - ref).abs().max().item()}")
```
