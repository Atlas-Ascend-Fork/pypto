# pypto_pro.language.matmul_acc

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

In-place accumulates the matrix product of lhs_tile and rhs_tile to dst_tile:

```text
dst_tile += lhs_tile × rhs_tile
```

## Prototype

```python
pypto_pro.language.matmul_acc(
    dst_tile: Tile,
    acc_tile: Tile,
    lhs_tile: Tile,
    rhs_tile: Tile,
    /,
    *,
    phase: Optional[AccPhase] = None,
) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| dst_tile | Input/Output| Accumulator operand (in-place accumulation result), of the Tile type, stored in the L0C Buffer, with a shape of [M, N]. The data type is DT_FP32 or DT_INT32, determined by the data type combination of the multiplication inputs. The layout must be NZ. When fractal is not explicitly specified, DT_FP32 and DT_INT32 use 1024. The valid value ranges of M and N are both [1, 4095].|
| acc_tile | Input| Accumulator parameter, of the Tile type, stored in the L0C Buffer, with a shape of [M, N]. The data type is DT_FP32 or DT_INT32, and the layout must be NZ.|
| lhs_tile | Input| Source operand (left matrix), of the Tile type, stored in the L0A Buffer, with a shape of [M, K]. The layout must be NZ. The valid value ranges of M and K are both [1, 4095]. The supported data type combinations are as follows: DT_FP16 × DT_FP16 → DT_FP32, DT_BF16 × DT_BF16 → DT_FP32, DT_FP32 × DT_FP32 → DT_FP32, DT_INT8 × DT_INT8 → DT_INT32, and DT_HF8 × DT_HF8 → DT_FP32. DT_FP8E4M3FN and DT_FP8E5M2 can be combined in any pairwise manner, with the output being DT_FP32.|
| rhs_tile | Input| Source operand (right matrix), of the Tile type, stored in the L0B Buffer, with a shape of [K, N]. The layout must be ZN. The valid value range of N is [1, 4095]. The K dimension must match that of lhs_tile. The data type must satisfy the supported combinations above together with lhs_tile.|
| phase | Input| K-dimensional tile accumulation phase, of the [pypto_pro.language.AccPhase](../basic_data_structures/AccPhase.md) type, optional. When used, it must be correctly paired with the [pypto_pro.language.STPhase](../basic_data_structures/STPhase.md) of store, store_tile, or move. For details, see [Advanced Cube Computation](../../../../guide/programming_guide/pro/advanced_programming/cube_computation_advanced_usage.md).|

## Constraints

- dst_tile and acc_tile must be the same tile.
- Before the matrix multiply-accumulate operation, call [pypto_pro.language.matmul](matmul.md) to initialize dst_tile, and then call this API to perform in-place accumulation.
- When phase is enabled, use pypto_pro.language.AccPhase.Partial for non-final tiles and pypto_pro.language.AccPhase.Final for the final tile, in conjunction with the [pypto_pro.language.STPhase](../basic_data_structures/STPhase.md) of store, store_tile, or move.

## Return Value

None

## Examples

```python
import pypto_pro.language as pl

TILE = 128
K_SIZE_ACC = 256     # Split into 2 tiles for accumulation


@pl.jit(auto_mutex=True)
def matmul_acc_kernel(
    a: pl.Tensor[[TILE, K_SIZE_ACC], pl.DT_FP16],
    b: pl.Tensor[[K_SIZE_ACC, TILE], pl.DT_FP16],
    c: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    # L1 Buffer, L0A Buffer, and L0B Buffer use double buffering (rotated by next()).
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0, 1])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000, mutex_ids=[2, 3])
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left,
                         layout=pl.NZ),
        addrs=0x0000, mutex_ids=[4, 5])
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[6, 7])
    # L0C Buffer: K-dimensional accumulation requires fractal=1024.
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                         fractal=1024),
        addrs=0x0000, mutex_ids=[8])

    with pl.section_cube():
        ac = acc.current()
        for k in pl.range(0, K_SIZE_ACC, TILE):     # K-dimensional tiling (accumulation)
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_left.next()
            br = b_right.next()
            pl.load(cur_a, a, [0, k])
            pl.load(cur_b, b, [k, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if k == 0:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)          # Write the first tile to the accumulator.
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)    # Accumulate the final tile (K = 2 tiles).
        pl.store(c, ac, [0, 0], phase=pl.STPhase.Final)
```
