# pypto_pro.language.mul_add_dst

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

Performs element-wise multiplication on lhs and rhs, adds the multiplication result to the original data in out element-wise, and writes the final result to out. The formula is out = lhs * rhs + out.

## Prototype

```python
pypto_pro.language.mul_add_dst(out: Tile, lhs: Tile, rhs: Tile) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| out | Output| Destination operand, a UB Tile, supporting DT_FP16 and DT_FP32. Its value before the call participates in the addition; after the call, it stores the computation result.|
| lhs | Input| Source UB Tile that participates in the multiplication. Its data type, Shape, and valid Shape must be consistent with those of out. When it uses a region that does not overlap with out, its original data is retained after the call.|
| rhs | Input| Source UB Tile. Its data type, Shape, and valid Shape must be consistent with those of out.|

## Return Value

None

## Constraints

- The data types of out, lhs, and rhs must be the same. DT_FP16 and DT_FP32 are supported.
- All three Tiles must use the same row-major layout, have consistent valid Shapes, and meet the alignment requirements of the Tile layout.
- The original data in out participates in the computation, so it must be initialized before the call.
- To preserve the source data, the source Tile and out must use non-overlapping storage regions.

## Examples

```python
import pypto_pro.language as pl

M, N = 64, 128


@pl.jit(auto_mutex=True)
def mul_add_dst_kernel(
    x: pl.Tensor[[M, N], pl.DT_FP16],
    y: pl.Tensor[[M, N], pl.DT_FP16],
    z: pl.Tensor[[M, N], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[1])
    tile_c = pl.make_tile_group(type=tt, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_c = tile_c.current()
        pl.load(cur_a, x, [0, 0])
        pl.load(cur_b, y, [0, 0])
        pl.load(cur_c, z, [0, 0])
        pl.mul_add_dst(cur_c, cur_a, cur_b)
        pl.store(z, cur_c, [0, 0])
```

Result example:

<!-- pypto-doc-output:mul_add_dst:start -->
```bash
Input x: [[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [65 65.25 65.5 65.75 66 66.25 66.5 66.75 ...], [97 97.25 97.5 97.75 98 98.25 98.5 98.75 ...], ...]
Input y: [[3 2.875 2.75 2.625 2.5 2.375 2.25 2.125 ...], [-13 -13.125 -13.25 -13.375 -13.5 -13.625 -13.75 -13.875 ...], [-29 -29.125 -29.25 -29.375 -29.5 -29.625 -29.75 -29.875 ...], [-45 -45.125 -45.25 -45.375 -45.5 -45.625 -45.75 -45.875 ...], ...]
Original value of input data z: [[-2 -1.9375 -1.875 -1.8125 -1.75 -1.6875 -1.625 -1.5625 ...], [6 6.0625 6.125 6.1875 6.25 6.3125 6.375 6.4375 ...], [14 14.0625 14.125 14.1875 14.25 14.3125 14.375 14.4375 ...], [22 22.0625 22.125 22.1875 22.25 22.3125 22.375 22.4375 ...], ...]
Output data z: [[1 1.65625 2.25 2.78125 3.25 3.65625 4 4.28125 ...], [-423 -430.5 -438 -445.25 -452.75 -460.5 -468 -475.75 ...], [-1.871000e+03 -1.886000e+03 -1.902000e+03 -1.917000e+03 -1.933000e+03 -1.949000e+03 -1.964000e+03 -1.980000e+03 ...], [-4.344000e+03 -4.364000e+03 -4.388000e+03 -4.412000e+03 -4.436000e+03 -4.460000e+03 -4.484000e+03 -4.508000e+03 ...], ...]
```
<!-- pypto-doc-output:mul_add_dst:end -->
