# pypto_pro.language.fused_mul_add_relu

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

Multiplies the original data in out by a element-wise, adds the multiplication result to b element-wise, and then applies ReLU activation, setting elements less than 0 to 0. The computation result is written back to out, overwriting the original data in out.

## Prototype

```python
pypto_pro.language.fused_mul_add_relu(out: Tile, a: Tile, b: Tile) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| out | Output| Destination operand, a UB Tile, supporting DT_FP16 and DT_FP32. Its value before the call participates in the multiplication; after the call, it stores the computation result.|
| a | Input| Source UB Tile that participates in the multiplication. Its data type, Shape, and valid Shape must be consistent with those of out. When it uses a region that does not overlap with out, its original data is retained after the call.|
| b | Input| Source UB Tile. Its data type, Shape, and valid Shape must be consistent with those of out.|

## Return Value

None

## Constraints

- The data types of out, a, and b must be the same. DT_FP16 and DT_FP32 are supported.
- All three Tiles must use the same row-major layout, have consistent valid Shapes, and meet the alignment requirements of the Tile layout.
- The original data in out participates in the computation, so it must be initialized before the call.
- To preserve the source data, the source Tile and out must use non-overlapping storage regions.

## Examples

```python
import pypto_pro.language as pl

M, N = 64, 128


@pl.jit(auto_mutex=True)
def fused_mul_add_relu_kernel(
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
        pl.fused_mul_add_relu(cur_c, cur_a, cur_b)
        pl.store(z, cur_c, [0, 0])
```

Result example:

<!-- pypto-doc-output:fused_mul_add_relu:start -->
```bash
Input x: [[-2 -1.75 -1.5 -1.25 -1 -0.75 -0.5 -0.25 ...], [30 30.25 30.5 30.75 31 31.25 31.5 31.75 ...], [62 62.25 62.5 62.75 63 63.25 63.5 63.75 ...], [94 94.25 94.5 94.75 95 95.25 95.5 95.75 ...], ...]
Input y: [[1 1.125 1.25 1.375 1.5 1.625 1.75 1.875 ...], [17 17.125 17.25 17.375 17.5 17.625 17.75 17.875 ...], [33 33.125 33.25 33.375 33.5 33.625 33.75 33.875 ...], [49 49.125 49.25 49.375 49.5 49.625 49.75 49.875 ...], ...]
Original value of input z: [[3 2.9375 2.875 2.8125 2.75 2.6875 2.625 2.5625 ...], [-5 -5.0625 -5.125 -5.1875 -5.25 -5.3125 -5.375 -5.4375 ...], [-13 -13.0625 -13.125 -13.1875 -13.25 -13.3125 -13.375 -13.4375 ...], [-21 -21.0625 -21.125 -21.1875 -21.25 -21.3125 -21.375 -21.4375 ...], ...]
Output z: [[0 0 0 0 0 0 0.4375 1.234375 ...], [0 0 0 0 0 0 0 0 ...], [0 0 0 0 0 0 0 0 ...], [0 0 0 0 0 0 0 0 ...], ...]
```
<!-- pypto-doc-output:fused_mul_add_relu:end -->
