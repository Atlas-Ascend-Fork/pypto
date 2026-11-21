# pypto_pro.language.expand_div

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

Broadcasts a single-element Tile along the specified dimension to the shape of the source Tile, and then performs element-wise division; that is, each element in the source Tile is divided by the scalar at the corresponding position after the single-element Tile is broadcast.

## Prototype

```python
pypto_pro.language.expand_div(out: Tile, src: Tile, scalar: Tile, *, dim: int=0) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| out | Output| Destination operand of type Tile, which stores the result of element-wise division.<br>The supported data types are DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_FP16, DT_FP32, and DT_BF16.<br>The shape must be the same as that of src.<br>It can be the same Tile as src to implement in-place computation.|
| src | Input| Source operand, which is of type Tile.<br>The supported data types are DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_FP16, DT_FP32, and DT_BF16.|
| scalar | Input| Specific value to be broadcast, which is of type Tile. It is broadcast to each column or each row for element-wise division.<br>The supported data types are DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_FP16, DT_FP32, and DT_BF16.<br>Its data types must be consistent with those of out and src.<br>If the shape of src is [M, N], then when dim=0, its shape is [M, 1]; when dim=1, its shape is [1, N].<br>Element values must not be 0; otherwise, the hardware behavior is undefined.|
| dim | Input| Broadcast direction. 0 means broadcasting along the column direction; 1 means broadcasting along the row direction.|

## Return Value

None

## Constraints

- dim can only be 0 or 1. Let the valid Shape of src be [M, N]: when dim=0, the valid Shape of scalar is [M, 1]; when dim=1, it is [1, N].
- out, src, and scalar must be located in UB and have the same data type; out and src must have consistent Shape and valid Shape.
- Valid elements of scalar must not be 0. Integer types use integer division and cannot be used to preserve fractional quotients.

## Examples

### dim = 0

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def row_expand_div_kernel(
    x: pl.Tensor[[64, 128], pl.DT_FP32],
    y: pl.Tensor[[64, 1], pl.DT_FP32],
    out: pl.Tensor[[64, 128], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tt_v = pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec,
                       layout=pl.DN)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_v = pl.make_tile_group(type=tt_v, addrs=0x8000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_v = tile_v.current()
        cur_out = tile_out.current()
        pl.load(cur_a, x, [0, 0])
        pl.load(cur_v, y, [0, 0])
        pl.expand_div(cur_out, cur_a, cur_v)
        pl.store(out, cur_out, [0, 0])
```

<!-- pypto-doc-output:row_expand_div:start -->
```bash
Input x: [[-3 -2.875 -2.75 -2.625 -2.5 -2.375 -2.25 -2.125 ...], [13 13.125 13.25 13.375 13.5 13.625 13.75 13.875 ...], [29 29.125 29.25 29.375 29.5 29.625 29.75 29.875 ...], [45 45.125 45.25 45.375 45.5 45.625 45.75 45.875 ...], ...]
Input y: [[1.25], [1.5], [1.75], [2], ...]
Output out: [[-2.4 -2.3 -2.2 -2.1 -2 -1.9 -1.8 -1.7 ...], [8.666667 8.75 8.833333 8.916667 9 9.083333 9.166667 9.25 ...], [16.571428 16.642857 16.714285 16.785715 16.857143 16.928572 17 17.071428 ...], [22.5 22.5625 22.625 22.6875 22.75 22.8125 22.875 22.9375 ...], ...]
```
<!-- pypto-doc-output:row_expand_div:end -->

### dim = 1

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def col_expand_div_kernel(
    x: pl.Tensor[[64, 128], pl.DT_FP32],
    y: pl.Tensor[[1, 128], pl.DT_FP32],
    out: pl.Tensor[[64, 128], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tt_v = pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_v = pl.make_tile_group(type=tt_v, addrs=0x8000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_v = tile_v.current()
        cur_out = tile_out.current()
        pl.load(cur_a, x, [0, 0])
        pl.load(cur_v, y, [0, 0])
        pl.expand_div(cur_out, cur_a, cur_v, dim=1)
        pl.store(out, cur_out, [0, 0])
```

<!-- pypto-doc-output:col_expand_div:start -->
```bash
Input x: [[-3 -2.875 -2.75 -2.625 -2.5 -2.375 -2.25 -2.125 ...], [13 13.125 13.25 13.375 13.5 13.625 13.75 13.875 ...], [29 29.125 29.25 29.375 29.5 29.625 29.75 29.875 ...], [45 45.125 45.25 45.375 45.5 45.625 45.75 45.875 ...], ...]
Input y: [[1.25 1.5 1.75 2 2.25 2.5 2.75 3 ...]]
Output out: [[-2.4 -1.916667 -1.571429 -1.3125 -1.111111 -0.95 -0.818182 -0.708333 ...], [10.4 8.75 7.571428 6.6875 6 5.45 5 4.625 ...], [23.200001 19.416666 16.714285 14.6875 13.111111 11.85 10.818182 9.958333 ...], [36 30.083332 25.857143 22.6875 20.222221 18.25 16.636364 15.291666 ...], ...]
```
<!-- pypto-doc-output:col_expand_div:end -->
