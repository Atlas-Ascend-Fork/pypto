# pypto_pro.language.expand_min

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

Broadcasts a single-element Tile along the specified dimension to the shape of the source Tile, and then takes the element-wise minimum; that is, each element in the source Tile is compared with the scalar at the corresponding position after broadcasting, and the smaller value is taken.

## Prototype

```python
pypto_pro.language.expand_min(out: Tile, src: Tile, scalar: Tile, *, dim: int=0) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| out | Output| Destination operand of the Tile type, which stores the element-wise minimum results.<br>The supported data types are DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_FP16, DT_FP32, and DT_BF16.<br>The shape must be the same as that of src.|
| src | Input| Source operand, which is of type Tile.<br>The supported data types are DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_FP16, DT_FP32, and DT_BF16.|
| scalar | Input| Specific value to be broadcast, which is of type Tile. It is broadcast to each column or each row for an element-wise minimum.<br>The supported data types are DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_FP16, DT_FP32, and DT_BF16.<br>Its data types must be consistent with those of out and src.<br>If the shape of src is [M, N], then when dim=0, the shape of out is [M, 1]; when dim=1, the shape of out is [1, N].|
| dim | Input| Broadcast direction. 0 means broadcasting along the column direction; 1 means broadcasting along the row direction.|

## Return Value

None

## Constraints

- dim can only be 0 or 1. Let the valid Shape of src be [M, N]: when dim=0, the valid Shape of scalar is [M, 1]; when dim=1, it is [1, N].
- out, src, and scalar must be located in UB and have the same data type; out and src must have consistent Shape and valid Shape.

## Examples

### dim = 0

```python
import pypto_pro.language as pl

M, N = 64, 128


@pl.jit(auto_mutex=True)
def row_expand_min_kernel(
    x: pl.Tensor[[M, N], pl.DT_FP16],
    y: pl.Tensor[[M, 1], pl.DT_FP16],
    z: pl.Tensor[[M, N], pl.DT_FP16],
):
    tile_a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000, mutex_ids=[0])
    tile_row = pl.make_tile_group(
        type=pl.TileType(shape=[64, 1], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_row = tile_row.current()
        cur_out = tile_out.current()
        pl.load(cur_a, x, [0, 0])
        pl.load(cur_row, y, [0, 0])
        pl.expand_min(cur_out, cur_a, cur_row)
        pl.store(z, cur_out, [0, 0])
```

<!-- pypto-doc-output:row_expand_min:start -->
```bash
Input a: [[-3 -2.875 -2.75 -2.625 -2.5 -2.375 -2.25 -2.125 ...], [13 13.125 13.25 13.375 13.5 13.625 13.75 13.875 ...], [29 29.125 29.25 29.375 29.5 29.625 29.75 29.875 ...], [45 45.125 45.25 45.375 45.5 45.625 45.75 45.875 ...], ...]
Input v: [[1.25], [1.5], [1.75], [2], ...]
Output z: [[-3 -2.875 -2.75 -2.625 -2.5 -2.375 -2.25 -2.125 ...], [1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 ...], [1.75 1.75 1.75 1.75 1.75 1.75 1.75 1.75 ...], [2 2 2 2 2 2 2 2 ...], ...]
```
<!-- pypto-doc-output:row_expand_min:end -->

### dim = 1

```python
import pypto_pro.language as pl

M, N = 64, 128


@pl.jit(auto_mutex=True)
def col_expand_min_kernel(
    x: pl.Tensor[[M, N], pl.DT_FP16],
    y: pl.Tensor[[1, N], pl.DT_FP16],
    z: pl.Tensor[[M, N], pl.DT_FP16],
):
    tile_a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000, mutex_ids=[0])
    tile_col = pl.make_tile_group(
        type=pl.TileType(shape=[1, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_col = tile_col.current()
        cur_out = tile_out.current()
        pl.load(cur_a, x, [0, 0])
        pl.load(cur_col, y, [0, 0])
        pl.expand_min(cur_out, cur_a, cur_col, dim=1)
        pl.store(z, cur_out, [0, 0])
```

<!-- pypto-doc-output:col_expand_min:start -->
```bash
Input x: [[-3 -2.875 -2.75 -2.625 -2.5 -2.375 -2.25 -2.125 ...], [13 13.125 13.25 13.375 13.5 13.625 13.75 13.875 ...], [29 29.125 29.25 29.375 29.5 29.625 29.75 29.875 ...], [45 45.125 45.25 45.375 45.5 45.625 45.75 45.875 ...], ...]
Input y: [[1.25 1.5 1.75 2 2.25 2.5 2.75 3 ...]]
Output z: [[-3 -2.875 -2.75 -2.625 -2.5 -2.375 -2.25 -2.125 ...], [1.25 1.5 1.75 2 2.25 2.5 2.75 3 ...], [1.25 1.5 1.75 2 2.25 2.5 2.75 3 ...], [1.25 1.5 1.75 2 2.25 2.5 2.75 3 ...], ...]
```
<!-- pypto-doc-output:col_expand_min:end -->
