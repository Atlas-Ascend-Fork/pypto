# pypto_pro.language.expand_sub

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

Broadcasts a single-element Tile along the specified dimension to the shape of the source Tile, and then performs element-wise subtraction; that is, each element in the source Tile is subtracted by the scalar at the corresponding position after broadcasting.

## Prototype

```python
pypto_pro.language.expand_sub(out: Tile, src: Tile, scalar: Tile, *, dim: int=0) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| out | Output| Destination operand of the Tile type, which stores the element-wise subtraction result.<br>The supported data types are DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_FP16, DT_FP32, and DT_BF16.<br>The shape must be the same as that of src.<br>It can be the same Tile as src to implement in-place computation.|
| src | Input| Source operand, which is of type Tile.<br>The supported data types are DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_FP16, DT_FP32, and DT_BF16.|
| scalar | Input| Specific value to be broadcast, which is of type Tile. It is broadcast to each column or each row for element-wise subtraction.<br>The supported data types are DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_FP16, DT_FP32, and DT_BF16.<br>Its data types must be consistent with those of out and src.<br>If the shape of src is [M, N], then when dim=0, the shape of out is [M, 1]; when dim=1, the shape of out is [1, N].|
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


@pl.jit(auto_mutex=True)
def row_expand_sub_kernel(
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
        pl.expand_sub(cur_out, cur_a, cur_v)
        pl.store(out, cur_out, [0, 0])
```

<!-- pypto-doc-output:row_expand_sub:start -->
```bash
Input x: [[-3 -2.875 -2.75 -2.625 -2.5 -2.375 -2.25 -2.125 ...], [13 13.125 13.25 13.375 13.5 13.625 13.75 13.875 ...], [29 29.125 29.25 29.375 29.5 29.625 29.75 29.875 ...], [45 45.125 45.25 45.375 45.5 45.625 45.75 45.875 ...], ...]
Input y: [[1.25], [1.5], [1.75], [2], ...]
Output out: [[-4.25 -4.125 -4 -3.875 -3.75 -3.625 -3.5 -3.375 ...], [11.5 11.625 11.75 11.875 12 12.125 12.25 12.375 ...], [27.25 27.375 27.5 27.625 27.75 27.875 28 28.125 ...], [43 43.125 43.25 43.375 43.5 43.625 43.75 43.875 ...], ...]
```
<!-- pypto-doc-output:row_expand_sub:end -->

### dim = 1

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def col_expand_sub_kernel(
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
        pl.expand_sub(cur_out, cur_a, cur_v, dim=1)
        pl.store(out, cur_out, [0, 0])
```

<!-- pypto-doc-output:col_expand_sub:start -->
```bash
Input x: [[-3 -2.875 -2.75 -2.625 -2.5 -2.375 -2.25 -2.125 ...], [13 13.125 13.25 13.375 13.5 13.625 13.75 13.875 ...], [29 29.125 29.25 29.375 29.5 29.625 29.75 29.875 ...], [45 45.125 45.25 45.375 45.5 45.625 45.75 45.875 ...], ...]
Input y: [[1.25 1.5 1.75 2 2.25 2.5 2.75 3 ...]]
Output out: [[-4.25 -4.375 -4.5 -4.625 -4.75 -4.875 -5 -5.125 ...], [11.75 11.625 11.5 11.375 11.25 11.125 11 10.875 ...], [27.75 27.625 27.5 27.375 27.25 27.125 27 26.875 ...], [43.75 43.625 43.5 43.375 43.25 43.125 43 42.875 ...], ...]
```
<!-- pypto-doc-output:col_expand_sub:end -->
