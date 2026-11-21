# pypto_pro.language.argmax

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

Retrieves the index of the maximum element of the source Tile along the specified dimension and writes it to the destination Tile.

## Prototype

```python
pypto_pro.language.argmax(out: Tile, src: Tile, tmp: Tile, *, dim: int=0) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| out | Output| Destination operand of the Tile type. It stores the index of the maximum element in each row or column. For details about the supported data types, see [Constraints](#constraints).<br>If the shape of src is [M, N], then when dim=0, the shape of out is [M, 1]; when dim=1, the shape of out is [1, N].|
| src | Input| Source operand of the Tile type. For details about the supported data types, see [Constraints](#constraints).|
| tmp | Input| Compatibility temporary Tile, which must still be supplied in the current call. Using the same data type, shape, and valid shape as src is recommended.|
| dim | Input| Lookup dimension. 0 means obtaining the column index corresponding to the maximum value in each row; 1 means obtaining the row index corresponding to the maximum value in each column.|

## Return Value

None

## Constraints

- dim can only be 0 or 1, which respectively looks up the column index of each row and the row index of each column.
- src supports only DT_FP16 and DT_FP32, while out supports only DT_INT32 and DT_UINT32. The type ranges for the two dimensions are the same.
- All tiles must be located in UB. The valid region of src must not be empty, and the valid shape of out must be consistent with the reduction result.

## Examples

### dim = 0

```python
import pypto_pro.language as pl

M, N = 64, 128


@pl.jit(auto_mutex=True)
def row_argmax_kernel(
    x: pl.Tensor[[M, N], pl.DT_FP16],
    z: pl.Tensor[[M, 1], pl.DT_INT32],
):
    tile_a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000, mutex_ids=[0])
    tile_tmp = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(
        type=pl.TileType(shape=[64, 1], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, x, [0, 0])
        pl.argmax(cur_out, cur_a, cur_tmp)
        pl.store(z, cur_out, [0, 0])
```

<!-- pypto-doc-output:row_argmax:start -->
```bash
Input a: [[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [24 24.25 24.5 24.75 25 25.25 25.5 25.75 ...], [56 56.25 56.5 56.75 57 57.25 57.5 57.75 ...], [88 88.25 88.5 88.75 89 89.25 89.5 89.75 ...], ...]
Output z: [[127], [127], [127], [127], ...]
```
<!-- pypto-doc-output:row_argmax:end -->

### dim = 1

```python
import pypto_pro.language as pl


M, N = 64, 128

@pl.jit(auto_mutex=True)
def col_argmax_kernel(
    x: pl.Tensor[[M, N], pl.DT_FP16],
    z: pl.Tensor[[1, N], pl.DT_INT32],
):
    tile_a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000, mutex_ids=[0])
    tile_tmp = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(
        type=pl.TileType(shape=[1, 128], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, x, [0, 0])
        pl.argmax(cur_out, cur_a, cur_tmp, dim=1)
        pl.store(z, cur_out, [0, 0])
```

<!-- pypto-doc-output:col_argmax:start -->
```bash
Input x: [[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [24 24.25 24.5 24.75 25 25.25 25.5 25.75 ...], [56 56.25 56.5 56.75 57 57.25 57.5 57.75 ...], [88 88.25 88.5 88.75 89 89.25 89.5 89.75 ...], ...]
Output z: [[63 63 63 63 63 63 63 63 ...]]
```
<!-- pypto-doc-output:col_argmax:end -->
