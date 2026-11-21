# pypto_pro.language.partmax

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

Performs element-wise maximum computation on the valid regions of two source Tiles. When a position is valid in both source Tiles, the maximum of the two values is output; when a position is valid in only one source Tile, the data at the corresponding position in that source Tile is copied.

## Prototype

```python
pypto_pro.language.partmax(out: Tile, src0: Tile, src1: Tile) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| out | Output| Target 2D UB Tile, which stores the computation result. The supported data types are DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_FP16, DT_BF16, and DT_FP32.|
| src0 | Input| First source 2D UB Tile. The supported data types are the same as those of out.|
| src1 | Input| Second source 2D UB Tile. The supported data types are the same as those of out.|

## Return Value

None

## Constraints

- The physical Shapes and data types of out, src0, and src1 must be the same.
- At least one of the two source Tiles must have a valid Shape consistent with that of out; the valid row count and valid column count of the other source Tile must not exceed those of out.
- A smaller valid Shape can be used to specify a partial computation region, but specifying a partial computation region by using a smaller physical Shape is not supported.
- When the valid region of out is empty, the API returns immediately.

## Examples

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def partmax_kernel(x: pl.Tensor[[64, 128], pl.DT_FP16],
                   y: pl.Tensor[[64, 128], pl.DT_FP16],
                   out: pl.Tensor[[64, 128], pl.DT_FP16]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16,
                     target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    gx = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    gy = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[1])
    gout = pl.make_tile_group(type=tt, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        tx, ty, tout = gx.current(), gy.current(), gout.current()
        pl.set_validshape(tx, [64, 128])
        pl.load(tx, x, [0, 0])
        pl.set_validshape(ty, [32, 128])
        pl.load(ty, y, [0, 0])
        pl.set_validshape(tout, [64, 128])
        pl.partmax(tout, tx, ty)
        pl.store(out, tout, [0, 0])
```

For the first 32 rows, maximum (x, y) is used. For the last 32 rows, only x is valid. Therefore, x is copied.
