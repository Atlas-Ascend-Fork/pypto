# pypto_pro.language.set_validshape

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

Sets the valid shape range of a Tile or TileGroup to handle tail blocks or non-full Tile scenarios. When a single Tile is passed, this API sets the valid data range of that Tile. When a TileGroup is passed, the same valid_shape is set for all Tiles in the group, which applies to scenarios where the entire TileGroup uses the same valid shape. When valid_shape of [pypto_pro.language.TileType](../basic_data_structures/TileType.md) is None, it is equivalent to [-1, -1] (dynamic mode), and the actual valid shape is set through this API at runtime.

## Prototype

```python
pypto_pro.language.set_validshape(tile: Tile | TileGroup, /, shape: Shape) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| tile | Output| Destination operand, of the Tile or TileGroup type.|
| shape | Input| Valid shape sequence (list or tuple) with a length of 2. Both elements are integer constants or runtime integer scalar expressions (loop variables supported), must be positive integers, and must not exceed the corresponding dimension of the Tile shape.|

## Return Value

None

## Constraints

- TileType must use a dynamic valid shape: valid_shape is omitted or valid_shape=[-1, -1] is declared. Static valid dimensions cannot be modified using this API.
- The shape contains two positive integers or runtime integer expressions, which must not exceed the corresponding physical shape. The range of the dynamic expression is ensured by the caller.
- When TileGroup is passed, all Tiles in the group must meet the preceding conditions.

## Examples

### Basic Usage

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def validshape_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP32],
    rows: pl.DT_INT64,
    cols: pl.DT_INT64,
    out: pl.Tensor[[64, 128], pl.DT_FP32],
):
    tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec,
                            valid_shape=[-1, -1])
    tile_group = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])
    with pl.section_vector():
        tile = tile_group.current()
        pl.set_validshape(tile, [rows, cols])
        pl.load(tile, a, [0, 0])
        pl.store(out, tile, [0, 0])
```

### TileGroup Batch Setting

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def tile_group_validshape_kernel(
    a: pl.Tensor[[128, 128], pl.DT_FP16],
    rows: pl.DT_INT64,
    cols: pl.DT_INT64,
    output: pl.Tensor[[128, 128], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, valid_shape=[-1, -1])
    tile_a = pl.make_tile_group(
        type=tile_type, addrs=0x0000, mutex_ids=[0])
    pl.set_validshape(tile_a, [rows, cols])
    with pl.section_vector():
        cur_a = tile_a.current()
        pl.load(cur_a, a, [0, 0])
        pl.store(output, cur_a, [0, 0])
```
