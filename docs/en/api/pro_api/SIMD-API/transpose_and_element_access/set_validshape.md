# pypto_pro.language.set_validshape

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T14:51:59.854Z pushedAt=2026-10-28T11:57:29.318Z nl=2 -->

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
pypto_pro.language.set_validshape(tile: Union[Tile, TileGroup], /, shape: List[int]) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| tile | Input/Output | Destination operand, of the Tile or TileGroup type. |
| shape | Input | Valid shape sequence of length 2. Both elements are integer constants or runtime integer scalar expressions (loop variables supported), must be positive integers, and must not exceed the corresponding dimension of the Tile shape. |

## Constraints

None

## Return Value

None

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

### Other Typical Usage

```python
# Handle the matmul tail block.
pl.set_validshape(q_mat_buf[q_count % 2], [TD, actual_sq])

# Both the TileGroup accessor and subscript return a Tile directly.
pl.set_validshape(a_wide_group.current(), [256, 64])
pl.set_validshape(a_wide_group[0], [256, 64])
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
