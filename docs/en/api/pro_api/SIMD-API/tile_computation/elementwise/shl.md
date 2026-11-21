# pypto_pro.language.shl

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

Shifts each element in lhs to the left and writes the result to out. If rhs is a Tile, each position uses the shift amount specified by the corresponding element. If rhs is an int or a runtime integer scalar, all elements use the same shift amount.

## Prototype

```python
pypto_pro.language.shl(out: Tile, lhs: Tile, rhs: Union[Tile, int]) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| out | Output| Destination operand, a UB Tile with a row-major layout. The supported data types are DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, and DT_UINT64. They are the same as those of lhs. Its Shape and valid Shape are the same as those of lhs.|
| lhs | Input| Shifted operand, which is of type Tile. It must be located in the UB and use a row-major layout. The data types are the same as those of out.|
| rhs | Input| Shift amount, which is of type Tile or int. Runtime integer scalar expressions are also supported. When a Tile is passed in, it must be in UB with a row-major layout, and its data type, shape, and valid shape must be the same as those of lhs. When a scalar is passed in, it must be representable by the element type of lhs. The shift amount must be non-negative and less than the element bit width of lhs.|

## Return Value

None

## Constraints

None

## Examples

### Tile-Tile Mode

Below is a complete Kernel. It loads two DT_INT32 inputs from GM to UB, performs left shift element-wise using pypto_pro.language.shl, and writes the result back to GM. The Vector Kernel has auto_mutex enabled, and synchronization is managed automatically by pypto_pro.language.make_tile_group.

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def shl_kernel(a: pl.Tensor[[64, 64], pl.DT_INT32],
               b: pl.Tensor[[64, 64], pl.DT_INT32],
               out: pl.Tensor[[64, 64], pl.DT_INT32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.shl(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```

### Tile-Scalar Mode

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def shl_scalar_kernel(
    a: pl.Tensor[[64, 64], pl.DT_INT32],
    out: pl.Tensor[[64, 64], pl.DT_INT32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.shl(cur_out, cur_a, 1)
        pl.store(out, cur_out, [0, 0])
```
