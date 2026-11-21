# pypto_pro.language.and_

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

Performs element-wise bitwise AND on the corresponding positions of two operands. Both Tile-Tile and Tile-Scalar modes are supported, where in the latter rhs is a scalar.

- **Tile-Tile mode**: A bitwise AND is performed on the elements at corresponding positions of lhs and rhs, and the result is written to out.
- **Tile-Scalar mode**: A bitwise AND is performed between each element in lhs and the scalar rhs, and the result is written to out.

## Prototype

```python
pypto_pro.language.and_(
    out: Tile,
    lhs: Tile,
    rhs: Union[Tile, int],
) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| out | Output| Destination operand, which is of type Tile and stored in UB, using a row-major layout. It holds the result of the element-wise bitwise AND. It supports DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, and DT_UINT32. In Tile-Tile mode, it may be the same Tile as lhs, enabling in-place computation.|
| lhs | Input| Source operand (left operand), which is of type Tile, stored in UB, and using a row-major layout. Its data type and valid_shape must be consistent with those of out.|
| rhs | Input| Source operand (right operand), which can be a Tile, an integer constant, or a runtime integer scalar expression. Both modes support DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, and DT_UINT32. When a Tile is passed in, Tile-Tile computation is performed; the Tile must be stored in UB and use a row-major layout, and its data type and valid_shape must be consistent with those of out and lhs. When a scalar is passed in, Tile-Scalar computation is performed; the value must be within the representable range of the out data type, and integer constants are treated as scalars.|

## Constraints

None

## Return Value

None

## Examples

### Tile-Scalar Mode

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def and_scalar_kernel(a: pl.Tensor[[64, 64], pl.DT_INT32],
                      out: pl.Tensor[[64, 64], pl.DT_INT32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.and_(cur_out, cur_a, 7)
        pl.store(out, cur_out, [0, 0])
```

### Tile-Tile Mode

```python
# Perform a bitwise AND on the corresponding positions of two Tiles.
pl.and_(tile_out, tile_a, tile_b)
```
