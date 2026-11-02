# pypto_pro.language.and_

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T14:40:14.078Z pushedAt=2026-10-28T11:57:29.230Z nl=2 -->

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

Performs element-wise bitwise AND on the corresponding positions of the two operands. Both Tile-Tile and Tile-Scalar modes are supported, where Scalar is a scalar.

- **Tile-Tile mode**: Performs bitwise AND on the elements at the corresponding positions of lhs and rhs, and writes the result to out.
- **Tile-Scalar mode**: Performs bitwise AND on each element of lhs and the rhs scalar, and writes the result to out.

## Prototype

```python
pypto_pro.language.and_(
    out: Tile,
    lhs: Tile,
    rhs: Union[Tile, Scalar],
) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| out | Output | Destination operand of the Tile type, stored in UB in row-major layout, holding the result of the element-wise bitwise AND. Supports DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, and DT_UINT32. In Tile-Tile mode, it can be the same Tile as lhs for in-place computation. |
| lhs | Input | Source operand (left operand) of the Tile type, stored in UB in row-major layout. The data type and valid_shape must be consistent with out. |
| rhs | Input | Source operand (right operand), supporting the Tile or Scalar type. Both modes support DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, and DT_UINT32. When a Tile is passed, Tile-Tile computation is performed; the Tile is stored in UB in row-major layout, and its data type and valid_shape must be consistent with out and lhs. When a Scalar is passed, Tile-Scalar computation is performed; its value must be within the representable range of the out data type, and integer constants are treated as Scalar. |

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
# Perform bitwise AND on the corresponding positions of two tiles.
pl.and_(tile_out, tile_a, tile_b)
```
