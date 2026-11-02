# pypto_pro.language.sub

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T14:44:39.734Z pushedAt=2026-10-28T11:57:29.265Z nl=2 -->

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

Performs element-wise subtraction on the corresponding positions of the two operands. Supports both Tile-Tile and Tile-Scalar modes, where Scalar is a scalar value. In-place computation is also supported.

- **Tile-Tile mode**: Subtracts the element in rhs from the element at the corresponding position in lhs, and writes the result to out.
- **Tile-Scalar mode**: Subtracts the rhs scalar from each element in lhs, and writes the result to out.

## Prototype

```python
pypto_pro.language.sub(
    out: Tile,
    lhs: Tile,
    rhs: Union[Tile, Scalar],
) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| out | Output | Destination operand of the Tile type, which stores the result of the element-wise subtraction. Its data type is the same as that of lhs, supporting DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_FP16, DT_BF16, and DT_FP32. It can be the same Tile as lhs or the Tile-type rhs to implement in-place computation. |
| lhs | Input | Left operand (minuend) of the Tile type. Its data type is the same as that of out. |
| rhs | Input | Right operand (subtrahend) of the Tile or Scalar type. When a Tile is passed, Tile-Tile computation is performed, and its data type is the same as that of out, with its shape consistent with out and lhs. When a Scalar is passed, Tile-Scalar computation is performed. |

## Constraints

None

## Return Value

None

## Examples

### Tile-Tile Mode

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def sub_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32], b: pl.Tensor[[64, 64], pl.DT_FP32],
               out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.sub(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```

The following shows an example of the actual result.

<!-- pypto-doc-output:sub:start -->
```bash
Input data a: [[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [17 17.25 17.5 17.75 18 18.25 18.5 18.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [49 49.25 49.5 49.75 50 50.25 50.5 50.75 ...], ...]
Input data b: [[10 10.5 11 11.5 12 12.5 13 13.5 ...], [42 42.5 43 43.5 44 44.5 45 45.5 ...], [74 74.5 75 75.5 76 76.5 77 77.5 ...], [106 106.5 107 107.5 108 108.5 109 109.5 ...], ...]
Output data out: [[-9 -9.25 -9.5 -9.75 -10 -10.25 -10.5 -10.75 ...], [-25 -25.25 -25.5 -25.75 -26 -26.25 -26.5 -26.75 ...], [-41 -41.25 -41.5 -41.75 -42 -42.25 -42.5 -42.75 ...], [-57 -57.25 -57.5 -57.75 -58 -58.25 -58.5 -58.75 ...], ...]
```
<!-- pypto-doc-output:sub:end -->

### Tile-Scalar Mode

```python
# Subtract the Scalar value from each element of the Tile.
pl.sub(out, lhs, 1.0)
```
