# pypto_pro.language.abs

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T14:40:32.392Z pushedAt=2026-10-28T11:57:29.233Z nl=2 -->

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

Computes the absolute value of each element in the source operand src and writes the result to the destination operand out. Negative values are negated, while positive values and zero remain unchanged.

## Prototype

```python
pypto_pro.language.abs(
    out: Tile,
    src: Tile,
) -> None
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| out | Output | Destination operand of the Tile type, which stores the result of computing the absolute value element by element. The data type is the same as that of src, supporting DT_FP16, DT_FP32, DT_INT8, DT_INT16, or DT_INT32. The shape and valid_shape are the same as those of src. It can be the same Tile as src for in-place computation. |
| src | Input | Source operand of the Tile type. The data type is the same as that of out. |

## Constraints

None

## Return Value

None

## Examples

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def abs_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32],
               out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.abs(cur_out, cur_a)
        pl.store(out, cur_out, [0, 0])
```
