# pypto_pro.language.ge

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

Compares lhs and rhs element-wise. When lhs is greater than or equal to rhs, true is generated. The result is output as a bit-packed mask. rhs can be a Tile or a scalar. The output is a bit-packed mask, typically stored in a Tile of type DT_UINT8, and is usually used in conjunction with [pypto_pro.language.select](../selection/select.md).

## Prototype

```python
pypto_pro.language.ge(out: Tile, lhs: Tile, rhs: Union[Tile, int, float]) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| out | Output| Destination operand, a UB Tile with a row-major layout, storing the bit-packed comparison result. The data type is not restricted to DT_UINT8; DT_UINT8 is typically used. The storage space must cover the mask bits corresponding to all valid elements.|
| lhs | Input| Left operand, a UB Tile with a row-major layout. Supported data types: DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_FP16, DT_FP32 and DT_BF16.|
| rhs | Input| Right operand, which is of type Tile, int, or float, and supports the corresponding runtime numeric scalar expression. When a Tile is passed in, the Tile must be located in UB and use a row-major layout, and the Tile's data type, shape, and valid shape must be consistent with lhs. When a scalar is passed in, the scalar value must be representable by the element type of lhs.|

## Return Value

None

## Constraints

out holds a bit-packed mask and must not be interpreted as a per-element 0/1 numeric Tile. To obtain numeric results, use it together with [pypto_pro.language.select](../selection/select.md).

## Examples

Provide the mask to pypto_pro.language.select: a is selected when the comparison result is true, and b is selected otherwise.

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def ge_select_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32],
    b: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    mask_type = pl.TileType(shape=[64, 64], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec)
    ga = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    gb = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    gout = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    gtmp = pl.make_tile_group(type=tt, addrs=0xC000, mutex_ids=[3])
    gmask = pl.make_tile_group(type=mask_type, addrs=0x10000, mutex_ids=[4])
    with pl.section_vector():
        ta = ga.current()
        tb = gb.current()
        tout = gout.current()
        tmp = gtmp.current()
        mask = gmask.current()
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.ge(mask, ta, tb)
        pl.select(tout, mask, ta, tb, tmp)
        pl.store(out, tout, [0, 0])
```
