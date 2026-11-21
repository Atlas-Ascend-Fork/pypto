# pypto_pro.language.sqrt

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

Computes the square root of each element in the source operand src and writes the result to the destination operand out.

## Prototype

```python
pypto_pro.language.sqrt(out: Tile, src: Tile) -> None
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| out | Output| Destination operand of type Tile, which stores the result of element-wise square root. The supported data types are DT_FP16 and DT_FP32. They are the same as those of src. The shape is the same as that of src. It can be the same Tile as src for in-place computation.|
| src | Input| Source operand, which is of type Tile. The data types are the same as those of out. The element value must be non-negative.|

## Return Value

None

## Constraints

None

## Examples

Below is a complete Kernel. It computes the square root of each element in the DT_FP32 source Tile and writes the result back to GM. The Vector Kernel has auto_mutex enabled, and synchronization is managed automatically by pypto_pro.language.make_tile_group.

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def sqrt_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32],
                out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.sqrt(cur_out, cur_a)
        pl.store(out, cur_out, [0, 0])
```
