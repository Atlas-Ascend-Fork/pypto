# pypto_pro.language.section_vector

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T12:55:33.367Z pushedAt=2026-10-28T11:57:28.679Z nl=2 -->

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

Identifies the Vector execution domain in a Kernel and must be used together with the with statement. The Vector execution domain organizes data movement between GM and UB as well as vector computation on UB tiles, including element-wise operations, reduction, and transposition.

Multiple Vector execution domains can be defined sequentially in the same Kernel, and they can also be used together with the Cube execution domain identified by [pypto_pro.language.section_cube](section_cube.md).

## Prototype

```python
pypto_pro.language.section_vector() -> ContextManager
```

## Parameters

None

## Constraints

- section_vector cannot be nested with section_vector or section_cube.
- All APIs that perform Tile operations must be inside section_vector or section_cube and cannot appear directly at the top level of the Kernel function body.
- In pipeline mode, the stage call chain must strictly alternate between Cube and Vector; stages in two consecutive regions of the same type are not supported.

## Return Value

Returns a context manager that defines the Vector execution domain.

## Examples

### Vector Region Vector Computation

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def with_section_vector_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.relu(cur_out, cur_a)
        pl.store(out, cur_out, [0, 0])
```
