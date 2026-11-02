# pypto_pro.language.section_cube

<!-- md-trans-meta sourceCommit=615e96d66b40a544ed5a7780011541c40cbe3ce3 translatedAt=2026-10-26T12:54:14.219Z pushedAt=2026-10-28T11:57:28.670Z nl=2 -->

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

Identifies the Cube execution domain in a kernel. It must be used together with the with statement. The Cube execution domain organizes matrix computation and related data movement operations, including data movement between GM and L1 Buffer, between L1 Buffer and L0A Buffer or L0B Buffer, matrix computation, and data movement out of L0C Buffer.

Multiple Cube execution domains can be defined sequentially in the same Kernel, and can also be used together with the Vector execution domain identified by [pypto_pro.language.section_vector](section_vector.md).

## Prototype

```python
pypto_pro.language.section_cube() -> ContextManager
```

## Parameters

None

## Constraints

- section_cube cannot be nested within section_cube or section_vector.
- All APIs that perform Tile operations must be located inside section_vector or section_cube, and cannot appear directly at the top level of the Kernel function body.
- In pipeline mode, the stage call chain must strictly alternate between Cube and Vector; stages in two consecutive regions of the same type are not supported.

## Return Value

Returns a context manager used to delimit the Cube execution domain.

## Examples

### Cube Region Matrix Computation

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def with_section_cube_fp16_kernel(
    x: pl.Tensor[[64, 32], pl.DT_FP16],
    y: pl.Tensor[[32, 64], pl.DT_FP16],
    z: pl.Tensor[[64, 64], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000, mutex_ids=[1])
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids=[2])
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[3])
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000, mutex_ids=[4])
    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.load(cur_a, x, [0, 0])
        pl.load(cur_b, y, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.store(z, ac, [0, 0])
```
