# pypto_pro.language.range

<!-- md-trans-meta sourceCommit=b055f6477ecc13f6fe4ce2ec802663fd5100ad23 translatedAt=2026-10-26T12:54:21.079Z pushedAt=2026-10-28T11:57:28.676Z nl=2 -->

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

Generates the iteration range of a for loop in the kernel. The iteration range is determined by the start value, stop value, and step, and excludes the stop value. The loop variable is a runtime integer scalar that can be used for tile indexing, data offset computation, and scalar operations.

## Prototype

```python
pypto_pro.language.range(
    start: Union[int, Scalar],
    stop: Optional[Union[int, Scalar]] = None,
    step: Union[int, Scalar] = 1,
    /,
) -> RangeIterator
```

## Parameters

| Parameter | Input/Output | Description |
|---|---|---|
| start | Input | Start value. Supports an integer constant, a runtime integer scalar, or an integer scalar expression. Omitted and defaults to 0 in the single-argument form. For the value range, see [Constraints](#constraints). |
| stop | Input | Stop value (exclusive). Supports an integer constant, a runtime integer scalar, or an integer scalar expression. For the value range, see [Constraints](#constraints). |
| step | Input | Step. Supports an integer constant, a runtime integer scalar, or an integer scalar expression. For the value range, see [Constraints](#constraints). |

## Constraints

- Only 1 to 3 positional arguments are supported. Keyword arguments are not supported.
- The value range requirements for the start, stop, and step parameters are as follows (only compile-time constants can be validated and intercepted; you must ensure the values of runtime scalars or expressions yourself).
    - start, stop, and step must be integer scalars.
    - The value ranges of start and stop depend on the execution domain.
        - Within the VF domain (the function body decorated by pypto_pro.language.vector_function): start and stop must be non-negative and must not exceed uint16, that is, the value range is [0, 65535].
        - Outside the VF domain: the value range of start and stop is int64, that is, [-9223372036854775808, 9223372036854775807].
    - step must be a positive integer and must not exceed the upper bound of the execution domain in which it resides.
        - Within the VF domain: the value range of step is [1, 65535].
        - Outside the VF domain: the value range of step is [1, 9223372036854775807].
    - After the last increment (i += step), the value of the loop variable must not exceed the upper bound of the execution domain in which it resides (65535 within the VF domain, and the int64 upper bound outside the VF domain). Otherwise, wraparound or overflow occurs, resulting in incorrect loop results.
- Nested loops are supported, and break and continue can be used in a for loop. break exits only the innermost loop in which it resides, and continue skips the current iteration.
- pypto_pro.language.range can be used in the common function body of a JIT function, or inside pypto_pro.language.section_vector or pypto_pro.language.section_cube. The specific operations in the loop body must still comply with the usage constraints of the corresponding execution domain.

## Return Value

Returns an iterator for use in a for loop.

## Examples

### Double-Layer Loop Tiling

```python
import pypto_pro.language as pl

# This example requires M and N to be integer multiples of TILE_M and TILE_N, respectively.
# The Vector Kernel enables auto_mutex, and synchronization is automatically managed by make_tile_group.
TILE_M = 64
TILE_N = 64


@pl.jit(auto_mutex=True)
def for_add_fp16_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    M = x.shape[0]
    N = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, M // TILE_M, 1):
            for j in pl.range(0, N // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])
```

### Single-Parameter Form

```python
# Equivalent to range(0, 10, 1).
for i in pl.range(10):
    ...
```
