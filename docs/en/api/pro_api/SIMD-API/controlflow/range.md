# pypto_pro.language.range

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

Generates the iteration range of the for loop in the kernel. The iteration range is determined by the start value, stop value, and step, and excludes the stop value. The loop variable is a runtime integer scalar, which can be used for tile indexing, data offset computation, and scalar operations.

## Prototype

```python
pypto_pro.language.range(
    start: int,
    stop: Optional[int] = None,
    step: int = 1,
    /,
) -> RangeIterator
```

## Parameters

| Parameter| Input/Output| Description|
|---|---|---|
| start | Input| Start value. It can be an integer constant, runtime integer scalar, or integer scalar expression. This parameter is omitted and defaults to 0 in the single-argument form. For details about the value range, see [Constraints](#constraints).|
| stop | Input| Stop value (excluded). It can be an integer constant, runtime integer scalar, or integer scalar expression. For details about the value range, see [Constraints](#constraints).|
| step | Input| Step. It can be an integer constant, runtime integer scalar, or integer scalar expression. For details about the value range, see [Constraints](#constraints).|

## Constraints

- Only one to three positional arguments are supported. Keyword arguments are not supported.
- The value ranges of the start, stop, and step parameters are as follows (only compile-time constants can be verified and intercepted, and the values of runtime scalars or expressions need to be ensured by users):
    - The start, stop, and step parameters must be integer scalars.
    - The value ranges of start and stop depend on the execution domain.
        - In the VF domain (function body decorated by pypto_pro.language.vector_function): The values of start and stop must be non-negative integers and cannot exceed uint16, which means that the value range is [0, 65535].
        - Outside the VF domain: The value ranges of start and stop are int64, which means that the value range is [–9223372036854775808, 9223372036854775807].
    - The value of step must be a positive integer and cannot exceed the upper bound of the execution domain in which it resides.
        - In the VF domain, the value range of step is [1, 65535].
        - Outside the VF domain: The value range of step is [1, 9223372036854775807].
    - The value of the loop variable after the last increment (i += step) cannot exceed the upper bound of the execution domain in which it resides (65535 in the VF domain, and the upper bound of int64 outside the VF domain). Otherwise, wraparound or overflow may occur, resulting in incorrect loop results.
- Nested loops are supported, and the break and continue statements can be used in for loops. The break statement exits only the innermost loop where it is located, and the continue statement skips the current iteration.
- pypto_pro.language.range can be used in the common function body of the JIT function, pypto_pro.language.section_vector, or pypto_pro.language.section_cube. The specific operations in the loop body must still meet the usage restrictions of the corresponding execution domain.

## Return Value

Returns an iterator for the for loop.

## Examples

### Double-Layer Loop Tiling

```python
import pypto_pro.language as pl

# In this example, M and N must be integer multiples of TILE_M and TILE_N, respectively.
# When auto_mutex is enabled for the vector kernel, synchronization is automatically managed by make_tile_group.
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

### Single-Argument Form

```python
# Equivalent to range(0, 10, 1)
for i in pl.range(10):
    ...
```
