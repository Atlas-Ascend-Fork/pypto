# pypto_pro.language.simt.grid_dim

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

Obtains the number of thread blocks in the current thread block grid along the X, Y, and Z dimensions.

## Prototype

```python
pypto_pro.language.simt.grid_dim() -> Any
```

## Parameters

None

## Constraints

This API can only be called in the SIMT kernel, SIMT vector function, or SIMT helper.

## Return Value

Returns a tuple object of the three-dimensional thread block grid size. You can read the size of each dimension through grid.x, grid.y, and grid.z. Each component is a scalar of the DT_UINT32 type.

- SIMT programming: The grid passed by the host is returned. For example, if grid is (2, 3, 4), (2, 3, 4) is returned.
- In SIMD and SIMT hybrid programming, grid.x is the number of logical Vector (AIV) cores that actually take effect in the outer kernel, and grid.y and grid.z are both 1. For details about the relationship between the actual value and the block_dim passed by the host, see [Meaning and Setting of blockDim](../../../../guide/programming_guide/pro/development/kernel_function.md#meaning-and-setting-of-blockdim).

## Examples

```python
import pypto_pro.language as pl

@pl.vector_function(mode="simt", max_threads=256)
def write_grid_dim_x(
    output: pl.Tensor[[1, 256], pl.DT_UINT32],
):
    tid = pl.simt.linear_thread_idx()
    grid = pl.simt.grid_dim()
    output[0, tid] = grid.x


@pl.jit()
def simt_grid_dim_kernel(
    output: pl.Tensor[[1, 256], pl.DT_UINT32],
):
    with pl.section_vector():
        write_grid_dim_x[256](output)
```
